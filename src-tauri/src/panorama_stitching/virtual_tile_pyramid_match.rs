//! Coarse-to-fine Virtual_Tile station matching.
//!
//! Runs the bounded probe through the match pyramid and polishes the resulting
//! relation on native station pixels.

use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stack_pipeline::residual_warp;
use crate::panorama_utils::stitching;
use image::GrayImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use rayon::prelude::*;
use std::collections::HashMap;

use super::{
    MatchInfo, STATION_RELATION_MAX_MEDIAN_ERROR_PX, STATION_RELATION_MIN_INLIERS,
    STATION_RELATION_MIN_SPATIAL_SUPPORT, STATION_RELATION_SCALE_RATIO_MAX,
    STATION_RELATION_SCALE_RATIO_MIN, StationOverlapMeasure,
    VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX, VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
    VirtualTileMatchPlane, VirtualTileMatchPyramid, estimate_affine,
    homography_preserves_focus_orientation, homography_scale_ratio, median_symmetric_error,
    normalized_homography, residual_distribution, station_relation_spatial_support_with,
    symmetric_point_error, transformed_point, virtual_tile_direct_station_match,
    virtual_tile_direct_station_match_with_structure, virtual_tile_polish_fit,
    virtual_tile_polished_station_match,
};

/// Symmetric residual-model tolerance of the octaves coarser than the finest,
/// in their own pixels.
pub(super) const VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX: f64 = 1.5;

/// A coarse pyramid failure may reflect the level's sparse residual model,
/// while a finer level can still measure the same prior.  Keep this retry
/// list deliberately narrow: a missing overlap or missing texture is not
/// evidence that becomes stronger at a finer level.
pub(super) fn virtual_tile_coarse_failure_can_retry(failure_stage: &str) -> bool {
    matches!(
        failure_stage,
        // At the coarsest octave a narrow, but valid, overlap can contain
        // fewer than the final minimum number of grid probes.  That is a
        // sampling-density failure of this octave, not evidence that the
        // station pair is unrelated.  Retry at the next finer bounded octave
        // and let the unchanged final polish/Station_Relation gates decide.
        "insufficient_grid_probes"
            | "insufficient_measurable_patches"
            | "insufficient_bidirectional_matches"
            | "insufficient_grid_coverage"
            | "residual_model_no_fit"
            | "insufficient_matches_after_model_quota"
            | "insufficient_fitted_hull_support"
    )
}

/// Probe one pair coarse to fine around `seed_left_to_right` (native tile
/// pixels). Every octave but the finest runs the bounded bidirectional probe
/// (edge orientation read on the octave's σ = 1 structural plane) and re-seeds
/// the next with its measured model (tolerance
/// [`VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX`] in its own pixels); the
/// finest octave polishes that model with model-warped templates
/// ([`virtual_tile_polished_station_match`]). A single native level (small
/// tiles) keeps the bounded probe with the requirement-6.3 tolerance. The
/// relation is expressed in native tile pixels; the diagnostic describes the
/// last stage that ran.
pub(super) fn virtual_tile_pyramid_station_match(
    left: usize,
    right: usize,
    pyramid: &VirtualTileMatchPyramid,
    seed_left_to_right: &Matrix3<f64>,
    seed_kind: &'static str,
) -> (Vec<MatchInfo>, stack_report::StationRelationCandidateRecord) {
    let empty_diagnostic = || {
        let (_, mut diagnostic) = virtual_tile_direct_station_match(
            left,
            right,
            &[],
            &[],
            &HashMap::new(),
            STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        );
        diagnostic.pyramid_factors = pyramid.factors.clone();
        diagnostic.seed = seed_kind.to_string();
        diagnostic
    };
    let level_count = pyramid.factors.len();
    if level_count == 0
        || left >= pyramid.native_dimensions.len()
        || right >= pyramid.native_dimensions.len()
    {
        return (Vec::new(), empty_diagnostic());
    }
    let single_level = level_count == 1;
    let bounded_levels = if single_level { 1 } else { level_count - 1 };
    let mut seed = *seed_left_to_right;
    let mut diagnostic = empty_diagnostic();
    for level in 0..bounded_levels {
        let factor = pyramid.factors[level];
        let to_native = VirtualTileMatchPlane::level_to_native(factor);
        let from_native = VirtualTileMatchPlane::native_to_level(factor);
        let level_tiles = &pyramid.levels[level];
        let Some(level_seed_inverse) = (from_native * seed * to_native).try_inverse() else {
            diagnostic.failure_stage = "non_invertible_seed".to_string();
            return (Vec::new(), diagnostic);
        };
        let initial = HashMap::from([
            (level_tiles[left].id, Matrix3::identity()),
            (level_tiles[right].id, level_seed_inverse),
        ]);
        let inlier_threshold_px = if single_level {
            STATION_RELATION_MAX_MEDIAN_ERROR_PX
        } else {
            VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX
        };
        // A coarse octave reads edge orientation on its structural (σ = 1)
        // plane, the rule of the relation-quality plane: on langyuan-10 ±1px
        // gradients of the unfiltered 1/8 planes read 13.4–15.5° for verified
        // relations (the prior-free 1→2 proposal failed the 15° gate at 15.5°),
        // the structural planes 5.1–6.0°, and a 2-level-pixel shift 27–36°.
        let structure = (!single_level).then(|| {
            (
                imageproc::filter::gaussian_blur_f32(
                    &level_tiles[left].alignment_image,
                    VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
                ),
                imageproc::filter::gaussian_blur_f32(
                    &level_tiles[right].alignment_image,
                    VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
                ),
            )
        });
        let (relation, level_diagnostic) = virtual_tile_direct_station_match_with_structure(
            left,
            right,
            level_tiles,
            &pyramid.coverages[level],
            &initial,
            inlier_threshold_px,
            structure
                .as_ref()
                .map(|(left_plane, right_plane)| (left_plane, right_plane)),
        );
        diagnostic = level_diagnostic;
        diagnostic.pyramid_factors = pyramid.factors.clone();
        diagnostic.measurement_factor = factor;
        diagnostic.seed = seed_kind.to_string();
        let Some(relation) = relation else {
            // A later octave has a smaller residual in its own pixels and can
            // recover a model that was under-supported at this coarse level.
            // Keep the original seed until a measured relation is available;
            // no prior is ever promoted to acceptance evidence.
            if !single_level && virtual_tile_coarse_failure_can_retry(&diagnostic.failure_stage) {
                if level + 1 < bounded_levels {
                    continue;
                }
                // The last bounded octave has no finer bounded probe left.
                // Its residual-model/support failure is still a recoverable
                // measurement failure: the finest polish has independent
                // patch, reverse-consistency, hull, scale, and orientation
                // gates and must get the chance to verify the original prior.
                // Do not return the coarse diagnostic here, because that
                // would make the finest evidence unreachable for exactly the
                // sparse, shifted overlaps this retry is meant to recover.
                break;
            }
            return (Vec::new(), diagnostic);
        };
        let Some(native) = normalized_homography(&(to_native * relation.homography * from_native))
        else {
            diagnostic.failure_stage = "non_finite_native_relation".to_string();
            return (Vec::new(), diagnostic);
        };
        seed = native;
        if single_level && factor == 1 {
            let points = relation
                .points
                .iter()
                .filter_map(|(source, target)| {
                    transformed_point(&to_native, *source)
                        .zip(transformed_point(&to_native, *target))
                })
                .collect::<Vec<_>>();
            return (
                vec![MatchInfo {
                    homography: native,
                    inliers: points.len(),
                    points,
                    // Key-direction fit; see `MatchInfo::canonical_homography`.
                    canonical_homography: None,
                    native_verified_points: Vec::new(),
                    ..relation
                }],
                diagnostic,
            );
        }
    }
    let finest = level_count - 1;
    let factor = pyramid.factors[finest];
    diagnostic.measurement_factor = factor;
    let relation = virtual_tile_polished_station_match(
        &pyramid.levels[finest][left],
        &pyramid.coverages[finest][left],
        &pyramid.levels[finest][right],
        &pyramid.coverages[finest][right],
        factor,
        &seed,
        (
            pyramid.native_dimensions[left],
            pyramid.native_dimensions[right],
        ),
        &mut diagnostic,
    );
    let relation = relation.into_iter().filter_map(|relation| {
        match (
            &pyramid.native_refinement[left],
            &pyramid.native_refinement[right],
        ) {
            (Some(left_native), Some(right_native)) => virtual_tile_native_station_polish(
                left_native,
                right_native,
                &relation,
                &mut diagnostic,
            ),
            _ => Some(relation),
        }
    });
    (relation.collect(), diagnostic)
}

/// Coarse pixels establish a search model. Final camera geometry is measured
/// on native pixels: a one-level-pixel reverse check at factor four otherwise
/// admits four-native-pixel errors and biases long projective tree chains.
pub(super) fn native_station_affine_candidate(
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    source: &stitching::CoverageMask,
    target: &stitching::CoverageMask,
) -> Option<Matrix3<f64>> {
    if points.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    robust_native_affine(points).filter(|affine| {
        median_symmetric_error(affine, points) <= STATION_RELATION_MAX_MEDIAN_ERROR_PX
            && homography_preserves_focus_orientation(affine, source_dimensions)
            && (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
                .contains(&homography_scale_ratio(affine).unwrap_or(f64::NAN))
            && station_relation_spatial_support_with(
                points,
                affine,
                source_dimensions,
                target_dimensions,
                StationOverlapMeasure::Covered {
                    source,
                    target,
                    factor: 1,
                },
            ) >= STATION_RELATION_MIN_SPATIAL_SUPPORT
    })
}

/// Downweight native point outliers without changing the observations used by
/// the acceptance gates. Keep the model with the lowest symmetric median.
pub(super) fn robust_native_affine(points: &[(Point2<f64>, Point2<f64>)]) -> Option<Matrix3<f64>> {
    let mut best = estimate_affine(points)?;
    let mut best_error = median_symmetric_error(&best, points);
    let mut current = best;
    for _ in 0..30 {
        let inverse = current.try_inverse()?;
        let errors = points
            .iter()
            .map(|&(a, b)| {
                Some(
                    ((transformed_point(&current, a)? - b).norm_squared()
                        + (transformed_point(&inverse, b)? - a).norm_squared())
                    .sqrt(),
                )
            })
            .collect::<Option<Vec<_>>>()?;
        let mut design = nalgebra::DMatrix::zeros(points.len(), 3);
        let mut targets = nalgebra::DMatrix::zeros(points.len(), 2);
        for (index, (&(a, b), error)) in points.iter().zip(&errors).enumerate() {
            let weight = (1.0 / error.max(1e-9)).min(1.0).sqrt();
            design[(index, 0)] = a.x * weight;
            design[(index, 1)] = a.y * weight;
            design[(index, 2)] = weight;
            targets[(index, 0)] = b.x * weight;
            targets[(index, 1)] = b.y * weight;
        }
        let solution = design.svd(true, true).solve(&targets, 1e-10).ok()?;
        current = Matrix3::new(
            solution[(0, 0)],
            solution[(1, 0)],
            solution[(2, 0)],
            solution[(0, 1)],
            solution[(1, 1)],
            solution[(2, 1)],
            0.0,
            0.0,
            1.0,
        );
        let error = median_symmetric_error(&current, points);
        if error < best_error {
            best = current;
            best_error = error;
        }
    }
    Some(best)
}

/// Use only the perspective needed by the native evidence. A noisy projective
/// fit can predict large corner motion outside the measured overlap even when
/// its inlier error is small. Every proposal is checked on the full retained
/// point set and against the existing coverage, scale and orientation gates.
pub(super) fn native_station_regularized_candidate(
    points: &[(Point2<f64>, Point2<f64>)],
    projective: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    source: &stitching::CoverageMask,
    target: &stitching::CoverageMask,
) -> Option<Matrix3<f64>> {
    if points.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    let affine = robust_native_affine(points)?;
    let projective = projective / projective[(2, 2)];
    for step in 0..=100 {
        let fraction = step as f64 / 100.0;
        let candidate = affine * (1.0 - fraction) + projective * fraction;
        if median_symmetric_error(&candidate, points) <= STATION_RELATION_MAX_MEDIAN_ERROR_PX
            && homography_preserves_focus_orientation(&candidate, source_dimensions)
            && (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
                .contains(&homography_scale_ratio(&candidate).unwrap_or(f64::NAN))
            && station_relation_spatial_support_with(
                points,
                &candidate,
                source_dimensions,
                target_dimensions,
                StationOverlapMeasure::Covered {
                    source,
                    target,
                    factor: 1,
                },
            ) >= STATION_RELATION_MIN_SPATIAL_SUPPORT
        {
            return Some(candidate);
        }
    }
    None
}

pub(super) fn virtual_tile_native_station_polish(
    left: &(GrayImage, stitching::CoverageMask),
    right: &(GrayImage, stitching::CoverageMask),
    seed: &MatchInfo,
    diagnostic: &mut stack_report::StationRelationCandidateRecord,
) -> Option<MatchInfo> {
    let gates = crate::panorama_utils::registration::PatchMatchGates::LEGACY;
    #[cfg(test)]
    let gates = crate::panorama_utils::registration::PatchMatchGates {
        min_correlation: std::env::var("RAW_EDITOR_TILE_REPLAY_NATIVE_CORRELATION")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
            .unwrap_or(gates.min_correlation),
        min_peak_margin: std::env::var("RAW_EDITOR_TILE_REPLAY_NATIVE_PEAK_MARGIN")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(gates.min_peak_margin),
        ..gates
    };
    let inverse = seed.homography.try_inverse()?;
    let step = (seed.points.len() / 1000).max(1);
    let supported = |mask: &stitching::CoverageMask, point: Point2<f64>| {
        let (w, h) = mask.dimensions();
        let radius = 40i32;
        let (x, y) = (point.x.round() as i32, point.y.round() as i32);
        x >= radius
            && y >= radius
            && x + radius < w as i32
            && y + radius < h as i32
            && (-radius..=radius).step_by(8).all(|dy| {
                (-radius..=radius)
                    .step_by(8)
                    .all(|dx| mask.is_covered((x + dx) as u32, (y + dy) as u32))
            })
    };
    let native_verified_points = seed
        .points
        .par_iter()
        .enumerate()
        .filter(|(index, _)| index % step == 0)
        .filter_map(|(_, &(source, target))| {
            if !supported(&left.1, source) || !supported(&right.1, target) {
                return None;
            }
            let measured = residual_warp::verify_round_trip_match_with_gates(
                &left.0,
                &right.0,
                &seed.homography,
                &inverse,
                source,
                31,
                gates,
            )?;
            supported(&right.1, measured.forward).then_some((
                source,
                measured.forward,
                measured.round_trip_error_px,
            ))
        })
        .collect::<Vec<_>>();
    let points = native_verified_points
        .iter()
        .map(|&(a, b, _)| (a, b))
        .collect::<Vec<_>>();
    diagnostic.failure_stage = "insufficient_native_round_trip_matches".to_string();
    diagnostic.bidirectional_match_count = points.len();
    diagnostic.measurement_factor = 1;
    if points.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    let (homography, indices) = virtual_tile_polish_fit(
        &points,
        VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX,
        left.0.dimensions(),
    )?;
    let native_verified_points = indices
        .into_iter()
        .map(|index| native_verified_points[index])
        .collect::<Vec<_>>();
    let points = native_verified_points
        .iter()
        .map(|&(a, b, _)| (a, b))
        .collect::<Vec<_>>();
    // Prefer fewer camera parameters only on the same verified native points
    // and with the existing median, scale, orientation and support gates.
    let regularized = native_station_regularized_candidate(
        &points,
        &homography,
        left.0.dimensions(),
        right.0.dimensions(),
        &left.1,
        &right.1,
    );
    let homography = regularized.unwrap_or(homography);
    diagnostic.fitted_inliers = points.len();
    diagnostic.hull_support = station_relation_spatial_support_with(
        &points,
        &homography,
        left.0.dimensions(),
        right.0.dimensions(),
        StationOverlapMeasure::Covered {
            source: &left.1,
            target: &right.1,
            factor: 1,
        },
    );
    diagnostic.residual_magnitude_px = residual_distribution(match homography.try_inverse() {
        Some(inverse) => points
            .iter()
            .map(|(a, b)| symmetric_point_error(&homography, &inverse, *a, *b))
            .collect(),
        None => Vec::new(),
    });
    diagnostic.failure_stage = "insufficient_native_polished_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT
        || median_symmetric_error(&homography, &points) > STATION_RELATION_MAX_MEDIAN_ERROR_PX
    {
        return None;
    }
    diagnostic.residual_model = if homography[(2, 0)] == 0.0 && homography[(2, 1)] == 0.0 {
        "native_polished_affine"
    } else if regularized.is_some() {
        "native_polished_regularized_projective"
    } else {
        "native_polished_projective"
    }
    .to_string();
    diagnostic.failure_stage = "fitted".to_string();
    Some(MatchInfo {
        homography,
        inliers: points.len(),
        points,
        canonical_homography: None,
        native_verified_points,
        ..seed.clone()
    })
}
