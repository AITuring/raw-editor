//! Prior-free and source-match proposals for Virtual_Tile relations.
//!
//! Masked global similarity search with peak selection, Source_RAW match seeds
//! and the prior-free covered-feature relation with its fallback conditions.

use crate::panorama_utils::processing;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::cmp::Ordering;
use std::collections::HashMap;

use super::{
    Feature, ImageInfo, MatchInfo, STATION_RELATION_MIN_INLIERS, VirtualTileGlobalSearchPlane,
    VirtualTileMatchPlane, homography_preserves_focus_orientation, normalized_homography,
    transformed_point, virtual_tile_ncc,
};

pub(super) const VIRTUAL_TILE_SEED_SOURCE_MATCH: &str = "source_pairwise_match";

pub(super) const VIRTUAL_TILE_SEED_GLOBAL_COVERAGE: &str = "global_coverage_similarity";

pub(super) const VIRTUAL_TILE_SOURCE_MATCH_MAX_SEEDS: usize = 4;

pub(super) const VIRTUAL_TILE_GLOBAL_TOP_K: usize = 8;

pub(super) const VIRTUAL_TILE_GLOBAL_MAX_SAMPLES: usize = 512;

pub(super) const VIRTUAL_TILE_GLOBAL_MIN_OVERLAP_SAMPLES: usize = 32;

#[derive(Clone, Copy, Debug)]
pub(super) struct VirtualTileSourceMatchProposal {
    pub(super) left_to_right: Matrix3<f64>,
    pub(super) inliers: usize,
    pub(super) left_source: usize,
    pub(super) right_source: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct VirtualTileGlobalProposal {
    pub(super) left_to_right: Matrix3<f64>,
    pub(super) score: f64,
    pub(super) overlap_samples: usize,
    pub(super) left_factor: u32,
    pub(super) right_factor: u32,
    pub(super) scale: f64,
    pub(super) rotation_degrees: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct VirtualTileGlobalPeak {
    pub(super) transform: Matrix3<f64>,
    pub(super) score: f64,
    pub(super) overlap_samples: usize,
    pub(super) translation: (f64, f64),
    pub(super) scale: f64,
    pub(super) rotation_degrees: f64,
}

pub(super) fn virtual_tile_translation(tx: f64, ty: f64) -> Matrix3<f64> {
    Matrix3::new(1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0)
}

/// Sparse masked zero-mean luminance/gradient NCC. This score only ranks
/// proposal basins; it is never consulted by a Station_Relation gate.
pub(super) fn virtual_tile_masked_similarity_score(
    left: &VirtualTileGlobalSearchPlane,
    right: &VirtualTileGlobalSearchPlane,
    transform: &Matrix3<f64>,
    samples: &[(u32, u32)],
) -> Option<(f64, usize)> {
    let mut left_values = Vec::new();
    let mut right_values = Vec::new();
    let mut left_gradients = Vec::new();
    let mut right_gradients = Vec::new();
    for &(x, y) in samples {
        let source_index = (y * left.width + x) as usize;
        let mapped = transformed_point(transform, Point2::new(f64::from(x), f64::from(y)))?;
        let (target_x, target_y) = (mapped.x.round() as i32, mapped.y.round() as i32);
        let Some(target_index) = right.index(target_x, target_y) else {
            continue;
        };
        if right.covered_count[target_index] == 0 {
            continue;
        }
        left_values.push(left.luminance[source_index]);
        right_values.push(right.luminance[target_index]);
        if left.gradient[source_index] > 0.0 && right.gradient[target_index] > 0.0 {
            left_gradients.push(left.gradient[source_index]);
            right_gradients.push(right.gradient[target_index]);
        }
    }
    let minimum_overlap = VIRTUAL_TILE_GLOBAL_MIN_OVERLAP_SAMPLES.max(samples.len().div_ceil(5));
    if left_values.len() < minimum_overlap {
        return None;
    }
    let luminance = virtual_tile_ncc(&left_values, &right_values)?;
    let score = virtual_tile_ncc(&left_gradients, &right_gradients)
        .map_or(luminance, |gradient| 0.7 * luminance + 0.3 * gradient);
    score.is_finite().then_some((score, left_values.len()))
}

pub(super) fn virtual_tile_global_peak_order(
    left: &VirtualTileGlobalPeak,
    right: &VirtualTileGlobalPeak,
) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| right.overlap_samples.cmp(&left.overlap_samples))
        .then_with(|| left.scale.total_cmp(&right.scale))
        .then_with(|| left.rotation_degrees.total_cmp(&right.rotation_degrees))
        .then_with(|| left.translation.1.total_cmp(&right.translation.1))
        .then_with(|| left.translation.0.total_cmp(&right.translation.0))
}

pub(super) fn virtual_tile_global_select_peaks(
    mut candidates: Vec<VirtualTileGlobalPeak>,
    limit: usize,
    translation_separation: f64,
) -> Vec<VirtualTileGlobalPeak> {
    candidates.sort_by(virtual_tile_global_peak_order);
    let mut selected = Vec::<VirtualTileGlobalPeak>::new();
    for candidate in candidates {
        if selected.iter().any(|existing| {
            (candidate.translation.0 - existing.translation.0)
                .hypot(candidate.translation.1 - existing.translation.1)
                < translation_separation
                && (candidate.scale - existing.scale).abs() < 0.0125
                && (candidate.rotation_degrees - existing.rotation_degrees).abs() < 2.5
        }) {
            continue;
        }
        selected.push(candidate);
        if selected.len() == limit {
            break;
        }
    }
    selected
}

/// Coverage-aware, full-translation, bounded-similarity, multi-peak search.
/// Returned matrices are native `left -> right` seeds for the existing dense
/// pyramid path; none is relation evidence by itself.
pub(super) fn virtual_tile_global_similarity_proposals(
    left: &VirtualTileGlobalSearchPlane,
    right: &VirtualTileGlobalSearchPlane,
) -> Vec<VirtualTileGlobalProposal> {
    let covered = (0..left.height)
        .flat_map(|y| (0..left.width).map(move |x| (x, y)))
        .filter(|&(x, y)| left.covered_count[(y * left.width + x) as usize] > 0)
        .collect::<Vec<_>>();
    if covered.len() < VIRTUAL_TILE_GLOBAL_MIN_OVERLAP_SAMPLES {
        return Vec::new();
    }
    let sample_stride = covered
        .len()
        .div_ceil(VIRTUAL_TILE_GLOBAL_MAX_SAMPLES)
        .max(1);
    let samples = covered
        .into_iter()
        .step_by(sample_stride)
        .take(VIRTUAL_TILE_GLOBAL_MAX_SAMPLES)
        .collect::<Vec<_>>();
    let mut coarse = Vec::new();
    for scale in [0.95f64, 1.0, 1.05] {
        for rotation_degrees in [-10.0f64, 0.0, 10.0] {
            let theta = rotation_degrees.to_radians();
            let (sin, cos) = theta.sin_cos();
            let level_scale = scale * f64::from(left.factor) / f64::from(right.factor);
            let linear = Matrix3::new(
                level_scale * cos,
                -level_scale * sin,
                0.0,
                level_scale * sin,
                level_scale * cos,
                0.0,
                0.0,
                0.0,
                1.0,
            );
            let corners = [
                Point2::new(0.0, 0.0),
                Point2::new(f64::from(left.width.saturating_sub(1)), 0.0),
                Point2::new(0.0, f64::from(left.height.saturating_sub(1))),
                Point2::new(
                    f64::from(left.width.saturating_sub(1)),
                    f64::from(left.height.saturating_sub(1)),
                ),
            ]
            .into_iter()
            .filter_map(|point| transformed_point(&linear, point))
            .collect::<Vec<_>>();
            let min_x = corners
                .iter()
                .map(|point| point.x)
                .fold(f64::INFINITY, f64::min);
            let max_x = corners
                .iter()
                .map(|point| point.x)
                .fold(f64::NEG_INFINITY, f64::max);
            let min_y = corners
                .iter()
                .map(|point| point.y)
                .fold(f64::INFINITY, f64::min);
            let max_y = corners
                .iter()
                .map(|point| point.y)
                .fold(f64::NEG_INFINITY, f64::max);
            let tx_min = (-max_x).floor() as i32;
            let tx_max = (f64::from(right.width.saturating_sub(1)) - min_x).ceil() as i32;
            let ty_min = (-max_y).floor() as i32;
            let ty_max = (f64::from(right.height.saturating_sub(1)) - min_y).ceil() as i32;
            let translation_step = if scale == 1.0 && rotation_degrees == 0.0 {
                1usize
            } else {
                4usize
            };
            for ty in (ty_min..=ty_max).step_by(translation_step) {
                for tx in (tx_min..=tx_max).step_by(translation_step) {
                    let transform = virtual_tile_translation(f64::from(tx), f64::from(ty)) * linear;
                    if let Some((score, overlap_samples)) =
                        virtual_tile_masked_similarity_score(left, right, &transform, &samples)
                    {
                        coarse.push(VirtualTileGlobalPeak {
                            transform,
                            score,
                            overlap_samples,
                            translation: (f64::from(tx), f64::from(ty)),
                            scale,
                            rotation_degrees,
                        });
                    }
                }
            }
        }
    }
    let coarse = virtual_tile_global_select_peaks(coarse, 32, 10.0);
    let mut refined = Vec::new();
    for peak in coarse {
        for scale in [peak.scale - 0.0125, peak.scale, peak.scale + 0.0125]
            .into_iter()
            .map(|scale| scale.clamp(0.95, 1.05))
        {
            for rotation_degrees in [
                peak.rotation_degrees - 2.5,
                peak.rotation_degrees,
                peak.rotation_degrees + 2.5,
            ]
            .into_iter()
            .map(|rotation| rotation.clamp(-10.0, 10.0))
            {
                let theta = rotation_degrees.to_radians();
                let (sin, cos) = theta.sin_cos();
                let level_scale = scale * f64::from(left.factor) / f64::from(right.factor);
                let linear = Matrix3::new(
                    level_scale * cos,
                    -level_scale * sin,
                    0.0,
                    level_scale * sin,
                    level_scale * cos,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                );
                for dy in -3..=3 {
                    for dx in -3..=3 {
                        let tx = peak.translation.0 + f64::from(dx);
                        let ty = peak.translation.1 + f64::from(dy);
                        let transform = virtual_tile_translation(tx, ty) * linear;
                        if let Some((score, overlap_samples)) =
                            virtual_tile_masked_similarity_score(left, right, &transform, &samples)
                        {
                            refined.push(VirtualTileGlobalPeak {
                                transform,
                                score,
                                overlap_samples,
                                translation: (tx, ty),
                                scale,
                                rotation_degrees,
                            });
                        }
                    }
                }
            }
        }
    }
    virtual_tile_global_select_peaks(refined, VIRTUAL_TILE_GLOBAL_TOP_K, 12.0)
        .into_iter()
        .filter_map(|peak| {
            let native = normalized_homography(
                &(right.level_to_native() * peak.transform * left.native_to_level()),
            )?;
            Some(VirtualTileGlobalProposal {
                left_to_right: native,
                score: peak.score,
                overlap_samples: peak.overlap_samples,
                left_factor: left.factor,
                right_factor: right.factor,
                scale: peak.scale,
                rotation_degrees: peak.rotation_degrees,
            })
        })
        .collect()
}

/// Lift independently measured Source_RAW relations into the two fused
/// station tiles' local pixel coordinates.  These transforms are search seeds
/// only: the Virtual_Tile pyramid still measures bidirectionally and the
/// shared Station_Relation selector remains the sole acceptance authority.
///
/// A source match is stored in `(min source id, max source id)` direction.  The
/// composition below maps a left-tile pixel to the corresponding right-tile
/// pixel through the measured source relation and the current source/world
/// poses.  Keeping this conversion explicit avoids treating a world-space
/// station prior as if it were already a local tile transform when a station
/// crop or pose has moved.
pub(super) fn virtual_tile_source_match_proposals(
    left_group: &[&ImageInfo],
    right_group: &[&ImageInfo],
    pairwise_matches: &HashMap<(usize, usize), MatchInfo>,
    source_to_world: &HashMap<usize, Matrix3<f64>>,
    left_tile_to_world: &Matrix3<f64>,
    right_tile_to_world: &Matrix3<f64>,
    source_render_scale: f64,
) -> Vec<VirtualTileSourceMatchProposal> {
    if !source_render_scale.is_finite() || source_render_scale <= 0.0 {
        return Vec::new();
    }
    let Some(right_world_to_tile) = right_tile_to_world.try_inverse() else {
        return Vec::new();
    };
    // `pairwise_matches` are measured in full-resolution Source_RAW pixels,
    // while `source_to_world` is built from `scaled_source_render_homographies`
    // and therefore consumes render-scaled source pixels. Move the measured
    // relation into that same coordinate system before composing it with the
    // station/world poses. At 1:1 this is the identity.
    let source_scale = Matrix3::new(
        source_render_scale,
        0.0,
        0.0,
        0.0,
        source_render_scale,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let Some(source_scale_inverse) = source_scale.try_inverse() else {
        return Vec::new();
    };
    let mut proposals = Vec::new();
    for left in left_group {
        let Some(left_world_to_source) = source_to_world
            .get(&left.id)
            .and_then(|matrix| matrix.try_inverse())
        else {
            continue;
        };
        for right in right_group {
            let Some(right_world) = source_to_world.get(&right.id) else {
                continue;
            };
            let key = (left.id.min(right.id), left.id.max(right.id));
            let Some(measurement) = pairwise_matches.get(&key) else {
                continue;
            };
            // Filename/sequence and an ordinary coarse retrieval edge are
            // ordering or basin hints, not independent pixel geometry. A
            // mixed-focal coarse edge with top-candidate consensus is the one
            // exception: it already carries multiple image-derived candidate
            // correspondences, so it may seed a search basin. The downstream
            // Virtual_Tile measurement and selector still gate the result.
            if measurement.sequence_bridge
                || (measurement.coarse_bridge && measurement.top_candidate_points.is_empty())
            {
                continue;
            }
            let Some(source_to_target) = (if left.id <= right.id {
                Some(measurement.homography)
            } else {
                measurement.homography.try_inverse()
            }) else {
                continue;
            };
            let source_to_target = source_scale * source_to_target * source_scale_inverse;
            let Some(left_to_right) = normalized_homography(
                &(right_world_to_tile
                    * *right_world
                    * source_to_target
                    * left_world_to_source
                    * *left_tile_to_world),
            ) else {
                continue;
            };
            if left_to_right.try_inverse().is_none() {
                continue;
            }
            proposals.push(VirtualTileSourceMatchProposal {
                left_to_right,
                inliers: measurement.inliers.max(measurement.points.len()),
                left_source: left.id,
                right_source: right.id,
            });
        }
    }
    proposals.sort_by(|left, right| {
        right
            .inliers
            .cmp(&left.inliers)
            .then_with(|| left.left_source.cmp(&right.left_source))
            .then_with(|| left.right_source.cmp(&right.right_source))
    });
    proposals.truncate(VIRTUAL_TILE_SOURCE_MATCH_MAX_SEEDS);
    proposals
}

/// Prior-free covered-feature relation between two finest planes, in native
/// tile pixels, with its RANSAC inlier count. It starts as a proposal; only a
/// subsequent independently polished hypothesis that passes the complete
/// shared Station_Relation gate may become authoritative.
pub(super) fn virtual_tile_prior_free_relation(
    left_factor: u32,
    left_features: &[Feature],
    right_factor: u32,
    right_features: &[Feature],
    left_native_dimensions: (u32, u32),
) -> Option<(Matrix3<f64>, usize)> {
    let matches = processing::match_features(left_features, right_features);
    let correspondences = matches
        .iter()
        .map(|matched| {
            let left = left_features[matched.index1].keypoint;
            let right = right_features[matched.index2].keypoint;
            (
                Point2::new(f64::from(left.x), f64::from(left.y)),
                Point2::new(f64::from(right.x), f64::from(right.y)),
            )
        })
        .collect::<Vec<_>>();
    let (homography, inliers) = processing::find_homography_ransac_points_stable_with_min_inliers(
        &correspondences,
        5.0,
        STATION_RELATION_MIN_INLIERS,
    )?;
    if inliers.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    let native = normalized_homography(
        &(VirtualTileMatchPlane::level_to_native(right_factor)
            * homography
            * VirtualTileMatchPlane::native_to_level(left_factor)),
    )?;
    (native.try_inverse().is_some()
        && homography_preserves_focus_orientation(&native, left_native_dimensions))
    .then_some((native, inliers.len()))
}

/// A covered-feature seed can still repair a rendering prior when the
/// subsequent finest-plane polish fitted a model but its inlier hull missed
/// the fixed spatial-support gate. The seed remains a proposal only: the
/// repaired station is rendered again and the ordinary bounded matcher must
/// accept the resulting relation before it enters the pose solver.
pub(super) fn virtual_tile_prior_free_seed_fallback_conditions(
    model_fitted: bool,
    failure_stage: &str,
    fitted_inliers: usize,
    prior_free_inliers: usize,
) -> bool {
    model_fitted
        && failure_stage == "insufficient_polished_support"
        && fitted_inliers >= STATION_RELATION_MIN_INLIERS
        // A covered-feature seed is only a repair proposal when it carries at
        // least as much support as the model that the finest-plane polish
        // already fitted.  A smaller seed can move an entire station cluster
        // on a repeated texture even when the polished model has hundreds of
        // mutually consistent matches but fails only its spatial-support gate.
        && prior_free_inliers >= fitted_inliers
}

pub(super) fn virtual_tile_prior_free_seed_fallback_allowed(
    diagnostic: &stack_report::StationRelationCandidateRecord,
) -> bool {
    virtual_tile_prior_free_seed_fallback_conditions(
        diagnostic.model_fitted,
        &diagnostic.failure_stage,
        diagnostic.fitted_inliers,
        diagnostic.prior_free_inliers,
    )
}
