//! Model selection for panorama and focus-stack pair transforms.
//!
//! Stability checks, the large-panorama model ladder, the mixed-focal similarity
//! RANSAC and the focus-stack transform choice with shifted-mosaic detection.

use crate::panorama_utils::processing;
use crate::panorama_utils::stack_pipeline::determinism;
use nalgebra::Matrix3;
use nalgebra::Point2;
use rand::prelude::*;

use super::{
    AlignmentMode, FOCUS_MODEL_INLIER_THRESHOLD, FOCUS_MODEL_MIN_INLIERS,
    FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT, MIXED_FOCAL_SCALE_MAX_RATIO, MIXED_FOCAL_SCALE_MIN_RATIO,
    PANORAMA_MODEL_INLIER_THRESHOLD, PANORAMA_MODEL_MIN_INLIERS, RobustTransformFit,
    estimate_affine, estimate_similarity, estimate_translation, median_symmetric_error,
    panorama_spatial_support, robust_transform_fit, symmetric_inlier_indices,
    transform_is_stable_for_focus_stack, transformed_point,
};

pub(super) const FOCUS_SHIFTED_MOSAIC_MOTION_RATIO: f64 = 0.015;

pub(super) const MIXED_FOCAL_SIMILARITY_RANSAC_ITERATIONS: usize = 2_500;

pub(super) const PANORAMA_MODEL_MIN_ERROR_GAIN: f64 = 0.12;

pub(super) const PANORAMA_MODEL_RELATIVE_ERROR_GAIN: f64 = 0.90;

pub(super) const PANORAMA_MODEL_SCALE_EPSILON: f64 = 0.012;

pub(super) const PANORAMA_MODEL_ROTATION_EPSILON: f64 = 0.004;

pub(super) fn panorama_transform_is_stable(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    if width <= 1.0 || height <= 1.0 || transform.try_inverse().is_none() {
        return false;
    }
    let corners = [
        Point2::new(0.0, 0.0),
        Point2::new(width, 0.0),
        Point2::new(width, height),
        Point2::new(0.0, height),
    ]
    .into_iter()
    .map(|point| transformed_point(transform, point))
    .collect::<Option<Vec<_>>>();
    let Some(corners) = corners else {
        return false;
    };

    let edge_scales = [
        (corners[1] - corners[0]).norm() / width.max(1.0),
        (corners[2] - corners[1]).norm() / height.max(1.0),
        (corners[2] - corners[3]).norm() / width.max(1.0),
        (corners[3] - corners[0]).norm() / height.max(1.0),
    ];
    if edge_scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale < 0.18 || *scale > 5.5)
    {
        return false;
    }
    let min_scale = edge_scales.iter().copied().fold(f64::INFINITY, f64::min);
    let max_scale = edge_scales
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !min_scale.is_finite() || max_scale / min_scale > 6.0 {
        return false;
    }

    let signed_double_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>();
    let area_ratio = signed_double_area / (2.0 * width * height);
    area_ratio.is_finite() && area_ratio.abs() >= 0.025 && area_ratio.abs() <= 30.0
}

pub(super) fn panorama_model_fit(
    transform: Matrix3<f64>,
    points: &[(Point2<f64>, Point2<f64>)],
) -> Option<(Matrix3<f64>, Vec<usize>, f64)> {
    let inliers = symmetric_inlier_indices(&transform, points, PANORAMA_MODEL_INLIER_THRESHOLD);
    if inliers.len() < PANORAMA_MODEL_MIN_INLIERS {
        return None;
    }
    let inlier_points = inliers
        .iter()
        .map(|&index| points[index])
        .collect::<Vec<_>>();
    // One deterministic refit removes the small bias introduced when a model
    // was estimated from a mixed set of descriptor inliers.
    let refitted = if transform[(2, 0)].abs() < 1e-14 && transform[(2, 1)].abs() < 1e-14 {
        // Keep the model family (translation/similarity/affine) selected by the
        // caller; a generic projective refit here would reintroduce drift.
        transform
    } else {
        processing::compute_homography(&inlier_points).unwrap_or(transform)
    };
    let error = median_symmetric_error(&refitted, &inlier_points);
    error.is_finite().then_some((refitted, inliers, error))
}

pub(super) fn panorama_model_linear_characteristics(transform: &Matrix3<f64>) -> (f64, f64, f64) {
    let a = transform[(0, 0)];
    let b = transform[(0, 1)];
    let c = transform[(1, 0)];
    let d = transform[(1, 1)];
    let scale_x = (a * a + c * c).sqrt();
    let scale_y = (b * b + d * d).sqrt();
    let scale = ((scale_x * scale_y).max(0.0)).sqrt();
    let rotation = 0.5 * (c - b).atan2(a + d);
    let anisotropy = if scale_x.max(scale_y) > f64::EPSILON {
        (scale_x - scale_y).abs() / scale_x.max(scale_y)
    } else {
        f64::INFINITY
    };
    (scale, rotation, anisotropy)
}

pub(super) fn mixed_focal_scale_is_plausible(
    transform: &Matrix3<f64>,
    source_focal: f64,
    target_focal: f64,
) -> bool {
    if !source_focal.is_finite()
        || !target_focal.is_finite()
        || source_focal <= 0.0
        || target_focal <= 0.0
    {
        return true;
    }
    let expected = target_focal / source_focal;
    let (scale, _, anisotropy) = panorama_model_linear_characteristics(transform);
    if !expected.is_finite() || !scale.is_finite() || anisotropy > 0.35 {
        return false;
    }
    let ratio = scale / expected;
    ratio.is_finite()
        && (MIXED_FOCAL_SCALE_MIN_RATIO..=MIXED_FOCAL_SCALE_MAX_RATIO).contains(&ratio)
}

pub(super) fn find_mixed_focal_similarity_ransac(
    points: &[(Point2<f64>, Point2<f64>)],
    source_focal: f64,
    target_focal: f64,
    inlier_threshold: f64,
    minimum_inliers: usize,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    if points.len() < 2 || !inlier_threshold.is_finite() || inlier_threshold <= 0.0 {
        return None;
    }

    // See `find_homography_ransac_points_with_solver`: site constant mixed with
    // the content derived run seed, no time and no thread id (需求 14.6).
    let mut rng = StdRng::seed_from_u64(
        0xB4D4_5EED_5CA1_E001u64
            ^ (points.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ determinism::run_random_seed_mix(),
    );
    let all_indices: Vec<usize> = (0..points.len()).collect();
    let required_inliers = minimum_inliers.max(2);
    let mut best_transform = None;
    let mut best_inliers = Vec::new();
    let mut best_median = f64::INFINITY;

    for _ in 0..MIXED_FOCAL_SIMILARITY_RANSAC_ITERATIONS {
        let sample_indices: Vec<usize> = all_indices.sample(&mut rng, 2).copied().collect();
        if sample_indices.len() != 2 {
            continue;
        }
        let sample = sample_indices
            .iter()
            .map(|&index| points[index])
            .collect::<Vec<_>>();
        let source_span = (sample[1].0 - sample[0].0).norm();
        let target_span = (sample[1].1 - sample[0].1).norm();
        if source_span < 8.0 || target_span < 8.0 {
            continue;
        }
        let Some(transform) = estimate_similarity(&sample) else {
            continue;
        };
        if !mixed_focal_scale_is_plausible(&transform, source_focal, target_focal) {
            continue;
        }
        let inliers = symmetric_inlier_indices(&transform, points, inlier_threshold);
        if inliers.len() < required_inliers {
            continue;
        }
        let inlier_points = inliers
            .iter()
            .map(|&index| points[index])
            .collect::<Vec<_>>();
        let median = median_symmetric_error(&transform, &inlier_points);
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && median < best_median)
        {
            best_transform = Some(transform);
            best_inliers = inliers;
            best_median = median;
        }
    }

    if best_inliers.len() < required_inliers {
        return None;
    }
    let mut transform = best_transform?;
    let mut inliers = best_inliers;
    for _ in 0..4 {
        let inlier_points = inliers
            .iter()
            .map(|&index| points[index])
            .collect::<Vec<_>>();
        let Some(refitted) = estimate_similarity(&inlier_points) else {
            break;
        };
        if !mixed_focal_scale_is_plausible(&refitted, source_focal, target_focal) {
            break;
        }
        let refitted_inliers = symmetric_inlier_indices(&refitted, points, inlier_threshold);
        if refitted_inliers.len() < required_inliers {
            break;
        }
        transform = refitted;
        if refitted_inliers == inliers {
            break;
        }
        inliers = refitted_inliers;
    }

    Some((transform, inliers))
}

pub(super) fn select_large_panorama_transform(
    projective: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    source_dimensions: (u32, u32),
    log_selection: bool,
) -> Matrix3<f64> {
    if points.len() < PANORAMA_MODEL_MIN_INLIERS {
        return *projective;
    }

    let translation = panorama_model_fit(estimate_translation(points), points);
    let similarity = estimate_similarity(points).and_then(|model| {
        panorama_transform_is_stable(&model, source_dimensions)
            .then(|| panorama_model_fit(model, points))
            .flatten()
    });
    let affine = estimate_affine(points).and_then(|model| {
        panorama_transform_is_stable(&model, source_dimensions)
            .then(|| panorama_model_fit(model, points))
            .flatten()
    });
    let projective_fit = panorama_transform_is_stable(projective, source_dimensions)
        .then(|| panorama_model_fit(*projective, points))
        .flatten();

    let mut selected = translation
        .clone()
        .or_else(|| similarity.clone())
        .or_else(|| affine.clone())
        .or_else(|| projective_fit.clone());
    let mut selected_name = if translation.is_some() {
        "translation"
    } else if similarity.is_some() {
        "similarity"
    } else if affine.is_some() {
        "affine"
    } else {
        "projective"
    };

    // A pure translation is still the safest model for a conventional long
    // panorama.  Promote it only when the data demonstrates a real scale or
    // rotation change, or when the lower-DOF model cannot explain the overlap.
    if let (Some(current), Some(candidate)) = (selected.as_ref(), similarity.as_ref()) {
        let (scale, rotation, _) = panorama_model_linear_characteristics(&candidate.0);
        let current_error = current.2;
        let meaningful_motion = (scale - 1.0).abs() > PANORAMA_MODEL_SCALE_EPSILON
            || rotation.abs() > PANORAMA_MODEL_ROTATION_EPSILON;
        let materially_better = candidate.2 + PANORAMA_MODEL_MIN_ERROR_GAIN < current_error
            || candidate.2 <= current_error * PANORAMA_MODEL_RELATIVE_ERROR_GAIN;
        let sufficient_support = candidate.1.len() + 2 >= current.1.len();
        if meaningful_motion && materially_better && sufficient_support {
            selected = Some(candidate.clone());
            selected_name = "similarity";
        }
    }

    if let (Some(current), Some(candidate)) = (selected.as_ref(), affine.as_ref()) {
        let (_, _, anisotropy) = panorama_model_linear_characteristics(&candidate.0);
        let meaningful_deformation = anisotropy > 0.018
            || candidate.0[(2, 0)].abs() > 1e-8
            || candidate.0[(2, 1)].abs() > 1e-8;
        let materially_better = candidate.2 + PANORAMA_MODEL_MIN_ERROR_GAIN < current.2
            && candidate.2 <= current.2 * 0.92;
        if meaningful_deformation && materially_better && candidate.1.len() + 2 >= current.1.len() {
            selected = Some(candidate.clone());
            selected_name = "affine";
        }
    }

    if let (Some(current), Some(candidate)) = (selected.as_ref(), projective_fit.as_ref()) {
        let perspective_strength = projective[(2, 0)].abs() + projective[(2, 1)].abs();
        let materially_better = candidate.2 + 0.2 < current.2 && candidate.2 <= current.2 * 0.82;
        if perspective_strength > 1e-8
            && materially_better
            && candidate.1.len() + 3 >= current.1.len()
        {
            selected = Some(candidate.clone());
            selected_name = "projective";
        }
    }

    let selected = selected.map(|fit| fit.0).unwrap_or_else(|| {
        // Never hand an unbounded projective fit to the graph merely because
        // every scored model missed the minimum support. A finite translation
        // is the conservative fallback and cannot fold the output canvas.
        if panorama_transform_is_stable(projective, source_dimensions) {
            *projective
        } else {
            estimate_translation(points)
        }
    });
    if log_selection {
        let describe = |fit: &Option<(Matrix3<f64>, Vec<usize>, f64)>| {
            fit.as_ref()
                .map(|(_, inliers, error)| format!("{error:.3}px/{}", inliers.len()))
                .unwrap_or_else(|| "n/a".to_string())
        };
        println!(
            "  - Large-panorama alignment selected {selected_name}: translation {}, similarity {}, affine {}, projective {}",
            describe(&translation),
            describe(&similarity),
            describe(&affine),
            describe(&projective_fit),
        );
    }
    selected
}

pub(super) fn focus_fit_is_competitive(
    candidate: &RobustTransformFit,
    selected: &RobustTransformFit,
) -> bool {
    if candidate.inlier_indices.len() < FOCUS_MODEL_MIN_INLIERS {
        return false;
    }
    // A focus stack is normally captured from one stable camera pose. Prefer the
    // lower-error, lower-DOF model instead of allowing a slightly larger inlier
    // consensus to select an affine warp that bends the paper differently at each
    // edge. Extra support may win only when it does not materially worsen the fit.
    let materially_more_precise = candidate.median_error + 0.25 < selected.median_error
        && candidate.median_error <= selected.median_error * 0.90;
    let similar_precision_with_more_support = candidate.inlier_indices.len()
        >= selected.inlier_indices.len() + 4
        && candidate.median_error <= selected.median_error * 1.04;
    materially_more_precise || similar_precision_with_more_support
}

pub(super) fn select_focus_stack_transform(
    projective: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    source_dimensions: (u32, u32),
    alignment_mode: AlignmentMode,
) -> Matrix3<f64> {
    // A scale change is a shifted mosaic even when the optical centres line up
    // perfectly.  Use the broader, bounded geometry guard for that case; the
    // normal focus-stack guard intentionally rejects large scale changes so a
    // bad repeated-stroke match cannot bend a fixed-camera stack.
    let shifted_mosaic = focus_stack_motion_is_shifted_mosaic(points, source_dimensions);
    let stable_transform = |transform: &Matrix3<f64>| {
        if shifted_mosaic {
            panorama_transform_is_stable(transform, source_dimensions)
        } else {
            transform_is_stable_for_focus_stack(transform, source_dimensions)
        }
    };
    let translation = estimate_translation(points);
    let translation_inliers =
        symmetric_inlier_indices(&translation, points, FOCUS_MODEL_INLIER_THRESHOLD);
    let translation_is_valid = translation_inliers.len() >= FOCUS_MODEL_MIN_INLIERS;
    let translation_points: Vec<_> = translation_inliers
        .iter()
        .map(|&index| points[index])
        .collect();
    let mut selected = RobustTransformFit {
        transform: translation,
        inlier_indices: if translation_is_valid {
            translation_inliers
        } else {
            Vec::new()
        },
        median_error: if translation_is_valid {
            median_symmetric_error(&translation, &translation_points)
        } else {
            f64::INFINITY
        },
    };
    let mut selected_name = "translation";

    let similarity = robust_transform_fit(points, 2, 0xA24B_AED4_963E_E407, estimate_similarity);
    if let Some(model) = similarity.as_ref()
        && stable_transform(&model.transform)
        && focus_fit_is_competitive(model, &selected)
    {
        selected = model.clone();
        selected_name = "similarity";
    }

    let affine = robust_transform_fit(points, 3, 0x9FB2_1C65_1E98_DF25, estimate_affine);
    if let Some(model) = affine.as_ref()
        && stable_transform(&model.transform)
        && focus_fit_is_competitive(model, &selected)
    {
        selected = model.clone();
        selected_name = "affine";
    }

    let projective_error = median_symmetric_error(projective, points);
    let explicit_projective = matches!(
        alignment_mode,
        AlignmentMode::Perspective | AlignmentMode::Cylindrical | AlignmentMode::Spherical
    );
    // Auto mode may still need a homography for a handheld view of a flat
    // artwork. Require the inliers to cover a substantial part of both
    // frames before accepting it; a compact repeated brush stroke must stay
    // on the lower-DOF model even if its projective fit has a tiny residual.
    let projective_spatial_support =
        panorama_spatial_support(points, source_dimensions, source_dimensions);
    let allow_projective =
        explicit_projective || projective_spatial_support >= FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT;
    if allow_projective
        && stable_transform(projective)
        && projective_error <= FOCUS_MODEL_INLIER_THRESHOLD * 0.5
        && projective_error + 0.15 < selected.median_error
        && projective_error <= selected.median_error * 0.80
    {
        selected.transform = *projective;
        selected.inlier_indices = (0..points.len()).collect();
        selected.median_error = projective_error;
        selected_name = "projective";
    }

    let similarity_summary = similarity
        .as_ref()
        .map(|fit| format!("{:.3}px/{}", fit.median_error, fit.inlier_indices.len()))
        .unwrap_or_else(|| "n/a".to_string());
    let affine_summary = affine
        .as_ref()
        .map(|fit| format!("{:.3}px/{}", fit.median_error, fit.inlier_indices.len()))
        .unwrap_or_else(|| "n/a".to_string());
    let translation_summary = if translation_is_valid {
        format!(
            "{:.3}px/{}",
            median_symmetric_error(&translation, &translation_points),
            translation_points.len()
        )
    } else {
        format!("invalid/{}", translation_points.len())
    };
    println!(
        "  - Focus alignment selected {selected_name}: median symmetric error {:.3}px with {} inliers (translation {translation_summary}, similarity {similarity_summary}, affine {affine_summary}, projective {:.3}px/{}, shifted-mosaic {})",
        selected.median_error,
        selected.inlier_indices.len(),
        projective_error,
        points.len(),
        shifted_mosaic
    );
    selected.transform
}

pub(super) fn focus_stack_motion_is_shifted_mosaic(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    source_dimensions: (u32, u32),
) -> bool {
    if points.len() < FOCUS_MODEL_MIN_INLIERS {
        return false;
    }
    let translation = estimate_translation(points);
    let motion = nalgebra::Vector2::new(translation[(0, 2)], translation[(1, 2)]).norm();
    let image_scale = source_dimensions.0.max(source_dimensions.1) as f64;
    let translation_shift = motion.is_finite()
        && image_scale > 1.0
        && motion > image_scale * FOCUS_SHIFTED_MOSAIC_MOTION_RATIO;
    let scale_or_rotation_shift = estimate_similarity(points)
        .map(|similarity| {
            let scale_x = similarity[(0, 0)].hypot(similarity[(1, 0)]);
            let scale_y = similarity[(0, 1)].hypot(similarity[(1, 1)]);
            let scale = (scale_x.max(0.0) * scale_y.max(0.0)).sqrt();
            let rotation = 0.5
                * (similarity[(1, 0)] - similarity[(0, 1)])
                    .atan2(similarity[(0, 0)] + similarity[(1, 1)]);
            (scale.is_finite() && (scale - 1.0).abs() > 0.10)
                || (rotation.is_finite() && rotation.abs() > 0.035)
        })
        .unwrap_or(false);
    translation_shift || scale_or_rotation_shift
}
