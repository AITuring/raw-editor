//! Pose parametrisation and observations of the global focus solve.
//!
//! Normalised pose and homography conversions and the regional, generic-region
//! and whole-frame correspondences fed to the global optimisation.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::MatchInfo;
use crate::panorama_utils::stack_pipeline::determinism::sorted_pairs;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;
use std::collections::HashMap;

use super::{
    FOCUS_LOCAL_MODEL_MIN_INLIERS, RobustTransformFit, estimate_affine,
    focus_match_is_local_bracket, robust_transform_fit, transform_is_stable_for_focus_stack,
};

pub(super) const FOCUS_GLOBAL_MAX_POINTS_PER_EDGE: usize = 256;

pub(super) const FOCUS_BRACKET_OBSERVATION_WEIGHT: f64 = 16.0;

pub(super) const FOCUS_LOCAL_MODEL_MAX_DISPLACEMENT_RATIO: f64 = 0.08;

// Regional matches are measured against the already registered global model.
// Keep this close to the dense-search radius after conversion back to the
// source coordinate system. A wide gate can accept a different repeated stroke
// and let an under-constrained local fit bend an entire band.
pub(super) const FOCUS_LOCAL_MATCH_RESIDUAL_RATIO: f64 = 0.008;

pub(super) type FocusPose = [f64; 8];

#[derive(Clone, Copy)]
pub(super) struct FocusGlobalObservation {
    pub(super) source_index: usize,
    pub(super) target_index: usize,
    pub(super) source: Point2<f64>,
    pub(super) target: Point2<f64>,
    pub(super) weight: f64,
}

pub(super) fn focus_pose_matrix(pose: &FocusPose) -> Matrix3<f64> {
    Matrix3::new(
        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], pose[7], 1.0,
    )
}

pub(super) fn focus_matrix_pose(matrix: &Matrix3<f64>) -> Option<FocusPose> {
    let normalization = matrix[(2, 2)];
    if !normalization.is_finite() || normalization.abs() < 1e-8 {
        return None;
    }
    let pose = [
        matrix[(0, 0)] / normalization,
        matrix[(0, 1)] / normalization,
        matrix[(0, 2)] / normalization,
        matrix[(1, 0)] / normalization,
        matrix[(1, 1)] / normalization,
        matrix[(1, 2)] / normalization,
        matrix[(2, 0)] / normalization,
        matrix[(2, 1)] / normalization,
    ];
    pose.iter().all(|value| value.is_finite()).then_some(pose)
}

pub(super) fn focus_local_norm_to_full(image: &ImageInfo, coordinate_scale: f64) -> Matrix3<f64> {
    Matrix3::new(
        coordinate_scale,
        0.0,
        image.width as f64 * 0.5,
        0.0,
        coordinate_scale,
        image.height as f64 * 0.5,
        0.0,
        0.0,
        1.0,
    )
}

pub(super) fn focus_world_full_to_norm(
    reference: &ImageInfo,
    coordinate_scale: f64,
) -> Matrix3<f64> {
    Matrix3::new(
        coordinate_scale.recip(),
        0.0,
        -(reference.width as f64 * 0.5) / coordinate_scale,
        0.0,
        coordinate_scale.recip(),
        -(reference.height as f64 * 0.5) / coordinate_scale,
        0.0,
        0.0,
        1.0,
    )
}

pub(super) fn focus_world_norm_to_full(
    reference: &ImageInfo,
    coordinate_scale: f64,
) -> Matrix3<f64> {
    focus_local_norm_to_full(reference, coordinate_scale)
}

pub(super) fn focus_normalized_point(
    image: &ImageInfo,
    point: Point2<f64>,
    coordinate_scale: f64,
) -> Point2<f64> {
    Point2::new(
        (point.x - image.width as f64 * 0.5) / coordinate_scale,
        (point.y - image.height as f64 * 0.5) / coordinate_scale,
    )
}

pub(super) fn focus_pose_to_full_homography(
    pose: &FocusPose,
    image: &ImageInfo,
    reference: &ImageInfo,
    coordinate_scale: f64,
) -> Option<Matrix3<f64>> {
    let local_full_to_norm = Matrix3::new(
        coordinate_scale.recip(),
        0.0,
        -(image.width as f64 * 0.5) / coordinate_scale,
        0.0,
        coordinate_scale.recip(),
        -(image.height as f64 * 0.5) / coordinate_scale,
        0.0,
        0.0,
        1.0,
    );
    let full = focus_world_norm_to_full(reference, coordinate_scale)
        * focus_pose_matrix(pose)
        * local_full_to_norm;
    focus_matrix_pose(&full).map(|normalized| focus_pose_matrix(&normalized))
}

pub(super) fn focus_homography_to_pose(
    homography: &Matrix3<f64>,
    image: &ImageInfo,
    reference: &ImageInfo,
    coordinate_scale: f64,
) -> Option<FocusPose> {
    let normalized = focus_world_full_to_norm(reference, coordinate_scale)
        * homography
        * focus_local_norm_to_full(image, coordinate_scale);
    focus_matrix_pose(&normalized)
}

pub(super) fn focus_pose_projection(
    pose: &FocusPose,
    point: Point2<f64>,
) -> Option<(Point2<f64>, [f64; 8], [f64; 8])> {
    let x = point.x;
    let y = point.y;
    let numerator_x = pose[0] * x + pose[1] * y + pose[2];
    let numerator_y = pose[3] * x + pose[4] * y + pose[5];
    let denominator = pose[6] * x + pose[7] * y + 1.0;
    if !denominator.is_finite() || denominator.abs() < 1e-8 {
        return None;
    }
    let inverse_denominator = denominator.recip();
    let projected = Point2::new(
        numerator_x * inverse_denominator,
        numerator_y * inverse_denominator,
    );
    if !projected.x.is_finite() || !projected.y.is_finite() {
        return None;
    }
    let x_jacobian = [
        x * inverse_denominator,
        y * inverse_denominator,
        inverse_denominator,
        0.0,
        0.0,
        0.0,
        -projected.x * x * inverse_denominator,
        -projected.x * y * inverse_denominator,
    ];
    let y_jacobian = [
        0.0,
        0.0,
        0.0,
        x * inverse_denominator,
        y * inverse_denominator,
        inverse_denominator,
        -projected.y * x * inverse_denominator,
        -projected.y * y * inverse_denominator,
    ];
    Some((projected, x_jacobian, y_jacobian))
}

pub(super) fn focus_match_points_for_region(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    match_info: &MatchInfo,
    normalized_y_range: (f64, f64),
) -> Vec<(Point2<f64>, Point2<f64>)> {
    if match_info.coarse_bridge {
        return Vec::new();
    }
    let Some(source_foreground_range) = source_image.foreground_range else {
        return Vec::new();
    };
    let Some(target_foreground_range) = target_image.foreground_range else {
        return Vec::new();
    };
    let mut selected_points = match_info.dense_focus_points.clone();
    selected_points.extend(match_info.foreground_feature_points.iter().copied());
    if selected_points.len() < FOCUS_LOCAL_MODEL_MIN_INLIERS {
        selected_points.extend(match_info.top_candidate_points.iter().copied());
    }
    let maximum_residual =
        source_image.width.max(source_image.height) as f64 * FOCUS_LOCAL_MATCH_RESIDUAL_RATIO;
    let filtered = selected_points
        .iter()
        .copied()
        .filter(|(source, target)| {
            let mapped = match_info.homography * Point3::new(source.x, source.y, 1.0);
            if mapped.z.abs() < 1e-8 {
                return false;
            }
            let predicted = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
            if (predicted - *target).norm() > maximum_residual {
                return false;
            }
            let source_y = source.y / source_image.height.max(1) as f64;
            let target_y = target.y / target_image.height.max(1) as f64;
            source_y >= normalized_y_range.0
                && source_y <= normalized_y_range.1
                && target_y >= normalized_y_range.0
                && target_y <= normalized_y_range.1
                && source_y >= source_foreground_range.0
                && source_y <= source_foreground_range.1
                && target_y >= target_foreground_range.0
                && target_y <= target_foreground_range.1
        })
        .collect::<Vec<_>>();
    if filtered.len() < FOCUS_LOCAL_MODEL_MIN_INLIERS {
        return Vec::new();
    }

    // A narrow depth layer can provide only a few long, mostly one-dimensional edges.
    // The general panorama homography solver intentionally requires 15 inliers,
    // which rejects otherwise useful local models from a narrow foreground band.
    // Fit an affine model here with the focus-specific minimum instead; the
    // stability check below still prevents a sparse or degenerate fit from being
    // applied to the rendered layer.
    let Some(regional_fit) =
        robust_transform_fit(&filtered, 3, 0x5EED_7A11_4F52_9B31, estimate_affine)
    else {
        return Vec::new();
    };
    let RobustTransformFit {
        transform: regional_homography,
        inlier_indices,
        median_error: _,
    } = regional_fit;
    let inlier_points = inlier_indices
        .iter()
        .map(|&index| filtered[index])
        .collect::<Vec<_>>();
    if inlier_indices.len() < FOCUS_LOCAL_MODEL_MIN_INLIERS
        || !transform_is_stable_for_focus_stack(&regional_homography, source_image.dimensions())
        || !focus_regional_support_is_diverse(
            &inlier_points,
            source_image.dimensions(),
            target_image.dimensions(),
        )
    {
        return Vec::new();
    }

    let maximum_displacement = [0.0, 0.5, 1.0]
        .into_iter()
        .flat_map(|x| {
            [normalized_y_range.0, normalized_y_range.1]
                .into_iter()
                .map(move |y| {
                    Point2::new(
                        source_image.width as f64 * x,
                        source_image.height as f64 * y,
                    )
                })
        })
        .filter_map(|point| {
            let local = regional_homography * nalgebra::Point3::new(point.x, point.y, 1.0);
            let global = match_info.homography * nalgebra::Point3::new(point.x, point.y, 1.0);
            if local.z.abs() < 1e-8 || global.z.abs() < 1e-8 {
                return None;
            }
            Some(
                (Point2::new(local.x / local.z, local.y / local.z)
                    - Point2::new(global.x / global.z, global.y / global.z))
                .norm(),
            )
        })
        .fold(0.0, f64::max);
    if maximum_displacement
        <= source_image.width.max(source_image.height) as f64
            * FOCUS_LOCAL_MODEL_MAX_DISPLACEMENT_RATIO
    {
        inlier_indices
            .into_iter()
            .map(|index| filtered[index])
            .collect()
    } else {
        Vec::new()
    }
}

pub(super) fn focus_regional_support_is_diverse(
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) -> bool {
    if points.len() < FOCUS_LOCAL_MODEL_MIN_INLIERS {
        return false;
    }
    let source_min_x = points
        .iter()
        .map(|(source, _)| source.x)
        .fold(f64::INFINITY, f64::min);
    let source_max_x = points
        .iter()
        .map(|(source, _)| source.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let source_min_y = points
        .iter()
        .map(|(source, _)| source.y)
        .fold(f64::INFINITY, f64::min);
    let source_max_y = points
        .iter()
        .map(|(source, _)| source.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let target_min_x = points
        .iter()
        .map(|(_, target)| target.x)
        .fold(f64::INFINITY, f64::min);
    let target_max_x = points
        .iter()
        .map(|(_, target)| target.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let target_min_y = points
        .iter()
        .map(|(_, target)| target.y)
        .fold(f64::INFINITY, f64::min);
    let target_max_y = points
        .iter()
        .map(|(_, target)| target.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let source_width = source_dimensions.0.max(1) as f64;
    let source_height = source_dimensions.1.max(1) as f64;
    let target_width = target_dimensions.0.max(1) as f64;
    let target_height = target_dimensions.1.max(1) as f64;
    let source_span_x = (source_max_x - source_min_x) / source_width;
    let source_span_y = (source_max_y - source_min_y) / source_height;
    let target_span_x = (target_max_x - target_min_x) / target_width;
    let target_span_y = (target_max_y - target_min_y) / target_height;
    let source_area = source_span_x * source_span_y;
    let target_area = target_span_x * target_span_y;

    // A local model needs support in both axes. The area fallback permits a
    // long, thin object such as a mat edge, but a single screw/corner cluster
    // cannot satisfy it and therefore cannot extrapolate a warp over the band.
    let source_supported = (source_span_x >= 0.10 && source_span_y >= 0.035)
        || source_area >= 0.006
        // A long, well-supported edge is intentionally one-dimensional. It
        // constrains the vertical translation/shear that causes a stepped
        // holder or paper boundary, while the global pose prior still keeps
        // the under-constrained horizontal parameters stable.
        || (source_span_x >= 0.40 && source_span_y >= 0.001);
    let target_supported = (target_span_x >= 0.10 && target_span_y >= 0.035)
        || target_area >= 0.006
        || (target_span_x >= 0.40 && target_span_y >= 0.001);
    source_supported && target_supported
}

pub(super) fn focus_global_observations(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    coordinate_scale: f64,
    normalized_y_range: Option<(f64, f64)>,
) -> Vec<FocusGlobalObservation> {
    focus_global_observations_with_mode(images, matches, coordinate_scale, normalized_y_range, true)
}

pub(super) fn focus_global_observations_for_generic_region(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    coordinate_scale: f64,
    normalized_y_range: (f64, f64),
) -> Vec<FocusGlobalObservation> {
    focus_global_observations_with_mode(
        images,
        matches,
        coordinate_scale,
        Some(normalized_y_range),
        false,
    )
}

pub(super) fn focus_global_observations_with_mode(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    coordinate_scale: f64,
    normalized_y_range: Option<(f64, f64)>,
    use_foreground_region_matches: bool,
) -> Vec<FocusGlobalObservation> {
    let mut observations = Vec::new();
    // The observation vector defines the summation order of the robust cost and
    // of the normal equations in the global pose solve.  Iterating the match map
    // directly made that order depend on the per-process hash seed, which
    // changed the solved poses in the last mantissa bits and therefore the
    // rendered canvas.  Walk the pairs in ascending key order instead (需求 14.6).
    for (&(source_index, target_index), match_info) in sorted_pairs(matches) {
        if source_index >= images.len() || target_index >= images.len() {
            continue;
        }
        // An overview/reference frame is permitted to fill an explicitly
        // missing row, but it is not an identity-bearing layer for the focus
        // pose solve. Its wide, earlier capture often contains repeated seals
        // and rails that can pull the close-up sequence to a false scale.
        if images[source_index].overview_reference || images[target_index].overview_reference {
            continue;
        }
        if match_info.sequence_bridge || match_info.coarse_bridge {
            // A gap bridge supplies capture order and a conservative initial
            // placement only. A coarse lens-switch candidate has synthetic
            // grid points as well. Neither is an observed artwork
            // correspondence, so allowing either into the global solve would
            // turn a nominal seam into a fake reprojection constraint.
            continue;
        }
        // The match has already passed full-resolution model, spatial-support
        // and overlap validation. Keep long-range closures in the global solve
        // as well: vertical links and cross-column links are often far apart
        // in capture order, and filtering them by filename gap leaves a snake
        // scan anchored to one drifting spanning tree.
        let distant_edge = source_index.abs_diff(target_index) > 1;
        let selected_points = normalized_y_range
            .map(|range| {
                if use_foreground_region_matches {
                    focus_match_points_for_region(
                        &images[source_index],
                        &images[target_index],
                        match_info,
                        range,
                    )
                } else {
                    focus_match_points_for_generic_region(
                        match_info,
                        &images[source_index],
                        &images[target_index],
                        range,
                    )
                }
            })
            .unwrap_or_else(|| match_info.points.clone());
        let maximum_points = if distant_edge {
            FOCUS_GLOBAL_MAX_POINTS_PER_EDGE.min(64)
        } else {
            FOCUS_GLOBAL_MAX_POINTS_PER_EDGE
        };
        let observation_weight = if focus_match_is_local_bracket(
            &images[source_index],
            &images[target_index],
            match_info,
        ) {
            FOCUS_BRACKET_OBSERVATION_WEIGHT
        } else {
            1.0
        };
        let point_count = selected_points.len().min(maximum_points);
        for sample_index in 0..point_count {
            let point_index = if point_count == selected_points.len() {
                sample_index
            } else {
                sample_index * (selected_points.len() - 1) / (point_count - 1).max(1)
            };
            let (source, target) = selected_points[point_index];
            if source.x.is_finite()
                && source.y.is_finite()
                && target.x.is_finite()
                && target.y.is_finite()
            {
                let source =
                    focus_normalized_point(&images[source_index], source, coordinate_scale);
                let target =
                    focus_normalized_point(&images[target_index], target, coordinate_scale);
                observations.push(FocusGlobalObservation {
                    source_index,
                    target_index,
                    source,
                    target,
                    weight: observation_weight,
                });
            }
        }
    }
    observations
}

pub(super) fn focus_match_points_for_generic_region(
    match_info: &MatchInfo,
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    normalized_y_range: (f64, f64),
) -> Vec<(Point2<f64>, Point2<f64>)> {
    if match_info.coarse_bridge {
        return Vec::new();
    }
    // Ordinary keypoint inliers provide the reliable identity signal, while the
    // dense patch matches carry the sub-pixel residual that a broad homography
    // cannot explain. Keep both for generic regions; restricting this to
    // keypoints made the "generic" bands numerically different but visually
    // inert on texture-poor or near-field parts of a scan.
    let mut candidates = match_info.points.clone();
    candidates.extend(match_info.dense_focus_points.iter().copied());
    let maximum_residual = source_image.width.max(source_image.height).max(1) as f64
        * FOCUS_LOCAL_MATCH_RESIDUAL_RATIO;
    candidates
        .into_iter()
        .filter(|(source, target)| {
            if !source.x.is_finite()
                || !source.y.is_finite()
                || !target.x.is_finite()
                || !target.y.is_finite()
            {
                return false;
            }
            let mapped = match_info.homography * Point3::new(source.x, source.y, 1.0);
            if mapped.z.abs() < 1e-8 {
                return false;
            }
            let predicted = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
            if (predicted - *target).norm() > maximum_residual {
                return false;
            }
            let source_y = source.y / source_image.height.max(1) as f64;
            let target_y = target.y / target_image.height.max(1) as f64;
            // A moving mosaic can move the same physical strip to a different
            // normalized y coordinate in the adjacent frame. Associate a
            // correspondence with a band when either endpoint lies in it;
            // requiring both endpoints silently dropped exactly those residuals.
            (normalized_y_range.0..=normalized_y_range.1).contains(&source_y)
                || (normalized_y_range.0..=normalized_y_range.1).contains(&target_y)
        })
        .collect()
}
