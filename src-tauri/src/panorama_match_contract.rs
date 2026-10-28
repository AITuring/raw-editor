//! Deterministic matching-input and relation contracts.
//!
//! The full pair matcher remains in `panorama_stitching.rs`; this module owns
//! direction, spatial-support, scale and adjacent-focus predicates shared by
//! matching and station solving.

use nalgebra::{Matrix2, Matrix3, Point2};
use std::hash::{Hash, Hasher};

use super::{
    ImageInfo, MatchInfo, PANORAMA_MODEL_MIN_INLIERS, STATION_POSE_LOCAL_SCALE_SAMPLES_PER_AXIS,
    trailing_capture_number,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CanonicalRelationView {
    pub(super) source_index: usize,
    pub(super) target_index: usize,
    pub(super) inverted: bool,
    pub(super) homography: Matrix3<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CanonicalRelationViewError {
    IndexOutOfBounds {
        left: usize,
        right: usize,
        image_count: usize,
    },
    NonInvertible,
}

pub(super) fn canonical_relation_view(
    images: &[ImageInfo],
    left: usize,
    right: usize,
    relation: &MatchInfo,
) -> Result<CanonicalRelationView, CanonicalRelationViewError> {
    if left >= images.len() || right >= images.len() {
        return Err(CanonicalRelationViewError::IndexOutOfBounds {
            left,
            right,
            image_count: images.len(),
        });
    }
    let (source_index, target_index, inverted) = canonical_match_direction(images, left, right);
    let homography = match relation.canonical_homography {
        Some(canonical) => canonical,
        None if inverted => relation
            .homography
            .try_inverse()
            .ok_or(CanonicalRelationViewError::NonInvertible)?,
        None => relation.homography,
    };
    if homography.try_inverse().is_none() {
        return Err(CanonicalRelationViewError::NonInvertible);
    }
    Ok(CanonicalRelationView {
        source_index,
        target_index,
        inverted,
        homography,
    })
}

pub(super) fn canonical_match_direction(
    images: &[ImageInfo],
    first: usize,
    second: usize,
) -> (usize, usize, bool) {
    let first_has_features =
        !images[first].features.is_empty() || !images[first].top_features.is_empty();
    let second_has_features =
        !images[second].features.is_empty() || !images[second].top_features.is_empty();
    let content_order = first_has_features && second_has_features;
    let first_before_second = if content_order {
        focus_image_direction_key(&images[first]) <= focus_image_direction_key(&images[second])
    } else {
        images[first].filename <= images[second].filename
    };
    if first_before_second {
        (first, second, false)
    } else {
        (second, first, true)
    }
}

pub(super) fn focus_image_direction_key(image: &ImageInfo) -> (u64, usize, u32, u32) {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    image.width.hash(&mut hasher);
    image.height.hash(&mut hasher);
    image.features.len().hash(&mut hasher);
    image.top_features.len().hash(&mut hasher);
    for feature in image
        .features
        .iter()
        .chain(image.top_features.iter())
        .take(16)
    {
        feature.keypoint.x.hash(&mut hasher);
        feature.keypoint.y.hash(&mut hasher);
        feature.descriptor.hash(&mut hasher);
        feature.support_scale.to_bits().hash(&mut hasher);
    }
    (
        hasher.finish(),
        image.features.len().max(image.top_features.len()),
        image.width,
        image.height,
    )
}

pub(super) fn homography_linear_singular_values(transform: &Matrix3<f64>) -> Option<(f64, f64)> {
    let normaliser = transform[(2, 2)];
    let scale = if normaliser.is_finite() && normaliser.abs() > f64::EPSILON {
        1.0 / normaliser
    } else {
        1.0
    };
    let linear = Matrix2::new(
        transform[(0, 0)] * scale,
        transform[(0, 1)] * scale,
        transform[(1, 0)] * scale,
        transform[(1, 1)] * scale,
    );
    if linear.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let values = linear.singular_values();
    let (first, second) = (values[0], values[1]);
    (first.is_finite() && second.is_finite() && second >= 0.0)
        .then_some((first.max(second), first.min(second)))
}

pub(super) fn homography_scale_ratio(transform: &Matrix3<f64>) -> Option<f64> {
    let (first, second) = homography_linear_singular_values(transform)?;
    let ratio = (first * second).sqrt();
    (ratio.is_finite() && ratio > 0.0).then_some(ratio)
}

pub(super) fn station_pose_local_scale_range(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> Option<(f64, f64)> {
    if dimensions.0 == 0 || dimensions.1 == 0 || transform.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0f64;
    let sample_denominator = (STATION_POSE_LOCAL_SCALE_SAMPLES_PER_AXIS - 1) as f64;
    let maximum_x = f64::from(dimensions.0.saturating_sub(1));
    let maximum_y = f64::from(dimensions.1.saturating_sub(1));
    for sample_y in 0..STATION_POSE_LOCAL_SCALE_SAMPLES_PER_AXIS {
        let y = maximum_y * sample_y as f64 / sample_denominator;
        for sample_x in 0..STATION_POSE_LOCAL_SCALE_SAMPLES_PER_AXIS {
            let x = maximum_x * sample_x as f64 / sample_denominator;
            let denominator = transform[(2, 0)] * x + transform[(2, 1)] * y + transform[(2, 2)];
            if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
                return None;
            }
            let world_x_numerator =
                transform[(0, 0)] * x + transform[(0, 1)] * y + transform[(0, 2)];
            let world_y_numerator =
                transform[(1, 0)] * x + transform[(1, 1)] * y + transform[(1, 2)];
            let denominator_squared = denominator * denominator;
            let dx_dx = (transform[(0, 0)] * denominator - world_x_numerator * transform[(2, 0)])
                / denominator_squared;
            let dx_dy = (transform[(0, 1)] * denominator - world_x_numerator * transform[(2, 1)])
                / denominator_squared;
            let dy_dx = (transform[(1, 0)] * denominator - world_y_numerator * transform[(2, 0)])
                / denominator_squared;
            let dy_dy = (transform[(1, 1)] * denominator - world_y_numerator * transform[(2, 1)])
                / denominator_squared;
            let local_scale = (dx_dx * dy_dy - dx_dy * dy_dx).abs().sqrt();
            if !local_scale.is_finite() || local_scale <= 0.0 {
                return None;
            }
            minimum = minimum.min(local_scale);
            maximum = maximum.max(local_scale);
        }
    }
    (minimum.is_finite() && maximum.is_finite() && minimum > 0.0).then_some((minimum, maximum))
}

pub(super) fn match_has_spatial_support(
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) -> bool {
    if points.len() < PANORAMA_MODEL_MIN_INLIERS {
        return false;
    }
    let spread = |side: usize, dimensions: (u32, u32)| {
        let mut xs = points
            .iter()
            .map(
                |(source, target)| {
                    if side == 0 { source.x } else { target.x }
                },
            )
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let mut ys = points
            .iter()
            .map(
                |(source, target)| {
                    if side == 0 { source.y } else { target.y }
                },
            )
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if xs.len() < PANORAMA_MODEL_MIN_INLIERS || ys.len() < PANORAMA_MODEL_MIN_INLIERS {
            return (0.0, 0.0, 0.0);
        }
        xs.sort_unstable_by(f64::total_cmp);
        ys.sort_unstable_by(f64::total_cmp);
        let lower = ((xs.len() as f64 - 1.0) * 0.05).round() as usize;
        let upper = ((xs.len() as f64 - 1.0) * 0.95).round() as usize;
        let y_lower = ((ys.len() as f64 - 1.0) * 0.05).round() as usize;
        let y_upper = ((ys.len() as f64 - 1.0) * 0.95).round() as usize;
        let span_x = (xs[upper] - xs[lower]) / dimensions.0.max(1) as f64;
        let span_y = (ys[y_upper] - ys[y_lower]) / dimensions.1.max(1) as f64;
        (span_x, span_y, span_x.max(0.0) * span_y.max(0.0))
    };
    let (source_x, source_y, source_area) = spread(0, source_dimensions);
    let (target_x, target_y, target_area) = spread(1, target_dimensions);
    let source_long = source_x.max(source_y);
    let target_long = target_x.max(target_y);
    let source_short = source_x.min(source_y);
    let target_short = target_x.min(target_y);
    // A valid tile overlap normally covers a broad strip.  Permit a very thin
    // strip (for a scan along a long edge), but reject compact repeated-stroke
    // clusters that otherwise satisfy the minimum inlier count.
    source_long >= 0.12
        && target_long >= 0.12
        && source_short >= 0.006
        && target_short >= 0.006
        && source_area >= 0.0012
        && target_area >= 0.0012
}

pub(super) fn focus_capture_number_gap(source: &ImageInfo, target: &ImageInfo) -> Option<u64> {
    trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .map(|(left, right)| left.abs_diff(right))
}

pub(super) fn focus_compatible_focal_length(source: &ImageInfo, target: &ImageInfo) -> bool {
    match (source.focal_length_35mm, target.focal_length_35mm) {
        (Some(left), Some(right))
            if left.is_finite() && right.is_finite() && left > 0.0 && right > 0.0 =>
        {
            (left / right).max(right / left) <= 1.03
        }
        _ => true,
    }
}

pub(super) fn focus_adjacent_same_focal_pair(source: &ImageInfo, target: &ImageInfo) -> bool {
    focus_capture_number_gap(source, target).is_some_and(|gap| gap == 1)
        && focus_compatible_focal_length(source, target)
}
