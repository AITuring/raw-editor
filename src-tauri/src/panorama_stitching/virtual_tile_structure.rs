//! Structural scores and coverage geometry of Virtual_Tile overlaps.
//!
//! Coverage tests, local Jacobians, gradient and patch structure scores and the
//! predicted overlap grid shared by the station matchers.

#[cfg(test)]
use crate::panorama_utils::processing;
use crate::panorama_utils::registration;
use crate::panorama_utils::stitching;
use image::GrayImage;
#[cfg(test)]
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;

#[cfg(test)]
use super::Feature;
#[cfg(test)]
use super::virtual_tile_covered_luminance;
use super::{
    KeyPoint, VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE, clip_polygon_to_rectangle, median_value,
    normalized_correlation, transformed_point,
};

pub(super) const VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS: u32 = 20;

/// Convert a rendered Virtual_Tile to its native-pixel station-matching
/// representation.
///
/// Uncovered RGB values are never read. The deterministic zero outside the
/// mask prevents transparent-corner payloads from changing detection, while
/// the final erosion keeps both FAST and BRIEF support wholly inside genuine
/// Coverage_Mask pixels (需求 6.1). Production matches the area-averaged planes
/// of [`install_virtual_tile_match_planes`]; this native form backs the tests.
#[cfg(test)]
pub(super) fn virtual_tile_covered_features(
    pixels: &Rgb32FImage,
    coverage: &stitching::CoverageMask,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> (GrayImage, Vec<Feature>) {
    let (width, height) = pixels.dimensions();
    if coverage.dimensions() != (width, height) {
        return (GrayImage::new(width, height), Vec::new());
    }
    let gray = virtual_tile_covered_luminance(pixels, coverage);
    let radius = VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS;
    let features = processing::find_features(&gray, brief_pairs)
        .into_iter()
        .filter(|feature| {
            let x = feature.keypoint.x;
            let y = feature.keypoint.y;
            x >= radius
                && y >= radius
                && x.saturating_add(radius) < width
                && y.saturating_add(radius) < height
                && (y - radius..=y + radius).all(|sample_y| {
                    (x - radius..=x + radius)
                        .all(|sample_x| coverage.is_covered(sample_x, sample_y))
                })
        })
        .collect();
    (gray, features)
}

pub(super) const VIRTUAL_TILE_GUIDED_PATCH_RADIUS: i32 = 6;

pub(super) const VIRTUAL_TILE_GUIDED_GRID_COLUMNS: usize = 6;

pub(super) const VIRTUAL_TILE_GUIDED_GRID_ROWS: usize = 6;

#[derive(Clone, Copy, Debug)]
pub(super) struct VirtualTileStructuralScore {
    pub(super) patch_ncc: f64,
    pub(super) gradient_ncc: f64,
    pub(super) orientation_difference_degrees: f64,
}

impl VirtualTileStructuralScore {
    pub(super) fn rank(self, descriptor_distance: u32) -> f64 {
        self.patch_ncc + 0.5 * self.gradient_ncc
            - f64::from(descriptor_distance)
                / f64::from(VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE.max(1))
                * 0.05
    }
}

pub(super) fn virtual_tile_point_is_covered(
    coverage: &stitching::CoverageMask,
    point: Point2<f64>,
) -> bool {
    let (width, height) = coverage.dimensions();
    point.x.is_finite()
        && point.y.is_finite()
        && point.x >= 0.0
        && point.y >= 0.0
        && point.x < width as f64
        && point.y < height as f64
        && coverage.is_covered(point.x.round() as u32, point.y.round() as u32)
}

pub(super) fn virtual_tile_local_jacobian(
    transform: &Matrix3<f64>,
    point: Point2<f64>,
) -> Option<nalgebra::Matrix2<f64>> {
    let center = transformed_point(transform, point)?;
    let horizontal = transformed_point(transform, point + nalgebra::Vector2::new(1.0, 0.0))?;
    let vertical = transformed_point(transform, point + nalgebra::Vector2::new(0.0, 1.0))?;
    let jacobian = nalgebra::Matrix2::new(
        horizontal.x - center.x,
        vertical.x - center.x,
        horizontal.y - center.y,
        vertical.y - center.y,
    );
    (jacobian.iter().all(|value| value.is_finite()) && jacobian.determinant().abs() > 1e-8)
        .then_some(jacobian)
}

pub(super) fn virtual_tile_gradient(
    image: &GrayImage,
    point: Point2<f64>,
) -> Option<nalgebra::Vector2<f64>> {
    let gx = registration::sample_gray(image, point.x + 1.0, point.y)?
        - registration::sample_gray(image, point.x - 1.0, point.y)?;
    let gy = registration::sample_gray(image, point.x, point.y + 1.0)?
        - registration::sample_gray(image, point.x, point.y - 1.0)?;
    let gradient = nalgebra::Vector2::new(gx, gy);
    gradient
        .iter()
        .all(|value| value.is_finite())
        .then_some(gradient)
}

/// A projective repair's scale belongs at the station, not at the unrelated
/// world origin. Reading its normalized linear block made the bound depend on
/// canvas translation and rejected a near-unit correction as a 1.5x jump.
pub(super) fn virtual_tile_repair_local_scale_ratio(
    transform: &Matrix3<f64>,
    center: Point2<f64>,
) -> Option<f64> {
    let ratio = virtual_tile_local_jacobian(transform, center)?
        .determinant()
        .abs()
        .sqrt();
    (ratio.is_finite() && ratio > 0.0).then_some(ratio)
}

pub(super) fn axial_orientation_difference_degrees(
    left: nalgebra::Vector2<f64>,
    right: nalgebra::Vector2<f64>,
) -> Option<f64> {
    let denominator = left.norm() * right.norm();
    if !denominator.is_finite() || denominator <= 1e-8 {
        return None;
    }
    let cosine = (left.dot(&right) / denominator).abs().clamp(0.0, 1.0);
    Some(cosine.acos().to_degrees())
}

pub(super) fn virtual_tile_gradient_magnitude_ncc(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    if let Some(correlation) = normalized_correlation(left, right) {
        return Some(correlation);
    }
    // A constant-strength edge has zero variance but still carries orientation
    // and location evidence. Treat equal non-zero magnitudes as perfect NCC;
    // unequal or flat gradients remain non-matches.
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let scale = left_mean.abs().max(right_mean.abs());
    (scale > 1e-8).then(|| (1.0 - (left_mean - right_mean).abs() / scale).clamp(-1.0, 1.0))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_structural_score_points(
    left: &GrayImage,
    right: &GrayImage,
    left_coverage: &stitching::CoverageMask,
    right_coverage: &stitching::CoverageMask,
    center: Point2<f64>,
    right_center: Point2<f64>,
    left_to_right: &Matrix3<f64>,
) -> Option<VirtualTileStructuralScore> {
    let projected_center = transformed_point(left_to_right, center)?;
    let residual = right_center - projected_center;
    let jacobian = virtual_tile_local_jacobian(left_to_right, center)?;
    let gradient_transform = jacobian.try_inverse()?.transpose();
    let radius = VIRTUAL_TILE_GUIDED_PATCH_RADIUS;
    let capacity = ((radius * 2 + 1) * (radius * 2 + 1)) as usize;
    let mut left_values = Vec::with_capacity(capacity);
    let mut right_values = Vec::with_capacity(capacity);
    let mut left_gradient_magnitudes = Vec::with_capacity(capacity);
    let mut right_gradient_magnitudes = Vec::with_capacity(capacity);
    let mut orientation_differences = Vec::with_capacity(capacity);

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let source = center + nalgebra::Vector2::new(f64::from(dx), f64::from(dy));
            let target = transformed_point(left_to_right, source)? + residual;
            if !virtual_tile_point_is_covered(left_coverage, source)
                || !virtual_tile_point_is_covered(right_coverage, target)
            {
                return None;
            }
            left_values.push(registration::sample_gray(left, source.x, source.y)?);
            right_values.push(registration::sample_gray(right, target.x, target.y)?);

            let left_gradient = virtual_tile_gradient(left, source)?;
            let right_gradient = virtual_tile_gradient(right, target)?;
            let transformed_gradient = gradient_transform * left_gradient;
            left_gradient_magnitudes.push(transformed_gradient.norm());
            right_gradient_magnitudes.push(right_gradient.norm());
            if transformed_gradient.norm().min(right_gradient.norm()) >= 4.0
                && let Some(difference) =
                    axial_orientation_difference_degrees(transformed_gradient, right_gradient)
            {
                orientation_differences.push(difference);
            }
        }
    }

    let patch_ncc = normalized_correlation(&left_values, &right_values)?;
    let gradient_ncc =
        virtual_tile_gradient_magnitude_ncc(&left_gradient_magnitudes, &right_gradient_magnitudes)?;
    let orientation_difference_degrees = median_value(&mut orientation_differences)?;
    Some(VirtualTileStructuralScore {
        patch_ncc,
        gradient_ncc,
        orientation_difference_degrees,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_structural_score(
    left: &GrayImage,
    right: &GrayImage,
    left_coverage: &stitching::CoverageMask,
    right_coverage: &stitching::CoverageMask,
    left_point: KeyPoint,
    right_point: KeyPoint,
    left_to_right: &Matrix3<f64>,
) -> Option<VirtualTileStructuralScore> {
    virtual_tile_structural_score_points(
        left,
        right,
        left_coverage,
        right_coverage,
        Point2::new(f64::from(left_point.x), f64::from(left_point.y)),
        Point2::new(f64::from(right_point.x), f64::from(right_point.y)),
        left_to_right,
    )
}

pub(super) fn virtual_tile_predicted_overlap_bounds(
    left_to_right: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
) -> Option<(f64, f64, f64, f64)> {
    let right_to_left = left_to_right.try_inverse()?;
    let footprint = [
        Point2::new(0.0, 0.0),
        Point2::new(f64::from(right_dimensions.0), 0.0),
        Point2::new(f64::from(right_dimensions.0), f64::from(right_dimensions.1)),
        Point2::new(0.0, f64::from(right_dimensions.1)),
    ]
    .into_iter()
    .map(|point| transformed_point(&right_to_left, point))
    .collect::<Option<Vec<_>>>()?;
    let clipped = clip_polygon_to_rectangle(&footprint, left_dimensions);
    if clipped.len() < 3 {
        return None;
    }
    let min_x = clipped
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let max_x = clipped
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = clipped
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = clipped
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    (min_x.is_finite()
        && max_x.is_finite()
        && min_y.is_finite()
        && max_y.is_finite()
        && max_x > min_x
        && max_y > min_y)
        .then_some((min_x, min_y, max_x, max_y))
}

pub(super) fn virtual_tile_overlap_grid_cell(
    point: Point2<f64>,
    bounds: (f64, f64, f64, f64),
) -> (usize, usize) {
    let (min_x, min_y, max_x, max_y) = bounds;
    let x = (((point.x - min_x) / (max_x - min_x)) * VIRTUAL_TILE_GUIDED_GRID_COLUMNS as f64)
        .floor()
        .clamp(0.0, (VIRTUAL_TILE_GUIDED_GRID_COLUMNS - 1) as f64) as usize;
    let y = (((point.y - min_y) / (max_y - min_y)) * VIRTUAL_TILE_GUIDED_GRID_ROWS as f64)
        .floor()
        .clamp(0.0, (VIRTUAL_TILE_GUIDED_GRID_ROWS - 1) as f64) as usize;
    (x, y)
}
