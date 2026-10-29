//! Overlap quality and spatial support of a pair transform.
//!
//! Intensity and edge correlation on the covered overlap, photometric
//! measurements, shifted-structure verification and the spatial-support measures
//! used by the match gates.

use crate::panorama_utils::processing;
use crate::panorama_utils::registration;
use crate::panorama_utils::stitching;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;
use std::collections::HashSet;

use super::{
    FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT, ImageInfo, STATION_RELATION_EDGE_STRENGTH_RATIO_MAX,
    STATION_RELATION_EDGE_STRENGTH_RATIO_MIN,
    STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE, focus_compatible_focal_length,
    focus_stack_motion_is_shifted_mosaic, median_symmetric_error, median_value,
    panorama_transform_is_stable, transformed_point, virtual_tile_coverage_contains_rectangle,
    virtual_tile_local_jacobian,
};

pub(super) const FOCUS_MATCH_MIN_INTENSITY_NCC: f64 = 0.25;

pub(super) const FOCUS_MATCH_MIN_EDGE_NCC: f64 = 0.10;

pub(super) const FOCUS_MATCH_MIN_EDGE_ORIENTATION: f64 = 0.18;

pub(super) fn focus_overlap_is_verified(
    quality: (f64, f64, f64, usize),
    inlier_count: usize,
    median_error: f64,
    allow_low_texture_boundary: bool,
) -> bool {
    let (intensity_ncc, edge_ncc, edge_orientation, samples) = quality;
    let normal = intensity_ncc >= FOCUS_MATCH_MIN_INTENSITY_NCC
        && edge_ncc >= FOCUS_MATCH_MIN_EDGE_NCC
        && edge_orientation >= FOCUS_MATCH_MIN_EDGE_ORIENTATION;
    let low_texture_boundary = allow_low_texture_boundary
        && intensity_ncc >= 0.82
        && edge_ncc >= 0.30
        && edge_orientation >= 0.10
        && samples >= 1_000
        && inlier_count >= 12
        && median_error.is_finite()
        && median_error <= 2.5;
    normal || low_texture_boundary
}

pub(super) fn focus_source_structure_plane(info: &ImageInfo) -> ImageInfo {
    let (width, height, scale_factor) = processing::calculate_downscale_dimensions_capped(
        info.width,
        info.height,
        info.alignment_image
            .width()
            .max(info.alignment_image.height())
            .min(600),
    );
    ImageInfo {
        id: info.id,
        filename: info.filename.clone(),
        width: info.width,
        height: info.height,
        scale_factor,
        alignment_image: imageproc::filter::gaussian_blur_f32(
            &image::imageops::resize(
                &info.alignment_image,
                width,
                height,
                image::imageops::FilterType::Triangle,
            ),
            1.0,
        ),
        full_image: None,
        focal_length_35mm: info.focal_length_35mm,
        source_footprint: info.source_footprint,
        features: Vec::new(),
        top_features: Vec::new(),
        overview_reference: false,
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

pub(super) fn shifted_focus_structure_is_verified(
    source: &ImageInfo,
    target: &ImageInfo,
    homography: &Matrix3<f64>,
    points: &[(Point2<f64>, Point2<f64>)],
    log_match: bool,
) -> bool {
    if !focus_compatible_focal_length(source, target)
        || points.len() < 30
        || !focus_stack_motion_is_shifted_mosaic(points, source.dimensions())
        || !panorama_transform_is_stable(homography, source.dimensions())
        || panorama_spatial_support(points, source.dimensions(), target.dimensions())
            < FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT
        || median_symmetric_error(homography, points) > 3.0
    {
        return false;
    }
    let quality = focus_overlap_quality(
        &focus_source_structure_plane(source),
        &focus_source_structure_plane(target),
        homography,
    );
    let verified = quality.is_some_and(|(intensity, edge, orientation, samples)| {
        intensity >= 0.65 && edge >= 0.30 && orientation >= 0.35 && samples >= 600
    });
    if log_match {
        println!(
            "  - Shifted focus structural validation: {:?} ({} distributed inliers, {})",
            quality,
            points.len(),
            if verified { "verified" } else { "rejected" }
        );
    }
    verified
}

pub(super) fn station_relation_overlap_scale(
    transform: &Matrix3<f64>,
    points: &[(Point2<f64>, Point2<f64>)],
) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let centre = points.iter().fold(Point2::origin(), |sum, point| {
        Point2::new(sum.x + point.0.x, sum.y + point.0.y)
    }) / points.len() as f64;
    let denominator =
        transform[(2, 0)] * centre.x + transform[(2, 1)] * centre.y + transform[(2, 2)];
    let determinant = transform.determinant() / denominator.powi(3);
    let scale = determinant.abs().sqrt();
    (denominator.is_finite()
        && denominator.abs() > f64::EPSILON
        && scale.is_finite()
        && scale > 0.0)
        .then_some(scale)
}

/// Validate a metadata-scaled translation against the actual alignment images.
/// The low-resolution search above is intentionally cheap, but repeated
/// calligraphy can make a wrong translation look unique. Sampling the proposed
/// overlap at a denser grid and comparing both luminance and edge structure
/// catches that failure before synthetic points can be promoted to a match.
#[derive(Clone, Copy, Debug)]
pub(super) struct FocusOverlapQuality {
    pub(super) intensity_ncc: f64,
    pub(super) edge_ncc: f64,
    pub(super) mean_edge_orientation_cosine: f64,
    pub(super) samples: usize,
    pub(super) low_frequency_mean_relative_difference: f64,
    pub(super) edge_strength_ratio: f64,
    pub(super) median_edge_orientation_difference_degrees: f64,
    pub(super) exposure_gain: f64,
    pub(super) exposure_normalized_difference: f64,
    pub(super) exposure_normalized_mean_relative_difference: f64,
}

impl FocusOverlapQuality {
    pub(super) fn legacy_tuple(self) -> (f64, f64, f64, usize) {
        (
            self.intensity_ncc,
            self.edge_ncc,
            self.mean_edge_orientation_cosine,
            self.samples,
        )
    }
}

pub(super) fn focus_overlap_quality_measurements(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<FocusOverlapQuality> {
    focus_overlap_quality_measurements_on_coverage(source, target, source_to_target, None)
}

/// [`focus_overlap_quality_measurements`] on the pixels both analysis planes
/// cover. `coverages` holds the `(source, target)` Coverage_Masks in
/// `alignment_image` pixels; a sample counts only when every pixel it reads,
/// directly or through the plane's blur, is covered in both (需求 6.1, 6.8:
/// uncovered payload is not overlap). `None` samples every in-bounds position.
pub(super) fn focus_overlap_quality_measurements_on_coverage(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
    coverages: Option<(&stitching::CoverageMask, &stitching::CoverageMask)>,
) -> Option<FocusOverlapQuality> {
    let inverse = source_to_target.try_inverse()?;
    let target_width = target.alignment_image.width();
    let target_height = target.alignment_image.height();
    if target_width < 32 || target_height < 32 {
        return None;
    }
    if coverages.is_some_and(|(source_coverage, target_coverage)| {
        source_coverage.dimensions() != source.alignment_image.dimensions()
            || target_coverage.dimensions() != target.alignment_image.dimensions()
    }) {
        return None;
    }
    let sample_is_covered = |coverage: &stitching::CoverageMask, x: f64, y: f64| {
        let radius = VIRTUAL_TILE_QUALITY_COVERAGE_RADIUS_PX;
        virtual_tile_coverage_contains_rectangle(
            coverage,
            x - radius,
            y - radius,
            x + radius,
            y + radius,
        )
    };
    let columns = 64u32;
    let rows = 40u32;
    let mut source_values = Vec::new();
    let mut target_values = Vec::new();
    let mut source_edges = Vec::new();
    let mut target_edges = Vec::new();
    let mut edge_orientation_cosines = Vec::new();
    let mut edge_orientation_differences = Vec::new();
    for row in 2..rows.saturating_sub(2) {
        let target_y = (row as f64 + 0.5) * target_height as f64 / rows as f64;
        for column in 2..columns.saturating_sub(2) {
            let target_x = (column as f64 + 0.5) * target_width as f64 / columns as f64;
            let target_full = Point2::new(
                target_x * target.scale_factor,
                target_y * target.scale_factor,
            );
            let Some(source_full) = transformed_point(&inverse, target_full) else {
                continue;
            };
            let source_x = source_full.x / source.scale_factor.max(f64::EPSILON);
            let source_y = source_full.y / source.scale_factor.max(f64::EPSILON);
            if let Some((source_coverage, target_coverage)) = coverages
                && !(sample_is_covered(target_coverage, target_x, target_y)
                    && sample_is_covered(source_coverage, source_x, source_y))
            {
                continue;
            }
            let Some(target_value) =
                registration::sample_gray(&target.alignment_image, target_x, target_y)
            else {
                continue;
            };
            let Some(source_value) =
                registration::sample_gray(&source.alignment_image, source_x, source_y)
            else {
                continue;
            };
            let Some(target_dx_left) =
                registration::sample_gray(&target.alignment_image, target_x - 1.0, target_y)
            else {
                continue;
            };
            let Some(target_dx_right) =
                registration::sample_gray(&target.alignment_image, target_x + 1.0, target_y)
            else {
                continue;
            };
            let Some(target_dy_up) =
                registration::sample_gray(&target.alignment_image, target_x, target_y - 1.0)
            else {
                continue;
            };
            let Some(target_dy_down) =
                registration::sample_gray(&target.alignment_image, target_x, target_y + 1.0)
            else {
                continue;
            };
            let Some(source_dx_left) =
                registration::sample_gray(&source.alignment_image, source_x - 1.0, source_y)
            else {
                continue;
            };
            let Some(source_dx_right) =
                registration::sample_gray(&source.alignment_image, source_x + 1.0, source_y)
            else {
                continue;
            };
            let Some(source_dy_up) =
                registration::sample_gray(&source.alignment_image, source_x, source_y - 1.0)
            else {
                continue;
            };
            let Some(source_dy_down) =
                registration::sample_gray(&source.alignment_image, source_x, source_y + 1.0)
            else {
                continue;
            };
            // Gradients are covectors: compare them in the target pixel grid,
            // not in the two independently rotated/scaled source grids. The
            // patch matcher already applies this inverse-transpose Jacobian;
            // the dense acceptance check must use the same convention.
            let Some(gradient_transform) =
                virtual_tile_local_jacobian(source_to_target, source_full)
                    .and_then(|jacobian| jacobian.try_inverse())
                    .map(|inverse| inverse.transpose())
            else {
                continue;
            };
            let source_gradient = gradient_transform
                * nalgebra::Vector2::new(
                    source_dx_right - source_dx_left,
                    source_dy_down - source_dy_up,
                )
                * (target.scale_factor / source.scale_factor.max(f64::EPSILON));
            let source_dx = source_gradient.x;
            let source_dy = source_gradient.y;
            let target_dx = target_dx_right - target_dx_left;
            let target_dy = target_dy_down - target_dy_up;
            let source_edge = source_dx.hypot(source_dy);
            let target_edge = target_dx.hypot(target_dy);
            if source_edge.is_finite() && target_edge.is_finite() {
                source_edges.push(source_edge);
                target_edges.push(target_edge);
                if source_edge > 4.0 && target_edge > 4.0 {
                    let cosine = ((source_dx * target_dx + source_dy * target_dy)
                        / (source_edge * target_edge))
                        .clamp(-1.0, 1.0);
                    edge_orientation_cosines.push(cosine);
                    // Edge orientation is axial rather than directional: a
                    // 180-degree gradient reversal describes the same edge.
                    edge_orientation_differences.push(cosine.abs().acos().to_degrees());
                }
            }
            source_values.push(source_value);
            target_values.push(target_value);
        }
    }
    if source_values.len() < 120 || source_edges.len() != source_values.len() {
        return None;
    }
    let intensity_ncc = normalized_correlation(&source_values, &target_values)?;
    let edge_ncc = normalized_correlation(&source_edges, &target_edges)?;
    let mean_edge_orientation_cosine = if edge_orientation_cosines.is_empty() {
        0.0
    } else {
        edge_orientation_cosines.iter().sum::<f64>() / edge_orientation_cosines.len() as f64
    };
    let source_mean = source_values.iter().sum::<f64>() / source_values.len() as f64;
    let target_mean = target_values.iter().sum::<f64>() / target_values.len() as f64;
    // A mean is the zero-frequency component of the sampled overlap. Dividing
    // by the larger magnitude makes the comparison symmetric and bounded.
    let low_frequency_mean_relative_difference = (source_mean - target_mean).abs()
        / source_mean.abs().max(target_mean.abs()).max(f64::EPSILON);
    // Estimate a scalar exposure gain on alternating overlap samples and
    // validate it on the held-out samples. Unlike matching the two means,
    // this residual cannot be made zero for arbitrary different images.
    let mut exposure_ratios = source_values
        .iter()
        .zip(&target_values)
        .enumerate()
        .filter_map(|(index, (&source, &target))| {
            (index % 2 == 0 && source > 8.0 && target > 8.0).then_some(target / source)
        })
        .collect::<Vec<_>>();
    let exposure_gain = median_value(&mut exposure_ratios).unwrap_or(f64::NAN);
    let mut exposure_residual = 0.0;
    let mut exposure_magnitude = 0.0;
    let mut predicted_sum = 0.0;
    let mut held_out_target_sum = 0.0;
    for (index, (&source, &target)) in source_values.iter().zip(&target_values).enumerate() {
        if index % 2 == 1 {
            let predicted = source * exposure_gain;
            exposure_residual += (predicted - target).abs();
            exposure_magnitude += predicted.abs().max(target.abs());
            predicted_sum += predicted;
            held_out_target_sum += target;
        }
    }
    let exposure_normalized_difference = exposure_residual / exposure_magnitude.max(f64::EPSILON);
    let exposure_normalized_mean_relative_difference = (predicted_sum - held_out_target_sum).abs()
        / predicted_sum
            .abs()
            .max(held_out_target_sum.abs())
            .max(f64::EPSILON);
    let source_edge_mean = source_edges.iter().sum::<f64>() / source_edges.len() as f64;
    let target_edge_mean = target_edges.iter().sum::<f64>() / target_edges.len() as f64;
    let edge_strength_ratio = if target_edge_mean > f64::EPSILON {
        source_edge_mean / target_edge_mean
    } else if source_edge_mean <= f64::EPSILON {
        1.0
    } else {
        f64::INFINITY
    };
    let median_edge_orientation_difference_degrees =
        median_value(&mut edge_orientation_differences).unwrap_or(f64::INFINITY);
    Some(FocusOverlapQuality {
        intensity_ncc,
        edge_ncc,
        mean_edge_orientation_cosine,
        samples: source_values.len(),
        low_frequency_mean_relative_difference,
        edge_strength_ratio,
        median_edge_orientation_difference_degrees,
        exposure_gain,
        exposure_normalized_difference,
        exposure_normalized_mean_relative_difference,
    })
}

/// Geometry is checked before the tone harmonizer runs. A confirmed structural
/// overlap must therefore survive a moderate camera-exposure change. Keep the
/// ordinary raw photometric gate unless correlation, held-out compensated
/// pixels and compensated edge strength independently verify that change.
pub(super) fn virtual_tile_photometric_measurements(quality: FocusOverlapQuality) -> (f64, f64) {
    let compensated_strength = quality.edge_strength_ratio * quality.exposure_gain;
    let exposure_structure_verified = (quality.intensity_ncc >= 0.95
        && quality.edge_ncc >= 0.65
        && quality.exposure_normalized_difference <= 0.20)
        || (quality.intensity_ncc >= 0.85
            && quality.edge_ncc >= 0.45
            && quality.exposure_normalized_difference <= 0.10);
    if quality.low_frequency_mean_relative_difference
        > STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE
        && exposure_structure_verified
        && (0.5..=2.0).contains(&quality.exposure_gain)
        && quality.exposure_normalized_mean_relative_difference
            <= STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE
        && (STATION_RELATION_EDGE_STRENGTH_RATIO_MIN..=STATION_RELATION_EDGE_STRENGTH_RATIO_MAX)
            .contains(&compensated_strength)
    {
        (
            quality.exposure_normalized_mean_relative_difference,
            compensated_strength,
        )
    } else {
        (
            quality.low_frequency_mean_relative_difference,
            quality.edge_strength_ratio,
        )
    }
}

/// Legacy overlap score used by non-Station_Relation paths. Station relation
/// acceptance reads the complete measurement structure above.
pub(super) fn focus_overlap_quality(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<(f64, f64, f64, usize)> {
    focus_overlap_quality_measurements(source, target, source_to_target)
        .map(FocusOverlapQuality::legacy_tuple)
}

pub(super) fn normalized_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (&left_value, &right_value) in left.iter().zip(right) {
        let left_delta = left_value - left_mean;
        let right_delta = right_value - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    (left_variance > f64::EPSILON && right_variance > f64::EPSILON)
        .then(|| covariance / (left_variance * right_variance).sqrt())
        .filter(|value| value.is_finite())
}

pub(super) fn panorama_spatial_support(
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) -> f64 {
    if points.len() < 4 {
        return 0.0;
    }
    let side_support = |side: usize, dimensions: (u32, u32)| {
        let width = dimensions.0.max(1) as f64;
        let height = dimensions.1.max(1) as f64;
        let mut xs = Vec::with_capacity(points.len());
        let mut ys = Vec::with_capacity(points.len());
        let mut occupied = HashSet::new();
        for &(source, target) in points {
            let point = if side == 0 { source } else { target };
            if !point.x.is_finite() || !point.y.is_finite() {
                continue;
            }
            xs.push(point.x);
            ys.push(point.y);
            let cell_x = ((point.x / width).clamp(0.0, 0.999_999) * 4.0).floor() as u8;
            let cell_y = ((point.y / height).clamp(0.0, 0.999_999) * 4.0).floor() as u8;
            occupied.insert((cell_x, cell_y));
        }
        if xs.len() < 4 {
            return 0.0;
        }
        xs.sort_unstable_by(f64::total_cmp);
        ys.sort_unstable_by(f64::total_cmp);
        let lower = ((xs.len() - 1) as f64 * 0.05).round() as usize;
        let upper = ((xs.len() - 1) as f64 * 0.95).round() as usize;
        let y_lower = ((ys.len() - 1) as f64 * 0.05).round() as usize;
        let y_upper = ((ys.len() - 1) as f64 * 0.95).round() as usize;
        let area = ((xs[upper] - xs[lower]) / width).max(0.0)
            * ((ys[y_upper] - ys[y_lower]) / height).max(0.0);
        let occupancy = occupied.len() as f64 / 16.0;
        (area.sqrt() * 0.7 + occupancy * 0.3).clamp(0.0, 1.0)
    };
    side_support(0, source_dimensions)
        .min(side_support(1, target_dimensions))
        .clamp(0.0, 1.0)
}

pub(super) fn panorama_transform_overlap_support(
    transform: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) -> f64 {
    let Some(inverse) = transform.try_inverse() else {
        return 0.0;
    };
    let inside = |mapped: Point3<f64>, dimensions: (u32, u32)| {
        mapped.z.abs() >= 1e-8 && {
            let x = mapped.x / mapped.z;
            let y = mapped.y / mapped.z;
            x >= 0.0 && y >= 0.0 && x < dimensions.0 as f64 && y < dimensions.1 as f64
        }
    };
    let mut forward = 0usize;
    let mut reverse = 0usize;
    for row in 0..=4 {
        for column in 0..=4 {
            let source = Point3::new(
                source_dimensions.0 as f64 * column as f64 / 4.0,
                source_dimensions.1 as f64 * row as f64 / 4.0,
                1.0,
            );
            if inside(*transform * source, target_dimensions) {
                forward += 1;
            }
            let target = Point3::new(
                target_dimensions.0 as f64 * column as f64 / 4.0,
                target_dimensions.1 as f64 * row as f64 / 4.0,
                1.0,
            );
            if inside(inverse * target, source_dimensions) {
                reverse += 1;
            }
        }
    }
    (forward.min(reverse) as f64 / 25.0).clamp(0.0, 1.0)
}

/// Half-width, in relation-quality plane pixels, of the neighbourhood that must
/// be covered around a requirement-6.8 sample: the ±1px gradient taps with
/// bilinear access read 2 pixels around it and the σ = 1 blur (imageproc
/// kernel radius ⌈2σ⌉) 2 more. On langyuan-10 the uncovered payload inside
/// that reach alone moved the verified 1→2 relation's edge-strength ratio
/// from 1.01 to 1.40 and its low-frequency difference from 0.002 to 0.13.
pub(super) const VIRTUAL_TILE_QUALITY_COVERAGE_RADIUS_PX: f64 = 4.0;
