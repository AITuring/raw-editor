//! Pure Quality_Gate measurements over already-decoded pixels and geometry.
//!
//! This module does not decode sources, read settings, mutate ownership, write
//! reports, or decide whether an export proceeds. Callers supply display-sRGB
//! float pixels, native/world geometry, coverage and immutable owner planes.
//! All error identifiers reuse the existing report/degradation vocabulary.
#![allow(dead_code)] // Production wiring belongs to tasks 15.7/15.8.

pub(crate) const QUALITY_LOCAL_SCALE_MEDIAN_MIN: f64 = 0.98;
pub(crate) const QUALITY_LOCAL_SCALE_PIXEL_MIN: f64 = 0.95;
pub(crate) const QUALITY_LOCAL_SCALE_PIXEL_RATIO_MIN: f64 = 0.99;
pub(crate) const QUALITY_EFFECTIVE_PIXEL_RATIO_MIN: f64 = 0.98;
pub(crate) const QUALITY_MTF50_RATIO_MIN: f64 = 0.93;
pub(crate) const QUALITY_GRADIENT_RATIO_MIN: f64 = 0.95;
pub(crate) const QUALITY_NOISE_RATIO_MIN: f64 = 0.85;
pub(crate) const QUALITY_NOISE_RATIO_MAX: f64 = 1.15;
pub(crate) const QUALITY_ROI_DELTA_E00_MAX: f64 = 2.0;
pub(crate) const QUALITY_BOUNDARY_P95_MAX: f64 = 1.5;
pub(crate) const QUALITY_BOUNDARY_MAX: f64 = 3.0;
pub(crate) const QUALITY_LOW_CONFIDENCE_RATIO_MAX: f64 = 0.01;
pub(crate) const OWNER_SHARPNESS_SHORTFALL_MAX: f32 = 0.05;
pub(crate) const OWNER_SHARPNESS_COVERAGE_MIN: f64 = 0.99;
pub(crate) const OWNERSHIP_DISAGREEMENT_VETO: f32 = 0.20;
pub(crate) const QUALITY_MIN_MEASURABLE: usize = 8;
pub(crate) const QUALITY_MAX_UNMEASURABLE_RATIO: f64 = 0.20;
pub(crate) const QUALITY_CRITERIA: [&str; 9] = [
    "local_scale_median",
    "local_scale_pixel_ratio",
    "effective_pixel_count",
    "mtf50_normalized",
    "gradient_energy_normalized",
    "noise_sigma_ratio",
    "roi_delta_e00",
    "boundary_stroke_alignment",
    "owner_sharpness_coverage",
];

/// Diagnostic measurements remain visible but cannot establish acceptance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RatioMeasurement {
    pub value: f64,
    pub passed: bool,
    pub diagnostic: bool,
}

/// Ratios require a strictly positive, measurable reference. Zero/zero is
/// not evidence of a passing MTF, gradient or noise comparison.
pub(crate) fn compare_ratio(
    measured: f64,
    reference: f64,
    minimum: f64,
    maximum: f64,
    diagnostic: bool,
) -> Option<RatioMeasurement> {
    if !measured.is_finite() || measured < 0.0 || !reference.is_finite() || reference <= 0.0 {
        return None;
    }
    let value = measured / reference;
    value.is_finite().then_some(RatioMeasurement {
        value,
        passed: !diagnostic && value >= minimum && value <= maximum,
        diagnostic,
    })
}

// ==================== task 15.1: deterministic ROI geometry ====================
// This fragment assumes sibling imports `degradation` and `residual_warp`.
use image::{GrayImage, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Vector3};
use std::collections::{BTreeSet, VecDeque};

use super::{degradation, residual_warp};

pub(crate) const QUALITY_ROI_SIDE: u32 = 512;
pub(crate) const QUALITY_ROI_MARGIN: f64 = 16.0;
pub(crate) const QUALITY_ROI_MAX_COUNT: usize = 256;
pub(crate) const QUALITY_ROI_MIN_TARGET: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OwnerRegion {
    pub label: u32,
    pub owner: u16,
    pub pixel_count: u64,
    pub first_pixel: (u32, u32),
}
#[derive(Debug, Clone)]
pub(crate) struct OwnerRegions {
    pub width: u32,
    pub height: u32,
    pub labels: Vec<u32>,
    pub regions: Vec<OwnerRegion>,
    pub distance_to_boundary: Vec<f64>,
}
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QualityRoi {
    pub x: u32,
    pub y: u32,
    pub side: u32,
    pub region_label: u32,
    pub owner: u16,
    pub world_origin: (f64, f64),
    pub boundary_clearance: f64,
}
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RoiSelection {
    pub rois: Vec<QualityRoi>,
    pub enumeration_step: u32,
    pub accepted_candidates: usize,
    pub mandatory_regions: Vec<u32>,
}

pub(crate) fn label_owner_regions(
    owners: &[u16],
    coverage: &GrayImage,
) -> Result<OwnerRegions, &'static str> {
    let (width, height) = coverage.dimensions();
    let count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(degradation::DIAGNOSTICS_ROI_INVALID)?;
    if owners.len() != count || count == 0 {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    if coverage
        .as_raw()
        .iter()
        .zip(owners)
        .any(|(&mask, &owner)| mask != 0 && owner == 0)
    {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    let mut labels = vec![0u32; count];
    let mut regions = Vec::new();
    for seed in 0..count {
        if labels[seed] != 0 || coverage.as_raw()[seed] == 0 {
            continue;
        }
        let label = u32::try_from(regions.len() + 1)
            .map_err(|_| degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE)?;
        let owner = owners[seed];
        let mut queue = VecDeque::from([seed]);
        labels[seed] = label;
        let mut pixel_count = 0;
        while let Some(index) = queue.pop_front() {
            pixel_count += 1;
            let x = index % width as usize;
            let y = index / width as usize;
            for next in [
                (x > 0).then(|| index - 1),
                (x + 1 < width as usize).then(|| index + 1),
                (y > 0).then(|| index - width as usize),
                (y + 1 < height as usize).then(|| index + width as usize),
            ]
            .into_iter()
            .flatten()
            {
                if labels[next] == 0 && coverage.as_raw()[next] != 0 && owners[next] == owner {
                    labels[next] = label;
                    queue.push_back(next);
                }
            }
        }
        regions.push(OwnerRegion {
            label,
            owner,
            pixel_count,
            first_pixel: (
                (seed % width as usize) as u32,
                (seed / width as usize) as u32,
            ),
        });
    }
    // Exact Euclidean distance transform from all 4-neighbour boundary pixels.
    // A border pixel has distance zero, intentionally conservative by one pixel.
    let far = (f64::from(width).hypot(f64::from(height)) + 1.0).powi(2);
    let mut squared = vec![far; count];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let i = y * width as usize + x;
            let label = labels[i];
            let boundary = label == 0
                || (x > 0 && labels[i - 1] != label)
                || (x + 1 < width as usize && labels[i + 1] != label)
                || (y > 0 && labels[i - width as usize] != label)
                || (y + 1 < height as usize && labels[i + width as usize] != label);
            if boundary {
                squared[i] = 0.0;
            }
        }
    }
    for row in squared.chunks_exact_mut(width as usize) {
        let transformed = squared_distance_1d(row);
        row.copy_from_slice(&transformed);
    }
    for x in 0..width as usize {
        let column = (0..height as usize)
            .map(|y| squared[y * width as usize + x])
            .collect::<Vec<_>>();
        let transformed = squared_distance_1d(&column);
        for (y, value) in transformed.into_iter().enumerate() {
            squared[y * width as usize + x] = value;
        }
    }
    Ok(OwnerRegions {
        width,
        height,
        labels,
        regions,
        distance_to_boundary: squared.into_iter().map(f64::sqrt).collect(),
    })
}

fn squared_distance_1d(input: &[f64]) -> Vec<f64> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut vertices = vec![0usize; input.len()];
    let mut intersections = vec![0.0; input.len() + 1];
    intersections[0] = f64::NEG_INFINITY;
    intersections[1] = f64::INFINITY;
    let mut k = 0;
    for q in 1..input.len() {
        let mut crossing;
        loop {
            let v = vertices[k];
            crossing = ((input[q] + (q as f64).powi(2)) - (input[v] + (v as f64).powi(2)))
                / (2.0 * (q as f64 - v as f64));
            if crossing > intersections[k] || k == 0 {
                break;
            }
            k -= 1;
        }
        k += 1;
        vertices[k] = q;
        intersections[k] = crossing;
        intersections[k + 1] = f64::INFINITY;
    }
    let mut output = vec![0.0; input.len()];
    k = 0;
    for (q, value) in output.iter_mut().enumerate() {
        while intersections[k + 1] < q as f64 {
            k += 1;
        }
        *value = (q as f64 - vertices[k] as f64).powi(2) + input[vertices[k]];
    }
    output
}

pub(crate) fn select_quality_rois(
    regions: &OwnerRegions,
    origin: (f64, f64),
) -> Result<RoiSelection, &'static str> {
    if !origin.0.is_finite() || !origin.1.is_finite() {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    let mut step: u32 = 256;
    let mut candidates = enumerate_quality_rois(regions, origin, step as usize);
    if candidates.len() < QUALITY_ROI_MIN_TARGET {
        step = 128;
        candidates = enumerate_quality_rois(regions, origin, step as usize);
    }
    if candidates.is_empty() {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    let mandatory_regions = regions
        .regions
        .iter()
        .filter(|r| r.pixel_count >= 4 * u64::from(QUALITY_ROI_SIDE).pow(2))
        .map(|r| r.label)
        .collect::<Vec<_>>();
    if mandatory_regions.len() > QUALITY_ROI_MAX_COUNT {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    let mut mandatory = BTreeSet::new();
    for label in &mandatory_regions {
        let best = candidates
            .iter()
            .enumerate()
            .filter(|(_, roi)| roi.region_label == *label)
            .max_by(|(li, left), (ri, right)| {
                left.boundary_clearance
                    .total_cmp(&right.boundary_clearance)
                    .then_with(|| ri.cmp(li))
            })
            .map(|(index, _)| index)
            .ok_or(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE)?;
        mandatory.insert(best);
    }
    let stride = candidates.len().div_ceil(QUALITY_ROI_MAX_COUNT).max(1);
    let mut selected = mandatory;
    for index in (0..candidates.len()).step_by(stride) {
        if selected.len() >= QUALITY_ROI_MAX_COUNT {
            break;
        }
        selected.insert(index);
    }
    Ok(RoiSelection {
        accepted_candidates: candidates.len(),
        enumeration_step: step,
        mandatory_regions,
        rois: selected
            .into_iter()
            .map(|i| candidates[i].clone())
            .collect(),
    })
}

fn enumerate_quality_rois(
    regions: &OwnerRegions,
    origin: (f64, f64),
    step: usize,
) -> Vec<QualityRoi> {
    if regions.width < QUALITY_ROI_SIDE || regions.height < QUALITY_ROI_SIDE {
        return Vec::new();
    }
    let mut result = Vec::new();
    for y in (0..=regions.height - QUALITY_ROI_SIDE).step_by(step) {
        for x in (0..=regions.width - QUALITY_ROI_SIDE).step_by(step) {
            let label = regions.labels[(y * regions.width + x) as usize];
            if label == 0 {
                continue;
            }
            let mut clearance = f64::INFINITY;
            let mut accepted = true;
            'rows: for ry in y..y + QUALITY_ROI_SIDE {
                let start = (ry * regions.width + x) as usize;
                for i in start..start + QUALITY_ROI_SIDE as usize {
                    if regions.labels[i] != label
                        || regions.distance_to_boundary[i] < QUALITY_ROI_MARGIN
                    {
                        accepted = false;
                        break 'rows;
                    }
                    clearance = clearance.min(regions.distance_to_boundary[i]);
                }
            }
            if accepted {
                result.push(QualityRoi {
                    x,
                    y,
                    side: QUALITY_ROI_SIDE,
                    region_label: label,
                    owner: regions.regions[(label - 1) as usize].owner,
                    world_origin: (origin.0 + f64::from(x), origin.1 + f64::from(y)),
                    boundary_clearance: clearance,
                });
            }
        }
    }
    result
}

#[derive(Debug, Clone)]
pub(crate) struct SourceGeometry {
    pub member_to_anchor: Matrix3<f64>,
    pub tile_to_world: Matrix3<f64>,
    pub station_id: usize,
}

fn project_geometry(
    matrix: &Matrix3<f64>,
    point: Point2<f64>,
) -> Result<Point2<f64>, &'static str> {
    let p = matrix * Vector3::new(point.x, point.y, 1.0);
    if !p.iter().all(|v| v.is_finite()) || p.z.abs() < 1e-12 {
        return Err(degradation::OWNER_SOURCE_UNDECODABLE);
    }
    Ok(Point2::new(p.x / p.z, p.y / p.z))
}

pub(crate) fn map_world_to_source(
    world: Point2<f64>,
    geometry: &SourceGeometry,
    residual: &residual_warp::ResidualWarp,
) -> Result<Point2<f64>, &'static str> {
    let inverse = (geometry.tile_to_world * geometry.member_to_anchor)
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    map_world_to_source_with_inverse(world, geometry.station_id, residual, &inverse)
}

/// Map a world point using a precomputed composite inverse.  Quality_Gate
/// samples millions of reference pixels per run; computing the same 3x3
/// inverse for every pixel is pure overhead and does not alter the mapping.
pub(crate) fn map_world_to_source_with_inverse(
    world: Point2<f64>,
    station_id: usize,
    residual: &residual_warp::ResidualWarp,
    inverse: &Matrix3<f64>,
) -> Result<Point2<f64>, &'static str> {
    let unwarped = residual.warp_inverse(world, station_id);
    project_geometry(inverse, unwarped)
}

pub(crate) fn map_roi_corners_to_source(
    roi: &QualityRoi,
    geometry: &SourceGeometry,
    residual: &residual_warp::ResidualWarp,
    residual_alignment_px: f64,
) -> Result<[Point2<f64>; 4], &'static str> {
    // NaN and infinities are outside the range as well.
    if !(0.0..=0.5).contains(&residual_alignment_px) {
        return Err(degradation::PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED);
    }
    let (x, y) = roi.world_origin;
    let side = f64::from(roi.side.saturating_sub(1));
    let corners = [
        Point2::new(x, y),
        Point2::new(x + side, y),
        Point2::new(x + side, y + side),
        Point2::new(x, y + side),
    ];
    let mut result = [Point2::origin(); 4];
    for (index, corner) in corners.into_iter().enumerate() {
        result[index] = map_world_to_source(corner, geometry, residual)?;
    }
    Ok(result)
}

// ==================== task 15.2: local scale and unique coverage ====================
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalScaleMeasurement {
    pub median: f64,
    pub fraction_at_least_095: f64,
    pub samples: usize,
    pub diagnostic_render_scale: bool,
}

pub(crate) fn local_scale_for_roi(
    roi: &QualityRoi,
    geometry: &SourceGeometry,
    residual: &residual_warp::ResidualWarp,
    acceptance_render_scale: f64,
) -> Result<LocalScaleMeasurement, &'static str> {
    if roi.side == 0 || !acceptance_render_scale.is_finite() || acceptance_render_scale <= 0.0 {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    let mut scales = Vec::new();
    // Differentiate the complete output->source map at +/-0.5 source-native
    // pixels, then invert its area determinant to obtain source->output scale.
    for y in (0..roi.side).step_by(16) {
        for x in (0..roi.side).step_by(16) {
            let center = Point2::new(
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            );
            let xp = map_world_to_source(
                center + nalgebra::Vector2::new(0.5, 0.0),
                geometry,
                residual,
            )?;
            let xm = map_world_to_source(
                center - nalgebra::Vector2::new(0.5, 0.0),
                geometry,
                residual,
            )?;
            let yp = map_world_to_source(
                center + nalgebra::Vector2::new(0.0, 0.5),
                geometry,
                residual,
            )?;
            let ym = map_world_to_source(
                center - nalgebra::Vector2::new(0.0, 0.5),
                geometry,
                residual,
            )?;
            let dx = xp - xm;
            let dy = yp - ym;
            let determinant = (dx.x * dy.y - dx.y * dy.x).abs();
            if !determinant.is_finite() || determinant <= 1e-12 {
                return Err(degradation::OWNER_SOURCE_UNDECODABLE);
            }
            scales.push(determinant.sqrt());
        }
    }
    scales.sort_by(f64::total_cmp);
    let median = if scales.len() % 2 == 0 {
        (scales[scales.len() / 2 - 1] + scales[scales.len() / 2]) * 0.5
    } else {
        scales[scales.len() / 2]
    };
    Ok(LocalScaleMeasurement {
        median,
        fraction_at_least_095: scales.iter().filter(|&&v| v >= 0.95).count() as f64
            / scales.len() as f64,
        samples: scales.len(),
        diagnostic_render_scale: acceptance_render_scale < 1.0,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EffectivePixelCoverage {
    pub output_nontransparent: u64,
    pub projected_union: u64,
    pub ratio: f64,
}

pub(crate) fn effective_pixel_coverage(
    coverage: &GrayImage,
    source_quadrilaterals: &[[Point2<f64>; 4]],
    canvas_world_origin: (f64, f64),
) -> Result<EffectivePixelCoverage, &'static str> {
    let (width, height) = coverage.dimensions();
    if width == 0 || height == 0 || source_quadrilaterals.is_empty() {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    let mut union = vec![false; width as usize * height as usize];
    for quad in source_quadrilaterals {
        if !quad.iter().all(|p| p.x.is_finite() && p.y.is_finite()) {
            return Err(degradation::DIAGNOSTICS_ROI_INVALID);
        }
        let min_y = quad
            .iter()
            .map(|p| p.y - canvas_world_origin.1)
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as u32;
        let max_y = quad
            .iter()
            .map(|p| p.y - canvas_world_origin.1)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            .max(0.0) as u32;
        for y in min_y..max_y.min(height) {
            let scan_y = canvas_world_origin.1 + f64::from(y) + 0.5;
            let mut intersections = Vec::with_capacity(4);
            for edge in 0..4 {
                let a = quad[edge];
                let b = quad[(edge + 1) % 4];
                if (a.y <= scan_y && scan_y < b.y) || (b.y <= scan_y && scan_y < a.y) {
                    intersections.push(a.x + (scan_y - a.y) * (b.x - a.x) / (b.y - a.y));
                }
            }
            intersections.sort_by(f64::total_cmp);
            for pair in intersections.chunks_exact(2) {
                let start = (pair[0] - canvas_world_origin.0 - 0.5).ceil().max(0.0) as u32;
                let end = (pair[1] - canvas_world_origin.0 - 0.5).ceil().max(0.0) as u32;
                for x in start..end.min(width) {
                    union[(y * width + x) as usize] = true;
                }
            }
        }
    }
    let projected_union = union.iter().filter(|&&covered| covered).count() as u64;
    if projected_union == 0 {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    let output_nontransparent = coverage
        .as_raw()
        .iter()
        .filter(|&&value| value != 0)
        .count() as u64;
    Ok(EffectivePixelCoverage {
        output_nontransparent,
        projected_union,
        ratio: output_nontransparent as f64 / projected_union as f64,
    })
}

// ==================== task 15.3: slanted edge and MTF50 ====================
// Stage 7 metric implementation fragment, to be inserted at quality_gate module scope.
// No run sinks or output mutations. Input RGB is final-output sRGB, not linear RGB.
// The installed imageproc supplies ordinary deterministic Hough (not probabilistic
// Hough). We validate finite, contiguous line support explicitly after Hough.

pub(crate) const MTF_EDGE_MIN_LENGTH_PX: f64 = 128.0;
pub(crate) const MTF_EDGE_MIN_ANGLE_DEG: f64 = 3.0;
pub(crate) const MTF_EDGE_MAX_ANGLE_DEG: f64 = 15.0;
pub(crate) const MTF_EDGE_MIN_CONTRAST: f64 = 0.20;
pub(crate) const MTF_LINE_MAX_RMS_PX: f64 = 0.5;
pub(crate) const FLAT_LOW_FREQUENCY_STD_MAX: f64 = 0.02;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SlantedEdgeEvidence {
    /// false: x = slope*y + intercept; true: y = slope*x + intercept.
    pub transpose: bool,
    pub slope: f64,
    pub intercept: f64,
    pub angle_deg: f64,
    pub length_px: f64,
    pub contrast: f64,
    pub line_fit_rms_px: f64,
    pub first_row: u32,
    pub last_row: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mtf50Measurement {
    pub f50_cycles_per_output_pixel: f64,
    pub normalized: f64,
    pub oversampling: usize,
    pub edge: SlantedEdgeEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FlatRoiEvidence {
    pub low_frequency_luma_std: f64,
    pub opaque_pixels: u64,
}

fn metric_linear_channel(value: f32) -> f64 {
    let value = f64::from(value);
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn metric_luminance(image: &image::Rgb32FImage) -> Option<Vec<f64>> {
    if image.width() == 0 || image.height() == 0 || image.as_raw().iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(
        image
            .pixels()
            .map(|pixel| {
                let rgb = pixel.0.map(metric_linear_channel);
                0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            })
            .collect(),
    )
}

fn metric_median(mut values: Vec<f64>) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let center = values.len() / 2;
    if values.is_empty() {
        0.0
    } else if values.len().is_multiple_of(2) {
        (values[center - 1] + values[center]) * 0.5
    } else {
        values[center]
    }
}

/// Gaussian sigma 2.0, radius 6, exactly 13 taps; replicated borders are used
/// identically for reference and output. The caller never resamples output.
fn metric_gaussian_sigma2(plane: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut kernel = [0.0f64; 13];
    for (index, weight) in kernel.iter_mut().enumerate() {
        *weight = (-0.5 * ((index as f64 - 6.0) / 2.0).powi(2)).exp();
    }
    let total: f64 = kernel.iter().sum();
    for weight in &mut kernel {
        *weight /= total;
    }
    let mut horizontal = vec![0.0; plane.len()];
    for y in 0..height {
        for x in 0..width {
            horizontal[y * width + x] = kernel
                .iter()
                .enumerate()
                .map(|(k, weight)| {
                    let nx = (x as isize + k as isize - 6).clamp(0, width as isize - 1) as usize;
                    weight * plane[y * width + nx]
                })
                .sum();
        }
    }
    let mut output = vec![0.0; plane.len()];
    for y in 0..height {
        for x in 0..width {
            output[y * width + x] = kernel
                .iter()
                .enumerate()
                .map(|(k, weight)| {
                    let ny = (y as isize + k as isize - 6).clamp(0, height as isize - 1) as usize;
                    weight * horizontal[ny * width + x]
                })
                .sum();
        }
    }
    output
}

fn metric_row_value(plane: &[f64], width: usize, transpose: bool, row: u32, column: usize) -> f64 {
    if transpose {
        plane[column * width + row as usize]
    } else {
        plane[row as usize * width + column]
    }
}

/// Catmull-Rom cubic interpolation in one row, used only to localise an edge;
/// it does not resample or replace any output ROI pixel.
fn metric_cubic_row(
    plane: &[f64],
    width: usize,
    columns: usize,
    transpose: bool,
    row: u32,
    x: f64,
) -> f64 {
    let base = x.floor() as isize;
    let fraction = x - base as f64;
    let p: [f64; 4] = std::array::from_fn(|i| {
        let column = (base + i as isize - 1).clamp(0, columns as isize - 1) as usize;
        metric_row_value(plane, width, transpose, row, column)
    });
    p[1] + 0.5
        * fraction
        * (p[2] - p[0]
            + fraction
                * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3]
                    + fraction * (3.0 * (p[1] - p[2]) + p[3] - p[0])))
}

fn metric_fit_line(points: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    if points.len() < 3 {
        return None;
    }
    let n = points.len() as f64;
    let mean_row = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_column = points.iter().map(|p| p.1).sum::<f64>() / n;
    let covariance = points
        .iter()
        .map(|p| (p.0 - mean_row) * (p.1 - mean_column))
        .sum::<f64>();
    let variance = points.iter().map(|p| (p.0 - mean_row).powi(2)).sum::<f64>();
    if variance <= 0.0 {
        return None;
    }
    let slope = covariance / variance;
    let intercept = mean_column - slope * mean_row;
    let rms = (points
        .iter()
        .map(|p| (p.1 - slope * p.0 - intercept).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    Some((slope, intercept, rms))
}

#[allow(clippy::too_many_arguments)]
fn metric_refine_edge(
    plane: &[f64],
    width: usize,
    height: usize,
    transpose: bool,
    slope: f64,
    intercept: f64,
    first_row: u32,
    last_row: u32,
) -> Option<SlantedEdgeEvidence> {
    let columns = if transpose { height } else { width };
    let mut points = Vec::new();
    let mut contrasts = Vec::new();
    for row in first_row..=last_row {
        let center = slope * f64::from(row) + intercept;
        if center < 64.0 || center + 64.0 >= columns as f64 {
            continue;
        }
        let mean = |side: f64| -> f64 {
            (24..=32)
                .map(|distance| {
                    metric_cubic_row(
                        plane,
                        width,
                        columns,
                        transpose,
                        row,
                        center + side * f64::from(distance),
                    )
                })
                .sum::<f64>()
                / 9.0
        };
        let contrast = mean(1.0) - mean(-1.0);
        if contrast.abs() < MTF_EDGE_MIN_CONTRAST {
            continue;
        }
        let direction = contrast.signum();
        // Integrate the derivative of the reconstructed cubic row in a 24px
        // search band. This is a subpixel gradient centroid, not an integer
        // threshold crossing; it treats both edge polarities identically.
        let mut weighted_x = 0.0;
        let mut total_weight = 0.0;
        for index in 0..96 {
            let x = center - 12.0 + index as f64 * 0.25;
            let left = metric_cubic_row(plane, width, columns, transpose, row, x);
            let right = metric_cubic_row(plane, width, columns, transpose, row, x + 0.25);
            let weight = (direction * (right - left)).max(0.0);
            weighted_x += (x + 0.125) * weight;
            total_weight += weight;
        }
        if total_weight > 0.05 {
            points.push((f64::from(row), weighted_x / total_weight));
            contrasts.push(contrast.abs());
        }
    }
    let &(start, _) = points.first()?;
    let &(end, _) = points.last()?;
    let (slope, intercept, rms) = metric_fit_line(&points)?;
    let length = (end - start) * (1.0 + slope * slope).sqrt();
    if length < MTF_EDGE_MIN_LENGTH_PX {
        return None;
    }
    Some(SlantedEdgeEvidence {
        transpose,
        slope,
        intercept,
        angle_deg: slope.abs().atan().to_degrees(),
        length_px: length,
        contrast: metric_median(contrasts),
        line_fit_rms_px: rms,
        first_row: start as u32,
        last_row: end as u32,
    })
}

/// Fixed Canny + ordinary Hough followed by explicit contiguous segment support.
/// The ordinary-Hough limitation is intentional and must remain visible in the
/// stage execution record; it is not called probabilistic Hough.
pub(crate) fn detect_slanted_edge(
    image: &image::Rgb32FImage,
) -> Result<SlantedEdgeEvidence, &'static str> {
    use super::degradation::{
        ROI_NOT_SLANTED_EDGE, SLANTED_EDGE_ANGLE_OUT_OF_RANGE, SLANTED_EDGE_CONTRAST_INSUFFICIENT,
        SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED, SLANTED_EDGE_TOO_SHORT,
    };
    let plane = metric_luminance(image).ok_or(ROI_NOT_SLANTED_EDGE)?;
    let (width, height) = (image.width() as usize, image.height() as usize);
    if width < 129 || height < 129 {
        return Err(ROI_NOT_SLANTED_EDGE);
    }
    let gray = image::GrayImage::from_fn(width as u32, height as u32, |x, y| {
        image::Luma([
            (plane[y as usize * width + x as usize].clamp(0.0, 1.0) * 255.0).round() as u8,
        ])
    });
    let edges = imageproc::edges::canny(&gray, 12.0, 24.0);
    let lines = imageproc::hough::detect_lines(
        &edges,
        imageproc::hough::LineDetectionOptions {
            vote_threshold: 64,
            suppression_radius: 4,
        },
    );
    let no_lines = lines.is_empty();
    let points: Vec<(f64, f64)> = edges
        .enumerate_pixels()
        .filter_map(|(x, y, p)| (p[0] != 0).then_some((f64::from(x), f64::from(y))))
        .collect();
    let mut candidates = Vec::new();
    let mut had_support = false;
    let mut had_long_support = false;
    let mut had_angle_candidate = false;
    for line in lines {
        let angle = f64::from(line.angle_in_degrees).to_radians();
        let (sine, cosine) = angle.sin_cos();
        let transpose = sine.abs() > cosine.abs();
        let (slope, intercept, rows) = if transpose {
            (-cosine / sine, f64::from(line.r) / sine, width)
        } else {
            (-sine / cosine, f64::from(line.r) / cosine, height)
        };
        // Keep near-axis candidates a little outside the final 3..15 degree
        // band; the subpixel line fit, not the 1-degree Hough bin, sets angle.
        if slope.abs().atan().to_degrees() > 17.0 {
            had_angle_candidate = true;
            continue;
        }
        let mut support = vec![false; rows];
        for &(x, y) in &points {
            if (x * cosine + y * sine - f64::from(line.r)).abs() <= 2.5 {
                let row = if transpose { x as usize } else { y as usize };
                support[row] = true;
            }
        }
        let mut runs = Vec::new();
        let mut start = None;
        let mut last = 0;
        for (row, present) in support.into_iter().enumerate() {
            if present {
                if start.is_none() {
                    start = Some(row);
                }
                last = row;
            } else if start.is_some() && row > last + 3 {
                runs.push((start.take().unwrap_or(0), last));
            }
        }
        if let Some(start) = start {
            runs.push((start, last));
        }
        for (start, end) in runs {
            had_support = true;
            if ((end - start) as f64) * (1.0 + slope * slope).sqrt() < MTF_EDGE_MIN_LENGTH_PX {
                continue;
            }
            had_long_support = true;
            if let Some(candidate) = metric_refine_edge(
                &plane,
                width,
                height,
                transpose,
                slope,
                intercept,
                start as u32,
                end as u32,
            ) {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.length_px
            .total_cmp(&a.length_px)
            .then_with(|| a.transpose.cmp(&b.transpose))
            .then_with(|| a.intercept.total_cmp(&b.intercept))
    });
    let mut found_bad_fit = false;
    for candidate in &candidates {
        if candidate.angle_deg < MTF_EDGE_MIN_ANGLE_DEG - 1e-6
            || candidate.angle_deg > MTF_EDGE_MAX_ANGLE_DEG + 1e-6
        {
            had_angle_candidate = true;
            continue;
        }
        if candidate.line_fit_rms_px > MTF_LINE_MAX_RMS_PX {
            found_bad_fit = true;
            continue;
        }
        let isolated = !candidates.iter().any(|other| {
            if other.transpose != candidate.transpose || other.contrast < MTF_EDGE_MIN_CONTRAST {
                return false;
            }
            let first = candidate.first_row.max(other.first_row);
            let last = candidate.last_row.min(other.last_row);
            if last <= first {
                return false;
            }
            // Ignore duplicate hypotheses for the same physical edge, but
            // reject a second contrasting edge anywhere in the ±32px band.
            let row = f64::from(first + last) * 0.5;
            let distance = ((candidate.slope - other.slope) * row + candidate.intercept
                - other.intercept)
                .abs()
                / (1.0 + candidate.slope * candidate.slope).sqrt();
            distance > 3.0 && distance <= 32.0
        });
        if isolated {
            return Ok(candidate.clone());
        }
    }
    Err(if found_bad_fit {
        SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED
    } else if had_angle_candidate {
        SLANTED_EDGE_ANGLE_OUT_OF_RANGE
    } else if had_long_support {
        SLANTED_EDGE_CONTRAST_INSUFFICIENT
    } else if had_support || no_lines {
        SLANTED_EDGE_TOO_SHORT
    } else {
        ROI_NOT_SLANTED_EDGE
    })
}

/// ISO 12233-style slanted-edge measurement. ESF bins represent physical
/// normal distance, so f50 is cycles/output-pixel before Local_Scale division.
pub(crate) fn slanted_edge_mtf50(
    image: &image::Rgb32FImage,
    local_scale: f64,
) -> Result<Mtf50Measurement, &'static str> {
    use super::degradation::{PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED, ROI_NOT_SLANTED_EDGE};
    if !local_scale.is_finite() || local_scale <= 0.0 {
        return Err(PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED);
    }
    let edge = detect_slanted_edge(image)?;
    let plane = metric_luminance(image).ok_or(ROI_NOT_SLANTED_EDGE)?;
    let oversampling = (1.0 / edge.angle_deg.to_radians().tan())
        .round()
        .clamp(4.0, 16.0) as usize;
    let bins = 128 * oversampling + 1;
    let mut sums = vec![0.0; bins];
    let mut counts = vec![0u32; bins];
    let width = image.width() as usize;
    let columns = if edge.transpose {
        image.height()
    } else {
        image.width()
    };
    let normalizer = (1.0 + edge.slope * edge.slope).sqrt();
    for row in edge.first_row..=edge.last_row {
        let center = edge.slope * f64::from(row) + edge.intercept;
        let start = (center - 64.0 * normalizer).ceil().max(0.0) as u32;
        let end = (center + 64.0 * normalizer)
            .floor()
            .min(f64::from(columns - 1)) as u32;
        for column in start..=end {
            let distance = (f64::from(column) - center) / normalizer;
            let bin = ((distance + 64.0) * oversampling as f64).round() as usize;
            if bin < bins {
                sums[bin] += metric_row_value(&plane, width, edge.transpose, row, column as usize);
                counts[bin] += 1;
            }
        }
    }
    let populated: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter_map(|(i, &count)| (count > 0).then_some(i))
        .collect();
    if populated.len() < bins / 2 {
        return Err(ROI_NOT_SLANTED_EDGE);
    }
    let mut esf = vec![0.0; bins];
    for &i in &populated {
        esf[i] = sums[i] / f64::from(counts[i]);
    }
    let first = populated[0];
    let last = *populated.last().ok_or(ROI_NOT_SLANTED_EDGE)?;
    let first_value = esf[first];
    let last_value = esf[last];
    esf[..first].fill(first_value);
    esf[last + 1..].fill(last_value);
    for pair in populated.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        for i in a + 1..b {
            esf[i] = esf[a] + (esf[b] - esf[a]) * (i - a) as f64 / (b - a) as f64;
        }
    }
    let smooth: Vec<f64> = esf
        .windows(4)
        .map(|window| window.iter().sum::<f64>() / 4.0)
        .collect();
    let lsf_len = smooth.len() - 1;
    let lsf: Vec<f64> = smooth
        .windows(2)
        .enumerate()
        .map(|(i, window)| {
            (window[1] - window[0])
                * (0.54 - 0.46 * (std::f64::consts::TAU * i as f64 / (lsf_len - 1) as f64).cos())
        })
        .collect();
    let dc = lsf.iter().sum::<f64>().abs();
    if dc <= 1e-12 {
        return Err(ROI_NOT_SLANTED_EDGE);
    }
    let nfft = lsf.len().next_power_of_two();
    let mut previous = (0.0, 1.0);
    // A real DFT with implicit zero padding: bins above output Nyquist are
    // unmeasurable as an output-image frequency and are never accepted.
    for k in 1..=nfft / (2 * oversampling) {
        let omega = std::f64::consts::TAU * k as f64 / nfft as f64;
        let (mut real, mut imaginary) = (0.0, 0.0);
        for (n, &value) in lsf.iter().enumerate() {
            let (sine, cosine) = (omega * n as f64).sin_cos();
            real += value * cosine;
            imaginary -= value * sine;
        }
        let frequency = k as f64 * oversampling as f64 / nfft as f64;
        let amplitude = real.hypot(imaginary) / dc;
        if amplitude <= 0.5 && previous.1 > 0.5 {
            let fraction = (previous.1 - 0.5) / (previous.1 - amplitude);
            let f50 = previous.0 + fraction * (frequency - previous.0);
            return Ok(Mtf50Measurement {
                f50_cycles_per_output_pixel: f50,
                normalized: f50 / local_scale,
                oversampling,
                edge,
            });
        }
        previous = (frequency, amplitude);
    }
    Err(ROI_NOT_SLANTED_EDGE)
}

// ==================== task 15.4: gradient, flat ROI and noise ====================
/// Uses exactly the production acutance primitive, with RGB replaced by equal
/// linear-luminance channels so chroma differences cannot masquerade as luma
/// gradient energy. Replication gives every ROI pixel a complete 13x13 patch.
pub(crate) fn normalized_gradient_energy(
    image: &image::Rgb32FImage,
    local_scale: f64,
) -> Result<f64, &'static str> {
    use super::degradation::{OWNER_SOURCE_UNDECODABLE, PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED};
    if !local_scale.is_finite() || local_scale <= 0.0 {
        return Err(PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED);
    }
    let plane = metric_luminance(image).ok_or(OWNER_SOURCE_UNDECODABLE)?;
    let (width, height) = (image.width() as usize, image.height() as usize);
    use rayon::prelude::*;
    let values: Vec<f64> = (0..width * height)
        .into_par_iter()
        .map(|index| {
            let x = index % width;
            let y = index / width;
            super::super::mosaic::acutance_with_step(
                |sx, sy| {
                    let nx = (sx.round() as isize).clamp(0, width as isize - 1) as usize;
                    let ny = (sy.round() as isize).clamp(0, height as isize - 1) as usize;
                    Some(image::Rgb([plane[ny * width + nx] as f32; 3]))
                },
                x as f64,
                y as f64,
                1.0,
            )
        })
        .collect();
    // Summing the indexed collection serially preserves the previous
    // floating-point reduction order and therefore the report values.
    let total = values.into_iter().sum::<f64>();
    Ok(total / plane.len() as f64 / local_scale)
}

pub(crate) fn flat_roi(
    image: &image::Rgb32FImage,
    coverage: &image::GrayImage,
) -> Result<FlatRoiEvidence, &'static str> {
    use super::degradation::ROI_NOT_FLAT;
    if image.dimensions() != coverage.dimensions() || coverage.pixels().any(|p| p[0] != 255) {
        return Err(ROI_NOT_FLAT);
    }
    let plane = metric_luminance(image).ok_or(ROI_NOT_FLAT)?;
    let low = metric_gaussian_sigma2(&plane, image.width() as usize, image.height() as usize);
    let mean = low.iter().sum::<f64>() / low.len() as f64;
    let std =
        (low.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / low.len() as f64).sqrt();
    if std > FLAT_LOW_FREQUENCY_STD_MAX || detect_slanted_edge(image).is_ok() {
        return Err(ROI_NOT_FLAT);
    }
    Ok(FlatRoiEvidence {
        low_frequency_luma_std: std,
        opaque_pixels: plane.len() as u64,
    })
}

/// Defined estimator (not variance-renormalised): sigma=1.4826*MAD(highpass),
/// with highpass = linear luminance - sigma2/radius6 Gaussian luminance.
pub(crate) fn noise_sigma(image: &image::Rgb32FImage) -> Result<f64, &'static str> {
    let plane = metric_luminance(image).ok_or(super::degradation::ROI_NOT_FLAT)?;
    let low = metric_gaussian_sigma2(&plane, image.width() as usize, image.height() as usize);
    let high: Vec<f64> = plane
        .iter()
        .zip(low)
        .map(|(value, blurred)| value - blurred)
        .collect();
    let center = metric_median(high.clone());
    Ok(1.4826 * metric_median(high.into_iter().map(|v| (v - center).abs()).collect()))
}

pub(crate) fn noise_sigma_ratio(
    output: &image::Rgb32FImage,
    reference: &image::Rgb32FImage,
) -> Result<f64, &'static str> {
    if output.dimensions() != reference.dimensions() {
        return Err(super::degradation::PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED);
    }
    let output_sigma = noise_sigma(output)?;
    let reference_sigma = noise_sigma(reference)?;
    if reference_sigma <= f64::EPSILON {
        return Err(super::degradation::ROI_NOT_FLAT);
    }
    Ok(output_sigma / reference_sigma)
}

// ==================== task 15.5: low-frequency CIEDE2000 ====================
/// The design fixes the ROI low-frequency mean to the mean of its fully opaque
/// RGB samples. Using compensated f64 summation avoids a large RGB lowpass
/// buffer and feeds the one canonical sRGB->D65 Lab->CIEDE2000 implementation.
pub(crate) fn roi_mean_rgb(image: &image::Rgb32FImage) -> Option<[f32; 3]> {
    if image.width() == 0 || image.height() == 0 {
        return None;
    }
    let mut sums = [0.0f64; 3];
    let mut compensation = [0.0f64; 3];
    for pixel in image.pixels() {
        for channel in 0..3 {
            if !pixel[channel].is_finite() {
                return None;
            }
            let value = f64::from(pixel[channel]) - compensation[channel];
            let next = sums[channel] + value;
            compensation[channel] = (next - sums[channel]) - value;
            sums[channel] = next;
        }
    }
    let count = f64::from(image.width()) * f64::from(image.height());
    Some(sums.map(|v| (v / count) as f32))
}

pub(crate) fn roi_low_frequency_delta_e00(
    output: &image::Rgb32FImage,
    reference: &image::Rgb32FImage,
) -> Result<f64, &'static str> {
    if output.dimensions() != reference.dimensions() {
        return Err(super::degradation::PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED);
    }
    let a = roi_mean_rgb(output).ok_or(super::degradation::OWNER_SOURCE_UNDECODABLE)?;
    let b = roi_mean_rgb(reference).ok_or(super::degradation::OWNER_SOURCE_UNDECODABLE)?;
    Ok(super::tone::delta_e00_rgb(a, b))
}

// ==================== task 15.6: boundary stroke and confidence ====================
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BoundaryStrokeMeasurement {
    pub world: (f64, f64),
    pub normal_error_px: f64,
    pub orientation_error_deg: f64,
}
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryStrokeReport {
    pub measurements: Vec<BoundaryStrokeMeasurement>,
    pub pairable_count: usize,
    pub unpairable_count: usize,
    pub unmeasurable: Vec<((f64, f64), &'static str)>,
    pub p95_error_px: f64,
    pub max_error_px: f64,
}

pub(crate) fn trace_moore_boundary(
    labels: &OwnerRegions,
    left_label: u32,
    right_label: u32,
) -> Result<Vec<(u32, u32)>, &'static str> {
    if left_label == 0 || right_label == 0 || left_label == right_label {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let w = labels.width as usize;
    let h = labels.height as usize;
    let touches = |x: usize, y: usize| -> bool {
        let i = y * w + x;
        if labels.labels[i] != left_label && labels.labels[i] != right_label {
            return false;
        }
        (-1i32..=1)
            .flat_map(|dy| (-1i32..=1).map(move |dx| (dx, dy)))
            .filter(|(dx, dy)| *dx != 0 || *dy != 0)
            .any(|(dx, dy)| {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                nx >= 0
                    && ny >= 0
                    && (nx as usize) < w
                    && (ny as usize) < h
                    && labels.labels[ny as usize * w + nx as usize]
                        == if labels.labels[i] == left_label {
                            right_label
                        } else {
                            left_label
                        }
            })
    };
    let start = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .find(|&(x, y)| touches(x, y))
        .ok_or(degradation::BOUNDARY_NO_PAIRABLE_EDGE)?;
    let dirs: [(i32, i32); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];
    let mut result = Vec::new();
    let mut current = start;
    let mut backtrack = 4usize;
    let mut states = BTreeSet::new();
    let max_steps = w.saturating_mul(h).saturating_mul(8).max(1);
    for _ in 0..max_steps {
        if !touches(current.0, current.1) {
            break;
        }
        if !states.insert((current, backtrack)) {
            break;
        }
        result.push((current.0 as u32, current.1 as u32));
        let mut found = None;
        for offset in 1..=8 {
            let index = (backtrack + offset) % 8;
            let nx = current.0 as i32 + dirs[index].0;
            let ny = current.1 as i32 + dirs[index].1;
            if nx >= 0
                && ny >= 0
                && (nx as usize) < w
                && (ny as usize) < h
                && touches(nx as usize, ny as usize)
            {
                found = Some(((nx as usize, ny as usize), (index + 4) % 8));
                break;
            }
        }
        let Some((next, next_backtrack)) = found else {
            break;
        };
        current = next;
        backtrack = next_backtrack;
        if current == start && result.len() > 2 {
            break;
        }
    }
    if result.len() < 2 {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    Ok(result)
}

fn luminance(pixel: image::Rgb<f32>) -> f64 {
    0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
}

pub(crate) fn measure_boundary_strokes(
    image: &Rgb32FImage,
    boundary: &[(u32, u32)],
    world_origin: (f64, f64),
) -> Result<BoundaryStrokeReport, &'static str> {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 || boundary.len() < 2 {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let mut measurements = Vec::new();
    let mut unpairable_count = 0;
    let mut unmeasurable = Vec::new();
    for (index, &(x, y)) in boundary.iter().enumerate().step_by(256) {
        let prev = boundary[(index + boundary.len() - 1) % boundary.len()];
        let next = boundary[(index + 1) % boundary.len()];
        let tangent = (
            f64::from(next.0) - f64::from(prev.0),
            f64::from(next.1) - f64::from(prev.1),
        );
        let length = tangent.0.hypot(tangent.1);
        if length <= f64::EPSILON {
            unpairable_count += 1;
            continue;
        }
        let normal = (-tangent.1 / length, tangent.0 / length);
        let mut best_left = (0.0f64, 0.0f64, 0.0f64);
        let mut best_right = (0.0f64, 0.0f64, 0.0f64);
        for distance in 1..=16 {
            let d = f64::from(distance);
            for side in [-1.0, 1.0] {
                let sx = f64::from(x) + side * normal.0 * d;
                let sy = f64::from(y) + side * normal.1 * d;
                let ix = sx.round() as i32;
                let iy = sy.round() as i32;
                if ix < 1 || iy < 1 || ix + 1 >= width as i32 || iy + 1 >= height as i32 {
                    continue;
                }
                let gx = luminance(*image.get_pixel((ix + 1) as u32, iy as u32))
                    - luminance(*image.get_pixel((ix - 1) as u32, iy as u32));
                let gy = luminance(*image.get_pixel(ix as u32, (iy + 1) as u32))
                    - luminance(*image.get_pixel(ix as u32, (iy - 1) as u32));
                let contrast = gx.hypot(gy) * 0.5;
                if contrast < 0.15 {
                    continue;
                }
                let orientation = gy.atan2(gx).to_degrees();
                if side < 0.0 {
                    best_left.0 += contrast;
                    best_left.1 += side * d * contrast;
                    best_left.2 += orientation * contrast;
                } else {
                    best_right.0 += contrast;
                    best_right.1 += side * d * contrast;
                    best_right.2 += orientation * contrast;
                }
            }
        }
        if best_left.0 <= 0.0 || best_right.0 <= 0.0 {
            unpairable_count += 1;
            unmeasurable.push((
                (world_origin.0 + f64::from(x), world_origin.1 + f64::from(y)),
                degradation::BOUNDARY_LOW_CONTRAST,
            ));
            continue;
        }
        let left_angle = best_left.2 / best_left.0;
        let right_angle = best_right.2 / best_right.0;
        let mut angle_error = (left_angle - right_angle).abs() % 180.0;
        if angle_error > 90.0 {
            angle_error = 180.0 - angle_error;
        }
        if angle_error > 10.0 {
            unpairable_count += 1;
            unmeasurable.push((
                (world_origin.0 + f64::from(x), world_origin.1 + f64::from(y)),
                degradation::BOUNDARY_ORIENTATION_MISMATCH,
            ));
            continue;
        }
        let left_position = best_left.1 / best_left.0;
        let right_position = best_right.1 / best_right.0;
        let normal_error = (left_position + right_position).abs();
        measurements.push(BoundaryStrokeMeasurement {
            world: (world_origin.0 + f64::from(x), world_origin.1 + f64::from(y)),
            normal_error_px: normal_error,
            orientation_error_deg: angle_error,
        });
    }
    if measurements.is_empty() {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let mut errors = measurements
        .iter()
        .map(|m| m.normal_error_px)
        .collect::<Vec<_>>();
    errors.sort_by(f64::total_cmp);
    let p95 = errors[((errors.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(errors.len() - 1)];
    Ok(BoundaryStrokeReport {
        pairable_count: measurements.len(),
        unpairable_count,
        unmeasurable,
        p95_error_px: p95,
        max_error_px: *errors.last().unwrap_or(&0.0),
        measurements,
    })
}

#[cfg(test)]
mod gradient_tests {
    use super::*;
    use image::{Rgb, Rgb32FImage};

    #[test]
    fn cached_gradient_matches_the_production_acutance_primitive() {
        let mut image = Rgb32FImage::new(17, 19);
        for (i, pixel) in image.pixels_mut().enumerate() {
            let value = ((i * 37 % 251) as f32) / 251.0;
            *pixel = Rgb([value, value * 0.93, value * 0.87]);
        }
        let plane = metric_luminance(&image).unwrap();
        let width = image.width() as usize;
        let height = image.height() as usize;
        let expected = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| {
                crate::panorama_utils::mosaic::acutance_with_step(
                    |sx, sy| {
                        let nx = (sx.round() as isize).clamp(0, width as isize - 1) as usize;
                        let ny = (sy.round() as isize).clamp(0, height as isize - 1) as usize;
                        Some(Rgb([plane[ny * width + nx] as f32; 3]))
                    },
                    x as f64,
                    y as f64,
                    1.0,
                )
            })
            .sum::<f64>()
            / plane.len() as f64;
        let actual = normalized_gradient_energy(&image, 1.0).unwrap();
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }
}
