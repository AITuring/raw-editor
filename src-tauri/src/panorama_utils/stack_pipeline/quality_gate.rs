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
    "sharpness_confidence_coverage",
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
use image::GrayImage;
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
            if labels[i] == 0
                || x == 0
                || y == 0
                || x + 1 == width as usize
                || y + 1 == height as usize
                || labels[i - 1] != labels[i]
                || labels[i + 1] != labels[i]
                || labels[i - width as usize] != labels[i]
                || labels[i + width as usize] != labels[i]
            {
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
    let unwarped = residual.warp_inverse(world, geometry.station_id);
    project_geometry(&inverse, unwarped)
}

pub(crate) fn map_roi_corners_to_source(
    roi: &QualityRoi,
    geometry: &SourceGeometry,
    residual: &residual_warp::ResidualWarp,
    residual_alignment_px: f64,
) -> Result<[Point2<f64>; 4], &'static str> {
    if !residual_alignment_px.is_finite()
        || residual_alignment_px < 0.0
        || residual_alignment_px > 0.5
    {
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
