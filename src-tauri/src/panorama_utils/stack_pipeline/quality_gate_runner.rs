//! Runtime adapter for the pure Quality_Gate measurements.
//!
//! `quality_gate.rs` deliberately contains no image loading or report policy.
//! This module supplies that small amount of orchestration: it selects ROIs
//! from the final owner planes, decodes one owner source at a time, samples the
//! corresponding reference ROI, and records all nine criteria.  The caller can
//! therefore use it in observation-only mode without changing export pixels.

use std::collections::BTreeMap;

use image::{GrayImage, ImageBuffer, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Vector3};

use super::degradation;
use super::quality_gate::{self, EffectivePixelCoverage, QualityRoi, SourceGeometry};
use super::report::{
    FailedMeasurementRecord, QualityGateCriterionRecord, QualityGateReport, QualityGateVerdict,
    UnmeasurableRecord, WorldPoint,
};
use super::residual_warp::ResidualWarp;

/// A source entry used by the runtime adapter. `owner` is the value copied to
/// the final ownership plane; IDs need not be contiguous.
#[derive(Debug, Clone)]
pub(crate) struct QualitySource {
    pub owner: u16,
    pub path: String,
    pub geometry: SourceGeometry,
    pub dimensions: (u32, u32),
}

/// Inputs retained by the compositor after the final image is assembled.
/// `textured` is optional for older callers; an absent plane treats every
/// covered pixel as textured, preserving the pre-Textured_Pixel behaviour.
pub(crate) struct QualityGateInput<'a> {
    pub output: &'a Rgb32FImage,
    pub coverage: &'a GrayImage,
    pub ownership: &'a [u16],
    pub confidence: &'a [f32],
    pub textured: Option<&'a [u8]>,
    pub world_origin: (f64, f64),
    pub sources: &'a [QualitySource],
    pub residual: &'a ResidualWarp,
    pub acceptance_render_scale: f64,
    pub final_sharpen_amount: f64,
}

pub(crate) type SourceLoader<'a> = dyn FnMut(&QualitySource) -> Result<Rgb32FImage, String> + 'a;

struct Criterion {
    name: &'static str,
    threshold: f64,
    min: Option<f64>,
    max: Option<f64>,
    measured: Vec<FailedMeasurementRecord>,
    failed: Vec<FailedMeasurementRecord>,
    unmeasurable: usize,
    diagnostic: bool,
    observed_count: usize,
    /// Keep reports bounded when a large canvas has many boundary samples.
    /// The count remains exact while only a deterministic prefix is retained
    /// for the per-point diagnostics.
    record_limit: usize,
}

impl Default for Criterion {
    fn default() -> Self {
        Self {
            name: "",
            threshold: 0.0,
            min: None,
            max: None,
            measured: Vec::new(),
            failed: Vec::new(),
            unmeasurable: 0,
            diagnostic: false,
            observed_count: 0,
            record_limit: usize::MAX,
        }
    }
}

impl Criterion {
    fn new(name: &'static str, threshold: f64) -> Self {
        Self {
            name,
            threshold,
            ..Self::default()
        }
    }

    fn bounded(name: &'static str, min: f64, max: f64) -> Self {
        Self {
            name,
            threshold: min,
            min: Some(min),
            max: Some(max),
            ..Self::default()
        }
    }

    fn p95_with_max(name: &'static str, p95: f64, max: f64) -> Self {
        Self {
            name,
            threshold: p95,
            max: Some(max),
            ..Self::default()
        }
    }

    fn push(&mut self, world: (f64, f64), value: f64, owner_path: &str, pass: bool) {
        self.observed_count = self.observed_count.saturating_add(1);
        let item = FailedMeasurementRecord {
            world: WorldPoint {
                x: world.0,
                y: world.1,
            },
            measured: value,
            owner_path: owner_path.to_string(),
        };
        if self.measured.len() < self.record_limit {
            self.measured.push(item.clone());
        }
        if !pass && self.failed.len() < self.record_limit {
            self.failed.push(item);
        }
    }

    fn miss(&mut self) {
        self.unmeasurable += 1;
    }

    fn finish(self, _unmeasurable: &mut Vec<UnmeasurableRecord>) -> QualityGateCriterionRecord {
        // `unmeasurable` records are appended by the caller while processing
        // each ROI; this method only carries the aggregate count.
        QualityGateCriterionRecord {
            name: self.name.to_string(),
            threshold: self.threshold,
            threshold_min: self.min,
            threshold_max: self.max,
            measurable_count: self.observed_count,
            unmeasurable_count: self.unmeasurable,
            measured: self.measured,
            diagnostic: self.diagnostic,
            failed: self.failed,
        }
    }
}

fn world(roi: &QualityRoi) -> (f64, f64) {
    roi.world_origin
}

fn output_roi(image: &Rgb32FImage, roi: &QualityRoi) -> Option<Rgb32FImage> {
    let side = roi.side;
    if roi.x.checked_add(side)? > image.width() || roi.y.checked_add(side)? > image.height() {
        return None;
    }
    Some(ImageBuffer::from_fn(side, side, |x, y| {
        *image.get_pixel(roi.x + x, roi.y + y)
    }))
}

fn project(matrix: &Matrix3<f64>, p: Point2<f64>) -> Option<Point2<f64>> {
    let v = matrix * Vector3::new(p.x, p.y, 1.0);
    if !v.iter().all(|x| x.is_finite()) || v.z.abs() < 1e-12 {
        return None;
    }
    Some(Point2::new(v.x / v.z, v.y / v.z))
}

fn source_quad(source: &QualitySource) -> Option<[Point2<f64>; 4]> {
    let matrix = source.geometry.tile_to_world * source.geometry.member_to_anchor;
    let (w, h) = source.dimensions;
    let corners = [
        Point2::new(0.0, 0.0),
        Point2::new(f64::from(w.saturating_sub(1)), 0.0),
        Point2::new(
            f64::from(w.saturating_sub(1)),
            f64::from(h.saturating_sub(1)),
        ),
        Point2::new(0.0, f64::from(h.saturating_sub(1))),
    ];
    Some([
        project(&matrix, corners[0])?,
        project(&matrix, corners[1])?,
        project(&matrix, corners[2])?,
        project(&matrix, corners[3])?,
    ])
}

/// Bilinear source sampling through the complete output-to-source mapping.
/// This is the single allowed reference resample; output pixels are copied as
/// is by [`output_roi`].
fn source_roi(
    source: &Rgb32FImage,
    roi: &QualityRoi,
    geometry: &SourceGeometry,
    residual: &ResidualWarp,
) -> Result<Rgb32FImage, &'static str> {
    let mut result = Rgb32FImage::new(roi.side, roi.side);
    for y in 0..roi.side {
        for x in 0..roi.side {
            let world = Point2::new(
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            );
            let mapped = quality_gate::map_world_to_source(world, geometry, residual)?;
            let max_x = f64::from(source.width().saturating_sub(1));
            let max_y = f64::from(source.height().saturating_sub(1));
            if !mapped.x.is_finite()
                || !mapped.y.is_finite()
                || mapped.x < 0.0
                || mapped.y < 0.0
                || mapped.x > max_x
                || mapped.y > max_y
            {
                return Err(degradation::OWNER_SOURCE_UNDECODABLE);
            }
            let sx = mapped.x;
            let sy = mapped.y;
            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let x1 = (x0 + 1).min(source.width().saturating_sub(1));
            let y1 = (y0 + 1).min(source.height().saturating_sub(1));
            let tx = (sx - f64::from(x0)) as f32;
            let ty = (sy - f64::from(y0)) as f32;
            let a = *source.get_pixel(x0, y0);
            let b = *source.get_pixel(x1, y0);
            let c = *source.get_pixel(x0, y1);
            let d = *source.get_pixel(x1, y1);
            let pixel = Rgb(std::array::from_fn(|channel| {
                let top = a[channel] * (1.0 - tx) + b[channel] * tx;
                let bottom = c[channel] * (1.0 - tx) + d[channel] * tx;
                top * (1.0 - ty) + bottom * ty
            }));
            result.put_pixel(x, y, pixel);
        }
    }
    Ok(result)
}

fn record_unmeasurable(
    criterion: &mut Criterion,
    records: &mut Vec<UnmeasurableRecord>,
    roi: &QualityRoi,
    reason: &'static str,
) {
    criterion.miss();
    records.push(UnmeasurableRecord {
        criterion: criterion.name.to_string(),
        world: WorldPoint {
            x: roi.world_origin.0,
            y: roi.world_origin.1,
        },
        reason: reason.to_string(),
    });
}

fn ratio(measured: f64, reference: f64, min: f64, max: f64) -> Option<(f64, bool)> {
    quality_gate::compare_ratio(measured, reference, min, max, false).map(|v| (v.value, v.passed))
}

/// Keep the observation-only gate bounded on the full-resolution acceptance
/// renders. The exact region labelling and distance transform are useful for
/// ordinary unit-sized fixtures, but a 9k by 11k canvas would allocate and
/// scan several gigabytes just to choose 512px ROIs. A deterministic lattice
/// still checks ownership and coverage over every pixel in a candidate ROI;
/// effective coverage and boundary tracing use separate streaming passes.
fn select_sparse_rois(
    coverage: &GrayImage,
    ownership: &[u16],
    origin: (f64, f64),
) -> Vec<QualityRoi> {
    let (width, height) = coverage.dimensions();
    let side = quality_gate::QUALITY_ROI_SIDE;
    if width < side || height < side {
        return Vec::new();
    }
    let mut result = Vec::new();
    for y in (0..=height - side).step_by(side as usize / 2) {
        for x in (0..=width - side).step_by(side as usize / 2) {
            let owner = ownership[(y * width + x) as usize];
            if owner == 0 || coverage.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let mut same_owner = true;
            for ry in 0..side {
                for rx in 0..side {
                    let index = ((y + ry) * width + x + rx) as usize;
                    if coverage.as_raw()[index] == 0 || ownership[index] != owner {
                        same_owner = false;
                        break;
                    }
                }
                if !same_owner {
                    break;
                }
            }
            if same_owner {
                // A candidate must have a 16px owner/coverage margin on all
                // four sides. Checking this narrow ring is equivalent to the
                // distance-transform predicate used by the exact small-plane
                // selector, while keeping the large-plane path bounded.
                let margin = quality_gate::QUALITY_ROI_MARGIN as u32;
                let margin_ok = x >= margin
                    && y >= margin
                    && x + side + margin <= width
                    && y + side + margin <= height
                    && (y - margin..y + side + margin).all(|row| {
                        (x - margin..x).all(|column| {
                            let index = (row * width + column) as usize;
                            coverage.as_raw()[index] != 0 && ownership[index] == owner
                        }) && (x + side..x + side + margin).all(|column| {
                            let index = (row * width + column) as usize;
                            coverage.as_raw()[index] != 0 && ownership[index] == owner
                        })
                    })
                    && (y - margin..y).all(|row| {
                        (x..x + side).all(|column| {
                            let index = (row * width + column) as usize;
                            coverage.as_raw()[index] != 0 && ownership[index] == owner
                        })
                    })
                    && (y + side..y + side + margin).all(|row| {
                        (x..x + side).all(|column| {
                            let index = (row * width + column) as usize;
                            coverage.as_raw()[index] != 0 && ownership[index] == owner
                        })
                    });
                if !margin_ok {
                    continue;
                }
                result.push(QualityRoi {
                    x,
                    y,
                    side,
                    region_label: result.len() as u32 + 1,
                    owner,
                    world_origin: (origin.0 + f64::from(x), origin.1 + f64::from(y)),
                    boundary_clearance: f64::from(side),
                });
                if result.len() == quality_gate::QUALITY_ROI_MAX_COUNT {
                    return result;
                }
            }
        }
    }
    result
}

fn streaming_confidence_coverage(coverage: &GrayImage, confidence: &[f32]) -> Option<f64> {
    let (width, height) = coverage.dimensions();
    let mut covered = 0usize;
    let mut low = 0usize;
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            // A non-finite value is an unknown confidence, not a sharp pixel.
            if coverage.as_raw()[index] == 0 || !confidence[index].is_finite() {
                continue;
            }
            covered += 1;
            if confidence[index] < 0.05 {
                low += 1;
            }
        }
    }
    (covered > 0).then(|| 1.0 - low as f64 / covered as f64)
}

/// Count the unique projected source coverage one scanline at a time. This is
/// equivalent to `quality_gate::effective_pixel_coverage`, but never allocates
/// a canvas-sized union bitmap (which is prohibitive for the acceptance
/// renders). Only the current row's sorted intervals are resident.
fn effective_pixel_coverage_streaming(
    coverage: &GrayImage,
    source_quadrilaterals: &[[Point2<f64>; 4]],
    canvas_world_origin: (f64, f64),
) -> Result<EffectivePixelCoverage, &'static str> {
    let (width, height) = coverage.dimensions();
    if width == 0 || height == 0 || source_quadrilaterals.is_empty() {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    let output_nontransparent = coverage
        .as_raw()
        .iter()
        .filter(|&&value| value != 0)
        .count() as u64;
    let mut projected_union = 0u64;
    for y in 0..height {
        let scan_y = canvas_world_origin.1 + f64::from(y) + 0.5;
        let mut intervals = Vec::<(f64, f64)>::new();
        for quad in source_quadrilaterals {
            if !quad
                .iter()
                .all(|point| point.x.is_finite() && point.y.is_finite())
            {
                return Err(degradation::DIAGNOSTICS_ROI_INVALID);
            }
            let mut intersections = [0.0f64; 4];
            let mut count = 0usize;
            for edge in 0..4 {
                let a = quad[edge];
                let b = quad[(edge + 1) % 4];
                if ((a.y <= scan_y && scan_y < b.y) || (b.y <= scan_y && scan_y < a.y))
                    && count < intersections.len()
                {
                    intersections[count] = a.x + (scan_y - a.y) * (b.x - a.x) / (b.y - a.y);
                    count += 1;
                }
            }
            if count >= 2 {
                intersections[..count].sort_by(f64::total_cmp);
                for pair in intersections[..count].chunks_exact(2) {
                    let start = (pair[0] - canvas_world_origin.0 - 0.5)
                        .ceil()
                        .max(0.0)
                        .min(f64::from(width)) as u32;
                    let end = (pair[1] - canvas_world_origin.0 - 0.5)
                        .ceil()
                        .max(0.0)
                        .min(f64::from(width)) as u32;
                    if end > start {
                        intervals.push((f64::from(start), f64::from(end)));
                    }
                }
            }
        }
        intervals.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
        });
        let mut union = 0u64;
        let mut active: Option<(f64, f64)> = None;
        for (start, end) in intervals {
            let Some((active_start, active_end)) = active else {
                active = Some((start, end));
                continue;
            };
            if start > active_end {
                union = union.saturating_add((active_end - active_start).max(0.0) as u64);
                active = Some((start, end));
            } else if end > active_end {
                active = Some((active_start, end));
            }
        }
        if let Some((active_start, active_end)) = active {
            union = union.saturating_add((active_end - active_start).max(0.0) as u64);
        }
        projected_union = projected_union.saturating_add(union);
    }
    if projected_union == 0 {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    Ok(EffectivePixelCoverage {
        output_nontransparent,
        projected_union,
        ratio: output_nontransparent as f64 / projected_union as f64,
    })
}

/// Return the two-sided subpixel edge displacement at one ownership boundary.
/// `normal` is either horizontal `(1,0)` or vertical `(0,1)` and is derived
/// directly from the neighbouring ownership pixels, so no full label image is
/// needed.
fn boundary_point_error(image: &Rgb32FImage, x: u32, y: u32, normal: (f64, f64)) -> Option<f64> {
    let (width, height) = image.dimensions();
    let luminance = |pixel: Rgb<f32>| {
        0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
    };
    let mut sides = [(0.0f64, 0.0f64), (0.0f64, 0.0f64)];
    for (side_index, side) in [-1.0f64, 1.0].into_iter().enumerate() {
        let mut contrast_sum = 0.0;
        let mut position_sum = 0.0;
        let mut orientation_sum = 0.0;
        for distance in 1..=16 {
            let distance = f64::from(distance);
            let sx = f64::from(x) + side * normal.0 * distance;
            let sy = f64::from(y) + side * normal.1 * distance;
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
            contrast_sum += contrast;
            position_sum += side * distance * contrast;
            orientation_sum += gy.atan2(gx).to_degrees() * contrast;
        }
        if contrast_sum <= 0.0 {
            return None;
        }
        sides[side_index] = (position_sum / contrast_sum, orientation_sum / contrast_sum);
    }
    let mut angle_error = (sides[0].1 - sides[1].1).abs() % 180.0;
    if angle_error > 90.0 {
        angle_error = 180.0 - angle_error;
    }
    (angle_error <= 10.0).then(|| (sides[0].0 + sides[1].0).abs())
}

/// Stream ownership transitions for large canvases. A fixed histogram keeps
/// the P95 exact to 0.01px while retaining only a bounded diagnostic prefix.
fn measure_large_boundaries(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    origin: (f64, f64),
    criterion: &mut Criterion,
    unmeasurable: &mut Vec<UnmeasurableRecord>,
) {
    let (width, height) = coverage.dimensions();
    let width_usize = width as usize;
    let mut seen = BTreeMap::<(u16, u16), u32>::new();
    let mut hist = [0u64; 301];
    let mut measured = 0u64;
    let mut max_error = 0.0f64;
    let mut first_world = origin;
    let mut first_path = String::new();
    let mut visit = |x: u32, y: u32, normal: (f64, f64), left: u16, right: u16| {
        if left == 0 || right == 0 || left == right {
            return;
        }
        let pair = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        let counter = seen.entry(pair).or_default();
        let sample = (*counter).is_multiple_of(256);
        *counter = counter.saturating_add(1);
        if !sample {
            return;
        }
        let world = (origin.0 + f64::from(x), origin.1 + f64::from(y));
        let Some(error) = boundary_point_error(image, x, y, normal) else {
            unmeasurable.push(UnmeasurableRecord {
                criterion: criterion.name.to_string(),
                world: WorldPoint {
                    x: world.0,
                    y: world.1,
                },
                reason: degradation::BOUNDARY_NO_PAIRABLE_EDGE.to_string(),
            });
            criterion.miss();
            return;
        };
        if measured == 0 {
            first_world = world;
            first_path = format!("owner:{}/owner:{}", pair.0, pair.1);
        }
        measured = measured.saturating_add(1);
        max_error = max_error.max(error);
        let bucket = (error.max(0.0) * 100.0).floor().min(300.0) as usize;
        hist[bucket] = hist[bucket].saturating_add(1);
        criterion.push(
            world,
            error,
            &format!("owner:{}/owner:{}", pair.0, pair.1),
            error <= quality_gate::QUALITY_BOUNDARY_MAX,
        );
    };
    for y in 0..height {
        for x in 0..width.saturating_sub(1) {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] != 0
                && coverage.as_raw()[index + 1] != 0
                && ownership[index] != ownership[index + 1]
            {
                visit(x, y, (1.0, 0.0), ownership[index], ownership[index + 1]);
            }
        }
    }
    for y in 0..height.saturating_sub(1) {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] != 0
                && coverage.as_raw()[index + width_usize] != 0
                && ownership[index] != ownership[index + width_usize]
            {
                visit(
                    x,
                    y,
                    (0.0, 1.0),
                    ownership[index],
                    ownership[index + width_usize],
                );
            }
        }
    }
    if measured == 0 {
        criterion.miss();
        unmeasurable.push(UnmeasurableRecord {
            criterion: criterion.name.to_string(),
            world: WorldPoint {
                x: origin.0,
                y: origin.1,
            },
            reason: degradation::BOUNDARY_NO_PAIRABLE_EDGE.to_string(),
        });
        return;
    }
    let target = (measured * 95).div_ceil(100).max(1);
    let mut cumulative = 0u64;
    let mut p95 = 3.0;
    for (index, count) in hist.iter().enumerate() {
        cumulative = cumulative.saturating_add(*count);
        if cumulative >= target {
            p95 = index as f64 / 100.0;
            break;
        }
    }
    criterion.push(
        first_world,
        p95,
        &first_path,
        p95 <= quality_gate::QUALITY_BOUNDARY_P95_MAX
            && max_error <= quality_gate::QUALITY_BOUNDARY_MAX,
    );
}

/// Execute all nine criteria in record-only mode. No decision from the return
/// value is applied to export; callers decide whether to block in a later task.
pub(crate) fn run_quality_gate(
    input: &QualityGateInput<'_>,
    load_source: &mut SourceLoader<'_>,
) -> QualityGateReport {
    let mut report = QualityGateReport::default();
    if input.output.dimensions() != input.coverage.dimensions()
        || input.ownership.len() != input.coverage.as_raw().len()
        || input.confidence.len() != input.coverage.as_raw().len()
    {
        report.verdict = QualityGateVerdict::InsufficientEvidence;
        report.unmeasurable.push(UnmeasurableRecord {
            criterion: "all".to_string(),
            world: WorldPoint {
                x: input.world_origin.0,
                y: input.world_origin.1,
            },
            reason: "diagnostics_roi_invalid".to_string(),
        });
        return report;
    }
    let large_plane =
        (input.coverage.width() as u64).saturating_mul(input.coverage.height() as u64) > 40_000_000;
    let (regions, rois) = if large_plane {
        (
            None,
            select_sparse_rois(input.coverage, input.ownership, input.world_origin),
        )
    } else {
        let regions = match quality_gate::label_owner_regions(input.ownership, input.coverage) {
            Ok(value) => value,
            Err(reason) => {
                report.verdict = QualityGateVerdict::InsufficientEvidence;
                report.unmeasurable.push(UnmeasurableRecord {
                    criterion: "all".to_string(),
                    world: WorldPoint {
                        x: input.world_origin.0,
                        y: input.world_origin.1,
                    },
                    reason: reason.to_string(),
                });
                return report;
            }
        };
        let rois = match quality_gate::select_quality_rois(&regions, input.world_origin) {
            Ok(value) => value.rois,
            Err(reason) => {
                report.verdict = QualityGateVerdict::InsufficientEvidence;
                report.unmeasurable.push(UnmeasurableRecord {
                    criterion: "all".to_string(),
                    world: WorldPoint {
                        x: input.world_origin.0,
                        y: input.world_origin.1,
                    },
                    reason: reason.to_string(),
                });
                return report;
            }
        };
        (Some(regions), rois)
    };
    report.roi_count = rois.len();

    let mut local_scale = Criterion::new(
        "local_scale_median",
        quality_gate::QUALITY_LOCAL_SCALE_MEDIAN_MIN,
    );
    local_scale.diagnostic = input.acceptance_render_scale < 1.0;
    let mut local_ratio = Criterion::new(
        "local_scale_pixel_ratio",
        quality_gate::QUALITY_LOCAL_SCALE_PIXEL_RATIO_MIN,
    );
    let mut mtf = Criterion::new("mtf50_normalized", quality_gate::QUALITY_MTF50_RATIO_MIN);
    let mut gradient = Criterion::new(
        "gradient_energy_normalized",
        quality_gate::QUALITY_GRADIENT_RATIO_MIN,
    );
    let mut noise = Criterion::bounded(
        "noise_sigma_ratio",
        quality_gate::QUALITY_NOISE_RATIO_MIN,
        quality_gate::QUALITY_NOISE_RATIO_MAX,
    );
    noise.diagnostic = input.final_sharpen_amount > 0.0;
    let mut delta = Criterion::new("roi_delta_e00", quality_gate::QUALITY_ROI_DELTA_E00_MAX);
    let mut unmeasurable = Vec::new();

    // Grouping keeps a decoded Source_RAW alive only for that owner's ROIs.
    let mut by_owner: BTreeMap<u16, Vec<&QualityRoi>> = BTreeMap::new();
    for roi in &rois {
        by_owner.entry(roi.owner).or_default().push(roi);
    }
    let source_map: BTreeMap<u16, &QualitySource> =
        input.sources.iter().map(|s| (s.owner, s)).collect();
    for (owner, owner_rois) in by_owner {
        let Some(source_info) = source_map.get(&owner).copied() else {
            for roi in owner_rois {
                for criterion in [
                    &mut local_scale,
                    &mut local_ratio,
                    &mut mtf,
                    &mut gradient,
                    &mut noise,
                    &mut delta,
                ] {
                    record_unmeasurable(
                        criterion,
                        &mut unmeasurable,
                        roi,
                        "owner_source_undecodable",
                    );
                }
            }
            continue;
        };
        let source = match load_source(source_info) {
            Ok(value) => value,
            Err(_) => {
                for roi in owner_rois {
                    for criterion in [
                        &mut local_scale,
                        &mut local_ratio,
                        &mut mtf,
                        &mut gradient,
                        &mut noise,
                        &mut delta,
                    ] {
                        record_unmeasurable(
                            criterion,
                            &mut unmeasurable,
                            roi,
                            "owner_source_undecodable",
                        );
                    }
                }
                continue;
            }
        };
        for roi in owner_rois {
            let where_ = world(roi);
            let output_roi = output_roi(input.output, roi);
            let reference = source_roi(&source, roi, &source_info.geometry, input.residual);
            let (Some(output_roi), Ok(reference)) = (output_roi, reference) else {
                for criterion in [
                    &mut local_scale,
                    &mut local_ratio,
                    &mut mtf,
                    &mut gradient,
                    &mut noise,
                    &mut delta,
                ] {
                    record_unmeasurable(
                        criterion,
                        &mut unmeasurable,
                        roi,
                        "owner_source_undecodable",
                    );
                }
                continue;
            };
            let scale = quality_gate::local_scale_for_roi(
                roi,
                &source_info.geometry,
                input.residual,
                input.acceptance_render_scale,
            );
            let local = scale.as_ref().ok().map(|value| value.median).unwrap_or(1.0);
            match scale {
                Ok(value) => {
                    local_scale.push(
                        where_,
                        value.median,
                        &source_info.path,
                        value.median >= quality_gate::QUALITY_LOCAL_SCALE_MEDIAN_MIN
                            && input.acceptance_render_scale >= 1.0,
                    );
                    local_ratio.push(
                        where_,
                        value.fraction_at_least_095,
                        &source_info.path,
                        value.fraction_at_least_095
                            >= quality_gate::QUALITY_LOCAL_SCALE_PIXEL_RATIO_MIN,
                    );
                }
                Err(reason) => {
                    record_unmeasurable(&mut local_scale, &mut unmeasurable, roi, reason);
                    record_unmeasurable(&mut local_ratio, &mut unmeasurable, roi, reason);
                }
            }
            match (
                quality_gate::slanted_edge_mtf50(&output_roi, local),
                quality_gate::slanted_edge_mtf50(&reference, 1.0),
            ) {
                (Ok(a), Ok(b)) => {
                    if let Some((value, pass)) = ratio(
                        a.normalized,
                        b.normalized,
                        quality_gate::QUALITY_MTF50_RATIO_MIN,
                        f64::INFINITY,
                    ) {
                        mtf.push(where_, value, &source_info.path, pass);
                    } else {
                        record_unmeasurable(&mut mtf, &mut unmeasurable, roi, "mtf_reference_zero");
                    }
                }
                (Err(reason), _) | (_, Err(reason)) => {
                    record_unmeasurable(&mut mtf, &mut unmeasurable, roi, reason)
                }
            }
            match (
                quality_gate::normalized_gradient_energy(&output_roi, local),
                quality_gate::normalized_gradient_energy(&reference, 1.0),
            ) {
                (Ok(a), Ok(b)) => {
                    if let Some((value, pass)) = ratio(
                        a,
                        b,
                        quality_gate::QUALITY_GRADIENT_RATIO_MIN,
                        f64::INFINITY,
                    ) {
                        gradient.push(where_, value, &source_info.path, pass);
                    } else {
                        record_unmeasurable(
                            &mut gradient,
                            &mut unmeasurable,
                            roi,
                            "gradient_reference_zero",
                        );
                    }
                }
                (Err(reason), _) | (_, Err(reason)) => {
                    record_unmeasurable(&mut gradient, &mut unmeasurable, roi, reason)
                }
            }
            let full = GrayImage::from_pixel(roi.side, roi.side, image::Luma([255]));
            match (
                quality_gate::noise_sigma_ratio(&output_roi, &reference),
                quality_gate::flat_roi(&output_roi, &full),
                quality_gate::flat_roi(&reference, &full),
            ) {
                (Ok(value), Ok(_), Ok(_)) => noise.push(
                    where_,
                    value,
                    &source_info.path,
                    (quality_gate::QUALITY_NOISE_RATIO_MIN..=quality_gate::QUALITY_NOISE_RATIO_MAX)
                        .contains(&value)
                        && input.final_sharpen_amount <= 0.0,
                ),
                _ => record_unmeasurable(&mut noise, &mut unmeasurable, roi, "roi_not_flat"),
            }
            match quality_gate::roi_low_frequency_delta_e00(&output_roi, &reference) {
                Ok(value) => delta.push(
                    where_,
                    value,
                    &source_info.path,
                    value <= quality_gate::QUALITY_ROI_DELTA_E00_MAX,
                ),
                Err(reason) => record_unmeasurable(&mut delta, &mut unmeasurable, roi, reason),
            }
        }
    }

    let mut effective = Criterion::new(
        "effective_pixel_count",
        quality_gate::QUALITY_EFFECTIVE_PIXEL_RATIO_MIN,
    );
    let quads = input
        .sources
        .iter()
        .filter_map(source_quad)
        .collect::<Vec<_>>();
    let effective_result =
        effective_pixel_coverage_streaming(input.coverage, &quads, input.world_origin);
    match effective_result {
        Ok(EffectivePixelCoverage { ratio, .. }) => effective.push(
            input.world_origin,
            ratio,
            "",
            ratio >= quality_gate::QUALITY_EFFECTIVE_PIXEL_RATIO_MIN,
        ),
        Err(reason) => {
            effective.miss();
            unmeasurable.push(UnmeasurableRecord {
                criterion: effective.name.to_string(),
                world: WorldPoint {
                    x: input.world_origin.0,
                    y: input.world_origin.1,
                },
                reason: reason.to_string(),
            });
        }
    }

    let mut boundary = Criterion::p95_with_max(
        "boundary_stroke_alignment",
        quality_gate::QUALITY_BOUNDARY_P95_MAX,
        quality_gate::QUALITY_BOUNDARY_MAX,
    );
    boundary.record_limit = 2_048;
    if let Some(regions) = regions.as_ref() {
        for left in &regions.regions {
            for right in &regions.regions {
                if left.label >= right.label || left.owner == right.owner {
                    continue;
                }
                let Ok(path) = quality_gate::trace_moore_boundary(regions, left.label, right.label)
                else {
                    continue;
                };
                match quality_gate::measure_boundary_strokes(
                    input.output,
                    &path,
                    input.world_origin,
                ) {
                    Ok(measurement) => {
                        for value in measurement.measurements {
                            boundary.push(
                                value.world,
                                value.normal_error_px,
                                "",
                                value.normal_error_px <= quality_gate::QUALITY_BOUNDARY_MAX,
                            );
                        }
                        // The criterion is defined by both the 95th percentile
                        // and the absolute maximum. Keep the sampled points for
                        // diagnostics and add one aggregate P95 measurement so a
                        // high but sub-3px tail cannot be reported as a pass.
                        boundary.push(
                            input.world_origin,
                            measurement.p95_error_px,
                            "",
                            measurement.p95_error_px <= quality_gate::QUALITY_BOUNDARY_P95_MAX
                                && measurement.max_error_px <= quality_gate::QUALITY_BOUNDARY_MAX,
                        );
                    }
                    Err(reason) => {
                        boundary.miss();
                        unmeasurable.push(UnmeasurableRecord {
                            criterion: boundary.name.to_string(),
                            world: WorldPoint {
                                x: input.world_origin.0,
                                y: input.world_origin.1,
                            },
                            reason: reason.to_string(),
                        });
                    }
                }
            }
        }
    } else {
        measure_large_boundaries(
            input.output,
            input.coverage,
            input.ownership,
            input.world_origin,
            &mut boundary,
            &mut unmeasurable,
        );
    }

    let mut confidence = Criterion::new(
        "sharpness_confidence_coverage",
        1.0 - quality_gate::QUALITY_LOW_CONFIDENCE_RATIO_MAX,
    );
    // Without any known confidence value the criterion has no evidence.
    let confidence_known = input.confidence.iter().any(|value| value.is_finite());
    let confidence_value = match input.textured {
        _ if !confidence_known => None,
        Some(textured) if textured.len() == input.confidence.len() => {
            let covered = input
                .coverage
                .as_raw()
                .iter()
                .zip(textured)
                .filter(|&(&mask, &textured)| mask != 0 && textured != 0)
                .count();
            if covered == 0 {
                None
            } else {
                let low = input
                    .coverage
                    .as_raw()
                    .iter()
                    .zip(input.confidence)
                    .zip(textured)
                    .filter(|&((&mask, &value), &textured)| {
                        mask != 0 && textured != 0 && value.is_finite() && value < 0.05
                    })
                    .count();
                Some(1.0 - low as f64 / covered as f64)
            }
        }
        _ if large_plane => streaming_confidence_coverage(input.coverage, input.confidence),
        _ => quality_gate::sharpness_confidence_coverage(input.coverage, input.confidence).ok(),
    };
    match confidence_value {
        Some(value) => {
            confidence.push(input.world_origin, value, "", value >= confidence.threshold)
        }
        None => {
            confidence.miss();
            unmeasurable.push(UnmeasurableRecord {
                criterion: confidence.name.to_string(),
                world: WorldPoint {
                    x: input.world_origin.0,
                    y: input.world_origin.1,
                },
                reason: "quality_gate_insufficient_evidence".to_string(),
            });
        }
    }

    let mut criteria = Vec::new();
    for criterion in [
        local_scale,
        local_ratio,
        effective,
        mtf,
        gradient,
        noise,
        delta,
        boundary,
        confidence,
    ] {
        criteria.push(criterion.finish(&mut unmeasurable));
    }
    let has_failure = criteria
        .iter()
        .any(|criterion| !criterion.diagnostic && !criterion.failed.is_empty());
    let insufficient = criteria.iter().any(|criterion| {
        criterion.measurable_count < quality_gate::QUALITY_MIN_MEASURABLE
            || (criterion.measurable_count + criterion.unmeasurable_count > 0
                && criterion.unmeasurable_count as f64
                    / (criterion.measurable_count + criterion.unmeasurable_count) as f64
                    > quality_gate::QUALITY_MAX_UNMEASURABLE_RATIO)
    });
    report.verdict = if has_failure {
        QualityGateVerdict::Fail
    } else if insufficient {
        QualityGateVerdict::InsufficientEvidence
    } else {
        QualityGateVerdict::Pass
    };
    report.criteria = criteria;
    report.unmeasurable = unmeasurable;
    report
}
