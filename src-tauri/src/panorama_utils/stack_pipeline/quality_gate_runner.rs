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
        let item = FailedMeasurementRecord {
            world: WorldPoint {
                x: world.0,
                y: world.1,
            },
            measured: value,
            owner_path: owner_path.to_string(),
        };
        self.measured.push(item.clone());
        if !pass {
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
            measurable_count: self.measured.len(),
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
            let sx = mapped
                .x
                .clamp(0.0, f64::from(source.width().saturating_sub(1)));
            let sy = mapped
                .y
                .clamp(0.0, f64::from(source.height().saturating_sub(1)));
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
/// still checks ownership and coverage over every candidate ROI; the two
/// criteria that require a complete canvas (effective coverage and boundary
/// tracing) are recorded as unmeasurable below.
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
            for ry in (0..side).step_by(32) {
                for rx in (0..side).step_by(32) {
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

fn sparse_confidence_coverage(coverage: &GrayImage, confidence: &[f32]) -> Option<f64> {
    let (width, height) = coverage.dimensions();
    let step = 16u32;
    let mut covered = 0usize;
    let mut low = 0usize;
    for y in (0..height).step_by(step as usize) {
        for x in (0..width).step_by(step as usize) {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] == 0 {
                continue;
            }
            covered += 1;
            if confidence[index].is_finite() && confidence[index] < 0.05 {
                low += 1;
            }
        }
    }
    (covered > 0).then(|| 1.0 - low as f64 / covered as f64)
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
            let local = quality_gate::local_scale_for_roi(
                roi,
                &source_info.geometry,
                input.residual,
                input.acceptance_render_scale,
            )
            .ok()
            .map(|v| v.median)
            .unwrap_or(1.0);
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
    let effective_result = if large_plane {
        Err("diagnostics_plane_too_large")
    } else {
        quality_gate::effective_pixel_coverage(input.coverage, &quads, input.world_origin)
    };
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
        boundary.miss();
        unmeasurable.push(UnmeasurableRecord {
            criterion: boundary.name.to_string(),
            world: WorldPoint {
                x: input.world_origin.0,
                y: input.world_origin.1,
            },
            reason: "diagnostics_plane_too_large".to_string(),
        });
    }

    let mut confidence = Criterion::new(
        "sharpness_confidence_coverage",
        1.0 - quality_gate::QUALITY_LOW_CONFIDENCE_RATIO_MAX,
    );
    let confidence_value = match input.textured {
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
        _ if large_plane => sparse_confidence_coverage(input.coverage, input.confidence),
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
