//! Runtime adapter for the pure Quality_Gate measurements.
//!
//! `quality_gate.rs` deliberately contains no image loading or report policy.
//! This module supplies that small amount of orchestration: it selects ROIs
//! from the final owner planes, decodes one owner source at a time, samples the
//! corresponding reference ROI, and records all nine criteria.  The caller can
//! therefore use it in observation-only mode without changing export pixels.

use std::collections::BTreeMap;
use std::time::Instant;

use image::{GrayImage, ImageBuffer, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Vector3};

use super::degradation;
use super::quality_gate::{self, EffectivePixelCoverage, QualityRoi, SourceGeometry};
use super::report::{
    FailedMeasurementRecord, QualityGateCriterionRecord, QualityGateReport,
    QualityGateTimingRecord, QualityGateVerdict, QualityStationStatistic, RoiPhotometryRecord,
    UnmeasurableCategory, UnmeasurableRecord, WorldPoint,
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
/// `textured` carries the cell-level Textured_Pixel evidence. An absent plane
/// is reported as insufficient evidence instead of widening the statistic to
/// all covered pixels.
pub(crate) struct QualityGateInput<'a> {
    pub output: &'a Rgb32FImage,
    pub coverage: &'a GrayImage,
    pub ownership: &'a [u16],
    pub confidence: &'a [f32],
    pub textured: Option<&'a [u8]>,
    /// Quantised owner shortfall in hundredths (0..=100) for each output
    /// pixel. Keeping this as bytes avoids another full-size f32 plane.
    pub owner_shortfall: Option<&'a [u8]>,
    /// Quantised ownership disagreement in hundredths (0..=100).
    pub owner_disagreement: Option<&'a [u8]>,
    /// Covered output pixels for which the source reverse lookup could not
    /// establish an owner. These are technical evidence gaps.
    pub unresolved_pixel_count: u64,
    pub world_origin: (f64, f64),
    pub sources: &'a [QualitySource],
    pub residual: &'a ResidualWarp,
    pub acceptance_render_scale: f64,
    pub final_sharpen_amount: f64,
}

pub(crate) type SourceLoader<'a> = dyn FnMut(&QualitySource) -> Result<Rgb32FImage, String> + 'a;

pub(crate) struct Criterion {
    name: &'static str,
    threshold: f64,
    min: Option<f64>,
    max: Option<f64>,
    measured: Vec<FailedMeasurementRecord>,
    failed: Vec<FailedMeasurementRecord>,
    unmeasurable: usize,
    content_not_applicable: usize,
    technical_unmeasurable: usize,
    unmeasurable_reasons: BTreeMap<String, usize>,
    pub(super) diagnostic: bool,
    observed_count: usize,
    /// Keep reports bounded when a large canvas has many boundary samples.
    /// The count remains exact while only a deterministic prefix is retained
    /// for the per-point diagnostics.
    record_limit: usize,
    textured_pixel_count: Option<u64>,
    excluded_flat_pixel_count: Option<u64>,
    textured_low_confidence_ratio: Option<f64>,
    flat_low_confidence_ratio: Option<f64>,
    textured_shortfall_ratio: Option<f64>,
    flat_shortfall_ratio: Option<f64>,
    shortfall_histogram: Vec<u64>,
    disagreement_veto_count: u64,
    unresolved_pixel_count: u64,
}

#[derive(Debug, Clone, Copy)]
struct ConfidenceCoverageStats {
    textured_pixels: u64,
    flat_pixels: u64,
    textured_low: u64,
    flat_low: u64,
    textured_unknown: u64,
    flat_unknown: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct OwnerSharpnessStats {
    pub textured_pixels: u64,
    pub flat_pixels: u64,
    pub textured_shortfall: u64,
    pub flat_shortfall: u64,
    pub shortfall_histogram: Vec<u64>,
    pub disagreement_veto_count: u64,
    pub unresolved_pixel_count: u64,
}

/// Marker of a covered output pixel without owner evidence in the quantised
/// owner planes.
pub(crate) const OWNER_EVIDENCE_UNKNOWN: u8 = u8::MAX;

/// Quantise a `[0, 1]` owner-evidence value to hundredths.  Rounding up keeps
/// every value strictly above a hundredth-aligned limit strictly above the
/// quantised limit, so `quantize_hundredths(v) > hundredths(limit)` is exactly
/// `v > limit`.  A non-finite value is evidence the lookup could not use.
pub(crate) fn quantize_hundredths(value: f32) -> u8 {
    if !value.is_finite() {
        return OWNER_EVIDENCE_UNKNOWN;
    }
    (value * 100.0).ceil().clamp(0.0, 100.0) as u8
}

/// A hundredth-aligned threshold in the units of [`quantize_hundredths`].
fn hundredths(limit: f64) -> u8 {
    (limit * 100.0).round().clamp(0.0, 100.0) as u8
}

/// Owner shortfall limit of 需求 11.12 in quantised units.
pub(crate) fn owner_shortfall_limit() -> u8 {
    hundredths(f64::from(quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX))
}

/// Disagreement above which the Focus_Fuser's mismatch veto may have kept a
/// less sharp owner, in quantised units.
pub(crate) fn owner_disagreement_veto_limit() -> u8 {
    hundredths(super::focus_fuser::OWNERSHIP_DISAGREEMENT_VETO)
}

/// Runtime owner-sharpness statistic used by the Quality_Gate and P71. The
/// quantised planes are produced during the output->station reverse lookup;
/// this function only counts them and never infers ownership.  The shortfall
/// histogram and the veto count describe the Textured_Pixel that exceed the
/// limit, which is the population the criterion judges.
pub(crate) fn owner_sharpness_stats(
    coverage: &[u8],
    textured: &[u8],
    shortfall: &[u8],
    disagreement: &[u8],
    unresolved_pixel_count: u64,
) -> Result<OwnerSharpnessStats, &'static str> {
    if coverage.len() != textured.len()
        || coverage.len() != shortfall.len()
        || coverage.len() != disagreement.len()
    {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    let shortfall_limit = owner_shortfall_limit();
    let veto_limit = owner_disagreement_veto_limit();
    let mut stats = OwnerSharpnessStats {
        shortfall_histogram: vec![0; 20],
        unresolved_pixel_count,
        ..OwnerSharpnessStats::default()
    };
    for (((&covered, &is_textured), &shortfall), &disagreement) in coverage
        .iter()
        .zip(textured)
        .zip(shortfall)
        .zip(disagreement)
    {
        if covered == 0 || shortfall == OWNER_EVIDENCE_UNKNOWN {
            continue;
        }
        let shortfall_exceeds = shortfall > shortfall_limit;
        if is_textured == 0 {
            stats.flat_pixels += 1;
            stats.flat_shortfall += u64::from(shortfall_exceeds);
            continue;
        }
        stats.textured_pixels += 1;
        if shortfall_exceeds {
            stats.textured_shortfall += 1;
            let bucket = usize::from(shortfall.min(100)) * 20 / 101;
            stats.shortfall_histogram[bucket.min(19)] += 1;
            if disagreement != OWNER_EVIDENCE_UNKNOWN && disagreement > veto_limit {
                stats.disagreement_veto_count += 1;
            }
        }
    }
    Ok(stats)
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
            content_not_applicable: 0,
            technical_unmeasurable: 0,
            unmeasurable_reasons: BTreeMap::new(),
            diagnostic: false,
            observed_count: 0,
            record_limit: usize::MAX,
            textured_pixel_count: None,
            excluded_flat_pixel_count: None,
            textured_low_confidence_ratio: None,
            flat_low_confidence_ratio: None,
            textured_shortfall_ratio: None,
            flat_shortfall_ratio: None,
            shortfall_histogram: Vec::new(),
            disagreement_veto_count: 0,
            unresolved_pixel_count: 0,
        }
    }
}

impl Criterion {
    pub(super) fn new(name: &'static str, threshold: f64) -> Self {
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

    pub(super) fn push(&mut self, world: (f64, f64), value: f64, owner_path: &str, pass: bool) {
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

    fn miss(&mut self, reason: &'static str) {
        self.miss_count(reason, 1);
    }

    fn miss_count(&mut self, reason: &'static str, count: u64) {
        let count = usize::try_from(count).unwrap_or(usize::MAX);
        self.unmeasurable = self.unmeasurable.saturating_add(count);
        match classify_unmeasurable_reason(reason) {
            UnmeasurableCategory::ContentNotApplicable => {
                self.content_not_applicable = self.content_not_applicable.saturating_add(count)
            }
            UnmeasurableCategory::Technical => {
                self.technical_unmeasurable = self.technical_unmeasurable.saturating_add(count)
            }
        }
        *self
            .unmeasurable_reasons
            .entry(reason.to_string())
            .or_default() += count;
    }

    pub(super) fn finish(
        self,
        _unmeasurable: &mut Vec<UnmeasurableRecord>,
    ) -> QualityGateCriterionRecord {
        // `unmeasurable` records are appended by the caller while processing
        // each ROI; this method only carries the aggregate count.
        let verdict = criterion_verdict(
            self.name,
            self.observed_count,
            self.technical_unmeasurable,
            !self.failed.is_empty(),
            self.diagnostic,
            self.observed_count == 0
                || (self.name == "owner_sharpness_coverage" && self.unresolved_pixel_count > 0),
        );
        QualityGateCriterionRecord {
            name: self.name.to_string(),
            threshold: self.threshold,
            threshold_min: self.min,
            threshold_max: self.max,
            measurable_count: self.observed_count,
            unmeasurable_count: self.unmeasurable,
            content_not_applicable_count: self.content_not_applicable,
            technical_unmeasurable_count: self.technical_unmeasurable,
            verdict,
            station_statistics: Vec::new(),
            unmeasurable_reasons: self.unmeasurable_reasons,
            measured: self.measured,
            diagnostic: self.diagnostic,
            failed: self.failed,
            textured_pixel_count: self.textured_pixel_count,
            excluded_flat_pixel_count: self.excluded_flat_pixel_count,
            textured_low_confidence_ratio: self.textured_low_confidence_ratio,
            flat_low_confidence_ratio: self.flat_low_confidence_ratio,
            textured_shortfall_ratio: self.textured_shortfall_ratio,
            flat_shortfall_ratio: self.flat_shortfall_ratio,
            shortfall_histogram: self.shortfall_histogram,
            disagreement_veto_count: self.disagreement_veto_count,
            unresolved_pixel_count: self.unresolved_pixel_count,
        }
    }
}

pub(crate) fn criterion_verdict(
    name: &str,
    measurable: usize,
    technical: usize,
    has_failed: bool,
    diagnostic: bool,
    statistics_base_empty: bool,
) -> QualityGateVerdict {
    // Diagnostic criteria describe an unavailable measurement path. They do
    // not establish a pass or fail result, even when a fallback value happens
    // to satisfy the nominal threshold.
    if diagnostic {
        return QualityGateVerdict::NotApplicable;
    }
    let technical_ratio = if measurable + technical == 0 {
        0.0
    } else {
        technical as f64 / (measurable + technical) as f64
    };
    let sparse = matches!(
        name,
        "mtf50_normalized" | "noise_sigma_ratio" | "boundary_stroke_alignment"
    );
    let whole_image = matches!(name, "effective_pixel_count" | "owner_sharpness_coverage");
    let dense = matches!(
        name,
        "local_scale_median"
            | "local_scale_pixel_ratio"
            | "gradient_energy_normalized"
            | "roi_delta_e00"
    );
    if (whole_image && statistics_base_empty)
        || (!whole_image && technical_ratio > quality_gate::QUALITY_MAX_UNMEASURABLE_RATIO)
    {
        QualityGateVerdict::InsufficientEvidence
    } else if has_failed {
        QualityGateVerdict::Fail
    } else if sparse && measurable < quality_gate::QUALITY_MIN_MEASURABLE {
        QualityGateVerdict::NotApplicable
    } else if dense && measurable < quality_gate::QUALITY_MIN_MEASURABLE {
        QualityGateVerdict::InsufficientEvidence
    } else {
        QualityGateVerdict::Pass
    }
}

pub(crate) fn overall_verdict(criteria: &[QualityGateCriterionRecord]) -> QualityGateVerdict {
    if criteria
        .iter()
        .any(|c| c.verdict == QualityGateVerdict::Fail)
    {
        QualityGateVerdict::Fail
    } else if criteria.is_empty()
        || criteria.iter().any(|c| {
            !matches!(
                c.verdict,
                QualityGateVerdict::Pass | QualityGateVerdict::NotApplicable
            )
        })
    {
        QualityGateVerdict::InsufficientEvidence
    } else {
        QualityGateVerdict::Pass
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
    let inverse = (geometry.tile_to_world * geometry.member_to_anchor)
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let mut result = Rgb32FImage::new(roi.side, roi.side);
    for y in 0..roi.side {
        for x in 0..roi.side {
            let world = Point2::new(
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            );
            let mapped = quality_gate::map_world_to_source_with_inverse(
                world,
                geometry.station_id,
                residual,
                &inverse,
            )?;
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

pub(super) fn record_unmeasurable(
    criterion: &mut Criterion,
    records: &mut Vec<UnmeasurableRecord>,
    roi: &QualityRoi,
    reason: &'static str,
) {
    let category = classify_unmeasurable_reason(reason);
    criterion.miss(reason);
    records.push(UnmeasurableRecord {
        criterion: criterion.name.to_string(),
        world: WorldPoint {
            x: roi.world_origin.0,
            y: roi.world_origin.1,
        },
        reason: reason.to_string(),
        category,
    });
}

pub(crate) fn classify_unmeasurable_reason(reason: &str) -> UnmeasurableCategory {
    match reason {
        degradation::ROI_NOT_SLANTED_EDGE
        | degradation::SLANTED_EDGE_ANGLE_OUT_OF_RANGE
        | degradation::SLANTED_EDGE_CONTRAST_INSUFFICIENT
        | degradation::SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED
        | degradation::SLANTED_EDGE_TOO_SHORT
        | degradation::ROI_NOT_FLAT
        | degradation::BOUNDARY_NO_PAIRABLE_EDGE
        | degradation::BOUNDARY_LOW_CONTRAST
        | degradation::BOUNDARY_ORIENTATION_MISMATCH => UnmeasurableCategory::ContentNotApplicable,
        _ => UnmeasurableCategory::Technical,
    }
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

fn confidence_coverage_stats(
    coverage: &GrayImage,
    confidence: &[f32],
    textured: &[u8],
) -> Option<ConfidenceCoverageStats> {
    let (width, height) = coverage.dimensions();
    if textured.len() != confidence.len() || confidence.len() != coverage.as_raw().len() {
        return None;
    }
    let mut stats = ConfidenceCoverageStats {
        textured_pixels: 0,
        flat_pixels: 0,
        textured_low: 0,
        flat_low: 0,
        textured_unknown: 0,
        flat_unknown: 0,
    };
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] == 0 {
                continue;
            }
            let is_textured = textured[index] != 0;
            if is_textured {
                stats.textured_pixels = stats.textured_pixels.saturating_add(1);
            } else {
                stats.flat_pixels = stats.flat_pixels.saturating_add(1);
            }
            if !confidence[index].is_finite() {
                if is_textured {
                    stats.textured_unknown = stats.textured_unknown.saturating_add(1);
                } else {
                    stats.flat_unknown = stats.flat_unknown.saturating_add(1);
                }
            } else if confidence[index] < 0.05 {
                if is_textured {
                    stats.textured_low = stats.textured_low.saturating_add(1);
                } else {
                    stats.flat_low = stats.flat_low.saturating_add(1);
                }
            }
        }
    }
    Some(stats)
}

/// Accumulate one output pixel while the compositor maps it back to its
/// ownership cell. Row-major calls preserve the diagnostic sum order without
/// retaining three additional full-resolution score/candidate planes.
pub(crate) fn accumulate_confidence_score(
    result: &mut super::report::ConfidenceScoreDistribution,
    covered: u8,
    confidence: f32,
    textured: u8,
    evidence: super::focus_fuser::SharpnessCellEvidence,
) {
    if covered == 0 || textured == 0 || !confidence.is_finite() || confidence >= 0.05 {
        return;
    }
    result.textured_low_count += 1;
    result.winner_sum += f64::from(evidence.winner_score);
    result.runner_up_sum += f64::from(evidence.runner_up_score);
    let wb = (f64::from(evidence.winner_score).clamp(0.0, 1.0) * 20.0).floor() as usize;
    let rb = (f64::from(evidence.runner_up_score).clamp(0.0, 1.0) * 20.0).floor() as usize;
    result.winner_histogram.resize(21, 0);
    result.runner_up_histogram.resize(21, 0);
    result.winner_histogram[wb.min(20)] += 1;
    result.runner_up_histogram[rb.min(20)] += 1;
    if evidence.winner_score < 0.10 {
        result.winner_below_0_10 += 1;
    } else if evidence.winner_score >= 0.10 && evidence.runner_up_score >= 0.10 {
        result.both_scores_at_least_0_10 += 1;
    } else if evidence.candidate_count <= 1 {
        result.single_candidate_count += 1;
    }
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
fn boundary_point_error(
    image: &Rgb32FImage,
    x: u32,
    y: u32,
    normal: (f64, f64),
) -> Result<f64, &'static str> {
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
            return Err(degradation::BOUNDARY_LOW_CONTRAST);
        }
        sides[side_index] = (position_sum / contrast_sum, orientation_sum / contrast_sum);
    }
    let mut angle_error = (sides[0].1 - sides[1].1).abs() % 180.0;
    if angle_error > 90.0 {
        angle_error = 180.0 - angle_error;
    }
    if angle_error > 10.0 {
        Err(degradation::BOUNDARY_ORIENTATION_MISMATCH)
    } else {
        Ok((sides[0].0 + sides[1].0).abs())
    }
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
    let unmeasurable_before = unmeasurable.len();
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
        let error = match boundary_point_error(image, x, y, normal) {
            Ok(error) => error,
            Err(reason) => {
                unmeasurable.push(UnmeasurableRecord {
                    criterion: criterion.name.to_string(),
                    world: WorldPoint {
                        x: world.0,
                        y: world.1,
                    },
                    reason: reason.to_string(),

                    category: classify_unmeasurable_reason(reason),
                });
                criterion.miss(reason);
                return;
            }
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
        if unmeasurable.len() == unmeasurable_before {
            criterion.miss(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
            unmeasurable.push(UnmeasurableRecord {
                criterion: criterion.name.to_string(),
                world: WorldPoint {
                    x: origin.0,
                    y: origin.1,
                },
                reason: degradation::BOUNDARY_NO_PAIRABLE_EDGE.to_string(),

                category: classify_unmeasurable_reason(degradation::BOUNDARY_NO_PAIRABLE_EDGE),
            });
        }
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

/// Build the owner-sharpness criterion from the exact planes retained by the
/// compositor.  Keeping this construction here makes the runtime report and
/// Property 71 exercise the same statistic, threshold, and evidence policy.
pub(crate) struct OwnerSharpnessCriterionInput<'a> {
    pub(crate) coverage: &'a GrayImage,
    pub(crate) confidence: &'a [f32],
    pub(crate) textured: Option<&'a [u8]>,
    pub(crate) shortfall: Option<&'a [u8]>,
    pub(crate) disagreement: Option<&'a [u8]>,
    pub(crate) unresolved_pixel_count: u64,
    pub(crate) world_origin: (f64, f64),
    pub(crate) unmeasurable: &'a mut Vec<UnmeasurableRecord>,
}

pub(crate) fn build_owner_sharpness_criterion(
    input: OwnerSharpnessCriterionInput<'_>,
) -> Criterion {
    let mut criterion = Criterion::new(
        "owner_sharpness_coverage",
        quality_gate::OWNER_SHARPNESS_COVERAGE_MIN,
    );
    let confidence_stats = input
        .textured
        .and_then(|textured| confidence_coverage_stats(input.coverage, input.confidence, textured));
    let owner_stats = match (input.textured, input.shortfall, input.disagreement) {
        (Some(textured), Some(shortfall), Some(disagreement)) => owner_sharpness_stats(
            input.coverage.as_raw(),
            textured,
            shortfall,
            disagreement,
            input.unresolved_pixel_count,
        )
        .ok(),
        _ => None,
    };
    match (confidence_stats, owner_stats) {
        (Some(confidence_stats), Some(stats)) => {
            criterion.textured_pixel_count = Some(stats.textured_pixels);
            criterion.excluded_flat_pixel_count = Some(stats.flat_pixels);
            criterion.textured_shortfall_ratio = (stats.textured_pixels > 0)
                .then_some(stats.textured_shortfall as f64 / stats.textured_pixels as f64);
            criterion.flat_shortfall_ratio = (stats.flat_pixels > 0)
                .then_some(stats.flat_shortfall as f64 / stats.flat_pixels as f64);
            criterion.shortfall_histogram = stats.shortfall_histogram.clone();
            criterion.disagreement_veto_count = stats.disagreement_veto_count;
            criterion.unresolved_pixel_count = stats.unresolved_pixel_count;
            criterion.textured_low_confidence_ratio = (confidence_stats.textured_pixels > 0
                && confidence_stats.textured_unknown == 0)
                .then_some(
                    confidence_stats.textured_low as f64 / confidence_stats.textured_pixels as f64,
                );
            criterion.flat_low_confidence_ratio = (confidence_stats.flat_pixels > 0
                && confidence_stats.flat_unknown == 0)
                .then_some(confidence_stats.flat_low as f64 / confidence_stats.flat_pixels as f64);
            if stats.unresolved_pixel_count > 0 {
                criterion.miss_count(
                    degradation::OWNER_REVERSE_LOOKUP_UNRESOLVED,
                    stats.unresolved_pixel_count,
                );
                input.unmeasurable.push(UnmeasurableRecord {
                    criterion: criterion.name.to_string(),
                    world: WorldPoint {
                        x: input.world_origin.0,
                        y: input.world_origin.1,
                    },
                    reason: degradation::OWNER_REVERSE_LOOKUP_UNRESOLVED.to_string(),
                    category: UnmeasurableCategory::Technical,
                });
            }
            if stats.textured_pixels == 0 {
                criterion.miss(degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE);
                input.unmeasurable.push(UnmeasurableRecord {
                    criterion: criterion.name.to_string(),
                    world: WorldPoint {
                        x: input.world_origin.0,
                        y: input.world_origin.1,
                    },
                    reason: degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE.to_string(),
                    category: classify_unmeasurable_reason(
                        degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE,
                    ),
                });
            } else {
                let value = 1.0 - stats.textured_shortfall as f64 / stats.textured_pixels as f64;
                criterion.push(input.world_origin, value, "", value >= criterion.threshold);
            }
        }
        _ => {
            criterion.miss(degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE);
            input.unmeasurable.push(UnmeasurableRecord {
                criterion: criterion.name.to_string(),
                world: WorldPoint {
                    x: input.world_origin.0,
                    y: input.world_origin.1,
                },
                reason: degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE.to_string(),
                category: classify_unmeasurable_reason(
                    degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE,
                ),
            });
        }
    }
    criterion
}

/// Execute all nine criteria in record-only mode. No decision from the return
/// value is applied to export; callers decide whether to block in a later task.
pub(crate) fn run_quality_gate(
    input: &QualityGateInput<'_>,
    load_source: &mut SourceLoader<'_>,
) -> QualityGateReport {
    let mut report = QualityGateReport::default();
    let mut timing = QualityGateTimingRecord::default();
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

            category: classify_unmeasurable_reason("diagnostics_roi_invalid"),
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

                    category: classify_unmeasurable_reason(reason),
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

                    category: classify_unmeasurable_reason(reason),
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
        let decode_started = Instant::now();
        let source = match load_source(source_info) {
            Ok(value) => value,
            Err(_) => {
                timing.decode_seconds += decode_started.elapsed().as_secs_f64();
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
        timing.decode_seconds += decode_started.elapsed().as_secs_f64();
        timing.owner_source_decode_count = timing.owner_source_decode_count.saturating_add(1);
        for roi in owner_rois {
            timing.roi_reference_count = timing.roi_reference_count.saturating_add(1);
            let roi_started = Instant::now();
            let where_ = world(roi);
            let output_roi = output_roi(input.output, roi);
            let reference = source_roi(&source, roi, &source_info.geometry, input.residual);
            let (Some(output_roi), Ok(reference)) = (output_roi, reference) else {
                timing.roi_sampling_seconds += roi_started.elapsed().as_secs_f64();
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
            timing.reference_resampling_seconds += roi_started.elapsed().as_secs_f64();
            if let (Some(output_mean_rgb), Some(reference_mean_rgb)) = (
                quality_gate::roi_mean_rgb(&output_roi),
                quality_gate::roi_mean_rgb(&reference),
            ) {
                report.roi_photometry.push(RoiPhotometryRecord {
                    world: WorldPoint {
                        x: where_.0,
                        y: where_.1,
                    },
                    station_id: source_info.geometry.station_id,
                    owner_path: source_info.path.clone(),
                    output_mean_rgb,
                    reference_mean_rgb,
                    channel_ratio: std::array::from_fn(|c| {
                        (reference_mean_rgb[c] != 0.0).then_some(
                            f64::from(output_mean_rgb[c]) / f64::from(reference_mean_rgb[c]),
                        )
                    }),
                });
            }
            let scale_started = Instant::now();
            let scale = quality_gate::local_scale_for_roi(
                roi,
                &source_info.geometry,
                input.residual,
                input.acceptance_render_scale,
            );
            let local = scale.as_ref().ok().map(|value| value.median);
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
            timing.local_scale_seconds += scale_started.elapsed().as_secs_f64();
            if let Some(local) = local {
                let mtf_started = Instant::now();
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
                            record_unmeasurable(
                                &mut mtf,
                                &mut unmeasurable,
                                roi,
                                "mtf_reference_zero",
                            );
                        }
                    }
                    (Err(reason), _) | (_, Err(reason)) => {
                        record_unmeasurable(&mut mtf, &mut unmeasurable, roi, reason)
                    }
                }
                timing.slanted_edge_mtf_seconds += mtf_started.elapsed().as_secs_f64();
                let gradient_started = Instant::now();
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
                timing.gradient_seconds += gradient_started.elapsed().as_secs_f64();
            } else {
                // MTF50 and gradient energy are scale-normalised quantities;
                // substituting 1.0 would turn missing geometry into evidence.
                let reason = degradation::LOCAL_SCALE_UNMEASURABLE;
                record_unmeasurable(&mut mtf, &mut unmeasurable, roi, reason);
                record_unmeasurable(&mut gradient, &mut unmeasurable, roi, reason);
            }
            let noise_started = Instant::now();
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
            timing.noise_seconds += noise_started.elapsed().as_secs_f64();
            let delta_started = Instant::now();
            match quality_gate::roi_low_frequency_delta_e00(&output_roi, &reference) {
                Ok(value) => delta.push(
                    where_,
                    value,
                    &source_info.path,
                    value <= quality_gate::QUALITY_ROI_DELTA_E00_MAX,
                ),
                Err(reason) => record_unmeasurable(&mut delta, &mut unmeasurable, roi, reason),
            }
            timing.delta_e_seconds += delta_started.elapsed().as_secs_f64();
            timing.roi_sampling_seconds += roi_started.elapsed().as_secs_f64();
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
    let effective_started = Instant::now();
    let effective_result =
        effective_pixel_coverage_streaming(input.coverage, &quads, input.world_origin);
    timing.effective_coverage_seconds = effective_started.elapsed().as_secs_f64();
    match effective_result {
        Ok(EffectivePixelCoverage { ratio, .. }) => effective.push(
            input.world_origin,
            ratio,
            "",
            ratio >= quality_gate::QUALITY_EFFECTIVE_PIXEL_RATIO_MIN,
        ),
        Err(reason) => {
            effective.miss(reason);
            unmeasurable.push(UnmeasurableRecord {
                criterion: effective.name.to_string(),
                world: WorldPoint {
                    x: input.world_origin.0,
                    y: input.world_origin.1,
                },
                reason: reason.to_string(),

                category: classify_unmeasurable_reason(reason),
            });
        }
    }

    let mut boundary = Criterion::p95_with_max(
        "boundary_stroke_alignment",
        quality_gate::QUALITY_BOUNDARY_P95_MAX,
        quality_gate::QUALITY_BOUNDARY_MAX,
    );
    boundary.record_limit = 2_048;
    let boundary_started = Instant::now();
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
                        for (point, reason) in &measurement.unmeasurable {
                            boundary.miss(reason);
                            unmeasurable.push(UnmeasurableRecord {
                                criterion: boundary.name.to_string(),
                                world: WorldPoint {
                                    x: point.0,
                                    y: point.1,
                                },
                                reason: (*reason).to_string(),

                                category: classify_unmeasurable_reason(reason),
                            });
                        }
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
                        boundary.miss(reason);
                        unmeasurable.push(UnmeasurableRecord {
                            criterion: boundary.name.to_string(),
                            world: WorldPoint {
                                x: input.world_origin.0,
                                y: input.world_origin.1,
                            },
                            reason: reason.to_string(),

                            category: classify_unmeasurable_reason(reason),
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
    timing.boundary_seconds = boundary_started.elapsed().as_secs_f64();
    timing.boundary_sample_count = boundary.observed_count + boundary.unmeasurable;

    let owner_sharpness = build_owner_sharpness_criterion(OwnerSharpnessCriterionInput {
        coverage: input.coverage,
        confidence: input.confidence,
        textured: input.textured,
        shortfall: input.owner_shortfall,
        disagreement: input.owner_disagreement,
        unresolved_pixel_count: input.unresolved_pixel_count,
        world_origin: input.world_origin,
        unmeasurable: &mut unmeasurable,
    });

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
        owner_sharpness,
    ] {
        criteria.push(criterion.finish(&mut unmeasurable));
    }
    let stations: std::collections::BTreeSet<_> = input
        .sources
        .iter()
        .map(|s| s.geometry.station_id)
        .collect();
    let shortfall_limit = owner_shortfall_limit();
    for station_id in stations {
        let owners: std::collections::BTreeSet<_> = input
            .sources
            .iter()
            .filter(|s| s.geometry.station_id == station_id)
            .map(|s| s.owner)
            .collect();
        let station_quads: Vec<_> = input
            .sources
            .iter()
            .filter(|s| s.geometry.station_id == station_id)
            .filter_map(source_quad)
            .collect();
        let covered = input
            .ownership
            .iter()
            .zip(input.coverage.as_raw())
            .filter(|(o, c)| **c != 0 && owners.contains(o))
            .count() as u64;
        let denominator =
            effective_pixel_coverage_streaming(input.coverage, &station_quads, input.world_origin)
                .ok()
                .map(|r| r.projected_union)
                .unwrap_or(0);
        criteria[2]
            .station_statistics
            .push(QualityStationStatistic {
                station_id,
                numerator: covered,
                denominator,
                value: (denominator > 0).then_some(covered as f64 / denominator as f64),
                ..Default::default()
            });
        if let (Some(textured), Some(shortfall)) = (input.textured, input.owner_shortfall) {
            let mut textured_count = 0u64;
            let mut flat_count = 0u64;
            let mut textured_low = 0u64;
            let mut flat_low = 0u64;
            let mut textured_shortfall = 0u64;
            let mut flat_shortfall = 0u64;
            for (index, owner) in input.ownership.iter().enumerate() {
                if input.coverage.as_raw()[index] == 0 || !owners.contains(owner) {
                    continue;
                }
                if shortfall[index] == OWNER_EVIDENCE_UNKNOWN {
                    continue;
                }
                let low = !input.confidence[index].is_finite() || input.confidence[index] < 0.05;
                let exceeds = shortfall[index] > shortfall_limit;
                if textured[index] != 0 {
                    textured_count += 1;
                    textured_low += u64::from(low);
                    textured_shortfall += u64::from(exceeds);
                } else {
                    flat_count += 1;
                    flat_low += u64::from(low);
                    flat_shortfall += u64::from(exceeds);
                }
            }
            if let Some(owner_criterion) = criteria
                .iter_mut()
                .find(|criterion| criterion.name == "owner_sharpness_coverage")
            {
                owner_criterion
                    .station_statistics
                    .push(QualityStationStatistic {
                        station_id,
                        numerator: textured_count.saturating_sub(textured_shortfall),
                        denominator: textured_count,
                        value: (textured_count > 0).then_some(
                            1.0 - textured_shortfall as f64 / textured_count.max(1) as f64,
                        ),
                        textured_pixel_count: textured_count,
                        excluded_flat_pixel_count: flat_count,
                        textured_low_confidence_count: textured_low,
                        flat_low_confidence_count: flat_low,
                        textured_low_confidence_ratio: (textured_count > 0)
                            .then_some(textured_low as f64 / textured_count as f64),
                        flat_low_confidence_ratio: (flat_count > 0)
                            .then_some(flat_low as f64 / flat_count as f64),
                        textured_shortfall_count: textured_shortfall,
                        flat_shortfall_count: flat_shortfall,
                        textured_shortfall_ratio: (textured_count > 0)
                            .then_some(textured_shortfall as f64 / textured_count as f64),
                        flat_shortfall_ratio: (flat_count > 0)
                            .then_some(flat_shortfall as f64 / flat_count as f64),
                    });
            }
        }
    }
    report.verdict = overall_verdict(&criteria);
    report.criteria = criteria;
    report.unmeasurable = unmeasurable;
    report.timing = timing;
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    /// 需求 11.12 judges `shortfall > 0.05` on quantised planes; the
    /// quantisation must keep that boundary exact, and the cell definition
    /// must give no shortfall to the best owner or to a lone candidate.
    #[test]
    fn owner_shortfall_quantisation_keeps_the_criterion_boundary() {
        let limit = owner_shortfall_limit();
        assert_eq!(limit, 5);
        assert_eq!(owner_disagreement_veto_limit(), 20);
        assert_eq!(quantize_hundredths(0.05), limit);
        assert!(quantize_hundredths(0.0501) > limit);
        assert!(quantize_hundredths(0.049) <= limit);
        assert_eq!(quantize_hundredths(0.0), 0);
        assert_eq!(quantize_hundredths(1.0), 100);
        assert_eq!(quantize_hundredths(f32::NAN), OWNER_EVIDENCE_UNKNOWN);
        assert_eq!(quantize_hundredths(f32::INFINITY), OWNER_EVIDENCE_UNKNOWN);

        let best_owner = super::super::focus_fuser::SharpnessCellEvidence {
            winner_score: 0.60,
            runner_up_score: 0.50,
            owner_score: 0.60,
            candidate_count: 2,
            disagreement: 0.0,
        };
        assert_eq!(best_owner.owner_shortfall(), 0.0);
        let lone_candidate = super::super::focus_fuser::SharpnessCellEvidence {
            owner_score: 0.30,
            candidate_count: 1,
            ..best_owner
        };
        assert_eq!(lone_candidate.owner_shortfall(), 0.0);
        let weaker_owner = super::super::focus_fuser::SharpnessCellEvidence {
            owner_score: 0.54,
            ..best_owner
        };
        assert!((weaker_owner.owner_shortfall() - 0.10).abs() < 1.0e-6);
        assert!(quantize_hundredths(weaker_owner.owner_shortfall()) > limit);
    }

    #[test]
    fn confidence_distribution_accumulates_compact_cell_evidence() {
        let mut result = super::super::report::ConfidenceScoreDistribution {
            winner_histogram: vec![0; 21],
            runner_up_histogram: vec![0; 21],
            ..Default::default()
        };
        let evidence = super::super::focus_fuser::SharpnessCellEvidence {
            winner_score: 0.60,
            runner_up_score: 0.59,
            owner_score: 0.60,
            candidate_count: 2,
            disagreement: 0.0,
        };
        accumulate_confidence_score(&mut result, 255, 0.01, 255, evidence);
        accumulate_confidence_score(&mut result, 255, 0.05, 255, evidence);
        accumulate_confidence_score(&mut result, 255, 0.01, 0, evidence);
        assert_eq!(result.textured_low_count, 1);
        assert_eq!(result.both_scores_at_least_0_10, 1);
        assert_eq!(result.winner_below_0_10, 0);
        assert_eq!(result.winner_histogram[12], 1);
        assert_eq!(result.runner_up_histogram[11], 1);
        assert!((result.winner_sum - 0.60).abs() < 1e-6);
        assert!((result.runner_up_sum - 0.59).abs() < 1e-6);
    }

    #[test]
    fn textured_partition_excludes_transparency_and_keeps_the_fixed_confidence_boundary() {
        let coverage = GrayImage::from_raw(6, 1, vec![255, 255, 255, 255, 0, 0]).unwrap();
        let confidence = [0.049, 0.05, 0.0, 0.8, 0.0, 0.0];
        let textured = [255, 255, 0, 0, 255, 0];
        let stats = confidence_coverage_stats(&coverage, &confidence, &textured).unwrap();
        assert_eq!(stats.textured_pixels, 2);
        assert_eq!(stats.flat_pixels, 2);
        assert_eq!(stats.textured_low, 1, "0.05 itself is not low confidence");
        assert_eq!(stats.flat_low, 1);
        assert_eq!(stats.textured_unknown, 0);
        assert_eq!(stats.flat_unknown, 0);
    }

    #[test]
    fn confidence_stats_report_all_flat_pixels_and_exclude_nan_from_ratios() {
        let coverage = GrayImage::from_raw(5, 1, vec![255; 5]).unwrap();
        let confidence = [f32::NAN, 0.02, f32::INFINITY, 0.049, 0.05];
        let textured = [0, 0, 0, 0, 0];
        let stats = confidence_coverage_stats(&coverage, &confidence, &textured).unwrap();
        assert_eq!(stats.textured_pixels, 0);
        assert_eq!(stats.flat_pixels, 5);
        assert_eq!(stats.flat_low, 2);
        assert_eq!(stats.flat_unknown, 2);
        assert_eq!(stats.textured_unknown, 0);

        let mixed =
            confidence_coverage_stats(&coverage, &confidence, &[255, 255, 0, 0, 0]).unwrap();
        assert_eq!(mixed.textured_pixels, 2);
        assert_eq!(mixed.textured_low, 1);
        assert_eq!(mixed.textured_unknown, 1);
    }

    #[test]
    fn texture_evidence_uses_all_covered_candidates_and_the_inclusive_sharpness_floor() {
        use super::super::focus_fuser::{CandidateCells, StationFusion};

        let mut fusion = StationFusion::new(24, 8, 2);
        assert_eq!(fusion.geometry().cell_count(), 3);
        fusion.seed(0, &[0.099, 0.10, 0.0], &[true, true, true]);
        assert_eq!(fusion.textured_cells(), vec![false, true, false]);
        let mut candidate = CandidateCells::new(3);
        candidate.sharpness = vec![0.10, 0.05, 0.9];
        candidate.covered = vec![true, true, false];
        fusion.fold(1, &candidate);
        assert_eq!(fusion.textured_cells(), vec![true, true, false]);
    }

    #[test]
    fn missing_local_scale_never_supplies_unit_scale_to_mtf_or_gradient() {
        let output = Rgb32FImage::from_pixel(1024, 1024, Rgb([0.5; 3]));
        let coverage = GrayImage::from_pixel(1024, 1024, Luma([255]));
        let ownership = vec![1; 1024 * 1024];
        let confidence = vec![0.8; 1024 * 1024];
        let textured = vec![255; 1024 * 1024];
        let residual = ResidualWarp::new();
        let sources = [QualitySource {
            owner: 1,
            path: "synthetic.raw".to_string(),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 0,
            },
            dimensions: (1024, 1024),
        }];
        let input = QualityGateInput {
            output: &output,
            coverage: &coverage,
            ownership: &ownership,
            confidence: &confidence,
            textured: Some(&textured),
            owner_shortfall: None,
            owner_disagreement: None,
            unresolved_pixel_count: 0,
            world_origin: (0.0, 0.0),
            sources: &sources,
            residual: &residual,
            acceptance_render_scale: f64::NAN,
            final_sharpen_amount: 0.0,
        };
        let mut decodes = 0;
        let report = run_quality_gate(&input, &mut |_| {
            decodes += 1;
            Ok(output.clone())
        });
        assert!(
            report.roi_count > 0,
            "the fixture must select an actual ROI"
        );
        assert_eq!(decodes, 1);
        for name in ["mtf50_normalized", "gradient_energy_normalized"] {
            let criterion = report
                .criteria
                .iter()
                .find(|item| item.name == name)
                .unwrap();
            assert_eq!(criterion.measurable_count, 0);
            assert_eq!(criterion.unmeasurable_count, report.roi_count);
            assert_eq!(
                criterion
                    .unmeasurable_reasons
                    .get(degradation::LOCAL_SCALE_UNMEASURABLE),
                Some(&report.roi_count),
            );
            assert!(criterion.measured.is_empty());
            assert!(criterion.failed.is_empty());
        }
        assert_eq!(
            report
                .unmeasurable
                .iter()
                .filter(|item| item.reason == degradation::LOCAL_SCALE_UNMEASURABLE)
                .count(),
            2 * report.roi_count,
        );
    }

    #[test]
    fn quality_gate_record_mode_preserves_output_and_records_all_criteria() {
        // The gate is observation-only at this boundary: it may return a
        // conclusion and diagnostics, but it cannot rewrite the composed
        // pixels or add a publication-blocking degradation of its own.
        let _run_scope = degradation::begin_run_scope();
        degradation::reset_run_ledger();
        let output = Rgb32FImage::from_pixel(1024, 1024, Rgb([0.4, 0.45, 0.5]));
        let before = output.clone();
        let coverage = GrayImage::from_pixel(1024, 1024, Luma([255]));
        let ownership = vec![1u16; 1024 * 1024];
        let confidence = vec![0.8f32; 1024 * 1024];
        let textured = vec![255u8; 1024 * 1024];
        let shortfall = vec![5u8; 1024 * 1024];
        let disagreement = vec![0u8; 1024 * 1024];
        let residual = ResidualWarp::new();
        let sources = [QualitySource {
            owner: 1,
            path: "record-only.raw".to_string(),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 0,
            },
            dimensions: (1024, 1024),
        }];
        let input = QualityGateInput {
            output: &output,
            coverage: &coverage,
            ownership: &ownership,
            confidence: &confidence,
            textured: Some(&textured),
            owner_shortfall: Some(&shortfall),
            owner_disagreement: Some(&disagreement),
            unresolved_pixel_count: 0,
            world_origin: (0.0, 0.0),
            sources: &sources,
            residual: &residual,
            acceptance_render_scale: 1.0,
            final_sharpen_amount: 0.0,
        };
        let report = run_quality_gate(&input, &mut |_source| Ok(output.clone()));

        assert_eq!(
            output, before,
            "record-only Quality_Gate must not rewrite pixels"
        );
        assert_eq!(report.criteria.len(), quality_gate::QUALITY_CRITERIA.len());
        assert!(
            report
                .criteria
                .iter()
                .all(|criterion| criterion.threshold.is_finite())
        );
        assert!(degradation::run_ledger_snapshot().entries().is_empty());
    }

    #[test]
    fn unmeasurable_reason_histograms_match_the_coordinate_records() {
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 512,
            region_label: 1,
            owner: 1,
            world_origin: (-15.0, 72.0),
            boundary_clearance: 16.0,
        };
        let mut criterion = Criterion::new("mtf50_normalized", 0.93);
        let mut records = Vec::new();
        for reason in [
            degradation::SLANTED_EDGE_TOO_SHORT,
            degradation::SLANTED_EDGE_TOO_SHORT,
            degradation::SLANTED_EDGE_CONTRAST_INSUFFICIENT,
            degradation::SLANTED_EDGE_ANGLE_OUT_OF_RANGE,
            degradation::SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED,
        ] {
            record_unmeasurable(&mut criterion, &mut records, &roi, reason);
        }
        let result = criterion.finish(&mut records);
        assert_eq!(result.measurable_count, 0);
        assert_eq!(result.unmeasurable_count, 5);
        assert_eq!(result.unmeasurable_reasons.values().sum::<usize>(), 5);
        assert_eq!(
            result.unmeasurable_reasons[degradation::SLANTED_EDGE_TOO_SHORT],
            2
        );
        for (reason, count) in &result.unmeasurable_reasons {
            assert_eq!(
                records
                    .iter()
                    .filter(|record| &record.reason == reason)
                    .count(),
                *count
            );
        }
        assert!(records.iter().all(|record| {
            record.criterion == "mtf50_normalized"
                && record.world == WorldPoint { x: -15.0, y: 72.0 }
        }));
    }

    #[test]
    fn a_flat_boundary_reports_one_reason_per_sample_without_an_extra_summary_miss() {
        let output = Rgb32FImage::from_pixel(40, 512, Rgb([0.5; 3]));
        let coverage = GrayImage::from_pixel(40, 512, Luma([255]));
        let ownership = (0..40 * 512)
            .map(|index| if index % 40 < 20 { 1 } else { 2 })
            .collect::<Vec<_>>();
        let mut criterion = Criterion::p95_with_max("boundary_stroke_alignment", 1.5, 3.0);
        let mut records = Vec::new();
        measure_large_boundaries(
            &output,
            &coverage,
            &ownership,
            (10.0, -30.0),
            &mut criterion,
            &mut records,
        );
        assert_eq!(criterion.observed_count, 0);
        assert_eq!(criterion.unmeasurable, 2, "512px boundary has two samples");
        assert_eq!(
            criterion.unmeasurable_reasons[degradation::BOUNDARY_LOW_CONTRAST],
            2
        );
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].world, WorldPoint { x: 29.0, y: -30.0 });
        assert_eq!(records[1].world, WorldPoint { x: 29.0, y: 226.0 });
    }
}
