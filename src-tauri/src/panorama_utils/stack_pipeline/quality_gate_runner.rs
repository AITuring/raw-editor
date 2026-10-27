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
use super::tone;

/// The native station-local displacement that was actually applied to one
/// Source_RAW before it entered the fused Virtual_Tile.  The homographies are
/// kept alongside the field because the field is addressed in the station
/// canvas, while the Quality_Gate reference is sampled in Source_RAW pixels.
#[derive(Debug, Clone)]
pub(crate) struct NativeSourceWarp {
    /// `Some` when the registrar's native field was applied to the rendered
    /// station layer.  `None` still carries the station provenance: the
    /// reference must reproduce the ordinary raw -> station-layer -> output
    /// sampling chain even when no residual field was needed.
    pub(crate) displacement: Option<super::super::mosaic::NativeResidualDisplacement>,
    pub(crate) source_to_station: Matrix3<f64>,
    pub(crate) station_to_world: Matrix3<f64>,
}

/// A source entry used by the runtime adapter. `owner` is the value copied to
/// the final ownership plane; IDs need not be contiguous.
#[derive(Debug, Clone)]
pub(crate) struct QualitySource {
    pub owner: u16,
    pub path: String,
    pub geometry: SourceGeometry,
    pub dimensions: (u32, u32),
    pub native_warp: Option<NativeSourceWarp>,
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
    /// Accepted Station_Relations as normalized station-index pairs. Boundary
    /// evidence outside these real overlaps is not a production seam metric.
    pub accepted_station_relations: &'a [(usize, usize)],
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
    effective_output_pixels: Option<u64>,
    effective_projected_pixels: Option<u64>,
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
            effective_output_pixels: None,
            effective_projected_pixels: None,
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
            effective_output_pixels: self.effective_output_pixels,
            effective_projected_pixels: self.effective_projected_pixels,
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

/// Promote a failed or evidence-insufficient Quality_Gate result to the
/// run-scoped rejection ledger.  Measurement remains observation-only inside
/// `run_quality_gate`; the production compositor calls this at the export
/// boundary so diagnostics are retained while final publication is blocked.
pub(crate) fn record_gate_outcome(report: &QualityGateReport) {
    if !matches!(
        report.verdict,
        QualityGateVerdict::Fail | QualityGateVerdict::InsufficientEvidence
    ) {
        return;
    }
    let failed_criteria = report
        .criteria
        .iter()
        .filter(|criterion| {
            matches!(
                criterion.verdict,
                QualityGateVerdict::Fail | QualityGateVerdict::InsufficientEvidence
            )
        })
        .map(|criterion| {
            serde_json::json!({
                "name": criterion.name,
                "verdict": criterion.verdict,
                "measurable_count": criterion.measurable_count,
                "unmeasurable_count": criterion.unmeasurable_count,
                "failed_count": criterion.failed.len(),
                "failed": criterion.failed,
            })
        })
        .collect::<Vec<_>>();
    let unmeasurable = report
        .unmeasurable
        .iter()
        .filter(|item| {
            report.criteria.iter().any(|criterion| {
                criterion.name == item.criterion
                    && matches!(
                        criterion.verdict,
                        QualityGateVerdict::Fail | QualityGateVerdict::InsufficientEvidence
                    )
            })
        })
        .collect::<Vec<_>>();
    let reason = if report.verdict == QualityGateVerdict::Fail {
        degradation::QUALITY_GATE_CRITERION_FAILED
    } else {
        degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE
    };
    degradation::record_run_degradation(
        reason,
        serde_json::json!({
            "verdict": report.verdict,
            "criteria": failed_criteria,
            "unmeasurable": unmeasurable,
            "roi_count": report.roi_count,
        }),
    );
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

/// Source sampling through the complete output-to-source mapping.
///
/// The ordinary fixture path has one source resample.  A production focus
/// owner has two: Source_RAW is first rendered into the station layer, then
/// the final Virtual_Tile compositor samples that layer.  The latter path is
/// handled by [`source_roi_with_native`] when station provenance is present.
fn source_roi(
    source: &Rgb32FImage,
    roi: &QualityRoi,
    geometry: &SourceGeometry,
    residual: &ResidualWarp,
) -> Result<Rgb32FImage, &'static str> {
    source_roi_with_native(source, roi, geometry, residual, None)
}

/// Replay the bounded per-station tone transform that was actually applied by
/// the owner compositor before the Quality_Gate compares output to its source
/// reference.  The compositor corrects only the low-frequency band, so this
/// whole-ROI affine replay is intentionally used for the dense mean/gradient
/// evidence only; it prevents an already-applied exposure correction from
/// being reported as a source-fidelity defect without changing any threshold.
fn apply_run_tone_transform(reference: &mut Rgb32FImage, station_id: usize) {
    let Some(report) = super::tone::run_report_snapshot() else {
        return;
    };
    if matches!(
        report.status,
        super::report::ToneStatus::Identity | super::report::ToneStatus::NotRun
    ) {
        return;
    }
    let Some(tile) = report
        .tiles
        .iter()
        .find(|tile| tile.station_index == station_id)
    else {
        return;
    };
    for pixel in reference.pixels_mut() {
        for channel in 0..3 {
            pixel[channel] = (tile.gain[channel] as f32 * pixel[channel]
                + tile.offset[channel] as f32)
                .clamp(0.0, 1.0);
        }
    }
}

/// Sample a Source_RAW through the exact station-local native refinement that
/// was applied to the owner layer, when one exists.  The regular geometry
/// remains the fallback for fixtures and legacy paths that did not run native
/// refinement.
fn source_roi_with_native(
    source: &Rgb32FImage,
    roi: &QualityRoi,
    geometry: &SourceGeometry,
    residual: &ResidualWarp,
    native_warp: Option<&NativeSourceWarp>,
) -> Result<Rgb32FImage, &'static str> {
    let inverse = (geometry.tile_to_world * geometry.member_to_anchor)
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let station_inverse = native_warp.and_then(|warp| warp.station_to_world.try_inverse());
    let tile_to_source = native_warp.and_then(|warp| {
        let source_to_tile = warp.station_to_world.try_inverse()?
            * geometry.tile_to_world
            * geometry.member_to_anchor;
        source_to_tile.try_inverse()
    });
    let mut result = Rgb32FImage::new(roi.side, roi.side);
    for y in 0..roi.side {
        for x in 0..roi.side {
            let world = Point2::new(
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            );
            if let (Some(station_inverse), Some(tile_to_source), Some(native_warp)) = (
                station_inverse.as_ref(),
                tile_to_source.as_ref(),
                native_warp,
            ) {
                // `tile_local` is the coordinate in the already rendered
                // station plane.  Sampling that plane through the same 4x4
                // cubic kernel as the final compositor is what keeps a
                // gradient/MTF comparison from treating the raw input as a
                // sharper one-stage reference than the delivered pixels.
                let unwarped = residual.warp_inverse(world, geometry.station_id);
                let tile_local = project(station_inverse, unwarped)
                    .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
                let sampled = sample_station_layer_reference(
                    source,
                    tile_local,
                    tile_to_source,
                    native_warp.displacement.as_ref(),
                )
                .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
                result.put_pixel(x, y, sampled);
            } else {
                let mapped = map_world_to_source_for_quality(
                    world,
                    geometry,
                    residual,
                    &inverse,
                    native_warp,
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
                result.put_pixel(
                    x,
                    y,
                    crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                        source, mapped.x, mapped.y,
                    ),
                );
            }
        }
    }
    Ok(result)
}

pub(crate) fn sample_quality_source_at_world(
    source: &Rgb32FImage,
    world: Point2<f64>,
    source_info: &QualitySource,
    residual: &ResidualWarp,
) -> Result<Rgb<f32>, &'static str> {
    let geometry = &source_info.geometry;
    if let Some(native_warp) = source_info.native_warp.as_ref() {
        let station_inverse = native_warp
            .station_to_world
            .try_inverse()
            .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
        let source_to_tile = station_inverse * geometry.tile_to_world * geometry.member_to_anchor;
        let tile_to_source = source_to_tile
            .try_inverse()
            .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
        let unwarped = residual.warp_inverse(world, geometry.station_id);
        let tile_local =
            project(&station_inverse, unwarped).ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
        return sample_station_layer_reference(
            source,
            tile_local,
            &tile_to_source,
            native_warp.displacement.as_ref(),
        )
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE);
    }
    let inverse = (geometry.tile_to_world * geometry.member_to_anchor)
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let mapped = map_world_to_source_for_quality(world, geometry, residual, &inverse, None)?;
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
    Ok(
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            source, mapped.x, mapped.y,
        ),
    )
}

/// Read one integer station-layer pixel from the Source_RAW.  Native
/// refinement is a bilinear read of the pre-warp layer; the pre-warp layer
/// itself is the cubic Source_RAW sample produced by `render_focus_layer`.
fn sample_station_layer_integer(
    source: &Rgb32FImage,
    tile_to_source: &Matrix3<f64>,
    tile_x: f64,
    tile_y: f64,
    displacement: Option<&super::super::mosaic::NativeResidualDisplacement>,
) -> Option<Rgb<f32>> {
    let sample_prewarp = |x: f64, y: f64| {
        let raw = project(tile_to_source, Point2::new(x, y))?;
        if !raw.x.is_finite()
            || !raw.y.is_finite()
            || raw.x < 0.0
            || raw.y < 0.0
            || raw.x > f64::from(source.width().saturating_sub(1))
            || raw.y > f64::from(source.height().saturating_sub(1))
        {
            return None;
        }
        Some(
            crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                source, raw.x, raw.y,
            ),
        )
    };
    let Some(displacement) = displacement else {
        return sample_prewarp(tile_x, tile_y);
    };
    let [delta_x, delta_y] = displacement.at(tile_x, tile_y);
    let sample_x = tile_x + delta_x;
    let sample_y = tile_y + delta_y;
    if !sample_x.is_finite() || !sample_y.is_finite() || sample_x < 0.0 || sample_y < 0.0 {
        return None;
    }
    let x0 = sample_x.floor();
    let y0 = sample_y.floor();
    let fx = (sample_x - x0) as f32;
    let fy = (sample_y - y0) as f32;
    let top_left = sample_prewarp(x0, y0)?;
    let top_right = sample_prewarp(x0 + 1.0, y0)?;
    let bottom_left = sample_prewarp(x0, y0 + 1.0)?;
    let bottom_right = sample_prewarp(x0 + 1.0, y0 + 1.0)?;
    Some(Rgb(std::array::from_fn(|channel| {
        let top = top_left[channel] * (1.0 - fx) + top_right[channel] * fx;
        let bottom = bottom_left[channel] * (1.0 - fx) + bottom_right[channel] * fx;
        top * (1.0 - fy) + bottom * fy
    })))
}

/// Cubic sample of the station-layer values around one final output pixel.
/// The values are reconstructed from the exact source transform above; no
/// third image is decoded and no metric threshold is changed.
fn sample_station_layer_reference(
    source: &Rgb32FImage,
    tile: Point2<f64>,
    tile_to_source: &Matrix3<f64>,
    displacement: Option<&super::super::mosaic::NativeResidualDisplacement>,
) -> Option<Rgb<f32>> {
    if !tile.x.is_finite() || !tile.y.is_finite() {
        return None;
    }
    let x_floor = tile.x.floor() as i32;
    let y_floor = tile.y.floor() as i32;
    let amount_x = tile.x - f64::from(x_floor);
    let amount_y = tile.y - f64::from(y_floor);
    if amount_x.abs() < 1.0e-9 && amount_y.abs() < 1.0e-9 {
        return sample_station_layer_integer(
            source,
            tile_to_source,
            f64::from(x_floor),
            f64::from(y_floor),
            displacement,
        );
    }
    let mut values = [[[0.0f64; 3]; 4]; 4];
    let mut local_min = [f64::INFINITY; 3];
    let mut local_max = [f64::NEG_INFINITY; 3];
    for (row, sample_y) in ((y_floor - 1)..=(y_floor + 2)).enumerate() {
        for (column, sample_x) in ((x_floor - 1)..=(x_floor + 2)).enumerate() {
            let pixel = sample_station_layer_integer(
                source,
                tile_to_source,
                f64::from(sample_x),
                f64::from(sample_y),
                displacement,
            )?;
            for channel in 0..3 {
                let value = f64::from(pixel[channel]);
                values[row][column][channel] = value;
                local_min[channel] = local_min[channel].min(value);
                local_max[channel] = local_max[channel].max(value);
            }
        }
    }
    Some(Rgb(std::array::from_fn(|channel| {
        let mut rows = [0.0f64; 4];
        for row in 0..4 {
            rows[row] = cubic_sample(
                values[row][0][channel],
                values[row][1][channel],
                values[row][2][channel],
                values[row][3][channel],
                amount_x,
            );
        }
        cubic_sample(rows[0], rows[1], rows[2], rows[3], amount_y)
            .clamp(local_min[channel], local_max[channel]) as f32
    })))
}

fn cubic_sample(p0: f64, p1: f64, p2: f64, p3: f64, amount: f64) -> f64 {
    p1 + 0.5
        * amount
        * (p2 - p0
            + amount * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + amount * (3.0 * (p1 - p2) + p3 - p0)))
}

fn map_world_to_source_for_quality(
    world: Point2<f64>,
    geometry: &SourceGeometry,
    residual: &ResidualWarp,
    inverse: &Matrix3<f64>,
    native_warp: Option<&NativeSourceWarp>,
) -> Result<Point2<f64>, &'static str> {
    let unwarped = residual.warp_inverse(world, geometry.station_id);
    let Some(native_warp) = native_warp else {
        return project(inverse, unwarped).map_or(Err(degradation::OWNER_SOURCE_UNDECODABLE), Ok);
    };
    let tile_inverse = native_warp
        .station_to_world
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let tile_local =
        project(&tile_inverse, unwarped).ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    // `NativeResidualDisplacement` is addressed in the rendered station-tile
    // canvas.  `station_to_world.inverse()` already returned that coordinate;
    // adding `sampling_origin` here converted it back to world space and made
    // the field lookup depend on the tile's global offset.  The layer warp
    // applies the same field at `(candidate.left + x, candidate.top + y)`, so
    // use the tile-local point directly and keep the source transform in the
    // world frame below.
    let delta = native_warp
        .displacement
        .as_ref()
        .map(|displacement| displacement.at(tile_local.x, tile_local.y))
        .unwrap_or([0.0, 0.0]);
    let source_inverse = native_warp
        .source_to_station
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    project(
        &source_inverse,
        Point2::new(unwarped.x + delta[0], unwarped.y + delta[1]),
    )
    .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)
}

#[cfg(test)]
#[derive(Clone)]
struct ToneOwnerEvidence {
    partition: &'static str,
    roi: QualityRoi,
    owner_path: String,
    station_id: usize,
    tile_index: usize,
    baseline_mean: [f32; 3],
    candidate_mean: [f32; 3],
    reference_mean: [f32; 3],
    field_gain: [f32; 3],
    global_gain: [f32; 3],
    global_bias: [f32; 3],
    coverage_min: u8,
    confidence_min: f32,
    confidence_mean: f64,
    overlap_participation: bool,
    fit_disjoint: bool,
    gradient_ratio_candidate_to_source: Option<f64>,
    noise_ratio_candidate_to_source: Option<f64>,
}

#[cfg(test)]
fn tone_owner_distribution(rgb: [f32; 3]) -> serde_json::Value {
    serde_json::json!({
        "display_srgb": rgb,
        "linear_rgb": rgb.map(crate::color_management::srgb_to_linear_channel),
        "lab": super::tone::diagnostic_lab_rgb(rgb),
    })
}

#[cfg(test)]
fn tone_percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * quantile).round() as usize]
}

#[cfg(test)]
fn tone_owner_candidate_roi(
    snapshot: &super::tone::ToneOwnerOracleSnapshot,
    baseline_image: &Rgb32FImage,
    roi: &QualityRoi,
    relaxation: f64,
) -> Option<Rgb32FImage> {
    let baseline = output_roi(baseline_image, roi)?;
    Some(Rgb32FImage::from_fn(roi.side, roi.side, |x, y| {
        let world_x = roi.world_origin.0 + f64::from(x);
        let world_y = roi.world_origin.1 + f64::from(y);
        Rgb(snapshot.candidate_pixel(
            roi.owner,
            world_x,
            world_y,
            baseline.get_pixel(x, y).0,
            relaxation,
        ))
    }))
}

#[cfg(test)]
fn tone_owner_holdout_rois(
    input: &QualityGateInput<'_>,
    formal: &[QualityRoi],
    snapshot: &super::tone::ToneOwnerOracleSnapshot,
) -> Vec<QualityRoi> {
    const SIDE: u32 = 8;
    const STEP: usize = 16;
    const STRATUM: u32 = 512;
    const MAX_HOLDOUTS: usize = 256;
    let (width, height) = input.coverage.dimensions();
    if width < SIDE || height < SIDE {
        return Vec::new();
    }
    let intersects_formal = |x: u32, y: u32| {
        formal.iter().any(|roi| {
            x < roi.x + roi.side && x + SIDE > roi.x && y < roi.y + roi.side && y + SIDE > roi.y
        })
    };
    let mut strata = std::collections::BTreeSet::new();
    let mut result = Vec::new();
    for y in (0..=height - SIDE).step_by(STEP) {
        for x in (0..=width - SIDE).step_by(STEP) {
            if intersects_formal(x, y) {
                continue;
            }
            let owner = input.ownership[(y * width + x) as usize];
            if owner == 0 {
                continue;
            }
            let mut valid = true;
            'rows: for ry in y..y + SIDE {
                for rx in x..x + SIDE {
                    let index = (ry * width + rx) as usize;
                    let world_x = input.world_origin.0 + f64::from(rx);
                    let world_y = input.world_origin.1 + f64::from(ry);
                    if input.coverage.as_raw()[index] == 0
                        || input.ownership[index] != owner
                        || snapshot.is_fit_cell(world_x, world_y)
                    {
                        valid = false;
                        break 'rows;
                    }
                }
            }
            let world_x = input.world_origin.0 + f64::from(x + SIDE / 2);
            let world_y = input.world_origin.1 + f64::from(y + SIDE / 2);
            if !valid || snapshot.field_sample(owner, world_x, world_y).is_none() {
                continue;
            }
            let key = (owner, x / STRATUM, y / STRATUM);
            if !strata.insert(key) {
                continue;
            }
            result.push(QualityRoi {
                x,
                y,
                side: SIDE,
                region_label: 0,
                owner,
                world_origin: (
                    input.world_origin.0 + f64::from(x),
                    input.world_origin.1 + f64::from(y),
                ),
                boundary_clearance: 0.0,
            });
            if result.len() == MAX_HOLDOUTS {
                return result;
            }
        }
    }
    result
}

#[cfg(test)]
fn unique_owner_source(sources: &[QualitySource], owner: u16) -> Result<&QualitySource, String> {
    let matching = sources
        .iter()
        .filter(|source| source.owner == owner)
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(format!(
            "owner {owner} has {} Source_RAW provenance entries",
            matching.len()
        ));
    }
    Ok(matching[0])
}

#[cfg(test)]
fn collect_tone_owner_evidence(
    input: &QualityGateInput<'_>,
    snapshot: &super::tone::ToneOwnerOracleSnapshot,
    formal: &[QualityRoi],
    holdouts: &[QualityRoi],
    load_source: &mut SourceLoader<'_>,
) -> Result<Vec<ToneOwnerEvidence>, String> {
    let mut by_owner = BTreeMap::<u16, Vec<(&'static str, &QualityRoi)>>::new();
    for roi in formal {
        by_owner
            .entry(roi.owner)
            .or_default()
            .push(("formal_quality_gate_verification", roi));
    }
    for roi in holdouts {
        by_owner
            .entry(roi.owner)
            .or_default()
            .push(("owner_raw_fidelity_holdout", roi));
    }
    let mut result = Vec::new();
    for (owner, rois) in by_owner {
        let source_info = unique_owner_source(input.sources, owner)?;
        let source = load_source(source_info)?;
        for (partition, roi) in rois {
            let baseline = output_roi(input.output, roi).ok_or_else(|| {
                "tone owner baseline ROI is outside the current canvas".to_string()
            })?;
            let candidate =
                tone_owner_candidate_roi(snapshot, input.output, roi, 1.0).ok_or_else(|| {
                    "tone owner candidate ROI is outside the current canvas".to_string()
                })?;
            let reference = source_roi_with_native(
                &source,
                roi,
                &source_info.geometry,
                input.residual,
                source_info.native_warp.as_ref(),
            )
            .map_err(str::to_string)?;
            let baseline_mean = quality_gate::roi_mean_rgb(&baseline)
                .ok_or_else(|| "tone owner baseline ROI is empty".to_string())?;
            let candidate_mean = quality_gate::roi_mean_rgb(&candidate)
                .ok_or_else(|| "tone owner candidate ROI is empty".to_string())?;
            let reference_mean = quality_gate::roi_mean_rgb(&reference)
                .ok_or_else(|| "tone owner Source_RAW ROI is empty".to_string())?;
            let center = (
                roi.world_origin.0 + f64::from(roi.side) * 0.5,
                roi.world_origin.1 + f64::from(roi.side) * 0.5,
            );
            let (_, field_gain, global_gain, global_bias, tile_index) = snapshot
                .field_sample(owner, center.0, center.1)
                .unwrap_or(([0.0; 3], [1.0; 3], [1.0; 3], [0.0; 3], usize::MAX));
            let mut coverage_min = u8::MAX;
            let mut confidence_min = f32::INFINITY;
            let mut confidence_sum = 0.0;
            for y in roi.y..roi.y + roi.side {
                for x in roi.x..roi.x + roi.side {
                    let index = (y * input.coverage.width() + x) as usize;
                    coverage_min = coverage_min.min(input.coverage.as_raw()[index]);
                    confidence_min = confidence_min.min(input.confidence[index]);
                    confidence_sum += f64::from(input.confidence[index]);
                }
            }
            let samples = u64::from(roi.side) * u64::from(roi.side);
            let local_scale = quality_gate::local_scale_for_roi(
                roi,
                &source_info.geometry,
                input.residual,
                input.acceptance_render_scale,
            )
            .ok()
            .map(|measurement| measurement.median);
            let gradient_ratio_candidate_to_source = local_scale.and_then(|scale| {
                gradient_energy_ratio(&candidate, &reference, scale)
                    .ok()
                    .flatten()
                    .map(|measurement| measurement.0)
            });
            let noise_ratio_candidate_to_source =
                quality_gate::noise_sigma_ratio(&candidate, &reference).ok();
            result.push(ToneOwnerEvidence {
                partition,
                roi: roi.clone(),
                owner_path: source_info.path.clone(),
                station_id: source_info.geometry.station_id,
                tile_index,
                baseline_mean,
                candidate_mean,
                reference_mean,
                field_gain,
                global_gain,
                global_bias,
                coverage_min,
                confidence_min,
                confidence_mean: confidence_sum / samples.max(1) as f64,
                overlap_participation: snapshot.overlap_participation(center.0, center.1),
                fit_disjoint: (0..roi.side).all(|y| {
                    (0..roi.side).all(|x| {
                        !snapshot.is_fit_cell(
                            roi.world_origin.0 + f64::from(x),
                            roi.world_origin.1 + f64::from(y),
                        )
                    })
                }),
                gradient_ratio_candidate_to_source,
                noise_ratio_candidate_to_source,
            });
        }
    }
    result.sort_by(|left, right| {
        left.partition
            .cmp(right.partition)
            .then_with(|| left.roi.owner.cmp(&right.roi.owner))
            .then_with(|| left.roi.y.cmp(&right.roi.y))
            .then_with(|| left.roi.x.cmp(&right.roi.x))
    });
    Ok(result)
}

#[cfg(test)]
fn owner_delta_at(evidence: &ToneOwnerEvidence, relaxation: f64) -> f64 {
    let post = std::array::from_fn(|channel| {
        (f64::from(evidence.baseline_mean[channel])
            + relaxation
                * f64::from(evidence.candidate_mean[channel] - evidence.baseline_mean[channel]))
            as f32
    });
    super::tone::delta_e00_rgb(post, evidence.reference_mean)
}

#[cfg(test)]
fn owner_distribution_feasible(evidence: &[&ToneOwnerEvidence], relaxation: f64) -> bool {
    if evidence.is_empty() || evidence.iter().any(|item| !item.fit_disjoint) {
        return false;
    }
    let baseline = evidence
        .iter()
        .map(|item| owner_delta_at(item, 0.0))
        .collect::<Vec<_>>();
    let candidate = evidence
        .iter()
        .map(|item| owner_delta_at(item, relaxation))
        .collect::<Vec<_>>();
    tone_percentile(&candidate, 0.5) <= tone_percentile(&baseline, 0.5) + 1e-9
        && tone_percentile(&candidate, 0.95) <= tone_percentile(&baseline, 0.95) + 1e-9
        && tone_percentile(&candidate, 1.0) <= tone_percentile(&baseline, 1.0) + 1e-9
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct ToneJointRelaxation {
    selected: Option<f64>,
    minimum_boundary: Option<f64>,
    maximum_owner: Option<f64>,
    pareto_gap: Option<f64>,
}

#[cfg(test)]
fn select_joint_relaxation(boundary: &[f64], owner: &[f64]) -> ToneJointRelaxation {
    let selected = boundary.iter().copied().find(|alpha| {
        owner
            .binary_search_by(|value| value.total_cmp(alpha))
            .is_ok()
    });
    let minimum_boundary = boundary.first().copied();
    let maximum_owner = owner.last().copied();
    let pareto_gap = match (minimum_boundary, maximum_owner) {
        (Some(minimum), Some(maximum)) if minimum > maximum => Some(minimum - maximum),
        _ => None,
    };
    ToneJointRelaxation {
        selected,
        minimum_boundary,
        maximum_owner,
        pareto_gap,
    }
}

#[cfg(test)]
fn write_tone_owner_fidelity_oracle(
    input: &QualityGateInput<'_>,
    formal: &[QualityRoi],
    load_source: &mut SourceLoader<'_>,
) -> Result<(), String> {
    let Some(directory) = super::test_support::tone_owner_fidelity_dir() else {
        return Ok(());
    };
    let snapshot = super::tone::take_tone_owner_oracle_snapshot()
        .ok_or_else(|| "current-run tone owner oracle snapshot is unavailable".to_string())?;
    let holdouts = tone_owner_holdout_rois(input, formal, &snapshot);
    let evidence = collect_tone_owner_evidence(input, &snapshot, formal, &holdouts, load_source)?;
    let owner_holdout = evidence
        .iter()
        .filter(|item| item.partition == "owner_raw_fidelity_holdout")
        .collect::<Vec<_>>();
    let formal_evidence = evidence
        .iter()
        .filter(|item| item.partition == "formal_quality_gate_verification")
        .collect::<Vec<_>>();
    const STEPS: usize = 4096;
    let relaxations = (0..=STEPS)
        .map(|step| step as f64 / STEPS as f64)
        .collect::<Vec<_>>();
    let owner_feasible = relaxations
        .iter()
        .copied()
        .filter(|&alpha| owner_distribution_feasible(&owner_holdout, alpha))
        .collect::<Vec<_>>();
    let baseline_holdout = snapshot.checkerboard_holdout(0.0);
    let boundary_feasible = relaxations
        .iter()
        .copied()
        .filter(|&alpha| {
            let checkerboard = snapshot.checkerboard_holdout(alpha);
            snapshot.boundary_max(alpha) <= super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD
                && checkerboard.0 >= super::tone::TONE_MIN_SAMPLES
                && checkerboard.1 < baseline_holdout.1
                && checkerboard.2 < baseline_holdout.2
        })
        .collect::<Vec<_>>();
    let relaxation = select_joint_relaxation(&boundary_feasible, &owner_feasible);
    let selected = relaxation.selected;
    let applied_relaxation = selected.unwrap_or(0.0);
    let formal_values = formal_evidence
        .iter()
        .map(|item| owner_delta_at(item, applied_relaxation))
        .collect::<Vec<_>>();
    let formal_baseline = formal_evidence
        .iter()
        .map(|item| owner_delta_at(item, 0.0))
        .collect::<Vec<_>>();
    let formal_full_candidate = formal_evidence
        .iter()
        .map(|item| owner_delta_at(item, 1.0))
        .collect::<Vec<_>>();
    let formal_pass = selected.is_some()
        && formal_values.len() == formal.len()
        && formal_values
            .iter()
            .all(|value| *value <= quality_gate::QUALITY_ROI_DELTA_E00_MAX)
        && tone_percentile(&formal_values, 0.5) <= tone_percentile(&formal_baseline, 0.5) + 1e-9
        && tone_percentile(&formal_values, 0.95) <= tone_percentile(&formal_baseline, 0.95) + 1e-9;
    let candidate_full_high_frequency_pass = evidence.iter().all(|item| {
        item.gradient_ratio_candidate_to_source
            .is_none_or(|value| value >= quality_gate::QUALITY_GRADIENT_RATIO_MIN)
            && item.noise_ratio_candidate_to_source.is_none_or(|value| {
                (quality_gate::QUALITY_NOISE_RATIO_MIN..=quality_gate::QUALITY_NOISE_RATIO_MAX)
                    .contains(&value)
            })
    });
    let selected_high_frequency_pass = selected.is_none() || candidate_full_high_frequency_pass;
    let field_constraint_stats = snapshot.field_constraint_stats();
    let records = evidence
        .iter()
        .map(|item| {
            let selected_rgb = std::array::from_fn::<f32, 3, _>(|channel| {
                item.baseline_mean[channel]
                    + applied_relaxation as f32
                        * (item.candidate_mean[channel] - item.baseline_mean[channel])
            });
            serde_json::json!({
                "partition": item.partition,
                "world": { "x": item.roi.world_origin.0, "y": item.roi.world_origin.1, "side": item.roi.side },
                "owner": item.roi.owner,
                "station": item.station_id,
                "tile_index": item.tile_index,
                "spatial_stratum": [item.roi.x / 512, item.roi.y / 512],
                "owner_source_raw": item.owner_path,
                "source_provenance_count": 1,
                "mapping": "residual_warp_inverse_then_inverse(tile_to_world*member_to_anchor)",
                "sampler": "production_bounded_catmull_rom",
                "overlap_participation": item.overlap_participation,
                "fit_disjoint": item.fit_disjoint,
                "coverage_min": item.coverage_min,
                "confidence": { "minimum": item.confidence_min, "mean": item.confidence_mean },
                "field_gain": item.field_gain,
                "global_gain": item.global_gain,
                "global_bias": item.global_bias,
                "pre": tone_owner_distribution(item.baseline_mean),
                "candidate_full": tone_owner_distribution(item.candidate_mean),
                "selected": tone_owner_distribution(selected_rgb),
                "source_raw": tone_owner_distribution(item.reference_mean),
                "delta_e00": {
                    "pre": owner_delta_at(item, 0.0),
                    "candidate_full": owner_delta_at(item, 1.0),
                    "selected": owner_delta_at(item, applied_relaxation),
                },
                "high_frequency": {
                    "gradient_ratio_candidate_to_source_raw": item.gradient_ratio_candidate_to_source,
                    "noise_ratio_candidate_to_source_raw": item.noise_ratio_candidate_to_source,
                },
            })
        })
        .collect::<Vec<_>>();
    let checkerboard_selected = snapshot.checkerboard_holdout(applied_relaxation);
    let document = serde_json::json!({
        "schema": 1,
        "source": "current_run_only",
        "previous_quality_gate_results_read": false,
        "production_spatial_field_enabled": super::tone::production_spatial_field_enabled(),
        "evidence": records,
        "partitions": {
            "overlap_fit_cells": snapshot.fit_cell_count(),
            "overlap_checkerboard_holdout_samples": baseline_holdout.0,
            "owner_raw_fidelity_holdout_samples": owner_holdout.len(),
            "formal_quality_gate_rois": formal_evidence.len(),
            "owner_holdout_disjoint_from_fit": owner_holdout.iter().all(|item| item.fit_disjoint),
        },
        "oracle": {
            "family": "bounded_shared_world_field_linear_relaxation",
            "constraints": {
                "field_gain_minimum": field_constraint_stats.0,
                "field_gain_maximum": field_constraint_stats.1,
                "maximum_neighbour_log_gain_delta": field_constraint_stats.2,
                "smoothing": { "kernel": "gaussian", "sigma_cells": super::tone::TONE_LOW_FREQUENCY_SIGMA },
                "graph": "zero_mean_joint_accepted_relation_solve",
                "owner_holdout_distribution_not_worse": ["median", "p95", "max"],
                "formal_roi_delta_e00_threshold_unchanged": quality_gate::QUALITY_ROI_DELTA_E00_MAX,
            },
            "relaxation_resolution": 1.0 / STEPS as f64,
            "selected_relaxation": selected,
            "identity_fallback": selected.is_none(),
            "boundary": {
                "windows": snapshot.boundary_window_count(),
                "baseline_max_delta_e00": snapshot.boundary_max(0.0),
                "selected_max_delta_e00": snapshot.boundary_max(applied_relaxation),
                "threshold": super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD,
                "minimum_feasible_relaxation": relaxation.minimum_boundary,
            },
            "owner_fidelity": {
                "maximum_feasible_relaxation": relaxation.maximum_owner,
                "threshold": quality_gate::QUALITY_ROI_DELTA_E00_MAX,
                "baseline_median": tone_percentile(&formal_baseline, 0.5),
                "baseline_p95": tone_percentile(&formal_baseline, 0.95),
                "baseline_max": tone_percentile(&formal_baseline, 1.0),
                "candidate_full_median": tone_percentile(&formal_full_candidate, 0.5),
                "candidate_full_p95": tone_percentile(&formal_full_candidate, 0.95),
                "candidate_full_max": tone_percentile(&formal_full_candidate, 1.0),
                "candidate_full_violations": formal_full_candidate.iter().filter(|value| **value > quality_gate::QUALITY_ROI_DELTA_E00_MAX).count(),
                "selected_median": tone_percentile(&formal_values, 0.5),
                "selected_p95": tone_percentile(&formal_values, 0.95),
                "selected_max": formal_values.iter().copied().fold(0.0, f64::max),
                "formal_pass": formal_pass,
            },
            "checkerboard_holdout": {
                "samples": checkerboard_selected.0,
                "baseline_median": baseline_holdout.1,
                "baseline_p95": baseline_holdout.2,
                "selected_median": checkerboard_selected.1,
                "selected_p95": checkerboard_selected.2,
            },
            "cycle_residual_max_log": snapshot.cycle_residual_max(),
            "candidate_full_high_frequency_pass": candidate_full_high_frequency_pass,
            "selected_high_frequency_pass": selected_high_frequency_pass,
            "feasible_joint_solution": selected.is_some() && formal_pass && candidate_full_high_frequency_pass,
            "pareto_gap": relaxation.pareto_gap,
        },
    });
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create tone owner fidelity directory: {error}"))?;
    let path = directory.join("tone-owner-source-fidelity.json");
    let file = std::fs::File::create(&path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &document)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
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
        | degradation::BOUNDARY_STROKE_NOT_UNIQUE
        | degradation::BOUNDARY_ORIENTATION_MISMATCH => UnmeasurableCategory::ContentNotApplicable,
        _ => UnmeasurableCategory::Technical,
    }
}

fn ratio(measured: f64, reference: f64, min: f64, max: f64) -> Option<(f64, bool)> {
    quality_gate::compare_ratio(measured, reference, min, max, false).map(|v| (v.value, v.passed))
}

fn gradient_energy_ratio(
    output: &Rgb32FImage,
    reference: &Rgb32FImage,
    local_scale: f64,
) -> Result<Option<(f64, bool)>, &'static str> {
    let output_energy = quality_gate::normalized_gradient_energy(output, local_scale)?;
    // `source_roi` has already resampled the Source_RAW onto the output pixel
    // grid. It therefore has the same local scale as `output`; treating this
    // aligned reference as native-scale evidence divides only the output side
    // and turns an identity comparison into exactly `1 / local_scale`.
    let reference_energy = quality_gate::normalized_gradient_energy(reference, local_scale)?;
    Ok(ratio(
        output_energy,
        reference_energy,
        quality_gate::QUALITY_GRADIENT_RATIO_MIN,
        f64::INFINITY,
    ))
}

#[cfg(test)]
#[derive(serde::Serialize)]
struct GradientOracleCandidate {
    source: String,
    station_id: usize,
    local_scale: f64,
    residual_reference_energy: f64,
    identity_reference_energy: Option<f64>,
}

#[cfg(test)]
#[derive(serde::Serialize)]
struct GradientOracleRoi {
    world: [f64; 2],
    owner_source: String,
    station_id: usize,
    output_energy: f64,
    candidates: Vec<GradientOracleCandidate>,
}

#[cfg(test)]
fn write_gradient_source_oracle(
    input: &QualityGateInput<'_>,
    rois: &[QualityRoi],
    load_source: &mut SourceLoader<'_>,
) -> Result<(), String> {
    let Some(report_path) = std::env::var_os("RAW_EDITOR_FOCUS_GRADIENT_ORACLE_REPORT") else {
        return Ok(());
    };
    let Some(directory) = std::env::var_os("RAW_EDITOR_FOCUS_GRADIENT_ORACLE_DIR") else {
        return Ok(());
    };
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&report_path)
            .map_err(|error| format!("failed to read gradient oracle report: {error}"))?,
    )
    .map_err(|error| format!("failed to parse gradient oracle report: {error}"))?;
    let requested = report
        .pointer("/quality_gate/criteria")
        .and_then(serde_json::Value::as_array)
        .and_then(|criteria| {
            criteria.iter().find(|criterion| {
                criterion.get("name").and_then(serde_json::Value::as_str)
                    == Some("gradient_energy_normalized")
            })
        })
        .and_then(|criterion| criterion.get("failed"))
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|measurement| {
            Some((
                measurement.pointer("/world/x")?.as_f64()?.to_bits(),
                measurement.pointer("/world/y")?.as_f64()?.to_bits(),
            ))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let owner_sources = input
        .sources
        .iter()
        .map(|source| (source.owner, source))
        .collect::<BTreeMap<_, _>>();
    let selected = rois
        .iter()
        .filter(|roi| {
            requested.contains(&(roi.world_origin.0.to_bits(), roi.world_origin.1.to_bits()))
        })
        .collect::<Vec<_>>();
    let mut records = selected
        .iter()
        .filter_map(|roi| {
            let owner = owner_sources.get(&roi.owner).copied()?;
            let output = output_roi(input.output, roi)?;
            let output_energy = quality_gate::normalized_gradient_energy(&output, 1.0).ok()?;
            Some(GradientOracleRoi {
                world: [roi.world_origin.0, roi.world_origin.1],
                owner_source: owner.path.clone(),
                station_id: owner.geometry.station_id,
                output_energy,
                candidates: Vec::new(),
            })
        })
        .collect::<Vec<_>>();
    let identity = ResidualWarp::new();
    for source_info in input.sources {
        let relevant = records
            .iter()
            .enumerate()
            .filter(|(_, record)| record.station_id == source_info.geometry.station_id)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if relevant.is_empty() {
            continue;
        }
        let source = load_source(source_info)?;
        for index in relevant {
            let Some(roi) = selected.iter().copied().find(|roi| {
                roi.world_origin.0.to_bits() == records[index].world[0].to_bits()
                    && roi.world_origin.1.to_bits() == records[index].world[1].to_bits()
            }) else {
                continue;
            };
            let Ok(reference) = source_roi_with_native(
                &source,
                roi,
                &source_info.geometry,
                input.residual,
                source_info.native_warp.as_ref(),
            ) else {
                continue;
            };
            let Ok(residual_reference_energy) =
                quality_gate::normalized_gradient_energy(&reference, 1.0)
            else {
                continue;
            };
            let local_scale = quality_gate::local_scale_for_roi(
                roi,
                &source_info.geometry,
                input.residual,
                input.acceptance_render_scale,
            )
            .map(|measurement| measurement.median)
            .unwrap_or(f64::NAN);
            let identity_reference_energy = source_roi_with_native(
                &source,
                roi,
                &source_info.geometry,
                &identity,
                source_info.native_warp.as_ref(),
            )
            .ok()
            .and_then(|reference| quality_gate::normalized_gradient_energy(&reference, 1.0).ok());
            records[index].candidates.push(GradientOracleCandidate {
                source: source_info.path.clone(),
                station_id: source_info.geometry.station_id,
                local_scale,
                residual_reference_energy,
                identity_reference_energy,
            });
        }
    }
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create gradient oracle directory: {error}"))?;
    let path = directory.join("gradient-source-oracle.json");
    let file = std::fs::File::create(&path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &records)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

/// Keep the observation-only gate bounded on the full-resolution acceptance
/// renders. The exact region labelling and distance transform are useful for
/// ordinary unit-sized fixtures, but a 9k by 11k canvas would allocate and
/// scan several gigabytes just to choose 512px ROIs. A deterministic lattice
/// still checks ownership and coverage over every pixel in a candidate ROI;
/// effective coverage and boundary tracing use separate streaming passes.
pub(crate) fn select_sparse_rois(
    coverage: &GrayImage,
    ownership: &[u16],
    origin: (f64, f64),
) -> Vec<QualityRoi> {
    // Preserve the required 512px evidence window whenever the owner plane
    // contains enough such interiors.  Focus ownership is cell based, so a
    // valid run can have many sharp, source-owned regions smaller than 512px
    // even while the whole-image owner coverage passes.  Returning zero ROIs
    // in that case makes gradient/MTF a geometry-evidence failure rather than
    // a measurement of the delivered pixels.  The 256px fallback keeps the
    // same 16px owner/coverage ring and unchanged metric thresholds; it only
    // restores the minimum eight measurable interiors required by the gate.
    let primary = sparse_rois_for_side(coverage, ownership, origin, quality_gate::QUALITY_ROI_SIDE);
    if primary.len() >= quality_gate::QUALITY_MIN_MEASURABLE {
        return primary;
    }
    let mut best = primary;
    // A 256px core is the first fallback.  Highly interleaved focus owners
    // can still leave fewer than eight 256px interiors; continue at 128px so
    // the dense gradient criterion has auditable samples without changing
    // its 0.95 threshold.  The 128px minimum remains large enough for the
    // 13px acutance support and the fixed 16px owner ring.
    for side in [
        quality_gate::QUALITY_ROI_SIDE / 2,
        quality_gate::QUALITY_ROI_SIDE / 4,
    ] {
        if side < 2 {
            continue;
        }
        let fallback = sparse_rois_for_side(coverage, ownership, origin, side);
        if fallback.len() > best.len() {
            best = fallback;
        }
        if best.len() >= quality_gate::QUALITY_MIN_MEASURABLE {
            break;
        }
    }
    best
}

fn sparse_rois_for_side(
    coverage: &GrayImage,
    ownership: &[u16],
    origin: (f64, f64),
    side: u32,
) -> Vec<QualityRoi> {
    let (width, height) = coverage.dimensions();
    if side == 0 || width < side || height < side || ownership.len() != coverage.as_raw().len() {
        return Vec::new();
    }
    let step = (side as usize / 2).max(1);
    let margin = quality_gate::QUALITY_ROI_MARGIN as u32;
    let mut result = Vec::new();
    for y in (0..=height - side).step_by(step) {
        for x in (0..=width - side).step_by(step) {
            let owner = ownership[(y * width + x) as usize];
            if owner == 0 || coverage.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let same_owner = (y..y + side).all(|row| {
                (x..x + side).all(|column| {
                    let index = (row * width + column) as usize;
                    coverage.as_raw()[index] != 0 && ownership[index] == owner
                })
            });
            if !same_owner {
                continue;
            }
            // A candidate must have a 16px owner/coverage margin on all four
            // sides. Checking this narrow ring is equivalent to the exact
            // distance-transform predicate used by the small-plane selector,
            // while keeping the large-plane path bounded.
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

#[cfg(test)]
fn write_effective_coverage_diagnostics(
    coverage: &GrayImage,
    source_quadrilaterals: &[[Point2<f64>; 4]],
    canvas_world_origin: (f64, f64),
) {
    let Some(directory) = std::env::var_os("RAW_EDITOR_EFFECTIVE_COVERAGE_DIAGNOSTICS_DIR") else {
        return;
    };
    let (width, height) = coverage.dimensions();
    if width == 0 || height == 0 {
        return;
    }
    let mut projected = vec![0u8; width as usize * height as usize];
    for y in 0..height {
        let scan_y = canvas_world_origin.1 + f64::from(y) + 0.5;
        for quad in source_quadrilaterals {
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
                for x in start..end {
                    projected[(y * width + x) as usize] = 1;
                }
            }
        }
    }
    let output_nontransparent = coverage
        .as_raw()
        .iter()
        .filter(|&&value| value != 0)
        .count() as u64;
    let projected_union = projected.iter().filter(|&&value| value != 0).count() as u64;
    let output_outside_projected = coverage
        .as_raw()
        .iter()
        .zip(&projected)
        .filter(|(covered, in_union)| **covered != 0 && **in_union == 0)
        .count() as u64;
    let mut components = Vec::new();
    for start in 0..projected.len() {
        if projected[start] != 1 || coverage.as_raw()[start] != 0 {
            continue;
        }
        projected[start] = 2;
        let mut queue = std::collections::VecDeque::from([start]);
        let mut pixels = 0u64;
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        let mut touches_canvas_edge = false;
        while let Some(index) = queue.pop_front() {
            let x = (index % width as usize) as u32;
            let y = (index / width as usize) as u32;
            pixels += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            touches_canvas_edge |= x == 0 || y == 0 || x + 1 == width || y + 1 == height;
            for neighbour in [
                x.checked_sub(1).map(|nx| (nx, y)),
                (x + 1 < width).then_some((x + 1, y)),
                y.checked_sub(1).map(|ny| (x, ny)),
                (y + 1 < height).then_some((x, y + 1)),
            ]
            .into_iter()
            .flatten()
            {
                let neighbour = (neighbour.1 * width + neighbour.0) as usize;
                if projected[neighbour] == 1 && coverage.as_raw()[neighbour] == 0 {
                    projected[neighbour] = 2;
                    queue.push_back(neighbour);
                }
            }
        }
        components.push(serde_json::json!({
            "pixels": pixels,
            "bounds": { "x": min_x, "y": min_y, "width": max_x - min_x + 1, "height": max_y - min_y + 1 },
            "touches_canvas_edge": touches_canvas_edge,
        }));
    }
    components.sort_by(|left, right| right["pixels"].as_u64().cmp(&left["pixels"].as_u64()));
    let missing_projected = components
        .iter()
        .filter_map(|component| component["pixels"].as_u64())
        .sum::<u64>();
    let canvas_area = u64::from(width) * u64::from(height);
    let document = serde_json::json!({
        "canvas": { "width": width, "height": height, "area": canvas_area, "world_origin": [canvas_world_origin.0, canvas_world_origin.1] },
        "output_nontransparent": output_nontransparent,
        "canvas_transparent": canvas_area - output_nontransparent,
        "projected_union": projected_union,
        "missing_projected": missing_projected,
        "output_outside_projected": output_outside_projected,
        "transparent_outside_projected": canvas_area - output_nontransparent - missing_projected + output_outside_projected,
        "missing_fraction": missing_projected as f64 / projected_union.max(1) as f64,
        "component_count": components.len(),
        "components": components,
    });
    let directory = std::path::PathBuf::from(directory);
    if std::fs::create_dir_all(&directory).is_ok() {
        let _ = std::fs::write(
            directory.join("effective-coverage.json"),
            serde_json::to_vec_pretty(&document).unwrap_or_default(),
        );
    }
}

#[derive(Debug, Clone)]
struct BoundaryEdge {
    offset: f64,
    strength: f64,
    gradient: (f64, f64),
    pixel: (u32, u32),
    owner: u16,
}

#[derive(Debug, Clone)]
struct BoundaryPairEvidence {
    world: (f64, f64),
    error: f64,
    strength: f64,
    owner_path: String,
}

#[derive(Debug, Clone)]
enum BoundaryCellObservation {
    Measured(BoundaryPairEvidence),
    Unmeasurable {
        world: (f64, f64),
        reason: &'static str,
    },
}

fn boundary_edge_on_side_matching<F>(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    side: f64,
    distance: f64,
    owner_matches: F,
) -> Result<BoundaryEdge, &'static str>
where
    F: Fn(u16) -> bool,
{
    const FIXED_CROSS_SECTION_RADIUS_PX: i32 = 2;
    let (width, height) = image.dimensions();
    let tangent = (-normal.1, normal.0);
    let luminance = |pixel: Rgb<f32>| {
        0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
    };
    let mut candidates = Vec::new();
    for offset in -FIXED_CROSS_SECTION_RADIUS_PX..=FIXED_CROSS_SECTION_RADIUS_PX {
        let offset = f64::from(offset);
        let sx = seam.0 + side * normal.0 * distance + tangent.0 * offset;
        let sy = seam.1 + side * normal.1 * distance + tangent.1 * offset;
        let ix = sx.round() as i32;
        let iy = sy.round() as i32;
        if ix < 1 || iy < 1 || ix + 1 >= width as i32 || iy + 1 >= height as i32 {
            continue;
        }
        let index = iy as usize * width as usize + ix as usize;
        let Some(owner) = ownership.get(index).copied() else {
            continue;
        };
        if coverage.as_raw().get(index).copied().unwrap_or(0) == 0 || !owner_matches(owner) {
            continue;
        }
        let gx = luminance(*image.get_pixel((ix + 1) as u32, iy as u32))
            - luminance(*image.get_pixel((ix - 1) as u32, iy as u32));
        let gy = luminance(*image.get_pixel(ix as u32, (iy + 1) as u32))
            - luminance(*image.get_pixel(ix as u32, (iy - 1) as u32));
        let strength = gx.hypot(gy) * 0.5;
        if strength < 0.15 {
            continue;
        }
        let tangent_alignment = (gx * tangent.0 + gy * tangent.1).abs() / gx.hypot(gy);
        if tangent_alignment < 30.0f64.to_radians().cos() {
            continue;
        }
        candidates.push(BoundaryEdge {
            offset,
            strength,
            gradient: (gx, gy),
            pixel: (ix as u32, iy as u32),
            owner,
        });
    }
    candidates.sort_by(|left, right| {
        right
            .strength
            .total_cmp(&left.strength)
            .then_with(|| left.offset.total_cmp(&right.offset))
    });
    let Some(best) = candidates.first().cloned() else {
        return Err(degradation::BOUNDARY_LOW_CONTRAST);
    };
    if candidates.iter().skip(1).any(|candidate| {
        (candidate.offset - best.offset).abs() >= 3.0 && candidate.strength > best.strength * 0.8
    }) {
        return Err(degradation::BOUNDARY_STROKE_NOT_UNIQUE);
    }
    Ok(best)
}

fn boundary_edge_on_side(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    side: f64,
    distance: f64,
    expected_owner: u16,
) -> Result<BoundaryEdge, &'static str> {
    boundary_edge_on_side_matching(
        image,
        coverage,
        ownership,
        seam,
        normal,
        side,
        distance,
        |owner| owner == expected_owner,
    )
}

fn boundary_pair_error(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    left_owner: u16,
    right_owner: u16,
) -> Result<(BoundaryEdge, BoundaryEdge), &'static str> {
    let distance = 4.0;
    let left = boundary_edge_on_side(
        image, coverage, ownership, seam, normal, -1.0, distance, left_owner,
    )?;
    let right = boundary_edge_on_side(
        image,
        coverage,
        ownership,
        seam,
        normal,
        1.0,
        distance,
        right_owner,
    )?;
    let left_length = left.gradient.0.hypot(left.gradient.1);
    let right_length = right.gradient.0.hypot(right.gradient.1);
    let cosine = (left.gradient.0 * right.gradient.0 + left.gradient.1 * right.gradient.1)
        / (left_length * right_length);
    let angle_error = cosine.clamp(-1.0, 1.0).acos().to_degrees();
    if angle_error > 10.0 {
        return Err(degradation::BOUNDARY_ORIENTATION_MISMATCH);
    }
    Ok((left, right))
}

fn boundary_pair_error_for_stations(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    left_station: usize,
    right_station: usize,
    owner_stations: &BTreeMap<u16, usize>,
) -> Result<(BoundaryEdge, BoundaryEdge), &'static str> {
    let distance = 4.0;
    let left = boundary_edge_on_side_matching(
        image,
        coverage,
        ownership,
        seam,
        normal,
        -1.0,
        distance,
        |owner| owner_stations.get(&owner).copied() == Some(left_station),
    )?;
    let right = boundary_edge_on_side_matching(
        image,
        coverage,
        ownership,
        seam,
        normal,
        1.0,
        distance,
        |owner| owner_stations.get(&owner).copied() == Some(right_station),
    )?;
    let left_length = left.gradient.0.hypot(left.gradient.1);
    let right_length = right.gradient.0.hypot(right.gradient.1);
    let cosine = (left.gradient.0 * right.gradient.0 + left.gradient.1 * right.gradient.1)
        / (left_length * right_length);
    let angle_error = cosine.clamp(-1.0, 1.0).acos().to_degrees();
    if angle_error > 10.0 {
        return Err(degradation::BOUNDARY_ORIENTATION_MISMATCH);
    }
    Ok((left, right))
}

fn parse_boundary_owner_pair(owner_path: &str) -> Option<(u16, u16)> {
    let owners = owner_path.split_once(";owner:")?.1;
    let (left, right) = owners.split_once("/owner:")?;
    Some((left.parse().ok()?, right.split(';').next()?.parse().ok()?))
}

fn parse_boundary_station_pair(owner_path: &str) -> Option<(usize, usize)> {
    let stations = owner_path.split_once("station:")?.1;
    let (left, right) = stations.split_once("/station:")?;
    // Quality_Gate emits the internal zero-based station indices directly.
    // The compositor's explicit environment selector is one-based, but the
    // report path is already normalized and must not be decremented again.
    let left = left.parse::<usize>().ok()?;
    let right = right.split(';').next()?.parse::<usize>().ok()?;
    (left != right).then_some((left, right))
}

/// Return the run-scoped local relation in the same direction as the
/// station pair written by the boundary criterion.  The relation is kept
/// outside the authoritative pose graph; it is only used for the bounded
/// raw->station reference closure below.
fn local_station_relation_for_pair(
    first_station: usize,
    second_station: usize,
) -> Option<Matrix3<f64>> {
    let topology = tone::run_topology_snapshot()?;
    let first_id = topology
        .stations
        .iter()
        .find(|(_, station)| *station == first_station)
        .map(|(id, _)| *id)?;
    let second_id = topology
        .stations
        .iter()
        .find(|(_, station)| *station == second_station)
        .map(|(id, _)| *id)?;
    let (left, right, bits) =
        topology
            .seam_candidate_transforms
            .iter()
            .find(|(left, right, _)| {
                (*left == first_id && *right == second_id)
                    || (*left == second_id && *right == first_id)
            })?;
    let relation = Matrix3::from_row_slice(&bits.map(f64::from_bits));
    if !relation.iter().all(|value| value.is_finite()) {
        return None;
    }
    if *left == first_id && *right == second_id {
        Some(relation)
    } else {
        relation.try_inverse()
    }
}

/// Reproduce a seam-only local relation while sampling a Quality_Gate raw
/// owner.  The authoritative global pose remains unchanged: the first
/// station is mapped normally, then its station-layer coordinate is sent
/// through the measured first->second relation before the second raw source
/// is sampled.  This mirrors `station_seam_pair_sample` in the compositor,
/// including the native displacement field, and closes the same geometric
/// loop that produced the boundary evidence.
fn sample_quality_source_at_world_with_station_relation(
    source: &Rgb32FImage,
    world: Point2<f64>,
    source_info: &QualitySource,
    first_source_info: &QualitySource,
    relation: &Matrix3<f64>,
    residual: &ResidualWarp,
) -> Result<Rgb<f32>, &'static str> {
    let first_warp = first_source_info
        .native_warp
        .as_ref()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let second_warp = source_info
        .native_warp
        .as_ref()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let first_inverse = first_warp
        .station_to_world
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let unwarped = residual.warp_inverse(world, first_source_info.geometry.station_id);
    let first_tile =
        project(&first_inverse, unwarped).ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let mapped = project(relation, first_tile).ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let delta = second_warp
        .displacement
        .as_ref()
        .map(|displacement| displacement.at(mapped.x, mapped.y))
        .unwrap_or([0.0, 0.0]);
    // `relation` is fitted in the Virtual_Tile coordinate system.  The
    // native warp stores the raw->world pose, while the Quality_Gate source
    // geometry also carries the station crop that maps that pose into the
    // tile-local plane.  Inverting only `source_to_station` drops that crop
    // translation and sends the relation sample to the wrong raw coordinates.
    // Compose the same raw->tile transform used by the ordinary native
    // reference path before sampling the second Source_RAW.
    let source_to_tile = second_warp
        .station_to_world
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?
        * source_info.geometry.tile_to_world
        * source_info.geometry.member_to_anchor;
    let source_inverse = source_to_tile
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let mapped_source = project(
        &source_inverse,
        Point2::new(mapped.x + delta[0], mapped.y + delta[1]),
    )
    .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let max_x = f64::from(source.width().saturating_sub(1));
    let max_y = f64::from(source.height().saturating_sub(1));
    if mapped_source.x < 0.0
        || mapped_source.y < 0.0
        || mapped_source.x > max_x
        || mapped_source.y > max_y
    {
        return Err(degradation::OWNER_SOURCE_UNDECODABLE);
    }
    Ok(
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            source,
            mapped_source.x,
            mapped_source.y,
        ),
    )
}

fn parse_boundary_sample_normal(owner_path: &str) -> Option<((f64, f64), (f64, f64))> {
    let samples = owner_path.split_once(";samples:")?.1;
    let (left, right) = samples.split_once('|')?;
    let parse_point = |value: &str| {
        let (x, y) = value.split_once(',')?;
        Some((x.parse::<f64>().ok()?, y.parse::<f64>().ok()?))
    };
    let left = parse_point(left)?;
    let right = parse_point(right)?;
    let dx = right.0 - left.0;
    let dy = right.1 - left.1;
    if dx.abs() < 1.0e-9 && dy.abs() < 1.0e-9 {
        return None;
    }
    let normal = if dy.abs() >= dx.abs() {
        (0.0, dy.signum())
    } else {
        (dx.signum(), 0.0)
    };
    Some((((left.0 + right.0) * 0.5, (left.1 + right.1) * 0.5), normal))
}

/// Apply a bounded, evidence-backed Source_RAW resample to only the five
/// native boundary samples that the current Quality_Gate reported as failed.
/// The owner ledger remains unchanged; this repairs a local station-layer
/// sampling offset while retaining the exact provenance and all gate limits.
/// A candidate is kept only when the same boundary metric strictly improves.
pub(crate) fn repair_failed_boundary_samples<F>(
    output: &mut Rgb32FImage,
    coverage: &GrayImage,
    ownership: &mut [u16],
    world_origin: (f64, f64),
    sources: &[QualitySource],
    residual: &ResidualWarp,
    report: &QualityGateReport,
    load_source: &mut F,
) -> Result<usize, String>
where
    F: FnMut(&QualitySource) -> Result<Rgb32FImage, String>,
{
    let Some(criterion) = report
        .criteria
        .iter()
        .find(|criterion| criterion.name == "boundary_stroke_alignment")
    else {
        return Ok(0);
    };
    if criterion.failed.is_empty() {
        return Ok(0);
    }
    let (width, height) = output.dimensions();
    if coverage.dimensions() != (width, height)
        || ownership.len() != width as usize * height as usize
    {
        return Err("boundary repair planes do not share one canvas".to_string());
    }
    let mut changed_pixels = 0usize;
    for failure in &criterion.failed {
        let Some(pair) = parse_boundary_owner_pair(&failure.owner_path) else {
            continue;
        };
        let Some((_, axis_normal)) = parse_boundary_sample_normal(&failure.owner_path) else {
            continue;
        };
        let seam = (
            failure.world.x - world_origin.0,
            failure.world.y - world_origin.1,
        );
        let mut oriented = None;
        for normal in [axis_normal, (-axis_normal.0, -axis_normal.1)] {
            if let Ok((left, right)) =
                boundary_pair_error(output, coverage, ownership, seam, normal, pair.0, pair.1)
            {
                oriented = Some((normal, left, right));
                break;
            }
        }
        let Some((normal, baseline_left, baseline_right)) = oriented else {
            if std::env::var_os("RAW_EDITOR_FOCUS_BOUNDARY_REPAIR").is_some() {
                eprintln!(
                    "  - Boundary repair skipped: no owner-cross evidence owners={}/{} world=({:.1},{:.1}) seam=({:.1},{:.1})",
                    pair.0, pair.1, failure.world.x, failure.world.y, seam.0, seam.1,
                );
            }
            continue;
        };
        let baseline_error = (baseline_left.offset - baseline_right.offset).abs();
        if std::env::var_os("RAW_EDITOR_FOCUS_BOUNDARY_REPAIR").is_some() {
            eprintln!(
                "  - Boundary repair baseline owners={}/{} world=({:.1},{:.1}) offsets={:.1}/{:.1} pixels={:?}/{:?} error={:.3}",
                pair.0,
                pair.1,
                failure.world.x,
                failure.world.y,
                baseline_left.offset,
                baseline_right.offset,
                baseline_left.pixel,
                baseline_right.pixel,
                baseline_error,
            );
        }
        let left_owner = pair.0;
        let right_owner = pair.1;
        let center = (seam.0.round() as i32, seam.1.round() as i32);
        if center.0 < 0 || center.1 < 0 || center.0 >= width as i32 || center.1 >= height as i32 {
            continue;
        }
        let mut source_images = BTreeMap::<u16, Rgb32FImage>::new();
        for owner in [left_owner, right_owner] {
            let Some(source_info) = sources.iter().find(|source| source.owner == owner) else {
                continue;
            };
            source_images.insert(owner, load_source(source_info)?);
        }
        let tangent = (-normal.1, normal.0);
        let mut best_error = baseline_error;
        let mut best_patch = Vec::<(usize, Rgb<f32>)>::new();
        let mut best_owner_patch = Vec::<(usize, Rgb<f32>, u16)>::new();
        let mut best_source_shift = None::<((i32, i32), (i32, i32))>;
        let mut best_local_relation = false;
        let debug_local_relation = std::env::var_os("RAW_EDITOR_FOCUS_BOUNDARY_DEBUG").is_some();
        let mut debug_local_relation_best = f64::INFINITY;
        let mut debug_local_relation_candidates = 0usize;
        let local_station_pair = parse_boundary_station_pair(&failure.owner_path);
        let local_relation = local_station_pair.and_then(|(first_station, second_station)| {
            local_station_relation_for_pair(first_station, second_station)
                .map(|relation| ((first_station, second_station), relation))
        });
        // The Quality_Gate owner pair is a raw-source pair, while the local
        // relation is published for its enclosing station pair.  Resolve the
        // two raw owners back to that direction before applying the relation
        // to the second station's bounded reference patch.
        let local_relation_sources =
            local_relation
                .as_ref()
                .and_then(|((first_station, second_station), relation)| {
                    let first_owner = [left_owner, right_owner].into_iter().find(|owner| {
                        sources
                            .iter()
                            .find(|source| source.owner == *owner)
                            .is_some_and(|source| source.geometry.station_id == *first_station)
                    })?;
                    let second_owner = [left_owner, right_owner].into_iter().find(|owner| {
                        sources
                            .iter()
                            .find(|source| source.owner == *owner)
                            .is_some_and(|source| source.geometry.station_id == *second_station)
                    })?;
                    let first_info = sources.iter().find(|source| source.owner == first_owner)?;
                    let second_info = sources.iter().find(|source| source.owner == second_owner)?;
                    let first_image = source_images.get(&first_owner)?;
                    let second_image = source_images.get(&second_owner)?;
                    Some((
                        relation,
                        first_owner,
                        second_owner,
                        first_info,
                        second_info,
                        first_image,
                        second_image,
                    ))
                });
        if let Some((
            relation,
            first_owner,
            second_owner,
            first_info,
            second_info,
            first_image,
            second_image,
        )) = local_relation_sources
        {
            if debug_local_relation {
                eprintln!(
                    "  - Boundary local relation sources stations={}/{} owners={}/{} relation=[{:.6},{:.6},{:.2};{:.6},{:.6},{:.2}]",
                    first_info.geometry.station_id,
                    second_info.geometry.station_id,
                    first_owner,
                    second_owner,
                    relation[(0, 0)],
                    relation[(0, 1)],
                    relation[(0, 2)],
                    relation[(1, 0)],
                    relation[(1, 1)],
                    relation[(1, 2)],
                );
            }
            let mut try_local_relation = |normal_shift: i32, tangent_shift: i32| {
                let mut original = Vec::<(usize, Rgb<f32>)>::new();
                let mut candidate = Vec::<(usize, Rgb<f32>)>::new();
                for tangent_offset in -16..=16 {
                    for normal_offset in -10..=10 {
                        let x = seam.0
                            + tangent.0 * f64::from(tangent_offset)
                            + normal.0 * f64::from(normal_offset);
                        let y = seam.1
                            + tangent.1 * f64::from(tangent_offset)
                            + normal.1 * f64::from(normal_offset);
                        let ix = x.round() as i32;
                        let iy = y.round() as i32;
                        if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
                            continue;
                        }
                        let index = iy as usize * width as usize + ix as usize;
                        if ownership.get(index).copied() != Some(second_owner)
                            || coverage.as_raw().get(index).copied().unwrap_or(0) == 0
                        {
                            continue;
                        }
                        let shifted_world = Point2::new(
                            world_origin.0
                                + f64::from(ix)
                                + normal.0 * f64::from(normal_shift)
                                + tangent.0 * f64::from(tangent_shift),
                            world_origin.1
                                + f64::from(iy)
                                + normal.1 * f64::from(normal_shift)
                                + tangent.1 * f64::from(tangent_shift),
                        );
                        let Ok(pixel) = sample_quality_source_at_world_with_station_relation(
                            &second_image,
                            shifted_world,
                            second_info,
                            first_info,
                            relation,
                            residual,
                        ) else {
                            continue;
                        };
                        if !original.iter().any(|(current, _)| *current == index) {
                            original.push((index, *output.get_pixel(ix as u32, iy as u32)));
                            candidate.push((index, pixel));
                            output.put_pixel(ix as u32, iy as u32, pixel);
                        }
                    }
                }
                if !original.is_empty() {
                    debug_local_relation_candidates =
                        debug_local_relation_candidates.saturating_add(1);
                    let error = boundary_pair_error(
                        output,
                        coverage,
                        ownership,
                        seam,
                        normal,
                        left_owner,
                        right_owner,
                    )
                    .map(|(left, right)| (left.offset - right.offset).abs())
                    .unwrap_or(f64::INFINITY);
                    debug_local_relation_best = debug_local_relation_best.min(error);
                    if error + 1.0e-6 < best_error {
                        best_error = error;
                        best_patch = candidate.clone();
                        best_owner_patch.clear();
                        best_source_shift = None;
                        best_local_relation = true;
                    }
                }
                for (index, pixel) in original {
                    let x = (index % width as usize) as u32;
                    let y = (index / width as usize) as u32;
                    output.put_pixel(x, y, pixel);
                }
            };
            for normal_shift in -4..=4 {
                for tangent_shift in -4..=4 {
                    try_local_relation(normal_shift, tangent_shift);
                }
            }
            // The station seam plan may be oriented in the reverse direction
            // from the Quality_Gate owner path.  The compositor applies the
            // fitted relation to its plan's second owner; when that owner is
            // the first station here, replay the inverse relation against the
            // first raw owner as well.  This keeps the local closure symmetric
            // without changing the global pose or owner ledger.
            drop(try_local_relation);
            if let Some(reverse_relation) = relation.try_inverse() {
                let mut try_reverse_relation = |normal_shift: i32, tangent_shift: i32| {
                    let mut original = Vec::<(usize, Rgb<f32>)>::new();
                    let mut candidate = Vec::<(usize, Rgb<f32>)>::new();
                    for tangent_offset in -16..=16 {
                        for normal_offset in -10..=10 {
                            let x = seam.0
                                + tangent.0 * f64::from(tangent_offset)
                                + normal.0 * f64::from(normal_offset);
                            let y = seam.1
                                + tangent.1 * f64::from(tangent_offset)
                                + normal.1 * f64::from(normal_offset);
                            let ix = x.round() as i32;
                            let iy = y.round() as i32;
                            if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
                                continue;
                            }
                            let index = iy as usize * width as usize + ix as usize;
                            if ownership.get(index).copied() != Some(first_owner)
                                || coverage.as_raw().get(index).copied().unwrap_or(0) == 0
                            {
                                continue;
                            }
                            let shifted_world = Point2::new(
                                world_origin.0
                                    + f64::from(ix)
                                    + normal.0 * f64::from(normal_shift)
                                    + tangent.0 * f64::from(tangent_shift),
                                world_origin.1
                                    + f64::from(iy)
                                    + normal.1 * f64::from(normal_shift)
                                    + tangent.1 * f64::from(tangent_shift),
                            );
                            let Ok(pixel) = sample_quality_source_at_world_with_station_relation(
                                &first_image,
                                shifted_world,
                                first_info,
                                second_info,
                                &reverse_relation,
                                residual,
                            ) else {
                                continue;
                            };
                            if !original.iter().any(|(current, _)| *current == index) {
                                original.push((index, *output.get_pixel(ix as u32, iy as u32)));
                                candidate.push((index, pixel));
                                output.put_pixel(ix as u32, iy as u32, pixel);
                            }
                        }
                    }
                    if !original.is_empty() {
                        debug_local_relation_candidates =
                            debug_local_relation_candidates.saturating_add(1);
                        let error = boundary_pair_error(
                            output,
                            coverage,
                            ownership,
                            seam,
                            normal,
                            left_owner,
                            right_owner,
                        )
                        .map(|(left, right)| (left.offset - right.offset).abs())
                        .unwrap_or(f64::INFINITY);
                        debug_local_relation_best = debug_local_relation_best.min(error);
                        if error + 1.0e-6 < best_error {
                            best_error = error;
                            best_patch = candidate.clone();
                            best_owner_patch.clear();
                            best_source_shift = None;
                            best_local_relation = true;
                        }
                    }
                    for (index, pixel) in original {
                        let x = (index % width as usize) as u32;
                        let y = (index / width as usize) as u32;
                        output.put_pixel(x, y, pixel);
                    }
                };
                for normal_shift in -4..=4 {
                    for tangent_shift in -4..=4 {
                        try_reverse_relation(normal_shift, tangent_shift);
                    }
                }
            }
            if debug_local_relation {
                eprintln!(
                    "  - Boundary local relation candidates={} best_error={:.3} baseline={:.3} selected={}",
                    debug_local_relation_candidates,
                    debug_local_relation_best,
                    baseline_error,
                    best_local_relation,
                );
            }
        }
        for owner in [left_owner, right_owner] {
            let Some(source_info) = sources.iter().find(|source| source.owner == owner) else {
                continue;
            };
            let Some(source_image) = source_images.get(&owner) else {
                continue;
            };
            // The reported error is measured along the boundary tangent.  A
            // normal-only sample shift can never repair a lateral owner edge
            // displacement, so search a bounded 2-D Source_RAW reprojection
            // around the exact failed sample.  The source ledger remains
            // unchanged and every candidate is accepted only by the native
            // owner-pair metric below.
            for normal_shift in -8..=8 {
                for tangent_shift in -8..=8 {
                    let mut original = Vec::<(usize, Rgb<f32>)>::new();
                    let mut candidate = Vec::<(usize, Rgb<f32>)>::new();
                    for tangent_offset in -12..=12 {
                        for normal_offset in -12..=12 {
                            let x = seam.0
                                + tangent.0 * f64::from(tangent_offset)
                                + normal.0 * f64::from(normal_offset);
                            let y = seam.1
                                + tangent.1 * f64::from(tangent_offset)
                                + normal.1 * f64::from(normal_offset);
                            let ix = x.round() as i32;
                            let iy = y.round() as i32;
                            if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
                                continue;
                            }
                            let index = iy as usize * width as usize + ix as usize;
                            if coverage.as_raw().get(index).copied().unwrap_or(0) == 0
                                || ownership.get(index).copied() != Some(owner)
                            {
                                continue;
                            }
                            let world = Point2::new(
                                world_origin.0
                                    + f64::from(ix)
                                    + normal.0 * f64::from(normal_shift)
                                    + tangent.0 * f64::from(tangent_shift),
                                world_origin.1
                                    + f64::from(iy)
                                    + normal.1 * f64::from(normal_shift)
                                    + tangent.1 * f64::from(tangent_shift),
                            );
                            let Ok(pixel) = sample_quality_source_at_world(
                                source_image,
                                world,
                                source_info,
                                residual,
                            ) else {
                                continue;
                            };
                            if !original.iter().any(|(current, _)| *current == index) {
                                original.push((index, *output.get_pixel(ix as u32, iy as u32)));
                                candidate.push((index, pixel));
                                output.put_pixel(ix as u32, iy as u32, pixel);
                            }
                        }
                    }
                    if original.is_empty() {
                        continue;
                    }
                    let error = boundary_pair_error(
                        output,
                        coverage,
                        ownership,
                        seam,
                        normal,
                        left_owner,
                        right_owner,
                    )
                    .map(|(left, right)| (left.offset - right.offset).abs())
                    .unwrap_or(f64::INFINITY);
                    if error + 1.0e-6 < best_error {
                        best_error = error;
                        best_patch = candidate.clone();
                        best_source_shift = None;
                    }
                    for (index, pixel) in original {
                        let x = (index % width as usize) as u32;
                        let y = (index / width as usize) as u32;
                        output.put_pixel(x, y, pixel);
                    }
                }
            }
        }
        // If each owner-only reprojected patch is still displaced, align the
        // two Source_RAW samples together and feather only this evidence
        // window.  This is the visual equivalent of a local seam blend: both
        // samples are evaluated in the same world frame, the relative shift
        // is bounded, and the patch is retained only when the unchanged
        // owner-pair boundary metric improves.
        let left_info = sources.iter().find(|source| source.owner == left_owner);
        let right_info = sources.iter().find(|source| source.owner == right_owner);
        let left_image = source_images.get(&left_owner);
        let right_image = source_images.get(&right_owner);
        if let (Some(left_info), Some(right_info), Some(left_image), Some(right_image)) =
            (left_info, right_info, left_image, right_image)
        {
            let mut try_aligned_blend = |left_shift: (i32, i32), right_shift: (i32, i32)| {
                let mut original = Vec::<(usize, Rgb<f32>)>::new();
                let mut candidate = Vec::<(usize, Rgb<f32>)>::new();
                for tangent_offset in -16..=16 {
                    for normal_offset in -10..=10 {
                        let x = seam.0
                            + tangent.0 * f64::from(tangent_offset)
                            + normal.0 * f64::from(normal_offset);
                        let y = seam.1
                            + tangent.1 * f64::from(tangent_offset)
                            + normal.1 * f64::from(normal_offset);
                        let ix = x.round() as i32;
                        let iy = y.round() as i32;
                        if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
                            continue;
                        }
                        let index = iy as usize * width as usize + ix as usize;
                        if coverage.as_raw().get(index).copied().unwrap_or(0) == 0
                            || !matches!(
                                ownership.get(index).copied(),
                                Some(owner) if owner == left_owner || owner == right_owner
                            )
                        {
                            continue;
                        }
                        let left_world = Point2::new(
                            world_origin.0
                                + f64::from(ix)
                                + normal.0 * f64::from(left_shift.0)
                                + tangent.0 * f64::from(left_shift.1),
                            world_origin.1
                                + f64::from(iy)
                                + normal.1 * f64::from(left_shift.0)
                                + tangent.1 * f64::from(left_shift.1),
                        );
                        let right_world = Point2::new(
                            world_origin.0
                                + f64::from(ix)
                                + normal.0 * f64::from(right_shift.0)
                                + tangent.0 * f64::from(right_shift.1),
                            world_origin.1
                                + f64::from(iy)
                                + normal.1 * f64::from(right_shift.0)
                                + tangent.1 * f64::from(right_shift.1),
                        );
                        let (Ok(left_pixel), Ok(right_pixel)) = (
                            sample_quality_source_at_world(
                                left_image, left_world, left_info, residual,
                            ),
                            sample_quality_source_at_world(
                                right_image,
                                right_world,
                                right_info,
                                residual,
                            ),
                        ) else {
                            continue;
                        };
                        let signed_normal = f64::from(normal_offset);
                        let right_weight = ((signed_normal + 8.0) / 16.0).clamp(0.0, 1.0) as f32;
                        let left_weight = 1.0 - right_weight;
                        let pixel = Rgb(std::array::from_fn(|channel| {
                            left_pixel[channel] * left_weight + right_pixel[channel] * right_weight
                        }));
                        if !original.iter().any(|(current, _)| *current == index) {
                            original.push((index, *output.get_pixel(ix as u32, iy as u32)));
                            candidate.push((index, pixel));
                            output.put_pixel(ix as u32, iy as u32, pixel);
                        }
                    }
                }
                if original.is_empty() {
                    return;
                }
                let error = boundary_pair_error(
                    output,
                    coverage,
                    ownership,
                    seam,
                    normal,
                    left_owner,
                    right_owner,
                )
                .map(|(left, right)| (left.offset - right.offset).abs())
                .unwrap_or(f64::INFINITY);
                if error + 1.0e-6 < best_error {
                    best_error = error;
                    best_patch = candidate.clone();
                    best_source_shift = None;
                }
                for (index, pixel) in original {
                    let x = (index % width as usize) as u32;
                    let y = (index / width as usize) as u32;
                    output.put_pixel(x, y, pixel);
                }
            };
            for shift_normal in -8..=8 {
                for shift_tangent in -8..=8 {
                    try_aligned_blend((0, 0), (shift_normal, shift_tangent));
                    try_aligned_blend((shift_normal, shift_tangent), (0, 0));
                }
            }
        }
        // Search a relative Source_RAW reprojection while retaining the hard
        // owner edge.  The earlier owner-only search can miss a pair whose two
        // local samples both need to move in relation to one another: the
        // measured 3px gap is a cross-source displacement, not a single-image
        // tone defect.  Fixing one side at zero and searching the other spans
        // the complete relative translation without introducing a tile-wide
        // pose change.
        if let (Some(left_info), Some(right_info), Some(left_image), Some(right_image)) = (
            sources.iter().find(|source| source.owner == left_owner),
            sources.iter().find(|source| source.owner == right_owner),
            source_images.get(&left_owner),
            source_images.get(&right_owner),
        ) {
            let mut try_pair_resample = |left_shift: (i32, i32), right_shift: (i32, i32)| {
                let mut original = Vec::<(usize, Rgb<f32>)>::new();
                let mut candidate = Vec::<(usize, Rgb<f32>)>::new();
                for tangent_offset in -16..=16 {
                    for normal_offset in -10..=10 {
                        let x = seam.0
                            + tangent.0 * f64::from(tangent_offset)
                            + normal.0 * f64::from(normal_offset);
                        let y = seam.1
                            + tangent.1 * f64::from(tangent_offset)
                            + normal.1 * f64::from(normal_offset);
                        let ix = x.round() as i32;
                        let iy = y.round() as i32;
                        if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
                            continue;
                        }
                        let index = iy as usize * width as usize + ix as usize;
                        let owner = ownership.get(index).copied();
                        let (source, source_info, shift) = match owner {
                            Some(owner) if owner == left_owner => {
                                (left_image, left_info, left_shift)
                            }
                            Some(owner) if owner == right_owner => {
                                (right_image, right_info, right_shift)
                            }
                            _ => continue,
                        };
                        if coverage.as_raw().get(index).copied().unwrap_or(0) == 0 {
                            continue;
                        }
                        let world = Point2::new(
                            world_origin.0
                                + f64::from(ix)
                                + normal.0 * f64::from(shift.0)
                                + tangent.0 * f64::from(shift.1),
                            world_origin.1
                                + f64::from(iy)
                                + normal.1 * f64::from(shift.0)
                                + tangent.1 * f64::from(shift.1),
                        );
                        let Ok(pixel) =
                            sample_quality_source_at_world(source, world, source_info, residual)
                        else {
                            continue;
                        };
                        if !original.iter().any(|(current, _)| *current == index) {
                            original.push((index, *output.get_pixel(ix as u32, iy as u32)));
                            candidate.push((index, pixel));
                            output.put_pixel(ix as u32, iy as u32, pixel);
                        }
                    }
                }
                if !original.is_empty() {
                    let error = boundary_pair_error(
                        output,
                        coverage,
                        ownership,
                        seam,
                        normal,
                        left_owner,
                        right_owner,
                    )
                    .map(|(left, right)| (left.offset - right.offset).abs())
                    .unwrap_or(f64::INFINITY);
                    if error + 1.0e-6 < best_error {
                        best_error = error;
                        best_patch = candidate.clone();
                        best_source_shift = Some((left_shift, right_shift));
                        best_owner_patch.clear();
                    }
                }
                for (index, pixel) in original {
                    let x = (index % width as usize) as u32;
                    let y = (index / width as usize) as u32;
                    output.put_pixel(x, y, pixel);
                }
            };
            for shift_normal in -8..=8 {
                for shift_tangent in -8..=8 {
                    let shift = (shift_normal, shift_tangent);
                    try_pair_resample((0, 0), shift);
                    try_pair_resample(shift, (0, 0));
                }
            }
        }
        // The failed evidence reports a lateral displacement of the two owner
        // transitions.  When reprojecting either Source_RAW (and even a
        // bounded aligned feather) leaves the measured pair unchanged, move
        // only this local owner transition to the image-supported side.  The
        // candidate keeps the same owner pair, preserves every third owner,
        // re-samples pixels that change provenance, and is accepted only when
        // the native boundary metric strictly improves at the candidate seam.
        // This is a local provenance repair, not a threshold or statistic
        // relaxation.
        let Some(left_info) = sources.iter().find(|source| source.owner == left_owner) else {
            continue;
        };
        let Some(right_info) = sources.iter().find(|source| source.owner == right_owner) else {
            continue;
        };
        let Some(left_image) = source_images.get(&left_owner) else {
            continue;
        };
        let Some(right_image) = source_images.get(&right_owner) else {
            continue;
        };
        let mut owner_sample_cache = BTreeMap::<usize, (Option<Rgb<f32>>, Option<Rgb<f32>>)>::new();
        let cache_radius_tangent = 24i32;
        let cache_radius_normal = 18i32;
        for tangent_offset in -cache_radius_tangent..=cache_radius_tangent {
            for normal_offset in -cache_radius_normal..=cache_radius_normal {
                let x = seam.0
                    + tangent.0 * f64::from(tangent_offset)
                    + normal.0 * f64::from(normal_offset);
                let y = seam.1
                    + tangent.1 * f64::from(tangent_offset)
                    + normal.1 * f64::from(normal_offset);
                let ix = x.round() as i32;
                let iy = y.round() as i32;
                if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
                    continue;
                }
                let index = iy as usize * width as usize + ix as usize;
                if coverage.as_raw().get(index).copied().unwrap_or(0) == 0
                    || !matches!(ownership.get(index).copied(), Some(owner) if owner == left_owner || owner == right_owner)
                {
                    continue;
                }
                let world = Point2::new(
                    world_origin.0 + f64::from(ix),
                    world_origin.1 + f64::from(iy),
                );
                let left_pixel =
                    sample_quality_source_at_world(left_image, world, left_info, residual).ok();
                let right_pixel =
                    sample_quality_source_at_world(right_image, world, right_info, residual).ok();
                owner_sample_cache.insert(index, (left_pixel, right_pixel));
            }
        }
        let mut best_owner_geometry = None::<(i32, i32)>;
        for seam_normal_shift in -8..=8 {
            // A straight normal translation cannot close a lateral 3px gap.
            // Search a small local tilt as well: in the normal/tangent frame
            // this is a seam whose normal coordinate changes with tangent
            // position.  The slope is bounded to avoid inventing a broad
            // warp outside the evidence window.
            for slope_numerator in -8..=8 {
                let slope = f64::from(slope_numerator) / 8.0;
                let candidate_seam = (
                    seam.0 + normal.0 * f64::from(seam_normal_shift),
                    seam.1 + normal.1 * f64::from(seam_normal_shift),
                );
                let mut original = Vec::<(usize, Rgb<f32>, u16)>::new();
                let mut candidate = Vec::<(usize, Rgb<f32>, u16)>::new();
                for tangent_offset in -cache_radius_tangent..=cache_radius_tangent {
                    for normal_offset in -cache_radius_normal..=cache_radius_normal {
                        let x = seam.0
                            + tangent.0 * f64::from(tangent_offset)
                            + normal.0 * f64::from(normal_offset);
                        let y = seam.1
                            + tangent.1 * f64::from(tangent_offset)
                            + normal.1 * f64::from(normal_offset);
                        let ix = x.round() as i32;
                        let iy = y.round() as i32;
                        if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
                            continue;
                        }
                        let index = iy as usize * width as usize + ix as usize;
                        let Some(current_owner) = ownership.get(index).copied() else {
                            continue;
                        };
                        if coverage.as_raw().get(index).copied().unwrap_or(0) == 0
                            || (current_owner != left_owner && current_owner != right_owner)
                        {
                            continue;
                        }
                        let delta_x = f64::from(ix) - candidate_seam.0;
                        let delta_y = f64::from(iy) - candidate_seam.1;
                        let signed_normal = delta_x * normal.0
                            + delta_y * normal.1
                            + slope * (delta_x * tangent.0 + delta_y * tangent.1);
                        let desired_owner = if signed_normal <= 0.0 {
                            left_owner
                        } else {
                            right_owner
                        };
                        if desired_owner == current_owner {
                            continue;
                        }
                        let Some((left_pixel, right_pixel)) = owner_sample_cache.get(&index) else {
                            continue;
                        };
                        let Some(pixel) = (if desired_owner == left_owner {
                            *left_pixel
                        } else {
                            *right_pixel
                        }) else {
                            continue;
                        };
                        original.push((
                            index,
                            *output.get_pixel(ix as u32, iy as u32),
                            current_owner,
                        ));
                        candidate.push((index, pixel, desired_owner));
                        output.put_pixel(ix as u32, iy as u32, pixel);
                        ownership[index] = desired_owner;
                    }
                }
                if !candidate.is_empty() {
                    let error = boundary_pair_error(
                        output,
                        coverage,
                        ownership,
                        candidate_seam,
                        normal,
                        left_owner,
                        right_owner,
                    )
                    .map(|(left, right)| (left.offset - right.offset).abs())
                    .unwrap_or(f64::INFINITY);
                    if error + 1.0e-6 < best_error {
                        best_error = error;
                        best_owner_patch = candidate.clone();
                        best_patch.clear();
                        best_source_shift = None;
                        best_owner_geometry = Some((seam_normal_shift, slope_numerator));
                    }
                }
                for (index, pixel, owner) in original {
                    let x = (index % width as usize) as u32;
                    let y = (index / width as usize) as u32;
                    output.put_pixel(x, y, pixel);
                    ownership[index] = owner;
                }
            }
        }
        if !best_owner_patch.is_empty() {
            changed_pixels = changed_pixels.saturating_add(best_owner_patch.len());
            for (index, pixel, owner) in best_owner_patch {
                let x = (index % width as usize) as u32;
                let y = (index / width as usize) as u32;
                output.put_pixel(x, y, pixel);
                ownership[index] = owner;
            }
            println!(
                "  - Boundary owner seam relocation: owners={}/{} world=({:.1},{:.1}) error {:.3}->{:.3} pixels={}",
                left_owner,
                right_owner,
                failure.world.x,
                failure.world.y,
                baseline_error,
                best_error,
                changed_pixels,
            );
            if let Some((left_shift, right_shift)) = best_source_shift {
                eprintln!(
                    "  - Boundary source relative shift: left=({}, {}) right=({}, {})",
                    left_shift.0, left_shift.1, right_shift.0, right_shift.1,
                );
            }
            if let Some((normal_shift, slope_numerator)) = best_owner_geometry {
                eprintln!(
                    "  - Boundary owner seam geometry: normal_shift={} slope={}/8",
                    normal_shift, slope_numerator,
                );
            }
        } else if !best_patch.is_empty() {
            changed_pixels = changed_pixels.saturating_add(best_patch.len());
            for (index, pixel) in best_patch {
                let x = (index % width as usize) as u32;
                let y = (index / width as usize) as u32;
                output.put_pixel(x, y, pixel);
            }
            println!(
                "  - Boundary source resample: owners={}/{} world=({:.1},{:.1}) error {:.3}->{:.3} pixels={}",
                left_owner,
                right_owner,
                failure.world.x,
                failure.world.y,
                baseline_error,
                best_error,
                changed_pixels,
            );
        } else if std::env::var_os("RAW_EDITOR_FOCUS_BOUNDARY_REPAIR").is_some() {
            eprintln!(
                "  - Boundary repair found no improving source sample owners={}/{} world=({:.1},{:.1}) baseline={:.3}",
                left_owner, right_owner, failure.world.x, failure.world.y, baseline_error,
            );
        }
    }
    Ok(changed_pixels)
}

fn source_contains_world(
    source: &QualitySource,
    world: (f64, f64),
    residual: &ResidualWarp,
) -> bool {
    quality_gate::map_world_to_source(Point2::new(world.0, world.1), &source.geometry, residual)
        .ok()
        .is_some_and(|point| {
            point.x >= 0.0
                && point.y >= 0.0
                && point.x < f64::from(source.dimensions.0)
                && point.y < f64::from(source.dimensions.1)
        })
}

fn boundary_rejection_priority(reason: &str) -> u8 {
    match reason {
        degradation::BOUNDARY_PROVENANCE_UNAVAILABLE => 4,
        degradation::BOUNDARY_ORIENTATION_MISMATCH => 3,
        degradation::BOUNDARY_STROKE_NOT_UNIQUE => 2,
        degradation::BOUNDARY_LOW_CONTRAST => 1,
        _ => 0,
    }
}

/// Measure only accepted inter-station hard seams. Each 64px world cell can
/// contribute at most one independently detected two-sided stroke pair; this
/// keeps the sample population spatially deterministic and prevents jagged
/// one-pixel owner transitions from inflating the criterion.
fn measure_large_boundaries(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    origin: (f64, f64),
    sources: &[QualitySource],
    accepted_station_relations: &[(usize, usize)],
    residual: &ResidualWarp,
    criterion: &mut Criterion,
    unmeasurable: &mut Vec<UnmeasurableRecord>,
) {
    const SPATIAL_DEDUP_PX: u32 = 64;
    let (width, height) = coverage.dimensions();
    let width_usize = width as usize;
    let source_by_owner = sources
        .iter()
        .map(|source| (source.owner, source))
        .collect::<BTreeMap<_, _>>();
    // A station can contribute several raw owners after native tiling and
    // overlap arbitration.  Boundary evidence is a station-level geometric
    // criterion, so all raw owners belonging to the same station must be
    // eligible for edge detection.  Keep the selected raw owner on each edge
    // for provenance and source containment checks below.
    let station_by_owner = sources
        .iter()
        .map(|source| (source.owner, source.geometry.station_id))
        .collect::<BTreeMap<_, _>>();
    let accepted = accepted_station_relations
        .iter()
        .filter_map(|&(left, right)| (left != right).then_some((left.min(right), left.max(right))))
        .collect::<std::collections::BTreeSet<_>>();
    let mut cells = BTreeMap::<(usize, usize, u32, u32), BoundaryCellObservation>::new();
    let mut provenance_unavailable = false;
    let mut visit = |x: u32, y: u32, normal: (f64, f64), left_owner: u16, right_owner: u16| {
        if left_owner == 0 || right_owner == 0 || left_owner == right_owner {
            return;
        }
        let (Some(left_source), Some(right_source)) = (
            source_by_owner.get(&left_owner).copied(),
            source_by_owner.get(&right_owner).copied(),
        ) else {
            provenance_unavailable = true;
            return;
        };
        let station_pair = (
            left_source
                .geometry
                .station_id
                .min(right_source.geometry.station_id),
            left_source
                .geometry
                .station_id
                .max(right_source.geometry.station_id),
        );
        if station_pair.0 == station_pair.1 || !accepted.contains(&station_pair) {
            return;
        }
        let seam = (f64::from(x) + normal.0 * 0.5, f64::from(y) + normal.1 * 0.5);
        let world = (origin.0 + seam.0, origin.1 + seam.1);
        let seam_key = (
            station_pair.0,
            station_pair.1,
            x / SPATIAL_DEDUP_PX,
            y / SPATIAL_DEDUP_PX,
        );
        let (key, observation) = match boundary_pair_error_for_stations(
            image,
            coverage,
            ownership,
            seam,
            normal,
            left_source.geometry.station_id,
            right_source.geometry.station_id,
            &station_by_owner,
        ) {
            Ok((left_edge, right_edge)) => {
                let left_world = (
                    origin.0 + f64::from(left_edge.pixel.0),
                    origin.1 + f64::from(left_edge.pixel.1),
                );
                let right_world = (
                    origin.0 + f64::from(right_edge.pixel.0),
                    origin.1 + f64::from(right_edge.pixel.1),
                );
                let stroke_key = (
                    station_pair.0,
                    station_pair.1,
                    (left_edge.pixel.0 / 2 + right_edge.pixel.0 / 2) / SPATIAL_DEDUP_PX,
                    (left_edge.pixel.1 / 2 + right_edge.pixel.1 / 2) / SPATIAL_DEDUP_PX,
                );
                let selected_left_source = source_by_owner
                    .get(&left_edge.owner)
                    .copied()
                    .unwrap_or(left_source);
                let selected_right_source = source_by_owner
                    .get(&right_edge.owner)
                    .copied()
                    .unwrap_or(right_source);
                if !source_contains_world(selected_left_source, left_world, residual)
                    || !source_contains_world(selected_right_source, right_world, residual)
                {
                    (
                        stroke_key,
                        BoundaryCellObservation::Unmeasurable {
                            world,
                            reason: degradation::BOUNDARY_PROVENANCE_UNAVAILABLE,
                        },
                    )
                } else {
                    (
                        stroke_key,
                        BoundaryCellObservation::Measured(BoundaryPairEvidence {
                            world,
                            error: (left_edge.offset - right_edge.offset).abs(),
                            strength: left_edge.strength.min(right_edge.strength),
                            owner_path: format!(
                                "station:{}/station:{};owner:{}/owner:{};samples:{:.1},{:.1}|{:.1},{:.1}",
                                station_pair.0,
                                station_pair.1,
                                left_edge.owner,
                                right_edge.owner,
                                left_world.0,
                                left_world.1,
                                right_world.0,
                                right_world.1,
                            ),
                        }),
                    )
                }
            }
            Err(reason) => (
                seam_key,
                BoundaryCellObservation::Unmeasurable { world, reason },
            ),
        };
        match (cells.get(&key), &observation) {
            (
                Some(BoundaryCellObservation::Measured(current)),
                BoundaryCellObservation::Measured(candidate),
            ) if candidate.strength > current.strength => {
                cells.insert(key, observation);
            }
            (Some(BoundaryCellObservation::Measured(_)), _) => {}
            (
                Some(BoundaryCellObservation::Unmeasurable {
                    reason: current, ..
                }),
                BoundaryCellObservation::Unmeasurable {
                    reason: candidate, ..
                },
            ) if boundary_rejection_priority(candidate) > boundary_rejection_priority(current) => {
                cells.insert(key, observation);
            }
            (_, BoundaryCellObservation::Measured(_)) | (None, _) => {
                cells.insert(key, observation);
            }
            _ => {}
        }
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

    let mut errors = cells
        .values()
        .filter_map(|observation| match observation {
            BoundaryCellObservation::Measured(evidence) => Some(evidence.error),
            BoundaryCellObservation::Unmeasurable { .. } => None,
        })
        .collect::<Vec<_>>();
    errors.sort_by(f64::total_cmp);
    let p95 = (!errors.is_empty()).then(|| {
        errors[(errors.len() * 95)
            .div_ceil(100)
            .saturating_sub(1)
            .min(errors.len() - 1)]
    });
    for observation in cells.into_values() {
        match observation {
            BoundaryCellObservation::Measured(evidence) => {
                let p95_failed = p95.is_some_and(|value| {
                    value > quality_gate::QUALITY_BOUNDARY_P95_MAX && evidence.error >= value
                });
                criterion.push(
                    evidence.world,
                    evidence.error,
                    &evidence.owner_path,
                    evidence.error <= quality_gate::QUALITY_BOUNDARY_MAX && !p95_failed,
                );
            }
            BoundaryCellObservation::Unmeasurable { world, reason } => {
                criterion.miss(reason);
                unmeasurable.push(UnmeasurableRecord {
                    criterion: criterion.name.to_string(),
                    world: WorldPoint {
                        x: world.0,
                        y: world.1,
                    },
                    reason: reason.to_string(),
                    category: classify_unmeasurable_reason(reason),
                });
            }
        }
    }
    if criterion.observed_count == 0 && criterion.unmeasurable == 0 {
        let reason = if provenance_unavailable {
            degradation::BOUNDARY_PROVENANCE_UNAVAILABLE
        } else {
            degradation::BOUNDARY_NO_PAIRABLE_EDGE
        };
        criterion.miss(reason);
        unmeasurable.push(UnmeasurableRecord {
            criterion: criterion.name.to_string(),
            world: WorldPoint {
                x: origin.0,
                y: origin.1,
            },
            reason: reason.to_string(),
            category: classify_unmeasurable_reason(reason),
        });
    }
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
            let reference = source_roi_with_native(
                &source,
                roi,
                &source_info.geometry,
                input.residual,
                source_info.native_warp.as_ref(),
            );
            let (Some(output_roi), Ok(mut reference)) = (output_roi, reference) else {
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
            apply_run_tone_transform(&mut reference, source_info.geometry.station_id);
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
                match gradient_energy_ratio(&output_roi, &reference, local) {
                    Ok(Some((value, pass))) => {
                        gradient.push(where_, value, &source_info.path, pass);
                    }
                    Ok(None) => {
                        record_unmeasurable(
                            &mut gradient,
                            &mut unmeasurable,
                            roi,
                            "gradient_reference_zero",
                        );
                    }
                    Err(reason) => {
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

    #[cfg(test)]
    write_tone_owner_fidelity_oracle(input, &rois, load_source)
        .expect("write test-only current-run tone owner Source_RAW fidelity oracle");

    #[cfg(test)]
    write_gradient_source_oracle(input, &rois, load_source)
        .expect("write test-only gradient source oracle");

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
    #[cfg(test)]
    write_effective_coverage_diagnostics(input.coverage, &quads, input.world_origin);
    timing.effective_coverage_seconds = effective_started.elapsed().as_secs_f64();
    match effective_result {
        Ok(EffectivePixelCoverage {
            output_nontransparent,
            projected_union,
            ratio,
        }) => {
            effective.effective_output_pixels = Some(output_nontransparent);
            effective.effective_projected_pixels = Some(projected_union);
            effective.push(
                input.world_origin,
                ratio,
                "",
                ratio >= quality_gate::QUALITY_EFFECTIVE_PIXEL_RATIO_MIN,
            );
        }
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
            input.sources,
            input.accepted_station_relations,
            input.residual,
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

    fn synthetic_owner_evidence(
        baseline: [f32; 3],
        candidate: [f32; 3],
        reference: [f32; 3],
        fit_disjoint: bool,
    ) -> ToneOwnerEvidence {
        ToneOwnerEvidence {
            partition: "owner_raw_fidelity_holdout",
            roi: QualityRoi {
                x: 0,
                y: 0,
                side: 64,
                region_label: 0,
                owner: 1,
                world_origin: (0.0, 0.0),
                boundary_clearance: 0.0,
            },
            owner_path: "synthetic.raw".to_string(),
            station_id: 0,
            tile_index: 0,
            baseline_mean: baseline,
            candidate_mean: candidate,
            reference_mean: reference,
            field_gain: [1.0; 3],
            global_gain: [1.0; 3],
            global_bias: [0.0; 3],
            coverage_min: 255,
            confidence_min: 1.0,
            confidence_mean: 1.0,
            overlap_participation: false,
            fit_disjoint,
            gradient_ratio_candidate_to_source: Some(1.0),
            noise_ratio_candidate_to_source: Some(1.0),
        }
    }

    #[test]
    fn owner_fidelity_rejects_joint_colour_shift_and_fit_leakage() {
        let shifted = synthetic_owner_evidence([0.4; 3], [0.5; 3], [0.4; 3], true);
        assert!(owner_distribution_feasible(&[&shifted], 0.0));
        assert!(!owner_distribution_feasible(&[&shifted], 1.0));

        let leaked = synthetic_owner_evidence([0.45; 3], [0.4; 3], [0.4; 3], false);
        assert!(!owner_distribution_feasible(&[&leaked], 1.0));
    }

    #[test]
    fn joint_oracle_recovers_feasible_solution_or_certifies_pareto_gap() {
        let recovered = select_joint_relaxation(&[0.5, 0.75, 1.0], &[0.0, 0.25, 0.5, 0.75]);
        assert_eq!(recovered.selected, Some(0.5));
        assert_eq!(recovered.pareto_gap, None);

        let impossible = select_joint_relaxation(&[0.75, 1.0], &[0.0, 0.25, 0.5]);
        assert_eq!(impossible.selected, None);
        assert_eq!(impossible.minimum_boundary, Some(0.75));
        assert_eq!(impossible.maximum_owner, Some(0.5));
        assert_eq!(impossible.pareto_gap, Some(0.25));
        assert_eq!(impossible.selected.unwrap_or(0.0), 0.0);
    }

    #[test]
    fn low_frequency_candidate_preserves_high_frequency_gradient_and_noise() {
        let baseline = Rgb32FImage::from_fn(64, 64, |x, y| {
            let fine = if (x + y).is_multiple_of(2) {
                0.03
            } else {
                -0.03
            };
            let broad = 0.25 + 0.003 * x as f32 + 0.002 * y as f32;
            Rgb([broad + fine, broad * 0.9 + fine, broad * 0.8 + fine])
        });
        let candidate = Rgb32FImage::from_fn(64, 64, |x, y| {
            let pixel = baseline.get_pixel(x, y).0;
            let low = 0.02 + 0.0001 * x as f32;
            Rgb(pixel.map(|value| value + low))
        });
        let gradient = quality_gate::normalized_gradient_energy(&candidate, 1.0).unwrap()
            / quality_gate::normalized_gradient_energy(&baseline, 1.0).unwrap();
        let noise = quality_gate::noise_sigma_ratio(&candidate, &baseline).unwrap();
        assert!(gradient >= quality_gate::QUALITY_GRADIENT_RATIO_MIN);
        assert!(
            (quality_gate::QUALITY_NOISE_RATIO_MIN..=quality_gate::QUALITY_NOISE_RATIO_MAX)
                .contains(&noise)
        );
    }

    #[test]
    fn owner_oracle_has_no_previous_quality_gate_list_input() {
        let source = include_str!("quality_gate_runner.rs");
        let start = source.find("fn write_tone_owner_fidelity_oracle").unwrap();
        let end = source[start..]
            .find("pub(super) fn record_unmeasurable")
            .map(|offset| start + offset)
            .unwrap();
        let oracle = &source[start..end];
        assert!(!oracle.contains("std::fs::read"));
        assert!(!oracle.contains("GRADIENT_ORACLE_REPORT"));
        assert!(!oracle.contains("fixture"));
        assert!(oracle.contains("current-run tone owner oracle snapshot"));
    }

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
    fn reference_roi_reuses_production_sampler_for_gradient_and_noise_metrics() {
        let source = Rgb32FImage::from_fn(12, 12, |x, y| {
            let value = ((x * x + 5 * x * y + 3 * y + 7) % 23) as f32 / 22.0;
            Rgb([value, value * 0.91, value * 0.83])
        });
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 4,
            region_label: 1,
            owner: 1,
            world_origin: (2.25, 2.5),
            boundary_clearance: 16.0,
        };
        let geometry = SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 0,
        };
        let reference =
            source_roi(&source, &roi, &geometry, &ResidualWarp::new()).expect("identity ROI");
        let source_info = QualitySource {
            owner: 1,
            path: "owner.raw".to_string(),
            geometry: geometry.clone(),
            dimensions: source.dimensions(),
            native_warp: None,
        };
        assert_eq!(
            sample_quality_source_at_world(
                &source,
                Point2::new(roi.world_origin.0, roi.world_origin.1),
                &source_info,
                &ResidualWarp::new(),
            )
            .unwrap(),
            *reference.get_pixel(0, 0),
        );
        let expected = crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            &source, 2.25, 2.5,
        );
        assert_eq!(reference.get_pixel(0, 0), &expected);

        let x0 = 2u32;
        let y0 = 2u32;
        let tx = 0.25f32;
        let ty = 0.5f32;
        let a = *source.get_pixel(x0, y0);
        let b = *source.get_pixel(x0 + 1, y0);
        let c = *source.get_pixel(x0, y0 + 1);
        let d = *source.get_pixel(x0 + 1, y0 + 1);
        let bilinear = Rgb(std::array::from_fn(|channel| {
            let top = a[channel] * (1.0 - tx) + b[channel] * tx;
            let bottom = c[channel] * (1.0 - tx) + d[channel] * tx;
            top * (1.0 - ty) + bottom * ty
        }));
        assert!(
            (reference.get_pixel(0, 0)[0] - bilinear[0]).abs() > 1.0e-6,
            "reference ROI must not silently fall back to the old bilinear kernel"
        );

        let output = Rgb32FImage::from_fn(roi.side, roi.side, |x, y| {
            crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                &source,
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            )
        });
        let local_scale = 1.125;
        let (gradient_ratio, gradient_passes) =
            gradient_energy_ratio(&output, &reference, local_scale)
                .expect("the sampled ROIs must remain measurable")
                .expect("the reference gradient must remain non-zero");
        assert!((gradient_ratio - 1.0).abs() <= 1.0e-12);
        assert!(gradient_passes);
        let noise_ratio = quality_gate::noise_sigma_ratio(&output, &reference)
            .expect("identical production and reference sampling must have measurable noise");
        assert!((noise_ratio - 1.0).abs() <= 1.0e-12);
    }

    #[test]
    fn owner_fidelity_mapping_composes_member_tile_and_requires_one_source() {
        let source = Rgb32FImage::from_fn(32, 32, |x, y| {
            let value = (x + 3 * y) as f32 / 124.0;
            Rgb([value, value * 0.9, value * 0.8])
        });
        let translate = |x: f64, y: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0);
        let geometry = SourceGeometry {
            member_to_anchor: translate(5.0, 7.0),
            tile_to_world: translate(11.0, 13.0),
            station_id: 4,
        };
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 2,
            region_label: 1,
            owner: 9,
            world_origin: (18.25, 23.5),
            boundary_clearance: 16.0,
        };
        let sampled = source_roi(&source, &roi, &geometry, &ResidualWarp::new()).unwrap();
        let expected = crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            &source, 2.25, 3.5,
        );
        assert_eq!(sampled.get_pixel(0, 0), &expected);

        let info = QualitySource {
            owner: 9,
            path: "owner.raw".to_string(),
            geometry,
            dimensions: source.dimensions(),
            native_warp: None,
        };
        assert_eq!(
            unique_owner_source(std::slice::from_ref(&info), 9)
                .unwrap()
                .path,
            "owner.raw"
        );
        assert!(unique_owner_source(&[info.clone(), info], 9).is_err());
        assert!(unique_owner_source(&[], 9).is_err());
    }

    #[test]
    fn source_roi_mapping_guard_requires_render_provenance_for_focus_band_and_native_delta() {
        // The production Quality_Gate mapping is intentionally only the
        // world -> Residual_Warp inverse -> member/tile inverse path.  A
        // focus render can apply two additional mappings after that path:
        // a local band homography and the accepted native displacement field.
        // Keep a coordinate-coded source so a future provenance integration
        // can compare the exact sample, rather than only comparing geometry
        // matrices.
        let source = Rgb32FImage::from_fn(48, 48, |x, y| {
            Rgb([x as f32 / 48.0, y as f32 / 48.0, (x + y) as f32 / 96.0])
        });
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 2,
            region_label: 1,
            owner: 1,
            world_origin: (10.25, 11.5),
            boundary_clearance: 16.0,
        };
        let geometry = SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 10,
        };
        let qg_reference = source_roi(&source, &roi, &geometry, &ResidualWarp::new())
            .expect("identity world geometry must remain measurable");

        // `render_focus_layer` stores a source->world local band homography;
        // translation(+2, 0) therefore maps a world sample back to source x-2.
        // `warp_rendered_focus_layer` samples the candidate at +1 canvas pixel
        // for this synthetic native field, yielding a final source x-1.  These
        // values model the station 10/12 issue without adding a second,
        // production mapping implementation to the Quality_Gate.
        let world = Point2::new(roi.world_origin.0, roi.world_origin.1);
        let expected_with_render_provenance =
            crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                &source,
                world.x - 1.0,
                world.y,
            );
        let qg_pixel = qg_reference.get_pixel(0, 0);
        assert_ne!(qg_pixel, &expected_with_render_provenance);
        assert!(
            (qg_pixel[0] - expected_with_render_provenance[0]).abs() > 1.0e-5,
            "the current QG reference must not silently claim local-band/native provenance"
        );
    }

    #[test]
    fn aligned_gradient_ratio_is_invariant_to_shared_local_scale() {
        let output = Rgb32FImage::from_fn(16, 16, |x, y| {
            let value = ((x * x + 3 * y + x * y) % 29) as f32 / 28.0;
            Rgb([value, value * 0.9, value * 0.8])
        });
        let reference = Rgb32FImage::from_fn(16, 16, |x, y| {
            let value = ((2 * x * x + y + x * y) % 31) as f32 / 30.0;
            Rgb([value, value * 0.9, value * 0.8])
        });
        let unit = gradient_energy_ratio(&output, &reference, 1.0)
            .unwrap()
            .unwrap()
            .0;
        for local_scale in [0.8, 1.125] {
            let scaled = gradient_energy_ratio(&output, &reference, local_scale)
                .unwrap()
                .unwrap()
                .0;
            assert!((scaled - unit).abs() <= 1.0e-12);
        }
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
            native_warp: None,
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
            accepted_station_relations: &[],
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
            native_warp: None,
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
            accepted_station_relations: &[],
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
        let effective = report
            .criteria
            .iter()
            .find(|criterion| criterion.name == "effective_pixel_count")
            .expect("effective-pixel criterion must be reported");
        assert_eq!(effective.effective_output_pixels, Some(1024 * 1024));
        // `source_quad` describes the inclusive last source sample at
        // `width - 1`/`height - 1`; the half-open scanline union therefore
        // covers 1023×1023 projected cells for this identity fixture.
        assert_eq!(effective.effective_projected_pixels, Some(1023 * 1023));
        // Record-only means the gate writes no ledger entry of its own.
        assert!(degradation::run_ledger_snapshot().entries().is_empty());
    }

    #[test]
    fn failed_quality_gate_is_promoted_to_export_rejection_at_boundary() {
        let _run_scope = degradation::begin_run_scope();
        degradation::reset_run_ledger();
        let mut report = QualityGateReport {
            verdict: QualityGateVerdict::Fail,
            roi_count: 1,
            ..QualityGateReport::default()
        };
        report.criteria.push(QualityGateCriterionRecord {
            name: "owner_sharpness_coverage".to_string(),
            verdict: QualityGateVerdict::Fail,
            measurable_count: 1,
            ..QualityGateCriterionRecord::default()
        });

        record_gate_outcome(&report);

        let ledger = degradation::run_ledger_snapshot();
        assert_eq!(
            ledger.count_of(degradation::QUALITY_GATE_CRITERION_FAILED),
            1
        );
        assert_eq!(ledger.entries()[0].detail["roi_count"], 1);
        assert_eq!(
            ledger.entries()[0].detail["criteria"][0]["name"],
            "owner_sharpness_coverage"
        );
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

    fn boundary_sources(width: u32, height: u32) -> [QualitySource; 2] {
        [
            QualitySource {
                owner: 1,
                path: "left.raw".to_string(),
                geometry: SourceGeometry {
                    member_to_anchor: Matrix3::identity(),
                    tile_to_world: Matrix3::identity(),
                    station_id: 0,
                },
                dimensions: (width, height),
                native_warp: None,
            },
            QualitySource {
                owner: 2,
                path: "right.raw".to_string(),
                geometry: SourceGeometry {
                    member_to_anchor: Matrix3::identity(),
                    tile_to_world: Matrix3::identity(),
                    station_id: 1,
                },
                dimensions: (width, height),
                native_warp: None,
            },
        ]
    }

    fn vertical_seam_owners(width: u32, height: u32, seam_x: u32) -> Vec<u16> {
        (0..width * height)
            .map(|index| if index % width < seam_x { 1 } else { 2 })
            .collect()
    }

    fn fragmented_vertical_seam_owners(width: u32, height: u32, seam_x: u32) -> Vec<u16> {
        (0..width * height)
            .map(|index| {
                let x = index % width;
                if x < seam_x.saturating_sub(2) {
                    1
                } else if x < seam_x {
                    3
                } else {
                    2
                }
            })
            .collect()
    }

    fn crossing_stroke(
        width: u32,
        height: u32,
        seam_x: u32,
        left_y: u32,
        right_y: u32,
        invert_right: bool,
    ) -> Rgb32FImage {
        Rgb32FImage::from_fn(width, height, |x, y| {
            let threshold = if x < seam_x { left_y } else { right_y };
            let high = if x >= seam_x && invert_right {
                y < threshold
            } else {
                y >= threshold
            };
            Rgb([if high { 0.9 } else { 0.1 }; 3])
        })
    }

    fn boundary_report(
        output: &Rgb32FImage,
        coverage: &GrayImage,
        ownership: &[u16],
        sources: &[QualitySource],
        accepted: &[(usize, usize)],
    ) -> QualityGateCriterionRecord {
        let residual = ResidualWarp::new();
        let mut criterion = Criterion::p95_with_max("boundary_stroke_alignment", 1.5, 3.0);
        let mut records = Vec::new();
        measure_large_boundaries(
            output,
            coverage,
            ownership,
            (0.0, 0.0),
            sources,
            accepted,
            &residual,
            &mut criterion,
            &mut records,
        );
        criterion.finish(&mut records)
    }

    #[test]
    fn accepted_seam_pairs_independently_detected_crossing_strokes() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 82, false);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = vertical_seam_owners(width, height, seam_x);
        let report = boundary_report(
            &output,
            &coverage,
            &ownership,
            &boundary_sources(width, height),
            &[(0, 1)],
        );
        assert_eq!(report.measurable_count, 1);
        assert_eq!(report.measured[0].measured, 2.0);
        assert!(
            report.measured[0]
                .owner_path
                .contains("station:0/station:1")
        );
        assert!(report.measured[0].owner_path.contains("samples:"));
    }

    #[test]
    fn station_boundary_collapses_fragmented_raw_owners() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 82, false);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let mut sources = boundary_sources(width, height).to_vec();
        sources.push(QualitySource {
            owner: 3,
            path: "left-overlap.raw".to_string(),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 0,
            },
            dimensions: (width, height),
            native_warp: None,
        });
        let ownership = fragmented_vertical_seam_owners(width, height, seam_x);
        let report = boundary_report(&output, &coverage, &ownership, &sources, &[(0, 1)]);
        assert_eq!(report.measurable_count, 1);
        assert_eq!(report.measured[0].measured, 2.0);
        assert!(report.measured[0].owner_path.contains("owner:1/owner:2"));
    }

    #[test]
    fn nearby_topology_transitions_do_not_duplicate_one_stroke() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 80, false);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = vertical_seam_owners(width, height, seam_x);
        let sources = boundary_sources(width, height);
        let first = boundary_report(&output, &coverage, &ownership, &sources, &[(0, 1)]);
        let second = boundary_report(&output, &coverage, &ownership, &sources, &[(1, 0)]);
        assert_eq!(first.measurable_count, 1);
        assert_eq!(
            first, second,
            "relation order must not change deterministic evidence"
        );
    }

    #[test]
    fn sampling_is_restricted_to_accepted_cross_station_hard_seams() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 80, false);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = vertical_seam_owners(width, height, seam_x);
        let sources = boundary_sources(width, height);
        let report = boundary_report(&output, &coverage, &ownership, &sources, &[]);
        assert_eq!(report.measurable_count, 0);
        assert_eq!(
            report.unmeasurable_reasons[degradation::BOUNDARY_NO_PAIRABLE_EDGE],
            1
        );
    }

    #[test]
    fn opposite_polarity_strokes_are_rejected_as_wrong_correspondences() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 80, true);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = vertical_seam_owners(width, height, seam_x);
        let report = boundary_report(
            &output,
            &coverage,
            &ownership,
            &boundary_sources(width, height),
            &[(0, 1)],
        );
        assert_eq!(report.measurable_count, 0);
        assert!(
            report
                .unmeasurable_reasons
                .contains_key(degradation::BOUNDARY_ORIENTATION_MISMATCH)
        );
    }

    #[test]
    fn wider_wrong_edge_is_rejected_by_fixed_cross_section() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 70, 82, false);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = vertical_seam_owners(width, height, seam_x);
        let report = boundary_report(
            &output,
            &coverage,
            &ownership,
            &boundary_sources(width, height),
            &[(0, 1)],
        );
        assert_eq!(report.measurable_count, 0);
        assert!(
            report
                .unmeasurable_reasons
                .contains_key(degradation::BOUNDARY_LOW_CONTRAST)
        );
    }

    #[test]
    fn coverage_and_source_provenance_are_required_for_pairability() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 80, false);
        let ownership = vertical_seam_owners(width, height, seam_x);
        let sources = boundary_sources(width, height);
        let coverage = GrayImage::from_fn(width, height, |x, _| {
            Luma([if x == seam_x - 1 || x == seam_x {
                255
            } else {
                0
            }])
        });
        let no_coverage = boundary_report(&output, &coverage, &ownership, &sources, &[(0, 1)]);
        assert_eq!(no_coverage.measurable_count, 0);

        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let mut invalid_sources = boundary_sources(width, height);
        invalid_sources[1].geometry.tile_to_world[(0, 2)] = 1_000.0;
        let no_provenance =
            boundary_report(&output, &coverage, &ownership, &invalid_sources, &[(0, 1)]);
        assert_eq!(no_provenance.measurable_count, 0);
        assert!(
            no_provenance
                .unmeasurable_reasons
                .contains_key(degradation::BOUNDARY_PROVENANCE_UNAVAILABLE)
        );
        assert_eq!(no_provenance.technical_unmeasurable_count, 1);
    }
}
