//! Quality_Gate criteria and verdict rules.
//!
//! The criterion accumulator, the owner-sharpness statistics and criterion,
//! the four evidence rules of one criterion, the overall verdict and the run
//! outcome record.

use super::super::degradation;
use super::super::quality_gate;
use super::super::report::FailedMeasurementRecord;
use super::super::report::QualityGateCriterionRecord;
use super::super::report::QualityGateReport;
use super::super::report::QualityGateVerdict;
use super::super::report::UnmeasurableCategory;
use super::super::report::UnmeasurableRecord;
use super::super::report::WorldPoint;
use image::GrayImage;
use std::collections::BTreeMap;

use super::{classify_unmeasurable_reason, confidence_coverage_stats};

pub(crate) struct Criterion {
    pub(super) name: &'static str,
    pub(super) threshold: f64,
    pub(super) min: Option<f64>,
    pub(super) max: Option<f64>,
    pub(super) measured: Vec<FailedMeasurementRecord>,
    pub(super) failed: Vec<FailedMeasurementRecord>,
    pub(super) unmeasurable: usize,
    pub(super) content_not_applicable: usize,
    pub(super) technical_unmeasurable: usize,
    pub(super) unmeasurable_reasons: BTreeMap<String, usize>,
    pub(in crate::panorama_utils::stack_pipeline) diagnostic: bool,
    pub(super) observed_count: usize,
    /// Keep reports bounded when a large canvas has many boundary samples.
    /// The count remains exact while only a deterministic prefix is retained
    /// for the per-point diagnostics.
    pub(super) record_limit: usize,
    pub(super) textured_pixel_count: Option<u64>,
    pub(super) excluded_flat_pixel_count: Option<u64>,
    pub(super) textured_low_confidence_ratio: Option<f64>,
    pub(super) flat_low_confidence_ratio: Option<f64>,
    pub(super) textured_shortfall_ratio: Option<f64>,
    pub(super) flat_shortfall_ratio: Option<f64>,
    pub(super) shortfall_histogram: Vec<u64>,
    pub(super) disagreement_veto_count: u64,
    pub(super) unresolved_pixel_count: u64,
    pub(super) effective_output_pixels: Option<u64>,
    pub(super) effective_projected_pixels: Option<u64>,
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
pub(super) fn hundredths(limit: f64) -> u8 {
    (limit * 100.0).round().clamp(0.0, 100.0) as u8
}

/// Owner shortfall limit of 需求 11.12 in quantised units.
pub(crate) fn owner_shortfall_limit() -> u8 {
    hundredths(f64::from(quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX))
}

/// Disagreement above which the Focus_Fuser's mismatch veto may have kept a
/// less sharp owner, in quantised units.
pub(crate) fn owner_disagreement_veto_limit() -> u8 {
    hundredths(crate::panorama_utils::stack_pipeline::focus_fuser::OWNERSHIP_DISAGREEMENT_VETO)
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
    pub(in crate::panorama_utils::stack_pipeline) fn new(
        name: &'static str,
        threshold: f64,
    ) -> Self {
        Self {
            name,
            threshold,
            ..Self::default()
        }
    }

    pub(super) fn bounded(name: &'static str, min: f64, max: f64) -> Self {
        Self {
            name,
            threshold: min,
            min: Some(min),
            max: Some(max),
            ..Self::default()
        }
    }

    pub(super) fn p95_with_max(name: &'static str, p95: f64, max: f64) -> Self {
        Self {
            name,
            threshold: p95,
            max: Some(max),
            ..Self::default()
        }
    }

    pub(in crate::panorama_utils::stack_pipeline) fn push(
        &mut self,
        world: (f64, f64),
        value: f64,
        owner_path: &str,
        pass: bool,
    ) {
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

    pub(super) fn miss(&mut self, reason: &'static str) {
        self.miss_count(reason, 1);
    }

    pub(super) fn miss_count(&mut self, reason: &'static str, count: u64) {
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

    pub(in crate::panorama_utils::stack_pipeline) fn finish(
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
