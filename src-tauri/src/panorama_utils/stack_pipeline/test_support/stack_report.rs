//! Stack_Report generators.
//!
//! A random but schema-legal Stack_Report: every failure reason comes from the
//! frozen identifier sets of degradation.rs, and the numeric fields straddle the
//! design thresholds so out-of-range combinations are representable.

use super::super::degradation::DegradationLedger;
use super::super::degradation::FAILURE_REASONS;
use super::super::degradation::Severity;
use super::super::degradation::UNMEASURABLE_REASONS;
use super::super::report::CanvasRecord;
use super::super::report::ClosureReport;
use super::super::report::ClosureStatus;
use super::super::report::CompositionReport;
use super::super::report::DegradationEntryRecord;
use super::super::report::DegradationReport;
use super::super::report::InputReport;
use super::super::report::QualityGateCriterionRecord;
use super::super::report::QualityGateReport;
use super::super::report::QualityGateVerdict;
use super::super::report::SelectedPath;
use super::super::report::SourceRecord;
use super::super::report::StackReport;
use super::super::report::StackRunResult;
use super::super::report::UnmeasurableRecord;
use super::super::report::WorldPoint;
use proptest::prelude::*;
use serde_json::Value;
use serde_json::json;

/// The stable `quality_gate.criteria[*].name` identifier set of the design
/// document.  Used so a generated report carries realistic criterion names.
pub const QUALITY_GATE_CRITERION_NAMES: &[&str] = &[
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

/// A failure reason identifier together with the severity the design matrix
/// declares for it.
pub fn arb_failure_reason() -> impl Strategy<Value = (&'static str, Severity)> {
    prop::sample::select(FAILURE_REASONS.to_vec())
}

/// An unmeasurable reason identifier (requirement 11.13).
pub fn arb_unmeasurable_reason() -> impl Strategy<Value = &'static str> {
    prop::sample::select(UNMEASURABLE_REASONS.to_vec())
}

pub(super) fn arb_criterion_name() -> impl Strategy<Value = &'static str> {
    prop::sample::select(QUALITY_GATE_CRITERION_NAMES.to_vec())
}

/// Free-form measured context of a degradation entry.  Values deliberately run
/// past the thresholds of the design document so a report can carry
/// out-of-range combinations.
pub fn arb_degradation_detail() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        (-1.0e6f64..1.0e6f64).prop_map(|measured| json!({ "measured": measured })),
        (0.0f64..1.0e4f64, 0.0f64..1.0e4f64).prop_map(
            |(threshold, measured)| json!({ "threshold": threshold, "measured": measured })
        ),
        (0usize..512, 0usize..512)
            .prop_map(|(left, right)| json!({ "left": left, "right": right })),
        any::<bool>().prop_map(|clamped| json!({ "clamped": clamped })),
        (0usize..1024).prop_map(|count| json!({ "count": count })),
    ]
}

/// A ledger holding 0..8 active degraded or rejected paths.
pub fn arb_degradation_ledger() -> impl Strategy<Value = DegradationLedger> {
    prop::collection::vec((arb_failure_reason(), arb_degradation_detail()), 0..8).prop_map(
        |records| {
            let mut ledger = DegradationLedger::new();
            for ((reason, _severity), detail) in records {
                ledger.record(reason, detail);
            }
            ledger
        },
    )
}

pub(super) fn degradation_entry_records(ledger: &DegradationLedger) -> Vec<DegradationEntryRecord> {
    ledger
        .entries()
        .iter()
        .map(|entry| DegradationEntryRecord {
            reason: entry.reason.to_string(),
            severity: entry.severity.as_identifier().to_string(),
            detail: entry.detail.clone(),
        })
        .collect()
}

pub(super) fn arb_selected_path() -> impl Strategy<Value = SelectedPath> {
    prop_oneof![
        Just(SelectedPath::LayeredVirtualTile),
        Just(SelectedPath::ProgressiveSeamTile),
        Just(SelectedPath::StreamingMosaic),
        Just(SelectedPath::LegacySingleLayerMosaic),
        Just(SelectedPath::SingleStation),
    ]
}

pub(super) fn arb_run_result() -> impl Strategy<Value = StackRunResult> {
    prop_oneof![
        Just(StackRunResult::Running),
        Just(StackRunResult::Success),
        Just(StackRunResult::Degraded),
        Just(StackRunResult::Rejected),
        Just(StackRunResult::Cancelled),
    ]
}

pub(super) fn arb_quality_gate_verdict() -> impl Strategy<Value = QualityGateVerdict> {
    prop_oneof![
        Just(QualityGateVerdict::Pass),
        Just(QualityGateVerdict::Fail),
        Just(QualityGateVerdict::InsufficientEvidence),
        Just(QualityGateVerdict::NotApplicable),
        Just(QualityGateVerdict::NotRun),
    ]
}

pub(super) fn arb_closure_status() -> impl Strategy<Value = ClosureStatus> {
    prop_oneof![
        Just(ClosureStatus::Converged),
        Just(ClosureStatus::Unreliable),
        Just(ClosureStatus::NoConstraints),
        Just(ClosureStatus::NotRun),
    ]
}

/// Source records with plausible absolute paths and digests.  The digests are
/// synthetic hex strings: no property in this stage recomputes them.
pub(super) fn arb_input_report() -> impl Strategy<Value = InputReport> {
    // `source_count` is free to exceed `max_sources` so the out-of-range
    // rejection combination is representable.
    (0usize..600, prop::collection::vec(0u8..16, 0..6)).prop_map(|(source_count, digits)| {
        let sources = digits
            .iter()
            .enumerate()
            .map(|(index, digit)| {
                let digest =
                    std::iter::repeat_n(char::from_digit(u32::from(*digit), 16).unwrap_or('0'), 64)
                        .collect::<String>();
                SourceRecord {
                    path: format!("/tmp/stack-pipeline/DSC_{:04}.NEF", 3680 + index),
                    sha256_before: digest.clone(),
                    sha256_after: digest,
                    decoded: index % 3 != 0,
                }
            })
            .collect();
        InputReport {
            source_count,
            max_sources: 500,
            sources,
            ..InputReport::default()
        }
    })
}

/// Canvas sizes straddle the 262,144 world-pixel long-side ceiling.
pub(super) fn arb_composition_report() -> impl Strategy<Value = CompositionReport> {
    (1u64..400_000, 1u64..400_000, 0u64..1_000_000).prop_map(|(width, height, opaque_pixels)| {
        CompositionReport {
            canvas: CanvasRecord {
                width,
                height,
                ..CanvasRecord::default()
            },
            opaque_pixels,
            ..CompositionReport::default()
        }
    })
}

/// Closure residuals straddle the 2.0 / 3.0 world-pixel limits and the 100
/// iteration ceiling.
pub(super) fn arb_closure_report() -> impl Strategy<Value = ClosureReport> {
    (
        arb_closure_status(),
        0usize..200,
        0.0f64..10.0,
        0.0f64..20.0,
        0.0f64..20.0,
        0.0f64..1024.0,
    )
        .prop_map(
            |(
                status,
                iterations,
                residual_median_px,
                residual_p95_px,
                max_direct_pair_p95_px,
                max_corner_correction_px,
            )| ClosureReport {
                status,
                iterations,
                residual_median_px,
                residual_p95_px,
                max_direct_pair_p95_px,
                max_corner_correction_px,
                ..ClosureReport::default()
            },
        )
}

pub(super) fn arb_quality_gate_report() -> impl Strategy<Value = QualityGateReport> {
    (
        arb_quality_gate_verdict(),
        0usize..64,
        prop::collection::vec(
            (arb_criterion_name(), 0.0f64..4.0, 0usize..32, 0usize..32),
            0..4,
        ),
        prop::collection::vec(
            (
                arb_criterion_name(),
                arb_unmeasurable_reason(),
                -1.0e5f64..1.0e5f64,
                -1.0e5f64..1.0e5f64,
            ),
            0..6,
        ),
    )
        .prop_map(
            |(verdict, roi_count, criteria, unmeasurable)| QualityGateReport {
                verdict,
                roi_count,
                criteria: criteria
                    .into_iter()
                    .map(|(name, threshold, measurable_count, unmeasurable_count)| {
                        QualityGateCriterionRecord {
                            name: name.to_string(),
                            threshold,
                            measurable_count,
                            unmeasurable_count,
                            ..QualityGateCriterionRecord::default()
                        }
                    })
                    .collect(),
                unmeasurable: unmeasurable
                    .into_iter()
                    .map(|(criterion, reason, x, y)| UnmeasurableRecord {
                        criterion: criterion.to_string(),
                        world: WorldPoint { x, y },
                        reason: reason.to_string(),
                        category: crate::panorama_utils::stack_pipeline::quality_gate_runner::classify_unmeasurable_reason(reason),
                    })
                    .collect(),
                timing: crate::panorama_utils::stack_pipeline::report::QualityGateTimingRecord::default(),
                confidence_scores: Default::default(),
                roi_photometry: Vec::new(),
                owner_reverse_lookup_failures: Default::default(),
            },
        )
}

/// A random but schema-legal [`StackReport`].
///
/// Every failure reason identifier the document carries is drawn from the
/// frozen identifier sets of `degradation.rs`, which is exactly the invariant
/// Property 79 relies on: the pipeline can only ever write identifiers from
/// that set.  Numeric fields intentionally cross the thresholds of the design
/// document so out-of-range combinations are representable.
pub fn arb_stack_report() -> impl Strategy<Value = StackReport> {
    (
        arb_degradation_ledger(),
        arb_run_result(),
        arb_selected_path(),
        arb_input_report(),
        arb_composition_report(),
        arb_closure_report(),
        arb_quality_gate_report(),
        0u64..1_000_000,
    )
        .prop_map(
            |(
                ledger,
                result,
                selected_path,
                input,
                composition,
                closure,
                quality_gate,
                started_at_epoch_ms,
            )| {
                StackReport {
                    pipeline_version: "stack-pipeline-test".to_string(),
                    run_id: format!("run-{started_at_epoch_ms}"),
                    started_at_epoch_ms,
                    finished_at_epoch_ms: started_at_epoch_ms,
                    selected_path,
                    input,
                    composition,
                    closure,
                    quality_gate,
                    degradation: DegradationReport {
                        entries: degradation_entry_records(&ledger),
                        result,
                    },
                    ..StackReport::default()
                }
            },
        )
}
