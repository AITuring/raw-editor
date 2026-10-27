//! Compatibility facade for the Stack_Report schema and recorder.
//!
//! The public-in-crate paths remain stable while the schema, recorder and I/O
//! responsibilities live in focused sibling modules.

#[path = "report_document.rs"]
mod report_document;
#[path = "report_io.rs"]
mod report_io;
#[path = "report_recorder.rs"]
mod report_recorder;
#[path = "report_schema_core.rs"]
mod report_schema_core;
#[path = "report_schema_metrics.rs"]
mod report_schema_metrics;
#[path = "report_schema_solver.rs"]
mod report_schema_solver;

pub(crate) use report_document::{STACK_REPORT_TOP_LEVEL_FIELDS, StackReport};
pub(crate) use report_io::*;
pub(crate) use report_recorder::*;
pub(crate) use report_schema_core::*;
pub(crate) use report_schema_metrics::*;
pub(crate) use report_schema_solver::*;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::fs;

    use super::super::degradation::{self, DegradationLedger};
    use super::report_recorder::ledger_run_result;

    fn default_report_value() -> Value {
        serde_json::to_value(StackReport::default()).expect("the default report must serialize")
    }

    #[test]
    fn stack_report_serializes_every_top_level_field() {
        let value = default_report_value();
        let object = value.as_object().expect("the report must be a JSON object");
        for field in STACK_REPORT_TOP_LEVEL_FIELDS {
            assert!(
                object.contains_key(*field),
                "the stack report schema must always write `{field}`"
            );
        }
        assert_eq!(
            object.len(),
            STACK_REPORT_TOP_LEVEL_FIELDS.len(),
            "the schema field list must stay in sync with the struct"
        );
    }

    #[test]
    fn stack_report_defaults_are_written_explicitly() {
        let value = default_report_value();
        assert_eq!(value["schema"], Value::from(STACK_REPORT_SCHEMA));
        assert_eq!(value["input"]["source_count"], Value::from(0));
        assert_eq!(
            value["input"]["sources"],
            Value::Array(Vec::new()),
            "an empty source list must still be present"
        );
        assert_eq!(
            value["input"]["feature_provider"]["selected"],
            Value::from("rust_fast_brief")
        );
        assert_eq!(
            value["input"]["feature_provider"]["explicit_copy_bytes"],
            Value::from(0)
        );
        assert_eq!(value["grouping"]["station_count"], Value::from(0));
        assert_eq!(value["closure"]["status"], Value::from("not_run"));
        assert_eq!(value["closure"]["fallback_reason"], Value::Null);
        assert_eq!(value["residual_warp"]["identity"], Value::from(true));
        assert_eq!(
            value["tone"]["boundary_delta_e"]["threshold"],
            Value::from(TONE_BOUNDARY_DELTA_E_THRESHOLD)
        );
        assert_eq!(
            value["composition"]["canvas"]["long_side_limit"],
            Value::from(CANVAS_LONG_SIDE_LIMIT)
        );
        assert_eq!(value["quality_gate"]["verdict"], Value::from("not_run"));
        assert_eq!(
            value["virtual_tile_cache"]["limit_bytes"],
            Value::from(VIRTUAL_TILE_CACHE_LIMIT_BYTES)
        );
        assert_eq!(
            value["resources"]["memory_threshold_bytes"],
            Value::from(MEMORY_THRESHOLD_DEFAULT_BYTES)
        );
        assert_eq!(value["degradation"]["result"], Value::from("running"));
    }

    #[test]
    fn stack_report_round_trips_through_json() {
        let report = StackReport::new("stack-test-1");
        let json = report.to_json_string().expect("serialization must succeed");
        let parsed: StackReport = serde_json::from_str(&json).expect("the schema must round-trip");
        assert_eq!(parsed, report);
    }

    #[test]
    fn selected_path_and_result_identifiers_are_stable() {
        for (variant, identifier) in [
            (SelectedPath::LayeredVirtualTile, "layered_virtual_tile"),
            (SelectedPath::ProgressiveSeamTile, "progressive_seam_tile"),
            (SelectedPath::StreamingMosaic, "streaming_mosaic"),
            (
                SelectedPath::LegacySingleLayerMosaic,
                "legacy_single_layer_mosaic",
            ),
            (SelectedPath::SingleStation, "single_station"),
        ] {
            assert_eq!(variant.as_identifier(), identifier);
            assert_eq!(
                serde_json::to_value(variant).expect("serialization must succeed"),
                Value::from(identifier),
                "the serialized form must match the stable identifier"
            );
        }
        for (variant, identifier) in [
            (StackRunResult::Running, "running"),
            (StackRunResult::Success, "success"),
            (StackRunResult::Degraded, "degraded"),
            (StackRunResult::Rejected, "rejected"),
            (StackRunResult::Cancelled, "cancelled"),
        ] {
            assert_eq!(variant.as_identifier(), identifier);
            assert_eq!(
                serde_json::to_value(variant).expect("serialization must succeed"),
                Value::from(identifier)
            );
        }
    }

    #[test]
    fn recorder_publishes_component_values_on_every_terminating_path() {
        // Property 93 is about the report emitted by the component, not only
        // serde round-tripping an arbitrary in-memory value.  Exercise the
        // recorder's actual terminal write for each result and verify that the
        // selected path, Quality_Gate result and terminal timestamps survive
        // the disk boundary together.
        for (result, selected_path) in [
            (StackRunResult::Success, SelectedPath::LayeredVirtualTile),
            (
                StackRunResult::Rejected,
                SelectedPath::LegacySingleLayerMosaic,
            ),
            (StackRunResult::Cancelled, SelectedPath::SingleStation),
        ] {
            let directory = tempfile::tempdir().expect("report directory");
            let recorder = StackReportRecorder::isolated(
                "property-93-terminal",
                Some(directory.path().to_path_buf()),
            );
            recorder.set_selected_path(selected_path);
            recorder.set_result(result);
            recorder.update(|report| {
                report.quality_gate.verdict = QualityGateVerdict::Pass;
                report.quality_gate.criteria = vec![QualityGateCriterionRecord {
                    name: "effective_pixel_count".to_string(),
                    verdict: QualityGateVerdict::Pass,
                    measurable_count: 1,
                    ..Default::default()
                }];
            });

            let path = recorder
                .write_once()
                .expect("terminal report must be written");
            let encoded = fs::read_to_string(path).expect("terminal report is readable");
            let published: StackReport =
                serde_json::from_str(&encoded).expect("published report schema");
            assert_eq!(published.selected_path, selected_path);
            assert_eq!(published.degradation.result, result);
            assert_eq!(published.quality_gate.verdict, QualityGateVerdict::Pass);
            assert_eq!(published.quality_gate.criteria.len(), 1);
            assert!(published.finished_at_epoch_ms >= published.started_at_epoch_ms);
        }
    }

    #[test]
    fn recorder_writes_exactly_one_report_on_drop() {
        let directory = tempfile::tempdir().expect("a temporary directory must be available");
        let path = {
            let recorder =
                StackReportRecorder::isolated("stack-test-1", Some(directory.path().to_path_buf()));
            recorder.update(|report| {
                report.input.source_count = 84;
                report.input.max_sources = 500;
            });
            recorder.set_selected_path(SelectedPath::ProgressiveSeamTile);
            recorder.set_result(StackRunResult::Success);
            recorder.write_once().expect("the report must be written")
        };
        assert!(path.exists(), "the report file must exist after the write");

        let entries = fs::read_dir(directory.path())
            .expect("the directory must be readable")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            entries.len(),
            1,
            "an atomic write must not leave a temporary file behind: {entries:?}"
        );

        let parsed: StackReport =
            serde_json::from_str(&fs::read_to_string(&path).expect("the report must be readable"))
                .expect("the written report must parse");
        assert_eq!(parsed.input.source_count, 84);
        assert_eq!(parsed.selected_path, SelectedPath::ProgressiveSeamTile);
        assert_eq!(parsed.degradation.result, StackRunResult::Success);
        assert!(
            parsed.finished_at_epoch_ms >= parsed.started_at_epoch_ms,
            "the finish timestamp must be recorded when the report is written"
        );
    }

    #[test]
    fn output_metadata_updates_only_the_matching_result_atomically() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let recorder = StackReportRecorder::isolated(
            "stack-test-output",
            Some(directory.path().to_path_buf()),
        );
        let report_path = recorder.write_once().expect("write report");
        assign_stack_report_result_id(&report_path, "result-42").expect("assign result ID");
        let output_path = directory.path().join("stack.tiff");
        update_stack_report_output(
            &report_path,
            "result-42",
            "TIFF".to_string(),
            16,
            true,
            "sRGB".to_string(),
            &output_path,
        )
        .expect("update output metadata");

        let parsed: StackReport = serde_json::from_str(
            &fs::read_to_string(&report_path).expect("updated report must be readable"),
        )
        .expect("updated report must parse");
        assert!(parsed.output.written);
        assert_eq!(parsed.output.result_id, "result-42");
        assert_eq!(parsed.output.format, "TIFF");
        assert_eq!(parsed.output.bit_depth, 16);
        assert!(parsed.output.alpha_preserved);
        assert_eq!(parsed.output.icc, "sRGB");
        assert_eq!(parsed.output.path, output_path.to_string_lossy());
        assert!(
            !report_path
                .with_file_name(".stack-report-stack-test-output.json.tmp")
                .exists(),
            "the atomic rewrite must clean up its temporary file"
        );
        let error = update_stack_report_output(
            &report_path,
            "different-result",
            "PNG".to_string(),
            8,
            false,
            "none".to_string(),
            &directory.path().join("other.png"),
        )
        .expect_err("a different result must not overwrite the report");
        assert!(error.contains("does not match"));
    }

    #[test]
    fn recorder_without_a_directory_stays_silent() {
        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        assert!(recorder.write_once().is_none());
        assert!(recorder.write_once().is_none());
    }

    #[test]
    fn ledger_entries_reach_the_report_with_their_severity() {
        let mut ledger = DegradationLedger::new();
        ledger.record(
            degradation::TONE_INSUFFICIENT_SAMPLES,
            serde_json::json!({ "retained_samples": 12 }),
        );
        ledger.record(
            degradation::GEOMETRY_DISCONNECTED,
            serde_json::json!({ "components": 2 }),
        );

        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.set_result(StackRunResult::Success);
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();

        assert_eq!(report.degradation.entries.len(), 2);
        assert_eq!(
            report.degradation.entries[0].reason,
            degradation::TONE_INSUFFICIENT_SAMPLES
        );
        assert_eq!(report.degradation.entries[0].severity, "degraded");
        assert_eq!(
            report.degradation.entries[0].detail["retained_samples"],
            Value::from(12)
        );
        assert_eq!(
            report.degradation.entries[1].reason,
            degradation::GEOMETRY_DISCONNECTED
        );
        assert_eq!(report.degradation.entries[1].severity, "rejected");
        assert_eq!(
            report.degradation.result,
            StackRunResult::Rejected,
            "a ledger rejection must override the compositor's success"
        );
    }

    #[test]
    fn discarded_single_layer_station_evidence_is_counted_per_station() {
        let mut ledger = DegradationLedger::new();
        ledger.record(
            degradation::STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED,
            serde_json::json!({
                "count": 1,
                "stations": [2, 5],
                "independent_support": 1,
            }),
        );
        ledger.record(
            degradation::STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED,
            serde_json::json!({
                "count": 2,
                "stations": [2, 7],
                "independent_support": 1,
            }),
        );

        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();

        assert_eq!(
            report.station_relations.discarded_single_layer_evidence,
            vec![
                DiscardedEvidenceRecord {
                    station_index: 2,
                    count: 3,
                },
                DiscardedEvidenceRecord {
                    station_index: 5,
                    count: 1,
                },
                DiscardedEvidenceRecord {
                    station_index: 7,
                    count: 2,
                },
            ]
        );
        assert_eq!(
            report
                .degradation
                .entries
                .iter()
                .filter(|entry| {
                    entry.reason == degradation::STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED
                })
                .count(),
            2,
            "the stable discard reason remains countable in the report ledger"
        );
    }

    #[test]
    fn an_empty_ledger_leaves_the_recorded_result_alone() {
        let ledger = DegradationLedger::new();
        for recorded in [
            StackRunResult::Success,
            StackRunResult::Rejected,
            StackRunResult::Cancelled,
        ] {
            let recorder = StackReportRecorder::isolated("stack-test-1", None);
            recorder.set_result(recorded);
            recorder.apply_degradation_ledger(&ledger);
            assert_eq!(recorder.snapshot().degradation.result, recorded);
            assert!(recorder.snapshot().degradation.entries.is_empty());
        }
    }

    #[test]
    fn cancellation_is_never_downgraded_by_the_ledger() {
        let mut ledger = DegradationLedger::new();
        ledger.record(degradation::TONE_GAIN_CLAMPED, Value::Null);
        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.set_result(StackRunResult::Cancelled);
        recorder.apply_degradation_ledger(&ledger);
        assert_eq!(
            recorder.snapshot().degradation.result,
            StackRunResult::Cancelled
        );

        // A ledger that saw the cancellation itself reports it as cancelled
        // rather than as the plain rejection its severity implies.
        let mut cancelled = DegradationLedger::new();
        cancelled.record(degradation::RUN_CANCELLED_BY_USER, Value::Null);
        assert_eq!(ledger_run_result(&cancelled), StackRunResult::Cancelled);

        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.set_result(StackRunResult::Success);
        recorder.apply_degradation_ledger(&cancelled);
        assert_eq!(
            recorder.snapshot().degradation.result,
            StackRunResult::Cancelled
        );
    }

    #[test]
    fn the_terminating_write_rechecks_every_source_digest() {
        let before = ["11".repeat(32), "22".repeat(32)];
        let after = ["11".repeat(32), "33".repeat(32)];
        let observed = after.clone();
        let recorder = StackReportRecorder::isolated("stack-test-1", None)
            .with_source_digest_recheck(Box::new(move |paths| {
                assert_eq!(paths, ["/tmp/a.NEF", "/tmp/b.NEF"]);
                observed.to_vec()
            }));
        recorder.update(|report| {
            report.input.sources = vec![
                SourceRecord {
                    path: "/tmp/a.NEF".to_string(),
                    sha256_before: before[0].clone(),
                    ..Default::default()
                },
                SourceRecord {
                    path: "/tmp/b.NEF".to_string(),
                    sha256_before: before[1].clone(),
                    ..Default::default()
                },
            ];
        });
        assert!(
            recorder
                .snapshot()
                .input
                .sources
                .iter()
                .all(|source| source.sha256_after.is_empty()),
            "the after digest is only taken on the terminating path"
        );

        recorder.write_once();
        let sources = recorder.snapshot().input.sources;
        assert_eq!(sources[0].sha256_before, sources[0].sha256_after);
        assert_eq!(sources[1].sha256_after, after[1]);
        assert_ne!(
            sources[1].sha256_before, sources[1].sha256_after,
            "a source rewritten behind the pipeline's back must stay visible in the report"
        );
    }

    #[test]
    fn a_recorder_without_a_recheck_leaves_the_after_digest_empty() {
        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.update(|report| {
            report.input.sources = vec![SourceRecord {
                path: "/tmp/a.NEF".to_string(),
                sha256_before: "44".repeat(32),
                ..Default::default()
            }];
        });
        recorder.write_once();
        let sources = recorder.snapshot().input.sources;
        assert_eq!(sources[0].sha256_before, "44".repeat(32));
        assert!(sources[0].sha256_after.is_empty());
    }

    #[test]
    fn the_resident_virtual_tile_peak_only_grows() {
        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        for observed in [1, 2, 1, 0] {
            recorder.record_max_resident_virtual_tiles(observed);
        }
        assert_eq!(
            recorder.snapshot().resources.max_resident_virtual_tiles,
            2,
            "the report keeps the peak, not the last observation"
        );
    }

    #[test]
    fn a_degraded_ledger_upgrades_success_but_not_rejection() {
        let mut ledger = DegradationLedger::new();
        ledger.record(degradation::CACHE_WRITE_UNAVAILABLE, Value::Null);

        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.set_result(StackRunResult::Success);
        recorder.apply_degradation_ledger(&ledger);
        assert_eq!(
            recorder.snapshot().degradation.result,
            StackRunResult::Degraded
        );

        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.set_result(StackRunResult::Rejected);
        recorder.apply_degradation_ledger(&ledger);
        assert_eq!(
            recorder.snapshot().degradation.result,
            StackRunResult::Rejected
        );
    }
}
