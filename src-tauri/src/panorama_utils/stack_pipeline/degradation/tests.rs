use super::*;
use serde_json::json;

#[test]
fn every_failure_identifier_matches_the_required_format() {
    for (reason, _) in FAILURE_REASONS {
        assert!(
            is_valid_reason_identifier(reason),
            "failure reason {reason} violates ^[a-z0-9_]{{1,64}}$"
        );
    }
    for reason in UNMEASURABLE_REASONS {
        assert!(
            is_valid_reason_identifier(reason),
            "unmeasurable reason {reason} violates ^[a-z0-9_]{{1,64}}$"
        );
    }
}

#[test]
fn identifiers_are_unique_within_and_across_the_two_sets() {
    let mut seen = std::collections::HashSet::new();
    for (reason, _) in FAILURE_REASONS {
        assert!(seen.insert(*reason), "duplicate failure reason {reason}");
    }
    for reason in UNMEASURABLE_REASONS {
        assert!(
            seen.insert(*reason),
            "duplicate unmeasurable reason {reason}"
        );
    }
    assert_eq!(
        seen.len(),
        FAILURE_REASONS.len() + UNMEASURABLE_REASONS.len()
    );
}

#[test]
fn quality_gate_rejection_detail_lists_failed_criteria_and_world_coordinates() {
    let mut ledger = DegradationLedger::new();
    ledger.record(
        QUALITY_GATE_CRITERION_FAILED,
        json!({
            "criteria": [{
                "name": "owner_sharpness_coverage",
                "verdict": "fail",
                "failed": [{
                    "world": {"x": 1171.5, "y": 6923.0},
                    "measured": 2.0,
                    "owner_path": "5/8"
                }]
            }]
        }),
    );

    let detail = ledger
        .quality_gate_rejection_detail()
        .expect("Quality_Gate failure should be actionable");
    assert!(detail.contains("owner_sharpness_coverage"));
    assert!(detail.contains("(1171.50,6923.00)"));
}

#[test]
fn rejection_severities_match_the_design_matrix() {
    let rejected = FAILURE_REASONS
        .iter()
        .filter(|(_, severity)| *severity == Severity::Rejected)
        .map(|(reason, _)| *reason)
        .collect::<Vec<_>>();
    assert_eq!(
        rejected,
        vec![
            INPUT_SOURCE_COUNT_OUT_OF_RANGE,
            INPUT_PIPELINE_VERSION_MISMATCH,
            GEOMETRY_DISCONNECTED,
            CANVAS_LONG_SIDE_EXCEEDED,
            QUALITY_GATE_CRITERION_FAILED,
            QUALITY_GATE_INSUFFICIENT_EVIDENCE,
            MEMORY_THRESHOLD_EXCEEDED,
            RUN_CANCELLED_BY_USER,
        ]
    );
}

#[test]
fn cache_capacity_eviction_is_the_only_informational_reason() {
    let informational = FAILURE_REASONS
        .iter()
        .filter(|(_, severity)| *severity == Severity::Informational)
        .map(|(reason, _)| *reason)
        .collect::<Vec<_>>();
    assert_eq!(informational, vec![CACHE_EVICTED_FOR_CAPACITY]);
    assert_eq!(
        Severity::Informational.as_identifier(),
        "informational",
        "the third severity needs a stable machine readable form too"
    );
}

#[test]
fn informational_entries_are_reported_without_degrading_the_run() {
    let mut ledger = DegradationLedger::new();
    ledger.record(
        CACHE_EVICTED_FOR_CAPACITY,
        json!({ "evicted_entries": 3, "total_bytes": 42 }),
    );
    assert_eq!(ledger.entries().len(), 1, "the entry is still reported");
    assert!(!ledger.is_empty());
    assert!(!ledger.has_degradation());
    assert_eq!(
        ledger.result(),
        RunResult::Success,
        "cache housekeeping must not turn a successful run into a degraded one"
    );

    // A real degradation alongside it still decides the verdict.
    ledger.record(TONE_GAIN_CLAMPED, Value::Null);
    assert_eq!(ledger.result(), RunResult::Degraded);
    ledger.record(GEOMETRY_DISCONNECTED, Value::Null);
    assert_eq!(ledger.result(), RunResult::Rejected);
}

#[test]
fn invalid_identifier_shapes_are_rejected() {
    assert!(!is_valid_reason_identifier(""));
    assert!(!is_valid_reason_identifier("Upper_Case"));
    assert!(!is_valid_reason_identifier("with-dash"));
    assert!(!is_valid_reason_identifier("with space"));
    assert!(!is_valid_reason_identifier(
        &"a".repeat(MAX_REASON_IDENTIFIER_LENGTH + 1)
    ));
    assert!(is_valid_reason_identifier(
        &"a".repeat(MAX_REASON_IDENTIFIER_LENGTH)
    ));
}

#[test]
fn severity_lookup_uses_the_table() {
    assert_eq!(severity_of(GEOMETRY_DISCONNECTED), Some(Severity::Rejected));
    assert_eq!(
        severity_of(COMPOSITION_NARROW_OVERLAP),
        Some(Severity::Degraded)
    );
    assert_eq!(severity_of("not_a_reason"), None);
}

#[test]
fn empty_ledger_reports_success() {
    let ledger = DegradationLedger::new();
    assert_eq!(ledger.result(), RunResult::Success);
    assert!(!ledger.has_rejection());
}

#[test]
fn degraded_entries_alone_report_degraded() {
    let mut ledger = DegradationLedger::new();
    ledger.record(TONE_INSUFFICIENT_SAMPLES, json!({ "samples": 12 }));
    ledger.record(COMPOSITION_NARROW_OVERLAP, Value::Null);
    assert_eq!(ledger.entries().len(), 2);
    assert_eq!(ledger.result(), RunResult::Degraded);
}

#[test]
fn a_single_rejection_outranks_every_degradation() {
    let mut ledger = DegradationLedger::new();
    ledger.record(TONE_GAIN_CLAMPED, Value::Null);
    ledger.record(GEOMETRY_DISCONNECTED, json!({ "components": 2 }));
    ledger.record(CLOSURE_UNRELIABLE_RESIDUAL, Value::Null);
    assert_eq!(ledger.result(), RunResult::Rejected);
    assert_eq!(ledger.entries().len(), 3, "all active paths are kept");
}

#[test]
fn repeated_reasons_are_counted_individually() {
    let mut ledger = DegradationLedger::new();
    ledger.record(INTRA_STATION_REGISTRATION_FAILED, json!({ "frame": 1 }));
    ledger.record(INTRA_STATION_REGISTRATION_FAILED, json!({ "frame": 4 }));
    assert_eq!(ledger.count_of(INTRA_STATION_REGISTRATION_FAILED), 2);
    assert_eq!(ledger.count_of(TONE_GAIN_CLAMPED), 0);
}

#[test]
fn entries_serialise_with_snake_case_severity() {
    let mut ledger = DegradationLedger::new();
    ledger.record(CLOSURE_UNRELIABLE_RESIDUAL, json!({ "median": 3.5 }));
    let value = serde_json::to_value(&ledger).expect("ledger serialises");
    assert_eq!(
        value["entries"][0]["reason"],
        json!("closure_unreliable_residual")
    );
    assert_eq!(value["entries"][0]["severity"], json!("degraded"));
    assert_eq!(value["entries"][0]["detail"]["median"], json!(3.5));
}

#[derive(Default)]
struct InjectedFileSystem {
    files: std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>,
    removed: Vec<std::path::PathBuf>,
    renames: Vec<(std::path::PathBuf, std::path::PathBuf)>,
    synced_directories: Vec<std::path::PathBuf>,
}

impl InjectedFileSystem {
    fn put(&mut self, path: impl Into<std::path::PathBuf>, contents: &[u8]) {
        self.files.insert(path.into(), contents.to_vec());
    }

    fn contents(&self, path: &Path) -> Option<&[u8]> {
        self.files.get(path).map(Vec::as_slice)
    }
}

impl FinalOutputFileSystem for InjectedFileSystem {
    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()> {
        let contents = self
            .files
            .remove(from)
            .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))?;
        self.files.insert(to.to_path_buf(), contents);
        self.renames.push((from.to_path_buf(), to.to_path_buf()));
        Ok(())
    }

    fn remove_file(&mut self, path: &Path) -> io::Result<()> {
        self.removed.push(path.to_path_buf());
        self.files
            .remove(path)
            .map(|_| ())
            .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))
    }

    fn sync_directory(&mut self, path: &Path) -> io::Result<()> {
        self.synced_directories.push(path.to_path_buf());
        Ok(())
    }
}

#[test]
fn injected_filesystem_rejection_outranks_degradation_and_retains_diagnostics() {
    let temporary = Path::new("/output/.stack-result.run-42.tmp");
    let final_path = Path::new("/output/stack-result.tiff");
    let report = Path::new("/diagnostics/stack-report-run-42.json");
    let preview = Path::new("/diagnostics/run-42-preview.jpg");
    let unrelated = Path::new("/output/user-notes.txt");
    let mut file_system = InjectedFileSystem::default();
    file_system.put(temporary, b"partial current run");
    file_system.put(report, b"stack report");
    file_system.put(preview, b"diagnostic preview");
    file_system.put(unrelated, b"pre-existing user file");

    let mut ledger = DegradationLedger::new();
    ledger.record(TONE_GAIN_CLAMPED, Value::Null);
    ledger.record(GEOMETRY_DISCONNECTED, json!({ "component_count": 2 }));
    ledger.record(CLOSURE_UNRELIABLE_RESIDUAL, Value::Null);
    let publication = DegradationManager::new(&ledger)
        .publish_staged_output(&mut file_system, temporary, final_path)
        .expect("rejected output cleanup must succeed");

    assert_eq!(publication, OutputPublication::Rejected);
    assert_eq!(
        ledger.entries().len(),
        3,
        "every active path remains reported"
    );
    assert_eq!(
        DegradationManager::new(&ledger).decision(),
        OutcomeDecision::Rejected
    );
    assert!(!file_system.files.contains_key(temporary));
    assert!(!file_system.files.contains_key(final_path));
    assert_eq!(
        file_system.contents(report),
        Some(b"stack report".as_slice())
    );
    assert_eq!(
        file_system.contents(preview),
        Some(b"diagnostic preview".as_slice())
    );
    assert_eq!(
        file_system.contents(unrelated),
        Some(b"pre-existing user file".as_slice())
    );
    assert_eq!(file_system.removed, vec![temporary.to_path_buf()]);
    assert!(file_system.renames.is_empty());
}

#[test]
fn injected_filesystem_rejection_never_deletes_a_preexisting_final_file() {
    let temporary = Path::new("/output/.stack-result.run-43.tmp");
    let final_path = Path::new("/output/stack-result.tiff");
    let mut file_system = InjectedFileSystem::default();
    file_system.put(temporary, b"rejected current run");
    file_system.put(final_path, b"older user result");
    let mut ledger = DegradationLedger::new();
    ledger.record(CANVAS_LONG_SIDE_EXCEEDED, Value::Null);

    let publication = DegradationManager::new(&ledger)
        .publish_staged_output(&mut file_system, temporary, final_path)
        .expect("rejected output cleanup must succeed");

    assert_eq!(publication, OutputPublication::Rejected);
    assert_eq!(
        file_system.contents(final_path),
        Some(b"older user result".as_slice())
    );
    assert_eq!(file_system.removed, vec![temporary.to_path_buf()]);
}

#[test]
fn every_existing_rejected_identifier_blocks_atomic_publication() {
    for &(reason, severity) in FAILURE_REASONS {
        if severity != Severity::Rejected || reason == RUN_CANCELLED_BY_USER {
            continue;
        }
        let temporary = Path::new("/output/.stack-result.current.tmp");
        let final_path = Path::new("/output/stack-result.tiff");
        let mut file_system = InjectedFileSystem::default();
        file_system.put(temporary, b"staged");
        let mut ledger = DegradationLedger::new();
        ledger.record(TONE_INSUFFICIENT_SAMPLES, Value::Null);
        ledger.record(reason, Value::Null);

        let publication = DegradationManager::new(&ledger)
            .publish_staged_output(&mut file_system, temporary, final_path)
            .expect("rejected output cleanup must succeed");
        assert_eq!(publication, OutputPublication::Rejected, "reason: {reason}");
        assert!(
            !file_system.files.contains_key(temporary),
            "reason: {reason}"
        );
        assert!(
            !file_system.files.contains_key(final_path),
            "reason: {reason}"
        );
    }
}

#[test]
fn injected_filesystem_success_uses_one_atomic_rename_after_acceptance() {
    let temporary = Path::new("/output/.stack-result.run-44.tmp");
    let final_path = Path::new("/output/stack-result.tiff");
    let report = Path::new("/diagnostics/stack-report-run-44.json");
    let preview = Path::new("/diagnostics/run-44-preview.jpg");
    let mut file_system = InjectedFileSystem::default();
    file_system.put(temporary, b"complete encoded result");
    file_system.put(report, b"stack report");
    file_system.put(preview, b"diagnostic preview");
    let mut ledger = DegradationLedger::new();
    ledger.record(TONE_GAIN_CLAMPED, Value::Null);

    let manager = DegradationManager::new(&ledger);
    assert_eq!(manager.decision(), OutcomeDecision::Degraded);
    let publication = manager
        .publish_staged_output(&mut file_system, temporary, final_path)
        .expect("accepted output must publish");

    assert_eq!(
        publication,
        OutputPublication::Published(OutcomeDecision::Degraded)
    );
    assert_eq!(
        file_system.contents(final_path),
        Some(b"complete encoded result".as_slice())
    );
    assert!(!file_system.files.contains_key(temporary));
    assert_eq!(
        file_system.renames,
        vec![(temporary.to_path_buf(), final_path.to_path_buf())]
    );
    assert!(file_system.removed.is_empty());
    assert_eq!(file_system.synced_directories, vec![Path::new("/output")]);
    assert_eq!(
        file_system.contents(report),
        Some(b"stack report".as_slice())
    );
    assert_eq!(
        file_system.contents(preview),
        Some(b"diagnostic preview".as_slice())
    );
}

#[test]
fn cancellation_remains_distinguishable_and_cleans_only_the_run_temp() {
    let temporary = Path::new("/output/.stack-result.cancelled.tmp");
    let final_path = Path::new("/output/stack-result.tiff");
    let mut file_system = InjectedFileSystem::default();
    file_system.put(temporary, b"cancelled current run");
    let mut ledger = DegradationLedger::new();
    ledger.record(GEOMETRY_DISCONNECTED, Value::Null);
    ledger.record(RUN_CANCELLED_BY_USER, Value::Null);

    let manager = DegradationManager::new(&ledger);
    assert_eq!(manager.decision(), OutcomeDecision::Cancelled);
    assert_eq!(
        manager
            .publish_staged_output(&mut file_system, temporary, final_path)
            .expect("cancelled output cleanup must succeed"),
        OutputPublication::Cancelled
    );
    assert!(!file_system.files.contains_key(temporary));
    assert!(!file_system.files.contains_key(final_path));
    assert!(file_system.renames.is_empty());
}

#[test]
fn rss_cancellation_rejects_only_the_staged_output_and_keeps_diagnostics() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    let rss = Arc::new(AtomicU64::new(128));
    let next = Arc::clone(&rss);
    let sampler = super::super::resources::RssSampler::start_with_interval(
        64,
        Duration::from_millis(1),
        move || next.load(Ordering::Relaxed),
    );
    for _ in 0..100 {
        if sampler.cancelled() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        sampler.cancelled(),
        "injected RSS must request cancellation"
    );
    let sample = sampler.stop(64);
    assert!(sample.threshold_exceeded);

    let temporary = Path::new("/output/.stack-result.rss-cancelled.tmp");
    let final_path = Path::new("/output/stack-result.tiff");
    let report = Path::new("/diagnostics/stack-report-rss-cancelled.json");
    let preview = Path::new("/diagnostics/rss-cancelled-preview.jpg");
    let mut file_system = InjectedFileSystem::default();
    file_system.put(temporary, b"partial output");
    file_system.put(report, b"stack report");
    file_system.put(preview, b"diagnostic preview");
    let mut ledger = DegradationLedger::new();
    ledger.record(
        MEMORY_THRESHOLD_EXCEEDED,
        json!({
            "threshold_bytes": 64,
            "peak_rss_bytes": sample.peak_rss_bytes,
            "sample_count": sample.sample_count,
        }),
    );

    // Exercise the same terminating evidence path as production: the
    // injected sampler's peak and sample count must reach Stack_Report,
    // alongside the machine-readable ledger detail that caused rejection.
    let recorder = super::super::report::StackReportRecorder::isolated("rss-cancelled", None);
    recorder.update(|report| {
        report.resources.memory_threshold_bytes = 64;
        report.resources.peak_rss_bytes = sample.peak_rss_bytes;
        report.resources.rss_sample_count = sample.sample_count;
        report.resources.memory_threshold_exceeded = sample.threshold_exceeded;
    });
    recorder.apply_degradation_ledger(&ledger);
    let stack_report = recorder.snapshot();
    assert_eq!(stack_report.resources.memory_threshold_bytes, 64);
    assert_eq!(stack_report.resources.peak_rss_bytes, sample.peak_rss_bytes);
    assert_eq!(stack_report.resources.rss_sample_count, sample.sample_count);
    assert!(stack_report.resources.memory_threshold_exceeded);
    let entry = stack_report
        .degradation
        .entries
        .iter()
        .find(|entry| entry.reason == MEMORY_THRESHOLD_EXCEEDED)
        .expect("the overrun reason must be present in Stack_Report");
    assert_eq!(entry.detail["threshold_bytes"], 64);
    assert_eq!(entry.detail["peak_rss_bytes"], sample.peak_rss_bytes);
    assert_eq!(entry.detail["sample_count"], sample.sample_count);

    assert_eq!(
        DegradationManager::new(&ledger)
            .publish_staged_output(&mut file_system, temporary, final_path)
            .expect("RSS cancellation cleanup must succeed"),
        OutputPublication::Rejected
    );
    assert!(!file_system.files.contains_key(temporary));
    assert!(!file_system.files.contains_key(final_path));
    assert_eq!(
        file_system.contents(report),
        Some(b"stack report".as_slice())
    );
    assert_eq!(
        file_system.contents(preview),
        Some(b"diagnostic preview".as_slice())
    );
    assert_eq!(file_system.removed, vec![temporary.to_path_buf()]);
    assert!(file_system.renames.is_empty());
}

#[test]
fn newer_generation_is_detected_without_cancelling_direct_runs() {
    assert!(!generation_changed_for(0, 1));
    assert!(!generation_changed_for(9, 9));
    assert!(generation_changed_for(9, 10));
}

#[test]
fn superseded_queued_generation_is_acknowledged_within_one_second() {
    set_active_run_generation(42);
    let started = std::time::Instant::now();
    let result = begin_run_scope_for_generation(41);
    assert!(matches!(result, Err(RunScopeCancelled)));
    assert!(started.elapsed() < std::time::Duration::from_secs(1));
    set_active_run_generation(0);
}
