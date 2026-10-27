//! Concurrent Stack_Report recorder and terminal-path publication.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::Value;

use super::super::degradation::DegradationLedger;
use super::super::{degradation, focus_fuser, intra_station, residual_warp, tone};
use super::report_document::{StackReport, epoch_millis};
use super::report_io::write_stack_report;
use super::report_schema_core::*;
use super::report_schema_metrics::*;
use super::report_schema_solver::DiscardedEvidenceRecord;

pub(super) fn ledger_run_result(ledger: &degradation::DegradationLedger) -> StackRunResult {
    if ledger.count_of(degradation::RUN_CANCELLED_BY_USER) > 0 {
        return StackRunResult::Cancelled;
    }
    match ledger.result() {
        degradation::RunResult::Success => StackRunResult::Success,
        degradation::RunResult::Degraded => StackRunResult::Degraded,
        degradation::RunResult::Rejected => StackRunResult::Rejected,
    }
}

fn degradation_entry_records(
    ledger: &degradation::DegradationLedger,
) -> Vec<DegradationEntryRecord> {
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

/// Accumulates a [`StackReport`] behind a shared lock and guarantees that the
/// document reaches disk on every terminating path.
///
/// Requirement 10.8 asks for a report on success, degraded output, rejection
/// and cancellation.  Rather than threading a write call through every `?` in
/// the stitching entry point, the recorder writes on `Drop`, which covers the
/// early-return and unwind paths as well.  Write failures never change the
/// run's result; they are reported on stdout only.
pub(crate) struct StackReportRecorder {
    report: Mutex<StackReport>,
    directory: Option<PathBuf>,
    written: AtomicBool,
    /// Whether [`Self::write_once`] folds the process-wide run ledger into
    /// `degradation`.  Production always does.  The unit tests below turn it off
    /// because `cargo test` drives the ledger observation points of other
    /// modules concurrently in the same process, which would make an assertion
    /// on the collected entries depend on unrelated tests.
    bridge_run_ledger: bool,
    /// Re-reads every Source_RAW and returns its digest in lower hex, in the
    /// order the paths were handed in (requirement 4.8 / 12.8).  Installed by
    /// the focus-stack entry point and called once, on the terminating path, so
    /// success, failure and cancellation all end up comparing the same two
    /// numbers.  `None` leaves `sha256_after` empty, which is what the panorama
    /// path and the unit tests want.
    source_digest_recheck: Option<SourceDigestRecheck>,
}

/// Maps Source_RAW paths to their current content digests in lower hex.  A path
/// that cannot be read contributes an empty string: a re-check that is itself
/// impossible must not hide the digest that *was* taken before the run.
pub(crate) type SourceDigestRecheck = Box<dyn Fn(&[String]) -> Vec<String> + Send + Sync>;

impl StackReportRecorder {
    pub(crate) fn new(pipeline_version: &str, directory: Option<PathBuf>) -> Self {
        Self {
            report: Mutex::new(StackReport::new(pipeline_version)),
            directory,
            written: AtomicBool::new(false),
            bridge_run_ledger: true,
            source_digest_recheck: None,
        }
    }

    /// Install the read-only digest re-check of requirement 4.8 / 12.8.
    pub(crate) fn with_source_digest_recheck(mut self, recheck: SourceDigestRecheck) -> Self {
        self.source_digest_recheck = Some(recheck);
        self
    }

    /// Test-only recorder that never reads the process-wide run ledger, so an
    /// assertion on `degradation` depends on this test alone.
    #[cfg(test)]
    pub(crate) fn isolated(pipeline_version: &str, directory: Option<PathBuf>) -> Self {
        Self {
            report: Mutex::new(StackReport::new(pipeline_version)),
            directory,
            written: AtomicBool::new(false),
            bridge_run_ledger: false,
            source_digest_recheck: None,
        }
    }

    /// Mutate the in-progress report.  A poisoned lock is recovered from
    /// because losing the report must never mask the original failure.
    pub(crate) fn update<F>(&self, apply: F)
    where
        F: FnOnce(&mut StackReport),
    {
        let mut report = self
            .report
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        apply(&mut report);
    }

    pub(crate) fn set_selected_path(&self, selected_path: SelectedPath) {
        self.update(|report| report.selected_path = selected_path);
    }

    /// Used by the explicit terminating paths a later phase introduces; today
    /// the result is derived from the degradation ledger alone.
    #[allow(dead_code)]
    pub(crate) fn set_result(&self, result: StackRunResult) {
        self.update(|report| report.degradation.result = result);
    }

    /// Copy `ledger` into `degradation.entries` and let it refine
    /// `degradation.result` (requirement 12.9).  Called once per report, right
    /// before the document is serialized, so late terminating paths are still
    /// represented.
    pub(crate) fn apply_degradation_ledger(&self, ledger: &DegradationLedger) {
        let entries = degradation_entry_records(ledger);
        let ledger_result = ledger_run_result(ledger);
        let mut discarded_by_station = BTreeMap::<usize, u64>::new();
        for entry in ledger.entries().iter().filter(|entry| {
            entry.reason == degradation::STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED
        }) {
            let count = entry
                .detail
                .get("count")
                .and_then(Value::as_u64)
                .unwrap_or(1);
            if let Some(stations) = entry.detail.get("stations").and_then(Value::as_array) {
                for station in stations.iter().filter_map(Value::as_u64) {
                    *discarded_by_station.entry(station as usize).or_default() += count;
                }
            }
        }
        let discarded_single_layer_evidence = discarded_by_station
            .into_iter()
            .map(|(station_index, count)| DiscardedEvidenceRecord {
                station_index,
                count,
            })
            .collect();
        self.update(|report| {
            report.degradation.entries = entries;
            report.degradation.result = report.degradation.result.merge(ledger_result);
            report.station_relations.discarded_single_layer_evidence =
                discarded_single_layer_evidence;
        });
    }

    /// Raise `resources.max_resident_virtual_tiles` to `observed`
    /// (requirement 14.4).  Monotonic: the report keeps the peak of every
    /// observation, so a lease counter that has already fallen back to zero
    /// cannot erase the maximum it reached.
    ///
    /// Called by the Tile_Compositor stage, which owns the
    /// [`super::super::virtual_tile::VirtualTileStore`] whose leases are counted; until
    /// then the field stays at its default of zero.
    #[allow(dead_code)]
    pub(crate) fn record_max_resident_virtual_tiles(&self, observed: usize) {
        self.update(|report| {
            report.resources.max_resident_virtual_tiles =
                report.resources.max_resident_virtual_tiles.max(observed);
        });
    }

    /// Re-read every Source_RAW listed in `input.sources` and record the digest
    /// in `sha256_after` (requirement 4.8 / 12.8).
    ///
    /// The digests are taken outside the report lock: hashing tens of RAW files
    /// must not block a concurrent observation point, and the closure has no
    /// business seeing the half-written report.  A digest that moved is printed,
    /// because the pipeline only ever opens a source read-only and a difference
    /// therefore means something outside this process rewrote the file mid-run.
    fn recheck_source_digests(&self) {
        let Some(recheck) = self.source_digest_recheck.as_ref() else {
            return;
        };
        let paths = self
            .report
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .input
            .sources
            .iter()
            .map(|source| source.path.clone())
            .collect::<Vec<_>>();
        if paths.is_empty() {
            return;
        }
        let digests = recheck(&paths);
        self.update(|report| {
            for (source, digest) in report.input.sources.iter_mut().zip(digests) {
                source.sha256_after = digest;
                if !source.sha256_before.is_empty()
                    && !source.sha256_after.is_empty()
                    && source.sha256_before != source.sha256_after
                {
                    println!(
                        "  - Source_RAW changed during the run: {} ({} -> {})",
                        source.path, source.sha256_before, source.sha256_after
                    );
                }
            }
        });
    }

    pub(crate) fn snapshot(&self) -> StackReport {
        self.report
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Write the report at most once.  Returns the written path when a file was
    /// produced by this call.
    pub(crate) fn write_once(&self) -> Option<PathBuf> {
        if self.written.swap(true, Ordering::SeqCst) {
            return None;
        }
        if self.bridge_run_ledger {
            self.apply_degradation_ledger(&degradation::run_ledger_snapshot());
            // The Intra_Station_Registrar observation points sit in the
            // refinement free functions and collect into a run scoped sink for
            // the same reason the ledger does (requirement 2.9).  Frames
            // already present win, so an explicit `update` from a later phase
            // is never overwritten by the sink.
            let intra_station = intra_station::run_records_snapshot();
            if !intra_station.is_empty() {
                self.update(|report| {
                    if report.intra_station.is_empty() {
                        report.intra_station = intra_station;
                    }
                });
            }
            // The Focus_Fuser sits inside the focus renderer, which has no
            // report parameter either (需求 3.5 / 3.8 / 3.9).
            let fusion = focus_fuser::run_records_snapshot();
            if !fusion.is_empty() {
                self.update(|report| {
                    if report.fusion.is_empty() {
                        report.fusion = fusion;
                    }
                });
            }
            // Residual_Warp is measured inside the stitching path, which has
            // no report parameter.  Copy the run-scoped records at the same
            // terminal point as Intra_Station and Focus_Fuser.
            let residual_report = residual_warp::run_report_snapshot();
            if !residual_report.regions.is_empty() || !residual_report.identity {
                self.update(|report| {
                    if report.residual_warp.regions.is_empty() {
                        report.residual_warp = residual_report;
                    }
                });
            }
            // Tone_Harmonizer records are collected without threading the
            // report through the compositor.  Snapshot them at the same
            // terminating boundary so success, degraded and cancellation all
            // expose the solved and bounded coefficients.
            if let Some(tone_report) = tone::run_report_snapshot() {
                self.update(|report| report.tone = tone_report);
            } else {
                let tone_tiles = tone::run_records_snapshot();
                if !tone_tiles.is_empty() {
                    self.update(|report| {
                        report.tone.status = ToneStatus::Applied;
                        report.tone.tiles = tone_tiles;
                    });
                }
            }
        }
        // Every terminating path lands here, so this single call covers the
        // success, failure and cancellation re-check of requirement 4.8 / 12.8.
        self.recheck_source_digests();
        self.update(|report| report.finished_at_epoch_ms = epoch_millis());
        let report = self.snapshot();
        let directory = self.directory.as_deref()?;
        match write_stack_report(&report, directory) {
            Ok(path) => {
                println!("  - Stack report written to {}", path.display());
                Some(path)
            }
            Err(error) => {
                println!("  - Stack report could not be written: {error}");
                None
            }
        }
    }
}

impl Drop for StackReportRecorder {
    fn drop(&mut self) {
        self.write_once();
    }
}
