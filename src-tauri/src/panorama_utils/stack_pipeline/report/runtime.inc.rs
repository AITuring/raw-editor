// output
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct PreviewRecord {
    pub path: String,
    pub max_roi_delta_e: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct OutputReport {
    pub written: bool,
    #[serde(default)]
    pub format: String,
    pub path: String,
    pub bit_depth: u8,
    pub alpha_preserved: bool,
    pub icc: String,
    pub result_id: String,
    pub preview: PreviewRecord,
}

impl Default for OutputReport {
    fn default() -> Self {
        Self {
            written: false,
            format: String::new(),
            path: String::new(),
            bit_depth: 16,
            alpha_preserved: false,
            icc: "sRGB".to_string(),
            result_id: String::new(),
            preview: PreviewRecord::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// resources
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ResourcesReport {
    pub memory_threshold_bytes: u64,
    /// `default`, `user_configured`, or `auto_calibrated` (requirement 14).
    pub memory_threshold_source: String,
    pub physical_memory_bytes: u64,
    pub peak_rss_bytes: u64,
    pub rss_sample_count: u64,
    /// The sampler observed the threshold before the run stopped at a
    /// cooperative stage boundary.
    pub memory_threshold_exceeded: bool,
    /// Highest number of simultaneously leased full-size Virtual_Tiles
    /// (requirement 14.4).
    pub max_resident_virtual_tiles: usize,
    pub network_requests: u64,
    pub worker_threads: usize,
    /// How the RANSAC seed was derived (requirement 14.6).
    pub random_seed_source: String,
}

impl Default for ResourcesReport {
    fn default() -> Self {
        Self {
            memory_threshold_bytes: MEMORY_THRESHOLD_DEFAULT_BYTES,
            memory_threshold_source: "default".to_string(),
            physical_memory_bytes: 0,
            peak_rss_bytes: 0,
            rss_sample_count: 0,
            memory_threshold_exceeded: false,
            max_resident_virtual_tiles: 0,
            network_requests: 0,
            worker_threads: 0,
            random_seed_source: "unrecorded".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// degradation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct DegradationEntryRecord {
    /// A stable machine readable failure identifier (requirement 12.7).
    pub reason: String,
    /// `degraded` or `rejected`.
    pub severity: String,
    pub detail: Value,
}

impl Default for DegradationEntryRecord {
    fn default() -> Self {
        Self {
            reason: String::new(),
            severity: "degraded".to_string(),
            detail: Value::Object(Map::new()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct DegradationReport {
    pub entries: Vec<DegradationEntryRecord>,
    pub result: StackRunResult,
}

impl Default for DegradationReport {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            result: StackRunResult::Running,
        }
    }
}

// ---------------------------------------------------------------------------
// Stack_Report
// ---------------------------------------------------------------------------

/// The full Stack_Report document.
///
/// Every top-level field is always serialized, including defaults, so the
/// structure can be asserted without a RAW decode (requirement 15.9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StackReport {
    pub schema: u32,
    pub pipeline_version: String,
    pub run_id: String,
    pub started_at_epoch_ms: u64,
    pub finished_at_epoch_ms: u64,
    pub selected_path: SelectedPath,

    pub input: InputReport,
    pub grouping: GroupingReport,
    pub intra_station: Vec<IntraStationReport>,
    pub fusion: Vec<FusionReport>,
    pub virtual_tiles: Vec<VirtualTileReport>,
    pub virtual_tile_cache: VirtualTileCacheSummary,
    pub topology: TopologyReport,
    pub station_relations: StationRelationsReport,
    pub closure: ClosureReport,
    pub residual_warp: ResidualWarpReport,
    pub tone: ToneReport,
    pub composition: CompositionReport,
    pub quality_gate: QualityGateReport,
    pub output: OutputReport,
    pub resources: ResourcesReport,
    pub degradation: DegradationReport,
}

impl Default for StackReport {
    fn default() -> Self {
        Self {
            schema: STACK_REPORT_SCHEMA,
            pipeline_version: String::new(),
            run_id: String::new(),
            started_at_epoch_ms: 0,
            finished_at_epoch_ms: 0,
            selected_path: SelectedPath::LayeredVirtualTile,

            input: InputReport::default(),
            grouping: GroupingReport::default(),
            intra_station: Vec::new(),
            fusion: Vec::new(),
            virtual_tiles: Vec::new(),
            virtual_tile_cache: VirtualTileCacheSummary::default(),
            topology: TopologyReport::default(),
            station_relations: StationRelationsReport::default(),
            closure: ClosureReport::default(),
            residual_warp: ResidualWarpReport::default(),
            tone: ToneReport::default(),
            composition: CompositionReport::default(),
            quality_gate: QualityGateReport::default(),
            output: OutputReport::default(),
            resources: ResourcesReport::default(),
            degradation: DegradationReport::default(),
        }
    }
}

/// Every top-level key the schema guarantees.  Used by the unit test below and
/// mirrored by `tests/focus-stack-quality-contract.mjs`.
#[allow(dead_code)]
pub(crate) const STACK_REPORT_TOP_LEVEL_FIELDS: &[&str] = &[
    "schema",
    "pipeline_version",
    "run_id",
    "started_at_epoch_ms",
    "finished_at_epoch_ms",
    "selected_path",
    "input",
    "grouping",
    "intra_station",
    "fusion",
    "virtual_tiles",
    "virtual_tile_cache",
    "topology",
    "station_relations",
    "closure",
    "residual_warp",
    "tone",
    "composition",
    "quality_gate",
    "output",
    "resources",
    "degradation",
];

impl StackReport {
    pub(crate) fn new(pipeline_version: &str) -> Self {
        Self {
            pipeline_version: pipeline_version.to_string(),
            run_id: Uuid::new_v4().to_string(),
            started_at_epoch_ms: epoch_millis(),
            ..Self::default()
        }
    }

    pub(crate) fn to_json_string(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("Failed to serialize the stack report: {error}"))
    }
}

fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

/// Resolve `<app_cache_dir>/stack-reports` without forcing the caller to
/// import `tauri::Manager`.
pub(crate) fn resolve_stack_report_dir<R: tauri::Runtime>(
    app_handle: &tauri::AppHandle<R>,
) -> Option<PathBuf> {
    use tauri::Manager;

    if let Some(directory) = std::env::var_os("RAW_EDITOR_STACK_REPORT_DIR") {
        return Some(PathBuf::from(directory));
    }

    app_handle
        .path()
        .app_cache_dir()
        .ok()
        .map(|directory| directory.join(STACK_REPORT_DIR_NAME))
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
    /// [`super::virtual_tile::VirtualTileStore`] whose leases are counted; until
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

/// Serialize `report` into `directory` using a temporary file plus rename so a
/// partially written document is never observable.
pub(crate) fn write_stack_report(
    report: &StackReport,
    directory: &Path,
) -> Result<PathBuf, String> {
    let run_id = if report.run_id.is_empty() {
        "unknown".to_string()
    } else {
        report.run_id.clone()
    };
    let final_path = directory.join(format!("stack-report-{run_id}.json"));
    write_stack_report_at(report, &final_path)
}

fn write_stack_report_at(report: &StackReport, final_path: &Path) -> Result<PathBuf, String> {
    let json = report.to_json_string()?;
    let directory = final_path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(directory).map_err(|error| {
        format!(
            "Failed to create the stack report directory {}: {error}",
            directory.display()
        )
    })?;
    let run_id = final_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("stack-report");
    let temporary_path = directory.join(format!(".{run_id}.tmp"));
    fs::write(&temporary_path, json.as_bytes()).map_err(|error| {
        format!(
            "Failed to write the stack report {}: {error}",
            temporary_path.display()
        )
    })?;
    fs::rename(&temporary_path, final_path).map_err(|error| {
        let _ = fs::remove_file(&temporary_path);
        format!(
            "Failed to publish the stack report {}: {error}",
            final_path.display()
        )
    })?;
    Ok(final_path.to_path_buf())
}

fn rewrite_stack_report<F>(report_path: &Path, apply: F) -> Result<(), String>
where
    F: FnOnce(&mut StackReport) -> Result<(), String>,
{
    let json = fs::read_to_string(report_path).map_err(|error| {
        format!(
            "Failed to read the Stack_Report {}: {error}",
            report_path.display()
        )
    })?;
    let mut report: StackReport = serde_json::from_str(&json).map_err(|error| {
        format!(
            "Failed to parse the Stack_Report {}: {error}",
            report_path.display()
        )
    })?;
    apply(&mut report)?;
    write_stack_report_at(&report, report_path)?;
    Ok(())
}

/// Attach the result identifier to the report written by the matching stack
/// run.  This is done before the result enters the export store, so a later
/// save can update only the report belonging to the visible result.
pub(crate) fn assign_stack_report_result_id(
    report_path: &Path,
    result_id: &str,
) -> Result<(), String> {
    rewrite_stack_report(report_path, |report| {
        if !report.output.result_id.is_empty() && report.output.result_id != result_id {
            return Err(format!(
                "Stack_Report result ID '{}' does not match '{}'.",
                report.output.result_id, result_id
            ));
        }
        report.output.result_id = result_id.to_string();
        Ok(())
    })
}

/// Record the metadata observed by reopening a successfully published output.
/// The rewrite uses the same temporary-file-plus-rename boundary as the initial
/// report, so readers never observe a partially updated document.
pub(crate) fn update_stack_report_output(
    report_path: &Path,
    result_id: &str,
    format: String,
    bit_depth: u8,
    alpha_preserved: bool,
    icc: String,
    output_path: &Path,
) -> Result<(), String> {
    rewrite_stack_report(report_path, |report| {
        if report.output.result_id != result_id {
            return Err(format!(
                "Stack_Report result ID '{}' does not match '{}'.",
                report.output.result_id, result_id
            ));
        }
        report.output.written = true;
        report.output.format = format;
        report.output.path = output_path.to_string_lossy().into_owned();
        report.output.bit_depth = bit_depth;
        report.output.alpha_preserved = alpha_preserved;
        report.output.icc = icc;
        Ok(())
    })
}

