// ---------------------------------------------------------------------------
// Run scoped observations
// ---------------------------------------------------------------------------

/// Fusion records of the run currently in flight.
///
/// The fusion runs inside the focus renderer, a free function with no report
/// parameter, so the records reach the document the same way the
/// Intra_Station_Registrar's do (see [`super::intra_station`]).
static RUN_FUSION: Mutex<Vec<FusionReport>> = Mutex::new(Vec::new());

fn with_run_records<T>(body: impl FnOnce(&mut Vec<FusionReport>) -> T) -> T {
    let mut guard = RUN_FUSION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    body(&mut guard)
}

/// Starts a fresh observation window, once per stitching run.
pub(crate) fn reset_run_records() {
    with_run_records(Vec::clear);
}

/// Record one Capture_Station's fusion outcome.  A station rendered twice (a
/// retry, or a cache miss after a partial run) replaces its earlier record.
pub(crate) fn record_run_station(report: FusionReport) {
    with_run_records(|records| {
        match records
            .iter_mut()
            .find(|existing| existing.station_index == report.station_index)
        {
            Some(existing) => *existing = report,
            None => records.push(report),
        }
    });
}

/// Copy of the records for report serialisation, ordered by station index.
pub(crate) fn run_records_snapshot() -> Vec<FusionReport> {
    let mut records = with_run_records(|records| records.clone());
    records.sort_by_key(|station| station.station_index);
    records
}
