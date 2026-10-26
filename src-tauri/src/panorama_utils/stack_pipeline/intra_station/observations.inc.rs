// Intra_Station_Registrar fragment; included at parent module scope.

// ---------------------------------------------------------------------------
// Run scoped observations
// ---------------------------------------------------------------------------

/// Intra-station records of the run currently in flight.
///
/// The refinement sites are free functions deep inside `mosaic.rs` with no
/// report parameter, exactly like the degradation observation points, so the
/// records are collected in a run scoped global and folded into the report by
/// [`super::report::StackReportRecorder::write_once`].
static RUN_INTRA_STATION: Mutex<Vec<IntraStationReport>> = Mutex::new(Vec::new());

fn with_run_records<T>(body: impl FnOnce(&mut Vec<IntraStationReport>) -> T) -> T {
    let mut guard = RUN_INTRA_STATION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    body(&mut guard)
}

/// Starts a fresh observation window, once per stitching run.
pub(crate) fn reset_run_records() {
    with_run_records(Vec::clear);
}

/// Add one frame record to a per-station collection (需求 2.9).  Frames keep
/// the order in which they were first registered and a re-registered path
/// replaces its record; the station entry is created on first use.  Separated
/// from the global sink so it is testable on its own.
pub(crate) fn push_frame_record(
    records: &mut Vec<IntraStationReport>,
    station_index: usize,
    anchor_path: &str,
    record: IntraStationFrameRecord,
) {
    let station = match records
        .iter_mut()
        .find(|station| station.station_index == station_index)
    {
        Some(station) => station,
        None => {
            records.push(IntraStationReport {
                station_index,
                anchor_path: anchor_path.to_string(),
                analysis_long_side: INTRA_STATION_ANALYSIS_LONG_SIDE,
                frames: Vec::new(),
            });
            records.last_mut().expect("just pushed")
        }
    };
    if station.anchor_path.is_empty() {
        station.anchor_path = anchor_path.to_string();
    }
    // A station is rendered more than once per run (station matching, a
    // rendering-prior repair, composition). Each render re-measures its frames,
    // so the latest measurement replaces the frame's earlier record in place
    // instead of listing the same Source_RAW twice in Stack_Report.
    match station
        .frames
        .iter_mut()
        .find(|frame| frame.path == record.path)
    {
        Some(existing) => *existing = record,
        None => station.frames.push(record),
    }
}

/// Record one frame of one Capture_Station of the current run (需求 2.9).
pub(crate) fn record_run_frame(
    station_index: usize,
    anchor_path: &str,
    record: IntraStationFrameRecord,
) {
    with_run_records(|records| push_frame_record(records, station_index, anchor_path, record))
}

/// Copy of the records for report serialisation.  Stations are ordered by
/// index so the document does not inherit the registration order.
pub(crate) fn run_records_snapshot() -> Vec<IntraStationReport> {
    let mut records = with_run_records(|records| records.clone());
    records.sort_by_key(|station| station.station_index);
    records
}

