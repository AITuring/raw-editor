//! Per-station candidate snapshots of the current run.
//!
//! The score-noise diagnostics record one snapshot per Capture_Station; the
//! final-world owner runner reads them back in station order.

use nalgebra::Matrix3;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::panorama_utils::stitching) struct StationCandidateCell {
    /// The production evidence grid stores these values as `f32`; retaining
    /// that representation here makes owner-shortfall replay bit-identical.
    pub sharpness: f32,
    pub data_cost: f64,
    pub disagreement: f32,
    /// Candidate supplied the production ownership-cell measurement.
    pub covered: bool,
    /// Every source pixel represented by the ownership cell is covered; this
    /// stricter flag gates coherent-region eligibility.
    pub fully_covered: bool,
    pub flat: bool,
    pub low_frequency_luma: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::panorama_utils::stitching) struct StationSourceSnapshot {
    pub source: String,
    pub local_owner: u16,
    pub source_to_station: Matrix3<f64>,
    pub cells: Vec<StationCandidateCell>,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::panorama_utils::stitching) struct StationSnapshot {
    pub station_index: usize,
    pub dimensions: (u32, u32),
    pub columns: u32,
    pub rows: u32,
    pub cell_side_px: u32,
    pub production_owners: Vec<u16>,
    pub sources: Vec<StationSourceSnapshot>,
}

pub(super) static STATION_SNAPSHOTS: Mutex<Vec<StationSnapshot>> = Mutex::new(Vec::new());

pub(in crate::panorama_utils::stitching) fn reset_station_snapshots() {
    STATION_SNAPSHOTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

pub(in crate::panorama_utils::stitching) fn record_station_snapshot(snapshot: StationSnapshot) {
    let mut snapshots = STATION_SNAPSHOTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match snapshots
        .iter_mut()
        .find(|current| current.station_index == snapshot.station_index)
    {
        Some(current) => *current = snapshot,
        None => snapshots.push(snapshot),
    }
}

pub(in crate::panorama_utils::stitching) fn station_snapshots() -> Vec<StationSnapshot> {
    let mut snapshots = STATION_SNAPSHOTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    snapshots.sort_by_key(|snapshot| snapshot.station_index);
    snapshots
}
