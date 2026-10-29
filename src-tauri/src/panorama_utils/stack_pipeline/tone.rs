//! Low-frequency affine tone harmonisation for owned focus-stack pixels.
//!
//! Tone correction is deliberately separated from ownership.  The owner image
//! supplies the high-frequency residual while a low-pass version of the source
//! supplies only broad exposure and white-balance changes.  This keeps strokes,
//! edges, and texture tied to the source selected by the ownership map.

#![allow(dead_code)]

use image::{GrayImage, Rgb32FImage};
#[cfg(test)]
use image::{Luma, Rgb};
use std::sync::Mutex;

#[cfg(test)]
use super::degradation;
#[cfg(test)]
use super::report::{TonePairWithoutEvidenceRecord, ToneStatus};
use super::report::{ToneReport, ToneTileRecord};

/// Build the low-frequency field on one sixteenth of the output grid.
pub(crate) const TONE_GRID_DIVISOR: u32 = 16;
/// Gaussian sigma in analysis-grid pixels (requirement 9.2/11.5).
pub(crate) const TONE_LOW_FREQUENCY_SIGMA: f32 = 4.0;
/// Minimum retained pair samples after robust outlier rejection.
pub(crate) const TONE_MIN_SAMPLES: usize = 1024;
/// Affine gain and offset limits shared by the stack Tone_Harmonizer.
pub(crate) const TONE_MIN_GAIN: f32 = 1.0 / 1.25;
pub(crate) const TONE_MAX_GAIN: f32 = 1.25;
pub(crate) const TONE_MAX_ABS_OFFSET: f32 = 0.02;
/// Width of the band on each side of an Owner_Region boundary (需求 9.8).
pub(crate) const TONE_BOUNDARY_BAND_PX: u32 = 16;
/// Boundary windows whose band means are compared separately, so a local
/// tone step is located instead of being averaged over a whole boundary.
pub(crate) const TONE_BOUNDARY_WINDOW_PX: u32 = 256;
/// Normalised luminance range of a usable tone sample on both sides (需求 9.1).
pub(crate) const TONE_MIN_SAMPLE_LUMINANCE: f32 = 0.02;
pub(crate) const TONE_MAX_SAMPLE_LUMINANCE: f32 = 0.98;
/// `ToneTile::validity` of a cell the tile covers entirely: tone evidence.
pub(crate) const TONE_CELL_SAMPLE: u8 = 255;
/// `ToneTile::validity` of a cell the tile covers in part: its covered-only
/// field is corrected with the tile, but it is no tone evidence.
pub(crate) const TONE_CELL_PARTIAL: u8 = 128;
/// Log-gain scatter at which a relation's joint-solve weight halves, and the
/// Huber scale floor of its residual (the values of `photometric.rs`).
const TONE_RELATION_SCATTER_SCALE: f64 = 0.04;
const TONE_RELATION_HUBER_LOG: f64 = 0.025;
/// The spatial residual experiment passed its overlap holdout but regressed
/// independent output-to-source ROI Delta-E on Wenyuan, so it must not enter
/// the production correction path.
const TONE_SPATIAL_FIELD_ENABLED: bool = false;

#[cfg(test)]
pub(crate) const fn production_spatial_field_enabled() -> bool {
    TONE_SPATIAL_FIELD_ENABLED
}
/// Fixed deterministic iterations used only while evaluating that experiment.
const TONE_SPATIAL_ITERATIONS: usize = 4;

// Tone solving is reached from the compositor's free functions, which do not
// carry a mutable Stack_Report reference.  Keep the same run-scoped sink used
// by Intra_Station and Focus_Fuser; report publication takes one snapshot at
// the terminating boundary.
static RUN_RECORDS: Mutex<Vec<ToneTileRecord>> = Mutex::new(Vec::new());
static RUN_REPORT: Mutex<Option<ToneReport>> = Mutex::new(None);
static RUN_TOPOLOGY: Mutex<Option<ToneStationTopology>> = Mutex::new(None);
// Radiometric eligibility is fixed after native station validation, before
// robust closure can assign an accepted geometric edge zero fitting weight.
static RUN_NATIVE_PHOTOMETRIC_TOPOLOGY: Mutex<Option<ToneStationTopology>> = Mutex::new(None);

/// The station context the owner compositor needs for tone evidence: each
/// Virtual_Tile's station index and the run's accepted Station_Relations,
/// both keyed by the tile's `ImageInfo::id`. Only the overlap of an accepted
/// relation yields tone samples (需求 9.1). It is published by the station
/// solve, which runs before the compositor, and like the residual model it is
/// read by the compositor rather than threaded through its signature.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ToneStationTopology {
    /// `(ImageInfo::id, station index)` of every Virtual_Tile.
    pub(crate) stations: Vec<(usize, usize)>,
    /// Accepted Station_Relations as `ImageInfo::id` pairs.
    pub(crate) accepted: Vec<(usize, usize)>,
    /// Locally measured, gate-passing adjacent station pairs that were
    /// removed by the global closure graph. These pairs are eligible for the
    /// ownership seam solver only; they are deliberately excluded from tone
    /// evidence and from the authoritative pose graph.
    pub(crate) seam_candidates: Vec<(usize, usize)>,
    /// Locally fitted source-to-target transforms for adjacent station pairs.
    /// This includes `seam_candidates` and may include accepted pairs whose
    /// seam-only refinement was explicitly requested by the compositor. The
    /// transforms never become tone evidence or pose-graph constraints.
    /// Matrices are stored as row-major IEEE-754 bits so this run-scoped
    /// topology remains deterministic and can keep its `Eq` contract without
    /// putting raw `f64` values in the public snapshot.
    pub(crate) seam_candidate_transforms: Vec<(usize, usize, [u64; 9])>,
}

impl ToneStationTopology {
    pub(crate) fn station_of(&self, image_id: usize) -> Option<usize> {
        self.stations
            .iter()
            .find(|(id, _)| *id == image_id)
            .map(|(_, station)| *station)
    }
}

/// Clear the station context at whole-run start. Kept apart from
/// [`reset_run_records`], which a compositor may call at its own start after
/// the station solve published this context.
pub(crate) fn reset_run_topology() {
    *RUN_TOPOLOGY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    *RUN_NATIVE_PHOTOMETRIC_TOPOLOGY
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = None;
}

pub(crate) fn record_native_photometric_topology(topology: ToneStationTopology) {
    *RUN_NATIVE_PHOTOMETRIC_TOPOLOGY
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = Some(topology);
}

pub(crate) fn native_photometric_topology_snapshot() -> Option<ToneStationTopology> {
    RUN_NATIVE_PHOTOMETRIC_TOPOLOGY
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}

pub(crate) fn record_run_topology(topology: ToneStationTopology) {
    *RUN_TOPOLOGY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(topology);
}

pub(crate) fn run_topology_snapshot() -> Option<ToneStationTopology> {
    RUN_TOPOLOGY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

pub(crate) fn reset_run_records() {
    RUN_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    *RUN_REPORT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

pub(crate) fn record_run_record(record: ToneTileRecord) {
    RUN_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(record);
}

pub(crate) fn run_records_snapshot() -> Vec<ToneTileRecord> {
    RUN_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

pub(crate) fn record_run_report(report: ToneReport) {
    *RUN_REPORT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(report);
}

pub(crate) fn run_report_snapshot() -> Option<ToneReport> {
    RUN_REPORT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// One source's low-frequency field in output world coordinates.  `low` and
/// `validity` stay at the one-sixteenth analysis resolution; no full-size
/// low-frequency image is allocated by the harmoniser.
#[derive(Debug, Clone)]
pub(crate) struct ToneTile {
    pub(crate) station_index: usize,
    pub(crate) owner_id: u16,
    pub(crate) low: Rgb32FImage,
    pub(crate) validity: GrayImage,
    pub(crate) world_origin: (f64, f64),
    pub(crate) world_size: (f64, f64),
    pub(crate) world_stride: f64,
    /// Unblurred mean of the covered tile pixels in the tile cell under each
    /// analysis cell, and that cell's covered fraction (row-major, the size
    /// of `low`). A pair of tiles is compared through [`PairField`], built
    /// from these over the cells both cover.
    pub(crate) cell_mean: Rgb32FImage,
    /// Original native quadrature samples retained for narrow contour seam costs.
    pub(crate) native_detail: Option<Rgb32FImage>,
    pub(crate) cell_coverage: Vec<f32>,
}

mod pair_evidence;

pub(crate) use pair_evidence::{
    PairField, ToneSample, ToneSolve, ToneSolveStatus, consistent_samples, solve_tone_pair,
};
use pair_evidence::{ToneRelation, log_channel, record_insufficient_samples};

mod tone_tiles;

#[cfg(test)]
use tone_tiles::apply_low_frequency_tone;
pub(super) use tone_tiles::{gaussian_cells, sample_tile_detail_bilinear};
pub(crate) use tone_tiles::{
    sample_tile_bilinear, world_aligned_native_tone_tile, world_aligned_tone_tile,
};

mod color_metrics;

pub(crate) use color_metrics::delta_e00_rgb;
#[cfg(test)]
pub(crate) use color_metrics::diagnostic_lab_rgb;
use color_metrics::linear_luma;
#[cfg(test)]
use color_metrics::rgb_to_lab;

mod transforms;

pub(super) use transforms::bounded_low_frequency_affine_coordinate;
use transforms::{
    ToneHoldoutGate, build_spatial_field_iterations, constrain_source_fidelity,
    sample_spatial_gain, solve_tone_transforms, spatial_holdout_gate,
};
#[cfg(test)]
use transforms::{ToneSpatialField, ToneTransform, percentile};

mod harmonize;

#[cfg(test)]
pub(crate) use harmonize::harmonize_tone_tiles;
pub(crate) use harmonize::harmonize_tone_tiles_with_fidelity;

#[cfg(test)]
mod owner_oracle;

#[cfg(test)]
use owner_oracle::{TONE_OWNER_ORACLE_SNAPSHOT, ToneOracleBoundaryWindow, tone_oracle_cell_sets};
#[cfg(test)]
pub(crate) use owner_oracle::{ToneOwnerOracleSnapshot, take_tone_owner_oracle_snapshot};

#[cfg(test)]
mod test_diagnostics;

#[cfg(test)]
use test_diagnostics::{tone_distribution_pair, write_tone_diagnostics};

#[cfg(test)]
#[test]
fn native_subcell_detail_keeps_a_shift_hidden_by_equal_cell_means() {
    let make = |shift: f64| {
        world_aligned_native_tone_tile(0, 1, (0, 0, 127, 127), |x, _| {
            Some([0.4 + 0.2 * ((x + shift) * std::f64::consts::TAU / 16.0).sin() as f32; 3])
        })
    };
    let mut a = make(0.0);
    let mut b = make(4.0);
    let native_a = sample_tile_detail_bilinear(&a, 65.5, 65.5).unwrap();
    let native_b = sample_tile_detail_bilinear(&b, 65.5, 65.5).unwrap();
    assert!((native_a[0] - native_b[0]).abs() > 0.05);
    assert_eq!(a.cell_coverage, b.cell_coverage);
    a.native_detail = None;
    b.native_detail = None;
    let coarse_a = sample_tile_detail_bilinear(&a, 65.5, 65.5).unwrap();
    let coarse_b = sample_tile_detail_bilinear(&b, 65.5, 65.5).unwrap();
    assert!((coarse_a[0] - coarse_b[0]).abs() < 1e-6);
}

fn median_f64(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<_> = values.filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn median(values: impl Iterator<Item = f32>) -> f32 {
    let mut values: Vec<_> = values.filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests;
