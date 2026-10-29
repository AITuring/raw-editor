//! Focus_Fuser: the ownership-cell decision metrics of the intra-station focus
//! fusion (需求 3.1, 3.2, 3.3, 3.9).
//!
//! # Why the metrics live here and not in `mosaic.rs`
//!
//! The task text points at `mosaic.rs`, but the default delivery path no longer
//! passes through the mosaic's ownership grid.  Since task 13.1 the default
//! compositor is `stitching::layered_virtual_tile_compositor`, which renders one
//! virtual tile per Capture_Station through `focus_stack_stitcher_unfilled` →
//! `focus_stack_stitcher_with_margin_policy`.  That function only hands over to
//! `mosaic::detail_preserving_mosaic` when `shifted_mosaic` is true, and a
//! virtual tile holds exactly one station by construction, so
//! `capture_group_count == 1` and `shifted_mosaic` is always false — a real run
//! logs "Compact focus capture detected (1 inferred group(s)); preserving
//! standard focus fusion" once per station.  `mosaic::acutance`,
//! `mosaic::cell_focus`, `mosaic::ownership_disagreement` and
//! `mosaic::ownership_grid` therefore only serve the legacy comparison path.
//!
//! This module reuses those measurements as *decision primitives*
//! ([`super::super::mosaic::acutance_with_step`],
//! [`super::super::mosaic::cell_focus_with_step`],
//! [`super::super::mosaic::cell_low_pass_differences`], and later
//! `seam_cut::cut_grid`) without calling `detail_preserving_mosaic` as a whole:
//! that entry point also applies a cell-wise local warp, which
//! `focus_stack_stitcher_with_margin_policy` already records as visibly
//! fragmenting strokes that are intentionally present at different focal depths.
//! The legacy reductions in `mosaic.rs` keep their old behaviour; every new rule
//! 需求 3 prescribes is a separate function here.
//!
//! Scope: tasks 7.6 and 7.7 build the metrics and the grid geometry, tasks 7.8
//! and 7.9 the multi-label graph cut over those costs together with its timeout
//! fallback and single-frame shortcut, and tasks 7.10 and 7.11 the hard
//! ownership decision the renderer consumes plus Sharpness_Confidence and the
//! low sharpness regions.  [`StationFusion`] is the entry point the production
//! focus renderer drives; see its documentation for why it folds one candidate
//! per exact two-label cut instead of refining a full multi-label grid.

// `solve_ownership` and the refinement sweeps around it are the whole-grid form
// of the same energy: they need every candidate's costs at once, which the
// render loop cannot afford (see `StationFusion`), so they are exercised by the
// tests and by the exhaustive-minimum comparison rather than by the renderer.
#![allow(dead_code)]

use std::sync::Mutex;
#[cfg(test)]
use std::time::Duration;

#[cfg(test)]
use image::Rgb;

#[cfg(test)]
use super::super::mosaic::{
    ACUTANCE_PATCH_POINTS, OWNERSHIP_MISMATCH_PENALTY, SELECTION_LONG_SIDE,
};
#[cfg(test)]
use super::degradation::{DegradationLedger, FUSION_GRAPH_CUT_TIMEOUT};
use super::report::FusionReport;
#[cfg(test)]
use super::report::{FusionSolverStatus, OwnershipGridSize, SharpnessConfidenceSummary};

/// Acutance value that maps to a normalised Sharpness_Score of `0.5`.
///
/// `acutance()` returns an unbounded "gradient energy RMS over local level"
/// ratio, while the Glossary and 需求 3.9 require a Sharpness_Score in `[0, 1]`.
/// [`normalized_sharpness`] is strictly increasing on `[0, ∞)`, so this constant
/// cannot change which candidate of a cell is sharper — it only fixes the
/// dimension the data term and Sharpness_Confidence are expressed in.  The value
/// puts a focused brush edge near the middle of the unit interval while quiet
/// substrate stays close to `0`.
pub(crate) const SHARPNESS_HALF_SCALE: f64 = 0.10;

/// Requirement 11.12's absolute Textured_Pixel floor.
pub(crate) const TEXTURED_PIXEL_SHARPNESS_FLOOR: f64 = 0.10;

/// Side of the Sharpness_Score sampling window in native pixels (需求 3.2).
pub(crate) const SHARPNESS_SAMPLE_WINDOW_PX: f64 = 32.0;

/// Smallest ownership cell side in native pixels (需求 3.1).
pub(crate) const OWNERSHIP_CELL_MIN_PX: u32 = 8;

/// Largest ownership cell side in native pixels (需求 3.1).
pub(crate) const OWNERSHIP_CELL_MAX_PX: u32 = 64;

/// Normalised pixel inconsistency above which a sharper candidate is penalised
/// instead of simply preferred (需求 3.3).
pub(crate) const OWNERSHIP_DISAGREEMENT_VETO: f64 = 0.2;

mod cell_metrics;

pub(crate) use cell_metrics::{
    CellSamplingPlan, OwnershipGridGeometry, candidate_wins_cell,
    cell_disagreement_with_seam_evidence, data_cost, mismatch_penalty,
};
#[cfg(test)]
pub(crate) use cell_metrics::{cell_disagreement, normalized_sharpness, ownership_cell_size_px};
#[cfg(test)]
use cell_metrics::{sharpness_sample_step_px, sharpness_window_px};

mod ownership_solver;

#[cfg(test)]
use ownership_solver::{
    FusionFrame, GraphCutClock, OwnershipSolution, candidate_order, record_solver_degradation,
};
pub(crate) use ownership_solver::{
    GRAPH_CUT_TIMEOUT, MonotonicGraphCutClock, NO_OWNER, OwnershipCostGrid, owner_of,
};
#[cfg(test)]
pub(crate) use ownership_solver::{skips_graph_cut, solve_ownership, solve_ownership_with_clock};

mod confidence;

#[cfg(test)]
use confidence::LowSharpnessRegion;
pub(crate) use confidence::{
    SharpnessCellEvidence, SharpnessEvidenceGrid, cell_confidence, confidence_summary,
    low_sharpness_region_records, low_sharpness_regions, low_sharpness_threshold,
};

mod decisive_cells;

#[cfg(test)]
use decisive_cells::{coherent_base_cells, coherent_decisive_cells};
use decisive_cells::{
    quality_gate_decisive_cells, quality_region_decisive_cells, region_decisive_cells,
};

mod station_fusion;

pub(crate) use station_fusion::{CandidateCells, StationFusion};

#[cfg(test)]
mod content_regions;

#[cfg(test)]
pub(crate) use content_regions::{CONTENT_OWNER_REGION_TARGET, ContentEvidence};
#[cfg(test)]
use content_regions::{ContentRegionCandidate, content_region_candidates, select_content_regions};

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

#[cfg(test)]
mod tests;
#[test]
fn cropped_native_probe_cannot_claim_comparable_focus() {
    let plan = CellSamplingPlan::for_cell(16.0);
    let texture = |x: f64, y: f64| Rgb([((x * 0.2).sin() * (y * 0.3).cos() * 0.2 + 0.4) as f32; 3]);
    let complete = plan.measure(32.0, 32.0, |x, y| Some(texture(x, y)));
    assert!(complete > 0.0);
    // The centre and lower probes remain covered; an upper corner of
    // the same required window leaves the sensor footprint.
    let partial = plan.measure(32.0, 32.0, |x, y| (y >= 27.0).then(|| texture(x, y)));
    assert_eq!(partial, 0.0);
    assert_eq!(plan.measure(32.0, 32.0, |_, _| Some(Rgb([0.0; 3]))), 0.0);
}
