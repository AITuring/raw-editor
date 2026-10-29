//! Test-only final-world Source_RAW ownership feasibility oracle.
//!
//! The caller composes current-run station candidates into [`WorldCell`]s by
//! using the production pose, residual warp, coverage and final sampling
//! origin. This module deliberately knows nothing about Quality-Gate ROI lists,
//! reports, fixture names or prior runs: coherent-region origins are derived
//! only from each source's coverage and low-frequency content mask.

#[cfg(test)]
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RegionGeometry {
    pub roi_side_px: u32,
    pub same_owner_ring_px: u32,
    pub sampling_phase_px: u32,
}

impl RegionGeometry {
    fn evidence_span_px(self) -> u32 {
        self.roi_side_px
            .saturating_add(self.same_owner_ring_px.saturating_mul(2))
    }

    fn protected_span_px(self) -> u32 {
        self.evidence_span_px()
            .saturating_add(self.sampling_phase_px.saturating_sub(1))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CandidateSample {
    /// Stable content identity of one real Source_RAW (normally its digest).
    pub source: u64,
    pub station: u16,
    pub sharpness: f64,
    pub data_cost: f64,
    /// Cost from the accepted final station seam's RGB/stroke term.
    pub seam_cost: f64,
    /// Exact Source_RAW sample corresponding to this final-world cell centre.
    pub source_x: f64,
    pub source_y: f64,
    /// Every output pixel represented by the cell has real source coverage.
    pub fully_covered: bool,
    /// Current-run low-frequency content classification.
    pub flat: bool,
    pub low_frequency_luma: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct WorldCell {
    pub production_source: u64,
    pub production_station: u16,
    pub textured_pixels: u64,
    pub candidates: Vec<CandidateSample>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct WorldGeometry {
    pub columns: u32,
    pub rows: u32,
    pub cell_side_px: u32,
    /// Use the current run's fixed Quality-Gate ROI lattice. Candidate
    /// footprints are 512px plus the unchanged 16px owner ring; no arbitrary
    /// 799px phase protection is required on this approved path.
    pub qg_aligned_region_grid: bool,
    /// World coordinate of final cell `(0, 0)`. Translation changes only these
    /// values, never candidate selection in local cell coordinates.
    pub world_origin: (i64, i64),
    pub regions: RegionGeometry,
}

impl WorldGeometry {
    fn cell_count(self) -> usize {
        self.columns as usize * self.rows as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct OracleInput {
    pub geometry: WorldGeometry,
    pub cells: Vec<WorldCell>,
    pub required_regions: usize,
    pub shortfall_limit: f64,
}

#[path = "final_world_owner_oracle/world_cells.rs"]
mod world_cells;

pub(super) use world_cells::{ProjectedStationCell, compose_station_cells};
use world_cells::{
    candidate, objective_cost, production_candidate, sacrifice, unconstrained_label,
};

#[path = "final_world_owner_oracle/region_solver.rs"]
mod region_solver;

use region_solver::region_options;
pub(super) use region_solver::solve;
#[cfg(test)]
use region_solver::{
    RegionOption, labeling_sacrifice, labels_for, reduced_exact, relaxation_lower_bound,
};

#[path = "final_world_owner_oracle/exact_regions.rs"]
mod exact_regions;

pub(super) use exact_regions::{
    ExactRegionEvaluation, ExactRegionProposal, exact_region_proposals, solve_exact_regions,
};

#[path = "final_world_owner_oracle/exact_runs.rs"]
mod exact_runs;

pub(super) use exact_runs::{
    ExactCandidateDecision, ExactPixelEvidence, ExactSourceSample, aggregate_exact_runs,
    exact_run_stats, exact_run_stats_data_min, exact_run_stats_for_source,
};

#[path = "final_world_owner_oracle/snapshots.rs"]
mod snapshots;

pub(super) use snapshots::{
    StationCandidateCell, StationSnapshot, StationSourceSnapshot, record_station_snapshot,
    reset_station_snapshots, station_snapshots,
};

#[cfg(test)]
#[path = "final_world_owner_oracle/tests.rs"]
mod tests;
