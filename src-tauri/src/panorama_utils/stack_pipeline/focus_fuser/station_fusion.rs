//! The station fusion driver (需求 3.4, 3.5, 3.6, 3.8, 3.9, 3.10).
//!
//! StationFusion folds one candidate frame at a time into the station's
//! ownership with one exact two-label cut per candidate.

use super::super::quality_gate;
use super::super::report::FusionSolverStatus;
use super::super::report::LowSharpnessRegionRecord;
use std::time::Duration;

use super::ownership_solver::GraphCutClock;
use super::{
    CellSamplingPlan, GRAPH_CUT_TIMEOUT, MonotonicGraphCutClock, NO_OWNER,
    OWNERSHIP_DISAGREEMENT_VETO, OwnershipCostGrid, OwnershipGridGeometry, SharpnessCellEvidence,
    SharpnessEvidenceGrid, TEXTURED_PIXEL_SHARPNESS_FLOOR, cell_confidence, data_cost,
    low_sharpness_region_records, low_sharpness_regions, low_sharpness_threshold, mismatch_penalty,
    owner_of, quality_gate_decisive_cells, quality_region_decisive_cells, region_decisive_cells,
};

// ---------------------------------------------------------------------------
// Station fusion driver (需求 3.4, 3.5, 3.6, 3.8, 3.9, 3.10)
// ---------------------------------------------------------------------------

/// Label of the current composite inside one round's two-label sub-problem.
pub(super) const BASE_LABEL: usize = 0;

/// Label of the candidate inside one round's two-label sub-problem.
pub(super) const CANDIDATE_LABEL: usize = 1;

/// The per-cell measurements of one candidate frame, in the cell order of
/// [`OwnershipGridGeometry`].
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct CandidateCells {
    /// Normalised Sharpness_Score (需求 3.2).
    pub sharpness: Vec<f64>,
    /// `true` where the candidate has valid projected pixels (需求 3.7).
    pub covered: Vec<bool>,
    /// Pixel inconsistency against the current composite (需求 3.3).
    pub disagreement: Vec<f64>,
    /// Narrow contour evidence used only by the seam smoothness term.
    pub seam_disagreement: Vec<f64>,
}

impl CandidateCells {
    pub(crate) fn new(cells: usize) -> Self {
        Self {
            sharpness: vec![0.0; cells],
            covered: vec![false; cells],
            disagreement: vec![0.0; cells],
            seam_disagreement: vec![0.0; cells],
        }
    }
}

/// The ownership decision of one Capture_Station, folded one candidate at a
/// time (需求 3.4).
///
/// # Why the rounds are not re-cut
///
/// [`solve_ownership`] refines a *complete* multi-label cost grid with
/// `O(labels²)` cuts per sweep.  That needs every candidate's cost available at
/// once, which in turn needs every reprojected frame resident: at a 20 MPixel
/// station plane a single `f32` RGB layer is ~240 MB, so the eight frame upper
/// bound of a Capture_Station would be ~2 GB of layers before the first cut.
/// The render loop instead copies each candidate's pixels as soon as its round
/// is decided and drops the layer, so a round can only be decided once.
///
/// Nothing is given up in the sub-problem: one round is a *two* label cut, and a
/// two label min-cut is the exact minimum of its energy.  What is given up is
/// the cross-round refinement, which is also where 7.8 measured the cost — a
/// 528×352 grid at the 48 frame bound would need hundreds of cuts and reach the
/// 120 s budget of 需求 3.5 on its own.  One exact cut per candidate keeps the
/// station at `frames − 1` cuts.
pub(crate) struct StationFusion {
    pub(super) geometry: OwnershipGridGeometry,
    pub(super) plan: CellSamplingPlan,
    pub(super) owners: Vec<u16>,
    /// Normalised Sharpness_Score of each cell's current owner, which is also
    /// the composite's score there because the write is a hard copy (需求 3.6).
    pub(super) owner_sharpness: Vec<f64>,
    pub(super) best: Vec<f64>,
    pub(super) second: Vec<f64>,
    pub(super) candidate_counts: Vec<u32>,
    pub(super) disagreement: Vec<f64>,
    pub(super) solver_status: FusionSolverStatus,
    pub(super) clock: MonotonicGraphCutClock,
    pub(super) timeout: Duration,
    pub(super) graph_cut_seconds: f64,
    #[cfg(test)]
    pub(super) forced_owners: Option<Vec<u16>>,
}

impl StationFusion {
    /// Start the fusion of a `width × height` station plane holding
    /// `frame_count` registration-successful frames.
    pub(crate) fn new(width: u32, height: u32, frame_count: usize) -> Self {
        let geometry = OwnershipGridGeometry::for_plane(width, height);
        let cells = geometry.cell_count();
        Self {
            plan: geometry.sampling_plan(),
            geometry,
            owners: vec![NO_OWNER; cells],
            owner_sharpness: vec![0.0; cells],
            best: vec![0.0; cells],
            second: vec![0.0; cells],
            candidate_counts: vec![0; cells],
            disagreement: vec![0.0; cells],
            // 需求 3.10: a single frame has nothing to cut.
            solver_status: match frame_count {
                0 => FusionSolverStatus::NotRun,
                1 => FusionSolverStatus::SingleFrame,
                _ => FusionSolverStatus::GraphCut,
            },
            clock: MonotonicGraphCutClock::start(),
            timeout: GRAPH_CUT_TIMEOUT,
            graph_cut_seconds: 0.0,
            #[cfg(test)]
            forced_owners: None,
        }
    }

    pub(crate) fn geometry(&self) -> OwnershipGridGeometry {
        self.geometry
    }

    /// The sampling plan every candidate of this station must measure with
    /// (需求 3.2).
    pub(crate) fn sampling_plan(&self) -> CellSamplingPlan {
        self.plan
    }

    pub(crate) fn owners(&self) -> &[u16] {
        &self.owners
    }

    #[cfg(test)]
    pub(crate) fn force_test_owners(&mut self, owners: Option<Vec<u16>>) {
        assert!(
            owners
                .as_ref()
                .is_none_or(|owners| owners.len() == self.geometry.cell_count())
        );
        self.forced_owners = owners;
    }

    pub(crate) fn solver_status(&self) -> FusionSolverStatus {
        self.solver_status
    }

    pub(crate) fn graph_cut_seconds(&self) -> f64 {
        self.graph_cut_seconds
    }

    /// Record one candidate's Sharpness_Score in the cell's evidence, keeping
    /// the winning and the runner-up score of 需求 3.9.
    pub(super) fn observe(&mut self, cell: usize, sharpness: f64) {
        self.candidate_counts[cell] += 1;
        if sharpness > self.best[cell] {
            self.second[cell] = self.best[cell];
            self.best[cell] = sharpness;
        } else if sharpness > self.second[cell] {
            self.second[cell] = sharpness;
        }
    }

    /// The first frame owns every cell it covers: there is no composite to cut
    /// against yet.
    pub(crate) fn seed(&mut self, label: usize, sharpness: &[f64], covered: &[bool]) {
        let owner = owner_of(label);
        for cell in 0..self.geometry.cell_count() {
            if !covered.get(cell).copied().unwrap_or(false) {
                continue;
            }
            let score = sharpness.get(cell).copied().unwrap_or(0.0);
            self.observe(cell, score);
            self.owners[cell] = owner;
            self.owner_sharpness[cell] = score;
        }
    }

    /// Fold one candidate into the composite with a single binary cut
    /// (需求 3.4), returning the cells the candidate took.
    ///
    /// The returned mask is the *only* thing the pixel write consults, so an
    /// owner boundary is exactly an ownership cell boundary: hard, at grid
    /// resolution, with no transition band (需求 3.6).
    pub(crate) fn fold(&mut self, label: usize, cells: &CandidateCells) -> Vec<bool> {
        let count = self.geometry.cell_count();
        for cell in 0..count {
            if !cells.covered.get(cell).copied().unwrap_or(false) {
                continue;
            }
            self.observe(cell, cells.sharpness[cell]);
            // The smoothness term is a property of the cell, so a seam is
            // pushed away from every place any pair of sources disagreed.
            self.disagreement[cell] = self.disagreement[cell]
                .max(cells.disagreement[cell])
                .max(cells.seam_disagreement.get(cell).copied().unwrap_or(0.0));
        }
        let owner = owner_of(label);
        let mut grid = OwnershipCostGrid::new(
            self.geometry.columns as usize,
            self.geometry.rows as usize,
            2,
        );
        let mut base_owners = vec![NO_OWNER; count];
        for cell in 0..count {
            let owned = self.owners[cell] != NO_OWNER;
            let candidate = cells.covered.get(cell).copied().unwrap_or(false);
            let penalty = if candidate {
                mismatch_penalty(
                    cells.disagreement[cell],
                    cells.sharpness[cell],
                    self.owner_sharpness[cell],
                )
            } else {
                0.0
            };
            // Smoothness may choose among equivalent focused samples, but
            // must not trade a measured detail loss for a cheaper seam. Use
            // the same shortfall limit as the independent quality gate. A
            // displaced sharper candidate remains vetoed, never forced into
            // the output by this eligibility constraint.
            let detail_floor = self.best[cell]
                * (1.0 - f64::from(crate::panorama_utils::stack_pipeline::quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX));
            let constrain_detail = candidate
                && penalty == 0.0
                && self.best[cell] >= TEXTURED_PIXEL_SHARPNESS_FLOOR
                && (self.owner_sharpness[cell] >= detail_floor
                    || cells.sharpness[cell] >= detail_floor);
            let base_eligible =
                owned && (!constrain_detail || self.owner_sharpness[cell] >= detail_floor);
            let candidate_eligible =
                candidate && (!constrain_detail || cells.sharpness[cell] >= detail_floor);
            if base_eligible {
                base_owners[cell] = owner_of(BASE_LABEL);
            }
            grid.set_cost(
                cell,
                BASE_LABEL,
                base_eligible.then(|| data_cost(self.owner_sharpness[cell], 0.0)),
            );
            grid.set_cost(
                cell,
                CANDIDATE_LABEL,
                candidate_eligible.then(|| data_cost(cells.sharpness[cell], penalty)),
            );
            grid.set_disagreement(cell, self.disagreement[cell]);
            // Keep the graph-cut smoothness for uncertain cells, but let a
            // clearly separated sharpness winner cross an old owner boundary.
            // The quality gate accepts an owner within 5% of the winner, so a
            // gap at or below that limit has no reason to pay for a weaker
            // seam; above it, the multiplier falls smoothly and never reaches
            // zero.  This is local evidence driven and leaves the mismatch
            // veto in the data term untouched.
            // The gate measures the selected owner's shortfall against the
            // best source at this same cell.  Scale the positive Potts term
            // from the candidate's actual advantage over the current owner,
            // rather than from the winner/runner-up confidence.  The latter
            // can be high even when the candidate is the only frame that
            // clears the gate margin, so seam cost used to suppress precisely
            // the owner moves the gate needs.  Keep a non-zero floor so this
            // remains a graph-cut energy, while a measured >5% improvement
            // is effectively data-term driven at the boundary.
            let candidate_advantage = if cells.sharpness[cell] > 0.0 {
                ((cells.sharpness[cell] - self.owner_sharpness[cell]) / cells.sharpness[cell])
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
            let smoothness_scale = if candidate_advantage
                > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                && mismatch_penalty(
                    cells.disagreement[cell],
                    cells.sharpness[cell],
                    self.owner_sharpness[cell],
                ) == 0.0
            {
                0.001
            } else {
                1.0
            };
            grid.set_smoothness_scale(cell, smoothness_scale);
        }
        // Offline diagnostic only: compare the production graph cut against the
        // exact data-term minimum without changing release behaviour. This is
        // deliberately cfg(test), so a user environment variable can never
        // turn off Requirement 3.4's smoothness term in production.
        #[cfg(test)]
        let data_term_only = std::env::var_os("RAW_EDITOR_FOCUS_DATA_TERM_ONLY").is_some();
        #[cfg(not(test))]
        let data_term_only = false;
        let decided = if data_term_only {
            grid.per_cell_minimum()
        } else if self.clock.elapsed() >= self.timeout {
            // 需求 3.5: an exhausted budget resolves the remaining rounds per cell.
            self.solver_status = FusionSolverStatus::PerCellFallback;
            grid.per_cell_minimum()
        } else {
            grid.fold_candidate(&base_owners, CANDIDATE_LABEL)
        };
        self.graph_cut_seconds = self.clock.elapsed().as_secs_f64();
        let candidate_owner = owner_of(CANDIDATE_LABEL);
        // The graph cut may route a boundary, but it must not erase a
        // measured, spatially coherent sharpness win merely because the
        // positive Potts term is larger than the local data-term gain. The
        // candidate still has to be covered, exceed the existing owner's
        // score by the unchanged 5% gate margin, and pass the same
        // disagreement veto used by the data term. Four-connected support
        // keeps isolated focus-score spikes under regularisation while letting
        // a real stroke or texture patch move as one owner region.
        let coherent_decisive = region_decisive_cells(self.geometry, &self.owner_sharpness, cells);
        let quality_region_decisive =
            quality_region_decisive_cells(self.geometry, &self.owner_sharpness, &self.best, cells);
        let quality_gate_decisive =
            quality_gate_decisive_cells(self.geometry, &self.owner_sharpness, &self.best, cells);
        // The Quality_Gate's owner evidence is a lower-bound contract: once a
        // cell has a sharper owner, a later graph-cut round must not replace it
        // with a softer sample merely to save a Potts seam.  That replacement
        // was the observed production failure mode (the data-term oracle was
        // above 0.99 while the graph-cut owner fell below it).  Keep the
        // unchanged mismatch veto symmetric with this guard: a sharper but
        // geometrically inconsistent candidate is not allowed to cross the
        // boundary through the graph-cut result either.
        let quality_gate_owner_protected = (0..count)
            .map(|cell| {
                if !cells.covered.get(cell).copied().unwrap_or(false) {
                    return false;
                }
                let candidate_score = cells.sharpness[cell];
                let owner_score = self.owner_sharpness[cell];
                if !candidate_score.is_finite()
                    || !owner_score.is_finite()
                    || candidate_score < owner_score
                {
                    return true;
                }
                candidate_score > owner_score
                    && mismatch_penalty(cells.disagreement[cell], candidate_score, owner_score)
                        != 0.0
            })
            .collect::<Vec<_>>();
        if std::env::var_os("RAW_EDITOR_FOCUS_OWNERSHIP_DEBUG").is_some() {
            let covered = cells.covered.iter().filter(|&&value| value).count();
            let mismatch_veto = cells
                .covered
                .iter()
                .enumerate()
                .filter(|&(cell, &covered)| {
                    covered
                        && cells.sharpness[cell] > self.owner_sharpness[cell]
                        && cells.disagreement[cell] > OWNERSHIP_DISAGREEMENT_VETO
                })
                .count();
            let sharpness_gap = cells
                .covered
                .iter()
                .enumerate()
                .filter(|&(cell, &covered)| {
                    covered
                        && cells.sharpness[cell] > 0.0
                        && self.owner_sharpness[cell] > 0.0
                        && (cells.sharpness[cell] - self.owner_sharpness[cell])
                            / cells.sharpness[cell]
                            > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                })
                .count();
            let sharpness_gap_without_veto = cells
                .covered
                .iter()
                .enumerate()
                .filter(|&(cell, &covered)| {
                    covered
                        && cells.sharpness[cell] > 0.0
                        && self.owner_sharpness[cell] > 0.0
                        && (cells.sharpness[cell] - self.owner_sharpness[cell])
                            / cells.sharpness[cell]
                            > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                        && mismatch_penalty(
                            cells.disagreement[cell],
                            cells.sharpness[cell],
                            self.owner_sharpness[cell],
                        ) == 0.0
                })
                .count();
            let graph_cut_candidate = decided
                .iter()
                .filter(|&&value| value == owner_of(CANDIDATE_LABEL))
                .count();
            let coherent_count = coherent_decisive.iter().filter(|&&value| value).count();
            let quality_count = quality_region_decisive
                .iter()
                .filter(|&&value| value)
                .count();
            let quality_gate_count = quality_gate_decisive.iter().filter(|&&value| value).count();
            println!(
                "    - Focus ownership round label={label}: covered={covered}, graph_cut_candidate={graph_cut_candidate}, coherent_override={coherent_count}, quality_roi_override={quality_count}, quality_gate_override={quality_gate_count}, mismatch_veto={mismatch_veto}, sharpness_gap={sharpness_gap}, gap_without_veto={sharpness_gap_without_veto}"
            );
        }
        let mut took = vec![false; count];
        for cell in 0..count {
            #[cfg(test)]
            let take_candidate = self.forced_owners.as_ref().map_or_else(
                || {
                    !quality_gate_owner_protected[cell]
                        && (decided[cell] == candidate_owner
                            || coherent_decisive[cell]
                            || quality_region_decisive[cell]
                            || quality_gate_decisive[cell])
                },
                |forced| forced[cell] == owner && cells.covered.get(cell).copied().unwrap_or(false),
            );
            #[cfg(not(test))]
            let take_candidate = !quality_gate_owner_protected[cell]
                && (decided[cell] == candidate_owner
                    || coherent_decisive[cell]
                    || quality_region_decisive[cell]
                    || quality_gate_decisive[cell]);
            if !take_candidate {
                continue;
            }
            took[cell] = true;
            self.owners[cell] = owner;
            self.owner_sharpness[cell] = cells.sharpness[cell];
        }
        took
    }

    /// Sharpness_Confidence of every cell (需求 3.9).
    pub(crate) fn confidence(&self) -> Vec<f32> {
        (0..self.geometry.cell_count())
            .map(|cell| {
                cell_confidence(
                    self.best[cell],
                    self.second[cell],
                    self.candidate_counts[cell] as usize,
                )
            })
            .collect()
    }

    /// Preserve the exact winner and runner-up measurements used by
    /// `confidence()` for report-only diagnostics.  This is deliberately a
    /// compact ownership-cell grid and cannot influence the solved owners.
    pub(crate) fn sharpness_evidence(&self) -> SharpnessEvidenceGrid {
        let cells = (0..self.geometry.cell_count())
            .map(|cell| SharpnessCellEvidence {
                winner_score: self.best[cell] as f32,
                runner_up_score: self.second[cell] as f32,
                owner_score: self.owner_sharpness[cell] as f32,
                candidate_count: self.candidate_counts[cell],
                disagreement: self.disagreement[cell] as f32,
            })
            .collect();
        SharpnessEvidenceGrid {
            columns: self.geometry.columns,
            rows: self.geometry.rows,
            cell_size_px: self.geometry.cell_size_px,
            cells,
        }
    }

    /// A cell is textured when any observed candidate reaches the glossary's
    /// fixed Sharpness_Score floor. `best` is updated for every candidate by
    /// `seed`/`fold`, so this does not infer texture from confidence.
    pub(crate) fn textured_cells(&self) -> Vec<bool> {
        (0..self.geometry.cell_count())
            .map(|cell| {
                self.owners[cell] != NO_OWNER && self.best[cell] >= TEXTURED_PIXEL_SHARPNESS_FLOOR
            })
            .collect()
    }

    /// Cells whose every candidate stayed below the station's P10 winning
    /// Sharpness_Score (需求 3.8).
    ///
    /// The winning score of a cell is the largest one its candidates reached,
    /// so "every candidate below the percentile" is exactly "the winner below
    /// the percentile".  Only owned cells take part: an uncovered cell has no
    /// candidate to be below anything.
    pub(crate) fn low_sharpness_cells(&self) -> Vec<bool> {
        let winning = (0..self.geometry.cell_count())
            .filter(|&cell| self.owners[cell] != NO_OWNER)
            .map(|cell| self.best[cell])
            .collect::<Vec<_>>();
        if winning.is_empty() {
            return vec![false; self.geometry.cell_count()];
        }
        let threshold = low_sharpness_threshold(&winning);
        (0..self.geometry.cell_count())
            .map(|cell| self.owners[cell] != NO_OWNER && self.best[cell] < threshold)
            .collect()
    }

    /// The low sharpness regions of 需求 3.8, in world coordinates.
    pub(crate) fn low_sharpness_region_records(
        &self,
        world_origin: (f64, f64),
    ) -> Vec<LowSharpnessRegionRecord> {
        let low = self.low_sharpness_cells();
        let regions = low_sharpness_regions(self.geometry.columns, self.geometry.rows, &low);
        low_sharpness_region_records(&self.geometry, &regions, world_origin)
    }
}
