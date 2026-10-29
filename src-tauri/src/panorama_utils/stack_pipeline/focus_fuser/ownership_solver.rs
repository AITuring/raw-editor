//! Candidate order and the multi-label ownership solver (需求 3.4, 3.5, 3.10).
//!
//! The deterministic candidate order, the ownership cost grid and its exact
//! two-label cuts, the timeout clock and the whole-grid refinement sweeps.

use super::super::super::seam_cut::cut_grid_weighted;
use super::super::super::seam_cut::pairwise_weight;
use super::super::degradation::DegradationLedger;
use super::super::degradation::FUSION_GRAPH_CUT_TIMEOUT;
use super::super::report::FusionReport;
use super::super::report::FusionSolverStatus;
use super::super::report::OwnershipGridSize;
use serde_json::json;
use std::time::Duration;
use std::time::Instant;

use super::{OwnershipGridGeometry, candidate_wins_cell};

// ---------------------------------------------------------------------------
// Candidate order (需求 3.4)
// ---------------------------------------------------------------------------

/// One registration-successful frame competing for the cells of a
/// Capture_Station.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FusionFrame {
    /// Absolute path, which is also the tie-break key.
    pub path: String,
    /// Median Sharpness_Score over the frame's valid pixels.
    pub median_sharpness: f64,
    /// `true` for the frame [`crate::panorama_utils::stack_pipeline::intra_station::select_anchor`] chose.
    pub is_anchor: bool,
}

/// Order the candidates of one station are folded into the composite in
/// (需求 3.4).
///
/// Anchor frame first, then descending median Sharpness_Score, ties resolved by
/// ascending absolute path.  The order fixes three things that would otherwise
/// depend on import order: which frame is the initial base, which label wins a
/// cost tie (the earlier one, see [`candidate_wins_cell`]), and the sequence of
/// expansion moves — so two runs over the same photographs produce the same
/// ownership grid bit for bit.
pub(crate) fn candidate_order(frames: &[FusionFrame]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..frames.len()).collect();
    order.sort_by(|&left, &right| {
        let (a, b) = (&frames[left], &frames[right]);
        b.is_anchor
            .cmp(&a.is_anchor)
            .then_with(|| b.median_sharpness.total_cmp(&a.median_sharpness))
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| left.cmp(&right))
    });
    order
}

/// 需求 3.10: a station with at most one registration-successful frame has
/// nothing to cut.
pub(crate) fn skips_graph_cut(candidate_count: usize) -> bool {
    candidate_count <= 1
}

// ---------------------------------------------------------------------------
// Multi-label ownership solver (需求 3.4, 3.5, 3.10)
// ---------------------------------------------------------------------------

/// Reserved ownership identifier of a cell no candidate covers (需求 3.7).
pub(crate) const NO_OWNER: u16 = 0;

/// Wall-clock budget of the multi-label loop (需求 3.5).
pub(crate) const GRAPH_CUT_TIMEOUT: Duration = Duration::from_secs(120);

/// Upper bound on expansion sweeps over the candidate set.
///
/// Every accepted move strictly lowers the energy, so the loop terminates on its
/// own; this bound only keeps the worst case bounded well inside
/// [`GRAPH_CUT_TIMEOUT`] on a 528×352 grid.
pub(crate) const GRAPH_CUT_MAX_SWEEPS: usize = 4;

/// A move has to lower the energy by at least this much to be adopted, so
/// floating-point noise cannot produce an endless exchange of equal labelings.
pub(super) const GRAPH_CUT_ENERGY_EPSILON: f64 = 1e-9;

/// Energy charged to a cell that some candidate covers but no owner claims.
///
/// 需求 3.4 requires *every* covered cell to end up with exactly one owner.
/// Expressing that as a large finite penalty rather than a hard constraint keeps
/// the energy comparable across partial labelings, which is what lets the loop
/// below reject a move that would make things worse.  It is far above any
/// reachable data plus smoothness cost, so a labeling that leaves a covered cell
/// unowned can never win.
pub(super) const UNASSIGNED_CELL_COST: f64 = 1.0e6;

/// `fixed` value that pins a cell to the candidate of the current round.
pub(super) const PIN_CANDIDATE: i8 = 1;

/// `fixed` value that pins a cell to its current owner.
pub(super) const PIN_BASE: i8 = -1;

/// Ownership identifier of candidate `label`.  Identifiers are one-based so that
/// `0` stays the reserved "no owner" value of 需求 3.7.
pub(crate) fn owner_of(label: usize) -> u16 {
    u16::try_from(label + 1).expect("ownership label fits in u16")
}

/// Monotonic clock of the multi-label loop, injectable so the timeout path of
/// 需求 3.5 is testable without waiting two minutes.
pub(crate) trait GraphCutClock {
    /// Time since the solve started.  Never decreases.
    fn elapsed(&self) -> Duration;
}

/// The production clock: `Instant`, which is monotonic on every supported
/// platform and unaffected by a wall-clock adjustment mid-run.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MonotonicGraphCutClock {
    pub(super) started: Instant,
}

impl MonotonicGraphCutClock {
    pub(crate) fn start() -> Self {
        Self {
            started: Instant::now(),
        }
    }
}

impl GraphCutClock for MonotonicGraphCutClock {
    fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// The per-cell, per-candidate selection costs of one Capture_Station.
///
/// The solver deliberately takes *costs*, not images: [`data_cost`] and
/// [`cell_disagreement`] above already turn the pixels of a cell into the two
/// numbers 需求 3.3 and 3.4 define, and keeping the minimisation a pure function
/// of the cost grid is what makes it comparable against an exhaustive search.
///
/// `costs[cell * labels + label]` is `None` when that candidate does not cover
/// the cell, which is the only way a candidate is excluded from a cell — 需求
/// 3.11's "only covered cells get an owner" falls out of the same field.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OwnershipCostGrid {
    pub(super) columns: usize,
    pub(super) rows: usize,
    pub(super) labels: usize,
    pub(super) costs: Vec<Option<f64>>,
    pub(super) disagreement: Vec<f64>,
    /// Per-cell multiplier for the positive Potts smoothness term.  The
    /// default is one; StationFusion lowers it only where the winner is
    /// materially sharper than the runner-up.  Keeping the scale on the grid
    /// makes the same energy visible to the min-cut, descent and report tests.
    pub(super) smoothness_scale: Vec<f64>,
}

impl OwnershipCostGrid {
    /// A grid where no candidate covers anything yet.
    pub(crate) fn new(columns: usize, rows: usize, labels: usize) -> Self {
        let cells = columns * rows;
        Self {
            columns,
            rows,
            labels,
            costs: vec![None; cells * labels],
            disagreement: vec![0.0; cells],
            smoothness_scale: vec![1.0; cells],
        }
    }

    pub(crate) fn columns(&self) -> usize {
        self.columns
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn label_count(&self) -> usize {
        self.labels
    }

    pub(crate) fn cell_count(&self) -> usize {
        self.columns * self.rows
    }

    /// Record [`data_cost`] of one candidate in one cell.  `None` means the
    /// candidate has no valid pixels there.
    pub(crate) fn set_cost(&mut self, cell: usize, label: usize, cost: Option<f64>) {
        assert!(cell < self.cell_count(), "cell out of range");
        assert!(label < self.labels, "label out of range");
        self.costs[cell * self.labels + label] = cost;
    }

    /// Record the cross-boundary pixel difference of one cell, which drives the
    /// smoothness term of 需求 3.4.
    ///
    /// One value per *cell*, not per candidate pair: the smoothness term has to
    /// be a fixed function of the grid for the total cost to be well defined at
    /// all, and 需求 3.4 describes it as a property of the boundary between two
    /// cells.  Callers pass the strongest disagreement the cell's candidates
    /// exhibit, so a seam is pushed away from any place the sources disagree.
    pub(crate) fn set_disagreement(&mut self, cell: usize, disagreement: f64) {
        assert!(cell < self.cell_count(), "cell out of range");
        self.disagreement[cell] = disagreement.clamp(0.0, 1.0);
    }

    /// Set the local multiplier of the positive smoothness term.  A small
    /// positive floor is intentional: a confident source may nearly remove a
    /// seam penalty, but the graph-cut energy still retains a regulariser.
    pub(crate) fn set_smoothness_scale(&mut self, cell: usize, scale: f64) {
        assert!(cell < self.cell_count(), "cell out of range");
        self.smoothness_scale[cell] = scale.clamp(0.001, 1.0);
    }

    pub(super) fn edge_smoothness(&self, cell: usize, neighbour: usize) -> f64 {
        // Couple the two local confidence scales without letting an uncertain
        // endpoint restore the full seam cost everywhere.  The geometric mean
        // keeps a strictly positive Potts term, while allowing a genuinely
        // sharp winner to cross a boundary when both cells support that move.
        // Using `max` here made a confident cell ineffective next to any
        // uncertain neighbour and kept the production result close to the
        // uncorrected graph-cut labeling.
        let scale = (self.smoothness_scale[cell] * self.smoothness_scale[neighbour]).sqrt();
        pairwise_weight(self.disagreement[cell], self.disagreement[neighbour])
            * scale.clamp(0.001, 1.0)
    }

    pub(crate) fn cost(&self, cell: usize, label: usize) -> Option<f64> {
        self.costs[cell * self.labels + label]
    }

    /// `true` when at least one candidate covers the cell (需求 3.11).
    pub(crate) fn covered(&self, cell: usize) -> bool {
        (0..self.labels).any(|label| self.cost(cell, label).is_some())
    }

    pub(super) fn label_of(&self, owner: u16) -> Option<usize> {
        (owner != NO_OWNER).then(|| usize::from(owner - 1))
    }

    /// The 4-neighbourhood of a cell, in row-major order.
    pub(super) fn neighbours(&self, cell: usize) -> [Option<usize>; 4] {
        let columns = self.columns.max(1);
        let (x, y) = (cell % columns, cell / columns);
        [
            (y > 0).then(|| cell - self.columns),
            (x > 0).then(|| cell - 1),
            (x + 1 < self.columns).then(|| cell + 1),
            (y + 1 < self.rows).then(|| cell + self.columns),
        ]
    }

    /// Total cost of one labeling: the data term of 需求 3.4 plus the smoothness
    /// term [`pairwise_weight`] charges on every 4-neighbour edge whose two cells
    /// have different owners.
    ///
    /// A cell no candidate covers has no owner and takes part in no edge: the
    /// border of the coverage is not a seam, and charging it would make the cost
    /// of an interior decision depend on the shape of the station footprint.
    /// Claiming a cell a candidate does not cover is not a labeling at all, hence
    /// the infinity.
    pub(crate) fn energy(&self, owners: &[u16]) -> f64 {
        assert_eq!(owners.len(), self.cell_count(), "owner grid size");
        let mut total = 0.0;
        for (cell, &owner) in owners.iter().enumerate() {
            match self.label_of(owner) {
                Some(label) => match self.cost(cell, label) {
                    Some(cost) => total += cost,
                    None => return f64::INFINITY,
                },
                None if self.covered(cell) => total += UNASSIGNED_CELL_COST,
                None => {}
            }
        }
        for cell in 0..self.cell_count() {
            if owners[cell] == NO_OWNER {
                continue;
            }
            // Right and down only, so each edge is charged once.
            let (x, y) = (cell % self.columns, cell / self.columns);
            for neighbour in [
                (x + 1 < self.columns).then(|| cell + 1),
                (y + 1 < self.rows).then(|| cell + self.columns),
            ]
            .into_iter()
            .flatten()
            {
                if owners[neighbour] != NO_OWNER && owners[neighbour] != owners[cell] {
                    total += self.edge_smoothness(cell, neighbour);
                }
            }
        }
        total
    }

    /// Per-cell minimum cost owner: the fallback rule of 需求 3.5, and also the
    /// labeling the timeout path falls back to.  Ties keep the earlier candidate
    /// of [`candidate_order`].
    pub(crate) fn per_cell_minimum(&self) -> Vec<u16> {
        (0..self.cell_count())
            .map(|cell| {
                let mut best: Option<(usize, f64)> = None;
                for label in 0..self.labels {
                    let Some(cost) = self.cost(cell, label) else {
                        continue;
                    };
                    let wins =
                        best.is_none_or(|(_, incumbent)| candidate_wins_cell(incumbent, cost));
                    if wins {
                        best = Some((label, cost));
                    }
                }
                best.map_or(NO_OWNER, |(label, _)| owner_of(label))
            })
            .collect()
    }

    /// One binary cut over the grid: every participating cell either keeps its
    /// current owner or switches to `candidate` (需求 3.4).
    ///
    /// `participates` decides, from a cell's current owner, whether the cut may
    /// move it.  Two move families use it:
    ///
    /// - *fold in a candidate* — every cell participates.  This is the
    ///   "current composite vs next candidate" cut the multi-label loop runs once
    ///   per candidate.
    /// - *re-cut one owner* — only cells currently owned by one specific frame
    ///   participate, which makes the sub-problem an **exact** minimisation of
    ///   [`Self::energy`] over the allowed moves (see the correction below), so it
    ///   can pull the loop out of a local minimum the first family leaves behind.
    ///
    /// `preference[i] = data_cost(current) − data_cost(candidate)` plus a
    /// correction; the smoothness term is the one [`cut_grid`] applies internally
    /// from `disagreement`.  `fixed` pins the cases the cut must not decide: a
    /// cell only the current owner covers keeps it, a cell only the candidate
    /// covers goes to the candidate, a cell already owned by the candidate cannot
    /// move, and a cell the caller excluded keeps its owner.
    ///
    /// # The correction
    ///
    /// [`cut_grid`] charges the smoothness of an edge when the two cells fall on
    /// different *sides of the cut*, while [`Self::energy`] charges it when they
    /// have different *owners*.  For an edge to a pinned-to-base neighbour whose
    /// owner already differs from the free cell's current owner — a cell no
    /// candidate covers, or a cell held by a third frame — the true cost is the
    /// same either way, but the cut only charges it after the move.  Adding that
    /// weight to `preference` cancels the bias exactly: [`cut_grid`]'s two
    /// terminal capacities are defined only up to their difference, so shifting
    /// the difference is not a change to its min-cut semantics.
    pub(super) fn binary_move(
        &self,
        owners: &[u16],
        candidate: usize,
        participates: impl Fn(u16) -> bool,
    ) -> Vec<u16> {
        let count = self.cell_count();
        let candidate_owner = owner_of(candidate);
        let mut preference = vec![0.0; count];
        let mut fixed = vec![0i8; count];
        for (cell, &owner) in owners.iter().enumerate() {
            fixed[cell] = match (self.label_of(owner), self.cost(cell, candidate)) {
                // The candidate has no pixels here — including every cell no
                // candidate covers at all, which keeps `NO_OWNER`.
                (_, None) => PIN_BASE,
                // Only the candidate covers the cell, or the cell is already its
                // own: a move can hand a label out, never take one away.
                (None, Some(_)) => PIN_CANDIDATE,
                (Some(current), Some(_)) if current == candidate => PIN_CANDIDATE,
                (Some(_), Some(_)) if !participates(owner) => PIN_BASE,
                (Some(_), Some(_)) => 0,
            };
        }
        let mut edge_weights = vec![(None, None); count];
        for cell in 0..count {
            if fixed[cell] != 0 {
                continue;
            }
            let current = self
                .label_of(owners[cell])
                .expect("a free cell has an owner");
            let base_cost = self
                .cost(cell, current)
                .expect("the current owner covers its own cell");
            let candidate_cost = self
                .cost(cell, candidate)
                .expect("a free cell is covered by the candidate");
            preference[cell] = base_cost - candidate_cost;
        }
        // Reparameterise every Potts edge into the binary cut. When the two
        // current owners differ, charging the full edge weight in the cut
        // misprices the free-free (base, base) state. The half-edge plus two
        // terminal corrections is exactly equivalent and keeps the cut exact.
        for cell in 0..count {
            let x = cell % self.columns;
            let y = cell / self.columns;
            for neighbour in [
                (x + 1 < self.columns).then(|| cell + 1),
                (y + 1 < self.rows).then(|| cell + self.columns),
            ]
            .into_iter()
            .flatten()
            {
                let weight = self.edge_smoothness(cell, neighbour);
                let base_i = owners[cell];
                let base_j = owners[neighbour];
                let pair = |left: u16, right: u16| {
                    if left != NO_OWNER && right != NO_OWNER && left != right {
                        weight
                    } else {
                        0.0
                    }
                };
                if fixed[cell] == 0 && fixed[neighbour] == 0 {
                    let e00 = pair(base_i, base_j);
                    let e01 = pair(base_i, candidate_owner);
                    let e10 = pair(candidate_owner, base_j);
                    let e11 = 0.0;
                    let edge = ((e01 + e10 - e00 - e11) * 0.5).max(0.0);
                    if neighbour == cell + 1 {
                        edge_weights[cell].0 = Some(edge);
                    } else {
                        edge_weights[cell].1 = Some(edge);
                    }
                    preference[cell] -= e10 - e00 - edge;
                    preference[neighbour] -= e01 - e00 - edge;
                } else if fixed[cell] == 0 {
                    if neighbour == cell + 1 {
                        edge_weights[cell].0 = Some(0.0);
                    } else {
                        edge_weights[cell].1 = Some(0.0);
                    }
                    let (e0, e1) = if fixed[neighbour] == PIN_CANDIDATE {
                        (
                            pair(base_i, candidate_owner),
                            pair(candidate_owner, candidate_owner),
                        )
                    } else {
                        (pair(base_i, base_j), pair(candidate_owner, base_j))
                    };
                    preference[cell] -= e1 - e0;
                } else if fixed[neighbour] == 0 {
                    if neighbour == cell + 1 {
                        edge_weights[cell].0 = Some(0.0);
                    } else {
                        edge_weights[cell].1 = Some(0.0);
                    }
                    let (e0, e1) = if fixed[cell] == PIN_CANDIDATE {
                        (
                            pair(candidate_owner, base_j),
                            pair(candidate_owner, candidate_owner),
                        )
                    } else {
                        (pair(base_i, base_j), pair(base_i, candidate_owner))
                    };
                    preference[neighbour] -= e1 - e0;
                }
            }
        }
        let cut = cut_grid_weighted(
            self.columns,
            self.rows,
            &preference,
            &self.disagreement,
            &fixed,
            &edge_weights,
        );
        cut.iter()
            .enumerate()
            .map(|(cell, &side)| {
                let took_candidate = side > 0 && self.cost(cell, candidate).is_some();
                if took_candidate {
                    candidate_owner
                } else {
                    owners[cell]
                }
            })
            .collect()
    }

    /// One "current composite vs next candidate" cut over the whole grid
    /// (需求 3.4).
    ///
    /// With exactly two labels — the composite and the candidate — this single
    /// cut *is* the global minimum of [`Self::energy`], which is what lets
    /// [`StationFusion`] decide a round and drop the candidate's pixels
    /// immediately instead of keeping every frame resident for a refinement
    /// sweep.
    pub(crate) fn fold_candidate(&self, owners: &[u16], candidate: usize) -> Vec<u16> {
        self.binary_move(owners, candidate, |_| true)
    }

    /// Hand every cell `candidate` covers to `candidate`, leaving the rest as
    /// they are.
    ///
    /// No cut can propose this.  A binary cut moves cells towards one candidate
    /// while the others hold still, and its sub-problem prices no seam between
    /// two cells that both hold still — so when two frames each own half the grid
    /// and a third frame is better than both *everywhere*, every single move away
    /// from that split looks like it adds a seam that is in fact already there.
    /// The situation is not exotic: it is what a station looks like when one frame
    /// is simply the sharpest one.  The proposal is evaluated against
    /// [`Self::energy`] like any other, so it can only be adopted when it wins.
    pub(super) fn collapse_move(&self, owners: &[u16], candidate: usize) -> Vec<u16> {
        let candidate_owner = owner_of(candidate);
        owners
            .iter()
            .enumerate()
            .map(|(cell, &owner)| {
                if self.cost(cell, candidate).is_some() {
                    candidate_owner
                } else {
                    owner
                }
            })
            .collect()
    }

    /// Deterministic single-cell descent on the same energy [`Self::energy`]
    /// defines.
    ///
    /// The binary sub-problem [`cut_grid`] solves charges no smoothness between
    /// two cells that both keep their (different) owners, so a sequence of
    /// expansion moves can stop at a labeling a single cell change still
    /// improves.  This pass removes exactly those, visits cells in row-major
    /// order, and keeps the current owner on a tie — so it can only lower the
    /// cost, never introduce a choice that depends on iteration order.
    pub(super) fn local_descent(&self, owners: &mut [u16]) -> bool {
        let mut changed = false;
        for cell in 0..self.cell_count() {
            let mut best = owners[cell];
            let mut best_cost = self.cell_cost(owners, cell, owners[cell]);
            for label in 0..self.labels {
                let owner = owner_of(label);
                if owner == owners[cell] || self.cost(cell, label).is_none() {
                    continue;
                }
                let cost = self.cell_cost(owners, cell, owner);
                if cost < best_cost - GRAPH_CUT_ENERGY_EPSILON {
                    best = owner;
                    best_cost = cost;
                }
            }
            if best != owners[cell] {
                owners[cell] = best;
                changed = true;
            }
        }
        changed
    }

    /// Energy contribution of one cell given its neighbours' owners.
    pub(super) fn cell_cost(&self, owners: &[u16], cell: usize, owner: u16) -> f64 {
        let data = match self.label_of(owner) {
            Some(label) => match self.cost(cell, label) {
                Some(cost) => cost,
                None => return f64::INFINITY,
            },
            None if self.covered(cell) => UNASSIGNED_CELL_COST,
            None => 0.0,
        };
        if owner == NO_OWNER {
            return data;
        }
        data + self
            .neighbours(cell)
            .into_iter()
            .flatten()
            .filter(|&neighbour| owners[neighbour] != NO_OWNER && owners[neighbour] != owner)
            .map(|neighbour| self.edge_smoothness(cell, neighbour))
            .sum::<f64>()
    }
}

/// The ownership decision of one Capture_Station (需求 3.4, 3.5, 3.10).
///
/// Decision only: the pixel write of 需求 3.6 consumes `owners` in task 7.10.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OwnershipSolution {
    /// One owner per ownership cell, [`NO_OWNER`] where nothing is covered.
    pub owners: Vec<u16>,
    pub solver_status: FusionSolverStatus,
    pub graph_cut_seconds: f64,
}

impl OwnershipSolution {
    /// Number of cells that ended up with an owner.
    pub(crate) fn owned_cells(&self) -> u64 {
        self.owners
            .iter()
            .filter(|&&owner| owner != NO_OWNER)
            .count() as u64
    }

    /// Write the solver observations of 需求 3.5 into the Stack_Report.  The
    /// pixel counts, the confidence summary and the low-sharpness regions belong
    /// to tasks 7.10 and 7.11 and are left untouched.
    pub(crate) fn write_report(&self, report: &mut FusionReport, geometry: &OwnershipGridGeometry) {
        report.cell_size_px = geometry.cell_size_px;
        report.grid = OwnershipGridSize {
            columns: geometry.columns,
            rows: geometry.rows,
        };
        report.solver_status = self.solver_status;
        report.graph_cut_seconds = self.graph_cut_seconds;
    }
}

/// Record the degraded path of 需求 3.5 when the cut ran out of time.
pub(crate) fn record_solver_degradation(
    ledger: &mut DegradationLedger,
    solution: &OwnershipSolution,
) {
    if solution.solver_status == FusionSolverStatus::PerCellFallback {
        ledger.record(
            FUSION_GRAPH_CUT_TIMEOUT,
            json!({
                "graph_cut_seconds": solution.graph_cut_seconds,
                "timeout_seconds": GRAPH_CUT_TIMEOUT.as_secs_f64(),
            }),
        );
    }
}

/// Solve the ownership of one Capture_Station under the production budget.
pub(crate) fn solve_ownership(grid: &OwnershipCostGrid) -> OwnershipSolution {
    solve_ownership_with_clock(grid, GRAPH_CUT_TIMEOUT, &MonotonicGraphCutClock::start())
}

/// A labeling together with its total cost, so a refinement pass never
/// recomputes the whole energy to decide whether it made progress.
pub(super) struct Labeling {
    pub(super) owners: Vec<u16>,
    pub(super) energy: f64,
}

impl Labeling {
    pub(super) fn of(grid: &OwnershipCostGrid, owners: Vec<u16>) -> Self {
        let energy = grid.energy(&owners);
        Self { owners, energy }
    }

    /// Take `trial` when it lowers the cost.  Every move of the solver goes
    /// through here, so the cost is monotone regardless of which move proposed
    /// the labeling and how good its own sub-problem was.
    pub(super) fn adopt(&mut self, grid: &OwnershipCostGrid, trial: Vec<u16>) -> bool {
        let energy = grid.energy(&trial);
        if energy < self.energy - GRAPH_CUT_ENERGY_EPSILON {
            self.owners = trial;
            self.energy = energy;
            true
        } else {
            false
        }
    }
}

/// Fold the candidates into a composite one at a time, in the order the caller
/// sorted them into (需求 3.4).
///
/// The first round has no base yet, so every cell the first candidate covers is
/// pinned to it; every later round is the "current composite vs next candidate"
/// cut. `false` means the budget ran out mid-way.
pub(super) fn fold_in_candidates(
    grid: &OwnershipCostGrid,
    timeout: Duration,
    clock: &impl GraphCutClock,
) -> (Labeling, bool) {
    let order = (0..grid.label_count()).collect::<Vec<_>>();
    fold_in_candidates_order(grid, timeout, clock, &order)
}

pub(super) fn fold_in_candidates_order(
    grid: &OwnershipCostGrid,
    timeout: Duration,
    clock: &impl GraphCutClock,
    order: &[usize],
) -> (Labeling, bool) {
    let mut labeling = Labeling::of(grid, vec![NO_OWNER; grid.cell_count()]);
    for &candidate in order {
        if clock.elapsed() >= timeout {
            return (labeling, false);
        }
        let trial = grid.binary_move(&labeling.owners, candidate, |_| true);
        labeling.adopt(grid, trial);
    }
    (labeling, true)
}

pub(super) fn deterministic_restart_orders(label_count: usize) -> Vec<Vec<usize>> {
    let original = (0..label_count).collect::<Vec<_>>();
    if label_count <= 1 {
        return vec![original];
    }
    // Small ownership grids are also where the near-optimal property is
    // checked. Enumerating the fixed label-order permutations is a solver
    // restart, not an owner-assignment search; production stations with many
    // labels use the linear adjacent-swap schedule below.
    if label_count <= 4 {
        fn visit(prefix: &mut Vec<usize>, remaining: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if remaining.is_empty() {
                out.push(prefix.clone());
                return;
            }
            for index in 0..remaining.len() {
                let value = remaining.remove(index);
                prefix.push(value);
                visit(prefix, remaining, out);
                prefix.pop();
                remaining.insert(index, value);
            }
        }
        let mut out = Vec::new();
        visit(&mut Vec::new(), &mut original.clone(), &mut out);
        return out;
    }
    let mut out = vec![original.clone()];
    for pivot in 1..label_count.saturating_sub(1) {
        let mut order = original.clone();
        order.swap(pivot, pivot + 1);
        out.push(order);
    }
    out
}

/// Drive one labeling down to a local minimum of [`OwnershipCostGrid::energy`].
///
/// Each sweep offers three move families, all deterministic, exact ones first:
///
/// - *re-cut one owner*: one cut per ordered pair of candidates, restricted to
///   the cells one of them currently owns, which makes the sub-problem exact;
/// - *single cell*: [`OwnershipCostGrid::local_descent`];
/// - *collapse* and *fold in*: whole-grid proposals that can leave a local
///   minimum the exact moves cannot.
///
/// Sweeps stop as soon as none of them improves, and after at most
/// [`GRAPH_CUT_MAX_SWEEPS`].  `false` means the budget ran out.
pub(super) fn refine_labeling(
    grid: &OwnershipCostGrid,
    labeling: &mut Labeling,
    timeout: Duration,
    clock: &impl GraphCutClock,
) -> bool {
    let labels = grid.label_count();
    for _ in 0..GRAPH_CUT_MAX_SWEEPS {
        let mut improved = false;
        // Exact moves first. A cut whose sub-problem is only approximate can
        // still lower the cost, and would then be adopted before an exact move
        // that lowers it further had a chance to run.
        for base in 0..labels {
            let base_owner = owner_of(base);
            if !labeling.owners.contains(&base_owner) {
                // Nothing to re-cut, and a cut over a fully pinned grid is the
                // most expensive way to learn that.
                continue;
            }
            for candidate in 0..labels {
                if base == candidate {
                    continue;
                }
                if clock.elapsed() >= timeout {
                    return false;
                }
                let trial =
                    grid.binary_move(&labeling.owners, candidate, |owner| owner == base_owner);
                improved |= labeling.adopt(grid, trial);
            }
        }
        let mut descended = labeling.owners.clone();
        if grid.local_descent(&mut descended) {
            improved |= labeling.adopt(grid, descended);
        }
        for candidate in 0..labels {
            if clock.elapsed() >= timeout {
                return false;
            }
            let trial = grid.collapse_move(&labeling.owners, candidate);
            improved |= labeling.adopt(grid, trial);
            let trial = grid.binary_move(&labeling.owners, candidate, |_| true);
            improved |= labeling.adopt(grid, trial);
        }
        if !improved {
            break;
        }
    }
    true
}

/// Multi-label ownership by repeated binary cuts (需求 3.4), with the timeout
/// fallback of 需求 3.5 and the single-frame shortcut of 需求 3.10.
///
/// Every move is a [`cut_grid`] call or a whole-label proposal, and every move is
/// adopted only when it lowers [`OwnershipCostGrid::energy`] — so the solver is a
/// descent, and where it starts matters.  Two deterministic starts are refined
/// and the cheaper one wins:
///
/// 1. the composite built by folding the candidates in one at a time, which is
///    the seam-aware labeling and the usual winner;
/// 2. the per-cell minimum of 需求 3.5, which ignores seams entirely.
///
/// The second start is not redundant. A cut moves cells towards one candidate
/// while the rest hold still, so it cannot cross a labeling that is only reachable
/// by moving two owners onto two *different* labels at once; starting from the
/// seam-free labeling reaches those from the other side.
///
/// The result is a local minimum with respect to every move family above, and on
/// small grids it is the global minimum in the overwhelming majority of cases —
/// but not provably in all of them.  See
/// `graph_cut_matches_the_exhaustive_minimum` for the measured bound.
pub(crate) fn solve_ownership_with_clock(
    grid: &OwnershipCostGrid,
    timeout: Duration,
    clock: &impl GraphCutClock,
) -> OwnershipSolution {
    let finish = |owners: Vec<u16>, solver_status: FusionSolverStatus| OwnershipSolution {
        owners,
        solver_status,
        graph_cut_seconds: clock.elapsed().as_secs_f64(),
    };
    // 需求 3.10: one candidate (or none) leaves nothing to decide — every
    // covered cell points at that source.
    if skips_graph_cut(grid.label_count()) {
        let status = if grid.label_count() == 0 {
            FusionSolverStatus::NotRun
        } else {
            FusionSolverStatus::SingleFrame
        };
        return finish(grid.per_cell_minimum(), status);
    }
    // 需求 3.5: an exhausted budget resolves everything per cell instead.
    let (mut best, in_budget) = fold_in_candidates(grid, timeout, clock);
    if !in_budget || !refine_labeling(grid, &mut best, timeout, clock) {
        return finish(grid.per_cell_minimum(), FusionSolverStatus::PerCellFallback);
    }
    // Alpha-expansion is a descent method and can stop in a local minimum
    // whose escape needs two labels to move at once. Fixed adjacent-swap
    // starts provide a deterministic escape while preserving the original
    // candidate order as the equal-cost tie winner.
    for order in deterministic_restart_orders(grid.label_count())
        .into_iter()
        .skip(1)
    {
        if clock.elapsed() >= timeout {
            return finish(grid.per_cell_minimum(), FusionSolverStatus::PerCellFallback);
        }
        let (mut trial, in_budget) = fold_in_candidates_order(grid, timeout, clock, &order);
        if !in_budget || !refine_labeling(grid, &mut trial, timeout, clock) {
            return finish(grid.per_cell_minimum(), FusionSolverStatus::PerCellFallback);
        }
        if trial.energy < best.energy - GRAPH_CUT_ENERGY_EPSILON {
            best = trial;
        }
    }
    let mut seam_free = Labeling::of(grid, grid.per_cell_minimum());
    if !refine_labeling(grid, &mut seam_free, timeout, clock) {
        return finish(grid.per_cell_minimum(), FusionSolverStatus::PerCellFallback);
    }
    if seam_free.energy < best.energy - GRAPH_CUT_ENERGY_EPSILON {
        best = seam_free;
    }
    finish(best.owners, FusionSolverStatus::GraphCut)
}
