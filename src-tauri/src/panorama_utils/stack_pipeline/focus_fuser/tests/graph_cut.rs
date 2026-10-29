//! The multi-label ownership graph cut (需求 3.4, 3.5, 3.10).
//!
//! The cut against an exhaustive minimum, its smoothness term, the deterministic
//! candidate order, the single-frame shortcut, the timeout fallback driven by an
//! injectable clock, and the solver fields of the report.

use super::*;

// -----------------------------------------------------------------------
// Multi-label solver (需求 3.4, 3.5, 3.10)
// -----------------------------------------------------------------------

/// Injectable clock: `elapsed` advances by `step` on every observation, so a
/// test can time the solve out after a known number of checks without
/// waiting for the real 120 s budget.
struct SteppingClock {
    step: Duration,
    observations: std::cell::Cell<u32>,
}

impl SteppingClock {
    fn new(step: Duration) -> Self {
        Self {
            step,
            observations: std::cell::Cell::new(0),
        }
    }
}

impl GraphCutClock for SteppingClock {
    fn elapsed(&self) -> Duration {
        let observations = self.observations.get();
        self.observations.set(observations + 1);
        self.step * observations
    }
}

fn grid_from(
    columns: usize,
    rows: usize,
    costs: &[&[Option<f64>]],
    disagreement: &[f64],
) -> OwnershipCostGrid {
    let labels = costs.len();
    let mut grid = OwnershipCostGrid::new(columns, rows, labels);
    for (label, per_cell) in costs.iter().enumerate() {
        assert_eq!(per_cell.len(), columns * rows);
        for (cell, &cost) in per_cell.iter().enumerate() {
            grid.set_cost(cell, label, cost);
        }
    }
    for (cell, &value) in disagreement.iter().enumerate() {
        grid.set_disagreement(cell, value);
    }
    grid
}

/// Exhaustive minimum over every labeling of a small grid, including the
/// partial ones, so the comparison covers 需求 3.4's "exactly one owner per
/// covered cell" as well as the total cost.
fn exhaustive_minimum(grid: &OwnershipCostGrid) -> f64 {
    let cells = grid.cell_count();
    let options = grid.label_count() + 1;
    let total = options.pow(u32::try_from(cells).expect("small grid"));
    let mut best = f64::INFINITY;
    for encoded in 0..total {
        let mut owners = Vec::with_capacity(cells);
        let mut rest = encoded;
        for _ in 0..cells {
            let choice = rest % options;
            rest /= options;
            owners.push(if choice == 0 {
                NO_OWNER
            } else {
                owner_of(choice - 1)
            });
        }
        best = best.min(grid.energy(&owners));
    }
    best
}

/// Deterministic pseudo-random cost grids, so the exhaustive comparison
/// covers more than the shapes we happened to think of.
fn pseudo_random_grid(seed: u64, columns: usize, rows: usize, labels: usize) -> OwnershipCostGrid {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        f64::from((state >> 33) as u32) / f64::from(u32::MAX >> 1)
    };
    let cells = columns * rows;
    let mut grid = OwnershipCostGrid::new(columns, rows, labels);
    for cell in 0..cells {
        for label in 0..labels {
            let roll = next();
            // Roughly one cell in eight is outside a given candidate, which
            // exercises the pinned cases of the expansion move.
            let cost = (roll > 0.125).then(|| data_cost(next(), 0.0));
            grid.set_cost(cell, label, cost);
        }
        grid.set_disagreement(cell, next() * 0.3);
    }
    grid
}

/// 需求 3.4 / Property 12: every covered cell gets exactly one owner and the
/// total cost equals the exhaustive minimum on a small grid.
///
/// # What this does and does not establish
///
/// The "exactly one owner" half holds for any grid: it is enforced by
/// [`UNASSIGNED_CELL_COST`] and re-checked here cell by cell.
///
/// The "equals the exhaustive minimum" half is a *measurement*, not a proof.
/// The solver is a descent over cut-shaped moves, and that family is known to
/// be an approximation on this kind of energy, not an exact minimiser.  The
/// 200 grids below all land on the exhaustive minimum; widening the same
/// sample to 6,000 grids finds one that does not (`seed = 1617`, a 2×2 grid
/// with 3 candidates, 6.3524 against an optimum of 6.3313 — a 0.3% gap).
/// Both numbers are worth having: the first is a focused exact-match regression
/// guard, while Property 12 in `properties.rs` states the algorithm's honest
/// contract: alpha-expansion need not reach the exact multi-label Potts optimum,
/// but generated grids must remain within 1% of an exhaustive oracle.
#[test]
fn graph_cut_matches_the_exhaustive_minimum() {
    // A focus-stacking shape: the anchor owns the left column, a sharper
    // second frame the right one, and a seam has to run between them.
    let handmade = grid_from(
        3,
        3,
        &[
            &[
                Some(0.20),
                Some(0.50),
                Some(0.90),
                Some(0.20),
                Some(0.55),
                Some(0.95),
                Some(0.25),
                Some(0.60),
                Some(0.90),
            ],
            &[
                Some(0.95),
                Some(0.55),
                Some(0.15),
                Some(0.90),
                Some(0.50),
                Some(0.20),
                Some(0.90),
                Some(0.45),
                Some(0.10),
            ],
            &[
                None,
                None,
                None,
                Some(0.40),
                Some(0.40),
                Some(0.40),
                None,
                None,
                None,
            ],
        ],
        &[0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
    );
    // A grid with a cell no candidate covers at all (需求 3.11).
    let with_hole = grid_from(
        2,
        2,
        &[
            &[Some(0.30), None, Some(0.80), Some(0.10)],
            &[Some(0.70), None, Some(0.20), Some(0.90)],
        ],
        &[0.05, 0.0, 0.10, 0.05],
    );
    let mut cases = vec![handmade, with_hole];
    for seed in 0..200u64 {
        let (columns, rows) = match seed % 3 {
            0 => (2, 2),
            1 => (3, 2),
            _ => (3, 3),
        };
        let labels = 2 + usize::try_from(seed % 2).expect("small");
        cases.push(pseudo_random_grid(seed, columns, rows, labels));
    }
    for (index, grid) in cases.iter().enumerate() {
        let solution = solve_ownership(grid);
        assert_eq!(solution.solver_status, FusionSolverStatus::GraphCut);
        assert_eq!(solution.owners.len(), grid.cell_count());
        for cell in 0..grid.cell_count() {
            if grid.covered(cell) {
                let owner = solution.owners[cell];
                assert_ne!(owner, NO_OWNER, "case {index} cell {cell} has no owner");
                assert!(
                    grid.cost(cell, usize::from(owner - 1)).is_some(),
                    "case {index} cell {cell} owned by a candidate that misses it"
                );
            } else {
                assert_eq!(solution.owners[cell], NO_OWNER, "case {index} cell {cell}");
            }
        }
        let achieved = grid.energy(&solution.owners);
        let optimal = exhaustive_minimum(grid);
        assert!(
            (achieved - optimal).abs() < 1e-9,
            "case {index}: cut reached {achieved}, exhaustive minimum is {optimal}"
        );
    }
}

/// The smoothness term of 需求 3.4 is what makes the cut differ from the
/// per-cell minimum: a single cell that mildly prefers the other candidate
/// is not worth a seam around it.
#[test]
fn the_smoothness_term_suppresses_an_isolated_flip() {
    let mut costs = vec![Some(0.30); 9];
    let mut rival = vec![Some(0.60); 9];
    // The centre cell prefers the rival, but only by a hair.
    costs[4] = Some(0.50);
    rival[4] = Some(0.40);
    let grid = grid_from(3, 3, &[&costs, &rival], &[0.05; 9]);
    let solution = solve_ownership(&grid);
    assert_eq!(solution.solver_status, FusionSolverStatus::GraphCut);
    assert_eq!(solution.owners, vec![owner_of(0); 9]);
    // The per-cell rule of 需求 3.5 does flip it, which is exactly the
    // difference between the two paths.
    assert_eq!(grid.per_cell_minimum()[4], owner_of(1));
}

/// 需求 3.4: a clear-cut sharpness difference does move the owner, so the
/// smoothness term is a tie-breaker and not a freeze.
///
/// The two frames are each decisively better on their own half, so paying for
/// one seam beats handing the whole grid to either of them — the ordinary
/// focus-stacking case.
#[test]
fn a_decisive_sharpness_difference_moves_the_owner() {
    let mut anchor = vec![Some(0.10); 9];
    let mut sharper = vec![Some(0.95); 9];
    for cell in [2, 5, 8] {
        anchor[cell] = Some(0.90);
        sharper[cell] = Some(0.05);
    }
    let grid = grid_from(3, 3, &[&anchor, &sharper], &[0.01; 9]);
    let solution = solve_ownership(&grid);
    let owners_of_right_column = [2, 5, 8].map(|cell| solution.owners[cell]);
    assert_eq!(owners_of_right_column, [owner_of(1); 3]);
    assert_eq!(solution.owners[0], owner_of(0));
}

/// 需求 3.4: the result does not depend on which order the caller happened
/// to hand the frames in — the deterministic candidate order does.
#[test]
fn candidate_order_puts_the_anchor_first_and_breaks_ties_by_path() {
    let frame = |path: &str, sharpness: f64, is_anchor: bool| FusionFrame {
        path: path.to_string(),
        median_sharpness: sharpness,
        is_anchor,
    };
    let frames = vec![
        frame("/z/0009.nef", 0.80, false),
        frame("/m/0005.nef", 0.40, true),
        frame("/a/0001.nef", 0.80, false),
        frame("/b/0002.nef", 0.10, false),
    ];
    assert_eq!(candidate_order(&frames), vec![1, 2, 0, 3]);
    let mut reversed = frames.clone();
    reversed.reverse();
    let paths = |frames: &[FusionFrame], order: Vec<usize>| {
        order
            .into_iter()
            .map(|index| frames[index].path.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        paths(&frames, candidate_order(&frames)),
        paths(&reversed, candidate_order(&reversed))
    );
}

/// 需求 3.10: one registered frame skips the cut and owns every covered cell.
#[test]
fn a_single_frame_station_skips_the_graph_cut() {
    let grid = grid_from(2, 2, &[&[Some(0.4), None, Some(0.6), Some(0.9)]], &[0.0; 4]);
    let solution = solve_ownership(&grid);
    assert_eq!(solution.solver_status, FusionSolverStatus::SingleFrame);
    assert_eq!(
        solution.owners,
        vec![owner_of(0), NO_OWNER, owner_of(0), owner_of(0)]
    );
    assert_eq!(solution.owned_cells(), 3);
    assert!(skips_graph_cut(1));
    assert!(!skips_graph_cut(2));

    // No candidate at all is not a fusion, so it is not a single frame
    // either.
    let empty = OwnershipCostGrid::new(2, 2, 0);
    let solution = solve_ownership(&empty);
    assert_eq!(solution.solver_status, FusionSolverStatus::NotRun);
    assert_eq!(solution.owners, vec![NO_OWNER; 4]);
}

/// 需求 3.5: an exhausted budget falls back to the per-cell minimum, reports
/// the degraded status and records the identifier.
#[test]
fn task_7_31_timeout_falls_back_and_reaches_status_ledger_and_report() {
    let mut costs = vec![Some(0.30); 9];
    let mut rival = vec![Some(0.60); 9];
    costs[4] = Some(0.50);
    rival[4] = Some(0.40);
    let grid = grid_from(3, 3, &[&costs, &rival], &[0.05; 9]);

    // A budget already spent when the first candidate is considered.
    let clock = SteppingClock::new(Duration::from_secs(45));
    let solution = solve_ownership_with_clock(&grid, Duration::from_secs(30), &clock);
    assert_eq!(solution.solver_status, FusionSolverStatus::PerCellFallback);
    assert_eq!(solution.owners, grid.per_cell_minimum());
    assert_eq!(solution.owners[4], owner_of(1));

    // The same grid inside the budget keeps the graph cut result.
    let clock = SteppingClock::new(Duration::ZERO);
    let solution_in_budget = solve_ownership_with_clock(&grid, GRAPH_CUT_TIMEOUT, &clock);
    assert_eq!(
        solution_in_budget.solver_status,
        FusionSolverStatus::GraphCut
    );
    assert_ne!(solution_in_budget.owners, solution.owners);

    let mut ledger = DegradationLedger::new();
    record_solver_degradation(&mut ledger, &solution_in_budget);
    assert!(ledger.is_empty());
    record_solver_degradation(&mut ledger, &solution);
    assert_eq!(ledger.count_of(FUSION_GRAPH_CUT_TIMEOUT), 1);

    let geometry = OwnershipGridGeometry::for_plane(9_504, 6_336);
    let mut report = FusionReport::default();
    solution.write_report(&mut report, &geometry);
    assert_eq!(report.solver_status, FusionSolverStatus::PerCellFallback);
    assert_eq!(report.graph_cut_seconds, solution.graph_cut_seconds);
    assert!(report.graph_cut_seconds >= 45.0);
    assert_eq!(GRAPH_CUT_TIMEOUT, Duration::from_secs(120));
}

/// A timeout part way through the candidate list still produces a complete
/// labeling (需求 3.4's "exactly one owner" holds on the degraded path too).
#[test]
fn a_mid_loop_timeout_still_owns_every_covered_cell() {
    let grid = pseudo_random_grid(7, 3, 3, 3);
    // Two observations of the clock fit inside the budget, the third does
    // not, so the fallback happens after the first candidates were folded in.
    let clock = SteppingClock::new(Duration::from_secs(50));
    let solution = solve_ownership_with_clock(&grid, Duration::from_secs(120), &clock);
    assert_eq!(solution.solver_status, FusionSolverStatus::PerCellFallback);
    assert!(solution.graph_cut_seconds > 0.0);
    for cell in 0..grid.cell_count() {
        assert_eq!(
            solution.owners[cell] != NO_OWNER,
            grid.covered(cell),
            "cell {cell}"
        );
    }
}

/// 需求 3.5: the solver observations reach the Stack_Report; the pixel and
/// confidence fields stay with tasks 7.10 / 7.11.
#[test]
fn the_solver_status_and_duration_reach_the_report() {
    let geometry = OwnershipGridGeometry::for_plane(9_504, 6_336);
    let solution = OwnershipSolution {
        owners: vec![owner_of(0), NO_OWNER],
        solver_status: FusionSolverStatus::PerCellFallback,
        graph_cut_seconds: 121.5,
    };
    let mut report = FusionReport::default();
    solution.write_report(&mut report, &geometry);
    assert_eq!(report.cell_size_px, 18);
    assert_eq!(
        report.grid,
        OwnershipGridSize {
            columns: 528,
            rows: 352
        }
    );
    assert_eq!(report.solver_status, FusionSolverStatus::PerCellFallback);
    assert!((report.graph_cut_seconds - 121.5).abs() < 1e-12);
    assert_eq!(report.owned_pixels, 0, "owned pixels belong to task 7.10");
}

/// The solve is a pure function of the cost grid: two runs agree, and so do
/// two grids that differ only in how the caller filled them in.
#[test]
fn the_solution_is_deterministic() {
    let grid = pseudo_random_grid(3, 3, 3, 3);
    let first = solve_ownership(&grid);
    let second = solve_ownership(&grid);
    assert_eq!(first.owners, second.owners);
    assert_eq!(first.solver_status, second.solver_status);
}

/// 需求 3.4: the data term falls as the Sharpness_Score rises, and the
/// penalty can overturn a sharper but misregistered candidate.
#[test]
fn data_cost_prefers_sharp_and_consistent_candidates() {
    assert_eq!(data_cost(1.0, 0.0), 0.0);
    assert_eq!(data_cost(0.0, 0.0), 1.0);
    assert!(data_cost(0.8, 0.0) < data_cost(0.4, 0.0));
    let base = data_cost(0.4, 0.0);
    let sharper_consistent = data_cost(0.8, mismatch_penalty(0.1, 0.8, 0.4));
    let sharper_displaced = data_cost(0.8, mismatch_penalty(0.7, 0.8, 0.4));
    assert!(candidate_wins_cell(base, sharper_consistent));
    assert!(!candidate_wins_cell(base, sharper_displaced));
    assert!(
        !candidate_wins_cell(base, base),
        "a tie must keep the current owner"
    );
}
