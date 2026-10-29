use super::*;

fn flat(value: f32) -> impl FnMut(f64, f64) -> Option<Rgb<f32>> {
    move |_, _| Some(Rgb([value, value, value]))
}

/// 需求 3.9: the Sharpness_Score is in `[0, 1]`, `0` for no measurable
/// gradient, and the mapping is order preserving.
#[test]
fn normalized_sharpness_is_bounded_and_order_preserving() {
    assert_eq!(normalized_sharpness(0.0), 0.0);
    assert_eq!(normalized_sharpness(-1.0), 0.0);
    assert_eq!(normalized_sharpness(f64::NAN), 0.0);
    assert_eq!(normalized_sharpness(SHARPNESS_HALF_SCALE), 0.5);
    assert!(normalized_sharpness(f64::MAX) <= 1.0);
    let mut previous = 0.0;
    for step in 1..2_000 {
        let raw = f64::from(step) * 0.01;
        let value = normalized_sharpness(raw);
        assert!((0.0..=1.0).contains(&value), "{raw} mapped to {value}");
        assert!(value > previous, "{raw} broke monotonicity");
        previous = value;
    }
}

/// 需求 3.2: the 13×13 sampler uses fractional native coordinates to span
/// the required 32-native-pixel window exactly.
#[test]
fn sharpness_window_spans_exactly_thirty_two_native_pixels() {
    assert_eq!(sharpness_window_px(), SHARPNESS_SAMPLE_WINDOW_PX);
    assert_eq!(
        sharpness_sample_step_px(),
        SHARPNESS_SAMPLE_WINDOW_PX / f64::from(ACUTANCE_PATCH_POINTS)
    );
}

/// 需求 3.1: cell side in `[8, 64]`, and at least 512 cells along the long
/// side of any plane large enough to allow it.
#[test]
fn ownership_cell_size_stays_within_the_required_band() {
    for long_side in [0u32, 1, 512, 4_095, 4_096, 9_504, 20_000, 32_768, 100_000] {
        let cell = ownership_cell_size_px(long_side);
        assert!(
            (OWNERSHIP_CELL_MIN_PX..=OWNERSHIP_CELL_MAX_PX).contains(&cell),
            "{long_side} gave a {cell}px cell"
        );
        if long_side >= OWNERSHIP_CELL_MIN_PX * SELECTION_LONG_SIDE {
            assert!(
                long_side.div_ceil(cell) >= SELECTION_LONG_SIDE,
                "{long_side} gave only {} cells",
                long_side.div_ceil(cell)
            );
        }
    }
    assert_eq!(ownership_cell_size_px(9_504), 18);
}

#[test]
fn grid_geometry_covers_the_whole_plane() {
    let geometry = OwnershipGridGeometry::for_plane(9_504, 6_336);
    assert_eq!(geometry.cell_size_px, 18);
    assert_eq!(geometry.columns, 528);
    assert_eq!(geometry.rows, 352);
    assert_eq!(geometry.cell_count(), 528 * 352);
    assert!(f64::from(geometry.columns * geometry.cell_size_px) >= 9_504.0);
    assert_eq!(geometry.cell_origin(2, 3), (36.0, 54.0));
}

/// 需求 3.2: a candidate and the current composite must agree on positions
/// and window size; the shared plan is what guarantees it.
#[test]
fn sampling_plan_is_shared_between_candidate_and_composite() {
    let geometry = OwnershipGridGeometry::for_plane(9_504, 6_336);
    let plan = geometry.sampling_plan();
    plan.assert_matches(&geometry.sampling_plan());
    assert_eq!(plan.probes().len(), 5);
    assert!(plan.probes().contains(&(0.0, 0.0)));
    let radius = plan.probes()[0].0.abs();
    assert!(
        plan.probes()
            .iter()
            .all(|(dx, dy)| dx.abs() == radius && dy.abs() == radius || (*dx, *dy) == (0.0, 0.0))
    );
}

#[test]
#[should_panic(expected = "same window size")]
fn mismatched_window_sizes_are_rejected() {
    let mut other = CellSamplingPlan::for_cell(19.0);
    other.step_px = 1.0;
    CellSamplingPlan::for_cell(19.0).assert_matches(&other);
}

#[test]
fn a_focused_cell_scores_above_a_flat_cell() {
    let plan = CellSamplingPlan::for_cell(19.0);
    let flat_score = plan.measure(0.0, 0.0, flat(0.5));
    let edge_score = plan.measure(0.0, 0.0, |x, _| {
        let value = if x.rem_euclid(16.0) < 8.0 { 0.15 } else { 0.85 };
        Some(Rgb([value, value, value]))
    });
    assert_eq!(flat_score, 0.0);
    assert!(edge_score > flat_score);
    assert!((0.0..=1.0).contains(&edge_score));
}

#[test]
fn an_uncovered_cell_scores_zero() {
    let plan = CellSamplingPlan::for_cell(19.0);
    assert_eq!(plan.measure(0.0, 0.0, |_, _| None), 0.0);
}

/// 需求 3.3: identical pixels disagree by 0, a fully displaced contour
/// disagrees strongly, and the value is normalised.
#[test]
fn cell_disagreement_is_a_normalised_median() {
    let identical = cell_disagreement(flat(0.4), flat(0.4));
    assert_eq!(identical, 0.0);
    let offset = cell_disagreement(flat(0.9), flat(0.3));
    assert!((offset - 0.6).abs() < 1e-6, "got {offset}");
    assert!((0.0..=1.0).contains(&offset));
    assert_eq!(cell_disagreement(flat(0.4), |_, _| None), 1.0);
    assert_eq!(cell_disagreement(|_, _| None, flat(0.4)), 1.0);
}

/// The median ignores a minority of differing probes, where the legacy high
/// percentile would have kept them. This is the behaviour change 需求 3.3 asks
/// for: a single stroke crossing one corner of the cell no longer vetoes an
/// otherwise consistent candidate.
#[test]
fn median_disagreement_ignores_a_minority_of_differing_probes() {
    let disagreement = cell_disagreement(
        |x_offset, y_offset| {
            let differs = x_offset < 0.2 && y_offset < 0.2;
            let value = if differs { 0.9 } else { 0.4 };
            Some(Rgb([value, value, value]))
        },
        flat(0.4),
    );
    assert!(disagreement < 0.2, "got {disagreement}");
}

#[test]
fn narrow_contour_prices_a_seam_without_vetoing_the_source() {
    let (median, seam) = cell_disagreement_with_seam_evidence(
        |x, y| Some(Rgb([if x < 0.2 && y < 0.2 { 0.9 } else { 0.4 }; 3])),
        flat(0.4),
    );
    assert_eq!(median, 0.0);
    assert!(seam > 0.1);
    assert_eq!(mismatch_penalty(median, 0.8, 0.4), 0.0);
}

/// 需求 3.3: the penalty is at least 1.0 and only applies to a sharper but
/// inconsistent candidate.
#[test]
fn mismatch_penalty_only_punishes_sharper_inconsistent_candidates() {
    const { assert!(OWNERSHIP_MISMATCH_PENALTY >= 1.0) };
    assert_eq!(mismatch_penalty(0.5, 0.8, 0.4), OWNERSHIP_MISMATCH_PENALTY);
    assert_eq!(mismatch_penalty(0.5, 0.3, 0.4), 0.0);
    assert_eq!(mismatch_penalty(0.2, 0.8, 0.4), 0.0);
    assert_eq!(
        mismatch_penalty(OWNERSHIP_DISAGREEMENT_VETO, 0.8, 0.4),
        0.0,
        "the veto is exclusive"
    );
    assert_eq!(mismatch_penalty(0.5, 0.4, 0.4), 0.0, "a tie is not sharper");
}

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

// -----------------------------------------------------------------------
// Hard ownership write, Sharpness_Confidence and low sharpness regions
// (tasks 7.10 / 7.11, 需求 3.6, 3.8, 3.9)
// -----------------------------------------------------------------------

/// 需求 3.9 / Property 15: the confidence is in `[0, 1]`, grows with the
/// winner/runner-up gap, and is exactly `0` for a single candidate.
#[test]
fn cell_confidence_is_bounded_monotone_and_zero_for_one_candidate() {
    assert_eq!(cell_confidence(0.9, 0.0, 1), 0.0);
    assert_eq!(cell_confidence(0.0, 0.0, 0), 0.0);
    // No measurable gradient anywhere in the cell: the quotient stays
    // defined and the cell carries no confidence.
    assert_eq!(cell_confidence(0.0, 0.0, 3), 0.0);
    assert_eq!(cell_confidence(0.8, 0.8, 2), 0.0);
    assert_eq!(cell_confidence(0.8, 0.0, 2), 1.0);
    let mut previous = 0.0;
    for step in 0..=20 {
        let second = 0.8 - f64::from(step) * 0.04;
        let value = cell_confidence(0.8, second, 2);
        assert!((0.0..=1.0).contains(&value), "{second} mapped to {value}");
        assert!(value >= previous, "{second} broke monotonicity");
        previous = value;
    }
    // A runner-up above the winner cannot happen, but the clamp must hold.
    assert_eq!(cell_confidence(0.4, 0.9, 2), 0.0);
}

/// 需求 3.8: the threshold is the 10th percentile of the winning scores and
/// is always a value the station produced, so a uniform station has no cell
/// strictly below it.
#[test]
fn low_sharpness_threshold_is_the_tenth_percentile() {
    assert_eq!(low_sharpness_threshold(&[]), 0.0);
    assert_eq!(low_sharpness_threshold(&[0.42]), 0.42);
    let uniform = vec![0.3; 50];
    let threshold = low_sharpness_threshold(&uniform);
    assert!(uniform.iter().all(|&value| value >= threshold));
    let ramp = (0..=100).map(|i| f64::from(i) * 0.01).collect::<Vec<_>>();
    assert!((low_sharpness_threshold(&ramp) - 0.10).abs() < 1e-9);
    // Order must not matter.
    let mut shuffled = ramp.clone();
    shuffled.reverse();
    assert_eq!(
        low_sharpness_threshold(&shuffled),
        low_sharpness_threshold(&ramp)
    );
}

/// 需求 3.8: low sharpness cells are merged into 4-connected regions; a
/// diagonal touch is two regions, and the area is the cell count.
#[test]
fn low_sharpness_regions_merge_only_four_connected_cells() {
    let low = [
        true, true, false, //
        false, false, false, //
        false, false, true,
    ];
    let regions = low_sharpness_regions(3, 3, &low);
    assert_eq!(regions.len(), 2);
    assert_eq!(
        regions[0],
        LowSharpnessRegion {
            first_column: 0,
            first_row: 0,
            last_column: 1,
            last_row: 0,
            cells: 2,
        }
    );
    assert_eq!(regions[1].cells, 1);
    assert!(low_sharpness_regions(3, 3, &[false; 9]).is_empty());
    // A mask of the wrong size is a programming error upstream, never a
    // region.
    assert!(low_sharpness_regions(3, 3, &[true; 4]).is_empty());
}

/// 需求 3.8: the recorded rectangle and area are in world coordinates and
/// world native pixels.
#[test]
fn low_sharpness_region_records_use_world_coordinates() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 16,
        columns: 4,
        rows: 4,
    };
    let regions = [LowSharpnessRegion {
        first_column: 1,
        first_row: 2,
        last_column: 2,
        last_row: 2,
        cells: 2,
    }];
    let records = low_sharpness_region_records(&geometry, &regions, (-100.0, 40.0));
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].world.left, -100.0 + 16.0);
    assert_eq!(records[0].world.top, 40.0 + 32.0);
    assert_eq!(records[0].world.width, 32.0);
    assert_eq!(records[0].world.height, 16.0);
    assert_eq!(records[0].area_px, 2.0 * 256.0);
}

/// 需求 11.12 measures the same ratio on the final output: uncovered pixels
/// are outside the statistic.
#[test]
fn confidence_summary_only_counts_covered_pixels() {
    let summary = confidence_summary(&[0.0, 0.5, 0.9, 0.01], &[0, 255, 255, 255]);
    assert!((summary.mean - (0.5 + 0.9 + 0.01) / 3.0).abs() < 1e-6);
    assert!((summary.below_0_05_ratio - 1.0 / 3.0).abs() < 1e-9);
    assert_eq!(
        confidence_summary(&[0.4, 0.4], &[0, 0]),
        SharpnessConfidenceSummary::default()
    );
}

/// One cell column per candidate so the cells are independent, with the
/// candidate sharper on the right half.
fn two_frame_station() -> (StationFusion, Vec<f64>, Vec<bool>, CandidateCells) {
    // 6x1 cells: `for_plane` clamps the cell side to 8px, so a 48x8 plane
    // gives exactly six cells in one row.
    let mut fusion = StationFusion::new(48, 8, 2);
    let cells = fusion.geometry().cell_count();
    assert_eq!(cells, 6);
    let base_sharpness = vec![0.5; cells];
    let base_covered = vec![true; cells];
    fusion.seed(0, &base_sharpness, &base_covered);
    let mut candidate = CandidateCells::new(cells);
    for cell in 0..cells {
        candidate.covered[cell] = true;
        candidate.sharpness[cell] = if cell >= 3 { 0.95 } else { 0.05 };
    }
    (fusion, base_sharpness, base_covered, candidate)
}

#[test]
fn station_smoothness_cannot_erase_consistent_sparse_focused_detail() {
    let mut fusion = StationFusion::new(64, 64, 2);
    let count = fusion.geometry().cell_count();
    fusion.seed(0, &vec![0.20; count], &vec![true; count]);
    let mut candidate = CandidateCells::new(count);
    for cell in 0..count {
        candidate.covered[cell] = true;
        candidate.sharpness[cell] = if cell % 7 == 0 { 0.24 } else { 0.18 };
    }
    fusion.fold(1, &candidate);
    for (cell, evidence) in fusion.sharpness_evidence().cells.iter().enumerate() {
        assert!(
            evidence.owner_shortfall() <= 0.05 + f32::EPSILON,
            "smoothness discarded focused detail at cell {cell}: {evidence:?}"
        );
    }
}

/// 需求 3.6 / 3.11: every covered cell ends with exactly one owner, taken
/// from the real candidate set — there is no blended or intermediate label.
#[test]
fn station_fusion_gives_every_covered_cell_one_real_owner() {
    let (mut fusion, _, _, candidate) = two_frame_station();
    let took = fusion.fold(1, &candidate);
    assert_eq!(took.len(), fusion.geometry().cell_count());
    // The sharper half switched, the softer half kept the anchor: a hard
    // split at a cell boundary, not a ramp across it.
    assert_eq!(took, vec![false, false, false, true, true, true]);
    assert_eq!(
        fusion.owners(),
        &[
            owner_of(0),
            owner_of(0),
            owner_of(0),
            owner_of(1),
            owner_of(1),
            owner_of(1)
        ]
    );
    assert_eq!(fusion.solver_status(), FusionSolverStatus::GraphCut);
}

/// 需求 3.7 / 3.11: a cell no frame covers keeps the reserved no-owner
/// identifier, and a cell only the candidate covers goes to the candidate.
#[test]
fn station_fusion_leaves_uncovered_cells_unowned() {
    let mut fusion = StationFusion::new(48, 8, 2);
    let cells = fusion.geometry().cell_count();
    let mut base_covered = vec![true; cells];
    base_covered[0] = false;
    base_covered[5] = false;
    fusion.seed(0, &vec![0.5; cells], &base_covered);
    let mut candidate = CandidateCells::new(cells);
    candidate.covered[0] = true;
    candidate.sharpness[0] = 0.2;
    fusion.fold(1, &candidate);
    assert_eq!(
        fusion.owners()[0],
        owner_of(1),
        "only the candidate covers cell 0"
    );
    assert_eq!(fusion.owners()[5], NO_OWNER, "no frame covers cell 5");
    assert!(
        fusion.owners()[1..5]
            .iter()
            .all(|&owner| owner == owner_of(0))
    );
}

/// 需求 3.3: an inconsistent candidate loses the cell even though it is
/// sharper, because the penalty is charged on top of its data term.
#[test]
fn station_fusion_penalises_a_sharper_but_displaced_candidate() {
    let (mut fusion, _, _, mut candidate) = two_frame_station();
    for cell in 0..candidate.covered.len() {
        candidate.disagreement[cell] = OWNERSHIP_DISAGREEMENT_VETO + 0.1;
    }
    let took = fusion.fold(1, &candidate);
    assert!(
        took.iter().all(|&took| !took),
        "a displaced candidate must not take any cell"
    );
}

/// 需求 3.9: the confidence of a cell follows its own winner/runner-up gap,
/// independently of which frame won.
#[test]
fn station_fusion_confidence_follows_the_winner_runner_up_gap() {
    let (mut fusion, _, _, candidate) = two_frame_station();
    fusion.fold(1, &candidate);
    let confidence = fusion.confidence();
    // Soft half: winner 0.5, runner-up 0.05 -> (0.5 - 0.05) / 0.5.
    assert!((f64::from(confidence[0]) - 0.9).abs() < 1e-6);
    // Sharp half: winner 0.95, runner-up 0.5 -> (0.95 - 0.5) / 0.95.
    assert!((f64::from(confidence[5]) - 0.45 / 0.95).abs() < 1e-6);
    assert!(confidence.iter().all(|&value| (0.0..=1.0).contains(&value)));
}

/// The report evidence is the same winner/runner pair that feeds
/// Sharpness_Confidence, and maps back to the ownership cell containing a
/// pixel without allocating a full-resolution diagnostic plane.
#[test]
fn sharpness_evidence_preserves_winner_runner_and_owner_scores() {
    let (mut fusion, _, _, candidate) = two_frame_station();
    fusion.fold(1, &candidate);
    let evidence = fusion.sharpness_evidence();
    assert_eq!(evidence.dimensions(), (6, 1, 8));
    let soft = evidence.at_pixel(4, 0).expect("cell evidence");
    assert_eq!(soft.winner_score, 0.5);
    assert_eq!(soft.runner_up_score, 0.05);
    assert_eq!(soft.owner_score, 0.5);
    assert_eq!(soft.candidate_count, 2);
    let sharp = evidence.at_pixel(40, 7).expect("cell evidence");
    assert_eq!(sharp.winner_score, 0.95);
    assert_eq!(sharp.runner_up_score, 0.5);
    assert_eq!(sharp.owner_score, 0.95);
}

/// 需求 3.10: a station of one frame carries no runner-up evidence at all,
/// so every cell's confidence is 0 and the solver never ran.
#[test]
fn a_single_frame_station_has_zero_confidence_everywhere() {
    let mut fusion = StationFusion::new(48, 8, 1);
    let cells = fusion.geometry().cell_count();
    fusion.seed(0, &vec![0.7; cells], &vec![true; cells]);
    assert_eq!(fusion.solver_status(), FusionSolverStatus::SingleFrame);
    assert!(fusion.confidence().iter().all(|&value| value == 0.0));
    assert!(fusion.owners().iter().all(|&owner| owner == owner_of(0)));
    assert_eq!(
        StationFusion::new(48, 8, 0).solver_status(),
        FusionSolverStatus::NotRun
    );
}

/// 需求 3.8: the cells whose winner is below the station P10 are exactly the
/// low sharpness cells, and an uncovered cell is never one of them.
#[test]
fn station_fusion_marks_the_below_percentile_cells() {
    let mut fusion = StationFusion::new(8 * 20, 8, 1);
    let cells = fusion.geometry().cell_count();
    assert_eq!(cells, 20);
    let mut sharpness = vec![0.8; cells];
    sharpness[7] = 0.01;
    let mut covered = vec![true; cells];
    covered[19] = false;
    fusion.seed(0, &sharpness, &covered);
    let low = fusion.low_sharpness_cells();
    assert!(low[7], "the single soft cell is below the P10 of the rest");
    assert_eq!(low.iter().filter(|&&low| low).count(), 1);
    assert!(!low[19], "an uncovered cell has no candidate to compare");
    let records = fusion.low_sharpness_region_records((0.0, 0.0));
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].area_px, 64.0);
}

#[test]
fn content_regions_come_only_from_current_masks_and_flat_components() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 100,
        columns: 16,
        rows: 8,
    };
    let count = geometry.cell_count();
    let sharpness = vec![0.02; count];
    let costs = vec![0.1; count];
    let full = ContentEvidence {
        fully_covered: vec![true; count],
        flat: vec![true; count],
        mean_luma: vec![0.5; count],
    };
    let candidates = content_region_candidates(geometry, owner_of(0), &sharpness, &costs, &full);
    assert!(!candidates.is_empty());
    assert!(candidates.iter().all(|candidate| {
        candidate
            .cells
            .iter()
            .all(|&cell| full.fully_covered[cell] && full.flat[cell])
    }));

    let mut masked = full.clone();
    masked.fully_covered[0] = false;
    let masked_candidates =
        content_region_candidates(geometry, owner_of(0), &sharpness, &costs, &masked);
    assert!(
        masked_candidates
            .iter()
            .all(|candidate| !candidate.cells.contains(&0))
    );
    let mut textured = full;
    textured.flat[0] = false;
    let textured_candidates =
        content_region_candidates(geometry, owner_of(0), &sharpness, &costs, &textured);
    assert!(
        textured_candidates
            .iter()
            .all(|candidate| candidate.origin != (0, 0))
    );
}

#[test]
fn content_region_selection_is_permutation_translation_and_thread_deterministic() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 100,
        columns: 24,
        rows: 8,
    };
    let count = geometry.cell_count();
    let evidence = ContentEvidence {
        fully_covered: vec![true; count],
        flat: vec![true; count],
        mean_luma: vec![0.4; count],
    };
    let candidates = content_region_candidates(
        geometry,
        owner_of(0),
        &vec![0.01; count],
        &vec![0.2; count],
        &evidence,
    );
    let expected = select_content_regions(candidates.clone(), 2);
    let mut reversed = candidates.clone();
    reversed.reverse();
    assert_eq!(expected, select_content_regions(reversed, 2));
    let threaded = std::thread::spawn(move || select_content_regions(candidates, 2))
        .join()
        .expect("content selector thread");
    assert_eq!(expected, threaded);

    let shifted_geometry = OwnershipGridGeometry {
        columns: 32,
        ..geometry
    };
    let shifted_count = shifted_geometry.cell_count();
    let mut shifted = ContentEvidence {
        fully_covered: vec![false; shifted_count],
        flat: vec![false; shifted_count],
        mean_luma: vec![f64::NAN; shifted_count],
    };
    for row in 0..8usize {
        for column in 8..32usize {
            let cell = row * 32 + column;
            shifted.fully_covered[cell] = true;
            shifted.flat[cell] = true;
            shifted.mean_luma[cell] = 0.4;
        }
    }
    let translated = select_content_regions(
        content_region_candidates(
            shifted_geometry,
            owner_of(0),
            &vec![0.01; shifted_count],
            &vec![0.2; shifted_count],
            &shifted,
        ),
        2,
    );
    assert_eq!(translated.len(), expected.len());
    assert!(
        translated
            .iter()
            .zip(&expected)
            .all(|(translated, original)| translated.origin.0 == original.origin.0 + 8)
    );
}

#[test]
fn disconnected_and_overlapping_content_regions_cannot_count_twice() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 100,
        columns: 8,
        rows: 8,
    };
    let count = geometry.cell_count();
    let mut evidence = ContentEvidence {
        fully_covered: vec![true; count],
        flat: vec![true; count],
        mean_luma: vec![0.5; count],
    };
    for row in 0..8usize {
        evidence.flat[row * 8 + 4] = false;
    }
    assert!(
        content_region_candidates(
            geometry,
            owner_of(0),
            &vec![0.01; count],
            &vec![0.1; count],
            &evidence,
        )
        .is_empty()
    );

    let duplicate = ContentRegionCandidate {
        owner: owner_of(0),
        cells: (0..64).collect(),
        origin: (0, 0),
        data_cost: 0.1,
    };
    let mut overlap = duplicate.clone();
    overlap.owner = owner_of(1);
    overlap.origin = (1, 0);
    assert_eq!(select_content_regions(vec![duplicate, overlap], 2).len(), 1);
}

#[test]
fn decisive_sharpness_requires_a_coherent_non_vetoed_component() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 8,
        columns: 4,
        rows: 2,
    };
    let mut candidate = CandidateCells::new(8);
    candidate.covered.fill(true);
    candidate.sharpness.fill(0.5);
    for cell in [0usize, 1, 4, 5] {
        candidate.sharpness[cell] = 0.9;
    }
    candidate.sharpness[7] = 0.9;
    let decisive = coherent_decisive_cells(geometry, &vec![0.5; 8], &candidate);
    assert_eq!(
        decisive,
        vec![true, true, false, false, true, true, false, false]
    );

    candidate.disagreement[0] = OWNERSHIP_DISAGREEMENT_VETO + 0.01;
    let vetoed = coherent_decisive_cells(geometry, &vec![0.5; 8], &candidate);
    assert!(vetoed.iter().all(|&value| !value));

    candidate.sharpness.fill(0.1);
    let base = coherent_base_cells(geometry, &vec![0.5; 8], &candidate);
    assert!(base.iter().all(|&value| value));
    candidate.sharpness[7] = 0.49;
    let still_coherent = coherent_base_cells(geometry, &vec![0.5; 8], &candidate);
    assert!(!still_coherent[7]);
    assert!(still_coherent[..7].iter().all(|&value| value));
}

#[test]
fn region_decision_keeps_a_quality_roi_sized_owner_block_together() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 16,
        columns: 50,
        rows: 50,
    };
    let count = geometry.cell_count();
    let mut candidate = CandidateCells::new(count);
    candidate.covered.fill(true);
    candidate.sharpness.fill(0.9);
    let decisive = region_decisive_cells(geometry, &vec![0.5; count], &candidate);
    assert!(decisive.iter().all(|&value| value));

    candidate.disagreement[0] = OWNERSHIP_DISAGREEMENT_VETO + 0.01;
    let vetoed = region_decisive_cells(geometry, &vec![0.5; count], &candidate);
    assert!(vetoed.iter().all(|&value| !value));
}

#[test]
fn quality_region_accepts_a_near_best_full_roi_without_relaxing_shortfall() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 16,
        columns: 40,
        rows: 40,
    };
    let count = geometry.cell_count();
    let mut candidate = CandidateCells::new(count);
    candidate.covered.fill(true);
    candidate.sharpness.fill(0.951);
    let owner = vec![0.94; count];
    let best = vec![1.0; count];
    let decisive = quality_region_decisive_cells(geometry, &owner, &best, &candidate);
    assert!(decisive.iter().all(|&value| value));

    // 0.94 is a six-percent shortfall from the observed winner, so the
    // same region is rejected at the unchanged five-percent gate boundary.
    candidate.sharpness.fill(0.94);
    let rejected = quality_region_decisive_cells(geometry, &owner, &best, &candidate);
    assert!(rejected.iter().all(|&value| !value));

    // A near-winner is still not allowed to lower an already sharper
    // owner just because it makes a larger connected ROI.  This is the
    // regression that let the smoothness override create gate shortfall.
    candidate.sharpness.fill(0.93);
    let softer = quality_region_decisive_cells(geometry, &vec![0.94; count], &best, &candidate);
    assert!(softer.iter().all(|&value| !value));
}

#[test]
fn quality_gate_decisive_cells_only_moves_a_consistent_five_percent_winner() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 16,
        columns: 2,
        rows: 2,
    };
    let count = geometry.cell_count();
    let mut candidate = CandidateCells::new(count);
    candidate.covered.fill(true);
    candidate.sharpness.fill(0.96);
    let owner = vec![0.90; count];
    let best = vec![0.96; count];
    let decisive = quality_gate_decisive_cells(geometry, &owner, &best, &candidate);
    assert!(decisive.iter().all(|&value| value));

    // A candidate that is sharper than the current owner but six percent
    // below the observed winner cannot manufacture a passing owner.
    candidate.sharpness.fill(0.90);
    let rejected = quality_gate_decisive_cells(geometry, &owner, &best, &candidate);
    assert!(rejected.iter().all(|&value| !value));

    candidate.sharpness.fill(0.96);
    candidate
        .disagreement
        .fill(OWNERSHIP_DISAGREEMENT_VETO + 0.01);
    let vetoed = quality_gate_decisive_cells(geometry, &owner, &best, &candidate);
    assert!(vetoed.iter().all(|&value| !value));
}

#[test]
fn content_constraint_keeps_real_provenance_without_forcing_an_oracle() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 100,
        columns: 24,
        rows: 8,
    };
    let count = geometry.cell_count();
    let evidence = ContentEvidence {
        fully_covered: vec![true; count],
        flat: vec![true; count],
        mean_luma: vec![0.5; count],
    };
    let mut candidates = content_region_candidates(
        geometry,
        owner_of(0),
        &vec![0.2; count],
        &vec![0.4; count],
        &evidence,
    );
    candidates.extend(content_region_candidates(
        geometry,
        owner_of(1),
        &vec![0.3; count],
        &vec![0.1; count],
        &evidence,
    ));
    let selected = select_content_regions(candidates.clone(), 2);
    assert_eq!(selected.len(), 2);
    assert!(selected.iter().all(|region| region.owner == owner_of(1)));
    assert!(selected.iter().all(|region| candidates.contains(region)));
}

#[test]
fn content_constraint_handles_feasible_and_impossible_cardinality() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 100,
        columns: 80,
        rows: 8,
    };
    let count = geometry.cell_count();
    let sharpness = vec![0.01; count];
    let costs = vec![0.1; count];
    let feasible = content_region_candidates(
        geometry,
        owner_of(0),
        &sharpness,
        &costs,
        &ContentEvidence {
            fully_covered: vec![true; count],
            flat: vec![true; count],
            mean_luma: vec![0.5; count],
        },
    );
    assert_eq!(
        select_content_regions(feasible, CONTENT_OWNER_REGION_TARGET).len(),
        CONTENT_OWNER_REGION_TARGET
    );

    let impossible = content_region_candidates(
        geometry,
        owner_of(0),
        &sharpness,
        &costs,
        &ContentEvidence {
            fully_covered: vec![true; count],
            flat: vec![false; count],
            mean_luma: vec![0.5; count],
        },
    );
    assert!(impossible.is_empty());
}

/// 需求 3.5: an exhausted budget resolves the round per cell and marks the
/// station degraded, without changing what a covered cell is.
#[test]
fn an_exhausted_station_budget_falls_back_to_the_per_cell_minimum() {
    let (mut fusion, _, _, candidate) = two_frame_station();
    fusion.timeout = Duration::ZERO;
    let took = fusion.fold(1, &candidate);
    assert_eq!(took, vec![false, false, false, true, true, true]);
    assert_eq!(fusion.solver_status(), FusionSolverStatus::PerCellFallback);
    let mut ledger = DegradationLedger::new();
    record_solver_degradation(
        &mut ledger,
        &OwnershipSolution {
            owners: fusion.owners().to_vec(),
            solver_status: fusion.solver_status(),
            graph_cut_seconds: fusion.graph_cut_seconds(),
        },
    );
    assert_eq!(ledger.entries().len(), 1);
}

/// The run scoped sink keeps one record per station, replaces a re-rendered
/// station instead of duplicating it, and orders the document by station
/// index rather than by render order.
///
/// The sink is a process global, so this is the only test that touches it.
#[test]
fn the_run_sink_keeps_one_record_per_station() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let record = |station_index: usize, cell_size_px: u32| FusionReport {
        station_index,
        cell_size_px,
        ..FusionReport::default()
    };
    reset_run_records();
    for report in [record(2, 8), record(0, 9), record(2, 10)] {
        record_run_station(report);
    }
    assert_eq!(
        run_records_snapshot()
            .iter()
            .map(|station| (station.station_index, station.cell_size_px))
            .collect::<Vec<_>>(),
        vec![(0, 9), (2, 10)]
    );
    reset_run_records();
    assert!(run_records_snapshot().is_empty());
}
