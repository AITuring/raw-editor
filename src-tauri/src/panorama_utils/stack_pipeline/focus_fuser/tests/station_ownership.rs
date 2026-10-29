//! `StationFusion`, the per-station fold the focus renderer drives.
//!
//! The pure confidence and low sharpness reductions, the hard ownership write,
//! Sharpness_Confidence and low sharpness cells of a station, the budget fallback
//! and the run scoped record sink.

use super::*;

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
