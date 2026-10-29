//! Property 12: the multi-label ownership graph cut.
//!
//! Complete, near-optimal against an exhaustive oracle on bounded cost grids and
//! deterministic; the saved counterexample stays within the property bound.

use super::super::focus_fuser;
use crate::panorama_utils::stitching;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Property 12 harness: bounded multi-label ownership cost grids
// ---------------------------------------------------------------------------

/// Generated input to the multi-label ownership solver.
///
/// Costs are positive hundredths, so every non-empty grid has a positive oracle
/// optimum and the multiplicative 1.01 bound is meaningful without an additive
/// escape hatch. `None` is a candidate that does not cover that cell.
#[derive(Clone, Debug)]
pub(super) struct CostGridDraw {
    pub(super) columns: usize,
    pub(super) rows: usize,
    pub(super) labels: usize,
    pub(super) costs: Vec<Option<u16>>,
    pub(super) disagreement: Vec<u8>,
}

impl CostGridDraw {
    pub(super) fn cell_count(&self) -> usize {
        self.columns * self.rows
    }

    pub(super) fn cost(&self, cell: usize, label: usize) -> Option<f64> {
        self.costs[cell * self.labels + label].map(|units| f64::from(units) / 100.0)
    }

    pub(super) fn disagreement(&self, cell: usize) -> f64 {
        f64::from(self.disagreement[cell]) / 100.0
    }

    pub(super) fn is_covered(&self, cell: usize) -> bool {
        (0..self.labels).any(|label| self.cost(cell, label).is_some())
    }

    pub(super) fn production_grid(&self) -> focus_fuser::OwnershipCostGrid {
        let mut grid = focus_fuser::OwnershipCostGrid::new(self.columns, self.rows, self.labels);
        for cell in 0..self.cell_count() {
            for label in 0..self.labels {
                grid.set_cost(cell, label, self.cost(cell, label));
            }
            grid.set_disagreement(cell, self.disagreement(cell));
        }
        grid
    }

    /// Evaluate the Potts energy independently of `OwnershipCostGrid::energy`.
    pub(super) fn energy(&self, owners: &[u16]) -> f64 {
        if owners.len() != self.cell_count() {
            return f64::INFINITY;
        }
        let mut total = 0.0;
        for (cell, &owner) in owners.iter().enumerate() {
            if owner == stitching::NO_OWNER {
                if self.is_covered(cell) {
                    return f64::INFINITY;
                }
                continue;
            }
            let label = usize::from(owner - 1);
            let Some(cost) = (label < self.labels)
                .then(|| self.cost(cell, label))
                .flatten()
            else {
                return f64::INFINITY;
            };
            total += cost;
        }
        for cell in 0..self.cell_count() {
            let (x, y) = (cell % self.columns, cell / self.columns);
            for neighbour in [
                (x + 1 < self.columns).then(|| cell + 1),
                (y + 1 < self.rows).then(|| cell + self.columns),
            ]
            .into_iter()
            .flatten()
            {
                if owners[cell] != stitching::NO_OWNER
                    && owners[neighbour] != stitching::NO_OWNER
                    && owners[cell] != owners[neighbour]
                {
                    total += crate::panorama_utils::seam_cut::pairwise_weight(
                        self.disagreement(cell),
                        self.disagreement(neighbour),
                    );
                }
            }
        }
        total
    }
}

/// At most `3^9` legal assignments: uncovered cells have only `NO_OWNER` as an
/// option, while covered cells enumerate only candidates that actually cover
/// them. This avoids the old unit oracle's deliberately-invalid fourth option
/// and keeps 100 exhaustive cases cheap enough for the normal library suite.
pub(super) const GRAPH_CUT_MAX_ORACLE_LABELINGS: usize = 19_683;

pub(super) const GRAPH_CUT_PROPERTY_CASES: usize = 100;

pub(super) fn arb_cost_grid() -> impl Strategy<Value = CostGridDraw> {
    (1usize..=3, 1usize..=3, 2usize..=3).prop_flat_map(|(columns, rows, labels)| {
        let cells = columns * rows;
        (
            Just((columns, rows, labels)),
            prop::collection::vec(
                prop_oneof![
                    1 => Just(None),
                    7 => (1u16..=100).prop_map(Some),
                ],
                cells * labels,
            ),
            prop::collection::vec(0u8..=30, cells),
        )
            .prop_map(|((columns, rows, labels), mut costs, disagreement)| {
                // Keep the multiplicative oracle bound non-vacuous: at least one
                // cell is covered and every finite data cost is strictly positive.
                costs[0] = Some(costs[0].unwrap_or(1));
                CostGridDraw {
                    columns,
                    rows,
                    labels,
                    costs,
                    disagreement,
                }
            })
    })
}

/// Exhaustively enumerate every legal owner assignment and return its minimum
/// energy together with the number of assignments examined.
pub(super) fn exhaustive_graph_cut_oracle(draw: &CostGridDraw) -> (f64, usize) {
    let options = (0..draw.cell_count())
        .map(|cell| {
            let labels = (0..draw.labels)
                .filter(|&label| draw.cost(cell, label).is_some())
                .map(focus_fuser::owner_of)
                .collect::<Vec<_>>();
            if labels.is_empty() {
                vec![stitching::NO_OWNER]
            } else {
                labels
            }
        })
        .collect::<Vec<_>>();
    let assignments = options.iter().map(Vec::len).product::<usize>();
    assert!(assignments <= GRAPH_CUT_MAX_ORACLE_LABELINGS);

    let mut best = f64::INFINITY;
    for encoded in 0..assignments {
        let mut rest = encoded;
        let owners = options
            .iter()
            .map(|cell_options| {
                let owner = cell_options[rest % cell_options.len()];
                rest /= cell_options.len();
                owner
            })
            .collect::<Vec<_>>();
        best = best.min(draw.energy(&owners));
    }
    (best, assignments)
}

/// A fixed focus-like witness where per-cell minima introduce an expensive
/// isolated seam. It makes a temporary "return per-cell minima" production
/// mutation fail deterministically instead of hoping a random case exposes it.
pub(super) fn graph_cut_non_vacuity_witness() -> CostGridDraw {
    let mut costs = Vec::with_capacity(18);
    for cell in 0..9 {
        costs.push(Some(30));
        costs.push(Some(if cell == 4 { 1 } else { 60 }));
    }
    CostGridDraw {
        columns: 3,
        rows: 3,
        labels: 2,
        costs,
        disagreement: vec![0; 9],
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: GRAPH_CUT_PROPERTY_CASES as u32,
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 12: 对于任意
    // ownership 代价网格与任意候选集合，每个已覆盖单元恰好得到一个 owner；在不超过
    // 3x3、候选数不超过 3 的网格上，生产总代价不超过穷举最优值的 1.01 倍；相同输入重复
    // 求解得到逐位相同的标签，代价完全相同时按确定的候选顺序解 tie。
    //
    // Exact multi-label Potts optimum is intentionally *not* required here:
    // alpha-expansion is an approximation algorithm and does not guarantee the
    // exact global optimum. The exhaustive oracle instead enforces the approved
    // 1% production bound on generated grids while retaining all owner-validity
    // and determinism conditions.
    //
    // **Validates: Requirements 3.4**
    #[test]
    fn multi_label_graph_cut_is_complete_near_optimal_and_deterministic(
        draw in arb_cost_grid(),
    ) {
        let grid = draw.production_grid();
        let first = focus_fuser::solve_ownership(&grid);
        let repeated = focus_fuser::solve_ownership(&grid);

        prop_assert_eq!(first.solver_status, crate::panorama_utils::stack_pipeline::report::FusionSolverStatus::GraphCut);
        prop_assert_eq!(
            &first.owners,
            &repeated.owners,
            "identical cost-grid input must produce bit-identical u16 labels"
        );
        prop_assert_eq!(first.owners.len(), draw.cell_count());
        for cell in 0..draw.cell_count() {
            let owner = first.owners[cell];
            if draw.is_covered(cell) {
                prop_assert_ne!(
                    owner,
                    stitching::NO_OWNER,
                    "covered cell {} has no owner",
                    cell
                );
                let label = usize::from(owner - 1);
                prop_assert!(
                    label < draw.labels && draw.cost(cell, label).is_some(),
                    "covered cell {cell} has invalid owner {owner}"
                );
            } else {
                prop_assert_eq!(
                    owner,
                    stitching::NO_OWNER,
                    "uncovered cell {} must keep the reserved owner",
                    cell
                );
            }
        }

        let achieved = draw.energy(&first.owners);
        let (optimal, assignments) = exhaustive_graph_cut_oracle(&draw);
        prop_assert!(optimal > 0.0 && optimal.is_finite());
        prop_assert!(assignments <= GRAPH_CUT_MAX_ORACLE_LABELINGS);
        prop_assert!(
            achieved <= optimal * 1.01,
            "production cost {achieved:.12} exceeds 1.01 * exhaustive optimum {optimal:.12} \
             on a {}x{} grid with {} labels ({assignments} legal assignments)",
            draw.columns,
            draw.rows,
            draw.labels,
        );

        // A full-grid tie has one seam-free optimum per label. Candidate order
        // is the sole tie-break, so the first candidate must win every cell.
        let mut tied = focus_fuser::OwnershipCostGrid::new(
            draw.columns,
            draw.rows,
            draw.labels,
        );
        for cell in 0..draw.cell_count() {
            for label in 0..draw.labels {
                tied.set_cost(cell, label, Some(0.5));
            }
            tied.set_disagreement(cell, draw.disagreement(cell));
        }
        let tied_first = focus_fuser::solve_ownership(&tied);
        let tied_repeated = focus_fuser::solve_ownership(&tied);
        prop_assert_eq!(&tied_first.owners, &tied_repeated.owners);
        prop_assert_eq!(
            tied_first.owners,
            vec![focus_fuser::owner_of(0); draw.cell_count()],
            "an exact tie must resolve to the first candidate deterministically"
        );

        // Mutation-sensitivity witness: unlike the per-cell fallback, the graph
        // cut must suppress an isolated label change whose seam costs more than
        // its data-term improvement.
        let witness = graph_cut_non_vacuity_witness();
        let witness_solution = focus_fuser::solve_ownership(&witness.production_grid());
        let witness_cost = witness.energy(&witness_solution.owners);
        let (witness_optimum, _) = exhaustive_graph_cut_oracle(&witness);
        prop_assert!(witness_cost <= witness_optimum * 1.01);
    }
}

#[test]
fn graph_cut_saved_counterexample_stays_within_property_12_bound() {
    let draw = CostGridDraw {
        columns: 2,
        rows: 3,
        labels: 3,
        costs: vec![
            Some(1),
            None,
            Some(1),
            None,
            Some(1),
            Some(1),
            None,
            Some(1),
            Some(1),
            Some(3),
            Some(1),
            None,
            None,
            None,
            None,
            Some(1),
            None,
            None,
        ],
        disagreement: vec![0, 0, 0, 24, 0, 22],
    };
    let solution = focus_fuser::solve_ownership(&draw.production_grid());
    let achieved = draw.energy(&solution.owners);
    assert!(
        achieved <= 6.53 * 1.01 + 1.0e-9,
        "counterexample energy {achieved}"
    );
}
