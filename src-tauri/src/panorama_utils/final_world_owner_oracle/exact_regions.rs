//! Exact-cost region proposals and the reduced exact solve.
//!
//! Lifts coarse region options to final-pixel rectangles and selects disjoint
//! regions from exact run costs.

use serde::Serialize;
use std::collections::BTreeMap;

use super::{OracleInput, region_options, sacrifice};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(in crate::panorama_utils::stitching) struct ExactRegionProposal {
    pub source: u64,
    pub station: u16,
    pub left: u32,
    pub top: u32,
    pub side: u32,
    /// Ranking hint only. Exact feasibility and costs never consume it.
    pub coarse_current_shortfall_pixels: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(in crate::panorama_utils::stitching) struct ExactRegionEvaluation {
    pub proposal: ExactRegionProposal,
    pub valid: bool,
    pub textured_pixels: u64,
    pub current_shortfall_pixels: u64,
    pub baseline_shortfall_pixels: u64,
    pub candidate_shortfall_pixels: u64,
    pub owner_change_pixels: u64,
    pub boundary_change_pixels: u64,
    pub data_cost_delta: f64,
    pub pairwise_cost: f64,
    pub sparse_run_count: u64,
    pub prediction_replay_mismatch_pixels: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(in crate::panorama_utils::stitching) struct ExactRegionSolveReport {
    pub feasible: bool,
    pub required_regions: usize,
    pub evaluated_options: usize,
    pub exact_valid_options: usize,
    pub reduced_options: usize,
    pub textured_pixels: u64,
    pub current_shortfall_pixels: u64,
    pub baseline_shortfall_pixels: u64,
    pub selected_shortfall_pixels: Option<u64>,
    pub current_sharpness_coverage: f64,
    pub baseline_sharpness_coverage: f64,
    pub relaxation_sharpness_coverage_upper_bound: f64,
    pub selected_sharpness_coverage: Option<f64>,
    pub selected: Vec<ExactRegionEvaluation>,
}

pub(super) fn exact_regions_overlap(left: ExactRegionProposal, right: ExactRegionProposal) -> bool {
    left.left < right.left.saturating_add(right.side)
        && right.left < left.left.saturating_add(left.side)
        && left.top < right.top.saturating_add(right.side)
        && right.top < left.top.saturating_add(left.side)
}

/// Lift generic mask/content origins into final-pixel rectangles. The coarse
/// grid proposes origins only; every pixel is independently revalidated by the
/// exact evaluator before an option can enter the solver.
pub(in crate::panorama_utils::stitching) fn exact_region_proposals(
    input: &OracleInput,
) -> Vec<ExactRegionProposal> {
    let cell_side = input.geometry.cell_side_px.max(1);
    let side = if input.geometry.qg_aligned_region_grid {
        input.geometry.regions.evidence_span_px()
    } else {
        input.geometry.regions.protected_span_px()
    };
    region_options(input)
        .into_iter()
        .map(|option| {
            let coarse_current_shortfall_pixels = option
                .cells
                .iter()
                .map(|&cell| {
                    sacrifice(
                        &input.cells[cell],
                        input.cells[cell].production_source,
                        input.shortfall_limit,
                    )
                })
                .sum();
            ExactRegionProposal {
                source: option.source,
                station: option.station,
                left: option.origin.0.saturating_mul(cell_side),
                top: option.origin.1.saturating_mul(cell_side),
                side,
                coarse_current_shortfall_pixels,
            }
        })
        .collect()
}

/// Deterministic exact-cost reduced solve. The relaxation ignores overlap and
/// therefore can only improve on a feasible solution (lower shortfall / upper
/// coverage); it never excludes a potentially better branch.
pub(in crate::panorama_utils::stitching) fn solve_exact_regions(
    textured_pixels: u64,
    current_shortfall_pixels: u64,
    baseline_shortfall_pixels: u64,
    required_regions: usize,
    evaluations: &[ExactRegionEvaluation],
) -> ExactRegionSolveReport {
    let coverage = |shortfall: u64| {
        if textured_pixels == 0 {
            0.0
        } else {
            1.0 - shortfall.min(textured_pixels) as f64 / textured_pixels as f64
        }
    };
    let mut valid = evaluations
        .iter()
        .copied()
        .filter(|evaluation| evaluation.valid && evaluation.textured_pixels != 0)
        .collect::<Vec<_>>();
    valid.sort_by(|left, right| {
        let left_delta =
            i128::from(left.candidate_shortfall_pixels) - i128::from(left.current_shortfall_pixels);
        let right_delta = i128::from(right.candidate_shortfall_pixels)
            - i128::from(right.current_shortfall_pixels);
        left_delta
            .cmp(&right_delta)
            .then_with(|| left.owner_change_pixels.cmp(&right.owner_change_pixels))
            .then_with(|| left.data_cost_delta.total_cmp(&right.data_cost_delta))
            .then_with(|| left.pairwise_cost.total_cmp(&right.pairwise_cost))
            .then_with(|| left.proposal.top.cmp(&right.proposal.top))
            .then_with(|| left.proposal.left.cmp(&right.proposal.left))
            .then_with(|| left.proposal.source.cmp(&right.proposal.source))
    });
    let mut by_origin = BTreeMap::<(u32, u32), u64>::new();
    for evaluation in &valid {
        by_origin
            .entry((evaluation.proposal.left, evaluation.proposal.top))
            .and_modify(|current| *current = (*current).min(evaluation.candidate_shortfall_pixels))
            .or_insert(evaluation.candidate_shortfall_pixels);
    }
    let mut relaxed = by_origin.into_values().collect::<Vec<_>>();
    relaxed.sort_unstable();
    // Every admissible labeling must pay the selected protected regions'
    // exact shortfall. Setting every pixel outside those regions to zero
    // shortfall is optimistic, so this is a safe global coverage upper bound.
    let relaxation_shortfall = if relaxed.len() < required_regions {
        textured_pixels
    } else {
        relaxed
            .iter()
            .take(required_regions)
            .copied()
            .sum::<u64>()
            .min(textured_pixels)
    };

    const REDUCED_OPTION_LIMIT: usize = 512;
    let mut reduced = Vec::<ExactRegionEvaluation>::new();
    for evaluation in &valid {
        if reduced.len() == REDUCED_OPTION_LIMIT.max(required_regions) {
            break;
        }
        // Keep the best source label for each fixed QG origin, but retain
        // overlapping origins. Compatibility belongs in branch-and-bound;
        // greedily discarding overlap here can remove every feasible set of 8.
        if reduced.iter().all(|current| {
            current.proposal.left != evaluation.proposal.left
                || current.proposal.top != evaluation.proposal.top
        }) {
            reduced.push(*evaluation);
        }
    }
    let mut greedy = Vec::<usize>::new();
    for index in 0..reduced.len() {
        if greedy.iter().all(|&selected| {
            !exact_regions_overlap(reduced[selected].proposal, reduced[index].proposal)
        }) {
            greedy.push(index);
            if greedy.len() == required_regions {
                break;
            }
        }
    }
    let mut best = (greedy.len() == required_regions).then(|| {
        let delta = greedy
            .iter()
            .map(|&index| {
                i128::from(reduced[index].candidate_shortfall_pixels)
                    - i128::from(reduced[index].baseline_shortfall_pixels)
            })
            .sum();
        let objective = greedy
            .iter()
            .map(|&index| {
                reduced[index].owner_change_pixels as f64
                    + (reduced[index].data_cost_delta + reduced[index].pairwise_cost) * 1.0e-9
            })
            .sum();
        (delta, objective, greedy)
    });
    fn visit(
        reduced: &[ExactRegionEvaluation],
        required: usize,
        start: usize,
        selected: &mut Vec<usize>,
        delta: i128,
        objective: f64,
        best: &mut Option<(i128, f64, Vec<usize>)>,
    ) {
        if selected.len() == required {
            if best.as_ref().is_none_or(|current| {
                delta < current.0
                    || (delta == current.0
                        && (objective < current.1
                            || (objective == current.1
                                && selected.as_slice() < current.2.as_slice())))
            }) {
                *best = Some((delta, objective, selected.clone()));
            }
            return;
        }
        let needed = required - selected.len();
        if reduced.len().saturating_sub(start) < needed {
            return;
        }
        if let Some(current) = best.as_ref() {
            let mut remaining = reduced[start..]
                .iter()
                .map(|evaluation| {
                    i128::from(evaluation.candidate_shortfall_pixels)
                        - i128::from(evaluation.baseline_shortfall_pixels)
                })
                .collect::<Vec<_>>();
            remaining.sort_unstable();
            if delta + remaining.into_iter().take(needed).sum::<i128>() > current.0 {
                return;
            }
        }
        for cursor in start..=reduced.len() - needed {
            if selected.iter().all(|&index| {
                !exact_regions_overlap(reduced[index].proposal, reduced[cursor].proposal)
            }) {
                selected.push(cursor);
                let evaluation = reduced[cursor];
                visit(
                    reduced,
                    required,
                    cursor + 1,
                    selected,
                    delta + i128::from(evaluation.candidate_shortfall_pixels)
                        - i128::from(evaluation.baseline_shortfall_pixels),
                    objective
                        + evaluation.owner_change_pixels as f64
                        + (evaluation.data_cost_delta + evaluation.pairwise_cost) * 1.0e-9,
                    best,
                );
                selected.pop();
            }
        }
    }
    visit(
        &reduced,
        required_regions,
        0,
        &mut Vec::new(),
        0,
        0.0,
        &mut best,
    );
    let (selected_shortfall_pixels, selected) =
        best.map_or((None, Vec::new()), |(delta, _, indices)| {
            let shortfall = (i128::from(baseline_shortfall_pixels) + delta)
                .clamp(0, i128::from(textured_pixels)) as u64;
            (
                Some(shortfall),
                indices.into_iter().map(|index| reduced[index]).collect(),
            )
        });
    let selected_coverage = selected_shortfall_pixels.map(coverage);
    ExactRegionSolveReport {
        feasible: selected.len() == required_regions
            && selected_coverage.is_some_and(|value| value >= 0.99),
        required_regions,
        evaluated_options: evaluations.len(),
        exact_valid_options: valid.len(),
        reduced_options: reduced.len(),
        textured_pixels,
        current_shortfall_pixels,
        baseline_shortfall_pixels,
        selected_shortfall_pixels,
        current_sharpness_coverage: coverage(current_shortfall_pixels),
        baseline_sharpness_coverage: coverage(baseline_shortfall_pixels),
        relaxation_sharpness_coverage_upper_bound: coverage(relaxation_shortfall),
        selected_sharpness_coverage: selected_coverage,
        selected,
    }
}
