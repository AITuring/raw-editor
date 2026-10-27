//! Test-only constrained owner-quality feasibility oracle.
//!
//! The oracle consumes candidate costs and source-valid ROI origins measured by
//! the production focus path. It never writes pixels or changes graph-cut
//! labels. ROI geometry is supplied by the caller from the production Quality
//! Gate constants so synthetic tests can exercise the same rules at small scale.

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub(super) struct OracleCandidate {
    pub owner: u16,
    pub source: String,
    pub sharpness: f64,
    pub data_cost: f64,
    pub covered: bool,
}

#[derive(Clone, Debug)]
pub(super) struct OracleCell {
    pub textured_pixels: u64,
    pub candidates: Vec<OracleCandidate>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct OracleGeometry {
    pub width: u32,
    pub height: u32,
    pub columns: u32,
    pub rows: u32,
    pub cell_size: u32,
    pub roi_side: u32,
    pub roi_step: u32,
    pub roi_margin: u32,
}

#[derive(Clone, Debug)]
pub(super) struct SourceRoiOrigin {
    pub owner: u16,
    pub source: String,
    pub x: u32,
    pub y: u32,
}

#[derive(Clone, Debug)]
pub(super) struct ExternalRoiOption {
    pub owner: u16,
    pub source: String,
    pub x: u32,
    pub y: u32,
    pub cells: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(super) struct OracleInput {
    pub station_index: usize,
    pub geometry: OracleGeometry,
    pub cells: Vec<OracleCell>,
    pub production_owners: Vec<u16>,
    /// Origins whose ROI plus production margin is fully covered by this exact
    /// source. These are measured from source masks, not inferred from cells.
    pub source_roi_origins: Vec<SourceRoiOrigin>,
    /// Optional exact final-output ROI footprints mapped back to this station's
    /// ownership cells. When present, these replace station-plane candidates.
    pub external_roi_options: Vec<ExternalRoiOption>,
}

#[derive(Clone, Debug)]
struct RoiOption {
    origin: (u32, u32),
    owner: u16,
    source: String,
    cells: Vec<usize>,
    individual_sacrifice: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct OracleCurvePoint {
    pub required_rois: usize,
    pub relaxation_lower_bound_sacrifice_pixels: u64,
    pub safely_conservative_coverage_upper_bound: f64,
    pub best_feasible_sacrifice_pixels: Option<u64>,
    pub best_feasible_coverage: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct OracleReport {
    pub schema: u32,
    pub station_index: usize,
    pub dimensions: [u32; 2],
    pub grid: [u32; 3],
    pub roi_side: u32,
    pub roi_step: u32,
    pub roi_margin: u32,
    pub textured_pixels: u64,
    pub unconstrained_pixelwise_coverage: f64,
    pub data_term_winner_coverage: f64,
    pub current_graph_cut_coverage: f64,
    pub exact_source_covered_roi_options: usize,
    pub distinct_roi_origins: usize,
    pub reduced_branch_and_bound_options: usize,
    pub curve: Vec<OracleCurvePoint>,
    pub best_eight_roi_origins: Vec<[u32; 2]>,
    pub best_eight_roi_sources: Vec<String>,
    pub best_eight_labeling: Option<Vec<u16>>,
    pub production_preserving_eight_sacrifice_pixels: Option<u64>,
    pub production_preserving_eight_coverage: Option<f64>,
    pub production_preserving_eight_roi_origins: Vec<[u32; 2]>,
    pub production_preserving_eight_roi_sources: Vec<String>,
    pub production_preserving_eight_labeling: Option<Vec<u16>>,
    pub proof_scope: String,
}

fn candidate_for_owner(cell: &OracleCell, owner: u16) -> Option<&OracleCandidate> {
    cell.candidates
        .iter()
        .filter(|candidate| candidate.covered && candidate.owner == owner)
        .min_by(|left, right| {
            left.data_cost
                .total_cmp(&right.data_cost)
                .then_with(|| left.source.cmp(&right.source))
        })
}

fn winner(cell: &OracleCell) -> Option<&OracleCandidate> {
    cell.candidates
        .iter()
        .filter(|candidate| candidate.covered)
        .max_by(|left, right| {
            left.sharpness
                .total_cmp(&right.sharpness)
                .then_with(|| right.source.cmp(&left.source))
        })
}

fn shortfall(cell: &OracleCell, owner: u16) -> f64 {
    let Some(best) = winner(cell) else {
        return 0.0;
    };
    if best.sharpness <= 0.0 || cell.candidates.iter().filter(|c| c.covered).count() <= 1 {
        return 0.0;
    }
    let score = candidate_for_owner(cell, owner).map_or(0.0, |candidate| candidate.sharpness);
    ((best.sharpness - score) / best.sharpness).clamp(0.0, 1.0)
}

fn sacrifice(cell: &OracleCell, owner: u16, shortfall_limit: f64) -> u64 {
    if shortfall(cell, owner) > shortfall_limit {
        cell.textured_pixels
    } else {
        0
    }
}

fn cells_for_padded_roi(geometry: OracleGeometry, x: u32, y: u32) -> Option<Vec<usize>> {
    let left = x.checked_sub(geometry.roi_margin)?;
    let top = y.checked_sub(geometry.roi_margin)?;
    let right = x
        .checked_add(geometry.roi_side)?
        .checked_add(geometry.roi_margin)?;
    let bottom = y
        .checked_add(geometry.roi_side)?
        .checked_add(geometry.roi_margin)?;
    if right > geometry.width || bottom > geometry.height || geometry.cell_size == 0 {
        return None;
    }
    let first_column = left / geometry.cell_size;
    let last_column = (right - 1) / geometry.cell_size;
    let first_row = top / geometry.cell_size;
    let last_row = (bottom - 1) / geometry.cell_size;
    if last_column >= geometry.columns || last_row >= geometry.rows {
        return None;
    }
    Some(
        (first_row..=last_row)
            .flat_map(|row| {
                (first_column..=last_column)
                    .map(move |column| (row * geometry.columns + column) as usize)
            })
            .collect(),
    )
}

fn options(input: &OracleInput, shortfall_limit: f64) -> Vec<RoiOption> {
    if !input.external_roi_options.is_empty() {
        let mut external = input.external_roi_options.clone();
        external.sort_by(|left, right| {
            (left.y, left.x, left.owner, &left.source).cmp(&(
                right.y,
                right.x,
                right.owner,
                &right.source,
            ))
        });
        external.dedup_by(|left, right| {
            left.x == right.x
                && left.y == right.y
                && left.owner == right.owner
                && left.cells == right.cells
        });
        return external
            .into_iter()
            .filter_map(|option| {
                if option.cells.is_empty()
                    || option.cells.iter().any(|&cell| {
                        input
                            .cells
                            .get(cell)
                            .and_then(|cell| candidate_for_owner(cell, option.owner))
                            .is_none()
                    })
                {
                    return None;
                }
                let individual_sacrifice = option
                    .cells
                    .iter()
                    .map(|&cell| sacrifice(&input.cells[cell], option.owner, shortfall_limit))
                    .sum();
                Some(RoiOption {
                    origin: (option.x, option.y),
                    owner: option.owner,
                    source: option.source,
                    cells: option.cells,
                    individual_sacrifice,
                })
            })
            .collect();
    }
    let mut origins = input.source_roi_origins.clone();
    origins.sort_by(|left, right| {
        (left.y, left.x, left.owner, &left.source).cmp(&(
            right.y,
            right.x,
            right.owner,
            &right.source,
        ))
    });
    origins.dedup_by(|left, right| {
        left.x == right.x
            && left.y == right.y
            && left.owner == right.owner
            && left.source == right.source
    });
    origins
        .into_iter()
        .filter_map(|origin| {
            let cells = cells_for_padded_roi(input.geometry, origin.x, origin.y)?;
            if cells.iter().any(|&cell| {
                input
                    .cells
                    .get(cell)
                    .and_then(|cell| candidate_for_owner(cell, origin.owner))
                    .is_none()
            }) {
                return None;
            }
            let individual_sacrifice = cells
                .iter()
                .map(|&cell| sacrifice(&input.cells[cell], origin.owner, shortfall_limit))
                .sum();
            Some(RoiOption {
                origin: (origin.x, origin.y),
                owner: origin.owner,
                source: origin.source,
                cells,
                individual_sacrifice,
            })
        })
        .collect()
}

fn labeling_cost(input: &OracleInput, labels: &[u16], shortfall_limit: f64) -> u64 {
    input
        .cells
        .iter()
        .zip(labels)
        .map(|(cell, &owner)| sacrifice(cell, owner, shortfall_limit))
        .sum()
}

fn apply_option(labels: &mut [u16], option: &RoiOption) -> bool {
    if option
        .cells
        .iter()
        .any(|&cell| labels[cell] != 0 && labels[cell] != option.owner)
    {
        return false;
    }
    for &cell in &option.cells {
        labels[cell] = option.owner;
    }
    true
}

fn roi_options_overlap(left: &RoiOption, right: &RoiOption, side: u32) -> bool {
    left.origin.0 < right.origin.0.saturating_add(side)
        && right.origin.0 < left.origin.0.saturating_add(side)
        && left.origin.1 < right.origin.1.saturating_add(side)
        && right.origin.1 < left.origin.1.saturating_add(side)
}

fn selected_options_are_distinct(
    input: &OracleInput,
    selected: &[usize],
    options: &[RoiOption],
) -> bool {
    selected.iter().enumerate().all(|(position, &left)| {
        selected[position + 1..].iter().all(|&right| {
            options[left].origin != options[right].origin
                && !roi_options_overlap(&options[left], &options[right], input.geometry.roi_side)
        })
    })
}

fn selected_cost(
    input: &OracleInput,
    selected: &[usize],
    options: &[RoiOption],
    limit: f64,
) -> Option<(u64, Vec<u16>)> {
    if !selected_options_are_distinct(input, selected, options) {
        return None;
    }
    let mut labels = vec![0u16; input.cells.len()];
    for &index in selected {
        if !apply_option(&mut labels, &options[index]) {
            return None;
        }
    }
    for (cell, label) in input.cells.iter().zip(&mut labels) {
        if *label == 0 {
            *label = winner(cell).map_or(0, |candidate| candidate.owner);
        }
    }
    Some((labeling_cost(input, &labels, limit), labels))
}

fn greedy_feasible(
    input: &OracleInput,
    options: &[RoiOption],
    count: usize,
    limit: f64,
) -> Option<(u64, Vec<usize>, Vec<u16>)> {
    if count == 0 {
        let labels = input
            .cells
            .iter()
            .map(|cell| winner(cell).map_or(0, |candidate| candidate.owner))
            .collect::<Vec<_>>();
        return Some((labeling_cost(input, &labels, limit), Vec::new(), labels));
    }
    let mut selected = Vec::new();
    let mut used_origins = BTreeSet::new();
    while selected.len() < count {
        let mut best = None;
        for index in 0..options.len() {
            if used_origins.contains(&options[index].origin) {
                continue;
            }
            let mut trial = selected.clone();
            trial.push(index);
            let Some((cost, _)) = selected_cost(input, &trial, options, limit) else {
                continue;
            };
            let key = (
                cost,
                options[index].origin.1,
                options[index].origin.0,
                options[index].owner,
                options[index].source.as_str(),
            );
            if best.as_ref().is_none_or(|(best_key, _)| key < *best_key) {
                best = Some((key, index));
            }
        }
        let (_, index) = best?;
        used_origins.insert(options[index].origin);
        selected.push(index);
    }
    let (cost, labels) = selected_cost(input, &selected, options, limit)?;
    Some((cost, selected, labels))
}

fn reduced_exact(
    input: &OracleInput,
    options: &[RoiOption],
    count: usize,
    limit: f64,
    maximum_options: usize,
) -> Option<(u64, Vec<usize>, Vec<u16>, usize)> {
    if count == 0 {
        let (cost, selected, labels) = greedy_feasible(input, options, 0, limit)?;
        return Some((cost, selected, labels, 0));
    }
    let mut ranked = (0..options.len()).collect::<Vec<_>>();
    ranked.sort_by(|&left, &right| {
        options[left]
            .individual_sacrifice
            .cmp(&options[right].individual_sacrifice)
            .then_with(|| options[left].origin.1.cmp(&options[right].origin.1))
            .then_with(|| options[left].origin.0.cmp(&options[right].origin.0))
            .then_with(|| options[left].owner.cmp(&options[right].owner))
            .then_with(|| options[left].source.cmp(&options[right].source))
    });
    ranked.truncate(maximum_options.max(count));
    let mut best: Option<(u64, Vec<usize>, Vec<u16>)> = None;
    fn visit(
        input: &OracleInput,
        options: &[RoiOption],
        ranked: &[usize],
        count: usize,
        limit: f64,
        start: usize,
        selected: &mut Vec<usize>,
        origins: &mut BTreeSet<(u32, u32)>,
        best: &mut Option<(u64, Vec<usize>, Vec<u16>)>,
    ) {
        if selected.len() == count {
            if let Some((cost, labels)) = selected_cost(input, selected, options, limit) {
                if best.as_ref().is_none_or(|(best_cost, best_selected, _)| {
                    cost < *best_cost
                        || (cost == *best_cost && selected.as_slice() < best_selected.as_slice())
                }) {
                    *best = Some((cost, selected.clone(), labels));
                }
            }
            return;
        }
        let needed = count - selected.len();
        if ranked.len().saturating_sub(start) < needed {
            return;
        }
        for cursor in start..=ranked.len() - needed {
            let index = ranked[cursor];
            if !origins.insert(options[index].origin) {
                continue;
            }
            selected.push(index);
            if selected_cost(input, selected, options, limit).is_some() {
                visit(
                    input,
                    options,
                    ranked,
                    count,
                    limit,
                    cursor + 1,
                    selected,
                    origins,
                    best,
                );
            }
            selected.pop();
            origins.remove(&options[index].origin);
        }
    }
    visit(
        input,
        options,
        &ranked,
        count,
        limit,
        0,
        &mut Vec::new(),
        &mut BTreeSet::new(),
        &mut best,
    );
    best.map(|(cost, selected, labels)| (cost, selected, labels, ranked.len()))
}

/// Safely conservative relaxation: every feasible labeling with `k` distinct
/// ROI origins must contain `k` origins whose individual forced-label costs do
/// not exceed its union cost. Ignoring owner conflicts and overlap can only
/// lower this sacrifice bound, never raise it.
fn relaxation_lower_bound(options: &[RoiOption], count: usize) -> Option<u64> {
    if count == 0 {
        return Some(0);
    }
    let mut by_origin = BTreeMap::<(u32, u32), u64>::new();
    for option in options {
        by_origin
            .entry(option.origin)
            .and_modify(|cost| *cost = (*cost).min(option.individual_sacrifice))
            .or_insert(option.individual_sacrifice);
    }
    let mut costs = by_origin.into_values().collect::<Vec<_>>();
    costs.sort_unstable();
    costs.get(count - 1).copied()
}

pub(super) fn solve(input: OracleInput, shortfall_limit: f64) -> OracleReport {
    assert_eq!(
        input.cells.len(),
        (input.geometry.columns * input.geometry.rows) as usize
    );
    assert_eq!(input.production_owners.len(), input.cells.len());
    let textured_pixels = input
        .cells
        .iter()
        .map(|cell| cell.textured_pixels)
        .sum::<u64>();
    let options = options(&input, shortfall_limit);
    let distinct_roi_origins = options
        .iter()
        .map(|option| option.origin)
        .collect::<BTreeSet<_>>()
        .len();
    let current_sacrifice = labeling_cost(&input, &input.production_owners, shortfall_limit);
    let data_term_owners = input
        .cells
        .iter()
        .map(|cell| {
            cell.candidates
                .iter()
                .filter(|candidate| candidate.covered)
                .min_by(|left, right| {
                    left.data_cost
                        .total_cmp(&right.data_cost)
                        .then_with(|| left.source.cmp(&right.source))
                })
                .map_or(0, |candidate| candidate.owner)
        })
        .collect::<Vec<_>>();
    let data_term_sacrifice = labeling_cost(&input, &data_term_owners, shortfall_limit);
    let coverage = |sacrifice: u64| {
        if textured_pixels == 0 {
            0.0
        } else {
            1.0 - sacrifice as f64 / textured_pixels as f64
        }
    };
    let mut curve = Vec::new();
    let mut best_eight = None;
    let mut reduced_count = 0;
    for required_rois in 1..=8 {
        let lower_bound =
            relaxation_lower_bound(&options, required_rois).unwrap_or(textured_pixels);
        let greedy = greedy_feasible(&input, &options, required_rois, shortfall_limit);
        // Exact search is deliberately reduced to the best twelve independent
        // windows. Its result is a feasible lower coverage, never the proof
        // bound; the separate relaxation above supplies the safe upper bound.
        let exact = reduced_exact(&input, &options, required_rois, shortfall_limit, 12);
        reduced_count = reduced_count.max(exact.as_ref().map_or(0, |value| value.3));
        let feasible = match (greedy, exact) {
            (Some(left), Some(right)) if right.0 < left.0 => Some((right.0, right.1, right.2)),
            (Some(left), _) => Some(left),
            (None, Some(right)) => Some((right.0, right.1, right.2)),
            (None, None) => None,
        };
        if required_rois == 8 {
            best_eight = feasible.clone();
        }
        curve.push(OracleCurvePoint {
            required_rois,
            relaxation_lower_bound_sacrifice_pixels: lower_bound,
            safely_conservative_coverage_upper_bound: coverage(lower_bound),
            best_feasible_sacrifice_pixels: feasible.as_ref().map(|value| value.0),
            best_feasible_coverage: feasible.as_ref().map(|value| coverage(value.0)),
        });
    }
    let (origins, sources, labels) = best_eight.map_or_else(
        || (Vec::new(), Vec::new(), None),
        |(_, selected, labels)| {
            (
                selected
                    .iter()
                    .map(|&index| [options[index].origin.0, options[index].origin.1])
                    .collect(),
                selected
                    .iter()
                    .map(|&index| options[index].source.clone())
                    .collect(),
                Some(labels),
            )
        },
    );
    let production_preserving_options = options
        .iter()
        .filter(|option| {
            option
                .cells
                .iter()
                .all(|&cell| input.production_owners.get(cell).copied() == Some(option.owner))
        })
        .cloned()
        .collect::<Vec<_>>();
    let production_preserving =
        greedy_feasible(&input, &production_preserving_options, 8, shortfall_limit);
    let (
        production_preserving_sacrifice,
        production_preserving_origins,
        production_preserving_sources,
        production_preserving_labels,
    ) = production_preserving.map_or_else(
        || (None, Vec::new(), Vec::new(), None),
        |(cost, selected, labels)| {
            (
                Some(cost),
                selected
                    .iter()
                    .map(|&index| {
                        [
                            production_preserving_options[index].origin.0,
                            production_preserving_options[index].origin.1,
                        ]
                    })
                    .collect(),
                selected
                    .iter()
                    .map(|&index| production_preserving_options[index].source.clone())
                    .collect(),
                Some(labels),
            )
        },
    );
    OracleReport {
        schema: 1,
        station_index: input.station_index,
        dimensions: [input.geometry.width, input.geometry.height],
        grid: [
            input.geometry.columns,
            input.geometry.rows,
            input.geometry.cell_size,
        ],
        roi_side: input.geometry.roi_side,
        roi_step: input.geometry.roi_step,
        roi_margin: input.geometry.roi_margin,
        textured_pixels,
        unconstrained_pixelwise_coverage: coverage(0),
        data_term_winner_coverage: coverage(data_term_sacrifice),
        current_graph_cut_coverage: coverage(current_sacrifice),
        exact_source_covered_roi_options: options.len(),
        distinct_roi_origins,
        reduced_branch_and_bound_options: reduced_count,
        curve,
        best_eight_roi_origins: origins,
        best_eight_roi_sources: sources,
        best_eight_labeling: labels,
        production_preserving_eight_sacrifice_pixels: production_preserving_sacrifice,
        production_preserving_eight_coverage: production_preserving_sacrifice.map(coverage),
        production_preserving_eight_roi_origins: production_preserving_origins,
        production_preserving_eight_roi_sources: production_preserving_sources,
        production_preserving_eight_labeling: production_preserving_labels,
        proof_scope: "ROI interiors are pairwise non-overlapping; upper bound is the independent distinct-origin order-statistic relaxation; feasible values are exact costs for explicit compatible labelings, with branch-and-bound exact only on the reported reduced option set".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(width: u32, height: u32, cell_size: u32, roi_side: u32) -> OracleInput {
        let columns = width.div_ceil(cell_size);
        let rows = height.div_ceil(cell_size);
        let cells = (0..columns * rows)
            .map(|_| OracleCell {
                textured_pixels: u64::from(cell_size) * u64::from(cell_size),
                candidates: vec![
                    OracleCandidate {
                        owner: 1,
                        source: "source-a".to_string(),
                        sharpness: 1.0,
                        data_cost: 0.0,
                        covered: true,
                    },
                    OracleCandidate {
                        owner: 2,
                        source: "source-b".to_string(),
                        sharpness: 0.5,
                        data_cost: 0.5,
                        covered: true,
                    },
                ],
            })
            .collect::<Vec<_>>();
        let geometry = OracleGeometry {
            width,
            height,
            columns,
            rows,
            cell_size,
            roi_side,
            roi_step: roi_side / 2,
            roi_margin: cell_size,
        };
        let mut source_roi_origins = Vec::new();
        for y in (geometry.roi_margin..height - roi_side).step_by(geometry.roi_step as usize) {
            for x in (geometry.roi_margin..width - roi_side).step_by(geometry.roi_step as usize) {
                source_roi_origins.push(SourceRoiOrigin {
                    owner: 1,
                    source: "source-a".to_string(),
                    x,
                    y,
                });
                source_roi_origins.push(SourceRoiOrigin {
                    owner: 2,
                    source: "source-b".to_string(),
                    x,
                    y,
                });
            }
        }
        OracleInput {
            station_index: 0,
            geometry,
            production_owners: vec![2; cells.len()],
            cells,
            source_roi_origins,
            external_roi_options: Vec::new(),
        }
    }

    fn brute_force_minimum(input: &OracleInput, count: usize) -> Option<u64> {
        let opts = options(input, 0.05);
        let mut best = None;
        fn visit(
            input: &OracleInput,
            opts: &[RoiOption],
            count: usize,
            start: usize,
            picked: &mut Vec<usize>,
            best: &mut Option<u64>,
        ) {
            if picked.len() == count {
                if picked
                    .iter()
                    .map(|&i| opts[i].origin)
                    .collect::<BTreeSet<_>>()
                    .len()
                    != count
                {
                    return;
                }
                if let Some((cost, _)) = selected_cost(input, picked, opts, 0.05) {
                    *best = Some(best.map_or(cost, |current: u64| current.min(cost)));
                }
                return;
            }
            for index in start..opts.len() {
                picked.push(index);
                visit(input, opts, count, index + 1, picked, best);
                picked.pop();
            }
        }
        visit(input, &opts, count, 0, &mut Vec::new(), &mut best);
        best
    }

    #[test]
    fn oracle_has_no_reserved_or_hardcoded_roi_positions() {
        let input = synthetic(18, 18, 2, 4);
        let report = solve(input.clone(), 0.05);
        assert!(!report.best_eight_roi_origins.is_empty());
        assert!(report.best_eight_roi_origins.iter().all(|origin| {
            input
                .source_roi_origins
                .iter()
                .any(|candidate| [candidate.x, candidate.y] == *origin)
        }));
    }

    #[test]
    fn permutation_and_world_translation_are_deterministic() {
        let input = synthetic(18, 18, 2, 4);
        let first = solve(input.clone(), 0.05);
        let mut permuted = input.clone();
        permuted.source_roi_origins.reverse();
        for cell in &mut permuted.cells {
            cell.candidates.reverse();
        }
        let second = solve(permuted, 0.05);
        assert_eq!(first.best_eight_roi_origins, second.best_eight_roi_origins);
        assert_eq!(first.best_eight_roi_sources, second.best_eight_roi_sources);
        assert_eq!(
            first.curve[7].best_feasible_sacrifice_pixels,
            second.curve[7].best_feasible_sacrifice_pixels
        );

        let mut translated = synthetic(20, 20, 2, 4);
        for cell in &mut translated.cells {
            cell.textured_pixels = 0;
        }
        translated.source_roi_origins = input
            .source_roi_origins
            .iter()
            .cloned()
            .map(|mut origin| {
                origin.x += 2;
                origin.y += 2;
                origin
            })
            .collect();
        for row in 0..input.geometry.rows as usize {
            for column in 0..input.geometry.columns as usize {
                let source = row * input.geometry.columns as usize + column;
                let destination = (row + 1) * translated.geometry.columns as usize + column + 1;
                translated.cells[destination] = input.cells[source].clone();
                translated.production_owners[destination] = input.production_owners[source];
            }
        }
        let shifted = solve(translated, 0.05);
        let shifted_local = shifted
            .best_eight_roi_origins
            .iter()
            .map(|origin| [origin[0] - 2, origin[1] - 2])
            .collect::<Vec<_>>();
        assert_eq!(first.best_eight_roi_origins, shifted_local);
        assert_eq!(
            first.curve[7].best_feasible_sacrifice_pixels,
            shifted.curve[7].best_feasible_sacrifice_pixels
        );
    }

    #[test]
    fn relaxation_upper_bound_never_understates_bruteforce_optimum() {
        let input = synthetic(14, 14, 2, 4);
        let opts = options(&input, 0.05);
        for count in 1..=3 {
            let optimum = brute_force_minimum(&input, count).unwrap();
            let lower = relaxation_lower_bound(&opts, count).unwrap();
            assert!(lower <= optimum);
        }
    }

    #[test]
    fn disconnected_duplicate_or_overlapping_origins_cannot_fake_regions() {
        let mut input = synthetic(10, 10, 2, 4);
        input.source_roi_origins = vec![
            SourceRoiOrigin {
                owner: 1,
                source: "source-a".to_string(),
                x: 2,
                y: 2,
            },
            SourceRoiOrigin {
                owner: 1,
                source: "source-a".to_string(),
                x: 4,
                y: 2,
            },
        ];
        let report = solve(input, 0.05);
        assert_eq!(report.distinct_roi_origins, 2);
        assert!(report.curve[1].best_feasible_coverage.is_none());
    }

    #[test]
    fn feasible_point_ninety_nine_with_eight_real_rois_is_found() {
        let mut input = synthetic(18, 18, 2, 4);
        input.production_owners.fill(1);
        let report = solve(input, 0.05);
        assert!(report.curve[7].best_feasible_coverage.unwrap() >= 0.99);
        assert_eq!(report.best_eight_roi_origins.len(), 8);
        assert!(
            report
                .best_eight_roi_sources
                .iter()
                .all(|source| source == "source-a")
        );
    }

    #[test]
    fn impossible_case_is_certified_by_independent_relaxation() {
        let mut input = synthetic(18, 18, 2, 4);
        input.source_roi_origins.retain(|origin| origin.owner == 2);
        let report = solve(input, 0.05);
        assert!(report.curve[7].safely_conservative_coverage_upper_bound < 0.99);
    }

    #[test]
    fn selected_labels_preserve_one_real_source_provenance() {
        let input = synthetic(18, 18, 2, 4);
        let report = solve(input.clone(), 0.05);
        let labels = report.best_eight_labeling.unwrap();
        assert!(labels.iter().all(|&owner| {
            input.cells[0]
                .candidates
                .iter()
                .any(|candidate| candidate.covered && candidate.owner == owner)
        }));
        assert!(
            report
                .best_eight_roi_sources
                .iter()
                .all(|source| source == "source-a" || source == "source-b")
        );
    }
}
