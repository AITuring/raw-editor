//! Coherent-region options and the reduced coarse-grid solve.
//!
//! Region origins come only from each source's coverage and low-frequency content
//! components; the solve pairs a reduced branch-and-bound with an independent
//! relaxation bound.

use serde::Serialize;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::VecDeque;

use super::{
    OracleInput, candidate, objective_cost, production_candidate, sacrifice, unconstrained_label,
};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(in crate::panorama_utils::stitching) struct SelectedRegion {
    pub source: u64,
    pub station: u16,
    pub local_origin: [u32; 2],
    pub world_origin: [i64; 2],
    pub protected_side_cells: u32,
    pub evidence_side_px: u32,
    pub protected_side_px: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(in crate::panorama_utils::stitching) struct OracleReport {
    pub feasible: bool,
    pub required_regions: usize,
    pub genuine_region_options: usize,
    pub distinct_region_origins: usize,
    pub reduced_options: usize,
    pub textured_pixels: u64,
    pub current_sharpness_coverage: f64,
    pub relaxation_sharpness_coverage_upper_bound: f64,
    pub feasible_sharpness_coverage: Option<f64>,
    pub objective_cost: Option<f64>,
    pub selected_regions: Vec<SelectedRegion>,
    pub labels: Option<Vec<u64>>,
    pub proof_scope: String,
}

#[derive(Clone, Debug)]
pub(super) struct RegionOption {
    pub(super) source: u64,
    pub(super) station: u16,
    pub(super) origin: (u32, u32),
    pub(super) cells: Vec<usize>,
    pub(super) individual_sacrifice: u64,
    pub(super) incremental_cost: f64,
}

pub(super) fn neighbour_indices(index: usize, columns: usize, rows: usize) -> [Option<usize>; 4] {
    let (x, y) = (index % columns, index / columns);
    [
        (x > 0).then(|| index - 1),
        (x + 1 < columns).then(|| index + 1),
        (y > 0).then(|| index - columns),
        (y + 1 < rows).then(|| index + columns),
    ]
}

pub(super) fn source_component_map(input: &OracleInput, source: u64) -> (Vec<usize>, Vec<usize>) {
    let columns = input.geometry.columns as usize;
    let rows = input.geometry.rows as usize;
    let mut ids = vec![usize::MAX; input.cells.len()];
    let mut sizes = Vec::new();
    for seed in 0..input.cells.len() {
        let eligible = candidate(&input.cells[seed], source)
            .is_some_and(|sample| sample.flat && sample.low_frequency_luma.is_finite());
        if !eligible || ids[seed] != usize::MAX {
            continue;
        }
        let id = sizes.len();
        let mut size = 0usize;
        let mut queue = VecDeque::from([seed]);
        ids[seed] = id;
        while let Some(cell) = queue.pop_front() {
            size += 1;
            for neighbour in neighbour_indices(cell, columns, rows).into_iter().flatten() {
                if ids[neighbour] != usize::MAX {
                    continue;
                }
                if candidate(&input.cells[neighbour], source)
                    .is_some_and(|sample| sample.flat && sample.low_frequency_luma.is_finite())
                {
                    ids[neighbour] = id;
                    queue.push_back(neighbour);
                }
            }
        }
        sizes.push(size);
    }
    (ids, sizes)
}

pub(super) fn region_options(input: &OracleInput) -> Vec<RegionOption> {
    let geometry = input.geometry;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    let cell_side = geometry.cell_side_px.max(1);
    let evidence_cells = geometry
        .regions
        .evidence_span_px()
        .div_ceil(cell_side)
        .max(1) as usize;
    let protected_span_px = if geometry.qg_aligned_region_grid {
        geometry.regions.evidence_span_px()
    } else {
        geometry.regions.protected_span_px()
    };
    let protected_cells = protected_span_px.div_ceil(cell_side).max(1) as usize;
    if protected_cells > columns || protected_cells > rows {
        return Vec::new();
    }
    let phase_cells = geometry
        .regions
        .sampling_phase_px
        .max(1)
        .div_ceil(cell_side)
        .max(1) as usize;
    let margin_cells = geometry
        .regions
        .same_owner_ring_px
        .div_ceil(cell_side)
        .max(u32::from(geometry.regions.same_owner_ring_px != 0)) as usize;
    // Quality_Gate places ROI top-left corners at 256px multiples and then
    // expands each side by the 16px owner ring. Therefore protected origins
    // are 240 mod 256, not arbitrary translated phases.
    let origin_start = if geometry.qg_aligned_region_grid {
        (phase_cells - margin_cells % phase_cells) % phase_cells
    } else {
        0
    };
    let sources = input
        .cells
        .iter()
        .flat_map(|cell| cell.candidates.iter().map(|sample| sample.source))
        .filter(|source| *source != 0)
        .collect::<BTreeSet<_>>();
    let mut options = Vec::new();
    for source in sources {
        let (components, sizes) = source_component_map(input, source);
        for top in (origin_start..=rows - protected_cells).step_by(phase_cells) {
            for left in (origin_start..=columns - protected_cells).step_by(phase_cells) {
                let component = components[top * columns + left];
                if component == usize::MAX
                    || sizes.get(component).copied().unwrap_or(0)
                        < evidence_cells.saturating_mul(evidence_cells)
                {
                    continue;
                }
                let mut cells = Vec::with_capacity(protected_cells * protected_cells);
                let mut luma_sum = 0.0;
                let mut luma_square_sum = 0.0;
                let mut source_station = None;
                let mut valid = true;
                let mut individual_sacrifice = 0u64;
                let mut incremental_cost = 0.0;
                for y in top..top + protected_cells {
                    for x in left..left + protected_cells {
                        let index = y * columns + x;
                        let Some(sample) = candidate(&input.cells[index], source) else {
                            valid = false;
                            break;
                        };
                        if components[index] != component {
                            valid = false;
                            break;
                        }
                        if source_station.is_some_and(|station| station != sample.station) {
                            valid = false;
                            break;
                        }
                        source_station = Some(sample.station);
                        // Moving a coherent region across a station boundary is
                        // forbidden when the current accepted seam's RGB/stroke
                        // evidence says this source is worse at any represented
                        // boundary cell.
                        if sample.station != input.cells[index].production_station
                            && production_candidate(&input.cells[index]).is_some_and(|current| {
                                sample.seam_cost > current.seam_cost + 1.0e-12
                            })
                        {
                            valid = false;
                            break;
                        }
                        if y < top + evidence_cells && x < left + evidence_cells {
                            luma_sum += sample.low_frequency_luma;
                            luma_square_sum += sample.low_frequency_luma.powi(2);
                        }
                        individual_sacrifice = individual_sacrifice.saturating_add(sacrifice(
                            &input.cells[index],
                            source,
                            input.shortfall_limit,
                        ));
                        let baseline = production_candidate(&input.cells[index])
                            .map_or(0.0, |current| objective_cost(&input.cells[index], current));
                        incremental_cost += objective_cost(&input.cells[index], sample) - baseline;
                        cells.push(index);
                    }
                    if !valid {
                        break;
                    }
                }
                let evidence_count = evidence_cells.saturating_mul(evidence_cells).max(1) as f64;
                let mean = luma_sum / evidence_count;
                let deviation = (luma_square_sum / evidence_count - mean * mean)
                    .max(0.0)
                    .sqrt();
                if !valid || deviation > 0.02 {
                    continue;
                }
                options.push(RegionOption {
                    source,
                    station: source_station.unwrap_or(0),
                    origin: (left as u32, top as u32),
                    cells,
                    individual_sacrifice,
                    incremental_cost,
                });
            }
        }
    }
    options.sort_by(|left, right| {
        left.individual_sacrifice
            .cmp(&right.individual_sacrifice)
            .then_with(|| left.incremental_cost.total_cmp(&right.incremental_cost))
            .then_with(|| left.origin.1.cmp(&right.origin.1))
            .then_with(|| left.origin.0.cmp(&right.origin.0))
            .then_with(|| left.source.cmp(&right.source))
            .then_with(|| left.station.cmp(&right.station))
    });
    options.dedup_by(|left, right| {
        left.source == right.source && left.origin == right.origin && left.cells == right.cells
    });
    options
}

pub(super) fn overlap(left: &RegionOption, right: &RegionOption) -> bool {
    let right_cells = right.cells.iter().copied().collect::<BTreeSet<_>>();
    left.cells.iter().any(|cell| right_cells.contains(cell))
}

pub(super) fn compatible(selected: &[usize], options: &[RegionOption]) -> bool {
    selected.iter().enumerate().all(|(position, &left)| {
        selected[position + 1..].iter().all(|&right| {
            options[left].origin != options[right].origin
                && !overlap(&options[left], &options[right])
        })
    })
}

pub(super) fn labels_for(
    input: &OracleInput,
    selected: &[usize],
    options: &[RegionOption],
) -> Option<Vec<u64>> {
    if !compatible(selected, options) {
        return None;
    }
    let mut forced = vec![0u64; input.cells.len()];
    for &option in selected {
        for &cell in &options[option].cells {
            if forced[cell] != 0 && forced[cell] != options[option].source {
                return None;
            }
            forced[cell] = options[option].source;
        }
    }
    Some(
        input
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                if forced[index] != 0 {
                    forced[index]
                } else {
                    // This is an objective minimum, not a hard sharpness-winner
                    // terminal; data and accepted seam costs remain free to keep
                    // today's owner.
                    unconstrained_label(cell)
                }
            })
            .collect(),
    )
}

pub(super) fn labeling_sacrifice(input: &OracleInput, labels: &[u64]) -> u64 {
    input
        .cells
        .iter()
        .zip(labels)
        .map(|(cell, &source)| sacrifice(cell, source, input.shortfall_limit))
        .sum()
}

pub(super) fn labeling_objective(input: &OracleInput, labels: &[u64]) -> f64 {
    input
        .cells
        .iter()
        .zip(labels)
        .map(|(cell, &source)| {
            if source == 0 || cell.candidates.is_empty() {
                0.0
            } else {
                candidate(cell, source).map_or(0.0, |sample| objective_cost(cell, sample))
            }
        })
        .sum()
}

pub(super) fn relaxation_lower_bound(options: &[RegionOption], count: usize) -> Option<u64> {
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

pub(super) fn reduced_exact(
    input: &OracleInput,
    options: &[RegionOption],
    maximum_options: usize,
) -> Option<(u64, f64, Vec<usize>, Vec<u64>, usize)> {
    let count = input.required_regions;
    if count == 0 {
        let labels = labels_for(input, &[], options)?;
        return Some((
            labeling_sacrifice(input, &labels),
            labeling_objective(input, &labels),
            Vec::new(),
            labels,
            0,
        ));
    }
    let mut ranked_all = (0..options.len()).collect::<Vec<_>>();
    ranked_all.sort_by(|&left, &right| {
        options[left]
            .individual_sacrifice
            .cmp(&options[right].individual_sacrifice)
            .then_with(|| {
                options[left]
                    .incremental_cost
                    .total_cmp(&options[right].incremental_cost)
            })
            .then_with(|| options[left].origin.1.cmp(&options[right].origin.1))
            .then_with(|| options[left].origin.0.cmp(&options[right].origin.0))
            .then_with(|| options[left].source.cmp(&options[right].source))
    });
    // Repeated translations of one broad component must not consume the
    // reduced pool. Keep the lowest-cost pairwise-disjoint options first; this
    // is only the feasible-search pool, while the independent relaxation above
    // still uses every genuine option.
    let mut ranked = Vec::new();
    for index in ranked_all {
        if ranked.len() == maximum_options.max(count) {
            break;
        }
        if ranked
            .iter()
            .all(|&current| !overlap(&options[index], &options[current]))
        {
            ranked.push(index);
        }
    }
    let base_labels = labels_for(input, &[], options)?;
    let deltas = ranked
        .iter()
        .map(|&option_index| {
            options[option_index].cells.iter().fold(
                (0i128, 0.0f64),
                |(sacrifice_delta, objective_delta), &cell| {
                    let old_source = base_labels[cell];
                    let new_source = options[option_index].source;
                    let old_sacrifice =
                        sacrifice(&input.cells[cell], old_source, input.shortfall_limit);
                    let new_sacrifice =
                        sacrifice(&input.cells[cell], new_source, input.shortfall_limit);
                    let old_objective = candidate(&input.cells[cell], old_source)
                        .map_or(f64::INFINITY, |sample| {
                            objective_cost(&input.cells[cell], sample)
                        });
                    let new_objective = candidate(&input.cells[cell], new_source)
                        .map_or(f64::INFINITY, |sample| {
                            objective_cost(&input.cells[cell], sample)
                        });
                    (
                        sacrifice_delta + i128::from(new_sacrifice) - i128::from(old_sacrifice),
                        objective_delta + new_objective - old_objective,
                    )
                },
            )
        })
        .collect::<Vec<_>>();
    let mut best = None::<(i128, f64, Vec<usize>)>;
    fn visit(
        input: &OracleInput,
        options: &[RegionOption],
        ranked: &[usize],
        deltas: &[(i128, f64)],
        start: usize,
        selected: &mut Vec<usize>,
        sacrifice_delta: i128,
        objective_delta: f64,
        best: &mut Option<(i128, f64, Vec<usize>)>,
    ) {
        if selected.len() == input.required_regions {
            let better = best.as_ref().is_none_or(|current| {
                sacrifice_delta < current.0
                    || (sacrifice_delta == current.0
                        && (objective_delta < current.1
                            || (objective_delta == current.1
                                && selected.as_slice() < current.2.as_slice())))
            });
            if better {
                *best = Some((sacrifice_delta, objective_delta, selected.clone()));
            }
            return;
        }
        let needed = input.required_regions - selected.len();
        if ranked.len().saturating_sub(start) < needed {
            return;
        }
        // Independent disjoint-region deltas provide a valid branch bound:
        // no completion can improve on the sum of the `needed` smallest
        // remaining sharpness deltas.
        if let Some(current) = best.as_ref() {
            let mut remaining = deltas[start..]
                .iter()
                .map(|delta| delta.0)
                .collect::<Vec<_>>();
            remaining.sort_unstable();
            let lower = remaining.into_iter().take(needed).sum::<i128>();
            if sacrifice_delta + lower > current.0 {
                return;
            }
        }
        for cursor in start..=ranked.len() - needed {
            selected.push(ranked[cursor]);
            if compatible(selected, options) {
                visit(
                    input,
                    options,
                    ranked,
                    deltas,
                    cursor + 1,
                    selected,
                    sacrifice_delta + deltas[cursor].0,
                    objective_delta + deltas[cursor].1,
                    best,
                );
            }
            selected.pop();
        }
    }
    visit(
        input,
        options,
        &ranked,
        &deltas,
        0,
        &mut Vec::new(),
        0,
        0.0,
        &mut best,
    );
    best.and_then(|(_, _, selected)| {
        let labels = labels_for(input, &selected, options)?;
        Some((
            labeling_sacrifice(input, &labels),
            labeling_objective(input, &labels),
            selected,
            labels,
            ranked.len(),
        ))
    })
}

pub(in crate::panorama_utils::stitching) fn solve(input: &OracleInput) -> OracleReport {
    assert_eq!(input.cells.len(), input.geometry.cell_count());
    let options = region_options(input);
    let textured_pixels = input
        .cells
        .iter()
        .map(|cell| cell.textured_pixels)
        .sum::<u64>();
    let coverage = |sacrifice: u64| {
        if textured_pixels == 0 {
            0.0
        } else {
            1.0 - sacrifice as f64 / textured_pixels as f64
        }
    };
    let current_sacrifice = input
        .cells
        .iter()
        .map(|cell| sacrifice(cell, cell.production_source, input.shortfall_limit))
        .sum();
    let lower = relaxation_lower_bound(&options, input.required_regions).unwrap_or(textured_pixels);
    const REDUCED_OPTION_LIMIT: usize = 24;
    let exact = reduced_exact(input, &options, REDUCED_OPTION_LIMIT);
    let distinct_region_origins = options
        .iter()
        .map(|option| option.origin)
        .collect::<BTreeSet<_>>()
        .len();
    let (feasible_sacrifice, objective, selected, labels, reduced_options) = exact.map_or(
        (
            None,
            None,
            Vec::new(),
            None,
            options.len().min(REDUCED_OPTION_LIMIT),
        ),
        |(sacrifice, objective, selected, labels, reduced)| {
            (
                Some(sacrifice),
                Some(objective),
                selected,
                Some(labels),
                reduced,
            )
        },
    );
    let cell_side = i64::from(input.geometry.cell_side_px);
    let feasible_coverage = feasible_sacrifice.map(coverage);
    let selected_regions: Vec<SelectedRegion> = selected
        .iter()
        .map(|&index| {
            let option = &options[index];
            SelectedRegion {
                source: option.source,
                station: option.station,
                local_origin: [option.origin.0, option.origin.1],
                world_origin: [
                    input.geometry.world_origin.0 + i64::from(option.origin.0) * cell_side,
                    input.geometry.world_origin.1 + i64::from(option.origin.1) * cell_side,
                ],
                protected_side_cells: input
                    .geometry
                    .regions
                    .protected_span_px()
                    .div_ceil(input.geometry.cell_side_px.max(1)),
                evidence_side_px: input.geometry.regions.evidence_span_px(),
                protected_side_px: input.geometry.regions.protected_span_px(),
            }
        })
        .collect();
    OracleReport {
        feasible: feasible_coverage.is_some_and(|value| value >= 0.99)
            && selected_regions.len() == input.required_regions,
        required_regions: input.required_regions,
        genuine_region_options: options.len(),
        distinct_region_origins,
        reduced_options,
        textured_pixels,
        current_sharpness_coverage: coverage(current_sacrifice),
        relaxation_sharpness_coverage_upper_bound: coverage(lower),
        feasible_sharpness_coverage: feasible_coverage,
        objective_cost: objective,
        selected_regions,
        labels,
        proof_scope: "final-world cells only; region origins come from current source coverage and low-frequency connected components; explicit labels are feasible lower evidence, while the distinct-origin order statistic is an independent conservative sharpness upper bound".to_string(),
    }
}
