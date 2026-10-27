//! Test-only final-world Source_RAW ownership feasibility oracle.
//!
//! The caller composes current-run station candidates into [`WorldCell`]s by
//! using the production pose, residual warp, coverage and final sampling
//! origin. This module deliberately knows nothing about Quality-Gate ROI lists,
//! reports, fixture names or prior runs: coherent-region origins are derived
//! only from each source's coverage and low-frequency content mask.

use nalgebra::Matrix3;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Mutex;

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

/// One station candidate projected through the production final-world map.
/// Tests use this adapter to prove station overlap is composed before solving;
/// the real caller builds the same records while reverse-mapping the canvas.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ProjectedStationCell {
    pub final_cell: usize,
    pub station: u16,
    pub selected_by_compositor: bool,
    pub samples: Vec<CandidateSample>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct StationCandidateCell {
    /// The production evidence grid stores these values as `f32`; retaining
    /// that representation here makes owner-shortfall replay bit-identical.
    pub sharpness: f32,
    pub data_cost: f64,
    pub disagreement: f32,
    /// Candidate supplied the production ownership-cell measurement.
    pub covered: bool,
    /// Every source pixel represented by the ownership cell is covered; this
    /// stricter flag gates coherent-region eligibility.
    pub fully_covered: bool,
    pub flat: bool,
    pub low_frequency_luma: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct StationSourceSnapshot {
    pub source: String,
    pub local_owner: u16,
    pub source_to_station: Matrix3<f64>,
    pub cells: Vec<StationCandidateCell>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct StationSnapshot {
    pub station_index: usize,
    pub dimensions: (u32, u32),
    pub columns: u32,
    pub rows: u32,
    pub cell_side_px: u32,
    pub production_owners: Vec<u16>,
    pub sources: Vec<StationSourceSnapshot>,
}

static STATION_SNAPSHOTS: Mutex<Vec<StationSnapshot>> = Mutex::new(Vec::new());

pub(super) fn reset_station_snapshots() {
    STATION_SNAPSHOTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

pub(super) fn record_station_snapshot(snapshot: StationSnapshot) {
    let mut snapshots = STATION_SNAPSHOTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match snapshots
        .iter_mut()
        .find(|current| current.station_index == snapshot.station_index)
    {
        Some(current) => *current = snapshot,
        None => snapshots.push(snapshot),
    }
}

pub(super) fn station_snapshots() -> Vec<StationSnapshot> {
    let mut snapshots = STATION_SNAPSHOTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    snapshots.sort_by_key(|snapshot| snapshot.station_index);
    snapshots
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SelectedRegion {
    pub source: u64,
    pub station: u16,
    pub local_origin: [u32; 2],
    pub world_origin: [i64; 2],
    pub protected_side_cells: u32,
    pub evidence_side_px: u32,
    pub protected_side_px: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OracleReport {
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
struct RegionOption {
    source: u64,
    station: u16,
    origin: (u32, u32),
    cells: Vec<usize>,
    individual_sacrifice: u64,
    incremental_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(super) struct ExactRegionProposal {
    pub source: u64,
    pub station: u16,
    pub left: u32,
    pub top: u32,
    pub side: u32,
    /// Ranking hint only. Exact feasibility and costs never consume it.
    pub coarse_current_shortfall_pixels: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(super) struct ExactRegionEvaluation {
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
pub(super) struct ExactRegionSolveReport {
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

fn exact_regions_overlap(left: ExactRegionProposal, right: ExactRegionProposal) -> bool {
    left.left < right.left.saturating_add(right.side)
        && right.left < left.left.saturating_add(left.side)
        && left.top < right.top.saturating_add(right.side)
        && right.top < left.top.saturating_add(left.side)
}

/// Lift generic mask/content origins into final-pixel rectangles. The coarse
/// grid proposes origins only; every pixel is independently revalidated by the
/// exact evaluator before an option can enter the solver.
pub(super) fn exact_region_proposals(input: &OracleInput) -> Vec<ExactRegionProposal> {
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
pub(super) fn solve_exact_regions(
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

fn canonicalize_samples(samples: &mut Vec<CandidateSample>) {
    samples.retain(|sample| {
        sample.source != 0
            && sample.sharpness.is_finite()
            && sample.data_cost.is_finite()
            && sample.seam_cost.is_finite()
            && sample.source_x.is_finite()
            && sample.source_y.is_finite()
            && sample.low_frequency_luma.is_finite()
    });
    samples.sort_by(|left, right| {
        left.source
            .cmp(&right.source)
            .then_with(|| left.station.cmp(&right.station))
            .then_with(|| left.data_cost.total_cmp(&right.data_cost))
            .then_with(|| right.sharpness.total_cmp(&left.sharpness))
            .then_with(|| left.source_y.total_cmp(&right.source_y))
            .then_with(|| left.source_x.total_cmp(&right.source_x))
    });
    samples.dedup_by(|left, right| left.source == right.source && left.station == right.station);
}

/// Compose station candidates in the exact final-cell decision domain.
/// Exactly one station must be selected by the current compositor at each
/// covered cell; all overlapping station/source candidates remain available.
pub(super) fn compose_station_cells(
    cell_count: usize,
    textured_pixels: &[u64],
    projected: &[ProjectedStationCell],
) -> Option<Vec<WorldCell>> {
    if textured_pixels.len() != cell_count {
        return None;
    }
    let mut grouped = vec![Vec::<&ProjectedStationCell>::new(); cell_count];
    for item in projected {
        grouped.get_mut(item.final_cell)?.push(item);
    }
    let mut cells = Vec::with_capacity(cell_count);
    for (index, mut station_cells) in grouped.into_iter().enumerate() {
        station_cells.sort_by_key(|item| item.station);
        let selected = station_cells
            .iter()
            .filter(|item| item.selected_by_compositor)
            .copied()
            .collect::<Vec<_>>();
        if station_cells.is_empty() {
            cells.push(WorldCell {
                production_source: 0,
                production_station: 0,
                textured_pixels: textured_pixels[index],
                candidates: Vec::new(),
            });
            continue;
        }
        if selected.len() != 1 {
            return None;
        }
        let selected = selected[0];
        let mut selected_samples = selected.samples.clone();
        canonicalize_samples(&mut selected_samples);
        let production = selected_samples.iter().min_by(|left, right| {
            left.data_cost
                .total_cmp(&right.data_cost)
                .then_with(|| left.source.cmp(&right.source))
        })?;
        let production_source = production.source;
        let production_station = selected.station;
        let mut candidates = station_cells
            .into_iter()
            .flat_map(|item| item.samples.clone())
            .collect::<Vec<_>>();
        canonicalize_samples(&mut candidates);
        cells.push(WorldCell {
            production_source,
            production_station,
            textured_pixels: textured_pixels[index],
            candidates,
        });
    }
    Some(cells)
}

fn candidate(cell: &WorldCell, source: u64) -> Option<&CandidateSample> {
    cell.candidates
        .iter()
        .filter(|sample| sample.source == source && sample.fully_covered)
        .min_by(|left, right| {
            left.data_cost
                .total_cmp(&right.data_cost)
                .then_with(|| left.station.cmp(&right.station))
        })
}

fn sharpest(cell: &WorldCell) -> Option<&CandidateSample> {
    cell.candidates
        .iter()
        .filter(|sample| sample.fully_covered)
        .max_by(|left, right| {
            left.sharpness
                .total_cmp(&right.sharpness)
                .then_with(|| right.data_cost.total_cmp(&left.data_cost))
                .then_with(|| right.source.cmp(&left.source))
        })
}

fn shortfall(cell: &WorldCell, source: u64) -> f64 {
    let Some(best) = sharpest(cell) else {
        return 0.0;
    };
    let available = cell
        .candidates
        .iter()
        .filter(|sample| sample.fully_covered)
        .count();
    if available <= 1 || best.sharpness <= 0.0 {
        return 0.0;
    }
    let score = candidate(cell, source).map_or(0.0, |sample| sample.sharpness);
    ((best.sharpness - score) / best.sharpness).clamp(0.0, 1.0)
}

fn sacrifice(cell: &WorldCell, source: u64, limit: f64) -> u64 {
    if shortfall(cell, source) > limit {
        cell.textured_pixels
    } else {
        0
    }
}

fn production_candidate(cell: &WorldCell) -> Option<&CandidateSample> {
    candidate(cell, cell.production_source)
}

/// The reduced objective is current-run data fidelity plus the accepted
/// station seam cost. A tiny deterministic stay cost resolves exact ties in
/// favour of today's output; it is not a hard terminal and cannot defeat a
/// measurable data/seam improvement.
fn objective_cost(cell: &WorldCell, sample: &CandidateSample) -> f64 {
    const OWNER_CHANGE_TIE_COST: f64 = 1.0e-9;
    sample.data_cost
        + sample.seam_cost
        + if sample.source == cell.production_source {
            0.0
        } else {
            OWNER_CHANGE_TIE_COST
        }
}

fn unconstrained_label(cell: &WorldCell) -> u64 {
    cell.candidates
        .iter()
        .filter(|sample| sample.fully_covered)
        .min_by(|left, right| {
            objective_cost(cell, left)
                .total_cmp(&objective_cost(cell, right))
                .then_with(|| left.source.cmp(&right.source))
        })
        .map_or(cell.production_source, |sample| sample.source)
}

fn neighbour_indices(index: usize, columns: usize, rows: usize) -> [Option<usize>; 4] {
    let (x, y) = (index % columns, index / columns);
    [
        (x > 0).then(|| index - 1),
        (x + 1 < columns).then(|| index + 1),
        (y > 0).then(|| index - columns),
        (y + 1 < rows).then(|| index + columns),
    ]
}

fn source_component_map(input: &OracleInput, source: u64) -> (Vec<usize>, Vec<usize>) {
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

fn region_options(input: &OracleInput) -> Vec<RegionOption> {
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

fn overlap(left: &RegionOption, right: &RegionOption) -> bool {
    let right_cells = right.cells.iter().copied().collect::<BTreeSet<_>>();
    left.cells.iter().any(|cell| right_cells.contains(cell))
}

fn compatible(selected: &[usize], options: &[RegionOption]) -> bool {
    selected.iter().enumerate().all(|(position, &left)| {
        selected[position + 1..].iter().all(|&right| {
            options[left].origin != options[right].origin
                && !overlap(&options[left], &options[right])
        })
    })
}

fn labels_for(
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

fn labeling_sacrifice(input: &OracleInput, labels: &[u64]) -> u64 {
    input
        .cells
        .iter()
        .zip(labels)
        .map(|(cell, &source)| sacrifice(cell, source, input.shortfall_limit))
        .sum()
}

fn labeling_objective(input: &OracleInput, labels: &[u64]) -> f64 {
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

fn relaxation_lower_bound(options: &[RegionOption], count: usize) -> Option<u64> {
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

fn reduced_exact(
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

pub(super) fn solve(input: &OracleInput) -> OracleReport {
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

/// Exact production decision inputs for one real Source_RAW at one final pixel.
/// Integer/bit fields make run equality independent of host floating-point
/// formatting. `station_cell` is the exact ownership cell reached after the
/// final-world, residual, source and source-to-station maps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(super) struct ExactCandidateDecision {
    pub source: u64,
    pub station: u16,
    pub station_cell: u32,
    pub sharpness_bits: u32,
    pub winner_sharpness_bits: u32,
    pub shortfall: u8,
    pub disagreement: u8,
    pub data_cost_bits: u64,
    pub pairwise_cost_bits: u64,
    pub seam_boundary: bool,
}

impl ExactCandidateDecision {
    fn data_cost(&self) -> f64 {
        f64::from_bits(self.data_cost_bits)
    }

    fn pairwise_cost(&self) -> f64 {
        f64::from_bits(self.pairwise_cost_bits)
    }
}

/// Exact source sample provenance retained separately from the decision key.
/// A run may share one decision while its projective source coordinates vary;
/// every represented final pixel still retains its own source sample bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(super) struct ExactSourceSample {
    pub source: u64,
    pub x_bits: u64,
    pub y_bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ExactPixelEvidence {
    pub x: u32,
    pub y: u32,
    pub covered: bool,
    pub textured: bool,
    pub production_source: u64,
    pub production_station: u16,
    pub production_shortfall: u8,
    pub production_disagreement: u8,
    pub candidates: Vec<ExactCandidateDecision>,
    pub source_samples: Vec<ExactSourceSample>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(super) struct ExactRun {
    pub y: u32,
    pub x_start: u32,
    pub len: u32,
    pub covered: bool,
    pub textured: bool,
    pub production_source: u64,
    pub production_station: u16,
    pub production_shortfall: u8,
    pub production_disagreement: u8,
    pub candidates: Vec<ExactCandidateDecision>,
    /// Row-major, then candidate order. Its length is `len * candidates.len()`.
    pub source_samples: Vec<ExactSourceSample>,
}

impl ExactRun {
    fn same_decision(&self, pixel: &ExactPixelEvidence) -> bool {
        self.y == pixel.y
            && self.x_start.saturating_add(self.len) == pixel.x
            && self.covered == pixel.covered
            && self.textured == pixel.textured
            && self.production_source == pixel.production_source
            && self.production_station == pixel.production_station
            && self.production_shortfall == pixel.production_shortfall
            && self.production_disagreement == pixel.production_disagreement
            && self.candidates == pixel.candidates
    }
}

/// Build horizontal sparse runs without centre inference. Pixels merge only
/// after all production owner inputs and all station/source candidate
/// provenance compare equal; exact source coordinates remain attached per
/// represented pixel.
pub(super) fn aggregate_exact_runs(
    width: u32,
    pixels: impl IntoIterator<Item = ExactPixelEvidence>,
) -> Result<Vec<ExactRun>, &'static str> {
    let mut runs = Vec::<ExactRun>::new();
    let mut previous_index = None;
    for mut pixel in pixels {
        if pixel.x >= width || pixel.candidates.len() != pixel.source_samples.len() {
            return Err("invalid exact owner pixel");
        }
        pixel
            .candidates
            .sort_by_key(|candidate| (candidate.station, candidate.source));
        pixel.source_samples.sort_by_key(|sample| sample.source);
        if !pixel
            .candidates
            .iter()
            .zip(&pixel.source_samples)
            .all(|(candidate, sample)| candidate.source == sample.source)
        {
            return Err("exact owner candidate provenance mismatch");
        }
        let index = u64::from(pixel.y)
            .checked_mul(u64::from(width))
            .and_then(|row| row.checked_add(u64::from(pixel.x)))
            .ok_or("invalid exact owner pixel")?;
        if previous_index.is_some_and(|previous| index <= previous) {
            return Err("exact owner pixels must be unique row-major samples");
        }
        previous_index = Some(index);
        if let Some(run) = runs.last_mut().filter(|run| run.same_decision(&pixel)) {
            run.len = run.len.saturating_add(1);
            run.source_samples.extend(pixel.source_samples);
            continue;
        }
        runs.push(ExactRun {
            y: pixel.y,
            x_start: pixel.x,
            len: 1,
            covered: pixel.covered,
            textured: pixel.textured,
            production_source: pixel.production_source,
            production_station: pixel.production_station,
            production_shortfall: pixel.production_shortfall,
            production_disagreement: pixel.production_disagreement,
            candidates: pixel.candidates,
            source_samples: pixel.source_samples,
        });
    }
    Ok(runs)
}

pub(super) fn exact_run_stats_data_min(runs: &[ExactRun], shortfall_limit: u8) -> ExactOwnerStats {
    let mut result = ExactOwnerStats::default();
    for run in runs {
        if !run.covered {
            continue;
        }
        let selected = run.candidates.iter().min_by(|left, right| {
            left.data_cost()
                .total_cmp(&right.data_cost())
                .then_with(|| left.source.cmp(&right.source))
        });
        let Some(selected) = selected else {
            result.unresolved_pixels = result.unresolved_pixels.saturating_add(u64::from(run.len));
            continue;
        };
        result.covered_pixels = result.covered_pixels.saturating_add(u64::from(run.len));
        if run.textured {
            result.textured_pixels = result.textured_pixels.saturating_add(u64::from(run.len));
            if selected.shortfall > shortfall_limit {
                result.shortfall_pixels =
                    result.shortfall_pixels.saturating_add(u64::from(run.len));
            }
        }
        result.data_cost += selected.data_cost() * f64::from(run.len);
        if selected.seam_boundary {
            result.pairwise_cost += selected.pairwise_cost() * f64::from(run.len);
        }
    }
    result
}

pub(super) fn exact_run_stats_for_source(
    runs: &[ExactRun],
    source: u64,
    shortfall_limit: u8,
) -> ExactOwnerStats {
    let mut result = ExactOwnerStats::default();
    for run in runs {
        if !run.covered {
            continue;
        }
        let selected = run
            .candidates
            .iter()
            .find(|candidate| candidate.source == source);
        let Some(selected) = selected else {
            result.unresolved_pixels = result.unresolved_pixels.saturating_add(u64::from(run.len));
            continue;
        };
        result.covered_pixels = result.covered_pixels.saturating_add(u64::from(run.len));
        if run.textured {
            result.textured_pixels = result.textured_pixels.saturating_add(u64::from(run.len));
            if selected.shortfall > shortfall_limit {
                result.shortfall_pixels =
                    result.shortfall_pixels.saturating_add(u64::from(run.len));
            }
        }
        result.data_cost += selected.data_cost() * f64::from(run.len);
        if selected.seam_boundary {
            result.pairwise_cost += selected.pairwise_cost() * f64::from(run.len);
        }
    }
    result
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(super) struct ExactOwnerStats {
    pub covered_pixels: u64,
    pub textured_pixels: u64,
    pub shortfall_pixels: u64,
    pub unresolved_pixels: u64,
    pub data_cost: f64,
    pub pairwise_cost: f64,
}

impl ExactOwnerStats {
    pub fn coverage(self) -> Option<f64> {
        (self.textured_pixels != 0)
            .then_some(1.0 - self.shortfall_pixels as f64 / self.textured_pixels as f64)
    }
}

/// Sum exact run evidence for the current labeling or an arbitrary row-major
/// synthetic labeling. This is intentionally independent of image replay;
/// the caller compares it with the unchanged Quality-Gate planes afterward.
pub(super) fn exact_run_stats(
    width: u32,
    runs: &[ExactRun],
    labels: Option<&[u64]>,
    shortfall_limit: u8,
) -> ExactOwnerStats {
    let mut result = ExactOwnerStats::default();
    for run in runs {
        for offset in 0..run.len {
            if !run.covered {
                continue;
            }
            result.covered_pixels = result.covered_pixels.saturating_add(1);
            let index = run.y as usize * width as usize + (run.x_start + offset) as usize;
            let source = labels
                .and_then(|labels| labels.get(index).copied())
                .unwrap_or(run.production_source);
            let selected = run
                .candidates
                .iter()
                .find(|candidate| candidate.source == source);
            let shortfall = if source == run.production_source {
                run.production_shortfall
            } else {
                selected.map_or(u8::MAX, |candidate| candidate.shortfall)
            };
            if selected.is_none() || shortfall == u8::MAX {
                result.unresolved_pixels = result.unresolved_pixels.saturating_add(1);
                continue;
            }
            if run.textured {
                result.textured_pixels = result.textured_pixels.saturating_add(1);
                result.shortfall_pixels = result
                    .shortfall_pixels
                    .saturating_add(u64::from(shortfall > shortfall_limit));
            }
            if let Some(selected) = selected {
                result.data_cost += selected.data_cost();
                if selected.seam_boundary {
                    result.pairwise_cost += selected.pairwise_cost();
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(source: u64, station: u16, sharpness: f64, cost: f64) -> CandidateSample {
        CandidateSample {
            source,
            station,
            sharpness,
            data_cost: cost,
            seam_cost: 0.0,
            source_x: source as f64 * 10.0,
            source_y: station as f64 * 10.0,
            fully_covered: true,
            flat: true,
            low_frequency_luma: 0.5,
        }
    }

    fn synthetic(columns: u32, rows: u32, required_regions: usize) -> OracleInput {
        let cells = (0..columns * rows)
            .map(|_| {
                let mut incumbent = sample(2, 1, 0.5, 0.5);
                incumbent.flat = false;
                WorldCell {
                    production_source: 2,
                    production_station: 1,
                    textured_pixels: 1,
                    candidates: vec![sample(1, 1, 1.0, 0.0), incumbent],
                }
            })
            .collect();
        OracleInput {
            geometry: WorldGeometry {
                columns,
                rows,
                cell_side_px: 1,
                qg_aligned_region_grid: false,
                world_origin: (100, -200),
                regions: RegionGeometry {
                    roi_side_px: 2,
                    same_owner_ring_px: 0,
                    sampling_phase_px: 1,
                },
            },
            cells,
            required_regions,
            shortfall_limit: 0.05,
        }
    }

    fn brute_force(input: &OracleInput) -> Option<u64> {
        let options = region_options(input);
        let mut best = None;
        fn visit(
            input: &OracleInput,
            options: &[RegionOption],
            start: usize,
            selected: &mut Vec<usize>,
            best: &mut Option<u64>,
        ) {
            if selected.len() == input.required_regions {
                if let Some(labels) = labels_for(input, selected, options) {
                    let cost = labeling_sacrifice(input, &labels);
                    *best = Some(best.map_or(cost, |current: u64| current.min(cost)));
                }
                return;
            }
            for index in start..options.len() {
                selected.push(index);
                visit(input, options, index + 1, selected, best);
                selected.pop();
            }
        }
        visit(input, &options, 0, &mut Vec::new(), &mut best);
        best
    }

    #[test]
    fn candidate_origins_come_only_from_coverage_and_content_components() {
        let mut input = synthetic(8, 8, 1);
        for y in 0..8usize {
            for x in 0..8usize {
                if x >= 4 || y >= 4 {
                    input.cells[y * 8 + x]
                        .candidates
                        .iter_mut()
                        .find(|candidate| candidate.source == 1)
                        .unwrap()
                        .fully_covered = false;
                }
            }
        }
        let report = solve(&input);
        assert!(report.feasible);
        assert!(
            report
                .selected_regions
                .iter()
                .all(|region| { region.local_origin[0] < 4 && region.local_origin[1] < 4 })
        );
        assert_eq!(report.selected_regions[0].source, 1);
    }

    #[test]
    fn source_order_permutation_world_translation_and_thread_counts_are_deterministic() {
        let input = synthetic(12, 12, 4);
        let expected = solve(&input);
        let mut permuted = input.clone();
        for cell in &mut permuted.cells {
            cell.candidates.reverse();
        }
        assert_eq!(expected, solve(&permuted));

        let mut translated = input.clone();
        translated.geometry.world_origin = (4_100, -7_200);
        let shifted = solve(&translated);
        assert_eq!(expected.labels, shifted.labels);
        assert_eq!(expected.objective_cost, shifted.objective_cost);
        assert!(
            expected
                .selected_regions
                .iter()
                .zip(&shifted.selected_regions)
                .all(|(left, right)| {
                    right.world_origin[0] - left.world_origin[0] == 4_000
                        && right.world_origin[1] - left.world_origin[1] == -7_000
                })
        );

        for threads in [1, 2, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let actual = pool.install(|| solve(&input));
            assert_eq!(expected, actual, "{threads} threads");
        }
    }

    #[test]
    fn disconnected_overlap_and_fake_regions_cannot_satisfy_cardinality() {
        let mut input = synthetic(8, 8, 2);
        for (index, cell) in input.cells.iter_mut().enumerate() {
            let (x, y) = (index % 8, index / 8);
            let candidate = cell
                .candidates
                .iter_mut()
                .find(|candidate| candidate.source == 1)
                .unwrap();
            candidate.fully_covered = (x < 2 && y < 2) || (x >= 6 && y >= 6);
        }
        let report = solve(&input);
        assert!(
            report.feasible,
            "two real disconnected components count twice"
        );

        input.required_regions = 3;
        assert!(
            !solve(&input).feasible,
            "duplicates cannot create a third region"
        );

        let mut overlap_only = synthetic(3, 3, 2);
        overlap_only.geometry.regions.sampling_phase_px = 1;
        assert!(
            !solve(&overlap_only).feasible,
            "overlapping windows cannot count twice"
        );
    }

    #[test]
    fn labels_always_retain_one_real_source_and_exact_coordinates() {
        let input = synthetic(8, 8, 1);
        let report = solve(&input);
        let labels = report.labels.unwrap();
        assert!(labels.iter().enumerate().all(|(index, &source)| {
            candidate(&input.cells[index], source)
                .is_some_and(|sample| sample.source_x.is_finite() && sample.source_y.is_finite())
        }));
    }

    #[test]
    fn final_station_overlap_composition_uses_selected_station_but_keeps_all_candidates() {
        let projected = vec![
            ProjectedStationCell {
                final_cell: 0,
                station: 1,
                selected_by_compositor: false,
                samples: vec![sample(11, 1, 1.0, 0.0)],
            },
            ProjectedStationCell {
                final_cell: 0,
                station: 2,
                selected_by_compositor: true,
                samples: vec![sample(22, 2, 0.8, 0.2)],
            },
        ];
        let cells = compose_station_cells(1, &[1], &projected).unwrap();
        assert_eq!(cells[0].production_source, 22);
        assert_eq!(cells[0].production_station, 2);
        assert_eq!(
            cells[0]
                .candidates
                .iter()
                .map(|candidate| candidate.source)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([11, 22])
        );
        let mut invalid = projected.clone();
        invalid[0].selected_by_compositor = true;
        assert!(compose_station_cells(1, &[1], &invalid).is_none());
    }

    #[test]
    fn accepted_seam_cost_rejects_a_sharper_region_that_worsens_strokes() {
        let mut input = synthetic(6, 6, 1);
        for cell in &mut input.cells {
            cell.production_station = 2;
            let current = cell
                .candidates
                .iter_mut()
                .find(|candidate| candidate.source == 2)
                .unwrap();
            current.station = 2;
            current.seam_cost = 0.1;
            let sharper = cell
                .candidates
                .iter_mut()
                .find(|candidate| candidate.source == 1)
                .unwrap();
            sharper.station = 1;
            sharper.seam_cost = 0.2;
        }
        assert!(!solve(&input).feasible);
        for cell in &mut input.cells {
            cell.candidates
                .iter_mut()
                .find(|candidate| candidate.source == 1)
                .unwrap()
                .seam_cost = 0.05;
        }
        assert!(solve(&input).feasible);
    }

    #[test]
    fn small_grid_optimum_and_relaxation_bound_match_independent_bruteforce() {
        for required in 1..=2 {
            let input = synthetic(6, 6, required);
            let options = region_options(&input);
            let brute = brute_force(&input).unwrap();
            let exact = reduced_exact(&input, &options, options.len()).unwrap().0;
            let lower = relaxation_lower_bound(&options, required).unwrap();
            assert_eq!(exact, brute);
            assert!(lower <= brute);
        }
    }

    #[test]
    fn feasible_and_impossible_sharpness_cases_are_reported_without_overstatement() {
        let feasible = solve(&synthetic(12, 12, 4));
        assert!(feasible.feasible);
        assert_eq!(feasible.selected_regions.len(), 4);
        assert!(feasible.feasible_sharpness_coverage.unwrap() >= 0.99);

        let mut impossible = synthetic(12, 12, 4);
        impossible.cells.iter_mut().for_each(|cell| {
            cell.candidates
                .iter_mut()
                .find(|candidate| candidate.source == 1)
                .unwrap()
                .flat = false;
        });
        let report = solve(&impossible);
        assert!(!report.feasible);
        assert!(report.labels.is_none());
    }

    fn exact_candidate(
        source: u64,
        station_cell: u32,
        sharpness: f32,
        winner: f32,
        shortfall: u8,
    ) -> ExactCandidateDecision {
        ExactCandidateDecision {
            source,
            station: 1,
            station_cell,
            sharpness_bits: sharpness.to_bits(),
            winner_sharpness_bits: winner.to_bits(),
            shortfall,
            disagreement: 0,
            data_cost_bits: (source as f64).to_bits(),
            pairwise_cost_bits: 0.0f64.to_bits(),
            seam_boundary: false,
        }
    }

    fn exact_pixel(x: u32, y: u32, station_cell: u32) -> ExactPixelEvidence {
        ExactPixelEvidence {
            x,
            y,
            covered: true,
            textured: true,
            production_source: 2,
            production_station: 1,
            production_shortfall: 10,
            production_disagreement: 0,
            candidates: vec![
                exact_candidate(1, station_cell, 1.0, 1.0, 0),
                exact_candidate(2, station_cell, 0.9, 1.0, 10),
            ],
            source_samples: vec![
                ExactSourceSample {
                    source: 1,
                    x_bits: f64::from(x).to_bits(),
                    y_bits: f64::from(y).to_bits(),
                },
                ExactSourceSample {
                    source: 2,
                    x_bits: (f64::from(x) + 0.25).to_bits(),
                    y_bits: (f64::from(y) + 0.25).to_bits(),
                },
            ],
        }
    }

    #[test]
    fn exact_runs_aggregate_to_bruteforce_owner_metric_for_current_and_arbitrary_labels() {
        let pixels = (0..8).map(|x| exact_pixel(x, 0, 4)).collect::<Vec<_>>();
        let runs = aggregate_exact_runs(8, pixels.clone()).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].len, 8);
        assert_eq!(runs[0].source_samples.len(), 16);

        let current = exact_run_stats(8, &runs, None, 5);
        assert_eq!(current.textured_pixels, 8);
        assert_eq!(current.shortfall_pixels, 8);
        assert_eq!(current.coverage(), Some(0.0));
        let production =
            super::super::super::stack_pipeline::quality_gate_runner::owner_sharpness_stats(
                &[255; 8], &[1; 8], &[10; 8], &[0; 8], 0,
            )
            .unwrap();
        assert_eq!(current.textured_pixels, production.textured_pixels);
        assert_eq!(current.shortfall_pixels, production.textured_shortfall);

        let labels = vec![1, 2, 1, 2, 1, 2, 1, 2];
        let replay = exact_run_stats(8, &runs, Some(&labels), 5);
        let brute_shortfall = pixels
            .iter()
            .zip(&labels)
            .filter(|(pixel, source)| {
                pixel
                    .candidates
                    .iter()
                    .find(|candidate| candidate.source == **source)
                    .is_none_or(|candidate| candidate.shortfall > 5)
            })
            .count() as u64;
        assert_eq!(replay.textured_pixels, 8);
        assert_eq!(replay.shortfall_pixels, brute_shortfall);
        assert_eq!(replay.shortfall_pixels, 4);
    }

    #[test]
    fn exact_runs_split_at_station_transform_coverage_and_quantization_boundaries() {
        let mut pixels = vec![
            exact_pixel(0, 0, 4),
            exact_pixel(1, 0, 4),
            exact_pixel(2, 0, 5),
            exact_pixel(3, 0, 5),
            exact_pixel(4, 0, 5),
        ];
        pixels[3].candidates[1].shortfall = 11;
        pixels[4].covered = false;
        let runs = aggregate_exact_runs(5, pixels).unwrap();
        assert_eq!(
            runs.iter().map(|run| run.len).collect::<Vec<_>>(),
            [2, 1, 1, 1]
        );
        assert_eq!(runs[0].candidates[0].station_cell, 4);
        assert_eq!(runs[1].candidates[0].station_cell, 5);
        assert_eq!(runs[2].candidates[1].shortfall, 11);
        assert!(!runs[3].covered);
    }

    #[test]
    fn exact_runs_preserve_source_provenance_station_seam_and_order_determinism() {
        let mut pixels = (0..4).map(|x| exact_pixel(x, 0, 9)).collect::<Vec<_>>();
        pixels[0].candidates[0].seam_boundary = true;
        pixels[0].candidates[0].pairwise_cost_bits = 0.25f64.to_bits();
        let expected = aggregate_exact_runs(4, pixels.clone()).unwrap();
        assert_eq!(expected[0].len, 1, "seam membership must split a run");
        assert_eq!(expected[0].source_samples[0].source, 1);
        for pixel in &mut pixels {
            pixel.candidates.reverse();
            pixel.source_samples.reverse();
        }
        let permuted = aggregate_exact_runs(4, pixels.clone()).unwrap();
        assert_eq!(expected, permuted);
        for threads in [1, 2, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(
                expected,
                pool.install(|| aggregate_exact_runs(4, pixels.clone()).unwrap()),
                "{threads} threads"
            );
        }
    }

    #[test]
    fn exact_region_relaxation_never_understates_bruteforce_coverage_bound() {
        let evaluations = (0..4)
            .map(|index| ExactRegionEvaluation {
                proposal: ExactRegionProposal {
                    source: 1,
                    station: 1,
                    left: index * 3,
                    top: 0,
                    side: 2,
                    coarse_current_shortfall_pixels: 2,
                },
                valid: true,
                textured_pixels: 2,
                current_shortfall_pixels: 2,
                baseline_shortfall_pixels: 2,
                candidate_shortfall_pixels: index as u64 % 2,
                owner_change_pixels: index as u64,
                boundary_change_pixels: 0,
                data_cost_delta: index as f64,
                pairwise_cost: 0.0,
                sparse_run_count: 1,
                prediction_replay_mismatch_pixels: 0,
            })
            .collect::<Vec<_>>();
        let report = solve_exact_regions(100, 10, 10, 2, &evaluations);
        assert_eq!(report.selected_shortfall_pixels, Some(6));
        assert_eq!(report.selected_sharpness_coverage, Some(0.94));
        assert!(
            report.relaxation_sharpness_coverage_upper_bound
                >= report.selected_sharpness_coverage.unwrap()
        );

        let mut impossible = evaluations.clone();
        impossible
            .iter_mut()
            .skip(1)
            .for_each(|evaluation| evaluation.valid = false);
        let report = solve_exact_regions(100, 10, 10, 2, &impossible);
        assert!(!report.feasible);
        assert!(report.selected_shortfall_pixels.is_none());
    }

    #[test]
    fn approved_qg_aligned_regions_use_544px_footprints_on_the_fixed_256px_grid() {
        let mut input = synthetic(100, 100, 0);
        input.geometry.cell_side_px = 16;
        input.geometry.qg_aligned_region_grid = true;
        input.geometry.regions = RegionGeometry {
            roi_side_px: 512,
            same_owner_ring_px: 16,
            sampling_phase_px: 256,
        };
        let proposals = exact_region_proposals(&input);
        assert!(!proposals.is_empty());
        assert!(proposals.iter().all(|proposal| proposal.side == 544));
        assert!(
            proposals.iter().all(|proposal| {
                (proposal.left + 16) % 256 == 0 && (proposal.top + 16) % 256 == 0
            })
        );
    }

    #[test]
    fn production_geometry_is_544_evidence_and_799_phase_protection() {
        let geometry = RegionGeometry {
            roi_side_px: 512,
            same_owner_ring_px: 16,
            sampling_phase_px: 256,
        };
        assert_eq!(geometry.evidence_span_px(), 544);
        assert_eq!(geometry.protected_span_px(), 799);
    }
}
