//! Final-world cells composed from station candidates.
//!
//! Composes the projected station candidates of every final cell, and holds the
//! per-cell candidate, sharpness-shortfall and objective-cost helpers that the
//! region and exact solvers share.

use super::{CandidateSample, WorldCell};

/// One station candidate projected through the production final-world map.
/// Tests use this adapter to prove station overlap is composed before solving;
/// the real caller builds the same records while reverse-mapping the canvas.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::panorama_utils::stitching) struct ProjectedStationCell {
    pub final_cell: usize,
    pub station: u16,
    pub selected_by_compositor: bool,
    pub samples: Vec<CandidateSample>,
}

pub(super) fn canonicalize_samples(samples: &mut Vec<CandidateSample>) {
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
pub(in crate::panorama_utils::stitching) fn compose_station_cells(
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

pub(super) fn candidate(cell: &WorldCell, source: u64) -> Option<&CandidateSample> {
    cell.candidates
        .iter()
        .filter(|sample| sample.source == source && sample.fully_covered)
        .min_by(|left, right| {
            left.data_cost
                .total_cmp(&right.data_cost)
                .then_with(|| left.station.cmp(&right.station))
        })
}

pub(super) fn sharpest(cell: &WorldCell) -> Option<&CandidateSample> {
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

pub(super) fn shortfall(cell: &WorldCell, source: u64) -> f64 {
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

pub(super) fn sacrifice(cell: &WorldCell, source: u64, limit: f64) -> u64 {
    if shortfall(cell, source) > limit {
        cell.textured_pixels
    } else {
        0
    }
}

pub(super) fn production_candidate(cell: &WorldCell) -> Option<&CandidateSample> {
    candidate(cell, cell.production_source)
}

/// The reduced objective is current-run data fidelity plus the accepted
/// station seam cost. A tiny deterministic stay cost resolves exact ties in
/// favour of today's output; it is not a hard terminal and cannot defeat a
/// measurable data/seam improvement.
pub(super) fn objective_cost(cell: &WorldCell, sample: &CandidateSample) -> f64 {
    const OWNER_CHANGE_TIE_COST: f64 = 1.0e-9;
    sample.data_cost
        + sample.seam_cost
        + if sample.source == cell.production_source {
            0.0
        } else {
            OWNER_CHANGE_TIE_COST
        }
}

pub(super) fn unconstrained_label(cell: &WorldCell) -> u64 {
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
