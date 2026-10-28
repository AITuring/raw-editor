//! Ownership planning for the planar scan mosaic.
//!
//! One owner per output cell, chosen by the detail each tile delivers in output
//! pixels, with seams priced by local structure through the shared multi-label
//! graph cut.

use super::super::stack_pipeline::focus_fuser::NO_OWNER;
use super::super::stack_pipeline::focus_fuser::OwnershipCostGrid;
use image::GrayImage;
use nalgebra::Point2;
use rayon::prelude::*;
use std::time::Instant;

use super::{
    Canvas, EDGE_OWNER_PENALTY, LOW_GRID, LowBand, OWNER_CELL, OWNER_MARGIN, SHARPNESS_BLOCK,
    SHARPNESS_COST, STRUCTURE_SEAM_WEIGHT, TONE_DEVIATION_COST, TONE_DEVIATION_FLOOR,
    TONE_DEVIATION_SPAN, TileAnalysis, TileWarp, inside_frame, quarter_valid,
};

// ---------------------------------------------------------------------------
// Ownership
// ---------------------------------------------------------------------------

pub(super) struct Ownership {
    pub(super) columns: usize,
    pub(super) rows: usize,
    /// One-based owner per cell, [`NO_OWNER`] where no tile covers the centre.
    pub(super) owners: Vec<u16>,
    /// Cells whose pixels may need a non-owner fallback sample.
    pub(super) fallback: Vec<bool>,
}

impl Ownership {
    pub(super) fn owner(&self, column: isize, row: isize) -> u16 {
        if column < 0 || row < 0 || column as usize >= self.columns || row as usize >= self.rows {
            return NO_OWNER;
        }
        self.owners[row as usize * self.columns + column as usize]
    }
}

pub(super) fn sample_sharpness(analysis: &TileAnalysis, source: Point2<f64>) -> Option<f32> {
    let x = source.x / SHARPNESS_BLOCK as f64 - 0.5;
    let y = source.y / SHARPNESS_BLOCK as f64 - 0.5;
    let x0 = x.floor().clamp(0.0, (analysis.sharp_columns - 1) as f64) as usize;
    let y0 = y.floor().clamp(0.0, (analysis.sharp_rows - 1) as f64) as usize;
    let x1 = (x0 + 1).min(analysis.sharp_columns - 1);
    let y1 = (y0 + 1).min(analysis.sharp_rows - 1);
    let fx = (x - x0 as f64).clamp(0.0, 1.0) as f32;
    let fy = (y - y0 as f64).clamp(0.0, 1.0) as f32;
    let mut sum = 0.0;
    let mut weight = 0.0;
    for (column, row, w) in [
        (x0, y0, (1.0 - fx) * (1.0 - fy)),
        (x1, y0, fx * (1.0 - fy)),
        (x0, y1, (1.0 - fx) * fy),
        (x1, y1, fx * fy),
    ] {
        let value = analysis.sharpness[row * analysis.sharp_columns + column];
        if value.is_finite() && w > 0.0 {
            sum += value * w;
            weight += w;
        }
    }
    (weight > 1e-4).then(|| sum / weight)
}

pub(super) fn sample_gray_bilinear(image: &GrayImage, x: f64, y: f64) -> Option<f32> {
    if x < 0.0 || y < 0.0 || x >= (image.width() - 1) as f64 || y >= (image.height() - 1) as f64 {
        return None;
    }
    let ix = x as u32;
    let iy = y as u32;
    let fx = (x - ix as f64) as f32;
    let fy = (y - iy as f64) as f32;
    let at = |x: u32, y: u32| f32::from(image.get_pixel(x, y)[0]);
    let top = at(ix, iy) * (1.0 - fx) + at(ix + 1, iy) * fx;
    let bottom = at(ix, iy + 1) * (1.0 - fx) + at(ix + 1, iy + 1) * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

/// Mean relative departure of one tile's low band from the consensus over an
/// ownership cell; zero where the tile has no valid low samples there.
pub(super) fn tone_deviation(
    band: &LowBand,
    consensus: &LowBand,
    column: usize,
    row: usize,
) -> f64 {
    let per_cell = (OWNER_CELL / LOW_GRID) as usize;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in row * per_cell..(row + 1) * per_cell {
        for x in column * per_cell..(column + 1) * per_cell {
            if !band.valid_at(x, y) || !consensus.valid_at(x, y) {
                continue;
            }
            let value = band.values[(y - band.y0) * band.columns + (x - band.x0)];
            let reference = consensus.values[y * consensus.columns + x];
            let deviation = (0..3)
                .map(|channel| {
                    (value[channel] - reference[channel]).abs() / reference[channel].max(0.05)
                })
                .fold(0.0f32, f32::max);
            sum += f64::from(deviation);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

pub(super) fn plan_ownership(
    analyses: &[TileAnalysis],
    warps: &[TileWarp],
    canvas: &Canvas,
    bands: &[LowBand],
    consensus: &LowBand,
) -> Ownership {
    let started = Instant::now();
    let columns = canvas.width.div_ceil(OWNER_CELL) as usize;
    let rows = canvas.height.div_ceil(OWNER_CELL) as usize;
    let labels = analyses.len();
    let cell = OWNER_CELL as f64;
    let reach = cell * 0.5 + OWNER_MARGIN;
    // (label, sharpness, eligible) of every tile covering a cell centre, and
    // the local structure that prices a seam through the cell.
    let measurements = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (column, row) = (index % columns, index / columns);
            let centre_x = column as f64 * cell + cell * 0.5 - 0.5;
            let centre_y = row as f64 * cell + cell * 0.5 - 0.5;
            let world = |dx: f64, dy: f64| {
                Point2::new(
                    centre_x + dx - canvas.offset_x,
                    centre_y + dy - canvas.offset_y,
                )
            };
            let mut candidates = Vec::new();
            for (label, (analysis, warp)) in analyses.iter().zip(warps).enumerate() {
                let Some(source) = warp.source(world(0.0, 0.0)) else {
                    continue;
                };
                if !quarter_valid(analysis, source, 0.0) {
                    continue;
                }
                let Some(sharpness) = sample_sharpness(analysis, source) else {
                    continue;
                };
                let eligible = [
                    (-reach, -reach),
                    (reach, -reach),
                    (-reach, reach),
                    (reach, reach),
                ]
                .iter()
                .all(|&(dx, dy)| {
                    warp.source(world(dx, dy)).is_some_and(|point| {
                        inside_frame(analysis, point, 2.0) && quarter_valid(analysis, point, 0.0)
                    })
                });
                let deviation = tone_deviation(&bands[label], consensus, column, row);
                candidates.push((label, sharpness, eligible, deviation));
            }
            // Structure of the sharpest candidate: local contrast of its
            // blurred quarter-resolution luma across the cell.
            let structure = candidates
                .iter()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .map(|&(label, _, _, _)| {
                    let analysis = &analyses[label];
                    let mut values = Vec::with_capacity(36);
                    for sy in 0..6 {
                        for sx in 0..6 {
                            let dx = (sx as f64 + 0.5) / 6.0 * cell - cell * 0.5;
                            let dy = (sy as f64 + 0.5) / 6.0 * cell - cell * 0.5;
                            if let Some(source) = warps[label].source(world(dx, dy))
                                && let Some(value) = sample_gray_bilinear(
                                    &analysis.reg_quarter,
                                    (source.x - 1.5) / 4.0,
                                    (source.y - 1.5) / 4.0,
                                )
                            {
                                values.push(value);
                            }
                        }
                    }
                    if values.len() < 8 {
                        return 0.0;
                    }
                    let mean = values.iter().sum::<f32>() / values.len() as f32;
                    (values
                        .iter()
                        .map(|value| (value - mean).powi(2))
                        .sum::<f32>()
                        / values.len() as f32)
                        .sqrt()
                })
                .unwrap_or(0.0);
            (candidates, structure)
        })
        .collect::<Vec<_>>();

    let mut structures = measurements
        .iter()
        .filter(|(candidates, _)| !candidates.is_empty())
        .map(|(_, structure)| f64::from(*structure))
        .collect::<Vec<_>>();
    structures.sort_by(f64::total_cmp);
    let quantile = |fraction: f64| {
        structures
            .get(((structures.len().max(1) - 1) as f64 * fraction).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let (structure_floor, structure_ceiling) = (quantile(0.5), quantile(0.95));
    let structure_span = (structure_ceiling - structure_floor).max(1e-6);

    let mut grid = OwnershipCostGrid::new(columns, rows, labels);
    let mut eligible_owner = vec![false; columns * rows * labels];
    for (index, (candidates, structure)) in measurements.iter().enumerate() {
        let best = candidates
            .iter()
            .map(|candidate| candidate.1)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        for &(label, sharpness, eligible, deviation) in candidates {
            let mut cost = SHARPNESS_COST * (1.0 - f64::from(sharpness / best)).clamp(0.0, 1.0);
            cost += TONE_DEVIATION_COST
                * ((deviation - TONE_DEVIATION_FLOOR) / TONE_DEVIATION_SPAN).clamp(0.0, 1.0);
            if !eligible {
                cost += EDGE_OWNER_PENALTY;
            }
            grid.set_cost(index, label, Some(cost));
            eligible_owner[index * labels + label] = eligible;
        }
        let disagreement = ((f64::from(*structure) - structure_floor) / structure_span)
            .clamp(0.0, 1.0)
            * STRUCTURE_SEAM_WEIGHT;
        grid.set_disagreement(index, disagreement);
    }
    let mut owners = grid.per_cell_minimum();
    let mut energy = grid.energy(&owners);
    let initial_energy = energy;
    for _ in 0..3 {
        let mut improved = false;
        for label in 0..labels {
            let candidate = grid.fold_candidate(&owners, label);
            let candidate_energy = grid.energy(&candidate);
            if candidate_energy < energy - 1e-9 {
                owners = candidate;
                energy = candidate_energy;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    let mut owned = vec![0usize; labels];
    let mut fallback = vec![false; columns * rows];
    for (index, &owner) in owners.iter().enumerate() {
        if owner == NO_OWNER {
            fallback[index] = true;
            continue;
        }
        let label = usize::from(owner - 1);
        owned[label] += 1;
        if !eligible_owner[index * labels + label] {
            fallback[index] = true;
        }
    }
    println!(
        "  - PSD ownership: {}x{} cells, energy {:.1} -> {:.1}, owned cells per tile {:?} ({:.2?})",
        columns,
        rows,
        initial_energy,
        energy,
        owned,
        started.elapsed()
    );
    Ownership {
        columns,
        rows,
        owners,
        fallback,
    }
}
