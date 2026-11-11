//! What the canvas reaches around a cell of a detail grid, on like material,
//! and how the cells of a grid hang together.
//!
//! The detail repair and the soft-region report both compare a cell with
//! the cells around it whose mean luma is close to its own: a smooth mount
//! or acrylic edge must not read as soft next to textured silk.

use rayon::prelude::*;
use std::collections::{HashSet, VecDeque};

use super::DetailGrid;

/// Connected groups of cells, 8-connected on a grid `columns` wide.
pub(super) fn components(cells: &[usize], columns: usize) -> Vec<Vec<usize>> {
    let set = cells.iter().copied().collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    let mut groups = Vec::new();
    for &start in cells {
        if !seen.insert(start) {
            continue;
        }
        let mut group = Vec::new();
        let mut pending = VecDeque::from([start]);
        while let Some(cell) = pending.pop_front() {
            group.push(cell);
            let (x, y) = ((cell % columns) as i64, (cell / columns) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= columns as i64 {
                        continue;
                    }
                    let neighbour = ny as usize * columns + nx as usize;
                    if set.contains(&neighbour) && seen.insert(neighbour) {
                        pending.push_back(neighbour);
                    }
                }
            }
        }
        group.sort_unstable();
        groups.push(group);
    }
    groups
}

/// Cells each way the neighbourhood reaches: about 2500 output pixels on
/// 祁序1, wider than a band no frame of a capture holds in focus (one ran
/// 9000 x 1500 px across 祁序1), so such a band is judged against the sharp
/// silk beyond it rather than against itself, which left every other cell of
/// it unrepaired.
pub(super) const NEIGHBOURHOOD_CELLS: usize = 24;

/// The 75th percentile of the finite `detail` within `NEIGHBOURHOOD_CELLS`
/// of every cell whose mean luma `level` lies within `similarity` of the
/// cell's, `NaN` where fewer than nine such cells were measured.
pub(super) fn neighbourhood_reference(
    detail: &[f64],
    level: &[f64],
    grid: &DetailGrid,
    similarity: f64,
) -> Vec<f64> {
    neighbourhood_percentiles(detail, level, grid, similarity, &[0.75])
        .into_iter()
        .map(|values| values.map_or(f64::NAN, |values| values[0]))
        .collect()
}

/// The `quantiles` of the finite `detail` within `NEIGHBOURHOOD_CELLS` of
/// every cell whose mean luma lies within `similarity` of the cell's, `None`
/// where fewer than nine such cells were measured.
pub(super) fn neighbourhood_percentiles(
    detail: &[f64],
    level: &[f64],
    grid: &DetailGrid,
    similarity: f64,
    quantiles: &[f64],
) -> Vec<Option<Vec<f64>>> {
    let (columns, rows) = grid.cell_dimensions();
    (0..columns * rows)
        .into_par_iter()
        .map(|cell| {
            if !detail[cell].is_finite() || !level[cell].is_finite() {
                return None;
            }
            let (x, y) = (cell % columns, cell / columns);
            let mut values = Vec::new();
            for ny in
                y.saturating_sub(NEIGHBOURHOOD_CELLS)..=(y + NEIGHBOURHOOD_CELLS).min(rows - 1)
            {
                for nx in x.saturating_sub(NEIGHBOURHOOD_CELLS)
                    ..=(x + NEIGHBOURHOOD_CELLS).min(columns - 1)
                {
                    let neighbour = ny * columns + nx;
                    let (value, neighbour_level) = (detail[neighbour], level[neighbour]);
                    if value.is_finite()
                        && neighbour_level.is_finite()
                        && neighbour_level <= level[cell] * similarity
                        && level[cell] <= neighbour_level * similarity
                    {
                        values.push(value);
                    }
                }
            }
            if values.len() < 9 {
                return None;
            }
            values.sort_by(f64::total_cmp);
            Some(
                quantiles
                    .iter()
                    .map(|quantile| values[((values.len() - 1) as f64 * quantile).round() as usize])
                    .collect(),
            )
        })
        .collect()
}
