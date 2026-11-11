//! Output-resolution focus evidence for focus-scan ownership.
//!
//! The streaming ownership grid is decided on analysis images of at most
//! 1400 px per layer, about one sample per five or six native pixels of a
//! 45 MP frame. Silk weave and the focus difference between two brackets of
//! one camera position are invisible at that resolution, so a focus decision
//! made there follows noise and can hand a whole region to a defocused frame.
//! This module measures the current canvas and one candidate layer on output
//! pixels instead, in the run's [`FocusBand`], and bins the evidence into the
//! cells of that ownership grid.

use rayon::prelude::*;

use super::{FocusBand, LayerSampler, StreamingMosaicStore, WeaveBand, luma, weave_band};

/// Lower bound of the luma a contrast-normalised focus is divided by, as in
/// the analysis-resolution acutance.
const FOCUS_LEVEL_FLOOR: f64 = 0.04;
/// Share of a cell window that must carry pixels both sources cover.
const FOCUS_MIN_SUPPORT: f64 = 0.25;

/// Per-cell focus of the current canvas and of one candidate layer, on the
/// streaming ownership grid of that layer. `NaN` where too few output pixels
/// carry both sources.
pub(super) struct OutputFocus {
    pub(super) base: Vec<f64>,
    pub(super) candidate: Vec<f64>,
}

#[derive(Clone, Copy, Default)]
struct CellSums {
    base_energy: f64,
    candidate_energy: f64,
    base_level: f64,
    candidate_level: f64,
    pixels: f64,
}

/// Mean squared neighbour difference at the centre of a 3×3 neighbourhood,
/// or `None` when any of its pixels is uncovered.
///
/// Neighbour differences respond up to the Nyquist frequency. Central
/// differences and the Sobel operator do not respond there at all, and silk
/// weave on a reduced render sits close to it.
fn difference_energy(rows: [&[f32]; 3], x: usize) -> Option<f64> {
    let mut values = [0.0f64; 9];
    for (row_index, row) in rows.iter().enumerate() {
        for column in 0..3 {
            let value = row[x + column - 1];
            if !value.is_finite() {
                return None;
            }
            values[row_index * 3 + column] = f64::from(value);
        }
    }
    let centre = values[4];
    let differences = [
        values[5] - centre,
        centre - values[3],
        values[7] - centre,
        centre - values[1],
    ];
    Some(differences.iter().map(|value| value * value).sum::<f64>() / 2.0)
}

/// Focus of the canvas and of `sampler`'s layer over the layer rectangle
/// `layer` (inclusive canvas bounds), binned into a `grid` of
/// `(columns, rows, analysis pixels per cell)`.
///
/// Both sources are compared only on pixels they both cover. A cell's value
/// is the contrast-normalised RMS of `band` over its 3×3 cell window.
pub(super) fn measure_output_focus(
    store: &StreamingMosaicStore,
    sampler: &LayerSampler<'_>,
    layer: (u32, u32, u32, u32),
    grid: (u32, u32, f64),
    band: FocusBand,
) -> Result<OutputFocus, String> {
    let (left, right, top, bottom) = layer;
    let width = (right - left + 1) as usize;
    let height = (bottom - top + 1) as usize;
    let (columns, rows, cell) = grid;
    let (columns, rows) = (columns as usize, rows as usize);
    if columns == 0 || rows == 0 {
        return Ok(OutputFocus {
            base: Vec::new(),
            candidate: Vec::new(),
        });
    }
    let base = store.luma_region(left, top, width as u32, height as u32)?;
    let mut candidate = vec![f32::NAN; width * height];
    candidate
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, value) in row.iter_mut().enumerate() {
                if let Some(pixel) = sampler.sample(x as f64, y as f64) {
                    *value = luma(pixel) as f32;
                }
            }
        });
    // Output pixels per ownership cell along either axis.
    let cell_px = (cell * sampler.scale).max(1.0);
    let (sums, minimum_pixels) = match band {
        FocusBand::Pixel => (
            pixel_band_sums(&base, &candidate, width, height, columns, rows, cell_px),
            FOCUS_MIN_SUPPORT * (3.0 * cell_px).powi(2),
        ),
        FocusBand::Weave => {
            let base_band = weave_band(&base, width, height);
            drop(base);
            let candidate_band = weave_band(&candidate, width, height);
            drop(candidate);
            // Binned pixels per ownership cell.
            let cell_px = cell_px * 0.5;
            (
                weave_band_sums(&base_band, &candidate_band, columns, rows, cell_px),
                FOCUS_MIN_SUPPORT * (3.0 * cell_px).powi(2),
            )
        }
    };
    let mut focus = OutputFocus {
        base: vec![f64::NAN; columns * rows],
        candidate: vec![f64::NAN; columns * rows],
    };
    for grid_y in 0..rows {
        for grid_x in 0..columns {
            let mut window = CellSums::default();
            for row in &sums[grid_y.saturating_sub(1)..=(grid_y + 1).min(rows - 1)] {
                for entry in &row[grid_x.saturating_sub(1)..=(grid_x + 1).min(columns - 1)] {
                    window.base_energy += entry.base_energy;
                    window.candidate_energy += entry.candidate_energy;
                    window.base_level += entry.base_level;
                    window.candidate_level += entry.candidate_level;
                    window.pixels += entry.pixels;
                }
            }
            if window.pixels < minimum_pixels {
                continue;
            }
            let index = grid_y * columns + grid_x;
            focus.base[index] = (window.base_energy / window.pixels).sqrt()
                / (window.base_level / window.pixels).max(FOCUS_LEVEL_FLOOR);
            focus.candidate[index] = (window.candidate_energy / window.pixels).sqrt()
                / (window.candidate_level / window.pixels).max(FOCUS_LEVEL_FLOOR);
        }
    }
    Ok(focus)
}

/// Neighbour-difference sums of every ownership cell, one row of cells per
/// entry, over output pixels both luma grids cover.
fn pixel_band_sums(
    base: &[f32],
    candidate: &[f32],
    width: usize,
    height: usize,
    columns: usize,
    rows: usize,
    cell_px: f64,
) -> Vec<Vec<CellSums>> {
    (0..rows)
        .into_par_iter()
        .map(|grid_y| {
            let mut sums = vec![CellSums::default(); columns];
            let first = ((grid_y as f64 * cell_px).ceil() as usize).max(1);
            let last = (((grid_y + 1) as f64 * cell_px).ceil() as usize).min(height - 1);
            for y in first..last {
                let base_rows = [
                    &base[(y - 1) * width..y * width],
                    &base[y * width..(y + 1) * width],
                    &base[(y + 1) * width..(y + 2) * width],
                ];
                let candidate_rows = [
                    &candidate[(y - 1) * width..y * width],
                    &candidate[y * width..(y + 1) * width],
                    &candidate[(y + 1) * width..(y + 2) * width],
                ];
                for x in 1..width.saturating_sub(1) {
                    let (Some(base_energy), Some(candidate_energy)) = (
                        difference_energy(base_rows, x),
                        difference_energy(candidate_rows, x),
                    ) else {
                        continue;
                    };
                    let column = ((x as f64 / cell_px) as usize).min(columns - 1);
                    let entry = &mut sums[column];
                    entry.base_energy += base_energy;
                    entry.candidate_energy += candidate_energy;
                    entry.base_level += f64::from(base_rows[1][x]);
                    entry.candidate_level += f64::from(candidate_rows[1][x]);
                    entry.pixels += 1.0;
                }
            }
            sums
        })
        .collect()
}

/// Weave-band sums of every ownership cell, one row of cells per entry, over
/// binned pixels both bands cover; `cell_px` is in binned pixels.
fn weave_band_sums(
    base: &WeaveBand,
    candidate: &WeaveBand,
    columns: usize,
    rows: usize,
    cell_px: f64,
) -> Vec<Vec<CellSums>> {
    let width = base.width.min(candidate.width);
    let height = base.height.min(candidate.height);
    (0..rows)
        .into_par_iter()
        .map(|grid_y| {
            let mut sums = vec![CellSums::default(); columns];
            let first = (grid_y as f64 * cell_px).ceil() as usize;
            let last = (((grid_y + 1) as f64 * cell_px).ceil() as usize).min(height);
            for y in first..last {
                let base_row = y * base.width;
                let candidate_row = y * candidate.width;
                for x in 0..width {
                    let base_band = base.band[base_row + x];
                    let candidate_band = candidate.band[candidate_row + x];
                    if !(base_band.is_finite() && candidate_band.is_finite()) {
                        continue;
                    }
                    let column = ((x as f64 / cell_px) as usize).min(columns - 1);
                    let entry = &mut sums[column];
                    entry.base_energy += f64::from(base_band * base_band);
                    entry.candidate_energy += f64::from(candidate_band * candidate_band);
                    entry.base_level += f64::from(base.level[base_row + x]);
                    entry.candidate_level += f64::from(candidate.level[candidate_row + x]);
                    entry.pixels += 1.0;
                }
            }
            sums
        })
        .collect()
}
