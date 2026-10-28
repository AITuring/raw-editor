//! Joint photometry and the low-frequency band.
//!
//! Solves the photometric model, builds every tile's low band on the low grid
//! and blends their consensus across a wide transition, so tone steps fade
//! while detail is never averaged.

use super::super::photometric::PhotometricModel;
use super::super::photometric::PhotometricOptions;
use super::super::photometric::calibrate_overlap_photometry;
use super::super::stitching::flat_scan_paper_median;
use nalgebra::Matrix3;
use nalgebra::Point2;
use rayon::prelude::*;

use super::{
    CONSENSUS_FEATHER, CONSENSUS_SIGMA, CONSENSUS_TOLERANCE, Canvas, LOW_BAND_SIGMA, LOW_GRID,
    TileAnalysis, TileWarp, blur_plane, project,
};

// ---------------------------------------------------------------------------
// Photometry
// ---------------------------------------------------------------------------

pub(super) fn solve_photometry(
    analyses: &[TileAnalysis],
    homographies: &[Matrix3<f64>],
) -> Vec<(PhotometricModel, f32)> {
    let tiles = analyses
        .iter()
        .zip(homographies)
        .map(|(analysis, homography)| {
            let scale_x = analysis.preview.width() as f64 / analysis.width.max(1) as f64;
            let scale_y = analysis.preview.height() as f64 / analysis.height.max(1) as f64;
            let preview_to_source = Matrix3::new(
                1.0 / scale_x.max(1e-9),
                0.0,
                0.0,
                0.0,
                1.0 / scale_y.max(1e-9),
                0.0,
                0.0,
                0.0,
                1.0,
            );
            (analysis.preview.clone(), homography * preview_to_source)
        })
        .collect::<Vec<_>>();
    let medians = tiles
        .iter()
        .map(|(preview, _)| flat_scan_paper_median(preview))
        .collect::<Vec<_>>();
    let mut samples = medians.iter().copied().flatten().collect::<Vec<_>>();
    samples.sort_by(f32::total_cmp);
    let target = samples.get(samples.len() / 2).copied().unwrap_or(0.5);
    let gains = medians
        .iter()
        .map(|median| {
            median
                .map(|value| (target / value.max(0.05)).clamp(0.78, 1.28))
                .unwrap_or(1.0)
        })
        .collect::<Vec<_>>();
    let options = PhotometricOptions {
        allow_linear: true,
        min_samples_per_pair: 32,
        max_samples_per_pair: 4096,
        max_pair_log_scatter: 0.30,
        ..PhotometricOptions::default()
    };
    let calibration = calibrate_overlap_photometry(&tiles, &options);
    println!(
        "  - PSD joint photometric solve: {} models, {} reliable pairs, linear={}, paper target {:.3}",
        calibration.models.len(),
        calibration
            .pairs
            .iter()
            .filter(|pair| pair.reliable)
            .count(),
        calibration.linear_used,
        target
    );
    calibration.models.into_iter().zip(gains).collect()
}

pub(super) fn corrected_low_value(
    analysis: &TileAnalysis,
    model: &PhotometricModel,
    gain: f32,
    source: Point2<f64>,
    value: [f32; 3],
) -> [f32; 3] {
    let preview_x = source.x * analysis.preview.width() as f64 / analysis.width.max(1) as f64;
    let preview_y = source.y * analysis.preview.height() as f64 / analysis.height.max(1) as f64;
    let channel_gain = model.gain_at(preview_x, preview_y);
    let offset = model.offset_at(preview_x, preview_y);
    std::array::from_fn(|channel| {
        ((value[channel] * channel_gain[channel] + offset[channel]) * gain).clamp(0.0, 1.0)
    })
}

// ---------------------------------------------------------------------------
// Low-frequency band
// ---------------------------------------------------------------------------

pub(super) struct LowBand {
    pub(super) x0: usize,
    pub(super) y0: usize,
    pub(super) columns: usize,
    pub(super) rows: usize,
    pub(super) values: Vec<[f32; 3]>,
    pub(super) valid: Vec<bool>,
}

impl LowBand {
    pub(super) fn empty() -> Self {
        Self {
            x0: 0,
            y0: 0,
            columns: 0,
            rows: 0,
            values: Vec::new(),
            valid: Vec::new(),
        }
    }

    /// Bilinear sample in global low-grid coordinates; clamped to the band.
    pub(super) fn sample(&self, x: f64, y: f64) -> [f32; 3] {
        if self.columns == 0 || self.rows == 0 {
            return [0.0; 3];
        }
        let lx = (x - self.x0 as f64).clamp(0.0, (self.columns - 1) as f64);
        let ly = (y - self.y0 as f64).clamp(0.0, (self.rows - 1) as f64);
        let ix = (lx as usize).min(self.columns.saturating_sub(2));
        let iy = (ly as usize).min(self.rows.saturating_sub(2));
        let ix1 = (ix + 1).min(self.columns - 1);
        let iy1 = (iy + 1).min(self.rows - 1);
        let fx = (lx - ix as f64).clamp(0.0, 1.0) as f32;
        let fy = (ly - iy as f64).clamp(0.0, 1.0) as f32;
        let a = self.values[iy * self.columns + ix];
        let b = self.values[iy * self.columns + ix1];
        let c = self.values[iy1 * self.columns + ix];
        let d = self.values[iy1 * self.columns + ix1];
        std::array::from_fn(|channel| {
            let top = a[channel] * (1.0 - fx) + b[channel] * fx;
            let bottom = c[channel] * (1.0 - fx) + d[channel] * fx;
            top * (1.0 - fy) + bottom * fy
        })
    }

    pub(super) fn valid_at(&self, x: usize, y: usize) -> bool {
        x >= self.x0
            && y >= self.y0
            && x < self.x0 + self.columns
            && y < self.y0 + self.rows
            && self.valid[(y - self.y0) * self.columns + (x - self.x0)]
    }
}

pub(super) fn low_grid_size(canvas: &Canvas) -> (usize, usize) {
    (
        canvas.width.div_ceil(LOW_GRID) as usize,
        canvas.height.div_ceil(LOW_GRID) as usize,
    )
}

pub(super) fn low_grid_centre(index: usize) -> f64 {
    index as f64 * LOW_GRID as f64 + (LOW_GRID as f64 - 1.0) * 0.5
}

/// One tile's photometrically corrected low band on the output low grid.
pub(super) fn low_band(
    analysis: &TileAnalysis,
    warp: &TileWarp,
    model: &PhotometricModel,
    gain: f32,
    canvas: &Canvas,
) -> LowBand {
    let (grid_width, grid_height) = low_grid_size(canvas);
    let forward = warp.forward;
    let mut bounds = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for (x, y) in [
        (0.0, 0.0),
        (analysis.width as f64, 0.0),
        (analysis.width as f64, analysis.height as f64),
        (0.0, analysis.height as f64),
    ] {
        if let Some(point) = project(&forward, Point2::new(x, y)) {
            let gx = (point.x + canvas.offset_x) / LOW_GRID as f64;
            let gy = (point.y + canvas.offset_y) / LOW_GRID as f64;
            bounds.0 = bounds.0.min(gx);
            bounds.1 = bounds.1.max(gx);
            bounds.2 = bounds.2.min(gy);
            bounds.3 = bounds.3.max(gy);
        }
    }
    if !bounds.0.is_finite() {
        return LowBand::empty();
    }
    let x0 = (bounds.0.floor() - 2.0).max(0.0) as usize;
    let y0 = (bounds.2.floor() - 2.0).max(0.0) as usize;
    let x1 = ((bounds.1.ceil() + 2.0).max(0.0) as usize).min(grid_width);
    let y1 = ((bounds.3.ceil() + 2.0).max(0.0) as usize).min(grid_height);
    if x1 <= x0 || y1 <= y0 {
        return LowBand::empty();
    }
    let columns = x1 - x0;
    let rows = y1 - y0;
    let lw = analysis.low_columns;
    let lh = analysis.low_rows;
    let sampled = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let gx = x0 + index % columns;
            let gy = y0 + index / columns;
            let world = Point2::new(
                low_grid_centre(gx) - canvas.offset_x,
                low_grid_centre(gy) - canvas.offset_y,
            );
            let Some(source) = warp.source(world) else {
                return ([0.0; 3], false);
            };
            let lx = (source.x - (LOW_GRID as f64 - 1.0) * 0.5) / LOW_GRID as f64;
            let ly = (source.y - (LOW_GRID as f64 - 1.0) * 0.5) / LOW_GRID as f64;
            if lx < 0.0 || ly < 0.0 || lx >= (lw - 1) as f64 || ly >= (lh - 1) as f64 {
                return ([0.0; 3], false);
            }
            let ix = lx as usize;
            let iy = ly as usize;
            let fx = (lx - ix as f64) as f32;
            let fy = (ly - iy as f64) as f32;
            let corners = [
                iy * lw + ix,
                iy * lw + ix + 1,
                (iy + 1) * lw + ix,
                (iy + 1) * lw + ix + 1,
            ];
            if corners.iter().any(|&corner| !analysis.low_valid[corner]) {
                return ([0.0; 3], false);
            }
            let weights = [
                (1.0 - fx) * (1.0 - fy),
                fx * (1.0 - fy),
                (1.0 - fx) * fy,
                fx * fy,
            ];
            let mut value = [0.0f32; 3];
            for (&corner, &weight) in corners.iter().zip(&weights) {
                for (sum, sample) in value.iter_mut().zip(&analysis.low[corner]) {
                    *sum += sample * weight;
                }
            }
            (
                corrected_low_value(analysis, model, gain, source, value),
                true,
            )
        })
        .collect::<Vec<_>>();
    let valid = sampled.iter().map(|(_, valid)| *valid).collect::<Vec<_>>();
    // Normalised low-pass on the grid, then extend the band a little past the
    // tile edge so a sample near the border never reads zeros.
    let mask = valid
        .iter()
        .map(|&flag| f32::from(u8::from(flag)))
        .collect::<Vec<_>>();
    let blurred_mask = blur_plane(&mask, columns, rows, LOW_BAND_SIGMA);
    let mut values = vec![[0.0f32; 3]; columns * rows];
    let channels = (0..3)
        .map(|channel| {
            let plane = sampled
                .iter()
                .map(|(value, valid)| if *valid { value[channel] } else { 0.0 })
                .collect::<Vec<_>>();
            blur_plane(&plane, columns, rows, LOW_BAND_SIGMA)
        })
        .collect::<Vec<_>>();
    let mut known = vec![false; columns * rows];
    for index in 0..columns * rows {
        if valid[index] && blurred_mask[index] > 1e-3 {
            for channel in 0..3 {
                values[index][channel] = channels[channel][index] / blurred_mask[index];
            }
            known[index] = true;
        }
    }
    fill_unknown(&mut values, &mut known, columns, rows, 24);
    LowBand {
        x0,
        y0,
        columns,
        rows,
        values,
        valid,
    }
}

/// Chamfer distance of every valid band sample to the nearest invalid sample
/// or band edge, in grid pixels.
pub(super) fn border_distance(band: &LowBand) -> Vec<f32> {
    let (columns, rows) = (band.columns, band.rows);
    let mut distance = band
        .valid
        .iter()
        .map(|&valid| if valid { f32::MAX } else { 0.0 })
        .collect::<Vec<_>>();
    for y in 0..rows {
        for x in 0..columns {
            let index = y * columns + x;
            if distance[index] == 0.0 {
                continue;
            }
            let edge = (x.min(y).min(columns - 1 - x).min(rows - 1 - y) + 1) as f32;
            let mut best = distance[index].min(edge);
            if x > 0 {
                best = best.min(distance[index - 1] + 1.0);
            }
            if y > 0 {
                best = best.min(distance[index - columns] + 1.0);
                if x > 0 {
                    best = best.min(distance[index - columns - 1] + 1.414);
                }
                if x + 1 < columns {
                    best = best.min(distance[index - columns + 1] + 1.414);
                }
            }
            distance[index] = best;
        }
    }
    for y in (0..rows).rev() {
        for x in (0..columns).rev() {
            let index = y * columns + x;
            if distance[index] == 0.0 {
                continue;
            }
            let mut best = distance[index];
            if x + 1 < columns {
                best = best.min(distance[index + 1] + 1.0);
            }
            if y + 1 < rows {
                best = best.min(distance[index + columns] + 1.0);
                if x + 1 < columns {
                    best = best.min(distance[index + columns + 1] + 1.414);
                }
                if x > 0 {
                    best = best.min(distance[index + columns - 1] + 1.414);
                }
            }
            distance[index] = best;
        }
    }
    distance
}

/// The low band every output pixel receives: a feathered mean of all tiles'
/// low bands that stay within [`CONSENSUS_TOLERANCE`] of their weighted
/// median. A glare field or reflection seen by one tile is outvoted, and
/// tile borders fade in over [`CONSENSUS_FEATHER`] instead of stepping.
pub(super) fn consensus_low_band(bands: &[LowBand], canvas: &Canvas) -> LowBand {
    let (grid_width, grid_height) = low_grid_size(canvas);
    let count = grid_width * grid_height;
    let feathers = bands
        .par_iter()
        .map(|band| {
            border_distance(band)
                .into_iter()
                .map(|distance| {
                    let ramp = (distance / CONSENSUS_FEATHER).clamp(0.0, 1.0);
                    ramp * ramp + 1e-3
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let samples = (0..count)
        .into_par_iter()
        .map(|index| {
            let (x, y) = (index % grid_width, index / grid_width);
            let mut candidates = Vec::with_capacity(bands.len());
            for (band, feather) in bands.iter().zip(&feathers) {
                if band.valid_at(x, y) {
                    let local = (y - band.y0) * band.columns + (x - band.x0);
                    candidates.push((band.values[local], feather[local]));
                }
            }
            if candidates.is_empty() {
                return None;
            }
            let total = candidates.iter().map(|(_, weight)| weight).sum::<f32>();
            let median: [f32; 3] = std::array::from_fn(|channel| {
                let mut ordered = candidates
                    .iter()
                    .map(|(value, weight)| (value[channel], *weight))
                    .collect::<Vec<_>>();
                ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
                let mut accumulated = 0.0;
                for (value, weight) in &ordered {
                    accumulated += weight;
                    if accumulated >= total * 0.5 {
                        return *value;
                    }
                }
                ordered.last().map_or(0.0, |(value, _)| *value)
            });
            let mut sum = [0.0f32; 3];
            let mut weight_sum = 0.0f32;
            for (value, weight) in &candidates {
                let agrees = (0..3).all(|channel| {
                    (value[channel] - median[channel]).abs()
                        <= CONSENSUS_TOLERANCE * median[channel].max(0.05)
                });
                if agrees {
                    for channel in 0..3 {
                        sum[channel] += weight * value[channel];
                    }
                    weight_sum += weight;
                }
            }
            Some(if weight_sum > 1e-6 {
                sum.map(|value| value / weight_sum)
            } else {
                median
            })
        })
        .collect::<Vec<_>>();
    let valid = samples.iter().map(Option::is_some).collect::<Vec<_>>();
    let mask = valid
        .iter()
        .map(|&flag| f32::from(u8::from(flag)))
        .collect::<Vec<_>>();
    let blurred_mask = blur_plane(&mask, grid_width, grid_height, CONSENSUS_SIGMA);
    let channels = (0..3)
        .map(|channel| {
            let plane = samples
                .iter()
                .map(|sample| sample.map_or(0.0, |value| value[channel]))
                .collect::<Vec<_>>();
            blur_plane(&plane, grid_width, grid_height, CONSENSUS_SIGMA)
        })
        .collect::<Vec<_>>();
    let mut values = vec![[0.0f32; 3]; count];
    let mut known = vec![false; count];
    for index in 0..count {
        if valid[index] && blurred_mask[index] > 1e-3 {
            for channel in 0..3 {
                values[index][channel] = channels[channel][index] / blurred_mask[index];
            }
            known[index] = true;
        }
    }
    fill_unknown(&mut values, &mut known, grid_width, grid_height, 4);
    LowBand {
        x0: 0,
        y0: 0,
        columns: grid_width,
        rows: grid_height,
        values,
        valid: known,
    }
}

/// Extend known samples into unknown ones by averaging known neighbours.
pub(super) fn fill_unknown(
    values: &mut [[f32; 3]],
    known: &mut [bool],
    columns: usize,
    rows: usize,
    iterations: usize,
) {
    for _ in 0..iterations {
        let snapshot = known.to_vec();
        let previous = values.to_vec();
        let mut changed = false;
        for index in 0..columns * rows {
            if snapshot[index] {
                continue;
            }
            let (x, y) = (index % columns, index / columns);
            let mut sum = [0.0f32; 3];
            let mut count = 0.0f32;
            for ny in y.saturating_sub(1)..(y + 2).min(rows) {
                for nx in x.saturating_sub(1)..(x + 2).min(columns) {
                    let neighbour = ny * columns + nx;
                    if snapshot[neighbour] {
                        for channel in 0..3 {
                            sum[channel] += previous[neighbour][channel];
                        }
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                values[index] = sum.map(|value| value / count);
                known[index] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}
