//! Owner detail rendering for the planar scan mosaic.
//!
//! Writes the owner's high-frequency detail with bicubic sampling and a
//! few-pixel transition across ownership seams.

use image::Rgb32FImage;
use nalgebra::Point2;
use rayon::prelude::*;

use super::{
    Canvas, DETAIL_TRANSITION, LOW_GRID, LowBand, OWNER_CELL, Ownership, ScanSource, TileWarp,
    validity_of,
};

// ---------------------------------------------------------------------------
// Detail rendering
// ---------------------------------------------------------------------------

/// Validity eroded by two pixels so a bicubic footprint never reads a
/// transparent or out-of-frame sample.
pub(super) fn eroded_validity(source: &ScanSource) -> Vec<bool> {
    let (width, height) = source.rgb.dimensions();
    let (w, h) = (width as usize, height as usize);
    let valid = validity_of(source);
    let radius = 2usize;
    let mut horizontal = vec![false; w * h];
    horizontal
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let line = &valid[y * w..(y + 1) * w];
            for (x, output) in row.iter_mut().enumerate() {
                *output = x >= radius
                    && x + radius < w
                    && line[x - radius..=x + radius].iter().all(|&value| value);
            }
        });
    let mut output = vec![false; w * h];
    output.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y < radius || y + radius >= h {
            return;
        }
        for (x, value) in row.iter_mut().enumerate() {
            *value = (y - radius..=y + radius).all(|sy| horizontal[sy * w + x]);
        }
    });
    output
}

#[inline]
pub(super) fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Catmull-Rom sample clamped to its 4×4 neighbourhood (no ringing), the
/// same contract as `get_high_quality_interpolated_pixel`.
#[inline]
pub(super) fn sample_bicubic(
    data: &[f32],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
) -> [f32; 3] {
    let xf = x.floor();
    let yf = y.floor();
    let ix = xf as isize;
    let iy = yf as isize;
    let wx = catmull_rom_weights((x - xf) as f32);
    let wy = catmull_rom_weights((y - yf) as f32);
    let mut output = [0.0f32; 3];
    let mut low = [f32::MAX; 3];
    let mut high = [f32::MIN; 3];
    for (j, weight_y) in wy.iter().enumerate() {
        let sy = (iy - 1 + j as isize).clamp(0, height as isize - 1) as usize;
        let line = sy * width;
        let mut row = [0.0f32; 3];
        for (i, weight_x) in wx.iter().enumerate() {
            let sx = (ix - 1 + i as isize).clamp(0, width as isize - 1) as usize;
            let start = (line + sx) * 3;
            for channel in 0..3 {
                let value = data[start + channel];
                row[channel] += weight_x * value;
                low[channel] = low[channel].min(value);
                high[channel] = high[channel].max(value);
            }
        }
        for channel in 0..3 {
            output[channel] += weight_y * row[channel];
        }
    }
    std::array::from_fn(|channel| output[channel].clamp(low[channel], high[channel]))
}

/// Weights of the previous, own and next cell along one axis for a pixel at
/// local position `local` (0..OWNER_CELL).
#[inline]
pub(super) fn axis_transition(local: u32) -> [f32; 3] {
    let position = local as f32 + 0.5;
    let to_previous = position;
    let to_next = OWNER_CELL as f32 - position;
    let ramp = |distance: f32| (0.5 + distance / (2.0 * DETAIL_TRANSITION)).clamp(0.0, 1.0);
    if to_previous < DETAIL_TRANSITION {
        let own = ramp(to_previous);
        [1.0 - own, own, 0.0]
    } else if to_next < DETAIL_TRANSITION {
        let own = ramp(to_next);
        [0.0, own, 1.0 - own]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_tile_detail(
    label: usize,
    rgb: &Rgb32FImage,
    valid: &[bool],
    warp: &TileWarp,
    band: &LowBand,
    ownership: &Ownership,
    canvas: &Canvas,
    detail: &mut [f32],
    weight: &mut [f32],
) {
    let owner = u16::try_from(label + 1).unwrap_or(u16::MAX);
    let (source_width, source_height) = (rgb.width() as usize, rgb.height() as usize);
    let data = rgb.as_raw();
    let width = canvas.width as usize;
    let height = canvas.height as usize;
    let cell = OWNER_CELL as usize;
    let rows_per_band = cell;
    detail
        .par_chunks_mut(width * 3 * rows_per_band)
        .zip(weight.par_chunks_mut(width * rows_per_band))
        .enumerate()
        .for_each(|(cell_row, (detail_band, weight_band))| {
            let band_rows = weight_band.len() / width;
            for column in 0..ownership.columns {
                let (c, r) = (column as isize, cell_row as isize);
                let mut near_owner = false;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if ownership.owner(c + dx, r + dy) == owner {
                            near_owner = true;
                        }
                    }
                }
                let fallback = ownership.fallback[cell_row * ownership.columns + column];
                if !near_owner && !fallback {
                    continue;
                }
                let x_start = column * cell;
                let x_end = ((column + 1) * cell).min(width);
                for local_y in 0..band_rows {
                    let y = cell_row * cell + local_y;
                    if y >= height {
                        break;
                    }
                    let wy = axis_transition(local_y as u32);
                    let world_y = y as f64 - canvas.offset_y;
                    let grid_y = (y as f64 - (LOW_GRID as f64 - 1.0) * 0.5) / LOW_GRID as f64;
                    for x in x_start..x_end {
                        let wx = axis_transition((x - x_start) as u32);
                        let mut own_weight = 0.0f32;
                        if near_owner {
                            for (j, weight_y) in wy.iter().enumerate() {
                                if *weight_y <= 0.0 {
                                    continue;
                                }
                                for (i, weight_x) in wx.iter().enumerate() {
                                    if *weight_x > 0.0
                                        && ownership.owner(c + i as isize - 1, r + j as isize - 1)
                                            == owner
                                    {
                                        own_weight += weight_x * weight_y;
                                    }
                                }
                            }
                        }
                        let pixel = local_y * width + x;
                        let current = weight_band[pixel];
                        if own_weight <= 0.0 && !(fallback && current == 0.0) {
                            continue;
                        }
                        let world_x = x as f64 - canvas.offset_x;
                        let Some(source) = warp.source(Point2::new(world_x, world_y)) else {
                            continue;
                        };
                        let (sx, sy) = (source.x, source.y);
                        if sx < 0.0
                            || sy < 0.0
                            || sx > (source_width - 1) as f64
                            || sy > (source_height - 1) as f64
                        {
                            continue;
                        }
                        let nearest = (sy.round() as usize).min(source_height - 1) * source_width
                            + (sx.round() as usize).min(source_width - 1);
                        if !valid[nearest] {
                            continue;
                        }
                        let value = sample_bicubic(data, source_width, source_height, sx, sy);
                        let start = pixel * 3;
                        if own_weight > 0.0 {
                            if current < 0.0 {
                                detail_band[start..start + 3].fill(0.0);
                                weight_band[pixel] = 0.0;
                            }
                            let grid_x =
                                (x as f64 - (LOW_GRID as f64 - 1.0) * 0.5) / LOW_GRID as f64;
                            let low = band.sample(grid_x, grid_y);
                            for channel in 0..3 {
                                detail_band[start + channel] +=
                                    own_weight * (value[channel] - low[channel]);
                            }
                            weight_band[pixel] += own_weight;
                        } else {
                            // First valid non-owner sample of a pixel no owner
                            // can serve; replaced as soon as an owner writes.
                            detail_band[start..start + 3].copy_from_slice(&value);
                            weight_band[pixel] = -1.0;
                        }
                    }
                }
            }
        });
}
