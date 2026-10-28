//! Per-tile analysis for the planar scan mosaic.
//!
//! Validity masks, registration luma, quarter-resolution search planes,
//! sharpness blocks measured in output pixels, low-band means and the bounded
//! photometric preview, all from one decode of each tile.

use image::GrayImage;
use image::Rgb32FImage;
use nalgebra::Point2;
use rayon::prelude::*;

use super::{LOW_GRID, PREVIEW_LONG_SIDE, SHARPNESS_BLOCK, ScanSource, TileAnalysis};

// ---------------------------------------------------------------------------
// Tile analysis
// ---------------------------------------------------------------------------

pub(super) fn validity_of(source: &ScanSource) -> Vec<bool> {
    let rgb = source.rgb.as_raw();
    let mask = source.valid.as_ref().map(|mask| mask.as_raw());
    (0..rgb.len() / 3)
        .into_par_iter()
        .map(|index| {
            let pixel = &rgb[index * 3..index * 3 + 3];
            mask.is_none_or(|mask| mask[index] > 0) && pixel[0].max(pixel[1]).max(pixel[2]) > 1e-6
        })
        .collect()
}

pub(super) fn analyse_tile(source: &ScanSource, output_scale: f64) -> TileAnalysis {
    let (width, height) = source.rgb.dimensions();
    let (w, h) = (width as usize, height as usize);
    let rgb = source.rgb.as_raw();
    let valid = validity_of(source);
    let luma = (0..w * h)
        .into_par_iter()
        .map(|index| {
            let pixel = &rgb[index * 3..index * 3 + 3];
            0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
        })
        .collect::<Vec<f32>>();
    // Sigmas are fixed in output pixels so a lower-magnification tile cannot
    // look sharper merely because its pixels are coarser.
    // The detail band stops short of every tile's Nyquist limit on the output
    // grid, so sensor noise of an upsampled tile is not mistaken for focus.
    let scale = output_scale.clamp(0.25, 4.0) as f32;
    let coarse = blur_plane(&luma, w, h, 2.5 / scale);
    let reg_full = to_gray(&coarse, width, height);
    let fine = blur_plane(&luma, w, h, 1.0 / scale);
    let (sharpness, sharp_columns, sharp_rows) =
        block_sharpness(&luma, &fine, &coarse, &valid, w, h, SHARPNESS_BLOCK);
    drop(fine);
    drop(coarse);
    let sharpness = smooth_finite_3x3(&sharpness, sharp_columns, sharp_rows);
    let (quarter, quarter_width, quarter_height) = area_downsample(&luma, w, h, 4);
    drop(luma);
    let quarter = blur_plane(&quarter, quarter_width, quarter_height, 1.0);
    let reg_quarter = to_gray(&quarter, quarter_width as u32, quarter_height as u32);
    let valid_quarter = valid_downsample(&valid, w, h, 4);
    let (low, low_valid, low_columns, low_rows) =
        area_rgb_with_validity(rgb, &valid, w, h, LOW_GRID as usize);
    let preview = photometric_preview(&low, &low_valid, low_columns, low_rows, width, height);
    TileAnalysis {
        width,
        height,
        reg_full,
        reg_quarter,
        valid_quarter,
        sharpness,
        sharp_columns,
        sharp_rows,
        low,
        low_valid,
        low_columns,
        low_rows,
        preview,
    }
}

pub(super) fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil().max(1.0) as i32;
    let mut kernel = (-radius..=radius)
        .map(|offset| (-(offset * offset) as f32 / (2.0 * sigma * sigma)).exp())
        .collect::<Vec<_>>();
    let sum = kernel.iter().sum::<f32>();
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

/// Separable Gaussian blur with clamped borders.
pub(super) fn blur_plane(data: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.3 || width == 0 || height == 0 {
        return data.to_vec();
    }
    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() / 2;
    let mut horizontal = vec![0.0f32; width * height];
    horizontal
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let source = &data[y * width..(y + 1) * width];
            for (x, output) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                if x >= radius && x + radius < width {
                    let window = &source[x - radius..=x + radius];
                    for (weight, value) in kernel.iter().zip(window) {
                        sum += weight * value;
                    }
                } else {
                    for (tap, weight) in kernel.iter().enumerate() {
                        let index = (x + tap).saturating_sub(radius).min(width - 1);
                        sum += weight * source[index];
                    }
                }
                *output = sum;
            }
        });
    let mut output = vec![0.0f32; width * height];
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (tap, weight) in kernel.iter().enumerate() {
                let source_y = (y + tap).saturating_sub(radius).min(height - 1);
                let source = &horizontal[source_y * width..(source_y + 1) * width];
                for (value, sample) in row.iter_mut().zip(source) {
                    *value += weight * sample;
                }
            }
        });
    output
}

pub(super) fn to_gray(data: &[f32], width: u32, height: u32) -> GrayImage {
    let bytes = data
        .par_iter()
        .map(|value| (value * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    GrayImage::from_raw(width, height, bytes).expect("gray buffer matches its dimensions")
}

pub(super) fn area_downsample(
    data: &[f32],
    width: usize,
    height: usize,
    factor: usize,
) -> (Vec<f32>, usize, usize) {
    let columns = width / factor;
    let rows = height / factor;
    let output = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (column, row) = (index % columns, index / columns);
            let mut sum = 0.0;
            for y in row * factor..(row + 1) * factor {
                let line = &data[y * width + column * factor..y * width + (column + 1) * factor];
                sum += line.iter().sum::<f32>();
            }
            sum / (factor * factor) as f32
        })
        .collect();
    (output, columns, rows)
}

pub(super) fn valid_downsample(
    valid: &[bool],
    width: usize,
    height: usize,
    factor: usize,
) -> GrayImage {
    let columns = width / factor;
    let rows = height / factor;
    let output = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (column, row) = (index % columns, index / columns);
            let all = (row * factor..(row + 1) * factor).all(|y| {
                valid[y * width + column * factor..y * width + (column + 1) * factor]
                    .iter()
                    .all(|&value| value)
            });
            if all { 255u8 } else { 0u8 }
        })
        .collect();
    GrayImage::from_raw(columns as u32, rows as u32, output)
        .expect("quarter validity matches its dimensions")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn block_sharpness(
    luma: &[f32],
    fine: &[f32],
    coarse: &[f32],
    valid: &[bool],
    width: usize,
    height: usize,
    block: usize,
) -> (Vec<f32>, usize, usize) {
    let columns = width.div_ceil(block);
    let rows = height.div_ceil(block);
    let values = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (column, row) = (index % columns, index / columns);
            let x0 = column * block;
            let y0 = row * block;
            let x1 = (x0 + block).min(width);
            let y1 = (y0 + block).min(height);
            let mut count = 0usize;
            let mut energy = 0.0f64;
            let mut mean = 0.0f64;
            for y in y0..y1 {
                let line = y * width;
                for x in x0..x1 {
                    let index = line + x;
                    if !valid[index] {
                        continue;
                    }
                    let band = f64::from(fine[index] - coarse[index]);
                    energy += band * band;
                    mean += f64::from(luma[index]);
                    count += 1;
                }
            }
            let area = (x1 - x0) * (y1 - y0);
            if count * 10 < area * 9 || count == 0 {
                return f32::NAN;
            }
            let mean = (mean / count as f64).max(0.03);
            ((energy / count as f64).sqrt() / mean) as f32
        })
        .collect();
    (values, columns, rows)
}

pub(super) fn smooth_finite_3x3(values: &[f32], columns: usize, rows: usize) -> Vec<f32> {
    (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            if !values[index].is_finite() {
                return f32::NAN;
            }
            let (column, row) = (index % columns, index / columns);
            let mut sum = 0.0;
            let mut count = 0.0;
            for y in row.saturating_sub(1)..(row + 2).min(rows) {
                for x in column.saturating_sub(1)..(column + 2).min(columns) {
                    let value = values[y * columns + x];
                    if value.is_finite() {
                        sum += value;
                        count += 1.0;
                    }
                }
            }
            sum / count
        })
        .collect()
}

pub(super) fn area_rgb_with_validity(
    rgb: &[f32],
    valid: &[bool],
    width: usize,
    height: usize,
    factor: usize,
) -> (Vec<[f32; 3]>, Vec<bool>, usize, usize) {
    let columns = width.div_ceil(factor);
    let rows = height.div_ceil(factor);
    let blocks = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (column, row) = (index % columns, index / columns);
            let x0 = column * factor;
            let y0 = row * factor;
            let x1 = (x0 + factor).min(width);
            let y1 = (y0 + factor).min(height);
            let mut sum = [0.0f64; 3];
            let mut count = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let pixel = y * width + x;
                    if !valid[pixel] {
                        continue;
                    }
                    for channel in 0..3 {
                        sum[channel] += f64::from(rgb[pixel * 3 + channel]);
                    }
                    count += 1;
                }
            }
            let area = (x1 - x0) * (y1 - y0);
            if count == 0 {
                return ([0.0; 3], false);
            }
            (
                sum.map(|value| (value / count as f64) as f32),
                count == area && area == factor * factor,
            )
        })
        .collect::<Vec<_>>();
    let (values, flags): (Vec<_>, Vec<_>) = blocks.into_iter().unzip();
    (values, flags, columns, rows)
}

pub(super) fn photometric_preview(
    low: &[[f32; 3]],
    low_valid: &[bool],
    columns: usize,
    rows: usize,
    width: u32,
    height: u32,
) -> Rgb32FImage {
    let grid = Rgb32FImage::from_fn(columns as u32, rows as u32, |x, y| {
        let index = y as usize * columns + x as usize;
        if low_valid[index] {
            image::Rgb(low[index])
        } else {
            image::Rgb([0.0; 3])
        }
    });
    let scale = PREVIEW_LONG_SIDE / f64::from(width.max(height).max(1));
    let preview_width = (f64::from(width) * scale).round().max(1.0) as u32;
    let preview_height = (f64::from(height) * scale).round().max(1.0) as u32;
    image::imageops::resize(
        &grid,
        preview_width,
        preview_height,
        image::imageops::FilterType::Triangle,
    )
}

pub(super) fn quarter_valid(analysis: &TileAnalysis, point: Point2<f64>, radius: f64) -> bool {
    let mask = &analysis.valid_quarter;
    let check = |x: f64, y: f64| {
        let qx = (x - 1.5) / 4.0;
        let qy = (y - 1.5) / 4.0;
        qx >= 0.0
            && qy >= 0.0
            && (qx as u32) < mask.width()
            && (qy as u32) < mask.height()
            && mask.get_pixel(qx as u32, qy as u32)[0] > 0
    };
    check(point.x, point.y)
        && check(point.x - radius, point.y - radius)
        && check(point.x + radius, point.y - radius)
        && check(point.x - radius, point.y + radius)
        && check(point.x + radius, point.y + radius)
}
