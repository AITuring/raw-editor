//! Low-frequency tone tiles and their bilinear sampling.
//!
//! Gaussian low-frequency fields, world-aligned (native) tone tiles, applying a
//! low-frequency transform to owned pixels, and the tile samplers.

use image::GrayImage;
use image::Luma;
use image::Rgb;
use image::Rgb32FImage;
use rayon::prelude::*;

use super::{
    TONE_CELL_PARTIAL, TONE_CELL_SAMPLE, TONE_GRID_DIVISOR, TONE_LOW_FREQUENCY_SIGMA, ToneSolve,
    ToneSolveStatus, ToneTile,
};

/// Estimate a low-frequency RGB field on a 1/16 grid and bilinearly expand it
/// to the input dimensions.
pub(crate) fn low_frequency_field(image: &Rgb32FImage) -> Rgb32FImage {
    let low = low_frequency_grid(image);
    bilinear_expand(&low, image.width(), image.height())
}

/// Analysis-resolution low-frequency field.  Callers that already retain an
/// ownership grid should keep this image and map it by world coordinates,
/// avoiding a full-resolution intermediate allocation.
pub(crate) fn low_frequency_grid(image: &Rgb32FImage) -> Rgb32FImage {
    if image.width() == 0 || image.height() == 0 {
        return Rgb32FImage::new(image.width(), image.height());
    }
    let low_width = image.width().div_ceil(TONE_GRID_DIVISOR).max(1);
    let low_height = image.height().div_ceil(TONE_GRID_DIVISOR).max(1);
    let low = image::imageops::resize(
        image,
        low_width,
        low_height,
        image::imageops::FilterType::Triangle,
    );
    image::imageops::blur(&low, TONE_LOW_FREQUENCY_SIGMA)
}

/// Separable Gaussian of a `width × height` grid of `N`-channel values with
/// zero padding, so that a normalised convolution sees no value outside it.
pub(in crate::panorama_utils::stack_pipeline) fn gaussian_cells<const N: usize>(
    values: &[[f64; N]],
    width: usize,
    height: usize,
    sigma: f64,
) -> Vec<[f64; N]> {
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel = (-radius..=radius)
        .map(|offset| (-(offset as f64).powi(2) / (2.0 * sigma * sigma)).exp())
        .collect::<Vec<_>>();
    let pass = |source: &[[f64; N]], horizontal: bool| {
        let mut output = vec![[0.0; N]; source.len()];
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0; N];
                for (tap, weight) in kernel.iter().enumerate() {
                    let offset = tap as isize - radius;
                    let (sx, sy) = if horizontal {
                        (x as isize + offset, y as isize)
                    } else {
                        (x as isize, y as isize + offset)
                    };
                    if sx < 0 || sy < 0 || sx >= width as isize || sy >= height as isize {
                        continue;
                    }
                    let value = source[sy as usize * width + sx as usize];
                    for channel in 0..N {
                        sum[channel] += weight * value[channel];
                    }
                }
                output[y * width + x] = sum;
            }
        }
        output
    };
    pass(&pass(values, true), false)
}

/// Integrate the renderer's native source sampler on the world analysis grid.
/// The deterministic quadrature averages each 16px cell before the 64px blur;
/// no fused raster or independently positioned source-grid cell is substituted.
pub(crate) fn world_aligned_native_tone_tile<F>(
    station_index: usize,
    owner_id: u16,
    bounds: (u32, u32, u32, u32),
    sample: F,
) -> ToneTile
where
    F: Fn(f64, f64) -> Option<[f32; 3]> + Sync,
{
    let (left, top, right, bottom) = bounds;
    let stride = TONE_GRID_DIVISOR.max(1);
    let width = (right.saturating_sub(left) + 1).div_ceil(stride).max(1);
    let height = (bottom.saturating_sub(top) + 1).div_ceil(stride).max(1);
    let cells = (0..height)
        .into_par_iter()
        .map(|row| {
            (0..width)
                .map(|column| {
                    let mut sum = [0.0f64; 3];
                    let mut covered = 0u32;
                    let mut native = [[0.0f32; 3]; 16];
                    for sy in 0..4 {
                        for sx in 0..4 {
                            let x = f64::from(left + column * stride)
                                + (f64::from(sx) + 0.5) * f64::from(stride) / 4.0
                                - 0.5;
                            let y = f64::from(top + row * stride)
                                + (f64::from(sy) + 0.5) * f64::from(stride) / 4.0
                                - 0.5;
                            if x > f64::from(right) || y > f64::from(bottom) {
                                continue;
                            }
                            if let Some(rgb) = sample(x, y) {
                                native[(sy * 4 + sx) as usize] = rgb;
                                for c in 0..3 {
                                    sum[c] += f64::from(rgb[c]);
                                }
                                covered += 1;
                            }
                        }
                    }
                    (sum, covered, native)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let weights = cells
        .iter()
        .map(|(_, covered, _)| [f64::from(*covered) / 16.0])
        .collect::<Vec<_>>();
    let sums = cells
        .iter()
        .map(|(sum, _, _)| sum.map(|v| v / 16.0))
        .collect::<Vec<_>>();
    let support = gaussian_cells(
        &weights,
        width as usize,
        height as usize,
        f64::from(TONE_LOW_FREQUENCY_SIGMA),
    );
    let blurred = gaussian_cells(
        &sums,
        width as usize,
        height as usize,
        f64::from(TONE_LOW_FREQUENCY_SIGMA),
    );
    let mut low = Rgb32FImage::new(width, height);
    let mut cell_mean = Rgb32FImage::new(width, height);
    let mut validity = GrayImage::new(width, height);
    for (i, (sum, covered, _)) in cells.iter().enumerate() {
        let (x, y) = (i as u32 % width, i as u32 / width);
        if *covered > 0 {
            cell_mean.put_pixel(x, y, Rgb(sum.map(|v| (v / f64::from(*covered)) as f32)));
        }
        if support[i][0] > 1e-9 {
            low.put_pixel(x, y, Rgb(blurred[i].map(|v| (v / support[i][0]) as f32)));
        }
        if *covered > 0 {
            validity.put_pixel(
                x,
                y,
                Luma([if *covered == 16 {
                    TONE_CELL_SAMPLE
                } else {
                    TONE_CELL_PARTIAL
                }]),
            );
        }
    }
    let native_detail = Rgb32FImage::from_fn(width * 4, height * 4, |x, y| {
        let cell = (y / 4 * width + x / 4) as usize;
        Rgb(cells[cell].2[(y % 4 * 4 + x % 4) as usize])
    });
    ToneTile {
        station_index,
        owner_id,
        low,
        validity,
        world_origin: (f64::from(left), f64::from(top)),
        world_size: (f64::from(width * stride), f64::from(height * stride)),
        world_stride: f64::from(stride),
        cell_mean,
        native_detail: Some(native_detail),
        cell_coverage: weights.iter().map(|v| v[0] as f32).collect(),
    }
}

/// Tone_Harmonizer field of one tile on a world-aligned analysis grid.
///
/// `tile` holds the tile's pixels with uncovered pixels exactly zero (the
/// compositor's transparency convention). `bounds` are the tile's inclusive
/// output-pixel bounds `(left, top, right, bottom)` and `to_tile` maps an
/// output pixel to tile pixels with the map the compositor samples through.
///
/// The low-frequency field averages covered tile pixels only (a normalised
/// convolution), so uncovered payload never darkens it, and it is resampled
/// through `to_tile`, so a rotated or projective tile is compared with its
/// neighbours at the right world position. A cell is tone evidence
/// ([`TONE_CELL_SAMPLE`]) only where the tile covers its whole analysis cell
/// (需求 9.1, 9.2, 9.7).
pub(crate) fn world_aligned_tone_tile(
    station_index: usize,
    owner_id: u16,
    tile: &Rgb32FImage,
    bounds: (u32, u32, u32, u32),
    to_tile: impl Fn(f64, f64) -> Option<(f64, f64)>,
) -> ToneTile {
    let divisor = TONE_GRID_DIVISOR.max(1);
    let stride = f64::from(divisor);
    let (left, top, right, bottom) = bounds;
    let world_width = (right.saturating_sub(left) + 1).div_ceil(divisor).max(1);
    let world_height = (bottom.saturating_sub(top) + 1).div_ceil(divisor).max(1);
    let mut low = Rgb32FImage::new(world_width, world_height);
    let mut validity = GrayImage::new(world_width, world_height);
    let mut cell_mean = Rgb32FImage::new(world_width, world_height);
    let mut cell_coverage = vec![0.0f32; world_width as usize * world_height as usize];
    let empty = |low, validity, cell_mean, cell_coverage| ToneTile {
        station_index,
        owner_id,
        low,
        validity,
        world_origin: (f64::from(left), f64::from(top)),
        world_size: (
            f64::from(world_width) * stride,
            f64::from(world_height) * stride,
        ),
        world_stride: stride,
        cell_mean,
        native_detail: None,
        cell_coverage,
    };
    let (tile_width, tile_height) = tile.dimensions();
    if tile_width == 0 || tile_height == 0 {
        return empty(low, validity, cell_mean, cell_coverage);
    }

    // Covered sums, covered counts and cell sizes on the tile's own grid.
    let grid_width = tile_width.div_ceil(divisor) as usize;
    let grid_height = tile_height.div_ceil(divisor) as usize;
    let rows = (0..grid_height)
        .into_par_iter()
        .map(|cell_y| {
            let mut row = vec![([0.0f64; 3], 0u32, 0u32); grid_width];
            let y_start = cell_y as u32 * divisor;
            for y in y_start..(y_start + divisor).min(tile_height) {
                for x in 0..tile_width {
                    let cell = &mut row[(x / divisor) as usize];
                    cell.2 += 1;
                    let pixel = tile.get_pixel(x, y).0;
                    if pixel[0].max(pixel[1]).max(pixel[2]) > 1e-6 {
                        for (sum, value) in cell.0.iter_mut().zip(pixel) {
                            *sum += f64::from(value);
                        }
                        cell.1 += 1;
                    }
                }
            }
            row
        })
        .collect::<Vec<_>>();
    let cells = rows.into_iter().flatten().collect::<Vec<_>>();
    let weights = cells
        .iter()
        .map(|&(_, covered, total)| [f64::from(covered) / f64::from(total.max(1))])
        .collect::<Vec<_>>();
    let weighted = cells
        .iter()
        .zip(&weights)
        .map(|(&(sum, covered, _), weight)| {
            std::array::from_fn(|channel| {
                if covered == 0 {
                    0.0
                } else {
                    sum[channel] / f64::from(covered) * weight[0]
                }
            })
        })
        .collect::<Vec<[f64; 3]>>();
    let sigma = f64::from(TONE_LOW_FREQUENCY_SIGMA);
    let blurred = gaussian_cells(&weighted, grid_width, grid_height, sigma);
    let support = gaussian_cells(&weights, grid_width, grid_height, sigma);
    let field = |cell: usize| -> Option<[f64; 3]> {
        (support[cell][0] > 1e-9)
            .then(|| std::array::from_fn(|channel| blurred[cell][channel] / support[cell][0]))
    };

    for world_y in 0..world_height {
        for world_x in 0..world_width {
            let x = f64::from(left) + f64::from(world_x) * stride + (stride - 1.0) * 0.5;
            let y = f64::from(top) + f64::from(world_y) * stride + (stride - 1.0) * 0.5;
            let Some((tile_x, tile_y)) = to_tile(x, y) else {
                continue;
            };
            if !(tile_x >= 0.0
                && tile_y >= 0.0
                && tile_x < f64::from(tile_width)
                && tile_y < f64::from(tile_height))
            {
                continue;
            }
            let cell_x = ((tile_x / stride) as usize).min(grid_width - 1);
            let cell_y = ((tile_y / stride) as usize).min(grid_height - 1);
            let cell = cell_y * grid_width + cell_x;
            let (cell_sum, covered, total) = cells[cell];
            if covered == 0 {
                continue;
            }
            cell_mean.put_pixel(
                world_x,
                world_y,
                Rgb(cell_sum.map(|channel| (channel / f64::from(covered)) as f32)),
            );
            cell_coverage[world_y as usize * world_width as usize + world_x as usize] =
                covered as f32 / total.max(1) as f32;
            // Bilinear over the covered-only field, renormalised over the
            // neighbours that carry any covered pixel.
            let u = (tile_x - (stride - 1.0) * 0.5) / stride;
            let v = (tile_y - (stride - 1.0) * 0.5) / stride;
            let (u0, v0) = (u.floor(), v.floor());
            let mut sum = [0.0f64; 3];
            let mut total_weight = 0.0f64;
            for (dy, weight_y) in [(0.0, 1.0 - (v - v0)), (1.0, v - v0)] {
                for (dx, weight_x) in [(0.0, 1.0 - (u - u0)), (1.0, u - u0)] {
                    let (nx, ny) = (u0 + dx, v0 + dy);
                    if nx < 0.0 || ny < 0.0 || nx >= grid_width as f64 || ny >= grid_height as f64 {
                        continue;
                    }
                    let neighbour = ny as usize * grid_width + nx as usize;
                    let Some(value) = field(neighbour) else {
                        continue;
                    };
                    let weight = weight_x * weight_y;
                    for channel in 0..3 {
                        sum[channel] += weight * value[channel];
                    }
                    total_weight += weight;
                }
            }
            let value = if total_weight > 1e-12 {
                sum.map(|channel| channel / total_weight)
            } else if let Some(value) = field(cell) {
                value
            } else {
                continue;
            };
            low.put_pixel(world_x, world_y, Rgb(value.map(|channel| channel as f32)));
            let level = if covered == total {
                TONE_CELL_SAMPLE
            } else {
                TONE_CELL_PARTIAL
            };
            validity.put_pixel(world_x, world_y, Luma([level]));
        }
    }
    empty(low, validity, cell_mean, cell_coverage)
}

/// Apply the low-frequency affine correction while preserving the owner's
/// high-frequency residual.  Pixels outside `evidence` are exact identity.
pub(crate) fn apply_low_frequency_tone(
    source: &Rgb32FImage,
    owner: &Rgb32FImage,
    evidence: &GrayImage,
    solve: &ToneSolve,
) -> Rgb32FImage {
    let width = owner.width().min(source.width()).min(evidence.width());
    let height = owner.height().min(source.height()).min(evidence.height());
    if width == 0 || height == 0 {
        return Rgb32FImage::new(width, height);
    }
    if solve.status == ToneSolveStatus::Identity {
        return image::imageops::crop_imm(owner, 0, 0, width, height).to_image();
    }
    let source_image = image::imageops::crop_imm(source, 0, 0, width, height).to_image();
    let source_low = low_frequency_field(&source_image);
    let owner_image = image::imageops::crop_imm(owner, 0, 0, width, height).to_image();
    let owner_low = low_frequency_field(&owner_image);
    Rgb32FImage::from_fn(width, height, |x, y| {
        if evidence.get_pixel(x, y)[0] == 0 {
            return *owner_image.get_pixel(x, y);
        }
        let low = source_low.get_pixel(x, y);
        let owned = owner_image.get_pixel(x, y);
        let owned_low = owner_low.get_pixel(x, y);
        Rgb(std::array::from_fn(|channel| {
            solve.gain[channel] * low[channel]
                + solve.offset[channel]
                + (owned[channel] - owned_low[channel])
        }))
    })
}

/// Explicitly named wrapper used by the compositor stage.
pub(crate) fn harmonize_owned_tone(
    source: &Rgb32FImage,
    owner: &Rgb32FImage,
    evidence: &GrayImage,
    solve: &ToneSolve,
) -> Rgb32FImage {
    apply_low_frequency_tone(source, owner, evidence, solve)
}

/// The tile's field at a world position, bilinear between the analysis-cell
/// centres the field was built at and renormalised over the neighbouring cells
/// that carry a field (validity ≠ 0). `None` where no neighbour has one.
pub(crate) fn sample_tile_bilinear(
    tile: &ToneTile,
    world_x: f64,
    world_y: f64,
) -> Option<[f32; 3]> {
    sample_tile_field_bilinear(tile, &tile.low, world_x, world_y)
}

pub(in crate::panorama_utils::stack_pipeline) fn sample_tile_detail_bilinear(
    tile: &ToneTile,
    world_x: f64,
    world_y: f64,
) -> Option<[f32; 3]> {
    let mean = if let Some(native) = tile.native_detail.as_ref() {
        let stride = tile.world_stride / 4.0;
        let u = (world_x - tile.world_origin.0 - (stride - 1.0) * 0.5) / stride;
        let v = (world_y - tile.world_origin.1 - (stride - 1.0) * 0.5) / stride;
        let (x, y) = (u.floor() as i64, v.floor() as i64);
        let mut sum = [0.0f64; 3];
        let mut mass = 0.0;
        for dy in 0..=1 {
            for dx in 0..=1 {
                let (px, py) = (x + dx, y + dy);
                if px < 0
                    || py < 0
                    || px >= native.width() as i64
                    || py >= native.height() as i64
                    || tile.validity.get_pixel(px as u32 / 4, py as u32 / 4)[0] != TONE_CELL_SAMPLE
                {
                    continue;
                }
                let w = (if dx == 0 {
                    1.0 - (u - u.floor())
                } else {
                    u - u.floor()
                }) * (if dy == 0 {
                    1.0 - (v - v.floor())
                } else {
                    v - v.floor()
                });
                for c in 0..3 {
                    sum[c] += w * f64::from(native.get_pixel(px as u32, py as u32)[c]);
                }
                mass += w;
            }
        }
        if mass < 1.0 - 1e-6 {
            return None;
        }
        sum.map(|v| (v / mass) as f32)
    } else {
        sample_tile_field_bilinear(tile, &tile.cell_mean, world_x, world_y)?
    };
    let low = sample_tile_bilinear(tile, world_x, world_y)?;
    Some(std::array::from_fn(|c| mean[c] - low[c]))
}

pub(super) fn sample_tile_field_bilinear(
    tile: &ToneTile,
    field: &Rgb32FImage,
    world_x: f64,
    world_y: f64,
) -> Option<[f32; 3]> {
    let stride = tile.world_stride;
    if !stride.is_finite() || stride <= 0.0 {
        return None;
    }
    let (width, height) = field.dimensions();
    let u = (world_x - tile.world_origin.0 - (stride - 1.0) * 0.5) / stride;
    let v = (world_y - tile.world_origin.1 - (stride - 1.0) * 0.5) / stride;
    let (u0, v0) = (u.floor(), v.floor());
    let mut sum = [0.0f64; 3];
    let mut total = 0.0f64;
    for (dy, weight_y) in [(0.0, 1.0 - (v - v0)), (1.0, v - v0)] {
        for (dx, weight_x) in [(0.0, 1.0 - (u - u0)), (1.0, u - u0)] {
            let (x, y) = (u0 + dx, v0 + dy);
            if x < 0.0 || y < 0.0 || x >= f64::from(width) || y >= f64::from(height) {
                continue;
            }
            let (x, y) = (x as u32, y as u32);
            if x >= tile.validity.width()
                || y >= tile.validity.height()
                || tile.validity.get_pixel(x, y)[0] == 0
            {
                continue;
            }
            let weight = weight_x * weight_y;
            let value = field.get_pixel(x, y).0;
            for channel in 0..3 {
                sum[channel] += weight * f64::from(value[channel]);
            }
            total += weight;
        }
    }
    (total > 1e-9).then(|| sum.map(|channel| (channel / total) as f32))
}

pub(super) fn bilinear_expand(low: &Rgb32FImage, width: u32, height: u32) -> Rgb32FImage {
    if low.width() == width && low.height() == height {
        return low.clone();
    }
    let x_scale = if width > 1 {
        (low.width().saturating_sub(1)) as f32 / (width - 1) as f32
    } else {
        0.0
    };
    let y_scale = if height > 1 {
        (low.height().saturating_sub(1)) as f32 / (height - 1) as f32
    } else {
        0.0
    };
    Rgb32FImage::from_fn(width, height, |x, y| {
        let fx = x as f32 * x_scale;
        let fy = y as f32 * y_scale;
        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(low.width().saturating_sub(1));
        let y1 = (y0 + 1).min(low.height().saturating_sub(1));
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let top = low.get_pixel(x0, y0);
        let top_right = low.get_pixel(x1, y0);
        let bottom = low.get_pixel(x0, y1);
        let bottom_right = low.get_pixel(x1, y1);
        Rgb(std::array::from_fn(|channel| {
            let a = top[channel] * (1.0 - tx) + top_right[channel] * tx;
            let b = bottom[channel] * (1.0 - tx) + bottom_right[channel] * tx;
            a * (1.0 - ty) + b * ty
        }))
    })
}
