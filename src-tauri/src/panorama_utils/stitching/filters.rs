//! Resampling and filtering primitives.
//!
//! Resizing, half-resolution pyramids, Laplacian band operations, masked box
//! blurs and multiband blending used by the compositors.

use image::GrayImage;
use image::Rgb32FImage;
use rayon::prelude::*;

use super::PANORAMA_GLOBAL_TONE_FIRST_BAND;

pub(super) fn resize_rgb(image: &Rgb32FImage, width: u32, height: u32) -> Rgb32FImage {
    image::imageops::resize(
        image,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Triangle,
    )
}

pub(in crate::panorama_utils) fn downsample_rgb_half(image: &Rgb32FImage) -> Rgb32FImage {
    let (source_width, source_height) = image.dimensions();
    let target_width = source_width.div_ceil(2).max(1);
    let target_height = source_height.div_ceil(2).max(1);
    if (source_width, source_height) == (target_width, target_height) {
        return image.clone();
    }

    let source = image.as_raw();
    let source_stride = source_width as usize * 3;
    let target_stride = target_width as usize * 3;
    let mut output = vec![0.0f32; target_stride * target_height as usize];
    output
        .par_chunks_mut(target_stride)
        .enumerate()
        .for_each(|(target_y, row)| {
            let source_y0 = (target_y * 2).min(source_height as usize - 1);
            let source_y1 = (source_y0 + 1).min(source_height as usize - 1);
            for target_x in 0..target_width as usize {
                let source_x0 = (target_x * 2).min(source_width as usize - 1);
                let source_x1 = (source_x0 + 1).min(source_width as usize - 1);
                let top_left = source_y0 * source_stride + source_x0 * 3;
                let top_right = source_y0 * source_stride + source_x1 * 3;
                let bottom_left = source_y1 * source_stride + source_x0 * 3;
                let bottom_right = source_y1 * source_stride + source_x1 * 3;
                let output_start = target_x * 3;
                for channel in 0..3 {
                    row[output_start + channel] = (source[top_left + channel]
                        + source[top_right + channel]
                        + source[bottom_left + channel]
                        + source[bottom_right + channel])
                        * 0.25;
                }
            }
        });
    Rgb32FImage::from_raw(target_width, target_height, output)
        .expect("half-resolution RGB buffer dimensions must match")
}

pub(super) fn downsample_mask_half(mask: &GrayImage) -> GrayImage {
    let (source_width, source_height) = mask.dimensions();
    let target_width = source_width.div_ceil(2).max(1);
    let target_height = source_height.div_ceil(2).max(1);
    if (source_width, source_height) == (target_width, target_height) {
        return mask.clone();
    }

    let source = mask.as_raw();
    let source_stride = source_width as usize;
    let target_stride = target_width as usize;
    let mut output = vec![0u8; target_stride * target_height as usize];
    output
        .par_chunks_mut(target_stride)
        .enumerate()
        .for_each(|(target_y, row)| {
            let source_y0 = (target_y * 2).min(source_height as usize - 1);
            let source_y1 = (source_y0 + 1).min(source_height as usize - 1);
            for (target_x, output) in row.iter_mut().enumerate() {
                let source_x0 = (target_x * 2).min(source_width as usize - 1);
                let source_x1 = (source_x0 + 1).min(source_width as usize - 1);
                let total = u16::from(source[source_y0 * source_stride + source_x0])
                    + u16::from(source[source_y0 * source_stride + source_x1])
                    + u16::from(source[source_y1 * source_stride + source_x0])
                    + u16::from(source[source_y1 * source_stride + source_x1]);
                *output = ((total + 2) / 4) as u8;
            }
        });
    GrayImage::from_raw(target_width, target_height, output)
        .expect("half-resolution mask buffer dimensions must match")
}

#[derive(Clone, Copy)]
pub(super) struct LinearSample {
    pub(super) lower: usize,
    pub(super) upper: usize,
    pub(super) upper_weight: f32,
}

pub(super) fn linear_samples(source_length: u32, target_length: u32) -> Vec<LinearSample> {
    let scale = source_length as f64 / target_length.max(1) as f64;
    (0..target_length.max(1))
        .map(|target| {
            let source = ((f64::from(target) + 0.5) * scale - 0.5)
                .clamp(0.0, f64::from(source_length.saturating_sub(1)));
            let lower = source.floor() as usize;
            let upper = (lower + 1).min(source_length.saturating_sub(1) as usize);
            LinearSample {
                lower,
                upper,
                upper_weight: (source - lower as f64) as f32,
            }
        })
        .collect()
}

pub(super) fn subtract_upsampled_rgb(fine: &Rgb32FImage, coarse: &Rgb32FImage) -> Rgb32FImage {
    let (width, height) = fine.dimensions();
    let x_samples = linear_samples(coarse.width(), width);
    let y_samples = linear_samples(coarse.height(), height);
    let coarse_stride = coarse.width() as usize * 3;
    let output_stride = width as usize * 3;
    let coarse_pixels = coarse.as_raw();
    let fine_pixels = fine.as_raw();
    let mut output = vec![0.0f32; output_stride * height as usize];

    output
        .par_chunks_mut(output_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            let y_weight = y_sample.upper_weight;
            let fine_row = &fine_pixels[y * output_stride..(y + 1) * output_stride];
            for (x, x_sample) in x_samples.iter().copied().enumerate() {
                let x_weight = x_sample.upper_weight;
                let top_left = y_sample.lower * coarse_stride + x_sample.lower * 3;
                let top_right = y_sample.lower * coarse_stride + x_sample.upper * 3;
                let bottom_left = y_sample.upper * coarse_stride + x_sample.lower * 3;
                let bottom_right = y_sample.upper * coarse_stride + x_sample.upper * 3;
                let output_start = x * 3;
                for channel in 0..3 {
                    let top = coarse_pixels[top_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[top_right + channel] * x_weight;
                    let bottom = coarse_pixels[bottom_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[bottom_right + channel] * x_weight;
                    let low_frequency = top * (1.0 - y_weight) + bottom * y_weight;
                    row[output_start + channel] = fine_row[output_start + channel] - low_frequency;
                }
            }
        });

    Rgb32FImage::from_raw(width, height, output)
        .expect("Laplacian detail buffer dimensions must match")
}

pub(super) fn upsample_and_add_rgb(coarse: &Rgb32FImage, detail: &Rgb32FImage) -> Rgb32FImage {
    let (width, height) = detail.dimensions();
    let x_samples = linear_samples(coarse.width(), width);
    let y_samples = linear_samples(coarse.height(), height);
    let coarse_stride = coarse.width() as usize * 3;
    let output_stride = width as usize * 3;
    let coarse_pixels = coarse.as_raw();
    let detail_pixels = detail.as_raw();
    let mut output = vec![0.0f32; output_stride * height as usize];

    output
        .par_chunks_mut(output_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            let y_weight = y_sample.upper_weight;
            let detail_row = &detail_pixels[y * output_stride..(y + 1) * output_stride];
            for (x, x_sample) in x_samples.iter().copied().enumerate() {
                let x_weight = x_sample.upper_weight;
                let top_left = y_sample.lower * coarse_stride + x_sample.lower * 3;
                let top_right = y_sample.lower * coarse_stride + x_sample.upper * 3;
                let bottom_left = y_sample.upper * coarse_stride + x_sample.lower * 3;
                let bottom_right = y_sample.upper * coarse_stride + x_sample.upper * 3;
                let output_start = x * 3;
                for channel in 0..3 {
                    let top = coarse_pixels[top_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[top_right + channel] * x_weight;
                    let bottom = coarse_pixels[bottom_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[bottom_right + channel] * x_weight;
                    row[output_start + channel] = top * (1.0 - y_weight)
                        + bottom * y_weight
                        + detail_row[output_start + channel];
                }
            }
        });

    Rgb32FImage::from_raw(width, height, output).expect("reconstructed image dimensions must match")
}

pub(super) fn combine_rgb(
    base: &Rgb32FImage,
    candidate: &Rgb32FImage,
    mask: &GrayImage,
    hard_mask: bool,
) -> Rgb32FImage {
    let (width, height) = base.dimensions();
    let mut output = vec![0.0f32; width as usize * height as usize * 3];
    output
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let mask_value = if hard_mask {
                    if mask.get_pixel(x, y)[0] >= 128 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    mask.get_pixel(x, y)[0] as f32 / 255.0
                };
                let base_pixel = base.get_pixel(x, y);
                let candidate_pixel = candidate.get_pixel(x, y);
                let start = x as usize * 3;
                for channel in 0..3 {
                    row[start + channel] = base_pixel[channel] * (1.0 - mask_value)
                        + candidate_pixel[channel] * mask_value;
                }
            }
        });
    Rgb32FImage::from_raw(width, height, output).expect("combined image dimensions must match")
}

pub(super) fn masked_box_blur_rgb(
    source: &Rgb32FImage,
    mask: &GrayImage,
    radius: usize,
) -> Rgb32FImage {
    let (width, height) = source.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) || radius == 0 {
        return source.clone();
    }

    let width = width as usize;
    let height = height as usize;
    let pixel_count = width * height;
    let window_size = radius.saturating_mul(2).saturating_add(1);
    let source_pixels = source.as_raw();
    let mask_pixels = mask.as_raw();
    let mut horizontal_values = vec![0.0f32; pixel_count * 3];
    let mut horizontal_weights = vec![0u32; pixel_count];

    horizontal_values
        .par_chunks_mut(width * 3)
        .zip(horizontal_weights.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (output_row, weight_row))| {
            let source_row_start = y * width * 3;
            let mask_row_start = y * width;
            let mut sums = [0.0f32; 3];
            let mut weight = 0u32;
            for offset in 0..window_size {
                let x = offset.saturating_sub(radius).min(width - 1);
                if mask_pixels[mask_row_start + x] > 0 {
                    weight += 1;
                    let start = source_row_start + x * 3;
                    for channel in 0..3 {
                        sums[channel] += source_pixels[start + channel];
                    }
                }
            }
            let write_pixel = |x: usize,
                               output_row: &mut [f32],
                               weight_row: &mut [u32],
                               sums: [f32; 3],
                               weight: u32| {
                let start = x * 3;
                if weight > 0 {
                    output_row[start..start + 3].copy_from_slice(&sums);
                }
                weight_row[x] = weight;
            };
            write_pixel(0, output_row, weight_row, sums, weight);
            for x in 1..width {
                let add_x = (x + radius).min(width - 1);
                let remove_x = x.saturating_sub(radius + 1);
                if mask_pixels[mask_row_start + add_x] > 0 {
                    weight += 1;
                    let start = source_row_start + add_x * 3;
                    for channel in 0..3 {
                        sums[channel] += source_pixels[start + channel];
                    }
                }
                if mask_pixels[mask_row_start + remove_x] > 0 {
                    weight = weight.saturating_sub(1);
                    let start = source_row_start + remove_x * 3;
                    for channel in 0..3 {
                        sums[channel] -= source_pixels[start + channel];
                    }
                }
                write_pixel(x, output_row, weight_row, sums, weight);
            }
        });

    let mut output = vec![0.0f32; pixel_count * 3];
    for x in 0..width {
        let mut sums = [0.0f32; 3];
        let mut weight = 0u32;
        for offset in 0..window_size {
            let y = offset.saturating_sub(radius).min(height - 1);
            let index = y * width + x;
            if horizontal_weights[index] > 0 {
                weight += horizontal_weights[index];
                let start = index * 3;
                for channel in 0..3 {
                    sums[channel] += horizontal_values[start + channel];
                }
            }
        }
        let write_pixel = |y: usize, output: &mut [f32], sums: [f32; 3], weight: u32| {
            let index = (y * width + x) * 3;
            if weight > 0 {
                for channel in 0..3 {
                    output[index + channel] = sums[channel] / weight as f32;
                }
            } else {
                let source_index = index;
                output[index..index + 3]
                    .copy_from_slice(&source_pixels[source_index..source_index + 3]);
            }
        };
        write_pixel(0, &mut output, sums, weight);
        for y in 1..height {
            let add_y = (y + radius).min(height - 1);
            let remove_y = y.saturating_sub(radius + 1);
            let add_index = add_y * width + x;
            let remove_index = remove_y * width + x;
            if horizontal_weights[add_index] > 0 {
                weight += horizontal_weights[add_index];
                let start = add_index * 3;
                for channel in 0..3 {
                    sums[channel] += horizontal_values[start + channel];
                }
            }
            if horizontal_weights[remove_index] > 0 {
                weight = weight.saturating_sub(horizontal_weights[remove_index]);
                let start = remove_index * 3;
                for channel in 0..3 {
                    sums[channel] -= horizontal_values[start + channel];
                }
            }
            write_pixel(y, &mut output, sums, weight);
        }
    }

    Rgb32FImage::from_raw(width as u32, height as u32, output)
        .expect("masked RGB blur dimensions must match")
}

pub(super) fn masked_box_blur_focus_map(
    source: &[f32],
    mask: &[u8],
    width: u32,
    height: u32,
    radius: usize,
) -> Vec<f32> {
    let width = width as usize;
    let height = height as usize;
    if width == 0 || height == 0 || source.len() != width * height || mask.len() != source.len() {
        return source.to_vec();
    }
    if radius == 0 {
        return source.to_vec();
    }

    let window_size = radius.saturating_mul(2).saturating_add(1);
    let mut horizontal_values = vec![0.0f32; source.len()];
    let mut horizontal_weights = vec![0u32; source.len()];
    horizontal_values
        .par_chunks_mut(width)
        .zip(horizontal_weights.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (output_row, weight_row))| {
            let row_start = y * width;
            let mut sum = 0.0f32;
            let mut weight = 0u32;
            for offset in 0..window_size {
                let x = offset.saturating_sub(radius).min(width - 1);
                if mask[row_start + x] > 0 {
                    sum += source[row_start + x];
                    weight += 1;
                }
            }
            output_row[0] = sum;
            weight_row[0] = weight;
            for x in 1..width {
                let add_x = (x + radius).min(width - 1);
                let remove_x = x.saturating_sub(radius + 1);
                if mask[row_start + add_x] > 0 {
                    sum += source[row_start + add_x];
                    weight += 1;
                }
                if mask[row_start + remove_x] > 0 {
                    sum -= source[row_start + remove_x];
                    weight = weight.saturating_sub(1);
                }
                output_row[x] = sum;
                weight_row[x] = weight;
            }
        });

    let mut output = vec![0.0f32; source.len()];
    for x in 0..width {
        let mut sum = 0.0f32;
        let mut weight = 0u32;
        for offset in 0..window_size {
            let y = offset.saturating_sub(radius).min(height - 1);
            let index = y * width + x;
            sum += horizontal_values[index];
            weight += horizontal_weights[index];
        }
        let first_index = x;
        output[first_index] = if weight > 0 {
            sum / weight as f32
        } else {
            source[first_index]
        };
        for y in 1..height {
            let add_y = (y + radius).min(height - 1);
            let remove_y = y.saturating_sub(radius + 1);
            let add_index = add_y * width + x;
            let remove_index = remove_y * width + x;
            sum += horizontal_values[add_index] - horizontal_values[remove_index];
            weight += horizontal_weights[add_index];
            weight = weight.saturating_sub(horizontal_weights[remove_index]);
            let index = y * width + x;
            output[index] = if weight > 0 {
                sum / weight as f32
            } else {
                source[index]
            };
        }
    }
    output
}

pub(super) fn multiband_blend(
    base: Rgb32FImage,
    candidate: Rgb32FImage,
    mask: GrayImage,
    low_frequency_mask: Option<GrayImage>,
    max_bands: usize,
    hard_finest_band: bool,
) -> Rgb32FImage {
    let (width, height) = base.dimensions();
    let mut current_base = base;
    let mut current_candidate = candidate;
    let mut current_mask = mask;
    let mut current_low_frequency_mask = low_frequency_mask.unwrap_or_else(|| current_mask.clone());
    let mut base_laplacian = Vec::new();
    let mut candidate_laplacian = Vec::new();
    let mut masks = Vec::new();
    let mut low_frequency_masks = Vec::new();

    while base_laplacian.len() + 1 < max_bands {
        // Continue far enough for the coarsest band to absorb broad illumination and
        // vignetting differences. Stopping at 32px left low-frequency exposure steps
        // visible even though the high-frequency seam itself was well placed.
        if current_base.width() <= 4 || current_base.height() <= 4 {
            break;
        }
        let next_base = downsample_rgb_half(&current_base);
        let next_candidate = downsample_rgb_half(&current_candidate);
        let next_mask = downsample_mask_half(&current_mask);
        let next_low_frequency_mask = downsample_mask_half(&current_low_frequency_mask);
        base_laplacian.push(subtract_upsampled_rgb(&current_base, &next_base));
        candidate_laplacian.push(subtract_upsampled_rgb(&current_candidate, &next_candidate));
        masks.push(current_mask);
        low_frequency_masks.push(current_low_frequency_mask);
        current_base = next_base;
        current_candidate = next_candidate;
        current_mask = next_mask;
        current_low_frequency_mask = next_low_frequency_mask;
    }

    let mut reconstructed = combine_rgb(
        &current_base,
        &current_candidate,
        &current_low_frequency_mask,
        false,
    );
    for level in (0..base_laplacian.len()).rev() {
        let blend_mask = if level >= PANORAMA_GLOBAL_TONE_FIRST_BAND {
            &low_frequency_masks[level]
        } else {
            &masks[level]
        };
        let blended_detail = combine_rgb(
            &base_laplacian[level],
            &candidate_laplacian[level],
            blend_mask,
            // Focus ownership is a source-selection problem. Once a seam has
            // been chosen, averaging middle-frequency detail across it can
            // create a second contour when the two registered samples differ
            // by even a fraction of a pixel. Keep all visible detail bands on
            // one side; only the broad tone bands may feather continuously.
            hard_finest_band && level < PANORAMA_GLOBAL_TONE_FIRST_BAND,
        );
        reconstructed = upsample_and_add_rgb(&reconstructed, &blended_detail);
    }
    if reconstructed.dimensions() == (width, height) {
        reconstructed
    } else {
        resize_rgb(&reconstructed, width, height)
    }
}
