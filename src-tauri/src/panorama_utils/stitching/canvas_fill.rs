//! Canvas cropping and invalid-margin filling.
//!
//! Valid-coverage bounds and rectangles for the comparison crops, and the
//! tone-matched reflection fill of uncovered focus-canvas margins.

use image::GrayImage;
use image::Rgb32FImage;
use rayon::prelude::*;

pub(in crate::panorama_utils) fn crop_to_valid_rectangle(
    image: Rgb32FImage,
    mask: &GrayImage,
) -> Rgb32FImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return image;
    }

    let mut heights = vec![0usize; width as usize];
    let mut stack = Vec::with_capacity(width as usize + 1);
    let mut best_area = 0usize;
    let mut best_left = 0usize;
    let mut best_top = 0usize;
    let mut best_width = width as usize;
    let mut best_height = height as usize;

    for y in 0..height as usize {
        for (x, height_value) in heights.iter_mut().enumerate() {
            *height_value = if mask.get_pixel(x as u32, y as u32)[0] > 0 {
                height_value.saturating_add(1)
            } else {
                0
            };
        }

        stack.clear();
        for x in 0..=width as usize {
            let current_height = if x < width as usize { heights[x] } else { 0 };
            while let Some(&bar_index) = stack.last() {
                if heights[bar_index] <= current_height {
                    break;
                }
                stack.pop();
                let left = stack.last().map_or(0, |&index| index + 1);
                let rectangle_width = x - left;
                let rectangle_height = heights[bar_index];
                let area = rectangle_width * rectangle_height;
                if area > best_area {
                    best_area = area;
                    best_left = left;
                    best_top = y + 1 - rectangle_height;
                    best_width = rectangle_width;
                    best_height = rectangle_height;
                }
            }
            stack.push(x);
        }
    }

    if best_area == 0 {
        return image;
    }

    if best_left == 0
        && best_top == 0
        && best_width == width as usize
        && best_height == height as usize
    {
        return image;
    }

    image::imageops::crop_imm(
        &image,
        best_left as u32,
        best_top as u32,
        best_width as u32,
        best_height as u32,
    )
    .to_image()
}

/// Return the mask bounding box and the fraction of that box covered by valid
/// pixels.  A projective tile union can have transparent corner wedges even
/// though its outer bounds are useful; the coverage lets callers choose a
/// source-union crop only when those wedges are negligible.
pub(super) fn valid_mask_bounds_and_coverage(mask: &GrayImage) -> ((u32, u32, u32, u32), f32) {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return ((0, 0, 0, 0), 0.0);
    }
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut valid = 0u64;
    for y in 0..height {
        for x in 0..width {
            if mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            valid += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if valid == 0 {
        return ((0, 0, 0, 0), 0.0);
    }
    let bbox_width = u64::from(max_x - min_x + 1);
    let bbox_height = u64::from(max_y - min_y + 1);
    let coverage = valid as f32 / (bbox_width * bbox_height).max(1) as f32;
    ((min_x, min_y, max_x, max_y), coverage.clamp(0.0, 1.0))
}

pub(in crate::panorama_utils) fn crop_to_valid_bounds(
    image: Rgb32FImage,
    mask: &GrayImage,
) -> Rgb32FImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return image;
    }
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut covered = false;
    for y in 0..height {
        for x in 0..width {
            if mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            covered = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !covered {
        return image;
    }
    let crop_width = max_x - min_x + 1;
    let crop_height = max_y - min_y + 1;
    if min_x == 0 && min_y == 0 && crop_width == width && crop_height == height {
        return image;
    }
    image::imageops::crop_imm(&image, min_x, min_y, crop_width, crop_height).to_image()
}

pub(super) fn reflected_run_source_index(
    target: usize,
    limit: usize,
    left: Option<usize>,
    right: Option<usize>,
) -> usize {
    match (left, right) {
        (Some(left), Some(right)) => {
            let distance_from_left = target - left;
            let distance_from_right = right - target;
            if distance_from_left <= distance_from_right {
                left.saturating_sub(distance_from_left)
            } else {
                right
                    .saturating_add(distance_from_right)
                    .min(limit.saturating_sub(1))
            }
        }
        (Some(left), None) => left.saturating_sub(target - left),
        (None, Some(right)) => right
            .saturating_add(right - target)
            .min(limit.saturating_sub(1)),
        (None, None) => target,
    }
}

pub(super) const FOCUS_FILL_TONE_RADIUS: usize = 64;

pub(super) fn fill_tone_adjusted_pixel(
    pixel: [f32; 3],
    source_tone: [f32; 3],
    left_tone: [f32; 3],
    right_tone: [f32; 3],
    progress: f32,
) -> [f32; 3] {
    let progress = progress.clamp(0.0, 1.0);
    let blend = progress * progress * (3.0 - 2.0 * progress);
    let mut output = [0.0f32; 3];
    for channel in 0..3 {
        let target_tone = left_tone[channel] * (1.0 - blend) + right_tone[channel] * blend;
        output[channel] = (pixel[channel] + target_tone - source_tone[channel]).clamp(0.0, 1.0);
    }
    output
}

pub(super) fn horizontal_fill_edge_tone(
    image_row: &[f32],
    edge: usize,
    from_left: bool,
    radius: usize,
) -> [f32; 3] {
    let width = image_row.len() / 3;
    let radius = radius.max(1).min(width);
    let (start, end) = if from_left {
        (edge.saturating_sub(radius - 1), edge)
    } else {
        (edge, (edge + radius - 1).min(width.saturating_sub(1)))
    };
    let mut tone = [0.0f32; 3];
    let sample_count = (end - start + 1) as f32;
    for x in start..=end {
        let pixel_start = x * 3;
        for channel in 0..3 {
            tone[channel] += image_row[pixel_start + channel];
        }
    }
    for channel in &mut tone {
        *channel /= sample_count;
    }
    tone
}

pub(super) fn vertical_fill_edge_tone(
    image_pixels: &[f32],
    width: usize,
    edge: usize,
    x: usize,
    from_top: bool,
    radius: usize,
) -> [f32; 3] {
    let height = image_pixels.len() / (width * 3);
    let radius = radius.max(1).min(height);
    let (start, end) = if from_top {
        (edge.saturating_sub(radius - 1), edge)
    } else {
        (edge, (edge + radius - 1).min(height.saturating_sub(1)))
    };
    let mut tone = [0.0f32; 3];
    let sample_count = (end - start + 1) as f32;
    for y in start..=end {
        let pixel_start = (y * width + x) * 3;
        for channel in 0..3 {
            tone[channel] += image_pixels[pixel_start + channel];
        }
    }
    for channel in &mut tone {
        *channel /= sample_count;
    }
    tone
}

pub(super) fn fill_invalid_runs_horizontally(
    image: &mut Rgb32FImage,
    mask: &mut GrayImage,
) -> usize {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return 0;
    }
    let width = width as usize;
    let image_stride = width * 3;
    // Determinism (需求 14.6): the writes are partitioned by destination row and
    // the reduction accumulates `usize`, which is associative, so neither the
    // pixels nor the returned count depend on the thread count. A floating
    // point reduction in this position would need
    // `stack_pipeline::determinism::deterministic_sum`.
    image
        .as_mut()
        .par_chunks_mut(image_stride)
        .zip(mask.as_mut().par_chunks_mut(width))
        .map(|(image_row, mask_row)| {
            let mut filled = 0usize;
            let mut run_start = 0usize;
            while run_start < width {
                if mask_row[run_start] > 0 {
                    run_start += 1;
                    continue;
                }
                let mut run_end = run_start + 1;
                while run_end < width && mask_row[run_end] == 0 {
                    run_end += 1;
                }
                let left = run_start.checked_sub(1).filter(|&x| mask_row[x] > 0);
                let right = (run_end < width && mask_row[run_end] > 0).then_some(run_end);
                if left.is_none() && right.is_none() {
                    run_start = run_end;
                    continue;
                }
                let edge_tones = left.zip(right).map(|(left, right)| {
                    (
                        horizontal_fill_edge_tone(image_row, left, true, FOCUS_FILL_TONE_RADIUS),
                        horizontal_fill_edge_tone(image_row, right, false, FOCUS_FILL_TONE_RADIUS),
                    )
                });
                for target in run_start..run_end {
                    let mut source = reflected_run_source_index(target, width, left, right);
                    if mask_row[source] == 0 {
                        source = left.or(right).expect("a fill run has a valid neighbour");
                    }
                    let source_start = source * 3;
                    let target_start = target * 3;
                    let pixel = [
                        image_row[source_start],
                        image_row[source_start + 1],
                        image_row[source_start + 2],
                    ];
                    let pixel = edge_tones.map_or(pixel, |(left_tone, right_tone)| {
                        let source_tone =
                            if left.is_some_and(|left| target - left <= right.unwrap() - target) {
                                left_tone
                            } else {
                                right_tone
                            };
                        let progress =
                            (target - run_start + 1) as f32 / (run_end - run_start + 1) as f32;
                        fill_tone_adjusted_pixel(
                            pixel,
                            source_tone,
                            left_tone,
                            right_tone,
                            progress,
                        )
                    });
                    image_row[target_start..target_start + 3].copy_from_slice(&pixel);
                    mask_row[target] = 255;
                    filled += 1;
                }
                run_start = run_end;
            }
            filled
        })
        .sum()
}

pub(super) fn fill_invalid_runs_vertically(image: &mut Rgb32FImage, mask: &mut GrayImage) -> usize {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return 0;
    }
    let width = width as usize;
    let height = height as usize;
    let image_pixels = image.as_mut();
    let mask_pixels = mask.as_mut();
    let mut filled = 0usize;
    for x in 0..width {
        let mut run_start = 0usize;
        while run_start < height {
            if mask_pixels[run_start * width + x] > 0 {
                run_start += 1;
                continue;
            }
            let mut run_end = run_start + 1;
            while run_end < height && mask_pixels[run_end * width + x] == 0 {
                run_end += 1;
            }
            let top = run_start
                .checked_sub(1)
                .filter(|&y| mask_pixels[y * width + x] > 0);
            let bottom =
                (run_end < height && mask_pixels[run_end * width + x] > 0).then_some(run_end);
            if top.is_none() && bottom.is_none() {
                run_start = run_end;
                continue;
            }
            let edge_tones = top.zip(bottom).map(|(top, bottom)| {
                (
                    vertical_fill_edge_tone(
                        image_pixels,
                        width,
                        top,
                        x,
                        true,
                        FOCUS_FILL_TONE_RADIUS,
                    ),
                    vertical_fill_edge_tone(
                        image_pixels,
                        width,
                        bottom,
                        x,
                        false,
                        FOCUS_FILL_TONE_RADIUS,
                    ),
                )
            });
            for target in run_start..run_end {
                let mut source = reflected_run_source_index(target, height, top, bottom);
                if mask_pixels[source * width + x] == 0 {
                    source = top.or(bottom).expect("a fill run has a valid neighbour");
                }
                let source_start = (source * width + x) * 3;
                let target_start = (target * width + x) * 3;
                let pixel = [
                    image_pixels[source_start],
                    image_pixels[source_start + 1],
                    image_pixels[source_start + 2],
                ];
                let pixel = edge_tones.map_or(pixel, |(top_tone, bottom_tone)| {
                    let source_tone =
                        if top.is_some_and(|top| target - top <= bottom.unwrap() - target) {
                            top_tone
                        } else {
                            bottom_tone
                        };
                    let progress =
                        (target - run_start + 1) as f32 / (run_end - run_start + 1) as f32;
                    fill_tone_adjusted_pixel(pixel, source_tone, top_tone, bottom_tone, progress)
                });
                image_pixels[target_start..target_start + 3].copy_from_slice(&pixel);
                mask_pixels[target * width + x] = 255;
                filled += 1;
            }
            run_start = run_end;
        }
    }
    filled
}

pub(super) fn fill_focus_canvas_margins(
    image: &mut Rgb32FImage,
    mask: &mut GrayImage,
) -> (usize, usize) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return (0, 0);
    }
    let mut filled = 0usize;
    // Reflection extends the local canvas weave and broad tone from the nearest
    // valid edge. Alternating directions also fills the triangular corner areas
    // created by a projective scan without allocating another full-size image.
    for _ in 0..3 {
        filled += fill_invalid_runs_horizontally(image, mask);
        filled += fill_invalid_runs_vertically(image, mask);
        if !mask.as_raw().contains(&0) {
            break;
        }
    }
    let remaining = mask.as_raw().iter().filter(|&&value| value == 0).count();
    (filled, remaining)
}
