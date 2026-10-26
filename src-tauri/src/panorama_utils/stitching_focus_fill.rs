use super::*;

pub(crate) fn combine_rgb(
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

pub(crate) fn reflected_run_source_index(
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

pub(crate) const FOCUS_FILL_TONE_RADIUS: usize = 64;

pub(crate) fn fill_tone_adjusted_pixel(
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

pub(crate) fn horizontal_fill_edge_tone(
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

pub(crate) fn vertical_fill_edge_tone(
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

pub(crate) fn fill_invalid_runs_horizontally(
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

pub(crate) fn fill_invalid_runs_vertically(image: &mut Rgb32FImage, mask: &mut GrayImage) -> usize {
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

pub(crate) fn fill_focus_canvas_margins(
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

pub(crate) fn masked_box_blur_rgb(
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

pub(crate) fn masked_box_blur_focus_map(
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
