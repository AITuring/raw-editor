use super::*;

pub(crate) fn hard_select_focus_layer(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    debug_assert_eq!(candidate.mask.dimensions(), (layer_width, layer_height));
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));
    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    let base_stride = base_width as usize * 3;
    let layer_stride = layer_width as usize * 3;
    base.as_mut()
        .par_chunks_mut(base_stride)
        .zip(base_mask.as_mut().par_chunks_mut(base_width as usize))
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(candidate.image.as_raw().par_chunks(layer_stride))
        .zip(candidate.mask.as_raw().par_chunks(layer_width as usize))
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(
            |((((base_row, base_mask_row), candidate_row), candidate_mask_row), decision_row)| {
                for local_x in 0..layer_width as usize {
                    if candidate_mask_row[local_x] == 0 {
                        continue;
                    }
                    let global_x = candidate.left as usize + local_x;
                    let base_pixel_start = global_x * 3;
                    let candidate_pixel_start = local_x * 3;
                    if base_mask_row[global_x] == 0 || decision_row[local_x] > 0 {
                        base_row[base_pixel_start..base_pixel_start + 3].copy_from_slice(
                            &candidate_row[candidate_pixel_start..candidate_pixel_start + 3],
                        );
                    }
                    base_mask_row[global_x] = 255;
                }
            },
        );
}

/// Record the Ownership_Map entries that the *following*
/// [`hard_select_focus_layer`] call is about to create.
///
/// `base_mask` must be the coverage state before that call, because the
/// selection replaces a pixel when the candidate is valid and either the base
/// has no coverage yet or the focus decision prefers the candidate.  The
/// predicate below is that same condition, kept in lockstep with the copy in
/// `hard_select_focus_layer`; nothing here writes a pixel.
pub(crate) fn record_focus_layer_ownership(
    ownership: &mut [u16],
    base_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
    owner_id: u16,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    let (base_width, base_height) = base_mask.dimensions();
    if layer_width == 0
        || layer_height == 0
        || decision_mask.dimensions() != (layer_width, layer_height)
        || candidate.mask.dimensions() != (layer_width, layer_height)
        || ownership.len() != base_width as usize * base_height as usize
        || candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }
    ownership
        .par_chunks_mut(base_width as usize)
        .zip(base_mask.as_raw().par_chunks(base_width as usize))
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(candidate.mask.as_raw().par_chunks(layer_width as usize))
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(
            |(((ownership_row, base_mask_row), candidate_mask_row), decision_row)| {
                let start = candidate.left as usize;
                for local_x in 0..layer_width as usize {
                    let global_x = start + local_x;
                    if candidate_mask_row[local_x] > 0
                        && (base_mask_row[global_x] == 0 || decision_row[local_x] > 0)
                    {
                        ownership_row[global_x] = owner_id;
                    }
                }
            },
        );
}

pub(crate) fn mark_focus_owner_pixels(
    owner: &mut GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
    owner_id: u8,
) {
    let (layer_width, layer_height) = candidate.mask.dimensions();
    if layer_width == 0
        || layer_height == 0
        || decision_mask.dimensions() != (layer_width, layer_height)
        || candidate.left + layer_width > owner.width()
        || candidate.top + layer_height > owner.height()
    {
        return;
    }
    let owner_width = owner.width() as usize;
    owner
        .as_mut()
        .par_chunks_mut(owner_width)
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(candidate.mask.as_raw().par_chunks(layer_width as usize))
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(|((owner_row, candidate_row), decision_row)| {
            let start = candidate.left as usize;
            for local_x in 0..layer_width as usize {
                if candidate_row[local_x] > 0 && decision_row[local_x] > 0 {
                    owner_row[start + local_x] = owner_id;
                }
            }
        });
}

pub(crate) fn scaled_region_start(value: u32, source_size: u32, target_size: u32) -> u32 {
    ((value as u64 * target_size as u64) / source_size.max(1) as u64) as u32
}

pub(crate) fn scaled_region_end(value: u32, source_size: u32, target_size: u32) -> u32 {
    (value as u64 * target_size as u64).div_ceil(source_size.max(1) as u64) as u32
}

pub(crate) fn render_focus_analysis_layer(
    layer: &RenderedFocusLayer,
    out_width: u32,
    out_height: u32,
    analysis_width: u32,
    analysis_height: u32,
) -> RenderedFocusAnalysisLayer {
    let mut analysis = Rgb32FImage::new(analysis_width, analysis_height);
    let mut analysis_mask = GrayImage::new(analysis_width, analysis_height);
    let mut analysis_foreground_mask = GrayImage::new(analysis_width, analysis_height);
    let (layer_width, layer_height) = layer.image.dimensions();
    if layer_width == 0 || layer_height == 0 || out_width == 0 || out_height == 0 {
        return RenderedFocusAnalysisLayer {
            image: analysis,
            mask: analysis_mask,
            foreground_mask: analysis_foreground_mask,
            relaxed_foreground_mask: GrayImage::new(analysis_width, analysis_height),
        };
    }

    let left = scaled_region_start(layer.left, out_width, analysis_width).min(analysis_width);
    let top = scaled_region_start(layer.top, out_height, analysis_height).min(analysis_height);
    let right = scaled_region_end(
        layer.left.saturating_add(layer_width),
        out_width,
        analysis_width,
    )
    .min(analysis_width);
    let bottom = scaled_region_end(
        layer.top.saturating_add(layer_height),
        out_height,
        analysis_height,
    )
    .min(analysis_height);
    if left >= right || top >= bottom {
        return RenderedFocusAnalysisLayer {
            image: analysis,
            mask: analysis_mask,
            foreground_mask: analysis_foreground_mask,
            relaxed_foreground_mask: GrayImage::new(analysis_width, analysis_height),
        };
    }

    let resized_width = right - left;
    let resized_height = bottom - top;
    let resized = resize_rgb(&layer.image, resized_width, resized_height);
    let resized_mask = resize_binary_mask(&layer.mask, resized_width, resized_height);
    let resized_foreground_mask =
        resize_binary_mask(&layer.foreground_mask, resized_width, resized_height);
    let resized_relaxed_foreground_mask = resize_binary_mask(
        &layer.relaxed_foreground_mask,
        resized_width,
        resized_height,
    );
    let analysis_stride = analysis_width as usize * 3;
    let resized_stride = resized_width as usize * 3;
    analysis
        .as_mut()
        .par_chunks_mut(analysis_stride)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(resized.as_raw().par_chunks(resized_stride))
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize * 3;
            let end = start + resized_stride;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    analysis_mask
        .as_mut()
        .par_chunks_mut(analysis_width as usize)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(resized_mask.as_raw().par_chunks(resized_width as usize))
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    analysis_foreground_mask
        .as_mut()
        .par_chunks_mut(analysis_width as usize)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(
            resized_foreground_mask
                .as_raw()
                .par_chunks(resized_width as usize),
        )
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    let analysis_relaxed_stride = analysis_width as usize;
    let mut relaxed_foreground_mask = GrayImage::new(analysis_width, analysis_height);
    relaxed_foreground_mask
        .as_mut()
        .par_chunks_mut(analysis_relaxed_stride)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(
            resized_relaxed_foreground_mask
                .as_raw()
                .par_chunks(resized_width as usize),
        )
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    RenderedFocusAnalysisLayer {
        image: analysis,
        mask: analysis_mask,
        foreground_mask: analysis_foreground_mask,
        relaxed_foreground_mask,
    }
}

/// Upsample an analysis-resolution focus decision onto a layer's pixel grid.
///
/// Superseded on the station path by [`focus_cell_decision_for_layer`], which
/// decides at ownership cell resolution instead (需求 3.6).  Kept because it is
/// the only reader of the analysis decision and the shifted-mosaic guards still
/// maintain that decision; the comparison path needs it back when it is
/// re-enabled.
#[allow(dead_code)]
pub(crate) fn focus_decision_for_layer(
    analysis_decision: &GrayImage,
    layer: &RenderedFocusLayer,
    out_width: u32,
    out_height: u32,
) -> GrayImage {
    let (width, height) = layer.image.dimensions();
    let analysis_width = analysis_decision.width();
    let analysis_height = analysis_decision.height();
    GrayImage::from_fn(width, height, |x, y| {
        if out_width == 0 || out_height == 0 || analysis_width == 0 || analysis_height == 0 {
            return image::Luma([0]);
        }
        let global_x = layer.left.saturating_add(x);
        let global_y = layer.top.saturating_add(y);
        let analysis_x = ((global_x as u64 * analysis_width as u64) / out_width as u64)
            .min(analysis_width.saturating_sub(1) as u64) as u32;
        let analysis_y = ((global_y as u64 * analysis_height as u64) / out_height as u64)
            .min(analysis_height.saturating_sub(1) as u64) as u32;
        *analysis_decision.get_pixel(analysis_x, analysis_y)
    })
}

pub(crate) fn commit_focus_analysis_scores(
    best_focus: &mut [f32],
    candidate_focus: &[f32],
    candidate_analysis_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    full_resolution_decision: &GrayImage,
    out_width: u32,
    out_height: u32,
    analysis_width: u32,
    analysis_height: u32,
) {
    if out_width == 0
        || out_height == 0
        || analysis_width == 0
        || analysis_height == 0
        || candidate_analysis_mask.dimensions() != (analysis_width, analysis_height)
        || candidate_focus.len() != analysis_width as usize * analysis_height as usize
        || best_focus.len() != candidate_focus.len()
        || full_resolution_decision.dimensions() != candidate.image.dimensions()
    {
        return;
    }

    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0
        || layer_height == 0
        || candidate.left >= out_width
        || candidate.top >= out_height
        || candidate.left.saturating_add(layer_width) > out_width
        || candidate.top.saturating_add(layer_height) > out_height
    {
        return;
    }

    // The full-resolution ownership guards can veto part of an analysis cell.
    // Downsample the committed decision, rather than blindly copying the
    // low-resolution decision, so the score table remains tied to pixels that
    // really entered the merged image.
    let left = scaled_region_start(candidate.left, out_width, analysis_width).min(analysis_width);
    let top = scaled_region_start(candidate.top, out_height, analysis_height).min(analysis_height);
    let right = scaled_region_end(
        candidate.left.saturating_add(layer_width),
        out_width,
        analysis_width,
    )
    .min(analysis_width);
    let bottom = scaled_region_end(
        candidate.top.saturating_add(layer_height),
        out_height,
        analysis_height,
    )
    .min(analysis_height);
    if left >= right || top >= bottom {
        return;
    }
    let committed_width = right - left;
    let committed_height = bottom - top;
    let committed = resize_binary_mask(full_resolution_decision, committed_width, committed_height);
    let candidate_mask = candidate_analysis_mask.as_raw();
    for local_y in 0..committed_height as usize {
        for local_x in 0..committed_width as usize {
            let analysis_x = left as usize + local_x;
            let analysis_y = top as usize + local_y;
            let analysis_index = analysis_y * analysis_width as usize + analysis_x;
            if committed.as_raw()[local_y * committed_width as usize + local_x] > 0
                && candidate_mask[analysis_index] > 0
            {
                best_focus[analysis_index] = candidate_focus[analysis_index];
            }
        }
    }
}

pub(crate) fn resize_rgb(image: &Rgb32FImage, width: u32, height: u32) -> Rgb32FImage {
    image::imageops::resize(
        image,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Triangle,
    )
}

pub(crate) fn downsample_rgb_half(image: &Rgb32FImage) -> Rgb32FImage {
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

pub(crate) fn downsample_mask_half(mask: &GrayImage) -> GrayImage {
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
pub(crate) struct LinearSample {
    pub(crate) lower: usize,
    pub(crate) upper: usize,
    pub(crate) upper_weight: f32,
}

pub(crate) fn linear_samples(source_length: u32, target_length: u32) -> Vec<LinearSample> {
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

pub(crate) fn subtract_upsampled_rgb(fine: &Rgb32FImage, coarse: &Rgb32FImage) -> Rgb32FImage {
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

pub(crate) fn upsample_and_add_rgb(coarse: &Rgb32FImage, detail: &Rgb32FImage) -> Rgb32FImage {
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
