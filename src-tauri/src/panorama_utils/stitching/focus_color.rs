//! Focus-stack colour correction between layers.
//!
//! Estimates bounded per-cell gain/offset fields on shared canvas evidence and
//! applies them before ownership is decided.

use image::GrayImage;
use image::Rgb32FImage;
use rayon::prelude::*;

use super::{
    FOCUS_COLOR_CELL_SIZE, FOCUS_COLOR_MAX_GAIN, FOCUS_COLOR_MAX_OFFSET,
    FOCUS_COLOR_MAX_RELAXED_LUMA, FOCUS_COLOR_MIN_CELL_SAMPLES, FOCUS_COLOR_MIN_CHANNEL_VALUE,
    FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MIN_SAMPLES, FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE,
    FOCUS_COLOR_PROPAGATION_RADIUS_CELLS, FOCUS_COLOR_SAMPLE_BUDGET,
    FOCUS_COLOR_SPATIAL_OFFSET_THRESHOLD, FOCUS_COLOR_SPATIAL_VARIATION_THRESHOLD,
    RenderedFocusLayer, focus_analysis_dimensions, focus_stack_pixel_is_canvas_like,
    linear_samples, luminance, median_f32, resize_binary_mask, resize_rgb,
};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct FocusColorCorrection {
    pub(super) gains: [f32; 3],
    pub(super) offsets: [f32; 3],
    pub(super) cell_size: u32,
    pub(super) grid_width: usize,
    pub(super) grid_height: usize,
    pub(super) spatial_gains: Option<Vec<[f32; 3]>>,
    pub(super) spatial_offsets: Option<Vec<[f32; 3]>>,
}

impl FocusColorCorrection {
    pub(super) const IDENTITY: Self = Self {
        gains: [1.0; 3],
        offsets: [0.0; 3],
        cell_size: 1,
        grid_width: 0,
        grid_height: 0,
        spatial_gains: None,
        spatial_offsets: None,
    };

    pub(super) fn interpolate_field(&self, field: &[[f32; 3]], x: u32, y: u32) -> [f32; 3] {
        if self.cell_size == 0 || self.grid_width == 0 || self.grid_height == 0 {
            return [0.0; 3];
        }

        let grid_x = x as f64 / self.cell_size as f64;
        let grid_y = y as f64 / self.cell_size as f64;
        let x0 = (grid_x.floor() as usize).min(self.grid_width - 1);
        let y0 = (grid_y.floor() as usize).min(self.grid_height - 1);
        let x1 = (x0 + 1).min(self.grid_width - 1);
        let y1 = (y0 + 1).min(self.grid_height - 1);
        let tx = if x0 == x1 {
            0.0
        } else {
            (grid_x - x0 as f64).clamp(0.0, 1.0) as f32
        };
        let ty = if y0 == y1 {
            0.0
        } else {
            (grid_y - y0 as f64).clamp(0.0, 1.0) as f32
        };
        let value = |gx: usize, gy: usize| field[gy * self.grid_width + gx];
        let top = value(x0, y0);
        let top_right = value(x1, y0);
        let bottom = value(x0, y1);
        let bottom_right = value(x1, y1);
        let mut interpolated = [0.0f32; 3];
        for channel in 0..3 {
            let top_value = top[channel] * (1.0 - tx) + top_right[channel] * tx;
            let bottom_value = bottom[channel] * (1.0 - tx) + bottom_right[channel] * tx;
            interpolated[channel] = top_value * (1.0 - ty) + bottom_value * ty;
        }
        interpolated
    }

    pub(super) fn gain_at(&self, x: u32, y: u32) -> [f32; 3] {
        self.spatial_gains
            .as_ref()
            .map_or(self.gains, |gains| self.interpolate_field(gains, x, y))
    }

    pub(super) fn offset_at(&self, x: u32, y: u32) -> [f32; 3] {
        self.spatial_offsets
            .as_ref()
            .map_or(self.offsets, |offsets| {
                self.interpolate_field(offsets, x, y)
            })
    }
}

pub(super) fn estimate_focus_color_correction(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    canvas_only: bool,
) -> FocusColorCorrection {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0
        || layer_height == 0
        || base.dimensions() != base_mask.dimensions()
        || merged_foreground_mask.dimensions() != base.dimensions()
        || candidate.left >= base.width()
        || candidate.top >= base.height()
        || candidate.left + layer_width > base.width()
        || candidate.top + layer_height > base.height()
    {
        return FocusColorCorrection::IDENTITY;
    }

    let sample_area = u64::from(layer_width) * u64::from(layer_height);
    let sample_step = ((sample_area as f64 / FOCUS_COLOR_SAMPLE_BUDGET as f64)
        .sqrt()
        .ceil() as u32)
        .max(1);
    let grid_width = layer_width.div_ceil(FOCUS_COLOR_CELL_SIZE) as usize;
    let grid_height = layer_height.div_ceil(FOCUS_COLOR_CELL_SIZE) as usize;
    let cell_count = grid_width * grid_height;
    let mut luminance_ratios = Vec::new();
    let mut channel_ratios = [Vec::new(), Vec::new(), Vec::new()];
    let mut channel_values = [Vec::new(), Vec::new(), Vec::new()];
    let mut cell_luminance_ratios: Vec<Vec<f32>> = (0..cell_count).map(|_| Vec::new()).collect();
    let mut cell_channel_ratios: Vec<[Vec<f32>; 3]> = (0..cell_count)
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    let mut cell_channel_values: Vec<[Vec<(f32, f32)>; 3]> = (0..cell_count)
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    for y in (0..layer_height).step_by(sample_step as usize) {
        for x in (0..layer_width).step_by(sample_step as usize) {
            if candidate.mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let global_x = candidate.left + x;
            let global_y = candidate.top + y;
            if base_mask.get_pixel(global_x, global_y)[0] == 0 {
                continue;
            }
            let base_is_foreground = merged_foreground_mask.get_pixel(global_x, global_y)[0] > 0;
            let candidate_is_foreground = candidate.foreground_mask.get_pixel(x, y)[0] > 0;
            let relaxed_foreground_overlap = candidate.relaxed_foreground_mask.get_pixel(x, y)[0]
                > 0
                && base_is_foreground
                && candidate_is_foreground;
            // Foreground is normally excluded because a repeated stroke or an
            // occluder is not a safe colour reference. A long-edge consensus is
            // different: it is an explicitly matched, repeated physical edge,
            // so its shared overlap is useful for correcting the rail/holder
            // colour that the paper-only samples cannot observe. Keep ordinary
            // foreground protected even when a focus stack contains it.
            if (base_is_foreground || candidate_is_foreground) && !relaxed_foreground_overlap {
                continue;
            }
            let base_pixel = base.get_pixel(global_x, global_y);
            let candidate_pixel = candidate.image.get_pixel(x, y);
            if canvas_only
                && (!focus_stack_pixel_is_canvas_like(base_pixel.0.as_slice())
                    || !focus_stack_pixel_is_canvas_like(candidate_pixel.0.as_slice()))
            {
                continue;
            }
            let base_luma = luminance(base_pixel);
            let candidate_luma = luminance(candidate_pixel);
            let maximum_luma = if relaxed_foreground_overlap {
                FOCUS_COLOR_MAX_RELAXED_LUMA
            } else {
                0.92
            };
            if !(0.04..=maximum_luma).contains(&base_luma)
                || !(0.04..=maximum_luma).contains(&candidate_luma)
            {
                continue;
            }
            let luma_ratio = base_luma / candidate_luma;
            if !(0.55..=1.8).contains(&luma_ratio) {
                continue;
            }
            luminance_ratios.push(luma_ratio);
            let cell_index = (y / FOCUS_COLOR_CELL_SIZE) as usize * grid_width
                + (x / FOCUS_COLOR_CELL_SIZE) as usize;
            cell_luminance_ratios[cell_index].push(luma_ratio);
            for channel in 0..3 {
                let base_value = base_pixel[channel];
                let candidate_value = candidate_pixel[channel];
                if base_value < FOCUS_COLOR_MIN_CHANNEL_VALUE
                    || candidate_value < FOCUS_COLOR_MIN_CHANNEL_VALUE
                {
                    continue;
                }
                let ratio = base_value / candidate_value;
                if (0.55..=1.8).contains(&ratio) {
                    channel_ratios[channel].push(ratio);
                    channel_values[channel].push((base_value, candidate_value));
                    cell_channel_ratios[cell_index][channel].push(ratio);
                    cell_channel_values[cell_index][channel].push((base_value, candidate_value));
                }
            }
        }
    }

    let luma_gain = if luminance_ratios.len() >= FOCUS_COLOR_MIN_SAMPLES {
        median_f32(&mut luminance_ratios)
            .unwrap_or(1.0)
            .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN)
    } else {
        1.0
    };
    let mut gains = [luma_gain; 3];
    for (channel, ratios) in channel_ratios.iter_mut().enumerate() {
        if ratios.len() >= FOCUS_COLOR_MIN_SAMPLES {
            let channel_gain = median_f32(ratios)
                .unwrap_or(luma_gain)
                .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
            // Most exposure changes are achromatic. Let the measured channel
            // ratio correct a real white-balance shift, but keep it anchored to
            // the luminance ratio so ink/seal colour cannot dominate a sparse
            // overlap.
            gains[channel] = (channel_gain * 0.72 + luma_gain * 0.28)
                .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
        }
    }

    // A ratio-only correction cannot remove a small black-level or flare
    // offset. Estimate it after the bounded gain, using the same robust
    // overlap samples, and keep it deliberately small so paper texture and
    // ink density are not flattened.
    let mut offsets = [0.0f32; 3];
    for channel in 0..3 {
        let mut differences = channel_values[channel]
            .iter()
            .map(|(base_value, candidate_value)| *base_value - *candidate_value * gains[channel])
            .collect::<Vec<_>>();
        if differences.len() >= FOCUS_COLOR_MIN_SAMPLES {
            offsets[channel] = median_f32(&mut differences)
                .unwrap_or(0.0)
                .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
        }
    }

    let mut spatial_gains = vec![gains; cell_count];
    let mut spatial_offsets = vec![offsets; cell_count];
    let mut known_cells = vec![false; cell_count];
    let mut known_cell_count = 0usize;
    for cell_index in 0..cell_count {
        let luma_ratios = &mut cell_luminance_ratios[cell_index];
        if luma_ratios.len() < FOCUS_COLOR_MIN_CELL_SAMPLES {
            continue;
        }
        let local_luma_gain = median_f32(luma_ratios)
            .unwrap_or(luma_gain)
            .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
        let mut local_gains = [local_luma_gain; 3];
        for (channel, ratios) in cell_channel_ratios[cell_index].iter_mut().enumerate() {
            if ratios.len() >= FOCUS_COLOR_MIN_CELL_SAMPLES {
                let channel_gain = median_f32(ratios)
                    .unwrap_or(local_luma_gain)
                    .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
                local_gains[channel] = (channel_gain * 0.72 + local_luma_gain * 0.28)
                    .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
            }
        }
        spatial_gains[cell_index] = local_gains;
        for channel in 0..3 {
            let values = &cell_channel_values[cell_index][channel];
            if values.len() >= FOCUS_COLOR_MIN_CELL_SAMPLES {
                let mut differences = values
                    .iter()
                    .map(|(base_value, candidate_value)| {
                        *base_value - *candidate_value * local_gains[channel]
                    })
                    .collect::<Vec<_>>();
                spatial_offsets[cell_index][channel] = median_f32(&mut differences)
                    .unwrap_or(offsets[channel])
                    .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
            }
        }
        known_cells[cell_index] = true;
        known_cell_count += 1;
    }

    // Smooth only measured cells first. Unknown cells start at the global
    // correction, so an overlap cannot impose its colour cast on a newly added
    // region without further evidence.
    for _ in 0..2 {
        let mut smoothed = spatial_gains.clone();
        let mut smoothed_offsets = spatial_offsets.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let index = grid_y * grid_width + grid_x;
                if !known_cells[index] {
                    continue;
                }
                let mut weight_sum = 0.0f32;
                let mut weighted = [0.0f32; 3];
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = if dx == 0 && dy == 0 { 4.0 } else { 1.0 };
                        weight_sum += weight;
                        for channel in 0..3 {
                            weighted[channel] += spatial_gains[neighbor_index][channel] * weight;
                        }
                    }
                }
                if weight_sum > 0.0 {
                    for channel in 0..3 {
                        smoothed[index][channel] = (weighted[channel] / weight_sum)
                            .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
                    }
                }

                let mut offset_weight_sum = 0.0f32;
                let mut weighted_offsets = [0.0f32; 3];
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = if dx == 0 && dy == 0 { 4.0 } else { 1.0 };
                        offset_weight_sum += weight;
                        for channel in 0..3 {
                            weighted_offsets[channel] +=
                                spatial_offsets[neighbor_index][channel] * weight;
                        }
                    }
                }
                if offset_weight_sum > 0.0 {
                    for channel in 0..3 {
                        smoothed_offsets[index][channel] = (weighted_offsets[channel]
                            / offset_weight_sum)
                            .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
                    }
                }
            }
        }
        spatial_gains = smoothed;
        spatial_offsets = smoothed_offsets;
    }

    // The measured overlap is often a narrow strip at one side of a shifted
    // frame. Leaving every other cell at the global value creates a visible
    // low-frequency block even though the correction itself is bilinearly
    // interpolated. Propagate only into nearby unknown cells, with confidence
    // determined by distance to measured cells and capped well below 1.0.
    // This keeps the correction continuous at the overlap boundary while
    // preserving the global calibration in genuinely unobserved areas.
    if known_cell_count >= 2 {
        let radius = FOCUS_COLOR_PROPAGATION_RADIUS_CELLS.max(1);
        let mut propagated = spatial_gains.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let index = grid_y * grid_width + grid_x;
                if known_cells[index] {
                    continue;
                }

                let mut weight_sum = 0.0f32;
                let mut weighted = [0.0f32; 3];
                let mut nearest_distance = f32::INFINITY;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let distance = ((dx * dx + dy * dy) as f32).sqrt();
                        if distance > radius as f32 {
                            continue;
                        }
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = (1.0 - distance / (radius as f32 + 1.0)).powi(2);
                        weight_sum += weight;
                        nearest_distance = nearest_distance.min(distance);
                        for channel in 0..3 {
                            weighted[channel] += spatial_gains[neighbor_index][channel] * weight;
                        }
                    }
                }
                if weight_sum == 0.0 || !weight_sum.is_finite() {
                    continue;
                }

                let distance_confidence =
                    (1.0 - nearest_distance / (radius as f32 + 1.0)).clamp(0.0, 1.0);
                let confidence = (distance_confidence * FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE)
                    .clamp(0.0, FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE);
                for channel in 0..3 {
                    let local_value = weighted[channel] / weight_sum;
                    propagated[index][channel] = (gains[channel] * (1.0 - confidence)
                        + local_value * confidence)
                        .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
                }
            }
        }
        let mut propagated_offsets = spatial_offsets.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let index = grid_y * grid_width + grid_x;
                if known_cells[index] {
                    continue;
                }

                let mut weight_sum = 0.0f32;
                let mut weighted = [0.0f32; 3];
                let mut nearest_distance = f32::INFINITY;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let distance = ((dx * dx + dy * dy) as f32).sqrt();
                        if distance > radius as f32 {
                            continue;
                        }
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = (1.0 - distance / (radius as f32 + 1.0)).powi(2);
                        weight_sum += weight;
                        nearest_distance = nearest_distance.min(distance);
                        for channel in 0..3 {
                            weighted[channel] += spatial_offsets[neighbor_index][channel] * weight;
                        }
                    }
                }
                if weight_sum == 0.0 || !weight_sum.is_finite() {
                    continue;
                }

                let distance_confidence =
                    (1.0 - nearest_distance / (radius as f32 + 1.0)).clamp(0.0, 1.0);
                let confidence = (distance_confidence * FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE)
                    .clamp(0.0, FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE);
                for channel in 0..3 {
                    let local_value = weighted[channel] / weight_sum;
                    propagated_offsets[index][channel] = (offsets[channel] * (1.0 - confidence)
                        + local_value * confidence)
                        .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
                }
            }
        }
        spatial_gains = propagated;
        spatial_offsets = propagated_offsets;
    }

    let has_spatial_variation = known_cell_count >= 2
        && known_cells.iter().enumerate().any(|(index, known)| {
            *known
                && spatial_gains[index]
                    .iter()
                    .zip(gains.iter())
                    .any(|(local_gain, global_gain)| {
                        (local_gain - global_gain).abs() >= FOCUS_COLOR_SPATIAL_VARIATION_THRESHOLD
                    })
        });
    let has_spatial_offset_variation = known_cell_count >= 2
        && known_cells.iter().enumerate().any(|(index, known)| {
            *known
                && spatial_offsets[index].iter().zip(offsets.iter()).any(
                    |(local_offset, global_offset)| {
                        (local_offset - global_offset).abs() >= FOCUS_COLOR_SPATIAL_OFFSET_THRESHOLD
                    },
                )
        });
    FocusColorCorrection {
        gains,
        offsets,
        cell_size: FOCUS_COLOR_CELL_SIZE,
        grid_width,
        grid_height,
        spatial_gains: has_spatial_variation.then_some(spatial_gains),
        spatial_offsets: has_spatial_offset_variation.then_some(spatial_offsets),
    }
}

pub(super) fn apply_focus_color_correction(
    image: &mut Rgb32FImage,
    correction: &FocusColorCorrection,
    protected_mask: &GrayImage,
    canvas_only: bool,
) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return;
    }
    let has_protected_mask = protected_mask.dimensions() == (width, height);

    // Apply the exposure match to a low-frequency copy and add only that
    // correction back to the source. Applying gain/offset directly to every
    // source pixel compresses brush-stroke and weave contrast, which is not
    // acceptable for a high-resolution cultural-relic scan.
    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(width, height, width.max(height));
    if (analysis_width, analysis_height) == (width, height) {
        let width = width as usize;
        image
            .as_mut()
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(index, pixel)| {
                let x = (index % width) as u32;
                let y = (index / width) as u32;
                if has_protected_mask && protected_mask.get_pixel(x, y)[0] > 0 {
                    return;
                }
                if canvas_only && !focus_stack_pixel_is_canvas_like(pixel) {
                    return;
                }
                let gains = correction.gain_at(x, y);
                let offsets = correction.offset_at(x, y);
                for (channel, value) in pixel.iter_mut().enumerate() {
                    *value = (*value * gains[channel] + offsets[channel]).clamp(0.0, 1.0);
                }
            });
        return;
    }

    let analysis = resize_rgb(image, analysis_width, analysis_height);
    let analysis_protected = if has_protected_mask {
        resize_binary_mask(protected_mask, analysis_width, analysis_height)
    } else {
        GrayImage::new(analysis_width, analysis_height)
    };
    let mut low_frequency_delta = Rgb32FImage::new(analysis_width, analysis_height);
    let analysis_width_usize = analysis_width as usize;
    let source_width = width as f64;
    let source_height = height as f64;
    low_frequency_delta
        .as_mut()
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(index, delta)| {
            if analysis_protected.as_raw()[index] > 0
                || (canvas_only
                    && !focus_stack_pixel_is_canvas_like(
                        &analysis.as_raw()[index * 3..index * 3 + 3],
                    ))
            {
                return;
            }
            let x = (index % analysis_width_usize) as f64;
            let y = (index / analysis_width_usize) as f64;
            let source_x = ((x + 0.5) * source_width / analysis_width as f64)
                .floor()
                .min(source_width - 1.0) as u32;
            let source_y = ((y + 0.5) * source_height / analysis_height as f64)
                .floor()
                .min(source_height - 1.0) as u32;
            let gains = correction.gain_at(source_x, source_y);
            let offsets = correction.offset_at(source_x, source_y);
            let source = &analysis.as_raw()[index * 3..index * 3 + 3];
            for channel in 0..3 {
                let corrected =
                    (source[channel] * gains[channel] + offsets[channel]).clamp(0.0, 1.0);
                delta[channel] = corrected - source[channel];
            }
        });
    // The correction field is already low-frequency. Sample it directly with
    // bilinear interpolation instead of allocating and resizing another full
    // source-sized RGB image for every frame in a long stack.
    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize * 3;
    let low_frequency_delta = low_frequency_delta.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            let top_row = y_sample.lower * analysis_stride;
            let bottom_row = y_sample.upper * analysis_stride;
            let y_weight = y_sample.upper_weight;
            for (x, x_sample) in x_samples.iter().copied().enumerate() {
                let index = y * width as usize + x;
                if has_protected_mask && protected_mask.as_raw()[index] > 0 {
                    continue;
                }
                let top_left = &low_frequency_delta
                    [top_row + x_sample.lower * 3..top_row + x_sample.lower * 3 + 3];
                let top_right = &low_frequency_delta
                    [top_row + x_sample.upper * 3..top_row + x_sample.upper * 3 + 3];
                let bottom_left = &low_frequency_delta
                    [bottom_row + x_sample.lower * 3..bottom_row + x_sample.lower * 3 + 3];
                let bottom_right = &low_frequency_delta
                    [bottom_row + x_sample.upper * 3..bottom_row + x_sample.upper * 3 + 3];
                let start = x * 3;
                if canvas_only && !focus_stack_pixel_is_canvas_like(&row[start..start + 3]) {
                    continue;
                }
                let x_weight = x_sample.upper_weight;
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_weight) + top_right[channel] * x_weight;
                    let bottom =
                        bottom_left[channel] * (1.0 - x_weight) + bottom_right[channel] * x_weight;
                    let delta = top * (1.0 - y_weight) + bottom * y_weight;
                    row[start + channel] = (row[start + channel] + delta).clamp(0.0, 1.0);
                }
            }
        });
}
