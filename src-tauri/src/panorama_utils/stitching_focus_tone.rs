use super::*;

pub(crate) fn harmonize_focus_background_tone_with_owners(
    image: &mut Rgb32FImage,
    image_mask: &GrayImage,
    foreground_mask: &GrayImage,
    owner_map: &GrayImage,
) {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || image_mask.dimensions() != (width, height)
        || foreground_mask.dimensions() != (width, height)
        || owner_map.dimensions() != (width, height)
    {
        return;
    }

    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(width, height, width.max(height));
    let analysis_image = resize_rgb(image, analysis_width, analysis_height);
    let analysis_mask = resize_binary_mask(image_mask, analysis_width, analysis_height);
    let analysis_foreground = resize_binary_mask(foreground_mask, analysis_width, analysis_height);
    let foreground_envelope = build_focus_background_foreground_envelope(
        &analysis_image,
        &analysis_mask,
        &analysis_foreground,
    );
    let background = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let index = y as usize * analysis_width as usize + x as usize;
        let start = index * 3;
        let pixel = &analysis_image.as_raw()[start..start + 3];
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0
                && foreground_envelope.as_raw()[index] == 0
                && focus_stack_pixel_is_canvas_like(pixel)
                && !focus_stack_pixel_is_tone_foreground(pixel),
        ) * 255])
    });
    let background_pixels = background
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if background_pixels < 128 {
        return;
    }

    let owner_analysis = image::imageops::resize(
        owner_map,
        analysis_width,
        analysis_height,
        image::imageops::FilterType::Nearest,
    );
    let sample_step = ((u64::from(analysis_width) * u64::from(analysis_height) / 120_000) as f64)
        .sqrt()
        .ceil()
        .max(1.0) as usize;
    let mut global_channels = [Vec::new(), Vec::new(), Vec::new()];
    let mut owner_channels: Vec<[Vec<f32>; 3]> = (0..=u8::MAX as usize)
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    for y in (0..analysis_height as usize).step_by(sample_step) {
        for x in (0..analysis_width as usize).step_by(sample_step) {
            let index = y * analysis_width as usize + x;
            if background.as_raw()[index] == 0 {
                continue;
            }
            let start = index * 3;
            let owner = owner_analysis.as_raw()[index] as usize;
            for channel in 0..3 {
                let value = analysis_image.as_raw()[start + channel];
                global_channels[channel].push(value);
                if owner > 0 {
                    owner_channels[owner][channel].push(value);
                }
            }
        }
    }
    let global_tone = [
        median_f32(&mut global_channels[0]).unwrap_or(0.0),
        median_f32(&mut global_channels[1]).unwrap_or(0.0),
        median_f32(&mut global_channels[2]).unwrap_or(0.0),
    ];
    let mut owner_corrections = vec![[0.0f32; 3]; u8::MAX as usize + 1];
    let mut corrected_owner_count = 0usize;
    let mut rejected_owner_count = 0usize;
    for (owner, channels) in owner_channels.iter_mut().enumerate().skip(1) {
        if channels[0].len() < FOCUS_COLOR_MIN_SAMPLES {
            continue;
        }
        // A source owner may cover a different painted region from the global
        // median. Its channel median is then content, not exposure. Require a
        // low robust spread before treating the owner as a tone reference.
        let mut owner_medians = [0.0f32; 3];
        let mut inconsistent = false;
        for channel in 0..3 {
            if channels[channel].len() < FOCUS_COLOR_MIN_SAMPLES {
                inconsistent = true;
                break;
            }
            owner_medians[channel] =
                median_f32(&mut channels[channel].clone()).unwrap_or(global_tone[channel]);
            let mut deviations = channels[channel]
                .iter()
                .map(|value| (*value - owner_medians[channel]).abs())
                .collect::<Vec<_>>();
            let mad = median_f32(&mut deviations).unwrap_or(0.0);
            if !mad.is_finite() || mad > FOCUS_OWNER_TONE_MAX_MAD {
                inconsistent = true;
                break;
            }
        }
        if inconsistent {
            rejected_owner_count += 1;
            continue;
        }
        let mut has_adjustment = false;
        for channel in 0..3 {
            let correction = (global_tone[channel] - owner_medians[channel]).clamp(
                -FOCUS_OWNER_TONE_MAX_ADJUSTMENT,
                FOCUS_OWNER_TONE_MAX_ADJUSTMENT,
            );
            owner_corrections[owner][channel] = correction;
            has_adjustment |= correction.abs() >= 0.002;
        }
        if has_adjustment {
            corrected_owner_count += 1;
        }
    }
    if std::env::var_os("RAW_EDITOR_FOCUS_TONE_DIAGNOSTICS").is_some() {
        let summary = owner_corrections
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, correction)| correction.iter().any(|value| value.abs() >= 0.001))
            .map(|(owner, correction)| {
                format!(
                    "{}:[{:.3},{:.3},{:.3}]",
                    owner, correction[0], correction[1], correction[2]
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!("  - Focus owner tone corrections: {summary}");
    }
    // The owner median is the exposure estimate. Do not derive a second
    // correction from the image's own low-frequency content: on artwork that
    // content is real paper ageing and brush tone, so pulling it toward one
    // global median creates the foggy, low-contrast result this pass is meant
    // to prevent. Extend each measured owner correction over that owner's
    // complete image and blur only the owner boundary. This preserves local
    // contrast while still removing a source-sized exposure step.
    let mut correction_pixels =
        vec![0.0f32; analysis_width as usize * analysis_height as usize * 3];
    for index in 0..background.as_raw().len() {
        if analysis_mask.as_raw()[index] == 0 {
            continue;
        }
        let start = index * 3;
        let owner_correction = owner_corrections[owner_analysis.as_raw()[index] as usize];
        for channel in 0..3 {
            correction_pixels[start + channel] = owner_correction[channel];
        }
    }
    let correction_image =
        Rgb32FImage::from_raw(analysis_width, analysis_height, correction_pixels)
            .expect("focus owner correction dimensions must match");
    let correction_radius = (analysis_width.max(analysis_height) as f32 * 0.006)
        .round()
        .clamp(4.0, 16.0) as usize;
    let smoothed_correction =
        masked_box_blur_rgb(&correction_image, &analysis_mask, correction_radius);
    let background_values = analysis_mask
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let background_radius = (analysis_width.max(analysis_height) as f32 * 0.006)
        .round()
        .clamp(4.0, 16.0) as usize;
    let background_weight = box_blur_focus_map(
        &background_values,
        analysis_width,
        analysis_height,
        background_radius,
    );
    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize;
    let correction_ref = smoothed_correction.as_raw();
    let background_weight_ref = &background_weight;
    let image_mask_ref = image_mask.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            for x in 0..width as usize {
                if image_mask_ref[y * width as usize + x] == 0 {
                    continue;
                }
                let start = x * 3;
                if focus_stack_pixel_is_tone_foreground(&row[start..start + 3]) {
                    continue;
                }
                let x_sample = x_samples[x];
                let top_weight = background_weight_ref
                    [y_sample.lower * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + background_weight_ref[y_sample.lower * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let bottom_weight = background_weight_ref
                    [y_sample.upper * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + background_weight_ref[y_sample.upper * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let background_weight = (top_weight * (1.0 - y_sample.upper_weight)
                    + bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                if background_weight <= 0.001 {
                    continue;
                }
                let top_left = &correction_ref[(y_sample.lower * analysis_stride + x_sample.lower)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.lower) * 3 + 3];
                let top_right = &correction_ref[(y_sample.lower * analysis_stride + x_sample.upper)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.upper) * 3 + 3];
                let bottom_left =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.lower) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.lower) * 3 + 3];
                let bottom_right =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.upper) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.upper) * 3 + 3];
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_sample.upper_weight)
                        + top_right[channel] * x_sample.upper_weight;
                    let bottom = bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + bottom_right[channel] * x_sample.upper_weight;
                    row[start + channel] = (row[start + channel]
                        + (top * (1.0 - y_sample.upper_weight) + bottom * y_sample.upper_weight)
                            * background_weight)
                        .clamp(0.0, 1.0);
                }
            }
        });
    println!(
        "  - Owner-based background harmonization: samples={} owners_adjusted={} owners_rejected={} radius={}px",
        background_pixels, corrected_owner_count, rejected_owner_count, correction_radius
    );
}

pub(crate) fn harmonize_focus_background_with_reference(
    image: &mut Rgb32FImage,
    image_mask: &GrayImage,
    foreground_mask: &GrayImage,
    reference_image: &Rgb32FImage,
    reference_mask: &GrayImage,
    owner_map: &GrayImage,
    owner_corrections: &[[f32; 3]],
) {
    let (width, height) = image.dimensions();
    let (analysis_width, analysis_height) = reference_image.dimensions();
    if width == 0
        || height == 0
        || analysis_width == 0
        || analysis_height == 0
        || image_mask.dimensions() != (width, height)
        || foreground_mask.dimensions() != (width, height)
        || reference_mask.dimensions() != (analysis_width, analysis_height)
        || owner_map.dimensions() != (width, height)
        || owner_corrections.is_empty()
    {
        return;
    }

    let analysis_image = resize_rgb(image, analysis_width, analysis_height);
    let analysis_mask = resize_binary_mask(image_mask, analysis_width, analysis_height);
    let analysis_foreground = resize_binary_mask(foreground_mask, analysis_width, analysis_height);
    let foreground_envelope = build_focus_background_foreground_envelope(
        &analysis_image,
        &analysis_mask,
        &analysis_foreground,
    );
    let background = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let index = y as usize * analysis_width as usize + x as usize;
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0
                && reference_mask.as_raw()[index] > 0
                && foreground_envelope.as_raw()[index] == 0,
        ) * 255])
    });
    let background_pixels = background
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if background_pixels < 128 {
        return;
    }

    let all_background = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let index = y as usize * analysis_width as usize + x as usize;
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0 && foreground_envelope.as_raw()[index] == 0,
        ) * 255])
    });
    let owner_analysis = image::imageops::resize(
        owner_map,
        analysis_width,
        analysis_height,
        image::imageops::FilterType::Nearest,
    );
    let owner_correction_pixels = all_background
        .as_raw()
        .iter()
        .enumerate()
        .flat_map(|(index, &value)| {
            if value == 0 {
                return [0.0; 3];
            }
            owner_corrections
                .get(owner_analysis.as_raw()[index] as usize)
                .copied()
                .unwrap_or([0.0; 3])
        })
        .collect::<Vec<_>>();
    let owner_correction_image =
        Rgb32FImage::from_raw(analysis_width, analysis_height, owner_correction_pixels)
            .expect("focus owner correction dimensions must match");
    let owner_correction_radius = (analysis_width.max(analysis_height) as f32 * 0.06)
        .round()
        .clamp(24.0, 96.0) as usize;
    let smoothed_owner_correction = masked_box_blur_rgb(
        &owner_correction_image,
        &all_background,
        owner_correction_radius,
    );

    // The reference is an average of every source's canvas-only pixels after a
    // robust per-frame offset. Compare it to the selected focus image only at a
    // broad scale. The final image keeps its own high-frequency samples, so the
    // operation cannot soften a face, sleeve, brush line, or the canvas weave.
    let focus_low = masked_box_blur_rgb(
        &analysis_image,
        &background,
        FOCUS_BACKGROUND_TONE_LOCAL_RADIUS,
    );
    let reference_low = masked_box_blur_rgb(
        reference_image,
        &background,
        FOCUS_BACKGROUND_TONE_LOCAL_RADIUS,
    );
    let focus_pixels = focus_low.as_raw();
    let reference_pixels = reference_low.as_raw();
    let mut correction_pixels =
        vec![0.0f32; analysis_width as usize * analysis_height as usize * 3];
    for index in 0..background.as_raw().len() {
        if background.as_raw()[index] == 0 {
            continue;
        }
        let start = index * 3;
        for channel in 0..3 {
            correction_pixels[start + channel] =
                (reference_pixels[start + channel] - focus_pixels[start + channel]).clamp(
                    -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                    FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                );
        }
    }
    let correction_image =
        Rgb32FImage::from_raw(analysis_width, analysis_height, correction_pixels)
            .expect("focus reference correction dimensions must match");
    let correction_radius = (analysis_width.max(analysis_height) as f32 * 0.06)
        .round()
        .clamp(24.0, 96.0) as usize;
    let smoothed_correction =
        masked_box_blur_rgb(&correction_image, &background, correction_radius);
    let reference_background_values = background
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let all_background_values = all_background
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let background_radius = (analysis_width.max(analysis_height) as f32 * 0.006)
        .round()
        .clamp(4.0, 16.0) as usize;
    let reference_background_weight = box_blur_focus_map(
        &reference_background_values,
        analysis_width,
        analysis_height,
        background_radius,
    );
    let all_background_weight = box_blur_focus_map(
        &all_background_values,
        analysis_width,
        analysis_height,
        background_radius,
    );
    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize;
    let correction_ref = smoothed_correction.as_raw();
    let owner_correction_ref = smoothed_owner_correction.as_raw();
    let reference_background_weight_ref = &reference_background_weight;
    let all_background_weight_ref = &all_background_weight;
    let image_mask_ref = image_mask.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            for x in 0..width as usize {
                if image_mask_ref[y * width as usize + x] == 0 {
                    continue;
                }
                let x_sample = x_samples[x];
                let top_weight = all_background_weight_ref
                    [y_sample.lower * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + all_background_weight_ref[y_sample.lower * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let bottom_weight = all_background_weight_ref
                    [y_sample.upper * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + all_background_weight_ref[y_sample.upper * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let all_background_weight = (top_weight * (1.0 - y_sample.upper_weight)
                    + bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                if all_background_weight <= 0.001 {
                    continue;
                }
                let reference_top_weight = reference_background_weight_ref
                    [y_sample.lower * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + reference_background_weight_ref
                        [y_sample.lower * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let reference_bottom_weight = reference_background_weight_ref
                    [y_sample.upper * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + reference_background_weight_ref
                        [y_sample.upper * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let reference_background_weight = (reference_top_weight
                    * (1.0 - y_sample.upper_weight)
                    + reference_bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                let top_left = &correction_ref[(y_sample.lower * analysis_stride + x_sample.lower)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.lower) * 3 + 3];
                let top_right = &correction_ref[(y_sample.lower * analysis_stride + x_sample.upper)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.upper) * 3 + 3];
                let bottom_left =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.lower) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.lower) * 3 + 3];
                let bottom_right =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.upper) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.upper) * 3 + 3];
                let start = x * 3;
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_sample.upper_weight)
                        + top_right[channel] * x_sample.upper_weight;
                    let bottom = bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + bottom_right[channel] * x_sample.upper_weight;
                    let delta = (top * (1.0 - y_sample.upper_weight)
                        + bottom * y_sample.upper_weight)
                        * reference_background_weight;
                    let owner_top_left = &owner_correction_ref[(y_sample.lower * analysis_stride
                        + x_sample.lower)
                        * 3
                        ..(y_sample.lower * analysis_stride + x_sample.lower) * 3 + 3];
                    let owner_top_right = &owner_correction_ref[(y_sample.lower * analysis_stride
                        + x_sample.upper)
                        * 3
                        ..(y_sample.lower * analysis_stride + x_sample.upper) * 3 + 3];
                    let owner_bottom_left = &owner_correction_ref[(y_sample.upper * analysis_stride
                        + x_sample.lower)
                        * 3
                        ..(y_sample.upper * analysis_stride + x_sample.lower) * 3 + 3];
                    let owner_bottom_right =
                        &owner_correction_ref[(y_sample.upper * analysis_stride + x_sample.upper)
                            * 3
                            ..(y_sample.upper * analysis_stride + x_sample.upper) * 3 + 3];
                    let owner_top = owner_top_left[channel] * (1.0 - x_sample.upper_weight)
                        + owner_top_right[channel] * x_sample.upper_weight;
                    let owner_bottom = owner_bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + owner_bottom_right[channel] * x_sample.upper_weight;
                    let owner_delta = (owner_top * (1.0 - y_sample.upper_weight)
                        + owner_bottom * y_sample.upper_weight)
                        * all_background_weight;
                    row[start + channel] =
                        (row[start + channel] + owner_delta + delta).clamp(0.0, 1.0);
                }
            }
        });
    println!(
        "  - Reference background harmonization: samples={} radius={}px owner_radius={}px",
        background_pixels, correction_radius, owner_correction_radius
    );
}
