use super::*;

pub(crate) fn build_focus_background_reference_mask(
    analysis_image: &Rgb32FImage,
    analysis_mask: &GrayImage,
    existing_foreground_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = analysis_image.dimensions();
    if width == 0
        || height == 0
        || analysis_mask.dimensions() != (width, height)
        || existing_foreground_mask.dimensions() != (width, height)
    {
        return GrayImage::new(width, height);
    }

    // Do not use a colour threshold here. The same paper can be considerably
    // darker or lighter in different source frames, which is exactly the
    // exposure step this reference is meant to remove. The foreground
    // envelope is content-derived and is the only exclusion needed here.
    let foreground_envelope = build_focus_background_foreground_envelope(
        analysis_image,
        analysis_mask,
        existing_foreground_mask,
    );
    GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0 && foreground_envelope.as_raw()[index] == 0,
        ) * 255])
    })
}

pub(crate) fn update_focus_background_reference(
    reference_image: &mut Rgb32FImage,
    reference_mask: &mut GrayImage,
    reference_counts: &mut [u16],
    candidate_image: &Rgb32FImage,
    candidate_mask: &GrayImage,
) -> [f32; 3] {
    let (width, height) = reference_image.dimensions();
    if width == 0
        || height == 0
        || candidate_image.dimensions() != (width, height)
        || reference_mask.dimensions() != (width, height)
        || candidate_mask.dimensions() != (width, height)
        || reference_counts.len() != width as usize * height as usize
    {
        return [0.0; 3];
    }

    let reference_pixels = reference_image.as_raw();
    let reference_mask_pixels = reference_mask.as_raw();
    let candidate_pixels = candidate_image.as_raw();
    let candidate_mask_pixels = candidate_mask.as_raw();
    let sample_step = ((u64::from(width) * u64::from(height) / 120_000) as f64)
        .sqrt()
        .ceil()
        .max(1.0) as usize;
    let mut channel_differences = [Vec::new(), Vec::new(), Vec::new()];
    for y in (0..height as usize).step_by(sample_step) {
        for x in (0..width as usize).step_by(sample_step) {
            let index = y * width as usize + x;
            if reference_mask_pixels[index] == 0 || candidate_mask_pixels[index] == 0 {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                channel_differences[channel]
                    .push(reference_pixels[start + channel] - candidate_pixels[start + channel]);
            }
        }
    }

    let correction = [
        median_f32(&mut channel_differences[0])
            .unwrap_or(0.0)
            .clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            ),
        median_f32(&mut channel_differences[1])
            .unwrap_or(0.0)
            .clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            ),
        median_f32(&mut channel_differences[2])
            .unwrap_or(0.0)
            .clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            ),
    ];

    let reference_pixels = reference_image.as_mut();
    let reference_mask_pixels = reference_mask.as_mut();
    for index in 0..candidate_mask_pixels.len() {
        if candidate_mask_pixels[index] == 0 {
            continue;
        }
        let start = index * 3;
        let mut normalized = [0.0f32; 3];
        for channel in 0..3 {
            normalized[channel] =
                (candidate_pixels[start + channel] + correction[channel]).clamp(0.0, 1.0);
        }
        let count = reference_counts[index];
        if count == 0 || reference_mask_pixels[index] == 0 {
            reference_pixels[start..start + 3].copy_from_slice(&normalized);
            reference_counts[index] = 1;
            reference_mask_pixels[index] = 255;
            continue;
        }
        let old_weight = f32::from(count);
        let next_weight = old_weight + 1.0;
        for channel in 0..3 {
            reference_pixels[start + channel] = (reference_pixels[start + channel] * old_weight
                + normalized[channel])
                / next_weight;
        }
        reference_counts[index] = count.saturating_add(1);
    }
    correction
}
