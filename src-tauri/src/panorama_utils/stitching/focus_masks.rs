//! Binary masks for focus foreground, background references and placement.
//!
//! Morphology, hole filling, component retention, background envelopes and the
//! placement of a layer or its foreground mask on the canvas.

use image::GrayImage;
use image::Rgb32FImage;
use rayon::prelude::*;

use super::{
    FOCUS_BACKGROUND_TONE_FOREGROUND_RADIUS_RATIO, FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
    RenderedFocusLayer, box_blur_focus_map, focus_stack_pixel_is_tone_foreground, median_f32,
};

pub(super) fn resize_binary_mask(mask: &GrayImage, width: u32, height: u32) -> GrayImage {
    image::imageops::resize(
        mask,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Nearest,
    )
}

pub(super) fn dilate_focus_binary_mask(mask: &GrayImage, radius: usize) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }
    let seed = mask
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let dilated = box_blur_focus_map(&seed, width, height, radius);
    GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        image::Luma([u8::from(dilated[index] > 0.0) * 255])
    })
}

pub(super) fn open_focus_binary_mask(mask: &GrayImage, radius: usize) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 || radius == 0 {
        return mask.clone();
    }
    let seed = mask
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let eroded = box_blur_focus_map(&seed, width, height, radius);
    let eroded = GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        image::Luma([u8::from(eroded[index] >= 0.999) * 255])
    });
    dilate_focus_binary_mask(&eroded, radius)
}

pub(super) fn fill_focus_foreground_holes(mask: &GrayImage) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return mask.clone();
    }
    let width = width as usize;
    let height = height as usize;
    let source = mask.as_raw();
    let mut outside = vec![false; source.len()];
    let mut pending = Vec::new();
    let enqueue = |index: usize, outside: &mut [bool], pending: &mut Vec<usize>| {
        if source[index] == 0 && !outside[index] {
            outside[index] = true;
            pending.push(index);
        }
    };
    for x in 0..width {
        enqueue(x, &mut outside, &mut pending);
        enqueue((height - 1) * width + x, &mut outside, &mut pending);
    }
    for y in 0..height {
        enqueue(y * width, &mut outside, &mut pending);
        enqueue(y * width + width - 1, &mut outside, &mut pending);
    }
    while let Some(index) = pending.pop() {
        let x = index % width;
        let y = index / width;
        if x > 0 {
            enqueue(index - 1, &mut outside, &mut pending);
        }
        if x + 1 < width {
            enqueue(index + 1, &mut outside, &mut pending);
        }
        if y > 0 {
            enqueue(index - width, &mut outside, &mut pending);
        }
        if y + 1 < height {
            enqueue(index + width, &mut outside, &mut pending);
        }
    }

    let mut filled = source.to_vec();
    for (index, value) in filled.iter_mut().enumerate() {
        if *value == 0 && !outside[index] {
            *value = 255;
        }
    }
    GrayImage::from_raw(width as u32, height as u32, filled)
        .expect("focus hole-filled foreground mask dimensions must match")
}

pub(super) fn retain_focus_foreground_components(
    foreground_mask: &GrayImage,
    seed_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = foreground_mask.dimensions();
    if (width, height) != seed_mask.dimensions() || width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let foreground = foreground_mask.as_raw();
    let seeds = seed_mask.as_raw();
    let mut visited = vec![false; width_usize * height_usize];
    let mut retained = vec![0u8; width_usize * height_usize];
    let minimum_component_size = ((width as u64 * height as u64) as f64 * 0.00018)
        .round()
        .clamp(24.0, 512.0) as usize;
    for start in 0..foreground.len() {
        if foreground[start] == 0 || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut pending = vec![start];
        let mut component = Vec::new();
        let mut has_seed = false;
        let mut minimum_x = start % width_usize;
        let mut maximum_x = minimum_x;
        let mut minimum_y = start / width_usize;
        let mut maximum_y = minimum_y;
        while let Some(index) = pending.pop() {
            component.push(index);
            has_seed |= seeds[index] > 0;
            let x = index % width_usize;
            let y = index / width_usize;
            minimum_x = minimum_x.min(x);
            maximum_x = maximum_x.max(x);
            minimum_y = minimum_y.min(y);
            maximum_y = maximum_y.max(y);
            if x > 0 {
                let neighbor = index - 1;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
            if x + 1 < width_usize {
                let neighbor = index + 1;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
            if y > 0 {
                let neighbor = index - width_usize;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
            if y + 1 < height_usize {
                let neighbor = index + width_usize;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
        }
        // Exposure-shifted canvas noise can satisfy a colour seed locally, but
        // it should not become a protected island. Real painted regions in this
        // stack are much larger and remain connected at analysis resolution.
        let component_width = maximum_x - minimum_x + 1;
        let component_height = maximum_y - minimum_y + 1;
        let minimum_component_width = (width as f32 * 0.006).round().clamp(8.0, 32.0) as usize;
        if has_seed
            && component.len() >= minimum_component_size
            && component_width >= minimum_component_width
            && component_height >= minimum_component_width
        {
            for index in component {
                retained[index] = 255;
            }
        }
    }

    GrayImage::from_raw(width, height, retained)
        .expect("focus retained foreground mask dimensions must match")
}

pub(super) fn build_focus_background_foreground_envelope(
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

    // A raw colour mask contains isolated canvas-weave outliers. Keep only
    // connected components that have enough support to be painted content,
    // then expand the retained silhouette so skin/highlights between the dark
    // hair and robe are protected as one object rather than treated as paper.
    let tone_seed = GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        let covered = analysis_mask.as_raw()[index] > 0;
        let start = index * 3;
        image::Luma([u8::from(
            covered
                && focus_stack_pixel_is_tone_foreground(&analysis_image.as_raw()[start..start + 3]),
        ) * 255])
    });
    // Canvas weave can create long, one-pixel colour runs that connect to a
    // real silhouette. Open the seed before connected-component filtering so
    // those runs cannot become foreground exclusions across an entire source
    // tile. The radius is only a few analysis pixels and is applied to the
    // mask, never to the final image.
    let seed_cleanup_radius = (width.max(height) as f32 * 0.0015).round().clamp(2.0, 4.0) as usize;
    let cleaned_tone_seed = open_focus_binary_mask(&tone_seed, seed_cleanup_radius);
    let retained_seed = retain_focus_foreground_components(&cleaned_tone_seed, &cleaned_tone_seed);
    let retained_seed = fill_focus_foreground_holes(&retained_seed);
    let retained_count = retained_seed
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    let mut envelope = if retained_count >= 24 {
        let radius = (width.max(height) as f32 * FOCUS_BACKGROUND_TONE_FOREGROUND_RADIUS_RATIO)
            .round()
            .clamp(8.0, 64.0) as usize;
        dilate_focus_binary_mask(&retained_seed, radius)
    } else {
        GrayImage::new(width, height)
    };

    // The geometric mask is an additional safety net when a caller has a
    // detected depth layer. It is intentionally dilated by the same bounded
    // halo so the tone pass cannot nibble at its silhouette edge.
    let existing_count = existing_foreground_mask
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if existing_count > 0 {
        let radius = (width.max(height) as f32 * 0.01).round().clamp(4.0, 32.0) as usize;
        let existing = dilate_focus_binary_mask(existing_foreground_mask, radius);
        for (destination, source) in envelope.as_mut().iter_mut().zip(existing.as_raw()) {
            *destination = (*destination).max(*source);
        }
    }
    envelope
}

pub(super) fn build_focus_background_reference_mask(
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

pub(super) fn update_focus_background_reference(
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

pub(super) fn hard_select_focus_pixels(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    decision_mask: &GrayImage,
) {
    debug_assert_eq!(base.dimensions(), candidate.dimensions());
    debug_assert_eq!(base.dimensions(), base_mask.dimensions());
    debug_assert_eq!(base.dimensions(), candidate_mask.dimensions());
    debug_assert_eq!(base.dimensions(), decision_mask.dimensions());
    let base_pixels: &mut [f32] = base.as_mut();
    let base_mask_pixels: &mut [u8] = base_mask.as_mut();
    base_pixels
        .par_chunks_mut(3)
        .zip(base_mask_pixels.par_iter_mut())
        .zip(candidate.as_raw().par_chunks(3))
        .zip(candidate_mask.as_raw().par_iter())
        .zip(decision_mask.as_raw().par_iter())
        .for_each(
            |((((base_pixel, base_valid), candidate_pixel), candidate_valid), decision)| {
                if *candidate_valid > 0 && (*base_valid == 0 || *decision > 0) {
                    base_pixel.copy_from_slice(candidate_pixel);
                }
                if *candidate_valid > 0 {
                    *base_valid = 255;
                }
            },
        );
}

pub(super) fn place_focus_layer(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &RenderedFocusLayer,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
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
        .for_each(
            |(((base_row, base_mask_row), candidate_row), candidate_mask_row)| {
                let base_start = candidate.left as usize * 3;
                let base_end = base_start + layer_stride;
                base_row[base_start..base_end].copy_from_slice(candidate_row);
                let mask_start = candidate.left as usize;
                let mask_end = mask_start + layer_width as usize;
                base_mask_row[mask_start..mask_end].copy_from_slice(candidate_mask_row);
            },
        );
}

pub(super) fn place_focus_foreground_mask(base: &mut GrayImage, candidate: &RenderedFocusLayer) {
    let (layer_width, layer_height) = candidate.foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    base.as_mut()
        .par_chunks_mut(base_width as usize)
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(
            candidate
                .foreground_mask
                .as_raw()
                .par_chunks(layer_width as usize),
        )
        .for_each(|(base_row, candidate_row)| {
            let start = candidate.left as usize;
            let end = start + layer_width as usize;
            base_row[start..end].copy_from_slice(candidate_row);
        });
}
