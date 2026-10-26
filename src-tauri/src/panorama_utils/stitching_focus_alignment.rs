use super::*;

pub(crate) fn hard_select_focus_pixels(
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

pub(crate) fn place_focus_layer(
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

pub(crate) fn place_focus_foreground_mask(base: &mut GrayImage, candidate: &RenderedFocusLayer) {
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

pub(crate) fn focus_rgb_luma_at(image: &Rgb32FImage, x: i32, y: i32) -> Option<f64> {
    let (width, height) = image.dimensions();
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return None;
    }
    let pixel = image.get_pixel(x as u32, y as u32);
    pixel
        .0
        .iter()
        .all(|value| value.is_finite())
        .then_some(pixel[0] as f64 * 0.299 + pixel[1] as f64 * 0.587 + pixel[2] as f64 * 0.114)
}

pub(crate) fn focus_alignment_gradient_at(
    image: &Rgb32FImage,
    mask: &GrayImage,
    x: i32,
    y: i32,
) -> Option<(f64, f64)> {
    let (width, height) = image.dimensions();
    if x < 1 || y < 1 || x + 1 >= width as i32 || y + 1 >= height as i32 {
        return None;
    }
    let valid =
        |sample_x: i32, sample_y: i32| mask.get_pixel(sample_x as u32, sample_y as u32)[0] > 0;
    if !valid(x, y) || !valid(x - 1, y) || !valid(x + 1, y) || !valid(x, y - 1) || !valid(x, y + 1)
    {
        return None;
    }
    let horizontal = focus_rgb_luma_at(image, x + 1, y)? - focus_rgb_luma_at(image, x - 1, y)?;
    let vertical = focus_rgb_luma_at(image, x, y + 1)? - focus_rgb_luma_at(image, x, y - 1)?;
    Some((horizontal, vertical))
}

pub(crate) fn focus_full_resolution_patch_score(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    center_x: i32,
    center_y: i32,
    delta_x: i32,
    delta_y: i32,
    radius: i32,
) -> Option<(f64, f64, f64)> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    let mut candidate_horizontal_sum = 0.0;
    let mut candidate_vertical_sum = 0.0;
    let mut merged_horizontal_sum = 0.0;
    let mut merged_vertical_sum = 0.0;
    let mut candidate_squared_sum = 0.0;
    let mut merged_squared_sum = 0.0;
    let mut product_sum = 0.0;
    let mut sample_count = 0usize;
    let mut foreground_count = 0usize;
    let mut patch_count = 0usize;
    for offset_y in -radius..=radius {
        for offset_x in -radius..=radius {
            let local_x = center_x + offset_x;
            let local_y = center_y + offset_y;
            if local_x >= 0
                && local_y >= 0
                && local_x < candidate_width as i32
                && local_y < candidate_height as i32
            {
                patch_count += 1;
                if candidate
                    .foreground_mask
                    .get_pixel(local_x as u32, local_y as u32)[0]
                    > 0
                {
                    foreground_count += 1;
                }
            }
            let Some((candidate_horizontal, candidate_vertical)) =
                focus_alignment_gradient_at(&candidate.image, &candidate.mask, local_x, local_y)
            else {
                continue;
            };
            let merged_x = candidate.left as i32 + local_x + delta_x;
            let merged_y = candidate.top as i32 + local_y + delta_y;
            let Some((merged_horizontal, merged_vertical)) =
                focus_alignment_gradient_at(merged, merged_mask, merged_x, merged_y)
            else {
                continue;
            };
            candidate_horizontal_sum += candidate_horizontal;
            candidate_vertical_sum += candidate_vertical;
            merged_horizontal_sum += merged_horizontal;
            merged_vertical_sum += merged_vertical;
            candidate_squared_sum += candidate_horizontal * candidate_horizontal
                + candidate_vertical * candidate_vertical;
            merged_squared_sum +=
                merged_horizontal * merged_horizontal + merged_vertical * merged_vertical;
            product_sum +=
                candidate_horizontal * merged_horizontal + candidate_vertical * merged_vertical;
            sample_count += 1;
        }
    }
    let minimum_samples = (((radius * 2 + 1) * (radius * 2 + 1)) as f64 * 0.45)
        .round()
        .max(48.0) as usize;
    if sample_count < minimum_samples || patch_count == 0 {
        return None;
    }
    let sample_count_f64 = sample_count as f64;
    let candidate_variance = candidate_squared_sum
        - (candidate_horizontal_sum * candidate_horizontal_sum
            + candidate_vertical_sum * candidate_vertical_sum)
            / sample_count_f64;
    let merged_variance = merged_squared_sum
        - (merged_horizontal_sum * merged_horizontal_sum
            + merged_vertical_sum * merged_vertical_sum)
            / sample_count_f64;
    if candidate_variance <= f64::EPSILON || merged_variance <= f64::EPSILON {
        return None;
    }
    let covariance = product_sum
        - (candidate_horizontal_sum * merged_horizontal_sum
            + candidate_vertical_sum * merged_vertical_sum)
            / sample_count_f64;
    let score = covariance / (candidate_variance * merged_variance).sqrt();
    let candidate_energy = (candidate_variance / sample_count_f64).sqrt();
    let merged_energy = (merged_variance / sample_count_f64).sqrt();
    let foreground_fraction = foreground_count as f64 / patch_count as f64;
    (score.is_finite() && candidate_energy.is_finite() && merged_energy.is_finite()).then_some((
        score,
        candidate_energy.min(merged_energy),
        foreground_fraction,
    ))
}

// Build only the two bounded patches used by one correspondence, rather than
// a second full-resolution copy of both 60MP layers. Gaussian decimation samples
// even pixel centers, so multiplying a displacement by two is exact between
// levels. A stride over native pixels is not a pyramid: it can skip an entire
// narrow correlation peak.
pub(crate) fn focus_alignment_patch_pyramid(
    image: &Rgb32FImage,
    mask: &GrayImage,
    foreground: Option<&GrayImage>,
    center_x: i32,
    center_y: i32,
    radius: i32,
    levels: usize,
) -> Vec<RenderedFocusLayer> {
    let size = (2 * radius + 1) as u32;
    let mut first = RenderedFocusLayer {
        image: Rgb32FImage::new(size, size),
        mask: GrayImage::new(size, size),
        foreground_mask: GrayImage::new(size, size),
        relaxed_foreground_mask: GrayImage::new(0, 0),
        left: 0,
        top: 0,
    };
    for y in 0..size {
        for x in 0..size {
            let sx = center_x + x as i32 - radius;
            let sy = center_y + y as i32 - radius;
            if sx < 0 || sy < 0 || sx >= image.width() as i32 || sy >= image.height() as i32 {
                continue;
            }
            first
                .image
                .put_pixel(x, y, *image.get_pixel(sx as u32, sy as u32));
            first
                .mask
                .put_pixel(x, y, *mask.get_pixel(sx as u32, sy as u32));
            if let Some(foreground) = foreground {
                first
                    .foreground_mask
                    .put_pixel(x, y, *foreground.get_pixel(sx as u32, sy as u32));
            }
        }
    }
    let mut pyramid = vec![first];
    for _ in 0..levels {
        let previous = pyramid.last().unwrap();
        let width = (previous.image.width() + 1) / 2;
        let height = (previous.image.height() + 1) / 2;
        let mut next = RenderedFocusLayer {
            image: Rgb32FImage::new(width, height),
            mask: GrayImage::new(width, height),
            foreground_mask: GrayImage::new(width, height),
            relaxed_foreground_mask: GrayImage::new(0, 0),
            left: 0,
            top: 0,
        };
        for y in 0..height {
            for x in 0..width {
                let mut pixel = [0.0f32; 3];
                let mut valid = true;
                let mut foreground = 0;
                for oy in -1i32..=1 {
                    for ox in -1i32..=1 {
                        let sx =
                            (2 * x as i32 + ox).clamp(0, previous.image.width() as i32 - 1) as u32;
                        let sy =
                            (2 * y as i32 + oy).clamp(0, previous.image.height() as i32 - 1) as u32;
                        let weight = (if ox == 0 { 2.0 } else { 1.0 })
                            * (if oy == 0 { 2.0 } else { 1.0 })
                            / 16.0;
                        let sample = previous.image.get_pixel(sx, sy);
                        for channel in 0..3 {
                            pixel[channel] += sample[channel] * weight;
                        }
                        valid &= previous.mask.get_pixel(sx, sy)[0] > 0;
                        foreground |= previous.foreground_mask.get_pixel(sx, sy)[0];
                    }
                }
                next.image.put_pixel(x, y, Rgb(pixel));
                next.mask
                    .put_pixel(x, y, image::Luma([u8::from(valid) * 255]));
                next.foreground_mask
                    .put_pixel(x, y, image::Luma([foreground]));
            }
        }
        pyramid.push(next);
    }
    pyramid
}

pub(crate) fn focus_patch_translation_candidates(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    center_x: i32,
    center_y: i32,
    max_shift: i32,
) -> Vec<(f64, f64, i32, i32)> {
    let mut levels = 0;
    while (max_shift >> levels) > 24 {
        levels += 1;
    }
    let scale = 1 << levels;
    let patch_radius = FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS;
    let candidate_radius = (patch_radius + 2) * scale;
    let merged_radius = candidate_radius + (max_shift + scale - 1) / scale * scale;
    let mut candidate_pyramid = focus_alignment_patch_pyramid(
        &candidate.image,
        &candidate.mask,
        Some(&candidate.foreground_mask),
        center_x,
        center_y,
        candidate_radius,
        levels,
    );
    let merged_pyramid = focus_alignment_patch_pyramid(
        merged,
        merged_mask,
        None,
        candidate.left as i32 + center_x,
        candidate.top as i32 + center_y,
        merged_radius,
        levels,
    );
    let mut beam: Vec<(f64, f64, i32, i32)> = Vec::new();
    for level in (0..=levels).rev() {
        let scale = 1 << level;
        let limit = (max_shift + scale - 1) / scale;
        let candidate_patch = &mut candidate_pyramid[level];
        candidate_patch.left = ((merged_radius - candidate_radius) / scale) as u32;
        candidate_patch.top = candidate_patch.left;
        let merged_patch = &merged_pyramid[level];
        let mut offsets = Vec::new();
        if beam.is_empty() && level == levels {
            for dy in -limit..=limit {
                for dx in -limit..=limit {
                    offsets.push((dx, dy));
                }
            }
        } else {
            for &(_, _, coarse_x, coarse_y) in &beam {
                // Coarse focus texture can shift by several pixels after
                // Gaussian decimation. Keep a broad native refinement window
                // around each of the few coarse seeds; a tiny ±3 window can
                // permanently exclude the true subpixel peak.
                for dy in -16..=16 {
                    for dx in -16..=16 {
                        let x = coarse_x * 2 + dx;
                        let y = coarse_y * 2 + dy;
                        if x.abs() <= limit && y.abs() <= limit {
                            offsets.push((x, y));
                        }
                    }
                }
            }
        }
        // The baseline must always be measured, including when the search
        // bounds are not an exact multiple of a sampling stride.
        offsets.push((0, 0));
        offsets.sort_unstable();
        offsets.dedup();
        let mut scored = offsets
            .into_iter()
            .filter_map(|(dx, dy)| {
                let (score, energy, foreground_fraction) = focus_full_resolution_patch_score(
                    candidate_patch,
                    &merged_patch.image,
                    &merged_patch.mask,
                    candidate_radius / scale,
                    candidate_radius / scale,
                    dx,
                    dy,
                    patch_radius,
                )?;
                // Coarse levels only propose locations. Native pixels supply the
                // final quality gate, allowing a low-contrast coarse feature to
                // lead to a sharply localized, well-supported native peak.
                (foreground_fraction <= FOCUS_FULL_RES_ALIGNMENT_MAX_FOREGROUND_FRACTION
                    && energy > 1e-6)
                    .then_some((score, energy, dx, dy))
            })
            .collect::<Vec<_>>();
        scored.sort_unstable_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| (a.2.abs() + a.3.abs()).cmp(&(b.2.abs() + b.3.abs())))
        });
        if level == 0 {
            return scored;
        }
        beam.clear();
        for entry in scored {
            if beam.iter().all(|previous| {
                (previous.2 - entry.2).abs() > 2 || (previous.3 - entry.3).abs() > 2
            }) {
                beam.push(entry);
                if beam.len() == 4 {
                    break;
                }
            }
        }
        if beam.is_empty() {
            return Vec::new();
        }
    }
    Vec::new()
}

/// Run the Intra_Station_Registrar over one already rendered layer (需求 2.2–2.9).
///
/// The measurement itself is `mosaic::register_station_frame_native`, the same
/// code the mosaic comparison path uses; this only reconstructs the pose the
/// caller actually rendered with.  `applied_translation` is the integer shift
/// `translate_rendered_focus_layer` has already put on those pixels, and a canvas
/// shift is a translation in the projection plane, so pre-multiplying the
/// station homography with it reproduces the layer exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn register_station_frame_native_layer(
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    image: &ImageInfo,
    source_image: &Rgb32FImage,
    homography: &Matrix3<f64>,
    projection: Projection,
    offset: (f64, f64),
    applied_translation: (i32, i32),
    candidate: &RenderedFocusLayer,
    anchor_long_side: u32,
    anchor_to_station_inverse: Option<Matrix3<f64>>,
) -> Option<super::super::mosaic::StationFrameRegistration> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    if candidate_width < 2 || candidate_height < 2 {
        return None;
    }
    let mut translation = Matrix3::identity();
    translation[(0, 2)] = f64::from(applied_translation.0);
    translation[(1, 2)] = f64::from(applied_translation.1);
    let rendered_pose = translation * homography;
    // 需求 2.1: every record is expressed in the anchor frame's coordinates.
    let into_anchor = anchor_to_station_inverse
        .map(|inverse| inverse * rendered_pose)
        .unwrap_or_else(Matrix3::identity);
    super::super::mosaic::register_station_frame_native(
        merged,
        merged_mask,
        image,
        source_image,
        &rendered_pose,
        projection,
        offset,
        (
            candidate.left,
            candidate.left + candidate_width - 1,
            candidate.top,
            candidate.top + candidate_height - 1,
        ),
        anchor_long_side,
        into_anchor,
    )
}

/// Apply a measured native displacement field to an already rendered layer
/// (需求 2.3–2.7).
///
/// The field is what [`super::super::mosaic::register_station_frame_native`] accepted,
/// expressed as a canvas-pixel offset to add before sampling, so resampling the
/// layer at that offset moves its pixels by exactly the accepted correction. A
/// pixel whose bilinear support is not fully covered loses coverage instead of
/// borrowing an invented neighbour.
pub(crate) fn warp_rendered_focus_layer(
    candidate: RenderedFocusLayer,
    displacement: &super::super::mosaic::NativeResidualDisplacement,
) -> RenderedFocusLayer {
    let (width, height) = candidate.image.dimensions();
    if width == 0 || height == 0 {
        return candidate;
    }
    let row_pixels = width as usize * 3;
    let mut pixels = vec![0.0f32; row_pixels * height as usize];
    let mut mask = vec![0u8; width as usize * height as usize];
    let mut foreground_mask = vec![0u8; width as usize * height as usize];
    let mut relaxed_foreground_mask = vec![0u8; width as usize * height as usize];
    pixels
        .par_chunks_mut(row_pixels)
        .zip(mask.par_chunks_mut(width as usize))
        .zip(foreground_mask.par_chunks_mut(width as usize))
        .zip(relaxed_foreground_mask.par_chunks_mut(width as usize))
        .enumerate()
        .for_each(|(y, (((row, mask_row), foreground_row), relaxed_row))| {
            for x in 0..width as usize {
                let [delta_x, delta_y] = displacement.at(
                    f64::from(candidate.left) + x as f64,
                    f64::from(candidate.top) + y as f64,
                );
                let sample_x = x as f64 + delta_x;
                let sample_y = y as f64 + delta_y;
                if !sample_x.is_finite() || !sample_y.is_finite() {
                    continue;
                }
                let x0 = sample_x.floor();
                let y0 = sample_y.floor();
                if x0 < 0.0
                    || y0 < 0.0
                    || x0 + 1.0 >= f64::from(width)
                    || y0 + 1.0 >= f64::from(height)
                {
                    continue;
                }
                let (x0, y0) = (x0 as u32, y0 as u32);
                if [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
                    .iter()
                    .any(|&(px, py)| candidate.mask.get_pixel(px, py)[0] == 0)
                {
                    continue;
                }
                let fx = (sample_x - f64::from(x0)) as f32;
                let fy = (sample_y - f64::from(y0)) as f32;
                for channel in 0..3 {
                    let at = |px: u32, py: u32| candidate.image.get_pixel(px, py)[channel];
                    let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
                    let bottom = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
                    row[x * 3 + channel] = top * (1.0 - fy) + bottom * fy;
                }
                mask_row[x] = 255;
                // The detected layer masks are categorical, so they follow
                // the nearest sample rather than an average of two classes.
                let nearest_x = (sample_x.round() as u32).min(width - 1);
                let nearest_y = (sample_y.round() as u32).min(height - 1);
                foreground_row[x] = candidate.foreground_mask.get_pixel(nearest_x, nearest_y)[0];
                relaxed_row[x] = candidate
                    .relaxed_foreground_mask
                    .get_pixel(nearest_x, nearest_y)[0];
            }
        });
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(width, height, pixels).expect("warped layer dimensions"),
        mask: GrayImage::from_raw(width, height, mask).expect("warped mask dimensions"),
        foreground_mask: GrayImage::from_raw(width, height, foreground_mask)
            .expect("warped foreground dimensions"),
        relaxed_foreground_mask: GrayImage::from_raw(width, height, relaxed_foreground_mask)
            .expect("warped relaxed foreground dimensions"),
        left: candidate.left,
        top: candidate.top,
    }
}

pub(crate) fn estimate_focus_layer_translation(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
) -> Option<(i32, i32, f64, usize, usize)> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    let (merged_width, merged_height) = merged.dimensions();
    if candidate_width < 2 * (FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS + 2) as u32
        || candidate_height < 2 * (FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS + 2) as u32
        || merged_width < 2
        || merged_height < 2
        || merged_mask.dimensions() != merged.dimensions()
    {
        return None;
    }

    let patch_radius = FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS;
    let overlap_left = candidate.left.max(0).min(merged_width.saturating_sub(1)) as i32;
    let overlap_top = candidate.top.max(0).min(merged_height.saturating_sub(1)) as i32;
    let overlap_right = (candidate
        .left
        .saturating_add(candidate_width)
        .saturating_sub(1))
    .min(merged_width.saturating_sub(1)) as i32;
    let overlap_bottom = (candidate
        .top
        .saturating_add(candidate_height)
        .saturating_sub(1))
    .min(merged_height.saturating_sub(1)) as i32;
    if overlap_right - overlap_left < patch_radius * 2 + 2
        || overlap_bottom - overlap_top < patch_radius * 2 + 2
    {
        return None;
    }

    let grid_size = FOCUS_FULL_RES_ALIGNMENT_GRID_SIZE;
    let mut patch_centers = Vec::with_capacity((grid_size * grid_size) as usize);
    for grid_y in 0..grid_size {
        let global_y =
            overlap_top + ((overlap_bottom - overlap_top) * (grid_y + 1) / (grid_size + 1));
        for grid_x in 0..grid_size {
            let global_x =
                overlap_left + ((overlap_right - overlap_left) * (grid_x + 1) / (grid_size + 1));
            let local_x = global_x - candidate.left as i32;
            let local_y = global_y - candidate.top as i32;
            patch_centers.push((local_x, local_y));
        }
    }

    let mut patch_deltas = Vec::new();
    let image_short_side = candidate_width.min(candidate_height) as i32;
    let max_shift = FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT
        .min((image_short_side as f32 * 0.08).round() as i32)
        .max(1);
    for (center_x, center_y) in patch_centers.iter().copied() {
        // Record zero-shift NCC before applying the acceptance threshold. A
        // genuinely displaced patch commonly has low baseline correlation;
        // rejecting that baseline would make large corrections impossible.
        let current_score = focus_full_resolution_patch_score(
            candidate,
            merged,
            merged_mask,
            center_x,
            center_y,
            0,
            0,
            patch_radius,
        )
        .map(|value| value.0);
        let scores = focus_patch_translation_candidates(
            candidate,
            merged,
            merged_mask,
            center_x,
            center_y,
            max_shift,
        );
        let Some(&(best_score, energy, best_x, best_y)) = scores.first() else {
            continue;
        };
        let second_best = scores
            .iter()
            .find(|entry| (entry.2 - best_x).abs() > 2 || (entry.3 - best_y).abs() > 2)
            .map(|entry| entry.0)
            .unwrap_or(f64::NEG_INFINITY);
        if best_score < FOCUS_FULL_RES_ALIGNMENT_MIN_NCC
            || energy < FOCUS_FULL_RES_ALIGNMENT_MIN_ENERGY
            || best_score - second_best < FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN
            || (best_x != 0 || best_y != 0)
                && current_score
                    .is_some_and(|score| best_score - score < FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN)
        {
            continue;
        }
        patch_deltas.push((best_x, best_y, best_score));
    }
    if std::env::var_os("RAW_EDITOR_FOCUS_TRANSLATION_DIAGNOSTICS").is_some() {
        println!(
            "  - Focus translation diagnostics: candidate={}x{} patches={} accepted={}",
            candidate_width,
            candidate_height,
            patch_centers.len(),
            patch_deltas.len()
        );
    }
    if patch_deltas.is_empty() {
        return None;
    }

    let mut x_values = patch_deltas.iter().map(|(x, _, _)| *x).collect::<Vec<_>>();
    let mut y_values = patch_deltas.iter().map(|(_, y, _)| *y).collect::<Vec<_>>();
    x_values.sort_unstable();
    y_values.sort_unstable();
    let median_x = x_values[x_values.len() / 2];
    let median_y = y_values[y_values.len() / 2];
    let support = patch_deltas
        .iter()
        .filter(|(x, y, _)| (x - median_x).abs() <= 1 && (y - median_y).abs() <= 1)
        .count();
    let minimum_support = FOCUS_FULL_RES_ALIGNMENT_MIN_PATCHES;
    if support < minimum_support
        || (support as f64) < patch_deltas.len() as f64 * 0.5
        || (median_x == 0 && median_y == 0)
    {
        return None;
    }
    Some((
        median_x,
        median_y,
        patch_deltas
            .iter()
            .filter(|(x, y, _)| (x - median_x).abs() <= 1 && (y - median_y).abs() <= 1)
            .map(|(_, _, score)| score)
            .sum::<f64>()
            / support as f64,
        support,
        patch_deltas.len(),
    ))
}
