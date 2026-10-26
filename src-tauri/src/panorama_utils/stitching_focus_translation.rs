use super::*;

pub(crate) fn translate_rendered_focus_layer(
    candidate: RenderedFocusLayer,
    delta_x: i32,
    delta_y: i32,
    out_width: u32,
    out_height: u32,
) -> RenderedFocusLayer {
    if delta_x == 0 && delta_y == 0 {
        return candidate;
    }
    let (width, height) = candidate.image.dimensions();
    if width == 0 || height == 0 || out_width == 0 || out_height == 0 {
        return candidate;
    }
    let old_left = candidate.left as i64;
    let old_top = candidate.top as i64;
    let old_right = old_left + width as i64 - 1;
    let old_bottom = old_top + height as i64 - 1;
    let new_left = (old_left + i64::from(delta_x)).clamp(0, i64::from(out_width - 1));
    let new_top = (old_top + i64::from(delta_y)).clamp(0, i64::from(out_height - 1));
    let new_right = (old_right + i64::from(delta_x)).clamp(0, i64::from(out_width - 1));
    let new_bottom = (old_bottom + i64::from(delta_y)).clamp(0, i64::from(out_height - 1));
    if new_left > new_right || new_top > new_bottom {
        return RenderedFocusLayer {
            image: Rgb32FImage::new(0, 0),
            mask: GrayImage::new(0, 0),
            foreground_mask: GrayImage::new(0, 0),
            relaxed_foreground_mask: GrayImage::new(0, 0),
            left: 0,
            top: 0,
        };
    }
    let new_width = (new_right - new_left + 1) as u32;
    let new_height = (new_bottom - new_top + 1) as u32;
    let mut image_pixels = vec![0.0f32; new_width as usize * new_height as usize * 3];
    let mut mask_pixels = vec![0u8; new_width as usize * new_height as usize];
    let mut foreground_pixels = vec![0u8; new_width as usize * new_height as usize];
    let mut relaxed_pixels = vec![0u8; new_width as usize * new_height as usize];
    let source_image = candidate.image.as_raw();
    let source_mask = candidate.mask.as_raw();
    let source_foreground = candidate.foreground_mask.as_raw();
    let source_relaxed = candidate.relaxed_foreground_mask.as_raw();
    for source_y in 0..height as i64 {
        for source_x in 0..width as i64 {
            let destination_x = old_left + source_x + i64::from(delta_x);
            let destination_y = old_top + source_y + i64::from(delta_y);
            if destination_x < new_left
                || destination_x > new_right
                || destination_y < new_top
                || destination_y > new_bottom
            {
                continue;
            }
            let source_index = source_y as usize * width as usize + source_x as usize;
            let destination_index = (destination_y - new_top) as usize * new_width as usize
                + (destination_x - new_left) as usize;
            let source_start = source_index * 3;
            let destination_start = destination_index * 3;
            image_pixels[destination_start..destination_start + 3]
                .copy_from_slice(&source_image[source_start..source_start + 3]);
            mask_pixels[destination_index] = source_mask[source_index];
            foreground_pixels[destination_index] = source_foreground[source_index];
            relaxed_pixels[destination_index] = source_relaxed[source_index];
        }
    }
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(new_width, new_height, image_pixels)
            .expect("translated focus layer image dimensions must match"),
        mask: GrayImage::from_raw(new_width, new_height, mask_pixels)
            .expect("translated focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(new_width, new_height, foreground_pixels)
            .expect("translated focus foreground mask dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(new_width, new_height, relaxed_pixels)
            .expect("translated focus relaxed foreground mask dimensions must match"),
        left: new_left as u32,
        top: new_top as u32,
    }
}

pub(crate) fn align_focus_foreground_layer_to_existing(
    candidate: RenderedFocusLayer,
    merged_foreground_mask: &GrayImage,
    refinement: &mut Option<FocusForegroundMove>,
) -> RenderedFocusLayer {
    let (layer_width, layer_height) = candidate.foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 || merged_foreground_mask.width() == 0 {
        return candidate;
    }

    let foreground = candidate.foreground_mask.as_raw();
    let foreground_count = foreground.iter().filter(|value| **value > 0).count();
    let existing_count = merged_foreground_mask
        .as_raw()
        .iter()
        .filter(|value| **value > 0)
        .count();
    if foreground_count < 32 || existing_count < 32 {
        return candidate;
    }

    // Use a bounded, deterministic set of mask anchors. This makes the
    // refinement cheap even for a full 60MP source while retaining enough
    // support to distinguish a real overlap from an accidental nearby edge.
    let anchor_stride = (foreground_count / 2_048).max(1);
    let mut anchors = Vec::with_capacity(foreground_count.min(2_048));
    let mut foreground_index = 0usize;
    for (index, value) in foreground.iter().enumerate() {
        if *value == 0 {
            continue;
        }
        if foreground_index.is_multiple_of(anchor_stride) {
            let local_x = (index % layer_width as usize) as u32;
            let local_y = (index / layer_width as usize) as u32;
            anchors.push((candidate.left + local_x, candidate.top + local_y));
        }
        foreground_index += 1;
    }
    if anchors.len() < 32 {
        return candidate;
    }

    let base_width = merged_foreground_mask.width() as i32;
    let base_height = merged_foreground_mask.height() as i32;
    let base_pixels = merged_foreground_mask.as_raw();
    let search_radius = ((layer_width.max(layer_height) as f32) * 0.01)
        .round()
        .clamp(4.0, FOCUS_FOREGROUND_REFINEMENT_MAX_SHIFT as f32) as i32;
    let coarse_step = (search_radius / 24).max(1);
    let hit_count = |delta_x: i32, delta_y: i32| -> usize {
        anchors
            .iter()
            .filter(|&&(x, y)| {
                let x = x as i32 + delta_x;
                let y = y as i32 + delta_y;
                x >= 0
                    && y >= 0
                    && x < base_width
                    && y < base_height
                    && base_pixels[y as usize * base_width as usize + x as usize] > 0
            })
            .count()
    };

    let current_hits = hit_count(0, 0);
    let mut best_delta = (0i32, 0i32);
    let mut best_hits = current_hits;
    for delta_y in (-search_radius..=search_radius).step_by(coarse_step as usize) {
        for delta_x in (-search_radius..=search_radius).step_by(coarse_step as usize) {
            let hits = hit_count(delta_x, delta_y);
            if hits > best_hits {
                best_hits = hits;
                best_delta = (delta_x, delta_y);
            }
        }
    }
    // Refine around the best coarse translation at pixel precision. The
    // foreground mask is only a registration cue; the final source pixels are
    // still selected by the focus score below.
    if coarse_step > 1 {
        let (coarse_x, coarse_y) = best_delta;
        let fine_min_y = (coarse_y - coarse_step + 1).max(-search_radius);
        let fine_max_y = (coarse_y + coarse_step - 1).min(search_radius);
        let fine_min_x = (coarse_x - coarse_step + 1).max(-search_radius);
        let fine_max_x = (coarse_x + coarse_step - 1).min(search_radius);
        for delta_y in fine_min_y..=fine_max_y {
            for delta_x in fine_min_x..=fine_max_x {
                let hits = hit_count(delta_x, delta_y);
                if hits > best_hits {
                    best_hits = hits;
                    best_delta = (delta_x, delta_y);
                }
            }
        }
    }

    let minimum_gain = (anchors.len() / 20).max(12);
    if best_delta == (0, 0)
        || best_hits < current_hits.saturating_add(minimum_gain)
        || best_hits < anchors.len() / 20
    {
        return candidate;
    }

    let (delta_x, delta_y) = best_delta;
    let mut image_pixels = candidate.image.into_raw();
    let mut candidate_mask = candidate.mask.into_raw();
    let mut foreground_mask = vec![0u8; foreground.len()];
    let relaxed_foreground = candidate.relaxed_foreground_mask.as_raw();
    let mut relaxed_foreground_mask = vec![0u8; foreground.len()];
    let mut moved_pixels = Vec::with_capacity(foreground_count);
    for (index, value) in foreground.iter().enumerate() {
        if *value == 0 {
            continue;
        }
        let local_x = (index % layer_width as usize) as i32;
        let local_y = (index / layer_width as usize) as i32;
        let destination_x = local_x + delta_x;
        let destination_y = local_y + delta_y;
        if destination_x < 0
            || destination_y < 0
            || destination_x >= layer_width as i32
            || destination_y >= layer_height as i32
        {
            continue;
        }
        let destination_index =
            destination_y as usize * layer_width as usize + destination_x as usize;
        let source_start = index * 3;
        moved_pixels.push((
            destination_index,
            [
                image_pixels[source_start],
                image_pixels[source_start + 1],
                image_pixels[source_start + 2],
            ],
        ));
        candidate_mask[index] = 0;
        foreground_mask[destination_index] = 255;
        if relaxed_foreground[index] > 0 {
            relaxed_foreground_mask[destination_index] = 255;
        }
    }
    *refinement = Some(FocusForegroundMove {
        delta: best_delta,
        left: candidate.left,
        top: candidate.top,
        destinations: GrayImage::from_raw(layer_width, layer_height, foreground_mask.clone())
            .expect("foreground refinement dimensions"),
    });
    for (destination_index, pixel) in moved_pixels {
        let destination_start = destination_index * 3;
        image_pixels[destination_start..destination_start + 3].copy_from_slice(&pixel);
        candidate_mask[destination_index] = 255;
    }

    println!(
        "  - Focus foreground mask refinement: delta=({delta_x},{delta_y}), overlap={current_hits}->{best_hits} anchors={}",
        anchors.len()
    );
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(layer_width, layer_height, image_pixels)
            .expect("aligned focus layer image dimensions must match"),
        mask: GrayImage::from_raw(layer_width, layer_height, candidate_mask)
            .expect("aligned focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(layer_width, layer_height, foreground_mask)
            .expect("aligned focus foreground mask dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(
            layer_width,
            layer_height,
            relaxed_foreground_mask,
        )
        .expect("aligned focus relaxed foreground mask dimensions must match"),
        left: candidate.left,
        top: candidate.top,
    }
}

pub(crate) fn suppress_focus_foreground_switches(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
) {
    suppress_focus_foreground_switches_in_region(
        decision_mask,
        merged_foreground_mask,
        candidate.left,
        candidate.top,
        &candidate.foreground_mask,
        &candidate.relaxed_foreground_mask,
    );
}

pub(crate) fn suppress_focus_foreground_switches_in_region(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    candidate_foreground_mask: &GrayImage,
    relaxed_foreground_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate_foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = merged_foreground_mask.width();
    let base_height = merged_foreground_mask.height();
    if candidate_left >= base_width
        || candidate_top >= base_height
        || candidate_left + layer_width > base_width
        || candidate_top + layer_height > base_height
    {
        return;
    }
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));

    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        candidate_left,
        candidate_top,
        (layer_width, layer_height),
    );
    if relaxed_foreground_mask.dimensions() != (layer_width, layer_height) {
        return;
    }

    decision_mask
        .as_mut()
        .par_chunks_mut(layer_width as usize)
        .enumerate()
        .for_each(|(local_y, decision_row)| {
            let foreground_row = &candidate_foreground_mask.as_raw()
                [local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            let overlap_row =
                &overlap_mask[local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            let relaxed_row = &relaxed_foreground_mask.as_raw()
                [local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            for (local_x, decision) in decision_row.iter_mut().enumerate() {
                if foreground_row[local_x] > 0
                    && relaxed_row[local_x] == 0
                    && overlap_row[local_x] > 0
                {
                    *decision = 0;
                }
            }
        });
}

pub(crate) fn suppress_focus_foreground_switches_on_canvas(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_foreground_mask: &GrayImage,
    relaxed_foreground_mask: &GrayImage,
) {
    if decision_mask.dimensions() != merged_foreground_mask.dimensions()
        || decision_mask.dimensions() != candidate_foreground_mask.dimensions()
        || candidate_foreground_mask.dimensions() != relaxed_foreground_mask.dimensions()
    {
        return;
    }
    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        0,
        0,
        candidate_foreground_mask.dimensions(),
    );
    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(candidate_foreground_mask.as_raw().par_iter())
        .zip(overlap_mask.par_iter())
        .zip(relaxed_foreground_mask.as_raw().par_iter())
        .for_each(|(((decision, candidate_foreground), overlap), relaxed)| {
            if *candidate_foreground > 0 && *relaxed == 0 && *overlap > 0 {
                *decision = 0;
            }
        });
}

pub(crate) fn suppress_focus_foreground_ownership_switches(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    relaxed_foreground_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate_mask.dimensions();
    if layer_width == 0
        || layer_height == 0
        || decision_mask.dimensions() != (layer_width, layer_height)
        || relaxed_foreground_mask.dimensions() != (layer_width, layer_height)
    {
        return;
    }
    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        candidate_left,
        candidate_top,
        (layer_width, layer_height),
    );
    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(candidate_mask.as_raw().par_iter())
        .zip(overlap_mask.par_iter())
        .zip(relaxed_foreground_mask.as_raw().par_iter())
        .for_each(|(((decision, candidate_valid), overlap), relaxed)| {
            if *candidate_valid > 0 && *relaxed == 0 && *overlap > 0 {
                *decision = 0;
            }
        });
}

pub(crate) fn suppress_focus_foreground_ownership_switches_on_canvas(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_mask: &GrayImage,
    relaxed_foreground_mask: &GrayImage,
) {
    if decision_mask.dimensions() != merged_foreground_mask.dimensions()
        || decision_mask.dimensions() != candidate_mask.dimensions()
        || candidate_mask.dimensions() != relaxed_foreground_mask.dimensions()
    {
        return;
    }
    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        0,
        0,
        candidate_mask.dimensions(),
    );
    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(candidate_mask.as_raw().par_iter())
        .zip(overlap_mask.par_iter())
        .zip(relaxed_foreground_mask.as_raw().par_iter())
        .for_each(|(((decision, candidate_valid), overlap), relaxed)| {
            if *candidate_valid > 0 && *relaxed == 0 && *overlap > 0 {
                *decision = 0;
            }
        });
}

pub(crate) fn focus_foreground_overlap_mask_for_region(
    merged_foreground_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    candidate_dimensions: (u32, u32),
) -> Vec<u8> {
    let (layer_width, layer_height) = candidate_dimensions;
    let pixel_count = layer_width as usize * layer_height as usize;
    let mut overlap = vec![0u8; pixel_count];
    if layer_width == 0 || layer_height == 0 {
        return overlap;
    }

    let radius = ((layer_width.max(layer_height) as f32) * FOCUS_FOREGROUND_OWNERSHIP_RADIUS_RATIO)
        .round()
        .clamp(8.0, 256.0) as usize;
    let base_width = merged_foreground_mask.width() as usize;
    let base_height = merged_foreground_mask.height() as usize;
    let left = candidate_left as usize;
    let top = candidate_top as usize;
    let width = layer_width as usize;
    let height = layer_height as usize;
    if left >= base_width
        || top >= base_height
        || left + width > base_width
        || top + height > base_height
    {
        return overlap;
    }

    // First apply a horizontal max filter to the already selected foreground
    // mask. This is equivalent to testing the whole rectangular neighbourhood,
    // but is linear in the candidate size and does not allocate a full canvas.
    let mut horizontal = vec![0u8; pixel_count];
    let base_pixels = merged_foreground_mask.as_raw();
    for local_y in 0..height {
        let base_row = &base_pixels[(top + local_y) * base_width..(top + local_y + 1) * base_width];
        let initial_start = left.saturating_sub(radius);
        let initial_end = (left + radius).min(base_width - 1);
        let mut window = base_row[initial_start..=initial_end]
            .iter()
            .filter(|value| **value > 0)
            .count() as u32;
        for local_x in 0..width {
            if local_x > 0 {
                let center_x = left + local_x;
                let add_x = (center_x + radius).min(base_width - 1);
                window += u32::from(base_row[add_x] > 0);
                if center_x > radius {
                    let remove_x = center_x - radius - 1;
                    window -= u32::from(base_row[remove_x] > 0);
                }
            }
            horizontal[local_y * width + local_x] = u8::from(window > 0);
        }
    }

    // Then apply the same max filter vertically. A later frame only loses a
    // foreground switch when it really overlaps this expanded ownership area;
    // unrelated artwork remains governed by the normal focus score.
    for local_x in 0..width {
        let initial_end = radius.min(height - 1);
        let mut window = (0..=initial_end)
            .map(|row| u32::from(horizontal[row * width + local_x] > 0))
            .sum::<u32>();
        for local_y in 0..height {
            if local_y > 0 {
                let add_y = (local_y + radius).min(height - 1);
                window += u32::from(horizontal[add_y * width + local_x] > 0);
                if local_y > radius {
                    let remove_y = local_y.saturating_sub(radius + 1);
                    window -= u32::from(horizontal[remove_y * width + local_x] > 0);
                }
            }
            overlap[local_y * width + local_x] = u8::from(window > 0);
        }
    }
    overlap
}

pub(crate) fn focus_foreground_edge_mask(
    mask: &[u8],
    width: u32,
    height: u32,
    radius: usize,
) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    if width == 0 || height == 0 || mask.len() != width * height {
        return Vec::new();
    }
    if radius == 0 {
        return mask.to_vec();
    }

    // Erode the protected foreground with separable prefix-sum windows. The
    // pixels left outside that erosion are the silhouette band; the interior
    // can participate in low-frequency exposure blending without averaging a
    // displaced edge.
    let mut horizontal = vec![0u8; mask.len()];
    for y in 0..height {
        let row_start = y * width;
        let mut prefix = vec![0u32; width + 1];
        for x in 0..width {
            prefix[x + 1] = prefix[x] + u32::from(mask[row_start + x] > 0);
        }
        for x in 0..width {
            let start = x.saturating_sub(radius);
            let end = (x + radius).min(width - 1);
            let window_length = (end - start + 1) as u32;
            if prefix[end + 1] - prefix[start] == window_length {
                horizontal[row_start + x] = 255;
            }
        }
    }

    let mut eroded = vec![0u8; mask.len()];
    for x in 0..width {
        let mut prefix = vec![0u32; height + 1];
        for y in 0..height {
            prefix[y + 1] = prefix[y] + u32::from(horizontal[y * width + x] > 0);
        }
        for y in 0..height {
            let start = y.saturating_sub(radius);
            let end = (y + radius).min(height - 1);
            let window_length = (end - start + 1) as u32;
            if prefix[end + 1] - prefix[start] == window_length {
                eroded[y * width + x] = 255;
            }
        }
    }

    mask.iter()
        .zip(eroded)
        .map(|(value, interior)| u8::from(*value > 0 && interior == 0))
        .collect()
}

pub(crate) fn mark_focus_foreground_pixels(
    merged_foreground_mask: &mut GrayImage,
    merged_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
) {
    mark_focus_foreground_pixels_in_region(
        merged_foreground_mask,
        merged_mask,
        candidate.left,
        candidate.top,
        &candidate.foreground_mask,
        decision_mask,
    );
}

pub(crate) fn mark_focus_foreground_pixels_in_region(
    merged_foreground_mask: &mut GrayImage,
    merged_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    candidate_foreground_mask: &GrayImage,
    decision_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate_foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = merged_foreground_mask.width();
    let base_height = merged_foreground_mask.height();
    if candidate_left >= base_width
        || candidate_top >= base_height
        || candidate_left + layer_width > base_width
        || candidate_top + layer_height > base_height
    {
        return;
    }
    debug_assert_eq!(
        merged_mask.dimensions(),
        merged_foreground_mask.dimensions()
    );
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));

    merged_foreground_mask
        .as_mut()
        .par_chunks_mut(base_width as usize)
        .skip(candidate_top as usize)
        .take(layer_height as usize)
        .zip(
            merged_mask
                .as_raw()
                .par_chunks(base_width as usize)
                .skip(candidate_top as usize),
        )
        .zip(
            candidate_foreground_mask
                .as_raw()
                .par_chunks(layer_width as usize),
        )
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(
            |(((foreground_row, merged_mask_row), candidate_foreground_row), decision_row)| {
                for local_x in 0..layer_width as usize {
                    if candidate_foreground_row[local_x] > 0
                        && (merged_mask_row[candidate_left as usize + local_x] == 0
                            || decision_row[local_x] > 0)
                    {
                        foreground_row[candidate_left as usize + local_x] = 255;
                    }
                }
            },
        );
}
