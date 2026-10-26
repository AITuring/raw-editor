use super::*;

pub(crate) fn multiband_blend(
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

pub(crate) fn blend_focus_seam_band(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width < 3 || layer_height < 3 {
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

    // A seam may cross a detected depth/occlusion layer, but that layer must
    // never be reconstructed by averaging two different source positions. The
    // mask is optional and content-derived; for ordinary flat stacks this is an
    // all-zero buffer and the generic seam path is unchanged.
    let foreground_overlap = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        candidate.left,
        candidate.top,
        (layer_width, layer_height),
    );
    let protected_foreground = candidate
        .foreground_mask
        .as_raw()
        .iter()
        .zip(foreground_overlap.iter())
        .zip(candidate.relaxed_foreground_mask.as_raw().iter())
        .map(
            |((candidate_foreground, existing_foreground), relaxed_foreground)| {
                u8::from(
                    (*candidate_foreground > 0 || *existing_foreground > 0)
                        && *relaxed_foreground == 0,
                )
            },
        )
        .collect::<Vec<_>>();
    let hard_foreground_edge_radius = ((layer_width.max(layer_height) as f32 / 2_400.0)
        * FOCUS_FOREGROUND_HARD_EDGE_RADIUS_AT_2400)
        .round()
        .clamp(3.0, 24.0) as usize;
    let protected_foreground_edges = focus_foreground_edge_mask(
        &protected_foreground,
        layer_width,
        layer_height,
        hard_foreground_edge_radius,
    );
    let is_protected = |x: u32, y: u32| -> bool {
        protected_foreground_edges[y as usize * layer_width as usize + x as usize] > 0
    };

    let owns_candidate = |x: u32, y: u32| -> bool {
        if x >= layer_width || y >= layer_height || candidate.mask.get_pixel(x, y)[0] == 0 {
            return false;
        }
        let global_x = candidate.left + x;
        let global_y = candidate.top + y;
        base_mask.get_pixel(global_x, global_y)[0] == 0 || decision_mask.get_pixel(x, y)[0] > 0
    };

    let mut seam_left = layer_width;
    let mut seam_right = 0u32;
    let mut seam_top = layer_height;
    let mut seam_bottom = 0u32;
    for y in 1..layer_height.saturating_sub(1) {
        for x in 1..layer_width.saturating_sub(1) {
            // Do not let a transition on a foreground silhouette expand the
            // seam's bounding box over the whole object. Its ownership is
            // restored pixel-for-pixel after the paper seam is blended.
            if is_protected(x, y) {
                continue;
            }
            let ownership = owns_candidate(x, y);
            let has_transition = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                .into_iter()
                .any(|(neighbor_x, neighbor_y)| {
                    !is_protected(neighbor_x, neighbor_y)
                        && owns_candidate(neighbor_x, neighbor_y) != ownership
                });
            if has_transition {
                seam_left = seam_left.min(x);
                seam_right = seam_right.max(x);
                seam_top = seam_top.min(y);
                seam_bottom = seam_bottom.max(y);
            }
        }
    }
    if seam_left > seam_right || seam_top > seam_bottom {
        hard_select_focus_layer(base, base_mask, candidate, decision_mask);
        return;
    }

    let radius = FOCUS_SEAM_BLEND_RADIUS.min(layer_width.max(layer_height) as usize);
    // The focus seam is used for detail ownership, but the exposure step can
    // extend across the complete shared canvas area. Build the low-frequency
    // pyramid over that overlap, not only over a rectangle around the sharpness
    // transition; otherwise a source-sized canvas block outside the seam box is
    // copied unchanged into the final stack.
    let mut overlap_left = layer_width;
    let mut overlap_right = 0u32;
    let mut overlap_top = layer_height;
    let mut overlap_bottom = 0u32;
    for y in 0..layer_height {
        for x in 0..layer_width {
            if candidate.mask.get_pixel(x, y)[0] == 0
                || base_mask.get_pixel(candidate.left + x, candidate.top + y)[0] == 0
            {
                continue;
            }
            overlap_left = overlap_left.min(x);
            overlap_right = overlap_right.max(x);
            overlap_top = overlap_top.min(y);
            overlap_bottom = overlap_bottom.max(y);
        }
    }
    let patch_left = if overlap_left <= overlap_right {
        overlap_left
    } else {
        seam_left.saturating_sub(radius as u32)
    };
    let patch_right = if overlap_left <= overlap_right {
        overlap_right
    } else {
        (seam_right + radius as u32).min(layer_width - 1)
    };
    let patch_top = if overlap_top <= overlap_bottom {
        overlap_top
    } else {
        seam_top.saturating_sub(radius as u32)
    };
    let patch_bottom = if overlap_top <= overlap_bottom {
        overlap_bottom
    } else {
        (seam_bottom + radius as u32).min(layer_height - 1)
    };
    let patch_width = patch_right - patch_left + 1;
    let patch_height = patch_bottom - patch_top + 1;
    if patch_width < 8 || patch_height < 8 {
        hard_select_focus_layer(base, base_mask, candidate, decision_mask);
        return;
    }

    let patch_pixel_count = patch_width as usize * patch_height as usize;
    let patch_rgb_stride = patch_width as usize * 3;
    let patch_mask_stride = patch_width as usize;
    let mut base_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut candidate_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut blend_values = vec![0.0f32; patch_pixel_count];
    let mut tone_blend_values = vec![0u8; patch_pixel_count];
    let mut tone_sample_values = vec![0u8; patch_pixel_count];
    let mut protected_values = vec![0u8; patch_pixel_count];
    let mut protected_pixels = vec![0.0f32; patch_pixel_count * 3];
    let base_ref: &Rgb32FImage = base;
    let base_mask_ref: &GrayImage = base_mask;

    base_pixels
        .par_chunks_mut(patch_rgb_stride)
        .zip(candidate_pixels.par_chunks_mut(patch_rgb_stride))
        .zip(blend_values.par_chunks_mut(patch_mask_stride))
        .zip(tone_blend_values.par_chunks_mut(patch_mask_stride))
        .zip(tone_sample_values.par_chunks_mut(patch_mask_stride))
        .zip(protected_values.par_chunks_mut(patch_mask_stride))
        .zip(protected_pixels.par_chunks_mut(patch_rgb_stride))
        .enumerate()
        .for_each(
            |(
                local_y,
                (
                    (
                        ((((base_row, candidate_row), blend_row), tone_blend_row), tone_sample_row),
                        protected_row,
                    ),
                    protected_pixel_row,
                ),
            )| {
                let source_y = patch_top + local_y as u32;
                for local_x in 0..patch_width {
                    let source_x = patch_left + local_x;
                    let global_x = candidate.left + source_x;
                    let global_y = candidate.top + source_y;
                    let base_valid = base_mask_ref.get_pixel(global_x, global_y)[0] > 0;
                    let candidate_valid = candidate.mask.get_pixel(source_x, source_y)[0] > 0;
                    let current_pixel = *base_ref.get_pixel(global_x, global_y);
                    let candidate_pixel = *candidate.image.get_pixel(source_x, source_y);
                    let base_pixel = if base_valid || !candidate_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };
                    let candidate_pixel = if candidate_valid {
                        candidate_pixel
                    } else {
                        base_pixel
                    };
                    let owns = candidate_valid
                        && (!base_valid || decision_mask.get_pixel(source_x, source_y)[0] > 0);
                    let candidate_foreground =
                        candidate.foreground_mask.get_pixel(source_x, source_y)[0] > 0;
                    let existing_foreground = foreground_overlap
                        [source_y as usize * layer_width as usize + source_x as usize]
                        > 0;
                    // Black ink, red seals, and other painted content are not part
                    // of the geometric depth mask. They still must not enter the
                    // low-frequency pyramid: averaging two sub-pixel-shifted
                    // strokes is exactly the blur/double-contour regression this
                    // path is intended to prevent. Keep the canvas gate separate
                    // so a darker/lighter paper exposure can still be harmonized.
                    let tone_foreground =
                        focus_stack_pixel_is_tone_foreground(base_pixel.0.as_slice())
                            || focus_stack_pixel_is_tone_foreground(candidate_pixel.0.as_slice());
                    let artwork_foreground =
                        candidate_foreground || existing_foreground || tone_foreground;
                    // Blend the broad tone over the union of the two valid
                    // canvas regions. When only one source exists, the
                    // missing side below is copied from that same source, so
                    // extending the mask is harmless and lets the low
                    // frequency transition continue through a tile edge.
                    // All artwork pixels use hard source ownership, including
                    // their low-frequency bands.
                    tone_blend_row[local_x as usize] =
                        u8::from((base_valid || candidate_valid) && !artwork_foreground);
                    tone_sample_row[local_x as usize] =
                        u8::from(base_valid && candidate_valid && !artwork_foreground);
                    protected_row[local_x as usize] = u8::from(artwork_foreground);
                    let start = local_x as usize * 3;
                    base_row[start..start + 3].copy_from_slice(&base_pixel.0);
                    candidate_row[start..start + 3].copy_from_slice(&candidate_pixel.0);
                    if artwork_foreground {
                        let selected_pixel = if owns { candidate_pixel } else { base_pixel };
                        protected_pixel_row[start..start + 3].copy_from_slice(&selected_pixel.0);
                    }
                    blend_row[local_x as usize] = if owns { 1.0 } else { 0.0 };
                }
            },
        );

    let low_frequency = masked_box_blur_focus_map(
        &blend_values,
        &tone_blend_values,
        patch_width,
        patch_height,
        (radius / 2).max(1),
    );

    // Match only the source-sized tone step measured from the shared canvas
    // samples. This is deliberately a constant correction for this local seam:
    // it is applied to the candidate before the pyramid is built, so it lands
    // in the coarsest tone band while the weave, brush strokes, and all hard
    // selected foreground detail remain in their original frequency bands.
    let tone_sample_count = tone_sample_values
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if tone_sample_count >= FOCUS_SEAM_TONE_MIN_SAMPLES {
        let mut channel_differences = [Vec::new(), Vec::new(), Vec::new()];
        for (index, &sample) in tone_sample_values.iter().enumerate() {
            if sample == 0 {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                channel_differences[channel]
                    .push(base_pixels[start + channel] - candidate_pixels[start + channel]);
            }
        }
        let mut tone_correction = [0.0f32; 3];
        for channel in 0..3 {
            let differences = &mut channel_differences[channel];
            differences.sort_unstable_by(f32::total_cmp);
            tone_correction[channel] = differences[differences.len() / 2].clamp(
                -FOCUS_SEAM_TONE_MAX_ADJUSTMENT,
                FOCUS_SEAM_TONE_MAX_ADJUSTMENT,
            );
        }
        for (index, &protected) in protected_values.iter().enumerate() {
            if protected > 0 {
                continue;
            }
            let local_x = (index % patch_width as usize) as u32;
            let local_y = (index / patch_width as usize) as u32;
            if candidate
                .mask
                .get_pixel(patch_left + local_x, patch_top + local_y)[0]
                == 0
            {
                continue;
            }
            let start = index * 3;
            // `low_frequency` is the smoothed hard ownership mask. Its middle
            // values identify the actual seam; using 4*a*(1-a) keeps the
            // measured correction out of the candidate's remote canvas and
            // out of the base side of the patch. The previous tone-union mask
            // was one across the whole rectangle, which silently turned a
            // large seam bounding box into a full candidate-wide recolour.
            let ownership = low_frequency[index].clamp(0.0, 1.0);
            let weight = (4.0 * ownership * (1.0 - ownership)).clamp(0.0, 1.0);
            if weight <= 0.0 {
                continue;
            }
            for channel in 0..3 {
                candidate_pixels[start + channel] += tone_correction[channel] * weight;
            }
        }
    }

    let blend_mask = GrayImage::from_fn(patch_width, patch_height, |x, y| {
        image::Luma([(blend_values[y as usize * patch_width as usize + x as usize] * 255.0) as u8])
    });
    let low_frequency_mask = GrayImage::from_fn(patch_width, patch_height, |x, y| {
        let index = y as usize * patch_width as usize + x as usize;
        let value = if tone_blend_values[index] > 0 {
            low_frequency[index]
        } else {
            blend_values[index]
        };
        image::Luma([(value.clamp(0.0, 1.0) * 255.0) as u8])
    });
    let blended = multiband_blend(
        Rgb32FImage::from_raw(patch_width, patch_height, base_pixels)
            .expect("focus seam base dimensions must match"),
        Rgb32FImage::from_raw(patch_width, patch_height, candidate_pixels)
            .expect("focus seam candidate dimensions must match"),
        blend_mask,
        Some(low_frequency_mask),
        PANORAMA_BLEND_BANDS,
        true,
    );

    let blended_pixels = blended.as_raw();
    base.as_mut()
        .par_chunks_mut(base_width as usize * 3)
        .enumerate()
        .skip((candidate.top + patch_top) as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, row)| {
            let local_y = global_y - (candidate.top + patch_top) as usize;
            let source_start = local_y * patch_rgb_stride;
            let destination_start = (candidate.left + patch_left) as usize * 3;
            row[destination_start..destination_start + patch_rgb_stride]
                .copy_from_slice(&blended_pixels[source_start..source_start + patch_rgb_stride]);
        });

    // Re-apply hard ownership to the protected layer. This is deliberately
    // after multiband reconstruction: the low-frequency seam mask is allowed
    // to smooth paper illumination, but not to average two displaced copies of
    // a depth-discontinuous object.
    base.as_mut()
        .par_chunks_mut(base_width as usize * 3)
        .enumerate()
        .skip((candidate.top + patch_top) as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, row)| {
            let local_y = global_y - (candidate.top + patch_top) as usize;
            for local_x in 0..patch_width as usize {
                let patch_index = local_y * patch_width as usize + local_x;
                if protected_values[patch_index] == 0 {
                    continue;
                }
                let source_start = patch_index * 3;
                let destination_start = (candidate.left + patch_left + local_x as u32) as usize * 3;
                row[destination_start..destination_start + 3]
                    .copy_from_slice(&protected_pixels[source_start..source_start + 3]);
            }
        });
    base_mask
        .as_mut()
        .par_chunks_mut(base_width as usize)
        .enumerate()
        .skip((candidate.top + patch_top) as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, row)| {
            let local_y = global_y - (candidate.top + patch_top) as usize;
            for local_x in 0..patch_width {
                let source_x = patch_left + local_x;
                if candidate
                    .mask
                    .get_pixel(source_x, patch_top + local_y as u32)[0]
                    > 0
                {
                    row[(candidate.left + source_x) as usize] = 255;
                }
            }
        });

    // Pixels outside the narrow seam patch still need the normal hard ownership
    // decision; only the low-frequency transition is allowed to cross the seam.
    hard_select_focus_layer_outside_patch(
        base,
        base_mask,
        candidate,
        decision_mask,
        patch_left,
        patch_right,
        patch_top,
        patch_bottom,
    );
}

pub(crate) fn hard_select_focus_layer_outside_patch(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
    excluded_left: u32,
    excluded_right: u32,
    excluded_top: u32,
    excluded_bottom: u32,
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
    let base_pixels = base.as_mut();
    let base_mask_pixels = base_mask.as_mut();
    let candidate_pixels = candidate.image.as_raw();
    let candidate_mask_pixels = candidate.mask.as_raw();
    let decision_pixels = decision_mask.as_raw();
    for local_y in 0..layer_height as usize {
        let global_y = candidate.top as usize + local_y;
        let base_row = &mut base_pixels[global_y * base_stride..(global_y + 1) * base_stride];
        let base_mask_row = &mut base_mask_pixels
            [global_y * base_width as usize..(global_y + 1) * base_width as usize];
        let candidate_row = &candidate_pixels[local_y * layer_stride..(local_y + 1) * layer_stride];
        let candidate_mask_row = &candidate_mask_pixels
            [local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
        let decision_row =
            &decision_pixels[local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
        for local_x in 0..layer_width as usize {
            if (excluded_left as usize..=excluded_right as usize).contains(&local_x)
                && (excluded_top as usize..=excluded_bottom as usize).contains(&local_y)
            {
                continue;
            }
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
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MappedFrameGeometry {
    pub(crate) center: Point2<f64>,
    pub(crate) scale: f64,
    pub(crate) rotation: f64,
    pub(crate) area: f64,
}

pub(crate) fn mapped_frame_geometry(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Option<MappedFrameGeometry> {
    let width = image.width.max(1) as f64;
    let height = image.height.max(1) as f64;
    let source_corners = [
        Point2::new(0.0, 0.0),
        Point2::new(width, 0.0),
        Point2::new(width, height),
        Point2::new(0.0, height),
    ];
    let mut corners = [Point2::new(0.0, 0.0); 4];
    for (slot, source) in source_corners.into_iter().enumerate() {
        let projected = project_point(image, source.x, source.y, projection)?;
        let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            return None;
        }
        let point = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
        if !point.x.is_finite() || !point.y.is_finite() {
            return None;
        }
        corners[slot] = point;
    }

    let center = corners
        .iter()
        .copied()
        .fold(Point2::new(0.0, 0.0), |sum, point| sum + point.coords)
        / 4.0;
    let horizontal_scale =
        ((corners[1] - corners[0]).norm() + (corners[2] - corners[3]).norm()) / (2.0 * width);
    let vertical_scale =
        ((corners[3] - corners[0]).norm() + (corners[2] - corners[1]).norm()) / (2.0 * height);
    let scale = (horizontal_scale.max(0.0) * vertical_scale.max(0.0)).sqrt();
    let top_edge = corners[1] - corners[0];
    let rotation = top_edge.y.atan2(top_edge.x);
    let signed_double_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>();
    let area = (signed_double_area / (2.0 * width * height)).abs();
    (scale.is_finite() && rotation.is_finite() && area.is_finite()).then_some(MappedFrameGeometry {
        center,
        scale,
        rotation,
        area,
    })
}

pub(crate) fn wrapped_angle_delta(left: f64, right: f64) -> f64 {
    let mut delta = left - right;
    while delta > std::f64::consts::PI {
        delta -= 2.0 * std::f64::consts::PI;
    }
    while delta < -std::f64::consts::PI {
        delta += 2.0 * std::f64::consts::PI;
    }
    delta.abs()
}
