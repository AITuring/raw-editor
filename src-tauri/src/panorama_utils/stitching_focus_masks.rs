use super::*;

pub(crate) struct SeamBandBlend<'a> {
    pub(crate) panorama: &'a mut Rgb32FImage,
    pub(crate) panorama_mask: &'a mut GrayImage,
    pub(crate) img_to_add_info: &'a ImageInfo,
    pub(crate) img_to_add: &'a Rgb32FImage,
    pub(crate) h_add: &'a Matrix3<f64>,
    pub(crate) projection: Projection,
    pub(crate) offset_x: f64,
    pub(crate) offset_y: f64,
    pub(crate) orientation: SeamOrientation,
    pub(crate) seam_coords: &'a [i32],
    pub(crate) new_image_is_dominant_side: bool,
    pub(crate) min_x: u32,
    pub(crate) max_x: u32,
    pub(crate) min_y: u32,
    pub(crate) max_y: u32,
    pub(crate) exposure: &'a ExposureCompensation,
}

pub(crate) fn blend_panorama_seam_band(ctx: SeamBandBlend<'_>) {
    let SeamBandBlend {
        panorama,
        panorama_mask,
        img_to_add_info,
        img_to_add,
        h_add,
        projection,
        offset_x,
        offset_y,
        orientation,
        seam_coords,
        new_image_is_dominant_side,
        min_x,
        max_x,
        min_y,
        max_y,
        exposure,
    } = ctx;
    let (out_width, out_height) = panorama.dimensions();
    if out_width == 0 || out_height == 0 || seam_coords.is_empty() {
        return;
    }

    let overlap_left = min_x.min(out_width - 1);
    let overlap_right = max_x.min(out_width - 1);
    let overlap_top = min_y.min(out_height - 1);
    let overlap_bottom = max_y.min(out_height - 1);
    if overlap_left > overlap_right || overlap_top > overlap_bottom {
        return;
    }

    let seam_value = |index: usize| -> Option<u32> {
        seam_coords
            .get(index)
            .copied()
            .map(|value| value.max(0) as u32)
    };

    match orientation {
        SeamOrientation::Horizontal if seam_coords.len() <= overlap_right as usize => return,
        SeamOrientation::Vertical if seam_coords.len() <= overlap_bottom as usize => return,
        _ => {}
    }
    // Keep the complete overlap so the low-frequency mask can transition across the
    // whole shared field rather than inheriting a tonal step around the detail seam.
    let (patch_left, patch_top, patch_right, patch_bottom) =
        (overlap_left, overlap_top, overlap_right, overlap_bottom);

    if patch_left > patch_right || patch_top > patch_bottom {
        return;
    }
    let patch_width = patch_right - patch_left + 1;
    let patch_height = patch_bottom - patch_top + 1;
    if patch_width < 8 || patch_height < 8 {
        return;
    }

    let Some(h_add_inv) = h_add.try_inverse() else {
        return;
    };
    let patch_pixel_count = patch_width as usize * patch_height as usize;
    let mut base_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut candidate_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut blend_mask = vec![0u8; patch_pixel_count];
    let mut low_frequency_mask = vec![0u8; patch_pixel_count];
    let image_width = img_to_add.width() as f64;
    let image_height = img_to_add.height() as f64;
    let patch_rgb_stride = patch_width as usize * 3;
    let patch_mask_stride = patch_width as usize;
    let panorama_ref: &Rgb32FImage = panorama;
    let panorama_mask_ref: &GrayImage = panorama_mask;

    base_pixels
        .par_chunks_mut(patch_rgb_stride)
        .zip(candidate_pixels.par_chunks_mut(patch_rgb_stride))
        .zip(blend_mask.par_chunks_mut(patch_mask_stride))
        .zip(low_frequency_mask.par_chunks_mut(patch_mask_stride))
        .enumerate()
        .for_each(
            |(local_y, (((base_row, candidate_row), blend_row), low_frequency_row))| {
                let local_y = local_y as u32;
                let global_y = patch_top + local_y;
                for local_x in 0..patch_width {
                    let global_x = patch_left + local_x;
                    let target =
                        Point3::new(global_x as f64 - offset_x, global_y as f64 - offset_y, 1.0);
                    let candidate_source =
                        map_target_to_source(&h_add_inv, target, img_to_add_info, projection);
                    let candidate_valid = candidate_source.as_ref().is_some_and(|source| {
                        source.x >= 0.0
                            && source.x < image_width
                            && source.y >= 0.0
                            && source.y < image_height
                            // A virtual tile is rendered into a rectangular
                            // buffer, but its projective corners are transparent
                            // zero support. Treat those samples as outside the
                            // candidate footprint so they cannot enter either
                            // the low-frequency pyramid or the detail seam.
                            && source_sample_has_support(img_to_add, source.x, source.y)
                    });
                    let candidate_pixel = if let Some(source) = candidate_source {
                        if candidate_valid {
                            apply_exposure_compensation(
                                get_high_quality_interpolated_pixel(img_to_add, source.x, source.y),
                                exposure,
                                global_x,
                                global_y,
                            )
                        } else {
                            Rgb([0.0, 0.0, 0.0])
                        }
                    } else {
                        Rgb([0.0, 0.0, 0.0])
                    };
                    let panorama_valid = panorama_mask_ref.get_pixel(global_x, global_y)[0] > 0;
                    let current_pixel = *panorama_ref.get_pixel(global_x, global_y);
                    let candidate_pixel = if !candidate_valid && panorama_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };
                    let base_pixel = if panorama_valid || !candidate_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };

                    let candidate_signed_distance = if !candidate_valid {
                        f32::NEG_INFINITY
                    } else if !panorama_valid {
                        f32::INFINITY
                    } else {
                        match orientation {
                            SeamOrientation::Horizontal => {
                                let seam_y = seam_value(global_x as usize).unwrap_or(global_y);
                                let distance = global_y as f32 - seam_y as f32;
                                if new_image_is_dominant_side {
                                    distance
                                } else {
                                    -distance
                                }
                            }
                            SeamOrientation::Vertical => {
                                let seam_x = seam_value(global_y as usize).unwrap_or(global_x);
                                let distance = global_x as f32 - seam_x as f32;
                                if new_image_is_dominant_side {
                                    distance
                                } else {
                                    -distance
                                }
                            }
                        }
                    };
                    let detail_alpha = panorama_detail_alpha(candidate_signed_distance);

                    let base_start = local_x as usize * 3;
                    base_row[base_start..base_start + 3].copy_from_slice(&base_pixel.0);
                    candidate_row[base_start..base_start + 3].copy_from_slice(&candidate_pixel.0);
                    blend_row[local_x as usize] = (detail_alpha * 255.0).round() as u8;
                    let low_frequency_alpha = if !candidate_valid {
                        0.0
                    } else if !panorama_valid {
                        1.0
                    } else {
                        match orientation {
                            SeamOrientation::Horizontal => {
                                let span = (overlap_bottom - overlap_top).max(1) as f32;
                                let position = (global_y.saturating_sub(overlap_top)) as f32 / span;
                                if new_image_is_dominant_side {
                                    position
                                } else {
                                    1.0 - position
                                }
                            }
                            SeamOrientation::Vertical => {
                                let span = (overlap_right - overlap_left).max(1) as f32;
                                let position =
                                    (global_x.saturating_sub(overlap_left)) as f32 / span;
                                if new_image_is_dominant_side {
                                    position
                                } else {
                                    1.0 - position
                                }
                            }
                        }
                    };
                    low_frequency_row[local_x as usize] =
                        (low_frequency_alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            },
        );

    let base = Rgb32FImage::from_raw(patch_width, patch_height, base_pixels)
        .expect("seam base patch dimensions must match");
    let candidate = Rgb32FImage::from_raw(patch_width, patch_height, candidate_pixels)
        .expect("seam candidate patch dimensions must match");
    let mask = GrayImage::from_raw(patch_width, patch_height, blend_mask.clone())
        .expect("seam mask dimensions must match");
    let low_frequency_mask = GrayImage::from_raw(patch_width, patch_height, low_frequency_mask)
        .expect("low-frequency seam mask dimensions must match");
    // Keep fine and middle-frequency Laplacian detail on one source side of the
    // path. Averaging those bands across the complete overlap softens strokes,
    // seals, foliage, and other structure whenever registration is not literally
    // pixel-identical. Only broad tonal bands may transition across the complete
    // overlap; visible detail transitions stay local to the optimal seam.
    let blended = multiband_blend(
        base,
        candidate,
        mask,
        Some(low_frequency_mask),
        PANORAMA_BLEND_BANDS,
        // A residual sub-pixel registration error must never be averaged into
        // the visible detail bands. Keep the source ownership hard for the
        // finest and middle frequencies; only the coarse tone pyramid is
        // allowed to feather across the seam.
        true,
    );

    let panorama_rgb_stride = out_width as usize * 3;
    let blended_pixels = blended.as_raw();
    panorama
        .as_mut()
        .par_chunks_mut(panorama_rgb_stride)
        .zip(panorama_mask.as_mut().par_chunks_mut(out_width as usize))
        .enumerate()
        .skip(patch_top as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, (panorama_row, panorama_mask_row))| {
            let local_y = global_y - patch_top as usize;
            let blended_row =
                &blended_pixels[local_y * patch_rgb_stride..(local_y + 1) * patch_rgb_stride];
            let destination_start = patch_left as usize * 3;
            let destination_end = destination_start + patch_rgb_stride;
            panorama_row[destination_start..destination_end].copy_from_slice(blended_row);

            let blend_row =
                &blend_mask[local_y * patch_mask_stride..(local_y + 1) * patch_mask_stride];
            for (local_x, candidate_owns_pixel) in blend_row.iter().copied().enumerate() {
                let global_x = patch_left as usize + local_x;
                if panorama_mask_row[global_x] > 0 || candidate_owns_pixel > 0 {
                    panorama_mask_row[global_x] = 255;
                }
            }
        });
}

pub(crate) fn luminance(pixel: &Rgb<f32>) -> f32 {
    pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114
}

pub(crate) fn focus_stack_pixel_is_canvas_like(pixel: &[f32]) -> bool {
    if pixel.len() < 3 || pixel[..3].iter().any(|value| !value.is_finite()) {
        return false;
    }
    let red = pixel[0].clamp(0.0, 1.0);
    let green = pixel[1].clamp(0.0, 1.0);
    let blue = pixel[2].clamp(0.0, 1.0);
    let luma = red * 0.299 + green * 0.587 + blue * 0.114;
    let red_floor = red.max(0.001);
    let green_ratio = green / red_floor;
    let blue_ratio = blue / red_floor;
    let red_dominance = red - 2.0 * green + blue;
    // The source canvas is a warm, moderately desaturated brown. This gate
    // deliberately rejects the red skirt, green sash, near-black robe, and
    // pale faces/hands before a low-frequency exposure field is applied. The
    // limits are broad enough to retain the darker and lighter canvas tiles.
    // Darkened paper from a long scan can fall below the original 0.12
    // floor.  Keep a small margin above true black ink so the low-frequency
    // tone field can still see those paper samples.
    luma >= 0.08
        && luma <= 0.78
        && red >= green
        && green >= blue
        && green_ratio >= 0.42
        && blue_ratio >= 0.24
        && red - green >= 0.025
        && green - blue >= 0.015
        && red_dominance <= 0.08
}

pub(crate) fn focus_stack_pixel_is_tone_foreground(pixel: &[f32]) -> bool {
    if pixel.len() < 3 || pixel[..3].iter().any(|value| !value.is_finite()) {
        return false;
    }
    let red = pixel[0].clamp(0.0, 1.0);
    let green = pixel[1].clamp(0.0, 1.0);
    let blue = pixel[2].clamp(0.0, 1.0);
    let luma = red * 0.299 + green * 0.587 + blue * 0.114;
    let chroma = red.max(green).max(blue) - red.min(green).min(blue);
    let red_dominance = red - 2.0 * green + blue;
    let red_paint = red - green >= 0.14 && red - blue >= 0.17 && red - 2.0 * green + blue >= 0.08;
    let green_or_cool_paint = green - red >= 0.05 || blue - green >= 0.05;
    // Brown paper can be dark and nearly neutral.  Reserve this guard for
    // genuinely near-black ink; otherwise dark paper is excluded from tone
    // estimation and remains as a visible tile-sized exposure block.
    let dark_neutral_paint =
        luma <= 0.12 && (red - green).abs() <= 0.07 && (green - blue).abs() <= 0.07;
    // Faces and hands can have nearly the same average luminance as the
    // canvas, but their blue channel is much closer to green and their red
    // dominance is stronger. Keep this warm-pale paint out of the canvas-only
    // correction without admitting the flatter brown paper tone.
    let warm_light_paint = luma >= 0.22
        && red - green >= 0.05
        && red - blue >= 0.08
        && green - blue <= 0.08
        && red_dominance >= 0.04;
    // Very pale sleeves, hands, and faces are outside the brown canvas range.
    // This is intentionally narrower than a generic bright-pixel detector so
    // a lighter canvas exposure can still be harmonized.
    let pale_paint = luma >= 0.78 && chroma <= 0.22;
    red_paint || green_or_cool_paint || dark_neutral_paint || warm_light_paint || pale_paint
}

pub(crate) struct RenderedFocusAnalysisLayer {
    pub(crate) image: Rgb32FImage,
    pub(crate) mask: GrayImage,
    pub(crate) foreground_mask: GrayImage,
    pub(crate) relaxed_foreground_mask: GrayImage,
}

pub(crate) fn focus_analysis_dimensions(
    width: u32,
    height: u32,
    source_longest_dimension: u32,
) -> (u32, u32) {
    let longest_side = width.max(height).max(1);
    // Keep the focus-analysis sampling density tied to an individual source,
    // not to the total mosaic length. Otherwise a long shifted stack compresses
    // each 9504px frame to a few hundred analysis pixels and erases the very
    // sharpness differences the stacker is supposed to select.
    let source_scale = FOCUS_ANALYSIS_MAX_DIMENSION as f64 / source_longest_dimension.max(1) as f64;
    let mut scale = source_scale.min(1.0);
    let scaled_pixels = width as f64 * height as f64 * scale * scale;
    if scaled_pixels > FOCUS_ANALYSIS_MAX_PIXELS as f64 {
        scale *= (FOCUS_ANALYSIS_MAX_PIXELS as f64 / scaled_pixels).sqrt();
    }
    if scale >= 1.0 && longest_side <= FOCUS_ANALYSIS_MAX_DIMENSION {
        return (width.max(1), height.max(1));
    }
    (
        (width as f64 * scale).round().max(1.0) as u32,
        (height as f64 * scale).round().max(1.0) as u32,
    )
}

pub(crate) fn box_blur_focus_map(
    source: &[f32],
    width: u32,
    height: u32,
    radius: usize,
) -> Vec<f32> {
    if radius == 0 || width == 0 || height == 0 {
        return source.to_vec();
    }
    let width = width as usize;
    let height = height as usize;
    let window_size = radius * 2 + 1;
    let divisor = window_size as f32;
    let mut horizontal = vec![0.0f32; source.len()];
    horizontal
        .par_chunks_mut(width)
        .zip(source.par_chunks(width))
        .for_each(|(output_row, source_row)| {
            let mut sum = 0.0f32;
            for offset in 0..window_size {
                sum += source_row[offset.saturating_sub(radius).min(width - 1)];
            }
            output_row[0] = sum / divisor;
            for (x, output) in output_row.iter_mut().enumerate().skip(1) {
                let add_x = (x + radius).min(width - 1);
                let remove_x = x.saturating_sub(radius + 1);
                sum += source_row[add_x] - source_row[remove_x];
                *output = sum / divisor;
            }
        });

    let mut output = vec![0.0f32; source.len()];
    for x in 0..width {
        let mut sum = 0.0f32;
        for offset in 0..window_size {
            let y = offset.saturating_sub(radius).min(height - 1);
            sum += horizontal[y * width + x];
        }
        output[x] = sum / divisor;
        for y in 1..height {
            let add_y = (y + radius).min(height - 1);
            let remove_y = y.saturating_sub(radius + 1);
            sum += horizontal[add_y * width + x] - horizontal[remove_y * width + x];
            output[y * width + x] = sum / divisor;
        }
    }
    output
}

pub(crate) fn focus_score_map(image: &Rgb32FImage, mask: &GrayImage) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let pixel_count = width as usize * height as usize;
    let mut luminance_map = vec![0.0f32; pixel_count];
    luminance_map
        .par_iter_mut()
        .zip(image.as_raw().par_chunks(3))
        .for_each(|(output, pixel)| {
            *output = pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114;
        });

    let mut raw_score = vec![0.0f32; pixel_count];
    if width >= 3 && height >= 3 {
        raw_score
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                if y == 0 || y + 1 >= height as usize {
                    return;
                }
                for (x, output) in row.iter_mut().enumerate().take(width as usize - 1).skip(1) {
                    let index = y * width as usize + x;
                    if mask.as_raw()[index] == 0
                        || mask.as_raw()[index - 1] == 0
                        || mask.as_raw()[index + 1] == 0
                        || mask.as_raw()[index - width as usize] == 0
                        || mask.as_raw()[index + width as usize] == 0
                    {
                        continue;
                    }
                    let laplacian = (4.0 * luminance_map[index]
                        - luminance_map[index - 1]
                        - luminance_map[index + 1]
                        - luminance_map[index - width as usize]
                        - luminance_map[index + width as usize])
                        .abs();
                    // A Laplacian alone under-scores broad, low-contrast
                    // brush strokes and woven paper texture.  Add a bounded
                    // Tenengrad term so a genuinely focused layer can win on
                    // those structures without making isolated sensor noise
                    // dominate the ownership map.
                    let gradient_x = luminance_map[index + 1] - luminance_map[index - 1];
                    let gradient_y = luminance_map[index + width as usize]
                        - luminance_map[index - width as usize];
                    let gradient = (gradient_x * gradient_x + gradient_y * gradient_y).sqrt();
                    *output = laplacian + gradient * 0.5;
                }
            });
    }

    // Focus ownership is decided on a bounded analysis canvas. Keep the score
    // window local enough that one clear stroke cannot claim an adjacent
    // character; the final pixels still come from the full-resolution aligned
    // layer.
    let radius = ((width.max(height) as f32 / FOCUS_SCORE_BLUR_RADIUS_DIVISOR).round() as usize)
        .clamp(1, FOCUS_SCORE_MAX_BLUR_RADIUS);
    box_blur_focus_map(&raw_score, width, height, radius)
}

pub(crate) fn focus_decision_mask(
    base_focus: &[f32],
    candidate_focus: &[f32],
    base_mask: &GrayImage,
    candidate_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = base_mask.dimensions();
    let pixel_count = width as usize * height as usize;
    let mut fine_advantage = vec![0.0f32; pixel_count];
    fine_advantage
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, output)| {
            let base_valid = base_mask.as_raw()[index] > 0;
            let candidate_valid = candidate_mask.as_raw()[index] > 0;
            *output = match (base_valid, candidate_valid) {
                (false, true) => 1.0,
                (true, false) => -1.0,
                (true, true) => {
                    let base = base_focus[index];
                    let candidate = candidate_focus[index];
                    (candidate - base) / (candidate + base + 1e-6)
                }
                (false, false) => 0.0,
            };
        });

    // The old radius was effectively ~100 source pixels for a 9504px frame.
    // That let a clear stroke in one region claim a neighbouring blurred
    // character. Keep the decision coherent, but make the neighbourhood local
    // enough that each character can select its own sharp source.
    let coherence_radius = ((width.max(height) as f32 / FOCUS_DECISION_COHERENCE_RADIUS_DIVISOR)
        .round() as usize)
        .clamp(2, FOCUS_DECISION_MAX_COHERENCE_RADIUS);
    let coarse_advantage = box_blur_focus_map(&fine_advantage, width, height, coherence_radius);

    let sample_step = (pixel_count / 4096).max(1);
    let mut magnitude_samples = (0..pixel_count)
        .step_by(sample_step)
        .filter_map(|index| {
            (base_mask.as_raw()[index] > 0 || candidate_mask.as_raw()[index] > 0)
                .then_some(base_focus[index].max(candidate_focus[index]))
        })
        .collect::<Vec<_>>();
    magnitude_samples.sort_unstable_by(f32::total_cmp);
    let confidence_floor = magnitude_samples
        .get(((magnitude_samples.len().saturating_sub(1)) as f32 * 0.9) as usize)
        .copied()
        .unwrap_or(0.0)
        * 0.04;

    // A defocused subject spills contrast beyond its true silhouette. That halo can
    // look locally sharper than the clean background in the focused layer, leaving
    // detached slivers around depth discontinuities even with hard pixel selection.
    // Let decisive, valid structure claim a small adjacent ambiguous band so the
    // focused silhouette also supplies the clean pixels immediately around it.
    let mut strong_base = vec![0.0f32; pixel_count];
    let mut strong_candidate = vec![0.0f32; pixel_count];
    strong_base
        .par_iter_mut()
        .zip(strong_candidate.par_iter_mut())
        .enumerate()
        .for_each(|(index, (base_claim, candidate_claim))| {
            let both_valid = base_mask.as_raw()[index] > 0 && candidate_mask.as_raw()[index] > 0;
            let confident = base_focus[index].max(candidate_focus[index]) > confidence_floor;
            if !both_valid || !confident {
                return;
            }
            let advantage = fine_advantage[index];
            if advantage < -FOCUS_DECISIVE_ADVANTAGE {
                *base_claim = 1.0;
            } else if advantage > FOCUS_DECISIVE_ADVANTAGE {
                *candidate_claim = 1.0;
            }
        });
    let protection_radius = ((width.max(height) as f32 / 2048.0)
        * (FOCUS_EDGE_PROTECTION_AT_1024 * FOCUS_EDGE_PROTECTION_SCALE))
        .round()
        .clamp(2.0, 6.0) as usize;
    let base_claim = box_blur_focus_map(&strong_base, width, height, protection_radius);
    let candidate_claim = box_blur_focus_map(&strong_candidate, width, height, protection_radius);

    GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        let base_valid = base_mask.as_raw()[index] > 0;
        let candidate_valid = candidate_mask.as_raw()[index] > 0;
        let confident = base_focus[index].max(candidate_focus[index]) > confidence_floor;
        // Always use the coherent neighborhood decision for ownership. A single
        // high-contrast stroke can be displaced by a couple of pixels between
        // captures; using its raw per-pixel score would then alternate sources
        // across the stroke and recreate a double contour.
        let advantage = coarse_advantage[index];
        let nearby_base_claim = base_claim[index];
        let nearby_candidate_claim = candidate_claim[index];
        let structural_winner = if nearby_candidate_claim > FOCUS_EDGE_CLAIM_THRESHOLD
            && nearby_candidate_claim > nearby_base_claim
        {
            Some(true)
        } else if nearby_base_claim > FOCUS_EDGE_CLAIM_THRESHOLD
            && nearby_base_claim > nearby_candidate_claim
        {
            Some(false)
        } else {
            None
        };
        let candidate_wins = candidate_valid
            && (!base_valid
                || structural_winner.unwrap_or(confident && advantage > FOCUS_CONFIDENCE_MARGIN));
        image::Luma([if candidate_wins { 255 } else { 0 }])
    })
}

pub(crate) fn suppress_focus_canvas_switches(
    decision_mask: &mut GrayImage,
    base: &Rgb32FImage,
    candidate: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate_mask: &GrayImage,
) {
    if decision_mask.dimensions() != base.dimensions()
        || base.dimensions() != candidate.dimensions()
        || base_mask.dimensions() != base.dimensions()
        || candidate_mask.dimensions() != base.dimensions()
    {
        return;
    }

    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(base.as_raw().par_chunks(3))
        .zip(candidate.as_raw().par_chunks(3))
        .zip(base_mask.as_raw().par_iter())
        .zip(candidate_mask.as_raw().par_iter())
        .for_each(
            |((((decision, base_pixel), candidate_pixel), base_valid), candidate_valid)| {
                // The canvas is one planar surface. Sharpness fluctuations in its
                // weave must not make the focus score swap the source for an
                // already-covered pixel; those swaps are the rectangular exposure
                // blocks visible in a shifted scan. A painted pixel can still win
                // normally, and a new candidate can still fill an uncovered hole.
                if *decision > 0
                    && *base_valid > 0
                    && *candidate_valid > 0
                    && focus_stack_pixel_is_canvas_like(base_pixel)
                    && focus_stack_pixel_is_canvas_like(candidate_pixel)
                    && !focus_stack_pixel_is_tone_foreground(base_pixel)
                    && !focus_stack_pixel_is_tone_foreground(candidate_pixel)
                {
                    *decision = 0;
                }
            },
        );
}

pub(crate) fn resize_binary_mask(mask: &GrayImage, width: u32, height: u32) -> GrayImage {
    image::imageops::resize(
        mask,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Nearest,
    )
}

pub(crate) fn dilate_focus_binary_mask(mask: &GrayImage, radius: usize) -> GrayImage {
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

pub(crate) fn open_focus_binary_mask(mask: &GrayImage, radius: usize) -> GrayImage {
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

pub(crate) fn fill_focus_foreground_holes(mask: &GrayImage) -> GrayImage {
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

pub(crate) fn retain_focus_foreground_components(
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

pub(crate) fn build_focus_background_foreground_envelope(
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
