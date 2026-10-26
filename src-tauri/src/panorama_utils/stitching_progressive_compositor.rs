use super::*;

pub fn progressive_seam_stitcher<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    // Keep the tone/residual ledgers scoped to this compositor invocation.
    // The panorama entry point also resets them at whole-run start (see the
    // handoff snippet in the stage5 report).
    tone::reset_run_records();
    residual_warp::reset_run_records();
    if images.is_empty() {
        return Ok(Rgb32FImage::new(0, 0));
    }

    let (min_x, max_x, min_y, max_y) = output_bounds(images, global_homographies, projection);

    let (offset_x, out_width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, out_height) = pixel_aligned_canvas(min_y, max_y);
    println!("  - Output canvas size: {}x{}", out_width, out_height);

    let mut panorama = Rgb32FImage::new(out_width, out_height);
    let mut panorama_mask = GrayImage::new(out_width, out_height);

    let base_img_info = images[0];
    let base_image = load_image(base_img_info)?;
    let h_base = &global_homographies[&base_img_info.id];
    let h_base_inv = h_base
        .try_inverse()
        .ok_or_else(|| "The base image alignment is not invertible.".to_string())?;
    println!("  - Placing base image: '{}'", base_img_info.filename);

    let num_pixels_per_row = out_width as usize * 3;
    let (base_left, base_right, base_top, base_bottom) = transformed_image_region(
        base_img_info,
        h_base,
        projection,
        offset_x,
        offset_y,
        out_width,
        out_height,
    )
    .unwrap_or((
        0,
        out_width.saturating_sub(1),
        0,
        out_height.saturating_sub(1),
    ));
    panorama
        .par_chunks_mut(num_pixels_per_row)
        .zip(panorama_mask.par_chunks_mut(out_width as usize))
        .enumerate()
        .skip(base_top as usize)
        .take((base_bottom - base_top + 1) as usize)
        .for_each(|(y, (row_slice, mask_row))| {
            for x in base_left..=base_right {
                let target_p = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
                if let Some(source) =
                    map_target_to_source(&h_base_inv, target_p, base_img_info, projection)
                {
                    let sx = source.x;
                    let sy = source.y;
                    if sx < 0.0
                        || sx >= base_image.width() as f64
                        || sy < 0.0
                        || sy >= base_image.height() as f64
                    {
                        continue;
                    }
                    let color = get_high_quality_interpolated_pixel(&base_image, sx, sy);
                    // Projected corners in a rendered focus tile are
                    // transparent black. They are outside the photographed
                    // support and must not become valid panorama pixels.
                    if color[0].max(color[1]).max(color[2]) <= 1e-6 {
                        continue;
                    }
                    let start = x as usize * 3;
                    row_slice[start..start + 3].copy_from_slice(&color.0);
                    mask_row[x as usize] = 255;
                }
            }
        });
    drop(base_image);

    for (i, &img_to_add_info) in images.iter().skip(1).enumerate() {
        let progress_msg = format!(
            "Stitching image {} of {}: {}",
            i + 2,
            images.len(),
            Path::new(&img_to_add_info.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        );
        let _ = app_handle.emit(progress_event, &progress_msg);
        println!("  - Progressively stitching '{}'", img_to_add_info.filename);

        let h_add = &global_homographies[&img_to_add_info.id];
        let h_add_inv = h_add.try_inverse().ok_or_else(|| {
            format!(
                "The alignment for '{}' is not invertible.",
                img_to_add_info.filename
            )
        })?;
        let img_to_add = load_image(img_to_add_info)?;
        let (candidate_left, candidate_right, candidate_top, candidate_bottom) =
            transformed_image_region(
                img_to_add_info,
                h_add,
                projection,
                offset_x,
                offset_y,
                out_width,
                out_height,
            )
            .unwrap_or((
                0,
                out_width.saturating_sub(1),
                0,
                out_height.saturating_sub(1),
            ));
        let exposure = if std::env::var_os("RAW_EDITOR_SKIP_PROGRESSIVE_EXPOSURE_GAIN").is_some() {
            ExposureCompensation {
                cell_size: 1,
                grid_width: 1,
                grid_height: 1,
                gains: vec![1.0],
                representative_gain: 1.0,
                channel_gains: vec![[1.0; 3]],
                representative_channel_gain: [1.0; 3],
            }
        } else {
            let estimated = estimate_overlap_exposure_compensation(ExposureOverlap {
                panorama: &panorama,
                panorama_mask: &panorama_mask,
                candidate: img_to_add_info,
                candidate_image: &img_to_add,
                candidate_inverse: &h_add_inv,
                projection,
                offset_x,
                offset_y,
            });
            let strength = std::env::var("RAW_EDITOR_PROGRESSIVE_EXPOSURE_STRENGTH")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite())
                .unwrap_or(1.0);
            estimated.damped(strength)
        };
        println!(
            "    - Overlap exposure gain: {:.3} with local illumination correction",
            exposure.representative_gain
        );

        let ctx = SeamContext {
            pano: &panorama,
            pano_mask: &panorama_mask,
            img_to_add_info,
            img_to_add: &img_to_add,
            h_add,
            projection,
            offset_x,
            offset_y,
            out_width,
            out_height,
            exposure: &exposure,
        };
        let seam_info = find_adaptive_seam(&ctx);

        let use_seam = if let Some(ref info) = seam_info {
            !info.coords.is_empty()
        } else {
            false
        };

        if !use_seam {
            println!("    - Warning: Could not find seam. Using simple overwrite.");
            // Observation only (requirement 12.7/12.9): the overwrite fallback
            // below is unchanged, the ledger just learns that this pair had no
            // usable overlap for a seam.
            degradation::record_run_degradation(
                degradation::COMPOSITION_NARROW_OVERLAP,
                serde_json::json!({
                    "stage": "pairwise_seam",
                    "trigger": "seam_unavailable",
                    "image": img_to_add_info.filename,
                }),
            );
        }

        let (orientation, seam_coords, new_image_is_dominant_side, seam_bounds) =
            if let Some(info) = seam_info {
                let dominant = match info.orientation {
                    SeamOrientation::Vertical => info.dx > 0.0,
                    SeamOrientation::Horizontal => info.dy > 0.0,
                };
                (
                    info.orientation,
                    info.coords,
                    dominant,
                    Some((info.min_x, info.max_x, info.min_y, info.max_y)),
                )
            } else {
                (SeamOrientation::Vertical, vec![], true, None)
            };

        // Extend the overlap correction smoothly into the newly covered side
        // of the candidate. A hard switch at the overlap bounding box makes
        // the low-frequency seam patch visible as a rectangle; this bounded
        // fade keeps the measured correction at the seam and returns to the
        // source exposure over part of one tile.
        let exposure_transition = seam_bounds.map(|(min_x, max_x, min_y, max_y)| {
            let axis_span = match orientation {
                SeamOrientation::Vertical => candidate_right.saturating_sub(candidate_left) + 1,
                SeamOrientation::Horizontal => candidate_bottom.saturating_sub(candidate_top) + 1,
            } as f32;
            let fade_length = (axis_span * 0.35).max(128.0);
            (min_x, max_x, min_y, max_y, fade_length)
        });

        if use_seam {
            let side = match orientation {
                SeamOrientation::Vertical => {
                    if new_image_is_dominant_side {
                        "right"
                    } else {
                        "left"
                    }
                }
                SeamOrientation::Horizontal => {
                    if new_image_is_dominant_side {
                        "bottom"
                    } else {
                        "top"
                    }
                }
            };
            println!("    - New image is on the {} side of the seam.", side);
        }

        panorama
            .par_chunks_mut(num_pixels_per_row)
            .zip(panorama_mask.par_chunks_mut(out_width as usize))
            .enumerate()
            .skip(candidate_top as usize)
            .take((candidate_bottom - candidate_top + 1) as usize)
            .for_each(|(y, (row_slice, mask_row))| {
                for x in candidate_left..=candidate_right {
                    let target_p = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
                    let Some(source_add) =
                        map_target_to_source(&h_add_inv, target_p, img_to_add_info, projection)
                    else {
                        continue;
                    };
                    let sx = source_add.x;
                    let sy = source_add.y;
                    let is_on_add = sx >= 0.0
                        && sx < img_to_add.width() as f64
                        && sy >= 0.0
                        && sy < img_to_add.height() as f64;
                    let is_on_pano = mask_row[x as usize] > 0;

                    if !is_on_add && !is_on_pano {
                        continue;
                    }
                    if is_on_add && is_on_pano && use_seam {
                        // Preserve both inputs across the overlap. The multiband
                        // stage below needs the untouched base to blend broad
                        // illumination independently from the detail seam.
                        continue;
                    }
                    if is_on_add {
                        let source_color = get_high_quality_interpolated_pixel(&img_to_add, sx, sy);
                        // The overlap is the only place with evidence for an
                        // exposure relation.  Apply the full robust gain
                        // there (and in the seam-band pyramid below), but do
                        // not carry that pairwise correction through the
                        // candidate's non-overlap area.  Doing so turns a
                        // legitimate scan illumination gradient into a
                        // source-sized dark/light rectangle.
                        let exposure_strength = if is_on_pano {
                            1.0
                        } else if let Some((min_x, max_x, min_y, max_y, fade_length)) =
                            exposure_transition
                        {
                            let distance = match orientation {
                                SeamOrientation::Vertical => {
                                    if x < min_x {
                                        (min_x - x) as f32
                                    } else {
                                        x.saturating_sub(max_x) as f32
                                    }
                                }
                                SeamOrientation::Horizontal => {
                                    let y_u32 = y as u32;
                                    if y_u32 < min_y {
                                        (min_y - y_u32) as f32
                                    } else {
                                        y_u32.saturating_sub(max_y) as f32
                                    }
                                }
                            };
                            (1.0 - distance / fade_length).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let color_to_add = apply_exposure_compensation_with_strength(
                            source_color,
                            &exposure,
                            x,
                            y as u32,
                            exposure_strength,
                        );
                        if color_to_add[0].max(color_to_add[1]).max(color_to_add[2]) <= 1e-6 {
                            continue;
                        }
                        let start = x as usize * 3;
                        row_slice[start..start + 3].copy_from_slice(&color_to_add.0);
                        mask_row[x as usize] = 255;
                    }
                }
            });

        if let (true, Some((min_x, max_x, min_y, max_y))) = (use_seam, seam_bounds) {
            blend_panorama_seam_band(SeamBandBlend {
                panorama: &mut panorama,
                panorama_mask: &mut panorama_mask,
                img_to_add_info,
                img_to_add: &img_to_add,
                h_add,
                projection,
                offset_x,
                offset_y,
                orientation,
                seam_coords: &seam_coords,
                new_image_is_dominant_side,
                min_x,
                max_x,
                min_y,
                max_y,
                exposure: &exposure,
            });
        }
    }

    // Virtual tiles are resampled once while being fused at the station and a
    // second time while they are projected onto the scan canvas.  The seam
    // pyramid deliberately keeps the fine bands source-owned, but that second
    // reconstruction still lowers native edge energy.  Restore only bounded
    // luma detail after all tile-level tone work; this cannot change ownership
    // or introduce colour halos at a seam.  The same pass is harmless for a
    // short ordinary panorama and can be tuned or disabled by callers that
    // need the exact pre-sharpened pixels.
    let final_sharpen_amount = std::env::var("RAW_EDITOR_FINAL_PANORAMA_SHARPEN_AMOUNT")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.42)
        .clamp(0.0, 1.25);
    if final_sharpen_amount > 0.0 {
        sharpen_focus_tile_detail(&mut panorama, &panorama_mask, final_sharpen_amount);
    }

    // Virtual tiles have projected corners and narrow ownership holes. Once
    // transparent/black corner samples are rejected, retain the full bounds
    // only when the union is densely covered; otherwise use the largest
    // all-valid rectangle so projective invalid wedges cannot become black
    // output corners.
    let panorama_dimensions = panorama.dimensions();
    let (valid_bounds, valid_coverage) = valid_mask_bounds_and_coverage(&panorama_mask);
    let cropped = if valid_coverage >= 0.995 {
        crop_to_valid_bounds(panorama, &panorama_mask)
    } else {
        crop_to_valid_rectangle(panorama, &panorama_mask)
    };
    if cropped.dimensions() != panorama_dimensions {
        println!(
            "  - Cropped progressive margins: {}x{} -> {}x{} (coverage {:.2}%, bbox {:?})",
            panorama_dimensions.0,
            panorama_dimensions.1,
            cropped.width(),
            cropped.height(),
            valid_coverage * 100.0,
            valid_bounds
        );
    }
    Ok(cropped)
}
