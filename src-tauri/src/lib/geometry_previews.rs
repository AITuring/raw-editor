//! Crop and geometry previews and the crop analysis commands.
//!
//! Uncropped, original-transformed and geometry-transform previews, straighten
//! and Upright analysis, and access to the loaded original image.

use crate::adjustment_utils::apply_all_transformations;
use crate::adjustment_utils::hydrate_adjustments;
use crate::app_settings::load_settings;
use crate::app_state::AppState;
use crate::app_state::SharedMaskBitmap;
use crate::cache_utils::GEOMETRY_KEYS;
use crate::cache_utils::calculate_full_job_hash;
use crate::cache_utils::calculate_visual_hash;
use crate::cache_utils::dynamic_image_weight;
use crate::color_management::srgb_preview_encoder;
use crate::crop_analysis::StraightenAnalysis;
use crate::crop_analysis::UprightAnalysis;
use crate::crop_analysis::UprightMode;
use crate::image_loader::composite_patches_on_image;
use crate::image_processing::GeometryParams;
use crate::image_processing::RenderRequest;
use crate::image_processing::apply_coarse_rotation;
use crate::image_processing::apply_cpu_default_raw_processing;
use crate::image_processing::apply_flip;
use crate::image_processing::apply_geometry_warp;
use crate::image_processing::downscale_f32_image;
use crate::image_processing::get_all_adjustments_from_json;
use crate::image_processing::get_or_init_gpu_context;
use crate::image_processing::process_and_get_dynamic_image;
use crate::image_processing::resolve_tonemapper_override_from_handle;
use crate::image_processing::warp_image_geometry;
use crate::mask_generation::MaskDefinition;
use crate::mask_generation::get_cached_or_generate_mask;
use image::DynamicImage;
use image::GenericImageView;
use image::Rgba;
use imageproc::drawing::draw_line_segment_mut;
use imageproc::edges::canny;
use imageproc::hough::LineDetectionOptions;
use imageproc::hough::detect_lines;
use mozjpeg_rs::Preset;
use mozjpeg_rs::Subsampling;
use std::borrow::Cow;
use tauri::Manager;
use tauri::ipc::Response;

use super::{crop_analysis, lut_processing};

#[tauri::command]
pub(super) async fn generate_uncropped_preview(
    js_adjustments: serde_json::Value,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Response, String> {
    let context = get_or_init_gpu_context(&state, &app_handle)?;
    let mut adjustments_clone = js_adjustments.clone();
    hydrate_adjustments(&state, &mut adjustments_clone);

    let loaded_image = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded")?;

    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
        let state = app_handle.state::<AppState>();
        let path = loaded_image.path.clone();
        let is_raw = loaded_image.is_raw;
        let unique_hash = calculate_full_job_hash(&path, &adjustments_clone);
        let settings = load_settings(app_handle.clone()).unwrap_or_default();
        let preview_dim = settings.editor_preview_resolution.unwrap_or(1920);
        let has_patches = adjustments_clone
            .get("aiPatches")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty());
        let patched_image = if has_patches {
            Cow::Owned(
                composite_patches_on_image(&loaded_image.image, &adjustments_clone).unwrap_or_else(
                    |e| {
                        eprintln!("Failed to composite patches for uncropped preview: {}", e);
                        loaded_image.image.as_ref().clone()
                    },
                ),
            )
        } else {
            Cow::Borrowed(loaded_image.image.as_ref())
        };

        let (source_width, source_height) = patched_image.dimensions();
        let preview_source = if source_width > preview_dim || source_height > preview_dim {
            Cow::Owned(downscale_f32_image(
                patched_image.as_ref(),
                preview_dim,
                preview_dim,
            ))
        } else {
            patched_image
        };

        // Geometry is resolution invariant because its perspective and offset
        // terms are normalized by the image dimensions. Warping the static
        // preview instead of a 40-60 MP original keeps crop sliders responsive
        // while the full-resolution export continues through the same mapping.
        let warped_image = apply_geometry_warp(preview_source, &adjustments_clone);
        let blurred_image = crate::lens_blur::apply_lens_blur(warped_image, &adjustments_clone);
        let orientation_steps = adjustments_clone["orientationSteps"].as_u64().unwrap_or(0) as u8;
        let coarse_rotated_image = apply_coarse_rotation(blurred_image, orientation_steps);

        let flip_horizontal = adjustments_clone["flipHorizontal"]
            .as_bool()
            .unwrap_or(false);
        let flip_vertical = adjustments_clone["flipVertical"].as_bool().unwrap_or(false);

        let flipped_image =
            apply_flip(coarse_rotated_image, flip_horizontal, flip_vertical).into_owned();

        let full_oriented_width = if orientation_steps % 2 == 1 {
            source_height
        } else {
            source_width
        };
        let scale_for_gpu = if full_oriented_width > 0 {
            flipped_image.width() as f32 / full_oriented_width as f32
        } else {
            1.0
        };
        let processing_base = flipped_image;

        let (preview_width, preview_height) = processing_base.dimensions();

        let mask_definitions: Vec<MaskDefinition> = adjustments_clone
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let mask_bitmaps: Vec<SharedMaskBitmap> = mask_definitions
            .iter()
            .filter_map(|def| {
                get_cached_or_generate_mask(
                    &state,
                    def,
                    preview_width,
                    preview_height,
                    scale_for_gpu,
                    (0.0, 0.0),
                    &adjustments_clone,
                )
            })
            .collect();

        let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
        let uncropped_adjustments =
            get_all_adjustments_from_json(&adjustments_clone, is_raw, tm_override);
        let lut_path = adjustments_clone["lutPath"].as_str();
        let lut = lut_path.and_then(|p| lut_processing::get_or_load_lut(&state, p).ok());

        let processed_image = process_and_get_dynamic_image(
            &context,
            &state,
            &processing_base,
            unique_hash,
            RenderRequest {
                adjustments: uncropped_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: None,
            },
            "generate_uncropped_preview",
        )?;

        let (width, height) = processed_image.dimensions();
        let rgba_pixels = processed_image.to_rgba8();
        let mut rgb_pixels = Vec::with_capacity((width * height * 3) as usize);
        for (index, pixel) in rgba_pixels.pixels().enumerate() {
            let x = index as u32 % width;
            let y = index as u32 / width;
            let checker = if ((x / 12) + (y / 12)).is_multiple_of(2) {
                35u8
            } else {
                47u8
            };
            let alpha = pixel[3] as u16;
            let inverse_alpha = 255 - alpha;
            for channel in 0..3 {
                let composited =
                    (pixel[channel] as u16 * alpha + checker as u16 * inverse_alpha + 127) / 255;
                rgb_pixels.push(composited as u8);
            }
        }
        let encoded = srgb_preview_encoder(Preset::BaselineFastest)
            .quality(82)
            .encode_rgb(&rgb_pixels, width, height)
            .map_err(|error| {
                format!("Failed to encode uncropped preview with mozjpeg-rs: {error}")
            })?;
        Ok(encoded)
    })
    .await
    .map_err(|e| format!("Uncropped preview worker failed: {}", e))??;

    Ok(Response::new(bytes))
}

#[tauri::command]
pub(super) async fn analyze_crop_straighten(
    js_adjustments: serde_json::Value,
    state: tauri::State<'_, AppState>,
) -> Result<StraightenAnalysis, String> {
    let mut adjustments_clone = js_adjustments;
    hydrate_adjustments(&state, &mut adjustments_clone);

    let loaded_image = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded")?;

    tokio::task::spawn_blocking(move || {
        let mut analysis_image = downscale_f32_image(loaded_image.image.as_ref(), 1200, 1200);
        if loaded_image.is_raw {
            apply_cpu_default_raw_processing(&mut analysis_image);
        }

        let warped_image = apply_geometry_warp(Cow::Owned(analysis_image), &adjustments_clone);
        let orientation_steps = adjustments_clone["orientationSteps"].as_u64().unwrap_or(0) as u8;
        let coarse_rotated_image = apply_coarse_rotation(warped_image, orientation_steps);
        let flip_horizontal = adjustments_clone["flipHorizontal"]
            .as_bool()
            .unwrap_or(false);
        let flip_vertical = adjustments_clone["flipVertical"].as_bool().unwrap_or(false);
        let analysis_image = apply_flip(coarse_rotated_image, flip_horizontal, flip_vertical);

        Ok(crop_analysis::analyze_straighten(analysis_image.as_ref()))
    })
    .await
    .map_err(|error| format!("Straighten analysis worker failed: {error}"))?
}

#[tauri::command]
pub(super) async fn analyze_crop_upright(
    mode: UprightMode,
    js_adjustments: serde_json::Value,
    state: tauri::State<'_, AppState>,
) -> Result<UprightAnalysis, String> {
    let mut analysis_adjustments = js_adjustments;
    hydrate_adjustments(&state, &mut analysis_adjustments);
    let orientation_steps = analysis_adjustments["orientationSteps"]
        .as_u64()
        .unwrap_or(0) as u8;

    if let Some(object) = analysis_adjustments.as_object_mut() {
        object.insert("crop".to_string(), serde_json::Value::Null);
        object.insert("rotation".to_string(), serde_json::json!(0.0));
        object.insert("transformProjection".to_string(), serde_json::json!(0.0));
        object.insert("transformVertical".to_string(), serde_json::json!(0.0));
        object.insert("transformHorizontal".to_string(), serde_json::json!(0.0));
        object.insert("transformRotate".to_string(), serde_json::json!(0.0));
        object.insert("transformAspect".to_string(), serde_json::json!(0.0));
        object.insert("transformScale".to_string(), serde_json::json!(100.0));
        object.insert("transformXOffset".to_string(), serde_json::json!(0.0));
        object.insert("transformYOffset".to_string(), serde_json::json!(0.0));
    }

    let loaded_image = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded")?;

    tokio::task::spawn_blocking(move || {
        let mut analysis_image = downscale_f32_image(loaded_image.image.as_ref(), 1200, 1200);
        if loaded_image.is_raw {
            apply_cpu_default_raw_processing(&mut analysis_image);
        }
        let lens_corrected = apply_geometry_warp(Cow::Owned(analysis_image), &analysis_adjustments);
        Ok(crop_analysis::analyze_upright(
            lens_corrected.as_ref(),
            mode,
            orientation_steps,
        ))
    })
    .await
    .map_err(|error| format!("Upright analysis worker failed: {error}"))?
}

#[tauri::command]
pub(super) fn generate_original_transformed_preview(
    js_adjustments: serde_json::Value,
    target_resolution: Option<u32>,
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Response, String> {
    let loaded_image = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded")?;

    let mut adjustments_clone = js_adjustments.clone();

    if let Some(obj) = adjustments_clone.as_object_mut() {
        obj.insert(
            "lensBlurEnabled".to_string(),
            serde_json::Value::Bool(false),
        );
    }

    hydrate_adjustments(&state, &mut adjustments_clone);

    let image_for_preview = if loaded_image.is_raw {
        let mut raw_preview = loaded_image.image.as_ref().clone();
        apply_cpu_default_raw_processing(&mut raw_preview);
        Cow::Owned(raw_preview)
    } else {
        Cow::Borrowed(loaded_image.image.as_ref())
    };

    let (transformed_full_res, _unscaled_crop_offset) =
        apply_all_transformations(image_for_preview, &adjustments_clone);

    let settings = load_settings(app_handle).unwrap_or_default();
    let default_dim = settings.editor_preview_resolution.unwrap_or(1920);
    let preview_dim = target_resolution.unwrap_or(default_dim);

    let (w, h) = transformed_full_res.dimensions();
    let transformed_image = if w > preview_dim || h > preview_dim {
        Cow::Owned(downscale_f32_image(
            transformed_full_res.as_ref(),
            preview_dim,
            preview_dim,
        ))
    } else {
        transformed_full_res
    };

    let (width, height) = transformed_image.dimensions();
    let rgb_pixels = transformed_image.to_rgb8().into_vec();

    let is_full_resolution = preview_dim >= w.max(h);
    let encode_quality = if is_full_resolution { 100 } else { 94 };
    let chroma_subsampling = if is_full_resolution {
        Subsampling::S444
    } else {
        Subsampling::S420
    };

    let bytes = srgb_preview_encoder(Preset::BaselineFastest)
        .quality(encode_quality)
        .subsampling(chroma_subsampling)
        .fast_color(true)
        .encode_rgb(&rgb_pixels, width, height)
        .map_err(|e| format!("Failed to encode with mozjpeg-rs: {}", e))?;

    Ok(Response::new(bytes))
}

#[tauri::command]
pub(super) async fn preview_geometry_transform(
    params: GeometryParams,
    js_adjustments: serde_json::Value,
    show_lines: bool,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Response, String> {
    let (loaded_image_path, is_raw) = {
        let guard = state.original_image.lock().unwrap();
        let loaded = guard.as_ref().ok_or("No image loaded")?;
        (loaded.path.clone(), loaded.is_raw)
    };

    let visual_hash = calculate_visual_hash(&loaded_image_path, &js_adjustments);

    let base_image_to_warp = {
        let maybe_cached_image = state
            .geometry_cache
            .lock()
            .unwrap()
            .get(&visual_hash)
            .cloned();

        if let Some(cached_image) = maybe_cached_image {
            cached_image
        } else {
            let context = get_or_init_gpu_context(&state, &app_handle)?;

            let original_image = {
                let guard = state.original_image.lock().unwrap();
                let loaded = guard.as_ref().ok_or("No image loaded")?;
                loaded.image.clone()
            };

            let settings = load_settings(app_handle.clone()).unwrap_or_default();
            let interactive_divisor = 1.5;
            let final_preview_dim = settings.editor_preview_resolution.unwrap_or(1920);
            let target_dim = (final_preview_dim as f32 / interactive_divisor) as u32;

            let preview_base = tokio::task::spawn_blocking(move || -> DynamicImage {
                downscale_f32_image(&original_image, target_dim, target_dim)
            })
            .await
            .map_err(|e| e.to_string())?;

            let mut temp_adjustments = js_adjustments.clone();
            hydrate_adjustments(&state, &mut temp_adjustments);

            if let Some(obj) = temp_adjustments.as_object_mut() {
                obj.insert("crop".to_string(), serde_json::Value::Null);
                obj.insert("rotation".to_string(), serde_json::json!(0.0));
                obj.insert("orientationSteps".to_string(), serde_json::json!(0));
                obj.insert("flipHorizontal".to_string(), serde_json::json!(false));
                obj.insert("flipVertical".to_string(), serde_json::json!(false));
                obj.insert("lensBlurEnabled".to_string(), serde_json::json!(false));
                for key in GEOMETRY_KEYS {
                    match *key {
                        "transformScale"
                        | "lensDistortionAmount"
                        | "lensVignetteAmount"
                        | "lensTcaAmount" => {
                            obj.insert(key.to_string(), serde_json::json!(100.0));
                        }
                        "lensDistortionParams" | "lensMaker" | "lensModel" => {
                            obj.insert(key.to_string(), serde_json::Value::Null);
                        }
                        "lensDistortionEnabled" | "lensTcaEnabled" | "lensVignetteEnabled" => {
                            obj.insert(key.to_string(), serde_json::json!(true));
                        }
                        _ => {
                            obj.insert(key.to_string(), serde_json::json!(0.0));
                        }
                    }
                }
            }

            let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
            let all_adjustments =
                get_all_adjustments_from_json(&temp_adjustments, is_raw, tm_override);
            let lut_path = temp_adjustments["lutPath"].as_str();
            let lut = lut_path.and_then(|p| lut_processing::get_or_load_lut(&state, p).ok());
            let mask_bitmaps = Vec::new();

            let processed_base = process_and_get_dynamic_image(
                &context,
                &state,
                &preview_base,
                visual_hash,
                RenderRequest {
                    adjustments: all_adjustments,
                    mask_bitmaps: &mask_bitmaps,
                    lut,
                    roi: None,
                },
                "preview_geometry_transform_base_gen",
            )?;

            let mut cache = state.geometry_cache.lock().unwrap();
            cache.insert(
                visual_hash,
                processed_base.clone(),
                dynamic_image_weight(&processed_base),
            );

            processed_base
        }
    };

    let final_image = tokio::task::spawn_blocking(move || -> DynamicImage {
        let mut adjusted_params = params;

        if is_raw {
            // approximate linear vignetting correction on gamma-baked & tonemapped geometry preview
            adjusted_params.lens_vignette_amount *= 0.4;
        } else {
            adjusted_params.lens_vignette_amount *= 0.8;
        }

        let warped_image = warp_image_geometry(&base_image_to_warp, adjusted_params);
        let orientation_steps = js_adjustments["orientationSteps"].as_u64().unwrap_or(0) as u8;
        let flip_horizontal = js_adjustments["flipHorizontal"].as_bool().unwrap_or(false);
        let flip_vertical = js_adjustments["flipVertical"].as_bool().unwrap_or(false);

        let coarse_rotated_image =
            apply_coarse_rotation(Cow::Owned(warped_image), orientation_steps);
        let flipped_image =
            apply_flip(coarse_rotated_image, flip_horizontal, flip_vertical).into_owned();

        if show_lines {
            let gray_image = flipped_image.to_luma8();
            let mut visualization = flipped_image.to_rgba8();
            let edges = canny(&gray_image, 50.0, 100.0);

            let min_dim = gray_image.width().min(gray_image.height());

            let options = LineDetectionOptions {
                vote_threshold: (min_dim as f32 * 0.24) as u32,
                suppression_radius: 15,
            };

            let lines = detect_lines(&edges, options);

            for line in lines {
                let angle_deg = line.angle_in_degrees as f32;
                let angle_norm = angle_deg % 180.0;
                let alignment_threshold = 0.5;
                let is_vertical =
                    angle_norm < alignment_threshold || angle_norm > (180.0 - alignment_threshold);
                let is_horizontal = (angle_norm - 90.0).abs() < alignment_threshold;

                let color = if is_vertical || is_horizontal {
                    Rgba([0, 255, 0, 255])
                } else {
                    Rgba([255, 0, 0, 255])
                };

                let r = line.r;
                let theta_rad = angle_deg.to_radians();
                let a = theta_rad.cos();
                let b = theta_rad.sin();
                let x0 = a * r;
                let y0 = b * r;

                let dist = (visualization.width().max(visualization.height()) * 2) as f32;

                let x1 = x0 + dist * (-b);
                let y1 = y0 + dist * (a);
                let x2 = x0 - dist * (-b);
                let y2 = y0 - dist * (a);

                draw_line_segment_mut(&mut visualization, (x1, y1), (x2, y2), color);
                draw_line_segment_mut(
                    &mut visualization,
                    (x1 + a, y1 + b),
                    (x2 + a, y2 + b),
                    color,
                );
            }

            DynamicImage::ImageRgba8(visualization)
        } else {
            flipped_image
        }
    })
    .await
    .map_err(|e| e.to_string())?;

    let (width, height) = final_image.dimensions();
    let rgb_pixels = final_image.to_rgb8().into_vec();

    let bytes = srgb_preview_encoder(Preset::BaselineFastest)
        .quality(75)
        .encode_rgb(&rgb_pixels, width, height)
        .map_err(|e| format!("Failed to encode with mozjpeg-rs: {}", e))?;

    Ok(Response::new(bytes))
}

pub fn get_original_image(
    state: &tauri::State<AppState>,
) -> Result<(std::sync::Arc<image::DynamicImage>, bool), String> {
    let original_image_lock = state.original_image.lock().unwrap();
    let loaded_image = original_image_lock
        .as_ref()
        .ok_or("No original image loaded")?;
    Ok((
        std::sync::Arc::clone(&loaded_image.image),
        loaded_image.is_raw,
    ))
}
