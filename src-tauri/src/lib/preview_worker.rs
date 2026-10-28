//! The editor preview worker and the analytics worker.
//!
//! Preview jobs render through the GPU pipeline and return JPEG bytes or a
//! viewport patch; the analytics worker computes histogram and waveform data for
//! settled frames.

use crate::adjustment_utils::hydrate_adjustments;
use crate::app_settings::load_settings;
use crate::app_state::AnalyticsJob;
use crate::app_state::AppState;
use crate::app_state::CachedPreview;
use crate::app_state::PreviewJob;
use crate::app_state::SharedMaskBitmap;
use crate::cache_utils::calculate_transform_hash;
use crate::color_management::srgb_preview_encoder;
use crate::image_processing::RenderRequest;
use crate::image_processing::get_all_adjustments_from_json;
use crate::image_processing::get_or_init_gpu_context;
use crate::image_processing::resolve_tonemapper_override_from_handle;
use crate::mask_generation::MaskDefinition;
use crate::mask_generation::get_cached_or_generate_mask;
use crate::preview_protocol::encode_preview_patch;
use crate::preview_protocol::normalized_roi_to_pixels;
use crate::render_strategy::RenderTier;
use crate::render_strategy::resolve_preview_render_tier;
use image::DynamicImage;
use image::GenericImageView;
use imgref::ImgRef;
use mozjpeg_rs::Preset;
use mozjpeg_rs::Subsampling;
use rgb::FromSlice;
use rgb::RGBA8;
use std::sync::Arc;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use tauri::Emitter;
use tauri::Manager;

use super::{generate_transformed_preview, image_processing, lut_processing};

pub(super) fn should_output_to_native_display(
    native_display_enabled: bool,
    has_native_display: bool,
    width: u32,
    height: u32,
    max_texture_dimension: u32,
) -> bool {
    native_display_enabled
        && has_native_display
        && width <= max_texture_dimension
        && height <= max_texture_dimension
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_preview_job(
    app_handle: &tauri::AppHandle,
    state: tauri::State<AppState>,
    mut adjustments_json: serde_json::Value,
    is_interactive: bool,
    target_resolution: Option<u32>,
    requested_render_tier: Option<RenderTier>,
    roi: Option<(f32, f32, f32, f32)>,
    compute_waveform: bool,
    active_waveform_channel: Option<&str>,
    prefer_native_display: bool,
) -> Result<Vec<u8>, String> {
    let fn_start = std::time::Instant::now();
    let context = get_or_init_gpu_context(&state, app_handle)?;
    hydrate_adjustments(&state, &mut adjustments_json);
    let adjustments_clone = adjustments_json;

    let loaded_image_guard = state.original_image.lock().unwrap();
    let loaded_image = loaded_image_guard
        .as_ref()
        .ok_or("No original image loaded")?
        .clone();
    drop(loaded_image_guard);

    let new_transform_hash = calculate_transform_hash(&adjustments_clone);
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let live_quality = settings.live_preview_quality.as_deref().unwrap_or("high");

    let default_preview_dim = settings.editor_preview_resolution.unwrap_or(1920);
    let preview_dim = target_resolution.unwrap_or(default_preview_dim);
    let source_longest_edge = loaded_image.image.width().max(loaded_image.image.height());
    let render_tier = resolve_preview_render_tier(
        requested_render_tier,
        is_interactive,
        roi.is_some(),
        preview_dim,
        source_longest_edge,
    )?;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    let native_display_enabled =
        prefer_native_display && settings.use_wgpu_renderer.unwrap_or(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let native_display_enabled = {
        let _ = prefer_native_display;
        false
    };

    let interactive_quality = match live_quality {
        "full" => 85_u8,
        "performance" => 65_u8,
        _ => 75_u8,
    };

    let mut cached_preview_lock = state.cached_preview.lock().unwrap();

    let base_valid = cached_preview_lock
        .as_ref()
        .is_some_and(|c| c.transform_hash == new_transform_hash && c.preview_dim == preview_dim);

    let (processing_image, effective_scale, unscaled_crop_offset) = if base_valid {
        let cached = cached_preview_lock.as_ref().unwrap();
        (
            Arc::clone(&cached.image),
            cached.scale,
            cached.unscaled_crop_offset,
        )
    } else {
        *state.gpu_image_cache.lock().unwrap() = None;

        generate_transformed_preview(&state, &loaded_image, &adjustments_clone, preview_dim)?
    };

    *cached_preview_lock = Some(CachedPreview {
        image: Arc::clone(&processing_image),
        transform_hash: new_transform_hash,
        scale: effective_scale,
        unscaled_crop_offset,
        preview_dim,
    });

    drop(cached_preview_lock);

    let jpeg_quality = if is_interactive {
        interactive_quality
    } else {
        94
    };

    let (preview_width, preview_height) = processing_image.dimensions();
    log::info!(
        "[process_preview_job] tier={} target={} resolved={}x{} roi={}",
        render_tier.as_str(),
        preview_dim,
        preview_width,
        preview_height,
        roi.is_some()
    );

    let has_native_display = context
        .display
        .lock()
        .map(|display| display.is_some())
        .unwrap_or(false);
    let output_to_native_display = should_output_to_native_display(
        native_display_enabled,
        has_native_display,
        preview_width,
        preview_height,
        context.limits.max_texture_dimension_2d,
    );

    let pixel_roi = normalized_roi_to_pixels(roi, preview_width, preview_height);

    let mask_definitions: Vec<MaskDefinition> = adjustments_clone
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let scaled_crop_offset = (
        unscaled_crop_offset.0 * effective_scale,
        unscaled_crop_offset.1 * effective_scale,
    );

    let mask_bitmaps: Vec<SharedMaskBitmap> = mask_definitions
        .iter()
        .filter_map(|def| {
            get_cached_or_generate_mask(
                &state,
                def,
                preview_width,
                preview_height,
                effective_scale,
                scaled_crop_offset,
                &adjustments_clone,
            )
        })
        .collect();

    let is_raw = loaded_image.is_raw;
    let tm_override = resolve_tonemapper_override_from_handle(app_handle, is_raw);
    let final_adjustments = get_all_adjustments_from_json(&adjustments_clone, is_raw, tm_override);
    let lut_path = adjustments_clone["lutPath"].as_str();
    let lut = lut_path.and_then(|p| lut_processing::get_or_load_lut(&state, p).ok());

    // A viewport patch is not representative of the whole image. Keep the
    // histogram/waveform from the most recent complete preview instead.
    let wants_analytics = pixel_roi.is_none();
    let channel_filter = if !is_interactive {
        active_waveform_channel.map(|s| s.to_string())
    } else {
        None
    };

    // Histogram/waveform readback copies the complete preview texture. It is
    // useful for a settled frame, but doing that work for every pointer move
    // competes with the image shown to the user. Keep analytics on the final
    // frame of a gesture; the existing analytics worker still coalesces those
    // settled updates to the newest image.
    let analytics_config = if wants_analytics && !is_interactive {
        state
            .analytics_worker_tx
            .lock()
            .unwrap()
            .clone()
            .map(|tx| crate::AnalyticsConfig {
                path: loaded_image.path.clone(),
                compute_waveform,
                active_waveform_channel: channel_filter,
                sender: tx,
            })
    } else {
        None
    };

    let final_processed_image_result =
        crate::image_processing::process_and_get_dynamic_image_with_analytics(
            &context,
            &state,
            &processing_image,
            new_transform_hash,
            RenderRequest {
                adjustments: final_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: pixel_roi,
            },
            "apply_adjustments",
            output_to_native_display,
            analytics_config,
        );

    if let Ok(final_processed_image) = final_processed_image_result {
        if output_to_native_display {
            // Rendering has already been submitted and presented in order.
            // A 500ms Wait here made every slider update look stuck even
            // though the native surface could display it immediately.
            let _ = context.device.poll(wgpu::PollType::Poll);
            let _ = app_handle.emit(
                "wgpu-frame-ready",
                serde_json::json!({ "path": loaded_image.path }),
            );
            return Ok(b"WGPU_RENDER".to_vec());
        }

        let final_rgba_image = match &*final_processed_image {
            DynamicImage::ImageRgba8(img) => img,
            _ => return Err("Expected Rgba8 image from GPU for encoding".to_string()),
        };

        let raw_bytes: &[u8] = final_rgba_image.as_raw();
        let rgba8_pixels: &[RGBA8] = raw_bytes.as_rgba();

        let img_ref = ImgRef::new(
            rgba8_pixels,
            final_rgba_image.width() as usize,
            final_rgba_image.height() as usize,
        );

        let step_start = std::time::Instant::now();

        let is_settled_viewport_patch = !is_interactive && pixel_roi.is_some();
        let encode_quality = if is_settled_viewport_patch {
            100
        } else {
            jpeg_quality
        };
        let chroma_subsampling = if is_settled_viewport_patch {
            Subsampling::S444
        } else {
            Subsampling::S420
        };

        let encode_result = srgb_preview_encoder(Preset::BaselineFastest)
            .quality(encode_quality)
            .subsampling(chroma_subsampling)
            .fast_color(true)
            .encode_imgref(img_ref);

        match encode_result {
            Ok(jpeg_bytes) => {
                if is_interactive || pixel_roi.is_some() {
                    let (roi_w, roi_h) = final_rgba_image.dimensions();
                    let (rx, ry) = if let Some(r) = pixel_roi {
                        (r.x, r.y)
                    } else {
                        (0, 0)
                    };

                    let response = encode_preview_patch(
                        &jpeg_bytes,
                        crate::gpu_processing::Roi {
                            x: rx,
                            y: ry,
                            width: roi_w,
                            height: roi_h,
                        },
                        preview_width,
                        preview_height,
                    );

                    log::info!(
                        "[process_preview_job] viewport ROI {}x{} encode in {:.2?}, total {:.2?}",
                        roi_w,
                        roi_h,
                        step_start.elapsed(),
                        fn_start.elapsed()
                    );
                    Ok(response)
                } else {
                    let (width, height) = final_rgba_image.dimensions();
                    log::info!(
                        "[process_preview_job] full {}x{} q={} encode in {:.2?}, total {:.2?}",
                        width,
                        height,
                        encode_quality,
                        step_start.elapsed(),
                        fn_start.elapsed()
                    );
                    Ok(jpeg_bytes)
                }
            }
            Err(e) => Err(format!("Failed to encode preview: {}", e)),
        }
    } else {
        log::error!(
            "[process_preview_job] processing failed after {:.2?}",
            fn_start.elapsed()
        );
        Err("Processing failed".to_string())
    }
}

pub(super) fn start_analytics_worker(app_handle: tauri::AppHandle) {
    let state = app_handle.state::<AppState>();
    let (tx, rx): (Sender<AnalyticsJob>, Receiver<AnalyticsJob>) = mpsc::channel();
    *state.analytics_worker_tx.lock().unwrap() = Some(tx);

    std::thread::spawn(move || {
        while let Ok(mut job) = rx.recv() {
            while let Ok(latest) = rx.try_recv() {
                job = latest;
            }

            let histogram_data = image_processing::calculate_histogram_from_image(&job.image).ok();

            let waveform_data = if job.compute_waveform {
                image_processing::calculate_waveform_from_image(
                    &job.image,
                    job.active_waveform_channel.as_deref(),
                )
                .ok()
            } else {
                None
            };

            if histogram_data.is_some() || waveform_data.is_some() {
                let _ = app_handle.emit(
                    "analytics-update",
                    serde_json::json!({
                        "path": job.path,
                        "histogram": histogram_data,
                        "waveform": waveform_data,
                    }),
                );
            }
        }
    });
}

pub(super) fn start_preview_worker(app_handle: tauri::AppHandle) {
    let state = app_handle.state::<AppState>();
    let (tx, rx): (Sender<PreviewJob>, Receiver<PreviewJob>) = mpsc::channel();

    *state.preview_worker_tx.lock().unwrap() = Some(tx);

    std::thread::spawn(move || {
        while let Ok(mut job) = rx.recv() {
            while let Ok(latest_job) = rx.try_recv() {
                job = latest_job;
            }

            let state = app_handle.state::<AppState>();
            let responder = job.responder;
            match process_preview_job(
                &app_handle,
                state,
                job.adjustments,
                job.is_interactive,
                job.target_resolution,
                job.render_tier,
                job.roi,
                job.compute_waveform,
                job.active_waveform_channel.as_deref(),
                job.prefer_native_display,
            ) {
                Ok(bytes) => {
                    let _ = responder.send(bytes);
                }
                Err(e) => {
                    log::error!("Preview worker error: {}", e);
                }
            }
        }
    });
}
