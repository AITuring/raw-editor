//! Editor preview commands.
//!
//! Queues interactive preview jobs, renders preset and per-path previews and
//! updates the native WGPU display transform.

use crate::adjustment_utils::apply_all_transformations;
use crate::app_settings::load_settings;
use crate::app_state::AppState;
use crate::app_state::PreviewJob;
use crate::app_state::SharedMaskBitmap;
use crate::cache_utils::calculate_full_job_hash;
use crate::color_management::srgb_preview_encoder;
use crate::file_management::parse_virtual_path;
use crate::file_management::read_file_mapped;
use crate::formats::is_raw_file;
use crate::image_loader::load_and_composite;
use crate::image_processing::GeometryWarpSampler;
use crate::image_processing::RenderRequest;
use crate::image_processing::get_all_adjustments_from_json;
use crate::image_processing::get_geometry_params_from_json;
use crate::image_processing::get_or_init_gpu_context;
use crate::image_processing::process_and_get_dynamic_image;
use crate::image_processing::resolve_tonemapper_override;
use crate::image_processing::resolve_tonemapper_override_from_handle;
use crate::mask_generation::MaskDefinition;
use crate::mask_generation::RangeMaskSource;
use crate::mask_generation::generate_mask_bitmap;
use crate::mask_generation::get_cached_or_generate_mask;
use crate::mask_generation::prepare_range_mask_source_image;
use crate::render_strategy::RenderTier;
use image::GenericImageView;
use mozjpeg_rs::Preset;
use serde_json::Value;
use std::borrow::Cow;
use std::fs;
use std::sync::Arc;
use tauri::Manager;
use tauri::ipc::Response;

use super::{generate_transformed_preview, lut_processing};

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WgpuTransformPayload {
    pub window_width: f32,
    pub window_height: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub clip_x: f32,
    pub clip_y: f32,
    pub clip_width: f32,
    pub clip_height: f32,
    pub bg_primary: [f32; 4],
    pub bg_secondary: [f32; 4],
    pub pixelated: bool,
}

#[tauri::command]
pub(super) async fn update_wgpu_transform(
    payload: WgpuTransformPayload,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let context = match state.gpu_context.lock().unwrap().as_ref() {
        Some(c) => c.clone(),
        None => return Ok(()),
    };

    tokio::task::spawn_blocking(move || {
        let mut display_lock = context.display.lock().unwrap();
        if let Some(display) = display_lock.as_mut() {
            display.latest_transform.rect = [payload.x, payload.y, payload.width, payload.height];
            display.latest_transform.clip = [
                payload.clip_x,
                payload.clip_y,
                payload.clip_width,
                payload.clip_height,
            ];
            display.latest_transform.window = [payload.window_width, payload.window_height];
            display.latest_transform.bg_primary = payload.bg_primary;
            display.latest_transform.bg_secondary = payload.bg_secondary;
            display.latest_transform.pixelated = if payload.pixelated { 1.0 } else { 0.0 };

            context.queue.write_buffer(
                &display.transform_buffer,
                0,
                bytemuck::bytes_of(&display.latest_transform),
            );
            display.render(&context.device, &context.queue);
        }
    })
    .await
    .map_err(|e| format!("Task panicked: {}", e))?;

    Ok(())
}

#[allow(clippy::too_many_arguments)] // Tauri exposes these fields as the stable IPC command payload.
#[tauri::command]
pub(super) async fn apply_adjustments(
    js_adjustments: serde_json::Value,
    is_interactive: bool,
    target_resolution: Option<u32>,
    render_tier: Option<RenderTier>,
    roi: Option<(f32, f32, f32, f32)>,
    compute_waveform: bool,
    active_waveform_channel: Option<String>,
    prefer_native_display: Option<bool>,
    state: tauri::State<'_, AppState>,
) -> Result<Response, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    {
        let tx_guard = state.preview_worker_tx.lock().unwrap();
        if let Some(worker_tx) = &*tx_guard {
            let job = PreviewJob {
                adjustments: js_adjustments,
                is_interactive,
                target_resolution,
                render_tier,
                roi,
                compute_waveform,
                active_waveform_channel,
                prefer_native_display: prefer_native_display.unwrap_or(true),
                responder: tx,
            };
            worker_tx
                .send(job)
                .map_err(|e| format!("Failed to send to preview worker: {}", e))?;
        } else {
            return Err("Preview worker not running".to_string());
        }
    }

    match rx.await {
        Ok(bytes) => Ok(Response::new(bytes)),
        Err(_) => Err("Superseded or worker failed".to_string()),
    }
}

#[tauri::command]
pub(super) fn generate_preset_preview(
    js_adjustments: serde_json::Value,
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Response, String> {
    let context = get_or_init_gpu_context(&state, &app_handle)?;

    let loaded_image = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded for preset preview")?;
    let is_raw = loaded_image.is_raw;
    let unique_hash = calculate_full_job_hash(&loaded_image.path, &js_adjustments);

    const PRESET_PREVIEW_DIM: u32 = 400;

    let (preview_image, scale_for_gpu, unscaled_crop_offset) =
        generate_transformed_preview(&state, &loaded_image, &js_adjustments, PRESET_PREVIEW_DIM)?;

    let (img_w, img_h) = preview_image.dimensions();

    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let scaled_crop_offset = (
        unscaled_crop_offset.0 * scale_for_gpu,
        unscaled_crop_offset.1 * scale_for_gpu,
    );

    let mask_bitmaps: Vec<SharedMaskBitmap> = mask_definitions
        .iter()
        .filter_map(|def| {
            get_cached_or_generate_mask(
                &state,
                def,
                img_w,
                img_h,
                scale_for_gpu,
                scaled_crop_offset,
                &js_adjustments,
            )
        })
        .collect();

    let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
    let all_adjustments = get_all_adjustments_from_json(&js_adjustments, is_raw, tm_override);
    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| lut_processing::get_or_load_lut(&state, p).ok());

    let processed_image = process_and_get_dynamic_image(
        &context,
        &state,
        &preview_image,
        unique_hash,
        RenderRequest {
            adjustments: all_adjustments,
            mask_bitmaps: &mask_bitmaps,
            lut,
            roi: None,
        },
        "generate_preset_preview",
    )?;

    let rgb = processed_image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let bytes = srgb_preview_encoder(Preset::BaselineFastest)
        .quality(80)
        .encode_rgb(rgb.as_raw(), width, height)
        .map_err(|e| e.to_string())?;

    Ok(Response::new(bytes))
}

#[tauri::command]
pub(super) async fn generate_preview_for_path(
    path: String,
    js_adjustments: Value,
    app_handle: tauri::AppHandle,
) -> Result<Response, String> {
    tokio::task::spawn_blocking(move || {
        let state = app_handle.state::<AppState>();
        let context = get_or_init_gpu_context(&state, &app_handle)?;
        let (source_path, _) = parse_virtual_path(&path);
        let source_path_str = source_path.to_string_lossy().to_string();
        let is_raw = is_raw_file(&source_path_str);
        let settings = load_settings(app_handle.clone()).unwrap_or_default();

        let base_image = match read_file_mapped(&source_path) {
            Ok(mmap) => load_and_composite(
                &mmap,
                &source_path_str,
                &js_adjustments,
                false,
                &settings,
                None,
            )
            .map_err(|e| e.to_string())?,
            Err(e) => {
                log::warn!(
                    "Failed to memory-map file '{}': {}. Falling back to standard read.",
                    source_path_str,
                    e
                );
                let bytes = fs::read(&source_path).map_err(|io_err| io_err.to_string())?;
                load_and_composite(
                    &bytes,
                    &source_path_str,
                    &js_adjustments,
                    false,
                    &settings,
                    None,
                )
                .map_err(|e| e.to_string())?
            }
        };

        let (transformed_image, unscaled_crop_offset) =
            apply_all_transformations(Cow::Borrowed(&base_image), &js_adjustments);
        let (img_w, img_h) = transformed_image.dimensions();
        let mask_definitions: Vec<MaskDefinition> = js_adjustments
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let range_source_image =
            prepare_range_mask_source_image(&base_image, is_raw, &mask_definitions);
        let range_source = range_source_image.as_deref().map(|image| {
            GeometryWarpSampler::new(image, get_geometry_params_from_json(&js_adjustments))
        });
        let mask_bitmaps: Vec<SharedMaskBitmap> = mask_definitions
            .iter()
            .filter_map(|def| {
                generate_mask_bitmap(
                    def,
                    img_w,
                    img_h,
                    1.0,
                    unscaled_crop_offset,
                    range_source
                        .as_ref()
                        .map(|source| source as &dyn RangeMaskSource),
                )
                .map(Arc::new)
            })
            .collect();

        let tm_override = resolve_tonemapper_override(&settings, is_raw);
        let all_adjustments = get_all_adjustments_from_json(&js_adjustments, is_raw, tm_override);
        let lut_path = js_adjustments["lutPath"].as_str();
        let lut = lut_path.and_then(|p| lut_processing::get_or_load_lut(&state, p).ok());
        let unique_hash = calculate_full_job_hash(&source_path_str, &js_adjustments);

        let final_image = process_and_get_dynamic_image(
            &context,
            &state,
            transformed_image.as_ref(),
            unique_hash,
            RenderRequest {
                adjustments: all_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: None,
            },
            "generate_preview_for_path",
        )?;

        let (width, height) = final_image.dimensions();
        let rgb_pixels = final_image.to_rgb8().into_vec();

        let bytes = srgb_preview_encoder(Preset::BaselineFastest)
            .quality(92)
            .encode_rgb(&rgb_pixels, width, height)
            .map_err(|e| format!("Failed to encode with mozjpeg-rs: {}", e))?;

        Ok(Response::new(bytes))
    })
    .await
    .map_err(|e| format!("Task execution failed: {}", e))?
}
