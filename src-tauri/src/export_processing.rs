use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(test)]
use image::DynamicImage;
use image::GenericImageView;
#[cfg(test)]
use image::ImageFormat;
#[cfg(test)]
#[cfg(not(target_os = "android"))]
use image::Rgba;
#[cfg(test)]
use image::RgbaImage;
#[cfg(test)]
use image::{ImageBuffer, imageops};
use serde_json::Value;
use tauri::Emitter;
use tauri::Manager;
#[cfg(test)]
#[cfg(not(target_os = "android"))]
use zenresize::{Filter as StreamingResizeFilter, PixelDescriptor, ResizeConfig};

use crate::AppState;
use crate::app_state::SharedMaskBitmap;
#[cfg(test)]
use crate::exif_processing;
use crate::file_management::{parse_virtual_path, read_file_mapped};
use crate::formats::is_raw_file;
use crate::image_loader::load_base_image_from_bytes;
#[cfg(test)]
use crate::image_processing::get_geometry_params_from_json;
use crate::image_processing::{
    Crop, RenderRequest, downscale_f32_image, get_all_adjustments_from_json,
    get_or_init_gpu_context, process_and_get_dynamic_image,
    resolve_tonemapper_override_from_handle,
};
use crate::lut_processing::get_or_load_lut;
use crate::mask_generation::MaskDefinition;

use crate::cache_utils::{calculate_full_job_hash, calculate_transform_hash};
#[cfg(test)]
use crate::color_management::srgb_v4_profile;
use crate::{
    apply_all_transformations, generate_transformed_preview, get_cached_or_generate_mask,
    hydrate_adjustments, load_settings,
};

mod settings;

pub use settings::ExportSettings;
pub(crate) use settings::{
    ExportAdjustmentsMode, calculate_resize_target, default_export_bit_depth_for_format,
    effective_export_bit_depth,
};
#[cfg(test)]
pub use settings::{ResizeMode, ResizeOptions};

mod watermark;

#[cfg(test)]
pub use watermark::WatermarkAnchor;
pub use watermark::WatermarkSettings;
#[cfg(test)]
use watermark::load_watermark_image;
pub(crate) use watermark::{DEFAULT_WATERMARK_PATH, apply_export_resize_and_watermark};
use watermark::{PreparedWatermark, import_watermark_image_impl, prepare_watermark};

mod tasks;

use tasks::{
    ExportCancellationRequest, ExportTaskEventScope, ExportTaskGuard, ensure_export_not_cancelled,
    finish_export_task, register_export_task, request_export_cancellation,
    summarize_batch_export_results,
};

mod export_render;

#[cfg(test)]
use export_render::can_stream_geometry_output;
#[cfg(not(target_os = "android"))]
use export_render::process_and_save_streaming_export;
use export_render::process_image_for_export;
#[cfg(test)]
#[cfg(not(target_os = "android"))]
use export_render::{StreamingExportMetadata, encode_streaming_rgba_rows};

mod file_output;

#[cfg(not(target_os = "android"))]
use file_output::{create_temporary_export, preserve_existing_export_permissions};
use file_output::{save_image_with_metadata, set_timestamps_from_exif, supports_streaming_export};

mod webp_output;

#[cfg(not(target_os = "android"))]
use webp_output::save_webp_with_bounded_output;
#[cfg(test)]
#[cfg(not(target_os = "android"))]
use webp_output::{
    WEBP_OUTPUT_COPY_BUFFER_BYTES, WebpFileWriterContext, WebpPictureGuard,
    check_webp_encoding_progress, rewrite_webp_icc_bounded,
};

mod streaming_encoders;

#[cfg(not(target_os = "android"))]
pub(crate) use streaming_encoders::encode_streaming_jpeg;
#[cfg(not(target_os = "android"))]
use streaming_encoders::{
    encode_streaming_png, transform_streaming_rgba_rows, validate_streamed_rgba_row,
};

mod tiff_streaming;

#[cfg(test)]
#[cfg(not(target_os = "android"))]
pub(crate) use tiff_streaming::encode_streaming_bigtiff_rgb8;
#[cfg(not(target_os = "android"))]
use tiff_streaming::{
    encode_streaming_tiff, has_writable_tiff_metadata_group, write_tiff_metadata_group,
};

mod tiff_writers;

#[cfg(not(target_os = "android"))]
pub(crate) use tiff_writers::{
    encode_rgb8_tiff_with_metadata, encode_rgb16_tiff_with_metadata,
    encode_rgba8_tiff_with_metadata, encode_rgba16_tiff_with_metadata,
};
#[cfg(test)]
#[cfg(not(target_os = "android"))]
pub(crate) use tiff_writers::{encode_rgb16_bigtiff_with_metadata, should_use_bigtiff};

mod encode_bytes;

#[cfg(test)]
pub(crate) use encode_bytes::encode_image_to_bytes;
#[cfg(target_os = "android")]
pub use encode_bytes::mime_type_for_extension;
use encode_bytes::{encode_grayscale_to_png, encode_image_to_bytes_with_profile};

mod masks_and_lut;

use masks_and_lut::{export_adjustments_as_lut, export_masks_for_image};

mod batch_export;

use batch_export::export_images_impl;
pub use batch_export::run_headless_export;

mod batch_geometry;

pub use batch_geometry::{
    BatchGeometryAnalysisResult, BatchGeometryCorrectionResult, BatchGeometryOrientationOverride,
    BatchGeometryPreviewItem,
};
use batch_geometry::{
    build_batch_geometry_adjustments, create_batch_geometry_output_folder,
    find_batch_distortion_params, has_usable_distortion_params,
    normalize_batch_geometry_output_format,
};

mod content_orientation;

#[cfg(test)]
use content_orientation::decide_content_orientation;
use content_orientation::{
    CONTENT_ORIENTATION_ANALYSIS_EDGE, ContentOrientationDecision,
    analyze_batch_content_orientation, load_batch_content_orientation_preview,
};

mod batch_geometry_preview;

use batch_geometry_preview::{
    batch_geometry_preview_item_from_prepared, prepare_batch_geometry_source,
};

const DEFAULT_WATERMARK_BYTES: &[u8] = include_bytes!("../../src/assets/default-watermark.png");

#[tauri::command]
pub async fn import_watermark_image(
    path: String,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let source_path = fs::canonicalize(PathBuf::from(path))
        .map_err(|error| format!("Failed to resolve the selected watermark image: {error}"))?;
    if !app_handle.asset_protocol_scope().is_allowed(&source_path) {
        return Err("Select the watermark image with the application file picker.".to_string());
    }
    let watermark_directory = app_handle
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("watermarks");

    tauri::async_runtime::spawn_blocking(move || {
        import_watermark_image_impl(&source_path, &watermark_directory)
            .map(|stored_path| stored_path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| format!("Watermark import task failed: {error}"))?
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn export_images(
    paths: Vec<String>,
    output_folder_or_file: String,
    is_explicit_file_path: bool,
    base_origin_folders: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    current_edit_path: Option<String>,
    current_edit_adjustments: Option<Value>,
    wait_for_completion: Option<bool>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let (completion_tx, completion_rx) = if wait_for_completion.unwrap_or(false) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    export_images_impl(
        paths,
        output_folder_or_file,
        is_explicit_file_path,
        base_origin_folders,
        export_settings,
        output_format,
        ExportAdjustmentsMode::UseSidecars {
            active_path: current_edit_path,
            active_adjustments: current_edit_adjustments,
        },
        state,
        app_handle,
        completion_tx,
        ExportTaskEventScope::Standard,
    )
    .await?;

    match completion_rx {
        Some(rx) => match rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error_count)) => Err(format!("Export completed with {error_count} error(s).")),
            Err(_) => Err("Export task ended before reporting completion.".to_string()),
        },
        None => Ok(()),
    }
}

/// Analyzes selected sources one at a time and streams each visual comparison
/// to the modal. Nothing is exported here: the user reviews these previews and
/// can override a direction before a separate export is started.
#[tauri::command]
pub async fn analyze_batch_geometry(
    paths: Vec<String>,
    apply_distortion: bool,
    show_guides: Option<bool>,
    use_content_orientation: Option<bool>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<BatchGeometryAnalysisResult, String> {
    if paths.is_empty() {
        return Err("Select at least one image to analyze.".to_string());
    }

    let show_guides = show_guides.unwrap_or(false);
    let use_content_orientation = use_content_orientation.unwrap_or(true);
    let clip_models = if use_content_orientation {
        let _ = app_handle.emit(
            "batch-geometry-orientation-progress",
            serde_json::json!({ "current": 0, "total": paths.len(), "path": "" }),
        );
        Some(
            crate::ai_processing::get_or_init_clip_models(
                &app_handle,
                &state.ai_state,
                &state.ai_init_lock,
            )
            .await
            .map_err(|error| format!("Could not start the content orientation model: {error}"))?,
        )
    } else {
        None
    };
    let preview_settings = load_settings(app_handle.clone())?;
    let lens_database = state
        .lens_db
        .lock()
        .map_err(|error| format!("Lens database lock poisoned: {error}"))?
        .clone();
    let mut items = Vec::with_capacity(paths.len());

    for (index, image_path) in paths.iter().enumerate() {
        let prepared = tokio::task::spawn_blocking({
            let image_path = image_path.clone();
            let preview_settings = preview_settings.clone();
            let clip_models = clip_models.clone();
            let lens_database = lens_database.clone();
            move || {
                prepare_batch_geometry_source(
                    image_path,
                    apply_distortion,
                    Some(preview_settings),
                    clip_models,
                    lens_database,
                    None,
                    true,
                    show_guides,
                )
            }
        })
        .await
        .map_err(|error| format!("Batch preview task failed: {error}"))??;
        let item = batch_geometry_preview_item_from_prepared(prepared)?;

        if use_content_orientation {
            let _ = app_handle.emit(
                "batch-geometry-orientation-progress",
                serde_json::json!({
                    "current": index + 1,
                    "total": paths.len(),
                    "path": item.source_path.clone(),
                }),
            );
        }
        let _ = app_handle.emit("batch-geometry-preview", item.clone());
        items.push(item);
    }

    Ok(BatchGeometryAnalysisResult { items })
}

/// Re-renders one reviewed source after a manual right-angle override without
/// invoking the content model again. This keeps the review controls immediate
/// while preserving the exact distortion and geometry path used for export.
#[tauri::command]
pub async fn preview_batch_geometry_correction(
    path: String,
    orientation_steps: u8,
    apply_distortion: bool,
    show_guides: Option<bool>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<BatchGeometryPreviewItem, String> {
    let preview_settings = load_settings(app_handle.clone())?;
    let lens_database = state
        .lens_db
        .lock()
        .map_err(|error| format!("Lens database lock poisoned: {error}"))?
        .clone();
    let prepared = tokio::task::spawn_blocking(move || {
        prepare_batch_geometry_source(
            path,
            apply_distortion,
            Some(preview_settings),
            None,
            lens_database,
            Some(orientation_steps),
            true,
            show_guides.unwrap_or(false),
        )
    })
    .await
    .map_err(|error| format!("Batch preview render task failed: {error}"))??;

    batch_geometry_preview_item_from_prepared(prepared)
}

/// Exports a clean batch of originals using content-aware orientation analysis and,
/// when available, the matching Lensfun distortion calibration. It deliberately
/// bypasses existing editor sidecars so the workflow never changes source edits or files.
#[tauri::command]
pub async fn batch_geometry_correction(
    paths: Vec<String>,
    output_folder: String,
    output_format: String,
    jpeg_quality: u8,
    apply_distortion: bool,
    use_content_orientation: Option<bool>,
    orientation_overrides: Option<Vec<BatchGeometryOrientationOverride>>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<BatchGeometryCorrectionResult, String> {
    if paths.is_empty() {
        return Err("Select at least one image to correct.".to_string());
    }

    let output_format = normalize_batch_geometry_output_format(&output_format)?;
    let use_content_orientation = use_content_orientation.unwrap_or(true);
    let orientation_overrides = orientation_overrides.map(|overrides| {
        let mut normalized = HashMap::with_capacity(overrides.len());
        for override_item in overrides {
            let (source_path, _) = parse_virtual_path(&override_item.path);
            normalized.insert(
                source_path.to_string_lossy().into_owned(),
                override_item.orientation_steps.min(3),
            );
        }
        normalized
    });
    // A reviewed override is authoritative. Do not make the user wait for a
    // second model pass after they have already inspected every preview.
    let should_analyze_content = use_content_orientation && orientation_overrides.is_none();
    let clip_models = if should_analyze_content {
        let _ = app_handle.emit(
            "batch-geometry-orientation-progress",
            serde_json::json!({ "current": 0, "total": paths.len(), "path": "" }),
        );
        Some(
            crate::ai_processing::get_or_init_clip_models(
                &app_handle,
                &state.ai_state,
                &state.ai_init_lock,
            )
            .await
            .map_err(|error| format!("Could not start the content orientation model: {error}"))?,
        )
    } else {
        None
    };
    let preview_settings = should_analyze_content
        .then(|| load_settings(app_handle.clone()))
        .transpose()?;
    let lens_database = state
        .lens_db
        .lock()
        .map_err(|error| format!("Lens database lock poisoned: {error}"))?
        .clone();

    let mut per_path_adjustments = HashMap::with_capacity(paths.len());
    let mut content_orientation_corrected_count = 0usize;
    let mut distortion_corrected_count = 0usize;
    let mut missing_distortion_profile_count = 0usize;

    for (index, image_path) in paths.iter().enumerate() {
        let (source_path, _) = parse_virtual_path(image_path);
        let source_path_key = source_path.to_string_lossy().into_owned();
        let orientation_override = match orientation_overrides.as_ref() {
            Some(overrides) => Some(*overrides.get(&source_path_key).ok_or_else(|| {
                format!(
                    "Missing a reviewed orientation for '{}'.",
                    source_path.display()
                )
            })?),
            None => None,
        };
        let prepared = tokio::task::spawn_blocking({
            let image_path = image_path.clone();
            let preview_settings = preview_settings.clone();
            let clip_models = clip_models.clone();
            let lens_database = lens_database.clone();
            move || {
                prepare_batch_geometry_source(
                    image_path,
                    apply_distortion,
                    preview_settings,
                    clip_models,
                    lens_database,
                    orientation_override,
                    false,
                    false,
                )
            }
        })
        .await
        .map_err(|error| format!("Batch source preparation task failed: {error}"))??;

        if should_analyze_content {
            let _ = app_handle.emit(
                "batch-geometry-orientation-progress",
                serde_json::json!({
                    "current": index + 1,
                    "total": paths.len(),
                    "path": prepared.source_path.clone(),
                }),
            );
        }

        if prepared.orientation_steps != 0 {
            content_orientation_corrected_count += 1;
        }

        let has_distortion = prepared
            .distortion_params
            .as_ref()
            .is_some_and(has_usable_distortion_params);

        if apply_distortion {
            if has_distortion {
                distortion_corrected_count += 1;
            } else {
                missing_distortion_profile_count += 1;
            }
        }

        per_path_adjustments.insert(
            prepared.source_path,
            build_batch_geometry_adjustments(
                prepared.orientation_steps,
                prepared.distortion_params,
                apply_distortion,
            ),
        );
    }
    let batch_output_folder = create_batch_geometry_output_folder(Path::new(&output_folder))?;
    let result = BatchGeometryCorrectionResult {
        output_folder: batch_output_folder.to_string_lossy().into_owned(),
        processed_count: paths.len(),
        content_orientation_corrected_count,
        distortion_corrected_count,
        missing_distortion_profile_count,
    };

    let export_settings = ExportSettings {
        bit_depth: default_export_bit_depth_for_format(&output_format),
        jpeg_quality: jpeg_quality.clamp(1, 100),
        resize: None,
        keep_metadata: true,
        metadata_overrides: None,
        preserve_timestamps: true,
        strip_gps: false,
        embed_color_profile: true,
        filename_template: Some("{original_filename}_corrected_{sequence}".to_string()),
        watermark: None,
        export_masks: false,
        preserve_folders: false,
    };
    let (completion_tx, completion_rx) = tokio::sync::oneshot::channel();

    export_images_impl(
        paths,
        result.output_folder.clone(),
        false,
        Vec::new(),
        export_settings,
        output_format.to_string(),
        ExportAdjustmentsMode::PerPathOverrides(Arc::new(per_path_adjustments)),
        state,
        app_handle,
        Some(completion_tx),
        ExportTaskEventScope::BatchGeometry,
    )
    .await?;

    match completion_rx.await {
        Ok(Ok(())) => Ok(result),
        Ok(Err(error_count)) => Err(format!(
            "Batch correction completed with {error_count} export error(s)."
        )),
        Err(_) => Err("Batch correction stopped before reporting completion.".to_string()),
    }
}

#[tauri::command]
pub fn cancel_export(
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    match request_export_cancellation(&state.export_task_token, || {
        let _ = app_handle.emit("export-cancelling", ());
    }) {
        ExportCancellationRequest::Requested => {
            log::info!("Export cancellation requested; workers will stop at the next checkpoint");
        }
        ExportCancellationRequest::AlreadyRequested => {
            log::info!("Export cancellation was already requested");
        }
        ExportCancellationRequest::NoActiveTask => {
            return Err("No export task is currently running.".to_string());
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn estimate_export_sizes(
    paths: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    current_edit_path: Option<String>,
    current_edit_adjustments: Option<Value>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<usize, String> {
    if output_format.to_lowercase() == "cube" {
        return Ok(1_050_000 * paths.len());
    }

    if paths.is_empty() {
        return Ok(0);
    }

    let first_path = &paths[0];
    let (source_path, sidecar_path) = parse_virtual_path(first_path);
    let source_path_str = source_path.to_string_lossy().to_string();

    let context = get_or_init_gpu_context(&state, &app_handle)?;
    let is_current_edit = Some(&source_path_str) == current_edit_path.as_ref();
    let is_raw = is_raw_file(&source_path_str);
    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let single_image_extrapolated_size: usize = if is_current_edit
        && current_edit_adjustments.is_some()
    {
        let loaded_image = state
            .original_image
            .lock()
            .unwrap()
            .clone()
            .ok_or("No original image loaded")?;
        let mut adjustments_clone = current_edit_adjustments.clone().unwrap();
        hydrate_adjustments(&state, &mut adjustments_clone);

        let new_transform_hash = calculate_transform_hash(&adjustments_clone);
        let cached_preview_lock = state.cached_preview.lock().unwrap();
        let preview_dim = settings.editor_preview_resolution.unwrap_or(1920);

        let (preview_image, scale, unscaled_crop_offset) = if let Some(cached) =
            &*cached_preview_lock
        {
            if cached.transform_hash == new_transform_hash && cached.preview_dim == preview_dim {
                let img = Arc::clone(&cached.image);
                let s = cached.scale;
                let offset = cached.unscaled_crop_offset;
                drop(cached_preview_lock);
                (img, s, offset)
            } else {
                drop(cached_preview_lock);
                generate_transformed_preview(
                    &state,
                    &loaded_image,
                    &adjustments_clone,
                    preview_dim,
                )?
            }
        } else {
            drop(cached_preview_lock);
            generate_transformed_preview(&state, &loaded_image, &adjustments_clone, preview_dim)?
        };

        let (img_w, img_h) = preview_image.dimensions();
        let mask_definitions: Vec<MaskDefinition> = adjustments_clone
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let scaled_crop_offset = (
            unscaled_crop_offset.0 * scale,
            unscaled_crop_offset.1 * scale,
        );

        let mask_bitmaps: Vec<SharedMaskBitmap> = mask_definitions
            .iter()
            .filter_map(|def| {
                get_cached_or_generate_mask(
                    &state,
                    def,
                    img_w,
                    img_h,
                    scale,
                    scaled_crop_offset,
                    &adjustments_clone,
                )
            })
            .collect();

        let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
        let mut all_adjustments =
            get_all_adjustments_from_json(&adjustments_clone, is_raw, tm_override);
        all_adjustments.global.show_clipping = 0;

        let lut = adjustments_clone["lutPath"]
            .as_str()
            .and_then(|p| get_or_load_lut(&state, p).ok());
        let unique_hash =
            calculate_full_job_hash(&loaded_image.path, &adjustments_clone).wrapping_add(1);

        let processed_preview = process_and_get_dynamic_image(
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
            "estimate_export_size",
        )?;

        let preview_bytes = encode_image_to_bytes_with_profile(
            &processed_preview,
            &output_format,
            export_settings.jpeg_quality,
            export_settings.embed_color_profile,
            export_settings.bit_depth,
        )?;
        let preview_byte_size = preview_bytes.len();

        let (transformed_full_res, _) =
            apply_all_transformations(&loaded_image.image, &adjustments_clone);
        let (full_w, full_h) = transformed_full_res.dimensions();

        let (final_full_w, final_full_h) = if let Some(resize_opts) = &export_settings.resize {
            calculate_resize_target(full_w, full_h, resize_opts)
        } else {
            (full_w, full_h)
        };

        let (processed_preview_w, processed_preview_h) = processed_preview.dimensions();
        let pixel_ratio = if processed_preview_w > 0 && processed_preview_h > 0 {
            (final_full_w as f64 * final_full_h as f64)
                / (processed_preview_w as f64 * processed_preview_h as f64)
        } else {
            1.0
        };

        (preview_byte_size as f64 * pixel_ratio) as usize
    } else {
        let metadata = crate::exif_processing::load_sidecar(&sidecar_path);
        let mut js_adjustments = metadata.adjustments;

        const ESTIMATE_DIM: u32 = 1280;

        let file_slice: Vec<u8>;
        let mmap_guard;
        let file_data: &[u8] = match read_file_mapped(Path::new(&source_path_str)) {
            Ok(mmap) => {
                mmap_guard = Some(mmap);
                mmap_guard.as_ref().unwrap()
            }
            Err(_) => {
                file_slice = fs::read(&source_path_str).map_err(|io_err| io_err.to_string())?;
                &file_slice
            }
        };

        let original_image =
            load_base_image_from_bytes(file_data, &source_path_str, true, &settings, None)
                .map_err(|e| e.to_string())?;

        let raw_scale_factor = if is_raw {
            crate::raw_processing::get_fast_demosaic_scale_factor(
                file_data,
                original_image.width(),
                original_image.height(),
            )
        } else {
            1.0
        };

        if let Some(crop_val) = js_adjustments.get_mut("crop")
            && let Ok(c) = serde_json::from_value::<Crop>(crop_val.clone())
        {
            *crop_val = serde_json::to_value(Crop {
                x: c.x * raw_scale_factor as f64,
                y: c.y * raw_scale_factor as f64,
                width: c.width * raw_scale_factor as f64,
                height: c.height * raw_scale_factor as f64,
            })
            .unwrap_or(serde_json::Value::Null);
        }

        let (transformed_shrunk_res, unscaled_crop_offset) =
            apply_all_transformations(Cow::Borrowed(&original_image), &js_adjustments);
        let (shrunk_w, shrunk_h) = transformed_shrunk_res.dimensions();

        let preview_base = if shrunk_w > ESTIMATE_DIM || shrunk_h > ESTIMATE_DIM {
            downscale_f32_image(transformed_shrunk_res.as_ref(), ESTIMATE_DIM, ESTIMATE_DIM)
        } else {
            transformed_shrunk_res.into_owned()
        };

        let (preview_w, preview_h) = preview_base.dimensions();
        let gpu_scale = if shrunk_w > 0 {
            preview_w as f32 / shrunk_w as f32
        } else {
            1.0
        };
        let total_scale = gpu_scale * raw_scale_factor;

        let mask_definitions: Vec<MaskDefinition> = js_adjustments
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();
        let scaled_crop_offset = (
            unscaled_crop_offset.0 * gpu_scale,
            unscaled_crop_offset.1 * gpu_scale,
        );

        let mask_bitmaps: Vec<SharedMaskBitmap> = mask_definitions
            .iter()
            .filter_map(|def| {
                get_cached_or_generate_mask(
                    &state,
                    def,
                    preview_w,
                    preview_h,
                    total_scale,
                    scaled_crop_offset,
                    &js_adjustments,
                )
            })
            .collect();

        let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
        let mut all_adjustments =
            get_all_adjustments_from_json(&js_adjustments, is_raw, tm_override);
        all_adjustments.global.show_clipping = 0;

        let lut = js_adjustments["lutPath"]
            .as_str()
            .and_then(|p| get_or_load_lut(&state, p).ok());
        let unique_hash =
            calculate_full_job_hash(&source_path_str, &js_adjustments).wrapping_add(1);

        let processed_preview = process_and_get_dynamic_image(
            &context,
            &state,
            &preview_base,
            unique_hash,
            RenderRequest {
                adjustments: all_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: None,
            },
            "estimate_batch_export_size",
        )?;

        let preview_bytes = encode_image_to_bytes_with_profile(
            &processed_preview,
            &output_format,
            export_settings.jpeg_quality,
            export_settings.embed_color_profile,
            export_settings.bit_depth,
        )?;
        let single_image_estimated_size = preview_bytes.len();

        let full_w = (shrunk_w as f32 / raw_scale_factor).round() as u32;
        let full_h = (shrunk_h as f32 / raw_scale_factor).round() as u32;

        let (final_full_w, final_full_h) = if let Some(resize_opts) = &export_settings.resize {
            calculate_resize_target(full_w, full_h, resize_opts)
        } else {
            (full_w, full_h)
        };

        let (processed_preview_w, processed_preview_h) = processed_preview.dimensions();
        let pixel_ratio = if processed_preview_w > 0 && processed_preview_h > 0 {
            (final_full_w as f64 * final_full_h as f64)
                / (processed_preview_w as f64 * processed_preview_h as f64)
        } else {
            1.0
        };

        (single_image_estimated_size as f64 * pixel_ratio) as usize
    };

    Ok(single_image_extrapolated_size * paths.len())
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{Cursor, Read, Seek, SeekFrom};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use crate::color_management::validate_icc_profile;
    use image::codecs::{jpeg::JpegDecoder, png::PngDecoder, tiff::TiffDecoder, webp::WebPDecoder};
    use image::{ImageDecoder, Rgb, RgbImage};
    use jxl_oxide::integration::JxlDecoder;
    use little_exif::exif_tag::ExifTag;
    use little_exif::filetype::FileExtension;
    use little_exif::metadata::Metadata;
    use little_exif::rational::uR64;
    use tempfile::tempdir;

    use super::*;

    fn fixture_image(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
            Rgb([
                ((x * 17 + y * 3) % 256) as u8,
                ((x * 5 + y * 13) % 256) as u8,
                ((x * 11 + y * 7) % 256) as u8,
            ])
        }))
    }

    #[test]
    fn bigtiff_budget_switches_before_standard_offset_limit() {
        assert!(!should_use_bigtiff(8_000, 6_000, 3, 2));
        assert!(should_use_bigtiff(1_000_000, 1_000_000, 3, 2));
        assert!(should_use_bigtiff(700_000, 700_000, 4, 2));
    }

    #[test]
    fn bigtiff_encoder_writes_a_decodable_16_bit_rgb_image() {
        let directory = tempdir().expect("temporary BigTIFF directory");
        let path = directory.path().join("sample.bigtiff");
        let mut output = File::create(&path).expect("create BigTIFF");
        encode_rgb16_bigtiff_with_metadata(
            &mut output,
            2,
            2,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            true,
            None,
        )
        .expect("encode BigTIFF");
        drop(output);

        let bytes = std::fs::read(&path).expect("read BigTIFF");
        assert!(
            bytes.get(2..4) == Some(&[43, 0][..]) || bytes.get(2..4) == Some(&[0, 43][..]),
            "expected TIFF version 43 (BigTIFF), header={:?}",
            bytes.get(..16)
        );
        let decoder = TiffDecoder::new(Cursor::new(bytes)).expect("decode BigTIFF");
        assert_eq!(decoder.dimensions(), (2, 2));
    }

    #[test]
    fn streaming_bigtiff_encoder_writes_rgba_rows_as_rgb8() {
        let directory = tempdir().expect("temporary BigTIFF directory");
        let path = directory.path().join("streamed.bigtiff");
        let mut output = File::create(&path).expect("create streamed BigTIFF");
        encode_streaming_bigtiff_rgb8(&mut output, 2, 2, |sink| {
            sink(&[10, 20, 30, 255, 40, 50, 60, 255])?;
            sink(&[70, 80, 90, 255, 100, 110, 120, 255])?;
            Ok(())
        })
        .expect("stream BigTIFF");
        drop(output);

        let bytes = std::fs::read(&path).expect("read streamed BigTIFF");
        assert_eq!(bytes.get(2..4), Some(&[43, 0][..]));
        let decoder = TiffDecoder::new(Cursor::new(bytes)).expect("decode streamed BigTIFF");
        assert_eq!(decoder.dimensions(), (2, 2));
    }

    #[test]
    fn export_adjustment_mode_normalizes_virtual_copy_paths() {
        let mode = ExportAdjustmentsMode::UseSidecars {
            active_path: Some("/photos/source.jpg?vc=3".to_string()),
            active_adjustments: Some(serde_json::json!({ "exposure": 1.0 })),
        }
        .normalize_active_path();

        match mode {
            ExportAdjustmentsMode::UseSidecars {
                active_path,
                active_adjustments,
            } => {
                assert_eq!(active_path.as_deref(), Some("/photos/source.jpg"));
                assert_eq!(
                    active_adjustments,
                    Some(serde_json::json!({ "exposure": 1.0 }))
                );
            }
            ExportAdjustmentsMode::GlobalOverride(_)
            | ExportAdjustmentsMode::PerPathOverrides(_) => panic!("unexpected override"),
        }
    }

    #[test]
    fn batch_geometry_adjustments_only_enable_a_usable_distortion_profile() {
        let usable_params = serde_json::json!({ "k1": 0.012, "k2": 0.0, "k3": 0.0, "model": 0 });
        let adjustments = build_batch_geometry_adjustments(2, Some(usable_params.clone()), true);

        assert_eq!(adjustments["lensDistortionEnabled"], true);
        assert_eq!(adjustments["lensDistortionParams"], usable_params);
        assert_eq!(adjustments["lensTcaEnabled"], false);
        assert_eq!(adjustments["lensVignetteEnabled"], false);
        assert_eq!(adjustments["orientationSteps"], 2);

        let unavailable =
            build_batch_geometry_adjustments(0, Some(serde_json::json!({ "k1": 0.0 })), true);
        assert_eq!(unavailable["lensDistortionEnabled"], false);

        let disabled =
            build_batch_geometry_adjustments(0, Some(serde_json::json!({ "k1": 0.012 })), false);
        assert_eq!(disabled["lensDistortionEnabled"], false);
    }

    #[test]
    fn batch_geometry_output_folders_are_created_without_overwriting_previous_runs() {
        let temporary_directory = tempfile::tempdir().expect("create temporary output root");

        let first = create_batch_geometry_output_folder(temporary_directory.path())
            .expect("create first batch output folder");
        let second = create_batch_geometry_output_folder(temporary_directory.path())
            .expect("create second batch output folder");

        assert_eq!(
            first.file_name().and_then(|name| name.to_str()),
            Some("RAW Editor Corrected")
        );
        assert_eq!(
            second.file_name().and_then(|name| name.to_str()),
            Some("RAW Editor Corrected 2")
        );
        assert!(first.is_dir());
        assert!(second.is_dir());
    }

    #[test]
    fn batch_geometry_content_orientation_requires_a_clear_win_over_the_current_direction() {
        let decision = decide_content_orientation([0.44, 0.48, 0.78, 0.53]);

        assert_eq!(
            decision,
            ContentOrientationDecision {
                orientation_steps: 2,
                corrected: true,
                confidence: 0.78,
            }
        );

        let ambiguous = decide_content_orientation([0.72, 0.74, 0.77, 0.73]);
        assert_eq!(ambiguous, ContentOrientationDecision::preserve_current());

        let weak = decide_content_orientation([0.35, 0.38, 0.57, 0.41]);
        assert_eq!(weak, ContentOrientationDecision::preserve_current());
    }

    #[test]
    fn geometry_output_streaming_supports_orthogonal_post_transforms_only() {
        let adjustments = serde_json::json!({
            "transformVertical": 18.0,
            "crop": { "unit": "px", "x": 4.0, "y": 3.0, "width": 80.0, "height": 60.0 }
        });
        let params = get_geometry_params_from_json(&adjustments);
        assert!(can_stream_geometry_output(&adjustments, &params));

        for compatible in [
            serde_json::json!({ "transformVertical": 18.0, "orientationSteps": 1 }),
            serde_json::json!({ "transformVertical": 18.0, "orientationSteps": 2, "flipHorizontal": true }),
            serde_json::json!({ "transformVertical": 18.0, "orientationSteps": 3, "flipVertical": true }),
            serde_json::json!({ "transformVertical": 18.0, "flipHorizontal": true, "flipVertical": true }),
        ] {
            let params = get_geometry_params_from_json(&compatible);
            assert!(can_stream_geometry_output(&compatible, &params));
        }

        for incompatible in [
            serde_json::json!({ "transformVertical": 18.0, "rotation": 0.5 }),
            serde_json::json!({ "transformVertical": 18.0, "lensBlurEnabled": true }),
        ] {
            let params = get_geometry_params_from_json(&incompatible);
            assert!(!can_stream_geometry_output(&incompatible, &params));
        }

        let color_mask: MaskDefinition = serde_json::from_value(serde_json::json!({
            "id": "color-mask",
            "name": "Color",
            "visible": true,
            "invert": false,
            "opacity": 100.0,
            "adjustments": null,
            "subMasks": [{
                "id": "color",
                "type": "color",
                "visible": true,
                "invert": false,
                "opacity": 100.0,
                "mode": "additive",
                "parameters": {}
            }]
        }))
        .expect("color mask fixture");
        assert!(color_mask.requires_range_source());
        assert!(can_stream_geometry_output(&adjustments, &params));
    }

    fn encode_streaming_fixture(
        image: &DynamicImage,
        output_format: &str,
        jpeg_quality: u8,
    ) -> Result<Vec<u8>, String> {
        encode_streaming_fixture_with_metadata(
            image,
            output_format,
            jpeg_quality,
            StreamingExportMetadata::None,
        )
    }

    fn encode_streaming_fixture_with_exif(
        image: &DynamicImage,
        output_format: &str,
        jpeg_quality: u8,
        export_exif: Option<&[u8]>,
    ) -> Result<Vec<u8>, String> {
        encode_streaming_fixture_with_metadata(
            image,
            output_format,
            jpeg_quality,
            export_exif.map_or(StreamingExportMetadata::None, |payload| {
                StreamingExportMetadata::ExifTiffPayload(payload)
            }),
        )
    }

    fn encode_streaming_fixture_with_tiff_metadata(
        image: &DynamicImage,
        export_metadata: Option<&Metadata>,
    ) -> Result<Vec<u8>, String> {
        encode_streaming_fixture_with_metadata(
            image,
            "tiff",
            100,
            export_metadata.map_or(StreamingExportMetadata::None, |metadata| {
                StreamingExportMetadata::TiffDirectories(metadata)
            }),
        )
    }

    fn encode_streaming_fixture_with_metadata(
        image: &DynamicImage,
        output_format: &str,
        jpeg_quality: u8,
        export_metadata: StreamingExportMetadata<'_>,
    ) -> Result<Vec<u8>, String> {
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let row_bytes = width as usize * 4;
        let mut file = tempfile::tempfile().map_err(|error| error.to_string())?;
        encode_streaming_rgba_rows(
            &mut file,
            width,
            height,
            output_format,
            jpeg_quality,
            default_export_bit_depth_for_format(output_format),
            true,
            export_metadata,
            |sink| {
                for row in rgba.as_raw().chunks_exact(row_bytes) {
                    sink(row)?;
                }
                Ok(())
            },
        )?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| error.to_string())?;
        let mut encoded = Vec::new();
        file.read_to_end(&mut encoded)
            .map_err(|error| error.to_string())?;
        Ok(encoded)
    }

    fn collect_streaming_transform(
        image: &DynamicImage,
        settings: &ExportSettings,
    ) -> Result<RgbaImage, String> {
        let rgba = image.to_rgba8();
        let (source_width, source_height) = rgba.dimensions();
        let (target_width, target_height) = settings
            .resize
            .as_ref()
            .map(|resize| calculate_resize_target(source_width, source_height, resize))
            .unwrap_or((source_width, source_height));
        let source_row_bytes = source_width as usize * 4;
        let mut transformed =
            Vec::with_capacity(target_width as usize * target_height as usize * 4);
        let mut output_sink = |row: &[u8]| -> Result<(), String> {
            transformed.extend_from_slice(row);
            Ok(())
        };
        transform_streaming_rgba_rows(
            source_width,
            source_height,
            target_width,
            target_height,
            settings.watermark.as_ref(),
            &mut output_sink,
            |source_sink| {
                for row in rgba.as_raw().chunks_exact(source_row_bytes) {
                    source_sink(row)?;
                }
                Ok(())
            },
        )?;

        RgbaImage::from_raw(target_width, target_height, transformed)
            .ok_or_else(|| "Streaming transform returned an invalid RGBA buffer".to_string())
    }

    fn assert_decoder_contract<D: ImageDecoder>(
        mut decoder: D,
        expected: (u32, u32),
        expect_exact_profile_bytes: bool,
    ) {
        assert_eq!(decoder.dimensions(), expected);
        let profile = decoder
            .icc_profile()
            .expect("read embedded ICC")
            .expect("export must contain an ICC profile");
        validate_icc_profile(&profile).expect("exported ICC profile must be valid RGB ICC");

        if expect_exact_profile_bytes {
            assert_eq!(profile, srgb_v4_profile());
        }
    }

    fn mean_rgb_abs_error(reference: &RgbImage, candidate: &RgbImage) -> f64 {
        assert_eq!(reference.dimensions(), candidate.dimensions());
        let total_delta: u64 = reference
            .pixels()
            .zip(candidate.pixels())
            .map(|(reference, candidate)| {
                (0..3)
                    .map(|channel| u64::from(reference[channel].abs_diff(candidate[channel])))
                    .sum::<u64>()
            })
            .sum();
        total_delta as f64 / (reference.width() as f64 * reference.height() as f64 * 3.0)
    }

    #[test]
    fn supported_color_managed_exports_round_trip_dimensions_and_icc() {
        let image = fixture_image(37, 23);

        let jpeg = encode_image_to_bytes(&image, "jpeg", 92).expect("encode JPEG");
        assert_decoder_contract(
            JpegDecoder::new(Cursor::new(jpeg)).expect("decode JPEG"),
            (37, 23),
            true,
        );

        let png = encode_image_to_bytes(&image, "png", 100).expect("encode PNG");
        assert_decoder_contract(
            PngDecoder::new(Cursor::new(png)).expect("decode PNG"),
            (37, 23),
            true,
        );

        let tiff = encode_image_to_bytes(&image, "tiff", 100).expect("encode TIFF");
        assert_decoder_contract(
            TiffDecoder::new(Cursor::new(tiff)).expect("decode TIFF"),
            (37, 23),
            true,
        );

        let webp = encode_image_to_bytes(&image, "webp", 90).expect("encode WebP");
        assert_decoder_contract(
            WebPDecoder::new(Cursor::new(webp)).expect("decode WebP"),
            (37, 23),
            true,
        );

        let jxl = encode_image_to_bytes(&image, "jxl", 90).expect("encode JPEG XL");
        assert_decoder_contract(
            JxlDecoder::new(Cursor::new(jxl)).expect("decode JPEG XL"),
            (37, 23),
            false,
        );
    }

    #[test]
    fn bounded_webp_output_matches_the_existing_encoder_bytes() {
        let rgb = fixture_image(37, 23);
        let rgba = DynamicImage::ImageRgba8(RgbaImage::from_fn(37, 23, |x, y| {
            Rgba([
                ((x * 17 + y * 3) % 256) as u8,
                ((x * 5 + y * 13) % 256) as u8,
                ((x * 11 + y * 7) % 256) as u8,
                ((x * 19 + y * 23) % 256) as u8,
            ])
        }));
        let directory = tempfile::tempdir().expect("temporary WebP directory");
        let cancellation_token = AtomicBool::new(false);

        for (index, image) in [rgb, rgba].iter().enumerate() {
            for quality in [50, 75, 90, 100] {
                let expected =
                    encode_image_to_bytes(image, "webp", quality).unwrap_or_else(|error| {
                        panic!("encode reference WebP {index} Q{quality}: {error}")
                    });
                let output_path = directory
                    .path()
                    .join(format!("bounded-{index}-q{quality}.webp"));
                save_webp_with_bounded_output(image, &output_path, quality, &cancellation_token)
                    .unwrap_or_else(|error| {
                        panic!("write bounded WebP {index} Q{quality}: {error}")
                    });
                let actual = fs::read(&output_path).expect("read bounded WebP");

                assert_eq!(actual, expected);
                assert_decoder_contract(
                    WebPDecoder::new(Cursor::new(actual)).expect("decode bounded WebP"),
                    image.dimensions(),
                    true,
                );
            }
        }
    }

    #[test]
    fn bounded_webp_icc_rewrite_replaces_existing_profile_and_rejects_bad_input() {
        let image = fixture_image(37, 23);
        let encoded = encode_image_to_bytes(&image, "webp", 90).expect("encode ICC WebP");
        let cancellation_token = AtomicBool::new(false);
        let mut source = Cursor::new(encoded.clone());
        let mut output = Cursor::new(Vec::new());
        let output_len = rewrite_webp_icc_bounded(
            &mut source,
            &mut output,
            srgb_v4_profile(),
            image.width(),
            image.height(),
            false,
            &cancellation_token,
        )
        .expect("rewrite existing WebP ICC");
        assert_eq!(output_len as usize, encoded.len());
        assert_eq!(output.into_inner(), encoded);
        assert_eq!(WEBP_OUTPUT_COPY_BUFFER_BYTES, 65_536);

        let mut truncated = source.into_inner();
        truncated.pop();
        let error = rewrite_webp_icc_bounded(
            &mut Cursor::new(truncated),
            &mut Cursor::new(Vec::new()),
            srgb_v4_profile(),
            image.width(),
            image.height(),
            false,
            &cancellation_token,
        )
        .expect_err("truncated WebP must fail");
        assert!(error.contains("RIFF length"), "unexpected error: {error}");

        let cancelled = AtomicBool::new(true);
        let error = rewrite_webp_icc_bounded(
            &mut Cursor::new(encoded),
            &mut Cursor::new(Vec::new()),
            srgb_v4_profile(),
            image.width(),
            image.height(),
            false,
            &cancelled,
        )
        .expect_err("cancelled WebP rewrite must fail");
        assert_eq!(error, "Export cancelled");
    }

    #[test]
    fn streaming_color_managed_exports_round_trip_dimensions_and_icc() {
        let image = fixture_image(37, 23);
        let expected_rgba = image.to_rgba8();

        let jpeg = encode_streaming_fixture(&image, "jpeg", 92).expect("stream JPEG");
        assert_decoder_contract(
            JpegDecoder::new(Cursor::new(&jpeg)).expect("decode streaming JPEG"),
            (37, 23),
            true,
        );

        let png = encode_streaming_fixture(&image, "png", 92).expect("stream PNG");
        assert_decoder_contract(
            PngDecoder::new(Cursor::new(&png)).expect("decode streaming PNG"),
            (37, 23),
            true,
        );
        let decoded_png = image::load_from_memory_with_format(&png, ImageFormat::Png)
            .expect("load streaming PNG pixels")
            .to_rgba8();
        assert_eq!(decoded_png, expected_rgba);

        let tiff = encode_streaming_fixture(&image, "tiff", 92).expect("stream TIFF");
        assert_decoder_contract(
            TiffDecoder::new(Cursor::new(&tiff)).expect("decode streaming TIFF"),
            (37, 23),
            true,
        );
        let decoded_tiff = image::load_from_memory_with_format(&tiff, ImageFormat::Tiff)
            .expect("load streaming TIFF pixels")
            .to_rgba8();
        assert_eq!(decoded_tiff, expected_rgba);
    }

    #[test]
    fn streaming_jpeg_png_and_tiff_embed_bounded_exif_and_strip_gps() {
        fn read_export_exif(encoded: &[u8]) -> exif::Exif {
            exif::Reader::new()
                .read_from_container(&mut Cursor::new(encoded))
                .expect("read streamed export EXIF")
        }

        fn ascii_tag(exif_data: &exif::Exif, tag: exif::Tag) -> Option<String> {
            let field = exif_data.get_field(tag, exif::In::PRIMARY)?;
            let exif::Value::Ascii(values) = &field.value else {
                return None;
            };
            Some(
                values
                    .iter()
                    .map(|value| {
                        String::from_utf8_lossy(value)
                            .trim_end_matches('\0')
                            .to_string()
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            )
        }

        let image = fixture_image(37, 23);
        let mut source = encode_image_to_bytes(&image, "jpeg", 92).expect("encode EXIF source");
        let mut source_metadata = Metadata::new();
        source_metadata.set_tag(ExifTag::Make("Bounded Camera".to_string()));
        source_metadata.set_tag(ExifTag::Model("Synthetic 60MP".to_string()));
        source_metadata.set_tag(ExifTag::Artist("Original Artist".to_string()));
        source_metadata.set_tag(ExifTag::ImageDescription(
            "Museum export fixture".to_string(),
        ));
        source_metadata.set_tag(ExifTag::DateTimeOriginal("2026:08:10 12:34:56".to_string()));
        source_metadata.set_tag(ExifTag::GPSLatitudeRef("N".to_string()));
        source_metadata.set_tag(ExifTag::GPSLatitude(vec![
            uR64 {
                nominator: 31,
                denominator: 1,
            },
            uR64 {
                nominator: 14,
                denominator: 1,
            },
            uR64 {
                nominator: 15,
                denominator: 1,
            },
        ]));
        source_metadata
            .write_to_vec(&mut source, FileExtension::JPEG)
            .expect("attach source EXIF");

        let directory = tempfile::tempdir().expect("create metadata test directory");
        let source_path = directory.path().join("source.jpg");
        std::fs::write(&source_path, source).expect("write EXIF source");
        let source_path = source_path.to_string_lossy();

        let retained =
            exif_processing::export_metadata_tiff_payload(&source_path, "jpeg", true, false, None)
                .expect("build retained EXIF")
                .expect("retained EXIF must exist");
        let jpeg = encode_streaming_fixture_with_exif(&image, "jpeg", 92, Some(&retained))
            .expect("stream JPEG with EXIF");
        assert_decoder_contract(
            JpegDecoder::new(Cursor::new(&jpeg)).expect("decode metadata JPEG"),
            (37, 23),
            true,
        );
        let jpeg_exif = read_export_exif(&jpeg);
        assert_eq!(
            ascii_tag(&jpeg_exif, exif::Tag::Make).as_deref(),
            Some("Bounded Camera")
        );
        assert_eq!(
            ascii_tag(&jpeg_exif, exif::Tag::ImageDescription).as_deref(),
            Some("Museum export fixture")
        );
        assert!(
            jpeg_exif
                .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                .is_some(),
            "JPEG must retain GPS when stripGps is false"
        );

        let cleared_metadata = exif_processing::export_metadata_tiff_payload(
            &source_path,
            "jpeg",
            true,
            false,
            Some(&exif_processing::ExportMetadataOverrides {
                artist: Some(String::new()),
                contact: None,
                copyright: None,
                description: None,
            }),
        )
        .expect("build EXIF with an explicit cleared artist")
        .expect("cleared EXIF must still contain retained metadata");
        let cleared_jpeg =
            encode_streaming_fixture_with_exif(&image, "jpeg", 92, Some(&cleared_metadata))
                .expect("stream JPEG with cleared artist");
        assert!(
            read_export_exif(&cleared_jpeg)
                .get_field(exif::Tag::Artist, exif::In::PRIMARY)
                .is_none(),
            "an empty editable artist field must remove the retained source artist"
        );

        let stripped =
            exif_processing::export_metadata_tiff_payload(&source_path, "png", true, true, None)
                .expect("build stripped EXIF")
                .expect("stripped EXIF must exist");
        let png = encode_streaming_fixture_with_exif(&image, "png", 92, Some(&stripped))
            .expect("stream PNG with EXIF");
        assert_decoder_contract(
            PngDecoder::new(Cursor::new(&png)).expect("decode metadata PNG"),
            (37, 23),
            true,
        );
        let png_exif = read_export_exif(&png);
        assert_eq!(
            ascii_tag(&png_exif, exif::Tag::Make).as_deref(),
            Some("Bounded Camera")
        );
        assert_eq!(
            ascii_tag(&png_exif, exif::Tag::Software).as_deref(),
            Some("RAW Editor")
        );
        assert!(
            png_exif
                .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                .is_none(),
            "PNG must remove GPS when stripGps is true"
        );

        let retained_tiff_metadata =
            exif_processing::export_metadata_for_streaming_tiff(&source_path, true, false, None)
                .expect("build retained TIFF metadata")
                .expect("retained TIFF metadata must exist");
        let tiff =
            encode_streaming_fixture_with_tiff_metadata(&image, Some(&retained_tiff_metadata))
                .expect("stream TIFF with metadata");
        assert_decoder_contract(
            TiffDecoder::new(Cursor::new(&tiff)).expect("decode metadata TIFF"),
            (37, 23),
            true,
        );
        assert_eq!(
            image::load_from_memory_with_format(&tiff, ImageFormat::Tiff)
                .expect("load metadata TIFF pixels")
                .to_rgba8(),
            image.to_rgba8()
        );
        let tiff_exif = read_export_exif(&tiff);
        assert_eq!(
            ascii_tag(&tiff_exif, exif::Tag::Make).as_deref(),
            Some("Bounded Camera")
        );
        assert_eq!(
            ascii_tag(&tiff_exif, exif::Tag::Software).as_deref(),
            Some("RAW Editor")
        );
        assert_eq!(
            ascii_tag(&tiff_exif, exif::Tag::DateTimeOriginal).as_deref(),
            Some("2026:08:10 12:34:56")
        );
        assert!(
            tiff_exif
                .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                .is_some(),
            "TIFF must retain GPS when stripGps is false"
        );

        let tiff_source_path = directory.path().join("source.tiff");
        std::fs::write(&tiff_source_path, &tiff).expect("write metadata TIFF source");
        let copied_tiff_metadata = exif_processing::export_metadata_for_streaming_tiff(
            &tiff_source_path.to_string_lossy(),
            true,
            false,
            None,
        )
        .expect("copy TIFF source metadata")
        .expect("copied TIFF metadata must exist");
        let copied_tiff =
            encode_streaming_fixture_with_tiff_metadata(&image, Some(&copied_tiff_metadata))
                .expect("stream TIFF copied from TIFF metadata");
        let copied_tiff_exif = read_export_exif(&copied_tiff);
        assert_eq!(
            ascii_tag(&copied_tiff_exif, exif::Tag::Make).as_deref(),
            Some("Bounded Camera")
        );
        assert!(
            copied_tiff_exif
                .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                .is_some(),
            "TIFF source metadata must remain readable and copyable"
        );

        let stripped_tiff_metadata =
            exif_processing::export_metadata_for_streaming_tiff(&source_path, true, true, None)
                .expect("build stripped TIFF metadata")
                .expect("stripped TIFF metadata must exist");
        let stripped_tiff =
            encode_streaming_fixture_with_tiff_metadata(&image, Some(&stripped_tiff_metadata))
                .expect("stream TIFF without GPS");
        let stripped_tiff_exif = read_export_exif(&stripped_tiff);
        assert_eq!(
            ascii_tag(&stripped_tiff_exif, exif::Tag::Make).as_deref(),
            Some("Bounded Camera")
        );
        assert!(
            stripped_tiff_exif
                .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                .is_none(),
            "TIFF must remove GPS when stripGps is true"
        );
    }

    #[test]
    fn streaming_jpeg_preserves_the_existing_quality_scale() {
        let image = fixture_image(257, 173);
        let reference = image.to_rgb8();
        let mut previous_streamed_error = f64::INFINITY;
        let mut previous_streamed_bytes = 0_usize;

        for quality in [50, 75, 92] {
            let legacy = encode_image_to_bytes(&image, "jpeg", quality)
                .unwrap_or_else(|error| panic!("encode legacy JPEG Q{quality}: {error}"));
            let streamed = encode_streaming_fixture(&image, "jpeg", quality)
                .unwrap_or_else(|error| panic!("stream JPEG Q{quality}: {error}"));
            let legacy_decoded = image::load_from_memory_with_format(&legacy, ImageFormat::Jpeg)
                .unwrap_or_else(|error| panic!("decode legacy JPEG Q{quality}: {error}"))
                .to_rgb8();
            let streamed_decoded =
                image::load_from_memory_with_format(&streamed, ImageFormat::Jpeg)
                    .unwrap_or_else(|error| panic!("decode streaming JPEG Q{quality}: {error}"))
                    .to_rgb8();
            let legacy_error = mean_rgb_abs_error(&reference, &legacy_decoded);
            let streamed_error = mean_rgb_abs_error(&reference, &streamed_decoded);
            let size_delta = streamed.len().abs_diff(legacy.len());
            let allowed_size_delta = (legacy.len() / 20).max(512);

            assert!(
                streamed_error <= legacy_error * 1.05 + 0.15,
                "streamed JPEG Q{quality} mean RGB error {streamed_error:.3} drifted from legacy {legacy_error:.3}"
            );
            assert!(
                size_delta <= allowed_size_delta,
                "streamed JPEG Q{quality} size {} drifted from legacy {}",
                streamed.len(),
                legacy.len()
            );
            assert!(
                streamed_error < previous_streamed_error,
                "streamed JPEG error did not improve at Q{quality}"
            );
            assert!(
                streamed.len() > previous_streamed_bytes,
                "streamed JPEG size did not increase at Q{quality}"
            );
            previous_streamed_error = streamed_error;
            previous_streamed_bytes = streamed.len();
        }
    }

    #[test]
    fn streaming_export_requires_exactly_one_complete_frame() {
        let mut file = tempfile::tempfile().expect("temporary output");
        let error = encode_streaming_rgba_rows(
            &mut file,
            4,
            3,
            "png",
            90,
            8,
            true,
            StreamingExportMetadata::None,
            |sink| {
                sink(&[0; 4 * 4])?;
                sink(&[0; 4 * 4])?;
                Ok(())
            },
        )
        .expect_err("missing final row must fail");
        assert!(error.contains("expected 3"), "unexpected error: {error}");

        let mut overflow_file = tempfile::tempfile().expect("temporary overflow output");
        let error = encode_streaming_rgba_rows(
            &mut overflow_file,
            4,
            1,
            "png",
            90,
            8,
            true,
            StreamingExportMetadata::None,
            |sink| {
                sink(&[0; 4 * 4])?;
                sink(&[0; 4 * 4])?;
                Ok(())
            },
        )
        .expect_err("extra row must fail");
        assert!(
            error.contains("more than the declared 1 rows"),
            "unexpected error: {error}"
        );

        let mut oversized_jpeg = tempfile::tempfile().expect("temporary oversized JPEG");
        let error = encode_streaming_rgba_rows(
            &mut oversized_jpeg,
            u32::from(u16::MAX) + 1,
            1,
            "jpeg",
            90,
            8,
            true,
            StreamingExportMetadata::None,
            |_| Ok(()),
        )
        .expect_err("oversized JPEG dimensions must fail before encoding");
        assert!(error.contains("65535"), "unexpected error: {error}");

        let settings = ExportSettings {
            bit_depth: 8,
            jpeg_quality: 90,
            resize: None,
            keep_metadata: false,
            metadata_overrides: None,
            preserve_timestamps: false,
            strip_gps: false,
            embed_color_profile: true,
            filename_template: None,
            watermark: None,
            export_masks: false,
            preserve_folders: false,
        };
        assert!(supports_streaming_export("jpeg", &settings));
        assert!(supports_streaming_export("png", &settings));
        assert!(supports_streaming_export("tiff", &settings));
        assert!(!supports_streaming_export("webp", &settings));

        let mut resized = settings.clone();
        resized.resize = Some(ResizeOptions {
            mode: ResizeMode::LongEdge,
            value: 2_048,
            dont_enlarge: true,
        });
        assert!(supports_streaming_export("png", &resized));

        let mut watermarked = resized;
        watermarked.watermark = Some(WatermarkSettings {
            path: "unused-routing-fixture.png".to_string(),
            anchor: WatermarkAnchor::BottomRight,
            scale: 10.0,
            spacing: 5.0,
            opacity: 50.0,
        });
        assert!(supports_streaming_export("jpeg", &watermarked));
    }

    #[test]
    fn streaming_resize_matches_the_bounded_batch_reference() {
        let image = fixture_image(37, 23);
        let settings = ExportSettings {
            bit_depth: 8,
            jpeg_quality: 92,
            resize: Some(ResizeOptions {
                mode: ResizeMode::Width,
                value: 19,
                dont_enlarge: false,
            }),
            keep_metadata: false,
            metadata_overrides: None,
            preserve_timestamps: false,
            strip_gps: false,
            embed_color_profile: true,
            filename_template: None,
            watermark: None,
            export_masks: false,
            preserve_folders: false,
        };
        let streamed = collect_streaming_transform(&image, &settings).expect("stream resize");
        let source = image.to_rgba8();
        let (target_width, target_height) = streamed.dimensions();
        let config =
            ResizeConfig::builder(source.width(), source.height(), target_width, target_height)
                .filter(StreamingResizeFilter::Lanczos)
                .format(PixelDescriptor::RGBA8_SRGB)
                .srgb()
                .build();
        let expected = zenresize::Resizer::new(&config).resize(source.as_raw());

        assert_eq!(streamed.as_raw(), &expected);
        assert_eq!(streamed.dimensions(), (19, 12));

        let legacy = image
            .resize(19, 12, imageops::FilterType::Lanczos3)
            .to_rgba8();
        let mut max_channel_delta = 0_u8;
        let mut total_channel_delta = 0_u64;
        for (streamed, legacy) in streamed.pixels().zip(legacy.pixels()) {
            for channel in 0..3 {
                let delta = streamed[channel].abs_diff(legacy[channel]);
                max_channel_delta = max_channel_delta.max(delta);
                total_channel_delta += u64::from(delta);
            }
        }
        let mean_channel_delta = total_channel_delta as f64 / (19 * 12 * 3) as f64;
        assert!(
            mean_channel_delta <= 0.5,
            "mean channel delta {mean_channel_delta:.3} exceeded the migration bound"
        );
        assert!(
            max_channel_delta <= 2,
            "max channel delta {max_channel_delta} exceeded the migration bound"
        );
    }

    #[test]
    fn imported_watermark_is_normalized_and_deduplicated_in_private_storage() {
        let directory = tempfile::tempdir().expect("watermark import directory");
        let source_path = directory.path().join("My Watermark.png");
        let source = RgbaImage::from_fn(11, 7, |x, y| {
            Rgba([
                (x * 17) as u8,
                (y * 29) as u8,
                ((x + y) * 13) as u8,
                (80 + x * 7 + y * 3) as u8,
            ])
        });
        source.save(&source_path).expect("save source watermark");

        let watermark_directory = directory.path().join("stored-watermarks");
        let first = import_watermark_image_impl(&source_path, &watermark_directory)
            .expect("import watermark");
        let second = import_watermark_image_impl(&source_path, &watermark_directory)
            .expect("deduplicate watermark");

        assert_eq!(first, second);
        assert_eq!(first.parent(), Some(watermark_directory.as_path()));
        assert!(
            first
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("My-Watermark-") && name.ends_with(".png"))
        );
        assert_eq!(
            image::image_dimensions(first).expect("stored dimensions"),
            (11, 7)
        );
    }

    #[test]
    fn bundled_default_watermark_matches_the_reference_asset() {
        let watermark = load_watermark_image(DEFAULT_WATERMARK_PATH)
            .expect("decode the bundled default watermark");
        assert_eq!(watermark.dimensions(), (1908, 1462));

        let prepared = prepare_watermark(
            1000,
            800,
            &WatermarkSettings {
                path: DEFAULT_WATERMARK_PATH.to_string(),
                anchor: WatermarkAnchor::Center,
                scale: 10.0,
                spacing: 5.0,
                opacity: 80.0,
            },
        )
        .expect("prepare the bundled default watermark")
        .expect("default watermark remains visible");

        assert_eq!(prepared.image.width(), 80);
        assert_eq!(prepared.x, 460);
        assert_eq!(prepared.y, (800 - i64::from(prepared.image.height())) / 2);
    }

    #[test]
    fn streaming_watermark_matches_the_existing_full_frame_blend() {
        let image = fixture_image(37, 23);
        let directory = tempfile::tempdir().expect("watermark directory");
        let watermark_path = directory.path().join("watermark.png");
        let watermark = RgbaImage::from_fn(7, 3, |x, y| {
            Rgba([
                (x * 31) as u8,
                (y * 67) as u8,
                ((x + y) * 23) as u8,
                (80 + x * 17 + y * 11) as u8,
            ])
        });
        watermark.save(&watermark_path).expect("save watermark");
        let settings = ExportSettings {
            bit_depth: 8,
            jpeg_quality: 92,
            resize: None,
            keep_metadata: false,
            metadata_overrides: None,
            preserve_timestamps: false,
            strip_gps: false,
            embed_color_profile: true,
            filename_template: None,
            watermark: Some(WatermarkSettings {
                path: watermark_path.to_string_lossy().into_owned(),
                anchor: WatermarkAnchor::BottomRight,
                scale: 24.0,
                spacing: 7.0,
                opacity: 63.0,
            }),
            export_masks: false,
            preserve_folders: false,
        };

        let streamed = collect_streaming_transform(&image, &settings).expect("stream watermark");
        let expected = apply_export_resize_and_watermark(image, &settings)
            .expect("apply existing watermark")
            .to_rgba8();
        assert_eq!(streamed, expected);
    }

    #[cfg(unix)]
    #[test]
    fn streamed_export_temp_preserves_existing_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temporary export directory");
        let output_path = directory.path().join("existing.png");
        fs::write(&output_path, b"old").expect("create existing export");
        fs::set_permissions(&output_path, fs::Permissions::from_mode(0o640))
            .expect("set existing permissions");

        let temporary = create_temporary_export(&output_path).expect("create adjacent temporary");
        preserve_existing_export_permissions(&temporary, &output_path)
            .expect("preserve existing permissions");
        let mode = temporary
            .as_file()
            .metadata()
            .expect("temporary metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o640);
    }

    #[cfg(unix)]
    #[test]
    fn bounded_webp_publish_preserves_permissions_and_cancel_keeps_target() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temporary WebP export directory");
        let output_path = directory.path().join("existing.webp");
        fs::write(&output_path, b"old").expect("create existing WebP export");
        fs::set_permissions(&output_path, fs::Permissions::from_mode(0o640))
            .expect("set existing permissions");

        let image = fixture_image(37, 23);
        save_webp_with_bounded_output(&image, &output_path, 90, &AtomicBool::new(false))
            .expect("publish bounded WebP");
        let published = fs::read(&output_path).expect("read published WebP");
        let mode = fs::metadata(&output_path)
            .expect("published metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o640);

        let error = save_webp_with_bounded_output(&image, &output_path, 90, &AtomicBool::new(true))
            .expect_err("cancelled WebP publish must fail");
        assert_eq!(error, "Export cancelled");
        assert_eq!(fs::read(&output_path).expect("reread WebP"), published);

        let mut callback_output = tempfile::tempfile().expect("callback output");
        let callback_token = AtomicBool::new(true);
        let mut callback_context = WebpFileWriterContext {
            output: &mut callback_output,
            cancellation_token: &callback_token,
            cancelled: false,
            first_error: None,
        };
        let picture = libwebp_sys::WebPPicture::new().expect("initialize callback picture");
        let mut picture = WebpPictureGuard(picture);
        picture.0.user_data = std::ptr::from_mut(&mut callback_context).cast();
        // SAFETY: the initialized picture points to the live callback context.
        let should_continue = unsafe { check_webp_encoding_progress(1, &picture.0) };
        assert_eq!(should_continue, 0);
        assert!(callback_context.cancelled);
    }

    #[test]
    #[ignore = "manual deterministic large-image encoder benchmark"]
    fn synthetic_60mp_webp_output_harness() {
        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode =
            std::env::var("RAW_EDITOR_BENCH_WEBP_MODE").unwrap_or_else(|_| "file".to_string());
        assert!(width > 0 && height > 0);

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(0));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let Ok(pid) = sysinfo::get_current_pid() else {
                return;
            };
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });

        let image = DynamicImage::ImageRgba8(RgbaImage::from_fn(width, height, |x, y| {
            Rgba([
                x.wrapping_mul(17).wrapping_add(y * 3) as u8,
                x.wrapping_mul(5).wrapping_add(y * 13) as u8,
                x.wrapping_mul(11).wrapping_add(y * 7) as u8,
                255,
            ])
        }));
        let benchmark_directory = tempfile::tempdir().expect("temporary WebP benchmark directory");
        let output_path = benchmark_directory.path().join("benchmark.webp");
        let started = Instant::now();
        let mut retained_memory = None;
        let output_bytes = match mode.as_str() {
            "memory" => {
                let encoded =
                    encode_image_to_bytes(&image, "webp", 90).expect("encode WebP in memory");
                let output_bytes = encoded.len() as u64;
                retained_memory = Some(encoded);
                output_bytes
            }
            "file" => {
                save_webp_with_bounded_output(&image, &output_path, 90, &AtomicBool::new(false))
                    .expect("encode WebP to bounded files");
                fs::metadata(&output_path)
                    .expect("benchmark WebP metadata")
                    .len()
            }
            _ => panic!("unsupported WebP benchmark mode: {mode}"),
        };
        let elapsed = started.elapsed();
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join RSS sampler");

        println!(
            "{{\"format\":\"webp\",\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"outputBytes\":{},\"peakRssBytes\":{},\"inputRgbaBytes\":{}}}",
            mode,
            width,
            height,
            elapsed.as_millis(),
            output_bytes,
            peak_rss.load(Ordering::Relaxed),
            u64::from(width) * u64::from(height) * 4,
        );

        std::hint::black_box(retained_memory);
        std::hint::black_box(benchmark_directory);
    }

    #[test]
    #[ignore = "manual deterministic large-image encoder benchmark"]
    fn synthetic_60mp_streaming_export_harness() {
        use crate::render_strategy::StreamingExportBufferPlan;

        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let output_format =
            std::env::var("RAW_EDITOR_BENCH_FORMAT").unwrap_or_else(|_| "png".to_string());
        let resize_long_edge = std::env::var("RAW_EDITOR_BENCH_RESIZE_LONG_EDGE")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0);
        let resize_options = resize_long_edge.map(|value| ResizeOptions {
            mode: ResizeMode::LongEdge,
            value,
            dont_enlarge: true,
        });
        let (target_width, target_height) = resize_options
            .as_ref()
            .map(|resize| calculate_resize_target(width, height, resize))
            .unwrap_or((width, height));
        let watermark_enabled = std::env::var("RAW_EDITOR_BENCH_WATERMARK")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
        let metadata_enabled = std::env::var("RAW_EDITOR_BENCH_METADATA")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
            && matches!(
                output_format.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "tif" | "tiff"
            );
        let benchmark_metadata = metadata_enabled.then(|| {
            let mut metadata = Metadata::new();
            metadata.set_tag(ExifTag::Make("Synthetic benchmark".to_string()));
            metadata.set_tag(ExifTag::Model("Bounded EXIF".to_string()));
            metadata.set_tag(ExifTag::Software("RAW Editor".to_string()));
            metadata
        });
        let is_tiff_output = matches!(output_format.to_ascii_lowercase().as_str(), "tif" | "tiff");
        let export_exif = benchmark_metadata
            .as_ref()
            .filter(|_| !is_tiff_output)
            .map(|metadata| {
                let encoded = metadata
                    .as_u8_vec(FileExtension::JPEG)
                    .expect("encode benchmark EXIF");
                assert_eq!(&encoded[..2], b"\xff\xe1");
                assert_eq!(&encoded[4..10], b"Exif\0\0");
                encoded[10..].to_vec()
            });
        let watermark_directory = watermark_enabled
            .then(|| tempfile::tempdir().expect("temporary benchmark watermark directory"));
        let watermark_settings = watermark_directory.as_ref().map(|directory| {
            let path = directory.path().join("benchmark-watermark.png");
            RgbaImage::from_fn(512, 128, |x, y| {
                Rgba([
                    x.wrapping_mul(29).wrapping_add(y * 3) as u8,
                    x.wrapping_mul(7).wrapping_add(y * 17) as u8,
                    x.wrapping_mul(13).wrapping_add(y * 11) as u8,
                    (96 + (x + y) % 160) as u8,
                ])
            })
            .save(&path)
            .expect("write benchmark watermark");
            WatermarkSettings {
                path: path.to_string_lossy().into_owned(),
                anchor: WatermarkAnchor::BottomRight,
                scale: 12.0,
                spacing: 5.0,
                opacity: 75.0,
            }
        });
        assert!(width > 0 && height > 0);

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(0));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let Ok(pid) = sysinfo::get_current_pid() else {
                return;
            };
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });

        let mut output = tempfile::tempfile().expect("temporary benchmark output");
        let buffer_plan = StreamingExportBufferPlan::new(width, height);
        let row_bytes = width as usize * 4;
        let mut band = vec![0_u8; buffer_plan.band_rgba_bytes()];
        let started = Instant::now();
        encode_streaming_rgba_rows(
            &mut output,
            target_width,
            target_height,
            &output_format,
            92,
            default_export_bit_depth_for_format(&output_format),
            true,
            if let Some(metadata) = benchmark_metadata.as_ref().filter(|_| is_tiff_output) {
                StreamingExportMetadata::TiffDirectories(metadata)
            } else if let Some(payload) = export_exif.as_deref() {
                StreamingExportMetadata::ExifTiffPayload(payload)
            } else {
                StreamingExportMetadata::None
            },
            |encoder_sink| {
                transform_streaming_rgba_rows(
                    width,
                    height,
                    target_width,
                    target_height,
                    watermark_settings.as_ref(),
                    encoder_sink,
                    |source_sink| {
                        for band_y in (0..height).step_by(buffer_plan.band_rows as usize) {
                            let rows = buffer_plan.band_rows.min(height - band_y);
                            let active_band = &mut band[..rows as usize * row_bytes];
                            for (local_y, row) in
                                active_band.chunks_exact_mut(row_bytes).enumerate()
                            {
                                let y = band_y + local_y as u32;
                                for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                                    let x = x as u32;
                                    pixel[0] = x.wrapping_mul(17).wrapping_add(y * 3) as u8;
                                    pixel[1] = x.wrapping_mul(5).wrapping_add(y * 13) as u8;
                                    pixel[2] = x.wrapping_mul(11).wrapping_add(y * 7) as u8;
                                    pixel[3] = 255;
                                }
                            }
                            for row in active_band.chunks_exact(row_bytes) {
                                source_sink(row)?;
                            }
                        }
                        Ok(())
                    },
                )
            },
        )
        .expect("run synthetic streaming export");
        let elapsed = started.elapsed();
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join RSS sampler");
        let output_bytes = output.metadata().expect("benchmark output metadata").len();

        println!(
            "{{\"format\":\"{}\",\"width\":{},\"height\":{},\"targetWidth\":{},\"targetHeight\":{},\"watermark\":{},\"metadata\":{},\"elapsedMs\":{},\"outputBytes\":{},\"peakRssBytes\":{},\"producerBandBytes\":{},\"legacyFullFrameBytes\":{}}}",
            output_format,
            width,
            height,
            target_width,
            target_height,
            watermark_enabled,
            metadata_enabled,
            elapsed.as_millis(),
            output_bytes,
            peak_rss.load(Ordering::Relaxed),
            band.len(),
            buffer_plan.legacy_full_rgba_bytes()
        );
    }

    #[test]
    fn export_bit_depth_controls_png_and_tiff_sample_precision() {
        let image = DynamicImage::ImageRgb16(ImageBuffer::from_fn(4, 3, |x, y| {
            Rgb([
                (x * 12_000 + 1_000) as u16,
                (y * 15_000 + 2_000) as u16,
                ((x + y) * 8_000 + 3_000) as u16,
            ])
        }));

        for (output_format, image_format) in [
            ("png", ImageFormat::Png),
            ("tif", ImageFormat::Tiff),
            ("tiff", ImageFormat::Tiff),
        ] {
            for bit_depth in [8_u8, 16_u8] {
                let bytes =
                    encode_image_to_bytes_with_profile(&image, output_format, 95, false, bit_depth)
                        .expect("encode selected export bit depth");
                let decoded = image::load_from_memory_with_format(&bytes, image_format)
                    .expect("decode selected export bit depth");
                assert_eq!(
                    decoded.color(),
                    if bit_depth == 16 {
                        image::ColorType::Rgb16
                    } else {
                        image::ColorType::Rgb8
                    }
                );
            }
        }
    }

    #[test]
    fn no_resize_export_keeps_the_full_input_dimensions() {
        let image = fixture_image(2049, 1367);
        let settings = ExportSettings {
            bit_depth: 8,
            jpeg_quality: 90,
            resize: None,
            keep_metadata: false,
            metadata_overrides: None,
            preserve_timestamps: false,
            strip_gps: false,
            embed_color_profile: true,
            filename_template: None,
            watermark: None,
            export_masks: false,
            preserve_folders: false,
        };

        let processed =
            apply_export_resize_and_watermark(image, &settings).expect("apply export options");
        assert_eq!(processed.dimensions(), (2049, 1367));

        let bytes = encode_image_to_bytes(&processed, "png", 100).expect("encode full image");
        let decoder = PngDecoder::new(Cursor::new(bytes)).expect("decode full image");
        assert_eq!(decoder.dimensions(), (2049, 1367));
    }

    #[test]
    fn batch_summary_retains_later_results_after_an_item_failure() {
        let summary = summarize_batch_export_results([
            Ok(()),
            Err("second.jpg: simulated encoder failure".to_string()),
            Ok(()),
            Err("fourth.jpg: simulated disk failure".to_string()),
            Ok(()),
        ]);

        assert_eq!(summary.completed, 5);
        assert_eq!(summary.errors.len(), 2);
        assert!(summary.errors[0].contains("second.jpg"));
        assert!(summary.errors[1].contains("fourth.jpg"));
    }
}
