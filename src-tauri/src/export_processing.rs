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
mod tests;
