//! Rendering one export source.
//!
//! Prepares the developed source, renders it as a full image or streams
//! geometry-warped GPU rows into the row encoders and publishes the result.

use crate::AppState;
use crate::app_state::SharedMaskBitmap;
use crate::apply_all_transformations;
use crate::cache_utils::calculate_full_job_hash;
use crate::exif_processing;
#[cfg(not(target_os = "android"))]
use crate::gpu_processing::process_and_stream_rgba_rows;
use crate::image_processing::AllAdjustments;
use crate::image_processing::GeometryWarpRows;
use crate::image_processing::GeometryWarpSampler;
use crate::image_processing::GpuContext;
use crate::image_processing::RenderRequest;
use crate::image_processing::get_all_adjustments_from_json;
use crate::image_processing::get_geometry_params_from_json;
use crate::image_processing::is_geometry_identity;
use crate::image_processing::process_and_get_dynamic_image;
use crate::image_processing::resolve_tonemapper_override_from_handle;
use crate::lut_processing::Lut;
use crate::lut_processing::get_or_load_lut;
use crate::mask_generation::MaskDefinition;
use crate::mask_generation::RangeMaskSource;
use crate::mask_generation::generate_mask_bitmap;
use crate::mask_generation::prepare_range_mask_source_image;
use crate::render_strategy::RenderTier;
use image::DynamicImage;
use image::GenericImageView;
#[cfg(not(target_os = "android"))]
use little_exif::metadata::Metadata as ExifMetadata;
use serde_json::Value;
use std::borrow::Cow;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::{
    ExportSettings, apply_export_resize_and_watermark, calculate_resize_target,
    ensure_export_not_cancelled,
};
#[cfg(not(target_os = "android"))]
use super::{
    create_temporary_export, encode_streaming_jpeg, encode_streaming_png, encode_streaming_tiff,
    preserve_existing_export_permissions, transform_streaming_rgba_rows,
};

pub(super) enum PreparedExportSource<'a> {
    Materialized(Cow<'a, DynamicImage>),
    StreamedGeometry(Box<GeometryWarpRows<'a>>),
}

impl PreparedExportSource<'_> {
    pub(super) fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Materialized(image) => image.dimensions(),
            Self::StreamedGeometry(rows) => rows.dimensions(),
        }
    }

    pub(super) fn gpu_input(&self) -> &dyn crate::gpu_processing::GpuInputSource {
        match self {
            Self::Materialized(image) => image.as_ref(),
            Self::StreamedGeometry(rows) => rows.as_ref(),
        }
    }
}

pub(super) struct PreparedExportRender<'a> {
    pub(super) source: PreparedExportSource<'a>,
    pub(super) mask_bitmaps: Vec<SharedMaskBitmap>,
    pub(super) adjustments: AllAdjustments,
    pub(super) lut: Option<Arc<Lut>>,
    pub(super) unique_hash: u64,
}

pub(super) fn can_stream_geometry_output(
    adjustments: &Value,
    geometry_params: &crate::image_processing::GeometryParams,
) -> bool {
    !is_geometry_identity(geometry_params)
        && adjustments["rotation"]
            .as_f64()
            .unwrap_or(0.0)
            .rem_euclid(360.0)
            == 0.0
        && !adjustments["lensBlurEnabled"].as_bool().unwrap_or(false)
}

#[allow(clippy::too_many_arguments)]
fn prepare_export_render<'a>(
    path: &str,
    base_image: &'a DynamicImage,
    js_adjustments: &Value,
    state: &tauri::State<AppState>,
    is_raw: bool,
    debug_tag: &str,
    app_handle: &tauri::AppHandle,
) -> PreparedExportRender<'a> {
    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();
    let geometry_params = get_geometry_params_from_json(js_adjustments);
    let can_stream_geometry = can_stream_geometry_output(js_adjustments, &geometry_params);

    let orientation_steps = js_adjustments["orientationSteps"].as_u64().unwrap_or(0) as u8;
    let flip_horizontal = js_adjustments["flipHorizontal"].as_bool().unwrap_or(false);
    let flip_vertical = js_adjustments["flipVertical"].as_bool().unwrap_or(false);
    let streamed_geometry = can_stream_geometry.then(|| {
        GeometryWarpRows::try_new_transformed_borrowed(
            base_image,
            geometry_params,
            orientation_steps,
            flip_horizontal,
            flip_vertical,
            &js_adjustments["crop"],
        )
    });
    let (source, unscaled_crop_offset) = match streamed_geometry.flatten() {
        Some(rows) => {
            let saved_bytes = rows.materialized_output_bytes();
            let checkpoint_bytes = rows.checkpoint_bytes();
            let band_scratch_bytes =
                rows.max_band_scratch_bytes(crate::render_strategy::GPU_INPUT_UPLOAD_BAND_ROWS);
            let crop_offset = rows.crop_offset();
            log::info!(
                "[{}] streaming transformed geometry bands directly to GPU input (avoiding {} B RGBA32F output; checkpoints={} B, transpose_scratch<={} B)",
                debug_tag,
                saved_bytes,
                checkpoint_bytes,
                band_scratch_bytes,
            );
            (
                PreparedExportSource::StreamedGeometry(Box::new(rows)),
                crop_offset,
            )
        }
        None => {
            let (image, crop_offset) =
                apply_all_transformations(Cow::Borrowed(base_image), js_adjustments);
            (PreparedExportSource::Materialized(image), crop_offset)
        }
    };
    let (img_w, img_h) = source.dimensions();
    log::info!(
        "[{}] tier={} source={}x{}",
        debug_tag,
        RenderTier::FullResolutionExport.as_str(),
        img_w,
        img_h
    );

    let range_source_image = prepare_range_mask_source_image(base_image, is_raw, &mask_definitions);
    let range_source = range_source_image.as_deref().map(|image| {
        GeometryWarpSampler::new(image, get_geometry_params_from_json(js_adjustments))
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
    if let Some(source) = range_source.as_ref() {
        log::info!(
            "[{}] range masks sampled geometry on demand (checkpoints={} B, maximum={} B)",
            debug_tag,
            source.checkpoint_bytes(),
            source.max_checkpoint_bytes(),
        );
    }

    let tm_override = resolve_tonemapper_override_from_handle(app_handle, is_raw);
    let mut all_adjustments = get_all_adjustments_from_json(js_adjustments, is_raw, tm_override);
    all_adjustments.global.show_clipping = 0;

    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| get_or_load_lut(state, p).ok());

    let unique_hash = calculate_full_job_hash(path, js_adjustments);

    PreparedExportRender {
        source,
        mask_bitmaps,
        adjustments: all_adjustments,
        lut,
        unique_hash,
    }
}

#[allow(clippy::too_many_arguments)]
fn process_image_for_export_pipeline(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    context: &GpuContext,
    state: &tauri::State<AppState>,
    is_raw: bool,
    debug_tag: &str,
    app_handle: &tauri::AppHandle,
) -> Result<DynamicImage, String> {
    let prepared = prepare_export_render(
        path,
        base_image,
        js_adjustments,
        state,
        is_raw,
        debug_tag,
        app_handle,
    );

    process_and_get_dynamic_image(
        context,
        state,
        prepared.source.gpu_input(),
        prepared.unique_hash,
        RenderRequest {
            adjustments: prepared.adjustments,
            mask_bitmaps: &prepared.mask_bitmaps,
            lut: prepared.lut,
            roi: None,
        },
        debug_tag,
    )
}

#[cfg(not(target_os = "android"))]
#[derive(Clone, Copy)]
pub(super) enum StreamingExportMetadata<'a> {
    None,
    ExifTiffPayload(&'a [u8]),
    TiffDirectories(&'a ExifMetadata),
}

#[cfg(not(target_os = "android"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_streaming_rgba_rows<F>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    output_format: &str,
    jpeg_quality: u8,
    bit_depth: u8,
    embed_color_profile: bool,
    export_metadata: StreamingExportMetadata<'_>,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    match output_format.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => {
            let exif = match export_metadata {
                StreamingExportMetadata::ExifTiffPayload(payload) => Some(payload),
                _ => None,
            };
            encode_streaming_jpeg(
                output,
                width,
                height,
                jpeg_quality,
                embed_color_profile,
                exif,
                render_rows,
            )
        }
        "png" => {
            let exif = match export_metadata {
                StreamingExportMetadata::ExifTiffPayload(payload) => Some(payload),
                _ => None,
            };
            encode_streaming_png(
                output,
                width,
                height,
                bit_depth,
                embed_color_profile,
                exif,
                render_rows,
            )
        }
        "tif" | "tiff" => {
            let metadata = match export_metadata {
                StreamingExportMetadata::TiffDirectories(metadata) => Some(metadata),
                _ => None,
            };
            encode_streaming_tiff(
                output,
                width,
                height,
                bit_depth,
                embed_color_profile,
                metadata,
                render_rows,
            )
        }
        _ => Err(format!(
            "Unsupported streaming export format: {output_format}"
        )),
    }
}

#[cfg(not(target_os = "android"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn process_and_save_streaming_export(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    output_path: &Path,
    output_format: &str,
    export_settings: &ExportSettings,
    context: &GpuContext,
    state: &tauri::State<AppState>,
    is_raw: bool,
    app_handle: &tauri::AppHandle,
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    let prepared = prepare_export_render(
        path,
        base_image,
        js_adjustments,
        state,
        is_raw,
        "process_and_save_streaming_export",
        app_handle,
    );
    let (source_width, source_height) = prepared.source.dimensions();
    let (target_width, target_height) = export_settings
        .resize
        .as_ref()
        .map(|resize| calculate_resize_target(source_width, source_height, resize))
        .unwrap_or((source_width, source_height));
    let is_tiff_output = matches!(output_format.to_ascii_lowercase().as_str(), "tif" | "tiff");
    let export_exif = if is_tiff_output {
        None
    } else {
        exif_processing::export_metadata_tiff_payload(
            path,
            output_format,
            export_settings.keep_metadata,
            export_settings.strip_gps,
            export_settings.metadata_overrides.as_ref(),
        )?
    };
    let export_tiff_metadata = if is_tiff_output {
        exif_processing::export_metadata_for_streaming_tiff(
            path,
            export_settings.keep_metadata,
            export_settings.strip_gps,
            export_settings.metadata_overrides.as_ref(),
        )?
    } else {
        None
    };
    let streaming_metadata = if let Some(metadata) = export_tiff_metadata.as_ref() {
        StreamingExportMetadata::TiffDirectories(metadata)
    } else if let Some(payload) = export_exif.as_deref() {
        StreamingExportMetadata::ExifTiffPayload(payload)
    } else {
        StreamingExportMetadata::None
    };
    let mut temporary = create_temporary_export(output_path)?;

    let PreparedExportRender {
        source,
        mask_bitmaps,
        adjustments,
        lut,
        unique_hash,
    } = prepared;
    encode_streaming_rgba_rows(
        temporary.as_file_mut(),
        target_width,
        target_height,
        output_format,
        export_settings.jpeg_quality,
        export_settings.bit_depth,
        export_settings.embed_color_profile,
        streaming_metadata,
        |encoder_sink| {
            let mut checked_encoder_sink = |row: &[u8]| -> Result<(), String> {
                ensure_export_not_cancelled(cancellation_token)?;
                encoder_sink(row)
            };
            transform_streaming_rgba_rows(
                source_width,
                source_height,
                target_width,
                target_height,
                export_settings.watermark.as_ref(),
                &mut checked_encoder_sink,
                |transform_sink| {
                    let mut checked_sink = |row: &[u8]| -> Result<(), String> {
                        ensure_export_not_cancelled(cancellation_token)?;
                        transform_sink(row)
                    };
                    let rendered_dimensions = process_and_stream_rgba_rows(
                        context,
                        state,
                        source.gpu_input(),
                        unique_hash,
                        RenderRequest {
                            adjustments,
                            mask_bitmaps: &mask_bitmaps,
                            lut,
                            roi: None,
                        },
                        "process_and_save_streaming_export",
                        &mut checked_sink,
                    )?;
                    if rendered_dimensions != (source_width, source_height) {
                        return Err(format!(
                            "Streamed GPU dimensions {:?} do not match source dimensions {}x{}",
                            rendered_dimensions, source_width, source_height
                        ));
                    }
                    Ok(())
                },
            )
        },
    )?;
    temporary
        .as_file_mut()
        .flush()
        .map_err(|error| format!("Failed to flush temporary export: {error}"))?;
    ensure_export_not_cancelled(cancellation_token)?;
    ensure_export_not_cancelled(cancellation_token)?;
    preserve_existing_export_permissions(&temporary, output_path)?;

    temporary.persist(output_path).map_err(|error| {
        format!(
            "Failed to atomically publish streamed export '{}': {}",
            output_path.display(),
            error.error
        )
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_image_for_export(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    export_settings: &ExportSettings,
    context: &GpuContext,
    state: &tauri::State<AppState>,
    is_raw: bool,
    app_handle: &tauri::AppHandle,
) -> Result<DynamicImage, String> {
    let processed_image = process_image_for_export_pipeline(
        path,
        base_image,
        js_adjustments,
        context,
        state,
        is_raw,
        "process_image_for_export",
        app_handle,
    )?;

    apply_export_resize_and_watermark(processed_image, export_settings)
}
