use crate::app_state::AppState;
use crate::export_processing::ExportSettings;
use crate::file_management::parse_virtual_path;
use crate::panorama_stitching::{AlignmentMode, BlendMode, stitch_images_with_options};
use image::codecs::jpeg::{JpegDecoder, JpegEncoder};
use image::codecs::png::{PngDecoder, PngEncoder};
use image::codecs::tiff::{TiffDecoder, TiffEncoder};
use image::imageops::FilterType;
use image::{ColorType, DynamicImage, GenericImageView, ImageDecoder, ImageEncoder, RgbImage};
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use tauri::{AppHandle, Emitter, Manager};
use tempfile::NamedTempFile;
use uuid::Uuid;

// Keep the UI proxy comfortably inside desktop WebView image-decoding budgets.
// The full-resolution result remains in AppState and is used for TIFF export.
const PREVIEW_MAX_LONG_SIDE: u32 = 4_096;
const PREVIEW_MAX_PIXELS: u64 = 12_000_000;
const PREVIEW_JPEG_QUALITY: u8 = 92;
const DETAIL_PREVIEW_MAX_LONG_SIDE: u32 = 8_192;
const DETAIL_PREVIEW_MAX_PIXELS: u64 = 32_000_000;
const DETAIL_PREVIEW_JPEG_QUALITY: u8 = 98;
#[cfg(test)]
const IMAGE_STACK_JPEG_QUALITY: u8 = 95;
const IMAGE_STACK_MAX_SOURCES: usize = 200;
const IMAGE_STACK_PIPELINE_VERSION: &str = "image-stack-2026.08.24.2";

fn validate_image_stack_source_count(count: usize) -> Result<(), String> {
    if count < 2 {
        return Err("Please select at least two images.".to_string());
    }
    if count > IMAGE_STACK_MAX_SOURCES {
        return Err(format!(
            "Image stack is limited to {IMAGE_STACK_MAX_SOURCES} source images."
        ));
    }
    Ok(())
}

fn validate_image_stack_pipeline_version(value: &str) -> Result<(), String> {
    if value == IMAGE_STACK_PIPELINE_VERSION {
        return Ok(());
    }

    Err(format!(
        "The image-stack backend is out of date (expected {IMAGE_STACK_PIPELINE_VERSION}, received {}). Fully quit and restart RAW Editor before stacking again.",
        if value.trim().is_empty() {
            "no version"
        } else {
            value
        }
    ))
}

fn resolve_blend_mode(value: &str) -> BlendMode {
    match value {
        "focus" => BlendMode::FocusStack,
        _ => BlendMode::Panorama,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageStackOutputFormat {
    Tiff,
    Png,
    Jpeg,
}

impl ImageStackOutputFormat {
    fn from_wire(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tif" | "tiff" => Ok(Self::Tiff),
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            _ => Err(format!(
                "Unsupported image-stack output format '{value}'. Choose TIFF, PNG, or JPEG."
            )),
        }
    }

    fn from_extension(extension: &str) -> Option<Self> {
        match extension.to_ascii_lowercase().as_str() {
            "tif" | "tiff" => Some(Self::Tiff),
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            _ => None,
        }
    }

    fn canonical_extension(self) -> &'static str {
        match self {
            Self::Tiff => "tiff",
            Self::Png => "png",
            Self::Jpeg => "jpg",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Tiff => "TIFF",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
        }
    }

    fn encoded_color_type(self) -> ColorType {
        match self {
            Self::Tiff | Self::Png => ColorType::Rgb16,
            Self::Jpeg => ColorType::Rgb8,
        }
    }
}

fn bounded_preview_dimensions(
    width: u32,
    height: u32,
    max_long_side: u32,
    max_pixels: u64,
) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (0, 0);
    }
    let long_side_scale = max_long_side as f64 / width.max(height) as f64;
    let pixel_scale = (max_pixels as f64 / (width as u64 * height as u64) as f64).sqrt();
    let scale = 1.0_f64.min(long_side_scale).min(pixel_scale);
    (
        (width as f64 * scale).round().max(1.0) as u32,
        (height as f64 * scale).round().max(1.0) as u32,
    )
}

fn preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    bounded_preview_dimensions(width, height, PREVIEW_MAX_LONG_SIDE, PREVIEW_MAX_PIXELS)
}

fn detail_preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    bounded_preview_dimensions(
        width,
        height,
        DETAIL_PREVIEW_MAX_LONG_SIDE,
        DETAIL_PREVIEW_MAX_PIXELS,
    )
}

/// Image stacking works on display-referred samples after RAW development. Freeze
/// those samples once at 16-bit precision so preview and export cannot diverge by
/// independently interpreting a floating-point TIFF as linear RGB.
pub(crate) fn canonicalize_image_stack_result(image: DynamicImage) -> DynamicImage {
    DynamicImage::ImageRgb16(image.to_rgb16())
}

fn encode_srgb_image_stack<W: Write + Seek>(
    image: &DynamicImage,
    writer: W,
    output_format: ImageStackOutputFormat,
    jpeg_quality: u8,
    embed_color_profile: bool,
    export_exif: Option<&[u8]>,
) -> Result<(), String> {
    let profile = crate::color_management::srgb_v4_profile().to_vec();
    match output_format {
        ImageStackOutputFormat::Tiff => {
            let mut encoder = TiffEncoder::new(writer);
            if embed_color_profile {
                encoder.set_icc_profile(profile).map_err(|error| {
                    format!("Failed to attach the image-stack TIFF color profile: {error}")
                })?;
            }
            image
                .write_with_encoder(encoder)
                .map_err(|error| format!("Failed to encode image-stack TIFF: {error}"))
        }
        ImageStackOutputFormat::Png => {
            let mut encoder = PngEncoder::new(writer);
            if embed_color_profile {
                encoder.set_icc_profile(profile).map_err(|error| {
                    format!("Failed to attach the image-stack PNG color profile: {error}")
                })?;
            }
            if let Some(exif) = export_exif {
                encoder.set_exif_metadata(exif.to_vec()).map_err(|error| {
                    format!("Failed to attach image-stack PNG metadata: {error}")
                })?;
            }
            image
                .write_with_encoder(encoder)
                .map_err(|error| format!("Failed to encode image-stack PNG: {error}"))
        }
        ImageStackOutputFormat::Jpeg => {
            let mut encoder = JpegEncoder::new_with_quality(writer, jpeg_quality.clamp(1, 100));
            if embed_color_profile {
                encoder.set_icc_profile(profile).map_err(|error| {
                    format!("Failed to attach the image-stack JPEG color profile: {error}")
                })?;
            }
            if let Some(exif) = export_exif {
                encoder.set_exif_metadata(exif.to_vec()).map_err(|error| {
                    format!("Failed to attach image-stack JPEG metadata: {error}")
                })?;
            }
            encoder
                .encode_image(&image.to_rgb8())
                .map_err(|error| format!("Failed to encode image-stack JPEG: {error}"))
        }
    }
}

#[cfg(test)]
fn encode_srgb_tiff<W: Write + Seek>(image: &DynamicImage, writer: W) -> Result<(), String> {
    encode_srgb_image_stack(image, writer, ImageStackOutputFormat::Tiff, 95, true, None)
}

#[cfg(not(target_os = "android"))]
fn encode_srgb_jpeg_streaming(
    image: &DynamicImage,
    output: &mut File,
    jpeg_quality: u8,
    embed_color_profile: bool,
    export_exif: Option<&[u8]>,
) -> Result<(), String> {
    let rgb16 = image
        .as_rgb16()
        .ok_or_else(|| "The canonical image-stack result is not RGB16.".to_string())?;
    let (width, height) = rgb16.dimensions();
    let source_row_samples = (width as usize)
        .checked_mul(3)
        .ok_or_else(|| "The image-stack JPEG row is too wide to encode.".to_string())?;
    let rgba_row_bytes = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| "The image-stack JPEG row is too wide to encode.".to_string())?;
    let mut rgba_row = vec![0_u8; rgba_row_bytes];

    crate::export_processing::encode_streaming_jpeg(
        output,
        width,
        height,
        jpeg_quality,
        embed_color_profile,
        export_exif,
        |sink| {
            for source_row in rgb16.as_raw().chunks_exact(source_row_samples) {
                for (source, target) in source_row.chunks_exact(3).zip(rgba_row.chunks_exact_mut(4))
                {
                    target[0] = ((u32::from(source[0]) + 128) / 257) as u8;
                    target[1] = ((u32::from(source[1]) + 128) / 257) as u8;
                    target[2] = ((u32::from(source[2]) + 128) / 257) as u8;
                    target[3] = 255;
                }
                sink(&rgba_row)?;
            }
            Ok(())
        },
    )
}

fn encode_image_stack_file(
    image: &DynamicImage,
    output: &mut File,
    output_format: ImageStackOutputFormat,
    export_settings: &ExportSettings,
    source_path: &str,
) -> Result<(), String> {
    let export_exif = if output_format == ImageStackOutputFormat::Tiff {
        None
    } else {
        crate::exif_processing::export_metadata_tiff_payload(
            source_path,
            output_format.canonical_extension(),
            export_settings.keep_metadata,
            export_settings.strip_gps,
            export_settings.metadata_overrides.as_ref(),
        )?
    };

    #[cfg(not(target_os = "android"))]
    if output_format == ImageStackOutputFormat::Jpeg {
        encode_srgb_jpeg_streaming(
            image,
            output,
            export_settings.jpeg_quality,
            export_settings.embed_color_profile,
            export_exif.as_deref(),
        )?;
        return output
            .flush()
            .map_err(|error| format!("Failed to finish writing the image-stack JPEG: {error}"));
    }

    #[cfg(not(target_os = "android"))]
    if output_format == ImageStackOutputFormat::Tiff {
        let metadata = crate::exif_processing::export_metadata_for_streaming_tiff(
            source_path,
            export_settings.keep_metadata,
            export_settings.strip_gps,
            export_settings.metadata_overrides.as_ref(),
        )?;
        let rgb16 = image
            .as_rgb16()
            .ok_or_else(|| "The canonical image-stack result is not RGB16.".to_string())?;
        return crate::export_processing::encode_rgb16_tiff_with_metadata(
            output,
            rgb16.width(),
            rgb16.height(),
            rgb16.as_raw(),
            export_settings.embed_color_profile,
            metadata.as_ref(),
        );
    }

    let mut writer = BufWriter::new(output);
    encode_srgb_image_stack(
        image,
        &mut writer,
        output_format,
        export_settings.jpeg_quality,
        export_settings.embed_color_profile,
        export_exif.as_deref(),
    )?;
    writer.flush().map_err(|error| {
        format!(
            "Failed to finish writing the image-stack {}: {error}",
            output_format.label()
        )
    })
}

fn validate_image_stack_decoder<D: ImageDecoder>(
    mut decoder: D,
    dimensions: (u32, u32),
    output_format: ImageStackOutputFormat,
    expect_color_profile: bool,
) -> Result<(), String> {
    if decoder.dimensions() != dimensions {
        return Err(format!(
            "Saved image-stack dimensions do not match the result (expected {}×{}, found {}×{}).",
            dimensions.0,
            dimensions.1,
            decoder.dimensions().0,
            decoder.dimensions().1
        ));
    }
    if decoder.color_type() != output_format.encoded_color_type() {
        return Err(format!(
            "Saved image-stack {} has an unexpected pixel format ({:?}).",
            output_format.label(),
            decoder.color_type()
        ));
    }
    let saved_profile = decoder.icc_profile().map_err(|error| {
        format!(
            "Failed to validate the image-stack {} color profile: {error}",
            output_format.label()
        )
    })?;
    if expect_color_profile {
        if saved_profile.as_deref() != Some(crate::color_management::srgb_v4_profile()) {
            return Err(format!(
                "The saved image-stack {} is missing its sRGB color profile.",
                output_format.label()
            ));
        }
    } else if saved_profile.is_some() {
        return Err(format!(
            "The saved image-stack {} contains a color profile even though embedding was disabled.",
            output_format.label()
        ));
    }
    Ok(())
}

fn validate_image_stack_output(
    output_path: &Path,
    dimensions: (u32, u32),
    output_format: ImageStackOutputFormat,
    expect_color_profile: bool,
) -> Result<(), String> {
    let open = || {
        File::open(output_path)
            .map(BufReader::new)
            .map_err(|error| {
                format!(
                    "Failed to reopen the saved image-stack {}: {error}",
                    output_format.label()
                )
            })
    };
    match output_format {
        ImageStackOutputFormat::Tiff => validate_image_stack_decoder(
            TiffDecoder::new(open()?).map_err(|error| {
                format!("Failed to validate the saved image-stack TIFF: {error}")
            })?,
            dimensions,
            output_format,
            expect_color_profile,
        ),
        ImageStackOutputFormat::Png => validate_image_stack_decoder(
            PngDecoder::new(open()?).map_err(|error| {
                format!("Failed to validate the saved image-stack PNG: {error}")
            })?,
            dimensions,
            output_format,
            expect_color_profile,
        ),
        ImageStackOutputFormat::Jpeg => validate_image_stack_decoder(
            JpegDecoder::new(open()?).map_err(|error| {
                format!("Failed to validate the saved image-stack JPEG: {error}")
            })?,
            dimensions,
            output_format,
            expect_color_profile,
        ),
    }
}

#[cfg(test)]
fn default_image_stack_export_settings() -> ExportSettings {
    ExportSettings {
        jpeg_quality: IMAGE_STACK_JPEG_QUALITY,
        resize: None,
        keep_metadata: false,
        metadata_overrides: None,
        preserve_timestamps: false,
        strip_gps: true,
        embed_color_profile: true,
        filename_template: None,
        watermark: None,
        export_masks: false,
        preserve_folders: false,
    }
}

fn write_image_stack_output_with_settings(
    image: &DynamicImage,
    output_path: &Path,
    output_format: ImageStackOutputFormat,
    export_settings: &ExportSettings,
    source_path: &str,
) -> Result<(), String> {
    let transformed_image =
        if export_settings.resize.is_some() || export_settings.watermark.is_some() {
            Some(crate::export_processing::apply_export_resize_and_watermark(
                image.clone(),
                export_settings,
            )?)
        } else {
            None
        };
    let image = transformed_image.as_ref().unwrap_or(image);
    let dimensions = image.dimensions();
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return Err("The image-stack result is empty and cannot be saved.".to_string());
    }

    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "Failed to create the image-stack output folder '{}': {error}",
            parent.display()
        )
    })?;

    let mut temporary = NamedTempFile::new_in(parent).map_err(|error| {
        format!(
            "Failed to create a temporary image-stack file beside '{}': {error}",
            output_path.display()
        )
    })?;
    encode_image_stack_file(
        image,
        temporary.as_file_mut(),
        output_format,
        export_settings,
        source_path,
    )?;
    temporary.as_file().sync_all().map_err(|error| {
        format!(
            "Failed to sync the image-stack {} to disk: {error}",
            output_format.label()
        )
    })?;

    let temporary_size = temporary
        .as_file()
        .metadata()
        .map_err(|error| {
            format!(
                "Failed to inspect the saved image-stack {}: {error}",
                output_format.label()
            )
        })?
        .len();
    if temporary_size == 0 {
        return Err(format!(
            "The encoded image-stack {} is empty.",
            output_format.label()
        ));
    }

    validate_image_stack_output(
        temporary.path(),
        dimensions,
        output_format,
        export_settings.embed_color_profile,
    )?;

    let persisted = temporary.persist(output_path).map_err(|error| {
        format!(
            "Failed to finalize the image-stack {} '{}': {}",
            output_format.label(),
            output_path.display(),
            error.error
        )
    })?;
    persisted.sync_all().map_err(|error| {
        format!(
            "Failed to finalize the image-stack {} on disk: {error}",
            output_format.label()
        )
    })?;

    #[cfg(unix)]
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("Failed to sync the image-stack output folder: {error}"))?;

    Ok(())
}

#[cfg(test)]
fn write_image_stack_output(
    image: &DynamicImage,
    output_path: &Path,
    output_format: ImageStackOutputFormat,
) -> Result<(), String> {
    write_image_stack_output_with_settings(
        image,
        output_path,
        output_format,
        &default_image_stack_export_settings(),
        "",
    )
}

#[cfg(test)]
pub(crate) fn write_srgb_tiff(image: &DynamicImage, output_path: &Path) -> Result<(), String> {
    write_image_stack_output(image, output_path, ImageStackOutputFormat::Tiff)
}

#[cfg(test)]
pub(crate) fn write_srgb_jpeg(image: &DynamicImage, output_path: &Path) -> Result<(), String> {
    write_image_stack_output(image, output_path, ImageStackOutputFormat::Jpeg)
}

fn resolve_image_stack_output_path(
    first_path: &Path,
    blend_mode: &str,
    output_path_str: Option<&str>,
    output_format: ImageStackOutputFormat,
) -> Result<PathBuf, String> {
    if let Some(requested_path) = output_path_str.filter(|path| !path.trim().is_empty()) {
        let mut output_path = PathBuf::from(requested_path);
        let path_format = output_path
            .extension()
            .and_then(|value| value.to_str())
            .and_then(ImageStackOutputFormat::from_extension);
        if path_format != Some(output_format) {
            output_path.set_extension(output_format.canonical_extension());
        }
        return Ok(output_path);
    }

    let parent = first_path
        .parent()
        .ok_or_else(|| "Could not determine the source image folder.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("image");
    let suffix = if blend_mode == "focus" {
        "FocusStack"
    } else {
        "Panorama"
    };
    Ok(parent.join(format!(
        "{stem}_{suffix}.{}",
        output_format.canonical_extension()
    )))
}

struct PreviewFiles {
    detail_height: u32,
    detail_path: String,
    detail_width: u32,
    interaction_height: u32,
    interaction_path: String,
    interaction_width: u32,
}

fn encode_preview_jpeg(image: &RgbImage, path: &Path, quality: u8) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|error| format!("Failed to create image-stack preview: {error}"))?;
    let mut writer = BufWriter::new(file);
    let mut encoder = JpegEncoder::new_with_quality(&mut writer, quality);
    encoder
        .set_icc_profile(crate::color_management::srgb_v4_profile().to_vec())
        .map_err(|error| {
            format!("Failed to attach the image-stack preview color profile: {error}")
        })?;
    encoder
        .encode_image(image)
        .map_err(|error| format!("Failed to encode image-stack preview: {error}"))?;
    drop(encoder);
    writer
        .flush()
        .map_err(|error| format!("Failed to finish image-stack preview: {error}"))
}

fn write_preview_files(
    image: &DynamicImage,
    app_handle: &AppHandle,
) -> Result<PreviewFiles, String> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err("The image result is empty.".to_string());
    }
    let (interaction_width, interaction_height) = preview_dimensions(width, height);
    let (detail_width, detail_height) = detail_preview_dimensions(width, height);
    let full_rgb = image.to_rgb8();
    let detail_rgb = if (detail_width, detail_height) == (width, height) {
        full_rgb
    } else {
        image::imageops::resize(&full_rgb, detail_width, detail_height, FilterType::Lanczos3)
    };

    let preview_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|error| format!("Failed to resolve image-stack preview cache: {error}"))?
        .join("image-stack-previews");
    fs::create_dir_all(&preview_dir)
        .map_err(|error| format!("Failed to create image-stack preview cache: {error}"))?;
    let preview_id = Uuid::new_v4();
    let detail_path = preview_dir.join(format!("{preview_id}-detail.jpg"));
    encode_preview_jpeg(&detail_rgb, &detail_path, DETAIL_PREVIEW_JPEG_QUALITY)?;

    let interaction_path =
        if (interaction_width, interaction_height) == (detail_width, detail_height) {
            detail_path.clone()
        } else {
            let interaction_rgb = image::imageops::resize(
                &detail_rgb,
                interaction_width,
                interaction_height,
                FilterType::Lanczos3,
            );
            let path = preview_dir.join(format!("{preview_id}-interaction.jpg"));
            encode_preview_jpeg(&interaction_rgb, &path, PREVIEW_JPEG_QUALITY)?;
            path
        };

    if let Ok(entries) = fs::read_dir(&preview_dir) {
        for entry in entries.flatten() {
            let stale_path = entry.path();
            if stale_path != interaction_path && stale_path != detail_path && stale_path.is_file() {
                let _ = fs::remove_file(stale_path);
            }
        }
    }

    Ok(PreviewFiles {
        detail_height,
        detail_path: detail_path.to_string_lossy().into_owned(),
        detail_width,
        interaction_height,
        interaction_path: interaction_path.to_string_lossy().into_owned(),
        interaction_width,
    })
}

#[tauri::command]
pub async fn process_image_stack(
    paths: Vec<String>,
    blend_mode: String,
    alignment_mode: String,
    pipeline_version: String,
    request_id: String,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    validate_image_stack_source_count(paths.len())?;
    if request_id.trim().is_empty() {
        return Err("Image-stack request ID is missing.".to_string());
    }
    validate_image_stack_pipeline_version(&pipeline_version)?;

    let source_paths: Vec<String> = paths
        .iter()
        .map(|path| parse_virtual_path(path).0.to_string_lossy().into_owned())
        .collect();
    let selected_blend_mode = resolve_blend_mode(&blend_mode);
    let selected_alignment_mode = AlignmentMode::from_wire(&alignment_mode);
    let result_handle = state.image_stack_result.clone();
    let generation_handle = state.image_stack_generation.clone();
    let generation = generation_handle.fetch_add(1, Ordering::SeqCst) + 1;
    *result_handle.lock().unwrap() = None;

    let task = tokio::task::spawn_blocking(move || {
        let result = stitch_images_with_options(
            source_paths,
            app_handle.clone(),
            selected_alignment_mode,
            selected_blend_mode,
            "image-stack-progress",
        );

        match result {
            Ok(outcome) => {
                if generation_handle.load(Ordering::SeqCst) != generation {
                    return Ok(());
                }

                let full_canvas_width = outcome.full_canvas_width;
                let full_canvas_height = outcome.full_canvas_height;
                let render_scale = outcome.render_scale;
                let image = canonicalize_image_stack_result(outcome.image);
                let _ = app_handle.emit("image-stack-progress", "Creating preview…");
                let previews = write_preview_files(&image, &app_handle)?;
                if generation_handle.load(Ordering::SeqCst) != generation {
                    let _ = fs::remove_file(&previews.interaction_path);
                    let _ = fs::remove_file(&previews.detail_path);
                    return Ok(());
                }

                let result_id = Uuid::new_v4().to_string();
                let (source_width, source_height) = image.dimensions();
                {
                    let mut stored_result = result_handle.lock().unwrap();
                    if generation_handle.load(Ordering::SeqCst) != generation {
                        let _ = fs::remove_file(&previews.interaction_path);
                        let _ = fs::remove_file(&previews.detail_path);
                        return Ok(());
                    }
                    *stored_result = Some((result_id.clone(), image));
                }
                let _ = app_handle.emit(
                    "image-stack-complete",
                    serde_json::json!({
                        "previewPath": previews.interaction_path,
                        "previewWidth": previews.interaction_width,
                        "previewHeight": previews.interaction_height,
                        "detailPreviewPath": previews.detail_path,
                        "detailPreviewWidth": previews.detail_width,
                        "detailPreviewHeight": previews.detail_height,
                        "sourceWidth": source_width,
                        "sourceHeight": source_height,
                        "fullCanvasWidth": full_canvas_width,
                        "fullCanvasHeight": full_canvas_height,
                        "renderScale": render_scale,
                        "pipelineVersion": IMAGE_STACK_PIPELINE_VERSION,
                        "requestId": request_id,
                        "resultId": result_id,
                    }),
                );
                Ok(())
            }
            Err(error) => {
                if generation_handle.load(Ordering::SeqCst) == generation {
                    let _ = app_handle.emit(
                        "image-stack-error",
                        serde_json::json!({
                            "message": error,
                            "requestId": request_id,
                        }),
                    );
                }
                Err(error)
            }
        }
    });

    match task.await {
        Ok(result) => result,
        Err(error) => Err(format!("Image stack task failed: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::io::{BufReader, Cursor};
    use std::path::Path;

    use image::codecs::{jpeg::JpegDecoder, tiff::TiffDecoder};
    use image::{DynamicImage, GenericImageView, ImageBuffer, ImageDecoder, Rgb, Rgb32FImage};

    use super::{
        DETAIL_PREVIEW_MAX_LONG_SIDE, DETAIL_PREVIEW_MAX_PIXELS, IMAGE_STACK_MAX_SOURCES,
        IMAGE_STACK_PIPELINE_VERSION, ImageStackOutputFormat, PREVIEW_MAX_LONG_SIDE,
        PREVIEW_MAX_PIXELS, canonicalize_image_stack_result, detail_preview_dimensions,
        encode_srgb_tiff, preview_dimensions, resolve_image_stack_output_path,
        validate_image_stack_pipeline_version, validate_image_stack_source_count,
        write_image_stack_output, write_image_stack_output_with_settings, write_srgb_tiff,
    };

    #[test]
    fn image_stack_pipeline_version_rejects_stale_frontends() {
        assert!(validate_image_stack_pipeline_version(IMAGE_STACK_PIPELINE_VERSION).is_ok());
        assert!(validate_image_stack_pipeline_version("").is_err());
        assert!(validate_image_stack_pipeline_version("image-stack-legacy").is_err());
    }

    #[test]
    fn image_stack_accepts_two_hundred_sources_but_rejects_more() {
        assert!(validate_image_stack_source_count(2).is_ok());
        assert!(validate_image_stack_source_count(IMAGE_STACK_MAX_SOURCES).is_ok());
        assert!(validate_image_stack_source_count(1).is_err());
        assert!(validate_image_stack_source_count(IMAGE_STACK_MAX_SOURCES + 1).is_err());
    }

    #[test]
    fn preview_dimensions_keep_images_within_the_quality_budget() {
        assert_eq!(preview_dimensions(4_000, 3_000), (4_000, 3_000));

        let (wide_width, wide_height) = preview_dimensions(20_000, 2_000);
        assert_eq!(wide_width, PREVIEW_MAX_LONG_SIDE);
        assert_eq!(wide_height, 410);

        let (large_width, large_height) = preview_dimensions(8_256, 5_504);
        assert!(large_width <= PREVIEW_MAX_LONG_SIDE);
        assert!(large_width as u64 * large_height as u64 <= PREVIEW_MAX_PIXELS + 10_000);

        let (portrait_width, portrait_height) = preview_dimensions(9_305, 12_618);
        assert!(portrait_height <= PREVIEW_MAX_LONG_SIDE);
        assert!(portrait_width as u64 * portrait_height as u64 <= PREVIEW_MAX_PIXELS + 10_000);

        let (detail_width, detail_height) = detail_preview_dimensions(9_305, 12_618);
        assert!(detail_height <= DETAIL_PREVIEW_MAX_LONG_SIDE);
        assert!(detail_width as u64 * detail_height as u64 <= DETAIL_PREVIEW_MAX_PIXELS + 10_000);
        assert!(detail_width > portrait_width);
        assert!(detail_height > portrait_height);
    }

    #[test]
    fn canonical_stack_result_is_display_encoded_rgb16() {
        let source =
            DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(2, 1, Rgb([0.25, 0.5, 0.75])));
        let canonical = canonicalize_image_stack_result(source);
        let pixels = canonical.as_rgb16().expect("canonical RGB16 result");

        assert_eq!(pixels.dimensions(), (2, 1));
        assert!((pixels.get_pixel(0, 0)[0] as i32 - 16_384).abs() <= 1);
        assert!((pixels.get_pixel(0, 0)[1] as i32 - 32_768).abs() <= 1);
        assert!((pixels.get_pixel(0, 0)[2] as i32 - 49_151).abs() <= 1);
    }

    #[test]
    fn exported_tiff_preserves_canonical_pixels_and_srgb_profile() {
        let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(3, 2, |x, y| {
            Rgb([
                (x as f32 + 1.0) / 4.0,
                (y as f32 + 1.0) / 3.0,
                (x as f32 + y as f32 + 1.0) / 6.0,
            ])
        }));
        let canonical = canonicalize_image_stack_result(source);
        let expected = canonical.to_rgb16();
        let mut encoded = Cursor::new(Vec::new());
        encode_srgb_tiff(&canonical, &mut encoded).expect("encode canonical TIFF");

        encoded.set_position(0);
        let mut metadata_decoder = TiffDecoder::new(&mut encoded).expect("decode TIFF metadata");
        let profile = metadata_decoder
            .icc_profile()
            .expect("read TIFF ICC")
            .expect("embedded TIFF ICC");
        assert_eq!(profile, crate::color_management::srgb_v4_profile());

        encoded.set_position(0);
        let decoded = DynamicImage::from_decoder(
            TiffDecoder::new(&mut encoded).expect("decode canonical TIFF"),
        )
        .expect("read canonical TIFF pixels")
        .to_rgb16();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn atomic_tiff_save_replaces_stale_output_with_a_valid_image() {
        let directory = tempfile::tempdir().expect("temporary image-stack output directory");
        let output_path = directory.path().join("stack-result.tiff");
        fs::write(&output_path, b"stale").expect("seed stale output");
        let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(4, 3, |x, y| {
            Rgb([
                (x as f32 + 1.0) / 5.0,
                (y as f32 + 1.0) / 4.0,
                (x as f32 + y as f32 + 1.0) / 8.0,
            ])
        }));
        let canonical = canonicalize_image_stack_result(source);

        write_srgb_tiff(&canonical, &output_path).expect("atomic image-stack TIFF save");
        assert!(
            fs::metadata(&output_path)
                .expect("saved TIFF metadata")
                .len()
                > 5
        );

        let mut decoder = TiffDecoder::new(BufReader::new(
            File::open(&output_path).expect("open persisted image-stack TIFF"),
        ))
        .expect("decode persisted image-stack TIFF");
        assert_eq!(decoder.dimensions(), (4, 3));
        assert_eq!(
            decoder
                .icc_profile()
                .expect("read persisted TIFF ICC")
                .as_deref(),
            Some(crate::color_management::srgb_v4_profile())
        );
    }

    #[test]
    fn save_path_prefers_the_requested_destination_and_normalizes_the_extension() {
        let source_path = Path::new("/photos/source.jpg");
        assert_eq!(
            resolve_image_stack_output_path(
                source_path,
                "focus",
                Some("/exports/custom-stack"),
                ImageStackOutputFormat::Png,
            )
            .expect("explicit PNG save destination"),
            Path::new("/exports/custom-stack.png")
        );
        assert_eq!(
            resolve_image_stack_output_path(
                source_path,
                "focus",
                Some("/exports/custom-stack.jpeg"),
                ImageStackOutputFormat::Jpeg,
            )
            .expect("explicit JPEG save destination"),
            Path::new("/exports/custom-stack.jpeg")
        );
        assert_eq!(
            resolve_image_stack_output_path(
                source_path,
                "focus",
                Some("/exports/custom-stack.png"),
                ImageStackOutputFormat::Tiff,
            )
            .expect("selected format overrides a mismatched extension"),
            Path::new("/exports/custom-stack.tiff")
        );
        assert_eq!(
            resolve_image_stack_output_path(
                source_path,
                "focus",
                None,
                ImageStackOutputFormat::Tiff,
            )
            .expect("fallback TIFF save destination"),
            Path::new("/photos/source_FocusStack.tiff")
        );
    }

    #[test]
    fn output_format_accepts_supported_aliases_and_rejects_other_values() {
        assert_eq!(
            ImageStackOutputFormat::from_wire("tif"),
            Ok(ImageStackOutputFormat::Tiff)
        );
        assert_eq!(
            ImageStackOutputFormat::from_wire("JPEG"),
            Ok(ImageStackOutputFormat::Jpeg)
        );
        assert!(ImageStackOutputFormat::from_wire("webp").is_err());
    }

    #[test]
    fn atomic_stack_save_supports_tiff_png_and_jpeg() {
        let directory = tempfile::tempdir().expect("temporary image-stack output directory");
        let source = DynamicImage::ImageRgb16(ImageBuffer::from_fn(8, 5, |x, y| {
            Rgb([
                ((x + 1) * 4_096) as u16,
                ((y + 1) * 8_192) as u16,
                ((x + y + 1) * 2_048) as u16,
            ])
        }));
        let expected_lossless_pixels = source.to_rgb16();

        for (format, extension) in [
            (ImageStackOutputFormat::Tiff, "tiff"),
            (ImageStackOutputFormat::Png, "png"),
            (ImageStackOutputFormat::Jpeg, "jpg"),
        ] {
            let output_path = directory.path().join(format!("stack-result.{extension}"));
            fs::write(&output_path, b"stale").expect("seed stale output");

            write_image_stack_output(&source, &output_path, format)
                .expect("atomic multi-format image-stack save");

            assert!(
                fs::metadata(&output_path)
                    .expect("saved output metadata")
                    .len()
                    > 5
            );
            let decoded = image::open(&output_path).expect("decode saved multi-format output");
            assert_eq!(decoded.dimensions(), source.dimensions());
            if format != ImageStackOutputFormat::Jpeg {
                assert_eq!(decoded.to_rgb16(), expected_lossless_pixels);
            }
        }
    }

    #[test]
    fn shared_export_settings_resize_stack_output_and_write_selected_metadata() {
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

        let directory = tempfile::tempdir().expect("temporary shared export directory");
        let output_path = directory.path().join("stack-settings.jpg");
        let source = DynamicImage::ImageRgb16(ImageBuffer::from_fn(8, 5, |x, y| {
            Rgb([
                ((x + 1) * 4_096) as u16,
                ((y + 1) * 8_192) as u16,
                ((x + y + 1) * 2_048) as u16,
            ])
        }));
        let settings = crate::export_processing::ExportSettings {
            jpeg_quality: 37,
            resize: Some(crate::export_processing::ResizeOptions {
                mode: crate::export_processing::ResizeMode::Width,
                value: 4,
                dont_enlarge: false,
            }),
            keep_metadata: false,
            metadata_overrides: Some(crate::exif_processing::ExportMetadataOverrides {
                artist: Some("Museum Imaging Team".to_string()),
                contact: Some("archive@example.test".to_string()),
                copyright: Some("Copyright 2026".to_string()),
                description: Some("Focus-stacked artifact".to_string()),
            }),
            preserve_timestamps: false,
            strip_gps: true,
            embed_color_profile: false,
            filename_template: None,
            watermark: None,
            export_masks: false,
            preserve_folders: false,
        };

        write_image_stack_output_with_settings(
            &source,
            &output_path,
            ImageStackOutputFormat::Jpeg,
            &settings,
            "/missing/source.jpg",
        )
        .expect("save stack through shared export settings");

        let encoded = fs::read(&output_path).expect("read shared export output");
        let mut decoder =
            JpegDecoder::new(Cursor::new(&encoded)).expect("decode resized stack JPEG");
        assert_eq!(decoder.dimensions(), (4, 3));
        assert_eq!(
            decoder.icc_profile().expect("read optional ICC profile"),
            None
        );

        let exif_data = exif::Reader::new()
            .read_from_container(&mut Cursor::new(&encoded))
            .expect("read selected stack metadata");
        assert_eq!(
            ascii_tag(&exif_data, exif::Tag::Artist).as_deref(),
            Some("Museum Imaging Team")
        );
        assert_eq!(
            ascii_tag(&exif_data, exif::Tag::Copyright).as_deref(),
            Some("Copyright 2026")
        );
        assert_eq!(
            ascii_tag(&exif_data, exif::Tag::ImageDescription).as_deref(),
            Some("Focus-stacked artifact")
        );
        let contact = exif_data
            .get_field(exif::Tag::UserComment, exif::In::PRIMARY)
            .and_then(|field| match &field.value {
                exif::Value::Undefined(value, _) => {
                    Some(String::from_utf8_lossy(value).to_string())
                }
                _ => None,
            });
        assert_eq!(contact.as_deref(), Some("Contact: archive@example.test"));
    }

    #[test]
    #[ignore = "requires RAW_EDITOR_STACK_EXPORT_SOURCE and RAW_EDITOR_STACK_EXPORT_OUTPUT_DIR"]
    fn real_stack_export_fixture_from_env() {
        let source_path = std::env::var_os("RAW_EDITOR_STACK_EXPORT_SOURCE")
            .map(std::path::PathBuf::from)
            .expect("RAW_EDITOR_STACK_EXPORT_SOURCE must point to a rendered stack image");
        let output_dir = std::env::var_os("RAW_EDITOR_STACK_EXPORT_OUTPUT_DIR")
            .map(std::path::PathBuf::from)
            .expect("RAW_EDITOR_STACK_EXPORT_OUTPUT_DIR must point to a writable directory");
        fs::create_dir_all(&output_dir).expect("create real image-stack export output directory");

        let mut reader = image::ImageReader::open(&source_path)
            .expect("open real image-stack export fixture")
            .with_guessed_format()
            .expect("detect real image-stack export fixture format");
        reader.no_limits();
        let decoded = reader
            .decode()
            .expect("decode real image-stack export fixture");
        let canonical = if decoded.as_rgb16().is_some() {
            decoded
        } else {
            canonicalize_image_stack_result(decoded)
        };

        for (format, extension) in [
            (ImageStackOutputFormat::Tiff, "tiff"),
            (ImageStackOutputFormat::Png, "png"),
            (ImageStackOutputFormat::Jpeg, "jpg"),
        ] {
            let output_path = output_dir.join(format!("real-stack-export.{extension}"));
            write_image_stack_output(&canonical, &output_path, format)
                .expect("export real image-stack fixture");
            println!("{}", output_path.display());
        }
    }
}

#[tauri::command]
pub async fn save_image_stack(
    first_path_str: String,
    blend_mode: String,
    output_format: String,
    export_settings: ExportSettings,
    result_id: String,
    output_path_str: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let (first_path, _) = parse_virtual_path(&first_path_str);
    let output_format = ImageStackOutputFormat::from_wire(&output_format)?;
    let output_path = resolve_image_stack_output_path(
        &first_path,
        &blend_mode,
        output_path_str.as_deref(),
        output_format,
    )?;
    let output_path_for_task = output_path.clone();
    let sidecar_source = first_path.to_string_lossy().into_owned();
    let result_handle = state.image_stack_result.clone();

    tokio::task::spawn_blocking(move || {
        let mut result = result_handle
            .lock()
            .map_err(|_| "The image-stack result store is unavailable.".to_string())?;
        let (stored_result_id, image) = result
            .as_ref()
            .ok_or_else(|| "No image-stack result is available to save.".to_string())?;
        if stored_result_id != &result_id {
            return Err(
                "The visible image-stack preview is no longer the current result. Please realign before saving."
                    .to_string(),
            );
        }

        write_image_stack_output_with_settings(
            image,
            &output_path_for_task,
            output_format,
            &export_settings,
            &sidecar_source,
        )?;
        *result = None;
        drop(result);
        // The encoded file is the export contract. Copying the source sidecar here would
        // silently restore EXIF/GPS fields that the user explicitly removed in the dialog.
        Ok(output_path_for_task.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| format!("Image-stack save task failed: {error}"))?
}
