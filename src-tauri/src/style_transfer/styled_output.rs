//! Writing the styled result.
//!
//! Encodes a full styled image, or streams a large desktop JPEG row by row
//! through the transform into a temporary file that is published atomically.

use crate::app_settings::AppSettings;
#[cfg(target_os = "android")]
use image::DynamicImage;
use image::ImageFormat;
use image::RgbImage;
#[cfg(target_os = "android")]
use image::codecs::jpeg::JpegEncoder;
#[cfg(not(target_os = "android"))]
use mozjpeg::Decompress;
use std::fs;
use std::fs::File;
#[cfg(target_os = "android")]
use std::io::BufWriter;
#[cfg(not(target_os = "android"))]
use std::io::Write;
use std::path::Path;
use tauri::AppHandle;
use tauri::Emitter;
#[cfg(not(target_os = "android"))]
use tempfile::NamedTempFile;

use super::{StyleTransferMode, estimate_transform, load_preview_for_path, real_path};
#[cfg(not(target_os = "android"))]
use super::{apply_transform_rgb_row, decode_jpeg_preview, jpeg_info};

pub(super) fn encode_output(image: &RgbImage, output_path: &Path) -> Result<(), String> {
    let format = ImageFormat::from_path(output_path).unwrap_or(ImageFormat::Jpeg);
    match format {
        ImageFormat::Jpeg => encode_jpeg_output(image, output_path)?,
        ImageFormat::Png => image
            .save_with_format(output_path, ImageFormat::Png)
            .map_err(|error| error.to_string())?,
        _ => image
            .save_with_format(output_path, format)
            .map_err(|error| error.to_string())?,
    }
    Ok(())
}

#[cfg(not(target_os = "android"))]
fn encode_jpeg_output(image: &RgbImage, output_path: &Path) -> Result<(), String> {
    let mut file = File::create(output_path).map_err(|error| error.to_string())?;
    let width = image.width();
    let height = image.height();
    let rgb_row_bytes = usize::try_from(width)
        .ok()
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "Styled JPEG row is too wide.".to_string())?;
    let rgba_row_bytes = usize::try_from(width)
        .ok()
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| "Styled JPEG output row is too wide.".to_string())?;
    let mut rgba_row = vec![0_u8; rgba_row_bytes];
    crate::export_processing::encode_streaming_jpeg(
        &mut file,
        width,
        height,
        94,
        true,
        None,
        |sink| {
            for rgb_row in image.as_raw().chunks_exact(rgb_row_bytes) {
                for (rgb, rgba) in rgb_row.chunks_exact(3).zip(rgba_row.chunks_exact_mut(4)) {
                    rgba[..3].copy_from_slice(rgb);
                    rgba[3] = 255;
                }
                sink(&rgba_row)?;
            }
            Ok(())
        },
    )?;
    file.flush()
        .map_err(|error| format!("Failed to flush styled JPEG: {error}"))
}

#[cfg(target_os = "android")]
fn encode_jpeg_output(image: &RgbImage, output_path: &Path) -> Result<(), String> {
    let file = File::create(output_path).map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(file);
    JpegEncoder::new_with_quality(&mut writer, 94)
        .encode_image(&DynamicImage::ImageRgb8(image.clone()))
        .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "android"))]
pub(super) fn stream_jpeg_export<R: tauri::Runtime>(
    reference_path: &str,
    target_path: &str,
    output_path: &str,
    strength: f32,
    mode: StyleTransferMode,
    settings: &AppSettings,
    progress_handle: &AppHandle<R>,
) -> Result<String, String> {
    let target_real_path = real_path(target_path);
    let target_info = jpeg_info(&target_real_path)
        .ok_or_else(|| "The target JPEG could not be inspected.".to_string())?;
    if !matches!(
        target_info.orientation,
        rawler::Orientation::Normal | rawler::Orientation::Unknown
    ) {
        return Err(
            "This JPEG uses an embedded rotation. Open it in the editor and export a copy first."
                .to_string(),
        );
    }

    let _ = progress_handle.emit("style-transfer-progress", "Preparing colour profile…");
    let reference_preview = load_preview_for_path(reference_path, settings)?;
    let target_preview = decode_jpeg_preview(&target_real_path, target_info)?;
    let transform = estimate_transform(&reference_preview, &target_preview);
    let output_real_path = real_path(output_path);
    if let Some(parent) = output_real_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }

    let output_parent = output_real_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = NamedTempFile::new_in(output_parent).map_err(|error| {
        format!(
            "Failed to create temporary styled image beside '{}': {error}",
            output_real_path.display()
        )
    })?;

    let _ = progress_handle.emit("style-transfer-progress", "Applying colour transfer…");
    crate::export_processing::encode_streaming_jpeg(
        temporary.as_file_mut(),
        target_info.width,
        target_info.height,
        94,
        true,
        None,
        |sink| {
            let decoder = Decompress::new_path(&target_real_path)
                .map_err(|error| format!("Failed to open target JPEG: {error}"))?;
            let mut started = decoder
                .rgb()
                .map_err(|error| format!("Failed to start target JPEG decode: {error}"))?;
            if (started.width(), started.height())
                != (target_info.width as usize, target_info.height as usize)
            {
                return Err("Target JPEG dimensions changed while exporting.".to_string());
            }

            let row_bytes = usize::try_from(target_info.width)
                .ok()
                .and_then(|value| value.checked_mul(3))
                .ok_or_else(|| "Target JPEG row is too wide.".to_string())?;
            let rgba_row_bytes = usize::try_from(target_info.width)
                .ok()
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| "Target JPEG output row is too wide.".to_string())?;
            let mut rgb_row = vec![0_u8; row_bytes];
            let mut rgba_row = vec![0_u8; rgba_row_bytes];

            for _ in 0..target_info.height {
                let decoded = started
                    .read_scanlines_into(&mut rgb_row)
                    .map_err(|error| format!("Failed to read target JPEG row: {error}"))?;
                if decoded.len() != row_bytes {
                    return Err("Target JPEG decoder returned an incomplete row.".to_string());
                }
                apply_transform_rgb_row(&mut rgb_row, transform, mode, strength);
                for (rgb, rgba) in rgb_row.chunks_exact(3).zip(rgba_row.chunks_exact_mut(4)) {
                    rgba[..3].copy_from_slice(rgb);
                    rgba[3] = 255;
                }
                sink(&rgba_row)?;
            }
            started
                .finish()
                .map_err(|error| format!("Failed to finish target JPEG decode: {error}"))?;
            Ok(())
        },
    )?;
    temporary
        .as_file_mut()
        .flush()
        .map_err(|error| format!("Failed to flush styled image: {error}"))?;
    if let Ok(metadata) = fs::metadata(&output_real_path) {
        temporary
            .as_file()
            .set_permissions(metadata.permissions())
            .map_err(|error| format!("Failed to preserve output permissions: {error}"))?;
    }
    let persisted_path = output_real_path.clone();
    temporary.persist(&output_real_path).map_err(|error| {
        format!(
            "Failed to publish styled image '{}': {}",
            persisted_path.display(),
            error.error
        )
    })?;
    let _ = progress_handle.emit("style-transfer-progress", "Export complete");
    Ok(output_real_path.to_string_lossy().to_string())
}
