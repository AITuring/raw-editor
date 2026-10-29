//! Local, non-destructive colour transfer for the Style Lab.
//!
//! The browser workbench renders a compact preview for immediate feedback.
//! This command repeats the same HSV/statistical transform against the full
//! target image so exporting does not silently reduce a large photograph to
//! the preview canvas size.

use std::fs;
use std::path::{Path, PathBuf};

use mozjpeg_rs::Preset;
use serde::Deserialize;
use tauri::{AppHandle, Emitter, ipc::Response};

use crate::app_settings::load_settings;
use crate::color_management::srgb_preview_encoder;

mod color_transfer;

#[cfg(not(target_os = "android"))]
use color_transfer::apply_transform_rgb_row;
#[cfg(test)]
use color_transfer::{TransferTransform, hsv_to_rgb, rgb_to_hsv};
use color_transfer::{apply_transform, estimate_transform};

mod preview_source;

#[cfg(test)]
use preview_source::{PREVIEW_EDGE, downscale_preview_to_edge};
#[cfg(not(target_os = "android"))]
use preview_source::{decode_jpeg_preview, is_jpeg_path, jpeg_info, large_jpeg_info};
use preview_source::{downscale_preview, load_image, load_preview_for_path};

mod styled_output;

use styled_output::encode_output;
#[cfg(not(target_os = "android"))]
use styled_output::stream_jpeg_export;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StyleTransferMode {
    #[default]
    Mood,
    Distribution,
}

fn real_path(path: &str) -> PathBuf {
    path.rsplit_once("?vc=")
        .map(|(source, _)| PathBuf::from(source))
        .unwrap_or_else(|| PathBuf::from(path))
}

fn paths_are_equal(first: &Path, second: &Path) -> bool {
    if first == second {
        return true;
    }
    match (fs::canonicalize(first), fs::canonicalize(second)) {
        (Ok(first), Ok(second)) => first == second,
        _ => first
            .to_string_lossy()
            .eq_ignore_ascii_case(&second.to_string_lossy()),
    }
}

#[tauri::command]
pub async fn generate_style_transfer_preview(
    path: String,
    app_handle: AppHandle,
) -> Result<Response, String> {
    if path.trim().is_empty() {
        return Err("An image path is required.".to_string());
    }

    tokio::task::spawn_blocking(move || {
        let settings = load_settings(app_handle.clone()).unwrap_or_default();
        let preview = load_preview_for_path(&path, &settings)?;
        let (width, height) = preview.dimensions();
        let bytes = srgb_preview_encoder(Preset::BaselineFastest)
            .quality(88)
            .encode_rgb(preview.as_raw(), width, height)
            .map_err(|error| error.to_string())?;
        Ok(Response::new(bytes))
    })
    .await
    .map_err(|error| format!("Style transfer preview task failed: {error}"))?
}

#[tauri::command]
pub async fn export_style_transfer(
    reference_path: String,
    target_path: String,
    output_path: String,
    strength: f32,
    mode: Option<StyleTransferMode>,
    app_handle: AppHandle,
) -> Result<String, String> {
    if reference_path.trim().is_empty()
        || target_path.trim().is_empty()
        || output_path.trim().is_empty()
    {
        return Err("Reference, target, and output paths are required.".to_string());
    }

    let reference_real_path = real_path(&reference_path);
    let target_real_path = real_path(&target_path);
    let output_real_path = real_path(&output_path);
    if paths_are_equal(&reference_real_path, &output_real_path)
        || paths_are_equal(&target_real_path, &output_real_path)
    {
        return Err("Choose a new output path so both source images stay untouched.".to_string());
    }

    let mode = mode.unwrap_or_default();
    let strength = if strength.is_finite() {
        strength.clamp(0.0, 1.0)
    } else {
        1.0
    };

    #[cfg(not(target_os = "android"))]
    {
        let target_real_path = real_path(&target_path);
        let output_real_path = real_path(&output_path);
        let can_stream_target = large_jpeg_info(&target_real_path).is_some_and(|info| {
            matches!(
                info.orientation,
                rawler::Orientation::Normal | rawler::Orientation::Unknown
            ) && is_jpeg_path(&output_real_path)
        });
        if can_stream_target {
            let progress_handle = app_handle.clone();
            return tokio::task::spawn_blocking(move || {
                let settings = load_settings(app_handle.clone()).unwrap_or_default();
                stream_jpeg_export(
                    &reference_path,
                    &target_path,
                    &output_path,
                    strength,
                    mode,
                    &settings,
                    &progress_handle,
                )
            })
            .await
            .map_err(|error| format!("Style transfer task failed: {error}"))?;
        }
    }

    let progress_handle = app_handle.clone();
    tokio::task::spawn_blocking(move || {
        let settings = load_settings(app_handle.clone()).unwrap_or_default();
        let _ = progress_handle.emit("style-transfer-progress", "Loading reference image…");
        let reference = load_image(&reference_path, &settings)?;
        let reference_preview = downscale_preview(&reference);
        drop(reference);

        let _ = progress_handle.emit("style-transfer-progress", "Loading target image…");
        let target = load_image(&target_path, &settings)?;
        let target_preview = downscale_preview(&target);
        let transform = estimate_transform(&reference_preview, &target_preview);
        drop(target_preview);

        let _ = progress_handle.emit("style-transfer-progress", "Applying colour transfer…");
        let processed = apply_transform(target.to_rgb8(), transform, mode, strength);

        let output_real_path = real_path(&output_path);
        if let Some(parent) = output_real_path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        encode_output(&processed, &output_real_path)
            .map_err(|error| format!("Failed to encode styled image: {error}"))?;

        let _ = progress_handle.emit("style-transfer-progress", "Export complete");
        Ok(output_real_path.to_string_lossy().to_string())
    })
    .await
    .map_err(|error| format!("Style transfer task failed: {error}"))?
}

#[cfg(test)]
mod tests;
