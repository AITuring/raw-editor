//! UI preview files for a finished image stack.
//!
//! Bounded preview and detail-preview JPEGs written next to the full result.

use image::DynamicImage;
use image::GenericImageView;
use image::ImageEncoder;
use image::RgbImage;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use tauri::AppHandle;
use tauri::Manager;

// Keep the UI proxy comfortably inside desktop WebView image-decoding budgets.
// The full-resolution result remains in AppState and is used for TIFF export.
pub(super) const PREVIEW_MAX_LONG_SIDE: u32 = 4_096;

pub(super) const PREVIEW_MAX_PIXELS: u64 = 12_000_000;

pub(super) const PREVIEW_JPEG_QUALITY: u8 = 92;

pub(super) const DETAIL_PREVIEW_MAX_LONG_SIDE: u32 = 8_192;

pub(super) const DETAIL_PREVIEW_MAX_PIXELS: u64 = 32_000_000;

pub(super) const DETAIL_PREVIEW_JPEG_QUALITY: u8 = 98;

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

pub(super) fn preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    bounded_preview_dimensions(width, height, PREVIEW_MAX_LONG_SIDE, PREVIEW_MAX_PIXELS)
}

pub(super) fn detail_preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    bounded_preview_dimensions(
        width,
        height,
        DETAIL_PREVIEW_MAX_LONG_SIDE,
        DETAIL_PREVIEW_MAX_PIXELS,
    )
}

pub(super) struct PreviewFiles {
    pub(super) detail_height: u32,
    pub(super) detail_path: String,
    pub(super) detail_width: u32,
    pub(super) interaction_height: u32,
    pub(super) interaction_path: String,
    pub(super) interaction_width: u32,
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

/// The detail and interaction previews of a canonical stack result (需求 10.7):
/// both are downsampled from the result's own display-encoded pixels and from
/// nothing else, so a preview can never show a different image than the one
/// that is exported. `None` means the interaction preview is the detail one.
pub(super) fn derive_preview_images(
    image: &DynamicImage,
    (detail_width, detail_height): (u32, u32),
    (interaction_width, interaction_height): (u32, u32),
) -> (RgbImage, Option<RgbImage>) {
    let (width, height) = image.dimensions();
    let full_rgb = image.to_rgb8();
    let detail_rgb = if (detail_width, detail_height) == (width, height) {
        full_rgb
    } else {
        image::imageops::resize(&full_rgb, detail_width, detail_height, FilterType::Lanczos3)
    };
    let interaction_rgb =
        ((interaction_width, interaction_height) != (detail_width, detail_height)).then(|| {
            image::imageops::resize(
                &detail_rgb,
                interaction_width,
                interaction_height,
                FilterType::Lanczos3,
            )
        });
    (detail_rgb, interaction_rgb)
}

/// Preview file names carry the result identifier, so the stored result and
/// its previews are linked by one identifier (需求 10.7).
pub(super) fn preview_file_names(result_id: &str) -> (String, String) {
    (
        format!("{result_id}-detail.jpg"),
        format!("{result_id}-interaction.jpg"),
    )
}

pub(super) fn write_preview_files(
    image: &DynamicImage,
    result_id: &str,
    app_handle: &AppHandle,
) -> Result<PreviewFiles, String> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err("The image result is empty.".to_string());
    }
    let (interaction_width, interaction_height) = preview_dimensions(width, height);
    let (detail_width, detail_height) = detail_preview_dimensions(width, height);
    let (detail_rgb, interaction_rgb) = derive_preview_images(
        image,
        (detail_width, detail_height),
        (interaction_width, interaction_height),
    );
    let (detail_name, interaction_name) = preview_file_names(result_id);

    let preview_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|error| format!("Failed to resolve image-stack preview cache: {error}"))?
        .join("image-stack-previews");
    fs::create_dir_all(&preview_dir)
        .map_err(|error| format!("Failed to create image-stack preview cache: {error}"))?;
    let detail_path = preview_dir.join(detail_name);
    encode_preview_jpeg(&detail_rgb, &detail_path, DETAIL_PREVIEW_JPEG_QUALITY)?;

    let interaction_path = match interaction_rgb {
        None => detail_path.clone(),
        Some(interaction_rgb) => {
            let path = preview_dir.join(interaction_name);
            encode_preview_jpeg(&interaction_rgb, &path, PREVIEW_JPEG_QUALITY)?;
            path
        }
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
