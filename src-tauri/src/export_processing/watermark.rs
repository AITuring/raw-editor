//! Watermarks: library import, preparation and compositing.
//!
//! Imports a picked image into the watermark library and composites a
//! prepared watermark onto full images or streamed rows.

use image::DynamicImage;
use image::GenericImageView;
use image::ImageFormat;
#[cfg(not(target_os = "android"))]
use image::Pixel;
#[cfg(not(target_os = "android"))]
use image::Rgba;
use image::RgbaImage;
use image::imageops;
use serde::Deserialize;
use serde::Serialize;
use std::fs;
use std::io::Cursor;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use tempfile::NamedTempFile;

use super::{DEFAULT_WATERMARK_BYTES, ExportSettings, calculate_resize_target};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum WatermarkAnchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WatermarkSettings {
    pub path: String,
    pub anchor: WatermarkAnchor,
    pub scale: f32,
    pub spacing: f32,
    pub opacity: f32,
}

pub(crate) const DEFAULT_WATERMARK_PATH: &str = "builtin://default-watermark";

const MAX_WATERMARK_SOURCE_BYTES: u64 = 64 * 1024 * 1024;

const MAX_WATERMARK_SOURCE_PIXELS: u64 = 100_000_000;

const MAX_WATERMARK_SOURCE_EDGE: u32 = 32_768;

const STORED_WATERMARK_EDGE: u32 = 4_096;

pub(super) struct PreparedWatermark {
    pub(super) image: RgbaImage,
    pub(super) x: i64,
    pub(super) y: i64,
}

pub(super) fn load_watermark_image(path: &str) -> Result<DynamicImage, String> {
    if path == DEFAULT_WATERMARK_PATH {
        return image::load_from_memory_with_format(DEFAULT_WATERMARK_BYTES, ImageFormat::Png)
            .map_err(|error| format!("Failed to decode the built-in watermark image: {error}"));
    }

    image::open(path).map_err(|error| format!("Failed to open watermark image: {error}"))
}

fn safe_watermark_stem(source_path: &Path) -> String {
    let stem = source_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("watermark");
    let sanitized: String = stem
        .chars()
        .filter_map(|character| {
            if character.is_alphanumeric() || matches!(character, '-' | '_') {
                Some(character)
            } else if character.is_whitespace() {
                Some('-')
            } else {
                None
            }
        })
        .take(48)
        .collect();

    if sanitized.is_empty() {
        "watermark".to_string()
    } else {
        sanitized
    }
}

pub(super) fn import_watermark_image_impl(
    source_path: &Path,
    watermark_directory: &Path,
) -> Result<PathBuf, String> {
    let metadata = fs::metadata(source_path)
        .map_err(|error| format!("Failed to inspect the selected watermark image: {error}"))?;
    if !metadata.is_file() {
        return Err("The selected watermark path is not a file.".to_string());
    }
    if metadata.len() > MAX_WATERMARK_SOURCE_BYTES {
        return Err("The selected watermark image exceeds the 64 MiB input limit.".to_string());
    }

    let source_file = fs::File::open(source_path)
        .map_err(|error| format!("Failed to open the selected watermark image: {error}"))?;
    let mut source_bytes = Vec::new();
    source_file
        .take(MAX_WATERMARK_SOURCE_BYTES + 1)
        .read_to_end(&mut source_bytes)
        .map_err(|error| format!("Failed to read the selected watermark image: {error}"))?;
    if source_bytes.len() as u64 > MAX_WATERMARK_SOURCE_BYTES {
        return Err("The selected watermark image exceeds the 64 MiB input limit.".to_string());
    }

    let (source_width, source_height) = image::ImageReader::new(Cursor::new(&source_bytes))
        .with_guessed_format()
        .map_err(|error| format!("Failed to detect the selected watermark format: {error}"))?
        .into_dimensions()
        .map_err(|error| format!("Failed to read the selected watermark dimensions: {error}"))?;
    let source_pixels = u64::from(source_width)
        .checked_mul(u64::from(source_height))
        .ok_or_else(|| {
            "The selected watermark dimensions overflow the supported range.".to_string()
        })?;
    if source_pixels > MAX_WATERMARK_SOURCE_PIXELS
        || source_width > MAX_WATERMARK_SOURCE_EDGE
        || source_height > MAX_WATERMARK_SOURCE_EDGE
    {
        return Err("The selected watermark dimensions exceed the supported limit.".to_string());
    }

    let source_image = image::load_from_memory(&source_bytes)
        .map_err(|error| format!("Failed to decode the selected watermark image: {error}"))?;
    let stored_image = if source_width.max(source_height) > STORED_WATERMARK_EDGE {
        source_image.resize(
            STORED_WATERMARK_EDGE,
            STORED_WATERMARK_EDGE,
            imageops::FilterType::Lanczos3,
        )
    } else {
        source_image
    };

    let mut encoded_png = Vec::new();
    stored_image
        .write_to(&mut Cursor::new(&mut encoded_png), ImageFormat::Png)
        .map_err(|error| format!("Failed to encode the selected watermark image: {error}"))?;

    fs::create_dir_all(watermark_directory)
        .map_err(|error| format!("Failed to create the watermark library: {error}"))?;
    let content_hash = blake3::hash(&encoded_png).to_hex().to_string();
    let destination = watermark_directory.join(format!(
        "{}-{}.png",
        safe_watermark_stem(source_path),
        content_hash
    ));

    let mut temporary = NamedTempFile::new_in(watermark_directory)
        .map_err(|error| format!("Failed to create a temporary watermark asset: {error}"))?;
    temporary
        .write_all(&encoded_png)
        .map_err(|error| format!("Failed to write the watermark asset: {error}"))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| format!("Failed to sync the watermark asset: {error}"))?;
    match temporary.persist_noclobber(&destination) {
        Ok(_) => Ok(destination),
        Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = fs::read(&destination).map_err(|read_error| {
                format!("Failed to verify the existing watermark asset: {read_error}")
            })?;
            if blake3::hash(&existing).to_hex().to_string() == content_hash {
                Ok(destination)
            } else {
                Err("The stored watermark asset failed its content-hash check.".to_string())
            }
        }
        Err(error) => Err(format!(
            "Failed to publish the watermark asset: {}",
            error.error
        )),
    }
}

pub(super) fn prepare_watermark(
    base_w: u32,
    base_h: u32,
    watermark_settings: &WatermarkSettings,
) -> Result<Option<PreparedWatermark>, String> {
    let watermark_img = load_watermark_image(&watermark_settings.path)?;

    let base_min_dim = base_w.min(base_h) as f32;

    let watermark_scale_factor =
        (base_min_dim * (watermark_settings.scale / 100.0)) / watermark_img.width().max(1) as f32;
    let new_wm_w = (watermark_img.width() as f32 * watermark_scale_factor).round() as u32;
    let new_wm_h = (watermark_img.height() as f32 * watermark_scale_factor).round() as u32;

    if new_wm_w == 0 || new_wm_h == 0 {
        return Ok(None);
    }

    let scaled_watermark =
        watermark_img.resize_exact(new_wm_w, new_wm_h, image::imageops::FilterType::Lanczos3);
    let mut scaled_watermark_rgba = scaled_watermark.to_rgba8();

    let opacity_factor = (watermark_settings.opacity / 100.0).clamp(0.0, 1.0);
    for pixel in scaled_watermark_rgba.pixels_mut() {
        pixel[3] = (pixel[3] as f32 * opacity_factor) as u8;
    }
    let spacing_pixels = (base_min_dim * (watermark_settings.spacing / 100.0)) as i64;
    let (wm_w, wm_h) = scaled_watermark_rgba.dimensions();

    let x = match watermark_settings.anchor {
        WatermarkAnchor::TopLeft | WatermarkAnchor::CenterLeft | WatermarkAnchor::BottomLeft => {
            spacing_pixels
        }
        WatermarkAnchor::TopCenter | WatermarkAnchor::Center | WatermarkAnchor::BottomCenter => {
            (base_w as i64 - wm_w as i64) / 2
        }
        WatermarkAnchor::TopRight | WatermarkAnchor::CenterRight | WatermarkAnchor::BottomRight => {
            base_w as i64 - wm_w as i64 - spacing_pixels
        }
    };

    let y = match watermark_settings.anchor {
        WatermarkAnchor::TopLeft | WatermarkAnchor::TopCenter | WatermarkAnchor::TopRight => {
            spacing_pixels
        }
        WatermarkAnchor::CenterLeft | WatermarkAnchor::Center | WatermarkAnchor::CenterRight => {
            (base_h as i64 - wm_h as i64) / 2
        }
        WatermarkAnchor::BottomLeft
        | WatermarkAnchor::BottomCenter
        | WatermarkAnchor::BottomRight => base_h as i64 - wm_h as i64 - spacing_pixels,
    };

    Ok(Some(PreparedWatermark {
        image: scaled_watermark_rgba,
        x,
        y,
    }))
}

fn apply_watermark(
    base_image: &mut DynamicImage,
    watermark_settings: &WatermarkSettings,
) -> Result<(), String> {
    let (base_w, base_h) = base_image.dimensions();
    if let Some(watermark) = prepare_watermark(base_w, base_h, watermark_settings)? {
        image::imageops::overlay(base_image, &watermark.image, watermark.x, watermark.y);
    }

    Ok(())
}

pub(crate) fn apply_export_resize_and_watermark(
    mut image: DynamicImage,
    export_settings: &ExportSettings,
) -> Result<DynamicImage, String> {
    if let Some(resize_opts) = &export_settings.resize {
        let (current_w, current_h) = image.dimensions();
        let (target_w, target_h) = calculate_resize_target(current_w, current_h, resize_opts);

        if target_w != current_w || target_h != current_h {
            image = image.resize(target_w, target_h, imageops::FilterType::Lanczos3);
        }
    }

    if let Some(watermark_settings) = &export_settings.watermark {
        apply_watermark(&mut image, watermark_settings)?;
    }
    Ok(image)
}

#[cfg(not(target_os = "android"))]
impl PreparedWatermark {
    pub(super) fn row_intersection(&self, base_width: u32, row_y: u32) -> Option<(u32, u32, u32)> {
        let watermark_y = i64::from(row_y) - self.y;
        if watermark_y < 0 || watermark_y >= i64::from(self.image.height()) {
            return None;
        }

        let start_x = self.x.max(0);
        let end_x = (self.x + i64::from(self.image.width())).min(i64::from(base_width));
        if start_x >= end_x {
            return None;
        }

        Some((start_x as u32, end_x as u32, watermark_y as u32))
    }

    pub(super) fn blend_row(&self, base_width: u32, row_y: u32, rgba_row: &mut [u8]) {
        let Some((start_x, end_x, watermark_y)) = self.row_intersection(base_width, row_y) else {
            return;
        };

        for base_x in start_x..end_x {
            let watermark_x = (i64::from(base_x) - self.x) as u32;
            let byte_offset = base_x as usize * 4;
            let mut base = Rgba([
                rgba_row[byte_offset],
                rgba_row[byte_offset + 1],
                rgba_row[byte_offset + 2],
                rgba_row[byte_offset + 3],
            ]);
            base.blend(self.image.get_pixel(watermark_x, watermark_y));
            rgba_row[byte_offset..byte_offset + 4].copy_from_slice(&base.0);
        }
    }
}
