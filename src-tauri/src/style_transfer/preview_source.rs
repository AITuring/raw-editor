//! Source loading and bounded previews for the Style Lab.
//!
//! Loads references and targets through the shared image loader, downsizes them
//! to the preview edge and, on desktop, decodes very large JPEGs at a reduced
//! scale instead of developing the full frame.

use crate::app_settings::AppSettings;
use crate::file_management::read_file_mapped;
use crate::formats::is_raw_file;
use crate::image_loader::load_base_image_from_bytes_without_exif_persistence;
use crate::image_processing::apply_linear_to_srgb;
use crate::image_processing::apply_orientation;
use image::DynamicImage;
use image::GenericImageView;
use image::RgbImage;
#[cfg(not(target_os = "android"))]
use mozjpeg::Decompress;
use std::fs;
use std::path::Path;

use super::real_path;

pub(super) const PREVIEW_EDGE: u32 = 1_600;

// The regular image loader intentionally keeps the editor's colour-managed
// float pipeline. A very large JPEG, however, would need multiple full-frame
// allocations just to produce a style-transfer export. Desktop JPEGs above
// this threshold use the bounded mozjpeg row path below; smaller files keep
// the shared loader's richer metadata/ICC behaviour.
#[cfg(not(target_os = "android"))]
const STREAMING_PIXEL_THRESHOLD: u64 = 40_000_000;

pub(super) fn load_image(path: &str, settings: &AppSettings) -> Result<DynamicImage, String> {
    let source_path = real_path(path);
    let source_path_string = source_path.to_string_lossy().to_string();
    let loaded = match read_file_mapped(&source_path) {
        Ok(mapped) => load_base_image_from_bytes_without_exif_persistence(
            &mapped,
            &source_path_string,
            false,
            settings,
            None,
        )
        .map_err(|error| error.to_string()),
        Err(map_error) => {
            log::warn!(
                "Style transfer could not map '{}': {}. Falling back to read.",
                source_path_string,
                map_error
            );
            let bytes = fs::read(&source_path).map_err(|error| error.to_string())?;
            load_base_image_from_bytes_without_exif_persistence(
                &bytes,
                &source_path_string,
                false,
                settings,
                None,
            )
            .map_err(|error| error.to_string())
        }
    }?;

    // RAW development feeds the editor's GPU in linear light, while the
    // colour-transfer controls (and the encoded output) are display-referred
    // sRGB. Convert only RAW sources here; raster inputs are already normalized
    // by `load_image_with_orientation` and must not be gamma-converted twice.
    Ok(if is_raw_file(&source_path) {
        apply_linear_to_srgb(loaded)
    } else {
        loaded
    })
}

pub(super) fn downscale_preview_to_edge(image: &DynamicImage, max_edge: u32) -> RgbImage {
    let max_edge = max_edge.max(1);
    let (width, height) = image.dimensions();
    if width.max(height) <= max_edge {
        return image.to_rgb8();
    }

    // The fast thumbnail path in image 0.25.10 applies an integer rounding
    // offset to floating-point samples. Our shared raster/RAW loader returns
    // RGB32F, so that path adds about 0.5 to every averaged channel and washes
    // out both the visible preview and the statistics used for style matching.
    // Filtered resize keeps the float samples in their original range.
    image
        .resize(max_edge, max_edge, image::imageops::FilterType::Triangle)
        .to_rgb8()
}

pub(super) fn downscale_preview(image: &DynamicImage) -> RgbImage {
    downscale_preview_to_edge(image, PREVIEW_EDGE)
}

#[cfg(not(target_os = "android"))]
#[derive(Debug, Clone, Copy)]
pub(super) struct JpegInfo {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) orientation: rawler::Orientation,
}

#[cfg(not(target_os = "android"))]
pub(super) fn is_jpeg_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| matches!(extension.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
}

#[cfg(not(target_os = "android"))]
fn orientation_from_bytes(bytes: &[u8]) -> rawler::Orientation {
    let Some(exif) = crate::exif_processing::read_exif(bytes) else {
        return rawler::Orientation::Normal;
    };
    exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .and_then(|field| field.value.get_uint(0))
        .map(|value| rawler::Orientation::from_u16(value as u16))
        .unwrap_or(rawler::Orientation::Normal)
}

#[cfg(not(target_os = "android"))]
fn jpeg_orientation(path: &Path) -> rawler::Orientation {
    if let Ok(mapped) = read_file_mapped(path) {
        return orientation_from_bytes(&mapped);
    }
    fs::read(path)
        .map(|bytes| orientation_from_bytes(&bytes))
        .unwrap_or(rawler::Orientation::Normal)
}

#[cfg(not(target_os = "android"))]
pub(super) fn jpeg_info(path: &Path) -> Option<JpegInfo> {
    if !is_jpeg_path(path) {
        return None;
    }

    let decoder = Decompress::new_path(path).ok()?;
    let width = u32::try_from(decoder.width()).ok()?;
    let height = u32::try_from(decoder.height()).ok()?;
    Some(JpegInfo {
        width,
        height,
        orientation: jpeg_orientation(path),
    })
}

#[cfg(not(target_os = "android"))]
pub(super) fn large_jpeg_info(path: &Path) -> Option<JpegInfo> {
    let info = jpeg_info(path)?;
    let pixels = u64::from(info.width).checked_mul(u64::from(info.height))?;
    (pixels >= STREAMING_PIXEL_THRESHOLD).then_some(info)
}

#[cfg(not(target_os = "android"))]
fn jpeg_preview_scale(width: u32, height: u32) -> u8 {
    let longest_edge = width.max(height);
    if longest_edge > PREVIEW_EDGE.saturating_mul(4) {
        1
    } else if longest_edge > PREVIEW_EDGE.saturating_mul(2) {
        2
    } else if longest_edge > PREVIEW_EDGE {
        4
    } else {
        8
    }
}

#[cfg(not(target_os = "android"))]
pub(super) fn decode_jpeg_preview(path: &Path, info: JpegInfo) -> Result<RgbImage, String> {
    let mut decoder = Decompress::new_path(path)
        .map_err(|error| format!("Failed to open JPEG preview: {error}"))?;
    decoder.scale(jpeg_preview_scale(info.width, info.height));
    let mut started = decoder
        .rgb()
        .map_err(|error| format!("Failed to start JPEG preview decode: {error}"))?;
    let width = u32::try_from(started.width())
        .map_err(|_| "JPEG preview width exceeds the supported range.".to_string())?;
    let height = u32::try_from(started.height())
        .map_err(|_| "JPEG preview height exceeds the supported range.".to_string())?;
    let row_bytes = usize::try_from(width)
        .ok()
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "JPEG preview row is too wide.".to_string())?;
    let pixel_bytes = row_bytes
        .checked_mul(usize::try_from(height).unwrap_or(usize::MAX))
        .ok_or_else(|| "JPEG preview is too large.".to_string())?;
    let mut pixels = vec![0_u8; pixel_bytes];
    for row in pixels.chunks_exact_mut(row_bytes) {
        let decoded = started
            .read_scanlines_into(row)
            .map_err(|error| format!("Failed to read JPEG preview row: {error}"))?;
        if decoded.len() != row_bytes {
            return Err("JPEG preview decoder returned an incomplete row.".to_string());
        }
    }
    started
        .finish()
        .map_err(|error| format!("Failed to finish JPEG preview decode: {error}"))?;
    let image = RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| "JPEG preview dimensions do not match the decoded pixels.".to_string())?;
    let oriented = apply_orientation(DynamicImage::ImageRgb8(image), info.orientation);
    Ok(if oriented.width().max(oriented.height()) > PREVIEW_EDGE {
        oriented.thumbnail(PREVIEW_EDGE, PREVIEW_EDGE).to_rgb8()
    } else {
        oriented.to_rgb8()
    })
}

pub(super) fn load_preview_for_path(
    path: &str,
    settings: &AppSettings,
) -> Result<RgbImage, String> {
    #[cfg(not(target_os = "android"))]
    {
        let source_path = real_path(path);
        if let Some(info) = large_jpeg_info(&source_path) {
            return decode_jpeg_preview(&source_path, info);
        }
    }

    let image = load_image(path, settings)?;
    Ok(downscale_preview(&image))
}
