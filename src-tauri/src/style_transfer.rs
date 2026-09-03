//! Local, non-destructive colour transfer for the Style Lab.
//!
//! The browser workbench renders a compact preview for immediate feedback.
//! This command repeats the same HSV/statistical transform against the full
//! target image so exporting does not silently reduce a large photograph to
//! the preview canvas size.

use std::fs::{self, File};
#[cfg(target_os = "android")]
use std::io::BufWriter;
#[cfg(not(target_os = "android"))]
use std::io::Write;
use std::path::{Path, PathBuf};

#[cfg(target_os = "android")]
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GenericImageView, ImageFormat, RgbImage};
#[cfg(not(target_os = "android"))]
use mozjpeg::Decompress;
use mozjpeg_rs::Preset;
use serde::Deserialize;
use tauri::{AppHandle, Emitter, ipc::Response};
#[cfg(not(target_os = "android"))]
use tempfile::NamedTempFile;

use crate::app_settings::{AppSettings, load_settings};
use crate::color_management::srgb_preview_encoder;
use crate::file_management::read_file_mapped;
use crate::formats::is_raw_file;
use crate::image_loader::load_base_image_from_bytes_without_exif_persistence;
use crate::image_processing::{apply_linear_to_srgb, apply_orientation};

const PREVIEW_EDGE: u32 = 1_600;

// The regular image loader intentionally keeps the editor's colour-managed
// float pipeline. A very large JPEG, however, would need multiple full-frame
// allocations just to produce a style-transfer export. Desktop JPEGs above
// this threshold use the bounded mozjpeg row path below; smaller files keep
// the shared loader's richer metadata/ICC behaviour.
#[cfg(not(target_os = "android"))]
const STREAMING_PIXEL_THRESHOLD: u64 = 40_000_000;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StyleTransferMode {
    #[default]
    Mood,
    Distribution,
}

#[derive(Debug, Clone, Copy)]
struct ImageStats {
    mean: [f32; 3],
    std: [f32; 3],
    mean_saturation: f32,
    mean_value: f32,
    value_std: f32,
    hue_sin: f32,
    hue_cos: f32,
    hue_weight: f32,
}

#[derive(Debug, Clone, Copy)]
struct TransferTransform {
    hue_shift: f32,
    saturation_scale: f32,
    value_scale: f32,
    value_contrast: f32,
    channel_scale: [f32; 3],
    channel_offset: [f32; 3],
    target_luma: f32,
}

#[derive(Debug, Clone, Copy)]
struct AppliedTransform {
    mode: StyleTransferMode,
    hue_shift: f32,
    saturation_scale: f32,
    value_scale: f32,
    value_contrast: f32,
    channel_scale: [f32; 3],
    channel_offset: [f32; 3],
    target_luma: f32,
}

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.clamp(min, max)
}

fn normalize_hue(value: f32) -> f32 {
    let mut normalized = value;
    while normalized > 0.5 {
        normalized -= 1.0;
    }
    while normalized < -0.5 {
        normalized += 1.0;
    }
    normalized
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let mut hue = 0.0;

    if delta > 1.0e-6 {
        if (max - r).abs() < 1.0e-6 {
            hue = ((g - b) / delta) % 6.0;
        } else if (max - g).abs() < 1.0e-6 {
            hue = (b - r) / delta + 2.0;
        } else {
            hue = (r - g) / delta + 4.0;
        }
        hue /= 6.0;
        if hue < 0.0 {
            hue += 1.0;
        }
    }

    (hue, if max <= 1.0e-6 { 0.0 } else { delta / max }, max)
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    let normalized_hue = hue.rem_euclid(1.0);
    let scaled = normalized_hue * 6.0;
    let sector = scaled.floor() as i32;
    let fraction = scaled - sector as f32;
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * fraction);
    let t = value * (1.0 - saturation * (1.0 - fraction));

    match sector.rem_euclid(6) {
        0 => [value, t, p],
        1 => [q, value, p],
        2 => [p, value, t],
        3 => [p, q, value],
        4 => [t, p, value],
        _ => [value, p, q],
    }
}

fn collect_stats(image: &RgbImage) -> ImageStats {
    let total = (image.width() as usize)
        .saturating_mul(image.height() as usize)
        .max(1);
    let stride = ((total as f32 / 120_000.0).sqrt().ceil() as u32).max(1);
    let mut sum = [0.0f32; 3];
    let mut sum_squares = [0.0f32; 3];
    let mut saturation_sum = 0.0;
    let mut value_sum = 0.0;
    let mut value_squares = 0.0;
    let mut hue_sin = 0.0;
    let mut hue_cos = 0.0;
    let mut hue_weight = 0.0;
    let mut count: f32 = 0.0;

    for y in (0..image.height()).step_by(stride as usize) {
        for x in (0..image.width()).step_by(stride as usize) {
            let pixel = image.get_pixel(x, y).0;
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let (hue, saturation, value) = rgb_to_hsv(r, g, b);
            let weight = (saturation * 0.65 + value * 0.35).max(0.001);

            sum[0] += r;
            sum[1] += g;
            sum[2] += b;
            sum_squares[0] += r * r;
            sum_squares[1] += g * g;
            sum_squares[2] += b * b;
            saturation_sum += saturation;
            value_sum += value;
            value_squares += value * value;
            hue_sin += (hue * std::f32::consts::TAU).sin() * weight;
            hue_cos += (hue * std::f32::consts::TAU).cos() * weight;
            hue_weight += weight;
            count += 1.0;
        }
    }

    let safe_count = count.max(1.0);
    let mean = [
        sum[0] / safe_count,
        sum[1] / safe_count,
        sum[2] / safe_count,
    ];
    let std = [
        (sum_squares[0] / safe_count - mean[0] * mean[0])
            .max(0.0)
            .sqrt(),
        (sum_squares[1] / safe_count - mean[1] * mean[1])
            .max(0.0)
            .sqrt(),
        (sum_squares[2] / safe_count - mean[2] * mean[2])
            .max(0.0)
            .sqrt(),
    ];
    let mean_value = value_sum / safe_count;

    ImageStats {
        mean,
        std,
        mean_saturation: saturation_sum / safe_count,
        mean_value,
        value_std: (value_squares / safe_count - mean_value * mean_value)
            .max(0.0)
            .sqrt(),
        hue_sin,
        hue_cos,
        hue_weight,
    }
}

fn estimate_transform(reference: &RgbImage, target: &RgbImage) -> TransferTransform {
    let reference_stats = collect_stats(reference);
    let target_stats = collect_stats(target);

    let reference_hue = if reference_stats.hue_weight > 0.0 {
        reference_stats.hue_sin.atan2(reference_stats.hue_cos) / std::f32::consts::TAU
    } else {
        0.0
    };
    let target_hue = if target_stats.hue_weight > 0.0 {
        target_stats.hue_sin.atan2(target_stats.hue_cos) / std::f32::consts::TAU
    } else {
        0.0
    };

    let saturation_scale = clamp(
        reference_stats.mean_saturation / target_stats.mean_saturation.max(0.04),
        0.35,
        2.4,
    );
    let value_scale = clamp(
        reference_stats.mean_value / target_stats.mean_value.max(0.08),
        0.55,
        1.65,
    );
    let value_contrast = clamp(
        reference_stats.value_std / target_stats.value_std.max(0.04),
        0.6,
        1.7,
    );
    let channel_scale = [
        clamp(
            reference_stats.std[0] / target_stats.std[0].max(0.025),
            0.55,
            1.8,
        ),
        clamp(
            reference_stats.std[1] / target_stats.std[1].max(0.025),
            0.55,
            1.8,
        ),
        clamp(
            reference_stats.std[2] / target_stats.std[2].max(0.025),
            0.55,
            1.8,
        ),
    ];
    let channel_offset = [
        reference_stats.mean[0] - target_stats.mean[0] * channel_scale[0],
        reference_stats.mean[1] - target_stats.mean[1] * channel_scale[1],
        reference_stats.mean[2] - target_stats.mean[2] * channel_scale[2],
    ];

    TransferTransform {
        hue_shift: normalize_hue(reference_hue - target_hue),
        saturation_scale,
        value_scale,
        value_contrast,
        channel_scale,
        channel_offset,
        target_luma: target_stats.mean[0] * 0.2126
            + target_stats.mean[1] * 0.7152
            + target_stats.mean[2] * 0.0722,
    }
}

fn apply_transform(
    mut target: RgbImage,
    transform: TransferTransform,
    mode: StyleTransferMode,
    strength: f32,
) -> RgbImage {
    let applied = AppliedTransform::new(transform, mode, strength);

    for pixel in target.pixels_mut() {
        let original = [
            pixel.0[0] as f32 / 255.0,
            pixel.0[1] as f32 / 255.0,
            pixel.0[2] as f32 / 255.0,
        ];
        let next = applied.apply_pixel(original);

        pixel.0[0] = (clamp(next[0], 0.0, 1.0) * 255.0).round() as u8;
        pixel.0[1] = (clamp(next[1], 0.0, 1.0) * 255.0).round() as u8;
        pixel.0[2] = (clamp(next[2], 0.0, 1.0) * 255.0).round() as u8;
    }

    target
}

impl AppliedTransform {
    fn new(transform: TransferTransform, mode: StyleTransferMode, strength: f32) -> Self {
        let amount = clamp(strength, 0.0, 1.0);
        Self {
            mode,
            hue_shift: transform.hue_shift * amount,
            saturation_scale: 1.0 + (transform.saturation_scale - 1.0) * amount,
            value_scale: 1.0 + (transform.value_scale - 1.0) * amount,
            value_contrast: 1.0 + (transform.value_contrast - 1.0) * amount,
            channel_scale: [
                1.0 + (transform.channel_scale[0] - 1.0) * amount,
                1.0 + (transform.channel_scale[1] - 1.0) * amount,
                1.0 + (transform.channel_scale[2] - 1.0) * amount,
            ],
            channel_offset: [
                transform.channel_offset[0] * amount,
                transform.channel_offset[1] * amount,
                transform.channel_offset[2] * amount,
            ],
            target_luma: transform.target_luma,
        }
    }

    fn apply_pixel(self, original: [f32; 3]) -> [f32; 3] {
        match self.mode {
            StyleTransferMode::Distribution => [
                original[0] * self.channel_scale[0] + self.channel_offset[0],
                original[1] * self.channel_scale[1] + self.channel_offset[1],
                original[2] * self.channel_scale[2] + self.channel_offset[2],
            ],
            StyleTransferMode::Mood => {
                let (hue, saturation, value) = rgb_to_hsv(original[0], original[1], original[2]);
                let adjusted_value = clamp(
                    self.target_luma + (value - self.target_luma) * self.value_contrast,
                    0.0,
                    1.0,
                );
                hsv_to_rgb(
                    hue + self.hue_shift,
                    clamp(saturation * self.saturation_scale, 0.0, 1.0),
                    clamp(adjusted_value * self.value_scale, 0.0, 1.0),
                )
            }
        }
    }
}

#[cfg(not(target_os = "android"))]
fn apply_transform_rgb_row(
    row: &mut [u8],
    transform: TransferTransform,
    mode: StyleTransferMode,
    strength: f32,
) {
    let applied = AppliedTransform::new(transform, mode, strength);
    for pixel in row.chunks_exact_mut(3) {
        let original = [
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        ];
        let next = applied.apply_pixel(original);
        pixel[0] = (clamp(next[0], 0.0, 1.0) * 255.0).round() as u8;
        pixel[1] = (clamp(next[1], 0.0, 1.0) * 255.0).round() as u8;
        pixel[2] = (clamp(next[2], 0.0, 1.0) * 255.0).round() as u8;
    }
}

fn load_image(path: &str, settings: &AppSettings) -> Result<DynamicImage, String> {
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

fn downscale_preview(image: &DynamicImage) -> RgbImage {
    let (width, height) = image.dimensions();
    if width.max(height) <= PREVIEW_EDGE {
        return image.to_rgb8();
    }
    image.thumbnail(PREVIEW_EDGE, PREVIEW_EDGE).to_rgb8()
}

#[cfg(not(target_os = "android"))]
#[derive(Debug, Clone, Copy)]
struct JpegInfo {
    width: u32,
    height: u32,
    orientation: rawler::Orientation,
}

#[cfg(not(target_os = "android"))]
fn is_jpeg_path(path: &Path) -> bool {
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
fn jpeg_info(path: &Path) -> Option<JpegInfo> {
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
fn large_jpeg_info(path: &Path) -> Option<JpegInfo> {
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
fn decode_jpeg_preview(path: &Path, info: JpegInfo) -> Result<RgbImage, String> {
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

fn real_path(path: &str) -> PathBuf {
    path.rsplit_once("?vc=")
        .map(|(source, _)| PathBuf::from(source))
        .unwrap_or_else(|| PathBuf::from(path))
}

fn load_preview_for_path(path: &str, settings: &AppSettings) -> Result<RgbImage, String> {
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

fn encode_output(image: &RgbImage, output_path: &Path) -> Result<(), String> {
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
fn stream_jpeg_export<R: tauri::Runtime>(
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
mod tests {
    use std::path::Path;

    use super::{StyleTransferMode, TransferTransform, apply_transform, hsv_to_rgb, rgb_to_hsv};
    use image::{Rgb, RgbImage};

    #[cfg(not(target_os = "android"))]
    use super::{PREVIEW_EDGE, decode_jpeg_preview, jpeg_info, stream_jpeg_export};
    #[cfg(not(target_os = "android"))]
    use image::{GenericImageView, ImageFormat};

    #[test]
    fn hsv_round_trip_preserves_primary_colours() {
        for rgb in [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 90, 40]] {
            let (hue, saturation, value) = rgb_to_hsv(
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
            );
            let result = hsv_to_rgb(hue, saturation, value);
            for (expected, actual) in rgb.into_iter().zip(result) {
                assert!((expected as f32 / 255.0 - actual).abs() < 0.01);
            }
        }
    }

    #[test]
    fn zero_strength_is_non_destructive() {
        let image = RgbImage::from_pixel(2, 2, Rgb([10, 20, 30]));
        let transform = TransferTransform {
            hue_shift: 0.2,
            saturation_scale: 0.4,
            value_scale: 1.4,
            value_contrast: 1.2,
            channel_scale: [1.4, 0.7, 1.3],
            channel_offset: [0.1, -0.1, 0.0],
            target_luma: 0.2,
        };
        let result = apply_transform(image.clone(), transform, StyleTransferMode::Mood, 0.0);
        assert_eq!(result, image);
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn scaled_jpeg_preview_uses_the_bounded_decoder_path() {
        let mut image = RgbImage::new(3_200, 2_100);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = Rgb([(x % 256) as u8, (y % 256) as u8, 128]);
        }
        let temporary = tempfile::Builder::new()
            .suffix(".jpg")
            .tempfile()
            .expect("create JPEG fixture");
        image
            .save_with_format(temporary.path(), ImageFormat::Jpeg)
            .expect("write JPEG fixture");

        let info = jpeg_info(temporary.path()).expect("inspect JPEG fixture");
        assert_eq!((info.width, info.height), (3_200, 2_100));
        let preview = decode_jpeg_preview(temporary.path(), info).expect("decode JPEG preview");
        assert!(preview.width().max(preview.height()) <= PREVIEW_EDGE);
        assert!(preview.width() > 0 && preview.height() > 0);
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn streaming_export_writes_a_valid_full_size_jpeg() {
        let reference = tempfile::Builder::new()
            .suffix(".jpg")
            .tempfile()
            .expect("create reference fixture");
        let target = tempfile::Builder::new()
            .suffix(".jpg")
            .tempfile()
            .expect("create target fixture");
        let output = tempfile::Builder::new()
            .suffix(".jpg")
            .tempfile()
            .expect("create output fixture");

        let reference_image = RgbImage::from_fn(48, 32, |x, y| Rgb([180, (x * 3) as u8, y as u8]));
        let target_image =
            RgbImage::from_fn(48, 32, |x, y| Rgb([(x * 2) as u8, 80, (y * 4) as u8]));
        reference_image
            .save_with_format(reference.path(), ImageFormat::Jpeg)
            .expect("write reference fixture");
        target_image
            .save_with_format(target.path(), ImageFormat::Jpeg)
            .expect("write target fixture");

        let app = tauri::test::mock_app();
        let settings = crate::app_settings::AppSettings::default();
        let saved = stream_jpeg_export(
            reference.path().to_str().expect("reference path"),
            target.path().to_str().expect("target path"),
            output.path().to_str().expect("output path"),
            0.8,
            StyleTransferMode::Mood,
            &settings,
            &app.handle().clone(),
        )
        .expect("stream styled JPEG");
        assert_eq!(Path::new(&saved), output.path());

        let decoded = image::open(output.path()).expect("decode styled JPEG");
        assert_eq!(decoded.dimensions(), target_image.dimensions());
    }
}
