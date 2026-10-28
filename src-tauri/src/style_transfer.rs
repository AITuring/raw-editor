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
mod tests {
    use std::path::Path;

    use super::{
        StyleTransferMode, TransferTransform, apply_transform, downscale_preview_to_edge,
        estimate_transform, hsv_to_rgb, rgb_to_hsv,
    };
    use image::{DynamicImage, Rgb, Rgb32FImage, RgbImage};

    #[test]
    fn floating_point_preview_downscale_preserves_midtone_levels() {
        let source =
            DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(4, 4, Rgb([0.25, 0.5, 0.75])));

        let preview = downscale_preview_to_edge(&source, 2);

        assert_eq!(preview.dimensions(), (2, 2));
        for pixel in preview.pixels() {
            for (actual, expected) in pixel.0.into_iter().zip([64_i16, 128, 191]) {
                assert!((i16::from(actual) - expected).abs() <= 1);
            }
        }
    }

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
            value_offset: 0.2,
            value_contrast: 1.2,
            target_value_mean: 0.2,
            channel_scale: [1.4, 0.7, 1.3],
            channel_offset: [0.1, -0.1, 0.0],
        };
        let result = apply_transform(image.clone(), transform, StyleTransferMode::Mood, 0.0);
        assert_eq!(result, image);
    }

    fn mean_horizontal_gradient(image: &RgbImage) -> f32 {
        let mut total = 0.0_f32;
        let mut count = 0_u32;
        for y in 0..image.height() {
            for x in 1..image.width() {
                let left = image.get_pixel(x - 1, y).0[0] as f32;
                let right = image.get_pixel(x, y).0[0] as f32;
                total += (right - left).abs();
                count += 1;
            }
        }
        total / count.max(1) as f32
    }

    #[test]
    fn tonal_transfer_preserves_natural_detail_contrast() {
        let target = RgbImage::from_fn(64, 64, |x, y| {
            let value = if (x + y) % 2 == 0 { 96 } else { 112 };
            Rgb([value, value, value])
        });
        let reference = RgbImage::from_fn(64, 64, |x, y| {
            let value = if (x + y) % 2 == 0 { 32 } else { 224 };
            Rgb([value, value, value])
        });
        let transform = estimate_transform(&reference, &target);

        assert!(transform.value_contrast <= 1.2);
        assert!(transform.channel_scale.iter().all(|scale| *scale <= 1.23));

        for mode in [StyleTransferMode::Mood, StyleTransferMode::Distribution] {
            let styled = apply_transform(target.clone(), transform, mode, 1.0);
            let gradient_ratio =
                mean_horizontal_gradient(&styled) / mean_horizontal_gradient(&target);
            assert!(
                gradient_ratio <= 1.25,
                "{mode:?} amplified detail contrast by {gradient_ratio:.3}x"
            );
        }
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
