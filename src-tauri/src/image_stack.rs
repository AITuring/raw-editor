use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use crate::panorama_stitching::{AlignmentMode, BlendMode, stitch_images_with_options};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageEncoder};
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use tauri::{AppHandle, Emitter, Manager};
use uuid::Uuid;

// Result inspection is a core part of the stacking workflow. Keep enough source
// pixels for meaningful 1:1-style zooming instead of stretching a small proxy.
const PREVIEW_MAX_LONG_SIDE: u32 = 12_288;
const PREVIEW_MAX_PIXELS: u64 = 100_000_000;
const PREVIEW_JPEG_QUALITY: u8 = 98;

fn resolve_blend_mode(value: &str) -> BlendMode {
    match value {
        "focus" => BlendMode::FocusStack,
        _ => BlendMode::Panorama,
    }
}

fn preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (0, 0);
    }
    let long_side_scale = PREVIEW_MAX_LONG_SIDE as f64 / width.max(height) as f64;
    let pixel_scale = (PREVIEW_MAX_PIXELS as f64 / (width as u64 * height as u64) as f64).sqrt();
    let scale = 1.0_f64.min(long_side_scale).min(pixel_scale);
    (
        (width as f64 * scale).round().max(1.0) as u32,
        (height as f64 * scale).round().max(1.0) as u32,
    )
}

fn write_preview_file(
    image: &DynamicImage,
    app_handle: &AppHandle,
) -> Result<(String, u32, u32), String> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err("The image result is empty.".to_string());
    }
    let (preview_width, preview_height) = preview_dimensions(width, height);
    let full_rgb = image.to_rgb8();
    let preview_rgb = if (preview_width, preview_height) == (width, height) {
        full_rgb
    } else {
        image::imageops::resize(
            &full_rgb,
            preview_width,
            preview_height,
            FilterType::Lanczos3,
        )
    };

    let preview_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|error| format!("Failed to resolve image-stack preview cache: {error}"))?
        .join("image-stack-previews");
    fs::create_dir_all(&preview_dir)
        .map_err(|error| format!("Failed to create image-stack preview cache: {error}"))?;
    let preview_path = preview_dir.join(format!("{}.jpg", Uuid::new_v4()));
    let file = File::create(&preview_path)
        .map_err(|error| format!("Failed to create image-stack preview: {error}"))?;
    let mut writer = BufWriter::new(file);
    let mut encoder = JpegEncoder::new_with_quality(&mut writer, PREVIEW_JPEG_QUALITY);
    encoder
        .set_icc_profile(crate::color_management::srgb_v4_profile().to_vec())
        .map_err(|error| {
            format!("Failed to attach the image-stack preview color profile: {error}")
        })?;
    encoder
        .encode_image(&preview_rgb)
        .map_err(|error| format!("Failed to encode image-stack preview: {error}"))?;
    drop(encoder);
    writer
        .flush()
        .map_err(|error| format!("Failed to finish image-stack preview: {error}"))?;

    if let Ok(entries) = fs::read_dir(&preview_dir) {
        for entry in entries.flatten() {
            let stale_path = entry.path();
            if stale_path != preview_path && stale_path.is_file() {
                let _ = fs::remove_file(stale_path);
            }
        }
    }

    Ok((
        preview_path.to_string_lossy().into_owned(),
        preview_width,
        preview_height,
    ))
}

#[tauri::command]
pub async fn process_image_stack(
    paths: Vec<String>,
    blend_mode: String,
    alignment_mode: String,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if paths.len() < 2 {
        return Err("Please select at least two images.".to_string());
    }
    if paths.len() > 30 {
        return Err("Image stack is limited to 30 source images.".to_string());
    }

    let source_paths: Vec<String> = paths
        .iter()
        .map(|path| parse_virtual_path(path).0.to_string_lossy().into_owned())
        .collect();
    let selected_blend_mode = resolve_blend_mode(&blend_mode);
    let selected_alignment_mode = AlignmentMode::from_wire(&alignment_mode);
    let result_handle = state.image_stack_result.clone();
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
            Ok(image) => {
                let _ = app_handle.emit("image-stack-progress", "Creating preview…");
                let (preview_path, preview_width, preview_height) =
                    write_preview_file(&image, &app_handle)?;
                *result_handle.lock().unwrap() = Some(image);
                let _ = app_handle.emit(
                    "image-stack-complete",
                    serde_json::json!({
                        "previewPath": preview_path,
                        "width": preview_width,
                        "height": preview_height,
                    }),
                );
                Ok(())
            }
            Err(error) => {
                let _ = app_handle.emit("image-stack-error", error.clone());
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
    use super::{PREVIEW_MAX_LONG_SIDE, PREVIEW_MAX_PIXELS, preview_dimensions};

    #[test]
    fn preview_dimensions_keep_images_within_the_quality_budget() {
        assert_eq!(preview_dimensions(4_000, 3_000), (4_000, 3_000));

        let (wide_width, wide_height) = preview_dimensions(20_000, 2_000);
        assert_eq!(wide_width, PREVIEW_MAX_LONG_SIDE);
        assert_eq!(wide_height, 1_229);

        let (large_width, large_height) = preview_dimensions(8_256, 5_504);
        assert!(large_width <= PREVIEW_MAX_LONG_SIDE);
        assert!(large_width as u64 * large_height as u64 <= PREVIEW_MAX_PIXELS + 10_000);

        let (stack_width, stack_height) = preview_dimensions(9_305, 12_618);
        assert!(
            stack_height > 11_000,
            "large stack previews must retain inspection detail"
        );
        assert!(stack_width as u64 * stack_height as u64 <= PREVIEW_MAX_PIXELS + 20_000);
    }
}

#[tauri::command]
pub async fn save_image_stack(
    first_path_str: String,
    blend_mode: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let image = state
        .image_stack_result
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| "No image-stack result is available to save.".to_string())?;
    let (first_path, _) = parse_virtual_path(&first_path_str);
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
    let (filename, output) = if image.color().has_alpha() {
        (
            format!("{stem}_{suffix}.png"),
            DynamicImage::ImageRgba8(image.to_rgba8()),
        )
    } else if image.as_rgb32f().is_some() {
        (format!("{stem}_{suffix}.tiff"), image)
    } else {
        (
            format!("{stem}_{suffix}.png"),
            DynamicImage::ImageRgb8(image.to_rgb8()),
        )
    };
    let output_path = parent.join(filename);
    output
        .save(&output_path)
        .map_err(|error| format!("Failed to save image-stack result: {error}"))?;
    let _ =
        crate::exif_processing::write_rrexif_sidecar(&first_path.to_string_lossy(), &output_path);
    Ok(output_path.to_string_lossy().to_string())
}
