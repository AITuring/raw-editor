use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use crate::panorama_stitching::{AlignmentMode, BlendMode, stitch_images_with_options};
use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, GenericImageView, ImageFormat};
use std::io::Cursor;
use tauri::{AppHandle, Emitter};

fn resolve_blend_mode(value: &str) -> BlendMode {
    match value {
        "focus" => BlendMode::FocusStack,
        _ => BlendMode::Panorama,
    }
}

fn preview_base64(image: &DynamicImage) -> Result<String, String> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err("The image result is empty.".to_string());
    }
    let longest_side = width.max(height);
    let (preview_width, preview_height) = if longest_side > 900 {
        (
            (width as f32 * 900.0 / longest_side as f32)
                .round()
                .max(1.0) as u32,
            (height as f32 * 900.0 / longest_side as f32)
                .round()
                .max(1.0) as u32,
        )
    } else {
        (width, height)
    };
    let preview = if (preview_width, preview_height) == (width, height) {
        image.clone()
    } else {
        crate::image_processing::downscale_f32_image(image, preview_width, preview_height)
    };
    let mut buffer = Cursor::new(Vec::new());
    preview
        .to_rgb8()
        .write_to(&mut buffer, ImageFormat::Png)
        .map_err(|error| format!("Failed to encode image-stack preview: {error}"))?;
    Ok(format!(
        "data:image/png;base64,{}",
        general_purpose::STANDARD.encode(buffer.get_ref())
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
                let base64 = preview_base64(&image)?;
                *result_handle.lock().unwrap() = Some(image);
                let _ = app_handle.emit(
                    "image-stack-complete",
                    serde_json::json!({ "base64": base64 }),
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
