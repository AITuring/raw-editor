//! HDR merge and save, and collage save commands.

use crate::app_settings::load_settings;
use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use crate::hdr_deghosting::align_hdr_frames;
use crate::hdr_deghosting::assert_uniform_dimensions;
use crate::hdr_deghosting::load_hdr_frames;
use crate::image_processing::apply_linear_to_srgb;
use base64::Engine as _;
use base64::engine::general_purpose;
use image::DynamicImage;
use image::ImageFormat;
use image_hdr::hdr_merge_images;
use image_hdr::input::HDRInput;
use std::fs;
use std::io::Cursor;
use tauri::Emitter;

#[tauri::command]
pub(super) async fn merge_hdr(
    paths: Vec<String>,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if paths.len() < 2 {
        return Err("Please select at least two images to merge.".to_string());
    }

    let hdr_result_handle = state.hdr_result.clone();
    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let mut frames = load_hdr_frames(&paths, &app_handle, &settings)?;
    assert_uniform_dimensions(&frames)?;
    align_hdr_frames(&mut frames, &app_handle);

    let images: Vec<HDRInput> = frames
        .iter()
        .map(|(path, img, exposure, gains)| {
            HDRInput::with_image(img, *exposure, *gains)
                .map_err(|e| format!("Failed to prepare HDR input for {}: {}", path, e))
        })
        .collect::<Result<Vec<HDRInput>, String>>()?;

    log::info!("Starting HDR merge of {} images", images.len());
    let _ = app_handle.emit("hdr-progress", "Merging exposures...");
    let mut hdr_merged = hdr_merge_images(&mut images.into()).map_err(|e| e.to_string())?;
    let _ = app_handle.emit("hdr-progress", "Tone mapping...");
    hdr_merged =
        image_hdr::stretch::apply_histogram_stretch(&hdr_merged).map_err(|e| e.to_string())?;
    hdr_merged = apply_linear_to_srgb(hdr_merged);
    log::info!("HDR merge completed");

    let mut buf = Cursor::new(Vec::new());
    if let Err(e) = hdr_merged.to_rgb8().write_to(&mut buf, ImageFormat::Png) {
        return Err(format!("Failed to encode hdr preview: {}", e));
    }

    let base64_str = general_purpose::STANDARD.encode(buf.get_ref());
    let final_base64 = format!("data:image/png;base64,{}", base64_str);

    let _ = app_handle.emit("hdr-progress", "Creating preview...");

    *hdr_result_handle.lock().unwrap() = Some(hdr_merged);

    let _ = app_handle.emit(
        "hdr-complete",
        serde_json::json!({
            "base64": final_base64,
        }),
    );
    Ok(())
}

#[tauri::command]
pub(super) async fn save_hdr(
    first_path_str: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let hdr_image = state.hdr_result.lock().unwrap().take().ok_or_else(|| {
        "No hdr image found in memory to save. It might have already been saved.".to_string()
    })?;

    let (first_path, _) = parse_virtual_path(&first_path_str);
    let parent_dir = first_path
        .parent()
        .ok_or_else(|| "Could not determine parent directory of the first image.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("hdr");

    let (output_filename, image_to_save): (String, DynamicImage) = if hdr_image.color().has_alpha()
    {
        (
            format!("{}_Hdr.png", stem),
            DynamicImage::ImageRgba8(hdr_image.to_rgba8()),
        )
    } else if hdr_image.as_rgb32f().is_some() {
        (format!("{}_Hdr.tiff", stem), hdr_image)
    } else {
        (
            format!("{}_Hdr.png", stem),
            DynamicImage::ImageRgb8(hdr_image.to_rgb8()),
        )
    };

    let output_path = parent_dir.join(output_filename);

    image_to_save
        .save(&output_path)
        .map_err(|e| format!("Failed to save hdr image: {}", e))?;

    let (real_path, _) = crate::file_management::parse_virtual_path(&first_path_str);
    let _ =
        crate::exif_processing::write_rrexif_sidecar(&real_path.to_string_lossy(), &output_path);

    Ok(output_path.to_string_lossy().to_string())
}

#[tauri::command]
pub(super) async fn save_collage(
    base64_data: String,
    first_path_str: String,
) -> Result<String, String> {
    let data_url_prefix = "data:image/png;base64,";
    if !base64_data.starts_with(data_url_prefix) {
        return Err("Invalid base64 data format".to_string());
    }
    let encoded_data = &base64_data[data_url_prefix.len()..];

    let decoded_bytes = general_purpose::STANDARD
        .decode(encoded_data)
        .map_err(|e| format!("Failed to decode base64: {}", e))?;

    let (first_path, _) = parse_virtual_path(&first_path_str);
    let parent_dir = first_path
        .parent()
        .ok_or_else(|| "Could not determine parent directory of the first image.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("collage");

    let output_filename = format!("{}_Collage.png", stem);
    let output_path = parent_dir.join(output_filename);

    fs::write(&output_path, &decoded_bytes)
        .map_err(|e| format!("Failed to save collage image: {}", e))?;

    Ok(output_path.to_string_lossy().to_string())
}
