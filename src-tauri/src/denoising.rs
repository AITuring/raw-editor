use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use image::DynamicImage;
#[cfg(test)]
use image::Rgb;
#[cfg(test)]
use image::Rgb32FImage;
use tauri::Emitter;

const BLOCK_SIZE: usize = 8;
const BLOCK_AREA: usize = 64;
const MAX_GROUP_SIZE: usize = 16;
const STRIDE: usize = 6;
const SEARCH_WINDOW: usize = 19;
const FIXED_POINT_SCALE: f32 = 100_000.0;

mod pipeline;

use pipeline::denoise_image;
#[cfg(test)]
use pipeline::run_bm3d_core;

mod bm3d;

use bm3d::{ProgressReporter, bm3d_process_joint};

mod block_matching;

use block_matching::{block_matching_joint, build_3d_group};

mod transforms;

use transforms::{DctTables, inverse_transform_3d, transform_3d};

#[derive(Clone, Copy)]
struct Bm3dParams {
    sigma: f32,
    hard_th_lambda: f32,
    max_dist_hard: f32,
    chroma_sigma_scale: f32,
}

impl Bm3dParams {
    fn from_intensity(i: f32) -> Self {
        let val = i.clamp(0.001, 1.0);
        Self {
            sigma: val * 80.0,
            hard_th_lambda: 2.0 + (val * 2.5),
            max_dist_hard: 3000.0 + (val * 20000.0),
            chroma_sigma_scale: 1.8,
        }
    }
}

#[tauri::command]
pub async fn apply_denoising(
    path: String,
    intensity: f32,
    method: String,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let (source_path, _) = parse_virtual_path(&path);
    let path_str = source_path.to_string_lossy().to_string();

    let mut ai_session = None;
    if method == "ai" {
        let session = crate::ai_processing::get_or_init_denoise_model(
            &app_handle,
            &state.ai_state,
            &state.ai_init_lock,
        )
        .await
        .map_err(|e| e.to_string())?;
        ai_session = Some(session);
    }

    let denoise_result_handle = state.denoise_result.clone();

    tokio::task::spawn_blocking(move || {
        match denoise_image(path_str, intensity, method, app_handle.clone(), ai_session) {
            Ok((image, _)) => {
                *denoise_result_handle.lock().unwrap() = Some(image);
            }
            Err(e) => {
                let _ = app_handle.emit("denoise-error", e);
            }
        }
    })
    .await
    .map_err(|e| format!("Denoising task failed: {}", e))
}

#[tauri::command]
pub async fn batch_denoise_images(
    paths: Vec<String>,
    intensity: f32,
    method: String,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<String>, String> {
    let mut ai_session = None;
    if method == "ai" {
        let session = crate::ai_processing::get_or_init_denoise_model(
            &app_handle,
            &state.ai_state,
            &state.ai_init_lock,
        )
        .await
        .map_err(|e| e.to_string())?;
        ai_session = Some(session);
    }

    tokio::task::spawn_blocking(move || {
        let mut results = Vec::new();

        for (i, path_str) in paths.iter().enumerate() {
            let emit_batch_progress = || {
                let _ = app_handle.emit(
                    "denoise-batch-progress",
                    serde_json::json!({
                        "current": i + 1,
                        "total": paths.len(),
                        "path": path_str
                    }),
                );
            };

            let (source_path, source_sidecar_path) =
                crate::file_management::parse_virtual_path(path_str);
            let real_path = source_path.to_string_lossy().to_string();

            match crate::denoising::denoise_image(
                real_path.clone(),
                intensity,
                method.clone(),
                app_handle.clone(),
                ai_session.clone(),
            ) {
                Ok((image, _)) => {
                    let is_raw = crate::formats::is_raw_file(&real_path);
                    let parent_dir = source_path.parent().unwrap_or(std::path::Path::new(""));
                    let stem = source_path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy();

                    let (output_filename, image_to_save) = if is_raw {
                        (
                            format!("{}_Denoised.tiff", stem),
                            DynamicImage::ImageRgb16(image.to_rgb16()),
                        )
                    } else {
                        (
                            format!("{}_Denoised.png", stem),
                            DynamicImage::ImageRgb8(image.to_rgb8()),
                        )
                    };

                    let output_path = parent_dir.join(output_filename);
                    if let Err(e) = image_to_save.save(&output_path) {
                        let _ = app_handle.emit(
                            "denoise-error",
                            format!("Failed to save {}: {}", real_path, e),
                        );
                        emit_batch_progress();
                        continue;
                    }

                    let _ = crate::exif_processing::write_rrexif_sidecar(&real_path, &output_path);

                    if source_sidecar_path.exists()
                        && let Some(output_path_str) = output_path.to_str()
                    {
                        let (_, dest_sidecar_path) =
                            crate::file_management::parse_virtual_path(output_path_str);
                        match crate::exif_processing::load_sidecar_checked(&source_sidecar_path) {
                            Ok(metadata) => {
                                if let Err(e) = crate::exif_processing::save_sidecar(
                                    &dest_sidecar_path,
                                    &metadata,
                                ) {
                                    log::warn!(
                                        "Failed to copy sidecar file for denoised image: {}",
                                        e
                                    );
                                }
                            }
                            Err(e) => {
                                log::warn!("Failed to read sidecar for denoised image: {}", e);
                            }
                        }
                    }

                    results.push(output_path.to_string_lossy().to_string());
                }
                Err(e) => {
                    let _ = app_handle.emit(
                        "denoise-error",
                        format!("Failed to denoise {}: {}", real_path, e),
                    );
                }
            }

            emit_batch_progress();
        }

        Ok(results)
    })
    .await
    .map_err(|e| format!("Batch denoising task failed: {}", e))?
}

#[tauri::command]
pub async fn save_denoised_image(
    original_path_str: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let denoised_image = state.denoise_result.lock().unwrap().take().ok_or_else(|| {
        "No denoised image found in memory. It might have already been saved or cleared."
            .to_string()
    })?;

    let is_raw = crate::formats::is_raw_file(&original_path_str);

    let (first_path, source_sidecar_path) =
        crate::file_management::parse_virtual_path(&original_path_str);
    let parent_dir = first_path
        .parent()
        .ok_or_else(|| "Could not determine parent directory.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("denoised");

    let (output_filename, image_to_save): (String, DynamicImage) = if is_raw {
        let filename = format!("{}_Denoised.tiff", stem);
        (
            filename,
            DynamicImage::ImageRgb16(denoised_image.to_rgb16()),
        )
    } else {
        let filename = format!("{}_Denoised.png", stem);
        (filename, DynamicImage::ImageRgb8(denoised_image.to_rgb8()))
    };

    let output_path = parent_dir.join(output_filename);

    image_to_save
        .save(&output_path)
        .map_err(|e| format!("Failed to save image: {}", e))?;

    let (real_path, _) = crate::file_management::parse_virtual_path(&original_path_str);
    let _ =
        crate::exif_processing::write_rrexif_sidecar(&real_path.to_string_lossy(), &output_path);

    if source_sidecar_path.exists()
        && let Some(output_path_str) = output_path.to_str()
    {
        let (_, dest_sidecar_path) = crate::file_management::parse_virtual_path(output_path_str);
        match crate::exif_processing::load_sidecar_checked(&source_sidecar_path) {
            Ok(metadata) => {
                if let Err(e) = crate::exif_processing::save_sidecar(&dest_sidecar_path, &metadata)
                {
                    log::warn!("Failed to copy sidecar file for denoised image: {}", e);
                }
            }
            Err(e) => {
                log::warn!("Failed to read sidecar for denoised image: {}", e);
            }
        }
    }

    Ok(output_path.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean_squared_error(left: &Rgb32FImage, right: &Rgb32FImage) -> f32 {
        left.pixels()
            .zip(right.pixels())
            .flat_map(|(left, right)| {
                (0..3).map(move |channel| {
                    let difference = left[channel] - right[channel];
                    difference * difference
                })
            })
            .sum::<f32>()
            / (left.width() * left.height() * 3) as f32
    }

    #[test]
    fn zero_intensity_is_pixel_exact() {
        let source = Rgb32FImage::from_fn(17, 13, |x, y| {
            let value = ((x * 17 + y * 11) % 97) as f32 / 96.0;
            Rgb([value, value * 0.8, value * 0.6])
        });
        let output = run_bm3d_core(&source, 0.0, None).to_rgb32f();
        assert_eq!(source.as_raw(), output.as_raw());
    }

    #[test]
    fn deterministic_bm3d_reduces_noise_without_destroying_primary_edge() {
        let clean = Rgb32FImage::from_fn(48, 48, |x, _| {
            let value = if x < 24 { 0.25 } else { 0.75 };
            Rgb([value, value, value])
        });
        let noisy = Rgb32FImage::from_fn(48, 48, |x, y| {
            let clean_value = if x < 24 { 0.25 } else { 0.75 };
            let hash = x
                .wrapping_mul(1_103_515_245)
                .wrapping_add(y.wrapping_mul(12_345))
                .rotate_left((y % 31) + 1);
            let noise = (hash % 10_001) as f32 / 10_000.0 * 0.16 - 0.08;
            let value = (clean_value + noise).clamp(0.0, 1.0);
            Rgb([value, value, value])
        });

        let denoised = run_bm3d_core(&noisy, 0.25, None).to_rgb32f();
        assert!(
            mean_squared_error(&denoised, &clean) < mean_squared_error(&noisy, &clean),
            "denoise output should be closer to the deterministic clean fixture"
        );
        assert!(
            denoised
                .pixels()
                .flat_map(|pixel| pixel.0)
                .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        );

        let left_mean = denoised
            .enumerate_pixels()
            .filter(|(x, _, _)| *x < 20)
            .map(|(_, _, pixel)| pixel[0])
            .sum::<f32>()
            / (20 * 48) as f32;
        let right_mean = denoised
            .enumerate_pixels()
            .filter(|(x, _, _)| *x >= 28)
            .map(|(_, _, pixel)| pixel[0])
            .sum::<f32>()
            / (20 * 48) as f32;
        assert!(right_mean - left_mean > 0.35);
    }

    #[test]
    fn higher_intensity_produces_a_materially_stronger_result() {
        let source = Rgb32FImage::from_fn(48, 48, |x, y| {
            let base = if x < 24 { 0.28 } else { 0.72 };
            let hash = x
                .wrapping_mul(747_796_405)
                .wrapping_add(y.wrapping_mul(2_891_336_453))
                .rotate_left((x + y) % 31 + 1);
            let noise = (hash % 10_001) as f32 / 10_000.0 * 0.18 - 0.09;
            let red = (base + noise).clamp(0.0, 1.0);
            let green = (base * 0.92 + noise * 0.75).clamp(0.0, 1.0);
            let blue = (base * 0.78 + noise * 0.55).clamp(0.0, 1.0);
            Rgb([red, green, blue])
        });

        let low = run_bm3d_core(&source, 0.15, None).to_rgb32f();
        let high = run_bm3d_core(&source, 0.85, None).to_rgb32f();
        let low_change = mean_squared_error(&low, &source);
        let high_change = mean_squared_error(&high, &source);
        let result_separation = mean_squared_error(&low, &high);

        assert!(
            high_change > low_change * 1.2,
            "high strength should alter the source more (low={low_change}, high={high_change})"
        );
        assert!(
            result_separation > 0.000_01,
            "15 and 85 strength should produce visibly distinct pixels (mse={result_separation})"
        );
    }
}
