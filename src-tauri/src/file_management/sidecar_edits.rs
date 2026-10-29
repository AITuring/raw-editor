//! Commands that edit image sidecars.
//!
//! Saves and resets adjustments, applies auto adjustments, ratings and colour
//! labels, regenerating thumbnails and syncing XMP where enabled.

use crate::AppState;
use crate::app_settings::load_settings;
use crate::formats::is_supported_image_file;
use crate::gpu_processing;
use crate::image_loader;
use crate::image_processing::ImageMetadata;
use crate::image_processing::auto_results_to_json;
use crate::image_processing::perform_auto_analysis;
use crate::tagging::COLOR_TAG_PREFIX;
use anyhow::Result;
use image::DynamicImage;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::thread;
use tauri::AppHandle;
use tauri::Manager;
use walkdir::WalkDir;

use super::{
    add_to_thumbnail_queue, emit_thumbnail_cache_setup_error, emit_thumbnail_generated,
    generate_single_thumbnail_and_cache, increment_thumbnail_progress, parse_virtual_path,
    resolve_thumbnail_cache_dir, sync_metadata_from_xmp, sync_metadata_to_xmp,
};

pub fn resolve_lens_params_in_adjustments(
    adjustments: &mut Value,
    exif_data: &Option<HashMap<String, String>>,
    lens_db: Option<&crate::lens_correction::LensDatabase>,
) {
    if let Some(map) = adjustments.as_object_mut() {
        let mode = map
            .get("lensCorrectionMode")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                if map
                    .get("lensMaker")
                    .and_then(|value| value.as_str())
                    .is_some()
                    && map
                        .get("lensModel")
                        .and_then(|value| value.as_str())
                        .is_some()
                {
                    "manual"
                } else {
                    "auto"
                }
            });

        if mode == "auto" {
            if let Some(exif) = exif_data {
                let exif_maker = exif.get("Make").map(|s| s.as_str()).unwrap_or("");
                let exif_model = exif.get("LensModel").map(|s| s.as_str()).unwrap_or("");
                if let Some(db) = lens_db {
                    if let Some((detected_maker, detected_model)) =
                        crate::lens_correction::find_best_lens_match(db, exif_maker, exif_model)
                    {
                        map.insert(
                            "lensMaker".to_string(),
                            serde_json::to_value(&detected_maker).unwrap(),
                        );
                        map.insert(
                            "lensModel".to_string(),
                            serde_json::to_value(&detected_model).unwrap(),
                        );
                    } else {
                        map.remove("lensMaker");
                        map.remove("lensModel");
                    }
                }
            } else {
                map.remove("lensMaker");
                map.remove("lensModel");
            }
        }

        if let Some(db) = lens_db {
            let has_valid_lens = match (
                map.get("lensMaker").and_then(|v| v.as_str()),
                map.get("lensModel").and_then(|v| v.as_str()),
            ) {
                (Some(maker), Some(model)) if !maker.is_empty() && !model.is_empty() => {
                    let mut focal_length = 50.0;
                    let mut aperture = None;
                    let mut distance = None;

                    if let Some(exif) = exif_data {
                        if let Some(fl_str) = exif
                            .get("FocalLength")
                            .or(exif.get("FocalLengthIn35mmFilm"))
                            && let Ok(fl) = fl_str.replace(" mm", "").trim().parse::<f32>()
                        {
                            focal_length = fl;
                        }
                        if let Some(ap_str) = exif.get("ApertureValue").or(exif.get("FNumber"))
                            && let Ok(ap) = ap_str.replace("f/", "").trim().parse::<f32>()
                        {
                            aperture = Some(ap);
                        }
                        if let Some(dist_str) = exif.get("SubjectDistance")
                            && let Ok(dist) = dist_str.replace(" m", "").trim().parse::<f32>()
                        {
                            distance = Some(dist);
                        }
                    }

                    if let Some(params) = crate::lens_correction::resolve_lens_params(
                        db,
                        maker,
                        model,
                        focal_length,
                        aperture,
                        distance,
                    ) {
                        map.insert(
                            "lensDistortionParams".to_string(),
                            serde_json::to_value(params).unwrap(),
                        );
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            };

            if !has_valid_lens {
                map.remove("lensDistortionParams");
            }
        }
    }
}

#[tauri::command]
pub fn save_metadata_and_update_thumbnail(
    path: String,
    adjustments: Value,
    app_handle: AppHandle,
    state: tauri::State<AppState>,
) -> Result<(), String> {
    let (source_path, sidecar_path) = parse_virtual_path(&path);

    let mut metadata = crate::exif_processing::load_sidecar(&sidecar_path);

    let mut final_adjustments = adjustments;
    {
        let lens_db_guard = state.lens_db.lock().unwrap();
        resolve_lens_params_in_adjustments(
            &mut final_adjustments,
            &metadata.exif,
            lens_db_guard.as_deref(),
        );
    }

    metadata.adjustments = final_adjustments;

    crate::exif_processing::save_sidecar(&sidecar_path, &metadata)?;

    if let Ok(settings) = load_settings(app_handle.clone())
        && settings.enable_xmp_sync.unwrap_or(false)
    {
        let create_if_missing = settings.create_xmp_if_missing.unwrap_or(false);
        sync_metadata_to_xmp(&source_path, &metadata, create_if_missing);
    }

    let loaded_image_lock = state.original_image.lock().unwrap();
    let preloaded_image_option = if let Some(loaded_image) = loaded_image_lock.as_ref() {
        if loaded_image.path == path {
            Some(loaded_image.image.clone())
        } else {
            None
        }
    } else {
        None
    };
    drop(loaded_image_lock);

    let gpu_context = gpu_processing::get_or_init_gpu_context(&state, &app_handle).ok();
    let app_handle_clone = app_handle.clone();
    let path_clone = path.clone();

    add_to_thumbnail_queue(&state, 1, &app_handle);

    thread::spawn(move || {
        let state = app_handle_clone.state::<AppState>();
        let settings = load_settings(app_handle_clone.clone()).unwrap_or_default();

        let thumb_cache_dir = match resolve_thumbnail_cache_dir(&app_handle_clone) {
            Ok(dir) => dir,
            Err(e) => {
                log::warn!(
                    "Unable to initialize thumbnail cache directory for '{}': {}",
                    path_clone,
                    e
                );
                emit_thumbnail_cache_setup_error(&app_handle_clone, &path_clone, &e);
                increment_thumbnail_progress(&state, &app_handle_clone);
                return;
            }
        };

        let result = generate_single_thumbnail_and_cache(
            &path_clone,
            &thumb_cache_dir,
            gpu_context.as_ref(),
            preloaded_image_option.as_deref(),
            true,
            &app_handle_clone,
            &settings,
        );

        if let Some((thumbnail_path, rating, is_edited)) = result {
            emit_thumbnail_generated(
                &app_handle_clone,
                &path_clone,
                &thumbnail_path,
                rating,
                is_edited,
            );
        }

        increment_thumbnail_progress(&state, &app_handle_clone);
    });

    Ok(())
}

#[tauri::command]
pub async fn apply_adjustments_to_paths(
    paths: Vec<String>,
    adjustments: Value,
    app_handle: AppHandle,
) -> Result<(), String> {
    let state = app_handle.state::<AppState>();
    add_to_thumbnail_queue(&state, paths.len(), &app_handle);

    tauri::async_runtime::spawn_blocking(move || {
        let settings = load_settings(app_handle.clone()).unwrap_or_default();
        let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);
        let create_xmp_if_missing = settings.create_xmp_if_missing.unwrap_or(false);

        let lens_db = app_handle
            .state::<AppState>()
            .lens_db
            .lock()
            .unwrap()
            .clone();

        paths.par_iter().for_each(|path| {
            let (_, sidecar_path) = parse_virtual_path(path);

            let mut existing_metadata = crate::exif_processing::load_sidecar(&sidecar_path);

            let mut new_adjustments = existing_metadata.adjustments;
            if new_adjustments.is_null() {
                new_adjustments = serde_json::json!({});
            }

            if let (Some(new_map), Some(pasted_map)) =
                (new_adjustments.as_object_mut(), adjustments.as_object())
            {
                for (k, v) in pasted_map {
                    new_map.insert(k.clone(), v.clone());
                }
            }

            resolve_lens_params_in_adjustments(
                &mut new_adjustments,
                &existing_metadata.exif,
                lens_db.as_deref(),
            );

            existing_metadata.adjustments = new_adjustments;

            let _ = crate::exif_processing::save_sidecar(&sidecar_path, &existing_metadata);

            if enable_xmp_sync {
                let source_path = parse_virtual_path(path).0;
                sync_metadata_to_xmp(&source_path, &existing_metadata, create_xmp_if_missing);
            }
        });

        let state = app_handle.state::<AppState>();
        let thumb_cache_dir = match resolve_thumbnail_cache_dir(&app_handle) {
            Ok(dir) => dir,
            Err(e) => {
                log::warn!("Unable to initialize thumbnail cache directory: {}", e);
                for path in &paths {
                    emit_thumbnail_cache_setup_error(&app_handle, path, &e);
                }
                for _ in 0..paths.len() {
                    increment_thumbnail_progress(&state, &app_handle);
                }
                return;
            }
        };

        let gpu_context = gpu_processing::get_or_init_gpu_context(&state, &app_handle).ok();

        paths.par_iter().for_each(|path_str| {
            let result = generate_single_thumbnail_and_cache(
                path_str,
                &thumb_cache_dir,
                gpu_context.as_ref(),
                None,
                true,
                &app_handle,
                &settings,
            );

            if let Some((thumbnail_path, rating, is_edited)) = result {
                emit_thumbnail_generated(&app_handle, path_str, &thumbnail_path, rating, is_edited);
            }

            increment_thumbnail_progress(&state, &app_handle);
        });
    });

    Ok(())
}

#[tauri::command]
pub async fn reset_adjustments_for_paths(
    paths: Vec<String>,
    app_handle: AppHandle,
) -> Result<(), String> {
    let state = app_handle.state::<AppState>();
    add_to_thumbnail_queue(&state, paths.len(), &app_handle);

    tauri::async_runtime::spawn_blocking(move || {
        let settings = load_settings(app_handle.clone()).unwrap_or_default();
        let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);
        let create_xmp_if_missing = settings.create_xmp_if_missing.unwrap_or(false);

        paths.par_iter().for_each(|path| {
            let (_, sidecar_path) = parse_virtual_path(path);

            let mut existing_metadata = crate::exif_processing::load_sidecar(&sidecar_path);

            existing_metadata.adjustments = serde_json::json!({});

            let _ = crate::exif_processing::save_sidecar(&sidecar_path, &existing_metadata);

            if enable_xmp_sync {
                let source_path = parse_virtual_path(path).0;
                sync_metadata_to_xmp(&source_path, &existing_metadata, create_xmp_if_missing);
            }
        });

        let state = app_handle.state::<AppState>();
        let thumb_cache_dir = match resolve_thumbnail_cache_dir(&app_handle) {
            Ok(dir) => dir,
            Err(e) => {
                log::warn!("Unable to initialize thumbnail cache directory: {}", e);
                for path in &paths {
                    emit_thumbnail_cache_setup_error(&app_handle, path, &e);
                }
                for _ in 0..paths.len() {
                    increment_thumbnail_progress(&state, &app_handle);
                }
                return;
            }
        };

        let gpu_context = gpu_processing::get_or_init_gpu_context(&state, &app_handle).ok();

        paths.par_iter().for_each(|path_str| {
            let result = generate_single_thumbnail_and_cache(
                path_str,
                &thumb_cache_dir,
                gpu_context.as_ref(),
                None,
                true,
                &app_handle,
                &settings,
            );

            if let Some((thumbnail_path, rating, is_edited)) = result {
                emit_thumbnail_generated(&app_handle, path_str, &thumbnail_path, rating, is_edited);
            }

            increment_thumbnail_progress(&state, &app_handle);
        });
    });

    Ok(())
}

#[tauri::command]
pub async fn apply_auto_adjustments_to_paths(
    paths: Vec<String>,
    app_handle: AppHandle,
) -> Result<(), String> {
    let state = app_handle.state::<AppState>();
    add_to_thumbnail_queue(&state, paths.len(), &app_handle);

    tauri::async_runtime::spawn_blocking(move || {
        let settings = load_settings(app_handle.clone()).unwrap_or_default();
        let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);
        let create_xmp_if_missing = settings.create_xmp_if_missing.unwrap_or(false);

        let state = app_handle.state::<AppState>();
        let thumb_cache_dir = match resolve_thumbnail_cache_dir(&app_handle) {
            Ok(dir) => dir,
            Err(e) => {
                log::warn!("Unable to initialize thumbnail cache directory: {}", e);
                for path in &paths {
                    emit_thumbnail_cache_setup_error(&app_handle, path, &e);
                }
                for _ in 0..paths.len() {
                    increment_thumbnail_progress(&state, &app_handle);
                }
                return;
            }
        };

        let gpu_context = gpu_processing::get_or_init_gpu_context(&state, &app_handle).ok();

        paths.par_iter().for_each(|path| {
            let loaded_image: Option<DynamicImage> = (|| -> Result<DynamicImage, String> {
                let (source_path, sidecar_path) = parse_virtual_path(path);
                let source_path_str = source_path.to_string_lossy().to_string();

                let file_bytes = fs::read(&source_path).map_err(|e| e.to_string())?;
                let image = image_loader::load_base_image_from_bytes(
                    &file_bytes,
                    &source_path_str,
                    true,
                    &settings,
                    None,
                )
                .map_err(|e| e.to_string())?;

                let auto_results = perform_auto_analysis(&image);
                let auto_adjustments_json = auto_results_to_json(&auto_results);

                let mut existing_metadata = crate::exif_processing::load_sidecar(&sidecar_path);

                if existing_metadata.adjustments.is_null() {
                    existing_metadata.adjustments = serde_json::json!({});
                }

                if let (Some(existing_map), Some(auto_map)) = (
                    existing_metadata.adjustments.as_object_mut(),
                    auto_adjustments_json.as_object(),
                ) {
                    for (k, v) in auto_map {
                        if k == "sectionVisibility" {
                            if let Some(existing_vis_val) = existing_map.get_mut(k) {
                                if let (Some(existing_vis), Some(auto_vis)) =
                                    (existing_vis_val.as_object_mut(), v.as_object())
                                {
                                    for (vis_k, vis_v) in auto_vis {
                                        existing_vis.insert(vis_k.clone(), vis_v.clone());
                                    }
                                }
                            } else {
                                existing_map.insert(k.clone(), v.clone());
                            }
                        } else {
                            existing_map.insert(k.clone(), v.clone());
                        }
                    }
                }

                let _ = crate::exif_processing::save_sidecar(&sidecar_path, &existing_metadata);

                if enable_xmp_sync {
                    sync_metadata_to_xmp(&source_path, &existing_metadata, create_xmp_if_missing);
                }
                Ok(image)
            })()
            .map_err(|e| eprintln!("Failed to apply auto adjustments to {}: {}", path, e))
            .ok();

            let result = generate_single_thumbnail_and_cache(
                path,
                &thumb_cache_dir,
                gpu_context.as_ref(),
                loaded_image.as_ref(),
                true,
                &app_handle,
                &settings,
            );

            if let Some((thumbnail_path, rating, is_edited)) = result {
                emit_thumbnail_generated(&app_handle, path, &thumbnail_path, rating, is_edited);
            }

            increment_thumbnail_progress(&state, &app_handle);
        });
    });

    Ok(())
}

#[tauri::command]
pub fn set_color_label_for_paths(
    paths: Vec<String>,
    color: Option<String>,
    app_handle: AppHandle,
) -> Result<(), String> {
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);
    let create_xmp_if_missing = settings.create_xmp_if_missing.unwrap_or(false);

    paths.par_iter().for_each(|path| {
        let (_, sidecar_path) = parse_virtual_path(path);

        let mut metadata = crate::exif_processing::load_sidecar(&sidecar_path);

        let mut tags = metadata.tags.unwrap_or_default();
        tags.retain(|tag| !tag.starts_with(COLOR_TAG_PREFIX));

        if let Some(c) = &color
            && !c.is_empty()
        {
            tags.push(format!("{}{}", COLOR_TAG_PREFIX, c));
        }

        if tags.is_empty() {
            metadata.tags = None;
        } else {
            metadata.tags = Some(tags);
        }

        let _ = crate::exif_processing::save_sidecar(&sidecar_path, &metadata);

        if enable_xmp_sync {
            let source_path = parse_virtual_path(path).0;
            sync_metadata_to_xmp(&source_path, &metadata, create_xmp_if_missing);
        }
    });

    Ok(())
}

#[tauri::command]
pub fn set_rating_for_paths(
    paths: Vec<String>,
    rating: u8,
    app_handle: AppHandle,
) -> Result<(), String> {
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);
    let create_xmp_if_missing = settings.create_xmp_if_missing.unwrap_or(false);

    paths.par_iter().for_each(|path| {
        let (_, sidecar_path) = parse_virtual_path(path);

        let mut metadata = crate::exif_processing::load_sidecar(&sidecar_path);

        metadata.rating = rating;

        let _ = crate::exif_processing::save_sidecar(&sidecar_path, &metadata);

        if enable_xmp_sync {
            let source_path = parse_virtual_path(path).0;
            sync_metadata_to_xmp(&source_path, &metadata, create_xmp_if_missing);
        }
    });

    Ok(())
}

#[tauri::command]
pub fn load_metadata(path: String, app_handle: AppHandle) -> Result<ImageMetadata, String> {
    let settings = load_settings(app_handle).unwrap_or_default();
    let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);

    let (source_path, sidecar_path) = parse_virtual_path(&path);
    let mut metadata = crate::exif_processing::load_sidecar(&sidecar_path);

    if enable_xmp_sync && sync_metadata_from_xmp(&source_path, &mut metadata) {
        let _ = crate::exif_processing::save_sidecar(&sidecar_path, &metadata);
    }

    Ok(metadata)
}

#[tauri::command]
pub fn clear_all_sidecars(root_path: String) -> Result<usize, String> {
    let root_path = Path::new(&root_path);
    if !root_path.exists() {
        return Err(format!("Root path does not exist: {}", root_path.display()));
    }

    let mut deleted_count = 0;
    let paths: Vec<PathBuf> = WalkDir::new(root_path)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.is_file())
        .collect();

    for image_path in paths.iter().filter(|path| is_supported_image_file(path)) {
        deleted_count += crate::sidecar_storage::remove_sidecars(image_path)?;
    }

    // Remove orphaned legacy sidecars whose source image no longer exists.
    for path in paths {
        if let Some(extension) = path.extension()
            && (extension == "rrdata" || extension == "rrexif")
            && path.exists()
        {
            fs::remove_file(&path).map_err(|error| {
                format!(
                    "Failed to delete legacy sidecar {}: {}",
                    path.display(),
                    error
                )
            })?;
            deleted_count += 1;
        }
    }

    Ok(deleted_count)
}
