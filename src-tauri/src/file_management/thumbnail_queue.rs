//! Thumbnail cache, queue and worker threads.
//!
//! Cache keys and directories, the prioritised generation queue, progress
//! events and the cache-backed thumbnail lookup used by tagging.

use crate::AppState;
use crate::app_settings::AppSettings;
use crate::app_settings::load_settings;
use crate::image_processing::GpuContext;
use anyhow::Result;
use image::DynamicImage;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use tauri::AppHandle;
use tauri::Emitter;
use tauri::Manager;

use super::{
    encode_thumbnail, enqueue_metadata, generate_thumbnail_data, is_cloud_placeholder,
    parse_virtual_path,
};

pub(super) fn resolve_thumbnail_cache_dir(
    app_handle: &AppHandle,
) -> std::result::Result<PathBuf, String> {
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?;
    let thumb_cache_dir = cache_dir.join("thumbnails");
    if !thumb_cache_dir.exists() {
        fs::create_dir_all(&thumb_cache_dir).map_err(|e| e.to_string())?;
    }
    Ok(thumb_cache_dir)
}

pub(super) fn emit_thumbnail_cache_setup_error(app_handle: &AppHandle, path: &str, reason: &str) {
    let _ = app_handle.emit(
        "thumbnail-generation-error",
        serde_json::json!({ "path": path, "reason": reason }),
    );
}

pub(super) fn compute_thumbnail_cache_hash(
    path_str: &str,
    adjustments_bytes: &[u8],
) -> Option<String> {
    let (source_path, _) = parse_virtual_path(path_str);

    let img_mod_time = fs::metadata(&source_path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();

    let mut hasher = blake3::Hasher::new();
    hasher.update(path_str.as_bytes());
    hasher.update(&img_mod_time.to_le_bytes());
    hasher.update(adjustments_bytes);
    Some(hasher.finalize().to_hex().to_string())
}

pub(super) fn generate_single_thumbnail_and_cache(
    path_str: &str,
    thumb_cache_dir: &Path,
    gpu_context: Option<&GpuContext>,
    preloaded_image: Option<&DynamicImage>,
    force_regenerate: bool,
    app_handle: &AppHandle,
    settings: &AppSettings,
) -> Option<(String, u8, bool)> {
    let (source_path, sidecar_path) = parse_virtual_path(path_str);

    let (rating, is_edited, adjustments_bytes) = if is_cloud_placeholder(&sidecar_path) {
        enqueue_metadata(
            app_handle,
            path_str.to_string(),
            source_path.clone(),
            sidecar_path.clone(),
        );
        (0, false, Vec::new())
    } else if let Ok(meta) = crate::exif_processing::load_sidecar_checked(&sidecar_path) {
        let is_raw = crate::formats::is_raw_file(path_str);
        let tm = crate::image_processing::resolve_tonemapper_override(settings, is_raw);

        (
            meta.rating,
            crate::image_processing::is_image_edited(&meta.adjustments, is_raw, tm),
            serde_json::to_vec(&meta.adjustments).unwrap_or_default(),
        )
    } else {
        (0, false, Vec::new())
    };

    let cache_hash = compute_thumbnail_cache_hash(path_str, &adjustments_bytes)?;

    let cache_filename = format!("{}.jpg", cache_hash);
    let cache_path = thumb_cache_dir.join(cache_filename);

    if !force_regenerate && cache_path.exists() {
        return Some((cache_path.to_string_lossy().into_owned(), rating, is_edited));
    }

    if is_cloud_placeholder(&source_path) {
        return None;
    }

    let target_width = settings.thumbnail_resolution.unwrap_or(720);

    if let Ok(thumb_image) =
        generate_thumbnail_data(path_str, gpu_context, preloaded_image, app_handle)
        && let Ok(thumb_data) = encode_thumbnail(&thumb_image, target_width)
    {
        let _ = fs::write(&cache_path, &thumb_data);
        return Some((cache_path.to_string_lossy().into_owned(), rating, is_edited));
    }
    None
}

pub(super) fn prefetch_source_file(path_str: &str) {
    let (source_path, _) = parse_virtual_path(path_str);
    let _ = fs::read(&source_path);
}

pub fn start_thumbnail_workers(app_handle: tauri::AppHandle) {
    let state = app_handle.state::<crate::AppState>();
    let manager = state.thumbnail_manager.clone();
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let thread_count = settings.thumbnail_worker_threads.unwrap_or(4).clamp(1, 16);

    for _ in 0..thread_count {
        let app_clone = app_handle.clone();
        let manager_clone = manager.clone();
        let worker_settings = settings.clone();

        std::thread::spawn(move || {
            loop {
                let path_to_process: String = {
                    let mut queue = manager_clone.queue.lock().unwrap();
                    while queue.is_empty() {
                        queue = manager_clone.cvar.wait(queue).unwrap();
                    }
                    let path = queue.pop_back().unwrap();

                    let mut processing = manager_clone.processing_now.lock().unwrap();
                    if processing.contains(&path) {
                        let state = app_clone.state::<crate::AppState>();
                        increment_thumbnail_progress(&state, &app_clone);
                        continue;
                    }
                    processing.insert(path.clone());
                    path
                };

                let state = app_clone.state::<crate::AppState>();
                let gpu_context =
                    crate::gpu_processing::get_or_init_gpu_context(&state, &app_clone).ok();

                if let Ok(cache_dir) = get_thumb_cache_dir(&app_clone) {
                    if manager_clone.rotational_disk.load(Ordering::Relaxed) {
                        let _io_permit = manager_clone.io_gate.lock().unwrap();
                        prefetch_source_file(&path_to_process);
                    }

                    let result = generate_single_thumbnail_and_cache(
                        &path_to_process,
                        &cache_dir,
                        gpu_context.as_ref(),
                        None,
                        false,
                        &app_clone,
                        &worker_settings,
                    );

                    if let Some((thumbnail_path, rating, is_edited)) = result {
                        emit_thumbnail_generated(
                            &app_clone,
                            &path_to_process,
                            &thumbnail_path,
                            rating,
                            is_edited,
                        );
                    }
                    increment_thumbnail_progress(&state, &app_clone);
                }
                manager_clone
                    .processing_now
                    .lock()
                    .unwrap()
                    .remove(&path_to_process);
            }
        });
    }
}

#[tauri::command]
pub fn update_thumbnail_queue(
    paths: Vec<String>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let state = app_handle.state::<crate::AppState>();

    let mut queue = state.thumbnail_manager.queue.lock().unwrap();

    if paths.is_empty() {
        queue.clear();
        let mut tracker = state.thumbnail_progress.lock().unwrap();
        tracker.total = 0;
        tracker.completed = 0;
        drop(tracker);

        let _ = app_handle.emit(
            "thumbnail-progress",
            serde_json::json!({ "current": 0, "total": 0 }),
        );
        state.thumbnail_manager.cvar.notify_all();
        return Ok(());
    }

    let mut unique_paths = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for path in paths {
        if seen.insert(path.clone()) {
            unique_paths.push(path);
        }
    }

    queue.retain(|p| !seen.contains(p));

    while queue.len() + unique_paths.len() > 500 {
        queue.pop_front();
    }

    if state
        .thumbnail_manager
        .rotational_disk
        .load(Ordering::Relaxed)
    {
        unique_paths.sort();
        for path in unique_paths.into_iter().rev() {
            queue.push_back(path);
        }
    } else {
        for path in unique_paths {
            queue.push_back(path);
        }
    }

    let queue_len = queue.len();
    drop(queue);

    let mut tracker = state.thumbnail_progress.lock().unwrap();
    tracker.total = tracker.completed + queue_len;

    let current = tracker.completed;
    let total = tracker.total;
    drop(tracker);

    let _ = app_handle.emit(
        "thumbnail-progress",
        serde_json::json!({ "current": current, "total": total }),
    );

    state.thumbnail_manager.cvar.notify_all();
    Ok(())
}

pub fn add_to_thumbnail_queue(state: &AppState, count: usize, app_handle: &AppHandle) {
    let mut tracker = state.thumbnail_progress.lock().unwrap();
    tracker.total += count;
    let current = tracker.completed;
    let total = tracker.total;
    drop(tracker);

    let _ = app_handle.emit(
        "thumbnail-progress",
        serde_json::json!({ "current": current, "total": total }),
    );
}

pub fn increment_thumbnail_progress(state: &AppState, app_handle: &AppHandle) {
    let mut tracker = state.thumbnail_progress.lock().unwrap();
    tracker.completed += 1;
    let current = tracker.completed;
    let total = tracker.total;

    if current >= total {
        tracker.total = 0;
        tracker.completed = 0;
        drop(tracker);

        let _ = app_handle.emit(
            "thumbnail-progress",
            serde_json::json!({ "current": 0, "total": 0 }),
        );
        let _ = app_handle.emit("thumbnail-generation-complete", true);
    } else {
        drop(tracker);
        let _ = app_handle.emit(
            "thumbnail-progress",
            serde_json::json!({ "current": current, "total": total }),
        );
    }
}

pub(super) fn emit_thumbnail_generated(
    app_handle: &AppHandle,
    path: &str,
    thumbnail_path: &str,
    rating: u8,
    is_edited: bool,
) {
    let _ = app_handle.emit(
        "thumbnail-generated",
        serde_json::json!({ "path": path, "thumbnailPath": thumbnail_path, "rating": rating, "is_edited": is_edited }),
    );
}

#[tauri::command]
pub fn clear_thumbnail_cache(app_handle: AppHandle) -> Result<(), String> {
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?;
    let thumb_cache_dir = cache_dir.join("thumbnails");

    if thumb_cache_dir.exists() {
        fs::remove_dir_all(&thumb_cache_dir)
            .map_err(|e| format!("Failed to remove thumbnail cache: {}", e))?;
    }

    fs::create_dir_all(&thumb_cache_dir)
        .map_err(|e| format!("Failed to recreate thumbnail cache directory: {}", e))?;

    Ok(())
}

pub fn get_thumb_cache_dir(app_handle: &AppHandle) -> Result<PathBuf, String> {
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?;
    let thumb_cache_dir = cache_dir.join("thumbnails");
    if !thumb_cache_dir.exists() {
        fs::create_dir_all(&thumb_cache_dir).map_err(|e| e.to_string())?;
    }
    Ok(thumb_cache_dir)
}

pub fn get_cache_key_hash(path_str: &str) -> Option<String> {
    let (_, sidecar_path) = parse_virtual_path(path_str);

    let adjustments_bytes = crate::exif_processing::load_sidecar_checked(&sidecar_path)
        .ok()
        .and_then(|metadata| serde_json::to_vec(&metadata.adjustments).ok())
        .unwrap_or_default();

    compute_thumbnail_cache_hash(path_str, &adjustments_bytes)
}

pub fn get_cached_or_generate_thumbnail_image(
    path_str: &str,
    app_handle: &AppHandle,
    gpu_context: Option<&GpuContext>,
) -> Result<DynamicImage> {
    let thumb_cache_dir = get_thumb_cache_dir(app_handle).map_err(|e| anyhow::anyhow!(e))?;
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let target_width = settings.thumbnail_resolution.unwrap_or(720);

    if let Some(cache_hash) = get_cache_key_hash(path_str) {
        let cache_filename = format!("{}.jpg", cache_hash);
        let cache_path = thumb_cache_dir.join(cache_filename);

        if cache_path.exists() {
            if let Ok(image) = image::open(&cache_path) {
                return Ok(image);
            }
            eprintln!(
                "Could not open cached thumbnail, regenerating: {:?}",
                cache_path
            );
        }

        let thumb_image = generate_thumbnail_data(path_str, gpu_context, None, app_handle)?;
        let thumb_data = encode_thumbnail(&thumb_image, target_width)?;
        fs::write(&cache_path, &thumb_data)?;

        Ok(thumb_image)
    } else {
        generate_thumbnail_data(path_str, gpu_context, None, app_handle)
    }
}
