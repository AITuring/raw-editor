//! Image metadata resolution and the background metadata workers.
//!
//! Resolves edited state, tags and rating from sidecars (with optional XMP
//! sync) and emits them to the frontend from a small worker pool.

use crate::PendingMetadata;
use crate::app_settings::AppSettings;
use crate::app_settings::load_settings;
use std::path::Path;
use std::path::PathBuf;
use tauri::AppHandle;
use tauri::Emitter;
use tauri::Manager;

use super::sync_metadata_from_xmp;

pub(super) struct ImageFileMetadata {
    pub(super) is_edited: bool,
    pub(super) tags: Option<Vec<String>>,
    pub(super) rating: u8,
    pub(super) is_raw: bool,
}

pub(super) fn resolve_image_metadata(
    image_path: &Path,
    sidecar_path: &Path,
    enable_xmp_sync: bool,
    settings: &AppSettings,
) -> ImageFileMetadata {
    if crate::exif_processing::get_rrexif_path(image_path).exists() {
        let _ = crate::exif_processing::read_rrexif_sidecar(image_path);
    }
    let mut metadata = crate::exif_processing::load_sidecar(sidecar_path);

    if enable_xmp_sync && sync_metadata_from_xmp(image_path, &mut metadata) {
        let _ = crate::exif_processing::save_sidecar(sidecar_path, &metadata);
    }

    let is_raw = crate::formats::is_raw_file(image_path);
    let tm_override = crate::image_processing::resolve_tonemapper_override(settings, is_raw);
    let is_edited =
        crate::image_processing::is_image_edited(&metadata.adjustments, is_raw, tm_override);
    ImageFileMetadata {
        is_edited,
        tags: metadata.tags,
        rating: metadata.rating,
        is_raw,
    }
}

pub(super) fn emit_image_metadata_loaded(
    app_handle: &AppHandle,
    path: &str,
    rating: u8,
    is_edited: bool,
    tags: &Option<Vec<String>>,
) {
    let _ = app_handle.emit(
        "image-metadata-loaded",
        serde_json::json!({ "path": path, "rating": rating, "is_edited": is_edited, "tags": tags }),
    );
}

pub(super) fn enqueue_metadata(
    app_handle: &AppHandle,
    virtual_path: String,
    image_path: PathBuf,
    sidecar_path: PathBuf,
) {
    let state = app_handle.state::<crate::AppState>();
    let manager = &state.metadata_manager;

    let mut pending = manager.pending.lock().unwrap();
    if !pending.insert(sidecar_path.clone()) {
        return;
    }
    drop(pending);

    manager.queue.lock().unwrap().push_back(PendingMetadata {
        virtual_path,
        image_path,
        sidecar_path,
    });
    manager.cvar.notify_one();
}

// Not compute-heavy — these threads mostly block waiting on iCloud to
// materialize a file, not burning CPU — so a small fixed pool is enough and
// doesn't need a user-facing setting the way thumbnail_worker_threads does.
pub(super) const METADATA_WORKER_THREADS: usize = 4;

pub fn start_metadata_workers(app_handle: tauri::AppHandle) {
    let state = app_handle.state::<crate::AppState>();
    let manager = state.metadata_manager.clone();

    for _ in 0..METADATA_WORKER_THREADS {
        let app_clone = app_handle.clone();
        let manager_clone = manager.clone();

        std::thread::spawn(move || {
            loop {
                let item = {
                    let mut queue = manager_clone.queue.lock().unwrap();
                    while queue.is_empty() {
                        queue = manager_clone.cvar.wait(queue).unwrap();
                    }
                    queue.pop_front().unwrap()
                };

                let settings = load_settings(app_clone.clone()).unwrap_or_default();
                let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);

                let metadata = resolve_image_metadata(
                    &item.image_path,
                    &item.sidecar_path,
                    enable_xmp_sync,
                    &settings,
                );

                emit_image_metadata_loaded(
                    &app_clone,
                    &item.virtual_path,
                    metadata.rating,
                    metadata.is_edited,
                    &metadata.tags,
                );

                manager_clone
                    .pending
                    .lock()
                    .unwrap()
                    .remove(&item.sidecar_path);
            }
        });
    }
}
