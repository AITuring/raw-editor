//! Folder listing commands.
//!
//! Lists images in a folder or recursively, groups files that share a stem,
//! summarises folder contents and records whether the folder is on a
//! rotational disk.

use crate::app_settings::load_settings;
use crate::formats::is_supported_image_file;
use anyhow::Result;
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::thread;
use sysinfo::Disks;
use tauri::AppHandle;
use tauri::Manager;
use walkdir::WalkDir;

use super::{
    ImageFileMetadata, enqueue_metadata, is_cloud_placeholder, parse_virtual_path,
    resolve_image_metadata, resolve_xmp_path,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImageFile {
    pub path: String,
    pub(super) modified: u64,
    pub(super) is_edited: bool,
    pub(super) rating: u8,
    pub(super) tags: Option<Vec<String>>,
    pub(super) exif: Option<HashMap<String, String>>,
    pub(super) is_virtual_copy: bool,
    pub(super) is_cloud_placeholder: bool,
    pub(super) is_raw: bool,
    pub(super) group_id: Option<String>,
}

pub(super) fn make_group_key(source_path: &Path) -> String {
    let parent = source_path.parent().unwrap_or(Path::new(""));
    let stem = source_path.file_stem().unwrap_or_default();
    format!("{}/{}", parent.to_string_lossy(), stem.to_string_lossy())
}

pub(super) fn assign_group_ids(
    files: &mut [ImageFile],
    settings: &crate::app_settings::AppSettings,
) {
    let require_matching_exif = settings.require_matching_exif.unwrap_or(false);
    let group_edited_files = settings.group_edited_files.unwrap_or(true);

    #[derive(Clone)]
    struct Candidate {
        index: usize,
        source_path: PathBuf,
        key: String,
    }

    let candidates: Vec<Candidate> = files
        .iter()
        .enumerate()
        .filter(|(_, file)| !file.is_virtual_copy && (group_edited_files || !file.is_edited))
        .map(|(index, file)| {
            let (source_path, _) = parse_virtual_path(&file.path);
            let key = make_group_key(&source_path);
            Candidate {
                index,
                source_path,
                key,
            }
        })
        .collect();

    let mut stem_groups: HashMap<String, Vec<Candidate>> = HashMap::new();
    for candidate in candidates {
        stem_groups
            .entry(candidate.key.clone())
            .or_default()
            .push(candidate);
    }

    if require_matching_exif {
        let groupable_paths: Vec<PathBuf> = stem_groups
            .values()
            .filter(|candidates| candidates.len() >= 2)
            .flat_map(|candidates| candidates.iter().map(|c| c.source_path.clone()))
            .collect();
        let exif_dates: HashMap<PathBuf, Option<chrono::DateTime<chrono::Utc>>> = groupable_paths
            .par_iter()
            .map(|p| {
                (
                    p.clone(),
                    crate::exif_processing::try_get_exif_creation_date(p),
                )
            })
            .collect();

        stem_groups.retain(|_, candidates| {
            if candidates.len() < 2 {
                return false;
            }
            let first = exif_dates
                .get(&candidates[0].source_path)
                .copied()
                .flatten();
            if first.is_none() {
                return false;
            }
            candidates
                .iter()
                .skip(1)
                .all(|c| exif_dates.get(&c.source_path).copied().flatten() == first)
        });
    } else {
        stem_groups.retain(|_, candidates| candidates.len() >= 2);
    }

    for (key, candidates) in stem_groups {
        for candidate in candidates {
            files[candidate.index].group_id = Some(key.clone());
        }
    }
}

pub(super) fn match_disk_kind(disks: &Disks, canonical: &Path) -> Option<bool> {
    let mut best_match: Option<(&Path, bool)> = None;

    for disk in disks.list() {
        let mount_point = disk.mount_point();
        if canonical.starts_with(mount_point) {
            let is_longer_match = best_match
                .map(|(current, _)| mount_point.as_os_str().len() > current.as_os_str().len())
                .unwrap_or(true);
            if is_longer_match {
                best_match = Some((mount_point, disk.kind() == sysinfo::DiskKind::HDD));
            }
        }
    }

    best_match.map(|(_, is_hdd)| is_hdd)
}

pub(super) fn update_rotational_disk_flag(path: &str, app_handle: &AppHandle) {
    let state = app_handle.state::<crate::AppState>();
    let Ok(canonical) = Path::new(path).canonicalize() else {
        return;
    };

    let cached_match = {
        let cache = state.disks_cache.lock().unwrap();
        cache
            .as_ref()
            .and_then(|disks| match_disk_kind(disks, &canonical))
    };

    match cached_match {
        Some(is_hdd) => {
            state
                .thumbnail_manager
                .rotational_disk
                .store(is_hdd, Ordering::Relaxed);
        }
        None => {
            if !state.disks_cache_refreshing.swap(true, Ordering::Relaxed) {
                let refresh_app_handle = app_handle.clone();
                thread::spawn(move || {
                    let disks = Disks::new_with_refreshed_list();
                    let state = refresh_app_handle.state::<crate::AppState>();
                    *state.disks_cache.lock().unwrap() = Some(disks);
                    state.disks_cache_refreshing.store(false, Ordering::Relaxed);
                });
            }
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FolderContentSummary {
    pub(super) total_files: usize,
    pub(super) supported_files: usize,
    pub(super) unreadable_entries: usize,
}

#[tauri::command]
pub fn inspect_folder_contents(
    path: String,
    recursive: bool,
) -> std::result::Result<FolderContentSummary, String> {
    let root = Path::new(&path);
    let mut summary = FolderContentSummary {
        total_files: 0,
        supported_files: 0,
        unreadable_entries: 0,
    };

    if recursive {
        for entry in WalkDir::new(root) {
            match entry {
                Ok(entry) if entry.file_type().is_file() => {
                    summary.total_files += 1;
                    if is_supported_image_file(entry.path()) {
                        summary.supported_files += 1;
                    }
                }
                Ok(_) => {}
                Err(_) => summary.unreadable_entries += 1,
            }
        }
    } else {
        let entries = fs::read_dir(root).map_err(|error| error.to_string())?;
        for entry in entries {
            match entry {
                Ok(entry) => match entry.file_type() {
                    Ok(file_type) if file_type.is_file() => {
                        summary.total_files += 1;
                        if is_supported_image_file(entry.path()) {
                            summary.supported_files += 1;
                        }
                    }
                    Ok(_) => {}
                    Err(_) => summary.unreadable_entries += 1,
                },
                Err(_) => summary.unreadable_entries += 1,
            }
        }
    }

    Ok(summary)
}

#[tauri::command]
pub fn list_images_in_dir(path: String, app_handle: AppHandle) -> Result<Vec<ImageFile>, String> {
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);

    update_rotational_disk_flag(&path, &app_handle);

    let entries = fs::read_dir(&path).map_err(|e| e.to_string())?;
    let mut images = Vec::new();

    for entry in entries.filter_map(Result::ok) {
        let entry_path = entry.path();
        let file_name = entry
            .file_name()
            .into_string()
            .unwrap_or_else(|os| os.to_string_lossy().into_owned());

        if is_supported_image_file(&file_name) {
            images.push(entry_path);
        }
    }

    let tasks: Vec<_> = images
        .into_iter()
        .map(|path_buf| {
            let mut sidecars = vec![None];
            sidecars.extend(
                crate::sidecar_storage::virtual_copy_ids(&path_buf)
                    .into_iter()
                    .map(Some),
            );
            let path_str = path_buf.to_string_lossy().into_owned();
            (path_str, path_buf, sidecars)
        })
        .collect();

    let mut result_list: Vec<ImageFile> = tasks
        .into_par_iter()
        .flat_map(|(path_str, path_buf, sidecars)| {
            let modified = fs::metadata(&path_buf)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let is_cloud_placeholder = is_cloud_placeholder(&path_buf);

            let mut file_results = Vec::with_capacity(sidecars.len());

            for copy_id_opt in sidecars {
                let (virtual_path, is_virtual_copy) = match &copy_id_opt {
                    Some(id) => (format!("{}?vc={}", path_str, id), true),
                    None => (path_str.clone(), false),
                };

                let sidecar_path =
                    crate::sidecar_storage::sidecar_path(&path_buf, copy_id_opt.as_deref());

                let xmp_is_placeholder = enable_xmp_sync
                    && resolve_xmp_path(&path_buf)
                        .is_some_and(|p| crate::file_management::is_cloud_placeholder(&p));

                let metadata = if crate::file_management::is_cloud_placeholder(&sidecar_path)
                    || xmp_is_placeholder
                {
                    enqueue_metadata(
                        &app_handle,
                        virtual_path.clone(),
                        path_buf.clone(),
                        sidecar_path.clone(),
                    );
                    ImageFileMetadata {
                        is_edited: false,
                        tags: None,
                        rating: 0,
                        is_raw: crate::formats::is_raw_file(&path_buf),
                    }
                } else {
                    resolve_image_metadata(&path_buf, &sidecar_path, enable_xmp_sync, &settings)
                };

                file_results.push(ImageFile {
                    path: virtual_path,
                    modified,
                    is_edited: metadata.is_edited,
                    tags: metadata.tags,
                    exif: None,
                    is_virtual_copy,
                    is_raw: metadata.is_raw,
                    group_id: None,
                    rating: metadata.rating,
                    is_cloud_placeholder,
                });
            }

            file_results
        })
        .collect();

    assign_group_ids(&mut result_list, &settings);
    Ok(result_list)
}

#[tauri::command]
pub fn list_images_recursive(
    path: String,
    app_handle: AppHandle,
) -> Result<Vec<ImageFile>, String> {
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let enable_xmp_sync = settings.enable_xmp_sync.unwrap_or(false);

    update_rotational_disk_flag(&path, &app_handle);

    let root_path = Path::new(&path);
    let mut images = Vec::new();

    for entry in WalkDir::new(root_path).into_iter().filter_map(Result::ok) {
        let entry_path = entry.path();
        if !entry_path.is_file() {
            continue;
        }

        if is_supported_image_file(entry_path.to_string_lossy().as_ref()) {
            images.push(entry_path.to_path_buf());
        }
    }

    let tasks: Vec<_> = images
        .into_iter()
        .map(|path_buf| {
            let mut sidecars = vec![None];
            sidecars.extend(
                crate::sidecar_storage::virtual_copy_ids(&path_buf)
                    .into_iter()
                    .map(Some),
            );
            let path_str = path_buf.to_string_lossy().into_owned();
            (path_str, path_buf, sidecars)
        })
        .collect();

    let mut result_list: Vec<ImageFile> = tasks
        .into_par_iter()
        .flat_map(|(path_str, path_buf, sidecars)| {
            let modified = fs::metadata(&path_buf)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let is_cloud_placeholder = is_cloud_placeholder(&path_buf);

            let mut file_results = Vec::with_capacity(sidecars.len());

            for copy_id_opt in sidecars {
                let (virtual_path, is_virtual_copy) = match &copy_id_opt {
                    Some(id) => (format!("{}?vc={}", path_str, id), true),
                    None => (path_str.clone(), false),
                };

                let sidecar_path =
                    crate::sidecar_storage::sidecar_path(&path_buf, copy_id_opt.as_deref());

                let xmp_is_placeholder = enable_xmp_sync
                    && resolve_xmp_path(&path_buf)
                        .is_some_and(|p| crate::file_management::is_cloud_placeholder(&p));

                let metadata = if crate::file_management::is_cloud_placeholder(&sidecar_path)
                    || xmp_is_placeholder
                {
                    enqueue_metadata(
                        &app_handle,
                        virtual_path.clone(),
                        path_buf.clone(),
                        sidecar_path.clone(),
                    );
                    ImageFileMetadata {
                        is_edited: false,
                        tags: None,
                        rating: 0,
                        is_raw: crate::formats::is_raw_file(&path_buf),
                    }
                } else {
                    resolve_image_metadata(&path_buf, &sidecar_path, enable_xmp_sync, &settings)
                };

                file_results.push(ImageFile {
                    path: virtual_path,
                    modified,
                    is_edited: metadata.is_edited,
                    tags: metadata.tags,
                    exif: None,
                    is_virtual_copy,
                    is_raw: metadata.is_raw,
                    group_id: None,
                    rating: metadata.rating,
                    is_cloud_placeholder,
                });
            }

            file_results
        })
        .collect();

    assign_group_ids(&mut result_list, &settings);
    Ok(result_list)
}
