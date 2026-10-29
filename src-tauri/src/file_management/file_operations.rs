//! File and folder operation commands.
//!
//! Create, rename, duplicate, copy, move and delete files and folders together
//! with their sidecars, virtual copies and album paths.

use crate::exif_processing;
use crate::formats::is_supported_image_file;
use crate::image_processing::ImageMetadata;
use anyhow::Result;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use tauri::AppHandle;
use uuid::Uuid;
use walkdir::WalkDir;

use super::{
    add_to_album, generate_filename_from_template, parse_virtual_path, sync_album_path_changes,
};

#[tauri::command]
pub fn create_folder(path: String) -> Result<(), String> {
    let path_obj = Path::new(&path);
    if let (Some(parent), Some(new_folder_name_os)) = (path_obj.parent(), path_obj.file_name())
        && let Some(new_folder_name) = new_folder_name_os.to_str()
        && parent.exists()
    {
        for entry in fs::read_dir(parent).map_err(|e| e.to_string())? {
            if let Ok(entry) = entry
                && entry.file_name().to_string_lossy().to_lowercase()
                    == new_folder_name.to_lowercase()
            {
                return Err("A folder with that name already exists.".to_string());
            }
        }
    }
    fs::create_dir_all(&path).map_err(|e| e.to_string())
}

pub(super) fn image_paths_in_tree(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.is_file() && is_supported_image_file(path))
        .collect()
}

#[tauri::command]
pub fn rename_folder(path: String, new_name: String, app_handle: AppHandle) -> Result<(), String> {
    let p = Path::new(&path);
    if !p.is_dir() {
        return Err("Path is not a directory.".to_string());
    }
    if let Some(parent) = p.parent() {
        for entry in fs::read_dir(parent).map_err(|e| e.to_string())? {
            if let Ok(entry) = entry
                && entry.file_name().to_string_lossy().to_lowercase() == new_name.to_lowercase()
                && entry.path() != p
            {
                return Err("A folder with that name already exists.".to_string());
            }
        }
        let image_paths = image_paths_in_tree(p);
        for image_path in &image_paths {
            if crate::exif_processing::get_rrexif_path(image_path).exists() {
                let _ = crate::exif_processing::read_rrexif_sidecar(image_path);
            }
        }
        let new_path = parent.join(&new_name);
        fs::rename(p, &new_path).map_err(|e| e.to_string())?;

        for old_image_path in image_paths {
            let Ok(relative_path) = old_image_path.strip_prefix(p) else {
                continue;
            };
            let new_image_path = new_path.join(relative_path);
            if let Err(error) =
                crate::sidecar_storage::move_sidecars(&old_image_path, &new_image_path)
            {
                log::error!(
                    "Failed to move private sidecars after renaming folder for {}: {}",
                    old_image_path.display(),
                    error
                );
            }
        }

        let new_folder_str = new_path.to_string_lossy().into_owned();
        sync_album_path_changes(&app_handle, None, None, Some((&path, &new_folder_str)));

        Ok(())
    } else {
        Err("Could not determine parent directory.".to_string())
    }
}

#[tauri::command]
pub fn delete_folder(path: String, app_handle: AppHandle) -> Result<(), String> {
    let image_paths = image_paths_in_tree(Path::new(&path));

    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
    {
        if let Err(trash_error) = trash::delete(&path) {
            log::warn!(
                "Failed to move folder to trash: {}. Falling back to permanent delete.",
                trash_error
            );
            fs::remove_dir_all(&path).map_err(|e| e.to_string())?;
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        fs::remove_dir_all(&path).map_err(|e| e.to_string())?;
    }

    for image_path in image_paths {
        crate::sidecar_storage::remove_sidecars(&image_path)?;
    }

    let mut deletions = HashSet::new();
    deletions.insert(path);
    sync_album_path_changes(&app_handle, None, Some(&deletions), None);

    Ok(())
}

#[tauri::command]
pub fn duplicate_file(
    path: String,
    target_album_id: Option<String>,
    app_handle: AppHandle,
) -> Result<String, String> {
    let (source_path, source_sidecar_path) = parse_virtual_path(&path);
    if !source_path.is_file() {
        return Err("Source path is not a file.".to_string());
    }

    let parent = source_path
        .parent()
        .ok_or("Could not get parent directory")?;
    let stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Could not get file stem")?;
    let extension = source_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let mut counter = 1;
    let mut dest_path;
    loop {
        let new_stem = if counter == 1 {
            format!("{}_copy", stem)
        } else {
            format!("{}_copy_{}", stem, counter - 1)
        };
        dest_path = parent.join(format!("{}.{}", new_stem, extension));
        if !dest_path.exists() {
            break;
        }
        counter += 1;
    }

    fs::copy(&source_path, &dest_path).map_err(|e| e.to_string())?;

    if crate::exif_processing::get_rrexif_path(&source_path).exists() {
        let _ = crate::exif_processing::read_rrexif_sidecar(&source_path);
    }

    if source_sidecar_path.exists()
        && let Some(dest_str) = dest_path.to_str()
    {
        let (_, dest_sidecar_path) = parse_virtual_path(dest_str);
        let metadata = crate::exif_processing::load_sidecar_checked(&source_sidecar_path)?;
        crate::exif_processing::save_sidecar(&dest_sidecar_path, &metadata)?;
    }

    let dest_path_str = dest_path.to_string_lossy().into_owned();

    if let Some(album_id) = target_album_id {
        let _ = add_to_album(album_id, vec![dest_path_str.clone()], app_handle);
    }

    Ok(dest_path_str)
}

#[tauri::command]
pub fn copy_files(source_paths: Vec<String>, destination_folder: String) -> Result<(), String> {
    let dest_path = Path::new(&destination_folder);
    if !dest_path.is_dir() {
        return Err(format!(
            "Destination is not a folder: {}",
            destination_folder
        ));
    }

    let unique_source_images: HashSet<PathBuf> = source_paths
        .iter()
        .map(|p| parse_virtual_path(p).0)
        .collect();

    for source_image_path in unique_source_images {
        let source_parent = source_image_path
            .parent()
            .ok_or("Could not get parent directory")?;
        let destination_image = if source_parent == dest_path {
            let stem = source_image_path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("Could not get file stem")?;
            let extension = source_image_path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("");

            let mut counter = 1;
            loop {
                let new_stem = format!("{}_copy_{}", stem, counter);
                let temp_path = if extension.is_empty() {
                    source_parent.join(new_stem)
                } else {
                    source_parent.join(format!("{}.{}", new_stem, extension))
                };
                if !temp_path.exists() {
                    break temp_path;
                }
                counter += 1;
            }
        } else {
            dest_path.join(
                source_image_path
                    .file_name()
                    .ok_or("Could not get source filename")?,
            )
        };

        if destination_image.exists() {
            return Err(format!(
                "File already exists at destination: {}",
                destination_image.display()
            ));
        }

        if crate::exif_processing::get_rrexif_path(&source_image_path).exists() {
            let _ = crate::exif_processing::read_rrexif_sidecar(&source_image_path);
        }
        fs::copy(&source_image_path, &destination_image).map_err(|e| e.to_string())?;
        if let Err(error) =
            crate::sidecar_storage::copy_sidecars(&source_image_path, &destination_image)
        {
            let _ = fs::remove_file(&destination_image);
            let _ = crate::sidecar_storage::remove_sidecars(&destination_image);
            return Err(error);
        }
    }
    Ok(())
}

#[tauri::command]
pub fn move_files(
    source_paths: Vec<String>,
    destination_folder: String,
    app_handle: AppHandle,
) -> Result<(), String> {
    let dest_path = Path::new(&destination_folder);
    if !dest_path.is_dir() {
        return Err(format!(
            "Destination is not a folder: {}",
            destination_folder
        ));
    }

    let unique_source_images: HashSet<PathBuf> = source_paths
        .iter()
        .map(|p| parse_virtual_path(p).0)
        .collect();

    let mut source_images_to_trash = Vec::new();
    let mut renames = HashMap::new();

    for source_image_path in unique_source_images {
        let source_parent = source_image_path
            .parent()
            .ok_or("Could not get parent directory")?;
        if source_parent == dest_path {
            return Err("Cannot move files into the same folder they are already in.".to_string());
        }

        let dest_image_path = dest_path.join(
            source_image_path
                .file_name()
                .ok_or("Could not get source filename")?,
        );
        if dest_image_path.exists() {
            return Err(format!(
                "File already exists at destination: {}",
                dest_image_path.display()
            ));
        }

        if crate::exif_processing::get_rrexif_path(&source_image_path).exists() {
            let _ = crate::exif_processing::read_rrexif_sidecar(&source_image_path);
        }
        fs::copy(&source_image_path, &dest_image_path).map_err(|e| e.to_string())?;
        if let Err(error) =
            crate::sidecar_storage::copy_sidecars(&source_image_path, &dest_image_path)
        {
            let _ = fs::remove_file(&dest_image_path);
            return Err(error);
        }

        renames.insert(
            source_image_path.to_string_lossy().into_owned(),
            dest_image_path.to_string_lossy().into_owned(),
        );

        source_images_to_trash.push(source_image_path);
    }

    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
    if !source_images_to_trash.is_empty()
        && let Err(trash_error) = trash::delete_all(&source_images_to_trash)
    {
        log::warn!(
            "Failed to move source files to trash: {}. Falling back to permanent delete.",
            trash_error
        );
        for path in &source_images_to_trash {
            if path.is_file() {
                fs::remove_file(path).map_err(|e| {
                    format!("Failed to delete source file {}: {}", path.display(), e)
                })?;
            }
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    for path in &source_images_to_trash {
        if path.is_file() {
            fs::remove_file(path)
                .map_err(|e| format!("Failed to delete source file {}: {}", path.display(), e))?;
        }
    }

    for source_image_path in &source_images_to_trash {
        crate::sidecar_storage::remove_sidecars(source_image_path)?;
    }

    sync_album_path_changes(&app_handle, Some(&renames), None, None);

    Ok(())
}

#[tauri::command]
pub fn show_in_finder(path: String) -> Result<(), String> {
    let (source_path, _) = parse_virtual_path(&path);

    #[cfg(target_os = "windows")]
    {
        let source_path_str = source_path.to_string_lossy().to_string();
        Command::new("explorer")
            .args(["/select,", &source_path_str])
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "macos")]
    {
        let source_path_str = source_path.to_string_lossy().to_string();
        Command::new("open")
            .args(["-R", &source_path_str])
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(parent) = source_path.parent() {
            Command::new("xdg-open")
                .arg(parent)
                .spawn()
                .map_err(|e| e.to_string())?;
        } else {
            return Err("Could not get parent directory".into());
        }
    }

    #[cfg(target_os = "android")]
    {
        return Err("Show in File Manager is not natively supported via CLI on Android.".into());
    }

    #[cfg(target_os = "ios")]
    {
        return Err("Show in File Manager is not supported on iOS.".into());
    }

    Ok(())
}

#[tauri::command]
pub fn delete_files_from_disk(paths: Vec<String>, app_handle: AppHandle) -> Result<(), String> {
    let mut images_to_trash = HashSet::new();
    let mut deletions = HashSet::new();

    for path_str in paths {
        let (source_path, _) = parse_virtual_path(&path_str);
        deletions.insert(path_str.clone());

        if let Some((_, copy_id)) = path_str.rsplit_once("?vc=") {
            crate::sidecar_storage::remove_virtual_copy(&source_path, copy_id)?;
        } else if source_path.exists() {
            images_to_trash.insert(source_path);
        }
    }

    if !images_to_trash.is_empty() {
        let final_paths_to_delete: Vec<PathBuf> = images_to_trash.iter().cloned().collect();
        #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
        if let Err(trash_error) = trash::delete_all(&final_paths_to_delete) {
            log::warn!(
                "Failed to move files to trash: {}. Falling back to permanent delete.",
                trash_error
            );
            for path in &final_paths_to_delete {
                if path.is_file() {
                    fs::remove_file(path)
                        .map_err(|e| format!("Failed to delete file {}: {}", path.display(), e))?;
                } else if path.is_dir() {
                    fs::remove_dir_all(path).map_err(|e| {
                        format!("Failed to delete directory {}: {}", path.display(), e)
                    })?;
                }
            }
        }

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        for path in &final_paths_to_delete {
            if path.is_file() {
                fs::remove_file(path)
                    .map_err(|e| format!("Failed to delete file {}: {}", path.display(), e))?;
            } else if path.is_dir() {
                fs::remove_dir_all(path)
                    .map_err(|e| format!("Failed to delete directory {}: {}", path.display(), e))?;
            }
        }

        for image_path in &images_to_trash {
            crate::sidecar_storage::remove_sidecars(image_path)?;
        }
    }

    sync_album_path_changes(&app_handle, None, Some(&deletions), None);

    Ok(())
}

pub(super) fn deletion_stem_for(filename: &str) -> Option<&str> {
    let image_filename = if filename.ends_with(".rrdata") {
        let without_rrdata = filename.trim_end_matches(".rrdata");
        if let Some(dot_pos) = without_rrdata.rfind('.') {
            let suffix = &without_rrdata[dot_pos + 1..];
            if suffix.len() == 6 && suffix.chars().all(|c| c.is_ascii_hexdigit()) {
                &without_rrdata[..dot_pos]
            } else {
                without_rrdata
            }
        } else {
            without_rrdata
        }
    } else if filename.ends_with(".rrexif") {
        filename.trim_end_matches(".rrexif")
    } else if is_supported_image_file(filename) {
        filename
    } else {
        return None;
    };
    Path::new(image_filename)
        .file_stem()
        .and_then(|s| s.to_str())
}

#[tauri::command]
pub fn delete_files_with_associated(
    paths: Vec<String>,
    app_handle: AppHandle,
) -> Result<(), String> {
    if paths.is_empty() {
        return Ok(());
    }

    let mut stems_to_delete = HashSet::new();
    let mut parent_dirs = HashSet::new();
    let mut deletions = HashSet::new();

    for path_str in &paths {
        deletions.insert(path_str.clone());
        let (source_path, _) = parse_virtual_path(path_str);
        if let Some(stem) = source_path.file_stem().and_then(|s| s.to_str()) {
            stems_to_delete.insert(stem.to_string());
        }
        if let Some(parent) = source_path.parent() {
            parent_dirs.insert(parent.to_path_buf());
        }
    }

    if stems_to_delete.is_empty() {
        return Ok(());
    }

    let mut files_to_trash = HashSet::new();

    for parent_dir in parent_dirs {
        if let Ok(entries) = fs::read_dir(parent_dir) {
            for entry in entries.filter_map(Result::ok) {
                let entry_path = entry.path();
                if !entry_path.is_file() {
                    continue;
                }

                let entry_filename = entry.file_name();
                let entry_filename_str = entry_filename.to_string_lossy();

                if let Some(stem) = deletion_stem_for(&entry_filename_str)
                    && stems_to_delete.contains(stem)
                {
                    files_to_trash.insert(entry_path);
                }
            }
        }
    }

    if files_to_trash.is_empty() {
        return Ok(());
    }

    let image_paths_to_cleanup: Vec<PathBuf> = files_to_trash
        .iter()
        .filter(|path| is_supported_image_file(path))
        .cloned()
        .collect();
    let final_paths_to_delete: Vec<PathBuf> = files_to_trash.into_iter().collect();
    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
    if let Err(trash_error) = trash::delete_all(&final_paths_to_delete) {
        log::warn!(
            "Failed to move files to trash: {}. Falling back to permanent delete.",
            trash_error
        );
        for path in final_paths_to_delete {
            if path.is_file() {
                fs::remove_file(&path)
                    .map_err(|e| format!("Failed to delete file {}: {}", path.display(), e))?;
            }
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    for path in final_paths_to_delete {
        if path.is_file() {
            fs::remove_file(&path)
                .map_err(|e| format!("Failed to delete file {}: {}", path.display(), e))?;
        }
    }

    for image_path in image_paths_to_cleanup {
        crate::sidecar_storage::remove_sidecars(&image_path)?;
    }

    sync_album_path_changes(&app_handle, None, Some(&deletions), None);

    Ok(())
}

#[tauri::command]
pub fn rename_files(
    paths: Vec<String>,
    name_template: String,
    app_handle: AppHandle,
) -> Result<Vec<String>, String> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }

    let mut operations: HashMap<PathBuf, PathBuf> = HashMap::new();
    let mut final_new_paths = Vec::with_capacity(paths.len());
    let mut renames = HashMap::new();

    for (i, path_str) in paths.iter().enumerate() {
        let (original_path, _) = parse_virtual_path(path_str);
        if !original_path.exists() {
            return Err(format!("File not found: {}", path_str));
        }

        let parent = original_path
            .parent()
            .ok_or("Could not get parent directory")?;
        let extension = original_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        let file_date = exif_processing::get_creation_date_from_path(&original_path);

        let new_stem = generate_filename_from_template(
            &name_template,
            &original_path,
            i + 1,
            paths.len(),
            &file_date,
        );
        let new_filename = format!("{}.{}", new_stem, extension);
        let new_path = parent.join(new_filename);

        if new_path.exists() && new_path != original_path {
            return Err(format!(
                "A file with the name {} already exists.",
                new_path.display()
            ));
        }

        if crate::exif_processing::get_rrexif_path(&original_path).exists() {
            let _ = crate::exif_processing::read_rrexif_sidecar(&original_path);
        }
        operations.insert(original_path, new_path);
    }

    for (old_path, new_path) in operations {
        fs::rename(&old_path, &new_path).map_err(|e| {
            format!(
                "Failed to rename {} to {}: {}",
                old_path.display(),
                new_path.display(),
                e
            )
        })?;

        if let Err(error) = crate::sidecar_storage::move_sidecars(&old_path, &new_path) {
            let rollback_result = fs::rename(&new_path, &old_path);
            return Err(match rollback_result {
                Ok(()) => format!("Failed to move edit data while renaming: {}", error),
                Err(rollback_error) => format!(
                    "File was renamed, but edit data could not be moved ({}), and rollback failed: {}",
                    error, rollback_error
                ),
            });
        }

        let old_str = old_path.to_string_lossy().into_owned();
        let new_str = new_path.to_string_lossy().into_owned();

        renames.insert(old_str, new_str.clone());

        if is_supported_image_file(&new_path) {
            final_new_paths.push(new_str);
        }
    }

    sync_album_path_changes(&app_handle, Some(&renames), None, None);

    Ok(final_new_paths)
}

#[tauri::command]
pub fn create_virtual_copy(
    source_virtual_path: String,
    target_album_id: Option<String>,
    app_handle: AppHandle,
) -> Result<String, String> {
    let (source_path, source_sidecar_path) = parse_virtual_path(&source_virtual_path);

    let new_copy_id = Uuid::new_v4().to_string()[..6].to_string();
    let new_virtual_path = format!("{}?vc={}", source_path.to_string_lossy(), new_copy_id);
    let (_, new_sidecar_path) = parse_virtual_path(&new_virtual_path);

    let metadata = if source_sidecar_path.exists() {
        crate::exif_processing::load_sidecar_checked(&source_sidecar_path)?
    } else {
        ImageMetadata::default()
    };
    crate::exif_processing::save_sidecar(&new_sidecar_path, &metadata)?;

    if let Some(album_id) = target_album_id {
        let _ = add_to_album(album_id, vec![new_virtual_path.clone()], app_handle);
    }

    Ok(new_virtual_path)
}
