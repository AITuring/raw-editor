//! The import command and its file-name templates.

#[cfg(target_os = "android")]
use crate::android_integration::is_android_content_uri;
#[cfg(target_os = "android")]
use crate::android_integration::read_android_content_uri;
#[cfg(target_os = "android")]
use crate::android_integration::resolve_android_content_uri_name;
use crate::exif_processing;
use anyhow::Result;
use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use tauri::AppHandle;
use tauri::Emitter;

use super::parse_virtual_path;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImportSettings {
    pub filename_template: String,
    pub organize_by_date: bool,
    pub date_folder_format: String,
    pub delete_after_import: bool,
}

#[tauri::command]
pub async fn import_files(
    source_paths: Vec<String>,
    destination_folder: String,
    settings: ImportSettings,
    app_handle: AppHandle,
) -> Result<(), String> {
    let total_files = source_paths.len();
    let _ = app_handle.emit("import-start", serde_json::json!({ "total": total_files }));

    tauri::async_runtime::spawn_blocking(move || {
        for (i, source_path_str) in source_paths.iter().enumerate() {
            let _ = app_handle.emit(
                "import-progress",
                serde_json::json!({ "current": i, "total": total_files, "path": source_path_str }),
            );

            let import_result: Result<(), String> = (|| {
                #[cfg(target_os = "android")]
                if is_android_content_uri(source_path_str) {
                    let resolved_name = resolve_android_content_uri_name(source_path_str)?;
                    let source_bytes = read_android_content_uri(source_path_str)?;
                    let source_name_path = Path::new(&resolved_name);
                    let file_date = exif_processing::get_creation_date_from_bytes(
                        &resolved_name,
                        &source_bytes,
                    );

                    let mut final_dest_folder = PathBuf::from(&destination_folder);
                    if settings.organize_by_date {
                        let date_format_str = settings
                            .date_folder_format
                            .replace("YYYY", "%Y")
                            .replace("MM", "%m")
                            .replace("DD", "%d");
                        let subfolder = file_date.format(&date_format_str).to_string();
                        final_dest_folder.push(subfolder);
                    }

                    fs::create_dir_all(&final_dest_folder)
                        .map_err(|e| format!("Failed to create destination folder: {}", e))?;

                    let new_stem = generate_filename_from_template(
                        &settings.filename_template,
                        source_name_path,
                        i + 1,
                        total_files,
                        &file_date,
                    );
                    let extension = source_name_path
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");
                    let new_filename = format!("{}.{}", new_stem, extension);
                    let dest_file_path = final_dest_folder.join(new_filename);

                    if dest_file_path.exists() {
                        return Err(format!(
                            "File already exists at destination: {}",
                            dest_file_path.display()
                        ));
                    }

                    fs::write(&dest_file_path, source_bytes).map_err(|e| e.to_string())?;

                    if settings.delete_after_import {
                        log::info!(
                            "Skipping delete_after_import for Android content URI source: {}",
                            source_path_str
                        );
                    }

                    return Ok(());
                }

                let (source_path, source_sidecar) = parse_virtual_path(source_path_str);
                if !source_path.exists() {
                    return Err(format!("Source file not found: {}", source_path_str));
                }

                let file_date = exif_processing::get_creation_date_from_path(&source_path);

                let mut final_dest_folder = PathBuf::from(&destination_folder);
                if settings.organize_by_date {
                    let date_format_str = settings
                        .date_folder_format
                        .replace("YYYY", "%Y")
                        .replace("MM", "%m")
                        .replace("DD", "%d");
                    let subfolder = file_date.format(&date_format_str).to_string();
                    final_dest_folder.push(subfolder);
                }

                fs::create_dir_all(&final_dest_folder)
                    .map_err(|e| format!("Failed to create destination folder: {}", e))?;

                let new_stem = generate_filename_from_template(
                    &settings.filename_template,
                    &source_path,
                    i + 1,
                    total_files,
                    &file_date,
                );
                let extension = source_path
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                let new_filename = format!("{}.{}", new_stem, extension);
                let dest_file_path = final_dest_folder.join(new_filename);

                if dest_file_path.exists() {
                    return Err(format!(
                        "File already exists at destination: {}",
                        dest_file_path.display()
                    ));
                }

                fs::copy(&source_path, &dest_file_path).map_err(|e| e.to_string())?;
                if crate::exif_processing::get_rrexif_path(&source_path).exists() {
                    let _ = crate::exif_processing::read_rrexif_sidecar(&source_path);
                }
                if source_sidecar.exists()
                    && let Some(dest_str) = dest_file_path.to_str()
                {
                    let (_, dest_sidecar) = parse_virtual_path(dest_str);
                    fs::copy(&source_sidecar, &dest_sidecar).map_err(|e| e.to_string())?;
                }

                if settings.delete_after_import {
                    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
                    {
                        if let Err(trash_error) = trash::delete(&source_path) {
                            log::warn!(
                                "Failed to trash source file {}: {}. Deleting permanently.",
                                source_path.display(),
                                trash_error
                            );
                            fs::remove_file(&source_path).map_err(|e| e.to_string())?;
                        }
                    }

                    #[cfg(not(any(
                        target_os = "windows",
                        target_os = "macos",
                        target_os = "linux"
                    )))]
                    {
                        fs::remove_file(&source_path).map_err(|e| e.to_string())?;
                    }

                    crate::sidecar_storage::remove_sidecars(&source_path)?;
                }

                Ok(())
            })();

            if let Err(e) = import_result {
                eprintln!("Failed to import {}: {}", source_path_str, e);
                let _ = app_handle.emit("import-error", e);
                return;
            }
        }

        let _ = app_handle.emit(
            "import-progress",
            serde_json::json!({ "current": total_files, "total": total_files, "path": "" }),
        );
        let _ = app_handle.emit("import-complete", ());
    });

    Ok(())
}

pub fn generate_filename_from_template(
    template: &str,
    original_path: &std::path::Path,
    sequence: usize,
    total: usize,
    file_date: &DateTime<Utc>,
) -> String {
    let stem = original_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    let sequence_str = format!(
        "{:0width$}",
        sequence,
        width = total.to_string().len().max(1)
    );
    let local_date = file_date.with_timezone(&chrono::Local);

    let mut result = template.to_string();
    result = result.replace("{original_filename}", stem);
    result = result.replace("{sequence}", &sequence_str);
    result = result.replace("{YYYY}", &local_date.format("%Y").to_string());
    result = result.replace("{MM}", &local_date.format("%m").to_string());
    result = result.replace("{DD}", &local_date.format("%d").to_string());
    result = result.replace("{hh}", &local_date.format("%H").to_string());
    result = result.replace("{mm}", &local_date.format("%M").to_string());

    result
}
