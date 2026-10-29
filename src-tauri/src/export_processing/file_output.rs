//! Writing encoded exports to disk.
//!
//! Metadata, timestamps, temporary files beside the destination and the
//! preserved permissions of an existing output.

use crate::exif_processing;
use image::DynamicImage;
use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use tempfile::NamedTempFile;

#[cfg(target_os = "android")]
use super::mime_type_for_extension;
#[cfg(not(target_os = "android"))]
use super::save_webp_with_bounded_output;
use super::{ExportSettings, encode_image_to_bytes_with_profile, ensure_export_not_cancelled};

pub(super) fn set_timestamps_from_exif(src: &Path, dst: &Path) {
    let capture_dt = exif_processing::get_creation_date_from_path(src);
    let ft = filetime::FileTime::from_unix_time(
        capture_dt.timestamp(),
        capture_dt.timestamp_subsec_nanos(),
    );
    if let Err(e) = filetime::set_file_times(dst, ft, ft) {
        log::warn!("Could not set timestamps on '{}': {}", dst.display(), e);
    }
}

pub(super) fn save_image_with_metadata(
    image: &DynamicImage,
    output_path: &std::path::Path,
    source_path_str: &str,
    export_settings: &ExportSettings,
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    ensure_export_not_cancelled(cancellation_token)?;
    let extension = output_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    #[cfg(not(target_os = "android"))]
    if extension == "webp" {
        return save_webp_with_bounded_output(
            image,
            output_path,
            export_settings.jpeg_quality,
            cancellation_token,
        );
    }

    let mut image_bytes = encode_image_to_bytes_with_profile(
        image,
        &extension,
        export_settings.jpeg_quality,
        export_settings.embed_color_profile,
        export_settings.bit_depth,
    )?;
    ensure_export_not_cancelled(cancellation_token)?;

    exif_processing::write_image_with_metadata(
        &mut image_bytes,
        source_path_str,
        &extension,
        export_settings.keep_metadata,
        export_settings.strip_gps,
        export_settings.metadata_overrides.as_ref(),
    )?;
    ensure_export_not_cancelled(cancellation_token)?;

    #[cfg(target_os = "android")]
    {
        let file_name = output_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "Missing Android export file name".to_string())?;
        crate::android_integration::save_image_bytes_to_android_gallery(
            file_name,
            mime_type_for_extension(&extension),
            &image_bytes,
        )?;
    }

    #[cfg(not(target_os = "android"))]
    fs::write(output_path, image_bytes).map_err(|e| e.to_string())?;

    Ok(())
}

pub(super) fn supports_streaming_export(
    output_format: &str,
    _export_settings: &ExportSettings,
) -> bool {
    #[cfg(target_os = "android")]
    {
        let _ = (output_format, _export_settings);
        false
    }

    #[cfg(not(target_os = "android"))]
    {
        matches!(
            output_format.to_ascii_lowercase().as_str(),
            "jpg" | "jpeg" | "png" | "tif" | "tiff"
        )
    }
}

#[cfg(not(target_os = "android"))]
pub(super) fn create_temporary_export(output_path: &Path) -> Result<NamedTempFile, String> {
    let output_parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    NamedTempFile::new_in(output_parent).map_err(|error| {
        format!(
            "Failed to create temporary export beside '{}': {error}",
            output_path.display()
        )
    })
}

#[cfg(not(target_os = "android"))]
pub(super) fn preserve_existing_export_permissions(
    temporary: &NamedTempFile,
    output_path: &Path,
) -> Result<(), String> {
    if let Ok(metadata) = fs::metadata(output_path) {
        temporary
            .as_file()
            .set_permissions(metadata.permissions())
            .map_err(|error| {
                format!(
                    "Failed to preserve permissions for '{}': {error}",
                    output_path.display()
                )
            })?;
    }
    Ok(())
}
