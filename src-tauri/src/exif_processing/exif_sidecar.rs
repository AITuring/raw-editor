//! The EXIF map cached in the primary sidecar.
//!
//! Reads EXIF maps from source bytes, stores them in the primary sidecar and
//! migrates legacy rrexif sidecars into it.

use crate::formats::is_raw_file;
use crate::image_processing::ImageMetadata;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use super::{
    clean_ascii_value, extract_metadata, format_exif_exposure_time, format_lens_specification,
    load_sidecar, read_exif, save_sidecar, truncate_large_exif,
};

pub fn get_primary_sidecar_path(image_path: &Path) -> PathBuf {
    crate::sidecar_storage::primary_sidecar_path(image_path)
}

pub fn get_rrexif_path(image_path: &Path) -> PathBuf {
    crate::sidecar_storage::legacy_rrexif_path(image_path)
}

fn load_primary_metadata(image_path: &Path) -> ImageMetadata {
    let primary = get_primary_sidecar_path(image_path);
    load_sidecar(&primary)
}

fn save_primary_metadata(image_path: &Path, metadata: &ImageMetadata) -> std::io::Result<()> {
    let primary = get_primary_sidecar_path(image_path);
    save_sidecar(&primary, metadata).map_err(std::io::Error::other)
}

pub fn read_rrexif_sidecar(image_path: &Path) -> Option<HashMap<String, String>> {
    if crate::sidecar_storage::is_initialized() {
        let metadata = load_primary_metadata(image_path);
        if let Some(exif) = metadata.exif {
            let legacy = get_rrexif_path(image_path);
            if legacy.exists() {
                let _ = fs::remove_file(legacy);
            }
            return Some(exif);
        }
    }

    let legacy = get_rrexif_path(image_path);
    if legacy.exists()
        && let Ok(content) = fs::read_to_string(&legacy)
        && let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content)
    {
        if crate::sidecar_storage::is_initialized() {
            let mut migrated = load_primary_metadata(image_path);
            migrated.exif = Some(map.clone());
            if save_primary_metadata(image_path, &migrated).is_ok() {
                let _ = fs::remove_file(&legacy);
            }
        }
        return Some(map);
    }

    None
}

pub fn read_exif_data_from_bytes(path: &str, file_bytes: &[u8]) -> HashMap<String, String> {
    if is_raw_file(path)
        && let Some(mut map) = extract_metadata(file_bytes)
    {
        enrich_raw_white_balance(path, file_bytes, &mut map);
        return map;
    }

    let mut exif_data = HashMap::new();
    if let Some(exif) = read_exif(file_bytes) {
        for field in exif.fields() {
            let raw_val = match &field.value {
                exif::Value::Ascii(_) => match clean_ascii_value(&field.value) {
                    Some(v) => v,
                    None => continue,
                },
                exif::Value::Rational(v) if field.tag == exif::Tag::ExposureTime => {
                    match v
                        .first()
                        .and_then(|r| format_exif_exposure_time(r.num, r.denom))
                    {
                        Some(s) => s,
                        None => continue,
                    }
                }
                exif::Value::Rational(v) if field.tag == exif::Tag::LensSpecification => {
                    match format_lens_specification(v) {
                        Some(s) => s,
                        None => continue,
                    }
                }
                _ => field.display_value().with_unit(&exif).to_string(),
            };
            exif_data.insert(field.tag.to_string(), truncate_large_exif(&raw_val));
        }
    }
    enrich_raw_white_balance(path, file_bytes, &mut exif_data);
    exif_data
}

/// Return the 35mm-equivalent focal length when the source embeds one. Image
/// stacks use this as a geometry hint only; the original metadata is never
/// rewritten. A physical focal length is intentionally not used as a fallback:
/// without the sensor crop factor it cannot be compared across camera modules.
pub fn focal_length_35mm_from_bytes(file_bytes: &[u8]) -> Option<f64> {
    let exif = read_exif(file_bytes)?;
    exif.get_field(exif::Tag::FocalLengthIn35mmFilm, exif::In::PRIMARY)
        .and_then(|field| field.value.get_uint(0))
        .map(f64::from)
        .filter(|value| value.is_finite() && (10.0..=300.0).contains(value))
}

fn enrich_raw_white_balance(
    path: &str,
    file_bytes: &[u8],
    exif_data: &mut HashMap<String, String>,
) -> bool {
    if !is_raw_file(path) || exif_data.contains_key("AsShotTemperature") {
        return false;
    }

    let Some(temperature) = crate::raw_processing::estimate_as_shot_temperature(file_bytes) else {
        return false;
    };

    exif_data.insert("AsShotTemperature".to_string(), temperature.to_string());
    true
}

pub fn read_exif_data(path: &str, file_bytes: &[u8]) -> HashMap<String, String> {
    let source_path = Path::new(path);
    if let Some(mut sidecar_exif) = read_rrexif_sidecar(source_path) {
        if enrich_raw_white_balance(path, file_bytes, &mut sidecar_exif) {
            let mut metadata = load_primary_metadata(source_path);
            metadata.exif = Some(sidecar_exif.clone());
            let _ = save_primary_metadata(source_path, &metadata);
        }
        return sidecar_exif;
    }

    let exif_map = read_exif_data_from_bytes(path, file_bytes);
    if !exif_map.is_empty() {
        let mut metadata = load_primary_metadata(source_path);
        metadata.exif = Some(exif_map.clone());
        let _ = save_primary_metadata(source_path, &metadata);
    }
    exif_map
}

pub fn persist_exif_if_missing(source_path: &Path, source_path_str: &str, file_bytes: &[u8]) {
    {
        let metadata = load_primary_metadata(source_path);
        if metadata.exif.is_some() {
            return;
        }
    }

    let legacy = get_rrexif_path(source_path);
    if legacy.exists()
        && let Ok(content) = fs::read_to_string(&legacy)
        && let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content)
    {
        let mut metadata = load_primary_metadata(source_path);
        metadata.exif = Some(map);
        if save_primary_metadata(source_path, &metadata).is_ok() {
            let _ = fs::remove_file(&legacy);
        }
        return;
    }

    let exif_map = read_exif_data_from_bytes(source_path_str, file_bytes);
    if exif_map.is_empty() {
        return;
    }

    let mut metadata = load_primary_metadata(source_path);

    if metadata.exif.is_none() {
        metadata.exif = Some(exif_map);
        let _ = save_primary_metadata(source_path, &metadata);
    }
}

pub fn write_rrexif_sidecar(source_path_str: &str, target_image_path: &Path) -> Result<(), String> {
    let source_path = Path::new(source_path_str);

    let exif_data = if let Some(existing) = read_rrexif_sidecar(source_path) {
        existing
    } else if let Ok(bytes) = fs::read(source_path) {
        read_exif_data_from_bytes(source_path_str, &bytes)
    } else {
        return Ok(());
    };

    if exif_data.is_empty() {
        return Ok(());
    }

    let mut metadata = load_primary_metadata(target_image_path);
    metadata.exif = Some(exif_data);
    save_primary_metadata(target_image_path, &metadata)
        .map_err(|e| format!("Failed to write sidecar: {}", e))
}
