//! Image sidecar documents on disk.
//!
//! Reads, migrates, validates and atomically writes sidecar documents, keeps a
//! last-known-good backup and recovers a damaged primary from it.

use crate::image_processing::ImageMetadata;
use crate::image_processing::SIDECAR_SCHEMA_VERSION;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use tempfile::NamedTempFile;

static SIDECAR_WRITE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
pub(super) enum SidecarReadError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
    FutureVersion(u32),
}

impl std::fmt::Display for SidecarReadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::Invalid(message) => formatter.write_str(message),
            Self::FutureVersion(version) => write!(
                formatter,
                "sidecar schema version {version} is newer than supported version {SIDECAR_SCHEMA_VERSION}"
            ),
        }
    }
}

pub(super) fn sidecar_backup_path(sidecar_path: &Path) -> PathBuf {
    let mut filename = sidecar_path.file_name().unwrap_or_default().to_os_string();
    filename.push(".bak");
    sidecar_path.with_file_name(filename)
}

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("Sidecar path has no parent directory"))?;
    fs::create_dir_all(parent)?;

    let mut temporary = NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.flush()?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;

    #[cfg(unix)]
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

pub(super) fn read_sidecar_document(
    path: &Path,
) -> Result<(ImageMetadata, bool), SidecarReadError> {
    let content = fs::read_to_string(path).map_err(SidecarReadError::Io)?;
    let mut document: serde_json::Value =
        serde_json::from_str(&content).map_err(SidecarReadError::Json)?;

    let version = match document.get("version") {
        None => 0,
        Some(version) => version.as_u64().ok_or_else(|| {
            SidecarReadError::Invalid("sidecar version must be a non-negative integer".to_string())
        })?,
    };
    let version = u32::try_from(version)
        .map_err(|_| SidecarReadError::Invalid("sidecar version is out of range".to_string()))?;
    if version > SIDECAR_SCHEMA_VERSION {
        return Err(SidecarReadError::FutureVersion(version));
    }

    let mut migrated = version != SIDECAR_SCHEMA_VERSION;
    if migrated {
        let object = document.as_object_mut().ok_or_else(|| {
            SidecarReadError::Invalid("sidecar root must be a JSON object".to_string())
        })?;
        object.insert(
            "version".to_string(),
            serde_json::Value::from(SIDECAR_SCHEMA_VERSION),
        );
    }

    let mut metadata: ImageMetadata =
        serde_json::from_value(document).map_err(SidecarReadError::Json)?;
    if metadata.rating > 5 {
        return Err(SidecarReadError::Invalid(format!(
            "sidecar rating {} is outside the supported 0-5 range",
            metadata.rating
        )));
    }
    metadata.version = SIDECAR_SCHEMA_VERSION;

    if let Some(ref mut exif_map) = metadata.exif {
        for value in exif_map.values_mut() {
            if value.len() > 500 {
                *value = truncate_large_exif(value);
                migrated = true;
            }
        }
    }

    Ok((metadata, migrated))
}

fn write_metadata_without_backup(
    sidecar_path: &Path,
    metadata: &ImageMetadata,
) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(metadata).map_err(|error| error.to_string())?;
    atomic_write_bytes(sidecar_path, &json).map_err(|error| error.to_string())
}

pub fn save_sidecar(sidecar_path: &Path, metadata: &ImageMetadata) -> Result<(), String> {
    if metadata.version != SIDECAR_SCHEMA_VERSION {
        return Err(format!(
            "cannot save sidecar schema version {}; expected {SIDECAR_SCHEMA_VERSION}",
            metadata.version
        ));
    }
    if metadata.rating > 5 {
        return Err(format!(
            "cannot save sidecar rating {}; expected 0-5",
            metadata.rating
        ));
    }

    let _write_guard = SIDECAR_WRITE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if sidecar_path.is_file() {
        match read_sidecar_document(sidecar_path) {
            Ok(_) => {
                let current = fs::read(sidecar_path).map_err(|error| error.to_string())?;
                atomic_write_bytes(&sidecar_backup_path(sidecar_path), &current)
                    .map_err(|error| error.to_string())?;
            }
            Err(SidecarReadError::FutureVersion(version)) => {
                return Err(format!(
                    "refusing to overwrite newer sidecar schema version {version}"
                ));
            }
            Err(_) => {
                // Preserve the last known-good backup when the primary is already damaged.
            }
        }
    }

    write_metadata_without_backup(sidecar_path, metadata)
}

pub fn truncate_large_exif(value: &str) -> String {
    if value.len() <= 500 {
        return value.to_string();
    }

    let mut start_idx = 200;
    while !value.is_char_boundary(start_idx) {
        start_idx -= 1;
    }

    let mut end_idx = value.len() - 200;
    while !value.is_char_boundary(end_idx) {
        end_idx += 1;
    }

    if start_idx < end_idx {
        let start_str = &value[..start_idx];
        let end_str = &value[end_idx..];
        return format!("{}...{}", start_str, end_str);
    }

    value.to_string()
}

pub fn load_sidecar_checked(sidecar_path: &Path) -> Result<ImageMetadata, String> {
    if !sidecar_path.exists() {
        return Ok(ImageMetadata::default());
    }

    match read_sidecar_document(sidecar_path) {
        Ok((metadata, migrated)) => {
            if migrated {
                save_sidecar(sidecar_path, &metadata)?;
                log::info!("Migrated sidecar to schema 1: {}", sidecar_path.display());
            }
            Ok(metadata)
        }
        Err(SidecarReadError::FutureVersion(version)) => Err(format!(
            "sidecar schema version {version} is newer than supported version {SIDECAR_SCHEMA_VERSION}"
        )),
        Err(primary_error) => {
            let backup_path = sidecar_backup_path(sidecar_path);
            let (metadata, _) = read_sidecar_document(&backup_path).map_err(|backup_error| {
                format!(
                    "failed to read sidecar {} ({primary_error}); recovery backup {} is unavailable ({backup_error})",
                    sidecar_path.display(),
                    backup_path.display()
                )
            })?;

            let _write_guard = SIDECAR_WRITE_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            write_metadata_without_backup(sidecar_path, &metadata)?;
            log::warn!(
                "Recovered damaged sidecar {} from {}",
                sidecar_path.display(),
                backup_path.display()
            );
            Ok(metadata)
        }
    }
}

pub fn load_sidecar(sidecar_path: &Path) -> ImageMetadata {
    match load_sidecar_checked(sidecar_path) {
        Ok(metadata) => metadata,
        Err(error) => {
            log::error!("Failed to load sidecar {}: {error}", sidecar_path.display());
            ImageMetadata::default()
        }
    }
}
