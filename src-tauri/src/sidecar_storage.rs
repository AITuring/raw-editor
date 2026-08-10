use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use tempfile::NamedTempFile;

const PRIMARY_SIDECAR_NAME: &str = "primary.rrdata";
const SIDECAR_DIRECTORY_NAME: &str = "sidecars-v1";

static SIDECAR_ROOT: OnceLock<PathBuf> = OnceLock::new();
static MIGRATION_LOCK: Mutex<()> = Mutex::new(());
static MIGRATED_LEGACY_DIRECTORIES: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();

pub fn initialize(app_data_dir: &Path) -> Result<(), String> {
    let root = app_data_dir.join(SIDECAR_DIRECTORY_NAME);
    fs::create_dir_all(&root).map_err(|error| {
        format!(
            "Failed to create private sidecar directory {}: {}",
            root.display(),
            error
        )
    })?;

    if let Some(existing_root) = SIDECAR_ROOT.get() {
        return if existing_root == &root {
            Ok(())
        } else {
            Err("Sidecar storage was initialized with a different directory".to_string())
        };
    }

    SIDECAR_ROOT
        .set(root)
        .map_err(|_| "Failed to initialize sidecar storage".to_string())
}

pub fn is_initialized() -> bool {
    SIDECAR_ROOT.get().is_some()
}

fn storage_root() -> &'static Path {
    SIDECAR_ROOT
        .get()
        .expect("sidecar storage must be initialized during application setup")
}

fn normalized_source_path(image_path: &Path) -> PathBuf {
    if image_path.is_absolute() {
        image_path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(image_path)
    }
}

fn image_storage_directory(image_path: &Path) -> PathBuf {
    let normalized_path = normalized_source_path(image_path);
    let identity = normalized_path.to_string_lossy().replace('\\', "/");
    #[cfg(target_os = "windows")]
    let identity = identity.to_ascii_lowercase();

    let hash = blake3::hash(identity.as_bytes()).to_hex().to_string();
    storage_root().join(&hash[..2]).join(hash)
}

fn valid_copy_id(copy_id: &str) -> bool {
    copy_id.len() == 6
        && copy_id
            .chars()
            .all(|character| character.is_ascii_hexdigit())
}

pub fn legacy_sidecar_path(image_path: &Path, copy_id: Option<&str>) -> PathBuf {
    let image_name = image_path.file_name().unwrap_or_default().to_string_lossy();
    let sidecar_name = match copy_id {
        Some(copy_id) => format!("{}.{}.rrdata", image_name, copy_id),
        None => format!("{}.rrdata", image_name),
    };
    image_path.with_file_name(sidecar_name)
}

pub fn legacy_rrexif_path(image_path: &Path) -> PathBuf {
    let mut filename = image_path.file_name().unwrap_or_default().to_os_string();
    filename.push(".rrexif");
    image_path.with_file_name(filename)
}

fn copy_file_atomically(source: &Path, destination: &Path) -> io::Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| io::Error::other("Sidecar destination has no parent directory"))?;
    fs::create_dir_all(parent)?;

    let mut temporary_file = NamedTempFile::new_in(parent)?;
    let mut source_file = fs::File::open(source)?;
    io::copy(&mut source_file, temporary_file.as_file_mut())?;
    temporary_file.as_file().sync_all()?;
    temporary_file
        .persist(destination)
        .map_err(|error| error.error)?;
    Ok(())
}

fn migrate_legacy_sidecar(legacy_path: &Path, private_path: &Path) {
    if !legacy_path.is_file() {
        return;
    }

    let _migration_guard = MIGRATION_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !legacy_path.is_file() {
        return;
    }

    let legacy_is_newer = match (
        fs::metadata(legacy_path).and_then(|metadata| metadata.modified()),
        fs::metadata(private_path).and_then(|metadata| metadata.modified()),
    ) {
        (Ok(legacy_modified), Ok(private_modified)) => legacy_modified > private_modified,
        (Ok(_), Err(_)) => true,
        _ => !private_path.exists(),
    };

    if legacy_is_newer && let Err(error) = copy_file_atomically(legacy_path, private_path) {
        log::warn!(
            "Failed to migrate sidecar {} to private storage: {}",
            legacy_path.display(),
            error
        );
        return;
    }

    if private_path.is_file()
        && let Err(error) = fs::remove_file(legacy_path)
    {
        log::warn!(
            "Sidecar was migrated but the legacy file {} could not be removed: {}",
            legacy_path.display(),
            error
        );
    }
}

pub fn sidecar_path(image_path: &Path, copy_id: Option<&str>) -> PathBuf {
    let directory = image_storage_directory(image_path);
    if let Err(error) = fs::create_dir_all(&directory) {
        log::error!(
            "Failed to create private sidecar directory {}: {}",
            directory.display(),
            error
        );
    }

    let filename = match copy_id {
        Some(copy_id) if valid_copy_id(copy_id) => format!("{}.rrdata", copy_id),
        Some(copy_id) => {
            let safe_id = blake3::hash(copy_id.as_bytes()).to_hex().to_string();
            format!("copy-{}.rrdata", &safe_id[..12])
        }
        None => PRIMARY_SIDECAR_NAME.to_string(),
    };
    let private_path = directory.join(filename);
    let legacy_path = legacy_sidecar_path(image_path, copy_id);
    migrate_legacy_sidecar(&legacy_path, &private_path);
    private_path
}

pub fn primary_sidecar_path(image_path: &Path) -> PathBuf {
    sidecar_path(image_path, None)
}

fn migrate_legacy_sidecars_in_directory(directory: &Path) {
    let directory_key = normalized_source_path(directory);
    let migrated_directories = MIGRATED_LEGACY_DIRECTORIES.get_or_init(Default::default);
    if !migrated_directories
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(directory_key.clone())
    {
        return;
    }

    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(_) => {
            migrated_directories
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&directory_key);
            return;
        }
    };
    for entry in entries.filter_map(Result::ok) {
        let filename = entry.file_name();
        let filename = filename.to_string_lossy();
        let Some(base_name) = filename.strip_suffix(".rrdata") else {
            continue;
        };

        let (image_filename, copy_id) = if let Some((image_filename, possible_copy_id)) =
            base_name.rsplit_once('.')
            && valid_copy_id(possible_copy_id)
        {
            (image_filename, Some(possible_copy_id))
        } else {
            (base_name, None)
        };
        let image_path = directory.join(image_filename);
        if image_path.is_file() {
            let _ = sidecar_path(&image_path, copy_id);
        }
    }
}

pub fn virtual_copy_ids(image_path: &Path) -> Vec<String> {
    if let Some(parent) = image_path.parent() {
        migrate_legacy_sidecars_in_directory(parent);
    }

    let directory = image_storage_directory(image_path);
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };

    let mut copy_ids: Vec<String> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let filename = entry.file_name();
            let filename = filename.to_string_lossy();
            let copy_id = filename.strip_suffix(".rrdata")?;
            valid_copy_id(copy_id).then(|| copy_id.to_string())
        })
        .collect();
    copy_ids.sort_unstable();
    copy_ids.dedup();
    copy_ids
}

pub fn sidecar_entries(image_path: &Path) -> Vec<(Option<String>, PathBuf)> {
    let mut entries = Vec::new();
    let primary_path = primary_sidecar_path(image_path);
    if primary_path.is_file() {
        entries.push((None, primary_path));
    }
    entries.extend(
        virtual_copy_ids(image_path)
            .into_iter()
            .filter_map(|copy_id| {
                let path = sidecar_path(image_path, Some(&copy_id));
                path.is_file().then_some((Some(copy_id), path))
            }),
    );
    entries
}

pub fn copy_sidecars(source_image: &Path, destination_image: &Path) -> Result<usize, String> {
    let mut copied_count = 0;
    for (copy_id, source_path) in sidecar_entries(source_image) {
        let destination_path = sidecar_path(destination_image, copy_id.as_deref());
        copy_file_atomically(&source_path, &destination_path).map_err(|error| {
            format!(
                "Failed to copy sidecar {} to {}: {}",
                source_path.display(),
                destination_path.display(),
                error
            )
        })?;
        copied_count += 1;
    }
    Ok(copied_count)
}

pub fn move_sidecars(source_image: &Path, destination_image: &Path) -> Result<usize, String> {
    if source_image == destination_image {
        return Ok(0);
    }
    let copied_count = copy_sidecars(source_image, destination_image)?;
    remove_sidecars(source_image)?;
    Ok(copied_count)
}

fn count_files(directory: &Path) -> usize {
    let Ok(entries) = fs::read_dir(directory) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() { count_files(&path) } else { 1 }
        })
        .sum()
}

fn legacy_sidecar_paths(image_path: &Path) -> Vec<PathBuf> {
    let mut paths = vec![
        legacy_sidecar_path(image_path, None),
        legacy_rrexif_path(image_path),
    ];
    let Some(parent) = image_path.parent() else {
        return paths;
    };
    let image_name = image_path.file_name().unwrap_or_default().to_string_lossy();
    let prefix = format!("{}.", image_name);

    if let Ok(entries) = fs::read_dir(parent) {
        paths.extend(entries.filter_map(Result::ok).filter_map(|entry| {
            let filename = entry.file_name();
            let filename = filename.to_string_lossy();
            let copy_id = filename.strip_prefix(&prefix)?.strip_suffix(".rrdata")?;
            valid_copy_id(copy_id).then(|| entry.path())
        }));
    }
    paths
}

pub fn remove_sidecars(image_path: &Path) -> Result<usize, String> {
    let directory = image_storage_directory(image_path);
    let mut removed_count = count_files(&directory);
    if directory.exists() {
        fs::remove_dir_all(&directory).map_err(|error| {
            format!(
                "Failed to remove private sidecars for {}: {}",
                image_path.display(),
                error
            )
        })?;
        if let Some(shard_directory) = directory.parent()
            && shard_directory
                .read_dir()
                .is_ok_and(|mut entries| entries.next().is_none())
        {
            let _ = fs::remove_dir(shard_directory);
        }
    }

    for legacy_path in legacy_sidecar_paths(image_path) {
        if legacy_path.is_file() {
            fs::remove_file(&legacy_path).map_err(|error| {
                format!(
                    "Failed to remove legacy sidecar {}: {}",
                    legacy_path.display(),
                    error
                )
            })?;
            removed_count += 1;
        }
    }
    Ok(removed_count)
}

pub fn remove_virtual_copy(image_path: &Path, copy_id: &str) -> Result<bool, String> {
    let private_path = sidecar_path(image_path, Some(copy_id));
    let legacy_path = legacy_sidecar_path(image_path, Some(copy_id));
    let mut removed = false;
    for path in [private_path, legacy_path] {
        if path.is_file() {
            fs::remove_file(&path)
                .map_err(|error| format!("Failed to remove {}: {}", path.display(), error))?;
            removed = true;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_and_copies_sidecars_without_writing_next_to_photos() {
        let test_directory = tempfile::tempdir().expect("create test directory");
        let app_data_directory = test_directory.path().join("app-data");
        let photo_directory = test_directory.path().join("photos");
        fs::create_dir_all(&photo_directory).expect("create photo directory");
        initialize(&app_data_directory).expect("initialize private sidecar storage");

        let source_image = photo_directory.join("source.jpg");
        fs::write(&source_image, b"image").expect("create source image");
        let legacy_primary = legacy_sidecar_path(&source_image, None);
        let legacy_copy = legacy_sidecar_path(&source_image, Some("a1b2c3"));
        fs::write(
            &legacy_primary,
            br#"{"version":1,"rating":3,"adjustments":null}"#,
        )
        .expect("create legacy primary sidecar");
        fs::write(
            &legacy_copy,
            br#"{"version":1,"rating":5,"adjustments":null}"#,
        )
        .expect("create legacy virtual-copy sidecar");

        let private_primary = primary_sidecar_path(&source_image);
        assert!(private_primary.starts_with(&app_data_directory));
        assert!(!private_primary.starts_with(&photo_directory));
        assert!(private_primary.is_file());
        assert!(!legacy_primary.exists());

        assert_eq!(virtual_copy_ids(&source_image), vec!["a1b2c3"]);
        assert!(!legacy_copy.exists());

        let destination_image = photo_directory.join("destination.jpg");
        fs::write(&destination_image, b"image").expect("create destination image");
        assert_eq!(
            copy_sidecars(&source_image, &destination_image).expect("copy sidecars"),
            2
        );
        assert!(primary_sidecar_path(&destination_image).is_file());
        assert!(sidecar_path(&destination_image, Some("a1b2c3")).is_file());
        assert!(!legacy_sidecar_path(&destination_image, None).exists());
        assert!(!legacy_sidecar_path(&destination_image, Some("a1b2c3")).exists());
    }
}
