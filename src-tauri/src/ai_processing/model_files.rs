//! Model and ONNX Runtime files on disk.
//!
//! The model directory, atomic writes of downloaded assets, SHA-256
//! verification, legacy file-name migration and the per-platform ONNX Runtime
//! asset.

use anyhow::Result;
use sha2::Digest;
use sha2::Sha256;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use tauri::Emitter;
use tauri::Manager;

use super::{SKYSEG_FILENAME, SKYSEG_LEGACY_FILENAME, SKYSEG_SHA256, download_model};

/// ONNX Runtime is normally bundled with release builds. Development builds can
/// intentionally omit it, so an on-device AI action provisions this verified
/// runtime beside its models before opening the first session.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub(super) const ONNX_RUNTIME_ASSET: (&str, &str, &str) = (
    "onnxruntime.dll",
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/onnxruntimes-v1.22.0/onnxruntime-windows-x86_64.dll?download=true",
    "579b636403983254346a5c1d80bd28f1519cd1e284cd204f8d4ff41f8d711559",
);

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
pub(super) const ONNX_RUNTIME_ASSET: (&str, &str, &str) = (
    "onnxruntime.dll",
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/onnxruntimes-v1.22.0/onnxruntime-windows-aarch64.dll?download=true",
    "79281671a386ed1baab9dbdbb09fe55f99577011472e9526cf9d0b468bb6bcc7",
);

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(super) const ONNX_RUNTIME_ASSET: (&str, &str, &str) = (
    "libonnxruntime.so",
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/onnxruntimes-v1.22.0/libonnxruntime-linux-x86_64.so?download=true",
    "3da6146e14e7b8aaec625dde11d6114c7457c87a5f93d744897da8781e35c673",
);

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub(super) const ONNX_RUNTIME_ASSET: (&str, &str, &str) = (
    "libonnxruntime.so",
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/onnxruntimes-v1.22.0/libonnxruntime-linux-aarch64.so?download=true",
    "0afd69a0ae38c5099fd0e8604dda398ac43dee67cd9c6394b5142b19e82528de",
);

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) const ONNX_RUNTIME_ASSET: (&str, &str, &str) = (
    "libonnxruntime.dylib",
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/onnxruntimes-v1.22.0/libonnxruntime-macos-x86_64.dylib?download=true",
    "283e595e61cf65df7a6b1d59a1616cbd35c8b6399dd90d799d99b71a3ff83160",
);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) const ONNX_RUNTIME_ASSET: (&str, &str, &str) = (
    "libonnxruntime.dylib",
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/onnxruntimes-v1.22.0/libonnxruntime-macos-aarch64.dylib?download=true",
    "2b885992d3d6fa4130d39ec84a80d7504ff52750027c547bb22c86165f19406a",
);

#[cfg(not(target_os = "android"))]
pub(super) fn onnx_runtime_asset() -> Result<(&'static str, &'static str, &'static str)> {
    #[cfg(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    ))]
    {
        Ok(ONNX_RUNTIME_ASSET)
    }

    #[cfg(not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    )))]
    {
        Err(anyhow::anyhow!(
            "No verified ONNX Runtime asset is available for this platform."
        ))
    }
}

pub(super) fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf> {
    let models_dir = app_handle.path().app_data_dir()?.join("models");
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir)?;
    }
    Ok(models_dir)
}

pub(super) fn persist_downloaded_asset(dest: &Path, bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        return Err(anyhow::anyhow!(
            "Downloaded asset for {} was empty",
            dest.display()
        ));
    }

    let parent = dest.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "Cannot determine parent directory for downloaded asset {}",
            dest.display()
        )
    })?;
    fs::create_dir_all(parent)?;

    let file_name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid downloaded asset path {}", dest.display()))?;
    let tmp_path = dest.with_file_name(format!(".{}.download", file_name));

    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }

    fs::rename(&tmp_path, dest).or_else(|rename_error| -> std::io::Result<()> {
        if dest.exists() {
            fs::remove_file(dest)?;
            fs::rename(&tmp_path, dest)?;
            Ok(())
        } else {
            Err(rename_error)
        }
    })?;
    Ok(())
}

pub(super) fn verify_sha256(path: &Path, expected_hash: &str) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let hash = hasher.finalize();
    let hex_hash = hex::encode(hash);
    Ok(hex_hash == expected_hash)
}

pub(super) fn promote_legacy_model_filename(
    models_dir: &Path,
    expected_filename: &str,
    legacy_filename: &str,
    expected_hash: &str,
) -> Result<()> {
    let expected_path = models_dir.join(expected_filename);
    if expected_path.exists() {
        return Ok(());
    }

    let legacy_path = models_dir.join(legacy_filename);
    if !legacy_path.exists() || !verify_sha256(&legacy_path, expected_hash)? {
        return Ok(());
    }

    fs::rename(&legacy_path, &expected_path).or_else(|rename_error| -> std::io::Result<()> {
        if expected_path.exists() {
            Ok(())
        } else {
            Err(rename_error)
        }
    })?;
    Ok(())
}

pub(super) async fn download_and_verify_model(
    app_handle: &tauri::AppHandle,
    models_dir: &Path,
    filename: &str,
    url: &str,
    expected_hash: &str,
    model_name: &str,
) -> Result<()> {
    let dest_path = models_dir.join(filename);
    if filename == SKYSEG_FILENAME {
        promote_legacy_model_filename(
            models_dir,
            SKYSEG_FILENAME,
            SKYSEG_LEGACY_FILENAME,
            SKYSEG_SHA256,
        )?;
    }
    let is_valid = verify_sha256(&dest_path, expected_hash)?;

    if !is_valid {
        if dest_path.exists() {
            println!("Model {} has incorrect hash. Re-downloading.", model_name);
            fs::remove_file(&dest_path)?;
        }
        let _ = app_handle.emit("ai-model-download-start", model_name);
        let download_result = download_model(url, &dest_path).await;
        let _ = app_handle.emit("ai-model-download-finish", model_name);
        download_result?;

        if !verify_sha256(&dest_path, expected_hash)? {
            return Err(anyhow::anyhow!(
                "Failed to verify model {} after download. Hash mismatch.",
                model_name
            ));
        }
    }
    Ok(())
}

/// Ensures a valid dynamic ONNX Runtime is selected before constructing a
/// session. Prefer the release-bundled library; development builds fall back
/// to a hash-verified, app-data copy downloaded on the first AI action.
#[cfg(not(target_os = "android"))]
pub(super) async fn ensure_onnx_runtime(
    app_handle: &tauri::AppHandle,
    models_dir: &Path,
) -> Result<()> {
    if let Some(configured_path) = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from)
        && configured_path.is_file()
    {
        return Ok(());
    }

    let (filename, url, expected_hash) = onnx_runtime_asset()?;
    let bundled_path = app_handle
        .path()
        .resolve("resources", tauri::path::BaseDirectory::Resource)
        .ok()
        .map(|resource_path| resource_path.join(filename));

    let runtime_path = match bundled_path.filter(|path| path.is_file()) {
        Some(path) => path,
        None => {
            download_and_verify_model(
                app_handle,
                models_dir,
                filename,
                url,
                expected_hash,
                "ONNX Runtime",
            )
            .await?;
            models_dir.join(filename)
        }
    };

    // `ort` reads this during its first global initialization. This function is
    // called while the shared AI initialization lock is held, before a session
    // is built, so the process-wide path cannot race another model setup.
    unsafe {
        std::env::set_var("ORT_DYLIB_PATH", &runtime_path);
    }
    Ok(())
}

#[cfg(target_os = "android")]
pub(super) async fn ensure_onnx_runtime(
    _app_handle: &tauri::AppHandle,
    _models_dir: &Path,
) -> Result<()> {
    // Android packages its native runtime with the app rather than loading a
    // desktop dylib path at run time.
    Ok(())
}
