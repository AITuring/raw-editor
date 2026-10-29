//! Batch geometry correction: result types, lens data and adjustments.
//!
//! Matches EXIF lens data against the lens database for distortion parameters,
//! creates the result folder and builds the geometry adjustments a correction
//! export applies.

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchGeometryCorrectionResult {
    pub output_folder: String,
    pub processed_count: usize,
    pub content_orientation_corrected_count: usize,
    pub distortion_corrected_count: usize,
    pub missing_distortion_profile_count: usize,
}

/// A compact, rendered comparison for one source. Both images are generated
/// from the same geometry adjustments that the full-resolution exporter uses.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BatchGeometryPreviewItem {
    pub source_path: String,
    pub before_preview: String,
    pub after_preview: String,
    pub suggested_orientation_steps: u8,
    pub orientation_steps: u8,
    pub auto_corrected: bool,
    pub content_confidence: f32,
    pub distortion_applied: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchGeometryAnalysisResult {
    pub items: Vec<BatchGeometryPreviewItem>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchGeometryOrientationOverride {
    pub path: String,
    pub orientation_steps: u8,
}

pub(super) fn normalize_batch_geometry_output_format(value: &str) -> Result<&'static str, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => Ok("jpg"),
        "png" => Ok("png"),
        "tif" | "tiff" => Ok("tiff"),
        _ => Err("Batch correction supports JPEG, PNG, and TIFF output.".to_string()),
    }
}

pub(super) fn create_batch_geometry_output_folder(output_root: &Path) -> Result<PathBuf, String> {
    if output_root.as_os_str().is_empty() {
        return Err("Choose a folder for the corrected images.".to_string());
    }

    fs::create_dir_all(output_root).map_err(|error| {
        format!(
            "Failed to create the selected output folder '{}': {error}",
            output_root.display()
        )
    })?;

    for index in 1..=9_999usize {
        let folder_name = if index == 1 {
            "RAW Editor Corrected".to_string()
        } else {
            format!("RAW Editor Corrected {index}")
        };
        let candidate = output_root.join(folder_name);

        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "Failed to create the result folder '{}': {error}",
                    candidate.display()
                ));
            }
        }
    }

    Err("Could not create a unique result folder. Please choose another location.".to_string())
}

fn parse_batch_geometry_exif_number(value: Option<&String>) -> Option<f32> {
    let value = value?.trim();
    let mut number = String::new();
    let mut started = false;

    for character in value.chars() {
        let is_number_character =
            character.is_ascii_digit() || character == '.' || character == '-';
        if !started {
            if is_number_character {
                started = true;
                number.push(character);
            }
        } else if is_number_character {
            number.push(character);
        } else {
            break;
        }
    }

    number
        .parse::<f32>()
        .ok()
        .filter(|number| number.is_finite())
}

pub(super) fn has_usable_distortion_params(params: &Value) -> bool {
    ["k1", "k2", "k3"].into_iter().any(|key| {
        params
            .get(key)
            .and_then(Value::as_f64)
            .is_some_and(|value| value.abs() > 1e-6)
    })
}

pub(super) fn find_batch_distortion_params(
    lens_database: Option<&crate::lens_correction::LensDatabase>,
    metadata: &HashMap<String, String>,
) -> Option<Value> {
    let lens_database = lens_database?;
    let maker = metadata
        .get("LensMake")
        .or_else(|| metadata.get("Make"))?
        .trim();
    let model = metadata.get("LensModel")?.trim();
    let focal_length = parse_batch_geometry_exif_number(metadata.get("FocalLength"))?;
    if maker.is_empty() || model.is_empty() || focal_length <= 0.0 {
        return None;
    }

    let (matched_maker, matched_model) =
        crate::lens_correction::find_best_lens_match(lens_database, maker, model)?;
    let aperture = parse_batch_geometry_exif_number(metadata.get("FNumber"));
    let distance = parse_batch_geometry_exif_number(metadata.get("SubjectDistance"));
    let params = crate::lens_correction::resolve_lens_params(
        lens_database,
        &matched_maker,
        &matched_model,
        focal_length,
        aperture,
        distance,
    )?;

    serde_json::to_value(params).ok()
}

pub(super) fn build_batch_geometry_adjustments(
    content_orientation_steps: u8,
    distortion_params: Option<Value>,
    apply_distortion: bool,
) -> Value {
    let has_distortion = apply_distortion
        && distortion_params
            .as_ref()
            .is_some_and(has_usable_distortion_params);

    serde_json::json!({
        "crop": Value::Null,
        "flipHorizontal": false,
        "flipVertical": false,
        "orientationSteps": content_orientation_steps.min(3),
        "rotation": 0.0,
        "sectionVisibility": {
            "geometry": true,
            "optics": true,
        },
        "transformAspect": 0.0,
        "transformDistortion": 0.0,
        "transformHorizontal": 0.0,
        "transformProjection": 0.0,
        "transformRotate": 0.0,
        "transformScale": 100.0,
        "transformVertical": 0.0,
        "transformXOffset": 0.0,
        "transformYOffset": 0.0,
        "lensDistortionAmount": 100.0,
        "lensDistortionEnabled": has_distortion,
        "lensDistortionParams": distortion_params,
        "lensTcaAmount": 100.0,
        "lensTcaEnabled": false,
        "lensVignetteAmount": 100.0,
        "lensVignetteEnabled": false,
        "masks": [],
    })
}
