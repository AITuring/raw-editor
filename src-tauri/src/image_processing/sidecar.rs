//! Sidecar metadata and the edited-state check.
//!
//! `ImageMetadata` is the versioned per-image sidecar record; `is_image_edited`
//! tells whether its adjustments differ from an untouched image.

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use super::{
    Crop, get_all_adjustments_from_json, get_geometry_params_from_json, is_geometry_identity,
};

pub const SIDECAR_SCHEMA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub struct ImageMetadata {
    pub version: u32,
    pub rating: u8,
    pub adjustments: Value,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exif: Option<std::collections::HashMap<String, String>>,
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

impl Default for ImageMetadata {
    fn default() -> Self {
        ImageMetadata {
            version: SIDECAR_SCHEMA_VERSION,
            rating: 0,
            adjustments: Value::Null,
            tags: None,
            exif: None,
            extra: std::collections::BTreeMap::new(),
        }
    }
}

pub fn is_image_edited(
    adj: &serde_json::Value,
    is_raw: bool,
    tonemapper_override: Option<u32>,
) -> bool {
    if adj.is_null() || adj.as_object().is_none() {
        return false;
    }

    if let Some(patches) = adj.get("aiPatches").and_then(|v| v.as_array())
        && !patches.is_empty()
    {
        return true;
    }
    if let Some(masks) = adj.get("masks").and_then(|v| v.as_array())
        && !masks.is_empty()
    {
        return true;
    }

    if let Some(crop_val) = adj.get("crop")
        && !crop_val.is_null()
        && let Ok(crop) = serde_json::from_value::<Crop>(crop_val.clone())
        && (crop.x.abs() > 0.1 || crop.y.abs() > 0.1)
    {
        return true;
    }

    if adj
        .get("orientationSteps")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        != 0
    {
        return true;
    }
    if adj
        .get("rotation")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .abs()
        > 0.001
    {
        return true;
    }
    if adj
        .get("flipHorizontal")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return true;
    }
    if adj
        .get("flipVertical")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return true;
    }

    let geo = get_geometry_params_from_json(adj);
    if !is_geometry_identity(&geo) {
        return true;
    }

    let current_adj = get_all_adjustments_from_json(adj, is_raw, tonemapper_override);
    let default_adj =
        get_all_adjustments_from_json(&serde_json::json!({}), is_raw, tonemapper_override);

    bytemuck::bytes_of(&current_adj) != bytemuck::bytes_of(&default_adj)
}
