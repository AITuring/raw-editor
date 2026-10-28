//! Export settings and their defaults.
//!
//! The serialised export settings, the serde defaults they rely on, the
//! export bit-depth rules and the resize target an export setting describes.

use crate::exif_processing;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use super::WatermarkSettings;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum ResizeMode {
    LongEdge,
    ShortEdge,
    Width,
    Height,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ResizeOptions {
    pub mode: ResizeMode,
    pub value: u32,
    pub dont_enlarge: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ExportSettings {
    #[serde(default = "default_export_bit_depth")]
    pub bit_depth: u8,
    pub jpeg_quality: u8,
    pub resize: Option<ResizeOptions>,
    pub keep_metadata: bool,
    #[serde(default)]
    pub metadata_overrides: Option<exif_processing::ExportMetadataOverrides>,
    #[serde(default)]
    pub preserve_timestamps: bool,
    pub strip_gps: bool,
    #[serde(default = "default_embed_color_profile")]
    pub embed_color_profile: bool,
    pub filename_template: Option<String>,
    pub watermark: Option<WatermarkSettings>,
    #[serde(default)]
    pub export_masks: bool,
    #[serde(default)]
    pub preserve_folders: bool,
}

const fn default_embed_color_profile() -> bool {
    true
}

pub(crate) fn default_export_bit_depth_for_format(output_format: &str) -> u8 {
    match output_format.to_ascii_lowercase().as_str() {
        "png" | "tif" | "tiff" => 16,
        _ => 8,
    }
}

pub(crate) fn effective_export_bit_depth(output_format: &str, requested_bit_depth: u8) -> u8 {
    if requested_bit_depth == 16
        && matches!(
            output_format.to_ascii_lowercase().as_str(),
            "png" | "tif" | "tiff"
        )
    {
        16
    } else {
        8
    }
}

const fn default_export_bit_depth() -> u8 {
    8
}

#[derive(Clone)]
pub(crate) enum ExportAdjustmentsMode {
    UseSidecars {
        active_path: Option<String>,
        active_adjustments: Option<Value>,
    },
    GlobalOverride(Value),
    /// A deliberate, per-source override for self-contained batch workflows.
    /// Unlike sidecars this never reads or mutates an editor adjustment.
    PerPathOverrides(Arc<HashMap<String, Value>>),
}

impl ExportAdjustmentsMode {
    pub(super) fn normalize_active_path(self) -> Self {
        match self {
            Self::UseSidecars {
                active_path,
                active_adjustments,
            } => Self::UseSidecars {
                active_path: active_path.map(|path| {
                    path.rsplit_once("?vc=")
                        .map(|(physical_path, _)| physical_path.to_string())
                        .unwrap_or(path)
                }),
                active_adjustments,
            },
            Self::GlobalOverride(adjustments) => Self::GlobalOverride(adjustments),
            Self::PerPathOverrides(overrides) => Self::PerPathOverrides(overrides),
        }
    }
}

pub(crate) fn calculate_resize_target(
    current_w: u32,
    current_h: u32,
    resize_opts: &ResizeOptions,
) -> (u32, u32) {
    if resize_opts.dont_enlarge {
        let exceeds = match resize_opts.mode {
            ResizeMode::LongEdge => current_w.max(current_h) > resize_opts.value,
            ResizeMode::ShortEdge => current_w.min(current_h) > resize_opts.value,
            ResizeMode::Width => current_w > resize_opts.value,
            ResizeMode::Height => current_h > resize_opts.value,
        };
        if !exceeds {
            return (current_w, current_h);
        }
    }

    let fix_width = match resize_opts.mode {
        ResizeMode::LongEdge => current_w >= current_h,
        ResizeMode::ShortEdge => current_w <= current_h,
        ResizeMode::Width => true,
        ResizeMode::Height => false,
    };

    let value = resize_opts.value;
    if fix_width {
        let h = (value as f32 * (current_h as f32 / current_w as f32)).round() as u32;
        (value, h)
    } else {
        let w = (value as f32 * (current_w as f32 / current_h as f32)).round() as u32;
        (w, value)
    }
}
