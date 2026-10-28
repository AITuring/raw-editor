//! Geometry parameters of an edit.
//!
//! Crop rectangle, the transform/lens parameter record read from the
//! adjustments JSON (respecting section visibility) and the identity check
//! that lets callers skip the warp.

use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Crop {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(default)]
pub struct GeometryParams {
    pub distortion: f32,
    pub projection: f32,
    pub vertical: f32,
    pub horizontal: f32,
    pub rotate: f32,
    pub aspect: f32,
    pub scale: f32,
    pub x_offset: f32,
    pub y_offset: f32,
    pub lens_distortion_amount: f32,
    pub lens_vignette_amount: f32,
    pub lens_tca_amount: f32,
    pub lens_distortion_enabled: bool,
    pub lens_tca_enabled: bool,
    pub lens_vignette_enabled: bool,
    pub lens_dist_k1: f32,
    pub lens_dist_k2: f32,
    pub lens_dist_k3: f32,
    pub lens_model: u32,
    pub tca_vr: f32,
    pub tca_vb: f32,
    pub vig_k1: f32,
    pub vig_k2: f32,
    pub vig_k3: f32,
}

impl Default for GeometryParams {
    fn default() -> Self {
        Self {
            distortion: 0.0,
            projection: 0.0,
            vertical: 0.0,
            horizontal: 0.0,
            rotate: 0.0,
            aspect: 0.0,
            scale: 100.0,
            x_offset: 0.0,
            y_offset: 0.0,
            lens_distortion_amount: 1.0,
            lens_vignette_amount: 1.0,
            lens_tca_amount: 1.0,
            lens_distortion_enabled: true,
            lens_tca_enabled: true,
            lens_vignette_enabled: true,
            lens_dist_k1: 0.0,
            lens_dist_k2: 0.0,
            lens_dist_k3: 0.0,
            lens_model: 0,
            tca_vr: 1.0,
            tca_vb: 1.0,
            vig_k1: 0.0,
            vig_k2: 0.0,
            vig_k3: 0.0,
        }
    }
}

pub fn get_geometry_params_from_json(adjustments: &serde_json::Value) -> GeometryParams {
    let visibility = adjustments.get("sectionVisibility");
    let section_visible = |section: &str, legacy: Option<&str>| -> bool {
        visibility
            .and_then(|v| v.get(section))
            .and_then(|s| s.as_bool())
            .or_else(|| {
                legacy.and_then(|legacy_section| {
                    visibility
                        .and_then(|v| v.get(legacy_section))
                        .and_then(|s| s.as_bool())
                })
            })
            .unwrap_or(true)
    };
    let geometry_visible = section_visible("geometry", None);
    let optics_visible = section_visible("optics", Some("details"));
    let lens_params = optics_visible
        .then(|| {
            adjustments
                .get("lensDistortionParams")
                .and_then(|v| v.as_object())
        })
        .flatten();

    GeometryParams {
        distortion: if optics_visible {
            adjustments["transformDistortion"].as_f64().unwrap_or(0.0) as f32
        } else {
            0.0
        },
        projection: if geometry_visible {
            adjustments["transformProjection"].as_f64().unwrap_or(0.0) as f32
        } else {
            0.0
        },
        vertical: if geometry_visible {
            adjustments["transformVertical"].as_f64().unwrap_or(0.0) as f32
        } else {
            0.0
        },
        horizontal: if geometry_visible {
            adjustments["transformHorizontal"].as_f64().unwrap_or(0.0) as f32
        } else {
            0.0
        },
        rotate: if geometry_visible {
            adjustments["transformRotate"].as_f64().unwrap_or(0.0) as f32
        } else {
            0.0
        },
        aspect: if geometry_visible {
            adjustments["transformAspect"].as_f64().unwrap_or(0.0) as f32
        } else {
            0.0
        },
        scale: if geometry_visible {
            adjustments["transformScale"].as_f64().unwrap_or(100.0) as f32
        } else {
            100.0
        },
        x_offset: if geometry_visible {
            adjustments["transformXOffset"].as_f64().unwrap_or(0.0) as f32
        } else {
            0.0
        },
        y_offset: if geometry_visible {
            adjustments["transformYOffset"].as_f64().unwrap_or(0.0) as f32
        } else {
            0.0
        },

        lens_distortion_amount: if optics_visible {
            adjustments["lensDistortionAmount"]
                .as_f64()
                .unwrap_or(100.0) as f32
                / 100.0
        } else {
            1.0
        },
        lens_vignette_amount: if optics_visible {
            adjustments["lensVignetteAmount"].as_f64().unwrap_or(100.0) as f32 / 100.0
        } else {
            1.0
        },
        lens_tca_amount: if optics_visible {
            adjustments["lensTcaAmount"].as_f64().unwrap_or(100.0) as f32 / 100.0
        } else {
            1.0
        },
        lens_distortion_enabled: optics_visible
            && adjustments["lensDistortionEnabled"]
                .as_bool()
                .unwrap_or(true),
        lens_tca_enabled: optics_visible && adjustments["lensTcaEnabled"].as_bool().unwrap_or(true),
        lens_vignette_enabled: optics_visible
            && adjustments["lensVignetteEnabled"].as_bool().unwrap_or(true),

        lens_dist_k1: lens_params
            .and_then(|p| p.get("k1").and_then(|k| k.as_f64()))
            .unwrap_or(0.0) as f32,
        lens_dist_k2: lens_params
            .and_then(|p| p.get("k2").and_then(|k| k.as_f64()))
            .unwrap_or(0.0) as f32,
        lens_dist_k3: lens_params
            .and_then(|p| p.get("k3").and_then(|k| k.as_f64()))
            .unwrap_or(0.0) as f32,
        lens_model: lens_params
            .and_then(|p| p.get("model").and_then(|m| m.as_u64()))
            .unwrap_or(0) as u32,
        tca_vr: lens_params
            .and_then(|p| p.get("tca_vr").and_then(|k| k.as_f64()))
            .unwrap_or(1.0) as f32,
        tca_vb: lens_params
            .and_then(|p| p.get("tca_vb").and_then(|k| k.as_f64()))
            .unwrap_or(1.0) as f32,
        vig_k1: lens_params
            .and_then(|p| p.get("vig_k1").and_then(|k| k.as_f64()))
            .unwrap_or(0.0) as f32,
        vig_k2: lens_params
            .and_then(|p| p.get("vig_k2").and_then(|k| k.as_f64()))
            .unwrap_or(0.0) as f32,
        vig_k3: lens_params
            .and_then(|p| p.get("vig_k3").and_then(|k| k.as_f64()))
            .unwrap_or(0.0) as f32,
    }
}

pub fn is_geometry_identity(params: &GeometryParams) -> bool {
    let dist_identity = !params.lens_distortion_enabled
        || ((params.lens_distortion_amount - 1.0).abs() < 1e-4
            && params.lens_dist_k1.abs() < 1e-6
            && params.lens_dist_k2.abs() < 1e-6
            && params.lens_dist_k3.abs() < 1e-6);

    let tca_identity = !params.lens_tca_enabled
        || ((params.lens_tca_amount - 1.0).abs() < 1e-4
            && (params.tca_vr - 1.0).abs() < 1e-6
            && (params.tca_vb - 1.0).abs() < 1e-6);

    let vig_identity = !params.lens_vignette_enabled
        || ((params.lens_vignette_amount - 1.0).abs() < 1e-4
            && params.vig_k1.abs() < 1e-6
            && params.vig_k2.abs() < 1e-6
            && params.vig_k3.abs() < 1e-6);

    params.distortion == 0.0
        && params.projection == 0.0
        && params.vertical == 0.0
        && params.horizontal == 0.0
        && params.rotate == 0.0
        && params.aspect == 0.0
        && params.scale == 100.0
        && params.x_offset == 0.0
        && params.y_offset == 0.0
        && dist_identity
        && tca_identity
        && vig_identity
}
