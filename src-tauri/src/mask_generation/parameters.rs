//! Sub-mask parameter types parsed from the adjustments JSON.
//!
//! Serde default functions stay next to the structs that name them.

use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub(super) struct GrowFeatherParameters {
    #[serde(default)]
    pub(super) grow: f32,
    #[serde(default)]
    pub(super) feather: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub(super) struct RadialMaskParameters {
    pub(super) center_x: f64,
    pub(super) center_y: f64,
    pub(super) radius_x: f64,
    pub(super) radius_y: f64,
    pub(super) rotation: f32,
    pub(super) feather: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub(super) struct LinearMaskParameters {
    pub(super) start_x: f64,
    pub(super) start_y: f64,
    pub(super) end_x: f64,
    pub(super) end_y: f64,
    #[serde(default = "default_range")]
    pub(super) range: f32,
}

pub(super) fn default_range() -> f32 {
    50.0
}

impl Default for LinearMaskParameters {
    fn default() -> Self {
        Self {
            start_x: 0.0,
            start_y: 0.0,
            end_x: 0.0,
            end_y: 0.0,
            range: default_range(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(super) struct Point {
    pub(super) x: f64,
    pub(super) y: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub(super) struct BrushLine {
    pub(super) tool: String,
    pub(super) brush_size: f32,
    pub(super) points: Vec<Point>,
    #[serde(default = "default_brush_feather")]
    pub(super) feather: f32,
}

pub(super) fn default_brush_feather() -> f32 {
    0.5
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub(super) struct BrushMaskParameters {
    #[serde(default)]
    pub(super) lines: Vec<BrushLine>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub(super) struct FlowLine {
    pub(super) tool: String,
    pub(super) brush_size: f32,
    pub(super) points: Vec<Point>,
    #[serde(default = "default_brush_feather")]
    pub(super) feather: f32,
    #[serde(default = "default_line_flow")]
    pub(super) flow: f32,
}

pub(super) fn default_line_flow() -> f32 {
    10.0
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub(super) struct FlowMaskParameters {
    #[serde(default)]
    pub(super) lines: Vec<FlowLine>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub(super) struct ParametricMaskParameters {
    pub(super) target_x: f64,
    pub(super) target_y: f64,
    #[serde(default = "default_tolerance")]
    pub(super) tolerance: f32,
    #[serde(default)]
    pub(super) grow: f32,
    #[serde(default)]
    pub(super) feather: f32,
    #[serde(default)]
    pub(super) rotation: f32,
    #[serde(default)]
    pub(super) flip_horizontal: bool,
    #[serde(default)]
    pub(super) flip_vertical: bool,
    #[serde(default)]
    pub(super) orientation_steps: u8,
}

pub(super) fn default_tolerance() -> f32 {
    20.0
}

impl Default for ParametricMaskParameters {
    fn default() -> Self {
        Self {
            target_x: 0.0,
            target_y: 0.0,
            tolerance: default_tolerance(),
            grow: 0.0,
            feather: 35.0,
            rotation: 0.0,
            flip_horizontal: false,
            flip_vertical: false,
            orientation_steps: 0,
        }
    }
}
