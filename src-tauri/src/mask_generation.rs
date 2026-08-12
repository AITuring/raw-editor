use crate::ai_processing::{
    AiDepthMaskParameters, AiForegroundMaskParameters, AiSkyMaskParameters, AiSubjectMaskParameters,
};
use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, GenericImageView, GrayImage, ImageFormat, Luma, Rgba, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::f32::consts::PI;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::sync::Arc; // Required for parallel rasterization
use tauri::ipc::Response;

use crate::app_state::{AppState, SharedMaskBitmap};
use crate::get_cached_full_warped_image;
use crate::render_strategy::GPU_TILE_SIZE;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub enum SubMaskMode {
    Additive,
    Subtractive,
    Intersect,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct SubMask {
    pub id: String,
    #[serde(rename = "type")]
    pub mask_type: String,
    pub visible: bool,
    #[serde(default)]
    pub invert: bool,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    pub mode: SubMaskMode,
    pub parameters: Value,
}

fn default_opacity() -> f32 {
    100.0
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct MaskDefinition {
    pub id: String,
    pub name: String,
    pub visible: bool,
    pub invert: bool,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    pub adjustments: Value,
    pub sub_masks: Vec<SubMask>,
}

impl MaskDefinition {
    pub fn requires_warped_image(&self) -> bool {
        self.sub_masks
            .iter()
            .any(|sm| sm.mask_type == "color" || sm.mask_type == "luminance")
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct PatchData {
    pub color: String,
    pub mask: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct AiPatchDefinition {
    pub id: String,
    pub name: String,
    pub visible: bool,
    pub invert: bool,
    pub prompt: String,
    #[serde(default)]
    pub patch_data: Option<PatchData>,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    pub sub_masks: Vec<SubMask>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct GrowFeatherParameters {
    #[serde(default)]
    grow: f32,
    #[serde(default)]
    feather: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct RadialMaskParameters {
    center_x: f64,
    center_y: f64,
    radius_x: f64,
    radius_y: f64,
    rotation: f32,
    feather: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct LinearMaskParameters {
    start_x: f64,
    start_y: f64,
    end_x: f64,
    end_y: f64,
    #[serde(default = "default_range")]
    range: f32,
}

fn default_range() -> f32 {
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
struct Point {
    x: f64,
    y: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct BrushLine {
    tool: String,
    brush_size: f32,
    points: Vec<Point>,
    #[serde(default = "default_brush_feather")]
    feather: f32,
}

fn default_brush_feather() -> f32 {
    0.5
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct BrushMaskParameters {
    #[serde(default)]
    lines: Vec<BrushLine>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct FlowLine {
    tool: String,
    brush_size: f32,
    points: Vec<Point>,
    #[serde(default = "default_brush_feather")]
    feather: f32,
    #[serde(default = "default_line_flow")]
    flow: f32,
}

fn default_line_flow() -> f32 {
    10.0
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct FlowMaskParameters {
    #[serde(default)]
    lines: Vec<FlowLine>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct ParametricMaskParameters {
    target_x: f64,
    target_y: f64,
    #[serde(default = "default_tolerance")]
    tolerance: f32,
    #[serde(default)]
    grow: f32,
    #[serde(default)]
    feather: f32,
    #[serde(default)]
    rotation: f32,
    #[serde(default)]
    flip_horizontal: bool,
    #[serde(default)]
    flip_vertical: bool,
    #[serde(default)]
    orientation_steps: u8,
}

fn default_tolerance() -> f32 {
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

fn grayscale_dilate(image: &GrayImage, k: u8) -> GrayImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return image.clone();
    }
    let w = width as usize;
    let h = height as usize;
    let r = k as i32;
    let src = image.as_raw();

    let mut temp = vec![0u8; w * h];
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let row_offset = y * w;
        for x in 0..w {
            let mut max_val = 0;
            let start = (x as i32 - r).max(0) as usize;
            let end = (x as i32 + r).min((w - 1) as i32) as usize;
            for xi in start..=end {
                max_val = max_val.max(src[row_offset + xi]);
            }
            temp[row_offset + x] = max_val;
        }
    }

    for x in 0..w {
        for y in 0..h {
            let mut max_val = 0;
            let start = (y as i32 - r).max(0) as usize;
            let end = (y as i32 + r).min((h - 1) as i32) as usize;
            for yi in start..=end {
                max_val = max_val.max(temp[yi * w + x]);
            }
            out[y * w + x] = max_val;
        }
    }

    GrayImage::from_raw(width, height, out).unwrap()
}

fn grayscale_erode(image: &GrayImage, k: u8) -> GrayImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return image.clone();
    }
    let w = width as usize;
    let h = height as usize;
    let r = k as i32;
    let src = image.as_raw();

    let mut temp = vec![0u8; w * h];
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let row_offset = y * w;
        for x in 0..w {
            let mut min_val = 255;
            let start = (x as i32 - r).max(0) as usize;
            let end = (x as i32 + r).min((w - 1) as i32) as usize;
            for xi in start..=end {
                min_val = min_val.min(src[row_offset + xi]);
            }
            temp[row_offset + x] = min_val;
        }
    }

    for x in 0..w {
        for y in 0..h {
            let mut min_val = 255;
            let start = (y as i32 - r).max(0) as usize;
            let end = (y as i32 + r).min((h - 1) as i32) as usize;
            for yi in start..=end {
                min_val = min_val.min(temp[yi * w + x]);
            }
            out[y * w + x] = min_val;
        }
    }

    GrayImage::from_raw(width, height, out).unwrap()
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct GrowFeatherPlan {
    grow_amount: u8,
    dilate: bool,
    feather_sigma: f32,
    feather_radius: u32,
}

impl GrowFeatherPlan {
    fn new(grow: f32, feather: f32, width: u32, height: u32) -> Self {
        let base_dimension = width.min(height) as f32;
        let grow_pixels = if grow.abs() > 0.01 {
            const MAX_GROW_PERCENTAGE: f32 = 0.01;
            (grow / 100.0) * base_dimension * MAX_GROW_PERCENTAGE
        } else {
            0.0
        };
        let grow_amount = grow_pixels.abs().round() as u8;

        let feather_sigma = if feather > 0.0 {
            const MAX_FEATHER_SIGMA_PERCENTAGE: f32 = 0.005;
            let sigma = (feather / 100.0) * base_dimension * MAX_FEATHER_SIGMA_PERCENTAGE;
            if sigma > 0.01 { sigma } else { 0.0 }
        } else {
            0.0
        };

        Self {
            grow_amount,
            dilate: grow_pixels > 0.0,
            feather_sigma,
            // imageproc::gaussian_blur_f32 constructs a finite kernel with this radius.
            feather_radius: (2.0 * feather_sigma).ceil() as u32,
        }
    }

    fn halo(self) -> u32 {
        u32::from(self.grow_amount).saturating_add(self.feather_radius)
    }

    fn has_effect(self) -> bool {
        self.grow_amount > 0 || self.feather_sigma > 0.0
    }

    fn apply(self, mask: &mut GrayImage) {
        if self.grow_amount > 0 {
            if self.dilate {
                *mask = grayscale_dilate(mask, self.grow_amount);
            } else {
                *mask = grayscale_erode(mask, self.grow_amount);
            }
        }

        if self.feather_sigma > 0.0 {
            *mask = imageproc::filter::gaussian_blur_f32(mask, self.feather_sigma);
        }
    }
}

fn apply_grow_and_feather(mask: &mut GrayImage, grow: f32, feather: f32, width: u32, height: u32) {
    GrowFeatherPlan::new(grow, feather, width, height).apply(mask);
}

fn stroke_bounds(
    points: &[Point],
    width: u32,
    height: u32,
    radius: f32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    if width == 0 || height == 0 || points.is_empty() {
        return None;
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let r_pad = radius.ceil() + 2.0;

    for p in points {
        let px = p.x as f32 * scale - crop_offset.0 - output_origin.0 as f32;
        let py = p.y as f32 * scale - crop_offset.1 - output_origin.1 as f32;

        min_x = min_x.min(px - r_pad);
        min_y = min_y.min(py - r_pad);
        max_x = max_x.max(px + r_pad);
        max_y = max_y.max(py + r_pad);
    }

    if max_x < 0.0 || max_y < 0.0 || min_x > (width - 1) as f32 || min_y > (height - 1) as f32 {
        return None;
    }

    let min_x = min_x.floor().max(0.0).min((width - 1) as f32) as u32;
    let min_y = min_y.floor().max(0.0).min((height - 1) as f32) as u32;
    let max_x = max_x.ceil().max(0.0).min((width - 1) as f32) as u32;
    let max_y = max_y.ceil().max(0.0).min((height - 1) as f32) as u32;

    if min_x > max_x || min_y > max_y {
        None
    } else {
        Some((min_x, min_y, max_x, max_y))
    }
}

#[allow(clippy::too_many_arguments)]
fn render_stroke_layer_parallel(
    points: &[Point],
    radius: f32,
    feather: f32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
    layer_offset: (f32, f32),
    bb_w: u32,
    bb_h: u32,
) -> GrayImage {
    let mut out_pixels = vec![0u8; (bb_w * bb_h) as usize];
    if points.is_empty() || radius <= 0.0 {
        return GrayImage::from_raw(bb_w, bb_h, out_pixels).unwrap();
    }

    struct Segment {
        x1: f32,
        y1: f32,
        dx: f32,
        dy: f32,
        len_sq: f32,
        bounds_left: i32,
        bounds_right: i32,
        bounds_top: i32,
        bounds_bottom: i32,
    }

    let mut segments = Vec::with_capacity(points.len().saturating_sub(1));
    for pair in points.windows(2) {
        let x1 = pair[0].x as f32 * scale - crop_offset.0 - output_origin.0 as f32 - layer_offset.0;
        let y1 = pair[0].y as f32 * scale - crop_offset.1 - output_origin.1 as f32 - layer_offset.1;
        let x2 = pair[1].x as f32 * scale - crop_offset.0 - output_origin.0 as f32 - layer_offset.0;
        let y2 = pair[1].y as f32 * scale - crop_offset.1 - output_origin.1 as f32 - layer_offset.1;

        let left = ((x1.min(x2) - radius).floor() as i32).max(0);
        let right = ((x1.max(x2) + radius).ceil() as i32).min(bb_w as i32 - 1);
        let top = ((y1.min(y2) - radius).floor() as i32).max(0);
        let bottom = ((y1.max(y2) + radius).ceil() as i32).min(bb_h as i32 - 1);

        if left > right || top > bottom {
            continue;
        }

        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;

        segments.push(Segment {
            x1,
            y1,
            dx,
            dy,
            len_sq,
            bounds_left: left,
            bounds_right: right,
            bounds_top: top,
            bounds_bottom: bottom,
        });
    }

    let mut single_point = None;
    if segments.is_empty() && !points.is_empty() {
        let x1 =
            points[0].x as f32 * scale - crop_offset.0 - output_origin.0 as f32 - layer_offset.0;
        let y1 =
            points[0].y as f32 * scale - crop_offset.1 - output_origin.1 as f32 - layer_offset.1;
        let left = ((x1 - radius).floor() as i32).max(0);
        let right = ((x1 + radius).ceil() as i32).min(bb_w as i32 - 1);
        let top = ((y1 - radius).floor() as i32).max(0);
        let bottom = ((y1 + radius).ceil() as i32).min(bb_h as i32 - 1);
        if left <= right && top <= bottom {
            single_point = Some((x1, y1, left, right, top, bottom));
        }
    }

    let feather_amount = feather.clamp(0.0, 1.0);
    let inner_radius = radius * (1.0 - feather_amount);
    let feather_range = (radius - inner_radius).max(0.01);
    let radius_sq = radius * radius;
    let inner_radius_sq = inner_radius * inner_radius;

    out_pixels
        .par_chunks_mut(bb_w as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let py = y as f32;
            let y_i32 = y as i32;

            let mut active_segments = Vec::new();
            for seg in &segments {
                if y_i32 >= seg.bounds_top && y_i32 <= seg.bounds_bottom {
                    active_segments.push(seg);
                }
            }

            let is_point_active = if let Some(pt) = &single_point {
                y_i32 >= pt.4 && y_i32 <= pt.5
            } else {
                false
            };

            if active_segments.is_empty() && !is_point_active {
                return;
            }

            for (x, pixel) in row.iter_mut().enumerate() {
                let px = x as f32;
                let x_i32 = x as i32;

                let mut min_dist_sq = radius_sq + 1.0;

                for seg in &active_segments {
                    if x_i32 >= seg.bounds_left && x_i32 <= seg.bounds_right {
                        let dist_sq = if seg.len_sq < 0.0001 {
                            (px - seg.x1) * (px - seg.x1) + (py - seg.y1) * (py - seg.y1)
                        } else {
                            let t = (((px - seg.x1) * seg.dx + (py - seg.y1) * seg.dy)
                                / seg.len_sq)
                                .clamp(0.0, 1.0);
                            let proj_x = seg.x1 + t * seg.dx;
                            let proj_y = seg.y1 + t * seg.dy;
                            (px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y)
                        };
                        if dist_sq < min_dist_sq {
                            min_dist_sq = dist_sq;
                        }
                    }
                }

                if is_point_active {
                    let pt = single_point.as_ref().unwrap();
                    if x_i32 >= pt.2 && x_i32 <= pt.3 {
                        let dist_sq = (px - pt.0) * (px - pt.0) + (py - pt.1) * (py - pt.1);
                        if dist_sq < min_dist_sq {
                            min_dist_sq = dist_sq;
                        }
                    }
                }

                if min_dist_sq <= radius_sq {
                    let intensity = if min_dist_sq <= inner_radius_sq {
                        1.0
                    } else {
                        let dist = min_dist_sq.sqrt();
                        let t = ((dist - inner_radius) / feather_range).clamp(0.0, 1.0);
                        1.0 - (t * t * (3.0 - 2.0 * t))
                    };
                    *pixel = (intensity * 255.0).round() as u8;
                }
            }
        });

    GrayImage::from_raw(bb_w, bb_h, out_pixels).unwrap()
}

fn generate_radial_bitmap(
    params: &RadialMaskParameters,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
) -> GrayImage {
    let mut mask = GrayImage::new(width, height);

    let center_x = (params.center_x as f32 * scale - crop_offset.0) as i32;
    let center_y = (params.center_y as f32 * scale - crop_offset.1) as i32;
    let radius_x = params.radius_x as f32 * scale;
    let radius_y = params.radius_y as f32 * scale;
    let rotation_rad = params.rotation * PI / 180.0;

    for y in 0..height {
        for x in 0..width {
            let dx = (x + output_origin.0) as f32 - center_x as f32;
            let dy = (y + output_origin.1) as f32 - center_y as f32;

            let cos_rot = rotation_rad.cos();
            let sin_rot = rotation_rad.sin();

            let rot_dx = dx * cos_rot + dy * sin_rot;
            let rot_dy = -dx * sin_rot + dy * cos_rot;

            let norm_x = rot_dx / radius_x.max(0.01);
            let norm_y = rot_dy / radius_y.max(0.01);

            let dist = (norm_x.powi(2) + norm_y.powi(2)).sqrt();

            let inner_bound = 1.0 - params.feather.clamp(0.0, 1.0);
            let intensity = 1.0 - (dist - inner_bound) / (1.0 - inner_bound).max(0.01);
            let clamped_intensity = intensity.clamp(0.0, 1.0);

            mask.put_pixel(x, y, Luma([(clamped_intensity * 255.0) as u8]));
        }
    }

    mask
}

fn generate_linear_bitmap(
    params: &LinearMaskParameters,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
) -> GrayImage {
    let mut mask = GrayImage::new(width, height);

    let start_x = params.start_x as f32 * scale - crop_offset.0;
    let start_y = params.start_y as f32 * scale - crop_offset.1;
    let end_x = params.end_x as f32 * scale - crop_offset.0;
    let end_y = params.end_y as f32 * scale - crop_offset.1;
    let range = params.range * scale;

    let line_vec_x = end_x - start_x;
    let line_vec_y = end_y - start_y;

    let len_sq = line_vec_x.powi(2) + line_vec_y.powi(2);

    if len_sq < 0.01 {
        return mask;
    }

    let perp_vec_x = -line_vec_y / len_sq.sqrt();
    let perp_vec_y = line_vec_x / len_sq.sqrt();

    let half_width = range.max(0.01);

    for y_u in 0..height {
        for x_u in 0..width {
            let x = (x_u + output_origin.0) as f32;
            let y = (y_u + output_origin.1) as f32;

            let pixel_vec_x = x - start_x;
            let pixel_vec_y = y - start_y;

            let dist_perp = pixel_vec_x * perp_vec_x + pixel_vec_y * perp_vec_y;

            let t = dist_perp / half_width;

            let intensity = 0.5 - t * 0.5;

            let clamped_intensity = intensity.clamp(0.0, 1.0);

            mask.put_pixel(x_u, y_u, Luma([(clamped_intensity * 255.0) as u8]));
        }
    }

    mask
}

fn generate_brush_bitmap(
    params: &BrushMaskParameters,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
) -> GrayImage {
    let mut final_mask = GrayImage::new(width, height);

    for line in &params.lines {
        if line.points.is_empty() {
            continue;
        }

        let is_eraser = line.tool == "eraser";
        let radius = (line.brush_size * scale / 2.0).max(0.0);
        let feather = line.feather.clamp(0.0, 1.0);

        let Some((min_x, min_y, max_x, max_y)) = stroke_bounds(
            &line.points,
            width,
            height,
            radius,
            scale,
            crop_offset,
            output_origin,
        ) else {
            continue;
        };

        let bb_w = max_x - min_x + 1;
        let bb_h = max_y - min_y + 1;
        let layer_offset = (min_x as f32, min_y as f32);

        let line_mask = render_stroke_layer_parallel(
            &line.points,
            radius,
            feather,
            scale,
            crop_offset,
            output_origin,
            layer_offset,
            bb_w,
            bb_h,
        );

        for y in 0..bb_h {
            for x in 0..bb_w {
                let src_val = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
                if src_val <= 0.0 {
                    continue;
                }

                let abs_x = min_x + x;
                let abs_y = min_y + y;
                let dst_pixel = final_mask.get_pixel_mut(abs_x, abs_y);
                let dst_val = dst_pixel[0] as f32 / 255.0;

                let blended = if is_eraser {
                    dst_val * (1.0 - src_val)
                } else {
                    dst_val + src_val - dst_val * src_val
                };

                dst_pixel[0] = (blended.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }

    final_mask
}

fn generate_flow_bitmap(
    params: &FlowMaskParameters,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
) -> GrayImage {
    let mut final_mask = GrayImage::new(width, height);

    for line in &params.lines {
        if line.points.is_empty() {
            continue;
        }

        let is_eraser = line.tool == "eraser";
        let flow_per_stroke = (line.flow.clamp(0.0, 100.0) / 100.0) * 255.0;
        let radius = (line.brush_size * scale / 2.0).max(0.0);
        let feather = line.feather.clamp(0.0, 1.0);

        let Some((min_x, min_y, max_x, max_y)) = stroke_bounds(
            &line.points,
            width,
            height,
            radius,
            scale,
            crop_offset,
            output_origin,
        ) else {
            continue;
        };

        let bb_w = max_x - min_x + 1;
        let bb_h = max_y - min_y + 1;
        let layer_offset = (min_x as f32, min_y as f32);

        let line_mask = render_stroke_layer_parallel(
            &line.points,
            radius,
            feather,
            scale,
            crop_offset,
            output_origin,
            layer_offset,
            bb_w,
            bb_h,
        );

        for y in 0..bb_h {
            for x in 0..bb_w {
                let stroke_pixel = line_mask.get_pixel(x, y)[0] as f32;
                if stroke_pixel <= 0.0 {
                    continue;
                }

                let abs_x = min_x + x;
                let abs_y = min_y + y;
                let pixel = final_mask.get_pixel_mut(abs_x, abs_y);

                let c_norm = pixel[0] as f32 / 255.0;
                let delta = ((stroke_pixel / 255.0) * flow_per_stroke).round();
                let d_norm = (delta / 255.0).clamp(0.0, 1.0);

                let next = if is_eraser {
                    c_norm * (1.0 - d_norm)
                } else {
                    c_norm + d_norm - c_norm * d_norm
                };

                pixel[0] = (next.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }

    final_mask
}

pub struct TransformParams {
    pub rotation: f32,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub orientation_steps: u8,
    pub width: u32,
    pub height: u32,
    pub scale: f32,
    pub crop_offset: (f32, f32),
}

fn generate_ai_bitmap_from_full_mask(
    full_mask_image: &GrayImage,
    tf: &TransformParams,
    output_origin: (u32, u32),
) -> GrayImage {
    let (full_mask_w, full_mask_h) = full_mask_image.dimensions();
    let mut final_mask = GrayImage::new(tf.width, tf.height);

    let angle_rad = tf.rotation.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let (coarse_rotated_w, coarse_rotated_h) = if tf.orientation_steps % 2 == 1 {
        (full_mask_h, full_mask_w)
    } else {
        (full_mask_w, full_mask_h)
    };

    let scaled_coarse_rotated_w = coarse_rotated_w as f32 * tf.scale;
    let scaled_coarse_rotated_h = coarse_rotated_h as f32 * tf.scale;
    let center_x = scaled_coarse_rotated_w / 2.0;
    let center_y = scaled_coarse_rotated_h / 2.0;

    for y_out in 0..tf.height {
        for x_out in 0..tf.width {
            let x_uncrop = (x_out + output_origin.0) as f32 + tf.crop_offset.0;
            let y_uncrop = (y_out + output_origin.1) as f32 + tf.crop_offset.1;

            let x_centered = x_uncrop - center_x;
            let y_centered = y_uncrop - center_y;

            let x_unrotated = x_centered * cos_a + y_centered * sin_a + center_x;
            let y_unrotated = -x_centered * sin_a + y_centered * cos_a + center_y;

            let x_unflipped = if tf.flip_horizontal {
                scaled_coarse_rotated_w - x_unrotated
            } else {
                x_unrotated
            };
            let y_unflipped = if tf.flip_vertical {
                scaled_coarse_rotated_h - y_unrotated
            } else {
                y_unrotated
            };

            let (x_unrotated_coarse, y_unrotated_coarse) = match tf.orientation_steps {
                0 => (x_unflipped, y_unflipped),
                1 => (y_unflipped, scaled_coarse_rotated_w - x_unflipped),
                2 => (
                    scaled_coarse_rotated_w - x_unflipped,
                    scaled_coarse_rotated_h - y_unflipped,
                ),
                3 => (scaled_coarse_rotated_h - y_unflipped, x_unflipped),
                _ => (x_unflipped, y_unflipped),
            };

            let x_src = x_unrotated_coarse / tf.scale;
            let y_src = y_unrotated_coarse / tf.scale;

            if x_src >= 0.0
                && x_src < full_mask_w as f32
                && y_src >= 0.0
                && y_src < full_mask_h as f32
            {
                let pixel = full_mask_image.get_pixel(x_src as u32, y_src as u32);
                final_mask.put_pixel(x_out, y_out, *pixel);
            }
        }
    }

    final_mask
}

fn decode_ai_bitmap_from_base64(data_url: &str) -> Option<GrayImage> {
    let b64_data = if let Some(idx) = data_url.find(',') {
        &data_url[idx + 1..]
    } else {
        data_url
    };

    let decoded_bytes = general_purpose::STANDARD.decode(b64_data).ok()?;
    Some(image::load_from_memory(&decoded_bytes).ok()?.to_luma8())
}

pub fn generate_ai_bitmap_from_base64(data_url: &str, tf: &TransformParams) -> Option<GrayImage> {
    let full_mask_image = decode_ai_bitmap_from_base64(data_url)?;
    Some(generate_ai_bitmap_from_full_mask(
        &full_mask_image,
        tf,
        (0, 0),
    ))
}

#[derive(Debug, Clone, Copy)]
struct AiMaskTransform {
    rotation: f32,
    flip_horizontal: bool,
    flip_vertical: bool,
    orientation_steps: u8,
}

impl AiMaskTransform {
    fn from_options(
        rotation: Option<f32>,
        flip_horizontal: Option<bool>,
        flip_vertical: Option<bool>,
        orientation_steps: Option<u8>,
    ) -> Self {
        Self {
            rotation: rotation.unwrap_or(0.0),
            flip_horizontal: flip_horizontal.unwrap_or(false),
            flip_vertical: flip_vertical.unwrap_or(false),
            orientation_steps: orientation_steps.unwrap_or(0),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rasterize(
        self,
        full_mask_image: &GrayImage,
        width: u32,
        height: u32,
        scale: f32,
        crop_offset: (f32, f32),
        output_origin: (u32, u32),
    ) -> GrayImage {
        generate_ai_bitmap_from_full_mask(
            full_mask_image,
            &TransformParams {
                rotation: self.rotation,
                flip_horizontal: self.flip_horizontal,
                flip_vertical: self.flip_vertical,
                orientation_steps: self.orientation_steps,
                width,
                height,
                scale,
                crop_offset,
            },
            output_origin,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct AiDepthSelection {
    min_depth: f32,
    max_depth: f32,
    min_fade: f32,
    max_fade: f32,
    feather_sigma: f32,
}

impl AiDepthSelection {
    fn new(min_depth: f32, max_depth: f32, min_fade: f32, max_fade: f32, feather: f32) -> Self {
        Self {
            min_depth,
            max_depth,
            min_fade,
            max_fade,
            feather_sigma: if feather > 0.0 { feather * 0.1 } else { 0.0 },
        }
    }

    fn blur_radius(self) -> u32 {
        image_gaussian_blur_radius(self.feather_sigma)
    }

    fn apply_pointwise(self, mask: &mut GrayImage) {
        fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
            let t = ((x - edge0) / (edge1 - edge0).max(0.0001)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        }

        for pixel in mask.pixels_mut() {
            let value_percentage = (pixel[0] as f32 / 255.0) * 100.0;
            let lower_bound = smoothstep(
                self.min_depth - self.min_fade,
                self.min_depth,
                value_percentage,
            );
            let upper_bound = 1.0
                - smoothstep(
                    self.max_depth,
                    self.max_depth + self.max_fade,
                    value_percentage,
                );
            let bandpass_weight = lower_bound * upper_bound;
            let depth_intensity = value_percentage / 100.0;
            pixel[0] = (bandpass_weight * depth_intensity * 255.0) as u8;
        }
    }

    fn apply_blur(self, mask: &mut GrayImage) {
        if self.feather_sigma > 0.0 {
            *mask = image::imageops::blur(mask, self.feather_sigma);
        }
    }
}

fn image_gaussian_blur_radius(sigma: f32) -> u32 {
    if sigma <= 0.0 {
        return 0;
    }

    // Keep this in lockstep with image 0.25.x GaussianBlurParameters::kernel_size_from_sigma.
    let possible_size = (((((sigma - 0.8) / 0.3) + 1.0) * 2.0) + 1.0).max(3.0) as u32;
    let kernel_size = if possible_size.is_multiple_of(2) {
        possible_size + 1
    } else {
        possible_size
    };
    kernel_size / 2
}

#[derive(Debug, Clone, Copy)]
enum AiMaskPostprocess {
    Binary,
    Depth(AiDepthSelection),
}

struct TiledAiMaskRasterizer {
    full_mask_image: GrayImage,
    transform: AiMaskTransform,
    grow: f32,
    feather: f32,
    postprocess: AiMaskPostprocess,
}

impl TiledAiMaskRasterizer {
    fn new(sub_mask: &SubMask) -> Option<Self> {
        let parameters = &sub_mask.parameters;
        let number = |camel_case: &str, snake_case: &str| {
            parameters
                .get(camel_case)
                .or_else(|| parameters.get(snake_case))
                .and_then(Value::as_f64)
                .unwrap_or_default() as f32
        };
        let boolean = |camel_case: &str, snake_case: &str| {
            parameters
                .get(camel_case)
                .or_else(|| parameters.get(snake_case))
                .and_then(Value::as_bool)
        };
        let integer = |camel_case: &str, snake_case: &str| {
            parameters
                .get(camel_case)
                .or_else(|| parameters.get(snake_case))
                .and_then(Value::as_u64)
                .map(|value| value as u8)
        };
        let data_url = parameters
            .get("maskDataBase64")
            .or_else(|| parameters.get("mask_data_base64"))
            .and_then(Value::as_str)?;
        let grow = number("grow", "grow");
        let feather = number("feather", "feather");
        let transform = AiMaskTransform::from_options(
            Some(number("rotation", "rotation")),
            boolean("flipHorizontal", "flip_horizontal"),
            boolean("flipVertical", "flip_vertical"),
            integer("orientationSteps", "orientation_steps"),
        );

        let postprocess = match sub_mask.mask_type.as_str() {
            "ai-subject" | "quick-eraser" => {
                parameters
                    .get("startX")
                    .or_else(|| parameters.get("start_x"))
                    .and_then(Value::as_f64)?;
                parameters
                    .get("startY")
                    .or_else(|| parameters.get("start_y"))
                    .and_then(Value::as_f64)?;
                parameters
                    .get("endX")
                    .or_else(|| parameters.get("end_x"))
                    .and_then(Value::as_f64)?;
                parameters
                    .get("endY")
                    .or_else(|| parameters.get("end_y"))
                    .and_then(Value::as_f64)?;
                AiMaskPostprocess::Binary
            }
            "ai-foreground" | "ai-sky" => AiMaskPostprocess::Binary,
            "ai-depth" => AiMaskPostprocess::Depth(AiDepthSelection::new(
                number("minDepth", "min_depth"),
                number("maxDepth", "max_depth"),
                number("minFade", "min_fade"),
                number("maxFade", "max_fade"),
                feather,
            )),
            _ => return None,
        };

        Some(Self {
            full_mask_image: decode_ai_bitmap_from_base64(data_url)?,
            transform,
            grow,
            feather,
            postprocess,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn rasterize(
        &self,
        width: u32,
        height: u32,
        scale: f32,
        crop_offset: (f32, f32),
        output_origin: (u32, u32),
    ) -> GrayImage {
        let mut mask = self.transform.rasterize(
            &self.full_mask_image,
            width,
            height,
            scale,
            crop_offset,
            output_origin,
        );
        if let AiMaskPostprocess::Depth(selection) = self.postprocess {
            selection.apply_pointwise(&mut mask);
        }
        mask
    }

    fn halo(&self, width: u32, height: u32) -> u32 {
        let depth_radius = match self.postprocess {
            AiMaskPostprocess::Binary => 0,
            AiMaskPostprocess::Depth(selection) => selection.blur_radius(),
        };
        depth_radius
            .saturating_add(GrowFeatherPlan::new(self.grow, self.feather, width, height).halo())
    }

    fn apply_cross_pixel_filters(&self, mask: &mut GrayImage, width: u32, height: u32) {
        if let AiMaskPostprocess::Depth(selection) = self.postprocess {
            selection.apply_blur(mask);
        }
        GrowFeatherPlan::new(self.grow, self.feather, width, height).apply(mask);
    }
}

fn generate_ai_sky_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Option<GrayImage> {
    let params: AiSkyMaskParameters = serde_json::from_value(params_value.clone()).ok()?;
    let grow_feather: GrowFeatherParameters =
        serde_json::from_value(params_value.clone()).unwrap_or_default();
    let data_url = params.mask_data_base64?;

    let tf = TransformParams {
        rotation: params.rotation.unwrap_or(0.0),
        flip_horizontal: params.flip_horizontal.unwrap_or(false),
        flip_vertical: params.flip_vertical.unwrap_or(false),
        orientation_steps: params.orientation_steps.unwrap_or(0),
        width,
        height,
        scale,
        crop_offset,
    };
    let mut mask = generate_ai_bitmap_from_base64(&data_url, &tf)?;

    apply_grow_and_feather(
        &mut mask,
        grow_feather.grow,
        grow_feather.feather,
        width,
        height,
    );

    Some(mask)
}

fn generate_ai_depth_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Option<GrayImage> {
    let params: AiDepthMaskParameters = serde_json::from_value(params_value.clone()).ok()?;
    let grow_feather: GrowFeatherParameters =
        serde_json::from_value(params_value.clone()).unwrap_or_default();
    let data_url = params.mask_data_base64?;

    let tf = TransformParams {
        rotation: params.rotation.unwrap_or(0.0),
        flip_horizontal: params.flip_horizontal.unwrap_or(false),
        flip_vertical: params.flip_vertical.unwrap_or(false),
        orientation_steps: params.orientation_steps.unwrap_or(0),
        width,
        height,
        scale,
        crop_offset,
    };

    let mut mask = generate_ai_bitmap_from_base64(&data_url, &tf)?;
    let depth_selection = AiDepthSelection::new(
        params.min_depth,
        params.max_depth,
        params.min_fade,
        params.max_fade,
        params.feather,
    );
    depth_selection.apply_pointwise(&mut mask);
    depth_selection.apply_blur(&mut mask);

    apply_grow_and_feather(
        &mut mask,
        grow_feather.grow,
        grow_feather.feather,
        width,
        height,
    );

    Some(mask)
}

fn generate_ai_foreground_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Option<GrayImage> {
    let params: AiForegroundMaskParameters = serde_json::from_value(params_value.clone()).ok()?;
    let grow_feather: GrowFeatherParameters =
        serde_json::from_value(params_value.clone()).unwrap_or_default();
    let data_url = params.mask_data_base64?;

    let tf = TransformParams {
        rotation: params.rotation.unwrap_or(0.0),
        flip_horizontal: params.flip_horizontal.unwrap_or(false),
        flip_vertical: params.flip_vertical.unwrap_or(false),
        orientation_steps: params.orientation_steps.unwrap_or(0),
        width,
        height,
        scale,
        crop_offset,
    };
    let mut mask = generate_ai_bitmap_from_base64(&data_url, &tf)?;

    apply_grow_and_feather(
        &mut mask,
        grow_feather.grow,
        grow_feather.feather,
        width,
        height,
    );

    Some(mask)
}

fn generate_ai_subject_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Option<GrayImage> {
    let params: AiSubjectMaskParameters = serde_json::from_value(params_value.clone()).ok()?;
    let grow_feather: GrowFeatherParameters =
        serde_json::from_value(params_value.clone()).unwrap_or_default();
    let data_url = params.mask_data_base64?;

    let tf = TransformParams {
        rotation: params.rotation.unwrap_or(0.0),
        flip_horizontal: params.flip_horizontal.unwrap_or(false),
        flip_vertical: params.flip_vertical.unwrap_or(false),
        orientation_steps: params.orientation_steps.unwrap_or(0),
        width,
        height,
        scale,
        crop_offset,
    };
    let mut mask = generate_ai_bitmap_from_base64(&data_url, &tf)?;

    apply_grow_and_feather(
        &mut mask,
        grow_feather.grow,
        grow_feather.feather,
        width,
        height,
    );

    Some(mask)
}

fn generate_color_bitmap_from_parameters(
    params: &ParametricMaskParameters,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
    warped: &image::DynamicImage,
) -> Option<GrayImage> {
    let (full_w, full_h) = warped.dimensions();

    let target_x = params.target_x.round() as i32;
    let target_y = params.target_y.round() as i32;
    if target_x < 0 || target_y < 0 || target_x >= full_w as i32 || target_y >= full_h as i32 {
        return None;
    }

    let ref_pixel = warped.get_pixel(target_x as u32, target_y as u32);
    let ref_r = ref_pixel[0] as f32;
    let ref_g = ref_pixel[1] as f32;
    let ref_b = ref_pixel[2] as f32;

    let mut mask = GrayImage::new(width, height);

    let angle_rad = params.rotation * PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let (coarse_rotated_w, coarse_rotated_h) = if params.orientation_steps % 2 == 1 {
        (full_h, full_w)
    } else {
        (full_w, full_h)
    };

    let scaled_coarse_rotated_w = coarse_rotated_w as f32 * scale;
    let scaled_coarse_rotated_h = coarse_rotated_h as f32 * scale;
    let center_x = scaled_coarse_rotated_w / 2.0;
    let center_y = scaled_coarse_rotated_h / 2.0;

    let tolerance_sq = (params.tolerance * 2.55).max(1.0).powi(2) * 3.0;
    let inv_scale = 1.0 / scale;

    for y_out in 0..height {
        let y_uncrop = (y_out + output_origin.1) as f32 + crop_offset.1;
        let y_centered = y_uncrop - center_y;
        let y_sin = y_centered * sin_a;
        let y_cos = y_centered * cos_a;

        for x_out in 0..width {
            let x_uncrop = (x_out + output_origin.0) as f32 + crop_offset.0;
            let x_centered = x_uncrop - center_x;

            let x_unrotated = x_centered * cos_a + y_sin + center_x;
            let y_unrotated = -x_centered * sin_a + y_cos + center_y;

            let x_unflipped = if params.flip_horizontal {
                scaled_coarse_rotated_w - x_unrotated
            } else {
                x_unrotated
            };
            let y_unflipped = if params.flip_vertical {
                scaled_coarse_rotated_h - y_unrotated
            } else {
                y_unrotated
            };

            let (x_unrotated_coarse, y_unrotated_coarse) = match params.orientation_steps {
                0 => (x_unflipped, y_unflipped),
                1 => (y_unflipped, scaled_coarse_rotated_w - x_unflipped),
                2 => (
                    scaled_coarse_rotated_w - x_unflipped,
                    scaled_coarse_rotated_h - y_unflipped,
                ),
                3 => (scaled_coarse_rotated_h - y_unflipped, x_unflipped),
                _ => (x_unflipped, y_unflipped),
            };

            if x_unrotated_coarse >= 0.0 && y_unrotated_coarse >= 0.0 {
                let x_src = (x_unrotated_coarse * inv_scale) as u32;
                let y_src = (y_unrotated_coarse * inv_scale) as u32;

                if x_src < full_w && y_src < full_h {
                    let pixel = warped.get_pixel(x_src, y_src);
                    let dist_sq = (pixel[0] as f32 - ref_r).powi(2)
                        + (pixel[1] as f32 - ref_g).powi(2)
                        + (pixel[2] as f32 - ref_b).powi(2);

                    if dist_sq <= tolerance_sq {
                        let intensity = 1.0 - (dist_sq.sqrt() / tolerance_sq.sqrt());
                        mask.put_pixel(x_out, y_out, Luma([(intensity * 255.0) as u8]));
                    }
                }
            }
        }
    }

    Some(mask)
}

fn generate_color_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&image::DynamicImage>,
) -> Option<GrayImage> {
    let params: ParametricMaskParameters = serde_json::from_value(params_value.clone()).ok()?;
    let mut mask = generate_color_bitmap_from_parameters(
        &params,
        width,
        height,
        scale,
        crop_offset,
        (0, 0),
        warped_image?,
    )?;
    apply_grow_and_feather(&mut mask, params.grow, params.feather, width, height);
    Some(mask)
}

fn generate_luminance_bitmap_from_parameters(
    params: &ParametricMaskParameters,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
    warped: &image::DynamicImage,
) -> Option<GrayImage> {
    let (full_w, full_h) = warped.dimensions();

    let target_x = params.target_x.round() as i32;
    let target_y = params.target_y.round() as i32;
    if target_x < 0 || target_y < 0 || target_x >= full_w as i32 || target_y >= full_h as i32 {
        return None;
    }

    let ref_pixel = warped.get_pixel(target_x as u32, target_y as u32);
    let ref_luma =
        0.299 * ref_pixel[0] as f32 + 0.587 * ref_pixel[1] as f32 + 0.114 * ref_pixel[2] as f32;

    let mut mask = GrayImage::new(width, height);

    let angle_rad = params.rotation * PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let (coarse_rotated_w, coarse_rotated_h) = if params.orientation_steps % 2 == 1 {
        (full_h, full_w)
    } else {
        (full_w, full_h)
    };

    let scaled_coarse_rotated_w = coarse_rotated_w as f32 * scale;
    let scaled_coarse_rotated_h = coarse_rotated_h as f32 * scale;
    let center_x = scaled_coarse_rotated_w / 2.0;
    let center_y = scaled_coarse_rotated_h / 2.0;

    let tolerance_val = (params.tolerance * 2.55).max(1.0);
    let inv_scale = 1.0 / scale;

    for y_out in 0..height {
        let y_uncrop = (y_out + output_origin.1) as f32 + crop_offset.1;
        let y_centered = y_uncrop - center_y;
        let y_sin = y_centered * sin_a;
        let y_cos = y_centered * cos_a;

        for x_out in 0..width {
            let x_uncrop = (x_out + output_origin.0) as f32 + crop_offset.0;
            let x_centered = x_uncrop - center_x;

            let x_unrotated = x_centered * cos_a + y_sin + center_x;
            let y_unrotated = -x_centered * sin_a + y_cos + center_y;

            let x_unflipped = if params.flip_horizontal {
                scaled_coarse_rotated_w - x_unrotated
            } else {
                x_unrotated
            };
            let y_unflipped = if params.flip_vertical {
                scaled_coarse_rotated_h - y_unrotated
            } else {
                y_unrotated
            };

            let (x_unrotated_coarse, y_unrotated_coarse) = match params.orientation_steps {
                0 => (x_unflipped, y_unflipped),
                1 => (y_unflipped, scaled_coarse_rotated_w - x_unflipped),
                2 => (
                    scaled_coarse_rotated_w - x_unflipped,
                    scaled_coarse_rotated_h - y_unflipped,
                ),
                3 => (scaled_coarse_rotated_h - y_unflipped, x_unflipped),
                _ => (x_unflipped, y_unflipped),
            };

            if x_unrotated_coarse >= 0.0 && y_unrotated_coarse >= 0.0 {
                let x_src = (x_unrotated_coarse * inv_scale) as u32;
                let y_src = (y_unrotated_coarse * inv_scale) as u32;

                if x_src < full_w && y_src < full_h {
                    let pixel = warped.get_pixel(x_src, y_src);
                    let luma =
                        0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
                    let dist = (luma - ref_luma).abs();

                    if dist <= tolerance_val {
                        let intensity = 1.0 - (dist / tolerance_val);
                        mask.put_pixel(x_out, y_out, Luma([(intensity * 255.0) as u8]));
                    }
                }
            }
        }
    }

    Some(mask)
}

fn generate_luminance_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&image::DynamicImage>,
) -> Option<GrayImage> {
    let params: ParametricMaskParameters = serde_json::from_value(params_value.clone()).ok()?;
    let mut mask = generate_luminance_bitmap_from_parameters(
        &params,
        width,
        height,
        scale,
        crop_offset,
        (0, 0),
        warped_image?,
    )?;
    apply_grow_and_feather(&mut mask, params.grow, params.feather, width, height);
    Some(mask)
}

fn generate_all_bitmap(width: u32, height: u32) -> GrayImage {
    GrayImage::from_pixel(width, height, Luma([255]))
}

fn generate_sub_mask_bitmap(
    sub_mask: &SubMask,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&DynamicImage>,
) -> Option<GrayImage> {
    if !sub_mask.visible {
        return None;
    }

    match sub_mask.mask_type.as_str() {
        "radial" => {
            let parameters =
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default();
            Some(generate_radial_bitmap(
                &parameters,
                width,
                height,
                scale,
                crop_offset,
                (0, 0),
            ))
        }
        "linear" => {
            let parameters =
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default();
            Some(generate_linear_bitmap(
                &parameters,
                width,
                height,
                scale,
                crop_offset,
                (0, 0),
            ))
        }
        "brush" | "clone" | "heal" => {
            let parameters =
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default();
            Some(generate_brush_bitmap(
                &parameters,
                width,
                height,
                scale,
                crop_offset,
                (0, 0),
            ))
        }
        "flow" => {
            let parameters =
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default();
            Some(generate_flow_bitmap(
                &parameters,
                width,
                height,
                scale,
                crop_offset,
                (0, 0),
            ))
        }
        "color" => generate_color_bitmap(
            &sub_mask.parameters,
            width,
            height,
            scale,
            crop_offset,
            warped_image,
        ),
        "luminance" => generate_luminance_bitmap(
            &sub_mask.parameters,
            width,
            height,
            scale,
            crop_offset,
            warped_image,
        ),
        "ai-subject" => {
            generate_ai_subject_bitmap(&sub_mask.parameters, width, height, scale, crop_offset)
        }
        "ai-foreground" => {
            generate_ai_foreground_bitmap(&sub_mask.parameters, width, height, scale, crop_offset)
        }
        "ai-sky" => generate_ai_sky_bitmap(&sub_mask.parameters, width, height, scale, crop_offset),
        "ai-depth" => {
            generate_ai_depth_bitmap(&sub_mask.parameters, width, height, scale, crop_offset)
        }
        "quick-eraser" => {
            generate_ai_subject_bitmap(&sub_mask.parameters, width, height, scale, crop_offset)
        }
        "all" => Some(generate_all_bitmap(width, height)),
        _ => None,
    }
}

enum TiledSubMaskRasterizer {
    Radial(RadialMaskParameters),
    Linear(LinearMaskParameters),
    Brush(BrushMaskParameters),
    Flow(FlowMaskParameters),
    Color(ParametricMaskParameters),
    Luminance(ParametricMaskParameters),
    Ai(TiledAiMaskRasterizer),
    All,
}

impl TiledSubMaskRasterizer {
    fn new(sub_mask: &SubMask) -> Option<Self> {
        match sub_mask.mask_type.as_str() {
            "radial" => Some(Self::Radial(
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default(),
            )),
            "linear" => Some(Self::Linear(
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default(),
            )),
            "brush" | "clone" | "heal" => Some(Self::Brush(
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default(),
            )),
            "flow" => Some(Self::Flow(
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default(),
            )),
            "color" => Some(Self::Color(
                serde_json::from_value(sub_mask.parameters.clone()).ok()?,
            )),
            "luminance" => Some(Self::Luminance(
                serde_json::from_value(sub_mask.parameters.clone()).ok()?,
            )),
            "ai-subject" | "ai-foreground" | "ai-sky" | "ai-depth" | "quick-eraser" => {
                Some(Self::Ai(TiledAiMaskRasterizer::new(sub_mask)?))
            }
            "all" => Some(Self::All),
            _ => None,
        }
    }

    fn rasterize(
        &self,
        width: u32,
        height: u32,
        scale: f32,
        crop_offset: (f32, f32),
        output_origin: (u32, u32),
        warped_image: Option<&DynamicImage>,
    ) -> Option<GrayImage> {
        match self {
            Self::Radial(parameters) => Some(generate_radial_bitmap(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
            )),
            Self::Linear(parameters) => Some(generate_linear_bitmap(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
            )),
            Self::Brush(parameters) => Some(generate_brush_bitmap(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
            )),
            Self::Flow(parameters) => Some(generate_flow_bitmap(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
            )),
            Self::Color(parameters) => generate_color_bitmap_from_parameters(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
                warped_image?,
            ),
            Self::Luminance(parameters) => generate_luminance_bitmap_from_parameters(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
                warped_image?,
            ),
            Self::Ai(rasterizer) => {
                Some(rasterizer.rasterize(width, height, scale, crop_offset, output_origin))
            }
            Self::All => Some(generate_all_bitmap(width, height)),
        }
    }

    fn halo(&self, width: u32, height: u32) -> u32 {
        match self {
            Self::Color(parameters) | Self::Luminance(parameters) => {
                GrowFeatherPlan::new(parameters.grow, parameters.feather, width, height).halo()
            }
            Self::Ai(rasterizer) => rasterizer.halo(width, height),
            _ => 0,
        }
    }

    fn apply_cross_pixel_filters(&self, mask: &mut GrayImage, width: u32, height: u32) {
        match self {
            Self::Color(parameters) | Self::Luminance(parameters) => {
                GrowFeatherPlan::new(parameters.grow, parameters.feather, width, height)
                    .apply(mask);
            }
            Self::Ai(rasterizer) => {
                rasterizer.apply_cross_pixel_filters(mask, width, height);
            }
            _ => {}
        }
    }

    fn is_ready(&self, warped_image: Option<&DynamicImage>) -> bool {
        match self {
            Self::Color(parameters) | Self::Luminance(parameters) => {
                warped_image.is_some_and(|warped| {
                    let (width, height) = warped.dimensions();
                    let target_x = parameters.target_x.round() as i32;
                    let target_y = parameters.target_y.round() as i32;
                    target_x >= 0
                        && target_y >= 0
                        && target_x < width as i32
                        && target_y < height as i32
                })
            }
            Self::Ai(_) => true,
            _ => true,
        }
    }
}

pub fn generate_mask_bitmap(
    mask_def: &MaskDefinition,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&DynamicImage>,
) -> Option<GrayImage> {
    if !mask_def.visible || mask_def.sub_masks.is_empty() {
        return None;
    }

    let mut final_mask: Option<GrayImage> = None;

    for sub_mask in &mask_def.sub_masks {
        // The historical composition starts from black. Subtractive and intersect masks cannot
        // change that state, so defer the first allocation/generation until an additive mask can
        // become the output buffer itself.
        if final_mask.is_none() && sub_mask.mode != SubMaskMode::Additive {
            continue;
        }
        if should_generate_sub_mask_in_tiles(sub_mask, final_mask.is_some(), width, height)
            && composite_sub_mask_tiled(
                &mut final_mask,
                sub_mask,
                width,
                height,
                scale,
                crop_offset,
                warped_image,
            )
        {
            continue;
        }
        if let Some(mut sub_bitmap) =
            generate_sub_mask_bitmap(sub_mask, width, height, scale, crop_offset, warped_image)
        {
            apply_sub_mask_modifiers(&mut sub_bitmap, sub_mask);
            composite_sub_mask(&mut final_mask, sub_bitmap, sub_mask.mode);
        }
    }

    let mut final_mask = final_mask.unwrap_or_else(|| GrayImage::new(width, height));

    if mask_def.invert {
        for pixel in final_mask.pixels_mut() {
            pixel[0] = 255 - pixel[0];
        }
    }

    let opacity_multiplier = (mask_def.opacity / 100.0).clamp(0.0, 1.0);
    if opacity_multiplier < 1.0 {
        for pixel in final_mask.pixels_mut() {
            pixel[0] = (pixel[0] as f32 * opacity_multiplier) as u8;
        }
    }

    Some(final_mask)
}

fn should_generate_sub_mask_in_tiles(
    sub_mask: &SubMask,
    has_composition: bool,
    width: u32,
    height: u32,
) -> bool {
    if !sub_mask.visible {
        return false;
    }
    match sub_mask.mask_type.as_str() {
        // Brush and flow generators otherwise hold their full-frame result plus a stroke layer.
        "brush" | "clone" | "heal" => {
            serde_json::from_value::<BrushMaskParameters>(sub_mask.parameters.clone())
                .is_ok_and(|parameters| parameters.lines.iter().any(|line| !line.points.is_empty()))
        }
        "flow" => serde_json::from_value::<FlowMaskParameters>(sub_mask.parameters.clone())
            .is_ok_and(|parameters| parameters.lines.iter().any(|line| !line.points.is_empty())),
        // The first pointwise additive mask can become the output allocation directly. Later
        // masks use bounded tiles so composition never needs a second full-frame bitmap.
        "radial" | "linear" | "all" => has_composition,
        // A first unfiltered range mask can likewise become the output directly. Cross-pixel
        // grow/feather and every later range mask use an exact finite halo around each tile.
        "color" | "luminance" => {
            serde_json::from_value::<ParametricMaskParameters>(sub_mask.parameters.clone())
                .is_ok_and(|parameters| {
                    has_composition
                        || GrowFeatherPlan::new(parameters.grow, parameters.feather, width, height)
                            .has_effect()
                })
        }
        // AI masks keep their decoded source bitmap, but bounded output/filter tiles avoid a
        // transformed full-frame temporary whenever composition or a cross-pixel filter needs it.
        "ai-subject" | "ai-foreground" | "ai-sky" | "quick-eraser" | "ai-depth" => {
            let grow = sub_mask
                .parameters
                .get("grow")
                .and_then(Value::as_f64)
                .unwrap_or_default() as f32;
            let feather = sub_mask
                .parameters
                .get("feather")
                .and_then(Value::as_f64)
                .unwrap_or_default() as f32;
            let grow_feather_effect =
                GrowFeatherPlan::new(grow, feather, width, height).has_effect();
            let depth_blur_effect = sub_mask.mask_type == "ai-depth" && feather > 0.0;
            has_composition || grow_feather_effect || depth_blur_effect
        }
        _ => false,
    }
}

fn apply_sub_mask_modifiers(bitmap: &mut GrayImage, sub_mask: &SubMask) {
    apply_sub_mask_modifiers_to_pixels(bitmap.as_mut(), sub_mask);
}

fn apply_sub_mask_modifiers_to_pixels(pixels: &mut [u8], sub_mask: &SubMask) {
    if sub_mask.invert {
        for pixel in pixels.iter_mut() {
            *pixel = 255 - *pixel;
        }
    }

    let opacity_multiplier = (sub_mask.opacity / 100.0).clamp(0.0, 1.0);
    if opacity_multiplier < 1.0 {
        for pixel in pixels {
            *pixel = (*pixel as f32 * opacity_multiplier) as u8;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn composite_sub_mask_tiled(
    final_mask: &mut Option<GrayImage>,
    sub_mask: &SubMask,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&DynamicImage>,
) -> bool {
    composite_sub_mask_tiled_with_edge(
        final_mask,
        sub_mask,
        width,
        height,
        scale,
        crop_offset,
        warped_image,
        GPU_TILE_SIZE,
    )
}

#[allow(clippy::too_many_arguments)]
fn composite_sub_mask_tiled_with_edge(
    final_mask: &mut Option<GrayImage>,
    sub_mask: &SubMask,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    warped_image: Option<&DynamicImage>,
    tile_edge: u32,
) -> bool {
    debug_assert!(tile_edge > 0);
    debug_assert!(final_mask.is_some() || sub_mask.mode == SubMaskMode::Additive);
    let Some(rasterizer) = TiledSubMaskRasterizer::new(sub_mask) else {
        return false;
    };
    if !rasterizer.is_ready(warped_image) {
        return false;
    }
    let halo = rasterizer.halo(width, height);
    let output = final_mask.get_or_insert_with(|| GrayImage::new(width, height));

    for tile_y in (0..height).step_by(tile_edge as usize) {
        let tile_height = (height - tile_y).min(tile_edge);
        for tile_x in (0..width).step_by(tile_edge as usize) {
            let tile_width = (width - tile_x).min(tile_edge);
            let expanded_x = tile_x.saturating_sub(halo);
            let expanded_y = tile_y.saturating_sub(halo);
            let expanded_right = tile_x
                .saturating_add(tile_width)
                .saturating_add(halo)
                .min(width);
            let expanded_bottom = tile_y
                .saturating_add(tile_height)
                .saturating_add(halo)
                .min(height);
            let expanded_width = expanded_right - expanded_x;
            let expanded_height = expanded_bottom - expanded_y;
            let mut tile = rasterizer
                .rasterize(
                    expanded_width,
                    expanded_height,
                    scale,
                    crop_offset,
                    (expanded_x, expanded_y),
                    warped_image,
                )
                .expect("validated tiled mask rasterizer");
            rasterizer.apply_cross_pixel_filters(&mut tile, width, height);
            composite_sub_mask_tile_region(
                output,
                &mut tile,
                tile_x - expanded_x,
                tile_y - expanded_y,
                tile_width,
                tile_height,
                tile_x,
                tile_y,
                sub_mask,
            );
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn composite_sub_mask_tile_region(
    final_mask: &mut GrayImage,
    tile: &mut GrayImage,
    source_x: u32,
    source_y: u32,
    width: u32,
    height: u32,
    tile_x: u32,
    tile_y: u32,
    sub_mask: &SubMask,
) {
    let (output_width, output_height) = final_mask.dimensions();
    let (source_width, source_height) = tile.dimensions();
    debug_assert!(tile_x.saturating_add(width) <= output_width);
    debug_assert!(tile_y.saturating_add(height) <= output_height);
    debug_assert!(source_x.saturating_add(width) <= source_width);
    debug_assert!(source_y.saturating_add(height) <= source_height);

    let output_width = output_width as usize;
    let source_width = source_width as usize;
    let width = width as usize;
    for row in 0..height as usize {
        let output_start = (tile_y as usize + row) * output_width + tile_x as usize;
        let source_start = (source_y as usize + row) * source_width + source_x as usize;
        let source_pixels = &mut tile.as_mut()[source_start..source_start + width];
        apply_sub_mask_modifiers_to_pixels(source_pixels, sub_mask);
        composite_mask_pixels(
            &mut final_mask.as_mut()[output_start..output_start + width],
            source_pixels,
            sub_mask.mode,
        );
    }
}

#[cfg(test)]
fn composite_sub_mask_tile(
    final_mask: &mut GrayImage,
    tile: &GrayImage,
    tile_x: u32,
    tile_y: u32,
    mode: SubMaskMode,
) {
    let (output_width, output_height) = final_mask.dimensions();
    let (tile_width, tile_height) = tile.dimensions();
    debug_assert!(tile_x.saturating_add(tile_width) <= output_width);
    debug_assert!(tile_y.saturating_add(tile_height) <= output_height);

    let output_width = output_width as usize;
    let tile_width = tile_width as usize;
    for row in 0..tile_height as usize {
        let output_start = (tile_y as usize + row) * output_width + tile_x as usize;
        let tile_start = row * tile_width;
        composite_mask_pixels(
            &mut final_mask.as_mut()[output_start..output_start + tile_width],
            &tile.as_raw()[tile_start..tile_start + tile_width],
            mode,
        );
    }
}

fn composite_sub_mask(
    final_mask: &mut Option<GrayImage>,
    sub_bitmap: GrayImage,
    mode: SubMaskMode,
) {
    if final_mask.is_none() {
        if mode == SubMaskMode::Additive {
            *final_mask = Some(sub_bitmap);
        }
        return;
    }

    debug_assert_eq!(
        final_mask
            .as_ref()
            .expect("mask allocation checked above")
            .dimensions(),
        sub_bitmap.dimensions()
    );
    let final_raw = final_mask
        .as_mut()
        .expect("mask allocation checked above")
        .as_mut();
    composite_mask_pixels(final_raw, sub_bitmap.as_raw(), mode);
}

fn composite_mask_pixels(output: &mut [u8], input: &[u8], mode: SubMaskMode) {
    debug_assert_eq!(output.len(), input.len());
    match mode {
        SubMaskMode::Additive => output
            .iter_mut()
            .zip(input)
            .for_each(|(output, input)| *output = (*output).max(*input)),
        SubMaskMode::Subtractive => output
            .iter_mut()
            .zip(input)
            .for_each(|(output, input)| *output = output.saturating_sub(*input)),
        SubMaskMode::Intersect => output
            .iter_mut()
            .zip(input)
            .for_each(|(output, input)| *output = (*output).min(*input)),
    }
}

#[tauri::command]
pub fn generate_mask_overlay(
    mut mask_def: serde_json::Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    mut js_adjustments: Option<serde_json::Value>,
    state: tauri::State<'_, AppState>,
) -> Result<Response, String> {
    if let Some(ref mut adj) = js_adjustments {
        crate::adjustment_utils::hydrate_adjustments(&state, adj);
    }

    if let Some(sub_masks) = mask_def.get_mut("subMasks").and_then(|v| v.as_array_mut()) {
        let mut cache = state.patch_cache.lock().unwrap();
        crate::adjustment_utils::hydrate_sub_masks(sub_masks, &mut cache);
    }

    let parsed_mask_def: MaskDefinition = serde_json::from_value(mask_def)
        .map_err(|e| format!("Failed to parse hydrated mask_def: {}", e))?;

    let scaled_crop_offset = (crop_offset.0 * scale, crop_offset.1 * scale);

    let warped_image = js_adjustments.as_ref().and_then(|adj| {
        resolve_warped_image_for_masks(&state, adj, std::slice::from_ref(&parsed_mask_def))
    });

    if let Some(gray_mask) = generate_mask_bitmap(
        &parsed_mask_def,
        width,
        height,
        scale,
        scaled_crop_offset,
        warped_image.as_deref(),
    ) {
        let mut rgba_mask = RgbaImage::new(width, height);
        for (x, y, pixel) in gray_mask.enumerate_pixels() {
            let intensity = pixel[0];
            let alpha = (intensity as f32 * 0.5) as u8;
            rgba_mask.put_pixel(x, y, Rgba([255, 0, 0, alpha]));
        }

        let mut buf = Cursor::new(Vec::new());
        rgba_mask
            .write_to(&mut buf, ImageFormat::Png)
            .map_err(|e| e.to_string())?;

        Ok(Response::new(buf.into_inner()))
    } else {
        Ok(Response::new(Vec::new()))
    }
}

pub fn resolve_warped_image_for_masks(
    state: &tauri::State<AppState>,
    adjustments: &serde_json::Value,
    masks: &[MaskDefinition],
) -> Option<Arc<DynamicImage>> {
    if masks.iter().any(|m| m.requires_warped_image()) {
        get_cached_full_warped_image(state, adjustments).ok()
    } else {
        None
    }
}

pub fn get_cached_or_generate_mask(
    state: &tauri::State<AppState>,
    def: &MaskDefinition,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    adjustments: &serde_json::Value,
) -> Option<SharedMaskBitmap> {
    let mut hasher = DefaultHasher::new();

    let mut def_for_hash = def.clone();
    def_for_hash.adjustments = serde_json::Value::Null;
    let def_json = serde_json::to_string(&def_for_hash).unwrap_or_default();
    def_json.hash(&mut hasher);

    width.hash(&mut hasher);
    height.hash(&mut hasher);
    scale.to_bits().hash(&mut hasher);
    crop_offset.0.to_bits().hash(&mut hasher);
    crop_offset.1.to_bits().hash(&mut hasher);

    let key = hasher.finish();

    {
        let mut cache = state.mask_cache.lock().unwrap();
        if let Some(img) = cache.get(&key) {
            return Some(Arc::clone(img));
        }
    }

    let warped_image =
        resolve_warped_image_for_masks(state, adjustments, std::slice::from_ref(def));

    let generated = generate_mask_bitmap(
        def,
        width,
        height,
        scale,
        crop_offset,
        warped_image.as_deref(),
    )
    .map(Arc::new);

    if let Some(img) = &generated {
        let mut cache = state.mask_cache.lock().unwrap();
        cache.insert(key, Arc::clone(img), img.as_raw().len());
    }

    generated
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::cache_utils::BudgetedCache;

    #[test]
    fn first_additive_submask_reuses_its_pixel_allocation() {
        let additive = GrayImage::from_fn(17, 11, |x, y| Luma([((x * 13 + y * 7) % 256) as u8]));
        let additive_ptr = additive.as_raw().as_ptr();
        let expected = additive.clone();
        let mut composed = None;

        composite_sub_mask(&mut composed, additive, SubMaskMode::Additive);

        let composed = composed.expect("additive bitmap becomes the composition base");
        assert_eq!(composed.as_raw().as_ptr(), additive_ptr);
        assert_eq!(composed, expected);
    }

    #[test]
    fn lazy_mask_composition_preserves_black_base_semantics() {
        let mut composed = None;
        composite_sub_mask(
            &mut composed,
            GrayImage::from_pixel(5, 3, Luma([200])),
            SubMaskMode::Subtractive,
        );
        composite_sub_mask(
            &mut composed,
            GrayImage::from_pixel(5, 3, Luma([100])),
            SubMaskMode::Intersect,
        );
        assert!(composed.is_none());

        composite_sub_mask(
            &mut composed,
            GrayImage::from_pixel(5, 3, Luma([180])),
            SubMaskMode::Additive,
        );
        composite_sub_mask(
            &mut composed,
            GrayImage::from_pixel(5, 3, Luma([30])),
            SubMaskMode::Subtractive,
        );
        composite_sub_mask(
            &mut composed,
            GrayImage::from_pixel(5, 3, Luma([120])),
            SubMaskMode::Intersect,
        );

        assert!(
            composed
                .expect("additive submask establishes the output")
                .pixels()
                .all(|pixel| pixel[0] == 120)
        );
    }

    #[test]
    fn shared_mask_cache_keeps_one_60mp_pixel_allocation() {
        const WIDTH: usize = 9_504;
        const HEIGHT: usize = 6_336;
        const MASK_BYTES: usize = 60_217_344;

        assert_eq!(WIDTH * HEIGHT, MASK_BYTES);
        let mask = Arc::new(GrayImage::from_pixel(7, 5, Luma([123])));
        let mut cache = BudgetedCache::new(2, 1_024);
        assert!(cache.insert(42_u64, Arc::clone(&mask), mask.as_raw().len()));
        let caller = Arc::clone(cache.get(&42).expect("shared mask cache hit"));

        assert!(Arc::ptr_eq(&mask, &caller));
        assert_eq!(mask.as_raw().as_ptr(), caller.as_raw().as_ptr());
        assert_eq!(MASK_BYTES * 2 - MASK_BYTES, MASK_BYTES);
    }

    fn test_sub_mask(mask_type: &str, parameters: Value, mode: SubMaskMode) -> SubMask {
        SubMask {
            id: format!("test-{mask_type}"),
            mask_type: mask_type.to_string(),
            visible: true,
            invert: true,
            opacity: 63.0,
            mode,
            parameters,
        }
    }

    fn grayscale_png_data_url(image: &GrayImage) -> String {
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, ImageFormat::Png)
            .expect("encode deterministic AI mask fixture");
        format!(
            "data:image/png;base64,{}",
            general_purpose::STANDARD.encode(encoded.get_ref())
        )
    }

    fn brush_parameters(flow: bool) -> Value {
        let first_line = if flow {
            serde_json::json!({
                "tool": "brush",
                "brushSize": 29.0,
                "feather": 0.37,
                "flow": 42.0,
                "points": [{ "x": 8.0, "y": 12.0 }, { "x": 128.0, "y": 91.0 }]
            })
        } else {
            serde_json::json!({
                "tool": "brush",
                "brushSize": 29.0,
                "feather": 0.37,
                "points": [{ "x": 8.0, "y": 12.0 }, { "x": 128.0, "y": 91.0 }]
            })
        };
        let eraser_line = if flow {
            serde_json::json!({
                "tool": "eraser",
                "brushSize": 17.0,
                "feather": 0.61,
                "flow": 73.0,
                "points": [{ "x": 76.0, "y": 5.0 }, { "x": 65.0, "y": 104.0 }]
            })
        } else {
            serde_json::json!({
                "tool": "eraser",
                "brushSize": 17.0,
                "feather": 0.61,
                "points": [{ "x": 76.0, "y": 5.0 }, { "x": 65.0, "y": 104.0 }]
            })
        };
        serde_json::json!({ "lines": [first_line, eraser_line] })
    }

    #[test]
    fn tiled_programmatic_and_brush_composition_matches_full_frame_across_seams() {
        const WIDTH: u32 = 141;
        const HEIGHT: u32 = 109;
        const TEST_TILE_EDGE: u32 = 64;
        const SCALE: f32 = 1.25;
        const CROP_OFFSET: (f32, f32) = (11.5, 7.25);

        let cases = [
            (
                "radial",
                serde_json::json!({
                    "centerX": 61.0,
                    "centerY": 47.0,
                    "radiusX": 54.0,
                    "radiusY": 31.0,
                    "rotation": 23.0,
                    "feather": 0.41
                }),
            ),
            (
                "linear",
                serde_json::json!({
                    "startX": 17.0,
                    "startY": 19.0,
                    "endX": 119.0,
                    "endY": 88.0,
                    "range": 38.0
                }),
            ),
            ("brush", brush_parameters(false)),
            ("clone", brush_parameters(false)),
            ("heal", brush_parameters(false)),
            ("flow", brush_parameters(true)),
            ("all", serde_json::json!({})),
        ];
        let modes = [
            SubMaskMode::Additive,
            SubMaskMode::Subtractive,
            SubMaskMode::Intersect,
        ];

        for (mask_type, parameters) in cases {
            for mode in modes {
                let sub_mask = test_sub_mask(mask_type, parameters.clone(), mode);
                let base = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
                    Luma([x.wrapping_mul(11).wrapping_add(y * 7) as u8])
                });

                let mut expected = (mode != SubMaskMode::Additive).then(|| base.clone());
                let mut full_sub_mask =
                    generate_sub_mask_bitmap(&sub_mask, WIDTH, HEIGHT, SCALE, CROP_OFFSET, None)
                        .expect("test mask type should rasterize");
                apply_sub_mask_modifiers(&mut full_sub_mask, &sub_mask);
                composite_sub_mask(&mut expected, full_sub_mask, mode);

                let mut actual = (mode != SubMaskMode::Additive).then_some(base);
                composite_sub_mask_tiled_with_edge(
                    &mut actual,
                    &sub_mask,
                    WIDTH,
                    HEIGHT,
                    SCALE,
                    CROP_OFFSET,
                    None,
                    TEST_TILE_EDGE,
                );

                let actual = actual.expect("tiled composition output");
                let expected = expected.expect("full-frame composition output");
                let first_difference = actual
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .position(|(actual, expected)| actual != expected)
                    .map(|index| {
                        (
                            index as u32 % WIDTH,
                            index as u32 / WIDTH,
                            actual.as_raw()[index],
                            expected.as_raw()[index],
                        )
                    });
                assert_eq!(
                    first_difference, None,
                    "{mask_type} {mode:?} changed at a tile boundary"
                );
            }
        }
    }

    #[test]
    fn first_brush_and_flow_submasks_match_full_frame_when_generated_in_tiles() {
        const WIDTH: u32 = 141;
        const HEIGHT: u32 = 109;

        for (mask_type, parameters) in [
            ("brush", brush_parameters(false)),
            ("flow", brush_parameters(true)),
        ] {
            let sub_mask = test_sub_mask(mask_type, parameters, SubMaskMode::Additive);
            let mut expected_sub_mask =
                generate_sub_mask_bitmap(&sub_mask, WIDTH, HEIGHT, 1.0, (0.0, 0.0), None)
                    .expect("brush-like test mask should rasterize");
            apply_sub_mask_modifiers(&mut expected_sub_mask, &sub_mask);

            let mut actual = None;
            composite_sub_mask_tiled_with_edge(
                &mut actual,
                &sub_mask,
                WIDTH,
                HEIGHT,
                1.0,
                (0.0, 0.0),
                None,
                64,
            );

            assert_eq!(actual, Some(expected_sub_mask), "{mask_type}");
        }
    }

    #[test]
    fn tiled_color_and_luminance_masks_match_full_frame_with_exact_halo() {
        const WIDTH: u32 = 141;
        const HEIGHT: u32 = 109;
        const TEST_TILE_EDGE: u32 = 37;
        const SCALE: f32 = 1.25;
        const CROP_OFFSET: (f32, f32) = (11.5, 7.25);

        let warped = DynamicImage::ImageRgba8(RgbaImage::from_fn(181, 137, |x, y| {
            Rgba([
                x.wrapping_mul(17).wrapping_add(y * 3) as u8,
                x.wrapping_mul(5).wrapping_add(y * 19) as u8,
                x.wrapping_mul(11).wrapping_add(y * 7) as u8,
                255,
            ])
        }));
        let cases = [
            (
                "color",
                serde_json::json!({
                    "targetX": 72.0,
                    "targetY": 58.0,
                    "tolerance": 72.0,
                    "grow": 100.0,
                    "feather": 100.0,
                    "rotation": 17.0,
                    "flipHorizontal": true,
                    "flipVertical": false,
                    "orientationSteps": 1
                }),
            ),
            (
                "luminance",
                serde_json::json!({
                    "targetX": 113.0,
                    "targetY": 41.0,
                    "tolerance": 32.0,
                    "grow": -100.0,
                    "feather": 100.0,
                    "rotation": -13.0,
                    "flipHorizontal": false,
                    "flipVertical": true,
                    "orientationSteps": 2
                }),
            ),
        ];

        for (mask_type, parameters) in cases {
            for mode in [
                SubMaskMode::Additive,
                SubMaskMode::Subtractive,
                SubMaskMode::Intersect,
            ] {
                let sub_mask = test_sub_mask(mask_type, parameters.clone(), mode);
                let base = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
                    Luma([x.wrapping_mul(11).wrapping_add(y * 7) as u8])
                });

                let mut expected = (mode != SubMaskMode::Additive).then(|| base.clone());
                let mut full_sub_mask = generate_sub_mask_bitmap(
                    &sub_mask,
                    WIDTH,
                    HEIGHT,
                    SCALE,
                    CROP_OFFSET,
                    Some(&warped),
                )
                .expect("range mask should rasterize");
                let first_pixel = full_sub_mask.get_pixel(0, 0)[0];
                assert!(
                    full_sub_mask.pixels().any(|pixel| pixel[0] != first_pixel),
                    "{mask_type} fixture must exercise non-uniform pixels"
                );
                apply_sub_mask_modifiers(&mut full_sub_mask, &sub_mask);
                composite_sub_mask(&mut expected, full_sub_mask, mode);

                let mut actual = (mode != SubMaskMode::Additive).then_some(base);
                assert!(composite_sub_mask_tiled_with_edge(
                    &mut actual,
                    &sub_mask,
                    WIDTH,
                    HEIGHT,
                    SCALE,
                    CROP_OFFSET,
                    Some(&warped),
                    TEST_TILE_EDGE,
                ));

                let actual = actual.expect("tiled composition output");
                let expected = expected.expect("full-frame composition output");
                let first_difference = actual
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .position(|(actual, expected)| actual != expected)
                    .map(|index| {
                        (
                            index as u32 % WIDTH,
                            index as u32 / WIDTH,
                            actual.as_raw()[index],
                            expected.as_raw()[index],
                        )
                    });
                assert_eq!(
                    first_difference, None,
                    "{mask_type} {mode:?} changed at an overlap seam"
                );
            }
        }
    }

    #[test]
    fn tiled_ai_masks_match_full_frame_with_exact_halo_and_one_decoded_source() {
        const WIDTH: u32 = 141;
        const HEIGHT: u32 = 109;
        const TEST_TILE_EDGE: u32 = 37;
        const SCALE: f32 = 1.25;
        const CROP_OFFSET: (f32, f32) = (11.5, 7.25);

        let full_mask = GrayImage::from_fn(181, 137, |x, y| {
            Luma([x.wrapping_mul(17).wrapping_add(y.wrapping_mul(13)) as u8])
        });
        let data_url = grayscale_png_data_url(&full_mask);
        let binary_parameters = serde_json::json!({
            "startX": 12.0,
            "startY": 9.0,
            "endX": 128.0,
            "endY": 96.0,
            "maskDataBase64": data_url,
            "grow": 100.0,
            "feather": 100.0,
            "rotation": 17.0,
            "flipHorizontal": true,
            "flipVertical": false,
            "orientationSteps": 1
        });
        let cases = vec![
            ("ai-subject", binary_parameters.clone()),
            ("ai-foreground", binary_parameters.clone()),
            ("ai-sky", binary_parameters.clone()),
            ("quick-eraser", binary_parameters),
            (
                "ai-depth",
                serde_json::json!({
                    "maskDataBase64": grayscale_png_data_url(&full_mask),
                    "minDepth": 18.0,
                    "maxDepth": 82.0,
                    "minFade": 11.0,
                    "maxFade": 9.0,
                    "grow": -100.0,
                    "feather": 60.0,
                    "rotation": -13.0,
                    "flipHorizontal": false,
                    "flipVertical": true,
                    "orientationSteps": 2
                }),
            ),
        ];

        for (mask_type, parameters) in cases {
            for mode in [
                SubMaskMode::Additive,
                SubMaskMode::Subtractive,
                SubMaskMode::Intersect,
            ] {
                let sub_mask = test_sub_mask(mask_type, parameters.clone(), mode);
                let base = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
                    Luma([x.wrapping_mul(11).wrapping_add(y * 7) as u8])
                });

                let mut expected = (mode != SubMaskMode::Additive).then(|| base.clone());
                let mut full_sub_mask =
                    generate_sub_mask_bitmap(&sub_mask, WIDTH, HEIGHT, SCALE, CROP_OFFSET, None)
                        .expect("AI mask should decode and rasterize");
                let first_pixel = full_sub_mask.get_pixel(0, 0)[0];
                assert!(
                    full_sub_mask.pixels().any(|pixel| pixel[0] != first_pixel),
                    "{mask_type} fixture must exercise non-uniform pixels"
                );
                apply_sub_mask_modifiers(&mut full_sub_mask, &sub_mask);
                composite_sub_mask(&mut expected, full_sub_mask, mode);

                let mut actual = (mode != SubMaskMode::Additive).then_some(base);
                assert!(composite_sub_mask_tiled_with_edge(
                    &mut actual,
                    &sub_mask,
                    WIDTH,
                    HEIGHT,
                    SCALE,
                    CROP_OFFSET,
                    None,
                    TEST_TILE_EDGE,
                ));

                let actual = actual.expect("tiled AI composition output");
                let expected = expected.expect("full-frame AI composition output");
                let first_difference = actual
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .position(|(actual, expected)| actual != expected)
                    .map(|index| {
                        (
                            index as u32 % WIDTH,
                            index as u32 / WIDTH,
                            actual.as_raw()[index],
                            expected.as_raw()[index],
                        )
                    });
                assert_eq!(
                    first_difference, None,
                    "{mask_type} {mode:?} changed at an overlap seam"
                );
            }
        }
    }

    #[test]
    fn ai_depth_halo_bounds_supported_60mp_filter_scratch() {
        const WIDTH: u32 = 9_504;
        const HEIGHT: u32 = 6_336;
        const FULL_MASK_BYTES: u64 = 60_217_344;
        const MAX_AI_DEPTH_HALO: u64 = 159;
        const MAX_EXPANDED_TILE_EDGE: u64 = 2_366;
        const MAX_EXPANDED_TILE_BYTES: u64 = 5_597_956;

        assert_eq!(image_gaussian_blur_radius(0.0), 0);
        assert_eq!(image_gaussian_blur_radius(0.1), 1);
        assert_eq!(image_gaussian_blur_radius(1.5), 3);
        assert_eq!(image_gaussian_blur_radius(10.0), 32);

        let grow_feather = GrowFeatherPlan::new(100.0, 100.0, WIDTH, HEIGHT);
        assert_eq!(
            u64::from(grow_feather.halo()) + u64::from(image_gaussian_blur_radius(10.0)),
            MAX_AI_DEPTH_HALO
        );
        assert_eq!(
            u64::from(GPU_TILE_SIZE) + MAX_AI_DEPTH_HALO * 2,
            MAX_EXPANDED_TILE_EDGE
        );
        assert_eq!(MAX_EXPANDED_TILE_EDGE.pow(2), MAX_EXPANDED_TILE_BYTES);
        assert_eq!(u64::from(WIDTH) * u64::from(HEIGHT), FULL_MASK_BYTES);
        assert_eq!(
            FULL_MASK_BYTES * 3 - MAX_EXPANDED_TILE_BYTES * 3,
            163_858_164
        );
    }

    #[test]
    fn grow_feather_halo_bounds_supported_60mp_range_mask_scratch() {
        const WIDTH: u32 = 9_504;
        const HEIGHT: u32 = 6_336;
        const FULL_MASK_BYTES: u64 = 60_217_344;
        const MAX_EXPANDED_TILE_EDGE: u64 = 2_302;
        const MAX_EXPANDED_TILE_BYTES: u64 = 5_299_204;

        let plan = GrowFeatherPlan::new(100.0, 100.0, WIDTH, HEIGHT);
        assert_eq!(plan.grow_amount, 63);
        assert!(plan.dilate);
        assert!((plan.feather_sigma - 31.68).abs() < 0.0001);
        assert_eq!(plan.feather_radius, 64);
        assert_eq!(plan.halo(), 127);
        assert_eq!(
            u64::from(GPU_TILE_SIZE) + u64::from(plan.halo()) * 2,
            MAX_EXPANDED_TILE_EDGE
        );
        assert_eq!(MAX_EXPANDED_TILE_EDGE.pow(2), MAX_EXPANDED_TILE_BYTES);
        assert_eq!(u64::from(WIDTH) * u64::from(HEIGHT), FULL_MASK_BYTES);
        assert_eq!(
            FULL_MASK_BYTES * 3 - MAX_EXPANDED_TILE_BYTES * 3,
            164_754_420
        );
    }

    #[test]
    fn tiled_generation_bounds_60mp_programmatic_and_brush_scratch() {
        const WIDTH: usize = 9_504;
        const HEIGHT: usize = 6_336;
        const FULL_MASK_BYTES: usize = 60_217_344;
        const MAX_TILE_BYTES: usize = 4_194_304;

        assert_eq!(WIDTH * HEIGHT, FULL_MASK_BYTES);
        assert_eq!(
            WIDTH.min(GPU_TILE_SIZE as usize) * HEIGHT.min(GPU_TILE_SIZE as usize),
            MAX_TILE_BYTES
        );
        assert_eq!(
            FULL_MASK_BYTES * 2 - (FULL_MASK_BYTES + MAX_TILE_BYTES),
            56_023_040
        );
        assert_eq!(
            FULL_MASK_BYTES * 3 - (FULL_MASK_BYTES + MAX_TILE_BYTES * 2),
            112_046_080
        );
    }

    #[test]
    fn tiled_generation_routes_supported_nonempty_submasks() {
        let radial = test_sub_mask("radial", serde_json::json!({}), SubMaskMode::Additive);
        assert!(!should_generate_sub_mask_in_tiles(&radial, false, 100, 80));
        assert!(should_generate_sub_mask_in_tiles(&radial, true, 100, 80));

        let brush = test_sub_mask("brush", brush_parameters(false), SubMaskMode::Additive);
        assert!(should_generate_sub_mask_in_tiles(&brush, false, 100, 80));

        let mut empty_brush = brush.clone();
        empty_brush.parameters = serde_json::json!({ "lines": [] });
        assert!(!should_generate_sub_mask_in_tiles(
            &empty_brush,
            false,
            100,
            80
        ));

        let mut hidden_brush = brush;
        hidden_brush.visible = false;
        assert!(!should_generate_sub_mask_in_tiles(
            &hidden_brush,
            true,
            100,
            80
        ));

        let color = test_sub_mask("color", serde_json::json!({}), SubMaskMode::Additive);
        assert!(!should_generate_sub_mask_in_tiles(&color, true, 100, 80));

        let mut color = test_sub_mask(
            "color",
            serde_json::json!({
                "targetX": 0.0,
                "targetY": 0.0,
                "grow": 0.0,
                "feather": 0.0
            }),
            SubMaskMode::Additive,
        );
        assert!(!should_generate_sub_mask_in_tiles(
            &color, false, 9_504, 6_336
        ));
        assert!(should_generate_sub_mask_in_tiles(
            &color, true, 9_504, 6_336
        ));
        color.parameters["feather"] = serde_json::json!(35.0);
        assert!(should_generate_sub_mask_in_tiles(
            &color, false, 9_504, 6_336
        ));

        let mut ai_subject = test_sub_mask(
            "ai-subject",
            serde_json::json!({ "grow": 0.0, "feather": 0.0 }),
            SubMaskMode::Additive,
        );
        assert!(!should_generate_sub_mask_in_tiles(
            &ai_subject,
            false,
            9_504,
            6_336
        ));
        assert!(should_generate_sub_mask_in_tiles(
            &ai_subject,
            true,
            9_504,
            6_336
        ));
        ai_subject.parameters["grow"] = serde_json::json!(50.0);
        assert!(should_generate_sub_mask_in_tiles(
            &ai_subject,
            false,
            9_504,
            6_336
        ));

        let ai_depth = test_sub_mask(
            "ai-depth",
            serde_json::json!({ "grow": 0.0, "feather": 15.0 }),
            SubMaskMode::Additive,
        );
        assert!(should_generate_sub_mask_in_tiles(
            &ai_depth, false, 9_504, 6_336
        ));
    }

    #[test]
    #[ignore = "manual deterministic 60MP range-mask overlap scratch benchmark"]
    fn synthetic_60mp_range_mask_overlap_scratch_harness() {
        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode = std::env::var("RAW_EDITOR_RANGE_MASK_BENCH_MODE")
            .unwrap_or_else(|_| "tiled".to_string());
        assert!(matches!(mode.as_str(), "full" | "tiled"));

        let grow_feather = GrowFeatherPlan::new(2.0, 1.0, width, height);
        let halo = grow_feather.halo();
        let mut composition = GrayImage::from_pixel(width, height, Luma([17]));
        let sub_mask = SubMask {
            id: "synthetic-range-mask".to_string(),
            mask_type: "color".to_string(),
            visible: true,
            invert: false,
            opacity: 100.0,
            mode: SubMaskMode::Additive,
            parameters: Value::Null,
        };
        let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
        let mut baseline_system = sysinfo::System::new();
        baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let baseline_rss = baseline_system
            .process(pid)
            .expect("read benchmark process after output allocation")
            .memory();

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let pixel =
            |x: u32, y: u32| Luma([x.wrapping_mul(17).wrapping_add(y.wrapping_mul(13)) as u8]);
        let started = Instant::now();
        if mode == "full" {
            let mut full_sub_mask = GrayImage::from_fn(width, height, pixel);
            grow_feather.apply(&mut full_sub_mask);
            composite_sub_mask_tile(
                &mut composition,
                &full_sub_mask,
                0,
                0,
                SubMaskMode::Additive,
            );
        } else {
            for tile_y in (0..height).step_by(GPU_TILE_SIZE as usize) {
                let tile_height = (height - tile_y).min(GPU_TILE_SIZE);
                for tile_x in (0..width).step_by(GPU_TILE_SIZE as usize) {
                    let tile_width = (width - tile_x).min(GPU_TILE_SIZE);
                    let expanded_x = tile_x.saturating_sub(halo);
                    let expanded_y = tile_y.saturating_sub(halo);
                    let expanded_right = tile_x
                        .saturating_add(tile_width)
                        .saturating_add(halo)
                        .min(width);
                    let expanded_bottom = tile_y
                        .saturating_add(tile_height)
                        .saturating_add(halo)
                        .min(height);
                    let mut tile = GrayImage::from_fn(
                        expanded_right - expanded_x,
                        expanded_bottom - expanded_y,
                        |x, y| pixel(x + expanded_x, y + expanded_y),
                    );
                    grow_feather.apply(&mut tile);
                    composite_sub_mask_tile_region(
                        &mut composition,
                        &mut tile,
                        tile_x - expanded_x,
                        tile_y - expanded_y,
                        tile_width,
                        tile_height,
                        tile_x,
                        tile_y,
                        &sub_mask,
                    );
                }
            }
        }
        let elapsed = started.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join range-mask RSS sampler");

        let sample_stride = (composition.as_raw().len() / 4_096).max(1);
        let sample_hash = composition
            .as_raw()
            .iter()
            .step_by(sample_stride)
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
                (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let peak_rss = peak_rss.load(Ordering::Relaxed);
        let expected_scratch_bytes = if mode == "full" {
            u64::from(width) * u64::from(height) * 3
        } else {
            let expanded_width = width.min(GPU_TILE_SIZE.saturating_add(halo * 2));
            let expanded_height = height.min(GPU_TILE_SIZE.saturating_add(halo * 2));
            u64::from(expanded_width) * u64::from(expanded_height) * 3
        };
        println!(
            "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"haloPixels\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"expectedScratchBytes\":{},\"sampleHash\":\"{:016x}\"}}",
            mode,
            width,
            height,
            halo,
            elapsed.as_millis(),
            baseline_rss,
            peak_rss,
            peak_rss.saturating_sub(baseline_rss),
            expected_scratch_bytes,
            sample_hash,
        );
        std::hint::black_box(composition);
    }

    #[test]
    #[ignore = "manual deterministic 60MP AI-mask overlap scratch benchmark"]
    fn synthetic_60mp_ai_mask_overlap_scratch_harness() {
        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode =
            std::env::var("RAW_EDITOR_AI_MASK_BENCH_MODE").unwrap_or_else(|_| "tiled".to_string());
        assert!(matches!(mode.as_str(), "full" | "tiled"));

        let source = GrayImage::from_fn(width, height, |x, y| {
            Luma([x.wrapping_mul(17).wrapping_add(y.wrapping_mul(13)) as u8])
        });
        let mut composition = GrayImage::from_pixel(width, height, Luma([17]));
        let transform = AiMaskTransform::from_options(None, None, None, None);
        let depth_selection = AiDepthSelection::new(18.0, 82.0, 10.0, 10.0, 1.0);
        let grow_feather = GrowFeatherPlan::new(2.0, 1.0, width, height);
        let halo = depth_selection
            .blur_radius()
            .saturating_add(grow_feather.halo());
        let sub_mask = SubMask {
            id: "synthetic-ai-mask".to_string(),
            mask_type: "ai-depth".to_string(),
            visible: true,
            invert: false,
            opacity: 100.0,
            mode: SubMaskMode::Additive,
            parameters: Value::Null,
        };
        let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
        let mut baseline_system = sysinfo::System::new();
        baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let baseline_rss = baseline_system
            .process(pid)
            .expect("read benchmark process after AI source and output allocation")
            .memory();

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let started = Instant::now();
        if mode == "full" {
            let mut full_sub_mask =
                transform.rasterize(&source, width, height, 1.0, (0.0, 0.0), (0, 0));
            depth_selection.apply_pointwise(&mut full_sub_mask);
            depth_selection.apply_blur(&mut full_sub_mask);
            grow_feather.apply(&mut full_sub_mask);
            composite_sub_mask_tile(
                &mut composition,
                &full_sub_mask,
                0,
                0,
                SubMaskMode::Additive,
            );
        } else {
            for tile_y in (0..height).step_by(GPU_TILE_SIZE as usize) {
                let tile_height = (height - tile_y).min(GPU_TILE_SIZE);
                for tile_x in (0..width).step_by(GPU_TILE_SIZE as usize) {
                    let tile_width = (width - tile_x).min(GPU_TILE_SIZE);
                    let expanded_x = tile_x.saturating_sub(halo);
                    let expanded_y = tile_y.saturating_sub(halo);
                    let expanded_right = tile_x
                        .saturating_add(tile_width)
                        .saturating_add(halo)
                        .min(width);
                    let expanded_bottom = tile_y
                        .saturating_add(tile_height)
                        .saturating_add(halo)
                        .min(height);
                    let mut tile = transform.rasterize(
                        &source,
                        expanded_right - expanded_x,
                        expanded_bottom - expanded_y,
                        1.0,
                        (0.0, 0.0),
                        (expanded_x, expanded_y),
                    );
                    depth_selection.apply_pointwise(&mut tile);
                    depth_selection.apply_blur(&mut tile);
                    grow_feather.apply(&mut tile);
                    composite_sub_mask_tile_region(
                        &mut composition,
                        &mut tile,
                        tile_x - expanded_x,
                        tile_y - expanded_y,
                        tile_width,
                        tile_height,
                        tile_x,
                        tile_y,
                        &sub_mask,
                    );
                }
            }
        }
        let elapsed = started.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join AI-mask RSS sampler");

        let sample_stride = (composition.as_raw().len() / 4_096).max(1);
        let sample_hash = composition
            .as_raw()
            .iter()
            .step_by(sample_stride)
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
                (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let pixel_bytes = u64::from(width) * u64::from(height);
        let peak_rss = peak_rss.load(Ordering::Relaxed);
        let expected_scratch_bytes = if mode == "full" {
            pixel_bytes * 3
        } else {
            let expanded_width = width.min(GPU_TILE_SIZE.saturating_add(halo * 2));
            let expanded_height = height.min(GPU_TILE_SIZE.saturating_add(halo * 2));
            u64::from(expanded_width) * u64::from(expanded_height) * 3
        };
        println!(
            "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"haloPixels\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"sourceAndOutputBytes\":{},\"expectedScratchBytes\":{},\"sampleHash\":\"{:016x}\"}}",
            mode,
            width,
            height,
            halo,
            elapsed.as_millis(),
            baseline_rss,
            peak_rss,
            peak_rss.saturating_sub(baseline_rss),
            pixel_bytes * 2,
            expected_scratch_bytes,
            sample_hash,
        );
        std::hint::black_box(source);
        std::hint::black_box(composition);
    }

    #[test]
    #[ignore = "manual deterministic 60MP mask composition scratch benchmark"]
    fn synthetic_60mp_mask_composition_scratch_harness() {
        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode = std::env::var("RAW_EDITOR_MASK_COMPOSITION_BENCH_MODE")
            .unwrap_or_else(|_| "tiled".to_string());
        assert!(matches!(mode.as_str(), "full" | "tiled"));

        let mut composition = GrayImage::from_pixel(width, height, Luma([17]));
        let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
        let mut baseline_system = sysinfo::System::new();
        baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let baseline_rss = baseline_system
            .process(pid)
            .expect("read benchmark process after output allocation")
            .memory();

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let started = Instant::now();
        if mode == "full" {
            let full_sub_mask = GrayImage::from_fn(width, height, |x, y| {
                Luma([x.wrapping_mul(17).wrapping_add(y * 13) as u8])
            });
            std::thread::sleep(Duration::from_millis(10));
            let mut output = Some(composition);
            composite_sub_mask(&mut output, full_sub_mask, SubMaskMode::Additive);
            composition = output.expect("full-frame composition output");
        } else {
            for tile_y in (0..height).step_by(GPU_TILE_SIZE as usize) {
                let tile_height = (height - tile_y).min(GPU_TILE_SIZE);
                for tile_x in (0..width).step_by(GPU_TILE_SIZE as usize) {
                    let tile_width = (width - tile_x).min(GPU_TILE_SIZE);
                    let tile = GrayImage::from_fn(tile_width, tile_height, |x, y| {
                        Luma([(x + tile_x)
                            .wrapping_mul(17)
                            .wrapping_add((y + tile_y) * 13) as u8])
                    });
                    if tile_x == 0 && tile_y == 0 {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    composite_sub_mask_tile(
                        &mut composition,
                        &tile,
                        tile_x,
                        tile_y,
                        SubMaskMode::Additive,
                    );
                }
            }
        }
        let elapsed = started.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join mask composition RSS sampler");

        let sample_stride = (composition.as_raw().len() / 4_096).max(1);
        let sample_hash = composition
            .as_raw()
            .iter()
            .step_by(sample_stride)
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
                (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let peak_rss = peak_rss.load(Ordering::Relaxed);
        let expected_scratch_bytes = if mode == "full" {
            u64::from(width) * u64::from(height)
        } else {
            u64::from(width.min(GPU_TILE_SIZE)) * u64::from(height.min(GPU_TILE_SIZE))
        };
        println!(
            "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"expectedScratchBytes\":{},\"sampleHash\":\"{:016x}\"}}",
            mode,
            width,
            height,
            elapsed.as_millis(),
            baseline_rss,
            peak_rss,
            peak_rss.saturating_sub(baseline_rss),
            expected_scratch_bytes,
            sample_hash,
        );
        std::hint::black_box(composition);
    }

    #[test]
    #[ignore = "manual deterministic 60MP mask cache ownership benchmark"]
    fn synthetic_60mp_mask_cache_ownership_harness() {
        enum Fixture {
            Cloned {
                cache: GrayImage,
                caller: GrayImage,
            },
            Shared {
                cache: Arc<GrayImage>,
                caller: Arc<GrayImage>,
            },
        }

        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode =
            std::env::var("RAW_EDITOR_MASK_BENCH_MODE").unwrap_or_else(|_| "shared".to_string());
        assert!(matches!(mode.as_str(), "shared" | "cloned"));

        let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
        let mut baseline_system = sysinfo::System::new();
        baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let baseline_rss = baseline_system
            .process(pid)
            .expect("read benchmark process before mask allocation")
            .memory();

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });

        let started = Instant::now();
        let source = GrayImage::from_fn(width, height, |x, y| {
            Luma([x.wrapping_mul(17).wrapping_add(y * 13) as u8])
        });
        let fixture = if mode == "cloned" {
            Fixture::Cloned {
                caller: source.clone(),
                cache: source,
            }
        } else {
            let cache = Arc::new(source);
            Fixture::Shared {
                caller: Arc::clone(&cache),
                cache,
            }
        };
        let elapsed = started.elapsed();
        std::thread::sleep(Duration::from_millis(25));
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join mask RSS sampler");

        let (cache_raw, caller_raw, shared_allocation) = match &fixture {
            Fixture::Cloned { cache, caller } => (cache.as_raw(), caller.as_raw(), false),
            Fixture::Shared { cache, caller } => {
                assert!(Arc::ptr_eq(cache, caller));
                (cache.as_raw(), caller.as_raw(), true)
            }
        };
        assert_eq!(cache_raw, caller_raw);
        let sample_stride = (cache_raw.len() / 4_096).max(1);
        let sample_hash = cache_raw
            .iter()
            .step_by(sample_stride)
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
                (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let mask_bytes = u64::from(width) * u64::from(height);
        let expected_live_pixel_bytes = mask_bytes * if shared_allocation { 1 } else { 2 };
        println!(
            "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"maskBytes\":{},\"expectedLivePixelBytes\":{},\"sampleHash\":\"{:016x}\"}}",
            mode,
            width,
            height,
            elapsed.as_millis(),
            baseline_rss,
            peak_rss.load(Ordering::Relaxed),
            peak_rss
                .load(Ordering::Relaxed)
                .saturating_sub(baseline_rss),
            mask_bytes,
            expected_live_pixel_bytes,
            sample_hash,
        );
        std::hint::black_box(fixture);
    }
}
