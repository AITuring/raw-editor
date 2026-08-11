use crate::color_management::{linear_to_srgb_channel, srgb_to_linear_channel};
use crate::gpu_processing::WgpuDisplay;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec2, Vec3};
use image::{DynamicImage, GenericImageView, Rgb32FImage, RgbImage, Rgba};
use imageproc::geometric_transformations::{Border, Interpolation, rotate_about_center};
use nalgebra::{Matrix3 as NaMatrix3, Vector3 as NaVector3};
use rawler::decoders::Orientation;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::json;
use std::borrow::Cow;
use std::f32::consts::PI;
use std::sync::Arc;

pub use crate::gpu_processing::{
    RenderRequest, get_or_init_gpu_context, process_and_get_dynamic_image,
    process_and_get_dynamic_image_with_analytics,
};
use crate::{AppState, mask_generation::MaskDefinition};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

pub trait IntoCowImage<'a> {
    fn into_cow(self) -> Cow<'a, DynamicImage>;
}

impl<'a> IntoCowImage<'a> for DynamicImage {
    fn into_cow(self) -> Cow<'a, DynamicImage> {
        Cow::Owned(self)
    }
}

impl<'a> IntoCowImage<'a> for &'a DynamicImage {
    fn into_cow(self) -> Cow<'a, DynamicImage> {
        Cow::Borrowed(self)
    }
}

impl<'a> IntoCowImage<'a> for Cow<'a, DynamicImage> {
    fn into_cow(self) -> Cow<'a, DynamicImage> {
        self
    }
}

impl<'a> IntoCowImage<'a> for &'a std::sync::Arc<DynamicImage> {
    fn into_cow(self) -> Cow<'a, DynamicImage> {
        Cow::Borrowed(self.as_ref())
    }
}

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

pub fn downscale_f32_image(image: &DynamicImage, nwidth: u32, nheight: u32) -> DynamicImage {
    let start = std::time::Instant::now();

    let (width, height) = image.dimensions();
    if nwidth == 0 || nheight == 0 || (nwidth >= width && nheight >= height) {
        return image.clone();
    }

    let ratio = (nwidth as f32 / width as f32).min(nheight as f32 / height as f32);
    let new_w = (width as f32 * ratio).round() as u32;
    let new_h = (height as f32 * ratio).round() as u32;

    if new_w == 0 || new_h == 0 {
        return image.clone();
    }

    let tmp_img;
    let img_ref = if let Some(rgb) = image.as_rgb32f() {
        rgb
    } else {
        tmp_img = image.to_rgb32f();
        &tmp_img
    };
    let src: &[f32] = img_ref.as_raw();

    let x_ratio = width as f32 / new_w as f32;
    let y_ratio = height as f32 / new_h as f32;
    let width_usize = width as usize;

    let mut x_bounds = Vec::with_capacity(new_w as usize);
    let mut x_weights = Vec::new();
    for x_out in 0..new_w as usize {
        let x_start = x_out as f32 * x_ratio;
        let x_end = (x_out + 1) as f32 * x_ratio;
        let x_in_start = x_start.floor() as usize;
        let x_in_end = (x_end.ceil() as usize).min(width as usize);

        let weight_start_idx = x_weights.len();
        let mut w_sum = 0.0;
        let mut tmp_w = Vec::with_capacity(x_in_end.saturating_sub(x_in_start));

        let mut actual_start = x_in_end;
        let mut actual_end = x_in_start;

        for x_in in x_in_start..x_in_end {
            let overlap_start = x_start.max(x_in as f32);
            let overlap_end = x_end.min((x_in + 1) as f32);
            let w = (overlap_end - overlap_start).max(0.0);
            if w > 0.0 {
                actual_start = actual_start.min(x_in);
                actual_end = actual_end.max(x_in + 1);
                tmp_w.push(w);
                w_sum += w;
            }
        }

        if w_sum > 0.0 {
            let inv_w = 1.0 / w_sum;
            for w in tmp_w {
                x_weights.push(w * inv_w);
            }
            x_bounds.push((actual_start, actual_end, weight_start_idx));
        } else {
            x_bounds.push((0, 0, weight_start_idx));
        }
    }

    let mut y_bounds = Vec::with_capacity(new_h as usize);
    let mut y_weights = Vec::new();
    for y_out in 0..new_h as usize {
        let y_start = y_out as f32 * y_ratio;
        let y_end = (y_out + 1) as f32 * y_ratio;
        let y_in_start = y_start.floor() as usize;
        let y_in_end = (y_end.ceil() as usize).min(height as usize);

        let weight_start_idx = y_weights.len();
        let mut w_sum = 0.0;
        let mut tmp_w = Vec::with_capacity(y_in_end.saturating_sub(y_in_start));

        let mut actual_start = y_in_end;
        let mut actual_end = y_in_start;

        for y_in in y_in_start..y_in_end {
            let overlap_start = y_start.max(y_in as f32);
            let overlap_end = y_end.min((y_in + 1) as f32);
            let w = (overlap_end - overlap_start).max(0.0);
            if w > 0.0 {
                actual_start = actual_start.min(y_in);
                actual_end = actual_end.max(y_in + 1);
                tmp_w.push(w);
                w_sum += w;
            }
        }

        if w_sum > 0.0 {
            let inv_w = 1.0 / w_sum;
            for w in tmp_w {
                y_weights.push(w * inv_w);
            }
            y_bounds.push((actual_start, actual_end, weight_start_idx));
        } else {
            y_bounds.push((0, 0, weight_start_idx));
        }
    }

    let mut out_buf = vec![0.0f32; (new_w * new_h * 3) as usize];

    out_buf
        .par_chunks_exact_mut(new_w as usize * 3)
        .enumerate()
        .for_each(|(y_out, row)| {
            let (y_in_start, y_in_end, y_wt_offset) = y_bounds[y_out];
            let y_len = y_in_end - y_in_start;
            let y_wts = &y_weights[y_wt_offset..y_wt_offset + y_len];

            for (x_out, &(x_in_start, x_in_end, x_wt_offset)) in x_bounds.iter().enumerate() {
                let mut r_sum = 0.0;
                let mut g_sum = 0.0;
                let mut b_sum = 0.0;

                let x_len = x_in_end - x_in_start;
                let x_wts = &x_weights[x_wt_offset..x_wt_offset + x_len];

                for (dy, &w_y) in y_wts.iter().enumerate() {
                    let y_in = y_in_start + dy;
                    let row_offset = y_in * width_usize * 3;

                    let src_start = row_offset + x_in_start * 3;
                    let src_end = row_offset + x_in_end * 3;
                    let src_slice = &src[src_start..src_end];

                    for (&w_x, chunk) in x_wts.iter().zip(src_slice.chunks_exact(3)) {
                        let w = w_x * w_y;

                        let r = chunk[0].max(0.0);
                        let g = chunk[1].max(0.0);
                        let b = chunk[2].max(0.0);

                        r_sum += r * r * w;
                        g_sum += g * g * w;
                        b_sum += b * b * w;
                    }
                }

                let out_idx = x_out * 3;
                row[out_idx] = r_sum.sqrt();
                row[out_idx + 1] = g_sum.sqrt();
                row[out_idx + 2] = b_sum.sqrt();
            }
        });

    let out = Rgb32FImage::from_raw(new_w, new_h, out_buf).expect("buffer size mismatch");
    let result = DynamicImage::ImageRgb32F(out);

    log::info!("downscale_f32_image took {:.2?}", start.elapsed());

    result
}

#[inline(always)]
fn interpolate_pixel(
    src_raw: &[f32],
    src_width: usize,
    src_height: usize,
    x: f32,
    y: f32,
    pixel_out: &mut [f32],
) {
    if x.is_nan()
        || y.is_nan()
        || x < 0.0
        || y < 0.0
        || x >= (src_width as f32 - 1.0)
        || y >= (src_height as f32 - 1.0)
    {
        return;
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;

    let wx = x - x0 as f32;
    let wy = y - y0 as f32;
    let one_minus_wx = 1.0 - wx;
    let one_minus_wy = 1.0 - wy;

    let stride = src_width * 3;
    let idx_row0 = y0 * stride;
    let idx_row1 = idx_row0 + stride;
    let idx_p00 = idx_row0 + x0 * 3;

    unsafe {
        let p00 = src_raw.get_unchecked(idx_p00..idx_p00 + 3);
        let p10 = src_raw.get_unchecked(idx_p00 + 3..idx_p00 + 6);
        let p01 = src_raw.get_unchecked(idx_row1 + x0 * 3..idx_row1 + x0 * 3 + 3);
        let p11 = src_raw.get_unchecked(idx_row1 + x0 * 3 + 3..idx_row1 + x0 * 3 + 6);

        let top_r = p00[0] * one_minus_wx + p10[0] * wx;
        let top_g = p00[1] * one_minus_wx + p10[1] * wx;
        let top_b = p00[2] * one_minus_wx + p10[2] * wx;

        let bot_r = p01[0] * one_minus_wx + p11[0] * wx;
        let bot_g = p01[1] * one_minus_wx + p11[1] * wx;
        let bot_b = p01[2] * one_minus_wx + p11[2] * wx;

        pixel_out[0] = top_r * one_minus_wy + bot_r * wy;
        pixel_out[1] = top_g * one_minus_wy + bot_g * wy;
        pixel_out[2] = top_b * one_minus_wy + bot_b * wy;
    }
}

fn build_transform_matrices(
    params: &GeometryParams,
    width: f32,
    height: f32,
) -> (NaMatrix3<f32>, f32, f32, f64) {
    let cx = width / 2.0;
    let cy = height / 2.0;
    let ref_dim = 2000.0;

    let p_vert = (params.vertical / 100000.0) * (ref_dim / height);
    let p_horiz = (-params.horizontal / 100000.0) * (ref_dim / width);
    let theta = params.rotate.to_radians();

    let aspect_factor = if params.aspect >= 0.0 {
        1.0 + params.aspect / 100.0
    } else {
        1.0 / (1.0 + params.aspect.abs() / 100.0)
    };

    let scale_factor = params.scale / 100.0;
    let off_x = (params.x_offset / 100.0) * width;
    let off_y = (params.y_offset / 100.0) * height;

    let t_center = NaMatrix3::new(1.0, 0.0, cx, 0.0, 1.0, cy, 0.0, 0.0, 1.0);
    let t_uncenter = NaMatrix3::new(1.0, 0.0, -cx, 0.0, 1.0, -cy, 0.0, 0.0, 1.0);
    let m_perspective = NaMatrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, p_horiz, p_vert, 1.0);

    let (sin_t, cos_t) = theta.sin_cos();
    let m_rotate = NaMatrix3::new(cos_t, -sin_t, 0.0, sin_t, cos_t, 0.0, 0.0, 0.0, 1.0);
    let m_scale = NaMatrix3::new(
        scale_factor * aspect_factor,
        0.0,
        0.0,
        0.0,
        scale_factor,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let m_offset = NaMatrix3::new(1.0, 0.0, off_x, 0.0, 1.0, off_y, 0.0, 0.0, 1.0);

    let forward = t_center * m_offset * m_perspective * m_rotate * m_scale * t_uncenter;
    let half_diagonal =
        ((width as f64 * width as f64 + height as f64 * height as f64).sqrt()) / 2.0;

    (forward, cx, cy, half_diagonal)
}

const MAX_PROJECTION_STRENGTH: f64 = 1.1;

#[inline(always)]
fn remap_projection_point(
    x: f64,
    y: f64,
    cx: f64,
    cy: f64,
    half_diagonal: f64,
    projection: f32,
    inverse: bool,
) -> (f64, f64) {
    if projection.abs() < 1e-5 || half_diagonal < 1e-6 {
        return (x, y);
    }

    let dx = x - cx;
    let dy = y - cy;
    let radius = (dx * dx + dy * dy).sqrt();
    if radius < 1e-6 {
        return (x, y);
    }

    let normalized_radius = radius / half_diagonal;
    let strength = (projection.abs() as f64 / 100.0) * MAX_PROJECTION_STRENGTH;
    let compress = if inverse {
        projection < 0.0
    } else {
        projection > 0.0
    };
    let mapped_radius = if compress {
        (strength * normalized_radius).atan() / strength
    } else {
        let angle = (strength * normalized_radius).min(std::f64::consts::FRAC_PI_2 - 1e-4);
        angle.tan() / strength
    };
    let scale = mapped_radius / normalized_radius;
    (cx + dx * scale, cy + dy * scale)
}

#[inline(always)]
fn project_geometry_point(
    x: f64,
    y: f64,
    cx: f64,
    cy: f64,
    half_diagonal: f64,
    projection: f32,
) -> (f64, f64) {
    remap_projection_point(x, y, cx, cy, half_diagonal, projection, false)
}

#[inline(always)]
fn unproject_geometry_point(
    x: f64,
    y: f64,
    cx: f64,
    cy: f64,
    half_diagonal: f64,
    projection: f32,
) -> (f64, f64) {
    remap_projection_point(x, y, cx, cy, half_diagonal, projection, true)
}

struct TcaContext<'a> {
    src_raw: &'a [f32],
    src_width: usize,
    src_height: usize,
    cx: f32,
    cy: f32,
}

#[inline(always)]
fn interpolate_pixel_with_tca(
    tca: &TcaContext,
    base_x: f32,
    base_y: f32,
    vr: f32,
    vb: f32,
    pixel_out: &mut [f32],
) {
    let src_raw = tca.src_raw;
    let src_width = tca.src_width;
    let src_height = tca.src_height;
    let cx = tca.cx;
    let cy = tca.cy;
    let gx = base_x;
    let gy = base_y;

    let rx = cx + (base_x - cx) * vr;
    let ry = cy + (base_y - cy) * vr;

    let bx = cx + (base_x - cx) * vb;
    let by = cy + (base_y - cy) * vb;

    let sample_channel = |target_x: f32, target_y: f32, channel_idx: usize| -> f32 {
        if target_x.is_nan() || target_y.is_nan() {
            return 0.0;
        }

        let x_clamped = target_x.clamp(0.0, src_width as f32 - 1.0);
        let y_clamped = target_y.clamp(0.0, src_height as f32 - 1.0);

        let mut x0 = x_clamped.floor() as usize;
        let mut y0 = y_clamped.floor() as usize;

        if x0 >= src_width - 1 {
            x0 = src_width.saturating_sub(2);
        }
        if y0 >= src_height - 1 {
            y0 = src_height.saturating_sub(2);
        }

        let wx = x_clamped - x0 as f32;
        let wy = y_clamped - y0 as f32;
        let one_minus_wx = 1.0 - wx;
        let one_minus_wy = 1.0 - wy;

        let stride = src_width * 3;
        let idx_row0 = y0 * stride;
        let idx_row1 = idx_row0 + stride;

        let idx_p00 = idx_row0 + x0 * 3 + channel_idx;

        unsafe {
            let p00 = *src_raw.get_unchecked(idx_p00);
            let p10 = *src_raw.get_unchecked(idx_p00 + 3);
            let p01 = *src_raw.get_unchecked(idx_row1 + x0 * 3 + channel_idx);
            let p11 = *src_raw.get_unchecked(idx_row1 + x0 * 3 + 3 + channel_idx);

            let top = p00 * one_minus_wx + p10 * wx;
            let bot = p01 * one_minus_wx + p11 * wx;
            top * one_minus_wy + bot * wy
        }
    };

    pixel_out[0] = sample_channel(rx, ry, 0);
    pixel_out[1] = sample_channel(gx, gy, 1);
    pixel_out[2] = sample_channel(bx, by, 2);
}

fn solve_generic_distortion_inv(r_target: f64, k_scaled: f64) -> f64 {
    if k_scaled.abs() < 1e-9 {
        return r_target;
    }

    let mut r = r_target;
    for _ in 0..10 {
        let r2 = r * r;
        let val = k_scaled * r2 * r + r - r_target;
        let slope = 3.0 * k_scaled * r2 + 1.0;

        if slope.abs() < 1e-9 {
            break;
        }
        let delta = val / slope;
        r -= delta;
        if delta.abs() < 1e-6 {
            break;
        }
    }
    r
}

fn compute_lens_auto_crop_scale(params: &GeometryParams, width: f32, height: f32) -> f64 {
    let cx = (width / 2.0) as f64;
    let cy = (height / 2.0) as f64;
    let half_diagonal = (cx * cx + cy * cy).sqrt();
    let max_radius_sq_inv = 1.0 / (cx * cx + cy * cy);

    let lk1 = params.lens_dist_k1 as f64;
    let lk2 = params.lens_dist_k2 as f64;
    let lk3 = params.lens_dist_k3 as f64;
    let lens_dist_amt = (params.lens_distortion_amount as f64) * 2.5;

    let k_distortion = (params.distortion as f64 / 100.0) * 2.5;

    let has_lens_correction = params.lens_distortion_enabled
        && (lk1.abs() > 1e-6 || lk2.abs() > 1e-6 || lk3.abs() > 1e-6);
    let is_ptlens = params.lens_model == 1;

    let sample_points: [(f64, f64); 8] = [
        (cx, 0.0),
        (cx, height as f64),
        (0.0, cy),
        (width as f64, cy),
        (0.0, 0.0),
        (width as f64, 0.0),
        (0.0, height as f64),
        (width as f64, height as f64),
    ];

    let mut max_scale: f64 = 1.0;

    for &(px, py) in &sample_points {
        let dx = px - cx;
        let dy = py - cy;
        let ru = (dx * dx + dy * dy).sqrt();
        if ru < 1e-6 {
            continue;
        }

        let mut mapped_dx = dx;
        let mut mapped_dy = dy;

        if has_lens_correction {
            let ru_norm = ru / half_diagonal;
            let ru_norm2 = ru_norm * ru_norm;

            let rd_norm = if is_ptlens {
                let a = lk1;
                let b = lk2;
                let c = lk3;
                let d = 1.0 - a - b - c;
                ru_norm * (a * ru_norm2 * ru_norm + b * ru_norm2 + c * ru_norm + d)
            } else {
                ru_norm
                    * (1.0
                        + lk1 * ru_norm2
                        + lk2 * (ru_norm2 * ru_norm2)
                        + lk3 * (ru_norm2 * ru_norm2 * ru_norm2))
            };

            let effective_r_norm = ru_norm + (rd_norm - ru_norm) * lens_dist_amt;
            let scale = effective_r_norm / ru_norm;

            mapped_dx *= scale;
            mapped_dy *= scale;
        }

        if k_distortion.abs() > 1e-5 {
            let r2_norm = (mapped_dx * mapped_dx + mapped_dy * mapped_dy) * max_radius_sq_inv;
            let f = 1.0 + k_distortion * r2_norm;
            mapped_dx *= f;
            mapped_dy *= f;
        }

        let mapped_ru = (mapped_dx * mapped_dx + mapped_dy * mapped_dy).sqrt();
        let scale = mapped_ru / ru;

        if scale > max_scale {
            max_scale = scale;
        }
    }

    if max_scale > 1.0 {
        max_scale * 1.002
    } else {
        max_scale
    }
}

pub fn warp_image_geometry(image: &DynamicImage, params: GeometryParams) -> DynamicImage {
    let src_img = image.to_rgb32f();
    let (width, height) = src_img.dimensions();
    let mut out_buffer = vec![0.0f32; (width * height * 3) as usize];

    let (forward_transform, cx, cy, half_diagonal) =
        build_transform_matrices(&params, width as f32, height as f32);
    let inv = forward_transform
        .try_inverse()
        .unwrap_or(NaMatrix3::identity());

    let step_vec_x = NaVector3::new(inv[(0, 0)], inv[(1, 0)], inv[(2, 0)]);
    let step_vec_y = NaVector3::new(inv[(0, 1)], inv[(1, 1)], inv[(2, 1)]);
    let origin_vec = NaVector3::new(inv[(0, 2)], inv[(1, 2)], inv[(2, 2)]);

    let max_radius_sq_inv = 1.0 / ((cx * cx + cy * cy) as f64);
    let hd = half_diagonal;

    let k_distortion = (params.distortion as f64 / 100.0) * 2.5;
    let lk1 = params.lens_dist_k1 as f64;
    let lk2 = params.lens_dist_k2 as f64;
    let lk3 = params.lens_dist_k3 as f64;
    let lens_dist_amt = (params.lens_distortion_amount as f64) * 2.5;

    let has_lens_correction = params.lens_distortion_enabled
        && (lk1.abs() > 1e-6 || lk2.abs() > 1e-6 || lk3.abs() > 1e-6);
    let is_ptlens = params.lens_model == 1;

    let auto_crop_scale = if has_lens_correction || k_distortion.abs() > 1e-5 {
        compute_lens_auto_crop_scale(&params, width as f32, height as f32) as f32
    } else {
        1.0
    };

    let vr = if (params.tca_vr - 1.0).abs() > 1e-5 {
        params.tca_vr + (1.0 - params.tca_vr) * (1.0 - params.lens_tca_amount)
    } else {
        1.0
    };
    let vb = if (params.tca_vb - 1.0).abs() > 1e-5 {
        params.tca_vb + (1.0 - params.tca_vb) * (1.0 - params.lens_tca_amount)
    } else {
        1.0
    };
    let has_tca = params.lens_tca_enabled && ((vr - 1.0).abs() > 1e-5 || (vb - 1.0).abs() > 1e-5);

    let vk1 = params.vig_k1 as f64;
    let vk2 = params.vig_k2 as f64;
    let vk3 = params.vig_k3 as f64;
    let lens_vig_amt = (params.lens_vignette_amount as f64) * 0.8;
    let has_vignetting = params.lens_vignette_enabled
        && (vk1.abs() > 1e-6 || vk2.abs() > 1e-6 || vk3.abs() > 1e-6)
        && lens_vig_amt > 0.01;

    let src_raw = src_img.as_raw();
    let width_usize = width as usize;
    let height_usize = height as usize;
    let tca_ctx = TcaContext {
        src_raw,
        src_width: width_usize,
        src_height: height_usize,
        cx,
        cy,
    };

    out_buffer
        .par_chunks_exact_mut(width_usize * 3)
        .enumerate()
        .for_each(|(y, row_pixel_data)| {
            let y_f = y as f32;
            let mut current_vec = origin_vec + (step_vec_y * y_f);

            for pixel in row_pixel_data.chunks_exact_mut(3) {
                if current_vec.z.abs() > 1e-6 {
                    let inv_z = 1.0 / current_vec.z;
                    let mut src_x = current_vec.x * inv_z;
                    let mut src_y = current_vec.y * inv_z;

                    let (unprojected_x, unprojected_y) = unproject_geometry_point(
                        src_x as f64,
                        src_y as f64,
                        cx as f64,
                        cy as f64,
                        hd,
                        params.projection,
                    );
                    src_x = unprojected_x as f32;
                    src_y = unprojected_y as f32;

                    if auto_crop_scale > 1.0 {
                        src_x = cx + (src_x - cx) / auto_crop_scale;
                        src_y = cy + (src_y - cy) / auto_crop_scale;
                    }

                    if has_lens_correction {
                        let dx = (src_x - cx) as f64;
                        let dy = (src_y - cy) as f64;
                        let ru = (dx * dx + dy * dy).sqrt();

                        if ru > 1e-6 {
                            let ru_norm = ru / hd;
                            let ru_norm2 = ru_norm * ru_norm;

                            let rd_norm = if is_ptlens {
                                let a = lk1;
                                let b = lk2;
                                let c = lk3;
                                let d = 1.0 - a - b - c;
                                ru_norm * (a * ru_norm2 * ru_norm + b * ru_norm2 + c * ru_norm + d)
                            } else {
                                ru_norm
                                    * (1.0
                                        + lk1 * ru_norm2
                                        + lk2 * (ru_norm2 * ru_norm2)
                                        + lk3 * (ru_norm2 * ru_norm2 * ru_norm2))
                            };

                            let effective_r_norm = ru_norm + (rd_norm - ru_norm) * lens_dist_amt;
                            let scale = effective_r_norm / ru_norm;

                            src_x = cx + (dx * scale) as f32;
                            src_y = cy + (dy * scale) as f32;
                        }
                    }

                    if k_distortion.abs() > 1e-5 {
                        let dx = (src_x - cx) as f64;
                        let dy = (src_y - cy) as f64;
                        let r2_norm = (dx * dx + dy * dy) * max_radius_sq_inv;
                        let f = 1.0 + k_distortion * r2_norm;

                        src_x = cx + (dx * f) as f32;
                        src_y = cy + (dy * f) as f32;
                    }

                    if has_tca {
                        interpolate_pixel_with_tca(&tca_ctx, src_x, src_y, vr, vb, pixel);
                    } else {
                        interpolate_pixel(src_raw, width_usize, height_usize, src_x, src_y, pixel);
                    }

                    if has_vignetting {
                        let dx = (src_x - cx) as f64;
                        let dy = (src_y - cy) as f64;
                        let ru = (dx * dx + dy * dy).sqrt();
                        let ru_norm = ru / hd;
                        let ru_norm2 = ru_norm * ru_norm;

                        let v_factor = 1.0
                            + vk1 * ru_norm2
                            + vk2 * (ru_norm2 * ru_norm2)
                            + vk3 * (ru_norm2 * ru_norm2 * ru_norm2);

                        if v_factor > 1e-6 {
                            let correction_gain = 1.0 / v_factor;
                            let final_gain = 1.0 + (correction_gain - 1.0) * lens_vig_amt;

                            pixel[0] *= final_gain as f32;
                            pixel[1] *= final_gain as f32;
                            pixel[2] *= final_gain as f32;
                        }
                    }
                }
                current_vec += step_vec_x;
            }
        });

    let out_img = Rgb32FImage::from_vec(width, height, out_buffer).unwrap();
    DynamicImage::ImageRgb32F(out_img)
}

pub fn unwarp_image_geometry(warped_image: &DynamicImage, params: GeometryParams) -> DynamicImage {
    let src_img = warped_image.to_rgb32f();
    let (width, height) = src_img.dimensions();
    let mut out_buffer = vec![0.0f32; (width * height * 3) as usize];

    let (forward_transform, cx, cy, half_diagonal) =
        build_transform_matrices(&params, width as f32, height as f32);
    let max_radius_sq_inv = 1.0 / ((cx * cx + cy * cy) as f64);
    let hd = half_diagonal;

    let k_distortion = (params.distortion as f64 / 100.0) * 2.5;
    let lk1 = params.lens_dist_k1 as f64;
    let lk2 = params.lens_dist_k2 as f64;
    let lk3 = params.lens_dist_k3 as f64;
    let lens_dist_amt = (params.lens_distortion_amount as f64) * 2.5;

    let has_lens_correction = params.lens_distortion_enabled
        && (lk1.abs() > 1e-6 || lk2.abs() > 1e-6 || lk3.abs() > 1e-6);
    let is_ptlens = params.lens_model == 1;

    let auto_crop_scale = if has_lens_correction || k_distortion.abs() > 1e-5 {
        compute_lens_auto_crop_scale(&params, width as f32, height as f32) as f32
    } else {
        1.0
    };

    let src_raw = src_img.as_raw();
    let width_usize = width as usize;
    let height_usize = height as usize;

    out_buffer
        .par_chunks_exact_mut(width_usize * 3)
        .enumerate()
        .for_each(|(y, row_pixel_data)| {
            let y_f = y as f32;

            for (x, pixel) in row_pixel_data.chunks_exact_mut(3).enumerate() {
                let x_f = x as f32;
                let mut current_x = x_f;
                let mut current_y = y_f;

                if k_distortion.abs() > 1e-5 {
                    let dx = (current_x - cx) as f64;
                    let dy = (current_y - cy) as f64;
                    let r_distorted = (dx * dx + dy * dy).sqrt();

                    if r_distorted > 1e-6 {
                        let k_effective = k_distortion * max_radius_sq_inv;
                        let r_straight = solve_generic_distortion_inv(r_distorted, k_effective);

                        let scale = r_straight / r_distorted;
                        current_x = cx + (dx * scale) as f32;
                        current_y = cy + (dy * scale) as f32;
                    }
                }

                if has_lens_correction {
                    let dx = (current_x - cx) as f64;
                    let dy = (current_y - cy) as f64;
                    let rd = (dx * dx + dy * dy).sqrt();

                    if rd > 1e-6 {
                        let mut ru = rd;

                        for _ in 0..8 {
                            let ru_norm = ru / hd;
                            let ru_norm2 = ru_norm * ru_norm;

                            let (f_val, f_prime) = if is_ptlens {
                                let a = lk1;
                                let b = lk2;
                                let c = lk3;
                                let d = 1.0 - a - b - c;
                                let poly = a * ru_norm2 * ru_norm + b * ru_norm2 + c * ru_norm + d;

                                let val = ru * poly;
                                let prime = 4.0 * a * ru_norm2 * ru_norm
                                    + 3.0 * b * ru_norm2
                                    + 2.0 * c * ru_norm
                                    + d;
                                (val, prime)
                            } else {
                                let poly = 1.0
                                    + lk1 * ru_norm2
                                    + lk2 * (ru_norm2 * ru_norm2)
                                    + lk3 * (ru_norm2 * ru_norm2 * ru_norm2);
                                let val = ru * poly;
                                let poly_prime = 2.0 * lk1 * ru_norm
                                    + 4.0 * lk2 * ru_norm2 * ru_norm
                                    + 6.0 * lk3 * (ru_norm2 * ru_norm2) * ru_norm;
                                let prime = poly + ru_norm * poly_prime;
                                (val, prime)
                            };

                            let g_val = ru + (f_val - ru) * lens_dist_amt - rd;
                            let g_prime = 1.0 + (f_prime - 1.0) * lens_dist_amt;

                            if g_prime.abs() < 1e-7 {
                                break;
                            }
                            let delta = g_val / g_prime;
                            ru -= delta;
                            if delta.abs() < 1e-4 {
                                break;
                            }
                        }

                        let scale = ru / rd;
                        current_x = cx + (dx * scale) as f32;
                        current_y = cy + (dy * scale) as f32;
                    }
                }

                if auto_crop_scale > 1.0 {
                    current_x = cx + (current_x - cx) * auto_crop_scale;
                    current_y = cy + (current_y - cy) * auto_crop_scale;
                }

                let (projected_x, projected_y) = project_geometry_point(
                    current_x as f64,
                    current_y as f64,
                    cx as f64,
                    cy as f64,
                    hd,
                    params.projection,
                );
                current_x = projected_x as f32;
                current_y = projected_y as f32;

                let target_vec = forward_transform * NaVector3::new(current_x, current_y, 1.0);

                if target_vec.z.abs() > 1e-6 {
                    let inv_z = 1.0 / target_vec.z;

                    let src_x = target_vec.x * inv_z;
                    let src_y = target_vec.y * inv_z;

                    interpolate_pixel(src_raw, width_usize, height_usize, src_x, src_y, pixel);
                }
            }
        });

    let out_img = Rgb32FImage::from_vec(width, height, out_buffer).unwrap();
    DynamicImage::ImageRgb32F(out_img)
}

pub fn inverse_transform_mask(
    mask: image::GrayImage,
    adjustments: &serde_json::Value,
) -> image::GrayImage {
    let rotation_degrees = adjustments
        .get("rotation")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let mask_dyn = image::DynamicImage::ImageLuma8(mask);

    let unrotated_fine = if rotation_degrees.abs() > 1e-5 {
        crate::image_processing::apply_rotation(mask_dyn, -rotation_degrees).into_owned()
    } else {
        mask_dyn
    };

    let flip_h = adjustments
        .get("flipHorizontal")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let flip_v = adjustments
        .get("flipVertical")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let flipped = apply_flip(unrotated_fine, flip_h, flip_v).into_owned();

    let steps = adjustments
        .get("orientationSteps")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u8;
    let inverse_steps = (4 - (steps % 4)) % 4;
    let unrotated_coarse = apply_coarse_rotation(flipped, inverse_steps).into_owned();

    let unwarped = apply_unwarp_geometry(unrotated_coarse, adjustments).into_owned();

    unwarped.into_luma8()
}

pub fn inverse_transform_point(
    mut x: f64,
    mut y: f64,
    mut curr_w: f64,
    mut curr_h: f64,
    adjustments: &serde_json::Value,
) -> (f64, f64) {
    let rotation_degrees = adjustments
        .get("rotation")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    if rotation_degrees.abs() > 1e-5 {
        let cx = curr_w / 2.0;
        let cy = curr_h / 2.0;
        let theta_rad = -rotation_degrees * std::f64::consts::PI / 180.0;
        let cos_t = theta_rad.cos();
        let sin_t = theta_rad.sin();

        let dx = x - cx;
        let dy = y - cy;
        x = cx + dx * cos_t - dy * sin_t;
        y = cy + dx * sin_t + dy * cos_t;
    }

    let flip_h = adjustments
        .get("flipHorizontal")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let flip_v = adjustments
        .get("flipVertical")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if flip_h {
        x = curr_w - x;
    }
    if flip_v {
        y = curr_h - y;
    }

    let steps = adjustments
        .get("orientationSteps")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u8;
    let inverse_steps = (4 - (steps % 4)) % 4;
    for _ in 0..inverse_steps {
        let new_x = curr_h - y;
        let new_y = x;
        x = new_x;
        y = new_y;
        std::mem::swap(&mut curr_w, &mut curr_h);
    }

    let params = get_geometry_params_from_json(adjustments);
    let width = curr_w as f32;
    let height = curr_h as f32;

    let (forward_transform, cx_f32, cy_f32, hd) = build_transform_matrices(&params, width, height);
    let cx = cx_f32 as f64;
    let cy = cy_f32 as f64;
    let inv = forward_transform
        .try_inverse()
        .unwrap_or(nalgebra::Matrix3::identity());

    let vec = inv * nalgebra::Vector3::new(x as f32, y as f32, 1.0);
    if vec.z.abs() > 1e-6 {
        let inv_z = 1.0 / (vec.z as f64);
        let mut src_x = (vec.x as f64) * inv_z;
        let mut src_y = (vec.y as f64) * inv_z;

        (src_x, src_y) = unproject_geometry_point(src_x, src_y, cx, cy, hd, params.projection);

        let k_distortion = (params.distortion as f64 / 100.0) * 2.5;
        let lk1 = params.lens_dist_k1 as f64;
        let lk2 = params.lens_dist_k2 as f64;
        let lk3 = params.lens_dist_k3 as f64;
        let lens_dist_amt = (params.lens_distortion_amount as f64) * 2.5;

        let has_lens_correction = params.lens_distortion_enabled
            && (lk1.abs() > 1e-6 || lk2.abs() > 1e-6 || lk3.abs() > 1e-6);
        let is_ptlens = params.lens_model == 1;

        let auto_crop_scale = if has_lens_correction || k_distortion.abs() > 1e-5 {
            compute_lens_auto_crop_scale(&params, width, height)
        } else {
            1.0
        };

        if auto_crop_scale > 1.0 {
            src_x = cx + (src_x - cx) / auto_crop_scale;
            src_y = cy + (src_y - cy) / auto_crop_scale;
        }

        if has_lens_correction {
            let dx = src_x - cx;
            let dy = src_y - cy;
            let ru = (dx * dx + dy * dy).sqrt();

            if ru > 1e-6 {
                let ru_norm = ru / hd;
                let ru_norm2 = ru_norm * ru_norm;

                let rd_norm = if is_ptlens {
                    let a = lk1;
                    let b = lk2;
                    let c = lk3;
                    let d = 1.0 - a - b - c;
                    ru_norm * (a * ru_norm2 * ru_norm + b * ru_norm2 + c * ru_norm + d)
                } else {
                    ru_norm
                        * (1.0
                            + lk1 * ru_norm2
                            + lk2 * (ru_norm2 * ru_norm2)
                            + lk3 * (ru_norm2 * ru_norm2 * ru_norm2))
                };

                let effective_r_norm = ru_norm + (rd_norm - ru_norm) * lens_dist_amt;
                let scale = effective_r_norm / ru_norm;

                src_x = cx + (dx * scale);
                src_y = cy + (dy * scale);
            }
        }

        if k_distortion.abs() > 1e-5 {
            let max_radius_sq_inv = 1.0 / (cx * cx + cy * cy);
            let dx = src_x - cx;
            let dy = src_y - cy;
            let r2_norm = (dx * dx + dy * dy) * max_radius_sq_inv;
            let f = 1.0 + k_distortion * r2_norm;

            src_x = cx + (dx * f);
            src_y = cy + (dy * f);
        }

        return (src_x, src_y);
    }

    (x, y)
}

pub fn apply_cpu_default_raw_processing(image: &mut DynamicImage) {
    let mut f32_image = image.to_rgb32f();

    const GAMMA: f32 = 2.38;
    const INV_GAMMA: f32 = 1.0 / GAMMA;
    const CONTRAST: f32 = 1.28;

    f32_image.par_chunks_mut(3).for_each(|pixel_chunk| {
        let r_gamma = pixel_chunk[0].powf(INV_GAMMA);
        let g_gamma = pixel_chunk[1].powf(INV_GAMMA);
        let b_gamma = pixel_chunk[2].powf(INV_GAMMA);

        let r_contrast = (r_gamma - 0.5) * CONTRAST + 0.5;
        let g_contrast = (g_gamma - 0.5) * CONTRAST + 0.5;
        let b_contrast = (b_gamma - 0.5) * CONTRAST + 0.5;

        pixel_chunk[0] = r_contrast.clamp(0.0, 1.0);
        pixel_chunk[1] = g_contrast.clamp(0.0, 1.0);
        pixel_chunk[2] = b_contrast.clamp(0.0, 1.0);
    });

    *image = DynamicImage::ImageRgb32F(f32_image);
}

pub fn apply_srgb_to_linear(mut image: DynamicImage) -> DynamicImage {
    match &mut image {
        DynamicImage::ImageRgb32F(img) => {
            img.as_mut()
                .par_iter_mut()
                .for_each(|c| *c = srgb_to_linear_channel(*c));
        }
        DynamicImage::ImageRgba32F(img) => {
            img.par_chunks_mut(4).for_each(|p| {
                p[0] = srgb_to_linear_channel(p[0]);
                p[1] = srgb_to_linear_channel(p[1]);
                p[2] = srgb_to_linear_channel(p[2]);
            });
        }
        _ => {}
    }
    image
}

pub fn apply_linear_to_srgb(mut image: DynamicImage) -> DynamicImage {
    match &mut image {
        DynamicImage::ImageRgb32F(img) => {
            img.as_mut()
                .par_iter_mut()
                .for_each(|c| *c = linear_to_srgb_channel(*c));
        }
        DynamicImage::ImageRgba32F(img) => {
            img.par_chunks_mut(4).for_each(|p| {
                p[0] = linear_to_srgb_channel(p[0]);
                p[1] = linear_to_srgb_channel(p[1]);
                p[2] = linear_to_srgb_channel(p[2]);
            });
        }
        _ => {}
    }
    image
}

pub fn apply_orientation(image: DynamicImage, orientation: Orientation) -> DynamicImage {
    match orientation {
        Orientation::Normal | Orientation::Unknown => image,
        Orientation::HorizontalFlip => image.fliph(),
        Orientation::Rotate180 => image.rotate180(),
        Orientation::VerticalFlip => image.flipv(),
        Orientation::Transpose => image.rotate90().fliph(),
        Orientation::Rotate90 => image.rotate90(),
        Orientation::Transverse => image.rotate270().fliph(),
        Orientation::Rotate270 => image.rotate270(),
    }
}

pub fn apply_geometry_warp<'a>(
    image: impl IntoCowImage<'a>,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    let params = get_geometry_params_from_json(adjustments);
    if !is_geometry_identity(&params) {
        Cow::Owned(warp_image_geometry(image.as_ref(), params))
    } else {
        image
    }
}

pub fn apply_unwarp_geometry<'a>(
    image: impl IntoCowImage<'a>,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    let params = get_geometry_params_from_json(adjustments);
    if !is_geometry_identity(&params) {
        Cow::Owned(unwarp_image_geometry(image.as_ref(), params))
    } else {
        image
    }
}

pub fn apply_coarse_rotation<'a>(
    image: impl IntoCowImage<'a>,
    orientation_steps: u8,
) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    match orientation_steps {
        1 => Cow::Owned(image.rotate90()),
        2 => Cow::Owned(image.rotate180()),
        3 => Cow::Owned(image.rotate270()),
        _ => image,
    }
}

pub fn apply_rotation<'a>(
    image: impl IntoCowImage<'a>,
    rotation_degrees: f32,
) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    if rotation_degrees % 360.0 == 0.0 {
        return image;
    }

    let rgba_image = image.to_rgba32f();
    let rotated = rotate_about_center(
        &rgba_image,
        rotation_degrees * PI / 180.0,
        Interpolation::Bilinear,
        Border::Constant(Rgba([0.0f32, 0.0, 0.0, 0.0])),
    );

    Cow::Owned(DynamicImage::ImageRgba32F(rotated))
}

pub fn apply_crop<'a>(image: impl IntoCowImage<'a>, crop_value: &Value) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    if crop_value.is_null() {
        return image;
    }

    if let Ok(crop) = serde_json::from_value::<Crop>(crop_value.clone()) {
        let x = crop.x.round() as u32;
        let y = crop.y.round() as u32;
        let width = crop.width.round() as u32;
        let height = crop.height.round() as u32;

        if width > 0 && height > 0 {
            let (img_w, img_h) = image.dimensions();
            if x < img_w && y < img_h {
                let new_width = (img_w - x).min(width);
                let new_height = (img_h - y).min(height);

                if new_width > 0 && new_height > 0 {
                    if x == 0 && y == 0 && new_width == img_w && new_height == img_h {
                        return image;
                    }
                    return Cow::Owned(image.crop_imm(x, y, new_width, new_height));
                }
            }
        }
    }
    image
}

pub fn apply_flip<'a>(
    image: impl IntoCowImage<'a>,
    horizontal: bool,
    vertical: bool,
) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    if !horizontal && !vertical {
        return image;
    }

    let mut img = image.into_owned();
    if horizontal {
        img = img.fliph();
    }
    if vertical {
        img = img.flipv();
    }
    Cow::Owned(img)
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AutoAdjustmentResults {
    pub exposure: f64,
    pub brightness: f64,
    pub contrast: f64,
    pub highlights: f64,
    pub shadows: f64,
    pub vibrancy: f64,
    pub vignette_amount: f64,
    pub temperature: f64,
    pub tint: f64,
    pub dehaze: f64,
    pub clarity: f64,
    pub centre: f64,
    pub blacks: f64,
    pub whites: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct Point {
    x: f32,
    y: f32,
    _pad1: f32,
    _pad2: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct HslColor {
    hue: f32,
    saturation: f32,
    luminance: f32,
    _pad: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct ColorGradeSettings {
    pub hue: f32,
    pub saturation: f32,
    pub luminance: f32,
    _pad: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct ColorCalibrationSettings {
    pub shadows_tint: f32,
    pub red_hue: f32,
    pub red_saturation: f32,
    pub green_hue: f32,
    pub green_saturation: f32,
    pub blue_hue: f32,
    pub blue_saturation: f32,
    _pad1: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuMat3 {
    col0: [f32; 4],
    col1: [f32; 4],
    col2: [f32; 4],
}

impl Default for GpuMat3 {
    fn default() -> Self {
        Self {
            col0: [1.0, 0.0, 0.0, 0.0],
            col1: [0.0, 1.0, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GlobalAdjustments {
    pub exposure: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub saturation: f32,
    pub temperature: f32,
    pub tint: f32,
    pub vibrance: f32,
    pub hue: f32,
    _pad_color1: f32,
    _pad_color2: f32,
    _pad_color3: f32,

    pub sharpness: f32,
    pub luma_noise_reduction: f32,
    pub color_noise_reduction: f32,
    pub clarity: f32,
    pub dehaze: f32,
    pub structure: f32,
    pub centré: f32,
    pub vignette_amount: f32,
    pub vignette_midpoint: f32,
    pub vignette_roundness: f32,
    pub vignette_feather: f32,
    pub grain_amount: f32,
    pub grain_size: f32,
    pub grain_roughness: f32,

    pub chromatic_aberration_red_cyan: f32,
    pub chromatic_aberration_blue_yellow: f32,
    pub show_clipping: u32,
    pub is_raw_image: u32,
    _pad_ca1: f32,

    pub has_lut: u32,
    pub lut_intensity: f32,
    pub tonemapper_mode: u32,
    _pad_lut2: f32,
    _pad_lut3: f32,
    _pad_lut4: f32,
    _pad_lut5: f32,

    _pad_agx1: f32,
    _pad_agx2: f32,
    _pad_agx3: f32,
    pub agx_pipe_to_rendering_matrix: GpuMat3,
    pub agx_rendering_to_pipe_matrix: GpuMat3,

    _pad_cg1: f32,
    _pad_cg2: f32,
    _pad_cg3: f32,
    _pad_cg4: f32,
    pub color_grading_shadows: ColorGradeSettings,
    pub color_grading_midtones: ColorGradeSettings,
    pub color_grading_highlights: ColorGradeSettings,
    pub color_grading_global: ColorGradeSettings,
    pub color_grading_blending: f32,
    pub color_grading_balance: f32,
    _pad2: f32,
    _pad3: f32,

    pub color_calibration: ColorCalibrationSettings,

    pub hsl: [HslColor; 8],
    pub luma_curve: [Point; 16],
    pub red_curve: [Point; 16],
    pub green_curve: [Point; 16],
    pub blue_curve: [Point; 16],
    pub luma_curve_count: u32,
    pub red_curve_count: u32,
    pub green_curve_count: u32,
    pub blue_curve_count: u32,
    _pad_end1: f32,
    _pad_end2: f32,
    _pad_end3: f32,
    _pad_end4: f32,

    pub glow_amount: f32,
    pub halation_amount: f32,
    pub flare_amount: f32,
    pub sharpness_threshold: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct MaskAdjustments {
    pub exposure: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub saturation: f32,
    pub temperature: f32,
    pub tint: f32,
    pub vibrance: f32,

    pub sharpness: f32,
    pub luma_noise_reduction: f32,
    pub color_noise_reduction: f32,
    pub clarity: f32,
    pub dehaze: f32,
    pub structure: f32,

    pub glow_amount: f32,
    pub halation_amount: f32,
    pub flare_amount: f32,
    pub sharpness_threshold: f32,

    pub hue: f32,
    _pad_cg1: f32,
    _pad_cg2: f32,
    pub color_grading_shadows: ColorGradeSettings,
    pub color_grading_midtones: ColorGradeSettings,
    pub color_grading_highlights: ColorGradeSettings,
    pub color_grading_global: ColorGradeSettings,
    pub color_grading_blending: f32,
    pub color_grading_balance: f32,
    _pad5: f32,
    _pad6: f32,

    pub hsl: [HslColor; 8],
    pub luma_curve: [Point; 16],
    pub red_curve: [Point; 16],
    pub green_curve: [Point; 16],
    pub blue_curve: [Point; 16],
    pub luma_curve_count: u32,
    pub red_curve_count: u32,
    pub green_curve_count: u32,
    pub blue_curve_count: u32,
    _pad_end4: f32,
    _pad_end5: f32,
    _pad_end6: f32,
    _pad_end7: f32,
}

pub const MAX_MASKS: usize = 32;

#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct AllAdjustments {
    pub global: GlobalAdjustments,
    pub mask_adjustments: [MaskAdjustments; MAX_MASKS],
    pub mask_count: u32,
    pub tile_offset_x: u32,
    pub tile_offset_y: u32,
    pub mask_atlas_cols: u32,
}

struct AdjustmentScales {
    exposure: f32,
    brightness: f32,
    contrast: f32,
    highlights: f32,
    shadows: f32,
    whites: f32,
    blacks: f32,
    saturation: f32,
    temperature: f32,
    tint: f32,
    vibrance: f32,

    sharpness: f32,
    sharpness_threshold: f32,
    luma_noise_reduction: f32,
    color_noise_reduction: f32,
    clarity: f32,
    dehaze: f32,
    structure: f32,
    centré: f32,

    vignette_amount: f32,
    vignette_midpoint: f32,
    vignette_roundness: f32,
    vignette_feather: f32,
    grain_amount: f32,
    grain_size: f32,
    grain_roughness: f32,

    chromatic_aberration: f32,

    hsl_hue_multiplier: f32,
    hsl_saturation: f32,
    hsl_luminance: f32,

    color_grading_saturation: f32,
    color_grading_luminance: f32,
    color_grading_blending: f32,
    color_grading_balance: f32,

    color_calibration_hue: f32,
    color_calibration_saturation: f32,

    glow: f32,
    halation: f32,
    flares: f32,
}

const SCALES: AdjustmentScales = AdjustmentScales {
    exposure: 0.8,
    brightness: 0.8,
    contrast: 100.0,
    highlights: 120.0,
    shadows: 120.0,
    whites: 30.0,
    blacks: 40.0,
    saturation: 100.0,
    temperature: 25.0,
    tint: 100.0,
    vibrance: 100.0,

    sharpness: 50.0,
    sharpness_threshold: 100.0,
    luma_noise_reduction: 100.0,
    color_noise_reduction: 100.0,
    clarity: 125.0,
    dehaze: 750.0,
    structure: 125.0,
    centré: 250.0,

    vignette_amount: 100.0,
    vignette_midpoint: 100.0,
    vignette_roundness: 100.0,
    vignette_feather: 100.0,
    grain_amount: 200.0,
    grain_size: 50.0,
    grain_roughness: 100.0,

    chromatic_aberration: 10000.0,

    hsl_hue_multiplier: 0.3,
    hsl_saturation: 100.0,
    hsl_luminance: 100.0,

    color_grading_saturation: 500.0,
    color_grading_luminance: 500.0,
    color_grading_blending: 100.0,
    color_grading_balance: 200.0,

    color_calibration_hue: 400.0,
    color_calibration_saturation: 120.0,

    glow: 100.0,
    halation: 100.0,
    flares: 100.0,
};

fn parse_hsl_adjustments(js_hsl: &serde_json::Value) -> [HslColor; 8] {
    let mut hsl_array = [HslColor::default(); 8];
    if let Some(hsl_map) = js_hsl.as_object() {
        let color_map = [
            ("reds", 0),
            ("oranges", 1),
            ("yellows", 2),
            ("greens", 3),
            ("aquas", 4),
            ("blues", 5),
            ("purples", 6),
            ("magentas", 7),
        ];
        for (name, index) in color_map.iter() {
            if let Some(color_data) = hsl_map.get(*name) {
                hsl_array[*index] = HslColor {
                    hue: color_data["hue"].as_f64().unwrap_or(0.0) as f32
                        * SCALES.hsl_hue_multiplier,
                    saturation: color_data["saturation"].as_f64().unwrap_or(0.0) as f32
                        / SCALES.hsl_saturation,
                    luminance: color_data["luminance"].as_f64().unwrap_or(0.0) as f32
                        / SCALES.hsl_luminance,
                    _pad: 0.0,
                };
            }
        }
    }
    hsl_array
}

fn parse_color_grade_settings(js_cg: &serde_json::Value) -> ColorGradeSettings {
    if js_cg.is_null() {
        return ColorGradeSettings::default();
    }
    ColorGradeSettings {
        hue: js_cg["hue"].as_f64().unwrap_or(0.0) as f32,
        saturation: js_cg["saturation"].as_f64().unwrap_or(0.0) as f32
            / SCALES.color_grading_saturation,
        luminance: js_cg["luminance"].as_f64().unwrap_or(0.0) as f32
            / SCALES.color_grading_luminance,
        _pad: 0.0,
    }
}

fn convert_points_to_aligned(frontend_points: Vec<serde_json::Value>) -> [Point; 16] {
    let mut aligned_points = [Point::default(); 16];
    for (i, point) in frontend_points.iter().enumerate().take(16) {
        if let (Some(x), Some(y)) = (point["x"].as_f64(), point["y"].as_f64()) {
            aligned_points[i] = Point {
                x: x as f32,
                y: y as f32,
                _pad1: 0.0,
                _pad2: 0.0,
            };
        }
    }
    aligned_points
}

const WP_D65: Vec2 = Vec2::new(0.3127, 0.3290);
const PRIMARIES_SRGB: [Vec2; 3] = [
    Vec2::new(0.64, 0.33),
    Vec2::new(0.30, 0.60),
    Vec2::new(0.15, 0.06),
];
const PRIMARIES_REC2020: [Vec2; 3] = [
    Vec2::new(0.708, 0.292),
    Vec2::new(0.170, 0.797),
    Vec2::new(0.131, 0.046),
];

fn xy_to_xyz(xy: Vec2) -> Vec3 {
    if xy.y < 1e-6 {
        Vec3::ZERO
    } else {
        Vec3::new(xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y)
    }
}

fn primaries_to_xyz_matrix(primaries: &[Vec2; 3], white_point: Vec2) -> Mat3 {
    let r_xyz = xy_to_xyz(primaries[0]);
    let g_xyz = xy_to_xyz(primaries[1]);
    let b_xyz = xy_to_xyz(primaries[2]);
    let primaries_matrix = Mat3::from_cols(r_xyz, g_xyz, b_xyz);
    let white_point_xyz = xy_to_xyz(white_point);
    let s = primaries_matrix.inverse() * white_point_xyz;
    Mat3::from_cols(r_xyz * s.x, g_xyz * s.y, b_xyz * s.z)
}

fn rotate_and_scale_primary(primary: Vec2, white_point: Vec2, scale: f32, rotation: f32) -> Vec2 {
    let p_rel = primary - white_point;
    let p_scaled = p_rel * scale;
    let (sin_r, cos_r) = rotation.sin_cos();
    let p_rotated = Vec2::new(
        p_scaled.x * cos_r - p_scaled.y * sin_r,
        p_scaled.x * sin_r + p_scaled.y * cos_r,
    );
    white_point + p_rotated
}

fn mat3_to_gpu_mat3(m: Mat3) -> GpuMat3 {
    GpuMat3 {
        col0: [m.x_axis.x, m.x_axis.y, m.x_axis.z, 0.0],
        col1: [m.y_axis.x, m.y_axis.y, m.y_axis.z, 0.0],
        col2: [m.z_axis.x, m.z_axis.y, m.z_axis.z, 0.0],
    }
}

fn calculate_agx_matrices_glam() -> (Mat3, Mat3) {
    let pipe_work_profile_to_xyz = primaries_to_xyz_matrix(&PRIMARIES_SRGB, WP_D65);
    let base_profile_to_xyz = primaries_to_xyz_matrix(&PRIMARIES_REC2020, WP_D65);
    let xyz_to_base_profile = base_profile_to_xyz.inverse();
    let pipe_to_base = xyz_to_base_profile * pipe_work_profile_to_xyz;

    let inset = [0.294_624_5, 0.25861925, 0.14641371];
    let rotation = [0.03540329, -0.02108586, -0.06305724];
    let outset = [0.290_776_4, 0.263_155_4, 0.045_810_72];
    let unrotation = [0.03540329, -0.02108586, -0.06305724];
    let master_outset_ratio = 1.0;
    let master_unrotation_ratio = 0.0;

    let mut inset_and_rotated_primaries = [Vec2::ZERO; 3];
    for i in 0..3 {
        inset_and_rotated_primaries[i] =
            rotate_and_scale_primary(PRIMARIES_REC2020[i], WP_D65, 1.0 - inset[i], rotation[i]);
    }
    let rendering_to_xyz = primaries_to_xyz_matrix(&inset_and_rotated_primaries, WP_D65);
    let base_to_rendering = xyz_to_base_profile * rendering_to_xyz;

    let mut outset_and_unrotated_primaries = [Vec2::ZERO; 3];
    for i in 0..3 {
        outset_and_unrotated_primaries[i] = rotate_and_scale_primary(
            PRIMARIES_REC2020[i],
            WP_D65,
            1.0 - master_outset_ratio * outset[i],
            master_unrotation_ratio * unrotation[i],
        );
    }
    let outset_to_xyz = primaries_to_xyz_matrix(&outset_and_unrotated_primaries, WP_D65);
    let temp_matrix = xyz_to_base_profile * outset_to_xyz;
    let rendering_to_base = temp_matrix.inverse();

    let pipe_to_rendering = base_to_rendering * pipe_to_base;
    let rendering_to_pipe = pipe_to_base.inverse() * rendering_to_base;

    (pipe_to_rendering, rendering_to_pipe)
}

fn calculate_agx_matrices() -> (GpuMat3, GpuMat3) {
    let (pipe_to_rendering, rendering_to_pipe) = calculate_agx_matrices_glam();
    (
        mat3_to_gpu_mat3(pipe_to_rendering),
        mat3_to_gpu_mat3(rendering_to_pipe),
    )
}

pub fn resolve_tonemapper_override(settings: &crate::AppSettings, is_raw: bool) -> Option<u32> {
    if !settings.tonemapper_override_enabled.unwrap_or(false) {
        return None;
    }
    let tm = if is_raw {
        settings.default_raw_tonemapper.as_deref().unwrap_or("agx")
    } else {
        settings
            .default_non_raw_tonemapper
            .as_deref()
            .unwrap_or("basic")
    };
    Some(if tm == "agx" { 1 } else { 0 })
}

pub fn resolve_tonemapper_override_from_handle(
    app_handle: &tauri::AppHandle,
    is_raw: bool,
) -> Option<u32> {
    let settings = crate::app_settings::load_settings(app_handle.clone()).unwrap_or_default();
    resolve_tonemapper_override(&settings, is_raw)
}

pub fn apply_cpu_agx_tonemap(image: &mut DynamicImage) {
    const AGX_EPSILON: f32 = 1.0e-6;
    const AGX_MIN_EV: f32 = -15.2;
    const AGX_MAX_EV: f32 = 5.0;
    const AGX_RANGE_EV: f32 = AGX_MAX_EV - AGX_MIN_EV;
    const AGX_GAMMA: f32 = 2.4;
    const AGX_SLOPE: f32 = 2.3843;
    const AGX_TOE_POWER: f32 = 1.5;
    const AGX_SHOULDER_POWER: f32 = 1.5;
    const AGX_TOE_TRANSITION_X: f32 = 0.6060606;
    const AGX_TOE_TRANSITION_Y: f32 = 0.43446;
    const AGX_SHOULDER_TRANSITION_X: f32 = 0.6060606;
    const AGX_SHOULDER_TRANSITION_Y: f32 = 0.43446;
    const AGX_INTERCEPT: f32 = -1.0112;
    const AGX_TOE_SCALE: f32 = -1.0359;
    const AGX_SHOULDER_SCALE: f32 = 1.3475;

    fn agx_sigmoid(x: f32, power: f32) -> f32 {
        x / (1.0 + x.powf(power)).powf(1.0 / power)
    }

    fn agx_scaled_sigmoid(x: f32, scale: f32, slope: f32, power: f32, tx: f32, ty: f32) -> f32 {
        scale * agx_sigmoid(slope * (x - tx) / scale, power) + ty
    }

    fn agx_curve_channel(x: f32) -> f32 {
        let result = if x < AGX_TOE_TRANSITION_X {
            agx_scaled_sigmoid(
                x,
                AGX_TOE_SCALE,
                AGX_SLOPE,
                AGX_TOE_POWER,
                AGX_TOE_TRANSITION_X,
                AGX_TOE_TRANSITION_Y,
            )
        } else if x <= AGX_SHOULDER_TRANSITION_X {
            AGX_SLOPE * x + AGX_INTERCEPT
        } else {
            agx_scaled_sigmoid(
                x,
                AGX_SHOULDER_SCALE,
                AGX_SLOPE,
                AGX_SHOULDER_POWER,
                AGX_SHOULDER_TRANSITION_X,
                AGX_SHOULDER_TRANSITION_Y,
            )
        };
        result.clamp(0.0, 1.0)
    }

    const LUT_SIZE: usize = 4096;
    let mut curve_lut = [0.0f32; LUT_SIZE];
    for (i, slot) in curve_lut.iter_mut().enumerate() {
        let x = i as f32 / (LUT_SIZE - 1) as f32;
        *slot = agx_curve_channel(x).max(0.0).powf(AGX_GAMMA);
    }

    let (pipe_to_rendering, rendering_to_pipe) = calculate_agx_matrices_glam();

    let mut f32_image = image.to_rgb32f();

    f32_image.par_chunks_mut(3).for_each(|pixel_chunk| {
        let r = pixel_chunk[0];
        let g = pixel_chunk[1];
        let b = pixel_chunk[2];

        let min_c = r.min(g).min(b);
        let (r, g, b) = if min_c < 0.0 {
            (r - min_c, g - min_c, b - min_c)
        } else {
            (r, g, b)
        };

        let in_rendering = pipe_to_rendering * Vec3::new(r, g, b);

        let x = Vec3::new(
            (in_rendering.x / 0.18).max(AGX_EPSILON),
            (in_rendering.y / 0.18).max(AGX_EPSILON),
            (in_rendering.z / 0.18).max(AGX_EPSILON),
        );
        let log_encoded = Vec3::new(
            (x.x.log2() - AGX_MIN_EV) / AGX_RANGE_EV,
            (x.y.log2() - AGX_MIN_EV) / AGX_RANGE_EV,
            (x.z.log2() - AGX_MIN_EV) / AGX_RANGE_EV,
        );
        let mapped = Vec3::new(
            log_encoded.x.clamp(0.0, 1.0),
            log_encoded.y.clamp(0.0, 1.0),
            log_encoded.z.clamp(0.0, 1.0),
        );

        let lut_lookup = |v: f32| -> f32 {
            let idx = (v * (LUT_SIZE - 1) as f32) as usize;
            curve_lut[idx.min(LUT_SIZE - 1)]
        };
        let curved = Vec3::new(
            lut_lookup(mapped.x),
            lut_lookup(mapped.y),
            lut_lookup(mapped.z),
        );

        let final_color = rendering_to_pipe * curved;

        pixel_chunk[0] = final_color.x.clamp(0.0, 1.0);
        pixel_chunk[1] = final_color.y.clamp(0.0, 1.0);
        pixel_chunk[2] = final_color.z.clamp(0.0, 1.0);
    });

    *image = DynamicImage::ImageRgb32F(f32_image);
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

fn get_global_adjustments_from_json(
    js_adjustments: &serde_json::Value,
    is_raw: bool,
    tonemapper_override: Option<u32>,
) -> GlobalAdjustments {
    let visibility = js_adjustments.get("sectionVisibility");
    let is_visible = |section: &str| -> bool {
        visibility
            .and_then(|v| v.get(section))
            .and_then(|s| s.as_bool())
            .unwrap_or(true)
    };
    let is_visible_with_legacy = |section: &str, legacy: &str| -> bool {
        visibility
            .and_then(|v| v.get(section))
            .and_then(|s| s.as_bool())
            .unwrap_or_else(|| is_visible(legacy))
    };
    let uses_camera_raw_sections = visibility
        .and_then(|v| v.get("colorMixer"))
        .and_then(|s| s.as_bool())
        .is_some();
    let is_reorganized_visible = |section: &str, legacy: &str| -> bool {
        if uses_camera_raw_sections {
            is_visible(section)
        } else {
            is_visible(legacy)
        }
    };

    let get_val = |section: &str, key: &str, scale: f32, default: Option<f64>| -> f32 {
        if is_visible(section) {
            js_adjustments[key]
                .as_f64()
                .unwrap_or(default.unwrap_or(0.0)) as f32
                / scale
        } else {
            if let Some(d) = default {
                d as f32 / scale
            } else {
                0.0
            }
        }
    };

    let default_curve = serde_json::json!([{"x": 0.0, "y": 0.0}, {"x": 255.0, "y": 255.0}]);
    let curves_obj = js_adjustments.get("curves").cloned().unwrap_or_default();

    let luma_points: Vec<serde_json::Value> = if is_visible("curves") {
        curves_obj
            .get("luma")
            .unwrap_or(&default_curve)
            .as_array()
            .cloned()
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let red_points: Vec<serde_json::Value> = if is_visible("curves") {
        curves_obj
            .get("red")
            .unwrap_or(&default_curve)
            .as_array()
            .cloned()
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let green_points: Vec<serde_json::Value> = if is_visible("curves") {
        curves_obj
            .get("green")
            .unwrap_or(&default_curve)
            .as_array()
            .cloned()
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let blue_points: Vec<serde_json::Value> = if is_visible("curves") {
        curves_obj
            .get("blue")
            .unwrap_or(&default_curve)
            .as_array()
            .cloned()
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let cg_obj = js_adjustments
        .get("colorGrading")
        .cloned()
        .unwrap_or_default();

    let cal_obj = js_adjustments
        .get("colorCalibration")
        .cloned()
        .unwrap_or_default();

    let color_cal_settings = if is_visible_with_legacy("calibration", "color") {
        ColorCalibrationSettings {
            shadows_tint: cal_obj["shadowsTint"].as_f64().unwrap_or(0.0) as f32
                / SCALES.color_calibration_hue,
            red_hue: cal_obj["redHue"].as_f64().unwrap_or(0.0) as f32
                / SCALES.color_calibration_hue,
            red_saturation: cal_obj["redSaturation"].as_f64().unwrap_or(0.0) as f32
                / SCALES.color_calibration_saturation,
            green_hue: cal_obj["greenHue"].as_f64().unwrap_or(0.0) as f32
                / SCALES.color_calibration_hue,
            green_saturation: cal_obj["greenSaturation"].as_f64().unwrap_or(0.0) as f32
                / SCALES.color_calibration_saturation,
            blue_hue: cal_obj["blueHue"].as_f64().unwrap_or(0.0) as f32
                / SCALES.color_calibration_hue,
            blue_saturation: cal_obj["blueSaturation"].as_f64().unwrap_or(0.0) as f32
                / SCALES.color_calibration_saturation,
            _pad1: 0.0,
        }
    } else {
        ColorCalibrationSettings::default()
    };

    let tone_mapper = if is_visible("basic") {
        js_adjustments["toneMapper"].as_str().unwrap_or("basic")
    } else {
        "basic"
    };
    let (pipe_to_rendering, rendering_to_pipe) = calculate_agx_matrices();

    let (has_lut, lut_intensity) = if is_visible("effects") {
        (
            if js_adjustments["lutPath"].is_string() {
                1
            } else {
                0
            },
            js_adjustments["lutIntensity"].as_f64().unwrap_or(100.0) as f32 / 100.0,
        )
    } else {
        (0, 1.0)
    };

    GlobalAdjustments {
        exposure: get_val("basic", "exposure", SCALES.exposure, None),
        brightness: get_val("basic", "brightness", SCALES.brightness, None),
        contrast: get_val("basic", "contrast", SCALES.contrast, None),
        highlights: get_val("basic", "highlights", SCALES.highlights, None),
        shadows: get_val("basic", "shadows", SCALES.shadows, None),
        whites: get_val("basic", "whites", SCALES.whites, None),
        blacks: get_val("basic", "blacks", SCALES.blacks, None),

        saturation: if is_visible("color") {
            js_adjustments["saturation"].as_f64().unwrap_or(0.0) as f32 / SCALES.saturation
        } else {
            0.0
        },
        temperature: if is_visible("color") {
            js_adjustments["temperature"].as_f64().unwrap_or(0.0) as f32 / SCALES.temperature
        } else {
            0.0
        },
        tint: if is_visible("color") {
            js_adjustments["tint"].as_f64().unwrap_or(0.0) as f32 / SCALES.tint
        } else {
            0.0
        },
        vibrance: if is_visible("color") {
            js_adjustments["vibrance"].as_f64().unwrap_or(0.0) as f32 / SCALES.vibrance
        } else {
            0.0
        },
        hue: if is_visible_with_legacy("colorMixer", "color") {
            js_adjustments["hue"].as_f64().unwrap_or(0.0) as f32
        } else {
            0.0
        },
        _pad_color1: 0.0,
        _pad_color2: 0.0,
        _pad_color3: 0.0,

        sharpness: get_val("details", "sharpness", SCALES.sharpness, None),
        luma_noise_reduction: get_val(
            "details",
            "lumaNoiseReduction",
            SCALES.luma_noise_reduction,
            None,
        ),
        color_noise_reduction: get_val(
            "details",
            "colorNoiseReduction",
            SCALES.color_noise_reduction,
            None,
        ),

        clarity: if is_reorganized_visible("effects", "details") {
            js_adjustments["clarity"].as_f64().unwrap_or(0.0) as f32 / SCALES.clarity
        } else {
            0.0
        },
        dehaze: if is_reorganized_visible("effects", "details") {
            js_adjustments["dehaze"].as_f64().unwrap_or(0.0) as f32 / SCALES.dehaze
        } else {
            0.0
        },
        structure: if is_reorganized_visible("effects", "details") {
            js_adjustments["structure"].as_f64().unwrap_or(0.0) as f32 / SCALES.structure
        } else {
            0.0
        },
        centré: if is_reorganized_visible("effects", "details") {
            js_adjustments["centré"].as_f64().unwrap_or(0.0) as f32 / SCALES.centré
        } else {
            0.0
        },
        vignette_amount: get_val("effects", "vignetteAmount", SCALES.vignette_amount, None),
        vignette_midpoint: get_val(
            "effects",
            "vignetteMidpoint",
            SCALES.vignette_midpoint,
            Some(50.0),
        ),
        vignette_roundness: get_val(
            "effects",
            "vignetteRoundness",
            SCALES.vignette_roundness,
            Some(0.0),
        ),
        vignette_feather: get_val(
            "effects",
            "vignetteFeather",
            SCALES.vignette_feather,
            Some(50.0),
        ),
        grain_amount: get_val("effects", "grainAmount", SCALES.grain_amount, None),
        grain_size: get_val("effects", "grainSize", SCALES.grain_size, Some(25.0)),
        grain_roughness: get_val(
            "effects",
            "grainRoughness",
            SCALES.grain_roughness,
            Some(50.0),
        ),

        chromatic_aberration_red_cyan: if is_visible_with_legacy("optics", "details") {
            js_adjustments["chromaticAberrationRedCyan"]
                .as_f64()
                .unwrap_or(0.0) as f32
                / SCALES.chromatic_aberration
        } else {
            0.0
        },
        chromatic_aberration_blue_yellow: if is_visible_with_legacy("optics", "details") {
            js_adjustments["chromaticAberrationBlueYellow"]
                .as_f64()
                .unwrap_or(0.0) as f32
                / SCALES.chromatic_aberration
        } else {
            0.0
        },
        show_clipping: if js_adjustments["showClipping"].as_bool().unwrap_or(false) {
            1
        } else {
            0
        },
        is_raw_image: if is_raw { 1 } else { 0 },
        _pad_ca1: 0.0,

        has_lut,
        lut_intensity,

        tonemapper_mode: tonemapper_override
            .unwrap_or_else(|| if tone_mapper == "agx" { 1 } else { 0 }),
        _pad_lut2: 0.0,
        _pad_lut3: 0.0,
        _pad_lut4: 0.0,
        _pad_lut5: 0.0,

        _pad_agx1: 0.0,
        _pad_agx2: 0.0,
        _pad_agx3: 0.0,
        agx_pipe_to_rendering_matrix: pipe_to_rendering,
        agx_rendering_to_pipe_matrix: rendering_to_pipe,

        _pad_cg1: 0.0,
        _pad_cg2: 0.0,
        _pad_cg3: 0.0,
        _pad_cg4: 0.0,
        color_grading_shadows: if is_visible_with_legacy("colorGrading", "color") {
            parse_color_grade_settings(&cg_obj["shadows"])
        } else {
            ColorGradeSettings::default()
        },
        color_grading_midtones: if is_visible_with_legacy("colorGrading", "color") {
            parse_color_grade_settings(&cg_obj["midtones"])
        } else {
            ColorGradeSettings::default()
        },
        color_grading_highlights: if is_visible_with_legacy("colorGrading", "color") {
            parse_color_grade_settings(&cg_obj["highlights"])
        } else {
            ColorGradeSettings::default()
        },
        color_grading_global: if is_visible_with_legacy("colorGrading", "color") {
            parse_color_grade_settings(&cg_obj["global"])
        } else {
            ColorGradeSettings::default()
        },
        color_grading_blending: if is_visible_with_legacy("colorGrading", "color") {
            cg_obj["blending"].as_f64().unwrap_or(50.0) as f32 / SCALES.color_grading_blending
        } else {
            0.5
        },
        color_grading_balance: if is_visible_with_legacy("colorGrading", "color") {
            cg_obj["balance"].as_f64().unwrap_or(0.0) as f32 / SCALES.color_grading_balance
        } else {
            0.0
        },
        _pad2: 0.0,
        _pad3: 0.0,

        color_calibration: color_cal_settings,

        hsl: if is_visible_with_legacy("colorMixer", "color") {
            parse_hsl_adjustments(&js_adjustments.get("hsl").cloned().unwrap_or_default())
        } else {
            [HslColor::default(); 8]
        },
        luma_curve: convert_points_to_aligned(luma_points.clone()),
        red_curve: convert_points_to_aligned(red_points.clone()),
        green_curve: convert_points_to_aligned(green_points.clone()),
        blue_curve: convert_points_to_aligned(blue_points.clone()),
        luma_curve_count: luma_points.len() as u32,
        red_curve_count: red_points.len() as u32,
        green_curve_count: green_points.len() as u32,
        blue_curve_count: blue_points.len() as u32,
        _pad_end1: 0.0,
        _pad_end2: 0.0,
        _pad_end3: 0.0,
        _pad_end4: 0.0,

        glow_amount: get_val("effects", "glowAmount", SCALES.glow, None),
        halation_amount: get_val("effects", "halationAmount", SCALES.halation, None),
        flare_amount: get_val("effects", "flareAmount", SCALES.flares, None),
        sharpness_threshold: get_val(
            "details",
            "sharpnessThreshold",
            SCALES.sharpness_threshold,
            Some(15.0),
        ),
    }
}

fn get_mask_adjustments_from_json(adj: &serde_json::Value) -> MaskAdjustments {
    if adj.is_null() {
        return MaskAdjustments::default();
    }

    let visibility = adj.get("sectionVisibility");
    let is_visible = |section: &str| -> bool {
        visibility
            .and_then(|v| v.get(section))
            .and_then(|s| s.as_bool())
            .unwrap_or(true)
    };

    let get_val = |section: &str, key: &str, scale: f32| -> f32 {
        if is_visible(section) {
            adj[key].as_f64().unwrap_or(0.0) as f32 / scale
        } else {
            0.0
        }
    };

    let curves_obj = adj.get("curves").cloned().unwrap_or_default();
    let luma_points: Vec<serde_json::Value> = if is_visible("curves") {
        curves_obj["luma"].as_array().cloned().unwrap_or_default()
    } else {
        Vec::new()
    };
    let red_points: Vec<serde_json::Value> = if is_visible("curves") {
        curves_obj["red"].as_array().cloned().unwrap_or_default()
    } else {
        Vec::new()
    };
    let green_points: Vec<serde_json::Value> = if is_visible("curves") {
        curves_obj["green"].as_array().cloned().unwrap_or_default()
    } else {
        Vec::new()
    };
    let blue_points: Vec<serde_json::Value> = if is_visible("curves") {
        curves_obj["blue"].as_array().cloned().unwrap_or_default()
    } else {
        Vec::new()
    };
    let cg_obj = adj.get("colorGrading").cloned().unwrap_or_default();

    MaskAdjustments {
        exposure: get_val("basic", "exposure", SCALES.exposure),
        brightness: get_val("basic", "brightness", SCALES.brightness),
        contrast: get_val("basic", "contrast", SCALES.contrast),
        highlights: get_val("basic", "highlights", SCALES.highlights),
        shadows: get_val("basic", "shadows", SCALES.shadows),
        whites: get_val("basic", "whites", SCALES.whites),
        blacks: get_val("basic", "blacks", SCALES.blacks),

        saturation: get_val("color", "saturation", SCALES.saturation),
        temperature: get_val("color", "temperature", SCALES.temperature),
        tint: get_val("color", "tint", SCALES.tint),
        vibrance: get_val("color", "vibrance", SCALES.vibrance),

        sharpness: get_val("details", "sharpness", SCALES.sharpness),
        luma_noise_reduction: get_val("details", "lumaNoiseReduction", SCALES.luma_noise_reduction),
        color_noise_reduction: get_val(
            "details",
            "colorNoiseReduction",
            SCALES.color_noise_reduction,
        ),

        clarity: get_val("details", "clarity", SCALES.clarity),
        dehaze: get_val("details", "dehaze", SCALES.dehaze),
        structure: get_val("details", "structure", SCALES.structure),

        glow_amount: get_val("effects", "glowAmount", SCALES.glow),
        halation_amount: get_val("effects", "halationAmount", SCALES.halation),
        flare_amount: get_val("effects", "flareAmount", SCALES.flares),
        sharpness_threshold: get_val("details", "sharpnessThreshold", SCALES.sharpness_threshold),

        hue: get_val("color", "hue", 1.0),
        _pad_cg1: 0.0,
        _pad_cg2: 0.0,
        color_grading_shadows: if is_visible("color") {
            parse_color_grade_settings(&cg_obj["shadows"])
        } else {
            ColorGradeSettings::default()
        },
        color_grading_midtones: if is_visible("color") {
            parse_color_grade_settings(&cg_obj["midtones"])
        } else {
            ColorGradeSettings::default()
        },
        color_grading_highlights: if is_visible("color") {
            parse_color_grade_settings(&cg_obj["highlights"])
        } else {
            ColorGradeSettings::default()
        },
        color_grading_global: if is_visible("color") {
            parse_color_grade_settings(&cg_obj["global"])
        } else {
            ColorGradeSettings::default()
        },
        color_grading_blending: if is_visible("color") {
            cg_obj["blending"].as_f64().unwrap_or(50.0) as f32 / SCALES.color_grading_blending
        } else {
            0.5
        },
        color_grading_balance: if is_visible("color") {
            cg_obj["balance"].as_f64().unwrap_or(0.0) as f32 / SCALES.color_grading_balance
        } else {
            0.0
        },
        _pad5: 0.0,
        _pad6: 0.0,

        hsl: if is_visible("color") {
            parse_hsl_adjustments(&adj.get("hsl").cloned().unwrap_or_default())
        } else {
            [HslColor::default(); 8]
        },
        luma_curve: convert_points_to_aligned(luma_points.clone()),
        red_curve: convert_points_to_aligned(red_points.clone()),
        green_curve: convert_points_to_aligned(green_points.clone()),
        blue_curve: convert_points_to_aligned(blue_points.clone()),
        luma_curve_count: luma_points.len() as u32,
        red_curve_count: red_points.len() as u32,
        green_curve_count: green_points.len() as u32,
        blue_curve_count: blue_points.len() as u32,
        _pad_end4: 0.0,
        _pad_end5: 0.0,
        _pad_end6: 0.0,
        _pad_end7: 0.0,
    }
}

pub fn get_all_adjustments_from_json(
    js_adjustments: &serde_json::Value,
    is_raw: bool,
    tonemapper_override: Option<u32>,
) -> AllAdjustments {
    let global = get_global_adjustments_from_json(js_adjustments, is_raw, tonemapper_override);
    let mut mask_adjustments = [MaskAdjustments::default(); MAX_MASKS];
    let mut mask_count = 0;

    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    for (i, mask_def) in mask_definitions
        .iter()
        .filter(|m| m.visible)
        .enumerate()
        .take(MAX_MASKS)
    {
        mask_adjustments[i] = get_mask_adjustments_from_json(&mask_def.adjustments);
        mask_count += 1;
    }

    AllAdjustments {
        global,
        mask_adjustments,
        mask_count,
        tile_offset_x: 0,
        tile_offset_y: 0,
        mask_atlas_cols: 1,
    }
}

#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub limits: wgpu::Limits,
    pub display: Arc<std::sync::Mutex<Option<WgpuDisplay>>>,
}

#[inline(always)]
fn rgb_to_yc_only(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168736 * r - 0.331264 * g + 0.5 * b;
    let cr = 0.5 * r - 0.418688 * g - 0.081312 * b;
    (y, cb, cr)
}

#[inline(always)]
fn yc_to_rgb(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;
    (r, g, b)
}

pub fn remove_raw_artifacts_and_enhance(
    image: &mut DynamicImage,
    color_nr_inv_sigma: f32,
    sharpening_amount: f32,
) {
    let mut buffer = image.to_rgb32f();
    let w = buffer.width() as usize;
    let h = buffer.height() as usize;

    let mut ycbcr_buffer = vec![0.0f32; w * h * 3];

    let src = buffer.as_raw();

    ycbcr_buffer
        .par_chunks_mut(3)
        .zip(src.par_chunks(3))
        .for_each(|(dest, pixel)| {
            let (y, cb, cr) = rgb_to_yc_only(pixel[0], pixel[1], pixel[2]);
            dest[0] = y;
            dest[1] = cb;
            dest[2] = cr;
        });

    if color_nr_inv_sigma > 0.0 {
        let base_inv_sigma = color_nr_inv_sigma;
        const OFFSETS: [isize; 3] = [-5, -1, 3];
        const OFFSET_SQUARES: [f32; 3] = [25.0, 1.0, 9.0];

        buffer
            .par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row)| {
                let row_offset = y * w;
                let h_isize = h as isize;
                let w_isize = w as isize;
                let y_isize = y as isize;

                for x in 0..w {
                    let center_idx = (row_offset + x) * 3;

                    let cy = ycbcr_buffer[center_idx];
                    let ccb = ycbcr_buffer[center_idx + 1];
                    let ccr = ycbcr_buffer[center_idx + 2];

                    let mut cb_sum = 0.0;
                    let mut cr_sum = 0.0;
                    let mut w_sum = 0.0;

                    for (ki, &ky) in OFFSETS.iter().enumerate() {
                        let sy = y_isize + ky;
                        if sy < 0 || sy >= h_isize {
                            continue;
                        }

                        let neighbor_row_idx = (sy as usize) * w;
                        let ky_sq_div_50 = OFFSET_SQUARES[ki] * 0.02;

                        for (kj, &kx) in OFFSETS.iter().enumerate() {
                            let sx = (x as isize) + kx;
                            if sx < 0 || sx >= w_isize {
                                continue;
                            }

                            let neighbor_idx = (neighbor_row_idx + sx as usize) * 3;

                            let neighbor_y = ycbcr_buffer[neighbor_idx];
                            let y_diff = (cy - neighbor_y).abs();

                            let val = y_diff * base_inv_sigma;
                            let spatial_penalty = OFFSET_SQUARES[kj] * 0.02 + ky_sq_div_50;

                            let weight = 1.0 / (1.0 + val * val + spatial_penalty);

                            cb_sum += ycbcr_buffer[neighbor_idx + 1] * weight;
                            cr_sum += ycbcr_buffer[neighbor_idx + 2] * weight;
                            w_sum += weight;
                        }
                    }

                    let (out_cb, out_cr) = if w_sum > 1e-4 {
                        let inv_w_sum = 1.0 / w_sum;
                        let filtered_cb = cb_sum * inv_w_sum;
                        let filtered_cr = cr_sum * inv_w_sum;

                        let orig_mag_sq = ccb * ccb + ccr * ccr;
                        let filt_mag_sq = filtered_cb * filtered_cb + filtered_cr * filtered_cr;

                        if filt_mag_sq > orig_mag_sq && orig_mag_sq > 1e-12 {
                            let scale = (orig_mag_sq / filt_mag_sq).sqrt();
                            (filtered_cb * scale, filtered_cr * scale)
                        } else {
                            (filtered_cb, filtered_cr)
                        }
                    } else {
                        (ccb, ccr)
                    };

                    let (r, g, b) = yc_to_rgb(cy, out_cb, out_cr);

                    let o = x * 3;
                    row[o] = r.clamp(0.0, 1.0);
                    row[o + 1] = g.clamp(0.0, 1.0);
                    row[o + 2] = b.clamp(0.0, 1.0);
                }
            });
    }

    if sharpening_amount > 0.0 {
        apply_gentle_detail_enhance(&mut buffer, &ycbcr_buffer, sharpening_amount);
    }

    *image = DynamicImage::ImageRgb32F(buffer);
}

fn apply_gentle_detail_enhance(
    buffer: &mut image::ImageBuffer<image::Rgb<f32>, Vec<f32>>,
    ycbcr_source: &[f32],
    amount: f32,
) {
    let w = buffer.width() as usize;
    let h = buffer.height() as usize;

    let mut temp_blur = vec![0.0; w * h];
    let radius = 2i32;

    temp_blur
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let row_offset = y * w;
            for (x, row_val) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                let mut count = 0;
                for kx in -radius..=radius {
                    let sx = (x as i32 + kx).clamp(0, (w as i32) - 1) as usize;
                    sum += ycbcr_source[(row_offset + sx) * 3];
                    count += 1;
                }
                *row_val = sum / count as f32;
            }
        });

    let output = buffer.as_mut();

    output
        .par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, rgb_row)| {
            for x in 0..w {
                let mut blur_sum = 0.0;
                let mut count = 0;
                for ky in -radius..=radius {
                    let sy = (y as i32 + ky).clamp(0, (h as i32) - 1) as usize;
                    blur_sum += temp_blur[sy * w + x];
                    count += 1;
                }
                let blurred_val = blur_sum / count as f32;

                let original_luma = ycbcr_source[(y * w + x) * 3];

                let detail = original_luma - blurred_val;

                let edge_strength = detail.abs();
                let adaptive_amount = if edge_strength > 0.1 {
                    amount * 0.3
                } else {
                    amount
                };
                let boost = detail * adaptive_amount;

                let r_idx = x * 3;
                let g_idx = r_idx + 1;
                let b_idx = r_idx + 2;

                let r = rgb_row[r_idx];
                let g = rgb_row[g_idx];
                let b = rgb_row[b_idx];

                let new_r = r + boost;
                let new_g = g + boost;
                let new_b = b + boost;

                let max_val = new_r.max(new_g).max(new_b);
                let min_val = new_r.min(new_g).min(new_b);

                let scale = if max_val > 1.0 || min_val < 0.0 {
                    if max_val > 1.0 && min_val < 0.0 {
                        0.0
                    } else if max_val > 1.0 {
                        (1.0 - r.max(g).max(b)) / boost.max(0.001)
                    } else {
                        r.min(g).min(b) / (-boost).max(0.001)
                    }
                } else {
                    1.0
                };

                let safe_boost = boost * scale.clamp(0.0, 1.0);

                rgb_row[r_idx] = (r + safe_boost).clamp(0.0, 1.0);
                rgb_row[g_idx] = (g + safe_boost).clamp(0.0, 1.0);
                rgb_row[b_idx] = (b + safe_boost).clamp(0.0, 1.0);
            }
        });
}

#[derive(Serialize, Clone)]
pub struct HistogramData {
    red: Vec<f32>,
    green: Vec<f32>,
    blue: Vec<f32>,
    luma: Vec<f32>,
}

pub fn calculate_histogram_from_image(image: &DynamicImage) -> Result<HistogramData, String> {
    let init_hist = || ([0u32; 256], [0u32; 256], [0u32; 256], [0u32; 256]);

    let reduce_hist = |mut a: ([u32; 256], [u32; 256], [u32; 256], [u32; 256]),
                       b: ([u32; 256], [u32; 256], [u32; 256], [u32; 256])| {
        for i in 0..256 {
            a.0[i] += b.0[i];
            a.1[i] += b.1[i];
            a.2[i] += b.2[i];
            a.3[i] += b.3[i];
        }
        a
    };

    let (r_c, g_c, b_c, l_c) = match image {
        DynamicImage::ImageRgb32F(f32_img) => {
            let raw = f32_img.as_raw();
            raw.par_chunks(30_000)
                .fold(init_hist, |mut acc, chunk| {
                    for pixel in chunk.chunks_exact(3).step_by(2) {
                        let r = (pixel[0].clamp(0.0, 1.0) * 255.0) as usize;
                        let g = (pixel[1].clamp(0.0, 1.0) * 255.0) as usize;
                        let b = (pixel[2].clamp(0.0, 1.0) * 255.0) as usize;

                        acc.0[r] += 1;
                        acc.1[g] += 1;
                        acc.2[b] += 1;

                        let luma = (r * 218 + g * 732 + b * 74) >> 10;
                        acc.3[luma.min(255)] += 1;
                    }
                    acc
                })
                .reduce(init_hist, reduce_hist)
        }
        _ => {
            let rgb = image.to_rgb8();
            let raw = rgb.as_raw();
            raw.par_chunks(30_000)
                .fold(init_hist, |mut acc, chunk| {
                    for pixel in chunk.chunks_exact(3).step_by(2) {
                        let r = pixel[0] as usize;
                        let g = pixel[1] as usize;
                        let b = pixel[2] as usize;

                        acc.0[r] += 1;
                        acc.1[g] += 1;
                        acc.2[b] += 1;

                        let luma = (r * 218 + g * 732 + b * 74) >> 10;
                        acc.3[luma.min(255)] += 1;
                    }
                    acc
                })
                .reduce(init_hist, reduce_hist)
        }
    };

    let mut red: Vec<f32> = r_c.into_iter().map(|c| c as f32).collect();
    let mut green: Vec<f32> = g_c.into_iter().map(|c| c as f32).collect();
    let mut blue: Vec<f32> = b_c.into_iter().map(|c| c as f32).collect();
    let mut luma: Vec<f32> = l_c.into_iter().map(|c| c as f32).collect();

    let smoothing_sigma = 2.0;
    apply_gaussian_smoothing(&mut red, smoothing_sigma);
    apply_gaussian_smoothing(&mut green, smoothing_sigma);
    apply_gaussian_smoothing(&mut blue, smoothing_sigma);
    apply_gaussian_smoothing(&mut luma, smoothing_sigma);

    normalize_histogram_range(&mut red, 0.99);
    normalize_histogram_range(&mut green, 0.99);
    normalize_histogram_range(&mut blue, 0.99);
    normalize_histogram_range(&mut luma, 0.99);

    Ok(HistogramData {
        red,
        green,
        blue,
        luma,
    })
}

fn apply_gaussian_smoothing(histogram: &mut [f32], sigma: f32) {
    if sigma <= 0.0 {
        return;
    }

    let kernel_radius = (sigma * 3.0).ceil() as usize;
    if kernel_radius == 0 || kernel_radius >= histogram.len() {
        return;
    }

    let kernel_size = 2 * kernel_radius + 1;
    let mut kernel = vec![0.0; kernel_size];
    let mut kernel_sum = 0.0;

    let two_sigma_sq = 2.0 * sigma * sigma;
    for (i, kernel_val) in kernel.iter_mut().enumerate() {
        let x = (i as i32 - kernel_radius as i32) as f32;
        let val = (-x * x / two_sigma_sq).exp();
        *kernel_val = val;
        kernel_sum += val;
    }

    if kernel_sum > 0.0 {
        for val in &mut kernel {
            *val /= kernel_sum;
        }
    }

    let original = histogram.to_owned();
    let len = histogram.len();

    for (i, hist_val) in histogram.iter_mut().enumerate() {
        let mut smoothed_val = 0.0;
        for (k, &kernel_val) in kernel.iter().enumerate() {
            let offset = k as i32 - kernel_radius as i32;
            let sample_index = i as i32 + offset;
            let clamped_index = sample_index.clamp(0, len as i32 - 1) as usize;
            smoothed_val += original[clamped_index] * kernel_val;
        }
        *hist_val = smoothed_val;
    }
}

fn normalize_histogram_range(histogram: &mut [f32], percentile_clip: f32) {
    if histogram.is_empty() {
        return;
    }

    let mut sorted_data = histogram.to_owned();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let clip_index = ((sorted_data.len() - 1) as f32 * percentile_clip).round() as usize;
    let max_val = sorted_data[clip_index.min(sorted_data.len() - 1)];

    if max_val > 1e-6 {
        let scale_factor = 1.0 / max_val;
        for value in histogram.iter_mut() {
            *value = (*value * scale_factor).min(1.0);
        }
    } else {
        for value in histogram.iter_mut() {
            *value = 0.0;
        }
    }
}

#[derive(serde::Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WaveformData {
    pub rgb: String,
    pub luma: String,
    pub parade: String,
    pub vectorscope: String,
    pub width: u32,
    pub height: u32,
}

pub fn calculate_waveform_from_image(
    image: &DynamicImage,
    active_channel: Option<&str>,
) -> Result<WaveformData, String> {
    const W: usize = 256;
    const H: usize = 256;

    let (orig_w, orig_h) = image.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return Err("Image has zero dimensions.".to_string());
    }

    let do_rgb = active_channel.is_none() || active_channel == Some("rgb");
    let do_luma =
        active_channel.is_none() || active_channel == Some("luma") || active_channel == Some("rgb");
    let do_parade = active_channel.is_none() || active_channel == Some("parade");
    let do_vectorscope = active_channel.is_none() || active_channel == Some("vectorscope");

    let mut red_bins = if do_rgb { vec![0u32; W * H] } else { vec![] };
    let mut green_bins = if do_rgb { vec![0u32; W * H] } else { vec![] };
    let mut blue_bins = if do_rgb { vec![0u32; W * H] } else { vec![] };
    let mut luma_bins = if do_luma { vec![0u32; W * H] } else { vec![] };
    let mut parade_bins = if do_parade { vec![0u32; W * H] } else { vec![] };
    let mut vector_bins = if do_vectorscope {
        vec![0u32; W * H]
    } else {
        vec![]
    };

    let x_scale = W as f32 / orig_w as f32;
    let mut x_buckets = vec![0usize; orig_w as usize];

    let mut x_buckets_parade_r = vec![0usize; orig_w as usize];
    let mut x_buckets_parade_g = vec![0usize; orig_w as usize];
    let mut x_buckets_parade_b = vec![0usize; orig_w as usize];

    for x in 0..(orig_w as usize) {
        x_buckets[x] = ((x as f32 * x_scale) as usize).min(W - 1);
        if do_parade {
            let relative_x = x as f32 / orig_w as f32;
            x_buckets_parade_r[x] = (relative_x * 82.0) as usize % 82;
            x_buckets_parade_g[x] = 87 + (relative_x * 82.0) as usize % 82;
            x_buckets_parade_b[x] = 174 + (relative_x * 82.0) as usize % 82;
        }
    }

    let mut process_pixel = |r: u8, g: u8, b: u8, out_x: usize, orig_x: usize| {
        if do_rgb {
            red_bins[(255 - r as usize) * W + out_x] += 1;
            green_bins[(255 - g as usize) * W + out_x] += 1;
            blue_bins[(255 - b as usize) * W + out_x] += 1;
        }
        if do_luma {
            let l = ((r as u32 * 218 + g as u32 * 732 + b as u32 * 74) >> 10).min(255) as usize;
            luma_bins[(255 - l) * W + out_x] += 1;
        }
        if do_parade {
            parade_bins[(255 - r as usize) * W + x_buckets_parade_r[orig_x]] += 1;
            parade_bins[(255 - g as usize) * W + x_buckets_parade_g[orig_x]] += 1;
            parade_bins[(255 - b as usize) * W + x_buckets_parade_b[orig_x]] += 1;
        }
        if do_vectorscope {
            let r_f = r as f32;
            let g_f = g as f32;
            let b_f = b as f32;

            let mut cb = (-0.1146 * r_f - 0.3854 * g_f + 0.5 * b_f) * 0.836;
            let mut cr = (0.5 * r_f - 0.4542 * g_f - 0.0458 * b_f) * 0.836;

            let dist_sq = cb * cb + cr * cr;
            if dist_sq > 16129.0 {
                let scale = 127.0 / dist_sq.sqrt();
                cb *= scale;
                cr *= scale;
            }

            let vx = (cb + 128.0).clamp(0.0, 255.0) as usize;
            let vy = (128.0 - cr).clamp(0.0, 255.0) as usize;
            vector_bins[vy * W + vx] += 1;
        }
    };

    match image {
        DynamicImage::ImageRgb32F(f32_img) => {
            let raw = f32_img.as_raw();
            let stride = orig_w as usize * 3;
            for y in 0..(orig_h as usize) {
                let row = y * stride;
                for (x, &x_bucket) in x_buckets.iter().enumerate() {
                    let i = row + x * 3;
                    process_pixel(
                        (raw[i].clamp(0.0, 1.0) * 255.0) as u8,
                        (raw[i + 1].clamp(0.0, 1.0) * 255.0) as u8,
                        (raw[i + 2].clamp(0.0, 1.0) * 255.0) as u8,
                        x_bucket,
                        x,
                    );
                }
            }
        }
        _ => {
            let rgb = image.to_rgb8();
            let raw = rgb.as_raw();
            let stride = orig_w as usize * 3;
            for y in 0..(orig_h as usize) {
                let row = y * stride;
                for (x, &x_bucket) in x_buckets.iter().enumerate() {
                    let i = row + x * 3;
                    process_pixel(raw[i], raw[i + 1], raw[i + 2], x_bucket, x);
                }
            }
        }
    }

    let build_lut = |bins: &[u32], do_calc: bool| -> (Vec<u8>, u32) {
        if !do_calc {
            return (vec![0; 1], 0);
        }
        let max_val = *bins.iter().max().unwrap_or(&0);
        if max_val == 0 {
            return (vec![0; 1], 0);
        }
        let scale = 255.0 / (1.0 + max_val as f32).ln();
        let lut = (0..=max_val)
            .map(|v| {
                if v == 0 {
                    0
                } else {
                    ((1.0 + v as f32).ln() * scale) as u8
                }
            })
            .collect();
        (lut, max_val)
    };

    let (lut_r, max_r) = build_lut(&red_bins, do_rgb);
    let (lut_g, max_g) = build_lut(&green_bins, do_rgb);
    let (lut_b, max_b) = build_lut(&blue_bins, do_rgb);
    let (lut_l, max_l) = build_lut(&luma_bins, do_luma);
    let (lut_p, max_p) = build_lut(&parade_bins, do_parade);
    let (lut_v, max_v) = build_lut(&vector_bins, do_vectorscope);

    let pixel_count = W * H;
    let byte_count = pixel_count * 4;

    let mut rgba_rgb = if do_rgb {
        vec![0u8; byte_count]
    } else {
        vec![]
    };
    let mut rgba_luma = if do_luma {
        vec![0u8; byte_count]
    } else {
        vec![]
    };
    let mut rgba_parade = if do_parade {
        vec![0u8; byte_count]
    } else {
        vec![]
    };
    let mut rgba_vector = if do_vectorscope {
        vec![0u8; byte_count]
    } else {
        vec![]
    };

    for i in 0..pixel_count {
        let x = i % W;
        let y = i / W;
        let off = i * 4;

        if do_rgb {
            let r = if red_bins[i] <= max_r {
                lut_r[red_bins[i] as usize]
            } else {
                0
            };
            let g = if green_bins[i] <= max_g {
                lut_g[green_bins[i] as usize]
            } else {
                0
            };
            let b = if blue_bins[i] <= max_b {
                lut_b[blue_bins[i] as usize]
            } else {
                0
            };
            if r > 0 || g > 0 || b > 0 {
                rgba_rgb[off] = r;
                rgba_rgb[off + 1] = g;
                rgba_rgb[off + 2] = b;
                rgba_rgb[off + 3] = r.max(g).max(b);
            }
        }

        if do_luma && luma_bins[i] > 0 && luma_bins[i] <= max_l {
            let l = lut_l[luma_bins[i] as usize];
            rgba_luma[off] = 255;
            rgba_luma[off + 1] = 255;
            rgba_luma[off + 2] = 255;
            rgba_luma[off + 3] = l;
        }

        if do_parade && parade_bins[i] > 0 && parade_bins[i] <= max_p {
            let bright = lut_p[parade_bins[i] as usize];
            if x < 82 {
                rgba_parade[off] = 255;
                rgba_parade[off + 3] = bright;
            } else if (87..169).contains(&x) {
                rgba_parade[off + 1] = 255;
                rgba_parade[off + 3] = bright;
            } else if x >= 174 {
                rgba_parade[off + 2] = 255;
                rgba_parade[off + 3] = bright;
            }
        }

        if do_vectorscope {
            let val = vector_bins[i];

            let dx = x as f32 - 128.0;
            let dy = 128.0 - y as f32;
            let min_d = dx.abs().min(dy.abs());
            let dist = (dx * dx + dy * dy).sqrt();

            if val > 0 && val <= max_v {
                let bright = lut_v[val as usize];

                let y_mid = 128.0;
                rgba_vector[off] = (y_mid + 1.402 * (dy / 0.836)).clamp(0.0, 255.0) as u8;
                rgba_vector[off + 1] = (y_mid - 0.344136 * (dx / 0.836) - 0.714136 * (dy / 0.836))
                    .clamp(0.0, 255.0) as u8;
                rgba_vector[off + 2] = (y_mid + 1.772 * (dx / 0.836)).clamp(0.0, 255.0) as u8;
                rgba_vector[off + 3] = bright;
            } else if min_d <= 1.0 {
                let alpha = (40.0 - min_d * 30.0).clamp(0.0, 255.0) as u8;
                rgba_vector[off] = 255;
                rgba_vector[off + 1] = 255;
                rgba_vector[off + 2] = 255;
                rgba_vector[off + 3] = alpha;
            } else if (dist - 127.0).abs() < 0.8 || (dist - 64.0).abs() < 0.8 {
                rgba_vector[off] = 255;
                rgba_vector[off + 1] = 255;
                rgba_vector[off + 2] = 255;
                rgba_vector[off + 3] = 15;
            } else if dx < 0.0 && dy > 0.0 && (dy + 1.53 * dx).abs() < 1.0 {
                rgba_vector[off] = 255;
                rgba_vector[off + 1] = 200;
                rgba_vector[off + 2] = 150;
                rgba_vector[off + 3] = 120;
            }
        }
    }

    Ok(WaveformData {
        rgb: if do_rgb {
            BASE64.encode(&rgba_rgb)
        } else {
            String::new()
        },
        luma: if do_luma {
            BASE64.encode(&rgba_luma)
        } else {
            String::new()
        },
        parade: if do_parade {
            BASE64.encode(&rgba_parade)
        } else {
            String::new()
        },
        vectorscope: if do_vectorscope {
            BASE64.encode(&rgba_vector)
        } else {
            String::new()
        },
        width: W as u32,
        height: H as u32,
    })
}

pub fn perform_auto_analysis(image: &DynamicImage) -> AutoAdjustmentResults {
    const ANALYSIS_MAX_DIM: u32 = 1024;

    const LUMA_R: f32 = 0.2126;
    const LUMA_G: f32 = 0.7152;
    const LUMA_B: f32 = 0.0722;

    const EXPOSURE_MIDPOINT: f64 = 128.0;
    const EXPOSURE_SCALE: f64 = 0.125;
    const WHITE_POINT_HARD_LIMIT: usize = 245;
    const HIGHLIGHT_LUMA_THRESHOLD: usize = 240;
    const CLIPPED_LUMA_THRESHOLD: usize = 250;
    const HIGHLIGHT_PERCENT_THRESHOLD: f64 = 0.02;
    const CLIPPED_PERCENT_THRESHOLD: f64 = 0.005;
    const EXPOSURE_CEILING: f64 = 250.0;

    const TARGET_RANGE: f64 = 220.0;
    const CONTRAST_SCALE: f64 = 10.0;
    const HIGHLIGHT_CONTRAST_REDUCE: f64 = 0.5;

    const SHADOW_LUMA_MAX: usize = 32;
    const SHADOW_PERCENT_THRESHOLD: f64 = 0.05;
    const SHADOW_BOOST_SCALE: f64 = 40.0;
    const SHADOW_MAX: f64 = 50.0;
    const HIGHLIGHT_BOOST_SCALE: f64 = 120.0;
    const HIGHLIGHT_MAX: f64 = 70.0;

    const VIBRANCY_SAT_THRESHOLD: f32 = 0.2;
    const VIBRANCY_SCALE: f64 = 120.0;

    const DEHAZE_RANGE_THRESHOLD: f64 = 120.0;
    const DEHAZE_SAT_THRESHOLD: f32 = 0.15;
    const DEHAZE_SCALE: f64 = 35.0;
    const CLARITY_RANGE_THRESHOLD: f64 = 180.0;
    const CLARITY_SCALE: f64 = 50.0;

    const VIGNETTE_CENTER_LOW: f32 = 0.25;
    const VIGNETTE_CENTER_HIGH: f32 = 0.75;

    const VIGNETTE_SCALE: f64 = 100.0;
    const VIGNETTE_CENTRE_DIFF_THRESHOLD: f32 = 0.05;
    const CENTRE_SCALE: f64 = 100.0;
    const CENTRE_MAX: f64 = 60.0;

    const MID_GRAY: f64 = 128.0;
    const BLACKS_SCALE: f64 = 0.5;
    const WHITES_SCALE: f64 = 0.2;
    const EXPOSURE_OUTPUT_SCALE: f64 = 20.0;
    const BRIGHTNESS_SCALE: f64 = 0.007;

    let analysis_preview = downscale_f32_image(image, ANALYSIS_MAX_DIM, ANALYSIS_MAX_DIM);
    let rgb_image = analysis_preview.to_rgb8();
    let (auto_temperature, auto_tint) = estimate_auto_white_balance(&rgb_image);
    let total_pixels = (rgb_image.width() * rgb_image.height()) as f64;

    let (width, height) = rgb_image.dimensions();
    let cx0 = (width as f32 * VIGNETTE_CENTER_LOW) as u32;
    let cx1 = (width as f32 * VIGNETTE_CENTER_HIGH) as u32;
    let cy0 = (height as f32 * VIGNETTE_CENTER_LOW) as u32;
    let cy1 = (height as f32 * VIGNETTE_CENTER_HIGH) as u32;

    let mut luma_hist = vec![0u32; 256];
    let mut mean_saturation = 0.0f32;
    let mut center_sum = 0.0f32;
    let mut edge_sum = 0.0f32;
    let mut center_n = 0u32;
    let mut edge_n = 0u32;

    for (x, y, pixel) in rgb_image.enumerate_pixels() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        let luma_f = LUMA_R * r + LUMA_G * g + LUMA_B * b;
        luma_hist[(luma_f.round() as usize).min(255)] += 1;

        let r_n = r / 255.0;
        let g_n = g / 255.0;
        let b_n = b / 255.0;
        let max_c = r_n.max(g_n).max(b_n);
        let min_c = r_n.min(g_n).min(b_n);
        if max_c > 0.0 {
            let s = (max_c - min_c) / max_c;
            mean_saturation += s;
        }

        let luma_norm = luma_f / 255.0;
        if x >= cx0 && x < cx1 && y >= cy0 && y < cy1 {
            center_sum += luma_norm;
            center_n += 1;
        } else {
            edge_sum += luma_norm;
            edge_n += 1;
        }
    }

    mean_saturation /= total_pixels as f32;

    let percentile = |hist: &Vec<u32>, p: f64| -> usize {
        let target = (total_pixels * p) as u32;
        let mut cumulative = 0u32;
        for (i, &v) in hist.iter().enumerate() {
            cumulative += v;
            if cumulative >= target {
                return i;
            }
        }
        255
    };

    let p1 = percentile(&luma_hist, 0.01);
    let p50 = percentile(&luma_hist, 0.50);
    let p99 = percentile(&luma_hist, 0.99);

    let black_point = p1;
    let white_point = p99;
    let range = (white_point as f64 - black_point as f64).max(1.0);

    let highlight_percent =
        luma_hist[HIGHLIGHT_LUMA_THRESHOLD..256].iter().sum::<u32>() as f64 / total_pixels;
    let clipped_percent =
        luma_hist[CLIPPED_LUMA_THRESHOLD..256].iter().sum::<u32>() as f64 / total_pixels;

    let mut exposure = (EXPOSURE_MIDPOINT - p50 as f64) * EXPOSURE_SCALE;

    if white_point > WHITE_POINT_HARD_LIMIT
        || highlight_percent > HIGHLIGHT_PERCENT_THRESHOLD
        || clipped_percent > CLIPPED_PERCENT_THRESHOLD
    {
        exposure = exposure.min(0.0);
    }

    if white_point as f64 + exposure > EXPOSURE_CEILING {
        exposure = EXPOSURE_CEILING - white_point as f64;
    }

    let mut contrast = 0.0f64;
    if range < TARGET_RANGE {
        contrast = ((TARGET_RANGE / range) - 1.0) * CONTRAST_SCALE;
    }
    if highlight_percent > HIGHLIGHT_PERCENT_THRESHOLD {
        contrast *= HIGHLIGHT_CONTRAST_REDUCE;
    }

    let shadow_percent = luma_hist[0..SHADOW_LUMA_MAX].iter().sum::<u32>() as f64 / total_pixels;

    let mut shadows = 0.0f64;
    if shadow_percent > SHADOW_PERCENT_THRESHOLD {
        shadows = (shadow_percent * SHADOW_BOOST_SCALE).min(SHADOW_MAX);
    }

    let mut highlights = 0.0f64;
    if highlight_percent > HIGHLIGHT_PERCENT_THRESHOLD {
        highlights = -(highlight_percent * HIGHLIGHT_BOOST_SCALE).min(HIGHLIGHT_MAX);
    }

    let mut vibrancy = 0.0f64;
    if mean_saturation < VIBRANCY_SAT_THRESHOLD {
        vibrancy = (VIBRANCY_SAT_THRESHOLD - mean_saturation) as f64 * VIBRANCY_SCALE;
    }

    let mut dehaze = 0.0f64;
    if range < DEHAZE_RANGE_THRESHOLD && mean_saturation < DEHAZE_SAT_THRESHOLD {
        dehaze = (1.0 - range / DEHAZE_RANGE_THRESHOLD) * DEHAZE_SCALE;
    }

    let mut clarity = 0.0f64;
    if range < CLARITY_RANGE_THRESHOLD {
        clarity = (1.0 - range / CLARITY_RANGE_THRESHOLD) * CLARITY_SCALE;
    }

    let mut vignette_amount = 0.0f64;
    let mut centre = 0.0f64;

    if center_n > 0 && edge_n > 0 {
        let c_avg = center_sum / center_n as f32;
        let e_avg = edge_sum / edge_n as f32;

        if e_avg < c_avg {
            let diff = c_avg - e_avg;
            vignette_amount = -(diff as f64 * VIGNETTE_SCALE);

            if diff > VIGNETTE_CENTRE_DIFF_THRESHOLD {
                centre = (diff as f64 * CENTRE_SCALE).min(CENTRE_MAX);
            }
        }
    }

    let mut adjusted_luma_hist = vec![0u32; 256];
    for pixel in rgb_image.pixels() {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;
        let mut luma = LUMA_R as f64 * r + LUMA_G as f64 * g + LUMA_B as f64 * b;
        luma += exposure;
        luma = (luma - MID_GRAY) * (1.0 + contrast / 100.0) + MID_GRAY;
        adjusted_luma_hist[luma.clamp(0.0, 255.0).round() as usize] += 1;
    }

    let adj_p1 = percentile(&adjusted_luma_hist, 0.01);
    let adj_p50 = percentile(&adjusted_luma_hist, 0.50);
    let adj_p99 = percentile(&adjusted_luma_hist, 0.99);
    let blacks: f64 = -(adj_p1 as f64 * BLACKS_SCALE);
    let whites: f64 = (adj_p99 as f64 - 255.0) * WHITES_SCALE;
    let brightness: f64 = (MID_GRAY - adj_p50 as f64) * BRIGHTNESS_SCALE;

    AutoAdjustmentResults {
        exposure: (exposure / EXPOSURE_OUTPUT_SCALE).clamp(-5.0, 5.0),
        brightness: brightness.clamp(-5.0, 5.0),
        contrast: contrast.clamp(-100.0, 100.0),
        highlights: highlights.clamp(-100.0, 100.0),
        shadows: shadows.clamp(-100.0, 100.0),
        vibrancy: vibrancy.clamp(-100.0, 100.0),
        vignette_amount: vignette_amount.clamp(-100.0, 100.0),
        temperature: auto_temperature,
        tint: auto_tint,
        dehaze: dehaze.clamp(-100.0, 100.0),
        clarity: clarity.clamp(-100.0, 100.0),
        centre: centre.clamp(-100.0, 100.0),
        whites: whites.clamp(-100.0, 100.0),
        blacks: blacks.clamp(-100.0, 100.0),
    }
}

pub fn auto_results_to_json(results: &AutoAdjustmentResults) -> serde_json::Value {
    json!({
        "exposure": results.exposure,
        "brightness": results.brightness,
        "contrast": results.contrast,
        "highlights": results.highlights,
        "shadows": results.shadows,
        "vibrance": results.vibrancy,
        "vignetteAmount": results.vignette_amount,
        "temperature": results.temperature,
        "tint": results.tint,
        "whiteBalanceMode": "auto",
        "clarity": results.clarity,
        "centré": results.centre,

        "dehaze": results.dehaze,
        "sectionVisibility": {
            "basic": true,
            "color": true,
            "effects": true
        },
        "whites": results.whites,
        "blacks": results.blacks
    })
}

fn white_balance_multipliers(temperature: f64, tint: f64) -> (f64, f64, f64) {
    let normalized_temperature = temperature / 25.0;
    let normalized_tint = tint / 100.0;
    let temperature_rgb = (
        1.0 + normalized_temperature * 0.2,
        1.0 + normalized_temperature * 0.05,
        1.0 - normalized_temperature * 0.2,
    );
    let tint_rgb = (
        1.0 + normalized_tint * 0.25,
        1.0 - normalized_tint * 0.25,
        1.0 + normalized_tint * 0.25,
    );
    (
        temperature_rgb.0 * tint_rgb.0,
        temperature_rgb.1 * tint_rgb.1,
        temperature_rgb.2 * tint_rgb.2,
    )
}

fn estimate_auto_white_balance(image: &RgbImage) -> (f64, f64) {
    const SATURATION_BINS: usize = 256;
    const MIN_LUMA: f64 = 0.08;
    const MAX_CHANNEL: f64 = 0.98;
    const MAX_NEUTRAL_SATURATION: f64 = 0.72;

    let mut saturation_histogram = [0usize; SATURATION_BINS];
    let mut eligible_pixels = 0usize;

    for pixel in image.pixels() {
        let r = pixel[0] as f64 / 255.0;
        let g = pixel[1] as f64 / 255.0;
        let b = pixel[2] as f64 / 255.0;
        let max_channel = r.max(g).max(b);
        let min_channel = r.min(g).min(b);
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if luma < MIN_LUMA || max_channel > MAX_CHANNEL || min_channel <= 0.0 {
            continue;
        }

        let saturation = (max_channel - min_channel) / max_channel.max(1.0e-6);
        let bin = (saturation * (SATURATION_BINS - 1) as f64).round() as usize;
        saturation_histogram[bin.min(SATURATION_BINS - 1)] += 1;
        eligible_pixels += 1;
    }

    if eligible_pixels < 16 {
        return (0.0, 0.0);
    }

    let neutral_target = (eligible_pixels as f64 * 0.35).ceil() as usize;
    let mut neutral_count = 0usize;
    let mut saturation_threshold = MAX_NEUTRAL_SATURATION;
    for (bin, count) in saturation_histogram.iter().enumerate() {
        neutral_count += count;
        if neutral_count >= neutral_target {
            saturation_threshold =
                (bin as f64 / (SATURATION_BINS - 1) as f64).clamp(0.08, MAX_NEUTRAL_SATURATION);
            break;
        }
    }

    let mut red_ratio_log = 0.0f64;
    let mut blue_ratio_log = 0.0f64;
    let mut total_weight = 0.0f64;

    for pixel in image.pixels() {
        let r = pixel[0] as f64 / 255.0;
        let g = pixel[1] as f64 / 255.0;
        let b = pixel[2] as f64 / 255.0;
        let max_channel = r.max(g).max(b);
        let min_channel = r.min(g).min(b);
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if luma < MIN_LUMA || max_channel > MAX_CHANNEL || min_channel <= 0.0 {
            continue;
        }

        let saturation = (max_channel - min_channel) / max_channel.max(1.0e-6);
        if saturation > saturation_threshold {
            continue;
        }

        let neutral_weight = (1.0 - saturation).powi(3);
        let exposure_weight = (luma / 0.6).clamp(0.2, 1.0);
        let weight = neutral_weight * exposure_weight;
        red_ratio_log += (g / r.max(1.0 / 255.0)).ln() * weight;
        blue_ratio_log += (g / b.max(1.0 / 255.0)).ln() * weight;
        total_weight += weight;
    }

    if total_weight <= 1.0e-6 {
        return (0.0, 0.0);
    }

    let target_red_log = (red_ratio_log / total_weight).clamp(-1.0, 1.0);
    let target_blue_log = (blue_ratio_log / total_weight).clamp(-1.0, 1.0);
    let mut best_temperature = 0.0f64;
    let mut best_tint = 0.0f64;
    let mut best_score = f64::INFINITY;

    for temperature in -85..=85 {
        for tint in -85..=85 {
            let (red_gain, green_gain, blue_gain) =
                white_balance_multipliers(temperature as f64, tint as f64);
            let red_log = (red_gain / green_gain).ln();
            let blue_log = (blue_gain / green_gain).ln();
            let score = (red_log - target_red_log).powi(2) + (blue_log - target_blue_log).powi(2);
            if score < best_score {
                best_score = score;
                best_temperature = temperature as f64;
                best_tint = tint as f64;
            }
        }
    }

    (best_temperature, best_tint)
}

#[tauri::command]
pub fn calculate_auto_adjustments(
    state: tauri::State<AppState>,
) -> Result<serde_json::Value, String> {
    let original_image = state
        .original_image
        .lock()
        .unwrap()
        .as_ref()
        .ok_or("No image loaded for auto adjustments")?
        .image
        .clone();

    let results = perform_auto_analysis(&original_image);

    Ok(auto_results_to_json(&results))
}

#[cfg(test)]
mod geometry_transform_tests {
    use super::*;

    fn coordinate_gradient(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(width, height, |x, y| {
            image::Rgb([
                x as f32 / (width - 1) as f32,
                y as f32 / (height - 1) as f32,
                ((x * 7 + y * 11) % 31) as f32 / 30.0,
            ])
        }))
    }

    fn absolute_difference(left: &DynamicImage, right: &DynamicImage) -> f64 {
        let left = left.to_rgb32f();
        let right = right.to_rgb32f();
        left.as_raw()
            .iter()
            .zip(right.as_raw())
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .sum()
    }

    #[test]
    fn every_geometry_menu_slider_changes_native_pixels() {
        let source = coordinate_gradient(161, 121);
        let mut cases = Vec::new();

        let mut projection = GeometryParams::default();
        projection.projection = 70.0;
        cases.push(("projection", projection));

        let mut vertical = GeometryParams::default();
        vertical.vertical = 45.0;
        cases.push(("vertical", vertical));

        let mut horizontal = GeometryParams::default();
        horizontal.horizontal = -45.0;
        cases.push(("horizontal", horizontal));

        let mut rotate = GeometryParams::default();
        rotate.rotate = 12.0;
        cases.push(("rotate", rotate));

        let mut aspect = GeometryParams::default();
        aspect.aspect = 25.0;
        cases.push(("aspect", aspect));

        let mut scale = GeometryParams::default();
        scale.scale = 82.0;
        cases.push(("scale", scale));

        let mut x_offset = GeometryParams::default();
        x_offset.x_offset = 12.0;
        cases.push(("x offset", x_offset));

        let mut y_offset = GeometryParams::default();
        y_offset.y_offset = -12.0;
        cases.push(("y offset", y_offset));

        for (name, params) in cases {
            let transformed = warp_image_geometry(&source, params);
            assert!(
                absolute_difference(&source, &transformed) > 100.0,
                "{name} unexpectedly produced an identity image"
            );
        }
    }

    #[test]
    fn preview_sampling_and_inverse_coordinates_share_the_same_geometry() {
        let width = 181;
        let height = 137;
        let source = coordinate_gradient(width, height);
        let adjustments = serde_json::json!({
            "transformProjection": 42.0,
            "transformVertical": 28.0,
            "transformHorizontal": -19.0,
            "transformRotate": 4.5,
            "transformAspect": 8.0,
            "transformScale": 96.0,
            "transformXOffset": 2.0,
            "transformYOffset": -3.0,
            "sectionVisibility": { "geometry": true, "optics": true }
        });
        let transformed = apply_geometry_warp(&source, &adjustments).into_owned();
        let target_x = 91;
        let target_y = 68;
        let (source_x, source_y) = inverse_transform_point(
            target_x as f64,
            target_y as f64,
            width as f64,
            height as f64,
            &adjustments,
        );
        let pixel = transformed.to_rgb32f().get_pixel(target_x, target_y).0;

        assert!((pixel[0] as f64 - source_x / (width - 1) as f64).abs() < 0.015);
        assert!((pixel[1] as f64 - source_y / (height - 1) as f64).abs() < 0.015);
    }

    #[test]
    fn projection_forward_and_inverse_round_trip() {
        let (cx, cy, half_diagonal) = (500.0, 400.0, 640.312_423_7);
        for projection in [-100.0, -35.0, 35.0, 100.0] {
            let projected = project_geometry_point(820.0, 145.0, cx, cy, half_diagonal, projection);
            let restored = unproject_geometry_point(
                projected.0,
                projected.1,
                cx,
                cy,
                half_diagonal,
                projection,
            );
            assert!((restored.0 - 820.0).abs() < 1e-6);
            assert!((restored.1 - 145.0).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
mod color_transfer_tests {
    use super::{apply_linear_to_srgb, apply_srgb_to_linear};
    use image::{DynamicImage, ImageBuffer, Rgba};

    #[test]
    fn float_rgba_transfer_round_trips_rgb_and_preserves_alpha() {
        let original = Rgba([0.0, 0.040_45, 0.5, 0.37]);
        let encoded = DynamicImage::ImageRgba32F(ImageBuffer::from_pixel(1, 1, original));

        let linear = apply_srgb_to_linear(encoded);
        let linear_pixel = *linear
            .as_rgba32f()
            .expect("sRGB decode keeps RGBA32F storage")
            .get_pixel(0, 0);
        assert!((linear_pixel[2] - 0.214_041_14).abs() <= 1.0e-7);
        assert_eq!(linear_pixel[3], original[3]);

        let round_trip = apply_linear_to_srgb(linear);
        let round_trip_pixel = round_trip
            .as_rgba32f()
            .expect("sRGB encode keeps RGBA32F storage")
            .get_pixel(0, 0);
        for channel in 0..3 {
            assert!((round_trip_pixel[channel] - original[channel]).abs() <= 2.0e-6);
        }
        assert_eq!(round_trip_pixel[3], original[3]);
    }
}

#[cfg(test)]
mod auto_white_balance_tests {
    use super::*;
    use image::Rgb;

    fn corrected_spread(pixel: Rgb<u8>, temperature: f64, tint: f64) -> f64 {
        let (red_gain, green_gain, blue_gain) = white_balance_multipliers(temperature, tint);
        let corrected = [
            pixel[0] as f64 * red_gain,
            pixel[1] as f64 * green_gain,
            pixel[2] as f64 * blue_gain,
        ];
        corrected.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - corrected.iter().copied().fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn neutral_image_keeps_neutral_white_balance() {
        let image = RgbImage::from_pixel(32, 32, Rgb([128, 128, 128]));
        assert_eq!(estimate_auto_white_balance(&image), (0.0, 0.0));
    }

    #[test]
    fn warm_cast_is_cooled_toward_neutral() {
        let cast = Rgb([190, 135, 85]);
        let image = RgbImage::from_pixel(32, 32, cast);
        let (temperature, tint) = estimate_auto_white_balance(&image);
        let original_spread = (cast[0] - cast[2]) as f64;

        assert!(temperature < -10.0);
        assert!(corrected_spread(cast, temperature, tint) < original_spread * 0.35);
    }

    #[test]
    fn green_cast_adds_magenta_tint() {
        let cast = Rgb([110, 170, 110]);
        let image = RgbImage::from_pixel(32, 32, cast);
        let (temperature, tint) = estimate_auto_white_balance(&image);
        let original_spread = (cast[1] - cast[0]) as f64;

        assert!(tint > 10.0);
        assert!(corrected_spread(cast, temperature, tint) < original_spread * 0.35);
    }
}
