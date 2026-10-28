//! AI sub-masks from stored mask bitmaps.
//!
//! Decodes the base64 mask, maps it through the image transform and applies the
//! depth band-pass, grow and feather; also provides the tiled AI rasterizer.

use crate::ai_processing::AiDepthMaskParameters;
use crate::ai_processing::AiForegroundMaskParameters;
use crate::ai_processing::AiSkyMaskParameters;
use crate::ai_processing::AiSubjectMaskParameters;
use base64::Engine as _;
use base64::engine::general_purpose;
use image::GrayImage;
use serde_json::Value;

use super::{GrowFeatherParameters, GrowFeatherPlan, SubMask, apply_grow_and_feather};

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

pub(super) fn generate_ai_bitmap_from_full_mask(
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

pub(super) fn decode_ai_bitmap_from_base64(data_url: &str) -> Option<GrayImage> {
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
pub(super) struct AiMaskTransform {
    pub(super) rotation: f32,
    pub(super) flip_horizontal: bool,
    pub(super) flip_vertical: bool,
    pub(super) orientation_steps: u8,
}

impl AiMaskTransform {
    pub(super) fn from_options(
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
    pub(super) fn rasterize(
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
pub(super) struct AiDepthSelection {
    pub(super) min_depth: f32,
    pub(super) max_depth: f32,
    pub(super) min_fade: f32,
    pub(super) max_fade: f32,
    pub(super) feather_sigma: f32,
}

impl AiDepthSelection {
    pub(super) fn new(
        min_depth: f32,
        max_depth: f32,
        min_fade: f32,
        max_fade: f32,
        feather: f32,
    ) -> Self {
        Self {
            min_depth,
            max_depth,
            min_fade,
            max_fade,
            feather_sigma: if feather > 0.0 { feather * 0.1 } else { 0.0 },
        }
    }

    pub(super) fn blur_radius(self) -> u32 {
        image_gaussian_blur_radius(self.feather_sigma)
    }

    pub(super) fn apply_pointwise(self, mask: &mut GrayImage) {
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

    pub(super) fn apply_blur(self, mask: &mut GrayImage) {
        if self.feather_sigma > 0.0 {
            *mask = image::imageops::blur(mask, self.feather_sigma);
        }
    }
}

pub(super) fn image_gaussian_blur_radius(sigma: f32) -> u32 {
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
pub(super) enum AiMaskPostprocess {
    Binary,
    Depth(AiDepthSelection),
}

pub(super) struct TiledAiMaskRasterizer {
    pub(super) full_mask_image: GrayImage,
    pub(super) transform: AiMaskTransform,
    pub(super) grow: f32,
    pub(super) feather: f32,
    pub(super) postprocess: AiMaskPostprocess,
}

impl TiledAiMaskRasterizer {
    pub(super) fn new(sub_mask: &SubMask) -> Option<Self> {
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
    pub(super) fn rasterize(
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

    pub(super) fn halo(&self, width: u32, height: u32) -> u32 {
        let depth_radius = match self.postprocess {
            AiMaskPostprocess::Binary => 0,
            AiMaskPostprocess::Depth(selection) => selection.blur_radius(),
        };
        depth_radius
            .saturating_add(GrowFeatherPlan::new(self.grow, self.feather, width, height).halo())
    }

    pub(super) fn apply_cross_pixel_filters(&self, mask: &mut GrayImage, width: u32, height: u32) {
        if let AiMaskPostprocess::Depth(selection) = self.postprocess {
            selection.apply_blur(mask);
        }
        GrowFeatherPlan::new(self.grow, self.feather, width, height).apply(mask);
    }
}

pub(super) fn generate_ai_sky_bitmap(
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

pub(super) fn generate_ai_depth_bitmap(
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

pub(super) fn generate_ai_foreground_bitmap(
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

pub(super) fn generate_ai_subject_bitmap(
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
