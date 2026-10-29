//! Colour and luminance range sub-masks.
//!
//! Maps output pixels back to the range source and scores each sample against
//! the picked reference colour or luminance.

use image::GrayImage;
use image::Rgba;
use rayon::prelude::*;
use serde_json::Value;
use std::f32::consts::PI;

use super::{ParametricMaskParameters, RangeMaskSource, apply_grow_and_feather};

#[derive(Clone, Copy)]
pub(super) struct RangeMaskCoordinateMapper {
    pub(super) full_width: u32,
    pub(super) full_height: u32,
    pub(super) scaled_coarse_width: f32,
    pub(super) scaled_coarse_height: f32,
    pub(super) center_x: f32,
    pub(super) center_y: f32,
    pub(super) cos_angle: f32,
    pub(super) sin_angle: f32,
    pub(super) inverse_scale: f32,
    pub(super) flip_horizontal: bool,
    pub(super) flip_vertical: bool,
    pub(super) orientation_steps: u8,
}

impl RangeMaskCoordinateMapper {
    pub(super) fn new(
        params: &ParametricMaskParameters,
        full_width: u32,
        full_height: u32,
        scale: f32,
    ) -> Self {
        let angle_radians = params.rotation * PI / 180.0;
        let cos_angle = angle_radians.cos();
        let sin_angle = angle_radians.sin();
        let (coarse_width, coarse_height) = if params.orientation_steps % 2 == 1 {
            (full_height, full_width)
        } else {
            (full_width, full_height)
        };
        let scaled_coarse_width = coarse_width as f32 * scale;
        let scaled_coarse_height = coarse_height as f32 * scale;
        Self {
            full_width,
            full_height,
            scaled_coarse_width,
            scaled_coarse_height,
            center_x: scaled_coarse_width / 2.0,
            center_y: scaled_coarse_height / 2.0,
            cos_angle,
            sin_angle,
            inverse_scale: 1.0 / scale,
            flip_horizontal: params.flip_horizontal,
            flip_vertical: params.flip_vertical,
            orientation_steps: params.orientation_steps,
        }
    }

    pub(super) fn source_coordinates(
        self,
        output_x: u32,
        output_y: u32,
        crop_offset: (f32, f32),
        output_origin: (u32, u32),
    ) -> Option<(u32, u32)> {
        let x_uncropped = (output_x + output_origin.0) as f32 + crop_offset.0;
        let y_uncropped = (output_y + output_origin.1) as f32 + crop_offset.1;
        let x_centered = x_uncropped - self.center_x;
        let y_centered = y_uncropped - self.center_y;
        let x_unrotated = x_centered * self.cos_angle + y_centered * self.sin_angle + self.center_x;
        let y_unrotated =
            -x_centered * self.sin_angle + y_centered * self.cos_angle + self.center_y;
        let x_unflipped = if self.flip_horizontal {
            self.scaled_coarse_width - x_unrotated
        } else {
            x_unrotated
        };
        let y_unflipped = if self.flip_vertical {
            self.scaled_coarse_height - y_unrotated
        } else {
            y_unrotated
        };
        let (x_unrotated_coarse, y_unrotated_coarse) = match self.orientation_steps {
            0 => (x_unflipped, y_unflipped),
            1 => (y_unflipped, self.scaled_coarse_width - x_unflipped),
            2 => (
                self.scaled_coarse_width - x_unflipped,
                self.scaled_coarse_height - y_unflipped,
            ),
            3 => (self.scaled_coarse_height - y_unflipped, x_unflipped),
            _ => (x_unflipped, y_unflipped),
        };
        if x_unrotated_coarse < 0.0 || y_unrotated_coarse < 0.0 {
            return None;
        }
        let source_x = (x_unrotated_coarse * self.inverse_scale) as u32;
        let source_y = (y_unrotated_coarse * self.inverse_scale) as u32;
        (source_x < self.full_width && source_y < self.full_height).then_some((source_x, source_y))
    }
}

#[derive(Clone, Copy)]
pub(super) enum RangeMaskMatch {
    Color { tolerance_squared: f32 },
    Luminance { tolerance: f32 },
}

impl RangeMaskMatch {
    pub(super) fn intensity(self, pixel: Rgba<u8>, reference: Rgba<u8>) -> u8 {
        match self {
            Self::Color { tolerance_squared } => {
                let distance_squared = (pixel[0] as f32 - reference[0] as f32).powi(2)
                    + (pixel[1] as f32 - reference[1] as f32).powi(2)
                    + (pixel[2] as f32 - reference[2] as f32).powi(2);
                if distance_squared <= tolerance_squared {
                    ((1.0 - distance_squared.sqrt() / tolerance_squared.sqrt()) * 255.0) as u8
                } else {
                    0
                }
            }
            Self::Luminance { tolerance } => {
                let reference_luma = 0.299 * reference[0] as f32
                    + 0.587 * reference[1] as f32
                    + 0.114 * reference[2] as f32;
                let luma =
                    0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
                let distance = (luma - reference_luma).abs();
                if distance <= tolerance {
                    ((1.0 - distance / tolerance) * 255.0) as u8
                } else {
                    0
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn generate_range_bitmap_from_parameters(
    params: &ParametricMaskParameters,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
    source: &dyn RangeMaskSource,
    range_match: RangeMaskMatch,
) -> Option<GrayImage> {
    let (full_w, full_h) = source.source_dimensions();

    let target_x = params.target_x.round() as i32;
    let target_y = params.target_y.round() as i32;
    if target_x < 0 || target_y < 0 || target_x >= full_w as i32 || target_y >= full_h as i32 {
        return None;
    }

    let reference = source.sample_rgba8(target_x as u32, target_y as u32)?;
    let mut mask = GrayImage::new(width, height);
    if width == 0 || height == 0 {
        return Some(mask);
    }
    let mapper = RangeMaskCoordinateMapper::new(params, full_w, full_h, scale);
    mask.as_mut()
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(output_y, mask_row)| {
            let mut source_coordinates = Vec::with_capacity(width as usize);
            let mut output_columns = Vec::with_capacity(width as usize);
            for output_x in 0..width {
                if let Some(coordinates) =
                    mapper.source_coordinates(output_x, output_y as u32, crop_offset, output_origin)
                {
                    source_coordinates.push(coordinates);
                    output_columns.push(output_x as usize);
                }
            }
            let mut source_pixels = vec![Rgba([0, 0, 0, 0]); source_coordinates.len()];
            let sampled = source.sample_rgba8_points(&source_coordinates, &mut source_pixels);
            debug_assert!(
                sampled,
                "validated range-mask source coordinates must be readable"
            );
            if !sampled {
                return;
            }
            for (output_x, pixel) in output_columns.into_iter().zip(source_pixels) {
                mask_row[output_x] = range_match.intensity(pixel, reference);
            }
        });

    Some(mask)
}

pub(super) fn generate_color_bitmap_from_parameters(
    params: &ParametricMaskParameters,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
    source: &dyn RangeMaskSource,
) -> Option<GrayImage> {
    let tolerance = (params.tolerance * 2.55).max(1.0);
    generate_range_bitmap_from_parameters(
        params,
        width,
        height,
        scale,
        crop_offset,
        output_origin,
        source,
        RangeMaskMatch::Color {
            tolerance_squared: tolerance.powi(2) * 3.0,
        },
    )
}

pub(super) fn generate_color_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    range_source: Option<&dyn RangeMaskSource>,
) -> Option<GrayImage> {
    let params: ParametricMaskParameters = serde_json::from_value(params_value.clone()).ok()?;
    let mut mask = generate_color_bitmap_from_parameters(
        &params,
        width,
        height,
        scale,
        crop_offset,
        (0, 0),
        range_source?,
    )?;
    apply_grow_and_feather(&mut mask, params.grow, params.feather, width, height);
    Some(mask)
}

pub(super) fn generate_luminance_bitmap_from_parameters(
    params: &ParametricMaskParameters,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    output_origin: (u32, u32),
    source: &dyn RangeMaskSource,
) -> Option<GrayImage> {
    generate_range_bitmap_from_parameters(
        params,
        width,
        height,
        scale,
        crop_offset,
        output_origin,
        source,
        RangeMaskMatch::Luminance {
            tolerance: (params.tolerance * 2.55).max(1.0),
        },
    )
}

pub(super) fn generate_luminance_bitmap(
    params_value: &Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    range_source: Option<&dyn RangeMaskSource>,
) -> Option<GrayImage> {
    let params: ParametricMaskParameters = serde_json::from_value(params_value.clone()).ok()?;
    let mut mask = generate_luminance_bitmap_from_parameters(
        &params,
        width,
        height,
        scale,
        crop_offset,
        (0, 0),
        range_source?,
    )?;
    apply_grow_and_feather(&mut mask, params.grow, params.feather, width, height);
    Some(mask)
}
