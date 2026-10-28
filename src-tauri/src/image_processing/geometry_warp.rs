//! The forward geometry warp.
//!
//! `GeometryWarpContext` maps output pixels to source positions with lens
//! distortion, TCA and vignetting; the sampler serves warped pixels on demand
//! for range masks, and `warp_image_geometry` materialises the whole output.

use half::f16;
use image::DynamicImage;
use image::GenericImageView;
use image::Rgba;
use image::Rgba32FImage;
use nalgebra::Matrix3 as NaMatrix3;
use nalgebra::Vector3 as NaVector3;
use rayon::prelude::*;
use std::mem::size_of;
use std::sync::OnceLock;

use super::{
    GeometryParams, GeometryPixelSource, GeometrySampleContext, GeometryWarpBufferPlan,
    build_transform_matrices, interpolate_geometry_pixel, interpolate_pixel_with_tca,
    unproject_geometry_point,
};

pub(super) struct GeometryWarpContext<'a> {
    pub(super) source: GeometryPixelSource<'a>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) step_vec_x: NaVector3<f32>,
    pub(super) step_vec_y: NaVector3<f32>,
    pub(super) origin_vec: NaVector3<f32>,
    pub(super) cx: f32,
    pub(super) cy: f32,
    pub(super) half_diagonal: f64,
    pub(super) max_radius_sq_inv: f64,
    pub(super) params: GeometryParams,
    pub(super) k_distortion: f64,
    pub(super) lens_dist_k1: f64,
    pub(super) lens_dist_k2: f64,
    pub(super) lens_dist_k3: f64,
    pub(super) lens_distortion_amount: f64,
    pub(super) has_lens_correction: bool,
    pub(super) is_ptlens: bool,
    pub(super) tca_vr: f32,
    pub(super) tca_vb: f32,
    pub(super) has_tca: bool,
    pub(super) vig_k1: f64,
    pub(super) vig_k2: f64,
    pub(super) vig_k3: f64,
    pub(super) lens_vignette_amount: f64,
    pub(super) has_vignetting: bool,
}

impl<'a> GeometryWarpContext<'a> {
    pub(super) fn new(image: &'a DynamicImage, params: GeometryParams) -> Self {
        Self::with_source(image, params, GeometryPixelSource::new(image))
    }

    pub(super) fn new_borrowed(image: &'a DynamicImage, params: GeometryParams) -> Self {
        Self::with_source(image, params, GeometryPixelSource::borrowed(image))
    }

    pub(super) fn with_source(
        image: &'a DynamicImage,
        params: GeometryParams,
        source: GeometryPixelSource<'a>,
    ) -> Self {
        let (width, height) = image.dimensions();
        let (forward_transform, cx, cy, half_diagonal) =
            build_transform_matrices(&params, width as f32, height as f32);
        let inv = forward_transform
            .try_inverse()
            .unwrap_or(NaMatrix3::identity());

        let step_vec_x = NaVector3::new(inv[(0, 0)], inv[(1, 0)], inv[(2, 0)]);
        let step_vec_y = NaVector3::new(inv[(0, 1)], inv[(1, 1)], inv[(2, 1)]);
        let origin_vec = NaVector3::new(inv[(0, 2)], inv[(1, 2)], inv[(2, 2)]);
        let max_radius_sq_inv = 1.0 / ((cx * cx + cy * cy) as f64);
        let k_distortion = (params.distortion as f64 / 100.0) * 2.5;
        let lens_dist_k1 = params.lens_dist_k1 as f64;
        let lens_dist_k2 = params.lens_dist_k2 as f64;
        let lens_dist_k3 = params.lens_dist_k3 as f64;
        let lens_distortion_amount = (params.lens_distortion_amount as f64) * 2.5;
        let has_lens_correction = params.lens_distortion_enabled
            && (lens_dist_k1.abs() > 1e-6
                || lens_dist_k2.abs() > 1e-6
                || lens_dist_k3.abs() > 1e-6);
        let is_ptlens = params.lens_model == 1;
        let tca_vr = if (params.tca_vr - 1.0).abs() > 1e-5 {
            params.tca_vr + (1.0 - params.tca_vr) * (1.0 - params.lens_tca_amount)
        } else {
            1.0
        };
        let tca_vb = if (params.tca_vb - 1.0).abs() > 1e-5 {
            params.tca_vb + (1.0 - params.tca_vb) * (1.0 - params.lens_tca_amount)
        } else {
            1.0
        };
        let has_tca =
            params.lens_tca_enabled && ((tca_vr - 1.0).abs() > 1e-5 || (tca_vb - 1.0).abs() > 1e-5);
        let vig_k1 = params.vig_k1 as f64;
        let vig_k2 = params.vig_k2 as f64;
        let vig_k3 = params.vig_k3 as f64;
        let lens_vignette_amount = (params.lens_vignette_amount as f64) * 0.8;
        let has_vignetting = params.lens_vignette_enabled
            && (vig_k1.abs() > 1e-6 || vig_k2.abs() > 1e-6 || vig_k3.abs() > 1e-6)
            && lens_vignette_amount > 0.01;

        Self {
            source,
            width,
            height,
            step_vec_x,
            step_vec_y,
            origin_vec,
            cx,
            cy,
            half_diagonal,
            max_radius_sq_inv,
            params,
            k_distortion,
            lens_dist_k1,
            lens_dist_k2,
            lens_dist_k3,
            lens_distortion_amount,
            has_lens_correction,
            is_ptlens,
            tca_vr,
            tca_vb,
            has_tca,
            vig_k1,
            vig_k2,
            vig_k3,
            lens_vignette_amount,
            has_vignetting,
        }
    }

    pub(super) fn sample_context(&self) -> GeometrySampleContext<'_> {
        GeometrySampleContext {
            source: self.source.sample_data(),
            src_width: self.width as usize,
            src_height: self.height as usize,
            cx: self.cx,
            cy: self.cy,
        }
    }

    pub(super) fn row_start(&self, y: u32, x_offset: u32) -> NaVector3<f32> {
        let mut current = self.origin_vec + self.step_vec_y * y as f32;
        // Match the materialized implementation's repeated f32 addition so a cropped
        // streamed row is bit-identical to slicing the corresponding full output row.
        for _ in 0..x_offset {
            current += self.step_vec_x;
        }
        current
    }

    pub(super) fn sample_pixel(
        &self,
        sample_context: &GeometrySampleContext<'_>,
        current_vec: NaVector3<f32>,
        pixel: &mut [f32; 4],
    ) {
        if current_vec.z.abs() <= 1e-6 {
            return;
        }

        let inv_z = 1.0 / current_vec.z;
        let mut src_x = current_vec.x * inv_z;
        let mut src_y = current_vec.y * inv_z;
        let (unprojected_x, unprojected_y) = unproject_geometry_point(
            src_x as f64,
            src_y as f64,
            self.cx as f64,
            self.cy as f64,
            self.half_diagonal,
            self.params.projection,
        );
        src_x = unprojected_x as f32;
        src_y = unprojected_y as f32;

        if self.has_lens_correction {
            let dx = (src_x - self.cx) as f64;
            let dy = (src_y - self.cy) as f64;
            let ru = (dx * dx + dy * dy).sqrt();
            if ru > 1e-6 {
                let ru_norm = ru / self.half_diagonal;
                let ru_norm2 = ru_norm * ru_norm;
                let rd_norm = if self.is_ptlens {
                    let d = 1.0 - self.lens_dist_k1 - self.lens_dist_k2 - self.lens_dist_k3;
                    ru_norm
                        * (self.lens_dist_k1 * ru_norm2 * ru_norm
                            + self.lens_dist_k2 * ru_norm2
                            + self.lens_dist_k3 * ru_norm
                            + d)
                } else {
                    ru_norm
                        * (1.0
                            + self.lens_dist_k1 * ru_norm2
                            + self.lens_dist_k2 * (ru_norm2 * ru_norm2)
                            + self.lens_dist_k3 * (ru_norm2 * ru_norm2 * ru_norm2))
                };
                let effective_r_norm = ru_norm + (rd_norm - ru_norm) * self.lens_distortion_amount;
                let scale = effective_r_norm / ru_norm;
                src_x = self.cx + (dx * scale) as f32;
                src_y = self.cy + (dy * scale) as f32;
            }
        }

        if self.k_distortion.abs() > 1e-5 {
            let dx = (src_x - self.cx) as f64;
            let dy = (src_y - self.cy) as f64;
            let r2_norm = (dx * dx + dy * dy) * self.max_radius_sq_inv;
            let factor = 1.0 + self.k_distortion * r2_norm;
            src_x = self.cx + (dx * factor) as f32;
            src_y = self.cy + (dy * factor) as f32;
        }

        if self.has_tca {
            interpolate_pixel_with_tca(
                sample_context,
                src_x,
                src_y,
                self.tca_vr,
                self.tca_vb,
                pixel,
            );
        } else {
            interpolate_geometry_pixel(sample_context, src_x, src_y, pixel);
        }

        if self.has_vignetting {
            let dx = (src_x - self.cx) as f64;
            let dy = (src_y - self.cy) as f64;
            let ru_norm = (dx * dx + dy * dy).sqrt() / self.half_diagonal;
            let ru_norm2 = ru_norm * ru_norm;
            let vignette_factor = 1.0
                + self.vig_k1 * ru_norm2
                + self.vig_k2 * (ru_norm2 * ru_norm2)
                + self.vig_k3 * (ru_norm2 * ru_norm2 * ru_norm2);
            if vignette_factor > 1e-6 {
                let correction_gain = 1.0 / vignette_factor;
                let final_gain = 1.0 + (correction_gain - 1.0) * self.lens_vignette_amount;
                pixel[0] *= final_gain as f32;
                pixel[1] *= final_gain as f32;
                pixel[2] *= final_gain as f32;
            }
        }
    }

    pub(super) fn validate_row(&self, y: u32, x_offset: u32, width: usize) -> Result<(), String> {
        if y >= self.height || x_offset > self.width || width > (self.width - x_offset) as usize {
            return Err(format!(
                "Geometry output row x={} width={} y={} is outside {}x{}",
                x_offset, width, y, self.width, self.height
            ));
        }
        Ok(())
    }

    pub(super) fn write_rgba32f_row(
        &self,
        y: u32,
        x_offset: u32,
        output: &mut [f32],
    ) -> Result<(), String> {
        if !output.len().is_multiple_of(4) {
            return Err("Geometry RGBA32F row must contain complete pixels".to_string());
        }
        let width = output.len() / 4;
        self.validate_row(y, x_offset, width)?;
        output.fill(0.0);
        let sample_context = self.sample_context();
        let mut current_vec = self.row_start(y, x_offset);
        for pixel in output.chunks_exact_mut(4) {
            let pixel: &mut [f32; 4] = pixel
                .try_into()
                .expect("RGBA32F row chunks always contain four samples");
            self.sample_pixel(&sample_context, current_vec, pixel);
            current_vec += self.step_vec_x;
        }
        Ok(())
    }

    pub(super) fn write_rgba16f_row(
        &self,
        y: u32,
        x_offset: u32,
        width: usize,
        output: &mut [f16],
    ) -> Result<(), String> {
        let current_vec = self.row_start(y, x_offset);
        self.write_rgba16f_row_from_current(y, x_offset, width, output, current_vec)
    }

    pub(super) fn write_rgba16f_row_from_current(
        &self,
        y: u32,
        x_offset: u32,
        width: usize,
        output: &mut [f16],
        mut current_vec: NaVector3<f32>,
    ) -> Result<(), String> {
        let samples = width.checked_mul(4).ok_or_else(|| {
            "Geometry RGBA16F row exceeds the addressable buffer size".to_string()
        })?;
        if output.len() < samples {
            return Err("Geometry RGBA16F row staging buffer is too small".to_string());
        }
        self.validate_row(y, x_offset, width)?;
        let sample_context = self.sample_context();
        for output_pixel in output[..samples].chunks_exact_mut(4) {
            let mut pixel = [0.0_f32; 4];
            self.sample_pixel(&sample_context, current_vec, &mut pixel);
            for (output, value) in output_pixel.iter_mut().zip(pixel) {
                *output = f16::from_f32(value);
            }
            current_vec += self.step_vec_x;
        }
        Ok(())
    }
}

/// A borrowed, pixel-addressable view of the geometry-warped image.
///
/// Unlike [`warp_image_geometry`], this source does not allocate the complete RGBA32F output.
/// Integer output pixels are sampled on demand with the same transform and interpolation kernel.
/// All current `DynamicImage` storage variants are borrowed directly, so range masks can consume
/// geometry pixels tile by tile without first expanding an 8-bit or 16-bit source to RGBA32F.
pub struct GeometryWarpSampler<'a> {
    pub(super) context: GeometryWarpContext<'a>,
    pub(super) rows: GeometrySamplerRows,
}

impl<'a> GeometryWarpSampler<'a> {
    pub fn new(image: &'a DynamicImage, params: GeometryParams) -> Self {
        let context = GeometryWarpContext::new_borrowed(image, params);
        Self {
            rows: GeometrySamplerRows::new(context.height),
            context,
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.context.width, self.context.height)
    }

    #[cfg(test)]
    pub fn sample_rgba8(&self, x: u32, y: u32) -> Option<Rgba<u8>> {
        if x >= self.context.width || y >= self.context.height {
            return None;
        }

        let mut output = [Rgba([0, 0, 0, 0])];
        self.sample_rgba8_points(&[(x, y)], &mut output)
            .then_some(output[0])
    }

    pub fn sample_rgba8_points(&self, points: &[(u32, u32)], output: &mut [Rgba<u8>]) -> bool {
        if points.len() != output.len()
            || points
                .iter()
                .any(|(x, y)| *x >= self.context.width || *y >= self.context.height)
        {
            return false;
        }

        let sample_context = self.context.sample_context();
        let mut cursor: Option<(u32, u32, NaVector3<f32>)> = None;
        for ((x, y), output) in points.iter().copied().zip(output) {
            let current_vec = match cursor {
                Some((cursor_x, cursor_y, mut current)) if cursor_y == y && x >= cursor_x => {
                    for _ in cursor_x..x {
                        current += self.context.step_vec_x;
                    }
                    current
                }
                _ => self.rows.row_start(&self.context, y, x),
            };
            let mut pixel = [0.0_f32; 4];
            self.context
                .sample_pixel(&sample_context, current_vec, &mut pixel);
            *output = Rgba(pixel.map(normalized_f32_to_u8));
            cursor = Some((x, y, current_vec));
        }
        true
    }

    pub fn checkpoint_bytes(&self) -> usize {
        self.rows.allocated_bytes()
    }

    pub fn max_checkpoint_bytes(&self) -> usize {
        (self.context.height as usize)
            .saturating_mul(
                (self.context.width as usize)
                    .div_ceil(GEOMETRY_SAMPLER_CHECKPOINT_COLUMNS as usize),
            )
            .saturating_mul(size_of::<NaVector3<f32>>())
    }
}

pub(super) const GEOMETRY_SAMPLER_CHECKPOINT_COLUMNS: u32 = 64;

pub(super) struct GeometrySamplerRows {
    pub(super) checkpoints: Vec<OnceLock<Box<[NaVector3<f32>]>>>,
}

impl GeometrySamplerRows {
    pub(super) fn new(height: u32) -> Self {
        Self {
            checkpoints: (0..height).map(|_| OnceLock::new()).collect(),
        }
    }

    pub(super) fn row_start(
        &self,
        context: &GeometryWarpContext<'_>,
        y: u32,
        x: u32,
    ) -> NaVector3<f32> {
        let checkpoints = self.checkpoints[y as usize].get_or_init(|| {
            let mut checkpoints = Vec::with_capacity(
                (context.width as usize).div_ceil(GEOMETRY_SAMPLER_CHECKPOINT_COLUMNS as usize),
            );
            let mut current = context.row_start(y, 0);
            for source_x in 0..context.width {
                if source_x.is_multiple_of(GEOMETRY_SAMPLER_CHECKPOINT_COLUMNS) {
                    checkpoints.push(current);
                }
                current += context.step_vec_x;
            }
            checkpoints.into_boxed_slice()
        });
        let checkpoint_column = x / GEOMETRY_SAMPLER_CHECKPOINT_COLUMNS;
        let checkpoint_x = checkpoint_column * GEOMETRY_SAMPLER_CHECKPOINT_COLUMNS;
        let mut current = checkpoints[checkpoint_column as usize];
        for _ in checkpoint_x..x {
            current += context.step_vec_x;
        }
        current
    }

    pub(super) fn allocated_bytes(&self) -> usize {
        self.checkpoints
            .iter()
            .filter_map(OnceLock::get)
            .map(|row| row.len() * size_of::<NaVector3<f32>>())
            .sum()
    }
}

#[inline]
pub(super) fn normalized_f32_to_u8(value: f32) -> u8 {
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let clamped = if !(value < 1.0) { 1.0 } else { value.max(0.0) };
    (clamped * 255.0).round() as u8
}

pub fn warp_image_geometry(image: &DynamicImage, params: GeometryParams) -> DynamicImage {
    let (width, height) = image.dimensions();
    let context = GeometryWarpContext::new(image, params);
    let buffer_plan = GeometryWarpBufferPlan::new(width, height, context.source.conversion_bytes())
        .expect("geometry warp dimensions exceed the addressable buffer size");
    log::debug!(
        "geometry warp buffers: output={} B source_conversion={} B transient={} B",
        buffer_plan.output_rgba32f_bytes,
        buffer_plan.source_conversion_bytes,
        buffer_plan.transient_bytes()
    );
    let mut out_buffer = vec![0.0f32; buffer_plan.output_samples];
    let width_usize = width as usize;
    out_buffer
        .par_chunks_exact_mut(width_usize * 4)
        .enumerate()
        .for_each(|(y, row_pixel_data)| {
            context
                .write_rgba32f_row(y as u32, 0, row_pixel_data)
                .expect("full geometry output row must fit the source canvas");
        });

    let out_img = Rgba32FImage::from_vec(width, height, out_buffer).unwrap();
    DynamicImage::ImageRgba32F(out_img)
}
