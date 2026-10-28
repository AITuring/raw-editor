//! Row/band geometry output for the full-resolution export path.
//!
//! Orthogonal orientation and flips as exact coordinate permutations, the
//! transformed crop and checkpointed row starts, so export can stream warped
//! rows without materialising the whole image.

use half::f16;
use image::DynamicImage;
use image::GenericImageView;
use image::Rgba32FImage;
use nalgebra::Vector3 as NaVector3;
use rayon::prelude::*;
use serde_json::Value;
use std::mem::size_of;

use super::{Crop, GeometryParams, GeometryWarpContext, apply_coarse_rotation, apply_flip};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GeometryCropRect {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

pub(super) const GEOMETRY_ROW_CHECKPOINT_COLUMNS: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GeometryOrthogonalTransform {
    pub(super) source_width: u32,
    pub(super) source_height: u32,
    pub(super) orientation_steps: u8,
    pub(super) flip_horizontal: bool,
    pub(super) flip_vertical: bool,
}

impl GeometryOrthogonalTransform {
    pub(super) fn new(
        source_width: u32,
        source_height: u32,
        orientation_steps: u8,
        flip_horizontal: bool,
        flip_vertical: bool,
    ) -> Self {
        Self {
            source_width,
            source_height,
            orientation_steps: match orientation_steps {
                1..=3 => orientation_steps,
                _ => 0,
            },
            flip_horizontal,
            flip_vertical,
        }
    }

    pub(super) fn dimensions(self) -> (u32, u32) {
        if self.swaps_axes() {
            (self.source_height, self.source_width)
        } else {
            (self.source_width, self.source_height)
        }
    }

    pub(super) fn swaps_axes(self) -> bool {
        matches!(self.orientation_steps, 1 | 3)
    }

    pub(super) fn is_identity(self) -> bool {
        self.orientation_steps == 0 && !self.flip_horizontal && !self.flip_vertical
    }

    pub(super) fn map_output_to_geometry(self, output_x: u32, output_y: u32) -> (u32, u32) {
        let (output_width, output_height) = self.dimensions();
        debug_assert!(output_x < output_width && output_y < output_height);
        let oriented_x = if self.flip_horizontal {
            output_width - output_x - 1
        } else {
            output_x
        };
        let oriented_y = if self.flip_vertical {
            output_height - output_y - 1
        } else {
            output_y
        };

        match self.orientation_steps {
            1 => (oriented_y, self.source_height - oriented_x - 1),
            2 => (
                self.source_width - oriented_x - 1,
                self.source_height - oriented_y - 1,
            ),
            3 => (self.source_width - oriented_y - 1, oriented_x),
            _ => (oriented_x, oriented_y),
        }
    }
}

pub(super) fn resolve_geometry_crop_rect(
    image_width: u32,
    image_height: u32,
    crop_value: &Value,
) -> GeometryCropRect {
    let full = GeometryCropRect {
        x: 0,
        y: 0,
        width: image_width,
        height: image_height,
    };
    let Ok(crop) = serde_json::from_value::<Crop>(crop_value.clone()) else {
        return full;
    };

    let x = crop.x.round() as u32;
    let y = crop.y.round() as u32;
    let width = crop.width.round() as u32;
    let height = crop.height.round() as u32;
    if width == 0 || height == 0 || x >= image_width || y >= image_height {
        return full;
    }

    let width = (image_width - x).min(width);
    let height = (image_height - y).min(height);
    if width == 0 || height == 0 {
        full
    } else {
        GeometryCropRect {
            x,
            y,
            width,
            height,
        }
    }
}

pub(super) struct GeometryRowCheckpoints {
    pub(super) checkpoints_per_row: usize,
    pub(super) current_vectors: Vec<NaVector3<f32>>,
}

impl GeometryRowCheckpoints {
    pub(super) fn new(context: &GeometryWarpContext<'_>) -> Option<Self> {
        let checkpoints_per_row =
            (context.width as usize).div_ceil(GEOMETRY_ROW_CHECKPOINT_COLUMNS as usize);
        let checkpoint_count = (context.height as usize).checked_mul(checkpoints_per_row)?;
        let mut current_vectors = vec![NaVector3::zeros(); checkpoint_count];

        current_vectors
            .par_chunks_exact_mut(checkpoints_per_row)
            .enumerate()
            .for_each(|(y, checkpoints)| {
                let mut current_vec = context.row_start(y as u32, 0);
                for x in 0..context.width {
                    if x.is_multiple_of(GEOMETRY_ROW_CHECKPOINT_COLUMNS) {
                        checkpoints[(x / GEOMETRY_ROW_CHECKPOINT_COLUMNS) as usize] = current_vec;
                    }
                    current_vec += context.step_vec_x;
                }
            });

        Some(Self {
            checkpoints_per_row,
            current_vectors,
        })
    }

    pub(super) fn row_start(
        &self,
        context: &GeometryWarpContext<'_>,
        y: u32,
        x: u32,
    ) -> NaVector3<f32> {
        let checkpoint_column = x / GEOMETRY_ROW_CHECKPOINT_COLUMNS;
        let checkpoint_x = checkpoint_column * GEOMETRY_ROW_CHECKPOINT_COLUMNS;
        let checkpoint_index = y as usize * self.checkpoints_per_row + checkpoint_column as usize;
        let mut current_vec = self.current_vectors[checkpoint_index];
        for _ in checkpoint_x..x {
            current_vec += context.step_vec_x;
        }
        current_vec
    }

    pub(super) fn allocated_bytes(&self) -> usize {
        self.current_vectors
            .capacity()
            .saturating_mul(size_of::<NaVector3<f32>>())
    }
}

/// A borrowed, row/band-addressable geometry result for the full-resolution export path.
///
/// The source must already use float RGB/RGBA storage. Orthogonal orientation and flips are
/// applied as exact coordinate permutations before the transformed crop is selected. Fine
/// rotation and other transforms that require a second resampling pass deliberately use the
/// materialized fallback.
pub struct GeometryWarpRows<'a> {
    pub(super) context: GeometryWarpContext<'a>,
    pub(super) transform: GeometryOrthogonalTransform,
    pub(super) crop: GeometryCropRect,
    pub(super) mask_crop_offset: (f32, f32),
    pub(super) row_checkpoints: Option<GeometryRowCheckpoints>,
}

impl<'a> GeometryWarpRows<'a> {
    pub fn try_new_transformed_borrowed(
        image: &'a DynamicImage,
        params: GeometryParams,
        orientation_steps: u8,
        flip_horizontal: bool,
        flip_vertical: bool,
        crop_value: &Value,
    ) -> Option<Self> {
        if !matches!(
            image,
            DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_)
        ) {
            return None;
        }
        let (width, height) = image.dimensions();
        let mask_crop_offset = serde_json::from_value::<Crop>(crop_value.clone())
            .map(|crop| (crop.x as f32, crop.y as f32))
            .unwrap_or((0.0, 0.0));
        let context = GeometryWarpContext::new(image, params);
        let transform = GeometryOrthogonalTransform::new(
            width,
            height,
            orientation_steps,
            flip_horizontal,
            flip_vertical,
        );
        let (output_width, output_height) = transform.dimensions();
        let row_checkpoints = if transform.swaps_axes() {
            Some(GeometryRowCheckpoints::new(&context)?)
        } else {
            None
        };
        Some(Self {
            context,
            transform,
            crop: resolve_geometry_crop_rect(output_width, output_height, crop_value),
            mask_crop_offset,
            row_checkpoints,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.crop.width, self.crop.height)
    }

    pub fn crop_offset(&self) -> (f32, f32) {
        self.mask_crop_offset
    }

    pub fn write_rgba16f_row(&self, y: u32, output: &mut [f16]) -> Result<(), String> {
        if y >= self.crop.height {
            return Err(format!(
                "Geometry output row {} is outside cropped height {}",
                y, self.crop.height
            ));
        }
        let width = self.crop.width as usize;
        let samples = width.checked_mul(4).ok_or_else(|| {
            "Geometry RGBA16F row exceeds the addressable buffer size".to_string()
        })?;
        if output.len() < samples {
            return Err("Geometry RGBA16F row staging buffer is too small".to_string());
        }

        let output_y = self.crop.y + y;
        if self.transform.swaps_axes() {
            let sample_context = self.context.sample_context();
            for (x, output_pixel) in output[..samples].chunks_exact_mut(4).enumerate() {
                let (geometry_x, geometry_y) = self
                    .transform
                    .map_output_to_geometry(self.crop.x + x as u32, output_y);
                let current_vec = self.exact_row_start(geometry_y, geometry_x);
                let mut pixel = [0.0_f32; 4];
                self.context
                    .sample_pixel(&sample_context, current_vec, &mut pixel);
                for (output, value) in output_pixel.iter_mut().zip(pixel) {
                    *output = f16::from_f32(value);
                }
            }
            return Ok(());
        }

        let (first_geometry_x, geometry_y) =
            self.transform.map_output_to_geometry(self.crop.x, output_y);
        let (last_geometry_x, last_geometry_y) = self
            .transform
            .map_output_to_geometry(self.crop.x + self.crop.width - 1, output_y);
        debug_assert_eq!(geometry_y, last_geometry_y);
        let geometry_x = first_geometry_x.min(last_geometry_x);
        self.context
            .write_rgba16f_row(geometry_y, geometry_x, width, output)?;
        if first_geometry_x > last_geometry_x {
            reverse_rgba16f_pixels(&mut output[..samples]);
        }
        Ok(())
    }

    pub fn write_rgba16f_band(
        &self,
        y: u32,
        rows: u32,
        row_stride_samples: usize,
        output: &mut [f16],
    ) -> Result<(), String> {
        if rows == 0 {
            return Ok(());
        }
        if y >= self.crop.height || rows > self.crop.height - y {
            return Err(format!(
                "Geometry output band y={} rows={} is outside cropped height {}",
                y, rows, self.crop.height
            ));
        }
        let row_samples = (self.crop.width as usize)
            .checked_mul(4)
            .ok_or_else(|| "Geometry output row exceeds the addressable buffer size".to_string())?;
        if row_stride_samples < row_samples {
            return Err("Geometry output band stride is smaller than one RGBA row".to_string());
        }
        let output_samples = row_stride_samples
            .checked_mul(rows as usize)
            .ok_or_else(|| {
                "Geometry output band exceeds the addressable buffer size".to_string()
            })?;
        if output.len() < output_samples {
            return Err("Geometry output band staging buffer is too small".to_string());
        }

        if !self.transform.swaps_axes() {
            return output[..output_samples]
                .par_chunks_exact_mut(row_stride_samples)
                .enumerate()
                .try_for_each(|(row, output)| self.write_rgba16f_row(y + row as u32, output));
        }

        let output_x_first = self.crop.x;
        let output_x_last = self.crop.x + self.crop.width - 1;
        let output_y_first = self.crop.y + y;
        let output_y_last = output_y_first + rows - 1;
        let (geometry_x_first, geometry_y_first) = self
            .transform
            .map_output_to_geometry(output_x_first, output_y_first);
        let (geometry_x_last, _) = self
            .transform
            .map_output_to_geometry(output_x_first, output_y_last);
        let (_, geometry_y_last) = self
            .transform
            .map_output_to_geometry(output_x_last, output_y_first);
        let geometry_x = geometry_x_first.min(geometry_x_last);
        let geometry_y = geometry_y_first.min(geometry_y_last);
        debug_assert_eq!(geometry_x_first.abs_diff(geometry_x_last) + 1, rows);
        debug_assert_eq!(
            geometry_y_first.abs_diff(geometry_y_last) + 1,
            self.crop.width
        );

        let tile_row_samples = (rows as usize)
            .checked_mul(4)
            .ok_or_else(|| "Geometry transpose tile row is too large".to_string())?;
        let tile_samples = (self.crop.width as usize)
            .checked_mul(tile_row_samples)
            .ok_or_else(|| "Geometry transpose tile is too large".to_string())?;
        let mut tile = vec![f16::ZERO; tile_samples];
        tile.par_chunks_exact_mut(tile_row_samples)
            .enumerate()
            .try_for_each(|(row, output)| {
                let geometry_y = geometry_y + row as u32;
                let current_vec = self.exact_row_start(geometry_y, geometry_x);
                self.context.write_rgba16f_row_from_current(
                    geometry_y,
                    geometry_x,
                    rows as usize,
                    output,
                    current_vec,
                )
            })?;

        output[..output_samples]
            .par_chunks_exact_mut(row_stride_samples)
            .enumerate()
            .for_each(|(band_row, output)| {
                let output_y = output_y_first + band_row as u32;
                for output_x in 0..self.crop.width {
                    let (geometry_pixel_x, geometry_pixel_y) = self
                        .transform
                        .map_output_to_geometry(self.crop.x + output_x, output_y);
                    let tile_pixel = (geometry_pixel_y - geometry_y) as usize * tile_row_samples
                        + (geometry_pixel_x - geometry_x) as usize * 4;
                    let output_pixel = output_x as usize * 4;
                    output[output_pixel..output_pixel + 4]
                        .copy_from_slice(&tile[tile_pixel..tile_pixel + 4]);
                }
            });
        Ok(())
    }

    pub fn materialized_output_bytes(&self) -> usize {
        (self.crop.width as usize)
            .saturating_mul(self.crop.height as usize)
            .saturating_mul(4)
            .saturating_mul(size_of::<f32>())
    }

    pub fn checkpoint_bytes(&self) -> usize {
        self.row_checkpoints
            .as_ref()
            .map_or(0, GeometryRowCheckpoints::allocated_bytes)
    }

    pub fn max_band_scratch_bytes(&self, band_rows: u32) -> usize {
        if !self.transform.swaps_axes() {
            return 0;
        }
        (self.crop.width as usize)
            .saturating_mul(self.crop.height.min(band_rows) as usize)
            .saturating_mul(4)
            .saturating_mul(size_of::<f16>())
    }

    pub fn materialize(&self) -> DynamicImage {
        if !self.transform.is_identity() {
            let row_samples = self.context.width as usize * 4;
            let mut output = vec![0.0_f32; row_samples * self.context.height as usize];
            output
                .par_chunks_exact_mut(row_samples)
                .enumerate()
                .for_each(|(y, row)| {
                    self.context
                        .write_rgba32f_row(y as u32, 0, row)
                        .expect("full geometry output row must fit the source canvas");
                });
            let warped = DynamicImage::ImageRgba32F(
                Rgba32FImage::from_vec(self.context.width, self.context.height, output)
                    .expect("geometry output dimensions must match the allocated buffer"),
            );
            let rotated = apply_coarse_rotation(warped, self.transform.orientation_steps);
            let flipped = apply_flip(
                rotated,
                self.transform.flip_horizontal,
                self.transform.flip_vertical,
            );
            let transformed = flipped.into_owned();
            let (transformed_width, transformed_height) = transformed.dimensions();
            if self.crop.x == 0
                && self.crop.y == 0
                && self.crop.width == transformed_width
                && self.crop.height == transformed_height
            {
                return transformed;
            }
            return transformed.crop_imm(
                self.crop.x,
                self.crop.y,
                self.crop.width,
                self.crop.height,
            );
        }

        let row_samples = self.crop.width as usize * 4;
        let mut output = vec![0.0_f32; row_samples * self.crop.height as usize];
        output
            .par_chunks_exact_mut(row_samples)
            .enumerate()
            .for_each(|(y, row)| {
                self.context
                    .write_rgba32f_row(self.crop.y + y as u32, self.crop.x, row)
                    .expect("cropped geometry output row must fit the source canvas");
            });
        DynamicImage::ImageRgba32F(
            Rgba32FImage::from_vec(self.crop.width, self.crop.height, output)
                .expect("geometry output dimensions must match the allocated buffer"),
        )
    }

    pub(super) fn exact_row_start(&self, y: u32, x: u32) -> NaVector3<f32> {
        self.row_checkpoints.as_ref().map_or_else(
            || self.context.row_start(y, x),
            |checkpoints| checkpoints.row_start(&self.context, y, x),
        )
    }
}

pub(super) fn reverse_rgba16f_pixels(row: &mut [f16]) {
    debug_assert!(row.len().is_multiple_of(4));
    let width = row.len() / 4;
    for left in 0..width / 2 {
        let right = width - left - 1;
        for channel in 0..4 {
            row.swap(left * 4 + channel, right * 4 + channel);
        }
    }
}
