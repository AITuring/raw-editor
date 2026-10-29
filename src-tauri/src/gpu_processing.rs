use std::sync::Arc;

use half::f16;
use image::{DynamicImage, GenericImageView};
use rayon::prelude::*;

use crate::app_state::SharedMaskBitmap;
use crate::image_processing::{AllAdjustments, GeometryWarpRows};
use crate::lut_processing::Lut;

#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct RenderRequest<'a> {
    pub adjustments: AllAdjustments,
    pub mask_bitmaps: &'a [SharedMaskBitmap],
    pub lut: Option<Arc<Lut>>,
    pub roi: Option<Roi>,
}

pub trait GpuInputSource: Sync {
    fn dimensions(&self) -> (u32, u32);
    fn write_rgba_f16_row(&self, y: u32, output: &mut [f16]) -> Result<(), String>;
    fn write_rgba_f16_band(
        &self,
        y: u32,
        rows: u32,
        row_stride_samples: usize,
        output: &mut [f16],
    ) -> Result<(), String> {
        let (width, height) = self.dimensions();
        if rows == 0 {
            return Ok(());
        }
        if y >= height || rows > height - y {
            return Err(format!(
                "GPU input band y={} rows={} is outside source height {}",
                y, rows, height
            ));
        }
        let row_samples = (width as usize)
            .checked_mul(4)
            .ok_or_else(|| "GPU input row exceeds the addressable buffer size".to_string())?;
        if row_stride_samples < row_samples {
            return Err("GPU input band stride is smaller than one RGBA row".to_string());
        }
        let required_samples = row_stride_samples
            .checked_mul(rows as usize)
            .ok_or_else(|| "GPU input band exceeds the addressable buffer size".to_string())?;
        if output.len() < required_samples {
            return Err("GPU input band staging buffer is too small".to_string());
        }
        output[..required_samples]
            .par_chunks_exact_mut(row_stride_samples)
            .enumerate()
            .try_for_each(|(row, output)| self.write_rgba_f16_row(y + row as u32, output))
    }
    fn materialize_cpu_fallback(&self) -> Option<DynamicImage> {
        None
    }
}

type RgbaRowSink<'a> = dyn FnMut(&[u8]) -> Result<(), String> + 'a;

enum GpuRenderOutput<'a> {
    CpuImage,
    NativeDisplay,
    StreamRows(&'a mut RgbaRowSink<'a>),
}

mod context;

pub use context::{WgpuDisplay, get_or_init_gpu_context};

mod texture_io;

#[cfg(test)]
use texture_io::to_rgba_f16;
use texture_io::{read_texture_data_roi, upload_source_to_texture_bounded, write_rgba_f16_row};

mod processor;

pub use processor::GpuProcessor;

mod render;

pub use render::{
    process_and_get_dynamic_image, process_and_get_dynamic_image_with_analytics,
    process_and_stream_rgba_rows, reclaim_gpu_resources_after_export,
};

impl GpuInputSource for DynamicImage {
    fn dimensions(&self) -> (u32, u32) {
        GenericImageView::dimensions(self)
    }

    fn write_rgba_f16_row(&self, y: u32, output: &mut [f16]) -> Result<(), String> {
        write_rgba_f16_row(self, y, output, self.width() as usize)
    }

    fn materialize_cpu_fallback(&self) -> Option<DynamicImage> {
        Some(self.clone())
    }
}

impl GpuInputSource for Arc<DynamicImage> {
    fn dimensions(&self) -> (u32, u32) {
        GenericImageView::dimensions(self.as_ref())
    }

    fn write_rgba_f16_row(&self, y: u32, output: &mut [f16]) -> Result<(), String> {
        write_rgba_f16_row(self.as_ref(), y, output, self.width() as usize)
    }

    fn materialize_cpu_fallback(&self) -> Option<DynamicImage> {
        Some(self.as_ref().clone())
    }
}

impl GpuInputSource for GeometryWarpRows<'_> {
    fn dimensions(&self) -> (u32, u32) {
        GeometryWarpRows::dimensions(self)
    }

    fn write_rgba_f16_row(&self, y: u32, output: &mut [f16]) -> Result<(), String> {
        self.write_rgba16f_row(y, output)
    }

    fn write_rgba_f16_band(
        &self,
        y: u32,
        rows: u32,
        row_stride_samples: usize,
        output: &mut [f16],
    ) -> Result<(), String> {
        self.write_rgba16f_band(y, rows, row_stride_samples, output)
    }

    fn materialize_cpu_fallback(&self) -> Option<DynamicImage> {
        Some(self.materialize())
    }
}

#[cfg(test)]
mod shader_tests;
