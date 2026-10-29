//! Texture uploads and readbacks.
//!
//! Converts CPU images row by row to RGBA16F, uploads GPU inputs in bounded
//! row bands and reads processed texture regions back as packed RGBA8 rows.

use crate::render_strategy::GPU_INPUT_UPLOAD_BAND_ROWS;
use crate::render_strategy::GpuInputUploadPlan;
use half::f16;
use image::DynamicImage;
use image::GenericImageView;

use super::GpuInputSource;

pub(super) fn read_texture_data_roi(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    origin: wgpu::Origin3d,
    size: wgpu::Extent3d,
) -> Result<Vec<u8>, String> {
    let unpadded_bytes_per_row = 4 * size.width;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);
    let output_buffer_size = (padded_bytes_per_row * size.height) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );

    queue.submit(Some(encoder.finish()));
    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        })
        .map_err(|e| format!("Failed while polling mapped GPU buffer: {}", e))?;
    let map_result = rx
        .recv()
        .map_err(|e| format!("Failed receiving GPU map result: {}", e))?;
    map_result.map_err(|e| e.to_string())?;

    let padded_data = buffer_slice
        .get_mapped_range()
        .map_err(|e| format!("Failed to get mapped GPU buffer range: {}", e))?
        .to_vec();
    output_buffer.unmap();

    if padded_bytes_per_row == unpadded_bytes_per_row {
        Ok(padded_data)
    } else {
        let mut unpadded_data = Vec::with_capacity((unpadded_bytes_per_row * size.height) as usize);
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        Ok(unpadded_data)
    }
}

#[cfg(test)]
pub(super) fn to_rgba_f16(img: &DynamicImage) -> Vec<f16> {
    match img {
        DynamicImage::ImageRgb32F(rgb) => {
            let mut output = Vec::with_capacity(rgb.as_raw().len() / 3 * 4);
            for pixel in rgb.as_raw().chunks_exact(3) {
                output.extend_from_slice(&[
                    f16::from_f32(pixel[0]),
                    f16::from_f32(pixel[1]),
                    f16::from_f32(pixel[2]),
                    f16::ONE,
                ]);
            }
            output
        }
        DynamicImage::ImageRgba32F(rgba) => {
            rgba.as_raw().iter().copied().map(f16::from_f32).collect()
        }
        _ => img
            .to_rgba32f()
            .into_raw()
            .into_iter()
            .map(f16::from_f32)
            .collect(),
    }
}

#[inline]
pub(super) fn write_rgba_f16_pixel(output: &mut [f16], pixel_index: usize, rgba: [f32; 4]) {
    let offset = pixel_index * 4;
    output[offset] = f16::from_f32(rgba[0]);
    output[offset + 1] = f16::from_f32(rgba[1]);
    output[offset + 2] = f16::from_f32(rgba[2]);
    output[offset + 3] = f16::from_f32(rgba[3]);
}

pub(super) fn write_rgba_f16_row_from_samples<T>(
    source: &[T],
    y: u32,
    output: &mut [f16],
    width: usize,
    channels: usize,
    max_value: f32,
) -> Result<(), String>
where
    T: Copy + Into<f32>,
{
    let row_samples = width
        .checked_mul(4)
        .ok_or_else(|| "GPU input row exceeds the addressable buffer size".to_string())?;
    if output.len() < row_samples {
        return Err("GPU input row staging buffer is too small".to_string());
    }
    if !(1..=4).contains(&channels) {
        return Err("GPU input row has an unsupported channel count".to_string());
    }

    let source_row_samples = width
        .checked_mul(channels)
        .ok_or_else(|| "GPU input source row exceeds the addressable buffer size".to_string())?;
    let start = (y as usize)
        .checked_mul(source_row_samples)
        .ok_or_else(|| "GPU input source row offset overflow".to_string())?;
    let end = start
        .checked_add(source_row_samples)
        .ok_or_else(|| "GPU input source row end overflow".to_string())?;
    let row = source
        .get(start..end)
        .ok_or_else(|| "GPU input source row is outside the source image".to_string())?;

    for (pixel_index, pixel) in row.chunks_exact(channels).enumerate() {
        let red = pixel[0].into() / max_value;
        let green = if channels >= 3 {
            pixel[1].into() / max_value
        } else {
            red
        };
        let blue = if channels >= 3 {
            pixel[2].into() / max_value
        } else {
            red
        };
        let alpha = match channels {
            2 => pixel[1].into() / max_value,
            4 => pixel[3].into() / max_value,
            _ => 1.0,
        };
        write_rgba_f16_pixel(output, pixel_index, [red, green, blue, alpha]);
    }

    Ok(())
}

pub(super) fn write_rgba_f16_row(
    image: &DynamicImage,
    y: u32,
    output: &mut [f16],
    width: usize,
) -> Result<(), String> {
    let max_u8 = u8::MAX as f32;
    let max_u16 = u16::MAX as f32;

    match image {
        DynamicImage::ImageLuma8(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 1, max_u8)?;
        }
        DynamicImage::ImageLumaA8(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 2, max_u8)?;
        }
        DynamicImage::ImageRgb8(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 3, max_u8)?;
        }
        DynamicImage::ImageRgba8(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 4, max_u8)?;
        }
        DynamicImage::ImageLuma16(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 1, max_u16)?;
        }
        DynamicImage::ImageLumaA16(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 2, max_u16)?;
        }
        DynamicImage::ImageRgb16(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 3, max_u16)?;
        }
        DynamicImage::ImageRgba16(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 4, max_u16)?;
        }
        DynamicImage::ImageRgb32F(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 3, 1.0)?;
        }
        DynamicImage::ImageRgba32F(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 4, 1.0)?;
        }
        _ => {
            for x in 0..width as u32 {
                let pixel = image.get_pixel(x, y);
                write_rgba_f16_pixel(
                    output,
                    x as usize,
                    [
                        f32::from(pixel[0]) / max_u8,
                        f32::from(pixel[1]) / max_u8,
                        f32::from(pixel[2]) / max_u8,
                        f32::from(pixel[3]) / max_u8,
                    ],
                );
            }
        }
    }

    Ok(())
}

pub(super) fn upload_source_to_texture_bounded(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    source: &dyn GpuInputSource,
) -> Result<GpuInputUploadPlan, String> {
    let (width, height) = source.dimensions();
    let plan = GpuInputUploadPlan::new(width, height)
        .ok_or_else(|| format!("Cannot upload invalid GPU input dimensions {width}x{height}"))?;
    let padded_row_samples = plan.padded_bytes_per_row as usize / std::mem::size_of::<f16>();
    let max_staging_samples = plan.max_staging_bytes / std::mem::size_of::<f16>();
    let mut staging = vec![f16::ZERO; max_staging_samples];

    for band_start in (0..height).step_by(GPU_INPUT_UPLOAD_BAND_ROWS as usize) {
        let band_height = (height - band_start).min(plan.band_rows);
        let band_samples = padded_row_samples
            .checked_mul(band_height as usize)
            .ok_or_else(|| "GPU input band exceeds the addressable buffer size".to_string())?;
        staging[..band_samples].fill(f16::ZERO);
        source.write_rgba_f16_band(
            band_start,
            band_height,
            padded_row_samples,
            &mut staging[..band_samples],
        )?;

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: band_start,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&staging[..band_samples]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(plan.padded_bytes_per_row),
                rows_per_image: Some(band_height),
            },
            wgpu::Extent3d {
                width,
                height: band_height,
                depth_or_array_layers: 1,
            },
        );
    }

    Ok(plan)
}
