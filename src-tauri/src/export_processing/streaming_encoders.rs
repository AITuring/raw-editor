//! Row-streaming export encoders.
//!
//! Validates streamed RGBA rows, applies resize and watermark per row and
//! encodes the rows as JPEG (mozjpeg) or PNG.

use crate::color_management::srgb_v4_profile;
#[cfg(not(target_os = "android"))]
use mozjpeg::ColorSpace as MozjpegColorSpace;
#[cfg(not(target_os = "android"))]
use mozjpeg::Compress as MozjpegCompressor;
#[cfg(not(target_os = "android"))]
use mozjpeg::Marker as MozjpegMarker;
#[cfg(not(target_os = "android"))]
use png::BitDepth;
#[cfg(not(target_os = "android"))]
use png::ColorType as PngColorType;
#[cfg(not(target_os = "android"))]
use png::Encoder as PngStreamEncoder;
#[cfg(not(target_os = "android"))]
use png::Info as PngInfo;
use std::borrow::Cow;
use std::fs;
#[cfg(not(target_os = "android"))]
use std::io::BufWriter;
use std::io::Write;
#[cfg(not(target_os = "android"))]
use zenresize::Filter as StreamingResizeFilter;
#[cfg(not(target_os = "android"))]
use zenresize::PixelDescriptor;
#[cfg(not(target_os = "android"))]
use zenresize::ResizeConfig;
#[cfg(not(target_os = "android"))]
use zenresize::StreamingResize;

use super::{PreparedWatermark, WatermarkSettings, effective_export_bit_depth, prepare_watermark};

#[cfg(not(target_os = "android"))]
pub(super) fn validate_streamed_rgba_row(
    row: &[u8],
    width: u32,
    height: u32,
    written_rows: u32,
) -> Result<(), String> {
    if written_rows >= height {
        return Err(format!(
            "Streaming encoder received more than the declared {height} rows"
        ));
    }
    let expected = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| format!("Streaming width {width} exceeds the RGBA row size limit"))?;
    if row.len() != expected {
        return Err(format!(
            "Streamed row {} has {} bytes; expected {} RGBA bytes",
            written_rows,
            row.len(),
            expected
        ));
    }
    Ok(())
}

#[cfg(not(target_os = "android"))]
pub(super) struct StreamingRowOutput<'a> {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) written_rows: u32,
    pub(super) watermark: Option<PreparedWatermark>,
    pub(super) scratch: Vec<u8>,
    pub(super) sink: &'a mut dyn FnMut(&[u8]) -> Result<(), String>,
}

#[cfg(not(target_os = "android"))]
impl<'a> StreamingRowOutput<'a> {
    pub(super) fn new(
        width: u32,
        height: u32,
        watermark: Option<PreparedWatermark>,
        sink: &'a mut dyn FnMut(&[u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        let row_bytes = (width as usize)
            .checked_mul(4)
            .ok_or_else(|| format!("Streaming output width {width} exceeds the row size limit"))?;
        let mut scratch = Vec::new();
        if watermark.is_some() {
            scratch
                .try_reserve_exact(row_bytes)
                .map_err(|error| format!("Failed to reserve the watermark row buffer: {error}"))?;
        }
        Ok(Self {
            width,
            height,
            written_rows: 0,
            watermark,
            scratch,
            sink,
        })
    }

    pub(super) fn emit(&mut self, rgba_row: &[u8]) -> Result<(), String> {
        validate_streamed_rgba_row(rgba_row, self.width, self.height, self.written_rows)?;
        let row_intersects_watermark = self
            .watermark
            .as_ref()
            .and_then(|watermark| watermark.row_intersection(self.width, self.written_rows))
            .is_some();

        if row_intersects_watermark {
            self.scratch.clear();
            self.scratch.extend_from_slice(rgba_row);
            if let Some(watermark) = &self.watermark {
                watermark.blend_row(self.width, self.written_rows, &mut self.scratch);
            }
            (self.sink)(&self.scratch)?;
        } else {
            (self.sink)(rgba_row)?;
        }
        self.written_rows += 1;
        Ok(())
    }

    pub(super) fn finish(&self) -> Result<(), String> {
        if self.written_rows != self.height {
            return Err(format!(
                "Streaming transform produced {} rows; expected {}",
                self.written_rows, self.height
            ));
        }
        Ok(())
    }
}

#[cfg(not(target_os = "android"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn transform_streaming_rgba_rows<F>(
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    watermark_settings: Option<&WatermarkSettings>,
    output_sink: &mut dyn FnMut(&[u8]) -> Result<(), String>,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    if source_width == 0 || source_height == 0 || target_width == 0 || target_height == 0 {
        return Err(format!(
            "Streaming transform requires non-zero dimensions; source={}x{}, target={}x{}",
            source_width, source_height, target_width, target_height
        ));
    }

    let watermark = watermark_settings
        .map(|settings| prepare_watermark(target_width, target_height, settings))
        .transpose()?
        .flatten();
    let mut output = StreamingRowOutput::new(target_width, target_height, watermark, output_sink)?;
    let mut source_rows = 0_u32;

    if (source_width, source_height) == (target_width, target_height) {
        let mut input_sink = |row: &[u8]| -> Result<(), String> {
            validate_streamed_rgba_row(row, source_width, source_height, source_rows)?;
            source_rows += 1;
            output.emit(row)
        };
        render_rows(&mut input_sink)?;
    } else {
        // The legacy full-frame path used encoded-space Lanczos3. Keep that
        // semantic while switching storage to zenresize's bounded row ring.
        let resize_config =
            ResizeConfig::builder(source_width, source_height, target_width, target_height)
                .filter(StreamingResizeFilter::Lanczos)
                .format(PixelDescriptor::RGBA8_SRGB)
                .srgb()
                .build();
        let mut resizer = StreamingResize::new(&resize_config);
        let mut input_sink = |row: &[u8]| -> Result<(), String> {
            validate_streamed_rgba_row(row, source_width, source_height, source_rows)?;
            resizer.push_row(row).map_err(|error| {
                format!("Failed to push source row {source_rows} into streaming resize: {error}")
            })?;
            source_rows += 1;
            while let Some(resized_row) = resizer.next_output_row() {
                output.emit(resized_row)?;
            }
            Ok(())
        };
        render_rows(&mut input_sink)?;
        if source_rows != source_height {
            return Err(format!(
                "Streaming transform received {source_rows} source rows; expected {source_height}"
            ));
        }

        resizer.finish();
        while let Some(resized_row) = resizer.next_output_row() {
            output.emit(resized_row)?;
        }
        if !resizer.is_complete() {
            return Err("Streaming resize did not produce a complete output frame".to_string());
        }
    }

    if source_rows != source_height {
        return Err(format!(
            "Streaming transform received {source_rows} source rows; expected {source_height}"
        ));
    }
    output.finish()
}

#[cfg(not(target_os = "android"))]
pub(crate) fn encode_streaming_jpeg<F>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    jpeg_quality: u8,
    embed_color_profile: bool,
    export_exif: Option<&[u8]>,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    if width > u32::from(u16::MAX) || height > u32::from(u16::MAX) {
        return Err(format!(
            "JPEG dimensions {width}x{height} exceed the format limit of 65535 pixels per axis"
        ));
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        let mut compressor = MozjpegCompressor::new(MozjpegColorSpace::JCS_RGB);
        compressor.set_fastest_defaults();
        compressor.set_size(width as usize, height as usize);
        compressor.set_quality(f32::from(jpeg_quality.clamp(1, 100)));
        compressor.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));
        compressor.set_optimize_coding(false);

        let mut encoder = compressor
            .start_compress(output)
            .map_err(|error| format!("Failed to configure streaming JPEG encoder: {error}"))?;
        const ICC_MARKER_OVERHEAD: usize = 14;
        const MAX_MARKER_PAYLOAD: usize = 65_533;
        if embed_color_profile {
            let icc_profile = srgb_v4_profile();
            // mozjpeg 0.10.13 numbers helper-generated ICC chunks from zero, while
            // the JPEG ICC convention and image decoders require one-based indices.
            let icc_chunk_size = MAX_MARKER_PAYLOAD - ICC_MARKER_OVERHEAD;
            let icc_chunk_count = icc_profile.len().div_ceil(icc_chunk_size);
            let icc_chunk_count = u8::try_from(icc_chunk_count)
                .map_err(|_| "Bundled ICC profile requires too many JPEG markers".to_string())?;
            for (chunk_index, chunk) in icc_profile.chunks(icc_chunk_size).enumerate() {
                let mut marker = Vec::with_capacity(ICC_MARKER_OVERHEAD + chunk.len());
                marker.extend_from_slice(b"ICC_PROFILE\0");
                marker.push(
                    u8::try_from(chunk_index + 1)
                        .map_err(|_| "Bundled ICC profile chunk index overflow".to_string())?,
                );
                marker.push(icc_chunk_count);
                marker.extend_from_slice(chunk);
                encoder.write_marker(MozjpegMarker::APP(2), &marker);
            }
        }
        if let Some(tiff_payload) = export_exif {
            const EXIF_HEADER: &[u8] = b"Exif\0\0";
            if tiff_payload.len() > MAX_MARKER_PAYLOAD - EXIF_HEADER.len() {
                return Err(format!(
                    "Encoded EXIF metadata exceeds the JPEG APP1 limit ({} bytes)",
                    tiff_payload.len()
                ));
            }
            let mut marker = Vec::with_capacity(EXIF_HEADER.len() + tiff_payload.len());
            marker.extend_from_slice(EXIF_HEADER);
            marker.extend_from_slice(tiff_payload);
            encoder.write_marker(MozjpegMarker::APP(1), &marker);
        }
        let mut written_rows = 0_u32;
        let mut rgb_row = vec![0_u8; width as usize * 3];
        {
            let mut sink = |rgba_row: &[u8]| -> Result<(), String> {
                validate_streamed_rgba_row(rgba_row, width, height, written_rows)?;
                for (rgba, rgb) in rgba_row.chunks_exact(4).zip(rgb_row.chunks_exact_mut(3)) {
                    rgb.copy_from_slice(&rgba[..3]);
                }
                encoder.write_scanlines(&rgb_row).map_err(|error| {
                    format!("Failed to stream JPEG row {written_rows}: {error}")
                })?;
                written_rows += 1;
                Ok(())
            };
            render_rows(&mut sink)?;
        }
        if written_rows != height {
            return Err(format!(
                "Streaming JPEG received {written_rows} rows; expected {height}"
            ));
        }
        encoder
            .finish()
            .map(|_| ())
            .map_err(|error| format!("Failed to finish streaming JPEG: {error}"))
    }))
    .map_err(|_| "Streaming JPEG encoder aborted while processing the image".to_string())?
}

#[cfg(not(target_os = "android"))]
pub(super) fn encode_streaming_png<F>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    bit_depth: u8,
    embed_color_profile: bool,
    export_exif: Option<&[u8]>,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    let bit_depth = effective_export_bit_depth("png", bit_depth);
    let mut info = PngInfo::with_size(width, height);
    info.color_type = PngColorType::Rgba;
    info.bit_depth = if bit_depth == 16 {
        BitDepth::Sixteen
    } else {
        BitDepth::Eight
    };
    info.icc_profile = embed_color_profile.then(|| Cow::Owned(srgb_v4_profile().to_vec()));
    info.exif_metadata = export_exif.map(|metadata| Cow::Owned(metadata.to_vec()));
    let writer = BufWriter::new(output);
    let encoder = PngStreamEncoder::with_info(writer, info)
        .map_err(|error| format!("Failed to configure streaming PNG encoder: {error}"))?;
    let mut png_writer = encoder
        .write_header()
        .map_err(|error| format!("Failed to write streaming PNG header: {error}"))?;
    let mut stream = png_writer
        .stream_writer_with_size(64 * 1024)
        .map_err(|error| format!("Failed to start streaming PNG encoder: {error}"))?;
    let mut rgba16_row = if bit_depth == 16 {
        vec![
            0_u8;
            (width as usize)
                .checked_mul(8)
                .ok_or_else(|| format!("PNG width {width} exceeds the row size limit"))?
        ]
    } else {
        Vec::new()
    };
    let mut written_rows = 0_u32;
    {
        let mut sink = |rgba_row: &[u8]| -> Result<(), String> {
            validate_streamed_rgba_row(rgba_row, width, height, written_rows)?;
            let row = if bit_depth == 16 {
                for (index, sample) in rgba_row.iter().copied().enumerate() {
                    let bytes = (u16::from(sample) * 257).to_be_bytes();
                    rgba16_row[index * 2..index * 2 + 2].copy_from_slice(&bytes);
                }
                rgba16_row.as_slice()
            } else {
                rgba_row
            };
            stream
                .write_all(row)
                .map_err(|error| format!("Failed to stream PNG row {written_rows}: {error}"))?;
            written_rows += 1;
            Ok(())
        };
        render_rows(&mut sink)?;
    }
    if written_rows != height {
        return Err(format!(
            "Streaming PNG received {written_rows} rows; expected {height}"
        ));
    }
    stream
        .finish()
        .map_err(|error| format!("Failed to finish streaming PNG data: {error}"))?;
    png_writer
        .finish()
        .map_err(|error| format!("Failed to finish streaming PNG: {error}"))
}
