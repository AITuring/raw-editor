//! In-memory export encoders.
//!
//! Encodes a finished image to JPEG, PNG, TIFF, WebP, AVIF or JPEG XL bytes with
//! the optional sRGB profile and embeds ICC chunks into WebP.

use crate::color_management::srgb_v4_profile;
use image::DynamicImage;
use image::GenericImageView;
use image::GrayImage;
use image::ImageEncoder;
use image::ImageFormat;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::tiff::TiffEncoder;
use jxl_encoder::ImageMetadata as JxlImageMetadata;
use jxl_encoder::LosslessConfig;
use jxl_encoder::LossyConfig;
use jxl_encoder::PixelLayout;
use jxl_encoder::api::calibrated_jxl_quality;
use jxl_encoder::api::quality_to_distance;
use std::io::Cursor;

#[cfg(test)]
use super::default_export_bit_depth_for_format;
use super::effective_export_bit_depth;

#[cfg(target_os = "android")]
pub fn mime_type_for_extension(extension: &str) -> &'static str {
    match extension {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "tif" | "tiff" => "image/tiff",
        "jxl" => "image/jxl",
        _ => "application/octet-stream",
    }
}

pub(super) fn encode_grayscale_to_png(bitmap: &GrayImage) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    bitmap
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(buf)
}

#[cfg(test)]
pub(crate) fn encode_image_to_bytes(
    image: &DynamicImage,
    output_format: &str,
    jpeg_quality: u8,
) -> Result<Vec<u8>, String> {
    encode_image_to_bytes_with_profile(
        image,
        output_format,
        jpeg_quality,
        true,
        default_export_bit_depth_for_format(output_format),
    )
}

pub(super) fn encode_image_to_bytes_with_profile(
    image: &DynamicImage,
    output_format: &str,
    jpeg_quality: u8,
    embed_color_profile: bool,
    bit_depth: u8,
) -> Result<Vec<u8>, String> {
    let mut image_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut image_bytes);

    match output_format.to_lowercase().as_str() {
        "jxl" => {
            let (width, height) = image.dimensions();
            let has_alpha = image.color().has_alpha();
            let metadata = if embed_color_profile {
                JxlImageMetadata::new().with_icc_profile(srgb_v4_profile())
            } else {
                JxlImageMetadata::new()
            };

            let jxl_data = if jpeg_quality == 100 {
                if has_alpha {
                    let rgba = image.to_rgba8();
                    LosslessConfig::new()
                        .encode_request(width, height, PixelLayout::Rgba8)
                        .with_metadata(&metadata)
                        .encode(rgba.as_raw())
                        .map_err(|e| format!("Failed to encode lossless JXL: {}", e))?
                } else {
                    let rgb = image.to_rgb8();
                    LosslessConfig::new()
                        .encode_request(width, height, PixelLayout::Rgb8)
                        .with_metadata(&metadata)
                        .encode(rgb.as_raw())
                        .map_err(|e| format!("Failed to encode lossless JXL: {}", e))?
                }
            } else {
                let jxl_quality = calibrated_jxl_quality(jpeg_quality as f32);
                let distance = quality_to_distance(jxl_quality);

                if has_alpha {
                    let rgba = image.to_rgba8();
                    LossyConfig::new(distance)
                        .encode_request(width, height, PixelLayout::Rgba8)
                        .with_metadata(&metadata)
                        .encode(rgba.as_raw())
                        .map_err(|e| format!("Failed to encode lossy JXL: {}", e))?
                } else {
                    let rgb = image.to_rgb8();
                    LossyConfig::new(distance)
                        .encode_request(width, height, PixelLayout::Rgb8)
                        .with_metadata(&metadata)
                        .encode(rgb.as_raw())
                        .map_err(|e| format!("Failed to encode lossy JXL: {}", e))?
                }
            };

            return Ok(jxl_data);
        }
        "webp" => {
            let encoder = webp::Encoder::from_image(image)
                .map_err(|_| "Failed to create WebP encoder".to_string())?;
            let webp_mem = encoder.encode(jpeg_quality as f32);
            return if embed_color_profile {
                embed_icc_in_webp(
                    webp_mem.as_ref(),
                    srgb_v4_profile(),
                    image.width(),
                    image.height(),
                    image.color().has_alpha(),
                )
            } else {
                Ok(webp_mem.to_vec())
            };
        }
        "jpg" | "jpeg" => {
            let rgb_image = image.to_rgb8();
            let mut encoder = JpegEncoder::new_with_quality(&mut cursor, jpeg_quality);
            if embed_color_profile {
                encoder
                    .set_icc_profile(srgb_v4_profile().to_vec())
                    .map_err(|e| format!("Failed to attach JPEG ICC profile: {e}"))?;
            }
            rgb_image
                .write_with_encoder(encoder)
                .map_err(|e| e.to_string())?;
        }
        "png" => {
            let image_to_encode = if effective_export_bit_depth("png", bit_depth) == 16 {
                if image.color().has_alpha() {
                    DynamicImage::ImageRgba16(image.to_rgba16())
                } else {
                    DynamicImage::ImageRgb16(image.to_rgb16())
                }
            } else {
                if image.color().has_alpha() {
                    DynamicImage::ImageRgba8(image.to_rgba8())
                } else {
                    DynamicImage::ImageRgb8(image.to_rgb8())
                }
            };

            let mut encoder = PngEncoder::new(&mut cursor);
            if embed_color_profile {
                encoder
                    .set_icc_profile(srgb_v4_profile().to_vec())
                    .map_err(|e| format!("Failed to attach PNG ICC profile: {e}"))?;
            }
            image_to_encode
                .write_with_encoder(encoder)
                .map_err(|e| e.to_string())?;
        }
        "tif" | "tiff" => {
            let mut encoder = TiffEncoder::new(&mut cursor);
            if embed_color_profile {
                encoder
                    .set_icc_profile(srgb_v4_profile().to_vec())
                    .map_err(|e| format!("Failed to attach TIFF ICC profile: {e}"))?;
            }
            if effective_export_bit_depth("tiff", bit_depth) == 16 {
                DynamicImage::ImageRgb16(image.to_rgb16())
            } else {
                DynamicImage::ImageRgb8(image.to_rgb8())
            }
            .write_with_encoder(encoder)
            .map_err(|e| e.to_string())?;
        }
        "avif" => {
            image
                .write_to(&mut cursor, image::ImageFormat::Avif)
                .map_err(|e| e.to_string())?;
        }
        _ => return Err(format!("Unsupported file format: {}", output_format)),
    };
    Ok(image_bytes)
}

fn write_webp_chunk(output: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    output.extend_from_slice(fourcc);
    output.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    output.extend_from_slice(payload);
    if !payload.len().is_multiple_of(2) {
        output.push(0);
    }
}

fn embed_icc_in_webp(
    encoded: &[u8],
    icc_profile: &[u8],
    width: u32,
    height: u32,
    has_alpha: bool,
) -> Result<Vec<u8>, String> {
    if encoded.len() < 12 || &encoded[0..4] != b"RIFF" || &encoded[8..12] != b"WEBP" {
        return Err("WebP encoder returned an invalid RIFF container".to_string());
    }
    if width == 0 || height == 0 || width > 0x01_00_00_00 || height > 0x01_00_00_00 {
        return Err("WebP dimensions cannot be represented by a VP8X header".to_string());
    }

    let mut output = Vec::with_capacity(encoded.len() + icc_profile.len() + 32);
    output.extend_from_slice(b"RIFF\0\0\0\0WEBP");

    let mut offset = 12;
    let mut wrote_icc = false;
    let mut found_vp8x = false;
    while offset + 8 <= encoded.len() {
        let fourcc: [u8; 4] = encoded[offset..offset + 4]
            .try_into()
            .map_err(|_| "Invalid WebP chunk identifier".to_string())?;
        let chunk_len = u32::from_le_bytes(
            encoded[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| "Invalid WebP chunk length".to_string())?,
        ) as usize;
        let padded_len = chunk_len + (chunk_len % 2);
        let chunk_end = offset
            .checked_add(8 + padded_len)
            .ok_or_else(|| "WebP chunk size overflow".to_string())?;
        if chunk_end > encoded.len() {
            return Err("WebP chunk extends past the RIFF container".to_string());
        }

        if &fourcc == b"ICCP" {
            if found_vp8x && !wrote_icc {
                write_webp_chunk(&mut output, b"ICCP", icc_profile);
                wrote_icc = true;
            }
        } else if &fourcc == b"VP8X" {
            if chunk_len != 10 {
                return Err("WebP VP8X chunk has an invalid length".to_string());
            }
            let mut payload = encoded[offset + 8..offset + 18].to_vec();
            payload[0] |= 0x20;
            write_webp_chunk(&mut output, b"VP8X", &payload);
            found_vp8x = true;
            if !wrote_icc {
                write_webp_chunk(&mut output, b"ICCP", icc_profile);
                wrote_icc = true;
            }
        } else {
            output.extend_from_slice(&encoded[offset..chunk_end]);
        }
        offset = chunk_end;
    }

    if offset != encoded.len() {
        return Err("WebP container has trailing partial chunk data".to_string());
    }

    if !found_vp8x {
        let image_chunks = output.split_off(12);
        let mut vp8x = [0u8; 10];
        vp8x[0] = 0x20 | if has_alpha { 0x10 } else { 0 };
        let width_minus_one = width - 1;
        let height_minus_one = height - 1;
        vp8x[4..7].copy_from_slice(&width_minus_one.to_le_bytes()[..3]);
        vp8x[7..10].copy_from_slice(&height_minus_one.to_le_bytes()[..3]);
        write_webp_chunk(&mut output, b"VP8X", &vp8x);
        write_webp_chunk(&mut output, b"ICCP", icc_profile);
        output.extend_from_slice(&image_chunks);
    } else if !wrote_icc {
        return Err("Failed to insert the WebP ICC profile".to_string());
    }

    let riff_size = u32::try_from(output.len().saturating_sub(8))
        .map_err(|_| "WebP output is too large for RIFF".to_string())?;
    output[4..8].copy_from_slice(&riff_size.to_le_bytes());
    Ok(output)
}
