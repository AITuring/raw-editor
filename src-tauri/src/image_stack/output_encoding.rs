//! Encoding the canonical image-stack result.
//!
//! Canonicalises the stitched image (and optional coverage alpha) and encodes
//! it as sRGB TIFF/PNG or streams it as JPEG.

use crate::export_processing::ExportSettings;
use image::DynamicImage;
use image::ImageEncoder;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::tiff::TiffEncoder;
use std::fs::File;
use std::io::BufWriter;
use std::io::Seek;
use std::io::Write;

use super::ImageStackOutputFormat;

/// The canonical result with the Coverage_Mask as alpha (需求 10.3 / 10.6):
/// uncovered pixels fully transparent, covered pixels fully opaque, and the
/// colour channels exactly those of [`canonicalize_image_stack_result`]. A
/// missing or mismatched mask keeps the opaque RGB canonical form.
pub(crate) fn canonicalize_image_stack_result_with_coverage(
    image: DynamicImage,
    coverage: Option<&image::GrayImage>,
) -> DynamicImage {
    let rgb = image.to_rgb16();
    let Some(coverage) = coverage.filter(|mask| mask.dimensions() == rgb.dimensions()) else {
        return DynamicImage::ImageRgb16(rgb);
    };
    let rgba = image::ImageBuffer::from_fn(rgb.width(), rgb.height(), |x, y| {
        let [red, green, blue] = rgb.get_pixel(x, y).0;
        let alpha = if coverage.get_pixel(x, y)[0] == 0 {
            0
        } else {
            u16::MAX
        };
        image::Rgba([red, green, blue, alpha])
    });
    DynamicImage::ImageRgba16(rgba)
}

/// The pixels a lossless format stores: RGBA when the result carries its
/// Coverage_Mask as alpha, RGB otherwise; colour channels are converted the
/// same way either way, so writing alpha never changes them (需求 10.6).
fn lossless_pixels(image: &DynamicImage, bit_depth: u8) -> DynamicImage {
    match (bit_depth == 16, image.color().has_alpha()) {
        (true, true) => DynamicImage::ImageRgba16(image.to_rgba16()),
        (true, false) => DynamicImage::ImageRgb16(image.to_rgb16()),
        (false, true) => DynamicImage::ImageRgba8(image.to_rgba8()),
        (false, false) => DynamicImage::ImageRgb8(image.to_rgb8()),
    }
}

pub(super) fn encode_srgb_image_stack<W: Write + Seek>(
    image: &DynamicImage,
    writer: W,
    output_format: ImageStackOutputFormat,
    jpeg_quality: u8,
    bit_depth: u8,
    embed_color_profile: bool,
    export_exif: Option<&[u8]>,
) -> Result<(), String> {
    let bit_depth = crate::export_processing::effective_export_bit_depth(
        output_format.canonical_extension(),
        bit_depth,
    );
    let profile = crate::color_management::srgb_v4_profile().to_vec();
    match output_format {
        ImageStackOutputFormat::Tiff => {
            let mut encoder = TiffEncoder::new(writer);
            if embed_color_profile {
                encoder.set_icc_profile(profile).map_err(|error| {
                    format!("Failed to attach the image-stack TIFF color profile: {error}")
                })?;
            }
            let image_to_encode = lossless_pixels(image, bit_depth);
            image_to_encode
                .write_with_encoder(encoder)
                .map_err(|error| format!("Failed to encode image-stack TIFF: {error}"))
        }
        ImageStackOutputFormat::Png => {
            let mut encoder = PngEncoder::new(writer);
            if embed_color_profile {
                encoder.set_icc_profile(profile).map_err(|error| {
                    format!("Failed to attach the image-stack PNG color profile: {error}")
                })?;
            }
            if let Some(exif) = export_exif {
                encoder.set_exif_metadata(exif.to_vec()).map_err(|error| {
                    format!("Failed to attach image-stack PNG metadata: {error}")
                })?;
            }
            let image_to_encode = lossless_pixels(image, bit_depth);
            image_to_encode
                .write_with_encoder(encoder)
                .map_err(|error| format!("Failed to encode image-stack PNG: {error}"))
        }
        ImageStackOutputFormat::Jpeg => {
            let mut encoder = JpegEncoder::new_with_quality(writer, jpeg_quality.clamp(1, 100));
            if embed_color_profile {
                encoder.set_icc_profile(profile).map_err(|error| {
                    format!("Failed to attach the image-stack JPEG color profile: {error}")
                })?;
            }
            if let Some(exif) = export_exif {
                encoder.set_exif_metadata(exif.to_vec()).map_err(|error| {
                    format!("Failed to attach image-stack JPEG metadata: {error}")
                })?;
            }
            encoder
                .encode_image(&image.to_rgb8())
                .map_err(|error| format!("Failed to encode image-stack JPEG: {error}"))
        }
    }
}

#[cfg(not(target_os = "android"))]
fn encode_srgb_jpeg_streaming(
    image: &DynamicImage,
    output: &mut File,
    jpeg_quality: u8,
    embed_color_profile: bool,
    export_exif: Option<&[u8]>,
) -> Result<(), String> {
    // JPEG has no alpha (需求 10.10 notices this before writing): drop the
    // Coverage_Mask channel, keeping the colour samples unchanged.
    let without_alpha;
    let rgb16 = match image.as_rgb16() {
        Some(rgb16) => rgb16,
        None if image.color().has_alpha() => {
            without_alpha = image.to_rgb16();
            &without_alpha
        }
        None => return Err("The canonical image-stack result is not RGB16.".to_string()),
    };
    let (width, height) = rgb16.dimensions();
    let source_row_samples = (width as usize)
        .checked_mul(3)
        .ok_or_else(|| "The image-stack JPEG row is too wide to encode.".to_string())?;
    let rgba_row_bytes = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| "The image-stack JPEG row is too wide to encode.".to_string())?;
    let mut rgba_row = vec![0_u8; rgba_row_bytes];

    crate::export_processing::encode_streaming_jpeg(
        output,
        width,
        height,
        jpeg_quality,
        embed_color_profile,
        export_exif,
        |sink| {
            for source_row in rgb16.as_raw().chunks_exact(source_row_samples) {
                for (source, target) in source_row.chunks_exact(3).zip(rgba_row.chunks_exact_mut(4))
                {
                    target[0] = ((u32::from(source[0]) + 128) / 257) as u8;
                    target[1] = ((u32::from(source[1]) + 128) / 257) as u8;
                    target[2] = ((u32::from(source[2]) + 128) / 257) as u8;
                    target[3] = 255;
                }
                sink(&rgba_row)?;
            }
            Ok(())
        },
    )
}

pub(super) fn encode_image_stack_file(
    image: &DynamicImage,
    output: &mut File,
    output_format: ImageStackOutputFormat,
    export_settings: &ExportSettings,
    source_path: &str,
) -> Result<(), String> {
    let bit_depth = crate::export_processing::effective_export_bit_depth(
        output_format.canonical_extension(),
        export_settings.bit_depth,
    );
    let export_exif = if output_format == ImageStackOutputFormat::Tiff {
        None
    } else {
        crate::exif_processing::export_metadata_tiff_payload(
            source_path,
            output_format.canonical_extension(),
            export_settings.keep_metadata,
            export_settings.strip_gps,
            export_settings.metadata_overrides.as_ref(),
        )?
    };

    #[cfg(not(target_os = "android"))]
    if output_format == ImageStackOutputFormat::Jpeg {
        encode_srgb_jpeg_streaming(
            image,
            output,
            export_settings.jpeg_quality,
            export_settings.embed_color_profile,
            export_exif.as_deref(),
        )?;
        return output
            .flush()
            .map_err(|error| format!("Failed to finish writing the image-stack JPEG: {error}"));
    }

    #[cfg(not(target_os = "android"))]
    if output_format == ImageStackOutputFormat::Tiff {
        let metadata = crate::exif_processing::export_metadata_for_streaming_tiff(
            source_path,
            export_settings.keep_metadata,
            export_settings.strip_gps,
            export_settings.metadata_overrides.as_ref(),
        )?;
        // The Coverage_Mask travels as alpha; the colour samples are the same
        // either way (需求 10.6).
        if image.color().has_alpha() {
            if bit_depth == 16 {
                let rgba16 = image.to_rgba16();
                return crate::export_processing::encode_rgba16_tiff_with_metadata(
                    output,
                    rgba16.width(),
                    rgba16.height(),
                    rgba16.as_raw(),
                    export_settings.embed_color_profile,
                    metadata.as_ref(),
                );
            }
            let rgba8 = image.to_rgba8();
            return crate::export_processing::encode_rgba8_tiff_with_metadata(
                output,
                rgba8.width(),
                rgba8.height(),
                rgba8.as_raw(),
                export_settings.embed_color_profile,
                metadata.as_ref(),
            );
        }
        let rgb16 = image
            .as_rgb16()
            .ok_or_else(|| "The canonical image-stack result is not RGB16.".to_string())?;
        if bit_depth == 16 {
            return crate::export_processing::encode_rgb16_tiff_with_metadata(
                output,
                rgb16.width(),
                rgb16.height(),
                rgb16.as_raw(),
                export_settings.embed_color_profile,
                metadata.as_ref(),
            );
        }
        let rgb8 = image.to_rgb8();
        return crate::export_processing::encode_rgb8_tiff_with_metadata(
            output,
            rgb8.width(),
            rgb8.height(),
            rgb8.as_raw(),
            export_settings.embed_color_profile,
            metadata.as_ref(),
        );
    }

    let mut writer = BufWriter::new(output);
    encode_srgb_image_stack(
        image,
        &mut writer,
        output_format,
        export_settings.jpeg_quality,
        bit_depth,
        export_settings.embed_color_profile,
        export_exif.as_deref(),
    )?;
    writer.flush().map_err(|error| {
        format!(
            "Failed to finish writing the image-stack {}: {error}",
            output_format.label()
        )
    })
}
