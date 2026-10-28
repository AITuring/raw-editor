//! Read-back validation of written image-stack files.
//!
//! Decodes the written file again and checks dimensions, colour type and the
//! embedded profile before the output is published.

use image::ColorType;
use image::ImageDecoder;
use image::codecs::jpeg::JpegDecoder;
use image::codecs::png::PngDecoder;
use image::codecs::tiff::TiffDecoder;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use super::ImageStackOutputFormat;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SavedOutputMetadata {
    pub(super) format: String,
    pub(super) bit_depth: u8,
    pub(super) alpha_preserved: bool,
    pub(super) icc: String,
}

fn saved_output_metadata(
    output_format: ImageStackOutputFormat,
    color_type: ColorType,
    icc_profile: Option<&[u8]>,
) -> Result<SavedOutputMetadata, String> {
    let channels = color_type.channel_count();
    if channels == 0 {
        return Err(format!(
            "Saved image-stack {} has no channels.",
            output_format.label()
        ));
    }
    let bits_per_channel = color_type.bits_per_pixel() / u16::from(channels);
    let bit_depth = u8::try_from(bits_per_channel).map_err(|_| {
        format!(
            "Saved image-stack {} has an unsupported channel bit depth ({} bits).",
            output_format.label(),
            bits_per_channel
        )
    })?;
    let icc = match icc_profile {
        Some(profile) if profile == crate::color_management::srgb_v4_profile() => "sRGB",
        Some(_) => "embedded",
        None => "none",
    };
    Ok(SavedOutputMetadata {
        format: output_format.label().to_string(),
        bit_depth,
        alpha_preserved: color_type.has_alpha(),
        icc: icc.to_string(),
    })
}

pub(super) type ImageStackDecoderParts = ((u32, u32), ColorType, Option<Vec<u8>>);

fn image_stack_decoder_parts<D: ImageDecoder>(
    mut decoder: D,
    output_format: ImageStackOutputFormat,
) -> Result<ImageStackDecoderParts, String> {
    let dimensions = decoder.dimensions();
    let color_type = decoder.color_type();
    let icc_profile = decoder.icc_profile().map_err(|error| {
        format!(
            "Failed to validate the image-stack {} color profile: {error}",
            output_format.label()
        )
    })?;
    Ok((dimensions, color_type, icc_profile))
}

fn validate_image_stack_decoder<D: ImageDecoder>(
    decoder: D,
    dimensions: (u32, u32),
    output_format: ImageStackOutputFormat,
    bit_depth: u8,
    alpha: bool,
    expect_color_profile: bool,
) -> Result<SavedOutputMetadata, String> {
    let (saved_dimensions, color_type, saved_profile) =
        image_stack_decoder_parts(decoder, output_format)?;
    if saved_dimensions != dimensions {
        return Err(format!(
            "Saved image-stack dimensions do not match the result (expected {}×{}, found {}×{}).",
            dimensions.0, dimensions.1, saved_dimensions.0, saved_dimensions.1
        ));
    }
    if color_type != output_format.encoded_color_type(bit_depth, alpha) {
        return Err(format!(
            "Saved image-stack {} has an unexpected pixel format ({:?}).",
            output_format.label(),
            color_type
        ));
    }
    if expect_color_profile {
        if saved_profile.as_deref() != Some(crate::color_management::srgb_v4_profile()) {
            return Err(format!(
                "The saved image-stack {} is missing its sRGB color profile.",
                output_format.label()
            ));
        }
    } else if saved_profile.is_some() {
        return Err(format!(
            "The saved image-stack {} contains a color profile even though embedding was disabled.",
            output_format.label()
        ));
    }
    saved_output_metadata(output_format, color_type, saved_profile.as_deref())
}

pub(super) fn validate_image_stack_output(
    output_path: &Path,
    dimensions: (u32, u32),
    output_format: ImageStackOutputFormat,
    bit_depth: u8,
    alpha: bool,
    expect_color_profile: bool,
) -> Result<SavedOutputMetadata, String> {
    let open = || {
        File::open(output_path)
            .map(BufReader::new)
            .map_err(|error| {
                format!(
                    "Failed to reopen the saved image-stack {}: {error}",
                    output_format.label()
                )
            })
    };
    match output_format {
        ImageStackOutputFormat::Tiff => validate_image_stack_decoder(
            TiffDecoder::new(open()?).map_err(|error| {
                format!("Failed to validate the saved image-stack TIFF: {error}")
            })?,
            dimensions,
            output_format,
            bit_depth,
            alpha,
            expect_color_profile,
        ),
        ImageStackOutputFormat::Png => validate_image_stack_decoder(
            PngDecoder::new(open()?).map_err(|error| {
                format!("Failed to validate the saved image-stack PNG: {error}")
            })?,
            dimensions,
            output_format,
            bit_depth,
            alpha,
            expect_color_profile,
        ),
        ImageStackOutputFormat::Jpeg => validate_image_stack_decoder(
            JpegDecoder::new(open()?).map_err(|error| {
                format!("Failed to validate the saved image-stack JPEG: {error}")
            })?,
            dimensions,
            output_format,
            bit_depth,
            alpha,
            expect_color_profile,
        ),
    }
}

pub(super) fn read_image_stack_output_metadata(
    output_path: &Path,
    output_format: ImageStackOutputFormat,
) -> Result<SavedOutputMetadata, String> {
    let open = || {
        File::open(output_path)
            .map(BufReader::new)
            .map_err(|error| {
                format!(
                    "Failed to reopen the saved image-stack {}: {error}",
                    output_format.label()
                )
            })
    };
    let (_, color_type, icc_profile) = match output_format {
        ImageStackOutputFormat::Tiff => image_stack_decoder_parts(
            TiffDecoder::new(open()?)
                .map_err(|error| format!("Failed to read the saved image-stack TIFF: {error}"))?,
            output_format,
        )?,
        ImageStackOutputFormat::Png => image_stack_decoder_parts(
            PngDecoder::new(open()?)
                .map_err(|error| format!("Failed to read the saved image-stack PNG: {error}"))?,
            output_format,
        )?,
        ImageStackOutputFormat::Jpeg => image_stack_decoder_parts(
            JpegDecoder::new(open()?)
                .map_err(|error| format!("Failed to read the saved image-stack JPEG: {error}"))?,
            output_format,
        )?,
    };
    saved_output_metadata(output_format, color_type, icc_profile.as_deref())
}
