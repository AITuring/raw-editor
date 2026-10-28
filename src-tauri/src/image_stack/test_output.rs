//! Test-only image-stack output helpers.
//!
//! Fixed-setting writers used by this module's tests and by acceptance
//! harnesses elsewhere in the crate.

use crate::export_processing::ExportSettings;
use image::DynamicImage;
use std::io::Seek;
use std::io::Write;
use std::path::Path;

use super::{
    ImageStackOutputFormat, encode_srgb_image_stack, write_image_stack_output_with_settings,
};

#[cfg(test)]
const IMAGE_STACK_JPEG_QUALITY: u8 = 95;

/// Image stacking works on display-referred samples after RAW development. Freeze
/// those samples once at 16-bit precision so preview and export cannot diverge by
/// independently interpreting a floating-point TIFF as linear RGB.
/// The opaque canonical form: production canonicalises through
/// [`canonicalize_image_stack_result_with_coverage`]; tests and the gate
/// harness use this RGB-only form.
#[cfg(test)]
pub(crate) fn canonicalize_image_stack_result(image: DynamicImage) -> DynamicImage {
    DynamicImage::ImageRgb16(image.to_rgb16())
}

#[cfg(test)]
pub(super) fn encode_srgb_tiff<W: Write + Seek>(
    image: &DynamicImage,
    writer: W,
) -> Result<(), String> {
    encode_srgb_image_stack(
        image,
        writer,
        ImageStackOutputFormat::Tiff,
        95,
        16,
        true,
        None,
    )
}

#[cfg(test)]
pub(super) fn default_image_stack_export_settings() -> ExportSettings {
    ExportSettings {
        bit_depth: 16,
        jpeg_quality: IMAGE_STACK_JPEG_QUALITY,
        resize: None,
        keep_metadata: false,
        metadata_overrides: None,
        preserve_timestamps: false,
        strip_gps: true,
        embed_color_profile: true,
        filename_template: None,
        watermark: None,
        export_masks: false,
        preserve_folders: false,
    }
}

#[cfg(test)]
pub(super) fn write_image_stack_output(
    image: &DynamicImage,
    output_path: &Path,
    output_format: ImageStackOutputFormat,
) -> Result<(), String> {
    write_image_stack_output_with_settings(
        image,
        output_path,
        output_format,
        &default_image_stack_export_settings(),
        "",
    )
}

#[cfg(test)]
pub(crate) fn write_srgb_tiff(image: &DynamicImage, output_path: &Path) -> Result<(), String> {
    write_image_stack_output(image, output_path, ImageStackOutputFormat::Tiff)
}

#[cfg(test)]
pub(crate) fn write_srgb_jpeg(image: &DynamicImage, output_path: &Path) -> Result<(), String> {
    write_image_stack_output(image, output_path, ImageStackOutputFormat::Jpeg)
}
