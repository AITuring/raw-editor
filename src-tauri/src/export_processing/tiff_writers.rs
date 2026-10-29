//! Whole-image TIFF and BigTIFF writers with metadata.
//!
//! Narrow RGB/RGBA 8/16-bit entry points that switch to BigTIFF before the
//! standard 32-bit offsets could overflow.

use crate::color_management::srgb_v4_profile;
#[cfg(not(target_os = "android"))]
use little_exif::ifd::ExifTagGroup;
#[cfg(not(target_os = "android"))]
use little_exif::metadata::Metadata as ExifMetadata;
use std::fs;
#[cfg(not(target_os = "android"))]
use tiff::encoder::TiffEncoder as StreamingTiffEncoder;
#[cfg(not(target_os = "android"))]
use tiff::encoder::TiffKindBig;
#[cfg(not(target_os = "android"))]
use tiff::encoder::TiffValue;
#[cfg(not(target_os = "android"))]
use tiff::encoder::colortype::ColorType as TiffColorType;
#[cfg(not(target_os = "android"))]
use tiff::encoder::colortype::RGB8;
#[cfg(not(target_os = "android"))]
use tiff::encoder::colortype::RGB16;
#[cfg(not(target_os = "android"))]
use tiff::encoder::colortype::RGBA8;
#[cfg(not(target_os = "android"))]
use tiff::encoder::colortype::RGBA16;
#[cfg(not(target_os = "android"))]
use tiff::tags::Tag as TiffTag;

#[cfg(not(target_os = "android"))]
use super::{has_writable_tiff_metadata_group, write_tiff_metadata_group};

#[cfg(not(target_os = "android"))]
pub(crate) fn encode_rgb16_tiff_with_metadata(
    output: &mut fs::File,
    width: u32,
    height: u32,
    rgb16: &[u16],
    embed_color_profile: bool,
    export_metadata: Option<&ExifMetadata>,
) -> Result<(), String> {
    if should_use_bigtiff(width, height, 3, 2) {
        return encode_rgb16_bigtiff_with_metadata(
            output,
            width,
            height,
            rgb16,
            embed_color_profile,
            export_metadata,
        );
    }
    encode_rgb_tiff_with_metadata::<RGB16>(
        output,
        width,
        height,
        rgb16,
        embed_color_profile,
        export_metadata,
    )
}

#[cfg(not(target_os = "android"))]
pub(crate) fn encode_rgb8_tiff_with_metadata(
    output: &mut fs::File,
    width: u32,
    height: u32,
    rgb8: &[u8],
    embed_color_profile: bool,
    export_metadata: Option<&ExifMetadata>,
) -> Result<(), String> {
    if should_use_bigtiff(width, height, 3, 1) {
        return encode_rgb8_bigtiff_with_metadata(
            output,
            width,
            height,
            rgb8,
            embed_color_profile,
            export_metadata,
        );
    }
    encode_rgb_tiff_with_metadata::<RGB8>(
        output,
        width,
        height,
        rgb8,
        embed_color_profile,
        export_metadata,
    )
}

/// RGBA16 variant of [`encode_rgb16_tiff_with_metadata`]: the fourth sample is
/// the unassociated alpha written from the Coverage_Mask (需求 10.6).
#[cfg(not(target_os = "android"))]
pub(crate) fn encode_rgba16_tiff_with_metadata(
    output: &mut fs::File,
    width: u32,
    height: u32,
    rgba16: &[u16],
    embed_color_profile: bool,
    export_metadata: Option<&ExifMetadata>,
) -> Result<(), String> {
    if should_use_bigtiff(width, height, 4, 2) {
        return encode_rgba16_bigtiff_with_metadata(
            output,
            width,
            height,
            rgba16,
            embed_color_profile,
            export_metadata,
        );
    }
    encode_rgb_tiff_with_metadata::<RGBA16>(
        output,
        width,
        height,
        rgba16,
        embed_color_profile,
        export_metadata,
    )
}

/// RGBA8 variant of [`encode_rgb8_tiff_with_metadata`].
#[cfg(not(target_os = "android"))]
pub(crate) fn encode_rgba8_tiff_with_metadata(
    output: &mut fs::File,
    width: u32,
    height: u32,
    rgba8: &[u8],
    embed_color_profile: bool,
    export_metadata: Option<&ExifMetadata>,
) -> Result<(), String> {
    if should_use_bigtiff(width, height, 4, 1) {
        return encode_rgba8_bigtiff_with_metadata(
            output,
            width,
            height,
            rgba8,
            embed_color_profile,
            export_metadata,
        );
    }
    encode_rgb_tiff_with_metadata::<RGBA8>(
        output,
        width,
        height,
        rgba8,
        embed_color_profile,
        export_metadata,
    )
}

/// Standard TIFF stores strip/tile offsets in 32-bit fields. Keep the normal
/// encoder for ordinary exports, but switch to BigTIFF before the uncompressed
/// sample payload could approach that limit. The outer image-stack publisher
/// still writes beside the destination, syncs, validates and atomically
/// renames the completed file.
#[cfg(not(target_os = "android"))]
pub(crate) fn should_use_bigtiff(
    width: u32,
    height: u32,
    samples_per_pixel: usize,
    bytes_per_sample: usize,
) -> bool {
    const STANDARD_TIFF_OFFSET_BUDGET: u64 = (u32::MAX as u64) - (64 * 1024 * 1024);
    let payload = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(samples_per_pixel as u64))
        .and_then(|samples| samples.checked_mul(bytes_per_sample as u64));
    payload.is_none_or(|bytes| bytes > STANDARD_TIFF_OFFSET_BUDGET)
}

/// 16-bit RGB BigTIFF writer. This intentionally exposes the same narrow API
/// as the standard TIFF writer so callers cannot accidentally pass a full
/// DynamicImage through a second conversion path.
#[cfg(not(target_os = "android"))]
pub(crate) fn encode_rgb16_bigtiff_with_metadata(
    output: &mut fs::File,
    width: u32,
    height: u32,
    rgb16: &[u16],
    embed_color_profile: bool,
    _export_metadata: Option<&ExifMetadata>,
) -> Result<(), String> {
    encode_bigtiff_with_metadata::<RGB16>(output, width, height, rgb16, embed_color_profile)
}

/// 8-bit RGB BigTIFF writer for explicitly downgraded exports whose pixel
/// count itself exceeds the standard TIFF offset budget.
#[cfg(not(target_os = "android"))]
pub(crate) fn encode_rgb8_bigtiff_with_metadata(
    output: &mut fs::File,
    width: u32,
    height: u32,
    rgb8: &[u8],
    embed_color_profile: bool,
    _export_metadata: Option<&ExifMetadata>,
) -> Result<(), String> {
    encode_bigtiff_with_metadata::<RGB8>(output, width, height, rgb8, embed_color_profile)
}

/// 16-bit RGBA BigTIFF writer. Alpha is written as an unassociated fourth
/// sample, preserving the Coverage_Mask semantics of the standard path.
#[cfg(not(target_os = "android"))]
pub(crate) fn encode_rgba16_bigtiff_with_metadata(
    output: &mut fs::File,
    width: u32,
    height: u32,
    rgba16: &[u16],
    embed_color_profile: bool,
    _export_metadata: Option<&ExifMetadata>,
) -> Result<(), String> {
    encode_bigtiff_with_metadata::<RGBA16>(output, width, height, rgba16, embed_color_profile)
}

/// 8-bit RGBA BigTIFF writer preserving the Coverage_Mask alpha channel.
#[cfg(not(target_os = "android"))]
pub(crate) fn encode_rgba8_bigtiff_with_metadata(
    output: &mut fs::File,
    width: u32,
    height: u32,
    rgba8: &[u8],
    embed_color_profile: bool,
    _export_metadata: Option<&ExifMetadata>,
) -> Result<(), String> {
    encode_bigtiff_with_metadata::<RGBA8>(output, width, height, rgba8, embed_color_profile)
}

#[cfg(not(target_os = "android"))]
fn encode_bigtiff_with_metadata<C>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    data: &[C::Inner],
    embed_color_profile: bool,
) -> Result<(), String>
where
    C: TiffColorType,
    [C::Inner]: TiffValue,
{
    let samples_per_pixel = C::BITS_PER_SAMPLE.len();
    let expected_samples = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(samples_per_pixel))
        .ok_or_else(|| "The BigTIFF dimensions exceed the addressable sample count".to_string())?;
    if data.len() != expected_samples {
        return Err(format!(
            "BigTIFF received {} samples; expected {expected_samples}",
            data.len()
        ));
    }

    let mut encoder = StreamingTiffEncoder::<_, TiffKindBig>::new_big(output)
        .map_err(|error| format!("Failed to start BigTIFF encoder: {error}"))?;
    let mut image = encoder
        .new_image::<C>(width, height)
        .map_err(|error| format!("Failed to configure BigTIFF image: {error}"))?;
    if embed_color_profile {
        image
            .encoder()
            .write_tag(TiffTag::IccProfile, srgb_v4_profile())
            .map_err(|error| format!("Failed to attach BigTIFF ICC profile: {error}"))?;
    }
    image
        .write_data(data)
        .map_err(|error| format!("Failed to encode BigTIFF pixels: {error}"))
}

#[cfg(not(target_os = "android"))]
fn encode_rgb_tiff_with_metadata<C>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    data: &[C::Inner],
    embed_color_profile: bool,
    export_metadata: Option<&ExifMetadata>,
) -> Result<(), String>
where
    C: TiffColorType,
    [C::Inner]: TiffValue,
{
    // Three samples for RGB, four for RGB with the Coverage_Mask as alpha.
    let samples_per_pixel = C::BITS_PER_SAMPLE.len();
    let expected_samples = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(samples_per_pixel))
        .ok_or_else(|| "The RGB TIFF dimensions exceed the addressable sample count".to_string())?;
    if data.len() != expected_samples {
        return Err(format!(
            "RGB TIFF received {} samples; expected {expected_samples}",
            data.len()
        ));
    }

    let mut encoder = StreamingTiffEncoder::new(output)
        .map_err(|error| format!("Failed to start RGB TIFF encoder: {error}"))?;
    let interoperability_directory = match export_metadata {
        Some(metadata) if has_writable_tiff_metadata_group(metadata, ExifTagGroup::INTEROP) => {
            let mut directory = encoder.extra_directory().map_err(|error| {
                format!("Failed to start TIFF interoperability metadata: {error}")
            })?;
            write_tiff_metadata_group(&mut directory, metadata, ExifTagGroup::INTEROP)?;
            Some(directory.finish_with_offsets().map_err(|error| {
                format!("Failed to finish TIFF interoperability metadata: {error}")
            })?)
        }
        _ => None,
    };
    let exif_directory = match export_metadata {
        Some(metadata) if has_writable_tiff_metadata_group(metadata, ExifTagGroup::EXIF) => {
            let mut directory = encoder
                .extra_directory()
                .map_err(|error| format!("Failed to start TIFF EXIF metadata: {error}"))?;
            write_tiff_metadata_group(&mut directory, metadata, ExifTagGroup::EXIF)?;
            if let Some(interoperability_directory) = interoperability_directory {
                directory
                    .write_tag(TiffTag::Unknown(0xa005), interoperability_directory.offset)
                    .map_err(|error| {
                        format!("Failed to link TIFF interoperability metadata: {error}")
                    })?;
            }
            Some(
                directory
                    .finish_with_offsets()
                    .map_err(|error| format!("Failed to finish TIFF EXIF metadata: {error}"))?,
            )
        }
        _ => None,
    };
    let gps_directory = match export_metadata {
        Some(metadata) if has_writable_tiff_metadata_group(metadata, ExifTagGroup::GPS) => {
            let mut directory = encoder
                .extra_directory()
                .map_err(|error| format!("Failed to start TIFF GPS metadata: {error}"))?;
            write_tiff_metadata_group(&mut directory, metadata, ExifTagGroup::GPS)?;
            Some(
                directory
                    .finish_with_offsets()
                    .map_err(|error| format!("Failed to finish TIFF GPS metadata: {error}"))?,
            )
        }
        _ => None,
    };

    let mut image = encoder
        .new_image::<C>(width, height)
        .map_err(|error| format!("Failed to configure RGB TIFF image: {error}"))?;
    if let Some(metadata) = export_metadata
        && has_writable_tiff_metadata_group(metadata, ExifTagGroup::GENERIC)
    {
        write_tiff_metadata_group(image.encoder(), metadata, ExifTagGroup::GENERIC)?;
    }
    if let Some(exif_directory) = exif_directory {
        image
            .encoder()
            .write_tag(TiffTag::ExifDirectory, exif_directory.offset)
            .map_err(|error| format!("Failed to link TIFF EXIF metadata: {error}"))?;
    }
    if let Some(gps_directory) = gps_directory {
        image
            .encoder()
            .write_tag(TiffTag::GpsDirectory, gps_directory.offset)
            .map_err(|error| format!("Failed to link TIFF GPS metadata: {error}"))?;
    }
    if embed_color_profile {
        image
            .encoder()
            .write_tag(TiffTag::IccProfile, srgb_v4_profile())
            .map_err(|error| format!("Failed to attach TIFF ICC profile: {error}"))?;
    }
    image
        .write_data(data)
        .map_err(|error| format!("Failed to encode RGB TIFF pixels: {error}"))
}
