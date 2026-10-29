//! Row-streaming TIFF and BigTIFF encoders.
//!
//! Writes strips as rows arrive and attaches the EXIF/GPS metadata directories.

use crate::color_management::srgb_v4_profile;
#[cfg(not(target_os = "android"))]
use little_exif::endian::Endian as ExifEndian;
#[cfg(not(target_os = "android"))]
use little_exif::ifd::ExifTagGroup;
#[cfg(not(target_os = "android"))]
use little_exif::metadata::Metadata as ExifMetadata;
use std::fs;
#[cfg(not(target_os = "android"))]
use std::io::Seek;
use std::io::Write;
#[cfg(not(target_os = "android"))]
use tiff::Directory as TiffDirectory;
#[cfg(not(target_os = "android"))]
use tiff::encoder::DirectoryEncoder as TiffDirectoryEncoder;
#[cfg(not(target_os = "android"))]
use tiff::encoder::TiffEncoder as StreamingTiffEncoder;
#[cfg(not(target_os = "android"))]
use tiff::encoder::TiffKindBig;
#[cfg(not(target_os = "android"))]
use tiff::encoder::TiffKindStandard;
#[cfg(not(target_os = "android"))]
use tiff::encoder::TiffValue;
#[cfg(not(target_os = "android"))]
use tiff::encoder::colortype::ColorType as TiffColorType;
#[cfg(not(target_os = "android"))]
use tiff::encoder::colortype::RGB8;
#[cfg(not(target_os = "android"))]
use tiff::encoder::colortype::RGB16;
#[cfg(not(target_os = "android"))]
use tiff::tags::Tag as TiffTag;
#[cfg(not(target_os = "android"))]
use tiff::tags::Type as TiffType;

use super::effective_export_bit_depth;
#[cfg(not(target_os = "android"))]
use super::validate_streamed_rgba_row;

#[cfg(not(target_os = "android"))]
pub(super) fn has_writable_tiff_metadata_group(
    metadata: &ExifMetadata,
    group: ExifTagGroup,
) -> bool {
    metadata
        .into_iter()
        .any(|tag| tag.get_group() == group && tag.is_writable())
}

#[cfg(not(target_os = "android"))]
pub(super) fn write_tiff_metadata_group<W: Write + Seek>(
    directory: &mut TiffDirectoryEncoder<'_, W, TiffKindStandard>,
    metadata: &ExifMetadata,
    group: ExifTagGroup,
) -> Result<(), String> {
    let endian = if cfg!(target_endian = "little") {
        ExifEndian::Little
    } else {
        ExifEndian::Big
    };
    let mut entries = TiffDirectory::empty();

    for tag in metadata
        .into_iter()
        .filter(|tag| tag.get_group() == group && tag.is_writable())
    {
        let format = tag.format();
        let data_type = TiffType::from_u16(format.as_u16()).ok_or_else(|| {
            format!(
                "Unsupported TIFF metadata type {} for tag 0x{:04x}",
                format.as_u16(),
                tag.as_u16()
            )
        })?;
        let expected_bytes = usize::try_from(tag.number_of_components())
            .ok()
            .and_then(|components| components.checked_mul(format.bytes_per_component() as usize))
            .ok_or_else(|| {
                format!(
                    "TIFF metadata tag 0x{:04x} exceeds the component size limit",
                    tag.as_u16()
                )
            })?;
        if expected_bytes == 0 {
            return Err(format!(
                "TIFF metadata tag 0x{:04x} has no components",
                tag.as_u16()
            ));
        }

        let mut value = tag.value_as_u8_vec(&endian);
        if value.len() > expected_bytes {
            return Err(format!(
                "TIFF metadata tag 0x{:04x} encoded {} bytes; expected at most {expected_bytes}",
                tag.as_u16(),
                value.len()
            ));
        }
        value.resize(expected_bytes, 0);

        let entry = directory
            .write_entry_bytes(data_type, &value)
            .map_err(|error| {
                format!(
                    "Failed to encode TIFF metadata tag 0x{:04x}: {error}",
                    tag.as_u16()
                )
            })?;
        entries.extend([(TiffTag::from_u16_exhaustive(tag.as_u16()), entry)]);
    }

    directory.extend_from(&entries);
    Ok(())
}

#[cfg(not(target_os = "android"))]
pub(super) fn encode_streaming_tiff<F>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    bit_depth: u8,
    embed_color_profile: bool,
    export_metadata: Option<&ExifMetadata>,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    if effective_export_bit_depth("tiff", bit_depth) == 16 {
        encode_streaming_tiff_typed::<RGB16, _>(
            output,
            width,
            height,
            embed_color_profile,
            export_metadata,
            |sample| u16::from(sample) * 257,
            render_rows,
        )
    } else {
        encode_streaming_tiff_typed::<RGB8, _>(
            output,
            width,
            height,
            embed_color_profile,
            export_metadata,
            |sample| sample,
            render_rows,
        )
    }
}

/// Stream RGB rows into a BigTIFF without materialising the canvas.
///
/// The ordinary streaming TIFF writer above is intentionally kept for small
/// exports, but its 32-bit strip offsets cannot address a multi-gigabyte
/// native artwork.  Native tiled mosaic callers can use this narrow RGB8
/// boundary after rendering one row (or one bounded strip) at a time.  Rows
/// are supplied as RGBA because that is the common compositor/export
/// contract; alpha is discarded at the file boundary.
#[cfg(not(target_os = "android"))]
pub(crate) fn encode_streaming_bigtiff_rgb8<F>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    const ROWS_PER_STRIP: u32 = 64;

    let mut encoder = StreamingTiffEncoder::<_, TiffKindBig>::new_big(output)
        .map_err(|error| format!("Failed to start streaming BigTIFF encoder: {error}"))?;
    let mut image = encoder
        .new_image::<RGB8>(width, height)
        .map_err(|error| format!("Failed to configure streaming BigTIFF image: {error}"))?;
    image
        .rows_per_strip(ROWS_PER_STRIP.min(height).max(1))
        .map_err(|error| format!("Failed to configure BigTIFF strips: {error}"))?;

    let strip_capacity = (width as usize)
        .checked_mul(3)
        .and_then(|samples_per_row| samples_per_row.checked_mul(ROWS_PER_STRIP as usize))
        .ok_or_else(|| format!("BigTIFF width {width} exceeds the strip size limit"))?;
    let mut strip = Vec::<u8>::with_capacity(strip_capacity);
    let mut written_rows = 0_u32;
    {
        let mut sink = |rgba_row: &[u8]| -> Result<(), String> {
            validate_streamed_rgba_row(rgba_row, width, height, written_rows)?;
            for pixel in rgba_row.chunks_exact(4) {
                strip.extend_from_slice(&pixel[..3]);
            }
            written_rows += 1;

            let expected_samples = image.next_strip_sample_count() as usize;
            if strip.len() == expected_samples {
                image
                    .write_strip(&strip)
                    .map_err(|error| format!("Failed to write BigTIFF strip: {error}"))?;
                strip.clear();
            } else if strip.len() > expected_samples {
                return Err(format!(
                    "Streaming BigTIFF strip exceeded its expected {expected_samples} samples"
                ));
            }
            Ok(())
        };
        render_rows(&mut sink)?;
    }
    if written_rows != height || !strip.is_empty() || image.next_strip_sample_count() != 0 {
        return Err(format!(
            "Streaming BigTIFF received {written_rows} complete rows; expected {height}"
        ));
    }
    image
        .finish()
        .map_err(|error| format!("Failed to finish streaming BigTIFF: {error}"))
}

#[cfg(not(target_os = "android"))]
fn encode_streaming_tiff_typed<C, F>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    embed_color_profile: bool,
    export_metadata: Option<&ExifMetadata>,
    to_sample: fn(u8) -> C::Inner,
    render_rows: F,
) -> Result<(), String>
where
    C: TiffColorType,
    [C::Inner]: TiffValue,
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    const ROWS_PER_STRIP: u32 = 64;

    let mut encoder = StreamingTiffEncoder::new(output)
        .map_err(|error| format!("Failed to start streaming TIFF encoder: {error}"))?;

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
        .map_err(|error| format!("Failed to configure streaming TIFF image: {error}"))?;
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
        .rows_per_strip(ROWS_PER_STRIP.min(height).max(1))
        .map_err(|error| format!("Failed to configure TIFF strips: {error}"))?;

    let strip_capacity = (width as usize)
        .checked_mul(3)
        .and_then(|samples_per_row| samples_per_row.checked_mul(ROWS_PER_STRIP as usize))
        .ok_or_else(|| format!("TIFF width {width} exceeds the strip size limit"))?;
    let mut strip = Vec::<C::Inner>::new();
    strip
        .try_reserve_exact(strip_capacity)
        .map_err(|error| format!("Failed to reserve the TIFF strip buffer: {error}"))?;
    let mut written_rows = 0_u32;
    {
        let mut sink = |rgba_row: &[u8]| -> Result<(), String> {
            validate_streamed_rgba_row(rgba_row, width, height, written_rows)?;
            for pixel in rgba_row.chunks_exact(4) {
                strip.push(to_sample(pixel[0]));
                strip.push(to_sample(pixel[1]));
                strip.push(to_sample(pixel[2]));
            }
            written_rows += 1;

            let expected_samples = image.next_strip_sample_count() as usize;
            if strip.len() == expected_samples {
                image
                    .write_strip(&strip)
                    .map_err(|error| format!("Failed to write TIFF strip: {error}"))?;
                strip.clear();
            } else if strip.len() > expected_samples {
                return Err(format!(
                    "Streaming TIFF strip exceeded its expected {} samples",
                    expected_samples
                ));
            }
            Ok(())
        };
        render_rows(&mut sink)?;
    }
    if written_rows != height || !strip.is_empty() || image.next_strip_sample_count() != 0 {
        return Err(format!(
            "Streaming TIFF received {written_rows} complete rows; expected {height}"
        ));
    }
    image
        .finish()
        .map_err(|error| format!("Failed to finish streaming TIFF: {error}"))
}
