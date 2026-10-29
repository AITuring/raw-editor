use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::color_management::validate_icc_profile;
use image::codecs::{jpeg::JpegDecoder, png::PngDecoder, tiff::TiffDecoder, webp::WebPDecoder};
use image::{ImageDecoder, Rgb, RgbImage};
use jxl_oxide::integration::JxlDecoder;
use little_exif::exif_tag::ExifTag;
use little_exif::filetype::FileExtension;
use little_exif::metadata::Metadata;
use little_exif::rational::uR64;
use tempfile::tempdir;

use super::*;

mod batch_jobs;
mod bounded_webp;
mod encoded_formats;
mod large_image_harnesses;
mod streaming_metadata;
mod streaming_output;
mod watermarking;

fn fixture_image(width: u32, height: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
        Rgb([
            ((x * 17 + y * 3) % 256) as u8,
            ((x * 5 + y * 13) % 256) as u8,
            ((x * 11 + y * 7) % 256) as u8,
        ])
    }))
}

fn encode_streaming_fixture_with_metadata(
    image: &DynamicImage,
    output_format: &str,
    jpeg_quality: u8,
    export_metadata: StreamingExportMetadata<'_>,
) -> Result<Vec<u8>, String> {
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let row_bytes = width as usize * 4;
    let mut file = tempfile::tempfile().map_err(|error| error.to_string())?;
    encode_streaming_rgba_rows(
        &mut file,
        width,
        height,
        output_format,
        jpeg_quality,
        default_export_bit_depth_for_format(output_format),
        true,
        export_metadata,
        |sink| {
            for row in rgba.as_raw().chunks_exact(row_bytes) {
                sink(row)?;
            }
            Ok(())
        },
    )?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    let mut encoded = Vec::new();
    file.read_to_end(&mut encoded)
        .map_err(|error| error.to_string())?;
    Ok(encoded)
}

fn collect_streaming_transform(
    image: &DynamicImage,
    settings: &ExportSettings,
) -> Result<RgbaImage, String> {
    let rgba = image.to_rgba8();
    let (source_width, source_height) = rgba.dimensions();
    let (target_width, target_height) = settings
        .resize
        .as_ref()
        .map(|resize| calculate_resize_target(source_width, source_height, resize))
        .unwrap_or((source_width, source_height));
    let source_row_bytes = source_width as usize * 4;
    let mut transformed = Vec::with_capacity(target_width as usize * target_height as usize * 4);
    let mut output_sink = |row: &[u8]| -> Result<(), String> {
        transformed.extend_from_slice(row);
        Ok(())
    };
    transform_streaming_rgba_rows(
        source_width,
        source_height,
        target_width,
        target_height,
        settings.watermark.as_ref(),
        &mut output_sink,
        |source_sink| {
            for row in rgba.as_raw().chunks_exact(source_row_bytes) {
                source_sink(row)?;
            }
            Ok(())
        },
    )?;

    RgbaImage::from_raw(target_width, target_height, transformed)
        .ok_or_else(|| "Streaming transform returned an invalid RGBA buffer".to_string())
}

fn assert_decoder_contract<D: ImageDecoder>(
    mut decoder: D,
    expected: (u32, u32),
    expect_exact_profile_bytes: bool,
) {
    assert_eq!(decoder.dimensions(), expected);
    let profile = decoder
        .icc_profile()
        .expect("read embedded ICC")
        .expect("export must contain an ICC profile");
    validate_icc_profile(&profile).expect("exported ICC profile must be valid RGB ICC");

    if expect_exact_profile_bytes {
        assert_eq!(profile, srgb_v4_profile());
    }
}
