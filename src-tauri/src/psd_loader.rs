use anyhow::{Context, Result, anyhow};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb, Rgba};
use rayon::prelude::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use zune_psd::PSDDecoder;
use zune_psd::zune_core::bytestream::ZCursor;
use zune_psd::zune_core::result::DecodingResult;

const MAX_PSD_PIXELS: u64 = 500_000_000;
/// Photoshop image resource that carries the document ICC profile.
const PSD_ICC_PROFILE_RESOURCE: u16 = 1039;
/// Rows converted per colour-management band; bounds the temporary RGB copy.
const PROFILE_BAND_ROWS: usize = 256;

/// The embedded ICC profile of a PSD, if the image resources carry one.
///
/// `zune-psd` skips the image resource section, so the document profile is
/// read here directly. Malformed sections simply yield no profile.
fn embedded_icc_profile(bytes: &[u8]) -> Option<&[u8]> {
    let read_u32 = |offset: usize| -> Option<usize> {
        let field: [u8; 4] = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
        usize::try_from(u32::from_be_bytes(field)).ok()
    };
    if bytes.get(0..4)? != b"8BPS" {
        return None;
    }
    // 26-byte file header, then the colour mode data section.
    let color_mode_length = read_u32(26)?;
    let resources = 30usize.checked_add(color_mode_length)?;
    let resources_length = read_u32(resources)?;
    let mut offset = resources.checked_add(4)?;
    let end = offset.checked_add(resources_length)?.min(bytes.len());
    while offset.checked_add(12)? <= end {
        if bytes.get(offset..offset + 4)? != b"8BIM" {
            return None;
        }
        let id = u16::from_be_bytes([*bytes.get(offset + 4)?, *bytes.get(offset + 5)?]);
        // Pascal name, padded so length byte + name is even.
        let name_length = usize::from(*bytes.get(offset + 6)?);
        let mut cursor = offset + 6 + 1 + name_length;
        if (1 + name_length) % 2 == 1 {
            cursor += 1;
        }
        let size = read_u32(cursor)?;
        cursor = cursor.checked_add(4)?;
        let data_end = cursor.checked_add(size)?;
        if data_end > end {
            return None;
        }
        if id == PSD_ICC_PROFILE_RESOURCE {
            return Some(&bytes[cursor..data_end]);
        }
        offset = data_end + (size % 2);
    }
    None
}

/// Fill one band of encoded RGB (and optional alpha) samples as `f32`.
fn fill_profile_band<T: Copy + Into<f32> + Sync>(
    samples: &[T],
    channels: usize,
    scale: f32,
    first_pixel: usize,
    rgb: &mut [f32],
    alpha: Option<&mut [f32]>,
) {
    let pixels = rgb.len() / 3;
    for pixel in 0..pixels {
        let source = (first_pixel + pixel) * channels;
        for channel in 0..3 {
            rgb[pixel * 3 + channel] = samples[source + channel].into() * scale;
        }
    }
    if let Some(alpha) = alpha {
        for (pixel, value) in alpha.iter_mut().enumerate() {
            *value = samples[(first_pixel + pixel) * channels + 3].into() * scale;
        }
    }
}

/// Convert a PSD's document profile to the editor's encoded-sRGB contract,
/// as JPEG/PNG/TIFF imports already do. Alpha is preserved so a scan
/// compositor can still tell transparent composite pixels apart.
fn normalize_psd_color_profile(image: DynamicImage, profile: &[u8]) -> DynamicImage {
    let channels = usize::from(image.color().channel_count());
    if !matches!(channels, 3 | 4) {
        return image;
    }
    let (width, height) = image.dimensions();
    let row = width as usize;
    let mut output = vec![0.0f32; row * height as usize * channels];
    let band_samples = row * channels * PROFILE_BAND_ROWS;
    let results = output
        .par_chunks_mut(band_samples)
        .enumerate()
        .map(|(band, chunk)| {
            let first_pixel = band * row * PROFILE_BAND_ROWS;
            let pixels = chunk.len() / channels;
            let mut rgb = vec![0.0f32; pixels * 3];
            let mut alpha = (channels == 4).then(|| vec![0.0f32; pixels]);
            match &image {
                DynamicImage::ImageRgb16(buffer) => fill_profile_band(
                    buffer.as_raw(),
                    3,
                    1.0 / 65535.0,
                    first_pixel,
                    &mut rgb,
                    None,
                ),
                DynamicImage::ImageRgba16(buffer) => fill_profile_band(
                    buffer.as_raw(),
                    4,
                    1.0 / 65535.0,
                    first_pixel,
                    &mut rgb,
                    alpha.as_deref_mut(),
                ),
                DynamicImage::ImageRgb8(buffer) => {
                    fill_profile_band(buffer.as_raw(), 3, 1.0 / 255.0, first_pixel, &mut rgb, None)
                }
                DynamicImage::ImageRgba8(buffer) => fill_profile_band(
                    buffer.as_raw(),
                    4,
                    1.0 / 255.0,
                    first_pixel,
                    &mut rgb,
                    alpha.as_deref_mut(),
                ),
                _ => return Err("unsupported PSD sample layout for ICC conversion".to_string()),
            }
            let converted =
                crate::color_management::normalize_encoded_rgb_profile_to_srgb(&mut rgb, profile)?;
            for pixel in 0..pixels {
                chunk[pixel * channels..pixel * channels + 3]
                    .copy_from_slice(&rgb[pixel * 3..pixel * 3 + 3]);
                if let Some(alpha) = alpha.as_ref() {
                    chunk[pixel * channels + 3] = alpha[pixel];
                }
            }
            Ok(converted)
        })
        .collect::<Vec<Result<bool, String>>>();
    if let Some(error) = results.iter().find_map(|result| result.as_ref().err()) {
        log::warn!("Ignoring unsupported PSD ICC profile: {error}");
        return image;
    }
    if !results.iter().any(|result| matches!(result, Ok(true))) {
        return image;
    }
    let converted = if channels == 4 {
        ImageBuffer::<Rgba<f32>, Vec<f32>>::from_raw(width, height, output)
            .map(DynamicImage::ImageRgba32F)
    } else {
        ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(width, height, output)
            .map(DynamicImage::ImageRgb32F)
    };
    converted.unwrap_or(image)
}

/// Decode a Photoshop document as one already-composited source image.
///
/// PSD layers intentionally are not exposed as stack frames here. The focus
/// stack treats each path as one capture, so importing layers separately would
/// change ordering, provenance, and matching semantics. `zune-psd` supports the
/// flattened image channel data needed by the current import contract, including
/// the 16-bit RGBA files used by the Mayuan regression set.
pub(crate) fn load_psd_image(
    bytes: &[u8],
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    let check_cancel = || -> Result<()> {
        if let Some((tracker, generation)) = &cancel_token
            && tracker.load(Ordering::SeqCst) != *generation
        {
            return Err(anyhow!("Load cancelled"));
        }
        Ok(())
    };

    check_cancel()?;
    let mut decoder = PSDDecoder::new(ZCursor::new(bytes));
    decoder
        .decode_headers()
        .map_err(|error| anyhow!("Failed to decode PSD header: {error:?}"))?;

    let (width, height) = decoder
        .dimensions()
        .ok_or_else(|| anyhow!("PSD decoder did not expose dimensions"))?;
    let width_u32 = u32::try_from(width).context("PSD width exceeds image limits")?;
    let height_u32 = u32::try_from(height).context("PSD height exceeds image limits")?;
    let pixel_count = (width as u64)
        .checked_mul(height as u64)
        .ok_or_else(|| anyhow!("PSD dimensions overflow pixel count"))?;
    if pixel_count > MAX_PSD_PIXELS {
        return Err(anyhow!(
            "PSD dimensions {width}x{height} exceed the {MAX_PSD_PIXELS} pixel import limit"
        ));
    }

    let color_space = decoder
        .colorspace()
        .ok_or_else(|| anyhow!("PSD decoder did not expose a color space"))?;
    let channels = color_space.num_components();
    if !matches!(channels, 1..=4) {
        return Err(anyhow!(
            "Unsupported PSD channel count {channels}; expected grayscale or RGB/RGBA"
        ));
    }
    if matches!(
        color_space,
        zune_psd::zune_core::colorspace::ColorSpace::CMYK
    ) {
        return Err(anyhow!("CMYK PSD files are not supported yet"));
    }

    let decoded = decoder
        .decode()
        .map_err(|error| anyhow!("Failed to decode PSD pixels: {error:?}"))?;
    check_cancel()?;

    let expected_samples = (pixel_count as usize)
        .checked_mul(channels)
        .ok_or_else(|| anyhow!("PSD sample count overflow"))?;
    let image = match decoded {
        DecodingResult::U8(samples) => {
            if samples.len() != expected_samples {
                return Err(anyhow!(
                    "PSD decoder returned {} 8-bit samples; expected {expected_samples}",
                    samples.len()
                ));
            }
            dynamic_image_from_psd_u8(width_u32, height_u32, channels, samples)?
        }
        DecodingResult::U16(samples) => {
            if samples.len() != expected_samples {
                return Err(anyhow!(
                    "PSD decoder returned {} 16-bit samples; expected {expected_samples}",
                    samples.len()
                ));
            }
            // zune-psd 0.5.1 stores each big-endian PSD sample with
            // `to_ne_bytes()` before returning it. On little-endian hosts this
            // leaves the numeric value byte-swapped; normalize it before
            // constructing the image buffer.
            let samples = samples.into_iter().map(u16::from_be).collect();
            dynamic_image_from_psd_u16(width_u32, height_u32, channels, samples)?
        }
        DecodingResult::F32(_) => return Err(anyhow!("Float PSD channels are not supported yet")),
        _ => return Err(anyhow!("Unsupported PSD sample format")),
    };
    check_cancel()?;
    // A PSD tagged with e.g. Display P3 must not be read as sRGB numbers.
    Ok(match embedded_icc_profile(bytes) {
        Some(profile) => normalize_psd_color_profile(image, profile),
        None => image,
    })
}

fn dynamic_image_from_psd_u8(
    width: u32,
    height: u32,
    channels: usize,
    samples: Vec<u8>,
) -> Result<DynamicImage> {
    match channels {
        1 => ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, samples)
            .map(DynamicImage::ImageLuma8)
            .ok_or_else(|| anyhow!("Invalid 8-bit grayscale PSD buffer")),
        2 => Err(anyhow!(
            "Two-channel grayscale PSD files are not supported yet"
        )),
        3 => ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width, height, samples)
            .map(DynamicImage::ImageRgb8)
            .ok_or_else(|| anyhow!("Invalid 8-bit RGB PSD buffer")),
        4 => ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, samples)
            .map(DynamicImage::ImageRgba8)
            .ok_or_else(|| anyhow!("Invalid 8-bit RGBA PSD buffer")),
        _ => Err(anyhow!("Unsupported 8-bit PSD channel count {channels}")),
    }
}

fn dynamic_image_from_psd_u16(
    width: u32,
    height: u32,
    channels: usize,
    samples: Vec<u16>,
) -> Result<DynamicImage> {
    match channels {
        1 => ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(width, height, samples)
            .map(DynamicImage::ImageLuma16)
            .ok_or_else(|| anyhow!("Invalid 16-bit grayscale PSD buffer")),
        2 => Err(anyhow!(
            "Two-channel grayscale PSD files are not supported yet"
        )),
        3 => ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(width, height, samples)
            .map(DynamicImage::ImageRgb16)
            .ok_or_else(|| anyhow!("Invalid 16-bit RGB PSD buffer")),
        4 => ImageBuffer::<Rgba<u16>, Vec<u16>>::from_raw(width, height, samples)
            .map(DynamicImage::ImageRgba16)
            .ok_or_else(|| anyhow!("Invalid 16-bit RGBA PSD buffer")),
        _ => Err(anyhow!("Unsupported 16-bit PSD channel count {channels}")),
    }
}
