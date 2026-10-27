use anyhow::{Context, Result, anyhow};
use image::{DynamicImage, ImageBuffer, Luma, Rgb, Rgba};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use zune_psd::PSDDecoder;
use zune_psd::zune_core::bytestream::ZCursor;
use zune_psd::zune_core::result::DecodingResult;

const MAX_PSD_PIXELS: u64 = 500_000_000;

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
    match decoded {
        DecodingResult::U8(samples) => {
            if samples.len() != expected_samples {
                return Err(anyhow!(
                    "PSD decoder returned {} 8-bit samples; expected {expected_samples}",
                    samples.len()
                ));
            }
            Ok(dynamic_image_from_psd_u8(
                width_u32, height_u32, channels, samples,
            )?)
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
            Ok(dynamic_image_from_psd_u16(
                width_u32, height_u32, channels, samples,
            )?)
        }
        DecodingResult::F32(_) => Err(anyhow!("Float PSD channels are not supported yet")),
        _ => Err(anyhow!("Unsupported PSD sample format")),
    }
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
