//! Decoding source bytes into the base image.
//!
//! RAW files develop through rawler with an embedded-preview fallback for files
//! that cannot be developed; other formats decode with their embedded ICC
//! profile and EXIF orientation applied.

use crate::Cursor;
use crate::app_settings::AppSettings;
use crate::color_management::normalize_encoded_rgb_profile_to_srgb;
use crate::formats::is_psd_file;
use crate::formats::is_raw_file;
use crate::image_processing::apply_orientation;
use crate::image_processing::apply_srgb_to_linear;
use crate::image_processing::remove_raw_artifacts_and_enhance;
use crate::psd_loader::load_psd_image;
use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use exif::Reader as ExifReader;
use exif::Tag;
use image::DynamicImage;
use image::ImageDecoder;
use image::ImageReader;
use image::codecs::tiff::TiffDecoder;
use rawler::Orientation;
use std::collections::HashMap;
use std::panic;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub fn load_base_image_from_bytes(
    bytes: &[u8],
    path_for_ext_check: &str,
    use_fast_raw_dev: bool,
    settings: &AppSettings,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    load_base_image_from_bytes_with_exif_persistence(
        bytes,
        path_for_ext_check,
        use_fast_raw_dev,
        settings,
        cancel_token,
        true,
    )
}

/// Decodes a source image without writing or migrating its EXIF sidecar.
/// Transient tools can use this when they do not need adjustment-patch
/// compositing, avoiding an otherwise unnecessary full-frame clone.
pub fn load_base_image_from_bytes_without_exif_persistence(
    bytes: &[u8],
    path_for_ext_check: &str,
    use_fast_raw_dev: bool,
    settings: &AppSettings,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    load_base_image_from_bytes_with_exif_persistence(
        bytes,
        path_for_ext_check,
        use_fast_raw_dev,
        settings,
        cancel_token,
        false,
    )
}

pub(super) fn load_base_image_from_bytes_with_exif_persistence(
    bytes: &[u8],
    path_for_ext_check: &str,
    use_fast_raw_dev: bool,
    settings: &AppSettings,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
    persist_exif: bool,
) -> Result<DynamicImage> {
    let highlight_compression = settings.raw_highlight_compression.unwrap_or(2.5);
    let linear_mode = settings.linear_raw_mode.clone();
    let color_nr_setting = settings.raw_preprocessing_color_nr.unwrap_or(0.5);
    let color_nr_amount = if color_nr_setting <= 0.0 {
        0.0
    } else {
        let x = color_nr_setting.clamp(0.01, 1.0);
        (12.0 / x - 10.0).max(0.1)
    };
    let sharpening_amount = settings.raw_preprocessing_sharpening.unwrap_or(0.35);
    let apply_to_non_raws = settings.apply_preprocessing_to_non_raws.unwrap_or(false);

    if persist_exif {
        crate::exif_processing::persist_exif_if_missing(
            Path::new(path_for_ext_check),
            path_for_ext_check,
            bytes,
        );
    }

    if is_raw_file(path_for_ext_check) {
        match panic::catch_unwind(move || {
            crate::raw_processing::develop_raw_image(
                bytes,
                use_fast_raw_dev,
                highlight_compression,
                linear_mode,
                cancel_token,
            )
        }) {
            Ok(Ok(mut image)) => {
                if !use_fast_raw_dev && (color_nr_amount > 0.0 || sharpening_amount > 0.0) {
                    let start = Instant::now();
                    remove_raw_artifacts_and_enhance(
                        &mut image,
                        color_nr_amount,
                        sharpening_amount,
                    );
                    let duration = start.elapsed();
                    log::info!(
                        "Raw enhancing for '{}' took {:?}",
                        path_for_ext_check,
                        duration
                    );
                }
                Ok(image)
            }
            Ok(Err(e)) => {
                let classified = classify_raw_develop_error(path_for_ext_check, e);

                if classified.to_string().contains("Load cancelled") {
                    return Err(classified);
                }

                log::warn!(
                    "Error developing RAW file '{}': {}",
                    path_for_ext_check,
                    classified
                );
                if let Some(preview) = safe_embedded_preview_fallback(bytes, path_for_ext_check) {
                    log::warn!(
                        "Using embedded preview fallback for '{}' ({}x{})",
                        path_for_ext_check,
                        preview.width(),
                        preview.height()
                    );

                    return Ok(linearize_embedded_preview(preview));
                }
                Err(classified)
            }
            Err(_) => {
                log::error!("Panic while processing RAW file: {}", path_for_ext_check);
                if let Some(preview) = safe_embedded_preview_fallback(bytes, path_for_ext_check) {
                    log::warn!(
                        "Using embedded preview fallback for '{}' after RAW decoder panic ({}x{})",
                        path_for_ext_check,
                        preview.width(),
                        preview.height()
                    );

                    return Ok(linearize_embedded_preview(preview));
                }
                Err(anyhow!(
                    "Failed to process RAW file: {}",
                    path_for_ext_check
                ))
            }
        }
    } else {
        let mut image = if is_psd_file(path_for_ext_check) {
            load_psd_image(bytes, cancel_token)?
        } else {
            load_image_with_orientation(bytes, cancel_token)?
        };

        if apply_to_non_raws
            && !use_fast_raw_dev
            && (color_nr_amount > 0.0 || sharpening_amount > 0.0)
        {
            let start = Instant::now();
            remove_raw_artifacts_and_enhance(&mut image, color_nr_amount, sharpening_amount);
            let duration = start.elapsed();
            log::info!(
                "Enhancing non-RAW '{}' took {:?}",
                path_for_ext_check,
                duration
            );
        }

        Ok(image)
    }
}

pub(super) fn classify_raw_develop_error(path: &str, err: anyhow::Error) -> anyhow::Error {
    let error_text = err.to_string();
    let lowered = error_text.to_ascii_lowercase();
    let unsupported_compression =
        lowered.contains("nef compression") && lowered.contains("not supported");

    if unsupported_compression {
        return anyhow!(
            "Unsupported RAW compression format for '{}'. Original error: {}",
            path,
            error_text
        );
    }

    err
}

pub(super) fn largest_tiff_jpeg_preview(buf: &[u8]) -> Option<DynamicImage> {
    let le = match buf.get(..4)? {
        [0x49, 0x49, 0x2A, 0x00] => true,
        [0x4D, 0x4D, 0x00, 0x2A] => false,
        _ => return None,
    };
    let rd16 = |o: usize| -> Option<u64> {
        let b: [u8; 2] = buf.get(o..o + 2)?.try_into().ok()?;
        Some(if le {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        } as u64)
    };
    let rd32 = |o: usize| -> Option<u64> {
        let b: [u8; 4] = buf.get(o..o + 4)?.try_into().ok()?;
        Some(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        } as u64)
    };

    let mut candidates: Vec<(u64, u64)> = Vec::new();
    let mut queue: Vec<u64> = vec![rd32(4)?];
    let mut seen = HashMap::new();

    while let Some(ifd) = queue.pop() {
        if seen.insert(ifd, ()).is_some() || seen.len() > 64 {
            continue;
        }
        let Some(n) = rd16(ifd as usize) else {
            continue;
        };

        let mut compression: u64 = 0;
        let mut strip: Option<(u64, u64)> = None;
        let mut old_jpeg: Option<(u64, u64)> = None;

        for i in 0..n {
            let e = ifd as usize + 2 + (i as usize) * 12;
            let (Some(tag), Some(count), Some(val)) = (rd16(e), rd32(e + 4), rd32(e + 8)) else {
                continue;
            };
            match tag {
                259 => compression = val,
                273 if count == 1 => strip = Some((val, strip.map_or(0, |s| s.1))),
                279 if count == 1 => strip = strip.map(|s| (s.0, val)).or(Some((0, val))),
                513 => old_jpeg = Some((val, old_jpeg.map_or(0, |s| s.1))),
                514 => old_jpeg = old_jpeg.map(|s| (s.0, val)).or(Some((0, val))),
                330 => {
                    if count == 1 {
                        queue.push(val);
                    } else {
                        for j in 0..count.min(8) {
                            if let Some(p) = rd32(val as usize + (j as usize) * 4) {
                                queue.push(p);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if matches!(compression, 6 | 7)
            && let Some(s) = strip
        {
            candidates.push(s);
        }
        if let Some(oj) = old_jpeg {
            candidates.push(oj);
        }
        if let Some(next) = rd32(ifd as usize + 2 + (n as usize) * 12)
            && next != 0
        {
            queue.push(next);
        }
    }

    candidates.sort_by_key(|&(_, len)| std::cmp::Reverse(len));

    for (off, len) in candidates {
        if let Some(bytes) = buf.get(off as usize..(off + len) as usize)
            && let Ok(img) = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg)
        {
            return Some(img);
        }
    }

    None
}

pub(super) fn embedded_preview_fallback(bytes: &[u8], path: &str) -> Option<DynamicImage> {
    let img = match largest_tiff_jpeg_preview(bytes) {
        Some(img) => img,
        None => rawler::analyze::extract_preview_pixels(
            path,
            &rawler::decoders::RawDecodeParams::default(),
        )
        .ok()?,
    };

    let orientation = ExifReader::new()
        .read_from_container(&mut Cursor::new(bytes))
        .ok()
        .and_then(|exif| {
            exif.get_field(Tag::Orientation, exif::In::PRIMARY)?
                .value
                .get_uint(0)
        });

    Some(match orientation {
        Some(o) if o > 1 => apply_orientation(img, Orientation::from_u16(o as u16)),
        _ => img,
    })
}

pub(super) fn safe_embedded_preview_fallback(bytes: &[u8], path: &str) -> Option<DynamicImage> {
    match panic::catch_unwind(panic::AssertUnwindSafe(|| {
        embedded_preview_fallback(bytes, path)
    })) {
        Ok(preview) => preview,
        Err(_) => {
            log::warn!("Embedded RAW preview extraction panicked for '{}'", path);
            None
        }
    }
}

pub(super) fn linearize_embedded_preview(preview: DynamicImage) -> DynamicImage {
    let preview = DynamicImage::ImageRgb32F(preview.to_rgb32f());
    let mut linear_preview = apply_srgb_to_linear(preview).into_rgb32f();
    for pixel in linear_preview.pixels_mut() {
        pixel[0] *= 0.4;
        pixel[1] *= 0.4;
        pixel[2] *= 0.4;
    }
    DynamicImage::ImageRgb32F(linear_preview)
}

pub(super) fn read_embedded_icc<D: ImageDecoder>(
    bytes: &[u8],
    format: Option<image::ImageFormat>,
    decoder: &mut D,
) -> image::ImageResult<Option<Vec<u8>>> {
    // ImageReader applies limits while constructing its opaque decoder. In image
    // 0.25.10 that rebuilds the TIFF decoder and loses access to the current IFD's
    // ICC tag, so use a metadata-only direct decoder for TIFF until upstream fixes it.
    if format == Some(image::ImageFormat::Tiff) {
        let mut metadata_decoder = TiffDecoder::new(Cursor::new(bytes))?;
        metadata_decoder.icc_profile()
    } else {
        decoder.icc_profile()
    }
}

pub fn load_image_with_orientation(
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

    let cursor = Cursor::new(bytes);
    let mut reader = ImageReader::new(cursor.clone())
        .with_guessed_format()
        .context("Failed to guess image format")?;

    reader.no_limits();

    check_cancel()?;

    let format = reader.format();
    let mut decoder = reader
        .into_decoder()
        .context("Failed to initialize image decoder")?;
    let embedded_icc = match read_embedded_icc(bytes, format, &mut decoder) {
        Ok(profile) => profile,
        Err(error) => {
            log::warn!("Ignoring unreadable embedded ICC profile: {error}");
            None
        }
    };

    check_cancel()?;
    let image = DynamicImage::from_decoder(decoder).context("Failed to decode image")?;
    check_cancel()?;

    let oriented_image = {
        let exif_reader = ExifReader::new();
        if let Ok(exif) = exif_reader.read_from_container(&mut cursor.clone()) {
            if let Some(orientation) = exif
                .get_field(Tag::Orientation, exif::In::PRIMARY)
                .and_then(|f| f.value.get_uint(0))
            {
                check_cancel()?;
                apply_orientation(image, Orientation::from_u16(orientation as u16))
            } else {
                image
            }
        } else {
            image
        }
    };

    check_cancel()?;
    let mut encoded_srgb = oriented_image.to_rgb32f();
    if let Some(profile) = embedded_icc {
        match normalize_encoded_rgb_profile_to_srgb(encoded_srgb.as_mut(), &profile) {
            Ok(true) => log::debug!("Converted embedded RGB ICC profile to sRGB"),
            Ok(false) => {}
            Err(error) => log::warn!("Ignoring unsupported embedded ICC profile: {error}"),
        }
    }
    check_cancel()?;

    Ok(DynamicImage::ImageRgb32F(encoded_srgb))
}
