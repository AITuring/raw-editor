//! Input preflight and source preparation.
//!
//! Skips generated and auxiliary files, detects mixed focal lengths, loads and
//! exposure-normalises Source_RAW frames and resamples sources to the render
//! scale.

use crate::app_settings::AppSettings;
use crate::file_management::read_file_mapped;
use crate::formats::is_raw_file;
use crate::image_processing::apply_cpu_default_raw_processing;
use crate::panorama_utils::processing;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use image::ColorType;
use image::DynamicImage;
use image::GenericImageView;
use image::GrayImage;
use image::Rgb32FImage;
use image::RgbImage;
use nalgebra::Matrix3;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use super::{
    ImageInfo, build_foreground_mask, detect_foreground_range, detect_vertical_edge_columns,
    find_alignment_features, find_top_alignment_features, horizontal_edge_row_candidates,
};

pub(super) const MIXED_FOCAL_LENGTH_RATIO: f64 = 1.12;

pub(super) fn focal_lengths_span_multiple_lenses<I>(focal_lengths: I) -> bool
where
    I: IntoIterator<Item = f64>,
{
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut count = 0usize;
    for focal_length in focal_lengths {
        if !focal_length.is_finite() || focal_length <= 0.0 {
            continue;
        }
        minimum = minimum.min(focal_length);
        maximum = maximum.max(focal_length);
        count += 1;
    }
    count >= 2
        && minimum.is_finite()
        && maximum.is_finite()
        && maximum / minimum >= MIXED_FOCAL_LENGTH_RATIO
}

pub(super) fn selection_has_mixed_focal_lengths(image_paths: &[String]) -> bool {
    // Large stacks already use the scale-robust path.  This lightweight EXIF
    // pass exists for small selections, where the old count-only switch would
    // otherwise miss a 35mm/85mm camera-module change.
    focal_lengths_span_multiple_lenses(image_paths.iter().filter_map(|filename| {
        read_file_mapped(Path::new(filename))
            .ok()
            .and_then(|bytes| crate::exif_processing::focal_length_35mm_from_bytes(&bytes))
    }))
}

pub(super) fn image_pair_has_mixed_focal_lengths(source: &ImageInfo, target: &ImageInfo) -> bool {
    focal_lengths_span_multiple_lenses(
        [source.focal_length_35mm, target.focal_length_35mm]
            .into_iter()
            .flatten(),
    )
}

pub(super) fn images_have_mixed_focal_lengths(images: &[ImageInfo]) -> bool {
    focal_lengths_span_multiple_lenses(images.iter().filter_map(|image| image.focal_length_35mm))
}

#[cfg(test)]
pub(super) fn scaled_homographies(
    homographies: &HashMap<usize, Matrix3<f64>>,
    scale: f64,
) -> HashMap<usize, Matrix3<f64>> {
    let scale_matrix = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
    homographies
        .iter()
        .map(|(&id, homography)| (id, scale_matrix * homography))
        .collect()
}

pub(super) fn scaled_source_render_homographies(
    homographies: &HashMap<usize, Matrix3<f64>>,
    scale: f64,
) -> HashMap<usize, Matrix3<f64>> {
    let output_scale = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
    let source_scale_inverse = Matrix3::new(
        scale.recip(),
        0.0,
        0.0,
        0.0,
        scale.recip(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    homographies
        .iter()
        .map(|(&id, homography)| (id, output_scale * homography * source_scale_inverse))
        .collect()
}

pub(super) fn scaled_render_image_info(image: &ImageInfo, scale: f64) -> ImageInfo {
    ImageInfo {
        id: image.id,
        filename: image.filename.clone(),
        width: (f64::from(image.width) * scale).round().max(1.0) as u32,
        height: (f64::from(image.height) * scale).round().max(1.0) as u32,
        alignment_image: GrayImage::new(0, 0),
        full_image: None,
        scale_factor: image.scale_factor,
        focal_length_35mm: image.focal_length_35mm,
        source_footprint: image.source_footprint,
        overview_reference: image.overview_reference,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: image.foreground_range,
        foreground_mask: image.foreground_mask.clone(),
        horizontal_edge_rows: image.horizontal_edge_rows.clone(),
        vertical_edge_columns: image.vertical_edge_columns.clone(),
    }
}

pub(super) struct AreaContributions {
    pub(super) offsets: Vec<usize>,
    pub(super) indices: Vec<usize>,
    pub(super) weights: Vec<f32>,
}

pub(super) fn area_contributions(source_length: u32, target_length: u32) -> AreaContributions {
    let mut offsets = Vec::with_capacity(target_length as usize + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let scale = f64::from(source_length) / f64::from(target_length.max(1));
    offsets.push(0);
    for target in 0..target_length.max(1) {
        let start = f64::from(target) * scale;
        let end = (f64::from(target) + 1.0) * scale;
        let first = start.floor() as usize;
        let last = (end.ceil() as usize)
            .saturating_sub(1)
            .min(source_length.saturating_sub(1) as usize);
        let normalization = (end - start).recip();
        for source in first..=last {
            let overlap = ((source + 1) as f64).min(end) - (source as f64).max(start);
            if overlap > 0.0 {
                indices.push(source);
                weights.push((overlap * normalization) as f32);
            }
        }
        offsets.push(indices.len());
    }
    AreaContributions {
        offsets,
        indices,
        weights,
    }
}

pub(super) fn resize_rgb8_area_to_rgb32f(
    source: &RgbImage,
    target_width: u32,
    target_height: u32,
) -> Rgb32FImage {
    let (source_width, source_height) = source.dimensions();
    debug_assert!(target_width > 0 && target_height > 0);
    debug_assert!(target_width <= source_width && target_height <= source_height);

    let horizontal_contributions = area_contributions(source_width, target_width);
    let vertical_contributions = area_contributions(source_height, target_height);
    let source_stride = source_width as usize * 3;
    let horizontal_stride = target_width as usize * 3;
    let mut horizontal = vec![0.0f32; horizontal_stride * source_height as usize];
    horizontal
        .par_chunks_mut(horizontal_stride)
        .zip(source.as_raw().par_chunks(source_stride))
        .for_each(|(output_row, source_row)| {
            for target_x in 0..target_width as usize {
                let contribution_start = horizontal_contributions.offsets[target_x];
                let contribution_end = horizontal_contributions.offsets[target_x + 1];
                let output_start = target_x * 3;
                for contribution in contribution_start..contribution_end {
                    let source_start = horizontal_contributions.indices[contribution] * 3;
                    let weight = horizontal_contributions.weights[contribution] / 255.0;
                    for channel in 0..3 {
                        output_row[output_start + channel] +=
                            f32::from(source_row[source_start + channel]) * weight;
                    }
                }
            }
        });

    let mut output = vec![0.0f32; horizontal_stride * target_height as usize];
    output
        .par_chunks_mut(horizontal_stride)
        .enumerate()
        .for_each(|(target_y, output_row)| {
            let contribution_start = vertical_contributions.offsets[target_y];
            let contribution_end = vertical_contributions.offsets[target_y + 1];
            for contribution in contribution_start..contribution_end {
                let source_y = vertical_contributions.indices[contribution];
                let weight = vertical_contributions.weights[contribution];
                let horizontal_row =
                    &horizontal[source_y * horizontal_stride..(source_y + 1) * horizontal_stride];
                for (output, source) in output_row.iter_mut().zip(horizontal_row) {
                    *output += source * weight;
                }
            }
        });

    Rgb32FImage::from_raw(target_width, target_height, output)
        .expect("area-resized RGB buffer dimensions must match")
}

pub(super) fn source_to_render_rgb32f(
    source: DynamicImage,
    target_width: u32,
    target_height: u32,
) -> Rgb32FImage {
    if source.dimensions() == (target_width, target_height) {
        return source.to_rgb32f();
    }
    if source.color() == ColorType::Rgb8
        && target_width <= source.width()
        && target_height <= source.height()
    {
        return resize_rgb8_area_to_rgb32f(&source.into_rgb8(), target_width, target_height);
    }
    source
        .resize_exact(
            target_width,
            target_height,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb32f()
}

pub(super) struct PreparedStackSource {
    pub(super) image: DynamicImage,
    pub(super) source_footprint: Option<crate::lens_correction::StackLensFootprint>,
    pub(super) focal_length_35mm: Option<f64>,
}

/// Photon exposure at the sensor, including the recorded ISO sensitivity.
/// This is a camera metadata calibration, not a frame histogram adjustment.
#[cfg(test)]
pub(super) fn stack_raw_exposure_value(metadata: &HashMap<String, String>) -> Option<f64> {
    let number = |text: &str| {
        let text = text.trim().trim_start_matches("f/").trim_end_matches(" s");
        if let Some((a, b)) = text.split_once('/') {
            Some(a.parse::<f64>().ok()? / b.parse::<f64>().ok()?)
        } else {
            text.parse::<f64>().ok()
        }
    };
    let time = number(metadata.get("ExposureTime")?)?;
    let aperture = number(metadata.get("FNumber")?)?;
    let iso = number(
        metadata
            .get("ISOSpeedRatings")
            .or_else(|| metadata.get("PhotographicSensitivity"))
            .or_else(|| metadata.get("ISOSpeed"))?,
    )?;
    let exposure = time * iso / aperture.powi(2);
    (time > 0.0 && aperture > 0.0 && iso > 0.0 && exposure.is_finite()).then_some(exposure)
}

pub(super) fn stack_raw_exposure_from_bytes(bytes: &[u8]) -> Option<f64> {
    let metadata = crate::exif_processing::read_raw_metadata(bytes)?;
    let time = metadata.exif.exposure_time?;
    let aperture = metadata.exif.fnumber?;
    if time.d == 0 || aperture.d == 0 {
        return None;
    }
    let iso = metadata
        .exif
        .iso_speed
        .or_else(|| metadata.exif.iso_speed_ratings.map(|iso| iso as u32))?;
    let time = f64::from(time.n) / f64::from(time.d);
    let aperture = f64::from(aperture.n) / f64::from(aperture.d);
    let exposure = time * f64::from(iso) / aperture.powi(2);
    (time > 0.0 && aperture > 0.0 && iso > 0 && exposure.is_finite()).then_some(exposure)
}

pub(super) fn stack_raw_exposure_reference(paths: &[String]) -> Option<f64> {
    let mut exposures = Vec::with_capacity(paths.len());
    let mut camera = None;
    for path in paths {
        if !is_raw_file(path) {
            return None;
        }
        let bytes = read_file_mapped(Path::new(path)).ok()?;
        let metadata = crate::exif_processing::read_exif_data_from_bytes(path, &bytes);
        let identity = (
            metadata.get("Make")?.clone(),
            metadata.get("Model")?.clone(),
        );
        if camera.as_ref().is_some_and(|camera| camera != &identity) {
            return None;
        }
        camera = Some(identity);
        exposures.push(stack_raw_exposure_from_bytes(&bytes)?);
    }
    exposures.sort_by(f64::total_cmp);
    exposures.get(exposures.len() / 2).copied()
}

pub(super) fn load_prepared_stack_source(
    filename: &str,
    settings: &AppSettings,
    raw_exposure_reference: Option<f64>,
) -> Result<PreparedStackSource, String> {
    let file_data = read_file_mapped(Path::new(filename))
        .map_err(|error| format!("Failed to read image {filename}: {error}"))?;
    let focal_length_35mm = crate::exif_processing::focal_length_35mm_from_bytes(&file_data);
    let mut image = crate::image_loader::load_base_image_from_bytes(
        &file_data, filename, false, settings, None,
    )
    .map_err(|error| format!("Failed to load image {filename}: {error}"))?;
    let mut source_footprint = None;
    if is_raw_file(filename) {
        let mut metadata = crate::exif_processing::read_exif_data_from_bytes(filename, &file_data);
        // RAW metadata extraction otherwise copies the physical focal length
        // into this field. Use the actual EXIF value to detect sensor crop mode.
        if let Some(equivalent) = focal_length_35mm {
            metadata.insert("FocalLengthIn35mmFilm".into(), equivalent.to_string());
        } else {
            metadata.remove("FocalLengthIn35mmFilm");
        }
        let calibration =
            crate::lens_correction::stack_lens_calibration(&metadata, image.dimensions());
        if let Some(reference) = raw_exposure_reference
            && let Some(exposure) = stack_raw_exposure_from_bytes(&file_data)
        {
            let gain = (reference / exposure) as f32;
            if gain.is_finite() && gain > 0.0 {
                if image.as_rgb32f().is_none() {
                    image = DynamicImage::ImageRgb32F(image.to_rgb32f());
                }
                image
                    .as_mut_rgb32f()
                    .unwrap()
                    .as_mut()
                    .par_iter_mut()
                    .for_each(|channel| *channel = (*channel * gain).max(0.0));
            }
        }
        if let Some(calibration) = calibration {
            if image.as_rgb32f().is_none() {
                image = DynamicImage::ImageRgb32F(image.to_rgb32f());
            }
            calibration.correct_linear_illumination(image.as_mut_rgb32f().unwrap());
        }
        apply_cpu_default_raw_processing(&mut image);
        if let Some(calibration) = calibration
            && let Some(corrected) = {
                // Developed RAW is already RGB32F. Borrow it so optical
                // preparation does not retain another full 45MP RGB buffer.
                let rgb = image
                    .as_rgb32f()
                    .map(Cow::Borrowed)
                    .unwrap_or_else(|| Cow::Owned(image.to_rgb32f()));
                calibration.correct(&rgb)
            }
        {
            println!(
                "  - Calibrated stack lens geometry: {} ({}x{} -> {}x{})",
                filename,
                image.width(),
                image.height(),
                corrected.width(),
                corrected.height()
            );
            source_footprint = Some(crate::lens_correction::StackLensFootprint {
                calibration,
                original: image.dimensions(),
                corrected: corrected.dimensions(),
            });
            image = DynamicImage::ImageRgb32F(corrected);
        }
    }
    Ok(PreparedStackSource {
        image,
        source_footprint,
        focal_length_35mm,
    })
}

pub(super) fn normalize_stack_raw_display_exposure(rgb: &mut Rgb32FImage) {
    const TARGET_BRIGHT_PERCENTILE: f32 = 0.90;
    const MAX_DISPLAY_GAIN: f32 = 2.10;
    // The RAW developer deliberately leaves headroom, while image-stack
    // export freezes these samples directly as display-referred sRGB. Restore
    // a conservative display white from the bright end of the actual frame;
    // use a percentile rather than a maximum so lamps and specular acrylic do
    // not suppress the correction. This is bounded and can only brighten.
    let step = ((u64::from(rgb.width()) * u64::from(rgb.height()) / 24_000).max(1) as f64)
        .sqrt()
        .floor()
        .max(1.0) as usize;
    let mut luma_samples = Vec::with_capacity(24_000);
    for y in (0..rgb.height()).step_by(step) {
        for x in (0..rgb.width()).step_by(step) {
            let pixel = rgb.get_pixel(x, y);
            let luma = pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722;
            if luma.is_finite() && luma > 0.02 {
                luma_samples.push(luma);
            }
        }
    }
    if luma_samples.len() < 64 {
        return;
    }
    luma_samples.sort_by(f32::total_cmp);
    let high = luma_samples[((luma_samples.len() - 1) * 97) / 100];
    if !high.is_finite() || high <= 0.0 {
        return;
    }
    let gain = (TARGET_BRIGHT_PERCENTILE / high).clamp(1.0, MAX_DISPLAY_GAIN);
    if gain <= 1.005 {
        return;
    }
    rgb.as_mut()
        .par_iter_mut()
        .for_each(|channel| *channel = (*channel * gain).clamp(0.0, 1.0));
}

/// Return the robust bright-end luma of a developed RAW frame without
/// allocating a second image.  This is used only during focus-stack
/// preparation to derive one shared display gain for the whole selection.
/// Each frame already excludes the brightest 3% of pixels. Protect the
/// brightest retained channel across the batch: a median frame white clips
/// brighter scan regions and changes their RGB ratios before tone matching.
pub(super) fn shared_stack_raw_display_gain(highs: &[f32]) -> Option<f32> {
    highs
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .max_by(f32::total_cmp)
        .map(|high| (0.90 / high).clamp(1.0, 2.10))
}

pub(super) fn raw_display_channel_percentile(image: &DynamicImage) -> Option<f32> {
    let rgb = image.as_rgb32f()?;
    let step = ((u64::from(rgb.width()) * u64::from(rgb.height()) / 24_000).max(1) as f64)
        .sqrt()
        .floor()
        .max(1.0) as usize;
    let mut samples = Vec::with_capacity(24_000);
    for y in (0..rgb.height()).step_by(step) {
        for x in (0..rgb.width()).step_by(step) {
            let pixel = rgb.get_pixel(x, y);
            let luma = pixel[0].max(pixel[1]).max(pixel[2]);
            if luma.is_finite() && luma > 0.02 {
                samples.push(luma);
            }
        }
    }
    if samples.len() < 64 {
        return None;
    }
    samples.sort_unstable_by(f32::total_cmp);
    samples.get(((samples.len() - 1) * 97) / 100).copied()
}

pub(super) fn prepare_focus_rescue_image(
    image: &ImageInfo,
    settings: &AppSettings,
    raw_exposure_reference: Option<f64>,
    brief_pairs: &[(nalgebra::Point2<i32>, nalgebra::Point2<i32>)],
) -> Result<ImageInfo, String> {
    let prepared = load_prepared_stack_source(&image.filename, settings, raw_exposure_reference)?;
    let (width, height) = prepared.image.dimensions();
    let (new_width, new_height, scale_factor) =
        processing::calculate_downscale_dimensions_capped(width, height, 2_400);
    let alignment_image = prepared
        .image
        .resize_exact(new_width, new_height, image::imageops::FilterType::Triangle)
        .to_luma8();
    let foreground_range = detect_foreground_range(&alignment_image);
    Ok(ImageInfo {
        id: image.id,
        filename: image.filename.clone(),
        width,
        height,
        features: find_alignment_features(
            &alignment_image,
            brief_pairs,
            1_600,
            true,
            prepared.focal_length_35mm,
        ),
        top_features: find_top_alignment_features(&alignment_image, brief_pairs, foreground_range),
        foreground_mask: build_foreground_mask(&alignment_image, foreground_range),
        horizontal_edge_rows: horizontal_edge_row_candidates(&alignment_image, foreground_range),
        vertical_edge_columns: detect_vertical_edge_columns(&alignment_image, foreground_range),
        alignment_image,
        full_image: None,
        scale_factor,
        focal_length_35mm: prepared.focal_length_35mm,
        source_footprint: prepared.source_footprint,
        overview_reference: false,
        foreground_range,
    })
}

pub(super) fn is_generated_stitch_output(path: &str) -> bool {
    let Some(stem) = Path::new(path).file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    let stem = stem.to_ascii_lowercase();
    ["_focusstack", "_panorama", "_pano"]
        .into_iter()
        .any(|suffix| {
            let Some(marker) = stem.rfind(suffix) else {
                return false;
            };
            let trailing = &stem[marker + suffix.len()..];
            trailing.is_empty()
                || trailing
                    .chars()
                    .all(|character| character.is_ascii_digit() || matches!(character, '-' | '_'))
        })
}

pub(super) fn is_auxiliary_stitch_file(path: &str) -> bool {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("._"))
}

/// The selected sources that produced no decoded image (需求 1.9).
///
/// Returned in selection order and by absolute path, which is what the
/// Stack_Report and the degradation ledger record.  Excluding them from every
/// Capture_Station is not a separate step: the grouper only ever sees the decoded
/// images, so a source named here cannot reach it.
pub(super) fn undecodable_source_paths(selection: &[String], decoded: &[ImageInfo]) -> Vec<String> {
    selection
        .iter()
        .filter(|path| !decoded.iter().any(|image| &image.filename == *path))
        .cloned()
        .collect()
}

/// The Stack_Report records of the sources one grouping excluded (需求 1.6 / 1.9).
///
/// Each record names the absolute path of the excluded Source_RAW and the stable
/// reason identifier it was excluded for, and nothing else: the exclusion itself
/// has already happened by the time this runs.
pub(super) fn excluded_source_records(
    images: &[ImageInfo],
    indices: &[usize],
    reason: &'static str,
) -> Vec<stack_report::ExcludedSourceRecord> {
    indices
        .iter()
        .filter_map(|&image_index| images.get(image_index))
        .map(|image| stack_report::ExcludedSourceRecord {
            path: image.filename.clone(),
            reason: reason.to_string(),
        })
        .collect()
}

pub(super) fn absolute_source_path(path: &str) -> String {
    let path = Path::new(path);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(path)
    };
    absolute.to_string_lossy().into_owned()
}
