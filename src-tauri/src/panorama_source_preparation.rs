//! Source preparation and RAW display calibration for panorama/focus runs.
//!
//! This module owns file decoding, camera exposure normalization, bounded
//! display gains, and the small rescue-image path used by disconnected focus
//! components. Matching and pose modules consume its prepared outputs.

use super::*;

pub(super) struct PreparedStackSource {
    pub(super) image: DynamicImage,
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
            image = DynamicImage::ImageRgb32F(corrected);
        }
    }
    Ok(PreparedStackSource {
        image,
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
        overview_reference: false,
        foreground_range,
    })
}
