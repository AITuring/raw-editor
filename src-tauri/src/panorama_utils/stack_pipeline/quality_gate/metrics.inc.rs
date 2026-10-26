// Quality_Gate module fragment; included at parent module scope.

// ==================== task 15.4: gradient, flat ROI and noise ====================
/// Uses exactly the production acutance primitive, with RGB replaced by equal
/// linear-luminance channels so chroma differences cannot masquerade as luma
/// gradient energy. Replication gives every ROI pixel a complete 13x13 patch.
pub(crate) fn normalized_gradient_energy(
    image: &image::Rgb32FImage,
    local_scale: f64,
) -> Result<f64, &'static str> {
    use super::degradation::{OWNER_SOURCE_UNDECODABLE, PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED};
    if !local_scale.is_finite() || local_scale <= 0.0 {
        return Err(PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED);
    }
    let plane = metric_luminance(image).ok_or(OWNER_SOURCE_UNDECODABLE)?;
    let (width, height) = (image.width() as usize, image.height() as usize);
    use rayon::prelude::*;
    let values: Vec<f64> = (0..width * height)
        .into_par_iter()
        .map(|index| {
            let x = index % width;
            let y = index / width;
            super::super::mosaic::acutance_with_step(
                |sx, sy| {
                    let nx = (sx.round() as isize).clamp(0, width as isize - 1) as usize;
                    let ny = (sy.round() as isize).clamp(0, height as isize - 1) as usize;
                    Some(image::Rgb([plane[ny * width + nx] as f32; 3]))
                },
                x as f64,
                y as f64,
                1.0,
            )
        })
        .collect();
    // Summing the indexed collection serially preserves the previous
    // floating-point reduction order and therefore the report values.
    let total = values.into_iter().sum::<f64>();
    Ok(total / plane.len() as f64 / local_scale)
}

pub(crate) fn flat_roi(
    image: &image::Rgb32FImage,
    coverage: &image::GrayImage,
) -> Result<FlatRoiEvidence, &'static str> {
    use super::degradation::ROI_NOT_FLAT;
    if image.dimensions() != coverage.dimensions() || coverage.pixels().any(|p| p[0] != 255) {
        return Err(ROI_NOT_FLAT);
    }
    let plane = metric_luminance(image).ok_or(ROI_NOT_FLAT)?;
    let low = metric_gaussian_sigma2(&plane, image.width() as usize, image.height() as usize);
    let mean = low.iter().sum::<f64>() / low.len() as f64;
    let std =
        (low.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / low.len() as f64).sqrt();
    if std > FLAT_LOW_FREQUENCY_STD_MAX || detect_slanted_edge(image).is_ok() {
        return Err(ROI_NOT_FLAT);
    }
    Ok(FlatRoiEvidence {
        low_frequency_luma_std: std,
        opaque_pixels: plane.len() as u64,
    })
}

/// Defined estimator (not variance-renormalised): sigma=1.4826*MAD(highpass),
/// with highpass = linear luminance - sigma2/radius6 Gaussian luminance.
pub(crate) fn noise_sigma(image: &image::Rgb32FImage) -> Result<f64, &'static str> {
    let plane = metric_luminance(image).ok_or(super::degradation::ROI_NOT_FLAT)?;
    let low = metric_gaussian_sigma2(&plane, image.width() as usize, image.height() as usize);
    let high: Vec<f64> = plane
        .iter()
        .zip(low)
        .map(|(value, blurred)| value - blurred)
        .collect();
    let center = metric_median(high.clone());
    Ok(1.4826 * metric_median(high.into_iter().map(|v| (v - center).abs()).collect()))
}

pub(crate) fn noise_sigma_ratio(
    output: &image::Rgb32FImage,
    reference: &image::Rgb32FImage,
) -> Result<f64, &'static str> {
    if output.dimensions() != reference.dimensions() {
        return Err(super::degradation::PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED);
    }
    let output_sigma = noise_sigma(output)?;
    let reference_sigma = noise_sigma(reference)?;
    if reference_sigma <= f64::EPSILON {
        return Err(super::degradation::ROI_NOT_FLAT);
    }
    Ok(output_sigma / reference_sigma)
}

// ==================== task 15.5: low-frequency CIEDE2000 ====================
/// The design fixes the ROI low-frequency mean to the mean of its fully opaque
/// RGB samples. Using compensated f64 summation avoids a large RGB lowpass
/// buffer and feeds the one canonical sRGB->D65 Lab->CIEDE2000 implementation.
pub(crate) fn roi_mean_rgb(image: &image::Rgb32FImage) -> Option<[f32; 3]> {
    if image.width() == 0 || image.height() == 0 {
        return None;
    }
    let mut sums = [0.0f64; 3];
    let mut compensation = [0.0f64; 3];
    for pixel in image.pixels() {
        for channel in 0..3 {
            if !pixel[channel].is_finite() {
                return None;
            }
            let value = f64::from(pixel[channel]) - compensation[channel];
            let next = sums[channel] + value;
            compensation[channel] = (next - sums[channel]) - value;
            sums[channel] = next;
        }
    }
    let count = f64::from(image.width()) * f64::from(image.height());
    Some(sums.map(|v| (v / count) as f32))
}

pub(crate) fn roi_low_frequency_delta_e00(
    output: &image::Rgb32FImage,
    reference: &image::Rgb32FImage,
) -> Result<f64, &'static str> {
    if output.dimensions() != reference.dimensions() {
        return Err(super::degradation::PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED);
    }
    let a = roi_mean_rgb(output).ok_or(super::degradation::OWNER_SOURCE_UNDECODABLE)?;
    let b = roi_mean_rgb(reference).ok_or(super::degradation::OWNER_SOURCE_UNDECODABLE)?;
    Ok(super::tone::delta_e00_rgb(a, b))
}

