//! Pure Quality_Gate pixel measurements on one ROI (tasks 15.3 to 15.5).
//!
//! Slanted-edge detection and MTF50, normalised gradient energy, flat-ROI noise
//! sigma and the low-frequency CIEDE2000 colour difference.

// ==================== task 15.3: slanted edge and MTF50 ====================
// Stage 7 metric implementation fragment, to be inserted at quality_gate module scope.
// No run sinks or output mutations. Input RGB is final-output sRGB, not linear RGB.
// The installed imageproc supplies ordinary deterministic Hough (not probabilistic
// Hough). We validate finite, contiguous line support explicitly after Hough.

pub(crate) const MTF_EDGE_MIN_LENGTH_PX: f64 = 128.0;

pub(crate) const MTF_EDGE_MIN_ANGLE_DEG: f64 = 3.0;

pub(crate) const MTF_EDGE_MAX_ANGLE_DEG: f64 = 15.0;

pub(crate) const MTF_EDGE_MIN_CONTRAST: f64 = 0.20;

pub(crate) const MTF_LINE_MAX_RMS_PX: f64 = 0.5;

pub(crate) const FLAT_LOW_FREQUENCY_STD_MAX: f64 = 0.02;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SlantedEdgeEvidence {
    /// false: x = slope*y + intercept; true: y = slope*x + intercept.
    pub transpose: bool,
    pub slope: f64,
    pub intercept: f64,
    pub angle_deg: f64,
    pub length_px: f64,
    pub contrast: f64,
    pub line_fit_rms_px: f64,
    pub first_row: u32,
    pub last_row: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mtf50Measurement {
    pub f50_cycles_per_output_pixel: f64,
    pub normalized: f64,
    pub oversampling: usize,
    pub edge: SlantedEdgeEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FlatRoiEvidence {
    pub low_frequency_luma_std: f64,
    pub opaque_pixels: u64,
}

pub(super) fn metric_linear_channel(value: f32) -> f64 {
    let value = f64::from(value);
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

pub(super) fn metric_luminance(image: &image::Rgb32FImage) -> Option<Vec<f64>> {
    if image.width() == 0 || image.height() == 0 || image.as_raw().iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(
        image
            .pixels()
            .map(|pixel| {
                let rgb = pixel.0.map(metric_linear_channel);
                0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            })
            .collect(),
    )
}

pub(super) fn metric_median(mut values: Vec<f64>) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let center = values.len() / 2;
    if values.is_empty() {
        0.0
    } else if values.len().is_multiple_of(2) {
        (values[center - 1] + values[center]) * 0.5
    } else {
        values[center]
    }
}

/// Gaussian sigma 2.0, radius 6, exactly 13 taps; replicated borders are used
/// identically for reference and output. The caller never resamples output.
pub(super) fn metric_gaussian_sigma2(plane: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut kernel = [0.0f64; 13];
    for (index, weight) in kernel.iter_mut().enumerate() {
        *weight = (-0.5 * ((index as f64 - 6.0) / 2.0).powi(2)).exp();
    }
    let total: f64 = kernel.iter().sum();
    for weight in &mut kernel {
        *weight /= total;
    }
    let mut horizontal = vec![0.0; plane.len()];
    for y in 0..height {
        for x in 0..width {
            horizontal[y * width + x] = kernel
                .iter()
                .enumerate()
                .map(|(k, weight)| {
                    let nx = (x as isize + k as isize - 6).clamp(0, width as isize - 1) as usize;
                    weight * plane[y * width + nx]
                })
                .sum();
        }
    }
    let mut output = vec![0.0; plane.len()];
    for y in 0..height {
        for x in 0..width {
            output[y * width + x] = kernel
                .iter()
                .enumerate()
                .map(|(k, weight)| {
                    let ny = (y as isize + k as isize - 6).clamp(0, height as isize - 1) as usize;
                    weight * horizontal[ny * width + x]
                })
                .sum();
        }
    }
    output
}

pub(super) fn metric_row_value(
    plane: &[f64],
    width: usize,
    transpose: bool,
    row: u32,
    column: usize,
) -> f64 {
    if transpose {
        plane[column * width + row as usize]
    } else {
        plane[row as usize * width + column]
    }
}

/// Catmull-Rom cubic interpolation in one row, used only to localise an edge;
/// it does not resample or replace any output ROI pixel.
pub(super) fn metric_cubic_row(
    plane: &[f64],
    width: usize,
    columns: usize,
    transpose: bool,
    row: u32,
    x: f64,
) -> f64 {
    let base = x.floor() as isize;
    let fraction = x - base as f64;
    let p: [f64; 4] = std::array::from_fn(|i| {
        let column = (base + i as isize - 1).clamp(0, columns as isize - 1) as usize;
        metric_row_value(plane, width, transpose, row, column)
    });
    p[1] + 0.5
        * fraction
        * (p[2] - p[0]
            + fraction
                * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3]
                    + fraction * (3.0 * (p[1] - p[2]) + p[3] - p[0])))
}

pub(super) fn metric_fit_line(points: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    if points.len() < 3 {
        return None;
    }
    let n = points.len() as f64;
    let mean_row = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_column = points.iter().map(|p| p.1).sum::<f64>() / n;
    let covariance = points
        .iter()
        .map(|p| (p.0 - mean_row) * (p.1 - mean_column))
        .sum::<f64>();
    let variance = points.iter().map(|p| (p.0 - mean_row).powi(2)).sum::<f64>();
    if variance <= 0.0 {
        return None;
    }
    let slope = covariance / variance;
    let intercept = mean_column - slope * mean_row;
    let rms = (points
        .iter()
        .map(|p| (p.1 - slope * p.0 - intercept).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    Some((slope, intercept, rms))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn metric_refine_edge(
    plane: &[f64],
    width: usize,
    height: usize,
    transpose: bool,
    slope: f64,
    intercept: f64,
    first_row: u32,
    last_row: u32,
) -> Option<SlantedEdgeEvidence> {
    let columns = if transpose { height } else { width };
    let mut points = Vec::new();
    let mut contrasts = Vec::new();
    for row in first_row..=last_row {
        let center = slope * f64::from(row) + intercept;
        if center < 64.0 || center + 64.0 >= columns as f64 {
            continue;
        }
        let mean = |side: f64| -> f64 {
            (24..=32)
                .map(|distance| {
                    metric_cubic_row(
                        plane,
                        width,
                        columns,
                        transpose,
                        row,
                        center + side * f64::from(distance),
                    )
                })
                .sum::<f64>()
                / 9.0
        };
        let contrast = mean(1.0) - mean(-1.0);
        if contrast.abs() < MTF_EDGE_MIN_CONTRAST {
            continue;
        }
        let direction = contrast.signum();
        // Integrate the derivative of the reconstructed cubic row in a 24px
        // search band. This is a subpixel gradient centroid, not an integer
        // threshold crossing; it treats both edge polarities identically.
        let mut weighted_x = 0.0;
        let mut total_weight = 0.0;
        for index in 0..96 {
            let x = center - 12.0 + index as f64 * 0.25;
            let left = metric_cubic_row(plane, width, columns, transpose, row, x);
            let right = metric_cubic_row(plane, width, columns, transpose, row, x + 0.25);
            let weight = (direction * (right - left)).max(0.0);
            weighted_x += (x + 0.125) * weight;
            total_weight += weight;
        }
        if total_weight > 0.05 {
            points.push((f64::from(row), weighted_x / total_weight));
            contrasts.push(contrast.abs());
        }
    }
    let &(start, _) = points.first()?;
    let &(end, _) = points.last()?;
    let (slope, intercept, rms) = metric_fit_line(&points)?;
    let length = (end - start) * (1.0 + slope * slope).sqrt();
    if length < MTF_EDGE_MIN_LENGTH_PX {
        return None;
    }
    Some(SlantedEdgeEvidence {
        transpose,
        slope,
        intercept,
        angle_deg: slope.abs().atan().to_degrees(),
        length_px: length,
        contrast: metric_median(contrasts),
        line_fit_rms_px: rms,
        first_row: start as u32,
        last_row: end as u32,
    })
}

/// Fixed Canny + ordinary Hough followed by explicit contiguous segment support.
/// The ordinary-Hough limitation is intentional and must remain visible in the
/// stage execution record; it is not called probabilistic Hough.
pub(crate) fn detect_slanted_edge(
    image: &image::Rgb32FImage,
) -> Result<SlantedEdgeEvidence, &'static str> {
    use crate::panorama_utils::stack_pipeline::degradation::{
        ROI_NOT_SLANTED_EDGE, SLANTED_EDGE_ANGLE_OUT_OF_RANGE, SLANTED_EDGE_CONTRAST_INSUFFICIENT,
        SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED, SLANTED_EDGE_TOO_SHORT,
    };
    let plane = metric_luminance(image).ok_or(ROI_NOT_SLANTED_EDGE)?;
    let (width, height) = (image.width() as usize, image.height() as usize);
    if width < 129 || height < 129 {
        return Err(ROI_NOT_SLANTED_EDGE);
    }
    let gray = image::GrayImage::from_fn(width as u32, height as u32, |x, y| {
        image::Luma([
            (plane[y as usize * width + x as usize].clamp(0.0, 1.0) * 255.0).round() as u8,
        ])
    });
    let edges = imageproc::edges::canny(&gray, 12.0, 24.0);
    let lines = imageproc::hough::detect_lines(
        &edges,
        imageproc::hough::LineDetectionOptions {
            vote_threshold: 64,
            suppression_radius: 4,
        },
    );
    let no_lines = lines.is_empty();
    let points: Vec<(f64, f64)> = edges
        .enumerate_pixels()
        .filter_map(|(x, y, p)| (p[0] != 0).then_some((f64::from(x), f64::from(y))))
        .collect();
    let mut candidates = Vec::new();
    let mut had_support = false;
    let mut had_long_support = false;
    let mut had_angle_candidate = false;
    for line in lines {
        let angle = f64::from(line.angle_in_degrees).to_radians();
        let (sine, cosine) = angle.sin_cos();
        let transpose = sine.abs() > cosine.abs();
        let (slope, intercept, rows) = if transpose {
            (-cosine / sine, f64::from(line.r) / sine, width)
        } else {
            (-sine / cosine, f64::from(line.r) / cosine, height)
        };
        // Keep near-axis candidates a little outside the final 3..15 degree
        // band; the subpixel line fit, not the 1-degree Hough bin, sets angle.
        if slope.abs().atan().to_degrees() > 17.0 {
            had_angle_candidate = true;
            continue;
        }
        let mut support = vec![false; rows];
        for &(x, y) in &points {
            if (x * cosine + y * sine - f64::from(line.r)).abs() <= 2.5 {
                let row = if transpose { x as usize } else { y as usize };
                support[row] = true;
            }
        }
        let mut runs = Vec::new();
        let mut start = None;
        let mut last = 0;
        for (row, present) in support.into_iter().enumerate() {
            if present {
                if start.is_none() {
                    start = Some(row);
                }
                last = row;
            } else if start.is_some() && row > last + 3 {
                runs.push((start.take().unwrap_or(0), last));
            }
        }
        if let Some(start) = start {
            runs.push((start, last));
        }
        for (start, end) in runs {
            had_support = true;
            if ((end - start) as f64) * (1.0 + slope * slope).sqrt() < MTF_EDGE_MIN_LENGTH_PX {
                continue;
            }
            had_long_support = true;
            if let Some(candidate) = metric_refine_edge(
                &plane,
                width,
                height,
                transpose,
                slope,
                intercept,
                start as u32,
                end as u32,
            ) {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.length_px
            .total_cmp(&a.length_px)
            .then_with(|| a.transpose.cmp(&b.transpose))
            .then_with(|| a.intercept.total_cmp(&b.intercept))
    });
    let mut found_bad_fit = false;
    for candidate in &candidates {
        if candidate.angle_deg < MTF_EDGE_MIN_ANGLE_DEG - 1e-6
            || candidate.angle_deg > MTF_EDGE_MAX_ANGLE_DEG + 1e-6
        {
            had_angle_candidate = true;
            continue;
        }
        if candidate.line_fit_rms_px > MTF_LINE_MAX_RMS_PX {
            found_bad_fit = true;
            continue;
        }
        let isolated = !candidates.iter().any(|other| {
            if other.transpose != candidate.transpose || other.contrast < MTF_EDGE_MIN_CONTRAST {
                return false;
            }
            let first = candidate.first_row.max(other.first_row);
            let last = candidate.last_row.min(other.last_row);
            if last <= first {
                return false;
            }
            // Ignore duplicate hypotheses for the same physical edge, but
            // reject a second contrasting edge anywhere in the ±32px band.
            let row = f64::from(first + last) * 0.5;
            let distance = ((candidate.slope - other.slope) * row + candidate.intercept
                - other.intercept)
                .abs()
                / (1.0 + candidate.slope * candidate.slope).sqrt();
            distance > 3.0 && distance <= 32.0
        });
        if isolated {
            return Ok(candidate.clone());
        }
    }
    Err(if found_bad_fit {
        SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED
    } else if had_angle_candidate {
        SLANTED_EDGE_ANGLE_OUT_OF_RANGE
    } else if had_long_support {
        SLANTED_EDGE_CONTRAST_INSUFFICIENT
    } else if had_support || no_lines {
        SLANTED_EDGE_TOO_SHORT
    } else {
        ROI_NOT_SLANTED_EDGE
    })
}

/// ISO 12233-style slanted-edge measurement. ESF bins represent physical
/// normal distance, so f50 is cycles/output-pixel before Local_Scale division.
pub(crate) fn slanted_edge_mtf50(
    image: &image::Rgb32FImage,
    local_scale: f64,
) -> Result<Mtf50Measurement, &'static str> {
    use crate::panorama_utils::stack_pipeline::degradation::{
        PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED, ROI_NOT_SLANTED_EDGE,
    };
    if !local_scale.is_finite() || local_scale <= 0.0 {
        return Err(PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED);
    }
    let edge = detect_slanted_edge(image)?;
    let plane = metric_luminance(image).ok_or(ROI_NOT_SLANTED_EDGE)?;
    let oversampling = (1.0 / edge.angle_deg.to_radians().tan())
        .round()
        .clamp(4.0, 16.0) as usize;
    let bins = 128 * oversampling + 1;
    let mut sums = vec![0.0; bins];
    let mut counts = vec![0u32; bins];
    let width = image.width() as usize;
    let columns = if edge.transpose {
        image.height()
    } else {
        image.width()
    };
    let normalizer = (1.0 + edge.slope * edge.slope).sqrt();
    for row in edge.first_row..=edge.last_row {
        let center = edge.slope * f64::from(row) + edge.intercept;
        let start = (center - 64.0 * normalizer).ceil().max(0.0) as u32;
        let end = (center + 64.0 * normalizer)
            .floor()
            .min(f64::from(columns - 1)) as u32;
        for column in start..=end {
            let distance = (f64::from(column) - center) / normalizer;
            let bin = ((distance + 64.0) * oversampling as f64).round() as usize;
            if bin < bins {
                sums[bin] += metric_row_value(&plane, width, edge.transpose, row, column as usize);
                counts[bin] += 1;
            }
        }
    }
    let populated: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter_map(|(i, &count)| (count > 0).then_some(i))
        .collect();
    if populated.len() < bins / 2 {
        return Err(ROI_NOT_SLANTED_EDGE);
    }
    let mut esf = vec![0.0; bins];
    for &i in &populated {
        esf[i] = sums[i] / f64::from(counts[i]);
    }
    let first = populated[0];
    let last = *populated.last().ok_or(ROI_NOT_SLANTED_EDGE)?;
    let first_value = esf[first];
    let last_value = esf[last];
    esf[..first].fill(first_value);
    esf[last + 1..].fill(last_value);
    for pair in populated.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        for i in a + 1..b {
            esf[i] = esf[a] + (esf[b] - esf[a]) * (i - a) as f64 / (b - a) as f64;
        }
    }
    let smooth: Vec<f64> = esf
        .windows(4)
        .map(|window| window.iter().sum::<f64>() / 4.0)
        .collect();
    let lsf_len = smooth.len() - 1;
    let lsf: Vec<f64> = smooth
        .windows(2)
        .enumerate()
        .map(|(i, window)| {
            (window[1] - window[0])
                * (0.54 - 0.46 * (std::f64::consts::TAU * i as f64 / (lsf_len - 1) as f64).cos())
        })
        .collect();
    let dc = lsf.iter().sum::<f64>().abs();
    if dc <= 1e-12 {
        return Err(ROI_NOT_SLANTED_EDGE);
    }
    let nfft = lsf.len().next_power_of_two();
    let mut previous = (0.0, 1.0);
    // A real DFT with implicit zero padding: bins above output Nyquist are
    // unmeasurable as an output-image frequency and are never accepted.
    for k in 1..=nfft / (2 * oversampling) {
        let omega = std::f64::consts::TAU * k as f64 / nfft as f64;
        let (mut real, mut imaginary) = (0.0, 0.0);
        for (n, &value) in lsf.iter().enumerate() {
            let (sine, cosine) = (omega * n as f64).sin_cos();
            real += value * cosine;
            imaginary -= value * sine;
        }
        let frequency = k as f64 * oversampling as f64 / nfft as f64;
        let amplitude = real.hypot(imaginary) / dc;
        if amplitude <= 0.5 && previous.1 > 0.5 {
            let fraction = (previous.1 - 0.5) / (previous.1 - amplitude);
            let f50 = previous.0 + fraction * (frequency - previous.0);
            return Ok(Mtf50Measurement {
                f50_cycles_per_output_pixel: f50,
                normalized: f50 / local_scale,
                oversampling,
                edge,
            });
        }
        previous = (frequency, amplitude);
    }
    Err(ROI_NOT_SLANTED_EDGE)
}

// ==================== task 15.4: gradient, flat ROI and noise ====================
/// Uses exactly the production acutance primitive, with RGB replaced by equal
/// linear-luminance channels so chroma differences cannot masquerade as luma
/// gradient energy. Replication gives every ROI pixel a complete 13x13 patch.
pub(crate) fn normalized_gradient_energy(
    image: &image::Rgb32FImage,
    local_scale: f64,
) -> Result<f64, &'static str> {
    use crate::panorama_utils::stack_pipeline::degradation::{
        OWNER_SOURCE_UNDECODABLE, PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED,
    };
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
            crate::panorama_utils::mosaic::acutance_with_step(
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
    use crate::panorama_utils::stack_pipeline::degradation::ROI_NOT_FLAT;
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
    let plane = metric_luminance(image)
        .ok_or(crate::panorama_utils::stack_pipeline::degradation::ROI_NOT_FLAT)?;
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
        return Err(
            crate::panorama_utils::stack_pipeline::degradation::PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED,
        );
    }
    let output_sigma = noise_sigma(output)?;
    let reference_sigma = noise_sigma(reference)?;
    if reference_sigma <= f64::EPSILON {
        return Err(crate::panorama_utils::stack_pipeline::degradation::ROI_NOT_FLAT);
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
        return Err(
            crate::panorama_utils::stack_pipeline::degradation::PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED,
        );
    }
    let a = roi_mean_rgb(output)
        .ok_or(crate::panorama_utils::stack_pipeline::degradation::OWNER_SOURCE_UNDECODABLE)?;
    let b = roi_mean_rgb(reference)
        .ok_or(crate::panorama_utils::stack_pipeline::degradation::OWNER_SOURCE_UNDECODABLE)?;
    Ok(crate::panorama_utils::stack_pipeline::tone::delta_e00_rgb(
        a, b,
    ))
}
