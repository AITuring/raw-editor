//! Low-frequency affine tone harmonisation for owned focus-stack pixels.
//!
//! Tone correction is deliberately separated from ownership.  The owner image
//! supplies the high-frequency residual while a low-pass version of the source
//! supplies only broad exposure and white-balance changes.  This keeps strokes,
//! edges, and texture tied to the source selected by the ownership map.

#![allow(dead_code)]

use image::{GrayImage, Rgb, Rgb32FImage};
use std::sync::Mutex;

use super::degradation;
use super::report::ToneTileRecord;

/// Build the low-frequency field on one sixteenth of the output grid.
pub(crate) const TONE_GRID_DIVISOR: u32 = 16;
/// Gaussian sigma in analysis-grid pixels (requirement 9.2/11.5).
pub(crate) const TONE_LOW_FREQUENCY_SIGMA: f32 = 4.0;
/// Minimum retained pair samples after robust outlier rejection.
pub(crate) const TONE_MIN_SAMPLES: usize = 1024;
/// Affine gain and offset limits shared by the stack Tone_Harmonizer.
pub(crate) const TONE_MIN_GAIN: f32 = 1.0 / 1.25;
pub(crate) const TONE_MAX_GAIN: f32 = 1.25;
pub(crate) const TONE_MAX_ABS_OFFSET: f32 = 0.02;

// Tone solving is reached from the compositor's free functions, which do not
// carry a mutable Stack_Report reference.  Keep the same run-scoped sink used
// by Intra_Station and Focus_Fuser; report publication takes one snapshot at
// the terminating boundary.
static RUN_RECORDS: Mutex<Vec<ToneTileRecord>> = Mutex::new(Vec::new());

pub(crate) fn reset_run_records() {
    RUN_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

pub(crate) fn record_run_record(record: ToneTileRecord) {
    RUN_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(record);
}

pub(crate) fn run_records_snapshot() -> Vec<ToneTileRecord> {
    RUN_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ToneSample {
    /// Pixel from the already selected owner source.
    pub(crate) owner: [f32; 3],
    /// Pixel from the source whose low-frequency tone is being corrected.
    pub(crate) source: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToneSolveStatus {
    Applied,
    Identity,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToneSolve {
    pub(crate) gain: [f32; 3],
    pub(crate) offset: [f32; 3],
    /// Raw affine coefficients before either bound is applied.
    pub(crate) solved_gain: [f32; 3],
    pub(crate) solved_offset: [f32; 3],
    pub(crate) retained_samples: usize,
    pub(crate) gain_clamped: bool,
    pub(crate) status: ToneSolveStatus,
}

impl ToneSolve {
    pub(crate) fn identity(retained_samples: usize) -> Self {
        Self {
            gain: [1.0; 3],
            offset: [0.0; 3],
            solved_gain: [1.0; 3],
            solved_offset: [0.0; 3],
            retained_samples,
            gain_clamped: false,
            status: ToneSolveStatus::Identity,
        }
    }
}

/// Solve one overlap's affine tone relation after a 3×MAD consistency pass.
///
/// The median/MAD gate is evaluated on the RGB difference vector.  It removes
/// painted content changes before the least-squares affine fit, while retaining
/// exactly the same samples for all three channels.  A pair with fewer than
/// [`TONE_MIN_SAMPLES`] retained samples is identity and records the stable
/// `tone_insufficient_samples` degradation identifier.
pub(crate) fn solve_tone_pair(samples: &[ToneSample]) -> ToneSolve {
    if samples.is_empty() {
        record_insufficient_samples(0);
        record_run_record(ToneTileRecord {
            samples: 0,
            ..ToneTileRecord::default()
        });
        return ToneSolve::identity(0);
    }
    let medians: [f32; 3] = std::array::from_fn(|channel| {
        median(
            samples
                .iter()
                .map(|sample| sample.owner[channel] - sample.source[channel]),
        )
    });
    let residuals: Vec<f32> = samples
        .iter()
        .map(|sample| {
            let difference: [f32; 3] = std::array::from_fn(|channel| {
                sample.owner[channel] - sample.source[channel] - medians[channel]
            });
            difference
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
        })
        .collect();
    let residual_median = median(residuals.iter().copied());
    let mad = median(
        residuals
            .iter()
            .map(|value| (value - residual_median).abs()),
    );
    // When MAD is zero, preserve exact-consensus samples while still allowing
    // finite floating-point noise around that consensus.
    let threshold = (residual_median + 3.0 * mad).max(1e-6);
    let retained: Vec<_> = samples
        .iter()
        .zip(residuals.iter())
        .filter_map(|(sample, residual)| (*residual <= threshold).then_some(*sample))
        .collect();
    if retained.len() < TONE_MIN_SAMPLES {
        record_insufficient_samples(retained.len());
        record_run_record(ToneTileRecord {
            samples: retained.len() as u64,
            ..ToneTileRecord::default()
        });
        return ToneSolve::identity(retained.len());
    }

    let mut solved_gain = [1.0f32; 3];
    let mut solved_offset = [0.0f32; 3];
    for channel in 0..3 {
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xx = 0.0f64;
        let mut sum_xy = 0.0f64;
        for sample in &retained {
            let x = f64::from(sample.source[channel]);
            let y = f64::from(sample.owner[channel]);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }
        let n = retained.len() as f64;
        let denominator = n.mul_add(sum_xx, -sum_x * sum_x);
        if denominator.abs() > f64::EPSILON {
            let gain = (n.mul_add(sum_xy, -sum_x * sum_y) / denominator) as f32;
            let offset = ((sum_y - f64::from(gain) * sum_x) / n) as f32;
            if gain.is_finite() {
                solved_gain[channel] = gain;
            }
            if offset.is_finite() {
                solved_offset[channel] = offset;
            }
        }
    }

    let gain =
        std::array::from_fn(|channel| solved_gain[channel].clamp(TONE_MIN_GAIN, TONE_MAX_GAIN));
    let offset = std::array::from_fn(|channel| {
        solved_offset[channel].clamp(-TONE_MAX_ABS_OFFSET, TONE_MAX_ABS_OFFSET)
    });
    let gain_clamped = solved_gain
        .iter()
        .zip(gain.iter())
        .any(|(solved, bounded)| (*solved - *bounded).abs() > 1e-6)
        || solved_offset
            .iter()
            .zip(offset.iter())
            .any(|(solved, bounded)| (*solved - *bounded).abs() > 1e-6);
    if gain_clamped {
        degradation::record_run_degradation(
            degradation::TONE_GAIN_CLAMPED,
            serde_json::json!({
                "stage": "tone_harmonizer",
                "retained_samples": retained.len(),
                "solved_gain": solved_gain,
                "gain": gain,
                "solved_offset": solved_offset,
                "offset": offset,
            }),
        );
    }
    record_run_record(ToneTileRecord {
        samples: retained.len() as u64,
        gain: gain.map(f64::from),
        offset: offset.map(f64::from),
        solved_gain: solved_gain.map(f64::from),
        solved_offset: solved_offset.map(f64::from),
        clamped: gain_clamped,
        ..ToneTileRecord::default()
    });
    ToneSolve {
        gain,
        offset,
        solved_gain,
        solved_offset,
        retained_samples: retained.len(),
        gain_clamped,
        status: ToneSolveStatus::Applied,
    }
}

/// Compatibility name for call sites that describe this operation as a tone
/// calibration rather than a pair solve.
pub(crate) fn calibrate_tone_pair(samples: &[ToneSample]) -> ToneSolve {
    solve_tone_pair(samples)
}

fn record_insufficient_samples(retained_samples: usize) {
    degradation::record_run_degradation(
        degradation::TONE_INSUFFICIENT_SAMPLES,
        serde_json::json!({
            "stage": "tone_harmonizer",
            "retained_samples": retained_samples,
            "minimum_samples": TONE_MIN_SAMPLES,
        }),
    );
}

/// Estimate a low-frequency RGB field on a 1/16 grid and bilinearly expand it
/// to the input dimensions.
pub(crate) fn low_frequency_field(image: &Rgb32FImage) -> Rgb32FImage {
    if image.width() == 0 || image.height() == 0 {
        return Rgb32FImage::new(image.width(), image.height());
    }
    let low_width = image.width().div_ceil(TONE_GRID_DIVISOR).max(1);
    let low_height = image.height().div_ceil(TONE_GRID_DIVISOR).max(1);
    let low = image::imageops::resize(
        image,
        low_width,
        low_height,
        image::imageops::FilterType::Triangle,
    );
    let low = image::imageops::blur(&low, TONE_LOW_FREQUENCY_SIGMA);
    bilinear_expand(&low, image.width(), image.height())
}

/// Apply the low-frequency affine correction while preserving the owner's
/// high-frequency residual.  Pixels outside `evidence` are exact identity.
pub(crate) fn apply_low_frequency_tone(
    source: &Rgb32FImage,
    owner: &Rgb32FImage,
    evidence: &GrayImage,
    solve: &ToneSolve,
) -> Rgb32FImage {
    let width = owner.width().min(source.width()).min(evidence.width());
    let height = owner.height().min(source.height()).min(evidence.height());
    if width == 0 || height == 0 {
        return Rgb32FImage::new(width, height);
    }
    if solve.status == ToneSolveStatus::Identity {
        return image::imageops::crop_imm(owner, 0, 0, width, height).to_image();
    }
    let source_image = image::imageops::crop_imm(source, 0, 0, width, height).to_image();
    let source_low = low_frequency_field(&source_image);
    let owner_image = image::imageops::crop_imm(owner, 0, 0, width, height).to_image();
    let owner_low = low_frequency_field(&owner_image);
    Rgb32FImage::from_fn(width, height, |x, y| {
        if evidence.get_pixel(x, y)[0] == 0 {
            return *owner_image.get_pixel(x, y);
        }
        let low = source_low.get_pixel(x, y);
        let owned = owner_image.get_pixel(x, y);
        let owned_low = owner_low.get_pixel(x, y);
        Rgb(std::array::from_fn(|channel| {
            solve.gain[channel] * low[channel]
                + solve.offset[channel]
                + (owned[channel] - owned_low[channel])
        }))
    })
}

/// Explicitly named wrapper used by the compositor stage.
pub(crate) fn harmonize_owned_tone(
    source: &Rgb32FImage,
    owner: &Rgb32FImage,
    evidence: &GrayImage,
    solve: &ToneSolve,
) -> Rgb32FImage {
    apply_low_frequency_tone(source, owner, evidence, solve)
}

/// CIEDE2000 for two normalised RGB samples.  Tone boundary diagnostics use
/// the same deterministic colour difference for every owner pair; keeping it
/// here avoids a second, subtly different Delta_E implementation in a later
/// quality-gate stage.
pub(crate) fn delta_e00_rgb(left: [f32; 3], right: [f32; 3]) -> f64 {
    ciede2000(rgb_to_lab(left), rgb_to_lab(right))
}

fn rgb_to_lab(rgb: [f32; 3]) -> [f64; 3] {
    let linear = rgb.map(|value| {
        let value = f64::from(value).clamp(0.0, 1.0);
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    });
    let x = (0.4124564 * linear[0] + 0.3575761 * linear[1] + 0.1804375 * linear[2]) / 0.95047;
    let y = 0.2126729 * linear[0] + 0.7151522 * linear[1] + 0.0721750 * linear[2];
    let z = (0.0193339 * linear[0] + 0.1191920 * linear[1] + 0.9503041 * linear[2]) / 1.08883;
    let f = |value: f64| {
        if value > 216.0 / 24389.0 {
            value.cbrt()
        } else {
            (24389.0 / 27.0 * value + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn ciede2000(left: [f64; 3], right: [f64; 3]) -> f64 {
    let (l1, a1, b1) = (left[0], left[1], left[2]);
    let (l2, a2, b2) = (right[0], right[1], right[2]);
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1 + c2) * 0.5;
    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25.0_f64.powi(7))).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();
    let hp = |a: f64, b: f64| {
        if a == 0.0 && b == 0.0 {
            0.0
        } else {
            let mut angle = b.atan2(a).to_degrees();
            if angle < 0.0 {
                angle += 360.0;
            }
            angle
        }
    };
    let h1p = hp(a1p, b1);
    let h2p = hp(a2p, b2);
    let d_lp = l2 - l1;
    let d_cp = c2p - c1p;
    let dh = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p <= h1p {
        h2p - h1p + 360.0
    } else {
        h2p - h1p - 360.0
    };
    let d_hp = 2.0 * (c1p * c2p).sqrt() * (0.5 * (dh.to_radians())).sin();
    let l_bar = (l1 + l2) * 0.5;
    let c_bar_p = (c1p + c2p) * 0.5;
    let h_bar = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) * 0.5
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) * 0.5
    } else {
        (h1p + h2p - 360.0) * 0.5
    };
    let t = 1.0 - 0.17 * (h_bar - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar).to_radians().cos()
        + 0.32 * (3.0 * h_bar + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar - 63.0).to_radians().cos();
    let delta_theta = 30.0 * (-(h_bar - 275.0).powi(2) / 25.0_f64.powi(2)).exp();
    let rc = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + 25.0_f64.powi(7))).sqrt();
    let sl = 1.0 + 0.015 * (l_bar - 50.0).powi(2) / (20.0 + (l_bar - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let rt = -(2.0 * delta_theta).to_radians().sin() * rc;
    ((d_lp / sl).powi(2)
        + (d_cp / sc).powi(2)
        + (d_hp / sh).powi(2)
        + rt * (d_cp / sc) * (d_hp / sh))
        .max(0.0)
        .sqrt()
}

fn bilinear_expand(low: &Rgb32FImage, width: u32, height: u32) -> Rgb32FImage {
    if low.width() == width && low.height() == height {
        return low.clone();
    }
    let x_scale = if width > 1 {
        (low.width().saturating_sub(1)) as f32 / (width - 1) as f32
    } else {
        0.0
    };
    let y_scale = if height > 1 {
        (low.height().saturating_sub(1)) as f32 / (height - 1) as f32
    } else {
        0.0
    };
    Rgb32FImage::from_fn(width, height, |x, y| {
        let fx = x as f32 * x_scale;
        let fy = y as f32 * y_scale;
        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(low.width().saturating_sub(1));
        let y1 = (y0 + 1).min(low.height().saturating_sub(1));
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let top = low.get_pixel(x0, y0);
        let top_right = low.get_pixel(x1, y0);
        let bottom = low.get_pixel(x0, y1);
        let bottom_right = low.get_pixel(x1, y1);
        Rgb(std::array::from_fn(|channel| {
            let a = top[channel] * (1.0 - tx) + top_right[channel] * tx;
            let b = bottom[channel] * (1.0 - tx) + bottom_right[channel] * tx;
            a * (1.0 - ty) + b * ty
        }))
    })
}

fn median(values: impl Iterator<Item = f32>) -> f32 {
    let mut values: Vec<_> = values.filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fewer_than_1024_samples_keep_identity() {
        let samples = vec![
            ToneSample {
                owner: [0.5; 3],
                source: [0.4; 3],
            };
            TONE_MIN_SAMPLES - 1
        ];
        let solve = solve_tone_pair(&samples);
        assert_eq!(solve.status, ToneSolveStatus::Identity);
        assert_eq!(solve.gain, [1.0; 3]);
        assert_eq!(solve.offset, [0.0; 3]);
    }

    #[test]
    fn affine_fit_rejects_outliers_and_records_raw_coefficients() {
        let mut samples = Vec::with_capacity(TONE_MIN_SAMPLES + 64);
        for index in 0..TONE_MIN_SAMPLES + 64 {
            let source = 0.1 + (index % 31) as f32 / 40.0;
            samples.push(ToneSample {
                owner: [source * 1.1 + 0.01; 3],
                source: [source; 3],
            });
        }
        for sample in &mut samples[..64] {
            sample.owner = [0.95; 3];
        }
        let solve = solve_tone_pair(&samples);
        assert_eq!(solve.status, ToneSolveStatus::Applied);
        assert!((solve.gain[0] - 1.1).abs() < 0.01);
        assert!((solve.offset[0] - 0.01).abs() < 0.01);
        assert!(solve.retained_samples >= TONE_MIN_SAMPLES);
    }

    #[test]
    fn identity_tone_preserves_owner_pixels_outside_evidence() {
        let source = Rgb32FImage::from_pixel(32, 32, Rgb([0.2, 0.3, 0.4]));
        let owner = Rgb32FImage::from_pixel(32, 32, Rgb([0.7, 0.6, 0.5]));
        let evidence = GrayImage::new(32, 32);
        let result = apply_low_frequency_tone(&source, &owner, &evidence, &ToneSolve::identity(0));
        assert_eq!(result, owner);
    }

    #[test]
    fn delta_e00_is_symmetric_and_zero_for_identical_rgb() {
        let left = [0.23, 0.41, 0.67];
        let right = [0.71, 0.36, 0.19];
        assert!(delta_e00_rgb(left, left).abs() < 1e-12);
        assert!((delta_e00_rgb(left, right) - delta_e00_rgb(right, left)).abs() < 1e-12);
    }
}
