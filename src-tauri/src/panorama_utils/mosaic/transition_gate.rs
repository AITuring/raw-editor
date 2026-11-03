//! The low-pass gate of a focus-scan transition seam.
//!
//! Two camera positions vote on focus only where their low-passed views
//! agree. A sharper candidate the agreement keeps out gets a shift search,
//! which tells defocus (lower correlation, same peak) from a registration
//! error (a moved peak).

use image::GrayImage;
use image::Rgb32FImage;

use super::{Field, adjusted, luma, rgb_at};

// Correlation of the low-passed views at and above which a transition cell
// counts as registered, and at and below which it keeps the base.
const SCAN_TRANSITION_AGREEING_CORRELATION: f64 = 0.9;
const SCAN_TRANSITION_DISAGREEING_CORRELATION: f64 = 0.5;
// Relative contrast under which a low-passed window has no structure that
// could be misregistered.
const SCAN_TRANSITION_FLAT_CONTRAST: f64 = 0.02;
// Registration check of a sharper candidate whose low-passed view does not
// correlate with the base: a window of ±8 analysis pixels sampled every 2
// pixels, searched over integer shifts of up to ±4 pixels, counts as
// registered when its best shift is at most one pixel and correlates at least
// 0.5. A frame defocused far beyond the low-pass (祁序2 DSC_6747 under the
// twice as sharp DSC_6760) agrees with no sharp frame there, yet lines up with
// it.
const SCAN_TRANSITION_SHIFT_WINDOW: i64 = 8;
const SCAN_TRANSITION_SHIFT_WINDOW_STEP: usize = 2;
const SCAN_TRANSITION_SHIFT_SEARCH: i64 = 4;
const SCAN_TRANSITION_SHIFT_TOLERANCE: i64 = 1;
const SCAN_TRANSITION_SHIFT_MIN_CORRELATION: f64 = 0.5;

/// Whether two camera positions agree after the low-pass of a focus-scan
/// transition, from 0 (structure displaced) to 1 (registered).
///
/// The views are compared by the correlation of their luma over a 5×5
/// window, so a frame-edge band that is darker from lens falloff still agrees
/// with a sharper registered frame and can be replaced; a displaced stroke
/// does not correlate. A window without structure in either view agrees.
pub(super) fn low_pass_agreement(
    low_base: &Rgb32FImage,
    low_candidate: &Rgb32FImage,
    tone: &Field<3>,
    x: f64,
    y: f64,
) -> f64 {
    const OFFSETS: [f64; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];
    let mut pairs = Vec::with_capacity(OFFSETS.len() * OFFSETS.len());
    for dy in OFFSETS {
        for dx in OFFSETS {
            let (sample_x, sample_y) = (x + dx, y + dy);
            let (Some(base), Some(candidate)) = (
                rgb_at(low_base, sample_x, sample_y),
                rgb_at(low_candidate, sample_x, sample_y),
            ) else {
                continue;
            };
            pairs.push((
                luma(base),
                luma(adjusted(candidate, tone.at(sample_x, sample_y))),
            ));
        }
    }
    if pairs.len() * 2 < OFFSETS.len() * OFFSETS.len() {
        return 0.0;
    }
    let count = pairs.len() as f64;
    let base_mean = pairs.iter().map(|pair| pair.0).sum::<f64>() / count;
    let candidate_mean = pairs.iter().map(|pair| pair.1).sum::<f64>() / count;
    let (mut base_variance, mut candidate_variance, mut covariance) = (0.0, 0.0, 0.0);
    for &(base, candidate) in &pairs {
        let (base, candidate) = (base - base_mean, candidate - candidate_mean);
        base_variance += base * base;
        candidate_variance += candidate * candidate;
        covariance += base * candidate;
    }
    let level = base_mean.max(candidate_mean).max(0.04);
    let flat = |variance: f64| (variance / count).sqrt() / level < SCAN_TRANSITION_FLAT_CONTRAST;
    if flat(base_variance) && flat(candidate_variance) {
        return 1.0;
    }
    let correlation = covariance / (base_variance * candidate_variance).sqrt().max(1e-12);
    ((correlation - SCAN_TRANSITION_DISAGREEING_CORRELATION)
        / (SCAN_TRANSITION_AGREEING_CORRELATION - SCAN_TRANSITION_DISAGREEING_CORRELATION))
        .clamp(0.0, 1.0)
}

/// Luma of the top-left `width`×`height` of a low-passed transition view,
/// row-major, `NaN` where `mask` leaves the pixel uncovered.
pub(super) fn masked_luma_plane(
    image: &Rgb32FImage,
    mask: &GrayImage,
    width: u32,
    height: u32,
) -> Vec<f32> {
    let mut plane = vec![f32::NAN; width as usize * height as usize];
    for y in 0..height.min(image.height()).min(mask.height()) {
        for x in 0..width.min(image.width()).min(mask.width()) {
            if mask.get_pixel(x, y)[0] != 0 {
                plane[(y * width + x) as usize] = luma(*image.get_pixel(x, y)) as f32;
            }
        }
    }
    plane
}

/// What the shift search of [`low_pass_registration`] shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LowPassRegistration {
    /// The best shift lies within the tolerance of zero.
    Registered,
    /// The best shift lies further out: the candidate is displaced.
    Displaced,
    /// No shift correlates well enough, or the window leaves the coverage.
    Unknown,
}

/// Whether the candidate's low-passed view lines up with the base's around
/// analysis position (`x`, `y`): of the integer shifts up to
/// [`SCAN_TRANSITION_SHIFT_SEARCH`] analysis pixels, the one that correlates a
/// window of the candidate best with the base's lies within
/// [`SCAN_TRANSITION_SHIFT_TOLERANCE`] of zero.
///
/// Defocus blurs a view symmetrically: it lowers the correlation of the two
/// views but does not move its peak, while a registration error does. A
/// window that reaches an uncovered pixel, or whose correlation peak stays
/// below [`SCAN_TRANSITION_SHIFT_MIN_CORRELATION`], shows nothing.
pub(super) fn low_pass_registration(
    base: &[f32],
    candidate: &[f32],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
) -> LowPassRegistration {
    let reach = SCAN_TRANSITION_SHIFT_WINDOW + SCAN_TRANSITION_SHIFT_SEARCH;
    let (cx, cy) = (x.round() as i64, y.round() as i64);
    if cx < reach || cy < reach || cx + reach >= width as i64 || cy + reach >= height as i64 {
        return LowPassRegistration::Unknown;
    }
    let at = |plane: &[f32], px: i64, py: i64| plane[py as usize * width + px as usize];
    let offsets = (-SCAN_TRANSITION_SHIFT_WINDOW..=SCAN_TRANSITION_SHIFT_WINDOW)
        .step_by(SCAN_TRANSITION_SHIFT_WINDOW_STEP)
        .collect::<Vec<_>>();
    let mut window = Vec::with_capacity(offsets.len() * offsets.len());
    for &oy in &offsets {
        for &ox in &offsets {
            let value = at(base, cx + ox, cy + oy);
            if !value.is_finite() {
                return LowPassRegistration::Unknown;
            }
            window.push(f64::from(value));
        }
    }
    let count = window.len() as f64;
    let base_mean = window.iter().sum::<f64>() / count;
    let base_centred = window
        .iter()
        .map(|value| value - base_mean)
        .collect::<Vec<_>>();
    let base_norm = base_centred
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if base_norm <= 1e-9 {
        return LowPassRegistration::Unknown;
    }
    let mut best = (f64::NEG_INFINITY, 0i64, 0i64);
    let mut samples = Vec::with_capacity(window.len());
    for dy in -SCAN_TRANSITION_SHIFT_SEARCH..=SCAN_TRANSITION_SHIFT_SEARCH {
        for dx in -SCAN_TRANSITION_SHIFT_SEARCH..=SCAN_TRANSITION_SHIFT_SEARCH {
            samples.clear();
            for &oy in &offsets {
                for &ox in &offsets {
                    samples.push(f64::from(at(candidate, cx + ox + dx, cy + oy + dy)));
                }
            }
            if samples.iter().any(|value| !value.is_finite()) {
                return LowPassRegistration::Unknown;
            }
            let mean = samples.iter().sum::<f64>() / count;
            let (mut covariance, mut variance) = (0.0, 0.0);
            for (sample, base) in samples.iter().zip(&base_centred) {
                let centred = sample - mean;
                covariance += centred * base;
                variance += centred * centred;
            }
            let correlation = covariance / (variance.sqrt() * base_norm).max(1e-12);
            if correlation > best.0 {
                best = (correlation, dx, dy);
            }
        }
    }
    if best.0 < SCAN_TRANSITION_SHIFT_MIN_CORRELATION {
        LowPassRegistration::Unknown
    } else if best.1.abs() <= SCAN_TRANSITION_SHIFT_TOLERANCE
        && best.2.abs() <= SCAN_TRANSITION_SHIFT_TOLERANCE
    {
        LowPassRegistration::Registered
    } else {
        LowPassRegistration::Displaced
    }
}

/// Mean of the finite values of a `width`×`height` grid over the
/// `(2 radius + 1)²` box around every cell; `NaN` where the box holds none.
/// Non-finite entries take no part, so fixed cells do not dilute the mean.
pub(super) fn box_mean_finite(
    values: &[f64],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<f64> {
    // Summed-area tables of the finite values and of their count.
    let stride = width + 1;
    let mut sums = vec![0.0f64; stride * (height + 1)];
    let mut counts = vec![0u32; stride * (height + 1)];
    for y in 0..height {
        for x in 0..width {
            let value = values[y * width + x];
            let (add, one) = if value.is_finite() {
                (value, 1)
            } else {
                (0.0, 0)
            };
            let index = (y + 1) * stride + x + 1;
            sums[index] = add + sums[index - 1] + sums[index - stride] - sums[index - stride - 1];
            counts[index] =
                one + counts[index - 1] + counts[index - stride] - counts[index - stride - 1];
        }
    }
    let mut means = vec![f64::NAN; width * height];
    for y in 0..height {
        let (y0, y1) = (y.saturating_sub(radius), (y + radius + 1).min(height));
        for x in 0..width {
            let (x0, x1) = (x.saturating_sub(radius), (x + radius + 1).min(width));
            let corner = |cx: usize, cy: usize| cy * stride + cx;
            let count = counts[corner(x1, y1)] + counts[corner(x0, y0)]
                - counts[corner(x0, y1)]
                - counts[corner(x1, y0)];
            if count > 0 {
                let sum = sums[corner(x1, y1)] + sums[corner(x0, y0)]
                    - sums[corner(x0, y1)]
                    - sums[corner(x1, y0)];
                means[y * width + x] = sum / f64::from(count);
            }
        }
    }
    means
}
