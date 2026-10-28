//! Automatic adjustment analysis.
//!
//! Estimates exposure, tone and white balance from a downscaled image and
//! reports them as adjustment values.

use image::DynamicImage;
use image::RgbImage;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;

use super::downscale_f32_image;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AutoAdjustmentResults {
    pub exposure: f64,
    pub brightness: f64,
    pub contrast: f64,
    pub highlights: f64,
    pub shadows: f64,
    pub vibrancy: f64,
    pub vignette_amount: f64,
    pub temperature: f64,
    pub tint: f64,
    pub dehaze: f64,
    pub clarity: f64,
    pub centre: f64,
    pub blacks: f64,
    pub whites: f64,
}

pub fn perform_auto_analysis(image: &DynamicImage) -> AutoAdjustmentResults {
    const ANALYSIS_MAX_DIM: u32 = 1024;

    const LUMA_R: f32 = 0.2126;
    const LUMA_G: f32 = 0.7152;
    const LUMA_B: f32 = 0.0722;

    const EXPOSURE_MIDPOINT: f64 = 128.0;
    const EXPOSURE_SCALE: f64 = 0.125;
    const WHITE_POINT_HARD_LIMIT: usize = 245;
    const HIGHLIGHT_LUMA_THRESHOLD: usize = 240;
    const CLIPPED_LUMA_THRESHOLD: usize = 250;
    const HIGHLIGHT_PERCENT_THRESHOLD: f64 = 0.02;
    const CLIPPED_PERCENT_THRESHOLD: f64 = 0.005;
    const EXPOSURE_CEILING: f64 = 250.0;

    const TARGET_RANGE: f64 = 220.0;
    const CONTRAST_SCALE: f64 = 10.0;
    const HIGHLIGHT_CONTRAST_REDUCE: f64 = 0.5;

    const SHADOW_LUMA_MAX: usize = 32;
    const SHADOW_PERCENT_THRESHOLD: f64 = 0.05;
    const SHADOW_BOOST_SCALE: f64 = 40.0;
    const SHADOW_MAX: f64 = 50.0;
    const HIGHLIGHT_BOOST_SCALE: f64 = 120.0;
    const HIGHLIGHT_MAX: f64 = 70.0;

    const VIBRANCY_SAT_THRESHOLD: f32 = 0.2;
    const VIBRANCY_SCALE: f64 = 120.0;

    const DEHAZE_RANGE_THRESHOLD: f64 = 120.0;
    const DEHAZE_SAT_THRESHOLD: f32 = 0.15;
    const DEHAZE_SCALE: f64 = 35.0;
    const CLARITY_RANGE_THRESHOLD: f64 = 180.0;
    const CLARITY_SCALE: f64 = 50.0;

    const VIGNETTE_CENTER_LOW: f32 = 0.25;
    const VIGNETTE_CENTER_HIGH: f32 = 0.75;

    const VIGNETTE_SCALE: f64 = 100.0;
    const VIGNETTE_CENTRE_DIFF_THRESHOLD: f32 = 0.05;
    const CENTRE_SCALE: f64 = 100.0;
    const CENTRE_MAX: f64 = 60.0;

    const MID_GRAY: f64 = 128.0;
    const BLACKS_SCALE: f64 = 0.5;
    const WHITES_SCALE: f64 = 0.2;
    const EXPOSURE_OUTPUT_SCALE: f64 = 20.0;
    const BRIGHTNESS_SCALE: f64 = 0.007;

    let analysis_preview = downscale_f32_image(image, ANALYSIS_MAX_DIM, ANALYSIS_MAX_DIM);
    let rgb_image = analysis_preview.to_rgb8();
    let (auto_temperature, auto_tint) = estimate_auto_white_balance(&rgb_image);
    let total_pixels = (rgb_image.width() * rgb_image.height()) as f64;

    let (width, height) = rgb_image.dimensions();
    let cx0 = (width as f32 * VIGNETTE_CENTER_LOW) as u32;
    let cx1 = (width as f32 * VIGNETTE_CENTER_HIGH) as u32;
    let cy0 = (height as f32 * VIGNETTE_CENTER_LOW) as u32;
    let cy1 = (height as f32 * VIGNETTE_CENTER_HIGH) as u32;

    let mut luma_hist = vec![0u32; 256];
    let mut mean_saturation = 0.0f32;
    let mut center_sum = 0.0f32;
    let mut edge_sum = 0.0f32;
    let mut center_n = 0u32;
    let mut edge_n = 0u32;

    for (x, y, pixel) in rgb_image.enumerate_pixels() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        let luma_f = LUMA_R * r + LUMA_G * g + LUMA_B * b;
        luma_hist[(luma_f.round() as usize).min(255)] += 1;

        let r_n = r / 255.0;
        let g_n = g / 255.0;
        let b_n = b / 255.0;
        let max_c = r_n.max(g_n).max(b_n);
        let min_c = r_n.min(g_n).min(b_n);
        if max_c > 0.0 {
            let s = (max_c - min_c) / max_c;
            mean_saturation += s;
        }

        let luma_norm = luma_f / 255.0;
        if x >= cx0 && x < cx1 && y >= cy0 && y < cy1 {
            center_sum += luma_norm;
            center_n += 1;
        } else {
            edge_sum += luma_norm;
            edge_n += 1;
        }
    }

    mean_saturation /= total_pixels as f32;

    let percentile = |hist: &Vec<u32>, p: f64| -> usize {
        let target = (total_pixels * p) as u32;
        let mut cumulative = 0u32;
        for (i, &v) in hist.iter().enumerate() {
            cumulative += v;
            if cumulative >= target {
                return i;
            }
        }
        255
    };

    let p1 = percentile(&luma_hist, 0.01);
    let p50 = percentile(&luma_hist, 0.50);
    let p99 = percentile(&luma_hist, 0.99);

    let black_point = p1;
    let white_point = p99;
    let range = (white_point as f64 - black_point as f64).max(1.0);

    let highlight_percent =
        luma_hist[HIGHLIGHT_LUMA_THRESHOLD..256].iter().sum::<u32>() as f64 / total_pixels;
    let clipped_percent =
        luma_hist[CLIPPED_LUMA_THRESHOLD..256].iter().sum::<u32>() as f64 / total_pixels;

    let mut exposure = (EXPOSURE_MIDPOINT - p50 as f64) * EXPOSURE_SCALE;

    if white_point > WHITE_POINT_HARD_LIMIT
        || highlight_percent > HIGHLIGHT_PERCENT_THRESHOLD
        || clipped_percent > CLIPPED_PERCENT_THRESHOLD
    {
        exposure = exposure.min(0.0);
    }

    if white_point as f64 + exposure > EXPOSURE_CEILING {
        exposure = EXPOSURE_CEILING - white_point as f64;
    }

    let mut contrast = 0.0f64;
    if range < TARGET_RANGE {
        contrast = ((TARGET_RANGE / range) - 1.0) * CONTRAST_SCALE;
    }
    if highlight_percent > HIGHLIGHT_PERCENT_THRESHOLD {
        contrast *= HIGHLIGHT_CONTRAST_REDUCE;
    }

    let shadow_percent = luma_hist[0..SHADOW_LUMA_MAX].iter().sum::<u32>() as f64 / total_pixels;

    let mut shadows = 0.0f64;
    if shadow_percent > SHADOW_PERCENT_THRESHOLD {
        shadows = (shadow_percent * SHADOW_BOOST_SCALE).min(SHADOW_MAX);
    }

    let mut highlights = 0.0f64;
    if highlight_percent > HIGHLIGHT_PERCENT_THRESHOLD {
        highlights = -(highlight_percent * HIGHLIGHT_BOOST_SCALE).min(HIGHLIGHT_MAX);
    }

    let mut vibrancy = 0.0f64;
    if mean_saturation < VIBRANCY_SAT_THRESHOLD {
        vibrancy = (VIBRANCY_SAT_THRESHOLD - mean_saturation) as f64 * VIBRANCY_SCALE;
    }

    let mut dehaze = 0.0f64;
    if range < DEHAZE_RANGE_THRESHOLD && mean_saturation < DEHAZE_SAT_THRESHOLD {
        dehaze = (1.0 - range / DEHAZE_RANGE_THRESHOLD) * DEHAZE_SCALE;
    }

    let mut clarity = 0.0f64;
    if range < CLARITY_RANGE_THRESHOLD {
        clarity = (1.0 - range / CLARITY_RANGE_THRESHOLD) * CLARITY_SCALE;
    }

    let mut vignette_amount = 0.0f64;
    let mut centre = 0.0f64;

    if center_n > 0 && edge_n > 0 {
        let c_avg = center_sum / center_n as f32;
        let e_avg = edge_sum / edge_n as f32;

        if e_avg < c_avg {
            let diff = c_avg - e_avg;
            vignette_amount = -(diff as f64 * VIGNETTE_SCALE);

            if diff > VIGNETTE_CENTRE_DIFF_THRESHOLD {
                centre = (diff as f64 * CENTRE_SCALE).min(CENTRE_MAX);
            }
        }
    }

    let mut adjusted_luma_hist = vec![0u32; 256];
    for pixel in rgb_image.pixels() {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;
        let mut luma = LUMA_R as f64 * r + LUMA_G as f64 * g + LUMA_B as f64 * b;
        luma += exposure;
        luma = (luma - MID_GRAY) * (1.0 + contrast / 100.0) + MID_GRAY;
        adjusted_luma_hist[luma.clamp(0.0, 255.0).round() as usize] += 1;
    }

    let adj_p1 = percentile(&adjusted_luma_hist, 0.01);
    let adj_p50 = percentile(&adjusted_luma_hist, 0.50);
    let adj_p99 = percentile(&adjusted_luma_hist, 0.99);
    let blacks: f64 = -(adj_p1 as f64 * BLACKS_SCALE);
    let whites: f64 = (adj_p99 as f64 - 255.0) * WHITES_SCALE;
    let brightness: f64 = (MID_GRAY - adj_p50 as f64) * BRIGHTNESS_SCALE;

    AutoAdjustmentResults {
        exposure: (exposure / EXPOSURE_OUTPUT_SCALE).clamp(-5.0, 5.0),
        brightness: brightness.clamp(-5.0, 5.0),
        contrast: contrast.clamp(-100.0, 100.0),
        highlights: highlights.clamp(-100.0, 100.0),
        shadows: shadows.clamp(-100.0, 100.0),
        vibrancy: vibrancy.clamp(-100.0, 100.0),
        vignette_amount: vignette_amount.clamp(-100.0, 100.0),
        temperature: auto_temperature,
        tint: auto_tint,
        dehaze: dehaze.clamp(-100.0, 100.0),
        clarity: clarity.clamp(-100.0, 100.0),
        centre: centre.clamp(-100.0, 100.0),
        whites: whites.clamp(-100.0, 100.0),
        blacks: blacks.clamp(-100.0, 100.0),
    }
}

pub fn auto_results_to_json(results: &AutoAdjustmentResults) -> serde_json::Value {
    json!({
        "exposure": results.exposure,
        "brightness": results.brightness,
        "contrast": results.contrast,
        "highlights": results.highlights,
        "shadows": results.shadows,
        "vibrance": results.vibrancy,
        "vignetteAmount": results.vignette_amount,
        "temperature": results.temperature,
        "tint": results.tint,
        "whiteBalanceMode": "auto",
        "clarity": results.clarity,
        "centré": results.centre,

        "dehaze": results.dehaze,
        "sectionVisibility": {
            "basic": true,
            "color": true,
            "effects": true
        },
        "whites": results.whites,
        "blacks": results.blacks
    })
}

pub(super) fn white_balance_multipliers(temperature: f64, tint: f64) -> (f64, f64, f64) {
    let normalized_temperature = temperature / 25.0;
    let normalized_tint = tint / 100.0;
    let temperature_rgb = (
        1.0 + normalized_temperature * 0.2,
        1.0 + normalized_temperature * 0.05,
        1.0 - normalized_temperature * 0.2,
    );
    let tint_rgb = (
        1.0 + normalized_tint * 0.25,
        1.0 - normalized_tint * 0.25,
        1.0 + normalized_tint * 0.25,
    );
    (
        temperature_rgb.0 * tint_rgb.0,
        temperature_rgb.1 * tint_rgb.1,
        temperature_rgb.2 * tint_rgb.2,
    )
}

pub(super) fn estimate_auto_white_balance(image: &RgbImage) -> (f64, f64) {
    const SATURATION_BINS: usize = 256;
    const MIN_LUMA: f64 = 0.08;
    const MAX_CHANNEL: f64 = 0.98;
    const MAX_NEUTRAL_SATURATION: f64 = 0.72;

    let mut saturation_histogram = [0usize; SATURATION_BINS];
    let mut eligible_pixels = 0usize;

    for pixel in image.pixels() {
        let r = pixel[0] as f64 / 255.0;
        let g = pixel[1] as f64 / 255.0;
        let b = pixel[2] as f64 / 255.0;
        let max_channel = r.max(g).max(b);
        let min_channel = r.min(g).min(b);
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if luma < MIN_LUMA || max_channel > MAX_CHANNEL || min_channel <= 0.0 {
            continue;
        }

        let saturation = (max_channel - min_channel) / max_channel.max(1.0e-6);
        let bin = (saturation * (SATURATION_BINS - 1) as f64).round() as usize;
        saturation_histogram[bin.min(SATURATION_BINS - 1)] += 1;
        eligible_pixels += 1;
    }

    if eligible_pixels < 16 {
        return (0.0, 0.0);
    }

    let neutral_target = (eligible_pixels as f64 * 0.35).ceil() as usize;
    let mut neutral_count = 0usize;
    let mut saturation_threshold = MAX_NEUTRAL_SATURATION;
    for (bin, count) in saturation_histogram.iter().enumerate() {
        neutral_count += count;
        if neutral_count >= neutral_target {
            saturation_threshold =
                (bin as f64 / (SATURATION_BINS - 1) as f64).clamp(0.08, MAX_NEUTRAL_SATURATION);
            break;
        }
    }

    let mut red_ratio_log = 0.0f64;
    let mut blue_ratio_log = 0.0f64;
    let mut total_weight = 0.0f64;

    for pixel in image.pixels() {
        let r = pixel[0] as f64 / 255.0;
        let g = pixel[1] as f64 / 255.0;
        let b = pixel[2] as f64 / 255.0;
        let max_channel = r.max(g).max(b);
        let min_channel = r.min(g).min(b);
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if luma < MIN_LUMA || max_channel > MAX_CHANNEL || min_channel <= 0.0 {
            continue;
        }

        let saturation = (max_channel - min_channel) / max_channel.max(1.0e-6);
        if saturation > saturation_threshold {
            continue;
        }

        let neutral_weight = (1.0 - saturation).powi(3);
        let exposure_weight = (luma / 0.6).clamp(0.2, 1.0);
        let weight = neutral_weight * exposure_weight;
        red_ratio_log += (g / r.max(1.0 / 255.0)).ln() * weight;
        blue_ratio_log += (g / b.max(1.0 / 255.0)).ln() * weight;
        total_weight += weight;
    }

    if total_weight <= 1.0e-6 {
        return (0.0, 0.0);
    }

    let target_red_log = (red_ratio_log / total_weight).clamp(-1.0, 1.0);
    let target_blue_log = (blue_ratio_log / total_weight).clamp(-1.0, 1.0);
    let mut best_temperature = 0.0f64;
    let mut best_tint = 0.0f64;
    let mut best_score = f64::INFINITY;

    for temperature in -85..=85 {
        for tint in -85..=85 {
            let (red_gain, green_gain, blue_gain) =
                white_balance_multipliers(temperature as f64, tint as f64);
            let red_log = (red_gain / green_gain).ln();
            let blue_log = (blue_gain / green_gain).ln();
            let score = (red_log - target_red_log).powi(2) + (blue_log - target_blue_log).powi(2);
            if score < best_score {
                best_score = score;
                best_temperature = temperature as f64;
                best_tint = tint as f64;
            }
        }
    }

    (best_temperature, best_tint)
}
