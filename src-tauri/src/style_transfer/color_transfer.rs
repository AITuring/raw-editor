//! Colour statistics and the transfer transform.
//!
//! Measures HSV and per-channel statistics on bounded previews and turns a
//! reference/target pair into the Mood (HSV) or Distribution (per-channel)
//! transform that is applied to whole images or streamed RGB rows.

use image::RgbImage;

use super::StyleTransferMode;

// Full variance matching exaggerates paper grain, noise, and brush edges until
// a colour transfer looks artificially sharpened. Transfer only the broad
// contrast character while retaining most of the target's native detail.
const DETAIL_CONTRAST_TRANSFER: f32 = 0.28;

#[derive(Debug, Clone, Copy)]
pub(super) struct ImageStats {
    pub(super) mean: [f32; 3],
    pub(super) std: [f32; 3],
    pub(super) mean_saturation: f32,
    pub(super) mean_value: f32,
    pub(super) value_std: f32,
    pub(super) hue_sin: f32,
    pub(super) hue_cos: f32,
    pub(super) hue_weight: f32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TransferTransform {
    pub(super) hue_shift: f32,
    pub(super) saturation_scale: f32,
    pub(super) value_offset: f32,
    pub(super) value_contrast: f32,
    pub(super) target_value_mean: f32,
    pub(super) channel_scale: [f32; 3],
    pub(super) channel_offset: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AppliedTransform {
    pub(super) mode: StyleTransferMode,
    pub(super) hue_shift: f32,
    pub(super) saturation_scale: f32,
    pub(super) value_offset: f32,
    pub(super) value_contrast: f32,
    pub(super) target_value_mean: f32,
    pub(super) channel_scale: [f32; 3],
    pub(super) channel_offset: [f32; 3],
}

pub(super) fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.clamp(min, max)
}

fn normalize_hue(value: f32) -> f32 {
    let mut normalized = value;
    while normalized > 0.5 {
        normalized -= 1.0;
    }
    while normalized < -0.5 {
        normalized += 1.0;
    }
    normalized
}

pub(super) fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let mut hue = 0.0;

    if delta > 1.0e-6 {
        if (max - r).abs() < 1.0e-6 {
            hue = ((g - b) / delta) % 6.0;
        } else if (max - g).abs() < 1.0e-6 {
            hue = (b - r) / delta + 2.0;
        } else {
            hue = (r - g) / delta + 4.0;
        }
        hue /= 6.0;
        if hue < 0.0 {
            hue += 1.0;
        }
    }

    (hue, if max <= 1.0e-6 { 0.0 } else { delta / max }, max)
}

pub(super) fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    let normalized_hue = hue.rem_euclid(1.0);
    let scaled = normalized_hue * 6.0;
    let sector = scaled.floor() as i32;
    let fraction = scaled - sector as f32;
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * fraction);
    let t = value * (1.0 - saturation * (1.0 - fraction));

    match sector.rem_euclid(6) {
        0 => [value, t, p],
        1 => [q, value, p],
        2 => [p, value, t],
        3 => [p, q, value],
        4 => [t, p, value],
        _ => [value, p, q],
    }
}

fn collect_stats(image: &RgbImage) -> ImageStats {
    let total = (image.width() as usize)
        .saturating_mul(image.height() as usize)
        .max(1);
    let stride = ((total as f32 / 120_000.0).sqrt().ceil() as u32).max(1);
    let mut sum = [0.0f32; 3];
    let mut sum_squares = [0.0f32; 3];
    let mut saturation_sum = 0.0;
    let mut value_sum = 0.0;
    let mut value_squares = 0.0;
    let mut hue_sin = 0.0;
    let mut hue_cos = 0.0;
    let mut hue_weight = 0.0;
    let mut count: f32 = 0.0;

    for y in (0..image.height()).step_by(stride as usize) {
        for x in (0..image.width()).step_by(stride as usize) {
            let pixel = image.get_pixel(x, y).0;
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let (hue, saturation, value) = rgb_to_hsv(r, g, b);
            let weight = (saturation * 0.65 + value * 0.35).max(0.001);

            sum[0] += r;
            sum[1] += g;
            sum[2] += b;
            sum_squares[0] += r * r;
            sum_squares[1] += g * g;
            sum_squares[2] += b * b;
            saturation_sum += saturation;
            value_sum += value;
            value_squares += value * value;
            hue_sin += (hue * std::f32::consts::TAU).sin() * weight;
            hue_cos += (hue * std::f32::consts::TAU).cos() * weight;
            hue_weight += weight;
            count += 1.0;
        }
    }

    let safe_count = count.max(1.0);
    let mean = [
        sum[0] / safe_count,
        sum[1] / safe_count,
        sum[2] / safe_count,
    ];
    let std = [
        (sum_squares[0] / safe_count - mean[0] * mean[0])
            .max(0.0)
            .sqrt(),
        (sum_squares[1] / safe_count - mean[1] * mean[1])
            .max(0.0)
            .sqrt(),
        (sum_squares[2] / safe_count - mean[2] * mean[2])
            .max(0.0)
            .sqrt(),
    ];
    let mean_value = value_sum / safe_count;

    ImageStats {
        mean,
        std,
        mean_saturation: saturation_sum / safe_count,
        mean_value,
        value_std: (value_squares / safe_count - mean_value * mean_value)
            .max(0.0)
            .sqrt(),
        hue_sin,
        hue_cos,
        hue_weight,
    }
}

pub(super) fn estimate_transform(reference: &RgbImage, target: &RgbImage) -> TransferTransform {
    let reference_stats = collect_stats(reference);
    let target_stats = collect_stats(target);

    let reference_hue = if reference_stats.hue_weight > 0.0 {
        reference_stats.hue_sin.atan2(reference_stats.hue_cos) / std::f32::consts::TAU
    } else {
        0.0
    };
    let target_hue = if target_stats.hue_weight > 0.0 {
        target_stats.hue_sin.atan2(target_stats.hue_cos) / std::f32::consts::TAU
    } else {
        0.0
    };

    let saturation_scale = clamp(
        reference_stats.mean_saturation / target_stats.mean_saturation.max(0.04),
        0.35,
        2.4,
    );
    let raw_value_contrast = clamp(
        reference_stats.value_std / target_stats.value_std.max(0.04),
        0.6,
        1.7,
    );
    let value_contrast = 1.0 + (raw_value_contrast - 1.0) * DETAIL_CONTRAST_TRANSFER;
    let raw_channel_scale = [
        clamp(
            reference_stats.std[0] / target_stats.std[0].max(0.025),
            0.55,
            1.8,
        ),
        clamp(
            reference_stats.std[1] / target_stats.std[1].max(0.025),
            0.55,
            1.8,
        ),
        clamp(
            reference_stats.std[2] / target_stats.std[2].max(0.025),
            0.55,
            1.8,
        ),
    ];
    let channel_scale =
        raw_channel_scale.map(|scale| 1.0 + (scale - 1.0) * DETAIL_CONTRAST_TRANSFER);
    let channel_offset = [
        reference_stats.mean[0] - target_stats.mean[0] * channel_scale[0],
        reference_stats.mean[1] - target_stats.mean[1] * channel_scale[1],
        reference_stats.mean[2] - target_stats.mean[2] * channel_scale[2],
    ];

    TransferTransform {
        hue_shift: normalize_hue(reference_hue - target_hue),
        saturation_scale,
        value_offset: reference_stats.mean_value - target_stats.mean_value,
        value_contrast,
        target_value_mean: target_stats.mean_value,
        channel_scale,
        channel_offset,
    }
}

pub(super) fn apply_transform(
    mut target: RgbImage,
    transform: TransferTransform,
    mode: StyleTransferMode,
    strength: f32,
) -> RgbImage {
    let applied = AppliedTransform::new(transform, mode, strength);

    for pixel in target.pixels_mut() {
        let original = [
            pixel.0[0] as f32 / 255.0,
            pixel.0[1] as f32 / 255.0,
            pixel.0[2] as f32 / 255.0,
        ];
        let next = applied.apply_pixel(original);

        pixel.0[0] = (clamp(next[0], 0.0, 1.0) * 255.0).round() as u8;
        pixel.0[1] = (clamp(next[1], 0.0, 1.0) * 255.0).round() as u8;
        pixel.0[2] = (clamp(next[2], 0.0, 1.0) * 255.0).round() as u8;
    }

    target
}

impl AppliedTransform {
    pub(super) fn new(
        transform: TransferTransform,
        mode: StyleTransferMode,
        strength: f32,
    ) -> Self {
        let amount = clamp(strength, 0.0, 1.0);
        Self {
            mode,
            hue_shift: transform.hue_shift * amount,
            saturation_scale: 1.0 + (transform.saturation_scale - 1.0) * amount,
            value_offset: transform.value_offset * amount,
            value_contrast: 1.0 + (transform.value_contrast - 1.0) * amount,
            target_value_mean: transform.target_value_mean,
            channel_scale: [
                1.0 + (transform.channel_scale[0] - 1.0) * amount,
                1.0 + (transform.channel_scale[1] - 1.0) * amount,
                1.0 + (transform.channel_scale[2] - 1.0) * amount,
            ],
            channel_offset: [
                transform.channel_offset[0] * amount,
                transform.channel_offset[1] * amount,
                transform.channel_offset[2] * amount,
            ],
        }
    }

    pub(super) fn apply_pixel(self, original: [f32; 3]) -> [f32; 3] {
        match self.mode {
            StyleTransferMode::Distribution => [
                original[0] * self.channel_scale[0] + self.channel_offset[0],
                original[1] * self.channel_scale[1] + self.channel_offset[1],
                original[2] * self.channel_scale[2] + self.channel_offset[2],
            ],
            StyleTransferMode::Mood => {
                let (hue, saturation, value) = rgb_to_hsv(original[0], original[1], original[2]);
                let adjusted_value = clamp(
                    self.target_value_mean
                        + (value - self.target_value_mean) * self.value_contrast
                        + self.value_offset,
                    0.0,
                    1.0,
                );
                hsv_to_rgb(
                    hue + self.hue_shift,
                    clamp(saturation * self.saturation_scale, 0.0, 1.0),
                    adjusted_value,
                )
            }
        }
    }
}

#[cfg(not(target_os = "android"))]
pub(super) fn apply_transform_rgb_row(
    row: &mut [u8],
    transform: TransferTransform,
    mode: StyleTransferMode,
    strength: f32,
) {
    let applied = AppliedTransform::new(transform, mode, strength);
    for pixel in row.chunks_exact_mut(3) {
        let original = [
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        ];
        let next = applied.apply_pixel(original);
        pixel[0] = (clamp(next[0], 0.0, 1.0) * 255.0).round() as u8;
        pixel[1] = (clamp(next[1], 0.0, 1.0) * 255.0).round() as u8;
        pixel[2] = (clamp(next[2], 0.0, 1.0) * 255.0).round() as u8;
    }
}
