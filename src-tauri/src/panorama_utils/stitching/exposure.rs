//! Overlap exposure compensation for progressive and focus compositing.
//!
//! Gains are measured on the overlap between a new source and the canvas, kept
//! as a smoothed scalar/RGB field and applied with an optional damping strength.

use super::super::coordinates::Projection;
use super::super::coordinates::map_target_to_source;
use super::super::render_geometry::transformed_image_region;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point3;

use super::{
    focus_stack_pixel_is_canvas_like, focus_stack_pixel_is_tone_foreground, get_interpolated_pixel,
    luminance,
};

#[allow(dead_code)]
pub(super) fn apply_exposure_gain(pixel: Rgb<f32>, gain: f32) -> Rgb<f32> {
    Rgb([pixel[0] * gain, pixel[1] * gain, pixel[2] * gain])
}

pub(super) fn apply_exposure_channel_gain(pixel: Rgb<f32>, gains: [f32; 3]) -> Rgb<f32> {
    Rgb([
        pixel[0] * gains[0],
        pixel[1] * gains[1],
        pixel[2] * gains[2],
    ])
}

pub(super) fn apply_exposure_compensation(
    pixel: Rgb<f32>,
    exposure: &ExposureCompensation,
    x: u32,
    y: u32,
) -> Rgb<f32> {
    apply_exposure_compensation_with_strength(pixel, exposure, x, y, 1.0)
}

pub(super) fn apply_exposure_compensation_with_strength(
    pixel: Rgb<f32>,
    exposure: &ExposureCompensation,
    x: u32,
    y: u32,
    strength: f32,
) -> Rgb<f32> {
    let strength = strength.clamp(0.0, 1.0);
    let gains = exposure.channel_gain_at(x, y).map(|gain| {
        if gain.is_finite() && gain > 0.0 {
            (gain.ln() * strength).exp()
        } else {
            1.0
        }
    });
    apply_exposure_channel_gain(pixel, gains)
}

pub(super) struct ExposureOverlap<'a> {
    pub(super) panorama: &'a Rgb32FImage,
    pub(super) panorama_mask: &'a GrayImage,
    pub(super) candidate: &'a ImageInfo,
    pub(super) candidate_image: &'a Rgb32FImage,
    pub(super) candidate_inverse: &'a Matrix3<f64>,
    pub(super) projection: Projection,
    pub(super) offset_x: f64,
    pub(super) offset_y: f64,
}

#[derive(Clone)]
pub(super) struct ExposureCompensation {
    pub(super) cell_size: u32,
    pub(super) grid_width: usize,
    pub(super) grid_height: usize,
    pub(super) gains: Vec<f32>,
    pub(super) representative_gain: f32,
    pub(super) channel_gains: Vec<[f32; 3]>,
    pub(super) representative_channel_gain: [f32; 3],
    /// Flat source scans often contain a broad illumination field that cannot
    /// be represented by one gain for the whole tile. Keep the historical
    /// representative gain for ordinary panoramas, but allow the PSD/source
    /// scan compositor to use the smoothed spatial field.
    pub(super) use_spatial_field: bool,
}

impl ExposureCompensation {
    /// Progressive scan compositing estimates a candidate against the current
    /// panorama.  Applying the full ratio to the whole candidate is correct at
    /// the overlap, but it can turn a real illumination gradient into a chain
    /// of source-sized tone blocks in the non-overlap area.  Pull the estimate
    /// toward one before the seam search; the seam still sees the direction of
    /// the correction while the low-frequency field remains stable across a
    /// long scan.  Focus-bracket fusion keeps the undamped estimator.
    pub(super) fn damped(mut self, strength: f32) -> Self {
        let strength = strength.clamp(0.0, 1.0);
        let damp = |value: f32| {
            if value.is_finite() && value > 0.0 {
                value.ln().mul_add(strength, 0.0).exp()
            } else {
                1.0
            }
        };
        self.representative_gain = damp(self.representative_gain);
        for gain in &mut self.gains {
            *gain = damp(*gain);
        }
        for channel in &mut self.representative_channel_gain {
            *channel = damp(*channel);
        }
        for gains in &mut self.channel_gains {
            for gain in gains {
                *gain = damp(*gain);
            }
        }
        self
    }

    #[allow(dead_code)]
    pub(super) fn gain_at(&self, _x: u32, _y: u32) -> f32 {
        if self.gains.is_empty() || self.grid_width == 0 || self.grid_height == 0 {
            return 1.0;
        }
        // The per-cell field is useful for diagnosing vignetting, but writing
        // it into every projected tile makes the 256px analysis cells visible
        // as rectangular tone steps on a long scan.  The default compositor
        // uses one robust overlap constant; spatial compensation is opt-in
        // for captures where illumination is known to vary within a station.
        // The spatial field remains available for diagnostics, but production
        // uses the one robust overlap constant.  `allow_linear = false` is the
        // single photometric wiring decision for the default Tone_Harmonizer.
        self.representative_gain
    }

    /// Diagnostic-only spatial scalar field retained for comparison reports.
    #[allow(dead_code)]
    pub(super) fn spatial_gain_at(&self, x: u32, y: u32) -> f32 {
        let grid_x = x as f64 / self.cell_size as f64;
        let grid_y = y as f64 / self.cell_size as f64;
        let x0 = (grid_x.floor() as usize).min(self.grid_width - 1);
        let y0 = (grid_y.floor() as usize).min(self.grid_height - 1);
        let x1 = (x0 + 1).min(self.grid_width - 1);
        let y1 = (y0 + 1).min(self.grid_height - 1);
        let tx = (grid_x - x0 as f64) as f32;
        let ty = (grid_y - y0 as f64) as f32;
        let value = |gx: usize, gy: usize| self.gains[gy * self.grid_width + gx];
        let top = value(x0, y0) * (1.0 - tx) + value(x1, y0) * tx;
        let bottom = value(x0, y1) * (1.0 - tx) + value(x1, y1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    pub(super) fn channel_gain_at(&self, x: u32, y: u32) -> [f32; 3] {
        if self.channel_gains.is_empty() || self.grid_width == 0 || self.grid_height == 0 {
            return [1.0; 3];
        }
        if self.use_spatial_field {
            return self.spatial_channel_gain_at(x, y);
        }
        // RGB gains use the same conservative default as the scalar field:
        // one robust overlap constant per candidate.  A per-cell RGB field is
        // useful for a measured vignette, but its 256px cells can otherwise
        // become visible chromatic rectangles across a long scan.
        // Keep the per-cell RGB field for comparison diagnostics, while the
        // shipped path uses the robust representative value.
        self.representative_channel_gain
    }

    /// Diagnostic-only spatial RGB field retained for comparison reports.
    #[allow(dead_code)]
    pub(super) fn spatial_channel_gain_at(&self, x: u32, y: u32) -> [f32; 3] {
        let grid_x = x as f64 / self.cell_size as f64;
        let grid_y = y as f64 / self.cell_size as f64;
        let x0 = (grid_x.floor() as usize).min(self.grid_width - 1);
        let y0 = (grid_y.floor() as usize).min(self.grid_height - 1);
        let x1 = (x0 + 1).min(self.grid_width - 1);
        let y1 = (y0 + 1).min(self.grid_height - 1);
        let tx = (grid_x - x0 as f64) as f32;
        let ty = (grid_y - y0 as f64) as f32;
        let value = |gx: usize, gy: usize| self.channel_gains[gy * self.grid_width + gx];
        std::array::from_fn(|channel| {
            let top = value(x0, y0)[channel] * (1.0 - tx) + value(x1, y0)[channel] * tx;
            let bottom = value(x0, y1)[channel] * (1.0 - tx) + value(x1, y1)[channel] * tx;
            top * (1.0 - ty) + bottom * ty
        })
    }
}

pub(super) fn estimate_overlap_exposure_compensation(
    ctx: ExposureOverlap<'_>,
) -> ExposureCompensation {
    const CELL_SIZE: u32 = 256;
    let (out_width, out_height) = ctx.panorama.dimensions();
    let grid_width = out_width.div_ceil(CELL_SIZE) as usize + 1;
    let grid_height = out_height.div_ceil(CELL_SIZE) as usize + 1;
    let cell_count = grid_width * grid_height;
    let sample_step = out_width.max(out_height).div_ceil(720).max(8) as usize;
    let mut log_sums = vec![0.0f64; cell_count];
    let mut channel_log_sums = vec![[0.0f64; 3]; cell_count];
    let mut counts = vec![0u32; cell_count];
    let mut ratios = Vec::new();
    let mut channel_ratios = [Vec::new(), Vec::new(), Vec::new()];
    let candidate_homography = ctx.candidate_inverse.try_inverse();
    let candidate_region = candidate_homography.as_ref().and_then(|homography| {
        transformed_image_region(
            ctx.candidate,
            homography,
            ctx.projection,
            ctx.offset_x,
            ctx.offset_y,
            out_width,
            out_height,
        )
    });
    let (left, right, top, bottom) = candidate_region.unwrap_or((
        0,
        out_width.saturating_sub(1),
        0,
        out_height.saturating_sub(1),
    ));
    let sample_step_u32 = sample_step as u32;
    let sample_start = |value: u32| value.div_ceil(sample_step_u32) * sample_step_u32;
    for y in (sample_start(top)..=bottom).step_by(sample_step) {
        for x in (sample_start(left)..=right).step_by(sample_step) {
            if ctx.panorama_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let target = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
            let Some(source) =
                map_target_to_source(ctx.candidate_inverse, target, ctx.candidate, ctx.projection)
            else {
                continue;
            };
            if source.x < 0.0
                || source.y < 0.0
                || source.x >= ctx.candidate_image.width() as f64 - 1.0
                || source.y >= ctx.candidate_image.height() as f64 - 1.0
            {
                continue;
            }
            let base_luma = luminance(ctx.panorama.get_pixel(x, y));
            let base_pixel = ctx.panorama.get_pixel(x, y);
            let candidate_pixel = get_interpolated_pixel(ctx.candidate_image, source.x, source.y);
            // Exposure estimation is a paper-plane operation.  A luma-only
            // ratio admits red seals, dark ink, and pale figures into the
            // overlap statistics; one such region can then produce a gain
            // step that is visible across the whole virtual tile.  Require
            // both samples to have the generic canvas signature used by the
            // final tone pass, while retaining the broad luma/range checks
            // below for darker or lighter paper exposures.
            if !focus_stack_pixel_is_canvas_like(base_pixel.0.as_slice())
                || !focus_stack_pixel_is_canvas_like(candidate_pixel.0.as_slice())
                || focus_stack_pixel_is_tone_foreground(base_pixel.0.as_slice())
                || focus_stack_pixel_is_tone_foreground(candidate_pixel.0.as_slice())
            {
                continue;
            }
            let candidate_luma = luminance(&candidate_pixel);
            if (0.025..0.92).contains(&base_luma) && (0.025..0.92).contains(&candidate_luma) {
                let ratio = base_luma / candidate_luma;
                if (0.55..1.8).contains(&ratio) {
                    ratios.push(ratio);
                    for channel in 0..3 {
                        let base_value = base_pixel[channel];
                        let candidate_value = candidate_pixel[channel];
                        if base_value > 0.025 && candidate_value > 0.025 {
                            let channel_ratio = base_value / candidate_value;
                            if (0.55..1.8).contains(&channel_ratio) {
                                channel_ratios[channel].push(channel_ratio);
                            }
                        }
                    }
                    let grid_x = (x / CELL_SIZE) as usize;
                    let grid_y = (y / CELL_SIZE) as usize;
                    let index = grid_y * grid_width + grid_x;
                    log_sums[index] += (ratio as f64).ln();
                    for channel in 0..3 {
                        let base_value = base_pixel[channel];
                        let candidate_value = candidate_pixel[channel];
                        if base_value > 0.025 && candidate_value > 0.025 {
                            let channel_ratio = base_value / candidate_value;
                            if (0.55..1.8).contains(&channel_ratio) {
                                channel_log_sums[index][channel] += (channel_ratio as f64).ln();
                            }
                        }
                    }
                    counts[index] += 1;
                }
            }
        }
    }
    // Some real scans use a very dark, nearly neutral paper whose warm-canvas
    // gate is intentionally conservative.  If that strict set is too small,
    // take a second robust pass over neutral overlap samples.  The fallback
    // rejects saturated artwork and trims both tails, so a red seal or a dark
    // ink stroke cannot become the tile-wide exposure estimate.
    if ratios.len() < 32 {
        let mut neutral_ratios = Vec::new();
        for y in (sample_start(top)..=bottom).step_by(sample_step) {
            for x in (sample_start(left)..=right).step_by(sample_step) {
                if ctx.panorama_mask.get_pixel(x, y)[0] == 0 {
                    continue;
                }
                let target = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
                let Some(source) = map_target_to_source(
                    ctx.candidate_inverse,
                    target,
                    ctx.candidate,
                    ctx.projection,
                ) else {
                    continue;
                };
                if source.x < 0.0
                    || source.y < 0.0
                    || source.x >= ctx.candidate_image.width() as f64 - 1.0
                    || source.y >= ctx.candidate_image.height() as f64 - 1.0
                {
                    continue;
                }
                let base_pixel = ctx.panorama.get_pixel(x, y);
                let candidate_pixel =
                    get_interpolated_pixel(ctx.candidate_image, source.x, source.y);
                let base_luma = luminance(base_pixel);
                let candidate_luma = luminance(&candidate_pixel);
                let base_chroma = base_pixel[0].max(base_pixel[1]).max(base_pixel[2])
                    - base_pixel[0].min(base_pixel[1]).min(base_pixel[2]);
                let candidate_chroma = candidate_pixel[0]
                    .max(candidate_pixel[1])
                    .max(candidate_pixel[2])
                    - candidate_pixel[0]
                        .min(candidate_pixel[1])
                        .min(candidate_pixel[2]);
                if !(0.025..0.92).contains(&base_luma)
                    || !(0.025..0.92).contains(&candidate_luma)
                    || base_chroma > 0.30
                    || candidate_chroma > 0.30
                {
                    continue;
                }
                let ratio = base_luma / candidate_luma;
                if (0.55..1.8).contains(&ratio) {
                    neutral_ratios.push(ratio);
                }
            }
        }
        neutral_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if neutral_ratios.len() >= 32 {
            let trim = neutral_ratios.len() / 5;
            ratios.extend(
                neutral_ratios[trim..neutral_ratios.len().saturating_sub(trim)]
                    .iter()
                    .copied(),
            );
        }
    }
    let representative_gain = if ratios.len() < 32 {
        1.0
    } else {
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ratios[ratios.len() / 2].clamp(0.75, 1.35)
    };
    let mut gains = vec![representative_gain; cell_count];
    let representative_channel_gain = std::array::from_fn(|channel| {
        if channel_ratios[channel].len() < 32 {
            representative_gain
        } else {
            channel_ratios[channel].sort_by(f32::total_cmp);
            channel_ratios[channel][channel_ratios[channel].len() / 2].clamp(0.75, 1.35)
        }
    });
    let mut channel_gains = vec![representative_channel_gain; cell_count];
    for index in 0..cell_count {
        if counts[index] >= 3 {
            gains[index] = (log_sums[index] / counts[index] as f64).exp().clamp(
                (representative_gain * 0.78) as f64,
                (representative_gain * 1.28) as f64,
            ) as f32;
            for channel in 0..3 {
                let channel_gain = (channel_log_sums[index][channel] / counts[index] as f64)
                    .exp()
                    .clamp(
                        (representative_channel_gain[channel] * 0.78) as f64,
                        (representative_channel_gain[channel] * 1.28) as f64,
                    );
                channel_gains[index][channel] = channel_gain as f32;
            }
        }
    }

    // The field models only broad illumination. Repeated neighbor averaging removes
    // local texture and registration noise while retaining vignetting and shadows.
    for _ in 0..5 {
        let mut smoothed = gains.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let mut weighted_sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        let weight = if dx == 0 && dy == 0 { 4.0 } else { 1.0 };
                        weighted_sum += gains[neighbor_index] * weight;
                        weight_sum += weight;
                    }
                }
                smoothed[grid_y * grid_width + grid_x] = weighted_sum / weight_sum;
            }
        }
        gains = smoothed;
    }

    ExposureCompensation {
        cell_size: CELL_SIZE,
        grid_width,
        grid_height,
        gains,
        representative_gain,
        channel_gains,
        representative_channel_gain,
        use_spatial_field: false,
    }
}
