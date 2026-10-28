//! Low-frequency illumination correction of a materialised mosaic.
//!
//! The broad illumination smoothing shared with the tile compositor and the
//! source-owner correction that removes frame-sized exposure blocks; both add
//! only a smooth field, so source-owned detail is untouched.

use super::super::stitching::get_high_quality_interpolated_pixel;
use image::Luma;
use image::Rgb;
use image::Rgb32FImage;
use rayon::prelude::*;

use super::{
    STREAMING_CONSENSUS_MAX_DELTA, STREAMING_GROUP_GAIN_FEATHER, STREAMING_HARMONIZATION_LONG_SIDE,
    STREAMING_ILLUMINATION_CONTINUOUS_SIGMA, STREAMING_ILLUMINATION_LOCAL_SIGMA,
    STREAMING_ILLUMINATION_MAX_GAIN, STREAMING_ILLUMINATION_MIN_GAIN, STREAMING_OWNER_GLOBAL_SIGMA,
    STREAMING_OWNER_LOCAL_SIGMA, StreamingMosaicStore, luma, median,
};

pub(in crate::panorama_utils) fn smooth_streaming_low_frequency_illumination(
    output: &mut Rgb32FImage,
) {
    let scale = (output.width().max(output.height()) as f64
        / STREAMING_HARMONIZATION_LONG_SIDE as f64)
        .max(1.0);
    let width = (output.width() as f64 / scale).ceil().max(1.0) as u32;
    let height = (output.height() as f64 / scale).ceil().max(1.0) as u32;
    let analysis =
        image::imageops::resize(output, width, height, image::imageops::FilterType::Triangle);
    let valid = image::ImageBuffer::from_fn(width, height, |x, y| {
        let value = (luma(*analysis.get_pixel(x, y)) > 0.025) as u8 as f32;
        Luma([value])
    });
    let premultiplied = Rgb32FImage::from_fn(width, height, |x, y| {
        let weight = valid.get_pixel(x, y)[0];
        let pixel = analysis.get_pixel(x, y);
        Rgb([pixel[0] * weight, pixel[1] * weight, pixel[2] * weight])
    });
    let normalized_blur = |sigma: f32| {
        let colour = image::imageops::blur(&premultiplied, sigma);
        let weight = image::imageops::blur(&valid, sigma);
        Rgb32FImage::from_fn(width, height, |x, y| {
            let divisor = weight.get_pixel(x, y)[0];
            if divisor <= 1e-4 {
                return Rgb([0.0; 3]);
            }
            let pixel = colour.get_pixel(x, y);
            Rgb([pixel[0] / divisor, pixel[1] / divisor, pixel[2] / divisor])
        })
    };
    let local_low = normalized_blur(STREAMING_ILLUMINATION_LOCAL_SIGMA);
    let continuous_low = normalized_blur(STREAMING_ILLUMINATION_CONTINUOUS_SIGMA);
    let gains = Rgb32FImage::from_fn(width, height, |x, y| {
        if valid.get_pixel(x, y)[0] == 0.0 {
            return Rgb([1.0; 3]);
        }
        let local_pixel = *local_low.get_pixel(x, y);
        let continuous_pixel = *continuous_low.get_pixel(x, y);
        let local = luma(local_pixel);
        let continuous = luma(continuous_pixel);
        let luminance_gain = if local > 0.025 && continuous > 0.025 {
            (continuous / local).clamp(
                STREAMING_ILLUMINATION_MIN_GAIN,
                STREAMING_ILLUMINATION_MAX_GAIN,
            ) as f32
        } else {
            1.0f32
        };
        // Exposure and white-balance changes affect channels together, but a
        // scalar luma gain leaves a coloured seam when two RAW frames have
        // slightly different white balance.  Use the broad luma correction as
        // the anchor and add a restrained per-channel low-frequency term. The
        // high-frequency detail is still taken verbatim from the selected
        // owner, so this cannot blur brush edges.
        Rgb(std::array::from_fn(|channel| {
            let channel_gain = if local_pixel[channel] > 0.025 && continuous_pixel[channel] > 0.025
            {
                (continuous_pixel[channel] / local_pixel[channel]).clamp(
                    STREAMING_ILLUMINATION_MIN_GAIN as f32,
                    STREAMING_ILLUMINATION_MAX_GAIN as f32,
                )
            } else {
                luminance_gain
            };
            (luminance_gain * 0.72 + channel_gain * 0.28).clamp(
                STREAMING_ILLUMINATION_MIN_GAIN as f32,
                STREAMING_ILLUMINATION_MAX_GAIN as f32,
            )
        }))
    });
    let output_width = output.width() as usize;
    output
        .as_mut()
        .par_chunks_mut(output_width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let gain_y = (y as f64 / scale).clamp(0.0, gains.height().saturating_sub(1) as f64);
            for x in 0..output_width {
                let pixel = &mut row[x * 3..x * 3 + 3];
                if pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722 <= 0.025 {
                    continue;
                }
                let gain_x = (x as f64 / scale).clamp(0.0, gains.width().saturating_sub(1) as f64);
                let gain = get_high_quality_interpolated_pixel(&gains, gain_x, gain_y);
                for channel in 0..3 {
                    pixel[channel] = (pixel[channel] * gain[channel]).clamp(0.0, 1.0);
                }
            }
        });
}

/// Correct source-sized exposure fields after ownership has selected the
/// pixels that carry the painting detail.  A single canvas blur cannot tell a
/// frame-sized exposure step from a real broad paper gradient: at the centre
/// of a source owner it simply follows that owner's rectangle.  This pass
/// keeps the owner labels, estimates a low-frequency field for each owner,
/// and compares it with a broad canvas field.  Only that smooth difference is
/// added to the final pixels, so brush strokes and wood grain remain the
/// source-owned samples.
pub(super) fn apply_streaming_source_owner_low_frequency_correction(
    store: &StreamingMosaicStore,
    crop: (u32, u32, u32, u32),
    output: &mut Rgb32FImage,
) -> Result<usize, String> {
    let (left, top, width, height) = crop;
    if width == 0 || height == 0 || output.width() == 0 || output.height() == 0 {
        return Ok(0);
    }
    let scale = (width.max(height) as f64 / STREAMING_HARMONIZATION_LONG_SIDE as f64).max(1.0);
    let analysis_width = (width as f64 / scale).ceil().max(1.0) as u32;
    let analysis_height = (height as f64 / scale).ceil().max(1.0) as u32;
    let (_, coverage, owner) =
        store.analysis_region(left, top, scale, analysis_width, analysis_height)?;
    let output_analysis = image::imageops::resize(
        output,
        analysis_width,
        analysis_height,
        image::imageops::FilterType::Triangle,
    );

    // Ignore capture borders and clipped hardware highlights.  Those samples
    // would otherwise become the reference field and spread a white/black
    // edge through the neighbouring source owners.
    let valid = image::ImageBuffer::<Luma<f32>, Vec<f32>>::from_fn(
        analysis_width,
        analysis_height,
        |x, y| {
            let covered = coverage.get_pixel(x, y)[0] != 0;
            let source_owner = owner.get_pixel(x, y)[0] != 0;
            let pixel = *output_analysis.get_pixel(x, y);
            let value = luma(pixel);
            Luma([f32::from(u8::from(
                covered && source_owner && value > 0.06 && value < 0.93,
            ))])
        },
    );
    let premultiplied = Rgb32FImage::from_fn(analysis_width, analysis_height, |x, y| {
        let weight = valid.get_pixel(x, y)[0];
        let pixel = output_analysis.get_pixel(x, y);
        Rgb([pixel[0] * weight, pixel[1] * weight, pixel[2] * weight])
    });
    let global_colour = image::imageops::blur(&premultiplied, STREAMING_OWNER_GLOBAL_SIGMA);
    let global_weight = image::imageops::blur(&valid, STREAMING_OWNER_GLOBAL_SIGMA);
    let global_low = Rgb32FImage::from_fn(analysis_width, analysis_height, |x, y| {
        let weight = global_weight.get_pixel(x, y)[0];
        if weight <= 1e-4 {
            return Rgb([0.0; 3]);
        }
        let pixel = global_colour.get_pixel(x, y);
        Rgb([pixel[0] / weight, pixel[1] / weight, pixel[2] / weight])
    });

    let mut owner_bounds = vec![None::<(u32, u32, u32, u32)>; 256];
    for y in 0..analysis_height {
        for x in 0..analysis_width {
            if valid.get_pixel(x, y)[0] <= 0.5 {
                continue;
            }
            let id = owner.get_pixel(x, y)[0] as usize;
            if id == 0 {
                continue;
            }
            owner_bounds[id] = Some(match owner_bounds[id] {
                Some((min_x, min_y, max_x, max_y)) => {
                    (min_x.min(x), min_y.min(y), max_x.max(x), max_y.max(y))
                }
                None => (x, y, x, y),
            });
        }
    }
    let owner_count = owner_bounds
        .iter()
        .skip(1)
        .filter(|bounds| bounds.is_some())
        .count();
    if owner_count < 2 {
        return Ok(0);
    }

    let mut delta_weighted = Rgb32FImage::new(analysis_width, analysis_height);
    let mut delta_support = image::ImageBuffer::<Luma<f32>, Vec<f32>>::from_pixel(
        analysis_width,
        analysis_height,
        Luma([0.0]),
    );
    // The owner fallback is multiplicative: a source whose paper median is
    // 0.18 must be lifted toward a 0.30 target by roughly 1.67x, rather than
    // by adding a fixed 0.12.  Store the correction in log space so the
    // feathered field remains a gain and never washes out the source detail.
    let mut owner_log_gain_weighted = Rgb32FImage::new(analysis_width, analysis_height);
    let mut owner_log_gain_support = image::ImageBuffer::<Luma<f32>, Vec<f32>>::from_pixel(
        analysis_width,
        analysis_height,
        Luma([0.0]),
    );
    let owner_radius = (3.0 * STREAMING_OWNER_LOCAL_SIGMA).ceil() as u32;
    let mut owner_level_samples = (0..256).map(|_| Vec::<[f64; 3]>::new()).collect::<Vec<_>>();
    let mut corrected_cells = 0usize;
    for (id, bounds) in owner_bounds.iter().enumerate().skip(1) {
        let Some((min_x, min_y, max_x, max_y)) = *bounds else {
            continue;
        };
        let local_left = min_x.saturating_sub(owner_radius);
        let local_top = min_y.saturating_sub(owner_radius);
        let local_right = (max_x + owner_radius).min(analysis_width.saturating_sub(1));
        let local_bottom = (max_y + owner_radius).min(analysis_height.saturating_sub(1));
        let local_width = local_right - local_left + 1;
        let local_height = local_bottom - local_top + 1;
        let local_colour = Rgb32FImage::from_fn(local_width, local_height, |x, y| {
            let gx = local_left + x;
            let gy = local_top + y;
            if owner.get_pixel(gx, gy)[0] as usize != id || valid.get_pixel(gx, gy)[0] <= 0.5 {
                return Rgb([0.0; 3]);
            }
            *output_analysis.get_pixel(gx, gy)
        });
        let local_weight = image::ImageBuffer::<Luma<f32>, Vec<f32>>::from_fn(
            local_width,
            local_height,
            |x, y| {
                let gx = local_left + x;
                let gy = local_top + y;
                Luma([f32::from(u8::from(
                    owner.get_pixel(gx, gy)[0] as usize == id && valid.get_pixel(gx, gy)[0] > 0.5,
                ))])
            },
        );
        let local_colour = image::imageops::blur(&local_colour, STREAMING_OWNER_LOCAL_SIGMA);
        let local_weight = image::imageops::blur(&local_weight, STREAMING_OWNER_LOCAL_SIGMA);
        for gy in min_y..=max_y {
            for gx in min_x..=max_x {
                if owner.get_pixel(gx, gy)[0] as usize != id || valid.get_pixel(gx, gy)[0] <= 0.5 {
                    continue;
                }
                let lx = gx - local_left;
                let ly = gy - local_top;
                let support = local_weight.get_pixel(lx, ly)[0];
                let global_support = global_weight.get_pixel(gx, gy)[0];
                if support <= 0.03 || global_support <= 0.03 {
                    continue;
                }
                let local_pixel = local_colour.get_pixel(lx, ly);
                let local = Rgb([
                    local_pixel[0] / support,
                    local_pixel[1] / support,
                    local_pixel[2] / support,
                ]);
                // Keep a sparse owner-level paper statistic.  This is used
                // below when a source tile is wider than the canvas blur and
                // the all-owner reference has become self-contaminated at its
                // centre.  The local low-pass removes brush detail before the
                // sample is admitted.
                if (gx + gy) % 4 == 0 && luma(local) > 0.10 {
                    owner_level_samples[id].push([
                        f64::from(local[0]),
                        f64::from(local[1]),
                        f64::from(local[2]),
                    ]);
                }
                let global = *global_low.get_pixel(gx, gy);
                let correction = Rgb(std::array::from_fn(|channel| {
                    (f64::from(global[channel]) - f64::from(local[channel])).clamp(
                        -STREAMING_CONSENSUS_MAX_DELTA,
                        STREAMING_CONSENSUS_MAX_DELTA,
                    ) as f32
                }));
                let weight = support.min(1.0);
                delta_weighted.put_pixel(
                    gx,
                    gy,
                    Rgb(std::array::from_fn(|channel| correction[channel] * weight)),
                );
                delta_support.put_pixel(gx, gy, Luma([weight]));
                corrected_cells += 1;
            }
        }
    }
    // Estimate one robust low-frequency target per owner.  Equal weighting
    // of the owner medians avoids letting a large source tile dominate the
    // target, while keeping the correction tied to the actual selected
    // source pixels instead of synthetic fill.
    let mut owner_medians = [[0.0f64; 3]; 256];
    let mut owner_median_ids = Vec::<u8>::new();
    for id in 1..256 {
        if owner_level_samples[id].len() < 8 {
            continue;
        }
        owner_medians[id] = std::array::from_fn(|channel| {
            median(
                &mut owner_level_samples[id]
                    .iter()
                    .map(|value| value[channel])
                    .collect::<Vec<_>>(),
            )
        });
        owner_median_ids.push(id as u8);
    }
    if owner_median_ids.len() >= 2 {
        let target: [f64; 3] = std::array::from_fn(|channel| {
            let mut values = owner_median_ids
                .iter()
                .map(|&id| owner_medians[id as usize][channel])
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            if values.len() % 2 == 0 {
                (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
            } else {
                values[values.len() / 2]
            }
        });
        let mut fallback_interior_cells = 0usize;
        // The owner-level fallback supplies the interior of a source block;
        // the boundary field remains in charge near a real paper gradient.
        // Blend in log-like channel space with a bounded additive delta so
        // fine detail and dark ink are never averaged.
        for &id in &owner_median_ids {
            let Some((min_x, min_y, max_x, max_y)) = owner_bounds[id as usize] else {
                continue;
            };
            for gy in min_y..=max_y {
                for gx in min_x..=max_x {
                    if owner.get_pixel(gx, gy)[0] != id || valid.get_pixel(gx, gy)[0] <= 0.5 {
                        continue;
                    }
                    let weight = delta_support.get_pixel(gx, gy)[0].max(0.001);
                    let existing = *delta_weighted.get_pixel(gx, gy);
                    let existing: [f32; 3] =
                        std::array::from_fn(|channel| existing[channel] / weight);
                    // At an owner centre the broad canvas reference is often
                    // self-contaminated, so `existing` is near zero.  Use the
                    // robust owner-level fallback fully there; retain the
                    // overlap-derived field near boundaries where it carries
                    // a real spatial illumination gradient.
                    let existing_magnitude = existing
                        .iter()
                        .map(|value| f64::from(value.abs()))
                        .sum::<f64>()
                        / 3.0;
                    // The multiplicative owner gain below is the robust
                    // interior correction.  Do not add the same exposure step
                    // a second time in linear space: doing both turns a
                    // 0.40/0.60 pair into an inverted 0.62/0.42 seam.  Keep
                    // the fallback counter for diagnostics, while retaining
                    // the overlap-derived additive field only at boundaries.
                    let gain_fallback_weight = if existing_magnitude < 0.02 {
                        fallback_interior_cells += 1;
                        0.95
                    } else {
                        0.65
                    };
                    // The multiplicative gain is the robust interior
                    // correction.  The overlap field remains available for a
                    // run without owner medians, but is not mixed into this
                    // owner-median path.
                    let log_gain: [f32; 3] = std::array::from_fn(|channel| {
                        let source = owner_medians[id as usize][channel].max(0.04);
                        let target = target[channel].max(0.04);
                        (target / source).ln().clamp(-0.70, 0.70) as f32
                    });
                    let gain_weight = weight * gain_fallback_weight;
                    owner_log_gain_weighted.put_pixel(
                        gx,
                        gy,
                        Rgb(std::array::from_fn(|channel| {
                            log_gain[channel] * gain_weight
                        })),
                    );
                    owner_log_gain_support.put_pixel(gx, gy, Luma([gain_weight]));
                    delta_weighted.put_pixel(
                        gx,
                        gy,
                        Rgb(std::array::from_fn(|channel| existing[channel] * weight)),
                    );
                    delta_support.put_pixel(gx, gy, Luma([weight]));
                }
            }
        }
        println!(
            "  - Owner median fallback: {} owners, {:.3?} target, {} interior cells",
            owner_median_ids.len(),
            target,
            fallback_interior_cells
        );
    }
    // Once every owner has a robust low-frequency median, the multiplicative
    // owner gain is the authoritative correction.  The broad additive field
    // is retained only as a fallback for runs without enough owner evidence;
    // applying both fields would double-count the same exposure step near a
    // boundary.
    let additive_delta_enabled = owner_median_ids.len() < 2;
    if corrected_cells == 0 {
        return Ok(0);
    }

    // Feather only the correction field at owner boundaries.  The support
    // normalisation keeps the feather from darkening where an owner has no
    // valid paper samples.
    let feather_sigma = STREAMING_GROUP_GAIN_FEATHER.max(1.0);
    let weighted_blur = image::imageops::blur(&delta_weighted, feather_sigma);
    let support_blur = image::imageops::blur(&delta_support, feather_sigma);
    let owner_log_gain_blur = image::imageops::blur(&owner_log_gain_weighted, feather_sigma);
    let owner_log_gain_support_blur = image::imageops::blur(&owner_log_gain_support, feather_sigma);
    let output_width = output.width() as usize;
    let output_scale_x =
        analysis_width.saturating_sub(1) as f64 / output.width().saturating_sub(1).max(1) as f64;
    let output_scale_y =
        analysis_height.saturating_sub(1) as f64 / output.height().saturating_sub(1).max(1) as f64;
    output
        .as_mut()
        .par_chunks_mut(output_width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let ay =
                (y as f64 * output_scale_y).clamp(0.0, analysis_height.saturating_sub(1) as f64);
            for x in 0..output_width {
                let ax =
                    (x as f64 * output_scale_x).clamp(0.0, analysis_width.saturating_sub(1) as f64);
                let support = sample_luma_f32(&support_blur, ax, ay);
                if support <= 0.03 {
                    continue;
                }
                let correction = get_high_quality_interpolated_pixel(&weighted_blur, ax, ay);
                let pixel = &mut row[x * 3..x * 3 + 3];
                let current_luma = pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114;
                if current_luma <= 0.06 || current_luma >= 0.93 {
                    continue;
                }
                if additive_delta_enabled {
                    for channel in 0..3 {
                        pixel[channel] =
                            (pixel[channel] + correction[channel] / support).clamp(0.0, 1.0);
                    }
                }
                let gain_support = sample_luma_f32(&owner_log_gain_support_blur, ax, ay);
                if gain_support > 0.03 {
                    let log_gain =
                        get_high_quality_interpolated_pixel(&owner_log_gain_blur, ax, ay);
                    for channel in 0..3 {
                        let gain = (log_gain[channel] / gain_support).exp().clamp(0.50, 2.00);
                        pixel[channel] = (pixel[channel] * gain).clamp(0.0, 1.0);
                    }
                }
            }
        });
    println!(
        "  - Source-owner low-frequency correction: {} owners, {} analysis cells",
        owner_count, corrected_cells
    );
    Ok(corrected_cells)
}

pub(super) fn sample_luma_f32(
    image: &image::ImageBuffer<Luma<f32>, Vec<f32>>,
    x: f64,
    y: f64,
) -> f32 {
    if image.width() == 0 || image.height() == 0 {
        return 0.0;
    }
    let x = x.clamp(0.0, image.width().saturating_sub(1) as f64);
    let y = y.clamp(0.0, image.height().saturating_sub(1) as f64);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let tx = (x - x0 as f64) as f32;
    let ty = (y - y0 as f64) as f32;
    let p00 = image.get_pixel(x0, y0)[0];
    let p10 = image.get_pixel(x1, y0)[0];
    let p01 = image.get_pixel(x0, y1)[0];
    let p11 = image.get_pixel(x1, y1)[0];
    let top = p00 * (1.0 - tx) + p10 * tx;
    let bottom = p01 * (1.0 - tx) + p11 * tx;
    top * (1.0 - ty) + bottom * ty
}
