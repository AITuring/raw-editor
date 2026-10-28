//! Overlap-based exposure harmonization of the streaming focus mosaic.
//!
//! Low-frequency consensus samples collected while sources are written, seam
//! gain fields between capture groups and the group tone relations measured on
//! corresponding overlap pixels; every correction is a smooth field applied to
//! the source-owned output.

use super::super::stitching::get_high_quality_interpolated_pixel;
use image::GrayImage;
use image::Luma;
use image::Rgb;
use image::Rgb32FImage;
use rayon::prelude::*;
use std::collections::BTreeMap;

use super::{
    Field, STREAMING_CONSENSUS_MAX_DELTA, STREAMING_CONSENSUS_MIN_SUPPORT,
    STREAMING_CONSENSUS_SIGMA, STREAMING_GROUP_GAIN_FEATHER, STREAMING_GROUP_GAIN_MAX_LOG,
    STREAMING_HARMONIZATION_LONG_SIDE, StreamingMosaicStore, adjusted, luma, median,
};

pub(super) type GroupToneRelation = ((u8, u8), [f64; 3], f64);

/// Low-frequency colour samples collected while the tiled source loop is
/// running.  Samples are accepted only where a new source overlaps an
/// already-covered canvas position, so the final field is a consensus of
/// observations rather than an extrapolation from a single frame.  The
/// source-owned output is corrected only after it has been materialised.
pub(super) struct StreamingLowFrequencyConsensus {
    pub(super) scale: f64,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) target_sum: Vec<[f64; 3]>,
    pub(super) support: Vec<f32>,
}

impl StreamingLowFrequencyConsensus {
    pub(super) fn new(canvas_width: u32, canvas_height: u32) -> Self {
        let scale = (canvas_width.max(canvas_height) as f64
            / STREAMING_HARMONIZATION_LONG_SIDE as f64)
            .max(1.0);
        let width = (canvas_width as f64 / scale).ceil().max(1.0) as u32;
        let height = (canvas_height as f64 / scale).ceil().max(1.0) as u32;
        let size = width as usize * height as usize;
        Self {
            scale,
            width,
            height,
            target_sum: vec![[0.0; 3]; size],
            support: vec![0.0; size],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_overlap(
        &mut self,
        left: u32,
        top: u32,
        layer_scale: f64,
        base: &Rgb32FImage,
        base_mask: &GrayImage,
        candidate: &Rgb32FImage,
        candidate_mask: &GrayImage,
        tone: &Field<3>,
    ) {
        let width = base
            .width()
            .min(candidate.width())
            .min(base_mask.width())
            .min(candidate_mask.width());
        let height = base
            .height()
            .min(candidate.height())
            .min(base_mask.height())
            .min(candidate_mask.height());
        // Analysis images are already low-pass samples for the projected
        // source.  Decimating the accumulator further makes the consensus
        // insensitive to individual brush strokes while keeping this pass
        // bounded for a large 100-frame stack.
        for y in (0..height).step_by(4) {
            for x in (0..width).step_by(4) {
                if base_mask.get_pixel(x, y)[0] == 0 || candidate_mask.get_pixel(x, y)[0] == 0 {
                    continue;
                }
                let base_pixel = *base.get_pixel(x, y);
                // Compare the same tone-adjusted candidate pixels that are
                // written to the source-owned output.  Feeding raw candidate
                // photometry here would apply the per-layer exposure field a
                // second time when the consensus delta is added below.
                let target_pixel =
                    adjusted(*candidate.get_pixel(x, y), tone.at(x as f64, y as f64));
                let base_luma = luma(base_pixel);
                let target_luma = luma(target_pixel);
                if base_luma < 0.06
                    || target_luma < 0.06
                    || target_luma > 0.93
                    // Keep the overlap solver useful across the larger
                    // exposure steps that remain after the per-layer tone
                    // field.  Dark ink is still excluded by the near-black
                    // guard above; this bound is for paper-toned samples.
                    || (base_luma - target_luma).abs() > 0.35
                    || base_pixel
                        .0
                        .iter()
                        .chain(target_pixel.0.iter())
                        .any(|value| !value.is_finite() || !(0.01..0.995).contains(value))
                {
                    continue;
                }
                let global_x = left as f64 + x as f64 * layer_scale;
                let global_y = top as f64 + y as f64 * layer_scale;
                let grid_x = (global_x / self.scale).floor() as u32;
                let grid_y = (global_y / self.scale).floor() as u32;
                if grid_x >= self.width || grid_y >= self.height {
                    continue;
                }
                let index = grid_y as usize * self.width as usize + grid_x as usize;
                for channel in 0..3 {
                    self.target_sum[index][channel] += f64::from(target_pixel[channel]);
                }
                self.support[index] += 1.0;
            }
        }
    }

    pub(super) fn apply(&self, output: &mut Rgb32FImage, crop: (u32, u32, u32, u32)) {
        let supported_cells = self.support.iter().filter(|&&value| value > 0.0).count();
        if supported_cells == 0 {
            return;
        }
        println!(
            "  - Low-frequency source consensus: {supported_cells} supported analysis cells at {}x{} (scale {:.2})",
            self.width, self.height, self.scale
        );
        let target_weight = Rgb32FImage::from_fn(self.width, self.height, |x, y| {
            let index = y as usize * self.width as usize + x as usize;
            let count = self.support[index];
            if count <= 0.0 {
                Rgb([0.0; 3])
            } else {
                Rgb(std::array::from_fn(|channel| {
                    (self.target_sum[index][channel] / f64::from(count)) as f32
                }))
            }
        });
        let support = image::ImageBuffer::from_fn(self.width, self.height, |x, y| {
            Luma([self.support[y as usize * self.width as usize + x as usize]])
        });
        let premultiplied = Rgb32FImage::from_fn(self.width, self.height, |x, y| {
            let weight = support.get_pixel(x, y)[0];
            let pixel = target_weight.get_pixel(x, y);
            Rgb([pixel[0] * weight, pixel[1] * weight, pixel[2] * weight])
        });
        let target_blur_sum = image::imageops::blur(&premultiplied, STREAMING_CONSENSUS_SIGMA);
        let support_blur = image::imageops::blur(&support, STREAMING_CONSENSUS_SIGMA);
        let target_blur = Rgb32FImage::from_fn(self.width, self.height, |x, y| {
            let weight = support_blur.get_pixel(x, y)[0];
            if weight <= 1e-5 {
                return Rgb([0.0; 3]);
            }
            let pixel = target_blur_sum.get_pixel(x, y);
            Rgb(std::array::from_fn(|channel| pixel[channel] / weight))
        });
        let output_width = (output.width() as f64 / self.scale).ceil().max(1.0) as u32;
        let output_height = (output.height() as f64 / self.scale).ceil().max(1.0) as u32;
        let output_low_resized = image::imageops::resize(
            output,
            output_width,
            output_height,
            image::imageops::FilterType::Triangle,
        );
        // Compare like with like: target_low is blurred on the consensus
        // grid, so the source-owned output must use the same low-pass before
        // subtraction.  Adding target_low - output_low then leaves every
        // high-frequency ownership pixel untouched.
        let output_low = image::imageops::blur(&output_low_resized, STREAMING_CONSENSUS_SIGMA);
        let crop_left = crop.0 as f64;
        let crop_top = crop.1 as f64;
        let output_width_u32 = output.width();
        let output_width_usize = output_width_u32 as usize;
        output
            .as_mut()
            .par_chunks_mut(output_width_usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                let global_y = crop_top + y as f64;
                let target_y =
                    (global_y / self.scale).clamp(0.0, self.height.saturating_sub(1) as f64);
                let output_y =
                    (y as f64 / self.scale).clamp(0.0, output_height.saturating_sub(1) as f64);
                for x in 0..output_width_usize {
                    let global_x = crop_left + x as f64;
                    let target_x =
                        (global_x / self.scale).clamp(0.0, self.width.saturating_sub(1) as f64);
                    let support_x = target_x
                        .round()
                        .clamp(0.0, self.width.saturating_sub(1) as f64)
                        as u32;
                    let support_y = target_y
                        .round()
                        .clamp(0.0, self.height.saturating_sub(1) as f64)
                        as u32;
                    let support_value = support_blur.get_pixel(support_x, support_y)[0];
                    if support_value < STREAMING_CONSENSUS_MIN_SUPPORT {
                        continue;
                    }
                    let target =
                        get_high_quality_interpolated_pixel(&target_blur, target_x, target_y);
                    let output_x =
                        (x as f64 / self.scale).clamp(0.0, output_width.saturating_sub(1) as f64);
                    let output_low_pixel =
                        get_high_quality_interpolated_pixel(&output_low, output_x, output_y);
                    let output_low_luma = luma(output_low_pixel);
                    // Do not diffuse an overlap correction into the dark
                    // capture border or clipped white frame hardware.  These
                    // pixels are outside the paper-tone solve and otherwise
                    // pull the support blur into visible edge blocks.
                    if !(0.06..0.93).contains(&output_low_luma) {
                        continue;
                    }
                    let start = x * 3;
                    for channel in 0..3 {
                        let delta = (f64::from(target[channel])
                            - f64::from(output_low_pixel[channel]))
                        .clamp(
                            -STREAMING_CONSENSUS_MAX_DELTA,
                            STREAMING_CONSENSUS_MAX_DELTA,
                        );
                        row[start + channel] =
                            (f64::from(row[start + channel]) + delta).clamp(0.0, 1.0) as f32;
                    }
                }
            });
    }
}

pub(super) fn streaming_seam_harmonization(
    store: &StreamingMosaicStore,
    crop: (u32, u32, u32, u32),
    measured_relations: &[GroupToneRelation],
) -> Result<Option<(Rgb32FImage, f64)>, String> {
    let (left, top, width, height) = crop;
    let scale = (width.max(height) as f64 / STREAMING_HARMONIZATION_LONG_SIDE as f64).max(1.0);
    let analysis_width = (width as f64 / scale).ceil() as u32;
    let analysis_height = (height as f64 / scale).ceil() as u32;
    let (analysis, mask, owner) =
        store.analysis_region(left, top, scale, analysis_width, analysis_height)?;
    let mut boundary_count = 0usize;
    let mut owner_pixels = [0usize; 256];
    for y in 0..analysis_height {
        for x in 0..analysis_width {
            let current = owner.get_pixel(x, y)[0];
            if current == 0 || mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            owner_pixels[current as usize] += 1;
            let boundary = [
                x.checked_sub(1).map(|nx| (nx, y)),
                (x + 1 < analysis_width).then_some((x + 1, y)),
                y.checked_sub(1).map(|ny| (x, ny)),
                (y + 1 < analysis_height).then_some((x, y + 1)),
            ]
            .into_iter()
            .flatten()
            .any(|(nx, ny)| {
                let neighbour = owner.get_pixel(nx, ny)[0];
                neighbour != 0 && neighbour != current
            });
            if boundary {
                boundary_count += 1;
            }
        }
    }
    if boundary_count == 0 {
        return Ok(None);
    }
    // Estimate one exposure/colour correction per camera position from paper
    // immediately across its ownership boundaries.  The previous per-pixel
    // ratio of two blurred images followed dark strokes and created a bright
    // unsharp-mask halo around calligraphy.  Group-constant gains cannot trace
    // glyph contours, and dark/strongly coloured foreground is excluded here.
    let paper = image::imageops::blur(&analysis, 2.0);
    let mut pair_samples: BTreeMap<(u8, u8), Vec<[f64; 3]>> = BTreeMap::new();
    let probe = 6u32;
    let mut sample_pair = |ax: u32, ay: u32, bx: u32, by: u32| {
        if mask.get_pixel(ax, ay)[0] == 0 || mask.get_pixel(bx, by)[0] == 0 {
            return;
        }
        let a_owner = owner.get_pixel(ax, ay)[0];
        let b_owner = owner.get_pixel(bx, by)[0];
        if a_owner == 0 || b_owner == 0 || a_owner == b_owner {
            return;
        }
        let a = paper.get_pixel(ax, ay);
        let b = paper.get_pixel(bx, by);
        let a_luma = luma(*a);
        let b_luma = luma(*b);
        // Museum scans are often deliberately underexposed. Requiring 0.18
        // luma discarded most of this painting and left the problematic
        // camera group disconnected from the exposure solve. Group-constant
        // gains cannot follow brush contours, so darker overlap samples are
        // safe here; retain only a strict near-black and mismatch guard.
        if a_luma < 0.06 || b_luma < 0.06 || (a_luma - b_luma).abs() > 0.35 {
            return;
        }
        let (key, sign) = if a_owner < b_owner {
            ((a_owner, b_owner), 1.0)
        } else {
            ((b_owner, a_owner), -1.0)
        };
        pair_samples
            .entry(key)
            .or_default()
            .push(std::array::from_fn(|channel| {
                ((f64::from(b[channel]).max(0.015) / f64::from(a[channel]).max(0.015)).ln() * sign)
                    .clamp(-STREAMING_GROUP_GAIN_MAX_LOG, STREAMING_GROUP_GAIN_MAX_LOG)
            }));
    };
    for y in 0..analysis_height {
        for x in 0..analysis_width.saturating_sub(1) {
            if owner.get_pixel(x, y)[0] != owner.get_pixel(x + 1, y)[0]
                && x >= probe
                && x + probe < analysis_width
            {
                sample_pair(x - probe, y, x + probe, y);
            }
        }
    }
    for y in 0..analysis_height.saturating_sub(1) {
        for x in 0..analysis_width {
            if owner.get_pixel(x, y)[0] != owner.get_pixel(x, y + 1)[0]
                && y >= probe
                && y + probe < analysis_height
            {
                sample_pair(x, y - probe, x, y + probe);
            }
        }
    }
    // The per-owner log gains below are solved by repeatedly accumulating
    // `total += target * edge_weight` over every pair relation.  That floating
    // point sum is order sensitive, so the relation list must be ordered by owner
    // pair rather than by hash seed (需求 14.6).
    let mut pair_offsets: BTreeMap<(u8, u8), ([f64; 3], f64)> = pair_samples
        .into_iter()
        .filter_map(|(pair, samples)| {
            (samples.len() >= 12).then(|| {
                let offset: [f64; 3] = std::array::from_fn(|channel| {
                    median(
                        &mut samples
                            .iter()
                            .map(|value| value[channel])
                            .collect::<Vec<_>>(),
                    )
                });
                (pair, (offset, samples.len().min(512) as f64))
            })
        })
        .collect();
    // Same-coordinate overlap measurements are more reliable than probing
    // opposite sides of a seam, where the painting content can differ. There
    // can be many observations for one source pair as later tiles overlap a
    // broad existing mosaic; reduce those observations robustly instead of
    // letting the last frame overwrite the pair relation.
    let mut measured_by_pair = BTreeMap::<(u8, u8), Vec<([f64; 3], f64)>>::new();
    for &(pair, offset, weight) in measured_relations {
        measured_by_pair
            .entry(pair)
            .or_default()
            .push((offset, weight));
    }
    for (pair, observations) in measured_by_pair {
        let offset = std::array::from_fn(|channel| {
            median(
                &mut observations
                    .iter()
                    .map(|(value, _)| value[channel])
                    .collect::<Vec<_>>(),
            )
        });
        let weight = observations
            .iter()
            .map(|(_, weight)| *weight)
            .sum::<f64>()
            .min(2048.0);
        pair_offsets.insert(pair, (offset, weight));
    }
    let pair_offsets = pair_offsets
        .into_iter()
        .map(|(pair, (offset, weight))| (pair, offset, weight))
        .collect::<Vec<_>>();
    let anchor = (1u8..=u8::MAX)
        .max_by_key(|&id| owner_pixels[id as usize])
        .unwrap_or(1);
    let mut log_gains = [[0.0f64; 3]; 256];
    for _ in 0..48 {
        let previous = log_gains;
        for id in 1u8..=u8::MAX {
            if id == anchor || owner_pixels[id as usize] == 0 {
                continue;
            }
            for channel in 0..3 {
                let mut total = 0.0;
                let mut weight = 0.0;
                for &((a, b), offset, edge_weight) in &pair_offsets {
                    let target = if id == a {
                        previous[b as usize][channel] + offset[channel]
                    } else if id == b {
                        previous[a as usize][channel] - offset[channel]
                    } else {
                        continue;
                    };
                    total += target * edge_weight;
                    weight += edge_weight;
                }
                if weight > 0.0 {
                    log_gains[id as usize][channel] = (total / weight)
                        .clamp(-STREAMING_GROUP_GAIN_MAX_LOG, STREAMING_GROUP_GAIN_MAX_LOG);
                }
            }
        }
    }
    let owner_gains = Rgb32FImage::from_fn(analysis_width, analysis_height, |x, y| {
        let id = owner.get_pixel(x, y)[0] as usize;
        if id == 0 {
            Rgb([1.0; 3])
        } else {
            Rgb(std::array::from_fn(|channel| {
                log_gains[id][channel].exp() as f32
            }))
        }
    });
    if std::env::var_os("RAW_EDITOR_DEBUG_GROUP_GAINS").is_some() {
        let mut values = Vec::new();
        for id in 1u8..=u8::MAX {
            if owner_pixels[id as usize] == 0 {
                continue;
            }
            values.push((id, log_gains[id as usize]));
        }
        println!("  - Debug group gains: {:?}", values);
    }
    let gains = image::imageops::blur(&owner_gains, STREAMING_GROUP_GAIN_FEATHER);
    println!(
        "  - Group exposure harmonization: {} boundary samples, {} robust group relations at {}x{}",
        boundary_count,
        pair_offsets.len(),
        analysis_width,
        analysis_height
    );
    Ok(Some((gains, scale)))
}

pub(super) fn streaming_group_tone_relations_from_analysis(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    base_owner: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    tone: &Field<3>,
    capture_group_id: u8,
) -> Vec<GroupToneRelation> {
    let width = base.width().min(candidate.width()).min(base_owner.width());
    let height = base
        .height()
        .min(candidate.height())
        .min(base_owner.height());
    // Ordered by owner pair: the returned relations are later folded into the
    // shared `pair_offsets` map and into a float accumulation (需求 14.6).
    let mut samples = BTreeMap::<(u8, u8), Vec<[f64; 3]>>::new();
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            if base_mask.get_pixel(x, y)[0] == 0 || candidate_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let base_group = base_owner.get_pixel(x, y)[0];
            if base_group == 0 || base_group == capture_group_id {
                continue;
            }
            let current = *base.get_pixel(x, y);
            // The candidate is written to the tiled store through
            // `adjusted(pixel, tone)` below.  Measure the same corrected
            // pixels here; comparing raw source values would apply the
            // overlap exposure correction a second time during the final
            // owner-gain pass and can blow out a whole source rectangle.
            let source = adjusted(*candidate.get_pixel(x, y), tone.at(x as f64, y as f64));
            let current_luma = luma(current);
            let source_luma = luma(source);
            if current_luma < 0.045
                || source_luma < 0.045
                || (current_luma - source_luma).abs() > 0.35
                || current
                    .0
                    .iter()
                    .chain(source.0.iter())
                    .any(|value| !(0.015..0.97).contains(value))
            {
                continue;
            }
            let (pair, sign) = if base_group < capture_group_id {
                ((base_group, capture_group_id), 1.0)
            } else {
                ((capture_group_id, base_group), -1.0)
            };
            samples
                .entry(pair)
                .or_default()
                .push(std::array::from_fn(|channel| {
                    (f64::from(source[channel] / current[channel]).ln() * sign)
                        .clamp(-STREAMING_GROUP_GAIN_MAX_LOG, STREAMING_GROUP_GAIN_MAX_LOG)
                }));
        }
    }
    samples
        .into_iter()
        .filter_map(|(pair, values)| {
            (values.len() >= 24).then(|| {
                let offset = std::array::from_fn(|channel| {
                    median(
                        &mut values
                            .iter()
                            .map(|value| value[channel])
                            .collect::<Vec<_>>(),
                    )
                });
                (pair, offset, values.len().min(2048) as f64)
            })
        })
        .collect()
}

pub(super) fn apply_streaming_seam_harmonization(
    output: &mut Rgb32FImage,
    gains: &Rgb32FImage,
    scale: f64,
) {
    let width = output.width() as usize;
    output
        .as_mut()
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let gain_y = (y as f64 / scale).clamp(0.0, gains.height().saturating_sub(1) as f64);
            for x in 0..width {
                let gain_x = (x as f64 / scale).clamp(0.0, gains.width().saturating_sub(1) as f64);
                let gain = get_high_quality_interpolated_pixel(gains, gain_x, gain_y);
                for channel in 0..3 {
                    row[x * 3 + channel] = (row[x * 3 + channel] * gain[channel]).clamp(0.0, 1.0);
                }
            }
        });
}
