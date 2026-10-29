//! Per-tile tone transforms.
//!
//! The joint affine solve over pair relations, the source-fidelity constraint,
//! and the disabled spatial residual field with its overlap holdout gate.

use crate::panorama_utils::photometric::PairwiseDifference;
use crate::panorama_utils::photometric::solve_pairwise_differences;
use image::GrayImage;
use image::Luma;
use image::Rgb;
use image::Rgb32FImage;

use super::{
    PairField, TONE_LOW_FREQUENCY_SIGMA, TONE_MAX_ABS_OFFSET, TONE_MAX_GAIN, TONE_MIN_GAIN,
    TONE_MIN_SAMPLES, TONE_RELATION_HUBER_LOG, TONE_SPATIAL_ITERATIONS, ToneRelation, ToneTile,
    delta_e00_rgb, gaussian_cells, linear_luma, log_channel, median_f64,
};

#[derive(Clone, Copy)]
pub(super) struct ToneTransform {
    /// Applied (bounded) correction `corrected = gain · value + offset`.
    pub(super) gain: [f32; 3],
    pub(super) offset: [f32; 3],
    /// The joint solution before the per-tile bounds (需求 9.9).
    pub(super) solved_gain: [f32; 3],
    pub(super) solved_offset: [f32; 3],
    /// The solved gain or offset was bounded.
    pub(super) clamped: bool,
    /// The tile is in at least one relation with evidence.
    pub(super) supported: bool,
}

impl ToneTransform {
    pub(super) const IDENTITY: Self = Self {
        gain: [1.0; 3],
        offset: [0.0; 3],
        solved_gain: [1.0; 3],
        solved_offset: [0.0; 3],
        clamped: false,
        supported: false,
    };
}

#[derive(Clone)]
pub(super) struct ToneSpatialField {
    pub(super) log_gain: Rgb32FImage,
    pub(super) validity: GrayImage,
    pub(super) world_origin: (f64, f64),
    pub(super) world_stride: f64,
}

#[derive(Clone, Copy, Default)]
pub(super) struct ToneHoldoutGate {
    pub(super) samples: usize,
    pub(super) before_median: f64,
    pub(super) before_p95: f64,
    pub(super) after_median: f64,
    pub(super) after_p95: f64,
    pub(super) applied: bool,
}

pub(super) fn spatial_cell(
    world_origin: (f64, f64),
    world_stride: f64,
    width: u32,
    height: u32,
    world_x: f64,
    world_y: f64,
) -> Option<usize> {
    let x = ((world_x - world_origin.0) / world_stride).floor();
    let y = ((world_y - world_origin.1) / world_stride).floor();
    if x < 0.0 || y < 0.0 || x >= f64::from(width) || y >= f64::from(height) {
        return None;
    }
    Some(y as usize * width as usize + x as usize)
}

pub(super) fn build_spatial_fields(
    tiles: &[ToneTile],
    relations: &[ToneRelation],
    transforms: &[ToneTransform],
    prior: Option<&[ToneSpatialField]>,
    checkerboard_train_only: bool,
) -> Vec<ToneSpatialField> {
    let world_stride = tiles
        .iter()
        .map(|tile| tile.world_stride)
        .fold(0.0f64, f64::max);
    let world_origin = (
        tiles
            .iter()
            .map(|tile| tile.world_origin.0)
            .fold(f64::INFINITY, f64::min),
        tiles
            .iter()
            .map(|tile| tile.world_origin.1)
            .fold(f64::INFINITY, f64::min),
    );
    let world_end = (
        tiles
            .iter()
            .map(|tile| tile.world_origin.0 + tile.world_size.0)
            .fold(f64::NEG_INFINITY, f64::max),
        tiles
            .iter()
            .map(|tile| tile.world_origin.1 + tile.world_size.1)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let width = ((world_end.0 - world_origin.0) / world_stride).ceil() as u32;
    let height = ((world_end.1 - world_origin.1) / world_stride).ceil() as u32;
    let field_len = width as usize * height as usize;
    let mut sums = vec![vec![[0.0f64; 3]; field_len]; tiles.len()];
    let mut degrees = vec![vec![0u32; field_len]; tiles.len()];
    for relation in relations {
        let Some(field) = PairField::new(&tiles[relation.first], &tiles[relation.second]) else {
            continue;
        };
        let mut edge_sums = [
            vec![[0.0f64; 3]; sums[relation.first].len()],
            vec![[0.0f64; 3]; sums[relation.second].len()],
        ];
        let mut edge_counts = [
            vec![0u32; sums[relation.first].len()],
            vec![0u32; sums[relation.second].len()],
        ];
        for index in 0..field.sample.len() {
            if !field.sample[index] {
                continue;
            }
            let column = index % field.width;
            let row = index / field.width;
            let world_x =
                field.origin.0 + column as f64 * field.stride + (field.stride - 1.0) * 0.5;
            let world_y = field.origin.1 + row as f64 * field.stride + (field.stride - 1.0) * 0.5;
            let Some(cell) =
                spatial_cell(world_origin, world_stride, width, height, world_x, world_y)
            else {
                continue;
            };
            if checkerboard_train_only
                && ((cell % width as usize) + (cell / width as usize)) % 2 != 0
            {
                continue;
            }
            let first = field.first[index];
            let second = field.second[index];
            let corrected = |tile: usize, pixel: [f32; 3]| {
                let local = prior
                    .and_then(|fields| sample_spatial_gain(&fields[tile], world_x, world_y))
                    .unwrap_or([1.0; 3]);
                std::array::from_fn::<f32, 3, _>(|channel| {
                    transforms[tile].gain[channel] * local[channel] * pixel[channel]
                        + transforms[tile].offset[channel]
                })
            };
            let first = corrected(relation.first, first);
            let second = corrected(relation.second, second);
            let error = std::array::from_fn::<f64, 3, _>(|channel| {
                log_channel(first[channel]) - log_channel(second[channel])
            });
            for (side, sign) in [(0usize, -0.5), (1usize, 0.5)] {
                let Some(cell) =
                    spatial_cell(world_origin, world_stride, width, height, world_x, world_y)
                else {
                    continue;
                };
                for channel in 0..3 {
                    edge_sums[side][cell][channel] += sign * error[channel];
                }
                edge_counts[side][cell] += 1;
            }
        }
        // A transformed pair grid need not share cell origins with either
        // tile. Average all pair samples landing in one tile cell before
        // counting the accepted relation as one graph edge; otherwise a
        // half-cell crop offset incorrectly changes the graph degree.
        for (side, tile_index) in [(0usize, relation.first), (1usize, relation.second)] {
            for cell in 0..edge_counts[side].len() {
                let count = edge_counts[side][cell];
                if count == 0 {
                    continue;
                }
                for channel in 0..3 {
                    sums[tile_index][cell][channel] +=
                        edge_sums[side][cell][channel] / f64::from(count);
                }
                degrees[tile_index][cell] += 1;
            }
        }
    }
    tiles
        .iter()
        .enumerate()
        .map(|(tile_index, _tile)| {
            let grid_width = width as usize;
            let grid_height = height as usize;
            let mut weights = vec![[0.0f64; 1]; field_len];
            let mut values = vec![[0.0f64; 3]; field_len];
            for index in 0..values.len() {
                let degree = degrees[tile_index][index];
                if degree == 0 {
                    continue;
                }
                // For one edge, split the correction equally between its two
                // stations. For a complete local graph this is the exact
                // zero-mean least-squares solution; it also remains bounded
                // and deterministic for a sparse accepted graph.
                let graph_factor = 2.0 / (f64::from(degree) + 1.0);
                values[index] = sums[tile_index][index].map(|value| value * graph_factor);
                weights[index] = [1.0];
            }
            let sigma = f64::from(TONE_LOW_FREQUENCY_SIGMA);
            let support = gaussian_cells(&weights, grid_width, grid_height, sigma);
            let blurred = gaussian_cells(&values, grid_width, grid_height, sigma);
            let mut log_gain = Rgb32FImage::new(width, height);
            let mut validity = GrayImage::new(width, height);
            for index in 0..support.len() {
                if support[index][0] <= 1e-9 {
                    continue;
                }
                let x = index as u32 % width;
                let y = index as u32 / width;
                let value = std::array::from_fn::<f32, 3, _>(|channel| {
                    let previous = prior
                        .map(|fields| {
                            f64::from(fields[tile_index].log_gain.get_pixel(x, y)[channel])
                        })
                        .unwrap_or(0.0);
                    let candidate = previous + blurred[index][channel] / support[index][0];
                    let minimum =
                        f64::from(TONE_MIN_GAIN / transforms[tile_index].gain[channel]).ln();
                    let maximum =
                        f64::from(TONE_MAX_GAIN / transforms[tile_index].gain[channel]).ln();
                    candidate.clamp(minimum, maximum) as f32
                });
                log_gain.put_pixel(x, y, Rgb(value));
                validity.put_pixel(x, y, Luma([255]));
            }
            ToneSpatialField {
                log_gain,
                validity,
                world_origin,
                world_stride,
            }
        })
        .collect()
}

pub(super) fn build_spatial_field_iterations(
    tiles: &[ToneTile],
    relations: &[ToneRelation],
    transforms: &[ToneTransform],
    checkerboard_train_only: bool,
) -> Vec<ToneSpatialField> {
    let mut fields = None;
    for _ in 0..TONE_SPATIAL_ITERATIONS {
        fields = Some(build_spatial_fields(
            tiles,
            relations,
            transforms,
            fields.as_deref(),
            checkerboard_train_only,
        ));
    }
    fields.unwrap_or_default()
}

pub(super) fn sample_spatial_gain(
    field: &ToneSpatialField,
    world_x: f64,
    world_y: f64,
) -> Option<[f32; 3]> {
    let (width, height) = field.log_gain.dimensions();
    let u =
        (world_x - field.world_origin.0 - (field.world_stride - 1.0) * 0.5) / field.world_stride;
    let v =
        (world_y - field.world_origin.1 - (field.world_stride - 1.0) * 0.5) / field.world_stride;
    let (u0, v0) = (u.floor(), v.floor());
    let mut sum = [0.0f64; 3];
    let mut total = 0.0f64;
    for (dy, weight_y) in [(0.0, 1.0 - (v - v0)), (1.0, v - v0)] {
        for (dx, weight_x) in [(0.0, 1.0 - (u - u0)), (1.0, u - u0)] {
            let (x, y) = (u0 + dx, v0 + dy);
            if x < 0.0 || y < 0.0 || x >= f64::from(width) || y >= f64::from(height) {
                continue;
            }
            let (x, y) = (x as u32, y as u32);
            if field.validity.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let weight = weight_x * weight_y;
            let value = field.log_gain.get_pixel(x, y).0;
            for channel in 0..3 {
                sum[channel] += weight * f64::from(value[channel]);
            }
            total += weight;
        }
    }
    (total > 1e-9).then(|| sum.map(|value| (value / total).exp() as f32))
}

pub(super) fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * quantile).round() as usize]
}

pub(super) fn spatial_holdout_gate(
    tiles: &[ToneTile],
    relations: &[ToneRelation],
    transforms: &[ToneTransform],
    fields: &[ToneSpatialField],
) -> ToneHoldoutGate {
    let mut before = Vec::new();
    let mut after = Vec::new();
    let Some(shared) = fields.first() else {
        return ToneHoldoutGate::default();
    };
    for relation in relations {
        let Some(field) = PairField::new(&tiles[relation.first], &tiles[relation.second]) else {
            continue;
        };
        for index in 0..field.sample.len() {
            let column = index % field.width;
            let row = index / field.width;
            if !field.sample[index] {
                continue;
            }
            let world_x =
                field.origin.0 + column as f64 * field.stride + (field.stride - 1.0) * 0.5;
            let world_y = field.origin.1 + row as f64 * field.stride + (field.stride - 1.0) * 0.5;
            let global_column =
                ((world_x - shared.world_origin.0) / shared.world_stride).floor() as i64;
            let global_row =
                ((world_y - shared.world_origin.1) / shared.world_stride).floor() as i64;
            if (global_column + global_row).rem_euclid(2) == 0 {
                continue;
            }
            let corrected = |tile: usize, pixel: [f32; 3], local: [f32; 3]| {
                std::array::from_fn::<f32, 3, _>(|channel| {
                    transforms[tile].gain[channel] * local[channel] * pixel[channel]
                        + transforms[tile].offset[channel]
                })
            };
            let first_global = corrected(relation.first, field.first[index], [1.0; 3]);
            let second_global = corrected(relation.second, field.second[index], [1.0; 3]);
            let Some(first_local) = sample_spatial_gain(&fields[relation.first], world_x, world_y)
            else {
                continue;
            };
            let Some(second_local) =
                sample_spatial_gain(&fields[relation.second], world_x, world_y)
            else {
                continue;
            };
            before.push(delta_e00_rgb(first_global, second_global));
            after.push(delta_e00_rgb(
                corrected(relation.first, field.first[index], first_local),
                corrected(relation.second, field.second[index], second_local),
            ));
        }
    }
    let samples = before.len();
    let before_median = percentile(&mut before.clone(), 0.5);
    let before_p95 = percentile(&mut before, 0.95);
    let after_median = percentile(&mut after.clone(), 0.5);
    let after_p95 = percentile(&mut after, 0.95);
    ToneHoldoutGate {
        samples,
        before_median,
        before_p95,
        after_median,
        after_p95,
        applied: samples >= TONE_MIN_SAMPLES
            && after_median < before_median
            && after_p95 < before_p95,
    }
}

/// Minimise one low-band affine coordinate's quadratic while retaining the
/// existing gain and offset bounds. `alpha` is the increment above gain 1.
/// Check every box edge as well as the unconstrained stationary point;
/// independently clamping the two free coefficients is not this minimum.
pub(in crate::panorama_utils::stack_pipeline) fn bounded_low_frequency_affine_coordinate(
    normal: [f64; 3],
    rhs: [f64; 2],
) -> (f64, f64) {
    let [a, b, c] = normal;
    let [x, y] = rhs;
    if !normal.iter().chain(rhs.iter()).all(|v| v.is_finite())
        || a <= 0.0
        || c <= 0.0
        || a * c < b * b
    {
        return (0.0, 0.0);
    }
    let lo = f64::from(TONE_MIN_GAIN) - 1.0;
    let hi = f64::from(TONE_MAX_GAIN) - 1.0;
    let bound = f64::from(TONE_MAX_ABS_OFFSET);
    let cost = |alpha: f64, offset: f64| {
        a * alpha * alpha + 2.0 * b * alpha * offset + c * offset * offset
            - 2.0 * x * alpha
            - 2.0 * y * offset
    };
    let mut best = (0.0, 0.0);
    let mut minimum = cost(best.0, best.1);
    let mut consider = |alpha: f64, offset: f64| {
        if !(lo..=hi).contains(&alpha) || !(-bound..=bound).contains(&offset) {
            return;
        }
        let value = cost(alpha, offset);
        if value < minimum {
            best = (alpha, offset);
            minimum = value;
        }
    };
    let determinant = a * c - b * b;
    if determinant > 1e-15 {
        consider((x * c - y * b) / determinant, (y * a - x * b) / determinant);
    }
    for alpha in [lo, hi] {
        consider(alpha, ((y - b * alpha) / c).clamp(-bound, bound));
    }
    for offset in [-bound, bound] {
        consider(((x - b * offset) / a).clamp(lo, hi), offset);
    }
    best
}

/// Keep calibrated source brightness and colour while applying the measured
/// broad correction. One strength per station avoids introducing a new local
/// edge. These rendering bounds are stricter than the unchanged export gates;
/// the Quality_Gate still compares the final pixels with the original RAW.
pub(super) fn constrain_source_fidelity(tile: &ToneTile, transform: &mut ToneTransform) -> f64 {
    let mut strength = 1.0f64;
    for (low, validity) in tile.low.pixels().zip(tile.validity.pixels()) {
        if validity[0] == 0 {
            continue;
        }
        let original = low.0;
        let luminance = linear_luma(original);
        if luminance <= 1e-8 {
            continue;
        }
        let delta = std::array::from_fn::<_, 3, _>(|channel| {
            (transform.gain[channel] - 1.0) * original[channel] + transform.offset[channel]
        });
        let allowed = |factor: f64| {
            let corrected =
                std::array::from_fn(|channel| original[channel] + delta[channel] * factor as f32);
            let ratio = linear_luma(corrected) / luminance;
            (0.96..=1.04).contains(&ratio)
                && corrected.iter().all(|&value| (0.0..=1.0).contains(&value))
                && delta_e00_rgb(original, corrected) <= 1.8
        };
        if allowed(strength) {
            continue;
        }
        let (mut low, mut high) = (0.0, strength);
        for _ in 0..20 {
            let middle = (low + high) * 0.5;
            if allowed(middle) {
                low = middle;
            } else {
                high = middle;
            }
        }
        strength = low;
    }
    if strength < 1.0 {
        for channel in 0..3 {
            transform.gain[channel] = 1.0 + (transform.gain[channel] - 1.0) * strength as f32;
            transform.offset[channel] *= strength as f32;
        }
        transform.clamped = true;
    }
    strength
}

/// Per-tile corrections from all relations at once (design: reuse the
/// `photometric.rs` joint solve).
///
/// Per channel, the log gains solve `ln gain[first] − ln gain[second] =
/// −log_gain` over every relation with Huber reweighting, each connected
/// component held at zero mean log gain, so neither station order nor an
/// anchor tile sets the level and a loop spreads its inconsistency instead of
/// passing it down a spanning tree. With the bounded gains fixed, the offsets
/// solve `offset[first] − offset[second] = median(gain[second] · second −
/// gain[first] · first)` with the same gauge. Both are then bounded per tile.
pub(super) fn solve_tone_transforms(
    tile_count: usize,
    relations: &[ToneRelation],
) -> Vec<ToneTransform> {
    let mut transforms = vec![ToneTransform::IDENTITY; tile_count];
    if relations.is_empty() {
        return transforms;
    }
    let mut log_gains = vec![[0.0f64; 3]; tile_count];
    for channel in 0..3 {
        let equations = relations
            .iter()
            .map(|relation| PairwiseDifference {
                first: relation.first,
                second: relation.second,
                value: -relation.log_gain[channel],
                weight: relation.weight(),
                huber_scale: Some(TONE_RELATION_HUBER_LOG + relation.scatter),
            })
            .collect::<Vec<_>>();
        let Some(solution) = solve_pairwise_differences(tile_count, &equations) else {
            return transforms;
        };
        for (tile_gains, value) in log_gains.iter_mut().zip(solution) {
            tile_gains[channel] = value;
        }
    }
    for relation in relations {
        transforms[relation.first].supported = true;
        transforms[relation.second].supported = true;
    }
    for (transform, log_gain) in transforms.iter_mut().zip(&log_gains) {
        transform.solved_gain = log_gain.map(|value| value.exp() as f32);
        transform.gain = transform
            .solved_gain
            .map(|value| value.clamp(TONE_MIN_GAIN, TONE_MAX_GAIN));
    }
    let mut offsets = vec![[0.0f64; 3]; tile_count];
    for channel in 0..3 {
        let equations = relations
            .iter()
            .map(|relation| {
                let first_gain = transforms[relation.first].gain[channel];
                let second_gain = transforms[relation.second].gain[channel];
                PairwiseDifference {
                    first: relation.first,
                    second: relation.second,
                    value: median_f64(relation.samples.iter().map(|sample| {
                        f64::from(second_gain * sample.source[channel])
                            - f64::from(first_gain * sample.owner[channel])
                    })),
                    weight: relation.weight(),
                    huber_scale: None,
                }
            })
            .collect::<Vec<_>>();
        if let Some(solution) = solve_pairwise_differences(tile_count, &equations) {
            for (tile_offsets, value) in offsets.iter_mut().zip(solution) {
                tile_offsets[channel] = value;
            }
        }
    }
    for (transform, offset) in transforms.iter_mut().zip(&offsets) {
        transform.solved_offset = offset.map(|value| value as f32);
        transform.offset = transform
            .solved_offset
            .map(|value| value.clamp(-TONE_MAX_ABS_OFFSET, TONE_MAX_ABS_OFFSET));
        transform.clamped =
            transform.gain != transform.solved_gain || transform.offset != transform.solved_offset;
    }
    transforms
}
