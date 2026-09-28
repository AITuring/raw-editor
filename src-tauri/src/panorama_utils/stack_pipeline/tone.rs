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
use super::report::{
    TONE_BOUNDARY_DELTA_E_THRESHOLD, ToneBoundaryViolationRecord, TonePairWithoutEvidenceRecord,
    ToneReport, ToneStatus, ToneTileRecord,
};

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
static RUN_REPORT: Mutex<Option<ToneReport>> = Mutex::new(None);

pub(crate) fn reset_run_records() {
    RUN_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    *RUN_REPORT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
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

pub(crate) fn record_run_report(report: ToneReport) {
    *RUN_REPORT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(report);
}

pub(crate) fn run_report_snapshot() -> Option<ToneReport> {
    RUN_REPORT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// One source's low-frequency field in output world coordinates.  `low` and
/// `validity` stay at the one-sixteenth analysis resolution; no full-size
/// low-frequency image is allocated by the harmoniser.
#[derive(Debug, Clone)]
pub(crate) struct ToneTile {
    pub(crate) station_index: usize,
    pub(crate) owner_id: u16,
    pub(crate) low: Rgb32FImage,
    pub(crate) validity: GrayImage,
    pub(crate) world_origin: (f64, f64),
    pub(crate) world_size: (f64, f64),
    pub(crate) world_stride: f64,
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
        if denominator.abs() > 1e-6 {
            let gain = (n.mul_add(sum_xy, -sum_x * sum_y) / denominator) as f32;
            let offset = ((sum_y - f64::from(gain) * sum_x) / n) as f32;
            if gain.is_finite() {
                solved_gain[channel] = gain;
            }
            if offset.is_finite() {
                solved_offset[channel] = offset;
            }
        } else if n > 0.0 {
            // A constant-gray overlap cannot identify two affine coefficients
            // independently.  Use the equivalent multiplicative solution,
            // which remains observable, finite, and preserves the measured
            // level instead of manufacturing a NaN or an arbitrary offset.
            let mean_x = sum_x / n;
            let mean_y = sum_y / n;
            if mean_x.abs() > f64::EPSILON {
                let gain = (mean_y / mean_x) as f32;
                if gain.is_finite() {
                    solved_gain[channel] = gain;
                    solved_offset[channel] = (mean_y - f64::from(gain) * mean_x) as f32;
                }
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
    let low = low_frequency_grid(image);
    bilinear_expand(&low, image.width(), image.height())
}

/// Analysis-resolution low-frequency field.  Callers that already retain an
/// ownership grid should keep this image and map it by world coordinates,
/// avoiding a full-resolution intermediate allocation.
pub(crate) fn low_frequency_grid(image: &Rgb32FImage) -> Rgb32FImage {
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
    image::imageops::blur(&low, TONE_LOW_FREQUENCY_SIGMA)
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

#[derive(Clone, Copy)]
struct ToneTransform {
    gain: [f32; 3],
    offset: [f32; 3],
    supported: bool,
}

impl ToneTransform {
    const IDENTITY: Self = Self {
        gain: [1.0; 3],
        offset: [0.0; 3],
        supported: false,
    };
}

/// Apply all owner-level tone corrections after the immutable ownership plane
/// has been selected.  Each `ToneTile` is an analysis-grid field in world
/// coordinates, so the function samples at most one low-frequency pixel per
/// output pixel and never allocates a second full-resolution RGB canvas.
pub(crate) fn harmonize_after_ownership(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    tiles: &[ToneTile],
    world_origin: (f64, f64),
) -> ToneReport {
    harmonize_after_ownership_with_evidence(panorama, owners, tiles, world_origin, None)
}

/// Production-facing entry point used by the owner compositor. `width` is
/// explicit because the ownership plane is commonly held as a flat immutable
/// slice; the height is derived from the output image.
pub(crate) fn harmonize_tone_tiles(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    width: u32,
    tiles: &[ToneTile],
    evidence: &GrayImage,
) -> ToneReport {
    if width != panorama.width() {
        let mut report = ToneReport::default();
        report.status = ToneStatus::Identity;
        record_run_report(report.clone());
        return report;
    }
    harmonize_after_ownership_with_evidence(panorama, owners, tiles, (0.0, 0.0), Some(evidence))
}

fn harmonize_after_ownership_with_evidence(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    tiles: &[ToneTile],
    world_origin: (f64, f64),
    evidence: Option<&GrayImage>,
) -> ToneReport {
    let mut report = ToneReport::default();
    if panorama.width() == 0 || panorama.height() == 0 || tiles.is_empty() {
        report.status = ToneStatus::Identity;
        record_run_report(report.clone());
        return report;
    }
    let pixel_count = panorama.width() as usize * panorama.height() as usize;
    if owners.len() != pixel_count {
        report.status = ToneStatus::Identity;
        record_run_report(report.clone());
        return report;
    }

    // Solve pairwise affine relations from the one-sixteenth fields, then
    // propagate each relation from a deterministic lowest station anchor.
    let mut adjacency: Vec<Vec<(usize, ToneSolve, bool)>> = vec![Vec::new(); tiles.len()];
    let mut sample_support = vec![0usize; tiles.len()];
    for left in 0..tiles.len() {
        for right in left + 1..tiles.len() {
            let samples = overlap_samples(&tiles[left], &tiles[right]);
            let retained_before = samples.len();
            let solve = solve_tone_pair(&samples);
            if solve.status == ToneSolveStatus::Identity {
                report
                    .pairs_without_evidence
                    .push(TonePairWithoutEvidenceRecord {
                        left: tiles[left].station_index,
                        right: tiles[right].station_index,
                        retained_samples: solve.retained_samples.min(retained_before) as u64,
                    });
                continue;
            }
            sample_support[left] = sample_support[left].max(solve.retained_samples);
            sample_support[right] = sample_support[right].max(solve.retained_samples);
            // `true` means the current node is the owner side of the relation
            // and the neighbour is the source side (owner ~= gain*source+offset).
            adjacency[left].push((right, solve.clone(), true));
            adjacency[right].push((left, solve, false));
        }
    }

    let mut transforms = vec![ToneTransform::IDENTITY; tiles.len()];
    let mut visited = vec![false; tiles.len()];
    for anchor in 0..tiles.len() {
        if visited[anchor] {
            continue;
        }
        visited[anchor] = true;
        transforms[anchor] = ToneTransform {
            ..ToneTransform::IDENTITY
        };
        let mut queue = std::collections::VecDeque::from([anchor]);
        while let Some(current) = queue.pop_front() {
            for &(next, ref relation, forward) in &adjacency[current] {
                if visited[next] {
                    continue;
                }
                let parent = transforms[current];
                let (gain, offset) = if forward {
                    (
                        std::array::from_fn(|channel| {
                            parent.gain[channel] * relation.gain[channel]
                        }),
                        std::array::from_fn(|channel| {
                            parent.gain[channel] * relation.offset[channel] + parent.offset[channel]
                        }),
                    )
                } else {
                    (
                        std::array::from_fn(|channel| {
                            parent.gain[channel] / relation.gain[channel].max(1e-6)
                        }),
                        std::array::from_fn(|channel| {
                            (parent.offset[channel] - relation.offset[channel])
                                / relation.gain[channel].max(1e-6)
                        }),
                    )
                };
                transforms[next] = ToneTransform {
                    gain,
                    offset,
                    supported: true,
                };
                visited[next] = true;
                queue.push_back(next);
            }
        }
    }

    for (index, tile) in tiles.iter().enumerate() {
        let transform = transforms[index];
        let clamped = transform
            .gain
            .iter()
            .any(|value| *value < TONE_MIN_GAIN || *value > TONE_MAX_GAIN)
            || transform
                .offset
                .iter()
                .any(|value| value.abs() > TONE_MAX_ABS_OFFSET);
        let gain = transform
            .gain
            .map(|value| value.clamp(TONE_MIN_GAIN, TONE_MAX_GAIN));
        let offset = transform
            .offset
            .map(|value| value.clamp(-TONE_MAX_ABS_OFFSET, TONE_MAX_ABS_OFFSET));
        if clamped {
            degradation::record_run_degradation(
                degradation::TONE_GAIN_CLAMPED,
                serde_json::json!({
                    "stage": "tone_harmonizer_graph",
                    "station_index": tile.station_index,
                    "solved_gain": transform.gain,
                    "gain": gain,
                    "solved_offset": transform.offset,
                    "offset": offset,
                }),
            );
        }
        report.tiles.push(ToneTileRecord {
            station_index: tile.station_index,
            gain: gain.map(f64::from),
            offset: offset.map(f64::from),
            samples: sample_support[index] as u64,
            clamped,
            solved_gain: transform.gain.map(f64::from),
            solved_offset: transform.offset.map(f64::from),
        });
    }

    // A tile owns its own high-frequency residual.  The low field is sampled
    // once and the residual is added back directly, preserving owner pixels in
    // every region where no overlap evidence exists.
    let output_width = panorama.width();
    let output_height = panorama.height();
    for (index, pixel) in panorama.as_mut().chunks_exact_mut(3).enumerate() {
        let owner_id = owners[index];
        let output_x = index as u32 % output_width;
        let output_y = index as u32 / output_width;
        let Some((tile_index, tile)) = tiles
            .iter()
            .enumerate()
            .find(|(_, tile)| tile.owner_id == owner_id)
        else {
            continue;
        };
        let transform = transforms[tile_index];
        if !transform.supported {
            continue;
        }
        if transform
            .gain
            .iter()
            .all(|value| (*value - 1.0).abs() <= f32::EPSILON)
            && transform
                .offset
                .iter()
                .all(|value| value.abs() <= f32::EPSILON)
        {
            continue;
        }
        if let Some(evidence) = evidence {
            if output_x >= evidence.width()
                || output_y >= evidence.height()
                || evidence.get_pixel(output_x, output_y)[0] == 0
            {
                continue;
            }
        }
        let world = (
            world_origin.0 + output_x as f64,
            world_origin.1 + output_y as f64,
        );
        let Some((low, valid)) = sample_tile(tile, world.0, world.1) else {
            continue;
        };
        if !valid {
            continue;
        }
        // `owner_low` is the same source field at this coordinate; only the
        // source's broad tone is changed, while the original owner residual is
        // retained exactly in the sum below.
        for channel in 0..3 {
            let owner = pixel[channel];
            let owner_low = low[channel];
            pixel[channel] = (transform.gain[channel] * low[channel]
                + transform.offset[channel]
                + (owner - owner_low))
                .clamp(0.0, 1.0);
        }
    }

    let corrected_low = |owner_id: u16, x: u32, y: u32| -> Option<[f32; 3]> {
        let (tile_index, tile) = tiles
            .iter()
            .enumerate()
            .find(|(_, tile)| tile.owner_id == owner_id)?;
        let (low, valid) = sample_tile(tile, world_origin.0 + x as f64, world_origin.1 + y as f64)?;
        if !valid {
            return None;
        }
        let transform = transforms[tile_index];
        Some(std::array::from_fn(|channel| {
            transform.gain[channel].clamp(TONE_MIN_GAIN, TONE_MAX_GAIN) * low[channel]
                + transform.offset[channel].clamp(-TONE_MAX_ABS_OFFSET, TONE_MAX_ABS_OFFSET)
        }))
    };
    for y in 0..output_height {
        for x in 0..output_width {
            let index = y as usize * output_width as usize + x as usize;
            let left_owner = owners[index];
            if left_owner == 0 {
                continue;
            }
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= output_width || ny >= output_height {
                    continue;
                }
                let right_index = ny as usize * panorama.width() as usize + nx as usize;
                let right_owner = owners[right_index];
                if right_owner == 0 || right_owner == left_owner {
                    continue;
                }
                // Inspect the 16px band on both sides of the ownership edge,
                // rather than only the two immediately adjacent pixels.
                for distance in 0..16u32 {
                    let (ax, ay) = if nx > x {
                        (x.saturating_sub(distance), y)
                    } else {
                        (x, y.saturating_sub(distance))
                    };
                    let (bx, by) = if nx > x {
                        (nx.saturating_add(distance).min(output_width - 1), ny)
                    } else {
                        (nx, ny.saturating_add(distance).min(output_height - 1))
                    };
                    let Some(left_low) = corrected_low(left_owner, ax, ay) else {
                        continue;
                    };
                    let Some(right_low) = corrected_low(right_owner, bx, by) else {
                        continue;
                    };
                    let delta = delta_e00_rgb(left_low, right_low);
                    report.boundary_delta_e.max = report.boundary_delta_e.max.max(delta);
                    if delta > TONE_BOUNDARY_DELTA_E_THRESHOLD
                        && report.boundary_delta_e.violations.len() < 4096
                    {
                        report
                            .boundary_delta_e
                            .violations
                            .push(ToneBoundaryViolationRecord {
                                left: left_owner as usize,
                                right: right_owner as usize,
                                delta_e00: delta,
                                world: super::report::WorldRect {
                                    left: world_origin.0 + ax as f64,
                                    top: world_origin.1 + ay as f64,
                                    width: 1.0,
                                    height: 1.0,
                                },
                            });
                        degradation::record_run_degradation(
                            degradation::TONE_BOUNDARY_DELTA_E_EXCEEDED,
                            serde_json::json!({
                                "left": left_owner,
                                "right": right_owner,
                                "delta_e00": delta,
                                "x": ax,
                                "y": ay,
                            }),
                        );
                    }
                }
            }
        }
    }
    let has_pair_evidence = adjacency.iter().any(|edges| !edges.is_empty());
    report.status = if !has_pair_evidence {
        ToneStatus::Identity
    } else if report.boundary_delta_e.violations.is_empty()
        && report.pairs_without_evidence.is_empty()
    {
        ToneStatus::Applied
    } else {
        ToneStatus::Degraded
    };
    for tile in &report.tiles {
        record_run_record(*tile);
    }
    record_run_report(report.clone());
    report
}

fn overlap_samples(left: &ToneTile, right: &ToneTile) -> Vec<ToneSample> {
    let left_stride = left.world_stride.max(f64::EPSILON);
    let right_stride = right.world_stride.max(f64::EPSILON);
    let left_end = (
        left.world_origin.0 + left.world_size.0,
        left.world_origin.1 + left.world_size.1,
    );
    let right_end = (
        right.world_origin.0 + right.world_size.0,
        right.world_origin.1 + right.world_size.1,
    );
    let min_x = left.world_origin.0.max(right.world_origin.0);
    let min_y = left.world_origin.1.max(right.world_origin.1);
    let max_x = left_end.0.min(right_end.0);
    let max_y = left_end.1.min(right_end.1);
    if max_x <= min_x || max_y <= min_y {
        return Vec::new();
    }
    let step = left_stride.max(right_stride);
    let columns = ((max_x - min_x) / step).floor() as usize + 1;
    let rows = ((max_y - min_y) / step).floor() as usize + 1;
    let mut samples = Vec::with_capacity(columns.saturating_mul(rows).min(16_384));
    for row in 0..rows {
        for column in 0..columns {
            if samples.len() >= 16_384 {
                return samples;
            }
            let x = min_x + (column as f64 + 0.5) * step;
            let y = min_y + (row as f64 + 0.5) * step;
            let Some((left_pixel, left_valid)) = sample_tile(left, x, y) else {
                continue;
            };
            let Some((right_pixel, right_valid)) = sample_tile(right, x, y) else {
                continue;
            };
            if left_valid && right_valid {
                samples.push(ToneSample {
                    owner: left_pixel,
                    source: right_pixel,
                });
            }
        }
    }
    samples
}

fn sample_tile(tile: &ToneTile, world_x: f64, world_y: f64) -> Option<([f32; 3], bool)> {
    if !tile.world_stride.is_finite() || tile.world_stride <= 0.0 {
        return None;
    }
    let x = ((world_x - tile.world_origin.0) / tile.world_stride).floor() as i64;
    let y = ((world_y - tile.world_origin.1) / tile.world_stride).floor() as i64;
    if x < 0 || y < 0 || x as u32 >= tile.low.width() || y as u32 >= tile.low.height() {
        return None;
    }
    let x = x as u32;
    let y = y as u32;
    if x >= tile.validity.width() || y >= tile.validity.height() {
        return Some((tile.low.get_pixel(x, y).0, false));
    }
    let valid = tile.validity.get_pixel(x, y)[0] != 0;
    Some((tile.low.get_pixel(x, y).0, valid))
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

    #[test]
    fn constant_gray_samples_have_a_finite_affine_solution() {
        let samples = vec![
            ToneSample {
                owner: [0.44; 3],
                source: [0.40; 3],
            };
            TONE_MIN_SAMPLES
        ];
        let solve = solve_tone_pair(&samples);
        assert!(solve.gain.iter().all(|value| value.is_finite()));
        assert!(solve.offset.iter().all(|value| value.is_finite()));
        for channel in 0..3 {
            let corrected = solve.gain[channel] * 0.40 + solve.offset[channel];
            assert!((corrected - 0.44).abs() < 1e-4);
        }
    }
}
