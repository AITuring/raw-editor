//! Low-frequency affine tone harmonisation for owned focus-stack pixels.
//!
//! Tone correction is deliberately separated from ownership.  The owner image
//! supplies the high-frequency residual while a low-pass version of the source
//! supplies only broad exposure and white-balance changes.  This keeps strokes,
//! edges, and texture tied to the source selected by the ownership map.

#![allow(dead_code)]

use image::{GrayImage, Luma, Rgb, Rgb32FImage};
use rayon::prelude::*;
use std::sync::Mutex;

use super::degradation;
use super::report::{
    TONE_BOUNDARY_DELTA_E_THRESHOLD, ToneBoundaryViolationRecord, TonePairWithoutEvidenceRecord,
    ToneReport, ToneStatus, ToneTileRecord,
};
use crate::panorama_utils::photometric::{PairwiseDifference, solve_pairwise_differences};

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
/// Width of the band on each side of an Owner_Region boundary (需求 9.8).
pub(crate) const TONE_BOUNDARY_BAND_PX: u32 = 16;
/// Boundary windows whose band means are compared separately, so a local
/// tone step is located instead of being averaged over a whole boundary.
pub(crate) const TONE_BOUNDARY_WINDOW_PX: u32 = 256;
/// Normalised luminance range of a usable tone sample on both sides (需求 9.1).
pub(crate) const TONE_MIN_SAMPLE_LUMINANCE: f32 = 0.02;
pub(crate) const TONE_MAX_SAMPLE_LUMINANCE: f32 = 0.98;
/// `ToneTile::validity` of a cell the tile covers entirely: tone evidence.
pub(crate) const TONE_CELL_SAMPLE: u8 = 255;
/// `ToneTile::validity` of a cell the tile covers in part: its covered-only
/// field is corrected with the tile, but it is no tone evidence.
pub(crate) const TONE_CELL_PARTIAL: u8 = 128;
/// Log-gain scatter at which a relation's joint-solve weight halves, and the
/// Huber scale floor of its residual (the values of `photometric.rs`).
const TONE_RELATION_SCATTER_SCALE: f64 = 0.04;
const TONE_RELATION_HUBER_LOG: f64 = 0.025;

// Tone solving is reached from the compositor's free functions, which do not
// carry a mutable Stack_Report reference.  Keep the same run-scoped sink used
// by Intra_Station and Focus_Fuser; report publication takes one snapshot at
// the terminating boundary.
static RUN_RECORDS: Mutex<Vec<ToneTileRecord>> = Mutex::new(Vec::new());
static RUN_REPORT: Mutex<Option<ToneReport>> = Mutex::new(None);
static RUN_TOPOLOGY: Mutex<Option<ToneStationTopology>> = Mutex::new(None);

/// The station context the owner compositor needs for tone evidence: each
/// Virtual_Tile's station index and the run's accepted Station_Relations,
/// both keyed by the tile's `ImageInfo::id`. Only the overlap of an accepted
/// relation yields tone samples (需求 9.1). It is published by the station
/// solve, which runs before the compositor, and like the residual model it is
/// read by the compositor rather than threaded through its signature.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ToneStationTopology {
    /// `(ImageInfo::id, station index)` of every Virtual_Tile.
    pub(crate) stations: Vec<(usize, usize)>,
    /// Accepted Station_Relations as `ImageInfo::id` pairs.
    pub(crate) accepted: Vec<(usize, usize)>,
}

impl ToneStationTopology {
    pub(crate) fn station_of(&self, image_id: usize) -> Option<usize> {
        self.stations
            .iter()
            .find(|(id, _)| *id == image_id)
            .map(|(_, station)| *station)
    }
}

/// Clear the station context at whole-run start. Kept apart from
/// [`reset_run_records`], which a compositor may call at its own start after
/// the station solve published this context.
pub(crate) fn reset_run_topology() {
    *RUN_TOPOLOGY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

pub(crate) fn record_run_topology(topology: ToneStationTopology) {
    *RUN_TOPOLOGY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(topology);
}

pub(crate) fn run_topology_snapshot() -> Option<ToneStationTopology> {
    RUN_TOPOLOGY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

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
    /// Unblurred mean of the covered tile pixels in the tile cell under each
    /// analysis cell, and that cell's covered fraction (row-major, the size
    /// of `low`). A pair of tiles is compared through [`PairField`], built
    /// from these over the cells both cover.
    pub(crate) cell_mean: Rgb32FImage,
    pub(crate) cell_coverage: Vec<f32>,
}

/// The low fields of two tiles over their common support only (需求 9.1).
///
/// Both are normalised convolutions of the same doubly covered cells, so they
/// see the same content window and differ by the tiles' tone and not by what
/// either tile covers alone. A tile's own field near its coverage edge
/// averages only its own side, which next to a content gradient separates it
/// from a neighbour whose field is still centred at that position.
struct PairField {
    origin: (f64, f64),
    stride: f64,
    width: usize,
    height: usize,
    first: Vec<[f32; 3]>,
    second: Vec<[f32; 3]>,
    /// Both tiles cover the cell entirely and the common field exists there.
    sample: Vec<bool>,
}

impl PairField {
    fn new(first: &ToneTile, second: &ToneTile) -> Option<Self> {
        let stride = first.world_stride.max(second.world_stride);
        if !stride.is_finite() || stride <= 0.0 {
            return None;
        }
        let origin = (
            first.world_origin.0.max(second.world_origin.0),
            first.world_origin.1.max(second.world_origin.1),
        );
        let end = (
            (first.world_origin.0 + first.world_size.0)
                .min(second.world_origin.0 + second.world_size.0),
            (first.world_origin.1 + first.world_size.1)
                .min(second.world_origin.1 + second.world_size.1),
        );
        if end.0 <= origin.0 || end.1 <= origin.1 {
            return None;
        }
        let width = ((end.0 - origin.0) / stride).ceil() as usize;
        let height = ((end.1 - origin.1) / stride).ceil() as usize;
        let cell = |tile: &ToneTile, x: f64, y: f64| -> Option<([f32; 3], f32)> {
            let cx = ((x - tile.world_origin.0) / tile.world_stride).floor();
            let cy = ((y - tile.world_origin.1) / tile.world_stride).floor();
            if cx < 0.0
                || cy < 0.0
                || cx >= f64::from(tile.cell_mean.width())
                || cy >= f64::from(tile.cell_mean.height())
            {
                return None;
            }
            let (cx, cy) = (cx as u32, cy as u32);
            let coverage = *tile
                .cell_coverage
                .get(cy as usize * tile.cell_mean.width() as usize + cx as usize)?;
            Some((tile.cell_mean.get_pixel(cx, cy).0, coverage))
        };
        let mut weights = vec![[0.0f64; 1]; width * height];
        let mut first_sums = vec![[0.0f64; 3]; width * height];
        let mut second_sums = vec![[0.0f64; 3]; width * height];
        let mut sample = vec![false; width * height];
        for row in 0..height {
            for column in 0..width {
                let x = origin.0 + column as f64 * stride + (stride - 1.0) * 0.5;
                let y = origin.1 + row as f64 * stride + (stride - 1.0) * 0.5;
                let (Some((first_mean, first_coverage)), Some((second_mean, second_coverage))) =
                    (cell(first, x, y), cell(second, x, y))
                else {
                    continue;
                };
                let index = row * width + column;
                let weight = f64::from(first_coverage.min(second_coverage));
                weights[index] = [weight];
                for channel in 0..3 {
                    first_sums[index][channel] = weight * f64::from(first_mean[channel]);
                    second_sums[index][channel] = weight * f64::from(second_mean[channel]);
                }
                sample[index] = first_coverage >= 1.0 && second_coverage >= 1.0;
            }
        }
        let sigma = f64::from(TONE_LOW_FREQUENCY_SIGMA);
        let support = gaussian_cells(&weights, width, height, sigma);
        let normalise = |sums: &[[f64; 3]]| {
            gaussian_cells(sums, width, height, sigma)
                .iter()
                .zip(&support)
                .map(|(sum, support)| {
                    if support[0] > 1e-9 {
                        sum.map(|value| (value / support[0]) as f32)
                    } else {
                        [0.0; 3]
                    }
                })
                .collect::<Vec<_>>()
        };
        let (first_field, second_field) = (normalise(&first_sums), normalise(&second_sums));
        for (index, flag) in sample.iter_mut().enumerate() {
            *flag &= support[index][0] > 1e-9;
        }
        Some(Self {
            origin,
            stride,
            width,
            height,
            first: first_field,
            second: second_field,
            sample,
        })
    }

    /// Both fields at the cell holding a world position, where it is a
    /// doubly and entirely covered cell.
    fn at(&self, world_x: f64, world_y: f64) -> Option<([f32; 3], [f32; 3])> {
        let column = ((world_x - self.origin.0) / self.stride).floor();
        let row = ((world_y - self.origin.1) / self.stride).floor();
        if column < 0.0 || row < 0.0 || column >= self.width as f64 || row >= self.height as f64 {
            return None;
        }
        let index = row as usize * self.width + column as usize;
        self.sample[index].then(|| (self.first[index], self.second[index]))
    }

    /// Every doubly and entirely covered cell whose two fields both lie in
    /// the usable luminance range, `first` as the owner side (需求 9.1).
    fn samples(&self) -> Vec<ToneSample> {
        let usable = |pixel: [f32; 3]| {
            let luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            (TONE_MIN_SAMPLE_LUMINANCE..=TONE_MAX_SAMPLE_LUMINANCE).contains(&luminance)
        };
        (0..self.sample.len())
            .filter(|&index| {
                self.sample[index] && usable(self.first[index]) && usable(self.second[index])
            })
            .map(|index| ToneSample {
                owner: self.first[index],
                source: self.second[index],
            })
            .collect()
    }
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

/// `ln` of a normalised channel value, bounded away from zero so a channel at
/// black gives a large but finite ratio that the MAD check then rejects.
fn log_channel(value: f32) -> f64 {
    f64::from(value.max(1.0 / 65_535.0)).ln()
}

/// Per-channel `ln owner − ln source` of one sample: the log gain that maps
/// the source side of the sample onto its owner side.
fn sample_log_gain(sample: &ToneSample) -> [f64; 3] {
    std::array::from_fn(|channel| {
        log_channel(sample.owner[channel]) - log_channel(sample.source[channel])
    })
}

/// The samples of one pair that pass the 3×MAD consistency check (需求 9.4).
///
/// The check runs on the per-sample RGB log gain, the quantity the tone
/// relation is estimated from. A sample is excluded when the distance of its
/// log-gain vector from the per-channel median exceeds three times the median
/// distance, so the three channels always keep the same samples.
fn consistent_samples(samples: &[ToneSample]) -> Vec<ToneSample> {
    if samples.is_empty() {
        return Vec::new();
    }
    let gains = samples.iter().map(sample_log_gain).collect::<Vec<_>>();
    let medians: [f64; 3] =
        std::array::from_fn(|channel| median_f64(gains.iter().map(|gain| gain[channel])));
    let deviations = gains
        .iter()
        .map(|gain| {
            (0..3)
                .map(|channel| (gain[channel] - medians[channel]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect::<Vec<_>>();
    let mad = median_f64(deviations.iter().copied());
    // An exact consensus (MAD 0) keeps the consensus up to rounding.
    let threshold = (3.0 * mad).max(1e-6);
    samples
        .iter()
        .zip(&deviations)
        .filter(|(_, deviation)| **deviation <= threshold)
        .map(|(sample, _)| *sample)
        .collect()
}

/// One accepted pair's multiplicative tone relation. `first` is the owner side
/// and `second` the source side of every retained sample.
struct ToneRelation {
    first: usize,
    second: usize,
    /// Per-channel median of `ln first − ln second` over `samples`.
    log_gain: [f64; 3],
    /// Median RMS distance of the sample log gains from `log_gain`.
    scatter: f64,
    samples: Vec<ToneSample>,
}

impl ToneRelation {
    fn new(first: usize, second: usize, samples: Vec<ToneSample>) -> Self {
        let gains = samples.iter().map(sample_log_gain).collect::<Vec<_>>();
        let log_gain: [f64; 3] =
            std::array::from_fn(|channel| median_f64(gains.iter().map(|gain| gain[channel])));
        let scatter = median_f64(gains.iter().map(|gain| {
            ((0..3)
                .map(|channel| (gain[channel] - log_gain[channel]).powi(2))
                .sum::<f64>()
                / 3.0)
                .sqrt()
        }));
        Self {
            first,
            second,
            log_gain,
            scatter,
            samples,
        }
    }

    /// The joint-solve weight of `photometric.rs`: more retained samples and a
    /// tighter log-gain scatter give a relation more pull.
    fn weight(&self) -> f64 {
        let support = self.samples.len() as f64;
        support
            / (support + TONE_MIN_SAMPLES as f64)
            / (1.0 + (self.scatter / TONE_RELATION_SCATTER_SCALE).powi(2))
    }
}

/// Solve one overlap's tone relation after the 3×MAD consistency check, the
/// two-tile case of the joint solve in [`harmonize_tone_tiles`].
///
/// The gain is the per-channel median log ratio of the retained samples and
/// the offset the robust median of `owner − gain · source` (design, Tone_Harmonizer:
/// "先用 log 增益求解得 gain，再对残差求稳健中位数得 offset"). A median ratio is
/// symmetric and does not flatten towards identity where the two low fields
/// disagree by position, which a least-squares affine fit of one field on the
/// other does. A pair with fewer than [`TONE_MIN_SAMPLES`] retained samples is
/// identity and records the stable `tone_insufficient_samples` identifier.
pub(crate) fn solve_tone_pair(samples: &[ToneSample]) -> ToneSolve {
    let retained = consistent_samples(samples);
    if retained.len() < TONE_MIN_SAMPLES {
        record_insufficient_samples(retained.len());
        return ToneSolve::identity(retained.len());
    }
    let relation = ToneRelation::new(0, 1, retained);
    let solved_gain = relation.log_gain.map(|value| value.exp() as f32);
    let gain =
        std::array::from_fn(|channel| solved_gain[channel].clamp(TONE_MIN_GAIN, TONE_MAX_GAIN));
    let solved_offset: [f32; 3] = std::array::from_fn(|channel| {
        median(
            relation
                .samples
                .iter()
                .map(|sample| sample.owner[channel] - gain[channel] * sample.source[channel]),
        )
    });
    let retained = relation.samples;
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

/// Separable Gaussian of a `width × height` grid of `N`-channel values with
/// zero padding, so that a normalised convolution sees no value outside it.
fn gaussian_cells<const N: usize>(
    values: &[[f64; N]],
    width: usize,
    height: usize,
    sigma: f64,
) -> Vec<[f64; N]> {
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel = (-radius..=radius)
        .map(|offset| (-(offset as f64).powi(2) / (2.0 * sigma * sigma)).exp())
        .collect::<Vec<_>>();
    let pass = |source: &[[f64; N]], horizontal: bool| {
        let mut output = vec![[0.0; N]; source.len()];
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0; N];
                for (tap, weight) in kernel.iter().enumerate() {
                    let offset = tap as isize - radius;
                    let (sx, sy) = if horizontal {
                        (x as isize + offset, y as isize)
                    } else {
                        (x as isize, y as isize + offset)
                    };
                    if sx < 0 || sy < 0 || sx >= width as isize || sy >= height as isize {
                        continue;
                    }
                    let value = source[sy as usize * width + sx as usize];
                    for channel in 0..N {
                        sum[channel] += weight * value[channel];
                    }
                }
                output[y * width + x] = sum;
            }
        }
        output
    };
    pass(&pass(values, true), false)
}

/// Tone_Harmonizer field of one tile on a world-aligned analysis grid.
///
/// `tile` holds the tile's pixels with uncovered pixels exactly zero (the
/// compositor's transparency convention). `bounds` are the tile's inclusive
/// output-pixel bounds `(left, top, right, bottom)` and `to_tile` maps an
/// output pixel to tile pixels with the map the compositor samples through.
///
/// The low-frequency field averages covered tile pixels only (a normalised
/// convolution), so uncovered payload never darkens it, and it is resampled
/// through `to_tile`, so a rotated or projective tile is compared with its
/// neighbours at the right world position. A cell is tone evidence
/// ([`TONE_CELL_SAMPLE`]) only where the tile covers its whole analysis cell
/// (需求 9.1, 9.2, 9.7).
pub(crate) fn world_aligned_tone_tile(
    station_index: usize,
    owner_id: u16,
    tile: &Rgb32FImage,
    bounds: (u32, u32, u32, u32),
    to_tile: impl Fn(f64, f64) -> Option<(f64, f64)>,
) -> ToneTile {
    let divisor = TONE_GRID_DIVISOR.max(1);
    let stride = f64::from(divisor);
    let (left, top, right, bottom) = bounds;
    let world_width = (right.saturating_sub(left) + 1).div_ceil(divisor).max(1);
    let world_height = (bottom.saturating_sub(top) + 1).div_ceil(divisor).max(1);
    let mut low = Rgb32FImage::new(world_width, world_height);
    let mut validity = GrayImage::new(world_width, world_height);
    let mut cell_mean = Rgb32FImage::new(world_width, world_height);
    let mut cell_coverage = vec![0.0f32; world_width as usize * world_height as usize];
    let empty = |low, validity, cell_mean, cell_coverage| ToneTile {
        station_index,
        owner_id,
        low,
        validity,
        world_origin: (f64::from(left), f64::from(top)),
        world_size: (
            f64::from(world_width) * stride,
            f64::from(world_height) * stride,
        ),
        world_stride: stride,
        cell_mean,
        cell_coverage,
    };
    let (tile_width, tile_height) = tile.dimensions();
    if tile_width == 0 || tile_height == 0 {
        return empty(low, validity, cell_mean, cell_coverage);
    }

    // Covered sums, covered counts and cell sizes on the tile's own grid.
    let grid_width = tile_width.div_ceil(divisor) as usize;
    let grid_height = tile_height.div_ceil(divisor) as usize;
    let rows = (0..grid_height)
        .into_par_iter()
        .map(|cell_y| {
            let mut row = vec![([0.0f64; 3], 0u32, 0u32); grid_width];
            let y_start = cell_y as u32 * divisor;
            for y in y_start..(y_start + divisor).min(tile_height) {
                for x in 0..tile_width {
                    let cell = &mut row[(x / divisor) as usize];
                    cell.2 += 1;
                    let pixel = tile.get_pixel(x, y).0;
                    if pixel[0].max(pixel[1]).max(pixel[2]) > 1e-6 {
                        for (sum, value) in cell.0.iter_mut().zip(pixel) {
                            *sum += f64::from(value);
                        }
                        cell.1 += 1;
                    }
                }
            }
            row
        })
        .collect::<Vec<_>>();
    let cells = rows.into_iter().flatten().collect::<Vec<_>>();
    let weights = cells
        .iter()
        .map(|&(_, covered, total)| [f64::from(covered) / f64::from(total.max(1))])
        .collect::<Vec<_>>();
    let weighted = cells
        .iter()
        .zip(&weights)
        .map(|(&(sum, covered, _), weight)| {
            std::array::from_fn(|channel| {
                if covered == 0 {
                    0.0
                } else {
                    sum[channel] / f64::from(covered) * weight[0]
                }
            })
        })
        .collect::<Vec<[f64; 3]>>();
    let sigma = f64::from(TONE_LOW_FREQUENCY_SIGMA);
    let blurred = gaussian_cells(&weighted, grid_width, grid_height, sigma);
    let support = gaussian_cells(&weights, grid_width, grid_height, sigma);
    let field = |cell: usize| -> Option<[f64; 3]> {
        (support[cell][0] > 1e-9)
            .then(|| std::array::from_fn(|channel| blurred[cell][channel] / support[cell][0]))
    };

    for world_y in 0..world_height {
        for world_x in 0..world_width {
            let x = f64::from(left) + f64::from(world_x) * stride + (stride - 1.0) * 0.5;
            let y = f64::from(top) + f64::from(world_y) * stride + (stride - 1.0) * 0.5;
            let Some((tile_x, tile_y)) = to_tile(x, y) else {
                continue;
            };
            if !(tile_x >= 0.0
                && tile_y >= 0.0
                && tile_x < f64::from(tile_width)
                && tile_y < f64::from(tile_height))
            {
                continue;
            }
            let cell_x = ((tile_x / stride) as usize).min(grid_width - 1);
            let cell_y = ((tile_y / stride) as usize).min(grid_height - 1);
            let cell = cell_y * grid_width + cell_x;
            let (cell_sum, covered, total) = cells[cell];
            if covered == 0 {
                continue;
            }
            cell_mean.put_pixel(
                world_x,
                world_y,
                Rgb(cell_sum.map(|channel| (channel / f64::from(covered)) as f32)),
            );
            cell_coverage[world_y as usize * world_width as usize + world_x as usize] =
                covered as f32 / total.max(1) as f32;
            // Bilinear over the covered-only field, renormalised over the
            // neighbours that carry any covered pixel.
            let u = (tile_x - (stride - 1.0) * 0.5) / stride;
            let v = (tile_y - (stride - 1.0) * 0.5) / stride;
            let (u0, v0) = (u.floor(), v.floor());
            let mut sum = [0.0f64; 3];
            let mut total_weight = 0.0f64;
            for (dy, weight_y) in [(0.0, 1.0 - (v - v0)), (1.0, v - v0)] {
                for (dx, weight_x) in [(0.0, 1.0 - (u - u0)), (1.0, u - u0)] {
                    let (nx, ny) = (u0 + dx, v0 + dy);
                    if nx < 0.0 || ny < 0.0 || nx >= grid_width as f64 || ny >= grid_height as f64 {
                        continue;
                    }
                    let neighbour = ny as usize * grid_width + nx as usize;
                    let Some(value) = field(neighbour) else {
                        continue;
                    };
                    let weight = weight_x * weight_y;
                    for channel in 0..3 {
                        sum[channel] += weight * value[channel];
                    }
                    total_weight += weight;
                }
            }
            let value = if total_weight > 1e-12 {
                sum.map(|channel| channel / total_weight)
            } else if let Some(value) = field(cell) {
                value
            } else {
                continue;
            };
            low.put_pixel(world_x, world_y, Rgb(value.map(|channel| channel as f32)));
            let level = if covered == total {
                TONE_CELL_SAMPLE
            } else {
                TONE_CELL_PARTIAL
            };
            validity.put_pixel(world_x, world_y, Luma([level]));
        }
    }
    empty(low, validity, cell_mean, cell_coverage)
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
    /// Applied (bounded) correction `corrected = gain · value + offset`.
    gain: [f32; 3],
    offset: [f32; 3],
    /// The joint solution before the per-tile bounds (需求 9.9).
    solved_gain: [f32; 3],
    solved_offset: [f32; 3],
    /// The solved gain or offset was bounded.
    clamped: bool,
    /// The tile is in at least one relation with evidence.
    supported: bool,
}

impl ToneTransform {
    const IDENTITY: Self = Self {
        gain: [1.0; 3],
        offset: [0.0; 3],
        solved_gain: [1.0; 3],
        solved_offset: [0.0; 3],
        clamped: false,
        supported: false,
    };
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
fn solve_tone_transforms(tile_count: usize, relations: &[ToneRelation]) -> Vec<ToneTransform> {
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

/// Apply all owner-level tone corrections after the immutable ownership plane
/// has been selected.  Each `ToneTile` is an analysis-grid field in world
/// coordinates, so the function samples at most one low-frequency pixel per
/// output pixel and never allocates a second full-resolution RGB canvas.
/// `relations` are the accepted Station_Relations as index pairs into `tiles`.
pub(crate) fn harmonize_after_ownership(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    tiles: &[ToneTile],
    relations: &[(usize, usize)],
    world_origin: (f64, f64),
) -> ToneReport {
    harmonize_after_ownership_with_evidence(panorama, owners, tiles, relations, world_origin, None)
}

/// Production-facing entry point used by the owner compositor. `width` is
/// explicit because the ownership plane is commonly held as a flat immutable
/// slice; the height is derived from the output image. `relations` are the
/// accepted Station_Relations as index pairs into `tiles`: only their
/// overlaps are tone evidence (需求 9.1).
pub(crate) fn harmonize_tone_tiles(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    width: u32,
    tiles: &[ToneTile],
    evidence: &GrayImage,
    relations: &[(usize, usize)],
) -> ToneReport {
    if width != panorama.width() {
        let report = ToneReport {
            status: ToneStatus::Identity,
            ..ToneReport::default()
        };
        record_run_report(report.clone());
        return report;
    }
    harmonize_after_ownership_with_evidence(
        panorama,
        owners,
        tiles,
        relations,
        (0.0, 0.0),
        Some(evidence),
    )
}

fn harmonize_after_ownership_with_evidence(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    tiles: &[ToneTile],
    relations: &[(usize, usize)],
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
    // One relation per accepted Station_Relation, measured on its doubly
    // covered overlap only (需求 9.1). Tiles that are merely placed next to
    // each other without an accepted relation contribute nothing.
    let mut pairs = relations
        .iter()
        .filter(|&&(left, right)| left != right && left < tiles.len() && right < tiles.len())
        .map(|&(left, right)| (left.min(right), left.max(right)))
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs.dedup();
    let mut solved_relations = Vec::with_capacity(pairs.len());
    let mut sample_support = vec![0usize; tiles.len()];
    for (first, second) in pairs {
        let samples = PairField::new(&tiles[first], &tiles[second])
            .map(|field| field.samples())
            .unwrap_or_default();
        let retained = consistent_samples(&samples);
        if retained.len() < TONE_MIN_SAMPLES {
            record_insufficient_samples(retained.len());
            report
                .pairs_without_evidence
                .push(TonePairWithoutEvidenceRecord {
                    left: tiles[first].station_index,
                    right: tiles[second].station_index,
                    retained_samples: retained.len() as u64,
                });
            continue;
        }
        sample_support[first] = sample_support[first].max(retained.len());
        sample_support[second] = sample_support[second].max(retained.len());
        solved_relations.push(ToneRelation::new(first, second, retained));
    }

    // One joint solve over every relation; each transform is bounded once, and
    // the pixels, the boundary re-check and the report read the same bounded
    // values while the report keeps the unbounded solution (需求 9.3, 9.9).
    let transforms = solve_tone_transforms(tiles.len(), &solved_relations);
    for (index, tile) in tiles.iter().enumerate() {
        let transform = &transforms[index];
        if transform.clamped {
            degradation::record_run_degradation(
                degradation::TONE_GAIN_CLAMPED,
                serde_json::json!({
                    "stage": "tone_harmonizer",
                    "station_index": tile.station_index,
                    "solved_gain": transform.solved_gain,
                    "gain": transform.gain,
                    "solved_offset": transform.solved_offset,
                    "offset": transform.offset,
                }),
            );
        }
        report.tiles.push(ToneTileRecord {
            station_index: tile.station_index,
            gain: transform.gain.map(f64::from),
            offset: transform.offset.map(f64::from),
            samples: sample_support[index] as u64,
            clamped: transform.clamped,
            solved_gain: transform.solved_gain.map(f64::from),
            solved_offset: transform.solved_offset.map(f64::from),
        });
    }

    // A tile owns its own high-frequency residual. The owner's low field is
    // expanded bilinearly from the analysis grid (design: "再双线性放大") and
    // only that band is corrected: `gain · low + offset + (owner − low)`.
    // Pixels without a correction, without evidence or without any field cell
    // of their owner keep their exact value.
    let output_width = panorama.width();
    let output_height = panorama.height();
    let mut tile_of_owner =
        vec![None; tiles.iter().map(|tile| tile.owner_id).max().unwrap_or(0) as usize + 1];
    for (index, tile) in tiles.iter().enumerate() {
        tile_of_owner[tile.owner_id as usize].get_or_insert(index);
    }
    let corrects = |tile_index: usize| {
        let transform = &transforms[tile_index];
        transform.supported
            && (transform
                .gain
                .iter()
                .any(|value| (*value - 1.0).abs() > f32::EPSILON)
                || transform
                    .offset
                    .iter()
                    .any(|value| value.abs() > f32::EPSILON))
    };
    panorama
        .as_mut()
        .par_chunks_mut(output_width as usize * 3)
        .enumerate()
        .for_each(|(output_y, row)| {
            for (output_x, pixel) in row.chunks_exact_mut(3).enumerate() {
                let owner_id = owners[output_y * output_width as usize + output_x];
                let Some(tile_index) = tile_of_owner.get(owner_id as usize).copied().flatten()
                else {
                    continue;
                };
                if !corrects(tile_index) {
                    continue;
                }
                if let Some(evidence) = evidence
                    && (output_x as u32 >= evidence.width()
                        || output_y as u32 >= evidence.height()
                        || evidence.get_pixel(output_x as u32, output_y as u32)[0] == 0)
                {
                    continue;
                }
                let Some(low) = sample_tile_bilinear(
                    &tiles[tile_index],
                    world_origin.0 + output_x as f64,
                    world_origin.1 + output_y as f64,
                ) else {
                    continue;
                };
                let transform = &transforms[tile_index];
                for channel in 0..3 {
                    pixel[channel] = (pixel[channel]
                        + (transform.gain[channel] - 1.0) * low[channel]
                        + transform.offset[channel])
                        .clamp(0.0, 1.0);
                }
            }
        });

    let correct = |tile_index: usize, low: [f32; 3]| -> [f32; 3] {
        let transform = &transforms[tile_index];
        std::array::from_fn(|channel| {
            transform.gain[channel] * low[channel] + transform.offset[channel]
        })
    };
    // Tile pair and boundary window → corrected low sums of both tiles. Each
    // adjacent pair is read through its common-support fields, so a band next
    // to either tile's coverage edge compares the same content window.
    let mut pair_fields = std::collections::BTreeMap::<(usize, usize), Option<PairField>>::new();
    let mut boundary_windows =
        std::collections::BTreeMap::<(usize, usize, u32, u32), ([f64; 3], [f64; 3], u64)>::new();
    for y in 0..output_height {
        for x in 0..output_width {
            let index = y as usize * output_width as usize + x as usize;
            let Some(left_tile) = tile_of_owner.get(owners[index] as usize).copied().flatten()
            else {
                continue;
            };
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= output_width || ny >= output_height {
                    continue;
                }
                let right_index = ny as usize * output_width as usize + nx as usize;
                let Some(right_tile) = tile_of_owner
                    .get(owners[right_index] as usize)
                    .copied()
                    .flatten()
                else {
                    continue;
                };
                if right_tile == left_tile {
                    continue;
                }
                // The 16px bands on both sides of this boundary pixel, along
                // its normal. Both owners' corrected low fields are read at
                // the same positions, so the window means differ by the tone
                // step at the boundary and not by the artwork's own gradient
                // across a 32px strip (需求 9.8).
                let (first, second) = (left_tile.min(right_tile), left_tile.max(right_tile));
                let Some(field) = pair_fields
                    .entry((first, second))
                    .or_insert_with(|| PairField::new(&tiles[first], &tiles[second]))
                    .as_ref()
                else {
                    continue;
                };
                let window = (
                    first,
                    second,
                    x / TONE_BOUNDARY_WINDOW_PX,
                    y / TONE_BOUNDARY_WINDOW_PX,
                );
                for distance in 0..TONE_BOUNDARY_BAND_PX {
                    let near = if nx > x {
                        (x.checked_sub(distance), Some(y))
                    } else {
                        (Some(x), y.checked_sub(distance))
                    };
                    let far = if nx > x {
                        (
                            Some(nx + distance).filter(|&px| px < output_width),
                            Some(ny),
                        )
                    } else {
                        (
                            Some(nx),
                            Some(ny + distance).filter(|&py| py < output_height),
                        )
                    };
                    for position in [near, far] {
                        let (Some(px), Some(py)) = position else {
                            continue;
                        };
                        let Some((first_low, second_low)) = field.at(
                            world_origin.0 + f64::from(px),
                            world_origin.1 + f64::from(py),
                        ) else {
                            continue;
                        };
                        let (first_low, second_low) =
                            (correct(first, first_low), correct(second, second_low));
                        let entry = boundary_windows.entry(window).or_insert((
                            [0.0f64; 3],
                            [0.0f64; 3],
                            0u64,
                        ));
                        for channel in 0..3 {
                            entry.0[channel] += f64::from(first_low[channel]);
                            entry.1[channel] += f64::from(second_low[channel]);
                        }
                        entry.2 += 1;
                    }
                }
            }
        }
    }
    for ((first, second, window_x, window_y), (first_sum, second_sum, count)) in boundary_windows {
        let mean = |sum: [f64; 3]| sum.map(|channel| (channel / count as f64) as f32);
        let delta = delta_e00_rgb(mean(first_sum), mean(second_sum));
        report.boundary_delta_e.max = report.boundary_delta_e.max.max(delta);
        if delta <= TONE_BOUNDARY_DELTA_E_THRESHOLD {
            continue;
        }
        let world = super::report::WorldRect {
            left: world_origin.0 + f64::from(window_x * TONE_BOUNDARY_WINDOW_PX),
            top: world_origin.1 + f64::from(window_y * TONE_BOUNDARY_WINDOW_PX),
            width: f64::from(TONE_BOUNDARY_WINDOW_PX),
            height: f64::from(TONE_BOUNDARY_WINDOW_PX),
        };
        if report.boundary_delta_e.violations.len() < 4096 {
            report
                .boundary_delta_e
                .violations
                .push(ToneBoundaryViolationRecord {
                    left: tiles[first].station_index,
                    right: tiles[second].station_index,
                    delta_e00: delta,
                    world,
                });
        }
        degradation::record_run_degradation(
            degradation::TONE_BOUNDARY_DELTA_E_EXCEEDED,
            serde_json::json!({
                "left": tiles[first].station_index,
                "right": tiles[second].station_index,
                "delta_e00": delta,
                "x": world.left,
                "y": world.top,
                "window_px": TONE_BOUNDARY_WINDOW_PX,
                "samples": count,
            }),
        );
    }
    report.status = if solved_relations.is_empty() {
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

/// The tile's field at a world position, bilinear between the analysis-cell
/// centres the field was built at and renormalised over the neighbouring cells
/// that carry a field (validity ≠ 0). `None` where no neighbour has one.
fn sample_tile_bilinear(tile: &ToneTile, world_x: f64, world_y: f64) -> Option<[f32; 3]> {
    let stride = tile.world_stride;
    if !stride.is_finite() || stride <= 0.0 {
        return None;
    }
    let (width, height) = tile.low.dimensions();
    let u = (world_x - tile.world_origin.0 - (stride - 1.0) * 0.5) / stride;
    let v = (world_y - tile.world_origin.1 - (stride - 1.0) * 0.5) / stride;
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
            if x >= tile.validity.width()
                || y >= tile.validity.height()
                || tile.validity.get_pixel(x, y)[0] == 0
            {
                continue;
            }
            let weight = weight_x * weight_y;
            let value = tile.low.get_pixel(x, y).0;
            for channel in 0..3 {
                sum[channel] += weight * f64::from(value[channel]);
            }
            total += weight;
        }
    }
    (total > 1e-9).then(|| sum.map(|channel| (channel / total) as f32))
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

fn median_f64(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<_> = values.filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
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
    fn pair_solve_rejects_outliers_and_keeps_the_median_gain() {
        let mut samples = Vec::with_capacity(TONE_MIN_SAMPLES + 64);
        for index in 0..TONE_MIN_SAMPLES + 64 {
            let source = 0.1 + (index % 31) as f32 / 40.0;
            samples.push(ToneSample {
                owner: [source * 1.1, source * 0.95, source * 1.05],
                source: [source; 3],
            });
        }
        for sample in &mut samples[..64] {
            sample.owner = [0.95; 3];
        }
        let solve = solve_tone_pair(&samples);
        assert_eq!(solve.status, ToneSolveStatus::Applied);
        for (channel, expected) in [1.1f32, 0.95, 1.05].into_iter().enumerate() {
            assert!((solve.gain[channel] - expected).abs() < 1e-3, "{solve:?}");
            assert!(
                (solve.solved_gain[channel] - expected).abs() < 1e-3,
                "{solve:?}"
            );
            assert!(solve.offset[channel].abs() < 1e-3, "{solve:?}");
        }
        assert!(!solve.gain_clamped);
        // Every consistent sample is kept; the painted-over ones are not.
        assert_eq!(solve.retained_samples, TONE_MIN_SAMPLES);
    }

    #[test]
    fn pair_solve_is_not_flattened_where_the_fields_disagree_by_position() {
        // Two exposures of the same content, each with its own ±6% low
        // frequency deviation (vignetting, a moved highlight). The true
        // relation is identity; a least-squares affine fit of one field on the
        // other attenuates its slope to ≈0.96 and compensates with an offset,
        // the station seam this solve used to create.
        let samples = (0..4096)
            .map(|index| {
                let content = 0.25 + 0.2 * ((index * 37) % 101) as f32 / 100.0;
                let owner = 1.0 + 0.06 * ((index as f32) * 0.71).sin();
                let source = 1.0 + 0.06 * ((index as f32) * 1.37 + 0.4).cos();
                ToneSample {
                    owner: [content * owner; 3],
                    source: [content * source; 3],
                }
            })
            .collect::<Vec<_>>();
        let solve = solve_tone_pair(&samples);
        assert_eq!(solve.status, ToneSolveStatus::Applied);
        for channel in 0..3 {
            assert!((solve.gain[channel] - 1.0).abs() < 0.01, "{solve:?}");
            assert!(solve.offset[channel].abs() < 0.005, "{solve:?}");
        }
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

    fn textured(x: f64, y: f64) -> f32 {
        (0.45 + 0.12 * (x * 0.013 + y * 0.007).sin() + 0.08 * (x * 0.004 - y * 0.011).cos()) as f32
    }

    /// A tile whose pixel `(x, y)` shows world `(x + origin_x, y)`, scaled by
    /// `gain`, with the columns `uncovered` left as transparent zeros.
    fn shifted_tile(
        width: u32,
        height: u32,
        origin_x: f64,
        gain: f32,
        uncovered: std::ops::Range<u32>,
    ) -> Rgb32FImage {
        Rgb32FImage::from_fn(width, height, |x, y| {
            if uncovered.contains(&x) {
                Rgb([0.0; 3])
            } else {
                Rgb([gain * textured(f64::from(x) + origin_x, f64::from(y)); 3])
            }
        })
    }

    #[test]
    fn world_aligned_tone_field_reads_covered_pixels_only() {
        // Left 64 columns uncovered; the covered part is a constant 0.5.
        let tile =
            Rgb32FImage::from_fn(
                256,
                128,
                |x, _| {
                    if x < 64 { Rgb([0.0; 3]) } else { Rgb([0.5; 3]) }
                },
            );
        let field = world_aligned_tone_tile(3, 7, &tile, (100, 40, 355, 167), |x, y| {
            Some((x - 100.0, y - 40.0))
        });
        assert_eq!((field.low.width(), field.low.height()), (16, 8));
        assert_eq!(field.world_origin, (100.0, 40.0));
        for y in 0..8 {
            for x in 0..16 {
                let level = field.validity.get_pixel(x, y)[0];
                if x < 4 {
                    assert_eq!(level, 0, "uncovered cell ({x}, {y})");
                } else {
                    assert_eq!(level, TONE_CELL_SAMPLE, "covered cell ({x}, {y})");
                    // Next to the uncovered edge the field is not darkened.
                    for channel in field.low.get_pixel(x, y).0 {
                        assert!((channel - 0.5).abs() < 1e-5, "({x}, {y}) {channel}");
                    }
                }
            }
        }
        // A cell straddling the coverage edge is corrected but no evidence.
        let straddling =
            Rgb32FImage::from_fn(
                64,
                16,
                |x, _| {
                    if x < 40 { Rgb([0.0; 3]) } else { Rgb([0.5; 3]) }
                },
            );
        let field = world_aligned_tone_tile(0, 1, &straddling, (0, 0, 63, 15), |x, y| Some((x, y)));
        let levels = (0..4)
            .map(|x| field.validity.get_pixel(x, 0)[0])
            .collect::<Vec<_>>();
        assert_eq!(levels, vec![0, 0, TONE_CELL_PARTIAL, TONE_CELL_SAMPLE]);
    }

    #[test]
    fn world_aligned_tone_field_follows_the_world_to_tile_map() {
        // The tile is drawn rotated by 90°: world (x, y) shows tile (y, 255 - x).
        let tile = Rgb32FImage::from_fn(256, 256, |x, y| {
            Rgb([textured(f64::from(x), f64::from(y)); 3])
        });
        let field =
            world_aligned_tone_tile(0, 1, &tile, (0, 0, 255, 255), |x, y| Some((y, 255.0 - x)));
        let axis = world_aligned_tone_tile(0, 1, &tile, (0, 0, 255, 255), |x, y| Some((x, y)));
        // World cell (cx, cy) of the rotated field is axis cell (cy, 15 - cx).
        for (cx, cy) in [(2u32, 3u32), (7, 12), (11, 5)] {
            let rotated = field.low.get_pixel(cx, cy)[0];
            let expected = axis.low.get_pixel(cy, 15 - cx)[0];
            assert!(
                (rotated - expected).abs() < 2e-3,
                "({cx}, {cy}) {rotated} vs {expected}"
            );
        }
    }

    /// Samples of one overlap where `first` shows the content at `first_gain`
    /// and `second` at `second_gain`.
    fn relation_between(
        first: usize,
        second: usize,
        first_gain: f32,
        second_gain: f32,
    ) -> ToneRelation {
        let samples = (0..2048)
            .map(|index| {
                let content = 0.2 + 0.5 * ((index * 53) % 97) as f32 / 96.0;
                ToneSample {
                    owner: [content * first_gain; 3],
                    source: [content * second_gain; 3],
                }
            })
            .collect::<Vec<_>>();
        ToneRelation::new(first, second, consistent_samples(&samples))
    }

    #[test]
    fn joint_solve_equalises_a_loop_without_an_anchor_or_order() {
        // Three stations exposed at 1.0, 0.9 and 1.1 with all three overlaps
        // accepted: every corrected tile shows the content at one level, the
        // component keeps zero mean log gain, and the relation order is
        // irrelevant.
        let exposure = [1.0f32, 0.9, 1.1];
        let relations = [(0, 1), (1, 2), (0, 2)].map(|(first, second)| {
            relation_between(first, second, exposure[first], exposure[second])
        });
        let transforms = solve_tone_transforms(3, &relations);
        let level = |tile: usize| transforms[tile].gain[0] * exposure[tile];
        let gains = transforms
            .iter()
            .map(|transform| transform.gain)
            .collect::<Vec<_>>();
        for tile in 1..3 {
            assert!((level(tile) - level(0)).abs() < 1e-4, "{gains:?}");
        }
        let mean_log_gain = transforms
            .iter()
            .map(|transform| f64::from(transform.gain[0]).ln())
            .sum::<f64>()
            / 3.0;
        assert!(mean_log_gain.abs() < 1e-6);
        assert!(
            transforms
                .iter()
                .all(|transform| transform.supported && !transform.clamped)
        );
        assert!(
            transforms
                .iter()
                .all(|transform| transform.offset[0].abs() < 1e-4)
        );

        let reversed = [(0, 2), (1, 2), (0, 1)].map(|(first, second)| {
            relation_between(first, second, exposure[first], exposure[second])
        });
        let reordered = solve_tone_transforms(3, &reversed);
        for (left, right) in transforms.iter().zip(&reordered) {
            assert!((left.gain[0] - right.gain[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn joint_solve_bounds_each_tile_and_keeps_the_solved_values() {
        // A 1.6× step exceeds the 1.25 gain bound on one side.
        let relations = [relation_between(0, 1, 1.0, 0.4)];
        let transforms = solve_tone_transforms(3, &relations);
        let corrected = &transforms[1];
        assert!(corrected.clamped);
        assert_eq!(corrected.gain, [TONE_MAX_GAIN; 3]);
        assert!((corrected.solved_gain[0] - (2.5f32).sqrt()).abs() < 1e-3);
        assert!(
            corrected
                .offset
                .iter()
                .all(|value| value.abs() <= TONE_MAX_ABS_OFFSET)
        );
        // A tile without an accepted relation stays exact identity.
        assert!(!transforms[2].supported);
        assert_eq!(transforms[2].gain, [1.0; 3]);
        assert_eq!(transforms[2].offset, [0.0; 3]);
    }

    #[test]
    fn harmonized_tiles_match_a_darker_neighbour_without_uncovered_bias() {
        // Tile 0 covers world x ∈ [0, 1024); tile 1 covers [512, 1408) and is
        // 10% darker. Its first 128 columns (world 384..512, inside tile 0)
        // are uncovered zeros that must not bias the pair relation.
        let (width, height) = (1_408u32, 640u32);
        let left = shifted_tile(1_024, height, 0.0, 1.0, 0..0);
        let right = shifted_tile(1_024, height, 384.0, 0.9, 0..128);
        let tiles = [
            world_aligned_tone_tile(10, 1, &left, (0, 0, 1_023, height - 1), |x, y| Some((x, y))),
            world_aligned_tone_tile(11, 2, &right, (384, 0, width - 1, height - 1), |x, y| {
                Some((x - 384.0, y))
            }),
        ];
        let owners = (0..height)
            .flat_map(|_| (0..width).map(|x| if x < 768 { 1u16 } else { 2 }))
            .collect::<Vec<_>>();
        let mut panorama = Rgb32FImage::from_fn(width, height, |x, y| {
            if x < 768 {
                *left.get_pixel(x, y)
            } else {
                *right.get_pixel(x - 384, y)
            }
        });
        let evidence = GrayImage::from_pixel(width, height, Luma([255]));
        let report =
            harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &[(0, 1)]);

        // Zero mean log gain: the darker tile is raised and the brighter one
        // lowered by the same factor, √(1/0.9) each.
        let (brighter, darker) = (&report.tiles[0], &report.tiles[1]);
        assert_eq!((brighter.station_index, darker.station_index), (10, 11));
        for channel in 0..3 {
            assert!(!darker.clamped && !brighter.clamped, "{report:?}");
            let step = darker.gain[channel] / brighter.gain[channel];
            assert!((step - 1.0 / 0.9).abs() < 0.01, "{report:?}");
            assert!((darker.gain[channel] * brighter.gain[channel] - 1.0).abs() < 1e-3);
            assert!(darker.offset[channel].abs() < 0.005, "{report:?}");
        }
        assert!(report.pairs_without_evidence.is_empty());
        assert_eq!(report.status, ToneStatus::Applied);
        // The corrected right half continues the corrected left half's tone.
        let level = brighter.gain[0] as f32;
        for (x, y) in [(100u32, 100u32), (700, 300), (900, 300), (1_200, 450)] {
            let expected = level * textured(f64::from(x), f64::from(y));
            let value = panorama.get_pixel(x, y)[0];
            assert!(
                (value - expected).abs() < 0.01,
                "({x}, {y}) {value} vs {expected}"
            );
        }
        assert!(report.boundary_delta_e.max <= 1.5, "{report:?}");
    }

    #[test]
    fn only_accepted_station_relations_are_tone_pairs() {
        // Two overlapping tiles 20% apart, but no accepted Station_Relation
        // between them: their overlap is no tone evidence (需求 9.1).
        let (width, height) = (1_536u32, 640u32);
        let left = shifted_tile(1_024, height, 0.0, 1.0, 0..0);
        let right = shifted_tile(1_024, height, 512.0, 0.8, 0..0);
        let tiles = [
            world_aligned_tone_tile(0, 1, &left, (0, 0, 1_023, height - 1), |x, y| Some((x, y))),
            world_aligned_tone_tile(1, 2, &right, (512, 0, width - 1, height - 1), |x, y| {
                Some((x - 512.0, y))
            }),
        ];
        let owners = (0..height)
            .flat_map(|_| (0..width).map(|x| if x < 768 { 1u16 } else { 2 }))
            .collect::<Vec<_>>();
        let mut panorama = Rgb32FImage::from_fn(width, height, |x, y| {
            if x < 768 {
                *left.get_pixel(x, y)
            } else {
                *right.get_pixel(x - 512, y)
            }
        });
        let before = panorama.clone();
        let evidence = GrayImage::from_pixel(width, height, Luma([255]));
        let report = harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &[]);
        assert_eq!(report.status, ToneStatus::Identity);
        assert!(report.pairs_without_evidence.is_empty());
        assert!(report.tiles.iter().all(|tile| tile.gain == [1.0; 3]));
        assert_eq!(panorama, before);

        // The same pair as an accepted relation is corrected.
        let report =
            harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &[(1, 0)]);
        assert_eq!(report.status, ToneStatus::Applied, "{report:?}");
        assert!((report.tiles[1].gain[0] / report.tiles[0].gain[0] - 1.25).abs() < 0.01);
    }

    #[test]
    fn an_accepted_relation_without_a_covered_overlap_is_recorded_without_evidence() {
        let left = shifted_tile(256, 256, 0.0, 1.0, 0..0);
        // The bounding boxes intersect on world x ∈ [192, 256), but tile 1 is
        // uncovered there: no doubly covered cell and so no retained sample.
        let right = shifted_tile(256, 256, 192.0, 0.8, 0..64);
        let tiles = [
            world_aligned_tone_tile(0, 1, &left, (0, 0, 255, 255), |x, y| Some((x, y))),
            world_aligned_tone_tile(1, 2, &right, (192, 0, 447, 255), |x, y| {
                Some((x - 192.0, y))
            }),
        ];
        let owners = (0..256u32)
            .flat_map(|_| (0..448u32).map(|x| if x < 256 { 1u16 } else { 2 }))
            .collect::<Vec<_>>();
        let mut panorama = Rgb32FImage::new(448, 256);
        let before = panorama.clone();
        let evidence = GrayImage::from_pixel(448, 256, Luma([255]));
        let report =
            harmonize_tone_tiles(&mut panorama, &owners, 448, &tiles, &evidence, &[(0, 1)]);
        assert_eq!(report.status, ToneStatus::Identity);
        assert_eq!(
            report.pairs_without_evidence,
            vec![TonePairWithoutEvidenceRecord {
                left: 0,
                right: 1,
                retained_samples: 0,
            }]
        );
        assert!(
            report
                .tiles
                .iter()
                .all(|tile| tile.gain == [1.0; 3] && !tile.clamped)
        );
        assert_eq!(panorama, before);
    }

    #[test]
    fn applied_correction_is_continuous_across_analysis_cells() {
        // A 30% step on a strong horizontal gradient. The correction is
        // `(gain − 1) · low + offset`; read from the analysis grid by nearest
        // cell it would step by ≈1e-3 every 16 px, bilinear it changes per
        // pixel only by the gradient's own share (≈7e-5).
        let (width, height) = (1_536u32, 640u32);
        let ramp = |x: f64| (0.1 + 0.8 * x / 1_535.0) as f32;
        let left = Rgb32FImage::from_fn(1_024, height, |x, _| Rgb([ramp(f64::from(x)); 3]));
        let right = Rgb32FImage::from_fn(1_024, height, |x, _| {
            Rgb([0.7 * ramp(f64::from(x) + 512.0); 3])
        });
        let tiles = [
            world_aligned_tone_tile(0, 1, &left, (0, 0, 1_023, height - 1), |x, y| Some((x, y))),
            world_aligned_tone_tile(1, 2, &right, (512, 0, width - 1, height - 1), |x, y| {
                Some((x - 512.0, y))
            }),
        ];
        let owners = (0..height)
            .flat_map(|_| (0..width).map(|x| if x < 768 { 1u16 } else { 2 }))
            .collect::<Vec<_>>();
        let mut panorama = Rgb32FImage::from_fn(width, height, |x, y| {
            if x < 768 {
                *left.get_pixel(x, y)
            } else {
                *right.get_pixel(x - 512, y)
            }
        });
        let before = panorama.clone();
        let evidence = GrayImage::from_pixel(width, height, Luma([255]));
        let report =
            harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &[(0, 1)]);
        assert_eq!(report.status, ToneStatus::Applied, "{report:?}");
        assert!(report.tiles.iter().all(|tile| !tile.clamped), "{report:?}");
        let y = height / 2;
        let delta = |x: u32| panorama.get_pixel(x, y)[0] - before.get_pixel(x, y)[0];
        for x in 0..width - 1 {
            // The owner boundary is where the correction is meant to change.
            if x + 1 == 768 {
                continue;
            }
            let step = (delta(x + 1) - delta(x)).abs();
            assert!(step < 3e-4, "correction steps by {step} at x = {x}");
        }
        // Across the owner boundary the corrected tiles meet.
        let seam = panorama.get_pixel(768, y)[0] - panorama.get_pixel(767, y)[0];
        assert!(seam.abs() < 3e-3, "seam step {seam}");
    }

    /// Colour scene with broad tone variation and fine texture.
    fn scan_scene(x: f64, y: f64) -> [f32; 3] {
        let broad = textured(x, y);
        let fine: f32 = if ((x as i64 / 3) + (y as i64 / 3)) % 2 == 0 {
            0.03
        } else {
            -0.03
        };
        [
            broad + fine,
            0.9 * broad + 0.03 + fine,
            0.75 * broad + 0.08 + fine,
        ]
    }

    // 检查点 12: on a synthetic scan grid with arbitrary per-station RGB
    // exposure differences, every Owner_Region boundary band meets
    // Delta_E00 ≤ 1.5 after harmonisation (需求 9.8) and the correction
    // carries no high frequency, i.e. every output pixel keeps its owner's
    // high-frequency residual (需求 9.2).
    #[test]
    fn synthetic_scan_grid_with_exposure_steps_meets_the_boundary_bound() {
        let (columns, rows) = (3usize, 2usize);
        let (tile_width, tile_height, step) = (1_024u32, 1_024u32, 512u32);
        let width = step * (columns as u32 - 1) + tile_width;
        let height = step * (rows as u32 - 1) + tile_height;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for _case in 0..3 {
            let stations = columns * rows;
            let origins = (0..stations)
                .map(|station| {
                    (
                        (station % columns) as u32 * step,
                        (station / columns) as u32 * step,
                    )
                })
                .collect::<Vec<_>>();
            // Independent RGB exposures in [0.88, 1.12] per station.
            let exposures = (0..stations)
                .map(|_| std::array::from_fn::<f32, 3, _>(|_| (0.88 + 0.24 * next()) as f32))
                .collect::<Vec<_>>();
            let sources = (0..stations)
                .map(|station| {
                    let (ox, oy) = origins[station];
                    Rgb32FImage::from_fn(tile_width, tile_height, |x, y| {
                        let scene = scan_scene(f64::from(x + ox), f64::from(y + oy));
                        Rgb(std::array::from_fn(|channel| {
                            scene[channel] * exposures[station][channel]
                        }))
                    })
                })
                .collect::<Vec<_>>();
            let tiles = (0..stations)
                .map(|station| {
                    let (ox, oy) = origins[station];
                    world_aligned_tone_tile(
                        station,
                        station as u16 + 1,
                        &sources[station],
                        (ox, oy, ox + tile_width - 1, oy + tile_height - 1),
                        |x, y| Some((x - f64::from(ox), y - f64::from(oy))),
                    )
                })
                .collect::<Vec<_>>();
            // Hard ownership by the nearest station centre.
            let owner_at = |x: u32, y: u32| {
                (0..stations)
                    .filter(|&station| {
                        let (ox, oy) = origins[station];
                        (ox..ox + tile_width).contains(&x) && (oy..oy + tile_height).contains(&y)
                    })
                    .min_by_key(|&station| {
                        let (ox, oy) = origins[station];
                        let dx = i64::from(x) - i64::from(ox + tile_width / 2);
                        let dy = i64::from(y) - i64::from(oy + tile_height / 2);
                        (dx * dx + dy * dy, station)
                    })
                    .expect("the grid covers the canvas")
            };
            let owners = (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .map(|(x, y)| owner_at(x, y) as u16 + 1)
                .collect::<Vec<_>>();
            let mut panorama = Rgb32FImage::from_fn(width, height, |x, y| {
                let station = owners[(y * width + x) as usize] as usize - 1;
                let (ox, oy) = origins[station];
                *sources[station].get_pixel(x - ox, y - oy)
            });
            let before = panorama.clone();
            // Accepted relations between grid neighbours only.
            let relations = (0..stations)
                .flat_map(|station| {
                    let right = (station % columns + 1 < columns).then_some((station, station + 1));
                    let below =
                        (station + columns < stations).then_some((station, station + columns));
                    [right, below].into_iter().flatten()
                })
                .collect::<Vec<_>>();
            let evidence = GrayImage::from_pixel(width, height, Luma([255]));
            let report =
                harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &relations);
            assert_eq!(
                report.status,
                ToneStatus::Applied,
                "{exposures:?} {report:?}"
            );
            assert!(
                report.boundary_delta_e.max <= 1.5,
                "{exposures:?} {report:?}"
            );
            assert!(report.tiles.iter().all(|tile| !tile.clamped), "{report:?}");
            // The correction is a low-frequency field inside each Owner_Region:
            // neighbouring pixels of one owner change by the same amount up to
            // the field's own slope, so the fine checker survives exactly.
            let delta = |x: u32, y: u32, channel: usize| {
                panorama.get_pixel(x, y)[channel] - before.get_pixel(x, y)[channel]
            };
            for y in (0..height).step_by(7) {
                for x in 0..width - 1 {
                    let index = (y * width + x) as usize;
                    if owners[index] != owners[index + 1] {
                        continue;
                    }
                    for channel in 0..3 {
                        let step = (delta(x + 1, y, channel) - delta(x, y, channel)).abs();
                        assert!(step < 1e-3, "({x}, {y}) channel {channel} steps by {step}");
                    }
                }
            }
        }
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
