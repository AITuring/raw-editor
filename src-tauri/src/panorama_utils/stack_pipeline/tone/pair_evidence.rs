//! Pairwise tone evidence between neighbouring tone tiles.
//!
//! The shared-overlap PairField, consistent samples and the robust per-pair
//! log-gain solve and calibration.

use super::super::degradation;
use super::super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD;

#[cfg(test)]
use super::linear_luma;
use super::{
    TONE_LOW_FREQUENCY_SIGMA, TONE_MAX_ABS_OFFSET, TONE_MAX_GAIN, TONE_MAX_SAMPLE_LUMINANCE,
    TONE_MIN_GAIN, TONE_MIN_SAMPLE_LUMINANCE, TONE_MIN_SAMPLES, TONE_RELATION_SCATTER_SCALE,
    ToneTile, delta_e00_rgb, gaussian_cells, median, median_f64,
};

/// The low fields of two tiles over their common support only (需求 9.1).
///
/// Both are normalised convolutions of the same doubly covered cells, so they
/// see the same content window and differ by the tiles' tone and not by what
/// either tile covers alone. A tile's own field near its coverage edge
/// averages only its own side, which next to a content gradient separates it
/// from a neighbour whose field is still centred at that position.
pub(crate) struct PairField {
    pub(super) origin: (f64, f64),
    pub(super) stride: f64,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) first: Vec<[f32; 3]>,
    pub(super) second: Vec<[f32; 3]>,
    /// Both tiles cover the cell entirely and the common field exists there.
    pub(super) sample: Vec<bool>,
}

impl PairField {
    #[cfg(test)]
    pub(crate) fn diagnostic_grid(&self) -> serde_json::Value {
        let bounded_consensus = self
            .first
            .iter()
            .zip(&self.second)
            .map(|(&first, &second)| {
                let target: [f32; 3] = std::array::from_fn(|c| (first[c] + second[c]) * 0.5);
                let bounded = |original: [f32; 3]| {
                    let valid = |strength: f32| {
                        let corrected = std::array::from_fn(|c| {
                            original[c] + (target[c] - original[c]) * strength
                        });
                        let ratio = linear_luma(corrected) / linear_luma(original).max(1e-8);
                        (
                            (0.96..=1.04).contains(&ratio)
                                && delta_e00_rgb(original, corrected) <= 1.8,
                            corrected,
                        )
                    };
                    let (mut low, mut high) = (0.0f32, 1.0f32);
                    for _ in 0..20 {
                        let middle = (low + high) * 0.5;
                        if valid(middle).0 {
                            low = middle;
                        } else {
                            high = middle;
                        }
                    }
                    valid(low).1
                };
                delta_e00_rgb(bounded(first), bounded(second))
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "origin": self.origin, "stride": self.stride,
            "width": self.width, "height": self.height,
            "first": self.first, "second": self.second, "sample": self.sample,
            "delta_e00": self.first.iter().zip(&self.second)
                .map(|(&first, &second)| delta_e00_rgb(first, second)).collect::<Vec<_>>(),
            "bounded_consensus_delta_e00": bounded_consensus,
            "diagnostic_only": true
        })
    }

    /// Seam selection uses the same common-support low-frequency comparison
    /// as the final tone boundary check, without changing either source.
    pub(crate) fn seam_disagreement(&self, world_x: f64, world_y: f64) -> Option<f64> {
        let (first, second) = self.at(world_x, world_y)?;
        Some(
            (delta_e00_rgb(first, second) / TONE_BOUNDARY_DELTA_E_THRESHOLD)
                .powi(2)
                .min(64.0),
        )
    }

    pub(crate) fn new(first: &ToneTile, second: &ToneTile) -> Option<Self> {
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
            let cx =
                (x - tile.world_origin.0 - (tile.world_stride - 1.0) * 0.5) / tile.world_stride;
            let cy =
                (y - tile.world_origin.1 - (tile.world_stride - 1.0) * 0.5) / tile.world_stride;
            if cx < 0.0
                || cy < 0.0
                || cx >= f64::from(tile.cell_mean.width())
                || cy >= f64::from(tile.cell_mean.height())
            {
                return None;
            }
            let (x0, y0) = (cx.floor() as u32, cy.floor() as u32);
            let mut rgb = [0.0f64; 3];
            let mut coverage = 1.0f32;
            for (dy, wy) in [(0, 1.0 - cy.fract()), (1, cy.fract())] {
                for (dx, wx) in [(0, 1.0 - cx.fract()), (1, cx.fract())] {
                    let weight = wx * wy;
                    if weight <= 1e-12 {
                        continue;
                    }
                    let (px, py) = (x0 + dx, y0 + dy);
                    if px >= tile.cell_mean.width() || py >= tile.cell_mean.height() {
                        return None;
                    }
                    coverage = coverage.min(
                        tile.cell_coverage
                            [py as usize * tile.cell_mean.width() as usize + px as usize],
                    );
                    for c in 0..3 {
                        rgb[c] += f64::from(tile.cell_mean.get_pixel(px, py)[c]) * weight;
                    }
                }
            }
            Some((rgb.map(|v| v as f32), coverage))
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
    pub(crate) fn at(&self, world_x: f64, world_y: f64) -> Option<([f32; 3], [f32; 3])> {
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
    pub(crate) fn samples(&self) -> Vec<ToneSample> {
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

    /// Robust evidence for a spatial illumination model. Compare RGB log
    /// gains inside the same world window so a smooth vignetting field is not
    /// mistaken for a global photometric outlier.
    pub(crate) fn spatial_consistency(
        &self,
        window: f64,
    ) -> (
        std::collections::BTreeMap<(i64, i64), ToneConsistency>,
        usize,
    ) {
        let mut groups = std::collections::BTreeMap::<(i64, i64), Vec<ToneSample>>::new();
        for i in 0..self.sample.len() {
            if !self.sample[i] {
                continue;
            }
            let sample = ToneSample {
                owner: self.first[i],
                source: self.second[i],
            };
            let usable = |rgb: [f32; 3]| {
                (TONE_MIN_SAMPLE_LUMINANCE..=TONE_MAX_SAMPLE_LUMINANCE)
                    .contains(&(rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722))
            };
            if !usable(sample.owner) || !usable(sample.source) {
                continue;
            }
            let x =
                self.origin.0 + (i % self.width) as f64 * self.stride + (self.stride - 1.0) * 0.5;
            let y =
                self.origin.1 + (i / self.width) as f64 * self.stride + (self.stride - 1.0) * 0.5;
            groups
                .entry(((x / window).floor() as i64, (y / window).floor() as i64))
                .or_default()
                .push(sample);
        }
        let mut retained = 0;
        let models = groups
            .into_iter()
            .filter_map(|(key, samples)| {
                let model = ToneConsistency::new(&samples)?;
                retained += samples.iter().filter(|s| model.accepts(s)).count();
                Some((key, model))
            })
            .collect();
        (models, retained)
    }

    pub(crate) fn spatial_window_key(&self, x: f64, y: f64, window: f64) -> Option<(i64, i64)> {
        let column = ((x - self.origin.0) / self.stride).floor();
        let row = ((y - self.origin.1) / self.stride).floor();
        if column < 0.0 || row < 0.0 || column >= self.width as f64 || row >= self.height as f64 {
            return None;
        }
        // The statistics are grouped by the native common-field cell centre,
        // which need not share the global analysis lattice's alignment.
        let cx = self.origin.0 + column * self.stride + (self.stride - 1.0) * 0.5;
        let cy = self.origin.1 + row * self.stride + (self.stride - 1.0) * 0.5;
        Some(((cx / window).floor() as i64, (cy / window).floor() as i64))
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
pub(super) fn log_channel(value: f32) -> f64 {
    f64::from(value.max(1.0 / 65_535.0)).ln()
}

/// Per-channel `ln owner − ln source` of one sample: the log gain that maps
/// the source side of the sample onto its owner side.
pub(super) fn sample_log_gain(sample: &ToneSample) -> [f64; 3] {
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
pub(crate) fn consistent_samples(samples: &[ToneSample]) -> Vec<ToneSample> {
    let Some(consistency) = ToneConsistency::new(samples) else {
        return Vec::new();
    };
    samples
        .iter()
        .filter(|sample| consistency.accepts(sample))
        .copied()
        .collect()
}

pub(crate) struct ToneConsistency {
    pub(super) medians: [f64; 3],
    pub(super) threshold: f64,
}

impl ToneConsistency {
    pub(crate) fn new(samples: &[ToneSample]) -> Option<Self> {
        if samples.is_empty() {
            return None;
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
        Some(Self { medians, threshold })
    }

    pub(crate) fn accepts(&self, sample: &ToneSample) -> bool {
        let gain = sample_log_gain(sample);
        (0..3)
            .map(|c| (gain[c] - self.medians[c]).powi(2))
            .sum::<f64>()
            .sqrt()
            <= self.threshold
    }
}

/// One accepted pair's multiplicative tone relation. `first` is the owner side
/// and `second` the source side of every retained sample.
#[derive(Clone)]
pub(super) struct ToneRelation {
    pub(super) first: usize,
    pub(super) second: usize,
    /// Per-channel median of `ln first − ln second` over `samples`.
    pub(super) log_gain: [f64; 3],
    /// Median RMS distance of the sample log gains from `log_gain`.
    pub(super) scatter: f64,
    pub(super) samples: Vec<ToneSample>,
}

impl ToneRelation {
    pub(super) fn new(first: usize, second: usize, samples: Vec<ToneSample>) -> Self {
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
    pub(super) fn weight(&self) -> f64 {
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

pub(super) fn record_insufficient_samples(retained_samples: usize) {
    degradation::record_run_degradation(
        degradation::TONE_INSUFFICIENT_SAMPLES,
        serde_json::json!({
            "stage": "tone_harmonizer",
            "retained_samples": retained_samples,
            "minimum_samples": TONE_MIN_SAMPLES,
        }),
    );
}
