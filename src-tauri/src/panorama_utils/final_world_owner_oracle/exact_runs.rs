//! Exact per-pixel owner evidence aggregated into sparse runs.
//!
//! Runs keep every candidate's decision inputs and exact source sample
//! provenance; the statistics replay the current, data-minimum, single-source or
//! an arbitrary labeling.

use serde::Serialize;

/// Exact production decision inputs for one real Source_RAW at one final pixel.
/// Integer/bit fields make run equality independent of host floating-point
/// formatting. `station_cell` is the exact ownership cell reached after the
/// final-world, residual, source and source-to-station maps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(in crate::panorama_utils::stitching) struct ExactCandidateDecision {
    pub source: u64,
    pub station: u16,
    pub station_cell: u32,
    pub sharpness_bits: u32,
    pub winner_sharpness_bits: u32,
    pub shortfall: u8,
    pub disagreement: u8,
    pub data_cost_bits: u64,
    pub pairwise_cost_bits: u64,
    pub seam_boundary: bool,
}

impl ExactCandidateDecision {
    pub(super) fn data_cost(&self) -> f64 {
        f64::from_bits(self.data_cost_bits)
    }

    pub(super) fn pairwise_cost(&self) -> f64 {
        f64::from_bits(self.pairwise_cost_bits)
    }
}

/// Exact source sample provenance retained separately from the decision key.
/// A run may share one decision while its projective source coordinates vary;
/// every represented final pixel still retains its own source sample bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(in crate::panorama_utils::stitching) struct ExactSourceSample {
    pub source: u64,
    pub x_bits: u64,
    pub y_bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::panorama_utils::stitching) struct ExactPixelEvidence {
    pub x: u32,
    pub y: u32,
    pub covered: bool,
    pub textured: bool,
    pub production_source: u64,
    pub production_station: u16,
    pub production_shortfall: u8,
    pub production_disagreement: u8,
    pub candidates: Vec<ExactCandidateDecision>,
    pub source_samples: Vec<ExactSourceSample>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(in crate::panorama_utils::stitching) struct ExactRun {
    pub y: u32,
    pub x_start: u32,
    pub len: u32,
    pub covered: bool,
    pub textured: bool,
    pub production_source: u64,
    pub production_station: u16,
    pub production_shortfall: u8,
    pub production_disagreement: u8,
    pub candidates: Vec<ExactCandidateDecision>,
    /// Row-major, then candidate order. Its length is `len * candidates.len()`.
    pub source_samples: Vec<ExactSourceSample>,
}

impl ExactRun {
    pub(super) fn same_decision(&self, pixel: &ExactPixelEvidence) -> bool {
        self.y == pixel.y
            && self.x_start.saturating_add(self.len) == pixel.x
            && self.covered == pixel.covered
            && self.textured == pixel.textured
            && self.production_source == pixel.production_source
            && self.production_station == pixel.production_station
            && self.production_shortfall == pixel.production_shortfall
            && self.production_disagreement == pixel.production_disagreement
            && self.candidates == pixel.candidates
    }
}

/// Build horizontal sparse runs without centre inference. Pixels merge only
/// after all production owner inputs and all station/source candidate
/// provenance compare equal; exact source coordinates remain attached per
/// represented pixel.
pub(in crate::panorama_utils::stitching) fn aggregate_exact_runs(
    width: u32,
    pixels: impl IntoIterator<Item = ExactPixelEvidence>,
) -> Result<Vec<ExactRun>, &'static str> {
    let mut runs = Vec::<ExactRun>::new();
    let mut previous_index = None;
    for mut pixel in pixels {
        if pixel.x >= width || pixel.candidates.len() != pixel.source_samples.len() {
            return Err("invalid exact owner pixel");
        }
        pixel
            .candidates
            .sort_by_key(|candidate| (candidate.station, candidate.source));
        pixel.source_samples.sort_by_key(|sample| sample.source);
        if !pixel
            .candidates
            .iter()
            .zip(&pixel.source_samples)
            .all(|(candidate, sample)| candidate.source == sample.source)
        {
            return Err("exact owner candidate provenance mismatch");
        }
        let index = u64::from(pixel.y)
            .checked_mul(u64::from(width))
            .and_then(|row| row.checked_add(u64::from(pixel.x)))
            .ok_or("invalid exact owner pixel")?;
        if previous_index.is_some_and(|previous| index <= previous) {
            return Err("exact owner pixels must be unique row-major samples");
        }
        previous_index = Some(index);
        if let Some(run) = runs.last_mut().filter(|run| run.same_decision(&pixel)) {
            run.len = run.len.saturating_add(1);
            run.source_samples.extend(pixel.source_samples);
            continue;
        }
        runs.push(ExactRun {
            y: pixel.y,
            x_start: pixel.x,
            len: 1,
            covered: pixel.covered,
            textured: pixel.textured,
            production_source: pixel.production_source,
            production_station: pixel.production_station,
            production_shortfall: pixel.production_shortfall,
            production_disagreement: pixel.production_disagreement,
            candidates: pixel.candidates,
            source_samples: pixel.source_samples,
        });
    }
    Ok(runs)
}

pub(in crate::panorama_utils::stitching) fn exact_run_stats_data_min(
    runs: &[ExactRun],
    shortfall_limit: u8,
) -> ExactOwnerStats {
    let mut result = ExactOwnerStats::default();
    for run in runs {
        if !run.covered {
            continue;
        }
        let selected = run.candidates.iter().min_by(|left, right| {
            left.data_cost()
                .total_cmp(&right.data_cost())
                .then_with(|| left.source.cmp(&right.source))
        });
        let Some(selected) = selected else {
            result.unresolved_pixels = result.unresolved_pixels.saturating_add(u64::from(run.len));
            continue;
        };
        result.covered_pixels = result.covered_pixels.saturating_add(u64::from(run.len));
        if run.textured {
            result.textured_pixels = result.textured_pixels.saturating_add(u64::from(run.len));
            if selected.shortfall > shortfall_limit {
                result.shortfall_pixels =
                    result.shortfall_pixels.saturating_add(u64::from(run.len));
            }
        }
        result.data_cost += selected.data_cost() * f64::from(run.len);
        if selected.seam_boundary {
            result.pairwise_cost += selected.pairwise_cost() * f64::from(run.len);
        }
    }
    result
}

pub(in crate::panorama_utils::stitching) fn exact_run_stats_for_source(
    runs: &[ExactRun],
    source: u64,
    shortfall_limit: u8,
) -> ExactOwnerStats {
    let mut result = ExactOwnerStats::default();
    for run in runs {
        if !run.covered {
            continue;
        }
        let selected = run
            .candidates
            .iter()
            .find(|candidate| candidate.source == source);
        let Some(selected) = selected else {
            result.unresolved_pixels = result.unresolved_pixels.saturating_add(u64::from(run.len));
            continue;
        };
        result.covered_pixels = result.covered_pixels.saturating_add(u64::from(run.len));
        if run.textured {
            result.textured_pixels = result.textured_pixels.saturating_add(u64::from(run.len));
            if selected.shortfall > shortfall_limit {
                result.shortfall_pixels =
                    result.shortfall_pixels.saturating_add(u64::from(run.len));
            }
        }
        result.data_cost += selected.data_cost() * f64::from(run.len);
        if selected.seam_boundary {
            result.pairwise_cost += selected.pairwise_cost() * f64::from(run.len);
        }
    }
    result
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(in crate::panorama_utils::stitching) struct ExactOwnerStats {
    pub covered_pixels: u64,
    pub textured_pixels: u64,
    pub shortfall_pixels: u64,
    pub unresolved_pixels: u64,
    pub data_cost: f64,
    pub pairwise_cost: f64,
}

impl ExactOwnerStats {
    pub fn coverage(self) -> Option<f64> {
        (self.textured_pixels != 0)
            .then_some(1.0 - self.shortfall_pixels as f64 / self.textured_pixels as f64)
    }
}

/// Sum exact run evidence for the current labeling or an arbitrary row-major
/// synthetic labeling. This is intentionally independent of image replay;
/// the caller compares it with the unchanged Quality-Gate planes afterward.
pub(in crate::panorama_utils::stitching) fn exact_run_stats(
    width: u32,
    runs: &[ExactRun],
    labels: Option<&[u64]>,
    shortfall_limit: u8,
) -> ExactOwnerStats {
    let mut result = ExactOwnerStats::default();
    for run in runs {
        for offset in 0..run.len {
            if !run.covered {
                continue;
            }
            result.covered_pixels = result.covered_pixels.saturating_add(1);
            let index = run.y as usize * width as usize + (run.x_start + offset) as usize;
            let source = labels
                .and_then(|labels| labels.get(index).copied())
                .unwrap_or(run.production_source);
            let selected = run
                .candidates
                .iter()
                .find(|candidate| candidate.source == source);
            let shortfall = if source == run.production_source {
                run.production_shortfall
            } else {
                selected.map_or(u8::MAX, |candidate| candidate.shortfall)
            };
            if selected.is_none() || shortfall == u8::MAX {
                result.unresolved_pixels = result.unresolved_pixels.saturating_add(1);
                continue;
            }
            if run.textured {
                result.textured_pixels = result.textured_pixels.saturating_add(1);
                result.shortfall_pixels = result
                    .shortfall_pixels
                    .saturating_add(u64::from(shortfall > shortfall_limit));
            }
            if let Some(selected) = selected {
                result.data_cost += selected.data_cost();
                if selected.seam_boundary {
                    result.pairwise_cost += selected.pairwise_cost();
                }
            }
        }
    }
    result
}
