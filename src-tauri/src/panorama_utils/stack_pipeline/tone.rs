//! Low-frequency affine tone harmonisation for owned focus-stack pixels.
//!
//! Tone correction is deliberately separated from ownership.  The owner image
//! supplies the high-frequency residual while a low-pass version of the source
//! supplies only broad exposure and white-balance changes.  This keeps strokes,
//! edges, and texture tied to the source selected by the ownership map.

#![allow(dead_code)]

use image::{GrayImage, Rgb32FImage};
#[cfg(test)]
use image::{Luma, Rgb};
use std::sync::Mutex;

#[cfg(test)]
use super::degradation;
#[cfg(test)]
use super::report::{TonePairWithoutEvidenceRecord, ToneStatus};
use super::report::{ToneReport, ToneTileRecord};

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
/// The spatial residual experiment passed its overlap holdout but regressed
/// independent output-to-source ROI Delta-E on Wenyuan, so it must not enter
/// the production correction path.
const TONE_SPATIAL_FIELD_ENABLED: bool = false;

#[cfg(test)]
pub(crate) const fn production_spatial_field_enabled() -> bool {
    TONE_SPATIAL_FIELD_ENABLED
}
/// Fixed deterministic iterations used only while evaluating that experiment.
const TONE_SPATIAL_ITERATIONS: usize = 4;

// Tone solving is reached from the compositor's free functions, which do not
// carry a mutable Stack_Report reference.  Keep the same run-scoped sink used
// by Intra_Station and Focus_Fuser; report publication takes one snapshot at
// the terminating boundary.
static RUN_RECORDS: Mutex<Vec<ToneTileRecord>> = Mutex::new(Vec::new());
static RUN_REPORT: Mutex<Option<ToneReport>> = Mutex::new(None);
static RUN_TOPOLOGY: Mutex<Option<ToneStationTopology>> = Mutex::new(None);
// Radiometric eligibility is fixed after native station validation, before
// robust closure can assign an accepted geometric edge zero fitting weight.
static RUN_NATIVE_PHOTOMETRIC_TOPOLOGY: Mutex<Option<ToneStationTopology>> = Mutex::new(None);

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
    /// Locally measured, gate-passing adjacent station pairs that were
    /// removed by the global closure graph. These pairs are eligible for the
    /// ownership seam solver only; they are deliberately excluded from tone
    /// evidence and from the authoritative pose graph.
    pub(crate) seam_candidates: Vec<(usize, usize)>,
    /// Locally fitted source-to-target transforms for adjacent station pairs.
    /// This includes `seam_candidates` and may include accepted pairs whose
    /// seam-only refinement was explicitly requested by the compositor. The
    /// transforms never become tone evidence or pose-graph constraints.
    /// Matrices are stored as row-major IEEE-754 bits so this run-scoped
    /// topology remains deterministic and can keep its `Eq` contract without
    /// putting raw `f64` values in the public snapshot.
    pub(crate) seam_candidate_transforms: Vec<(usize, usize, [u64; 9])>,
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
    *RUN_NATIVE_PHOTOMETRIC_TOPOLOGY
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = None;
}

pub(crate) fn record_native_photometric_topology(topology: ToneStationTopology) {
    *RUN_NATIVE_PHOTOMETRIC_TOPOLOGY
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = Some(topology);
}

pub(crate) fn native_photometric_topology_snapshot() -> Option<ToneStationTopology> {
    RUN_NATIVE_PHOTOMETRIC_TOPOLOGY
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
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
    /// Original native quadrature samples retained for narrow contour seam costs.
    pub(crate) native_detail: Option<Rgb32FImage>,
    pub(crate) cell_coverage: Vec<f32>,
}

mod pair_evidence;

pub(crate) use pair_evidence::{
    PairField, ToneSample, ToneSolve, ToneSolveStatus, consistent_samples, solve_tone_pair,
};
use pair_evidence::{ToneRelation, log_channel, record_insufficient_samples};

mod tone_tiles;

#[cfg(test)]
use tone_tiles::apply_low_frequency_tone;
pub(super) use tone_tiles::{gaussian_cells, sample_tile_detail_bilinear};
pub(crate) use tone_tiles::{
    sample_tile_bilinear, world_aligned_native_tone_tile, world_aligned_tone_tile,
};

mod color_metrics;

pub(crate) use color_metrics::delta_e00_rgb;
#[cfg(test)]
pub(crate) use color_metrics::diagnostic_lab_rgb;
use color_metrics::linear_luma;
#[cfg(test)]
use color_metrics::rgb_to_lab;

mod transforms;

pub(super) use transforms::bounded_low_frequency_affine_coordinate;
use transforms::{
    ToneHoldoutGate, build_spatial_field_iterations, constrain_source_fidelity,
    sample_spatial_gain, solve_tone_transforms, spatial_holdout_gate,
};
#[cfg(test)]
use transforms::{ToneSpatialField, ToneTransform, percentile};

mod harmonize;

#[cfg(test)]
pub(crate) use harmonize::harmonize_tone_tiles;
pub(crate) use harmonize::harmonize_tone_tiles_with_fidelity;

#[cfg(test)]
#[test]
fn native_subcell_detail_keeps_a_shift_hidden_by_equal_cell_means() {
    let make = |shift: f64| {
        world_aligned_native_tone_tile(0, 1, (0, 0, 127, 127), |x, _| {
            Some([0.4 + 0.2 * ((x + shift) * std::f64::consts::TAU / 16.0).sin() as f32; 3])
        })
    };
    let mut a = make(0.0);
    let mut b = make(4.0);
    let native_a = sample_tile_detail_bilinear(&a, 65.5, 65.5).unwrap();
    let native_b = sample_tile_detail_bilinear(&b, 65.5, 65.5).unwrap();
    assert!((native_a[0] - native_b[0]).abs() > 0.05);
    assert_eq!(a.cell_coverage, b.cell_coverage);
    a.native_detail = None;
    b.native_detail = None;
    let coarse_a = sample_tile_detail_bilinear(&a, 65.5, 65.5).unwrap();
    let coarse_b = sample_tile_detail_bilinear(&b, 65.5, 65.5).unwrap();
    assert!((coarse_a[0] - coarse_b[0]).abs() < 1e-6);
}

#[cfg(test)]
#[derive(Clone)]
struct ToneOracleBoundaryWindow {
    key: (usize, usize, u32, u32),
    baseline_first_sum: [f64; 3],
    baseline_second_sum: [f64; 3],
    field_first_delta_sum: [f64; 3],
    field_second_delta_sum: [f64; 3],
    samples: u64,
}

/// Test-only current-run state consumed by the Quality Gate owner-RAW oracle.
/// It contains no fixture coordinates or previous-run report data.
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct ToneOwnerOracleSnapshot {
    baseline: Rgb32FImage,
    owners: Vec<u16>,
    tiles: Vec<ToneTile>,
    relations: Vec<ToneRelation>,
    transforms: Vec<ToneTransform>,
    fields: Vec<ToneSpatialField>,
    fit_cells: std::collections::BTreeSet<(i64, i64)>,
    overlap_cells: std::collections::BTreeSet<(i64, i64)>,
    boundaries: Vec<ToneOracleBoundaryWindow>,
    world_origin: (f64, f64),
}

#[cfg(test)]
static TONE_OWNER_ORACLE_SNAPSHOT: Mutex<Option<ToneOwnerOracleSnapshot>> = Mutex::new(None);

#[cfg(test)]
impl ToneOwnerOracleSnapshot {
    fn shared_cell(&self, world_x: f64, world_y: f64) -> Option<(i64, i64)> {
        let field = self.fields.first()?;
        if !field.world_stride.is_finite() || field.world_stride <= 0.0 {
            return None;
        }
        Some((
            ((world_x - field.world_origin.0) / field.world_stride).floor() as i64,
            ((world_y - field.world_origin.1) / field.world_stride).floor() as i64,
        ))
    }

    pub(crate) fn is_fit_cell(&self, world_x: f64, world_y: f64) -> bool {
        self.shared_cell(world_x, world_y)
            .is_some_and(|cell| self.fit_cells.contains(&cell))
    }

    pub(crate) fn overlap_participation(&self, world_x: f64, world_y: f64) -> bool {
        self.shared_cell(world_x, world_y)
            .is_some_and(|cell| self.overlap_cells.contains(&cell))
    }

    pub(crate) fn baseline(&self) -> &Rgb32FImage {
        &self.baseline
    }

    pub(crate) fn owner_at_world(&self, world_x: f64, world_y: f64) -> Option<u16> {
        let x = (world_x - self.world_origin.0).round();
        let y = (world_y - self.world_origin.1).round();
        if x < 0.0
            || y < 0.0
            || x >= f64::from(self.baseline.width())
            || y >= f64::from(self.baseline.height())
        {
            return None;
        }
        self.owners
            .get(y as usize * self.baseline.width() as usize + x as usize)
            .copied()
    }

    pub(crate) fn field_sample(
        &self,
        owner: u16,
        world_x: f64,
        world_y: f64,
    ) -> Option<([f32; 3], [f32; 3], [f32; 3], [f32; 3], usize)> {
        let tile_index = self.tiles.iter().position(|tile| tile.owner_id == owner)?;
        let low = sample_tile_bilinear(&self.tiles[tile_index], world_x, world_y)?;
        let local_gain = sample_spatial_gain(&self.fields[tile_index], world_x, world_y)?;
        let transform = self.transforms[tile_index];
        Some((
            low,
            local_gain,
            transform.gain,
            transform.offset,
            tile_index,
        ))
    }

    pub(crate) fn candidate_pixel(
        &self,
        owner: u16,
        world_x: f64,
        world_y: f64,
        baseline: [f32; 3],
        relaxation: f64,
    ) -> [f32; 3] {
        let Some((low, local_gain, gain, _, _)) = self.field_sample(owner, world_x, world_y) else {
            return baseline;
        };
        std::array::from_fn(|channel| {
            (f64::from(baseline[channel])
                + relaxation
                    * f64::from(gain[channel] * (local_gain[channel] - 1.0) * low[channel]))
            .clamp(0.0, 1.0) as f32
        })
    }

    pub(crate) fn boundary_max(&self, relaxation: f64) -> f64 {
        self.boundaries
            .iter()
            .filter(|window| window.samples > 0)
            .map(|window| {
                let mean = |baseline: [f64; 3], delta: [f64; 3]| {
                    std::array::from_fn(|channel| {
                        ((baseline[channel] + relaxation * delta[channel]) / window.samples as f64)
                            as f32
                    })
                };
                delta_e00_rgb(
                    mean(window.baseline_first_sum, window.field_first_delta_sum),
                    mean(window.baseline_second_sum, window.field_second_delta_sum),
                )
            })
            .fold(0.0, f64::max)
    }

    pub(crate) fn boundary_window_count(&self) -> usize {
        self.boundaries.len()
    }

    pub(crate) fn fit_cell_count(&self) -> usize {
        self.fit_cells.len()
    }

    pub(crate) fn field_constraint_stats(&self) -> (f64, f64, f64) {
        let mut minimum_gain = f64::INFINITY;
        let mut maximum_gain = f64::NEG_INFINITY;
        let mut maximum_neighbour_log_delta = 0.0f64;
        for field in &self.fields {
            let (width, height) = field.log_gain.dimensions();
            for y in 0..height {
                for x in 0..width {
                    if field.validity.get_pixel(x, y)[0] == 0 {
                        continue;
                    }
                    let value = field.log_gain.get_pixel(x, y).0;
                    for channel in 0..3 {
                        let gain = f64::from(value[channel]).exp();
                        minimum_gain = minimum_gain.min(gain);
                        maximum_gain = maximum_gain.max(gain);
                    }
                    for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                        if nx >= width || ny >= height || field.validity.get_pixel(nx, ny)[0] == 0 {
                            continue;
                        }
                        let neighbour = field.log_gain.get_pixel(nx, ny).0;
                        for channel in 0..3 {
                            maximum_neighbour_log_delta = maximum_neighbour_log_delta
                                .max(f64::from(value[channel] - neighbour[channel]).abs());
                        }
                    }
                }
            }
        }
        if !minimum_gain.is_finite() {
            minimum_gain = 1.0;
            maximum_gain = 1.0;
        }
        (minimum_gain, maximum_gain, maximum_neighbour_log_delta)
    }

    pub(crate) fn checkerboard_holdout(&self, relaxation: f64) -> (usize, f64, f64) {
        let mut values = Vec::new();
        for relation in &self.relations {
            let Some(field) =
                PairField::new(&self.tiles[relation.first], &self.tiles[relation.second])
            else {
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
                let world_y =
                    field.origin.1 + row as f64 * field.stride + (field.stride - 1.0) * 0.5;
                let Some(cell) = self.shared_cell(world_x, world_y) else {
                    continue;
                };
                if (cell.0 + cell.1).rem_euclid(2) == 0 {
                    continue;
                }
                let corrected = |tile: usize, pixel: [f32; 3]| {
                    let transform = self.transforms[tile];
                    let local = sample_spatial_gain(&self.fields[tile], world_x, world_y)
                        .unwrap_or([1.0; 3]);
                    std::array::from_fn(|channel| {
                        let baseline =
                            transform.gain[channel] * pixel[channel] + transform.offset[channel];
                        baseline
                            + relaxation as f32
                                * transform.gain[channel]
                                * (local[channel] - 1.0)
                                * pixel[channel]
                    })
                };
                values.push(delta_e00_rgb(
                    corrected(relation.first, field.first[index]),
                    corrected(relation.second, field.second[index]),
                ));
            }
        }
        let count = values.len();
        let median = percentile(&mut values.clone(), 0.5);
        let p95 = percentile(&mut values, 0.95);
        (count, median, p95)
    }

    pub(crate) fn cycle_residual_max(&self) -> f64 {
        self.relations
            .iter()
            .flat_map(|relation| {
                (0..3).map(move |channel| {
                    (f64::from(self.transforms[relation.first].gain[channel]).ln()
                        - f64::from(self.transforms[relation.second].gain[channel]).ln()
                        + relation.log_gain[channel])
                        .abs()
                })
            })
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
pub(crate) fn take_tone_owner_oracle_snapshot() -> Option<ToneOwnerOracleSnapshot> {
    TONE_OWNER_ORACLE_SNAPSHOT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
}

#[cfg(test)]
fn tone_oracle_cell_sets(
    tiles: &[ToneTile],
    relations: &[ToneRelation],
    fields: &[ToneSpatialField],
) -> (
    std::collections::BTreeSet<(i64, i64)>,
    std::collections::BTreeSet<(i64, i64)>,
) {
    let mut fit = std::collections::BTreeSet::new();
    let mut overlap = std::collections::BTreeSet::new();
    let Some(shared) = fields.first() else {
        return (fit, overlap);
    };
    for relation in relations {
        let Some(field) = PairField::new(&tiles[relation.first], &tiles[relation.second]) else {
            continue;
        };
        for index in 0..field.sample.len() {
            if !field.sample[index] {
                continue;
            }
            let column = index % field.width;
            let row = index / field.width;
            let world_x =
                field.origin.0 + column as f64 * field.stride + (field.stride - 1.0) * 0.5;
            let world_y = field.origin.1 + row as f64 * field.stride + (field.stride - 1.0) * 0.5;
            let cell = (
                ((world_x - shared.world_origin.0) / shared.world_stride).floor() as i64,
                ((world_y - shared.world_origin.1) / shared.world_stride).floor() as i64,
            );
            overlap.insert(cell);
            if (cell.0 + cell.1).rem_euclid(2) == 0 {
                fit.insert(cell);
            }
        }
    }
    (fit, overlap)
}

#[cfg(test)]
fn tone_distribution_pair(first: [f32; 3], second: [f32; 3]) -> serde_json::Value {
    let linear = |rgb: [f32; 3]| rgb.map(crate::color_management::srgb_to_linear_channel);
    serde_json::json!({
        "display_srgb_mean": [first, second],
        "linear_srgb_mean": [linear(first), linear(second)],
        "lab_mean": [rgb_to_lab(first), rgb_to_lab(second)],
        "channel_ratio": std::array::from_fn::<f64, 3, _>(|channel| {
            f64::from(first[channel]) / f64::from(second[channel]).max(1.0 / 65_535.0)
        }),
        "delta_e00": delta_e00_rgb(first, second),
    })
}

#[cfg(test)]
fn write_tone_diagnostics(
    before: &Rgb32FImage,
    after: &Rgb32FImage,
    tiles: &[ToneTile],
    relations: &[ToneRelation],
    transforms: &[ToneTransform],
    spatial_fields: Option<&[ToneSpatialField]>,
    holdout_gate: ToneHoldoutGate,
    world_origin: (f64, f64),
    windows: &[serde_json::Value],
    report: &ToneReport,
) {
    let Some(directory) = super::test_support::tone_diagnostics_dir() else {
        return;
    };
    if let Err(error) = std::fs::create_dir_all(&directory) {
        eprintln!("cannot create tone diagnostics directory: {error}");
        return;
    }
    let relation_records = relations
        .iter()
        .map(|relation| {
            let first = &tiles[relation.first];
            let second = &tiles[relation.second];
            let residual = std::array::from_fn::<f64, 3, _>(|channel| {
                f64::from(transforms[relation.first].gain[channel]).ln()
                    - f64::from(transforms[relation.second].gain[channel]).ln()
                    + relation.log_gain[channel]
            });
            let corrected_mean = |tile: usize, owner_side: bool| {
                let transform = &transforms[tile];
                let samples = &relation.samples;
                std::array::from_fn::<f32, 3, _>(|channel| {
                    samples
                        .iter()
                        .map(|sample| {
                            let value = if owner_side {
                                sample.owner[channel]
                            } else {
                                sample.source[channel]
                            };
                            f64::from(transform.gain[channel] * value + transform.offset[channel])
                        })
                        .sum::<f64>() as f32
                        / samples.len().max(1) as f32
                })
            };
            let first_pre = std::array::from_fn::<f32, 3, _>(|channel| {
                relation.samples.iter().map(|sample| sample.owner[channel]).sum::<f32>()
                    / relation.samples.len().max(1) as f32
            });
            let second_pre = std::array::from_fn::<f32, 3, _>(|channel| {
                relation.samples.iter().map(|sample| sample.source[channel]).sum::<f32>()
                    / relation.samples.len().max(1) as f32
            });
            let representative = relation
                .samples
                .iter()
                .take(16)
                .map(|sample| serde_json::json!({ "first": sample.owner, "second": sample.source }))
                .collect::<Vec<_>>();
            serde_json::json!({
                "tile_indices": [relation.first, relation.second],
                "stations": [first.station_index, second.station_index],
                "owner_ids": [first.owner_id, second.owner_id],
                "overlap_world": {
                    "left": first.world_origin.0.max(second.world_origin.0),
                    "top": first.world_origin.1.max(second.world_origin.1),
                    "right": (first.world_origin.0 + first.world_size.0).min(second.world_origin.0 + second.world_size.0),
                    "bottom": (first.world_origin.1 + first.world_size.1).min(second.world_origin.1 + second.world_size.1),
                },
                "pair_samples": {
                    "retained": relation.samples.len(),
                    "minimum": TONE_MIN_SAMPLES,
                    "texture_rejection_enabled": false,
                },
                "log_gain": relation.log_gain,
                "scatter": relation.scatter,
                "weight": relation.weight(),
                "solver_log_residual": residual,
                "pre": tone_distribution_pair(first_pre, second_pre),
                "post": tone_distribution_pair(
                    corrected_mean(relation.first, true),
                    corrected_mean(relation.second, false),
                ),
                "representative_low_frequency_samples": representative,
                "holdout": { "enabled": false },
            })
        })
        .collect::<Vec<_>>();
    let tile_records = tiles
        .iter()
        .zip(transforms)
        .map(|(tile, transform)| {
            serde_json::json!({
                "station": tile.station_index,
                "owner_id": tile.owner_id,
                "world_origin": tile.world_origin,
                "world_size": tile.world_size,
                "world_stride": tile.world_stride,
                "field_dimensions": tile.low.dimensions(),
                "gain": transform.gain,
                "bias": transform.offset,
                "solved_gain": transform.solved_gain,
                "solved_bias": transform.solved_offset,
                "supported": transform.supported,
                "clamped": transform.clamped,
            })
        })
        .collect::<Vec<_>>();
    let cycle_consistency = if relations.len() == 3 {
        let relation_gain = |first: usize, second: usize, channel: usize| {
            relations.iter().find_map(|relation| {
                if relation.first == first && relation.second == second {
                    Some(relation.log_gain[channel])
                } else if relation.first == second && relation.second == first {
                    Some(-relation.log_gain[channel])
                } else {
                    None
                }
            })
        };
        (0..3)
            .map(|channel| {
                relation_gain(0, 1, channel).unwrap_or(0.0)
                    + relation_gain(1, 2, channel).unwrap_or(0.0)
                    + relation_gain(2, 0, channel).unwrap_or(0.0)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let payload = serde_json::json!({
        "schema": 1,
        "color_encoding": "display_srgb_with_explicit_linear_and_lab_diagnostics",
        "canvas": { "width": after.width(), "height": after.height(), "world_origin": world_origin },
        "graph": {
            "reference": "zero_mean_log_gain_per_connected_component",
            "relations": relations.len(),
            "cycle_log_residual": cycle_consistency,
            "deterministic_relation_order": true,
        },
        "tiles": tile_records,
        "relations": relation_records,
        "boundary_windows": windows,
        "summary": {
            "status": report.status,
            "maximum_delta_e00": report.boundary_delta_e.max,
            "threshold": report.boundary_delta_e.threshold,
            "window_count": windows.len(),
            "violations": report.boundary_delta_e.violations.len(),
            "spatial_field_enabled": spatial_fields.is_some(),
            "spatial_iterations": TONE_SPATIAL_ITERATIONS,
            "held_out_improvement_gate": {
                "samples": holdout_gate.samples,
                "before_median_delta_e00": holdout_gate.before_median,
                "before_p95_delta_e00": holdout_gate.before_p95,
                "after_median_delta_e00": holdout_gate.after_median,
                "after_p95_delta_e00": holdout_gate.after_p95,
                "applied": holdout_gate.applied,
            },
        },
    });
    if let Ok(bytes) = serde_json::to_vec_pretty(&payload) {
        if let Err(error) = std::fs::write(directory.join("tone-final-world.json"), bytes) {
            eprintln!("cannot write tone diagnostics: {error}");
        }
    }

    let maximum = windows.iter().max_by(|left, right| {
        left["delta_e00"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&right["delta_e00"].as_f64().unwrap_or(0.0))
    });
    if let Some(maximum) = maximum {
        let world = &maximum["world"];
        let left = world["left"].as_f64().unwrap_or(world_origin.0);
        let top = world["top"].as_f64().unwrap_or(world_origin.1);
        let x = (left - world_origin.0).max(0.0) as u32;
        let y = (top - world_origin.1).max(0.0) as u32;
        let crop_left = x.saturating_sub(TONE_BOUNDARY_WINDOW_PX);
        let crop_top = y.saturating_sub(TONE_BOUNDARY_WINDOW_PX);
        let crop_width = (TONE_BOUNDARY_WINDOW_PX * 3).min(after.width() - crop_left);
        let crop_height = (TONE_BOUNDARY_WINDOW_PX * 3).min(after.height() - crop_top);
        for (name, image) in [("pre", before), ("post", after)] {
            let crop =
                image::imageops::crop_imm(image, crop_left, crop_top, crop_width, crop_height)
                    .to_image();
            let encoded = image::DynamicImage::ImageRgb32F(crop).to_rgb16();
            if let Err(error) = encoded.save(directory.join(format!("tone-max-{name}.png"))) {
                eprintln!("cannot write tone crop: {error}");
            }
        }
    }

    let heat_width = after.width().div_ceil(TONE_BOUNDARY_WINDOW_PX);
    let heat_height = after.height().div_ceil(TONE_BOUNDARY_WINDOW_PX);
    let mut heat = image::RgbImage::new(heat_width, heat_height);
    for window in windows {
        let world = &window["world"];
        let x = ((world["left"].as_f64().unwrap_or(world_origin.0) - world_origin.0)
            / f64::from(TONE_BOUNDARY_WINDOW_PX)) as u32;
        let y = ((world["top"].as_f64().unwrap_or(world_origin.1) - world_origin.1)
            / f64::from(TONE_BOUNDARY_WINDOW_PX)) as u32;
        if x >= heat_width || y >= heat_height {
            continue;
        }
        let value = (window["delta_e00"].as_f64().unwrap_or(0.0) / 20.0).clamp(0.0, 1.0);
        let old = heat.get_pixel(x, y).0;
        heat.put_pixel(
            x,
            y,
            image::Rgb([old[0].max((255.0 * value) as u8), old[1], old[2]]),
        );
    }
    if let Err(error) = heat.save(directory.join("tone-boundary-heatmap.png")) {
        eprintln!("cannot write tone heatmap: {error}");
    }
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
    #[test]
    fn spatial_window_lookup_uses_the_measured_native_cell_alignment() {
        let field = PairField {
            origin: (113.0, 7.0),
            stride: 16.0,
            width: 2,
            height: 2,
            first: vec![[0.4; 3]; 4],
            second: vec![[0.35; 3]; 4],
            sample: vec![true; 4],
        };
        let key = field.spatial_window_key(128.0, 17.0, 128.0).unwrap();
        assert_eq!(key, (0, 0));
        assert_ne!(key, ((128.0f64 / 128.0).floor() as i64, 0));
        let (models, _) = field.spatial_consistency(128.0);
        let (a, b) = field.at(128.0, 17.0).unwrap();
        assert!(models[&key].accepts(&ToneSample {
            owner: a,
            source: b
        }));
    }
    #[test]
    fn spatial_mad_rejects_corruption_without_rejecting_the_illumination_field() {
        let mut first = (0..4096)
            .map(|i| [0.4 * (0.8 + 0.4 * (i % 64) as f32 / 63.0); 3])
            .collect::<Vec<_>>();
        let bad = 32 * 64 + 32;
        first[bad] = [0.9, 0.1, 0.7];
        let field = PairField {
            origin: (0.0, 0.0),
            stride: 16.0,
            width: 64,
            height: 64,
            first,
            second: vec![[0.4; 3]; 4096],
            sample: vec![true; 4096],
        };
        let (models, retained) = field.spatial_consistency(128.0);
        assert!(retained >= TONE_MIN_SAMPLES);
        assert!(!models[&(4, 4)].accepts(&ToneSample {
            owner: field.first[bad],
            source: field.second[bad]
        }));
        for i in [8 * 64 + 8, 8 * 64 + 56, 56 * 64 + 8, 56 * 64 + 56] {
            assert!(
                models[&((i % 64 / 8) as i64, (i / 64 / 8) as i64)].accepts(&ToneSample {
                    owner: field.first[i],
                    source: field.second[i]
                })
            );
        }
    }
    use super::*;

    #[test]
    fn affine_coordinate_refits_offset_when_gain_reaches_its_bound() {
        let normal = [0.46, 1.0, 3.0];
        let rhs = [normal[0] * 0.4 - 0.005, 0.4 - 0.015];
        let fitted = bounded_low_frequency_affine_coordinate(normal, rhs);
        assert!((fitted.0 - 0.25).abs() < 1e-12);
        assert!((fitted.1 - f64::from(TONE_MAX_ABS_OFFSET)).abs() < 1e-12);
        let unbounded_rhs = [normal[0] * 0.12 - 0.008, 0.12 - 0.024];
        let fitted = bounded_low_frequency_affine_coordinate(normal, unbounded_rhs);
        assert!((fitted.0 - 0.12).abs() < 1e-12);
        assert!((fitted.1 + 0.008).abs() < 1e-12);
    }

    #[test]
    fn affine_coordinate_agrees_with_an_independent_dense_feasible_oracle() {
        for (normal, rhs) in [
            ([0.46, 1.0, 3.0], [0.179, 0.385]),
            ([0.46, 1.0, 3.0], [-0.15, -0.25]),
            ([0.12, 0.1, 0.7], [0.01, -0.04]),
        ] {
            let cost = |alpha: f64, offset: f64| {
                normal[0] * alpha * alpha
                    + 2.0 * normal[1] * alpha * offset
                    + normal[2] * offset * offset
                    - 2.0 * rhs[0] * alpha
                    - 2.0 * rhs[1] * offset
            };
            let fitted = bounded_low_frequency_affine_coordinate(normal, rhs);
            let minimum = (0..=450)
                .flat_map(|i| {
                    (0..=400).map(move |j| cost(-0.2 + i as f64 * 0.001, -0.02 + j as f64 * 0.0001))
                })
                .min_by(f64::total_cmp)
                .unwrap();
            assert!(cost(fitted.0, fitted.1) <= minimum + 1e-8);
        }
        assert_eq!(
            bounded_low_frequency_affine_coordinate([0.0; 3], [0.0; 2]),
            (0.0, 0.0)
        );
    }

    #[test]
    fn fewer_than_1024_samples_keep_identity() {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
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

    #[test]
    fn source_fidelity_bounds_preserve_the_unchanged_reference_metrics() {
        let low_at = |x: u32| {
            let b = 0.3 + 0.03 * x as f32 / 127.0;
            [b, b * 0.95, b * 0.9]
        };
        let source = Rgb32FImage::from_fn(128, 128, |x, y| {
            let noise = ((x * 37 + y * 53) % 101) as f32 / 100.0 * 0.012 - 0.006;
            Rgb(low_at(x).map(|value| value + noise))
        });
        let low = Rgb32FImage::from_fn(128, 128, |x, _| Rgb(low_at(x)));
        let tile = ToneTile {
            station_index: 0,
            owner_id: 1,
            cell_mean: low.clone(),
            low,
            validity: GrayImage::from_pixel(128, 128, Luma([255])),
            world_origin: (0.0, 0.0),
            world_size: (128.0, 128.0),
            world_stride: 1.0,
            native_detail: None,
            cell_coverage: vec![1.0; 128 * 128],
        };
        let mut transform = ToneTransform {
            gain: [1.2; 3],
            offset: [0.01; 3],
            solved_gain: [1.2; 3],
            solved_offset: [0.01; 3],
            clamped: false,
            supported: true,
        };
        let render = |transform: &ToneTransform| {
            Rgb32FImage::from_fn(128, 128, |x, y| {
                let before = source.get_pixel(x, y).0;
                let low = low_at(x);
                Rgb(std::array::from_fn(|c| {
                    before[c] + (transform.gain[c] - 1.0) * low[c] + transform.offset[c]
                }))
            })
        };
        let gradient =
            super::super::quality_gate::normalized_gradient_energy(&source, 1.0).unwrap();
        assert!(
            super::super::quality_gate::normalized_gradient_energy(&render(&transform), 1.0)
                .unwrap()
                / gradient
                < 0.95,
            "unbounded tone must expose the real loss of local contrast"
        );
        let strength = constrain_source_fidelity(&tile, &mut transform);
        assert!(strength > 0.0 && strength < 1.0);
        let output = render(&transform);
        assert!(
            super::super::quality_gate::normalized_gradient_energy(&output, 1.0).unwrap()
                / gradient
                >= 0.95
        );
        let noise = super::super::quality_gate::noise_sigma_ratio(&output, &source).unwrap();
        assert!((0.85..=1.15).contains(&noise), "{noise:?}");
        assert!(
            super::super::quality_gate::roi_low_frequency_delta_e00(&output, &source).unwrap()
                <= 2.0
        );
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
    fn native_tone_fields_compare_the_same_world_window_across_grid_phases() {
        let sample = |x: f64, y: f64| Some([0.12 + x as f32 * 0.0003 + y as f32 * 0.0002; 3]);
        let first = world_aligned_native_tone_tile(0, 1, (0, 0, 511, 511), sample);
        let second = world_aligned_native_tone_tile(1, 2, (7, 11, 518, 522), sample);
        let pair = PairField::new(&first, &second).unwrap();
        for y in (64..448).step_by(16) {
            for x in (64..448).step_by(16) {
                let (a, b) = pair.at(x as f64, y as f64).unwrap();
                assert!(delta_e00_rgb(a, b) < 0.0001, "{x},{y}: {a:?}/{b:?}");
            }
        }
        let dark = world_aligned_native_tone_tile(2, 3, (7, 11, 518, 522), |x, y| {
            sample(x, y).map(|v| v.map(|c| c * 0.8))
        });
        let shifted = PairField::new(&first, &dark).unwrap();
        let (a, b) = shifted.at(256.0, 256.0).unwrap();
        assert!(
            delta_e00_rgb(a, b) > 3.0,
            "a real source tone step must remain visible"
        );
    }

    #[test]
    fn seam_cost_detects_low_frequency_tone_mismatch_without_altering_pixels() {
        let image = Rgb32FImage::from_fn(512, 512, |x, y| {
            let v = 0.1 + 0.05 * x as f32 / 511.0 + 0.02 * y as f32 / 511.0;
            Rgb([v; 3])
        });
        let brighter = Rgb32FImage::from_fn(512, 512, |x, y| {
            Rgb(image.get_pixel(x, y).0.map(|v| v + 0.08))
        });
        let make = |owner, image: &Rgb32FImage| {
            world_aligned_tone_tile(owner as usize, owner, image, (0, 0, 511, 511), |x, y| {
                Some((x, y))
            })
        };
        let first = make(1, &image);
        let same = make(2, &image);
        let different = make(3, &brighter);
        let same_field = PairField::new(&first, &same).unwrap();
        let mismatch = PairField::new(&first, &different).unwrap();
        assert!(same_field.seam_disagreement(256.0, 256.0).unwrap() < 1e-8);
        assert!(mismatch.seam_disagreement(256.0, 256.0).unwrap() > 1.0);
        assert_eq!(image.get_pixel(0, 0).0, [0.1; 3]);
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
        // `harmonize_tone_tiles` writes the process-wide run ledger.
        let _run_scope = degradation::begin_run_scope();
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
        let _run_scope = degradation::begin_run_scope();
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
        let _run_scope = degradation::begin_run_scope();
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
        let _run_scope = degradation::begin_run_scope();
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
        let _run_scope = degradation::begin_run_scope();
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
    fn harmonization_is_deterministic_across_station_order_and_thread_count() {
        fn run(
            order: [usize; 3],
            threads: usize,
        ) -> (Rgb32FImage, Vec<ToneTileRecord>, ToneStatus, f64) {
            let _run_scope = degradation::begin_run_scope();
            let (width, height) = (1_536u32, 640u32);
            let origins = [0u32, 256, 512];
            let exposures = [1.0f32, 0.9, 1.1];
            let sources = origins
                .iter()
                .zip(exposures)
                .map(|(&origin, exposure)| {
                    shifted_tile(1_024, height, f64::from(origin), exposure, 0..0)
                })
                .collect::<Vec<_>>();
            let canonical_tiles = (0..3)
                .map(|station| {
                    let origin = origins[station];
                    world_aligned_tone_tile(
                        station,
                        station as u16 + 1,
                        &sources[station],
                        (origin, 0, origin + 1_023, height - 1),
                        |x, y| Some((x - f64::from(origin), y)),
                    )
                })
                .collect::<Vec<_>>();
            let owners = (0..height)
                .flat_map(|_| {
                    (0..width).map(|x| {
                        if x < 512 {
                            1u16
                        } else if x < 1_024 {
                            2
                        } else {
                            3
                        }
                    })
                })
                .collect::<Vec<_>>();
            let mut panorama = Rgb32FImage::from_fn(width, height, |x, y| {
                let station = owners[(y * width + x) as usize] as usize - 1;
                *sources[station].get_pixel(x - origins[station], y)
            });
            let tiles = order
                .iter()
                .map(|&station| canonical_tiles[station].clone())
                .collect::<Vec<_>>();
            let mut position = [0usize; 3];
            for (index, station) in order.into_iter().enumerate() {
                position[station] = index;
            }
            let relations = [(0usize, 2usize), (1, 2), (0, 1)]
                .map(|(left, right)| (position[left], position[right]));
            let evidence = GrayImage::from_pixel(width, height, Luma([255]));
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let report = pool.install(|| {
                harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &relations)
            });
            let mut tile_records = report.tiles;
            tile_records.sort_by_key(|tile| tile.station_index);
            (
                panorama,
                tile_records,
                report.status,
                report.boundary_delta_e.max,
            )
        }

        let single_thread = run([0, 1, 2], 1);
        let reordered_parallel = run([2, 0, 1], 4);
        assert_eq!(single_thread, reordered_parallel);
    }

    #[test]
    fn owner_oracle_checkerboard_is_globally_disjoint_and_thread_deterministic() {
        fn build(threads: usize) -> (Vec<Vec<f32>>, std::collections::BTreeSet<(i64, i64)>) {
            let width = 1_536u32;
            let height = 640u32;
            let left = shifted_tile(1_024, height, 0.0, 1.0, 0..0);
            let right = shifted_tile(1_024, height, 512.0, 0.9, 0..0);
            let tiles = vec![
                world_aligned_tone_tile(0, 1, &left, (0, 0, 1_023, height - 1), |x, y| {
                    Some((x, y))
                }),
                world_aligned_tone_tile(1, 2, &right, (512, 0, width - 1, height - 1), |x, y| {
                    Some((x - 512.0, y))
                }),
            ];
            let pair = PairField::new(&tiles[0], &tiles[1]).unwrap();
            let relation = ToneRelation::new(0, 1, consistent_samples(&pair.samples()));
            let transforms = solve_tone_transforms(2, std::slice::from_ref(&relation));
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let fields = pool.install(|| {
                build_spatial_field_iterations(
                    &tiles,
                    std::slice::from_ref(&relation),
                    &transforms,
                    true,
                )
            });
            let (fit, overlap) =
                tone_oracle_cell_sets(&tiles, std::slice::from_ref(&relation), &fields);
            assert!(!fit.is_empty());
            assert!(fit.is_subset(&overlap));
            assert!(
                overlap
                    .iter()
                    .filter(|cell| !fit.contains(cell))
                    .all(|cell| (cell.0 + cell.1).rem_euclid(2) != 0)
            );
            (
                fields
                    .iter()
                    .map(|field| field.log_gain.as_raw().clone())
                    .collect(),
                fit,
            )
        }

        assert_eq!(build(1), build(4));
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
