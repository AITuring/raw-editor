//! Test-only Source_RAW owner-fidelity oracle state of the Tone_Harmonizer.
//!
//! Snapshots the current-run tone tiles and boundary windows for the owner
//! fidelity oracle; never part of a production render.

use image::Rgb32FImage;
use std::sync::Mutex;

use super::{
    PairField, ToneRelation, ToneSpatialField, ToneTile, ToneTransform, delta_e00_rgb, percentile,
    sample_spatial_gain, sample_tile_bilinear,
};

#[cfg(test)]
#[derive(Clone)]
pub(super) struct ToneOracleBoundaryWindow {
    pub(super) key: (usize, usize, u32, u32),
    pub(super) baseline_first_sum: [f64; 3],
    pub(super) baseline_second_sum: [f64; 3],
    pub(super) field_first_delta_sum: [f64; 3],
    pub(super) field_second_delta_sum: [f64; 3],
    pub(super) samples: u64,
}

/// Test-only current-run state consumed by the Quality Gate owner-RAW oracle.
/// It contains no fixture coordinates or previous-run report data.
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct ToneOwnerOracleSnapshot {
    pub(super) baseline: Rgb32FImage,
    pub(super) owners: Vec<u16>,
    pub(super) tiles: Vec<ToneTile>,
    pub(super) relations: Vec<ToneRelation>,
    pub(super) transforms: Vec<ToneTransform>,
    pub(super) fields: Vec<ToneSpatialField>,
    pub(super) fit_cells: std::collections::BTreeSet<(i64, i64)>,
    pub(super) overlap_cells: std::collections::BTreeSet<(i64, i64)>,
    pub(super) boundaries: Vec<ToneOracleBoundaryWindow>,
    pub(super) world_origin: (f64, f64),
}

#[cfg(test)]
pub(super) static TONE_OWNER_ORACLE_SNAPSHOT: Mutex<Option<ToneOwnerOracleSnapshot>> =
    Mutex::new(None);

#[cfg(test)]
impl ToneOwnerOracleSnapshot {
    pub(super) fn shared_cell(&self, world_x: f64, world_y: f64) -> Option<(i64, i64)> {
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
pub(super) fn tone_oracle_cell_sets(
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
