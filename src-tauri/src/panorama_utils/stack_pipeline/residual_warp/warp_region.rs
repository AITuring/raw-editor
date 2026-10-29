//! One overlap's bounded residual field: WarpRegion.
//!
//! Builds the control-node grid from round-trip verified observations, solves
//! and clamps the node displacements, reverts cells that do not improve the
//! global model, samples the faded field and reports the region record.

use super::super::super::mosaic::Field;
use super::super::degradation;
use super::super::report::RESIDUAL_WARP_NODE_STEP_PX;
use super::super::report::WarpRegionRecord;
use super::super::report::WorldPoint;
use super::super::report::WorldRect;

use super::{
    OverlapResidual, RESIDUAL_WARP_EDGE_FADE_PX, RESIDUAL_WARP_EXTRAPOLATION_RADIUS_NODES,
    RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX, RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX,
    RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX, RESIDUAL_WARP_MIN_VERIFIED_POINTS, WarpNode,
    WarpObservation, contains, sane_extent,
};

pub(super) const MIN_GRID_NODES: usize = 4;

pub(super) const GAUSSIAN_SIGMA_NODES: f64 = 1.5;

/// A bounded residual field for one overlap.
#[derive(Debug, Clone)]
pub(crate) struct WarpRegion {
    pub left_station: usize,
    pub right_station: usize,
    pub world: WorldRect,
    pub p95_before_px: f64,
    pub node_step_px: u32,
    pub columns: u32,
    pub rows: u32,
    pub(super) field: Field<2>,
    pub(super) nodes: Vec<WarpNode>,
    pub reverted_cells: u64,
    pub insufficient_evidence: bool,
    pub max_node_displacement_px: f64,
    pub max_neighbour_delta_px: f64,
}

impl WarpRegion {
    /// Accept a jointly estimated grid only where measured observations
    /// support it. The caller must independently validate the resulting
    /// compositor map after these support/boundary guards have been applied.
    pub(crate) fn from_verified_grid(
        overlap: OverlapResidual,
        values: &[[f64; 2]],
        measured: &[WarpObservation],
    ) -> Option<Self> {
        if measured.len() < RESIDUAL_WARP_MIN_VERIFIED_POINTS {
            return None;
        }
        let mut region = Self::identity(overlap);
        if values.len() != region.nodes.len()
            || values.iter().any(|v| {
                !v[0].is_finite()
                    || !v[1].is_finite()
                    || v[0].hypot(v[1]) > RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX
            })
        {
            return None;
        }
        let mut support = vec![false; values.len()];
        let mut round_trip_errors = vec![0.0f64; values.len()];
        for observation in measured {
            let point = &observation.world;
            if !observation_is_valid(*observation) || !contains(&region.world, point.x, point.y) {
                continue;
            }
            let x = ((point.x - region.world.left) / region.field.step).floor() as usize;
            let y = ((point.y - region.world.top) / region.field.step).floor() as usize;
            for dy in 0..=1 {
                for dx in 0..=1 {
                    if x + dx < region.field.width && y + dy < region.field.height {
                        support[(y + dy) * region.field.width + x + dx] = true;
                        let index = (y + dy) * region.field.width + x + dx;
                        round_trip_errors[index] =
                            round_trip_errors[index].max(observation.round_trip_error_px);
                    }
                }
            }
        }
        if support.iter().filter(|&&valid| valid).count() < RESIDUAL_WARP_MIN_VERIFIED_POINTS {
            return None;
        }
        let radius = RESIDUAL_WARP_EXTRAPOLATION_RADIUS_NODES as isize;
        for index in 0..values.len() {
            let (x, y) = (
                (index % region.field.width) as isize,
                (index / region.field.width) as isize,
            );
            let nearby = (-radius..=radius).any(|dy| {
                (-radius..=radius).any(|dx| {
                    let (nx, ny) = (x + dx, y + dy);
                    dx * dx + dy * dy <= radius * radius
                        && nx >= 0
                        && ny >= 0
                        && nx < region.field.width as isize
                        && ny < region.field.height as isize
                        && support[ny as usize * region.field.width + nx as usize]
                })
            });
            region.nodes[index].valid = support[index];
            region.nodes[index].round_trip_error_px = round_trip_errors[index];
            region.nodes[index].displacement = if nearby { values[index] } else { [0.0; 2] };
        }
        region.enforce_all_neighbour_bound();
        region.rebuild_field();
        region.update_metrics();
        (region.max_node_displacement_px <= RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX
            && region.max_neighbour_delta_px <= RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX + 1e-8)
            .then_some(region)
    }
    /// Create an identity region.  The four-node floor is intentional: callers
    /// may populate a tiny overlap without changing field topology.
    pub(crate) fn identity(overlap: OverlapResidual) -> Self {
        let width = sane_extent(overlap.world.width);
        let height = sane_extent(overlap.world.height);
        let field = Field::new_with_min_nodes(
            width.ceil() as u32,
            height.ceil() as u32,
            f64::from(RESIDUAL_WARP_NODE_STEP_PX),
            MIN_GRID_NODES,
        );
        let mut nodes = Vec::with_capacity(field.width * field.height);
        for row in 0..field.height {
            for column in 0..field.width {
                nodes.push(WarpNode::identity(node_world(
                    &overlap.world,
                    column,
                    row,
                    field.step,
                )));
            }
        }
        Self {
            left_station: overlap.left_station,
            right_station: overlap.right_station,
            world: overlap.world,
            p95_before_px: overlap.p95_px,
            node_step_px: RESIDUAL_WARP_NODE_STEP_PX,
            columns: field.width as u32,
            rows: field.height as u32,
            field,
            nodes,
            reverted_cells: 0,
            insufficient_evidence: false,
            max_node_displacement_px: 0.0,
            max_neighbour_delta_px: 0.0,
        }
    }

    /// Build a region from measurements.  Measurements that fail all three
    /// point gates are ignored before any field value is written.
    pub(crate) fn from_observations(
        overlap: OverlapResidual,
        observations: &[WarpObservation],
    ) -> Self {
        let mut region = Self::identity(overlap);
        if observations.len() < RESIDUAL_WARP_MIN_VERIFIED_POINTS {
            region.insufficient_evidence = true;
            degradation::record_run_degradation(
                degradation::RESIDUAL_WARP_INSUFFICIENT_EVIDENCE,
                serde_json::json!({
                    "left_station": overlap.left_station,
                    "right_station": overlap.right_station,
                    "verified_points": observations.len(),
                    "minimum_points": RESIDUAL_WARP_MIN_VERIFIED_POINTS,
                }),
            );
            return region;
        }

        let mut accepted = observations
            .iter()
            .copied()
            .filter(|observation| observation_is_valid(*observation))
            .collect::<Vec<_>>();
        if accepted.len() < RESIDUAL_WARP_MIN_VERIFIED_POINTS {
            region.insufficient_evidence = true;
            degradation::record_run_degradation(
                degradation::RESIDUAL_WARP_INSUFFICIENT_EVIDENCE,
                serde_json::json!({
                    "left_station": overlap.left_station,
                    "right_station": overlap.right_station,
                    "verified_points": accepted.len(),
                    "minimum_points": RESIDUAL_WARP_MIN_VERIFIED_POINTS,
                }),
            );
            return region;
        }

        // Stable ordering makes the accepted grid independent of the source's
        // parallel iteration order.
        accepted.sort_by(|a, b| {
            a.world
                .y
                .total_cmp(&b.world.y)
                .then_with(|| a.world.x.total_cmp(&b.world.x))
        });
        let mut candidate = vec![None; region.nodes.len()];
        for observation in accepted.iter().copied() {
            let Some(index) = region.nearest_node(observation.world) else {
                continue;
            };
            // If several samples land in one cell, retain the one with the
            // smaller round-trip error; ties are resolved by input order.
            if candidate[index]
                .map(|current: WarpObservation| {
                    observation.round_trip_error_px < current.round_trip_error_px
                })
                .unwrap_or(true)
            {
                candidate[index] = Some(observation);
            }
        }

        for (index, observation) in candidate.iter().enumerate() {
            let Some(observation) = observation else {
                continue;
            };
            region.nodes[index] = WarpNode {
                world: region.nodes[index].world,
                displacement: observation.displacement,
                valid: true,
                round_trip_error_px: observation.round_trip_error_px,
            };
        }
        let cell_columns = region.columns.saturating_sub(1) as usize;
        let cell_rows = region.rows.saturating_sub(1) as usize;
        let mut global_cell_samples = vec![Vec::new(); cell_columns * cell_rows];
        for observation in &accepted {
            if !observation.global_error_px.is_finite() || cell_columns == 0 || cell_rows == 0 {
                continue;
            }
            let Some(index) = region.nearest_node(observation.world) else {
                continue;
            };
            let column = (index % region.field.width).min(cell_columns - 1);
            let row = (index / region.field.width).min(cell_rows - 1);
            let cell = row * cell_columns + column;
            global_cell_samples[cell].push(observation.global_error_px);
        }
        let global_cell_error = global_cell_samples
            .iter_mut()
            .map(|samples| percentile95(samples))
            .collect::<Vec<_>>();
        let unique_valid_nodes = region.nodes.iter().filter(|node| node.valid).count();
        if unique_valid_nodes < RESIDUAL_WARP_MIN_VERIFIED_POINTS {
            region.insufficient_evidence = true;
            degradation::record_run_degradation(
                degradation::RESIDUAL_WARP_INSUFFICIENT_EVIDENCE,
                serde_json::json!({
                    "left_station": overlap.left_station,
                    "right_station": overlap.right_station,
                    "verified_points": unique_valid_nodes,
                    "minimum_points": RESIDUAL_WARP_MIN_VERIFIED_POINTS,
                }),
            );
            return region;
        }
        region.reject_neighbour_spikes();
        if region.nodes.iter().filter(|node| node.valid).count() < RESIDUAL_WARP_MIN_VERIFIED_POINTS
        {
            region.insufficient_evidence = true;
            degradation::record_run_degradation(
                degradation::RESIDUAL_WARP_INSUFFICIENT_EVIDENCE,
                serde_json::json!({
                    "left_station": overlap.left_station,
                    "right_station": overlap.right_station,
                    "verified_points": region.nodes.iter().filter(|node| node.valid).count(),
                    "minimum_points": RESIDUAL_WARP_MIN_VERIFIED_POINTS,
                }),
            );
            return region;
        }
        region.extrapolate_invalid_nodes();
        region.rebuild_field();
        // A producer may know the global correspondence error but not the
        // residual error until the bounded field has been built. Derive that
        // value from the same observation: the residual is the difference
        // between its measured displacement and the displacement actually
        // returned by the constrained field (including edge fade). Explicit
        // finite evidence remains authoritative; legacy callers that provide
        // neither error keep their previous unmeasurable-cell behaviour.
        let mut residual_cell_samples = vec![Vec::new(); cell_columns * cell_rows];
        if cell_columns > 0 && cell_rows > 0 {
            for observation in &accepted {
                let residual_error = if observation.residual_error_px.is_finite() {
                    observation.residual_error_px
                } else if observation.global_error_px.is_finite()
                    && observation
                        .displacement
                        .iter()
                        .all(|value| value.is_finite())
                {
                    let applied = region.displacement_at(observation.world.x, observation.world.y);
                    [
                        observation.displacement[0] - applied[0],
                        observation.displacement[1] - applied[1],
                    ]
                    .into_iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
                } else {
                    continue;
                };
                if !residual_error.is_finite() {
                    continue;
                }
                let Some(index) = region.nearest_node(observation.world) else {
                    continue;
                };
                let column = (index % region.field.width).min(cell_columns - 1);
                let row = (index / region.field.width).min(cell_rows - 1);
                residual_cell_samples[row * cell_columns + column].push(residual_error);
            }
        }
        region.update_metrics();
        let residual_cell_error = residual_cell_samples
            .iter_mut()
            .map(|samples| percentile95(samples))
            .collect::<Vec<_>>();
        region.revert_cells(&global_cell_error, &residual_cell_error);
        region
    }

    /// Number of nodes in the world-space field.
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn nodes(&self) -> &[WarpNode] {
        &self.nodes
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.field
            .values
            .iter()
            .all(|value| value[0] == 0.0 && value[1] == 0.0)
    }

    /// Bilinearly sample the bounded residual at a world point.  The edge
    /// smoothstep is applied after interpolation so values remain zero exactly
    /// on the activation boundary.
    pub(crate) fn displacement_at(&self, x: f64, y: f64) -> [f64; 2] {
        if !x.is_finite() || !y.is_finite() || !contains(&self.world, x, y) {
            return [0.0, 0.0];
        }
        let raw = self.field.at(x - self.world.left, y - self.world.top);
        let fade = edge_fade(&self.world, x, y);
        [raw[0] * fade, raw[1] * fade]
    }

    /// Apply the residual to a world point, returning the point used to sample
    /// the source image.
    pub(crate) fn warp_point(&self, point: WorldPoint) -> WorldPoint {
        let delta = self.displacement_at(point.x, point.y);
        WorldPoint {
            x: point.x + delta[0],
            y: point.y + delta[1],
        }
    }

    /// Revert cells where the measured local error is not lower than the global
    /// homography error.  Both slices use row-major field-cell order.  A NaN
    /// evidence pair is left untouched because it is unmeasurable, not a proof
    /// that the residual is worse.
    pub(crate) fn revert_cells(&mut self, global_error_px: &[f64], residual_error_px: &[f64]) {
        let cell_columns = self.columns.saturating_sub(1) as usize;
        let cell_rows = self.rows.saturating_sub(1) as usize;
        let cell_count = cell_columns * cell_rows;
        for cell in 0..cell_count
            .min(global_error_px.len())
            .min(residual_error_px.len())
        {
            let global = global_error_px[cell];
            let residual = residual_error_px[cell];
            if !global.is_finite() || !residual.is_finite() || residual < global {
                continue;
            }
            let column = cell % cell_columns;
            let row = cell / cell_columns;
            for dy in 0..=1 {
                for dx in 0..=1 {
                    self.nodes[(row + dy) * self.field.width + column + dx].displacement =
                        [0.0, 0.0];
                }
            }
            self.reverted_cells += 1;
            degradation::record_run_degradation(
                degradation::RESIDUAL_WARP_CELL_REVERTED,
                serde_json::json!({
                    "left_station": self.left_station,
                    "right_station": self.right_station,
                    "cell": cell,
                    "global_error_px": global,
                    "residual_error_px": residual,
                }),
            );
        }
        self.enforce_all_neighbour_bound();
        self.rebuild_field();
        self.update_metrics();
    }

    /// Convert this model into the report record consumed by Stack_Report.
    pub(crate) fn report_record(&self) -> WarpRegionRecord {
        WarpRegionRecord {
            world: self.world,
            node_step_px: self.node_step_px,
            columns: self.columns,
            rows: self.rows,
            p95_before_px: self.p95_before_px,
            max_node_displacement_px: self.max_node_displacement_px,
            max_neighbour_delta_px: self.max_neighbour_delta_px,
            reverted_cells: self.reverted_cells,
        }
    }

    pub(super) fn nearest_node(&self, point: WorldPoint) -> Option<usize> {
        if !contains(&self.world, point.x, point.y) {
            return None;
        }
        let column = ((point.x - self.world.left) / self.field.step)
            .round()
            .clamp(0.0, self.field.width as f64 - 1.0) as usize;
        let row = ((point.y - self.world.top) / self.field.step)
            .round()
            .clamp(0.0, self.field.height as f64 - 1.0) as usize;
        Some(row * self.field.width + column)
    }

    pub(super) fn reject_neighbour_spikes(&mut self) {
        // Iterate to a fixed point: rejecting one spike can expose another
        // spike next to it, and the bounded field must satisfy every edge.
        loop {
            let mut rejected = false;
            for row in 0..self.field.height {
                for column in 0..self.field.width {
                    let index = row * self.field.width + column;
                    if !self.nodes[index].valid {
                        continue;
                    }
                    let neighbours =
                        neighbour_indices(row, column, self.field.width, self.field.height);
                    if neighbours.iter().any(|&other| {
                        self.nodes[other].valid
                            && displacement_delta(
                                self.nodes[index].displacement,
                                self.nodes[other].displacement,
                            ) > RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX
                    }) {
                        self.nodes[index].valid = false;
                        self.nodes[index].displacement = [0.0, 0.0];
                        rejected = true;
                    }
                }
            }
            if !rejected {
                break;
            }
        }
    }

    pub(super) fn extrapolate_invalid_nodes(&mut self) {
        let radius = self.field.step * RESIDUAL_WARP_EXTRAPOLATION_RADIUS_NODES;
        let original = self.nodes.clone();
        for row in 0..self.field.height {
            for column in 0..self.field.width {
                let index = row * self.field.width + column;
                if original[index].valid {
                    continue;
                }
                let point = original[index].world;
                let mut weighted = [0.0; 2];
                let mut weight_sum = 0.0;
                for node in &original {
                    if !node.valid {
                        continue;
                    }
                    let distance = (node.world.x - point.x).hypot(node.world.y - point.y);
                    if distance > radius {
                        continue;
                    }
                    let distance_nodes = distance / self.field.step;
                    let weight =
                        (-distance_nodes.powi(2) / (2.0 * GAUSSIAN_SIGMA_NODES.powi(2))).exp();
                    weighted[0] += node.displacement[0] * weight;
                    weighted[1] += node.displacement[1] * weight;
                    weight_sum += weight;
                }
                if weight_sum > 0.0 {
                    self.nodes[index].displacement =
                        [weighted[0] / weight_sum, weighted[1] / weight_sum];
                } else {
                    self.nodes[index].displacement = [0.0, 0.0];
                }
            }
        }
        self.enforce_invalid_neighbour_bound();
    }

    /// Invalid-node extrapolation is itself subject to the same 32/8 gates as
    /// measured nodes. Keep measured nodes fixed and pull only extrapolated
    /// nodes towards their neighbours; samples with no support remain zero.
    pub(super) fn enforce_invalid_neighbour_bound(&mut self) {
        // Keep the support set fixed.  Using an already extrapolated invalid
        // node as a neighbour would propagate one valid sample arbitrarily far
        // across the grid and violate the three-node-radius requirement.
        let snapshot = self.nodes.clone();
        let width = self.field.width;
        let height = self.field.height;
        for index in 0..self.nodes.len() {
            if snapshot[index].valid {
                continue;
            }
            let node = &mut self.nodes[index];
            let magnitude = node.displacement[0].hypot(node.displacement[1]);
            if magnitude > RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX {
                let scale = RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX / magnitude;
                node.displacement[0] *= scale;
                node.displacement[1] *= scale;
            }
            let row = index / width;
            let column = index % width;
            for other in neighbour_indices(row, column, width, height) {
                if !snapshot[other].valid {
                    continue;
                }
                let displacement = snapshot[other].displacement;
                let delta = displacement_delta(node.displacement, displacement);
                if delta <= RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX {
                    continue;
                }
                let direction = [
                    node.displacement[0] - displacement[0],
                    node.displacement[1] - displacement[1],
                ];
                let scale = RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX / delta;
                node.displacement = [
                    displacement[0] + direction[0] * scale,
                    displacement[1] + direction[1] * scale,
                ];
            }
        }
    }

    /// Reverted cells are zeroed global-homography cells. Pull any adjacent
    /// high residual toward the zero fallback so the 8px neighbour gate remains
    /// true at the boundary of a reverted cell.
    pub(super) fn enforce_all_neighbour_bound(&mut self) {
        for _ in 0..self.nodes.len().max(1) {
            let mut changed = false;
            for row in 0..self.field.height {
                for column in 0..self.field.width {
                    let index = row * self.field.width + column;
                    for other in neighbour_indices(row, column, self.field.width, self.field.height)
                    {
                        if other <= index {
                            continue;
                        }
                        let delta = displacement_delta(
                            self.nodes[index].displacement,
                            self.nodes[other].displacement,
                        );
                        // Projection onto the bound can round a few ulps
                        // above 8. Re-projecting that same pair indefinitely
                        // never changes the field and makes large grids stall.
                        if delta <= RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX + 1e-10 {
                            continue;
                        }
                        let index_magnitude = self.nodes[index].displacement[0]
                            .hypot(self.nodes[index].displacement[1]);
                        let other_magnitude = self.nodes[other].displacement[0]
                            .hypot(self.nodes[other].displacement[1]);
                        let (high, low) = if index_magnitude >= other_magnitude {
                            (index, other)
                        } else {
                            (other, index)
                        };
                        let direction = [
                            self.nodes[high].displacement[0] - self.nodes[low].displacement[0],
                            self.nodes[high].displacement[1] - self.nodes[low].displacement[1],
                        ];
                        let scale = RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX / delta;
                        self.nodes[high].displacement = [
                            self.nodes[low].displacement[0] + direction[0] * scale,
                            self.nodes[low].displacement[1] + direction[1] * scale,
                        ];
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    pub(super) fn rebuild_field(&mut self) {
        self.field.values.fill([0.0, 0.0]);
        for row in 0..self.field.height {
            for column in 0..self.field.width {
                let index = row * self.field.width + column;
                self.field.values[index] = [
                    self.nodes[index].displacement[0],
                    self.nodes[index].displacement[1],
                ];
            }
        }
    }

    pub(super) fn update_metrics(&mut self) {
        self.max_node_displacement_px = self
            .nodes
            .iter()
            .map(|node| node.displacement[0].hypot(node.displacement[1]))
            .fold(0.0, f64::max);
        let width = self.field.width;
        let height = self.field.height;
        let mut max_neighbour_delta: f64 = 0.0;
        for row in 0..height {
            for column in 0..width {
                let index = row * width + column;
                let displacement = self.nodes[index].displacement;
                for other in neighbour_indices(row, column, width, height) {
                    if other > index {
                        max_neighbour_delta = max_neighbour_delta.max(displacement_delta(
                            displacement,
                            self.nodes[other].displacement,
                        ));
                    }
                }
            }
        }
        self.max_neighbour_delta_px = max_neighbour_delta;
    }
}

pub(super) fn observation_is_valid(observation: WarpObservation) -> bool {
    observation.world.x.is_finite()
        && observation.world.y.is_finite()
        && observation
            .displacement
            .iter()
            .all(|value| value.is_finite())
        && observation.displacement[0].hypot(observation.displacement[1])
            <= RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX
        && observation.round_trip_error_px.is_finite()
        && observation.round_trip_error_px <= RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX
}

pub(super) fn node_world(world: &WorldRect, column: usize, row: usize, step: f64) -> WorldPoint {
    WorldPoint {
        x: world.left + column as f64 * step,
        y: world.top + row as f64 * step,
    }
}

pub(super) fn neighbour_indices(
    row: usize,
    column: usize,
    width: usize,
    height: usize,
) -> Vec<usize> {
    let mut result = Vec::with_capacity(4);
    if row > 0 {
        result.push((row - 1) * width + column);
    }
    if column > 0 {
        result.push(row * width + column - 1);
    }
    if row + 1 < height {
        result.push((row + 1) * width + column);
    }
    if column + 1 < width {
        result.push(row * width + column + 1);
    }
    result
}

pub(super) fn displacement_delta(left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - right[0]).hypot(left[1] - right[1])
}

pub(super) fn percentile95(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let rank = (values.len() * 95).div_ceil(100).max(1);
    let index = rank.saturating_sub(1).min(values.len() - 1);
    values[index]
}

pub(super) fn edge_fade(world: &WorldRect, x: f64, y: f64) -> f64 {
    let width = sane_extent(world.width);
    let height = sane_extent(world.height);
    let distance = (x - world.left)
        .min(world.left + width - x)
        .min(y - world.top)
        .min(world.top + height - y);
    let t = (distance / RESIDUAL_WARP_EDGE_FADE_PX).clamp(0.0, 1.0);
    // Smoothstep gives a zero derivative at both ends and avoids a visible
    // crease where the local field meets the global homography.
    t * t * (3.0 - 2.0 * t)
}
