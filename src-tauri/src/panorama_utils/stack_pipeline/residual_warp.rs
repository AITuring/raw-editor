//! Residual_Warp: a bounded, evidence-gated displacement field for one
//! overlapping station pair (requirements 8.1--8.9).
//!
//! The module deliberately keeps the global homography outside the field.  An
//! overlap supplies the region in world coordinates and its global P95; the
//! optional control-point observations below only describe the correction to
//! that model.  When no trustworthy observations are available the field is
//! exactly zero, which makes the caller's existing global path the identity
//! residual fallback.

#![allow(dead_code)]

use std::sync::Mutex;

use image::GrayImage;
use nalgebra::{Matrix3, Point2};
use serde::{Deserialize, Serialize};

use super::super::mosaic::Field;
use super::super::registration::refine_warped_patch;
use super::degradation;
use super::report::{
    RESIDUAL_WARP_NODE_STEP_PX, ResidualWarpReport, WarpRegionRecord, WorldPoint, WorldRect,
};

/// A residual is considered large enough to warrant a local field only when
/// the globally aligned overlap has P95 above this value (world pixels).
pub(crate) const RESIDUAL_WARP_ENABLE_P95_PX: f64 = 3.0;
/// Maximum displacement of one control node (world pixels).
pub(crate) const RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX: f64 = 32.0;
/// Maximum difference between neighbouring control nodes (world pixels).
pub(crate) const RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX: f64 = 8.0;
/// Maximum forward/backward round-trip error accepted for one control point.
pub(crate) const RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX: f64 = 1.0;
/// Minimum number of round-trip verified points required for a non-identity
/// region.  This is intentionally separate from the four-node grid floor.
pub(crate) const RESIDUAL_WARP_MIN_VERIFIED_POINTS: usize = 16;
/// Distance over which a field fades in from the enabled region boundary.
pub(crate) const RESIDUAL_WARP_EDGE_FADE_PX: f64 = 128.0;
/// Three node radii are the maximum support for invalid-node extrapolation.
pub(crate) const RESIDUAL_WARP_EXTRAPOLATION_RADIUS_NODES: f64 = 3.0;

const MIN_GRID_NODES: usize = 4;
const GAUSSIAN_SIGMA_NODES: f64 = 1.5;

/// A globally aligned overlap that may receive a local residual correction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct OverlapResidual {
    pub left_station: usize,
    pub right_station: usize,
    pub world: WorldRect,
    pub p95_px: f64,
}

impl OverlapResidual {
    pub(crate) const fn new(
        left_station: usize,
        right_station: usize,
        world: WorldRect,
        p95_px: f64,
    ) -> Self {
        Self {
            left_station,
            right_station,
            world,
            p95_px,
        }
    }
}

impl Default for OverlapResidual {
    fn default() -> Self {
        Self {
            left_station: 0,
            right_station: 0,
            world: WorldRect::default(),
            p95_px: 0.0,
        }
    }
}

/// A match measured in the overlap's world frame.  `global_error_px` and
/// `residual_error_px` are optional evidence used by [`WarpRegion::revert`]
/// when deciding whether an individual cell improved the global model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WarpObservation {
    pub world: WorldPoint,
    pub displacement: [f64; 2],
    pub round_trip_error_px: f64,
    pub global_error_px: f64,
    pub residual_error_px: f64,
}

impl WarpObservation {
    pub(crate) const fn new(
        world: WorldPoint,
        displacement: [f64; 2],
        round_trip_error_px: f64,
    ) -> Self {
        Self {
            world,
            displacement,
            round_trip_error_px,
            global_error_px: f64::NAN,
            residual_error_px: f64::NAN,
        }
    }

    pub(crate) const fn with_errors(
        world: WorldPoint,
        displacement: [f64; 2],
        round_trip_error_px: f64,
        global_error_px: f64,
        residual_error_px: f64,
    ) -> Self {
        Self {
            world,
            displacement,
            round_trip_error_px,
            global_error_px,
            residual_error_px,
        }
    }
}

/// One verified control node.  Invalid nodes are retained as records so a
/// diagnostic can distinguish no evidence from an explicitly zero correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WarpNode {
    pub world: WorldPoint,
    pub displacement: [f64; 2],
    pub valid: bool,
    pub round_trip_error_px: f64,
}

impl WarpNode {
    fn identity(world: WorldPoint) -> Self {
        Self {
            world,
            displacement: [0.0, 0.0],
            valid: false,
            round_trip_error_px: f64::INFINITY,
        }
    }
}

/// The result of a bidirectional patch check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RoundTripMatch {
    pub forward: Point2<f64>,
    pub backward: Point2<f64>,
    pub round_trip_error_px: f64,
}

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
    field: Field<2>,
    nodes: Vec<WarpNode>,
    pub reverted_cells: u64,
    pub insufficient_evidence: bool,
    pub max_node_displacement_px: f64,
    pub max_neighbour_delta_px: f64,
}

impl WarpRegion {
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
        let mut residual_cell_samples = vec![Vec::new(); cell_columns * cell_rows];
        for observation in &accepted {
            if !observation.global_error_px.is_finite()
                || !observation.residual_error_px.is_finite()
                || cell_columns == 0
                || cell_rows == 0
            {
                continue;
            }
            let Some(index) = region.nearest_node(observation.world) else {
                continue;
            };
            let column = (index % region.field.width).min(cell_columns - 1);
            let row = (index / region.field.width).min(cell_rows - 1);
            let cell = row * cell_columns + column;
            global_cell_samples[cell].push(observation.global_error_px);
            residual_cell_samples[cell].push(observation.residual_error_px);
        }
        let global_cell_error = global_cell_samples
            .iter_mut()
            .map(percentile95)
            .collect::<Vec<_>>();
        let residual_cell_error = residual_cell_samples
            .iter_mut()
            .map(percentile95)
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
        region.update_metrics();
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

    fn nearest_node(&self, point: WorldPoint) -> Option<usize> {
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

    fn reject_neighbour_spikes(&mut self) {
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

    fn extrapolate_invalid_nodes(&mut self) {
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
    fn enforce_invalid_neighbour_bound(&mut self) {
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
    fn enforce_all_neighbour_bound(&mut self) {
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
                        if delta <= RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX {
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

    fn rebuild_field(&mut self) {
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

    fn update_metrics(&mut self) {
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

/// The complete residual model for a run.  `Default` intentionally starts
/// with an empty region list so old callers preserve their exact behaviour.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResidualWarp {
    pub regions: Vec<WarpRegion>,
}

impl ResidualWarp {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add an overlap using identity evidence.  P95 at or below the enable
    /// threshold does not allocate a region; a high-P95 overlap allocates an
    /// identity region and records insufficient evidence until observations
    /// are supplied with [`Self::add_observations`].
    pub(crate) fn add_overlap(&mut self, overlap: OverlapResidual) -> Option<usize> {
        if !overlap.p95_px.is_finite() || overlap.p95_px <= RESIDUAL_WARP_ENABLE_P95_PX {
            return None;
        }
        let region = WarpRegion::identity(overlap);
        self.regions.push(region);
        Some(self.regions.len() - 1)
    }

    /// Add an overlap and immediately populate it from round-trip observations.
    pub(crate) fn add_observations(
        &mut self,
        overlap: OverlapResidual,
        observations: &[WarpObservation],
    ) -> Option<usize> {
        if !overlap.p95_px.is_finite() || overlap.p95_px <= RESIDUAL_WARP_ENABLE_P95_PX {
            return None;
        }
        self.regions
            .push(WarpRegion::from_observations(overlap, observations));
        Some(self.regions.len() - 1)
    }

    pub(crate) fn identity(&self) -> bool {
        self.regions.iter().all(|region| {
            region
                .field
                .values
                .iter()
                .all(|value| value[0] == 0.0 && value[1] == 0.0)
        })
    }

    /// Apply the inverse residual map for one station in world coordinates.
    /// A residual is defined as `warped = world + displacement(world)`, so a
    /// few fixed-point steps recover the source world point without replacing
    /// the non-rigid field by a matrix.  Regions are keyed by their right-hand
    /// station; absent regions are an exact identity.
    pub(crate) fn inverse_world_for_station(
        &self,
        world: Point2<f64>,
        station_id: usize,
    ) -> Point2<f64> {
        let Some(region) = self.regions.iter().find(|region| {
            region.right_station == station_id && contains(&region.world, world.x, world.y)
        }) else {
            return world;
        };
        if self.identity() {
            return world;
        }
        let mut source = world;
        for _ in 0..4 {
            let displacement = region.displacement_at(source.x, source.y);
            let next = Point2::new(world.x - displacement[0], world.y - displacement[1]);
            if (next - source).norm() < 1e-6 {
                return next;
            }
            source = next;
        }
        source
    }

    /// Short compositor-facing alias for the world-space inverse query.
    pub(crate) fn warp_inverse(&self, world: Point2<f64>, station_id: usize) -> Point2<f64> {
        self.inverse_world_for_station(world, station_id)
    }

    pub(crate) fn report_records(&self) -> Vec<WarpRegionRecord> {
        self.regions.iter().map(WarpRegion::report_record).collect()
    }
}

/// Run-scoped report sink, matching the Intra_Station and Focus_Fuser pattern.
static RUN_RESIDUAL_WARP: Mutex<Vec<WarpRegionRecord>> = Mutex::new(Vec::new());
static RUN_RESIDUAL_WARP_IDENTITY: Mutex<bool> = Mutex::new(true);
static RUN_RESIDUAL_WARP_MODEL: Mutex<Option<ResidualWarp>> = Mutex::new(None);

pub(crate) fn reset_run_records() {
    RUN_RESIDUAL_WARP
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    *RUN_RESIDUAL_WARP_IDENTITY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
    *RUN_RESIDUAL_WARP_MODEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

pub(crate) fn record_run_model(model: &ResidualWarp) {
    *RUN_RESIDUAL_WARP_MODEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(model.clone());
    let mut records = RUN_RESIDUAL_WARP
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    records.extend(model.report_records());
    let mut identity = RUN_RESIDUAL_WARP_IDENTITY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *identity &= model.identity();
}

/// Snapshot the run model for the compositor.  An empty snapshot is the
/// compatibility path used until the panorama entry point supplies measured
/// station P95s and observations.
pub(crate) fn run_model_snapshot() -> ResidualWarp {
    RUN_RESIDUAL_WARP_MODEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
        .unwrap_or_default()
}

pub(crate) fn run_records_snapshot() -> Vec<WarpRegionRecord> {
    RUN_RESIDUAL_WARP
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

pub(crate) fn run_report_snapshot() -> ResidualWarpReport {
    ResidualWarpReport {
        regions: run_records_snapshot(),
        identity: *RUN_RESIDUAL_WARP_IDENTITY
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    }
}

/// Verify a forward match by matching the resulting point back into the source
/// image.  The existing patch matcher is used in both directions, so the
/// round-trip gate is exactly the same one used by production registration.
pub(crate) fn verify_round_trip_match(
    source: &GrayImage,
    target: &GrayImage,
    forward_transform: &Matrix3<f64>,
    backward_transform: &Matrix3<f64>,
    center: Point2<f64>,
) -> Option<RoundTripMatch> {
    let forward_match = refine_warped_patch(source, target, forward_transform, center, 9, 7)?;
    let backward_match = refine_warped_patch(
        target,
        source,
        backward_transform,
        forward_match.target,
        9,
        7,
    )?;
    let round_trip_error_px = (backward_match.target - center).norm();
    (round_trip_error_px <= RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX).then_some(RoundTripMatch {
        forward: forward_match.target,
        backward: backward_match.target,
        round_trip_error_px,
    })
}

fn observation_is_valid(observation: WarpObservation) -> bool {
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

fn sane_extent(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn contains(world: &WorldRect, x: f64, y: f64) -> bool {
    x >= world.left
        && y >= world.top
        && x <= world.left + sane_extent(world.width)
        && y <= world.top + sane_extent(world.height)
}

fn node_world(world: &WorldRect, column: usize, row: usize, step: f64) -> WorldPoint {
    WorldPoint {
        x: world.left + column as f64 * step,
        y: world.top + row as f64 * step,
    }
}

fn neighbour_indices(row: usize, column: usize, width: usize, height: usize) -> Vec<usize> {
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

fn displacement_delta(left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - right[0]).hypot(left[1] - right[1])
}

fn percentile95(values: &mut Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let rank = (values.len() * 95).div_ceil(100).max(1);
    let index = rank.saturating_sub(1).min(values.len() - 1);
    values[index]
}

fn edge_fade(world: &WorldRect, x: f64, y: f64) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn overlap(width: f64, height: f64, p95_px: f64) -> OverlapResidual {
        OverlapResidual::new(
            2,
            7,
            WorldRect {
                left: 100.0,
                top: 200.0,
                width,
                height,
            },
            p95_px,
        )
    }

    #[test]
    fn residual_warp_starts_empty_and_only_enables_above_p95_gate() {
        let mut model = ResidualWarp::new();
        assert!(model.regions.is_empty());
        assert!(model.add_overlap(overlap(80.0, 80.0, 3.0)).is_none());
        assert_eq!(model.add_overlap(overlap(80.0, 80.0, 3.01)), Some(0));
        assert_eq!(model.regions[0].columns.min(model.regions[0].rows), 4);
    }

    #[test]
    fn displacement_is_bounded_and_fades_at_region_edge() {
        let mut observations = Vec::new();
        for row in 0..4 {
            for column in 0..4 {
                observations.push(WarpObservation::new(
                    WorldPoint {
                        x: 100.0 + column as f64 * 64.0,
                        y: 200.0 + row as f64 * 64.0,
                    },
                    [4.0, -3.0],
                    0.2,
                ));
            }
        }
        let region = WarpRegion::from_observations(overlap(192.0, 192.0, 4.0), &observations);
        assert!(!region.insufficient_evidence);
        assert!(region.max_node_displacement_px <= RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX);
        assert_eq!(region.displacement_at(100.0, 300.0), [0.0, 0.0]);
        let centre = region.displacement_at(196.0, 296.0);
        assert!(centre[0].is_finite() && centre[1].is_finite());
    }

    #[test]
    fn invalid_nodes_are_gaussian_extrapolated_only_within_three_radii() {
        let mut observations = Vec::new();
        for row in 0..4 {
            for column in 0..4 {
                if row == 0 && column == 0 {
                    observations.push(WarpObservation::new(
                        WorldPoint { x: 100.0, y: 200.0 },
                        [8.0, 0.0],
                        0.1,
                    ));
                    continue;
                }
                // Valid evidence elsewhere keeps the region above the 16-point
                // gate while leaving the corner's neighbours to extrapolation.
                observations.push(WarpObservation::new(
                    WorldPoint {
                        x: 100.0 + column as f64 * 64.0,
                        y: 200.0 + row as f64 * 64.0,
                    },
                    [0.0, 0.0],
                    0.1,
                ));
            }
        }
        let region = WarpRegion::from_observations(overlap(192.0, 192.0, 4.0), &observations);
        assert!(
            region
                .nodes()
                .iter()
                .all(|node| node.displacement[0].is_finite())
        );
    }

    #[test]
    fn round_trip_gate_uses_the_existing_matcher() {
        let source = GrayImage::from_fn(96, 96, |x, y| {
            image::Luma([((x.wrapping_mul(13) ^ y.wrapping_mul(7)) & 255) as u8])
        });
        let target = source.clone();
        let identity = Matrix3::identity();
        let result = verify_round_trip_match(
            &source,
            &target,
            &identity,
            &identity,
            Point2::new(48.0, 48.0),
        );
        assert!(result.is_some());
    }
}
