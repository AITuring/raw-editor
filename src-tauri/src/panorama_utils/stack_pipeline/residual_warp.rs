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

use std::{collections::BTreeMap, sync::Mutex};

use image::GrayImage;
use nalgebra::{Matrix3, Point2};
use serde::{Deserialize, Serialize};

use super::super::registration::{PatchMatchGates, probe_warped_patch};
#[cfg(test)]
use super::degradation;
use super::report::{ResidualWarpReport, WarpRegionRecord, WorldPoint, WorldRect};

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

mod warp_region;

pub(crate) use warp_region::WarpRegion;

/// The complete residual model for a run.  `Default` intentionally starts
/// with an empty region list so old callers preserve their exact behaviour.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResidualWarp {
    pub regions: Vec<WarpRegion>,
    pub(crate) line_layers: super::line_layers::LineLayers,
    station_aliases: BTreeMap<usize, usize>,
}

impl ResidualWarp {
    pub(crate) fn register_station_alias(&mut self, image_id: usize, station: usize) {
        self.station_aliases.insert(image_id, station);
    }
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
        self.line_layers.reports.is_empty()
            && self.regions.iter().all(|region| {
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
        let station_id = self
            .station_aliases
            .get(&station_id)
            .copied()
            .unwrap_or(station_id);
        let world = self.line_layers.inverse(world, station_id);
        let Some(region) = self.regions.iter().find(|region| {
            region.right_station == station_id && contains(&region.world, world.x, world.y)
        }) else {
            return world;
        };
        let mut source = world;
        // The 32/8 grid and edge fade can approach a 0.625 contraction.
        // Four steps are not subpixel-accurate there; sixteen bound the
        // remaining inverse error below 0.02px even at maximum displacement.
        for _ in 0..16 {
            let displacement = region.displacement_at(source.x, source.y);
            let next = Point2::new(world.x - displacement[0], world.y - displacement[1]);
            if (next - source).norm() < 1e-6 {
                return next;
            }
            source = next;
        }
        source
    }

    pub(crate) fn forward_world_for_station(
        &self,
        source: Point2<f64>,
        image_id: usize,
    ) -> Point2<f64> {
        let station = self
            .station_aliases
            .get(&image_id)
            .copied()
            .unwrap_or(image_id);
        self.regions
            .iter()
            .find(|r| r.right_station == station && contains(&r.world, source.x, source.y))
            .map_or(source, |r| {
                let d = r.displacement_at(source.x, source.y);
                Point2::new(source.x + d[0], source.y + d[1])
            })
    }
    pub(crate) fn has_line_layer(&self, image_id: usize) -> bool {
        let station = self
            .station_aliases
            .get(&image_id)
            .copied()
            .unwrap_or(image_id);
        self.line_layers.has_station(station)
    }
    pub(crate) fn line_layer_affects(&self, world: Point2<f64>, image_id: usize) -> bool {
        let station = self
            .station_aliases
            .get(&image_id)
            .copied()
            .unwrap_or(image_id);
        self.line_layers.affects(world, station)
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

/// Replace the complete compositor model without duplicating its existing
/// camera-field records when a separately validated depth layer is attached.
pub(crate) fn replace_run_model(model: &ResidualWarp) {
    *RUN_RESIDUAL_WARP_MODEL
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = Some(model.clone());
    *RUN_RESIDUAL_WARP.lock().unwrap_or_else(|p| p.into_inner()) = model.report_records();
    *RUN_RESIDUAL_WARP_IDENTITY
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = model.identity();
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
        normal_layers: run_model_snapshot().line_layers.reports,
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
    verify_round_trip_match_with_template(
        source,
        target,
        forward_transform,
        backward_transform,
        center,
        9,
    )
}

pub(crate) fn verify_round_trip_match_with_template(
    source: &GrayImage,
    target: &GrayImage,
    forward_transform: &Matrix3<f64>,
    backward_transform: &Matrix3<f64>,
    center: Point2<f64>,
    half: i32,
) -> Option<RoundTripMatch> {
    verify_round_trip_match_with_gates(
        source,
        target,
        forward_transform,
        backward_transform,
        center,
        half,
        PatchMatchGates::LEGACY,
    )
}

pub(crate) fn verify_round_trip_match_with_gates(
    source: &GrayImage,
    target: &GrayImage,
    forward_transform: &Matrix3<f64>,
    backward_transform: &Matrix3<f64>,
    center: Point2<f64>,
    half: i32,
    gates: PatchMatchGates,
) -> Option<RoundTripMatch> {
    let forward_match =
        probe_warped_patch(source, target, forward_transform, center, half, 7, gates).matched?;
    let backward_match = probe_warped_patch(
        target,
        source,
        backward_transform,
        forward_match.target,
        half,
        7,
        gates,
    )
    .matched?;
    let round_trip_error_px = (backward_match.target - center).norm();
    (round_trip_error_px <= RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX).then_some(RoundTripMatch {
        forward: forward_match.target,
        backward: backward_match.target,
        round_trip_error_px,
    })
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
    fn virtual_tile_alias_and_source_station_use_the_same_inverse_field() {
        let world = WorldPoint { x: 356.0, y: 456.0 };
        let observations = (0..8)
            .flat_map(|row| {
                (0..8).map(move |column| {
                    WarpObservation::new(
                        WorldPoint {
                            x: 100.0 + column as f64 * 64.0,
                            y: 200.0 + row as f64 * 64.0,
                        },
                        [4.0, -3.0],
                        0.2,
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut model = ResidualWarp::new();
        model
            .add_observations(overlap(448.0, 448.0, 10.0), &observations)
            .unwrap();
        model.register_station_alias(usize::MAX - 7, 7);
        let warped = model.regions[0].warp_point(world);
        let target = Point2::new(warped.x, warped.y);
        let source = model.inverse_world_for_station(target, 7);
        assert!((source - Point2::new(world.x, world.y)).norm() < 0.01);
        assert_eq!(
            source,
            model.inverse_world_for_station(target, usize::MAX - 7)
        );
        assert_eq!(target, model.inverse_world_for_station(target, 6));
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
    fn missing_residual_error_is_derived_after_field_constraints() {
        let _run_scope = degradation::begin_run_scope();
        let overlap = overlap(192.0, 192.0, 4.0);
        let observations = (0..4)
            .flat_map(|row| {
                (0..4).map(move |column| {
                    WarpObservation::with_errors(
                        WorldPoint {
                            x: 100.0 + column as f64 * 64.0,
                            y: 200.0 + row as f64 * 64.0,
                        },
                        [4.0, 0.0],
                        0.1,
                        4.0,
                        f64::NAN,
                    )
                })
            })
            .collect::<Vec<_>>();
        let region = WarpRegion::from_observations(overlap, &observations);
        assert!(!region.insufficient_evidence);
        assert!(region.reverted_cells > 0);
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
