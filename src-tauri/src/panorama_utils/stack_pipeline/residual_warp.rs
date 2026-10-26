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

// Overlap, observation, node and region data types.
include!("residual_warp/types.inc.rs");
// Bounded residual field construction, interpolation and cell gates.
include!("residual_warp/region.inc.rs");
// Run-level residual model and inverse world mapping.
include!("residual_warp/model.inc.rs");
// Run-scoped report/model observation sink.
include!("residual_warp/runtime.inc.rs");
// Round-trip matching and shared geometry helpers.
include!("residual_warp/matching.inc.rs");

#[cfg(test)]
include!("residual_warp/tests.inc.rs");
