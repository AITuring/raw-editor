//! Facade for the focus-stack station pose solver.
//!
//! The implementation is split by responsibility into geometry/topology,
//! station graph validation, and evidence/closure solving modules. Re-exports
//! keep the parent `panorama_stitching` API and test accessors stable.

use super::*;

#[path = "panorama_focus_pose_geometry.rs"]
mod geometry;
pub(super) use geometry::*;

#[path = "panorama_focus_station_graph.rs"]
mod station_graph;
pub(super) use station_graph::*;

#[path = "panorama_focus_station_solver.rs"]
mod station_solver;
pub(super) use station_solver::*;
