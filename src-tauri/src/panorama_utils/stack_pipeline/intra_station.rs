//! Intra_Station_Registrar: anchor frame selection, the two-level registration
//! budget and the control-point gates of 需求 2.1–2.9.
//!
//! The registrar itself is not a new parallel pipeline: the measurement work
//! lives in [`super::super::mosaic::refine_native_layer`] and
//! [`super::super::registration::refine_warped_patch`].  This module owns the
//! *decisions* — which frame is the anchor, at which resolution the two global
//! and local rounds run, which control point survives, and when a whole frame
//! falls back to its global model — so each rule is a small deterministic
//! function a property test can drive without decoding a RAW file.

use std::sync::Mutex;

use image::{GrayImage, Rgb};

use super::super::registration::PatchMatchGates;
use super::report::{
    IntraStationFrameRecord, IntraStationFrameStatus, IntraStationLocalFieldVerdict,
    IntraStationReport,
};

// Configuration and patch-matching budget.
include!("intra_station/config.inc.rs");
// Focal-plane matched low-pass and sharpness parity.
include!("intra_station/focal_plane.inc.rs");
// Control-point acceptance gates and anchor analysis.
include!("intra_station/gates.inc.rs");
// Refinement-grid data structure and deterministic medians.
include!("intra_station/grid.inc.rs");
// Run-scoped Stack_Report observations.
include!("intra_station/observations.inc.rs");

#[cfg(test)]
include!("intra_station/tests.inc.rs");
