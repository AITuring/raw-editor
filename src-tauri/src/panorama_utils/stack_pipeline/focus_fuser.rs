//! Focus_Fuser: the ownership-cell decision metrics of the intra-station focus
//! fusion (需求 3.1, 3.2, 3.3, 3.9).
//!
//! # Why the metrics live here and not in `mosaic.rs`
//!
//! The task text points at `mosaic.rs`, but the default delivery path no longer
//! passes through the mosaic's ownership grid.  Since task 13.1 the default
//! compositor is `stitching::layered_virtual_tile_compositor`, which renders one
//! virtual tile per Capture_Station through `focus_stack_stitcher_unfilled` →
//! `focus_stack_stitcher_with_margin_policy`.  That function only hands over to
//! `mosaic::detail_preserving_mosaic` when `shifted_mosaic` is true, and a
//! virtual tile holds exactly one station by construction, so
//! `capture_group_count == 1` and `shifted_mosaic` is always false — a real run
//! logs "Compact focus capture detected (1 inferred group(s)); preserving
//! standard focus fusion" once per station.  `mosaic::acutance`,
//! `mosaic::cell_focus`, `mosaic::ownership_disagreement` and
//! `mosaic::ownership_grid` therefore only serve the legacy comparison path.
//!
//! This module reuses those measurements as *decision primitives*
//! ([`super::super::mosaic::acutance_with_step`],
//! [`super::super::mosaic::cell_focus_with_step`],
//! [`super::super::mosaic::cell_low_pass_differences`], and later
//! `seam_cut::cut_grid`) without calling `detail_preserving_mosaic` as a whole:
//! that entry point also applies a cell-wise local warp, which
//! `focus_stack_stitcher_with_margin_policy` already records as visibly
//! fragmenting strokes that are intentionally present at different focal depths.
//! The legacy reductions in `mosaic.rs` keep their old behaviour; every new rule
//! 需求 3 prescribes is a separate function here.
//!
//! Scope: tasks 7.6 and 7.7 build the metrics and the grid geometry, tasks 7.8
//! and 7.9 the multi-label graph cut over those costs together with its timeout
//! fallback and single-frame shortcut, and tasks 7.10 and 7.11 the hard
//! ownership decision the renderer consumes plus Sharpness_Confidence and the
//! low sharpness regions.  [`StationFusion`] is the entry point the production
//! focus renderer drives; see its documentation for why it folds one candidate
//! per exact two-label cut instead of refining a full multi-label grid.

// `solve_ownership` and the refinement sweeps around it are the whole-grid form
// of the same energy: they need every candidate's costs at once, which the
// render loop cannot afford (see `StationFusion`), so they are exercised by the
// tests and by the exhaustive-minimum comparison rather than by the renderer.
#![allow(dead_code)]

use std::sync::Mutex;
use std::time::{Duration, Instant};

use image::Rgb;
use serde_json::json;

use super::super::mosaic::{
    ACUTANCE_PATCH_POINTS, OWNERSHIP_MISMATCH_PENALTY, SELECTION_LONG_SIDE, acutance_with_step,
    cell_focus_with_step, cell_low_pass_differences, cell_probe_offsets,
};
use super::super::seam_cut::{cut_grid_weighted, pairwise_weight};
use super::degradation::{DegradationLedger, FUSION_GRAPH_CUT_TIMEOUT};
use super::report::{
    FusionReport, FusionSolverStatus, LowSharpnessRegionRecord, OwnershipGridSize,
    SharpnessConfidenceSummary, WorldRect,
};

// Geometry and sampling contracts shared by solver and station driver.
include!("focus_fuser/geometry.inc.rs");
// Cost grid, graph-cut refinement, timeout, and deterministic fallback.
include!("focus_fuser/solver.inc.rs");
// Confidence, low-sharpness regions, and per-cell evidence surfaces.
include!("focus_fuser/evidence.inc.rs");
// Run-scoped report observations.
include!("focus_fuser/observations.inc.rs");

#[cfg(test)]
include!("focus_fuser/tests.inc.rs");
