//! Focus mosaics: refine residual geometry, choose sharp source detail, match
//! only broad colour. The output is never an average of displaced fine detail.
use super::registration::{PatchMatchGates, probe_warped_patch, refine_warped_patch};
use super::seam_cut;
use super::stack_pipeline::degradation;
use super::stack_pipeline::intra_station;
use super::stack_pipeline::report::{IntraStationFrameRecord, IntraStationFrameStatus};
use super::stack_pipeline::station_degradation::{
    GroupJoinEvidence, StationMember, group_join_rejection, plan_station_fusion,
    record_run_entries, record_run_group_join_rejection, station_plan_entries,
};
use super::stitching::{
    Projection, downsample_rgb_half, get_high_quality_interpolated_pixel, map_target_to_source,
    output_bounds, pixel_aligned_canvas, transformed_image_region,
};
use crate::panorama_stitching::ImageInfo;
use image::{GrayImage, Luma, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Runtime};

#[cfg(test)]
#[path = "mosaic_diagnostics.rs"]
mod diagnostics;

// Actual configured intra-station analysis long side. Keep both compositor
// paths on the same source of truth so Stack_Report records what production
// used rather than the 2048 requirement ceiling (需求 2.2).
const ANALYSIS_LONG_SIDE: u32 = intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE;
// Bound the grid per source layer, not by the full panorama width. At a
// roughly 9504px layer footprint this yields approximately 19px cells, fine
// enough for seams to route around individual strokes.
// Also the Focus_Fuser's ownership cell count (需求 3.1): the station plane's
// long side is divided into at least this many cells.
pub(super) const SELECTION_LONG_SIDE: u32 = 512;
const LONG_SEQUENCE_SELECTION_LONG_SIDE: u32 = 768;
// Streaming still makes one continuous decision for the whole source layer.
// Keeping the analysis bounded preserves memory while preventing independent
// 1024px tiles from choosing incompatible owners along their shared border.
const STREAMING_ANALYSIS_LONG_SIDE: u32 = intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE;
const STREAMING_SELECTION_LONG_SIDE: u32 = 768;
// Ownership is already regularised by the graph cut. A wide blur here mixes
// focused and defocused focal planes across tens of native pixels on a 45MP
// source. Keep only a sub-cell antialiasing transition.
const STREAMING_OWNERSHIP_FEATHER: f32 = 0.65;
const STREAMING_HARMONIZATION_LONG_SIDE: u32 = 2400;
// Keep this narrow enough to cancel the actual source discontinuity at an
// ownership boundary. A broad gain blur follows nearby dark strokes and draws
// a visible halo even though the gain itself is constant per capture group.
const STREAMING_GROUP_GAIN_FEATHER: f32 = 3.0;
const STREAMING_GROUP_GAIN_MAX_LOG: f64 = 0.223_143_551_314_209_76;
// Camera-position lighting varies smoothly across a frame, so a seam needs a
// much broader low-frequency transition than the ownership antialiasing.
// These radii operate on the bounded 2400px analysis image.
const STREAMING_ILLUMINATION_LOCAL_SIGMA: f32 = 16.0;
const STREAMING_ILLUMINATION_CONTINUOUS_SIGMA: f32 = 320.0;
const STREAMING_ILLUMINATION_MIN_GAIN: f64 = 0.70;
const STREAMING_ILLUMINATION_MAX_GAIN: f64 = 1.40;
const GRID_STEP: u32 = 32;
const NATIVE_REFINE_STEP: u32 = 16;
const NATIVE_FIELD_RADIUS: f64 = 32.0;
// Selection cost added to a candidate that is sharper than the current
// composite but photometrically inconsistent with it (需求 3.3 requires ≥ 1.0).
pub(super) const OWNERSHIP_MISMATCH_PENALTY: f64 = 2.4;
const STREAMING_TILE_SIZE: u32 = 1024;
// A float RGB canvas costs twelve bytes per pixel before the ownership mask,
// decoded RAW layer, and canonical 16-bit result are counted.  Switch to the
// tiled backing store by canvas area as well as by capture-sequence metadata:
// evidence-only graph alignment intentionally has no synthetic sequence
// bridges, so using that flag alone made large, valid focus mosaics allocate
// the entire sparse bounding box in memory.
const STREAMING_CANVAS_MIN_PIXELS: u64 = 120_000_000;

type GroupToneRelation = ((u8, u8), [f64; 3], f64);

// Storage and tiled backing store.
include!("mosaic_storage.inc.rs");

// Sampling, photometric correction and low-level image analysis.
include!("mosaic_photometric.inc.rs");
include!("mosaic_sampling.inc.rs");

// Native residual refinement is split into model, pass and runner phases.
include!("mosaic_refinement_model.inc.rs");
include!("mosaic_refinement_pass.inc.rs");
include!("mosaic_refinement_runner.inc.rs");

// Tone and ownership decisions.
include!("mosaic_tone.inc.rs");
include!("mosaic_ownership.inc.rs");

// Top-level compositor and streaming renderer.
include!("mosaic_compositor.inc.rs");

// Test-only accessors and focused regression suites.
include!("mosaic_tests_access.inc.rs");

#[cfg(test)]
mod tests {
    use super::*;
    include!("mosaic_tests_ownership.inc.rs");
    include!("mosaic_tests_refinement.inc.rs");
}
