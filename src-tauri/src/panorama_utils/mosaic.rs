//! Focus mosaics: refine residual geometry, choose sharp source detail, match
//! only broad colour. The output is never an average of displaced fine detail.
use super::stack_pipeline::intra_station;
#[cfg(test)]
use super::stack_pipeline::report::IntraStationFrameStatus;
#[cfg(test)]
use super::stitching::downsample_rgb_half;
use super::stitching::{Projection, get_high_quality_interpolated_pixel};
use crate::panorama_utils::shared::ImageInfo;
#[cfg(test)]
use image::Luma;
use image::{GrayImage, Rgb, Rgb32FImage};
use nalgebra::Matrix3;
#[cfg(test)]
use nalgebra::{Point2, Point3};
#[cfg(test)]
use rayon::prelude::*;
#[cfg(test)]
use std::collections::HashMap;

#[cfg(test)]
#[path = "mosaic_diagnostics.rs"]
mod diagnostics;

mod sampling;

pub(crate) use sampling::Field;
use sampling::LayerSampler;
#[cfg(test)]
use sampling::is_lower_left_black_capture_border;

mod streaming_store;

#[cfg(test)]
use streaming_store::sample_decision_alpha;
use streaming_store::{
    StreamingMosaicStore, copy_streaming_owned_tile_region, copy_streaming_tile_region,
    fill_streaming_uncovered_tile_region, materialize_streaming_focus_output,
};

mod streaming_harmonization;

#[cfg(test)]
use streaming_harmonization::streaming_group_tone_relations_from_analysis;
use streaming_harmonization::{
    StreamingLowFrequencyConsensus, apply_streaming_seam_harmonization,
    streaming_seam_harmonization,
};

mod streaming_illumination;

use streaming_illumination::apply_streaming_source_owner_low_frequency_correction;
pub(super) use streaming_illumination::smooth_streaming_low_frequency_illumination;

mod layer_refine;

use layer_refine::{
    refine_layer, refine_layer_from_analysis, streaming_candidate_analysis,
    streaming_refinement_pair,
};

mod layer_tone;

use layer_tone::{streaming_tone_field_from_analysis, tone_field};

mod source_ownership;

use source_ownership::{ownership, ownership_with_long_side, streaming_ownership_from_analysis};
#[cfg(test)]
use source_ownership::{
    ownership_disagreement, ownership_grid, streaming_ownership_with_long_side,
    streaming_transition_structure,
};

mod focus_measure;

pub(super) use focus_measure::{
    ACUTANCE_PATCH_POINTS, acutance_with_step, cell_focus_with_step, cell_low_pass_differences,
    cell_probe_offsets,
};
use focus_measure::{cell_focus, streaming_cell_focus};
pub(crate) use focus_measure::{gray_analysis_sharpness, station_member_median_sharpness};

mod native_control_points;

use native_control_points::{SimilarityStats, collect_native_observations};
#[cfg(test)]
use native_control_points::{contrasted_native_control_location, measure_native_control_point};

mod native_refine;

pub(crate) use native_refine::NativeRefineMode;
#[cfg(test)]
pub(crate) use native_refine::native_control_point_spacing_px;
use native_refine::{
    NativeRefineBase, NativeRefinePass, intra_station_frame_record, refine_native_layer,
    refine_native_layer_streaming,
};

mod station_registration;

pub(crate) use station_registration::{
    NativeResidualDisplacement, StationFrameRegistration,
    register_station_frame_native_with_mapping,
};

mod in_memory_mosaic;

pub(crate) use in_memory_mosaic::detail_preserving_mosaic;
#[cfg(test)]
use in_memory_mosaic::mask_covered_bounds;

mod streaming_mosaic;

use streaming_mosaic::detail_preserving_mosaic_streaming;

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
// Allow a bounded correction for a real source exposure step while rejecting
// the extreme ratios produced by a displaced brush stroke.  All relation
// paths (seam probes, direct overlap measurements, and the robust group solve)
// share this symmetric cap.
const STREAMING_GROUP_GAIN_MAX_LOG: f64 = 0.70;
// Camera-position lighting varies smoothly across a frame, so a seam needs a
// much broader low-frequency transition than the ownership antialiasing.
// These radii operate on the bounded 2400px analysis image.
const STREAMING_ILLUMINATION_LOCAL_SIGMA: f32 = 16.0;
const STREAMING_ILLUMINATION_CONTINUOUS_SIGMA: f32 = 640.0;
const STREAMING_ILLUMINATION_MIN_GAIN: f64 = 0.70;
const STREAMING_ILLUMINATION_MAX_GAIN: f64 = 1.40;
// The consensus pass runs on a bounded low-frequency grid.  It only supplies
// a smooth additive correction after source ownership is final; all brush and
// paper detail remains the pixel chosen by the ownership compositor.
// Keep this wider than the ownership antialiasing. Exposure blocks span
// hundreds of native pixels; a short kernel would only retouch the seam edge
// and leave the source rectangle intact. This is still low-frequency because
// the final correction is added to the untouched source-owned pixels.
const STREAMING_CONSENSUS_SIGMA: f32 = 64.0;
const STREAMING_CONSENSUS_MIN_SUPPORT: f32 = 0.001;
const STREAMING_CONSENSUS_MAX_DELTA: f64 = 0.18;
// Source-owner correction compares each owner's local low-frequency field
// with a much broader canvas reference.  It removes frame-sized exposure
// blocks while adding only a smooth field to source-owned pixels.
const STREAMING_OWNER_LOCAL_SIGMA: f32 = 64.0;
const STREAMING_OWNER_GLOBAL_SIGMA: f32 = 640.0;
// Flat-artwork source exposures can vary over a whole frame.  The overlap
// observations therefore need to influence the non-overlap interior over
// several 112px tone-field cells; a one-cell kernel leaves a source-sized
// exposure rectangle intact.  Focus-stack layers keep the narrower kernel.
const STREAMING_FLAT_TONE_SIGMA_CELLS: f64 = 16.0;
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

fn luma(p: Rgb<f32>) -> f64 {
    f64::from(p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114)
}

fn rgb_at(image: &Rgb32FImage, x: f64, y: f64) -> Option<Rgb<f32>> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= image.width() as f64
        || y >= image.height() as f64
    {
        return None;
    }
    Some(get_high_quality_interpolated_pixel(image, x, y))
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Register one non-anchor frame of a Capture_Station against the station's
/// already composited canvas at native resolution (需求 2.2–2.9).
///
/// This is the entry point of the *default* station path.  It runs the same
/// measurement as the mosaic comparison path — [`refine_layer`] twice at the
/// analysis resolution of 需求 2.2, then [`refine_native_layer`] with the
/// control point gates of 需求 2.3–2.8 — over a [`LayerSampler`] built from the
/// frame's own source pixels, and hands back both the report record and the
/// displacement field, so the caller's already rendered layer can be corrected
/// by exactly what was measured.
///
/// `layer` is the frame's canvas footprint as the caller rendered it, and
/// `transform` must be the pose that produced it: the sampler reproduces that
/// geometry, which is what makes the measured displacement applicable to those
/// pixels.
#[allow(clippy::too_many_arguments)]
pub(crate) fn register_station_frame_native(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    info: &ImageInfo,
    source: &Rgb32FImage,
    transform: &Matrix3<f64>,
    projection: Projection,
    offset: (f64, f64),
    layer: (u32, u32, u32, u32),
    anchor_long_side: u32,
    into_anchor: Matrix3<f64>,
) -> Option<StationFrameRegistration> {
    register_station_frame_native_with_mapping(
        base,
        base_mask,
        info,
        source,
        transform,
        projection,
        offset,
        layer,
        anchor_long_side,
        into_anchor,
        None,
    )
}

fn adjusted(pixel: Rgb<f32>, delta: [f64; 3]) -> Rgb<f32> {
    Rgb(std::array::from_fn(|c| {
        (pixel[c] * delta[c].exp() as f32).clamp(0.0, 1.0)
    }))
}

/// Test-only access to the Intra_Station_Registrar control point internals of
/// 需求 2.4 to 2.8.
///
/// Property 6 lives in `super::stack_pipeline::properties`, a sibling module that
/// cannot see `LayerSampler`, `NativeRefinePass` or `measure_native_control_point`.
/// The two drivers below assemble exactly what the in-memory compositor assembles
/// — a native-resolution base with a full coverage mask, a layer sampler at
/// `scale = 1.0` and an identity residual field — and then call the production
/// functions unchanged, so the property tests the refinement rather than a model
/// of it.
#[cfg(test)]
pub(crate) mod intra_station_test_access;

#[cfg(test)]
mod tests;
