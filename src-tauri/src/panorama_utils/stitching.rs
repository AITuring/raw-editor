pub(crate) use super::coordinates::output_canvas_dimensions;
pub use super::coordinates::{Projection, project_point};
pub(super) use super::coordinates::{map_target_to_source, output_bounds, pixel_aligned_canvas};
#[cfg(test)]
pub(super) use super::focus_geometry::focus_stack_is_shifted_mosaic;
pub(super) use super::render_geometry::transformed_image_region;
#[cfg(test)]
use super::stack_pipeline::compositor::seam_candidate_cost_with_alignment;
#[cfg(test)]
use super::stack_pipeline::degradation;
#[cfg(test)]
use super::stack_pipeline::focus_fuser;
#[cfg(test)]
use super::stack_pipeline::focus_fuser::{CellSamplingPlan, OwnershipGridGeometry};
#[cfg(test)]
use super::stack_pipeline::report::{IntraStationFrameRecord, IntraStationFrameStatus};
#[cfg(test)]
use super::stack_pipeline::residual_warp;
#[cfg(test)]
use super::stack_pipeline::station_degradation::{plan_station_fusion, station_plan_entries};
#[cfg(test)]
use crate::panorama_utils::shared::{FocusLayerWarp, ImageInfo};
use image::Rgb;
#[cfg(test)]
use image::{GrayImage, Rgb32FImage};
#[cfg(test)]
use nalgebra::{Matrix3, Point2, Point3};
#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
#[path = "final_world_owner_oracle.rs"]
mod final_world_owner_oracle;
#[cfg(test)]
#[path = "owner_quality_oracle.rs"]
mod owner_quality_oracle;

mod canvas_maps;

pub(crate) use canvas_maps::{
    ColorEncoding, ConfidenceMap, CoverageMask, NO_OWNER, OwnershipMap, SourceProvenance,
    map_target_to_source_with_residual, output_canvas_dimensions_with_focus_warp,
};

mod virtual_tiles;

use virtual_tiles::native_station_geometry;
pub(crate) use virtual_tiles::{
    FocusVirtualTileGeometry, VirtualTile, focus_stack_virtual_tile_geometry,
};

mod exposure;

use exposure::{
    ExposureCompensation, ExposureOverlap, apply_exposure_compensation,
    apply_exposure_compensation_with_strength, estimate_overlap_exposure_compensation,
};

mod station_seam_sampling;

#[cfg(test)]
use station_seam_sampling::station_seam_luma;
use station_seam_sampling::{
    StationSeamPlan, projected_coverage_boundary_distance, seam_candidate_relation_for_pair,
    source_to_world_point, station_local_relation_resample_band_px, station_seam_edge_misalignment,
    station_seam_edge_strength_weight, station_seam_foreground_penalty, station_seam_local_tone,
    station_seam_pair_sample, station_seam_sample, station_seam_world_gradient,
};

mod station_seam_plan;

use station_seam_plan::{
    build_station_seam_plans, station_current_cross_at, station_current_transition_candidates,
    station_seam_cross_at, station_transition_first_on_low_side,
};

mod station_seam_apply;

use station_seam_apply::apply_station_seam_plans;
#[cfg(test)]
use station_seam_apply::write_station_owner_trace;

mod progressive_seam;

pub use progressive_seam::progressive_seam_stitcher;
use progressive_seam::{SeamContext, SeamInfo, SeamOrientation};
pub(crate) use progressive_seam::{apply_photometric_model_to_image, flat_scan_paper_median};

mod panorama_seam_band;

#[cfg(test)]
use panorama_seam_band::panorama_detail_alpha;
use panorama_seam_band::{SeamBandBlend, blend_panorama_seam_band};

mod layered_compositor;

#[cfg(test)]
use layered_compositor::OwnershipCompositorSwitches;
pub(crate) use layered_compositor::{
    FocusTilePayload, FocusTileSource, FocusTileSources, LayeredOwnershipRender,
    layered_final_sharpen_amount, layered_virtual_tile_compositor_with_ownership,
};
use layered_compositor::{
    TileCompositorFinishing, focus_tile_ownership_should_replace, ownership_compositor_switches,
    tile_boundary_distance_world, tile_covered_boundary_distance,
};

mod tile_ownership_stitcher;

use tile_ownership_stitcher::focus_tile_ownership_stitcher_with_finishing;

#[cfg(test)]
mod station_stroke_match;

#[cfg(test)]
use station_stroke_match::{
    ClosedStrokeMatch, STROKE_MATCH_CLOSURE_MAX_PX, STROKE_MATCH_CONTRAST_MIN,
    STROKE_MATCH_CURVATURE_MAX, STROKE_MATCH_ORIENTATION_MAX_DEGREES, STROKE_MATCH_PATCH_RMS_MAX,
    STROKE_MATCH_SEAM_RADIUS_PX, STROKE_MATCH_SEARCH_RADIUS_PX, STROKE_MATCH_UNIQUENESS_MARGIN_MIN,
    STROKE_MATCH_UNIQUENESS_RATIO_MAX, StationStrokeView, closed_station_stroke_match,
    station_stroke_correspondences, stroke_feature, stroke_translation_holdout,
};

#[cfg(test)]
mod station_seam_diagnostics;

#[cfg(test)]
use station_seam_diagnostics::write_station_seam_diagnostics;

#[cfg(test)]
mod test_tone_capture;

#[cfg(test)]
pub(crate) use test_tone_capture::set_test_tone_capture_directory;
#[cfg(test)]
use test_tone_capture::{
    TEST_TONE_CAPTURE, capture_test_contour_probe, capture_test_seam_grid,
    capture_test_tone_fields, capture_test_tone_tile,
};

#[cfg(test)]
mod final_world_owner_runner;

#[cfg(test)]
pub(crate) use final_world_owner_runner::run_final_world_owner_oracle;

#[cfg(test)]
pub(crate) fn reset_final_world_owner_oracle() {
    final_world_owner_oracle::reset_station_snapshots();
}

const PANORAMA_BLEND_BANDS: usize = 9;
const PANORAMA_DETAIL_SEAM_FEATHER_RADIUS: f32 = 4.0;
// Laplacian levels 0..4 contain structure up to roughly 32 px wide. Keep those
// levels tied to the optimal seam; broader bands may follow the overlap-wide
// illumination ramp without visibly doubling normal photographic detail.
const PANORAMA_GLOBAL_TONE_FIRST_BAND: usize = 5;
// Keep enough native detail in the focus decision for thin brush strokes and
// seal edges. The final canvas may be much wider than one source frame, so the
// analysis scale is still bounded by the per-source dimension in
// `focus_analysis_dimensions`.
// Keep enough samples for narrow brush strokes and paper weave to influence
// focus ownership.  The analysis image is still bounded independently of the
// full panorama, so increasing this from the old 2048px cap does not make a
// long scan allocate one score map per full-resolution source frame.
const FOCUS_ANALYSIS_MAX_DIMENSION: u32 = 3072;
const FOCUS_ANALYSIS_MAX_PIXELS: u64 = 40_000_000;
const FOCUS_DECISIVE_ADVANTAGE: f32 = 0.20;
const FOCUS_CONFIDENCE_MARGIN: f32 = 0.04;
const FOCUS_EDGE_PROTECTION_AT_1024: f32 = 12.0;
const FOCUS_EDGE_CLAIM_THRESHOLD: f32 = 0.08;
// The analysis canvas is still sampled from the native frame (roughly 4-5
// source pixels per analysis pixel for the Lanting fixture). Keep the focus
// metric local at that scale: a large blur makes a sharp character claim its
// neighbour and is the main way a focus stack can become softer than either
// input frame.
const FOCUS_SCORE_BLUR_RADIUS_DIVISOR: f32 = 1_024.0;
// Keep the sharpness support local.  At the normal 3x analysis reduction a
// four-pixel analysis blur covered roughly a 25px native neighbourhood and
// made adjacent out-of-focus brush strokes contribute to one another.  A
// two-pixel cap still suppresses sensor noise while retaining a usable
// focus boundary for thin ink and woven paper.
const FOCUS_SCORE_MAX_BLUR_RADIUS: usize = 2;
const FOCUS_DECISION_COHERENCE_RADIUS_DIVISOR: f32 = 1_024.0;
const FOCUS_DECISION_MAX_COHERENCE_RADIUS: usize = 3;
const FOCUS_EDGE_PROTECTION_SCALE: f32 = 0.35;
// The broad tone transition is limited to the canvas-only low-frequency mask
// below. Keep enough room for a source-sized exposure step to meet smoothly,
// while all foreground pixels and all detail bands remain source-selected.
const FOCUS_SEAM_BLEND_RADIUS: usize = 512;
const FOCUS_FOREGROUND_HARD_EDGE_RADIUS_AT_2400: f32 = 8.0;
const FOCUS_SEAM_TONE_MIN_SAMPLES: usize = 128;
const FOCUS_SEAM_TONE_MAX_ADJUSTMENT: f32 = 0.075;
const FOCUS_COLOR_SAMPLE_BUDGET: u64 = 12_000;
const FOCUS_COLOR_MIN_SAMPLES: usize = 96;
const FOCUS_COLOR_MIN_CHANNEL_VALUE: f32 = 0.025;
const FOCUS_COLOR_MIN_GAIN: f32 = 0.78;
const FOCUS_COLOR_MAX_GAIN: f32 = 1.28;
const FOCUS_COLOR_MAX_OFFSET: f32 = 0.06;
// Bright plastic, glass, or paper edges can sit above the normal colour-sample
// ceiling. Admit those pixels only inside a matched foreground consensus band;
// ordinary bright highlights remain excluded from exposure estimation.
const FOCUS_COLOR_MAX_RELAXED_LUMA: f32 = 0.995;
// Focus stacks must not average a displaced subject at a seam. The seam path
// therefore blends only the low-frequency tone bands and restores protected
// foreground pixels after reconstruction; the visible detail bands stay on a
// single source. This lets a shifted scan remove exposure blocks without
// turning a displaced brush stroke into a double contour.
const FOCUS_ALLOW_LOW_FREQUENCY_SEAM_BLEND: bool = false;
// Correct broad local illumination differences without allowing individual
// brush strokes or the canvas weave to become a colour reference. The field is
// sampled in candidate-image space, then interpolated while the layer is
// copied into the final canvas.
const FOCUS_COLOR_CELL_SIZE: u32 = 768;
const FOCUS_COLOR_MIN_CELL_SAMPLES: usize = 12;
const FOCUS_COLOR_SPATIAL_VARIATION_THRESHOLD: f32 = 0.015;
const FOCUS_COLOR_SPATIAL_OFFSET_THRESHOLD: f32 = 0.004;
// Overlap samples do not cover the newly exposed side of every shifted frame.
// Let a measured local correction fade into nearby unsampled cells instead of
// stopping at a grid boundary, but keep the reach and strength bounded so an
// isolated overlap cannot recolour an entire new region.
const FOCUS_COLOR_PROPAGATION_RADIUS_CELLS: i32 = 3;
const FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE: f32 = 0.65;
// The final canvas can still contain a broad, source-sized exposure step when
// a newly exposed background region has no direct overlap samples. Work on a
// bounded analysis image and correct only a slowly varying background field;
// the canvas weave and all selected foreground detail remain untouched.
const FOCUS_BACKGROUND_TONE_LOCAL_RADIUS: usize = 4;
const FOCUS_BACKGROUND_TONE_SMOOTH_RADIUS: usize = 256;
// Prefer a continuous canvas tone over owner-specific medians.  The selected
// detail pixels are still copied from one source; this only changes the broad
// illumination field that otherwise exposes virtual-tile rectangles.
const FOCUS_BACKGROUND_TONE_GLOBAL_WEIGHT: f32 = 0.82;
const FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT: f32 = 0.24;
// Owner-level tone estimates are only trustworthy when the selected owner
// contains a locally consistent canvas sample.  A whole owner may cover a
// different painted region from the global median; use a tighter correction
// bound and reject high-dispersion owners instead of recolouring their detail.
const FOCUS_OWNER_TONE_MAX_ADJUSTMENT: f32 = 0.08;
const FOCUS_OWNER_TONE_MAX_MAD: f32 = 0.045;
const FOCUS_BACKGROUND_TONE_FOREGROUND_RADIUS_RATIO: f32 = 0.018;
// A depth layer can move by more than a pixel when its source frame is aligned
// against the paper plane. Keep an ownership buffer around an already selected
// foreground object so a later frame cannot re-introduce the same object as a
// parallel strip. The radius is derived from the source layer, not from any
// particular colour or holder shape.
const FOCUS_FOREGROUND_OWNERSHIP_RADIUS_RATIO: f32 = 0.025;
// The mask-only foreground refinement is a last-mile correction, not a second
// registration pass. A larger search can match a repeated rail/seal edge and
// move an otherwise valid foreground layer onto the wrong instance. Residual
// motion beyond this bound must be resolved by the image-backed homography and
// edge-consensus stages above.
const FOCUS_FOREGROUND_REFINEMENT_MAX_SHIFT: i32 = 32;
// Scalable stacks register on a reduced analysis image. Refine the committed
// full-resolution layer only when several independent overlap patches agree on
// the same small translation; ambiguous texture and isolated repeated strokes
// are deliberately left unchanged.
const FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS: i32 = 8;
// Global pose registration is intentionally conservative on a long scan, so a
// focus layer can still arrive at the tile renderer tens or hundreds of pixels
// away from the already selected layer. The bounded per-patch pyramid searches
// this range before refining native pixels; a fixed +-16px window leaves
// doubled brush strokes whenever the residual exceeds that bound.
const FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT: i32 = 384;
const FOCUS_FULL_RES_ALIGNMENT_GRID_SIZE: i32 = 4;
const FOCUS_FULL_RES_ALIGNMENT_MIN_NCC: f64 = 0.52;
const FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN: f64 = 0.025;
const FOCUS_FULL_RES_ALIGNMENT_MIN_ENERGY: f64 = 0.008;
const FOCUS_FULL_RES_ALIGNMENT_MIN_PATCHES: usize = 4;
const FOCUS_FULL_RES_ALIGNMENT_MAX_FOREGROUND_FRACTION: f64 = 0.25;
/// Smallest native-resolution correction worth resampling a rendered layer for
/// (需求 2.3).  Below a twentieth of a pixel the warp is indistinguishable from
/// the interpolation it costs, so the already rendered pixels are kept.
const FOCUS_NATIVE_REFINEMENT_MIN_OFFSET_PX: f64 = 0.05;

#[cfg(test)]
mod compositor_switch_contract_tests {
    use super::*;

    #[test]
    fn default_ownership_switches_are_all_off_for_the_layered_path() {
        let switches = ownership_compositor_switches(TileCompositorFinishing::LayeredVirtualTile);
        assert!(!switches.tone_all_pixels);
        assert!(!switches.soft_owner_boundary);
        assert!(!switches.skip_exposure_gain);
        assert!(!switches.low_frequency_consensus);
        assert!(!switches.owner_tone_harmonization);
        assert!(!switches.skip_owner_tone_harmonization);
        assert!(!switches.final_low_frequency_illumination);
        assert_eq!(switches.legacy_final_sharpen_amount, 0.85);
        assert_eq!(
            crate::panorama_utils::stack_pipeline::report::SelectedPath::LegacySingleLayerMosaic
                .as_identifier(),
            "legacy_single_layer_mosaic"
        );
    }

    #[test]
    fn layered_entry_uses_default_switches_while_legacy_entry_keeps_its_diagnostic_defaults() {
        let layered = ownership_compositor_switches(TileCompositorFinishing::LayeredVirtualTile);
        let legacy = ownership_compositor_switches(TileCompositorFinishing::LegacyOwnership);

        assert_eq!(layered, OwnershipCompositorSwitches::default());
        assert_eq!(legacy.legacy_final_sharpen_amount, 0.85);
        assert!(!legacy.tone_all_pixels);
        assert!(!legacy.soft_owner_boundary);
        assert!(!legacy.skip_exposure_gain);
    }
}

#[cfg(test)]
mod layered_coverage_distance_tests {
    use super::*;
    #[test]
    fn distance_uses_real_coverage_holes_and_native_world_units() {
        let (w, h) = (64, 64);
        let owners = (0..h)
            .flat_map(|_| (0..w).map(|x| if x == 32 { 0 } else { 1 }))
            .collect::<Vec<_>>();
        let sources = FocusTileSources {
            ownership: OwnershipMap::new(
                w,
                h,
                owners,
                vec![std::path::PathBuf::from("covered-source.nef")],
            )
            .unwrap(),
            origin: (0.0, 0.0),
            originals: Vec::new(),
        };
        // Black is a covered source sample here; its colour cannot turn it
        // into transparent padding in the seam's coverage-distance evidence.
        let tile = Rgb32FImage::new(w, h);
        let distance = tile_covered_boundary_distance(&tile, Some(&sources));
        assert_eq!(distance.get_pixel(32, 20)[0], 0);
        assert_eq!(distance.get_pixel(31, 20)[0], 1);
        assert_eq!(distance.get_pixel(16, 16)[0], 16);
        let pose = Matrix3::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0);
        let near = tile_boundary_distance_world(&distance, Point2::new(31.0, 20.0), &pose, false);
        assert!(near < 16.0);
        assert!(super::super::stack_pipeline::compositor::seam_candidate_cost(0.0, near) >= 1.0);
        let interior =
            tile_boundary_distance_world(&distance, Point2::new(16.0, 16.0), &pose, false);
        assert!(interior > 16.0 && interior <= 32.0);
    }
}

fn luminance(pixel: &Rgb<f32>) -> f32 {
    pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114
}

fn focus_stack_pixel_is_canvas_like(pixel: &[f32]) -> bool {
    if pixel.len() < 3 || pixel[..3].iter().any(|value| !value.is_finite()) {
        return false;
    }
    let red = pixel[0].clamp(0.0, 1.0);
    let green = pixel[1].clamp(0.0, 1.0);
    let blue = pixel[2].clamp(0.0, 1.0);
    let luma = red * 0.299 + green * 0.587 + blue * 0.114;
    let red_floor = red.max(0.001);
    let green_ratio = green / red_floor;
    let blue_ratio = blue / red_floor;
    let red_dominance = red - 2.0 * green + blue;
    // The source canvas is a warm, moderately desaturated brown. This gate
    // deliberately rejects the red skirt, green sash, near-black robe, and
    // pale faces/hands before a low-frequency exposure field is applied. The
    // limits are broad enough to retain the darker and lighter canvas tiles.
    // Darkened paper from a long scan can fall below the original 0.12
    // floor.  Keep a small margin above true black ink so the low-frequency
    // tone field can still see those paper samples.
    luma >= 0.08
        && luma <= 0.78
        && red >= green
        && green >= blue
        && green_ratio >= 0.42
        && blue_ratio >= 0.24
        && red - green >= 0.025
        && green - blue >= 0.015
        && red_dominance <= 0.08
}

fn focus_stack_pixel_is_tone_foreground(pixel: &[f32]) -> bool {
    if pixel.len() < 3 || pixel[..3].iter().any(|value| !value.is_finite()) {
        return false;
    }
    let red = pixel[0].clamp(0.0, 1.0);
    let green = pixel[1].clamp(0.0, 1.0);
    let blue = pixel[2].clamp(0.0, 1.0);
    let luma = red * 0.299 + green * 0.587 + blue * 0.114;
    let chroma = red.max(green).max(blue) - red.min(green).min(blue);
    let red_dominance = red - 2.0 * green + blue;
    let red_paint = red - green >= 0.14 && red - blue >= 0.17 && red - 2.0 * green + blue >= 0.08;
    let green_or_cool_paint = green - red >= 0.05 || blue - green >= 0.05;
    // Brown paper can be dark and nearly neutral.  Reserve this guard for
    // genuinely near-black ink; otherwise dark paper is excluded from tone
    // estimation and remains as a visible tile-sized exposure block.
    let dark_neutral_paint =
        luma <= 0.12 && (red - green).abs() <= 0.07 && (green - blue).abs() <= 0.07;
    // Faces and hands can have nearly the same average luminance as the
    // canvas, but their blue channel is much closer to green and their red
    // dominance is stronger. Keep this warm-pale paint out of the canvas-only
    // correction without admitting the flatter brown paper tone.
    let warm_light_paint = luma >= 0.22
        && red - green >= 0.05
        && red - blue >= 0.08
        && green - blue <= 0.08
        && red_dominance >= 0.04;
    // Very pale sleeves, hands, and faces are outside the brown canvas range.
    // This is intentionally narrower than a generic bright-pixel detector so
    // a lighter canvas exposure can still be harmonized.
    let pale_paint = luma >= 0.78 && chroma <= 0.22;
    red_paint || green_or_cool_paint || dark_neutral_paint || warm_light_paint || pale_paint
}

mod focus_layers;

pub(crate) use focus_layers::FocusSourceSampling;
use focus_layers::{
    RenderedFocusAnalysisLayer, RenderedFocusLayer, render_focus_layer,
    render_focus_layer_with_refinement, source_is_focus_foreground,
};

mod focus_color;

#[cfg(test)]
use focus_color::{apply_focus_color_correction, estimate_focus_color_correction};

mod focus_decision;

use focus_decision::{
    box_blur_focus_map, focus_analysis_dimensions, focus_decision_mask, focus_score_map,
    suppress_focus_canvas_switches,
};

mod focus_masks;

#[cfg(test)]
use focus_masks::hard_select_focus_pixels;
use focus_masks::{
    build_focus_background_foreground_envelope, place_focus_foreground_mask, place_focus_layer,
    resize_binary_mask,
};

mod focus_alignment;

#[cfg(test)]
use focus_alignment::warp_rendered_focus_layer;
use focus_alignment::{
    align_focus_foreground_layer_to_existing, estimate_focus_layer_translation,
    register_station_frame_native_layer, translate_rendered_focus_layer,
};

mod focus_foreground;

use focus_foreground::{
    focus_foreground_edge_mask, focus_foreground_overlap_mask_for_region, hard_select_focus_layer,
    mark_focus_foreground_pixels, mark_focus_owner_pixels, record_focus_layer_ownership,
    scaled_region_end, scaled_region_start, suppress_focus_foreground_ownership_switches,
    suppress_focus_foreground_ownership_switches_on_canvas, suppress_focus_foreground_switches,
    suppress_focus_foreground_switches_on_canvas,
};

mod focus_analysis;

use focus_analysis::{
    commit_focus_analysis_scores, focus_cell_decision_for_layer, focus_cell_plane_to_pixels,
    focus_cell_texture_to_pixels, measure_focus_cell_disagreement, measure_focus_cells,
    render_focus_analysis_layer,
};

mod filters;

pub(super) use filters::downsample_rgb_half;
use filters::{
    linear_samples, masked_box_blur_focus_map, masked_box_blur_rgb, multiband_blend, resize_rgb,
};
#[cfg(test)]
use filters::{subtract_upsampled_rgb, upsample_and_add_rgb};

mod canvas_fill;

pub(super) use canvas_fill::{crop_to_valid_bounds, crop_to_valid_rectangle};
use canvas_fill::{fill_focus_canvas_margins, valid_mask_bounds_and_coverage};

mod focus_tone;

use focus_tone::{harmonize_focus_background_tone, harmonize_focus_background_tone_with_owners};

mod focus_seam_blend;

use focus_seam_blend::blend_focus_seam_band;

mod focus_stack_entry;

pub use focus_stack_entry::focus_stack_stitcher;
#[cfg(test)]
use focus_stack_entry::station_members_from_frame_records;
#[cfg(test)]
pub(crate) use focus_stack_entry::station_members_from_measured_group_join_evidence;
pub(crate) use focus_stack_entry::{
    FocusForegroundMove, FocusSourceRefinement, FocusStackTileMasks, FocusStackTileRender,
    focus_stack_stitcher_unfilled,
};
use focus_stack_entry::{
    order_focus_station_frames, sample_focus_plane, station_members_from_records,
};

mod focus_stack_stitcher;

use focus_stack_stitcher::{focus_stack_stitcher_with_margin_policy, sharpen_focus_tile_detail};

mod seam_path;

use seam_path::find_adaptive_seam;

mod sampling;

pub(crate) use sampling::get_high_quality_interpolated_pixel;
pub use sampling::warp_image_homography;
use sampling::{get_interpolated_pixel, source_sample_has_support};

#[cfg(test)]
mod score_noise_probes;

#[cfg(test)]
use score_noise_probes::{
    FinalOwnerRoiConstraint, SCORE_NOISE_PHASE_OFFSETS, ScoreNoiseAttributionCounts,
    ScoreNoiseCandidateCell, ScoreNoiseCellRecord, ScoreNoiseRegistrationCell,
    ScoreNoiseRoiSummary, ScoreNoiseSummary, owner_quality_forced_labeling, score_noise_aggregate,
    score_noise_owner_regions, score_noise_primary_attribution, score_noise_probe_values,
    score_noise_registration_cells, score_noise_single_owner_rois, score_noise_smoothness_scale,
    score_noise_source_roi_origins, score_noise_variance, score_noise_winner_owners,
};

#[cfg(test)]
mod score_noise_diagnostics;

#[cfg(test)]
use score_noise_diagnostics::ScoreNoiseDiagnostics;

fn median_f32(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f32::total_cmp);
    Some(values[values.len() / 2])
}

#[cfg(test)]
mod interpolation_tests;

/// Test-only access to the Focus_Fuser measurement the default station path
/// drives (需求 3.2, 3.3).
///
/// Property 10 lives in `stack_pipeline::properties`, which cannot see
/// [`measure_focus_cells`] or [`sample_focus_plane`].  Both entry points below
/// call the production functions unchanged with the production geometry and the
/// production sampling plan, so the property tests the measurement the composite
/// and every candidate really go through rather than a re-implementation of it.
#[cfg(test)]
pub(crate) mod focus_fusion_test_access;

/// 需求 12.1 / 12.2 / 12.5 on the *default* station path.
///
/// The decisions themselves are unit tested in
/// `stack_pipeline::station_degradation`; these tests cover the wiring, which is
/// where the same trap has been hit before: a rule connected to
/// `mosaic::detail_preserving_mosaic` alone never runs, because a Virtual_Tile
/// holds one Capture_Station and therefore always takes the standard focus
/// fusion branch of `focus_stack_stitcher_with_margin_policy`.
#[cfg(test)]
mod station_degradation_wiring_tests;
