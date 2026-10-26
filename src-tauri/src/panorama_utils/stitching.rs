#[path = "stitching_adaptive_seam.rs"]
mod adaptive_seam;
#[path = "stitching_focus_alignment.rs"]
mod focus_alignment;
#[path = "stitching_focus_blend.rs"]
mod focus_blend;
#[path = "stitching_focus_fill.rs"]
mod focus_fill;
#[path = "stitching_focus_masks.rs"]
mod focus_masks;
#[path = "stitching_focus_ownership.rs"]
mod focus_ownership;
#[path = "stitching_focus_pipeline.rs"]
mod focus_pipeline;
#[path = "stitching_focus_reference.rs"]
mod focus_reference;
#[path = "stitching_focus_station.rs"]
mod focus_station;
#[path = "stitching_focus_tone.rs"]
mod focus_tone;
#[path = "stitching_focus_translation.rs"]
mod focus_translation;
#[path = "stitching_ownership_compositor.rs"]
mod ownership_compositor;
#[path = "stitching_progressive_compositor.rs"]
mod progressive_compositor;
#[path = "stitching_seam_planning.rs"]
mod seam_planning;
#[path = "stitching_seam_sampling.rs"]
mod seam_sampling;
pub(crate) use adaptive_seam::*;
pub(crate) use focus_alignment::*;
pub(crate) use focus_blend::*;
pub(crate) use focus_fill::*;
pub(crate) use focus_masks::*;
pub(crate) use focus_ownership::*;
pub(crate) use focus_pipeline::*;
pub(crate) use focus_reference::*;
pub(crate) use focus_station::*;
pub(crate) use focus_tone::*;
pub(crate) use focus_translation::*;
pub(crate) use ownership_compositor::*;
pub(crate) use progressive_compositor::*;
pub(crate) use seam_planning::*;
pub(crate) use seam_sampling::*;

use super::photometric::{PhotometricModel, PhotometricOptions, calibrate_overlap_photometry};
#[path = "stitching_compositor_helpers.rs"]
mod compositor_helpers;
#[path = "stitching_compositor_options.rs"]
mod compositor_options;
#[path = "stitching_focus_color.rs"]
mod focus_color;
#[path = "stitching_focus_sampling_render.rs"]
mod focus_sampling_render;
#[path = "virtual_tile_types.rs"]
mod virtual_tile_types;
use super::stack_pipeline::compositor::seam_candidate_cost_with_alignment;
use super::stack_pipeline::degradation;
use super::stack_pipeline::focus_fuser::{
    self, CandidateCells, CellSamplingPlan, OwnershipGridGeometry, StationFusion, cell_disagreement,
};
use super::stack_pipeline::intra_station;
use super::stack_pipeline::report::{
    FusionReport, IntraStationFrameRecord, IntraStationFrameStatus, OwnershipGridSize,
};
use super::stack_pipeline::residual_warp;
use super::stack_pipeline::station_degradation::{
    GroupJoinEvidence, GroupJoinRejection, StationMember, group_join_rejection,
    plan_station_fusion, record_run_entries, record_run_group_join_rejection, station_plan_entries,
};
use super::stack_pipeline::tone;
use crate::panorama_stitching::{FocusLayerWarp, ImageInfo};
pub(super) use compositor_helpers::{
    crop_to_valid_bounds, crop_to_valid_rectangle, valid_mask_bounds_and_coverage,
};
use compositor_options::{TileCompositorFinishing, ownership_compositor_switches};
use focus_color::median_f32;
#[cfg(test)]
use focus_color::{apply_focus_color_correction, estimate_focus_color_correction};
pub(crate) use focus_sampling_render::FocusSourceSampling;
use focus_sampling_render::{
    RenderedFocusLayer, render_focus_layer, render_focus_layer_with_refinement,
};
use image::{GrayImage, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use tauri::{AppHandle, Emitter, Runtime};
pub(crate) use virtual_tile_types::{
    ColorEncoding, ConfidenceMap, CoverageMask, NO_OWNER, OwnershipMap, SourceProvenance,
    VirtualTile,
};

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

#[path = "stitching_geometry.rs"]
mod stitching_geometry;
pub(crate) use compositor_options::layered_final_sharpen_amount;
pub(crate) use stitching_geometry::{
    FocusVirtualTileGeometry, focus_stack_virtual_tile_geometry,
    map_target_to_source_with_residual, output_canvas_dimensions,
    output_canvas_dimensions_with_focus_warp,
};
pub use stitching_geometry::{Projection, project_point};
use stitching_geometry::{focus_output_bounds, unproject_point};
pub(crate) use stitching_geometry::{
    map_target_to_source, output_bounds, pixel_aligned_canvas, transformed_image_region,
};

/// Fuse each capture group independently into one in-memory Virtual_Tile.
///
/// Group membership is supplied by the evidence-backed capture grouping in
/// `panorama_stitching`; this helper does not inspect filenames or assume a
/// particular number of layers.  Every group's existing global homographies
/// are retained, so the returned translation maps tile pixels back into the
/// same world coordinates used by the normal pose graph.
pub(crate) fn focus_stack_virtual_tiles<R: Runtime, F>(
    groups: &[&[&ImageInfo]],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Vec<VirtualTile>, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let mut tiles = Vec::with_capacity(groups.len());
    for (group_index, group) in groups.iter().enumerate() {
        if group.is_empty() {
            continue;
        }
        let geometry =
            focus_stack_virtual_tile_geometry(group, global_homographies, projection, focus_warp)
                .ok_or_else(|| {
                format!(
                    "Capture group {} has invalid focus-tile bounds",
                    group_index + 1
                )
            })?;

        // The regular focus renderer computes its own bounds and uses exactly
        // these pixel-aligned offsets.  Render it once, then expose the
        // inverse translation as the tile-to-world placement.  The unfilled
        // form is the Virtual_Tile form: it keeps the projected validity mask
        // instead of filling the trapezoid margins, which is exactly the
        // Coverage_Mask semantics of 需求 3.7.
        let rendered = focus_stack_stitcher_unfilled(
            group,
            global_homographies,
            projection,
            focus_warp,
            None,
            group_index,
            false,
            app_handle.clone(),
            progress_event,
            load_image,
        )?;
        tiles.push(virtual_tile_from_render(
            group_index,
            geometry.tile_to_world,
            rendered,
        )?);
    }
    Ok(tiles)
}

/// Assemble a Virtual_Tile from a station render.  Every invariant is checked
/// by [`VirtualTile::new`]; this only turns the renderer's internal masks into
/// the published structure (需求 4.1 / 4.2 / 4.6 / 4.7).
pub(crate) fn virtual_tile_from_render(
    station_index: usize,
    tile_to_world: Matrix3<f64>,
    rendered: FocusStackTileRender,
) -> Result<VirtualTile, String> {
    let FocusStackTileRender {
        image,
        masks,
        sampling_origin: _,
    } = rendered;
    let masks = masks.ok_or_else(|| {
        format!(
            "Capture station {} produced no ownership/coverage masks",
            station_index + 1
        )
    })?;
    let mut fusion = masks.fusion;
    fusion.station_index = station_index;
    focus_fuser::record_run_station(fusion);
    let owned_pixels = masks.ownership.owned_pixel_counts();
    let provenance = masks
        .ownership
        .legend()
        .iter()
        .zip(owned_pixels)
        .map(|(path, owned_pixels)| SourceProvenance {
            absolute_path: path.clone(),
            // TODO(task 5.4): the read-only source access records the digest.
            sha256: None,
            owned_pixels,
        })
        .collect::<Vec<_>>();
    VirtualTile::new(
        station_index,
        tile_to_world,
        image,
        masks.ownership,
        masks.confidence,
        masks.coverage,
        // The focus renderer copies decoded render pixels, which are the
        // display-encoded sRGB values used by the rest of the pipeline.
        ColorEncoding::DisplaySrgb,
        provenance,
    )
}

#[allow(dead_code)]
fn apply_exposure_gain(pixel: Rgb<f32>, gain: f32) -> Rgb<f32> {
    Rgb([pixel[0] * gain, pixel[1] * gain, pixel[2] * gain])
}

fn apply_exposure_channel_gain(pixel: Rgb<f32>, gains: [f32; 3]) -> Rgb<f32> {
    Rgb([
        pixel[0] * gains[0],
        pixel[1] * gains[1],
        pixel[2] * gains[2],
    ])
}

fn apply_exposure_compensation(
    pixel: Rgb<f32>,
    exposure: &ExposureCompensation,
    x: u32,
    y: u32,
) -> Rgb<f32> {
    apply_exposure_compensation_with_strength(pixel, exposure, x, y, 1.0)
}

fn apply_exposure_compensation_with_strength(
    pixel: Rgb<f32>,
    exposure: &ExposureCompensation,
    x: u32,
    y: u32,
    strength: f32,
) -> Rgb<f32> {
    let strength = strength.clamp(0.0, 1.0);
    let gains = exposure.channel_gain_at(x, y).map(|gain| {
        if gain.is_finite() && gain > 0.0 {
            (gain.ln() * strength).exp()
        } else {
            1.0
        }
    });
    apply_exposure_channel_gain(pixel, gains)
}

fn panorama_detail_alpha(candidate_signed_distance: f32) -> f32 {
    if candidate_signed_distance.is_infinite() {
        return if candidate_signed_distance.is_sign_positive() {
            1.0
        } else {
            0.0
        };
    }
    (0.5 + candidate_signed_distance / (PANORAMA_DETAIL_SEAM_FEATHER_RADIUS * 2.0)).clamp(0.0, 1.0)
}

struct ExposureOverlap<'a> {
    panorama: &'a Rgb32FImage,
    panorama_mask: &'a GrayImage,
    candidate: &'a ImageInfo,
    candidate_image: &'a Rgb32FImage,
    candidate_inverse: &'a Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
}

#[derive(Clone)]
struct ExposureCompensation {
    cell_size: u32,
    grid_width: usize,
    grid_height: usize,
    gains: Vec<f32>,
    representative_gain: f32,
    channel_gains: Vec<[f32; 3]>,
    representative_channel_gain: [f32; 3],
}

/// Decide whether a projected focus tile should replace the current owner at
/// one output pixel.  Interior distance is the primary quality signal; exact
/// ties are resolved by the stable tile identity so a compositor cannot make
/// the visible seam depend on the order in which groups happened to load.
fn focus_tile_ownership_should_replace(
    current_quality: u8,
    current_owner: u8,
    candidate_quality: u8,
    candidate_owner: u8,
) -> bool {
    current_owner == 0
        || candidate_quality > current_quality
        || (candidate_quality == current_quality && candidate_owner < current_owner)
}

impl ExposureCompensation {
    /// Progressive scan compositing estimates a candidate against the current
    /// panorama.  Applying the full ratio to the whole candidate is correct at
    /// the overlap, but it can turn a real illumination gradient into a chain
    /// of source-sized tone blocks in the non-overlap area.  Pull the estimate
    /// toward one before the seam search; the seam still sees the direction of
    /// the correction while the low-frequency field remains stable across a
    /// long scan.  Focus-bracket fusion keeps the undamped estimator.
    fn damped(mut self, strength: f32) -> Self {
        let strength = strength.clamp(0.0, 1.0);
        let damp = |value: f32| {
            if value.is_finite() && value > 0.0 {
                value.ln().mul_add(strength, 0.0).exp()
            } else {
                1.0
            }
        };
        self.representative_gain = damp(self.representative_gain);
        for gain in &mut self.gains {
            *gain = damp(*gain);
        }
        for channel in &mut self.representative_channel_gain {
            *channel = damp(*channel);
        }
        for gains in &mut self.channel_gains {
            for gain in gains {
                *gain = damp(*gain);
            }
        }
        self
    }

    #[allow(dead_code)]
    fn gain_at(&self, _x: u32, _y: u32) -> f32 {
        if self.gains.is_empty() || self.grid_width == 0 || self.grid_height == 0 {
            return 1.0;
        }
        // The per-cell field is useful for diagnosing vignetting, but writing
        // it into every projected tile makes the 256px analysis cells visible
        // as rectangular tone steps on a long scan.  The default compositor
        // uses one robust overlap constant; spatial compensation is opt-in
        // for captures where illumination is known to vary within a station.
        // The spatial field remains available for diagnostics, but production
        // uses the one robust overlap constant.  `allow_linear = false` is the
        // single photometric wiring decision for the default Tone_Harmonizer.
        self.representative_gain
    }

    /// Diagnostic-only spatial scalar field retained for comparison reports.
    #[allow(dead_code)]
    fn spatial_gain_at(&self, x: u32, y: u32) -> f32 {
        let grid_x = x as f64 / self.cell_size as f64;
        let grid_y = y as f64 / self.cell_size as f64;
        let x0 = (grid_x.floor() as usize).min(self.grid_width - 1);
        let y0 = (grid_y.floor() as usize).min(self.grid_height - 1);
        let x1 = (x0 + 1).min(self.grid_width - 1);
        let y1 = (y0 + 1).min(self.grid_height - 1);
        let tx = (grid_x - x0 as f64) as f32;
        let ty = (grid_y - y0 as f64) as f32;
        let value = |gx: usize, gy: usize| self.gains[gy * self.grid_width + gx];
        let top = value(x0, y0) * (1.0 - tx) + value(x1, y0) * tx;
        let bottom = value(x0, y1) * (1.0 - tx) + value(x1, y1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    fn channel_gain_at(&self, _x: u32, _y: u32) -> [f32; 3] {
        if self.channel_gains.is_empty() || self.grid_width == 0 || self.grid_height == 0 {
            return [1.0; 3];
        }
        // RGB gains use the same conservative default as the scalar field:
        // one robust overlap constant per candidate.  A per-cell RGB field is
        // useful for a measured vignette, but its 256px cells can otherwise
        // become visible chromatic rectangles across a long scan.
        // Keep the per-cell RGB field for comparison diagnostics, while the
        // shipped path uses the robust representative value.
        self.representative_channel_gain
    }

    /// Diagnostic-only spatial RGB field retained for comparison reports.
    #[allow(dead_code)]
    fn spatial_channel_gain_at(&self, x: u32, y: u32) -> [f32; 3] {
        let grid_x = x as f64 / self.cell_size as f64;
        let grid_y = y as f64 / self.cell_size as f64;
        let x0 = (grid_x.floor() as usize).min(self.grid_width - 1);
        let y0 = (grid_y.floor() as usize).min(self.grid_height - 1);
        let x1 = (x0 + 1).min(self.grid_width - 1);
        let y1 = (y0 + 1).min(self.grid_height - 1);
        let tx = (grid_x - x0 as f64) as f32;
        let ty = (grid_y - y0 as f64) as f32;
        let value = |gx: usize, gy: usize| self.channel_gains[gy * self.grid_width + gx];
        std::array::from_fn(|channel| {
            let top = value(x0, y0)[channel] * (1.0 - tx) + value(x1, y0)[channel] * tx;
            let bottom = value(x0, y1)[channel] * (1.0 - tx) + value(x1, y1)[channel] * tx;
            top * (1.0 - ty) + bottom * ty
        })
    }
}

fn estimate_overlap_exposure_compensation(ctx: ExposureOverlap<'_>) -> ExposureCompensation {
    const CELL_SIZE: u32 = 256;
    let (out_width, out_height) = ctx.panorama.dimensions();
    let grid_width = out_width.div_ceil(CELL_SIZE) as usize + 1;
    let grid_height = out_height.div_ceil(CELL_SIZE) as usize + 1;
    let cell_count = grid_width * grid_height;
    let sample_step = out_width.max(out_height).div_ceil(720).max(8) as usize;
    let mut log_sums = vec![0.0f64; cell_count];
    let mut channel_log_sums = vec![[0.0f64; 3]; cell_count];
    let mut counts = vec![0u32; cell_count];
    let mut ratios = Vec::new();
    let mut channel_ratios = [Vec::new(), Vec::new(), Vec::new()];
    let candidate_homography = ctx.candidate_inverse.try_inverse();
    let candidate_region = candidate_homography.as_ref().and_then(|homography| {
        transformed_image_region(
            ctx.candidate,
            homography,
            ctx.projection,
            ctx.offset_x,
            ctx.offset_y,
            out_width,
            out_height,
        )
    });
    let (left, right, top, bottom) = candidate_region.unwrap_or((
        0,
        out_width.saturating_sub(1),
        0,
        out_height.saturating_sub(1),
    ));
    let sample_step_u32 = sample_step as u32;
    let sample_start = |value: u32| value.div_ceil(sample_step_u32) * sample_step_u32;
    for y in (sample_start(top)..=bottom).step_by(sample_step) {
        for x in (sample_start(left)..=right).step_by(sample_step) {
            if ctx.panorama_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let target = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
            let Some(source) =
                map_target_to_source(ctx.candidate_inverse, target, ctx.candidate, ctx.projection)
            else {
                continue;
            };
            if source.x < 0.0
                || source.y < 0.0
                || source.x >= ctx.candidate_image.width() as f64 - 1.0
                || source.y >= ctx.candidate_image.height() as f64 - 1.0
            {
                continue;
            }
            let base_luma = luminance(ctx.panorama.get_pixel(x, y));
            let base_pixel = ctx.panorama.get_pixel(x, y);
            let candidate_pixel = get_interpolated_pixel(ctx.candidate_image, source.x, source.y);
            // Exposure estimation is a paper-plane operation.  A luma-only
            // ratio admits red seals, dark ink, and pale figures into the
            // overlap statistics; one such region can then produce a gain
            // step that is visible across the whole virtual tile.  Require
            // both samples to have the generic canvas signature used by the
            // final tone pass, while retaining the broad luma/range checks
            // below for darker or lighter paper exposures.
            if !focus_stack_pixel_is_canvas_like(base_pixel.0.as_slice())
                || !focus_stack_pixel_is_canvas_like(candidate_pixel.0.as_slice())
                || focus_stack_pixel_is_tone_foreground(base_pixel.0.as_slice())
                || focus_stack_pixel_is_tone_foreground(candidate_pixel.0.as_slice())
            {
                continue;
            }
            let candidate_luma = luminance(&candidate_pixel);
            if (0.025..0.92).contains(&base_luma) && (0.025..0.92).contains(&candidate_luma) {
                let ratio = base_luma / candidate_luma;
                if (0.55..1.8).contains(&ratio) {
                    ratios.push(ratio);
                    for channel in 0..3 {
                        let base_value = base_pixel[channel];
                        let candidate_value = candidate_pixel[channel];
                        if base_value > 0.025 && candidate_value > 0.025 {
                            let channel_ratio = base_value / candidate_value;
                            if (0.55..1.8).contains(&channel_ratio) {
                                channel_ratios[channel].push(channel_ratio);
                            }
                        }
                    }
                    let grid_x = (x / CELL_SIZE) as usize;
                    let grid_y = (y / CELL_SIZE) as usize;
                    let index = grid_y * grid_width + grid_x;
                    log_sums[index] += (ratio as f64).ln();
                    for channel in 0..3 {
                        let base_value = base_pixel[channel];
                        let candidate_value = candidate_pixel[channel];
                        if base_value > 0.025 && candidate_value > 0.025 {
                            let channel_ratio = base_value / candidate_value;
                            if (0.55..1.8).contains(&channel_ratio) {
                                channel_log_sums[index][channel] += (channel_ratio as f64).ln();
                            }
                        }
                    }
                    counts[index] += 1;
                }
            }
        }
    }
    // Some real scans use a very dark, nearly neutral paper whose warm-canvas
    // gate is intentionally conservative.  If that strict set is too small,
    // take a second robust pass over neutral overlap samples.  The fallback
    // rejects saturated artwork and trims both tails, so a red seal or a dark
    // ink stroke cannot become the tile-wide exposure estimate.
    if ratios.len() < 32 {
        let mut neutral_ratios = Vec::new();
        for y in (sample_start(top)..=bottom).step_by(sample_step) {
            for x in (sample_start(left)..=right).step_by(sample_step) {
                if ctx.panorama_mask.get_pixel(x, y)[0] == 0 {
                    continue;
                }
                let target = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
                let Some(source) = map_target_to_source(
                    ctx.candidate_inverse,
                    target,
                    ctx.candidate,
                    ctx.projection,
                ) else {
                    continue;
                };
                if source.x < 0.0
                    || source.y < 0.0
                    || source.x >= ctx.candidate_image.width() as f64 - 1.0
                    || source.y >= ctx.candidate_image.height() as f64 - 1.0
                {
                    continue;
                }
                let base_pixel = ctx.panorama.get_pixel(x, y);
                let candidate_pixel =
                    get_interpolated_pixel(ctx.candidate_image, source.x, source.y);
                let base_luma = luminance(base_pixel);
                let candidate_luma = luminance(&candidate_pixel);
                let base_chroma = base_pixel[0].max(base_pixel[1]).max(base_pixel[2])
                    - base_pixel[0].min(base_pixel[1]).min(base_pixel[2]);
                let candidate_chroma = candidate_pixel[0]
                    .max(candidate_pixel[1])
                    .max(candidate_pixel[2])
                    - candidate_pixel[0]
                        .min(candidate_pixel[1])
                        .min(candidate_pixel[2]);
                if !(0.025..0.92).contains(&base_luma)
                    || !(0.025..0.92).contains(&candidate_luma)
                    || base_chroma > 0.30
                    || candidate_chroma > 0.30
                {
                    continue;
                }
                let ratio = base_luma / candidate_luma;
                if (0.55..1.8).contains(&ratio) {
                    neutral_ratios.push(ratio);
                }
            }
        }
        neutral_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if neutral_ratios.len() >= 32 {
            let trim = neutral_ratios.len() / 5;
            ratios.extend(
                neutral_ratios[trim..neutral_ratios.len().saturating_sub(trim)]
                    .iter()
                    .copied(),
            );
        }
    }
    let representative_gain = if ratios.len() < 32 {
        1.0
    } else {
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ratios[ratios.len() / 2].clamp(0.75, 1.35)
    };
    let mut gains = vec![representative_gain; cell_count];
    let representative_channel_gain = std::array::from_fn(|channel| {
        if channel_ratios[channel].len() < 32 {
            representative_gain
        } else {
            channel_ratios[channel].sort_by(f32::total_cmp);
            channel_ratios[channel][channel_ratios[channel].len() / 2].clamp(0.75, 1.35)
        }
    });
    let mut channel_gains = vec![representative_channel_gain; cell_count];
    for index in 0..cell_count {
        if counts[index] >= 3 {
            gains[index] = (log_sums[index] / counts[index] as f64).exp().clamp(
                (representative_gain * 0.78) as f64,
                (representative_gain * 1.28) as f64,
            ) as f32;
            for channel in 0..3 {
                let channel_gain = (channel_log_sums[index][channel] / counts[index] as f64)
                    .exp()
                    .clamp(
                        (representative_channel_gain[channel] * 0.78) as f64,
                        (representative_channel_gain[channel] * 1.28) as f64,
                    );
                channel_gains[index][channel] = channel_gain as f32;
            }
        }
    }

    // The field models only broad illumination. Repeated neighbor averaging removes
    // local texture and registration noise while retaining vignetting and shadows.
    for _ in 0..5 {
        let mut smoothed = gains.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let mut weighted_sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        let weight = if dx == 0 && dy == 0 { 4.0 } else { 1.0 };
                        weighted_sum += gains[neighbor_index] * weight;
                        weight_sum += weight;
                    }
                }
                smoothed[grid_y * grid_width + grid_x] = weighted_sum / weight_sum;
            }
        }
        gains = smoothed;
    }

    ExposureCompensation {
        cell_size: CELL_SIZE,
        grid_width,
        grid_height,
        gains,
        representative_gain,
        channel_gains,
        representative_channel_gain,
    }
}

#[cfg(test)]
#[path = "stitching_interpolation_tests.rs"]
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
pub(crate) mod focus_fusion_test_access {
    use super::*;

    /// Per-cell normalised Sharpness_Score and coverage of one projected layer,
    /// exactly as the composite and every candidate of a Capture_Station are
    /// measured.
    pub(crate) fn measure_cells(
        geometry: &OwnershipGridGeometry,
        plan: &CellSamplingPlan,
        image: &Rgb32FImage,
        mask: &GrayImage,
        left: u32,
        top: u32,
    ) -> (Vec<f64>, Vec<bool>) {
        measure_focus_cells(geometry, plan, image, mask, left, top)
    }
}

/// 需求 12.1 / 12.2 / 12.5 on the *default* station path.
///
/// The decisions themselves are unit tested in
/// `stack_pipeline::station_degradation`; these tests cover the wiring, which is
/// where the same trap has been hit before: a rule connected to
/// `mosaic::detail_preserving_mosaic` alone never runs, because a Virtual_Tile
/// holds one Capture_Station and therefore always takes the standard focus
/// fusion branch of `focus_stack_stitcher_with_margin_policy`.
#[cfg(test)]
#[path = "stitching_station_degradation_wiring_tests.rs"]
mod station_degradation_wiring_tests;
