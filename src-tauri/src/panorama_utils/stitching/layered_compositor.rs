//! Layered Virtual_Tile compositor entry points and finishing switches.
//!
//! Public tile-ownership compositor entries, tile payload/source records, the
//! production finishing resolver and covered-boundary distances.

use super::super::coordinates::Projection;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;
use std::collections::HashMap;
use tauri::AppHandle;
use tauri::Runtime;

use super::{
    CoverageMask, FocusSourceRefinement, NO_OWNER, OwnershipMap,
    focus_tile_ownership_stitcher_with_finishing, get_high_quality_interpolated_pixel,
};

/// Decide whether a projected focus tile should replace the current owner at
/// one output pixel.  Interior distance is the primary quality signal; exact
/// ties are resolved by the stable tile identity so a compositor cannot make
/// the visible seam depend on the order in which groups happened to load.
pub(super) fn focus_tile_ownership_should_replace(
    current_quality: u8,
    current_owner: u8,
    candidate_quality: u8,
    candidate_owner: u8,
) -> bool {
    current_owner == 0
        || candidate_quality > current_quality
        || (candidate_quality == current_quality && candidate_owner < current_owner)
}

/// Compose already-fused focus tiles in one ownership pass.
///
/// The progressive seam compositor is appropriate for a short photographic
/// panorama, but it is order dependent: every new tile inherits the previous
/// tile's exposure estimate and seam decision.  A long scan made from virtual
/// focus tiles can therefore accumulate rectangular tone steps even when the
/// tile poses are correct.  This compositor keeps the geometry and ownership
/// decisions independent of import/path order.  In an overlap, the tile whose
/// source sample is farther from its own border owns the pixel; this favours
/// native, sharp interior pixels and prevents averaging two slightly displaced
/// focus results.
pub fn focus_tile_ownership_stitcher<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    focus_tile_ownership_stitcher_with_finishing(
        images,
        global_homographies,
        projection,
        TileCompositorFinishing::LegacyOwnership,
        app_handle,
        progress_event,
        load_image,
    )
    .map(LayeredOwnershipRender::into_image)
}

/// The default Tile_Compositor of the layered station pipeline (任务 13.1).
///
/// Geometry and ownership are decided exactly as in
/// [`focus_tile_ownership_stitcher`] — that compositor's semantics are what
/// became the default — but the finishing stage differs:
///
/// * no `crop_to_valid_rectangle()` fallback.  The canvas is the union
///   axis-aligned bounding box of the Coverage_Mask and nothing else
///   (需求 10.2 / 10.3, 任务 13.3).  Projective corners stay uncovered instead
///   of being cropped away.
/// * no unconditional final sharpen.  The default amount is 0.0 and the
///   environment variable is diagnostic only (需求 11.6 / 11.7, 任务 13.4).
///
/// Known defect on real material, to be repaired by the remaining stage-6 and
/// stage-3 tasks: source-interior ownership turns a projected tile footprint
/// into a visible polygon on a long scan.  任务 13.2 adds the
/// coverage-boundary-distance term to the seam cost and 任务 7.10 moves pixel
/// writes onto the hard Ownership_Map; until then a station boundary can be
/// visible where the two stations disagree photometrically.
#[allow(dead_code)]
pub fn layered_virtual_tile_compositor<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    layered_virtual_tile_compositor_with_ownership(
        images,
        global_homographies,
        projection,
        app_handle,
        progress_event,
        load_image,
    )
    .map(LayeredOwnershipRender::into_image)
}

/// The default layered compositor result before the public image-only API drops
/// its semantic planes. `sampling_origin` is the world/projected coordinate of
/// output pixel `(0, 0)` and therefore defines the exact Source_RAW
/// correspondence for a non-integer projective sample (需求 3.6).
#[derive(Debug, Clone)]
pub(crate) struct LayeredOwnershipRender {
    pub(crate) image: Rgb32FImage,
    pub(crate) coverage: CoverageMask,
    pub(crate) ownership: OwnershipMap,
    pub(crate) sampling_origin: (f64, f64),
    /// Test-only evidence from the production full-resolution sampling loop.
    /// This field is not present in release builds and cannot affect output.
    #[cfg(test)]
    pub(crate) full_resolution_samples: usize,
    /// Per-output-pixel sampling counts from the production loop. This is
    /// test-only evidence for checking Property 44 without aggregate masking.
    #[cfg(test)]
    pub(crate) full_resolution_sample_counts: Vec<u16>,
}

impl LayeredOwnershipRender {
    #[allow(dead_code)]
    pub(super) fn into_image(self) -> Rgb32FImage {
        self.image
    }
}

/// A station supplies geometry/ownership; delivered pixels can be sampled
/// directly from its prepared originals, without filtering the fused raster
/// a second time. Only one station's original buffers are resident here.
pub(crate) struct FocusTilePayload {
    pub(crate) image: Rgb32FImage,
    pub(crate) sources: Option<FocusTileSources>,
}

impl From<Rgb32FImage> for FocusTilePayload {
    fn from(image: Rgb32FImage) -> Self {
        Self {
            image,
            sources: None,
        }
    }
}

pub(crate) struct FocusTileSource {
    pub(crate) image: Rgb32FImage,
    pub(crate) refinement: FocusSourceRefinement,
}

pub(crate) struct FocusTileSources {
    pub(crate) ownership: OwnershipMap,
    pub(crate) origin: (f64, f64),
    /// Same legend order as the station's exact ownership plane.
    pub(crate) originals: Vec<FocusTileSource>,
}

impl FocusTileSources {
    pub(super) fn sample(&self, point: Point2<f64>) -> Option<Rgb<f32>> {
        let (width, height) = self.ownership.dimensions();
        let mut selected = None;
        let mut minimum_distance = f64::INFINITY;
        let rounded = (point.x.round(), point.y.round());
        for dy in -1..=1 {
            for dx in -1..=1 {
                let x = rounded.0 + f64::from(dx);
                let y = rounded.1 + f64::from(dy);
                if x < 0.0 || y < 0.0 || x >= f64::from(width) || y >= f64::from(height) {
                    continue;
                }
                let owner = self.ownership.owner_at(x as u32, y as u32);
                let Some(index) = owner.checked_sub(1) else {
                    continue;
                };
                let distance = (x - point.x).powi(2) + (y - point.y).powi(2);
                if distance < minimum_distance {
                    minimum_distance = distance;
                    selected = Some(usize::from(index));
                }
            }
        }
        let original = self.originals.get(selected?)?;
        let corrected = original.refinement.inverse_canvas_point(point);
        let source = original
            .refinement
            .source_sampling
            .as_ref()?
            .source_point(Point3::new(
                corrected.x + self.origin.0,
                corrected.y + self.origin.1,
                1.0,
            ))?;
        if source.x < 0.0
            || source.y < 0.0
            || source.x >= f64::from(original.image.width())
            || source.y >= f64::from(original.image.height())
        {
            return None;
        }
        Some(get_high_quality_interpolated_pixel(
            &original.image,
            source.x,
            source.y,
        ))
    }
}

/// Provenance-preserving form of [`layered_virtual_tile_compositor`]. The
/// image-only production entry point above calls this exact function; exposing
/// the masks prevents tests and later pipeline stages from reconstructing
/// ownership from pixel colours.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layered_virtual_tile_compositor_with_ownership<R: Runtime, F, T>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<LayeredOwnershipRender, String>
where
    F: FnMut(&ImageInfo) -> Result<T, String>,
    T: Into<FocusTilePayload>,
{
    focus_tile_ownership_stitcher_with_finishing(
        images,
        global_homographies,
        projection,
        TileCompositorFinishing::LayeredVirtualTile,
        app_handle,
        progress_event,
        load_image,
    )
}

/// How the ownership compositor finishes a composed canvas.
///
/// The two arms exist so the retired path keeps its exact pixels while the
/// default path drops the second-pass sharpen and the rectangle crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TileCompositorFinishing {
    /// Comparison path: 0.85 final sharpen plus the valid-rectangle crop.
    LegacyOwnership,
    /// Default path: no sharpen, no crop beyond the covered union bounds.
    LayeredVirtualTile,
}

/// The final sharpen amount of the default layered path (任务 13.4).
///
/// 需求 11.6 / 11.7 measure sharpness with MTF50 and gradient energy, so the
/// default path must not pre-sharpen its own evidence: the default is 0.0.
/// `RAW_EDITOR_FINAL_PANORAMA_SHARPEN_AMOUNT` survives as a diagnostic knob and
/// the resolved value is reported in `composition.final_sharpen_amount`.
pub(crate) fn layered_final_sharpen_amount() -> f32 {
    std::env::var("RAW_EDITOR_FINAL_PANORAMA_SHARPEN_AMOUNT")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.0)
        .clamp(0.0, 1.25)
}

/// Diagnostic switches of the ownership compositor. The retired comparison
/// path reads them once (`from_env`); the default layered path always uses
/// `Default`, all off, so no environment variable changes its output
/// (需求 15.2 / 15.10, 任务 13.7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct OwnershipCompositorSwitches {
    pub(super) tone_all_pixels: bool,
    pub(super) soft_owner_boundary: bool,
    pub(super) skip_exposure_gain: bool,
    pub(super) low_frequency_consensus: bool,
    pub(super) owner_tone_harmonization: bool,
    pub(super) skip_owner_tone_harmonization: bool,
    pub(super) final_low_frequency_illumination: bool,
    pub(super) legacy_final_sharpen_amount: f32,
}

impl Default for OwnershipCompositorSwitches {
    fn default() -> Self {
        Self {
            tone_all_pixels: false,
            soft_owner_boundary: false,
            skip_exposure_gain: false,
            low_frequency_consensus: false,
            owner_tone_harmonization: false,
            skip_owner_tone_harmonization: false,
            final_low_frequency_illumination: false,
            legacy_final_sharpen_amount: 0.85,
        }
    }
}

impl OwnershipCompositorSwitches {
    pub(super) fn from_env() -> Self {
        let set = |name: &str| std::env::var_os(name).is_some();
        Self {
            tone_all_pixels: set("RAW_EDITOR_FOCUS_TONE_ALL_PIXELS"),
            soft_owner_boundary: set("RAW_EDITOR_FOCUS_SOFT_OWNER_BOUNDARY"),
            skip_exposure_gain: set("RAW_EDITOR_SKIP_FOCUS_TILE_EXPOSURE_GAIN"),
            low_frequency_consensus: set("RAW_EDITOR_ENABLE_FOCUS_TILE_LOW_FREQUENCY_CONSENSUS"),
            owner_tone_harmonization: set("RAW_EDITOR_ENABLE_FOCUS_TILE_OWNER_TONE_HARMONIZATION"),
            skip_owner_tone_harmonization: set("RAW_EDITOR_SKIP_FOCUS_TONE_HARMONIZATION"),
            final_low_frequency_illumination: set(
                "RAW_EDITOR_ENABLE_FINAL_LOW_FREQUENCY_ILLUMINATION",
            ),
            legacy_final_sharpen_amount: std::env::var("RAW_EDITOR_FINAL_FOCUS_SHARPEN_AMOUNT")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite())
                .unwrap_or(0.85)
                .clamp(0.0, 1.25),
        }
    }
}

/// Resolve the finishing mode's switches at the production entry point.
/// Keeping this decision in one function lets the layered-path test exercise
/// the same branch used by the compositor, rather than reconstructing it in a
/// test-only helper.
pub(super) fn ownership_compositor_switches(
    finishing: TileCompositorFinishing,
) -> OwnershipCompositorSwitches {
    match finishing {
        TileCompositorFinishing::LegacyOwnership => OwnershipCompositorSwitches::from_env(),
        TileCompositorFinishing::LayeredVirtualTile => OwnershipCompositorSwitches::default(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tile_covered_boundary_distance(
    tile: &Rgb32FImage,
    sources: Option<&FocusTileSources>,
) -> GrayImage {
    let (w, h) = (tile.width() as usize, tile.height() as usize);
    let mut distance = vec![0u8; w * h];
    for (index, pixel) in tile.pixels().enumerate() {
        let covered = sources.map_or_else(
            || pixel[0].max(pixel[1]).max(pixel[2]) > 1e-6,
            |sources| {
                sources
                    .ownership
                    .owners()
                    .get(index)
                    .is_some_and(|&owner| owner != NO_OWNER)
            },
        );
        if covered {
            distance[index] = 255;
        }
    }
    // Eight-neighbour distance is a lower bound on Euclidean distance. The
    // exterior is uncovered even when the tile's first row is fully opaque.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if distance[i] == 0 {
                continue;
            }
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                distance[i] = 1;
            }
            for (dx, dy) in [(-1, 0), (-1, -1), (0, -1), (1, -1)] {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx >= 0 && ny >= 0 && nx < w as isize && ny < h as isize {
                    distance[i] =
                        distance[i].min(distance[ny as usize * w + nx as usize].saturating_add(1));
                }
            }
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if distance[i] == 0 {
                continue;
            }
            for (dx, dy) in [(1, 0), (1, 1), (0, 1), (-1, 1)] {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx >= 0 && ny >= 0 && nx < w as isize && ny < h as isize {
                    distance[i] =
                        distance[i].min(distance[ny as usize * w + nx as usize].saturating_add(1));
                }
            }
        }
    }
    GrayImage::from_raw(w as u32, h as u32, distance).unwrap()
}

pub(super) fn tile_boundary_distance_world(
    distance: &GrayImage,
    source: Point2<f64>,
    pose: &Matrix3<f64>,
    nonrigid: bool,
) -> f64 {
    if source.x < 0.0
        || source.y < 0.0
        || source.x >= distance.width() as f64
        || source.y >= distance.height() as f64
    {
        return 0.0;
    }
    let x = source
        .x
        .round()
        .clamp(0.0, distance.width().saturating_sub(1) as f64) as u32;
    let y = source
        .y
        .round()
        .clamp(0.0, distance.height().saturating_sub(1) as f64) as u32;
    let world = pose * Point3::new(source.x, source.y, 1.0);
    if !world.z.is_finite() || world.z.abs() < 1e-8 {
        return 0.0;
    }
    let (wx, wy) = (world.x / world.z, world.y / world.z);
    let a = (pose[(0, 0)] - wx * pose[(2, 0)]) / world.z;
    let b = (pose[(0, 1)] - wx * pose[(2, 1)]) / world.z;
    let c = (pose[(1, 0)] - wy * pose[(2, 0)]) / world.z;
    let d = (pose[(1, 1)] - wy * pose[(2, 1)]) / world.z;
    let norm = (a * a + b * b + c * c + d * d).sqrt();
    if !norm.is_finite() || norm <= 1e-8 {
        return 0.0;
    }
    // |det J| / ||J||F is a conservative lower singular-value bound. The
    // residual grid's 32/8 limits plus its 128px smoothstep permit at most
    // 0.625 operator deviation, hence the 0.375 lower scale bound.
    let scale = (a * d - b * c).abs() / norm * if nonrigid { 0.375 } else { 1.0 };
    (f64::from(distance.get_pixel(x, y)[0]) - 2.0).max(0.0) * scale
}
