use super::photometric::{PhotometricModel, PhotometricOptions, calibrate_overlap_photometry};
use super::stack_pipeline::compositor::tile_coordinate_with_residual;
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
use crate::panorama_stitching::{FOCUS_FOREGROUND_LUMA_THRESHOLD, FocusLayerWarp, ImageInfo};
use image::{GrayImage, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use tauri::{AppHandle, Emitter, Runtime};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    Planar,
    Cylindrical,
    Spherical,
}

pub fn project_point(
    image: &ImageInfo,
    x: f64,
    y: f64,
    projection: Projection,
) -> Option<Point2<f64>> {
    let width = image.width() as f64;
    let height = image.height() as f64;
    let center_x = width * 0.5;
    let center_y = height * 0.5;
    let focal = width.max(height).max(1.0) * 0.85;
    let normalized_x = (x - center_x) / focal;
    let normalized_y = (y - center_y) / focal;

    let (projected_x, projected_y) = match projection {
        Projection::Planar => (x, y),
        Projection::Cylindrical => (
            normalized_x.atan() * focal + center_x,
            (normalized_y / (1.0 + normalized_x * normalized_x).sqrt()) * focal + center_y,
        ),
        Projection::Spherical => (
            normalized_x.atan() * focal + center_x,
            (normalized_y / (1.0 + normalized_x * normalized_x).sqrt()).atan() * focal + center_y,
        ),
    };

    if projected_x.is_finite() && projected_y.is_finite() {
        Some(Point2::new(projected_x, projected_y))
    } else {
        None
    }
}

fn unproject_point(
    image: &ImageInfo,
    x: f64,
    y: f64,
    projection: Projection,
) -> Option<Point2<f64>> {
    if projection == Projection::Planar {
        return if x.is_finite() && y.is_finite() {
            Some(Point2::new(x, y))
        } else {
            None
        };
    }

    let width = image.width() as f64;
    let height = image.height() as f64;
    let center_x = width * 0.5;
    let center_y = height * 0.5;
    let focal = width.max(height).max(1.0) * 0.85;
    let projected_x = (x - center_x) / focal;
    let projected_y = (y - center_y) / focal;

    let normalized_x = projected_x.tan();
    let vertical_scale = (1.0 + normalized_x * normalized_x).sqrt();
    let normalized_y = match projection {
        Projection::Cylindrical => projected_y * vertical_scale,
        Projection::Spherical => projected_y.tan() * vertical_scale,
        Projection::Planar => unreachable!("planar projection returned above"),
    };
    let source_x = normalized_x * focal + center_x;
    let source_y = normalized_y * focal + center_y;

    if source_x.is_finite() && source_y.is_finite() {
        Some(Point2::new(source_x, source_y))
    } else {
        None
    }
}

pub(super) fn map_target_to_source(
    inverse_homography: &Matrix3<f64>,
    target: Point3<f64>,
    image: &ImageInfo,
    projection: Projection,
) -> Option<Point2<f64>> {
    let projected_source = inverse_homography * target;
    if projected_source.z.abs() < 1e-8 {
        return None;
    }
    unproject_point(
        image,
        projected_source.x / projected_source.z,
        projected_source.y / projected_source.z,
        projection,
    )
}

/// Map one output-world coordinate into a source tile while applying the
/// optional run-scoped residual field. The empty model takes the exact legacy
/// path. Planar output uses the compositor's single composed inverse; curved
/// projections first move the world target and retain their existing inverse
/// projection.
pub(crate) fn map_target_to_source_with_residual(
    inverse_homography: &Matrix3<f64>,
    target: Point3<f64>,
    image: &ImageInfo,
    projection: Projection,
    residual: &residual_warp::ResidualWarp,
) -> Option<Point2<f64>> {
    if residual.identity() {
        return map_target_to_source(inverse_homography, target, image, projection);
    }
    let world = Point2::new(target.x, target.y);
    if projection == Projection::Planar {
        return tile_coordinate_with_residual(world, inverse_homography, residual, image.id);
    }
    let warped = residual.warp_inverse(world, image.id);
    map_target_to_source(
        inverse_homography,
        Point3::new(warped.x, warped.y, target.z),
        image,
        projection,
    )
}

pub(super) fn output_bounds(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for &image in images {
        let h = global_homographies[&image.id];
        let (width, height) = image.dimensions();
        let corners = [
            (0.0, 0.0),
            (width as f64, 0.0),
            (width as f64, height as f64),
            (0.0, height as f64),
        ];
        for (x, y) in corners {
            let Some(projected) = project_point(image, x, y, projection) else {
                continue;
            };
            let mapped = h * Point3::new(projected.x, projected.y, 1.0);
            if mapped.z.abs() < 1e-8 {
                continue;
            }
            let mapped_x = mapped.x / mapped.z;
            let mapped_y = mapped.y / mapped.z;
            min_x = min_x.min(mapped_x);
            max_x = max_x.max(mapped_x);
            min_y = min_y.min(mapped_y);
            max_y = max_y.max(mapped_y);
        }
    }

    (min_x, max_x, min_y, max_y)
}

pub(super) fn pixel_aligned_canvas(minimum: f64, maximum: f64) -> (f64, u32) {
    // The first image is the global reference and normally has an identity
    // transform. Keep its samples on integer output coordinates; using the exact
    // fractional bound as the offset would unnecessarily interpolate every
    // reference pixel.
    let offset = (-minimum).ceil();
    let size = (maximum + offset).ceil().max(1.0) as u32;
    (offset, size)
}

pub(crate) fn output_canvas_dimensions(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> (u32, u32) {
    if images.is_empty() {
        return (0, 0);
    }
    let (min_x, max_x, min_y, max_y) = output_bounds(images, global_homographies, projection);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return (0, 0);
    }
    let (_, width) = pixel_aligned_canvas(min_x, max_x);
    let (_, height) = pixel_aligned_canvas(min_y, max_y);
    (width, height)
}

pub(crate) fn output_canvas_dimensions_with_focus_warp(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
) -> (u32, u32) {
    if images.is_empty() {
        return (0, 0);
    }
    let (min_x, max_x, min_y, max_y) =
        focus_output_bounds(images, global_homographies, projection, focus_warp);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return (0, 0);
    }
    let (_, width) = pixel_aligned_canvas(min_x, max_x);
    let (_, height) = pixel_aligned_canvas(min_y, max_y);
    (width, height)
}

/// Reserved Ownership_Map identifier for a pixel that no Source_RAW owns
/// (需求 3.7).  Every other identifier is `source index + 1`, so
/// `legend[id - 1]` is the owning file.
pub(crate) const NO_OWNER: u16 = 0;

/// Colour encoding of the Virtual_Tile pixels (需求 4.2).  The focus fuser
/// copies decoded source pixels, so the tile inherits the encoding of the
/// render pipeline rather than choosing one of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorEncoding {
    LinearSrgb,
    DisplaySrgb,
}

/// Maps every Virtual_Tile pixel to at most one Source_RAW (需求 4.1 / 4.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OwnershipMap {
    width: u32,
    height: u32,
    owners: Vec<u16>,
    /// `legend[id - 1]` is the absolute path of owner identifier `id`.
    legend: Vec<PathBuf>,
}

impl OwnershipMap {
    pub(crate) fn new(
        width: u32,
        height: u32,
        owners: Vec<u16>,
        legend: Vec<PathBuf>,
    ) -> Result<Self, String> {
        let expected = width as usize * height as usize;
        if owners.len() != expected {
            return Err(format!(
                "ownership map holds {} entries for a {}x{} tile",
                owners.len(),
                width,
                height
            ));
        }
        if let Some(&invalid) = owners.iter().find(|&&owner| owner as usize > legend.len()) {
            return Err(format!(
                "ownership map references owner {} outside a legend of {} source(s)",
                invalid,
                legend.len()
            ));
        }
        Ok(Self {
            width,
            height,
            owners,
            legend,
        })
    }

    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub(crate) fn owners(&self) -> &[u16] {
        &self.owners
    }

    pub(crate) fn legend(&self) -> &[PathBuf] {
        &self.legend
    }

    pub(crate) fn owner_at(&self, x: u32, y: u32) -> u16 {
        if x >= self.width || y >= self.height {
            return NO_OWNER;
        }
        self.owners[y as usize * self.width as usize + x as usize]
    }

    /// Owned pixel count per legend entry, in legend order (需求 4.1).
    pub(crate) fn owned_pixel_counts(&self) -> Vec<u64> {
        let mut counts = vec![0u64; self.legend.len()];
        for &owner in &self.owners {
            if owner == NO_OWNER {
                continue;
            }
            counts[owner as usize - 1] += 1;
        }
        counts
    }

    pub(crate) fn assigned_pixels(&self) -> u64 {
        self.owners
            .iter()
            .filter(|&&owner| owner != NO_OWNER)
            .count() as u64
    }
}

/// Binary Coverage_Mask: whether a Virtual_Tile pixel is covered by at least
/// one valid source projection (需求 3.7 / 4.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CoverageMask {
    width: u32,
    height: u32,
    /// Non-zero means covered; the focus renderer already keeps this mask as
    /// an 8-bit image, so the tile stores it without a re-encoding step.
    covered: Vec<u8>,
}

impl CoverageMask {
    pub(crate) fn from_gray(mask: GrayImage) -> Self {
        let (width, height) = mask.dimensions();
        Self {
            width,
            height,
            covered: mask.into_raw(),
        }
    }

    /// Rebuild a mask from a raw coverage plane, one byte per pixel, non-zero
    /// meaning covered.  Used by the Virtual_Tile disk cache, which stores the
    /// mask as one bit per pixel and expands it on read (需求 4.3).
    pub(crate) fn from_bytes(width: u32, height: u32, covered: Vec<u8>) -> Result<Self, String> {
        let expected = width as usize * height as usize;
        if covered.len() != expected {
            return Err(format!(
                "coverage mask holds {} entries for a {}x{} tile",
                covered.len(),
                width,
                height
            ));
        }
        Ok(Self {
            width,
            height,
            covered,
        })
    }

    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub(crate) fn is_covered(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.covered[y as usize * self.width as usize + x as usize] > 0
    }

    pub(crate) fn covered(&self) -> &[u8] {
        &self.covered
    }

    pub(crate) fn covered_pixels(&self) -> u64 {
        self.covered.iter().filter(|&&value| value > 0).count() as u64
    }

    /// Union bounds of the covered pixels as `(x, y, width, height)`
    /// (需求 4.7).  `None` when nothing is covered.
    pub(crate) fn covered_bounds(&self) -> Option<(u32, u32, u32, u32)> {
        let width = self.width as usize;
        let mut min_x = self.width;
        let mut min_y = self.height;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        let mut any = false;
        for (index, &value) in self.covered.iter().enumerate() {
            if value == 0 {
                continue;
            }
            any = true;
            let x = (index % width) as u32;
            let y = (index / width) as u32;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        any.then(|| (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
    }
}

/// Sharpness_Confidence per Virtual_Tile pixel (需求 4.6).  The Focus_Fuser
/// produces one value per ownership cell and the station render expands it to
/// `PerPixel`; `Uniform` stays for the paths that never ran a fusion (a
/// cache-restored tile of uniform confidence, and the test fixtures), where it
/// keeps the required geometry without allocating a full-canvas `f32` plane.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ConfidenceMap {
    Uniform {
        width: u32,
        height: u32,
        value: f32,
    },
    PerPixel {
        width: u32,
        height: u32,
        values: Vec<f32>,
    },
}

impl ConfidenceMap {
    /// Placeholder map: every pixel reads 0.0 (需求 3.9 treats "no runner-up
    /// evidence" as zero confidence).
    #[cfg(test)]
    pub(crate) fn zero(width: u32, height: u32) -> Self {
        Self::Uniform {
            width,
            height,
            value: 0.0,
        }
    }

    pub(crate) fn per_pixel(width: u32, height: u32, values: Vec<f32>) -> Result<Self, String> {
        let expected = width as usize * height as usize;
        if values.len() != expected {
            return Err(format!(
                "confidence map holds {} entries for a {}x{} tile",
                values.len(),
                width,
                height
            ));
        }
        Ok(Self::PerPixel {
            width,
            height,
            values,
        })
    }

    pub(crate) fn dimensions(&self) -> (u32, u32) {
        match *self {
            Self::Uniform { width, height, .. } | Self::PerPixel { width, height, .. } => {
                (width, height)
            }
        }
    }

    pub(crate) fn value_at(&self, x: u32, y: u32) -> f32 {
        let (width, height) = self.dimensions();
        if x >= width || y >= height {
            return 0.0;
        }
        match self {
            Self::Uniform { value, .. } => *value,
            Self::PerPixel { values, .. } => values[y as usize * width as usize + x as usize],
        }
    }

    /// Row-major copy of every confidence value.  The Virtual_Tile disk cache
    /// stores one plane whatever representation the map happens to use.
    pub(crate) fn to_row_major(&self) -> Vec<f32> {
        let (width, height) = self.dimensions();
        let count = width as usize * height as usize;
        match self {
            Self::Uniform { value, .. } => vec![*value; count],
            Self::PerPixel { values, .. } => values.clone(),
        }
    }

    fn in_unit_range(&self) -> bool {
        match self {
            Self::Uniform { value, .. } => value.is_finite() && (0.0..=1.0).contains(value),
            Self::PerPixel { values, .. } => values
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
        }
    }
}

/// Source_RAW provenance of one Virtual_Tile (需求 4.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceProvenance {
    pub(crate) absolute_path: PathBuf,
    /// SHA-256 over every byte of the source file, taken with the read-only
    /// digest of `virtual_tile::source_file_sha256`.  Filled by the fusion call
    /// site that builds the tile; `None` until that stage lands.
    pub(crate) sha256: Option<[u8; 32]>,
    pub(crate) owned_pixels: u64,
}

/// A fully focus-fused capture station in the coordinate system of the
/// complete mosaic.  The tile pixels are owned by the station's focus
/// decision; `tile_to_world` only places that result in the global planar
/// coordinate system.  Keeping the ownership, coverage, confidence and
/// provenance beside the pixels lets the caller perform tile-level
/// registration and seam/ownership, and lets every output pixel be traced
/// back to one real Source_RAW, without falling back to the individual focus
/// layers.
#[derive(Debug, Clone)]
pub(crate) struct VirtualTile {
    pub(crate) station_index: usize,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tile_to_world: Matrix3<f64>,
    pub(crate) pixels: Rgb32FImage,
    pub(crate) ownership: OwnershipMap,
    pub(crate) sharpness_confidence: ConfidenceMap,
    pub(crate) coverage: CoverageMask,
    pub(crate) color_encoding: ColorEncoding,
    pub(crate) provenance: Vec<SourceProvenance>,
}

impl VirtualTile {
    /// Build a Virtual_Tile, enforcing the four structural invariants of the
    /// design document:
    ///
    /// 1. `provenance` has one entry per participating Source_RAW and matches
    ///    the Ownership_Map legend (需求 4.1);
    /// 2. every `provenance[i].owned_pixels` equals the number of pixels owned
    ///    by that source, so the sum equals the number of pixels with an owner
    ///    (需求 4.1);
    /// 3. Coverage_Mask and Ownership_Map agree in both directions and share
    ///    the pixel geometry of `pixels` (需求 3.11 / 4.6);
    /// 4. the valid pixel extent equals the union bounds of the covered pixels
    ///    (需求 4.7).
    pub(crate) fn new(
        station_index: usize,
        tile_to_world: Matrix3<f64>,
        pixels: Rgb32FImage,
        ownership: OwnershipMap,
        sharpness_confidence: ConfidenceMap,
        coverage: CoverageMask,
        color_encoding: ColorEncoding,
        provenance: Vec<SourceProvenance>,
    ) -> Result<Self, String> {
        let (width, height) = pixels.dimensions();
        // Invariant 3a: one pixel coordinate system for pixels and masks.
        if ownership.dimensions() != (width, height) {
            return Err(format!(
                "ownership map is {:?} for a {}x{} tile",
                ownership.dimensions(),
                width,
                height
            ));
        }
        if coverage.dimensions() != (width, height) {
            return Err(format!(
                "coverage mask is {:?} for a {}x{} tile",
                coverage.dimensions(),
                width,
                height
            ));
        }
        if sharpness_confidence.dimensions() != (width, height) {
            return Err(format!(
                "confidence map is {:?} for a {}x{} tile",
                sharpness_confidence.dimensions(),
                width,
                height
            ));
        }
        if !sharpness_confidence.in_unit_range() {
            return Err("confidence values must stay inside [0, 1]".to_string());
        }
        // Invariant 1: provenance is the Ownership_Map legend.
        if provenance.len() != ownership.legend().len() {
            return Err(format!(
                "{} provenance record(s) for an ownership legend of {} source(s)",
                provenance.len(),
                ownership.legend().len()
            ));
        }
        if let Some((index, record)) = provenance
            .iter()
            .enumerate()
            .find(|(index, record)| ownership.legend()[*index] != record.absolute_path)
        {
            return Err(format!(
                "provenance {} ({}) does not match its ownership legend entry ({})",
                index,
                record.absolute_path.display(),
                ownership.legend()[index].display()
            ));
        }
        // Invariant 2: the provenance pixel counts are the Ownership_Map.
        let counts = ownership.owned_pixel_counts();
        if let Some((index, record)) = provenance
            .iter()
            .enumerate()
            .find(|(index, record)| counts[*index] != record.owned_pixels)
        {
            return Err(format!(
                "provenance {} ({}) claims {} owned pixel(s) but owns {}",
                index,
                record.absolute_path.display(),
                record.owned_pixels,
                counts[index]
            ));
        }
        // Invariant 3b: covered <=> owned, in both directions.
        if let Some(index) = coverage
            .covered()
            .iter()
            .zip(ownership.owners())
            .position(|(&covered, &owner)| (covered > 0) != (owner != NO_OWNER))
        {
            let x = index % width.max(1) as usize;
            let y = index / width.max(1) as usize;
            return Err(format!(
                "pixel ({x}, {y}) is covered={} but owned={}",
                coverage.covered()[index] > 0,
                ownership.owners()[index] != NO_OWNER
            ));
        }
        // Invariant 4: the valid extent is exactly the covered union bounds.
        // Everything outside those bounds must stay untouched, so the tile
        // never carries a pixel that no source produced.
        match coverage.covered_bounds() {
            Some((min_x, min_y, bounds_width, bounds_height)) => {
                let tight =
                    min_x == 0 && min_y == 0 && bounds_width == width && bounds_height == height;
                if !tight {
                    let outside_is_clear = pixels.enumerate_pixels().all(|(x, y, pixel)| {
                        let inside = x >= min_x
                            && y >= min_y
                            && x < min_x + bounds_width
                            && y < min_y + bounds_height;
                        inside || pixel.0 == [0.0, 0.0, 0.0]
                    });
                    if !outside_is_clear {
                        return Err(format!(
                            "tile pixels extend beyond the coverage union bounds ({min_x}, {min_y}, {bounds_width}, {bounds_height})"
                        ));
                    }
                }
            }
            None => {
                if pixels.as_raw().iter().any(|value| *value != 0.0) {
                    return Err("tile has pixels but no coverage".to_string());
                }
            }
        }
        Ok(Self {
            station_index,
            width,
            height,
            tile_to_world,
            pixels,
            ownership,
            sharpness_confidence,
            coverage,
            color_encoding,
            provenance,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FocusVirtualTileGeometry {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tile_to_world: Matrix3<f64>,
}

/// Return the projected bounds needed to place one virtual tile without
/// decoding or fusing any source pixels.  This is used by the bounded-memory
/// compositor to create tile metadata first and render each group lazily.
pub(crate) fn focus_stack_virtual_tile_geometry(
    group: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
) -> Option<FocusVirtualTileGeometry> {
    if group.is_empty() {
        return None;
    }
    if group
        .iter()
        .any(|image| !global_homographies.contains_key(&image.id))
    {
        return None;
    }
    let (min_x, max_x, min_y, max_y) =
        focus_output_bounds(group, global_homographies, projection, focus_warp);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return None;
    }
    let (offset_x, width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, height) = pixel_aligned_canvas(min_y, max_y);
    Some(FocusVirtualTileGeometry {
        width,
        height,
        tile_to_world: Matrix3::new(1.0, 0.0, -offset_x, 0.0, 1.0, -offset_y, 0.0, 0.0, 1.0),
    })
}

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

pub(super) fn transformed_image_region(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    if out_width == 0 || out_height == 0 {
        return None;
    }
    let (width, height) = image.dimensions();
    let corners = [
        (0.0, 0.0),
        (width as f64, 0.0),
        (width as f64, height as f64),
        (0.0, height as f64),
    ];
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in corners {
        let projected = project_point(image, x, y, projection)?;
        let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            continue;
        }
        let mapped_x = mapped.x / mapped.z + offset_x;
        let mapped_y = mapped.y / mapped.z + offset_y;
        if !mapped_x.is_finite() || !mapped_y.is_finite() {
            continue;
        }
        min_x = min_x.min(mapped_x);
        max_x = max_x.max(mapped_x);
        min_y = min_y.min(mapped_y);
        max_y = max_y.max(mapped_y);
    }
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return None;
    }
    let left = min_x.floor().max(0.0).min((out_width - 1) as f64) as u32;
    let right = max_x.ceil().max(0.0).min((out_width - 1) as f64) as u32;
    let top = min_y.floor().max(0.0).min((out_height - 1) as f64) as u32;
    let bottom = max_y.ceil().max(0.0).min((out_height - 1) as f64) as u32;
    (left <= right && top <= bottom).then_some((left, right, top, bottom))
}

fn transformed_source_bounds(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    minimum_source_y: f64,
    maximum_source_y: f64,
) -> Option<(f64, f64, f64, f64)> {
    transformed_source_bounds_xy(
        image,
        homography,
        projection,
        0.0,
        1.0,
        minimum_source_y,
        maximum_source_y,
    )
}

fn transformed_source_bounds_xy(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    minimum_source_x: f64,
    maximum_source_x: f64,
    minimum_source_y: f64,
    maximum_source_y: f64,
) -> Option<(f64, f64, f64, f64)> {
    let (width, height) = image.dimensions();
    let x0 = (width as f64 * minimum_source_x).clamp(0.0, width as f64);
    let x1 = (width as f64 * maximum_source_x).clamp(0.0, width as f64);
    let y0 = (height as f64 * minimum_source_y).clamp(0.0, height as f64);
    let y1 = (height as f64 * maximum_source_y).clamp(0.0, height as f64);
    let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in corners {
        let projected = project_point(image, x, y, projection)?;
        let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            continue;
        }
        let mapped_x = mapped.x / mapped.z;
        let mapped_y = mapped.y / mapped.z;
        if !mapped_x.is_finite() || !mapped_y.is_finite() {
            continue;
        }
        min_x = min_x.min(mapped_x);
        max_x = max_x.max(mapped_x);
        min_y = min_y.min(mapped_y);
        max_y = max_y.max(mapped_y);
    }
    if min_x.is_finite() && max_x.is_finite() && min_y.is_finite() && max_y.is_finite() {
        Some((min_x, max_x, min_y, max_y))
    } else {
        None
    }
}

fn focus_output_bounds(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
) -> (f64, f64, f64, f64) {
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        output_bounds(images, global_homographies, projection);
    let Some(focus_warp) = focus_warp else {
        return (min_x, max_x, min_y, max_y);
    };
    for image in images {
        for band in &focus_warp.bands {
            let Some(homography) = band.homographies.get(&image.id) else {
                continue;
            };
            let Some(&(minimum_source_y, maximum_source_y)) = band.source_ranges.get(&image.id)
            else {
                continue;
            };
            let (minimum_source_x, maximum_source_x) = band
                .source_x_ranges
                .get(&image.id)
                .copied()
                .unwrap_or((0.0, 1.0));
            let Some((band_min_x, band_max_x, band_min_y, band_max_y)) =
                transformed_source_bounds_xy(
                    image,
                    homography,
                    projection,
                    minimum_source_x,
                    maximum_source_x,
                    minimum_source_y,
                    maximum_source_y,
                )
            else {
                continue;
            };
            min_x = min_x.min(band_min_x);
            max_x = max_x.max(band_max_x);
            min_y = min_y.min(band_min_y);
            max_y = max_y.max(band_max_y);
        }
    }
    (min_x, max_x, min_y, max_y)
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

struct SeamContext<'a> {
    pano: &'a Rgb32FImage,
    pano_mask: &'a GrayImage,
    img_to_add_info: &'a ImageInfo,
    img_to_add: &'a Rgb32FImage,
    h_add: &'a Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
    exposure: &'a ExposureCompensation,
}

#[derive(Clone, Copy)]
enum SeamOrientation {
    Vertical,
    Horizontal,
}

struct SeamInfo {
    orientation: SeamOrientation,
    coords: Vec<i32>,
    dx: f64,
    dy: f64,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
}

pub fn progressive_seam_stitcher<R: Runtime, F>(
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
    // Keep the tone/residual ledgers scoped to this compositor invocation.
    // The panorama entry point also resets them at whole-run start (see the
    // handoff snippet in the stage5 report).
    tone::reset_run_records();
    residual_warp::reset_run_records();
    if images.is_empty() {
        return Ok(Rgb32FImage::new(0, 0));
    }

    let (min_x, max_x, min_y, max_y) = output_bounds(images, global_homographies, projection);

    let (offset_x, out_width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, out_height) = pixel_aligned_canvas(min_y, max_y);
    println!("  - Output canvas size: {}x{}", out_width, out_height);

    let mut panorama = Rgb32FImage::new(out_width, out_height);
    let mut panorama_mask = GrayImage::new(out_width, out_height);

    let base_img_info = images[0];
    let base_image = load_image(base_img_info)?;
    let h_base = &global_homographies[&base_img_info.id];
    let h_base_inv = h_base
        .try_inverse()
        .ok_or_else(|| "The base image alignment is not invertible.".to_string())?;
    println!("  - Placing base image: '{}'", base_img_info.filename);

    let num_pixels_per_row = out_width as usize * 3;
    let (base_left, base_right, base_top, base_bottom) = transformed_image_region(
        base_img_info,
        h_base,
        projection,
        offset_x,
        offset_y,
        out_width,
        out_height,
    )
    .unwrap_or((
        0,
        out_width.saturating_sub(1),
        0,
        out_height.saturating_sub(1),
    ));
    panorama
        .par_chunks_mut(num_pixels_per_row)
        .zip(panorama_mask.par_chunks_mut(out_width as usize))
        .enumerate()
        .skip(base_top as usize)
        .take((base_bottom - base_top + 1) as usize)
        .for_each(|(y, (row_slice, mask_row))| {
            for x in base_left..=base_right {
                let target_p = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
                if let Some(source) =
                    map_target_to_source(&h_base_inv, target_p, base_img_info, projection)
                {
                    let sx = source.x;
                    let sy = source.y;
                    if sx < 0.0
                        || sx >= base_image.width() as f64
                        || sy < 0.0
                        || sy >= base_image.height() as f64
                    {
                        continue;
                    }
                    let color = get_high_quality_interpolated_pixel(&base_image, sx, sy);
                    // Projected corners in a rendered focus tile are
                    // transparent black. They are outside the photographed
                    // support and must not become valid panorama pixels.
                    if color[0].max(color[1]).max(color[2]) <= 1e-6 {
                        continue;
                    }
                    let start = x as usize * 3;
                    row_slice[start..start + 3].copy_from_slice(&color.0);
                    mask_row[x as usize] = 255;
                }
            }
        });
    drop(base_image);

    for (i, &img_to_add_info) in images.iter().skip(1).enumerate() {
        let progress_msg = format!(
            "Stitching image {} of {}: {}",
            i + 2,
            images.len(),
            Path::new(&img_to_add_info.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        );
        let _ = app_handle.emit(progress_event, &progress_msg);
        println!("  - Progressively stitching '{}'", img_to_add_info.filename);

        let h_add = &global_homographies[&img_to_add_info.id];
        let h_add_inv = h_add.try_inverse().ok_or_else(|| {
            format!(
                "The alignment for '{}' is not invertible.",
                img_to_add_info.filename
            )
        })?;
        let img_to_add = load_image(img_to_add_info)?;
        let (candidate_left, candidate_right, candidate_top, candidate_bottom) =
            transformed_image_region(
                img_to_add_info,
                h_add,
                projection,
                offset_x,
                offset_y,
                out_width,
                out_height,
            )
            .unwrap_or((
                0,
                out_width.saturating_sub(1),
                0,
                out_height.saturating_sub(1),
            ));
        let exposure = if std::env::var_os("RAW_EDITOR_SKIP_PROGRESSIVE_EXPOSURE_GAIN").is_some() {
            ExposureCompensation {
                cell_size: 1,
                grid_width: 1,
                grid_height: 1,
                gains: vec![1.0],
                representative_gain: 1.0,
                channel_gains: vec![[1.0; 3]],
                representative_channel_gain: [1.0; 3],
            }
        } else {
            let estimated = estimate_overlap_exposure_compensation(ExposureOverlap {
                panorama: &panorama,
                panorama_mask: &panorama_mask,
                candidate: img_to_add_info,
                candidate_image: &img_to_add,
                candidate_inverse: &h_add_inv,
                projection,
                offset_x,
                offset_y,
            });
            let strength = std::env::var("RAW_EDITOR_PROGRESSIVE_EXPOSURE_STRENGTH")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite())
                .unwrap_or(1.0);
            estimated.damped(strength)
        };
        println!(
            "    - Overlap exposure gain: {:.3} with local illumination correction",
            exposure.representative_gain
        );

        let ctx = SeamContext {
            pano: &panorama,
            pano_mask: &panorama_mask,
            img_to_add_info,
            img_to_add: &img_to_add,
            h_add,
            projection,
            offset_x,
            offset_y,
            out_width,
            out_height,
            exposure: &exposure,
        };
        let seam_info = find_adaptive_seam(&ctx);

        let use_seam = if let Some(ref info) = seam_info {
            !info.coords.is_empty()
        } else {
            false
        };

        if !use_seam {
            println!("    - Warning: Could not find seam. Using simple overwrite.");
            // Observation only (requirement 12.7/12.9): the overwrite fallback
            // below is unchanged, the ledger just learns that this pair had no
            // usable overlap for a seam.
            degradation::record_run_degradation(
                degradation::COMPOSITION_NARROW_OVERLAP,
                serde_json::json!({
                    "stage": "pairwise_seam",
                    "trigger": "seam_unavailable",
                    "image": img_to_add_info.filename,
                }),
            );
        }

        let (orientation, seam_coords, new_image_is_dominant_side, seam_bounds) =
            if let Some(info) = seam_info {
                let dominant = match info.orientation {
                    SeamOrientation::Vertical => info.dx > 0.0,
                    SeamOrientation::Horizontal => info.dy > 0.0,
                };
                (
                    info.orientation,
                    info.coords,
                    dominant,
                    Some((info.min_x, info.max_x, info.min_y, info.max_y)),
                )
            } else {
                (SeamOrientation::Vertical, vec![], true, None)
            };

        // Extend the overlap correction smoothly into the newly covered side
        // of the candidate. A hard switch at the overlap bounding box makes
        // the low-frequency seam patch visible as a rectangle; this bounded
        // fade keeps the measured correction at the seam and returns to the
        // source exposure over part of one tile.
        let exposure_transition = seam_bounds.map(|(min_x, max_x, min_y, max_y)| {
            let axis_span = match orientation {
                SeamOrientation::Vertical => candidate_right.saturating_sub(candidate_left) + 1,
                SeamOrientation::Horizontal => candidate_bottom.saturating_sub(candidate_top) + 1,
            } as f32;
            let fade_length = (axis_span * 0.35).max(128.0);
            (min_x, max_x, min_y, max_y, fade_length)
        });

        if use_seam {
            let side = match orientation {
                SeamOrientation::Vertical => {
                    if new_image_is_dominant_side {
                        "right"
                    } else {
                        "left"
                    }
                }
                SeamOrientation::Horizontal => {
                    if new_image_is_dominant_side {
                        "bottom"
                    } else {
                        "top"
                    }
                }
            };
            println!("    - New image is on the {} side of the seam.", side);
        }

        panorama
            .par_chunks_mut(num_pixels_per_row)
            .zip(panorama_mask.par_chunks_mut(out_width as usize))
            .enumerate()
            .skip(candidate_top as usize)
            .take((candidate_bottom - candidate_top + 1) as usize)
            .for_each(|(y, (row_slice, mask_row))| {
                for x in candidate_left..=candidate_right {
                    let target_p = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
                    let Some(source_add) =
                        map_target_to_source(&h_add_inv, target_p, img_to_add_info, projection)
                    else {
                        continue;
                    };
                    let sx = source_add.x;
                    let sy = source_add.y;
                    let is_on_add = sx >= 0.0
                        && sx < img_to_add.width() as f64
                        && sy >= 0.0
                        && sy < img_to_add.height() as f64;
                    let is_on_pano = mask_row[x as usize] > 0;

                    if !is_on_add && !is_on_pano {
                        continue;
                    }
                    if is_on_add && is_on_pano && use_seam {
                        // Preserve both inputs across the overlap. The multiband
                        // stage below needs the untouched base to blend broad
                        // illumination independently from the detail seam.
                        continue;
                    }
                    if is_on_add {
                        let source_color = get_high_quality_interpolated_pixel(&img_to_add, sx, sy);
                        // The overlap is the only place with evidence for an
                        // exposure relation.  Apply the full robust gain
                        // there (and in the seam-band pyramid below), but do
                        // not carry that pairwise correction through the
                        // candidate's non-overlap area.  Doing so turns a
                        // legitimate scan illumination gradient into a
                        // source-sized dark/light rectangle.
                        let exposure_strength = if is_on_pano {
                            1.0
                        } else if let Some((min_x, max_x, min_y, max_y, fade_length)) =
                            exposure_transition
                        {
                            let distance = match orientation {
                                SeamOrientation::Vertical => {
                                    if x < min_x {
                                        (min_x - x) as f32
                                    } else {
                                        x.saturating_sub(max_x) as f32
                                    }
                                }
                                SeamOrientation::Horizontal => {
                                    let y_u32 = y as u32;
                                    if y_u32 < min_y {
                                        (min_y - y_u32) as f32
                                    } else {
                                        y_u32.saturating_sub(max_y) as f32
                                    }
                                }
                            };
                            (1.0 - distance / fade_length).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let color_to_add = apply_exposure_compensation_with_strength(
                            source_color,
                            &exposure,
                            x,
                            y as u32,
                            exposure_strength,
                        );
                        if color_to_add[0].max(color_to_add[1]).max(color_to_add[2]) <= 1e-6 {
                            continue;
                        }
                        let start = x as usize * 3;
                        row_slice[start..start + 3].copy_from_slice(&color_to_add.0);
                        mask_row[x as usize] = 255;
                    }
                }
            });

        if let (true, Some((min_x, max_x, min_y, max_y))) = (use_seam, seam_bounds) {
            blend_panorama_seam_band(SeamBandBlend {
                panorama: &mut panorama,
                panorama_mask: &mut panorama_mask,
                img_to_add_info,
                img_to_add: &img_to_add,
                h_add,
                projection,
                offset_x,
                offset_y,
                orientation,
                seam_coords: &seam_coords,
                new_image_is_dominant_side,
                min_x,
                max_x,
                min_y,
                max_y,
                exposure: &exposure,
            });
        }
    }

    // Virtual tiles are resampled once while being fused at the station and a
    // second time while they are projected onto the scan canvas.  The seam
    // pyramid deliberately keeps the fine bands source-owned, but that second
    // reconstruction still lowers native edge energy.  Restore only bounded
    // luma detail after all tile-level tone work; this cannot change ownership
    // or introduce colour halos at a seam.  The same pass is harmless for a
    // short ordinary panorama and can be tuned or disabled by callers that
    // need the exact pre-sharpened pixels.
    let final_sharpen_amount = std::env::var("RAW_EDITOR_FINAL_PANORAMA_SHARPEN_AMOUNT")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.42)
        .clamp(0.0, 1.25);
    if final_sharpen_amount > 0.0 {
        sharpen_focus_tile_detail(&mut panorama, &panorama_mask, final_sharpen_amount);
    }

    // Virtual tiles have projected corners and narrow ownership holes. Once
    // transparent/black corner samples are rejected, retain the full bounds
    // only when the union is densely covered; otherwise use the largest
    // all-valid rectangle so projective invalid wedges cannot become black
    // output corners.
    let panorama_dimensions = panorama.dimensions();
    let (valid_bounds, valid_coverage) = valid_mask_bounds_and_coverage(&panorama_mask);
    let cropped = if valid_coverage >= 0.995 {
        crop_to_valid_bounds(panorama, &panorama_mask)
    } else {
        crop_to_valid_rectangle(panorama, &panorama_mask)
    };
    if cropped.dimensions() != panorama_dimensions {
        println!(
            "  - Cropped progressive margins: {}x{} -> {}x{} (coverage {:.2}%, bbox {:?})",
            panorama_dimensions.0,
            panorama_dimensions.1,
            cropped.width(),
            cropped.height(),
            valid_coverage * 100.0,
            valid_bounds
        );
    }
    Ok(cropped)
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
    fn into_image(self) -> Rgb32FImage {
        self.image
    }
}

/// Provenance-preserving form of [`layered_virtual_tile_compositor`]. The
/// image-only production entry point above calls this exact function; exposing
/// the masks prevents tests and later pipeline stages from reconstructing
/// ownership from pixel colours.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layered_virtual_tile_compositor_with_ownership<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<LayeredOwnershipRender, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
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
enum TileCompositorFinishing {
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
struct OwnershipCompositorSwitches {
    tone_all_pixels: bool,
    soft_owner_boundary: bool,
    skip_exposure_gain: bool,
    low_frequency_consensus: bool,
    owner_tone_harmonization: bool,
    skip_owner_tone_harmonization: bool,
    final_low_frequency_illumination: bool,
    legacy_final_sharpen_amount: f32,
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
    fn from_env() -> Self {
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
fn ownership_compositor_switches(
    finishing: TileCompositorFinishing,
) -> OwnershipCompositorSwitches {
    match finishing {
        TileCompositorFinishing::LegacyOwnership => OwnershipCompositorSwitches::from_env(),
        TileCompositorFinishing::LayeredVirtualTile => OwnershipCompositorSwitches::default(),
    }
}

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

#[allow(clippy::too_many_arguments)]
fn focus_tile_ownership_stitcher_with_finishing<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    finishing: TileCompositorFinishing,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<LayeredOwnershipRender, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let residual_model = residual_warp::run_model_snapshot();
    if images.is_empty() {
        return Ok(LayeredOwnershipRender {
            image: Rgb32FImage::new(0, 0),
            coverage: CoverageMask::from_gray(GrayImage::new(0, 0)),
            ownership: OwnershipMap::new(0, 0, Vec::new(), Vec::new())?,
            sampling_origin: (0.0, 0.0),
            #[cfg(test)]
            full_resolution_samples: 0,
            #[cfg(test)]
            full_resolution_sample_counts: Vec::new(),
        });
    }
    let (min_x, max_x, min_y, max_y) = output_bounds(images, global_homographies, projection);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return Ok(LayeredOwnershipRender {
            image: Rgb32FImage::new(0, 0),
            coverage: CoverageMask::from_gray(GrayImage::new(0, 0)),
            ownership: OwnershipMap::new(0, 0, Vec::new(), Vec::new())?,
            sampling_origin: (0.0, 0.0),
            #[cfg(test)]
            full_resolution_samples: 0,
            #[cfg(test)]
            full_resolution_sample_counts: Vec::new(),
        });
    }
    let (offset_x, out_width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, out_height) = pixel_aligned_canvas(min_y, max_y);
    // 需求 10.11: an unsupported canvas is rejected before any buffer exists.
    if finishing == TileCompositorFinishing::LayeredVirtualTile
        && let Some(rejection) = super::stack_pipeline::compositor::reject_oversized_canvas(
            u64::from(out_width),
            u64::from(out_height),
        )
    {
        return Err(rejection);
    }
    // Only the retired comparison path reads its diagnostic switches; the
    // default layered path has no environment precondition (任务 13.7).
    let switches = ownership_compositor_switches(finishing);
    let mut panorama = Rgb32FImage::new(out_width, out_height);
    let mut panorama_mask = GrayImage::new(out_width, out_height);
    // An 8-bit interior-distance score is sufficient for deterministic tile
    // ownership and is substantially smaller than a full floating-point score
    // plane for a 40k-pixel-wide scan.
    let mut ownership = vec![0u8; out_width as usize * out_height as usize];
    // Keep the virtual-tile identity separately from the interior-distance
    // quality.  The latter decides which tile supplies detail; the former is
    // needed after composition to solve one global low-frequency tone field
    // instead of accumulating pairwise exposure steps in import order.
    let mut owner_map = GrayImage::new(out_width, out_height);
    // Source-level semantic plane for the exact same write decisions. The u16
    // identifier is not capped at the 8-bit tone-group key above.
    let mut source_owner_ids = vec![NO_OWNER; out_width as usize * out_height as usize];
    #[cfg(test)]
    let full_resolution_samples = AtomicUsize::new(0);
    #[cfg(test)]
    let full_resolution_sample_counts = (0..out_width as usize * out_height as usize)
        .map(|_| AtomicU16::new(0))
        .collect::<Vec<_>>();
    let mut tone_tiles = Vec::with_capacity(images.len());
    let mut tone_tile_ids = Vec::with_capacity(images.len());
    // Station indices and accepted Station_Relations of this run, published by
    // the station solve; without them no tile pair is tone evidence.
    let tone_topology = tone::run_topology_snapshot().unwrap_or_default();
    let row_stride = out_width as usize * 3;
    // A low-frequency consensus is accumulated from every virtual tile while
    // ownership is decided.  The final image keeps one tile's high-frequency
    // detail, while this analysis-resolution field retains the shared paper
    // illumination across overlaps and removes projective tile polygons.
    let (tone_width, tone_height) =
        focus_analysis_dimensions(out_width, out_height, out_width.max(out_height));
    let tone_stride = tone_width as usize;
    let mut tone_sum = vec![[0.0f32; 3]; tone_stride * tone_height as usize];
    let mut tone_weight = vec![0.0f32; tone_stride * tone_height as usize];
    let mut tone_count = vec![0u16; tone_stride * tone_height as usize];
    // A diagnostic/general fallback for material whose canvas tone is not
    // separable from the painted foreground.  The default semantic gate keeps
    // strokes out of the illumination solve; when disabled, the same
    // analysis-resolution consensus is still low-pass filtered before it is
    // applied, so it cannot replace the owned high-frequency detail.
    let tone_all_pixels = switches.tone_all_pixels;
    let soft_owner_boundary = switches.soft_owner_boundary;

    // Pairwise exposure estimates against the progressively built panorama
    // are order dependent.  An optional bounded preview pass solves one
    // global RGB log-gain per virtual tile from all geometric overlaps, then
    // the full-resolution pass below applies that fixed model.  It is
    // opt-in while the cost is being evaluated because a tile preview is an
    // additional decode/render of each virtual station.
    let photometric_models: Option<HashMap<usize, PhotometricModel>> = if images.len() > 1 {
        let mut previews = Vec::with_capacity(images.len());
        for image_info in images {
            let image = load_image(image_info)?;
            let longest = image.width().max(image.height()).max(1);
            let scale = (1536.0 / longest as f64).min(1.0);
            let preview_width = ((image.width() as f64 * scale).round() as u32).max(2);
            let preview_height = ((image.height() as f64 * scale).round() as u32).max(2);
            let preview = resize_rgb(&image, preview_width, preview_height);
            let source_scale = Matrix3::new(
                image.width() as f64 / preview_width as f64,
                0.0,
                0.0,
                0.0,
                image.height() as f64 / preview_height as f64,
                0.0,
                0.0,
                0.0,
                1.0,
            );
            let transform = global_homographies
                .get(&image_info.id)
                .copied()
                .ok_or_else(|| format!("Missing focus tile pose for '{}'", image_info.filename))?
                * source_scale;
            previews.push((preview, transform));
        }
        let calibration = calibrate_overlap_photometry(&previews, &PhotometricOptions::default());
        let reliable_pairs = calibration
            .pairs
            .iter()
            .filter(|pair| pair.reliable)
            .count();
        println!(
            "  - Global virtual-tile photometry: reliable_pairs={}/{} error={:.4}->{:.4}",
            reliable_pairs,
            calibration.pairs.len(),
            calibration.held_out_constant_error,
            calibration.held_out_corrected_error
        );
        // Observation only: unreliable pairs already fall back to the
        // identity model inside `calibrate_overlap_photometry`.
        if reliable_pairs < calibration.pairs.len() {
            degradation::record_run_degradation(
                degradation::TONE_INSUFFICIENT_SAMPLES,
                serde_json::json!({
                    "stage": "global_virtual_tile_photometry",
                    "reliable_pairs": reliable_pairs,
                    "pairs": calibration.pairs.len(),
                }),
            );
        }
        Some(
            images
                .iter()
                .enumerate()
                .map(|(index, image)| (image.id, calibration.models[index].clone()))
                .collect(),
        )
    } else {
        None
    };

    for (index, image_info) in images.iter().enumerate() {
        let _ = app_handle.emit(
            progress_event,
            format!("Composing focus tile {} of {}", index + 1, images.len()),
        );
        println!(
            "  - Ownership composing '{}', tile {} of {}",
            image_info.filename,
            index + 1,
            images.len()
        );
        let homography = global_homographies
            .get(&image_info.id)
            .ok_or_else(|| format!("Missing focus tile pose for '{}'", image_info.filename))?;
        let inverse = homography.try_inverse().ok_or_else(|| {
            format!(
                "Focus tile pose for '{}' is not invertible",
                image_info.filename
            )
        })?;
        let tile = load_image(image_info)?;
        let tile_width = tile.width().max(1) as f64;
        let tile_height = tile.height().max(1) as f64;
        let tile_short = tile_width.min(tile_height).max(1.0);
        // The layered compositor keeps hard Source_RAW ownership exact.  The
        // calibration is still solved unconditionally for the Tone_Harmonizer
        // report, while the retired comparison path is the only one that
        // applies its pixels during this legacy ownership pass.
        let photo_model = (finishing == TileCompositorFinishing::LegacyOwnership)
            .then(|| {
                photometric_models
                    .as_ref()
                    .and_then(|models| models.get(&image_info.id))
            })
            .flatten();
        // Requirement 3.6: the default ownership path may choose a sample, but
        // it must not change that sample's RGB while continuing to name the
        // source as owner. Keep exposure compensation only on the retired
        // comparison path.
        let exposure_enabled = finishing == TileCompositorFinishing::LegacyOwnership
            && photo_model.is_none()
            && !switches.skip_exposure_gain;
        let exposure = if !exposure_enabled {
            ExposureCompensation {
                cell_size: 1,
                grid_width: 1,
                grid_height: 1,
                gains: vec![1.0],
                representative_gain: 1.0,
                channel_gains: vec![[1.0; 3]],
                representative_channel_gain: [1.0; 3],
            }
        } else {
            estimate_overlap_exposure_compensation(ExposureOverlap {
                panorama: &panorama,
                panorama_mask: &panorama_mask,
                candidate: image_info,
                candidate_image: &tile,
                candidate_inverse: &inverse,
                projection,
                offset_x,
                offset_y,
            })
        };
        // Sample the candidate at the bounded tone-analysis resolution. Only
        // canvas-like pixels contribute; brush strokes and seals cannot steer
        // the exposure field. A raised-cosine edge weight favours native tile
        // interiors and makes the consensus independent of tile order.
        for tone_y in 0..tone_height {
            let canvas_y = (tone_y as f64 + 0.5) * out_height as f64 / tone_height.max(1) as f64;
            for tone_x in 0..tone_width {
                let canvas_x = (tone_x as f64 + 0.5) * out_width as f64 / tone_width.max(1) as f64;
                let target = Point3::new(canvas_x - offset_x, canvas_y - offset_y, 1.0);
                let Some(source) = map_target_to_source_with_residual(
                    &inverse,
                    target,
                    image_info,
                    projection,
                    &residual_model,
                ) else {
                    continue;
                };
                if source.x < 0.0
                    || source.y < 0.0
                    || source.x >= tile_width
                    || source.y >= tile_height
                {
                    continue;
                }
                let mut pixel = get_high_quality_interpolated_pixel(&tile, source.x, source.y);
                if let Some(model) = photo_model {
                    let preview_x = source.x * model.preview_width as f64 / tile_width;
                    let preview_y = source.y * model.preview_height as f64 / tile_height;
                    pixel = model.apply(pixel, preview_x, preview_y);
                }
                if !tone_all_pixels
                    && (!focus_stack_pixel_is_canvas_like(&pixel.0)
                        || focus_stack_pixel_is_tone_foreground(&pixel.0))
                {
                    continue;
                }
                let edge_distance = source
                    .x
                    .min(source.y)
                    .min((tile_width - 1.0) - source.x)
                    .min((tile_height - 1.0) - source.y)
                    .max(0.0);
                let edge_t = (edge_distance / (tile_short * 0.20).max(1.0)).clamp(0.0, 1.0);
                let weight = (edge_t * edge_t * (3.0 - 2.0 * edge_t)) as f32;
                if weight <= 0.01 {
                    continue;
                }
                let tone_index = tone_y as usize * tone_stride + tone_x as usize;
                for channel in 0..3 {
                    tone_sum[tone_index][channel] += pixel[channel] * weight;
                }
                tone_weight[tone_index] += weight;
                tone_count[tone_index] = tone_count[tone_index].saturating_add(1);
            }
        }
        // Do not estimate a gain against the progressively accumulated
        // panorama here.  That would make this supposedly order-independent
        // ownership pass inherit a chain of pairwise exposure decisions: a
        // tile selected later could be corrected against a previous owner's
        // already corrected tone, then replaced by a third tile whose gain was
        // measured against a different surface.  This is the source of broad
        // tile-sized tone bands on long scans.  Ownership depends only on the
        // source geometry/edge distance; the owner-wide harmonization below
        // solves the low-frequency tone once, after all owners are final.
        let (left, right, top, bottom) = transformed_image_region(
            image_info, homography, projection, offset_x, offset_y, out_width, out_height,
        )
        .unwrap_or((
            0,
            out_width.saturating_sub(1),
            0,
            out_height.saturating_sub(1),
        ));
        if left > right || top > bottom {
            continue;
        }
        let left = left.min(out_width.saturating_sub(1));
        let right = right.min(out_width.saturating_sub(1));
        let top = top.min(out_height.saturating_sub(1));
        let bottom = bottom.min(out_height.saturating_sub(1));
        // The tone field reads covered tile pixels only and is resampled
        // through the same world-to-tile map the pixels below are drawn with,
        // so neighbouring tiles are compared where they show the same content.
        tone_tile_ids.push(image_info.id);
        tone_tiles.push(tone::world_aligned_tone_tile(
            tone_topology
                .station_of(image_info.id)
                .unwrap_or(image_info.id),
            (index + 1).min(u16::MAX as usize) as u16,
            &tile,
            (left, top, right, bottom),
            |x, y| {
                map_target_to_source_with_residual(
                    &inverse,
                    Point3::new(x - offset_x, y - offset_y, 1.0),
                    image_info,
                    projection,
                    &residual_model,
                )
                .map(|source| (source.x, source.y))
            },
        ));
        let image_width = tile.width() as f64;
        let image_height = tile.height() as f64;
        panorama
            .as_mut()
            .par_chunks_mut(row_stride)
            .zip(ownership.par_chunks_mut(out_width as usize))
            .zip(panorama_mask.as_mut().par_chunks_mut(out_width as usize))
            .zip(owner_map.as_mut().par_chunks_mut(out_width as usize))
            .zip(source_owner_ids.par_chunks_mut(out_width as usize))
            .enumerate()
            .skip(top as usize)
            .take((bottom - top + 1) as usize)
            .for_each(
                |(y, ((((row, quality_row), mask_row), owner_row), source_owner_row))| {
                    for x in left..=right {
                        let target = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
                        let Some(source) = map_target_to_source_with_residual(
                            &inverse,
                            target,
                            image_info,
                            projection,
                            &residual_model,
                        ) else {
                            continue;
                        };
                        if source.x < 0.0
                            || source.y < 0.0
                            || source.x >= image_width
                            || source.y >= image_height
                        {
                            continue;
                        }
                        let edge_distance = source
                            .x
                            .min(source.y)
                            .min((image_width - 1.0) - source.x)
                            .min((image_height - 1.0) - source.y)
                            .max(0.0);
                        let quality = (1.0 + (edge_distance / tile_short * 510.0).round())
                            .clamp(1.0, 255.0) as u8;
                        let quality_slot = &mut quality_row[x as usize];
                        let candidate_owner = (index + 1).min(255) as u8;
                        if !focus_tile_ownership_should_replace(
                            *quality_slot,
                            owner_row[x as usize],
                            quality,
                            candidate_owner,
                        ) {
                            continue;
                        }
                        #[cfg(test)]
                        full_resolution_sample_counts[y as usize * out_width as usize + x as usize]
                            .fetch_add(1, Ordering::Relaxed);
                        #[cfg(test)]
                        full_resolution_samples.fetch_add(1, Ordering::Relaxed);
                        let mut color =
                            get_high_quality_interpolated_pixel(&tile, source.x, source.y);
                        if let Some(model) = photo_model {
                            let preview_x = source.x * model.preview_width as f64 / image_width;
                            let preview_y = source.y * model.preview_height as f64 / image_height;
                            color = model.apply(color, preview_x, preview_y);
                        }
                        // Unfilled projective corners are transparent zeros.  Do
                        // not let them win ownership over a neighbouring tile.
                        if color[0].max(color[1]).max(color[2]) <= 1e-6 {
                            continue;
                        }
                        color = apply_exposure_compensation(color, &exposure, x, y as u32);
                        // Interior distance is a useful prior, but selecting a
                        // whole projective polygon solely by that prior cuts the
                        // paper weave at a visible hard edge. If the candidate is
                        // only marginally better and disagrees strongly with the
                        // already selected sample, keep the existing owner and
                        // let the seam remain in a locally coherent region.
                        if owner_row[x as usize] > 0 {
                            let start = x as usize * 3;
                            let current = &row[start..start + 3];
                            let current_rgb = [current[0], current[1], current[2]];
                            let current_luma =
                                current[0] * 0.299 + current[1] * 0.587 + current[2] * 0.114;
                            let candidate_luma =
                                color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114;
                            let luma_delta = (candidate_luma - current_luma).abs();
                            let quality_ratio =
                                f32::from(quality) / f32::from((*quality_slot).max(1));
                            if luma_delta > 0.075 && quality_ratio < 1.35 {
                                continue;
                            }
                            // Interior distance chooses the sharper source, but a
                            // projective tile boundary should not become a hard
                            // tone edge.  Feather only small, low-contrast canvas
                            // disagreements; saturated or high-contrast detail
                            // remains single-owner so strokes cannot double.
                            if soft_owner_boundary
                                && luma_delta < 0.06
                                && focus_stack_pixel_is_canvas_like(&current_rgb)
                                && focus_stack_pixel_is_canvas_like(&color.0)
                            {
                                let quality_advantage = (f32::from(quality)
                                    - f32::from((*quality_slot).max(1)))
                                .max(0.0);
                                let alpha =
                                    (0.45 + quality_advantage / 255.0 * 0.45).clamp(0.45, 0.90);
                                for channel in 0..3 {
                                    color[channel] = current_rgb[channel] * (1.0 - alpha)
                                        + color[channel] * alpha;
                                }
                            }
                        }
                        let start = x as usize * 3;
                        row[start..start + 3].copy_from_slice(&color.0);
                        *quality_slot = quality;
                        mask_row[x as usize] = 255;
                        owner_row[x as usize] = candidate_owner;
                        source_owner_row[x as usize] = (index + 1).min(u16::MAX as usize) as u16;
                    }
                },
            );
    }

    // Solve a low-frequency consensus from all candidate tiles. The high
    // frequency ownership below remains hard, but its broad paper tone is
    // replaced by the overlap-supported field so tile-shaped vignetting does
    // not survive as a polygonal boundary.
    let tone_mask = GrayImage::from_fn(tone_width, tone_height, |x, y| {
        let index = y as usize * tone_stride + x as usize;
        image::Luma([u8::from(tone_weight[index] > 0.05) * 255])
    });
    let tone_samples = tone_mask
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if switches.low_frequency_consensus && tone_samples >= 128 {
        let tone_reference = Rgb32FImage::from_fn(tone_width, tone_height, |x, y| {
            let index = y as usize * tone_stride + x as usize;
            let weight = tone_weight[index];
            if weight > 0.05 {
                let sum = tone_sum[index];
                Rgb([sum[0] / weight, sum[1] / weight, sum[2] / weight])
            } else {
                Rgb([0.0; 3])
            }
        });
        // The illumination step is source-sized on a scan. A tiny blur only
        // feathers the ownership edge and leaves a broad diagonal tile visible;
        // use a bounded fraction of the analysis canvas so the correction is
        // continuous across one station while preserving brush-scale detail.
        let tone_radius = (tone_width.max(tone_height) as f32 * 0.05)
            .round()
            .clamp(32.0, 192.0) as usize;
        let target_low = masked_box_blur_rgb(&tone_reference, &tone_mask, tone_radius);
        let output_analysis = resize_rgb(&panorama, tone_width, tone_height);
        let output_low = masked_box_blur_rgb(&output_analysis, &tone_mask, tone_radius);
        let target_pixels = target_low.as_raw();
        let output_pixels = output_low.as_raw();
        let mut correction_pixels = vec![0.0f32; tone_width as usize * tone_height as usize * 3];
        for index in 0..tone_mask.as_raw().len() {
            if tone_mask.as_raw()[index] == 0 {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                correction_pixels[start + channel] = (target_pixels[start + channel]
                    - output_pixels[start + channel])
                    .clamp(-0.24, 0.24);
            }
        }
        // A correction measured only where two tiles overlap must also reach
        // the newly exposed part of that owner. Propagate a robust median per
        // owner, then keep the local overlap residual as a small supplement.
        let owner_analysis = image::imageops::resize(
            &owner_map,
            tone_width,
            tone_height,
            image::imageops::FilterType::Nearest,
        );
        let mut owner_deltas: Vec<[Vec<f32>; 3]> = (0..=u8::MAX as usize)
            .map(|_| [Vec::new(), Vec::new(), Vec::new()])
            .collect();
        for index in 0..tone_mask.as_raw().len() {
            if tone_mask.as_raw()[index] == 0 || tone_count[index] < 2 {
                continue;
            }
            let owner = owner_analysis.as_raw()[index] as usize;
            if owner == 0 {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                owner_deltas[owner][channel].push(correction_pixels[start + channel]);
            }
        }
        let mut owner_delta = vec![[0.0f32; 3]; u8::MAX as usize + 1];
        let mut owner_delta_valid = vec![false; u8::MAX as usize + 1];
        for owner in 1..owner_deltas.len() {
            if owner_deltas[owner][0].len() < 8 {
                continue;
            }
            for channel in 0..3 {
                owner_delta[owner][channel] = median_f32(&mut owner_deltas[owner][channel])
                    .unwrap_or(0.0)
                    .clamp(-0.24, 0.24);
            }
            owner_delta_valid[owner] = true;
        }
        for index in 0..tone_mask.as_raw().len() {
            let owner = owner_analysis.as_raw()[index] as usize;
            if !owner_delta_valid.get(owner).copied().unwrap_or(false) {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                correction_pixels[start + channel] =
                    owner_delta[owner][channel] * 0.75 + correction_pixels[start + channel] * 0.25;
            }
        }
        let correction_image = Rgb32FImage::from_raw(tone_width, tone_height, correction_pixels)
            .expect("focus low-frequency correction dimensions must match");
        let correction_low = masked_box_blur_rgb(&correction_image, &tone_mask, tone_radius / 2);
        let tone_weights = tone_mask
            .as_raw()
            .iter()
            .map(|&value| f32::from(value > 0))
            .collect::<Vec<_>>();
        let tone_weight_blur = box_blur_focus_map(
            &tone_weights,
            tone_width,
            tone_height,
            (tone_radius / 2).max(4),
        );
        let x_samples = linear_samples(tone_width, out_width);
        let y_samples = linear_samples(tone_height, out_height);
        let correction_pixels = correction_low.as_raw();
        let tone_weight_ref = &tone_weight_blur;
        let image_mask_ref = panorama_mask.as_raw();
        panorama
            .as_mut()
            .par_chunks_mut(out_width as usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                let y_sample = y_samples[y];
                for x in 0..out_width as usize {
                    if image_mask_ref[y * out_width as usize + x] == 0 {
                        continue;
                    }
                    let x_sample = x_samples[x];
                    let top_left = y_sample.lower * tone_stride + x_sample.lower;
                    let top_right = y_sample.lower * tone_stride + x_sample.upper;
                    let bottom_left = y_sample.upper * tone_stride + x_sample.lower;
                    let bottom_right = y_sample.upper * tone_stride + x_sample.upper;
                    let top_weight = tone_weight_ref[top_left] * (1.0 - x_sample.upper_weight)
                        + tone_weight_ref[top_right] * x_sample.upper_weight;
                    let bottom_weight = tone_weight_ref[bottom_left]
                        * (1.0 - x_sample.upper_weight)
                        + tone_weight_ref[bottom_right] * x_sample.upper_weight;
                    let weight = (top_weight * (1.0 - y_sample.upper_weight)
                        + bottom_weight * y_sample.upper_weight)
                        .clamp(0.0, 1.0);
                    if weight <= 0.001 {
                        continue;
                    }
                    let mut delta = [0.0f32; 3];
                    for channel in 0..3 {
                        let top = correction_pixels[top_left * 3 + channel]
                            * (1.0 - x_sample.upper_weight)
                            + correction_pixels[top_right * 3 + channel] * x_sample.upper_weight;
                        let bottom = correction_pixels[bottom_left * 3 + channel]
                            * (1.0 - x_sample.upper_weight)
                            + correction_pixels[bottom_right * 3 + channel] * x_sample.upper_weight;
                        let local =
                            top * (1.0 - y_sample.upper_weight) + bottom * y_sample.upper_weight;
                        delta[channel] = local * weight;
                    }
                    let start = x * 3;
                    for channel in 0..3 {
                        row[start + channel] =
                            (row[start + channel] + delta[channel]).clamp(0.0, 1.0);
                    }
                }
            });
        println!(
            "  - Low-frequency focus consensus: samples={} radius={}px",
            tone_samples, tone_radius
        );
    }

    // Detail ownership is already final at this point. Correct only the
    // remaining owner-level tone difference using selected virtual-tile
    // regions at once; doing this after the consensus pass removes residual
    // source steps without averaging or replacing brush pixels.
    if finishing == TileCompositorFinishing::LayeredVirtualTile {
        let tone_tile_index = |image_id: usize| tone_tile_ids.iter().position(|&id| id == image_id);
        let tone_relations = tone_topology
            .accepted
            .iter()
            .filter_map(|&(left, right)| Some((tone_tile_index(left)?, tone_tile_index(right)?)))
            .collect::<Vec<_>>();
        let tone_report = tone::harmonize_tone_tiles(
            &mut panorama,
            &source_owner_ids,
            out_width,
            &tone_tiles,
            &panorama_mask,
            &tone_relations,
        );
        println!(
            "  - Tone_Harmonizer: status={:?}, tiles={}, boundary_max_delta_e00={:.3}",
            tone_report.status,
            tone_report.tiles.len(),
            tone_report.boundary_delta_e.max
        );
    }
    let empty_foreground = GrayImage::new(out_width, out_height);
    if switches.owner_tone_harmonization {
        if !switches.skip_owner_tone_harmonization {
            harmonize_focus_background_tone_with_owners(
                &mut panorama,
                &panorama_mask,
                &empty_foreground,
                &owner_map,
            );
        } else {
            println!("  - Skipping final focus owner tone harmonization (diagnostic override)");
        }
    } else {
        println!("  - Skipping focus-tile owner tone harmonization (disabled by default)");
    }
    // A hard owner map keeps brush detail from averaging across a displaced
    // seam. The streaming illumination solve remains an opt-in diagnostic for
    // scenes where a broad lighting field is preferred; the owner-based solve
    // above is the default so it cannot flatten genuine artwork contrast.
    if switches.final_low_frequency_illumination {
        super::mosaic::smooth_streaming_low_frequency_illumination(&mut panorama);
    }
    // Tile warps and the final cubic sampling soften native edges once more.
    // Restore a bounded amount of luma detail after all tone work; the mask
    // keeps the pass out of invalid projective corners and it never averages
    // neighbouring owners.
    // The virtual-tile renderer resamples each source once for the tile and
    // once again while placing that tile on the global canvas.  A restrained
    // luma-only unsharp pass restores the native edge energy lost to those two
    // cubic samples without inventing cross-owner detail.  Keep this below the
    // per-station pass (1.35) so long scans do not acquire halos at tile seams.
    // 任务 13.4: the default layered path resolves to 0.0 here, so
    // `sharpen_focus_tile_detail` is not executed at all; the comparison path
    // keeps its 0.85 second pass.
    let final_sharpen_amount = match finishing {
        TileCompositorFinishing::LayeredVirtualTile => layered_final_sharpen_amount(),
        TileCompositorFinishing::LegacyOwnership => switches.legacy_final_sharpen_amount,
    };
    if final_sharpen_amount > 0.0 {
        sharpen_focus_tile_detail(&mut panorama, &panorama_mask, final_sharpen_amount);
    }

    let panorama_dimensions = panorama.dimensions();
    // The comparison path keeps the complete source union only when its
    // bounding box is genuinely covered and otherwise falls back to the largest
    // valid rectangle.  The default path must not: 需求 10.2 fixes the canvas at
    // the union axis-aligned bounding box of every Coverage_Mask, and 需求 10.3
    // keeps the uncovered projective corners transparent instead of cropping
    // real coverage away to hide them (任务 13.3).
    let (valid_bounds, valid_coverage) = valid_mask_bounds_and_coverage(&panorama_mask);
    let cropped = match finishing {
        TileCompositorFinishing::LayeredVirtualTile => {
            crop_to_valid_bounds(panorama, &panorama_mask)
        }
        TileCompositorFinishing::LegacyOwnership => {
            if valid_coverage >= 0.995 {
                crop_to_valid_bounds(panorama, &panorama_mask)
            } else {
                crop_to_valid_rectangle(panorama, &panorama_mask)
            }
        }
    };
    if cropped.dimensions() != panorama_dimensions {
        println!(
            "  - Cropped ownership margins: {}x{} -> {}x{} (bounds coverage {:.2}%, bbox {:?})",
            panorama_dimensions.0,
            panorama_dimensions.1,
            cropped.width(),
            cropped.height(),
            valid_coverage * 100.0,
            valid_bounds
        );
    }

    // The production default crops only to the covered union bounds, so the
    // same rectangle can be applied losslessly to all semantic planes. The
    // legacy image-only caller discards these planes.
    let (crop_x, crop_y, crop_width, crop_height) = if valid_coverage > 0.0
        && (finishing == TileCompositorFinishing::LayeredVirtualTile || valid_coverage >= 0.995)
    {
        (
            valid_bounds.0,
            valid_bounds.1,
            valid_bounds.2 - valid_bounds.0 + 1,
            valid_bounds.3 - valid_bounds.1 + 1,
        )
    } else {
        (0, 0, cropped.width(), cropped.height())
    };
    let coverage_gray = if crop_x + crop_width <= panorama_mask.width()
        && crop_y + crop_height <= panorama_mask.height()
    {
        image::imageops::crop_imm(&panorama_mask, crop_x, crop_y, crop_width, crop_height)
            .to_image()
    } else {
        GrayImage::new(cropped.width(), cropped.height())
    };
    let mut cropped_owners = vec![NO_OWNER; cropped.width() as usize * cropped.height() as usize];
    #[cfg(test)]
    let mut cropped_sample_counts =
        vec![0u16; cropped.width() as usize * cropped.height() as usize];
    if crop_x + cropped.width() <= out_width && crop_y + cropped.height() <= out_height {
        cropped_owners
            .par_chunks_mut(cropped.width() as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let source_start = (crop_y as usize + y) * out_width as usize + crop_x as usize;
                row.copy_from_slice(&source_owner_ids[source_start..source_start + row.len()]);
            });
        #[cfg(test)]
        cropped_sample_counts
            .iter_mut()
            .enumerate()
            .for_each(|(index, count)| {
                let x = index % cropped.width() as usize;
                let y = index / cropped.width() as usize;
                *count = full_resolution_sample_counts
                    [(crop_y as usize + y) * out_width as usize + crop_x as usize + x]
                    .load(Ordering::Relaxed);
            });
    }
    let legend = images
        .iter()
        .map(|image| PathBuf::from(&image.filename))
        .collect::<Vec<_>>();
    Ok(LayeredOwnershipRender {
        image: cropped,
        coverage: CoverageMask::from_gray(coverage_gray),
        ownership: OwnershipMap::new(crop_width, crop_height, cropped_owners, legend)?,
        sampling_origin: (crop_x as f64 - offset_x, crop_y as f64 - offset_y),
        #[cfg(test)]
        full_resolution_samples: full_resolution_samples.load(Ordering::Relaxed),
        #[cfg(test)]
        full_resolution_sample_counts: cropped_sample_counts,
    })
}

struct SeamBandBlend<'a> {
    panorama: &'a mut Rgb32FImage,
    panorama_mask: &'a mut GrayImage,
    img_to_add_info: &'a ImageInfo,
    img_to_add: &'a Rgb32FImage,
    h_add: &'a Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    orientation: SeamOrientation,
    seam_coords: &'a [i32],
    new_image_is_dominant_side: bool,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
    exposure: &'a ExposureCompensation,
}

fn blend_panorama_seam_band(ctx: SeamBandBlend<'_>) {
    let SeamBandBlend {
        panorama,
        panorama_mask,
        img_to_add_info,
        img_to_add,
        h_add,
        projection,
        offset_x,
        offset_y,
        orientation,
        seam_coords,
        new_image_is_dominant_side,
        min_x,
        max_x,
        min_y,
        max_y,
        exposure,
    } = ctx;
    let (out_width, out_height) = panorama.dimensions();
    if out_width == 0 || out_height == 0 || seam_coords.is_empty() {
        return;
    }

    let overlap_left = min_x.min(out_width - 1);
    let overlap_right = max_x.min(out_width - 1);
    let overlap_top = min_y.min(out_height - 1);
    let overlap_bottom = max_y.min(out_height - 1);
    if overlap_left > overlap_right || overlap_top > overlap_bottom {
        return;
    }

    let seam_value = |index: usize| -> Option<u32> {
        seam_coords
            .get(index)
            .copied()
            .map(|value| value.max(0) as u32)
    };

    match orientation {
        SeamOrientation::Horizontal if seam_coords.len() <= overlap_right as usize => return,
        SeamOrientation::Vertical if seam_coords.len() <= overlap_bottom as usize => return,
        _ => {}
    }
    // Keep the complete overlap so the low-frequency mask can transition across the
    // whole shared field rather than inheriting a tonal step around the detail seam.
    let (patch_left, patch_top, patch_right, patch_bottom) =
        (overlap_left, overlap_top, overlap_right, overlap_bottom);

    if patch_left > patch_right || patch_top > patch_bottom {
        return;
    }
    let patch_width = patch_right - patch_left + 1;
    let patch_height = patch_bottom - patch_top + 1;
    if patch_width < 8 || patch_height < 8 {
        return;
    }

    let Some(h_add_inv) = h_add.try_inverse() else {
        return;
    };
    let patch_pixel_count = patch_width as usize * patch_height as usize;
    let mut base_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut candidate_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut blend_mask = vec![0u8; patch_pixel_count];
    let mut low_frequency_mask = vec![0u8; patch_pixel_count];
    let image_width = img_to_add.width() as f64;
    let image_height = img_to_add.height() as f64;
    let patch_rgb_stride = patch_width as usize * 3;
    let patch_mask_stride = patch_width as usize;
    let panorama_ref: &Rgb32FImage = panorama;
    let panorama_mask_ref: &GrayImage = panorama_mask;

    base_pixels
        .par_chunks_mut(patch_rgb_stride)
        .zip(candidate_pixels.par_chunks_mut(patch_rgb_stride))
        .zip(blend_mask.par_chunks_mut(patch_mask_stride))
        .zip(low_frequency_mask.par_chunks_mut(patch_mask_stride))
        .enumerate()
        .for_each(
            |(local_y, (((base_row, candidate_row), blend_row), low_frequency_row))| {
                let local_y = local_y as u32;
                let global_y = patch_top + local_y;
                for local_x in 0..patch_width {
                    let global_x = patch_left + local_x;
                    let target =
                        Point3::new(global_x as f64 - offset_x, global_y as f64 - offset_y, 1.0);
                    let candidate_source =
                        map_target_to_source(&h_add_inv, target, img_to_add_info, projection);
                    let candidate_valid = candidate_source.as_ref().is_some_and(|source| {
                        source.x >= 0.0
                            && source.x < image_width
                            && source.y >= 0.0
                            && source.y < image_height
                            // A virtual tile is rendered into a rectangular
                            // buffer, but its projective corners are transparent
                            // zero support. Treat those samples as outside the
                            // candidate footprint so they cannot enter either
                            // the low-frequency pyramid or the detail seam.
                            && source_sample_has_support(img_to_add, source.x, source.y)
                    });
                    let candidate_pixel = if let Some(source) = candidate_source {
                        if candidate_valid {
                            apply_exposure_compensation(
                                get_high_quality_interpolated_pixel(img_to_add, source.x, source.y),
                                exposure,
                                global_x,
                                global_y,
                            )
                        } else {
                            Rgb([0.0, 0.0, 0.0])
                        }
                    } else {
                        Rgb([0.0, 0.0, 0.0])
                    };
                    let panorama_valid = panorama_mask_ref.get_pixel(global_x, global_y)[0] > 0;
                    let current_pixel = *panorama_ref.get_pixel(global_x, global_y);
                    let candidate_pixel = if !candidate_valid && panorama_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };
                    let base_pixel = if panorama_valid || !candidate_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };

                    let candidate_signed_distance = if !candidate_valid {
                        f32::NEG_INFINITY
                    } else if !panorama_valid {
                        f32::INFINITY
                    } else {
                        match orientation {
                            SeamOrientation::Horizontal => {
                                let seam_y = seam_value(global_x as usize).unwrap_or(global_y);
                                let distance = global_y as f32 - seam_y as f32;
                                if new_image_is_dominant_side {
                                    distance
                                } else {
                                    -distance
                                }
                            }
                            SeamOrientation::Vertical => {
                                let seam_x = seam_value(global_y as usize).unwrap_or(global_x);
                                let distance = global_x as f32 - seam_x as f32;
                                if new_image_is_dominant_side {
                                    distance
                                } else {
                                    -distance
                                }
                            }
                        }
                    };
                    let detail_alpha = panorama_detail_alpha(candidate_signed_distance);

                    let base_start = local_x as usize * 3;
                    base_row[base_start..base_start + 3].copy_from_slice(&base_pixel.0);
                    candidate_row[base_start..base_start + 3].copy_from_slice(&candidate_pixel.0);
                    blend_row[local_x as usize] = (detail_alpha * 255.0).round() as u8;
                    let low_frequency_alpha = if !candidate_valid {
                        0.0
                    } else if !panorama_valid {
                        1.0
                    } else {
                        match orientation {
                            SeamOrientation::Horizontal => {
                                let span = (overlap_bottom - overlap_top).max(1) as f32;
                                let position = (global_y.saturating_sub(overlap_top)) as f32 / span;
                                if new_image_is_dominant_side {
                                    position
                                } else {
                                    1.0 - position
                                }
                            }
                            SeamOrientation::Vertical => {
                                let span = (overlap_right - overlap_left).max(1) as f32;
                                let position =
                                    (global_x.saturating_sub(overlap_left)) as f32 / span;
                                if new_image_is_dominant_side {
                                    position
                                } else {
                                    1.0 - position
                                }
                            }
                        }
                    };
                    low_frequency_row[local_x as usize] =
                        (low_frequency_alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            },
        );

    let base = Rgb32FImage::from_raw(patch_width, patch_height, base_pixels)
        .expect("seam base patch dimensions must match");
    let candidate = Rgb32FImage::from_raw(patch_width, patch_height, candidate_pixels)
        .expect("seam candidate patch dimensions must match");
    let mask = GrayImage::from_raw(patch_width, patch_height, blend_mask.clone())
        .expect("seam mask dimensions must match");
    let low_frequency_mask = GrayImage::from_raw(patch_width, patch_height, low_frequency_mask)
        .expect("low-frequency seam mask dimensions must match");
    // Keep fine and middle-frequency Laplacian detail on one source side of the
    // path. Averaging those bands across the complete overlap softens strokes,
    // seals, foliage, and other structure whenever registration is not literally
    // pixel-identical. Only broad tonal bands may transition across the complete
    // overlap; visible detail transitions stay local to the optimal seam.
    let blended = multiband_blend(
        base,
        candidate,
        mask,
        Some(low_frequency_mask),
        PANORAMA_BLEND_BANDS,
        // A residual sub-pixel registration error must never be averaged into
        // the visible detail bands. Keep the source ownership hard for the
        // finest and middle frequencies; only the coarse tone pyramid is
        // allowed to feather across the seam.
        true,
    );

    let panorama_rgb_stride = out_width as usize * 3;
    let blended_pixels = blended.as_raw();
    panorama
        .as_mut()
        .par_chunks_mut(panorama_rgb_stride)
        .zip(panorama_mask.as_mut().par_chunks_mut(out_width as usize))
        .enumerate()
        .skip(patch_top as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, (panorama_row, panorama_mask_row))| {
            let local_y = global_y - patch_top as usize;
            let blended_row =
                &blended_pixels[local_y * patch_rgb_stride..(local_y + 1) * patch_rgb_stride];
            let destination_start = patch_left as usize * 3;
            let destination_end = destination_start + patch_rgb_stride;
            panorama_row[destination_start..destination_end].copy_from_slice(blended_row);

            let blend_row =
                &blend_mask[local_y * patch_mask_stride..(local_y + 1) * patch_mask_stride];
            for (local_x, candidate_owns_pixel) in blend_row.iter().copied().enumerate() {
                let global_x = patch_left as usize + local_x;
                if panorama_mask_row[global_x] > 0 || candidate_owns_pixel > 0 {
                    panorama_mask_row[global_x] = 255;
                }
            }
        });
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

struct RenderedFocusLayer {
    image: Rgb32FImage,
    mask: GrayImage,
    foreground_mask: GrayImage,
    // Pixels in a repeated long-edge silhouette may be switched between
    // aligned sources. Other detected foreground remains first-owner-wins.
    relaxed_foreground_mask: GrayImage,
    left: u32,
    top: u32,
}

struct RenderedFocusAnalysisLayer {
    image: Rgb32FImage,
    mask: GrayImage,
    foreground_mask: GrayImage,
    relaxed_foreground_mask: GrayImage,
}

#[derive(Clone, Debug, PartialEq)]
struct FocusColorCorrection {
    gains: [f32; 3],
    offsets: [f32; 3],
    cell_size: u32,
    grid_width: usize,
    grid_height: usize,
    spatial_gains: Option<Vec<[f32; 3]>>,
    spatial_offsets: Option<Vec<[f32; 3]>>,
}

impl FocusColorCorrection {
    const IDENTITY: Self = Self {
        gains: [1.0; 3],
        offsets: [0.0; 3],
        cell_size: 1,
        grid_width: 0,
        grid_height: 0,
        spatial_gains: None,
        spatial_offsets: None,
    };

    fn interpolate_field(&self, field: &[[f32; 3]], x: u32, y: u32) -> [f32; 3] {
        if self.cell_size == 0 || self.grid_width == 0 || self.grid_height == 0 {
            return [0.0; 3];
        }

        let grid_x = x as f64 / self.cell_size as f64;
        let grid_y = y as f64 / self.cell_size as f64;
        let x0 = (grid_x.floor() as usize).min(self.grid_width - 1);
        let y0 = (grid_y.floor() as usize).min(self.grid_height - 1);
        let x1 = (x0 + 1).min(self.grid_width - 1);
        let y1 = (y0 + 1).min(self.grid_height - 1);
        let tx = if x0 == x1 {
            0.0
        } else {
            (grid_x - x0 as f64).clamp(0.0, 1.0) as f32
        };
        let ty = if y0 == y1 {
            0.0
        } else {
            (grid_y - y0 as f64).clamp(0.0, 1.0) as f32
        };
        let value = |gx: usize, gy: usize| field[gy * self.grid_width + gx];
        let top = value(x0, y0);
        let top_right = value(x1, y0);
        let bottom = value(x0, y1);
        let bottom_right = value(x1, y1);
        let mut interpolated = [0.0f32; 3];
        for channel in 0..3 {
            let top_value = top[channel] * (1.0 - tx) + top_right[channel] * tx;
            let bottom_value = bottom[channel] * (1.0 - tx) + bottom_right[channel] * tx;
            interpolated[channel] = top_value * (1.0 - ty) + bottom_value * ty;
        }
        interpolated
    }

    fn gain_at(&self, x: u32, y: u32) -> [f32; 3] {
        self.spatial_gains
            .as_ref()
            .map_or(self.gains, |gains| self.interpolate_field(gains, x, y))
    }

    fn offset_at(&self, x: u32, y: u32) -> [f32; 3] {
        self.spatial_offsets
            .as_ref()
            .map_or(self.offsets, |offsets| {
                self.interpolate_field(offsets, x, y)
            })
    }
}

fn median_f32(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f32::total_cmp);
    Some(values[values.len() / 2])
}

fn estimate_focus_color_correction(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    canvas_only: bool,
) -> FocusColorCorrection {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0
        || layer_height == 0
        || base.dimensions() != base_mask.dimensions()
        || merged_foreground_mask.dimensions() != base.dimensions()
        || candidate.left >= base.width()
        || candidate.top >= base.height()
        || candidate.left + layer_width > base.width()
        || candidate.top + layer_height > base.height()
    {
        return FocusColorCorrection::IDENTITY;
    }

    let sample_area = u64::from(layer_width) * u64::from(layer_height);
    let sample_step = ((sample_area as f64 / FOCUS_COLOR_SAMPLE_BUDGET as f64)
        .sqrt()
        .ceil() as u32)
        .max(1);
    let grid_width = layer_width.div_ceil(FOCUS_COLOR_CELL_SIZE) as usize;
    let grid_height = layer_height.div_ceil(FOCUS_COLOR_CELL_SIZE) as usize;
    let cell_count = grid_width * grid_height;
    let mut luminance_ratios = Vec::new();
    let mut channel_ratios = [Vec::new(), Vec::new(), Vec::new()];
    let mut channel_values = [Vec::new(), Vec::new(), Vec::new()];
    let mut cell_luminance_ratios: Vec<Vec<f32>> = (0..cell_count).map(|_| Vec::new()).collect();
    let mut cell_channel_ratios: Vec<[Vec<f32>; 3]> = (0..cell_count)
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    let mut cell_channel_values: Vec<[Vec<(f32, f32)>; 3]> = (0..cell_count)
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    for y in (0..layer_height).step_by(sample_step as usize) {
        for x in (0..layer_width).step_by(sample_step as usize) {
            if candidate.mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let global_x = candidate.left + x;
            let global_y = candidate.top + y;
            if base_mask.get_pixel(global_x, global_y)[0] == 0 {
                continue;
            }
            let base_is_foreground = merged_foreground_mask.get_pixel(global_x, global_y)[0] > 0;
            let candidate_is_foreground = candidate.foreground_mask.get_pixel(x, y)[0] > 0;
            let relaxed_foreground_overlap = candidate.relaxed_foreground_mask.get_pixel(x, y)[0]
                > 0
                && base_is_foreground
                && candidate_is_foreground;
            // Foreground is normally excluded because a repeated stroke or an
            // occluder is not a safe colour reference. A long-edge consensus is
            // different: it is an explicitly matched, repeated physical edge,
            // so its shared overlap is useful for correcting the rail/holder
            // colour that the paper-only samples cannot observe. Keep ordinary
            // foreground protected even when a focus stack contains it.
            if (base_is_foreground || candidate_is_foreground) && !relaxed_foreground_overlap {
                continue;
            }
            let base_pixel = base.get_pixel(global_x, global_y);
            let candidate_pixel = candidate.image.get_pixel(x, y);
            if canvas_only
                && (!focus_stack_pixel_is_canvas_like(base_pixel.0.as_slice())
                    || !focus_stack_pixel_is_canvas_like(candidate_pixel.0.as_slice()))
            {
                continue;
            }
            let base_luma = luminance(base_pixel);
            let candidate_luma = luminance(candidate_pixel);
            let maximum_luma = if relaxed_foreground_overlap {
                FOCUS_COLOR_MAX_RELAXED_LUMA
            } else {
                0.92
            };
            if !(0.04..=maximum_luma).contains(&base_luma)
                || !(0.04..=maximum_luma).contains(&candidate_luma)
            {
                continue;
            }
            let luma_ratio = base_luma / candidate_luma;
            if !(0.55..=1.8).contains(&luma_ratio) {
                continue;
            }
            luminance_ratios.push(luma_ratio);
            let cell_index = (y / FOCUS_COLOR_CELL_SIZE) as usize * grid_width
                + (x / FOCUS_COLOR_CELL_SIZE) as usize;
            cell_luminance_ratios[cell_index].push(luma_ratio);
            for channel in 0..3 {
                let base_value = base_pixel[channel];
                let candidate_value = candidate_pixel[channel];
                if base_value < FOCUS_COLOR_MIN_CHANNEL_VALUE
                    || candidate_value < FOCUS_COLOR_MIN_CHANNEL_VALUE
                {
                    continue;
                }
                let ratio = base_value / candidate_value;
                if (0.55..=1.8).contains(&ratio) {
                    channel_ratios[channel].push(ratio);
                    channel_values[channel].push((base_value, candidate_value));
                    cell_channel_ratios[cell_index][channel].push(ratio);
                    cell_channel_values[cell_index][channel].push((base_value, candidate_value));
                }
            }
        }
    }

    let luma_gain = if luminance_ratios.len() >= FOCUS_COLOR_MIN_SAMPLES {
        median_f32(&mut luminance_ratios)
            .unwrap_or(1.0)
            .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN)
    } else {
        1.0
    };
    let mut gains = [luma_gain; 3];
    for (channel, ratios) in channel_ratios.iter_mut().enumerate() {
        if ratios.len() >= FOCUS_COLOR_MIN_SAMPLES {
            let channel_gain = median_f32(ratios)
                .unwrap_or(luma_gain)
                .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
            // Most exposure changes are achromatic. Let the measured channel
            // ratio correct a real white-balance shift, but keep it anchored to
            // the luminance ratio so ink/seal colour cannot dominate a sparse
            // overlap.
            gains[channel] = (channel_gain * 0.72 + luma_gain * 0.28)
                .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
        }
    }

    // A ratio-only correction cannot remove a small black-level or flare
    // offset. Estimate it after the bounded gain, using the same robust
    // overlap samples, and keep it deliberately small so paper texture and
    // ink density are not flattened.
    let mut offsets = [0.0f32; 3];
    for channel in 0..3 {
        let mut differences = channel_values[channel]
            .iter()
            .map(|(base_value, candidate_value)| *base_value - *candidate_value * gains[channel])
            .collect::<Vec<_>>();
        if differences.len() >= FOCUS_COLOR_MIN_SAMPLES {
            offsets[channel] = median_f32(&mut differences)
                .unwrap_or(0.0)
                .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
        }
    }

    let mut spatial_gains = vec![gains; cell_count];
    let mut spatial_offsets = vec![offsets; cell_count];
    let mut known_cells = vec![false; cell_count];
    let mut known_cell_count = 0usize;
    for cell_index in 0..cell_count {
        let luma_ratios = &mut cell_luminance_ratios[cell_index];
        if luma_ratios.len() < FOCUS_COLOR_MIN_CELL_SAMPLES {
            continue;
        }
        let local_luma_gain = median_f32(luma_ratios)
            .unwrap_or(luma_gain)
            .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
        let mut local_gains = [local_luma_gain; 3];
        for (channel, ratios) in cell_channel_ratios[cell_index].iter_mut().enumerate() {
            if ratios.len() >= FOCUS_COLOR_MIN_CELL_SAMPLES {
                let channel_gain = median_f32(ratios)
                    .unwrap_or(local_luma_gain)
                    .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
                local_gains[channel] = (channel_gain * 0.72 + local_luma_gain * 0.28)
                    .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
            }
        }
        spatial_gains[cell_index] = local_gains;
        for channel in 0..3 {
            let values = &cell_channel_values[cell_index][channel];
            if values.len() >= FOCUS_COLOR_MIN_CELL_SAMPLES {
                let mut differences = values
                    .iter()
                    .map(|(base_value, candidate_value)| {
                        *base_value - *candidate_value * local_gains[channel]
                    })
                    .collect::<Vec<_>>();
                spatial_offsets[cell_index][channel] = median_f32(&mut differences)
                    .unwrap_or(offsets[channel])
                    .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
            }
        }
        known_cells[cell_index] = true;
        known_cell_count += 1;
    }

    // Smooth only measured cells first. Unknown cells start at the global
    // correction, so an overlap cannot impose its colour cast on a newly added
    // region without further evidence.
    for _ in 0..2 {
        let mut smoothed = spatial_gains.clone();
        let mut smoothed_offsets = spatial_offsets.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let index = grid_y * grid_width + grid_x;
                if !known_cells[index] {
                    continue;
                }
                let mut weight_sum = 0.0f32;
                let mut weighted = [0.0f32; 3];
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
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = if dx == 0 && dy == 0 { 4.0 } else { 1.0 };
                        weight_sum += weight;
                        for channel in 0..3 {
                            weighted[channel] += spatial_gains[neighbor_index][channel] * weight;
                        }
                    }
                }
                if weight_sum > 0.0 {
                    for channel in 0..3 {
                        smoothed[index][channel] = (weighted[channel] / weight_sum)
                            .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
                    }
                }

                let mut offset_weight_sum = 0.0f32;
                let mut weighted_offsets = [0.0f32; 3];
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
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = if dx == 0 && dy == 0 { 4.0 } else { 1.0 };
                        offset_weight_sum += weight;
                        for channel in 0..3 {
                            weighted_offsets[channel] +=
                                spatial_offsets[neighbor_index][channel] * weight;
                        }
                    }
                }
                if offset_weight_sum > 0.0 {
                    for channel in 0..3 {
                        smoothed_offsets[index][channel] = (weighted_offsets[channel]
                            / offset_weight_sum)
                            .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
                    }
                }
            }
        }
        spatial_gains = smoothed;
        spatial_offsets = smoothed_offsets;
    }

    // The measured overlap is often a narrow strip at one side of a shifted
    // frame. Leaving every other cell at the global value creates a visible
    // low-frequency block even though the correction itself is bilinearly
    // interpolated. Propagate only into nearby unknown cells, with confidence
    // determined by distance to measured cells and capped well below 1.0.
    // This keeps the correction continuous at the overlap boundary while
    // preserving the global calibration in genuinely unobserved areas.
    if known_cell_count >= 2 {
        let radius = FOCUS_COLOR_PROPAGATION_RADIUS_CELLS.max(1);
        let mut propagated = spatial_gains.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let index = grid_y * grid_width + grid_x;
                if known_cells[index] {
                    continue;
                }

                let mut weight_sum = 0.0f32;
                let mut weighted = [0.0f32; 3];
                let mut nearest_distance = f32::INFINITY;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let distance = ((dx * dx + dy * dy) as f32).sqrt();
                        if distance > radius as f32 {
                            continue;
                        }
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
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = (1.0 - distance / (radius as f32 + 1.0)).powi(2);
                        weight_sum += weight;
                        nearest_distance = nearest_distance.min(distance);
                        for channel in 0..3 {
                            weighted[channel] += spatial_gains[neighbor_index][channel] * weight;
                        }
                    }
                }
                if weight_sum == 0.0 || !weight_sum.is_finite() {
                    continue;
                }

                let distance_confidence =
                    (1.0 - nearest_distance / (radius as f32 + 1.0)).clamp(0.0, 1.0);
                let confidence = (distance_confidence * FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE)
                    .clamp(0.0, FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE);
                for channel in 0..3 {
                    let local_value = weighted[channel] / weight_sum;
                    propagated[index][channel] = (gains[channel] * (1.0 - confidence)
                        + local_value * confidence)
                        .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
                }
            }
        }
        let mut propagated_offsets = spatial_offsets.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let index = grid_y * grid_width + grid_x;
                if known_cells[index] {
                    continue;
                }

                let mut weight_sum = 0.0f32;
                let mut weighted = [0.0f32; 3];
                let mut nearest_distance = f32::INFINITY;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let distance = ((dx * dx + dy * dy) as f32).sqrt();
                        if distance > radius as f32 {
                            continue;
                        }
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
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = (1.0 - distance / (radius as f32 + 1.0)).powi(2);
                        weight_sum += weight;
                        nearest_distance = nearest_distance.min(distance);
                        for channel in 0..3 {
                            weighted[channel] += spatial_offsets[neighbor_index][channel] * weight;
                        }
                    }
                }
                if weight_sum == 0.0 || !weight_sum.is_finite() {
                    continue;
                }

                let distance_confidence =
                    (1.0 - nearest_distance / (radius as f32 + 1.0)).clamp(0.0, 1.0);
                let confidence = (distance_confidence * FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE)
                    .clamp(0.0, FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE);
                for channel in 0..3 {
                    let local_value = weighted[channel] / weight_sum;
                    propagated_offsets[index][channel] = (offsets[channel] * (1.0 - confidence)
                        + local_value * confidence)
                        .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
                }
            }
        }
        spatial_gains = propagated;
        spatial_offsets = propagated_offsets;
    }

    let has_spatial_variation = known_cell_count >= 2
        && known_cells.iter().enumerate().any(|(index, known)| {
            *known
                && spatial_gains[index]
                    .iter()
                    .zip(gains.iter())
                    .any(|(local_gain, global_gain)| {
                        (local_gain - global_gain).abs() >= FOCUS_COLOR_SPATIAL_VARIATION_THRESHOLD
                    })
        });
    let has_spatial_offset_variation = known_cell_count >= 2
        && known_cells.iter().enumerate().any(|(index, known)| {
            *known
                && spatial_offsets[index].iter().zip(offsets.iter()).any(
                    |(local_offset, global_offset)| {
                        (local_offset - global_offset).abs() >= FOCUS_COLOR_SPATIAL_OFFSET_THRESHOLD
                    },
                )
        });
    FocusColorCorrection {
        gains,
        offsets,
        cell_size: FOCUS_COLOR_CELL_SIZE,
        grid_width,
        grid_height,
        spatial_gains: has_spatial_variation.then_some(spatial_gains),
        spatial_offsets: has_spatial_offset_variation.then_some(spatial_offsets),
    }
}

fn apply_focus_color_correction(
    image: &mut Rgb32FImage,
    correction: &FocusColorCorrection,
    protected_mask: &GrayImage,
    canvas_only: bool,
) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return;
    }
    let has_protected_mask = protected_mask.dimensions() == (width, height);

    // Apply the exposure match to a low-frequency copy and add only that
    // correction back to the source. Applying gain/offset directly to every
    // source pixel compresses brush-stroke and weave contrast, which is not
    // acceptable for a high-resolution cultural-relic scan.
    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(width, height, width.max(height));
    if (analysis_width, analysis_height) == (width, height) {
        let width = width as usize;
        image
            .as_mut()
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(index, pixel)| {
                let x = (index % width) as u32;
                let y = (index / width) as u32;
                if has_protected_mask && protected_mask.get_pixel(x, y)[0] > 0 {
                    return;
                }
                if canvas_only && !focus_stack_pixel_is_canvas_like(pixel) {
                    return;
                }
                let gains = correction.gain_at(x, y);
                let offsets = correction.offset_at(x, y);
                for (channel, value) in pixel.iter_mut().enumerate() {
                    *value = (*value * gains[channel] + offsets[channel]).clamp(0.0, 1.0);
                }
            });
        return;
    }

    let analysis = resize_rgb(image, analysis_width, analysis_height);
    let analysis_protected = if has_protected_mask {
        resize_binary_mask(protected_mask, analysis_width, analysis_height)
    } else {
        GrayImage::new(analysis_width, analysis_height)
    };
    let mut low_frequency_delta = Rgb32FImage::new(analysis_width, analysis_height);
    let analysis_width_usize = analysis_width as usize;
    let source_width = width as f64;
    let source_height = height as f64;
    low_frequency_delta
        .as_mut()
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(index, delta)| {
            if analysis_protected.as_raw()[index] > 0
                || (canvas_only
                    && !focus_stack_pixel_is_canvas_like(
                        &analysis.as_raw()[index * 3..index * 3 + 3],
                    ))
            {
                return;
            }
            let x = (index % analysis_width_usize) as f64;
            let y = (index / analysis_width_usize) as f64;
            let source_x = ((x + 0.5) * source_width / analysis_width as f64)
                .floor()
                .min(source_width - 1.0) as u32;
            let source_y = ((y + 0.5) * source_height / analysis_height as f64)
                .floor()
                .min(source_height - 1.0) as u32;
            let gains = correction.gain_at(source_x, source_y);
            let offsets = correction.offset_at(source_x, source_y);
            let source = &analysis.as_raw()[index * 3..index * 3 + 3];
            for channel in 0..3 {
                let corrected =
                    (source[channel] * gains[channel] + offsets[channel]).clamp(0.0, 1.0);
                delta[channel] = corrected - source[channel];
            }
        });
    // The correction field is already low-frequency. Sample it directly with
    // bilinear interpolation instead of allocating and resizing another full
    // source-sized RGB image for every frame in a long stack.
    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize * 3;
    let low_frequency_delta = low_frequency_delta.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            let top_row = y_sample.lower * analysis_stride;
            let bottom_row = y_sample.upper * analysis_stride;
            let y_weight = y_sample.upper_weight;
            for (x, x_sample) in x_samples.iter().copied().enumerate() {
                let index = y * width as usize + x;
                if has_protected_mask && protected_mask.as_raw()[index] > 0 {
                    continue;
                }
                let top_left = &low_frequency_delta
                    [top_row + x_sample.lower * 3..top_row + x_sample.lower * 3 + 3];
                let top_right = &low_frequency_delta
                    [top_row + x_sample.upper * 3..top_row + x_sample.upper * 3 + 3];
                let bottom_left = &low_frequency_delta
                    [bottom_row + x_sample.lower * 3..bottom_row + x_sample.lower * 3 + 3];
                let bottom_right = &low_frequency_delta
                    [bottom_row + x_sample.upper * 3..bottom_row + x_sample.upper * 3 + 3];
                let start = x * 3;
                if canvas_only && !focus_stack_pixel_is_canvas_like(&row[start..start + 3]) {
                    continue;
                }
                let x_weight = x_sample.upper_weight;
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_weight) + top_right[channel] * x_weight;
                    let bottom =
                        bottom_left[channel] * (1.0 - x_weight) + bottom_right[channel] * x_weight;
                    let delta = top * (1.0 - y_weight) + bottom * y_weight;
                    row[start + channel] = (row[start + channel] + delta).clamp(0.0, 1.0);
                }
            }
        });
}

fn source_is_focus_foreground(image: &ImageInfo, source: Point2<f64>) -> bool {
    let Some((minimum_y, maximum_y)) = image.foreground_range else {
        return false;
    };
    let source_width = image.width().max(1) as f64;
    let source_height = image.height().max(1) as f64;
    if source.x < 0.0 || source.y < 0.0 || source.x >= source_width || source.y >= source_height {
        return false;
    }
    let source_y = source.y / source_height;
    if !(minimum_y..=maximum_y).contains(&source_y) {
        return false;
    }
    let alignment_scale = image.scale_factor.max(1.0);
    let alignment_x = (source.x / alignment_scale).round() as i32;
    let alignment_y = (source.y / alignment_scale).round() as i32;
    let (alignment_width, alignment_height) = image.alignment_image.dimensions();
    if alignment_x < 0
        || alignment_y < 0
        || alignment_x >= alignment_width as i32
        || alignment_y >= alignment_height as i32
    {
        return false;
    }
    image.foreground_mask.as_ref().map_or_else(
        || {
            image
                .alignment_image
                .get_pixel(alignment_x as u32, alignment_y as u32)[0]
                >= FOCUS_FOREGROUND_LUMA_THRESHOLD
        },
        |mask| mask.get_pixel(alignment_x as u32, alignment_y as u32)[0] > 0,
    )
}

fn focus_transformed_image_region(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    if out_width == 0 || out_height == 0 {
        return None;
    }
    let Some((mut min_x, mut max_x, mut min_y, mut max_y)) =
        transformed_source_bounds(image, homography, projection, 0.0, 1.0)
    else {
        return None;
    };
    if let Some(focus_warp) = focus_warp {
        for band in &focus_warp.bands {
            let Some(band_homography) = band.homographies.get(&image.id) else {
                continue;
            };
            let Some(&(minimum_source_y, maximum_source_y)) = band.source_ranges.get(&image.id)
            else {
                continue;
            };
            let (minimum_source_x, maximum_source_x) = band
                .source_x_ranges
                .get(&image.id)
                .copied()
                .unwrap_or((0.0, 1.0));
            let Some((band_min_x, band_max_x, band_min_y, band_max_y)) =
                transformed_source_bounds_xy(
                    image,
                    band_homography,
                    projection,
                    minimum_source_x,
                    maximum_source_x,
                    minimum_source_y,
                    maximum_source_y,
                )
            else {
                continue;
            };
            min_x = min_x.min(band_min_x);
            max_x = max_x.max(band_max_x);
            min_y = min_y.min(band_min_y);
            max_y = max_y.max(band_max_y);
        }
    }
    let min_x = min_x + offset_x;
    let max_x = max_x + offset_x;
    let min_y = min_y + offset_y;
    let max_y = max_y + offset_y;
    let left = min_x.floor().max(0.0).min((out_width - 1) as f64) as u32;
    let right = max_x.ceil().max(0.0).min((out_width - 1) as f64) as u32;
    let top = min_y.floor().max(0.0).min((out_height - 1) as f64) as u32;
    let bottom = max_y.ceil().max(0.0).min((out_height - 1) as f64) as u32;
    (left <= right && top <= bottom).then_some((left, right, top, bottom))
}

#[allow(clippy::too_many_arguments)]
fn render_focus_layer(
    image: &ImageInfo,
    source_image: &Rgb32FImage,
    homography: &Matrix3<f64>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
) -> RenderedFocusLayer {
    let global_inverse = homography.try_inverse().unwrap_or_else(Matrix3::identity);
    let Some((left, right, top, bottom)) = focus_transformed_image_region(
        image, homography, projection, focus_warp, offset_x, offset_y, out_width, out_height,
    ) else {
        return RenderedFocusLayer {
            image: Rgb32FImage::new(0, 0),
            mask: GrayImage::new(0, 0),
            foreground_mask: GrayImage::new(0, 0),
            relaxed_foreground_mask: GrayImage::new(0, 0),
            left: 0,
            top: 0,
        };
    };
    let band_inverses = focus_warp
        .into_iter()
        .flat_map(|warp| warp.bands.iter())
        .filter_map(|band| {
            let homography = band.homographies.get(&image.id)?;
            let &(minimum_source_y, maximum_source_y) = band.source_ranges.get(&image.id)?;
            let (minimum_source_x, maximum_source_x) = band
                .source_x_ranges
                .get(&image.id)
                .copied()
                .unwrap_or((0.0, 1.0));
            let inverse = homography.try_inverse()?;
            let correction_inverse = (*homography * global_inverse).try_inverse()?;
            let axis = if maximum_source_x - minimum_source_x < maximum_source_y - minimum_source_y
            {
                0u8 // vertical edge: narrow source-x band
            } else {
                1u8 // horizontal/depth band: narrow source-y band
            };
            Some((
                minimum_source_x,
                maximum_source_x,
                minimum_source_y,
                maximum_source_y,
                inverse,
                correction_inverse,
                axis,
                band.relax_foreground_seam,
                band.foreground_only,
            ))
        })
        .collect::<Vec<_>>();
    let layer_width = right - left + 1;
    let layer_height = bottom - top + 1;
    let mut pixels = vec![0.0f32; layer_width as usize * layer_height as usize * 3];
    let mut mask = vec![0u8; layer_width as usize * layer_height as usize];
    let mut foreground_mask = vec![0u8; layer_width as usize * layer_height as usize];
    let mut relaxed_foreground_mask = vec![0u8; layer_width as usize * layer_height as usize];

    pixels
        .par_chunks_mut(layer_width as usize * 3)
        .zip(mask.par_chunks_mut(layer_width as usize))
        .zip(foreground_mask.par_chunks_mut(layer_width as usize))
        .zip(relaxed_foreground_mask.par_chunks_mut(layer_width as usize))
        .enumerate()
        .for_each(
            |(y, (((row, mask_row), foreground_mask_row), relaxed_row))| {
                let global_y = top + y as u32;
                for local_x in 0..layer_width {
                    let global_x = left + local_x;
                    let target =
                        Point3::new(global_x as f64 - offset_x, global_y as f64 - offset_y, 1.0);
                    let global_source =
                        map_target_to_source(&global_inverse, target, image, projection).filter(
                            |candidate| {
                                candidate.x >= 0.0
                                    && candidate.y >= 0.0
                                    && candidate.x < source_image.width() as f64
                                    && candidate.y < source_image.height() as f64
                            },
                        );
                    // A local model moves a depth layer relative to the paper. Do
                    // not use the global inverse to decide whether that moved pixel
                    // belongs to the band: doing so drops the local model exactly
                    // along the displaced silhouette and creates a step back to the
                    // paper transform. Instead, validate each local inverse in its
                    // own source domain, then let the narrowest active band win.
                    // This is content-agnostic; when no local bands exist the global
                    // inverse remains the only mapping.
                    let mut local_candidates = Vec::new();
                    for &(
                        minimum_source_x,
                        maximum_source_x,
                        minimum_source_y,
                        maximum_source_y,
                        ref band_inverse,
                        ref correction_inverse,
                        axis,
                        relax_foreground_seam,
                        foreground_only,
                    ) in &band_inverses
                    {
                        let Some(candidate) =
                            map_target_to_source(band_inverse, target, image, projection)
                        else {
                            continue;
                        };
                        if candidate.x < 0.0
                            || candidate.y < 0.0
                            || candidate.x >= source_image.width() as f64
                            || candidate.y >= source_image.height() as f64
                        {
                            continue;
                        }
                        if foreground_only && !source_is_focus_foreground(image, candidate) {
                            continue;
                        }
                        let normalized_x = candidate.x / image.width().max(1) as f64;
                        let normalized_y = candidate.y / image.height().max(1) as f64;
                        if normalized_x < minimum_source_x - 0.02
                            || normalized_x > maximum_source_x + 0.02
                            || normalized_y < minimum_source_y - 0.02
                            || normalized_y > maximum_source_y + 0.02
                        {
                            continue;
                        }
                        let x_span = (maximum_source_x - minimum_source_x).max(1e-6);
                        let y_span = (maximum_source_y - minimum_source_y).max(1e-6);
                        let (span, normalized_axis, center_axis) = if x_span < y_span {
                            (
                                x_span,
                                normalized_x,
                                (minimum_source_x + maximum_source_x) * 0.5,
                            )
                        } else {
                            (
                                y_span,
                                normalized_y,
                                (minimum_source_y + maximum_source_y) * 0.5,
                            )
                        };
                        local_candidates.push((
                            span,
                            normalized_axis,
                            center_axis,
                            axis,
                            candidate,
                            *correction_inverse,
                            relax_foreground_seam,
                            foreground_only,
                        ));
                    }
                    let select_local_candidate = |axis: u8| {
                        let narrowest_span = local_candidates
                            .iter()
                            .filter(|candidate| candidate.3 == axis)
                            .map(|candidate| candidate.0)
                            .fold(f64::INFINITY, f64::min);
                        if !narrowest_span.is_finite() {
                            return None;
                        }
                        let maximum_span = narrowest_span * 1.25;
                        // Several generic/edge bands can overlap in source space.
                        // Keep one candidate per axis: prefer the narrowest valid
                        // band, then the band whose centre is closest to this
                        // sample. Orthogonal bands are composed below rather than
                        // allowing one axis to discard the other.
                        local_candidates
                            .iter()
                            .filter(|candidate| candidate.3 == axis && candidate.0 <= maximum_span)
                            .min_by(|left, right| {
                                left.0.total_cmp(&right.0).then_with(|| {
                                    (left.1 - left.2)
                                        .abs()
                                        .total_cmp(&(right.1 - right.2).abs())
                                })
                            })
                    };
                    let selected_vertical = select_local_candidate(0);
                    let selected_horizontal = select_local_candidate(1);
                    let local_strength_for =
                        |candidate: &(f64, f64, f64, u8, Point2<f64>, Matrix3<f64>, bool, bool)| {
                            (1.0 - (candidate.1 - candidate.2).abs() / (candidate.0 * 0.5))
                                .clamp(0.0, 1.0)
                        };
                    let vertical_strength = selected_vertical.map(local_strength_for);
                    let horizontal_strength = selected_horizontal.map(local_strength_for);
                    let mut local_source = None;
                    let mut local_strength = 0.0f64;
                    let mut relax_foreground_seam = false;
                    match (selected_vertical, selected_horizontal) {
                        (Some(vertical), Some(horizontal)) => {
                            // The stored correction is a world-space correction
                            // on top of the global pose. Apply inverse horizontal
                            // then inverse vertical correction before the global
                            // inverse; this preserves both sides of a detected
                            // corner/endpoint without introducing an averaged pose.
                            let corrected_target = horizontal.5 * target;
                            let corrected_target = vertical.5 * corrected_target;
                            local_source = map_target_to_source(
                                &global_inverse,
                                corrected_target,
                                image,
                                projection,
                            )
                            .filter(|candidate| {
                                candidate.x >= 0.0
                                    && candidate.y >= 0.0
                                    && candidate.x < source_image.width() as f64
                                    && candidate.y < source_image.height() as f64
                            });
                            local_strength = vertical_strength
                                .unwrap_or(0.0)
                                .min(horizontal_strength.unwrap_or(0.0));
                            relax_foreground_seam = vertical.6 || horizontal.6;
                        }
                        (Some(vertical), None) => {
                            local_source = Some(vertical.4);
                            local_strength = vertical_strength.unwrap_or(0.0);
                            relax_foreground_seam = vertical.6;
                        }
                        (None, Some(horizontal)) => {
                            local_source = Some(horizontal.4);
                            local_strength = horizontal_strength.unwrap_or(0.0);
                            relax_foreground_seam = horizontal.6;
                        }
                        (None, None) => {}
                    }
                    if local_strength <= 0.0 {
                        local_source = None;
                        relax_foreground_seam = false;
                    }
                    // Blend each selected regional correction into the global
                    // mapping at the edges of its source band. A hard switch is
                    // itself visible as a seam, especially when the corrected
                    // object is a rail or paper boundary. Each axis' influence
                    // falls continuously to zero at its band boundary; when both
                    // axes are active their composed correction uses the smaller
                    // overlap weight. If the global inverse is unavailable, the
                    // local mapping remains a valid fallback.
                    let source = match (global_source, local_source) {
                        (Some(global), Some(local)) => {
                            let local_weight = local_strength.clamp(0.0, 1.0);
                            Some(Point2::new(
                                global.x * (1.0 - local_weight) + local.x * local_weight,
                                global.y * (1.0 - local_weight) + local.y * local_weight,
                            ))
                        }
                        (Some(global), None) => Some(global),
                        (None, Some(local)) => Some(local),
                        (None, None) => None,
                    };
                    let Some(source) = source else {
                        continue;
                    };
                    if source.x < 1.0
                        || source.y < 1.0
                        || source.x >= source_image.width() as f64 - 2.0
                        || source.y >= source_image.height() as f64 - 2.0
                    {
                        continue;
                    }
                    let pixel =
                        get_high_quality_interpolated_pixel(source_image, source.x, source.y);
                    // Requirement 3.6 forbids replacing an owned sample with a
                    // neighbouring "corrected" paper sample: the owner must
                    // identify the photograph and coordinate that supplied RGB.
                    let start = local_x as usize * 3;
                    row[start..start + 3].copy_from_slice(&pixel.0);
                    mask_row[local_x as usize] = 255;
                    // This mask controls focus ownership and duplicate suppression;
                    // it must stay geometric. Colour classification belongs only
                    // to the final canvas-tone protection below. Marking every
                    // black stroke or red seal here would make the first soft
                    // sample win forever and produce the reported ghosting.
                    if source_is_focus_foreground(image, source) {
                        foreground_mask_row[local_x as usize] = 255;
                        if relax_foreground_seam {
                            relaxed_row[local_x as usize] = 255;
                        }
                    }
                }
            },
        );

    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(layer_width, layer_height, pixels)
            .expect("focus layer buffer dimensions must match"),
        mask: GrayImage::from_raw(layer_width, layer_height, mask)
            .expect("focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(layer_width, layer_height, foreground_mask)
            .expect("focus foreground mask buffer dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(
            layer_width,
            layer_height,
            relaxed_foreground_mask,
        )
        .expect("focus relaxed foreground mask dimensions must match"),
        left,
        top,
    }
}

fn focus_analysis_dimensions(width: u32, height: u32, source_longest_dimension: u32) -> (u32, u32) {
    let longest_side = width.max(height).max(1);
    // Keep the focus-analysis sampling density tied to an individual source,
    // not to the total mosaic length. Otherwise a long shifted stack compresses
    // each 9504px frame to a few hundred analysis pixels and erases the very
    // sharpness differences the stacker is supposed to select.
    let source_scale = FOCUS_ANALYSIS_MAX_DIMENSION as f64 / source_longest_dimension.max(1) as f64;
    let mut scale = source_scale.min(1.0);
    let scaled_pixels = width as f64 * height as f64 * scale * scale;
    if scaled_pixels > FOCUS_ANALYSIS_MAX_PIXELS as f64 {
        scale *= (FOCUS_ANALYSIS_MAX_PIXELS as f64 / scaled_pixels).sqrt();
    }
    if scale >= 1.0 && longest_side <= FOCUS_ANALYSIS_MAX_DIMENSION {
        return (width.max(1), height.max(1));
    }
    (
        (width as f64 * scale).round().max(1.0) as u32,
        (height as f64 * scale).round().max(1.0) as u32,
    )
}

fn box_blur_focus_map(source: &[f32], width: u32, height: u32, radius: usize) -> Vec<f32> {
    if radius == 0 || width == 0 || height == 0 {
        return source.to_vec();
    }
    let width = width as usize;
    let height = height as usize;
    let window_size = radius * 2 + 1;
    let divisor = window_size as f32;
    let mut horizontal = vec![0.0f32; source.len()];
    horizontal
        .par_chunks_mut(width)
        .zip(source.par_chunks(width))
        .for_each(|(output_row, source_row)| {
            let mut sum = 0.0f32;
            for offset in 0..window_size {
                sum += source_row[offset.saturating_sub(radius).min(width - 1)];
            }
            output_row[0] = sum / divisor;
            for (x, output) in output_row.iter_mut().enumerate().skip(1) {
                let add_x = (x + radius).min(width - 1);
                let remove_x = x.saturating_sub(radius + 1);
                sum += source_row[add_x] - source_row[remove_x];
                *output = sum / divisor;
            }
        });

    let mut output = vec![0.0f32; source.len()];
    for x in 0..width {
        let mut sum = 0.0f32;
        for offset in 0..window_size {
            let y = offset.saturating_sub(radius).min(height - 1);
            sum += horizontal[y * width + x];
        }
        output[x] = sum / divisor;
        for y in 1..height {
            let add_y = (y + radius).min(height - 1);
            let remove_y = y.saturating_sub(radius + 1);
            sum += horizontal[add_y * width + x] - horizontal[remove_y * width + x];
            output[y * width + x] = sum / divisor;
        }
    }
    output
}

fn focus_score_map(image: &Rgb32FImage, mask: &GrayImage) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let pixel_count = width as usize * height as usize;
    let mut luminance_map = vec![0.0f32; pixel_count];
    luminance_map
        .par_iter_mut()
        .zip(image.as_raw().par_chunks(3))
        .for_each(|(output, pixel)| {
            *output = pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114;
        });

    let mut raw_score = vec![0.0f32; pixel_count];
    if width >= 3 && height >= 3 {
        raw_score
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                if y == 0 || y + 1 >= height as usize {
                    return;
                }
                for (x, output) in row.iter_mut().enumerate().take(width as usize - 1).skip(1) {
                    let index = y * width as usize + x;
                    if mask.as_raw()[index] == 0
                        || mask.as_raw()[index - 1] == 0
                        || mask.as_raw()[index + 1] == 0
                        || mask.as_raw()[index - width as usize] == 0
                        || mask.as_raw()[index + width as usize] == 0
                    {
                        continue;
                    }
                    let laplacian = (4.0 * luminance_map[index]
                        - luminance_map[index - 1]
                        - luminance_map[index + 1]
                        - luminance_map[index - width as usize]
                        - luminance_map[index + width as usize])
                        .abs();
                    // A Laplacian alone under-scores broad, low-contrast
                    // brush strokes and woven paper texture.  Add a bounded
                    // Tenengrad term so a genuinely focused layer can win on
                    // those structures without making isolated sensor noise
                    // dominate the ownership map.
                    let gradient_x = luminance_map[index + 1] - luminance_map[index - 1];
                    let gradient_y = luminance_map[index + width as usize]
                        - luminance_map[index - width as usize];
                    let gradient = (gradient_x * gradient_x + gradient_y * gradient_y).sqrt();
                    *output = laplacian + gradient * 0.5;
                }
            });
    }

    // Focus ownership is decided on a bounded analysis canvas. Keep the score
    // window local enough that one clear stroke cannot claim an adjacent
    // character; the final pixels still come from the full-resolution aligned
    // layer.
    let radius = ((width.max(height) as f32 / FOCUS_SCORE_BLUR_RADIUS_DIVISOR).round() as usize)
        .clamp(1, FOCUS_SCORE_MAX_BLUR_RADIUS);
    box_blur_focus_map(&raw_score, width, height, radius)
}

fn focus_decision_mask(
    base_focus: &[f32],
    candidate_focus: &[f32],
    base_mask: &GrayImage,
    candidate_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = base_mask.dimensions();
    let pixel_count = width as usize * height as usize;
    let mut fine_advantage = vec![0.0f32; pixel_count];
    fine_advantage
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, output)| {
            let base_valid = base_mask.as_raw()[index] > 0;
            let candidate_valid = candidate_mask.as_raw()[index] > 0;
            *output = match (base_valid, candidate_valid) {
                (false, true) => 1.0,
                (true, false) => -1.0,
                (true, true) => {
                    let base = base_focus[index];
                    let candidate = candidate_focus[index];
                    (candidate - base) / (candidate + base + 1e-6)
                }
                (false, false) => 0.0,
            };
        });

    // The old radius was effectively ~100 source pixels for a 9504px frame.
    // That let a clear stroke in one region claim a neighbouring blurred
    // character. Keep the decision coherent, but make the neighbourhood local
    // enough that each character can select its own sharp source.
    let coherence_radius = ((width.max(height) as f32 / FOCUS_DECISION_COHERENCE_RADIUS_DIVISOR)
        .round() as usize)
        .clamp(2, FOCUS_DECISION_MAX_COHERENCE_RADIUS);
    let coarse_advantage = box_blur_focus_map(&fine_advantage, width, height, coherence_radius);

    let sample_step = (pixel_count / 4096).max(1);
    let mut magnitude_samples = (0..pixel_count)
        .step_by(sample_step)
        .filter_map(|index| {
            (base_mask.as_raw()[index] > 0 || candidate_mask.as_raw()[index] > 0)
                .then_some(base_focus[index].max(candidate_focus[index]))
        })
        .collect::<Vec<_>>();
    magnitude_samples.sort_unstable_by(f32::total_cmp);
    let confidence_floor = magnitude_samples
        .get(((magnitude_samples.len().saturating_sub(1)) as f32 * 0.9) as usize)
        .copied()
        .unwrap_or(0.0)
        * 0.04;

    // A defocused subject spills contrast beyond its true silhouette. That halo can
    // look locally sharper than the clean background in the focused layer, leaving
    // detached slivers around depth discontinuities even with hard pixel selection.
    // Let decisive, valid structure claim a small adjacent ambiguous band so the
    // focused silhouette also supplies the clean pixels immediately around it.
    let mut strong_base = vec![0.0f32; pixel_count];
    let mut strong_candidate = vec![0.0f32; pixel_count];
    strong_base
        .par_iter_mut()
        .zip(strong_candidate.par_iter_mut())
        .enumerate()
        .for_each(|(index, (base_claim, candidate_claim))| {
            let both_valid = base_mask.as_raw()[index] > 0 && candidate_mask.as_raw()[index] > 0;
            let confident = base_focus[index].max(candidate_focus[index]) > confidence_floor;
            if !both_valid || !confident {
                return;
            }
            let advantage = fine_advantage[index];
            if advantage < -FOCUS_DECISIVE_ADVANTAGE {
                *base_claim = 1.0;
            } else if advantage > FOCUS_DECISIVE_ADVANTAGE {
                *candidate_claim = 1.0;
            }
        });
    let protection_radius = ((width.max(height) as f32 / 2048.0)
        * (FOCUS_EDGE_PROTECTION_AT_1024 * FOCUS_EDGE_PROTECTION_SCALE))
        .round()
        .clamp(2.0, 6.0) as usize;
    let base_claim = box_blur_focus_map(&strong_base, width, height, protection_radius);
    let candidate_claim = box_blur_focus_map(&strong_candidate, width, height, protection_radius);

    GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        let base_valid = base_mask.as_raw()[index] > 0;
        let candidate_valid = candidate_mask.as_raw()[index] > 0;
        let confident = base_focus[index].max(candidate_focus[index]) > confidence_floor;
        // Always use the coherent neighborhood decision for ownership. A single
        // high-contrast stroke can be displaced by a couple of pixels between
        // captures; using its raw per-pixel score would then alternate sources
        // across the stroke and recreate a double contour.
        let advantage = coarse_advantage[index];
        let nearby_base_claim = base_claim[index];
        let nearby_candidate_claim = candidate_claim[index];
        let structural_winner = if nearby_candidate_claim > FOCUS_EDGE_CLAIM_THRESHOLD
            && nearby_candidate_claim > nearby_base_claim
        {
            Some(true)
        } else if nearby_base_claim > FOCUS_EDGE_CLAIM_THRESHOLD
            && nearby_base_claim > nearby_candidate_claim
        {
            Some(false)
        } else {
            None
        };
        let candidate_wins = candidate_valid
            && (!base_valid
                || structural_winner.unwrap_or(confident && advantage > FOCUS_CONFIDENCE_MARGIN));
        image::Luma([if candidate_wins { 255 } else { 0 }])
    })
}

fn suppress_focus_canvas_switches(
    decision_mask: &mut GrayImage,
    base: &Rgb32FImage,
    candidate: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate_mask: &GrayImage,
) {
    if decision_mask.dimensions() != base.dimensions()
        || base.dimensions() != candidate.dimensions()
        || base_mask.dimensions() != base.dimensions()
        || candidate_mask.dimensions() != base.dimensions()
    {
        return;
    }

    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(base.as_raw().par_chunks(3))
        .zip(candidate.as_raw().par_chunks(3))
        .zip(base_mask.as_raw().par_iter())
        .zip(candidate_mask.as_raw().par_iter())
        .for_each(
            |((((decision, base_pixel), candidate_pixel), base_valid), candidate_valid)| {
                // The canvas is one planar surface. Sharpness fluctuations in its
                // weave must not make the focus score swap the source for an
                // already-covered pixel; those swaps are the rectangular exposure
                // blocks visible in a shifted scan. A painted pixel can still win
                // normally, and a new candidate can still fill an uncovered hole.
                if *decision > 0
                    && *base_valid > 0
                    && *candidate_valid > 0
                    && focus_stack_pixel_is_canvas_like(base_pixel)
                    && focus_stack_pixel_is_canvas_like(candidate_pixel)
                    && !focus_stack_pixel_is_tone_foreground(base_pixel)
                    && !focus_stack_pixel_is_tone_foreground(candidate_pixel)
                {
                    *decision = 0;
                }
            },
        );
}

fn resize_binary_mask(mask: &GrayImage, width: u32, height: u32) -> GrayImage {
    image::imageops::resize(
        mask,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Nearest,
    )
}

fn dilate_focus_binary_mask(mask: &GrayImage, radius: usize) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }
    let seed = mask
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let dilated = box_blur_focus_map(&seed, width, height, radius);
    GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        image::Luma([u8::from(dilated[index] > 0.0) * 255])
    })
}

fn open_focus_binary_mask(mask: &GrayImage, radius: usize) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 || radius == 0 {
        return mask.clone();
    }
    let seed = mask
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let eroded = box_blur_focus_map(&seed, width, height, radius);
    let eroded = GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        image::Luma([u8::from(eroded[index] >= 0.999) * 255])
    });
    dilate_focus_binary_mask(&eroded, radius)
}

fn fill_focus_foreground_holes(mask: &GrayImage) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return mask.clone();
    }
    let width = width as usize;
    let height = height as usize;
    let source = mask.as_raw();
    let mut outside = vec![false; source.len()];
    let mut pending = Vec::new();
    let enqueue = |index: usize, outside: &mut [bool], pending: &mut Vec<usize>| {
        if source[index] == 0 && !outside[index] {
            outside[index] = true;
            pending.push(index);
        }
    };
    for x in 0..width {
        enqueue(x, &mut outside, &mut pending);
        enqueue((height - 1) * width + x, &mut outside, &mut pending);
    }
    for y in 0..height {
        enqueue(y * width, &mut outside, &mut pending);
        enqueue(y * width + width - 1, &mut outside, &mut pending);
    }
    while let Some(index) = pending.pop() {
        let x = index % width;
        let y = index / width;
        if x > 0 {
            enqueue(index - 1, &mut outside, &mut pending);
        }
        if x + 1 < width {
            enqueue(index + 1, &mut outside, &mut pending);
        }
        if y > 0 {
            enqueue(index - width, &mut outside, &mut pending);
        }
        if y + 1 < height {
            enqueue(index + width, &mut outside, &mut pending);
        }
    }

    let mut filled = source.to_vec();
    for (index, value) in filled.iter_mut().enumerate() {
        if *value == 0 && !outside[index] {
            *value = 255;
        }
    }
    GrayImage::from_raw(width as u32, height as u32, filled)
        .expect("focus hole-filled foreground mask dimensions must match")
}

fn retain_focus_foreground_components(
    foreground_mask: &GrayImage,
    seed_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = foreground_mask.dimensions();
    if (width, height) != seed_mask.dimensions() || width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let foreground = foreground_mask.as_raw();
    let seeds = seed_mask.as_raw();
    let mut visited = vec![false; width_usize * height_usize];
    let mut retained = vec![0u8; width_usize * height_usize];
    let minimum_component_size = ((width as u64 * height as u64) as f64 * 0.00018)
        .round()
        .clamp(24.0, 512.0) as usize;
    for start in 0..foreground.len() {
        if foreground[start] == 0 || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut pending = vec![start];
        let mut component = Vec::new();
        let mut has_seed = false;
        let mut minimum_x = start % width_usize;
        let mut maximum_x = minimum_x;
        let mut minimum_y = start / width_usize;
        let mut maximum_y = minimum_y;
        while let Some(index) = pending.pop() {
            component.push(index);
            has_seed |= seeds[index] > 0;
            let x = index % width_usize;
            let y = index / width_usize;
            minimum_x = minimum_x.min(x);
            maximum_x = maximum_x.max(x);
            minimum_y = minimum_y.min(y);
            maximum_y = maximum_y.max(y);
            if x > 0 {
                let neighbor = index - 1;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
            if x + 1 < width_usize {
                let neighbor = index + 1;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
            if y > 0 {
                let neighbor = index - width_usize;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
            if y + 1 < height_usize {
                let neighbor = index + width_usize;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
        }
        // Exposure-shifted canvas noise can satisfy a colour seed locally, but
        // it should not become a protected island. Real painted regions in this
        // stack are much larger and remain connected at analysis resolution.
        let component_width = maximum_x - minimum_x + 1;
        let component_height = maximum_y - minimum_y + 1;
        let minimum_component_width = (width as f32 * 0.006).round().clamp(8.0, 32.0) as usize;
        if has_seed
            && component.len() >= minimum_component_size
            && component_width >= minimum_component_width
            && component_height >= minimum_component_width
        {
            for index in component {
                retained[index] = 255;
            }
        }
    }

    GrayImage::from_raw(width, height, retained)
        .expect("focus retained foreground mask dimensions must match")
}

fn build_focus_background_foreground_envelope(
    analysis_image: &Rgb32FImage,
    analysis_mask: &GrayImage,
    existing_foreground_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = analysis_image.dimensions();
    if width == 0
        || height == 0
        || analysis_mask.dimensions() != (width, height)
        || existing_foreground_mask.dimensions() != (width, height)
    {
        return GrayImage::new(width, height);
    }

    // A raw colour mask contains isolated canvas-weave outliers. Keep only
    // connected components that have enough support to be painted content,
    // then expand the retained silhouette so skin/highlights between the dark
    // hair and robe are protected as one object rather than treated as paper.
    let tone_seed = GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        let covered = analysis_mask.as_raw()[index] > 0;
        let start = index * 3;
        image::Luma([u8::from(
            covered
                && focus_stack_pixel_is_tone_foreground(&analysis_image.as_raw()[start..start + 3]),
        ) * 255])
    });
    // Canvas weave can create long, one-pixel colour runs that connect to a
    // real silhouette. Open the seed before connected-component filtering so
    // those runs cannot become foreground exclusions across an entire source
    // tile. The radius is only a few analysis pixels and is applied to the
    // mask, never to the final image.
    let seed_cleanup_radius = (width.max(height) as f32 * 0.0015).round().clamp(2.0, 4.0) as usize;
    let cleaned_tone_seed = open_focus_binary_mask(&tone_seed, seed_cleanup_radius);
    let retained_seed = retain_focus_foreground_components(&cleaned_tone_seed, &cleaned_tone_seed);
    let retained_seed = fill_focus_foreground_holes(&retained_seed);
    let retained_count = retained_seed
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    let mut envelope = if retained_count >= 24 {
        let radius = (width.max(height) as f32 * FOCUS_BACKGROUND_TONE_FOREGROUND_RADIUS_RATIO)
            .round()
            .clamp(8.0, 64.0) as usize;
        dilate_focus_binary_mask(&retained_seed, radius)
    } else {
        GrayImage::new(width, height)
    };

    // The geometric mask is an additional safety net when a caller has a
    // detected depth layer. It is intentionally dilated by the same bounded
    // halo so the tone pass cannot nibble at its silhouette edge.
    let existing_count = existing_foreground_mask
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if existing_count > 0 {
        let radius = (width.max(height) as f32 * 0.01).round().clamp(4.0, 32.0) as usize;
        let existing = dilate_focus_binary_mask(existing_foreground_mask, radius);
        for (destination, source) in envelope.as_mut().iter_mut().zip(existing.as_raw()) {
            *destination = (*destination).max(*source);
        }
    }
    envelope
}

fn build_focus_background_reference_mask(
    analysis_image: &Rgb32FImage,
    analysis_mask: &GrayImage,
    existing_foreground_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = analysis_image.dimensions();
    if width == 0
        || height == 0
        || analysis_mask.dimensions() != (width, height)
        || existing_foreground_mask.dimensions() != (width, height)
    {
        return GrayImage::new(width, height);
    }

    // Do not use a colour threshold here. The same paper can be considerably
    // darker or lighter in different source frames, which is exactly the
    // exposure step this reference is meant to remove. The foreground
    // envelope is content-derived and is the only exclusion needed here.
    let foreground_envelope = build_focus_background_foreground_envelope(
        analysis_image,
        analysis_mask,
        existing_foreground_mask,
    );
    GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0 && foreground_envelope.as_raw()[index] == 0,
        ) * 255])
    })
}

fn update_focus_background_reference(
    reference_image: &mut Rgb32FImage,
    reference_mask: &mut GrayImage,
    reference_counts: &mut [u16],
    candidate_image: &Rgb32FImage,
    candidate_mask: &GrayImage,
) -> [f32; 3] {
    let (width, height) = reference_image.dimensions();
    if width == 0
        || height == 0
        || candidate_image.dimensions() != (width, height)
        || reference_mask.dimensions() != (width, height)
        || candidate_mask.dimensions() != (width, height)
        || reference_counts.len() != width as usize * height as usize
    {
        return [0.0; 3];
    }

    let reference_pixels = reference_image.as_raw();
    let reference_mask_pixels = reference_mask.as_raw();
    let candidate_pixels = candidate_image.as_raw();
    let candidate_mask_pixels = candidate_mask.as_raw();
    let sample_step = ((u64::from(width) * u64::from(height) / 120_000) as f64)
        .sqrt()
        .ceil()
        .max(1.0) as usize;
    let mut channel_differences = [Vec::new(), Vec::new(), Vec::new()];
    for y in (0..height as usize).step_by(sample_step) {
        for x in (0..width as usize).step_by(sample_step) {
            let index = y * width as usize + x;
            if reference_mask_pixels[index] == 0 || candidate_mask_pixels[index] == 0 {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                channel_differences[channel]
                    .push(reference_pixels[start + channel] - candidate_pixels[start + channel]);
            }
        }
    }

    let correction = [
        median_f32(&mut channel_differences[0])
            .unwrap_or(0.0)
            .clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            ),
        median_f32(&mut channel_differences[1])
            .unwrap_or(0.0)
            .clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            ),
        median_f32(&mut channel_differences[2])
            .unwrap_or(0.0)
            .clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            ),
    ];

    let reference_pixels = reference_image.as_mut();
    let reference_mask_pixels = reference_mask.as_mut();
    for index in 0..candidate_mask_pixels.len() {
        if candidate_mask_pixels[index] == 0 {
            continue;
        }
        let start = index * 3;
        let mut normalized = [0.0f32; 3];
        for channel in 0..3 {
            normalized[channel] =
                (candidate_pixels[start + channel] + correction[channel]).clamp(0.0, 1.0);
        }
        let count = reference_counts[index];
        if count == 0 || reference_mask_pixels[index] == 0 {
            reference_pixels[start..start + 3].copy_from_slice(&normalized);
            reference_counts[index] = 1;
            reference_mask_pixels[index] = 255;
            continue;
        }
        let old_weight = f32::from(count);
        let next_weight = old_weight + 1.0;
        for channel in 0..3 {
            reference_pixels[start + channel] = (reference_pixels[start + channel] * old_weight
                + normalized[channel])
                / next_weight;
        }
        reference_counts[index] = count.saturating_add(1);
    }
    correction
}

fn hard_select_focus_pixels(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    decision_mask: &GrayImage,
) {
    debug_assert_eq!(base.dimensions(), candidate.dimensions());
    debug_assert_eq!(base.dimensions(), base_mask.dimensions());
    debug_assert_eq!(base.dimensions(), candidate_mask.dimensions());
    debug_assert_eq!(base.dimensions(), decision_mask.dimensions());
    let base_pixels: &mut [f32] = base.as_mut();
    let base_mask_pixels: &mut [u8] = base_mask.as_mut();
    base_pixels
        .par_chunks_mut(3)
        .zip(base_mask_pixels.par_iter_mut())
        .zip(candidate.as_raw().par_chunks(3))
        .zip(candidate_mask.as_raw().par_iter())
        .zip(decision_mask.as_raw().par_iter())
        .for_each(
            |((((base_pixel, base_valid), candidate_pixel), candidate_valid), decision)| {
                if *candidate_valid > 0 && (*base_valid == 0 || *decision > 0) {
                    base_pixel.copy_from_slice(candidate_pixel);
                }
                if *candidate_valid > 0 {
                    *base_valid = 255;
                }
            },
        );
}

fn place_focus_layer(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &RenderedFocusLayer,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    let base_stride = base_width as usize * 3;
    let layer_stride = layer_width as usize * 3;
    base.as_mut()
        .par_chunks_mut(base_stride)
        .zip(base_mask.as_mut().par_chunks_mut(base_width as usize))
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(candidate.image.as_raw().par_chunks(layer_stride))
        .zip(candidate.mask.as_raw().par_chunks(layer_width as usize))
        .for_each(
            |(((base_row, base_mask_row), candidate_row), candidate_mask_row)| {
                let base_start = candidate.left as usize * 3;
                let base_end = base_start + layer_stride;
                base_row[base_start..base_end].copy_from_slice(candidate_row);
                let mask_start = candidate.left as usize;
                let mask_end = mask_start + layer_width as usize;
                base_mask_row[mask_start..mask_end].copy_from_slice(candidate_mask_row);
            },
        );
}

fn place_focus_foreground_mask(base: &mut GrayImage, candidate: &RenderedFocusLayer) {
    let (layer_width, layer_height) = candidate.foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    base.as_mut()
        .par_chunks_mut(base_width as usize)
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(
            candidate
                .foreground_mask
                .as_raw()
                .par_chunks(layer_width as usize),
        )
        .for_each(|(base_row, candidate_row)| {
            let start = candidate.left as usize;
            let end = start + layer_width as usize;
            base_row[start..end].copy_from_slice(candidate_row);
        });
}

fn focus_rgb_luma_at(image: &Rgb32FImage, x: i32, y: i32) -> Option<f64> {
    let (width, height) = image.dimensions();
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return None;
    }
    let pixel = image.get_pixel(x as u32, y as u32);
    pixel
        .0
        .iter()
        .all(|value| value.is_finite())
        .then_some(pixel[0] as f64 * 0.299 + pixel[1] as f64 * 0.587 + pixel[2] as f64 * 0.114)
}

fn focus_alignment_gradient_at(
    image: &Rgb32FImage,
    mask: &GrayImage,
    x: i32,
    y: i32,
) -> Option<(f64, f64)> {
    let (width, height) = image.dimensions();
    if x < 1 || y < 1 || x + 1 >= width as i32 || y + 1 >= height as i32 {
        return None;
    }
    let valid =
        |sample_x: i32, sample_y: i32| mask.get_pixel(sample_x as u32, sample_y as u32)[0] > 0;
    if !valid(x, y) || !valid(x - 1, y) || !valid(x + 1, y) || !valid(x, y - 1) || !valid(x, y + 1)
    {
        return None;
    }
    let horizontal = focus_rgb_luma_at(image, x + 1, y)? - focus_rgb_luma_at(image, x - 1, y)?;
    let vertical = focus_rgb_luma_at(image, x, y + 1)? - focus_rgb_luma_at(image, x, y - 1)?;
    Some((horizontal, vertical))
}

fn focus_full_resolution_patch_score(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    center_x: i32,
    center_y: i32,
    delta_x: i32,
    delta_y: i32,
    radius: i32,
) -> Option<(f64, f64, f64)> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    let mut candidate_horizontal_sum = 0.0;
    let mut candidate_vertical_sum = 0.0;
    let mut merged_horizontal_sum = 0.0;
    let mut merged_vertical_sum = 0.0;
    let mut candidate_squared_sum = 0.0;
    let mut merged_squared_sum = 0.0;
    let mut product_sum = 0.0;
    let mut sample_count = 0usize;
    let mut foreground_count = 0usize;
    let mut patch_count = 0usize;
    for offset_y in -radius..=radius {
        for offset_x in -radius..=radius {
            let local_x = center_x + offset_x;
            let local_y = center_y + offset_y;
            if local_x >= 0
                && local_y >= 0
                && local_x < candidate_width as i32
                && local_y < candidate_height as i32
            {
                patch_count += 1;
                if candidate
                    .foreground_mask
                    .get_pixel(local_x as u32, local_y as u32)[0]
                    > 0
                {
                    foreground_count += 1;
                }
            }
            let Some((candidate_horizontal, candidate_vertical)) =
                focus_alignment_gradient_at(&candidate.image, &candidate.mask, local_x, local_y)
            else {
                continue;
            };
            let merged_x = candidate.left as i32 + local_x + delta_x;
            let merged_y = candidate.top as i32 + local_y + delta_y;
            let Some((merged_horizontal, merged_vertical)) =
                focus_alignment_gradient_at(merged, merged_mask, merged_x, merged_y)
            else {
                continue;
            };
            candidate_horizontal_sum += candidate_horizontal;
            candidate_vertical_sum += candidate_vertical;
            merged_horizontal_sum += merged_horizontal;
            merged_vertical_sum += merged_vertical;
            candidate_squared_sum += candidate_horizontal * candidate_horizontal
                + candidate_vertical * candidate_vertical;
            merged_squared_sum +=
                merged_horizontal * merged_horizontal + merged_vertical * merged_vertical;
            product_sum +=
                candidate_horizontal * merged_horizontal + candidate_vertical * merged_vertical;
            sample_count += 1;
        }
    }
    let minimum_samples = (((radius * 2 + 1) * (radius * 2 + 1)) as f64 * 0.45)
        .round()
        .max(48.0) as usize;
    if sample_count < minimum_samples || patch_count == 0 {
        return None;
    }
    let sample_count_f64 = sample_count as f64;
    let candidate_variance = candidate_squared_sum
        - (candidate_horizontal_sum * candidate_horizontal_sum
            + candidate_vertical_sum * candidate_vertical_sum)
            / sample_count_f64;
    let merged_variance = merged_squared_sum
        - (merged_horizontal_sum * merged_horizontal_sum
            + merged_vertical_sum * merged_vertical_sum)
            / sample_count_f64;
    if candidate_variance <= f64::EPSILON || merged_variance <= f64::EPSILON {
        return None;
    }
    let covariance = product_sum
        - (candidate_horizontal_sum * merged_horizontal_sum
            + candidate_vertical_sum * merged_vertical_sum)
            / sample_count_f64;
    let score = covariance / (candidate_variance * merged_variance).sqrt();
    let candidate_energy = (candidate_variance / sample_count_f64).sqrt();
    let merged_energy = (merged_variance / sample_count_f64).sqrt();
    let foreground_fraction = foreground_count as f64 / patch_count as f64;
    (score.is_finite() && candidate_energy.is_finite() && merged_energy.is_finite()).then_some((
        score,
        candidate_energy.min(merged_energy),
        foreground_fraction,
    ))
}

// Build only the two bounded patches used by one correspondence, rather than
// a second full-resolution copy of both 60MP layers. Gaussian decimation samples
// even pixel centers, so multiplying a displacement by two is exact between
// levels. A stride over native pixels is not a pyramid: it can skip an entire
// narrow correlation peak.
fn focus_alignment_patch_pyramid(
    image: &Rgb32FImage,
    mask: &GrayImage,
    foreground: Option<&GrayImage>,
    center_x: i32,
    center_y: i32,
    radius: i32,
    levels: usize,
) -> Vec<RenderedFocusLayer> {
    let size = (2 * radius + 1) as u32;
    let mut first = RenderedFocusLayer {
        image: Rgb32FImage::new(size, size),
        mask: GrayImage::new(size, size),
        foreground_mask: GrayImage::new(size, size),
        relaxed_foreground_mask: GrayImage::new(0, 0),
        left: 0,
        top: 0,
    };
    for y in 0..size {
        for x in 0..size {
            let sx = center_x + x as i32 - radius;
            let sy = center_y + y as i32 - radius;
            if sx < 0 || sy < 0 || sx >= image.width() as i32 || sy >= image.height() as i32 {
                continue;
            }
            first
                .image
                .put_pixel(x, y, *image.get_pixel(sx as u32, sy as u32));
            first
                .mask
                .put_pixel(x, y, *mask.get_pixel(sx as u32, sy as u32));
            if let Some(foreground) = foreground {
                first
                    .foreground_mask
                    .put_pixel(x, y, *foreground.get_pixel(sx as u32, sy as u32));
            }
        }
    }
    let mut pyramid = vec![first];
    for _ in 0..levels {
        let previous = pyramid.last().unwrap();
        let width = (previous.image.width() + 1) / 2;
        let height = (previous.image.height() + 1) / 2;
        let mut next = RenderedFocusLayer {
            image: Rgb32FImage::new(width, height),
            mask: GrayImage::new(width, height),
            foreground_mask: GrayImage::new(width, height),
            relaxed_foreground_mask: GrayImage::new(0, 0),
            left: 0,
            top: 0,
        };
        for y in 0..height {
            for x in 0..width {
                let mut pixel = [0.0f32; 3];
                let mut valid = true;
                let mut foreground = 0;
                for oy in -1i32..=1 {
                    for ox in -1i32..=1 {
                        let sx =
                            (2 * x as i32 + ox).clamp(0, previous.image.width() as i32 - 1) as u32;
                        let sy =
                            (2 * y as i32 + oy).clamp(0, previous.image.height() as i32 - 1) as u32;
                        let weight = (if ox == 0 { 2.0 } else { 1.0 })
                            * (if oy == 0 { 2.0 } else { 1.0 })
                            / 16.0;
                        let sample = previous.image.get_pixel(sx, sy);
                        for channel in 0..3 {
                            pixel[channel] += sample[channel] * weight;
                        }
                        valid &= previous.mask.get_pixel(sx, sy)[0] > 0;
                        foreground |= previous.foreground_mask.get_pixel(sx, sy)[0];
                    }
                }
                next.image.put_pixel(x, y, Rgb(pixel));
                next.mask
                    .put_pixel(x, y, image::Luma([u8::from(valid) * 255]));
                next.foreground_mask
                    .put_pixel(x, y, image::Luma([foreground]));
            }
        }
        pyramid.push(next);
    }
    pyramid
}

fn focus_patch_translation_candidates(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    center_x: i32,
    center_y: i32,
    max_shift: i32,
) -> Vec<(f64, f64, i32, i32)> {
    let mut levels = 0;
    while (max_shift >> levels) > 24 {
        levels += 1;
    }
    let scale = 1 << levels;
    let patch_radius = FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS;
    let candidate_radius = (patch_radius + 2) * scale;
    let merged_radius = candidate_radius + (max_shift + scale - 1) / scale * scale;
    let mut candidate_pyramid = focus_alignment_patch_pyramid(
        &candidate.image,
        &candidate.mask,
        Some(&candidate.foreground_mask),
        center_x,
        center_y,
        candidate_radius,
        levels,
    );
    let merged_pyramid = focus_alignment_patch_pyramid(
        merged,
        merged_mask,
        None,
        candidate.left as i32 + center_x,
        candidate.top as i32 + center_y,
        merged_radius,
        levels,
    );
    let mut beam: Vec<(f64, f64, i32, i32)> = Vec::new();
    for level in (0..=levels).rev() {
        let scale = 1 << level;
        let limit = (max_shift + scale - 1) / scale;
        let candidate_patch = &mut candidate_pyramid[level];
        candidate_patch.left = ((merged_radius - candidate_radius) / scale) as u32;
        candidate_patch.top = candidate_patch.left;
        let merged_patch = &merged_pyramid[level];
        let mut offsets = Vec::new();
        if beam.is_empty() && level == levels {
            for dy in -limit..=limit {
                for dx in -limit..=limit {
                    offsets.push((dx, dy));
                }
            }
        } else {
            for &(_, _, coarse_x, coarse_y) in &beam {
                // Coarse focus texture can shift by several pixels after
                // Gaussian decimation. Keep a broad native refinement window
                // around each of the few coarse seeds; a tiny ±3 window can
                // permanently exclude the true subpixel peak.
                for dy in -16..=16 {
                    for dx in -16..=16 {
                        let x = coarse_x * 2 + dx;
                        let y = coarse_y * 2 + dy;
                        if x.abs() <= limit && y.abs() <= limit {
                            offsets.push((x, y));
                        }
                    }
                }
            }
        }
        // The baseline must always be measured, including when the search
        // bounds are not an exact multiple of a sampling stride.
        offsets.push((0, 0));
        offsets.sort_unstable();
        offsets.dedup();
        let mut scored = offsets
            .into_iter()
            .filter_map(|(dx, dy)| {
                let (score, energy, foreground_fraction) = focus_full_resolution_patch_score(
                    candidate_patch,
                    &merged_patch.image,
                    &merged_patch.mask,
                    candidate_radius / scale,
                    candidate_radius / scale,
                    dx,
                    dy,
                    patch_radius,
                )?;
                // Coarse levels only propose locations. Native pixels supply the
                // final quality gate, allowing a low-contrast coarse feature to
                // lead to a sharply localized, well-supported native peak.
                (foreground_fraction <= FOCUS_FULL_RES_ALIGNMENT_MAX_FOREGROUND_FRACTION
                    && energy > 1e-6)
                    .then_some((score, energy, dx, dy))
            })
            .collect::<Vec<_>>();
        scored.sort_unstable_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| (a.2.abs() + a.3.abs()).cmp(&(b.2.abs() + b.3.abs())))
        });
        if level == 0 {
            return scored;
        }
        beam.clear();
        for entry in scored {
            if beam.iter().all(|previous| {
                (previous.2 - entry.2).abs() > 2 || (previous.3 - entry.3).abs() > 2
            }) {
                beam.push(entry);
                if beam.len() == 4 {
                    break;
                }
            }
        }
        if beam.is_empty() {
            return Vec::new();
        }
    }
    Vec::new()
}

/// Run the Intra_Station_Registrar over one already rendered layer (需求 2.2–2.9).
///
/// The measurement itself is `mosaic::register_station_frame_native`, the same
/// code the mosaic comparison path uses; this only reconstructs the pose the
/// caller actually rendered with.  `applied_translation` is the integer shift
/// `translate_rendered_focus_layer` has already put on those pixels, and a canvas
/// shift is a translation in the projection plane, so pre-multiplying the
/// station homography with it reproduces the layer exactly.
#[allow(clippy::too_many_arguments)]
fn register_station_frame_native_layer(
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    image: &ImageInfo,
    source_image: &Rgb32FImage,
    homography: &Matrix3<f64>,
    projection: Projection,
    offset: (f64, f64),
    applied_translation: (i32, i32),
    candidate: &RenderedFocusLayer,
    anchor_long_side: u32,
    anchor_to_station_inverse: Option<Matrix3<f64>>,
) -> Option<super::mosaic::StationFrameRegistration> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    if candidate_width < 2 || candidate_height < 2 {
        return None;
    }
    let mut translation = Matrix3::identity();
    translation[(0, 2)] = f64::from(applied_translation.0);
    translation[(1, 2)] = f64::from(applied_translation.1);
    let rendered_pose = translation * homography;
    // 需求 2.1: every record is expressed in the anchor frame's coordinates.
    let into_anchor = anchor_to_station_inverse
        .map(|inverse| inverse * rendered_pose)
        .unwrap_or_else(Matrix3::identity);
    super::mosaic::register_station_frame_native(
        merged,
        merged_mask,
        image,
        source_image,
        &rendered_pose,
        projection,
        offset,
        (
            candidate.left,
            candidate.left + candidate_width - 1,
            candidate.top,
            candidate.top + candidate_height - 1,
        ),
        anchor_long_side,
        into_anchor,
    )
}

/// Apply a measured native displacement field to an already rendered layer
/// (需求 2.3–2.7).
///
/// The field is what [`super::mosaic::register_station_frame_native`] accepted,
/// expressed as a canvas-pixel offset to add before sampling, so resampling the
/// layer at that offset moves its pixels by exactly the accepted correction. A
/// pixel whose bilinear support is not fully covered loses coverage instead of
/// borrowing an invented neighbour.
fn warp_rendered_focus_layer(
    candidate: RenderedFocusLayer,
    displacement: &super::mosaic::NativeResidualDisplacement,
) -> RenderedFocusLayer {
    let (width, height) = candidate.image.dimensions();
    if width == 0 || height == 0 {
        return candidate;
    }
    let row_pixels = width as usize * 3;
    let mut pixels = vec![0.0f32; row_pixels * height as usize];
    let mut mask = vec![0u8; width as usize * height as usize];
    let mut foreground_mask = vec![0u8; width as usize * height as usize];
    let mut relaxed_foreground_mask = vec![0u8; width as usize * height as usize];
    pixels
        .par_chunks_mut(row_pixels)
        .zip(mask.par_chunks_mut(width as usize))
        .zip(foreground_mask.par_chunks_mut(width as usize))
        .zip(relaxed_foreground_mask.par_chunks_mut(width as usize))
        .enumerate()
        .for_each(|(y, (((row, mask_row), foreground_row), relaxed_row))| {
            for x in 0..width as usize {
                let [delta_x, delta_y] = displacement.at(
                    f64::from(candidate.left) + x as f64,
                    f64::from(candidate.top) + y as f64,
                );
                let sample_x = x as f64 + delta_x;
                let sample_y = y as f64 + delta_y;
                if !sample_x.is_finite() || !sample_y.is_finite() {
                    continue;
                }
                let x0 = sample_x.floor();
                let y0 = sample_y.floor();
                if x0 < 0.0
                    || y0 < 0.0
                    || x0 + 1.0 >= f64::from(width)
                    || y0 + 1.0 >= f64::from(height)
                {
                    continue;
                }
                let (x0, y0) = (x0 as u32, y0 as u32);
                if [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
                    .iter()
                    .any(|&(px, py)| candidate.mask.get_pixel(px, py)[0] == 0)
                {
                    continue;
                }
                let fx = (sample_x - f64::from(x0)) as f32;
                let fy = (sample_y - f64::from(y0)) as f32;
                for channel in 0..3 {
                    let at = |px: u32, py: u32| candidate.image.get_pixel(px, py)[channel];
                    let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
                    let bottom = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
                    row[x * 3 + channel] = top * (1.0 - fy) + bottom * fy;
                }
                mask_row[x] = 255;
                // The detected layer masks are categorical, so they follow
                // the nearest sample rather than an average of two classes.
                let nearest_x = (sample_x.round() as u32).min(width - 1);
                let nearest_y = (sample_y.round() as u32).min(height - 1);
                foreground_row[x] = candidate.foreground_mask.get_pixel(nearest_x, nearest_y)[0];
                relaxed_row[x] = candidate
                    .relaxed_foreground_mask
                    .get_pixel(nearest_x, nearest_y)[0];
            }
        });
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(width, height, pixels).expect("warped layer dimensions"),
        mask: GrayImage::from_raw(width, height, mask).expect("warped mask dimensions"),
        foreground_mask: GrayImage::from_raw(width, height, foreground_mask)
            .expect("warped foreground dimensions"),
        relaxed_foreground_mask: GrayImage::from_raw(width, height, relaxed_foreground_mask)
            .expect("warped relaxed foreground dimensions"),
        left: candidate.left,
        top: candidate.top,
    }
}

fn estimate_focus_layer_translation(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
) -> Option<(i32, i32, f64, usize, usize)> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    let (merged_width, merged_height) = merged.dimensions();
    if candidate_width < 2 * (FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS + 2) as u32
        || candidate_height < 2 * (FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS + 2) as u32
        || merged_width < 2
        || merged_height < 2
        || merged_mask.dimensions() != merged.dimensions()
    {
        return None;
    }

    let patch_radius = FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS;
    let overlap_left = candidate.left.max(0).min(merged_width.saturating_sub(1)) as i32;
    let overlap_top = candidate.top.max(0).min(merged_height.saturating_sub(1)) as i32;
    let overlap_right = (candidate
        .left
        .saturating_add(candidate_width)
        .saturating_sub(1))
    .min(merged_width.saturating_sub(1)) as i32;
    let overlap_bottom = (candidate
        .top
        .saturating_add(candidate_height)
        .saturating_sub(1))
    .min(merged_height.saturating_sub(1)) as i32;
    if overlap_right - overlap_left < patch_radius * 2 + 2
        || overlap_bottom - overlap_top < patch_radius * 2 + 2
    {
        return None;
    }

    let grid_size = FOCUS_FULL_RES_ALIGNMENT_GRID_SIZE;
    let mut patch_centers = Vec::with_capacity((grid_size * grid_size) as usize);
    for grid_y in 0..grid_size {
        let global_y =
            overlap_top + ((overlap_bottom - overlap_top) * (grid_y + 1) / (grid_size + 1));
        for grid_x in 0..grid_size {
            let global_x =
                overlap_left + ((overlap_right - overlap_left) * (grid_x + 1) / (grid_size + 1));
            let local_x = global_x - candidate.left as i32;
            let local_y = global_y - candidate.top as i32;
            patch_centers.push((local_x, local_y));
        }
    }

    let mut patch_deltas = Vec::new();
    let image_short_side = candidate_width.min(candidate_height) as i32;
    let max_shift = FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT
        .min((image_short_side as f32 * 0.08).round() as i32)
        .max(1);
    for (center_x, center_y) in patch_centers.iter().copied() {
        // Record zero-shift NCC before applying the acceptance threshold. A
        // genuinely displaced patch commonly has low baseline correlation;
        // rejecting that baseline would make large corrections impossible.
        let current_score = focus_full_resolution_patch_score(
            candidate,
            merged,
            merged_mask,
            center_x,
            center_y,
            0,
            0,
            patch_radius,
        )
        .map(|value| value.0);
        let scores = focus_patch_translation_candidates(
            candidate,
            merged,
            merged_mask,
            center_x,
            center_y,
            max_shift,
        );
        let Some(&(best_score, energy, best_x, best_y)) = scores.first() else {
            continue;
        };
        let second_best = scores
            .iter()
            .find(|entry| (entry.2 - best_x).abs() > 2 || (entry.3 - best_y).abs() > 2)
            .map(|entry| entry.0)
            .unwrap_or(f64::NEG_INFINITY);
        if best_score < FOCUS_FULL_RES_ALIGNMENT_MIN_NCC
            || energy < FOCUS_FULL_RES_ALIGNMENT_MIN_ENERGY
            || best_score - second_best < FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN
            || (best_x != 0 || best_y != 0)
                && current_score
                    .is_some_and(|score| best_score - score < FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN)
        {
            continue;
        }
        patch_deltas.push((best_x, best_y, best_score));
    }
    if std::env::var_os("RAW_EDITOR_FOCUS_TRANSLATION_DIAGNOSTICS").is_some() {
        println!(
            "  - Focus translation diagnostics: candidate={}x{} patches={} accepted={}",
            candidate_width,
            candidate_height,
            patch_centers.len(),
            patch_deltas.len()
        );
    }
    if patch_deltas.is_empty() {
        return None;
    }

    let mut x_values = patch_deltas.iter().map(|(x, _, _)| *x).collect::<Vec<_>>();
    let mut y_values = patch_deltas.iter().map(|(_, y, _)| *y).collect::<Vec<_>>();
    x_values.sort_unstable();
    y_values.sort_unstable();
    let median_x = x_values[x_values.len() / 2];
    let median_y = y_values[y_values.len() / 2];
    let support = patch_deltas
        .iter()
        .filter(|(x, y, _)| (x - median_x).abs() <= 1 && (y - median_y).abs() <= 1)
        .count();
    let minimum_support = FOCUS_FULL_RES_ALIGNMENT_MIN_PATCHES;
    if support < minimum_support
        || (support as f64) < patch_deltas.len() as f64 * 0.5
        || (median_x == 0 && median_y == 0)
    {
        return None;
    }
    Some((
        median_x,
        median_y,
        patch_deltas
            .iter()
            .filter(|(x, y, _)| (x - median_x).abs() <= 1 && (y - median_y).abs() <= 1)
            .map(|(_, _, score)| score)
            .sum::<f64>()
            / support as f64,
        support,
        patch_deltas.len(),
    ))
}

fn translate_rendered_focus_layer(
    candidate: RenderedFocusLayer,
    delta_x: i32,
    delta_y: i32,
    out_width: u32,
    out_height: u32,
) -> RenderedFocusLayer {
    if delta_x == 0 && delta_y == 0 {
        return candidate;
    }
    let (width, height) = candidate.image.dimensions();
    if width == 0 || height == 0 || out_width == 0 || out_height == 0 {
        return candidate;
    }
    let old_left = candidate.left as i64;
    let old_top = candidate.top as i64;
    let old_right = old_left + width as i64 - 1;
    let old_bottom = old_top + height as i64 - 1;
    let new_left = (old_left + i64::from(delta_x)).clamp(0, i64::from(out_width - 1));
    let new_top = (old_top + i64::from(delta_y)).clamp(0, i64::from(out_height - 1));
    let new_right = (old_right + i64::from(delta_x)).clamp(0, i64::from(out_width - 1));
    let new_bottom = (old_bottom + i64::from(delta_y)).clamp(0, i64::from(out_height - 1));
    if new_left > new_right || new_top > new_bottom {
        return RenderedFocusLayer {
            image: Rgb32FImage::new(0, 0),
            mask: GrayImage::new(0, 0),
            foreground_mask: GrayImage::new(0, 0),
            relaxed_foreground_mask: GrayImage::new(0, 0),
            left: 0,
            top: 0,
        };
    }
    let new_width = (new_right - new_left + 1) as u32;
    let new_height = (new_bottom - new_top + 1) as u32;
    let mut image_pixels = vec![0.0f32; new_width as usize * new_height as usize * 3];
    let mut mask_pixels = vec![0u8; new_width as usize * new_height as usize];
    let mut foreground_pixels = vec![0u8; new_width as usize * new_height as usize];
    let mut relaxed_pixels = vec![0u8; new_width as usize * new_height as usize];
    let source_image = candidate.image.as_raw();
    let source_mask = candidate.mask.as_raw();
    let source_foreground = candidate.foreground_mask.as_raw();
    let source_relaxed = candidate.relaxed_foreground_mask.as_raw();
    for source_y in 0..height as i64 {
        for source_x in 0..width as i64 {
            let destination_x = old_left + source_x + i64::from(delta_x);
            let destination_y = old_top + source_y + i64::from(delta_y);
            if destination_x < new_left
                || destination_x > new_right
                || destination_y < new_top
                || destination_y > new_bottom
            {
                continue;
            }
            let source_index = source_y as usize * width as usize + source_x as usize;
            let destination_index = (destination_y - new_top) as usize * new_width as usize
                + (destination_x - new_left) as usize;
            let source_start = source_index * 3;
            let destination_start = destination_index * 3;
            image_pixels[destination_start..destination_start + 3]
                .copy_from_slice(&source_image[source_start..source_start + 3]);
            mask_pixels[destination_index] = source_mask[source_index];
            foreground_pixels[destination_index] = source_foreground[source_index];
            relaxed_pixels[destination_index] = source_relaxed[source_index];
        }
    }
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(new_width, new_height, image_pixels)
            .expect("translated focus layer image dimensions must match"),
        mask: GrayImage::from_raw(new_width, new_height, mask_pixels)
            .expect("translated focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(new_width, new_height, foreground_pixels)
            .expect("translated focus foreground mask dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(new_width, new_height, relaxed_pixels)
            .expect("translated focus relaxed foreground mask dimensions must match"),
        left: new_left as u32,
        top: new_top as u32,
    }
}

fn align_focus_foreground_layer_to_existing(
    candidate: RenderedFocusLayer,
    merged_foreground_mask: &GrayImage,
) -> RenderedFocusLayer {
    let (layer_width, layer_height) = candidate.foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 || merged_foreground_mask.width() == 0 {
        return candidate;
    }

    let foreground = candidate.foreground_mask.as_raw();
    let foreground_count = foreground.iter().filter(|value| **value > 0).count();
    let existing_count = merged_foreground_mask
        .as_raw()
        .iter()
        .filter(|value| **value > 0)
        .count();
    if foreground_count < 32 || existing_count < 32 {
        return candidate;
    }

    // Use a bounded, deterministic set of mask anchors. This makes the
    // refinement cheap even for a full 60MP source while retaining enough
    // support to distinguish a real overlap from an accidental nearby edge.
    let anchor_stride = (foreground_count / 2_048).max(1);
    let mut anchors = Vec::with_capacity(foreground_count.min(2_048));
    let mut foreground_index = 0usize;
    for (index, value) in foreground.iter().enumerate() {
        if *value == 0 {
            continue;
        }
        if foreground_index.is_multiple_of(anchor_stride) {
            let local_x = (index % layer_width as usize) as u32;
            let local_y = (index / layer_width as usize) as u32;
            anchors.push((candidate.left + local_x, candidate.top + local_y));
        }
        foreground_index += 1;
    }
    if anchors.len() < 32 {
        return candidate;
    }

    let base_width = merged_foreground_mask.width() as i32;
    let base_height = merged_foreground_mask.height() as i32;
    let base_pixels = merged_foreground_mask.as_raw();
    let search_radius = ((layer_width.max(layer_height) as f32) * 0.01)
        .round()
        .clamp(4.0, FOCUS_FOREGROUND_REFINEMENT_MAX_SHIFT as f32) as i32;
    let coarse_step = (search_radius / 24).max(1);
    let hit_count = |delta_x: i32, delta_y: i32| -> usize {
        anchors
            .iter()
            .filter(|&&(x, y)| {
                let x = x as i32 + delta_x;
                let y = y as i32 + delta_y;
                x >= 0
                    && y >= 0
                    && x < base_width
                    && y < base_height
                    && base_pixels[y as usize * base_width as usize + x as usize] > 0
            })
            .count()
    };

    let current_hits = hit_count(0, 0);
    let mut best_delta = (0i32, 0i32);
    let mut best_hits = current_hits;
    for delta_y in (-search_radius..=search_radius).step_by(coarse_step as usize) {
        for delta_x in (-search_radius..=search_radius).step_by(coarse_step as usize) {
            let hits = hit_count(delta_x, delta_y);
            if hits > best_hits {
                best_hits = hits;
                best_delta = (delta_x, delta_y);
            }
        }
    }
    // Refine around the best coarse translation at pixel precision. The
    // foreground mask is only a registration cue; the final source pixels are
    // still selected by the focus score below.
    if coarse_step > 1 {
        let (coarse_x, coarse_y) = best_delta;
        let fine_min_y = (coarse_y - coarse_step + 1).max(-search_radius);
        let fine_max_y = (coarse_y + coarse_step - 1).min(search_radius);
        let fine_min_x = (coarse_x - coarse_step + 1).max(-search_radius);
        let fine_max_x = (coarse_x + coarse_step - 1).min(search_radius);
        for delta_y in fine_min_y..=fine_max_y {
            for delta_x in fine_min_x..=fine_max_x {
                let hits = hit_count(delta_x, delta_y);
                if hits > best_hits {
                    best_hits = hits;
                    best_delta = (delta_x, delta_y);
                }
            }
        }
    }

    let minimum_gain = (anchors.len() / 20).max(12);
    if best_delta == (0, 0)
        || best_hits < current_hits.saturating_add(minimum_gain)
        || best_hits < anchors.len() / 20
    {
        return candidate;
    }

    let (delta_x, delta_y) = best_delta;
    let mut image_pixels = candidate.image.into_raw();
    let mut candidate_mask = candidate.mask.into_raw();
    let mut foreground_mask = vec![0u8; foreground.len()];
    let relaxed_foreground = candidate.relaxed_foreground_mask.as_raw();
    let mut relaxed_foreground_mask = vec![0u8; foreground.len()];
    let mut moved_pixels = Vec::with_capacity(foreground_count);
    for (index, value) in foreground.iter().enumerate() {
        if *value == 0 {
            continue;
        }
        let local_x = (index % layer_width as usize) as i32;
        let local_y = (index / layer_width as usize) as i32;
        let destination_x = local_x + delta_x;
        let destination_y = local_y + delta_y;
        if destination_x < 0
            || destination_y < 0
            || destination_x >= layer_width as i32
            || destination_y >= layer_height as i32
        {
            continue;
        }
        let destination_index =
            destination_y as usize * layer_width as usize + destination_x as usize;
        let source_start = index * 3;
        moved_pixels.push((
            destination_index,
            [
                image_pixels[source_start],
                image_pixels[source_start + 1],
                image_pixels[source_start + 2],
            ],
        ));
        candidate_mask[index] = 0;
        foreground_mask[destination_index] = 255;
        if relaxed_foreground[index] > 0 {
            relaxed_foreground_mask[destination_index] = 255;
        }
    }
    for (destination_index, pixel) in moved_pixels {
        let destination_start = destination_index * 3;
        image_pixels[destination_start..destination_start + 3].copy_from_slice(&pixel);
        candidate_mask[destination_index] = 255;
    }

    println!(
        "  - Focus foreground mask refinement: delta=({delta_x},{delta_y}), overlap={current_hits}->{best_hits} anchors={}",
        anchors.len()
    );
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(layer_width, layer_height, image_pixels)
            .expect("aligned focus layer image dimensions must match"),
        mask: GrayImage::from_raw(layer_width, layer_height, candidate_mask)
            .expect("aligned focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(layer_width, layer_height, foreground_mask)
            .expect("aligned focus foreground mask dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(
            layer_width,
            layer_height,
            relaxed_foreground_mask,
        )
        .expect("aligned focus relaxed foreground mask dimensions must match"),
        left: candidate.left,
        top: candidate.top,
    }
}

fn suppress_focus_foreground_switches(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
) {
    suppress_focus_foreground_switches_in_region(
        decision_mask,
        merged_foreground_mask,
        candidate.left,
        candidate.top,
        &candidate.foreground_mask,
        &candidate.relaxed_foreground_mask,
    );
}

fn suppress_focus_foreground_switches_in_region(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    candidate_foreground_mask: &GrayImage,
    relaxed_foreground_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate_foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = merged_foreground_mask.width();
    let base_height = merged_foreground_mask.height();
    if candidate_left >= base_width
        || candidate_top >= base_height
        || candidate_left + layer_width > base_width
        || candidate_top + layer_height > base_height
    {
        return;
    }
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));

    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        candidate_left,
        candidate_top,
        (layer_width, layer_height),
    );
    if relaxed_foreground_mask.dimensions() != (layer_width, layer_height) {
        return;
    }

    decision_mask
        .as_mut()
        .par_chunks_mut(layer_width as usize)
        .enumerate()
        .for_each(|(local_y, decision_row)| {
            let foreground_row = &candidate_foreground_mask.as_raw()
                [local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            let overlap_row =
                &overlap_mask[local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            let relaxed_row = &relaxed_foreground_mask.as_raw()
                [local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            for (local_x, decision) in decision_row.iter_mut().enumerate() {
                if foreground_row[local_x] > 0
                    && relaxed_row[local_x] == 0
                    && overlap_row[local_x] > 0
                {
                    *decision = 0;
                }
            }
        });
}

fn suppress_focus_foreground_switches_on_canvas(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_foreground_mask: &GrayImage,
    relaxed_foreground_mask: &GrayImage,
) {
    if decision_mask.dimensions() != merged_foreground_mask.dimensions()
        || decision_mask.dimensions() != candidate_foreground_mask.dimensions()
        || candidate_foreground_mask.dimensions() != relaxed_foreground_mask.dimensions()
    {
        return;
    }
    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        0,
        0,
        candidate_foreground_mask.dimensions(),
    );
    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(candidate_foreground_mask.as_raw().par_iter())
        .zip(overlap_mask.par_iter())
        .zip(relaxed_foreground_mask.as_raw().par_iter())
        .for_each(|(((decision, candidate_foreground), overlap), relaxed)| {
            if *candidate_foreground > 0 && *relaxed == 0 && *overlap > 0 {
                *decision = 0;
            }
        });
}

fn suppress_focus_foreground_ownership_switches(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    relaxed_foreground_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate_mask.dimensions();
    if layer_width == 0
        || layer_height == 0
        || decision_mask.dimensions() != (layer_width, layer_height)
        || relaxed_foreground_mask.dimensions() != (layer_width, layer_height)
    {
        return;
    }
    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        candidate_left,
        candidate_top,
        (layer_width, layer_height),
    );
    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(candidate_mask.as_raw().par_iter())
        .zip(overlap_mask.par_iter())
        .zip(relaxed_foreground_mask.as_raw().par_iter())
        .for_each(|(((decision, candidate_valid), overlap), relaxed)| {
            if *candidate_valid > 0 && *relaxed == 0 && *overlap > 0 {
                *decision = 0;
            }
        });
}

fn suppress_focus_foreground_ownership_switches_on_canvas(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_mask: &GrayImage,
    relaxed_foreground_mask: &GrayImage,
) {
    if decision_mask.dimensions() != merged_foreground_mask.dimensions()
        || decision_mask.dimensions() != candidate_mask.dimensions()
        || candidate_mask.dimensions() != relaxed_foreground_mask.dimensions()
    {
        return;
    }
    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        0,
        0,
        candidate_mask.dimensions(),
    );
    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(candidate_mask.as_raw().par_iter())
        .zip(overlap_mask.par_iter())
        .zip(relaxed_foreground_mask.as_raw().par_iter())
        .for_each(|(((decision, candidate_valid), overlap), relaxed)| {
            if *candidate_valid > 0 && *relaxed == 0 && *overlap > 0 {
                *decision = 0;
            }
        });
}

fn focus_foreground_overlap_mask_for_region(
    merged_foreground_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    candidate_dimensions: (u32, u32),
) -> Vec<u8> {
    let (layer_width, layer_height) = candidate_dimensions;
    let pixel_count = layer_width as usize * layer_height as usize;
    let mut overlap = vec![0u8; pixel_count];
    if layer_width == 0 || layer_height == 0 {
        return overlap;
    }

    let radius = ((layer_width.max(layer_height) as f32) * FOCUS_FOREGROUND_OWNERSHIP_RADIUS_RATIO)
        .round()
        .clamp(8.0, 256.0) as usize;
    let base_width = merged_foreground_mask.width() as usize;
    let base_height = merged_foreground_mask.height() as usize;
    let left = candidate_left as usize;
    let top = candidate_top as usize;
    let width = layer_width as usize;
    let height = layer_height as usize;
    if left >= base_width
        || top >= base_height
        || left + width > base_width
        || top + height > base_height
    {
        return overlap;
    }

    // First apply a horizontal max filter to the already selected foreground
    // mask. This is equivalent to testing the whole rectangular neighbourhood,
    // but is linear in the candidate size and does not allocate a full canvas.
    let mut horizontal = vec![0u8; pixel_count];
    let base_pixels = merged_foreground_mask.as_raw();
    for local_y in 0..height {
        let base_row = &base_pixels[(top + local_y) * base_width..(top + local_y + 1) * base_width];
        let initial_start = left.saturating_sub(radius);
        let initial_end = (left + radius).min(base_width - 1);
        let mut window = base_row[initial_start..=initial_end]
            .iter()
            .filter(|value| **value > 0)
            .count() as u32;
        for local_x in 0..width {
            if local_x > 0 {
                let center_x = left + local_x;
                let add_x = (center_x + radius).min(base_width - 1);
                window += u32::from(base_row[add_x] > 0);
                if center_x > radius {
                    let remove_x = center_x - radius - 1;
                    window -= u32::from(base_row[remove_x] > 0);
                }
            }
            horizontal[local_y * width + local_x] = u8::from(window > 0);
        }
    }

    // Then apply the same max filter vertically. A later frame only loses a
    // foreground switch when it really overlaps this expanded ownership area;
    // unrelated artwork remains governed by the normal focus score.
    for local_x in 0..width {
        let initial_end = radius.min(height - 1);
        let mut window = (0..=initial_end)
            .map(|row| u32::from(horizontal[row * width + local_x] > 0))
            .sum::<u32>();
        for local_y in 0..height {
            if local_y > 0 {
                let add_y = (local_y + radius).min(height - 1);
                window += u32::from(horizontal[add_y * width + local_x] > 0);
                if local_y > radius {
                    let remove_y = local_y.saturating_sub(radius + 1);
                    window -= u32::from(horizontal[remove_y * width + local_x] > 0);
                }
            }
            overlap[local_y * width + local_x] = u8::from(window > 0);
        }
    }
    overlap
}

fn focus_foreground_edge_mask(mask: &[u8], width: u32, height: u32, radius: usize) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    if width == 0 || height == 0 || mask.len() != width * height {
        return Vec::new();
    }
    if radius == 0 {
        return mask.to_vec();
    }

    // Erode the protected foreground with separable prefix-sum windows. The
    // pixels left outside that erosion are the silhouette band; the interior
    // can participate in low-frequency exposure blending without averaging a
    // displaced edge.
    let mut horizontal = vec![0u8; mask.len()];
    for y in 0..height {
        let row_start = y * width;
        let mut prefix = vec![0u32; width + 1];
        for x in 0..width {
            prefix[x + 1] = prefix[x] + u32::from(mask[row_start + x] > 0);
        }
        for x in 0..width {
            let start = x.saturating_sub(radius);
            let end = (x + radius).min(width - 1);
            let window_length = (end - start + 1) as u32;
            if prefix[end + 1] - prefix[start] == window_length {
                horizontal[row_start + x] = 255;
            }
        }
    }

    let mut eroded = vec![0u8; mask.len()];
    for x in 0..width {
        let mut prefix = vec![0u32; height + 1];
        for y in 0..height {
            prefix[y + 1] = prefix[y] + u32::from(horizontal[y * width + x] > 0);
        }
        for y in 0..height {
            let start = y.saturating_sub(radius);
            let end = (y + radius).min(height - 1);
            let window_length = (end - start + 1) as u32;
            if prefix[end + 1] - prefix[start] == window_length {
                eroded[y * width + x] = 255;
            }
        }
    }

    mask.iter()
        .zip(eroded)
        .map(|(value, interior)| u8::from(*value > 0 && interior == 0))
        .collect()
}

fn mark_focus_foreground_pixels(
    merged_foreground_mask: &mut GrayImage,
    merged_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
) {
    mark_focus_foreground_pixels_in_region(
        merged_foreground_mask,
        merged_mask,
        candidate.left,
        candidate.top,
        &candidate.foreground_mask,
        decision_mask,
    );
}

fn mark_focus_foreground_pixels_in_region(
    merged_foreground_mask: &mut GrayImage,
    merged_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    candidate_foreground_mask: &GrayImage,
    decision_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate_foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = merged_foreground_mask.width();
    let base_height = merged_foreground_mask.height();
    if candidate_left >= base_width
        || candidate_top >= base_height
        || candidate_left + layer_width > base_width
        || candidate_top + layer_height > base_height
    {
        return;
    }
    debug_assert_eq!(
        merged_mask.dimensions(),
        merged_foreground_mask.dimensions()
    );
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));

    merged_foreground_mask
        .as_mut()
        .par_chunks_mut(base_width as usize)
        .skip(candidate_top as usize)
        .take(layer_height as usize)
        .zip(
            merged_mask
                .as_raw()
                .par_chunks(base_width as usize)
                .skip(candidate_top as usize),
        )
        .zip(
            candidate_foreground_mask
                .as_raw()
                .par_chunks(layer_width as usize),
        )
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(
            |(((foreground_row, merged_mask_row), candidate_foreground_row), decision_row)| {
                for local_x in 0..layer_width as usize {
                    if candidate_foreground_row[local_x] > 0
                        && (merged_mask_row[candidate_left as usize + local_x] == 0
                            || decision_row[local_x] > 0)
                    {
                        foreground_row[candidate_left as usize + local_x] = 255;
                    }
                }
            },
        );
}

fn hard_select_focus_layer(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    debug_assert_eq!(candidate.mask.dimensions(), (layer_width, layer_height));
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));
    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    let base_stride = base_width as usize * 3;
    let layer_stride = layer_width as usize * 3;
    base.as_mut()
        .par_chunks_mut(base_stride)
        .zip(base_mask.as_mut().par_chunks_mut(base_width as usize))
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(candidate.image.as_raw().par_chunks(layer_stride))
        .zip(candidate.mask.as_raw().par_chunks(layer_width as usize))
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(
            |((((base_row, base_mask_row), candidate_row), candidate_mask_row), decision_row)| {
                for local_x in 0..layer_width as usize {
                    if candidate_mask_row[local_x] == 0 {
                        continue;
                    }
                    let global_x = candidate.left as usize + local_x;
                    let base_pixel_start = global_x * 3;
                    let candidate_pixel_start = local_x * 3;
                    if base_mask_row[global_x] == 0 || decision_row[local_x] > 0 {
                        base_row[base_pixel_start..base_pixel_start + 3].copy_from_slice(
                            &candidate_row[candidate_pixel_start..candidate_pixel_start + 3],
                        );
                    }
                    base_mask_row[global_x] = 255;
                }
            },
        );
}

/// Record the Ownership_Map entries that the *following*
/// [`hard_select_focus_layer`] call is about to create.
///
/// `base_mask` must be the coverage state before that call, because the
/// selection replaces a pixel when the candidate is valid and either the base
/// has no coverage yet or the focus decision prefers the candidate.  The
/// predicate below is that same condition, kept in lockstep with the copy in
/// `hard_select_focus_layer`; nothing here writes a pixel.
fn record_focus_layer_ownership(
    ownership: &mut [u16],
    base_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
    owner_id: u16,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    let (base_width, base_height) = base_mask.dimensions();
    if layer_width == 0
        || layer_height == 0
        || decision_mask.dimensions() != (layer_width, layer_height)
        || candidate.mask.dimensions() != (layer_width, layer_height)
        || ownership.len() != base_width as usize * base_height as usize
        || candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }
    ownership
        .par_chunks_mut(base_width as usize)
        .zip(base_mask.as_raw().par_chunks(base_width as usize))
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(candidate.mask.as_raw().par_chunks(layer_width as usize))
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(
            |(((ownership_row, base_mask_row), candidate_mask_row), decision_row)| {
                let start = candidate.left as usize;
                for local_x in 0..layer_width as usize {
                    let global_x = start + local_x;
                    if candidate_mask_row[local_x] > 0
                        && (base_mask_row[global_x] == 0 || decision_row[local_x] > 0)
                    {
                        ownership_row[global_x] = owner_id;
                    }
                }
            },
        );
}

fn mark_focus_owner_pixels(
    owner: &mut GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
    owner_id: u8,
) {
    let (layer_width, layer_height) = candidate.mask.dimensions();
    if layer_width == 0
        || layer_height == 0
        || decision_mask.dimensions() != (layer_width, layer_height)
        || candidate.left + layer_width > owner.width()
        || candidate.top + layer_height > owner.height()
    {
        return;
    }
    let owner_width = owner.width() as usize;
    owner
        .as_mut()
        .par_chunks_mut(owner_width)
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(candidate.mask.as_raw().par_chunks(layer_width as usize))
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(|((owner_row, candidate_row), decision_row)| {
            let start = candidate.left as usize;
            for local_x in 0..layer_width as usize {
                if candidate_row[local_x] > 0 && decision_row[local_x] > 0 {
                    owner_row[start + local_x] = owner_id;
                }
            }
        });
}

fn scaled_region_start(value: u32, source_size: u32, target_size: u32) -> u32 {
    ((value as u64 * target_size as u64) / source_size.max(1) as u64) as u32
}

fn scaled_region_end(value: u32, source_size: u32, target_size: u32) -> u32 {
    (value as u64 * target_size as u64).div_ceil(source_size.max(1) as u64) as u32
}

fn render_focus_analysis_layer(
    layer: &RenderedFocusLayer,
    out_width: u32,
    out_height: u32,
    analysis_width: u32,
    analysis_height: u32,
) -> RenderedFocusAnalysisLayer {
    let mut analysis = Rgb32FImage::new(analysis_width, analysis_height);
    let mut analysis_mask = GrayImage::new(analysis_width, analysis_height);
    let mut analysis_foreground_mask = GrayImage::new(analysis_width, analysis_height);
    let (layer_width, layer_height) = layer.image.dimensions();
    if layer_width == 0 || layer_height == 0 || out_width == 0 || out_height == 0 {
        return RenderedFocusAnalysisLayer {
            image: analysis,
            mask: analysis_mask,
            foreground_mask: analysis_foreground_mask,
            relaxed_foreground_mask: GrayImage::new(analysis_width, analysis_height),
        };
    }

    let left = scaled_region_start(layer.left, out_width, analysis_width).min(analysis_width);
    let top = scaled_region_start(layer.top, out_height, analysis_height).min(analysis_height);
    let right = scaled_region_end(
        layer.left.saturating_add(layer_width),
        out_width,
        analysis_width,
    )
    .min(analysis_width);
    let bottom = scaled_region_end(
        layer.top.saturating_add(layer_height),
        out_height,
        analysis_height,
    )
    .min(analysis_height);
    if left >= right || top >= bottom {
        return RenderedFocusAnalysisLayer {
            image: analysis,
            mask: analysis_mask,
            foreground_mask: analysis_foreground_mask,
            relaxed_foreground_mask: GrayImage::new(analysis_width, analysis_height),
        };
    }

    let resized_width = right - left;
    let resized_height = bottom - top;
    let resized = resize_rgb(&layer.image, resized_width, resized_height);
    let resized_mask = resize_binary_mask(&layer.mask, resized_width, resized_height);
    let resized_foreground_mask =
        resize_binary_mask(&layer.foreground_mask, resized_width, resized_height);
    let resized_relaxed_foreground_mask = resize_binary_mask(
        &layer.relaxed_foreground_mask,
        resized_width,
        resized_height,
    );
    let analysis_stride = analysis_width as usize * 3;
    let resized_stride = resized_width as usize * 3;
    analysis
        .as_mut()
        .par_chunks_mut(analysis_stride)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(resized.as_raw().par_chunks(resized_stride))
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize * 3;
            let end = start + resized_stride;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    analysis_mask
        .as_mut()
        .par_chunks_mut(analysis_width as usize)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(resized_mask.as_raw().par_chunks(resized_width as usize))
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    analysis_foreground_mask
        .as_mut()
        .par_chunks_mut(analysis_width as usize)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(
            resized_foreground_mask
                .as_raw()
                .par_chunks(resized_width as usize),
        )
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    let analysis_relaxed_stride = analysis_width as usize;
    let mut relaxed_foreground_mask = GrayImage::new(analysis_width, analysis_height);
    relaxed_foreground_mask
        .as_mut()
        .par_chunks_mut(analysis_relaxed_stride)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(
            resized_relaxed_foreground_mask
                .as_raw()
                .par_chunks(resized_width as usize),
        )
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    RenderedFocusAnalysisLayer {
        image: analysis,
        mask: analysis_mask,
        foreground_mask: analysis_foreground_mask,
        relaxed_foreground_mask,
    }
}

/// Upsample an analysis-resolution focus decision onto a layer's pixel grid.
///
/// Superseded on the station path by [`focus_cell_decision_for_layer`], which
/// decides at ownership cell resolution instead (需求 3.6).  Kept because it is
/// the only reader of the analysis decision and the shifted-mosaic guards still
/// maintain that decision; the comparison path needs it back when it is
/// re-enabled.
#[allow(dead_code)]
fn focus_decision_for_layer(
    analysis_decision: &GrayImage,
    layer: &RenderedFocusLayer,
    out_width: u32,
    out_height: u32,
) -> GrayImage {
    let (width, height) = layer.image.dimensions();
    let analysis_width = analysis_decision.width();
    let analysis_height = analysis_decision.height();
    GrayImage::from_fn(width, height, |x, y| {
        if out_width == 0 || out_height == 0 || analysis_width == 0 || analysis_height == 0 {
            return image::Luma([0]);
        }
        let global_x = layer.left.saturating_add(x);
        let global_y = layer.top.saturating_add(y);
        let analysis_x = ((global_x as u64 * analysis_width as u64) / out_width as u64)
            .min(analysis_width.saturating_sub(1) as u64) as u32;
        let analysis_y = ((global_y as u64 * analysis_height as u64) / out_height as u64)
            .min(analysis_height.saturating_sub(1) as u64) as u32;
        *analysis_decision.get_pixel(analysis_x, analysis_y)
    })
}

fn commit_focus_analysis_scores(
    best_focus: &mut [f32],
    candidate_focus: &[f32],
    candidate_analysis_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    full_resolution_decision: &GrayImage,
    out_width: u32,
    out_height: u32,
    analysis_width: u32,
    analysis_height: u32,
) {
    if out_width == 0
        || out_height == 0
        || analysis_width == 0
        || analysis_height == 0
        || candidate_analysis_mask.dimensions() != (analysis_width, analysis_height)
        || candidate_focus.len() != analysis_width as usize * analysis_height as usize
        || best_focus.len() != candidate_focus.len()
        || full_resolution_decision.dimensions() != candidate.image.dimensions()
    {
        return;
    }

    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0
        || layer_height == 0
        || candidate.left >= out_width
        || candidate.top >= out_height
        || candidate.left.saturating_add(layer_width) > out_width
        || candidate.top.saturating_add(layer_height) > out_height
    {
        return;
    }

    // The full-resolution ownership guards can veto part of an analysis cell.
    // Downsample the committed decision, rather than blindly copying the
    // low-resolution decision, so the score table remains tied to pixels that
    // really entered the merged image.
    let left = scaled_region_start(candidate.left, out_width, analysis_width).min(analysis_width);
    let top = scaled_region_start(candidate.top, out_height, analysis_height).min(analysis_height);
    let right = scaled_region_end(
        candidate.left.saturating_add(layer_width),
        out_width,
        analysis_width,
    )
    .min(analysis_width);
    let bottom = scaled_region_end(
        candidate.top.saturating_add(layer_height),
        out_height,
        analysis_height,
    )
    .min(analysis_height);
    if left >= right || top >= bottom {
        return;
    }
    let committed_width = right - left;
    let committed_height = bottom - top;
    let committed = resize_binary_mask(full_resolution_decision, committed_width, committed_height);
    let candidate_mask = candidate_analysis_mask.as_raw();
    for local_y in 0..committed_height as usize {
        for local_x in 0..committed_width as usize {
            let analysis_x = left as usize + local_x;
            let analysis_y = top as usize + local_y;
            let analysis_index = analysis_y * analysis_width as usize + analysis_x;
            if committed.as_raw()[local_y * committed_width as usize + local_x] > 0
                && candidate_mask[analysis_index] > 0
            {
                best_focus[analysis_index] = candidate_focus[analysis_index];
            }
        }
    }
}

fn resize_rgb(image: &Rgb32FImage, width: u32, height: u32) -> Rgb32FImage {
    image::imageops::resize(
        image,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Triangle,
    )
}

pub(super) fn downsample_rgb_half(image: &Rgb32FImage) -> Rgb32FImage {
    let (source_width, source_height) = image.dimensions();
    let target_width = source_width.div_ceil(2).max(1);
    let target_height = source_height.div_ceil(2).max(1);
    if (source_width, source_height) == (target_width, target_height) {
        return image.clone();
    }

    let source = image.as_raw();
    let source_stride = source_width as usize * 3;
    let target_stride = target_width as usize * 3;
    let mut output = vec![0.0f32; target_stride * target_height as usize];
    output
        .par_chunks_mut(target_stride)
        .enumerate()
        .for_each(|(target_y, row)| {
            let source_y0 = (target_y * 2).min(source_height as usize - 1);
            let source_y1 = (source_y0 + 1).min(source_height as usize - 1);
            for target_x in 0..target_width as usize {
                let source_x0 = (target_x * 2).min(source_width as usize - 1);
                let source_x1 = (source_x0 + 1).min(source_width as usize - 1);
                let top_left = source_y0 * source_stride + source_x0 * 3;
                let top_right = source_y0 * source_stride + source_x1 * 3;
                let bottom_left = source_y1 * source_stride + source_x0 * 3;
                let bottom_right = source_y1 * source_stride + source_x1 * 3;
                let output_start = target_x * 3;
                for channel in 0..3 {
                    row[output_start + channel] = (source[top_left + channel]
                        + source[top_right + channel]
                        + source[bottom_left + channel]
                        + source[bottom_right + channel])
                        * 0.25;
                }
            }
        });
    Rgb32FImage::from_raw(target_width, target_height, output)
        .expect("half-resolution RGB buffer dimensions must match")
}

fn downsample_mask_half(mask: &GrayImage) -> GrayImage {
    let (source_width, source_height) = mask.dimensions();
    let target_width = source_width.div_ceil(2).max(1);
    let target_height = source_height.div_ceil(2).max(1);
    if (source_width, source_height) == (target_width, target_height) {
        return mask.clone();
    }

    let source = mask.as_raw();
    let source_stride = source_width as usize;
    let target_stride = target_width as usize;
    let mut output = vec![0u8; target_stride * target_height as usize];
    output
        .par_chunks_mut(target_stride)
        .enumerate()
        .for_each(|(target_y, row)| {
            let source_y0 = (target_y * 2).min(source_height as usize - 1);
            let source_y1 = (source_y0 + 1).min(source_height as usize - 1);
            for (target_x, output) in row.iter_mut().enumerate() {
                let source_x0 = (target_x * 2).min(source_width as usize - 1);
                let source_x1 = (source_x0 + 1).min(source_width as usize - 1);
                let total = u16::from(source[source_y0 * source_stride + source_x0])
                    + u16::from(source[source_y0 * source_stride + source_x1])
                    + u16::from(source[source_y1 * source_stride + source_x0])
                    + u16::from(source[source_y1 * source_stride + source_x1]);
                *output = ((total + 2) / 4) as u8;
            }
        });
    GrayImage::from_raw(target_width, target_height, output)
        .expect("half-resolution mask buffer dimensions must match")
}

#[derive(Clone, Copy)]
struct LinearSample {
    lower: usize,
    upper: usize,
    upper_weight: f32,
}

fn linear_samples(source_length: u32, target_length: u32) -> Vec<LinearSample> {
    let scale = source_length as f64 / target_length.max(1) as f64;
    (0..target_length.max(1))
        .map(|target| {
            let source = ((f64::from(target) + 0.5) * scale - 0.5)
                .clamp(0.0, f64::from(source_length.saturating_sub(1)));
            let lower = source.floor() as usize;
            let upper = (lower + 1).min(source_length.saturating_sub(1) as usize);
            LinearSample {
                lower,
                upper,
                upper_weight: (source - lower as f64) as f32,
            }
        })
        .collect()
}

fn subtract_upsampled_rgb(fine: &Rgb32FImage, coarse: &Rgb32FImage) -> Rgb32FImage {
    let (width, height) = fine.dimensions();
    let x_samples = linear_samples(coarse.width(), width);
    let y_samples = linear_samples(coarse.height(), height);
    let coarse_stride = coarse.width() as usize * 3;
    let output_stride = width as usize * 3;
    let coarse_pixels = coarse.as_raw();
    let fine_pixels = fine.as_raw();
    let mut output = vec![0.0f32; output_stride * height as usize];

    output
        .par_chunks_mut(output_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            let y_weight = y_sample.upper_weight;
            let fine_row = &fine_pixels[y * output_stride..(y + 1) * output_stride];
            for (x, x_sample) in x_samples.iter().copied().enumerate() {
                let x_weight = x_sample.upper_weight;
                let top_left = y_sample.lower * coarse_stride + x_sample.lower * 3;
                let top_right = y_sample.lower * coarse_stride + x_sample.upper * 3;
                let bottom_left = y_sample.upper * coarse_stride + x_sample.lower * 3;
                let bottom_right = y_sample.upper * coarse_stride + x_sample.upper * 3;
                let output_start = x * 3;
                for channel in 0..3 {
                    let top = coarse_pixels[top_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[top_right + channel] * x_weight;
                    let bottom = coarse_pixels[bottom_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[bottom_right + channel] * x_weight;
                    let low_frequency = top * (1.0 - y_weight) + bottom * y_weight;
                    row[output_start + channel] = fine_row[output_start + channel] - low_frequency;
                }
            }
        });

    Rgb32FImage::from_raw(width, height, output)
        .expect("Laplacian detail buffer dimensions must match")
}

fn upsample_and_add_rgb(coarse: &Rgb32FImage, detail: &Rgb32FImage) -> Rgb32FImage {
    let (width, height) = detail.dimensions();
    let x_samples = linear_samples(coarse.width(), width);
    let y_samples = linear_samples(coarse.height(), height);
    let coarse_stride = coarse.width() as usize * 3;
    let output_stride = width as usize * 3;
    let coarse_pixels = coarse.as_raw();
    let detail_pixels = detail.as_raw();
    let mut output = vec![0.0f32; output_stride * height as usize];

    output
        .par_chunks_mut(output_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            let y_weight = y_sample.upper_weight;
            let detail_row = &detail_pixels[y * output_stride..(y + 1) * output_stride];
            for (x, x_sample) in x_samples.iter().copied().enumerate() {
                let x_weight = x_sample.upper_weight;
                let top_left = y_sample.lower * coarse_stride + x_sample.lower * 3;
                let top_right = y_sample.lower * coarse_stride + x_sample.upper * 3;
                let bottom_left = y_sample.upper * coarse_stride + x_sample.lower * 3;
                let bottom_right = y_sample.upper * coarse_stride + x_sample.upper * 3;
                let output_start = x * 3;
                for channel in 0..3 {
                    let top = coarse_pixels[top_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[top_right + channel] * x_weight;
                    let bottom = coarse_pixels[bottom_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[bottom_right + channel] * x_weight;
                    row[output_start + channel] = top * (1.0 - y_weight)
                        + bottom * y_weight
                        + detail_row[output_start + channel];
                }
            }
        });

    Rgb32FImage::from_raw(width, height, output).expect("reconstructed image dimensions must match")
}

fn combine_rgb(
    base: &Rgb32FImage,
    candidate: &Rgb32FImage,
    mask: &GrayImage,
    hard_mask: bool,
) -> Rgb32FImage {
    let (width, height) = base.dimensions();
    let mut output = vec![0.0f32; width as usize * height as usize * 3];
    output
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let mask_value = if hard_mask {
                    if mask.get_pixel(x, y)[0] >= 128 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    mask.get_pixel(x, y)[0] as f32 / 255.0
                };
                let base_pixel = base.get_pixel(x, y);
                let candidate_pixel = candidate.get_pixel(x, y);
                let start = x as usize * 3;
                for channel in 0..3 {
                    row[start + channel] = base_pixel[channel] * (1.0 - mask_value)
                        + candidate_pixel[channel] * mask_value;
                }
            }
        });
    Rgb32FImage::from_raw(width, height, output).expect("combined image dimensions must match")
}

pub(super) fn crop_to_valid_rectangle(image: Rgb32FImage, mask: &GrayImage) -> Rgb32FImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return image;
    }

    let mut heights = vec![0usize; width as usize];
    let mut stack = Vec::with_capacity(width as usize + 1);
    let mut best_area = 0usize;
    let mut best_left = 0usize;
    let mut best_top = 0usize;
    let mut best_width = width as usize;
    let mut best_height = height as usize;

    for y in 0..height as usize {
        for (x, height_value) in heights.iter_mut().enumerate() {
            *height_value = if mask.get_pixel(x as u32, y as u32)[0] > 0 {
                height_value.saturating_add(1)
            } else {
                0
            };
        }

        stack.clear();
        for x in 0..=width as usize {
            let current_height = if x < width as usize { heights[x] } else { 0 };
            while let Some(&bar_index) = stack.last() {
                if heights[bar_index] <= current_height {
                    break;
                }
                stack.pop();
                let left = stack.last().map_or(0, |&index| index + 1);
                let rectangle_width = x - left;
                let rectangle_height = heights[bar_index];
                let area = rectangle_width * rectangle_height;
                if area > best_area {
                    best_area = area;
                    best_left = left;
                    best_top = y + 1 - rectangle_height;
                    best_width = rectangle_width;
                    best_height = rectangle_height;
                }
            }
            stack.push(x);
        }
    }

    if best_area == 0 {
        return image;
    }

    if best_left == 0
        && best_top == 0
        && best_width == width as usize
        && best_height == height as usize
    {
        return image;
    }

    image::imageops::crop_imm(
        &image,
        best_left as u32,
        best_top as u32,
        best_width as u32,
        best_height as u32,
    )
    .to_image()
}

/// Return the mask bounding box and the fraction of that box covered by valid
/// pixels.  A projective tile union can have transparent corner wedges even
/// though its outer bounds are useful; the coverage lets callers choose a
/// source-union crop only when those wedges are negligible.
fn valid_mask_bounds_and_coverage(mask: &GrayImage) -> ((u32, u32, u32, u32), f32) {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return ((0, 0, 0, 0), 0.0);
    }
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut valid = 0u64;
    for y in 0..height {
        for x in 0..width {
            if mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            valid += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if valid == 0 {
        return ((0, 0, 0, 0), 0.0);
    }
    let bbox_width = u64::from(max_x - min_x + 1);
    let bbox_height = u64::from(max_y - min_y + 1);
    let coverage = valid as f32 / (bbox_width * bbox_height).max(1) as f32;
    ((min_x, min_y, max_x, max_y), coverage.clamp(0.0, 1.0))
}

pub(super) fn crop_to_valid_bounds(image: Rgb32FImage, mask: &GrayImage) -> Rgb32FImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return image;
    }
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut covered = false;
    for y in 0..height {
        for x in 0..width {
            if mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            covered = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !covered {
        return image;
    }
    let crop_width = max_x - min_x + 1;
    let crop_height = max_y - min_y + 1;
    if min_x == 0 && min_y == 0 && crop_width == width && crop_height == height {
        return image;
    }
    image::imageops::crop_imm(&image, min_x, min_y, crop_width, crop_height).to_image()
}

fn reflected_run_source_index(
    target: usize,
    limit: usize,
    left: Option<usize>,
    right: Option<usize>,
) -> usize {
    match (left, right) {
        (Some(left), Some(right)) => {
            let distance_from_left = target - left;
            let distance_from_right = right - target;
            if distance_from_left <= distance_from_right {
                left.saturating_sub(distance_from_left)
            } else {
                right
                    .saturating_add(distance_from_right)
                    .min(limit.saturating_sub(1))
            }
        }
        (Some(left), None) => left.saturating_sub(target - left),
        (None, Some(right)) => right
            .saturating_add(right - target)
            .min(limit.saturating_sub(1)),
        (None, None) => target,
    }
}

const FOCUS_FILL_TONE_RADIUS: usize = 64;

fn fill_tone_adjusted_pixel(
    pixel: [f32; 3],
    source_tone: [f32; 3],
    left_tone: [f32; 3],
    right_tone: [f32; 3],
    progress: f32,
) -> [f32; 3] {
    let progress = progress.clamp(0.0, 1.0);
    let blend = progress * progress * (3.0 - 2.0 * progress);
    let mut output = [0.0f32; 3];
    for channel in 0..3 {
        let target_tone = left_tone[channel] * (1.0 - blend) + right_tone[channel] * blend;
        output[channel] = (pixel[channel] + target_tone - source_tone[channel]).clamp(0.0, 1.0);
    }
    output
}

fn horizontal_fill_edge_tone(
    image_row: &[f32],
    edge: usize,
    from_left: bool,
    radius: usize,
) -> [f32; 3] {
    let width = image_row.len() / 3;
    let radius = radius.max(1).min(width);
    let (start, end) = if from_left {
        (edge.saturating_sub(radius - 1), edge)
    } else {
        (edge, (edge + radius - 1).min(width.saturating_sub(1)))
    };
    let mut tone = [0.0f32; 3];
    let sample_count = (end - start + 1) as f32;
    for x in start..=end {
        let pixel_start = x * 3;
        for channel in 0..3 {
            tone[channel] += image_row[pixel_start + channel];
        }
    }
    for channel in &mut tone {
        *channel /= sample_count;
    }
    tone
}

fn vertical_fill_edge_tone(
    image_pixels: &[f32],
    width: usize,
    edge: usize,
    x: usize,
    from_top: bool,
    radius: usize,
) -> [f32; 3] {
    let height = image_pixels.len() / (width * 3);
    let radius = radius.max(1).min(height);
    let (start, end) = if from_top {
        (edge.saturating_sub(radius - 1), edge)
    } else {
        (edge, (edge + radius - 1).min(height.saturating_sub(1)))
    };
    let mut tone = [0.0f32; 3];
    let sample_count = (end - start + 1) as f32;
    for y in start..=end {
        let pixel_start = (y * width + x) * 3;
        for channel in 0..3 {
            tone[channel] += image_pixels[pixel_start + channel];
        }
    }
    for channel in &mut tone {
        *channel /= sample_count;
    }
    tone
}

fn fill_invalid_runs_horizontally(image: &mut Rgb32FImage, mask: &mut GrayImage) -> usize {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return 0;
    }
    let width = width as usize;
    let image_stride = width * 3;
    // Determinism (需求 14.6): the writes are partitioned by destination row and
    // the reduction accumulates `usize`, which is associative, so neither the
    // pixels nor the returned count depend on the thread count. A floating
    // point reduction in this position would need
    // `stack_pipeline::determinism::deterministic_sum`.
    image
        .as_mut()
        .par_chunks_mut(image_stride)
        .zip(mask.as_mut().par_chunks_mut(width))
        .map(|(image_row, mask_row)| {
            let mut filled = 0usize;
            let mut run_start = 0usize;
            while run_start < width {
                if mask_row[run_start] > 0 {
                    run_start += 1;
                    continue;
                }
                let mut run_end = run_start + 1;
                while run_end < width && mask_row[run_end] == 0 {
                    run_end += 1;
                }
                let left = run_start.checked_sub(1).filter(|&x| mask_row[x] > 0);
                let right = (run_end < width && mask_row[run_end] > 0).then_some(run_end);
                if left.is_none() && right.is_none() {
                    run_start = run_end;
                    continue;
                }
                let edge_tones = left.zip(right).map(|(left, right)| {
                    (
                        horizontal_fill_edge_tone(image_row, left, true, FOCUS_FILL_TONE_RADIUS),
                        horizontal_fill_edge_tone(image_row, right, false, FOCUS_FILL_TONE_RADIUS),
                    )
                });
                for target in run_start..run_end {
                    let mut source = reflected_run_source_index(target, width, left, right);
                    if mask_row[source] == 0 {
                        source = left.or(right).expect("a fill run has a valid neighbour");
                    }
                    let source_start = source * 3;
                    let target_start = target * 3;
                    let pixel = [
                        image_row[source_start],
                        image_row[source_start + 1],
                        image_row[source_start + 2],
                    ];
                    let pixel = edge_tones.map_or(pixel, |(left_tone, right_tone)| {
                        let source_tone =
                            if left.is_some_and(|left| target - left <= right.unwrap() - target) {
                                left_tone
                            } else {
                                right_tone
                            };
                        let progress =
                            (target - run_start + 1) as f32 / (run_end - run_start + 1) as f32;
                        fill_tone_adjusted_pixel(
                            pixel,
                            source_tone,
                            left_tone,
                            right_tone,
                            progress,
                        )
                    });
                    image_row[target_start..target_start + 3].copy_from_slice(&pixel);
                    mask_row[target] = 255;
                    filled += 1;
                }
                run_start = run_end;
            }
            filled
        })
        .sum()
}

fn fill_invalid_runs_vertically(image: &mut Rgb32FImage, mask: &mut GrayImage) -> usize {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return 0;
    }
    let width = width as usize;
    let height = height as usize;
    let image_pixels = image.as_mut();
    let mask_pixels = mask.as_mut();
    let mut filled = 0usize;
    for x in 0..width {
        let mut run_start = 0usize;
        while run_start < height {
            if mask_pixels[run_start * width + x] > 0 {
                run_start += 1;
                continue;
            }
            let mut run_end = run_start + 1;
            while run_end < height && mask_pixels[run_end * width + x] == 0 {
                run_end += 1;
            }
            let top = run_start
                .checked_sub(1)
                .filter(|&y| mask_pixels[y * width + x] > 0);
            let bottom =
                (run_end < height && mask_pixels[run_end * width + x] > 0).then_some(run_end);
            if top.is_none() && bottom.is_none() {
                run_start = run_end;
                continue;
            }
            let edge_tones = top.zip(bottom).map(|(top, bottom)| {
                (
                    vertical_fill_edge_tone(
                        image_pixels,
                        width,
                        top,
                        x,
                        true,
                        FOCUS_FILL_TONE_RADIUS,
                    ),
                    vertical_fill_edge_tone(
                        image_pixels,
                        width,
                        bottom,
                        x,
                        false,
                        FOCUS_FILL_TONE_RADIUS,
                    ),
                )
            });
            for target in run_start..run_end {
                let mut source = reflected_run_source_index(target, height, top, bottom);
                if mask_pixels[source * width + x] == 0 {
                    source = top.or(bottom).expect("a fill run has a valid neighbour");
                }
                let source_start = (source * width + x) * 3;
                let target_start = (target * width + x) * 3;
                let pixel = [
                    image_pixels[source_start],
                    image_pixels[source_start + 1],
                    image_pixels[source_start + 2],
                ];
                let pixel = edge_tones.map_or(pixel, |(top_tone, bottom_tone)| {
                    let source_tone =
                        if top.is_some_and(|top| target - top <= bottom.unwrap() - target) {
                            top_tone
                        } else {
                            bottom_tone
                        };
                    let progress =
                        (target - run_start + 1) as f32 / (run_end - run_start + 1) as f32;
                    fill_tone_adjusted_pixel(pixel, source_tone, top_tone, bottom_tone, progress)
                });
                image_pixels[target_start..target_start + 3].copy_from_slice(&pixel);
                mask_pixels[target * width + x] = 255;
                filled += 1;
            }
            run_start = run_end;
        }
    }
    filled
}

fn fill_focus_canvas_margins(image: &mut Rgb32FImage, mask: &mut GrayImage) -> (usize, usize) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return (0, 0);
    }
    let mut filled = 0usize;
    // Reflection extends the local canvas weave and broad tone from the nearest
    // valid edge. Alternating directions also fills the triangular corner areas
    // created by a projective scan without allocating another full-size image.
    for _ in 0..3 {
        filled += fill_invalid_runs_horizontally(image, mask);
        filled += fill_invalid_runs_vertically(image, mask);
        if !mask.as_raw().contains(&0) {
            break;
        }
    }
    let remaining = mask.as_raw().iter().filter(|&&value| value == 0).count();
    (filled, remaining)
}

fn masked_box_blur_rgb(source: &Rgb32FImage, mask: &GrayImage, radius: usize) -> Rgb32FImage {
    let (width, height) = source.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) || radius == 0 {
        return source.clone();
    }

    let width = width as usize;
    let height = height as usize;
    let pixel_count = width * height;
    let window_size = radius.saturating_mul(2).saturating_add(1);
    let source_pixels = source.as_raw();
    let mask_pixels = mask.as_raw();
    let mut horizontal_values = vec![0.0f32; pixel_count * 3];
    let mut horizontal_weights = vec![0u32; pixel_count];

    horizontal_values
        .par_chunks_mut(width * 3)
        .zip(horizontal_weights.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (output_row, weight_row))| {
            let source_row_start = y * width * 3;
            let mask_row_start = y * width;
            let mut sums = [0.0f32; 3];
            let mut weight = 0u32;
            for offset in 0..window_size {
                let x = offset.saturating_sub(radius).min(width - 1);
                if mask_pixels[mask_row_start + x] > 0 {
                    weight += 1;
                    let start = source_row_start + x * 3;
                    for channel in 0..3 {
                        sums[channel] += source_pixels[start + channel];
                    }
                }
            }
            let write_pixel = |x: usize,
                               output_row: &mut [f32],
                               weight_row: &mut [u32],
                               sums: [f32; 3],
                               weight: u32| {
                let start = x * 3;
                if weight > 0 {
                    output_row[start..start + 3].copy_from_slice(&sums);
                }
                weight_row[x] = weight;
            };
            write_pixel(0, output_row, weight_row, sums, weight);
            for x in 1..width {
                let add_x = (x + radius).min(width - 1);
                let remove_x = x.saturating_sub(radius + 1);
                if mask_pixels[mask_row_start + add_x] > 0 {
                    weight += 1;
                    let start = source_row_start + add_x * 3;
                    for channel in 0..3 {
                        sums[channel] += source_pixels[start + channel];
                    }
                }
                if mask_pixels[mask_row_start + remove_x] > 0 {
                    weight = weight.saturating_sub(1);
                    let start = source_row_start + remove_x * 3;
                    for channel in 0..3 {
                        sums[channel] -= source_pixels[start + channel];
                    }
                }
                write_pixel(x, output_row, weight_row, sums, weight);
            }
        });

    let mut output = vec![0.0f32; pixel_count * 3];
    for x in 0..width {
        let mut sums = [0.0f32; 3];
        let mut weight = 0u32;
        for offset in 0..window_size {
            let y = offset.saturating_sub(radius).min(height - 1);
            let index = y * width + x;
            if horizontal_weights[index] > 0 {
                weight += horizontal_weights[index];
                let start = index * 3;
                for channel in 0..3 {
                    sums[channel] += horizontal_values[start + channel];
                }
            }
        }
        let write_pixel = |y: usize, output: &mut [f32], sums: [f32; 3], weight: u32| {
            let index = (y * width + x) * 3;
            if weight > 0 {
                for channel in 0..3 {
                    output[index + channel] = sums[channel] / weight as f32;
                }
            } else {
                let source_index = index;
                output[index..index + 3]
                    .copy_from_slice(&source_pixels[source_index..source_index + 3]);
            }
        };
        write_pixel(0, &mut output, sums, weight);
        for y in 1..height {
            let add_y = (y + radius).min(height - 1);
            let remove_y = y.saturating_sub(radius + 1);
            let add_index = add_y * width + x;
            let remove_index = remove_y * width + x;
            if horizontal_weights[add_index] > 0 {
                weight += horizontal_weights[add_index];
                let start = add_index * 3;
                for channel in 0..3 {
                    sums[channel] += horizontal_values[start + channel];
                }
            }
            if horizontal_weights[remove_index] > 0 {
                weight = weight.saturating_sub(horizontal_weights[remove_index]);
                let start = remove_index * 3;
                for channel in 0..3 {
                    sums[channel] -= horizontal_values[start + channel];
                }
            }
            write_pixel(y, &mut output, sums, weight);
        }
    }

    Rgb32FImage::from_raw(width as u32, height as u32, output)
        .expect("masked RGB blur dimensions must match")
}

fn masked_box_blur_focus_map(
    source: &[f32],
    mask: &[u8],
    width: u32,
    height: u32,
    radius: usize,
) -> Vec<f32> {
    let width = width as usize;
    let height = height as usize;
    if width == 0 || height == 0 || source.len() != width * height || mask.len() != source.len() {
        return source.to_vec();
    }
    if radius == 0 {
        return source.to_vec();
    }

    let window_size = radius.saturating_mul(2).saturating_add(1);
    let mut horizontal_values = vec![0.0f32; source.len()];
    let mut horizontal_weights = vec![0u32; source.len()];
    horizontal_values
        .par_chunks_mut(width)
        .zip(horizontal_weights.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (output_row, weight_row))| {
            let row_start = y * width;
            let mut sum = 0.0f32;
            let mut weight = 0u32;
            for offset in 0..window_size {
                let x = offset.saturating_sub(radius).min(width - 1);
                if mask[row_start + x] > 0 {
                    sum += source[row_start + x];
                    weight += 1;
                }
            }
            output_row[0] = sum;
            weight_row[0] = weight;
            for x in 1..width {
                let add_x = (x + radius).min(width - 1);
                let remove_x = x.saturating_sub(radius + 1);
                if mask[row_start + add_x] > 0 {
                    sum += source[row_start + add_x];
                    weight += 1;
                }
                if mask[row_start + remove_x] > 0 {
                    sum -= source[row_start + remove_x];
                    weight = weight.saturating_sub(1);
                }
                output_row[x] = sum;
                weight_row[x] = weight;
            }
        });

    let mut output = vec![0.0f32; source.len()];
    for x in 0..width {
        let mut sum = 0.0f32;
        let mut weight = 0u32;
        for offset in 0..window_size {
            let y = offset.saturating_sub(radius).min(height - 1);
            let index = y * width + x;
            sum += horizontal_values[index];
            weight += horizontal_weights[index];
        }
        let first_index = x;
        output[first_index] = if weight > 0 {
            sum / weight as f32
        } else {
            source[first_index]
        };
        for y in 1..height {
            let add_y = (y + radius).min(height - 1);
            let remove_y = y.saturating_sub(radius + 1);
            let add_index = add_y * width + x;
            let remove_index = remove_y * width + x;
            sum += horizontal_values[add_index] - horizontal_values[remove_index];
            weight += horizontal_weights[add_index];
            weight = weight.saturating_sub(horizontal_weights[remove_index]);
            let index = y * width + x;
            output[index] = if weight > 0 {
                sum / weight as f32
            } else {
                source[index]
            };
        }
    }
    output
}

#[cfg(test)]
fn harmonize_focus_background_tone(
    image: &mut Rgb32FImage,
    image_mask: &GrayImage,
    foreground_mask: &GrayImage,
) {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || image_mask.dimensions() != (width, height)
        || foreground_mask.dimensions() != (width, height)
    {
        return;
    }

    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(width, height, width.max(height));
    let analysis_image = resize_rgb(image, analysis_width, analysis_height);
    let analysis_mask = resize_binary_mask(image_mask, analysis_width, analysis_height);
    // Do not derive the correction domain from the focus ownership mask. That
    // mask is made from sharpness decisions and may contain source-shaped
    // islands on the canvas. Instead, sample only pixels that still look like
    // the warm-brown paper itself; this makes an exposure block eligible for
    // correction while keeping painted colours out of the estimator.
    let analysis_background_like = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let covered = analysis_mask.get_pixel(x, y)[0] > 0;
        let index = y as usize * analysis_width as usize + x as usize;
        let pixel_start = index * 3;
        let pixel = &analysis_image.as_raw()[pixel_start..pixel_start + 3];
        let canvas_like = focus_stack_pixel_is_canvas_like(pixel);
        let tone_foreground = focus_stack_pixel_is_tone_foreground(pixel);
        image::Luma([u8::from(covered && canvas_like && !tone_foreground) * 255])
    });
    let foreground_soft_radius = (analysis_width.max(analysis_height) as f32 * 0.005)
        .round()
        .clamp(2.0, 12.0) as usize;
    let analysis_background_values = analysis_background_like
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let analysis_background_weight = box_blur_focus_map(
        &analysis_background_values,
        analysis_width,
        analysis_height,
        foreground_soft_radius,
    )
    .into_iter()
    .map(|value| value.clamp(0.0, 1.0))
    .collect::<Vec<_>>();
    let analysis_background = analysis_background_like.clone();
    let background_samples = analysis_background
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if background_samples < 128 {
        return;
    }

    let background_pixels = analysis_background.as_raw();
    let mut global_channels = [Vec::new(), Vec::new(), Vec::new()];
    for (index, &background) in background_pixels.iter().enumerate() {
        if background == 0 {
            continue;
        }
        let start = index * 3;
        for channel in 0..3 {
            global_channels[channel].push(analysis_image.as_raw()[start + channel]);
        }
    }
    let global_tone = [
        median_f32(&mut global_channels[0]).unwrap_or(0.0),
        median_f32(&mut global_channels[1]).unwrap_or(0.0),
        median_f32(&mut global_channels[2]).unwrap_or(0.0),
    ];
    let analysis_background_like = analysis_background.clone();
    let background_like_pixels = analysis_background_like
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    println!(
        "  - Background tone sample coverage: {background_like_pixels}/{background_samples} analysis pixels"
    );
    if background_like_pixels < 128 {
        return;
    }

    // The first blur removes weave-scale variation from the estimate. The
    // second, much wider blur provides a continuous target across source-sized
    // exposure blocks. Their difference is a low-frequency correction, so the
    // original high-frequency canvas texture is retained pixel-for-pixel.
    let local_tone = masked_box_blur_rgb(
        &analysis_image,
        &analysis_background_like,
        FOCUS_BACKGROUND_TONE_LOCAL_RADIUS,
    );
    let smooth_tone = masked_box_blur_rgb(
        &local_tone,
        &analysis_background_like,
        FOCUS_BACKGROUND_TONE_SMOOTH_RADIUS,
    );
    let local_pixels = local_tone.as_raw();
    let smooth_pixels = smooth_tone.as_raw();
    let mut correction = vec![[0.0f32; 3]; analysis_width as usize * analysis_height as usize];
    correction
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, output)| {
            if background_like_pixels == 0 || analysis_background_like.as_raw()[index] == 0 {
                return;
            }
            let start = index * 3;
            for channel in 0..3 {
                let target = smooth_pixels[start + channel]
                    * (1.0 - FOCUS_BACKGROUND_TONE_GLOBAL_WEIGHT)
                    + global_tone[channel] * FOCUS_BACKGROUND_TONE_GLOBAL_WEIGHT;
                output[channel] = (target - local_pixels[start + channel]).clamp(
                    -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                    FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                );
            }
        });

    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize;
    let background_weight_ref = &analysis_background_weight;
    let correction_ref = &correction;
    let full_mask = image_mask.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            for x in 0..width as usize {
                let full_index = y * width as usize + x;
                if full_mask[full_index] == 0 {
                    continue;
                }
                let start = x * 3;
                if !focus_stack_pixel_is_canvas_like(&row[start..start + 3])
                    || focus_stack_pixel_is_tone_foreground(&row[start..start + 3])
                {
                    continue;
                }
                let x_sample = x_samples[x];
                let top_left_weight =
                    background_weight_ref[y_sample.lower * analysis_stride + x_sample.lower];
                let top_right_weight =
                    background_weight_ref[y_sample.lower * analysis_stride + x_sample.upper];
                let bottom_left_weight =
                    background_weight_ref[y_sample.upper * analysis_stride + x_sample.lower];
                let bottom_right_weight =
                    background_weight_ref[y_sample.upper * analysis_stride + x_sample.upper];
                let top_weight = top_left_weight * (1.0 - x_sample.upper_weight)
                    + top_right_weight * x_sample.upper_weight;
                let bottom_weight = bottom_left_weight * (1.0 - x_sample.upper_weight)
                    + bottom_right_weight * x_sample.upper_weight;
                let background_weight = (top_weight * (1.0 - y_sample.upper_weight)
                    + bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                if background_weight <= 0.001 {
                    continue;
                }
                let top_left = correction_ref[y_sample.lower * analysis_stride + x_sample.lower];
                let top_right = correction_ref[y_sample.lower * analysis_stride + x_sample.upper];
                let bottom_left = correction_ref[y_sample.upper * analysis_stride + x_sample.lower];
                let bottom_right =
                    correction_ref[y_sample.upper * analysis_stride + x_sample.upper];
                let mut delta = [0.0f32; 3];
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_sample.upper_weight)
                        + top_right[channel] * x_sample.upper_weight;
                    let bottom = bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + bottom_right[channel] * x_sample.upper_weight;
                    delta[channel] = (top * (1.0 - y_sample.upper_weight)
                        + bottom * y_sample.upper_weight)
                        * background_weight;
                }
                for channel in 0..3 {
                    row[start + channel] = (row[start + channel] + delta[channel]).clamp(0.0, 1.0);
                }
            }
        });
}

fn harmonize_focus_background_tone_with_owners(
    image: &mut Rgb32FImage,
    image_mask: &GrayImage,
    foreground_mask: &GrayImage,
    owner_map: &GrayImage,
) {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || image_mask.dimensions() != (width, height)
        || foreground_mask.dimensions() != (width, height)
        || owner_map.dimensions() != (width, height)
    {
        return;
    }

    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(width, height, width.max(height));
    let analysis_image = resize_rgb(image, analysis_width, analysis_height);
    let analysis_mask = resize_binary_mask(image_mask, analysis_width, analysis_height);
    let analysis_foreground = resize_binary_mask(foreground_mask, analysis_width, analysis_height);
    let foreground_envelope = build_focus_background_foreground_envelope(
        &analysis_image,
        &analysis_mask,
        &analysis_foreground,
    );
    let background = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let index = y as usize * analysis_width as usize + x as usize;
        let start = index * 3;
        let pixel = &analysis_image.as_raw()[start..start + 3];
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0
                && foreground_envelope.as_raw()[index] == 0
                && focus_stack_pixel_is_canvas_like(pixel)
                && !focus_stack_pixel_is_tone_foreground(pixel),
        ) * 255])
    });
    let background_pixels = background
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if background_pixels < 128 {
        return;
    }

    let owner_analysis = image::imageops::resize(
        owner_map,
        analysis_width,
        analysis_height,
        image::imageops::FilterType::Nearest,
    );
    let sample_step = ((u64::from(analysis_width) * u64::from(analysis_height) / 120_000) as f64)
        .sqrt()
        .ceil()
        .max(1.0) as usize;
    let mut global_channels = [Vec::new(), Vec::new(), Vec::new()];
    let mut owner_channels: Vec<[Vec<f32>; 3]> = (0..=u8::MAX as usize)
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    for y in (0..analysis_height as usize).step_by(sample_step) {
        for x in (0..analysis_width as usize).step_by(sample_step) {
            let index = y * analysis_width as usize + x;
            if background.as_raw()[index] == 0 {
                continue;
            }
            let start = index * 3;
            let owner = owner_analysis.as_raw()[index] as usize;
            for channel in 0..3 {
                let value = analysis_image.as_raw()[start + channel];
                global_channels[channel].push(value);
                if owner > 0 {
                    owner_channels[owner][channel].push(value);
                }
            }
        }
    }
    let global_tone = [
        median_f32(&mut global_channels[0]).unwrap_or(0.0),
        median_f32(&mut global_channels[1]).unwrap_or(0.0),
        median_f32(&mut global_channels[2]).unwrap_or(0.0),
    ];
    let mut owner_corrections = vec![[0.0f32; 3]; u8::MAX as usize + 1];
    let mut corrected_owner_count = 0usize;
    let mut rejected_owner_count = 0usize;
    for (owner, channels) in owner_channels.iter_mut().enumerate().skip(1) {
        if channels[0].len() < FOCUS_COLOR_MIN_SAMPLES {
            continue;
        }
        // A source owner may cover a different painted region from the global
        // median. Its channel median is then content, not exposure. Require a
        // low robust spread before treating the owner as a tone reference.
        let mut owner_medians = [0.0f32; 3];
        let mut inconsistent = false;
        for channel in 0..3 {
            if channels[channel].len() < FOCUS_COLOR_MIN_SAMPLES {
                inconsistent = true;
                break;
            }
            owner_medians[channel] =
                median_f32(&mut channels[channel].clone()).unwrap_or(global_tone[channel]);
            let mut deviations = channels[channel]
                .iter()
                .map(|value| (*value - owner_medians[channel]).abs())
                .collect::<Vec<_>>();
            let mad = median_f32(&mut deviations).unwrap_or(0.0);
            if !mad.is_finite() || mad > FOCUS_OWNER_TONE_MAX_MAD {
                inconsistent = true;
                break;
            }
        }
        if inconsistent {
            rejected_owner_count += 1;
            continue;
        }
        let mut has_adjustment = false;
        for channel in 0..3 {
            let correction = (global_tone[channel] - owner_medians[channel]).clamp(
                -FOCUS_OWNER_TONE_MAX_ADJUSTMENT,
                FOCUS_OWNER_TONE_MAX_ADJUSTMENT,
            );
            owner_corrections[owner][channel] = correction;
            has_adjustment |= correction.abs() >= 0.002;
        }
        if has_adjustment {
            corrected_owner_count += 1;
        }
    }
    if std::env::var_os("RAW_EDITOR_FOCUS_TONE_DIAGNOSTICS").is_some() {
        let summary = owner_corrections
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, correction)| correction.iter().any(|value| value.abs() >= 0.001))
            .map(|(owner, correction)| {
                format!(
                    "{}:[{:.3},{:.3},{:.3}]",
                    owner, correction[0], correction[1], correction[2]
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!("  - Focus owner tone corrections: {summary}");
    }
    // The owner median is the exposure estimate. Do not derive a second
    // correction from the image's own low-frequency content: on artwork that
    // content is real paper ageing and brush tone, so pulling it toward one
    // global median creates the foggy, low-contrast result this pass is meant
    // to prevent. Extend each measured owner correction over that owner's
    // complete image and blur only the owner boundary. This preserves local
    // contrast while still removing a source-sized exposure step.
    let mut correction_pixels =
        vec![0.0f32; analysis_width as usize * analysis_height as usize * 3];
    for index in 0..background.as_raw().len() {
        if analysis_mask.as_raw()[index] == 0 {
            continue;
        }
        let start = index * 3;
        let owner_correction = owner_corrections[owner_analysis.as_raw()[index] as usize];
        for channel in 0..3 {
            correction_pixels[start + channel] = owner_correction[channel];
        }
    }
    let correction_image =
        Rgb32FImage::from_raw(analysis_width, analysis_height, correction_pixels)
            .expect("focus owner correction dimensions must match");
    let correction_radius = (analysis_width.max(analysis_height) as f32 * 0.006)
        .round()
        .clamp(4.0, 16.0) as usize;
    let smoothed_correction =
        masked_box_blur_rgb(&correction_image, &analysis_mask, correction_radius);
    let background_values = analysis_mask
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let background_radius = (analysis_width.max(analysis_height) as f32 * 0.006)
        .round()
        .clamp(4.0, 16.0) as usize;
    let background_weight = box_blur_focus_map(
        &background_values,
        analysis_width,
        analysis_height,
        background_radius,
    );
    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize;
    let correction_ref = smoothed_correction.as_raw();
    let background_weight_ref = &background_weight;
    let image_mask_ref = image_mask.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            for x in 0..width as usize {
                if image_mask_ref[y * width as usize + x] == 0 {
                    continue;
                }
                let start = x * 3;
                if focus_stack_pixel_is_tone_foreground(&row[start..start + 3]) {
                    continue;
                }
                let x_sample = x_samples[x];
                let top_weight = background_weight_ref
                    [y_sample.lower * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + background_weight_ref[y_sample.lower * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let bottom_weight = background_weight_ref
                    [y_sample.upper * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + background_weight_ref[y_sample.upper * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let background_weight = (top_weight * (1.0 - y_sample.upper_weight)
                    + bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                if background_weight <= 0.001 {
                    continue;
                }
                let top_left = &correction_ref[(y_sample.lower * analysis_stride + x_sample.lower)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.lower) * 3 + 3];
                let top_right = &correction_ref[(y_sample.lower * analysis_stride + x_sample.upper)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.upper) * 3 + 3];
                let bottom_left =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.lower) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.lower) * 3 + 3];
                let bottom_right =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.upper) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.upper) * 3 + 3];
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_sample.upper_weight)
                        + top_right[channel] * x_sample.upper_weight;
                    let bottom = bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + bottom_right[channel] * x_sample.upper_weight;
                    row[start + channel] = (row[start + channel]
                        + (top * (1.0 - y_sample.upper_weight) + bottom * y_sample.upper_weight)
                            * background_weight)
                        .clamp(0.0, 1.0);
                }
            }
        });
    println!(
        "  - Owner-based background harmonization: samples={} owners_adjusted={} owners_rejected={} radius={}px",
        background_pixels, corrected_owner_count, rejected_owner_count, correction_radius
    );
}

fn harmonize_focus_background_with_reference(
    image: &mut Rgb32FImage,
    image_mask: &GrayImage,
    foreground_mask: &GrayImage,
    reference_image: &Rgb32FImage,
    reference_mask: &GrayImage,
    owner_map: &GrayImage,
    owner_corrections: &[[f32; 3]],
) {
    let (width, height) = image.dimensions();
    let (analysis_width, analysis_height) = reference_image.dimensions();
    if width == 0
        || height == 0
        || analysis_width == 0
        || analysis_height == 0
        || image_mask.dimensions() != (width, height)
        || foreground_mask.dimensions() != (width, height)
        || reference_mask.dimensions() != (analysis_width, analysis_height)
        || owner_map.dimensions() != (width, height)
        || owner_corrections.is_empty()
    {
        return;
    }

    let analysis_image = resize_rgb(image, analysis_width, analysis_height);
    let analysis_mask = resize_binary_mask(image_mask, analysis_width, analysis_height);
    let analysis_foreground = resize_binary_mask(foreground_mask, analysis_width, analysis_height);
    let foreground_envelope = build_focus_background_foreground_envelope(
        &analysis_image,
        &analysis_mask,
        &analysis_foreground,
    );
    let background = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let index = y as usize * analysis_width as usize + x as usize;
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0
                && reference_mask.as_raw()[index] > 0
                && foreground_envelope.as_raw()[index] == 0,
        ) * 255])
    });
    let background_pixels = background
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if background_pixels < 128 {
        return;
    }

    let all_background = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let index = y as usize * analysis_width as usize + x as usize;
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0 && foreground_envelope.as_raw()[index] == 0,
        ) * 255])
    });
    let owner_analysis = image::imageops::resize(
        owner_map,
        analysis_width,
        analysis_height,
        image::imageops::FilterType::Nearest,
    );
    let owner_correction_pixels = all_background
        .as_raw()
        .iter()
        .enumerate()
        .flat_map(|(index, &value)| {
            if value == 0 {
                return [0.0; 3];
            }
            owner_corrections
                .get(owner_analysis.as_raw()[index] as usize)
                .copied()
                .unwrap_or([0.0; 3])
        })
        .collect::<Vec<_>>();
    let owner_correction_image =
        Rgb32FImage::from_raw(analysis_width, analysis_height, owner_correction_pixels)
            .expect("focus owner correction dimensions must match");
    let owner_correction_radius = (analysis_width.max(analysis_height) as f32 * 0.06)
        .round()
        .clamp(24.0, 96.0) as usize;
    let smoothed_owner_correction = masked_box_blur_rgb(
        &owner_correction_image,
        &all_background,
        owner_correction_radius,
    );

    // The reference is an average of every source's canvas-only pixels after a
    // robust per-frame offset. Compare it to the selected focus image only at a
    // broad scale. The final image keeps its own high-frequency samples, so the
    // operation cannot soften a face, sleeve, brush line, or the canvas weave.
    let focus_low = masked_box_blur_rgb(
        &analysis_image,
        &background,
        FOCUS_BACKGROUND_TONE_LOCAL_RADIUS,
    );
    let reference_low = masked_box_blur_rgb(
        reference_image,
        &background,
        FOCUS_BACKGROUND_TONE_LOCAL_RADIUS,
    );
    let focus_pixels = focus_low.as_raw();
    let reference_pixels = reference_low.as_raw();
    let mut correction_pixels =
        vec![0.0f32; analysis_width as usize * analysis_height as usize * 3];
    for index in 0..background.as_raw().len() {
        if background.as_raw()[index] == 0 {
            continue;
        }
        let start = index * 3;
        for channel in 0..3 {
            correction_pixels[start + channel] =
                (reference_pixels[start + channel] - focus_pixels[start + channel]).clamp(
                    -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                    FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                );
        }
    }
    let correction_image =
        Rgb32FImage::from_raw(analysis_width, analysis_height, correction_pixels)
            .expect("focus reference correction dimensions must match");
    let correction_radius = (analysis_width.max(analysis_height) as f32 * 0.06)
        .round()
        .clamp(24.0, 96.0) as usize;
    let smoothed_correction =
        masked_box_blur_rgb(&correction_image, &background, correction_radius);
    let reference_background_values = background
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let all_background_values = all_background
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let background_radius = (analysis_width.max(analysis_height) as f32 * 0.006)
        .round()
        .clamp(4.0, 16.0) as usize;
    let reference_background_weight = box_blur_focus_map(
        &reference_background_values,
        analysis_width,
        analysis_height,
        background_radius,
    );
    let all_background_weight = box_blur_focus_map(
        &all_background_values,
        analysis_width,
        analysis_height,
        background_radius,
    );
    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize;
    let correction_ref = smoothed_correction.as_raw();
    let owner_correction_ref = smoothed_owner_correction.as_raw();
    let reference_background_weight_ref = &reference_background_weight;
    let all_background_weight_ref = &all_background_weight;
    let image_mask_ref = image_mask.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            for x in 0..width as usize {
                if image_mask_ref[y * width as usize + x] == 0 {
                    continue;
                }
                let x_sample = x_samples[x];
                let top_weight = all_background_weight_ref
                    [y_sample.lower * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + all_background_weight_ref[y_sample.lower * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let bottom_weight = all_background_weight_ref
                    [y_sample.upper * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + all_background_weight_ref[y_sample.upper * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let all_background_weight = (top_weight * (1.0 - y_sample.upper_weight)
                    + bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                if all_background_weight <= 0.001 {
                    continue;
                }
                let reference_top_weight = reference_background_weight_ref
                    [y_sample.lower * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + reference_background_weight_ref
                        [y_sample.lower * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let reference_bottom_weight = reference_background_weight_ref
                    [y_sample.upper * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + reference_background_weight_ref
                        [y_sample.upper * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let reference_background_weight = (reference_top_weight
                    * (1.0 - y_sample.upper_weight)
                    + reference_bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                let top_left = &correction_ref[(y_sample.lower * analysis_stride + x_sample.lower)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.lower) * 3 + 3];
                let top_right = &correction_ref[(y_sample.lower * analysis_stride + x_sample.upper)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.upper) * 3 + 3];
                let bottom_left =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.lower) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.lower) * 3 + 3];
                let bottom_right =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.upper) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.upper) * 3 + 3];
                let start = x * 3;
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_sample.upper_weight)
                        + top_right[channel] * x_sample.upper_weight;
                    let bottom = bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + bottom_right[channel] * x_sample.upper_weight;
                    let delta = (top * (1.0 - y_sample.upper_weight)
                        + bottom * y_sample.upper_weight)
                        * reference_background_weight;
                    let owner_top_left = &owner_correction_ref[(y_sample.lower * analysis_stride
                        + x_sample.lower)
                        * 3
                        ..(y_sample.lower * analysis_stride + x_sample.lower) * 3 + 3];
                    let owner_top_right = &owner_correction_ref[(y_sample.lower * analysis_stride
                        + x_sample.upper)
                        * 3
                        ..(y_sample.lower * analysis_stride + x_sample.upper) * 3 + 3];
                    let owner_bottom_left = &owner_correction_ref[(y_sample.upper * analysis_stride
                        + x_sample.lower)
                        * 3
                        ..(y_sample.upper * analysis_stride + x_sample.lower) * 3 + 3];
                    let owner_bottom_right =
                        &owner_correction_ref[(y_sample.upper * analysis_stride + x_sample.upper)
                            * 3
                            ..(y_sample.upper * analysis_stride + x_sample.upper) * 3 + 3];
                    let owner_top = owner_top_left[channel] * (1.0 - x_sample.upper_weight)
                        + owner_top_right[channel] * x_sample.upper_weight;
                    let owner_bottom = owner_bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + owner_bottom_right[channel] * x_sample.upper_weight;
                    let owner_delta = (owner_top * (1.0 - y_sample.upper_weight)
                        + owner_bottom * y_sample.upper_weight)
                        * all_background_weight;
                    row[start + channel] =
                        (row[start + channel] + owner_delta + delta).clamp(0.0, 1.0);
                }
            }
        });
    println!(
        "  - Reference background harmonization: samples={} radius={}px owner_radius={}px",
        background_pixels, correction_radius, owner_correction_radius
    );
}

fn multiband_blend(
    base: Rgb32FImage,
    candidate: Rgb32FImage,
    mask: GrayImage,
    low_frequency_mask: Option<GrayImage>,
    max_bands: usize,
    hard_finest_band: bool,
) -> Rgb32FImage {
    let (width, height) = base.dimensions();
    let mut current_base = base;
    let mut current_candidate = candidate;
    let mut current_mask = mask;
    let mut current_low_frequency_mask = low_frequency_mask.unwrap_or_else(|| current_mask.clone());
    let mut base_laplacian = Vec::new();
    let mut candidate_laplacian = Vec::new();
    let mut masks = Vec::new();
    let mut low_frequency_masks = Vec::new();

    while base_laplacian.len() + 1 < max_bands {
        // Continue far enough for the coarsest band to absorb broad illumination and
        // vignetting differences. Stopping at 32px left low-frequency exposure steps
        // visible even though the high-frequency seam itself was well placed.
        if current_base.width() <= 4 || current_base.height() <= 4 {
            break;
        }
        let next_base = downsample_rgb_half(&current_base);
        let next_candidate = downsample_rgb_half(&current_candidate);
        let next_mask = downsample_mask_half(&current_mask);
        let next_low_frequency_mask = downsample_mask_half(&current_low_frequency_mask);
        base_laplacian.push(subtract_upsampled_rgb(&current_base, &next_base));
        candidate_laplacian.push(subtract_upsampled_rgb(&current_candidate, &next_candidate));
        masks.push(current_mask);
        low_frequency_masks.push(current_low_frequency_mask);
        current_base = next_base;
        current_candidate = next_candidate;
        current_mask = next_mask;
        current_low_frequency_mask = next_low_frequency_mask;
    }

    let mut reconstructed = combine_rgb(
        &current_base,
        &current_candidate,
        &current_low_frequency_mask,
        false,
    );
    for level in (0..base_laplacian.len()).rev() {
        let blend_mask = if level >= PANORAMA_GLOBAL_TONE_FIRST_BAND {
            &low_frequency_masks[level]
        } else {
            &masks[level]
        };
        let blended_detail = combine_rgb(
            &base_laplacian[level],
            &candidate_laplacian[level],
            blend_mask,
            // Focus ownership is a source-selection problem. Once a seam has
            // been chosen, averaging middle-frequency detail across it can
            // create a second contour when the two registered samples differ
            // by even a fraction of a pixel. Keep all visible detail bands on
            // one side; only the broad tone bands may feather continuously.
            hard_finest_band && level < PANORAMA_GLOBAL_TONE_FIRST_BAND,
        );
        reconstructed = upsample_and_add_rgb(&reconstructed, &blended_detail);
    }
    if reconstructed.dimensions() == (width, height) {
        reconstructed
    } else {
        resize_rgb(&reconstructed, width, height)
    }
}

fn blend_focus_seam_band(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width < 3 || layer_height < 3 {
        return;
    }
    debug_assert_eq!(candidate.mask.dimensions(), (layer_width, layer_height));
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));

    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    // A seam may cross a detected depth/occlusion layer, but that layer must
    // never be reconstructed by averaging two different source positions. The
    // mask is optional and content-derived; for ordinary flat stacks this is an
    // all-zero buffer and the generic seam path is unchanged.
    let foreground_overlap = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        candidate.left,
        candidate.top,
        (layer_width, layer_height),
    );
    let protected_foreground = candidate
        .foreground_mask
        .as_raw()
        .iter()
        .zip(foreground_overlap.iter())
        .zip(candidate.relaxed_foreground_mask.as_raw().iter())
        .map(
            |((candidate_foreground, existing_foreground), relaxed_foreground)| {
                u8::from(
                    (*candidate_foreground > 0 || *existing_foreground > 0)
                        && *relaxed_foreground == 0,
                )
            },
        )
        .collect::<Vec<_>>();
    let hard_foreground_edge_radius = ((layer_width.max(layer_height) as f32 / 2_400.0)
        * FOCUS_FOREGROUND_HARD_EDGE_RADIUS_AT_2400)
        .round()
        .clamp(3.0, 24.0) as usize;
    let protected_foreground_edges = focus_foreground_edge_mask(
        &protected_foreground,
        layer_width,
        layer_height,
        hard_foreground_edge_radius,
    );
    let is_protected = |x: u32, y: u32| -> bool {
        protected_foreground_edges[y as usize * layer_width as usize + x as usize] > 0
    };

    let owns_candidate = |x: u32, y: u32| -> bool {
        if x >= layer_width || y >= layer_height || candidate.mask.get_pixel(x, y)[0] == 0 {
            return false;
        }
        let global_x = candidate.left + x;
        let global_y = candidate.top + y;
        base_mask.get_pixel(global_x, global_y)[0] == 0 || decision_mask.get_pixel(x, y)[0] > 0
    };

    let mut seam_left = layer_width;
    let mut seam_right = 0u32;
    let mut seam_top = layer_height;
    let mut seam_bottom = 0u32;
    for y in 1..layer_height.saturating_sub(1) {
        for x in 1..layer_width.saturating_sub(1) {
            // Do not let a transition on a foreground silhouette expand the
            // seam's bounding box over the whole object. Its ownership is
            // restored pixel-for-pixel after the paper seam is blended.
            if is_protected(x, y) {
                continue;
            }
            let ownership = owns_candidate(x, y);
            let has_transition = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                .into_iter()
                .any(|(neighbor_x, neighbor_y)| {
                    !is_protected(neighbor_x, neighbor_y)
                        && owns_candidate(neighbor_x, neighbor_y) != ownership
                });
            if has_transition {
                seam_left = seam_left.min(x);
                seam_right = seam_right.max(x);
                seam_top = seam_top.min(y);
                seam_bottom = seam_bottom.max(y);
            }
        }
    }
    if seam_left > seam_right || seam_top > seam_bottom {
        hard_select_focus_layer(base, base_mask, candidate, decision_mask);
        return;
    }

    let radius = FOCUS_SEAM_BLEND_RADIUS.min(layer_width.max(layer_height) as usize);
    // The focus seam is used for detail ownership, but the exposure step can
    // extend across the complete shared canvas area. Build the low-frequency
    // pyramid over that overlap, not only over a rectangle around the sharpness
    // transition; otherwise a source-sized canvas block outside the seam box is
    // copied unchanged into the final stack.
    let mut overlap_left = layer_width;
    let mut overlap_right = 0u32;
    let mut overlap_top = layer_height;
    let mut overlap_bottom = 0u32;
    for y in 0..layer_height {
        for x in 0..layer_width {
            if candidate.mask.get_pixel(x, y)[0] == 0
                || base_mask.get_pixel(candidate.left + x, candidate.top + y)[0] == 0
            {
                continue;
            }
            overlap_left = overlap_left.min(x);
            overlap_right = overlap_right.max(x);
            overlap_top = overlap_top.min(y);
            overlap_bottom = overlap_bottom.max(y);
        }
    }
    let patch_left = if overlap_left <= overlap_right {
        overlap_left
    } else {
        seam_left.saturating_sub(radius as u32)
    };
    let patch_right = if overlap_left <= overlap_right {
        overlap_right
    } else {
        (seam_right + radius as u32).min(layer_width - 1)
    };
    let patch_top = if overlap_top <= overlap_bottom {
        overlap_top
    } else {
        seam_top.saturating_sub(radius as u32)
    };
    let patch_bottom = if overlap_top <= overlap_bottom {
        overlap_bottom
    } else {
        (seam_bottom + radius as u32).min(layer_height - 1)
    };
    let patch_width = patch_right - patch_left + 1;
    let patch_height = patch_bottom - patch_top + 1;
    if patch_width < 8 || patch_height < 8 {
        hard_select_focus_layer(base, base_mask, candidate, decision_mask);
        return;
    }

    let patch_pixel_count = patch_width as usize * patch_height as usize;
    let patch_rgb_stride = patch_width as usize * 3;
    let patch_mask_stride = patch_width as usize;
    let mut base_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut candidate_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut blend_values = vec![0.0f32; patch_pixel_count];
    let mut tone_blend_values = vec![0u8; patch_pixel_count];
    let mut tone_sample_values = vec![0u8; patch_pixel_count];
    let mut protected_values = vec![0u8; patch_pixel_count];
    let mut protected_pixels = vec![0.0f32; patch_pixel_count * 3];
    let base_ref: &Rgb32FImage = base;
    let base_mask_ref: &GrayImage = base_mask;

    base_pixels
        .par_chunks_mut(patch_rgb_stride)
        .zip(candidate_pixels.par_chunks_mut(patch_rgb_stride))
        .zip(blend_values.par_chunks_mut(patch_mask_stride))
        .zip(tone_blend_values.par_chunks_mut(patch_mask_stride))
        .zip(tone_sample_values.par_chunks_mut(patch_mask_stride))
        .zip(protected_values.par_chunks_mut(patch_mask_stride))
        .zip(protected_pixels.par_chunks_mut(patch_rgb_stride))
        .enumerate()
        .for_each(
            |(
                local_y,
                (
                    (
                        ((((base_row, candidate_row), blend_row), tone_blend_row), tone_sample_row),
                        protected_row,
                    ),
                    protected_pixel_row,
                ),
            )| {
                let source_y = patch_top + local_y as u32;
                for local_x in 0..patch_width {
                    let source_x = patch_left + local_x;
                    let global_x = candidate.left + source_x;
                    let global_y = candidate.top + source_y;
                    let base_valid = base_mask_ref.get_pixel(global_x, global_y)[0] > 0;
                    let candidate_valid = candidate.mask.get_pixel(source_x, source_y)[0] > 0;
                    let current_pixel = *base_ref.get_pixel(global_x, global_y);
                    let candidate_pixel = *candidate.image.get_pixel(source_x, source_y);
                    let base_pixel = if base_valid || !candidate_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };
                    let candidate_pixel = if candidate_valid {
                        candidate_pixel
                    } else {
                        base_pixel
                    };
                    let owns = candidate_valid
                        && (!base_valid || decision_mask.get_pixel(source_x, source_y)[0] > 0);
                    let candidate_foreground =
                        candidate.foreground_mask.get_pixel(source_x, source_y)[0] > 0;
                    let existing_foreground = foreground_overlap
                        [source_y as usize * layer_width as usize + source_x as usize]
                        > 0;
                    // Black ink, red seals, and other painted content are not part
                    // of the geometric depth mask. They still must not enter the
                    // low-frequency pyramid: averaging two sub-pixel-shifted
                    // strokes is exactly the blur/double-contour regression this
                    // path is intended to prevent. Keep the canvas gate separate
                    // so a darker/lighter paper exposure can still be harmonized.
                    let tone_foreground =
                        focus_stack_pixel_is_tone_foreground(base_pixel.0.as_slice())
                            || focus_stack_pixel_is_tone_foreground(candidate_pixel.0.as_slice());
                    let artwork_foreground =
                        candidate_foreground || existing_foreground || tone_foreground;
                    // Blend the broad tone over the union of the two valid
                    // canvas regions. When only one source exists, the
                    // missing side below is copied from that same source, so
                    // extending the mask is harmless and lets the low
                    // frequency transition continue through a tile edge.
                    // All artwork pixels use hard source ownership, including
                    // their low-frequency bands.
                    tone_blend_row[local_x as usize] =
                        u8::from((base_valid || candidate_valid) && !artwork_foreground);
                    tone_sample_row[local_x as usize] =
                        u8::from(base_valid && candidate_valid && !artwork_foreground);
                    protected_row[local_x as usize] = u8::from(artwork_foreground);
                    let start = local_x as usize * 3;
                    base_row[start..start + 3].copy_from_slice(&base_pixel.0);
                    candidate_row[start..start + 3].copy_from_slice(&candidate_pixel.0);
                    if artwork_foreground {
                        let selected_pixel = if owns { candidate_pixel } else { base_pixel };
                        protected_pixel_row[start..start + 3].copy_from_slice(&selected_pixel.0);
                    }
                    blend_row[local_x as usize] = if owns { 1.0 } else { 0.0 };
                }
            },
        );

    let low_frequency = masked_box_blur_focus_map(
        &blend_values,
        &tone_blend_values,
        patch_width,
        patch_height,
        (radius / 2).max(1),
    );

    // Match only the source-sized tone step measured from the shared canvas
    // samples. This is deliberately a constant correction for this local seam:
    // it is applied to the candidate before the pyramid is built, so it lands
    // in the coarsest tone band while the weave, brush strokes, and all hard
    // selected foreground detail remain in their original frequency bands.
    let tone_sample_count = tone_sample_values
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if tone_sample_count >= FOCUS_SEAM_TONE_MIN_SAMPLES {
        let mut channel_differences = [Vec::new(), Vec::new(), Vec::new()];
        for (index, &sample) in tone_sample_values.iter().enumerate() {
            if sample == 0 {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                channel_differences[channel]
                    .push(base_pixels[start + channel] - candidate_pixels[start + channel]);
            }
        }
        let mut tone_correction = [0.0f32; 3];
        for channel in 0..3 {
            let differences = &mut channel_differences[channel];
            differences.sort_unstable_by(f32::total_cmp);
            tone_correction[channel] = differences[differences.len() / 2].clamp(
                -FOCUS_SEAM_TONE_MAX_ADJUSTMENT,
                FOCUS_SEAM_TONE_MAX_ADJUSTMENT,
            );
        }
        for (index, &protected) in protected_values.iter().enumerate() {
            if protected > 0 {
                continue;
            }
            let local_x = (index % patch_width as usize) as u32;
            let local_y = (index / patch_width as usize) as u32;
            if candidate
                .mask
                .get_pixel(patch_left + local_x, patch_top + local_y)[0]
                == 0
            {
                continue;
            }
            let start = index * 3;
            // `low_frequency` is the smoothed hard ownership mask. Its middle
            // values identify the actual seam; using 4*a*(1-a) keeps the
            // measured correction out of the candidate's remote canvas and
            // out of the base side of the patch. The previous tone-union mask
            // was one across the whole rectangle, which silently turned a
            // large seam bounding box into a full candidate-wide recolour.
            let ownership = low_frequency[index].clamp(0.0, 1.0);
            let weight = (4.0 * ownership * (1.0 - ownership)).clamp(0.0, 1.0);
            if weight <= 0.0 {
                continue;
            }
            for channel in 0..3 {
                candidate_pixels[start + channel] += tone_correction[channel] * weight;
            }
        }
    }

    let blend_mask = GrayImage::from_fn(patch_width, patch_height, |x, y| {
        image::Luma([(blend_values[y as usize * patch_width as usize + x as usize] * 255.0) as u8])
    });
    let low_frequency_mask = GrayImage::from_fn(patch_width, patch_height, |x, y| {
        let index = y as usize * patch_width as usize + x as usize;
        let value = if tone_blend_values[index] > 0 {
            low_frequency[index]
        } else {
            blend_values[index]
        };
        image::Luma([(value.clamp(0.0, 1.0) * 255.0) as u8])
    });
    let blended = multiband_blend(
        Rgb32FImage::from_raw(patch_width, patch_height, base_pixels)
            .expect("focus seam base dimensions must match"),
        Rgb32FImage::from_raw(patch_width, patch_height, candidate_pixels)
            .expect("focus seam candidate dimensions must match"),
        blend_mask,
        Some(low_frequency_mask),
        PANORAMA_BLEND_BANDS,
        true,
    );

    let blended_pixels = blended.as_raw();
    base.as_mut()
        .par_chunks_mut(base_width as usize * 3)
        .enumerate()
        .skip((candidate.top + patch_top) as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, row)| {
            let local_y = global_y - (candidate.top + patch_top) as usize;
            let source_start = local_y * patch_rgb_stride;
            let destination_start = (candidate.left + patch_left) as usize * 3;
            row[destination_start..destination_start + patch_rgb_stride]
                .copy_from_slice(&blended_pixels[source_start..source_start + patch_rgb_stride]);
        });

    // Re-apply hard ownership to the protected layer. This is deliberately
    // after multiband reconstruction: the low-frequency seam mask is allowed
    // to smooth paper illumination, but not to average two displaced copies of
    // a depth-discontinuous object.
    base.as_mut()
        .par_chunks_mut(base_width as usize * 3)
        .enumerate()
        .skip((candidate.top + patch_top) as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, row)| {
            let local_y = global_y - (candidate.top + patch_top) as usize;
            for local_x in 0..patch_width as usize {
                let patch_index = local_y * patch_width as usize + local_x;
                if protected_values[patch_index] == 0 {
                    continue;
                }
                let source_start = patch_index * 3;
                let destination_start = (candidate.left + patch_left + local_x as u32) as usize * 3;
                row[destination_start..destination_start + 3]
                    .copy_from_slice(&protected_pixels[source_start..source_start + 3]);
            }
        });
    base_mask
        .as_mut()
        .par_chunks_mut(base_width as usize)
        .enumerate()
        .skip((candidate.top + patch_top) as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, row)| {
            let local_y = global_y - (candidate.top + patch_top) as usize;
            for local_x in 0..patch_width {
                let source_x = patch_left + local_x;
                if candidate
                    .mask
                    .get_pixel(source_x, patch_top + local_y as u32)[0]
                    > 0
                {
                    row[(candidate.left + source_x) as usize] = 255;
                }
            }
        });

    // Pixels outside the narrow seam patch still need the normal hard ownership
    // decision; only the low-frequency transition is allowed to cross the seam.
    hard_select_focus_layer_outside_patch(
        base,
        base_mask,
        candidate,
        decision_mask,
        patch_left,
        patch_right,
        patch_top,
        patch_bottom,
    );
}

fn hard_select_focus_layer_outside_patch(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
    excluded_left: u32,
    excluded_right: u32,
    excluded_top: u32,
    excluded_bottom: u32,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    let base_stride = base_width as usize * 3;
    let layer_stride = layer_width as usize * 3;
    let base_pixels = base.as_mut();
    let base_mask_pixels = base_mask.as_mut();
    let candidate_pixels = candidate.image.as_raw();
    let candidate_mask_pixels = candidate.mask.as_raw();
    let decision_pixels = decision_mask.as_raw();
    for local_y in 0..layer_height as usize {
        let global_y = candidate.top as usize + local_y;
        let base_row = &mut base_pixels[global_y * base_stride..(global_y + 1) * base_stride];
        let base_mask_row = &mut base_mask_pixels
            [global_y * base_width as usize..(global_y + 1) * base_width as usize];
        let candidate_row = &candidate_pixels[local_y * layer_stride..(local_y + 1) * layer_stride];
        let candidate_mask_row = &candidate_mask_pixels
            [local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
        let decision_row =
            &decision_pixels[local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
        for local_x in 0..layer_width as usize {
            if (excluded_left as usize..=excluded_right as usize).contains(&local_x)
                && (excluded_top as usize..=excluded_bottom as usize).contains(&local_y)
            {
                continue;
            }
            if candidate_mask_row[local_x] == 0 {
                continue;
            }
            let global_x = candidate.left as usize + local_x;
            let base_pixel_start = global_x * 3;
            let candidate_pixel_start = local_x * 3;
            if base_mask_row[global_x] == 0 || decision_row[local_x] > 0 {
                base_row[base_pixel_start..base_pixel_start + 3].copy_from_slice(
                    &candidate_row[candidate_pixel_start..candidate_pixel_start + 3],
                );
            }
            base_mask_row[global_x] = 255;
        }
    }
}

#[derive(Clone, Copy)]
struct MappedFrameGeometry {
    center: Point2<f64>,
    scale: f64,
    rotation: f64,
    area: f64,
}

fn mapped_frame_geometry(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Option<MappedFrameGeometry> {
    let width = image.width.max(1) as f64;
    let height = image.height.max(1) as f64;
    let source_corners = [
        Point2::new(0.0, 0.0),
        Point2::new(width, 0.0),
        Point2::new(width, height),
        Point2::new(0.0, height),
    ];
    let mut corners = [Point2::new(0.0, 0.0); 4];
    for (slot, source) in source_corners.into_iter().enumerate() {
        let projected = project_point(image, source.x, source.y, projection)?;
        let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            return None;
        }
        let point = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
        if !point.x.is_finite() || !point.y.is_finite() {
            return None;
        }
        corners[slot] = point;
    }

    let center = corners
        .iter()
        .copied()
        .fold(Point2::new(0.0, 0.0), |sum, point| sum + point.coords)
        / 4.0;
    let horizontal_scale =
        ((corners[1] - corners[0]).norm() + (corners[2] - corners[3]).norm()) / (2.0 * width);
    let vertical_scale =
        ((corners[3] - corners[0]).norm() + (corners[2] - corners[1]).norm()) / (2.0 * height);
    let scale = (horizontal_scale.max(0.0) * vertical_scale.max(0.0)).sqrt();
    let top_edge = corners[1] - corners[0];
    let rotation = top_edge.y.atan2(top_edge.x);
    let signed_double_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>();
    let area = (signed_double_area / (2.0 * width * height)).abs();
    (scale.is_finite() && rotation.is_finite() && area.is_finite()).then_some(MappedFrameGeometry {
        center,
        scale,
        rotation,
        area,
    })
}

fn wrapped_angle_delta(left: f64, right: f64) -> f64 {
    let mut delta = left - right;
    while delta > std::f64::consts::PI {
        delta -= 2.0 * std::f64::consts::PI;
    }
    while delta < -std::f64::consts::PI {
        delta += 2.0 * std::f64::consts::PI;
    }
    delta.abs()
}

fn focus_stack_is_shifted_mosaic(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> bool {
    let Some(first) = images.first() else {
        return false;
    };
    let Some(reference_homography) = global_homographies.get(&first.id) else {
        return false;
    };
    let Some(reference) = mapped_frame_geometry(first, reference_homography, projection) else {
        return false;
    };
    let reference_width = first.width().max(1) as f64;
    let reference_height = first.height().max(1) as f64;
    // Some exported frames lose the equivalent-focal tag. Use the first tag
    // available in the stack as a comparison anchor so a missing tag on the
    // first frame does not hide a later 35mm/85mm switch.
    let reference_focal = first
        .focal_length_35mm
        .or_else(|| images.iter().find_map(|image| image.focal_length_35mm));
    images.iter().skip(1).any(|image| {
        let Some(homography) = global_homographies.get(&image.id) else {
            return false;
        };
        let Some(current) = mapped_frame_geometry(image, homography, projection) else {
            return false;
        };
        let center_shift = (current.center.x - reference.center.x).abs() > reference_width * 0.08
            || (current.center.y - reference.center.y).abs() > reference_height * 0.08;
        let relative_scale = current.scale / reference.scale.max(1e-8);
        let scale_shift = relative_scale.is_finite() && !(0.90..=1.10).contains(&relative_scale);
        let relative_area = current.area / reference.area.max(1e-8);
        let area_shift = relative_area.is_finite() && !(0.82..=1.22).contains(&relative_area);
        let rotation_shift = wrapped_angle_delta(current.rotation, reference.rotation) > 0.035;
        let focal_shift = reference_focal
            .zip(image.focal_length_35mm)
            .map(|(reference_focal, current_focal)| {
                let ratio = current_focal / reference_focal.max(1e-8);
                ratio.is_finite() && !(0.90..=1.10).contains(&ratio)
            })
            .unwrap_or(false);
        center_shift || scale_shift || area_shift || rotation_shift || focal_shift
    })
}

pub fn focus_stack_stitcher<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    sequence_gap_aware: bool,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    focus_stack_stitcher_with_margin_policy(
        images,
        global_homographies,
        projection,
        focus_warp,
        capture_group_ids,
        // The margin-filling form is the single-canvas comparison output: it
        // invents pixels for the projected trapezoid corners and publishes no
        // Ownership_Map, so it carries no Capture_Station identity of its own.
        0,
        sequence_gap_aware,
        true,
        app_handle,
        progress_event,
        load_image,
    )
    .map(FocusStackTileRender::into_image)
}

/// The Coverage_Mask and Ownership_Map the focus renderer already maintains
/// internally (需求 3.7 / 3.11 / 4.6).  They are recorded at the exact pixel
/// write sites of the hard focus selection, so publishing them does not change
/// a single pixel write.
#[derive(Debug, Clone)]
pub(crate) struct FocusStackTileMasks {
    pub(crate) coverage: CoverageMask,
    pub(crate) ownership: OwnershipMap,
    /// Per-pixel Sharpness_Confidence of the owning ownership cell (需求 3.9).
    pub(crate) confidence: ConfidenceMap,
    /// Per-pixel Textured_Pixel evidence. A value of 255 means that at least
    /// one candidate observed a Sharpness_Score at or above the fixed 0.10
    /// floor in the owning cell (需求 11.12).
    pub(crate) textured: Vec<u8>,
    /// Compact winner/runner-up Sharpness_Score evidence at ownership-cell
    /// resolution.  This is report-only and is never consulted while writing
    /// the fused pixels.
    pub(crate) sharpness_evidence: focus_fuser::SharpnessEvidenceGrid,
    /// The Focus_Fuser observations of 需求 3.5 / 3.8 / 3.9.  `station_index` is
    /// filled by the call site that knows it.
    pub(crate) fusion: FusionReport,
}

/// One focus-fused station: the pixels plus, for the unfilled Virtual_Tile
/// form, the masks that describe where they came from.  `masks` is `None`
/// whenever the render did not go through the hard focus selection (empty
/// input, degenerate bounds, the shifted-mosaic path, or the margin-filling
/// single-layer output), because those paths cannot attest per-pixel ownership.
#[derive(Debug, Clone)]
pub(crate) struct FocusStackTileRender {
    pub(crate) image: Rgb32FImage,
    pub(crate) masks: Option<FocusStackTileMasks>,
    pub(crate) sampling_origin: (f64, f64),
}

impl FocusStackTileRender {
    fn unattributed(image: Rgb32FImage) -> Self {
        Self {
            image,
            masks: None,
            sampling_origin: (0.0, 0.0),
        }
    }

    pub(crate) fn into_image(self) -> Rgb32FImage {
        self.image
    }
}

/// Focus-fuse a station while preserving the projected validity mask.  Virtual
/// tiles use this form because their non-rectangular projective corners must
/// remain transparent until the tile-level compositor owns them, and because
/// the Virtual_Tile needs the Coverage_Mask and Ownership_Map that the fusion
/// builds on the way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn focus_stack_stitcher_unfilled<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    station_index: usize,
    sequence_gap_aware: bool,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<FocusStackTileRender, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    focus_stack_stitcher_with_margin_policy(
        images,
        global_homographies,
        projection,
        focus_warp,
        capture_group_ids,
        station_index,
        sequence_gap_aware,
        false,
        app_handle,
        progress_event,
        load_image,
    )
}

/// Nearest-neighbour read of a canvas-space position out of a layer that lives
/// at `(left, top)`, returning `None` outside the layer's validity mask.
///
/// The Sharpness_Score probes sit on a fixed grid whose spacing is a whole
/// number of native pixels, so there is nothing between samples to interpolate;
/// a bilinear read here would only low-pass the very gradients 需求 3.2 measures.
#[inline]
fn sample_focus_plane(
    image: &Rgb32FImage,
    mask: &GrayImage,
    left: u32,
    top: u32,
    x: f64,
    y: f64,
) -> Option<Rgb<f32>> {
    let local_x = x - f64::from(left);
    let local_y = y - f64::from(top);
    if !local_x.is_finite() || !local_y.is_finite() || local_x < 0.0 || local_y < 0.0 {
        return None;
    }
    let (px, py) = (local_x.round(), local_y.round());
    if px < 0.0 || py < 0.0 || px >= f64::from(image.width()) || py >= f64::from(image.height()) {
        return None;
    }
    let (px, py) = (px as u32, py as u32);
    if mask.get_pixel(px, py).0[0] == 0 {
        return None;
    }
    Some(*image.get_pixel(px, py))
}

/// Normalised Sharpness_Score and coverage of every ownership cell of one
/// projected layer (需求 3.2, 3.7).
///
/// A cell counts as covered when the layer's validity mask is set at the cell
/// centre.  The cell is the *decision* unit; a cell only partly inside the
/// projected trapezoid still gets one owner, and the per-pixel write below
/// never copies a pixel the layer does not actually have, so the Coverage_Mask
/// stays exact at pixel resolution.
fn measure_focus_cells(
    geometry: &OwnershipGridGeometry,
    plan: &CellSamplingPlan,
    image: &Rgb32FImage,
    mask: &GrayImage,
    left: u32,
    top: u32,
) -> (Vec<f64>, Vec<bool>) {
    // 需求 3.2: the candidate and the current composite are comparable only when
    // they are measured at the same positions with the same window, and the only
    // way to guarantee that across two call sites is to insist that both measure
    // with the plan this station's ownership grid defines.  A plan from anywhere
    // else would make defocus indistinguishable from a change of sampling
    // geometry, which is the one confusion 需求 3.2 exists to rule out.
    plan.assert_matches(&geometry.sampling_plan());
    let count = geometry.cell_count();
    let mut sharpness = vec![0.0f64; count];
    let mut covered = vec![false; count];
    let columns = geometry.columns as usize;
    let half = f64::from(geometry.cell_size_px) * 0.5;
    sharpness
        .par_iter_mut()
        .zip(covered.par_iter_mut())
        .enumerate()
        .for_each(|(cell, (sharpness, covered))| {
            let (column, row) = ((cell % columns) as u32, (cell / columns) as u32);
            let (origin_x, origin_y) = geometry.cell_origin(column, row);
            let sample = |x: f64, y: f64| sample_focus_plane(image, mask, left, top, x, y);
            if sample(origin_x + half, origin_y + half).is_none() {
                return;
            }
            *covered = true;
            *sharpness = plan.measure(origin_x, origin_y, sample);
        });
    (sharpness, covered)
}

/// Pixel inconsistency of every ownership cell between a candidate layer and
/// the current composite (需求 3.3).
fn measure_focus_cell_disagreement(
    geometry: &OwnershipGridGeometry,
    candidate: &RenderedFocusLayer,
    base: &Rgb32FImage,
    base_mask: &GrayImage,
) -> Vec<f64> {
    let count = geometry.cell_count();
    let columns = geometry.columns as usize;
    let cell_size = f64::from(geometry.cell_size_px);
    let mut disagreement = vec![0.0f64; count];
    disagreement
        .par_iter_mut()
        .enumerate()
        .for_each(|(cell, disagreement)| {
            let (column, row) = ((cell % columns) as u32, (cell / columns) as u32);
            let (origin_x, origin_y) = geometry.cell_origin(column, row);
            // Both closures receive the same fractional cell offsets and map
            // them into their own plane, which is how 需求 3.3's "same cell"
            // comparison stays a property of the code.
            *disagreement = cell_disagreement(
                |fx, fy| {
                    sample_focus_plane(
                        &candidate.image,
                        &candidate.mask,
                        candidate.left,
                        candidate.top,
                        origin_x + fx * cell_size,
                        origin_y + fy * cell_size,
                    )
                },
                |fx, fy| {
                    sample_focus_plane(
                        base,
                        base_mask,
                        0,
                        0,
                        origin_x + fx * cell_size,
                        origin_y + fy * cell_size,
                    )
                },
            );
        });
    disagreement
}

/// Rasterise an ownership cell decision onto a layer's own pixel grid
/// (需求 3.6).
///
/// The result is a hard mask whose boundaries are exactly ownership cell
/// boundaries: no feather, no alpha ramp, no transition band.  The legacy
/// streaming mosaic keeps its `STREAMING_OWNERSHIP_FEATHER` sub-cell blur; the
/// station layer no longer has one.
fn focus_cell_decision_for_layer(
    geometry: &OwnershipGridGeometry,
    took: &[bool],
    layer: &RenderedFocusLayer,
) -> GrayImage {
    let (width, height) = layer.image.dimensions();
    let mut decision = GrayImage::new(width, height);
    let cell_size = geometry.cell_size_px.max(1);
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    let width_usize = width as usize;
    decision
        .as_mut()
        .par_chunks_mut(width_usize)
        .enumerate()
        .for_each(|(y, row)| {
            let global_y = layer.top as usize + y;
            let cell_row = global_y / cell_size as usize;
            if cell_row >= rows {
                return;
            }
            let base = cell_row * columns;
            for (x, value) in row.iter_mut().enumerate() {
                let cell_column = (layer.left as usize + x) / cell_size as usize;
                if cell_column < columns && took[base + cell_column] {
                    *value = 255;
                }
            }
        });
    decision
}

/// Expand an ownership cell plane to one value per tile pixel (需求 4.6).
fn focus_cell_plane_to_pixels(
    geometry: &OwnershipGridGeometry,
    cells: &[f32],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let mut values = vec![0.0f32; width as usize * height as usize];
    let cell_size = geometry.cell_size_px.max(1) as usize;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    values
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let cell_row = y / cell_size;
            if cell_row >= rows {
                return;
            }
            let base = cell_row * columns;
            for (x, value) in row.iter_mut().enumerate() {
                let cell_column = x / cell_size;
                if cell_column < columns {
                    *value = cells[base + cell_column];
                }
            }
        });
    values
}

/// Expand the cell-level Textured_Pixel evidence onto the station plane.
fn focus_cell_texture_to_pixels(
    geometry: &OwnershipGridGeometry,
    cells: &[bool],
    width: u32,
    height: u32,
) -> Vec<u8> {
    let mut values = vec![0u8; width as usize * height as usize];
    let cell_size = geometry.cell_size_px.max(1) as usize;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    values
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let cell_row = y / cell_size;
            if cell_row >= rows {
                return;
            }
            let base = cell_row * columns;
            for (x, value) in row.iter_mut().enumerate() {
                let cell_column = x / cell_size;
                if cell_column < columns && cells[base + cell_column] {
                    *value = 255;
                }
            }
        });
    values
}

/// The Capture_Station members of 需求 12.1 / 12.2 as the default station path
/// can see them, together with every 需求 12.5 rejection its evidence supports.
///
/// The registration evidence is what the Intra_Station_Registrar recorded for
/// 需求 2.9 through [`intra_station::record_run_frame`].  Since the default path
/// runs the native patch refinement itself — `register_station_frame_native_layer`
/// below, which is `mosaic::refine_native_layer` over this path's own layer pose
/// — this function is what a *later* render of the same station reads, and what
/// the up-front plan of a station whose frames have already been measured is
/// built from.  A frame with a record is judged by the numbers in that record; a
/// frame without one has not been measured yet and is placed by its verified
/// global model plus the bounded integer translation of
/// `estimate_focus_layer_translation`, which has no control point field to accept
/// or reject and is therefore exactly `GlobalFallback`.  Nothing here invents a
/// substitute measurement: 需求 12.5 stays silent rather than judging a frame on a
/// number that was never taken.
fn station_members_from_records(
    images: &[&ImageInfo],
    station_index: usize,
) -> (Vec<StationMember>, Vec<GroupJoinRejection>) {
    let recorded = intra_station::run_records_snapshot()
        .into_iter()
        .find(|station| station.station_index == station_index)
        .map(|station| {
            station
                .frames
                .into_iter()
                .map(|frame| (frame.path.clone(), frame))
                .collect::<HashMap<String, IntraStationFrameRecord>>()
        })
        .unwrap_or_default();
    station_members_from_frame_records(images, &recorded)
}

/// [`station_members_from_records`] over an injected record set.
///
/// The run scoped sink is a global that concurrent tests share, so the rule
/// itself takes the records as data: production passes the snapshot of the
/// station it is rendering, a test passes a local map.
fn station_members_from_frame_records(
    images: &[&ImageInfo],
    recorded: &HashMap<String, IntraStationFrameRecord>,
) -> (Vec<StationMember>, Vec<GroupJoinRejection>) {
    // Layer 0 is the frame every later layer registers against, so it is this
    // station's anchor by construction and 需求 12.5 scales its error limit by
    // that frame's native long side.
    let anchor_long_side = images
        .first()
        .map(|info| info.width.max(info.height))
        .unwrap_or(0);
    let mut rejections = Vec::new();
    let members = images
        .iter()
        .enumerate()
        .map(|(index, info)| {
            let record = recorded.get(&info.filename);
            // 需求 12.5: inlier spatial support below 20% of the overlap area, or
            // an inlier median symmetric reprojection error above 0.01 × the long
            // side, keeps this Source_RAW out of every Capture_Station.  Only a
            // record with an accepted inlier set carries both measurements; with
            // none accepted the two fields are structural zeros rather than
            // measurements, and 需求 12.2's registration-failure exclusion below
            // already covers that frame.
            let rejection = record
                .filter(|record| index > 0 && record.inliers > 0)
                .and_then(|record| {
                    group_join_rejection(&GroupJoinEvidence {
                        path: info.filename.clone(),
                        inlier_area_coverage: record.inlier_area_coverage,
                        median_symmetric_error_px: record.median_symmetric_error_px,
                        anchor_long_side_px: anchor_long_side,
                    })
                });
            let status = match (
                index,
                rejection.is_some(),
                record.map(|record| record.status),
            ) {
                // Rejected from every Capture_Station, so this station cannot
                // fuse it either (需求 12.2 / 12.5).
                (_, true, _) => IntraStationFrameStatus::Failed,
                (0, _, _) => IntraStationFrameStatus::Anchor,
                (_, false, Some(status)) => status,
                (_, false, None) => IntraStationFrameStatus::GlobalFallback,
            };
            if let Some(rejection) = rejection {
                rejections.push(rejection);
            }
            StationMember {
                path: info.filename.clone(),
                median_sharpness: super::mosaic::station_member_median_sharpness(info),
                status,
            }
        })
        .collect();
    (members, rejections)
}

/// Test-only entry point for Property 78. This deliberately delegates to the
/// production default-station wiring above, rather than duplicating the
/// rejection predicate in the property harness.
#[cfg(test)]
pub(crate) fn station_members_from_measured_group_join_evidence(
    images: &[&ImageInfo],
    recorded: &HashMap<String, IntraStationFrameRecord>,
) -> (Vec<StationMember>, Vec<GroupJoinRejection>) {
    station_members_from_frame_records(images, recorded)
}

#[allow(clippy::too_many_arguments)]
fn focus_stack_stitcher_with_margin_policy<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    station_index: usize,
    sequence_gap_aware: bool,
    fill_margins: bool,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<FocusStackTileRender, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    if images.is_empty() {
        return Ok(FocusStackTileRender::unattributed(Rgb32FImage::new(0, 0)));
    }
    let capture_group_count = capture_group_ids
        .map(|ids| {
            ids.values()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len()
        })
        .unwrap_or(0);
    // Focus breathing and small tripod movement can trip the framing/scale
    // heuristic even when every source belongs to one focus bracket.  The
    // hard, position-owned mosaic is only appropriate when registration has
    // independently recovered multiple camera positions (or an explicit
    // sequence gap).  A single capture group must retain the mature focus
    // fusion path; otherwise its cell-wise local warp visibly fragments
    // strokes that are intentionally present at different focal depths.
    let shifted_mosaic = sequence_gap_aware
        || (capture_group_count > 1
            && focus_stack_is_shifted_mosaic(images, global_homographies, projection));
    if !shifted_mosaic {
        println!(
            "  - Compact focus capture detected ({} inferred group(s)); preserving standard focus fusion",
            capture_group_count.max(1)
        );
    }
    if shifted_mosaic {
        println!("  - Framing/lens shift detected; using local registration and detail ownership");
        let _ = app_handle.emit(
            progress_event,
            "Framing or lens change detected; refining registration and selecting sharp detail...",
        );
        return super::mosaic::detail_preserving_mosaic(
            images,
            global_homographies,
            projection,
            capture_group_ids,
            sequence_gap_aware,
            app_handle,
            progress_event,
            load_image,
        )
        .map(FocusStackTileRender::unattributed);
    }
    let (min_x, max_x, min_y, max_y) =
        focus_output_bounds(images, global_homographies, projection, focus_warp);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return Ok(FocusStackTileRender::unattributed(Rgb32FImage::new(0, 0)));
    }
    let (offset_x, out_width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, out_height) = pixel_aligned_canvas(min_y, max_y);
    // 需求 12.1 / 12.2 / 12.5: which members this Capture_Station can fuse, as
    // the records of earlier runs of this station already describe them.  The
    // loop below measures this run's evidence itself (需求 2.2–2.9) and updates
    // `station_members` frame by frame, so the plan that is finally *recorded* is
    // the one the pixels followed.  The canvas bounds above deliberately stay
    // over the whole group: the Virtual_Tile geometry the caller already
    // published is computed from the same set, and a degraded station must not
    // silently resize its tile.
    let (mut station_members, group_join_rejections) =
        station_members_from_records(images, station_index);
    for rejection in &group_join_rejections {
        println!(
            "  - Rejected from every Capture_Station: '{}' inlier support {:.3} (minimum {:.2}), median symmetric error {:.2}px (limit {:.2}px)",
            rejection.path,
            rejection.inlier_area_coverage,
            rejection.minimum_inlier_area_coverage,
            rejection.median_symmetric_error_px,
            rejection.symmetric_error_limit_px
        );
        record_run_group_join_rejection(station_index, rejection);
    }
    let station_plan = plan_station_fusion(&station_members);
    // 需求 3.10 and 12.1 both retain source-exact pixel writes; the former
    // physically contains one successful frame, while the latter selects one
    // survivor from a larger station.
    // 需求 12.1: a degraded station keeps one Source_RAW, so `fused` holds that
    // frame alone and every covered pixel below is copied from it.
    let seed_index = station_plan.fused.first().copied().unwrap_or(0);
    // 需求 2.1 / 2.9: the seed frame is the one every later frame of this station
    // is composited against, so it is this station's anchor; it carries the
    // identity transform and its record is the reference the other frames'
    // transforms are expressed against.
    let anchor_path = images[seed_index].filename.clone();
    let anchor_long_side = images[seed_index].width.max(images[seed_index].height);
    let anchor_to_station_inverse = global_homographies
        .get(&images[seed_index].id)
        .and_then(|transform| transform.try_inverse());
    intra_station::record_run_frame(
        station_index,
        &anchor_path,
        IntraStationFrameRecord {
            path: anchor_path.clone(),
            status: IntraStationFrameStatus::Anchor,
            inlier_area_coverage: 1.0,
            ..Default::default()
        },
    );
    if let Some(member) = station_members.get_mut(seed_index) {
        member.status = IntraStationFrameStatus::Anchor;
    }
    let first_source = load_image(images[seed_index])?;
    let first_layer = render_focus_layer(
        images[seed_index],
        &first_source,
        &global_homographies[&images[seed_index].id],
        projection,
        focus_warp,
        offset_x,
        offset_y,
        out_width,
        out_height,
    );
    drop(first_source);
    // The merged image occupies the final canvas, but each subsequent source
    // layer is kept local to its transformed bounds. This is important for a
    // long scan: allocating a full-canvas candidate for every 60MP source can
    // multiply memory use by the number of frames.
    let mut merged = Rgb32FImage::new(out_width, out_height);
    let mut merged_mask = GrayImage::new(out_width, out_height);
    let mut merged_foreground_mask = GrayImage::new(out_width, out_height);
    let mut merged_owner = GrayImage::new(out_width, out_height);
    place_focus_layer(&mut merged, &mut merged_mask, &first_layer);
    place_focus_foreground_mask(&mut merged_foreground_mask, &first_layer);
    // Published Ownership_Map (需求 3.11 / 4.1).  `merged_owner` above stays
    // exactly as it was: it is an 8-bit tone-harmonisation grouping key with
    // its own identifier scheme, while this map is the per-pixel Source_RAW
    // attribution the Virtual_Tile publishes.  It is only tracked for the
    // unfilled tile form, where every covered pixel is a hard selection from
    // one source; the margin-filling output invents pixels for the projected
    // trapezoid corners and therefore cannot attest ownership.
    let seed_owner = (seed_index + 1).min(u16::MAX as usize) as u16;
    let mut ownership_ids = (!fill_margins).then(|| {
        let mut owners = vec![NO_OWNER; out_width as usize * out_height as usize];
        owners
            .par_iter_mut()
            .zip(merged_mask.as_raw().par_iter())
            .for_each(|(owner, &covered)| {
                if covered > 0 {
                    *owner = seed_owner;
                }
            });
        owners
    });
    merged_owner
        .as_mut()
        .par_chunks_mut(out_width as usize)
        .zip(merged_mask.as_raw().par_chunks(out_width as usize))
        .for_each(|(owner_row, mask_row)| {
            for (owner, mask) in owner_row.iter_mut().zip(mask_row) {
                if *mask > 0 {
                    *owner = 1;
                }
            }
        });
    // Focus_Fuser (需求 3.1–3.11).  The station's ownership grid decides which
    // source owns each cell; the loop below copies that source's pixels and
    // nothing else, so an owner boundary is an ownership cell boundary.
    // 需求 3.10 / 12.1: the frame count is the number of frames the plan fuses,
    // so a single-frame degraded station reports `SingleFrame` and skips the cut
    // it has nothing to cut against.
    let mut fusion = StationFusion::new(
        out_width,
        out_height,
        station_plan.fused.len().min(images.len()).max(1),
    );
    let fusion_geometry = fusion.geometry();
    let fusion_plan = fusion.sampling_plan();
    {
        // The composite currently holds exactly the seed frame, so measuring it
        // here is measuring that frame — with the same plan every candidate
        // will use (需求 3.2).
        let (sharpness, covered) =
            measure_focus_cells(&fusion_geometry, &fusion_plan, &merged, &merged_mask, 0, 0);
        fusion.seed(seed_index, &sharpness, &covered);
    }
    println!(
        "  - Focus ownership grid: {}x{} cells of {}px over a {}x{} station plane",
        fusion_geometry.columns,
        fusion_geometry.rows,
        fusion_geometry.cell_size_px,
        out_width,
        out_height
    );
    let source_longest_dimension = images
        .iter()
        .map(|image| image.width().max(image.height()))
        .max()
        .unwrap_or(1);
    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(out_width, out_height, source_longest_dimension);
    let mut merged_analysis = resize_rgb(&merged, analysis_width, analysis_height);
    let mut merged_analysis_mask =
        resize_binary_mask(&merged_mask, analysis_width, analysis_height);
    let mut merged_analysis_foreground_mask =
        resize_binary_mask(&merged_foreground_mask, analysis_width, analysis_height);
    let mut merged_focus = focus_score_map(&merged_analysis, &merged_analysis_mask);

    // 需求 12.2: only the frames the plan fused are loaded, rendered and offered
    // to the cut.  An excluded Source_RAW is never sampled, so no pixel of it can
    // reach the Virtual_Tile while the report calls it excluded.
    for &index in station_plan.fused.iter().skip(1) {
        let image = images[index];
        let _ = app_handle.emit(
            progress_event,
            format!("Focus-stacking image {} of {}", index + 1, images.len()),
        );
        let source_image = load_image(image)?;
        let mut candidate = render_focus_layer(
            image,
            &source_image,
            &global_homographies[&image.id],
            projection,
            focus_warp,
            offset_x,
            offset_y,
            out_width,
            out_height,
        );
        let mut applied_translation = (0i32, 0i32);
        if let Some((delta_x, delta_y, score, support, patch_count)) =
            estimate_focus_layer_translation(&candidate, &merged, &merged_mask)
        {
            println!(
                "  - Full-resolution focus refinement: delta=({delta_x},{delta_y}), score={score:.3}, consensus={support}/{patch_count} patches"
            );
            candidate =
                translate_rendered_focus_layer(candidate, delta_x, delta_y, out_width, out_height);
            applied_translation = (delta_x, delta_y);
        }
        // Intra_Station_Registrar (需求 2.2–2.9).  The integer translation above
        // is this frame's *global* model on this path; the registrar now measures
        // and corrects what is left of it at native resolution, against the
        // station canvas the anchor seeded.
        let registration = register_station_frame_native_layer(
            &merged,
            &merged_mask,
            image,
            &source_image,
            &global_homographies[&image.id],
            projection,
            (offset_x, offset_y),
            applied_translation,
            &candidate,
            anchor_long_side,
            anchor_to_station_inverse,
        );
        drop(source_image);
        if let Some(registration) = registration {
            let record = registration.record.clone();
            println!(
                "    - Intra-station registration '{}': {:?}, {} inliers of {} control points, coverage {:.3}, rejected {:.3}, median symmetric error {:.2}px, control point spacing {:.1}px",
                image.filename,
                record.status,
                record.inliers,
                registration.evaluated_control_points,
                record.inlier_area_coverage,
                record.rejected_control_point_ratio,
                record.median_symmetric_error_px,
                record.control_point_spacing_px
            );
            intra_station::record_run_frame(station_index, &anchor_path, record.clone());
            // 需求 12.5: inlier spatial support below 20%, or an inlier median
            // symmetric reprojection error above 0.01 × the anchor's long side,
            // keeps this Source_RAW out of every Capture_Station — so this
            // station cannot fuse it either (需求 12.2).  A frame whose evidence
            // was never measured has nothing to judge and keeps the behaviour it
            // had before.
            let rejection = (registration.evaluated_control_points > 0 && record.inliers > 0)
                .then(|| {
                    group_join_rejection(&GroupJoinEvidence {
                        path: image.filename.clone(),
                        inlier_area_coverage: record.inlier_area_coverage,
                        median_symmetric_error_px: record.median_symmetric_error_px,
                        anchor_long_side_px: anchor_long_side,
                    })
                })
                .flatten();
            if let Some(member) = station_members.get_mut(index) {
                member.status = match (rejection.is_some(), record.status) {
                    (true, _) => IntraStationFrameStatus::Failed,
                    (false, IntraStationFrameStatus::Failed) => {
                        IntraStationFrameStatus::GlobalFallback
                    }
                    (false, status) => status,
                };
            }
            if let Some(rejection) = rejection {
                println!(
                    "    - Rejected from every Capture_Station: inlier support {:.3} (minimum {:.2}), median symmetric error {:.2}px (limit {:.2}px)",
                    rejection.inlier_area_coverage,
                    rejection.minimum_inlier_area_coverage,
                    rejection.median_symmetric_error_px,
                    rejection.symmetric_error_limit_px
                );
                record_run_group_join_rejection(station_index, &rejection);
                continue;
            }
            // Apply exactly what was measured: the same displacement field the
            // gates above accepted, resampled onto the layer that produced it.
            if let Some(displacement) = registration.displacement.as_ref() {
                let max_offset = displacement.max_offset_px();
                if max_offset >= FOCUS_NATIVE_REFINEMENT_MIN_OFFSET_PX {
                    println!("    - Applying native refinement field: up to {max_offset:.2}px");
                    candidate = warp_rendered_focus_layer(candidate, displacement);
                }
            }
        }
        // A regional warp can still leave a small residual translation on a
        // depth-discontinuous layer. Refine only the detected foreground
        // ownership mask against the already selected foreground when the
        // stack is a normal, mostly-overlapping focus capture. In a shifted
        // mosaic the existing mask can contain a different physical instance
        // of a repeated rail/seal, so mask-only matching is not image-backed
        // registration and can move the whole foreground onto the wrong copy.
        // The shifted-mosaic path has already used the bounded edge-consensus
        // geometry above; leave its residuals to that model and the focus
        // ownership pass.
        if !shifted_mosaic {
            candidate =
                align_focus_foreground_layer_to_existing(candidate, &merged_foreground_mask);
        }
        // Requirement 3.6: registration chooses the corresponding sample and
        // ownership chooses the frame; neither step may tone-modify the
        // candidate while continuing to identify that Source_RAW as owner.
        // Inter-station low-frequency harmonisation is a later pipeline stage
        // and must not alter this Virtual_Tile's source-level provenance.
        let candidate_analysis = render_focus_analysis_layer(
            &candidate,
            out_width,
            out_height,
            analysis_width,
            analysis_height,
        );
        let candidate_focus = focus_score_map(&candidate_analysis.image, &candidate_analysis.mask);
        let mut analysis_decision = focus_decision_mask(
            &merged_focus,
            &candidate_focus,
            &merged_analysis_mask,
            &candidate_analysis.mask,
        );
        // Keep the low-resolution ownership model in lockstep with the final
        // image. If focus scores are allowed to switch a near-field object at
        // analysis resolution and the full-resolution pass vetoes that switch,
        // the next source can still see the stale analysis winner and add a
        // second copy of the object. The mask is deliberately generic and
        // applies to any detected near-field/occlusion layer.
        if shifted_mosaic {
            suppress_focus_foreground_switches_on_canvas(
                &mut analysis_decision,
                &merged_analysis_foreground_mask,
                &candidate_analysis.foreground_mask,
                &candidate_analysis.relaxed_foreground_mask,
            );
            suppress_focus_foreground_ownership_switches_on_canvas(
                &mut analysis_decision,
                &merged_analysis_foreground_mask,
                &candidate_analysis.mask,
                &candidate_analysis.relaxed_foreground_mask,
            );
        }
        // In a compact focus bracket, the paper/canvas plane is exactly what
        // the stack is supposed to sharpen, so later focused layers must be
        // allowed to replace an already covered canvas pixel.  The guard is
        // only needed for a shifted scan, where a canvas-like overlap can be
        // a different physical camera position and switching it creates broad
        // rectangular exposure blocks.
        if shifted_mosaic {
            suppress_focus_canvas_switches(
                &mut analysis_decision,
                &merged_analysis,
                &candidate_analysis.image,
                &merged_analysis_mask,
                &candidate_analysis.mask,
            );
        }
        // Focus_Fuser (需求 3.2–3.6): on the standard focus path the ownership
        // grid, not the analysis-resolution focus map, decides this candidate.
        // `analysis_decision` above keeps feeding the shifted-mosaic guards and
        // the analysis score table; the pixel write follows the cut alone, at
        // ownership cell resolution and with no transition band.  The cut runs
        // on the *photometrically corrected* candidate, so the inconsistency of
        // 需求 3.3 measures geometry rather than exposure.
        let mut full_resolution_decision = {
            let mut cells = CandidateCells::new(fusion_geometry.cell_count());
            let (sharpness, covered) = measure_focus_cells(
                &fusion_geometry,
                &fusion_plan,
                &candidate.image,
                &candidate.mask,
                candidate.left,
                candidate.top,
            );
            cells.sharpness = sharpness;
            cells.covered = covered;
            cells.disagreement = measure_focus_cell_disagreement(
                &fusion_geometry,
                &candidate,
                &merged,
                &merged_mask,
            );
            let took = fusion.fold(index, &cells);
            focus_cell_decision_for_layer(&fusion_geometry, &took, &candidate)
        };
        // A detected near-field layer is geometrically separate from the main
        // image plane. Do not let focus sharpness make later frames replace an
        // already placed instance of that layer; doing so creates block-shaped
        // brightness and edge jumps in a moving scan. New pixels can still be
        // added where the mosaic has no ownership yet.
        if shifted_mosaic {
            suppress_focus_foreground_switches(
                &mut full_resolution_decision,
                &merged_foreground_mask,
                &candidate,
            );
            suppress_focus_foreground_ownership_switches(
                &mut full_resolution_decision,
                &merged_foreground_mask,
                &candidate.mask,
                candidate.left,
                candidate.top,
                &candidate.relaxed_foreground_mask,
            );
        }
        // A shifted scan can put the ownership boundary through a face, sleeve,
        // or painted contour. Averaging the full-resolution detail there turns
        // two slightly displaced sharp samples into a soft double exposure.
        // The seam path therefore blends only low-frequency canvas tone while
        // keeping detail-band source ownership hard and deterministic.
        let seam_blend_enabled = shifted_mosaic && FOCUS_ALLOW_LOW_FREQUENCY_SEAM_BLEND;
        if seam_blend_enabled {
            // A blended seam band mixes two sources in one pixel, so no single
            // Source_RAW owns it.  Drop the published map instead of claiming
            // an owner that is not the only contributor.
            ownership_ids = None;
            blend_focus_seam_band(
                &mut merged,
                &mut merged_mask,
                &merged_foreground_mask,
                &candidate,
                &full_resolution_decision,
            );
        } else {
            if let Some(ownership_ids) = ownership_ids.as_mut() {
                // Recorded from the coverage state *before* the selection, so
                // it marks exactly the pixels the call below copies.
                record_focus_layer_ownership(
                    ownership_ids,
                    &merged_mask,
                    &candidate,
                    &full_resolution_decision,
                    (index + 1).min(u16::MAX as usize) as u16,
                );
            }
            hard_select_focus_layer(
                &mut merged,
                &mut merged_mask,
                &candidate,
                &full_resolution_decision,
            );
        }
        mark_focus_foreground_pixels(
            &mut merged_foreground_mask,
            &merged_mask,
            &candidate,
            &full_resolution_decision,
        );
        mark_focus_owner_pixels(
            &mut merged_owner,
            &candidate,
            &full_resolution_decision,
            (index + 2).min(u8::MAX as usize) as u8,
        );
        commit_focus_analysis_scores(
            &mut merged_focus,
            &candidate_focus,
            &candidate_analysis.mask,
            &candidate,
            &full_resolution_decision,
            out_width,
            out_height,
            analysis_width,
            analysis_height,
        );
        // Rebuild the analysis image from the authoritative full-resolution
        // result for the canvas/foreground guards. Focus scores intentionally
        // stay in the separate best-score table above: recomputing them from
        // this image would make a selected seam, a sub-pixel warp, or a cubic
        // resample look like a new focus layer and would reintroduce order-
        // dependent blur.
        merged_analysis = resize_rgb(&merged, analysis_width, analysis_height);
        merged_analysis_mask = resize_binary_mask(&merged_mask, analysis_width, analysis_height);
        merged_analysis_foreground_mask =
            resize_binary_mask(&merged_foreground_mask, analysis_width, analysis_height);
    }
    // 需求 12.1 / 12.2: the station's fusion path, with every excluded
    // Source_RAW listed individually.  `station_members` now carries what the
    // loop really fused — the registrar's own verdict for every frame it
    // measured — so the recorded plan and the pixels agree.
    //
    // A station of one Source_RAW has no non-anchor frame, so neither 需求 12.1
    // ("all non-anchor frames failed") nor 需求 12.2 ("at least one failed") has
    // a premise: nothing degraded and nothing is reported.  The fusion above
    // still saw exactly one frame, which is 需求 3.10's single-frame short
    // circuit and not a degradation.
    if station_members.len() >= 2 {
        let measured_plan = plan_station_fusion(&station_members);
        let entries = station_plan_entries(station_index, &station_members, &measured_plan);
        if !entries.is_empty() {
            println!(
                "  - Capture_Station {station_index} fused {} of {} frame(s) ({:?})",
                measured_plan.fused.len(),
                station_members.len(),
                measured_plan.mode
            );
            for excluded in &measured_plan.excluded {
                println!("    - Excluded '{}': {}", excluded.path, excluded.reason);
            }
        }
        record_run_entries(&entries);
    }
    let merged_dimensions = merged.dimensions();
    if fill_margins {
        let (filled_pixels, remaining_invalid_pixels) =
            fill_focus_canvas_margins(&mut merged, &mut merged_mask);
        println!(
            "  - Kept full focus-stack canvas {}x{}; filled {} invalid margin pixels{}",
            merged_dimensions.0,
            merged_dimensions.1,
            filled_pixels,
            if remaining_invalid_pixels > 0 {
                format!(" ({} remain)", remaining_invalid_pixels)
            } else {
                String::new()
            }
        );
    } else {
        println!(
            "  - Kept unfilled focus-tile canvas {}x{}; retaining projected invalid margins",
            merged_dimensions.0, merged_dimensions.1
        );
        // Requirement 3.6: an attributed Virtual_Tile is the selected source
        // sample itself. Sharpening would create a value that no owner supplied.
        println!("  - Source-owned Capture_Station tile; skipping tile sharpening");
    }
    if std::env::var_os("RAW_EDITOR_FOCUS_OWNERSHIP_DIAGNOSTICS").is_some() {
        let mut counts = vec![0usize; images.len() + 2];
        for (&owner, &valid) in merged_owner.as_raw().iter().zip(merged_mask.as_raw()) {
            if valid == 0 {
                continue;
            }
            let index = owner as usize;
            if index >= counts.len() {
                counts.resize(index + 1, 0);
            }
            counts[index] += 1;
        }
        let total: usize = counts.iter().sum();
        let summary = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(index, count)| {
                format!(
                    "{}:{:.1}%",
                    index,
                    *count as f64 * 100.0 / total.max(1) as f64
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  - Focus ownership diagnostics: valid={} owners=[{}]",
            total, summary
        );
    }
    // Requirement 3.6: source-level ownership and tone modification are
    // mutually exclusive. Tone harmonisation belongs after inter-station seam
    // ownership and cannot run inside an attributed Capture_Station tile.
    println!("  - Source-owned Capture_Station tile; skipping owner tone harmonization");
    // 需求 3.8 / 3.9 / 4.6: publish the fuser's per-cell confidence at pixel
    // resolution and the low sharpness regions in world coordinates.  The tile
    // plane is a pure translation of the world plane, so the tile's top-left
    // pixel sits at `(-offset_x, -offset_y)`.
    let confidence_cells = fusion.confidence();
    let textured_cells = fusion.textured_cells();
    let sharpness_evidence = fusion.sharpness_evidence();
    let low_sharpness_regions = fusion.low_sharpness_region_records((-offset_x, -offset_y));
    if !low_sharpness_regions.is_empty() {
        println!(
            "  - Focus fusion found {} low-sharpness region(s) below the station P10",
            low_sharpness_regions.len()
        );
        degradation::record_run_degradation(
            degradation::FUSION_LOW_SHARPNESS_REGION,
            serde_json::json!({ "regions": low_sharpness_regions.len() }),
        );
    }
    let confidence_values =
        focus_cell_plane_to_pixels(&fusion_geometry, &confidence_cells, out_width, out_height);
    let textured_values =
        focus_cell_texture_to_pixels(&fusion_geometry, &textured_cells, out_width, out_height);
    let mut fusion_report = FusionReport {
        cell_size_px: fusion_geometry.cell_size_px,
        grid: OwnershipGridSize {
            columns: fusion_geometry.columns,
            rows: fusion_geometry.rows,
        },
        solver_status: fusion.solver_status(),
        graph_cut_seconds: fusion.graph_cut_seconds(),
        confidence: focus_fuser::confidence_summary(&confidence_values, merged_mask.as_raw()),
        low_sharpness_regions,
        ..FusionReport::default()
    };
    println!(
        "  - Focus ownership solved in {:.2}s ({:?}); mean confidence {:.3}, below 0.05 {:.2}%",
        fusion_report.graph_cut_seconds,
        fusion_report.solver_status,
        fusion_report.confidence.mean,
        fusion_report.confidence.below_0_05_ratio * 100.0
    );
    let masks = ownership_ids
        .map(|owners| -> Result<FocusStackTileMasks, String> {
            let legend = images
                .iter()
                .map(|image| PathBuf::from(&image.filename))
                .collect::<Vec<_>>();
            let confidence = ConfidenceMap::per_pixel(out_width, out_height, confidence_values)?;
            let ownership = OwnershipMap::new(out_width, out_height, owners, legend)?;
            fusion_report.owned_pixels = ownership.assigned_pixels();
            fusion_report.uncovered_pixels = u64::from(out_width) * u64::from(out_height)
                - merged_mask
                    .as_raw()
                    .par_iter()
                    .filter(|&&covered| covered > 0)
                    .count() as u64;
            Ok(FocusStackTileMasks {
                coverage: CoverageMask::from_gray(merged_mask),
                ownership,
                confidence,
                textured: textured_values,
                sharpness_evidence,
                fusion: fusion_report,
            })
        })
        .transpose()?;
    Ok(FocusStackTileRender {
        image: merged,
        masks,
        sampling_origin: (-offset_x, -offset_y),
    })
}

/// Restore a small amount of native edge contrast after focus ownership.  The
/// focus score intentionally uses a local blur and the ownership mask is
/// upsampled from an analysis canvas, both of which are stable but slightly
/// conservative around thin brush strokes.  This bounded luma-only unsharp
/// pass runs per virtual tile, never across a projected invalid corner, and
/// therefore cannot blend two camera positions together.
fn sharpen_focus_tile_detail(image: &mut Rgb32FImage, mask: &GrayImage, amount: f32) {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 || mask.dimensions() != (width, height) || amount <= 0.0 {
        return;
    }
    let source = image.clone();
    let source_pixels = source.as_raw();
    let source_mask = mask.as_raw();
    let width_usize = width as usize;
    let height_usize = height as usize;
    image
        .as_mut()
        .par_chunks_mut(width_usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            if y == 0 || y + 1 >= height_usize {
                return;
            }
            for x in 1..width_usize - 1 {
                let center_index = y * width_usize + x;
                if source_mask[center_index] == 0 {
                    continue;
                }
                let center_start = center_index * 3;
                let center_luma = source_pixels[center_start] * 0.299
                    + source_pixels[center_start + 1] * 0.587
                    + source_pixels[center_start + 2] * 0.114;
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        let index = ny * width_usize + nx;
                        if source_mask[index] == 0 {
                            continue;
                        }
                        let start = index * 3;
                        sum += source_pixels[start] * 0.299
                            + source_pixels[start + 1] * 0.587
                            + source_pixels[start + 2] * 0.114;
                        count += 1;
                    }
                }
                if count < 5 {
                    continue;
                }
                let detail = center_luma - sum / count as f32;
                let boost = (detail * amount).clamp(-0.12, 0.12);
                let start = x * 3;
                row[start] = (row[start] + boost).clamp(0.0, 1.0);
                row[start + 1] = (row[start + 1] + boost).clamp(0.0, 1.0);
                row[start + 2] = (row[start + 2] + boost).clamp(0.0, 1.0);
            }
        });
}

fn find_adaptive_seam(ctx: &SeamContext) -> Option<SeamInfo> {
    let h_add_inv = ctx.h_add.try_inverse()?;
    let (w_add, h_add_img) = ctx.img_to_add.dimensions();

    let mut min_ox = u32::MAX;
    let mut max_ox = 0;
    let mut min_oy = u32::MAX;
    let mut max_oy = 0;
    let mut has_overlap = false;

    let (candidate_left, candidate_right, candidate_top, candidate_bottom) =
        transformed_image_region(
            ctx.img_to_add_info,
            ctx.h_add,
            ctx.projection,
            ctx.offset_x,
            ctx.offset_y,
            ctx.out_width,
            ctx.out_height,
        )?;
    for y in candidate_top..=candidate_bottom {
        for x in candidate_left..=candidate_right {
            if ctx.pano_mask.get_pixel(x, y)[0] > 0 {
                let target_p = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
                let Some(source) =
                    map_target_to_source(&h_add_inv, target_p, ctx.img_to_add_info, ctx.projection)
                else {
                    continue;
                };
                let sx = source.x;
                let sy = source.y;
                if sx >= 0.0
                    && sx < w_add as f64
                    && sy >= 0.0
                    && sy < h_add_img as f64
                    && source_sample_has_support(ctx.img_to_add, sx, sy)
                {
                    has_overlap = true;
                    min_ox = min_ox.min(x);
                    max_ox = max_ox.max(x);
                    min_oy = min_oy.min(y);
                    max_oy = max_oy.max(y);
                }
            }
        }
    }

    if !has_overlap {
        return None;
    }

    println!(
        "    - Overlap bounds: x={}..{}, y={}..{}",
        min_ox, max_ox, min_oy, max_oy
    );

    let center_source = project_point(
        ctx.img_to_add_info,
        w_add as f64 / 2.0,
        h_add_img as f64 / 2.0,
        ctx.projection,
    )?;
    let center_p_source = Point3::new(center_source.x, center_source.y, 1.0);
    let center_p_target = ctx.h_add * center_p_source;
    let center_add_x = (center_p_target.x / center_p_target.z) + ctx.offset_x;
    let center_add_y = (center_p_target.y / center_p_target.z) + ctx.offset_y;

    let center_overlap_x = (min_ox + max_ox) as f64 / 2.0;
    let center_overlap_y = (min_oy + max_oy) as f64 / 2.0;

    let dx = center_add_x - center_overlap_x;
    let dy = center_add_y - center_overlap_y;

    if dx.abs() > dy.abs() {
        println!("    - Overlap is vertical. Finding vertical seam...");
        let seam = find_pairwise_seam_dp(
            ctx,
            SeamOrientation::Vertical,
            min_ox,
            max_ox,
            min_oy,
            max_oy,
        );
        if let Some(active) = seam.get(min_oy as usize..=max_oy as usize) {
            let minimum = active.iter().copied().min().unwrap_or_default();
            let maximum = active.iter().copied().max().unwrap_or_default();
            println!("    - Vertical seam range: x={minimum}..{maximum}");
        }
        Some(SeamInfo {
            orientation: SeamOrientation::Vertical,
            coords: seam,
            dx,
            dy,
            min_x: min_ox,
            max_x: max_ox,
            min_y: min_oy,
            max_y: max_oy,
        })
    } else {
        println!("    - Overlap is horizontal. Finding horizontal seam...");
        let seam = find_pairwise_seam_dp(
            ctx,
            SeamOrientation::Horizontal,
            min_ox,
            max_ox,
            min_oy,
            max_oy,
        );
        if let Some(active) = seam.get(min_ox as usize..=max_ox as usize) {
            let minimum = active.iter().copied().min().unwrap_or_default();
            let maximum = active.iter().copied().max().unwrap_or_default();
            println!("    - Horizontal seam range: y={minimum}..{maximum}");
        }
        Some(SeamInfo {
            orientation: SeamOrientation::Horizontal,
            coords: seam,
            dx,
            dy,
            min_x: min_ox,
            max_x: max_ox,
            min_y: min_oy,
            max_y: max_oy,
        })
    }
}

fn seam_energy_at(ctx: &SeamContext, h_add_inv: &Matrix3<f64>, x: u32, y: u32) -> Option<f64> {
    if ctx.pano_mask.get_pixel(x, y)[0] == 0 {
        return None;
    }
    let target = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
    let source = map_target_to_source(h_add_inv, target, ctx.img_to_add_info, ctx.projection)?;
    if source.x < 0.0
        || source.y < 0.0
        || source.x >= ctx.img_to_add.width() as f64 - 1.0
        || source.y >= ctx.img_to_add.height() as f64 - 1.0
        || !source_sample_has_support(ctx.img_to_add, source.x, source.y)
    {
        return None;
    }
    let base = ctx.pano.get_pixel(x, y);
    let candidate = apply_exposure_compensation(
        get_interpolated_pixel(ctx.img_to_add, source.x, source.y),
        ctx.exposure,
        x,
        y,
    );
    Some(
        ((base[0] as f64 - candidate[0] as f64).powi(2)
            + (base[1] as f64 - candidate[1] as f64).powi(2)
            + (base[2] as f64 - candidate[2] as f64).powi(2))
        .sqrt(),
    )
}

fn find_pairwise_seam_dp(
    ctx: &SeamContext,
    orientation: SeamOrientation,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
) -> Vec<i32> {
    const MAX_GRID_DIMENSION: u32 = 2_400;
    let (along_min, along_max, cross_min, cross_max, output_length) = match orientation {
        SeamOrientation::Vertical => (min_y, max_y, min_x, max_x, ctx.out_height),
        SeamOrientation::Horizontal => (min_x, max_x, min_y, max_y, ctx.out_width),
    };
    if along_min > along_max || cross_min > cross_max {
        return Vec::new();
    }

    let along_span = along_max - along_min;
    let cross_span = cross_max - cross_min;
    let step = along_span
        .max(cross_span)
        .div_ceil(MAX_GRID_DIMENSION)
        .max(1);
    let along_count = along_span.div_ceil(step) as usize + 1;
    let cross_count = cross_span.div_ceil(step) as usize + 1;
    if along_count < 2 || cross_count < 2 {
        return Vec::new();
    }
    println!(
        "    - Seam search grid: {}x{} ({}px sampling)",
        cross_count, along_count, step
    );

    let coordinate = |minimum: u32, maximum: u32, index: usize| {
        minimum
            .saturating_add((index as u32).saturating_mul(step))
            .min(maximum)
    };
    let Some(h_add_inv) = ctx.h_add.try_inverse() else {
        return Vec::new();
    };
    let mut previous = vec![f64::INFINITY; cross_count];
    let mut current = vec![f64::INFINITY; cross_count];
    let mut predecessors = vec![i8::MAX; along_count * cross_count];
    let mut last_active_index = None;
    let mut last_active_costs = Vec::new();

    for along_index in 0..along_count {
        current.fill(f64::INFINITY);
        let has_previous_path = previous.iter().any(|cost| cost.is_finite());
        let along = coordinate(along_min, along_max, along_index);
        for cross_index in 0..cross_count {
            let cross = coordinate(cross_min, cross_max, cross_index);
            let (x, y) = match orientation {
                SeamOrientation::Vertical => (cross, along),
                SeamOrientation::Horizontal => (along, cross),
            };
            let Some(mut energy) = seam_energy_at(ctx, &h_add_inv, x, y) else {
                continue;
            };

            // Discourage paths that merely trace a warped image border. Such paths make
            // rectangular exposure changes visible even when the geometry is correct.
            let edge_distance = cross_index.min(cross_count - 1 - cross_index) as f64;
            energy += (6.0 - edge_distance).max(0.0) * 0.01;

            if !has_previous_path {
                current[cross_index] = energy;
                predecessors[along_index * cross_count + cross_index] = 2;
                continue;
            }
            let first_neighbor = cross_index.saturating_sub(1);
            let last_neighbor = (cross_index + 1).min(cross_count - 1);
            let mut best_previous = f64::INFINITY;
            let mut best_index = cross_index;
            for (previous_index, &previous_cost) in previous
                .iter()
                .enumerate()
                .take(last_neighbor + 1)
                .skip(first_neighbor)
            {
                if previous_cost < best_previous {
                    best_previous = previous_cost;
                    best_index = previous_index;
                }
            }
            if best_previous.is_finite() {
                current[cross_index] = best_previous + energy;
                predecessors[along_index * cross_count + cross_index] =
                    (best_index as i32 - cross_index as i32) as i8;
            }
        }
        std::mem::swap(&mut previous, &mut current);
        if previous.iter().any(|cost| cost.is_finite()) {
            last_active_index = Some(along_index);
            last_active_costs.clone_from(&previous);
        }
    }

    let Some(last_active_index) = last_active_index else {
        return Vec::new();
    };
    let Some((mut current_cross, _)) = last_active_costs
        .iter()
        .enumerate()
        .filter(|(_, cost)| cost.is_finite())
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Vec::new();
    };
    let mut sampled_path = vec![0usize; along_count];
    sampled_path[last_active_index] = current_cross;
    let mut first_active_index = last_active_index;
    for along_index in (1..=last_active_index).rev() {
        let predecessor = predecessors[along_index * cross_count + current_cross];
        if predecessor == 2 {
            first_active_index = along_index;
            break;
        }
        if predecessor == i8::MAX {
            return Vec::new();
        }
        current_cross =
            (current_cross as i32 + predecessor as i32).clamp(0, cross_count as i32 - 1) as usize;
        sampled_path[along_index - 1] = current_cross;
        first_active_index = along_index - 1;
    }
    for index in 0..first_active_index {
        sampled_path[index] = sampled_path[first_active_index];
    }
    for index in (last_active_index + 1)..along_count {
        sampled_path[index] = sampled_path[last_active_index];
    }

    let sampled_cross: Vec<f64> = sampled_path
        .iter()
        .map(|&index| coordinate(cross_min, cross_max, index) as f64)
        .collect();
    let mut seam = vec![sampled_cross[0].round() as i32; output_length as usize];
    for along in along_min..=along_max {
        let relative = along - along_min;
        let lower_index = ((relative / step) as usize).min(along_count - 2);
        let lower_along = coordinate(along_min, along_max, lower_index);
        let upper_along = coordinate(along_min, along_max, lower_index + 1);
        let interpolation = if upper_along == lower_along {
            0.0
        } else {
            (along - lower_along) as f64 / (upper_along - lower_along) as f64
        };
        let cross = sampled_cross[lower_index] * (1.0 - interpolation)
            + sampled_cross[lower_index + 1] * interpolation;
        seam[along as usize] = cross.round() as i32;
    }
    let last_cross = sampled_cross.last().copied().unwrap_or(sampled_cross[0]);
    for value in seam.iter_mut().skip(along_max as usize + 1) {
        *value = last_cross.round() as i32;
    }
    seam
}

pub fn warp_image_homography(
    source: &Rgb32FImage,
    homography: &Matrix3<f64>,
    width: u32,
    height: u32,
) -> Rgb32FImage {
    assert!(width > 0 && height > 0, "warp output must be non-empty");
    let mut buffer = vec![0.0f32; (width as usize) * (height as usize) * 3];
    buffer
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let mapped = homography * Point3::new(x as f64, y as f64, 1.0);
                let pixel = if mapped.z.abs() < 1e-8 {
                    Rgb([0.0, 0.0, 0.0])
                } else {
                    get_high_quality_interpolated_pixel(
                        source,
                        mapped.x / mapped.z,
                        mapped.y / mapped.z,
                    )
                };
                let base = x as usize * 3;
                row[base] = pixel[0];
                row[base + 1] = pixel[1];
                row[base + 2] = pixel[2];
            }
        });
    Rgb32FImage::from_raw(width, height, buffer)
        .expect("warp buffer dimensions must match output image")
}

fn get_interpolated_pixel(img: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
    let (width, height) = img.dimensions();
    let x_floor = x.floor() as u32;
    let y_floor = y.floor() as u32;
    if x_floor + 1 >= width || y_floor + 1 >= height || x < 0.0 || y < 0.0 {
        return *img.get_pixel(
            x.max(0.0).min(width as f64 - 1.0) as u32,
            y.max(0.0).min(height as f64 - 1.0) as u32,
        );
    }
    let dx = x - x_floor as f64;
    let dy = y - y_floor as f64;
    let p00 = img.get_pixel(x_floor, y_floor);
    let p10 = img.get_pixel(x_floor + 1, y_floor);
    let p01 = img.get_pixel(x_floor, y_floor + 1);
    let p11 = img.get_pixel(x_floor + 1, y_floor + 1);
    let mut final_pixel = [0.0; 3];
    for i in 0..3 {
        let c00 = p00[i] as f64;
        let c10 = p10[i] as f64;
        let c01 = p01[i] as f64;
        let c11 = p11[i] as f64;
        let top = c00 * (1.0 - dx) + c10 * dx;
        let bottom = c01 * (1.0 - dx) + c11 * dx;
        final_pixel[i] = top * (1.0 - dy) + bottom * dy;
    }
    Rgb([
        final_pixel[0] as f32,
        final_pixel[1] as f32,
        final_pixel[2] as f32,
    ])
}

fn cubic_sample(p0: f64, p1: f64, p2: f64, p3: f64, amount: f64) -> f64 {
    p1 + 0.5
        * amount
        * (p2 - p0
            + amount * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + amount * (3.0 * (p1 - p2) + p3 - p0)))
}

/// Return whether a source coordinate has actual image support.
///
/// Warped virtual tiles are stored in a rectangular RGB buffer even when the
/// projective footprint is triangular or otherwise clipped.  Those uncovered
/// corners are represented by exact black pixels, so a rectangle-only bounds
/// check would incorrectly feed them into the seam and its low-frequency
/// pyramid.  Real artwork may contain very dark strokes, but a decoded image
/// sample still has non-zero energy in at least one channel; requiring a small
/// positive floor keeps those samples while rejecting the transparent zero
/// support used by the renderer.
#[inline]
fn source_sample_has_support(image: &Rgb32FImage, x: f64, y: f64) -> bool {
    const SUPPORT_EPSILON: f32 = 1e-6;
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= width as f64
        || y >= height as f64
    {
        return false;
    }
    // Use the nearest source sample rather than the interpolated value.  A
    // cubic sample near a transparent corner can be non-zero solely because
    // its support straddles the valid footprint; the nearest source pixel is
    // the stable ownership signal and avoids re-introducing that corner.
    let source_x = (x.floor() as u32).min(width - 1);
    let source_y = (y.floor() as u32).min(height - 1);
    let pixel = image.get_pixel(source_x, source_y);
    pixel.0.iter().all(|value| value.is_finite())
        && pixel.0.iter().any(|value| *value > SUPPORT_EPSILON)
}

pub(super) fn get_high_quality_interpolated_pixel(img: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
    let (width, height) = img.dimensions();
    if width < 4
        || height < 4
        || x < 1.0
        || y < 1.0
        || x >= width as f64 - 2.0
        || y >= height as f64 - 2.0
    {
        return get_interpolated_pixel(img, x, y);
    }

    let x_floor = x.floor() as i32;
    let y_floor = y.floor() as i32;
    let amount_x = x - x_floor as f64;
    let amount_y = y - y_floor as f64;
    if amount_x.abs() < 1e-9 && amount_y.abs() < 1e-9 {
        return *img.get_pixel(x_floor as u32, y_floor as u32);
    }

    let mut output = [0.0f32; 3];
    for (channel, output_channel) in output.iter_mut().enumerate() {
        let mut rows = [0.0f64; 4];
        let mut local_min = f64::INFINITY;
        let mut local_max = f64::NEG_INFINITY;
        for (row_index, sample_y) in ((y_floor - 1)..=(y_floor + 2)).enumerate() {
            let mut samples = [0.0f64; 4];
            for (column_index, sample_x) in ((x_floor - 1)..=(x_floor + 2)).enumerate() {
                let value = img.get_pixel(sample_x as u32, sample_y as u32)[channel] as f64;
                samples[column_index] = value;
                local_min = local_min.min(value);
                local_max = local_max.max(value);
            }
            rows[row_index] =
                cubic_sample(samples[0], samples[1], samples[2], samples[3], amount_x);
        }
        *output_channel = cubic_sample(rows[0], rows[1], rows[2], rows[3], amount_y)
            .clamp(local_min, local_max) as f32;
    }
    Rgb(output)
}

#[cfg(test)]
mod interpolation_tests {
    use super::*;

    fn geometry_test_image(id: usize, focal_length_35mm: Option<f64>) -> ImageInfo {
        ImageInfo {
            id,
            filename: format!("geometry-{id}.jpg"),
            width: 1_000,
            height: 800,
            alignment_image: GrayImage::new(1, 1),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }

    #[test]
    fn pixel_aligned_canvas_preserves_integer_reference_coordinates() {
        let (offset, size) = pixel_aligned_canvas(-123.4, 876.2);
        assert_eq!(offset, 124.0);
        assert_eq!(size, 1001);
        assert_eq!(offset.fract(), 0.0);
        assert!(offset - 123.4 >= 0.0);
    }

    #[test]
    fn virtual_tile_geometry_preserves_nonzero_world_origin() {
        let image = geometry_test_image(7, None);
        let image_ref = &image;
        let image_to_world = Matrix3::new(1.0, 0.0, -123.4, 0.0, 1.0, 55.2, 0.0, 0.0, 1.0);
        let homographies = HashMap::from([(image.id, image_to_world)]);
        let geometry = focus_stack_virtual_tile_geometry(
            &[image_ref],
            &homographies,
            Projection::Planar,
            None,
        )
        .expect("translated source must have valid Virtual_Tile geometry");
        assert_eq!(geometry.tile_to_world[(0, 2)], -124.0);
        assert_eq!(geometry.tile_to_world[(1, 2)], 55.0);
        for source in [Point2::new(0.0, 0.0), Point2::new(999.0, 799.0)] {
            let world = image_to_world * Point3::new(source.x, source.y, 1.0);
            let tile = Point2::new(world.x + 124.0, world.y - 55.0);
            let recovered = geometry.tile_to_world * Point3::new(tile.x, tile.y, 1.0);
            assert!((recovered.x - world.x).abs() < 1e-9);
            assert!((recovered.y - world.y).abs() < 1e-9);
        }
    }

    #[test]
    fn cubic_warp_sampling_preserves_more_edge_contrast_than_bilinear() {
        let image = Rgb32FImage::from_fn(8, 8, |x, _| {
            let value = if x < 4 { 0.0 } else { 1.0 };
            Rgb([value, value, value])
        });
        let bilinear_low = get_interpolated_pixel(&image, 3.25, 3.5)[0];
        let bilinear_high = get_interpolated_pixel(&image, 3.75, 3.5)[0];
        let cubic_low = get_high_quality_interpolated_pixel(&image, 3.25, 3.5)[0];
        let cubic_high = get_high_quality_interpolated_pixel(&image, 3.75, 3.5)[0];

        assert!(cubic_high - cubic_low > bilinear_high - bilinear_low);
        assert!((0.0..=1.0).contains(&cubic_low));
        assert!((0.0..=1.0).contains(&cubic_high));
    }

    #[test]
    fn focus_box_blur_preserves_a_constant_score_map() {
        let source = vec![3.5f32; 5 * 4];
        let blurred = box_blur_focus_map(&source, 5, 4, 2);

        assert!(blurred.iter().all(|value| (*value - 3.5).abs() < 1e-6));
    }

    #[test]
    fn overlap_exposure_ignores_coloured_artwork_samples() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 64;
        let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.60, 0.50, 0.40]));
        let candidate = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
            if x < WIDTH * 3 / 4 {
                // A saturated red seal must not determine the paper gain.
                Rgb([1.0, 0.02, 0.01])
            } else {
                Rgb([0.60, 0.50, 0.40])
            }
        });
        let mut info = geometry_test_image(0, None);
        info.width = WIDTH;
        info.height = HEIGHT;
        let panorama_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let compensation = estimate_overlap_exposure_compensation(ExposureOverlap {
            panorama: &base,
            panorama_mask: &panorama_mask,
            candidate: &info,
            candidate_image: &candidate,
            candidate_inverse: &Matrix3::identity(),
            projection: Projection::Planar,
            offset_x: 0.0,
            offset_y: 0.0,
        });

        assert!(
            (compensation.representative_gain - 1.0).abs() < 0.02,
            "artwork contaminated exposure estimate: gain={:.3}",
            compensation.representative_gain
        );
    }

    #[test]
    fn overlap_exposure_recovers_bounded_rgb_channel_gain() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 64;
        let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.66, 0.52, 0.40]));
        let candidate = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.60, 0.55, 0.44]));
        let mut info = geometry_test_image(0, None);
        info.width = WIDTH;
        info.height = HEIGHT;
        let panorama_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let compensation = estimate_overlap_exposure_compensation(ExposureOverlap {
            panorama: &base,
            panorama_mask: &panorama_mask,
            candidate: &info,
            candidate_image: &candidate,
            candidate_inverse: &Matrix3::identity(),
            projection: Projection::Planar,
            offset_x: 0.0,
            offset_y: 0.0,
        });
        let gains = compensation.representative_channel_gain;
        assert!((gains[0] - 1.10).abs() < 0.02, "red gain={:.3}", gains[0]);
        assert!((gains[1] - 0.95).abs() < 0.02, "green gain={:.3}", gains[1]);
        assert!((gains[2] - 0.91).abs() < 0.02, "blue gain={:.3}", gains[2]);
    }

    #[test]
    fn progressive_exposure_damping_keeps_direction_without_chain_drift() {
        let compensation = ExposureCompensation {
            cell_size: 1,
            grid_width: 1,
            grid_height: 1,
            gains: vec![0.75],
            representative_gain: 0.75,
            channel_gains: vec![[0.75, 1.20, 1.0]],
            representative_channel_gain: [0.75, 1.20, 1.0],
        }
        .damped(0.45);
        let expected = 0.75f32.ln().mul_add(0.45, 0.0).exp();
        assert!((compensation.representative_gain - expected).abs() < 1e-6);
        assert!(compensation.representative_gain < 1.0);
        assert!(compensation.representative_gain > 0.85);
        assert!(compensation.representative_channel_gain[1] > 1.0);
        assert_eq!(compensation.gain_at(0, 0), compensation.representative_gain);
    }

    #[test]
    fn seam_blend_rejects_a_triangular_zero_support_corner() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 64;
        let paper = Rgb([0.62, 0.51, 0.43]);
        let mut panorama = Rgb32FImage::from_pixel(WIDTH, HEIGHT, paper);
        let mut panorama_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let candidate = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            if x + y < 28 {
                Rgb([0.0, 0.0, 0.0])
            } else {
                paper
            }
        });
        let mut info = geometry_test_image(0, None);
        info.width = WIDTH;
        info.height = HEIGHT;
        let exposure = ExposureCompensation {
            cell_size: 1,
            grid_width: 1,
            grid_height: 1,
            gains: vec![1.0],
            representative_gain: 1.0,
            channel_gains: vec![[1.0; 3]],
            representative_channel_gain: [1.0; 3],
        };
        let seam_coords = vec![HEIGHT as i32 / 2; WIDTH as usize];
        blend_panorama_seam_band(SeamBandBlend {
            panorama: &mut panorama,
            panorama_mask: &mut panorama_mask,
            img_to_add_info: &info,
            img_to_add: &candidate,
            h_add: &Matrix3::identity(),
            projection: Projection::Planar,
            offset_x: 0.0,
            offset_y: 0.0,
            orientation: SeamOrientation::Horizontal,
            seam_coords: &seam_coords,
            new_image_is_dominant_side: true,
            min_x: 0,
            max_x: WIDTH - 1,
            min_y: 0,
            max_y: HEIGHT - 1,
            exposure: &exposure,
        });

        let maximum_error = panorama
            .as_raw()
            .iter()
            .zip(std::iter::repeat(&paper.0).flatten())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            maximum_error < 1e-4,
            "zero-support corner leaked into the seam pyramid: error={maximum_error}"
        );
        assert!(!source_sample_has_support(&candidate, 4.0, 4.0));
        assert!(source_sample_has_support(&candidate, 40.0, 40.0));
    }

    #[test]
    fn focus_tile_tie_break_is_stable_and_quality_first() {
        assert!(focus_tile_ownership_should_replace(0, 0, 1, 3));
        assert!(focus_tile_ownership_should_replace(80, 4, 81, 9));
        assert!(!focus_tile_ownership_should_replace(81, 4, 80, 1));
        // Equal edge distance must resolve to the same tile identity even if
        // the caller's loading order changes.
        assert!(focus_tile_ownership_should_replace(120, 9, 120, 4));
        assert!(!focus_tile_ownership_should_replace(120, 4, 120, 9));
    }

    #[test]
    fn focus_score_prefers_native_edges_over_a_blurred_layer() {
        const WIDTH: u32 = 128;
        const HEIGHT: u32 = 128;
        let sharp = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            // Use aperiodic, high-frequency structure so the metric cannot
            // win by matching a single periodic phase after blur.
            let cell = ((x / 3 + y / 5 + (x * 7 + y * 11) % 5) & 1) as f32;
            let value = 0.18 + cell * 0.62;
            Rgb([value, value * 0.92, value * 0.84])
        });
        let blurred = image::imageops::blur(&sharp, 2.2);
        let mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let sharp_score = focus_score_map(&sharp, &mask);
        let blurred_score = focus_score_map(&blurred, &mask);
        let interior = |scores: &[f32]| {
            let mut values = Vec::new();
            for y in 8..(HEIGHT - 8) as usize {
                let row = y * WIDTH as usize;
                values.extend_from_slice(&scores[row + 8..row + (WIDTH - 8) as usize]);
            }
            values.iter().copied().sum::<f32>() / values.len().max(1) as f32
        };
        let sharp_mean = interior(&sharp_score);
        let blurred_mean = interior(&blurred_score);
        assert!(
            sharp_mean > blurred_mean * 1.8,
            "focus metric did not separate sharp={sharp_mean:.5} from blurred={blurred_mean:.5}"
        );
    }

    #[test]
    fn focus_analysis_keeps_source_sampling_density_on_long_mosaics() {
        let (analysis_width, analysis_height) = focus_analysis_dimensions(31_121, 11_034, 9_504);

        assert!(analysis_width >= 4_500);
        assert!(analysis_height >= 1_500);
        assert!(
            u64::from(analysis_width) * u64::from(analysis_height) <= FOCUS_ANALYSIS_MAX_PIXELS
        );
    }

    #[test]
    fn shifted_mosaic_detection_catches_a_lens_scale_change() {
        let first = geometry_test_image(0, Some(35.0));
        let second = geometry_test_image(1, Some(85.0));
        let center_x = first.width as f64 * 0.5;
        let center_y = first.height as f64 * 0.5;
        let scale = 1.25;
        let scale_about_center = Matrix3::new(
            scale,
            0.0,
            center_x * (1.0 - scale),
            0.0,
            scale,
            center_y * (1.0 - scale),
            0.0,
            0.0,
            1.0,
        );
        let homographies = HashMap::from([(0, Matrix3::identity()), (1, scale_about_center)]);
        let images = vec![&first, &second];

        assert!(focus_stack_is_shifted_mosaic(
            &images,
            &homographies,
            Projection::Planar
        ));
    }

    #[test]
    fn shifted_mosaic_detection_uses_focal_metadata_when_geometry_is_centered() {
        let first = geometry_test_image(0, Some(35.0));
        let second = geometry_test_image(1, Some(85.0));
        let homographies = HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())]);
        let images = vec![&first, &second];

        assert!(focus_stack_is_shifted_mosaic(
            &images,
            &homographies,
            Projection::Planar
        ));
    }

    #[test]
    fn shifted_mosaic_detection_keeps_small_focus_breathing_on_the_focus_path() {
        let first = geometry_test_image(0, Some(85.0));
        let second = geometry_test_image(1, Some(85.0));
        let center_x = first.width as f64 * 0.5;
        let center_y = first.height as f64 * 0.5;
        let scale = 1.025;
        let small_breathing = Matrix3::new(
            scale,
            0.0,
            center_x * (1.0 - scale) + 18.0,
            0.0,
            scale,
            center_y * (1.0 - scale) + 12.0,
            0.0,
            0.0,
            1.0,
        );
        let homographies = HashMap::from([(0, Matrix3::identity()), (1, small_breathing)]);
        let images = vec![&first, &second];

        assert!(!focus_stack_is_shifted_mosaic(
            &images,
            &homographies,
            Projection::Planar
        ));
    }

    #[test]
    fn parallel_pyramid_downsample_preserves_constant_pixels_and_odd_edges() {
        let source = Rgb32FImage::from_pixel(5, 7, Rgb([0.2, 0.4, 0.8]));
        let downsampled = downsample_rgb_half(&source);

        assert_eq!(downsampled.dimensions(), (3, 4));
        assert!(downsampled.pixels().all(|pixel| pixel.0 == [0.2, 0.4, 0.8]));
    }

    #[test]
    fn parallel_pyramid_detail_reconstructs_the_original_pixels() {
        let source = Rgb32FImage::from_fn(17, 11, |x, y| {
            let value = (x as f32 * 0.07 + y as f32 * 0.03).sin();
            Rgb([value, value * 0.5, 1.0 - value])
        });
        let coarse = downsample_rgb_half(&source);
        let detail = subtract_upsampled_rgb(&source, &coarse);
        let reconstructed = upsample_and_add_rgb(&coarse, &detail);

        assert_eq!(reconstructed.dimensions(), source.dimensions());
        let maximum_error = reconstructed
            .as_raw()
            .iter()
            .zip(source.as_raw())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 1e-6, "maximum error: {maximum_error}");
    }

    #[test]
    fn panorama_detail_feather_is_local_and_symmetric() {
        let radius = PANORAMA_DETAIL_SEAM_FEATHER_RADIUS;

        assert_eq!(panorama_detail_alpha(f32::NEG_INFINITY), 0.0);
        assert_eq!(panorama_detail_alpha(-radius), 0.0);
        assert_eq!(panorama_detail_alpha(0.0), 0.5);
        assert_eq!(panorama_detail_alpha(radius), 1.0);
        assert_eq!(panorama_detail_alpha(f32::INFINITY), 1.0);
    }

    #[test]
    fn panorama_global_tone_blend_preserves_medium_frequency_detail() {
        const WIDTH: u32 = 256;
        const HEIGHT: u32 = 64;
        let pattern = |x: u32| if (x / 4).is_multiple_of(2) { 0.2 } else { 0.8 };
        let base = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
            let value = pattern(x);
            Rgb([value, value, value])
        });
        let candidate = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
            let value = (pattern((x + 4).min(WIDTH - 1)) + 0.08).min(1.0);
            Rgb([value, value, value])
        });
        let seam_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
            image::Luma([if x >= WIDTH / 2 { 255 } else { 0 }])
        });
        let tone_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
            image::Luma([((x as f32 / (WIDTH - 1) as f32) * 255.0).round() as u8])
        });
        let blended = multiband_blend(
            base.clone(),
            candidate,
            seam_mask,
            Some(tone_mask),
            PANORAMA_BLEND_BANDS,
            false,
        );
        let horizontal_variation = |image: &Rgb32FImage| {
            (32..96)
                .map(|x| {
                    (image.get_pixel(x, HEIGHT / 2)[0] - image.get_pixel(x - 1, HEIGHT / 2)[0])
                        .abs()
                })
                .sum::<f32>()
        };

        let source_variation = horizontal_variation(&base);
        let blended_variation = horizontal_variation(&blended);
        assert!(
            blended_variation >= source_variation * 0.9,
            "medium-frequency contrast fell from {source_variation:.3} to {blended_variation:.3}",
        );
    }

    #[test]
    fn focus_translation_recovers_a_non_grid_aligned_displacement() {
        const WIDTH: u32 = 800;
        const HEIGHT: u32 = 800;
        const LEFT: i32 = 21;
        const TOP: i32 = 17;
        const SHIFT_X: i32 = 19;
        const SHIFT_Y: i32 = 11;
        let pattern = |x: i32, y: i32| {
            let x = x as f32;
            let y = y as f32;
            // A deterministic aperiodic texture keeps the correspondence
            // unique instead of letting a periodic texture pass the margin at
            // another offset.
            let xi = x as i32;
            let yi = y as i32;
            let hash = ((xi.wrapping_mul(73_856_093)
                ^ yi.wrapping_mul(19_349_663)
                ^ (xi.wrapping_mul(yi)).wrapping_mul(83_492_791))
                & 255) as f32
                / 255.0;
            let wave = (x * 0.017 + y * 0.043).sin() * 0.12;
            (0.12 + hash * 0.72 + wave).clamp(0.03, 0.97)
        };
        let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            let value = pattern(x as i32, y as i32);
            Rgb([value, value * 0.91, value * 0.78])
        });
        let merged_image = Rgb32FImage::from_fn(WIDTH + 80, HEIGHT + 80, |x, y| {
            let source_x = x as i32 - LEFT - SHIFT_X;
            let source_y = y as i32 - TOP - SHIFT_Y;
            let value = pattern(source_x, source_y);
            Rgb([value, value * 0.91, value * 0.78])
        });
        let candidate = RenderedFocusLayer {
            image: candidate_image,
            mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            left: LEFT as u32,
            top: TOP as u32,
        };
        let merged_mask = GrayImage::from_pixel(WIDTH + 80, HEIGHT + 80, image::Luma([255]));

        let result = estimate_focus_layer_translation(&candidate, &merged_image, &merged_mask)
            .expect("synthetic overlap should produce a translation");

        assert_eq!((result.0, result.1), (SHIFT_X, SHIFT_Y));
        assert!(result.3 >= 8, "insufficient patch consensus: {:?}", result);
    }

    #[test]
    fn focus_translation_accepts_aligned_layers_without_forcing_motion() {
        const WIDTH: u32 = 640;
        const HEIGHT: u32 = 640;
        let pattern = |x: i32, y: i32| {
            let v = ((x * 17 + y * 29).rem_euclid(97) as f32) / 97.0;
            0.15 + 0.7 * v
        };
        let image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            let value = pattern(x as i32, y as i32);
            Rgb([value, value * 0.8, value * 0.65])
        });
        let candidate = RenderedFocusLayer {
            image: image.clone(),
            mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            left: 0,
            top: 0,
        };
        let mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));

        let result = estimate_focus_layer_translation(&candidate, &image, &mask);

        assert!(
            result.is_none(),
            "already aligned layers should remain untouched: {result:?}"
        );
    }

    #[test]
    fn focus_decision_prefers_the_locally_sharper_layer() {
        let base_focus = vec![0.9, 0.8, 0.1, 0.1];
        let candidate_focus = vec![0.1, 0.1, 0.8, 0.9];
        let base_mask = GrayImage::from_pixel(4, 1, image::Luma([255]));
        let candidate_mask = GrayImage::from_pixel(4, 1, image::Luma([255]));

        let decision =
            focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

        assert_eq!(decision.as_raw(), &[0, 0, 255, 255]);
    }

    #[test]
    fn focus_decision_extends_clear_structure_over_an_adjacent_halo() {
        let base_focus = vec![0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2];
        let candidate_focus = vec![0.1, 0.1, 0.1, 0.8, 0.8, 0.22, 0.2];
        let base_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));
        let candidate_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));

        let decision =
            focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

        assert_eq!(decision.as_raw()[2], 0);
        assert_eq!(decision.as_raw()[4], 255);
        assert_eq!(decision.as_raw()[5], 255);
    }

    #[test]
    fn focus_decision_protects_base_structure_symmetrically() {
        let base_focus = vec![0.1, 0.1, 0.8, 0.8, 0.22, 0.2, 0.2];
        let candidate_focus = vec![0.9, 0.9, 0.1, 0.1, 0.2, 0.2, 0.2];
        let base_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));
        let candidate_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));

        let decision =
            focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

        assert_eq!(decision.as_raw()[0], 255);
        assert_eq!(decision.as_raw()[2], 0);
        assert_eq!(decision.as_raw()[4], 0);
    }

    #[test]
    fn focus_canvas_switch_suppression_keeps_canvas_owner_but_allows_ink() {
        let mut decision = GrayImage::from_pixel(2, 1, image::Luma([255]));
        let base = Rgb32FImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgb([0.32, 0.28, 0.24])
            } else {
                Rgb([0.05, 0.05, 0.05])
            }
        });
        let candidate = Rgb32FImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgb([0.40, 0.36, 0.32])
            } else {
                Rgb([0.01, 0.01, 0.01])
            }
        });
        let mask = GrayImage::from_pixel(2, 1, image::Luma([255]));

        suppress_focus_canvas_switches(&mut decision, &base, &candidate, &mask, &mask);

        assert_eq!(decision.as_raw(), &[0, 255]);
    }

    #[test]
    fn focus_seam_keeps_painted_detail_on_one_source() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 64;
        let canvas = Rgb([0.32, 0.28, 0.24]);
        let candidate_canvas = Rgb([0.40, 0.36, 0.32]);
        let mut base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, canvas);
        let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            if x == 32 && (16..48).contains(&y) {
                Rgb([0.01, 0.01, 0.01])
            } else {
                candidate_canvas
            }
        });
        let candidate = RenderedFocusLayer {
            image: candidate_image,
            mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            left: 0,
            top: 0,
        };
        let mut base_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let merged_foreground_mask = GrayImage::new(WIDTH, HEIGHT);
        let decision = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
            image::Luma([u8::from(x >= WIDTH / 2) * 255])
        });

        blend_focus_seam_band(
            &mut base,
            &mut base_mask,
            &merged_foreground_mask,
            &candidate,
            &decision,
        );

        assert_eq!(base.get_pixel(32, 32).0, [0.01, 0.01, 0.01]);
    }

    #[test]
    fn hard_focus_selection_copies_source_pixels_without_averaging() {
        let mut base =
            Rgb32FImage::from_raw(3, 1, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
        let candidate =
            Rgb32FImage::from_raw(3, 1, vec![0.0, 0.0, 1.0, 0.2, 0.3, 0.9, 0.7, 0.1, 0.8]).unwrap();
        let mut base_mask = GrayImage::from_raw(3, 1, vec![255, 255, 0]).unwrap();
        let candidate_mask = GrayImage::from_pixel(3, 1, image::Luma([255]));
        let decision = GrayImage::from_raw(3, 1, vec![0, 255, 0]).unwrap();

        hard_select_focus_pixels(
            &mut base,
            &mut base_mask,
            &candidate,
            &candidate_mask,
            &decision,
        );

        assert_eq!(base.get_pixel(0, 0).0, [1.0, 0.0, 0.0]);
        assert_eq!(base.get_pixel(1, 0).0, [0.2, 0.3, 0.9]);
        assert_eq!(base.get_pixel(2, 0).0, [0.7, 0.1, 0.8]);
        assert_eq!(base_mask.as_raw(), &[255, 255, 255]);
    }

    #[test]
    fn focus_color_correction_removes_a_measured_channel_gain() {
        let base = Rgb32FImage::from_pixel(40, 16, Rgb([0.4, 0.5, 0.6]));
        let mut candidate_image = Rgb32FImage::from_pixel(32, 16, Rgb([0.5, 0.625, 0.75]));
        let base_mask = GrayImage::from_pixel(40, 16, image::Luma([255]));
        let candidate = RenderedFocusLayer {
            image: candidate_image.clone(),
            mask: GrayImage::from_pixel(32, 16, image::Luma([255])),
            foreground_mask: GrayImage::new(32, 16),
            relaxed_foreground_mask: GrayImage::new(32, 16),
            left: 4,
            top: 0,
        };

        let correction = estimate_focus_color_correction(
            &base,
            &base_mask,
            &GrayImage::new(40, 16),
            &candidate,
            false,
        );
        assert!(
            correction
                .gains
                .iter()
                .all(|gain| (*gain - 0.8).abs() < 0.01)
        );

        apply_focus_color_correction(
            &mut candidate_image,
            &correction,
            &GrayImage::new(32, 16),
            false,
        );
        let corrected = candidate_image.get_pixel(0, 0);
        assert!((corrected[0] - 0.4).abs() < 0.01);
        assert!((corrected[1] - 0.5).abs() < 0.01);
        assert!((corrected[2] - 0.6).abs() < 0.01);
    }

    #[test]
    fn focus_color_correction_uses_only_consensus_foreground_overlap() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 32;
        let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.4, 0.4, 0.4]));
        let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |_, y| {
            if y < 24 {
                Rgb([0.5, 0.5, 0.5])
            } else {
                Rgb([0.4, 0.4, 0.4])
            }
        });
        let foreground_mask = GrayImage::from_fn(WIDTH, HEIGHT, |_, y| {
            image::Luma([if y < 24 { 255 } else { 0 }])
        });
        let candidate = RenderedFocusLayer {
            image: candidate_image,
            mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            foreground_mask: foreground_mask.clone(),
            relaxed_foreground_mask: foreground_mask.clone(),
            left: 0,
            top: 0,
        };

        let correction = estimate_focus_color_correction(
            &base,
            &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            &foreground_mask,
            &candidate,
            false,
        );

        assert!(
            correction
                .gains
                .iter()
                .all(|gain| (*gain - 0.8).abs() < 0.01)
        );
    }

    #[test]
    fn focus_color_correction_can_use_bright_consensus_edge_pixels() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 32;
        let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.94, 0.94, 0.94]));
        let candidate = RenderedFocusLayer {
            image: Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.98, 0.98, 0.98])),
            mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            foreground_mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            relaxed_foreground_mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            left: 0,
            top: 0,
        };

        let correction = estimate_focus_color_correction(
            &base,
            &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            &candidate,
            false,
        );

        assert!(
            correction
                .gains
                .iter()
                .all(|gain| (*gain - (0.94 / 0.98)).abs() < 0.01)
        );
    }

    #[test]
    fn focus_color_correction_reduces_a_broad_local_gain_step() {
        const WIDTH: u32 = 1_536;
        const HEIGHT: u32 = 32;
        let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.4, 0.5, 0.6]));
        let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
            if x < WIDTH / 2 {
                Rgb([0.5, 0.625, 0.75])
            } else {
                Rgb([0.4, 0.5, 0.6])
            }
        });
        let candidate = RenderedFocusLayer {
            image: candidate_image.clone(),
            mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            left: 0,
            top: 0,
        };
        let base_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let correction = estimate_focus_color_correction(
            &base,
            &base_mask,
            &GrayImage::new(WIDTH, HEIGHT),
            &candidate,
            false,
        );
        assert!(correction.spatial_gains.is_some());

        let mut corrected = candidate_image;
        apply_focus_color_correction(
            &mut corrected,
            &correction,
            &GrayImage::new(WIDTH, HEIGHT),
            false,
        );
        let input_step = 0.5f32 - 0.4;
        let corrected_step =
            (corrected.get_pixel(100, 10)[0] - corrected.get_pixel(1_300, 10)[0]).abs();
        assert!(corrected_step < input_step * 0.8);
    }

    #[test]
    fn focus_background_tone_harmonization_reduces_step_without_touching_foreground() {
        const WIDTH: u32 = 192;
        const HEIGHT: u32 = 96;
        let mut image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            if (80..112).contains(&x) && (32..64).contains(&y) {
                Rgb([0.65, 0.10, 0.08])
            } else if x < WIDTH / 2 {
                Rgb([0.32, 0.28, 0.24])
            } else {
                Rgb([0.40, 0.36, 0.32])
            }
        });
        let original_foreground_pixel = *image.get_pixel(96, 48);
        let image_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let foreground_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
            image::Luma([u8::from((80..112).contains(&x) && (32..64).contains(&y)) * 255])
        });

        let input_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();
        harmonize_focus_background_tone(&mut image, &image_mask, &foreground_mask);
        let output_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();

        assert!(output_step < input_step * 0.5);
        assert_eq!(*image.get_pixel(96, 48), original_foreground_pixel);
    }

    #[test]
    fn focus_owner_tone_harmonization_matches_source_blocks_without_touching_detail() {
        const WIDTH: u32 = 192;
        const HEIGHT: u32 = 96;
        let mut image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            if (80..112).contains(&x) && (32..64).contains(&y) {
                Rgb([0.65, 0.10, 0.08])
            } else if x < WIDTH / 2 {
                Rgb([0.32, 0.28, 0.24])
            } else {
                Rgb([0.40, 0.36, 0.32])
            }
        });
        let original_foreground_pixel = *image.get_pixel(96, 48);
        let image_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let foreground_mask = GrayImage::new(WIDTH, HEIGHT);
        let owner_map = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
            image::Luma([if x < WIDTH / 2 { 1 } else { 2 }])
        });

        let input_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();
        harmonize_focus_background_tone_with_owners(
            &mut image,
            &image_mask,
            &foreground_mask,
            &owner_map,
        );
        let output_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();

        assert!(output_step < input_step * 0.5);
        assert_eq!(*image.get_pixel(96, 48), original_foreground_pixel);
    }

    #[test]
    fn focus_canvas_margin_fill_keeps_the_full_canvas_and_extends_texture() {
        let mut image = Rgb32FImage::from_fn(8, 6, |x, y| {
            let value = (x as f32 * 0.07 + y as f32 * 0.11).fract();
            Rgb([value, value * 0.8, value * 0.6])
        });
        let mut mask = GrayImage::new(8, 6);
        for y in 1..5 {
            for x in 1..7 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }

        let dimensions = image.dimensions();
        let (filled, remaining) = fill_focus_canvas_margins(&mut image, &mut mask);

        assert_eq!(image.dimensions(), dimensions);
        assert!(filled > 0);
        assert_eq!(remaining, 0);
        assert!(mask.as_raw().iter().all(|&value| value > 0));
        assert!(image.get_pixel(0, 0).0.iter().any(|&value| value != 0.0));
    }
}

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
mod station_degradation_wiring_tests {
    use super::*;
    use crate::panorama_utils::stack_pipeline::degradation::INTRA_STATION_REGISTRATION_FAILED;
    use crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode;

    fn station_frame(id: usize) -> ImageInfo {
        ImageInfo {
            id,
            filename: format!("/station/000{id}.nef"),
            width: 6_000,
            height: 4_000,
            alignment_image: GrayImage::new(1, 1),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }

    fn record(
        path: &str,
        status: IntraStationFrameStatus,
        inliers: usize,
        coverage: f64,
        error_px: f64,
    ) -> (String, IntraStationFrameRecord) {
        (
            path.to_string(),
            IntraStationFrameRecord {
                path: path.to_string(),
                inliers,
                inlier_area_coverage: coverage,
                median_symmetric_error_px: error_px,
                status,
                ..Default::default()
            },
        )
    }

    #[test]
    fn a_station_with_no_registration_record_fuses_every_frame_in_order() {
        // This is the no-change guarantee of the wiring: the default path does
        // not run the native patch refinement, so it records nothing, so the
        // plan is the full bracket and the render loop is exactly what it was.
        let frames = (1..=4).map(station_frame).collect::<Vec<_>>();
        let images = frames.iter().collect::<Vec<_>>();
        let (members, rejections) = station_members_from_frame_records(&images, &HashMap::new());
        assert!(rejections.is_empty());
        assert_eq!(members[0].status, IntraStationFrameStatus::Anchor);
        assert!(
            members[1..]
                .iter()
                .all(|member| member.status == IntraStationFrameStatus::GlobalFallback),
            "a frame placed by its global model alone is a global fallback, not a failure"
        );
        let plan = plan_station_fusion(&members);
        assert_eq!(plan.mode, StationFusionMode::AllFrames);
        assert_eq!(plan.fused, vec![0, 1, 2, 3]);
        assert!(plan.excluded.is_empty());
        assert!(station_plan_entries(0, &members, &plan).is_empty());
    }

    #[test]
    fn a_recorded_failure_beside_two_survivors_leaves_the_fusion_to_the_survivors() {
        // 需求 12.2.
        let frames = (1..=4).map(station_frame).collect::<Vec<_>>();
        let images = frames.iter().collect::<Vec<_>>();
        let recorded = HashMap::from([
            record(
                "/station/0002.nef",
                IntraStationFrameStatus::Failed,
                0,
                0.05,
                0.4,
            ),
            record(
                "/station/0003.nef",
                IntraStationFrameStatus::Local,
                900,
                0.80,
                0.4,
            ),
        ]);
        let (members, rejections) = station_members_from_frame_records(&images, &recorded);
        assert!(
            rejections.is_empty(),
            "no accepted inlier means no measured 需求 12.5 evidence to judge"
        );
        let plan = plan_station_fusion(&members);
        assert_eq!(plan.mode, StationFusionMode::SuccessfulFramesOnly);
        assert_eq!(
            plan.fused,
            vec![0, 2, 3],
            "the excluded frame is never offered to the cut"
        );
        assert_eq!(
            plan.excluded
                .iter()
                .map(|record| (record.path.as_str(), record.reason.as_str()))
                .collect::<Vec<_>>(),
            vec![("/station/0002.nef", INTRA_STATION_REGISTRATION_FAILED)],
            "需求 12.2 names the excluded Source_RAW by absolute path"
        );
        let entries = station_plan_entries(7, &members, &plan);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, INTRA_STATION_REGISTRATION_FAILED);
        assert_eq!(entries[0].1["station_index"], serde_json::json!(7));
        assert_eq!(entries[0].1["fused_frames"], serde_json::json!(3));
    }

    #[test]
    fn every_non_anchor_frame_failing_degrades_the_station_to_one_source() {
        // 需求 12.1: the plan keeps one member, so the fusion is constructed with
        // one frame and reports `SingleFrame`, and every covered pixel is copied
        // from that Source_RAW.
        let frames = (1..=3).map(station_frame).collect::<Vec<_>>();
        let images = frames.iter().collect::<Vec<_>>();
        let recorded = HashMap::from([
            record(
                "/station/0002.nef",
                IntraStationFrameStatus::Failed,
                0,
                0.01,
                9.0,
            ),
            record(
                "/station/0003.nef",
                IntraStationFrameStatus::Failed,
                0,
                0.02,
                9.0,
            ),
        ]);
        let (members, _) = station_members_from_frame_records(&images, &recorded);
        let plan = plan_station_fusion(&members);
        assert_eq!(plan.mode, StationFusionMode::SingleFrameDegraded);
        assert_eq!(plan.fused, vec![0]);
        assert_eq!(
            plan.single_frame_path(&members).as_deref(),
            Some("/station/0001.nef")
        );
        assert_eq!(plan.excluded.len(), 2);
        assert_eq!(
            focus_fuser::StationFusion::new(64, 64, plan.fused.len()).solver_status(),
            crate::panorama_utils::stack_pipeline::report::FusionSolverStatus::SingleFrame,
            "需求 12.1 / 3.10: one frame has nothing to cut against"
        );
    }

    #[test]
    fn recorded_evidence_below_the_group_join_limits_rejects_the_frame() {
        // 需求 12.5, on a 6000px anchor: support below 0.20, or a median
        // symmetric error above 60px, keeps the Source_RAW out of the station.
        let frames = (1..=3).map(station_frame).collect::<Vec<_>>();
        let images = frames.iter().collect::<Vec<_>>();
        let recorded = HashMap::from([
            record(
                "/station/0002.nef",
                IntraStationFrameStatus::Local,
                120,
                0.1999,
                0.5,
            ),
            record(
                "/station/0003.nef",
                IntraStationFrameStatus::Local,
                900,
                0.85,
                60.1,
            ),
        ]);
        let (members, rejections) = station_members_from_frame_records(&images, &recorded);
        assert_eq!(rejections.len(), 2);
        assert!(rejections[0].support_below_minimum && !rejections[0].error_above_limit);
        assert!(rejections[1].error_above_limit && !rejections[1].support_below_minimum);
        assert!((rejections[1].symmetric_error_limit_px - 60.0).abs() < 1e-12);
        assert!(
            members[1..]
                .iter()
                .all(|member| member.status == IntraStationFrameStatus::Failed),
            "a frame that joins no Capture_Station cannot be fused by this one"
        );
        // A frame exactly at both limits still joins, so the thresholds are not
        // loosened in either direction.
        let recorded = HashMap::from([
            record(
                "/station/0002.nef",
                IntraStationFrameStatus::Local,
                120,
                0.20,
                60.0,
            ),
            record(
                "/station/0003.nef",
                IntraStationFrameStatus::Local,
                900,
                0.85,
                0.5,
            ),
        ]);
        let (members, rejections) = station_members_from_frame_records(&images, &recorded);
        assert!(rejections.is_empty());
        assert_eq!(
            plan_station_fusion(&members).mode,
            StationFusionMode::AllFrames
        );
    }
}
