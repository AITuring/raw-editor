//! Planar scan mosaic for flattened source tiles (Photoshop PSD composites).
//!
//! One PSD is one already-fused capture station of a flat artwork. Its
//! neighbours overlap it heavily but differ in focus, magnification and
//! illumination, and the feature graph only knows their poses from 2400px
//! proxies, so the global poses still disagree by several native pixels.
//! Averaging two such tiles across an overlap doubles or softens brush and
//! silk detail. This compositor instead
//!
//! 1. refines the planar homographies with native-resolution patch
//!    correspondences and one joint robust least-squares adjustment, then
//!    fits every tile a smooth correction on a control grid for what a
//!    planar pose cannot follow (lens distortion, a scroll that is not
//!    perfectly flat), validated on held-out correspondences;
//! 2. renders on an output grid on which the highest-resolution tile is not
//!    downsampled (bounded by the in-memory pixel budget);
//! 3. chooses one owner per output cell by the detail each tile delivers in
//!    *output* pixels, with seams priced by local structure (the multi-label
//!    graph cut shared with the focus fuser);
//! 4. writes the owner's high-frequency detail with a few-pixel seam
//!    transition and blends only the low-frequency band across a wide
//!    transition, so tone steps fade while detail is never averaged.
//!
//! Sources are decoded twice (analysis, then rendering) so only one
//! full-resolution tile is resident at a time.
use image::{GrayImage, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Vector3};

/// Side of one ownership cell in output pixels.
const OWNER_CELL: u32 = 32;
/// A cell may be owned without penalty only when this much valid source
/// surrounds it, so resampling and the low band never read past a tile edge.
const OWNER_MARGIN: f64 = 48.0;
/// Data cost added when a tile owns a cell too close to its own border.
const EDGE_OWNER_PENALTY: f64 = 4.0;
/// Data cost of a tile that delivers no detail where the best tile delivers
/// all of it. The graph cut compares it with the Potts seam weight (0.35+).
const SHARPNESS_COST: f64 = 4.0;
/// Upper bound of the per-cell structure term that prices a seam. Kept low:
/// with the smooth registration correction a seam through a stroke stays
/// aligned, so a softer tile must not win a stroke merely to avoid a seam.
const STRUCTURE_SEAM_WEIGHT: f64 = 0.05;
/// Side of one sharpness measurement block in source pixels.
const SHARPNESS_BLOCK: usize = 32;
/// Decimation of the low-frequency band grid in output pixels.
const LOW_GRID: u32 = 8;
/// Low-pass of every tile's low band on the low grid (grid pixels). Wide
/// enough that a residual registration offset of a pixel or two cannot move
/// the band, narrow enough to carry glare and illumination fields.
const LOW_BAND_SIGMA: f32 = 4.0;
/// Final smoothing of the consensus low band (grid pixels).
const CONSENSUS_SIGMA: f32 = 1.5;
/// Feather distance of a tile's consensus weight from its border (grid px).
const CONSENSUS_FEATHER: f32 = 32.0;
/// A tile's low band joins the consensus only within this relative distance
/// of the per-channel median, so one tile's glass reflection is outvoted.
const CONSENSUS_TOLERANCE: f32 = 0.04;
/// Data cost of a tile whose low band departs from the consensus by more
/// than [`TONE_DEVIATION_SPAN`] beyond [`TONE_DEVIATION_FLOOR`]. The owner's
/// tone never reaches the output (the consensus carries it), so only a
/// departure well beyond the few percent photometric residual every tile
/// shows (a reflection) is priced.
const TONE_DEVIATION_COST: f64 = 2.0;
const TONE_DEVIATION_FLOOR: f64 = 0.03;
const TONE_DEVIATION_SPAN: f64 = 0.05;
/// Half-width of the detail transition across an ownership seam (pixels).
const DETAIL_TRANSITION: f32 = 3.0;
/// Patch spacing of the registration correspondences (world pixels).
const REGISTRATION_STEP: f64 = 384.0;
/// Patch centres stay this far inside both source frames.
const REGISTRATION_INSET: f64 = 96.0;
const REGISTRATION_HUBER_PX: f64 = 1.5;
/// Node spacing of the smooth per-tile registration correction (source
/// pixels). A planar pose cannot follow lens distortion or a scroll that is
/// not perfectly flat; the residual it leaves is smooth over this scale.
const MESH_SPACING: f64 = 256.0;
/// Correspondence spacing of the correction solve (world pixels).
const MESH_STEP: f64 = 160.0;
/// Membrane smoothness between neighbouring correction nodes and the pull of
/// every node towards no correction, both relative to one correspondence of
/// unit weight.
const MESH_SMOOTHNESS: f64 = 1.0;
const MESH_DAMPING: f64 = 0.01;
const MESH_HUBER_PX: f64 = 1.0;
/// Every n-th correspondence stays out of the correction solve; the
/// correction is only used when it also aligns those held-out points.
const MESH_HOLDOUT: usize = 7;
const MAX_OUTPUT_PIXELS: f64 = 240_000_000.0;
const MAX_OUTPUT_UPSCALE: f64 = 1.25;
const PREVIEW_LONG_SIDE: f64 = 512.0;
/// Coordinate unit of the homography solve; keeps the normal equations
/// well conditioned for 10k-pixel frames.
const NORMALIZED_UNIT: f64 = 1000.0;

/// One decoded tile handed to the compositor.
pub(crate) struct ScanSource {
    pub(crate) rgb: Rgb32FImage,
    /// `255` where the source has an opaque sample; `None` when fully opaque.
    pub(crate) valid: Option<GrayImage>,
}

pub(crate) struct ScanMosaicResult {
    pub(crate) image: Rgb32FImage,
    /// `255` where an output pixel is backed by source content.
    pub(crate) coverage: GrayImage,
}

struct TileAnalysis {
    width: u32,
    height: u32,
    /// Luma blurred to two output pixels, for native registration.
    reg_full: GrayImage,
    /// Quarter-resolution luma for the coarse registration search.
    reg_quarter: GrayImage,
    /// `255` where all sixteen source pixels of a quarter sample are valid.
    valid_quarter: GrayImage,
    /// Detail energy per [`SHARPNESS_BLOCK`] block, measured in a band fixed
    /// in *output* pixels; NaN where the block is not measurable.
    sharpness: Vec<f32>,
    sharp_columns: usize,
    sharp_rows: usize,
    /// Mean RGB of every `LOW_GRID`×`LOW_GRID` source block.
    low: Vec<[f32; 3]>,
    low_valid: Vec<bool>,
    low_columns: usize,
    low_rows: usize,
    /// Bounded preview for the joint photometric solve.
    preview: Rgb32FImage,
}

#[derive(Clone, Copy)]
struct Canvas {
    width: u32,
    height: u32,
    offset_x: f64,
    offset_y: f64,
}

/// Smooth correction of one tile's planar pose: a displacement in source
/// pixels on a regular control grid, bilinearly interpolated. The tile maps
/// to world as `H(p + F(p))`.
#[derive(Clone, Debug)]
struct DisplacementGrid {
    spacing: f64,
    columns: usize,
    rows: usize,
    values: Vec<[f64; 2]>,
}

impl DisplacementGrid {
    fn zero(width: u32, height: u32, spacing: f64) -> Self {
        let columns = (f64::from(width) / spacing).ceil() as usize + 1;
        let rows = (f64::from(height) / spacing).ceil() as usize + 1;
        Self {
            spacing,
            columns: columns.max(2),
            rows: rows.max(2),
            values: vec![[0.0; 2]; columns.max(2) * rows.max(2)],
        }
    }

    fn len(&self) -> usize {
        self.columns * self.rows
    }

    /// Node indices and bilinear weights of a source point, clamped to the
    /// grid.
    fn stencil(&self, point: Point2<f64>) -> [(usize, f64); 4] {
        let gx = (point.x / self.spacing).clamp(0.0, (self.columns - 1) as f64);
        let gy = (point.y / self.spacing).clamp(0.0, (self.rows - 1) as f64);
        let ix = (gx.floor() as usize).min(self.columns - 2);
        let iy = (gy.floor() as usize).min(self.rows - 2);
        let fx = gx - ix as f64;
        let fy = gy - iy as f64;
        let base = iy * self.columns + ix;
        [
            (base, (1.0 - fx) * (1.0 - fy)),
            (base + 1, fx * (1.0 - fy)),
            (base + self.columns, (1.0 - fx) * fy),
            (base + self.columns + 1, fx * fy),
        ]
    }

    fn at(&self, point: Point2<f64>) -> [f64; 2] {
        let mut value = [0.0; 2];
        for (node, weight) in self.stencil(point) {
            value[0] += weight * self.values[node][0];
            value[1] += weight * self.values[node][1];
        }
        value
    }

    /// The source point `p` with `p + F(p) == point`.
    fn invert(&self, point: Point2<f64>) -> Point2<f64> {
        let mut source = point;
        for _ in 0..4 {
            let offset = self.at(source);
            source = Point2::new(point.x - offset[0], point.y - offset[1]);
        }
        source
    }

    fn largest(&self) -> f64 {
        self.values
            .iter()
            .map(|value| value[0].hypot(value[1]))
            .fold(0.0, f64::max)
    }
}

/// Output-world → source mapping of one tile: the inverse planar pose and,
/// when registration found one, the inverse of the smooth correction.
struct TileWarp {
    forward: Matrix3<f64>,
    inverse: Matrix3<f64>,
    correction: Option<DisplacementGrid>,
}

impl TileWarp {
    #[inline]
    fn source(&self, world: Point2<f64>) -> Option<Point2<f64>> {
        let planar = project(&self.inverse, world)?;
        Some(match &self.correction {
            Some(grid) => grid.invert(planar),
            None => planar,
        })
    }
}

fn project(homography: &Matrix3<f64>, point: Point2<f64>) -> Option<Point2<f64>> {
    let mapped = homography * Vector3::new(point.x, point.y, 1.0);
    (mapped.z.abs() > 1e-12).then(|| Point2::new(mapped.x / mapped.z, mapped.y / mapped.z))
}

mod compositor;

pub(crate) use compositor::{opaque_source_mask, planar_scan_mosaic};

mod tile_analysis;

use tile_analysis::{analyse_tile, blur_plane, quarter_valid, validity_of};

mod homography_refine;

use homography_refine::{
    Correspondence, collect_correspondences, correspondence_residuals, inside_frame, percentile,
    refine_homographies,
};

mod mesh_correction;

use mesh_correction::refine_mesh;

mod ownership;

use ownership::{Ownership, plan_ownership};

mod low_frequency;

use low_frequency::{LowBand, consensus_low_band, low_band, solve_photometry};

mod detail_render;

use detail_render::{eroded_validity, render_tile_detail};
