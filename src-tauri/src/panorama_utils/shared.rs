//! Shared focus-stack data contracts.
//!
//! These types are used by the top-level preparation flow and by the geometry,
//! registration, mosaic and stack-pipeline modules. Keeping them here prevents
//! the facade from owning data definitions that every lower-level stage needs.

use image::{GrayImage, Rgb32FImage};
use nalgebra::{Matrix3, Point2};
use std::collections::HashMap;

pub const BRIEF_DESCRIPTOR_SIZE: usize = 256;
pub type Descriptor = [u8; BRIEF_DESCRIPTOR_SIZE / 8];

#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    pub x: u32,
    pub y: u32,
}

#[derive(Clone)]
pub struct Feature {
    pub keypoint: KeyPoint,
    pub descriptor: Descriptor,
    /// Pyramid level used to build the descriptor. The keypoint is mapped
    /// back to the input image, but the support scale is retained so a
    /// different focal module can be matched against a physically compatible
    /// BRIEF neighbourhood instead of an arbitrary level.
    pub support_scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Match {
    pub index1: usize,
    pub index2: usize,
}

pub struct ImageInfo {
    pub id: usize,
    pub filename: String,
    pub width: u32,
    pub height: u32,
    pub alignment_image: GrayImage,
    pub full_image: Option<Rgb32FImage>,
    pub scale_factor: f64,
    /// 35mm-equivalent focal length when available in source metadata.  This
    /// is an alignment hint; it does not alter exported metadata or pixels.
    pub focal_length_35mm: Option<f64>,
    pub(crate) source_footprint: Option<crate::lens_correction::StackLensFootprint>,
    /// A supplied wide frame used to bridge a missing focal-length row is a
    /// coverage fallback, never the preferred sharp source over a close-up.
    pub overview_reference: bool,
    pub features: Vec<Feature>,
    pub top_features: Vec<Feature>,
    pub foreground_range: Option<(f64, f64)>,
    pub foreground_mask: Option<GrayImage>,
    pub horizontal_edge_rows: Vec<f64>,
    pub vertical_edge_columns: Vec<f64>,
}

impl ImageInfo {
    pub(crate) fn source_is_supported(&self, point: Point2<f64>) -> bool {
        point.x >= 0.0
            && point.y >= 0.0
            && point.x < f64::from(self.width)
            && point.y < f64::from(self.height)
            && self
                .source_footprint
                .is_none_or(|footprint| footprint.contains(point.x, point.y, self.dimensions()))
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

#[derive(Clone)]
pub struct MatchInfo {
    pub homography: Matrix3<f64>,
    pub inliers: usize,
    /// A filename-sequence bridge is an ordering cue, not a visual
    /// correspondence. It may connect adjacent captures after a lens switch
    /// left no literal overlap, but it must never enter the pixel-level global
    /// registration solve as if its points were measured from the artwork.
    pub sequence_bridge: bool,
    /// A metadata-scaled image-correlation candidate is an ordering cue only
    /// until its overlap passes the full-resolution validation. Its grid
    /// points are synthetic and must not pull the artwork pose optimizer.
    pub coarse_bridge: bool,
    pub points: Vec<(Point2<f64>, Point2<f64>)>,
    pub candidate_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub top_candidate_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub dense_focus_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub foreground_feature_points: Vec<(Point2<f64>, Point2<f64>)>,
    /// The fit exactly as measured, in the direction
    /// [`canonical_match_direction`] chose for the two sources.
    ///
    /// `homography` is stored in the `(min index, max index)` orientation of the
    /// `matches` key, so whenever that orientation disagrees with the content
    /// derived measurement direction it holds `try_inverse()` of the fit.  That
    /// round trip is not bit exact — `inv(inv(M))` differs from `M` in the last
    /// mantissa bits — so a consumer that rotated `homography` back into the
    /// canonical direction still inherited which of the two sources happened to
    /// be imported first, and the station pose solve propagated it into the
    /// rendered canvas.  Keeping the untouched fit here lets such a consumer
    /// read the same bits for either import order (需求 14.6).
    ///
    /// `None` for relations that never passed through a canonical-direction
    /// storage step — panorama matching and filename-sequence bridges; those
    /// keep the previous inverse round trip — and for Virtual_Tile station
    /// relations, which are measured untouched in the `(left, right)` key
    /// direction: a consumer derives the canonical direction from `homography`
    /// with at most one inversion. (A key-direction fit stored here was read as
    /// the reverse relation whenever the content-derived direction ran
    /// right→left: wenyuan-10's verified 1→2 then re-checked at 8329px.)
    pub canonical_homography: Option<Matrix3<f64>>,
    /// Native patch measurements retained with their actual reverse-check
    /// errors. Coarse and synthetic matches leave this empty and cannot
    /// establish support for the bounded station deformation field.
    pub native_verified_points: Vec<(Point2<f64>, Point2<f64>, f64)>,
}

#[derive(Clone)]
pub(crate) struct FocusWarpBand {
    pub(crate) homographies: HashMap<usize, Matrix3<f64>>,
    pub(crate) source_ranges: HashMap<usize, (f64, f64)>,
    pub(crate) source_x_ranges: HashMap<usize, (f64, f64)>,
    // A repeated, content-independent long edge may safely change ownership
    // at its narrow silhouette. Generic/depth bands keep the stricter
    // foreground ownership rule so a repeated stroke cannot be reintroduced.
    pub(crate) relax_foreground_seam: bool,
    // A depth-layer correction is only valid for pixels that belong to the
    // detected layer. Keeping this bit on the band prevents a near-field
    // correction from bending the continuous paper plane beside it.
    pub(crate) foreground_only: bool,
    // Long-edge consensus is a separate, content-independent geometry cue.
    // It may be retained in a shifted mosaic to keep a physical frame/rail
    // continuous, while generic regional focus fits remain disabled there.
    pub(crate) physical_edge: bool,
}

#[derive(Clone)]
pub(crate) struct FocusLayerWarp {
    pub(crate) bands: Vec<FocusWarpBand>,
}
