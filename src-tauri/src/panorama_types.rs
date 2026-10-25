use image::{GrayImage, Rgb32FImage};
use nalgebra::{Matrix3, Point2};

/// Number of bits in the BRIEF descriptor used by the panorama matcher.
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
    /// 35mm-equivalent focal length when available in source metadata. This
    /// is an alignment hint; it does not alter exported metadata or pixels.
    pub focal_length_35mm: Option<f64>,
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
    /// The fit exactly as measured in the canonical content direction. The
    /// optional copy keeps deterministic consumers independent of key order.
    pub canonical_homography: Option<Matrix3<f64>>,
}
