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
    /// Pyramid level used to build the descriptor.
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
    pub focal_length_35mm: Option<f64>,
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
    pub sequence_bridge: bool,
    pub coarse_bridge: bool,
    pub points: Vec<(Point2<f64>, Point2<f64>)>,
    pub candidate_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub top_candidate_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub dense_focus_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub foreground_feature_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub canonical_homography: Option<Matrix3<f64>>,
}

#[derive(Clone)]
pub(crate) struct FocusWarpBand {
    pub(crate) homographies: HashMap<usize, Matrix3<f64>>,
    pub(crate) source_ranges: HashMap<usize, (f64, f64)>,
    pub(crate) source_x_ranges: HashMap<usize, (f64, f64)>,
    pub(crate) relax_foreground_seam: bool,
    pub(crate) foreground_only: bool,
    pub(crate) physical_edge: bool,
}

#[derive(Clone)]
pub(crate) struct FocusLayerWarp {
    pub(crate) bands: Vec<FocusWarpBand>,
}
