//! Stitching modes and the outcome of a stitch run.
//!
//! `AlignmentMode` and `BlendMode` select the registration model and the
//! compositor; `StitchOutcome` is what `stitch_images_with_options` returns.

use crate::panorama_utils::stitching::Projection;
use image::DynamicImage;
use image::GrayImage;
use std::path::PathBuf;

pub(crate) struct StitchOutcome {
    pub image: DynamicImage,
    /// The layered path's output Coverage_Mask, the same size as `image`;
    /// the export writes it as alpha (需求 10.3 / 10.6).
    pub coverage: Option<GrayImage>,
    pub full_canvas_width: u32,
    pub full_canvas_height: u32,
    pub render_scale: f64,
    pub ordered_paths: Vec<String>,
    /// The report written for this focus-stack run.  Image-stack export fills
    /// in the result identifier and output metadata after the file is
    /// published, while keeping the report tied to the same run.
    pub stack_report_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentMode {
    Auto,
    Perspective,
    Cylindrical,
    Spherical,
    Position,
    /// Alignment for a flat artwork photographed as a mostly orthogonal grid.
    /// Keep the per-pair model at similarity (translation + uniform scale +
    /// rotation) instead of allowing the panorama selector to overfit paper
    /// texture with a projective warp.
    FlatArtwork,
}

impl AlignmentMode {
    pub fn from_wire(value: &str) -> Self {
        match value {
            "perspective" => Self::Perspective,
            "cylindrical" => Self::Cylindrical,
            "spherical" => Self::Spherical,
            "position" => Self::Position,
            "flat-artwork" | "flat_artwork" | "artwork" => Self::FlatArtwork,
            _ => Self::Auto,
        }
    }

    pub(super) fn projection_for(self, blend_mode: BlendMode) -> Projection {
        match self {
            Self::Cylindrical => Projection::Cylindrical,
            Self::Spherical => Projection::Spherical,
            Self::Perspective | Self::Position | Self::FlatArtwork => Projection::Planar,
            Self::Auto => match blend_mode {
                BlendMode::Panorama => Projection::Planar,
                BlendMode::FocusStack => Projection::Planar,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Panorama,
    FocusStack,
}
