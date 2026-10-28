//! Canvas ownership, coverage and confidence maps shared by every compositor.
//!
//! The maps hold per-pixel source ownership, coverage and confidence in output
//! coordinates; the two mapping helpers resolve an output pixel to its source
//! pixel through the optional residual field and focus warp.

use super::super::coordinates::Projection;
use super::super::coordinates::map_target_to_source;
use super::super::coordinates::pixel_aligned_canvas;
use super::super::render_geometry::focus_output_bounds;
use super::super::stack_pipeline::compositor::tile_coordinate_with_residual;
use super::super::stack_pipeline::residual_warp;
use crate::panorama_utils::shared::FocusLayerWarp;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;
use std::collections::HashMap;
use std::path::PathBuf;

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
    if residual.regions.is_empty() && residual.line_layers.reports.is_empty() {
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
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) owners: Vec<u16>,
    /// `legend[id - 1]` is the absolute path of owner identifier `id`.
    pub(super) legend: Vec<PathBuf>,
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
    pub(super) width: u32,
    pub(super) height: u32,
    /// Non-zero means covered; the focus renderer already keeps this mask as
    /// an 8-bit image, so the tile stores it without a re-encoding step.
    pub(super) covered: Vec<u8>,
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

    pub(super) fn in_unit_range(&self) -> bool {
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
