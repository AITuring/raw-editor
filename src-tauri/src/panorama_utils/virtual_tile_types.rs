//! Core Virtual_Tile data structures and invariants.
//!
//! These types deliberately have no renderer or compositor behavior. Keeping
//! the pixel metadata contracts together makes the tile cache, diagnostics,
//! and focus renderer depend on one small, testable boundary.

use image::{GrayImage, Rgb32FImage};
use nalgebra::Matrix3;
use std::path::PathBuf;

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
