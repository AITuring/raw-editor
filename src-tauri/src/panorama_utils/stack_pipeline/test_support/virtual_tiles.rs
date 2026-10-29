//! Coverage_Mask shapes and Virtual_Tile generators.
//!
//! Every generated tile satisfies the Virtual_Tile construction invariants; the
//! properties derive the illegal variants themselves.

use crate::panorama_utils::stitching::ColorEncoding;
use crate::panorama_utils::stitching::ConfidenceMap;
use crate::panorama_utils::stitching::CoverageMask;
use crate::panorama_utils::stitching::OwnershipMap;
use crate::panorama_utils::stitching::SourceProvenance;
use crate::panorama_utils::stitching::VirtualTile;
use image::GrayImage;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use proptest::prelude::*;
use std::path::PathBuf;

use super::{splitmix64, unit_interval};

// ---------------------------------------------------------------------------
// Virtual_Tile generators (设计 `Testing Strategy > 生成器设计`)
// ---------------------------------------------------------------------------
//
// `arb_coverage_shape()` draws a Coverage_Mask shape, `arb_virtual_tile()`
// grows one into a complete [`VirtualTile`].  The two are separate because
// Property 13 only needs the mask geometry while Properties 18, 19 and 23 need
// the whole tile, and because the cache properties want a larger tile than the
// mask-only ones (`arb_*_in` takes the side range).
//
// Every tile a generator returns passes `VirtualTile::new`, i.e. it satisfies
// the four construction invariants.  That is deliberate: the generator produces
// *legal* tiles, and each property test derives its own illegal variants from
// them so the rejection direction is exercised too.

/// Which shape family a [`CoverageShape`] was drawn from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverageFamily {
    /// Nothing covered: the "no covered pixel at all" boundary of 需求 4.7.
    Empty,
    /// Every pixel covered, so the coverage bounds are the whole tile.
    Full,
    /// One axis aligned rectangle somewhere inside the tile.
    Rectangle,
    /// Two rectangles, so the union bounds are wider than either blob *and*
    /// uncovered pixels exist inside those bounds.
    TwoBlobs,
    /// Independent per pixel draw: a ragged, non convex, possibly disconnected
    /// mask.  This is the family a real projected trapezoid intersection plus a
    /// low-sharpness rejection produces.
    Ragged,
    /// Exactly one covered pixel: 1x1 bounds anywhere inside the tile.
    SinglePixel,
}

/// A Coverage_Mask shape: the pixel grid plus the covered predicate.
///
/// The covered flags are held as `bool` rather than as a [`CoverageMask`] so a
/// property test can compare the production mask against an independently
/// computed answer instead of against itself.
#[derive(Clone, PartialEq, Eq)]
pub struct CoverageShape {
    pub width: u32,
    pub height: u32,
    /// Row major, `width * height` entries.
    pub covered: Vec<bool>,
    pub family: CoverageFamily,
}

/// A summary rather than the whole bitmap: `proptest` prints this on failure.
impl std::fmt::Debug for CoverageShape {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CoverageShape")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("family", &self.family)
            .field("covered_pixels", &self.covered_count())
            .field("bounds", &self.bounds())
            .finish()
    }
}

impl CoverageShape {
    pub fn index(&self, x: u32, y: u32) -> usize {
        y as usize * self.width as usize + x as usize
    }

    pub fn is_covered(&self, x: u32, y: u32) -> bool {
        self.covered[self.index(x, y)]
    }

    pub fn covered_count(&self) -> u64 {
        self.covered.iter().filter(|&&covered| covered).count() as u64
    }

    /// Union axis aligned bounds of the covered pixels as `(x, y, w, h)`,
    /// computed here so it is an independent answer to compare
    /// `CoverageMask::covered_bounds` against (需求 4.7).
    pub fn bounds(&self) -> Option<(u32, u32, u32, u32)> {
        let mut min_x = u32::MAX;
        let mut min_y = u32::MAX;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        let mut any = false;
        for y in 0..self.height {
            for x in 0..self.width {
                if !self.is_covered(x, y) {
                    continue;
                }
                any = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
        any.then(|| (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
    }

    /// The 8 bit plane the focus renderer hands to [`CoverageMask::from_gray`].
    pub fn gray(&self) -> GrayImage {
        GrayImage::from_raw(
            self.width,
            self.height,
            self.covered
                .iter()
                .map(|&covered| if covered { 255u8 } else { 0u8 })
                .collect(),
        )
        .expect("the coverage plane has one byte per pixel")
    }

    pub fn mask(&self) -> CoverageMask {
        CoverageMask::from_gray(self.gray())
    }
}

pub(super) fn filled_rectangle(shape: &mut CoverageShape, state: &mut u64) {
    let left = (splitmix64(state) % u64::from(shape.width)) as u32;
    let top = (splitmix64(state) % u64::from(shape.height)) as u32;
    let width = 1 + (splitmix64(state) % u64::from(shape.width - left)) as u32;
    let height = 1 + (splitmix64(state) % u64::from(shape.height - top)) as u32;
    for y in top..top + height {
        for x in left..left + width {
            let index = shape.index(x, y);
            shape.covered[index] = true;
        }
    }
}

pub(super) fn coverage_shape(
    width: u32,
    height: u32,
    family: CoverageFamily,
    seed: u64,
    density: f64,
) -> CoverageShape {
    let mut shape = CoverageShape {
        width,
        height,
        covered: vec![false; width as usize * height as usize],
        family,
    };
    let mut state = seed | 1;
    match family {
        CoverageFamily::Empty => {}
        CoverageFamily::Full => shape.covered.fill(true),
        CoverageFamily::Rectangle => filled_rectangle(&mut shape, &mut state),
        CoverageFamily::TwoBlobs => {
            filled_rectangle(&mut shape, &mut state);
            filled_rectangle(&mut shape, &mut state);
        }
        CoverageFamily::Ragged => {
            for slot in shape.covered.iter_mut() {
                *slot = unit_interval(&mut state) < density;
            }
        }
        CoverageFamily::SinglePixel => {
            let x = (splitmix64(&mut state) % u64::from(width)) as u32;
            let y = (splitmix64(&mut state) % u64::from(height)) as u32;
            let index = shape.index(x, y);
            shape.covered[index] = true;
        }
    }
    shape
}

/// Coverage_Mask shapes over tiles of `min_side..=max_side` pixels a side.
///
/// Non square tiles are reachable (the two sides are drawn independently),
/// which is what makes a swapped row/column index in any of the four planes
/// observable.
pub fn arb_coverage_shape_in(min_side: u32, max_side: u32) -> impl Strategy<Value = CoverageShape> {
    (
        min_side..=max_side,
        min_side..=max_side,
        prop_oneof![
            1 => Just(CoverageFamily::Empty),
            1 => Just(CoverageFamily::Full),
            3 => Just(CoverageFamily::Rectangle),
            3 => Just(CoverageFamily::TwoBlobs),
            3 => Just(CoverageFamily::Ragged),
            1 => Just(CoverageFamily::SinglePixel),
        ],
        any::<u64>(),
        0.05f64..0.95,
    )
        .prop_map(|(width, height, family, seed, density)| {
            coverage_shape(width, height, family, seed, density)
        })
}

/// A fully covered shape of an explicit size, for properties that need an
/// entry of a chosen payload size rather than a drawn mask.
pub fn full_coverage_shape(width: u32, height: u32) -> CoverageShape {
    coverage_shape(width, height, CoverageFamily::Full, 1, 1.0)
}

/// Coverage_Mask shapes over 1x1..17x17 tiles.
///
/// The small side range is on purpose: the mask invariants are per pixel, and
/// 17 is not a multiple of 8, so the tail byte of the one-bit-per-pixel cache
/// encoding is exercised.  The odd sizes also reach the 1x1 and single-row
/// boundaries a large tile never visits.
pub fn arb_coverage_shape() -> impl Strategy<Value = CoverageShape> {
    arb_coverage_shape_in(1, 17)
}

/// Channel values written into covered Virtual_Tile pixels.
///
/// Each one is a value an image codec would destroy — a subnormal, a long
/// mantissa, a negative, the f32 maximum — and none of them is zero, so the
/// non-zero pixel extent of a generated tile is exactly its covered extent
/// (需求 4.7).
pub const VIRTUAL_TILE_PIXEL_PALETTE: [f32; 8] = [
    f32::from_bits(0x0000_0001),
    0.123_456_79_f32,
    -2.5_f32,
    1.0 / 3.0,
    f32::MAX,
    f32::MIN_POSITIVE,
    -1.0e-30_f32,
    9_999.999_f32,
];

/// Absolute path of legend entry `index`.
pub fn synthetic_source_path(index: usize) -> PathBuf {
    PathBuf::from(format!("/tmp/stack-pipeline/DSC_{:04}.NEF", 3680 + index))
}

pub(super) fn synthetic_digest(seed: u64) -> [u8; 32] {
    let mut state = seed | 1;
    let mut digest = [0u8; 32];
    for slot in digest.iter_mut() {
        *slot = (splitmix64(&mut state) % 256) as u8;
    }
    digest
}

/// Grow a [`CoverageShape`] into a complete, invariant satisfying Virtual_Tile.
#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_from(
    shape: &CoverageShape,
    legend_len: usize,
    station_index: usize,
    color_encoding: ColorEncoding,
    uniform_confidence: bool,
    exact_confidence: bool,
    seed: u64,
) -> VirtualTile {
    let (width, height) = (shape.width, shape.height);
    let legend = (0..legend_len)
        .map(synthetic_source_path)
        .collect::<Vec<_>>();
    let mut state = seed | 1;
    let mut owners = vec![0u16; shape.covered.len()];
    let mut pixels = Rgb32FImage::new(width, height);
    let mut confidence = vec![0.0f32; shape.covered.len()];
    for y in 0..height {
        for x in 0..width {
            let index = shape.index(x, y);
            if !shape.covered[index] {
                // Uncovered pixels stay `NO_OWNER`, confidence 0.0 and black,
                // which is what 需求 3.11 demands and what invariant 4 of
                // `VirtualTile::new` checks outside the coverage bounds.
                continue;
            }
            owners[index] = 1 + (splitmix64(&mut state) % legend_len as u64) as u16;
            let palette = splitmix64(&mut state) as usize;
            pixels.put_pixel(
                x,
                y,
                Rgb([
                    VIRTUAL_TILE_PIXEL_PALETTE[palette % VIRTUAL_TILE_PIXEL_PALETTE.len()],
                    VIRTUAL_TILE_PIXEL_PALETTE[(palette / 8) % VIRTUAL_TILE_PIXEL_PALETTE.len()],
                    VIRTUAL_TILE_PIXEL_PALETTE[(palette / 64) % VIRTUAL_TILE_PIXEL_PALETTE.len()],
                ]),
            );
            confidence[index] = if exact_confidence {
                // A multiple of 2^-10 is exactly representable as f16, so the
                // cache round trip of this value must be bit exact.
                (splitmix64(&mut state) % 1025) as f32 / 1024.0
            } else {
                // Generally *not* representable as f16: the round trip is then
                // only exact for the rounded value the cache wrote.
                (splitmix64(&mut state) % 997) as f32 / 996.0
            };
        }
    }
    let ownership = OwnershipMap::new(width, height, owners, legend.clone())
        .expect("the generated ownership map has one owner per pixel inside its legend");
    let sharpness_confidence = if uniform_confidence {
        ConfidenceMap::zero(width, height)
    } else {
        ConfidenceMap::per_pixel(width, height, confidence)
            .expect("the generated confidence map has one value per pixel")
    };
    let provenance = legend
        .iter()
        .zip(ownership.owned_pixel_counts())
        .enumerate()
        .map(|(index, (path, owned_pixels))| SourceProvenance {
            absolute_path: path.clone(),
            sha256: Some(synthetic_digest(seed ^ (index as u64 + 1))),
            owned_pixels,
        })
        .collect();
    // A translation with a long mantissa, so a lossy round trip of
    // `tile_to_world` shows up as a changed bit pattern.
    let tile_to_world = Matrix3::new(
        1.0,
        0.0,
        -1_234.567_890_123_456_7,
        0.0,
        1.0,
        -5_678.901_234_567_89,
        0.0,
        0.0,
        1.0,
    );
    VirtualTile::new(
        station_index,
        tile_to_world,
        pixels,
        ownership,
        sharpness_confidence,
        shape.mask(),
        color_encoding,
        provenance,
    )
    .expect("the generated tile satisfies every Virtual_Tile invariant")
}

/// Virtual_Tiles over tiles of `min_side..=max_side` pixels a side.
pub fn arb_virtual_tile_in(min_side: u32, max_side: u32) -> impl Strategy<Value = VirtualTile> {
    (
        arb_coverage_shape_in(min_side, max_side),
        1usize..=4,
        0usize..16,
        prop_oneof![
            Just(ColorEncoding::LinearSrgb),
            Just(ColorEncoding::DisplaySrgb)
        ],
        any::<bool>(),
        any::<bool>(),
        any::<u64>(),
    )
        .prop_map(
            |(
                shape,
                legend_len,
                station_index,
                color_encoding,
                uniform_confidence,
                exact_confidence,
                seed,
            )| {
                virtual_tile_from(
                    &shape,
                    legend_len,
                    station_index,
                    color_encoding,
                    uniform_confidence,
                    exact_confidence,
                    seed,
                )
            },
        )
}

/// Virtual_Tiles over 1x1..17x17 tiles: 1..4 Source_RAWs, both colour
/// encodings, both Sharpness_Confidence representations and every coverage
/// family.
pub fn arb_virtual_tile() -> impl Strategy<Value = VirtualTile> {
    arb_virtual_tile_in(1, 17)
}

/// A Virtual_Tile over an explicit [`CoverageShape`], for properties that draw
/// the mask themselves (Property 13).
pub fn virtual_tile_with_shape(shape: &CoverageShape, legend_len: usize) -> VirtualTile {
    virtual_tile_from(
        shape,
        legend_len,
        0,
        ColorEncoding::LinearSrgb,
        false,
        true,
        shape.covered_count().wrapping_add(u64::from(shape.width)) | 1,
    )
}

/// The smallest legal Virtual_Tile, for properties about *when* a tile is
/// resident rather than about what it holds (需求 14.4).
pub fn minimal_virtual_tile(station_index: usize) -> VirtualTile {
    let shape = coverage_shape(2, 2, CoverageFamily::SinglePixel, 1, 1.0);
    virtual_tile_from(
        &shape,
        1,
        station_index,
        ColorEncoding::LinearSrgb,
        true,
        true,
        station_index as u64 + 1,
    )
}
