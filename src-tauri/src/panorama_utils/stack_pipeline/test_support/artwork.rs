//! The synthetic artwork plane every scan generator crops from.
//!
//! Paper grain plus variable-width strokes; the painted variant adds more strokes
//! and one blur so the finest structure is band limited like a real analysis
//! image.

use image::GrayImage;
use image::Luma;
use nalgebra::Point2;
use proptest::prelude::*;
use std::sync::Arc;

use super::{in_range, splitmix64};

/// Side of the procedurally generated artwork plane, in plane pixels.
pub const SYNTHETIC_PLANE_SIDE: u32 = 640;

/// The synthetic "书画"底图: paper grain plus a handful of variable width
/// Bézier strokes and large empty areas.
#[derive(Clone)]
pub struct ArtworkPlane {
    pub(super) image: Arc<GrayImage>,
}

impl ArtworkPlane {
    pub fn image(&self) -> &GrayImage {
        &self.image
    }
}

/// A summary rather than 400 KiB of pixels: `proptest` prints this on failure.
impl std::fmt::Debug for ArtworkPlane {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (width, height) = self.image.dimensions();
        let mean = self
            .image
            .as_raw()
            .iter()
            .map(|&value| u64::from(value))
            .sum::<u64>()
            / u64::from(width * height).max(1);
        formatter
            .debug_struct("ArtworkPlane")
            .field("width", &width)
            .field("height", &height)
            .field("mean_luma", &mean)
            .finish()
    }
}

pub(super) fn paint_stroke_sample(image: &mut GrayImage, x: f64, y: f64, radius: f64, ink: u8) {
    let (width, height) = image.dimensions();
    let min_x = (x - radius).floor().max(0.0) as u32;
    let min_y = (y - radius).floor().max(0.0) as u32;
    let max_x = ((x + radius).ceil() as i64).clamp(0, i64::from(width) - 1) as u32;
    let max_y = ((y + radius).ceil() as i64).clamp(0, i64::from(height) - 1) as u32;
    for pixel_y in min_y..=max_y {
        for pixel_x in min_x..=max_x {
            let dx = f64::from(pixel_x) + 0.5 - x;
            let dy = f64::from(pixel_y) + 0.5 - y;
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            let current = image.get_pixel(pixel_x, pixel_y)[0];
            image.put_pixel(pixel_x, pixel_y, Luma([current.min(ink)]));
        }
    }
}

pub(super) fn artwork_plane(seed: u64, stroke_count: usize) -> ArtworkPlane {
    let side = SYNTHETIC_PLANE_SIDE;
    // Paper: a bright base with a fine grain, so even the 留白 carries enough
    // signal for the dense overlap correlation to be measurable.
    let mut image = GrayImage::from_fn(side, side, |x, y| {
        let mut local = seed
            ^ (u64::from(x).wrapping_mul(0x2545_F491_4F6C_DD1D))
            ^ (u64::from(y).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        Luma([238u8.saturating_sub((splitmix64(&mut local) % 13) as u8)])
    });
    let mut state = seed | 1;
    for _ in 0..stroke_count {
        let control: [Point2<f64>; 3] = std::array::from_fn(|_| {
            Point2::new(
                in_range(&mut state, 0.0, f64::from(side)),
                in_range(&mut state, 0.0, f64::from(side)),
            )
        });
        let ink = in_range(&mut state, 12.0, 92.0) as u8;
        let half_width = in_range(&mut state, 1.2, 4.0);
        let steps = 320usize;
        for step in 0..=steps {
            let t = step as f64 / steps as f64;
            let inverse = 1.0 - t;
            let x = inverse * inverse * control[0].x
                + 2.0 * inverse * t * control[1].x
                + t * t * control[2].x;
            let y = inverse * inverse * control[0].y
                + 2.0 * inverse * t * control[1].y
                + t * t * control[2].y;
            // A brush tapers at both ends.
            let radius = half_width * (0.45 + 0.55 * (t * std::f64::consts::PI).sin());
            paint_stroke_sample(&mut image, x, y, radius, ink);
        }
    }
    ArtworkPlane {
        image: Arc::new(image),
    }
}

/// Random artwork plane: 6..14 strokes over a grained paper base.
pub fn arb_artwork_plane() -> impl Strategy<Value = ArtworkPlane> {
    (any::<u64>(), 6usize..14).prop_map(|(seed, strokes)| artwork_plane(seed, strokes))
}

/// A densely painted artwork plane whose grain is band limited.
///
/// The Capture_Station properties measure the 需求 1.1 overlap correlation between
/// two focus layers of one camera position, and those two layers are crops of the
/// plane through *different sub-pixel* poses.  Per-pixel white grain cannot
/// survive that: two different resamplings of it decorrelate, and on a sparsely
/// painted plane the grain is most of the signal, so the measured correlation of a
/// genuine intra-station pair drops below the 0.60 bar for reasons that have
/// nothing to do with the grouping.
///
/// A real analysis image cannot behave that way either — it is downsampled from a
/// 45MP RAW, so its grain is band limited by construction.  This plane models
/// that: 18..34 strokes instead of 6..14, and one Gaussian pass over the whole
/// plane so the finest structure spans more than one pixel.
///
/// [`arb_artwork_plane`] stays as it is, because Properties 2 and 90 are about
/// determinism rather than about correlation and their fixtures are pinned.
pub fn arb_painted_artwork_plane() -> impl Strategy<Value = ArtworkPlane> {
    (any::<u64>(), 18usize..34).prop_map(|(seed, strokes)| {
        let plane = artwork_plane(seed, strokes);
        ArtworkPlane {
            image: Arc::new(image::imageops::blur(plane.image(), 0.9)),
        }
    })
}
