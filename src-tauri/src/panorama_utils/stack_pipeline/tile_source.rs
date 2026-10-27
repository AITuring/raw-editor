//! Full-resolution source-window and output-tile contracts (任务 21.7).
//!
//! This module deliberately owns only the geometry and streaming boundary. A
//! production [`FullResSource`] implementation must decode the requested RAW
//! window without materialising the complete source image. Keeping the
//! contract independent of the RAW decoder lets the compositor test tile
//! ordering, halo expansion and projective bounds without silently falling
//! back to a whole-frame allocation.

use image::Rgb32FImage;
use nalgebra::{Matrix3, Point2, Point3};
use serde::{Deserialize, Serialize};

/// The default output tile side required by task 21.7.
pub(crate) const DEFAULT_TILE_SIDE: u32 = 4096;

/// A conservative default for a cubic/bicubic sampling kernel. Callers may
/// choose a larger halo when their filter reaches farther than two pixels.
pub(crate) const DEFAULT_FILTER_HALO: u32 = 2;

/// A half-open rectangle in output-world or source-pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TileRect {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl TileRect {
    pub(crate) const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub(crate) fn end_x(self) -> u64 {
        u64::from(self.x) + u64::from(self.width)
    }

    pub(crate) fn end_y(self) -> u64 {
        u64::from(self.y) + u64::from(self.height)
    }

    pub(crate) fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Full-resolution source window requested for one output tile.
///
/// The window is clamped to the source dimensions and includes the requested
/// filter halo. The caller must retain the origin when mapping samples back to
/// source coordinates.
pub(crate) type SourceReadWindow = TileRect;

/// An output tile plan. `tiles` is always row-major (top to bottom, then left
/// to right), so parallel execution and cache-hit order cannot change the
/// sequence handed to an encoder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TilePlan {
    pub(crate) canvas_width: u32,
    pub(crate) canvas_height: u32,
    pub(crate) tile_width: u32,
    pub(crate) tile_height: u32,
    pub(crate) filter_halo: u32,
    pub(crate) tiles: Vec<TileRect>,
}

impl TilePlan {
    pub(crate) fn new(
        canvas_width: u32,
        canvas_height: u32,
        tile_width: u32,
        tile_height: u32,
        filter_halo: u32,
    ) -> Result<Self, String> {
        if canvas_width == 0 || canvas_height == 0 {
            return Err("tile plan canvas dimensions must be non-zero".to_string());
        }
        if tile_width == 0 || tile_height == 0 {
            return Err("tile plan tile dimensions must be non-zero".to_string());
        }

        let mut tiles = Vec::new();
        let mut y = 0_u32;
        while y < canvas_height {
            let height = tile_height.min(canvas_height - y);
            let mut x = 0_u32;
            while x < canvas_width {
                let width = tile_width.min(canvas_width - x);
                tiles.push(TileRect::new(x, y, width, height));
                x = x.saturating_add(tile_width);
            }
            y = y.saturating_add(tile_height);
        }

        Ok(Self {
            canvas_width,
            canvas_height,
            tile_width,
            tile_height,
            filter_halo,
            tiles,
        })
    }

    pub(crate) fn default_for(canvas_width: u32, canvas_height: u32) -> Result<Self, String> {
        Self::new(
            canvas_width,
            canvas_height,
            DEFAULT_TILE_SIDE,
            DEFAULT_TILE_SIDE,
            DEFAULT_FILTER_HALO,
        )
    }
}

/// A source provider used by a tile compositor. Implementations must perform
/// bounded, read-only region reads; returning a full-frame decode here would
/// violate the 21.7 memory contract even if the caller later crops it.
pub(crate) trait FullResSource {
    fn dimensions(&self) -> (u32, u32);
    fn source_sha256(&self) -> &str;
    fn color_encoding(&self) -> &str;
    fn read_window(&mut self, window: SourceReadWindow) -> Result<Rgb32FImage, String>;
}

fn project(matrix: &Matrix3<f64>, point: Point2<f64>) -> Option<Point2<f64>> {
    let projected = matrix * Point3::new(point.x, point.y, 1.0);
    if !projected.x.is_finite() || !projected.y.is_finite() || !projected.z.is_finite() {
        return None;
    }
    if projected.z.abs() < 1e-12 {
        return None;
    }
    Some(Point2::new(
        projected.x / projected.z,
        projected.y / projected.z,
    ))
}

/// Calculate the bounded source rectangle needed for one output tile.
///
/// `target_to_source` must be the complete inverse mapping, including global
/// pose and residual warp. The four corners plus centre are checked; a
/// projective singularity or non-finite mapping rejects the request instead of
/// returning an unsafe under-sized read window.
pub(crate) fn source_window_for_tile(
    target: TileRect,
    target_to_source: &Matrix3<f64>,
    source_width: u32,
    source_height: u32,
    filter_halo: u32,
) -> Option<SourceReadWindow> {
    if target.is_empty() || source_width == 0 || source_height == 0 {
        return None;
    }

    let x0 = f64::from(target.x);
    let y0 = f64::from(target.y);
    let x1 = target.end_x() as f64;
    let y1 = target.end_y() as f64;
    let points = [
        Point2::new(x0, y0),
        Point2::new(x1, y0),
        Point2::new(x0, y1),
        Point2::new(x1, y1),
        Point2::new((x0 + x1) * 0.5, (y0 + y1) * 0.5),
    ];
    let depths = points
        .iter()
        .map(|point| {
            let projected = target_to_source * Point3::new(point.x, point.y, 1.0);
            projected.z
        })
        .collect::<Vec<_>>();
    if depths
        .iter()
        .any(|depth| !depth.is_finite() || depth.abs() < 1e-12)
    {
        return None;
    }
    let has_positive_depth = depths.iter().any(|depth| *depth > 0.0);
    let has_negative_depth = depths.iter().any(|depth| *depth < 0.0);
    if has_positive_depth && has_negative_depth {
        // A linear projective denominator that changes sign over the tile has
        // a pole inside it. Corner-only bounds would under-read that tile.
        return None;
    }

    let mut mapped = points
        .into_iter()
        .map(|point| project(target_to_source, point));
    let first = mapped.next()??;
    let (mut min_x, mut max_x) = (first.x, first.x);
    let (mut min_y, mut max_y) = (first.y, first.y);
    for point in mapped {
        let point = point?;
        min_x = min_x.min(point.x);
        max_x = max_x.max(point.x);
        min_y = min_y.min(point.y);
        max_y = max_y.max(point.y);
    }

    let halo = f64::from(filter_halo);
    min_x = (min_x - halo).floor().max(0.0);
    min_y = (min_y - halo).floor().max(0.0);
    max_x = (max_x + halo).ceil().min(f64::from(source_width));
    max_y = (max_y + halo).ceil().min(f64::from(source_height));
    if !(max_x > min_x && max_y > min_y) {
        return None;
    }

    Some(SourceReadWindow::new(
        min_x as u32,
        min_y as u32,
        (max_x - min_x) as u32,
        (max_y - min_y) as u32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_plan_is_row_major_and_clips_edge_tiles() {
        let plan = TilePlan::default_for(8193, 4097).expect("valid plan");
        assert_eq!(plan.filter_halo, DEFAULT_FILTER_HALO);
        assert_eq!(plan.tiles.len(), 6);
        assert_eq!(
            plan.tiles,
            vec![
                TileRect::new(0, 0, 4096, 4096),
                TileRect::new(4096, 0, 4096, 4096),
                TileRect::new(8192, 0, 1, 4096),
                TileRect::new(0, 4096, 4096, 1),
                TileRect::new(4096, 4096, 4096, 1),
                TileRect::new(8192, 4096, 1, 1),
            ]
        );
    }

    #[test]
    fn source_window_expands_halo_and_clamps_to_source() {
        let target = TileRect::new(10, 20, 30, 40);
        let matrix = Matrix3::identity();
        let window = source_window_for_tile(target, &matrix, 32, 64, 2).expect("window");
        assert_eq!(window, TileRect::new(8, 18, 24, 44));
    }

    #[test]
    fn source_window_uses_projective_inverse_corners() {
        let target = TileRect::new(100, 50, 20, 10);
        let matrix = Matrix3::new(
            2.0, 0.0, -10.0, // x_source = 2*x - 10
            0.0, 3.0, 5.0, // y_source = 3*y + 5
            0.0, 0.0, 1.0,
        );
        let window = source_window_for_tile(target, &matrix, 512, 512, 0).expect("window");
        assert_eq!(window, TileRect::new(190, 155, 40, 30));
    }

    #[test]
    fn source_window_rejects_a_projective_singularity() {
        let target = TileRect::new(0, 0, 10, 10);
        let matrix = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        assert!(source_window_for_tile(target, &matrix, 100, 100, 2).is_none());
    }

    #[test]
    fn source_window_rejects_a_denominator_crossing_between_samples() {
        let target = TileRect::new(0, 0, 10, 10);
        let matrix = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.15, 0.0, 1.0);
        assert!(source_window_for_tile(target, &matrix, 100, 100, 2).is_none());
    }
}
