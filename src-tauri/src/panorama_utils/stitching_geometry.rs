//! Projection and focus-stack geometry helpers.
//!
//! This module owns pure coordinate/bounds calculations used by the
//! compositor and station renderers. It deliberately keeps source decoding
//! and ownership decisions in the parent stitching module.

use super::super::stack_pipeline::compositor::tile_coordinate_with_residual;
use super::super::stack_pipeline::residual_warp;
use super::{FocusLayerWarp, ImageInfo};
use nalgebra::{Matrix3, Point2, Point3};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    Planar,
    Cylindrical,
    Spherical,
}

pub fn project_point(
    image: &ImageInfo,
    x: f64,
    y: f64,
    projection: Projection,
) -> Option<Point2<f64>> {
    let width = image.width() as f64;
    let height = image.height() as f64;
    let center_x = width * 0.5;
    let center_y = height * 0.5;
    let focal = width.max(height).max(1.0) * 0.85;
    let normalized_x = (x - center_x) / focal;
    let normalized_y = (y - center_y) / focal;

    let (projected_x, projected_y) = match projection {
        Projection::Planar => (x, y),
        Projection::Cylindrical => (
            normalized_x.atan() * focal + center_x,
            (normalized_y / (1.0 + normalized_x * normalized_x).sqrt()) * focal + center_y,
        ),
        Projection::Spherical => (
            normalized_x.atan() * focal + center_x,
            (normalized_y / (1.0 + normalized_x * normalized_x).sqrt()).atan() * focal + center_y,
        ),
    };

    if projected_x.is_finite() && projected_y.is_finite() {
        Some(Point2::new(projected_x, projected_y))
    } else {
        None
    }
}

pub(super) fn unproject_point(
    image: &ImageInfo,
    x: f64,
    y: f64,
    projection: Projection,
) -> Option<Point2<f64>> {
    if projection == Projection::Planar {
        return if x.is_finite() && y.is_finite() {
            Some(Point2::new(x, y))
        } else {
            None
        };
    }

    let width = image.width() as f64;
    let height = image.height() as f64;
    let center_x = width * 0.5;
    let center_y = height * 0.5;
    let focal = width.max(height).max(1.0) * 0.85;
    let projected_x = (x - center_x) / focal;
    let projected_y = (y - center_y) / focal;

    let normalized_x = projected_x.tan();
    let vertical_scale = (1.0 + normalized_x * normalized_x).sqrt();
    let normalized_y = match projection {
        Projection::Cylindrical => projected_y * vertical_scale,
        Projection::Spherical => projected_y.tan() * vertical_scale,
        Projection::Planar => unreachable!("planar projection returned above"),
    };
    let source_x = normalized_x * focal + center_x;
    let source_y = normalized_y * focal + center_y;

    if source_x.is_finite() && source_y.is_finite() {
        Some(Point2::new(source_x, source_y))
    } else {
        None
    }
}

pub(crate) fn map_target_to_source(
    inverse_homography: &Matrix3<f64>,
    target: Point3<f64>,
    image: &ImageInfo,
    projection: Projection,
) -> Option<Point2<f64>> {
    let projected_source = inverse_homography * target;
    if projected_source.z.abs() < 1e-8 {
        return None;
    }
    unproject_point(
        image,
        projected_source.x / projected_source.z,
        projected_source.y / projected_source.z,
        projection,
    )
}

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
    if residual.identity() {
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

pub(crate) fn output_bounds(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for &image in images {
        let h = global_homographies[&image.id];
        let (width, height) = image.dimensions();
        let corners = [
            (0.0, 0.0),
            (width as f64, 0.0),
            (width as f64, height as f64),
            (0.0, height as f64),
        ];
        for (x, y) in corners {
            let Some(projected) = project_point(image, x, y, projection) else {
                continue;
            };
            let mapped = h * Point3::new(projected.x, projected.y, 1.0);
            if mapped.z.abs() < 1e-8 {
                continue;
            }
            let mapped_x = mapped.x / mapped.z;
            let mapped_y = mapped.y / mapped.z;
            min_x = min_x.min(mapped_x);
            max_x = max_x.max(mapped_x);
            min_y = min_y.min(mapped_y);
            max_y = max_y.max(mapped_y);
        }
    }

    (min_x, max_x, min_y, max_y)
}

pub(crate) fn pixel_aligned_canvas(minimum: f64, maximum: f64) -> (f64, u32) {
    // The first image is the global reference and normally has an identity
    // transform. Keep its samples on integer output coordinates; using the exact
    // fractional bound as the offset would unnecessarily interpolate every
    // reference pixel.
    let offset = (-minimum).ceil();
    let size = (maximum + offset).ceil().max(1.0) as u32;
    (offset, size)
}

pub(crate) fn output_canvas_dimensions(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> (u32, u32) {
    if images.is_empty() {
        return (0, 0);
    }
    let (min_x, max_x, min_y, max_y) = output_bounds(images, global_homographies, projection);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return (0, 0);
    }
    let (_, width) = pixel_aligned_canvas(min_x, max_x);
    let (_, height) = pixel_aligned_canvas(min_y, max_y);
    (width, height)
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct FocusVirtualTileGeometry {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tile_to_world: Matrix3<f64>,
}

/// Return the projected bounds needed to place one virtual tile without
/// decoding or fusing any source pixels.  This is used by the bounded-memory
/// compositor to create tile metadata first and render each group lazily.
pub(crate) fn focus_stack_virtual_tile_geometry(
    group: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
) -> Option<FocusVirtualTileGeometry> {
    if group.is_empty() {
        return None;
    }
    if group
        .iter()
        .any(|image| !global_homographies.contains_key(&image.id))
    {
        return None;
    }
    let (min_x, max_x, min_y, max_y) =
        focus_output_bounds(group, global_homographies, projection, focus_warp);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return None;
    }
    let (offset_x, width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, height) = pixel_aligned_canvas(min_y, max_y);
    Some(FocusVirtualTileGeometry {
        width,
        height,
        tile_to_world: Matrix3::new(1.0, 0.0, -offset_x, 0.0, 1.0, -offset_y, 0.0, 0.0, 1.0),
    })
}
pub(crate) fn transformed_image_region(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    if out_width == 0 || out_height == 0 {
        return None;
    }
    let (width, height) = image.dimensions();
    let corners = [
        (0.0, 0.0),
        (width as f64, 0.0),
        (width as f64, height as f64),
        (0.0, height as f64),
    ];
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in corners {
        let projected = project_point(image, x, y, projection)?;
        let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            continue;
        }
        let mapped_x = mapped.x / mapped.z + offset_x;
        let mapped_y = mapped.y / mapped.z + offset_y;
        if !mapped_x.is_finite() || !mapped_y.is_finite() {
            continue;
        }
        min_x = min_x.min(mapped_x);
        max_x = max_x.max(mapped_x);
        min_y = min_y.min(mapped_y);
        max_y = max_y.max(mapped_y);
    }
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return None;
    }
    let left = min_x.floor().max(0.0).min((out_width - 1) as f64) as u32;
    let right = max_x.ceil().max(0.0).min((out_width - 1) as f64) as u32;
    let top = min_y.floor().max(0.0).min((out_height - 1) as f64) as u32;
    let bottom = max_y.ceil().max(0.0).min((out_height - 1) as f64) as u32;
    (left <= right && top <= bottom).then_some((left, right, top, bottom))
}

pub(super) fn transformed_source_bounds(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    minimum_source_y: f64,
    maximum_source_y: f64,
) -> Option<(f64, f64, f64, f64)> {
    transformed_source_bounds_xy(
        image,
        homography,
        projection,
        0.0,
        1.0,
        minimum_source_y,
        maximum_source_y,
    )
}

pub(super) fn transformed_source_bounds_xy(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    minimum_source_x: f64,
    maximum_source_x: f64,
    minimum_source_y: f64,
    maximum_source_y: f64,
) -> Option<(f64, f64, f64, f64)> {
    let (width, height) = image.dimensions();
    let x0 = (width as f64 * minimum_source_x).clamp(0.0, width as f64);
    let x1 = (width as f64 * maximum_source_x).clamp(0.0, width as f64);
    let y0 = (height as f64 * minimum_source_y).clamp(0.0, height as f64);
    let y1 = (height as f64 * maximum_source_y).clamp(0.0, height as f64);
    let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in corners {
        let projected = project_point(image, x, y, projection)?;
        let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            continue;
        }
        let mapped_x = mapped.x / mapped.z;
        let mapped_y = mapped.y / mapped.z;
        if !mapped_x.is_finite() || !mapped_y.is_finite() {
            continue;
        }
        min_x = min_x.min(mapped_x);
        max_x = max_x.max(mapped_x);
        min_y = min_y.min(mapped_y);
        max_y = max_y.max(mapped_y);
    }
    if min_x.is_finite() && max_x.is_finite() && min_y.is_finite() && max_y.is_finite() {
        Some((min_x, max_x, min_y, max_y))
    } else {
        None
    }
}

pub(super) fn focus_output_bounds(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
) -> (f64, f64, f64, f64) {
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        output_bounds(images, global_homographies, projection);
    let Some(focus_warp) = focus_warp else {
        return (min_x, max_x, min_y, max_y);
    };
    for image in images {
        for band in &focus_warp.bands {
            let Some(homography) = band.homographies.get(&image.id) else {
                continue;
            };
            let Some(&(minimum_source_y, maximum_source_y)) = band.source_ranges.get(&image.id)
            else {
                continue;
            };
            let (minimum_source_x, maximum_source_x) = band
                .source_x_ranges
                .get(&image.id)
                .copied()
                .unwrap_or((0.0, 1.0));
            let Some((band_min_x, band_max_x, band_min_y, band_max_y)) =
                transformed_source_bounds_xy(
                    image,
                    homography,
                    projection,
                    minimum_source_x,
                    maximum_source_x,
                    minimum_source_y,
                    maximum_source_y,
                )
            else {
                continue;
            };
            min_x = min_x.min(band_min_x);
            max_x = max_x.max(band_max_x);
            min_y = min_y.min(band_min_y);
            max_y = max_y.max(band_max_y);
        }
    }
    (min_x, max_x, min_y, max_y)
}
