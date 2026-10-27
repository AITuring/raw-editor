//! Rendering-space geometry shared by panorama and focus compositors.
//!
//! These helpers only transform source rectangles into output-world bounds. They
//! deliberately do not sample pixels or choose ownership, so moving them out of
//! the stitching facade does not change the render algorithm.

use super::coordinates::{Projection, output_bounds, project_point};
use super::shared::{FocusLayerWarp, ImageInfo};
use nalgebra::{Matrix3, Point3};
use std::collections::HashMap;

pub(super) fn transformed_image_region(
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

pub(super) fn focus_transformed_image_region(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    if out_width == 0 || out_height == 0 {
        return None;
    }
    let Some((mut min_x, mut max_x, mut min_y, mut max_y)) =
        transformed_source_bounds(image, homography, projection, 0.0, 1.0)
    else {
        return None;
    };
    if let Some(focus_warp) = focus_warp {
        for band in &focus_warp.bands {
            let Some(band_homography) = band.homographies.get(&image.id) else {
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
                    band_homography,
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
    let min_x = min_x + offset_x;
    let max_x = max_x + offset_x;
    let min_y = min_y + offset_y;
    let max_y = max_y + offset_y;
    let left = min_x.floor().max(0.0).min((out_width - 1) as f64) as u32;
    let right = max_x.ceil().max(0.0).min((out_width - 1) as f64) as u32;
    let top = min_y.floor().max(0.0).min((out_height - 1) as f64) as u32;
    let bottom = max_y.ceil().max(0.0).min((out_height - 1) as f64) as u32;
    (left <= right && top <= bottom).then_some((left, right, top, bottom))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    fn image_info() -> ImageInfo {
        ImageInfo {
            id: 0,
            filename: "/test/geometry.jpg".to_string(),
            width: 100,
            height: 50,
            alignment_image: GrayImage::new(100, 50),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }

    #[test]
    fn transformed_region_clamps_to_output_canvas() {
        let image = image_info();
        let region = transformed_image_region(
            &image,
            &Matrix3::identity(),
            Projection::Planar,
            10.0,
            20.0,
            120,
            90,
        );
        assert_eq!(region, Some((10, 110, 20, 70)));
    }

    #[test]
    fn source_bounds_respect_fractional_source_window() {
        let image = image_info();
        let bounds =
            transformed_source_bounds(&image, &Matrix3::identity(), Projection::Planar, 0.2, 0.8);
        assert_eq!(bounds, Some((0.0, 100.0, 10.0, 40.0)));
    }
}
