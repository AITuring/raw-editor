use super::shared::ImageInfo;
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

pub(crate) fn unproject_point(
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    fn image_info(id: usize, width: u32, height: u32) -> ImageInfo {
        ImageInfo {
            id,
            filename: format!("/test/{id}.jpg"),
            width,
            height,
            alignment_image: GrayImage::new(width, height),
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
    fn planar_projection_is_identity() {
        let image = image_info(0, 100, 80);
        let point = project_point(&image, 12.5, 33.25, Projection::Planar).unwrap();
        assert_eq!(point, Point2::new(12.5, 33.25));
        assert_eq!(
            unproject_point(&image, point.x, point.y, Projection::Planar),
            Some(point)
        );
    }

    #[test]
    fn curved_projection_round_trips_inside_working_range() {
        let image = image_info(0, 400, 300);
        let center = Point2::new(200.0, 150.0);
        for projection in [Projection::Cylindrical, Projection::Spherical] {
            let source = Point2::new(center.x + 40.0, center.y + 24.0);
            let projected = project_point(&image, source.x, source.y, projection).unwrap();
            let restored = unproject_point(&image, projected.x, projected.y, projection).unwrap();
            assert!((restored.x - source.x).abs() < 1e-8, "{projection:?}: x");
            assert!((restored.y - source.y).abs() < 1e-8, "{projection:?}: y");
        }
    }

    #[test]
    fn translated_canvas_uses_projected_union() {
        let image = image_info(0, 100, 50);
        let images = [&image];
        let transform = nalgebra::Translation2::new(20.0, -5.0).to_homogeneous();
        let transforms = HashMap::from([(0usize, transform)]);
        assert_eq!(
            output_bounds(&images, &transforms, Projection::Planar),
            (20.0, 120.0, -5.0, 45.0)
        );
        assert_eq!(
            output_canvas_dimensions(&images, &transforms, Projection::Planar),
            (100, 50)
        );
    }
}
