//! Focus-stack geometry classification helpers.
//!
//! This module detects whether a stack behaves like a shifted mosaic from
//! mapped frame geometry and focal metadata. It does not inspect pixels or
//! change the compositor decision thresholds.

use super::coordinates::{Projection, project_point};
use super::shared::ImageInfo;
use nalgebra::{Matrix3, Point2, Point3};
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct MappedFrameGeometry {
    center: Point2<f64>,
    scale: f64,
    rotation: f64,
    area: f64,
}

fn mapped_frame_geometry(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Option<MappedFrameGeometry> {
    let width = image.width.max(1) as f64;
    let height = image.height.max(1) as f64;
    let source_corners = [
        Point2::new(0.0, 0.0),
        Point2::new(width, 0.0),
        Point2::new(width, height),
        Point2::new(0.0, height),
    ];
    let mut corners = [Point2::new(0.0, 0.0); 4];
    for (slot, source) in source_corners.into_iter().enumerate() {
        let projected = project_point(image, source.x, source.y, projection)?;
        let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            return None;
        }
        let point = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
        if !point.x.is_finite() || !point.y.is_finite() {
            return None;
        }
        corners[slot] = point;
    }

    let center = corners
        .iter()
        .copied()
        .fold(Point2::new(0.0, 0.0), |sum, point| sum + point.coords)
        / 4.0;
    let horizontal_scale =
        ((corners[1] - corners[0]).norm() + (corners[2] - corners[3]).norm()) / (2.0 * width);
    let vertical_scale =
        ((corners[3] - corners[0]).norm() + (corners[2] - corners[1]).norm()) / (2.0 * height);
    let scale = (horizontal_scale.max(0.0) * vertical_scale.max(0.0)).sqrt();
    let top_edge = corners[1] - corners[0];
    let rotation = top_edge.y.atan2(top_edge.x);
    let signed_double_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>();
    let area = (signed_double_area / (2.0 * width * height)).abs();
    (scale.is_finite() && rotation.is_finite() && area.is_finite()).then_some(MappedFrameGeometry {
        center,
        scale,
        rotation,
        area,
    })
}

fn wrapped_angle_delta(left: f64, right: f64) -> f64 {
    let mut delta = left - right;
    while delta > std::f64::consts::PI {
        delta -= 2.0 * std::f64::consts::PI;
    }
    while delta < -std::f64::consts::PI {
        delta += 2.0 * std::f64::consts::PI;
    }
    delta.abs()
}

pub(super) fn focus_stack_is_shifted_mosaic(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> bool {
    let Some(first) = images.first() else {
        return false;
    };
    let Some(reference_homography) = global_homographies.get(&first.id) else {
        return false;
    };
    let Some(reference) = mapped_frame_geometry(first, reference_homography, projection) else {
        return false;
    };
    let reference_width = first.width().max(1) as f64;
    let reference_height = first.height().max(1) as f64;
    // Some exported frames lose the equivalent-focal tag. Use the first tag
    // available in the stack as a comparison anchor so a missing tag on the
    // first frame does not hide a later 35mm/85mm switch.
    let reference_focal = first
        .focal_length_35mm
        .or_else(|| images.iter().find_map(|image| image.focal_length_35mm));
    images.iter().skip(1).any(|image| {
        let Some(homography) = global_homographies.get(&image.id) else {
            return false;
        };
        let Some(current) = mapped_frame_geometry(image, homography, projection) else {
            return false;
        };
        let center_shift = (current.center.x - reference.center.x).abs() > reference_width * 0.08
            || (current.center.y - reference.center.y).abs() > reference_height * 0.08;
        let relative_scale = current.scale / reference.scale.max(1e-8);
        let scale_shift = relative_scale.is_finite() && !(0.90..=1.10).contains(&relative_scale);
        let relative_area = current.area / reference.area.max(1e-8);
        let area_shift = relative_area.is_finite() && !(0.82..=1.22).contains(&relative_area);
        let rotation_shift = wrapped_angle_delta(current.rotation, reference.rotation) > 0.035;
        let focal_shift = reference_focal
            .zip(image.focal_length_35mm)
            .map(|(reference_focal, current_focal)| {
                let ratio = current_focal / reference_focal.max(1e-8);
                ratio.is_finite() && !(0.90..=1.10).contains(&ratio)
            })
            .unwrap_or(false);
        center_shift || scale_shift || area_shift || rotation_shift || focal_shift
    })
}
