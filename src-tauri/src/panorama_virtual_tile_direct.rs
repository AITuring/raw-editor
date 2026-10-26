//! Virtual-tile direct geometry helpers.

use super::*;

pub(crate) const VIRTUAL_TILE_DIRECT_PATCH_RADIUS: i32 = 6;
pub(crate) const VIRTUAL_TILE_DIRECT_GRID_SPACING_PX: f64 = 24.0;
pub(crate) const VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES: usize = 1_024;
pub(crate) const VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS: usize = 64;
pub(crate) const VIRTUAL_TILE_DIRECT_MAX_SEARCH_RADIUS_PX: f64 = 16.0;
pub(crate) const VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX: f64 = 1.0;
pub(crate) const VIRTUAL_TILE_DIRECT_PATCH_GATES: registration::PatchMatchGates =
    registration::PatchMatchGates {
        min_correlation: VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC,
        min_peak_margin: VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN,
        min_source_variance: 2.0,
        min_target_variance: 1.0,
    };

pub(crate) fn virtual_tile_direct_grid_points(bounds: (f64, f64, f64, f64)) -> Vec<Point2<f64>> {
    let (min_x, min_y, max_x, max_y) = bounds;
    let width = max_x - min_x;
    let height = max_y - min_y;
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return Vec::new();
    }
    let area = width * height;
    let desired = (area
        / (VIRTUAL_TILE_DIRECT_GRID_SPACING_PX * VIRTUAL_TILE_DIRECT_GRID_SPACING_PX))
        .ceil()
        .clamp(
            STATION_RELATION_MIN_INLIERS as f64,
            VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES as f64,
        ) as usize;
    let aspect = width / height;
    let mut columns = ((desired as f64 * aspect).sqrt().round() as usize)
        .clamp(1, VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS);
    let mut rows = desired
        .div_ceil(columns)
        .clamp(1, VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS);
    while columns * rows > VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES {
        if columns >= rows && columns > 1 {
            columns -= 1;
        } else if rows > 1 {
            rows -= 1;
        } else {
            break;
        }
    }
    let mut points = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        let y = min_y + (row as f64 + 0.5) * height / rows as f64;
        for column in 0..columns {
            let x = min_x + (column as f64 + 0.5) * width / columns as f64;
            points.push(Point2::new(x, y));
        }
    }
    points
}

pub(crate) fn virtual_tile_coverage_contains_rectangle(
    coverage: &stitching::CoverageMask,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> bool {
    let (width, height) = coverage.dimensions();
    if ![min_x, min_y, max_x, max_y]
        .iter()
        .all(|value| value.is_finite())
        || min_x < 0.0
        || min_y < 0.0
        || max_x >= f64::from(width.saturating_sub(1))
        || max_y >= f64::from(height.saturating_sub(1))
        || max_x < min_x
        || max_y < min_y
    {
        return false;
    }
    let start_x = min_x.floor() as u32;
    let end_x = max_x.ceil() as u32;
    let start_y = min_y.floor() as u32;
    let end_y = max_y.ceil() as u32;
    (start_y..=end_y).all(|y| (start_x..=end_x).all(|x| coverage.is_covered(x, y)))
}

pub(crate) fn virtual_tile_direct_patch_support_is_covered(
    source_coverage: &stitching::CoverageMask,
    target_coverage: &stitching::CoverageMask,
    source_to_target: &Matrix3<f64>,
    center: Point2<f64>,
    patch_radius: i32,
    search_radius: i32,
) -> bool {
    let Some(jacobian) = virtual_tile_local_jacobian(source_to_target, center) else {
        return false;
    };
    let scale = (jacobian.column(0).norm() * jacobian.column(1).norm()).sqrt();
    if !(0.15..6.0).contains(&scale) {
        return false;
    }
    let spacing = scale.recip().max(1.0);
    // Include one pixel for gradient sampling and another for bilinear access.
    let source_extent = f64::from(patch_radius) * spacing + 2.0;
    if !virtual_tile_coverage_contains_rectangle(
        source_coverage,
        center.x - source_extent,
        center.y - source_extent,
        center.x + source_extent,
        center.y + source_extent,
    ) {
        return false;
    }
    let corners = [
        Point2::new(center.x - source_extent, center.y - source_extent),
        Point2::new(center.x + source_extent, center.y - source_extent),
        Point2::new(center.x + source_extent, center.y + source_extent),
        Point2::new(center.x - source_extent, center.y + source_extent),
    ];
    let Some(projected) = corners
        .into_iter()
        .map(|point| transformed_point(source_to_target, point))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let padding = f64::from(search_radius) + 2.0;
    let min_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min)
        - padding;
    let max_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max)
        + padding;
    let min_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min)
        - padding;
    let max_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max)
        + padding;
    virtual_tile_coverage_contains_rectangle(target_coverage, min_x, min_y, max_x, max_y)
}

pub(crate) fn virtual_tile_round_trip_is_consistent(
    source: Point2<f64>,
    returned: Point2<f64>,
) -> bool {
    let distance = (returned - source).norm();
    distance.is_finite() && distance <= VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX
}

/// Invert the *measured* local forward map, not merely the rough station prior.
///
/// `probe_warped_patch` returns `target = H_initial(source) + residual` in the
/// target tile's coordinates. The corresponding local forward homography is
/// therefore `T(residual) * H_initial`; its inverse is the only transform whose
/// source/target convention maps that measured target back to the original
/// source. Using `H_initial.inverse()` here incorrectly asks the reverse probe
/// to rediscover the already measured residual and made valid pairs depend on
/// which station happened to be the source (任务 10 checkpoint).
pub(crate) fn virtual_tile_measured_reverse_transform(
    initial_source_to_target: &Matrix3<f64>,
    source: Point2<f64>,
    measured_target: Point2<f64>,
) -> Option<Matrix3<f64>> {
    let predicted_target = transformed_point(initial_source_to_target, source)?;
    let residual = measured_target - predicted_target;
    if !residual.iter().all(|value| value.is_finite()) {
        return None;
    }
    let residual_translation =
        Matrix3::new(1.0, 0.0, residual.x, 0.0, 1.0, residual.y, 0.0, 0.0, 1.0);
    (residual_translation * initial_source_to_target).try_inverse()
}

pub(crate) const VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX: f64 = 8.0;
pub(crate) const VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX: f64 = 24.0;
pub(crate) const VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MIN: f64 = 0.98;
pub(crate) const VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MAX: f64 = 1.02;
