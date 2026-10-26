//! Station relation measurements, overlap geometry, and relation acceptance.
//!
//! This module is part of the station-solver facade and preserves the
//! production helper visibility used by the focus-stack pipeline.

use super::*;

#[derive(Clone, Copy, Debug)]

pub(crate) struct StationRelationMeasurements {
    pub(crate) inliers: usize,
    pub(crate) median_error_px: f64,
    pub(crate) scale_ratio: f64,
    pub(crate) spatial_support: f64,
    pub(crate) low_frequency_mean_relative_difference: f64,
    pub(crate) edge_strength_ratio: f64,
    pub(crate) median_edge_orientation_difference_degrees: f64,
    pub(crate) preserves_convex_orientation: bool,
}

pub(crate) fn station_relation_rejection_reasons(
    measurements: &StationRelationMeasurements,
) -> Vec<&'static str> {
    let mut reasons = Vec::with_capacity(8);
    if measurements.inliers < STATION_RELATION_MIN_INLIERS {
        reasons.push(degradation::STATION_RELATION_LOW_INLIERS);
    }
    if !measurements.median_error_px.is_finite()
        || measurements.median_error_px > STATION_RELATION_MAX_MEDIAN_ERROR_PX
    {
        reasons.push(degradation::STATION_RELATION_HIGH_REPROJECTION_ERROR);
    }
    if !measurements.scale_ratio.is_finite()
        || !(STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
            .contains(&measurements.scale_ratio)
    {
        reasons.push(degradation::STATION_RELATION_SCALE_OUT_OF_RANGE);
    }
    if !measurements.spatial_support.is_finite()
        || measurements.spatial_support < STATION_RELATION_MIN_SPATIAL_SUPPORT
    {
        reasons.push(degradation::STATION_RELATION_LOW_SPATIAL_SUPPORT);
    }
    if !measurements
        .low_frequency_mean_relative_difference
        .is_finite()
        || measurements.low_frequency_mean_relative_difference
            > STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE
    {
        reasons.push(degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH);
    }
    if !measurements.edge_strength_ratio.is_finite()
        || !(STATION_RELATION_EDGE_STRENGTH_RATIO_MIN..=STATION_RELATION_EDGE_STRENGTH_RATIO_MAX)
            .contains(&measurements.edge_strength_ratio)
    {
        reasons.push(degradation::STATION_RELATION_EDGE_STRENGTH_MISMATCH);
    }
    if !measurements
        .median_edge_orientation_difference_degrees
        .is_finite()
        || measurements.median_edge_orientation_difference_degrees
            > STATION_RELATION_MAX_MEDIAN_EDGE_ORIENTATION_DIFFERENCE_DEGREES
    {
        reasons.push(degradation::STATION_RELATION_EDGE_ORIENTATION_MISMATCH);
    }
    if !measurements.preserves_convex_orientation {
        reasons.push(degradation::STATION_RELATION_NON_CONVEX_QUAD);
    }
    reasons
}

/// Apply the production Station_Relation rejection decision to its report.
///
/// One candidate increments `rejected_count` once, while every independently
/// failed criterion increments its stable reason bucket.  Keeping this at the
/// solver boundary makes combined defects visible without allowing a rejected
/// candidate to reach pose construction (requirements 6.5, 6.9, 6.10).
pub(crate) fn station_relation_is_accepted_with_report(
    left: usize,
    right: usize,
    consensus_error_px: f64,
    measurements: &StationRelationMeasurements,
    report: &mut stack_report::StationRelationsReport,
) -> bool {
    let reasons = station_relation_rejection_reasons(measurements);
    if reasons.is_empty() {
        return true;
    }
    report.rejected_count += 1;
    for reason in reasons {
        *report
            .rejected_by_reason
            .entry(reason.to_string())
            .or_default() += 1;
        degradation::record_run_degradation(
            reason,
            serde_json::json!({
                "left": left,
                "right": right,
                "inliers": measurements.inliers,
                "median_error_px": measurements.median_error_px,
                "consensus_error_px": consensus_error_px,
                "scale_ratio": measurements.scale_ratio,
                "spatial_support": measurements.spatial_support,
                "low_frequency_mean_relative_difference": measurements.low_frequency_mean_relative_difference,
                "edge_strength_ratio": measurements.edge_strength_ratio,
                "median_edge_orientation_difference_degrees": measurements.median_edge_orientation_difference_degrees,
                "preserves_convex_orientation": measurements.preserves_convex_orientation,
            }),
        );
    }
    false
}

/// A topology candidate without a measured model was rejected, but no
/// Requirement 6 acceptance criterion was measured. Its matcher diagnostics
/// supply the separate unmeasured and failure-stage summaries at publication.
pub(crate) fn record_station_relation_no_fit(
    _left: usize,
    _right: usize,
    report: &mut stack_report::StationRelationsReport,
) {
    report.rejected_count += 1;
}

pub(crate) fn polygon_signed_double_area(points: &[Point2<f64>]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum()
}

pub(crate) fn convex_hull_area(points: impl IntoIterator<Item = Point2<f64>>) -> f64 {
    polygon_signed_double_area(&convex_hull_vertices(points)).abs() * 0.5
}

/// Counter-clockwise convex hull vertices (Andrew's monotone chain); fewer
/// than three distinct finite points give an empty hull.
pub(crate) fn convex_hull_vertices(
    points: impl IntoIterator<Item = Point2<f64>>,
) -> Vec<Point2<f64>> {
    let mut points = points
        .into_iter()
        .filter(|point| point.x.is_finite() && point.y.is_finite())
        .collect::<Vec<_>>();
    points.sort_by(|left, right| {
        left.x
            .total_cmp(&right.x)
            .then_with(|| left.y.total_cmp(&right.y))
    });
    points.dedup_by(|left, right| left.x == right.x && left.y == right.y);
    if points.len() < 3 {
        return Vec::new();
    }
    let cross = |origin: Point2<f64>, left: Point2<f64>, right: Point2<f64>| {
        (left.x - origin.x) * (right.y - origin.y) - (left.y - origin.y) * (right.x - origin.x)
    };
    let mut lower = Vec::with_capacity(points.len());
    for &point in &points {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0.0
        {
            lower.pop();
        }
        lower.push(point);
    }
    let mut upper = Vec::with_capacity(points.len());
    for &point in points.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0.0
        {
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

pub(crate) fn clip_polygon_to_rectangle(
    polygon: &[Point2<f64>],
    dimensions: (u32, u32),
) -> Vec<Point2<f64>> {
    let mut clipped = polygon.to_vec();
    let boundaries: [(usize, f64, bool); 4] = [
        (0, 0.0, true),
        (0, dimensions.0 as f64, false),
        (1, 0.0, true),
        (1, dimensions.1 as f64, false),
    ];
    for (axis, limit, keep_greater) in boundaries {
        if clipped.is_empty() {
            break;
        }
        let input = std::mem::take(&mut clipped);
        let coordinate = |point: Point2<f64>| if axis == 0 { point.x } else { point.y };
        let inside = |point: Point2<f64>| {
            if keep_greater {
                coordinate(point) >= limit
            } else {
                coordinate(point) <= limit
            }
        };
        for index in 0..input.len() {
            let current = input[index];
            let previous = input[(index + input.len() - 1) % input.len()];
            let current_inside = inside(current);
            let previous_inside = inside(previous);
            if current_inside != previous_inside {
                let previous_coordinate = coordinate(previous);
                let current_coordinate = coordinate(current);
                let denominator = current_coordinate - previous_coordinate;
                if denominator.abs() > f64::EPSILON {
                    let t = (limit - previous_coordinate) / denominator;
                    clipped.push(previous + (current - previous) * t);
                }
            }
            if current_inside {
                clipped.push(current);
            }
        }
    }
    clipped
}

pub(crate) fn station_relation_overlap_area(
    source_to_target: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    source_side: bool,
) -> Option<f64> {
    let target_to_source = source_to_target.try_inverse()?;
    let (footprint_dimensions, clip_dimensions, transform) = if source_side {
        (target_dimensions, source_dimensions, target_to_source)
    } else {
        (source_dimensions, target_dimensions, *source_to_target)
    };
    let footprint = [
        Point2::new(0.0, 0.0),
        Point2::new(footprint_dimensions.0 as f64, 0.0),
        Point2::new(footprint_dimensions.0 as f64, footprint_dimensions.1 as f64),
        Point2::new(0.0, footprint_dimensions.1 as f64),
    ]
    .into_iter()
    .map(|point| transformed_point(&transform, point))
    .collect::<Option<Vec<_>>>()?;
    let clipped = clip_polygon_to_rectangle(&footprint, clip_dimensions);
    let area = polygon_signed_double_area(&clipped).abs() * 0.5;
    (area.is_finite() && area > f64::EPSILON).then_some(area)
}

/// What requirement 6.4 counts as a relation's overlap area.
#[derive(Clone, Copy)]
pub(crate) enum StationOverlapMeasure<'a> {
    /// Source-image evidence: the geometric intersection of the two frames,
    /// every pixel of which is image content.
    Rectangles,
    /// Virtual_Tile evidence: only pixels both Coverage_Masks cover. A tile's
    /// bounding rectangle also holds uncovered corners, which carry no content,
    /// can never hold an inlier and are never composited from both tiles
    /// (需求 6.1). Mask pixel `p` is the block `[f·p, f·p + f)` of the
    /// relation's pixels (`f = factor`, the [`VirtualTileMatchPlane`] layout).
    Covered {
        source: &'a stitching::CoverageMask,
        target: &'a stitching::CoverageMask,
        factor: u32,
    },
}

/// Area, in the relation's pixels, of the source-plane pixels covered in
/// `source_mask` whose image under `source_to_target` lands on a pixel
/// covered in `target_mask`. `None` when a mask does not tile its plane.
pub(crate) fn virtual_tile_covered_overlap_area(
    source_mask: &stitching::CoverageMask,
    source_dimensions: (u32, u32),
    target_mask: &stitching::CoverageMask,
    target_dimensions: (u32, u32),
    factor: u32,
    source_to_target: &Matrix3<f64>,
) -> Option<f64> {
    let factor = factor.max(1);
    let tiles_plane = |mask: &stitching::CoverageMask, (width, height): (u32, u32)| {
        mask.dimensions() == (width / factor, height / factor)
    };
    if !tiles_plane(source_mask, source_dimensions) || !tiles_plane(target_mask, target_dimensions)
    {
        return None;
    }
    let scale = f64::from(factor);
    let offset = (scale - 1.0) * 0.5;
    let (width, height) = source_mask.dimensions();
    let (target_width, target_height) = target_mask.dimensions();
    // Rows are independent and integer counts add exactly, so the parallel
    // reduction does not depend on scheduling.
    let covered = (0..height)
        .into_par_iter()
        .map(|y| {
            (0..width)
                .filter(|&x| source_mask.is_covered(x, y))
                .filter(|&x| {
                    let centre =
                        Point2::new(scale * f64::from(x) + offset, scale * f64::from(y) + offset);
                    transformed_point(source_to_target, centre).is_some_and(|target| {
                        let column = ((target.x - offset) / scale).round();
                        let row = ((target.y - offset) / scale).round();
                        column >= 0.0
                            && row >= 0.0
                            && column < f64::from(target_width)
                            && row < f64::from(target_height)
                            && target_mask.is_covered(column as u32, row as u32)
                    })
                })
                .count() as u64
        })
        .sum::<u64>();
    Some(covered as f64 * scale * scale)
}

/// Requirement 6.4's literal support: inlier convex-hull area divided by the
/// overlap area, checked on both station planes. This is the source-image
/// measure (the frames' geometric intersection); Virtual_Tile evidence uses
/// [`station_relation_spatial_support_with`] over the doubly covered overlap.
pub(crate) fn station_relation_spatial_support(
    points: &[(Point2<f64>, Point2<f64>)],
    source_to_target: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) -> f64 {
    station_relation_spatial_support_with(
        points,
        source_to_target,
        source_dimensions,
        target_dimensions,
        StationOverlapMeasure::Rectangles,
    )
}

/// [`station_relation_spatial_support`] over the overlap `measure` defines.
pub(crate) fn station_relation_spatial_support_with(
    points: &[(Point2<f64>, Point2<f64>)],
    source_to_target: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    measure: StationOverlapMeasure<'_>,
) -> f64 {
    let overlaps = match measure {
        StationOverlapMeasure::Rectangles => station_relation_overlap_area(
            source_to_target,
            source_dimensions,
            target_dimensions,
            true,
        )
        .zip(station_relation_overlap_area(
            source_to_target,
            source_dimensions,
            target_dimensions,
            false,
        )),
        StationOverlapMeasure::Covered {
            source,
            target,
            factor,
        } => source_to_target.try_inverse().and_then(|target_to_source| {
            virtual_tile_covered_overlap_area(
                source,
                source_dimensions,
                target,
                target_dimensions,
                factor,
                source_to_target,
            )
            .zip(virtual_tile_covered_overlap_area(
                target,
                target_dimensions,
                source,
                source_dimensions,
                factor,
                &target_to_source,
            ))
        }),
    };
    let Some((source_overlap, target_overlap)) =
        overlaps.filter(|&(source, target)| source > f64::EPSILON && target > f64::EPSILON)
    else {
        return 0.0;
    };
    let source_hull = convex_hull_area(points.iter().map(|point| point.0));
    let target_hull = convex_hull_area(points.iter().map(|point| point.1));
    (source_hull / source_overlap)
        .min(target_hull / target_overlap)
        .clamp(0.0, 1.0)
}

#[derive(Clone, Copy)]
pub(crate) struct MeasuredStationRelationCandidate {
    pub(crate) measurements: StationRelationMeasurements,
    pub(crate) quality: f64,
}

/// Build the complete canonical Requirement 6 Station_Relation measurement.
/// Both provisional Virtual_Tile clustering and the final solver consume this
/// typed result and apply [`station_relation_rejection_reasons`] to it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_station_relation_candidate(
    inliers: usize,
    source_to_target: &Matrix3<f64>,
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    overlap_measure: Option<StationOverlapMeasure<'_>>,
    dense_measurements: Option<FocusOverlapQuality>,
) -> Option<MeasuredStationRelationCandidate> {
    let median_error_px = median_symmetric_error(source_to_target, points);
    let precision = if median_error_px.is_finite() {
        1.0 / (1.0 + median_error_px / STATION_RELATION_MAX_MEDIAN_ERROR_PX)
    } else {
        0.0
    };
    let spatial = panorama_spatial_support(points, source_dimensions, target_dimensions);
    let overlap =
        panorama_transform_overlap_support(source_to_target, source_dimensions, target_dimensions);
    let dense_quality = dense_measurements.map(FocusOverlapQuality::legacy_tuple);
    let dense_weight = dense_quality.map_or(1.0, |(intensity, edge, orientation, _)| {
        0.25 + intensity.max(0.0) + 0.5 * edge.max(0.0) + 0.25 * orientation.max(0.0)
    });
    let quality = (inliers as f64).sqrt()
        * precision
        * (0.25 + 0.75 * spatial)
        * (0.25 + 0.75 * overlap)
        * dense_weight;
    if !quality.is_finite() || quality <= 0.0 {
        return None;
    }
    let virtual_tile_measurement =
        matches!(overlap_measure, Some(StationOverlapMeasure::Covered { .. }));
    let photometric = dense_measurements.map(|measurements| {
        if virtual_tile_measurement {
            virtual_tile_photometric_measurements(measurements)
        } else {
            (
                measurements.low_frequency_mean_relative_difference,
                measurements.edge_strength_ratio,
            )
        }
    });
    let edge_strength_ratio = photometric.map_or(f64::INFINITY, |(_, strength)| strength);
    Some(MeasuredStationRelationCandidate {
        measurements: StationRelationMeasurements {
            inliers,
            median_error_px,
            scale_ratio: if virtual_tile_measurement {
                station_relation_overlap_scale(source_to_target, points)
            } else {
                homography_scale_ratio(source_to_target)
            }
            .unwrap_or(f64::INFINITY),
            spatial_support: overlap_measure.map_or(0.0, |measure| {
                station_relation_spatial_support_with(
                    points,
                    source_to_target,
                    source_dimensions,
                    target_dimensions,
                    measure,
                )
            }),
            low_frequency_mean_relative_difference: photometric
                .map_or(f64::INFINITY, |(difference, _)| difference),
            edge_strength_ratio,
            median_edge_orientation_difference_degrees: dense_measurements
                .map_or(f64::INFINITY, |measurements| {
                    measurements.median_edge_orientation_difference_degrees
                }),
            preserves_convex_orientation: homography_preserves_focus_orientation(
                source_to_target,
                source_dimensions,
            ),
        },
        quality,
    })
}

#[derive(Clone, Copy)]
pub(crate) struct FocusGroupEdgeCandidate {
    pub(crate) transform: Matrix3<f64>,
    pub(crate) quality: f64,
    pub(crate) independent_support: usize,
    pub(crate) inlier_count: usize,
    pub(crate) median_error_px: f64,
    pub(crate) scale_ratio: f64,
    pub(crate) spatial_support: f64,
    pub(crate) low_frequency_mean_relative_difference: f64,
    pub(crate) edge_strength_ratio: f64,
    pub(crate) median_edge_orientation_difference_degrees: f64,
    pub(crate) preserves_convex_orientation: bool,
}
