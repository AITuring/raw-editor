//! Closure evidence sampling, bounded corrections and the closure acceptance gate.
//!
//! Residuals are sampled inside each relation's overlap polygon, corrections are
//! clamped to the requirement-7 corner and overlap limits, and an unreliable
//! closure rejects the offending relation only when the station graph stays
//! connected. Nothing here relaxes a Station_Relation threshold.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stack_pipeline::degradation;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stack_pipeline::topology;
use nalgebra::Matrix3;
use nalgebra::Point2;

use super::{
    CLOSURE_DIRECT_PAIR_P95_LIMIT_PX, CLOSURE_MAX_CORNER_CORRECTION_PX, CLOSURE_MAX_ITERATIONS,
    CLOSURE_OVERLAP_SHORT_SIDE_CORRECTION_RATIO, CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX, Dsu,
    FocusCaptureGroup, affine_delta_matrix, clip_polygon_to_rectangle, projective_delta_matrix,
    transformed_point,
};

pub(super) fn station_relation_overlap_polygon(
    source_to_target: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    source_side: bool,
) -> Option<Vec<Point2<f64>>> {
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
    (!clipped.is_empty()).then_some(clipped)
}

/// Sample the part of a relation that is actually shared by both station
/// tiles.  The closure solver only corrects world translation, so measuring a
/// relation at a point outside the target tile would turn an extrapolated
/// homography into a false translation constraint.  This is especially easy
/// to trigger for long, narrow overlaps where the fixed 20/50/80% lattice
/// contains mostly out-of-bounds points.
pub(super) fn closure_relation_sample_points(
    left_to_right: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
) -> Vec<Point2<f64>> {
    let Some(overlap) =
        station_relation_overlap_polygon(left_to_right, left_dimensions, right_dimensions, true)
    else {
        return Vec::new();
    };
    if overlap.len() < 3 {
        return Vec::new();
    }

    // The clipped polygon is convex.  Its vertices and edge midpoints cover
    // narrow strip overlaps, while the mean of the vertices supplies one
    // interior sample.  Every point is therefore measured inside both tile
    // bounds; no closure constraint is extrapolated beyond the relation's
    // actual shared footprint.
    let mut samples = overlap.clone();
    for index in 0..overlap.len() {
        let next = overlap[(index + 1) % overlap.len()];
        samples.push(Point2::from((overlap[index].coords + next.coords) / 2.0));
    }
    let centroid = overlap
        .iter()
        .copied()
        .fold(nalgebra::Vector2::zeros(), |sum, point| sum + point.coords)
        / overlap.len() as f64;
    samples.push(Point2::from(centroid));
    samples.retain(|point| {
        transformed_point(left_to_right, *point).is_some_and(|target| {
            target.x >= 0.0
                && target.y >= 0.0
                && target.x <= right_dimensions.0 as f64
                && target.y <= right_dimensions.1 as f64
        })
    });
    samples.sort_by(|left, right| {
        left.y
            .total_cmp(&right.y)
            .then_with(|| left.x.total_cmp(&right.x))
    });
    samples.dedup_by(|left, right| {
        (left.x - right.x).abs() <= 1.0e-9 && (left.y - right.y).abs() <= 1.0e-9
    });
    samples
}

/// Return the same clipped 11×11 lattice used by the strict direct-pair gate.
/// The closure optimizer otherwise sees only polygon vertices and a centroid,
/// which can leave a projective correction that looks good on the sparse
/// boundary samples but still fails the gate's interior lattice.
pub(super) fn closure_relation_direct_sample_points(
    left_to_right: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
) -> Vec<Point2<f64>> {
    const SAMPLES_PER_AXIS: usize = 11;
    let mut samples = Vec::with_capacity(SAMPLES_PER_AXIS * SAMPLES_PER_AXIS);
    for sample_y in 0..SAMPLES_PER_AXIS {
        let y = left_dimensions.1.saturating_sub(1) as f64 * sample_y as f64
            / (SAMPLES_PER_AXIS - 1) as f64;
        for sample_x in 0..SAMPLES_PER_AXIS {
            let x = left_dimensions.0.saturating_sub(1) as f64 * sample_x as f64
                / (SAMPLES_PER_AXIS - 1) as f64;
            let point = Point2::new(x, y);
            let Some(target) = transformed_point(left_to_right, point) else {
                continue;
            };
            if target.x >= 0.0
                && target.y >= 0.0
                && target.x < right_dimensions.0 as f64
                && target.y < right_dimensions.1 as f64
            {
                samples.push(point);
            }
        }
    }
    samples
}

pub(super) fn projected_polygon_short_side(
    polygon: &[Point2<f64>],
    local_to_world: &Matrix3<f64>,
) -> Option<f64> {
    let projected = polygon
        .iter()
        .copied()
        .map(|point| transformed_point(local_to_world, point))
        .collect::<Option<Vec<_>>>()?;
    let min_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let max_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let short_side = (max_x - min_x).min(max_y - min_y);
    (short_side.is_finite() && short_side > 0.0).then_some(short_side)
}

pub(super) fn maximum_station_corner_displacement(
    initial_pose: &Matrix3<f64>,
    corrected_pose: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> Option<f64> {
    [
        Point2::new(0.0, 0.0),
        Point2::new(dimensions.0 as f64, 0.0),
        Point2::new(dimensions.0 as f64, dimensions.1 as f64),
        Point2::new(0.0, dimensions.1 as f64),
    ]
    .into_iter()
    .try_fold(0.0f64, |maximum, corner| {
        let initial = transformed_point(initial_pose, corner)?;
        let corrected = transformed_point(corrected_pose, corner)?;
        let displacement = (corrected - initial).norm();
        displacement
            .is_finite()
            .then_some(maximum.max(displacement))
    })
}

pub(super) fn clamp_station_translation_corrections(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    initial_poses: &[Matrix3<f64>],
    corrections: &mut [[f64; 2]],
) -> Option<(f64, usize)> {
    if groups.len() != initial_poses.len() || groups.len() != corrections.len() {
        return None;
    }
    let mut overlap_short_sides = vec![None::<f64>; groups.len()];
    for &(_, left, right, left_to_right, _, _) in group_edges {
        if left >= groups.len() || right >= groups.len() {
            continue;
        }
        let left_dimensions = images[groups[left].anchor].dimensions();
        let right_dimensions = images[groups[right].anchor].dimensions();
        for (station, source_side) in [(left, true), (right, false)] {
            let polygon = station_relation_overlap_polygon(
                &left_to_right,
                left_dimensions,
                right_dimensions,
                source_side,
            )?;
            let short_side = projected_polygon_short_side(&polygon, &initial_poses[station])?;
            overlap_short_sides[station] = Some(
                overlap_short_sides[station].map_or(short_side, |current| current.min(short_side)),
            );
        }
    }

    let mut clamped_stations = 0usize;
    let mut actual_maximum = 0.0f64;
    for station in 0..groups.len() {
        let correction = corrections[station];
        if !correction[0].is_finite() || !correction[1].is_finite() {
            return None;
        }
        let overlap_short_side = overlap_short_sides[station]?;
        let limit = (overlap_short_side * CLOSURE_OVERLAP_SHORT_SIDE_CORRECTION_RATIO)
            .min(CLOSURE_MAX_CORNER_CORRECTION_PX);
        let corrected_pose = Matrix3::new(
            1.0,
            0.0,
            correction[0],
            0.0,
            1.0,
            correction[1],
            0.0,
            0.0,
            1.0,
        ) * initial_poses[station];
        let displacement = maximum_station_corner_displacement(
            &initial_poses[station],
            &corrected_pose,
            images[groups[station].anchor].dimensions(),
        )?;
        if displacement > limit {
            let scale = if displacement > 0.0 {
                limit / displacement
            } else {
                0.0
            };
            corrections[station][0] *= scale;
            corrections[station][1] *= scale;
            clamped_stations += 1;
        }
        let clamped = corrections[station];
        let clamped_pose = Matrix3::new(1.0, 0.0, clamped[0], 0.0, 1.0, clamped[1], 0.0, 0.0, 1.0)
            * initial_poses[station];
        actual_maximum = actual_maximum.max(maximum_station_corner_displacement(
            &initial_poses[station],
            &clamped_pose,
            images[groups[station].anchor].dimensions(),
        )?);
    }
    Some((actual_maximum, clamped_stations))
}

/// Bound a global affine closure correction by the same physical overlap
/// displacement budget as the translation closure.  Scaling all six deltas
/// together preserves the measured shear/scale direction while ensuring that
/// one weak edge cannot move an entire station outside its overlap evidence.
pub(super) fn clamp_station_affine_corrections(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    initial_poses: &[Matrix3<f64>],
    corrections: &mut [[f64; 6]],
    coordinate_scale: f64,
) -> Option<(f64, usize)> {
    if groups.len() != initial_poses.len() || groups.len() != corrections.len() {
        return None;
    }
    let mut overlap_short_sides = vec![None::<f64>; groups.len()];
    for &(_, left, right, left_to_right, _, _) in group_edges {
        if left >= groups.len() || right >= groups.len() {
            continue;
        }
        let left_dimensions = images[groups[left].anchor].dimensions();
        let right_dimensions = images[groups[right].anchor].dimensions();
        for (station, source_side) in [(left, true), (right, false)] {
            let polygon = station_relation_overlap_polygon(
                &left_to_right,
                left_dimensions,
                right_dimensions,
                source_side,
            )?;
            let short_side = projected_polygon_short_side(&polygon, &initial_poses[station])?;
            overlap_short_sides[station] = Some(
                overlap_short_sides[station].map_or(short_side, |current| current.min(short_side)),
            );
        }
    }
    let mut clamped_stations = 0usize;
    let mut actual_maximum = 0.0f64;
    for station in 0..groups.len() {
        if station >= corrections.len() || !corrections[station].iter().all(|v| v.is_finite()) {
            return None;
        }
        let overlap_short_side = overlap_short_sides[station]?;
        let limit = (overlap_short_side * CLOSURE_OVERLAP_SHORT_SIDE_CORRECTION_RATIO)
            .min(CLOSURE_MAX_CORNER_CORRECTION_PX);
        let correction = affine_delta_matrix(&corrections[station], coordinate_scale);
        let corrected_pose = correction * initial_poses[station];
        let displacement = maximum_station_corner_displacement(
            &initial_poses[station],
            &corrected_pose,
            images[groups[station].anchor].dimensions(),
        )?;
        if displacement > limit {
            let scale = if displacement > 0.0 {
                limit / displacement
            } else {
                0.0
            };
            for value in &mut corrections[station] {
                *value *= scale;
            }
            clamped_stations += 1;
        }
        let corrected_pose =
            affine_delta_matrix(&corrections[station], coordinate_scale) * initial_poses[station];
        actual_maximum = actual_maximum.max(maximum_station_corner_displacement(
            &initial_poses[station],
            &corrected_pose,
            images[groups[station].anchor].dimensions(),
        )?);
    }
    Some((actual_maximum, clamped_stations))
}

pub(super) fn clamp_station_projective_corrections(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    initial_poses: &[Matrix3<f64>],
    corrections: &mut [[f64; 8]],
    coordinate_scale: f64,
) -> Option<(f64, usize)> {
    if groups.len() != initial_poses.len() || groups.len() != corrections.len() {
        return None;
    }
    let mut overlap_short_sides = vec![None::<f64>; groups.len()];
    for &(_, left, right, left_to_right, _, _) in group_edges {
        if left >= groups.len() || right >= groups.len() {
            continue;
        }
        let left_dimensions = images[groups[left].anchor].dimensions();
        let right_dimensions = images[groups[right].anchor].dimensions();
        for (station, source_side) in [(left, true), (right, false)] {
            let polygon = station_relation_overlap_polygon(
                &left_to_right,
                left_dimensions,
                right_dimensions,
                source_side,
            )?;
            let short_side = projected_polygon_short_side(&polygon, &initial_poses[station])?;
            overlap_short_sides[station] = Some(
                overlap_short_sides[station].map_or(short_side, |current| current.min(short_side)),
            );
        }
    }
    let mut clamped_stations = 0usize;
    let mut actual_maximum = 0.0f64;
    for station in 0..groups.len() {
        if !corrections[station].iter().all(|value| value.is_finite()) {
            return None;
        }
        let overlap_short_side = overlap_short_sides[station]?;
        let limit = (overlap_short_side * CLOSURE_OVERLAP_SHORT_SIDE_CORRECTION_RATIO)
            .min(CLOSURE_MAX_CORNER_CORRECTION_PX);
        let corrected_pose = projective_delta_matrix(&corrections[station], coordinate_scale)
            * initial_poses[station];
        let displacement = maximum_station_corner_displacement(
            &initial_poses[station],
            &corrected_pose,
            images[groups[station].anchor].dimensions(),
        )?;
        if displacement > limit {
            let scale = if displacement > 0.0 {
                limit / displacement
            } else {
                0.0
            };
            for value in &mut corrections[station] {
                *value *= scale;
            }
            clamped_stations += 1;
        }
        let corrected_pose = projective_delta_matrix(&corrections[station], coordinate_scale)
            * initial_poses[station];
        actual_maximum = actual_maximum.max(maximum_station_corner_displacement(
            &initial_poses[station],
            &corrected_pose,
            images[groups[station].anchor].dimensions(),
        )?);
    }
    Some((actual_maximum, clamped_stations))
}

pub(super) fn closure_residual_summary(residuals: &mut [f64]) -> Option<(f64, f64)> {
    if residuals.is_empty() || residuals.iter().any(|residual| !residual.is_finite()) {
        return None;
    }
    residuals.sort_by(f64::total_cmp);
    let median = residuals[residuals.len() / 2];
    let p95_index = residuals
        .len()
        .saturating_mul(95)
        .div_ceil(100)
        .saturating_sub(1);
    Some((median, residuals[p95_index]))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ClosureDirectPairP95 {
    pub(super) left: usize,
    pub(super) right: usize,
    pub(super) p95_px: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ClosureAcceptance {
    Accepted,
    NoConstraints,
    UnreliableResidual,
    UnreliableIterations,
    UnreliablePairP95(ClosureDirectPairP95),
}

pub(super) fn closure_acceptance(
    station_count: usize,
    accepted_relation_count: usize,
    converged: bool,
    iterations: usize,
    residual_median_px: f64,
    direct_pairs: &[ClosureDirectPairP95],
) -> ClosureAcceptance {
    if accepted_relation_count <= station_count.saturating_sub(1) {
        return ClosureAcceptance::NoConstraints;
    }
    if !converged || iterations > CLOSURE_MAX_ITERATIONS {
        return ClosureAcceptance::UnreliableIterations;
    }
    if !residual_median_px.is_finite() || residual_median_px > CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX {
        return ClosureAcceptance::UnreliableResidual;
    }
    direct_pairs
        .iter()
        .copied()
        .filter(|pair| !pair.p95_px.is_finite() || pair.p95_px > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX)
        .max_by(|left, right| {
            left.p95_px
                .total_cmp(&right.p95_px)
                .then_with(|| (right.left, right.right).cmp(&(left.left, left.right)))
        })
        .map_or(
            ClosureAcceptance::Accepted,
            ClosureAcceptance::UnreliablePairP95,
        )
}

/// Measure overlap reprojection disagreement for every accepted direct
/// Station_Relation. The fixed 11×11 source-plane lattice is clipped by the
/// relation into the target tile, so every retained sample belongs to the
/// geometric overlap and repeated runs inspect identical world positions.
pub(super) fn closure_direct_pair_p95s(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    poses: &[Matrix3<f64>],
) -> Vec<ClosureDirectPairP95> {
    const SAMPLES_PER_AXIS: usize = 11;

    group_edges
        .iter()
        .map(|&(_, left, right, left_to_right, _, _)| {
            let mut errors = Vec::with_capacity(SAMPLES_PER_AXIS * SAMPLES_PER_AXIS);
            if let (Some(left_group), Some(right_group), Some(left_pose), Some(right_pose)) = (
                groups.get(left),
                groups.get(right),
                poses.get(left),
                poses.get(right),
            ) {
                let left_dimensions = images[left_group.anchor].dimensions();
                let right_dimensions = images[right_group.anchor].dimensions();
                for sample_y in 0..SAMPLES_PER_AXIS {
                    let y = (left_dimensions.1.saturating_sub(1) as f64) * sample_y as f64
                        / (SAMPLES_PER_AXIS - 1) as f64;
                    for sample_x in 0..SAMPLES_PER_AXIS {
                        let x = (left_dimensions.0.saturating_sub(1) as f64) * sample_x as f64
                            / (SAMPLES_PER_AXIS - 1) as f64;
                        let source = Point2::new(x, y);
                        let Some(target) = transformed_point(&left_to_right, source) else {
                            continue;
                        };
                        if target.x < 0.0
                            || target.y < 0.0
                            || target.x >= right_dimensions.0 as f64
                            || target.y >= right_dimensions.1 as f64
                        {
                            continue;
                        }
                        let (Some(left_world), Some(right_world)) = (
                            transformed_point(left_pose, source),
                            transformed_point(right_pose, target),
                        ) else {
                            continue;
                        };
                        let error = (left_world - right_world).norm();
                        if error.is_finite() {
                            errors.push(error);
                        }
                    }
                }
            }
            let p95_px = closure_residual_summary(&mut errors)
                .map(|(_, p95)| p95)
                .unwrap_or(f64::INFINITY);
            ClosureDirectPairP95 {
                left,
                right,
                p95_px,
            }
        })
        .collect()
}

pub(super) fn closure_offending_pair_record(
    pair: ClosureDirectPairP95,
    station_topology: &topology::StationTopology,
) -> stack_report::ClosureOffendingPairRecord {
    stack_report::ClosureOffendingPairRecord {
        left: pair.left,
        left_row: station_topology
            .row
            .get(pair.left)
            .copied()
            .unwrap_or(u32::MAX),
        left_column: station_topology
            .column
            .get(pair.left)
            .copied()
            .unwrap_or(u32::MAX),
        right: pair.right,
        right_row: station_topology
            .row
            .get(pair.right)
            .copied()
            .unwrap_or(u32::MAX),
        right_column: station_topology
            .column
            .get(pair.right)
            .copied()
            .unwrap_or(u32::MAX),
        p95_px: pair.p95_px,
    }
}

pub(super) fn station_relation_pair_matches(
    left: usize,
    right: usize,
    pair: (usize, usize),
) -> bool {
    (left.min(right), left.max(right)) == (pair.0.min(pair.1), pair.0.max(pair.1))
}

pub(super) fn relation_edge_removal_preserves_connectivity(
    station_count: usize,
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    remove_index: usize,
) -> bool {
    if station_count == 0 || remove_index >= group_edges.len() {
        return false;
    }
    let mut connectivity = Dsu::new(station_count);
    for (index, &(_, left, right, _, _, _)) in group_edges.iter().enumerate() {
        if index == remove_index || left >= station_count || right >= station_count || left == right
        {
            continue;
        }
        connectivity.union(left, right);
    }
    let reference = connectivity.find(0);
    (0..station_count).all(|station| connectivity.find(station) == reference)
}

/// Remove an accepted relation only after the strict global direct-pair gate
/// has identified it as incompatible with the connected pose graph.  Local
/// Station_Relation evidence remains intact in the candidate diagnostics, but
/// the relation no longer participates in the authoritative graph or closure
/// report.  The graph must stay connected, so this cannot silently discard a
/// tree edge and turn the solve into independent components.
pub(super) fn reject_global_closure_relation(
    station_count: usize,
    group_edges: &mut Vec<(f64, usize, usize, Matrix3<f64>, usize, f64)>,
    station_relations_report: &mut stack_report::StationRelationsReport,
    pair: (usize, usize),
) -> bool {
    let Some(index) = group_edges
        .iter()
        .position(|&(_, left, right, _, _, _)| station_relation_pair_matches(left, right, pair))
    else {
        return false;
    };
    if !relation_edge_removal_preserves_connectivity(station_count, group_edges, index) {
        return false;
    }
    let accepted_before = station_relations_report.accepted.len();
    let accepted_matches = station_relations_report
        .accepted
        .iter()
        .any(|relation| station_relation_pair_matches(relation.left, relation.right, pair));
    if !accepted_matches {
        return false;
    }
    group_edges.remove(index);
    station_relations_report
        .accepted
        .retain(|relation| !station_relation_pair_matches(relation.left, relation.right, pair));
    if station_relations_report.accepted.len() == accepted_before {
        return false;
    }
    station_relations_report.rejected_count =
        station_relations_report.rejected_count.saturating_add(1);
    *station_relations_report
        .rejected_by_reason
        .entry(degradation::CLOSURE_UNRELIABLE_PAIR_P95.to_string())
        .or_default() += 1;
    true
}
