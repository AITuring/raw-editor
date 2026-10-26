//! Station closure translation solving, acceptance, and spanning-tree selection.
//!
//! This module is part of the station-solver facade and preserves the
//! production helper visibility used by the focus-stack pipeline.

use super::*;

/// Exact Closure_Optimizer M-estimator multiplier from requirement 7.3.
///
/// Residuals are measured in world-coordinate native pixels. Keeping this
/// independent of image dimensions makes the same geometric error receive the
/// same weight at every station and in the final report.

pub(crate) fn closure_residual_weight(residual_px: f64) -> f64 {
    if residual_px.is_nan() {
        return 0.0;
    }
    let residual_px = residual_px.max(0.0);
    if residual_px <= 2.0 {
        1.0
    } else if residual_px <= 6.0 {
        1.0 - 0.9 * (residual_px - 2.0) / 4.0
    } else {
        0.1 * (6.0 / residual_px)
    }
}

pub(crate) fn closure_weighted_total_residual(
    corrections: &[[f64; 2]],
    constraints: &[(usize, usize, [f64; 2], f64)],
) -> f64 {
    constraints
        .iter()
        .map(|&(left, right, observed, base_weight)| {
            let residual = (corrections[right][0] - corrections[left][0] - observed[0])
                .hypot(corrections[right][1] - corrections[left][1] - observed[1]);
            base_weight * closure_residual_weight(residual) * residual * residual
        })
        .sum()
}

pub(crate) fn closure_relative_residual_decrease_converged(previous: f64, current: f64) -> bool {
    if !previous.is_finite() || !current.is_finite() || current > previous {
        return false;
    }
    if previous <= f64::EPSILON {
        return current <= f64::EPSILON;
    }
    (previous - current) / previous < CLOSURE_RELATIVE_RESIDUAL_DECREASE
}

#[derive(Debug)]
pub(crate) struct ClosureTranslationSolve {
    pub(crate) corrections: Vec<[f64; 2]>,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
}

pub(crate) fn solve_robust_closure_translations(
    station_count: usize,
    reference_station: usize,
    constraints: &[(usize, usize, [f64; 2], f64)],
) -> Option<ClosureTranslationSolve> {
    if station_count < 2 || reference_station >= station_count || constraints.is_empty() {
        return None;
    }
    let variable_count = station_count.saturating_sub(1) * 2;
    let variable_index = |station: usize, axis: usize| {
        let compact = if station < reference_station {
            station
        } else {
            station - 1
        };
        compact * 2 + axis
    };
    let mut corrections = vec![[0.0f64; 2]; station_count];
    let mut previous_total = closure_weighted_total_residual(&corrections, constraints);
    let mut iterations = 0usize;
    let mut converged = false;

    for iteration in 1..=CLOSURE_MAX_ITERATIONS {
        let mut normal = nalgebra::DMatrix::<f64>::zeros(variable_count, variable_count);
        let mut rhs = nalgebra::DVector::<f64>::zeros(variable_count);
        for &(left, right, observed, base_weight) in constraints {
            let residual = [
                corrections[right][0] - corrections[left][0] - observed[0],
                corrections[right][1] - corrections[left][1] - observed[1],
            ];
            let magnitude = residual[0].hypot(residual[1]);
            let weight = base_weight * closure_residual_weight(magnitude);
            for axis in 0..2 {
                let left_variable = (left != reference_station).then(|| variable_index(left, axis));
                let right_variable =
                    (right != reference_station).then(|| variable_index(right, axis));
                if let Some(variable) = left_variable {
                    normal[(variable, variable)] += weight;
                    rhs[variable] -= weight * observed[axis];
                }
                if let Some(variable) = right_variable {
                    normal[(variable, variable)] += weight;
                    rhs[variable] += weight * observed[axis];
                }
                if let (Some(left_variable), Some(right_variable)) = (left_variable, right_variable)
                {
                    normal[(left_variable, right_variable)] -= weight;
                    normal[(right_variable, left_variable)] -= weight;
                }
            }
        }
        for variable in 0..variable_count {
            normal[(variable, variable)] += FOCUS_GLOBAL_DAMPING;
        }
        let solution = normal.lu().solve(&rhs)?;
        for station in 0..station_count {
            if station == reference_station {
                continue;
            }
            corrections[station] = [
                solution[variable_index(station, 0)],
                solution[variable_index(station, 1)],
            ];
        }
        iterations = iteration;
        let current_total = closure_weighted_total_residual(&corrections, constraints);
        if closure_relative_residual_decrease_converged(previous_total, current_total) {
            converged = true;
            break;
        }
        previous_total = current_total;
    }

    Some(ClosureTranslationSolve {
        corrections,
        iterations,
        converged,
    })
}

pub(crate) fn station_relation_overlap_polygon(
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

pub(crate) fn projected_polygon_short_side(
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

pub(crate) fn maximum_station_corner_displacement(
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

pub(crate) fn clamp_station_translation_corrections(
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

pub(crate) fn closure_residual_summary(residuals: &mut [f64]) -> Option<(f64, f64)> {
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
pub(crate) struct ClosureDirectPairP95 {
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) p95_px: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ClosureAcceptance {
    Accepted,
    NoConstraints,
    UnreliableResidual,
    UnreliableIterations,
    UnreliablePairP95(ClosureDirectPairP95),
}

pub(crate) fn closure_acceptance(
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
pub(crate) fn closure_direct_pair_p95s(
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

pub(crate) fn closure_offending_pair_record(
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

pub(crate) fn station_tree_position_key(
    station_topology: &topology::StationTopology,
    station: usize,
) -> (u32, u32, usize) {
    (
        station_topology
            .row
            .get(station)
            .copied()
            .unwrap_or(u32::MAX),
        station_topology
            .column
            .get(station)
            .copied()
            .unwrap_or(u32::MAX),
        station,
    )
}

pub(crate) fn station_tree_edge_key(
    station_topology: &topology::StationTopology,
    left: usize,
    right: usize,
) -> ((u32, u32, usize), (u32, u32, usize)) {
    let left = station_tree_position_key(station_topology, left);
    let right = station_tree_position_key(station_topology, right);
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

/// Select a maximum-inlier spanning tree. The returned values are indices into
/// `relations`; relation score and input order never participate. Equal inlier
/// counts are ordered only by the stations' canonical row/column indices.
pub(crate) fn maximum_inlier_spanning_tree_indices(
    station_count: usize,
    station_topology: &topology::StationTopology,
    relations: &[(usize, usize, usize)],
) -> Vec<usize> {
    let mut ordered = relations
        .iter()
        .enumerate()
        .filter(|(_, (left, right, _))| {
            *left < station_count && *right < station_count && left != right
        })
        .collect::<Vec<_>>();
    ordered.sort_by(|(_, left), (_, right)| {
        right.2.cmp(&left.2).then_with(|| {
            station_tree_edge_key(station_topology, left.0, left.1).cmp(&station_tree_edge_key(
                station_topology,
                right.0,
                right.1,
            ))
        })
    });

    let mut connectivity = Dsu::new(station_count);
    let mut selected = Vec::with_capacity(station_count.saturating_sub(1));
    for (index, &(left, right, _)) in ordered {
        if connectivity.find(left) == connectivity.find(right) {
            continue;
        }
        connectivity.union(left, right);
        selected.push(index);
        if selected.len() + 1 == station_count {
            break;
        }
    }
    selected
}
