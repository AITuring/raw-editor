//! Closure_Optimizer driver: refine spanning-tree station poses over all relations.
//!
//! Starting from the maximum-inlier spanning-tree seed, the driver alternates the
//! robust closure solves with the closure gate until the residual converges or
//! the iteration budget ends, and records the outcome in the ClosureReport.
//! Protected tree edges are never rejected, so the accepted graph stays connected.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stack_pipeline::degradation;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stack_pipeline::topology;
use nalgebra::Matrix3;
use std::collections::BTreeSet;

use super::{
    CLOSURE_DIRECT_PAIR_P95_LIMIT_PX, CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX, ClosureAcceptance,
    ClosureAffineObservation, ClosureProjectiveObservation, Dsu, FocusCaptureGroup,
    STATION_RELATION_MAX_MEDIAN_ERROR_PX, affine_delta_apply, affine_delta_matrix,
    clamp_station_affine_corrections, clamp_station_projective_corrections,
    clamp_station_translation_corrections, closure_acceptance, closure_direct_pair_p95s,
    closure_offending_pair_record, closure_relation_direct_sample_points,
    closure_relation_sample_points, closure_residual_summary, closure_residual_weight,
    homography_preserves_focus_orientation, median_value, projective_delta_matrix,
    projective_delta_projection, solve_robust_closure_affine, solve_robust_closure_projective,
    solve_robust_closure_translations, transformed_point,
};

/// Refine the maximum-support group spanning-tree poses with all available
/// group relations.  The spanning tree is still used as a stable projective
/// initialisation (it preserves focal scale/roll), but every additional edge
/// contributes a robust translation closure residual.  This prevents one
/// tree edge from carrying the accumulated drift of a long snake scan while
/// retaining the projective geometry measured by the image-level matcher.
///
/// The measured homography remains the base pose. Closure first tries the
/// bounded projective correction needed for denominator drift, then the
/// bounded affine and translation fallbacks. Each correction is M-estimator
/// weighted and must pass the existing residual, corner-displacement, and
/// direct-pair gates, so repeated-content edges are downweighted without
/// changing the local focus ownership.
#[cfg(test)]
pub(super) fn refine_focus_group_projective_tree_poses(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    base_poses: &[Matrix3<f64>],
    reference_group: usize,
    coordinate_scale: f64,
) -> Option<Vec<Matrix3<f64>>> {
    let mut closure_report = stack_report::ClosureReport::default();
    let station_topology = topology::StationTopology {
        row: vec![0; groups.len()],
        column: (0..groups.len() as u32).collect(),
        ..topology::StationTopology::default()
    };
    refine_focus_group_projective_tree_poses_with_report(
        images,
        groups,
        group_edges,
        base_poses,
        reference_group,
        coordinate_scale,
        &station_topology,
        2,
        &mut closure_report,
    )
}

pub(super) fn refine_focus_group_projective_tree_poses_with_report(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    base_poses: &[Matrix3<f64>],
    reference_group: usize,
    coordinate_scale: f64,
    station_topology: &topology::StationTopology,
    minimum_independent_support: usize,
    closure_report: &mut stack_report::ClosureReport,
) -> Option<Vec<Matrix3<f64>>> {
    refine_focus_group_projective_tree_poses_with_report_protected(
        images,
        groups,
        group_edges,
        base_poses,
        reference_group,
        coordinate_scale,
        station_topology,
        minimum_independent_support,
        &BTreeSet::new(),
        closure_report,
    )
}

pub(super) fn refine_focus_group_projective_tree_poses_with_report_protected(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    base_poses: &[Matrix3<f64>],
    reference_group: usize,
    coordinate_scale: f64,
    station_topology: &topology::StationTopology,
    minimum_independent_support: usize,
    protected_tree_edges: &BTreeSet<(usize, usize)>,
    closure_report: &mut stack_report::ClosureReport,
) -> Option<Vec<Matrix3<f64>>> {
    closure_report.participating_constraints = group_edges.len();
    closure_report.constraint_weights.clear();
    closure_report.status = stack_report::ClosureStatus::NotRun;
    closure_report.iterations = 0;
    closure_report.residual_median_px = 0.0;
    closure_report.residual_p95_px = 0.0;
    closure_report.max_direct_pair_p95_px = 0.0;
    closure_report.max_corner_correction_px = 0.0;
    closure_report.clamped_stations = 0;
    closure_report.fallback_reason = None;
    closure_report.offending_pair = None;
    if groups.len() < 2 || base_poses.len() != groups.len() || reference_group >= groups.len() {
        return None;
    }

    // Requirement 7.10: a connected graph with no edge beyond its spanning
    // tree has no cycle constraint. Do not enter either joint solver; returning
    // this clone preserves every f64 bit of the tree seed.
    if group_edges.len() <= groups.len().saturating_sub(1) {
        closure_report.status = stack_report::ClosureStatus::NoConstraints;
        closure_report.fallback_reason = Some(degradation::CLOSURE_NO_CONSTRAINTS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_NO_CONSTRAINTS,
            serde_json::json!({
                "accepted_relations": group_edges.len(),
                "station_count": groups.len(),
            }),
        );
        return Some(base_poses.to_vec());
    }

    let mut constraints = Vec::<(usize, usize, [f64; 2], f64)>::new();
    for &(score, left_group, right_group, left_to_right, support, median_error) in group_edges {
        if support < minimum_independent_support
            || left_group >= groups.len()
            || right_group >= groups.len()
            || left_group == right_group
        {
            continue;
        }
        let left_image = &images[groups[left_group].anchor];
        let left_dimensions = left_image.dimensions();
        let right_dimensions = images[groups[right_group].anchor].dimensions();
        let samples =
            closure_relation_sample_points(&left_to_right, left_dimensions, right_dimensions);
        let mut dx = Vec::new();
        let mut dy = Vec::new();
        // A relation is measured only on its overlap. Extrapolating its
        // projective fit to the rest of the source plane invents closure
        // observations and can pull even a well-aligned overlap apart.
        for point in samples {
            let Some(left_world) = transformed_point(&base_poses[left_group], point) else {
                continue;
            };
            let Some(right_local) = transformed_point(&left_to_right, point) else {
                continue;
            };
            if right_local.x < 0.0
                || right_local.y < 0.0
                || right_local.x >= f64::from(right_dimensions.0)
                || right_local.y >= f64::from(right_dimensions.1)
            {
                continue;
            }
            let Some(right_world) = transformed_point(&base_poses[right_group], right_local) else {
                continue;
            };
            let delta = left_world - right_world;
            if delta.x.is_finite() && delta.y.is_finite() {
                dx.push(delta.x);
                dy.push(delta.y);
            }
        }
        let (Some(dx), Some(dy)) = (median_value(&mut dx), median_value(&mut dy)) else {
            continue;
        };
        let precision = 1.0 / (1.0 + median_error / STATION_RELATION_MAX_MEDIAN_ERROR_PX);
        let protected = protected_tree_edges
            .contains(&(left_group.min(right_group), left_group.max(right_group)));
        let protection_weight = if protected { 16.0 } else { 1.0 };
        let weight = support as f64 * score.max(1e-8).sqrt() * precision * protection_weight;
        if weight.is_finite() && weight > 0.0 {
            constraints.push((left_group, right_group, [dx, dy], weight));
        }
    }

    let constraints_are_connected = if constraints.len() + 1 >= groups.len() {
        let mut connectivity = Dsu::new(groups.len());
        for &(left, right, _, _) in &constraints {
            connectivity.union(left, right);
        }
        let reference_root = connectivity.find(reference_group);
        (0..groups.len()).all(|group| connectivity.find(group) == reference_root)
    } else {
        false
    };
    if !constraints_are_connected {
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason =
            Some(degradation::CLOSURE_UNRELIABLE_ITERATIONS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            serde_json::json!({
                "iterations": 0,
                "converged": false,
                "accepted_relations": group_edges.len(),
                "usable_constraints": constraints.len(),
            }),
        );
        return Some(base_poses.to_vec());
    }

    // Translation closure is insufficient when a valid local affine relation
    // carries a small scale/shear difference across a long station chain.  Use
    // every measured overlap sample to solve one bounded affine correction per
    // station.  The measured projective edge remains authoritative; this stage
    // only reconciles its world-space placement globally and is accepted by
    // the unchanged residual and direct-pair gates below.
    // Do not let the affine fallback turn a pure-translation contradiction
    // into an accepted solve.  It is only applicable when the current tree
    // already violates the unchanged direct-pair gate *and* the measured
    // geometry contains a real linear (scale/shear/projective) signal.  This
    // preserves the strict tree fallback for ordinary translation cycles.
    let base_direct_pairs = closure_direct_pair_p95s(images, groups, group_edges, base_poses);
    let base_direct_pair_failed = base_direct_pairs
        .iter()
        .any(|pair| pair.p95_px > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX);
    let reference_pose = base_poses[reference_group];
    let affine_signal = base_poses.iter().any(|pose| {
        (0..2).any(|row| {
            (0..2).any(|column| (pose[(row, column)] - reference_pose[(row, column)]).abs() > 1e-4)
        }) || pose[(2, 0)].abs() > 1e-8
            || pose[(2, 1)].abs() > 1e-8
    }) || group_edges.iter().any(|&(_, _, _, transform, _, _)| {
        ((transform[(0, 0)] - 1.0).abs()
            + transform[(0, 1)].abs()
            + transform[(1, 0)].abs()
            + (transform[(1, 1)] - 1.0).abs()
            + transform[(2, 0)].abs()
            + transform[(2, 1)].abs())
            > 1e-4
    });
    let affine_needed = base_direct_pair_failed && affine_signal;
    let mut affine_observations = Vec::new();
    let mut affine_coordinate_scale = coordinate_scale.max(1.0);
    for &(_score, left_group, right_group, left_to_right, support, median_error) in group_edges {
        if support < minimum_independent_support
            || left_group >= groups.len()
            || right_group >= groups.len()
            || left_group == right_group
        {
            continue;
        }
        let left_dimensions = images[groups[left_group].anchor].dimensions();
        let right_dimensions = images[groups[right_group].anchor].dimensions();
        let mut samples =
            closure_relation_sample_points(&left_to_right, left_dimensions, right_dimensions);
        samples.extend(closure_relation_direct_sample_points(
            &left_to_right,
            left_dimensions,
            right_dimensions,
        ));
        samples.sort_by(|left, right| {
            left.y
                .total_cmp(&right.y)
                .then_with(|| left.x.total_cmp(&right.x))
        });
        samples.dedup_by(|left, right| {
            (left.x - right.x).abs() <= 1.0e-9 && (left.y - right.y).abs() <= 1.0e-9
        });
        let precision = 1.0 / (1.0 + median_error / STATION_RELATION_MAX_MEDIAN_ERROR_PX);
        // Every accepted relation is subject to the same strict direct-pair
        // P95 gate.  Keep independent support and measured precision in the
        // joint objective, but do not let the inlier-count quality score make
        // a dense edge dominate the closure solve while a lower-support
        // accepted edge remains geometrically open.  The local acceptance
        // gates have already rejected relations without sufficient evidence.
        let protected = protected_tree_edges
            .contains(&(left_group.min(right_group), left_group.max(right_group)));
        let protection_weight = if protected { 16.0 } else { 1.0 };
        let weight = support as f64 * precision * protection_weight;
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        for point in samples {
            let Some(left_world) = transformed_point(&base_poses[left_group], point) else {
                continue;
            };
            let Some(right_local) = transformed_point(&left_to_right, point) else {
                continue;
            };
            let Some(right_world) = transformed_point(&base_poses[right_group], right_local) else {
                continue;
            };
            if !left_world.x.is_finite()
                || !left_world.y.is_finite()
                || !right_world.x.is_finite()
                || !right_world.y.is_finite()
            {
                continue;
            }
            affine_coordinate_scale = affine_coordinate_scale
                .max(left_world.x.abs())
                .max(left_world.y.abs())
                .max(right_world.x.abs())
                .max(right_world.y.abs());
            affine_observations.push(ClosureAffineObservation {
                left: left_group,
                right: right_group,
                left_world: [left_world.x, left_world.y],
                right_world: [right_world.x, right_world.y],
                weight,
            });
        }
    }
    // Native Station_Relation edges can be projective. A projective edge in a
    // spanning tree accumulates denominator drift over a long path that a
    // six-parameter affine correction cannot represent. Solve the same
    // bounded eight-parameter homography delta used by the global pose
    // optimizer before trying the affine fallback. All existing residual,
    // corner-displacement, and direct-pair gates remain unchanged.
    let projective_observations = affine_observations
        .iter()
        .map(|observation| ClosureProjectiveObservation {
            left: observation.left,
            right: observation.right,
            left_world: observation.left_world,
            right_world: observation.right_world,
            weight: observation.weight,
        })
        .collect::<Vec<_>>();
    if affine_needed && projective_observations.len() >= groups.len() {
        if let Some(projective) = solve_robust_closure_projective(
            groups.len(),
            reference_group,
            &projective_observations,
            affine_coordinate_scale,
        ) {
            let mut corrections = projective.corrections;
            let projective_clamp = clamp_station_projective_corrections(
                images,
                groups,
                group_edges,
                base_poses,
                &mut corrections,
                affine_coordinate_scale,
            );
            if let Some((maximum_correction, clamped_stations)) = projective_clamp {
                let mut projective_poses = Vec::with_capacity(groups.len());
                for (station, pose) in base_poses.iter().enumerate() {
                    let corrected =
                        projective_delta_matrix(&corrections[station], affine_coordinate_scale)
                            * pose;
                    if corrected.try_inverse().is_none()
                        || !homography_preserves_focus_orientation(
                            &corrected,
                            images[groups[station].anchor].dimensions(),
                        )
                    {
                        projective_poses.clear();
                        break;
                    }
                    projective_poses.push(corrected);
                }
                if projective_poses.len() == groups.len() {
                    let mut projective_residuals = projective_observations
                        .iter()
                        .filter_map(|observation| {
                            let left_point = [
                                observation.left_world[0] / affine_coordinate_scale,
                                observation.left_world[1] / affine_coordinate_scale,
                            ];
                            let right_point = [
                                observation.right_world[0] / affine_coordinate_scale,
                                observation.right_world[1] / affine_coordinate_scale,
                            ];
                            let left = projective_delta_projection(
                                &corrections[observation.left],
                                left_point,
                            )?;
                            let right = projective_delta_projection(
                                &corrections[observation.right],
                                right_point,
                            )?;
                            Some(
                                (left.0[0] - right.0[0]).hypot(left.0[1] - right.0[1])
                                    * affine_coordinate_scale,
                            )
                        })
                        .collect::<Vec<_>>();
                    if let Some((projective_median, projective_p95)) =
                        closure_residual_summary(&mut projective_residuals)
                    {
                        let direct_pairs = closure_direct_pair_p95s(
                            images,
                            groups,
                            group_edges,
                            &projective_poses,
                        );
                        if let Some(pair) = direct_pairs
                            .iter()
                            .copied()
                            .filter(|pair| {
                                !pair.p95_px.is_finite()
                                    || pair.p95_px > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX
                            })
                            .max_by(|left, right| left.p95_px.total_cmp(&right.p95_px))
                        {
                            closure_report.max_direct_pair_p95_px = pair.p95_px;
                            closure_report.offending_pair =
                                Some(closure_offending_pair_record(pair, station_topology));
                        }
                        if closure_acceptance(
                            groups.len(),
                            group_edges.len(),
                            projective.converged,
                            projective.iterations,
                            projective_median,
                            &direct_pairs,
                        ) == ClosureAcceptance::Accepted
                        {
                            closure_report.iterations = projective.iterations;
                            closure_report.residual_median_px = projective_median;
                            closure_report.residual_p95_px = projective_p95;
                            closure_report.max_direct_pair_p95_px = direct_pairs
                                .iter()
                                .map(|pair| pair.p95_px)
                                .max_by(f64::total_cmp)
                                .unwrap_or(0.0);
                            closure_report.status = stack_report::ClosureStatus::Converged;
                            closure_report.max_corner_correction_px = maximum_correction;
                            closure_report.clamped_stations = clamped_stations;
                            println!(
                                "  - Focus capture-group projective closure accepted: {} samples, {} iteration(s), median/P95 residual {:.2}/{:.2}px, maximum direct-pair P95 {:.2}px",
                                projective_observations.len(),
                                projective.iterations,
                                projective_median,
                                projective_p95,
                                closure_report.max_direct_pair_p95_px,
                            );
                            return Some(projective_poses);
                        }
                    }
                }
            }
        }
    }
    if affine_needed && affine_observations.len() >= groups.len() {
        if let Some(affine) = solve_robust_closure_affine(
            groups.len(),
            reference_group,
            &affine_observations,
            affine_coordinate_scale,
        ) {
            let mut corrections = affine.corrections;
            let affine_clamp = clamp_station_affine_corrections(
                images,
                groups,
                group_edges,
                base_poses,
                &mut corrections,
                affine_coordinate_scale,
            );
            if let Some((maximum_correction, clamped_stations)) = affine_clamp {
                let mut affine_poses = Vec::with_capacity(groups.len());
                for (station, pose) in base_poses.iter().enumerate() {
                    let corrected =
                        affine_delta_matrix(&corrections[station], affine_coordinate_scale) * pose;
                    if corrected.try_inverse().is_none()
                        || !homography_preserves_focus_orientation(
                            &corrected,
                            images[groups[station].anchor].dimensions(),
                        )
                    {
                        affine_poses.clear();
                        break;
                    }
                    affine_poses.push(corrected);
                }
                if affine_poses.len() == groups.len() {
                    let mut affine_residuals = affine_observations
                        .iter()
                        .map(|observation| {
                            let left = affine_delta_apply(
                                &corrections[observation.left],
                                [
                                    observation.left_world[0] / affine_coordinate_scale,
                                    observation.left_world[1] / affine_coordinate_scale,
                                ],
                            );
                            let right = affine_delta_apply(
                                &corrections[observation.right],
                                [
                                    observation.right_world[0] / affine_coordinate_scale,
                                    observation.right_world[1] / affine_coordinate_scale,
                                ],
                            );
                            (left[0] - right[0]).hypot(left[1] - right[1]) * affine_coordinate_scale
                        })
                        .collect::<Vec<_>>();
                    if let Some((affine_median, affine_p95)) =
                        closure_residual_summary(&mut affine_residuals)
                    {
                        let direct_pairs =
                            closure_direct_pair_p95s(images, groups, group_edges, &affine_poses);
                        if let Some(pair) = direct_pairs
                            .iter()
                            .copied()
                            .filter(|pair| {
                                !pair.p95_px.is_finite()
                                    || pair.p95_px > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX
                            })
                            .max_by(|left, right| left.p95_px.total_cmp(&right.p95_px))
                        {
                            closure_report.max_direct_pair_p95_px = pair.p95_px;
                            closure_report.offending_pair =
                                Some(closure_offending_pair_record(pair, station_topology));
                        }
                        if closure_acceptance(
                            groups.len(),
                            group_edges.len(),
                            affine.converged,
                            affine.iterations,
                            affine_median,
                            &direct_pairs,
                        ) == ClosureAcceptance::Accepted
                        {
                            closure_report.iterations = affine.iterations;
                            closure_report.residual_median_px = affine_median;
                            closure_report.residual_p95_px = affine_p95;
                            closure_report.max_direct_pair_p95_px = direct_pairs
                                .iter()
                                .map(|pair| pair.p95_px)
                                .max_by(f64::total_cmp)
                                .unwrap_or(0.0);
                            closure_report.status = stack_report::ClosureStatus::Converged;
                            closure_report.max_corner_correction_px = maximum_correction;
                            closure_report.clamped_stations = clamped_stations;
                            println!(
                                "  - Focus capture-group affine closure accepted: {} samples, {} iteration(s), median/P95 residual {:.2}/{:.2}px, maximum direct-pair P95 {:.2}px",
                                affine_observations.len(),
                                affine.iterations,
                                affine_median,
                                affine_p95,
                                closure_report.max_direct_pair_p95_px,
                            );
                            return Some(affine_poses);
                        }
                    }
                }
            }
        }
    }

    let Some(mut solve) =
        solve_robust_closure_translations(groups.len(), reference_group, &constraints)
    else {
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason =
            Some(degradation::CLOSURE_UNRELIABLE_ITERATIONS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            serde_json::json!({ "iterations": 0, "converged": false }),
        );
        return Some(base_poses.to_vec());
    };
    let iterations = solve.iterations;
    closure_report.iterations = iterations;
    let Some((maximum_correction, clamped_stations)) = clamp_station_translation_corrections(
        images,
        groups,
        group_edges,
        base_poses,
        &mut solve.corrections,
    ) else {
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason =
            Some(degradation::CLOSURE_UNRELIABLE_ITERATIONS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            serde_json::json!({
                "iterations": solve.iterations,
                "converged": false,
                "invalid_correction": true,
            }),
        );
        return Some(base_poses.to_vec());
    };

    closure_report.constraint_weights = constraints
        .iter()
        .map(|&(left, right, observed, _)| {
            let residual = (solve.corrections[right][0] - solve.corrections[left][0] - observed[0])
                .hypot(solve.corrections[right][1] - solve.corrections[left][1] - observed[1]);
            stack_report::ClosureConstraintWeightRecord {
                left,
                right,
                residual_px: residual,
                weight: closure_residual_weight(residual),
            }
        })
        .collect();
    let mut residuals = closure_report
        .constraint_weights
        .iter()
        .map(|relation| relation.residual_px)
        .collect::<Vec<_>>();
    let (median_residual, residual_p95) =
        closure_residual_summary(&mut residuals).unwrap_or((f64::INFINITY, f64::INFINITY));
    closure_report.residual_median_px = median_residual;
    closure_report.residual_p95_px = residual_p95;

    let preliminary = closure_acceptance(
        groups.len(),
        group_edges.len(),
        solve.converged,
        solve.iterations,
        median_residual,
        &[],
    );
    if preliminary != ClosureAcceptance::Accepted {
        let reason = match preliminary {
            ClosureAcceptance::UnreliableResidual => degradation::CLOSURE_UNRELIABLE_RESIDUAL,
            ClosureAcceptance::UnreliableIterations => degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            ClosureAcceptance::NoConstraints => degradation::CLOSURE_NO_CONSTRAINTS,
            ClosureAcceptance::Accepted | ClosureAcceptance::UnreliablePairP95(_) => unreachable!(),
        };
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason = Some(reason.to_string());
        degradation::record_run_degradation(
            reason,
            serde_json::json!({
                "iterations": solve.iterations,
                "converged": solve.converged,
                "residual_median_px": median_residual,
                "residual_limit_px": CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX,
            }),
        );
        return Some(base_poses.to_vec());
    }

    let mut refined = Vec::with_capacity(groups.len());
    for (group_index, pose) in base_poses.iter().enumerate() {
        let correction = solve.corrections[group_index];
        let translation = Matrix3::new(
            1.0,
            0.0,
            correction[0],
            0.0,
            1.0,
            correction[1],
            0.0,
            0.0,
            1.0,
        );
        let corrected = translation * pose;
        if corrected.try_inverse().is_none()
            || !homography_preserves_focus_orientation(
                &corrected,
                images[groups[group_index].anchor].dimensions(),
            )
        {
            closure_report.status = stack_report::ClosureStatus::Unreliable;
            closure_report.fallback_reason =
                Some(degradation::CLOSURE_UNRELIABLE_PAIR_P95.to_string());
            degradation::record_run_degradation(
                degradation::CLOSURE_UNRELIABLE_PAIR_P95,
                serde_json::json!({
                    "station": group_index,
                    "invalid_corrected_pose": true,
                }),
            );
            return Some(base_poses.to_vec());
        }
        refined.push(corrected);
    }

    let direct_pairs = closure_direct_pair_p95s(images, groups, group_edges, &refined);
    closure_report.max_direct_pair_p95_px = direct_pairs
        .iter()
        .map(|pair| pair.p95_px)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    match closure_acceptance(
        groups.len(),
        group_edges.len(),
        solve.converged,
        solve.iterations,
        median_residual,
        &direct_pairs,
    ) {
        ClosureAcceptance::Accepted => {}
        ClosureAcceptance::UnreliablePairP95(pair) => {
            let offending = closure_offending_pair_record(pair, station_topology);
            closure_report.status = stack_report::ClosureStatus::Unreliable;
            closure_report.fallback_reason =
                Some(degradation::CLOSURE_UNRELIABLE_PAIR_P95.to_string());
            closure_report.offending_pair = Some(offending);
            closure_report.max_corner_correction_px = 0.0;
            closure_report.clamped_stations = 0;
            degradation::record_run_degradation(
                degradation::CLOSURE_UNRELIABLE_PAIR_P95,
                serde_json::json!({
                    "left": offending.left,
                    "left_row": offending.left_row,
                    "left_column": offending.left_column,
                    "right": offending.right,
                    "right_row": offending.right_row,
                    "right_column": offending.right_column,
                    "p95_px": offending.p95_px,
                    "limit_px": CLOSURE_DIRECT_PAIR_P95_LIMIT_PX,
                }),
            );
            return Some(base_poses.to_vec());
        }
        ClosureAcceptance::NoConstraints
        | ClosureAcceptance::UnreliableResidual
        | ClosureAcceptance::UnreliableIterations => unreachable!(),
    }

    closure_report.status = stack_report::ClosureStatus::Converged;
    closure_report.max_corner_correction_px = maximum_correction;
    closure_report.clamped_stations = clamped_stations;
    if clamped_stations > 0 {
        degradation::record_run_degradation(
            degradation::CLOSURE_CORRECTION_CLAMPED,
            serde_json::json!({
                "clamped_stations": clamped_stations,
                "max_corner_correction_px": maximum_correction,
            }),
        );
    }
    println!(
        "  - Focus capture-group closure accepted: {} relations, {} iteration(s), median/P95 residual {:.2}/{:.2}px, maximum direct-pair P95 {:.2}px, maximum corner correction {:.2}px, {} clamped station(s)",
        constraints.len(),
        solve.iterations,
        median_residual,
        residual_p95,
        closure_report.max_direct_pair_p95_px,
        maximum_correction,
        clamped_stations,
    );
    Some(refined)
}
