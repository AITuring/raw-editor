//! Closure_Optimizer robust station pose solves (requirement 7).
//!
//! Translation, affine and projective corrections are fitted to accepted
//! Station_Relation observations with the requirement-7.3 M-estimator weights;
//! convergence uses the shared relative residual decrease. Acceptance, bounds
//! and relation rejection live in the closure gate module.

use nalgebra::Matrix3;

use super::{
    CLOSURE_DIRECT_PAIR_P95_LIMIT_PX, CLOSURE_MAX_ITERATIONS, CLOSURE_RELATIVE_RESIDUAL_DECREASE,
    FOCUS_GLOBAL_DAMPING, FOCUS_GLOBAL_MAX_LINEAR_ADJUSTMENT,
    FOCUS_GLOBAL_MAX_PROJECTIVE_ADJUSTMENT, FOCUS_GLOBAL_MAX_TRANSLATION_ADJUSTMENT,
    FOCUS_GLOBAL_PRIOR_WEIGHT, FOCUS_GLOBAL_PROJECTIVE_PRIOR_WEIGHT,
};

/// Exact Closure_Optimizer M-estimator multiplier from requirement 7.3.
///
/// Residuals are measured in world-coordinate native pixels. Keeping this
/// independent of image dimensions makes the same geometric error receive the
/// same weight at every station and in the final report.
pub(super) fn closure_residual_weight(residual_px: f64) -> f64 {
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

pub(super) fn closure_weighted_total_residual(
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

pub(super) fn closure_relative_residual_decrease_converged(previous: f64, current: f64) -> bool {
    if !previous.is_finite() || !current.is_finite() || current > previous {
        return false;
    }
    if previous <= f64::EPSILON {
        return current <= f64::EPSILON;
    }
    (previous - current) / previous < CLOSURE_RELATIVE_RESIDUAL_DECREASE
}

#[derive(Debug)]
pub(super) struct ClosureTranslationSolve {
    pub(super) corrections: Vec<[f64; 2]>,
    pub(super) iterations: usize,
    pub(super) converged: bool,
}

#[derive(Debug)]
pub(super) struct ClosureAffineSolve {
    /// Per-station affine deltas in normalized world coordinates:
    /// `[m00-1, m01, tx, m10, m11-1, ty]`.
    pub(super) corrections: Vec<[f64; 6]>,
    pub(super) iterations: usize,
    pub(super) converged: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ClosureAffineObservation {
    pub(super) left: usize,
    pub(super) right: usize,
    pub(super) left_world: [f64; 2],
    pub(super) right_world: [f64; 2],
    pub(super) weight: f64,
}

#[derive(Debug)]
pub(super) struct ClosureProjectiveSolve {
    /// Per-station homography deltas in normalized world coordinates:
    /// `[m00-1, m01, tx, m10, m11-1, ty, h20, h21]`.
    pub(super) corrections: Vec<[f64; 8]>,
    pub(super) iterations: usize,
    pub(super) converged: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ClosureProjectiveObservation {
    pub(super) left: usize,
    pub(super) right: usize,
    pub(super) left_world: [f64; 2],
    pub(super) right_world: [f64; 2],
    pub(super) weight: f64,
}

pub(super) fn affine_delta_matrix(delta: &[f64; 6], coordinate_scale: f64) -> Matrix3<f64> {
    Matrix3::new(
        1.0 + delta[0],
        delta[1],
        delta[2] * coordinate_scale,
        delta[3],
        1.0 + delta[4],
        delta[5] * coordinate_scale,
        0.0,
        0.0,
        1.0,
    )
}

pub(super) fn affine_delta_apply(delta: &[f64; 6], point: [f64; 2]) -> [f64; 2] {
    [
        (1.0 + delta[0]) * point[0] + delta[1] * point[1] + delta[2],
        delta[3] * point[0] + (1.0 + delta[4]) * point[1] + delta[5],
    ]
}

pub(super) fn projective_delta_matrix(delta: &[f64; 8], coordinate_scale: f64) -> Matrix3<f64> {
    Matrix3::new(
        1.0 + delta[0],
        delta[1],
        delta[2] * coordinate_scale,
        delta[3],
        1.0 + delta[4],
        delta[5] * coordinate_scale,
        delta[6] / coordinate_scale,
        delta[7] / coordinate_scale,
        1.0,
    )
}

pub(super) fn projective_delta_projection(
    delta: &[f64; 8],
    point: [f64; 2],
) -> Option<([f64; 2], [f64; 8], [f64; 8])> {
    let x = point[0];
    let y = point[1];
    let numerator_x = (1.0 + delta[0]) * x + delta[1] * y + delta[2];
    let numerator_y = delta[3] * x + (1.0 + delta[4]) * y + delta[5];
    let denominator = delta[6] * x + delta[7] * y + 1.0;
    if !denominator.is_finite() || denominator.abs() < 1e-8 {
        return None;
    }
    let inverse_denominator = denominator.recip();
    let projected = [
        numerator_x * inverse_denominator,
        numerator_y * inverse_denominator,
    ];
    if projected.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let x_jacobian = [
        x * inverse_denominator,
        y * inverse_denominator,
        inverse_denominator,
        0.0,
        0.0,
        0.0,
        -projected[0] * x * inverse_denominator,
        -projected[0] * y * inverse_denominator,
    ];
    let y_jacobian = [
        0.0,
        0.0,
        0.0,
        x * inverse_denominator,
        y * inverse_denominator,
        inverse_denominator,
        -projected[1] * x * inverse_denominator,
        -projected[1] * y * inverse_denominator,
    ];
    Some((projected, x_jacobian, y_jacobian))
}

/// Keep the projective closure solve sensitive to a relation that is still
/// visibly open under the strict direct-pair gate.  The factor is bounded so a
/// single bad match cannot overwhelm the graph, but it prevents the ordinary
/// least-squares median from treating one failing relation as a harmless tail.
pub(super) fn closure_projective_edge_focus_weight(residual_px: f64) -> f64 {
    if !residual_px.is_finite() {
        return 1.0;
    }
    let gate_ratio = residual_px / CLOSURE_DIRECT_PAIR_P95_LIMIT_PX.max(1.0);
    (1.0 + gate_ratio * gate_ratio).min(16.0)
}

pub(super) fn projective_delta_cost(
    corrections: &[[f64; 8]],
    reference_station: usize,
    observations: &[ClosureProjectiveObservation],
    coordinate_scale: f64,
    use_robust_weights: bool,
) -> f64 {
    let mut cost = 0.0;
    for observation in observations {
        let left_point = [
            observation.left_world[0] / coordinate_scale,
            observation.left_world[1] / coordinate_scale,
        ];
        let right_point = [
            observation.right_world[0] / coordinate_scale,
            observation.right_world[1] / coordinate_scale,
        ];
        let Some((left, _, _)) =
            projective_delta_projection(&corrections[observation.left], left_point)
        else {
            return f64::INFINITY;
        };
        let Some((right, _, _)) =
            projective_delta_projection(&corrections[observation.right], right_point)
        else {
            return f64::INFINITY;
        };
        let residual_px = (left[0] - right[0]).hypot(left[1] - right[1]) * coordinate_scale;
        if !residual_px.is_finite() {
            return f64::INFINITY;
        }
        // Keep the first few projective steps on the same raw least-squares
        // objective used to build their normal equations.  If the initial
        // spanning-tree drift is already beyond the redescending cutoff, a
        // robust cost comparison can reject the useful first step even though
        // it reduces the measured closure error.  The strict robust objective
        // is restored before the final acceptance gates are evaluated.
        let robust = if use_robust_weights {
            closure_residual_weight(residual_px)
        } else {
            1.0
        };
        let edge_focus = closure_projective_edge_focus_weight(residual_px);
        cost += observation.weight * robust * edge_focus * residual_px * residual_px;
    }
    for (station, correction) in corrections.iter().enumerate() {
        if station == reference_station {
            continue;
        }
        for (parameter, value) in correction.iter().enumerate() {
            let prior = if parameter >= 6 {
                FOCUS_GLOBAL_PROJECTIVE_PRIOR_WEIGHT
            } else {
                FOCUS_GLOBAL_PRIOR_WEIGHT
            };
            cost += prior * value * value;
        }
    }
    cost
}

pub(super) fn projective_delta_parameter_limit(parameter: usize) -> f64 {
    match parameter {
        2 | 5 => FOCUS_GLOBAL_MAX_TRANSLATION_ADJUSTMENT,
        6 | 7 => FOCUS_GLOBAL_MAX_PROJECTIVE_ADJUSTMENT,
        _ => FOCUS_GLOBAL_MAX_LINEAR_ADJUSTMENT,
    }
}

pub(super) fn solve_robust_closure_projective(
    station_count: usize,
    reference_station: usize,
    observations: &[ClosureProjectiveObservation],
    coordinate_scale: f64,
) -> Option<ClosureProjectiveSolve> {
    if station_count < 2
        || reference_station >= station_count
        || observations.len() < station_count
        || !coordinate_scale.is_finite()
        || coordinate_scale <= 0.0
    {
        return None;
    }
    let variable_count = station_count.saturating_sub(1) * 8;
    let variable_index = |station: usize, parameter: usize| {
        let compact = if station < reference_station {
            station
        } else {
            station - 1
        };
        compact * 8 + parameter
    };
    let zero_corrections = vec![[0.0f64; 8]; station_count];
    let zero_cost = projective_delta_cost(
        &zero_corrections,
        reference_station,
        observations,
        coordinate_scale,
        false,
    );
    if !zero_cost.is_finite() {
        return None;
    }
    // An affine solve is a strong first-order approximation of the same
    // closure objective.  Starting the projective refinement from it avoids
    // spending the first Gauss-Newton step merely recovering scale/shear that
    // the affine branch already knows, while the projective parameters remain
    // zero and can still absorb denominator drift.
    let affine_observations = observations
        .iter()
        .map(|observation| ClosureAffineObservation {
            left: observation.left,
            right: observation.right,
            left_world: observation.left_world,
            right_world: observation.right_world,
            weight: observation.weight,
        })
        .collect::<Vec<_>>();
    let mut corrections = zero_corrections.clone();
    if let Some(affine_seed) = solve_robust_closure_affine(
        station_count,
        reference_station,
        &affine_observations,
        coordinate_scale,
    ) {
        for station in 0..station_count {
            if station == reference_station {
                continue;
            }
            for parameter in 0..6 {
                corrections[station][parameter] = affine_seed.corrections[station][parameter]
                    .clamp(
                        -projective_delta_parameter_limit(parameter),
                        projective_delta_parameter_limit(parameter),
                    );
            }
        }
        let seeded_cost = projective_delta_cost(
            &corrections,
            reference_station,
            observations,
            coordinate_scale,
            false,
        );
        if !seeded_cost.is_finite() || seeded_cost >= zero_cost {
            corrections = zero_corrections;
        }
    }
    let initial_cost = projective_delta_cost(
        &corrections,
        reference_station,
        observations,
        coordinate_scale,
        false,
    );
    if !initial_cost.is_finite() {
        return None;
    }
    let mut iterations = 0usize;
    let mut converged = false;

    for iteration in 1..=CLOSURE_MAX_ITERATIONS {
        let use_robust_weights = iteration > 3;
        let current_cost = projective_delta_cost(
            &corrections,
            reference_station,
            observations,
            coordinate_scale,
            use_robust_weights,
        );
        if !current_cost.is_finite() {
            break;
        }
        let mut normal = nalgebra::DMatrix::<f64>::zeros(variable_count, variable_count);
        let mut gradient = nalgebra::DVector::<f64>::zeros(variable_count);
        for observation in observations {
            if observation.left >= station_count
                || observation.right >= station_count
                || observation.left == observation.right
                || !observation.weight.is_finite()
                || observation.weight <= 0.0
            {
                continue;
            }
            let left_point = [
                observation.left_world[0] / coordinate_scale,
                observation.left_world[1] / coordinate_scale,
            ];
            let right_point = [
                observation.right_world[0] / coordinate_scale,
                observation.right_world[1] / coordinate_scale,
            ];
            let Some((left, left_x, left_y)) =
                projective_delta_projection(&corrections[observation.left], left_point)
            else {
                continue;
            };
            let Some((right, right_x, right_y)) =
                projective_delta_projection(&corrections[observation.right], right_point)
            else {
                continue;
            };
            let residual = [left[0] - right[0], left[1] - right[1]];
            let magnitude = residual[0].hypot(residual[1]) * coordinate_scale;
            if !magnitude.is_finite() {
                continue;
            }
            // Start the projective solve with the measured edges at their
            // declared support.  Applying the redescending M-estimator to the
            // uncorrected spanning-tree drift can suppress the very long-range
            // closure edge that identifies the accumulated perspective term;
            // after the first few Gauss-Newton steps, restore the unchanged
            // robust weighting for the strict acceptance decision.
            let robust = if use_robust_weights {
                closure_residual_weight(magnitude)
            } else {
                1.0
            };
            let edge_focus = closure_projective_edge_focus_weight(magnitude);
            let weight = observation.weight * robust * edge_focus;
            for (residual_component, left_jacobian, right_jacobian) in [
                (residual[0], left_x, right_x),
                (residual[1], left_y, right_y),
            ]
            .into_iter()
            {
                let mut entries = Vec::with_capacity(16);
                if observation.left != reference_station {
                    let base = variable_index(observation.left, 0);
                    entries.extend(
                        left_jacobian
                            .iter()
                            .enumerate()
                            .map(|(parameter, value)| (base + parameter, *value)),
                    );
                }
                if observation.right != reference_station {
                    let base = variable_index(observation.right, 0);
                    entries.extend(
                        right_jacobian
                            .iter()
                            .enumerate()
                            .map(|(parameter, value)| (base + parameter, -*value)),
                    );
                }
                for &(column, jacobian) in &entries {
                    gradient[column] += weight * jacobian * residual_component;
                    for &(row, other) in &entries {
                        normal[(column, row)] += weight * jacobian * other;
                    }
                }
            }
        }
        for station in 0..station_count {
            if station == reference_station {
                continue;
            }
            let base = variable_index(station, 0);
            for parameter in 0..8 {
                let prior = if parameter >= 6 {
                    FOCUS_GLOBAL_PROJECTIVE_PRIOR_WEIGHT
                } else {
                    FOCUS_GLOBAL_PRIOR_WEIGHT
                };
                normal[(base + parameter, base + parameter)] += prior + FOCUS_GLOBAL_DAMPING;
                gradient[base + parameter] += prior * corrections[station][parameter];
            }
        }
        let delta = normal.lu().solve(&(-gradient))?;
        if delta.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let delta_norm = delta.norm();
        if delta_norm < 1e-9 {
            converged = true;
            iterations = iteration;
            break;
        }
        let mut accepted = None;
        for step in [1.0, 0.5, 0.25, 0.125, 0.0625] {
            let mut candidate = corrections.clone();
            for station in 0..station_count {
                if station == reference_station {
                    continue;
                }
                let base = variable_index(station, 0);
                for parameter in 0..8 {
                    // A Gauss-Newton step can point beyond the bounded
                    // projective correction envelope even when its projected
                    // point is a useful descent direction.  Rejecting the
                    // whole station in that case makes the solve stop after
                    // its first step.  Project each parameter independently;
                    // the unchanged corner-displacement clamp remains the
                    // physical acceptance guard after the solve.
                    let limit = projective_delta_parameter_limit(parameter);
                    let value = candidate[station][parameter] + step * delta[base + parameter];
                    if !value.is_finite() {
                        candidate.clear();
                        break;
                    }
                    candidate[station][parameter] = value.clamp(-limit, limit);
                }
                if candidate.is_empty() {
                    break;
                }
            }
            if candidate.len() != station_count {
                continue;
            }
            let candidate_cost = projective_delta_cost(
                &candidate,
                reference_station,
                observations,
                coordinate_scale,
                use_robust_weights,
            );
            if candidate_cost.is_finite() && candidate_cost + 1e-12 < current_cost {
                accepted = Some((candidate, candidate_cost));
                break;
            }
        }
        let Some((candidate, candidate_cost)) = accepted else {
            break;
        };
        let relative_decrease = if current_cost > f64::EPSILON {
            (current_cost - candidate_cost) / current_cost
        } else {
            0.0
        };
        corrections = candidate;
        iterations = iteration;
        if relative_decrease < CLOSURE_RELATIVE_RESIDUAL_DECREASE {
            converged = true;
            break;
        }
    }
    Some(ClosureProjectiveSolve {
        corrections,
        iterations,
        converged,
    })
}

pub(super) fn solve_robust_closure_affine(
    station_count: usize,
    reference_station: usize,
    observations: &[ClosureAffineObservation],
    coordinate_scale: f64,
) -> Option<ClosureAffineSolve> {
    if station_count < 2
        || reference_station >= station_count
        || observations.len() < station_count
        || !coordinate_scale.is_finite()
        || coordinate_scale <= 0.0
    {
        return None;
    }
    let variable_count = station_count.saturating_sub(1) * 6;
    let variable_index = |station: usize, parameter: usize| {
        let compact = if station < reference_station {
            station
        } else {
            station - 1
        };
        compact * 6 + parameter
    };
    let mut corrections = vec![[0.0f64; 6]; station_count];
    let mut previous_total = f64::INFINITY;
    let mut iterations = 0usize;
    let mut converged = false;

    for iteration in 1..=CLOSURE_MAX_ITERATIONS {
        let mut normal = nalgebra::DMatrix::<f64>::zeros(variable_count, variable_count);
        let mut rhs = nalgebra::DVector::<f64>::zeros(variable_count);
        let mut total = 0.0;
        for observation in observations {
            if observation.left >= station_count
                || observation.right >= station_count
                || observation.left == observation.right
                || !observation.weight.is_finite()
                || observation.weight <= 0.0
            {
                continue;
            }
            let left_point = [
                observation.left_world[0] / coordinate_scale,
                observation.left_world[1] / coordinate_scale,
            ];
            let right_point = [
                observation.right_world[0] / coordinate_scale,
                observation.right_world[1] / coordinate_scale,
            ];
            let left = affine_delta_apply(&corrections[observation.left], left_point);
            let right = affine_delta_apply(&corrections[observation.right], right_point);
            let residual = [left[0] - right[0], left[1] - right[1]];
            let magnitude = residual[0].hypot(residual[1]) * coordinate_scale;
            if !magnitude.is_finite() {
                continue;
            }
            let robust = closure_residual_weight(magnitude);
            let weight = observation.weight * robust;
            total += observation.weight * robust * magnitude * magnitude;
            for axis in 0..2 {
                let mut entries = Vec::with_capacity(12);
                let left_base = (observation.left != reference_station)
                    .then(|| variable_index(observation.left, axis * 3));
                let right_base = (observation.right != reference_station)
                    .then(|| variable_index(observation.right, axis * 3));
                if let Some(base) = left_base {
                    entries.extend([
                        (base, left_point[0]),
                        (base + 1, left_point[1]),
                        (base + 2, 1.0),
                    ]);
                }
                if let Some(base) = right_base {
                    entries.extend([
                        (base, -right_point[0]),
                        (base + 1, -right_point[1]),
                        (base + 2, -1.0),
                    ]);
                }
                let residual_component = residual[axis];
                for &(column, jacobian) in &entries {
                    rhs[column] -= weight * jacobian * residual_component;
                    for &(row, other) in &entries {
                        normal[(column, row)] += weight * jacobian * other;
                    }
                }
            }
        }
        if !total.is_finite() {
            return None;
        }
        for variable in 0..variable_count {
            normal[(variable, variable)] += FOCUS_GLOBAL_DAMPING;
        }
        let delta = normal.lu().solve(&rhs)?;
        if delta.iter().any(|value| !value.is_finite()) {
            return None;
        }
        for station in 0..station_count {
            if station == reference_station {
                continue;
            }
            let base = variable_index(station, 0);
            for parameter in 0..6 {
                corrections[station][parameter] += delta[base + parameter];
            }
        }
        iterations = iteration;
        if previous_total.is_finite()
            && closure_relative_residual_decrease_converged(previous_total, total)
        {
            converged = true;
            break;
        }
        previous_total = total;
    }
    Some(ClosureAffineSolve {
        corrections,
        iterations,
        converged,
    })
}

pub(super) fn solve_robust_closure_translations(
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
