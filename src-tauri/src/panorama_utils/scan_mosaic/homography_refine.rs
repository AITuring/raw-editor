//! Registration refinement of the planar tile poses.
//!
//! Native patch correspondences between overlapping tiles and one joint robust
//! least-squares adjustment of all homographies.

use super::super::registration::PatchMatchGates;
use super::super::registration::probe_warped_patch;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Matrix3;
use nalgebra::Point2;
use rayon::prelude::*;
use std::time::Instant;

use super::{
    NORMALIZED_UNIT, REGISTRATION_HUBER_PX, REGISTRATION_INSET, REGISTRATION_STEP, TileAnalysis,
    project, quarter_valid,
};

// ---------------------------------------------------------------------------
// Registration refinement
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(super) struct Correspondence {
    pub(super) first: usize,
    pub(super) second: usize,
    pub(super) first_point: Point2<f64>,
    pub(super) second_point: Point2<f64>,
}

pub(super) fn inside_frame(analysis: &TileAnalysis, point: Point2<f64>, inset: f64) -> bool {
    point.x >= inset
        && point.y >= inset
        && point.x < analysis.width as f64 - inset
        && point.y < analysis.height as f64 - inset
}

pub(super) fn collect_correspondences(
    analyses: &[TileAnalysis],
    homographies: &[Matrix3<f64>],
    step: f64,
) -> Vec<Correspondence> {
    let inverses = homographies
        .iter()
        .map(|homography| homography.try_inverse())
        .collect::<Vec<_>>();
    let mut jobs = Vec::new();
    for first in 0..analyses.len() {
        for second in first + 1..analyses.len() {
            let (Some(first_inverse), Some(second_inverse)) = (inverses[first], inverses[second])
            else {
                continue;
            };
            let bounds = |index: usize| {
                let analysis = &analyses[index];
                let mut bounds = (
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                );
                for (x, y) in [
                    (0.0, 0.0),
                    (analysis.width as f64, 0.0),
                    (analysis.width as f64, analysis.height as f64),
                    (0.0, analysis.height as f64),
                ] {
                    if let Some(point) = project(&homographies[index], Point2::new(x, y)) {
                        bounds.0 = bounds.0.min(point.x);
                        bounds.1 = bounds.1.max(point.x);
                        bounds.2 = bounds.2.min(point.y);
                        bounds.3 = bounds.3.max(point.y);
                    }
                }
                bounds
            };
            let a = bounds(first);
            let b = bounds(second);
            let x0 = a.0.max(b.0);
            let x1 = a.1.min(b.1);
            let y0 = a.2.max(b.2);
            let y1 = a.3.min(b.3);
            if x1 - x0 < REGISTRATION_INSET * 3.0 || y1 - y0 < REGISTRATION_INSET * 3.0 {
                continue;
            }
            let mut y = y0 + REGISTRATION_INSET;
            while y <= y1 - REGISTRATION_INSET {
                let mut x = x0 + REGISTRATION_INSET;
                while x <= x1 - REGISTRATION_INSET {
                    let world = Point2::new(x, y);
                    if let (Some(a_point), Some(b_point)) = (
                        project(&first_inverse, world),
                        project(&second_inverse, world),
                    ) && inside_frame(&analyses[first], a_point, REGISTRATION_INSET)
                        && inside_frame(&analyses[second], b_point, REGISTRATION_INSET)
                    {
                        jobs.push((first, second, a_point));
                    }
                    x += step;
                }
                y += step;
            }
        }
    }
    let quarter = Matrix3::new(0.25, 0.0, -0.375, 0.0, 0.25, -0.375, 0.0, 0.0, 1.0);
    let quarter_inverse = Matrix3::new(4.0, 0.0, 1.5, 0.0, 4.0, 1.5, 0.0, 0.0, 1.0);
    jobs.par_iter()
        .filter_map(|&(first, second, first_point)| {
            let transform = inverses[second]? * homographies[first];
            let source = &analyses[first];
            let target = &analyses[second];
            if !quarter_valid(source, first_point, 48.0) {
                return None;
            }
            let predicted = project(&transform, first_point)?;
            if !quarter_valid(target, predicted, 48.0) {
                return None;
            }
            // Coarse search on quarter-resolution luma covers the residual
            // the 2400px feature graph leaves (tens of native pixels).
            let coarse_transform = quarter * transform * quarter_inverse;
            let coarse_centre = project(&quarter, first_point)?;
            let coarse = probe_warped_patch(
                &source.reg_quarter,
                &target.reg_quarter,
                &coarse_transform,
                coarse_centre,
                10,
                10,
                PatchMatchGates::LEGACY,
            )
            .matched?;
            let coarse_target = project(&quarter_inverse, coarse.target)?;
            let shift = coarse_target - predicted;
            let fine_transform =
                Matrix3::new(1.0, 0.0, shift.x, 0.0, 1.0, shift.y, 0.0, 0.0, 1.0) * transform;
            let fine = probe_warped_patch(
                &source.reg_full,
                &target.reg_full,
                &fine_transform,
                first_point,
                16,
                4,
                PatchMatchGates::LEGACY,
            )
            .matched?;
            Some(Correspondence {
                first,
                second,
                first_point,
                second_point: fine.target,
            })
        })
        .collect()
}

pub(super) fn correspondence_residuals(
    homographies: &[Matrix3<f64>],
    correspondences: &[Correspondence],
) -> Vec<f64> {
    correspondences
        .iter()
        .map(|item| {
            match (
                project(&homographies[item.first], item.first_point),
                project(&homographies[item.second], item.second_point),
            ) {
                (Some(a), Some(b)) => (a - b).norm(),
                _ => f64::INFINITY,
            }
        })
        .collect()
}

pub(super) fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return f64::INFINITY;
    }
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

pub(super) fn refine_homographies(
    analyses: &[TileAnalysis],
    initial: &[Matrix3<f64>],
    fixed: usize,
) -> Vec<Matrix3<f64>> {
    let mut current = initial.to_vec();
    for round in 0..2 {
        let started = Instant::now();
        let correspondences = collect_correspondences(analyses, &current, REGISTRATION_STEP);
        let before = correspondence_residuals(&current, &correspondences);
        let Some(solved) = solve_joint_homographies(&current, &correspondences, fixed) else {
            println!(
                "  - PSD registration round {}: joint solve unavailable ({} correspondences); keeping current poses",
                round + 1,
                correspondences.len()
            );
            break;
        };
        let after = correspondence_residuals(&solved, &correspondences);
        let (before_median, after_median) = (percentile(&before, 0.5), percentile(&after, 0.5));
        println!(
            "  - PSD registration round {}: {} native correspondences, residual median {:.2}px -> {:.2}px, p90 {:.2}px -> {:.2}px ({:.2?})",
            round + 1,
            correspondences.len(),
            before_median,
            after_median,
            percentile(&before, 0.9),
            percentile(&after, 0.9),
            started.elapsed()
        );
        // Keep a round only when it strictly helped (a NaN never does).
        if after_median.partial_cmp(&before_median) != Some(std::cmp::Ordering::Less) {
            break;
        }
        current = solved;
    }
    current
}

pub(super) fn parameters_of(homography: &Matrix3<f64>) -> [f64; 8] {
    let normalize = Matrix3::new(
        1.0 / NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0 / NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let denormalize = Matrix3::new(
        NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let normalized = normalize * homography * denormalize;
    let normalized = normalized / normalized[(2, 2)];
    [
        normalized[(0, 0)],
        normalized[(0, 1)],
        normalized[(0, 2)],
        normalized[(1, 0)],
        normalized[(1, 1)],
        normalized[(1, 2)],
        normalized[(2, 0)],
        normalized[(2, 1)],
    ]
}

pub(super) fn homography_of(parameters: &[f64; 8]) -> Matrix3<f64> {
    let normalized = Matrix3::new(
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
        parameters[4],
        parameters[5],
        parameters[6],
        parameters[7],
        1.0,
    );
    let normalize = Matrix3::new(
        1.0 / NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0 / NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let denormalize = Matrix3::new(
        NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    denormalize * normalized * normalize
}

/// Normalised projection and its Jacobian with respect to the eight
/// parameters.
pub(super) fn project_normalized(
    parameters: &[f64; 8],
    x: f64,
    y: f64,
) -> (f64, f64, [f64; 8], [f64; 8]) {
    let denominator = parameters[6] * x + parameters[7] * y + 1.0;
    let u = (parameters[0] * x + parameters[1] * y + parameters[2]) / denominator;
    let v = (parameters[3] * x + parameters[4] * y + parameters[5]) / denominator;
    let inverse = 1.0 / denominator;
    (
        u,
        v,
        [
            x * inverse,
            y * inverse,
            inverse,
            0.0,
            0.0,
            0.0,
            -u * x * inverse,
            -u * y * inverse,
        ],
        [
            0.0,
            0.0,
            0.0,
            x * inverse,
            y * inverse,
            inverse,
            -v * x * inverse,
            -v * y * inverse,
        ],
    )
}

pub(super) fn huber_cost(residual: f64) -> f64 {
    if residual <= REGISTRATION_HUBER_PX {
        0.5 * residual * residual
    } else {
        REGISTRATION_HUBER_PX * (residual - 0.5 * REGISTRATION_HUBER_PX)
    }
}

/// Joint robust Levenberg–Marquardt adjustment of all free tile poses.
pub(super) fn solve_joint_homographies(
    initial: &[Matrix3<f64>],
    correspondences: &[Correspondence],
    fixed: usize,
) -> Option<Vec<Matrix3<f64>>> {
    let count = initial.len();
    let mut support = vec![0usize; count];
    for item in correspondences {
        support[item.first] += 1;
        support[item.second] += 1;
    }
    let free = (0..count)
        .filter(|&index| index != fixed && support[index] >= 24)
        .collect::<Vec<_>>();
    if free.is_empty() {
        return None;
    }
    let mut slot = vec![usize::MAX; count];
    for (position, &index) in free.iter().enumerate() {
        slot[index] = position;
    }
    let dimension = free.len() * 8;
    let mut parameters = initial.iter().map(parameters_of).collect::<Vec<_>>();
    let points = correspondences
        .iter()
        .map(|item| {
            (
                item.first,
                item.second,
                item.first_point.x / NORMALIZED_UNIT,
                item.first_point.y / NORMALIZED_UNIT,
                item.second_point.x / NORMALIZED_UNIT,
                item.second_point.y / NORMALIZED_UNIT,
            )
        })
        .collect::<Vec<_>>();
    let mut active = vec![true; points.len()];
    let residual_of = |parameters: &[[f64; 8]], point: &(usize, usize, f64, f64, f64, f64)| {
        let (first, second, ax, ay, bx, by) = *point;
        let (u1, v1, _, _) = project_normalized(&parameters[first], ax, ay);
        let (u2, v2, _, _) = project_normalized(&parameters[second], bx, by);
        ((u1 - u2) * NORMALIZED_UNIT, (v1 - v2) * NORMALIZED_UNIT)
    };
    let total_cost = |parameters: &[[f64; 8]], active: &[bool]| -> f64 {
        points
            .iter()
            .zip(active)
            .filter(|(_, active)| **active)
            .map(|(point, _)| {
                let (rx, ry) = residual_of(parameters, point);
                huber_cost(rx.hypot(ry))
            })
            .sum()
    };
    for pass in 0..2 {
        let mut damping = 1e-3;
        let mut cost = total_cost(&parameters, &active);
        for _ in 0..50 {
            let mut normal = DMatrix::<f64>::zeros(dimension, dimension);
            let mut gradient = DVector::<f64>::zeros(dimension);
            for (point, _) in points.iter().zip(&active).filter(|(_, active)| **active) {
                let (first, second, ax, ay, bx, by) = *point;
                let (u1, v1, ju1, jv1) = project_normalized(&parameters[first], ax, ay);
                let (u2, v2, ju2, jv2) = project_normalized(&parameters[second], bx, by);
                let rx = (u1 - u2) * NORMALIZED_UNIT;
                let ry = (v1 - v2) * NORMALIZED_UNIT;
                let magnitude = rx.hypot(ry);
                let weight = if magnitude <= REGISTRATION_HUBER_PX {
                    1.0
                } else {
                    REGISTRATION_HUBER_PX / magnitude
                };
                let blocks = [(slot[first], ju1, jv1, 1.0), (slot[second], ju2, jv2, -1.0)];
                for &(row_slot, ru, rv, row_sign) in &blocks {
                    if row_slot == usize::MAX {
                        continue;
                    }
                    for i in 0..8 {
                        let jxi = ru[i] * NORMALIZED_UNIT * row_sign;
                        let jyi = rv[i] * NORMALIZED_UNIT * row_sign;
                        gradient[row_slot * 8 + i] += weight * (jxi * rx + jyi * ry);
                        for &(column_slot, cu, cv, column_sign) in &blocks {
                            if column_slot == usize::MAX {
                                continue;
                            }
                            for j in 0..8 {
                                let jxj = cu[j] * NORMALIZED_UNIT * column_sign;
                                let jyj = cv[j] * NORMALIZED_UNIT * column_sign;
                                normal[(row_slot * 8 + i, column_slot * 8 + j)] +=
                                    weight * (jxi * jxj + jyi * jyj);
                            }
                        }
                    }
                }
            }
            let mut improved = false;
            for _ in 0..8 {
                let mut system = normal.clone();
                for d in 0..dimension {
                    system[(d, d)] += damping * normal[(d, d)].max(1e-9);
                }
                let Some(cholesky) = system.cholesky() else {
                    damping *= 10.0;
                    continue;
                };
                let step = cholesky.solve(&(-&gradient));
                let mut trial = parameters.clone();
                for &index in &free {
                    for i in 0..8 {
                        trial[index][i] += step[slot[index] * 8 + i];
                    }
                }
                let trial_cost = total_cost(&trial, &active);
                if trial_cost.is_finite() && trial_cost < cost {
                    let largest = step.amax();
                    parameters = trial;
                    cost = trial_cost;
                    damping = (damping * 0.3).max(1e-9);
                    improved = largest > 1e-12;
                    break;
                }
                damping *= 10.0;
            }
            if !improved {
                break;
            }
        }
        if pass == 0 {
            let residuals = points
                .iter()
                .map(|point| {
                    let (rx, ry) = residual_of(&parameters, point);
                    rx.hypot(ry)
                })
                .collect::<Vec<_>>();
            let median = percentile(&residuals, 0.5);
            let limit = (median * 4.0).max(4.0);
            for (flag, residual) in active.iter_mut().zip(&residuals) {
                *flag = *residual <= limit;
            }
        }
    }
    let solved = parameters.iter().map(homography_of).collect::<Vec<_>>();
    solved
        .iter()
        .all(|matrix| {
            matrix.iter().all(|value| value.is_finite()) && matrix.try_inverse().is_some()
        })
        .then_some(solved)
}
