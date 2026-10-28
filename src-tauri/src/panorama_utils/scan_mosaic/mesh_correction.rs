//! Smooth per-tile registration correction.
//!
//! Fits a displacement on a control grid for what a planar pose cannot follow
//! (lens distortion, a scroll that is not perfectly flat) and keeps it only
//! when held-out correspondences confirm it.

use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::time::Instant;

use super::{
    Correspondence, DisplacementGrid, MESH_DAMPING, MESH_HOLDOUT, MESH_HUBER_PX, MESH_SMOOTHNESS,
    MESH_SPACING, MESH_STEP, TileAnalysis, collect_correspondences, correspondence_residuals,
    percentile, project,
};

// ---------------------------------------------------------------------------
// Smooth registration correction
// ---------------------------------------------------------------------------

/// Jacobian `[[du/dx, du/dy], [dv/dx, dv/dy]]` of a homography at `point`.
pub(super) fn homography_jacobian(
    homography: &Matrix3<f64>,
    point: Point2<f64>,
) -> Option<[[f64; 2]; 2]> {
    let mapped = homography * Vector3::new(point.x, point.y, 1.0);
    if mapped.z.abs() < 1e-12 {
        return None;
    }
    let (u, v, w) = (mapped.x / mapped.z, mapped.y / mapped.z, mapped.z);
    Some([
        [
            (homography[(0, 0)] - u * homography[(2, 0)]) / w,
            (homography[(0, 1)] - u * homography[(2, 1)]) / w,
        ],
        [
            (homography[(1, 0)] - v * homography[(2, 0)]) / w,
            (homography[(1, 1)] - v * homography[(2, 1)]) / w,
        ],
    ])
}

/// One correspondence of the correction solve, linearised at no correction:
/// `r(x) = r0 + Ja·F_a(p_a) − Jb·F_b(p_b)` with both corrections bilinear in
/// the node unknowns.
pub(super) struct MeshRow {
    pub(super) first: [(usize, f64); 4],
    pub(super) second: [(usize, f64); 4],
    pub(super) first_jacobian: [[f64; 2]; 2],
    pub(super) second_jacobian: [[f64; 2]; 2],
    pub(super) initial: [f64; 2],
}

impl MeshRow {
    #[inline]
    pub(super) fn correction(stencil: &[(usize, f64); 4], unknowns: &[f64]) -> [f64; 2] {
        let mut value = [0.0; 2];
        for &(node, weight) in stencil {
            value[0] += weight * unknowns[node * 2];
            value[1] += weight * unknowns[node * 2 + 1];
        }
        value
    }

    #[inline]
    pub(super) fn linear_part(&self, unknowns: &[f64]) -> [f64; 2] {
        let a = Self::correction(&self.first, unknowns);
        let b = Self::correction(&self.second, unknowns);
        let (ja, jb) = (&self.first_jacobian, &self.second_jacobian);
        [
            ja[0][0] * a[0] + ja[0][1] * a[1] - jb[0][0] * b[0] - jb[0][1] * b[1],
            ja[1][0] * a[0] + ja[1][1] * a[1] - jb[1][0] * b[0] - jb[1][1] * b[1],
        ]
    }

    #[inline]
    pub(super) fn residual(&self, unknowns: &[f64]) -> [f64; 2] {
        let linear = self.linear_part(unknowns);
        [self.initial[0] + linear[0], self.initial[1] + linear[1]]
    }

    /// Adds `weight · Aᵀ·value` of this row to `output`.
    #[inline]
    pub(super) fn scatter(&self, value: [f64; 2], weight: f64, output: &mut [f64]) {
        let (ja, jb) = (&self.first_jacobian, &self.second_jacobian);
        let a = [
            weight * (ja[0][0] * value[0] + ja[1][0] * value[1]),
            weight * (ja[0][1] * value[0] + ja[1][1] * value[1]),
        ];
        let b = [
            weight * (jb[0][0] * value[0] + jb[1][0] * value[1]),
            weight * (jb[0][1] * value[0] + jb[1][1] * value[1]),
        ];
        for &(node, stencil_weight) in &self.first {
            output[node * 2] += stencil_weight * a[0];
            output[node * 2 + 1] += stencil_weight * a[1];
        }
        for &(node, stencil_weight) in &self.second {
            output[node * 2] -= stencil_weight * b[0];
            output[node * 2 + 1] -= stencil_weight * b[1];
        }
    }
}

/// `Aᵀ·W·(A·x + shift)` over all rows, accumulated in parallel.
pub(super) fn mesh_normal_product(
    rows: &[MeshRow],
    weights: &[f64],
    unknowns: &[f64],
    with_initial: bool,
) -> Vec<f64> {
    let length = unknowns.len();
    rows.par_chunks(1024)
        .zip(weights.par_chunks(1024))
        .fold(
            || vec![0.0; length],
            |mut output, (rows, weights)| {
                for (row, &weight) in rows.iter().zip(weights) {
                    if weight <= 0.0 {
                        continue;
                    }
                    let value = if with_initial {
                        row.residual(unknowns)
                    } else {
                        row.linear_part(unknowns)
                    };
                    row.scatter(value, weight, &mut output);
                }
                output
            },
        )
        .reduce(
            || vec![0.0; length],
            |mut left, right| {
                for (value, other) in left.iter_mut().zip(&right) {
                    *value += other;
                }
                left
            },
        )
}

/// `(AᵀWA + λs·L + λ0·I)·x`.
pub(super) fn mesh_system_product(
    rows: &[MeshRow],
    weights: &[f64],
    edges: &[(usize, usize)],
    unknowns: &[f64],
) -> Vec<f64> {
    let mut output = mesh_normal_product(rows, weights, unknowns, false);
    for &(first, second) in edges {
        for component in 0..2 {
            let difference = MESH_SMOOTHNESS
                * (unknowns[first * 2 + component] - unknowns[second * 2 + component]);
            output[first * 2 + component] += difference;
            output[second * 2 + component] -= difference;
        }
    }
    for (value, unknown) in output.iter_mut().zip(unknowns) {
        *value += MESH_DAMPING * unknown;
    }
    output
}

/// Jacobi-preconditioned conjugate gradients on the correction normal
/// equations, warm-started from `unknowns`.
pub(super) fn solve_mesh_system(
    rows: &[MeshRow],
    weights: &[f64],
    edges: &[(usize, usize)],
    unknowns: &mut [f64],
) -> usize {
    let length = unknowns.len();
    let mut diagonal = vec![MESH_DAMPING; length];
    for &(first, second) in edges {
        for component in 0..2 {
            diagonal[first * 2 + component] += MESH_SMOOTHNESS;
            diagonal[second * 2 + component] += MESH_SMOOTHNESS;
        }
    }
    for (row, &weight) in rows.iter().zip(weights) {
        if weight <= 0.0 {
            continue;
        }
        for (stencil, jacobian) in [
            (&row.first, &row.first_jacobian),
            (&row.second, &row.second_jacobian),
        ] {
            let column_x = jacobian[0][0] * jacobian[0][0] + jacobian[1][0] * jacobian[1][0];
            let column_y = jacobian[0][1] * jacobian[0][1] + jacobian[1][1] * jacobian[1][1];
            for &(node, stencil_weight) in stencil {
                let squared = weight * stencil_weight * stencil_weight;
                diagonal[node * 2] += squared * column_x;
                diagonal[node * 2 + 1] += squared * column_y;
            }
        }
    }
    // Right-hand side −AᵀW·r0 and the initial residual b − M·x.
    let zero = vec![0.0; length];
    let rhs = mesh_normal_product(rows, weights, &zero, true)
        .into_iter()
        .map(|value| -value)
        .collect::<Vec<_>>();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    if rhs_norm < 1e-12 {
        return 0;
    }
    let product = mesh_system_product(rows, weights, edges, unknowns);
    let mut residual = rhs
        .iter()
        .zip(&product)
        .map(|(b, ax)| b - ax)
        .collect::<Vec<_>>();
    let mut preconditioned = residual
        .iter()
        .zip(&diagonal)
        .map(|(r, d)| r / d)
        .collect::<Vec<_>>();
    let mut direction = preconditioned.clone();
    let mut rz = residual
        .iter()
        .zip(&preconditioned)
        .map(|(r, z)| r * z)
        .sum::<f64>();
    let mut iterations = 0;
    for iteration in 0..3000 {
        iterations = iteration;
        let residual_norm = residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if residual_norm <= 1e-7 * rhs_norm {
            break;
        }
        let product = mesh_system_product(rows, weights, edges, &direction);
        let curvature = direction
            .iter()
            .zip(&product)
            .map(|(p, ap)| p * ap)
            .sum::<f64>();
        if curvature <= 0.0 || !curvature.is_finite() {
            break;
        }
        let alpha = rz / curvature;
        for (unknown, step) in unknowns.iter_mut().zip(&direction) {
            *unknown += alpha * step;
        }
        for (((r, z), ap), d) in residual
            .iter_mut()
            .zip(preconditioned.iter_mut())
            .zip(&product)
            .zip(&diagonal)
        {
            *r -= alpha * ap;
            *z = *r / d;
        }
        let next = residual
            .iter()
            .zip(&preconditioned)
            .map(|(r, z)| r * z)
            .sum::<f64>();
        let beta = next / rz.max(1e-300);
        rz = next;
        for (p, z) in direction.iter_mut().zip(&preconditioned) {
            *p = z + beta * *p;
        }
    }
    iterations
}

/// Fit one smooth correction per tile so that every native correspondence
/// lands on the same world point. Returns `None` when the correction does not
/// also align the held-out correspondences better than the planar poses.
pub(super) fn refine_mesh(
    analyses: &[TileAnalysis],
    homographies: &[Matrix3<f64>],
) -> Option<Vec<DisplacementGrid>> {
    let started = Instant::now();
    let correspondences = collect_correspondences(analyses, homographies, MESH_STEP);
    if correspondences.len() < 64 {
        println!(
            "  - PSD registration correction skipped: {} correspondences",
            correspondences.len()
        );
        return None;
    }
    let mut grids = analyses
        .iter()
        .map(|analysis| DisplacementGrid::zero(analysis.width, analysis.height, MESH_SPACING))
        .collect::<Vec<_>>();
    let mut offsets = Vec::with_capacity(grids.len());
    let mut nodes = 0usize;
    for grid in &grids {
        offsets.push(nodes);
        nodes += grid.len();
    }
    let mut edges = Vec::new();
    for (grid, &offset) in grids.iter().zip(&offsets) {
        for row in 0..grid.rows {
            for column in 0..grid.columns {
                let node = offset + row * grid.columns + column;
                if column + 1 < grid.columns {
                    edges.push((node, node + 1));
                }
                if row + 1 < grid.rows {
                    edges.push((node, node + grid.columns));
                }
            }
        }
    }
    let (correspondences, rows): (Vec<_>, Vec<_>) = correspondences
        .into_iter()
        .filter_map(|item| {
            let a = project(&homographies[item.first], item.first_point)?;
            let b = project(&homographies[item.second], item.second_point)?;
            let shift = |stencil: [(usize, f64); 4], offset: usize| {
                stencil.map(|(node, weight)| (node + offset, weight))
            };
            let row = MeshRow {
                first: shift(
                    grids[item.first].stencil(item.first_point),
                    offsets[item.first],
                ),
                second: shift(
                    grids[item.second].stencil(item.second_point),
                    offsets[item.second],
                ),
                first_jacobian: homography_jacobian(&homographies[item.first], item.first_point)?,
                second_jacobian: homography_jacobian(
                    &homographies[item.second],
                    item.second_point,
                )?,
                initial: [a.x - b.x, a.y - b.y],
            };
            Some((item, row))
        })
        .unzip();
    let holdout = (0..rows.len())
        .map(|index| index % MESH_HOLDOUT == MESH_HOLDOUT - 1)
        .collect::<Vec<_>>();
    let mut active = holdout.iter().map(|&held| !held).collect::<Vec<_>>();
    let mut unknowns = vec![0.0; nodes * 2];
    let mut iterations = 0;
    for pass in 0..5 {
        let residuals = rows
            .iter()
            .map(|row| {
                let r = row.residual(&unknowns);
                r[0].hypot(r[1])
            })
            .collect::<Vec<_>>();
        if pass == 3 {
            // Reject what the fitted correction cannot explain (repeated
            // texture, glare) and refit without it.
            let fitted = residuals
                .iter()
                .zip(&active)
                .filter(|(_, active)| **active)
                .map(|(residual, _)| *residual)
                .collect::<Vec<_>>();
            let limit = (percentile(&fitted, 0.5) * 4.0).max(2.0);
            for (flag, residual) in active.iter_mut().zip(&residuals) {
                *flag = *flag && *residual <= limit;
            }
        }
        let weights = residuals
            .iter()
            .zip(&active)
            .map(|(&residual, &active)| {
                if !active {
                    0.0
                } else if residual <= MESH_HUBER_PX {
                    1.0
                } else {
                    MESH_HUBER_PX / residual
                }
            })
            .collect::<Vec<_>>();
        iterations += solve_mesh_system(&rows, &weights, &edges, &mut unknowns);
    }
    for (grid, &offset) in grids.iter_mut().zip(&offsets) {
        for (index, value) in grid.values.iter_mut().enumerate() {
            let node = offset + index;
            *value = [unknowns[node * 2], unknowns[node * 2 + 1]];
        }
    }
    if grids.iter().any(|grid| {
        grid.values
            .iter()
            .any(|value| !value[0].is_finite() || !value[1].is_finite())
    }) {
        println!("  - PSD registration correction rejected: non-finite solution");
        return None;
    }
    // Exact (not linearised) residuals of the corrected poses.
    let corrected = |item: &Correspondence| -> f64 {
        let moved = |index: usize, point: Point2<f64>| {
            let offset = grids[index].at(point);
            project(
                &homographies[index],
                Point2::new(point.x + offset[0], point.y + offset[1]),
            )
        };
        match (
            moved(item.first, item.first_point),
            moved(item.second, item.second_point),
        ) {
            (Some(a), Some(b)) => (a - b).norm(),
            _ => f64::INFINITY,
        }
    };
    let planar = correspondence_residuals(homographies, &correspondences);
    let after = correspondences.iter().map(corrected).collect::<Vec<_>>();
    let pick = |values: &[f64], held: bool| {
        values
            .iter()
            .zip(&holdout)
            .filter(|(_, flag)| **flag == held)
            .map(|(value, _)| *value)
            .collect::<Vec<_>>()
    };
    let (held_before, held_after) = (pick(&planar, true), pick(&after, true));
    let (fit_before, fit_after) = (pick(&planar, false), pick(&after, false));
    let largest = grids
        .iter()
        .map(DisplacementGrid::largest)
        .fold(0.0, f64::max);
    println!(
        "  - PSD registration correction: {} correspondences ({} active), fitted median {:.2}px -> {:.2}px p90 {:.2}px -> {:.2}px; held-out median {:.2}px -> {:.2}px p90 {:.2}px -> {:.2}px; largest correction {:.2}px; {} CG iterations ({:.2?})",
        rows.len(),
        active.iter().filter(|&&flag| flag).count(),
        percentile(&fit_before, 0.5),
        percentile(&fit_after, 0.5),
        percentile(&fit_before, 0.9),
        percentile(&fit_after, 0.9),
        percentile(&held_before, 0.5),
        percentile(&held_after, 0.5),
        percentile(&held_before, 0.9),
        percentile(&held_after, 0.9),
        largest,
        iterations,
        started.elapsed()
    );
    let improves = percentile(&held_after, 0.9) < percentile(&held_before, 0.9)
        && percentile(&held_after, 0.5) <= percentile(&held_before, 0.5);
    if !improves {
        println!("  - PSD registration correction rejected: held-out residuals did not improve");
        return None;
    }
    Some(grids)
}
