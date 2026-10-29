//! Projective/local decomposition of the joint station fields.
//!
//! Fits each field's smooth projective part into its camera pose, chooses the
//! common world gauge within the original corner budgets and recovers the
//! native pixel density the gauge allows.

use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::SMatrix;
use nalgebra::SVector;

use super::{STEP, StationField, corners, project, projection_jacobian};

pub(super) fn normalized_points(
    points: &[Point2<f64>],
) -> Option<(Vec<Point2<f64>>, Matrix3<f64>)> {
    if points.len() < 4 {
        return None;
    }
    let center = points.iter().fold(Point2::origin(), |sum, p| {
        Point2::from(sum.coords + p.coords)
    }) / points.len() as f64;
    let distance = points.iter().map(|p| (p - center).norm()).sum::<f64>() / points.len() as f64;
    if !distance.is_finite() || distance <= 1e-8 {
        return None;
    }
    let scale = 2.0f64.sqrt() / distance;
    let transform = Matrix3::new(
        scale,
        0.0,
        -scale * center.x,
        0.0,
        scale,
        -scale * center.y,
        0.0,
        0.0,
        1.0,
    );
    Some((
        points
            .iter()
            .map(|p| Point2::from((p - center) * scale))
            .collect(),
        transform,
    ))
}

pub(super) fn fit_homography(
    source: &[Point2<f64>],
    target: &[Point2<f64>],
    weights: &[f64],
) -> Option<Matrix3<f64>> {
    if source.len() != target.len() || source.len() != weights.len() {
        return None;
    }
    let (source, s) = normalized_points(source)?;
    let (target, t) = normalized_points(target)?;
    let mut normal = SMatrix::<f64, 9, 9>::zeros();
    for ((a, b), &weight) in source.iter().zip(&target).zip(weights) {
        let rows = [
            [-a.x, -a.y, -1.0, 0.0, 0.0, 0.0, b.x * a.x, b.x * a.y, b.x],
            [0.0, 0.0, 0.0, -a.x, -a.y, -1.0, b.y * a.x, b.y * a.y, b.y],
        ];
        for row in rows {
            let row = SVector::<f64, 9>::from_row_slice(&row);
            normal += weight * (row * row.transpose());
        }
    }
    let eigen = normal.symmetric_eigen();
    let index = (0..9).min_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]))?;
    let vector = eigen.eigenvectors.column(index);
    let h = t.try_inverse()? * Matrix3::from_fn(|row, column| vector[row * 3 + column]) * s;
    if h.iter().any(|v| !v.is_finite()) || h[(2, 2)].abs() < 1e-8 {
        return None;
    }
    let h = h / h[(2, 2)];
    h.try_inverse()?;
    Some(h)
}

pub(crate) struct Decomposition {
    pub poses: Vec<Matrix3<f64>>,
    pub gauge: Matrix3<f64>,
    pub maximum_corner_motion: f64,
}

/// The common coordinate gauge is free, but it must not needlessly discard
/// native pixel density. Recover as much of the native 1.0 density floor as the
/// original camera corner budgets allow. The physical map is transformed
/// together with its local field and is still independently validated below.
pub(super) fn native_density_gauge(
    mut result: Decomposition,
    dimensions: &[(u32, u32)],
    initial: &[Matrix3<f64>],
    limits: &[f64],
) -> Option<Decomposition> {
    let mut density = f64::INFINITY;
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for (pose, &size) in result.poses.iter().zip(dimensions) {
        for point in corners(size) {
            let world = project(pose, point)?;
            bounds[0] = bounds[0].min(world.x);
            bounds[1] = bounds[1].min(world.y);
            bounds[2] = bounds[2].max(world.x);
            bounds[3] = bounds[3].max(world.y);
            let z = pose[(2, 0)] * point.x + pose[(2, 1)] * point.y + pose[(2, 2)];
            let local = (pose.determinant() / z.powi(3)).abs().sqrt();
            if !local.is_finite() || local <= 0.0 {
                return None;
            }
            density = density.min(local);
        }
    }
    let target = (1.0 / density).max(1.0);
    if target <= 1.0 + 1e-10 {
        return Some(result);
    }
    let center = ((bounds[0] + bounds[2]) * 0.5, (bounds[1] + bounds[3]) * 0.5);
    // Uniform output density is a common gauge. Its translation is free;
    // pinning the magnification at the canvas centre can spend a camera's
    // entire corner budget even when another common pivot is feasible.
    let motion = |scale: f64| -> Option<(Matrix3<f64>, f64)> {
        let mut disks = Vec::new();
        for (station, pose) in result.poses.iter().enumerate() {
            for point in corners(dimensions[station]) {
                disks.push((
                    Point2::from(
                        project(&initial[station], point)? - project(pose, point)? * scale,
                    ),
                    limits[station],
                ));
            }
        }
        let mut t = Point2::new(center.0 * (1.0 - scale), center.1 * (1.0 - scale));
        let feasible = |t: Point2<f64>| disks.iter().all(|(c, r)| (t - *c).norm() <= *r + 1e-7);
        if !feasible(t) {
            let mut corrections = vec![nalgebra::Vector2::zeros(); disks.len()];
            for _ in 0..2048 {
                for ((c, r), correction) in disks.iter().zip(&mut corrections) {
                    let proposal = t + *correction;
                    let delta = proposal - *c;
                    let distance = delta.norm();
                    let next = if distance > *r {
                        *c + delta * ((*r - 1e-8).max(0.0) / distance)
                    } else {
                        proposal
                    };
                    *correction = proposal - next;
                    t = next;
                }
                if feasible(t) {
                    break;
                }
            }
        }
        if !feasible(t) {
            return None;
        }
        let common = Matrix3::new(scale, 0.0, t.x, 0.0, scale, t.y, 0.0, 0.0, 1.0);
        let maximum = disks
            .iter()
            .map(|(c, _)| (t - *c).norm())
            .fold(0.0, f64::max);
        Some((common, maximum))
    };
    let (mut low, mut high) = (1.0, target);
    for _ in 0..40 {
        let middle = (low + high) * 0.5;
        if motion(middle).is_some() {
            low = middle;
        } else {
            high = middle;
        }
    }
    let scale = 1.0 + (low - 1.0) * 0.999999;
    let (common, maximum) = motion(scale)?;
    result
        .poses
        .iter_mut()
        .for_each(|pose| *pose = common * (*pose));
    result.gauge = common * result.gauge;
    result.maximum_corner_motion = maximum;
    Some(result)
}

pub(super) fn geometric_gauge(
    source: &[Point2<f64>],
    target: &[Point2<f64>],
    limits: &[f64],
    initial: Matrix3<f64>,
) -> Option<Matrix3<f64>> {
    let center = Point2::from(
        source
            .iter()
            .chain(target)
            .map(|p| p.coords)
            .fold(nalgebra::Vector2::zeros(), |sum, p| sum + p)
            / (source.len() + target.len()) as f64,
    );
    let scale = source
        .iter()
        .chain(target)
        .map(|p| (p - center).norm())
        .sum::<f64>()
        / (source.len() + target.len()) as f64;
    if !scale.is_finite() || scale <= 1e-8 {
        return None;
    }
    let t = Matrix3::new(
        1.0 / scale,
        0.0,
        -center.x / scale,
        0.0,
        1.0 / scale,
        -center.y / scale,
        0.0,
        0.0,
        1.0,
    );
    let inverse = t.try_inverse()?;
    let a = source
        .iter()
        .map(|&p| project(&t, p))
        .collect::<Option<Vec<_>>>()?;
    let b = target
        .iter()
        .map(|&p| project(&t, p))
        .collect::<Option<Vec<_>>>()?;
    let limits = limits.iter().map(|v| v / scale).collect::<Vec<_>>();
    let mut h = t * initial * inverse;
    h /= h[(2, 2)];
    for power in [2, 4, 8, 16, 32] {
        for _ in 0..100 {
            let errors = a
                .iter()
                .zip(&b)
                .map(|(&a, &b)| Some(project(&h, a)? - b))
                .collect::<Option<Vec<_>>>()?;
            let ratios = errors
                .iter()
                .zip(&limits)
                .map(|(e, l)| e.norm() / l)
                .collect::<Vec<_>>();
            let maximum = ratios.iter().copied().fold(1e-8, f64::max);
            let loss = ratios.iter().map(|r| r.powi(power)).sum::<f64>();
            let mut normal = SMatrix::<f64, 8, 8>::zeros();
            let mut rhs = SVector::<f64, 8>::zeros();
            for i in 0..a.len() {
                let (_, jacobian) = projection_jacobian(&h, a[i])?;
                let weight = (ratios[i] / maximum).max(0.001).powi(power - 2) / limits[i].powi(2);
                for axis in 0..2 {
                    let row = SVector::<f64, 8>::from_row_slice(&jacobian[axis]);
                    normal += weight * (row * row.transpose());
                    rhs -= weight * row * errors[i][axis];
                }
            }
            for i in 0..8 {
                normal[(i, i)] += normal[(i, i)].max(1.0) * 1e-9;
            }
            let delta = normal.lu().solve(&rhs)?;
            let mut accepted = None;
            for factor in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125] {
                let mut candidate = h;
                for i in 0..8 {
                    candidate[(i / 3, i % 3)] += factor * delta[i];
                }
                let candidate_losses = a
                    .iter()
                    .zip(&b)
                    .zip(&limits)
                    .map(|((&a, &b), limit)| {
                        Some(((project(&candidate, a)? - b).norm() / limit).powi(power))
                    })
                    .collect::<Option<Vec<_>>>();
                let Some(candidate_losses) = candidate_losses else {
                    continue;
                };
                let candidate_loss = candidate_losses.into_iter().sum::<f64>();
                if candidate_loss < loss {
                    accepted = Some(candidate);
                    break;
                }
            }
            let Some(candidate) = accepted else {
                break;
            };
            h = candidate;
        }
    }
    Some(inverse * h * t)
}

pub(super) fn corner_bounded_pose(
    candidate: Matrix3<f64>,
    initial: Matrix3<f64>,
    dimensions: (u32, u32),
    limit: f64,
) -> Option<(Matrix3<f64>, f64)> {
    let candidate = candidate / candidate[(2, 2)];
    let initial = initial / initial[(2, 2)];
    let motion = |pose: &Matrix3<f64>| {
        corners(dimensions)
            .into_iter()
            .map(|point| Some((project(pose, point)? - project(&initial, point)?).norm()))
            .collect::<Option<Vec<_>>>()
            .map(|values| values.into_iter().fold(0.0, f64::max))
    };
    let mut pose = candidate;
    if motion(&pose)? > limit {
        let (mut low, mut high) = (0.0, 1.0);
        for _ in 0..50 {
            let alpha = (low + high) * 0.5;
            let trial = candidate * (1.0 - alpha) + initial * alpha;
            if motion(&trial)? <= limit * (1.0 - 1e-9) {
                high = alpha;
            } else {
                low = alpha;
            }
        }
        pose = candidate * (1.0 - high) + initial * high;
    }
    pose.try_inverse()?;
    Some((pose, motion(&pose)?))
}

/// Move the smooth projective part of each field into its camera pose and
/// choose a common world gauge. This leaves the fitted correspondence map
/// unchanged, but exposes the real global/local displacement bounds.
pub(crate) fn decompose(
    proposal: &[Matrix3<f64>],
    fields: &[StationField],
    dimensions: &[(u32, u32)],
    initial: &[Matrix3<f64>],
    limits: &[f64],
) -> Option<Decomposition> {
    let count = proposal.len();
    if [fields.len(), dimensions.len(), initial.len(), limits.len()]
        .iter()
        .any(|&n| n != count)
    {
        return None;
    }
    let mut fitted = Vec::with_capacity(count);
    let mut samples = Vec::with_capacity(count);
    for station in 0..count {
        let field = &fields[station];
        let inverse = proposal[station].try_inverse()?;
        let mut source = Vec::new();
        let mut target = Vec::new();
        for row in 0..field.rows {
            for column in 0..field.columns {
                let world =
                    field.origin + nalgebra::Vector2::new(column as f64 * STEP, row as f64 * STEP);
                let point = project(&inverse, world)?;
                if point.x < 0.0
                    || point.y < 0.0
                    || point.x > f64::from(dimensions[station].0)
                    || point.y > f64::from(dimensions[station].1)
                {
                    continue;
                }
                let value = field.values[row * field.columns + column];
                source.push(point);
                target.push(world + nalgebra::Vector2::new(value[0], value[1]));
            }
        }
        let mut h = fit_homography(&source, &target, &vec![1.0; source.len()])?;
        for _ in 0..25 {
            let errors = source
                .iter()
                .zip(&target)
                .map(|(a, b)| Some((project(&h, *a)? - *b).norm()))
                .collect::<Option<Vec<_>>>()?;
            let maximum = errors.iter().copied().fold(1e-8, f64::max);
            let weights = errors
                .iter()
                .map(|e| (e / maximum).max(0.1).powi(4))
                .collect::<Vec<_>>();
            let candidate = fit_homography(&source, &target, &weights)?;
            h = h * 0.8 + candidate * 0.2;
            h /= h[(2, 2)];
        }
        fitted.push(h);
        samples.push((source, target));
    }
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut corner_limits = Vec::new();
    for station in 0..count {
        if !limits[station].is_finite() || limits[station] <= 0.0 {
            return None;
        }
        for corner in corners(dimensions[station]) {
            a.push(project(&fitted[station], corner)?);
            b.push(project(&initial[station], corner)?);
            corner_limits.push(limits[station]);
        }
    }
    // Repeated camera corners may make the algebraic proposal singular.
    // Identity remains a valid starting gauge for the geometric solve;
    // publication still requires all final corner/local bounds.
    let mut gauge = fit_homography(&a, &b, &vec![1.0; a.len()]).unwrap_or_else(Matrix3::identity);
    for _ in 0..100 {
        let errors = a
            .iter()
            .zip(&b)
            .zip(&corner_limits)
            .map(|((a, b), limit)| Some((project(&gauge, *a)? - *b).norm() / limit))
            .collect::<Option<Vec<_>>>()?;
        let maximum = errors.iter().copied().fold(1e-8, f64::max);
        let weights = errors
            .iter()
            .map(|e| (e / maximum).max(0.1).powi(6))
            .collect::<Vec<_>>();
        let Some(candidate) = fit_homography(&a, &b, &weights) else {
            break;
        };
        gauge = gauge * 0.9 + candidate * 0.1;
        gauge /= gauge[(2, 2)];
    }
    let needs_geometric_fit = a
        .iter()
        .zip(&b)
        .zip(&corner_limits)
        .any(|((&a, &b), limit)| project(&gauge, a).is_none_or(|p| (p - b).norm() > *limit));
    if needs_geometric_fit {
        if let Some(refined) = geometric_gauge(&a, &b, &corner_limits, gauge) {
            gauge = refined;
        }
    }
    let translations: &[f64] = if needs_geometric_fit {
        &[-4.0, -2.0, 0.0, 2.0, 4.0]
    } else {
        &[0.0]
    };
    let mut best = None;
    for &x in translations {
        for &y in translations {
            let current = Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0) * gauge;
            let mut poses = Vec::with_capacity(count);
            let mut max_corner = 0.0f64;
            let mut max_local = 0.0f64;
            for station in 0..count {
                let (pose, motion) = corner_bounded_pose(
                    current * fitted[station],
                    initial[station],
                    dimensions[station],
                    limits[station],
                )?;
                max_corner = max_corner.max(motion);
                for (&source, &target) in samples[station].0.iter().zip(&samples[station].1) {
                    max_local = max_local
                        .max((project(&current, target)? - project(&pose, source)?).norm());
                }
                poses.push(pose);
            }
            if best.as_ref().is_none_or(|(cost, _)| max_local < *cost) {
                best = Some((
                    max_local,
                    Decomposition {
                        poses,
                        gauge: current,
                        maximum_corner_motion: max_corner,
                    },
                ));
            }
        }
    }
    let (cost, result) = best?;
    #[cfg(test)]
    if cost > crate::panorama_utils::stack_pipeline::residual_warp::RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX {
        eprintln!("Joint decomposition rejected: required local displacement {cost:.6}px");
    }
    if cost > crate::panorama_utils::stack_pipeline::residual_warp::RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX {
        return None;
    }
    native_density_gauge(result, dimensions, initial, limits)
}
