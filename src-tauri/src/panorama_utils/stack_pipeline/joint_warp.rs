//! Joint station fields fitted to measured correspondences, with a separate
//! validation set. The reference station fixes the coordinate gauge.

use super::report::{WorldPoint, WorldRect};
use super::residual_warp::{OverlapResidual, ResidualWarp, WarpObservation, WarpRegion};
use nalgebra::{DMatrix, DVector, Matrix3, Point2};

const STEP: f64 = 64.0;
const SMOOTHNESS: f64 = 0.001;
const RIDGE: f64 = 1e-7;

#[derive(Clone, Debug)]
pub(crate) struct Relation {
    pub left: usize,
    pub right: usize,
    pub points: Vec<(Point2<f64>, Point2<f64>)>,
}

#[derive(Clone, Debug)]
pub(crate) struct StationField {
    pub origin: Point2<f64>,
    pub columns: usize,
    pub rows: usize,
    pub values: Vec<[f64; 2]>,
    offset: usize,
}

impl StationField {
    fn taps(&self, point: Point2<f64>) -> ([usize; 4], [f64; 4]) {
        let x = ((point.x - self.origin.x) / STEP).clamp(0.0, (self.columns - 1) as f64);
        let y = ((point.y - self.origin.y) / STEP).clamp(0.0, (self.rows - 1) as f64);
        let column = (x.floor() as usize).min(self.columns - 2);
        let row = (y.floor() as usize).min(self.rows - 2);
        let (fx, fy) = (x - column as f64, y - row as f64);
        let index = self.offset + row * self.columns + column;
        (
            [
                index,
                index + 1,
                index + self.columns,
                index + self.columns + 1,
            ],
            [
                (1.0 - fx) * (1.0 - fy),
                fx * (1.0 - fy),
                (1.0 - fx) * fy,
                fx * fy,
            ],
        )
    }

    pub(crate) fn at(&self, point: Point2<f64>) -> [f64; 2] {
        let (indices, weights) = self.taps(point);
        let mut result = [0.0; 2];
        for (index, weight) in indices.into_iter().zip(weights) {
            let value = self.values[index - self.offset];
            for axis in 0..2 {
                result[axis] += value[axis] * weight;
            }
        }
        result
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PairValidation {
    pub left: usize,
    pub right: usize,
    pub count: usize,
    pub median: f64,
    pub p95: f64,
}

pub(crate) struct Fit {
    pub fields: Vec<StationField>,
    pub validation: Vec<PairValidation>,
}

/// Native bidirectional evidence, retained separately from fitting points.
/// The error must come from a patch measurement, never from a fitted model.
#[derive(Clone, Copy)]
pub(crate) struct VerifiedSource {
    pub point: Point2<f64>,
    pub round_trip_error_px: f64,
}

pub(crate) struct BoundedFit {
    pub poses: Vec<Matrix3<f64>>,
    pub residual: ResidualWarp,
    pub validation: Vec<PairValidation>,
    pub validation_errors: Vec<f64>,
    pub maximum_corner_motion: f64,
}

/// Materialize the model the compositor will actually sample. In particular,
/// validation runs after support truncation, neighbour guards and edge fading.
/// A failed validation stays visible to the caller; it cannot be published as
/// an accepted camera solution merely because the unguarded fit was accurate.
pub(crate) fn materialize(
    proposal: &[Matrix3<f64>],
    fit: &Fit,
    dimensions: &[(u32, u32)],
    initial: &[Matrix3<f64>],
    limits: &[f64],
    relations: &[Relation],
    verified: &[Vec<VerifiedSource>],
) -> Option<BoundedFit> {
    if verified.len() != proposal.len() {
        return None;
    }
    let decomposition =
        decompose(proposal, &fit.fields, dimensions, initial, limits).or_else(|| {
            #[cfg(test)]
            eprintln!("Joint materialization refused at projective/local decomposition");
            None
        })?;
    let mut global_p95_by_station = vec![0.0f64; proposal.len()];
    for relation in relations {
        let mut errors = relation
            .points
            .iter()
            .map(|&(a, b)| {
                Some(
                    (project(&decomposition.poses[relation.left], a)?
                        - project(&decomposition.poses[relation.right], b)?)
                    .norm(),
                )
            })
            .collect::<Option<Vec<_>>>()?;
        if errors.is_empty() {
            return None;
        }
        errors.sort_by(f64::total_cmp);
        let p95 = errors[(errors.len() * 95).div_ceil(100) - 1];
        for station in [relation.left, relation.right] {
            global_p95_by_station[station] = global_p95_by_station[station].max(p95);
        }
    }
    let mut residual = ResidualWarp::new();
    for station in 0..proposal.len() {
        if global_p95_by_station[station] <= super::residual_warp::RESIDUAL_WARP_ENABLE_P95_PX {
            continue;
        }
        let pose = &decomposition.poses[station];
        let inverse = pose.try_inverse()?;
        let bounds = corners(dimensions[station])
            .into_iter()
            .map(|p| project(pose, p))
            .collect::<Option<Vec<_>>>()?;
        let left = (bounds.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) / STEP).floor() * STEP;
        let top = (bounds.iter().map(|p| p.y).fold(f64::INFINITY, f64::min) / STEP).floor() * STEP;
        let right =
            (bounds.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) / STEP).ceil() * STEP;
        let bottom =
            (bounds.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) / STEP).ceil() * STEP;
        let overlap = OverlapResidual::new(
            station,
            station,
            WorldRect {
                left,
                top,
                width: right - left,
                height: bottom - top,
            },
            global_p95_by_station[station],
        );
        let grid = WarpRegion::identity(overlap);
        let delta_at = |world: Point2<f64>| -> Option<[f64; 2]> {
            let source = project(&inverse, world)?;
            let old_world = project(&proposal[station], source)?;
            let delta = fit.fields[station].at(old_world);
            let target = project(
                &decomposition.gauge,
                old_world + nalgebra::Vector2::new(delta[0], delta[1]),
            )?;
            Some([target.x - world.x, target.y - world.y])
        };
        let values = grid
            .nodes()
            .iter()
            .map(|node| {
                let world = Point2::new(node.world.x, node.world.y);
                let source = project(&inverse, world)?;
                if source.x < 0.0
                    || source.y < 0.0
                    || source.x > f64::from(dimensions[station].0)
                    || source.y > f64::from(dimensions[station].1)
                {
                    return Some([0.0; 2]);
                }
                delta_at(world)
            })
            .collect::<Option<Vec<_>>>()?;
        if values.iter().all(|v| v[0].hypot(v[1]) < 1e-6) {
            continue;
        }
        let observations = verified[station]
            .iter()
            .filter_map(|evidence| {
                if !evidence.round_trip_error_px.is_finite()
                    || evidence.round_trip_error_px > 1.0
                    || evidence.round_trip_error_px < 0.0
                {
                    return None;
                }
                let world = project(pose, evidence.point)?;
                Some(WarpObservation::new(
                    WorldPoint {
                        x: world.x,
                        y: world.y,
                    },
                    delta_at(world)?,
                    evidence.round_trip_error_px,
                ))
            })
            .collect::<Vec<_>>();
        let region=WarpRegion::from_verified_grid(overlap,&values,&observations).or_else(|| {
            #[cfg(test)]
            eprintln!("Joint region {station} refused: {} observations, maximum proposed node {:.6}px",
                observations.len(),values.iter().map(|v|v[0].hypot(v[1])).fold(0.0,f64::max));
            None
        })?;
        residual.regions.push(region);
    }
    let mut validation = Vec::new();
    let mut validation_errors = Vec::new();
    for relation in relations {
        let mut points = relation.points.clone();
        sort_points(&mut points);
        let mut errors = points
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 5 == 0)
            .map(|(_, &(a, b))| {
                let forward = |station: usize, source| -> Option<Point2<f64>> {
                    let world = project(&decomposition.poses[station], source)?;
                    let delta = residual
                        .regions
                        .iter()
                        .find(|r| r.right_station == station)
                        .map(|r| r.displacement_at(world.x, world.y))
                        .unwrap_or([0.0; 2]);
                    Some(world + nalgebra::Vector2::new(delta[0], delta[1]))
                };
                Some((forward(relation.left, a)? - forward(relation.right, b)?).norm())
            })
            .collect::<Option<Vec<_>>>()?;
        if errors.is_empty() {
            return None;
        }
        errors.sort_by(f64::total_cmp);
        validation_errors.extend_from_slice(&errors);
        validation.push(PairValidation {
            left: relation.left,
            right: relation.right,
            count: errors.len(),
            median: errors[errors.len() / 2],
            p95: errors[(errors.len() * 95).div_ceil(100) - 1],
        });
    }
    Some(BoundedFit {
        poses: decomposition.poses,
        residual,
        validation,
        validation_errors,
        maximum_corner_motion: decomposition.maximum_corner_motion,
    })
}

fn sort_points(points: &mut [(Point2<f64>, Point2<f64>)]) {
    points.sort_by(|a, b| {
        a.0.x
            .total_cmp(&b.0.x)
            .then_with(|| a.0.y.total_cmp(&b.0.y))
            .then_with(|| a.1.x.total_cmp(&b.1.x))
            .then_with(|| a.1.y.total_cmp(&b.1.y))
    });
}

mod decomposition;

#[cfg(test)]
use decomposition::Decomposition;
pub(crate) use decomposition::decompose;
#[cfg(test)]
use decomposition::native_density_gauge;

fn corners(dimensions: (u32, u32)) -> [Point2<f64>; 4] {
    let (w, h) = (f64::from(dimensions.0), f64::from(dimensions.1));
    [
        Point2::origin(),
        Point2::new(w, 0.0),
        Point2::new(w, h),
        Point2::new(0.0, h),
    ]
}

fn projection_jacobian(
    h: &Matrix3<f64>,
    point: Point2<f64>,
) -> Option<(Point2<f64>, [[f64; 8]; 2])> {
    let q = project(h, point)?;
    let z = h[(2, 0)] * point.x + h[(2, 1)] * point.y + 1.0;
    let b = [point.x / z, point.y / z, 1.0 / z];
    let mut jacobian = [[0.0; 8]; 2];
    jacobian[0][..3].copy_from_slice(&b);
    jacobian[1][3..6].copy_from_slice(&b);
    for axis in 0..2 {
        jacobian[axis][6] = -q[axis] * point.x / z;
        jacobian[axis][7] = -q[axis] * point.y / z;
    }
    Some((q, jacobian))
}

/// Projective component of the joint fit. This is an internal proposal: it
/// must be decomposed and checked against the original corner limits before
/// any renderer can consume it. Validation points never enter this solve.
pub(crate) fn projective_proposal(
    initial: &[Matrix3<f64>],
    relations: &[Relation],
    reference: usize,
    scale: f64,
) -> Option<Vec<Matrix3<f64>>> {
    if initial.len() < 2 || reference >= initial.len() || !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let s = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
    let inverse_s = s.try_inverse()?;
    let mut poses = initial
        .iter()
        .map(|h| {
            let h = inverse_s * h * s;
            h / h[(2, 2)]
        })
        .collect::<Vec<_>>();
    let mut samples = Vec::new();
    for relation in relations {
        if relation.left >= poses.len() || relation.right >= poses.len() {
            return None;
        }
        let mut points = relation.points.clone();
        points.sort_by(|a, b| {
            a.0.x
                .total_cmp(&b.0.x)
                .then_with(|| a.0.y.total_cmp(&b.0.y))
        });
        for (index, (a, b)) in points.iter().enumerate() {
            if index % 5 == 0 {
                continue;
            }
            samples.push((
                relation.left,
                relation.right,
                *a / scale,
                *b / scale,
                1.0 / points.len().max(1) as f64,
            ));
        }
    }
    let cost = |poses: &[Matrix3<f64>]| -> Option<f64> {
        let mut total = 0.0;
        for &(a, b, x, y, weight) in &samples {
            let error = (project(&poses[a], x)? - project(&poses[b], y)?).norm() * scale;
            total += weight
                * if error <= 3.0 {
                    0.5 * error * error
                } else {
                    3.0 * (error - 1.5)
                };
        }
        Some(total)
    };
    let variables = (poses.len() - 1) * 8;
    let block = |station: usize| {
        if station < reference {
            station * 8
        } else {
            (station - 1) * 8
        }
    };
    let mut damping = 1e-5;
    let mut current_cost = cost(&poses)?;
    for _ in 0..100 {
        let mut normal = DMatrix::<f64>::zeros(variables, variables);
        let mut rhs = DVector::<f64>::zeros(variables);
        for &(a, b, x, y, base_weight) in &samples {
            let (pa, ja) = projection_jacobian(&poses[a], x)?;
            let (pb, jb) = projection_jacobian(&poses[b], y)?;
            let error = pa - pb;
            let weight = base_weight * (3.0 / (error.norm() * scale).max(1e-12)).min(1.0);
            for (station, jacobian, sign) in [(a, &ja, 1.0), (b, &jb, -1.0)] {
                if station == reference {
                    continue;
                }
                let offset = block(station);
                for parameter in 0..8 {
                    rhs[offset + parameter] -= weight
                        * sign
                        * (jacobian[0][parameter] * error.x + jacobian[1][parameter] * error.y);
                    for (other, other_jacobian, other_sign) in [(a, &ja, 1.0), (b, &jb, -1.0)] {
                        if other == reference {
                            continue;
                        }
                        let other_offset = block(other);
                        for column in 0..8 {
                            normal[(offset + parameter, other_offset + column)] += weight
                                * sign
                                * other_sign
                                * (jacobian[0][parameter] * other_jacobian[0][column]
                                    + jacobian[1][parameter] * other_jacobian[1][column]);
                        }
                    }
                }
            }
        }
        for i in 0..variables {
            normal[(i, i)] += damping * normal[(i, i)].max(1e-10);
        }
        let delta = normal.lu().solve(&rhs)?;
        if delta.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut candidate = poses.clone();
        for (station, h) in candidate.iter_mut().enumerate() {
            if station == reference {
                continue;
            }
            let offset = block(station);
            for (parameter, (row, column)) in [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
            ]
            .into_iter()
            .enumerate()
            {
                h[(row, column)] += delta[offset + parameter];
            }
        }
        let Some(next_cost) = cost(&candidate) else {
            damping *= 10.0;
            continue;
        };
        if next_cost >= current_cost {
            damping *= 10.0;
            continue;
        }
        let decrease = (current_cost - next_cost) / current_cost.max(1e-12);
        poses = candidate;
        current_cost = next_cost;
        damping = (damping * 0.3).max(1e-9);
        if decrease < 1e-5 {
            break;
        }
    }
    Some(poses.into_iter().map(|h| s * h * inverse_s).collect())
}

struct Row {
    indices: [usize; 8],
    coefficients: [f64; 8],
    delta: [f64; 2],
    base_weight: f64,
    training: bool,
    pair: usize,
}

pub(crate) fn project(h: &Matrix3<f64>, point: Point2<f64>) -> Option<Point2<f64>> {
    let result = h * point.to_homogeneous();
    (result.iter().all(|v| v.is_finite()) && result.z.abs() > 1e-8)
        .then(|| Point2::new(result.x / result.z, result.y / result.z))
}

fn prediction(row: &Row, values: &[f64]) -> f64 {
    row.indices
        .iter()
        .zip(row.coefficients)
        .map(|(&index, c)| values[index] * c)
        .sum()
}

fn multiply(rows: &[Row], weights: &[f64], edges: &[(usize, usize)], x: &[f64]) -> Vec<f64> {
    let mut output = x.iter().map(|v| RIDGE * v).collect::<Vec<_>>();
    for (row, &weight) in rows.iter().zip(weights) {
        let value = weight * prediction(row, x);
        for (&index, coefficient) in row.indices.iter().zip(row.coefficients) {
            output[index] += coefficient * value;
        }
    }
    for &(a, b) in edges {
        let value = SMOOTHNESS * (x[a] - x[b]);
        output[a] += value;
        output[b] -= value;
    }
    output
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn solve_axis(
    rows: &[Row],
    weights: &[f64],
    edges: &[(usize, usize)],
    axis: usize,
    x: &mut Vec<f64>,
) -> Option<()> {
    let mut rhs = vec![0.0; x.len()];
    let mut diagonal = vec![RIDGE; x.len()];
    for (row, &weight) in rows.iter().zip(weights) {
        for (&index, c) in row.indices.iter().zip(row.coefficients) {
            rhs[index] += c * weight * row.delta[axis];
            diagonal[index] += c * c * weight;
        }
    }
    for &(a, b) in edges {
        diagonal[a] += SMOOTHNESS;
        diagonal[b] += SMOOTHNESS;
    }
    let ax = multiply(rows, weights, edges, x);
    let mut residual = rhs.iter().zip(ax).map(|(b, a)| b - a).collect::<Vec<_>>();
    let mut z = residual
        .iter()
        .zip(&diagonal)
        .map(|(r, d)| r / d)
        .collect::<Vec<_>>();
    let mut direction = z.clone();
    let mut rz = dot(&residual, &z);
    for _ in 0..300 {
        if dot(&residual, &residual).sqrt() < 1e-8 {
            break;
        }
        let ad = multiply(rows, weights, edges, &direction);
        let denominator = dot(&direction, &ad);
        if !denominator.is_finite() || denominator <= 0.0 {
            return None;
        }
        let alpha = rz / denominator;
        for index in 0..x.len() {
            x[index] += alpha * direction[index];
            residual[index] -= alpha * ad[index];
            z[index] = residual[index] / diagonal[index];
        }
        let next = dot(&residual, &z);
        if !next.is_finite() || rz <= 0.0 {
            return None;
        }
        let beta = next / rz;
        for index in 0..x.len() {
            direction[index] = z[index] + beta * direction[index];
        }
        rz = next;
    }
    x.iter().all(|v| v.is_finite()).then_some(())
}

pub(crate) fn fit(
    poses: &[Matrix3<f64>],
    dimensions: &[(u32, u32)],
    relations: &[Relation],
    reference: usize,
) -> Option<Fit> {
    if poses.len() != dimensions.len() || reference >= poses.len() || poses.len() < 2 {
        return None;
    }
    let mut fields = Vec::with_capacity(poses.len());
    let mut offset = 0;
    let mut edges = Vec::new();
    for (station, (pose, &(width, height))) in poses.iter().zip(dimensions).enumerate() {
        let corners = [
            (0.0, 0.0),
            (f64::from(width), 0.0),
            (f64::from(width), f64::from(height)),
            (0.0, f64::from(height)),
        ]
        .into_iter()
        .map(|(x, y)| project(pose, Point2::new(x, y)))
        .collect::<Option<Vec<_>>>()?;
        let low = Point2::new(
            corners.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
            corners.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
        );
        let high = Point2::new(
            corners
                .iter()
                .map(|p| p.x)
                .fold(f64::NEG_INFINITY, f64::max),
            corners
                .iter()
                .map(|p| p.y)
                .fold(f64::NEG_INFINITY, f64::max),
        );
        let origin = Point2::new((low.x / STEP).floor() * STEP, (low.y / STEP).floor() * STEP);
        let columns = ((high.x / STEP).ceil() - origin.x / STEP) as usize + 1;
        let rows = ((high.y / STEP).ceil() - origin.y / STEP) as usize + 1;
        if columns < 2 || rows < 2 || columns.checked_mul(rows)? > 200_000 {
            return None;
        }
        let field = StationField {
            origin,
            columns,
            rows,
            values: vec![[0.0; 2]; columns * rows],
            offset,
        };
        if station != reference {
            for row in 0..rows {
                for column in 0..columns {
                    let index = offset + row * columns + column;
                    if column + 1 < columns {
                        edges.push((index, index + 1));
                    }
                    if row + 1 < rows {
                        edges.push((index, index + columns));
                    }
                }
            }
            offset += columns * rows;
        }
        fields.push(field);
    }
    if offset == 0 || offset > 2_000_000 {
        return None;
    }
    let mut rows = Vec::new();
    for (pair, relation) in relations.iter().enumerate() {
        if relation.left >= poses.len()
            || relation.right >= poses.len()
            || relation.left == relation.right
        {
            return None;
        }
        let mut points = relation.points.clone();
        points.sort_by(|a, b| {
            a.0.x
                .total_cmp(&b.0.x)
                .then_with(|| a.0.y.total_cmp(&b.0.y))
                .then_with(|| a.1.x.total_cmp(&b.1.x))
                .then_with(|| a.1.y.total_cmp(&b.1.y))
        });
        for (index, &(a, b)) in points.iter().enumerate() {
            let a = project(&poses[relation.left], a)?;
            let b = project(&poses[relation.right], b)?;
            let (ai, mut ac) = fields[relation.left].taps(a);
            let (bi, mut bc) = fields[relation.right].taps(b);
            if relation.left == reference {
                ac.fill(0.0);
            }
            if relation.right == reference {
                bc.fill(0.0);
            }
            let mut indices = [0; 8];
            let mut coefficients = [0.0; 8];
            for k in 0..4 {
                indices[k] = if relation.left == reference { 0 } else { ai[k] };
                indices[k + 4] = if relation.right == reference {
                    0
                } else {
                    bi[k]
                };
                coefficients[k] = ac[k];
                coefficients[k + 4] = -bc[k];
            }
            rows.push(Row {
                indices,
                coefficients,
                delta: [b.x - a.x, b.y - a.y],
                base_weight: 1.0 / points.len().max(1) as f64,
                training: index % 5 != 0,
                pair,
            });
        }
    }
    if rows.is_empty() {
        return None;
    }
    let mut weights = rows
        .iter()
        .map(|r| if r.training { r.base_weight } else { 0.0 })
        .collect::<Vec<_>>();
    let mut coordinates = [vec![0.0; offset], vec![0.0; offset]];
    for _ in 0..8 {
        for axis in 0..2 {
            solve_axis(&rows, &weights, &edges, axis, &mut coordinates[axis])?;
        }
        for (row, weight) in rows.iter().zip(&mut weights) {
            let error = (prediction(row, &coordinates[0]) - row.delta[0])
                .hypot(prediction(row, &coordinates[1]) - row.delta[1]);
            *weight = if row.training {
                row.base_weight * (3.0 / error.max(1e-8)).min(1.0)
            } else {
                0.0
            };
        }
    }
    for (station, field) in fields.iter_mut().enumerate() {
        if station == reference {
            continue;
        }
        for (index, value) in field.values.iter_mut().enumerate() {
            *value = [
                coordinates[0][field.offset + index],
                coordinates[1][field.offset + index],
            ];
        }
    }
    let mut validation = Vec::with_capacity(relations.len());
    for (pair, relation) in relations.iter().enumerate() {
        let mut errors = rows
            .iter()
            .filter(|r| r.pair == pair && !r.training)
            .map(|r| {
                (prediction(r, &coordinates[0]) - r.delta[0])
                    .hypot(prediction(r, &coordinates[1]) - r.delta[1])
            })
            .collect::<Vec<_>>();
        if errors.is_empty() {
            return None;
        }
        errors.sort_by(f64::total_cmp);
        validation.push(PairValidation {
            left: relation.left,
            right: relation.right,
            count: errors.len(),
            median: errors[errors.len() / 2],
            p95: errors[(errors.len() * 95).div_ceil(100) - 1],
        });
    }
    Some(Fit { fields, validation })
}

#[cfg(test)]
mod tests;
