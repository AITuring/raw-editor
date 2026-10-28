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
mod tests {
    use super::*;

    #[test]
    fn density_gauge_uses_a_feasible_common_pivot_without_relaxing_corner_limits() {
        let initial = Matrix3::new(0.96, 0.0, 0.0, 0.0, 0.96, 0.0, 0.0, 0.0, 1.0);
        let pose = Matrix3::new(0.96, 0.0, 100.0, 0.0, 0.96, 0.0, 0.0, 0.0, 1.0);
        let result = native_density_gauge(
            Decomposition {
                poses: vec![pose],
                gauge: Matrix3::identity(),
                maximum_corner_motion: 100.0,
            },
            &[(1024, 1024)],
            &[initial],
            &[112.0],
        )
        .unwrap();
        assert!(result.poses[0].determinant().sqrt() > 1.0 - 1e-7);
        assert!(result.maximum_corner_motion <= 112.0 + 1e-7);
        for p in corners((1024, 1024)) {
            assert!(
                (project(&result.poses[0], p).unwrap() - project(&initial, p).unwrap()).norm()
                    <= 112.0 + 1e-7
            );
        }
    }

    #[test]
    fn density_gauge_recovers_pixels_without_exceeding_camera_budgets() {
        let pose = Matrix3::new(0.96, 0.0, 0.0, 0.0, 0.96, 0.0, 0.0, 0.0, 1.0);
        let run = |limit| {
            native_density_gauge(
                Decomposition {
                    poses: vec![pose],
                    gauge: Matrix3::identity(),
                    maximum_corner_motion: 0.0,
                },
                &[(1024, 1024)],
                &[pose],
                &[limit],
            )
            .unwrap()
        };
        let recovered = run(32.0);
        assert!(recovered.poses[0].determinant().sqrt() >= 0.98 - 1e-7);
        assert!(recovered.maximum_corner_motion <= 32.0);
        let limited = run(2.0);
        assert!(limited.poses[0].determinant().sqrt() > 0.96);
        assert!(limited.poses[0].determinant().sqrt() < 0.98);
        assert!(limited.maximum_corner_motion <= 2.0);
        for point in corners((1024, 1024)) {
            assert!(
                (project(&limited.poses[0], point).unwrap() - project(&pose, point).unwrap())
                    .norm()
                    <= 2.0
            );
        }
    }
    #[test]
    fn corner_budget_can_move_into_a_bounded_local_field_but_cannot_be_forced() {
        let proposal = [Matrix3::identity(); 2];
        let initial = [
            Matrix3::identity(),
            Matrix3::new(1.06, 0.0, 0.0, 0.0, 1.06, 0.0, 0.0, 0.0, 1.0),
        ];
        let points = (128..896)
            .step_by(64)
            .flat_map(|x| {
                (128..896).step_by(64).map(move |y| {
                    let p = Point2::new(x as f64, y as f64);
                    (p, p)
                })
            })
            .collect::<Vec<_>>();
        let relations = [Relation {
            left: 0,
            right: 1,
            points,
        }];
        let fit = fit(&proposal, &[(1024, 1024); 2], &relations, 0).unwrap();
        let verified = (0..2)
            .map(|_| {
                relations[0]
                    .points
                    .iter()
                    .map(|&(point, _)| VerifiedSource {
                        point,
                        round_trip_error_px: 0.2,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let decomposed = decompose(
            &proposal,
            &fit.fields,
            &[(1024, 1024); 2],
            &initial,
            &[24.0; 2],
        );
        assert!(
            decomposed.is_some(),
            "the bounded camera/local decomposition is feasible"
        );
        let candidate = materialize(
            &proposal,
            &fit,
            &[(1024, 1024); 2],
            &initial,
            &[24.0; 2],
            &relations,
            &verified,
        )
        .unwrap();
        assert!(candidate.maximum_corner_motion <= 24.0);
        assert!(!candidate.residual.identity());
        assert!(
            candidate
                .residual
                .regions
                .iter()
                .all(|r| r.max_node_displacement_px <= 32.0)
        );
        assert!(
            materialize(
                &proposal,
                &fit,
                &[(1024, 1024); 2],
                &initial,
                &[1.0; 2],
                &relations,
                &verified
            )
            .is_none(),
            "an infeasible budget must still reject the model"
        );
    }
    #[test]
    fn guarded_model_requires_native_evidence_and_rechecks_actual_sampling() {
        let poses = [Matrix3::identity(); 2];
        let mut points = Vec::new();
        for x in (32..992).step_by(32) {
            for y in (32..992).step_by(32) {
                let a = Point2::new(x as f64, y as f64);
                points.push((a, Point2::new(a.x + 12.0 * (a.y / 150.0).sin(), a.y)));
            }
        }
        let relation = Relation {
            left: 0,
            right: 1,
            points,
        };
        let relations = [relation];
        let fields = fit(&poses, &[(1024, 1024); 2], &relations, 0).unwrap();
        assert!(
            materialize(
                &poses,
                &fields,
                &[(1024, 1024); 2],
                &poses,
                &[256.0; 2],
                &relations,
                &[Vec::new(), Vec::new()]
            )
            .is_none()
        );
        let verified = (0..2)
            .map(|station| {
                relations[0]
                    .points
                    .iter()
                    .map(|&(a, b)| VerifiedSource {
                        point: if station == 0 { a } else { b },
                        round_trip_error_px: 0.4,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let bounded = materialize(
            &poses,
            &fields,
            &[(1024, 1024); 2],
            &poses,
            &[256.0; 2],
            &relations,
            &verified,
        )
        .unwrap();
        assert!(!bounded.residual.identity());
        assert!(bounded.validation[0].p95.is_finite());
        // Boundary fading is a real change to the fitted mapping. Its errors
        // must be measured instead of copying the unguarded fit statistics.
        assert!((bounded.validation[0].p95 - fields.validation[0].p95).abs() > 0.01);
        for region in &bounded.residual.regions {
            assert!(region.max_node_displacement_px <= 32.0);
            assert!(region.max_neighbour_delta_px <= 8.0 + 1e-8);
            assert_eq!(
                region.displacement_at(
                    region.world.left,
                    region.world.top + region.world.height / 2.0
                ),
                [0.0; 2]
            );
        }
        let invalid = verified
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|v| VerifiedSource {
                        round_trip_error_px: 1.1,
                        ..*v
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(
            materialize(
                &poses,
                &fields,
                &[(1024, 1024); 2],
                &poses,
                &[256.0; 2],
                &relations,
                &invalid
            )
            .is_none()
        );
    }
    #[test]
    fn projective_component_recovers_camera_motion_without_moving_the_reference() {
        let initial = vec![
            Matrix3::identity(),
            Matrix3::new(
                1.003, 0.002, 317.0, -0.001, 0.997, 9.0, 0.000002, -0.000001, 1.0,
            ),
            Matrix3::new(
                0.998, -0.003, 583.0, 0.001, 1.003, -7.0, -0.000001, 0.000002, 1.0,
            ),
        ];
        let mut relations = Vec::new();
        for (left, right) in [(0, 1), (1, 2), (0, 2)] {
            let mut points = Vec::new();
            for y in (80..920).step_by(40) {
                for x in (640..960).step_by(40) {
                    points.push((
                        Point2::new(x as f64 - left as f64 * 300.0, y as f64),
                        Point2::new(x as f64 - right as f64 * 300.0, y as f64),
                    ));
                }
            }
            relations.push(Relation {
                left,
                right,
                points,
            });
        }
        let result = projective_proposal(&initial, &relations, 0, 1024.0).unwrap();
        assert_eq!(result[0], initial[0]);
        for relation in &relations {
            for &(a, b) in &relation.points {
                let error = (project(&result[relation.left], a).unwrap()
                    - project(&result[relation.right], b).unwrap())
                .norm();
                assert!(error < 0.05, "camera motion did not converge: {error}");
            }
        }
        let fields = fit(&result, &[(1024, 1024); 3], &relations, 0).unwrap();
        let decomposed = decompose(
            &result,
            &fields.fields,
            &[(1024, 1024); 3],
            &initial,
            &[256.0; 3],
        )
        .unwrap();
        assert!(decomposed.maximum_corner_motion < 40.0);
        assert!(decomposed.poses.iter().all(|h| h.try_inverse().is_some()));
    }
    #[test]
    #[ignore = "requires saved native station correspondences and a diagnostic pose set"]
    fn real_joint_station_field_diagnostics_from_env() {
        let input = std::env::var("RAW_EDITOR_JOINT_WARP_INPUT").unwrap();
        let output = std::env::var("RAW_EDITOR_JOINT_WARP_OUTPUT").unwrap();
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
        let mut poses = data["poses"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| Matrix3::from_fn(|row, column| p[row][column].as_f64().unwrap()))
            .collect::<Vec<_>>();
        let dimensions = data["dimensions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| (d[0].as_u64().unwrap() as u32, d[1].as_u64().unwrap() as u32))
            .collect::<Vec<_>>();
        let relations = data["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| Relation {
                left: r["left"].as_u64().unwrap() as usize,
                right: r["right"].as_u64().unwrap() as usize,
                points: r["points"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| {
                        (
                            Point2::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()),
                            Point2::new(p[2].as_f64().unwrap(), p[3].as_f64().unwrap()),
                        )
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let initial = poses.clone();
        if let Some(proposal) = data["proposal_poses"].as_array() {
            poses = proposal
                .iter()
                .map(|p| Matrix3::from_fn(|row, column| p[row][column].as_f64().unwrap()))
                .collect();
        } else if std::env::var_os("RAW_EDITOR_JOINT_WARP_OPTIMIZE").is_some() {
            poses = projective_proposal(&poses, &relations, 0, 9000.0).unwrap();
        }
        let result = fit(&poses, &dimensions, &relations, 0).unwrap();
        let bounded =
            if let Some(limits) = data["limits"].as_array() {
                let limits = limits
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>();
                let mut verified = vec![Vec::new(); poses.len()];
                for record in data["records"].as_array().unwrap() {
                    let (left, right) = (
                        record["left"].as_u64().unwrap() as usize,
                        record["right"].as_u64().unwrap() as usize,
                    );
                    if let Some(points) = record["native_verified_points"].as_array() {
                        for point in points {
                            let error = point[4].as_f64().unwrap();
                            verified[left].push(VerifiedSource {
                                point: Point2::new(
                                    point[0].as_f64().unwrap(),
                                    point[1].as_f64().unwrap(),
                                ),
                                round_trip_error_px: error,
                            });
                            verified[right].push(VerifiedSource {
                                point: Point2::new(
                                    point[2].as_f64().unwrap(),
                                    point[3].as_f64().unwrap(),
                                ),
                                round_trip_error_px: error,
                            });
                        }
                    }
                }
                materialize(&poses, &result, &dimensions, &initial, &limits, &relations, &verified)
                .map(|bounded| serde_json::json!({
                    "max_corner": bounded.maximum_corner_motion,
                    "regions": bounded.residual.report_records(),
                    "validation": bounded.validation.iter().map(|v| serde_json::json!({
                        "left":v.left,"right":v.right,"count":v.count,"median":v.median,"p95":v.p95,
                    })).collect::<Vec<_>>(),
                }))
            } else {
                None
            };
        let fields = result.fields.iter().map(|f| serde_json::json!({
            "origin":[f.origin.x,f.origin.y],"columns":f.columns,"rows":f.rows,"values":f.values,
        })).collect::<Vec<_>>();
        let validation = result
            .validation
            .iter()
            .map(|v| {
                serde_json::json!({
                    "left":v.left,"right":v.right,"count":v.count,"median":v.median,"p95":v.p95,
                })
            })
            .collect::<Vec<_>>();
        let max_node = result
            .fields
            .iter()
            .flat_map(|f| &f.values)
            .map(|v| v[0].hypot(v[1]))
            .fold(0.0, f64::max);
        println!(
            "Joint station diagnostic: {} relations, max unbounded node {max_node:.3}px",
            relations.len()
        );
        for v in &result.validation {
            println!(
                "PAIR {} {} count={} median={:.3} p95={:.3}",
                v.left, v.right, v.count, v.median, v.p95
            );
        }
        std::fs::write(
            output,
            serde_json::to_vec(&serde_json::json!({
                "fields":fields,"validation":validation,"max_node":max_node,"bounded":bounded,
                "poses":poses.iter().map(|h| (0..3).map(|r| (0..3).map(|c| h[(r,c)]).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>(),
            }))
            .unwrap(),
        )
        .unwrap();
    }
    #[test]
    fn validation_observations_cannot_be_fitted_away() {
        let poses = [Matrix3::identity(); 2];
        let mut points = Vec::new();
        for x in (64..960).step_by(32) {
            for y in (64..960).step_by(32) {
                let point = Point2::new(x as f64, y as f64);
                let offset = if points.len() % 5 == 0 { 15.0 } else { 0.0 };
                points.push((point, Point2::new(point.x + offset, point.y)));
            }
        }
        let result = fit(
            &poses,
            &[(1024, 1024); 2],
            &[Relation {
                left: 0,
                right: 1,
                points,
            }],
            0,
        )
        .unwrap();
        assert!(
            result.validation[0].p95 >= 14.9,
            "held-out errors were hidden by fitting: {:?}",
            result.validation
        );
        assert!(
            result
                .fields
                .iter()
                .all(|f| f.values.iter().all(|v| *v == [0.0; 2]))
        );
    }
    #[test]
    fn joint_fit_uses_validation_points_and_recovers_a_smooth_nonrigid_overlap() {
        let poses = [Matrix3::identity(); 2];
        let mut points = Vec::new();
        for y in (64..960).step_by(32) {
            for x in (64..960).step_by(32) {
                let a = Point2::new(x as f64, y as f64);
                let delta = 2.0 * (a.y / 300.0).sin();
                points.push((a, Point2::new(a.x + delta, a.y)));
            }
        }
        let fit = fit(
            &poses,
            &[(1024, 1024); 2],
            &[Relation {
                left: 0,
                right: 1,
                points,
            }],
            0,
        )
        .unwrap();
        assert!(fit.validation[0].count > 100);
        assert!(
            fit.validation[0].p95 < 0.5,
            "validation: {:?}",
            fit.validation
        );
        assert!(fit.fields[0].values.iter().all(|v| *v == [0.0; 2]));
        assert!(fit.fields[1].at(Point2::new(512.0, 256.0))[0].abs() > 1.0);
    }
}
