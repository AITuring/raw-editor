//! Transform estimation and robust-fit primitives.
//!
//! Translation/similarity/affine estimators, symmetric point errors, the robust
//! transform fit, union-find and the small matrix/median helpers shared by the
//! matching, ordering and pose stages.

use crate::panorama_utils::processing;
use crate::panorama_utils::stack_pipeline::determinism;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;
use rand::prelude::*;

use super::FOCUS_MODEL_INLIER_THRESHOLD;

pub(super) const FOCUS_MODEL_RANSAC_ITERATIONS: usize = 1_500;

pub(super) fn estimate_translation(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> Matrix3<f64> {
    let mut dx: Vec<f64> = points.iter().map(|(a, b)| b.x - a.x).collect();
    let mut dy: Vec<f64> = points.iter().map(|(a, b)| b.y - a.y).collect();
    // `partial_cmp(...).unwrap_or(Equal)` is not a valid sort order when a
    // degenerate homography produces NaN: NaN compares equal to every finite
    // value but finite values still compare against one another.  Use the
    // IEEE total order so a real mixed-focal stack cannot panic in the
    // geometry fallback.
    dx.sort_by(f64::total_cmp);
    dy.sort_by(f64::total_cmp);
    let median = |values: &[f64]| values[values.len() / 2];
    Matrix3::new(1.0, 0.0, median(&dx), 0.0, 1.0, median(&dy), 0.0, 0.0, 1.0)
}

pub(super) fn estimate_similarity(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> Option<Matrix3<f64>> {
    if points.len() < 2 {
        return None;
    }
    let count = points.len() as f64;
    let source_center = points
        .iter()
        .fold(nalgebra::Vector2::zeros(), |sum, (source, _)| {
            sum + source.coords
        })
        / count;
    let target_center = points
        .iter()
        .fold(nalgebra::Vector2::zeros(), |sum, (_, target)| {
            sum + target.coords
        })
        / count;

    let mut denominator = 0.0;
    let mut dot = 0.0;
    let mut cross = 0.0;
    for (source, target) in points {
        let source_delta = source.coords - source_center;
        let target_delta = target.coords - target_center;
        denominator += source_delta.norm_squared();
        dot += source_delta.x * target_delta.x + source_delta.y * target_delta.y;
        cross += source_delta.x * target_delta.y - source_delta.y * target_delta.x;
    }
    if denominator <= f64::EPSILON {
        return None;
    }
    let a = dot / denominator;
    let b = cross / denominator;
    let translation_x = target_center.x - a * source_center.x + b * source_center.y;
    let translation_y = target_center.y - b * source_center.x - a * source_center.y;
    let transform = Matrix3::new(a, -b, translation_x, b, a, translation_y, 0.0, 0.0, 1.0);
    transform
        .iter()
        .all(|value| value.is_finite())
        .then_some(transform)
}

pub(super) fn estimate_affine(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> Option<Matrix3<f64>> {
    if points.len() < 3 {
        return None;
    }
    let mut design_data = Vec::with_capacity(points.len() * 3);
    let mut target_x = Vec::with_capacity(points.len());
    let mut target_y = Vec::with_capacity(points.len());
    for (source, target) in points {
        design_data.extend_from_slice(&[source.x, source.y, 1.0]);
        target_x.push(target.x);
        target_y.push(target.y);
    }
    let design = nalgebra::DMatrix::from_row_slice(points.len(), 3, &design_data);
    let decomposition = design.svd(true, true);
    let x_solution = decomposition
        .solve(&nalgebra::DVector::from_vec(target_x), 1e-10)
        .ok()?;
    let y_solution = decomposition
        .solve(&nalgebra::DVector::from_vec(target_y), 1e-10)
        .ok()?;
    let transform = Matrix3::new(
        x_solution[0],
        x_solution[1],
        x_solution[2],
        y_solution[0],
        y_solution[1],
        y_solution[2],
        0.0,
        0.0,
        1.0,
    );
    transform
        .iter()
        .all(|value| value.is_finite())
        .then_some(transform)
}

pub(super) fn transformed_point(
    transform: &Matrix3<f64>,
    point: nalgebra::Point2<f64>,
) -> Option<nalgebra::Point2<f64>> {
    let mapped = transform * nalgebra::Point3::new(point.x, point.y, 1.0);
    if mapped.z.abs() < 1e-8 {
        return None;
    }
    let mapped = nalgebra::Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
    (mapped.x.is_finite() && mapped.y.is_finite()).then_some(mapped)
}

pub(super) fn homography_preserves_focus_orientation(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    if width <= 1.0 || height <= 1.0 {
        return false;
    }
    let source_corners = [
        Point2::new(0.0, 0.0),
        Point2::new(width, 0.0),
        Point2::new(width, height),
        Point2::new(0.0, height),
    ];
    let mut mapped_corners = Vec::with_capacity(source_corners.len());
    let mut depth_sign = 0.0;
    for corner in source_corners {
        let mapped = transform * Point3::new(corner.x, corner.y, 1.0);
        if !mapped.iter().all(|value| value.is_finite()) || mapped.z.abs() < 1e-8 {
            return false;
        }
        let sign = mapped.z.signum();
        if depth_sign == 0.0 {
            depth_sign = sign;
        } else if sign != depth_sign {
            // A corner crossing the projective horizon can fold the image even
            // when the matrix is numerically invertible. Such a pose is never
            // a valid focus-stack registration.
            return false;
        }
        mapped_corners.push(Point2::new(mapped.x / mapped.z, mapped.y / mapped.z));
    }
    let signed_double_area = mapped_corners
        .iter()
        .zip(mapped_corners.iter().cycle().skip(1))
        .take(4)
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>();
    signed_double_area.is_finite() && signed_double_area > 0.0
}

pub(super) fn transform_is_stable_for_focus_stack(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    if width <= 1.0 || height <= 1.0 || transform.try_inverse().is_none() {
        return false;
    }
    let source_corners = [
        nalgebra::Point2::new(0.0, 0.0),
        nalgebra::Point2::new(width, 0.0),
        nalgebra::Point2::new(width, height),
        nalgebra::Point2::new(0.0, height),
    ];
    let Some(corners) = source_corners
        .into_iter()
        .map(|point| transformed_point(transform, point))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };

    let top = (corners[1] - corners[0]).norm() / width;
    let right = (corners[2] - corners[1]).norm() / height;
    let bottom = (corners[2] - corners[3]).norm() / width;
    let left = (corners[3] - corners[0]).norm() / height;
    let edge_scales = [top, right, bottom, left];
    if edge_scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale < 0.65 || *scale > 1.45)
    {
        return false;
    }
    let min_scale = edge_scales.iter().copied().fold(f64::INFINITY, f64::min);
    let max_scale = edge_scales
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_scale / min_scale > 1.18 {
        return false;
    }

    let top_vector = corners[1] - corners[0];
    let left_vector = corners[3] - corners[0];
    let orthogonality = top_vector.dot(&left_vector) / (top_vector.norm() * left_vector.norm());
    if !orthogonality.is_finite() || orthogonality.abs() > 0.22 {
        return false;
    }

    let signed_double_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .map(|(a, b)| a.x * b.y - b.x * a.y)
        .sum::<f64>();
    let area_ratio = signed_double_area / (2.0 * width * height);
    area_ratio.is_finite() && (0.5..=2.0).contains(&area_ratio)
}

pub(super) fn median_symmetric_error(
    transform: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> f64 {
    let Some(inverse) = transform.try_inverse() else {
        return f64::INFINITY;
    };
    let mut errors: Vec<f64> = points
        .iter()
        .map(|(source, target)| symmetric_point_error(transform, &inverse, *source, *target))
        .filter(|error| error.is_finite())
        .collect();
    if errors.is_empty() {
        return f64::INFINITY;
    }
    errors.sort_by(f64::total_cmp);
    errors[errors.len() / 2]
}

#[derive(Debug, Clone)]
pub(super) struct RobustTransformFit {
    pub(super) transform: Matrix3<f64>,
    pub(super) inlier_indices: Vec<usize>,
    pub(super) median_error: f64,
}

pub(super) fn symmetric_inlier_indices(
    transform: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    threshold: f64,
) -> Vec<usize> {
    let Some(inverse) = transform.try_inverse() else {
        return Vec::new();
    };
    points
        .iter()
        .enumerate()
        .filter_map(|(index, (source, target))| {
            (symmetric_point_error(transform, &inverse, *source, *target) <= threshold)
                .then_some(index)
        })
        .collect()
}

pub(super) fn robust_transform_fit<F>(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    sample_size: usize,
    seed: u64,
    estimator: F,
) -> Option<RobustTransformFit>
where
    F: Fn(&[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)]) -> Option<Matrix3<f64>>,
{
    robust_transform_fit_with_threshold(
        points,
        sample_size,
        seed,
        FOCUS_MODEL_INLIER_THRESHOLD,
        estimator,
    )
}

/// [`robust_transform_fit`] with a caller-chosen symmetric inlier threshold.
pub(super) fn robust_transform_fit_with_threshold<F>(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    sample_size: usize,
    seed: u64,
    inlier_threshold: f64,
    estimator: F,
) -> Option<RobustTransformFit>
where
    F: Fn(&[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)]) -> Option<Matrix3<f64>>,
{
    if points.len() < sample_size {
        return None;
    }

    // `seed` is the caller's site constant; the run seed below is derived from
    // the input content (需求 14.6).
    let mut rng = StdRng::seed_from_u64(
        seed ^ (points.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ determinism::run_random_seed_mix(),
    );
    let all_indices: Vec<usize> = (0..points.len()).collect();
    let mut best_transform = None;
    let mut best_inliers = Vec::new();
    let mut best_median = f64::INFINITY;

    let iterations = if points.len() == sample_size {
        1
    } else {
        FOCUS_MODEL_RANSAC_ITERATIONS
    };
    for _ in 0..iterations {
        let sample_indices: Vec<usize> =
            all_indices.sample(&mut rng, sample_size).copied().collect();
        if sample_indices.len() != sample_size {
            continue;
        }
        let sample: Vec<_> = sample_indices.iter().map(|&index| points[index]).collect();
        let Some(transform) = estimator(&sample) else {
            continue;
        };
        let inlier_indices = symmetric_inlier_indices(&transform, points, inlier_threshold);
        if inlier_indices.len() < sample_size {
            continue;
        }
        let inlier_points: Vec<_> = inlier_indices.iter().map(|&index| points[index]).collect();
        let median = median_symmetric_error(&transform, &inlier_points);
        if inlier_indices.len() > best_inliers.len()
            || (inlier_indices.len() == best_inliers.len() && median < best_median)
        {
            best_transform = Some(transform);
            best_inliers = inlier_indices;
            best_median = median;
        }
    }

    let mut transform = best_transform?;
    let mut inlier_indices = best_inliers;
    for _ in 0..4 {
        let inlier_points: Vec<_> = inlier_indices.iter().map(|&index| points[index]).collect();
        let Some(refitted) = estimator(&inlier_points) else {
            break;
        };
        let refitted_inliers = symmetric_inlier_indices(&refitted, points, inlier_threshold);
        if refitted_inliers.len() < sample_size {
            break;
        }
        transform = refitted;
        if refitted_inliers == inlier_indices {
            break;
        }
        inlier_indices = refitted_inliers;
    }

    let inlier_points: Vec<_> = inlier_indices.iter().map(|&index| points[index]).collect();
    Some(RobustTransformFit {
        transform,
        inlier_indices,
        median_error: median_symmetric_error(&transform, &inlier_points),
    })
}

pub(super) fn refine_homography_inliers(
    points: &mut Vec<(nalgebra::Point2<f64>, nalgebra::Point2<f64>)>,
    refinement_threshold: f64,
    minimum_inliers: usize,
) -> Option<Matrix3<f64>> {
    if points.len() < minimum_inliers {
        return None;
    }

    // Patch correlation can occasionally lock onto a nearby repeated stroke or
    // texture. Starting the least-squares refinement from every correlation lets
    // a coherent minority bend a projective model enough that none of those bad
    // points exceeds the later residual threshold. Re-run a deterministic robust
    // fit at full resolution before optimizing all remaining correspondences.
    let (_, robust_inlier_indices) =
        processing::find_homography_ransac_points_stable_with_min_inliers(
            points,
            refinement_threshold,
            minimum_inliers,
        )?;
    if robust_inlier_indices.len() < minimum_inliers {
        return None;
    }
    if robust_inlier_indices.len() < points.len() {
        *points = robust_inlier_indices
            .into_iter()
            .map(|index| points[index])
            .collect();
    }

    let mut homography = processing::compute_homography(points)?;
    for _ in 0..3 {
        let Some(inverse) = homography.try_inverse() else {
            break;
        };
        let refined_points: Vec<_> = points
            .iter()
            .copied()
            .filter(|(source, target)| {
                symmetric_point_error(&homography, &inverse, *source, *target)
                    <= refinement_threshold
            })
            .collect();
        if refined_points.len() < minimum_inliers || refined_points.len() == points.len() {
            break;
        }
        *points = refined_points;
        homography = processing::compute_homography(points)?;
    }
    Some(homography)
}

pub(super) fn retain_model_inliers(
    points: &mut Vec<(nalgebra::Point2<f64>, nalgebra::Point2<f64>)>,
    transform: &Matrix3<f64>,
    threshold: f64,
    minimum_inliers: usize,
) -> bool {
    let Some(inverse) = transform.try_inverse() else {
        return false;
    };
    let filtered = points
        .iter()
        .copied()
        .filter(|(source, target)| {
            symmetric_point_error(transform, &inverse, *source, *target) <= threshold
        })
        .collect::<Vec<_>>();
    if filtered.len() < minimum_inliers {
        return false;
    }
    *points = filtered;
    true
}

pub(super) fn symmetric_point_error(
    homography: &Matrix3<f64>,
    inverse: &Matrix3<f64>,
    source: nalgebra::Point2<f64>,
    target: nalgebra::Point2<f64>,
) -> f64 {
    let forward = homography * nalgebra::Point3::new(source.x, source.y, 1.0);
    let reverse = inverse * nalgebra::Point3::new(target.x, target.y, 1.0);
    if forward.z.abs() < 1e-8 || reverse.z.abs() < 1e-8 {
        return f64::INFINITY;
    }
    let forward_point = nalgebra::Point2::new(forward.x / forward.z, forward.y / forward.z);
    let reverse_point = nalgebra::Point2::new(reverse.x / reverse.z, reverse.y / reverse.z);
    ((forward_point - target).norm_squared() + (reverse_point - source).norm_squared()).sqrt()
}

pub(super) fn symmetric_reprojection_rmse(
    homography: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> f64 {
    let Some(inverse) = homography.try_inverse() else {
        return f64::INFINITY;
    };
    let sum_squared_error = points
        .iter()
        .filter_map(|(source, target)| {
            let forward = homography * nalgebra::Point3::new(source.x, source.y, 1.0);
            let reverse = inverse * nalgebra::Point3::new(target.x, target.y, 1.0);
            if forward.z.abs() < 1e-8 || reverse.z.abs() < 1e-8 {
                return None;
            }
            let forward_point = nalgebra::Point2::new(forward.x / forward.z, forward.y / forward.z);
            let reverse_point = nalgebra::Point2::new(reverse.x / reverse.z, reverse.y / reverse.z);
            Some((forward_point - target).norm_squared() + (reverse_point - source).norm_squared())
        })
        .sum::<f64>();
    if points.is_empty() {
        f64::INFINITY
    } else {
        (sum_squared_error / (points.len() as f64 * 2.0)).sqrt()
    }
}

pub(super) struct Dsu {
    pub(super) parent: Vec<usize>,
}

impl Dsu {
    pub(super) fn new(n: usize) -> Self {
        Dsu {
            parent: (0..n).collect(),
        }
    }

    pub(super) fn find(&mut self, i: usize) -> usize {
        if self.parent[i] == i {
            i
        } else {
            self.parent[i] = self.find(self.parent[i]);
            self.parent[i]
        }
    }

    pub(super) fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            self.parent[root_i] = root_j;
        }
    }
}

pub(super) fn normalized_homography(transform: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let scale = transform[(2, 2)];
    if !scale.is_finite() || scale.abs() <= f64::EPSILON {
        return None;
    }
    let normalized = transform / scale;
    normalized
        .iter()
        .all(|value| value.is_finite())
        .then_some(normalized)
}

pub(super) fn focus_transform_disagreement_px(
    left: &Matrix3<f64>,
    right: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> Option<f64> {
    let (width, height) = dimensions;
    let samples = [
        Point2::new(width as f64 * 0.2, height as f64 * 0.2),
        Point2::new(width as f64 * 0.8, height as f64 * 0.2),
        Point2::new(width as f64 * 0.5, height as f64 * 0.5),
        Point2::new(width as f64 * 0.2, height as f64 * 0.8),
        Point2::new(width as f64 * 0.8, height as f64 * 0.8),
    ];
    let mut errors = samples
        .into_iter()
        .filter_map(|point| {
            let left = transformed_point(left, point)?;
            let right = transformed_point(right, point)?;
            let error = (left - right).norm();
            error.is_finite().then_some(error)
        })
        .collect::<Vec<_>>();
    median_value(&mut errors)
}

pub(super) fn matrix3_row_major(transform: &Matrix3<f64>) -> [f64; 9] {
    [
        transform[(0, 0)],
        transform[(0, 1)],
        transform[(0, 2)],
        transform[(1, 0)],
        transform[(1, 1)],
        transform[(1, 2)],
        transform[(2, 0)],
        transform[(2, 1)],
        transform[(2, 2)],
    ]
}

pub(super) fn median_value(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}
