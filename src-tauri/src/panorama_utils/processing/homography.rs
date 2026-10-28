//! Robust homography estimation.
//!
//! RANSAC over descriptor matches or point correspondences with
//! content-derived seeding, the normalised four-point and least-squares DLT
//! solvers and the collinearity guard.

use crate::panorama_utils::shared::KeyPoint;
use crate::panorama_utils::shared::Match;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::SVD;
use rand::prelude::*;
use rayon::prelude::*;

use super::{
    MIN_INLIERS_FOR_CONNECTION, RANSAC_INLIER_MIN_POINTS_PER_TASK, RANSAC_INLIER_THRESHOLD,
    RANSAC_ITERATIONS,
};

pub fn find_homography_ransac(
    matches: &[Match],
    keypoints1: &[KeyPoint],
    keypoints2: &[KeyPoint],
) -> Option<(Matrix3<f64>, Vec<Match>)> {
    let points: Vec<(Point2<f64>, Point2<f64>)> = matches
        .iter()
        .map(|m| {
            let p1 = keypoints1[m.index1];
            let p2 = keypoints2[m.index2];
            (
                Point2::new(p1.x as f64, p1.y as f64),
                Point2::new(p2.x as f64, p2.y as f64),
            )
        })
        .collect();

    let (homography, inlier_indices) =
        find_homography_ransac_points(&points, RANSAC_INLIER_THRESHOLD)?;
    let inliers = inlier_indices
        .into_iter()
        .map(|index| matches[index])
        .collect();
    Some((homography, inliers))
}

pub fn find_homography_ransac_points(
    points: &[(Point2<f64>, Point2<f64>)],
    inlier_threshold: f64,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    find_homography_ransac_points_with_solver(
        points,
        inlier_threshold,
        false,
        MIN_INLIERS_FOR_CONNECTION,
    )
}

pub fn find_homography_ransac_points_stable(
    points: &[(Point2<f64>, Point2<f64>)],
    inlier_threshold: f64,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    // Solve the normalized four-point system directly. This keeps RANSAC's
    // minimum sample size while avoiding the missing null-space row in
    // nalgebra's thin 8x9 SVD. The legacy solver remains above so established
    // <=30-image output bytes do not change.
    find_homography_ransac_points_stable_with_min_inliers(
        points,
        inlier_threshold,
        MIN_INLIERS_FOR_CONNECTION,
    )
}

pub fn find_homography_ransac_points_stable_with_min_inliers(
    points: &[(Point2<f64>, Point2<f64>)],
    inlier_threshold: f64,
    minimum_inliers: usize,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    find_homography_ransac_points_with_solver(
        points,
        inlier_threshold,
        true,
        minimum_inliers.max(4),
    )
}

pub(super) fn find_homography_ransac_points_with_solver(
    points: &[(Point2<f64>, Point2<f64>)],
    inlier_threshold: f64,
    stable_four_point_solver: bool,
    minimum_inliers: usize,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    // Site constant mixed with the run seed derived from the input content
    // (需求 14.6). No wall clock and no thread id take part: with the run seed
    // absent the expression collapses to the site constant, which is what the
    // unit tests and the non focus stack path run with.
    let mut rng = StdRng::seed_from_u64(
        0x9E37_79B9_7F4A_7C15u64
            ^ (points.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ crate::panorama_utils::stack_pipeline::determinism::run_random_seed_mix(),
    );
    let mut best_h: Option<Matrix3<f64>> = None;
    let mut best_inliers: Vec<usize> = Vec::new();

    if points.len() < 4 {
        return None;
    }

    let ransac_inlier_threshold_sq = inlier_threshold.powi(2);

    for _ in 0..RANSAC_ITERATIONS {
        let sample_indices: Vec<usize> = (0..points.len()).collect();
        let sample_indices = sample_indices
            .sample(&mut rng, 4)
            .cloned()
            .collect::<Vec<_>>();
        if sample_indices.len() < 4 {
            continue;
        }

        let sample_points: Vec<(Point2<f64>, Point2<f64>)> =
            sample_indices.iter().map(|&i| points[i]).collect();

        if are_points_collinear(sample_points[0].0, sample_points[1].0, sample_points[2].0)
            || are_points_collinear(sample_points[0].0, sample_points[1].0, sample_points[3].0)
            || are_points_collinear(sample_points[0].0, sample_points[2].0, sample_points[3].0)
            || are_points_collinear(sample_points[1].0, sample_points[2].0, sample_points[3].0)
        {
            continue;
        }

        let homography = if stable_four_point_solver {
            compute_homography_four_points(&sample_points)
        } else {
            compute_homography(&sample_points)
        };
        if let Some(h) = homography {
            let Some(h_inverse) = h.try_inverse() else {
                continue;
            };
            let current_inliers: Vec<usize> = points
                .par_iter()
                .enumerate()
                // Scheduling granularity only, exactly like
                // `DETERMINISTIC_SUM_MIN_BLOCKS_PER_TASK`: this loop body runs
                // `RANSAC_ITERATIONS` times per image pair, and splitting a
                // match set of a few dozen points into one `rayon` task per
                // point costs orders of magnitude more in dispatch than the
                // arithmetic it distributes.  `collect` keeps the ascending
                // index order whatever the split shape is, so the inlier set is
                // bit identical for any `min_len` and any thread count.
                .with_min_len(RANSAC_INLIER_MIN_POINTS_PER_TASK)
                .filter_map(|(i, (p1, p2))| {
                    let p1_h = nalgebra::Point3::new(p1.x, p1.y, 1.0);
                    let p2_h_transformed = h * p1_h;
                    if p2_h_transformed.z.abs() < 1e-8 {
                        return None;
                    }
                    let p2_transformed = Point2::new(
                        p2_h_transformed.x / p2_h_transformed.z,
                        p2_h_transformed.y / p2_h_transformed.z,
                    );
                    let dist_sq =
                        (p2.x - p2_transformed.x).powi(2) + (p2.y - p2_transformed.y).powi(2);
                    let p1_h_transformed = h_inverse * nalgebra::Point3::new(p2.x, p2.y, 1.0);
                    if p1_h_transformed.z.abs() < 1e-8 {
                        return None;
                    }
                    let p1_transformed = Point2::new(
                        p1_h_transformed.x / p1_h_transformed.z,
                        p1_h_transformed.y / p1_h_transformed.z,
                    );
                    let reverse_dist_sq =
                        (p1.x - p1_transformed.x).powi(2) + (p1.y - p1_transformed.y).powi(2);
                    if dist_sq < ransac_inlier_threshold_sq
                        && reverse_dist_sq < ransac_inlier_threshold_sq
                    {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();

            if current_inliers.len() > best_inliers.len() {
                best_inliers = current_inliers;
                best_h = Some(h);
            }
        }
    }

    if best_inliers.len() >= minimum_inliers {
        Some((best_h.unwrap(), best_inliers))
    } else {
        None
    }
}

pub(super) fn are_points_collinear(p1: Point2<f64>, p2: Point2<f64>, p3: Point2<f64>) -> bool {
    let area = p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y);
    area.abs() < 1e-6
}

pub(super) fn compute_homography_four_points(
    points: &[(Point2<f64>, Point2<f64>)],
) -> Option<Matrix3<f64>> {
    if points.len() != 4 {
        return None;
    }
    let source_points: Vec<_> = points.iter().map(|(source, _)| *source).collect();
    let target_points: Vec<_> = points.iter().map(|(_, target)| *target).collect();
    let (normalized_source, source_transform) = normalize_points(&source_points)?;
    let (normalized_target, target_transform) = normalize_points(&target_points)?;
    let mut coefficients = nalgebra::SMatrix::<f64, 8, 8>::zeros();
    let mut values = nalgebra::SVector::<f64, 8>::zeros();

    for (index, (source, target)) in normalized_source
        .iter()
        .zip(normalized_target.iter())
        .enumerate()
    {
        let row_x = index * 2;
        coefficients[(row_x, 0)] = source.x;
        coefficients[(row_x, 1)] = source.y;
        coefficients[(row_x, 2)] = 1.0;
        coefficients[(row_x, 6)] = -source.x * target.x;
        coefficients[(row_x, 7)] = -source.y * target.x;
        values[row_x] = target.x;

        let row_y = row_x + 1;
        coefficients[(row_y, 3)] = source.x;
        coefficients[(row_y, 4)] = source.y;
        coefficients[(row_y, 5)] = 1.0;
        coefficients[(row_y, 6)] = -source.x * target.y;
        coefficients[(row_y, 7)] = -source.y * target.y;
        values[row_y] = target.y;
    }

    let solution = coefficients.lu().solve(&values)?;
    let normalized_h = Matrix3::new(
        solution[0],
        solution[1],
        solution[2],
        solution[3],
        solution[4],
        solution[5],
        solution[6],
        solution[7],
        1.0,
    );
    let homography = target_transform.try_inverse()? * normalized_h * source_transform;
    let scale = homography[(2, 2)];
    if scale.abs() < 1e-12 {
        Some(homography)
    } else {
        Some(homography / scale)
    }
}

pub fn compute_homography(points: &[(Point2<f64>, Point2<f64>)]) -> Option<Matrix3<f64>> {
    if points.len() < 4 {
        return None;
    }
    let source_points: Vec<Point2<f64>> = points.iter().map(|(source, _)| *source).collect();
    let target_points: Vec<Point2<f64>> = points.iter().map(|(_, target)| *target).collect();
    let (normalized_source, source_transform) = normalize_points(&source_points)?;
    let (normalized_target, target_transform) = normalize_points(&target_points)?;
    let mut a_rows = Vec::with_capacity(points.len() * 2);
    for (p1, p2) in normalized_source.iter().zip(normalized_target.iter()) {
        let (x, y) = (p1.x, p1.y);
        let (xp, yp) = (p2.x, p2.y);
        a_rows.push(nalgebra::RowDVector::from_vec(vec![
            -x,
            -y,
            -1.0,
            0.0,
            0.0,
            0.0,
            x * xp,
            y * xp,
            xp,
        ]));
        a_rows.push(nalgebra::RowDVector::from_vec(vec![
            0.0,
            0.0,
            0.0,
            -x,
            -y,
            -1.0,
            x * yp,
            y * yp,
            yp,
        ]));
    }
    let a = nalgebra::DMatrix::from_rows(&a_rows);
    let svd = SVD::new(a, true, true);
    let v_t = svd.v_t.expect("SVD failed to compute V_t");
    let h_vec = v_t.row(v_t.nrows() - 1).transpose();
    let normalized_h = Matrix3::from_iterator(h_vec.iter().cloned()).transpose();
    let denormalized_h = target_transform
        .try_inverse()
        .unwrap_or_else(Matrix3::identity)
        * normalized_h
        * source_transform;
    let scale = denormalized_h[(2, 2)];
    if scale.abs() < 1e-12 {
        Some(denormalized_h)
    } else {
        Some(denormalized_h / scale)
    }
}

pub(super) fn normalize_points(points: &[Point2<f64>]) -> Option<(Vec<Point2<f64>>, Matrix3<f64>)> {
    if points.is_empty() {
        return None;
    }
    let centroid = points
        .iter()
        .fold(Point2::new(0.0, 0.0), |sum, point| sum + point.coords)
        / points.len() as f64;
    let mean_distance = points
        .iter()
        .map(|point| (point - centroid).norm())
        .sum::<f64>()
        / points.len() as f64;
    if mean_distance < 1e-12 {
        return None;
    }
    let scale = 2.0f64.sqrt() / mean_distance;
    let transform = Matrix3::new(
        scale,
        0.0,
        -scale * centroid.x,
        0.0,
        scale,
        -scale * centroid.y,
        0.0,
        0.0,
        1.0,
    );
    let normalized = points
        .iter()
        .map(|point| {
            let transformed = transform * nalgebra::Point3::new(point.x, point.y, 1.0);
            Point2::new(transformed.x, transformed.y)
        })
        .collect();
    Some((normalized, transform))
}
