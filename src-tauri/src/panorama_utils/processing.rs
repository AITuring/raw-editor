use crate::panorama_stitching::{BRIEF_DESCRIPTOR_SIZE, Descriptor, Feature, KeyPoint, Match};
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::corners::{Corner, corners_fast9};
use imageproc::filter::gaussian_blur_f32;
use nalgebra::{Matrix3, Point2, SVD};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

const MAX_PROCESSING_DIMENSION: u32 = 2400;
const FAST_THRESHOLD: u8 = 15;
const NON_MAXIMA_SUPPRESSION_RADIUS: f32 = 15.0;
const BRIEF_PATCH_SIZE: u32 = 32;
const MATCH_RATIO_THRESHOLD: f32 = 0.8;
const RANSAC_ITERATIONS: usize = 2500;
const RANSAC_INLIER_THRESHOLD: f64 = 5.0;
const MAX_FEATURES: usize = 2000;
pub const MIN_INLIERS_FOR_CONNECTION: usize = 15;

pub fn calculate_downscale_dimensions(width: u32, height: u32) -> (u32, u32, f64) {
    calculate_downscale_dimensions_capped(width, height, MAX_PROCESSING_DIMENSION)
}

pub fn calculate_downscale_dimensions_capped(
    width: u32,
    height: u32,
    max_dimension: u32,
) -> (u32, u32, f64) {
    assert!(max_dimension > 0, "max_dimension must be positive");
    let long_side = width.max(height);
    if long_side <= max_dimension {
        return (width, height, 1.0);
    }
    let scale_factor = long_side as f64 / max_dimension as f64;
    let new_width = (width as f64 / scale_factor).round() as u32;
    let new_height = (height as f64 / scale_factor).round() as u32;
    (new_width, new_height, scale_factor)
}

pub fn normalize_grayscale(img: &GrayImage) -> GrayImage {
    let mut minimum = u8::MAX;
    let mut maximum = u8::MIN;
    for pixel in img.pixels() {
        let value = pixel[0];
        if value < minimum {
            minimum = value;
        }
        if value > maximum {
            maximum = value;
        }
    }
    if maximum <= minimum {
        return img.clone();
    }
    let span = (maximum - minimum) as f32;
    let (width, height) = img.dimensions();
    GrayImage::from_fn(width, height, |x, y| {
        let value = img.get_pixel(x, y)[0];
        let stretched = ((value - minimum) as f32 / span * 255.0).round();
        Luma([stretched as u8])
    })
}

pub fn find_features(img: &GrayImage, brief_pairs: &[(Point2<i32>, Point2<i32>)]) -> Vec<Feature> {
    find_features_tuned(
        img,
        brief_pairs,
        FAST_THRESHOLD,
        NON_MAXIMA_SUPPRESSION_RADIUS,
    )
}

/// Detect BRIEF features on a small image pyramid and map every keypoint back
/// into the input image's coordinate system.
///
/// FAST/BRIEF by itself has no scale invariant descriptor.  That is especially
/// noticeable when a phone changes from its wide camera to a telephoto camera:
/// the same brush stroke can be two or three times larger in the second frame.
/// Sampling a few downscaled copies gives the matcher descriptors whose support
/// covers the same physical area in both images, while retaining the original
/// (full-size) level for ordinary stacks.  The returned coordinates always use
/// the input image coordinate system, so callers do not need a second transform.
pub fn find_features_multiscale(
    img: &GrayImage,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
    max_features: usize,
    scales: &[f32],
) -> Vec<Feature> {
    if img.width() == 0 || img.height() == 0 || max_features == 0 {
        return Vec::new();
    }

    // Always include the native level even when a caller supplies a custom
    // list.  Stable ordering makes the result deterministic across rayon
    // worker counts and gives native descriptors first choice when a location
    // is represented at more than one scale.
    let mut levels = Vec::with_capacity(scales.len() + 1);
    levels.push(1.0f32);
    for &scale in scales {
        if scale.is_finite() && scale > 0.05 && scale < 4.0 {
            let duplicate = levels
                .iter()
                .any(|existing| (*existing - scale).abs() < 0.01);
            if !duplicate {
                levels.push(scale);
            }
        }
    }

    let mut per_level = Vec::<Vec<Feature>>::with_capacity(levels.len());
    for &scale in &levels {
        let width = ((img.width() as f32 * scale).round() as u32).max(1);
        let height = ((img.height() as f32 * scale).round() as u32).max(1);
        let level = if (width, height) == img.dimensions() {
            img.clone()
        } else {
            image::imageops::resize(img, width, height, image::imageops::FilterType::Triangle)
        };
        let mut features = find_features(&level, brief_pairs);
        let scale_x = level.width() as f64 / img.width().max(1) as f64;
        let scale_y = level.height() as f64 / img.height().max(1) as f64;
        for feature in &mut features {
            feature.support_scale = scale;
            feature.keypoint.x = ((feature.keypoint.x as f64 / scale_x).round())
                .clamp(0.0, img.width().saturating_sub(1) as f64)
                as u32;
            feature.keypoint.y = ((feature.keypoint.y as f64 / scale_y).round())
                .clamp(0.0, img.height().saturating_sub(1) as f64)
                as u32;
        }
        per_level.push(features);
    }

    // Reserve most slots for the native image and distribute the remainder
    // across pyramid levels.  This preserves the established behaviour on
    // normal same-scale pairs while still guaranteeing cross-scale support.
    let native_quota = if per_level.len() == 1 {
        max_features
    } else {
        (max_features * 2 / 5).max(1)
    };
    let remaining_levels = per_level.len().saturating_sub(1);
    let secondary_quota = max_features
        .saturating_sub(native_quota)
        .checked_div(remaining_levels)
        .unwrap_or(0)
        .max(1);
    let mut selected = Vec::with_capacity(max_features);
    let mut quotas = Vec::with_capacity(per_level.len());
    quotas.push(native_quota);
    quotas.extend(std::iter::repeat_n(secondary_quota, remaining_levels));
    for (features, quota) in per_level.iter_mut().zip(quotas) {
        // Drain only the consumed prefix. `drain(..).take(quota)` also drops
        // every unused feature when the iterator is destroyed, so low-texture
        // levels could never lend their spare quota to the useful scales.
        let count = quota.min(features.len());
        selected.extend(features.drain(..count));
    }

    // If a low-texture level did not fill its quota, use the unused candidates
    // from all levels before giving up.  Do not deduplicate by position: two
    // descriptors at one location but different support scales are precisely
    // what lets a 35mm/85mm pair match.
    if selected.len() < max_features {
        for features in &mut per_level {
            if selected.len() >= max_features {
                break;
            }
            let count = max_features - selected.len();
            let count = count.min(features.len());
            selected.extend(features.drain(..count));
        }
    }
    selected.truncate(max_features);
    selected
}

pub fn find_features_tuned(
    img: &GrayImage,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
    fast_threshold: u8,
    non_maxima_suppression_radius: f32,
) -> Vec<Feature> {
    let blurred_img_u8 = imageproc::filter::gaussian_blur_f32(img, 1.5);
    let corners = corners_fast9(&blurred_img_u8, fast_threshold);
    let keypoints = non_maximal_suppression(&corners, non_maxima_suppression_radius);
    let blurred_img_f32 = gaussian_blur_f32(&convert_gray_u8_to_f32(img), 2.0);
    let mut features: Vec<Feature> = keypoints
        .par_iter()
        .filter_map(|kp| {
            compute_brief_descriptor(&blurred_img_f32, kp, BRIEF_PATCH_SIZE, brief_pairs).map(
                |descriptor| Feature {
                    keypoint: *kp,
                    descriptor,
                    support_scale: 1.0,
                },
            )
        })
        .collect();
    features.truncate(MAX_FEATURES);
    features
}

fn non_maximal_suppression(corners: &[Corner], radius: f32) -> Vec<KeyPoint> {
    let mut sorted_corners = corners.to_vec();
    sorted_corners.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut result = Vec::new();
    let radius_sq = radius * radius;
    let cell_size = radius.max(1.0);
    let mut accepted_by_cell: HashMap<(i32, i32), Vec<KeyPoint>> = HashMap::new();

    for corner in sorted_corners {
        let cell_x = (corner.x as f32 / cell_size).floor() as i32;
        let cell_y = (corner.y as f32 / cell_size).floor() as i32;
        let is_suppressed = (-1..=1).any(|cell_dy| {
            (-1..=1).any(|cell_dx| {
                accepted_by_cell
                    .get(&(cell_x + cell_dx, cell_y + cell_dy))
                    .is_some_and(|accepted| {
                        accepted.iter().any(|point| {
                            let dx = point.x as f32 - corner.x as f32;
                            let dy = point.y as f32 - corner.y as f32;
                            dx * dx + dy * dy < radius_sq
                        })
                    })
            })
        });
        if !is_suppressed {
            let point = KeyPoint {
                x: corner.x,
                y: corner.y,
            };
            result.push(point);
            accepted_by_cell
                .entry((cell_x, cell_y))
                .or_default()
                .push(point);
        }
    }
    result
}

pub fn generate_brief_pairs() -> Vec<(Point2<i32>, Point2<i32>)> {
    let mut rng = StdRng::seed_from_u64(12345);
    let half_patch = BRIEF_PATCH_SIZE as i32 / 2;
    let distribution = match rand::distr::Uniform::new(-half_patch, half_patch) {
        Ok(dist) => dist,
        Err(e) => panic!("Failed to create uniform distribution: {}", e),
    };

    (0..BRIEF_DESCRIPTOR_SIZE)
        .map(|_| {
            (
                Point2::new(distribution.sample(&mut rng), distribution.sample(&mut rng)),
                Point2::new(distribution.sample(&mut rng), distribution.sample(&mut rng)),
            )
        })
        .collect()
}

fn compute_brief_descriptor(
    img: &ImageBuffer<Luma<f32>, Vec<f32>>,
    kp: &KeyPoint,
    patch_size: u32,
    pairs: &[(Point2<i32>, Point2<i32>)],
) -> Option<Descriptor> {
    let mut descriptor = [0u8; BRIEF_DESCRIPTOR_SIZE / 8];
    let (width, height) = img.dimensions();
    let max_pair_radius = (pairs
        .iter()
        .flat_map(|pair| [pair.0, pair.1])
        .map(|point| ((point.x * point.x + point.y * point.y) as f32).sqrt())
        .fold(0.0f32, f32::max)
        .ceil() as i32)
        .max((patch_size / 2) as i32)
        + 1;
    if width <= (max_pair_radius * 2) as u32
        || height <= (max_pair_radius * 2) as u32
        || kp.x < max_pair_radius as u32
        || kp.y < max_pair_radius as u32
        || kp.x + max_pair_radius as u32 >= width
        || kp.y + max_pair_radius as u32 >= height
    {
        return None;
    }
    let orientation = estimate_patch_orientation(img, kp, (patch_size / 2) as i32);
    let (sin_theta, cos_theta) = orientation.sin_cos();
    for (i, pair) in pairs.iter().enumerate() {
        let rotate = |point: nalgebra::Point2<i32>| {
            let x = point.x as f32 * cos_theta - point.y as f32 * sin_theta;
            let y = point.x as f32 * sin_theta + point.y as f32 * cos_theta;
            (x.round() as i32, y.round() as i32)
        };
        let (p1_dx, p1_dy) = rotate(pair.0);
        let (p2_dx, p2_dy) = rotate(pair.1);
        let p1_x = (kp.x as i32 + p1_dx) as u32;
        let p1_y = (kp.y as i32 + p1_dy) as u32;
        let p2_x = (kp.x as i32 + p2_dx) as u32;
        let p2_y = (kp.y as i32 + p2_dy) as u32;
        let intensity1 = img.get_pixel(p1_x, p1_y)[0];
        let intensity2 = img.get_pixel(p2_x, p2_y)[0];
        if intensity1 < intensity2 {
            let byte_index = i / 8;
            let bit_index = i % 8;
            descriptor[byte_index] |= 1 << bit_index;
        }
    }
    Some(descriptor)
}

fn estimate_patch_orientation(
    img: &ImageBuffer<Luma<f32>, Vec<f32>>,
    kp: &KeyPoint,
    radius: i32,
) -> f32 {
    let mut moment_x = 0.0f32;
    let mut moment_y = 0.0f32;
    let center_x = kp.x as i32;
    let center_y = kp.y as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let x = (center_x + dx) as u32;
            let y = (center_y + dy) as u32;
            let intensity = img.get_pixel(x, y)[0];
            moment_x += dx as f32 * intensity;
            moment_y += dy as f32 * intensity;
        }
    }
    if moment_x.abs() + moment_y.abs() < f32::EPSILON {
        0.0
    } else {
        moment_y.atan2(moment_x)
    }
}

fn hamming_distance(d1: &Descriptor, d2: &Descriptor) -> u32 {
    d1.iter()
        .zip(d2.iter())
        .map(|(b1, b2)| (b1 ^ b2).count_ones())
        .sum()
}

pub fn match_features(features1: &[Feature], features2: &[Feature]) -> Vec<Match> {
    match_features_with_ratio(features1, features2, MATCH_RATIO_THRESHOLD)
}

/// Match BRIEF descriptors with a caller-selected Lowe ratio.  The default
/// matcher keeps the historical threshold; scalable stacks can use a slightly
/// looser ratio because pyramid levels intentionally contribute near-duplicate
/// descriptors at different support sizes.
pub fn match_features_with_ratio(
    features1: &[Feature],
    features2: &[Feature],
    ratio_threshold: f32,
) -> Vec<Match> {
    if features1.is_empty() || features2.is_empty() {
        return Vec::new();
    }
    // Compute each Hamming distance once. The previous bidirectional search
    // recalculated the full descriptor matrix twice, which dominates large
    // ordered stacks even after candidate-pair pruning.
    let distances: Vec<Vec<u16>> = features1
        .par_iter()
        .map(|f1| {
            features2
                .iter()
                .map(|f2| hamming_distance(&f1.descriptor, &f2.descriptor) as u16)
                .collect()
        })
        .collect();
    let best_for_first: Vec<(usize, u32, u32)> = distances
        .par_iter()
        .map(|row| {
            let mut best_dist = u32::MAX;
            let mut second_best_dist = u32::MAX;
            let mut best_idx = 0;
            for (j, &distance) in row.iter().enumerate() {
                let dist = u32::from(distance);
                if dist < best_dist {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_idx = j;
                } else if dist < second_best_dist {
                    second_best_dist = dist;
                }
            }
            (best_idx, best_dist, second_best_dist)
        })
        .collect();
    let best_for_second: Vec<usize> = (0..features2.len())
        .into_par_iter()
        .map(|second_index| {
            let mut best_dist = u32::MAX;
            let mut best_idx = 0;
            for (first_index, row) in distances.iter().enumerate() {
                let dist = u32::from(row[second_index]);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = first_index;
                }
            }
            best_idx
        })
        .collect();

    best_for_first
        .into_iter()
        .enumerate()
        .filter_map(|(index1, (index2, best_dist, second_best_dist))| {
            if second_best_dist > 0
                && (best_dist as f32 / second_best_dist as f32) < ratio_threshold
                && best_for_second[index2] == index1
            {
                Some(Match { index1, index2 })
            } else {
                None
            }
        })
        .collect()
}

/// Match only descriptor levels whose support sizes are compatible with the
/// expected optical scale change. The ordinary matcher remains the fallback:
/// metadata is a prior, not proof, because phones can crop a module or change
/// their distance to the artwork at the same time as the focal length.
pub fn match_features_with_ratio_and_scale(
    features1: &[Feature],
    features2: &[Feature],
    ratio_threshold: f32,
    expected_support_scale_ratio: f32,
    tolerance: f32,
) -> Vec<Match> {
    if features1.is_empty()
        || features2.is_empty()
        || !ratio_threshold.is_finite()
        || !expected_support_scale_ratio.is_finite()
        || expected_support_scale_ratio <= 0.0
        || !tolerance.is_finite()
        || tolerance < 1.0
    {
        return Vec::new();
    }
    let minimum_ratio = expected_support_scale_ratio / tolerance;
    let maximum_ratio = expected_support_scale_ratio * tolerance;
    let distances: Vec<Vec<u16>> = features1
        .par_iter()
        .map(|f1| {
            features2
                .iter()
                .map(|f2| hamming_distance(&f1.descriptor, &f2.descriptor) as u16)
                .collect()
        })
        .collect();
    let best_for_first: Vec<(usize, u32, u32)> = distances
        .par_iter()
        .enumerate()
        .map(|(index1, row)| {
            let mut best = (usize::MAX, u32::MAX, u32::MAX);
            for (index2, &distance) in row.iter().enumerate() {
                let support_ratio = features2[index2].support_scale
                    / features1[index1].support_scale.max(f32::EPSILON);
                if !support_ratio.is_finite()
                    || support_ratio < minimum_ratio
                    || support_ratio > maximum_ratio
                {
                    continue;
                }
                let distance = u32::from(distance);
                if distance < best.1 {
                    best = (index2, distance, best.1);
                } else if distance < best.2 {
                    best.2 = distance;
                }
            }
            best
        })
        .collect();
    let best_for_second: Vec<usize> = (0..features2.len())
        .into_par_iter()
        .map(|index2| {
            let mut best = (usize::MAX, u32::MAX);
            for (index1, row) in distances.iter().enumerate() {
                let support_ratio = features2[index2].support_scale
                    / features1[index1].support_scale.max(f32::EPSILON);
                if !support_ratio.is_finite()
                    || support_ratio < minimum_ratio
                    || support_ratio > maximum_ratio
                {
                    continue;
                }
                let distance = u32::from(row[index2]);
                if distance < best.1 {
                    best = (index1, distance);
                }
            }
            best.0
        })
        .collect();

    best_for_first
        .into_iter()
        .enumerate()
        .filter_map(|(index1, (index2, best_dist, second_best_dist))| {
            (index2 != usize::MAX
                && second_best_dist > 0
                && (best_dist as f32 / second_best_dist as f32) < ratio_threshold
                && best_for_second[index2] == index1)
                .then_some(Match { index1, index2 })
        })
        .collect()
}

/// Count reliable descriptor correspondences without allocating the complete
/// distance matrix.  Large stacks use this as a cheap image-retrieval pass to
/// discover a few non-neighbouring overlap candidates; the selected pairs are
/// then passed through the full geometric/RANSAC matcher.  Keeping this pass
/// allocation-free is important for 100–200 image selections.
pub fn count_mutual_descriptor_matches_with_ratio(
    features1: &[Feature],
    features2: &[Feature],
    ratio_threshold: f32,
) -> usize {
    if features1.is_empty() || features2.is_empty() {
        return 0;
    }

    let mut best_for_first = Vec::with_capacity(features1.len());
    for feature1 in features1 {
        let mut best_distance = u32::MAX;
        let mut second_distance = u32::MAX;
        let mut best_index = 0usize;
        for (index, feature2) in features2.iter().enumerate() {
            let distance = hamming_distance(&feature1.descriptor, &feature2.descriptor);
            if distance < best_distance {
                second_distance = best_distance;
                best_distance = distance;
                best_index = index;
            } else if distance < second_distance {
                second_distance = distance;
            }
        }
        best_for_first.push((best_index, best_distance, second_distance));
    }

    let mut best_for_second = vec![0usize; features2.len()];
    for (second_index, best_index) in best_for_second.iter_mut().enumerate() {
        let mut distance = u32::MAX;
        for (first_index, feature1) in features1.iter().enumerate() {
            let candidate =
                hamming_distance(&feature1.descriptor, &features2[second_index].descriptor);
            if candidate < distance {
                distance = candidate;
                *best_index = first_index;
            }
        }
    }

    best_for_first
        .into_iter()
        .enumerate()
        .filter(
            |(first_index, (second_index, best_distance, second_distance))| {
                *second_distance > 0
                    && (*best_distance as f32 / *second_distance as f32) < ratio_threshold
                    && best_for_second[*second_index] == *first_index
            },
        )
        .count()
}

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

fn find_homography_ransac_points_with_solver(
    points: &[(Point2<f64>, Point2<f64>)],
    inlier_threshold: f64,
    stable_four_point_solver: bool,
    minimum_inliers: usize,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    let mut rng = StdRng::seed_from_u64(
        0x9E37_79B9_7F4A_7C15u64 ^ (points.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93),
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

fn are_points_collinear(p1: Point2<f64>, p2: Point2<f64>, p3: Point2<f64>) -> bool {
    let area = p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y);
    area.abs() < 1e-6
}

fn compute_homography_four_points(points: &[(Point2<f64>, Point2<f64>)]) -> Option<Matrix3<f64>> {
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

fn normalize_points(points: &[Point2<f64>]) -> Option<(Vec<Point2<f64>>, Matrix3<f64>)> {
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

fn convert_gray_u8_to_f32(img: &GrayImage) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (width, height) = img.dimensions();
    ImageBuffer::from_fn(width, height, |x, y| {
        Luma([img.get_pixel(x, y)[0] as f32 / 255.0])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature(descriptor: Descriptor) -> Feature {
        Feature {
            keypoint: KeyPoint { x: 0, y: 0 },
            descriptor,
            support_scale: 1.0,
        }
    }

    fn apply_homography(h: &Matrix3<f64>, point: Point2<f64>) -> Point2<f64> {
        let transformed = h * nalgebra::Point3::new(point.x, point.y, 1.0);
        Point2::new(transformed.x / transformed.z, transformed.y / transformed.z)
    }

    #[test]
    fn normalized_homography_remains_stable_at_full_resolution_coordinates() {
        let expected = Matrix3::new(
            1.08, 0.015, 1800.0, -0.012, 0.97, -620.0, 0.0000012, -0.0000008, 1.0,
        );
        let source = [
            Point2::new(120.0, 80.0),
            Point2::new(8800.0, 120.0),
            Point2::new(9000.0, 5800.0),
            Point2::new(160.0, 6000.0),
            Point2::new(4200.0, 3000.0),
            Point2::new(7200.0, 4400.0),
        ];
        let pairs: Vec<_> = source
            .iter()
            .copied()
            .map(|point| (point, apply_homography(&expected, point)))
            .collect();

        let fitted = compute_homography(&pairs).expect("six non-collinear points should fit");
        for (source_point, target_point) in pairs {
            let fitted_point = apply_homography(&fitted, source_point);
            assert!((fitted_point - target_point).norm() < 1e-5);
        }
    }

    #[test]
    fn cached_distance_matrix_preserves_mutual_ratio_matches() {
        let source = vec![feature([0; 32]), feature([u8::MAX; 32])];
        let mut near_zero = [0; 32];
        near_zero[0] = 1;
        let mut near_max = [u8::MAX; 32];
        near_max[0] = u8::MAX - 1;
        let target = vec![feature(near_zero), feature(near_max)];

        let matches = match_features(&source, &target);
        assert_eq!(matches.len(), 2);
        assert_eq!((matches[0].index1, matches[0].index2), (0, 0));
        assert_eq!((matches[1].index1, matches[1].index2), (1, 1));
    }

    #[test]
    fn multiscale_features_bridge_a_scaled_pair() {
        let source = GrayImage::from_fn(640, 480, |x, y| {
            let mut value = 28.0f32;
            // Distinct, comfortably sized marks make the test exercise the
            // pyramid support rather than high-frequency aliasing. Their
            // centres are deterministic and spread over the complete frame.
            for index in 0..24u32 {
                let cx = 32.0 + ((index * 193) % 576) as f32;
                let cy = 30.0 + ((index * 157) % 420) as f32;
                let radius = 8.0 + (index % 5) as f32 * 2.5;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let distance = (dx * dx + dy * dy).sqrt();
                if distance <= radius {
                    value = 210.0 + (index % 4) as f32 * 10.0;
                }
                if (dx + dy * 0.37).abs() < 1.5 && distance < radius * 2.2 {
                    value = 120.0 + (index % 5) as f32 * 12.0;
                }
            }
            Luma([value.round() as u8])
        });
        let target =
            image::imageops::resize(&source, 320, 240, image::imageops::FilterType::Triangle);
        let pairs = crate::panorama_utils::processing::generate_brief_pairs();
        let source_features = find_features_multiscale(&source, &pairs, 1_000, &[0.5]);
        let target_features = find_features_multiscale(&target, &pairs, 1_000, &[0.5]);
        let matches = match_features_with_ratio(&source_features, &target_features, 0.9);
        assert!(
            source_features.len() >= 100,
            "source={}",
            source_features.len()
        );
        assert!(
            target_features.len() >= 20,
            "target={}",
            target_features.len()
        );
        assert!(matches.len() >= 8, "matches={}", matches.len());

        let points = matches
            .iter()
            .map(|matched| {
                (
                    Point2::new(
                        source_features[matched.index1].keypoint.x as f64,
                        source_features[matched.index1].keypoint.y as f64,
                    ),
                    Point2::new(
                        target_features[matched.index2].keypoint.x as f64,
                        target_features[matched.index2].keypoint.y as f64,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let (transform, inliers) = find_homography_ransac_points_stable(&points, 8.0)
            .expect("the scaled texture should produce a geometric consensus");
        assert!(inliers.len() >= 8, "inliers={}", inliers.len());
        assert!(
            (transform[(0, 0)] - 0.5).abs() < 0.08,
            "transform={transform:?}"
        );
        assert!(
            (transform[(1, 1)] - 0.5).abs() < 0.08,
            "transform={transform:?}"
        );
    }

    #[test]
    fn stable_ransac_recovers_exact_translation() {
        let points: Vec<_> = (0..5)
            .flat_map(|row| {
                (0..6).map(move |column| {
                    let source = Point2::new(40.0 + column as f64 * 31.0, 35.0 + row as f64 * 29.0);
                    let target = Point2::new(source.x - 2.0, source.y + 3.0);
                    (source, target)
                })
            })
            .collect();

        let (homography, inliers) = find_homography_ransac_points_stable(&points, 1.0)
            .expect("the normalized four-point solver should recover an exact translation");
        assert_eq!(inliers.len(), points.len());
        let probe = Point2::new(123.0, 87.0);
        let mapped = apply_homography(&homography, probe);
        assert!((mapped - Point2::new(121.0, 90.0)).norm() < 1e-6);
    }
}
