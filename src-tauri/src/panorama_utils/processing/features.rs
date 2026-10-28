//! FAST/BRIEF feature detection and descriptor matching.
//!
//! Corner detection with a local-contrast detector image and an image pyramid,
//! BRIEF descriptors with patch orientation, and Hamming ratio-test matching
//! with its attrition counts.

use crate::panorama_utils::shared::BRIEF_DESCRIPTOR_SIZE;
use crate::panorama_utils::shared::Descriptor;
use crate::panorama_utils::shared::Feature;
use crate::panorama_utils::shared::KeyPoint;
use crate::panorama_utils::shared::Match;
use image::GrayImage;
use image::ImageBuffer;
use image::Luma;
use imageproc::corners::Corner;
use imageproc::corners::corners_fast9;
use imageproc::filter::gaussian_blur_f32;
use nalgebra::Point2;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

use super::{
    BRIEF_PATCH_SIZE, FAST_THRESHOLD, LOCAL_CONTRAST_CELL_SIZE, LOCAL_CONTRAST_FAST_THRESHOLD,
    LOCAL_CONTRAST_MAX_GAIN, MATCH_RATIO_THRESHOLD, MAX_FEATURES, NON_MAXIMA_SUPPRESSION_RADIUS,
};

pub fn find_features(img: &GrayImage, brief_pairs: &[(Point2<i32>, Point2<i32>)]) -> Vec<Feature> {
    find_features_tuned(
        img,
        brief_pairs,
        FAST_THRESHOLD,
        NON_MAXIMA_SUPPRESSION_RADIUS,
    )
}

/// A bright mount and a dark display case can consume the global intensity
/// range while the artwork between them has only a few levels of contrast.
/// Stretch a bounded local range for corner detection; descriptors continue
/// to sample the original image, so this cannot invent a matching texture.
pub(super) fn local_contrast_detector_image(img: &GrayImage) -> GrayImage {
    let (width, height) = img.dimensions();
    if width == 0 || height == 0 {
        return img.clone();
    }
    let columns = width.div_ceil(LOCAL_CONTRAST_CELL_SIZE);
    let rows = height.div_ceil(LOCAL_CONTRAST_CELL_SIZE);
    let mut cells = Vec::with_capacity((columns * rows) as usize);
    for row in 0..rows {
        for column in 0..columns {
            let left = column * LOCAL_CONTRAST_CELL_SIZE;
            let top = row * LOCAL_CONTRAST_CELL_SIZE;
            let right = (left + LOCAL_CONTRAST_CELL_SIZE).min(width);
            let bottom = (top + LOCAL_CONTRAST_CELL_SIZE).min(height);
            let mut histogram = [0u32; 256];
            for y in top..bottom {
                for x in left..right {
                    histogram[img.get_pixel(x, y)[0] as usize] += 1;
                }
            }
            let count = (right - left) * (bottom - top);
            let percentile = |numerator: u32| {
                let target = (count * numerator / 100).max(1);
                let mut cumulative = 0;
                histogram
                    .iter()
                    .enumerate()
                    .find_map(|(value, &frequency)| {
                        cumulative += frequency;
                        (cumulative >= target).then_some(value as f32)
                    })
                    .unwrap_or(255.0)
            };
            let low = percentile(5);
            let high = percentile(95);
            let gain = (192.0 / (high - low).max(1.0)).clamp(1.0, LOCAL_CONTRAST_MAX_GAIN);
            cells.push(((low + high) * 0.5, gain));
        }
    }
    GrayImage::from_fn(width, height, |x, y| {
        let gx = (x as f32 / LOCAL_CONTRAST_CELL_SIZE as f32 - 0.5)
            .clamp(0.0, columns.saturating_sub(1) as f32);
        let gy = (y as f32 / LOCAL_CONTRAST_CELL_SIZE as f32 - 0.5)
            .clamp(0.0, rows.saturating_sub(1) as f32);
        let x0 = gx.floor() as u32;
        let y0 = gy.floor() as u32;
        let x1 = (x0 + 1).min(columns - 1);
        let y1 = (y0 + 1).min(rows - 1);
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let mut center = 0.0;
        let mut gain = 0.0;
        for (cx, cy, weight) in [
            (x0, y0, (1.0 - fx) * (1.0 - fy)),
            (x1, y0, fx * (1.0 - fy)),
            (x0, y1, (1.0 - fx) * fy),
            (x1, y1, fx * fy),
        ] {
            let cell = cells[(cy * columns + cx) as usize];
            center += cell.0 * weight;
            gain += cell.1 * weight;
        }
        let value = (img.get_pixel(x, y)[0] as f32 - center) * gain + 128.0;
        Luma([value.round().clamp(0.0, 255.0) as u8])
    })
}

pub(super) fn find_features_with_local_contrast(
    img: &GrayImage,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> Vec<Feature> {
    let mut features = find_features(img, brief_pairs);
    // Count against source area rather than a tiny absolute threshold: one
    // hundred corners on the mount should not suppress recovery over a two
    // megapixel low-contrast painting.
    let target = ((u64::from(img.width()) * u64::from(img.height()) / 2_400) as usize)
        .clamp(64, MAX_FEATURES);
    if features.len() >= target {
        return features;
    }
    let normalized = local_contrast_detector_image(img);
    let blurred_detector = gaussian_blur_f32(&normalized, 1.5);
    let corners = corners_fast9(&blurred_detector, LOCAL_CONTRAST_FAST_THRESHOLD);
    let keypoints = non_maximal_suppression(&corners, NON_MAXIMA_SUPPRESSION_RADIUS);
    let descriptor_image = gaussian_blur_f32(&convert_gray_u8_to_f32(img), 2.0);
    let minimum_distance_squared = (NON_MAXIMA_SUPPRESSION_RADIUS * 0.5).powi(2);
    for keypoint in keypoints {
        if features.iter().any(|existing| {
            let dx = existing.keypoint.x as f32 - keypoint.x as f32;
            let dy = existing.keypoint.y as f32 - keypoint.y as f32;
            dx * dx + dy * dy < minimum_distance_squared
        }) {
            continue;
        }
        if let Some(descriptor) =
            compute_brief_descriptor(&descriptor_image, &keypoint, BRIEF_PATCH_SIZE, brief_pairs)
        {
            features.push(Feature {
                keypoint,
                descriptor,
                support_scale: 1.0,
            });
        }
        if features.len() >= MAX_FEATURES {
            break;
        }
    }
    features
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
        let mut features = find_features_with_local_contrast(&level, brief_pairs);
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

pub(super) fn non_maximal_suppression(corners: &[Corner], radius: f32) -> Vec<KeyPoint> {
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

pub(super) fn compute_brief_descriptor(
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

pub(super) fn estimate_patch_orientation(
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

pub(super) fn hamming_distance(d1: &Descriptor, d2: &Descriptor) -> u32 {
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

/// Diagnostic mirror of [`match_features_with_ratio`] that exposes exact
/// descriptor attrition without changing the production matcher. Retrieval is
/// counted once a source descriptor has both a nearest and second-nearest
/// target; ratio survivors are counted before the reverse-best mutual gate.
pub(crate) struct DescriptorMatchAttrition {
    pub(crate) retrieval_count: usize,
    pub(crate) ratio_filter_count: usize,
    pub(crate) mutual_matches: Vec<Match>,
}

pub(crate) fn match_features_with_ratio_attrition(
    features1: &[Feature],
    features2: &[Feature],
    ratio_threshold: f32,
) -> DescriptorMatchAttrition {
    if features1.is_empty() || features2.len() < 2 {
        return DescriptorMatchAttrition {
            retrieval_count: 0,
            ratio_filter_count: 0,
            mutual_matches: Vec::new(),
        };
    }
    let distances = features1
        .par_iter()
        .map(|feature1| {
            features2
                .iter()
                .map(|feature2| hamming_distance(&feature1.descriptor, &feature2.descriptor) as u16)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let best_for_first = distances
        .par_iter()
        .map(|row| {
            let mut best_dist = u32::MAX;
            let mut second_best_dist = u32::MAX;
            let mut best_index = 0usize;
            for (index, &distance) in row.iter().enumerate() {
                let distance = u32::from(distance);
                if distance < best_dist {
                    second_best_dist = best_dist;
                    best_dist = distance;
                    best_index = index;
                } else if distance < second_best_dist {
                    second_best_dist = distance;
                }
            }
            (best_index, best_dist, second_best_dist)
        })
        .collect::<Vec<_>>();
    let best_for_second = (0..features2.len())
        .into_par_iter()
        .map(|second_index| {
            distances
                .iter()
                .enumerate()
                .min_by_key(|(_, row)| row[second_index])
                .map_or(0, |(first_index, _)| first_index)
        })
        .collect::<Vec<_>>();
    let ratio_filter_count = best_for_first
        .iter()
        .filter(|(_, best, second)| {
            *second > 0 && (*best as f32 / *second as f32) < ratio_threshold
        })
        .count();
    let mutual_matches = best_for_first
        .into_iter()
        .enumerate()
        .filter_map(|(index1, (index2, best, second))| {
            (second > 0
                && (best as f32 / second as f32) < ratio_threshold
                && best_for_second[index2] == index1)
                .then_some(Match { index1, index2 })
        })
        .collect();
    DescriptorMatchAttrition {
        retrieval_count: features1.len(),
        ratio_filter_count,
        mutual_matches,
    }
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

pub(super) fn convert_gray_u8_to_f32(img: &GrayImage) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (width, height) = img.dimensions();
    ImageBuffer::from_fn(width, height, |x, y| {
        Luma([img.get_pixel(x, y)[0] as f32 / 255.0])
    })
}
