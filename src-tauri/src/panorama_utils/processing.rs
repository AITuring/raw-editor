#[cfg(test)]
use crate::panorama_utils::shared::{Descriptor, Feature, KeyPoint};
use image::{GrayImage, Luma};
#[cfg(test)]
use imageproc::corners::corners_fast9;
#[cfg(test)]
use imageproc::filter::gaussian_blur_f32;
#[cfg(test)]
use nalgebra::{Matrix3, Point2};

const MAX_PROCESSING_DIMENSION: u32 = 2400;
const FAST_THRESHOLD: u8 = 15;
const NON_MAXIMA_SUPPRESSION_RADIUS: f32 = 15.0;
const BRIEF_PATCH_SIZE: u32 = 32;
const MATCH_RATIO_THRESHOLD: f32 = 0.8;
const RANSAC_ITERATIONS: usize = 2500;
const RANSAC_INLIER_THRESHOLD: f64 = 5.0;
/// Minimum number of correspondences handed to one `rayon` task while scoring a
/// RANSAC hypothesis.
///
/// Scoring one hypothesis is a handful of floating point operations per point,
/// so the per-task dispatch cost dominates unless a task owns a sizeable run of
/// points.  This is a scheduling granularity floor only: the inlier set is
/// collected in ascending index order regardless of how the range is split, so
/// no value here can change the result (需求 14.6).
const RANSAC_INLIER_MIN_POINTS_PER_TASK: usize = 4096;
const MAX_FEATURES: usize = 2000;
const LOCAL_CONTRAST_CELL_SIZE: u32 = 160;
const LOCAL_CONTRAST_MAX_GAIN: f32 = 4.0;
const LOCAL_CONTRAST_FAST_THRESHOLD: u8 = 7;
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

mod features;

#[cfg(test)]
use features::local_contrast_detector_image;
pub(crate) use features::match_features_with_ratio_attrition;
pub use features::{
    count_mutual_descriptor_matches_with_ratio, find_features, find_features_multiscale,
    find_features_tuned, generate_brief_pairs, match_features, match_features_with_ratio,
    match_features_with_ratio_and_scale,
};

mod homography;

#[cfg(test)]
pub use homography::find_homography_ransac_points_stable;
pub use homography::{
    compute_homography, find_homography_ransac, find_homography_ransac_points,
    find_homography_ransac_points_stable_with_min_inliers,
};

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
    fn local_contrast_detector_stretches_sub_threshold_texture() {
        // A two-level patch is deliberately below FAST's native 15-level
        // threshold. It still has enough area in every 160px cell for bounded
        // local stretching to produce a useful detector image.
        let image = GrayImage::from_fn(640, 480, |x, y| {
            let in_patch =
                (x % 160).clamp(24, 103) == x % 160 && (y % 160).clamp(24, 103) == y % 160;
            Luma([if in_patch { 108 } else { 100 }])
        });
        let detector_image = local_contrast_detector_image(&image);
        let blurred = gaussian_blur_f32(&detector_image, 1.5);
        let _corners = corners_fast9(&blurred, FAST_THRESHOLD);
        let low_corners = corners_fast9(&blurred, 1);
        let pairs = generate_brief_pairs();
        let recovered = find_features_multiscale(&image, &pairs, 256, &[]);
        let (minimum, maximum) = detector_image
            .pixels()
            .map(|pixel| pixel[0])
            .fold((u8::MAX, u8::MIN), |(minimum, maximum), value| {
                (minimum.min(value), maximum.max(value))
            });
        assert!(
            maximum.saturating_sub(minimum) >= 30,
            "local contrast stretch was too weak: {minimum}..{maximum}"
        );
        assert!(
            low_corners.len() >= 256,
            "low-threshold corners={}",
            low_corners.len()
        );
        assert!(
            recovered.len() >= 32,
            "recovered features={}",
            recovered.len()
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
