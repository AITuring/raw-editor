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
mod tests;
