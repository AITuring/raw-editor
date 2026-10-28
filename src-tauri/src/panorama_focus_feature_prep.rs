//! Focus feature preparation and foreground-mask contracts.
//!
//! This module owns the bounded analysis-image feature path used by focus stacks.
//! Edge geometry and source decoding remain in the facade until their dedicated
//! N1 stages are migrated.

use image::GrayImage;

use crate::panorama_utils::processing;

use super::{
    FOCUS_FOREGROUND_LUMA_THRESHOLD, FOCUS_FOREGROUND_MIN_BRIGHT_FRACTION,
    FOCUS_FOREGROUND_MIN_HEIGHT_RATIO, FOCUS_FOREGROUND_SCAN_MAX_Y, Feature, ImageInfo, KeyPoint,
    SCALABLE_LOW_TEXTURE_FAST_THRESHOLD, SCALABLE_LOW_TEXTURE_FEATURE_TARGET,
    SCALABLE_LOW_TEXTURE_NMS_RADIUS,
};

pub(super) fn find_alignment_features(
    alignment_image: &GrayImage,
    brief_pairs: &[(nalgebra::Point2<i32>, nalgebra::Point2<i32>)],
    max_features: usize,
    scalable_stack: bool,
    focal_length_35mm: Option<f64>,
) -> Vec<Feature> {
    // A fixed-size BRIEF patch is not scale invariant.  Large phone stacks can
    // switch lenses midway through a capture (for example 35mm -> 85mm), so
    // keep descriptors from a small image pyramid on the bounded-memory path.
    // The native level remains dominant; the extra levels only provide a bridge
    // when the same detail is rendered at a different magnification.
    let mut features = if scalable_stack {
        let mut scales = vec![0.38, 0.5, std::f32::consts::FRAC_1_SQRT_2];
        // A metadata-guided level places the two camera modules on a common
        // support scale.  The generic levels remain in place for cropped or
        // non-EXIF sources.
        if let Some(focal) = focal_length_35mm.filter(|value| value.is_finite() && *value > 0.0) {
            let metadata_scale = (50.0 / focal).clamp(0.28, 2.5) as f32;
            if scales
                .iter()
                .all(|existing| (*existing - metadata_scale).abs() >= 0.04)
            {
                scales.push(metadata_scale);
            }
        }
        processing::find_features_multiscale(alignment_image, brief_pairs, max_features, &scales)
    } else {
        processing::find_features(alignment_image, brief_pairs)
    };
    if scalable_stack && features.len() < SCALABLE_LOW_TEXTURE_FEATURE_TARGET {
        let normalized = processing::normalize_grayscale(alignment_image);
        let fallback = processing::find_features_tuned(
            &normalized,
            brief_pairs,
            SCALABLE_LOW_TEXTURE_FAST_THRESHOLD,
            SCALABLE_LOW_TEXTURE_NMS_RADIUS,
        );
        if fallback.len() > features.len() {
            // Keep the pyramid descriptors even when the normalized native
            // pass finds more corners. Replacing the vector here used to erase
            // the only scale bridge on a low-texture frame—the exact case in
            // which the 35mm/85mm pair is hardest to register.
            let remaining = max_features.saturating_sub(features.len());
            features.extend(fallback.into_iter().take(remaining));
        }
    }
    features.truncate(max_features);
    features
}

pub(super) fn detect_foreground_range(alignment_image: &GrayImage) -> Option<(f64, f64)> {
    let width = alignment_image.width();
    let height = alignment_image.height();
    if width < 64 || height < 128 {
        return None;
    }

    let scan_end = ((height as f64 * FOCUS_FOREGROUND_SCAN_MAX_Y).round() as u32)
        .clamp(1, height.saturating_sub(1));
    let minimum_bright_pixels = (width as f64 * FOCUS_FOREGROUND_MIN_BRIGHT_FRACTION).ceil() as u32;
    let bright_rows = (0..scan_end)
        .map(|y| {
            let bright_pixels = alignment_image
                .rows()
                .nth(y as usize)
                .into_iter()
                .flatten()
                .filter(|pixel| pixel[0] >= FOCUS_FOREGROUND_LUMA_THRESHOLD)
                .count() as u32;
            bright_pixels >= minimum_bright_pixels
        })
        .collect::<Vec<_>>();

    // This is only a conservative seed for an optional depth mask. The main
    // focus registration does not depend on it and samples the complete frame.
    let max_gap = ((height as f64 * 0.012).round() as usize).max(2);
    let mut first = None;
    let mut last = None;
    let mut gap = 0usize;
    for (row, is_bright) in bright_rows.iter().copied().enumerate() {
        if is_bright {
            if first.is_none() {
                first = Some(row);
            }
            last = Some(row);
            gap = 0;
        } else if first.is_some() {
            gap += 1;
            if gap > max_gap {
                break;
            }
        }
    }
    let (first, last) = first.zip(last)?;
    let run_height = last.saturating_sub(first) + 1;
    if (run_height as f64) < height as f64 * FOCUS_FOREGROUND_MIN_HEIGHT_RATIO {
        return None;
    }

    let padding = (height as f64 * 0.012).max(3.0);
    let minimum = ((first as f64 - padding) / height as f64).clamp(0.0, 1.0);
    let maximum = (((last + 1) as f64 + padding) / height as f64).clamp(0.0, 1.0);
    Some((minimum, maximum))
}

pub(super) fn dilate_binary_mask(
    mask: &GrayImage,
    horizontal_radius: usize,
    vertical_radius: usize,
) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return mask.clone();
    }
    let width = width as usize;
    let height = height as usize;
    let source = mask.as_raw();
    let mut horizontal = vec![0u8; width * height];
    let mut output = vec![0u8; width * height];

    for y in 0..height {
        let row_start = y * width;
        let mut prefix = vec![0u32; width + 1];
        for x in 0..width {
            prefix[x + 1] = prefix[x] + u32::from(source[row_start + x] > 0);
        }
        for x in 0..width {
            let start = x.saturating_sub(horizontal_radius);
            let end = (x + horizontal_radius).min(width - 1);
            if prefix[end + 1] > prefix[start] {
                horizontal[row_start + x] = 255;
            }
        }
    }

    for x in 0..width {
        let mut prefix = vec![0u32; height + 1];
        for y in 0..height {
            prefix[y + 1] = prefix[y] + u32::from(horizontal[y * width + x] > 0);
        }
        for y in 0..height {
            let start = y.saturating_sub(vertical_radius);
            let end = (y + vertical_radius).min(height - 1);
            if prefix[end + 1] > prefix[start] {
                output[y * width + x] = 255;
            }
        }
    }

    GrayImage::from_raw(width as u32, height as u32, output)
        .expect("foreground mask dimensions must match")
}

pub(super) fn build_foreground_mask(
    alignment_image: &GrayImage,
    foreground_range: Option<(f64, f64)>,
) -> Option<GrayImage> {
    let (minimum_y, maximum_y) = foreground_range?;
    let (width, height) = alignment_image.dimensions();
    if width == 0 || height == 0 || minimum_y >= maximum_y {
        return None;
    }
    let base = GrayImage::from_fn(width, height, |x, y| {
        let normalized_y = y as f64 / height.max(1) as f64;
        let value = alignment_image.get_pixel(x, y)[0];
        image::Luma([
            if (minimum_y..=maximum_y).contains(&normalized_y)
                && value >= FOCUS_FOREGROUND_LUMA_THRESHOLD
            {
                255
            } else {
                0
            },
        ])
    });
    // Include the narrow shadow and metallic lip around the bright core so
    // ownership remains stable at its silhouette. This is a mask-space margin;
    // it does not alter the generic registration path.
    let horizontal_radius = ((width as f64 * 0.012).round() as usize).clamp(2, 32);
    let vertical_radius = ((height as f64 * 0.025).round() as usize).clamp(3, 48);
    Some(dilate_binary_mask(
        &base,
        horizontal_radius,
        vertical_radius,
    ))
}

pub(super) fn focus_alignment_keypoint_is_foreground(
    image: &ImageInfo,
    keypoint: KeyPoint,
) -> bool {
    let Some((minimum_y, maximum_y)) = image.foreground_range else {
        return false;
    };
    let (width, height) = image.alignment_image.dimensions();
    if keypoint.x >= width || keypoint.y >= height {
        return false;
    }
    let normalized_y = keypoint.y as f64 / height.max(1) as f64;
    if !(minimum_y..=maximum_y).contains(&normalized_y) {
        return false;
    }
    image.foreground_mask.as_ref().map_or_else(
        || {
            image.alignment_image.get_pixel(keypoint.x, keypoint.y)[0]
                >= FOCUS_FOREGROUND_LUMA_THRESHOLD
        },
        |mask| mask.get_pixel(keypoint.x, keypoint.y)[0] > 0,
    )
}

pub(super) fn find_top_alignment_features(
    alignment_image: &GrayImage,
    brief_pairs: &[(nalgebra::Point2<i32>, nalgebra::Point2<i32>)],
    foreground_range: Option<(f64, f64)>,
) -> Vec<Feature> {
    let Some((minimum_y, maximum_y)) = foreground_range else {
        return Vec::new();
    };
    if alignment_image.height() < 128 || minimum_y >= maximum_y {
        return Vec::new();
    }
    let top_start = (alignment_image.height() as f64 * minimum_y).round() as u32;
    let top_end = (alignment_image.height() as f64 * maximum_y)
        .round()
        .max((top_start + 64) as f64)
        .min(alignment_image.height() as f64) as u32;
    let top_region = image::imageops::crop_imm(
        alignment_image,
        0,
        top_start,
        alignment_image.width(),
        top_end.saturating_sub(top_start),
    )
    .to_image();
    let mut features = processing::find_features_tuned(
        &top_region,
        brief_pairs,
        SCALABLE_LOW_TEXTURE_FAST_THRESHOLD,
        (SCALABLE_LOW_TEXTURE_NMS_RADIUS * 0.75).max(6.0),
    );
    let normalized_region = processing::normalize_grayscale(&top_region);
    features.extend(processing::find_features_tuned(
        &normalized_region,
        brief_pairs,
        5,
        8.0,
    ));
    for feature in &mut features {
        feature.keypoint.y = feature.keypoint.y.saturating_add(top_start);
    }
    features.retain(|feature| {
        let y = feature.keypoint.y as f64 / alignment_image.height().max(1) as f64;
        (minimum_y..=maximum_y).contains(&y)
            && alignment_image
                .get_pixel(feature.keypoint.x, feature.keypoint.y)
                .0[0]
                >= FOCUS_FOREGROUND_LUMA_THRESHOLD
    });
    features
}
