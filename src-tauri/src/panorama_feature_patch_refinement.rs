//! Feature matching patch refinement helpers.
//!
//! This module is a focused implementation slice re-exported by the
//! `panorama_feature_matching` facade.

use super::*;

pub(crate) fn focus_stack_motion_is_shifted_mosaic(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    source_dimensions: (u32, u32),
) -> bool {
    if points.len() < FOCUS_MODEL_MIN_INLIERS {
        return false;
    }
    let translation = estimate_translation(points);
    let motion = nalgebra::Vector2::new(translation[(0, 2)], translation[(1, 2)]).norm();
    let image_scale = source_dimensions.0.max(source_dimensions.1) as f64;
    let translation_shift = motion.is_finite()
        && image_scale > 1.0
        && motion > image_scale * FOCUS_SHIFTED_MOSAIC_MOTION_RATIO;
    let scale_or_rotation_shift = estimate_similarity(points)
        .map(|similarity| {
            let scale_x = similarity[(0, 0)].hypot(similarity[(1, 0)]);
            let scale_y = similarity[(0, 1)].hypot(similarity[(1, 1)]);
            let scale = (scale_x.max(0.0) * scale_y.max(0.0)).sqrt();
            let rotation = 0.5
                * (similarity[(1, 0)] - similarity[(0, 1)])
                    .atan2(similarity[(0, 0)] + similarity[(1, 1)]);
            (scale.is_finite() && (scale - 1.0).abs() > 0.10)
                || (rotation.is_finite() && rotation.abs() > 0.035)
        })
        .unwrap_or(false);
    translation_shift || scale_or_rotation_shift
}

pub(crate) fn refine_match_point_from_homography(
    image1: &ImageInfo,
    image2: &ImageInfo,
    keypoint1: KeyPoint,
    keypoint2: KeyPoint,
    projection: Projection,
    homography: &Matrix3<f64>,
    prefer_feature_center: bool,
    high_precision: bool,
) -> Option<(nalgebra::Point2<f64>, nalgebra::Point2<f64>)> {
    let source_full_x = (keypoint1.x as f64 * image1.scale_factor).round() as i32;
    let source_full_y = (keypoint1.y as f64 * image1.scale_factor).round() as i32;
    let fallback_target_full_x = (keypoint2.x as f64 * image2.scale_factor).round() as i32;
    let fallback_target_full_y = (keypoint2.y as f64 * image2.scale_factor).round() as i32;
    let (target_full_x, target_full_y) =
        if projection == Projection::Planar && !prefer_feature_center {
            let predicted =
                homography * nalgebra::Point3::new(source_full_x as f64, source_full_y as f64, 1.0);
            if predicted.z.abs() < 1e-8 {
                (fallback_target_full_x, fallback_target_full_y)
            } else {
                (
                    (predicted.x / predicted.z).round() as i32,
                    (predicted.y / predicted.z).round() as i32,
                )
            }
        } else {
            (fallback_target_full_x, fallback_target_full_y)
        };

    let (
        source_plane,
        target_plane,
        source_x,
        source_y,
        target_x,
        target_y,
        target_scale,
        patch_radius,
        search_radius,
    ) = if let (Some(source), Some(target)) = (&image1.full_image, &image2.full_image) {
        (
            LumaPlane::Rgb(source),
            LumaPlane::Rgb(target),
            source_full_x,
            source_full_y,
            target_full_x,
            target_full_y,
            1.0,
            if high_precision {
                FOCUS_MATCH_REFINE_PATCH_RADIUS
            } else {
                MATCH_REFINE_PATCH_RADIUS
            },
            if high_precision {
                FOCUS_MATCH_REFINE_SEARCH_RADIUS
            } else {
                MATCH_REFINE_SEARCH_RADIUS
            },
        )
    } else {
        (
            LumaPlane::Gray(&image1.alignment_image),
            LumaPlane::Gray(&image2.alignment_image),
            keypoint1.x as i32,
            keypoint1.y as i32,
            (target_full_x as f64 / image2.scale_factor).round() as i32,
            (target_full_y as f64 / image2.scale_factor).round() as i32,
            image2.scale_factor,
            if high_precision { 8 } else { 4 },
            if high_precision { 12 } else { 6 },
        )
    };

    let (best_x, best_y, subpixel_x, subpixel_y) = refine_patch_position(
        &source_plane,
        &target_plane,
        source_x,
        source_y,
        target_x,
        target_y,
        patch_radius,
        search_radius,
    )?;
    let source = project_point(
        image1,
        source_full_x as f64,
        source_full_y as f64,
        projection,
    )?;
    let target = project_point(
        image2,
        (best_x as f64 + subpixel_x) * target_scale,
        (best_y as f64 + subpixel_y) * target_scale,
        projection,
    )?;
    Some((source, target))
}

pub(crate) enum LumaPlane<'a> {
    Rgb(&'a Rgb32FImage),
    Gray(&'a GrayImage),
}

impl LumaPlane<'_> {
    fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Rgb(image) => image.dimensions(),
            Self::Gray(image) => image.dimensions(),
        }
    }

    pub(crate) fn luma_at(&self, x: i32, y: i32) -> Option<f64> {
        let (width, height) = self.dimensions();
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            return None;
        }
        match self {
            Self::Rgb(image) => {
                let pixel = image.get_pixel(x as u32, y as u32);
                Some(
                    (pixel[0] as f64 * 0.299)
                        + (pixel[1] as f64 * 0.587)
                        + (pixel[2] as f64 * 0.114),
                )
            }
            Self::Gray(image) => Some(image.get_pixel(x as u32, y as u32)[0] as f64),
        }
    }

    fn gradient_at(&self, x: i32, y: i32) -> Option<(f64, f64)> {
        let horizontal = self.luma_at(x + 1, y)? - self.luma_at(x - 1, y)?;
        let vertical = self.luma_at(x, y + 1)? - self.luma_at(x, y - 1)?;
        Some((horizontal, vertical))
    }
}

pub(crate) fn gradient_patch_energy(
    image: &LumaPlane<'_>,
    center_x: i32,
    center_y: i32,
    radius: i32,
) -> f64 {
    let mut energy = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some((horizontal, vertical)) = image.gradient_at(center_x + dx, center_y + dy)
            else {
                return 0.0;
            };
            energy += horizontal * horizontal + vertical * vertical;
            sample_count += 1;
        }
    }
    if sample_count == 0 {
        0.0
    } else {
        (energy / sample_count as f64).sqrt()
    }
}

pub(crate) fn gradient_patch_corner_energy(
    image: &LumaPlane<'_>,
    center_x: i32,
    center_y: i32,
    radius: i32,
) -> f64 {
    let mut sum_horizontal_squared = 0.0;
    let mut sum_vertical_squared = 0.0;
    let mut sum_cross = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some((horizontal, vertical)) = image.gradient_at(center_x + dx, center_y + dy)
            else {
                return 0.0;
            };
            sum_horizontal_squared += horizontal * horizontal;
            sum_vertical_squared += vertical * vertical;
            sum_cross += horizontal * vertical;
            sample_count += 1;
        }
    }
    if sample_count == 0 {
        return 0.0;
    }
    let sample_count = sample_count as f64;
    let horizontal_squared = sum_horizontal_squared / sample_count;
    let vertical_squared = sum_vertical_squared / sample_count;
    let cross = sum_cross / sample_count;
    (horizontal_squared * vertical_squared - cross * cross)
        .max(0.0)
        .sqrt()
}

pub(crate) fn gradient_patch_ncc(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    center1_x: i32,
    center1_y: i32,
    center2_x: i32,
    center2_y: i32,
    radius: i32,
) -> f64 {
    let mut sum1_horizontal = 0.0;
    let mut sum1_vertical = 0.0;
    let mut sum2_horizontal = 0.0;
    let mut sum2_vertical = 0.0;
    let mut sum_squared1 = 0.0;
    let mut sum_squared2 = 0.0;
    let mut sum_product = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some((horizontal1, vertical1)) = image1.gradient_at(center1_x + dx, center1_y + dy)
            else {
                return f64::NEG_INFINITY;
            };
            let Some((horizontal2, vertical2)) = image2.gradient_at(center2_x + dx, center2_y + dy)
            else {
                return f64::NEG_INFINITY;
            };
            sum1_horizontal += horizontal1;
            sum1_vertical += vertical1;
            sum2_horizontal += horizontal2;
            sum2_vertical += vertical2;
            sum_squared1 += horizontal1 * horizontal1 + vertical1 * vertical1;
            sum_squared2 += horizontal2 * horizontal2 + vertical2 * vertical2;
            sum_product += horizontal1 * horizontal2 + vertical1 * vertical2;
            sample_count += 1;
        }
    }

    let sample_count = sample_count as f64;
    let covariance = sum_product
        - (sum1_horizontal * sum2_horizontal + sum1_vertical * sum2_vertical) / sample_count;
    let variance1 = sum_squared1
        - (sum1_horizontal * sum1_horizontal + sum1_vertical * sum1_vertical) / sample_count;
    let variance2 = sum_squared2
        - (sum2_horizontal * sum2_horizontal + sum2_vertical * sum2_vertical) / sample_count;
    if variance1 <= f64::EPSILON || variance2 <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        covariance / (variance1 * variance2).sqrt()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_foreground_patch_position(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    source_x: i32,
    source_y: i32,
    target_x: i32,
    target_y: i32,
    patch_radius: i32,
    search_radius: i32,
) -> Option<(i32, i32, f64, f64)> {
    let source_energy = gradient_patch_energy(image1, source_x, source_y, patch_radius);
    if source_energy < FOCUS_FOREGROUND_MIN_GRADIENT_ENERGY {
        return None;
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best_target = None;
    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let candidate_x = target_x + dx;
            let candidate_y = target_y + dy;
            let gradient_score = gradient_patch_ncc(
                image1,
                image2,
                source_x,
                source_y,
                candidate_x,
                candidate_y,
                patch_radius,
            );
            if !gradient_score.is_finite() {
                continue;
            }
            // Gradient correlation identifies stable depth-layer edges and
            // structure; a small luminance contribution keeps the position from
            // becoming arbitrary on a long, nearly uniform surface.
            let luminance_score = patch_ncc(
                image1,
                image2,
                source_x,
                source_y,
                candidate_x,
                candidate_y,
                patch_radius,
            );
            let score = if luminance_score.is_finite() {
                gradient_score * 0.75 + luminance_score * 0.25
            } else {
                gradient_score
            };
            if score > best_score {
                best_score = score;
                best_target = Some((candidate_x, candidate_y));
            }
        }
    }

    let (best_x, best_y) = best_target?;
    if !best_score.is_finite() {
        return None;
    }
    let sample = |x, y| {
        let gradient_score =
            gradient_patch_ncc(image1, image2, source_x, source_y, x, y, patch_radius);
        let luminance_score = patch_ncc(image1, image2, source_x, source_y, x, y, patch_radius);
        if luminance_score.is_finite() {
            gradient_score * 0.75 + luminance_score * 0.25
        } else {
            gradient_score
        }
    };
    let subpixel_offset = |negative: f64, center: f64, positive: f64| {
        if !negative.is_finite() || !center.is_finite() || !positive.is_finite() {
            return 0.0;
        }
        let denominator = negative - 2.0 * center + positive;
        if denominator.abs() < 1e-8 {
            0.0
        } else {
            (0.5 * (negative - positive) / denominator).clamp(-1.0, 1.0)
        }
    };
    let subpixel_x = subpixel_offset(
        sample(best_x - 1, best_y),
        best_score,
        sample(best_x + 1, best_y),
    );
    let subpixel_y = subpixel_offset(
        sample(best_x, best_y - 1),
        best_score,
        sample(best_x, best_y + 1),
    );
    Some((best_x, best_y, subpixel_x, subpixel_y))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_patch_position(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    source_x: i32,
    source_y: i32,
    target_x: i32,
    target_y: i32,
    patch_radius: i32,
    search_radius: i32,
) -> Option<(i32, i32, f64, f64)> {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_target = None;

    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let candidate_x = target_x + dx;
            let candidate_y = target_y + dy;
            let score = patch_ncc(
                image1,
                image2,
                source_x,
                source_y,
                candidate_x,
                candidate_y,
                patch_radius,
            );
            if score > best_score {
                best_score = score;
                best_target = Some((candidate_x, candidate_y));
            }
        }
    }

    let (best_x, best_y) = best_target?;
    if !best_score.is_finite() {
        return None;
    }
    let subpixel_offset = |negative: f64, center: f64, positive: f64| {
        if !negative.is_finite() || !center.is_finite() || !positive.is_finite() {
            return 0.0;
        }
        let denominator = negative - 2.0 * center + positive;
        if denominator.abs() < 1e-8 {
            0.0
        } else {
            (0.5 * (negative - positive) / denominator).clamp(-1.0, 1.0)
        }
    };
    let sample = |x, y| patch_ncc(image1, image2, source_x, source_y, x, y, patch_radius);
    let subpixel_x = subpixel_offset(
        sample(best_x - 1, best_y),
        best_score,
        sample(best_x + 1, best_y),
    );
    let subpixel_y = subpixel_offset(
        sample(best_x, best_y - 1),
        best_score,
        sample(best_x, best_y + 1),
    );
    Some((best_x, best_y, subpixel_x, subpixel_y))
}

pub(crate) fn patch_ncc(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    center1_x: i32,
    center1_y: i32,
    center2_x: i32,
    center2_y: i32,
    radius: i32,
) -> f64 {
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum_squares1 = 0.0;
    let mut sum_squares2 = 0.0;
    let mut sum_products = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some(value1) = image1.luma_at(center1_x + dx, center1_y + dy) else {
                return f64::NEG_INFINITY;
            };
            let Some(value2) = image2.luma_at(center2_x + dx, center2_y + dy) else {
                return f64::NEG_INFINITY;
            };
            sum1 += value1;
            sum2 += value2;
            sum_squares1 += value1 * value1;
            sum_squares2 += value2 * value2;
            sum_products += value1 * value2;
            sample_count += 1;
        }
    }

    let sample_count = sample_count as f64;
    let covariance = sum_products - sum1 * sum2 / sample_count;
    let variance1 = sum_squares1 - sum1 * sum1 / sample_count;
    let variance2 = sum_squares2 - sum2 * sum2 / sample_count;
    if variance1 <= f64::EPSILON || variance2 <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        covariance / (variance1 * variance2).sqrt()
    }
}

pub(crate) fn refine_homography_inliers(
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

pub(crate) fn retain_model_inliers(
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

pub(crate) fn symmetric_point_error(
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

pub(crate) fn symmetric_reprojection_rmse(
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
