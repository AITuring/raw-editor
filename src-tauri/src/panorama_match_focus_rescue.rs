//! Focus-specific post-geometry rescue and overlap decisions.

use nalgebra::Point2;
use std::path::Path;

use super::match_geometry::MatchGeometry;
use super::match_result::build_match_info;
use super::{
    BlendMode, ImageInfo, MIXED_FOCAL_MAX_MEDIAN_ERROR, MIXED_FOCAL_MIN_INLIERS, MatchInfo,
    estimate_mixed_focal_coarse_registration, focus_adjacent_same_focal_pair,
    focus_overlap_is_verified, focus_overlap_quality, median_symmetric_error,
    panorama_transform_overlap_support, shifted_focus_structure_is_verified,
    trailing_capture_number, transformed_point,
};

pub(super) enum FocusRescueResult {
    Rejected,
    Rescued(MatchInfo),
    Accepted {
        geometry: MatchGeometry,
        inlier_count: usize,
    },
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_focus_rescue(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    geometry: MatchGeometry,
    mixed_focal_pair: bool,
    blend_mode: BlendMode,
    stable_four_point_solver: bool,
    allow_low_texture_boundary: bool,
    log_match: bool,
) -> FocusRescueResult {
    let MatchGeometry {
        homography,
        inlier_points,
        projected_match_points,
        top_candidate_points,
        dense_focus_points,
        foreground_feature_points,
        mixed_focal_fallback_used,
    } = geometry;
    let inlier_count = inlier_points.len();
    let median_error = median_symmetric_error(&homography, &inlier_points);
    let focus_overlap_quality = (blend_mode == BlendMode::FocusStack && stable_four_point_solver)
        .then(|| focus_overlap_quality(source_image, target_image, &homography))
        .flatten();
    let focus_overlap_verified = focus_overlap_quality.is_some_and(|quality| {
        focus_overlap_is_verified(
            quality,
            inlier_count,
            median_error,
            allow_low_texture_boundary,
        )
    }) || (blend_mode == BlendMode::FocusStack
        && stable_four_point_solver
        && !mixed_focal_pair
        && shifted_focus_structure_is_verified(
            source_image,
            target_image,
            &homography,
            &inlier_points,
            log_match,
        ));
    if log_match
        && let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) = focus_overlap_quality
    {
        println!(
            "  - Focus overlap validation: intensity NCC {:.3}, edge NCC {:.3}, edge orientation {:.3} ({} samples, {})",
            intensity_ncc,
            edge_ncc,
            edge_orientation,
            samples,
            if focus_overlap_verified {
                "verified"
            } else {
                "rejected"
            }
        );
    }
    if mixed_focal_pair
        && blend_mode == BlendMode::FocusStack
        && (inlier_count < MIXED_FOCAL_MIN_INLIERS
            || !median_error.is_finite()
            || median_error > MIXED_FOCAL_MAX_MEDIAN_ERROR
            || (stable_four_point_solver && !focus_overlap_verified))
    {
        if mixed_focal_fallback_used && focus_overlap_verified {
            // Detector coordinates are heavily quantized when a small
            // overview is matched to a long-lens frame. The focal-constrained
            // similarity can therefore miss the full-resolution point-error
            // cutoff while the actual warped luminance and edge structure are
            // clearly the same region. Preserve that image-verified pose as a
            // graph-only bridge; its noisy points must not enter bundle
            // adjustment.
            if log_match {
                println!(
                    "  - Accepting image-verified mixed-focal graph bridge despite {:.3}px quantized point error",
                    median_error
                );
            }
            return FocusRescueResult::Rescued(build_match_info(
                homography,
                inlier_points.clone(),
                projected_match_points,
                top_candidate_points,
                dense_focus_points,
                foreground_feature_points,
                true,
            ));
        }
        if let (Some(source_focal), Some(target_focal)) = (
            source_image.focal_length_35mm,
            target_image.focal_length_35mm,
        ) && let Some((coarse_homography, coarse_points, coarse_score)) =
            estimate_mixed_focal_coarse_registration(
                source_image,
                target_image,
                source_focal,
                target_focal,
                false,
                log_match,
            )
        {
            if log_match {
                println!(
                    "  - Mixed-focal coarse image registration recovered {} points (NCC {:.3})",
                    coarse_points.len(),
                    coarse_score
                );
            }
            return FocusRescueResult::Rescued(build_match_info(
                coarse_homography,
                coarse_points.clone(),
                coarse_points,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                true,
            ));
        }
        if log_match {
            println!(
                "  - Rejecting weak mixed-focal bridge: {} inliers, {:.3}px median symmetric error (need at least {} and <= {:.2}px)",
                inlier_count, median_error, MIXED_FOCAL_MIN_INLIERS, MIXED_FOCAL_MAX_MEDIAN_ERROR
            );
        }
        return FocusRescueResult::Rejected;
    }
    if blend_mode == BlendMode::FocusStack && stable_four_point_solver && !focus_overlap_verified {
        if focus_adjacent_same_focal_pair(source_image, target_image)
            && let Some(match_info) =
                recover_adjacent_focus_bracket_match(source_image, target_image, log_match)
        {
            if log_match {
                println!(
                    "  - Focus overlap check failed; recovered same-pose adjacent bracket instead"
                );
            }
            return FocusRescueResult::Rescued(match_info);
        }
        if log_match {
            println!(
                "  - Rejecting focus match: the proposed overlap does not preserve the source structure"
            );
        }
        return FocusRescueResult::Rejected;
    }
    FocusRescueResult::Accepted {
        geometry: MatchGeometry {
            homography,
            inlier_points,
            projected_match_points,
            top_candidate_points,
            dense_focus_points,
            foreground_feature_points,
            mixed_focal_fallback_used,
        },
        inlier_count,
    }
}

fn recover_adjacent_focus_bracket_match(
    source: &ImageInfo,
    target: &ImageInfo,
    log_match: bool,
) -> Option<MatchInfo> {
    let adjacent_capture = trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .is_some_and(|(left, right)| left.abs_diff(right) == 1);
    if !adjacent_capture {
        return None;
    }
    let compatible_focal_length = source
        .focal_length_35mm
        .zip(target.focal_length_35mm)
        .map(|(left, right)| {
            left.is_finite()
                && right.is_finite()
                && left > 0.0
                && right > 0.0
                && (left / right).max(right / left) <= 1.03
        })
        // Missing EXIF must not disable focus-bracket recovery. The strict
        // same-pose checks below remain the authority.
        .unwrap_or(true);
    if !compatible_focal_length {
        return None;
    }
    let source_focal = source
        .focal_length_35mm
        .or(target.focal_length_35mm)
        .unwrap_or(50.0);
    let target_focal = target
        .focal_length_35mm
        .or(source.focal_length_35mm)
        .unwrap_or(source_focal);
    let (homography, points, score) = estimate_mixed_focal_coarse_registration(
        source,
        target,
        source_focal,
        target_focal,
        true,
        log_match,
    )?;
    // This fallback is only allowed to attach another focal plane at the same
    // camera pose. A spatial scan step or repeated motif must continue through
    // the ordinary feature/cycle-verified panorama graph.
    let scale = homography[(0, 0)].abs();
    let source_center = Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5);
    let target_center = Point2::new(target.width as f64 * 0.5, target.height as f64 * 0.5);
    let mapped_center = transformed_point(&homography, source_center)?;
    let normalization = source
        .width
        .max(source.height)
        .max(target.width.max(target.height))
        .max(1) as f64;
    let center_motion = (mapped_center - target_center).norm() / normalization;
    let overlap =
        panorama_transform_overlap_support(&homography, source.dimensions(), target.dimensions());
    // The overlap estimator deliberately ignores an outer validation margin;
    // identical full-frame images therefore report about 0.64, not 1.0.
    if !(0.88..=1.12).contains(&scale) || center_motion > 0.12 || overlap < 0.60 {
        if log_match {
            println!(
                "  - Rejecting adjacent focus-layer recovery: scale {scale:.3}, motion {:.3}%, overlap {:.1}%",
                center_motion * 100.0,
                overlap * 100.0,
            );
        }
        return None;
    }
    if log_match {
        println!(
            "  - Recovered same-pose adjacent focus layer: '{}' <-> '{}' (NCC {:.3}, motion {:.3}%, overlap {:.1}%)",
            Path::new(&source.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            score,
            center_motion * 100.0,
            overlap * 100.0,
        );
    }
    Some(build_match_info(
        homography,
        points.clone(),
        points,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        true,
    ))
}
