//! Pairwise image matching.
//!
//! Matches one image pair under the selected alignment mode, with the coarse
//! registration used for mixed focal lengths.

use crate::panorama_utils::stitching::Projection;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::path::Path;

use super::{
    AlignmentMode, BlendMode, ImageInfo, MIXED_FOCAL_COARSE_MIN_EDGE_NCC,
    MIXED_FOCAL_COARSE_MIN_EDGE_ORIENTATION, MIXED_FOCAL_COARSE_MIN_INTENSITY_NCC,
    MIXED_FOCAL_MIN_INLIERS, Match, MatchCandidates, MatchInfo, build_match_candidates,
    build_match_info, focus_overlap_quality, match_focus_rescue, match_geometry,
    symmetric_reprojection_rmse, transformed_point,
};

pub(super) fn match_image_pair(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    projection: Projection,
    blend_mode: BlendMode,
    alignment_mode: AlignmentMode,
    stable_four_point_solver: bool,
    log_match: bool,
    allow_low_texture_boundary: bool,
) -> Option<MatchInfo> {
    match_image_pair_with_seed(
        source_image,
        target_image,
        projection,
        blend_mode,
        alignment_mode,
        stable_four_point_solver,
        log_match,
        allow_low_texture_boundary,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn match_image_pair_with_seed(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    projection: Projection,
    blend_mode: BlendMode,
    alignment_mode: AlignmentMode,
    stable_four_point_solver: bool,
    log_match: bool,
    allow_low_texture_boundary: bool,
    seeded_initial_matches: Option<Vec<Match>>,
) -> Option<MatchInfo> {
    let candidates = build_match_candidates(
        source_image,
        target_image,
        blend_mode,
        stable_four_point_solver,
        log_match,
        seeded_initial_matches,
    );
    let MatchCandidates {
        initial_matches,
        mixed_focal_pair,
        minimum_inliers,
        adjacent_same_focal_focus_pair,
    } = candidates;
    let features1 = &source_image.features;
    let features2 = &target_image.features;
    // Filename adjacency is only a candidate-order hint. It must not enable
    // a coarse transform during pair matching: with shuffled imports, an
    // arbitrary numeric suffix can otherwise turn two scan positions into a
    // false focus-layer edge before geometry has been established.
    let severe_focus_feature_imbalance = features1.len().min(features2.len()).saturating_mul(4)
        < features1.len().max(features2.len());
    if adjacent_same_focal_focus_pair
        && severe_focus_feature_imbalance
        && let Some((homography, points, score)) = {
            let source_focal = source_image
                .focal_length_35mm
                .or(target_image.focal_length_35mm)
                .unwrap_or(50.0);
            let target_focal = target_image
                .focal_length_35mm
                .or(source_image.focal_length_35mm)
                .unwrap_or(source_focal);
            estimate_mixed_focal_coarse_registration(
                source_image,
                target_image,
                source_focal,
                target_focal,
                true,
                log_match,
            )
        }
    {
        if log_match {
            println!(
                "  - Defocused adjacent bracket registration recovered {} points (NCC {:.3})",
                points.len(),
                score
            );
        }
        return Some(build_match_info(
            homography,
            points.clone(),
            points,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            true,
        ));
    }
    if initial_matches.len() < minimum_inliers {
        // A focus bracket can contain one deliberately defocused endpoint
        // whose local descriptors collapse even though the frame has the same
        // camera pose as its immediate neighbour. Photoshop can still align
        // these pairs from their global image structure. Give only adjacent,
        // same-focal captures that bounded fallback: the coarse registrar
        // validates luminance and edges before returning a transform, while
        // the filename/focal gates prevent it from connecting unrelated scan
        // tiles merely because they contain repeated artwork texture.
        if adjacent_same_focal_focus_pair
            && let Some((homography, points, score)) = {
                let source_focal = source_image
                    .focal_length_35mm
                    .or(target_image.focal_length_35mm)
                    .unwrap_or(50.0);
                let target_focal = target_image
                    .focal_length_35mm
                    .or(source_image.focal_length_35mm)
                    .unwrap_or(source_focal);
                estimate_mixed_focal_coarse_registration(
                    source_image,
                    target_image,
                    source_focal,
                    target_focal,
                    true,
                    log_match,
                )
            }
        {
            if log_match {
                println!(
                    "  - Adjacent focus-bracket image registration recovered {} points (NCC {:.3})",
                    points.len(),
                    score
                );
            }
            return Some(build_match_info(
                homography,
                points.clone(),
                points,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                true,
            ));
        }
        if log_match && mixed_focal_pair {
            println!(
                "  - Rejecting mixed-focal pair before geometry: {} descriptor candidates (need {})",
                initial_matches.len(),
                minimum_inliers
            );
        }
        return None;
    }

    let geometry = match match_geometry::verify_match_geometry(
        source_image,
        target_image,
        projection,
        blend_mode,
        alignment_mode,
        stable_four_point_solver,
        log_match,
        allow_low_texture_boundary,
        mixed_focal_pair,
        minimum_inliers,
        &initial_matches,
    ) {
        None => return None,
        Some(match_geometry::MatchGeometryResult::CoarseBridge { homography, points }) => {
            return Some(build_match_info(
                homography,
                points.clone(),
                points,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                true,
            ));
        }
        Some(match_geometry::MatchGeometryResult::Verified(geometry)) => geometry,
    };
    let (geometry, inlier_count) = match match_focus_rescue::apply_focus_rescue(
        source_image,
        target_image,
        geometry,
        mixed_focal_pair,
        blend_mode,
        stable_four_point_solver,
        allow_low_texture_boundary,
        log_match,
    ) {
        match_focus_rescue::FocusRescueResult::Rejected => return None,
        match_focus_rescue::FocusRescueResult::Rescued(match_info) => return Some(match_info),
        match_focus_rescue::FocusRescueResult::Accepted {
            geometry,
            inlier_count,
        } => (geometry, inlier_count),
    };
    let match_geometry::MatchGeometry {
        homography,
        inlier_points,
        projected_match_points,
        top_candidate_points,
        dense_focus_points,
        foreground_feature_points,
        ..
    } = geometry;
    if log_match {
        println!(
            "  - Good match found: '{}' <-> '{}' ({} inliers)",
            Path::new(&source_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            inlier_count
        );
        let reprojection_error = symmetric_reprojection_rmse(&homography, &inlier_points);
        println!(
            "  - Refined match: '{}' <-> '{}' ({} inliers, {:.3}px symmetric RMS)",
            Path::new(&source_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            inlier_count,
            reprojection_error
        );
    }
    Some(build_match_info(
        homography,
        inlier_points,
        projected_match_points,
        top_candidate_points,
        dense_focus_points,
        foreground_feature_points,
        false,
    ))
}

pub(super) fn estimate_mixed_focal_coarse_registration(
    source: &ImageInfo,
    target: &ImageInfo,
    source_focal: f64,
    target_focal: f64,
    allow_adjacent_defocus: bool,
    log_match: bool,
) -> Option<(Matrix3<f64>, Vec<(Point2<f64>, Point2<f64>)>, f64)> {
    if !source_focal.is_finite()
        || !target_focal.is_finite()
        || source_focal <= 0.0
        || target_focal <= 0.0
    {
        return None;
    }
    let focal_scale = target_focal / source_focal;
    if !focal_scale.is_finite() || focal_scale <= 0.0 {
        return None;
    }

    // Work at a deliberately small common resolution. The source is resized
    // by the metadata-predicted focal ratio before the translation search, so
    // this is a real coarse registration rather than a same-size image hash.
    // It recovers a close-up inside a wider frame even when BRIEF selects a
    // repeated character as its only cross-lens consensus.
    let target_alignment_width = target.alignment_image.width().max(1) as f64;
    let target_alignment_height = target.alignment_image.height().max(1) as f64;
    let target_low_width = 128u32.min(target.alignment_image.width().max(32));
    let target_low_height = ((target_alignment_height / target_alignment_width)
        * target_low_width as f64)
        .round()
        .clamp(32.0, 128.0) as u32;
    let target_low = image::imageops::resize(
        &target.alignment_image,
        target_low_width,
        target_low_height,
        image::imageops::FilterType::Triangle,
    );
    let target_low_scale = target_low_width as f64 / target_alignment_width;
    // EXIF focal ratio is a useful prior, but a handheld capture can also
    // change camera distance while zooming. Search a bounded scale band around
    // that prior; otherwise a valid close-up in a wide overview is rejected by
    // the fixed-ratio translation search before dense validation sees it.
    let focal_ratio = (source_focal / target_focal).max(target_focal / source_focal);
    let scale_candidates = if focal_ratio <= 1.03 {
        // Same lens module: do not let EXIF noise expand the search to 1.08/1.20
        // scales that mimic a false lens switch on dark, low-texture artwork.
        [0.94_f64, 0.97, 1.0, 1.03, 1.06]
            .into_iter()
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .collect::<Vec<_>>()
    } else {
        [0.70, 0.82, 0.94, 1.0, 1.08, 1.20, 1.34]
            .into_iter()
            .map(|adjustment| focal_scale * adjustment)
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .collect::<Vec<_>>()
    };
    let mut best: Option<(i32, i32, f64, f64)> = None;
    let mut candidate_scores: Vec<(i32, i32, f64, f64)> = Vec::new();

    for scale in scale_candidates {
        let source_alignment_scale =
            scale * source.scale_factor / target.scale_factor.max(f64::EPSILON);
        let source_low_width =
            (source.alignment_image.width() as f64 * source_alignment_scale * target_low_scale)
                .round()
                .clamp(24.0, 360.0) as u32;
        let source_low_height =
            (source.alignment_image.height() as f64 * source_alignment_scale * target_low_scale)
                .round()
                .clamp(24.0, 280.0) as u32;
        let source_low = image::imageops::resize(
            &source.alignment_image,
            source_low_width,
            source_low_height,
            image::imageops::FilterType::Triangle,
        );
        let minimum_overlap_width = (source_low.width().min(target_low.width()) as f64 * 0.30)
            .round()
            .max(12.0) as u32;
        let minimum_overlap_height = (source_low.height().min(target_low.height()) as f64 * 0.30)
            .round()
            .max(12.0) as u32;
        let x_min = -(source_low.width() as i32 - minimum_overlap_width as i32);
        let x_max = target_low.width() as i32 - minimum_overlap_width as i32;
        let y_min = -(source_low.height() as i32 - minimum_overlap_height as i32);
        let y_max = target_low.height() as i32 - minimum_overlap_height as i32;

        for offset_y in y_min..=y_max {
            for offset_x in x_min..=x_max {
                let source_start_x = offset_x.max(0) as u32;
                let source_start_y = offset_y.max(0) as u32;
                let target_start_x = (-offset_x).max(0) as u32;
                let target_start_y = (-offset_y).max(0) as u32;
                if source_start_x >= source_low.width()
                    || source_start_y >= source_low.height()
                    || target_start_x >= target_low.width()
                    || target_start_y >= target_low.height()
                {
                    continue;
                }
                let overlap_width =
                    (source_low.width() - source_start_x).min(target_low.width() - target_start_x);
                let overlap_height = (source_low.height() - source_start_y)
                    .min(target_low.height() - target_start_y);
                if overlap_width < minimum_overlap_width || overlap_height < minimum_overlap_height
                {
                    continue;
                }
                let stride = 2u32;
                let mut values = Vec::new();
                let mut target_values = Vec::new();
                for y in (0..overlap_height).step_by(stride as usize) {
                    for x in (0..overlap_width).step_by(stride as usize) {
                        values.push(f64::from(
                            source_low.get_pixel(source_start_x + x, source_start_y + y)[0],
                        ));
                        target_values.push(f64::from(
                            target_low.get_pixel(target_start_x + x, target_start_y + y)[0],
                        ));
                    }
                }
                if values.len() < 120 {
                    continue;
                }
                let mean_source = values.iter().sum::<f64>() / values.len() as f64;
                let mean_target = target_values.iter().sum::<f64>() / values.len() as f64;
                let mut covariance = 0.0;
                let mut source_variance = 0.0;
                let mut target_variance = 0.0;
                for (source_value, target_value) in values.iter().zip(target_values.iter()) {
                    let source_delta = source_value - mean_source;
                    let target_delta = target_value - mean_target;
                    covariance += source_delta * target_delta;
                    source_variance += source_delta * source_delta;
                    target_variance += target_delta * target_delta;
                }
                if source_variance <= f64::EPSILON || target_variance <= f64::EPSILON {
                    continue;
                }
                let score = covariance / (source_variance * target_variance).sqrt();
                if !score.is_finite() {
                    continue;
                }
                candidate_scores.push((offset_x, offset_y, scale, score));
                if best
                    .as_ref()
                    .is_none_or(|(_, _, _, best_score)| score > *best_score)
                {
                    best = Some((offset_x, offset_y, scale, score));
                }
            }
        }
    }

    let (best_offset_x, best_offset_y, best_scale, best_score) = best?;
    let alternate = candidate_scores
        .iter()
        .filter(|(candidate_x, candidate_y, candidate_scale, _)| {
            (*candidate_x - best_offset_x).abs() >= 8
                || (*candidate_y - best_offset_y).abs() >= 8
                || (*candidate_scale - best_scale).abs() >= best_scale * 0.08
        })
        .map(|(_, _, _, candidate_score)| *candidate_score)
        .fold(f64::NEG_INFINITY, f64::max);
    if log_match {
        println!(
            "  - Mixed-focal coarse search best offset=({}, {}), NCC {:.3}, margin {:.3}",
            best_offset_x,
            best_offset_y,
            best_score,
            best_score - alternate
        );
    }
    if best_score < 0.40 {
        return None;
    }
    // The best low-resolution luminance peak is not necessarily the best
    // geometric peak: a repeated stroke can win the cheap search while a
    // weaker peak is the actual shared patch. Re-score a small set of
    // spatially separated peaks against the denser source/target images.
    let mut ranked_candidates = candidate_scores;
    ranked_candidates.sort_by(|left, right| right.3.total_cmp(&left.3));
    let mut selected = None;
    let mut best_overlap = None;
    let mut tried: Vec<(i32, i32, f64)> = Vec::new();
    for (offset_x, offset_y, scale, score) in ranked_candidates {
        if score < (best_score - 0.16).max(0.40) || tried.len() >= 16 {
            break;
        }
        if tried
            .iter()
            .any(|(previous_x, previous_y, previous_scale)| {
                (offset_x - *previous_x).abs() < 8
                    && (offset_y - *previous_y).abs() < 8
                    && (scale - *previous_scale).abs() < scale * 0.08
            })
        {
            continue;
        }
        tried.push((offset_x, offset_y, scale));
        // `offset_*` above is expressed as the crop displacement applied to
        // the scaled source. A positive source crop aligns with target zero,
        // so the source-to-target transform has the opposite translation.
        // Using the same sign validates a mirrored search pose and was the
        // reason genuine overview/close-up bridges never survived the dense
        // overlap check.
        let tx_alignment = -offset_x as f64 / target_low_scale;
        let ty_alignment = -offset_y as f64 / target_low_scale;
        let tx_full = tx_alignment * target.scale_factor;
        let ty_full = ty_alignment * target.scale_factor;
        let homography = Matrix3::new(scale, 0.0, tx_full, 0.0, scale, ty_full, 0.0, 0.0, 1.0);
        let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) =
            focus_overlap_quality(source, target, &homography)
        else {
            continue;
        };
        let validation_score = intensity_ncc * 0.45 + edge_ncc * 0.35 + edge_orientation * 0.20;
        if best_overlap
            .as_ref()
            .is_none_or(|(_, _, _, _, _, best_validation)| validation_score > *best_validation)
        {
            best_overlap = Some((
                offset_x,
                offset_y,
                intensity_ncc,
                edge_ncc,
                edge_orientation,
                validation_score,
            ));
        }
        let structurally_verified = if allow_adjacent_defocus {
            // A heavily defocused bracket endpoint preserves coarse luminance
            // and edge magnitude but not reliable gradient orientation. This
            // relaxed profile is reachable only for adjacent, same-focal
            // captures at the caller and still requires both image channels.
            (score >= 0.60
                && intensity_ncc >= 0.30
                && edge_ncc >= -0.10
                && edge_orientation >= -0.10)
                || (intensity_ncc >= 0.60 && edge_ncc >= 0.20 && edge_orientation >= 0.0)
                || (intensity_ncc >= 0.85 && edge_ncc >= 0.05 && edge_orientation >= 0.10)
        } else {
            intensity_ncc >= MIXED_FOCAL_COARSE_MIN_INTENSITY_NCC
                && edge_ncc >= MIXED_FOCAL_COARSE_MIN_EDGE_NCC
                && edge_orientation >= MIXED_FOCAL_COARSE_MIN_EDGE_ORIENTATION
        };
        if !structurally_verified {
            continue;
        }
        if selected
            .as_ref()
            .is_none_or(|(_, _, _, _, _, _, _, _, _, best_validation)| {
                validation_score > *best_validation
            })
        {
            selected = Some((
                offset_x,
                offset_y,
                scale,
                score,
                homography,
                intensity_ncc,
                edge_ncc,
                edge_orientation,
                samples,
                validation_score,
            ));
        }
    }
    let Some((_, _, _, score, homography, intensity_ncc, edge_ncc, edge_orientation, samples, _)) =
        selected
    else {
        if log_match {
            if let Some((offset_x, offset_y, intensity_ncc, edge_ncc, edge_orientation, _)) =
                best_overlap
            {
                println!(
                    "  - Rejecting mixed-focal coarse bridge: best structural candidate at ({offset_x}, {offset_y}) had intensity NCC {intensity_ncc:.3}, edge NCC {edge_ncc:.3}, edge orientation {edge_orientation:.3}"
                );
            } else {
                println!(
                    "  - Rejecting mixed-focal coarse bridge: overlap structure is not verified"
                );
            }
        }
        return None;
    };
    if log_match {
        println!(
            "  - Mixed-focal coarse overlap validation: intensity NCC {:.3}, edge NCC {:.3}, edge orientation {:.3} ({} samples)",
            intensity_ncc, edge_ncc, edge_orientation, samples
        );
    }
    let mut points = Vec::new();
    for row in 1..=6 {
        for column in 1..=8 {
            let source_point = Point2::new(
                source.width as f64 * column as f64 / 9.0,
                source.height as f64 * row as f64 / 7.0,
            );
            let Some(target_point) = transformed_point(&homography, source_point) else {
                continue;
            };
            if target_point.x >= 0.0
                && target_point.y >= 0.0
                && target_point.x < target.width as f64
                && target_point.y < target.height as f64
            {
                points.push((source_point, target_point));
            }
        }
    }
    // A narrow same-focal scan strip can cover only a handful of the coarse
    // grid cells.  It is still independently validated by the dense
    // luminance/edge checks above; do not discard it solely because the
    // synthetic grid has fewer than the cross-lens minimum correspondences.
    let minimum_points = if allow_adjacent_defocus && focal_ratio <= 1.03 {
        // A very narrow but well-correlated bracket boundary may expose no
        // coarse grid cell after the transform is scaled back to the source
        // dimensions. The dense overlap validator remains the evidence gate;
        // these points are optional and only carry the recovered placement.
        0
    } else {
        MIXED_FOCAL_MIN_INLIERS
    };
    (points.len() >= minimum_points).then_some((homography, points, score))
}
