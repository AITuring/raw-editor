//! Geometric verification for one descriptor candidate set.

use nalgebra::{Matrix3, Point2};

use crate::panorama_utils::processing;
use crate::panorama_utils::registration;
use crate::panorama_utils::stitching::{Projection, project_point};

use super::{
    AlignmentMode, BlendMode, FOCUS_MODEL_INLIER_THRESHOLD, FULL_RES_RANSAC_INLIER_THRESHOLD,
    FULL_RES_REFINEMENT_THRESHOLD, ImageInfo, MIXED_FOCAL_COARSE_MIN_EDGE_NCC,
    MIXED_FOCAL_COARSE_MIN_EDGE_ORIENTATION, MIXED_FOCAL_COARSE_MIN_INTENSITY_NCC,
    MIXED_FOCAL_SIMILARITY_RANSAC_THRESHOLD, Match, PANORAMA_MODEL_INLIER_THRESHOLD,
    collect_dense_focus_region_points, collect_foreground_feature_points,
    estimate_mixed_focal_coarse_registration, estimate_similarity, estimate_translation,
    find_mixed_focal_similarity_ransac, focus_alignment_keypoint_is_foreground,
    focus_overlap_is_verified, focus_overlap_quality, match_has_spatial_support,
    median_symmetric_error, mixed_focal_scale_is_plausible, panorama_model_fit,
    refine_homography_inliers, refine_match_point_from_homography, retain_model_inliers,
    select_focus_stack_transform, select_large_panorama_transform,
    shifted_focus_structure_is_verified, symmetric_inlier_indices,
};

#[derive(Debug)]
pub(super) struct MatchGeometry {
    pub(super) homography: Matrix3<f64>,
    pub(super) inlier_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub(super) projected_match_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub(super) top_candidate_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub(super) dense_focus_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub(super) foreground_feature_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub(super) mixed_focal_fallback_used: bool,
}

#[derive(Debug)]
pub(super) enum MatchGeometryResult {
    CoarseBridge {
        homography: Matrix3<f64>,
        points: Vec<(Point2<f64>, Point2<f64>)>,
    },
    Verified(MatchGeometry),
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_match_geometry(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    projection: Projection,
    blend_mode: BlendMode,
    alignment_mode: AlignmentMode,
    stable_four_point_solver: bool,
    log_match: bool,
    allow_low_texture_boundary: bool,
    mixed_focal_pair: bool,
    minimum_inliers: usize,
    initial_matches: &[Match],
) -> Option<MatchGeometryResult> {
    let features1 = &source_image.features;
    let features2 = &target_image.features;
    let keypoints1 = features1
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let keypoints2 = features2
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let projected_points1 = keypoints1
        .iter()
        .map(|point| {
            project_point(
                source_image,
                point.x as f64 * source_image.scale_factor,
                point.y as f64 * source_image.scale_factor,
                projection,
            )
            .expect("projection should produce finite feature coordinates")
        })
        .collect::<Vec<_>>();
    let projected_points2 = keypoints2
        .iter()
        .map(|point| {
            project_point(
                target_image,
                point.x as f64 * target_image.scale_factor,
                point.y as f64 * target_image.scale_factor,
                projection,
            )
            .expect("projection should produce finite feature coordinates")
        })
        .collect::<Vec<_>>();
    let projected_match_points = initial_matches
        .iter()
        .map(|matched| {
            (
                projected_points1[matched.index1],
                projected_points2[matched.index2],
            )
        })
        .collect::<Vec<_>>();
    let background_match_indices = if blend_mode == BlendMode::FocusStack
        && (source_image.foreground_range.is_some() || target_image.foreground_range.is_some())
    {
        initial_matches
            .iter()
            .enumerate()
            .filter_map(|(index, matched)| {
                (!focus_alignment_keypoint_is_foreground(source_image, keypoints1[matched.index1])
                    && !focus_alignment_keypoint_is_foreground(
                        target_image,
                        keypoints2[matched.index2],
                    ))
                .then_some(index)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    // A depth-discontinuous object can dominate a local descriptor match (the
    // bright holder in this fixture is one example). Solve the global camera
    // pose from the paper/background whenever there is enough background
    // support, then let the optional regional model handle the object itself.
    // If a pair contains too little background, retain the old all-point solver
    // so unusual stacks do not become disconnected merely because an object
    // detector fired.
    let solver_indices = if background_match_indices.len() >= minimum_inliers {
        background_match_indices
    } else {
        (0..projected_match_points.len()).collect::<Vec<_>>()
    };
    let solver_points = solver_indices
        .iter()
        .map(|&index| projected_match_points[index])
        .collect::<Vec<_>>();
    let top_matches =
        processing::match_features(&source_image.top_features, &target_image.top_features);
    let top_candidate_points = top_matches
        .iter()
        .filter_map(|matched| {
            let source = source_image.top_features.get(matched.index1)?.keypoint;
            let target = target_image.top_features.get(matched.index2)?.keypoint;
            Some((
                project_point(
                    source_image,
                    source.x as f64 * source_image.scale_factor,
                    source.y as f64 * source_image.scale_factor,
                    projection,
                )?,
                project_point(
                    target_image,
                    target.x as f64 * target_image.scale_factor,
                    target.y as f64 * target_image.scale_factor,
                    projection,
                )?,
            ))
        })
        .collect::<Vec<_>>();
    // The detector works in analysis pixels. The 12px native-image gate is
    // smaller than two quantized detector pixels on 200MP phone photographs.
    let seed_threshold = FULL_RES_RANSAC_INLIER_THRESHOLD
        .max(source_image.scale_factor.max(target_image.scale_factor) * 3.5);
    let mut solver_indices_used = solver_indices.clone();
    let ransac_result = if stable_four_point_solver {
        processing::find_homography_ransac_points_stable_with_min_inliers(
            &solver_points,
            seed_threshold,
            minimum_inliers,
        )
    } else {
        processing::find_homography_ransac_points(&solver_points, seed_threshold)
    };
    let mut mixed_focal_fallback_used = false;
    let ransac_result = ransac_result.or_else(|| {
        if !(mixed_focal_pair && projection == Projection::Planar) {
            return None;
        }
        let (Some(source_focal), Some(target_focal)) = (
            source_image.focal_length_35mm,
            target_image.focal_length_35mm,
        ) else {
            return None;
        };
        let fallback = find_mixed_focal_similarity_ransac(
            &solver_points,
            source_focal,
            target_focal,
            MIXED_FOCAL_SIMILARITY_RANSAC_THRESHOLD,
            minimum_inliers,
        )
        .or_else(|| {
            // Foreground masking is normally valuable for focus stacks, but a
            // lens switch can leave the paper/object boundary outside one
            // mask. Retry the complete candidate set before declaring the
            // focal-length component disconnected.
            if solver_indices.len() == projected_match_points.len() {
                return None;
            }
            solver_indices_used = (0..projected_match_points.len()).collect();
            find_mixed_focal_similarity_ransac(
                &projected_match_points,
                source_focal,
                target_focal,
                MIXED_FOCAL_SIMILARITY_RANSAC_THRESHOLD,
                minimum_inliers,
            )
        });
        if log_match && fallback.is_some() {
            mixed_focal_fallback_used = true;
            println!(
                "  - Mixed-focal similarity fallback recovered a geometric bridge ({} inliers)",
                fallback.as_ref().map_or(0, |(_, indices)| indices.len())
            );
        }
        fallback
    });
    let Some((projected_homography, projected_inlier_indices)) = ransac_result else {
        if mixed_focal_pair
            && projection == Projection::Planar
            && let (Some(source_focal), Some(target_focal)) = (
                source_image.focal_length_35mm,
                target_image.focal_length_35mm,
            )
            && let Some((coarse_homography, coarse_points, coarse_score)) =
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
            return Some(MatchGeometryResult::CoarseBridge {
                homography: coarse_homography,
                points: coarse_points,
            });
        }
        if log_match && mixed_focal_pair {
            println!(
                "  - Rejecting mixed-focal pair after RANSAC: {} descriptor candidates",
                initial_matches.len()
            );
        }
        return None;
    };
    if mixed_focal_pair
        && projection == Projection::Planar
        && let (Some(source_focal), Some(target_focal)) = (
            source_image.focal_length_35mm,
            target_image.focal_length_35mm,
        )
        && !mixed_focal_scale_is_plausible(&projected_homography, source_focal, target_focal)
    {
        return None;
    }
    let projected_inlier_indices = projected_inlier_indices
        .into_iter()
        .map(|index| solver_indices_used[index])
        .collect::<Vec<_>>();
    if mixed_focal_pair
        && projection == Projection::Planar
        && (source_image.foreground_range.is_some() || target_image.foreground_range.is_some())
    {
        let foreground_support = projected_inlier_indices
            .iter()
            .filter(|&&index| {
                let matched = initial_matches[index];
                focus_alignment_keypoint_is_foreground(source_image, keypoints1[matched.index1])
                    || focus_alignment_keypoint_is_foreground(
                        target_image,
                        keypoints2[matched.index2],
                    )
            })
            .count();
        if log_match {
            println!(
                "  - Mixed-focal foreground support: {foreground_support}/{} inliers",
                projected_inlier_indices.len()
            );
        }
        // A scale-correct homography supported only by a repeated wall weave
        // is still a false bridge between independent phone captures. When
        // foreground masks exist, require a few identity-bearing points too.
        if foreground_support < minimum_inliers.min(8) {
            let descriptor_pose_verified = focus_overlap_quality(
                source_image,
                target_image,
                &projected_homography,
            )
            .is_some_and(|(intensity_ncc, edge_ncc, edge_orientation, samples)| {
                if log_match {
                    println!(
                        "  - Rail-supported mixed-focal pose validation: intensity NCC {:.3}, edge NCC {:.3}, edge orientation {:.3} ({} samples)",
                        intensity_ncc, edge_ncc, edge_orientation, samples
                    );
                }
                intensity_ncc >= MIXED_FOCAL_COARSE_MIN_INTENSITY_NCC
                    && edge_ncc >= MIXED_FOCAL_COARSE_MIN_EDGE_NCC
                    && edge_orientation >= MIXED_FOCAL_COARSE_MIN_EDGE_ORIENTATION
            });
            if !descriptor_pose_verified {
                // The descriptor consensus may have landed entirely on the
                // display rail even when a genuine cross-lens artwork overlap
                // is present elsewhere. Give the denser image-level validator
                // one chance; it will reject a repeated rail-only pose by its
                // full-overlap structure score.
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
                    return Some(MatchGeometryResult::CoarseBridge {
                        homography: coarse_homography,
                        points: coarse_points,
                    });
                }
                return None;
            }
        }
    }
    let (dense_focus_points, foreground_feature_points) = if blend_mode == BlendMode::FocusStack {
        let dense_focus_points = collect_dense_focus_region_points(
            source_image,
            target_image,
            &projected_homography,
            projection,
        );
        let mut foreground_feature_points = collect_foreground_feature_points(
            source_image,
            target_image,
            &projected_homography,
            projection,
        );
        // Descriptor matches are sparse but carry a stronger identity signal
        // than a patch search on a long edge. Keep them alongside the strictly
        // corner-filtered matches; the regional residual/RANSAC checks below
        // decide whether a point is geometrically usable.
        foreground_feature_points.extend(top_candidate_points.iter().copied());
        (dense_focus_points, foreground_feature_points)
    } else {
        (Vec::new(), Vec::new())
    };
    let mut inlier_points = projected_inlier_indices
        .iter()
        .map(|&index| {
            let matched = initial_matches[index];
            if registration::native_patch_refinement_disabled_for_tests()
                || (mixed_focal_pair && projection == Projection::Planar)
            {
                // The test-only switch keeps the descriptor/RANSAC observation
                // untouched. Mixed-focal pairs do the same until the
                // scale-aware patch pass below replaces these points; that pass
                // is also covered by the switch.
                projected_match_points[index]
            } else {
                refine_match_point_from_homography(
                    source_image,
                    target_image,
                    keypoints1[matched.index1],
                    keypoints2[matched.index2],
                    projection,
                    &projected_homography,
                    false,
                    blend_mode == BlendMode::FocusStack,
                )
                .unwrap_or(projected_match_points[index])
            }
        })
        .collect::<Vec<_>>();
    if mixed_focal_pair
        && projection == Projection::Planar
        && !registration::native_patch_refinement_disabled_for_tests()
    {
        // The descriptors live in the reduced alignment images while the RANSAC
        // model above lives in source pixels. Convert the model back into the
        // descriptor coordinate system, warp every patch sample, then convert
        // the refined target observations to source pixels. This removes the
        // several-pixel quantisation error that otherwise becomes a visible
        // soft edge at 85mm/35mm overlaps.
        let source_scale = source_image.scale_factor.max(1e-9);
        let target_scale = target_image.scale_factor.max(1e-9);
        let to_target_alignment = Matrix3::new(
            target_scale.recip(),
            0.0,
            0.0,
            0.0,
            target_scale.recip(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let from_source_alignment = Matrix3::new(
            source_scale,
            0.0,
            0.0,
            0.0,
            source_scale,
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let alignment_homography =
            to_target_alignment * projected_homography * from_source_alignment;
        let mut refined_points = Vec::new();
        for &index in &projected_inlier_indices {
            let matched = initial_matches[index];
            let center = Point2::new(
                keypoints1[matched.index1].x as f64,
                keypoints1[matched.index1].y as f64,
            );
            if let Some(refined) = registration::refine_warped_patch(
                &source_image.alignment_image,
                &target_image.alignment_image,
                &alignment_homography,
                center,
                7,
                4,
            ) {
                let source = projected_match_points[index].0;
                let target = Point2::new(
                    refined.target.x * target_scale,
                    refined.target.y * target_scale,
                );
                if target.x.is_finite() && target.y.is_finite() {
                    refined_points.push((source, target));
                }
            }
        }
        if refined_points.len() >= minimum_inliers
            && match_has_spatial_support(
                &refined_points,
                source_image.dimensions(),
                target_image.dimensions(),
            )
        {
            println!(
                "  - Cross-scale warped-patch refinement retained {}/{} observations",
                refined_points.len(),
                projected_inlier_indices.len()
            );
            inlier_points = refined_points;
        }
    }
    let model_refinement_threshold = if blend_mode == BlendMode::FocusStack {
        FOCUS_MODEL_INLIER_THRESHOLD
    } else {
        FULL_RES_REFINEMENT_THRESHOLD
    };
    let alignment_quantization = source_image.scale_factor.max(target_image.scale_factor) * 1.5;
    let refinement_threshold = if source_image.full_image.is_some()
        && target_image.full_image.is_some()
        && !mixed_focal_pair
    {
        model_refinement_threshold
    } else {
        model_refinement_threshold
            .max(alignment_quantization)
            .max(if mixed_focal_pair {
                PANORAMA_MODEL_INLIER_THRESHOLD
            } else {
                0.0
            })
    };
    let refined_homography = if mixed_focal_fallback_used {
        // The two-point scale-constrained fallback has already performed its
        // own robust inlier/refit loop. Re-running the four-point homography
        // refit on only a dozen cross-scale observations can reject the valid
        // bridge because the sparse close-up has different lens distortion.
        projected_homography
    } else {
        refine_homography_inliers(&mut inlier_points, refinement_threshold, minimum_inliers)?
    };
    if stable_four_point_solver
        && !match_has_spatial_support(
            &inlier_points,
            source_image.dimensions(),
            target_image.dimensions(),
        )
    {
        // Descriptor consensus concentrated on one repeated face/stroke is not
        // enough to place a large tile. Reject it before it can become a strong
        // graph edge and pull an otherwise coherent mosaic into a false overlap.
        return None;
    }
    let mut homography = if alignment_mode == AlignmentMode::Position {
        estimate_translation(&inlier_points)
    } else if alignment_mode == AlignmentMode::FlatArtwork {
        // The projective RANSAC above is only a candidate generator.  A flat
        // artwork must be rendered with a similarity pose, so validate that
        // selected model again before it can become an MST edge.  Repeated
        // brush strokes often produce a large projective consensus while the
        // constrained similarity fit has a much larger residual; retaining
        // those points would let a false edge reorder the whole mosaic.
        let Some(similarity) = estimate_similarity(&inlier_points) else {
            return None;
        };
        let Some((similarity, model_inliers, _)) = panorama_model_fit(similarity, &inlier_points)
        else {
            return None;
        };
        if model_inliers.len() < minimum_inliers {
            return None;
        }
        inlier_points = model_inliers
            .into_iter()
            .filter_map(|index| inlier_points.get(index).copied())
            .collect();
        similarity
    } else if mixed_focal_fallback_used {
        // The fallback is already a scale-constrained similarity fit. Running
        // the generic focus model selector on a sparse lens-switch bridge can
        // turn its valid scale into an arbitrary affine shear, so preserve the
        // model that passed the focal-length plausibility gate.
        projected_homography
    } else if blend_mode == BlendMode::FocusStack {
        select_focus_stack_transform(
            &refined_homography,
            &inlier_points,
            source_image.dimensions(),
            alignment_mode,
        )
    } else if stable_four_point_solver && alignment_mode == AlignmentMode::Auto {
        select_large_panorama_transform(
            &refined_homography,
            &inlier_points,
            source_image.dimensions(),
            log_match,
        )
    } else {
        refined_homography
    };
    // A shifted artwork scan can require perspective even when every frame
    // uses the same lens. The low-DOF selector may favour a precise minority
    // while rejecting the correctly registered full overlap. Keep the original
    // feature/RANSAC fit as a candidate; only a distributed, independently
    // image-verified fit may supersede that minority.
    if blend_mode == BlendMode::FocusStack
        && stable_four_point_solver
        && !mixed_focal_pair
        && !mixed_focal_fallback_used
        && alignment_mode == AlignmentMode::Auto
    {
        let selected_indices =
            symmetric_inlier_indices(&homography, &inlier_points, FOCUS_MODEL_INLIER_THRESHOLD);
        let selected_points = selected_indices
            .iter()
            .map(|&index| inlier_points[index])
            .collect::<Vec<_>>();
        let selected_verified = focus_overlap_quality(source_image, target_image, &homography)
            .is_some_and(|quality| {
                focus_overlap_is_verified(
                    quality,
                    selected_points.len(),
                    median_symmetric_error(&homography, &selected_points),
                    allow_low_texture_boundary,
                )
            });
        if !selected_verified {
            if shifted_focus_structure_is_verified(
                source_image,
                target_image,
                &homography,
                &selected_points,
                log_match,
            ) {
                // Keep the simpler model when its structure is independently
                // supported; do not promote perspective solely for more points.
            } else if shifted_focus_structure_is_verified(
                source_image,
                target_image,
                &refined_homography,
                &inlier_points,
                log_match,
            ) {
                homography = refined_homography;
                if log_match {
                    println!(
                        "  - Retaining structure-verified shifted focus projective fit instead of a low-DOF minority"
                    );
                }
            }
        }
    }
    if blend_mode == BlendMode::FocusStack
        && !mixed_focal_fallback_used
        && !retain_model_inliers(
            &mut inlier_points,
            &homography,
            FOCUS_MODEL_INLIER_THRESHOLD,
            minimum_inliers,
        )
    {
        // The projective RANSAC fit is only an intermediate consensus. A
        // lower-DOF focus-stack model must get its own residual-consistent
        // observations; otherwise the global pose solve receives points that
        // support a different warp and can introduce a soft/double edge.
        if log_match {
            println!(
                "  - Rejecting focus match after selected-model validation: fewer than {} consistent observations",
                minimum_inliers
            );
        }
        return None;
    }
    Some(MatchGeometryResult::Verified(MatchGeometry {
        homography,
        inlier_points,
        projected_match_points,
        top_candidate_points,
        dense_focus_points,
        foreground_feature_points,
        mixed_focal_fallback_used,
    }))
}
