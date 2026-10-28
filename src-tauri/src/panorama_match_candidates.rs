//! Candidate generation for one image pair.
//!
//! This module owns descriptor matching and the bounded mixed-focal fallback.
//! Geometry verification remains in the top-level match orchestrator.

use crate::panorama_utils::processing;

use super::{
    BlendMode, ImageInfo, MIXED_FOCAL_EMERGENCY_MATCH_RATIO_THRESHOLD,
    MIXED_FOCAL_MATCH_RATIO_THRESHOLD, Match, SCALABLE_MATCH_RATIO_THRESHOLD,
    SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION, image_pair_has_mixed_focal_lengths,
};

#[derive(Debug)]
pub(super) struct MatchCandidates {
    pub(super) initial_matches: Vec<Match>,
    pub(super) mixed_focal_pair: bool,
    pub(super) minimum_inliers: usize,
    pub(super) adjacent_same_focal_focus_pair: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_match_candidates(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    blend_mode: BlendMode,
    stable_four_point_solver: bool,
    log_match: bool,
    seeded_initial_matches: Option<Vec<Match>>,
) -> MatchCandidates {
    let mixed_focal_pair = image_pair_has_mixed_focal_lengths(source_image, target_image);
    let features1 = &source_image.features;
    let features2 = &target_image.features;
    let minimum_inliers = if stable_four_point_solver {
        SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION
    } else {
        processing::MIN_INLIERS_FOR_CONNECTION
    };
    let initial_matches = if let Some(seeded) = seeded_initial_matches {
        seeded
    } else if stable_four_point_solver {
        let ratio_threshold = if mixed_focal_pair {
            MIXED_FOCAL_MATCH_RATIO_THRESHOLD
        } else {
            SCALABLE_MATCH_RATIO_THRESHOLD
        };
        let mut matches = if mixed_focal_pair {
            match (
                source_image.focal_length_35mm,
                target_image.focal_length_35mm,
            ) {
                (Some(source_focal), Some(target_focal)) if source_focal > 0.0 => {
                    processing::match_features_with_ratio_and_scale(
                        features1,
                        features2,
                        MIXED_FOCAL_EMERGENCY_MATCH_RATIO_THRESHOLD,
                        (source_focal / target_focal) as f32,
                        1.65,
                    )
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        if matches.len() < minimum_inliers {
            matches = processing::match_features_with_ratio(features1, features2, ratio_threshold);
        }
        if mixed_focal_pair && matches.len() < minimum_inliers {
            matches = processing::match_features_with_ratio(
                features1,
                features2,
                MIXED_FOCAL_EMERGENCY_MATCH_RATIO_THRESHOLD,
            );
            if log_match {
                println!(
                    "  - Mixed-focal descriptor fallback produced {} candidates",
                    matches.len()
                );
            }
        }
        matches
    } else {
        processing::match_features(features1, features2)
    };

    // Filename adjacency is only a candidate-order hint. Keep this gate
    // explicit so geometry cannot be enabled by import order alone.
    let adjacent_same_focal_focus_pair = false;
    let _ = blend_mode;
    MatchCandidates {
        initial_matches,
        mixed_focal_pair,
        minimum_inliers,
        adjacent_same_focal_focus_pair,
    }
}
