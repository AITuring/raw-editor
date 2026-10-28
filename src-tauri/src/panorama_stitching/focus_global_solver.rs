//! Global focus-stack homography optimisation.
//!
//! Robust cost, normal equations and the damped iterations that refine every
//! focus-layer homography against a reference, globally or per region.

use crate::panorama_utils::processing;
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::MatchInfo;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::HashMap;

use super::{
    FOCUS_GLOBAL_DAMPING, FOCUS_GLOBAL_MAX_LINEAR_ADJUSTMENT,
    FOCUS_GLOBAL_MAX_PROJECTIVE_ADJUSTMENT, FOCUS_GLOBAL_MAX_TRANSLATION_ADJUSTMENT,
    FOCUS_GLOBAL_PRIOR_WEIGHT, FOCUS_GLOBAL_PROJECTIVE_PRIOR_WEIGHT, FOCUS_LOCAL_MODEL_MIN_INLIERS,
    FOCUS_MODEL_MIN_INLIERS, FocusGlobalObservation, FocusPose, focus_global_observations,
    focus_global_observations_for_generic_region, focus_homography_to_pose, focus_normalized_point,
    focus_pose_projection, focus_pose_to_full_homography, homography_preserves_focus_orientation,
    images_have_mixed_focal_lengths,
};

pub(super) const FOCUS_GLOBAL_MAX_ITERATIONS: usize = 12;

// The global pose solve operates in coordinates normalised by the largest
// source dimension. A 0.0012 Huber transition is roughly 19px on the phone
// frames used by the focus-stack path; that is wide enough for a repeated
// canvas stroke to pull a whole layer before the local mosaic refinement gets
// a chance to correct it. Keep the robust solve focused on the identity
// correspondences and let the regional/native passes handle the remaining
// sub-pixel deformation.
pub(super) const FOCUS_GLOBAL_HUBER_THRESHOLD: f64 = 0.00065;

pub(super) fn focus_global_poses_are_valid(
    poses: &[FocusPose],
    images: &[ImageInfo],
    reference: &ImageInfo,
    coordinate_scale: f64,
) -> bool {
    if poses.len() != images.len()
        || poses.iter().any(|pose| {
            pose.iter()
                .any(|value| !value.is_finite() || value.abs() > 100.0)
        })
    {
        return false;
    }
    poses.iter().zip(images).all(|(pose, image)| {
        let corners = [
            Point2::new(0.0, 0.0),
            Point2::new(image.width as f64, 0.0),
            Point2::new(image.width as f64, image.height as f64),
            Point2::new(0.0, image.height as f64),
        ];
        let normalized_corners = corners
            .into_iter()
            .map(|corner| focus_normalized_point(image, corner, coordinate_scale));
        if normalized_corners
            .filter_map(|corner| focus_pose_projection(pose, corner))
            .count()
            != 4
        {
            return false;
        }
        let Some(full) = focus_pose_to_full_homography(pose, image, reference, coordinate_scale)
        else {
            return false;
        };
        full.try_inverse().is_some()
            && homography_preserves_focus_orientation(&full, (image.width, image.height))
    })
}

pub(super) fn focus_global_robust_cost(
    poses: &[FocusPose],
    initial_poses: &[FocusPose],
    observations: &[FocusGlobalObservation],
    reference_index: usize,
) -> f64 {
    let mut cost = 0.0;
    for observation in observations {
        let Some((source, _, _)) =
            focus_pose_projection(&poses[observation.source_index], observation.source)
        else {
            return f64::INFINITY;
        };
        let Some((target, _, _)) =
            focus_pose_projection(&poses[observation.target_index], observation.target)
        else {
            return f64::INFINITY;
        };
        let residual = source - target;
        let magnitude = residual.norm();
        if !magnitude.is_finite() {
            return f64::INFINITY;
        }
        let robust_cost = if magnitude <= FOCUS_GLOBAL_HUBER_THRESHOLD {
            0.5 * magnitude * magnitude
        } else {
            FOCUS_GLOBAL_HUBER_THRESHOLD * (magnitude - 0.5 * FOCUS_GLOBAL_HUBER_THRESHOLD)
        };
        cost += observation.weight * robust_cost;
    }
    for (image_index, pose) in poses.iter().enumerate() {
        if image_index == reference_index {
            continue;
        }
        for parameter in 0..8 {
            let prior_weight = if parameter >= 6 {
                FOCUS_GLOBAL_PROJECTIVE_PRIOR_WEIGHT
            } else {
                FOCUS_GLOBAL_PRIOR_WEIGHT
            };
            let difference = pose[parameter] - initial_poses[image_index][parameter];
            cost += 0.5 * prior_weight * difference * difference;
        }
    }
    cost
}

pub(super) fn focus_global_normal_equations(
    poses: &[FocusPose],
    initial_poses: &[FocusPose],
    observations: &[FocusGlobalObservation],
    reference_index: usize,
) -> (nalgebra::DMatrix<f64>, nalgebra::DVector<f64>) {
    let variable_count = poses.len().saturating_sub(1) * 8;
    let mut normal = nalgebra::DMatrix::zeros(variable_count, variable_count);
    let mut gradient = nalgebra::DVector::zeros(variable_count);

    for observation in observations {
        let Some((source, source_x_jacobian, source_y_jacobian)) =
            focus_pose_projection(&poses[observation.source_index], observation.source)
        else {
            continue;
        };
        let Some((target, target_x_jacobian, target_y_jacobian)) =
            focus_pose_projection(&poses[observation.target_index], observation.target)
        else {
            continue;
        };
        let residual = source - target;
        let magnitude = residual.norm();
        if !magnitude.is_finite() {
            continue;
        }
        let robust_weight = observation.weight
            * if magnitude <= FOCUS_GLOBAL_HUBER_THRESHOLD {
                1.0
            } else {
                FOCUS_GLOBAL_HUBER_THRESHOLD / magnitude
            };

        for (residual_component, source_jacobian, target_jacobian) in [
            (residual.x, source_x_jacobian, target_x_jacobian),
            (residual.y, source_y_jacobian, target_y_jacobian),
        ] {
            let mut jacobian_entries = Vec::with_capacity(16);
            if observation.source_index != reference_index {
                let base = if observation.source_index < reference_index {
                    observation.source_index * 8
                } else {
                    (observation.source_index - 1) * 8
                };
                for (parameter, &value) in source_jacobian.iter().enumerate() {
                    jacobian_entries.push((base + parameter, value));
                }
            }
            if observation.target_index != reference_index {
                let base = if observation.target_index < reference_index {
                    observation.target_index * 8
                } else {
                    (observation.target_index - 1) * 8
                };
                for (parameter, &value) in target_jacobian.iter().enumerate() {
                    jacobian_entries.push((base + parameter, -value));
                }
            }
            for &(column, jacobian) in &jacobian_entries {
                gradient[column] += robust_weight * jacobian * residual_component;
                for &(row, other_jacobian) in &jacobian_entries {
                    normal[(column, row)] += robust_weight * jacobian * other_jacobian;
                }
            }
        }
    }

    for (image_index, pose) in poses.iter().enumerate() {
        if image_index == reference_index {
            continue;
        }
        let base = if image_index < reference_index {
            image_index * 8
        } else {
            (image_index - 1) * 8
        };
        for parameter in 0..8 {
            let prior_weight = if parameter >= 6 {
                FOCUS_GLOBAL_PROJECTIVE_PRIOR_WEIGHT
            } else {
                FOCUS_GLOBAL_PRIOR_WEIGHT
            };
            normal[(base + parameter, base + parameter)] += prior_weight + FOCUS_GLOBAL_DAMPING;
            gradient[base + parameter] +=
                prior_weight * (pose[parameter] - initial_poses[image_index][parameter]);
        }
    }
    (normal, gradient)
}

pub(super) fn optimize_focus_stack_global_homographies_with_reference(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
) -> HashMap<usize, Matrix3<f64>> {
    if images.iter().any(|image| image.overview_reference) {
        // The wide frame is a bounded coverage source, not a registration
        // anchor. It has no verified identity correspondences with the
        // close-up sequence, so letting bundle adjustment see the disconnected
        // components would invent a scale for the whole stack.
        println!(
            "  - Skipping global pose refinement: overview reference has no verified identity overlap"
        );
        return initial_homographies.clone();
    }
    optimize_focus_stack_global_homographies_in_region_mode_with_reference(
        images,
        matches,
        initial_homographies,
        None,
        false,
        reference_index,
    )
}

pub(super) fn optimize_focus_stack_global_homographies_in_region(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    normalized_y_range: Option<(f64, f64)>,
) -> HashMap<usize, Matrix3<f64>> {
    optimize_focus_stack_global_homographies_in_region_mode_with_reference(
        images,
        matches,
        initial_homographies,
        normalized_y_range,
        true,
        0,
    )
}

pub(super) fn optimize_focus_stack_global_homographies_in_generic_region(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    normalized_y_range: (f64, f64),
) -> HashMap<usize, Matrix3<f64>> {
    optimize_focus_stack_global_homographies_in_region_mode_with_reference(
        images,
        matches,
        initial_homographies,
        Some(normalized_y_range),
        false,
        0,
    )
}

pub(super) fn optimize_focus_stack_global_homographies_in_region_mode_with_reference(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    normalized_y_range: Option<(f64, f64)>,
    use_foreground_region_matches: bool,
    reference_index: usize,
) -> HashMap<usize, Matrix3<f64>> {
    if images.len() < 2 || reference_index >= images.len() {
        return initial_homographies.clone();
    }
    let mixed_focal_stack = images_have_mixed_focal_lengths(images);
    let reference = &images[reference_index];
    let coordinate_scale = images
        .iter()
        .map(|image| image.width.max(image.height) as f64)
        .fold(1.0, f64::max);
    let mut initial_poses = Vec::with_capacity(images.len());
    for image in images {
        let Some(homography) = initial_homographies.get(&image.id) else {
            return initial_homographies.clone();
        };
        let Some(pose) = focus_homography_to_pose(homography, image, reference, coordinate_scale)
        else {
            return initial_homographies.clone();
        };
        initial_poses.push(pose);
    }
    initial_poses[reference_index] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let initial_pose_maxima = (0..8)
        .map(|parameter| {
            initial_poses
                .iter()
                .map(|pose| pose[parameter].abs())
                .fold(0.0, f64::max)
        })
        .collect::<Vec<_>>();
    println!(
        "  - Focus global initial pose maxima: {:?}",
        initial_pose_maxima
    );

    let observations = if use_foreground_region_matches {
        focus_global_observations(images, matches, coordinate_scale, normalized_y_range)
    } else if let Some(range) = normalized_y_range {
        focus_global_observations_for_generic_region(images, matches, coordinate_scale, range)
    } else {
        focus_global_observations(images, matches, coordinate_scale, None)
    };
    if mixed_focal_stack {
        println!(
            "  - Focus global registration for mixed-focal stack: solving validated same-/cross-lens closures"
        );
    }
    let minimum_observations = if normalized_y_range.is_some() {
        if use_foreground_region_matches {
            FOCUS_LOCAL_MODEL_MIN_INLIERS
        } else {
            FOCUS_MODEL_MIN_INLIERS
        }
    } else {
        processing::MIN_INLIERS_FOR_CONNECTION
    };
    if observations.len() < minimum_observations {
        return initial_homographies.clone();
    }
    let mut poses = initial_poses.clone();
    let mut current_cost =
        focus_global_robust_cost(&poses, &initial_poses, &observations, reference_index);
    if !current_cost.is_finite() {
        return initial_homographies.clone();
    }
    let initial_poses_valid =
        focus_global_poses_are_valid(&poses, images, reference, coordinate_scale);
    println!("  - Focus global registration initial geometry valid: {initial_poses_valid}");
    if !initial_poses_valid {
        // Do not let the optimizer turn a bad pairwise registration into a
        // plausible-looking mirrored canvas. The original homographies are the
        // safer fallback; pairwise model validation remains responsible for
        // finding a usable non-mirrored path on the next stage.
        return initial_homographies.clone();
    }
    let initial_cost = current_cost;
    let mut accepted_iterations = 0;

    for _ in 0..FOCUS_GLOBAL_MAX_ITERATIONS {
        let (normal, gradient) =
            focus_global_normal_equations(&poses, &initial_poses, &observations, reference_index);
        let right_hand_side = -gradient;
        let Some(delta) = normal.lu().solve(&right_hand_side) else {
            break;
        };
        let delta_norm = delta.norm();
        if !delta_norm.is_finite() || delta_norm < 1e-9 {
            break;
        }
        let max_parameter_delta = delta
            .as_slice()
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        println!(
            "  - Focus global pose step: norm {:.6}, max {:.6}",
            delta_norm, max_parameter_delta
        );

        let mut accepted = None;
        for step in [1.0, 0.5, 0.25, 0.125, 0.0625] {
            let mut candidate = poses.clone();
            for (image_index, pose) in candidate.iter_mut().enumerate() {
                if image_index == reference_index {
                    continue;
                }
                let base = if image_index < reference_index {
                    image_index * 8
                } else {
                    (image_index - 1) * 8
                };
                for parameter in 0..8 {
                    pose[parameter] += step * delta[base + parameter];
                }
            }
            if candidate
                .iter()
                .enumerate()
                .filter(|(image_index, _)| *image_index != reference_index)
                .any(|(image_index, pose)| {
                    pose.iter().enumerate().any(|(parameter, value)| {
                        let limit = match parameter {
                            2 | 5 => FOCUS_GLOBAL_MAX_TRANSLATION_ADJUSTMENT,
                            6 | 7 => FOCUS_GLOBAL_MAX_PROJECTIVE_ADJUSTMENT,
                            _ => FOCUS_GLOBAL_MAX_LINEAR_ADJUSTMENT,
                        };
                        (value - initial_poses[image_index][parameter]).abs() > limit
                    })
                })
            {
                continue;
            }
            if !focus_global_poses_are_valid(&candidate, images, reference, coordinate_scale) {
                continue;
            }
            let candidate_cost = focus_global_robust_cost(
                &candidate,
                &initial_poses,
                &observations,
                reference_index,
            );
            if candidate_cost.is_finite() && candidate_cost + 1e-12 < current_cost {
                accepted = Some((candidate, candidate_cost));
                break;
            }
        }
        let Some((candidate, candidate_cost)) = accepted else {
            break;
        };
        poses = candidate;
        current_cost = candidate_cost;
        accepted_iterations += 1;
    }

    if accepted_iterations == 0 || current_cost >= initial_cost {
        return initial_homographies.clone();
    }

    let mut optimized = HashMap::new();
    for (image_index, image) in images.iter().enumerate() {
        let Some(homography) =
            focus_pose_to_full_homography(&poses[image_index], image, reference, coordinate_scale)
        else {
            return initial_homographies.clone();
        };
        optimized.insert(image.id, homography);
    }
    println!(
        "  - Focus global registration used {} observations over {} edges: robust cost {:.6} -> {:.6} in {} iteration(s)",
        observations.len(),
        matches
            .values()
            .filter(|match_info| !match_info.points.is_empty())
            .count(),
        initial_cost,
        current_cost,
        accepted_iterations,
    );
    optimized
}
