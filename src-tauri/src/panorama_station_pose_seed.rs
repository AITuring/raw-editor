//! Station pose seeds and the Local_Scale pose-candidate guard.
//!
//! Translation poses give the shifted-scan seed, long-edge roll correction keeps
//! station horizons consistent, and a solved projective candidate replaces the
//! spanning-tree seed only when its sampled Local_Scale stays bounded
//! (requirement 6.6).

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stack_pipeline::degradation;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::HashMap;
use std::path::Path;

use super::{
    Dsu, FOCUS_GROUP_ROLL_MAX_CORRECTION_RADIANS, FOCUS_GROUP_ROLL_MAX_SLOPE_MAD,
    FOCUS_GROUP_ROLL_MIN_LINES, FocusCaptureGroup, STATION_POSE_MAX_LOCAL_SCALE_RATIO,
    STATION_RELATION_MAX_MEDIAN_ERROR_PX, absolute_source_path, fit_focus_horizontal_edge_line,
    median_value, solve_robust_closure_translations, station_pose_local_scale_range,
    transform_is_stable_for_focus_stack, transformed_point,
};

pub(super) fn solve_focus_group_translation_poses(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    _locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_group: usize,
    coordinate_scale: f64,
) -> Option<Vec<Matrix3<f64>>> {
    if groups.len() < 2 || reference_group >= groups.len() {
        return None;
    }
    let Some(reference_focal) = images[groups[reference_group].anchor].focal_length_35mm else {
        println!(
            "  - Focus capture-group translation geometry unavailable: missing reference focal length"
        );
        return None;
    };
    if groups.iter().any(|group| {
        images[group.anchor].focal_length_35mm.is_some_and(|focal| {
            let ratio = (focal / reference_focal).max(reference_focal / focal);
            !ratio.is_finite() || ratio > 1.03
        })
    }) {
        println!(
            "  - Focus capture-group translation geometry unavailable: mixed focal-length groups"
        );
        return None;
    }

    let mut consensus_connectivity = Dsu::new(groups.len());
    // Only multi-layer Consensus_Feature evidence may connect stations.
    // A single Source_RAW observation is supplemental evidence, never a
    // station relation on its own (需求 6.2).
    for &(_, left, right, _, support, _) in group_edges {
        if support >= 2 {
            consensus_connectivity.union(left, right);
        }
    }
    let mut consensus_sizes = HashMap::<usize, usize>::new();
    for group in 0..groups.len() {
        *consensus_sizes
            .entry(consensus_connectivity.find(group))
            .or_default() += 1;
    }
    let consensus_covered = consensus_sizes.values().copied().max().unwrap_or(0);
    // The checks below still require every group to be connected and every
    // translation constraint to pass robust closure validation.  Requiring
    // an 85% multi-layer core here made the solver fall back to a projective
    // spanning tree precisely on long, low-texture scans where scale drift
    // is most damaging.
    if consensus_covered < groups.len() {
        println!(
            "  - Focus capture-group translation geometry unavailable: multi-layer graph covers only {consensus_covered}/{} groups",
            groups.len()
        );
        return None;
    }

    let constraints = group_edges
        .iter()
        .filter_map(
            |&(score, left, right, left_to_right, support, median_error)| {
                if support < 2 || left >= groups.len() || right >= groups.len() {
                    return None;
                }
                let left_image = &images[groups[left].anchor];
                let left_center = Point2::new(
                    left_image.width as f64 * 0.5,
                    left_image.height as f64 * 0.5,
                );
                let right_point = transformed_point(&left_to_right, left_center)?;
                let observed = left_center - right_point;
                let weight = support as f64 * score.max(1e-8).sqrt()
                    / (1.0 + median_error / STATION_RELATION_MAX_MEDIAN_ERROR_PX);
                (observed.x.is_finite()
                    && observed.y.is_finite()
                    && weight.is_finite()
                    && weight > 0.0)
                    .then_some((left, right, [observed.x, observed.y], weight))
            },
        )
        .collect::<Vec<_>>();
    if constraints.len() + 1 < groups.len() {
        println!(
            "  - Focus capture-group translation geometry unavailable: {} relations for {} groups",
            constraints.len(),
            groups.len()
        );
        return None;
    }
    let mut connectivity = Dsu::new(groups.len());
    for &(left, right, _, _) in &constraints {
        connectivity.union(left, right);
    }
    let root = connectivity.find(reference_group);
    let disconnected = (0..groups.len())
        .filter(|&group| connectivity.find(group) != root)
        .collect::<Vec<_>>();
    if !disconnected.is_empty() {
        let names = disconnected
            .iter()
            .map(|&group| {
                Path::new(&images[groups[group].anchor].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        println!(
            "  - Focus capture-group translation geometry unavailable: disconnected group anchors {:?}",
            names
        );
        return None;
    }

    let solve = solve_robust_closure_translations(groups.len(), reference_group, &constraints)?;
    let positions = solve.corrections;
    let iterations = solve.iterations;
    let mut residuals = constraints
        .iter()
        .map(|&(left, right, observed, _)| {
            (positions[right][0] - positions[left][0] - observed[0])
                .hypot(positions[right][1] - positions[left][1] - observed[1])
        })
        .collect::<Vec<_>>();
    let Some(median_residual) = median_value(&mut residuals) else {
        println!(
            "  - Focus capture-group translation geometry unavailable: no finite closure residuals"
        );
        return None;
    };
    let robust_limit = coordinate_scale * 0.04;
    if !median_residual.is_finite() || median_residual > robust_limit {
        println!(
            "  - Focus capture-group translation geometry rejected: median closure error {:.2}px exceeds {:.2}px",
            median_residual, robust_limit
        );
        return None;
    }
    println!(
        "  - Focus capture-group translation geometry seeded from {} relation(s), multi-layer core {consensus_covered}/{}, {iterations} iteration(s), median closure error {:.2}px",
        constraints.len(),
        groups.len(),
        median_residual
    );
    Some(
        positions
            .into_iter()
            .map(|position| {
                Matrix3::new(1.0, 0.0, position[0], 0.0, 1.0, position[1], 0.0, 0.0, 1.0)
            })
            .collect(),
    )
}

pub(super) fn correct_focus_group_roll_from_long_edges(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_poses: &mut [Matrix3<f64>],
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_group: usize,
) -> usize {
    if groups.len() != group_poses.len() || reference_group >= groups.len() {
        return 0;
    }
    let mut group_angles = vec![None; groups.len()];
    for (group_index, group) in groups.iter().enumerate() {
        let image = &images[group.anchor];
        let pose = &group_poses[group_index];
        let mut slopes = image
            .horizontal_edge_rows
            .iter()
            .filter_map(|&row| fit_focus_horizontal_edge_line(image, pose, row))
            .map(|line| line.slope)
            .filter(|slope| slope.is_finite())
            .collect::<Vec<_>>();
        if slopes.len() < FOCUS_GROUP_ROLL_MIN_LINES {
            continue;
        }
        let Some(slope) = median_value(&mut slopes) else {
            continue;
        };
        let mut deviations = slopes
            .iter()
            .map(|value| (value - slope).abs())
            .collect::<Vec<_>>();
        let Some(mad) = median_value(&mut deviations) else {
            continue;
        };
        if mad > FOCUS_GROUP_ROLL_MAX_SLOPE_MAD {
            continue;
        }
        group_angles[group_index] = Some(slope.atan());
    }
    let mut reliable_angles = group_angles.iter().flatten().copied().collect::<Vec<_>>();
    let target_angle = median_value(&mut reliable_angles);
    let reference_pose_angle = locked_homographies
        .get(&images[groups[reference_group].anchor].id)
        .map(|pose| pose[(1, 0)].atan2(pose[(0, 0)]));

    let mut corrected_count = 0usize;
    let mut edge_corrected_count = 0usize;
    for (group_index, angle) in group_angles.into_iter().enumerate() {
        let edge_correction = angle
            .zip(target_angle)
            .map(|(angle, target)| target - angle);
        let pose_correction = reference_pose_angle.and_then(|reference| {
            locked_homographies
                .get(&images[groups[group_index].anchor].id)
                .map(|pose| pose[(1, 0)].atan2(pose[(0, 0)]) - reference)
        });
        let Some(raw_correction) = edge_correction.or(pose_correction) else {
            continue;
        };
        let correction_angle = raw_correction.clamp(
            -FOCUS_GROUP_ROLL_MAX_CORRECTION_RADIANS,
            FOCUS_GROUP_ROLL_MAX_CORRECTION_RADIANS,
        );
        if correction_angle.abs() < 1e-5 {
            continue;
        }
        let image = &images[groups[group_index].anchor];
        let source_center = Point2::new(image.width as f64 * 0.5, image.height as f64 * 0.5);
        let Some(world_center) = transformed_point(&group_poses[group_index], source_center) else {
            continue;
        };
        let (sin, cos) = correction_angle.sin_cos();
        let rotate_about_center = Matrix3::new(
            cos,
            -sin,
            world_center.x - cos * world_center.x + sin * world_center.y,
            sin,
            cos,
            world_center.y - sin * world_center.x - cos * world_center.y,
            0.0,
            0.0,
            1.0,
        );
        let corrected = rotate_about_center * group_poses[group_index];
        if transform_is_stable_for_focus_stack(&corrected, image.dimensions()) {
            group_poses[group_index] = corrected;
            corrected_count += 1;
            edge_corrected_count += usize::from(edge_correction.is_some());
        }
    }
    if corrected_count > 0 {
        println!(
            "  - Focus capture-group roll correction: {corrected_count}/{} groups ({edge_corrected_count} from long edges, {} pose-only), long-edge target {}, maximum correction {:.3} degrees",
            groups.len(),
            corrected_count.saturating_sub(edge_corrected_count),
            target_angle
                .map(|angle| format!("{:.3} degrees", angle.to_degrees()))
                .unwrap_or_else(|| "unavailable".to_string()),
            FOCUS_GROUP_ROLL_MAX_CORRECTION_RADIANS.to_degrees(),
        );
    }
    corrected_count
}

pub(super) fn station_pose_candidate_has_bounded_local_scale(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    poses: &[Matrix3<f64>],
    stage: &str,
) -> bool {
    if poses.len() != groups.len() {
        return false;
    }
    let mut accepted = true;
    for (station_index, (group, pose)) in groups.iter().zip(poses).enumerate() {
        let anchor = &images[group.anchor];
        let range = station_pose_local_scale_range(pose, anchor.dimensions());
        let ratio = range.map(|(minimum, maximum)| maximum / minimum);
        if ratio.is_some_and(|ratio| ratio <= STATION_POSE_MAX_LOCAL_SCALE_RATIO) {
            continue;
        }
        accepted = false;
        degradation::record_run_degradation(
            degradation::STATION_POSE_LOCAL_SCALE_EXCEEDED,
            serde_json::json!({
                "station_index": station_index,
                "anchor_path": absolute_source_path(&anchor.filename),
                "stage": stage,
                "minimum_local_scale": range.map(|range| range.0),
                "maximum_local_scale": range.map(|range| range.1),
                "local_scale_ratio": ratio,
                "maximum_ratio": STATION_POSE_MAX_LOCAL_SCALE_RATIO,
            }),
        );
        println!(
            "  - Station pose candidate rejected at station {station_index}: Local_Scale ratio {} exceeds {:.2}; retaining spanning-tree seed",
            ratio
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "unmeasurable".to_string()),
            STATION_POSE_MAX_LOCAL_SCALE_RATIO,
        );
    }
    accepted
}

/// Keep a general projective candidate exactly as solved when its sampled
/// Local_Scale is bounded; otherwise preserve every bit of the spanning-tree
/// seed (requirement 6.6).
pub(super) fn select_station_pose_candidate(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    candidate: &[Matrix3<f64>],
    spanning_tree_seed: &[Matrix3<f64>],
    stage: &str,
) -> Vec<Matrix3<f64>> {
    if station_pose_candidate_has_bounded_local_scale(images, groups, candidate, stage) {
        candidate.to_vec()
    } else {
        spanning_tree_seed.to_vec()
    }
}
