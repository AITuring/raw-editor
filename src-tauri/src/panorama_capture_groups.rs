//! Capture groups: station membership, bracket pose locking, anchors and render order.
//!
//! The grouping outcome becomes FocusCaptureGroup records with one anchor and
//! member-to-anchor transforms; locked local bracket poses keep focus layers of a
//! station registered to each other, and render order follows the capture groups.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::MatchInfo;
use crate::panorama_utils::stack_pipeline::intra_station;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use super::{
    Dsu, FOCUS_BRACKET_INTERNAL_MOTION_RATIO, FOCUS_BRACKET_MAX_COMPONENT_SOURCES,
    FOCUS_MODEL_INLIER_THRESHOLD, StationEvidence, StationGroupingOutcome,
    focus_compatible_focal_length, focus_frame_median_sharpness, focus_image_direction_key,
    focus_local_bracket_components, focus_match_is_capture_station_link,
    focus_match_is_local_bracket, focus_station_grouping, focus_station_link_evidence,
    median_symmetric_error, natural_path_cmp, transform_is_stable_for_focus_stack,
    transformed_point,
};

/// Binding 需求 1.1 evidence of one Capture_Station: the weakest accepted
/// relation inside it.
///
/// Every member joined the station through at least this much evidence, so it is
/// the number 需求 1.7 should carry. Returns `None` for a single-member station,
/// which has no intra-station relation to report.
pub(super) fn focus_station_binding_evidence(
    members: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Option<StationEvidence> {
    let mut weakest: Option<StationEvidence> = None;
    for (offset, &left) in members.iter().enumerate() {
        for &right in &members[offset + 1..] {
            let key = (left.min(right), left.max(right));
            let Some(match_info) = matches.get(&key) else {
                continue;
            };
            let Some(evidence) = focus_station_link_evidence(images, key.0, key.1, match_info)
                .filter(StationEvidence::accepts)
            else {
                continue;
            };
            if weakest.is_none_or(|current| evidence.score < current.score) {
                weakest = Some(evidence);
            }
        }
    }
    weakest
}

pub(super) fn lock_focus_local_bracket_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    global_homographies: &HashMap<usize, Matrix3<f64>>,
) -> HashMap<usize, Matrix3<f64>> {
    let components = focus_local_bracket_components(images, matches);
    if components.is_empty() {
        return global_homographies.clone();
    }
    let mut locked = global_homographies.clone();
    let mut locked_sources = 0usize;
    let mut maximum_center_shift = 0.0f64;

    for component in &components {
        let members = component.iter().copied().collect::<HashSet<_>>();
        let mut edges = matches
            .iter()
            .filter_map(|(&(source_index, target_index), match_info)| {
                if !members.contains(&source_index)
                    || !members.contains(&target_index)
                    || !focus_match_is_local_bracket(
                        &images[source_index],
                        &images[target_index],
                        match_info,
                    )
                {
                    return None;
                }
                let error = median_symmetric_error(&match_info.homography, &match_info.points);
                let precision = if error.is_finite() {
                    1.0 / (1.0 + error / FOCUS_MODEL_INLIER_THRESHOLD)
                } else {
                    0.0
                };
                Some((
                    match_info.inliers.max(match_info.points.len()) as f64 * precision,
                    source_index,
                    target_index,
                    match_info.homography,
                ))
            })
            .collect::<Vec<_>>();
        edges.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| natural_path_cmp(&images[left.1].filename, &images[right.1].filename))
                .then_with(|| natural_path_cmp(&images[left.2].filename, &images[right.2].filename))
        });

        let mut support = HashMap::<usize, f64>::new();
        for &(weight, source_index, target_index, _) in &edges {
            *support.entry(source_index).or_default() += weight;
            *support.entry(target_index).or_default() += weight;
        }
        let Some(anchor) = component.iter().copied().min_by(|left, right| {
            support
                .get(right)
                .copied()
                .unwrap_or(0.0)
                .total_cmp(&support.get(left).copied().unwrap_or(0.0))
                .then_with(|| natural_path_cmp(&images[*left].filename, &images[*right].filename))
        }) else {
            continue;
        };
        let Some(anchor_global) = global_homographies.get(&images[anchor].id).copied() else {
            continue;
        };

        let mut local_dsu = Dsu::new(images.len());
        let mut adjacency = HashMap::<usize, Vec<(usize, Matrix3<f64>)>>::new();
        for &(_, source_index, target_index, source_to_target) in &edges {
            if local_dsu.find(source_index) == local_dsu.find(target_index) {
                continue;
            }
            let Some(target_to_source) = source_to_target.try_inverse() else {
                continue;
            };
            local_dsu.union(source_index, target_index);
            adjacency
                .entry(source_index)
                .or_default()
                .push((target_index, target_to_source));
            adjacency
                .entry(target_index)
                .or_default()
                .push((source_index, source_to_target));
        }

        let mut relative_to_anchor = HashMap::new();
        relative_to_anchor.insert(anchor, Matrix3::identity());
        let mut pending = VecDeque::from([anchor]);
        while let Some(current) = pending.pop_front() {
            let current_to_anchor = relative_to_anchor[&current];
            for &(neighbor, neighbor_to_current) in adjacency.get(&current).into_iter().flatten() {
                if relative_to_anchor.contains_key(&neighbor) {
                    continue;
                }
                relative_to_anchor.insert(neighbor, current_to_anchor * neighbor_to_current);
                pending.push_back(neighbor);
            }
        }
        if relative_to_anchor.len() != component.len() {
            continue;
        }

        for &image_index in component {
            let replacement = anchor_global * relative_to_anchor[&image_index];
            if !transform_is_stable_for_focus_stack(&replacement, images[image_index].dimensions())
            {
                continue;
            }
            if let Some(previous) = global_homographies.get(&images[image_index].id) {
                let center = Point2::new(
                    images[image_index].width as f64 * 0.5,
                    images[image_index].height as f64 * 0.5,
                );
                if let (Some(before), Some(after)) = (
                    transformed_point(previous, center),
                    transformed_point(&replacement, center),
                ) {
                    maximum_center_shift = maximum_center_shift.max((after - before).norm());
                }
            }
            locked.insert(images[image_index].id, replacement);
            locked_sources += 1;
        }
    }
    println!(
        "  - Locked {} local focus-bracket component(s), {} source poses; maximum center correction {:.2}px",
        components.len(),
        locked_sources,
        maximum_center_shift
    );
    locked
}

pub(super) struct FocusCaptureGroup {
    pub(super) members: Vec<usize>,
    pub(super) anchor: usize,
    pub(super) local_to_anchor: HashMap<usize, Matrix3<f64>>,
}

pub(super) fn focus_capture_groups(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<(Vec<FocusCaptureGroup>, Vec<usize>)> {
    focus_capture_groups_with_grouping(images, matches, locked_homographies)
        .map(|(groups, group_for_image, _)| (groups, group_for_image))
}

/// [`focus_capture_groups`] together with everything the Stack_Report needs to
/// carry about the grouping decision (需求 1.4 / 1.6 / 1.7 / 1.10).
pub(super) fn focus_capture_groups_with_grouping(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<(Vec<FocusCaptureGroup>, Vec<usize>, StationGroupingOutcome)> {
    // Direct local links already provide the strongest station membership
    // evidence. Keep those components intact; solved world centres may only
    // select among attachments backed by same-pose evidence. The previous implementation
    // sorted each connected component by filename before splitting it; a
    // renamed or shuffled import therefore changed the station count even
    // though every geometric match was identical.
    let grouping = focus_station_grouping(images, matches);
    let mut components = grouping.components.clone();
    let mut assigned = components
        .iter()
        .flat_map(|component| component.iter().copied())
        .collect::<HashSet<_>>();
    // 需求 1.6: a source with neither same-pose nor verified scan overlap is
    // isolated. It joins no Capture_Station at all — not even
    // by its solved world centre, which is only as trustworthy as the pose the
    // missing evidence could not confirm — while every other source keeps the
    // grouping it would have had anyway.  A single member left over by the
    // accumulated-motion split is *not* isolated: it had verified evidence, so
    // it still becomes a station of its own (需求 1.5).
    let isolated = grouping.isolated.iter().copied().collect::<HashSet<_>>();
    for image_index in 0..images.len() {
        if assigned.contains(&image_index) || isolated.contains(&image_index) {
            continue;
        }
        components.push(vec![image_index]);
        assigned.insert(image_index);
    }

    let world_center = |image_index: usize| {
        locked_homographies
            .get(&images[image_index].id)
            .and_then(|pose| {
                transformed_point(
                    pose,
                    Point2::new(
                        images[image_index].width as f64 * 0.5,
                        images[image_index].height as f64 * 0.5,
                    ),
                )
            })
    };
    let component_center = |component: &[usize]| {
        let centers = component
            .iter()
            .filter_map(|&index| world_center(index))
            .collect::<Vec<_>>();
        if centers.is_empty() {
            return None;
        }
        let x = centers.iter().map(|point| point.x).sum::<f64>() / centers.len() as f64;
        let y = centers.iter().map(|point| point.y).sum::<f64>() / centers.len() as f64;
        Some(Point2::new(x, y))
    };

    // Absorb a singleton into the nearest established station only with
    // same-pose evidence and a centre within the bracket-sized radius. A coarse-only
    // edge is an explicit boundary: it can suggest scan order, but cannot
    // make two focus layers compete for ownership.
    for singleton_index in (0..components.len())
        .filter(|&index| components[index].len() == 1)
        .collect::<Vec<_>>()
    {
        let image_index = components[singleton_index][0];
        let Some(center) = world_center(image_index) else {
            continue;
        };
        let mut best = None::<(f64, usize)>;
        for (candidate_index, candidate) in components.iter().enumerate() {
            if candidate_index == singleton_index
                || candidate.is_empty()
                || candidate.len() >= FOCUS_BRACKET_MAX_COMPONENT_SOURCES
                || !focus_compatible_focal_length(&images[image_index], &images[candidate[0]])
            {
                continue;
            }
            // A world-centre prior is not camera-pose evidence. In particular,
            // a connected single-frame scan station must not be folded back
            // into a neighbouring focus bracket just because their solved
            // centres drift close together. Only a verified same-pose link
            // may absorb the singleton into a multi-frame station.
            let same_pose_evidence = candidate.iter().any(|&member| {
                let key = (image_index.min(member), image_index.max(member));
                matches.get(&key).is_some_and(|edge| {
                    focus_match_is_capture_station_link(images, key.0, key.1, edge)
                })
            });
            if !same_pose_evidence {
                continue;
            }
            let blocked = candidate.iter().any(|&member| {
                matches
                    .get(&(image_index.min(member), image_index.max(member)))
                    .is_some_and(|edge| edge.coarse_bridge)
            });
            if blocked {
                continue;
            }
            let Some(candidate_center) = component_center(candidate) else {
                continue;
            };
            let scale = images[image_index]
                .width
                .max(images[image_index].height)
                .max(images[candidate[0]].width.max(images[candidate[0]].height))
                .max(1) as f64;
            let distance = (center - candidate_center).norm() / scale;
            if distance <= FOCUS_BRACKET_INTERNAL_MOTION_RATIO
                && best.is_none_or(|(best_distance, _)| distance < best_distance)
            {
                best = Some((distance, candidate_index));
            }
        }
        if let Some((_, candidate_index)) = best {
            let singleton = std::mem::take(&mut components[singleton_index]);
            components[candidate_index].extend(singleton);
        }
    }

    // Establish a deterministic geometric order for diagnostics and the
    // station pose graph.  Content-derived keys only break exact-centre ties;
    // filenames and import indices do not participate.
    for component in &mut components {
        component.sort_by(|left, right| {
            match (world_center(*left), world_center(*right)) {
                (Some(left_center), Some(right_center)) => left_center
                    .y
                    .total_cmp(&right_center.y)
                    .then_with(|| left_center.x.total_cmp(&right_center.x)),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
            .then_with(|| {
                focus_image_direction_key(&images[*left])
                    .cmp(&focus_image_direction_key(&images[*right]))
            })
            .then_with(|| left.cmp(right))
        });
    }
    components.retain(|component| !component.is_empty());
    components.sort_by(|left, right| {
        match (component_center(left), component_center(right)) {
            (Some(left_center), Some(right_center)) => left_center
                .y
                .total_cmp(&right_center.y)
                .then_with(|| left_center.x.total_cmp(&right_center.x)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
        .then_with(|| {
            left.first()
                .copied()
                .unwrap_or(usize::MAX)
                .cmp(&right.first().copied().unwrap_or(usize::MAX))
        })
    });

    let mut group_for_image = vec![usize::MAX; images.len()];
    let mut groups = Vec::with_capacity(components.len());
    for (group_index, members) in components.into_iter().enumerate() {
        // 需求 2.1: the anchor frame is the member whose valid pixels have the
        // highest median Sharpness_Score, with near-equal scores resolved by
        // ascending absolute path.  The previous `members[0]` took the first
        // frame in world-centre order, which is a geometric accident of the
        // bracket: every other frame is then registered into the coordinate
        // system of a possibly defocused reference, and the station's own
        // representative frame (its dimensions, focal length and edge lines
        // feed the station pose graph) carried the least detail of the bracket.
        let anchor_candidates = members
            .iter()
            .map(|&image_index| intra_station::AnchorCandidate {
                path: images[image_index].filename.clone(),
                median_sharpness: focus_frame_median_sharpness(&images[image_index]),
            })
            .collect::<Vec<_>>();
        let anchor = intra_station::select_anchor(&anchor_candidates)
            .map(|position| members[position])
            .unwrap_or(members[0]);
        let anchor_global = locked_homographies.get(&images[anchor].id)?;
        let anchor_inverse = anchor_global.try_inverse()?;
        let mut local_to_anchor = HashMap::new();
        for &image_index in &members {
            let image_global = locked_homographies.get(&images[image_index].id)?;
            local_to_anchor.insert(image_index, anchor_inverse * image_global);
            group_for_image[image_index] = group_index;
        }
        groups.push(FocusCaptureGroup {
            members,
            anchor,
            local_to_anchor,
        });
    }
    Some((groups, group_for_image, grouping))
}

pub(super) fn canonicalize_station_geometry_anchors(
    images: &[ImageInfo],
    groups: &mut [FocusCaptureGroup],
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<()> {
    let pose_order = |left: usize, right: usize| {
        let left_pose = locked_homographies.get(&images[left].id);
        let right_pose = locked_homographies.get(&images[right].id);
        (0..9)
            .map(|index| {
                left_pose
                    .map(|pose| pose[index])
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&right_pose.map(|pose| pose[index]).unwrap_or(f64::INFINITY))
            })
            .find(|order| !order.is_eq())
            .unwrap_or(std::cmp::Ordering::Equal)
    };

    for group in groups {
        let geometry_anchor = group.members.iter().copied().min_by(|&left, &right| {
            focus_image_direction_key(&images[left])
                .cmp(&focus_image_direction_key(&images[right]))
                .then_with(|| pose_order(left, right))
        })?;
        let anchor_inverse = locked_homographies
            .get(&images[geometry_anchor].id)?
            .try_inverse()?;
        let mut local_to_anchor = HashMap::new();
        for &image_index in &group.members {
            let image_global = locked_homographies.get(&images[image_index].id)?;
            local_to_anchor.insert(image_index, anchor_inverse * image_global);
        }
        group.anchor = geometry_anchor;
        group.local_to_anchor = local_to_anchor;
    }
    Some(())
}

pub(super) fn order_focus_render_sources_by_capture_group(
    images: &mut [&ImageInfo],
    capture_group_ids: &HashMap<usize, u8>,
) {
    images.sort_by(|left, right| {
        capture_group_ids
            .get(&left.id)
            .copied()
            .unwrap_or(u8::MAX)
            .cmp(&capture_group_ids.get(&right.id).copied().unwrap_or(u8::MAX))
            .then_with(|| focus_image_direction_key(left).cmp(&focus_image_direction_key(right)))
            .then_with(|| natural_path_cmp(&left.filename, &right.filename))
    });
}
