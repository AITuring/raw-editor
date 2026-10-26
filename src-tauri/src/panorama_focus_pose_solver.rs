//! Focus-stack pose responsibilities extracted from `panorama_stitching`.
//!
//! This file intentionally depends on the parent facade (`super::*`) while
//! keeping its implementation available to the parent and sibling pipeline
//! modules through `pub(super)`. The parent module can re-export these symbols
//! without changing the existing test and orchestration APIs.

use super::*;

pub(super) fn focus_frame_median_sharpness(image: &ImageInfo) -> f64 {
    let analysis = &image.alignment_image;
    let (width, height) = (analysis.width(), analysis.height());
    let spacing = intra_station::analysis_probe_spacing(width.max(height));
    intra_station::median_valid_sharpness(width, height, |x, y| {
        crate::panorama_utils::mosaic::gray_analysis_sharpness(analysis, x, y, spacing)
    })
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
            // into a neighbouring focus bracket just because their priors drift.
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
            world_center(*left)
                .zip(world_center(*right))
                .map(|(left_center, right_center)| {
                    left_center
                        .y
                        .total_cmp(&right_center.y)
                        .then_with(|| left_center.x.total_cmp(&right_center.x))
                })
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    focus_image_direction_key(&images[*left])
                        .cmp(&focus_image_direction_key(&images[*right]))
                })
        });
    }
    components.retain(|component| !component.is_empty());
    components.sort_by(|left, right| {
        component_center(left)
            .zip(component_center(right))
            .map(|(left_center, right_center)| {
                left_center
                    .y
                    .total_cmp(&right_center.y)
                    .then_with(|| left_center.x.total_cmp(&right_center.x))
            })
            .unwrap_or(std::cmp::Ordering::Equal)
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

/// Refine the maximum-support group spanning-tree poses with all available
/// group relations.  The spanning tree is still used as a stable projective
/// initialisation (it preserves focal scale/roll), but every additional edge
/// contributes a robust translation closure residual.  This prevents one
/// tree edge from carrying the accumulated drift of a long snake scan while
/// retaining the projective geometry measured by the image-level matcher.
///
/// We intentionally solve only a world-space translation correction here. A
/// full projective bundle adjustment at station level is under-constrained on
/// flat artwork and can be pulled by repeated calligraphy. The measured
/// homography remains the base pose; the closure correction is bounded and
/// M-estimator weighted, so an occasional repeated-content edge is strongly
/// downweighted without changing the camera scale or local focus ownership.
#[cfg(test)]
pub(super) fn refine_focus_group_projective_tree_poses(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    base_poses: &[Matrix3<f64>],
    reference_group: usize,
    coordinate_scale: f64,
) -> Option<Vec<Matrix3<f64>>> {
    let mut closure_report = stack_report::ClosureReport::default();
    let station_topology = topology::StationTopology {
        row: vec![0; groups.len()],
        column: (0..groups.len() as u32).collect(),
        ..topology::StationTopology::default()
    };
    refine_focus_group_projective_tree_poses_with_report(
        images,
        groups,
        group_edges,
        base_poses,
        reference_group,
        coordinate_scale,
        &station_topology,
        2,
        &mut closure_report,
    )
}

pub(super) fn refine_focus_group_projective_tree_poses_with_report(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    base_poses: &[Matrix3<f64>],
    reference_group: usize,
    _coordinate_scale: f64,
    station_topology: &topology::StationTopology,
    minimum_independent_support: usize,
    closure_report: &mut stack_report::ClosureReport,
) -> Option<Vec<Matrix3<f64>>> {
    closure_report.participating_constraints = group_edges.len();
    closure_report.constraint_weights.clear();
    closure_report.status = stack_report::ClosureStatus::NotRun;
    closure_report.iterations = 0;
    closure_report.residual_median_px = 0.0;
    closure_report.residual_p95_px = 0.0;
    closure_report.max_direct_pair_p95_px = 0.0;
    closure_report.max_corner_correction_px = 0.0;
    closure_report.clamped_stations = 0;
    closure_report.fallback_reason = None;
    closure_report.offending_pair = None;
    if groups.len() < 2 || base_poses.len() != groups.len() || reference_group >= groups.len() {
        return None;
    }

    // Requirement 7.10: a connected graph with no edge beyond its spanning
    // tree has no cycle constraint. Do not enter either joint solver; returning
    // this clone preserves every f64 bit of the tree seed.
    if group_edges.len() <= groups.len().saturating_sub(1) {
        closure_report.status = stack_report::ClosureStatus::NoConstraints;
        closure_report.fallback_reason = Some(degradation::CLOSURE_NO_CONSTRAINTS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_NO_CONSTRAINTS,
            serde_json::json!({
                "accepted_relations": group_edges.len(),
                "station_count": groups.len(),
            }),
        );
        return Some(base_poses.to_vec());
    }

    let mut constraints = Vec::<(usize, usize, [f64; 2], f64)>::new();
    for &(score, left_group, right_group, left_to_right, support, median_error) in group_edges {
        if support < minimum_independent_support
            || left_group >= groups.len()
            || right_group >= groups.len()
            || left_group == right_group
        {
            continue;
        }
        let left_image = &images[groups[left_group].anchor];
        let dimensions = left_image.dimensions();
        let samples = [
            Point2::new(dimensions.0 as f64 * 0.2, dimensions.1 as f64 * 0.2),
            Point2::new(dimensions.0 as f64 * 0.8, dimensions.1 as f64 * 0.2),
            Point2::new(dimensions.0 as f64 * 0.5, dimensions.1 as f64 * 0.5),
            Point2::new(dimensions.0 as f64 * 0.2, dimensions.1 as f64 * 0.8),
            Point2::new(dimensions.0 as f64 * 0.8, dimensions.1 as f64 * 0.8),
        ];
        let mut dx = Vec::new();
        let mut dy = Vec::new();
        for point in samples {
            let Some(left_world) = transformed_point(&base_poses[left_group], point) else {
                continue;
            };
            let Some(right_local) = transformed_point(&left_to_right, point) else {
                continue;
            };
            let Some(right_world) = transformed_point(&base_poses[right_group], right_local) else {
                continue;
            };
            let delta = left_world - right_world;
            if delta.x.is_finite() && delta.y.is_finite() {
                dx.push(delta.x);
                dy.push(delta.y);
            }
        }
        let (Some(dx), Some(dy)) = (median_value(&mut dx), median_value(&mut dy)) else {
            continue;
        };
        let precision = 1.0 / (1.0 + median_error / STATION_RELATION_MAX_MEDIAN_ERROR_PX);
        let weight = support as f64 * score.max(1e-8).sqrt() * precision;
        if weight.is_finite() && weight > 0.0 {
            constraints.push((left_group, right_group, [dx, dy], weight));
        }
    }

    let constraints_are_connected = if constraints.len() + 1 >= groups.len() {
        let mut connectivity = Dsu::new(groups.len());
        for &(left, right, _, _) in &constraints {
            connectivity.union(left, right);
        }
        let reference_root = connectivity.find(reference_group);
        (0..groups.len()).all(|group| connectivity.find(group) == reference_root)
    } else {
        false
    };
    if !constraints_are_connected {
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason =
            Some(degradation::CLOSURE_UNRELIABLE_ITERATIONS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            serde_json::json!({
                "iterations": 0,
                "converged": false,
                "accepted_relations": group_edges.len(),
                "usable_constraints": constraints.len(),
            }),
        );
        return Some(base_poses.to_vec());
    }

    let Some(mut solve) =
        solve_robust_closure_translations(groups.len(), reference_group, &constraints)
    else {
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason =
            Some(degradation::CLOSURE_UNRELIABLE_ITERATIONS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            serde_json::json!({ "iterations": 0, "converged": false }),
        );
        return Some(base_poses.to_vec());
    };
    let iterations = solve.iterations;
    closure_report.iterations = iterations;
    let Some((maximum_correction, clamped_stations)) = clamp_station_translation_corrections(
        images,
        groups,
        group_edges,
        base_poses,
        &mut solve.corrections,
    ) else {
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason =
            Some(degradation::CLOSURE_UNRELIABLE_ITERATIONS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            serde_json::json!({
                "iterations": solve.iterations,
                "converged": false,
                "invalid_correction": true,
            }),
        );
        return Some(base_poses.to_vec());
    };

    closure_report.constraint_weights = constraints
        .iter()
        .map(|&(left, right, observed, _)| {
            let residual = (solve.corrections[right][0] - solve.corrections[left][0] - observed[0])
                .hypot(solve.corrections[right][1] - solve.corrections[left][1] - observed[1]);
            stack_report::ClosureConstraintWeightRecord {
                left,
                right,
                residual_px: residual,
                weight: closure_residual_weight(residual),
            }
        })
        .collect();
    let mut residuals = closure_report
        .constraint_weights
        .iter()
        .map(|relation| relation.residual_px)
        .collect::<Vec<_>>();
    let (median_residual, residual_p95) =
        closure_residual_summary(&mut residuals).unwrap_or((f64::INFINITY, f64::INFINITY));
    closure_report.residual_median_px = median_residual;
    closure_report.residual_p95_px = residual_p95;

    let preliminary = closure_acceptance(
        groups.len(),
        group_edges.len(),
        solve.converged,
        solve.iterations,
        median_residual,
        &[],
    );
    if preliminary != ClosureAcceptance::Accepted {
        let reason = match preliminary {
            ClosureAcceptance::UnreliableResidual => degradation::CLOSURE_UNRELIABLE_RESIDUAL,
            ClosureAcceptance::UnreliableIterations => degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            ClosureAcceptance::NoConstraints => degradation::CLOSURE_NO_CONSTRAINTS,
            ClosureAcceptance::Accepted | ClosureAcceptance::UnreliablePairP95(_) => unreachable!(),
        };
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason = Some(reason.to_string());
        degradation::record_run_degradation(
            reason,
            serde_json::json!({
                "iterations": solve.iterations,
                "converged": solve.converged,
                "residual_median_px": median_residual,
                "residual_limit_px": CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX,
            }),
        );
        return Some(base_poses.to_vec());
    }

    let mut refined = Vec::with_capacity(groups.len());
    for (group_index, pose) in base_poses.iter().enumerate() {
        let correction = solve.corrections[group_index];
        let translation = Matrix3::new(
            1.0,
            0.0,
            correction[0],
            0.0,
            1.0,
            correction[1],
            0.0,
            0.0,
            1.0,
        );
        let corrected = translation * pose;
        if corrected.try_inverse().is_none()
            || !homography_preserves_focus_orientation(
                &corrected,
                images[groups[group_index].anchor].dimensions(),
            )
        {
            closure_report.status = stack_report::ClosureStatus::Unreliable;
            closure_report.fallback_reason =
                Some(degradation::CLOSURE_UNRELIABLE_PAIR_P95.to_string());
            degradation::record_run_degradation(
                degradation::CLOSURE_UNRELIABLE_PAIR_P95,
                serde_json::json!({
                    "station": group_index,
                    "invalid_corrected_pose": true,
                }),
            );
            return Some(base_poses.to_vec());
        }
        refined.push(corrected);
    }

    let direct_pairs = closure_direct_pair_p95s(images, groups, group_edges, &refined);
    closure_report.max_direct_pair_p95_px = direct_pairs
        .iter()
        .map(|pair| pair.p95_px)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    match closure_acceptance(
        groups.len(),
        group_edges.len(),
        solve.converged,
        solve.iterations,
        median_residual,
        &direct_pairs,
    ) {
        ClosureAcceptance::Accepted => {}
        ClosureAcceptance::UnreliablePairP95(pair) => {
            let offending = closure_offending_pair_record(pair, station_topology);
            closure_report.status = stack_report::ClosureStatus::Unreliable;
            closure_report.fallback_reason =
                Some(degradation::CLOSURE_UNRELIABLE_PAIR_P95.to_string());
            closure_report.offending_pair = Some(offending);
            closure_report.max_corner_correction_px = 0.0;
            closure_report.clamped_stations = 0;
            degradation::record_run_degradation(
                degradation::CLOSURE_UNRELIABLE_PAIR_P95,
                serde_json::json!({
                    "left": offending.left,
                    "left_row": offending.left_row,
                    "left_column": offending.left_column,
                    "right": offending.right,
                    "right_row": offending.right_row,
                    "right_column": offending.right_column,
                    "p95_px": offending.p95_px,
                    "limit_px": CLOSURE_DIRECT_PAIR_P95_LIMIT_PX,
                }),
            );
            return Some(base_poses.to_vec());
        }
        ClosureAcceptance::NoConstraints
        | ClosureAcceptance::UnreliableResidual
        | ClosureAcceptance::UnreliableIterations => unreachable!(),
    }

    closure_report.status = stack_report::ClosureStatus::Converged;
    closure_report.max_corner_correction_px = maximum_correction;
    closure_report.clamped_stations = clamped_stations;
    if clamped_stations > 0 {
        degradation::record_run_degradation(
            degradation::CLOSURE_CORRECTION_CLAMPED,
            serde_json::json!({
                "clamped_stations": clamped_stations,
                "max_corner_correction_px": maximum_correction,
            }),
        );
    }
    println!(
        "  - Focus capture-group closure accepted: {} relations, {} iteration(s), median/P95 residual {:.2}/{:.2}px, maximum direct-pair P95 {:.2}px, maximum corner correction {:.2}px, {} clamped station(s)",
        constraints.len(),
        solve.iterations,
        median_residual,
        residual_p95,
        closure_report.max_direct_pair_p95_px,
        maximum_correction,
        clamped_stations,
    );
    Some(refined)
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

pub(super) fn focus_station_topology(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    station_homographies: &HashMap<usize, Matrix3<f64>>,
) -> topology::StationTopology {
    let observations = groups
        .iter()
        .enumerate()
        .map(|(station_index, group)| {
            let anchor = &images[group.anchor];
            let pose = station_homographies.get(&anchor.id);
            let center = pose.and_then(|pose| {
                transformed_point(
                    pose,
                    Point2::new(anchor.width as f64 * 0.5, anchor.height as f64 * 0.5),
                )
            });
            let effective_width = pose.and_then(|pose| {
                let corners = [
                    Point2::new(0.0, 0.0),
                    Point2::new(anchor.width as f64, 0.0),
                    Point2::new(0.0, anchor.height as f64),
                    Point2::new(anchor.width as f64, anchor.height as f64),
                ];
                let projected = corners
                    .into_iter()
                    .map(|corner| transformed_point(pose, corner))
                    .collect::<Option<Vec<_>>>()?;
                let min_x = projected
                    .iter()
                    .map(|point| point.x)
                    .fold(f64::INFINITY, f64::min);
                let max_x = projected
                    .iter()
                    .map(|point| point.x)
                    .fold(f64::NEG_INFINITY, f64::max);
                let width = max_x - min_x;
                (width.is_finite() && width > 0.0).then_some(width)
            });
            topology::StationObservation {
                station_index,
                world_x: center.map(|point| point.x),
                world_y: center.map(|point| point.y),
                effective_width,
                filename: anchor.filename.clone(),
            }
        })
        .collect::<Vec<_>>();

    // Candidate scores are the transform-estimated overlap area divided by one
    // station's effective area. Keep the matrix outside StationTopology so the
    // topology result remains pose-free (requirements 5.4 and 5.5).
    let mut overlap_scores = BTreeMap::new();
    for left in 0..groups.len() {
        let left_anchor = &images[groups[left].anchor];
        let Some(left_to_world) = station_homographies.get(&left_anchor.id) else {
            continue;
        };
        for right in (left + 1)..groups.len() {
            let right_anchor = &images[groups[right].anchor];
            let Some(world_to_right) = station_homographies
                .get(&right_anchor.id)
                .and_then(|pose| pose.try_inverse())
            else {
                continue;
            };
            let left_to_right = world_to_right * left_to_world;
            overlap_scores.insert(
                (left, right),
                panorama_transform_overlap_support(
                    &left_to_right,
                    left_anchor.dimensions(),
                    right_anchor.dimensions(),
                ),
            );
        }
    }

    topology::infer_station_topology(&observations, &overlap_scores)
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

pub(super) fn absolute_source_path(path: &str) -> String {
    let path = Path::new(path);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(path)
    };
    absolute.to_string_lossy().into_owned()
}

pub(super) fn station_graph_components_from_members(
    station_members: &[Vec<String>],
    relation_pairs: &[(usize, usize)],
) -> Vec<Vec<String>> {
    let mut connectivity = Dsu::new(station_members.len());
    for &(left, right) in relation_pairs {
        if left < station_members.len() && right < station_members.len() {
            connectivity.union(left, right);
        }
    }
    let mut stations_by_root = BTreeMap::<usize, Vec<usize>>::new();
    for station_index in 0..station_members.len() {
        let root = connectivity.find(station_index);
        stations_by_root
            .entry(root)
            .or_default()
            .push(station_index);
    }
    stations_by_root
        .into_values()
        .map(|station_indices| {
            let mut members = station_indices
                .into_iter()
                .flat_map(|station_index| station_members[station_index].iter().cloned())
                .collect::<Vec<_>>();
            members.sort();
            members
        })
        .collect()
}

pub(super) fn station_graph_components(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    relation_pairs: &[(usize, usize)],
) -> Vec<Vec<String>> {
    let station_members = groups
        .iter()
        .map(|group| {
            group
                .members
                .iter()
                .map(|&image_index| absolute_source_path(&images[image_index].filename))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    station_graph_components_from_members(&station_members, relation_pairs)
}

/// Populate the canonical connectivity report and return the exact rejected
/// diagnostic payload when the accepted Station_Relation graph is disconnected.
pub(super) fn station_graph_connectivity_rejection(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    relation_pairs: &[(usize, usize)],
    report: &mut stack_report::StationRelationsReport,
) -> Option<serde_json::Value> {
    let components = station_graph_components(images, groups, relation_pairs);
    report.connectivity.components = components.len();
    report.connectivity.component_member_counts = components.iter().map(Vec::len).collect();
    report.connectivity.component_members = components.clone();
    (components.len() > 1).then(|| {
        let component_details = components
            .iter()
            .map(|members| {
                serde_json::json!({
                    "member_count": members.len(),
                    "members": members,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "component_count": components.len(),
            "components": component_details,
            "actions": ["按连续场景重新分组", "补拍增加重叠"],
        })
    })
}

pub(super) fn geometry_disconnected_user_error(component_count: usize) -> String {
    format!(
        "机位几何图包含 {component_count} 个不连通分量，无法导出完整拼图。可执行操作：“按连续场景重新分组”或“补拍增加重叠”。"
    )
}

#[cfg(test)]
pub(super) fn solve_focus_capture_group_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
    translation_geometry_verified: &mut bool,
) -> HashMap<usize, Matrix3<f64>> {
    let mut closure_report = stack_report::ClosureReport::default();
    let mut station_relations_report = stack_report::StationRelationsReport::default();
    solve_focus_capture_group_poses_with_report(
        images,
        matches,
        locked_homographies,
        reference_index,
        translation_geometry_verified,
        &mut closure_report,
        &mut station_relations_report,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StationRelationEvidenceKind {
    SourceRaw,
    VirtualTile,
}

impl StationRelationEvidenceKind {
    fn minimum_independent_support(self) -> usize {
        match self {
            Self::SourceRaw => 2,
            // One fused station tile is already the requirement-6.1 primary
            // observation.  This is not a fabricated second source: the
            // evidence kind explicitly exempts it from requirement 6.2.
            Self::VirtualTile => 1,
        }
    }
}

pub(super) fn solve_focus_capture_group_poses_with_report(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
    translation_geometry_verified: &mut bool,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
) -> HashMap<usize, Matrix3<f64>> {
    solve_station_poses_for_evidence_with_report(
        images,
        matches,
        locked_homographies,
        reference_index,
        translation_geometry_verified,
        closure_report,
        station_relations_report,
        StationRelationEvidenceKind::SourceRaw,
        None,
        None,
    )
}

/// Coverage_Masks the station solver reads for Virtual_Tile evidence: every
/// re-check it makes on a tile pair looks only at pixels both tiles cover
/// (需求 6.1).
pub(super) fn solve_virtual_tile_station_hypotheses_with_report(
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
    hypotheses: &VirtualTileRelationHypotheses,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
) -> HashMap<usize, Matrix3<f64>> {
    let selected = select_virtual_tile_relation_hypotheses(tiles, hypotheses, coverages);
    solve_virtual_tile_station_poses_with_report(
        tiles,
        coverages,
        &selected,
        initial_homographies,
        station_topology,
        closure_report,
        station_relations_report,
    )
}

pub(super) fn solve_virtual_tile_station_poses_with_report(
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
) -> HashMap<usize, Matrix3<f64>> {
    let mut verified = false;
    solve_station_poses_for_evidence_with_report(
        tiles,
        matches,
        initial_homographies,
        0,
        &mut verified,
        closure_report,
        station_relations_report,
        StationRelationEvidenceKind::VirtualTile,
        Some(station_topology),
        Some(coverages),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve_station_poses_for_evidence_with_report(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
    translation_geometry_verified: &mut bool,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
    evidence_kind: StationRelationEvidenceKind,
    supplied_topology: Option<&topology::StationTopology>,
    coverages: Option<VirtualTileSolverCoverages<'_>>,
) -> HashMap<usize, Matrix3<f64>> {
    *translation_geometry_verified = false;
    *station_relations_report = stack_report::StationRelationsReport::default();
    let (mut groups, group_for_image) = match evidence_kind {
        StationRelationEvidenceKind::SourceRaw => {
            let Some(groups) = focus_capture_groups(images, matches, locked_homographies) else {
                return locked_homographies.clone();
            };
            groups
        }
        StationRelationEvidenceKind::VirtualTile => {
            let groups = (0..images.len())
                .map(|index| FocusCaptureGroup {
                    members: vec![index],
                    anchor: index,
                    local_to_anchor: HashMap::from([(index, Matrix3::identity())]),
                })
                .collect::<Vec<_>>();
            (groups, (0..images.len()).collect::<Vec<_>>())
        }
    };
    // The Focus_Fuser anchor follows requirement 2.1 and may legitimately use
    // the source path for a near-sharpness tie. Station geometry cannot inherit
    // that rename-sensitive coordinate choice (requirements 1.3/5.8), so the
    // pose graph uses an equivalent content/geometry-derived representative.
    if canonicalize_station_geometry_anchors(images, &mut groups, locked_homographies).is_none() {
        return locked_homographies.clone();
    }
    if groups.len() < 2 || reference_index >= images.len() {
        return locked_homographies.clone();
    }
    // 需求 1.6: an isolated source belongs to no Capture_Station, so it cannot
    // anchor the station pose graph.  Keep the image-level geometry instead of
    // rooting the tree at a station that does not exist.
    if group_for_image[reference_index] == usize::MAX {
        return locked_homographies.clone();
    }
    // Task 9.1 is part of the production station-relation path: infer the
    // displacement-only topology before collecting relation evidence, retain
    // it for Stack_Report, and use it below as the candidate eligibility gate.
    // Matrix-valued poses are reduced to centres/widths before entering the
    // topology module (需求 5.2 / 5.4).
    let station_topology = supplied_topology
        .cloned()
        .unwrap_or_else(|| focus_station_topology(images, &groups, locked_homographies));
    if evidence_kind == StationRelationEvidenceKind::SourceRaw {
        topology::record_run_topology(station_topology.clone());
    }
    // An isolated source keeps the image-level pose the station solve cannot
    // refine, so the returned map still covers the whole selection.
    let carry_unstationed_poses = |solved: &mut HashMap<usize, Matrix3<f64>>| {
        for (image_index, image) in images.iter().enumerate() {
            if group_for_image[image_index] != usize::MAX {
                continue;
            }
            if let Some(&pose) = locked_homographies.get(&image.id) {
                solved.insert(image.id, pose);
            }
        }
    };

    #[cfg(test)]
    if std::env::var_os("RAW_EDITOR_STACK_GROUP_DIAGNOSTICS").is_some() {
        println!("  - Focus capture groups in capture order:");
        for (group_index, group) in groups.iter().enumerate() {
            let names = group
                .members
                .iter()
                .map(|&image_index| {
                    Path::new(&images[image_index].filename)
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect::<Vec<_>>()
                .join(",");
            println!("    group {group_index}: {names}");
        }
    }

    // Station-pair candidates are keyed by a group pair and later compared by
    // score; both the candidate order inside a pair and the order of the pairs
    // themselves must not depend on the hash seed (需求 14.6).  The map is one
    // entry per station pair, so an ordered container costs nothing here.
    let mut candidates = BTreeMap::<(usize, usize), Vec<FocusGroupEdgeCandidate>>::new();
    let mut non_invertible_candidate_pairs = BTreeSet::<(usize, usize)>::new();
    for (&(source_index, target_index), match_info) in sorted_pairs(matches) {
        if source_index >= images.len()
            || target_index >= images.len()
            || match_info.sequence_bridge
        {
            continue;
        }
        // A single coarse correlation peak is not geometry evidence.  A
        // component-recovery edge, however, is inserted only after at least
        // two different focus layers on both sides agree on the same planar
        // transform.  Preserve that independent support in the station-level
        // solver instead of using the edge merely to make the image graph
        // connected and then falling back to a repeated-texture image pose.
        let consensus_coarse =
            match_info.coarse_bridge && !match_info.top_candidate_points.is_empty();
        if (match_info.coarse_bridge && !consensus_coarse)
            || (!consensus_coarse && match_info.points.len() < FOCUS_MODEL_MIN_INLIERS)
        {
            continue;
        }
        // 需求 1.6: an isolated source is a member of no Capture_Station, so it
        // contributes no station relation either.
        if group_for_image[source_index] == usize::MAX
            || group_for_image[target_index] == usize::MAX
        {
            continue;
        }
        if group_for_image[source_index] == group_for_image[target_index] {
            continue;
        }
        if !station_topology
            .pair_is_eligible(group_for_image[source_index], group_for_image[target_index])
        {
            continue;
        }
        let canonical =
            match canonical_relation_view(images, source_index, target_index, match_info) {
                Ok(view) => view,
                Err(CanonicalRelationViewError::NonInvertible) => {
                    let source_group = group_for_image[source_index];
                    let target_group = group_for_image[target_index];
                    non_invertible_candidate_pairs.insert((
                        source_group.min(target_group),
                        source_group.max(target_group),
                    ));
                    continue;
                }
                Err(CanonicalRelationViewError::IndexOutOfBounds {
                    left,
                    right,
                    image_count,
                }) => panic!(
                    "canonical relation pair ({left}, {right}) is outside {image_count} images"
                ),
            };
        let quality_source = canonical.source_index;
        let quality_target = canonical.target_index;
        let quality_inverted = canonical.inverted;
        let quality_homography = canonical.homography;
        let source_group = group_for_image[quality_source];
        let target_group = group_for_image[quality_target];
        let quality_points: Cow<'_, [(Point2<f64>, Point2<f64>)]> = if quality_inverted {
            Cow::Owned(
                match_info
                    .points
                    .iter()
                    .map(|&(source, target)| (target, source))
                    .collect(),
            )
        } else {
            Cow::Borrowed(&match_info.points)
        };
        let Some(source_local) = groups[source_group].local_to_anchor.get(&quality_source) else {
            continue;
        };
        let Some(source_local_inverse) = (*source_local).try_inverse() else {
            continue;
        };
        let Some(target_local) = groups[target_group].local_to_anchor.get(&quality_target) else {
            continue;
        };
        let source_anchor_to_target_anchor =
            target_local * quality_homography * source_local_inverse;
        let (key, transform) = if source_group < target_group {
            ((source_group, target_group), source_anchor_to_target_anchor)
        } else {
            let Some(inverse) = source_anchor_to_target_anchor.try_inverse() else {
                continue;
            };
            ((target_group, source_group), inverse)
        };
        // Acceptance evidence belongs to the immutable measured source pair,
        // not to whichever near-tied focus frames happen to become station
        // anchors. Requirement 2.1 permits a path tie-break for that anchor;
        // requirements 1.3/5.8 therefore require these measurements to remain
        // in the canonical content-derived source direction. The composed
        // anchor transform and anchor points remain the pose-graph payload.
        let measurement_source_dimensions = images[quality_source].dimensions();
        let measurement_target_dimensions = images[quality_target].dimensions();
        // Virtual_Tile evidence is re-checked only where both tiles are
        // covered; a missing mask leaves the relation unmeasured (rejected).
        let quality_coverage_pair = match coverages {
            None => Some(None),
            Some(coverages) => coverages
                .quality
                .get(quality_source)
                .zip(coverages.quality.get(quality_target))
                .map(Some),
        };
        // Requirement 6.4's overlap area: the frames' geometric intersection
        // for source images, the doubly covered pixels for Virtual_Tiles.
        let overlap_measure = match coverages {
            None => Some(StationOverlapMeasure::Rectangles),
            Some(coverages) => coverages.overlap_measure(quality_source, quality_target),
        };
        let dense_measurements = quality_coverage_pair.and_then(|coverages| {
            focus_overlap_quality_measurements_on_coverage(
                &images[quality_source],
                &images[quality_target],
                &quality_homography,
                coverages,
            )
        });
        let Some(measured) = measure_station_relation_candidate(
            match_info.inliers.max(quality_points.len()),
            &quality_homography,
            &quality_points,
            measurement_source_dimensions,
            measurement_target_dimensions,
            overlap_measure,
            dense_measurements,
        ) else {
            continue;
        };
        let measurements = measured.measurements;
        candidates
            .entry(key)
            .or_default()
            .push(FocusGroupEdgeCandidate {
                transform,
                quality: measured.quality,
                independent_support: if consensus_coarse { 2 } else { 1 },
                inlier_count: measurements.inliers,
                median_error_px: measurements.median_error_px,
                scale_ratio: measurements.scale_ratio,
                spatial_support: measurements.spatial_support,
                low_frequency_mean_relative_difference: measurements
                    .low_frequency_mean_relative_difference,
                edge_strength_ratio: measurements.edge_strength_ratio,
                median_edge_orientation_difference_degrees: measurements
                    .median_edge_orientation_difference_degrees,
                preserves_convex_orientation: measurements.preserves_convex_orientation,
            });
    }

    let final_non_invertible_pairs = non_invertible_candidate_pairs
        .into_iter()
        .filter(|pair| !candidates.contains_key(pair))
        .collect::<BTreeSet<_>>();
    for &(left, right) in &final_non_invertible_pairs {
        station_relations_report.record_non_invertible_canonical_pair(left, right);
    }

    // One station pair collects one candidate per accepted focus layer
    // relation, and the consensus search below reads that vector in order: it
    // accumulates `supporting_quality` along it and keeps the first candidate on
    // a score tie.  Pushing in `matches` key order would make both inherit the
    // import positions the keys are built from, so re-order every candidate
    // list by its own measured content (需求 14.6).
    for pair_candidates in candidates.values_mut() {
        pair_candidates.sort_by(|left, right| {
            right
                .quality
                .total_cmp(&left.quality)
                .then_with(|| right.independent_support.cmp(&left.independent_support))
                .then_with(|| {
                    focus_group_candidate_key(left).cmp(&focus_group_candidate_key(right))
                })
        });
    }

    let coordinate_scale = images
        .iter()
        .map(|image| image.width.max(image.height) as f64)
        .fold(1.0, f64::max);
    let consensus_threshold = STATION_RELATION_MAX_MEDIAN_ERROR_PX;
    let mut group_edges = Vec::<(f64, usize, usize, Matrix3<f64>, usize, f64)>::new();
    let mut station_relation_inlier_counts = BTreeMap::<(usize, usize), usize>::new();
    let mut evaluated_candidate_pairs = final_non_invertible_pairs
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut consensus_pair_count = 0usize;
    for ((left_group, right_group), pair_candidates) in candidates {
        evaluated_candidate_pairs.insert((left_group, right_group));
        let dimensions = images[groups[left_group].anchor].dimensions();
        let mut best: Option<(f64, FocusGroupEdgeCandidate, usize, f64, usize)> = None;
        for candidate in &pair_candidates {
            let mut supporting_errors = Vec::new();
            let mut supporting_quality = 0.0;
            let mut independent_support = 0usize;
            let mut inlier_count = 0usize;
            for other in &pair_candidates {
                let Some(error) = focus_transform_disagreement_px(
                    &candidate.transform,
                    &other.transform,
                    dimensions,
                ) else {
                    continue;
                };
                if error <= consensus_threshold {
                    supporting_errors.push(error);
                    supporting_quality += other.quality;
                    independent_support += other.independent_support;
                    inlier_count = inlier_count.saturating_add(other.inlier_count);
                }
            }
            let support = independent_support;
            let discard = (evidence_kind == StationRelationEvidenceKind::SourceRaw)
                .then(|| single_layer_station_evidence_discard(support, left_group, right_group))
                .flatten();
            if discard.is_none() {
                let consensus_error =
                    median_value(&mut supporting_errors).unwrap_or(consensus_threshold);
                let score = (support * support) as f64 * supporting_quality
                    / (1.0 + consensus_error / consensus_threshold.max(1.0));
                if best
                    .as_ref()
                    .is_none_or(|(best_score, _, _, _, _)| score > *best_score)
                {
                    best = Some((score, *candidate, support, consensus_error, inlier_count));
                }
                continue;
            }
            degradation::record_run_degradation(
                degradation::STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED,
                discard.expect("source evidence discard was checked above"),
            );
        }
        let Some((score, candidate, support, consensus_error, inlier_count)) = best else {
            if evidence_kind == StationRelationEvidenceKind::VirtualTile {
                record_station_relation_no_fit(left_group, right_group, station_relations_report);
            }
            continue;
        };
        let measurements = StationRelationMeasurements {
            inliers: inlier_count,
            median_error_px: candidate.median_error_px,
            scale_ratio: candidate.scale_ratio,
            spatial_support: candidate.spatial_support,
            low_frequency_mean_relative_difference: candidate
                .low_frequency_mean_relative_difference,
            edge_strength_ratio: candidate.edge_strength_ratio,
            median_edge_orientation_difference_degrees: candidate
                .median_edge_orientation_difference_degrees,
            preserves_convex_orientation: candidate.preserves_convex_orientation,
        };
        if !station_relation_is_accepted_with_report(
            left_group,
            right_group,
            consensus_error,
            &measurements,
            station_relations_report,
        ) {
            continue;
        }
        consensus_pair_count += 1;
        station_relations_report
            .accepted
            .push(stack_report::AcceptedStationRelationRecord {
                left: left_group,
                right: right_group,
                evidence_kind: match evidence_kind {
                    StationRelationEvidenceKind::SourceRaw => {
                        stack_report::StationRelationEvidenceKind::SourceRawConsensus
                    }
                    StationRelationEvidenceKind::VirtualTile => {
                        stack_report::StationRelationEvidenceKind::VirtualTile
                    }
                },
                inliers: measurements.inliers,
                spatial_support: measurements.spatial_support,
                scale_ratio: measurements.scale_ratio,
                median_error_px: measurements.median_error_px,
                independent_support: support,
            });
        station_relation_inlier_counts.insert((left_group, right_group), inlier_count);
        group_edges.push((
            score,
            left_group,
            right_group,
            candidate.transform,
            support,
            measurements.median_error_px,
        ));
    }
    if evidence_kind == StationRelationEvidenceKind::VirtualTile {
        for candidate in &station_topology.candidates {
            let pair = (
                candidate.left.min(candidate.right),
                candidate.left.max(candidate.right),
            );
            if pair.0 >= groups.len()
                || pair.1 >= groups.len()
                || !evaluated_candidate_pairs.insert(pair)
            {
                continue;
            }
            record_station_relation_no_fit(pair.0, pair.1, station_relations_report);
        }
    }
    closure_report.participating_constraints = group_edges.len();
    closure_report.constraint_weights.clear();

    // Requirements 6.7 / 12.4: an accepted Station_Relation graph must cover
    // every Capture_Station.  Image-level locked poses are diagnostics and a
    // spanning-tree seed, not permission to render independent components into
    // one apparently complete mosaic.
    let relation_pairs = group_edges
        .iter()
        .map(|edge| (edge.1, edge.2))
        .collect::<Vec<_>>();
    if let Some(rejection) = station_graph_connectivity_rejection(
        images,
        &groups,
        &relation_pairs,
        station_relations_report,
    ) {
        let component_count = station_relations_report.connectivity.components;
        degradation::record_run_degradation(degradation::GEOMETRY_DISCONNECTED, rejection);
        println!(
            "  - Focus station geometry rejected: {component_count} disconnected component(s)"
        );
        return HashMap::new();
    }

    let reference_group = group_for_image[reference_index];
    #[cfg(test)]
    if std::env::var_os("RAW_EDITOR_STACK_GROUP_DIAGNOSTICS").is_some() {
        let mut diagnostics = group_edges.clone();
        diagnostics.sort_by_key(|edge| (edge.1, edge.2));
        println!("  - Focus capture-group relations:");
        for &(_, left, right, transform, support, median_error) in &diagnostics {
            let image = &images[groups[left].anchor];
            let center = Point2::new(image.width as f64 * 0.5, image.height as f64 * 0.5);
            let mapped = transformed_point(&transform, center).unwrap_or(center);
            println!(
                "    {left}->{right}: gap={} support={support} error={median_error:.1}px center=({:.1},{:.1})",
                right.saturating_sub(left),
                center.x - mapped.x,
                center.y - mapped.y,
            );
        }
    }
    // Requirement 7.1: accepted-relation inlier count is the only spanning-tree
    // weight. Composite quality continues to inform the joint optimizer, but it
    // cannot change the projective seed. Equal inlier counts use the inferred
    // station row/column coordinates as the complete deterministic tie-break.
    let tree_relations = group_edges
        .iter()
        .map(|&(_, left, right, transform, _, _)| {
            let key = (left.min(right), left.max(right));
            let inlier_count = station_relation_inlier_counts
                .get(&key)
                .copied()
                .unwrap_or_default();
            (
                left,
                right,
                transform.try_inverse().is_some().then_some(inlier_count),
            )
        })
        .collect::<Vec<_>>();
    let selectable_relations = tree_relations
        .iter()
        .filter_map(|&(left, right, inlier_count)| {
            inlier_count.map(|inlier_count| (left, right, inlier_count))
        })
        .collect::<Vec<_>>();
    let selected_tree_indices = maximum_inlier_spanning_tree_indices(
        groups.len(),
        &station_topology,
        &selectable_relations,
    );
    let selectable_group_edge_indices = tree_relations
        .iter()
        .enumerate()
        .filter_map(|(index, relation)| relation.2.map(|_| index))
        .collect::<Vec<_>>();

    let mut group_adjacency = HashMap::<usize, Vec<(usize, Matrix3<f64>)>>::new();
    let mut tree_edges = 0usize;
    for selectable_index in selected_tree_indices {
        let group_edge_index = selectable_group_edge_indices[selectable_index];
        let (_, left_group, right_group, left_to_right, _, _) = group_edges[group_edge_index];
        let right_to_left = left_to_right
            .try_inverse()
            .expect("tree candidates were filtered to invertible transforms");
        group_adjacency
            .entry(left_group)
            .or_default()
            .push((right_group, right_to_left));
        group_adjacency
            .entry(right_group)
            .or_default()
            .push((left_group, left_to_right));
        tree_edges += 1;
    }
    closure_report.spanning_tree_edges = tree_edges;
    let mut group_poses = vec![None; groups.len()];
    group_poses[reference_group] = locked_homographies
        .get(&images[groups[reference_group].anchor].id)
        .copied();
    let mut pending = VecDeque::from([reference_group]);
    while let Some(current) = pending.pop_front() {
        let Some(current_pose) = group_poses[current] else {
            continue;
        };
        for &(neighbor, neighbor_to_current) in group_adjacency.get(&current).into_iter().flatten()
        {
            if group_poses[neighbor].is_some() {
                continue;
            }
            group_poses[neighbor] = Some(current_pose * neighbor_to_current);
            pending.push_back(neighbor);
        }
    }
    if tree_edges + 1 == groups.len() && group_poses.iter().all(Option::is_some) {
        let base_group_poses = group_poses
            .iter()
            .map(|pose| pose.expect("connected group tree must provide every pose"))
            .collect::<Vec<_>>();
        // The maximum-support tree keeps each station's full 8-DOF measured
        // homography. Task 9.9 integrates the 9.6-9.8 robust closure path here
        // as the only joint update, so no earlier projective solve can bypass
        // its convergence, median-residual, and direct-pair P95 acceptance
        // gates.
        // Keep the maximum-support tree as the projective initialisation, but
        // let every reliable closure edge vote on a bounded world translation
        // correction. This is the generic path for both horizontal/vertical
        // scan closures and snake cross-column links; no filename topology is
        // assumed here.
        let refined_group_poses = refine_focus_group_projective_tree_poses_with_report(
            images,
            &groups,
            &group_edges,
            &base_group_poses,
            reference_group,
            coordinate_scale,
            &station_topology,
            evidence_kind.minimum_independent_support(),
            closure_report,
        )
        .unwrap_or_else(|| base_group_poses.clone());
        let selected_group_poses = select_station_pose_candidate(
            images,
            &groups,
            &refined_group_poses,
            &base_group_poses,
            "closure_refinement",
        );
        let build_solved = |poses: &[Matrix3<f64>]| {
            let mut solved = HashMap::new();
            for (group_index, group) in groups.iter().enumerate() {
                let group_pose = poses[group_index];
                for &image_index in &group.members {
                    let replacement = group_pose * group.local_to_anchor[&image_index];
                    if replacement.try_inverse().is_none()
                        || !homography_preserves_focus_orientation(
                            &replacement,
                            images[image_index].dimensions(),
                        )
                    {
                        return None;
                    }
                    solved.insert(images[image_index].id, replacement);
                }
            }
            carry_unstationed_poses(&mut solved);
            Some(solved)
        };
        let solved =
            build_solved(&selected_group_poses).or_else(|| build_solved(&base_group_poses));
        let Some(solved) = solved else {
            println!(
                "  - Focus capture-group planar geometry produced an unstable pose; keeping image-level geometry"
            );
            return locked_homographies.clone();
        };
        println!(
            "  - Focus capture-group planar geometry seeded from {tree_edges} pixel-verified relation(s)"
        );
        *translation_geometry_verified = true;
        return solved;
    }
    if let Some(mut translation_poses) = solve_focus_group_translation_poses(
        images,
        &groups,
        &group_edges,
        locked_homographies,
        reference_group,
        coordinate_scale,
    ) {
        // The translation solve has already required a connected group graph
        // and robust closure. Do not subsequently "refine" it with every
        // long-range planar relation: repeated figures and calligraphy can
        // agree as homographies yet pull a valid scan position by hundreds of
        // pixels. Keep the solved camera positions and only retain the local
        // focus-plane transform inside each capture group.
        correct_focus_group_roll_from_long_edges(
            images,
            &groups,
            &mut translation_poses,
            locked_homographies,
            reference_group,
        );
        let mut solved = HashMap::new();
        for (group_index, group) in groups.iter().enumerate() {
            let group_pose = translation_poses[group_index];
            for &image_index in &group.members {
                let replacement = group_pose * group.local_to_anchor[&image_index];
                if replacement.try_inverse().is_none()
                    || !homography_preserves_focus_orientation(
                        &replacement,
                        images[image_index].dimensions(),
                    )
                {
                    println!(
                        "  - Focus capture-group translation geometry produced an unstable pose for '{}'",
                        images[image_index].filename
                    );
                    return locked_homographies.clone();
                }
                solved.insert(images[image_index].id, replacement);
            }
        }
        carry_unstationed_poses(&mut solved);
        println!(
            "  - Focus capture-group translation geometry accepted for {} groups without projective re-refinement",
            groups.len()
        );
        *translation_geometry_verified = true;
        return solved;
    }
    let translation_geometry_seeded = false;
    let use_consensus_group_poses = translation_geometry_seeded
        || (tree_edges + 1 == groups.len()
            && group_poses.iter().all(Option::is_some)
            && group_poses.iter().all(|pose| {
                pose.is_some_and(|pose| {
                    pose.try_inverse().is_some()
                        && homography_preserves_focus_orientation(
                            &pose,
                            images[groups[reference_group].anchor].dimensions(),
                        )
                })
            }));
    if use_consensus_group_poses {
        println!(
            "  - Focus capture-group geometry seeded from {tree_edges} multi-member consensus relation(s)"
        );
    }
    let mut constraints = Vec::<(usize, usize, [f64; 2], f64, usize)>::new();
    for &(score, left_group, right_group, left_to_right, support, median_error) in &group_edges {
        // One repeated character can produce one excellent-looking edge. A
        // group may move only when two independent member pairs agree on the
        // same anchor-to-anchor transform.
        if support < 2 {
            continue;
        }
        let left_global = if use_consensus_group_poses {
            group_poses[left_group].unwrap()
        } else {
            let Some(pose) = locked_homographies
                .get(&images[groups[left_group].anchor].id)
                .copied()
            else {
                continue;
            };
            pose
        };
        let right_global = if use_consensus_group_poses {
            group_poses[right_group].unwrap()
        } else {
            let Some(pose) = locked_homographies
                .get(&images[groups[right_group].anchor].id)
                .copied()
            else {
                continue;
            };
            pose
        };
        let dimensions = images[groups[left_group].anchor].dimensions();
        let samples = [
            Point2::new(dimensions.0 as f64 * 0.2, dimensions.1 as f64 * 0.2),
            Point2::new(dimensions.0 as f64 * 0.8, dimensions.1 as f64 * 0.2),
            Point2::new(dimensions.0 as f64 * 0.5, dimensions.1 as f64 * 0.5),
            Point2::new(dimensions.0 as f64 * 0.2, dimensions.1 as f64 * 0.8),
            Point2::new(dimensions.0 as f64 * 0.8, dimensions.1 as f64 * 0.8),
        ];
        let mut dx = Vec::new();
        let mut dy = Vec::new();
        for point in samples {
            let Some(left_world) = transformed_point(&left_global, point) else {
                continue;
            };
            let Some(right_local) = transformed_point(&left_to_right, point) else {
                continue;
            };
            let Some(right_world) = transformed_point(&right_global, right_local) else {
                continue;
            };
            let delta = right_world - left_world;
            if delta.x.is_finite() && delta.y.is_finite() {
                dx.push(delta.x);
                dy.push(delta.y);
            }
        }
        let (Some(dx), Some(dy)) = (median_value(&mut dx), median_value(&mut dy)) else {
            continue;
        };
        let weight =
            (support as f64) * score.sqrt() / (1.0 + median_error / consensus_threshold.max(1.0));
        if weight.is_finite() && weight > 0.0 {
            constraints.push((left_group, right_group, [dx, dy], weight, support));
        }
    }
    if constraints.is_empty() {
        println!(
            "  - Focus capture-group translation solve: no multi-edge constraints; keeping image-level geometry"
        );
        return locked_homographies.clone();
    }

    let closure_constraints = constraints
        .iter()
        .map(|&(left, right, observed, base_weight, _)| {
            (left, right, [-observed[0], -observed[1]], base_weight)
        })
        .collect::<Vec<_>>();
    let Some(mut closure_solve) =
        solve_robust_closure_translations(groups.len(), reference_group, &closure_constraints)
    else {
        return locked_homographies.clone();
    };
    let iterations = closure_solve.iterations;
    let corrections = &mut closure_solve.corrections;
    let initial_group_poses = groups
        .iter()
        .enumerate()
        .map(|(group_index, group)| {
            if use_consensus_group_poses {
                group_poses[group_index]
            } else {
                locked_homographies.get(&images[group.anchor].id).copied()
            }
        })
        .collect::<Option<Vec<_>>>();
    let Some(initial_group_poses) = initial_group_poses else {
        return locked_homographies.clone();
    };
    let Some((maximum_corner_correction, clamped_stations)) = clamp_station_translation_corrections(
        images,
        &groups,
        &group_edges,
        &initial_group_poses,
        corrections,
    ) else {
        return locked_homographies.clone();
    };
    if clamped_stations > 0 {
        degradation::record_run_degradation(
            degradation::CLOSURE_CORRECTION_CLAMPED,
            serde_json::json!({
                "clamped_stations": clamped_stations,
                "max_corner_correction_px": maximum_corner_correction,
            }),
        );
    }

    let mut solved = HashMap::new();
    for (group_index, group) in groups.iter().enumerate() {
        let original_group_pose = initial_group_poses[group_index];
        let correction = corrections[group_index];
        let group_pose = Matrix3::new(
            1.0,
            0.0,
            correction[0],
            0.0,
            1.0,
            correction[1],
            0.0,
            0.0,
            1.0,
        ) * original_group_pose;
        for &image_index in &group.members {
            let replacement = group_pose * group.local_to_anchor[&image_index];
            if replacement.try_inverse().is_none()
                || !homography_preserves_focus_orientation(
                    &replacement,
                    images[image_index].dimensions(),
                )
            {
                println!(
                    "  - Focus capture-group translation solve produced an unstable pose for '{}'",
                    images[image_index].filename
                );
                return locked_homographies.clone();
            }
            solved.insert(images[image_index].id, replacement);
        }
    }
    carry_unstationed_poses(&mut solved);
    println!(
        "  - Focus capture-group translation solve: {} groups, {} candidate group pairs, {} consensus pairs, {} accepted constraints; {iterations} iteration(s), maximum corner correction {:.2}px, {} clamped station(s)",
        groups.len(),
        group_edges.len(),
        consensus_pair_count,
        constraints.len(),
        maximum_corner_correction,
        clamped_stations,
    );
    solved
}
