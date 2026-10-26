//! Focus pose geometry, capture-group refinement, and topology estimation.
//!
//! The parent `panorama_focus_pose_solver` module re-exports these symbols so
//! existing callers and test accessors retain their historical paths.

use super::*;

pub(crate) fn focus_frame_median_sharpness(image: &ImageInfo) -> f64 {
    let analysis = &image.alignment_image;
    let (width, height) = (analysis.width(), analysis.height());
    let spacing = intra_station::analysis_probe_spacing(width.max(height));
    intra_station::median_valid_sharpness(width, height, |x, y| {
        crate::panorama_utils::mosaic::gray_analysis_sharpness(analysis, x, y, spacing)
    })
}

pub(crate) fn focus_capture_groups(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<(Vec<FocusCaptureGroup>, Vec<usize>)> {
    focus_capture_groups_with_grouping(images, matches, locked_homographies)
        .map(|(groups, group_for_image, _)| (groups, group_for_image))
}

/// [`focus_capture_groups`] together with everything the Stack_Report needs to
/// carry about the grouping decision (需求 1.4 / 1.6 / 1.7 / 1.10).
pub(crate) fn focus_capture_groups_with_grouping(
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

pub(crate) fn canonicalize_station_geometry_anchors(
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

pub(crate) fn order_focus_render_sources_by_capture_group(
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

pub(crate) fn solve_focus_group_translation_poses(
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
pub(crate) fn refine_focus_group_projective_tree_poses(
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

pub(crate) fn refine_focus_group_projective_tree_poses_with_report(
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

pub(crate) fn correct_focus_group_roll_from_long_edges(
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

pub(crate) fn focus_station_topology(
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
