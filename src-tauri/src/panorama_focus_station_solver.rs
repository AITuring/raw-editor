//! Station evidence pose solving, closure refinement, and translation fallback.
//!
//! The parent `panorama_focus_pose_solver` module re-exports these symbols so
//! existing callers and test accessors retain their historical paths.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_station_poses_for_evidence_with_report(
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
