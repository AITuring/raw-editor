//! Source_RAW layer-pair diagnostics.
//!
//! Independent observations, consensus clusters, correlation-guided descriptor
//! matches and matcher attrition records for each layer pair.

use crate::panorama_utils::processing;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stitching::Projection;
use crate::panorama_utils::stitching::project_point;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;

use super::{
    AlignmentMode, BlendMode, FOCUS_MODEL_MIN_INLIERS, FULL_RES_RANSAC_INLIER_THRESHOLD,
    FocusCaptureGroup, ImageInfo, Match, MatchInfo, SCALABLE_MATCH_RATIO_THRESHOLD,
    SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION, STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    STATION_RELATION_MIN_SPATIAL_SUPPORT, STATION_RELATION_SCALE_RATIO_MAX,
    STATION_RELATION_SCALE_RATIO_MIN, SourceRawCorrelationPyramid,
    VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE, VirtualTileGlobalProposal, absolute_source_path,
    canonical_match_direction, focus_transform_disagreement_px, homography_scale_ratio,
    match_image_pair_with_seed, matrix3_row_major, median_symmetric_error, median_value,
    source_raw_correlation_proposals, station_relation_spatial_support, transformed_point,
    virtual_tile_descriptor_distance, virtual_tile_hypothesis_matrix_key,
};

#[derive(Clone)]
pub(super) struct SourceRawLayerDiagnosticModel {
    pub(super) left_source: usize,
    pub(super) right_source: usize,
    pub(super) record_index: usize,
    pub(super) transform: Matrix3<f64>,
    pub(super) spatial_support: f64,
}

/// Select the largest set of cluster observations that shares no Source_RAW
/// on either station. Thus many patches, alternate fits, or matches from one
/// particularly sharp RAW can contribute only one unit of independent support.
pub(super) fn source_raw_independent_observations(
    models: &[SourceRawLayerDiagnosticModel],
    cluster: &[usize],
) -> Vec<usize> {
    fn augment(
        left: usize,
        adjacency: &BTreeMap<usize, Vec<(usize, usize)>>,
        seen_right: &mut HashSet<usize>,
        right_matches: &mut BTreeMap<usize, (usize, usize)>,
    ) -> bool {
        for &(right, model_index) in adjacency.get(&left).into_iter().flatten() {
            if !seen_right.insert(right) {
                continue;
            }
            let previous = right_matches.get(&right).copied();
            if previous.is_none_or(|(previous_left, _)| {
                augment(previous_left, adjacency, seen_right, right_matches)
            }) {
                right_matches.insert(right, (left, model_index));
                return true;
            }
        }
        false
    }

    let mut adjacency = BTreeMap::<usize, Vec<(usize, usize)>>::new();
    for &model_index in cluster {
        let model = &models[model_index];
        adjacency
            .entry(model.left_source)
            .or_default()
            .push((model.right_source, model_index));
    }
    for edges in adjacency.values_mut() {
        edges.sort_unstable();
    }
    let mut right_matches = BTreeMap::<usize, (usize, usize)>::new();
    for &left in adjacency.keys() {
        augment(left, &adjacency, &mut HashSet::new(), &mut right_matches);
    }
    let mut selected = right_matches
        .into_values()
        .map(|(_, model_index)| model_index)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
}

pub(super) fn source_raw_consensus_clusters(
    models: &[SourceRawLayerDiagnosticModel],
    layer_records: &mut [stack_report::SourceRawLayerModelRecord],
    dimensions: (u32, u32),
) -> Vec<stack_report::SourceRawConsensusClusterRecord> {
    let mut remaining = (0..models.len()).collect::<BTreeSet<_>>();
    let mut clusters = Vec::new();
    while !remaining.is_empty() {
        let mut best: Option<(usize, usize, f64, [u64; 9], usize, Vec<usize>, Vec<usize>)> = None;
        for &representative in &remaining {
            let members = remaining
                .iter()
                .copied()
                .filter(|&other| {
                    focus_transform_disagreement_px(
                        &models[representative].transform,
                        &models[other].transform,
                        dimensions,
                    )
                    .is_some_and(|error| error <= STATION_RELATION_MAX_MEDIAN_ERROR_PX)
                })
                .collect::<Vec<_>>();
            let independent = source_raw_independent_observations(models, &members);
            let mut disagreements = members
                .iter()
                .filter_map(|&other| {
                    focus_transform_disagreement_px(
                        &models[representative].transform,
                        &models[other].transform,
                        dimensions,
                    )
                })
                .collect::<Vec<_>>();
            let median_disagreement = median_value(&mut disagreements).unwrap_or(f64::INFINITY);
            let matrix_key = virtual_tile_hypothesis_matrix_key(&models[representative].transform);
            let rank = (
                independent.len(),
                members.len(),
                median_disagreement,
                matrix_key,
                representative,
                members,
                independent,
            );
            let replace = best.as_ref().is_none_or(|current| {
                rank.0 > current.0
                    || (rank.0 == current.0 && rank.1 > current.1)
                    || (rank.0 == current.0
                        && rank.1 == current.1
                        && rank.2.total_cmp(&current.2).is_lt())
                    || (rank.0 == current.0
                        && rank.1 == current.1
                        && rank.2.total_cmp(&current.2).is_eq()
                        && rank.3 < current.3)
            });
            if replace {
                best = Some(rank);
            }
        }
        let Some((support, _, median_disagreement, _, representative, members, independent)) = best
        else {
            break;
        };
        if support < 2 {
            break;
        }
        let cluster_index = clusters.len();
        for &model_index in &members {
            layer_records[models[model_index].record_index].consensus_cluster = Some(cluster_index);
            remaining.remove(&model_index);
        }
        let representative_model = &models[representative];
        let scale_ratio =
            homography_scale_ratio(&representative_model.transform).unwrap_or(f64::INFINITY);
        let spatial_support_potential = independent
            .iter()
            .map(|&index| models[index].spatial_support)
            .fold(0.0, f64::max);
        let supporting_layer_pairs = independent
            .iter()
            .map(|&index| {
                let record = &layer_records[models[index].record_index];
                [record.left_source.clone(), record.right_source.clone()]
            })
            .collect();
        clusters.push(stack_report::SourceRawConsensusClusterRecord {
            cluster: cluster_index,
            observation_count: members.len(),
            independent_support: support,
            supporting_layer_pairs,
            median_disagreement_px: median_disagreement,
            scale_ratio,
            spatial_support_potential,
            scale_compliant: (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
                .contains(&scale_ratio),
            spatially_broad: spatial_support_potential >= STATION_RELATION_MIN_SPATIAL_SUPPORT,
            anchor_transform: matrix3_row_major(&representative_model.transform),
        });
    }
    clusters
}

/// Convert one correlation peak into deterministic, one-to-one RAW descriptor
/// correspondences. Geometry only narrows each descriptor search window; the
/// unchanged descriptor distance and ratio limits prevent a peak from creating
/// support by itself.
pub(super) fn source_raw_correlation_guided_matches(
    source: &ImageInfo,
    target: &ImageInfo,
    proposal: &VirtualTileGlobalProposal,
) -> Vec<Match> {
    let search_radius = f64::from(proposal.left_factor.max(proposal.right_factor)) * 2.0;
    let mut candidates = Vec::<(u32, u64, usize, usize)>::new();
    for (source_index, source_feature) in source.features.iter().enumerate() {
        let Some(predicted) = transformed_point(
            &proposal.left_to_right,
            Point2::new(
                f64::from(source_feature.keypoint.x),
                f64::from(source_feature.keypoint.y),
            ),
        ) else {
            continue;
        };
        let mut nearby = target
            .features
            .iter()
            .enumerate()
            .filter_map(|(target_index, target_feature)| {
                let dx = f64::from(target_feature.keypoint.x) - predicted.x;
                let dy = f64::from(target_feature.keypoint.y) - predicted.y;
                let residual_squared = dx * dx + dy * dy;
                (residual_squared <= search_radius * search_radius).then(|| {
                    (
                        virtual_tile_descriptor_distance(
                            &source_feature.descriptor,
                            &target_feature.descriptor,
                        ),
                        residual_squared,
                        target_index,
                    )
                })
            })
            .collect::<Vec<_>>();
        nearby.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        let Some(&(best_distance, residual_squared, target_index)) = nearby.first() else {
            continue;
        };
        let second_distance = nearby.get(1).map_or(u32::MAX, |entry| entry.0);
        if best_distance > VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE
            || f64::from(best_distance)
                >= f64::from(second_distance) * SCALABLE_MATCH_RATIO_THRESHOLD as f64
        {
            continue;
        }
        candidates.push((
            best_distance,
            residual_squared.to_bits(),
            source_index,
            target_index,
        ));
    }
    candidates.sort_unstable();
    let mut used_source = HashSet::new();
    let mut used_target = HashSet::new();
    candidates
        .into_iter()
        .filter_map(|(_, _, index1, index2)| {
            (used_source.insert(index1) && used_target.insert(index2))
                .then_some(Match { index1, index2 })
        })
        .collect()
}

pub(super) fn source_raw_seed_points(
    source: &ImageInfo,
    target: &ImageInfo,
    matches: &[Match],
) -> Vec<(Point2<f64>, Point2<f64>)> {
    matches
        .iter()
        .filter_map(|matched| {
            let source_point = source.features.get(matched.index1)?.keypoint;
            let target_point = target.features.get(matched.index2)?.keypoint;
            project_point(
                source,
                f64::from(source_point.x) * source.scale_factor,
                f64::from(source_point.y) * source.scale_factor,
                Projection::Planar,
            )
            .zip(project_point(
                target,
                f64::from(target_point.x) * target.scale_factor,
                f64::from(target_point.y) * target.scale_factor,
                Projection::Planar,
            ))
        })
        .collect()
}

pub(super) fn source_raw_normalize_validated_match(
    relation: &MatchInfo,
    match_direction_reversed: bool,
) -> Option<(Matrix3<f64>, Vec<(Point2<f64>, Point2<f64>)>, usize)> {
    if match_direction_reversed {
        relation.homography.try_inverse().map(|transform| {
            (
                transform,
                relation
                    .points
                    .iter()
                    .map(|&(source, target)| (target, source))
                    .collect(),
                relation.inliers,
            )
        })
    } else {
        Some((
            relation.homography,
            relation.points.clone(),
            relation.inliers,
        ))
    }
}

#[derive(Default)]
pub(super) struct SourceRawMatcherAttrition {
    pub(super) left_descriptor_count: usize,
    pub(super) right_descriptor_count: usize,
    pub(super) match_direction_reversed: bool,
    pub(super) descriptor_retrieval_count: usize,
    pub(super) ratio_filter_count: usize,
    pub(super) mutual_filter_count: usize,
    pub(super) ransac_model_fitted: bool,
    pub(super) ransac_inliers: usize,
    pub(super) ransac_anchor_transform: Option<[f64; 9]>,
    pub(super) canonical_conversion_succeeded: bool,
    pub(super) proposal_candidate_count: usize,
    pub(super) proposal_score: Option<f64>,
    pub(super) proposal_match_count: usize,
    pub(super) proposal_match: Option<MatchInfo>,
    pub(super) correlation_proposals: Vec<stack_report::SourceRawCorrelationProposalRecord>,
}

/// Independent per-RAW spatial proposal recall. Descriptor correspondences are
/// grouped only by their observed displacement on this RAW pair; no fused tile,
/// station prior, filename, or proposal from another layer is visible here.
/// Returned subsets are merely seeds for the unchanged stable RANSAC and full
/// image validation in `match_image_pair_with_seed`.
pub(super) fn source_raw_descriptor_proposals(
    matches: &[Match],
    points: &[(Point2<f64>, Point2<f64>)],
    dimensions: (u32, u32),
) -> Vec<(f64, Vec<Match>)> {
    if matches.len() != points.len() || matches.len() < SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION {
        return Vec::new();
    }
    let bin_size = f64::from(dimensions.0.max(dimensions.1).max(1)) * 0.01;
    let mut bins = BTreeMap::<(i32, i32), Vec<usize>>::new();
    for (index, (source, target)) in points.iter().enumerate() {
        let displacement = target - source;
        if displacement.x.is_finite() && displacement.y.is_finite() {
            bins.entry((
                (displacement.x / bin_size).floor() as i32,
                (displacement.y / bin_size).floor() as i32,
            ))
            .or_default()
            .push(index);
        }
    }
    let mut candidates = bins
        .keys()
        .map(|&(bin_x, bin_y)| {
            let mut indices = Vec::new();
            for y in bin_y - 1..=bin_y + 1 {
                for x in bin_x - 1..=bin_x + 1 {
                    indices.extend(bins.get(&(x, y)).into_iter().flatten().copied());
                }
            }
            indices.sort_unstable();
            indices.dedup();
            ((bin_x, bin_y), indices)
        })
        .filter(|(_, indices)| indices.len() >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION)
        .collect::<Vec<_>>();
    candidates.sort_by(|(left_bin, left), (right_bin, right)| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| left_bin.cmp(right_bin))
    });
    let mut seen = HashSet::new();
    candidates
        .into_iter()
        .filter_map(|(_, indices)| {
            let proposal = indices
                .into_iter()
                .map(|index| matches[index])
                .collect::<Vec<_>>();
            let provenance = proposal
                .iter()
                .map(|matched| (matched.index1, matched.index2))
                .collect::<Vec<_>>();
            seen.insert(provenance)
                .then_some((proposal.len() as f64 / matches.len() as f64, proposal))
        })
        .take(8)
        .collect()
}

pub(super) fn source_raw_record_correlation_oracle(
    source: &ImageInfo,
    target: &ImageInfo,
    seed_threshold: f64,
    attrition: &mut SourceRawMatcherAttrition,
) {
    let proposals = SourceRawCorrelationPyramid::from_image(source)
        .zip(SourceRawCorrelationPyramid::from_image(target))
        .map(|(source_pyramid, target_pyramid)| {
            source_raw_correlation_proposals(&source_pyramid, &target_pyramid)
        })
        .unwrap_or_default();
    attrition.proposal_candidate_count = proposals.len();
    for (rank, correlation) in proposals.into_iter().enumerate() {
        let proposal = correlation.proposal;
        let guided = source_raw_correlation_guided_matches(source, target, &proposal);
        let seed_points = source_raw_seed_points(source, target, &guided);
        let stable_inliers = (seed_points.len() >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION)
            .then(|| {
                processing::find_homography_ransac_points_stable_with_min_inliers(
                    &seed_points,
                    seed_threshold,
                    SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION,
                )
            })
            .flatten()
            .map_or(0, |(_, inliers)| inliers.len());
        let validation = (stable_inliers >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION)
            .then(|| {
                match_image_pair_with_seed(
                    source,
                    target,
                    Projection::Planar,
                    BlendMode::FocusStack,
                    AlignmentMode::Auto,
                    true,
                    false,
                    false,
                    Some(guided.clone()),
                )
            })
            .flatten();
        let authoritative = validation.as_ref().filter(|relation| {
            !relation.sequence_bridge
                && !relation.coarse_bridge
                && relation.points.len() >= FOCUS_MODEL_MIN_INLIERS
        });
        let validation_stage = if guided.len() < SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION {
            "guided_feature_matching"
        } else if stable_inliers < SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION {
            "stable_ransac"
        } else if validation.is_none() {
            "original_full_resolution_validation"
        } else if authoritative.is_none() {
            "original_full_resolution_non_authoritative"
        } else {
            "accepted"
        };
        attrition
            .correlation_proposals
            .push(stack_report::SourceRawCorrelationProposalRecord {
                rank,
                score: proposal.score,
                second_peak_margin: correlation.second_peak_margin,
                overlap_samples: proposal.overlap_samples,
                left_factor: proposal.left_factor,
                right_factor: proposal.right_factor,
                translation_x_px: proposal.left_to_right[(0, 2)],
                translation_y_px: proposal.left_to_right[(1, 2)],
                scale: proposal.scale,
                rotation_degrees: proposal.rotation_degrees,
                guided_match_count: guided.len(),
                stable_ransac_inliers: stable_inliers,
                full_resolution_validated: authoritative.is_some(),
                validation_stage: validation_stage.to_string(),
                match_source: absolute_source_path(&source.filename),
                match_target: absolute_source_path(&target.filename),
                source_to_target_transform: matrix3_row_major(&proposal.left_to_right),
            });
        if attrition.proposal_match.is_none()
            && let Some(validated) = authoritative.cloned()
        {
            attrition.proposal_score = Some(proposal.score);
            attrition.proposal_match_count = guided.len();
            attrition.proposal_match = Some(validated);
        }
    }
}

/// Replays only the immutable descriptor and provisional geometry stages used
/// by the stable RAW-pair matcher. The final validation result still comes from
/// `matches`, so this oracle cannot create or promote a relation.
pub(super) fn source_raw_matcher_attrition(
    images: &[ImageInfo],
    left_source: usize,
    right_source: usize,
    left_group: &FocusCaptureGroup,
    right_group: &FocusCaptureGroup,
) -> SourceRawMatcherAttrition {
    let left_image = &images[left_source];
    let right_image = &images[right_source];
    let (match_source, match_target, reversed) =
        canonical_match_direction(images, left_source, right_source);
    let source = &images[match_source];
    let target = &images[match_target];
    let descriptor = processing::match_features_with_ratio_attrition(
        &source.features,
        &target.features,
        SCALABLE_MATCH_RATIO_THRESHOLD,
    );
    let mut attrition = SourceRawMatcherAttrition {
        left_descriptor_count: left_image.features.len(),
        right_descriptor_count: right_image.features.len(),
        match_direction_reversed: reversed,
        descriptor_retrieval_count: descriptor.retrieval_count,
        ratio_filter_count: descriptor.ratio_filter_count,
        mutual_filter_count: descriptor.mutual_matches.len(),
        ..Default::default()
    };
    let source_points = source
        .features
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let target_points = target
        .features
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let projected = descriptor
        .mutual_matches
        .iter()
        .filter_map(|matched| {
            let source_point = source_points.get(matched.index1)?;
            let target_point = target_points.get(matched.index2)?;
            project_point(
                source,
                source_point.x as f64 * source.scale_factor,
                source_point.y as f64 * source.scale_factor,
                Projection::Planar,
            )
            .zip(project_point(
                target,
                target_point.x as f64 * target.scale_factor,
                target_point.y as f64 * target.scale_factor,
                Projection::Planar,
            ))
        })
        .collect::<Vec<_>>();
    let seed_threshold =
        FULL_RES_RANSAC_INLIER_THRESHOLD.max(source.scale_factor.max(target.scale_factor) * 3.5);
    let stable = (projected.len() >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION)
        .then(|| {
            processing::find_homography_ransac_points_stable_with_min_inliers(
                &projected,
                seed_threshold,
                SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION,
            )
        })
        .flatten();
    source_raw_record_correlation_oracle(source, target, seed_threshold, &mut attrition);
    let Some((match_transform, inliers)) = stable else {
        return attrition;
    };
    attrition.ransac_model_fitted = true;
    attrition.ransac_inliers = inliers.len();
    let left_to_right = if reversed {
        match_transform.try_inverse()
    } else {
        Some(match_transform)
    };
    let anchor_transform = left_to_right
        .zip(left_group.local_to_anchor.get(&left_source))
        .zip(right_group.local_to_anchor.get(&right_source))
        .and_then(|((left_to_right, left_local), right_local)| {
            left_local
                .try_inverse()
                .map(|left_inverse| right_local * left_to_right * left_inverse)
        });
    if let Some(anchor_transform) = anchor_transform {
        attrition.canonical_conversion_succeeded = true;
        attrition.ransac_anchor_transform = Some(matrix3_row_major(&anchor_transform));
    }
    attrition
}

/// Experiment E's read-only layer oracle. It consumes each Source_RAW's own
/// luminance/gradient pyramid, immutable descriptors, already fitted RAW-pair
/// models, and each station's verified local-to-anchor maps. No result is
/// inserted into the Virtual_Tile graph or prior repair.
pub(super) fn source_raw_layer_pair_diagnostics(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    groups: &[FocusCaptureGroup],
) -> Vec<stack_report::SourceRawPairDiagnosticRecord> {
    let mut diagnostics = Vec::new();
    for left_station in 0..groups.len() {
        for right_station in left_station + 1..groups.len() {
            let left_group = &groups[left_station];
            let right_group = &groups[right_station];
            let mut layer_records = Vec::new();
            let mut models = Vec::new();
            for &left_source in &left_group.members {
                for &right_source in &right_group.members {
                    let record_index = layer_records.len();
                    let attrition = source_raw_matcher_attrition(
                        images,
                        left_source,
                        right_source,
                        left_group,
                        right_group,
                    );
                    let key = (left_source.min(right_source), left_source.max(right_source));
                    let fitted = matches.get(&key).filter(|relation| {
                        !relation.sequence_bridge
                            && !relation.coarse_bridge
                            && relation.points.len() >= FOCUS_MODEL_MIN_INLIERS
                    });
                    let full_resolution_validated =
                        fitted.is_some() || attrition.proposal_match.is_some();
                    let failure_stage = if attrition.left_descriptor_count == 0
                        || attrition.right_descriptor_count == 0
                    {
                        "descriptor_extraction"
                    } else if attrition.descriptor_retrieval_count == 0 {
                        "descriptor_retrieval"
                    } else if fitted.is_some() {
                        "accepted"
                    } else if attrition.proposal_match.is_some() {
                        "accepted_by_raw_correlation_oracle"
                    } else if !attrition.correlation_proposals.is_empty() {
                        "raw_correlation_oracle_no_full_validation"
                    } else if attrition.mutual_filter_count
                        < SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION
                    {
                        "ratio_mutual_filtering"
                    } else if !attrition.ransac_model_fitted {
                        "model_ransac"
                    } else if attrition.ransac_model_fitted
                        && !attrition.canonical_conversion_succeeded
                    {
                        "canonical_local_to_anchor"
                    } else if !full_resolution_validated {
                        "full_resolution_validation"
                    } else {
                        "accepted"
                    };
                    let mut record = stack_report::SourceRawLayerModelRecord {
                        left_source: absolute_source_path(&images[left_source].filename),
                        right_source: absolute_source_path(&images[right_source].filename),
                        left_descriptor_count: attrition.left_descriptor_count,
                        right_descriptor_count: attrition.right_descriptor_count,
                        match_direction_reversed: attrition.match_direction_reversed,
                        descriptor_retrieval_count: attrition.descriptor_retrieval_count,
                        ratio_filter_count: attrition.ratio_filter_count,
                        mutual_filter_count: attrition.mutual_filter_count,
                        ransac_model_fitted: attrition.ransac_model_fitted,
                        ransac_inliers: attrition.ransac_inliers,
                        ransac_anchor_transform: attrition.ransac_anchor_transform,
                        canonical_conversion_succeeded: attrition.canonical_conversion_succeeded,
                        full_resolution_validated,
                        failure_stage: failure_stage.to_string(),
                        model_provenance: if fitted.is_some() {
                            "stable_matcher".to_string()
                        } else if attrition.proposal_match.is_some() {
                            "per_raw_coarse_luminance_gradient_correlation".to_string()
                        } else {
                            "stable_matcher".to_string()
                        },
                        proposal_seeded: attrition.proposal_match.is_some(),
                        proposal_score: attrition.proposal_score,
                        proposal_candidate_count: attrition.proposal_candidate_count,
                        proposal_match_count: attrition.proposal_match_count,
                        proposal_ransac_inliers: attrition
                            .correlation_proposals
                            .iter()
                            .filter(|proposal| proposal.full_resolution_validated)
                            .map(|proposal| proposal.stable_ransac_inliers)
                            .next()
                            .unwrap_or_default(),
                        correlation_proposals: attrition.correlation_proposals.clone(),
                        ..Default::default()
                    };
                    let normalized = if let Some(relation) = fitted {
                        if left_source == key.0 {
                            Some((
                                relation.homography,
                                relation.points.clone(),
                                relation.inliers,
                            ))
                        } else {
                            relation.homography.try_inverse().map(|transform| {
                                (
                                    transform,
                                    relation
                                        .points
                                        .iter()
                                        .map(|&(source, target)| (target, source))
                                        .collect(),
                                    relation.inliers,
                                )
                            })
                        }
                    } else if let Some(relation) = attrition.proposal_match.as_ref() {
                        source_raw_normalize_validated_match(
                            relation,
                            attrition.match_direction_reversed,
                        )
                    } else {
                        None
                    };
                    if let Some((left_to_right, raw_points, relation_inliers)) = normalized {
                        let left_local = left_group.local_to_anchor.get(&left_source);
                        let right_local = right_group.local_to_anchor.get(&right_source);
                        if let Some((left_local, left_local_inverse, right_local)) = left_local
                            .and_then(|transform| {
                                transform.try_inverse().map(|inverse| (transform, inverse))
                            })
                            .zip(right_local)
                            .map(|((left, inverse), right)| (left, inverse, right))
                        {
                            let transform = right_local * left_to_right * left_local_inverse;
                            let anchor_points = raw_points
                                .iter()
                                .filter_map(|&(left, right)| {
                                    transformed_point(left_local, left)
                                        .zip(transformed_point(right_local, right))
                                })
                                .collect::<Vec<_>>();
                            let median_error_px =
                                median_symmetric_error(&transform, &anchor_points);
                            let scale_ratio = homography_scale_ratio(&transform);
                            if anchor_points.len() >= FOCUS_MODEL_MIN_INLIERS
                                && median_error_px.is_finite()
                                && scale_ratio.is_some_and(f64::is_finite)
                            {
                                let spatial_support = station_relation_spatial_support(
                                    &anchor_points,
                                    &transform,
                                    images[left_group.anchor].dimensions(),
                                    images[right_group.anchor].dimensions(),
                                );
                                record.model_fitted = true;
                                record.canonical_conversion_succeeded = true;
                                if record.ransac_anchor_transform.is_none() {
                                    record.ransac_anchor_transform =
                                        Some(matrix3_row_major(&transform));
                                }
                                record.inliers = relation_inliers.max(anchor_points.len());
                                record.median_error_px = median_error_px;
                                record.scale_ratio = scale_ratio.unwrap_or_default();
                                record.spatial_support = spatial_support;
                                record.anchor_transform = matrix3_row_major(&transform);
                                models.push(SourceRawLayerDiagnosticModel {
                                    left_source,
                                    right_source,
                                    record_index,
                                    transform,
                                    spatial_support,
                                });
                            }
                        }
                    }
                    layer_records.push(record);
                }
            }
            let dimensions = images[left_group.anchor].dimensions();
            let consensus_clusters =
                source_raw_consensus_clusters(&models, &mut layer_records, dimensions);
            diagnostics.push(stack_report::SourceRawPairDiagnosticRecord {
                left: left_station,
                right: right_station,
                layer_models: layer_records,
                consensus_clusters,
            });
        }
    }
    diagnostics
}
