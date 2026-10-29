//! Direct Source_RAW pair observations and their consensus.
//!
//! One direct observation per layer pair, clustered into consensus models for
//! the pair diagnostics.

use crate::panorama_utils::stack_pipeline::report as stack_report;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::BTreeSet;

use super::{
    FocusCaptureGroup, ImageInfo, STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    SourceRawCorrelationPyramid, SourceRawDirectPyramid, SourceRawLayerDiagnosticModel,
    StationOverlapMeasure, VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX, absolute_source_path,
    focus_overlap_quality_measurements, focus_transform_disagreement_px,
    homography_preserves_focus_orientation, homography_scale_ratio, matrix3_row_major,
    measure_station_relation_candidate, median_symmetric_error, median_value,
    source_raw_correlation_proposals, source_raw_direct_closure, source_raw_direct_raw_transform,
    source_raw_direct_select_basin, source_raw_independent_observations,
    station_relation_rejection_reasons, station_relation_spatial_support, transformed_point,
};

#[derive(Clone)]
pub(super) struct SourceRawDirectObservation {
    pub(super) record: stack_report::SourceRawDirectObservationRecord,
    pub(super) left_source: usize,
    pub(super) right_source: usize,
    pub(super) transform: Matrix3<f64>,
    pub(super) reverse_transform: Matrix3<f64>,
    pub(super) anchor_points: Vec<(Point2<f64>, Point2<f64>)>,
}

pub(super) fn source_raw_direct_finite_diagnostic(value: f64) -> f64 {
    if value.is_finite() { value } else { f64::MAX }
}

pub(super) fn source_raw_direct_observation(
    images: &[ImageInfo],
    left_source: usize,
    right_source: usize,
    left_group: &FocusCaptureGroup,
    right_group: &FocusCaptureGroup,
) -> Option<SourceRawDirectObservation> {
    let left = &images[left_source];
    let right = &images[right_source];
    let left_pyramid = SourceRawDirectPyramid::from_image(left)?;
    let right_pyramid = SourceRawDirectPyramid::from_image(right)?;
    let left_coarse = SourceRawCorrelationPyramid::from_image(left)?;
    let right_coarse = SourceRawCorrelationPyramid::from_image(right)?;
    let forward_seeds = source_raw_correlation_proposals(&left_coarse, &right_coarse);
    let reverse_seeds = source_raw_correlation_proposals(&right_coarse, &left_coarse);
    if forward_seeds.is_empty() || reverse_seeds.is_empty() {
        return None;
    }
    let selection = source_raw_direct_select_basin(
        left,
        right,
        &left_pyramid,
        &right_pyramid,
        &forward_seeds,
        &reverse_seeds,
    )?;
    let mut fold_records = Vec::new();
    let mut fold_transforms = Vec::new();
    let mut all_points = Vec::new();
    let mut reverse_transforms = Vec::new();
    for fold in 0..2 {
        let forward = &selection.forward[fold];
        let reverse = &selection.reverse[fold];
        let raw_forward = source_raw_direct_raw_transform(
            &forward.transform,
            left.scale_factor,
            right.scale_factor,
        );
        let raw_reverse = source_raw_direct_raw_transform(
            &reverse.transform,
            right.scale_factor,
            left.scale_factor,
        );
        let closure = selection.closures[fold];
        let points = selection.points[fold].clone();
        fold_records.push(stack_report::SourceRawDirectFoldRecord {
            fold,
            seed_rank: forward.seed_rank,
            reverse_seed_rank: reverse.seed_rank,
            seed_pair_coarse_closure_px: selection.coarse_closure,
            seed_provenance: "reciprocal_coarse_multi_peak_fold_consistent".to_string(),
            pyramid_factors: forward.factors.clone(),
            iterations: forward.iterations + reverse.iterations,
            fit_cell_count: forward.fit.cells.min(reverse.fit.cells),
            holdout_cell_count: forward.holdout.cells.min(reverse.holdout.cells),
            fit_loss: forward.fit.loss.max(reverse.fit.loss),
            holdout_loss: forward.holdout.loss.max(reverse.holdout.loss),
            gain: forward.holdout.gain,
            bias: forward.holdout.bias,
            model: format!("{}+independent_reverse_{}", forward.model, reverse.model),
            raw_transform: matrix3_row_major(&raw_forward),
            reverse_raw_transform: matrix3_row_major(&raw_reverse),
            bidirectional_closure_px: closure,
            points: points.clone(),
        });
        all_points.extend(points);
        fold_transforms.push(raw_forward);
        reverse_transforms.push(raw_reverse);
    }
    let fold_disagreement = selection.forward_fold_disagreement;
    let representative = if fold_records[1].holdout_loss < fold_records[0].holdout_loss {
        1
    } else {
        0
    };
    let raw_transform = fold_transforms[representative];
    let reverse_transform = reverse_transforms[representative];
    let closure = fold_records
        .iter()
        .map(|record| record.bidirectional_closure_px)
        .max_by(f64::total_cmp)?;
    all_points.sort_by(|left, right| {
        left.cell
            .cmp(&right.cell)
            .then_with(|| left.source[1].total_cmp(&right.source[1]))
            .then_with(|| left.source[0].total_cmp(&right.source[0]))
    });
    all_points.dedup_by_key(|point| point.cell);
    let raw_points = all_points
        .iter()
        .map(|point| {
            (
                Point2::new(point.source[0], point.source[1]),
                Point2::new(point.target[0], point.target[1]),
            )
        })
        .collect::<Vec<_>>();
    let left_local = left_group.local_to_anchor.get(&left_source)?;
    let right_local = right_group.local_to_anchor.get(&right_source)?;
    let anchor_transform = right_local * raw_transform * left_local.try_inverse()?;
    let anchor_reverse = left_local * reverse_transform * right_local.try_inverse()?;
    let anchor_points = raw_points
        .iter()
        .filter_map(|&(source, target)| {
            transformed_point(left_local, source).zip(transformed_point(right_local, target))
        })
        .collect::<Vec<_>>();
    let median_error_px = median_symmetric_error(&anchor_transform, &anchor_points);
    let scale_ratio = homography_scale_ratio(&anchor_transform).unwrap_or(f64::INFINITY);
    let spatial_support = station_relation_spatial_support(
        &anchor_points,
        &anchor_transform,
        images[left_group.anchor].dimensions(),
        images[right_group.anchor].dimensions(),
    );
    let preserves_convex_orientation = homography_preserves_focus_orientation(
        &anchor_transform,
        images[left_group.anchor].dimensions(),
    );
    let dense = focus_overlap_quality_measurements(left, right, &raw_transform);
    let measured = measure_station_relation_candidate(
        raw_points.len(),
        &raw_transform,
        &raw_points,
        left.dimensions(),
        right.dimensions(),
        Some(StationOverlapMeasure::Rectangles),
        dense,
        false,
    );
    let mut rejection_reasons = measured
        .as_ref()
        .map(|measured| {
            station_relation_rejection_reasons(&measured.measurements)
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["direct_measurement_unavailable".to_string()]);
    if fold_disagreement > STATION_RELATION_MAX_MEDIAN_ERROR_PX {
        rejection_reasons.push("direct_fold_disagreement".to_string());
    }
    if closure > VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX {
        rejection_reasons.push("direct_bidirectional_closure".to_string());
    }
    rejection_reasons.sort();
    rejection_reasons.dedup();
    let record = stack_report::SourceRawDirectObservationRecord {
        left_source: absolute_source_path(&left.filename),
        right_source: absolute_source_path(&right.filename),
        folds: fold_records,
        model: if all_points.is_empty() {
            "none".to_string()
        } else {
            "similarity_affine_projective_heldout".to_string()
        },
        holdout_points: anchor_points.len(),
        median_error_px: source_raw_direct_finite_diagnostic(median_error_px),
        scale_ratio: source_raw_direct_finite_diagnostic(scale_ratio),
        spatial_support: source_raw_direct_finite_diagnostic(spatial_support),
        preserves_convex_orientation,
        fold_disagreement_px: source_raw_direct_finite_diagnostic(fold_disagreement),
        bidirectional_closure_px: source_raw_direct_finite_diagnostic(closure),
        anchor_transform: matrix3_row_major(&anchor_transform),
        rejection_reasons,
    };
    Some(SourceRawDirectObservation {
        record,
        left_source,
        right_source,
        transform: anchor_transform,
        reverse_transform: anchor_reverse,
        anchor_points,
    })
}

pub(super) fn source_raw_direct_consensus_clusters(
    images: &[ImageInfo],
    left_group: &FocusCaptureGroup,
    right_group: &FocusCaptureGroup,
    observations: &[SourceRawDirectObservation],
) -> Vec<stack_report::SourceRawDirectConsensusRecord> {
    let valid = observations
        .iter()
        .enumerate()
        .filter(|(_, observation)| observation.record.rejection_reasons.is_empty())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut clusters = Vec::new();
    let mut remaining = valid.into_iter().collect::<BTreeSet<_>>();
    while let Some(&representative) = remaining.first() {
        let members = remaining
            .iter()
            .copied()
            .filter(|&index| {
                focus_transform_disagreement_px(
                    &observations[representative].transform,
                    &observations[index].transform,
                    images[left_group.anchor].dimensions(),
                )
                .is_some_and(|error| error <= STATION_RELATION_MAX_MEDIAN_ERROR_PX)
            })
            .collect::<Vec<_>>();
        let support_models = members
            .iter()
            .enumerate()
            .map(|(record_index, &index)| SourceRawLayerDiagnosticModel {
                left_source: observations[index].left_source,
                right_source: observations[index].right_source,
                record_index,
                transform: observations[index].transform,
                spatial_support: observations[index].record.spatial_support,
            })
            .collect::<Vec<_>>();
        let selected_local = source_raw_independent_observations(
            &support_models,
            &(0..support_models.len()).collect::<Vec<_>>(),
        );
        let selected = selected_local
            .iter()
            .map(|&index| members[index])
            .collect::<Vec<_>>();
        for member in &members {
            remaining.remove(member);
        }
        if selected.len() < 2 {
            continue;
        }
        let representative = selected[0];
        let transform = observations[representative].transform;
        let reverse = observations[representative].reverse_transform;
        let points = selected
            .iter()
            .flat_map(|&index| observations[index].anchor_points.iter().copied())
            .collect::<Vec<_>>();
        let mut disagreements = selected
            .iter()
            .filter_map(|&index| {
                focus_transform_disagreement_px(
                    &transform,
                    &observations[index].transform,
                    images[left_group.anchor].dimensions(),
                )
            })
            .collect::<Vec<_>>();
        let canonical_disagreement_px = median_value(&mut disagreements).unwrap_or(f64::INFINITY);
        let closure =
            source_raw_direct_closure(&transform, &reverse, images[left_group.anchor].dimensions())
                .unwrap_or(f64::INFINITY);
        let dense = focus_overlap_quality_measurements(
            &images[left_group.anchor],
            &images[right_group.anchor],
            &transform,
        );
        let measured = measure_station_relation_candidate(
            points.len(),
            &transform,
            &points,
            images[left_group.anchor].dimensions(),
            images[right_group.anchor].dimensions(),
            Some(StationOverlapMeasure::Rectangles),
            dense,
            false,
        );
        let mut reasons = measured
            .as_ref()
            .map(|measured| {
                station_relation_rejection_reasons(&measured.measurements)
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec!["direct_station_measurement_unavailable".to_string()]);
        if canonical_disagreement_px > STATION_RELATION_MAX_MEDIAN_ERROR_PX {
            reasons.push("direct_canonical_disagreement".to_string());
        }
        if closure > VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX {
            reasons.push("direct_bidirectional_closure".to_string());
        }
        reasons.sort();
        reasons.dedup();
        let measurements = measured.map(|measured| measured.measurements);
        clusters.push(stack_report::SourceRawDirectConsensusRecord {
            cluster: clusters.len(),
            observation_count: members.len(),
            independent_support: selected.len(),
            supporting_layer_pairs: selected
                .iter()
                .map(|&index| {
                    [
                        observations[index].record.left_source.clone(),
                        observations[index].record.right_source.clone(),
                    ]
                })
                .collect(),
            canonical_disagreement_px: source_raw_direct_finite_diagnostic(
                canonical_disagreement_px,
            ),
            anchor_transform: matrix3_row_major(&transform),
            station_inliers: measurements.as_ref().map_or(0, |value| value.inliers),
            station_median_error_px: source_raw_direct_finite_diagnostic(
                measurements
                    .as_ref()
                    .map_or(f64::MAX, |value| value.median_error_px),
            ),
            station_scale_ratio: source_raw_direct_finite_diagnostic(
                measurements
                    .as_ref()
                    .map_or(f64::MAX, |value| value.scale_ratio),
            ),
            station_spatial_support: source_raw_direct_finite_diagnostic(
                measurements
                    .as_ref()
                    .map_or(0.0, |value| value.spatial_support),
            ),
            station_preserves_convex_orientation: measurements
                .as_ref()
                .is_some_and(|value| value.preserves_convex_orientation),
            station_bidirectional_closure_px: source_raw_direct_finite_diagnostic(closure),
            station_rejection_reasons: reasons,
        });
    }
    clusters
}

pub(super) fn source_raw_direct_pair_diagnostics(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
) -> Vec<stack_report::SourceRawDirectPairDiagnosticRecord> {
    let mut diagnostics = Vec::new();
    for left_station in 0..groups.len() {
        for right_station in left_station + 1..groups.len() {
            let start = std::time::Instant::now();
            let left_group = &groups[left_station];
            let right_group = &groups[right_station];
            let mut observations = Vec::new();
            let mut peak_working_set = 0usize;
            for &left_source in &left_group.members {
                for &right_source in &right_group.members {
                    if let (Some(left), Some(right)) = (
                        SourceRawDirectPyramid::from_image(&images[left_source]),
                        SourceRawDirectPyramid::from_image(&images[right_source]),
                    ) {
                        peak_working_set = peak_working_set
                            .max(left.working_set_bytes() + right.working_set_bytes());
                    }
                    if let Some(observation) = source_raw_direct_observation(
                        images,
                        left_source,
                        right_source,
                        left_group,
                        right_group,
                    ) {
                        observations.push(observation);
                    }
                }
            }
            observations
                .sort_by_key(|observation| (observation.left_source, observation.right_source));
            let consensus_clusters = source_raw_direct_consensus_clusters(
                images,
                left_group,
                right_group,
                &observations,
            );
            diagnostics.push(stack_report::SourceRawDirectPairDiagnosticRecord {
                left: left_station,
                right: right_station,
                observations: observations
                    .into_iter()
                    .map(|observation| observation.record)
                    .collect(),
                consensus_clusters,
                elapsed_ms: start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
                peak_working_set_bytes: peak_working_set as u64,
            });
        }
    }
    diagnostics
}
