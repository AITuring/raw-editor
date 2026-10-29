//! Basin selection of the direct Source_RAW refinement.
//!
//! Holdout patch localisation, closure checks and the comparison that selects
//! one reciprocal basin among the refined peaks.

use crate::panorama_utils::registration;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use image::GrayImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::BTreeSet;
use std::collections::HashSet;

use super::{
    ImageInfo, SOURCE_RAW_DIRECT_MAX_SEEDS, SOURCE_RAW_DIRECT_PATCH_GATES,
    SOURCE_RAW_DIRECT_PATCH_RADIUS, SOURCE_RAW_DIRECT_SEARCH_RADIUS,
    STATION_RELATION_MAX_MEDIAN_ERROR_PX, SourceRawCorrelationProposal, SourceRawDirectFold,
    SourceRawDirectPyramid, VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX, VirtualTileMatchPlane,
    focus_transform_disagreement_px, source_raw_direct_grid_centers,
    source_raw_direct_level_transform, source_raw_direct_raw_transform, source_raw_direct_refine,
    station_relation_spatial_support, transformed_point,
    virtual_tile_direct_patch_support_is_covered, virtual_tile_hypothesis_matrix_key,
};

pub(super) fn source_raw_direct_closure(
    forward: &Matrix3<f64>,
    reverse: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> Option<f64> {
    let samples = [
        Point2::new(0.0, 0.0),
        Point2::new(f64::from(dimensions.0), 0.0),
        Point2::new(0.0, f64::from(dimensions.1)),
        Point2::new(f64::from(dimensions.0), f64::from(dimensions.1)),
        Point2::new(f64::from(dimensions.0) * 0.5, f64::from(dimensions.1) * 0.5),
    ];
    samples
        .into_iter()
        .map(|point| {
            transformed_point(forward, point)
                .and_then(|mapped| transformed_point(reverse, mapped))
                .map(|returned| (returned - point).norm())
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max_by(f64::total_cmp)
}

pub(super) fn source_raw_direct_patch_fingerprint(
    image: &GrayImage,
    center: Point2<f64>,
) -> Option<u32> {
    let mut values = Vec::with_capacity(25);
    for dy in [-4.0, -2.0, 0.0, 2.0, 4.0] {
        for dx in [-4.0, -2.0, 0.0, 2.0, 4.0] {
            values.push(registration::sample_gray(
                image,
                center.x + dx,
                center.y + dy,
            )?);
        }
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    if variance < 2.0 {
        return None;
    }
    Some(
        values
            .into_iter()
            .enumerate()
            .fold(0u32, |bits, (index, value)| {
                bits | (u32::from(value >= mean) << index)
            }),
    )
}

pub(super) fn source_raw_direct_localize_holdout(
    left: &ImageInfo,
    right: &ImageInfo,
    left_pyramid: &SourceRawDirectPyramid,
    right_pyramid: &SourceRawDirectPyramid,
    forward: &Matrix3<f64>,
    reverse: &Matrix3<f64>,
    fold: usize,
) -> Vec<stack_report::SourceRawDirectPointRecord> {
    let Some(left_level) = left_pyramid.levels.last() else {
        return Vec::new();
    };
    let Some(right_level) = right_pyramid.levels.last() else {
        return Vec::new();
    };
    let level_forward = source_raw_direct_level_transform(
        forward,
        left_level.plane.factor,
        right_level.plane.factor,
    );
    let level_reverse = source_raw_direct_level_transform(
        reverse,
        right_level.plane.factor,
        left_level.plane.factor,
    );
    let left_level_to_native = VirtualTileMatchPlane::level_to_native(left_level.plane.factor);
    let right_level_to_native = VirtualTileMatchPlane::level_to_native(right_level.plane.factor);
    let mut fingerprints = HashSet::new();
    source_raw_direct_grid_centers(left_level.plane.gray.dimensions(), fold)
        .into_iter()
        .filter_map(|(cell, source)| {
            let fingerprint = source_raw_direct_patch_fingerprint(&left_level.plane.gray, source)?;
            if !virtual_tile_direct_patch_support_is_covered(
                &left_level.plane.coverage,
                &right_level.plane.coverage,
                &level_forward,
                source,
                SOURCE_RAW_DIRECT_PATCH_RADIUS,
                SOURCE_RAW_DIRECT_SEARCH_RADIUS,
            ) {
                return None;
            }
            let forward_probe = registration::probe_warped_patch(
                &left_level.plane.gray,
                &right_level.plane.gray,
                &level_forward,
                source,
                SOURCE_RAW_DIRECT_PATCH_RADIUS,
                SOURCE_RAW_DIRECT_SEARCH_RADIUS,
                SOURCE_RAW_DIRECT_PATCH_GATES,
            );
            let measured = forward_probe.matched?;
            if !virtual_tile_direct_patch_support_is_covered(
                &right_level.plane.coverage,
                &left_level.plane.coverage,
                &level_reverse,
                measured.target,
                SOURCE_RAW_DIRECT_PATCH_RADIUS,
                SOURCE_RAW_DIRECT_SEARCH_RADIUS,
            ) {
                return None;
            }
            let reverse_probe = registration::probe_warped_patch(
                &right_level.plane.gray,
                &left_level.plane.gray,
                &level_reverse,
                measured.target,
                SOURCE_RAW_DIRECT_PATCH_RADIUS,
                SOURCE_RAW_DIRECT_SEARCH_RADIUS,
                SOURCE_RAW_DIRECT_PATCH_GATES,
            );
            let returned = reverse_probe.matched?.target;
            let source_native = transformed_point(&left_level_to_native, source)?;
            let target_native = transformed_point(&right_level_to_native, measured.target)?;
            let returned_native = transformed_point(&left_level_to_native, returned)?;
            let source_raw = source_native * left.scale_factor;
            let target_raw = target_native * right.scale_factor;
            let returned_raw = returned_native * left.scale_factor;
            let round_trip = (returned_raw - source_raw).norm();
            (round_trip <= VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX).then_some((
                fingerprint,
                stack_report::SourceRawDirectPointRecord {
                    cell: [cell.0, cell.1],
                    source: [source_raw.x, source_raw.y],
                    target: [target_raw.x, target_raw.y],
                    reverse_source: [returned_raw.x, returned_raw.y],
                    correlation: measured.correlation,
                    peak_margin: forward_probe.peak_margin.unwrap_or_default(),
                    round_trip_error_px: round_trip,
                },
            ))
        })
        .filter_map(|(fingerprint, record)| fingerprints.insert(fingerprint).then_some(record))
        .collect()
}

#[derive(Clone)]
pub(super) struct SourceRawDirectBasinSelection {
    pub(super) forward: Vec<SourceRawDirectFold>,
    pub(super) reverse: Vec<SourceRawDirectFold>,
    pub(super) points: Vec<Vec<stack_report::SourceRawDirectPointRecord>>,
    pub(super) closures: Vec<f64>,
    pub(super) forward_fold_disagreement: f64,
    pub(super) reverse_fold_disagreement: f64,
    pub(super) coarse_closure: f64,
    pub(super) minimum_points: usize,
    pub(super) union_points: usize,
    pub(super) minimum_spatial_support: f64,
    pub(super) covered_overlap: usize,
    pub(super) maximum_holdout_loss: f64,
}

pub(super) fn source_raw_direct_basin_is_better(
    candidate: &SourceRawDirectBasinSelection,
    current: &SourceRawDirectBasinSelection,
) -> bool {
    let reciprocal = |selection: &SourceRawDirectBasinSelection| {
        selection.closures.iter().copied().fold(0.0, f64::max)
            <= VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX
            && selection.forward_fold_disagreement <= STATION_RELATION_MAX_MEDIAN_ERROR_PX
            && selection.reverse_fold_disagreement <= STATION_RELATION_MAX_MEDIAN_ERROR_PX
    };
    let candidate_reciprocal = reciprocal(candidate);
    let current_reciprocal = reciprocal(current);
    if candidate_reciprocal != current_reciprocal {
        return candidate_reciprocal;
    }
    if candidate.minimum_points != current.minimum_points {
        return candidate.minimum_points > current.minimum_points;
    }
    if candidate.union_points != current.union_points {
        return candidate.union_points > current.union_points;
    }
    if candidate
        .minimum_spatial_support
        .total_cmp(&current.minimum_spatial_support)
        .is_ne()
    {
        return candidate.minimum_spatial_support > current.minimum_spatial_support;
    }
    let candidate_closure = candidate.closures.iter().copied().fold(0.0, f64::max);
    let current_closure = current.closures.iter().copied().fold(0.0, f64::max);
    if candidate_closure.total_cmp(&current_closure).is_ne() {
        return candidate_closure < current_closure;
    }
    let candidate_disagreement = candidate
        .forward_fold_disagreement
        .max(candidate.reverse_fold_disagreement);
    let current_disagreement = current
        .forward_fold_disagreement
        .max(current.reverse_fold_disagreement);
    if candidate_disagreement
        .total_cmp(&current_disagreement)
        .is_ne()
    {
        return candidate_disagreement < current_disagreement;
    }
    if candidate.covered_overlap != current.covered_overlap {
        return candidate.covered_overlap > current.covered_overlap;
    }
    if candidate
        .coarse_closure
        .total_cmp(&current.coarse_closure)
        .is_ne()
    {
        return candidate.coarse_closure < current.coarse_closure;
    }
    if candidate
        .maximum_holdout_loss
        .total_cmp(&current.maximum_holdout_loss)
        .is_ne()
    {
        return candidate.maximum_holdout_loss < current.maximum_holdout_loss;
    }
    let candidate_key = (
        candidate.forward[0].seed_rank,
        candidate.reverse[0].seed_rank,
        virtual_tile_hypothesis_matrix_key(&candidate.forward[0].transform),
        virtual_tile_hypothesis_matrix_key(&candidate.reverse[0].transform),
    );
    let current_key = (
        current.forward[0].seed_rank,
        current.reverse[0].seed_rank,
        virtual_tile_hypothesis_matrix_key(&current.forward[0].transform),
        virtual_tile_hypothesis_matrix_key(&current.reverse[0].transform),
    );
    candidate_key < current_key
}

/// Select one reciprocal basin from independently generated and independently
/// refined forward/reverse peaks. The accepted station relation is never read:
/// ranking uses only transform composition, covered overlap, spatially spread
/// measured hold-out localizations, fold stability, and direct image losses.
pub(super) fn source_raw_direct_select_basin(
    left: &ImageInfo,
    right: &ImageInfo,
    left_pyramid: &SourceRawDirectPyramid,
    right_pyramid: &SourceRawDirectPyramid,
    forward_seeds: &[SourceRawCorrelationProposal],
    reverse_seeds: &[SourceRawCorrelationProposal],
) -> Option<SourceRawDirectBasinSelection> {
    let forward_seeds = &forward_seeds[..forward_seeds.len().min(SOURCE_RAW_DIRECT_MAX_SEEDS)];
    let reverse_seeds = &reverse_seeds[..reverse_seeds.len().min(SOURCE_RAW_DIRECT_MAX_SEEDS)];
    let forward_folds = (0..2)
        .map(|fold| {
            forward_seeds
                .iter()
                .enumerate()
                .map(|(rank, seed)| {
                    source_raw_direct_refine(
                        left_pyramid,
                        right_pyramid,
                        seed.proposal.left_to_right,
                        rank,
                        fold,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let reverse_folds = (0..2)
        .map(|fold| {
            reverse_seeds
                .iter()
                .enumerate()
                .map(|(rank, seed)| {
                    source_raw_direct_refine(
                        right_pyramid,
                        left_pyramid,
                        seed.proposal.left_to_right,
                        rank,
                        fold,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut best = None;
    for (forward_rank, forward_seed) in forward_seeds.iter().enumerate() {
        for (reverse_rank, reverse_seed) in reverse_seeds.iter().enumerate() {
            let forward = (0..2)
                .map(|fold| forward_folds[fold][forward_rank].clone())
                .collect::<Option<Vec<_>>>();
            let reverse = (0..2)
                .map(|fold| reverse_folds[fold][reverse_rank].clone())
                .collect::<Option<Vec<_>>>();
            let (Some(forward), Some(reverse)) = (forward, reverse) else {
                continue;
            };
            let coarse_forward = source_raw_direct_raw_transform(
                &forward_seed.proposal.left_to_right,
                left.scale_factor,
                right.scale_factor,
            );
            let coarse_reverse = source_raw_direct_raw_transform(
                &reverse_seed.proposal.left_to_right,
                right.scale_factor,
                left.scale_factor,
            );
            let Some(coarse_closure) =
                source_raw_direct_closure(&coarse_forward, &coarse_reverse, left.dimensions())
            else {
                continue;
            };
            let mut points = Vec::with_capacity(2);
            let mut closures = Vec::with_capacity(2);
            let mut spatial_support = Vec::with_capacity(2);
            for fold in 0..2 {
                let raw_forward = source_raw_direct_raw_transform(
                    &forward[fold].transform,
                    left.scale_factor,
                    right.scale_factor,
                );
                let raw_reverse = source_raw_direct_raw_transform(
                    &reverse[fold].transform,
                    right.scale_factor,
                    left.scale_factor,
                );
                let Some(closure) =
                    source_raw_direct_closure(&raw_forward, &raw_reverse, left.dimensions())
                else {
                    points.clear();
                    break;
                };
                let measured = source_raw_direct_localize_holdout(
                    left,
                    right,
                    left_pyramid,
                    right_pyramid,
                    &forward[fold].transform,
                    &reverse[fold].transform,
                    fold,
                );
                let measured_pairs = measured
                    .iter()
                    .map(|point| {
                        (
                            Point2::new(point.source[0], point.source[1]),
                            Point2::new(point.target[0], point.target[1]),
                        )
                    })
                    .collect::<Vec<_>>();
                spatial_support.push(station_relation_spatial_support(
                    &measured_pairs,
                    &raw_forward,
                    left.dimensions(),
                    right.dimensions(),
                ));
                closures.push(closure);
                points.push(measured);
            }
            if points.len() != 2 {
                continue;
            }
            let forward_raw = forward
                .iter()
                .map(|fold| {
                    source_raw_direct_raw_transform(
                        &fold.transform,
                        left.scale_factor,
                        right.scale_factor,
                    )
                })
                .collect::<Vec<_>>();
            let reverse_raw = reverse
                .iter()
                .map(|fold| {
                    source_raw_direct_raw_transform(
                        &fold.transform,
                        right.scale_factor,
                        left.scale_factor,
                    )
                })
                .collect::<Vec<_>>();
            let Some(forward_fold_disagreement) = focus_transform_disagreement_px(
                &forward_raw[0],
                &forward_raw[1],
                left.dimensions(),
            ) else {
                continue;
            };
            let Some(reverse_fold_disagreement) = focus_transform_disagreement_px(
                &reverse_raw[0],
                &reverse_raw[1],
                right.dimensions(),
            ) else {
                continue;
            };
            let union_points = points
                .iter()
                .flat_map(|fold| fold.iter().map(|point| point.cell))
                .collect::<BTreeSet<_>>()
                .len();
            let selection = SourceRawDirectBasinSelection {
                minimum_points: points.iter().map(Vec::len).min().unwrap_or_default(),
                union_points,
                minimum_spatial_support: spatial_support.into_iter().fold(1.0, f64::min),
                covered_overlap: forward_seed
                    .proposal
                    .overlap_samples
                    .min(reverse_seed.proposal.overlap_samples),
                maximum_holdout_loss: forward
                    .iter()
                    .chain(&reverse)
                    .map(|fold| fold.holdout.loss)
                    .fold(0.0, f64::max),
                forward,
                reverse,
                points,
                closures,
                forward_fold_disagreement,
                reverse_fold_disagreement,
                coarse_closure,
            };
            if best
                .as_ref()
                .is_none_or(|current| source_raw_direct_basin_is_better(&selection, current))
            {
                best = Some(selection);
            }
        }
    }
    best
}
