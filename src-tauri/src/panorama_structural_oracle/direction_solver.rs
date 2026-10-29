//! One solve direction: mutual descriptor matches, geometric-hash similarity proposals, model
//! inliers and correspondences, and the final measurement of each model.

use super::*;

pub(super) const HASH_MODELS: usize = 32;
pub(super) const MODEL_INLIER_PX: f64 = 6.0;

#[derive(Clone)]
pub(super) struct PrimitiveMatch {
    pub(super) source: usize,
    pub(super) target: usize,
    pub(super) distance: u64,
}

pub(super) fn mutual_structural_matches(
    source: &[StructuralPrimitive],
    target: &[StructuralPrimitive],
) -> Vec<PrimitiveMatch> {
    let nearest = |needle: &StructuralPrimitive, haystack: &[StructuralPrimitive]| {
        let mut distances = haystack
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                candidate.record.factor == needle.record.factor
                    && candidate.record.kind == needle.record.kind
            })
            .map(|(index, candidate)| {
                (
                    descriptor_distance(&needle.record.descriptor, &candidate.record.descriptor),
                    index,
                )
            })
            .collect::<Vec<_>>();
        distances.sort_unstable();
        let &(best, index) = distances.first()?;
        let second = distances.get(1).map_or(u64::MAX, |entry| entry.0);
        // Keep exact unique descriptors, otherwise require a modest margin.
        (best == 0 || best.saturating_mul(100) <= second.saturating_mul(96))
            .then_some((index, best))
    };
    let source_to_target = source
        .iter()
        .map(|primitive| nearest(primitive, target))
        .collect::<Vec<_>>();
    let target_to_source = target
        .iter()
        .map(|primitive| nearest(primitive, source))
        .collect::<Vec<_>>();
    let mut matches = source_to_target
        .into_iter()
        .enumerate()
        .filter_map(|(source_index, match_target)| {
            let (target_index, distance) = match_target?;
            (target_to_source
                .get(target_index)
                .and_then(|entry| *entry)
                .map(|entry| entry.0)
                == Some(source_index))
            .then_some(PrimitiveMatch {
                source: source_index,
                target: target_index,
                distance,
            })
        })
        .collect::<Vec<_>>();
    matches.sort_by_key(|item| {
        (
            item.distance,
            source[item.source].record.id,
            target[item.target].record.id,
        )
    });
    matches
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct HashBin(i32, i32, i16, i16);

pub(super) fn similarity_from_pair(
    source_a: Point2<f64>,
    source_b: Point2<f64>,
    target_a: Point2<f64>,
    target_b: Point2<f64>,
) -> Option<Matrix3<f64>> {
    let source = source_b - source_a;
    let target = target_b - target_a;
    let source_length = source.norm();
    let target_length = target.norm();
    if source_length < 32.0 || target_length < 32.0 {
        return None;
    }
    let scale = target_length / source_length;
    if !(STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX).contains(&scale) {
        return None;
    }
    let theta = target.y.atan2(target.x) - source.y.atan2(source.x);
    if theta.abs() > 20.0f64.to_radians() {
        return None;
    }
    let (sin, cos) = theta.sin_cos();
    let tx = target_a.x - scale * (cos * source_a.x - sin * source_a.y);
    let ty = target_a.y - scale * (sin * source_a.x + cos * source_a.y);
    Some(Matrix3::new(
        scale * cos,
        -scale * sin,
        tx,
        scale * sin,
        scale * cos,
        ty,
        0.0,
        0.0,
        1.0,
    ))
}

pub(super) fn geometric_hash_proposals(
    source: &[StructuralPrimitive],
    target: &[StructuralPrimitive],
    matches: &[PrimitiveMatch],
) -> Vec<Matrix3<f64>> {
    let bounded = &matches[..matches.len().min(256)];
    let mut bins = BTreeMap::<HashBin, (usize, Matrix3<f64>)>::new();
    for left in 0..bounded.len() {
        for right in left + 1..bounded.len() {
            let a = &bounded[left];
            let b = &bounded[right];
            if a.source == b.source || a.target == b.target {
                continue;
            }
            let Some(transform) = similarity_from_pair(
                source[a.source].point,
                source[b.source].point,
                target[a.target].point,
                target[b.target].point,
            ) else {
                continue;
            };
            let scale = homography_scale_ratio(&transform).unwrap_or(0.0);
            let angle = transform[(1, 0)].atan2(transform[(0, 0)]);
            let key = HashBin(
                (transform[(0, 2)] / 12.0).round() as i32,
                (transform[(1, 2)] / 12.0).round() as i32,
                (angle.to_degrees() / 1.5).round() as i16,
                ((scale - 1.0) / 0.0025).round() as i16,
            );
            bins.entry(key)
                .and_modify(|entry| entry.0 += 1)
                .or_insert((1, transform));
        }
    }
    let mut ranked = bins
        .into_iter()
        .map(|(key, (votes, transform))| (std::cmp::Reverse(votes), key, transform))
        .collect::<Vec<_>>();
    ranked.sort_by_key(|entry| (entry.0, entry.1));
    ranked
        .into_iter()
        .take(HASH_MODELS)
        .map(|entry| entry.2)
        .collect()
}

pub(super) fn model_inlier_matches(
    transform: &Matrix3<f64>,
    source: &[StructuralPrimitive],
    target: &[StructuralPrimitive],
    matches: &[PrimitiveMatch],
) -> Vec<PrimitiveMatch> {
    let Some(inverse) = transform.try_inverse() else {
        return Vec::new();
    };
    let mut inliers = matches
        .iter()
        .filter(|item| {
            let Some(projected_target) = transformed_point(transform, source[item.source].point)
            else {
                return false;
            };
            let Some(projected_source) = transformed_point(&inverse, target[item.target].point)
            else {
                return false;
            };
            (projected_target - target[item.target].point).norm() <= MODEL_INLIER_PX
                && (projected_source - source[item.source].point).norm() <= MODEL_INLIER_PX
        })
        .cloned()
        .collect::<Vec<_>>();
    inliers.sort_by_key(|item| (source[item.source].record.id, target[item.target].record.id));
    inliers
}

pub(super) fn correspondence_record(
    source: &StructuralPrimitive,
    target: &StructuralPrimitive,
    distance: u64,
) -> StructuralCorrespondenceRecord {
    StructuralCorrespondenceRecord {
        source_primitive_id: source.record.id,
        target_primitive_id: target.record.id,
        source_x_bits: source.point.x.to_bits(),
        source_y_bits: source.point.y.to_bits(),
        target_x_bits: target.point.x.to_bits(),
        target_y_bits: target.point.y.to_bits(),
        descriptor_distance_q: distance.min(u64::from(u32::MAX)) as u32,
    }
}

pub(super) fn solve_structural_direction(
    source_station: usize,
    target_station: usize,
    _tiles: &[ImageInfo],
    pyramid: &VirtualTileMatchPyramid,
) -> (
    StructuralDirectionRecord,
    Vec<(Matrix3<f64>, Vec<PrimitiveMatch>)>,
) {
    let source = extract_structural_primitives(pyramid, source_station);
    let target = extract_structural_primitives(pyramid, target_station);
    let matches = mutual_structural_matches(&source, &target);
    let proposals = geometric_hash_proposals(&source, &target, &matches);
    let mut fitted = Vec::<(Matrix3<f64>, Vec<PrimitiveMatch>)>::new();
    for proposal in &proposals {
        let seed_inliers = model_inlier_matches(proposal, &source, &target, &matches);
        if seed_inliers.len() < 4 {
            continue;
        }
        let points = seed_inliers
            .iter()
            .map(|item| (source[item.source].point, target[item.target].point))
            .collect::<Vec<_>>();
        let Some((fit, indices)) =
            processing::find_homography_ransac_points_stable_with_min_inliers(
                &points,
                STATION_RELATION_MAX_MEDIAN_ERROR_PX,
                4,
            )
        else {
            continue;
        };
        let fit_matches = indices
            .into_iter()
            .map(|index| seed_inliers[index].clone())
            .collect::<Vec<_>>();
        let Some(refit) = processing::compute_homography(
            &fit_matches
                .iter()
                .map(|item| (source[item.source].point, target[item.target].point))
                .collect::<Vec<_>>(),
        )
        .or(Some(fit)) else {
            continue;
        };
        let inliers = model_inlier_matches(&refit, &source, &target, &matches);
        if inliers.len() < 4 {
            continue;
        }
        let duplicate = fitted.iter().any(|(existing, _)| {
            focus_transform_disagreement_px(
                existing,
                &refit,
                pyramid.native_dimensions[source_station],
            )
            .is_some_and(|value| value < 1.0)
        });
        if !duplicate {
            fitted.push((refit, inliers));
        }
    }
    fitted.sort_by(|left, right| {
        right.1.len().cmp(&left.1.len()).then_with(|| {
            matrix3_row_major(&left.0)
                .iter()
                .map(|value| value.to_bits())
                .cmp(
                    matrix3_row_major(&right.0)
                        .iter()
                        .map(|value| value.to_bits()),
                )
        })
    });
    let models = fitted
        .iter()
        .map(|(transform, inliers)| {
            let points = inliers
                .iter()
                .map(|item| (source[item.source].point, target[item.target].point))
                .collect::<Vec<_>>();
            StructuralModelRecord {
                transform: matrix3_row_major(transform),
                correspondences: inliers
                    .iter()
                    .map(|item| {
                        correspondence_record(
                            &source[item.source],
                            &target[item.target],
                            item.distance,
                        )
                    })
                    .collect(),
                median_symmetric_error_px: median_symmetric_error(transform, &points),
                scale_ratio: homography_scale_ratio(transform).unwrap_or(f64::INFINITY),
                covered_overlap_hull: 0.0,
                preserves_convex_orientation: homography_preserves_focus_orientation(
                    transform,
                    pyramid.native_dimensions[source_station],
                ),
                low_frequency_mean_relative_difference: f64::INFINITY,
                edge_strength_ratio: f64::INFINITY,
                median_edge_orientation_difference_degrees: f64::INFINITY,
                rejection_reasons: vec!["structural_model_not_finally_measured".to_string()],
            }
        })
        .collect();
    (
        StructuralDirectionRecord {
            source_primitives: source
                .iter()
                .map(|primitive| primitive.record.clone())
                .collect(),
            target_primitives: target
                .iter()
                .map(|primitive| primitive.record.clone())
                .collect(),
            mutual_matches: matches.len(),
            hash_proposals: proposals.len(),
            models,
        },
        fitted,
    )
}

pub(super) fn point_from_bits(x: u64, y: u64) -> Point2<f64> {
    Point2::new(f64::from_bits(x), f64::from_bits(y))
}

pub(super) fn finally_measure_models(
    direction: &mut StructuralDirectionRecord,
    source_station: usize,
    target_station: usize,
    tiles: &[ImageInfo],
    pyramid: &VirtualTileMatchPyramid,
) {
    for model in &mut direction.models {
        let transform = Matrix3::from_row_slice(&model.transform);
        let points = model
            .correspondences
            .iter()
            .map(|point| {
                (
                    point_from_bits(point.source_x_bits, point.source_y_bits),
                    point_from_bits(point.target_x_bits, point.target_y_bits),
                )
            })
            .collect::<Vec<_>>();
        let quality_transform = transform;
        let dense = focus_overlap_quality_measurements_on_coverage(
            &tiles[source_station],
            &tiles[target_station],
            &quality_transform,
            Some((
                &pyramid.quality_coverages[source_station],
                &pyramid.quality_coverages[target_station],
            )),
        );
        let overlap = StationOverlapMeasure::Covered {
            source: &pyramid.coverages.last().unwrap()[source_station],
            target: &pyramid.coverages.last().unwrap()[target_station],
            factor: *pyramid.factors.last().unwrap(),
        };
        if let Some(measured) = measure_station_relation_candidate(
            points.len(),
            &transform,
            &points,
            pyramid.native_dimensions[source_station],
            pyramid.native_dimensions[target_station],
            Some(overlap),
            dense,
            false,
        ) {
            let values = measured.measurements;
            model.median_symmetric_error_px = values.median_error_px;
            model.scale_ratio = values.scale_ratio;
            model.covered_overlap_hull = values.spatial_support;
            model.preserves_convex_orientation = values.preserves_convex_orientation;
            model.low_frequency_mean_relative_difference =
                values.low_frequency_mean_relative_difference;
            model.edge_strength_ratio = values.edge_strength_ratio;
            model.median_edge_orientation_difference_degrees =
                values.median_edge_orientation_difference_degrees;
            model.rejection_reasons = station_relation_rejection_reasons(&values)
                .into_iter()
                .map(str::to_string)
                .collect();
        }
    }
}

pub(super) fn best_empty_rejection_model(
    direction: &StructuralDirectionRecord,
) -> Option<&StructuralModelRecord> {
    direction
        .models
        .iter()
        .filter(|model| model.rejection_reasons.is_empty())
        .max_by_key(|model| model.correspondences.len())
}
