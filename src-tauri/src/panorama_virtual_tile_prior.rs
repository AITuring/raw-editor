//! Virtual-tile prior relation selection and hypothesis ranking.

use super::*;

pub(crate) fn normalized_homography(transform: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let scale = transform[(2, 2)];
    if !scale.is_finite() || scale.abs() <= f64::EPSILON {
        return None;
    }
    let normalized = transform / scale;
    normalized
        .iter()
        .all(|value| value.is_finite())
        .then_some(normalized)
}

/// Rendering-prior relation `left tile -> right tile` in native tile pixels.
pub(crate) fn virtual_tile_prior_relation(
    left: &ImageInfo,
    right: &ImageInfo,
    tile_to_world: &HashMap<usize, Matrix3<f64>>,
) -> Option<Matrix3<f64>> {
    let left_to_world = tile_to_world.get(&left.id)?;
    let world_to_right = tile_to_world.get(&right.id)?.try_inverse()?;
    normalized_homography(&(world_to_right * left_to_world))
}

/// Prior-free covered-feature relation between two feature planes, in native
/// tile pixels, with its RANSAC inlier count. It is a proposal only.
pub(crate) fn virtual_tile_prior_free_relation(
    left_factor: u32,
    left_features: &[Feature],
    right_factor: u32,
    right_features: &[Feature],
    left_native_dimensions: (u32, u32),
) -> Option<(Matrix3<f64>, usize)> {
    let matches = processing::match_features(left_features, right_features);
    let correspondences = matches
        .iter()
        .map(|matched| {
            let left = left_features[matched.index1].keypoint;
            let right = right_features[matched.index2].keypoint;
            (
                Point2::new(f64::from(left.x), f64::from(left.y)),
                Point2::new(f64::from(right.x), f64::from(right.y)),
            )
        })
        .collect::<Vec<_>>();
    let (homography, inliers) = processing::find_homography_ransac_points_stable_with_min_inliers(
        &correspondences,
        5.0,
        STATION_RELATION_MIN_INLIERS,
    )?;
    if inliers.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    let native = normalized_homography(
        &(VirtualTileMatchPlane::level_to_native(right_factor)
            * homography
            * VirtualTileMatchPlane::native_to_level(left_factor)),
    )?;
    (native.try_inverse().is_some()
        && homography_preserves_focus_orientation(&native, left_native_dimensions))
    .then_some((native, inliers.len()))
}

/// A covered-feature seed can still repair a rendering prior when the
/// subsequent finest-plane polish fitted a model but its inlier hull missed
/// the fixed spatial-support gate. The seed remains a proposal only: the
/// repaired station is rendered again and the ordinary bounded matcher must
/// accept the resulting relation before it enters the pose solver.
pub(crate) fn virtual_tile_prior_free_seed_fallback_conditions(
    model_fitted: bool,
    failure_stage: &str,
    fitted_inliers: usize,
    prior_free_inliers: usize,
) -> bool {
    model_fitted
        && failure_stage == "insufficient_polished_support"
        && fitted_inliers >= STATION_RELATION_MIN_INLIERS
        // A covered-feature seed is only a repair proposal when it carries at
        // least as much support as the model that the finest-plane polish
        // already fitted.  A smaller seed can move an entire station cluster
        // on a repeated texture even when the polished model has hundreds of
        // mutually consistent matches but fails only its spatial-support gate.
        && prior_free_inliers >= fitted_inliers
}

pub(crate) fn virtual_tile_prior_free_seed_fallback_allowed(
    diagnostic: &stack_report::StationRelationCandidateRecord,
) -> bool {
    virtual_tile_prior_free_seed_fallback_conditions(
        diagnostic.model_fitted,
        &diagnostic.failure_stage,
        diagnostic.fitted_inliers,
        diagnostic.prior_free_inliers,
    )
}

pub(crate) fn log_virtual_tile_candidate(
    diagnostic: &stack_report::StationRelationCandidateRecord,
    accepted: bool,
) {
    println!(
        "  - Virtual_Tile candidate {}->{}: seed={} level=1/{} of {:?} accepted={} prior_free_inliers={} repaired={} grid_probes={} measurable={} bidirectional={} cells={} radius={:.1}px residual_mag=[{:.2},{:.2},{:.2}] translation={}/{:.4}/{:?}/{} affine={}/{:.4}/{:?}/{} projective={}/{:.4}/{:?}/{} residual_model={} fit_inliers={} support={:.4} prefit_structural_orientation={} fitted_orientation={} stage={}",
        diagnostic.left,
        diagnostic.right,
        diagnostic.seed,
        diagnostic.measurement_factor,
        diagnostic.pyramid_factors,
        accepted,
        diagnostic.prior_free_inliers,
        diagnostic.after_prior_repair,
        diagnostic.grid_probe_count,
        diagnostic.measurable_patch_count,
        diagnostic.bidirectional_match_count,
        diagnostic.occupied_grid_cells,
        diagnostic.search_radius_px,
        diagnostic.residual_magnitude_px.p10,
        diagnostic.residual_magnitude_px.median,
        diagnostic.residual_magnitude_px.p90,
        diagnostic.residual_translation.candidate_inliers,
        diagnostic.residual_translation.final_support,
        diagnostic.residual_translation.bounded_correction,
        diagnostic.residual_translation.rejection_reason,
        diagnostic.residual_affine.candidate_inliers,
        diagnostic.residual_affine.final_support,
        diagnostic.residual_affine.bounded_correction,
        diagnostic.residual_affine.rejection_reason,
        diagnostic.residual_projective.candidate_inliers,
        diagnostic.residual_projective.final_support,
        diagnostic.residual_projective.bounded_correction,
        diagnostic.residual_projective.rejection_reason,
        diagnostic.residual_model,
        diagnostic.fitted_inliers,
        diagnostic.hull_support,
        diagnostic
            .median_prefit_structural_orientation_difference_degrees
            .map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}deg")),
        diagnostic
            .median_fitted_orientation_difference_degrees
            .map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}deg")),
        diagnostic.failure_stage,
    );
}

pub(crate) type VirtualTileRelationHypotheses = HashMap<(usize, usize), Vec<MatchInfo>>;

pub(crate) fn station_relation_tile_geometry_record(
    tile: &ImageInfo,
    coverage: Option<&stitching::CoverageMask>,
    factor: u32,
    tile_to_world: Option<&Matrix3<f64>>,
) -> stack_report::StationRelationTileGeometryRecord {
    let factor = factor.max(1);
    let effective_bounds = coverage
        .and_then(stitching::CoverageMask::covered_bounds)
        .map(|(x, y, width, height)| {
            let native_x = x.saturating_mul(factor).min(tile.width);
            let native_y = y.saturating_mul(factor).min(tile.height);
            let native_right = x
                .saturating_add(width)
                .saturating_mul(factor)
                .min(tile.width);
            let native_bottom = y
                .saturating_add(height)
                .saturating_mul(factor)
                .min(tile.height);
            [
                native_x,
                native_y,
                native_right.saturating_sub(native_x),
                native_bottom.saturating_sub(native_y),
            ]
        });
    stack_report::StationRelationTileGeometryRecord {
        dimensions: [tile.width, tile.height],
        effective_bounds,
        tile_to_world: tile_to_world.map(matrix3_row_major),
    }
}

pub(crate) fn populate_station_relation_candidate_geometry(
    diagnostic: &mut stack_report::StationRelationCandidateRecord,
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
    tile_to_world: &HashMap<usize, Matrix3<f64>>,
) {
    let factor = coverages.overlap_factor;
    if let Some(tile) = tiles.get(diagnostic.left) {
        diagnostic.left_tile = station_relation_tile_geometry_record(
            tile,
            coverages.overlap.get(diagnostic.left),
            factor,
            tile_to_world.get(&tile.id),
        );
    }
    if let Some(tile) = tiles.get(diagnostic.right) {
        diagnostic.right_tile = station_relation_tile_geometry_record(
            tile,
            coverages.overlap.get(diagnostic.right),
            factor,
            tile_to_world.get(&tile.id),
        );
    }
}

/// Authoritative relations: bounded coarse-to-fine hypotheses around the
/// rendering prior for every topology candidate (or for `only` when given).
/// Returns `(relations, diagnostics)`, diagnostics in candidate order.
pub(crate) fn virtual_tile_measure_prior_relations(
    tiles: &[ImageInfo],
    pyramid: &VirtualTileMatchPyramid,
    tile_to_world: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
    only: Option<&BTreeSet<(usize, usize)>>,
    after_prior_repair: bool,
) -> (
    VirtualTileRelationHypotheses,
    Vec<stack_report::StationRelationCandidateRecord>,
) {
    let mut relations = VirtualTileRelationHypotheses::new();
    let mut diagnostics = Vec::with_capacity(station_topology.candidates.len());
    for candidate in &station_topology.candidates {
        let (left, right) = (candidate.left, candidate.right);
        if only.is_some_and(|pairs| !pairs.contains(&(left.min(right), left.max(right)))) {
            continue;
        }
        let prior = tiles
            .get(left)
            .zip(tiles.get(right))
            .and_then(|(left_tile, right_tile)| {
                virtual_tile_prior_relation(left_tile, right_tile, tile_to_world)
            });
        let (relation, mut diagnostic) = match prior {
            Some(prior) => virtual_tile_pyramid_station_match(
                left,
                right,
                pyramid,
                &prior,
                VIRTUAL_TILE_SEED_RENDERING_PRIOR,
            ),
            None => {
                let (_, mut diagnostic) = virtual_tile_direct_station_match(
                    left,
                    right,
                    &[],
                    &[],
                    &HashMap::new(),
                    STATION_RELATION_MAX_MEDIAN_ERROR_PX,
                );
                diagnostic.failure_stage = "missing_initial_pose".to_string();
                (Vec::new(), diagnostic)
            }
        };
        diagnostic.after_prior_repair = after_prior_repair;
        if let Some(selection) = select_virtual_tile_relation_hypothesis(
            tiles,
            left,
            right,
            &relation,
            pyramid.solver_coverages(),
        ) {
            diagnostic.hypotheses = selection.outcomes;
        }
        populate_station_relation_candidate_geometry(
            &mut diagnostic,
            tiles,
            pyramid.solver_coverages(),
            tile_to_world,
        );
        log_virtual_tile_candidate(&diagnostic, !relation.is_empty());
        if !relation.is_empty() {
            relations.insert((left, right), relation);
        }
        diagnostics.push(diagnostic);
    }
    (relations, diagnostics)
}

/// A densely verified prior-free relation between two stations whose rendering
/// priors disagree with it. It may only repair a prior; it is never a
/// Station_Relation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualTilePriorProposal {
    pub(crate) left: usize,
    pub(crate) right: usize,
    /// Measured `left tile -> right tile` relation in native tile pixels.
    pub(crate) left_to_right: Matrix3<f64>,
    pub(crate) inliers: usize,
}

/// One cluster member's rendering-prior repair: `world_correction` maps the
/// station's current world placement onto the placement the proposal measured
/// relative to an already consistent station.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VirtualTilePriorRepair {
    pub(crate) station: usize,
    pub(crate) cluster_root: usize,
    pub(crate) fixed_station: usize,
    pub(crate) relation: (usize, usize),
    pub(crate) relation_inliers: usize,
    pub(crate) world_correction: Matrix3<f64>,
}

/// A rendering-prior repair may move a station only within the same bounded
/// geometry envelope used by closure corrections and the station scale gate.
/// Keep this decision pure so production and property tests share the exact
/// finite-value and inclusive-boundary semantics.
pub(crate) fn virtual_tile_prior_repair_is_bounded(
    center_shift_px: f64,
    correction_scale_ratio: f64,
) -> bool {
    center_shift_px.is_finite()
        && center_shift_px >= 0.0
        && center_shift_px <= CLOSURE_MAX_CORNER_CORRECTION_PX
        && correction_scale_ratio.is_finite()
        && (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
            .contains(&correction_scale_ratio)
}

pub(crate) fn virtual_tile_prior_repair_measurements(
    correction: &Matrix3<f64>,
    station_world_center: Point2<f64>,
) -> Option<(f64, f64)> {
    let corrected_center = transformed_point(correction, station_world_center)?;
    let center_shift_px = (corrected_center - station_world_center).norm();
    let correction_scale_ratio =
        virtual_tile_repair_local_scale_ratio(correction, station_world_center)?;
    (center_shift_px.is_finite() && correction_scale_ratio.is_finite())
        .then_some((center_shift_px, correction_scale_ratio))
}

pub(crate) fn virtual_tile_prior_repair_is_bounded_at_center(
    correction: &Matrix3<f64>,
    station_world_center: Point2<f64>,
) -> bool {
    virtual_tile_prior_repair_measurements(correction, station_world_center).is_some_and(
        |(center_shift_px, correction_scale_ratio)| {
            virtual_tile_prior_repair_is_bounded(center_shift_px, correction_scale_ratio)
        },
    )
}

/// Apply the exact pure Station_Relation gates to a Virtual_Tile match before
/// it is allowed to merge rendering-prior repair clusters. This mirrors the
/// measurements in `solve_station_poses_for_evidence_with_report` without
/// recording a second report or running any degradation side effects.
pub(crate) fn virtual_tile_relation_is_authoritative(
    left: usize,
    right: usize,
    relation: &MatchInfo,
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> bool {
    if left >= tiles.len()
        || right >= tiles.len()
        || left == right
        || relation.sequence_bridge
        || relation.points.len() < FOCUS_MODEL_MIN_INLIERS
    {
        return false;
    }
    let canonical = match canonical_relation_view(tiles, left, right, relation) {
        Ok(view) => view,
        Err(CanonicalRelationViewError::NonInvertible) => return false,
        Err(CanonicalRelationViewError::IndexOutOfBounds {
            left,
            right,
            image_count,
        }) => panic!("canonical relation pair ({left}, {right}) is outside {image_count} tiles"),
    };
    let quality_source = canonical.source_index;
    let quality_target = canonical.target_index;
    let quality_points = if canonical.inverted {
        relation
            .points
            .iter()
            .map(|&(source, target)| (target, source))
            .collect::<Vec<_>>()
    } else {
        relation.points.clone()
    };
    let quality_coverages = coverages
        .quality
        .get(quality_source)
        .zip(coverages.quality.get(quality_target));
    let dense_measurements = focus_overlap_quality_measurements_on_coverage(
        &tiles[quality_source],
        &tiles[quality_target],
        &canonical.homography,
        quality_coverages,
    );
    let overlap_measure = coverages.overlap_measure(quality_source, quality_target);
    measure_station_relation_candidate(
        relation.inliers.max(quality_points.len()),
        &canonical.homography,
        &quality_points,
        tiles[quality_source].dimensions(),
        tiles[quality_target].dimensions(),
        overlap_measure,
        dense_measurements,
    )
    .is_some_and(|candidate| station_relation_rejection_reasons(&candidate.measurements).is_empty())
}

pub(crate) fn matrix3_row_major(transform: &Matrix3<f64>) -> [f64; 9] {
    [
        transform[(0, 0)],
        transform[(0, 1)],
        transform[(0, 2)],
        transform[(1, 0)],
        transform[(1, 1)],
        transform[(1, 2)],
        transform[(2, 0)],
        transform[(2, 1)],
        transform[(2, 2)],
    ]
}

pub(crate) fn virtual_tile_hypothesis_model_rank(transform: &Matrix3<f64>) -> u8 {
    let Some(transform) = normalized_homography(transform) else {
        return u8::MAX;
    };
    if transform[(2, 0)].abs() > 1e-12 || transform[(2, 1)].abs() > 1e-12 {
        return 2;
    }
    let first = nalgebra::Vector2::new(transform[(0, 0)], transform[(1, 0)]);
    let second = nalgebra::Vector2::new(transform[(0, 1)], transform[(1, 1)]);
    let scale = first.norm().max(second.norm()).max(f64::EPSILON);
    if first.dot(&second).abs() <= 1e-10 * scale * scale
        && (first.norm() - second.norm()).abs() <= 1e-10 * scale
    {
        0
    } else {
        1
    }
}

pub(crate) fn virtual_tile_hypothesis_matrix_key(transform: &Matrix3<f64>) -> [u64; 9] {
    let mut key = [0; 9];
    for (slot, value) in key.iter_mut().zip(transform.iter()) {
        *slot = value.to_bits();
    }
    key
}

#[derive(Clone)]
pub(crate) struct MeasuredVirtualTileHypothesis {
    pub(crate) relation: MatchInfo,
    pub(crate) canonical_source: usize,
    pub(crate) canonical_target: usize,
    pub(crate) canonical_homography: Matrix3<f64>,
    pub(crate) correspondence_count: usize,
    pub(crate) correspondence_sample: Vec<stack_report::StationRelationCorrespondenceRecord>,
    pub(crate) holdout_fit_count: usize,
    pub(crate) holdout_count: usize,
    pub(crate) holdout_median_error_px: Option<f64>,
    pub(crate) holdout_spatial_support: Option<f64>,
    pub(crate) holdout_transform: Option<Matrix3<f64>>,
    pub(crate) measured: MeasuredStationRelationCandidate,
    pub(crate) rejection_count: usize,
    pub(crate) model_rank: u8,
}

#[derive(Clone)]
pub(crate) struct VirtualTileHypothesisSelection {
    pub(crate) relation: MatchInfo,
    pub(crate) passes: bool,
    pub(crate) outcomes: Vec<stack_report::StationRelationHypothesisRecord>,
}

#[allow(clippy::type_complexity)]
pub(crate) fn virtual_tile_hypothesis_holdout(
    model_rank: u8,
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    overlap_measure: Option<StationOverlapMeasure<'_>>,
) -> (usize, usize, Option<f64>, Option<f64>, Option<Matrix3<f64>>) {
    let cell = f64::from(source_dimensions.0.max(source_dimensions.1).max(1)) / 12.0;
    let mut fit = Vec::new();
    let mut holdout = Vec::new();
    for &point in points {
        let column = (point.0.x / cell).floor() as i64;
        let row = (point.0.y / cell).floor() as i64;
        if (column + row).rem_euclid(2) == 0 {
            fit.push(point);
        } else {
            holdout.push(point);
        }
    }
    let model = match model_rank {
        0 => VirtualTilePolishModel::Similarity,
        1 => VirtualTilePolishModel::Affine,
        2 => VirtualTilePolishModel::Projective,
        _ => return (fit.len(), holdout.len(), None, None, None),
    };
    let transform = model
        .fit(
            &fit,
            VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX,
            source_dimensions,
        )
        .map(|(transform, _)| transform);
    let median_error_px = transform
        .as_ref()
        .map(|transform| median_symmetric_error(transform, &holdout))
        .filter(|error| error.is_finite());
    let spatial_support = transform.as_ref().and_then(|transform| {
        overlap_measure.map(|measure| {
            station_relation_spatial_support_with(
                &holdout,
                transform,
                source_dimensions,
                target_dimensions,
                measure,
            )
        })
    });
    (
        fit.len(),
        holdout.len(),
        median_error_px,
        spatial_support,
        transform,
    )
}

/// Measure, canonicalize, deduplicate, and rank one pair's independent model
/// hypotheses. Fully passing hypotheses always outrank failing hypotheses;
/// ties use the complete deterministic quality tuple and then intrinsic model
/// complexity/matrix bits, never input order. Both provisional clustering and
/// the final solver call this exact selector.
pub(crate) fn select_virtual_tile_relation_hypothesis(
    tiles: &[ImageInfo],
    left: usize,
    right: usize,
    hypotheses: &[MatchInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> Option<VirtualTileHypothesisSelection> {
    let mut measured = Vec::<MeasuredVirtualTileHypothesis>::new();
    for relation in hypotheses {
        if left == right
            || relation.sequence_bridge
            || relation.points.len() < FOCUS_MODEL_MIN_INLIERS
        {
            continue;
        }
        let canonical = match canonical_relation_view(tiles, left, right, relation) {
            Ok(view) => view,
            Err(CanonicalRelationViewError::NonInvertible) => continue,
            Err(CanonicalRelationViewError::IndexOutOfBounds {
                left,
                right,
                image_count,
            }) => {
                panic!("canonical relation pair ({left}, {right}) is outside {image_count} tiles")
            }
        };
        let points: Cow<'_, [(Point2<f64>, Point2<f64>)]> = if canonical.inverted {
            Cow::Owned(
                relation
                    .points
                    .iter()
                    .map(|&(source, target)| (target, source))
                    .collect(),
            )
        } else {
            Cow::Borrowed(&relation.points)
        };
        let quality_coverages = coverages
            .quality
            .get(canonical.source_index)
            .zip(coverages.quality.get(canonical.target_index));
        let dense_measurements = focus_overlap_quality_measurements_on_coverage(
            &tiles[canonical.source_index],
            &tiles[canonical.target_index],
            &canonical.homography,
            quality_coverages,
        );
        let overlap_measure =
            coverages.overlap_measure(canonical.source_index, canonical.target_index);
        let Some(candidate) = measure_station_relation_candidate(
            relation.inliers.max(points.len()),
            &canonical.homography,
            &points,
            tiles[canonical.source_index].dimensions(),
            tiles[canonical.target_index].dimensions(),
            overlap_measure,
            dense_measurements,
        ) else {
            continue;
        };
        let correspondence_count = points.len();
        let sample_count = correspondence_count.min(64);
        let correspondence_sample = (0..sample_count)
            .map(|sample| {
                let index = if sample_count <= 1 {
                    0
                } else {
                    sample * (correspondence_count - 1) / (sample_count - 1)
                };
                let (source, target) = points[index];
                let local_scale = virtual_tile_local_jacobian(&canonical.homography, source)
                    .map(|jacobian| jacobian.determinant().abs().sqrt());
                stack_report::StationRelationCorrespondenceRecord {
                    source: [source.x, source.y],
                    target: [target.x, target.y],
                    local_scale,
                }
            })
            .collect();
        let (
            holdout_fit_count,
            holdout_count,
            holdout_median_error_px,
            holdout_spatial_support,
            holdout_transform,
        ) = virtual_tile_hypothesis_holdout(
            virtual_tile_hypothesis_model_rank(&canonical.homography),
            &points,
            tiles[canonical.source_index].dimensions(),
            tiles[canonical.target_index].dimensions(),
            overlap_measure,
        );
        measured.push(MeasuredVirtualTileHypothesis {
            relation: relation.clone(),
            canonical_source: canonical.source_index,
            canonical_target: canonical.target_index,
            canonical_homography: canonical.homography,
            correspondence_count,
            correspondence_sample,
            holdout_fit_count,
            holdout_count,
            holdout_median_error_px,
            holdout_spatial_support,
            holdout_transform,
            measured: candidate,
            rejection_count: station_relation_rejection_reasons(&candidate.measurements).len(),
            model_rank: virtual_tile_hypothesis_model_rank(&canonical.homography),
        });
    }
    measured.sort_by(|left, right| {
        left.rejection_count
            .cmp(&right.rejection_count)
            .then_with(|| right.measured.quality.total_cmp(&left.measured.quality))
            .then_with(|| {
                right
                    .measured
                    .measurements
                    .spatial_support
                    .total_cmp(&left.measured.measurements.spatial_support)
            })
            .then_with(|| {
                right
                    .measured
                    .measurements
                    .inliers
                    .cmp(&left.measured.measurements.inliers)
            })
            .then_with(|| {
                left.measured
                    .measurements
                    .median_error_px
                    .total_cmp(&right.measured.measurements.median_error_px)
            })
            .then_with(|| left.model_rank.cmp(&right.model_rank))
            .then_with(|| {
                virtual_tile_hypothesis_matrix_key(&left.canonical_homography).cmp(
                    &virtual_tile_hypothesis_matrix_key(&right.canonical_homography),
                )
            })
    });
    let mut deduplicated = Vec::<MeasuredVirtualTileHypothesis>::new();
    for hypothesis in measured {
        if deduplicated.iter().any(|existing| {
            virtual_tile_hypotheses_are_equivalent(
                &existing.canonical_homography,
                &hypothesis.canonical_homography,
                tiles[left].dimensions(),
            )
        }) {
            continue;
        }
        deduplicated.push(hypothesis);
    }
    let outcomes = deduplicated
        .iter()
        .enumerate()
        .map(|(index, hypothesis)| {
            let measurements = hypothesis.measured.measurements;
            stack_report::StationRelationHypothesisRecord {
                model: match hypothesis.model_rank {
                    0 => "similarity",
                    1 => "affine",
                    2 => "projective",
                    _ => "unknown",
                }
                .to_string(),
                selected: index == 0,
                inliers: measurements.inliers,
                median_error_px: measurements.median_error_px,
                scale_ratio: measurements.scale_ratio,
                spatial_support: measurements.spatial_support,
                low_frequency_mean_relative_difference: measurements
                    .low_frequency_mean_relative_difference,
                edge_strength_ratio: measurements.edge_strength_ratio,
                median_edge_orientation_difference_degrees: measurements
                    .median_edge_orientation_difference_degrees,
                preserves_convex_orientation: measurements.preserves_convex_orientation,
                rejection_reasons: station_relation_rejection_reasons(&measurements)
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                canonical_source: hypothesis.canonical_source,
                canonical_target: hypothesis.canonical_target,
                source_to_target_transform: matrix3_row_major(&hypothesis.canonical_homography),
                correspondence_count: hypothesis.correspondence_count,
                correspondence_sample: hypothesis.correspondence_sample.clone(),
                holdout_fit_count: hypothesis.holdout_fit_count,
                holdout_count: hypothesis.holdout_count,
                holdout_median_error_px: hypothesis.holdout_median_error_px,
                holdout_spatial_support: hypothesis.holdout_spatial_support,
                holdout_transform: hypothesis.holdout_transform.as_ref().map(matrix3_row_major),
            }
        })
        .collect::<Vec<_>>();
    deduplicated
        .first()
        .map(|hypothesis| VirtualTileHypothesisSelection {
            relation: hypothesis.relation.clone(),
            passes: hypothesis.rejection_count == 0,
            outcomes,
        })
}

pub(crate) fn select_virtual_tile_relation_hypotheses(
    tiles: &[ImageInfo],
    relations: &VirtualTileRelationHypotheses,
    coverages: VirtualTileSolverCoverages<'_>,
) -> HashMap<(usize, usize), MatchInfo> {
    let mut selected = HashMap::new();
    let mut pairs = relations.iter().collect::<Vec<_>>();
    pairs.sort_by_key(|(pair, _)| **pair);
    for (&pair, hypotheses) in pairs {
        if let Some(selection) =
            select_virtual_tile_relation_hypothesis(tiles, pair.0, pair.1, hypotheses, coverages)
        {
            selected.insert(pair, selection.relation);
        } else if let Some(relation) = hypotheses.first() {
            // Preserve one singular measured hypothesis so the final solver can
            // report the pair once as a canonical inversion failure.
            selected.insert(pair, relation.clone());
        }
    }
    selected
}

/// Apply the complete canonical Station_Relation measurement and rejection
/// gate before a prior-seeded Virtual_Tile relation may merge repair clusters.
/// This is intentionally side-effect free: final report counts remain owned by
/// the solver, so provisional filtering cannot double-count a rejection.
pub(crate) fn virtual_tile_provisional_authoritative_candidate(
    tiles: &[ImageInfo],
    left: usize,
    right: usize,
    hypotheses: &[MatchInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> bool {
    select_virtual_tile_relation_hypothesis(tiles, left, right, hypotheses, coverages)
        .is_some_and(|selection| selection.passes)
}

pub(crate) fn virtual_tile_authoritative_pairs(
    relations: &VirtualTileRelationHypotheses,
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> Vec<(usize, usize)> {
    let mut pairs = relations
        .iter()
        .filter_map(|(&(left, right), hypotheses)| {
            virtual_tile_provisional_authoritative_candidate(
                tiles, left, right, hypotheses, coverages,
            )
            .then_some((left, right))
        })
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs
}

/// Topology pairs that remain eligible for an actual prior-free proposal after
/// authoritative relations form their prior-consistent clusters.
pub(crate) fn virtual_tile_prior_free_candidate_pairs(
    station_count: usize,
    authoritative_pairs: &[(usize, usize)],
    station_topology: &topology::StationTopology,
) -> Vec<(usize, usize)> {
    let mut clusters = Dsu::new(station_count);
    for &(left, right) in authoritative_pairs {
        if left < station_count && right < station_count && left != right {
            clusters.union(left, right);
        }
    }
    station_topology
        .candidates
        .iter()
        .filter_map(|candidate| {
            let (left, right) = (candidate.left, candidate.right);
            (left < station_count
                && right < station_count
                && clusters.find(left) != clusters.find(right))
            .then_some((left, right))
        })
        .collect()
}
