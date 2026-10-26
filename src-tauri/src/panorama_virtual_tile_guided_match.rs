//! Virtual-tile descriptor-guided matching.

use super::*;

#[allow(dead_code)]
pub(crate) fn virtual_tile_descriptor_guided_station_match(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
) -> (
    Option<MatchInfo>,
    stack_report::StationRelationCandidateRecord,
) {
    let mut diagnostic = stack_report::StationRelationCandidateRecord {
        left,
        right,
        left_tile: Default::default(),
        right_tile: Default::default(),
        predicted_overlap_area_px: 0.0,
        model_fitted: false,
        evidence_kind: stack_report::StationRelationEvidenceKind::VirtualTile,
        grid_probe_count: 0,
        measurable_patch_count: 0,
        bidirectional_match_count: 0,
        occupied_grid_cells: 0,
        search_radius_px: 0.0,
        residual_dx_px: Default::default(),
        residual_dy_px: Default::default(),
        residual_magnitude_px: Default::default(),
        max_residual_translation_consensus: 0,
        residual_translation: Default::default(),
        residual_affine: Default::default(),
        residual_projective: Default::default(),
        residual_model: "not_run".to_string(),
        fitted_inliers: 0,
        hull_support: 0.0,
        median_prefit_structural_orientation_difference_degrees: None,
        median_fitted_orientation_difference_degrees: None,
        failure_stage: "missing_input".to_string(),
        pyramid_factors: Vec::new(),
        measurement_factor: 1,
        seed: VIRTUAL_TILE_SEED_RENDERING_PRIOR.to_string(),
        prior_free_inliers: 0,
        after_prior_repair: false,
        hypotheses: Vec::new(),
    };
    let Some(left_tile) = tiles.get(left) else {
        return (None, diagnostic);
    };
    let Some(right_tile) = tiles.get(right) else {
        return (None, diagnostic);
    };
    let Some(left_coverage) = coverages.get(left) else {
        return (None, diagnostic);
    };
    let Some(right_coverage) = coverages.get(right) else {
        return (None, diagnostic);
    };
    diagnostic.failure_stage = "missing_initial_pose".to_string();
    let Some(left_to_world) = initial_homographies.get(&left_tile.id) else {
        return (None, diagnostic);
    };
    let Some(right_to_world) = initial_homographies.get(&right_tile.id) else {
        return (None, diagnostic);
    };
    let Some(world_to_right) = right_to_world.try_inverse() else {
        return (None, diagnostic);
    };
    let left_to_right = world_to_right * left_to_world;
    let Some(right_to_left) = left_to_right.try_inverse() else {
        return (None, diagnostic);
    };
    diagnostic.failure_stage = "no_predicted_overlap".to_string();
    let Some(overlap_bounds) = virtual_tile_predicted_overlap_bounds(
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
    ) else {
        return (None, diagnostic);
    };
    let Some(predicted_overlap_area) = station_relation_overlap_area(
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        true,
    ) else {
        return (None, diagnostic);
    };
    diagnostic.predicted_overlap_area_px = predicted_overlap_area;

    let left_eligible = left_tile
        .features
        .iter()
        .map(|feature| {
            let point = Point2::new(f64::from(feature.keypoint.x), f64::from(feature.keypoint.y));
            virtual_tile_point_is_covered(left_coverage, point)
                && transformed_point(&left_to_right, point)
                    .is_some_and(|mapped| virtual_tile_point_is_covered(right_coverage, mapped))
        })
        .collect::<Vec<_>>();
    let right_eligible = right_tile
        .features
        .iter()
        .map(|feature| {
            let point = Point2::new(f64::from(feature.keypoint.x), f64::from(feature.keypoint.y));
            virtual_tile_point_is_covered(right_coverage, point)
                && transformed_point(&right_to_left, point)
                    .is_some_and(|mapped| virtual_tile_point_is_covered(left_coverage, mapped))
        })
        .collect::<Vec<_>>();
    let instrument_count = left_eligible
        .iter()
        .filter(|&&eligible| eligible)
        .count()
        .min(right_eligible.iter().filter(|&&eligible| eligible).count());
    diagnostic.grid_probe_count = instrument_count;
    diagnostic.failure_stage = "insufficient_jointly_covered_probes".to_string();
    if instrument_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }

    let search_radius = virtual_tile_guided_search_radius(
        left_tile,
        right_tile,
        predicted_overlap_area,
        instrument_count,
    );
    diagnostic.search_radius_px = search_radius;
    let left_buckets =
        virtual_tile_feature_buckets(&left_tile.features, &left_eligible, search_radius);
    let right_buckets =
        virtual_tile_feature_buckets(&right_tile.features, &right_eligible, search_radius);
    let mut by_cell = BTreeMap::<(usize, usize), Vec<(f64, usize, usize, f64)>>::new();
    let mut orientation_observations = Vec::new();
    for (left_index, &eligible) in left_eligible.iter().enumerate() {
        if !eligible {
            continue;
        }
        let Some(forward) = virtual_tile_best_guided_feature(
            left_index,
            &left_tile.features,
            &right_tile.features,
            &right_buckets,
            &left_tile.alignment_image,
            &right_tile.alignment_image,
            left_coverage,
            right_coverage,
            &left_to_right,
            search_radius,
            &mut orientation_observations,
        ) else {
            continue;
        };
        let mut reverse_orientations = Vec::new();
        let reverse = virtual_tile_best_guided_feature(
            forward.target_index,
            &right_tile.features,
            &left_tile.features,
            &left_buckets,
            &right_tile.alignment_image,
            &left_tile.alignment_image,
            right_coverage,
            left_coverage,
            &right_to_left,
            search_radius,
            &mut reverse_orientations,
        );
        orientation_observations.extend(reverse_orientations);
        if reverse.is_none_or(|reverse| reverse.target_index != left_index) {
            continue;
        }
        let source = left_tile.features[left_index].keypoint;
        let cell = virtual_tile_overlap_grid_cell(
            Point2::new(f64::from(source.x), f64::from(source.y)),
            overlap_bounds,
        );
        by_cell.entry(cell).or_default().push((
            forward.rank,
            left_index,
            forward.target_index,
            forward.orientation_difference_degrees,
        ));
    }
    diagnostic.median_prefit_structural_orientation_difference_degrees =
        median_value(&mut orientation_observations);
    diagnostic.bidirectional_match_count = by_cell.values().map(Vec::len).sum();
    diagnostic.occupied_grid_cells = by_cell.len();
    diagnostic.failure_stage = "insufficient_bidirectional_match_count".to_string();
    if diagnostic.bidirectional_match_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    diagnostic.failure_stage = "insufficient_grid_coverage".to_string();
    if diagnostic.occupied_grid_cells < VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS {
        return (None, diagnostic);
    }

    for proposals in by_cell.values_mut() {
        proposals.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| {
                    let left_source = left_tile.features[left.1].keypoint;
                    let right_source = left_tile.features[right.1].keypoint;
                    (left_source.y, left_source.x).cmp(&(right_source.y, right_source.x))
                })
                .then_with(|| {
                    let left_target = right_tile.features[left.2].keypoint;
                    let right_target = right_tile.features[right.2].keypoint;
                    (left_target.y, left_target.x).cmp(&(right_target.y, right_target.x))
                })
        });
        proposals.truncate(VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA);
    }
    // Round-robin by quota rank gives RANSAC one observation from every cell
    // before any repeated structure receives its second slot.
    let mut selected = Vec::new();
    for rank in 0..VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA {
        for proposals in by_cell.values() {
            if let Some(proposal) = proposals.get(rank) {
                selected.push(*proposal);
            }
        }
    }
    diagnostic.failure_stage = "insufficient_matches_after_cell_quota".to_string();
    if selected.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let candidate_points = selected
        .iter()
        .map(|&(_, left_index, right_index, _)| {
            let source = left_tile.features[left_index].keypoint;
            let target = right_tile.features[right_index].keypoint;
            (
                Point2::new(f64::from(source.x), f64::from(source.y)),
                Point2::new(f64::from(target.x), f64::from(target.y)),
            )
        })
        .collect::<Vec<_>>();
    diagnostic.hull_support = station_relation_spatial_support(
        &candidate_points,
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
    );
    diagnostic.failure_stage = "insufficient_prefit_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    // The initial pose ends here. RANSAC and its refit receive only measured
    // fused-pixel correspondences, so the prior can neither become an inlier
    // nor contribute acceptance evidence.
    diagnostic.failure_stage = "ransac_no_fit".to_string();
    let Some((seed_fit, seed_inliers)) =
        processing::find_homography_ransac_points_stable_with_min_inliers(
            &candidate_points,
            STATION_RELATION_MAX_MEDIAN_ERROR_PX,
            STATION_RELATION_MIN_INLIERS,
        )
    else {
        return (None, diagnostic);
    };
    diagnostic.model_fitted = true;
    let seed_points = seed_inliers
        .iter()
        .map(|&index| candidate_points[index])
        .collect::<Vec<_>>();
    let refined = processing::compute_homography(&seed_points).unwrap_or(seed_fit);
    let refined_inliers = symmetric_inlier_indices(
        &refined,
        &candidate_points,
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    diagnostic.fitted_inliers = refined_inliers.len();
    diagnostic.failure_stage = "insufficient_fitted_inliers".to_string();
    if refined_inliers.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let points = refined_inliers
        .iter()
        .map(|&index| candidate_points[index])
        .collect::<Vec<_>>();
    diagnostic.hull_support = station_relation_spatial_support(
        &points,
        &refined,
        left_tile.dimensions(),
        right_tile.dimensions(),
    );
    diagnostic.failure_stage = "insufficient_fitted_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    // Orientation becomes an acceptance gate only after fused-pixel
    // correspondences have produced a measured homography. Re-measure every
    // accepted inlier patch with the fitted local Jacobian; the rough prior is
    // deliberately absent from this decision.
    let mut fitted_orientation_observations = refined_inliers
        .iter()
        .filter_map(|&index| {
            let &(_, left_index, right_index, _) = selected.get(index)?;
            virtual_tile_structural_score(
                &left_tile.alignment_image,
                &right_tile.alignment_image,
                left_coverage,
                right_coverage,
                left_tile.features[left_index].keypoint,
                right_tile.features[right_index].keypoint,
                &refined,
            )
            .map(|score| score.orientation_difference_degrees)
        })
        .collect::<Vec<_>>();
    diagnostic.median_fitted_orientation_difference_degrees =
        median_value(&mut fitted_orientation_observations);
    diagnostic.failure_stage = "fitted_orientation_mismatch".to_string();
    if diagnostic
        .median_fitted_orientation_difference_degrees
        .is_some_and(|difference| difference > VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES)
    {
        return (None, diagnostic);
    }

    diagnostic.failure_stage = "fitted".to_string();
    (
        Some(MatchInfo {
            homography: refined,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: false,
            points,
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            // Key-direction Virtual_Tile fit; see `MatchInfo::canonical_homography`.
            canonical_homography: None,
        }),
        diagnostic,
    )
}
