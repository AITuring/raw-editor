//! Virtual-tile direct station matching.

use super::*;

/// Bounded bidirectional grid probe of one tile pair in the pixels of the
/// supplied planes. `inlier_threshold_px` is the symmetric residual-model
/// tolerance in those pixels (see [`virtual_tile_pyramid_station_match`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn virtual_tile_direct_station_match(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    inlier_threshold_px: f64,
) -> (
    Option<MatchInfo>,
    stack_report::StationRelationCandidateRecord,
) {
    virtual_tile_direct_station_match_with_structure(
        left,
        right,
        tiles,
        coverages,
        initial_homographies,
        inlier_threshold_px,
        None,
    )
}

/// [`virtual_tile_direct_station_match`] whose structural edge-orientation
/// measurements (prefit diagnostic and fitted gate) read `structure_planes`
/// `(left, right)` instead of the probe planes. The patch and gradient NCC
/// filters always read the probe planes. `None` measures on the probe planes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn virtual_tile_direct_station_match_with_structure(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    inlier_threshold_px: f64,
    structure_planes: Option<(&GrayImage, &GrayImage)>,
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
    let (Some(left_tile), Some(right_tile), Some(left_coverage), Some(right_coverage)) = (
        tiles.get(left),
        tiles.get(right),
        coverages.get(left),
        coverages.get(right),
    ) else {
        return (None, diagnostic);
    };
    diagnostic.failure_stage = "missing_initial_pose".to_string();
    let (Some(left_to_world), Some(right_to_world)) = (
        initial_homographies.get(&left_tile.id),
        initial_homographies.get(&right_tile.id),
    ) else {
        return (None, diagnostic);
    };
    let Some(world_to_right) = right_to_world.try_inverse() else {
        return (None, diagnostic);
    };
    let left_to_right = world_to_right * left_to_world;
    if left_to_right.try_inverse().is_none() {
        return (None, diagnostic);
    }
    let (left_structure, right_structure) =
        structure_planes.unwrap_or((&left_tile.alignment_image, &right_tile.alignment_image));
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
    let grid = virtual_tile_direct_grid_points(overlap_bounds);
    diagnostic.grid_probe_count = grid.len();
    diagnostic.failure_stage = "insufficient_grid_probes".to_string();
    if grid.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let search_radius = virtual_tile_guided_search_radius(
        left_tile,
        right_tile,
        predicted_overlap_area,
        grid.len(),
    )
    .min(VIRTUAL_TILE_DIRECT_MAX_SEARCH_RADIUS_PX)
    .ceil() as i32;
    diagnostic.search_radius_px = f64::from(search_radius);

    let mut by_cell = BTreeMap::<(usize, usize), Vec<(f64, Point2<f64>, Point2<f64>, f64)>>::new();
    let mut prefit_orientations = Vec::new();
    for source in grid {
        if !virtual_tile_direct_patch_support_is_covered(
            left_coverage,
            right_coverage,
            &left_to_right,
            source,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
        ) {
            continue;
        }
        let forward_probe = registration::probe_warped_patch(
            &left_tile.alignment_image,
            &right_tile.alignment_image,
            &left_to_right,
            source,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
            VIRTUAL_TILE_DIRECT_PATCH_GATES,
        );
        if forward_probe.best_correlation.is_some() {
            diagnostic.measurable_patch_count += 1;
        }
        let Some(forward) = forward_probe.matched else {
            continue;
        };
        let Some(structure) = virtual_tile_structural_score_points(
            &left_tile.alignment_image,
            &right_tile.alignment_image,
            left_coverage,
            right_coverage,
            source,
            forward.target,
            &left_to_right,
        ) else {
            continue;
        };
        let prefit_orientation = if structure_planes.is_some() {
            virtual_tile_structural_score_points(
                left_structure,
                right_structure,
                left_coverage,
                right_coverage,
                source,
                forward.target,
                &left_to_right,
            )
            .map(|score| score.orientation_difference_degrees)
        } else {
            Some(structure.orientation_difference_degrees)
        };
        if let Some(orientation) = prefit_orientation {
            prefit_orientations.push(orientation);
        }
        if structure.patch_ncc < VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC
            || structure.gradient_ncc < VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC
        {
            continue;
        }
        let Some(measured_right_to_left) =
            virtual_tile_measured_reverse_transform(&left_to_right, source, forward.target)
        else {
            continue;
        };
        if !virtual_tile_direct_patch_support_is_covered(
            right_coverage,
            left_coverage,
            &measured_right_to_left,
            forward.target,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
        ) {
            continue;
        }
        let reverse_probe = registration::probe_warped_patch(
            &right_tile.alignment_image,
            &left_tile.alignment_image,
            &measured_right_to_left,
            forward.target,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
            VIRTUAL_TILE_DIRECT_PATCH_GATES,
        );
        let Some(reverse) = reverse_probe.matched else {
            continue;
        };
        if !virtual_tile_round_trip_is_consistent(source, reverse.target) {
            continue;
        }
        diagnostic.bidirectional_match_count += 1;
        let cell = virtual_tile_overlap_grid_cell(source, overlap_bounds);
        let rank = structure.patch_ncc + 0.5 * structure.gradient_ncc
            - 0.05 * (reverse.target - source).norm();
        by_cell.entry(cell).or_default().push((
            rank,
            source,
            forward.target,
            structure.orientation_difference_degrees,
        ));
    }
    diagnostic.median_prefit_structural_orientation_difference_degrees =
        median_value(&mut prefit_orientations);
    diagnostic.occupied_grid_cells = by_cell.len();
    diagnostic.failure_stage = "insufficient_measurable_patches".to_string();
    if diagnostic.measurable_patch_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    diagnostic.failure_stage = "insufficient_bidirectional_matches".to_string();
    if diagnostic.bidirectional_match_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    diagnostic.failure_stage = "insufficient_grid_coverage".to_string();
    if diagnostic.occupied_grid_cells < VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS {
        return (None, diagnostic);
    }

    // Rank locally, but do not discard candidates before finding a globally
    // coherent residual model. Repeated artwork can give every cell a strong
    // yet different local peak; truncating each cell first made those modes
    // indistinguishable to RANSAC. The strict measured model selects one mode,
    // then the per-cell quota only limits its already verified inliers.
    for proposals in by_cell.values_mut() {
        proposals.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.y.total_cmp(&right.1.y))
                .then_with(|| left.1.x.total_cmp(&right.1.x))
                .then_with(|| left.2.y.total_cmp(&right.2.y))
                .then_with(|| left.2.x.total_cmp(&right.2.x))
        });
    }
    let maximum_cell_candidates = by_cell.values().map(Vec::len).max().unwrap_or(0);
    let mut all_selected = Vec::with_capacity(diagnostic.bidirectional_match_count);
    for rank in 0..maximum_cell_candidates {
        for (&cell, proposals) in &by_cell {
            if let Some(proposal) = proposals.get(rank) {
                all_selected.push((cell, *proposal));
            }
        }
    }
    let candidate_points = all_selected
        .iter()
        .map(|(_, (_, source, target, _))| (*source, *target))
        .collect::<Vec<_>>();
    // Requirement 6.4 on the pixels both tiles cover, in this plane's pixels.
    let overlap_measure = StationOverlapMeasure::Covered {
        source: left_coverage,
        target: right_coverage,
        factor: 1,
    };
    diagnostic.hull_support = station_relation_spatial_support_with(
        &candidate_points,
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        overlap_measure,
    );
    diagnostic.failure_stage = "insufficient_prefit_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    let measured_residuals = virtual_tile_measured_residuals(&candidate_points, &left_to_right);
    diagnostic.residual_dx_px = residual_distribution(
        measured_residuals
            .iter()
            .map(|(_, residual)| residual.x)
            .collect(),
    );
    diagnostic.residual_dy_px = residual_distribution(
        measured_residuals
            .iter()
            .map(|(_, residual)| residual.y)
            .collect(),
    );
    diagnostic.residual_magnitude_px = residual_distribution(
        measured_residuals
            .iter()
            .map(|(_, residual)| residual.norm())
            .collect(),
    );

    diagnostic.failure_stage = "residual_model_no_fit".to_string();
    // Bound the correction where the probes measured it: the extent of the
    // bidirectional matches inside the predicted overlap. The far corners of a
    // tile that overlaps along one strip would turn any rotation or perspective
    // remainder of the seed into an apparent large correction there.
    let evidence_bounds = candidate_points.iter().fold(
        (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ),
        |(min_x, min_y, max_x, max_y), (source, _)| {
            (
                min_x.min(source.x),
                min_y.min(source.y),
                max_x.max(source.x),
                max_y.max(source.y),
            )
        },
    );
    let model = virtual_tile_fit_measured_residual_model_with_threshold(
        &candidate_points,
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        inlier_threshold_px,
        evidence_bounds,
        VirtualTileResidualSelection::MostInliers,
        overlap_measure,
    );
    diagnostic.max_residual_translation_consensus = model.max_translation_consensus;
    diagnostic.residual_translation = model.translation.clone();
    diagnostic.residual_affine = model.affine.clone();
    diagnostic.residual_projective = model.projective.clone();
    let Some(fit) = model.fit else {
        return (None, diagnostic);
    };
    diagnostic.model_fitted = true;
    diagnostic.residual_model = fit.model.to_string();
    let refined = fit.measured_homography;

    let fitted_indices = fit.inlier_indices.into_iter().collect::<HashSet<_>>();
    let mut fitted_by_cell =
        BTreeMap::<(usize, usize), Vec<(f64, Point2<f64>, Point2<f64>, f64)>>::new();
    let mut fitted_proposals = Vec::with_capacity(fitted_indices.len());
    for (index, (cell, proposal)) in all_selected.into_iter().enumerate() {
        if fitted_indices.contains(&index) {
            fitted_by_cell.entry(cell).or_default().push(proposal);
            fitted_proposals.push(proposal);
        }
    }
    for proposals in fitted_by_cell.values_mut() {
        proposals.truncate(VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA);
    }
    let mut selected = Vec::new();
    for rank in 0..VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA {
        for proposals in fitted_by_cell.values() {
            if let Some(proposal) = proposals.get(rank) {
                selected.push(*proposal);
            }
        }
    }
    // Requirement 6.4 measures the convex hull of the *inliers*. The per-cell
    // quota only limits how strongly a textured cell weighs in the pose graph,
    // so it must not shrink that hull: keep the fitted inliers that are hull
    // vertices on either station plane as well.
    let hull_vertex = |vertices: &[Point2<f64>], point: Point2<f64>| {
        vertices
            .iter()
            .any(|vertex| vertex.x == point.x && vertex.y == point.y)
    };
    let source_hull =
        convex_hull_vertices(fitted_proposals.iter().map(|&(_, source, _, _)| source));
    let target_hull =
        convex_hull_vertices(fitted_proposals.iter().map(|&(_, _, target, _)| target));
    for &proposal in &fitted_proposals {
        let (_, source, target, _) = proposal;
        let already = selected
            .iter()
            .any(|&(_, selected_source, selected_target, _)| {
                selected_source == source && selected_target == target
            });
        if !already && (hull_vertex(&source_hull, source) || hull_vertex(&target_hull, target)) {
            selected.push(proposal);
        }
    }
    diagnostic.failure_stage = "insufficient_matches_after_model_quota".to_string();
    if selected.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let points = selected
        .iter()
        .map(|&(_, source, target, _)| (source, target))
        .collect::<Vec<_>>();
    diagnostic.fitted_inliers = points.len();
    diagnostic.hull_support = station_relation_spatial_support_with(
        &points,
        &refined,
        left_tile.dimensions(),
        right_tile.dimensions(),
        overlap_measure,
    );
    diagnostic.failure_stage = "insufficient_fitted_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    let mut fitted_orientations = selected
        .iter()
        .filter_map(|&(_, source, target, _)| {
            virtual_tile_structural_score_points(
                left_structure,
                right_structure,
                left_coverage,
                right_coverage,
                source,
                target,
                &refined,
            )
            .map(|score| score.orientation_difference_degrees)
        })
        .collect::<Vec<_>>();
    diagnostic.median_fitted_orientation_difference_degrees =
        median_value(&mut fitted_orientations);
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

// ---------------------------------------------------------------------------
// Coarse-to-fine Virtual_Tile station matching (任务 10 checkpoint)
//
// `langyuan-10` measured two independent failures of a native-pixel probe
// centred on the rendering prior (the source-level world poses the tiles are
// rendered with):
//
// * the fine texture of two focus-fused stations is not comparable. Seeded by
//   an independent SIFT relation (median error ~1.6px), 13px and 33px native
//   windows still gave only 58–92 of ~900 bidirectional matches, with a 32–35°
//   structural orientation difference (spurious peaks). The same 13-sample
//   window on 4×4 area averages gave 408–605 with 13–22°, and 0→2 was
//   accepted from the rough prior alone at 1/8 and 1/4;
// * the prior itself can be grossly wrong: a false source-level coarse bridge
//   put one station ~3,400px away at the wrong scale, where no bounded search
//   can reach it. Covered-pixel FAST/BRIEF at analysis resolution found its
//   true neighbour (253–260 RANSAC inliers) without any prior.
//
// The authoritative relation always comes from a bounded pixel probe, run
// coarse to fine on area-averaged planes. A prior-free feature relation only
// seeds that measurement; it never becomes a Station_Relation itself. The
// default anchor-grid fusion keeps placement out of the pixels, so the
// independently measured relation can go through the solver's gates directly.
// Globally warped comparison tiles instead repair the whole cluster's prior,
// render again, and measure around the repaired prior.
// ---------------------------------------------------------------------------
