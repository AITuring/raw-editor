//! Bounded direct probes between Virtual_Tile stations.
//!
//! Probes a grid of covered patches around the prior, checks round trips and
//! patch support, and fits the measured relation.

use crate::panorama_utils::registration;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stitching;
use image::GrayImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use super::{
    ImageInfo, MatchInfo, STATION_RELATION_MIN_INLIERS, STATION_RELATION_MIN_SPATIAL_SUPPORT,
    StationOverlapMeasure, VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES,
    VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC, VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS,
    VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC, VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN,
    VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA, VIRTUAL_TILE_SEED_RENDERING_PRIOR,
    VirtualTileResidualSelection, convex_hull_vertices, median_value, residual_distribution,
    station_relation_overlap_area, station_relation_spatial_support_with, transformed_point,
    virtual_tile_fit_measured_residual_model_with_threshold, virtual_tile_guided_search_radius,
    virtual_tile_local_jacobian, virtual_tile_measured_residuals, virtual_tile_overlap_grid_cell,
    virtual_tile_predicted_overlap_bounds, virtual_tile_structural_score_points,
};

pub(super) const VIRTUAL_TILE_DIRECT_PATCH_RADIUS: i32 = 6;

pub(super) const VIRTUAL_TILE_DIRECT_GRID_SPACING_PX: f64 = 24.0;

pub(super) const VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES: usize = 1_024;

pub(super) const VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS: usize = 64;

pub(super) const VIRTUAL_TILE_DIRECT_MAX_SEARCH_RADIUS_PX: f64 = 16.0;

pub(super) const VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX: f64 = 1.0;

pub(super) const VIRTUAL_TILE_DIRECT_PATCH_GATES: registration::PatchMatchGates =
    registration::PatchMatchGates {
        min_correlation: VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC,
        min_peak_margin: VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN,
        min_source_variance: 2.0,
        min_target_variance: 1.0,
    };

pub(super) fn virtual_tile_direct_grid_points(bounds: (f64, f64, f64, f64)) -> Vec<Point2<f64>> {
    let (min_x, min_y, max_x, max_y) = bounds;
    let width = max_x - min_x;
    let height = max_y - min_y;
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return Vec::new();
    }
    let area = width * height;
    let desired = (area
        / (VIRTUAL_TILE_DIRECT_GRID_SPACING_PX * VIRTUAL_TILE_DIRECT_GRID_SPACING_PX))
        .ceil()
        .clamp(
            STATION_RELATION_MIN_INLIERS as f64,
            VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES as f64,
        ) as usize;
    let aspect = width / height;
    let mut columns = ((desired as f64 * aspect).sqrt().round() as usize)
        .clamp(1, VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS);
    let mut rows = desired
        .div_ceil(columns)
        .clamp(1, VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS);
    while columns * rows > VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES {
        if columns >= rows && columns > 1 {
            columns -= 1;
        } else if rows > 1 {
            rows -= 1;
        } else {
            break;
        }
    }
    let mut points = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        let y = min_y + (row as f64 + 0.5) * height / rows as f64;
        for column in 0..columns {
            let x = min_x + (column as f64 + 0.5) * width / columns as f64;
            points.push(Point2::new(x, y));
        }
    }
    points
}

pub(super) fn virtual_tile_coverage_contains_rectangle(
    coverage: &stitching::CoverageMask,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> bool {
    let (width, height) = coverage.dimensions();
    if ![min_x, min_y, max_x, max_y]
        .iter()
        .all(|value| value.is_finite())
        || min_x < 0.0
        || min_y < 0.0
        || max_x >= f64::from(width.saturating_sub(1))
        || max_y >= f64::from(height.saturating_sub(1))
        || max_x < min_x
        || max_y < min_y
    {
        return false;
    }
    let start_x = min_x.floor() as u32;
    let end_x = max_x.ceil() as u32;
    let start_y = min_y.floor() as u32;
    let end_y = max_y.ceil() as u32;
    (start_y..=end_y).all(|y| (start_x..=end_x).all(|x| coverage.is_covered(x, y)))
}

pub(super) fn virtual_tile_direct_patch_support_is_covered(
    source_coverage: &stitching::CoverageMask,
    target_coverage: &stitching::CoverageMask,
    source_to_target: &Matrix3<f64>,
    center: Point2<f64>,
    patch_radius: i32,
    search_radius: i32,
) -> bool {
    let Some(jacobian) = virtual_tile_local_jacobian(source_to_target, center) else {
        return false;
    };
    let scale = (jacobian.column(0).norm() * jacobian.column(1).norm()).sqrt();
    if !(0.15..6.0).contains(&scale) {
        return false;
    }
    let spacing = scale.recip().max(1.0);
    // Include one pixel for gradient sampling and another for bilinear access.
    let source_extent = f64::from(patch_radius) * spacing + 2.0;
    if !virtual_tile_coverage_contains_rectangle(
        source_coverage,
        center.x - source_extent,
        center.y - source_extent,
        center.x + source_extent,
        center.y + source_extent,
    ) {
        return false;
    }
    let corners = [
        Point2::new(center.x - source_extent, center.y - source_extent),
        Point2::new(center.x + source_extent, center.y - source_extent),
        Point2::new(center.x + source_extent, center.y + source_extent),
        Point2::new(center.x - source_extent, center.y + source_extent),
    ];
    let Some(projected) = corners
        .into_iter()
        .map(|point| transformed_point(source_to_target, point))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let padding = f64::from(search_radius) + 2.0;
    let min_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min)
        - padding;
    let max_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max)
        + padding;
    let min_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min)
        - padding;
    let max_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max)
        + padding;
    virtual_tile_coverage_contains_rectangle(target_coverage, min_x, min_y, max_x, max_y)
}

pub(super) fn virtual_tile_round_trip_is_consistent(
    source: Point2<f64>,
    returned: Point2<f64>,
) -> bool {
    let distance = (returned - source).norm();
    distance.is_finite() && distance <= VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX
}

/// Invert the *measured* local forward map, not merely the rough station prior.
///
/// `probe_warped_patch` returns `target = H_initial(source) + residual` in the
/// target tile's coordinates. The corresponding local forward homography is
/// therefore `T(residual) * H_initial`; its inverse is the only transform whose
/// source/target convention maps that measured target back to the original
/// source. Using `H_initial.inverse()` here incorrectly asks the reverse probe
/// to rediscover the already measured residual and made valid pairs depend on
/// which station happened to be the source (任务 10 checkpoint).
pub(super) fn virtual_tile_measured_reverse_transform(
    initial_source_to_target: &Matrix3<f64>,
    source: Point2<f64>,
    measured_target: Point2<f64>,
) -> Option<Matrix3<f64>> {
    let predicted_target = transformed_point(initial_source_to_target, source)?;
    let residual = measured_target - predicted_target;
    if !residual.iter().all(|value| value.is_finite()) {
        return None;
    }
    let residual_translation =
        Matrix3::new(1.0, 0.0, residual.x, 0.0, 1.0, residual.y, 0.0, 0.0, 1.0);
    (residual_translation * initial_source_to_target).try_inverse()
}

/// Bounded bidirectional grid probe of one tile pair in the pixels of the
/// supplied planes. `inlier_threshold_px` is the symmetric residual-model
/// tolerance in those pixels (see [`virtual_tile_pyramid_station_match`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_direct_station_match(
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
pub(super) fn virtual_tile_direct_station_match_with_structure(
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
        global_proposals: Vec::new(),
        after_prior_repair: false,
        hypotheses: Vec::new(),
    };
    // Keep the full-resolution Virtual_Tile station probe available for normal
    // builds, while allowing test fixtures to isolate the descriptor/coarse
    // geometry path. The helper is hard-wired to `false` outside `cfg(test)`,
    // so a production environment variable cannot weaken station evidence.
    if registration::native_patch_refinement_disabled_for_tests() {
        diagnostic.failure_stage = "native_patch_refinement_disabled".to_string();
        return (None, diagnostic);
    }
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
            native_verified_points: Vec::new(),
        }),
        diagnostic,
    )
}
