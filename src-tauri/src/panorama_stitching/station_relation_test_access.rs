use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MeasurementsView {
    pub(crate) inliers: usize,
    pub(crate) median_error_px: f64,
    pub(crate) scale_ratio: f64,
    pub(crate) spatial_support: f64,
    pub(crate) low_frequency_mean_relative_difference: f64,
    pub(crate) edge_strength_ratio: f64,
    pub(crate) median_edge_orientation_difference_degrees: f64,
    pub(crate) preserves_convex_orientation: bool,
}

pub(crate) fn covered_features(
    pixels: &Rgb32FImage,
    coverage: &stitching::CoverageMask,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> (GrayImage, Vec<Feature>) {
    virtual_tile_covered_features(pixels, coverage, brief_pairs)
}

pub(crate) fn station_matches(tiles: &[ImageInfo]) -> HashMap<(usize, usize), MatchInfo> {
    all_image_pairs(tiles.len())
        .into_iter()
        .filter_map(|(left, right)| {
            virtual_tile_unseeded_station_match(left, right, tiles)
                .map(|relation| ((left, right), relation))
        })
        .collect()
}

pub(crate) fn guided_station_matches(
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
) -> HashMap<(usize, usize), MatchInfo> {
    virtual_tile_station_matches(tiles, coverages, initial_homographies, station_topology)
}

pub(crate) fn single_source_discard(
    independent_support: usize,
    left_station: usize,
    right_station: usize,
) -> Option<serde_json::Value> {
    single_layer_station_evidence_discard(independent_support, left_station, right_station)
}

pub(crate) fn rejection_reasons(measurements: MeasurementsView) -> Vec<&'static str> {
    station_relation_rejection_reasons(&StationRelationMeasurements {
        inliers: measurements.inliers,
        median_error_px: measurements.median_error_px,
        scale_ratio: measurements.scale_ratio,
        spatial_support: measurements.spatial_support,
        low_frequency_mean_relative_difference: measurements.low_frequency_mean_relative_difference,
        edge_strength_ratio: measurements.edge_strength_ratio,
        median_edge_orientation_difference_degrees: measurements
            .median_edge_orientation_difference_degrees,
        preserves_convex_orientation: measurements.preserves_convex_orientation,
    })
}

fn production_measurements(measurements: MeasurementsView) -> StationRelationMeasurements {
    StationRelationMeasurements {
        inliers: measurements.inliers,
        median_error_px: measurements.median_error_px,
        scale_ratio: measurements.scale_ratio,
        spatial_support: measurements.spatial_support,
        low_frequency_mean_relative_difference: measurements.low_frequency_mean_relative_difference,
        edge_strength_ratio: measurements.edge_strength_ratio,
        median_edge_orientation_difference_degrees: measurements
            .median_edge_orientation_difference_degrees,
        preserves_convex_orientation: measurements.preserves_convex_orientation,
    }
}

pub(crate) struct RelationDecisionBatchView {
    pub(crate) report: stack_report::StationRelationsReport,
    pub(crate) pose_writes: Vec<(usize, usize)>,
}

/// Drive the exact production reject/report gate for a batch of candidate
/// adjacencies. `pose_writes` models the immediately following pose-graph
/// insertion and therefore contains accepted candidates only.
pub(crate) fn relation_decision_batch(
    candidates: &[(usize, usize, MeasurementsView)],
) -> RelationDecisionBatchView {
    let mut report = stack_report::StationRelationsReport::default();
    let mut pose_writes = Vec::new();
    for &(left, right, measurements) in candidates {
        if station_relation_is_accepted_with_report(
            left,
            right,
            0.0,
            &production_measurements(measurements),
            &mut report,
        ) {
            pose_writes.push((left, right));
        }
    }
    RelationDecisionBatchView {
        report,
        pose_writes,
    }
}

fn synthetic_image(id: usize, path: String, dimensions: (u32, u32)) -> ImageInfo {
    ImageInfo {
        id,
        filename: path,
        width: dimensions.0,
        height: dimensions.1,
        alignment_image: GrayImage::new(dimensions.0.max(1), dimensions.1.max(1)),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: None,
        source_footprint: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

fn singleton_station_inputs(
    dimensions: (u32, u32),
    station_count: usize,
) -> (Vec<ImageInfo>, Vec<FocusCaptureGroup>) {
    let images = (0..station_count)
        .map(|station| {
            synthetic_image(
                station,
                format!("/synthetic/property/station-{station}.raw"),
                dimensions,
            )
        })
        .collect::<Vec<_>>();
    let groups = (0..station_count)
        .map(|station| FocusCaptureGroup {
            members: vec![station],
            anchor: station,
            local_to_anchor: HashMap::from([(station, Matrix3::identity())]),
        })
        .collect::<Vec<_>>();
    (images, groups)
}

pub(crate) struct PoseCandidateView {
    pub(crate) selected: Matrix3<f64>,
    pub(crate) minimum_local_scale: Option<f64>,
    pub(crate) maximum_local_scale: Option<f64>,
    pub(crate) rejection_reason: Option<&'static str>,
}

/// Exercise the production Local_Scale publication gate for one station.
pub(crate) fn select_pose_candidate(
    candidate: Matrix3<f64>,
    spanning_tree_seed: Matrix3<f64>,
    dimensions: (u32, u32),
) -> PoseCandidateView {
    let (images, groups) = singleton_station_inputs(dimensions, 1);
    let accepted = station_pose_local_scale_range(&candidate, dimensions)
        .is_some_and(|(minimum, maximum)| maximum / minimum <= STATION_POSE_MAX_LOCAL_SCALE_RATIO);
    let selected = select_station_pose_candidate(
        &images,
        &groups,
        &[candidate],
        &[spanning_tree_seed],
        "property_31",
    )[0];
    let range = station_pose_local_scale_range(&candidate, dimensions);
    PoseCandidateView {
        selected,
        minimum_local_scale: range.map(|range| range.0),
        maximum_local_scale: range.map(|range| range.1),
        rejection_reason: (!accepted).then_some(degradation::STATION_POSE_LOCAL_SCALE_EXCEEDED),
    }
}

pub(crate) struct StationGraphOutcomeView {
    pub(crate) report: stack_report::StationRelationsReport,
    pub(crate) rejection_reason: Option<&'static str>,
    pub(crate) rejection_detail: Option<serde_json::Value>,
    pub(crate) user_error: Option<String>,
}

/// Run the production component canonicalisation/report boundary on an
/// arbitrary station graph supplied by a property test.
pub(crate) fn station_graph_outcome(
    station_member_paths: &[Vec<String>],
    relation_pairs: &[(usize, usize)],
) -> StationGraphOutcomeView {
    let mut images = Vec::new();
    let mut groups = Vec::new();
    for members in station_member_paths {
        let first = images.len();
        let mut member_indices = Vec::new();
        for path in members {
            let image_index = images.len();
            images.push(synthetic_image(image_index, path.clone(), (64, 64)));
            member_indices.push(image_index);
        }
        groups.push(FocusCaptureGroup {
            members: member_indices,
            anchor: first,
            local_to_anchor: HashMap::from([(first, Matrix3::identity())]),
        });
    }
    let mut report = stack_report::StationRelationsReport::default();
    let rejection_detail =
        station_graph_connectivity_rejection(&images, &groups, relation_pairs, &mut report);
    let rejection_reason = rejection_detail
        .as_ref()
        .map(|_| degradation::GEOMETRY_DISCONNECTED);
    let user_error = rejection_detail
        .as_ref()
        .map(|_| geometry_disconnected_user_error(report.connectivity.components));
    StationGraphOutcomeView {
        report,
        rejection_reason,
        rejection_detail,
        user_error,
    }
}

/// Production Kruskal selection, returned as canonical station pairs so
/// relation input order is not observable to the property test.
pub(crate) fn maximum_inlier_tree(
    station_count: usize,
    rows: &[u32],
    columns: &[u32],
    relations: &[(usize, usize, usize, f64)],
) -> Vec<(usize, usize)> {
    let station_topology = topology::StationTopology {
        row: rows.to_vec(),
        column: columns.to_vec(),
        ..topology::StationTopology::default()
    };
    let production_relations = relations
        .iter()
        .map(|&(left, right, inliers, _score)| (left, right, inliers))
        .collect::<Vec<_>>();
    maximum_inlier_spanning_tree_indices(station_count, &station_topology, &production_relations)
        .into_iter()
        .map(|index| {
            let (left, right, _, _) = relations[index];
            (left.min(right), left.max(right))
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ClosureRelationView {
    pub(crate) score: f64,
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) left_to_right: Matrix3<f64>,
    pub(crate) independent_support: usize,
    pub(crate) median_error_px: f64,
}

pub(crate) struct ClosureRunView {
    pub(crate) poses: Vec<Matrix3<f64>>,
    pub(crate) report: stack_report::ClosureReport,
}

fn production_group_edges(
    relations: &[ClosureRelationView],
) -> Vec<(f64, usize, usize, Matrix3<f64>, usize, f64)> {
    relations
        .iter()
        .map(|relation| {
            (
                relation.score,
                relation.left,
                relation.right,
                relation.left_to_right,
                relation.independent_support,
                relation.median_error_px,
            )
        })
        .collect()
}

/// Exact production M-estimator access for Property 35.
pub(crate) fn closure_weight(residual_px: f64) -> f64 {
    closure_residual_weight(residual_px)
}

/// Exact production stopping predicate access for Property 37.
pub(crate) fn closure_converged(previous: f64, current: f64) -> bool {
    closure_relative_residual_decrease_converged(previous, current)
}

pub(crate) struct ClosureSolveView {
    pub(crate) corrections: Vec<[f64; 2]>,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
}

/// Exercise the production robust translation solver without image setup.
pub(crate) fn closure_solve(
    station_count: usize,
    reference_station: usize,
    constraints: &[(usize, usize, [f64; 2], f64)],
) -> Option<ClosureSolveView> {
    solve_robust_closure_translations(station_count, reference_station, constraints).map(|solve| {
        ClosureSolveView {
            corrections: solve.corrections,
            iterations: solve.iterations,
            converged: solve.converged,
        }
    })
}

pub(crate) struct ClosureClampView {
    pub(crate) corrections: Vec<[f64; 2]>,
    pub(crate) maximum_corner_displacement_px: f64,
    pub(crate) clamped_stations: usize,
}

/// Exercise the production four-corner correction clamp for Property 36.
pub(crate) fn clamp_closure_corrections(
    dimensions: (u32, u32),
    base_poses: &[Matrix3<f64>],
    relations: &[ClosureRelationView],
    corrections: &[[f64; 2]],
) -> Option<ClosureClampView> {
    let (images, groups) = singleton_station_inputs(dimensions, base_poses.len());
    let group_edges = production_group_edges(relations);
    let mut corrections = corrections.to_vec();
    let (maximum_corner_displacement_px, clamped_stations) = clamp_station_translation_corrections(
        &images,
        &groups,
        &group_edges,
        base_poses,
        &mut corrections,
    )?;
    Some(ClosureClampView {
        corrections,
        maximum_corner_displacement_px,
        clamped_stations,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClosureAcceptanceView {
    Accepted,
    NoConstraints,
    UnreliableResidual,
    UnreliableIterations,
    UnreliablePairP95(usize, usize),
}

/// Exact production acceptance gate access for Properties 38 and 39.
pub(crate) fn closure_acceptance_view(
    station_count: usize,
    accepted_relation_count: usize,
    converged: bool,
    iterations: usize,
    residual_median_px: f64,
    direct_pairs: &[(usize, usize, f64)],
) -> ClosureAcceptanceView {
    let direct_pairs = direct_pairs
        .iter()
        .map(|&(left, right, p95_px)| ClosureDirectPairP95 {
            left,
            right,
            p95_px,
        })
        .collect::<Vec<_>>();
    match closure_acceptance(
        station_count,
        accepted_relation_count,
        converged,
        iterations,
        residual_median_px,
        &direct_pairs,
    ) {
        ClosureAcceptance::Accepted => ClosureAcceptanceView::Accepted,
        ClosureAcceptance::NoConstraints => ClosureAcceptanceView::NoConstraints,
        ClosureAcceptance::UnreliableResidual => ClosureAcceptanceView::UnreliableResidual,
        ClosureAcceptance::UnreliableIterations => ClosureAcceptanceView::UnreliableIterations,
        ClosureAcceptance::UnreliablePairP95(pair) => {
            ClosureAcceptanceView::UnreliablePairP95(pair.left, pair.right)
        }
    }
}

/// Run the actual closure/report boundary with one singleton source per
/// station; no image payload is decoded or mocked.
pub(crate) fn closure_run(
    dimensions: (u32, u32),
    rows: &[u32],
    columns: &[u32],
    base_poses: &[Matrix3<f64>],
    relations: &[ClosureRelationView],
) -> ClosureRunView {
    let (images, groups) = singleton_station_inputs(dimensions, base_poses.len());
    let group_edges = production_group_edges(relations);
    let station_topology = topology::StationTopology {
        row: rows.to_vec(),
        column: columns.to_vec(),
        ..topology::StationTopology::default()
    };
    let mut report = stack_report::ClosureReport::default();
    let poses = refine_focus_group_projective_tree_poses_with_report(
        &images,
        &groups,
        &group_edges,
        base_poses,
        0,
        f64::from(dimensions.0.max(dimensions.1)),
        &station_topology,
        2,
        &mut report,
    )
    .expect("valid property closure fixture returns a pose vector");
    ClosureRunView { poses, report }
}
