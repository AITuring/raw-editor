//! Focus-stack scan geometry and stitching order: geometry guards, shifted-structure and
//! low-texture overlap verification, capture-sequence edges, dense and sparse overlap graphs,
//! compact station runs and component bridges.

use super::*;

#[test]
fn narrow_same_focal_scan_boundary_rejects_low_texture_overlap() {
    assert!(!focus_position_recovery_accepts(
        1, true, 1.0, 0.045, 0.12, 120, 0.35, -0.05, -0.05, false,
    ));
    assert!(!focus_position_recovery_accepts(
        2, true, 1.0, 0.045, 0.12, 120, 0.35, -0.05, -0.05, false,
    ));
    assert!(!focus_position_recovery_accepts(
        1, false, 1.0, 0.20, 0.12, 120, 0.35, -0.05, -0.05, false,
    ));
}

#[test]
fn focus_geometry_guard_rejects_incomplete_or_drifting_scan() {
    let images = (0..4)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let stable = (0..images.len())
        .map(|index| (images[index].id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();
    assert!(focus_capture_geometry_passes(&images, 3, 3, 0.64, &stable));
    assert!(!focus_capture_geometry_passes(&images, 2, 3, 0.64, &stable));

    let drifting = HashMap::from([
        (0, Matrix3::identity()),
        (
            1,
            Matrix3::new(0.78, 0.0, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0),
        ),
        (2, Matrix3::identity()),
        (3, Matrix3::identity()),
    ]);
    assert!(!focus_capture_geometry_passes(
        &images, 3, 3, 0.64, &drifting
    ));
}

#[test]
fn focus_station_geometry_guard_does_not_hide_broken_boundaries() {
    assert!(focus_capture_group_geometry_passes(25, 25, 24, 0.50));
    assert!(!focus_capture_group_geometry_passes(24, 25, 24, 0.50));
    assert!(!focus_capture_group_geometry_passes(25, 25, 23, 0.50));
    assert!(!focus_capture_group_geometry_passes(25, 25, 24, 0.20));
}

#[test]
fn shifted_focus_structure_requires_geometry_and_rejects_wrong_overlap() {
    let mut source = test_image(0, "source.nef");
    let mut target = test_image(1, "target.nef");
    let texture = |x: i32, y: i32, seed: u32| {
        let xf = x as f64;
        let yf = y as f64;
        let grain = ((x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663) ^ seed)
            .wrapping_mul(83492791);
        let value = 128.0
            + 45.0 * (xf * 0.029 + yf * 0.013).sin()
            + 35.0 * (xf * 0.017 - yf * 0.037).cos()
            + 25.0 * (xf * 0.053 + yf * 0.041).sin()
            + ((grain >> 24) as f64 - 128.0) * 0.30;
        image::Luma([value.clamp(0.0, 255.0) as u8])
    };
    for image in [&mut source, &mut target] {
        image.width = 4800;
        image.height = 3200;
        image.scale_factor = 4.0;
    }
    source.alignment_image = GrayImage::from_fn(1200, 800, |x, y| texture(x as i32, y as i32, 7));
    target.alignment_image =
        GrayImage::from_fn(1200, 800, |x, y| texture(x as i32 - 160, y as i32 - 40, 91));
    let transform = Matrix3::new(1.0, 0.0, 640.0, 0.0, 1.0, 160.0, 0.0, 0.0, 1.0);
    let points = (0..6)
        .flat_map(|row| {
            (0..8).map(move |col| {
                let p = Point2::new(400.0 + col as f64 * 450.0, 400.0 + row as f64 * 450.0);
                (p, Point2::new(p.x + 640.0, p.y + 160.0))
            })
        })
        .collect::<Vec<_>>();
    assert!(shifted_focus_structure_is_verified(
        &source, &target, &transform, &points, true
    ));
    source.focal_length_35mm = Some(24.0);
    target.focal_length_35mm = Some(50.0);
    assert!(!shifted_focus_structure_is_verified(
        &source, &target, &transform, &points, false
    ));
    target.focal_length_35mm = Some(24.0);
    assert!(!shifted_focus_structure_is_verified(
        &source,
        &target,
        &transform,
        &points[..29],
        false
    ));
    let wrong = Matrix3::new(1.0, 0.0, 960.0, 0.0, 1.0, 160.0, 0.0, 0.0, 1.0);
    let wrong_points = points
        .iter()
        .map(|&(p, _)| (p, Point2::new(p.x + 960.0, p.y + 160.0)))
        .collect::<Vec<_>>();
    assert!(!shifted_focus_structure_is_verified(
        &source,
        &target,
        &wrong,
        &wrong_points,
        true
    ));
    let concentrated = (0..48)
        .map(|i| {
            let p = Point2::new(1000.0 + (i % 8) as f64 * 5.0, 1000.0 + (i / 8) as f64 * 5.0);
            (p, Point2::new(p.x + 640.0, p.y + 160.0))
        })
        .collect::<Vec<_>>();
    assert!(!shifted_focus_structure_is_verified(
        &source,
        &target,
        &transform,
        &concentrated,
        false
    ));
    let fixed_points = points.iter().map(|&(p, _)| (p, p)).collect::<Vec<_>>();
    assert!(!shifted_focus_structure_is_verified(
        &source,
        &source,
        &Matrix3::identity(),
        &fixed_points,
        false
    ));
    let scan_match = MatchInfo {
        homography: transform,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: false,
        points: points.clone(),
        candidate_points: points.clone(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
        native_verified_points: Vec::new(),
    };
    let matches = HashMap::from([((0, 1), scan_match)]);
    let images = [source, target];
    let grouped = focus_station_grouping(&images, &matches);
    assert!(
        grouped.isolated.is_empty(),
        "a verified scan source is not an isolated source"
    );
    assert_eq!(
        grouped.components,
        [vec![0], vec![1]],
        "scan positions must remain separate stations"
    );
    let misleading_priors = HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())]);
    let (groups, _) = focus_capture_groups(&images, &matches, &misleading_priors)
        .expect("connected scan stations");
    assert_eq!(
        groups.len(),
        2,
        "coincident world priors cannot merge distinct scan positions"
    );
}

#[test]
fn low_texture_overlap_requires_boundary_rescue_context() {
    let zhang_haohao_boundary = (0.886, 0.323, 0.134, 1_313);
    assert!(!focus_overlap_is_verified(
        zhang_haohao_boundary,
        13,
        1.887,
        false
    ));
    assert!(focus_overlap_is_verified(
        zhang_haohao_boundary,
        13,
        1.887,
        true
    ));
    assert!(!focus_overlap_is_verified(
        (0.886, 0.323, 0.099, 1_313),
        13,
        1.887,
        true
    ));
    assert!(!focus_overlap_is_verified(
        zhang_haohao_boundary,
        11,
        1.887,
        true
    ));
}

#[test]
fn capture_sequence_edges_outrank_remote_calligraphy_lookalikes() {
    let first = focus_test_image(0, "DSC_1000.NEF");
    let adjacent = focus_test_image(1, "DSC_1001.NEF");
    let remote = focus_test_image(2, "DSC_1200.NEF");
    let local_match = observed_translation_match(180.0, 0.0, 16);
    let remote_match = observed_translation_match(20.0, 0.0, 300);

    assert_eq!(
        focus_graph_capture_sequence_boost(&first, &adjacent, &local_match),
        FOCUS_CAPTURE_SEQUENCE_GRAPH_WEIGHT_BOOST
    );
    assert_eq!(
        focus_graph_capture_sequence_boost(&first, &remote, &remote_match),
        1.0
    );
}

#[test]
fn weak_interior_focus_frame_can_be_skipped_only_across_verified_overlap() {
    let images = vec![
        focus_test_image(0, "DSC_2740.NEF"),
        focus_test_image(1, "DSC_2741.NEF"),
        focus_test_image(2, "DSC_2742.NEF"),
    ];
    let stitched = HashSet::from([0, 2]);
    let verified_bypass = HashMap::from([((0, 2), observed_translation_match(20.0, 0.0, 24))]);

    assert!(focus_unstitched_sources_are_redundant(
        &images,
        &verified_bypass,
        &stitched
    ));
    assert!(!focus_unstitched_sources_are_redundant(
        &images,
        &HashMap::new(),
        &stitched
    ));
}

#[test]
fn focus_stack_geometry_uses_overlap_evidence_and_rendering_uses_capture_order() {
    let images = vec![
        focus_test_image(0, "random-c.jpg"),
        focus_test_image(1, "random-a.jpg"),
        focus_test_image(2, "random-d.jpg"),
        focus_test_image(3, "random-b.jpg"),
    ];
    let matches = HashMap::from([
        ((0, 2), translation_match(100.0, 0.0, 30)),
        ((2, 3), translation_match(100.0, 0.0, 35)),
        ((3, 1), translation_match(100.0, 0.0, 25)),
        // A long overlap can have more descriptors than its immediate
        // neighbor. Motion-aware graph weights must still prefer the
        // local capture path.
        ((0, 3), translation_match(200.0, 0.0, 400)),
        ((0, 1), translation_match(300.0, 0.0, 350)),
    ]);

    let (order, homographies) = build_focus_stack_stitching_order(&images, &matches);

    assert_eq!(order.len(), images.len());
    assert_eq!(
        order.iter().copied().collect::<HashSet<_>>().len(),
        images.len()
    );
    assert!(order.windows(2).all(|pair| {
        matches.contains_key(&(pair[0], pair[1])) || matches.contains_key(&(pair[1], pair[0]))
    }));
    assert_eq!(homographies.len(), images.len());
    assert!(
        homographies
            .values()
            .any(|matrix| *matrix == Matrix3::identity())
    );
}

#[test]
fn dense_focus_overlap_graph_identifies_a_continuous_scan() {
    let images = (0..12)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let mut matches = HashMap::new();
    for source in 0..images.len() {
        for target in source + 1..images.len() {
            matches.insert(
                (source, target),
                observed_translation_match(0.0, (target - source) as f64 * 12.0, 32),
            );
        }
    }

    assert!(focus_match_graph_is_dense_continuous_scan(
        &images, &matches
    ));
}

#[test]
fn sparse_focus_overlap_graph_keeps_evidence_selected_render_order() {
    let images = (0..12)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let matches = (0..images.len() - 1)
        .map(|source| {
            (
                (source, source + 1),
                observed_translation_match(0.0, 12.0, 32),
            )
        })
        .collect::<HashMap<_, _>>();

    assert!(!focus_match_graph_is_dense_continuous_scan(
        &images, &matches
    ));
}

#[test]
fn compact_three_station_focus_run_executes_strict_station_solver() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let capture_numbers = [1000, 1009, 1001, 1008, 1002, 1007, 1003, 1006, 1004, 1005];
    let images = capture_numbers
        .iter()
        .enumerate()
        .map(|(index, number)| focus_textured_test_image(index, &format!("DSC_{number:04}.NEF")))
        .collect::<Vec<_>>();
    assert!(images.len() < SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES);

    // Three focus brackets (4 + 3 + 3 frames) are connected by two
    // multi-layer consensus bridges. The bridges seed station geometry but
    // cannot merge the brackets into one Capture_Station.
    let mut matches = HashMap::new();
    for &(start, end) in &[(0usize, 4usize), (4, 7), (7, 10)] {
        for index in start..end - 1 {
            matches.insert((index, index + 1), observed_translation_match(0.0, 0.0, 36));
        }
    }
    for &(left, right) in &[(3usize, 4usize), (6, 7)] {
        let mut bridge = observed_translation_match(0.0, 0.0, 36);
        bridge.coarse_bridge = true;
        bridge.top_candidate_points = bridge.points.clone();
        matches.insert((left, right), bridge);
    }

    let mut closure_report = stack_report::ClosureReport::default();
    let mut relation_report = stack_report::StationRelationsReport::default();
    let (render_order, solved) = build_focus_stack_stitching_order_with_report(
        &images,
        &matches,
        &mut closure_report,
        &mut relation_report,
        true,
    );

    assert_eq!(solved.len(), images.len());
    assert_eq!(relation_report.accepted.len(), 2);
    assert_eq!(relation_report.connectivity.components, 1);
    assert_eq!(closure_report.participating_constraints, 2);
    assert_eq!(
        closure_report.status,
        stack_report::ClosureStatus::NoConstraints
    );

    // Solving compact station geometry must not opt the ownership renderer
    // into the large-scan filename heuristic.
    let filename_order = focus_filename_order_with_overview_gaps(&images, &matches);
    assert_eq!(render_order.len(), images.len());
    assert_ne!(render_order, filename_order);
}

#[test]
fn compact_single_station_focus_run_skips_inter_station_solver() {
    let images = (0..10)
        .map(|index| focus_textured_test_image(index, &format!("DSC_{:04}.NEF", 2000 + index)))
        .collect::<Vec<_>>();
    let matches = (0..images.len() - 1)
        .map(|index| ((index, index + 1), observed_translation_match(0.0, 0.0, 36)))
        .collect::<HashMap<_, _>>();

    let mut closure_report = stack_report::ClosureReport::default();
    let mut relation_report = stack_report::StationRelationsReport::default();
    let (render_order, solved) = build_focus_stack_stitching_order_with_report(
        &images,
        &matches,
        &mut closure_report,
        &mut relation_report,
        true,
    );

    assert_eq!(render_order.len(), images.len());
    assert_eq!(solved.len(), images.len());
    assert!(relation_report.accepted.is_empty());
    assert_eq!(relation_report.rejected_count, 0);
    assert_eq!(relation_report.connectivity.components, 0);
    assert_eq!(closure_report.status, stack_report::ClosureStatus::NotRun);
    assert_eq!(closure_report.participating_constraints, 0);
}

#[test]
fn large_focus_scan_keeps_capture_order_when_one_weak_frame_is_disconnected() {
    let images = (0..65)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let skipped = 32usize;
    let retained = (0..images.len())
        .filter(|index| *index != skipped)
        .collect::<Vec<_>>();
    let matches = retained
        .windows(2)
        .map(|pair| {
            (
                (pair[0].min(pair[1]), pair[0].max(pair[1])),
                translation_match(20.0, 0.0, 32),
            )
        })
        .collect::<HashMap<_, _>>();

    let (order, _) = build_focus_stack_stitching_order(&images, &matches);

    assert_eq!(order, retained);
}

#[test]
fn focus_component_bridge_candidates_keep_both_sequence_boundaries() {
    let images = (0..12)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let component = (0..images.len()).collect::<Vec<_>>();

    let candidates = focus_component_bridge_candidates(&component, &images);

    assert!(candidates.contains(&0));
    assert!(candidates.contains(&1));
    assert!(candidates.contains(&10));
    assert!(candidates.contains(&11));
}
