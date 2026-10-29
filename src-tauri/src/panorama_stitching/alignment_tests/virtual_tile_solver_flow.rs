//! The Virtual_Tile station solver end to end: key-direction relations in either content
//! direction, unmeasured no-fit candidates, the default two-stage flow, true disconnection, a
//! single tile and two-layer station evidence.

use super::*;

/// Solver coverages of tiles whose single analysis plane is covered
/// everywhere and doubles as their finest match plane.
fn fully_covered_solver_coverages<'a>(
    tiles: &[ImageInfo],
    masks: &'a [stitching::CoverageMask],
) -> VirtualTileSolverCoverages<'a> {
    VirtualTileSolverCoverages {
        quality: masks,
        overlap: masks,
        overlap_factor: tiles
            .first()
            .map_or(1, |tile| tile.scale_factor.round().max(1.0) as u32),
    }
}
/// Relation-quality Coverage_Masks of tiles whose analysis planes are
/// covered everywhere.
fn fully_covered_quality_coverages(tiles: &[ImageInfo]) -> Vec<stitching::CoverageMask> {
    tiles
        .iter()
        .map(|tile| {
            let (width, height) = tile.alignment_image.dimensions();
            stitching::CoverageMask::from_bytes(
                width,
                height,
                vec![u8::MAX; width as usize * height as usize],
            )
            .expect("a fully covered mask matches its plane")
        })
        .collect()
}

#[test]
fn virtual_tile_solver_reads_a_key_direction_relation_in_either_content_direction() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    // Tile 1 shows tile 0's content shifted by `shift`: left p ↔ right p − shift.
    let (side, shift) = (1_000u32, nalgebra::Vector2::new(40.0, 6.0));
    let texture = |x: f64, y: f64| {
        (126.0
            + 47.0 * (x * 0.091 + y * 0.037).sin()
            + 38.0 * (x * 0.027 - y * 0.113).cos()
            + 24.0 * (x * 0.173 + y * 0.149).sin())
        .clamp(0.0, 255.0)
        .round() as u8
    };
    let mut tiles = synthetic_virtual_tiles(2);
    for (tile, offset) in tiles.iter_mut().zip([nalgebra::Vector2::zeros(), shift]) {
        tile.alignment_image = GrayImage::from_fn(side, side, |x, y| {
            image::Luma([texture(f64::from(x) + offset.x, f64::from(y) + offset.y)])
        });
        tile.width = side;
        tile.height = side;
        tile.scale_factor = 1.0;
    }
    let coverages = fully_covered_quality_coverages(&tiles);
    let left_to_right = Matrix3::new(1.0, 0.0, -shift.x, 0.0, 1.0, -shift.y, 0.0, 0.0, 1.0);
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, left_to_right.try_inverse().unwrap()),
    ]);
    // The production matcher's key-direction relation, not a hand-built one.
    let (relation, diagnostic) = virtual_tile_direct_station_match(
        0,
        1,
        &tiles,
        &coverages,
        &initial,
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    let relation = relation.unwrap_or_else(|| panic!("measurable shift: {diagnostic:?}"));
    let matches = HashMap::from([((0, 1), relation)]);
    let station_topology = synthetic_virtual_tile_topology(2);

    let mut accepted = Vec::new();
    for (left_name, inverted) in [
        ("virtual://synthetic-station-0", false),
        ("virtual://z", true),
    ] {
        tiles[0].filename = left_name.to_string();
        // The content-derived measurement direction runs right→left when
        // `inverted`; the key-direction fit must then be read inverted.
        assert_eq!(canonical_match_direction(&tiles, 0, 1).2, inverted);
        let mut closure = stack_report::ClosureReport::default();
        let mut report = stack_report::StationRelationsReport::default();
        let solved = solve_virtual_tile_station_poses_with_report(
            &tiles,
            fully_covered_solver_coverages(&tiles, &coverages),
            &matches,
            &initial,
            &station_topology,
            &mut closure,
            &mut report,
        );
        assert_eq!(solved.len(), 2, "inverted {inverted}: {report:?}");
        assert_eq!(report.accepted.len(), 1, "inverted {inverted}: {report:?}");
        accepted.push(report.accepted[0]);
    }
    let (forward, reversed) = (&accepted[0], &accepted[1]);
    assert!(forward.median_error_px <= STATION_RELATION_MAX_MEDIAN_ERROR_PX);
    assert!((forward.median_error_px - reversed.median_error_px).abs() < 1e-6);
    assert_eq!(forward.inliers, reversed.inliers);
}

#[test]
fn virtual_tile_no_fit_candidate_reports_unmeasured_without_inventing_reasons() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let tiles = synthetic_virtual_tiles(2);
    let initial = tiles
        .iter()
        .map(|tile| (tile.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();
    let mut closure = stack_report::ClosureReport::default();
    let mut report = stack_report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_poses_with_report(
        &tiles,
        fully_covered_solver_coverages(&tiles, &fully_covered_quality_coverages(&tiles)),
        &HashMap::new(),
        &initial,
        &synthetic_virtual_tile_topology(2),
        &mut closure,
        &mut report,
    );
    assert!(solved.is_empty());
    assert_eq!(report.accepted.len(), 0);
    assert_eq!(report.rejected_count, 1);
    assert!(report.rejected_by_reason.is_empty());
    let (_, diagnostic) = virtual_tile_direct_station_match(
        0,
        1,
        &[],
        &[],
        &HashMap::new(),
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    report.candidates = vec![diagnostic];
    report.summarize_candidate_measurements();
    assert_eq!(report.unmeasured_count, 1);
    assert_eq!(
        report.rejected_by_failure_stage,
        BTreeMap::from([("missing_input".to_string(), 1)])
    );
    assert!(report.rejected_by_reason.is_empty());
}

#[test]
fn default_two_stage_flow_solves_ten_sources_as_three_fused_stations() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let images = (0..10)
        .map(|index| focus_textured_test_image(index, &format!("source-{index}.raw")))
        .collect::<Vec<_>>();
    let mut source_matches = HashMap::new();
    for &(start, end) in &[(0usize, 4usize), (4, 7), (7, 10)] {
        for index in start..end - 1 {
            source_matches.insert((index, index + 1), observed_translation_match(0.0, 0.0, 36));
        }
    }
    // Each station boundary has only one Source_RAW observation.  The old
    // early authoritative solve discarded both as support=1 and rejected
    // the three components before the fused station tiles existed.
    source_matches.insert((3, 4), observed_translation_match(120.0, 0.0, 36));
    source_matches.insert((6, 7), observed_translation_match(120.0, 0.0, 36));
    let mut early_closure = stack_report::ClosureReport::default();
    let mut early_relations = stack_report::StationRelationsReport::default();
    let (order, local_geometry) = build_focus_stack_stitching_order_with_report(
        &images,
        &source_matches,
        &mut early_closure,
        &mut early_relations,
        false,
    );
    assert_eq!(order.len(), 10);
    assert_eq!(local_geometry.len(), 10);
    assert!(early_relations.accepted.is_empty());
    assert_eq!(early_relations.connectivity.components, 0);
    assert_eq!(early_closure.status, stack_report::ClosureStatus::NotRun);

    let tiles = synthetic_virtual_tiles(3);
    let tile_matches = HashMap::from([
        ((0, 1), observed_translation_match(0.0, 0.0, 36)),
        ((1, 2), observed_translation_match(0.0, 0.0, 36)),
    ]);
    let initial = tiles
        .iter()
        .map(|tile| (tile.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();
    let mut final_closure = stack_report::ClosureReport::default();
    let mut final_relations = stack_report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_poses_with_report(
        &tiles,
        fully_covered_solver_coverages(&tiles, &fully_covered_quality_coverages(&tiles)),
        &tile_matches,
        &initial,
        &synthetic_virtual_tile_topology(3),
        &mut final_closure,
        &mut final_relations,
    );

    assert_eq!(solved.len(), 3);
    assert_eq!(final_relations.connectivity.components, 1);
    assert_eq!(final_relations.accepted.len(), 2);
    assert!(final_relations.discarded_single_layer_evidence.is_empty());
    assert!(final_relations.accepted.iter().all(|relation| {
        relation.evidence_kind == stack_report::StationRelationEvidenceKind::VirtualTile
            && relation.independent_support == 1
    }));
    assert_eq!(
        final_closure.status,
        stack_report::ClosureStatus::NoConstraints
    );
}

#[test]
fn true_virtual_tile_disconnection_rejects_before_composition() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let tiles = synthetic_virtual_tiles(3);
    let tile_matches = HashMap::from([((0, 1), observed_translation_match(0.0, 0.0, 36))]);
    let initial = tiles
        .iter()
        .map(|tile| (tile.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();
    let mut closure = stack_report::ClosureReport::default();
    let mut relations = stack_report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_poses_with_report(
        &tiles,
        fully_covered_solver_coverages(&tiles, &fully_covered_quality_coverages(&tiles)),
        &tile_matches,
        &initial,
        &synthetic_virtual_tile_topology(3),
        &mut closure,
        &mut relations,
    );

    assert!(solved.is_empty());
    assert_eq!(relations.connectivity.components, 2);
    assert_eq!(relations.connectivity.component_member_counts, vec![2, 1]);
}

#[test]
fn single_virtual_tile_skips_station_relations_and_closure() {
    let tiles = synthetic_virtual_tiles(1);
    let initial = HashMap::from([(tiles[0].id, Matrix3::identity())]);
    let mut closure = stack_report::ClosureReport::default();
    let mut relations = stack_report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_poses_with_report(
        &tiles,
        fully_covered_solver_coverages(&tiles, &fully_covered_quality_coverages(&tiles)),
        &HashMap::new(),
        &initial,
        &synthetic_virtual_tile_topology(1),
        &mut closure,
        &mut relations,
    );

    assert_eq!(solved, initial);
    assert!(relations.accepted.is_empty());
    assert_eq!(relations.connectivity.components, 0);
    assert_eq!(closure.status, stack_report::ClosureStatus::NotRun);
}

#[test]
fn station_source_evidence_requires_two_independent_layers() {
    let discarded = single_layer_station_evidence_discard(1, 3, 8)
        .expect("one Source_RAW without consensus must be discarded");
    assert_eq!(discarded["count"], serde_json::json!(1));
    assert_eq!(discarded["stations"], serde_json::json!([3, 8]));
    assert_eq!(discarded["independent_support"], serde_json::json!(1));
    assert!(
        single_layer_station_evidence_discard(2, 3, 8).is_none(),
        "two independent focus layers are valid supplemental evidence"
    );
}
