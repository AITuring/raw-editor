//! Rendering-prior repair planning and the authoritative relation pairs it starts from: seed
//! fallback, quality gates and the canonical scale contract, connectivity, stable prior-free
//! RANSAC and cluster repairs.

use super::*;

fn line_topology(station_count: usize) -> topology::StationTopology {
    topology::StationTopology {
        row: vec![0; station_count],
        column: (0..station_count as u32).collect(),
        ..topology::StationTopology::default()
    }
}

fn prior_proposal(
    left: usize,
    right: usize,
    left_to_right: Matrix3<f64>,
    inliers: usize,
) -> VirtualTilePriorProposal {
    VirtualTilePriorProposal {
        left,
        right,
        left_to_right,
        inliers,
    }
}

/// Four station placements with small projective parts.
fn sample_tile_to_world() -> Vec<Matrix3<f64>> {
    vec![
        Matrix3::identity(),
        Matrix3::new(1.0, 0.0, 380.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0),
        Matrix3::new(1.01, 0.002, 770.0, -0.001, 0.99, -8.0, 0.000_001, 0.0, 1.0),
        Matrix3::new(1.0, 0.0, 1_150.0, 0.0, 1.0, 12.0, 0.0, 0.0, 1.0),
    ]
}

fn sample_relation(dx: f64, dy: f64) -> Matrix3<f64> {
    Matrix3::new(
        0.998, 0.004, dx, 0.003, 1.001, dy, 0.000_002, -0.000_001, 1.0,
    )
}

fn repaired_stations(repairs: &[VirtualTilePriorRepair]) -> Vec<usize> {
    repairs.iter().map(|repair| repair.station).collect()
}

// ---------------------------------------------------------------------------
// 5. rendering-prior repair planning
// ---------------------------------------------------------------------------

#[test]
fn prior_free_seed_fallback_requires_a_fitted_polish_support_failure() {
    assert!(virtual_tile_prior_free_seed_fallback_conditions(
        true,
        "insufficient_polished_support",
        STATION_RELATION_MIN_INLIERS,
        STATION_RELATION_MIN_INLIERS,
    ));
    for (model_fitted, stage, fitted, prior_free) in [
        (
            false,
            "insufficient_polished_support",
            STATION_RELATION_MIN_INLIERS,
            STATION_RELATION_MIN_INLIERS,
        ),
        (
            true,
            "fitted",
            STATION_RELATION_MIN_INLIERS,
            STATION_RELATION_MIN_INLIERS,
        ),
        (
            true,
            "insufficient_polished_support",
            STATION_RELATION_MIN_INLIERS - 1,
            STATION_RELATION_MIN_INLIERS,
        ),
        (
            true,
            "insufficient_polished_support",
            STATION_RELATION_MIN_INLIERS,
            STATION_RELATION_MIN_INLIERS - 1,
        ),
    ] {
        assert!(!virtual_tile_prior_free_seed_fallback_conditions(
            model_fitted,
            stage,
            fitted,
            prior_free,
        ));
    }
}

#[test]
fn plan_virtual_tile_prior_repairs_needs_proposals_and_two_clusters() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(4);
    let relation = sample_relation(-372.0, 9.0);
    assert!(
        plan_virtual_tile_prior_repairs(4, &[(0, 1), (2, 3)], &[], &tile_to_world, &topology)
            .is_empty()
    );
    assert!(
        plan_virtual_tile_prior_repairs(
            4,
            &[(0, 1), (1, 2), (2, 3)],
            &[prior_proposal(1, 2, relation, 50)],
            &tile_to_world,
            &topology,
        )
        .is_empty(),
        "one authoritative cluster needs no repair"
    );
    assert!(
        plan_virtual_tile_prior_repairs(
            3,
            &[],
            &[prior_proposal(0, 1, relation, 50)],
            &tile_to_world,
            &topology,
        )
        .is_empty(),
        "placements must match the station count"
    );
    assert!(
        plan_virtual_tile_prior_repairs(
            1,
            &[],
            &[prior_proposal(0, 1, relation, 50)],
            &tile_to_world[..1],
            &topology,
        )
        .is_empty()
    );
}

#[test]
fn authoritative_quality_gates_control_production_cluster_and_proposal_behavior() {
    let dimensions = (96, 96);
    let coverage = full_coverage(dimensions.0, dimensions.1);
    let quality_coverages = vec![coverage.clone(), coverage.clone()];
    let overlap_coverages = vec![coverage.clone(), coverage];
    let coverages = VirtualTileSolverCoverages {
        quality: &quality_coverages,
        overlap: &overlap_coverages,
        overlap_factor: 1,
    };
    let points = (0..24)
        .map(|index| {
            let x = 8.0 + f64::from(index % 8) * 10.0;
            let y = 8.0 + f64::from(index / 8) * 30.0;
            (Point2::new(x, y), Point2::new(x, y))
        })
        .collect::<Vec<_>>();
    let relation = measured_relation(Matrix3::identity(), points);
    let mut topology = line_topology(2);
    topology.candidates.push(topology::CandidateAdjacency {
        left: 0,
        right: 1,
        kind: report::AdjacencyKind::SameRow,
        score: 1.0,
    });
    let proposal_pairs = |tiles: &[ImageInfo], relation: MatchInfo| {
        let authoritative = virtual_tile_authoritative_pairs(
            &HashMap::from([((0, 1), vec![relation])]),
            tiles,
            coverages,
        );
        (
            authoritative.clone(),
            virtual_tile_prior_free_candidate_pairs(tiles.len(), &authoritative, &topology),
        )
    };

    let texture = textured_plane(dimensions.0, dimensions.1);
    let mut reversed_tiles = vec![
        plane_info(0, texture.clone()),
        plane_info(1, texture.clone()),
    ];
    reversed_tiles[0].filename = "virtual://z".to_string();
    reversed_tiles[1].filename = "virtual://a".to_string();
    let (authoritative, proposals) = proposal_pairs(&reversed_tiles, relation.clone());
    assert_eq!(
        authoritative,
        [(0, 1)],
        "reversed canonical direction remains authoritative"
    );
    assert!(
        proposals.is_empty(),
        "an accepted pair merges the production clusters"
    );

    let low_contrast = GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
        image::Luma([40 + texture.get_pixel(x, y)[0] / 4])
    });
    let brighter = GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
        image::Luma([140 + texture.get_pixel(x, y)[0] / 4])
    });
    let photometric_tiles = vec![plane_info(0, low_contrast), plane_info(1, brighter)];
    let (authoritative, proposals) = proposal_pairs(&photometric_tiles, relation.clone());
    assert!(
        authoritative.is_empty(),
        "photometric mismatch cannot merge clusters"
    );
    assert_eq!(
        proposals,
        [(0, 1)],
        "photometric failure keeps the prior-free proposal"
    );

    let weak_edges = GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
        let value = i16::from(texture.get_pixel(x, y)[0]);
        image::Luma([(128 + (value - 128) / 4) as u8])
    });
    let edge_tiles = vec![plane_info(0, texture.clone()), plane_info(1, weak_edges)];
    let (authoritative, proposals) = proposal_pairs(&edge_tiles, relation.clone());
    assert!(
        authoritative.is_empty(),
        "edge-strength mismatch cannot merge clusters"
    );
    assert_eq!(
        proposals,
        [(0, 1)],
        "edge failure keeps the prior-free proposal"
    );

    let flat = GrayImage::from_pixel(dimensions.0, dimensions.1, image::Luma([128]));
    let orientation_missing_tiles = vec![plane_info(0, flat.clone()), plane_info(1, flat)];
    let (authoritative, proposals) = proposal_pairs(&orientation_missing_tiles, relation);
    assert!(
        authoritative.is_empty(),
        "missing dense orientation cannot merge clusters"
    );
    assert_eq!(
        proposals,
        [(0, 1)],
        "missing orientation keeps the prior-free proposal"
    );
}

#[test]
fn authoritative_pairs_keep_the_inclusive_canonical_scale_contract() {
    let dimensions = (96, 96);
    let gray = textured_plane(dimensions.0, dimensions.1);
    let mut tiles = vec![plane_info(0, gray.clone()), plane_info(1, gray)];
    tiles[0].filename = "virtual://z".to_string();
    tiles[1].filename = "virtual://a".to_string();
    let coverage = full_coverage(dimensions.0, dimensions.1);
    let quality_coverages = vec![coverage.clone(), coverage.clone()];
    let overlap_coverages = vec![coverage.clone(), coverage];
    let coverages = VirtualTileSolverCoverages {
        quality: &quality_coverages,
        overlap: &overlap_coverages,
        overlap_factor: 1,
    };
    let points = (0..24)
        .map(|index| {
            let x = 8.0 + f64::from(index % 8) * 10.0;
            let y = 8.0 + f64::from(index / 8) * 30.0;
            (Point2::new(x, y), Point2::new(x, y))
        })
        .collect::<Vec<_>>();

    assert_eq!(STATION_RELATION_SCALE_RATIO_MIN, 0.95);
    assert_eq!(STATION_RELATION_SCALE_RATIO_MAX, 1.05);
    assert_eq!(
        virtual_tile_authoritative_pairs(
            &HashMap::from([(
                (0, 1,),
                vec![measured_relation(Matrix3::identity(), points.clone())],
            )]),
            &tiles,
            coverages,
        ),
        [(0, 1)]
    );

    for canonical_scale in [
        f64::from_bits(STATION_RELATION_SCALE_RATIO_MIN.to_bits() - 1),
        f64::from_bits(STATION_RELATION_SCALE_RATIO_MAX.to_bits() + 1),
    ] {
        let relation = measured_relation(uniform_scale(1.0 / canonical_scale), points.clone());
        assert!(
            virtual_tile_authoritative_pairs(
                &HashMap::from([((0, 1), vec![relation])]),
                &tiles,
                coverages,
            )
            .is_empty(),
            "canonical scale {canonical_scale}"
        );
    }
}

#[test]
fn virtual_tile_non_invertible_canonical_transform_is_summarized_once() {
    let _run_scope = degradation::begin_run_scope();
    let mut report = report::StationRelationsReport::default();
    report.record_non_invertible_canonical_pair(0, 1);
    report.record_non_invertible_canonical_pair(1, 0);
    report.candidates = vec![sample_candidate_record()];
    report.summarize_candidate_measurements();
    assert_eq!(report.rejected_count, 1);
    assert_eq!(report.unmeasured_count, 0);
    assert_eq!(
        report.rejected_by_reason[degradation::STATION_RELATION_NON_INVERTIBLE_CANONICAL_TRANSFORM],
        1
    );
    assert_eq!(
        report.rejected_by_failure_stage["canonical_transform_non_invertible"],
        1
    );
    assert_eq!(
        report.candidates[0].failure_stage,
        "canonical_transform_non_invertible"
    );
    assert!(report.candidates[0].model_fitted);
    assert!(
        !report
            .rejected_by_reason
            .contains_key(degradation::STATION_RELATION_SCALE_OUT_OF_RANGE)
    );
}

#[test]
fn virtual_tile_connectivity_reports_original_source_members() {
    let stations = vec![
        vec!["/raw/a.nef".to_string(), "/raw/b.nef".to_string()],
        vec!["/raw/c.nef".to_string()],
        vec!["/raw/d.nef".to_string(), "/raw/e.nef".to_string()],
    ];
    assert_eq!(
        station_graph_components_from_members(&stations, &[(0, 1)]),
        vec![
            vec![
                "/raw/a.nef".to_string(),
                "/raw/b.nef".to_string(),
                "/raw/c.nef".to_string(),
            ],
            vec!["/raw/d.nef".to_string(), "/raw/e.nef".to_string()],
        ]
    );
}

#[test]
fn prior_free_relation_uses_stable_ransac_for_large_projective_coordinates() {
    let dimensions = (30_000, 30_000);
    let expected = Matrix3::new(
        1.018, 0.012, 73.0, -0.008, 0.997, -41.0, 1.5e-6, -1.2e-6, 1.0,
    );
    let points = (0..32)
        .map(|index| {
            let source = Point2::new(
                10_000.0 + f64::from(index % 8) * 230.0,
                20_000.0 + f64::from(index / 8) * 190.0,
            );
            let target = transformed_point(&expected, source).expect("finite target");
            (source, target)
        })
        .collect::<Vec<_>>();
    let mut all_points = points.clone();
    all_points.extend((0..12).map(|index| {
        (
            Point2::new(10_100.0 + f64::from(index) * 120.0, 24_500.0),
            Point2::new(
                400.0 + f64::from(index) * 137.0,
                600.0 + f64::from(index) * 29.0,
            ),
        )
    }));
    let descriptor_for = |index: usize| {
        let mut descriptor = [0u8; BRIEF_DESCRIPTOR_SIZE / 8];
        descriptor[index / 8] = 1u8 << (index % 8);
        descriptor
    };
    let left_features = all_points
        .iter()
        .enumerate()
        .map(|(index, &(source, _))| Feature {
            keypoint: KeyPoint {
                x: source.x as u32,
                y: source.y as u32,
            },
            descriptor: descriptor_for(index),
            support_scale: 1.0,
        })
        .collect::<Vec<_>>();
    let right_features = all_points
        .iter()
        .enumerate()
        .map(|(index, &(_, target))| Feature {
            keypoint: KeyPoint {
                x: target.x as u32,
                y: target.y as u32,
            },
            descriptor: descriptor_for(index),
            support_scale: 1.0,
        })
        .collect::<Vec<_>>();
    let (relation, inliers) =
        virtual_tile_prior_free_relation(1, &left_features, 1, &right_features, dimensions)
            .expect("stable RANSAC recovers the large-coordinate projective relation");
    assert_eq!(inliers, points.len());
    let probe = Point2::new(11_234.0, 20_456.0);
    let expected_probe = transformed_point(&expected, probe).expect("finite expected probe");
    let actual_probe = transformed_point(&relation, probe).expect("finite fitted probe");
    assert!((actual_probe - expected_probe).norm() < 2.0);
}

#[test]
fn plan_virtual_tile_prior_repairs_moves_the_whole_unfixed_cluster_onto_the_proposal() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(4);
    let authoritative = [(0, 1), (2, 3)];
    let relation = sample_relation(-372.0, 9.0);

    // The fixed station is the proposal's left: H(u→f) = (left→right)⁻¹.
    let repairs = plan_virtual_tile_prior_repairs(
        4,
        &authoritative,
        &[prior_proposal(1, 2, relation, 50)],
        &tile_to_world,
        &topology,
    );
    assert_eq!(repaired_stations(&repairs), vec![2, 3]);
    for repair in &repairs {
        assert_eq!(repair.fixed_station, 1);
        assert_eq!(repair.relation, (1, 2));
        assert_eq!(repair.relation_inliers, 50);
        assert_eq!(repair.world_correction, repairs[0].world_correction);
    }
    let correction = repairs[0].world_correction;
    assert_projectively_equal(
        &(correction * tile_to_world[2]),
        &(tile_to_world[1] * relation.try_inverse().unwrap()),
        "left fixed",
    );

    // The fixed station is the proposal's right: H(u→f) = left→right.
    let repairs = plan_virtual_tile_prior_repairs(
        4,
        &authoritative,
        &[prior_proposal(2, 1, relation, 50)],
        &tile_to_world,
        &topology,
    );
    assert_eq!(repaired_stations(&repairs), vec![2, 3]);
    assert!(repairs.iter().all(|repair| repair.fixed_station == 1
        && repair.relation == (2, 1)
        && repair.world_correction == repairs[0].world_correction));
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[2]),
        &(tile_to_world[1] * relation),
        "right fixed",
    );
}

#[test]
fn plan_virtual_tile_prior_repairs_keeps_the_largest_cluster_and_breaks_ties_by_position() {
    let tile_to_world = sample_tile_to_world();
    let relation = sample_relation(-372.0, 9.0);
    let proposals = [prior_proposal(0, 1, relation, 40)];

    // {0} against {1, 2}: the larger cluster is the reference.
    let repairs = plan_virtual_tile_prior_repairs(
        3,
        &[(1, 2)],
        &proposals,
        &tile_to_world[..3],
        &line_topology(3),
    );
    assert_eq!(repaired_stations(&repairs), vec![0]);
    assert_eq!(repairs[0].fixed_station, 1);

    // Equal sizes: the smallest row/column key is the reference.
    let repairs =
        plan_virtual_tile_prior_repairs(2, &[], &proposals, &tile_to_world[..2], &line_topology(2));
    assert_eq!(repaired_stations(&repairs), vec![1]);
    assert_eq!(repairs[0].fixed_station, 0);
    let reordered = topology::StationTopology {
        row: vec![1, 0],
        column: vec![0, 3],
        ..topology::StationTopology::default()
    };
    let repairs =
        plan_virtual_tile_prior_repairs(2, &[], &proposals, &tile_to_world[..2], &reordered);
    assert_eq!(repaired_stations(&repairs), vec![0]);
    assert_eq!(repairs[0].fixed_station, 1);
}

#[test]
fn plan_virtual_tile_prior_repairs_prefers_inliers_and_composes_repairs_transitively() {
    let tile_to_world = sample_tile_to_world();
    let weak_relation = sample_relation(-372.0, 9.0);
    let strong_relation = sample_relation(-375.5, 6.0);
    let weak = prior_proposal(0, 1, weak_relation, 50);
    let strong = prior_proposal(0, 1, strong_relation, 80);
    let topology = line_topology(2);
    let repairs =
        plan_virtual_tile_prior_repairs(2, &[], &[weak, strong], &tile_to_world[..2], &topology);
    assert_eq!(
        repaired_stations(&repairs),
        vec![1],
        "repaired at most once"
    );
    assert_eq!(repairs[0].relation_inliers, 80);
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[1]),
        &(tile_to_world[0] * strong_relation.try_inverse().unwrap()),
        "strongest proposal",
    );
    assert_eq!(
        plan_virtual_tile_prior_repairs(2, &[], &[strong, weak], &tile_to_world[..2], &topology),
        repairs,
        "proposal order is irrelevant"
    );

    // Three singleton clusters: 0→2 (90) first, then 2→1 (60) beats 0→1 (40)
    // and composes station 2's correction.
    let relation_01 = sample_relation(-372.0, 9.0);
    let relation_02 = sample_relation(-765.0, 14.0);
    let relation_21 = sample_relation(391.0, 12.0);
    let repairs = plan_virtual_tile_prior_repairs(
        3,
        &[],
        &[
            prior_proposal(0, 1, relation_01, 40),
            prior_proposal(0, 2, relation_02, 90),
            prior_proposal(2, 1, relation_21, 60),
        ],
        &tile_to_world[..3],
        &line_topology(3),
    );
    assert_eq!(repaired_stations(&repairs), vec![1, 2], "sorted by station");
    let (repair_1, repair_2) = (&repairs[0], &repairs[1]);
    assert_eq!(
        (
            repair_2.fixed_station,
            repair_2.relation,
            repair_2.relation_inliers
        ),
        (0, (0, 2), 90)
    );
    assert_projectively_equal(
        &(repair_2.world_correction * tile_to_world[2]),
        &(tile_to_world[0] * relation_02.try_inverse().unwrap()),
        "first repair",
    );
    assert_eq!(
        (
            repair_1.fixed_station,
            repair_1.relation,
            repair_1.relation_inliers
        ),
        (2, (2, 1), 60)
    );
    assert_projectively_equal(
        &(repair_1.world_correction * tile_to_world[1]),
        &(repair_2.world_correction * tile_to_world[2] * relation_21.try_inverse().unwrap()),
        "transitive repair",
    );
}

#[test]
fn plan_virtual_tile_prior_repairs_skips_an_unbounded_strong_proposal_for_a_bounded_fallback() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(2);
    let bounded = sample_relation(-372.0, 9.0);
    // Station 0 is fixed. This relation would require station 1's world pose
    // to shrink by 20%, outside the existing inclusive [0.95, 1.05] envelope.
    let unbounded = uniform_scale(1.25) * bounded;
    let repairs = plan_virtual_tile_prior_repairs(
        2,
        &[],
        &[
            prior_proposal(0, 1, unbounded, 99),
            prior_proposal(0, 1, bounded, 30),
        ],
        &tile_to_world[..2],
        &topology,
    );

    assert_eq!(repaired_stations(&repairs), vec![1]);
    assert_eq!(repairs[0].relation_inliers, 30);
    let correction_scale = homography_scale_ratio(&repairs[0].world_correction).unwrap();
    assert!(
        (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
            .contains(&correction_scale),
        "fallback correction scale {correction_scale}"
    );
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[1]),
        &(tile_to_world[0] * bounded.try_inverse().unwrap()),
        "bounded fallback",
    );
}

#[test]
fn plan_virtual_tile_prior_repairs_ignores_invalid_proposals_and_skips_singular_relations() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(2);
    let relation = sample_relation(-372.0, 9.0);
    assert!(
        plan_virtual_tile_prior_repairs(
            2,
            &[],
            &[
                prior_proposal(0, 2, relation, 99),
                prior_proposal(5, 0, relation, 99),
                prior_proposal(1, 1, relation, 99),
            ],
            &tile_to_world[..2],
            &topology,
        )
        .is_empty()
    );
    // Station 0 is the reference. A zero relation cannot be inverted (fixed
    // left) nor normalised (fixed right); the valid weaker proposal repairs.
    let repairs = plan_virtual_tile_prior_repairs(
        2,
        &[],
        &[
            prior_proposal(0, 1, Matrix3::zeros(), 99),
            prior_proposal(1, 0, Matrix3::zeros(), 98),
            prior_proposal(0, 1, relation, 30),
        ],
        &tile_to_world[..2],
        &topology,
    );
    assert_eq!(repaired_stations(&repairs), vec![1]);
    assert_eq!(repairs[0].relation_inliers, 30);
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[1]),
        &(tile_to_world[0] * relation.try_inverse().unwrap()),
        "valid fallback",
    );
}
