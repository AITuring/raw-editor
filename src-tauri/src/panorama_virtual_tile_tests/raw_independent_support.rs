//! Source_RAW independent support: per-pair spatial proposals with match provenance, proposal
//! scores and duplicate provenance, one RAW never counted twice, and layer models in verified
//! local-to-anchor coordinates.

use super::*;

#[test]
fn source_raw_spatial_proposals_are_per_pair_and_preserve_match_provenance() {
    let mut matches = Vec::new();
    let mut points = Vec::new();
    for index in 0..12 {
        let source = Point2::new((index % 4) as f64 * 100.0, (index / 4) as f64 * 120.0);
        matches.push(Match {
            index1: index,
            index2: 100 + index,
        });
        points.push((
            source,
            Point2::new(source.x + 305.0 + (index % 2) as f64, source.y - 181.0),
        ));
    }
    for index in 12..48 {
        let source = Point2::new((index % 8) as f64 * 80.0, (index / 8) as f64 * 90.0);
        matches.push(Match {
            index1: index,
            index2: 100 + index,
        });
        points.push((
            source,
            Point2::new(
                source.x - 900.0 + index as f64 * 47.0,
                source.y + 800.0 - index as f64 * 53.0,
            ),
        ));
    }
    let proposals = source_raw_descriptor_proposals(&matches, &points, (1000, 800));
    assert!(!proposals.is_empty());
    let recovered = &proposals[0].1;
    assert_eq!(recovered.len(), 12);
    assert_eq!(
        recovered
            .iter()
            .map(|matched| (matched.index1, matched.index2))
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        recovered.len(),
        "proposal filtering must preserve one-to-one descriptor provenance"
    );
    assert!(recovered.iter().all(|matched| matched.index1 < 12));
}

#[test]
fn source_raw_proposal_scores_and_duplicate_provenance_never_inflate_support() {
    let mut records = (0..32)
        .map(|_| {
            let mut record = source_raw_test_record(0, 10);
            record.model_provenance = "per_raw_coarse_luminance_gradient_correlation".to_string();
            record.proposal_seeded = true;
            record.proposal_score = Some(f64::MAX);
            record.correlation_proposals = (0..8)
                .map(|rank| report::SourceRawCorrelationProposalRecord {
                    rank,
                    score: f64::MAX,
                    full_resolution_validated: true,
                    ..Default::default()
                })
                .collect();
            record
        })
        .collect::<Vec<_>>();
    let models = records
        .iter()
        .enumerate()
        .map(|(record_index, _)| SourceRawLayerDiagnosticModel {
            left_source: 0,
            right_source: 10,
            record_index,
            transform: Matrix3::identity(),
            spatial_support: 0.9,
        })
        .collect::<Vec<_>>();
    let cluster = (0..models.len()).collect::<Vec<_>>();
    assert_eq!(
        source_raw_independent_observations(&models, &cluster).len(),
        1
    );
    let clusters = source_raw_consensus_clusters(&models, &mut records, (100, 100));
    assert!(
        clusters.is_empty(),
        "proposal count and score cannot create support=2"
    );
}

#[test]
fn source_raw_independent_support_never_counts_one_raw_twice() {
    let pairs = [(0, 10), (0, 11), (1, 11)];
    let models = pairs
        .iter()
        .enumerate()
        .map(
            |(record_index, &(left_source, right_source))| SourceRawLayerDiagnosticModel {
                left_source,
                right_source,
                record_index,
                transform: Matrix3::identity(),
                spatial_support: 0.4,
            },
        )
        .collect::<Vec<_>>();
    assert_eq!(
        source_raw_independent_observations(&models, &[0, 1]).len(),
        1,
        "two layer pairs sharing one left RAW are one independent observation"
    );
    let selected = source_raw_independent_observations(&models, &[0, 1, 2]);
    assert_eq!(selected.len(), 2);
    let selected_left = selected
        .iter()
        .map(|&index| models[index].left_source)
        .collect::<std::collections::BTreeSet<_>>();
    let selected_right = selected
        .iter()
        .map(|&index| models[index].right_source)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(selected_left.len(), 2);
    assert_eq!(selected_right.len(), 2);

    let mut records = pairs
        .iter()
        .map(|&(left, right)| source_raw_test_record(left, right))
        .collect::<Vec<_>>();
    let clusters = source_raw_consensus_clusters(&models, &mut records, (100, 100));
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].observation_count, 3);
    assert_eq!(clusters[0].independent_support, 2);
    assert!(clusters[0].scale_compliant);
    assert!(clusters[0].spatially_broad);
    assert_eq!(clusters[0].supporting_layer_pairs.len(), 2);
}

#[test]
fn source_raw_layer_models_use_verified_local_to_anchor_coordinates() {
    let images = (0..4)
        .map(|id| plane_info(id, textured_plane(100, 100)))
        .collect::<Vec<_>>();
    let left_locals = [Matrix3::identity(), translation(5.0, 0.0)];
    let right_locals = [Matrix3::identity(), translation(-4.0, 0.0)];
    let expected_anchor = translation(20.0, 0.0);
    let mut matches = HashMap::new();
    for (left, right, left_local, right_local) in [
        (0usize, 2usize, left_locals[0], right_locals[0]),
        (1usize, 3usize, left_locals[1], right_locals[1]),
    ] {
        let raw_transform = right_local.try_inverse().unwrap() * expected_anchor * left_local;
        let points = [
            (10.0, 10.0),
            (50.0, 10.0),
            (80.0, 10.0),
            (10.0, 50.0),
            (80.0, 50.0),
            (10.0, 80.0),
            (50.0, 80.0),
            (80.0, 80.0),
        ]
        .into_iter()
        .map(|(x, y)| {
            let source = Point2::new(x, y);
            (source, transformed_point(&raw_transform, source).unwrap())
        })
        .collect::<Vec<_>>();
        matches.insert(
            (left, right),
            MatchInfo {
                native_verified_points: Vec::new(),
                homography: raw_transform,
                inliers: points.len(),
                sequence_bridge: false,
                coarse_bridge: false,
                points,
                candidate_points: Vec::new(),
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
                canonical_homography: Some(raw_transform),
            },
        );
    }
    let groups = vec![
        FocusCaptureGroup {
            members: vec![0, 1],
            anchor: 0,
            local_to_anchor: HashMap::from([(0, left_locals[0]), (1, left_locals[1])]),
        },
        FocusCaptureGroup {
            members: vec![2, 3],
            anchor: 2,
            local_to_anchor: HashMap::from([(2, right_locals[0]), (3, right_locals[1])]),
        },
    ];
    let diagnostics = source_raw_layer_pair_diagnostics(&images, &matches, &groups);
    assert_eq!(diagnostics.len(), 1);
    let fitted = diagnostics[0]
        .layer_models
        .iter()
        .filter(|model| model.model_fitted)
        .collect::<Vec<_>>();
    assert_eq!(fitted.len(), 2);
    for model in fitted {
        assert_projectively_equal(
            &Matrix3::from_row_slice(&model.anchor_transform),
            &expected_anchor,
            "layer model canonicalization",
        );
    }
    assert_eq!(diagnostics[0].consensus_clusters.len(), 1);
    assert_eq!(diagnostics[0].consensus_clusters[0].independent_support, 2);
}
