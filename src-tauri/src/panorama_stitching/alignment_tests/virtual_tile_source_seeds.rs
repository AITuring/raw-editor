use super::*;
use crate::panorama_utils::stitching::FocusVirtualTileGeometry;

#[test]
fn source_seed_uses_local_crops_and_preserves_raw_direction_at_both_render_scales() {
    // Different global station placements must never be composed a second time
    // into an independently observed source relation.
    let global = HashMap::from([
        (
            0,
            Matrix3::new(0.9, -0.02, 230.0, 0.01, 0.91, 3100.0, 0.0, 0.0, 1.0),
        ),
        (
            1,
            Matrix3::new(0.8, 0.03, -400.0, -0.02, 0.82, 5200.0, 0.0, 0.0, 1.0),
        ),
    ]);
    let images = vec![focus_test_image(0, "a.raw"), focus_test_image(1, "b.raw")];
    let groups = (0..2)
        .map(|anchor| FocusCaptureGroup {
            anchor,
            members: vec![anchor],
            local_to_anchor: HashMap::from([(anchor, Matrix3::identity())]),
        })
        .collect::<Vec<_>>();
    let frames = focus_anchor_render_frames(&groups, &images, &global, None).unwrap();
    let raw_relation = Matrix3::new(1.02, 0.01, 180.0, -0.01, 1.02, -240.0, 0.0, 0.0, 1.0);
    let matches = HashMap::from([((0, 1), observed_transform_match(raw_relation, 36))]);
    for scale in [1.0, 0.5] {
        let left = FocusVirtualTileGeometry {
            width: 1200,
            height: 1100,
            tile_to_world: virtual_tile_translation(-23.0, -31.0),
        };
        let right = FocusVirtualTileGeometry {
            width: 1300,
            height: 1200,
            tile_to_world: virtual_tile_translation(-47.0, -19.0),
        };
        let proposals = virtual_tile_source_match_proposals(
            &[&images[0]],
            &[&images[1]],
            &matches,
            &frames.homographies,
            &left,
            &right,
            scale,
        );
        assert_eq!(proposals.len(), 1);
        let point = Point2::new(300.0, 400.0);
        let raw_left = Point2::new((point.x - 23.0) / scale, (point.y - 31.0) / scale);
        let raw_right = transformed_point(&raw_relation, raw_left).unwrap();
        let expected = Point2::new(raw_right.x * scale + 47.0, raw_right.y * scale + 19.0);
        let measured = transformed_point(&proposals[0].left_to_right, point).unwrap();
        assert!((measured - expected).norm() < 1e-9);
        let reverse = virtual_tile_source_match_proposals(
            &[&images[1]],
            &[&images[0]],
            &matches,
            &frames.homographies,
            &right,
            &left,
            scale,
        );
        let recovered = transformed_point(&reverse[0].left_to_right, measured).unwrap();
        assert!((recovered - point).norm() < 1e-9);
        // This is the former call-site error: a globally placed tile crop is
        // incompatible with the per-station source poses.
        let wrong = (frames.anchor_to_world[1] * right.tile_to_world)
            .try_inverse()
            .unwrap()
            * raw_relation
            * frames.anchor_to_world[0]
            * left.tile_to_world;
        assert!((transformed_point(&wrong, point).unwrap() - expected).norm() > 100.0);
    }
}

#[test]
fn source_seed_rejects_unmeasured_bridges_and_invalid_frames() {
    let images = [focus_test_image(0, "a.raw"), focus_test_image(1, "b.raw")];
    let poses = HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())]);
    let geometry = FocusVirtualTileGeometry {
        width: 1000,
        height: 1000,
        tile_to_world: Matrix3::identity(),
    };
    let mut relation = observed_translation_match(100.0, 0.0, 36);
    for sequence in [true, false] {
        relation.sequence_bridge = sequence;
        relation.coarse_bridge = !sequence;
        let matches = HashMap::from([((0, 1), relation.clone())]);
        assert!(
            virtual_tile_source_match_proposals(
                &[&images[0]],
                &[&images[1]],
                &matches,
                &poses,
                &geometry,
                &geometry,
                1.0
            )
            .is_empty()
        );
    }
    relation.sequence_bridge = false;
    relation.coarse_bridge = false;
    let matches = HashMap::from([((0, 1), relation)]);
    for scale in [0.0, -1.0, f64::NAN] {
        assert!(
            virtual_tile_source_match_proposals(
                &[&images[0]],
                &[&images[1]],
                &matches,
                &poses,
                &geometry,
                &geometry,
                scale
            )
            .is_empty()
        );
    }
    let singular = FocusVirtualTileGeometry {
        tile_to_world: Matrix3::zeros(),
        ..geometry
    };
    assert!(
        virtual_tile_source_match_proposals(
            &[&images[0]],
            &[&images[1]],
            &matches,
            &poses,
            &geometry,
            &singular,
            1.0
        )
        .is_empty()
    );
}

#[test]
fn source_seed_candidate_history_keeps_earlier_fits_flat_and_round_trips() {
    let (_, mut first) = virtual_tile_direct_station_match(0, 1, &[], &[], &HashMap::new(), 3.0);
    first.seed = "rendering_prior".into();
    first.model_fitted = true;
    first.fitted_inliers = 471;
    let mut records = vec![first.clone()];
    for (seed, count) in [
        ("source_pairwise_match", 397),
        ("global_coverage_similarity", 0),
    ] {
        let mut next = first.clone();
        next.left = 1;
        next.right = 0;
        next.seed = seed.into();
        next.fitted_inliers = count;
        next.model_fitted = count > 0;
        replace_virtual_tile_candidate_record(&mut records, next);
    }
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].seed, "global_coverage_similarity");
    let history = &records[0].previous_attempts;
    assert_eq!(
        history.iter().map(|r| r.fitted_inliers).collect::<Vec<_>>(),
        vec![471, 397]
    );
    assert!(history.iter().all(|r| r.previous_attempts.is_empty()));
    let mut report = stack_report::StationRelationsReport {
        candidates: records.clone(),
        ..Default::default()
    };
    report.summarize_candidate_measurements();
    assert_eq!(report.unmeasured_count, 0);
    assert_eq!(
        report.rejected_by_failure_stage["measured_relation_not_accepted"],
        1
    );
    let value = serde_json::to_value(&records[0]).unwrap();
    let restored: stack_report::StationRelationCandidateRecord =
        serde_json::from_value(value).unwrap();
    assert_eq!(restored, records[0]);
    // Old reports without history still deserialize, and a single attempt
    // adds no field to the already accepted nine-NEF report.
    let old = serde_json::to_value(&first).unwrap();
    assert!(old.get("previous_attempts").is_none());
    assert!(
        serde_json::from_value::<stack_report::StationRelationCandidateRecord>(old)
            .unwrap()
            .previous_attempts
            .is_empty()
    );
}
