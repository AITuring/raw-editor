//! Intra-station frame records and Station_Relation record compatibility: a re-measured frame
//! replaces its record in place, and records without the checkpoint-10 fields deserialise.

use super::*;

// ---------------------------------------------------------------------------
// 9. intra-station frame records
// ---------------------------------------------------------------------------

#[test]
fn push_frame_record_replaces_a_re_measured_frame_in_place() {
    let frame = |path: &str, inliers: usize| report::IntraStationFrameRecord {
        path: path.to_string(),
        inliers,
        ..report::IntraStationFrameRecord::default()
    };
    let summary = |station: &report::IntraStationReport| {
        station
            .frames
            .iter()
            .map(|frame| (frame.path.clone(), frame.inliers))
            .collect::<Vec<_>>()
    };
    let mut records = Vec::new();
    intra_station::push_frame_record(&mut records, 3, "anchor-a.raw", frame("a.raw", 10));
    intra_station::push_frame_record(&mut records, 3, "anchor-a.raw", frame("b.raw", 20));
    intra_station::push_frame_record(&mut records, 5, "anchor-c.raw", frame("c.raw", 30));
    // A later render re-measures a.raw: replaced in place, anchor unchanged.
    intra_station::push_frame_record(&mut records, 3, "other-anchor.raw", frame("a.raw", 11));
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].station_index, 3);
    assert_eq!(records[0].anchor_path, "anchor-a.raw");
    assert_eq!(
        summary(&records[0]),
        vec![("a.raw".to_string(), 11), ("b.raw".to_string(), 20)]
    );
    intra_station::push_frame_record(&mut records, 3, "anchor-a.raw", frame("d.raw", 40));
    assert_eq!(
        summary(&records[0]),
        vec![
            ("a.raw".to_string(), 11),
            ("b.raw".to_string(), 20),
            ("d.raw".to_string(), 40)
        ]
    );
    assert_eq!(records[1].station_index, 5);
    assert_eq!(summary(&records[1]), vec![("c.raw".to_string(), 30)]);

    // An empty anchor is filled by the first record that names one.
    intra_station::push_frame_record(&mut records, 7, "", frame("e.raw", 1));
    assert_eq!(records[2].anchor_path, "");
    intra_station::push_frame_record(&mut records, 7, "anchor-e.raw", frame("e.raw", 2));
    assert_eq!(records[2].anchor_path, "anchor-e.raw");
    assert_eq!(summary(&records[2]), vec![("e.raw".to_string(), 2)]);
}

#[test]
fn station_relation_records_deserialise_without_checkpoint_ten_fields() {
    let candidate = sample_candidate_record();
    let value = serde_json::to_value(&candidate).unwrap();
    assert_eq!(
        serde_json::from_value::<report::StationRelationCandidateRecord>(value.clone()).unwrap(),
        candidate
    );
    let mut legacy = value;
    let object = legacy.as_object_mut().unwrap();
    for key in [
        "predicted_overlap_area_px",
        "model_fitted",
        "left_tile",
        "right_tile",
        "pyramid_factors",
        "measurement_factor",
        "seed",
        "prior_free_inliers",
        "global_proposals",
        "after_prior_repair",
        "hypotheses",
    ] {
        assert!(object.remove(key).is_some(), "{key} is serialised");
    }
    let parsed = serde_json::from_value::<report::StationRelationCandidateRecord>(legacy).unwrap();
    assert_eq!(
        parsed,
        report::StationRelationCandidateRecord {
            predicted_overlap_area_px: 0.0,
            model_fitted: false,
            left_tile: Default::default(),
            right_tile: Default::default(),
            pyramid_factors: Vec::new(),
            measurement_factor: 0,
            seed: String::new(),
            prior_free_inliers: 0,
            global_proposals: Vec::new(),
            after_prior_repair: false,
            hypotheses: Vec::new(),
            ..candidate.clone()
        }
    );

    let repair = report::StationPriorRepairRecord {
        station: 2,
        fixed_station: 1,
        relation_left: 1,
        relation_right: 2,
        relation_inliers: 64,
        center_shift_px: 12.5,
        correction_scale_ratio: 1.004,
    };
    let text = serde_json::to_string(&repair).unwrap();
    assert_eq!(
        serde_json::from_str::<report::StationPriorRepairRecord>(&text).unwrap(),
        repair
    );

    let source_raw_diagnostic = report::SourceRawPairDiagnosticRecord {
        left: 1,
        right: 2,
        layer_models: vec![source_raw_test_record(3, 7)],
        consensus_clusters: vec![report::SourceRawConsensusClusterRecord {
            cluster: 0,
            observation_count: 2,
            independent_support: 2,
            supporting_layer_pairs: vec![["left-3".to_string(), "right-7".to_string()]],
            median_disagreement_px: 0.5,
            scale_ratio: 1.0,
            spatial_support_potential: 0.4,
            scale_compliant: true,
            spatially_broad: true,
            anchor_transform: matrix3_row_major(&Matrix3::identity()),
        }],
    };
    let relations = report::StationRelationsReport {
        candidates: vec![candidate],
        source_raw_pair_diagnostics: vec![source_raw_diagnostic],
        prior_repairs: vec![repair],
        ..report::StationRelationsReport::default()
    };
    let value = serde_json::to_value(&relations).unwrap();
    assert_eq!(
        serde_json::from_value::<report::StationRelationsReport>(value.clone()).unwrap(),
        relations
    );
    let mut legacy = value;
    legacy.as_object_mut().unwrap().remove("prior_repairs");
    legacy
        .as_object_mut()
        .unwrap()
        .remove("source_raw_pair_diagnostics");
    assert_eq!(
        serde_json::from_value::<report::StationRelationsReport>(legacy).unwrap(),
        report::StationRelationsReport {
            source_raw_pair_diagnostics: Vec::new(),
            prior_repairs: Vec::new(),
            ..relations
        }
    );
}
