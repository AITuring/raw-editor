use super::*;
use crate::panorama_utils::stack_pipeline::degradation::INTRA_STATION_REGISTRATION_FAILED;
use crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode;

fn station_frame(id: usize) -> ImageInfo {
    ImageInfo {
        id,
        filename: format!("/station/000{id}.nef"),
        width: 6_000,
        height: 4_000,
        alignment_image: GrayImage::new(1, 1),
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

fn record(
    path: &str,
    status: IntraStationFrameStatus,
    inliers: usize,
    coverage: f64,
    error_px: f64,
) -> (String, IntraStationFrameRecord) {
    (
        path.to_string(),
        IntraStationFrameRecord {
            path: path.to_string(),
            inliers,
            inlier_area_coverage: coverage,
            median_symmetric_error_px: error_px,
            status,
            ..Default::default()
        },
    )
}

#[test]
fn a_station_with_no_registration_record_fuses_every_frame_in_order() {
    // This is the no-change guarantee of the wiring: the default path does
    // not run the native patch refinement, so it records nothing, so the
    // plan is the full bracket and the render loop is exactly what it was.
    let frames = (1..=4).map(station_frame).collect::<Vec<_>>();
    let images = frames.iter().collect::<Vec<_>>();
    let (members, rejections) = station_members_from_frame_records(&images, &HashMap::new());
    assert!(rejections.is_empty());
    assert_eq!(members[0].status, IntraStationFrameStatus::Anchor);
    assert!(
        members[1..]
            .iter()
            .all(|member| member.status == IntraStationFrameStatus::GlobalFallback),
        "a frame placed by its global model alone is a global fallback, not a failure"
    );
    let plan = plan_station_fusion(&members);
    assert_eq!(plan.mode, StationFusionMode::AllFrames);
    assert_eq!(plan.fused, vec![0, 1, 2, 3]);
    assert!(plan.excluded.is_empty());
    assert!(station_plan_entries(0, &members, &plan).is_empty());
}

#[test]
fn a_recorded_failure_beside_two_survivors_leaves_the_fusion_to_the_survivors() {
    // 需求 12.2.
    let frames = (1..=4).map(station_frame).collect::<Vec<_>>();
    let images = frames.iter().collect::<Vec<_>>();
    let recorded = HashMap::from([
        record(
            "/station/0002.nef",
            IntraStationFrameStatus::Failed,
            0,
            0.05,
            0.4,
        ),
        record(
            "/station/0003.nef",
            IntraStationFrameStatus::Local,
            900,
            0.80,
            0.4,
        ),
    ]);
    let (members, rejections) = station_members_from_frame_records(&images, &recorded);
    assert!(
        rejections.is_empty(),
        "no accepted inlier means no measured 需求 12.5 evidence to judge"
    );
    let plan = plan_station_fusion(&members);
    assert_eq!(plan.mode, StationFusionMode::SuccessfulFramesOnly);
    assert_eq!(
        plan.fused,
        vec![0, 2, 3],
        "the excluded frame is never offered to the cut"
    );
    assert_eq!(
        plan.excluded
            .iter()
            .map(|record| (record.path.as_str(), record.reason.as_str()))
            .collect::<Vec<_>>(),
        vec![("/station/0002.nef", INTRA_STATION_REGISTRATION_FAILED)],
        "需求 12.2 names the excluded Source_RAW by absolute path"
    );
    let entries = station_plan_entries(7, &members, &plan);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, INTRA_STATION_REGISTRATION_FAILED);
    assert_eq!(entries[0].1["station_index"], serde_json::json!(7));
    assert_eq!(entries[0].1["fused_frames"], serde_json::json!(3));
}

#[test]
fn every_non_anchor_frame_failing_degrades_the_station_to_one_source() {
    // 需求 12.1: the plan keeps one member, so the fusion is constructed with
    // one frame and reports `SingleFrame`, and every covered pixel is copied
    // from that Source_RAW.
    let frames = (1..=3).map(station_frame).collect::<Vec<_>>();
    let images = frames.iter().collect::<Vec<_>>();
    let recorded = HashMap::from([
        record(
            "/station/0002.nef",
            IntraStationFrameStatus::Failed,
            0,
            0.01,
            9.0,
        ),
        record(
            "/station/0003.nef",
            IntraStationFrameStatus::Failed,
            0,
            0.02,
            9.0,
        ),
    ]);
    let (members, _) = station_members_from_frame_records(&images, &recorded);
    let plan = plan_station_fusion(&members);
    assert_eq!(plan.mode, StationFusionMode::SingleFrameDegraded);
    assert_eq!(plan.fused, vec![0]);
    assert_eq!(
        plan.single_frame_path(&members).as_deref(),
        Some("/station/0001.nef")
    );
    assert_eq!(plan.excluded.len(), 2);
    assert_eq!(
        focus_fuser::StationFusion::new(64, 64, plan.fused.len()).solver_status(),
        crate::panorama_utils::stack_pipeline::report::FusionSolverStatus::SingleFrame,
        "需求 12.1 / 3.10: one frame has nothing to cut against"
    );
}

#[test]
fn score_noise_diagnostics_preserve_production_probe_aggregation_and_window() {
    let geometry = OwnershipGridGeometry::for_plane(128, 128);
    let plan = geometry.sampling_plan();
    let image = Rgb32FImage::from_fn(128, 128, |x, y| {
        let edge = if x > 63 { 0.82 } else { 0.18 };
        let texture = ((x * 17 + y * 29) % 19) as f32 / 500.0;
        Rgb([edge + texture, edge * 0.9, edge * 0.8])
    });
    let mask = GrayImage::from_pixel(128, 128, image::Luma([255]));
    let origin = (56.0, 56.0);
    let production = plan.measure(origin.0, origin.1, |x, y| {
        sample_focus_plane(&image, &mask, 0, 0, x, y)
    });
    let probes = score_noise_probe_values(&plan, origin.0, origin.1, (0.0, 0.0), |x, y| {
        sample_focus_plane(&image, &mask, 0, 0, x, y)
    });
    let diagnostic = focus_fuser::normalized_sharpness(score_noise_aggregate(&probes));

    assert_eq!(plan.window_px(), focus_fuser::SHARPNESS_SAMPLE_WINDOW_PX);
    assert_eq!(production, diagnostic);
    assert_eq!(
        probes,
        score_noise_probe_values(&plan, origin.0, origin.1, (0.0, 0.0), |x, y| {
            sample_focus_plane(&image, &mask, 0, 0, x, y)
        }),
        "repeated diagnostics must be bit deterministic"
    );
    assert!(SCORE_NOISE_PHASE_OFFSETS.contains(&(0.0, 0.0)));
}

#[test]
fn score_noise_primary_attribution_keeps_direct_pairwise_cause_distinct_from_signals() {
    assert_eq!(
        score_noise_primary_attribution(true, true, true, true, true, true),
        Some("pairwise_smoothness_override"),
        "a phase-unstable cell still has a direct pairwise cause when its data-term owner differs"
    );
    assert_eq!(
        score_noise_primary_attribution(true, false, false, true, false, false),
        Some("sampling_phase_instability")
    );
    assert_eq!(
        score_noise_primary_attribution(false, true, true, true, true, true),
        None,
        "non-shortfall cells must not inflate causal counts"
    );
}

#[test]
fn score_noise_winner_runs_use_sharpness_winners_not_production_owners() {
    let candidates = vec![vec![
        ScoreNoiseCandidateCell {
            label: 0,
            covered: true,
            normalized_sharpness: 0.4,
            ..Default::default()
        },
        ScoreNoiseCandidateCell {
            label: 1,
            covered: true,
            normalized_sharpness: 0.8,
            ..Default::default()
        },
    ]];
    assert_eq!(
        score_noise_winner_owners(&candidates),
        vec![focus_fuser::owner_of(1)]
    );
}

#[test]
fn score_noise_region_and_roi_diagnostics_are_deterministic_and_detect_fragmentation() {
    let geometry = OwnershipGridGeometry::for_plane(1024, 1024);
    let mut coherent = vec![focus_fuser::owner_of(0); geometry.cell_count()];
    let first = score_noise_owner_regions(&geometry, &coherent);
    let second = score_noise_owner_regions(&geometry, &coherent);
    assert_eq!(first, second);
    assert!(
        first
            .0
            .iter()
            .all(|&size| size == geometry.cell_count() as u64)
    );
    assert_eq!(
        score_noise_single_owner_rois((1024, 1024), &geometry, &coherent),
        4,
        "a coherent owner must preserve all four aligned 512px ROIs"
    );

    for row in 0..geometry.rows as usize {
        for column in 0..geometry.columns as usize {
            coherent[row * geometry.columns as usize + column] =
                focus_fuser::owner_of((row + column) % 2);
        }
    }
    assert_eq!(
        score_noise_single_owner_rois((1024, 1024), &geometry, &coherent),
        0,
        "cell-scale owner noise must not be mistaken for preserved ROI evidence"
    );
}

#[test]
fn recorded_evidence_below_the_group_join_limits_rejects_the_frame() {
    // 需求 12.5, on a 6000px anchor: support below 0.20, or a median
    // symmetric error above 60px, keeps the Source_RAW out of the station.
    let frames = (1..=3).map(station_frame).collect::<Vec<_>>();
    let images = frames.iter().collect::<Vec<_>>();
    let recorded = HashMap::from([
        record(
            "/station/0002.nef",
            IntraStationFrameStatus::Local,
            120,
            0.1999,
            0.5,
        ),
        record(
            "/station/0003.nef",
            IntraStationFrameStatus::Local,
            900,
            0.85,
            60.1,
        ),
    ]);
    let (members, rejections) = station_members_from_frame_records(&images, &recorded);
    assert_eq!(rejections.len(), 2);
    assert!(rejections[0].support_below_minimum && !rejections[0].error_above_limit);
    assert!(rejections[1].error_above_limit && !rejections[1].support_below_minimum);
    assert!((rejections[1].symmetric_error_limit_px - 60.0).abs() < 1e-12);
    assert!(
        members[1..]
            .iter()
            .all(|member| member.status == IntraStationFrameStatus::Failed),
        "a frame that joins no Capture_Station cannot be fused by this one"
    );
    // A frame exactly at both limits still joins, so the thresholds are not
    // loosened in either direction.
    let recorded = HashMap::from([
        record(
            "/station/0002.nef",
            IntraStationFrameStatus::Local,
            120,
            0.20,
            60.0,
        ),
        record(
            "/station/0003.nef",
            IntraStationFrameStatus::Local,
            900,
            0.85,
            0.5,
        ),
    ]);
    let (members, rejections) = station_members_from_frame_records(&images, &recorded);
    assert!(rejections.is_empty());
    assert_eq!(
        plan_station_fusion(&members).mode,
        StationFusionMode::AllFrames
    );
}
