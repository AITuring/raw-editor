use super::*;

fn candidate(path: &str, sharpness: f64) -> AnchorCandidate {
    AnchorCandidate {
        path: path.to_string(),
        median_sharpness: sharpness,
    }
}

#[test]
fn anchor_is_the_highest_median_sharpness_frame() {
    let candidates = vec![
        candidate("/b.nef", 0.20),
        candidate("/a.nef", 0.50),
        candidate("/c.nef", 0.35),
    ];
    assert_eq!(select_anchor(&candidates), Some(1));
}

#[test]
fn near_equal_scores_are_resolved_by_ascending_absolute_path() {
    // Within the 0.01 window, so the lowest path wins even though it is
    // neither the first candidate nor the numerically highest score.
    let candidates = vec![
        candidate("/z/0009.nef", 0.5000),
        candidate("/a/0001.nef", 0.4995),
        candidate("/m/0005.nef", 0.4999),
    ];
    assert_eq!(select_anchor(&candidates), Some(1));

    // Outside the window the sharpest frame keeps the anchor role.
    let candidates = vec![
        candidate("/z/0009.nef", 0.5000),
        candidate("/a/0001.nef", 0.4800),
    ];
    assert_eq!(select_anchor(&candidates), Some(0));
}

#[test]
fn anchor_selection_is_independent_of_candidate_order() {
    let mut candidates = vec![
        candidate("/a.nef", 0.10),
        candidate("/b.nef", 0.42),
        candidate("/c.nef", 0.30),
    ];
    let chosen = candidates[select_anchor(&candidates).unwrap()].path.clone();
    candidates.reverse();
    assert_eq!(candidates[select_anchor(&candidates).unwrap()].path, chosen);
}

#[test]
fn empty_and_unmeasurable_candidate_sets_are_handled() {
    assert_eq!(select_anchor(&[]), None);
    let candidates = vec![candidate("/b.nef", f64::NAN), candidate("/a.nef", f64::NAN)];
    // No measurable score: the lowest absolute path is still deterministic.
    assert_eq!(select_anchor(&candidates), Some(1));
}

#[test]
fn task_7_31_patch_geometry_satisfies_window_and_search_bounds() {
    // 需求 2.3: window side ≥ 32 native pixels at the finest pass.
    const { assert!(2 * INTRA_STATION_MATCH_RADIUS + 1 >= 32) };
    // 需求 2.3: search reach ≤ 64 native pixels, and inside our own bound.
    assert!((native_search_reach_px(2.0) - 64.0).abs() < 1e-12);
    assert!(native_search_reach_px(2.0) <= INTRA_STATION_SEARCH_RADIUS);
    const { assert!(INTRA_STATION_SEARCH_RADIUS <= 64.0) };
    // The regression this guards: the matcher needs the window and the whole
    // search support inside the patch buffer, so a "radius" of 64 cannot be
    // matched at all.
    assert!(patch_support_fits(
        INTRA_STATION_MATCH_RADIUS,
        INTRA_STATION_SEARCH_SAMPLES,
        INTRA_STATION_PATCH_HALF
    ));
    assert!(!patch_support_fits(
        INTRA_STATION_SEARCH_RADIUS as i32,
        INTRA_STATION_SEARCH_SAMPLES,
        INTRA_STATION_PATCH_HALF
    ));
    // The legacy mosaic geometry is untouched.
    assert_eq!(LEGACY_MOSAIC_PATCH_HALF, 28);
    assert!(patch_support_fits(10, 7, LEGACY_MOSAIC_PATCH_HALF));
}

#[test]
fn control_point_gates_match_the_requirement_thresholds() {
    // 需求 2.4 allows 1.0 native pixels, and the intra-station path uses all
    // of it: a defocused frame cannot be localised to a fraction of that.
    assert!((INTRA_STATION_BIDIRECTIONAL_TOLERANCE_PX - 1.00).abs() < 1e-12);
    assert!(bidirectional_consistent(1.00));
    assert!(!bidirectional_consistent(1.01));
    assert!(!bidirectional_consistent(f64::NAN));

    // 需求 2.6 on a 6000px anchor: the limit is 60 native pixels.
    assert!(symmetric_error_accepted(60.0, 6_000));
    assert!(!symmetric_error_accepted(60.1, 6_000));

    assert!(neighbour_median_consistent([3.0, 0.0], None));
    assert!(neighbour_median_consistent([4.0, 0.0], Some([0.0, 0.0])));
    assert!(!neighbour_median_consistent([4.1, 0.0], Some([0.0, 0.0])));
}

#[test]
fn neighbour_median_uses_the_accepted_eight_neighbourhood_only() {
    let mut grid = ControlPointGrid::new(3, 3);
    assert_eq!(grid.neighbour_median(1, 1), None);
    grid.accept(0, 0, [1.0, -1.0]);
    grid.accept(1, 0, [3.0, -3.0]);
    grid.accept(2, 2, [99.0, 99.0]);
    // (2, 2) is not in the 8-neighbourhood of (0, 1).
    assert_eq!(grid.neighbour_median(0, 1), Some([1.0, -3.0]));
    assert_eq!(grid.accepted_count(), 3);
}

#[test]
fn coverage_and_status_follow_the_requirement_definitions() {
    // 40 cells of 16x16 native pixels over a 256x256 anchor area.
    let coverage = inlier_area_coverage(40, 256.0, 65_536.0);
    assert!((coverage - 0.15625).abs() < 1e-12);
    assert_eq!(
        frame_status(true, coverage),
        IntraStationFrameStatus::Failed,
        "coverage below 0.20 fails the frame regardless of the local field"
    );
    assert_eq!(
        frame_status(true, 0.20),
        IntraStationFrameStatus::Local,
        "0.20 is inside the accepted range"
    );
    assert_eq!(
        frame_status(false, 0.75),
        IntraStationFrameStatus::GlobalFallback
    );
    assert_eq!(inlier_area_coverage(10, 16.0, 0.0), 0.0);
}

#[test]
fn local_field_is_kept_only_when_it_lowers_the_median_error() {
    assert!(local_field_improves(0.9, 1.0));
    assert!(!local_field_improves(1.0, 1.0));
    assert!(!local_field_improves(1.1, 1.0));
    assert!(!local_field_improves(f64::NAN, 1.0));
}

/// 需求 2.7 compares two medians.  An absent median is not `0.00px`, and the
/// regression this guards produced exactly that: a coarse pass that accepted
/// too few control points left the baseline empty, `median_of(&[])` returned
/// `0.0`, and every single frame was then reverted for "not being better
/// than the global model's 0.00px".
#[test]
fn an_absent_baseline_is_indeterminate_and_never_a_revert() {
    let many = INTRA_STATION_MIN_BASELINE_SAMPLES;
    assert_eq!(
        local_field_verdict(0.30, many, 0.00, 0),
        LocalFieldVerdict::Indeterminate,
        "an empty global baseline cannot make a refinement look worse"
    );
    assert_eq!(
        local_field_verdict(0.30, many, 0.50, many - 1),
        LocalFieldVerdict::Indeterminate,
        "a baseline below the median's own sample floor is not a value either"
    );
    assert_eq!(
        local_field_verdict(0.30, 0, 0.50, many),
        LocalFieldVerdict::Indeterminate
    );
    assert!(
        local_field_verdict(0.30, many, 0.00, 0).field_kept(),
        "an unevaluable comparison keeps the measured field"
    );

    // With both medians formed, 需求 2.7 applies exactly as written.
    assert_eq!(
        local_field_verdict(0.30, many, 0.50, many),
        LocalFieldVerdict::Kept
    );
    assert_eq!(
        local_field_verdict(0.50, many, 0.50, many),
        LocalFieldVerdict::Reverted,
        "需求 2.7 reverts a refinement that is not better, not merely a worse one"
    );
    assert!(!local_field_verdict(0.50, many, 0.50, many).field_kept());
}

fn checkerboard_patch(side: usize, period: f64, amplitude: f64) -> Vec<f64> {
    (0..side * side)
        .map(|index| {
            let x = (index % side) as f64;
            let y = (index / side) as f64;
            let phase = std::f64::consts::TAU / period;
            128.0 + amplitude * ((x * phase).sin() + (y * phase * 0.7).cos())
        })
        .collect()
}

#[test]
fn a_three_tap_low_pass_removes_high_frequency_energy_monotonically() {
    let side = (2 * INTRA_STATION_PATCH_HALF + 1) as usize;
    let mut patch = checkerboard_patch(side, 4.0, 40.0);
    let mut scratch = vec![0.0; side * side];
    let mut energy = patch_high_frequency_energy(&patch, side);
    assert!(energy > 0.0);
    for _ in 0..6 {
        blur_variance_in_place(&mut patch, side, 0.5, &mut scratch);
        let next = patch_high_frequency_energy(&patch, side);
        assert!(next < energy, "{next} should be below {energy}");
        energy = next;
    }
    // A zero-variance step is the identity, so the parameter really is the
    // amount of blur and not a pass count.
    let before = patch.clone();
    blur_variance_in_place(&mut patch, side, 0.0, &mut scratch);
    assert_eq!(patch, before);
}

/// The point of the matched low-pass: a sharp patch and its own defocused
/// counterpart must end up at a comparable scale, and two patches already at
/// the same scale must be left alone.
#[test]
fn the_focal_plane_match_brings_a_defocused_pair_to_a_comparable_scale() {
    let side = (2 * INTRA_STATION_PATCH_HALF + 1) as usize;
    let sharp = checkerboard_patch(side, 5.0, 40.0);
    let mut defocused = sharp.clone();
    let mut scratch = vec![0.0; side * side];
    for _ in 0..4 {
        blur_variance_in_place(&mut defocused, side, 0.5, &mut scratch);
    }

    let mut left = sharp.clone();
    let mut right = defocused.clone();
    let matched = match_focal_plane(&mut left, &mut right, side);
    assert!(
        matched.sharpness_ratio_before < 0.5,
        "the fixture must start far from parity ({})",
        matched.sharpness_ratio_before
    );
    assert!(
        matched.sharpness_ratio_after > 0.9,
        "the two sides must end up comparable ({})",
        matched.sharpness_ratio_after
    );
    assert!(
        matched.matched_variance > 0.0 && !matched.capped,
        "{matched:?}"
    );

    // Two identical patches need no matching blur at all.
    let mut a = sharp.clone();
    let mut b = sharp.clone();
    let same = match_focal_plane(&mut a, &mut b, side);
    assert_eq!(same.matched_variance, 0.0);
    assert!((same.sharpness_ratio_after - 1.0).abs() < 1e-9);
    assert_eq!(a, b, "the same filter is applied to both sides");
}

/// The matched low-pass must not manufacture agreement: two unrelated
/// patches stay uncorrelated after it.
#[test]
fn the_focal_plane_match_does_not_make_unrelated_content_agree() {
    let side = (2 * INTRA_STATION_PATCH_HALF + 1) as usize;
    let mut left = checkerboard_patch(side, 5.0, 40.0);
    let mut right = checkerboard_patch(side, 11.0, 40.0)
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    match_focal_plane(&mut left, &mut right, side);
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (mean_left, mean_right) = (mean(&left), mean(&right));
    let mut product = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (a, b) in left.iter().zip(right.iter()) {
        product += (a - mean_left) * (b - mean_right);
        left_variance += (a - mean_left).powi(2);
        right_variance += (b - mean_right).powi(2);
    }
    let correlation = product / (left_variance * right_variance).sqrt();
    assert!(
        correlation.abs() < INTRA_STATION_MIN_CORRELATION,
        "unrelated content must stay below the similarity floor ({correlation})"
    );
}

/// 需求 2.8 counts the control point cells that contain an accepted local
/// match, over the whole refinement rather than over its last spacing pass.
#[test]
fn accepted_cells_accumulate_over_the_refinement_passes() {
    let mut union = ControlPointGrid::new(4, 4);
    let mut coarse = ControlPointGrid::new(4, 4);
    coarse.accept(0, 0, [1.0, 0.0]);
    coarse.accept(1, 1, [1.0, 0.0]);
    let mut fine = ControlPointGrid::new(4, 4);
    fine.accept(1, 1, [0.1, 0.0]);
    fine.accept(3, 3, [0.1, 0.0]);
    union.absorb(&coarse);
    union.absorb(&fine);
    assert_eq!(union.accepted_count(), 3);
    // A grid of another shape is ignored rather than silently misaligned.
    union.absorb(&ControlPointGrid::new(2, 2));
    assert_eq!(union.accepted_count(), 3);
}

#[test]
fn task_7_31_analysis_budget_and_report_use_actual_configured_value() {
    assert_eq!(INTRA_STATION_ANALYSIS_LONG_SIDE, 1_400);
    assert!(INTRA_STATION_ANALYSIS_LONG_SIDE <= INTRA_STATION_ANALYSIS_MAX_LONG_SIDE);
    assert!((analysis_probe_spacing(1_024) - 1.0).abs() < 1e-12);
    assert!((analysis_probe_spacing(2_800) - 2.0).abs() < 1e-12);
    // A probe grid of 48 samples stays well inside a small analysis image.
    let measured = median_valid_sharpness(64, 64, |_, _| Some(0.25));
    assert!((measured - 0.25).abs() < 1e-12);
    assert_eq!(median_valid_sharpness(4, 4, |_, _| Some(1.0)), 0.0);

    let mut records = Vec::new();
    push_frame_record(
        &mut records,
        0,
        "/anchor.nef",
        IntraStationFrameRecord::default(),
    );
    assert_eq!(
        records[0].analysis_long_side,
        INTRA_STATION_ANALYSIS_LONG_SIDE
    );
    assert_ne!(
        records[0].analysis_long_side,
        INTRA_STATION_ANALYSIS_MAX_LONG_SIDE
    );
}

#[test]
fn task_7_31_measured_control_point_spacing_obeys_bounded_rule() {
    use crate::panorama_utils::mosaic::native_control_point_spacing_px;
    use crate::panorama_utils::stack_pipeline::focus_fuser::ownership_cell_size_px;

    // Production dimensions measured on the supported camera fixtures.
    for (long_side, expected_cell) in [(7_088, 13), (8_256, 16), (9_504, 18)] {
        let analysis_scale =
            (f64::from(long_side) / f64::from(INTRA_STATION_ANALYSIS_LONG_SIDE)).max(1.0);
        let spacing = native_control_point_spacing_px(analysis_scale);
        let cell = ownership_cell_size_px(long_side);
        assert_eq!(cell, expected_cell);
        assert!(
            control_point_spacing_is_bounded(spacing, cell),
            "{long_side}px produced spacing {spacing:.3}px for a {cell}px ownership cell"
        );
    }

    let measured = [
        native_control_point_spacing_px(7_088.0 / 1_400.0),
        native_control_point_spacing_px(8_256.0 / 1_400.0),
        native_control_point_spacing_px(9_504.0 / 1_400.0),
    ];
    assert!((measured[0] - 81.005_714_285_714_29).abs() < 1e-9);
    assert!((measured[1] - 94.354_285_714_285_71).abs() < 1e-9);
    assert!((measured[2] - 108.617_142_857_142_85).abs() < 1e-9);

    // Both clauses are independently effective: one ratio violation still
    // fits under 112px, and one absolute-cap violation still fits within 8c.
    assert!(!control_point_spacing_is_bounded(104.1, 13));
    assert!(!control_point_spacing_is_bounded(112.1, 18));
    assert!(!control_point_spacing_is_bounded(f64::NAN, 18));
}

#[test]
fn frame_records_are_grouped_per_station() {
    // Driven on a local collection: `cargo test` runs the mosaic tests
    // concurrently in the same process, so an assertion on the run scoped
    // sink would depend on unrelated tests.
    let mut records = Vec::new();
    let frame = |path: &str| IntraStationFrameRecord {
        path: path.to_string(),
        ..Default::default()
    };
    push_frame_record(&mut records, 1, "/station-1-anchor.nef", frame("/b.nef"));
    push_frame_record(&mut records, 0, "/station-0-anchor.nef", frame("/a.nef"));
    push_frame_record(&mut records, 1, "/station-1-anchor.nef", frame("/c.nef"));
    records.sort_by_key(|station| station.station_index);
    let stations = records
        .iter()
        .map(|station| (station.station_index, station.frames.len()))
        .collect::<Vec<_>>();
    assert_eq!(stations, vec![(0, 1), (1, 2)]);
    assert_eq!(
        records[1].analysis_long_side,
        INTRA_STATION_ANALYSIS_LONG_SIDE
    );
    assert_eq!(records[1].anchor_path, "/station-1-anchor.nef");
}
