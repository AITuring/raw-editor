//! Sharpness_Score and the ownership cell metrics (需求 3.1, 3.2, 3.3, 3.9).
//!
//! The normalised score and its 32-native-pixel window, the cell size band and
//! grid geometry, the sampling plan shared by candidate and composite, and the
//! median cell disagreement with its mismatch penalty.

use super::*;

fn flat(value: f32) -> impl FnMut(f64, f64) -> Option<Rgb<f32>> {
    move |_, _| Some(Rgb([value, value, value]))
}

/// 需求 3.9: the Sharpness_Score is in `[0, 1]`, `0` for no measurable
/// gradient, and the mapping is order preserving.
#[test]
fn normalized_sharpness_is_bounded_and_order_preserving() {
    assert_eq!(normalized_sharpness(0.0), 0.0);
    assert_eq!(normalized_sharpness(-1.0), 0.0);
    assert_eq!(normalized_sharpness(f64::NAN), 0.0);
    assert_eq!(normalized_sharpness(SHARPNESS_HALF_SCALE), 0.5);
    assert!(normalized_sharpness(f64::MAX) <= 1.0);
    let mut previous = 0.0;
    for step in 1..2_000 {
        let raw = f64::from(step) * 0.01;
        let value = normalized_sharpness(raw);
        assert!((0.0..=1.0).contains(&value), "{raw} mapped to {value}");
        assert!(value > previous, "{raw} broke monotonicity");
        previous = value;
    }
}

/// 需求 3.2: the 13×13 sampler uses fractional native coordinates to span
/// the required 32-native-pixel window exactly.
#[test]
fn sharpness_window_spans_exactly_thirty_two_native_pixels() {
    assert_eq!(sharpness_window_px(), SHARPNESS_SAMPLE_WINDOW_PX);
    assert_eq!(
        sharpness_sample_step_px(),
        SHARPNESS_SAMPLE_WINDOW_PX / f64::from(ACUTANCE_PATCH_POINTS)
    );
}

/// 需求 3.1: cell side in `[8, 64]`, and at least 512 cells along the long
/// side of any plane large enough to allow it.
#[test]
fn ownership_cell_size_stays_within_the_required_band() {
    for long_side in [0u32, 1, 512, 4_095, 4_096, 9_504, 20_000, 32_768, 100_000] {
        let cell = ownership_cell_size_px(long_side);
        assert!(
            (OWNERSHIP_CELL_MIN_PX..=OWNERSHIP_CELL_MAX_PX).contains(&cell),
            "{long_side} gave a {cell}px cell"
        );
        if long_side >= OWNERSHIP_CELL_MIN_PX * SELECTION_LONG_SIDE {
            assert!(
                long_side.div_ceil(cell) >= SELECTION_LONG_SIDE,
                "{long_side} gave only {} cells",
                long_side.div_ceil(cell)
            );
        }
    }
    assert_eq!(ownership_cell_size_px(9_504), 18);
}

#[test]
fn grid_geometry_covers_the_whole_plane() {
    let geometry = OwnershipGridGeometry::for_plane(9_504, 6_336);
    assert_eq!(geometry.cell_size_px, 18);
    assert_eq!(geometry.columns, 528);
    assert_eq!(geometry.rows, 352);
    assert_eq!(geometry.cell_count(), 528 * 352);
    assert!(f64::from(geometry.columns * geometry.cell_size_px) >= 9_504.0);
    assert_eq!(geometry.cell_origin(2, 3), (36.0, 54.0));
}

/// 需求 3.2: a candidate and the current composite must agree on positions
/// and window size; the shared plan is what guarantees it.
#[test]
fn sampling_plan_is_shared_between_candidate_and_composite() {
    let geometry = OwnershipGridGeometry::for_plane(9_504, 6_336);
    let plan = geometry.sampling_plan();
    plan.assert_matches(&geometry.sampling_plan());
    assert_eq!(plan.probes().len(), 5);
    assert!(plan.probes().contains(&(0.0, 0.0)));
    let radius = plan.probes()[0].0.abs();
    assert!(
        plan.probes()
            .iter()
            .all(|(dx, dy)| dx.abs() == radius && dy.abs() == radius || (*dx, *dy) == (0.0, 0.0))
    );
}

#[test]
#[should_panic(expected = "same window size")]
fn mismatched_window_sizes_are_rejected() {
    let mut other = CellSamplingPlan::for_cell(19.0);
    other.step_px = 1.0;
    CellSamplingPlan::for_cell(19.0).assert_matches(&other);
}

#[test]
fn a_focused_cell_scores_above_a_flat_cell() {
    let plan = CellSamplingPlan::for_cell(19.0);
    let flat_score = plan.measure(0.0, 0.0, flat(0.5));
    let edge_score = plan.measure(0.0, 0.0, |x, _| {
        let value = if x.rem_euclid(16.0) < 8.0 { 0.15 } else { 0.85 };
        Some(Rgb([value, value, value]))
    });
    assert_eq!(flat_score, 0.0);
    assert!(edge_score > flat_score);
    assert!((0.0..=1.0).contains(&edge_score));
}

#[test]
fn an_uncovered_cell_scores_zero() {
    let plan = CellSamplingPlan::for_cell(19.0);
    assert_eq!(plan.measure(0.0, 0.0, |_, _| None), 0.0);
}

/// 需求 3.3: identical pixels disagree by 0, a fully displaced contour
/// disagrees strongly, and the value is normalised.
#[test]
fn cell_disagreement_is_a_normalised_median() {
    let identical = cell_disagreement(flat(0.4), flat(0.4));
    assert_eq!(identical, 0.0);
    let offset = cell_disagreement(flat(0.9), flat(0.3));
    assert!((offset - 0.6).abs() < 1e-6, "got {offset}");
    assert!((0.0..=1.0).contains(&offset));
    assert_eq!(cell_disagreement(flat(0.4), |_, _| None), 1.0);
    assert_eq!(cell_disagreement(|_, _| None, flat(0.4)), 1.0);
}

/// The median ignores a minority of differing probes, where the legacy high
/// percentile would have kept them. This is the behaviour change 需求 3.3 asks
/// for: a single stroke crossing one corner of the cell no longer vetoes an
/// otherwise consistent candidate.
#[test]
fn median_disagreement_ignores_a_minority_of_differing_probes() {
    let disagreement = cell_disagreement(
        |x_offset, y_offset| {
            let differs = x_offset < 0.2 && y_offset < 0.2;
            let value = if differs { 0.9 } else { 0.4 };
            Some(Rgb([value, value, value]))
        },
        flat(0.4),
    );
    assert!(disagreement < 0.2, "got {disagreement}");
}

#[test]
fn narrow_contour_prices_a_seam_without_vetoing_the_source() {
    let (median, seam) = cell_disagreement_with_seam_evidence(
        |x, y| Some(Rgb([if x < 0.2 && y < 0.2 { 0.9 } else { 0.4 }; 3])),
        flat(0.4),
    );
    assert_eq!(median, 0.0);
    assert!(seam > 0.1);
    assert_eq!(mismatch_penalty(median, 0.8, 0.4), 0.0);
}

/// 需求 3.3: the penalty is at least 1.0 and only applies to a sharper but
/// inconsistent candidate.
#[test]
fn mismatch_penalty_only_punishes_sharper_inconsistent_candidates() {
    const { assert!(OWNERSHIP_MISMATCH_PENALTY >= 1.0) };
    assert_eq!(mismatch_penalty(0.5, 0.8, 0.4), OWNERSHIP_MISMATCH_PENALTY);
    assert_eq!(mismatch_penalty(0.5, 0.3, 0.4), 0.0);
    assert_eq!(mismatch_penalty(0.2, 0.8, 0.4), 0.0);
    assert_eq!(
        mismatch_penalty(OWNERSHIP_DISAGREEMENT_VETO, 0.8, 0.4),
        0.0,
        "the veto is exclusive"
    );
    assert_eq!(mismatch_penalty(0.5, 0.4, 0.4), 0.0, "a tie is not sharper");
}
