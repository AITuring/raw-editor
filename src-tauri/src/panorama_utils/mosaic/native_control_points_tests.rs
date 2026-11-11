//! Diagnostic accounting must distinguish unmeasured reverse matches from a
//! measured zero-pixel round trip, without changing control-point acceptance.

use super::*;

fn reverse_attempt() -> NativeControlPoint {
    NativeControlPoint {
        covered: true,
        evaluated: true,
        best_correlation: Some(0.95),
        reverse_attempted: true,
        peak_margin: Some(0.08),
        rejection: Some(NativeControlPointRejection::ReverseMatch),
        ..Default::default()
    }
}

fn completed_match(round_trip_px: f64, accepted: bool) -> NativeControlPoint {
    let measured = Some(([0.0, 0.0], 0.95, round_trip_px, 0.0));
    NativeControlPoint {
        round_trip_px,
        round_trip_closed: accepted,
        measured,
        accepted: if accepted { measured } else { None },
        rejection: if accepted {
            None
        } else {
            Some(NativeControlPointRejection::RoundTrip)
        },
        ..reverse_attempt()
    }
}

#[test]
fn failed_reverse_probe_is_unmeasured_instead_of_a_zero_round_trip() {
    let mut stats = SimilarityStats::default();
    let point = reverse_attempt();
    stats.absorb(&point);

    let bucket = similarity_bucket(0.95).unwrap();
    assert_eq!(stats.reverse_attempted, 1);
    assert_eq!(stats.reverse_completed, 0);
    assert_eq!(stats.reverse_rejected, 1);
    assert_eq!(stats.bucket_total[bucket], 1);
    assert_eq!(stats.bucket_completed[bucket], 0);
    assert_eq!(stats.bucket_round_trip[bucket], 0);
    assert!(stats.accepted_round_trips.is_empty());
    assert!(stats.accepted_peak_margins.is_empty());
    assert!(point.accepted.is_none());
    let summary = stats.summary();
    assert!(summary.contains("unmeasured (0 completed matches)"));
    assert!(summary.contains("reverse attempted/completed/rejected 1/0/1"));
    assert!(summary.contains("round trip closed/completed/attempted by bucket"));
    assert!(!summary.contains("within 0.40/0.70/1.00px"));
}

#[test]
fn measured_zero_and_rejected_completed_round_trip_both_enter_distribution() {
    let mut stats = SimilarityStats::default();
    stats.absorb(&reverse_attempt());
    stats.absorb(&completed_match(0.0, true));
    stats.absorb(&completed_match(1.2, false));

    let bucket = similarity_bucket(0.95).unwrap();
    assert_eq!(stats.reverse_attempted, 3);
    assert_eq!(stats.reverse_completed, 2);
    assert_eq!(stats.round_trip_rejected, 1);
    assert_eq!(stats.bucket_total[bucket], 3);
    assert_eq!(stats.bucket_completed[bucket], 2);
    assert_eq!(stats.bucket_round_trip[bucket], 1);
    assert_eq!(stats.accepted_round_trips, vec![0.0, 1.2]);
    assert_eq!(stats.accepted_peak_margins, vec![0.08, 0.08]);
    assert_eq!(stats.accepted_before_neighbour, 1);
    let summary = stats.summary();
    assert!(summary.contains("within 0.40/0.70/1.00px 1/1/1 of 2"));
    assert!(summary.contains("reverse attempted/completed/rejected 3/2/1"));
}

#[test]
fn failed_reverse_probe_keeps_focal_plane_observations() {
    let mut stats = SimilarityStats::default();
    let point = NativeControlPoint {
        focal_plane: Some(intra_station::FocalPlaneMatch {
            matched_variance: 4.0,
            sharpness_ratio_before: 0.4,
            sharpness_ratio_matched: 0.8,
            sharpness_ratio_after: 0.9,
            capped: true,
        }),
        ..reverse_attempt()
    };
    stats.absorb(&point);

    assert_eq!(stats.focal_ratio_before, vec![0.4]);
    assert_eq!(stats.focal_ratio_matched, vec![0.8]);
    assert_eq!(stats.focal_ratio_after, vec![0.9]);
    assert_eq!(stats.focal_passes, vec![2.0]);
    assert_eq!(stats.focal_capped, 1);
    assert!(stats.accepted_round_trips.is_empty());
}

#[test]
fn forward_rejection_categories_distinguish_missing_signal_and_peak_ambiguity() {
    use NativeControlPointRejection as Rejection;
    let classify = |correlation, margin| forward_rejection(correlation, margin, 0.70, 0.03);
    assert_eq!(classify(None, None), Rejection::ForwardUnmeasurable);
    // A probe may fail both floors: the explicit diagnostic precedence is
    // correlation first, rather than claiming the peak alone rejected it.
    assert_eq!(
        classify(Some(0.60), Some(0.01)),
        Rejection::ForwardCorrelation
    );
    assert_eq!(
        classify(Some(0.95), Some(0.01)),
        Rejection::ForwardPeakMargin
    );
    assert_eq!(classify(Some(0.95), Some(0.08)), Rejection::ForwardOther);
}

#[test]
fn rejection_ledger_separates_forward_reverse_round_trip_and_neighbour_gates() {
    use NativeControlPointRejection as Rejection;
    let mut stats = SimilarityStats::default();
    for rejection in [
        Rejection::PatchUnavailable,
        Rejection::ForwardUnmeasurable,
        Rejection::ForwardCorrelation,
        Rejection::ForwardPeakMargin,
        Rejection::ForwardOther,
    ] {
        stats.absorb(&NativeControlPoint {
            rejection: Some(rejection),
            ..Default::default()
        });
    }
    stats.absorb(&reverse_attempt());
    stats.absorb(&completed_match(1.2, false));
    stats.absorb(&NativeControlPoint {
        accepted: None,
        rejection: Some(Rejection::GlobalDisplacement),
        ..completed_match(0.1, true)
    });
    stats.absorb(&completed_match(0.1, true));
    stats.record_neighbour_verdict(false);
    stats.absorb(&completed_match(0.2, true));
    stats.record_neighbour_verdict(true);

    assert_eq!(stats.patch_unavailable, 1);
    assert_eq!(stats.forward_unmeasurable, 1);
    assert_eq!(stats.forward_correlation_rejected, 1);
    assert_eq!(stats.forward_peak_margin_rejected, 1);
    assert_eq!(stats.forward_other_rejected, 1);
    assert_eq!(stats.reverse_rejected, 1);
    assert_eq!(stats.round_trip_rejected, 1);
    assert_eq!(stats.global_displacement_rejected, 1);
    assert_eq!(stats.neighbour_rejected, 1);
    assert_eq!(stats.accepted_before_neighbour, 2);
    assert_eq!(stats.accepted_after_neighbour, 1);
    let summary = stats.summary();
    assert!(summary.contains("forward unmeasurable/correlation/peak/other rejected 1/1/1/1"));
    assert!(summary.contains("reverse attempted/completed/rejected 5/4/1"));
    assert!(summary.contains("roundtrip/global/neighbor rejected 1/1/1"));
    assert!(summary.contains("accepted before/after neighbor 2/1"));
}

#[test]
fn diagnostic_summary_handles_missing_correlation_or_peak_margin() {
    let mut unavailable = SimilarityStats::default();
    unavailable.absorb(&NativeControlPoint {
        rejection: Some(NativeControlPointRejection::PatchUnavailable),
        ..Default::default()
    });
    assert!(
        unavailable
            .summary()
            .contains("no measurable patch pair; patch unavailable 1")
    );

    let mut completed = SimilarityStats::default();
    completed.absorb(&NativeControlPoint {
        peak_margin: None,
        ..completed_match(0.0, true)
    });
    let summary = completed.summary();
    assert!(summary.contains("within 0.40/0.70/1.00px 1/1/1 of 1"));
    assert!(!summary.contains("second-peak margin"));
}
