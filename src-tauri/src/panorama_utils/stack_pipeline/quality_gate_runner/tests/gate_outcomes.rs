//! Criterion boundaries and whole-gate outcomes: owner shortfall quantisation,
//! `run_quality_gate` runs (missing Local_Scale, world-resampled references,
//! record-only mode), export rejection and the unmeasurable reason records.

use super::*;

/// 需求 11.12 judges `shortfall > 0.05` on quantised planes; the
/// quantisation must keep that boundary exact, and the cell definition
/// must give no shortfall to the best owner or to a lone candidate.
#[test]
fn owner_shortfall_quantisation_keeps_the_criterion_boundary() {
    let limit = owner_shortfall_limit();
    assert_eq!(limit, 5);
    assert_eq!(owner_disagreement_veto_limit(), 20);
    assert_eq!(quantize_hundredths(0.05), limit);
    assert!(quantize_hundredths(0.0501) > limit);
    assert!(quantize_hundredths(0.049) <= limit);
    assert_eq!(quantize_hundredths(0.0), 0);
    assert_eq!(quantize_hundredths(1.0), 100);
    assert_eq!(quantize_hundredths(f32::NAN), OWNER_EVIDENCE_UNKNOWN);
    assert_eq!(quantize_hundredths(f32::INFINITY), OWNER_EVIDENCE_UNKNOWN);

    let best_owner = crate::panorama_utils::stack_pipeline::focus_fuser::SharpnessCellEvidence {
        winner_score: 0.60,
        runner_up_score: 0.50,
        owner_score: 0.60,
        candidate_count: 2,
        disagreement: 0.0,
    };
    assert_eq!(best_owner.owner_shortfall(), 0.0);
    let lone_candidate =
        crate::panorama_utils::stack_pipeline::focus_fuser::SharpnessCellEvidence {
            owner_score: 0.30,
            candidate_count: 1,
            ..best_owner
        };
    assert_eq!(lone_candidate.owner_shortfall(), 0.0);
    let weaker_owner = crate::panorama_utils::stack_pipeline::focus_fuser::SharpnessCellEvidence {
        owner_score: 0.54,
        ..best_owner
    };
    assert!((weaker_owner.owner_shortfall() - 0.10).abs() < 1.0e-6);
    assert!(quantize_hundredths(weaker_owner.owner_shortfall()) > limit);
}

#[test]
fn missing_local_scale_never_supplies_unit_scale_to_mtf_or_gradient() {
    let output = Rgb32FImage::from_pixel(1024, 1024, Rgb([0.5; 3]));
    let coverage = GrayImage::from_pixel(1024, 1024, Luma([255]));
    let ownership = vec![1; 1024 * 1024];
    let confidence = vec![0.8; 1024 * 1024];
    let textured = vec![255; 1024 * 1024];
    let residual = ResidualWarp::new();
    let sources = [QualitySource {
        owner: 1,
        path: "synthetic.raw".to_string(),
        geometry: SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 0,
        },
        dimensions: (1024, 1024),
        native_warp: None,
        native_registration: None,
    }];
    let input = QualityGateInput {
        output: &output,
        coverage: &coverage,
        ownership: &ownership,
        confidence: &confidence,
        textured: Some(&textured),
        owner_shortfall: None,
        owner_disagreement: None,
        unresolved_pixel_count: 0,
        world_origin: (0.0, 0.0),
        sources: &sources,
        accepted_station_relations: &[],
        residual: &residual,
        acceptance_render_scale: f64::NAN,
        final_sharpen_amount: 0.0,
    };
    let mut decodes = 0;
    let report = run_quality_gate(&input, &mut |_| {
        decodes += 1;
        Ok(output.clone())
    });
    assert!(
        report.roi_count > 0,
        "the fixture must select an actual ROI"
    );
    assert_eq!(decodes, 1);
    for name in ["mtf50_normalized", "gradient_energy_normalized"] {
        let criterion = report
            .criteria
            .iter()
            .find(|item| item.name == name)
            .unwrap();
        assert_eq!(criterion.measurable_count, 0);
        assert_eq!(criterion.unmeasurable_count, report.roi_count);
        assert_eq!(
            criterion
                .unmeasurable_reasons
                .get(degradation::LOCAL_SCALE_UNMEASURABLE),
            Some(&report.roi_count),
        );
        assert!(criterion.measured.is_empty());
        assert!(criterion.failed.is_empty());
    }
    assert_eq!(
        report
            .unmeasurable
            .iter()
            .filter(|item| item.reason == degradation::LOCAL_SCALE_UNMEASURABLE)
            .count(),
        2 * report.roi_count,
    );
}

#[test]
fn world_resampled_reference_uses_the_same_scale_and_still_detects_blur() {
    let _scope = degradation::begin_run_scope();
    // `source_roi` replays the run's recorded tone transform. A previous
    // test that ran the tone harmonizer inside its own run scope leaves
    // that report behind, which would turn this identity comparison into
    // a tone mismatch; production resets it at every run start.
    crate::panorama_utils::stack_pipeline::tone::reset_run_records();
    let source = Rgb32FImage::from_fn(1024, 1024, |x, y| {
        let value = 0.42 + 0.12 * (x as f32 * 0.27).sin() + 0.08 * (y as f32 * 0.19).cos();
        Rgb([value; 3])
    });
    let scale = 1.2;
    let exact = Rgb32FImage::from_fn(1024, 1024, |x, y| {
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            &source,
            f64::from(x) / scale,
            f64::from(y) / scale,
        )
    });
    let blurred = image::imageops::blur(&exact, 3.0);
    let coverage = GrayImage::from_pixel(1024, 1024, Luma([255]));
    let ownership = vec![1; 1024 * 1024];
    let confidence = vec![0.8; 1024 * 1024];
    let textured = vec![255; 1024 * 1024];
    let residual = ResidualWarp::new();
    let sources = [QualitySource {
        native_warp: None,
        owner: 1,
        path: "world-grid.raw".to_string(),
        geometry: SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0),
            station_id: 0,
        },
        dimensions: source.dimensions(),
        native_registration: None,
    }];
    for (output, expect_pass) in [(&exact, true), (&blurred, false)] {
        let input = QualityGateInput {
            output,
            coverage: &coverage,
            ownership: &ownership,
            confidence: &confidence,
            textured: Some(&textured),
            owner_shortfall: None,
            owner_disagreement: None,
            unresolved_pixel_count: 0,
            world_origin: (0.0, 0.0),
            sources: &sources,
            accepted_station_relations: &[],
            residual: &residual,
            acceptance_render_scale: 1.0,
            final_sharpen_amount: 0.0,
        };
        let report = run_quality_gate(&input, &mut |_| Ok(source.clone()));
        let gradient = report
            .criteria
            .iter()
            .find(|criterion| criterion.name == "gradient_energy_normalized")
            .unwrap();
        assert!(gradient.measurable_count >= 8);
        if expect_pass {
            assert!(
                gradient.failed.is_empty(),
                "identical world samples lost detail"
            );
            assert!(
                gradient
                    .measured
                    .iter()
                    .all(|m| (m.measured - 1.0).abs() < 1e-6)
            );
        } else {
            assert!(
                !gradient.failed.is_empty(),
                "actual blur must still fail the unchanged gate"
            );
        }
    }
}

#[test]
fn quality_gate_record_mode_preserves_output_and_records_all_criteria() {
    // The gate is observation-only at this boundary: it may return a
    // conclusion and diagnostics, but it cannot rewrite the composed
    // pixels or add a publication-blocking degradation of its own.
    let _run_scope = degradation::begin_run_scope();
    degradation::reset_run_ledger();
    let output = Rgb32FImage::from_pixel(1024, 1024, Rgb([0.4, 0.45, 0.5]));
    let before = output.clone();
    let coverage = GrayImage::from_pixel(1024, 1024, Luma([255]));
    let ownership = vec![1u16; 1024 * 1024];
    let confidence = vec![0.8f32; 1024 * 1024];
    let textured = vec![255u8; 1024 * 1024];
    let shortfall = vec![5u8; 1024 * 1024];
    let disagreement = vec![0u8; 1024 * 1024];
    let residual = ResidualWarp::new();
    let sources = [QualitySource {
        owner: 1,
        path: "record-only.raw".to_string(),
        geometry: SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 0,
        },
        dimensions: (1024, 1024),
        native_warp: None,
        native_registration: None,
    }];
    let input = QualityGateInput {
        output: &output,
        coverage: &coverage,
        ownership: &ownership,
        confidence: &confidence,
        textured: Some(&textured),
        owner_shortfall: Some(&shortfall),
        owner_disagreement: Some(&disagreement),
        unresolved_pixel_count: 0,
        world_origin: (0.0, 0.0),
        sources: &sources,
        accepted_station_relations: &[],
        residual: &residual,
        acceptance_render_scale: 1.0,
        final_sharpen_amount: 0.0,
    };
    let report = run_quality_gate(&input, &mut |_source| Ok(output.clone()));

    assert_eq!(
        output, before,
        "record-only Quality_Gate must not rewrite pixels"
    );
    assert_eq!(report.criteria.len(), quality_gate::QUALITY_CRITERIA.len());
    assert!(
        report
            .criteria
            .iter()
            .all(|criterion| criterion.threshold.is_finite())
    );
    let effective = report
        .criteria
        .iter()
        .find(|criterion| criterion.name == "effective_pixel_count")
        .expect("effective-pixel criterion must be reported");
    assert_eq!(effective.effective_output_pixels, Some(1024 * 1024));
    // `source_quad` describes the inclusive last source sample at
    // `width - 1`/`height - 1`; the half-open scanline union therefore
    // covers 1023×1023 projected cells for this identity fixture.
    assert_eq!(effective.effective_projected_pixels, Some(1023 * 1023));
    // Record-only means the gate writes no ledger entry of its own.
    assert!(degradation::run_ledger_snapshot().entries().is_empty());
}

#[test]
fn failed_quality_gate_is_promoted_to_export_rejection_at_boundary() {
    let _run_scope = degradation::begin_run_scope();
    degradation::reset_run_ledger();
    let mut report = QualityGateReport {
        verdict: QualityGateVerdict::Fail,
        roi_count: 1,
        ..QualityGateReport::default()
    };
    report.criteria.push(QualityGateCriterionRecord {
        name: "owner_sharpness_coverage".to_string(),
        verdict: QualityGateVerdict::Fail,
        measurable_count: 1,
        ..QualityGateCriterionRecord::default()
    });

    record_gate_outcome(&report);

    let ledger = degradation::run_ledger_snapshot();
    assert_eq!(
        ledger.count_of(degradation::QUALITY_GATE_CRITERION_FAILED),
        1
    );
    assert_eq!(ledger.entries()[0].detail["roi_count"], 1);
    assert_eq!(
        ledger.entries()[0].detail["criteria"][0]["name"],
        "owner_sharpness_coverage"
    );
}

#[test]
fn unmeasurable_reason_histograms_match_the_coordinate_records() {
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 512,
        region_label: 1,
        owner: 1,
        world_origin: (-15.0, 72.0),
        boundary_clearance: 16.0,
    };
    let mut criterion = Criterion::new("mtf50_normalized", 0.93);
    let mut records = Vec::new();
    for reason in [
        degradation::SLANTED_EDGE_TOO_SHORT,
        degradation::SLANTED_EDGE_TOO_SHORT,
        degradation::SLANTED_EDGE_CONTRAST_INSUFFICIENT,
        degradation::SLANTED_EDGE_ANGLE_OUT_OF_RANGE,
        degradation::SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED,
    ] {
        record_unmeasurable(&mut criterion, &mut records, &roi, reason);
    }
    let result = criterion.finish(&mut records);
    assert_eq!(result.measurable_count, 0);
    assert_eq!(result.unmeasurable_count, 5);
    assert_eq!(result.unmeasurable_reasons.values().sum::<usize>(), 5);
    assert_eq!(
        result.unmeasurable_reasons[degradation::SLANTED_EDGE_TOO_SHORT],
        2
    );
    for (reason, count) in &result.unmeasurable_reasons {
        assert_eq!(
            records
                .iter()
                .filter(|record| &record.reason == reason)
                .count(),
            *count
        );
    }
    assert!(records.iter().all(|record| {
        record.criterion == "mtf50_normalized" && record.world == WorldPoint { x: -15.0, y: 72.0 }
    }));
}
