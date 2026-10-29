use super::*;
use image::Luma;

#[test]
fn merged_quality_reference_prefers_direct_raw_provenance() {
    let image = Rgb32FImage::from_fn(64, 64, |x, y| Rgb([x as f32 / 64.0, y as f32 / 64.0, 0.25]));
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 4,
        region_label: 1,
        owner: 1,
        world_origin: (20.0, 20.0),
        boundary_clearance: 16.0,
    };
    let source = QualitySource {
        owner: 1,
        path: "merged-reference.raw".into(),
        dimensions: image.dimensions(),
        geometry: SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 0,
        },
        native_warp: Some(NativeSourceWarp {
            displacement: None,
            source_to_station: Matrix3::identity(),
            station_to_world: Matrix3::identity(),
        }),
        native_registration: Some(NativeReferenceMap {
            world_to_station: Matrix3::identity(),
            station_to_world: Matrix3::identity(),
            station_origin: (0.0, 0.0),
            refinement: crate::panorama_utils::stitching::FocusSourceRefinement {
                translation: (3, -2),
                ..Default::default()
            },
        }),
    };
    let residual = ResidualWarp::new();
    let reference = source_roi_for_quality(&image, &roi, &source, &residual).unwrap();
    for y in 0..roi.side {
        for x in 0..roi.side {
            let pixel = reference.get_pixel(x, y);
            assert!((pixel[0] - (17.0 + x as f32) / 64.0).abs() < 1e-6);
            assert!((pixel[1] - (22.0 + y as f32) / 64.0).abs() < 1e-6);
        }
    }
    let legacy = QualitySource {
        native_registration: None,
        ..source
    };
    let reference = source_roi_for_quality(&image, &roi, &legacy, &residual).unwrap();
    assert!((reference.get_pixel(0, 0)[0] - 20.0 / 64.0).abs() < 1e-6);
}

fn synthetic_owner_evidence(
    baseline: [f32; 3],
    candidate: [f32; 3],
    reference: [f32; 3],
    fit_disjoint: bool,
) -> ToneOwnerEvidence {
    ToneOwnerEvidence {
        partition: "owner_raw_fidelity_holdout",
        roi: QualityRoi {
            x: 0,
            y: 0,
            side: 64,
            region_label: 0,
            owner: 1,
            world_origin: (0.0, 0.0),
            boundary_clearance: 0.0,
        },
        owner_path: "synthetic.raw".to_string(),
        station_id: 0,
        tile_index: 0,
        baseline_mean: baseline,
        candidate_mean: candidate,
        reference_mean: reference,
        field_gain: [1.0; 3],
        global_gain: [1.0; 3],
        global_bias: [0.0; 3],
        coverage_min: 255,
        confidence_min: 1.0,
        confidence_mean: 1.0,
        overlap_participation: false,
        fit_disjoint,
        gradient_ratio_candidate_to_source: Some(1.0),
        noise_ratio_candidate_to_source: Some(1.0),
    }
}

#[test]
fn owner_reference_uses_the_rendering_kernel_for_fractional_fine_texture() {
    let source = Rgb32FImage::from_fn(32, 32, |x, y| {
        Rgb([if (x + y) % 2 == 0 { 0.15 } else { 0.85 }; 3])
    });
    let geometry = SourceGeometry {
        member_to_anchor: Matrix3::identity(),
        tile_to_world: Matrix3::new(1.0, 0.0, -0.37, 0.0, 1.0, 0.23, 0.0, 0.0, 1.0),
        station_id: 0,
    };
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 4,
        region_label: 1,
        owner: 1,
        world_origin: (10.0, 10.0),
        boundary_clearance: 8.0,
    };
    let sampled =
        source_roi_with_registration(&source, &roi, &geometry, &ResidualWarp::new(), None).unwrap();
    for y in 0..4 {
        for x in 0..4 {
            let expected = crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                &source,
                10.37 + f64::from(x),
                9.77 + f64::from(y),
            );
            assert!((sampled.get_pixel(x, y)[0] - expected[0]).abs() < 1e-6);
        }
    }
}

#[test]
fn owner_fidelity_rejects_joint_colour_shift_and_fit_leakage() {
    let shifted = synthetic_owner_evidence([0.4; 3], [0.5; 3], [0.4; 3], true);
    assert!(owner_distribution_feasible(&[&shifted], 0.0));
    assert!(!owner_distribution_feasible(&[&shifted], 1.0));

    let leaked = synthetic_owner_evidence([0.45; 3], [0.4; 3], [0.4; 3], false);
    assert!(!owner_distribution_feasible(&[&leaked], 1.0));
}

#[test]
fn joint_oracle_recovers_feasible_solution_or_certifies_pareto_gap() {
    let recovered = select_joint_relaxation(&[0.5, 0.75, 1.0], &[0.0, 0.25, 0.5, 0.75]);
    assert_eq!(recovered.selected, Some(0.5));
    assert_eq!(recovered.pareto_gap, None);

    let impossible = select_joint_relaxation(&[0.75, 1.0], &[0.0, 0.25, 0.5]);
    assert_eq!(impossible.selected, None);
    assert_eq!(impossible.minimum_boundary, Some(0.75));
    assert_eq!(impossible.maximum_owner, Some(0.5));
    assert_eq!(impossible.pareto_gap, Some(0.25));
    assert_eq!(impossible.selected.unwrap_or(0.0), 0.0);
}

#[test]
fn low_frequency_candidate_preserves_high_frequency_gradient_and_noise() {
    let baseline = Rgb32FImage::from_fn(64, 64, |x, y| {
        let fine = if (x + y).is_multiple_of(2) {
            0.03
        } else {
            -0.03
        };
        let broad = 0.25 + 0.003 * x as f32 + 0.002 * y as f32;
        Rgb([broad + fine, broad * 0.9 + fine, broad * 0.8 + fine])
    });
    let candidate = Rgb32FImage::from_fn(64, 64, |x, y| {
        let pixel = baseline.get_pixel(x, y).0;
        let low = 0.02 + 0.0001 * x as f32;
        Rgb(pixel.map(|value| value + low))
    });
    let gradient = quality_gate::normalized_gradient_energy(&candidate, 1.0).unwrap()
        / quality_gate::normalized_gradient_energy(&baseline, 1.0).unwrap();
    let noise = quality_gate::noise_sigma_ratio(&candidate, &baseline).unwrap();
    assert!(gradient >= quality_gate::QUALITY_GRADIENT_RATIO_MIN);
    assert!(
        (quality_gate::QUALITY_NOISE_RATIO_MIN..=quality_gate::QUALITY_NOISE_RATIO_MAX)
            .contains(&noise)
    );
}

#[test]
fn owner_oracle_has_no_previous_quality_gate_list_input() {
    let source = include_str!("../quality_gate_runner.rs");
    let start = source.find("fn write_tone_owner_fidelity_oracle").unwrap();
    let end = source[start..]
        .find("pub(super) fn record_unmeasurable")
        .map(|offset| start + offset)
        .unwrap();
    let oracle = &source[start..end];
    assert!(!oracle.contains("std::fs::read"));
    assert!(!oracle.contains("GRADIENT_ORACLE_REPORT"));
    assert!(!oracle.contains("fixture"));
    assert!(oracle.contains("current-run tone owner oracle snapshot"));
}

#[test]
fn owner_reference_replays_foreground_native_field_and_translation_in_render_order() {
    use crate::panorama_utils::{
        mosaic::{Field, NativeResidualDisplacement},
        stitching::{FocusForegroundMove, FocusSourceRefinement},
    };
    let source = Rgb32FImage::from_fn(128, 128, |x, y| {
        Rgb([x as f32 / 128.0, y as f32 / 128.0, 0.25])
    });
    let station_to_world = Matrix3::new(1.0, 0.0, 100.0, 0.0, 1.0, 200.0, 0.0, 0.0, 1.0);
    let mut field = Field::<2>::new(128, 128, 16.0);
    field.values.fill([1.5, -0.5]);
    let native = NativeReferenceMap {
        world_to_station: station_to_world.try_inverse().unwrap(),
        station_to_world,
        station_origin: (0.0, 0.0),
        refinement: FocusSourceRefinement {
            translation: (3, -2),
            displacement: Some(NativeResidualDisplacement::from_test_field(field)),
            foreground_move: Some(FocusForegroundMove {
                delta: (4, 0),
                left: 0,
                top: 0,
                destinations: GrayImage::from_pixel(128, 128, Luma([255])),
            }),
            source_sampling: None,
        },
    };
    let geometry = SourceGeometry {
        member_to_anchor: Matrix3::identity(),
        tile_to_world: station_to_world,
        station_id: 0,
    };
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 4,
        region_label: 1,
        owner: 1,
        world_origin: (112.0, 224.0),
        boundary_clearance: 8.0,
    };
    let actual = source_roi_with_registration(
        &source,
        &roi,
        &geometry,
        &ResidualWarp::new(),
        Some(&native),
    )
    .unwrap();
    for y in 0..4 {
        for x in 0..4 {
            let pixel = actual.get_pixel(x, y);
            assert!((pixel[0] - (6.5 + x as f32) / 128.0).abs() < 1e-6);
            assert!((pixel[1] - (25.5 + y as f32) / 128.0).abs() < 1e-6);
        }
    }
    let wrong =
        source_roi_with_registration(&source, &roi, &geometry, &ResidualWarp::new(), None).unwrap();
    assert!((wrong.get_pixel(0, 0)[0] - actual.get_pixel(0, 0)[0]).abs() > 0.04);
}

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

    let best_owner = super::super::focus_fuser::SharpnessCellEvidence {
        winner_score: 0.60,
        runner_up_score: 0.50,
        owner_score: 0.60,
        candidate_count: 2,
        disagreement: 0.0,
    };
    assert_eq!(best_owner.owner_shortfall(), 0.0);
    let lone_candidate = super::super::focus_fuser::SharpnessCellEvidence {
        owner_score: 0.30,
        candidate_count: 1,
        ..best_owner
    };
    assert_eq!(lone_candidate.owner_shortfall(), 0.0);
    let weaker_owner = super::super::focus_fuser::SharpnessCellEvidence {
        owner_score: 0.54,
        ..best_owner
    };
    assert!((weaker_owner.owner_shortfall() - 0.10).abs() < 1.0e-6);
    assert!(quantize_hundredths(weaker_owner.owner_shortfall()) > limit);
}

#[test]
fn reference_roi_reuses_production_sampler_for_gradient_and_noise_metrics() {
    let source = Rgb32FImage::from_fn(12, 12, |x, y| {
        let value = ((x * x + 5 * x * y + 3 * y + 7) % 23) as f32 / 22.0;
        Rgb([value, value * 0.91, value * 0.83])
    });
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 4,
        region_label: 1,
        owner: 1,
        world_origin: (2.25, 2.5),
        boundary_clearance: 16.0,
    };
    let geometry = SourceGeometry {
        member_to_anchor: Matrix3::identity(),
        tile_to_world: Matrix3::identity(),
        station_id: 0,
    };
    let reference =
        source_roi(&source, &roi, &geometry, &ResidualWarp::new()).expect("identity ROI");
    let source_info = QualitySource {
        native_registration: None,
        owner: 1,
        path: "owner.raw".to_string(),
        geometry: geometry.clone(),
        dimensions: source.dimensions(),
        native_warp: None,
    };
    assert_eq!(
        sample_quality_source_at_world(
            &source,
            Point2::new(roi.world_origin.0, roi.world_origin.1),
            &source_info,
            &ResidualWarp::new(),
        )
        .unwrap(),
        *reference.get_pixel(0, 0),
    );
    let expected =
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(&source, 2.25, 2.5);
    assert_eq!(reference.get_pixel(0, 0), &expected);

    let x0 = 2u32;
    let y0 = 2u32;
    let tx = 0.25f32;
    let ty = 0.5f32;
    let a = *source.get_pixel(x0, y0);
    let b = *source.get_pixel(x0 + 1, y0);
    let c = *source.get_pixel(x0, y0 + 1);
    let d = *source.get_pixel(x0 + 1, y0 + 1);
    let bilinear = Rgb(std::array::from_fn(|channel| {
        let top = a[channel] * (1.0 - tx) + b[channel] * tx;
        let bottom = c[channel] * (1.0 - tx) + d[channel] * tx;
        top * (1.0 - ty) + bottom * ty
    }));
    assert!(
        (reference.get_pixel(0, 0)[0] - bilinear[0]).abs() > 1.0e-6,
        "reference ROI must not silently fall back to the old bilinear kernel"
    );

    let output = Rgb32FImage::from_fn(roi.side, roi.side, |x, y| {
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            &source,
            roi.world_origin.0 + f64::from(x),
            roi.world_origin.1 + f64::from(y),
        )
    });
    let local_scale = 1.125;
    let (gradient_ratio, gradient_passes) = gradient_energy_ratio(&output, &reference, local_scale)
        .expect("the sampled ROIs must remain measurable")
        .expect("the reference gradient must remain non-zero");
    assert!((gradient_ratio - 1.0).abs() <= 1.0e-12);
    assert!(gradient_passes);
    let noise_ratio = quality_gate::noise_sigma_ratio(&output, &reference)
        .expect("identical production and reference sampling must have measurable noise");
    assert!((noise_ratio - 1.0).abs() <= 1.0e-12);
}

#[test]
fn owner_fidelity_mapping_composes_member_tile_and_requires_one_source() {
    let source = Rgb32FImage::from_fn(32, 32, |x, y| {
        let value = (x + 3 * y) as f32 / 124.0;
        Rgb([value, value * 0.9, value * 0.8])
    });
    let translate = |x: f64, y: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0);
    let geometry = SourceGeometry {
        member_to_anchor: translate(5.0, 7.0),
        tile_to_world: translate(11.0, 13.0),
        station_id: 4,
    };
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 2,
        region_label: 1,
        owner: 9,
        world_origin: (18.25, 23.5),
        boundary_clearance: 16.0,
    };
    let sampled = source_roi(&source, &roi, &geometry, &ResidualWarp::new()).unwrap();
    let expected =
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(&source, 2.25, 3.5);
    assert_eq!(sampled.get_pixel(0, 0), &expected);

    let info = QualitySource {
        native_registration: None,
        owner: 9,
        path: "owner.raw".to_string(),
        geometry,
        dimensions: source.dimensions(),
        native_warp: None,
    };
    assert_eq!(
        unique_owner_source(std::slice::from_ref(&info), 9)
            .unwrap()
            .path,
        "owner.raw"
    );
    assert!(unique_owner_source(&[info.clone(), info], 9).is_err());
    assert!(unique_owner_source(&[], 9).is_err());
}

#[test]
fn source_roi_mapping_guard_requires_render_provenance_for_focus_band_and_native_delta() {
    // The production Quality_Gate mapping is intentionally only the
    // world -> Residual_Warp inverse -> member/tile inverse path.  A
    // focus render can apply two additional mappings after that path:
    // a local band homography and the accepted native displacement field.
    // Keep a coordinate-coded source so a future provenance integration
    // can compare the exact sample, rather than only comparing geometry
    // matrices.
    let source = Rgb32FImage::from_fn(48, 48, |x, y| {
        Rgb([x as f32 / 48.0, y as f32 / 48.0, (x + y) as f32 / 96.0])
    });
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 2,
        region_label: 1,
        owner: 1,
        world_origin: (10.25, 11.5),
        boundary_clearance: 16.0,
    };
    let geometry = SourceGeometry {
        member_to_anchor: Matrix3::identity(),
        tile_to_world: Matrix3::identity(),
        station_id: 10,
    };
    let qg_reference = source_roi(&source, &roi, &geometry, &ResidualWarp::new())
        .expect("identity world geometry must remain measurable");

    // `render_focus_layer` stores a source->world local band homography;
    // translation(+2, 0) therefore maps a world sample back to source x-2.
    // `warp_rendered_focus_layer` samples the candidate at +1 canvas pixel
    // for this synthetic native field, yielding a final source x-1.  These
    // values model the station 10/12 issue without adding a second,
    // production mapping implementation to the Quality_Gate.
    let world = Point2::new(roi.world_origin.0, roi.world_origin.1);
    let expected_with_render_provenance =
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            &source,
            world.x - 1.0,
            world.y,
        );
    let qg_pixel = qg_reference.get_pixel(0, 0);
    assert_ne!(qg_pixel, &expected_with_render_provenance);
    assert!(
        (qg_pixel[0] - expected_with_render_provenance[0]).abs() > 1.0e-5,
        "the current QG reference must not silently claim local-band/native provenance"
    );
}

#[test]
fn aligned_gradient_ratio_is_invariant_to_shared_local_scale() {
    let output = Rgb32FImage::from_fn(16, 16, |x, y| {
        let value = ((x * x + 3 * y + x * y) % 29) as f32 / 28.0;
        Rgb([value, value * 0.9, value * 0.8])
    });
    let reference = Rgb32FImage::from_fn(16, 16, |x, y| {
        let value = ((2 * x * x + y + x * y) % 31) as f32 / 30.0;
        Rgb([value, value * 0.9, value * 0.8])
    });
    let unit = gradient_energy_ratio(&output, &reference, 1.0)
        .unwrap()
        .unwrap()
        .0;
    for local_scale in [0.8, 1.125] {
        let scaled = gradient_energy_ratio(&output, &reference, local_scale)
            .unwrap()
            .unwrap()
            .0;
        assert!((scaled - unit).abs() <= 1.0e-12);
    }
}

#[test]
fn confidence_distribution_accumulates_compact_cell_evidence() {
    let mut result = super::super::report::ConfidenceScoreDistribution {
        winner_histogram: vec![0; 21],
        runner_up_histogram: vec![0; 21],
        ..Default::default()
    };
    let evidence = super::super::focus_fuser::SharpnessCellEvidence {
        winner_score: 0.60,
        runner_up_score: 0.59,
        owner_score: 0.60,
        candidate_count: 2,
        disagreement: 0.0,
    };
    accumulate_confidence_score(&mut result, 255, 0.01, 255, evidence);
    accumulate_confidence_score(&mut result, 255, 0.05, 255, evidence);
    accumulate_confidence_score(&mut result, 255, 0.01, 0, evidence);
    assert_eq!(result.textured_low_count, 1);
    assert_eq!(result.both_scores_at_least_0_10, 1);
    assert_eq!(result.winner_below_0_10, 0);
    assert_eq!(result.winner_histogram[12], 1);
    assert_eq!(result.runner_up_histogram[11], 1);
    assert!((result.winner_sum - 0.60).abs() < 1e-6);
    assert!((result.runner_up_sum - 0.59).abs() < 1e-6);
}

#[test]
fn textured_partition_excludes_transparency_and_keeps_the_fixed_confidence_boundary() {
    let coverage = GrayImage::from_raw(6, 1, vec![255, 255, 255, 255, 0, 0]).unwrap();
    let confidence = [0.049, 0.05, 0.0, 0.8, 0.0, 0.0];
    let textured = [255, 255, 0, 0, 255, 0];
    let stats = confidence_coverage_stats(&coverage, &confidence, &textured).unwrap();
    assert_eq!(stats.textured_pixels, 2);
    assert_eq!(stats.flat_pixels, 2);
    assert_eq!(stats.textured_low, 1, "0.05 itself is not low confidence");
    assert_eq!(stats.flat_low, 1);
    assert_eq!(stats.textured_unknown, 0);
    assert_eq!(stats.flat_unknown, 0);
}

#[test]
fn confidence_stats_report_all_flat_pixels_and_exclude_nan_from_ratios() {
    let coverage = GrayImage::from_raw(5, 1, vec![255; 5]).unwrap();
    let confidence = [f32::NAN, 0.02, f32::INFINITY, 0.049, 0.05];
    let textured = [0, 0, 0, 0, 0];
    let stats = confidence_coverage_stats(&coverage, &confidence, &textured).unwrap();
    assert_eq!(stats.textured_pixels, 0);
    assert_eq!(stats.flat_pixels, 5);
    assert_eq!(stats.flat_low, 2);
    assert_eq!(stats.flat_unknown, 2);
    assert_eq!(stats.textured_unknown, 0);

    let mixed = confidence_coverage_stats(&coverage, &confidence, &[255, 255, 0, 0, 0]).unwrap();
    assert_eq!(mixed.textured_pixels, 2);
    assert_eq!(mixed.textured_low, 1);
    assert_eq!(mixed.textured_unknown, 1);
}

#[test]
fn texture_evidence_uses_all_covered_candidates_and_the_inclusive_sharpness_floor() {
    use super::super::focus_fuser::{CandidateCells, StationFusion};

    let mut fusion = StationFusion::new(24, 8, 2);
    assert_eq!(fusion.geometry().cell_count(), 3);
    fusion.seed(0, &[0.099, 0.10, 0.0], &[true, true, true]);
    assert_eq!(fusion.textured_cells(), vec![false, true, false]);
    let mut candidate = CandidateCells::new(3);
    candidate.sharpness = vec![0.10, 0.05, 0.9];
    candidate.covered = vec![true, true, false];
    fusion.fold(1, &candidate);
    assert_eq!(fusion.textured_cells(), vec![true, true, false]);
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
    super::super::tone::reset_run_records();
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

fn boundary_sources(width: u32, height: u32) -> [QualitySource; 2] {
    [
        QualitySource {
            native_registration: None,
            owner: 1,
            path: "left.raw".to_string(),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 0,
            },
            dimensions: (width, height),
            native_warp: None,
        },
        QualitySource {
            native_registration: None,
            owner: 2,
            path: "right.raw".to_string(),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 1,
            },
            dimensions: (width, height),
            native_warp: None,
        },
    ]
}

fn vertical_seam_owners(width: u32, height: u32, seam_x: u32) -> Vec<u16> {
    (0..width * height)
        .map(|index| if index % width < seam_x { 1 } else { 2 })
        .collect()
}

fn fragmented_vertical_seam_owners(width: u32, height: u32, seam_x: u32) -> Vec<u16> {
    (0..width * height)
        .map(|index| {
            let x = index % width;
            if x < seam_x.saturating_sub(2) {
                1
            } else if x < seam_x {
                3
            } else {
                2
            }
        })
        .collect()
}

fn crossing_stroke(
    width: u32,
    height: u32,
    seam_x: u32,
    left_y: u32,
    right_y: u32,
    invert_right: bool,
) -> Rgb32FImage {
    Rgb32FImage::from_fn(width, height, |x, y| {
        let threshold = if x < seam_x { left_y } else { right_y };
        let high = if x >= seam_x && invert_right {
            y < threshold
        } else {
            y >= threshold
        };
        Rgb([if high { 0.9 } else { 0.1 }; 3])
    })
}

fn boundary_report(
    output: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    sources: &[QualitySource],
    accepted: &[(usize, usize)],
) -> QualityGateCriterionRecord {
    let residual = ResidualWarp::new();
    let mut criterion = Criterion::p95_with_max("boundary_stroke_alignment", 1.5, 3.0);
    let mut records = Vec::new();
    measure_large_boundaries(
        output,
        coverage,
        ownership,
        (0.0, 0.0),
        sources,
        accepted,
        &residual,
        &mut criterion,
        &mut records,
    );
    criterion.finish(&mut records)
}

#[test]
fn accepted_seam_pairs_independently_detected_crossing_strokes() {
    let (width, height, seam_x) = (192, 192, 96);
    let output = crossing_stroke(width, height, seam_x, 80, 82, false);
    let coverage = GrayImage::from_pixel(width, height, Luma([255]));
    let ownership = vertical_seam_owners(width, height, seam_x);
    let report = boundary_report(
        &output,
        &coverage,
        &ownership,
        &boundary_sources(width, height),
        &[(0, 1)],
    );
    assert_eq!(report.measurable_count, 1);
    assert_eq!(report.measured[0].measured, 2.0);
    assert!(
        report.measured[0]
            .owner_path
            .contains("station:0/station:1")
    );
    assert!(report.measured[0].owner_path.contains("samples:"));
}

#[test]
fn station_boundary_collapses_fragmented_raw_owners() {
    let (width, height, seam_x) = (192, 192, 96);
    let output = crossing_stroke(width, height, seam_x, 80, 82, false);
    let coverage = GrayImage::from_pixel(width, height, Luma([255]));
    let mut sources = boundary_sources(width, height).to_vec();
    sources.push(QualitySource {
        native_registration: None,
        owner: 3,
        path: "left-overlap.raw".to_string(),
        geometry: SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 0,
        },
        dimensions: (width, height),
        native_warp: None,
    });
    let ownership = fragmented_vertical_seam_owners(width, height, seam_x);
    let report = boundary_report(&output, &coverage, &ownership, &sources, &[(0, 1)]);
    assert_eq!(report.measurable_count, 1);
    assert_eq!(report.measured[0].measured, 2.0);
    assert!(report.measured[0].owner_path.contains("owner:1/owner:2"));
}

#[test]
fn nearby_topology_transitions_do_not_duplicate_one_stroke() {
    let (width, height, seam_x) = (192, 192, 96);
    let output = crossing_stroke(width, height, seam_x, 80, 80, false);
    let coverage = GrayImage::from_pixel(width, height, Luma([255]));
    let ownership = vertical_seam_owners(width, height, seam_x);
    let sources = boundary_sources(width, height);
    let first = boundary_report(&output, &coverage, &ownership, &sources, &[(0, 1)]);
    let second = boundary_report(&output, &coverage, &ownership, &sources, &[(1, 0)]);
    assert_eq!(first.measurable_count, 1);
    assert_eq!(
        first, second,
        "relation order must not change deterministic evidence"
    );
}

#[test]
fn sampling_is_restricted_to_accepted_cross_station_hard_seams() {
    let (width, height, seam_x) = (192, 192, 96);
    let output = crossing_stroke(width, height, seam_x, 80, 80, false);
    let coverage = GrayImage::from_pixel(width, height, Luma([255]));
    let ownership = vertical_seam_owners(width, height, seam_x);
    let sources = boundary_sources(width, height);
    let report = boundary_report(&output, &coverage, &ownership, &sources, &[]);
    assert_eq!(report.measurable_count, 0);
    assert_eq!(
        report.unmeasurable_reasons[degradation::BOUNDARY_NO_PAIRABLE_EDGE],
        1
    );
}

#[test]
fn opposite_polarity_strokes_are_rejected_as_wrong_correspondences() {
    let (width, height, seam_x) = (192, 192, 96);
    let output = crossing_stroke(width, height, seam_x, 80, 80, true);
    let coverage = GrayImage::from_pixel(width, height, Luma([255]));
    let ownership = vertical_seam_owners(width, height, seam_x);
    let report = boundary_report(
        &output,
        &coverage,
        &ownership,
        &boundary_sources(width, height),
        &[(0, 1)],
    );
    assert_eq!(report.measurable_count, 0);
    assert!(
        report
            .unmeasurable_reasons
            .contains_key(degradation::BOUNDARY_ORIENTATION_MISMATCH)
    );
}

#[test]
fn wider_wrong_edge_is_rejected_by_fixed_cross_section() {
    let (width, height, seam_x) = (192, 192, 96);
    let output = crossing_stroke(width, height, seam_x, 70, 82, false);
    let coverage = GrayImage::from_pixel(width, height, Luma([255]));
    let ownership = vertical_seam_owners(width, height, seam_x);
    let report = boundary_report(
        &output,
        &coverage,
        &ownership,
        &boundary_sources(width, height),
        &[(0, 1)],
    );
    assert_eq!(report.measurable_count, 0);
    assert!(
        report
            .unmeasurable_reasons
            .contains_key(degradation::BOUNDARY_LOW_CONTRAST)
    );
}

#[test]
fn coverage_and_source_provenance_are_required_for_pairability() {
    let (width, height, seam_x) = (192, 192, 96);
    let output = crossing_stroke(width, height, seam_x, 80, 80, false);
    let ownership = vertical_seam_owners(width, height, seam_x);
    let sources = boundary_sources(width, height);
    let coverage = GrayImage::from_fn(width, height, |x, _| {
        Luma([if x == seam_x - 1 || x == seam_x {
            255
        } else {
            0
        }])
    });
    let no_coverage = boundary_report(&output, &coverage, &ownership, &sources, &[(0, 1)]);
    assert_eq!(no_coverage.measurable_count, 0);

    let coverage = GrayImage::from_pixel(width, height, Luma([255]));
    let mut invalid_sources = boundary_sources(width, height);
    invalid_sources[1].geometry.tile_to_world[(0, 2)] = 1_000.0;
    let no_provenance =
        boundary_report(&output, &coverage, &ownership, &invalid_sources, &[(0, 1)]);
    assert_eq!(no_provenance.measurable_count, 0);
    assert!(
        no_provenance
            .unmeasurable_reasons
            .contains_key(degradation::BOUNDARY_PROVENANCE_UNAVAILABLE)
    );
    assert_eq!(no_provenance.technical_unmeasurable_count, 1);
}

#[test]
fn sparse_roi_cap_covers_the_full_canvas_and_late_source_owners() {
    let (width, height) = (4096, 12288);
    let coverage = GrayImage::from_pixel(width, height, Luma([255]));
    let ownership = (0..height)
        .flat_map(|y| std::iter::repeat_n(if y < height / 2 { 1 } else { 2 }, width as usize))
        .collect::<Vec<_>>();
    let selected = select_sparse_rois(&coverage, &ownership, (10.0, -30.0));
    assert_eq!(selected.len(), quality_gate::QUALITY_ROI_MAX_COUNT);
    assert_eq!(
        selected.iter().map(|r| r.owner).collect::<BTreeSet<_>>(),
        BTreeSet::from([1, 2])
    );
    assert!(selected.iter().map(|r| r.y).max().unwrap() >= height - 1024);
    assert!(selected.iter().map(|r| r.x).max().unwrap() >= width - 1024);
    assert_eq!(
        selected,
        select_sparse_rois(&coverage, &ownership, (10.0, -30.0))
    );
}
