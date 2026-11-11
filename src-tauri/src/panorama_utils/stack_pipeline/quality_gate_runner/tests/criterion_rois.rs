use super::super::criterion_rois::measure_criterion_specific_rois;
use super::*;
use crate::panorama_utils::stack_pipeline::report::QualityGateTimingRecord;
use std::collections::BTreeMap;

fn content_image() -> Rgb32FImage {
    // The only slanted edge lies in the right half of the canvas. Flat noisy
    // patches occupy both sides. A shared upper-left ROI cannot measure MTF.
    let sharp = Rgb32FImage::from_fn(1536, 640, |x, y| {
        let edge_x = 1030.0 + f64::from(y) * 7.0f64.to_radians().tan();
        let baseline = if f64::from(x) >= edge_x { 0.85 } else { 0.15 };
        let noise = if (x * 17 + y * 31) % 7 < 3 {
            0.002
        } else {
            -0.002
        };
        Rgb([baseline + noise; 3])
    });
    image::imageops::blur(&sharp, 0.9)
}

fn fixture_input<'a>(
    output: &'a Rgb32FImage,
    coverage: &'a GrayImage,
    ownership: &'a [u16],
    confidence: &'a [f32],
    sources: &'a [QualitySource],
    residual: &'a ResidualWarp,
) -> QualityGateInput<'a> {
    QualityGateInput {
        output,
        coverage,
        ownership,
        confidence,
        sources,
        residual,
        textured: None,
        owner_shortfall: None,
        owner_disagreement: None,
        unresolved_pixel_count: 0,
        world_origin: (0.0, 0.0),
        accepted_station_relations: &[],
        acceptance_render_scale: 1.0,
        final_sharpen_amount: 0.0,
    }
}

fn check_counts(record: &QualityGateCriterionRecord) {
    assert_eq!(
        record.candidate_count,
        record.eligible_count + record.rejected_content_count + record.technical_unmeasurable_count
    );
    assert_eq!(record.eligible_count, record.measurable_count);
    assert_eq!(
        record.rejected_content_reasons.values().sum::<usize>(),
        record.rejected_content_count
    );
    assert!(record.rejected_content_examples.len() <= 32);
}

#[test]
fn independent_search_finds_edges_and_flat_patches_outside_dense_rois_and_reuses_references() {
    let _scope = degradation::begin_run_scope();
    crate::panorama_utils::stack_pipeline::tone::reset_run_records();
    let output = content_image();
    let coverage = GrayImage::from_pixel(output.width(), output.height(), Luma([255]));
    let ownership = vec![1; output.width() as usize * output.height() as usize];
    let confidence = vec![0.8; ownership.len()];
    let mut source = cache_source("criterion-content.raw");
    source.dimensions = output.dimensions();
    let sources = [source];
    let residual = ResidualWarp::new();
    let input = fixture_input(
        &output,
        &coverage,
        &ownership,
        &confidence,
        &sources,
        &residual,
    );
    let mut cache = QualityRoiCache::new(64 * 1024 * 1024);
    let mut loads = 0;
    let mut previous = None;
    for pass in 0..2 {
        let mut mtf = Criterion::new("mtf50_normalized", 0.93);
        let mut noise = Criterion::bounded("noise_sigma_ratio", 0.85, 1.15);
        let mut records = Vec::new();
        let mut timing = QualityGateTimingRecord::default();
        measure_criterion_specific_rois(
            &input,
            &ownership,
            &mut |_| {
                loads += 1;
                Ok(output.clone())
            },
            &mut cache,
            &mut BTreeMap::new(),
            &mut mtf,
            &mut noise,
            &mut records,
            &mut timing,
        );
        let mtf = mtf.finish(&mut records);
        let noise = noise.finish(&mut records);
        check_counts(&mtf);
        check_counts(&noise);
        assert!(
            mtf.measurable_count > 0,
            "new candidate lattice must find the far-right edge"
        );
        assert!(mtf.measured.iter().all(|item| item.world.x >= 768.0));
        assert!(noise.measurable_count >= quality_gate::QUALITY_MIN_MEASURABLE);
        assert!(mtf.failed.is_empty() && noise.failed.is_empty());
        assert_eq!(mtf.content_not_applicable_count, 0);
        assert_eq!(noise.content_not_applicable_count, 0);
        assert_ne!(mtf.selection_mode, noise.selection_mode);
        let value = serde_json::to_value((&mtf, &noise)).unwrap();
        let round_trip: (QualityGateCriterionRecord, QualityGateCriterionRecord) =
            serde_json::from_value(value.clone()).unwrap();
        assert_eq!(round_trip, (mtf, noise));
        if pass == 0 {
            previous = Some(value);
        } else {
            assert_eq!(previous.as_ref(), Some(&value));
        }
    }
    assert_eq!(
        loads, 1,
        "second pass must reuse retained source-reference windows"
    );
}

#[test]
fn sparse_search_keeps_measured_failure_and_decode_failure_visible() {
    let _scope = degradation::begin_run_scope();
    crate::panorama_utils::stack_pipeline::tone::reset_run_records();
    let reference = content_image();
    let output = image::imageops::blur(&reference, 2.0);
    let coverage = GrayImage::from_pixel(output.width(), output.height(), Luma([255]));
    let ownership = vec![1; output.width() as usize * output.height() as usize];
    let confidence = vec![0.8; ownership.len()];
    let mut source = cache_source("criterion-blur.raw");
    source.dimensions = output.dimensions();
    let sources = [source];
    let residual = ResidualWarp::new();
    let input = fixture_input(
        &output,
        &coverage,
        &ownership,
        &confidence,
        &sources,
        &residual,
    );
    for decode_fails in [false, true] {
        let mut mtf = Criterion::new("mtf50_normalized", 0.93);
        let mut noise = Criterion::bounded("noise_sigma_ratio", 0.85, 1.15);
        let mut records = Vec::new();
        measure_criterion_specific_rois(
            &input,
            &ownership,
            &mut |_| {
                if decode_fails {
                    Err("fixture decode failed".to_string())
                } else {
                    Ok(reference.clone())
                }
            },
            &mut QualityRoiCache::new(0),
            &mut BTreeMap::new(),
            &mut mtf,
            &mut noise,
            &mut records,
            &mut QualityGateTimingRecord::default(),
        );
        let mtf = mtf.finish(&mut records);
        let noise = noise.finish(&mut records);
        check_counts(&mtf);
        check_counts(&noise);
        if decode_fails {
            assert!(mtf.technical_unmeasurable_count > 0);
            assert_eq!(mtf.verdict, QualityGateVerdict::InsufficientEvidence);
        } else {
            assert!(
                !mtf.failed.is_empty(),
                "selection must not replace a blurred measurable edge"
            );
            assert_eq!(mtf.verdict, QualityGateVerdict::Fail);
        }
    }
}
