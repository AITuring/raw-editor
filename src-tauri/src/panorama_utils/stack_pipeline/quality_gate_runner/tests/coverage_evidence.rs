//! Confidence and texture evidence over covered pixels, and the sparse ROI
//! selection over the whole canvas.

use super::*;

#[test]
fn confidence_distribution_accumulates_compact_cell_evidence() {
    let mut result = crate::panorama_utils::stack_pipeline::report::ConfidenceScoreDistribution {
        winner_histogram: vec![0; 21],
        runner_up_histogram: vec![0; 21],
        ..Default::default()
    };
    let evidence = crate::panorama_utils::stack_pipeline::focus_fuser::SharpnessCellEvidence {
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
    use crate::panorama_utils::stack_pipeline::focus_fuser::{CandidateCells, StationFusion};

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
