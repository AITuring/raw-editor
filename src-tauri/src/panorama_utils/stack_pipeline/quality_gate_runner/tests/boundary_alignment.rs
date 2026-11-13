//! Boundary stroke alignment across accepted cross-station seams: the synthetic
//! seam fixtures, pairing, deduplication, wrong correspondences and the coverage
//! and provenance a pair needs.

use super::*;

#[test]
fn stroke_cell_merge_cannot_replace_failure_or_technical_evidence_from_another_seam_cell() {
    use super::super::boundary_pairs::{
        BoundaryCellObservation, BoundaryPairEvidence, boundary_observation_can_replace,
    };
    let measured = |error, strength| {
        BoundaryCellObservation::Measured(BoundaryPairEvidence {
            world: (64.0, 128.0),
            error,
            strength,
            owner_path: "station:0/station:1".to_string(),
        })
    };
    let failure = measured(4.0, 0.2);
    let stronger_pass = measured(0.0, 0.9);
    assert!(!boundary_observation_can_replace(&failure, &stronger_pass));
    let technical = BoundaryCellObservation::Unmeasurable {
        world: (64.0, 128.0),
        reason: degradation::BOUNDARY_PROVENANCE_UNAVAILABLE,
    };
    assert!(!boundary_observation_can_replace(
        &technical,
        &stronger_pass
    ));
    let content = BoundaryCellObservation::Unmeasurable {
        world: (64.0, 128.0),
        reason: degradation::BOUNDARY_LOW_CONTRAST,
    };
    assert!(boundary_observation_can_replace(&content, &failure));
    assert!(boundary_observation_can_replace(&content, &technical));
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
    boundary_report_with_search(output, coverage, ownership, sources, accepted, false)
}

fn boundary_report_with_search(
    output: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    sources: &[QualitySource],
    accepted: &[(usize, usize)],
    criterion_specific_pool: bool,
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
        1.0,
        criterion_specific_pool,
        &mut criterion,
        &mut records,
    );
    criterion.finish(&mut records)
}

#[test]
fn boundary_candidate_search_counts_cells_and_retains_first_measurable_failure() {
    let (width, height, seam_x) = (192, 192, 96);
    let coverage = GrayImage::from_pixel(width, height, Luma([255]));
    let ownership = vertical_seam_owners(width, height, seam_x);
    let sources = boundary_sources(width, height);
    let flat = Rgb32FImage::from_pixel(width, height, Rgb([0.5; 3]));
    let rejected =
        boundary_report_with_search(&flat, &coverage, &ownership, &sources, &[(0, 1)], true);
    assert_eq!(
        rejected.candidate_count, 3,
        "192 seam pixels are three 64px candidate cells"
    );
    assert_eq!(rejected.eligible_count, 0);
    assert_eq!(rejected.rejected_content_count, 3);
    assert_eq!(rejected.content_not_applicable_count, 0);
    assert_eq!(
        rejected.rejected_content_reasons[degradation::BOUNDARY_LOW_CONTRAST],
        3
    );
    assert_eq!(rejected.rejected_content_examples.len(), 3);

    // Two strokes share one candidate cell. The first is misaligned by four
    // pixels; a later aligned stroke must not replace its measured failure.
    let output = Rgb32FImage::from_fn(width, height, |x, y| {
        let first = if x < seam_x { 80 } else { 84 };
        Rgb([if y >= first && y < 100 { 0.9 } else { 0.1 }; 3])
    });
    let result =
        boundary_report_with_search(&output, &coverage, &ownership, &sources, &[(0, 1)], true);
    assert_eq!(
        result.candidate_count,
        result.eligible_count + result.rejected_content_count + result.technical_unmeasurable_count
    );
    assert!(
        !result.failed.is_empty(),
        "do not replace a measurable failure with a later passing edge"
    );
    assert_eq!(result.verdict, QualityGateVerdict::Fail);
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
    assert!(report.candidate_count >= 1);
    assert_eq!(report.eligible_count, report.measurable_count);
    assert_eq!(
        report.rejected_content_count,
        report.content_not_applicable_count
    );
    assert_eq!(
        report.selection_mode,
        "station_boundary_streaming_candidates"
    );
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
fn boundary_gradient_requires_a_fully_covered_single_owner_stencil() {
    use super::super::boundary_pairs::boundary_edge_on_side;
    let (width, height) = (32, 32);
    let output = Rgb32FImage::from_fn(width, height, |x, y| {
        let value = if x == 12 && y == 17 { 0.9 } else { 0.1 };
        Rgb([value; 3])
    });
    let coverage = GrayImage::from_pixel(width, height, Luma([255]));
    let ownership = vec![1u16; (width * height) as usize];
    let sample = |coverage: &GrayImage, ownership: &[u16]| {
        boundary_edge_on_side(
            &output,
            coverage,
            ownership,
            (16.0, 16.0),
            (1.0, 0.0),
            -1.0,
            4.0,
            1,
        )
    };

    assert!(sample(&coverage, &ownership).is_ok());

    let mut jagged_ownership = ownership.clone();
    jagged_ownership[(17 * width + 12) as usize] = 2;
    assert_eq!(
        sample(&coverage, &jagged_ownership).unwrap_err(),
        degradation::BOUNDARY_LOW_CONTRAST,
        "an opposite-owner gradient neighbour must not contaminate the expected source"
    );

    let mut holed_coverage = coverage.clone();
    holed_coverage.put_pixel(12, 17, Luma([0]));
    assert_eq!(
        sample(&holed_coverage, &ownership).unwrap_err(),
        degradation::BOUNDARY_LOW_CONTRAST,
        "an uncovered gradient neighbour must make the stencil unmeasurable"
    );
}
