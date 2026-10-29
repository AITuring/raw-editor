//! Boundary stroke alignment across accepted cross-station seams: the synthetic
//! seam fixtures, pairing, deduplication, wrong correspondences and the coverage
//! and provenance a pair needs.

use super::*;

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
