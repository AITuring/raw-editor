//! Regression tests for the checkpoint-10 Virtual_Tile station matcher:
//! coarse-to-fine match planes, the fine polish stage, prior-free relation
//! proposals, rendering-prior repair planning and the structural quality plane.

use super::*;
use crate::panorama_utils::stack_pipeline::{intra_station, report};
use nalgebra::Point2;

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

fn full_coverage(width: u32, height: u32) -> stitching::CoverageMask {
    stitching::CoverageMask::from_bytes(
        width,
        height,
        vec![u8::MAX; width as usize * height as usize],
    )
    .expect("a fully covered mask matches its plane")
}

fn coverage_from_fn(
    width: u32,
    height: u32,
    covered: impl Fn(u32, u32) -> bool,
) -> stitching::CoverageMask {
    let bytes = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| if covered(x, y) { u8::MAX } else { 0 })
        .collect::<Vec<_>>();
    stitching::CoverageMask::from_bytes(width, height, bytes).expect("mask dimensions match")
}

fn coverage_bits(mask: &stitching::CoverageMask) -> Vec<bool> {
    let (width, height) = mask.dimensions();
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| mask.is_covered(x, y))
        .collect()
}

fn textured_value(x: f64, y: f64) -> f64 {
    (126.0
        + 47.0 * (x * 0.091 + y * 0.037).sin()
        + 38.0 * (x * 0.027 - y * 0.113).cos()
        + 24.0 * (x * 0.173 + y * 0.149).sin()
        + 13.0 * (x * 0.311 - y * 0.071).cos())
    .clamp(0.0, 255.0)
}

fn textured_plane(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        image::Luma([textured_value(f64::from(x), f64::from(y)).round() as u8])
    })
}

fn plane_info(id: usize, gray: GrayImage) -> ImageInfo {
    ImageInfo {
        id,
        filename: format!("virtual://pyramid-test-{id}"),
        width: gray.width(),
        height: gray.height(),
        alignment_image: gray,
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

fn planes_for(
    factors: &[u32],
    gray: &GrayImage,
    coverage: &stitching::CoverageMask,
) -> Vec<VirtualTileMatchPlane> {
    factors
        .iter()
        .map(|&factor| {
            VirtualTileMatchPlane::from_native(gray, coverage, factor).expect("usable level")
        })
        .collect()
}

fn translation(dx: f64, dy: f64) -> Matrix3<f64> {
    Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0)
}

/// `linear` applied about `center`: T(c) · L · T(−c).
fn about_center(center: (f64, f64), linear: Matrix3<f64>) -> Matrix3<f64> {
    translation(center.0, center.1) * linear * translation(-center.0, -center.1)
}

fn rotation(theta: f64) -> Matrix3<f64> {
    let (sin, cos) = theta.sin_cos();
    Matrix3::new(cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0)
}

fn uniform_scale(scale: f64) -> Matrix3<f64> {
    Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0)
}

/// `sqrt(|det J|)` of a homography at `point`, by central differences.
fn numeric_local_scale(transform: &Matrix3<f64>, point: Point2<f64>) -> f64 {
    let step = 1e-3;
    let at = |dx: f64, dy: f64| {
        transformed_point(transform, Point2::new(point.x + dx, point.y + dy)).expect("finite point")
    };
    let ddx = (at(step, 0.0) - at(-step, 0.0)) / (2.0 * step);
    let ddy = (at(0.0, step) - at(0.0, -step)) / (2.0 * step);
    (ddx.x * ddy.y - ddx.y * ddy.x).abs().sqrt()
}

fn assert_projectively_equal(left: &Matrix3<f64>, right: &Matrix3<f64>, context: &str) {
    for (x, y) in [
        (0.0, 0.0),
        (400.0, 0.0),
        (0.0, 300.0),
        (400.0, 300.0),
        (123.0, 77.0),
    ] {
        let point = Point2::new(x, y);
        let a = transformed_point(left, point).expect("finite left point");
        let b = transformed_point(right, point).expect("finite right point");
        assert!(
            (a - b).norm() < 1e-6,
            "{context}: {a:?} vs {b:?} at ({x}, {y})"
        );
    }
}

fn line_topology(station_count: usize) -> topology::StationTopology {
    topology::StationTopology {
        row: vec![0; station_count],
        column: (0..station_count as u32).collect(),
        ..topology::StationTopology::default()
    }
}

fn prior_proposal(
    left: usize,
    right: usize,
    left_to_right: Matrix3<f64>,
    inliers: usize,
) -> VirtualTilePriorProposal {
    VirtualTilePriorProposal {
        left,
        right,
        left_to_right,
        inliers,
    }
}

/// Four station placements with small projective parts.
fn sample_tile_to_world() -> Vec<Matrix3<f64>> {
    vec![
        Matrix3::identity(),
        Matrix3::new(1.0, 0.0, 380.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0),
        Matrix3::new(1.01, 0.002, 770.0, -0.001, 0.99, -8.0, 0.000_001, 0.0, 1.0),
        Matrix3::new(1.0, 0.0, 1_150.0, 0.0, 1.0, 12.0, 0.0, 0.0, 1.0),
    ]
}

fn sample_relation(dx: f64, dy: f64) -> Matrix3<f64> {
    Matrix3::new(
        0.998, 0.004, dx, 0.003, 1.001, dy, 0.000_002, -0.000_001, 1.0,
    )
}

fn repaired_stations(repairs: &[VirtualTilePriorRepair]) -> Vec<usize> {
    repairs.iter().map(|repair| repair.station).collect()
}

// ---------------------------------------------------------------------------
// 1. pyramid factors
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_pyramid_factors_keep_the_probe_window_and_the_smallest_level_size() {
    let window = f64::from(2 * VIRTUAL_TILE_DIRECT_PATCH_RADIUS + 1);
    let wanted = (VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX / window).ceil() as u32;
    assert_eq!(wanted, 4, "ceil(48 / 13)");
    assert_eq!(VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE, 384);

    let cases = [
        (0u32, vec![1u32]),
        (383, vec![1]),
        (700, vec![1]),
        (767, vec![1]),
        (768, vec![2]),
        (1_151, vec![2]),
        (1_152, vec![3]),
        (1_200, vec![3]),
        (1_535, vec![3]),
        (1_536, vec![4]),
        (3_071, vec![4]),
        (3_072, vec![8, 4]),
        (6_143, vec![8, 4]),
        (6_144, vec![16, 8, 4]),
        (7_818, vec![16, 8, 4]),
        (12_287, vec![16, 8, 4]),
        (12_288, vec![32, 16, 8, 4]),
    ];
    for (long_side, expected) in cases {
        assert_eq!(
            virtual_tile_pyramid_factors(long_side),
            expected,
            "long side {long_side}"
        );
    }

    for long_side in (0..20_000u32).step_by(37) {
        let factors = virtual_tile_pyramid_factors(long_side);
        let largest = (long_side / VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE).max(1);
        let finest = *factors.last().expect("at least one level");
        assert_eq!(finest, wanted.min(largest), "long side {long_side}");
        assert!(
            factors.windows(2).all(|pair| pair[0] == 2 * pair[1]),
            "octaves coarsest first: {factors:?}"
        );
        assert!(factors[0] <= largest && factors[0] * 2 > largest);
        if factors[0] > 1 {
            assert!(long_side / factors[0] >= VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. match planes and pyramid
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_match_plane_rounds_block_means_and_never_leaks_uncovered_payload() {
    // Three 2x2 blocks: means 1.75 -> 2, 1.25 -> 1, 1.5 -> 2 (round half up).
    let gray = GrayImage::from_raw(6, 2, vec![1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2]).unwrap();
    let plane = VirtualTileMatchPlane::from_native(&gray, &full_coverage(6, 2), 2).unwrap();
    assert_eq!(plane.gray.dimensions(), (3, 1));
    assert_eq!(plane.gray.as_raw(), &vec![2, 1, 2]);
    assert_eq!(coverage_bits(&plane.coverage), vec![true, true, true]);

    // One uncovered pixel carrying a bright payload uncovers only its block.
    let mut leaking = gray.clone();
    leaking.put_pixel(3, 1, image::Luma([255]));
    let coverage = coverage_from_fn(6, 2, |x, y| (x, y) != (3, 1));
    let plane = VirtualTileMatchPlane::from_native(&leaking, &coverage, 2).unwrap();
    assert_eq!(plane.gray.as_raw(), &vec![2, 0, 2]);
    assert_eq!(coverage_bits(&plane.coverage), vec![true, false, true]);

    // Reference block average on a larger plane with scattered holes.
    let (width, height) = (9u32, 7u32);
    let uncovered = [(1u32, 0u32), (6, 5), (4, 4)];
    let mut gray = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([((x * 37 + y * 11) % 200) as u8 + 20])
    });
    for &(x, y) in &uncovered {
        gray.put_pixel(x, y, image::Luma([255]));
    }
    let coverage = coverage_from_fn(width, height, |x, y| !uncovered.contains(&(x, y)));
    for factor in [2u32, 3] {
        let plane = VirtualTileMatchPlane::from_native(&gray, &coverage, factor).unwrap();
        assert_eq!(plane.factor, factor);
        assert_eq!(plane.gray.dimensions(), (width / factor, height / factor));
        assert_eq!(plane.coverage.dimensions(), plane.gray.dimensions());
        for level_y in 0..height / factor {
            for level_x in 0..width / factor {
                let block = (0..factor)
                    .flat_map(|dy| {
                        (0..factor).map(move |dx| (level_x * factor + dx, level_y * factor + dy))
                    })
                    .collect::<Vec<_>>();
                let all_covered = block.iter().all(|&(x, y)| coverage.is_covered(x, y));
                let expected = if all_covered {
                    let sum = block
                        .iter()
                        .map(|&(x, y)| u32::from(gray.get_pixel(x, y)[0]))
                        .sum::<u32>();
                    (f64::from(sum) / f64::from(factor * factor)).round() as u8
                } else {
                    0
                };
                assert_eq!(
                    plane.coverage.is_covered(level_x, level_y),
                    all_covered,
                    "factor {factor} level ({level_x}, {level_y})"
                );
                assert_eq!(
                    plane.gray.get_pixel(level_x, level_y)[0],
                    expected,
                    "factor {factor} level ({level_x}, {level_y})"
                );
            }
        }
    }
}

#[test]
fn virtual_tile_match_plane_identity_factor_floor_dimensions_and_mismatches() {
    let (width, height) = (9u32, 7u32);
    let gray = textured_plane(width, height);
    let coverage = coverage_from_fn(width, height, |x, y| (x + y) % 5 != 0);
    for factor in [0u32, 1] {
        let plane = VirtualTileMatchPlane::from_native(&gray, &coverage, factor).unwrap();
        assert_eq!(plane.factor, 1);
        assert_eq!(plane.gray, gray, "factor {factor} keeps native pixels");
        assert_eq!(coverage_bits(&plane.coverage), coverage_bits(&coverage));
    }
    assert_eq!(
        VirtualTileMatchPlane::from_native(&gray, &coverage, 4)
            .unwrap()
            .gray
            .dimensions(),
        (2, 1)
    );
    assert!(
        VirtualTileMatchPlane::from_native(&gray, &coverage, 8).is_none(),
        "a level without rows is unusable"
    );
    let narrower = full_coverage(width - 1, height);
    for factor in [1u32, 2] {
        assert!(VirtualTileMatchPlane::from_native(&gray, &narrower, factor).is_none());
    }
}

#[test]
fn virtual_tile_match_plane_level_and_native_maps_are_exact_inverses_on_block_centres() {
    for factor in [0u32, 1, 2, 3, 4, 8, 16] {
        let to_native = VirtualTileMatchPlane::level_to_native(factor);
        let to_level = VirtualTileMatchPlane::native_to_level(factor);
        assert!((to_native * to_level - Matrix3::identity()).amax() < 1e-12);
        assert!((to_level * to_native - Matrix3::identity()).amax() < 1e-12);
        let scale = f64::from(factor.max(1));
        for (x, y) in [(0.0, 0.0), (3.0, 2.0), (17.0, 5.0), (0.25, 9.5)] {
            let native = transformed_point(&to_native, Point2::new(x, y)).unwrap();
            assert!((native.x - (scale * x + (scale - 1.0) * 0.5)).abs() < 1e-12);
            assert!((native.y - (scale * y + (scale - 1.0) * 0.5)).abs() < 1e-12);
            let back = transformed_point(&to_level, native).unwrap();
            assert!((back - Point2::new(x, y)).norm() < 1e-12);
        }
    }
    // Level pixel (0, 0) of a 4x level is the centre of native block [0, 4).
    let centre =
        transformed_point(&VirtualTileMatchPlane::level_to_native(4), Point2::origin()).unwrap();
    assert_eq!((centre.x, centre.y), (1.5, 1.5));
}

#[test]
fn virtual_tile_match_pyramid_set_station_fills_every_level_of_that_station_only() {
    let factors = vec![8u32, 4];
    let mut pyramid = VirtualTileMatchPyramid::new(factors.clone(), 3);
    assert_eq!(pyramid.factors, factors);
    assert_eq!(pyramid.levels.len(), 2);
    assert_eq!(pyramid.coverages.len(), 2);
    assert!(pyramid.levels.iter().all(|level| level.len() == 3));
    assert!(pyramid.coverages.iter().all(|level| level.len() == 3));
    assert_eq!(pyramid.native_dimensions, vec![(1, 1); 3]);
    assert_eq!(pyramid.quality_coverages.len(), 3);

    let (width, height) = (130u32, 70u32);
    let gray = textured_plane(width, height);
    let coverage = coverage_from_fn(width, height, |x, y| x >= 9 || y >= 5);
    let mut tile = plane_info(42, gray.clone());
    tile.scale_factor = 3.0;
    let planes = pyramid
        .set_station(1, &tile, &gray, &coverage)
        .expect("both levels are usable");
    assert_eq!(
        planes.iter().map(|plane| plane.factor).collect::<Vec<_>>(),
        factors
    );
    for (level, plane) in planes.iter().enumerate() {
        let factor = factors[level];
        let expected = VirtualTileMatchPlane::from_native(&gray, &coverage, factor).unwrap();
        assert_eq!(plane.gray, expected.gray);
        let info = &pyramid.levels[level][1];
        assert_eq!(info.id, 42);
        assert_eq!((info.width, info.height), (width / factor, height / factor));
        assert_eq!(info.alignment_image, expected.gray);
        assert_eq!(info.scale_factor, 1.0, "a level is its own pixel grid");
        assert_eq!(
            coverage_bits(&pyramid.coverages[level][1]),
            coverage_bits(&expected.coverage)
        );
        for untouched in [0, 2] {
            assert_eq!(pyramid.levels[level][untouched].id, usize::MAX);
            assert_eq!(
                pyramid.levels[level][untouched]
                    .alignment_image
                    .dimensions(),
                (1, 1)
            );
            assert_eq!(pyramid.coverages[level][untouched].dimensions(), (1, 1));
        }
    }
    assert_eq!(
        pyramid.native_dimensions,
        vec![(1, 1), (width, height), (1, 1)]
    );

    let tiny = GrayImage::new(6, 6);
    assert!(
        pyramid
            .set_station(2, &tile, &tiny, &full_coverage(6, 6))
            .is_none(),
        "a level below one pixel makes the station unusable"
    );
}

// ---------------------------------------------------------------------------
// 3. relation-quality plane and its installation
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_relation_quality_plane_blurs_the_plane_two_octaves_above_the_finest() {
    assert_eq!(VIRTUAL_TILE_RELATION_QUALITY_OCTAVES, 2);
    assert_eq!(VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA, 1.0);
    let gray = textured_plane(512, 256);
    let coverage = full_coverage(512, 256);

    let (empty, empty_factor) = virtual_tile_relation_quality_plane(&[]);
    assert_eq!((empty.dimensions(), empty_factor), ((1, 1), 1));

    for factor in [1u32, 4] {
        let planes = planes_for(&[factor], &gray, &coverage);
        let (plane, plane_factor) = virtual_tile_relation_quality_plane(&planes);
        assert_eq!(plane_factor, factor);
        assert_eq!(plane, planes[0].gray, "a single level stays unfiltered");
    }

    for (factors, expected) in [
        (vec![16u32, 8, 4], 16u32),
        (vec![8, 4], 8),
        (vec![2, 1], 2),
        (vec![4, 2], 4),
        (vec![32, 16, 8, 4], 16),
    ] {
        let planes = planes_for(&factors, &gray, &coverage);
        let (plane, plane_factor) = virtual_tile_relation_quality_plane(&planes);
        assert_eq!(plane_factor, expected, "factors {factors:?}");
        let source = planes
            .iter()
            .find(|plane| plane.factor == expected)
            .unwrap();
        assert_eq!(
            plane,
            imageproc::filter::gaussian_blur_f32(&source.gray, 1.0),
            "factors {factors:?}"
        );
        assert_ne!(plane, source.gray, "factors {factors:?} must be blurred");
    }
}

#[test]
fn install_virtual_tile_match_planes_pairs_the_quality_plane_with_its_own_coverage() {
    let (width, height) = (320u32, 200u32);
    let coverage = coverage_from_fn(width, height, |x, y| x >= 37 && y < 170);
    let texture = textured_plane(width, height);
    let gray = GrayImage::from_fn(width, height, |x, y| {
        if coverage.is_covered(x, y) {
            *texture.get_pixel(x, y)
        } else {
            image::Luma([0])
        }
    });
    let brief_pairs = processing::generate_brief_pairs();

    let mut pyramid = VirtualTileMatchPyramid::new(vec![8, 4], 2);
    let mut tile = plane_info(7, gray.clone());
    install_virtual_tile_match_planes(&mut tile, &mut pyramid, 1, &gray, &coverage, &brief_pairs)
        .expect("installed");
    let quality_level = VirtualTileMatchPlane::from_native(&gray, &coverage, 8).unwrap();
    assert_eq!(tile.scale_factor, 8.0);
    assert_eq!(tile.alignment_image.dimensions(), (40, 25));
    assert_eq!(
        tile.alignment_image,
        imageproc::filter::gaussian_blur_f32(&quality_level.gray, 1.0)
    );
    assert_eq!(
        pyramid.quality_coverages[1].dimensions(),
        tile.alignment_image.dimensions()
    );
    assert_eq!(
        coverage_bits(&pyramid.quality_coverages[1]),
        coverage_bits(&quality_level.coverage)
    );
    assert_eq!(pyramid.quality_coverages[0].dimensions(), (1, 1));
    let finest = VirtualTileMatchPlane::from_native(&gray, &coverage, 4).unwrap();
    assert!(tile.features.iter().all(|feature| {
        feature.keypoint.x < finest.gray.width() && feature.keypoint.y < finest.gray.height()
    }));

    let mut single = VirtualTileMatchPyramid::new(vec![1], 1);
    let mut native_tile = plane_info(8, gray.clone());
    install_virtual_tile_match_planes(
        &mut native_tile,
        &mut single,
        0,
        &gray,
        &coverage,
        &brief_pairs,
    )
    .expect("installed");
    assert_eq!(native_tile.scale_factor, 1.0);
    assert_eq!(native_tile.alignment_image, gray);
    assert_eq!(
        coverage_bits(&single.quality_coverages[0]),
        coverage_bits(&coverage)
    );

    let mut failing = VirtualTileMatchPyramid::new(vec![8, 4], 1);
    let tiny = GrayImage::new(5, 5);
    let mut tiny_tile = plane_info(9, tiny.clone());
    assert!(
        install_virtual_tile_match_planes(
            &mut tiny_tile,
            &mut failing,
            0,
            &tiny,
            &full_coverage(5, 5),
            &brief_pairs,
        )
        .is_err()
    );
}

// ---------------------------------------------------------------------------
// 4. coverage-aware requirement-6.8 measurements
// ---------------------------------------------------------------------------

/// 256x160 planes: the 60x36 sample grid sits at x = 4c + 2, y = 4r + 2.
const QUALITY_PLANE: (u32, u32) = (256, 160);
const QUALITY_GRID_SAMPLES: usize = 60 * 36;

#[test]
fn focus_overlap_quality_on_coverage_reads_only_covered_payload() {
    let (width, height) = QUALITY_PLANE;
    let texture = textured_plane(width, height);
    let hole = |x: u32| (96..176).contains(&x);
    let source = plane_info(1, texture.clone());
    let target = plane_info(
        2,
        GrayImage::from_fn(width, height, |x, y| {
            if hole(x) {
                image::Luma([0])
            } else {
                *texture.get_pixel(x, y)
            }
        }),
    );
    let source_coverage = full_coverage(width, height);
    let target_coverage = coverage_from_fn(width, height, |x, _| !hole(x));
    let identity = Matrix3::identity();

    let unmasked = focus_overlap_quality_measurements(&source, &target, &identity).unwrap();
    assert_eq!(unmasked.samples, QUALITY_GRID_SAMPLES);
    assert!(
        unmasked.low_frequency_mean_relative_difference > 0.1,
        "uncovered zeros contaminate the mean: {unmasked:?}"
    );
    assert!(
        unmasked.edge_strength_ratio > 1.1,
        "uncovered zeros contaminate the edges: {unmasked:?}"
    );

    let masked = focus_overlap_quality_measurements_on_coverage(
        &source,
        &target,
        &identity,
        Some((&source_coverage, &target_coverage)),
    )
    .unwrap();
    assert!(
        masked.low_frequency_mean_relative_difference < 1e-12,
        "{masked:?}"
    );
    assert!(
        (masked.edge_strength_ratio - 1.0).abs() < 1e-12,
        "{masked:?}"
    );
    assert!(masked.intensity_ncc > 0.999_999, "{masked:?}");
    assert!(masked.median_edge_orientation_difference_degrees < 1e-6);
    // Columns whose ±4px neighbourhood meets [96, 176) are excluded.
    let kept_columns = (2..62u32)
        .map(|column| 4 * column + 2)
        .filter(|&x| x + 4 < 96 || x - 4 > 175)
        .count();
    assert_eq!(kept_columns, 38);
    assert_eq!(masked.samples, kept_columns * 36);

    // Fully covered masks sample exactly what the unmasked measurement does.
    let full = focus_overlap_quality_measurements_on_coverage(
        &source,
        &target,
        &identity,
        Some((&source_coverage, &source_coverage)),
    )
    .unwrap();
    assert_eq!(full.samples, unmasked.samples);
    assert_eq!(
        full.low_frequency_mean_relative_difference,
        unmasked.low_frequency_mean_relative_difference
    );
    assert_eq!(full.edge_strength_ratio, unmasked.edge_strength_ratio);
}

#[test]
fn focus_overlap_quality_on_coverage_excludes_samples_within_four_pixels_of_uncovered_pixels() {
    assert_eq!(VIRTUAL_TILE_QUALITY_COVERAGE_RADIUS_PX, 4.0);
    let (width, height) = QUALITY_PLANE;
    let texture = textured_plane(width, height);
    let source = plane_info(1, texture.clone());
    let target = plane_info(2, texture);
    let full = full_coverage(width, height);
    let identity = Matrix3::identity();
    let baseline = focus_overlap_quality_measurements_on_coverage(
        &source,
        &target,
        &identity,
        Some((&full, &full)),
    )
    .unwrap()
    .samples;
    assert_eq!(baseline, QUALITY_GRID_SAMPLES);
    // (102, 50) is itself a grid sample: x ∈ {98, 102, 106} × y ∈ {46, 50, 54}
    // lie within 4px (radius 3 would exclude 1). (103, 51): only
    // x ∈ {102, 106} × y ∈ {50, 54} (radius 5 would exclude 9).
    for ((hole_x, hole_y), excluded) in [((102u32, 50u32), 9usize), ((103, 51), 4)] {
        let hole = coverage_from_fn(width, height, |x, y| (x, y) != (hole_x, hole_y));
        for (source_coverage, target_coverage, side) in
            [(&hole, &full, "source"), (&full, &hole, "target")]
        {
            let quality = focus_overlap_quality_measurements_on_coverage(
                &source,
                &target,
                &identity,
                Some((source_coverage, target_coverage)),
            )
            .unwrap();
            assert_eq!(
                quality.samples,
                baseline - excluded,
                "{side} hole at ({hole_x}, {hole_y})"
            );
        }
    }
}

#[test]
fn focus_overlap_quality_on_coverage_rejects_mismatched_masks_and_sparse_overlap() {
    let (width, height) = QUALITY_PLANE;
    let texture = textured_plane(width, height);
    let source = plane_info(1, texture.clone());
    let target = plane_info(2, texture);
    let full = full_coverage(width, height);
    let identity = Matrix3::identity();
    let measure = |source_coverage: &stitching::CoverageMask,
                   target_coverage: &stitching::CoverageMask| {
        focus_overlap_quality_measurements_on_coverage(
            &source,
            &target,
            &identity,
            Some((source_coverage, target_coverage)),
        )
    };
    let narrower = full_coverage(width - 1, height);
    let shorter = full_coverage(width, height - 1);
    assert!(measure(&narrower, &full).is_none());
    assert!(measure(&full, &narrower).is_none());
    assert!(measure(&shorter, &full).is_none());
    // Covered strip [0, 24): x ∈ {10, 14, 18} → 108 samples < 120.
    let sparse = coverage_from_fn(width, height, |x, _| x < 24);
    assert!(measure(&full, &sparse).is_none());
    assert!(measure(&sparse, &full).is_none());
    // Covered strip [0, 28): x ∈ {10, 14, 18, 22} → 144 samples.
    let enough = coverage_from_fn(width, height, |x, _| x < 28);
    assert_eq!(measure(&full, &enough).unwrap().samples, 144);
}

// ---------------------------------------------------------------------------
// 5. rendering-prior repair planning
// ---------------------------------------------------------------------------

#[test]
fn prior_free_seed_fallback_requires_a_fitted_polish_support_failure() {
    assert!(virtual_tile_prior_free_seed_fallback_conditions(
        true,
        "insufficient_polished_support",
        STATION_RELATION_MIN_INLIERS,
        STATION_RELATION_MIN_INLIERS,
    ));
    for (model_fitted, stage, fitted, prior_free) in [
        (false, "insufficient_polished_support", 24, 24),
        (true, "fitted", 24, 24),
        (true, "insufficient_polished_support", 23, 24),
        (true, "insufficient_polished_support", 24, 23),
    ] {
        assert!(!virtual_tile_prior_free_seed_fallback_conditions(
            model_fitted,
            stage,
            fitted,
            prior_free,
        ));
    }
}

#[test]
fn plan_virtual_tile_prior_repairs_needs_proposals_and_two_clusters() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(4);
    let relation = sample_relation(-372.0, 9.0);
    assert!(
        plan_virtual_tile_prior_repairs(4, &[(0, 1), (2, 3)], &[], &tile_to_world, &topology)
            .is_empty()
    );
    assert!(
        plan_virtual_tile_prior_repairs(
            4,
            &[(0, 1), (1, 2), (2, 3)],
            &[prior_proposal(1, 2, relation, 50)],
            &tile_to_world,
            &topology,
        )
        .is_empty(),
        "one authoritative cluster needs no repair"
    );
    assert!(
        plan_virtual_tile_prior_repairs(
            3,
            &[],
            &[prior_proposal(0, 1, relation, 50)],
            &tile_to_world,
            &topology,
        )
        .is_empty(),
        "placements must match the station count"
    );
    assert!(
        plan_virtual_tile_prior_repairs(
            1,
            &[],
            &[prior_proposal(0, 1, relation, 50)],
            &tile_to_world[..1],
            &topology,
        )
        .is_empty()
    );
}

#[test]
fn plan_virtual_tile_prior_repairs_moves_the_whole_unfixed_cluster_onto_the_proposal() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(4);
    let authoritative = [(0, 1), (2, 3)];
    let relation = sample_relation(-372.0, 9.0);

    // The fixed station is the proposal's left: H(u→f) = (left→right)⁻¹.
    let repairs = plan_virtual_tile_prior_repairs(
        4,
        &authoritative,
        &[prior_proposal(1, 2, relation, 50)],
        &tile_to_world,
        &topology,
    );
    assert_eq!(repaired_stations(&repairs), vec![2, 3]);
    for repair in &repairs {
        assert_eq!(repair.fixed_station, 1);
        assert_eq!(repair.relation, (1, 2));
        assert_eq!(repair.relation_inliers, 50);
        assert_eq!(repair.world_correction, repairs[0].world_correction);
    }
    let correction = repairs[0].world_correction;
    assert_projectively_equal(
        &(correction * tile_to_world[2]),
        &(tile_to_world[1] * relation.try_inverse().unwrap()),
        "left fixed",
    );

    // The fixed station is the proposal's right: H(u→f) = left→right.
    let repairs = plan_virtual_tile_prior_repairs(
        4,
        &authoritative,
        &[prior_proposal(2, 1, relation, 50)],
        &tile_to_world,
        &topology,
    );
    assert_eq!(repaired_stations(&repairs), vec![2, 3]);
    assert!(repairs.iter().all(|repair| repair.fixed_station == 1
        && repair.relation == (2, 1)
        && repair.world_correction == repairs[0].world_correction));
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[2]),
        &(tile_to_world[1] * relation),
        "right fixed",
    );
}

#[test]
fn plan_virtual_tile_prior_repairs_keeps_the_largest_cluster_and_breaks_ties_by_position() {
    let tile_to_world = sample_tile_to_world();
    let relation = sample_relation(-372.0, 9.0);
    let proposals = [prior_proposal(0, 1, relation, 40)];

    // {0} against {1, 2}: the larger cluster is the reference.
    let repairs = plan_virtual_tile_prior_repairs(
        3,
        &[(1, 2)],
        &proposals,
        &tile_to_world[..3],
        &line_topology(3),
    );
    assert_eq!(repaired_stations(&repairs), vec![0]);
    assert_eq!(repairs[0].fixed_station, 1);

    // Equal sizes: the smallest row/column key is the reference.
    let repairs =
        plan_virtual_tile_prior_repairs(2, &[], &proposals, &tile_to_world[..2], &line_topology(2));
    assert_eq!(repaired_stations(&repairs), vec![1]);
    assert_eq!(repairs[0].fixed_station, 0);
    let reordered = topology::StationTopology {
        row: vec![1, 0],
        column: vec![0, 3],
        ..topology::StationTopology::default()
    };
    let repairs =
        plan_virtual_tile_prior_repairs(2, &[], &proposals, &tile_to_world[..2], &reordered);
    assert_eq!(repaired_stations(&repairs), vec![0]);
    assert_eq!(repairs[0].fixed_station, 1);
}

#[test]
fn plan_virtual_tile_prior_repairs_anchors_a_disconnected_proposal_component() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(4);
    let repairs = plan_virtual_tile_prior_repairs(
        4,
        &[(0, 1)],
        &[prior_proposal(2, 3, sample_relation(-372.0, 9.0), 80)],
        &tile_to_world,
        &topology,
    );
    assert_eq!(repaired_stations(&repairs), vec![3]);
    assert_eq!(repairs[0].fixed_station, 2);
    assert_eq!(repairs[0].relation, (2, 3));
}

#[test]
fn plan_virtual_tile_prior_repairs_prefers_inliers_and_composes_repairs_transitively() {
    let tile_to_world = sample_tile_to_world();
    let weak_relation = sample_relation(-372.0, 9.0);
    let strong_relation = sample_relation(-375.5, 6.0);
    let weak = prior_proposal(0, 1, weak_relation, 50);
    let strong = prior_proposal(0, 1, strong_relation, 80);
    let topology = line_topology(2);
    let repairs =
        plan_virtual_tile_prior_repairs(2, &[], &[weak, strong], &tile_to_world[..2], &topology);
    assert_eq!(
        repaired_stations(&repairs),
        vec![1],
        "repaired at most once"
    );
    assert_eq!(repairs[0].relation_inliers, 80);
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[1]),
        &(tile_to_world[0] * strong_relation.try_inverse().unwrap()),
        "strongest proposal",
    );
    assert_eq!(
        plan_virtual_tile_prior_repairs(2, &[], &[strong, weak], &tile_to_world[..2], &topology),
        repairs,
        "proposal order is irrelevant"
    );

    // Three singleton clusters: 0→2 (90) first, then 2→1 (60) beats 0→1 (40)
    // and composes station 2's correction.
    let relation_01 = sample_relation(-372.0, 9.0);
    let relation_02 = sample_relation(-765.0, 14.0);
    let relation_21 = sample_relation(391.0, 12.0);
    let repairs = plan_virtual_tile_prior_repairs(
        3,
        &[],
        &[
            prior_proposal(0, 1, relation_01, 40),
            prior_proposal(0, 2, relation_02, 90),
            prior_proposal(2, 1, relation_21, 60),
        ],
        &tile_to_world[..3],
        &line_topology(3),
    );
    assert_eq!(repaired_stations(&repairs), vec![1, 2], "sorted by station");
    let (repair_1, repair_2) = (&repairs[0], &repairs[1]);
    assert_eq!(
        (
            repair_2.fixed_station,
            repair_2.relation,
            repair_2.relation_inliers
        ),
        (0, (0, 2), 90)
    );
    assert_projectively_equal(
        &(repair_2.world_correction * tile_to_world[2]),
        &(tile_to_world[0] * relation_02.try_inverse().unwrap()),
        "first repair",
    );
    assert_eq!(
        (
            repair_1.fixed_station,
            repair_1.relation,
            repair_1.relation_inliers
        ),
        (2, (2, 1), 60)
    );
    assert_projectively_equal(
        &(repair_1.world_correction * tile_to_world[1]),
        &(repair_2.world_correction * tile_to_world[2] * relation_21.try_inverse().unwrap()),
        "transitive repair",
    );
}

#[test]
fn plan_virtual_tile_prior_repairs_ignores_invalid_proposals_and_skips_singular_relations() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(2);
    let relation = sample_relation(-372.0, 9.0);
    assert!(
        plan_virtual_tile_prior_repairs(
            2,
            &[],
            &[
                prior_proposal(0, 2, relation, 99),
                prior_proposal(5, 0, relation, 99),
                prior_proposal(1, 1, relation, 99),
            ],
            &tile_to_world[..2],
            &topology,
        )
        .is_empty()
    );
    // Station 0 is the reference. A zero relation cannot be inverted (fixed
    // left) nor normalised (fixed right); the valid weaker proposal repairs.
    let repairs = plan_virtual_tile_prior_repairs(
        2,
        &[],
        &[
            prior_proposal(0, 1, Matrix3::zeros(), 99),
            prior_proposal(1, 0, Matrix3::zeros(), 98),
            prior_proposal(0, 1, relation, 30),
        ],
        &tile_to_world[..2],
        &topology,
    );
    assert_eq!(repaired_stations(&repairs), vec![1]);
    assert_eq!(repairs[0].relation_inliers, 30);
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[1]),
        &(tile_to_world[0] * relation.try_inverse().unwrap()),
        "valid fallback",
    );
}

// ---------------------------------------------------------------------------
// 6. bounded residual correction
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_residual_correction_is_bounded_reads_local_scale_over_the_evidence_region() {
    assert_eq!(VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX, 24.0);
    assert_eq!(VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX, 8.0);
    assert_eq!(VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MIN, 0.98);
    assert_eq!(VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MAX, 1.02);
    let identity = Matrix3::identity();
    let evidence = (400.0, 200.0, 600.0, 400.0);
    let centre = Point2::new(500.0, 300.0);

    // Upper-left block 1.035, but sqrt(det J) = a / w^1.5 = 1 at the centre.
    let block: f64 = 1.035;
    let perspective = (block.powf(2.0 / 3.0) - 1.0) / centre.x;
    let projective = Matrix3::new(block, 0.0, 0.0, 0.0, block, 0.0, perspective, 0.0, 1.0);
    assert!((numeric_local_scale(&projective, centre) - 1.0).abs() < 1e-6);
    assert!(virtual_tile_residual_correction_is_bounded(
        &projective,
        &identity,
        evidence
    ));
    assert_eq!(
        virtual_tile_full_tile_bounds((1000, 700)),
        (0.0, 0.0, 1000.0, 700.0)
    );
    assert!(
        !virtual_tile_residual_correction_is_bounded(
            &projective,
            &identity,
            virtual_tile_full_tile_bounds((1000, 700)),
        ),
        "the full tile reaches the origin, where the local scale is 1.035"
    );
    // The bounds are evaluated where the initial relation predicts the evidence.
    assert!(!virtual_tile_residual_correction_is_bounded(
        &projective,
        &translation(-300.0, 0.0),
        evidence
    ));
    let shifted_perspective = (block.powf(2.0 / 3.0) - 1.0) / 200.0;
    let shifted = Matrix3::new(
        block,
        0.0,
        0.0,
        0.0,
        block,
        0.0,
        shifted_perspective,
        0.0,
        1.0,
    );
    assert!(virtual_tile_residual_correction_is_bounded(
        &shifted,
        &translation(-300.0, 0.0),
        evidence
    ));
}

#[test]
fn virtual_tile_residual_correction_is_bounded_rejects_scale_displacement_and_non_translation() {
    let identity = Matrix3::identity();
    let evidence = (400.0, 200.0, 600.0, 400.0);
    let centre = (500.0, 300.0);
    let bounded = |correction: Matrix3<f64>| {
        virtual_tile_residual_correction_is_bounded(&correction, &identity, evidence)
    };
    // True scale about the centre moves corners by at most 4.3px.
    for (scale, expected) in [
        (1.0, true),
        (1.015, true),
        (1.019, true),
        (1.021, false),
        (1.03, false),
        (0.981, true),
        (0.979, false),
        (0.97, false),
    ] {
        assert_eq!(
            bounded(about_center(centre, uniform_scale(scale))),
            expected,
            "scale {scale}"
        );
    }
    for (dx, dy, expected) in [
        (23.5, 0.0, true),
        (0.0, -23.5, true),
        (24.5, 0.0, false),
        (17.0, 17.0, false),
    ] {
        assert_eq!(
            bounded(translation(dx, dy)),
            expected,
            "translation ({dx}, {dy})"
        );
    }
    // A rotation about the centre moves the corners by 2·sin(θ/2)·141.4px.
    for (theta, expected) in [(0.05, true), (0.06, false), (-0.06, false)] {
        assert_eq!(
            bounded(about_center(centre, rotation(theta))),
            expected,
            "rotation {theta}"
        );
    }
    // Each part passes alone; together a corner moves 25.6px.
    assert!(bounded(translation(20.0, 0.0)));
    assert!(!bounded(
        translation(20.0, 0.0) * about_center(centre, rotation(0.05))
    ));
}

// ---------------------------------------------------------------------------
// 9. intra-station frame records
// ---------------------------------------------------------------------------

#[test]
fn push_frame_record_replaces_a_re_measured_frame_in_place() {
    let frame = |path: &str, inliers: usize| report::IntraStationFrameRecord {
        path: path.to_string(),
        inliers,
        ..report::IntraStationFrameRecord::default()
    };
    let summary = |station: &report::IntraStationReport| {
        station
            .frames
            .iter()
            .map(|frame| (frame.path.clone(), frame.inliers))
            .collect::<Vec<_>>()
    };
    let mut records = Vec::new();
    intra_station::push_frame_record(&mut records, 3, "anchor-a.raw", frame("a.raw", 10));
    intra_station::push_frame_record(&mut records, 3, "anchor-a.raw", frame("b.raw", 20));
    intra_station::push_frame_record(&mut records, 5, "anchor-c.raw", frame("c.raw", 30));
    // A later render re-measures a.raw: replaced in place, anchor unchanged.
    intra_station::push_frame_record(&mut records, 3, "other-anchor.raw", frame("a.raw", 11));
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].station_index, 3);
    assert_eq!(records[0].anchor_path, "anchor-a.raw");
    assert_eq!(
        summary(&records[0]),
        vec![("a.raw".to_string(), 11), ("b.raw".to_string(), 20)]
    );
    intra_station::push_frame_record(&mut records, 3, "anchor-a.raw", frame("d.raw", 40));
    assert_eq!(
        summary(&records[0]),
        vec![
            ("a.raw".to_string(), 11),
            ("b.raw".to_string(), 20),
            ("d.raw".to_string(), 40)
        ]
    );
    assert_eq!(records[1].station_index, 5);
    assert_eq!(summary(&records[1]), vec![("c.raw".to_string(), 30)]);

    // An empty anchor is filled by the first record that names one.
    intra_station::push_frame_record(&mut records, 7, "", frame("e.raw", 1));
    assert_eq!(records[2].anchor_path, "");
    intra_station::push_frame_record(&mut records, 7, "anchor-e.raw", frame("e.raw", 2));
    assert_eq!(records[2].anchor_path, "anchor-e.raw");
    assert_eq!(summary(&records[2]), vec![("e.raw".to_string(), 2)]);
}

// ---------------------------------------------------------------------------
// 10. report serde defaults
// ---------------------------------------------------------------------------

fn sample_candidate_record() -> report::StationRelationCandidateRecord {
    let distribution = report::ResidualDistributionRecord {
        count: 3,
        min: -1.0,
        p10: -0.5,
        median: 0.25,
        p90: 0.75,
        max: 1.5,
    };
    let stage = |reason: &str| report::ResidualModelStageRecord {
        candidate_inliers: 42,
        final_support: 0.4,
        bounded_correction: Some(true),
        rejection_reason: reason.to_string(),
    };
    report::StationRelationCandidateRecord {
        left: 0,
        right: 1,
        predicted_overlap_area_px: 12_345.0,
        model_fitted: true,
        evidence_kind: report::StationRelationEvidenceKind::VirtualTile,
        grid_probe_count: 120,
        measurable_patch_count: 90,
        bidirectional_match_count: 70,
        occupied_grid_cells: 12,
        search_radius_px: 16.0,
        residual_dx_px: distribution,
        residual_dy_px: distribution,
        residual_magnitude_px: distribution,
        max_residual_translation_consensus: 60,
        residual_translation: stage("accepted"),
        residual_affine: stage("not_run"),
        residual_projective: stage("not_run"),
        residual_model: "polished_projective".to_string(),
        fitted_inliers: 64,
        hull_support: 0.31,
        median_prefit_structural_orientation_difference_degrees: Some(4.5),
        median_fitted_orientation_difference_degrees: None,
        failure_stage: "fitted".to_string(),
        pyramid_factors: vec![16, 8, 4],
        measurement_factor: 4,
        seed: "prior_free_features".to_string(),
        prior_free_inliers: 77,
        after_prior_repair: true,
    }
}

#[test]
fn station_relation_records_deserialise_without_checkpoint_ten_fields() {
    let candidate = sample_candidate_record();
    let value = serde_json::to_value(&candidate).unwrap();
    assert_eq!(
        serde_json::from_value::<report::StationRelationCandidateRecord>(value.clone()).unwrap(),
        candidate
    );
    let mut legacy = value;
    let object = legacy.as_object_mut().unwrap();
    for key in [
        "predicted_overlap_area_px",
        "model_fitted",
        "pyramid_factors",
        "measurement_factor",
        "seed",
        "prior_free_inliers",
        "after_prior_repair",
    ] {
        assert!(object.remove(key).is_some(), "{key} is serialised");
    }
    let parsed = serde_json::from_value::<report::StationRelationCandidateRecord>(legacy).unwrap();
    assert_eq!(
        parsed,
        report::StationRelationCandidateRecord {
            predicted_overlap_area_px: 0.0,
            model_fitted: false,
            pyramid_factors: Vec::new(),
            measurement_factor: 0,
            seed: String::new(),
            prior_free_inliers: 0,
            after_prior_repair: false,
            ..candidate.clone()
        }
    );

    let repair = report::StationPriorRepairRecord {
        station: 2,
        fixed_station: 1,
        relation_left: 1,
        relation_right: 2,
        relation_inliers: 64,
        center_shift_px: 12.5,
        correction_scale_ratio: 1.004,
    };
    let text = serde_json::to_string(&repair).unwrap();
    assert_eq!(
        serde_json::from_str::<report::StationPriorRepairRecord>(&text).unwrap(),
        repair
    );

    let relations = report::StationRelationsReport {
        candidates: vec![candidate],
        prior_repairs: vec![repair],
        ..report::StationRelationsReport::default()
    };
    let value = serde_json::to_value(&relations).unwrap();
    assert_eq!(
        serde_json::from_value::<report::StationRelationsReport>(value.clone()).unwrap(),
        relations
    );
    let mut legacy = value;
    legacy.as_object_mut().unwrap().remove("prior_repairs");
    assert_eq!(
        serde_json::from_value::<report::StationRelationsReport>(legacy).unwrap(),
        report::StationRelationsReport {
            prior_repairs: Vec::new(),
            ..relations
        }
    );
}

// ---------------------------------------------------------------------------
// 8. polish helpers
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_ncc_and_best_window_offset_recover_integer_offsets() {
    let run = [1.0, 4.0, 2.0, 8.0, 5.0];
    assert!((virtual_tile_ncc(&run, &run).unwrap() - 1.0).abs() < 1e-12);
    assert!((virtual_tile_ncc(&run, &run.map(|value| -value)).unwrap() + 1.0).abs() < 1e-12);
    assert!(
        (virtual_tile_ncc(&run, &run.map(|value| 3.0 * value + 7.0)).unwrap() - 1.0).abs() < 1e-12
    );
    assert!(virtual_tile_ncc(&run, &[5.0; 5]).is_none(), "constant run");
    assert!(
        virtual_tile_ncc(&run, &run[..4]).is_none(),
        "length mismatch"
    );
    assert!(virtual_tile_ncc(&[], &[]).is_none());

    let (half, search) = (3i32, 4i32);
    let side = (2 * (half + search) + 1) as usize;
    let template_side = (2 * half + 1) as usize;
    let window = (0..side * side)
        .map(|index| textured_value((index % side) as f64 * 3.1, (index / side) as f64 * 2.7))
        .collect::<Vec<_>>();
    for (offset_x, offset_y) in [(0, 0), (2, -3), (-4, 4), (1, 1)] {
        let template = (0..template_side * template_side)
            .map(|index| {
                let x = (index % template_side) as i32 + search + offset_x;
                let y = (index / template_side) as i32 + search + offset_y;
                window[y as usize * side + x as usize]
            })
            .collect::<Vec<_>>();
        let ((best_x, best_y), best, second, scores) =
            virtual_tile_best_window_offset(&template, &window, half, search).unwrap();
        assert_eq!((best_x, best_y), (offset_x, offset_y));
        assert!((best - 1.0).abs() < 1e-9);
        assert!(second < best);
        assert_eq!(scores.len(), ((2 * search + 1) * (2 * search + 1)) as usize);
    }
}

#[test]
fn virtual_tile_polish_correspondences_skip_low_texture_templates() {
    let full = full_coverage(96, 96);
    let textured = textured_plane(96, 96);
    // Alternating 127/129 has a standard deviation of 1 < 3 levels.
    let low_texture = GrayImage::from_fn(96, 96, |x, y| {
        image::Luma([if (x + y) % 2 == 0 { 127 } else { 129 }])
    });
    let bounds = (0.0, 0.0, 95.0, 95.0);
    let mut counts = VirtualTilePolishCounts::default();
    let matches = virtual_tile_polish_correspondences(
        &low_texture,
        &full,
        &textured,
        &full,
        &Matrix3::identity(),
        bounds,
        4,
        &mut counts,
    );
    assert_eq!(counts.probes, 25);
    assert_eq!(counts.textured, 0);
    assert_eq!(counts.matched, 0);
    assert!(matches.is_empty());

    let mut counts = VirtualTilePolishCounts::default();
    virtual_tile_polish_correspondences(
        &textured,
        &full,
        &textured,
        &full,
        &Matrix3::identity(),
        bounds,
        4,
        &mut counts,
    );
    assert_eq!(counts.probes, 25);
    assert_eq!(counts.textured, counts.probes);
}

/// Deterministic zero-mean pseudo-noise in `[-amplitude, amplitude]`.
fn hashed_noise(x: u32, y: u32, seed: u32, amplitude: f64) -> f64 {
    let mut value = x
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(y.wrapping_mul(0x85EB_CA77))
        .wrapping_add(seed.wrapping_mul(0xC2B2_AE3D));
    value ^= value >> 15;
    value = value.wrapping_mul(0x2C1B_3C6D);
    value ^= value >> 12;
    (f64::from(value % 2_001) / 1_000.0 - 1.0) * amplitude
}

fn candidate_topology(station_count: usize) -> topology::StationTopology {
    topology::StationTopology {
        row: vec![0; station_count],
        column: (0..station_count as u32).collect(),
        candidates: (0..station_count)
            .flat_map(|left| {
                (left + 1..station_count).map(move |right| topology::CandidateAdjacency {
                    left,
                    right,
                    kind: report::AdjacencyKind::SameRow,
                    score: 1.0,
                })
            })
            .collect(),
        ..topology::StationTopology::default()
    }
}

fn measured_relation(
    homography: Matrix3<f64>,
    points: Vec<(Point2<f64>, Point2<f64>)>,
) -> MatchInfo {
    MatchInfo {
        homography,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: false,
        candidate_points: points.clone(),
        points,
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
    }
}

// ---------------------------------------------------------------------------
// 7. residual model selection
// ---------------------------------------------------------------------------

/// `rows × columns` measured pairs of `correction · initial` on a 500×400 tile.
fn residual_selection_points(
    initial: &Matrix3<f64>,
    correction: &Matrix3<f64>,
    columns: usize,
    rows: usize,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let measured = correction * initial;
    (0..rows)
        .flat_map(|row| {
            (0..columns).map(move |column| {
                let source = Point2::new(
                    40.0 + column as f64 * 420.0 / (columns - 1) as f64,
                    35.0 + row as f64 * 330.0 / (rows - 1) as f64,
                );
                (source, transformed_point(&measured, source).unwrap())
            })
        })
        .collect()
}

fn fit_residual_model(
    points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    selection: VirtualTileResidualSelection,
) -> VirtualTileResidualModelOutcome {
    virtual_tile_fit_measured_residual_model_with_threshold(
        points,
        initial,
        (500, 400),
        (500, 400),
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        virtual_tile_full_tile_bounds((500, 400)),
        selection,
        StationOverlapMeasure::Rectangles,
    )
}

#[test]
fn most_inliers_selection_keeps_the_richer_model_a_passing_translation_would_truncate() {
    let initial = translation(-90.0, 5.0);
    // A 0.9° rotation about the tile centre: a translation explains only the
    // central band within 3px; the bounded affine explains every pair.
    let correction = translation(1.5, -2.0) * about_center((250.0, 200.0), rotation(0.0157));
    let points = residual_selection_points(&initial, &correction, 10, 8);

    let lowest = fit_residual_model(
        &points,
        &initial,
        VirtualTileResidualSelection::LowestComplexity,
    );
    let lowest_fit = lowest.fit.expect("the translation stage passes on its own");
    assert_eq!(lowest_fit.model, "translation");
    assert_eq!(lowest.translation.rejection_reason, "accepted");
    assert_eq!(lowest.affine.rejection_reason, "not_run");
    assert!(lowest_fit.inlier_indices.len() < points.len());

    let most = fit_residual_model(&points, &initial, VirtualTileResidualSelection::MostInliers);
    let most_fit = most.fit.expect("every stage is evaluated");
    assert_eq!(most.translation.rejection_reason, "accepted");
    assert_eq!(most.affine.rejection_reason, "accepted");
    assert_ne!(most_fit.model, "translation");
    assert_eq!(most_fit.inlier_indices.len(), points.len());
    assert!(most_fit.inlier_indices.len() > lowest_fit.inlier_indices.len());
    for (source, target) in &points {
        let predicted = transformed_point(&most_fit.measured_homography, *source).unwrap();
        assert!((predicted - target).norm() < 1e-6);
    }
}

#[test]
fn most_inliers_selection_breaks_an_inlier_tie_by_strictly_lower_median_error() {
    let initial = translation(-90.0, 5.0);
    // Small enough that the translation keeps every pair within 3px, yet
    // leaves a measurable error that the affine stage removes.
    let correction = translation(1.5, -2.0) * about_center((250.0, 200.0), rotation(0.004));
    let points = residual_selection_points(&initial, &correction, 10, 8);

    let lowest = fit_residual_model(
        &points,
        &initial,
        VirtualTileResidualSelection::LowestComplexity,
    );
    let lowest_fit = lowest.fit.expect("translation passes");
    assert_eq!(lowest_fit.model, "translation");
    assert_eq!(lowest_fit.inlier_indices.len(), points.len());
    assert!(lowest_fit.median_error_px > 0.1);

    let most = fit_residual_model(&points, &initial, VirtualTileResidualSelection::MostInliers);
    let most_fit = most.fit.expect("every stage is evaluated");
    assert_eq!(most_fit.inlier_indices.len(), points.len());
    assert_ne!(
        most_fit.model, "translation",
        "the tie goes to the lower error"
    );
    assert!(most_fit.median_error_px < lowest_fit.median_error_px);
}

// ---------------------------------------------------------------------------
// 8. polish fit and polished station match
// ---------------------------------------------------------------------------

fn polish_truth() -> Matrix3<f64> {
    translation(-60.0, 8.0)
        * about_center((240.0, 180.0), rotation(0.008) * uniform_scale(1.004))
        * Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.000_012, -0.000_008, 1.0)
}

#[test]
fn virtual_tile_polish_fit_recovers_a_projective_relation_and_drops_outliers() {
    let truth = polish_truth();
    let mut correspondences = (0..8)
        .flat_map(|row| (0..8).map(move |column| (row, column)))
        .map(|(row, column)| {
            let source = Point2::new(90.0 + 45.0 * column as f64, 40.0 + 40.0 * row as f64);
            let target = transformed_point(&truth, source).unwrap()
                + nalgebra::Vector2::new(
                    hashed_noise(column, row, 1, 0.3),
                    hashed_noise(column, row, 2, 0.3),
                );
            (source, target)
        })
        .collect::<Vec<_>>();
    let true_count = correspondences.len();
    for outlier in 0..16u32 {
        let source = Point2::new(
            100.0 + 21.0 * f64::from(outlier),
            60.0 + 13.0 * f64::from(outlier),
        );
        let offset = nalgebra::Vector2::new(
            25.0 + 2.0 * f64::from(outlier),
            -30.0 + 3.0 * f64::from(outlier % 5),
        );
        correspondences.push((source, transformed_point(&truth, source).unwrap() + offset));
    }
    let threshold = VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX;
    let (fitted, inliers) = virtual_tile_polish_fit(&correspondences, threshold, (480, 360))
        .expect("64 consistent pairs among 16 outliers");
    assert_eq!(inliers, (0..true_count).collect::<Vec<_>>());
    for (x, y) in [
        (90.0, 40.0),
        (405.0, 40.0),
        (90.0, 320.0),
        (405.0, 320.0),
        (240.0, 180.0),
    ] {
        let point = Point2::new(x, y);
        let error = (transformed_point(&fitted, point).unwrap()
            - transformed_point(&truth, point).unwrap())
        .norm();
        assert!(error < 0.3, "{error} px at ({x}, {y})");
    }

    // Fewer than 24 consistent pairs is no relation.
    assert!(virtual_tile_polish_fit(&correspondences[..20], threshold, (480, 360)).is_none());
    // A mirrored relation is never a valid planar station relation.
    let mirror = Matrix3::new(-1.0, 0.0, 480.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let mirrored = correspondences[..true_count]
        .iter()
        .map(|&(source, _)| (source, transformed_point(&mirror, source).unwrap()))
        .collect::<Vec<_>>();
    assert!(virtual_tile_polish_fit(&mirrored, threshold, (480, 360)).is_none());
}

#[test]
fn virtual_tile_polished_station_match_recovers_a_perturbed_seed_on_noisy_planes() {
    let (width, height) = (480u32, 360u32);
    let truth = polish_truth();
    let inverse = truth.try_inverse().unwrap();
    let left = GrayImage::from_fn(width, height, |x, y| {
        let value = textured_value(f64::from(x), f64::from(y)) + hashed_noise(x, y, 3, 3.0);
        image::Luma([value.clamp(0.0, 255.0).round() as u8])
    });
    let right = GrayImage::from_fn(width, height, |x, y| {
        let source = transformed_point(&inverse, Point2::new(f64::from(x), f64::from(y))).unwrap();
        let value = textured_value(source.x, source.y) + hashed_noise(x, y, 4, 3.0);
        image::Luma([value.clamp(0.0, 255.0).round() as u8])
    });
    let (left_info, right_info) = (plane_info(0, left), plane_info(1, right));
    let coverage = full_coverage(width, height);
    // The seed misses by a 2.5px shift and a 0.15° rotation: inside the
    // first ±8px pass, outside the requirement-6.3 3px tolerance at the edges.
    let seed = translation(2.5, -1.5) * about_center((240.0, 180.0), rotation(0.0026)) * truth;
    let mut diagnostic = sample_candidate_record();
    let relation = virtual_tile_polished_station_match(
        &left_info,
        &coverage,
        &right_info,
        &coverage,
        1,
        &seed,
        ((width, height), (width, height)),
        &mut diagnostic,
    )
    .unwrap_or_else(|| panic!("polish must recover the seed: {diagnostic:?}"));
    assert_eq!(diagnostic.failure_stage, "fitted");
    assert_eq!(diagnostic.residual_model, "polished_projective");
    assert!(relation.inliers >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.hull_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT);
    assert!(relation.canonical_homography.is_none());
    for (x, y) in [
        (120.0, 60.0),
        (400.0, 60.0),
        (120.0, 300.0),
        (400.0, 300.0),
        (260.0, 180.0),
    ] {
        let point = Point2::new(x, y);
        let error = (transformed_point(&relation.homography, point).unwrap()
            - transformed_point(&truth, point).unwrap())
        .norm();
        assert!(error < 0.75, "{error} px at ({x}, {y})");
    }
}

// ---------------------------------------------------------------------------
// 4b. coverage-aware requirement-6.8 re-check through the station solver
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_solver_rechecks_requirement_6_8_on_covered_pixels_only() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    // Tile 1 shows tile 0 shifted by (40, 6) but carries a zero-valued,
    // uncovered band x ∈ [150, 250) inside the overlap.
    let (width, height) = (400u32, 300u32);
    let band = |x: u32| (150..250).contains(&x);
    let left = textured_plane(width, height);
    let right = GrayImage::from_fn(width, height, |x, y| {
        if band(x) {
            image::Luma([0])
        } else {
            image::Luma([textured_value(f64::from(x) + 40.0, f64::from(y) + 6.0).round() as u8])
        }
    });
    let tiles = vec![plane_info(0, left), plane_info(1, right)];
    let left_to_right = translation(-40.0, -6.0);
    let points = [60.0, 100.0, 140.0, 180.0, 300.0, 340.0, 380.0]
        .into_iter()
        .flat_map(|x| (0..6).map(move |row| Point2::new(x, 30.0 + 50.0 * f64::from(row))))
        .map(|source| (source, transformed_point(&left_to_right, source).unwrap()))
        .collect::<Vec<_>>();
    let matches = HashMap::from([((0, 1), measured_relation(left_to_right, points))]);
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, left_to_right.try_inverse().unwrap()),
    ]);
    let station_topology = candidate_topology(2);
    let solve = |coverages: &[stitching::CoverageMask]| {
        let mut closure = report::ClosureReport::default();
        let mut relations = report::StationRelationsReport::default();
        let solved = solve_virtual_tile_station_poses_with_report(
            &tiles,
            VirtualTileSolverCoverages {
                quality: coverages,
                overlap: coverages,
                overlap_factor: 1,
            },
            &matches,
            &initial,
            &station_topology,
            &mut closure,
            &mut relations,
        );
        (solved, relations)
    };

    let masked = [
        full_coverage(width, height),
        coverage_from_fn(width, height, |x, _| !band(x)),
    ];
    let (solved, relations) = solve(&masked);
    assert_eq!(solved.len(), 2, "{relations:?}");
    assert_eq!(relations.accepted.len(), 1);

    // Treating the band as content lets its zeros reach the low-frequency and
    // edge-strength statistics, which then reject the same relation.
    let unmasked = [full_coverage(width, height), full_coverage(width, height)];
    let (solved, relations) = solve(&unmasked);
    assert!(solved.is_empty(), "{relations:?}");
    assert!(relations.accepted.is_empty());
    assert!(
        relations
            .rejected_by_reason
            .contains_key("station_relation_photometric_mismatch"),
        "{relations:?}"
    );
}

// ---------------------------------------------------------------------------
// checkpoint 10: closure on a synthetic two-dimensional scan grid
// ---------------------------------------------------------------------------

/// Worst reprojection disagreement of one translation relation under `poses`,
/// on a lattice over the left tile clipped to the right tile.
fn translation_pair_error(
    dimensions: (u32, u32),
    poses: &[Matrix3<f64>],
    left: usize,
    right: usize,
    left_to_right: &Matrix3<f64>,
) -> f64 {
    let (width, height) = (f64::from(dimensions.0), f64::from(dimensions.1));
    let mut worst = 0.0f64;
    for row in 0..=10 {
        for column in 0..=10 {
            let source = Point2::new(
                (width - 1.0) * f64::from(column) / 10.0,
                (height - 1.0) * f64::from(row) / 10.0,
            );
            let target = transformed_point(left_to_right, source).unwrap();
            if !(0.0..width).contains(&target.x) || !(0.0..height).contains(&target.y) {
                continue;
            }
            let error = (transformed_point(&poses[left], source).unwrap()
                - transformed_point(&poses[right], target).unwrap())
            .norm();
            worst = worst.max(error);
        }
    }
    worst
}

#[test]
fn checkpoint_ten_scan_grid_closure_corrects_tree_drift_within_the_residual_bounds() {
    use super::station_relation_test_access::{self, ClosureRelationView};

    // A 3×3 serpentine scan; each spanning-tree step along the scan path adds
    // (0.6, −0.4)px of drift, so the tree alone misplaces loop partners by up
    // to five steps (3.6px) while every relation measures the true geometry.
    let dimensions = (512u32, 384u32);
    let serpentine = [
        (0u32, 0u32),
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 1),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ];
    let world = |station: usize| {
        let (row, column) = serpentine[station];
        (f64::from(column) * 360.0, f64::from(row) * 270.0)
    };
    let drift = nalgebra::Vector2::new(0.6, -0.4);
    let base = (0..serpentine.len())
        .map(|station| {
            let (x, y) = world(station);
            translation(x + drift.x * station as f64, y + drift.y * station as f64)
        })
        .collect::<Vec<_>>();
    let station_of = |row: u32, column: u32| {
        serpentine
            .iter()
            .position(|&cell| cell == (row, column))
            .unwrap()
    };
    let mut relations = Vec::new();
    for row in 0..3u32 {
        for column in 0..3u32 {
            for (next_row, next_column) in [(row, column + 1), (row + 1, column)] {
                if next_row > 2 || next_column > 2 {
                    continue;
                }
                let (a, b) = (station_of(row, column), station_of(next_row, next_column));
                let (left, right) = (a.min(b), a.max(b));
                let ((left_x, left_y), (right_x, right_y)) = (world(left), world(right));
                let noise = |salt: u32| hashed_noise(left as u32, right as u32, salt, 0.25);
                let tree_edge = right == left + 1;
                // Tree edges carry exactly the drifted placement the tree was
                // built from; loop edges carry the true geometry.
                let (dx, dy) = if tree_edge {
                    (-drift.x, -drift.y)
                } else {
                    (noise(1), noise(2))
                };
                relations.push(ClosureRelationView {
                    score: 1.0,
                    left,
                    right,
                    left_to_right: translation(left_x - right_x + dx, left_y - right_y + dy),
                    independent_support: 3,
                    median_error_px: 0.5,
                });
            }
        }
    }
    assert_eq!(relations.len(), 12, "6 horizontal + 6 vertical relations");
    let tree_worst = relations
        .iter()
        .map(|relation| {
            translation_pair_error(
                dimensions,
                &base,
                relation.left,
                relation.right,
                &relation.left_to_right,
            )
        })
        .fold(0.0f64, f64::max);
    assert!(
        tree_worst > 3.0,
        "the drifted tree alone violates the pair bound: {tree_worst}"
    );

    let rows = serpentine.iter().map(|&(row, _)| row).collect::<Vec<_>>();
    let columns = serpentine
        .iter()
        .map(|&(_, column)| column)
        .collect::<Vec<_>>();
    let run =
        station_relation_test_access::closure_run(dimensions, &rows, &columns, &base, &relations);
    assert_eq!(
        run.report.status,
        report::ClosureStatus::Converged,
        "{:?}",
        run.report
    );
    assert_eq!(run.report.participating_constraints, relations.len());
    assert!(run.report.residual_median_px <= 2.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px <= 3.0, "{:?}", run.report);
    assert_eq!(run.report.offending_pair, None);
    assert_eq!(run.report.clamped_stations, 0);
    let closed_worst = relations
        .iter()
        .map(|relation| {
            translation_pair_error(
                dimensions,
                &run.poses,
                relation.left,
                relation.right,
                &relation.left_to_right,
            )
        })
        .fold(0.0f64, f64::max);
    assert!(closed_worst <= 3.0, "{closed_worst}");
    assert!(closed_worst < tree_worst);
}

// ---------------------------------------------------------------------------
// requirement 6.4 over the doubly covered overlap
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_covered_overlap_area_counts_only_doubly_covered_pixels() {
    let (width, height) = (100u32, 80u32);
    let source = coverage_from_fn(width, height, |x, y| !(10..30).contains(&x) || y >= 40);
    let target = coverage_from_fn(width, height, |x, y| x < 90 && !(50..60).contains(&y));
    let count = |shift: u32| {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                x + shift < width && source.is_covered(x, y) && target.is_covered(x + shift, y)
            })
            .count() as f64
    };
    let area = |relation: &Matrix3<f64>| {
        virtual_tile_covered_overlap_area(
            &source,
            (width, height),
            &target,
            (width, height),
            1,
            relation,
        )
        .unwrap()
    };
    assert_eq!(area(&Matrix3::identity()), count(0));
    assert_eq!(area(&translation(20.0, 0.0)), count(20));

    // A factor-2 mask pixel is a 2×2 block of relation pixels.
    let coarse_source = full_coverage(50, 40);
    let coarse_target = coverage_from_fn(50, 40, |x, _| x < 25);
    let coarse = virtual_tile_covered_overlap_area(
        &coarse_source,
        (100, 80),
        &coarse_target,
        (101, 81),
        2,
        &Matrix3::identity(),
    )
    .unwrap();
    assert_eq!(coarse, 25.0 * 40.0 * 4.0);

    // A mask that does not tile its plane is no measurement.
    assert!(
        virtual_tile_covered_overlap_area(
            &source,
            (width + 2, height),
            &target,
            (width, height),
            1,
            &Matrix3::identity(),
        )
        .is_none()
    );
}

#[test]
fn covered_overlap_support_ignores_the_uncovered_part_of_the_rectangle_overlap() {
    let (width, height) = (200u32, 100u32);
    // Identity relation, the right half of the source uncovered: the covered
    // overlap is half of the rectangle overlap.
    let source = coverage_from_fn(width, height, |x, _| x < 100);
    let target = full_coverage(width, height);
    let points = [
        (20.0, 20.0),
        (80.0, 20.0),
        (80.0, 70.0),
        (20.0, 70.0),
        (50.0, 45.0),
    ]
    .map(|(x, y)| (Point2::new(x, y), Point2::new(x, y)));
    let hull = 60.0 * 50.0;
    let rectangles = station_relation_spatial_support(
        &points,
        &Matrix3::identity(),
        (width, height),
        (width, height),
    );
    let covered = station_relation_spatial_support_with(
        &points,
        &Matrix3::identity(),
        (width, height),
        (width, height),
        StationOverlapMeasure::Covered {
            source: &source,
            target: &target,
            factor: 1,
        },
    );
    assert!((rectangles - hull / 20_000.0).abs() < 1e-9, "{rectangles}");
    assert!((covered - hull / 10_000.0).abs() < 1e-9, "{covered}");
    // Fully covered masks reproduce the rectangle measure.
    let full = station_relation_spatial_support_with(
        &points,
        &Matrix3::identity(),
        (width, height),
        (width, height),
        StationOverlapMeasure::Covered {
            source: &target,
            target: &target,
            factor: 1,
        },
    );
    assert!((full - rectangles).abs() < 1e-9);
    // No doubly covered pixel: no support at all.
    let empty = coverage_from_fn(width, height, |_, _| false);
    assert_eq!(
        station_relation_spatial_support_with(
            &points,
            &Matrix3::identity(),
            (width, height),
            (width, height),
            StationOverlapMeasure::Covered {
                source: &empty,
                target: &target,
                factor: 1,
            },
        ),
        0.0
    );
}

#[test]
fn virtual_tile_solver_measures_requirement_6_4_on_the_doubly_covered_overlap() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    // Tile 1 shows tile 0 shifted by (40, 6) everywhere, but tile 0 is only
    // covered for x < 220: half of the rectangle overlap holds no tile content.
    let (width, height) = (400u32, 300u32);
    let left = textured_plane(width, height);
    let right = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([textured_value(f64::from(x) + 40.0, f64::from(y) + 6.0).round() as u8])
    });
    let tiles = vec![plane_info(0, left), plane_info(1, right)];
    let left_to_right = translation(-40.0, -6.0);
    // 36 inliers spanning a 100×150 hull: 14% of the 360×294 rectangle
    // overlap, 28% of the 180×294 doubly covered overlap.
    let points = (0..6)
        .flat_map(|row| (0..6).map(move |column| (row, column)))
        .map(|(row, column)| {
            let source = Point2::new(
                80.0 + 20.0 * f64::from(column),
                60.0 + 30.0 * f64::from(row),
            );
            (source, transformed_point(&left_to_right, source).unwrap())
        })
        .collect::<Vec<_>>();
    let matches = HashMap::from([((0, 1), measured_relation(left_to_right, points))]);
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, left_to_right.try_inverse().unwrap()),
    ]);
    let station_topology = candidate_topology(2);
    let solve = |coverages: &[stitching::CoverageMask]| {
        let mut closure = report::ClosureReport::default();
        let mut relations = report::StationRelationsReport::default();
        let solved = solve_virtual_tile_station_poses_with_report(
            &tiles,
            VirtualTileSolverCoverages {
                quality: coverages,
                overlap: coverages,
                overlap_factor: 1,
            },
            &matches,
            &initial,
            &station_topology,
            &mut closure,
            &mut relations,
        );
        (solved, relations)
    };

    let covered = [
        coverage_from_fn(width, height, |x, _| x < 220),
        full_coverage(width, height),
    ];
    let (solved, relations) = solve(&covered);
    assert_eq!(solved.len(), 2, "{relations:?}");
    let support = relations.accepted[0].spatial_support;
    assert!(
        (support - 15_000.0 / (180.0 * 294.0)).abs() < 0.01,
        "{support}"
    );

    // Counting the uncovered half as overlap drops the same evidence to 14%.
    let pretended = [full_coverage(width, height), full_coverage(width, height)];
    let (solved, relations) = solve(&pretended);
    assert!(solved.is_empty(), "{relations:?}");
    assert!(
        relations
            .rejected_by_reason
            .contains_key("station_relation_low_spatial_support"),
        "{relations:?}"
    );
}
