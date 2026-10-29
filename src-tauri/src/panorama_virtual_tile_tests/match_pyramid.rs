//! Match planes and the pyramid: pyramid factors, block-mean match planes without uncovered
//! payload, level and native maps, per-station levels, and the relation-quality plane with its
//! own coverage.

use super::*;

fn coverage_bits(mask: &stitching::CoverageMask) -> Vec<bool> {
    let (width, height) = mask.dimensions();
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| mask.is_covered(x, y))
        .collect()
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

    let (empty, empty_factor) = virtual_tile_relation_quality_plane(&[], &gray, &coverage);
    assert_eq!((empty.dimensions(), empty_factor), ((1, 1), 1));

    for factor in [1u32, 4] {
        let planes = planes_for(&[factor], &gray, &coverage);
        let (plane, plane_factor) = virtual_tile_relation_quality_plane(&planes, &gray, &coverage);
        assert_eq!(plane_factor, 4);
        let structural = VirtualTileMatchPlane::from_native(&gray, &coverage, 4).unwrap();
        assert_eq!(
            plane,
            imageproc::filter::gaussian_blur_f32(&structural.gray, 1.0),
            "a single matching level still verifies covered structure"
        );
    }

    for (factors, expected) in [
        (vec![16u32, 8, 4], 16u32),
        (vec![8, 4], 8),
        (vec![2, 1], 2),
        (vec![4, 2], 4),
        (vec![32, 16, 8, 4], 16),
    ] {
        let planes = planes_for(&factors, &gray, &coverage);
        let (plane, plane_factor) = virtual_tile_relation_quality_plane(&planes, &gray, &coverage);
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
    let feature_plane =
        VirtualTileMatchPlane::from_native(&gray, &coverage, pyramid.feature_factors[1]).unwrap();
    assert!(tile.features.iter().all(|feature| {
        feature.keypoint.x < feature_plane.gray.width()
            && feature.keypoint.y < feature_plane.gray.height()
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
    let structural = VirtualTileMatchPlane::from_native(&gray, &coverage, 3).unwrap();
    assert_eq!(native_tile.scale_factor, 3.0);
    assert_eq!(
        native_tile.alignment_image,
        imageproc::filter::gaussian_blur_f32(&structural.gray, 1.0)
    );
    assert_eq!(
        single.quality_coverages[0].dimensions(),
        native_tile.alignment_image.dimensions()
    );
    assert_eq!(
        coverage_bits(&single.quality_coverages[0]),
        coverage_bits(&structural.coverage)
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
