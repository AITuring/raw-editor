//! Virtual_Tile station matching: descriptor-guided and direct grids, polished and
//! coarse-pyramid matches, measured reverse transforms, structure planes and relation quality
//! on repeated-texture, projective and offset-origin fixtures.

use super::*;

fn guided_repeated_texture_fixture() -> (
    Vec<ImageInfo>,
    Vec<stitching::CoverageMask>,
    HashMap<usize, Matrix3<f64>>,
    topology::StationTopology,
) {
    let (width, height) = (720u32, 560u32);
    let world_texture = |world_x: u32, y: u32| {
        let coarse = ((world_x / 11 + y / 13) % 2) as u8 * 150;
        let fine = ((world_x / 5 + y / 7) % 2) as u8 * 70;
        20u8.saturating_add(coarse).saturating_add(fine)
    };
    let left_gray = GrayImage::from_fn(width, height, |x, y| image::Luma([world_texture(x, y)]));
    let right_gray = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([world_texture(x + 200, y)])
    });
    let points = (70..=490)
        .step_by(70)
        .flat_map(|y| (240..=660).step_by(70).map(move |left_x| (left_x, y)))
        .collect::<Vec<_>>();
    let descriptor_for = |index: usize| {
        let mut descriptor = [0u8; BRIEF_DESCRIPTOR_SIZE / 8];
        descriptor[index / 8] = 1u8 << (index % 8);
        descriptor
    };
    let left_features = points
        .iter()
        .enumerate()
        .map(|(index, &(left_x, y))| Feature {
            keypoint: KeyPoint { x: left_x, y },
            descriptor: descriptor_for(index),
            support_scale: 1.0,
        })
        .collect::<Vec<_>>();
    // Deliberately permute otherwise perfect descriptor identities. A
    // descriptor matcher therefore gets broad but geometrically
    // inconsistent repeated-texture pairs; direct pixels remain aligned.
    let right_features = points
        .iter()
        .enumerate()
        .map(|(index, &(left_x, y))| Feature {
            keypoint: KeyPoint { x: left_x - 200, y },
            descriptor: descriptor_for((index * 17 + 5) % points.len()),
            support_scale: 1.0,
        })
        .collect::<Vec<_>>();
    let mut tiles = synthetic_virtual_tiles(2);
    for tile in &mut tiles {
        tile.width = width;
        tile.height = height;
        tile.scale_factor = 1.0;
    }
    tiles[0].alignment_image = left_gray;
    tiles[0].features = left_features;
    tiles[1].alignment_image = right_gray;
    tiles[1].features = right_features;
    let coverage_bytes = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                u8::from((20..width - 20).contains(&x) && (20..height - 20).contains(&y)) * u8::MAX
            })
        })
        .collect::<Vec<_>>();
    let coverage = stitching::CoverageMask::from_bytes(width, height, coverage_bytes).unwrap();
    let right_to_world = Matrix3::new(1.0, 0.0, 200.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, right_to_world),
    ]);
    (
        tiles,
        vec![coverage.clone(), coverage],
        initial,
        synthetic_virtual_tile_topology(2),
    )
}

fn direct_projective_fixture() -> (
    Vec<ImageInfo>,
    Vec<stitching::CoverageMask>,
    HashMap<usize, Matrix3<f64>>,
    topology::StationTopology,
    Matrix3<f64>,
) {
    let (width, height) = (420u32, 340u32);
    let texture = |x: f64, y: f64| {
        let value = 126.0
            + 47.0 * (x * 0.091 + y * 0.037).sin()
            + 38.0 * (x * 0.027 - y * 0.113).cos()
            + 24.0 * (x * 0.173 + y * 0.149).sin()
            + 13.0 * (x * 0.311 - y * 0.071).cos();
        value.clamp(0.0, 255.0)
    };
    let actual = Matrix3::new(
        1.006, 0.009, -82.0, -0.006, 0.996, 11.0, 0.000_018, -0.000_012, 1.0,
    );
    let inverse = actual.try_inverse().unwrap();
    let left = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([texture(f64::from(x), f64::from(y)).round() as u8])
    });
    let right = GrayImage::from_fn(width, height, |x, y| {
        let point = transformed_point(&inverse, Point2::new(f64::from(x), f64::from(y)));
        let value = point.map_or(0.0, |point| texture(point.x, point.y) * 0.82 + 19.0);
        image::Luma([value.clamp(0.0, 255.0).round() as u8])
    });
    let mut tiles = synthetic_virtual_tiles(2);
    for tile in &mut tiles {
        tile.width = width;
        tile.height = height;
        tile.scale_factor = 2.0;
        tile.features.clear();
    }
    tiles[0].alignment_image = left;
    tiles[1].alignment_image = right;
    let coverage = stitching::CoverageMask::from_bytes(
        width,
        height,
        vec![u8::MAX; width as usize * height as usize],
    )
    .unwrap();
    let mut initial_left_to_right = actual;
    initial_left_to_right[(0, 2)] += 4.5;
    initial_left_to_right[(1, 2)] -= 3.75;
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, initial_left_to_right.try_inverse().unwrap()),
    ]);
    (
        tiles,
        vec![coverage.clone(), coverage],
        initial,
        synthetic_virtual_tile_topology(2),
        actual,
    )
}

fn nonzero_origin_virtual_tile_fixture() -> (
    Vec<ImageInfo>,
    Vec<stitching::CoverageMask>,
    HashMap<usize, Matrix3<f64>>,
    topology::StationTopology,
    Matrix3<f64>,
) {
    let left_dimensions = (560u32, 420u32);
    let right_dimensions = (500u32, 460u32);
    let left_to_world = Matrix3::new(1.0, 0.0, 140.0, 0.0, 1.0, -80.0, 0.0, 0.0, 1.0);
    let right_to_world = Matrix3::new(1.0, 0.0, 260.0, 0.0, 1.0, -55.0, 0.0, 0.0, 1.0);
    let actual = right_to_world.try_inverse().unwrap() * left_to_world;
    let world_texture = |x: f64, y: f64| {
        (127.0
            + 48.0 * (x * 0.071 + y * 0.043).sin()
            + 37.0 * (x * 0.029 - y * 0.109).cos()
            + 22.0 * (x * 0.163 + y * 0.137).sin())
        .clamp(0.0, 255.0)
    };
    let render = |dimensions: (u32, u32), tile_to_world: &Matrix3<f64>| {
        GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
            let world =
                transformed_point(tile_to_world, Point2::new(f64::from(x), f64::from(y))).unwrap();
            image::Luma([world_texture(world.x, world.y).round() as u8])
        })
    };
    let mut tiles = synthetic_virtual_tiles(2);
    tiles[0].width = left_dimensions.0;
    tiles[0].height = left_dimensions.1;
    tiles[0].alignment_image = render(left_dimensions, &left_to_world);
    tiles[1].width = right_dimensions.0;
    tiles[1].height = right_dimensions.1;
    tiles[1].alignment_image = render(right_dimensions, &right_to_world);
    let coverages = [left_dimensions, right_dimensions]
        .into_iter()
        .map(|(width, height)| {
            stitching::CoverageMask::from_bytes(
                width,
                height,
                vec![u8::MAX; width as usize * height as usize],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let initial = HashMap::from([(tiles[0].id, left_to_world), (tiles[1].id, right_to_world)]);
    (
        tiles,
        coverages,
        initial,
        synthetic_virtual_tile_topology(2),
        actual,
    )
}

#[test]
fn direct_grid_nonzero_origins_and_different_dimensions_keep_station_mapping() {
    let (tiles, coverages, initial, topology, actual) = nonzero_origin_virtual_tile_fixture();
    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    let relation = outcome
        .relations
        .get(&(0, 1))
        .expect("nonzero Virtual_Tile origins must remain in the same station frame");
    assert!(relation.inliers >= STATION_RELATION_MIN_INLIERS);
    for point in [
        Point2::new(180.0, 90.0),
        Point2::new(310.0, 210.0),
        Point2::new(455.0, 350.0),
    ] {
        let expected = transformed_point(&actual, point).unwrap();
        let measured = transformed_point(&relation.homography, point).unwrap();
        assert!((expected - measured).norm() <= 0.75);
        let left_world = transformed_point(&initial[&tiles[0].id], point).unwrap();
        let right_world = transformed_point(&initial[&tiles[1].id], measured).unwrap();
        assert!((left_world - right_world).norm() <= 0.75);
    }

    let (mut reordered_tiles, mut reordered_coverages, reordered_initial, reordered_topology, _) =
        nonzero_origin_virtual_tile_fixture();
    reordered_tiles.swap(0, 1);
    reordered_coverages.swap(0, 1);
    let reordered = virtual_tile_station_matches(
        &reordered_tiles,
        &reordered_coverages,
        &reordered_initial,
        &reordered_topology,
    );
    let reordered_inverse = reordered
        .get(&(0, 1))
        .expect("station import reordering must retain measured geometry")
        .homography
        .try_inverse()
        .unwrap();
    for (forward, reverse) in relation.homography.iter().zip(reordered_inverse.iter()) {
        assert!((forward - reverse).abs() < 1e-8);
    }

    let mut wrong_mapping = initial.clone();
    wrong_mapping.insert(tiles[0].id, initial[&tiles[1].id]);
    wrong_mapping.insert(tiles[1].id, initial[&tiles[0].id]);
    assert!(
        virtual_tile_station_matches(&tiles, &coverages, &wrong_mapping, &topology).is_empty(),
        "swapping station origins must not be accepted as measured evidence"
    );
}

#[test]
fn polished_matcher_accepts_a_forward_shift_beyond_reverse_probe_radius() {
    // A five-pixel forward residual is within the production polish pass
    // (±8px), but outside the old reverse probe (±2px) when that probe was
    // centred on the forward match.  The reverse check must use the source
    // centre so it can verify the same correspondence without widening
    // its independent radius.
    let (width, height) = (420u32, 320u32);
    let texture = |x: f64, y: f64| {
        (127.0
            + 55.0 * (x * 0.071 + y * 0.043).sin()
            + 41.0 * (x * 0.029 - y * 0.109).cos()
            + 23.0 * (x * 0.163 + y * 0.137).sin())
        .clamp(0.0, 255.0)
        .round() as u8
    };
    let left = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([texture(f64::from(x), f64::from(y))])
    });
    let shift = 5.0;
    let right = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([texture(f64::from(x) - shift, f64::from(y))])
    });
    let coverage = stitching::CoverageMask::from_bytes(
        width,
        height,
        vec![u8::MAX; width as usize * height as usize],
    )
    .expect("the synthetic polish planes are fully covered");
    let mut planes = synthetic_virtual_tiles(2);
    let mut left_plane = planes.remove(0);
    let mut right_plane = planes.remove(0);
    left_plane.width = width;
    left_plane.height = height;
    left_plane.alignment_image = left;
    right_plane.width = width;
    right_plane.height = height;
    right_plane.alignment_image = right;
    let (_, mut diagnostic) = virtual_tile_direct_station_match(
        0,
        1,
        &[],
        &[],
        &HashMap::new(),
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    let relation = virtual_tile_polished_station_match(
        &left_plane,
        &coverage,
        &right_plane,
        &coverage,
        1,
        &Matrix3::identity(),
        ((width, height), (width, height)),
        &mut diagnostic,
    )
    .into_iter()
    .last()
    .expect("the production polish matcher must verify the five-pixel shift");
    assert_eq!(diagnostic.failure_stage, "fitted");
    assert!(relation.inliers >= STATION_RELATION_MIN_INLIERS);
    let measured = transformed_point(&relation.homography, Point2::new(210.0, 160.0))
        .expect("fitted homography maps the source point");
    assert!((measured.x - 215.0).abs() < 1.0);
    assert!((measured.y - 160.0).abs() < 1.0);
}

#[test]
fn coarse_pyramid_retry_accepts_finer_sampling_failures_without_relaxing_gates() {
    for stage in [
        "insufficient_grid_probes",
        "insufficient_measurable_patches",
        "insufficient_bidirectional_matches",
        "insufficient_grid_coverage",
        "residual_model_no_fit",
        "insufficient_matches_after_model_quota",
        "insufficient_fitted_hull_support",
    ] {
        assert!(virtual_tile_coarse_failure_can_retry(stage));
    }
    for stage in [
        "no_predicted_overlap",
        "fitted_orientation_mismatch",
        "non_invertible_seed",
    ] {
        assert!(!virtual_tile_coarse_failure_can_retry(stage));
    }
}

#[test]
fn direct_grid_rejects_inverse_coordinate_direction() {
    let (tiles, coverages, mut initial, topology, actual) = nonzero_origin_virtual_tile_fixture();
    initial.insert(tiles[0].id, Matrix3::identity());
    initial.insert(tiles[1].id, actual);
    assert!(
        virtual_tile_station_matches(&tiles, &coverages, &initial, &topology).is_empty(),
        "using target-to-source as source-to-target must fail strict pixel evidence"
    );
}

#[test]
fn measured_reverse_transform_is_the_inverse_of_forward_residual_composition() {
    let initial = Matrix3::new(
        1.004, 0.009, -83.0, -0.006, 0.997, 12.0, 0.000_01, -0.000_02, 1.0,
    );
    let source = Point2::new(210.5, 143.25);
    let predicted = transformed_point(&initial, source).unwrap();
    let measured = predicted + nalgebra::Vector2::new(11.25, -7.5);
    let reverse = virtual_tile_measured_reverse_transform(&initial, source, measured).unwrap();
    assert!((transformed_point(&reverse, measured).unwrap() - source).norm() < 1e-9);
    assert!(
        (transformed_point(&initial.try_inverse().unwrap(), measured).unwrap() - source).norm()
            > VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX,
        "the inverse prior is not the inverse of a measured residual match"
    );
}

#[test]
fn direct_grid_recovers_projective_photometric_overlap() {
    let (tiles, coverages, initial, topology, actual) = direct_projective_fixture();
    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    let relation = outcome
        .relations
        .get(&(0, 1))
        .expect("direct fused pixels must recover the projective overlap");
    let diagnostic = &outcome.diagnostics[0];
    assert!(diagnostic.grid_probe_count <= VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES);
    assert!(diagnostic.measurable_patch_count >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.bidirectional_match_count >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.fitted_inliers >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.hull_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT);
    for point in [
        Point2::new(100.0, 70.0),
        Point2::new(210.0, 170.0),
        Point2::new(335.0, 275.0),
    ] {
        let expected = transformed_point(&actual, point).unwrap();
        let measured = transformed_point(&relation.homography, point).unwrap();
        assert!(
            (expected - measured).norm() <= 1.0,
            "{expected:?} != {measured:?}"
        );
    }
}

#[test]
fn virtual_tile_direct_relations_leave_the_canonical_direction_to_the_solver() {
    let (tiles, coverages, initial, _, _) = direct_projective_fixture();
    let (relation, _) = virtual_tile_direct_station_match(
        0,
        1,
        &tiles,
        &coverages,
        &initial,
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    let relation = relation.expect("the fixture overlap is measurable");
    // The fit is stored untouched in the (left, right) key direction.
    assert!(relation.canonical_homography.is_none());
}

#[test]
fn single_level_relation_quality_distinguishes_registered_and_shifted_structure() {
    let (mut tiles, coverages, _, _, actual) = direct_projective_fixture();
    let mut quality_coverages = Vec::new();
    for (tile, coverage) in tiles.iter_mut().zip(&coverages) {
        let plane = VirtualTileMatchPlane::from_native(&tile.alignment_image, coverage, 1).unwrap();
        let (quality, factor) =
            virtual_tile_relation_quality_plane(&[plane], &tile.alignment_image, coverage);
        assert_eq!(factor, 4);
        quality_coverages.push(
            VirtualTileMatchPlane::from_native(&tile.alignment_image, coverage, factor)
                .unwrap()
                .coverage,
        );
        tile.scale_factor *= f64::from(factor);
        tile.alignment_image = quality;
    }
    let scale = Matrix3::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0);
    let native = scale * actual * scale.try_inverse().unwrap();
    let good = focus_overlap_quality_measurements_on_coverage(
        &tiles[0],
        &tiles[1],
        &native,
        Some((&quality_coverages[0], &quality_coverages[1])),
    )
    .unwrap();
    let mut shifted = native;
    shifted[(0, 2)] += 80.0;
    let bad = focus_overlap_quality_measurements_on_coverage(
        &tiles[0],
        &tiles[1],
        &shifted,
        Some((&quality_coverages[0], &quality_coverages[1])),
    )
    .unwrap();
    assert!(
        good.median_edge_orientation_difference_degrees < 10.0,
        "{:?}",
        good.median_edge_orientation_difference_degrees
    );
    assert!(
        bad.median_edge_orientation_difference_degrees > 10.0,
        "{:?}",
        bad.median_edge_orientation_difference_degrees
    );
    assert!(good.intensity_ncc > bad.intensity_ncc + 0.2);
}

#[test]
fn direct_grid_structure_planes_only_carry_the_orientation_measurement() {
    let (tiles, coverages, initial, _, _) = direct_projective_fixture();
    let threshold = STATION_RELATION_MAX_MEDIAN_ERROR_PX;
    let (probe, probe_diagnostic) =
        virtual_tile_direct_station_match(0, 1, &tiles, &coverages, &initial, threshold);
    let probe = probe.expect("the probe planes measure the projective overlap");

    // σ = 1 structural planes (the coarse-octave rule) keep the probe and
    // the fitted relation; only the orientation measurement moves.
    let structure = tiles
        .iter()
        .map(|tile| {
            imageproc::filter::gaussian_blur_f32(
                &tile.alignment_image,
                VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
            )
        })
        .collect::<Vec<_>>();
    let (structural, structural_diagnostic) = virtual_tile_direct_station_match_with_structure(
        0,
        1,
        &tiles,
        &coverages,
        &initial,
        threshold,
        Some((&structure[0], &structure[1])),
    );
    let structural = structural.expect("structural planes keep the verified relation");
    assert_eq!(
        structural_diagnostic.bidirectional_match_count,
        probe_diagnostic.bidirectional_match_count
    );
    assert!(
        structural_diagnostic
            .median_fitted_orientation_difference_degrees
            .is_some_and(|degrees| degrees <= VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES)
    );
    for point in [Point2::new(100.0, 70.0), Point2::new(335.0, 275.0)] {
        let expected = transformed_point(&probe.homography, point).unwrap();
        let measured = transformed_point(&structural.homography, point).unwrap();
        assert!(
            (expected - measured).norm() <= 1.0,
            "{expected:?} != {measured:?}"
        );
    }

    // The gate reads the structure planes: edges that cross the probe
    // planes' edges reject the same probe matches.
    let (width, height) = tiles[1].alignment_image.dimensions();
    let rows = GrayImage::from_fn(width, height, |_, y| {
        image::Luma([(127.0 + 100.0 * (f64::from(y) * 0.2).sin()).round() as u8])
    });
    let columns = GrayImage::from_fn(width, height, |x, _| {
        image::Luma([(127.0 + 100.0 * (f64::from(x) * 0.2).sin()).round() as u8])
    });
    let (crossed, crossed_diagnostic) = virtual_tile_direct_station_match_with_structure(
        0,
        1,
        &tiles,
        &coverages,
        &initial,
        threshold,
        Some((&rows, &columns)),
    );
    assert!(crossed.is_none());
    assert_eq!(
        crossed_diagnostic.failure_stage,
        "fitted_orientation_mismatch"
    );
    assert_eq!(
        crossed_diagnostic.bidirectional_match_count,
        probe_diagnostic.bidirectional_match_count
    );
    assert!(
        crossed_diagnostic
            .median_fitted_orientation_difference_degrees
            .is_some_and(|degrees| degrees > VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES)
    );
}

#[test]
fn direct_grid_round_trip_mutation_is_rejected() {
    let source = Point2::new(113.25, 84.5);
    assert!(virtual_tile_round_trip_is_consistent(
        source,
        source + nalgebra::Vector2::new(0.6, -0.6)
    ));
    assert!(!virtual_tile_round_trip_is_consistent(
        source,
        source + nalgebra::Vector2::new(1.01, 0.0)
    ));
}

#[test]
fn direct_grid_recovers_repeated_texture_despite_inconsistent_descriptors() {
    let (tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    assert!(
        virtual_tile_unseeded_station_match(0, 1, &tiles).is_none(),
        "global ratio matching must remain ambiguous on repeated descriptors"
    );
    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    let relation = outcome
        .relations
        .get(&(0, 1))
        .expect("guided overlap must fit");
    let diagnostic = &outcome.diagnostics[0];
    assert_eq!(
        diagnostic.evidence_kind,
        stack_report::StationRelationEvidenceKind::VirtualTile
    );
    assert_eq!(diagnostic.failure_stage, "fitted");
    assert!(diagnostic.grid_probe_count >= 24);
    assert!(diagnostic.measurable_patch_count >= 24);
    assert!(diagnostic.bidirectional_match_count >= 24);
    assert!(diagnostic.occupied_grid_cells >= VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS);
    assert!(diagnostic.search_radius_px > 0.0);
    assert_eq!(diagnostic.residual_model, "translation");
    assert_eq!(
        diagnostic.max_residual_translation_consensus,
        diagnostic.residual_translation.candidate_inliers
    );
    assert!(
        diagnostic.max_residual_translation_consensus >= diagnostic.fitted_inliers,
        "the post-model spatial quota may only reduce a measured consensus"
    );
    assert!(diagnostic.residual_dx_px.count >= diagnostic.fitted_inliers);
    assert_eq!(
        diagnostic.residual_dx_px.count,
        diagnostic.residual_dy_px.count
    );
    assert_eq!(
        diagnostic.residual_dx_px.count,
        diagnostic.residual_magnitude_px.count
    );
    assert!(diagnostic.fitted_inliers >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.hull_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT);
    assert!(relation.inliers >= STATION_RELATION_MIN_INLIERS);
    assert!(relation.points.iter().all(|(left, right)| {
        (right.x - (left.x - 200.0)).abs() < 1e-9 && (right.y - left.y).abs() < 1e-9
    }));
    assert!(
        station_relation_spatial_support(
            &relation.points,
            &relation.homography,
            tiles[0].dimensions(),
            tiles[1].dimensions(),
        ) >= STATION_RELATION_MIN_SPATIAL_SUPPORT
    );
}

#[test]
fn direct_grid_ignores_descriptor_matches_concentrated_in_repeated_stroke_cells() {
    let (mut tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    let descriptor = [0u8; BRIEF_DESCRIPTOR_SIZE / 8];
    tiles[0].features = (0..30)
        .map(|index| Feature {
            keypoint: KeyPoint {
                x: 240 + index * 14,
                y: 270 + index % 3,
            },
            descriptor,
            support_scale: 1.0,
        })
        .collect();
    tiles[1].features = tiles[0]
        .features
        .iter()
        .map(|feature| Feature {
            keypoint: KeyPoint {
                x: feature.keypoint.x - 200,
                y: feature.keypoint.y,
            },
            descriptor,
            support_scale: 1.0,
        })
        .collect();

    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    let relation = outcome
        .relations
        .get(&(0, 1))
        .expect("direct grid pixels must not depend on descriptor concentration");
    let diagnostic = &outcome.diagnostics[0];
    assert!(diagnostic.grid_probe_count > tiles[0].features.len());
    assert!(diagnostic.measurable_patch_count >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.bidirectional_match_count >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.occupied_grid_cells >= VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS);
    assert!(relation.inliers >= STATION_RELATION_MIN_INLIERS);
}

#[test]
fn guided_overlap_rejects_wrong_prior_and_wrong_content() {
    let (tiles, coverages, mut initial, topology) = guided_repeated_texture_fixture();
    initial.insert(
        tiles[1].id,
        Matrix3::new(1.0, 0.0, 260.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    );
    assert!(
        virtual_tile_station_matches(&tiles, &coverages, &initial, &topology).is_empty(),
        "a prior outside the bounded structural search must not become evidence"
    );

    let (mut tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    tiles[1].alignment_image =
        GrayImage::from_pixel(tiles[1].width, tiles[1].height, image::Luma([127]));
    assert!(
        virtual_tile_station_matches(&tiles, &coverages, &initial, &topology).is_empty(),
        "descriptor identity without native patch agreement must reject"
    );
}

#[test]
fn direct_grid_rejects_locally_wrong_45_degree_orientation() {
    let (mut tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    let (width, height) = tiles[0].dimensions();
    tiles[0].alignment_image = GrayImage::from_fn(width, height, |x, _| {
        image::Luma([(40.0 + f64::from(x % 100) * 2.0).round() as u8])
    });
    tiles[1].alignment_image = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([(40.0 + f64::from((x + y) % 100) * std::f64::consts::SQRT_2).round() as u8])
    });
    let left_to_right = Matrix3::new(1.0, 0.0, -200.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let structure = virtual_tile_structural_score(
        &tiles[0].alignment_image,
        &tiles[1].alignment_image,
        &coverages[0],
        &coverages[1],
        tiles[0].features[1].keypoint,
        tiles[1].features[1].keypoint,
        &left_to_right,
    )
    .expect("the two ramp patches are measurable");
    assert!(structure.patch_ncc >= VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC);
    assert!(structure.gradient_ncc >= VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC);
    assert!(structure.orientation_difference_degrees > 35.0);

    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    assert!(outcome.relations.is_empty());
    let diagnostic = &outcome.diagnostics[0];
    assert_ne!(diagnostic.failure_stage, "fitted");
    // A one-dimensional ramp can be rejected earlier as an ambiguous patch
    // peak. If enough unique peaks survive to produce a model, the retained
    // fitted-Jacobian orientation gate must be the rejecting stage.
    if diagnostic.fitted_inliers >= STATION_RELATION_MIN_INLIERS {
        assert_eq!(diagnostic.failure_stage, "fitted_orientation_mismatch");
        assert!(
            diagnostic
                .median_fitted_orientation_difference_degrees
                .is_some_and(|difference| difference > 35.0)
        );
    }
}

#[test]
fn direct_grid_flat_overlap_reports_unmeasurable_patches() {
    let (mut tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    for tile in &mut tiles {
        tile.alignment_image = GrayImage::from_pixel(tile.width, tile.height, image::Luma([127]));
    }
    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    assert!(outcome.relations.is_empty());
    assert_eq!(outcome.diagnostics.len(), 1);
    let diagnostic = &outcome.diagnostics[0];
    assert_eq!(
        diagnostic.evidence_kind,
        stack_report::StationRelationEvidenceKind::VirtualTile
    );
    assert!(diagnostic.grid_probe_count >= STATION_RELATION_MIN_INLIERS);
    assert_eq!(diagnostic.measurable_patch_count, 0);
    assert_eq!(diagnostic.bidirectional_match_count, 0);
    assert_eq!(diagnostic.fitted_inliers, 0);
    assert_eq!(diagnostic.failure_stage, "insufficient_measurable_patches");
}

#[test]
fn guided_overlap_ignores_uncovered_payload_and_import_order() {
    let (tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    let original = virtual_tile_station_matches(&tiles, &coverages, &initial, &topology);
    let mut mutated = tiles;
    let coverage = &coverages[1];
    for (x, y, pixel) in mutated[1].alignment_image.enumerate_pixels_mut() {
        if !coverage.is_covered(x, y) {
            pixel[0] = ((x.wrapping_mul(31) + y.wrapping_mul(17)) % 251) as u8;
        }
    }
    let repeated = virtual_tile_station_matches(&mutated, &coverages, &initial, &topology);
    let original_relation = original.get(&(0, 1)).unwrap();
    let repeated_relation = repeated.get(&(0, 1)).unwrap();
    assert_eq!(original_relation.points, repeated_relation.points);
    assert_eq!(
        original_relation
            .homography
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        repeated_relation
            .homography
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let (mut swapped_tiles, mut swapped_coverages, swapped_initial, swapped_topology) =
        guided_repeated_texture_fixture();
    swapped_tiles.swap(0, 1);
    swapped_coverages.swap(0, 1);
    let swapped = virtual_tile_station_matches(
        &swapped_tiles,
        &swapped_coverages,
        &swapped_initial,
        &swapped_topology,
    );
    let reverse = swapped.get(&(0, 1)).unwrap();
    let reverse_inverse = reverse.homography.try_inverse().unwrap();
    for (left, right) in original_relation
        .homography
        .iter()
        .zip(reverse_inverse.iter())
    {
        assert!((left - right).abs() < 1e-8);
    }
}
