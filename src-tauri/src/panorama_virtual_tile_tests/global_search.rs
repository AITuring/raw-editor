//! Prior-free global search: similarity proposals against an independent brute-force oracle,
//! closed scale and rotation boundaries, partial coverage, source-pair seeds in station and
//! render coordinates, repeated texture and the dense Station_Relation gates.

use super::*;

/// Deliberately independent of the production scorer: exhaustive integer
/// translation and a direct Pearson correlation over native covered pixels.
fn brute_force_translation_oracle(
    left: &GrayImage,
    right: &GrayImage,
    right_coverage: &stitching::CoverageMask,
) -> (i32, i32) {
    let (width, height) = left.dimensions();
    let mut best = (f64::NEG_INFINITY, 0, 0);
    for dy in -(height as i32 - 1)..height as i32 {
        for dx in -(width as i32 - 1)..width as i32 {
            let mut pairs = Vec::new();
            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    let (tx, ty) = (x + dx, y + dy);
                    if tx >= 0
                        && ty >= 0
                        && tx < width as i32
                        && ty < height as i32
                        && right_coverage.is_covered(tx as u32, ty as u32)
                    {
                        pairs.push((
                            f64::from(left.get_pixel(x as u32, y as u32)[0]),
                            f64::from(right.get_pixel(tx as u32, ty as u32)[0]),
                        ));
                    }
                }
            }
            if pairs.len() < VIRTUAL_TILE_GLOBAL_MIN_OVERLAP_SAMPLES {
                continue;
            }
            let count = pairs.len() as f64;
            let left_mean = pairs.iter().map(|pair| pair.0).sum::<f64>() / count;
            let right_mean = pairs.iter().map(|pair| pair.1).sum::<f64>() / count;
            let numerator = pairs
                .iter()
                .map(|pair| (pair.0 - left_mean) * (pair.1 - right_mean))
                .sum::<f64>();
            let left_energy = pairs
                .iter()
                .map(|pair| (pair.0 - left_mean).powi(2))
                .sum::<f64>();
            let right_energy = pairs
                .iter()
                .map(|pair| (pair.1 - right_mean).powi(2))
                .sum::<f64>();
            let score = numerator / (left_energy * right_energy).sqrt();
            if score > best.0 {
                best = (score, dx, dy);
            }
        }
    }
    (best.1, best.2)
}

fn global_plane(
    gray: &GrayImage,
    coverage: &stitching::CoverageMask,
) -> VirtualTileGlobalSearchPlane {
    VirtualTileGlobalSearchPlane::from_native(gray, coverage).expect("global plane")
}

fn proposal_reaches(
    proposals: &[VirtualTileGlobalProposal],
    expected: &Matrix3<f64>,
    tolerance: f64,
) -> bool {
    let centre = Point2::new(24.0, 20.0);
    let expected = transformed_point(expected, centre).unwrap();
    proposals.iter().any(|proposal| {
        transformed_point(&proposal.left_to_right, centre)
            .is_some_and(|actual| (actual - expected).norm() <= tolerance)
    })
}

#[test]
fn global_similarity_recalls_full_translation_and_agrees_with_independent_oracle() {
    let (width, height) = (56, 44);
    let left = textured_plane(width, height);
    let expected = translation(40.0, 2.0);
    let (right, right_coverage) = transformed_fixture(&left, &expected);
    let oracle = brute_force_translation_oracle(&left, &right, &right_coverage);
    assert_eq!(oracle, (40, 2));
    let proposals = virtual_tile_global_similarity_proposals(
        &global_plane(&left, &full_coverage(width, height)),
        &global_plane(&right, &right_coverage),
    );
    assert!(
        proposal_reaches(&proposals, &expected, 1.5),
        "{proposals:?}"
    );
}

#[test]
fn global_similarity_includes_closed_scale_and_rotation_boundaries() {
    let (width, height) = (56, 44);
    let left = textured_plane(width, height);
    for (scale, angle, dx, dy) in [
        (0.95f64, -10.0f64, 18.0f64, 11.0f64),
        (1.05, 10.0, 14.0, -3.0),
    ] {
        let expected = translation(dx, dy) * rotation(angle.to_radians()) * uniform_scale(scale);
        let (right, right_coverage) = transformed_fixture(&left, &expected);
        let proposals = virtual_tile_global_similarity_proposals(
            &global_plane(&left, &full_coverage(width, height)),
            &global_plane(&right, &right_coverage),
        );
        assert!(
            proposal_reaches(&proposals, &expected, 2.0),
            "{scale} {angle}: {proposals:?}"
        );
        assert!(
            proposals
                .iter()
                .all(|proposal| (0.95..=1.05).contains(&proposal.scale))
        );
        assert!(
            proposals
                .iter()
                .all(|proposal| (-10.0..=10.0).contains(&proposal.rotation_degrees))
        );
    }
}

#[test]
fn global_plane_uses_partial_coverage_without_reading_uncovered_payload() {
    let (width, height) = (320, 240);
    let coverage = coverage_from_fn(width, height, |x, y| x % 17 == 0 || (80..90).contains(&y));
    let first = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([if coverage.is_covered(x, y) {
            ((x * 7 + y * 3) % 251) as u8
        } else {
            0
        }])
    });
    let second = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([if coverage.is_covered(x, y) {
            ((x * 7 + y * 3) % 251) as u8
        } else {
            255
        }])
    });
    let first = global_plane(&first, &coverage);
    let second = global_plane(&second, &coverage);
    assert_eq!(first.factor, 2);
    assert_eq!(first.luminance, second.luminance);
    assert_eq!(first.gradient, second.gradient);
    assert_eq!(first.covered_count, second.covered_count);
    assert!(
        first
            .covered_count
            .iter()
            .any(|&count| count > 0 && count < first.factor * first.factor)
    );
}

#[test]
fn source_pairwise_seed_is_composed_in_station_tile_coordinates() {
    let left = plane_info(7, textured_plane(100, 80));
    let right = plane_info(11, textured_plane(100, 80));
    let source_to_target = translation(23.0, -11.0) * uniform_scale(1.01);
    let points = hypothesis_points(&source_to_target, (100, 80));
    let matches = HashMap::from([(
        (7usize, 11usize),
        MatchInfo {
            native_verified_points: Vec::new(),
            homography: source_to_target,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: false,
            points,
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            canonical_homography: Some(source_to_target),
        },
    )]);
    let source_to_world = HashMap::from([
        (7usize, translation(120.0, 40.0)),
        (11usize, translation(310.0, 95.0)),
    ]);
    let left_tile_to_world = translation(-35.0, 12.0);
    let right_tile_to_world = translation(80.0, -26.0);
    let proposals = virtual_tile_source_match_proposals(
        &[&left],
        &[&right],
        &matches,
        &source_to_world,
        &left_tile_to_world,
        &right_tile_to_world,
        1.0,
    );
    assert_eq!(proposals.len(), 1);
    let expected = right_tile_to_world.try_inverse().unwrap()
        * source_to_world[&11]
        * source_to_target
        * source_to_world[&7].try_inverse().unwrap()
        * left_tile_to_world;
    assert_projectively_equal(
        &proposals[0].left_to_right,
        &expected,
        "source pair seed should use tile-local coordinates",
    );
    assert_eq!(proposals[0].inliers, 64);
}

#[test]
fn source_pairwise_seed_scales_full_resolution_match_into_render_coordinates() {
    let left = plane_info(7, textured_plane(100, 80));
    let right = plane_info(11, textured_plane(100, 80));
    let source_to_target = translation(23.0, -11.0) * uniform_scale(1.01);
    let points = hypothesis_points(&source_to_target, (100, 80));
    let matches = HashMap::from([(
        (7usize, 11usize),
        MatchInfo {
            native_verified_points: Vec::new(),
            homography: source_to_target,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: false,
            points,
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            canonical_homography: Some(source_to_target),
        },
    )]);
    let source_to_world = HashMap::from([
        (7usize, translation(120.0, 40.0)),
        (11usize, translation(310.0, 95.0)),
    ]);
    let left_tile_to_world = translation(-35.0, 12.0);
    let right_tile_to_world = translation(80.0, -26.0);
    let render_scale = 0.5;
    let proposals = virtual_tile_source_match_proposals(
        &[&left],
        &[&right],
        &matches,
        &source_to_world,
        &left_tile_to_world,
        &right_tile_to_world,
        render_scale,
    );
    assert_eq!(proposals.len(), 1);
    let source_scale = Matrix3::new(
        render_scale,
        0.0,
        0.0,
        0.0,
        render_scale,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let expected_source_to_target =
        source_scale * source_to_target * source_scale.try_inverse().unwrap();
    let expected = right_tile_to_world.try_inverse().unwrap()
        * source_to_world[&11]
        * expected_source_to_target
        * source_to_world[&7].try_inverse().unwrap()
        * left_tile_to_world;
    assert_projectively_equal(
        &proposals[0].left_to_right,
        &expected,
        "source pair seed should scale full-resolution matches into render coordinates",
    );
}

#[test]
fn source_pairwise_seed_requires_consensus_for_coarse_bridge() {
    let left = plane_info(7, textured_plane(100, 80));
    let right = plane_info(11, textured_plane(100, 80));
    let source_to_target = translation(23.0, -11.0) * uniform_scale(1.01);
    let points = hypothesis_points(&source_to_target, (100, 80));
    let mut match_info = MatchInfo {
        native_verified_points: Vec::new(),
        homography: source_to_target,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: true,
        points: points.clone(),
        candidate_points: Vec::new(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: Some(source_to_target),
    };
    let mut matches = HashMap::from([((7usize, 11usize), match_info.clone())]);
    let source_to_world = HashMap::from([
        (7usize, translation(120.0, 40.0)),
        (11usize, translation(310.0, 95.0)),
    ]);
    let left_tile_to_world = translation(-35.0, 12.0);
    let right_tile_to_world = translation(80.0, -26.0);
    assert!(
        virtual_tile_source_match_proposals(
            &[&left],
            &[&right],
            &matches,
            &source_to_world,
            &left_tile_to_world,
            &right_tile_to_world,
            1.0,
        )
        .is_empty(),
        "ordinary coarse bridges must not seed station geometry"
    );
    match_info.top_candidate_points = points.into_iter().take(2).collect();
    matches.insert((7, 11), match_info);
    assert_eq!(
        virtual_tile_source_match_proposals(
            &[&left],
            &[&right],
            &matches,
            &source_to_world,
            &left_tile_to_world,
            &right_tile_to_world,
            1.0,
        )
        .len(),
        1,
        "consensus coarse bridges may seed a basin, but still need the final selector"
    );
}

#[test]
fn global_similarity_keeps_true_peak_for_repeated_texture_and_is_deterministic() {
    let (width, height) = (64, 48);
    let left = GrayImage::from_fn(width, height, |x, y| {
        let periodic = if (x / 4 + y / 4) % 2 == 0 { 55 } else { 205 };
        image::Luma([if (9..15).contains(&x) && (7..13).contains(&y) {
            ((x * 19 + y * 23) % 251) as u8
        } else {
            periodic
        }])
    });
    let expected = translation(29.0, 3.0);
    let (right, right_coverage) = transformed_fixture(&left, &expected);
    let left_global = global_plane(&left, &full_coverage(width, height));
    let right_global = global_plane(&right, &right_coverage);
    let covered = (0..left_global.height)
        .flat_map(|y| (0..left_global.width).map(move |x| (x, y)))
        .filter(|&(x, y)| left_global.covered_count[(y * left_global.width + x) as usize] > 0)
        .collect::<Vec<_>>();
    let stride = covered
        .len()
        .div_ceil(VIRTUAL_TILE_GLOBAL_MAX_SAMPLES)
        .max(1);
    let samples = covered
        .into_iter()
        .step_by(stride)
        .take(VIRTUAL_TILE_GLOBAL_MAX_SAMPLES)
        .collect::<Vec<_>>();
    let expected_score =
        virtual_tile_masked_similarity_score(&left_global, &right_global, &expected, &samples);
    assert!(
        expected_score.is_some_and(|(score, _)| score > 0.95),
        "expected score: {expected_score:?}"
    );
    let run = || virtual_tile_global_similarity_proposals(&left_global, &right_global);
    let first = run();
    let second = run();
    assert!(proposal_reaches(&first, &expected, 1.5), "{first:?}");
    let threaded = [1usize, 2, 8, 16]
        .into_iter()
        .map(|threads| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(run)
        })
        .collect::<Vec<_>>();
    let proposal_bits = |proposals: &[VirtualTileGlobalProposal]| {
        proposals
            .iter()
            .map(|proposal| {
                (
                    proposal.score.to_bits(),
                    proposal.overlap_samples,
                    virtual_tile_hypothesis_matrix_key(&proposal.left_to_right),
                )
            })
            .collect::<Vec<_>>()
    };
    for proposals in &threaded {
        assert_eq!(proposal_bits(proposals), proposal_bits(&first));
    }
    let inverse = expected.try_inverse().unwrap();
    let reverse = virtual_tile_global_similarity_proposals(&right_global, &left_global);
    assert!(proposal_reaches(&reverse, &inverse, 1.5), "{reverse:?}");
    assert_eq!(first.len(), second.len());
    for (left, right) in first.iter().zip(second.iter()) {
        assert_eq!(left.score.to_bits(), right.score.to_bits());
        assert_eq!(
            virtual_tile_hypothesis_matrix_key(&left.left_to_right),
            virtual_tile_hypothesis_matrix_key(&right.left_to_right)
        );
    }
}

#[test]
fn unrelated_global_proposals_do_not_bypass_dense_station_relation_gates() {
    let (width, height) = (64, 48);
    let left = textured_plane(width, height);
    let right = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([((x * 71 + y * 29 + x * y * 3) % 251) as u8])
    });
    let coverages = vec![full_coverage(width, height), full_coverage(width, height)];
    let tiles = vec![plane_info(0, left.clone()), plane_info(1, right.clone())];
    let mut pyramid = VirtualTileMatchPyramid::new(vec![1], 2);
    pyramid
        .set_station(0, &tiles[0], &left, &coverages[0])
        .unwrap();
    pyramid
        .set_station(1, &tiles[1], &right, &coverages[1])
        .unwrap();
    let proposals = virtual_tile_global_similarity_proposals(
        pyramid.global_planes[0].as_ref().unwrap(),
        pyramid.global_planes[1].as_ref().unwrap(),
    );
    assert!(!proposals.is_empty());
    let hypotheses = proposals
        .iter()
        .flat_map(|proposal| {
            virtual_tile_pyramid_station_match(
                0,
                1,
                &pyramid,
                &proposal.left_to_right,
                VIRTUAL_TILE_SEED_GLOBAL_COVERAGE,
            )
            .0
        })
        .collect::<Vec<_>>();
    assert!(
        !select_virtual_tile_relation_hypothesis(
            &tiles,
            0,
            1,
            &hypotheses,
            pyramid.solver_coverages()
        )
        .is_some_and(|selection| selection.passes)
    );
}
