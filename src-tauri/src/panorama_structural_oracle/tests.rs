//! Independent synthetic tests of the structural oracle: auditable primitives, uncovered
//! payload, geometric hashing on repetition, target provenance, left/right inversion,
//! unrelated structure and determinism at 1, 2, 8 and 16 threads.

use super::*;

// ---------------------------------------------------------------------------
// Independent synthetic oracle tests
// ---------------------------------------------------------------------------

fn synthetic_plane(width: u32, height: u32, shift: (i32, i32), repeated: bool) -> GrayImage {
    let mut image = GrayImage::from_pixel(width, height, image::Luma([245]));
    let mut draw = |x: i32, y: i32| {
        let x = x + shift.0;
        let y = y + shift.1;
        if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
            image.put_pixel(x as u32, y as u32, image::Luma([12]));
        }
    };
    for motif in 0..if repeated { 4 } else { 1 } {
        let ox = 30 + motif * 48;
        for t in 0..45 {
            draw(ox + t, 35);
            draw(ox + 10, 25 + t);
        }
        for t in 0..35 {
            draw(ox + t, 75 + t / 2);
        }
    }
    for t in 0..100 {
        draw(45 + t, 145);
    }
    for t in 0..70 {
        draw(150, 105 + t);
    }
    image
}

fn synthetic_pyramid(
    left: GrayImage,
    right: GrayImage,
    left_coverage: stitching::CoverageMask,
    right_coverage: stitching::CoverageMask,
) -> (Vec<ImageInfo>, VirtualTileMatchPyramid) {
    let mut tiles = vec![
        plane_info_for_oracle(0, left.dimensions()),
        plane_info_for_oracle(1, right.dimensions()),
    ];
    let mut pyramid = VirtualTileMatchPyramid::new(vec![4], 2);
    install_virtual_tile_match_planes(&mut tiles[0], &mut pyramid, 0, &left, &left_coverage, &[])
        .unwrap();
    install_virtual_tile_match_planes(&mut tiles[1], &mut pyramid, 1, &right, &right_coverage, &[])
        .unwrap();
    (tiles, pyramid)
}

fn plane_info_for_oracle(id: usize, dimensions: (u32, u32)) -> ImageInfo {
    ImageInfo {
        source_footprint: None,
        id,
        filename: String::new(),
        width: dimensions.0,
        height: dimensions.1,
        alignment_image: GrayImage::new(1, 1),
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

#[test]
fn structural_oracle_localizes_auditable_primitives() {
    let image = synthetic_plane(256, 208, (0, 0), false);
    let coverage = stitching::CoverageMask::from_bytes(256, 208, vec![255; 256 * 208]).unwrap();
    let (_, pyramid) = synthetic_pyramid(image.clone(), image, coverage.clone(), coverage);
    let primitives = extract_structural_primitives(&pyramid, 0);
    assert!(primitives.len() >= 8, "primitives={}", primitives.len());
    assert!(primitives.iter().any(|primitive| {
        primitive.record.kind == StructuralPrimitiveKind::LineEndpoint
            && (primitive.point - Point2::new(30.0, 35.0)).norm() <= 8.0
    }));
    assert!(
        primitives
            .iter()
            .all(|primitive| primitive.record.chain_length >= MIN_CHAIN_POINTS as u32)
    );
}

#[test]
fn structural_oracle_ignores_uncovered_payload_bits() {
    let width = 256;
    let height = 208;
    let coverage = stitching::CoverageMask::from_bytes(
        width,
        height,
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    if x > 20 && x < 230 && y > 20 && y < 190 {
                        255
                    } else {
                        0
                    }
                })
            })
            .collect(),
    )
    .unwrap();
    let first = synthetic_plane(width, height, (0, 0), false);
    let mut second = first.clone();
    for y in 0..height {
        for x in 0..width {
            if !coverage.is_covered(x, y) {
                second.put_pixel(x, y, image::Luma([((x * 17 + y * 31) & 255) as u8]));
            }
        }
    }
    let (first_tiles, first_pyramid) =
        synthetic_pyramid(first.clone(), first, coverage.clone(), coverage.clone());
    let (second_tiles, second_pyramid) =
        synthetic_pyramid(second.clone(), second, coverage.clone(), coverage);
    let first_bits =
        serde_json::to_vec(&solve_structural_direction(0, 1, &first_tiles, &first_pyramid).0)
            .unwrap();
    let second_bits =
        serde_json::to_vec(&solve_structural_direction(0, 1, &second_tiles, &second_pyramid).0)
            .unwrap();
    assert_eq!(first_bits, second_bits);
}

#[test]
fn structural_oracle_geometric_hash_disambiguates_repetition_with_real_points() {
    let left = synthetic_plane(320, 240, (0, 0), true);
    let right = synthetic_plane(320, 240, (12, 8), true);
    let coverage = stitching::CoverageMask::from_bytes(320, 240, vec![255; 320 * 240]).unwrap();
    let (_, pyramid) = synthetic_pyramid(left, right, coverage.clone(), coverage);
    let (direction, fits) = solve_structural_direction(0, 1, &[], &pyramid);
    let (transform, inliers) = fits.first().expect("a repeated fixture has a hashed model");
    assert!(inliers.len() >= 4);
    assert!(
        focus_transform_disagreement_px(
            transform,
            &Matrix3::new(1.0, 0.0, 12.0, 0.0, 1.0, 8.0, 0.0, 0.0, 1.0),
            (320, 240),
        )
        .unwrap()
            <= 2.0
    );
    let ids = direction.models[0]
        .correspondences
        .iter()
        .map(|point| (point.source_primitive_id, point.target_primitive_id))
        .collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), direction.models[0].correspondences.len());
}

#[test]
fn structural_oracle_correspondences_keep_independent_target_provenance() {
    let left = synthetic_plane(320, 240, (0, 0), false);
    let right = synthetic_plane(320, 240, (13, 7), false);
    let coverage = stitching::CoverageMask::from_bytes(320, 240, vec![255; 320 * 240]).unwrap();
    let (_, pyramid) = synthetic_pyramid(left, right, coverage.clone(), coverage);
    let (direction, _) = solve_structural_direction(0, 1, &[], &pyramid);
    let target = direction
        .target_primitives
        .iter()
        .map(|primitive| {
            (
                primitive.id,
                primitive.x_native_bits,
                primitive.y_native_bits,
            )
        })
        .collect::<BTreeSet<_>>();
    for model in direction.models {
        for point in model.correspondences {
            assert!(target.contains(&(
                point.target_primitive_id,
                point.target_x_bits,
                point.target_y_bits
            )));
        }
    }
}

#[test]
fn structural_oracle_left_right_solve_inverts() {
    let left = synthetic_plane(320, 240, (0, 0), false);
    let right = synthetic_plane(320, 240, (12, 8), false);
    let coverage = stitching::CoverageMask::from_bytes(320, 240, vec![255; 320 * 240]).unwrap();
    let (_, pyramid) = synthetic_pyramid(left, right, coverage.clone(), coverage);
    let (_, forward) = solve_structural_direction(0, 1, &[], &pyramid);
    let (_, reverse) = solve_structural_direction(1, 0, &[], &pyramid);
    let reverse_inverse = reverse[0].0.try_inverse().unwrap();
    assert!(
        focus_transform_disagreement_px(&forward[0].0, &reverse_inverse, (320, 240)).unwrap()
            <= 1.0
    );
}

#[test]
fn structural_oracle_rejects_unrelated_structure() {
    let left = synthetic_plane(320, 240, (0, 0), false);
    let mut right = GrayImage::from_pixel(320, 240, image::Luma([245]));
    for x in 20..300 {
        right.put_pixel(x, 20 + x / 3, image::Luma([10]));
    }
    let coverage = stitching::CoverageMask::from_bytes(320, 240, vec![255; 320 * 240]).unwrap();
    let (_, pyramid) = synthetic_pyramid(left, right, coverage.clone(), coverage);
    let (direction, _) = solve_structural_direction(0, 1, &[], &pyramid);
    assert!(
        direction
            .models
            .iter()
            .all(|model| model.correspondences.len() < STATION_RELATION_MIN_INLIERS)
    );
}

#[test]
fn structural_oracle_is_deterministic_at_1_2_8_16_threads() {
    let left = synthetic_plane(320, 240, (0, 0), true);
    let right = synthetic_plane(320, 240, (12, 8), true);
    let coverage = stitching::CoverageMask::from_bytes(320, 240, vec![255; 320 * 240]).unwrap();
    let (_, pyramid) = synthetic_pyramid(left, right, coverage.clone(), coverage);
    let expected = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            serde_json::to_vec(&solve_structural_direction(0, 1, &[], &pyramid).0).unwrap()
        });
    for threads in [2, 8, 16] {
        let actual = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                serde_json::to_vec(&solve_structural_direction(0, 1, &[], &pyramid).0).unwrap()
            });
        assert_eq!(actual, expected, "thread count {threads}");
    }
}
