//! Pair matching contracts: candidate pairs for large and medium stacks, natural filename
//! order, canonical match direction and relation views, the homography scale ratio and the
//! station pose Jacobian.

use super::*;

#[test]
fn focus_adjacent_same_focal_pair_allows_missing_exif() {
    let mut left = focus_test_image(0, "DSC_3743.png");
    left.focal_length_35mm = Some(50.0);
    let mut right = focus_test_image(1, "DSC_3744.png");
    right.focal_length_35mm = None;
    assert!(focus_adjacent_same_focal_pair(&left, &right));
    let mut mismatched = focus_test_image(2, "DSC_3745.png");
    mismatched.focal_length_35mm = Some(85.0);
    assert!(!focus_compatible_focal_length(&left, &mismatched));
}

#[test]
fn canonical_match_direction_is_independent_of_import_order() {
    let reversed = vec![test_image(0, "z.jpg"), test_image(1, "a.jpg")];
    let (source, target, invert_for_storage) = canonical_match_direction(&reversed, 0, 1);

    assert_eq!(reversed[source].filename, "a.jpg");
    assert_eq!(reversed[target].filename, "z.jpg");
    assert!(invert_for_storage);

    let forward = vec![test_image(0, "a.jpg"), test_image(1, "z.jpg")];
    let (source, target, invert_for_storage) = canonical_match_direction(&forward, 0, 1);

    assert_eq!((source, target), (0, 1));
    assert_eq!(forward[source].filename, "a.jpg");
    assert_eq!(forward[target].filename, "z.jpg");
    assert!(!invert_for_storage);
}

#[test]
fn canonical_relation_view_distinguishes_bounds_and_singular_matrices() {
    let images = vec![test_image(0, "z.jpg"), test_image(1, "a.jpg")];
    let relation = translation_match(12.0, -4.0, 24);
    assert_eq!(
        canonical_relation_view(&images, 0, 2, &relation),
        Err(CanonicalRelationViewError::IndexOutOfBounds {
            left: 0,
            right: 2,
            image_count: 2,
        })
    );

    let singular = MatchInfo {
        homography: Matrix3::zeros(),
        ..relation.clone()
    };
    assert_eq!(
        canonical_relation_view(&images, 0, 1, &singular),
        Err(CanonicalRelationViewError::NonInvertible)
    );

    let canonical = canonical_relation_view(&images, 0, 1, &relation)
        .expect("invertible relation has a canonical view");
    assert_eq!((canonical.source_index, canonical.target_index), (1, 0));
    assert!(canonical.inverted);
    assert_eq!(
        canonical.homography,
        relation.homography.try_inverse().unwrap()
    );
}

#[test]
fn large_stack_matching_scales_with_neighbors_instead_of_all_pairs() {
    let small_pairs = pairs_to_match(SCALABLE_STACK_THRESHOLD);
    assert_eq!(
        small_pairs.len(),
        SCALABLE_STACK_THRESHOLD * (SCALABLE_STACK_THRESHOLD - 1) / 2
    );

    let large_pairs = pairs_to_match(200);
    assert!(large_pairs.len() <= 200 * LARGE_STACK_NEIGHBOR_WINDOW);
    assert!(large_pairs.len() * 10 < 200 * 199 / 2);
    for index in 0..199 {
        assert!(large_pairs.contains(&(index, index + 1)));
    }
}

#[test]
fn large_stack_candidates_include_natural_filename_neighbors() {
    let images = (0..31)
        .map(|index| {
            let number = (index * 13) % 31;
            test_image(index, &format!("tile-{number}.jpg"))
        })
        .collect::<Vec<_>>();
    let pairs = pairs_to_match_for_images(&images, BlendMode::Panorama);
    let index_for_number = |number: usize| {
        images
            .iter()
            .position(|image| image.filename == format!("tile-{number}.jpg"))
            .expect("generated filename should exist")
    };

    for number in 0..30 {
        let left = index_for_number(number);
        let right = index_for_number(number + 1);
        assert!(pairs.contains(&(left.min(right), left.max(right))));
    }
    assert!(pairs.len() <= images.len() * LARGE_STACK_NEIGHBOR_WINDOW * 2);
}

#[test]
fn medium_focus_stack_candidates_are_independent_of_upload_order() {
    let images = (0..45)
        .map(|index| test_image(index, &format!("upload-random-{index}.jpg")))
        .collect::<Vec<_>>();
    let pairs = pairs_to_match_for_images(&images, BlendMode::FocusStack);

    assert_eq!(
        pairs.len(),
        45 * 44 / 2,
        "a medium focus stack must inspect every possible overlap before ordering"
    );
    assert!(pairs.contains(&(0, 44)));
    assert!(pairs.contains(&(7, 31)));
}

#[test]
fn medium_panorama_candidates_cover_a_possible_lens_switch() {
    let images = (0..69)
        .map(|index| test_image(index, &format!("frame-{index}.jpg")))
        .collect::<Vec<_>>();
    let pairs = pairs_to_match_for_images(&images, BlendMode::Panorama);

    assert_eq!(pairs.len(), 69 * 68 / 2);
    assert!(pairs.contains(&(0, 68)));
}

#[test]
fn natural_path_order_compares_numeric_filename_runs() {
    let mut paths = ["tile-10.jpg", "tile-2.jpg", "tile-001.jpg", "tile-1.jpg"];
    paths.sort_by(|left, right| natural_path_cmp(left, right));

    assert_eq!(
        paths,
        ["tile-1.jpg", "tile-001.jpg", "tile-2.jpg", "tile-10.jpg"]
    );
}

#[test]
fn homography_scale_ratio_is_the_singular_value_geometric_mean() {
    assert!((homography_scale_ratio(&Matrix3::identity()).unwrap() - 1.0).abs() < 1e-12);

    // A uniform 3% enlargement reports 1.03 regardless of rotation.
    let (sin, cos) = 0.4f64.sin_cos();
    let rotated_scale = Matrix3::new(
        1.03 * cos,
        -1.03 * sin,
        120.0,
        1.03 * sin,
        1.03 * cos,
        -40.0,
        0.0,
        0.0,
        1.0,
    );
    assert!((homography_scale_ratio(&rotated_scale).unwrap() - 1.03).abs() < 1e-9);

    // An anisotropic stretch reports the geometric mean of both axes.
    let anisotropic = Matrix3::new(1.20, 0.0, 0.0, 0.0, 0.80, 0.0, 0.0, 0.0, 1.0);
    let expected = (1.20f64 * 0.80).sqrt();
    assert!((homography_scale_ratio(&anisotropic).unwrap() - expected).abs() < 1e-12);

    // A homography is defined up to an overall scale, so an unnormalised
    // matrix must report the same ratio as its normalised form.
    let unnormalised = Matrix3::identity() * 4.0;
    assert!((homography_scale_ratio(&unnormalised).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn station_pose_jacobian_accepts_bounded_projective_local_scale_without_mutation() {
    // Every one of the eight homography parameters is free and non-trivial;
    // the Local_Scale gate observes the matrix but never projects it onto a
    // similarity/affine model.
    let pose = Matrix3::new(
        1.012, -0.018, 73.0, 0.021, 0.997, -41.0, 0.000_015, -0.000_009, 1.0,
    );
    let original = pose;
    let (minimum, maximum) = station_pose_local_scale_range(&pose, (1200, 800))
        .expect("a nonsingular projective pose has a measurable Jacobian");
    let ratio = maximum / minimum;

    assert!(minimum > 0.0);
    assert!(ratio <= STATION_POSE_MAX_LOCAL_SCALE_RATIO);
    assert_eq!(pose, original, "the gate must preserve all eight pose DOF");
    assert_ne!(pose[(2, 0)], 0.0);
    assert_ne!(pose[(2, 1)], 0.0);
}

#[test]
fn station_pose_jacobian_rejects_excessive_local_scale_variation() {
    let pose = Matrix3::new(1.0, 0.01, 0.0, -0.02, 1.0, 0.0, 0.000_35, 0.0, 1.0);
    let (minimum, maximum) = station_pose_local_scale_range(&pose, (1200, 800))
        .expect("the perspective pose remains nonsingular over the tile");
    let ratio = maximum / minimum;

    assert!(ratio > STATION_POSE_MAX_LOCAL_SCALE_RATIO);
}
