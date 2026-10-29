//! Capture_Station grouping: the four pieces of link evidence, sequence and motion boundaries,
//! member-count limits, isolated sources and layers, bracket pose locks and capture-group
//! render order.

use super::*;

#[test]
fn local_focus_bracket_lock_restores_the_direct_relative_pose() {
    let mut images = vec![
        focus_test_image(0, "DSC_1001.NEF"),
        focus_test_image(1, "DSC_1002.NEF"),
    ];
    for image in &mut images {
        image.focal_length_35mm = Some(90.0);
    }
    let matches = HashMap::from([((0, 1), observed_translation_match(10.0, 4.0, 32))]);
    let global = HashMap::from([
        (0, Matrix3::identity()),
        (
            1,
            Matrix3::new(1.0, 0.0, 90.0, 0.0, 1.0, 40.0, 0.0, 0.0, 1.0),
        ),
    ]);

    let locked = lock_focus_local_bracket_poses(&images, &matches, &global);
    let relative = locked[&1]
        .try_inverse()
        .expect("locked target pose should be invertible")
        * locked[&0];

    assert!((relative[(0, 2)] - 10.0).abs() < 1e-9);
    assert!((relative[(1, 2)] - 4.0).abs() < 1e-9);
}

#[test]
fn capture_stations_do_not_merge_across_a_coarse_sequence_boundary() {
    let images = (0..7)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let mut coarse_boundary = observed_translation_match(120.0, 0.0, 48);
    coarse_boundary.coarse_bridge = true;
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(3.0, 1.0, 600)),
        ((1, 2), observed_translation_match(2.0, 1.0, 540)),
        ((2, 3), coarse_boundary),
        ((3, 4), observed_translation_match(3.0, 0.0, 520)),
        ((4, 5), observed_translation_match(2.0, 1.0, 490)),
        // A weak direct boundary match also cannot define a bracket.
        ((5, 6), observed_translation_match(140.0, 0.0, 95)),
    ]);

    let stations = focus_capture_stations(&images, &matches);
    assert_eq!(stations, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]);
}

#[test]
fn capture_stations_keep_a_low_texture_direct_focus_layer() {
    let images = (0..3)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(3.0, 1.0, 540)),
        ((1, 2), observed_translation_match(4.0, 2.0, 64)),
    ]);

    assert_eq!(
        focus_capture_stations(&images, &matches),
        vec![vec![0, 1, 2]]
    );
}

#[test]
fn capture_groups_keep_long_focus_brackets() {
    let images = (0..12)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    // 需求 1.4 budgets 0.02 long sides of *accumulated* centre displacement
    // per Capture_Station, so a 12-frame bracket may drift at most
    // 20px/11 ≈ 1.8px per layer.  0.4px of refocus breathing per layer adds
    // up to 0.0049 and stays one station.
    let matches = (0..images.len() - 1)
        .map(|index| ((index, index + 1), observed_translation_match(0.4, 0.2, 80)))
        .collect::<HashMap<_, _>>();
    let locked = images
        .iter()
        .map(|image| (image.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();

    let (groups, group_for_image) = focus_capture_groups(&images, &matches, &locked).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].members, (0..12).collect::<Vec<_>>());
    assert_eq!(group_for_image, vec![0; 12]);
}

#[test]
fn capture_groups_use_geometry_when_filenames_have_unrelated_numbers() {
    // A file picker can rename or re-number captures.  The same-position
    // decision must therefore come from the verified overlap and solved
    // centre motion, rather than treating a numeric filename gap as a
    // camera-position boundary.
    let mut first = focus_test_image(0, "capture-1000.NEF");
    first.focal_length_35mm = Some(75.0);
    let mut second = focus_test_image(1, "capture-9000.NEF");
    second.focal_length_35mm = Some(75.0);
    let images = vec![first, second];
    let matches = HashMap::from([((0, 1), observed_translation_match(0.0, 0.0, 16))]);
    let locked = images
        .iter()
        .map(|image| (image.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();

    let (groups, group_for_image) =
        focus_capture_groups(&images, &matches, &locked).expect("poses are invertible");

    assert_eq!(groups.len(), 1);
    assert_eq!(
        groups[0].members.iter().copied().collect::<HashSet<_>>(),
        HashSet::from([0, 1])
    );
    assert_eq!(group_for_image, vec![0, 0]);
}

#[test]
fn capture_groups_split_accumulated_scan_motion() {
    let images = (0..12)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let matches = (0..images.len() - 1)
        .map(|index| {
            (
                (index, index + 1),
                observed_translation_match(20.0, 0.0, 80),
            )
        })
        .collect::<HashMap<_, _>>();

    let groups = focus_local_bracket_components(&images, &matches);

    assert!(groups.len() > 1);
    assert!(groups.iter().all(|group| group.len() <= 4));
    assert_eq!(groups.iter().map(Vec::len).sum::<usize>(), images.len());

    let locked = images
        .iter()
        .enumerate()
        .map(|(index, image)| {
            (
                image.id,
                Matrix3::new(1.0, 0.0, index as f64 * 20.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            )
        })
        .collect::<HashMap<_, _>>();
    let (groups, _) = focus_capture_groups(&images, &matches, &locked).unwrap();
    assert!(groups.len() > 1);
    assert!(groups.iter().all(|group| group.members.len() <= 4));
}

#[test]
fn capture_station_link_requires_all_four_pieces_of_evidence() {
    let images = (0..2)
        .map(|index| focus_textured_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();

    // Focus breathing: 36 inliers, high overlap correlation, broad support,
    // unit scale.
    let accepted = observed_translation_match(2.0, 1.0, 36);
    let evidence =
        focus_station_link_evidence(&images, 0, 1, &accepted).expect("the overlap is measurable");
    assert!(evidence.overlap_ncc_measured);
    assert!(evidence.inliers >= STATION_MIN_INLIER_MATCHES);
    assert!(evidence.overlap_ncc >= STATION_MIN_OVERLAP_NCC);
    assert!(evidence.spatial_support >= STATION_MIN_INLIER_SPATIAL_SUPPORT);
    assert!((STATION_SCALE_RATIO_MIN..=STATION_SCALE_RATIO_MAX).contains(&evidence.scale_ratio));
    assert!(focus_match_is_capture_station_link(
        &images, 0, 1, &accepted
    ));

    // Fewer than 30 verified correspondences is not station evidence, even
    // with a perfect transform.
    let mut too_few_inliers = accepted.clone();
    too_few_inliers
        .points
        .truncate(STATION_MIN_INLIER_MATCHES - 1);
    assert!(!focus_match_is_capture_station_link(
        &images,
        0,
        1,
        &too_few_inliers
    ));

    // A 5% scale step is a different camera distance, not a focus layer.
    let scaled = observed_transform_match(
        Matrix3::new(1.05, 0.0, 0.0, 0.0, 1.05, 0.0, 0.0, 0.0, 1.0),
        36,
    );
    let scaled_evidence = focus_station_link_evidence(&images, 0, 1, &scaled).expect("measurable");
    assert!(scaled_evidence.scale_ratio > STATION_SCALE_RATIO_MAX);
    assert!(!focus_match_is_capture_station_link(&images, 0, 1, &scaled));

    // Inliers packed into one corner carry no spatial support.
    let mut clustered = accepted.clone();
    clustered.points = (0..36)
        .map(|index| {
            let source = Point2::new(
                120.0 + f64::from(index % 6) * 2.0,
                120.0 + f64::from(index / 6) * 2.0,
            );
            (source, Point2::new(source.x + 2.0, source.y + 1.0))
        })
        .collect();
    let clustered_evidence =
        focus_station_link_evidence(&images, 0, 1, &clustered).expect("measurable");
    assert!(clustered_evidence.spatial_support < STATION_MIN_INLIER_SPATIAL_SUPPORT);
    assert!(!focus_match_is_capture_station_link(
        &images, 0, 1, &clustered
    ));

    // A displacement of a quarter frame decorrelates the overlap well below
    // the 0.60 bar even though the transform is a clean translation.
    let neighbouring_station = observed_translation_match(60.0, 0.0, 36);
    let neighbour_evidence =
        focus_station_link_evidence(&images, 0, 1, &neighbouring_station).expect("measurable");
    assert!(neighbour_evidence.overlap_ncc < STATION_MIN_OVERLAP_NCC);
    assert!(!focus_match_is_capture_station_link(
        &images,
        0,
        1,
        &neighbouring_station
    ));
}

#[test]
fn capture_sequence_weight_boost_never_reaches_station_membership() {
    // 需求 1.2: `FOCUS_BRACKET_MAX_CAPTURE_GAP` is a candidate-order hint for
    // the pose graph.  Two filename-adjacent captures whose overlap evidence
    // fails 需求 1.1 must stay in different Capture_Stations even while the
    // graph builder keeps boosting their edge weight.
    let images = vec![
        focus_textured_test_image(0, "DSC_1001.NEF"),
        focus_textured_test_image(1, "DSC_1002.NEF"),
    ];
    let capture_gap = trailing_capture_number(&images[0].filename)
        .zip(trailing_capture_number(&images[1].filename))
        .map(|(left, right)| left.abs_diff(right))
        .expect("both names carry a trailing capture number");
    assert!(capture_gap <= FOCUS_BRACKET_MAX_CAPTURE_GAP);

    let neighbouring_station = observed_translation_match(60.0, 0.0, 36);
    assert!(
        focus_graph_capture_sequence_boost(&images[0], &images[1], &neighbouring_station) > 1.0,
        "the capture-sequence hint must keep boosting the graph weight"
    );
    assert!(
        !focus_match_is_capture_station_link(&images, 0, 1, &neighbouring_station),
        "a capture-sequence hint must not create station membership"
    );
    assert!(
        focus_station_grouping(&images, &HashMap::from([((0, 1), neighbouring_station)]))
            .components
            .is_empty()
    );
}

#[test]
fn grouping_reports_an_isolated_source_without_moving_the_others() {
    // 需求 1.6: the isolated source is named, and the remaining sources keep
    // the grouping they would have had on their own.
    let images = (0..3)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let bracket = HashMap::from([((0, 1), observed_translation_match(2.0, 1.0, 36))]);
    let reference = focus_station_grouping(&images[..2], &bracket);

    let with_isolated = focus_station_grouping(&images, &bracket);
    assert_eq!(with_isolated.components, reference.components);
    assert_eq!(with_isolated.isolated, vec![2]);
}

#[test]
fn task_7_31_station_member_count_boundaries_are_exact() {
    let outcomes = [1usize, 2, 47, 48, 49].map(station_member_count_is_supported);
    assert_eq!(outcomes, [true, true, true, true, false]);
    assert_eq!(FOCUS_BRACKET_MAX_COMPONENT_SOURCES, 48);
    assert!(!station_member_count_is_supported(0));
}

#[test]
fn oversized_station_is_split_until_every_member_count_fits() {
    // 需求 1.5 / 1.10: a static 50-frame bracket has no motion to split on,
    // so the member ceiling has to do it, and every split is recorded.
    let count = FOCUS_BRACKET_MAX_COMPONENT_SOURCES + 2;
    let images = (0..count)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let matches = (0..count - 1)
        .map(|index| ((index, index + 1), observed_translation_match(0.0, 0.0, 36)))
        .collect::<HashMap<_, _>>();

    let grouping = focus_station_grouping(&images, &matches);

    assert_eq!(
        grouping.components.iter().map(Vec::len).sum::<usize>(),
        count
    );
    assert!(
        grouping
            .components
            .iter()
            .all(|station| station.len() <= FOCUS_BRACKET_MAX_COMPONENT_SOURCES)
    );
    assert!(
        grouping
            .splits
            .iter()
            .any(|split| split.reason == degradation::GROUPING_MEMBER_LIMIT_SPLIT)
    );
}

#[test]
fn accumulated_motion_split_records_its_position_and_ratio() {
    // 需求 1.4: 12 frames drifting 2.236px each cross 0.02 long sides of
    // accumulated displacement at the ninth step (9 × 0.002236 = 0.0201).
    let images = (0..12)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let matches = (0..images.len() - 1)
        .map(|index| ((index, index + 1), observed_translation_match(2.0, 1.0, 36)))
        .collect::<HashMap<_, _>>();

    let grouping = focus_station_grouping(&images, &matches);

    let motion_splits = grouping
        .splits
        .iter()
        .filter(|split| split.reason == degradation::GROUPING_ACCUMULATED_MOTION)
        .collect::<Vec<_>>();
    assert_eq!(motion_splits.len(), 1);
    assert_eq!(motion_splits[0].position, 9);
    assert!(motion_splits[0].motion_ratio > STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO);
    // The candidate order runs along the recovered world centres, whose
    // direction is a property of the measured relations rather than of the
    // import order, so assert the partition rather than which end it starts
    // from.
    let mut sizes = grouping.components.iter().map(Vec::len).collect::<Vec<_>>();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![3, 9]);
    let mut covered = grouping.components.concat();
    covered.sort_unstable();
    assert_eq!(covered, (0..12).collect::<Vec<_>>());
}

#[test]
fn focus_render_order_finishes_each_capture_group_before_the_next() {
    let images = (0..6)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let groups = HashMap::from([(0, 1), (1, 1), (2, 1), (3, 2), (4, 2), (5, 2)]);
    let mut render_order = vec![
        &images[4], &images[1], &images[3], &images[0], &images[5], &images[2],
    ];

    order_focus_render_sources_by_capture_group(&mut render_order, &groups);

    assert_eq!(
        render_order
            .iter()
            .map(|image| image.id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5]
    );
}

#[test]
fn capture_groups_keep_an_isolated_layer_out_of_every_station() {
    // 需求 1.6: a source that satisfies the 需求 1.1 overlap evidence with no
    // other source is marked isolated and merged into no Capture_Station —
    // a close solved world centre is not a substitute for the missing
    // evidence — while the remaining sources keep the grouping they would
    // have had anyway.
    let images = (0..5)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(3.0, 1.0, 540)),
        ((3, 4), observed_translation_match(2.0, 1.0, 520)),
    ]);
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let locked = HashMap::from([
        (0, translation(0.0)),
        (1, translation(3.0)),
        // No accepted relation at all for image 2, even though its solved
        // centre sits at the same physical station as images 0 and 1.
        (2, translation(7.0)),
        (3, translation(900.0)),
        (4, translation(904.0)),
    ]);

    let (groups, group_for_image) = focus_capture_groups(&images, &matches, &locked).unwrap();
    assert_eq!(
        groups
            .iter()
            .map(|group| group.members.clone())
            .collect::<Vec<_>>(),
        vec![vec![0, 1], vec![3, 4]]
    );
    assert_eq!(group_for_image[2], usize::MAX);
    assert!(groups.iter().all(|group| !group.members.contains(&2)));
    assert_eq!(
        focus_station_grouping(&images, &matches).isolated,
        vec![2],
        "需求 1.6: the isolated source is still named for the Stack_Report"
    );
}

#[test]
fn capture_groups_keep_a_coarse_boundary_despite_close_solved_centers() {
    let images = (0..4)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let mut boundary = observed_translation_match(100.0, 0.0, 48);
    boundary.coarse_bridge = true;
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(3.0, 1.0, 540)),
        ((1, 2), boundary),
        ((2, 3), observed_translation_match(2.0, 1.0, 520)),
    ]);
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let locked = HashMap::from([
        (0, translation(0.0)),
        (1, translation(3.0)),
        (2, translation(100.0)),
        (3, translation(104.0)),
    ]);

    let (groups, _) = focus_capture_groups(&images, &matches, &locked).unwrap();
    assert_eq!(
        groups
            .iter()
            .map(|group| group.members.clone())
            .collect::<Vec<_>>(),
        vec![vec![0, 1], vec![2, 3]]
    );
}
