//! Render and stitching order: the flat-artwork overlap walk, compositor order over the station
//! topology, anchor-frame fusion and the panorama stitching order.

use super::*;

#[test]
fn flat_artwork_render_order_starts_at_center_and_walks_overlaps() {
    let mut images = (0..4)
        .map(|id| test_image(id, &format!("/synthetic/DSC_{id:04}.ARW")))
        .collect::<Vec<_>>();
    for image in &mut images {
        image.width = 100;
        image.height = 100;
    }
    let translations = [(0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (90.0, 90.0)];
    let homographies = translations
        .into_iter()
        .enumerate()
        .map(|(id, (x, y))| (id, Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0)))
        .collect::<HashMap<_, _>>();
    let groups = HashMap::from([(0, 1u8), (1, 1u8), (2, 2u8), (3, 2u8)]);

    let order =
        flat_artwork_overlap_render_order(&images, &[0, 1, 2, 3], &homographies, Some(&groups));

    assert_eq!(order, vec![1, 0, 2, 3]);
    for (position, &index) in order.iter().enumerate().skip(1) {
        let current = flat_artwork_projected_rect(&images[index], &homographies).unwrap();
        let overlaps_previous = order[..position].iter().any(|previous| {
            let previous = flat_artwork_projected_rect(&images[*previous], &homographies).unwrap();
            current.overlap_area(previous) > 0.0
        });
        assert!(
            overlaps_previous,
            "source {index} should overlap prior coverage"
        );
    }
}

fn identity_match(inliers: usize) -> MatchInfo {
    MatchInfo {
        homography: Matrix3::identity(),
        inliers,
        sequence_bridge: false,
        coarse_bridge: false,
        points: Vec::new(),
        candidate_points: Vec::new(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
        native_verified_points: Vec::new(),
    }
}

/// Build single-source stations plus image-level world poses for focused
/// compositor-order tests. `centers` are world centres of 1000×1000
/// stations, exactly the observations consumed by production topology
/// inference (任务 9.10).
fn compositor_order_fixture(
    centers: &[(f64, f64)],
) -> (
    Vec<ImageInfo>,
    Vec<FocusCaptureGroup>,
    HashMap<usize, Matrix3<f64>>,
) {
    let images = (0..centers.len())
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 3680 + index)))
        .collect::<Vec<_>>();
    let groups = (0..centers.len())
        .map(|index| FocusCaptureGroup {
            members: vec![index],
            anchor: index,
            local_to_anchor: HashMap::from([(index, Matrix3::identity())]),
        })
        .collect::<Vec<_>>();
    let poses = centers
        .iter()
        .enumerate()
        .map(|(index, &(x, y))| {
            (
                images[index].id,
                Matrix3::new(1.0, 0.0, x - 500.0, 0.0, 1.0, y - 500.0, 0.0, 0.0, 1.0),
            )
        })
        .collect::<HashMap<_, _>>();
    (images, groups, poses)
}

#[test]
fn compositor_order_uses_topology_rows_and_columns_not_world_coordinate_drift() {
    // Per-station drift makes a continuous y/x sort interleave each row in
    // the wrong order. Production topology first clusters columns, assigns
    // rows inside them, then composes top-to-bottom and right-to-left.
    let (images, groups, poses) =
        compositor_order_fixture(&[(900.0, 20.0), (900.0, 900.0), (0.0, 0.0), (0.0, 880.0)]);
    let station_topology = focus_station_topology(&images, &groups, &poses);

    assert_eq!(station_topology.row, vec![0, 1, 0, 1]);
    assert_eq!(station_topology.column, vec![0, 0, 1, 1]);
    assert_eq!(
        focus_source_order_group_compositor_order(groups.len(), &station_topology),
        vec![0, 2, 1, 3]
    );
}

#[test]
fn compositor_order_uses_station_index_only_for_exact_topology_ties() {
    // Requirement 5.1 prevents this malformed tie in production. Keeping a
    // final station key nevertheless makes synthetic/partial topology
    // deterministic without consulting filenames, world coordinates or
    // candidate/import order.
    let station_topology = topology::StationTopology {
        row: vec![0, 0, 1],
        column: vec![0, 0, 0],
        ..topology::StationTopology::default()
    };

    assert_eq!(
        focus_source_order_group_compositor_order(3, &station_topology),
        vec![0, 1, 2]
    );
}

#[test]
fn compositor_order_is_invariant_to_rename_and_import_order() {
    // Physical station identities in canonical topology order: right-top,
    // left-top, right-bottom, left-bottom.
    let physical_centers = [(900.0, 10.0), (0.0, 0.0), (900.0, 900.0), (0.0, 890.0)];
    let (original_images, original_groups, original_poses) =
        compositor_order_fixture(&physical_centers);
    let original_topology =
        focus_station_topology(&original_images, &original_groups, &original_poses);
    let original_order =
        focus_source_order_group_compositor_order(original_groups.len(), &original_topology);

    // Reindex stations by shuffled import order and rename every source.
    // Map the resulting station indices back to physical identities before
    // comparing, because import indices themselves intentionally differ.
    let permutation = [3usize, 0, 2, 1];
    let shuffled_centers = permutation
        .iter()
        .map(|&physical| physical_centers[physical])
        .collect::<Vec<_>>();
    let (mut shuffled_images, shuffled_groups, shuffled_poses) =
        compositor_order_fixture(&shuffled_centers);
    let shuffled_count = shuffled_images.len();
    for (index, image) in shuffled_images.iter_mut().enumerate() {
        image.filename = format!("renamed-{}.NEF", shuffled_count - index);
    }
    let shuffled_topology =
        focus_station_topology(&shuffled_images, &shuffled_groups, &shuffled_poses);
    let shuffled_order =
        focus_source_order_group_compositor_order(shuffled_groups.len(), &shuffled_topology);

    let original_physical = original_order;
    let shuffled_physical = shuffled_order
        .into_iter()
        .map(|station| permutation[station])
        .collect::<Vec<_>>();
    assert_eq!(original_physical, vec![0, 1, 2, 3]);
    assert_eq!(shuffled_physical, original_physical);
}

#[test]
fn anchor_frame_fusion_is_independent_of_global_station_placement() {
    let sources = (0..4)
        .map(|id| test_image(id, &format!("{id}.nef")))
        .collect::<Vec<_>>();
    let groups = vec![
        FocusCaptureGroup {
            members: vec![0, 1],
            anchor: 0,
            local_to_anchor: HashMap::new(),
        },
        FocusCaptureGroup {
            members: vec![2, 3],
            anchor: 2,
            local_to_anchor: HashMap::new(),
        },
    ];
    let child = Matrix3::new(1.001, 0.0, 5.0, 0.0, 1.001, -3.0, 0.0, 0.0, 1.0);
    let first = Matrix3::new(1.4, 0.2, 1000.0, 0.1, 1.2, 700.0, 0.0003, 0.0002, 1.0);
    let second = Matrix3::new(0.8, -0.3, 3000.0, 0.2, 0.9, 200.0, -0.0001, 0.0003, 1.0);
    let global = HashMap::from([
        (0, first),
        (1, first * child),
        (2, second),
        (3, second * child),
    ]);
    let local = focus_anchor_render_frames(&groups, &sources, &global, None).unwrap();
    for (id, expected) in [
        (0, Matrix3::identity()),
        (1, child),
        (2, Matrix3::identity()),
        (3, child),
    ] {
        assert!((local.homographies[&id] - expected).norm() < 1e-8);
    }
    assert!((local.anchor_to_world[0] - Matrix3::identity()).norm() < 1e-8);
    let mut moved = global.clone();
    let movement = Matrix3::new(1.3, 0.1, -800.0, 0.0, 1.1, 600.0, 0.0002, 0.0, 1.0);
    moved.insert(2, movement * second);
    moved.insert(3, movement * second * child);
    let other = focus_anchor_render_frames(&groups, &sources, &moved, None).unwrap();
    assert!((other.homographies[&3] - local.homographies[&3]).norm() < 1e-8);
    // Local source pixels and source provenance share the same composition.
    let expected_world = first.try_inverse().unwrap() * moved[&3];
    assert!(
        (normalized_homography(&(other.anchor_to_world[1] * other.homographies[&3])).unwrap()
            - normalized_homography(&expected_world).unwrap())
        .norm()
            < 1e-8
    );
}

#[test]
fn stitching_order_uses_stable_filenames_for_equivalent_graphs() {
    let first_images = vec![
        test_image(0, "c.jpg"),
        test_image(1, "a.jpg"),
        test_image(2, "b.jpg"),
    ];
    let first_matches = HashMap::from([((1, 2), identity_match(30)), ((0, 2), identity_match(20))]);
    let (first_order, _) = build_stitching_order(&first_images, &first_matches);

    let second_images = vec![
        test_image(0, "b.jpg"),
        test_image(1, "c.jpg"),
        test_image(2, "a.jpg"),
    ];
    let second_matches =
        HashMap::from([((0, 2), identity_match(30)), ((0, 1), identity_match(20))]);
    let (second_order, _) = build_stitching_order(&second_images, &second_matches);

    let first_names = first_order
        .iter()
        .map(|&index| first_images[index].filename.as_str())
        .collect::<Vec<_>>();
    let second_names = second_order
        .iter()
        .map(|&index| second_images[index].filename.as_str())
        .collect::<Vec<_>>();

    assert_eq!(first_names, vec!["a.jpg", "b.jpg", "c.jpg"]);
    assert_eq!(second_names, first_names);
}

#[test]
fn disconnected_stitch_graph_selects_the_largest_component() {
    let images = (0..7)
        .map(|index| test_image(index, &format!("tile-{index}.jpg")))
        .collect::<Vec<_>>();
    let matches = HashMap::from([
        ((0, 1), identity_match(80)),
        // The four-image component must win even though every one of its
        // edges has fewer inliers than the isolated two-image component.
        ((2, 3), identity_match(30)),
        ((3, 4), identity_match(30)),
        ((4, 5), identity_match(30)),
    ]);

    let (order, homographies) = build_stitching_order(&images, &matches);

    assert_eq!(order.len(), 4);
    assert_eq!(
        order.iter().copied().collect::<HashSet<_>>(),
        HashSet::from([2, 3, 4, 5])
    );
    assert_eq!(homographies.len(), 4);
}
