use super::*;

fn overlap(width: f64, height: f64, p95_px: f64) -> OverlapResidual {
    OverlapResidual::new(
        2,
        7,
        WorldRect {
            left: 100.0,
            top: 200.0,
            width,
            height,
        },
        p95_px,
    )
}

#[test]
fn residual_warp_starts_empty_and_only_enables_above_p95_gate() {
    let mut model = ResidualWarp::new();
    assert!(model.regions.is_empty());
    assert!(model.add_overlap(overlap(80.0, 80.0, 3.0)).is_none());
    assert_eq!(model.add_overlap(overlap(80.0, 80.0, 3.01)), Some(0));
    assert_eq!(model.regions[0].columns.min(model.regions[0].rows), 4);
}

#[test]
fn virtual_tile_alias_and_source_station_use_the_same_inverse_field() {
    let world = WorldPoint { x: 356.0, y: 456.0 };
    let observations = (0..8)
        .flat_map(|row| {
            (0..8).map(move |column| {
                WarpObservation::new(
                    WorldPoint {
                        x: 100.0 + column as f64 * 64.0,
                        y: 200.0 + row as f64 * 64.0,
                    },
                    [4.0, -3.0],
                    0.2,
                )
            })
        })
        .collect::<Vec<_>>();
    let mut model = ResidualWarp::new();
    model
        .add_observations(overlap(448.0, 448.0, 10.0), &observations)
        .unwrap();
    model.register_station_alias(usize::MAX - 7, 7);
    let warped = model.regions[0].warp_point(world);
    let target = Point2::new(warped.x, warped.y);
    let source = model.inverse_world_for_station(target, 7);
    assert!((source - Point2::new(world.x, world.y)).norm() < 0.01);
    assert_eq!(
        source,
        model.inverse_world_for_station(target, usize::MAX - 7)
    );
    assert_eq!(target, model.inverse_world_for_station(target, 6));
}

#[test]
fn displacement_is_bounded_and_fades_at_region_edge() {
    let mut observations = Vec::new();
    for row in 0..4 {
        for column in 0..4 {
            observations.push(WarpObservation::new(
                WorldPoint {
                    x: 100.0 + column as f64 * 64.0,
                    y: 200.0 + row as f64 * 64.0,
                },
                [4.0, -3.0],
                0.2,
            ));
        }
    }
    let region = WarpRegion::from_observations(overlap(192.0, 192.0, 4.0), &observations);
    assert!(!region.insufficient_evidence);
    assert!(region.max_node_displacement_px <= RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX);
    assert_eq!(region.displacement_at(100.0, 300.0), [0.0, 0.0]);
    let centre = region.displacement_at(196.0, 296.0);
    assert!(centre[0].is_finite() && centre[1].is_finite());
}

#[test]
fn missing_residual_error_is_derived_after_field_constraints() {
    let _run_scope = degradation::begin_run_scope();
    let overlap = overlap(192.0, 192.0, 4.0);
    let observations = (0..4)
        .flat_map(|row| {
            (0..4).map(move |column| {
                WarpObservation::with_errors(
                    WorldPoint {
                        x: 100.0 + column as f64 * 64.0,
                        y: 200.0 + row as f64 * 64.0,
                    },
                    [4.0, 0.0],
                    0.1,
                    4.0,
                    f64::NAN,
                )
            })
        })
        .collect::<Vec<_>>();
    let region = WarpRegion::from_observations(overlap, &observations);
    assert!(!region.insufficient_evidence);
    assert!(region.reverted_cells > 0);
}

#[test]
fn invalid_nodes_are_gaussian_extrapolated_only_within_three_radii() {
    let mut observations = Vec::new();
    for row in 0..4 {
        for column in 0..4 {
            if row == 0 && column == 0 {
                observations.push(WarpObservation::new(
                    WorldPoint { x: 100.0, y: 200.0 },
                    [8.0, 0.0],
                    0.1,
                ));
                continue;
            }
            // Valid evidence elsewhere keeps the region above the 16-point
            // gate while leaving the corner's neighbours to extrapolation.
            observations.push(WarpObservation::new(
                WorldPoint {
                    x: 100.0 + column as f64 * 64.0,
                    y: 200.0 + row as f64 * 64.0,
                },
                [0.0, 0.0],
                0.1,
            ));
        }
    }
    let region = WarpRegion::from_observations(overlap(192.0, 192.0, 4.0), &observations);
    assert!(
        region
            .nodes()
            .iter()
            .all(|node| node.displacement[0].is_finite())
    );
}

#[test]
fn round_trip_gate_uses_the_existing_matcher() {
    let source = GrayImage::from_fn(96, 96, |x, y| {
        image::Luma([((x.wrapping_mul(13) ^ y.wrapping_mul(7)) & 255) as u8])
    });
    let target = source.clone();
    let identity = Matrix3::identity();
    let result = verify_round_trip_match(
        &source,
        &target,
        &identity,
        &identity,
        Point2::new(48.0, 48.0),
    );
    assert!(result.is_some());
}
