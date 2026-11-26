//! Residual_Warp Jacobian and round-trip contract properties (需求 8.3/8.4/8.5).
//!
//! The generated fields use constant, contract-safe node values.  Their
//! single-region Jacobian is exactly the identity; for a multi-region station
//! the edge-weighted partition adds only a bounded smooth blend term.  The
//! sampled floor below is deliberately conservative and remains test-only.

use nalgebra::{Matrix2, Point2};
use proptest::prelude::*;

use super::super::report::{WorldPoint, WorldRect};
use super::super::residual_warp::{OverlapResidual, ResidualWarp, WarpObservation};

// For one contract-safe field, bilinear interpolation contributes at most
// sqrt(2) * δ / step to the operator-norm perturbation. The smoothstep edge
// contributes at most D * 1.5 / fade. Subtracting that bound from identity
// gives the auditable lower bound used by the property (the generated blend
// fields use smaller constant displacements, so this full-contract bound is
// deliberately conservative for their partition-of-unity composition).
const CONTRACT_JACOBIAN_PERTURBATION_BOUND: f64 =
    std::f64::consts::SQRT_2 * 8.0 / 64.0 + 32.0 * 1.5 / 128.0;
const JACOBIAN_SIGMA_FLOOR: f64 = 1.0 - CONTRACT_JACOBIAN_PERTURBATION_BOUND;

fn observations(world: WorldRect, displacement: [f64; 2]) -> Vec<WarpObservation> {
    (0..4)
        .flat_map(|row| {
            (0..4).map(move |column| {
                WarpObservation::new(
                    WorldPoint {
                        x: world.left + column as f64 * 64.0,
                        y: world.top + row as f64 * 64.0,
                    },
                    displacement,
                    0.1,
                )
            })
        })
        .collect()
}

fn map_jacobian(model: &ResidualWarp, point: Point2<f64>) -> Matrix2<f64> {
    let dx = model.forward_world_for_station(point + nalgebra::Vector2::new(0.5, 0.0), 7)
        - model.forward_world_for_station(point - nalgebra::Vector2::new(0.5, 0.0), 7);
    let dy = model.forward_world_for_station(point + nalgebra::Vector2::new(0.0, 0.5), 7)
        - model.forward_world_for_station(point - nalgebra::Vector2::new(0.0, 0.5), 7);
    Matrix2::from_columns(&[dx, dy])
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]

    #[test]
    fn overlapping_contract_preserves_jacobian_floor_and_round_trip(
        regions in prop::collection::vec(
            (-64i32..=64, -64i32..=64, -2.0f64..2.0f64, -2.0f64..2.0f64),
            1..=4,
        )
    ) {
        let mut model = ResidualWarp::new();
        let mut probes = Vec::new();
        for (index, (left_offset, top_offset, dx, dy)) in regions.iter().copied().enumerate() {
            let world = WorldRect {
                left: 256.0 + f64::from(left_offset),
                top: 384.0 + f64::from(top_offset),
                width: 192.0,
                height: 192.0,
            };
            model
                .add_observations(
                    OverlapResidual::new(100 + index, 7, world, 10.0),
                    &observations(world, [dx, dy]),
                )
                .expect("contract-safe generated region should be enabled");
            let mid_y = world.top + 96.0;
            probes.extend([
                Point2::new(world.left - 0.5, mid_y),
                Point2::new(world.left + 0.5, mid_y),
                Point2::new(world.left + world.width - 0.5, mid_y),
                Point2::new(world.left + world.width + 0.5, mid_y),
                Point2::new(world.left + 96.0, mid_y),
            ]);
        }

        for point in probes {
            let target = model.forward_world_for_station(point, 7);
            let recovered = model.inverse_world_for_station(target, 7);
            prop_assert!((recovered - point).norm() <= 1.0,
                "round-trip error {:.4} at {point:?}: {recovered:?} -> {target:?}",
                (recovered - point).norm());

            let sigma_min = map_jacobian(&model, point).svd(false, false).singular_values.min();
            prop_assert!(sigma_min.is_finite() && sigma_min >= JACOBIAN_SIGMA_FLOOR,
                "sigma_min={sigma_min:.6} at {point:?}");
        }
    }
}

#[test]
fn one_region_and_empty_station_keep_the_legacy_bit_pattern() {
    let world = WorldRect {
        left: 100.0,
        top: 200.0,
        width: 192.0,
        height: 192.0,
    };
    let mut model = ResidualWarp::new();
    model
        .add_observations(
            OverlapResidual::new(2, 7, world, 10.0),
            &observations(world, [4.0, -3.0]),
        )
        .unwrap();
    let point = Point2::new(196.5, 296.5);
    let expected = model.regions[0].warp_point(WorldPoint {
        x: point.x,
        y: point.y,
    });
    assert_eq!(
        model.forward_world_for_station(point, 7),
        Point2::new(expected.x, expected.y)
    );
    assert_eq!(
        ResidualWarp::new().forward_world_for_station(point, 7),
        point
    );
}
