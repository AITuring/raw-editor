//! Measured residual models of Virtual_Tile relations: translation first, bounded affine
//! escalation, incoherent measurements, multimodal texture and the rejected stages.

use super::*;

fn residual_model_fixture(
    initial: Matrix3<f64>,
    correction: Matrix3<f64>,
    columns: usize,
    rows: usize,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let measured = correction * initial;
    (0..rows)
        .flat_map(|row| {
            (0..columns).map(move |column| {
                let source = Point2::new(70.0 + column as f64 * 65.0, 55.0 + row as f64 * 52.0);
                (source, transformed_point(&measured, source).unwrap())
            })
        })
        .collect()
}

#[test]
fn direct_residual_translation_uses_only_measured_correspondences() {
    let initial = Matrix3::new(
        1.002, 0.004, -96.0, -0.003, 0.998, 8.0, 0.000_006, -0.000_004, 1.0,
    );
    let correction = Matrix3::new(1.0, 0.0, 6.25, 0.0, 1.0, -4.5, 0.0, 0.0, 1.0);
    let points = residual_model_fixture(initial, correction, 6, 6);
    let outcome =
        virtual_tile_fit_measured_residual_model(&points, &initial, (500, 400), (500, 400));
    let maximum_consensus = outcome.max_translation_consensus;
    let fit = outcome.fit.expect("coherent measured residuals must fit");
    assert_eq!(fit.model, "translation");
    assert_eq!(maximum_consensus, points.len());
    assert_eq!(fit.inlier_indices.len(), points.len());
    assert!(fit.spatial_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT);
    for (source, target) in &points {
        let predicted = transformed_point(&fit.measured_homography, *source).unwrap();
        assert!((predicted - target).norm() < 1e-9);
    }

    // Mutation proof: a perfect prior plus only 23 measured pairs still has
    // only 23 pieces of evidence. The prior itself must never become the
    // 24th point or an accepted inlier.
    let prior_only_points = residual_model_fixture(initial, Matrix3::identity(), 23, 1);
    let outcome = virtual_tile_fit_measured_residual_model(
        &prior_only_points,
        &initial,
        (1_600, 400),
        (1_600, 400),
    );
    assert!(outcome.fit.is_none());
    assert_eq!(
        outcome.max_translation_consensus,
        STATION_RELATION_MIN_INLIERS - 1
    );
}

#[test]
fn direct_residual_escalates_to_bounded_affine_only_when_translation_fails() {
    let initial = Matrix3::new(1.0, 0.0, -90.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0);
    let correction = Matrix3::new(1.015, 0.008, 1.5, -0.006, 1.012, -2.0, 0.0, 0.0, 1.0);
    let points = residual_model_fixture(initial, correction, 6, 6);
    let outcome =
        virtual_tile_fit_measured_residual_model(&points, &initial, (500, 400), (500, 400));
    let translation_consensus = outcome.max_translation_consensus;
    let fit = outcome
        .fit
        .expect("bounded spatially coherent affine residual must fit");
    assert!(translation_consensus < STATION_RELATION_MIN_INLIERS);
    assert_eq!(fit.model, "affine");
    assert_eq!(fit.inlier_indices.len(), points.len());
    assert!(virtual_tile_residual_correction_is_bounded(
        &correction,
        &initial,
        virtual_tile_full_tile_bounds((500, 400))
    ));
}

#[test]
fn direct_residual_rejects_incoherent_measurements_and_is_order_deterministic() {
    let initial = Matrix3::new(1.0, 0.0, -80.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0);
    let correction = Matrix3::new(1.0, 0.0, 5.0, 0.0, 1.0, -3.0, 0.0, 0.0, 1.0);
    let points = residual_model_fixture(initial, correction, 6, 6);
    let forward =
        virtual_tile_fit_measured_residual_model(&points, &initial, (500, 400), (500, 400));
    let mut reversed = points.clone();
    reversed.reverse();
    let reverse =
        virtual_tile_fit_measured_residual_model(&reversed, &initial, (500, 400), (500, 400));
    let forward = forward.fit.unwrap();
    let reverse = reverse.fit.unwrap();
    assert_eq!(forward.model, reverse.model);
    assert_eq!(forward.inlier_indices.len(), reverse.inlier_indices.len());
    assert_eq!(
        forward
            .measured_homography
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        reverse
            .measured_homography
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let incoherent = (0..48)
        .map(|index| {
            let source = Point2::new(
                60.0 + f64::from(index % 8) * 52.0,
                45.0 + f64::from(index / 8) * 58.0,
            );
            let predicted = transformed_point(&initial, source).unwrap();
            let dx = f64::from((index * 37) % 31) - 15.0;
            let dy = f64::from((index * 53) % 29) - 14.0;
            (source, predicted + nalgebra::Vector2::new(dx, dy))
        })
        .collect::<Vec<_>>();
    let outcome =
        virtual_tile_fit_measured_residual_model(&incoherent, &initial, (500, 400), (500, 400));
    assert!(outcome.max_translation_consensus < STATION_RELATION_MIN_INLIERS);
    assert!(
        outcome.fit.is_none(),
        "an incorrect prior is not acceptance evidence"
    );
}

#[test]
fn residual_model_selects_a_spatially_broad_mode_from_multimodal_texture() {
    let initial = Matrix3::new(
        1.001, 0.003, -74.0, -0.002, 0.999, 6.0, 0.000_004, -0.000_003, 1.0,
    );
    let coherent_correction = Matrix3::new(1.0, 0.0, 6.5, 0.0, 1.0, -4.25, 0.0, 0.0, 1.0);
    let mut points = residual_model_fixture(initial, coherent_correction, 6, 5);
    // Add stronger-looking but mutually inconsistent repeated-texture modes.
    // None may borrow the prior or combine with the 30-point broad mode.
    points.extend((0..72).map(|index| {
        let source = Point2::new(
            45.0 + f64::from(index % 12) * 36.0,
            38.0 + f64::from(index / 12) * 57.0,
        );
        let predicted = transformed_point(&initial, source).unwrap();
        let mode = index % 6;
        let residual = nalgebra::Vector2::new(
            -14.0 + f64::from(mode) * 5.1,
            13.0 - f64::from((index * 5) % 7) * 4.2,
        );
        (source, predicted + residual)
    }));
    let outcome =
        virtual_tile_fit_measured_residual_model(&points, &initial, (520, 390), (500, 410));
    let fit = outcome
        .fit
        .expect("the spatially broad measured residual mode must be recovered");
    assert_eq!(fit.model, "translation");
    assert!(fit.inlier_indices.len() >= 30);
    assert!(fit.spatial_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT);
    assert_eq!(outcome.translation.rejection_reason, "accepted");
    assert_eq!(outcome.translation.bounded_correction, Some(true));
}

#[test]
fn residual_model_reports_each_rejected_bounded_stage() {
    let initial = Matrix3::new(1.0, 0.0, -80.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0);
    let unbounded = Matrix3::new(1.0, 0.0, 40.0, 0.0, 1.0, -35.0, 0.0, 0.0, 1.0);
    let points = residual_model_fixture(initial, unbounded, 6, 6);
    let outcome =
        virtual_tile_fit_measured_residual_model(&points, &initial, (500, 400), (500, 400));
    assert!(outcome.fit.is_none());
    assert_eq!(outcome.translation.candidate_inliers, points.len());
    assert_eq!(outcome.translation.bounded_correction, Some(false));
    assert_eq!(
        outcome.translation.rejection_reason,
        "correction_out_of_bounds"
    );
    assert!(!outcome.affine.rejection_reason.is_empty());
    assert!(!outcome.projective.rejection_reason.is_empty());
}
