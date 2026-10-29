//! Bounded residual correction and residual model selection: local scale over the evidence
//! region, scale, displacement and non-translation rejection, and most-inliers selection with
//! its median-error tie-break.

use super::*;

/// `sqrt(|det J|)` of a homography at `point`, by central differences.
fn numeric_local_scale(transform: &Matrix3<f64>, point: Point2<f64>) -> f64 {
    let step = 1e-3;
    let at = |dx: f64, dy: f64| {
        transformed_point(transform, Point2::new(point.x + dx, point.y + dy)).expect("finite point")
    };
    let ddx = (at(step, 0.0) - at(-step, 0.0)) / (2.0 * step);
    let ddy = (at(0.0, step) - at(0.0, -step)) / (2.0 * step);
    (ddx.x * ddy.y - ddx.y * ddy.x).abs().sqrt()
}

// ---------------------------------------------------------------------------
// 6. bounded residual correction
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_residual_correction_is_bounded_reads_local_scale_over_the_evidence_region() {
    assert_eq!(VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX, 24.0);
    assert_eq!(VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX, 8.0);
    assert_eq!(VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MIN, 0.98);
    assert_eq!(VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MAX, 1.02);
    let identity = Matrix3::identity();
    let evidence = (400.0, 200.0, 600.0, 400.0);
    let centre = Point2::new(500.0, 300.0);

    // Upper-left block 1.035, but sqrt(det J) = a / w^1.5 = 1 at the centre.
    let block: f64 = 1.035;
    let perspective = (block.powf(2.0 / 3.0) - 1.0) / centre.x;
    let projective = Matrix3::new(block, 0.0, 0.0, 0.0, block, 0.0, perspective, 0.0, 1.0);
    assert!((numeric_local_scale(&projective, centre) - 1.0).abs() < 1e-6);
    assert!(virtual_tile_residual_correction_is_bounded(
        &projective,
        &identity,
        evidence
    ));
    assert_eq!(
        virtual_tile_full_tile_bounds((1000, 700)),
        (0.0, 0.0, 1000.0, 700.0)
    );
    assert!(
        !virtual_tile_residual_correction_is_bounded(
            &projective,
            &identity,
            virtual_tile_full_tile_bounds((1000, 700)),
        ),
        "the full tile reaches the origin, where the local scale is 1.035"
    );
    // The bounds are evaluated where the initial relation predicts the evidence.
    assert!(!virtual_tile_residual_correction_is_bounded(
        &projective,
        &translation(-300.0, 0.0),
        evidence
    ));
    let shifted_perspective = (block.powf(2.0 / 3.0) - 1.0) / 200.0;
    let shifted = Matrix3::new(
        block,
        0.0,
        0.0,
        0.0,
        block,
        0.0,
        shifted_perspective,
        0.0,
        1.0,
    );
    assert!(virtual_tile_residual_correction_is_bounded(
        &shifted,
        &translation(-300.0, 0.0),
        evidence
    ));
}

#[test]
fn virtual_tile_residual_correction_is_bounded_rejects_scale_displacement_and_non_translation() {
    let identity = Matrix3::identity();
    let evidence = (400.0, 200.0, 600.0, 400.0);
    let centre = (500.0, 300.0);
    let bounded = |correction: Matrix3<f64>| {
        virtual_tile_residual_correction_is_bounded(&correction, &identity, evidence)
    };
    // True scale about the centre moves corners by at most 4.3px.
    for (scale, expected) in [
        (1.0, true),
        (1.015, true),
        (1.019, true),
        (1.021, false),
        (1.03, false),
        (0.981, true),
        (0.979, false),
        (0.97, false),
    ] {
        assert_eq!(
            bounded(about_center(centre, uniform_scale(scale))),
            expected,
            "scale {scale}"
        );
    }
    for (dx, dy, expected) in [
        (23.5, 0.0, true),
        (0.0, -23.5, true),
        (24.5, 0.0, false),
        (17.0, 17.0, false),
    ] {
        assert_eq!(
            bounded(translation(dx, dy)),
            expected,
            "translation ({dx}, {dy})"
        );
    }
    // A rotation about the centre moves the corners by 2·sin(θ/2)·141.4px.
    for (theta, expected) in [(0.05, true), (0.06, false), (-0.06, false)] {
        assert_eq!(
            bounded(about_center(centre, rotation(theta))),
            expected,
            "rotation {theta}"
        );
    }
    // Each part passes alone; together a corner moves 25.6px.
    assert!(bounded(translation(20.0, 0.0)));
    assert!(!bounded(
        translation(20.0, 0.0) * about_center(centre, rotation(0.05))
    ));
}

// ---------------------------------------------------------------------------
// 7. residual model selection
// ---------------------------------------------------------------------------

/// `rows × columns` measured pairs of `correction · initial` on a 500×400 tile.
fn residual_selection_points(
    initial: &Matrix3<f64>,
    correction: &Matrix3<f64>,
    columns: usize,
    rows: usize,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let measured = correction * initial;
    (0..rows)
        .flat_map(|row| {
            (0..columns).map(move |column| {
                let source = Point2::new(
                    40.0 + column as f64 * 420.0 / (columns - 1) as f64,
                    35.0 + row as f64 * 330.0 / (rows - 1) as f64,
                );
                (source, transformed_point(&measured, source).unwrap())
            })
        })
        .collect()
}

fn fit_residual_model(
    points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    selection: VirtualTileResidualSelection,
) -> VirtualTileResidualModelOutcome {
    virtual_tile_fit_measured_residual_model_with_threshold(
        points,
        initial,
        (500, 400),
        (500, 400),
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        virtual_tile_full_tile_bounds((500, 400)),
        selection,
        StationOverlapMeasure::Rectangles,
    )
}

#[test]
fn most_inliers_selection_keeps_the_richer_model_a_passing_translation_would_truncate() {
    let initial = translation(-90.0, 5.0);
    // A 0.9° rotation about the tile centre: a translation explains only the
    // central band within 3px; the bounded affine explains every pair.
    let correction = translation(1.5, -2.0) * about_center((250.0, 200.0), rotation(0.0157));
    let points = residual_selection_points(&initial, &correction, 10, 8);

    let lowest = fit_residual_model(
        &points,
        &initial,
        VirtualTileResidualSelection::LowestComplexity,
    );
    let lowest_fit = lowest.fit.expect("the translation stage passes on its own");
    assert_eq!(lowest_fit.model, "translation");
    assert_eq!(lowest.translation.rejection_reason, "accepted");
    assert_eq!(lowest.affine.rejection_reason, "not_run");
    assert!(lowest_fit.inlier_indices.len() < points.len());

    let most = fit_residual_model(&points, &initial, VirtualTileResidualSelection::MostInliers);
    let most_fit = most.fit.expect("every stage is evaluated");
    assert_eq!(most.translation.rejection_reason, "accepted");
    assert_eq!(most.affine.rejection_reason, "accepted");
    assert_ne!(most_fit.model, "translation");
    assert_eq!(most_fit.inlier_indices.len(), points.len());
    assert!(most_fit.inlier_indices.len() > lowest_fit.inlier_indices.len());
    for (source, target) in &points {
        let predicted = transformed_point(&most_fit.measured_homography, *source).unwrap();
        assert!((predicted - target).norm() < 1e-6);
    }
}

#[test]
fn most_inliers_selection_breaks_an_inlier_tie_by_strictly_lower_median_error() {
    let initial = translation(-90.0, 5.0);
    // Small enough that the translation keeps every pair within 3px, yet
    // leaves a measurable error that the affine stage removes.
    let correction = translation(1.5, -2.0) * about_center((250.0, 200.0), rotation(0.004));
    let points = residual_selection_points(&initial, &correction, 10, 8);

    let lowest = fit_residual_model(
        &points,
        &initial,
        VirtualTileResidualSelection::LowestComplexity,
    );
    let lowest_fit = lowest.fit.expect("translation passes");
    assert_eq!(lowest_fit.model, "translation");
    assert_eq!(lowest_fit.inlier_indices.len(), points.len());
    assert!(lowest_fit.median_error_px > 0.1);

    let most = fit_residual_model(&points, &initial, VirtualTileResidualSelection::MostInliers);
    let most_fit = most.fit.expect("every stage is evaluated");
    assert_eq!(most_fit.inlier_indices.len(), points.len());
    assert_ne!(
        most_fit.model, "translation",
        "the tie goes to the lower error"
    );
    assert!(most_fit.median_error_px < lowest_fit.median_error_px);
}
