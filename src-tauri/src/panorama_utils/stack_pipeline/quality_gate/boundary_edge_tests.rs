use super::*;
fn edge_image(shift: f64) -> Rgb32FImage {
    Rgb32FImage::from_fn(128, 128, |x, y| {
        let amplitude = 0.5 + 0.4 * ((x as f64 - 56.0) / 4.0).tanh();
        let position = 64.35 + if x >= 64 { shift } else { 0.0 };
        let value = 0.05 + amplitude * (0.5 + 0.5 * ((y as f64 - position) * 2.0).tanh());
        image::Rgb([value as f32; 3])
    })
}
#[test]
fn continuous_edge_keeps_its_position_when_contrast_changes() {
    let image = edge_image(0.0);
    let (error, _) = paired_boundary_edge_error(&image, 64, 64, (1.0, 0.0)).unwrap();
    assert!(
        error < 0.35,
        "illumination was mistaken for edge displacement: {error}"
    );
}
#[test]
fn common_contour_is_not_replaced_by_a_different_stronger_parallel_edge() {
    let image = Rgb32FImage::from_fn(128, 128, |x, y| {
        let (first, second) = if x < 64 { (0.4, 0.5) } else { (0.5, 0.4) };
        let step = |position: f64| 0.5 + 0.5 * ((y as f64 - position) * 2.0).tanh();
        image::Rgb([(0.05 + first * step(51.35) + second * step(64.35)) as f32; 3])
    });
    let (error, _) = paired_boundary_edge_error(&image, 64, 64, (1.0, 0.0)).unwrap();
    assert!(
        error < 0.35,
        "different continuous contours were paired: {error}"
    );
}

#[test]
fn isolated_fifteen_pixel_edge_jump_remains_a_measurable_failure() {
    let (error, _) = paired_boundary_edge_error(&edge_image(15.0), 64, 64, (1.0, 0.0)).unwrap();
    // This edge sits one pixel from the 16px band's end, so its
    // truncated derivative footprint can bias the subpixel fit.
    assert!(
        (14.0..16.0).contains(&error),
        "large jump was hidden: {error}"
    );
    assert!(error > QUALITY_BOUNDARY_MAX);
}
#[test]
fn one_unrelated_bright_pixel_cannot_hide_an_isolated_edge_jump() {
    let mut image = edge_image(15.0);
    image.put_pixel(78, 64, image::Rgb([1.0; 3]));
    let (error, _) = paired_boundary_edge_error(&image, 64, 64, (1.0, 0.0)).unwrap();
    assert!(
        error > QUALITY_BOUNDARY_MAX,
        "an unrelated glint hid the jump: {error}"
    );
}
#[test]
fn covered_interpolation_halo_does_not_create_a_photographic_stroke() {
    let image = Rgb32FImage::from_fn(128, 128, |_, y| {
        image::Rgb([if y < 62 { 0.0 } else { 0.9 }; 3])
    });
    let halo = GrayImage::from_fn(128, 128, |_, y| image::Luma([if y < 60 { 0 } else { 255 }]));
    assert!(
        paired_boundary_edge_error_with_coverage(&image, Some(&halo), 64, 64, (1.0, 0.0)).is_err()
    );
    let full_support =
        GrayImage::from_fn(128, 128, |_, y| image::Luma([if y < 40 { 0 } else { 255 }]));
    assert!(
        paired_boundary_edge_error_with_coverage(&image, Some(&full_support), 64, 64, (1.0, 0.0))
            .unwrap()
            .0
            < 1e-6
    );
}

#[test]
fn transparency_edges_are_not_strokes_but_covered_black_edges_are() {
    let image = Rgb32FImage::from_fn(128, 128, |_, y| {
        image::Rgb([if y < 64 { 0.0 } else { 0.9 }; 3])
    });
    let coverage = GrayImage::from_fn(128, 128, |_, y| image::Luma([if y < 64 { 0 } else { 255 }]));
    assert!(
        paired_boundary_edge_error_with_coverage(&image, Some(&coverage), 64, 64, (1.0, 0.0))
            .is_err()
    );
    let covered = GrayImage::from_pixel(128, 128, image::Luma([255]));
    let (error, _) =
        paired_boundary_edge_error_with_coverage(&image, Some(&covered), 64, 64, (1.0, 0.0))
            .unwrap();
    assert!(error < 1e-6);
}

#[test]
fn parallel_segments_outside_the_seam_window_are_not_a_crossing_pair() {
    let image = Rgb32FImage::from_fn(128, 128, |x, y| {
        let d = y as f64 - 64.0;
        let edge = 64.0
            + if y < 64 {
                25.0 + 2.0 * d
            } else {
                -25.0 + 2.0 * d
            };
        image::Rgb([if x as f64 >= edge { 0.9 } else { 0.1 }; 3])
    });
    assert_eq!(
        paired_boundary_edge_error(&image, 64, 64, (0.0, 1.0)),
        Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE)
    );
}

#[test]
fn different_parallel_edges_are_not_one_crossing_stroke() {
    let image = Rgb32FImage::from_fn(128, 128, |_, y| {
        image::Rgb([if y <= 20 || y >= 44 { 0.9 } else { 0.1 }; 3])
    });
    assert!(paired_boundary_edge_error(&image, 64, 32, (0.0, 1.0)).is_err());
}

#[test]
fn shifted_edge_is_still_rejected_at_the_original_threshold() {
    let image = edge_image(3.0);
    let (error, _) = paired_boundary_edge_error(&image, 64, 64, (1.0, 0.0)).unwrap();
    assert!(
        (error - 3.0).abs() < 0.35,
        "wrong native edge displacement: {error}"
    );
    assert!(error > QUALITY_BOUNDARY_P95_MAX);
}
