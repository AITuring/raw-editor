//! Virtual_Tile geometry, source sampling, focus-layer rendering and
//! shifted-mosaic detection tests.

use super::*;

#[test]
fn virtual_tile_geometry_uses_the_same_validated_station_bands_as_rendering() {
    let info = geometry_test_image(0, None);
    let poses = HashMap::from([(0, Matrix3::identity())]);
    let mut warp = FocusLayerWarp {
        bands: vec![crate::panorama_stitching::FocusWarpBand {
            homographies: HashMap::from([(
                0,
                Matrix3::new(1.0, 0.0, 50.0, 0.0, 1.0, 50.0, 0.0, 0.0, 1.0),
            )]),
            source_ranges: HashMap::from([(0, (0.0, 1.0))]),
            source_x_ranges: HashMap::new(),
            relax_foreground_seam: false,
            foreground_only: true,
            physical_edge: false,
        }],
    };
    let baseline =
        focus_stack_virtual_tile_geometry(&[&info], &poses, Projection::Planar, None).unwrap();
    let ignored =
        focus_stack_virtual_tile_geometry(&[&info], &poses, Projection::Planar, Some(&warp))
            .unwrap();
    assert_eq!(
        (ignored.width, ignored.height),
        (baseline.width, baseline.height)
    );
    assert_eq!(ignored.tile_to_world, baseline.tile_to_world);
    assert!(native_station_geometry(&warp).bands.is_empty());
    warp.bands[0].physical_edge = true;
    let retained =
        focus_stack_virtual_tile_geometry(&[&info], &poses, Projection::Planar, Some(&warp))
            .unwrap();
    assert_eq!(
        (retained.width, retained.height),
        (baseline.width + 50, baseline.height + 50)
    );
    assert_eq!(native_station_geometry(&warp).bands.len(), 1);
}

#[test]
fn retained_foreground_mask_survives_empty_alignment_pixels_and_render_scaling() {
    let mut info = geometry_test_image(0, None);
    info.alignment_image = GrayImage::new(0, 0);
    info.foreground_range = Some((0.0, 1.0));
    info.foreground_mask = Some(GrayImage::from_fn(100, 80, |x, y| {
        image::Luma([if (40..60).contains(&x) && (30..50).contains(&y) {
            255
        } else {
            0
        }])
    }));
    assert!(source_is_focus_foreground(&info, Point2::new(500.0, 400.0)));
    assert!(!source_is_focus_foreground(
        &info,
        Point2::new(100.0, 400.0)
    ));
    info.width = 500;
    info.height = 400;
    assert!(source_is_focus_foreground(&info, Point2::new(250.0, 200.0)));
    assert!(!source_is_focus_foreground(&info, Point2::new(50.0, 200.0)));
}

#[test]
fn quality_scale_uses_the_native_sampling_map_instead_of_the_seed_pose() {
    use crate::panorama_utils::stack_pipeline::{quality_gate, quality_gate_runner};
    let info = geometry_test_image(0, None);
    let roi = quality_gate::QualityRoi {
        x: 0,
        y: 0,
        side: 32,
        region_label: 1,
        owner: 1,
        world_origin: (64.0, 64.0),
        boundary_clearance: 16.0,
    };
    for scale in [0.5, 2.0] {
        let sampling = FocusSourceSampling::new(
            &info,
            &Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0),
            Projection::Planar,
            None,
        );
        let source = quality_gate_runner::QualitySource {
            native_warp: None,
            owner: 1,
            path: "native-sampling".into(),
            dimensions: info.dimensions(),
            geometry: quality_gate::SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 0,
            },
            native_registration: Some(quality_gate_runner::NativeReferenceMap {
                world_to_station: Matrix3::identity(),
                station_to_world: Matrix3::identity(),
                station_origin: (0.0, 0.0),
                refinement: FocusSourceRefinement {
                    translation: (0, 0),
                    displacement: None,
                    foreground_move: None,
                    source_sampling: Some(sampling),
                },
            }),
        };
        let measured = quality_gate_runner::measure_source_local_scale(
            &roi,
            &source,
            &residual_warp::ResidualWarp::new(),
            1.0,
        )
        .unwrap();
        assert!((measured.median - scale).abs() < 1e-10);
        assert_eq!(
            measured.fraction_at_least_095,
            if scale >= 0.95 { 1.0 } else { 0.0 }
        );
    }
}

#[test]
fn native_focus_refinement_samples_fine_texture_once_from_the_original() {
    let mut info = geometry_test_image(0, None);
    info.width = 64;
    info.height = 48;
    let source = Rgb32FImage::from_fn(64, 48, |x, y| {
        let value = if (x + y) % 2 == 0 { 0.15 } else { 0.85 };
        Rgb([value, value * 0.8, value * 0.6])
    });
    let mut field = crate::panorama_utils::mosaic::Field::<2>::new(64, 48, 16.0);
    field.values.fill([0.37, -0.23]);
    let refinement = FocusSourceRefinement {
        translation: (2, -1),
        displacement: Some(
            crate::panorama_utils::mosaic::NativeResidualDisplacement::from_test_field(field),
        ),
        ..Default::default()
    };
    let h = Matrix3::new(1.0, 0.0, 0.31, 0.0, 1.0, -0.19, 0.0, 0.0, 1.0);
    let direct = render_focus_layer_with_refinement(
        &info,
        &source,
        &h,
        Projection::Planar,
        None,
        0.0,
        0.0,
        64,
        48,
        Some(&refinement),
        None,
    );
    for y in 12..28 {
        for x in 12..44 {
            let expected = get_high_quality_interpolated_pixel(
                &source,
                x as f64 + 0.37 - 2.0 - 0.31,
                y as f64 - 0.23 + 1.0 + 0.19,
            );
            let actual = direct.image.get_pixel(x - direct.left, y - direct.top);
            for channel in 0..3 {
                assert!((actual[channel] - expected[channel]).abs() < 1e-6);
            }
        }
    }
    // The old two-filter path attenuates this detail despite identical geometry.
    let projected = render_focus_layer(
        &info,
        &source,
        &h,
        Projection::Planar,
        None,
        0.0,
        0.0,
        64,
        48,
    );
    let translated = translate_rendered_focus_layer(projected, 2, -1, 64, 48);
    let old = warp_rendered_focus_layer(translated, refinement.displacement.as_ref().unwrap());
    let actual = direct.image.get_pixel(24 - direct.left, 20 - direct.top)[0];
    let blurred = old.image.get_pixel(24 - old.left, 20 - old.top)[0];
    assert!((actual - blurred).abs() > 0.02);
}

#[test]
fn focus_layer_masks_optical_padding_before_ownership() {
    let mut info = geometry_test_image(0, None);
    info.width = 64;
    info.height = 48;
    let source = Rgb32FImage::from_fn(64, 48, |x, y| {
        if (10..54).contains(&x) && (8..40).contains(&y) {
            Rgb([0.002, 0.003, 0.001])
        } else {
            Rgb([0.0; 3])
        }
    });
    let layer = render_focus_layer(
        &info,
        &source,
        &Matrix3::identity(),
        Projection::Planar,
        None,
        0.0,
        0.0,
        64,
        48,
    );
    assert_eq!(layer.mask.get_pixel(4, 4)[0], 0);
    assert_eq!(layer.mask.get_pixel(32, 24)[0], 255);
    assert_eq!(layer.image.get_pixel(4, 4).0, [0.0; 3]);
    assert_eq!(layer.image.get_pixel(32, 24).0, [0.002, 0.003, 0.001]);
}

#[test]
fn pixel_aligned_canvas_preserves_integer_reference_coordinates() {
    let (offset, size) = pixel_aligned_canvas(-123.4, 876.2);
    assert_eq!(offset, 124.0);
    assert_eq!(size, 1001);
    assert_eq!(offset.fract(), 0.0);
    assert!(offset - 123.4 >= 0.0);
}

#[test]
fn virtual_tile_geometry_preserves_nonzero_world_origin() {
    let image = geometry_test_image(7, None);
    let image_ref = &image;
    let image_to_world = Matrix3::new(1.0, 0.0, -123.4, 0.0, 1.0, 55.2, 0.0, 0.0, 1.0);
    let homographies = HashMap::from([(image.id, image_to_world)]);
    let geometry =
        focus_stack_virtual_tile_geometry(&[image_ref], &homographies, Projection::Planar, None)
            .expect("translated source must have valid Virtual_Tile geometry");
    assert_eq!(geometry.tile_to_world[(0, 2)], -124.0);
    assert_eq!(geometry.tile_to_world[(1, 2)], 55.0);
    for source in [Point2::new(0.0, 0.0), Point2::new(999.0, 799.0)] {
        let world = image_to_world * Point3::new(source.x, source.y, 1.0);
        let tile = Point2::new(world.x + 124.0, world.y - 55.0);
        let recovered = geometry.tile_to_world * Point3::new(tile.x, tile.y, 1.0);
        assert!((recovered.x - world.x).abs() < 1e-9);
        assert!((recovered.y - world.y).abs() < 1e-9);
    }
}

#[test]
fn cubic_warp_sampling_preserves_more_edge_contrast_than_bilinear() {
    let image = Rgb32FImage::from_fn(8, 8, |x, _| {
        let value = if x < 4 { 0.0 } else { 1.0 };
        Rgb([value, value, value])
    });
    let bilinear_low = get_interpolated_pixel(&image, 3.25, 3.5)[0];
    let bilinear_high = get_interpolated_pixel(&image, 3.75, 3.5)[0];
    let cubic_low = get_high_quality_interpolated_pixel(&image, 3.25, 3.5)[0];
    let cubic_high = get_high_quality_interpolated_pixel(&image, 3.75, 3.5)[0];

    assert!(cubic_high - cubic_low > bilinear_high - bilinear_low);
    assert!((0.0..=1.0).contains(&cubic_low));
    assert!((0.0..=1.0).contains(&cubic_high));
}

#[test]
fn focus_tile_tie_break_is_stable_and_quality_first() {
    assert!(focus_tile_ownership_should_replace(0, 0, 1, 3));
    assert!(focus_tile_ownership_should_replace(80, 4, 81, 9));
    assert!(!focus_tile_ownership_should_replace(81, 4, 80, 1));
    // Equal edge distance must resolve to the same tile identity even if
    // the caller's loading order changes.
    assert!(focus_tile_ownership_should_replace(120, 9, 120, 4));
    assert!(!focus_tile_ownership_should_replace(120, 4, 120, 9));
}

#[test]
fn final_source_sampling_does_not_filter_a_fused_owner_boundary() {
    let mut info = geometry_test_image(0, None);
    info.width = 64;
    info.height = 64;
    let original = Rgb32FImage::from_fn(64, 64, |x, y| {
        Rgb([if (x + y) % 2 == 0 { 0.25 } else { 0.55 }; 3])
    });
    let refinement = FocusSourceRefinement {
        source_sampling: Some(FocusSourceSampling::new(
            &info,
            &Matrix3::identity(),
            Projection::Planar,
            None,
        )),
        ..Default::default()
    };
    let sources = FocusTileSources {
        ownership: OwnershipMap::new(
            64,
            64,
            (0..4096).map(|i| if i % 64 < 32 { 1 } else { 2 }).collect(),
            vec![PathBuf::from("/a.nef"), PathBuf::from("/b.nef")],
        )
        .unwrap(),
        origin: (0.0, 0.0),
        originals: vec![
            FocusTileSource {
                image: original.clone(),
                refinement: refinement.clone(),
            },
            FocusTileSource {
                image: Rgb32FImage::from_pixel(64, 64, Rgb([0.9; 3])),
                refinement,
            },
        ],
    };
    let point = Point2::new(31.49, 30.1);
    let expected = get_high_quality_interpolated_pixel(&original, point.x, point.y);
    assert_eq!(sources.sample(point), Some(expected));
    let fused = Rgb32FImage::from_fn(64, 64, |x, y| {
        if x < 32 {
            *original.get_pixel(x, y)
        } else {
            Rgb([0.9; 3])
        }
    });
    let filtered = get_high_quality_interpolated_pixel(&fused, point.x, point.y);
    assert!(
        (filtered[0] - expected[0]).abs() > 0.1,
        "fixture must expose cross-owner colour mixing"
    );
}

#[test]
fn shifted_mosaic_detection_catches_a_lens_scale_change() {
    let first = geometry_test_image(0, Some(35.0));
    let second = geometry_test_image(1, Some(85.0));
    let center_x = first.width as f64 * 0.5;
    let center_y = first.height as f64 * 0.5;
    let scale = 1.25;
    let scale_about_center = Matrix3::new(
        scale,
        0.0,
        center_x * (1.0 - scale),
        0.0,
        scale,
        center_y * (1.0 - scale),
        0.0,
        0.0,
        1.0,
    );
    let homographies = HashMap::from([(0, Matrix3::identity()), (1, scale_about_center)]);
    let images = vec![&first, &second];

    assert!(focus_stack_is_shifted_mosaic(
        &images,
        &homographies,
        Projection::Planar
    ));
}

#[test]
fn shifted_mosaic_detection_uses_focal_metadata_when_geometry_is_centered() {
    let first = geometry_test_image(0, Some(35.0));
    let second = geometry_test_image(1, Some(85.0));
    let homographies = HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())]);
    let images = vec![&first, &second];

    assert!(focus_stack_is_shifted_mosaic(
        &images,
        &homographies,
        Projection::Planar
    ));
}

#[test]
fn shifted_mosaic_detection_keeps_small_focus_breathing_on_the_focus_path() {
    let first = geometry_test_image(0, Some(85.0));
    let second = geometry_test_image(1, Some(85.0));
    let center_x = first.width as f64 * 0.5;
    let center_y = first.height as f64 * 0.5;
    let scale = 1.025;
    let small_breathing = Matrix3::new(
        scale,
        0.0,
        center_x * (1.0 - scale) + 18.0,
        0.0,
        scale,
        center_y * (1.0 - scale) + 12.0,
        0.0,
        0.0,
        1.0,
    );
    let homographies = HashMap::from([(0, Matrix3::identity()), (1, small_breathing)]);
    let images = vec![&first, &second];

    assert!(!focus_stack_is_shifted_mosaic(
        &images,
        &homographies,
        Projection::Planar
    ));
}
