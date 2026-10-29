//! Source_RAW reference ROIs: provenance, the world-to-source mapping and the
//! production sampler, and the gradient ratio of aligned ROIs.

use super::*;

#[test]
fn merged_quality_reference_prefers_direct_raw_provenance() {
    let image = Rgb32FImage::from_fn(64, 64, |x, y| Rgb([x as f32 / 64.0, y as f32 / 64.0, 0.25]));
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 4,
        region_label: 1,
        owner: 1,
        world_origin: (20.0, 20.0),
        boundary_clearance: 16.0,
    };
    let source = QualitySource {
        owner: 1,
        path: "merged-reference.raw".into(),
        dimensions: image.dimensions(),
        geometry: SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 0,
        },
        native_warp: Some(NativeSourceWarp {
            displacement: None,
            source_to_station: Matrix3::identity(),
            station_to_world: Matrix3::identity(),
        }),
        native_registration: Some(NativeReferenceMap {
            world_to_station: Matrix3::identity(),
            station_to_world: Matrix3::identity(),
            station_origin: (0.0, 0.0),
            refinement: crate::panorama_utils::stitching::FocusSourceRefinement {
                translation: (3, -2),
                ..Default::default()
            },
        }),
    };
    let residual = ResidualWarp::new();
    let reference = source_roi_for_quality(&image, &roi, &source, &residual).unwrap();
    for y in 0..roi.side {
        for x in 0..roi.side {
            let pixel = reference.get_pixel(x, y);
            assert!((pixel[0] - (17.0 + x as f32) / 64.0).abs() < 1e-6);
            assert!((pixel[1] - (22.0 + y as f32) / 64.0).abs() < 1e-6);
        }
    }
    let legacy = QualitySource {
        native_registration: None,
        ..source
    };
    let reference = source_roi_for_quality(&image, &roi, &legacy, &residual).unwrap();
    assert!((reference.get_pixel(0, 0)[0] - 20.0 / 64.0).abs() < 1e-6);
}

#[test]
fn owner_reference_uses_the_rendering_kernel_for_fractional_fine_texture() {
    let source = Rgb32FImage::from_fn(32, 32, |x, y| {
        Rgb([if (x + y) % 2 == 0 { 0.15 } else { 0.85 }; 3])
    });
    let geometry = SourceGeometry {
        member_to_anchor: Matrix3::identity(),
        tile_to_world: Matrix3::new(1.0, 0.0, -0.37, 0.0, 1.0, 0.23, 0.0, 0.0, 1.0),
        station_id: 0,
    };
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 4,
        region_label: 1,
        owner: 1,
        world_origin: (10.0, 10.0),
        boundary_clearance: 8.0,
    };
    let sampled =
        source_roi_with_registration(&source, &roi, &geometry, &ResidualWarp::new(), None).unwrap();
    for y in 0..4 {
        for x in 0..4 {
            let expected = crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                &source,
                10.37 + f64::from(x),
                9.77 + f64::from(y),
            );
            assert!((sampled.get_pixel(x, y)[0] - expected[0]).abs() < 1e-6);
        }
    }
}

#[test]
fn owner_reference_replays_foreground_native_field_and_translation_in_render_order() {
    use crate::panorama_utils::{
        mosaic::{Field, NativeResidualDisplacement},
        stitching::{FocusForegroundMove, FocusSourceRefinement},
    };
    let source = Rgb32FImage::from_fn(128, 128, |x, y| {
        Rgb([x as f32 / 128.0, y as f32 / 128.0, 0.25])
    });
    let station_to_world = Matrix3::new(1.0, 0.0, 100.0, 0.0, 1.0, 200.0, 0.0, 0.0, 1.0);
    let mut field = Field::<2>::new(128, 128, 16.0);
    field.values.fill([1.5, -0.5]);
    let native = NativeReferenceMap {
        world_to_station: station_to_world.try_inverse().unwrap(),
        station_to_world,
        station_origin: (0.0, 0.0),
        refinement: FocusSourceRefinement {
            translation: (3, -2),
            displacement: Some(NativeResidualDisplacement::from_test_field(field)),
            foreground_move: Some(FocusForegroundMove {
                delta: (4, 0),
                left: 0,
                top: 0,
                destinations: GrayImage::from_pixel(128, 128, Luma([255])),
            }),
            source_sampling: None,
        },
    };
    let geometry = SourceGeometry {
        member_to_anchor: Matrix3::identity(),
        tile_to_world: station_to_world,
        station_id: 0,
    };
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 4,
        region_label: 1,
        owner: 1,
        world_origin: (112.0, 224.0),
        boundary_clearance: 8.0,
    };
    let actual = source_roi_with_registration(
        &source,
        &roi,
        &geometry,
        &ResidualWarp::new(),
        Some(&native),
    )
    .unwrap();
    for y in 0..4 {
        for x in 0..4 {
            let pixel = actual.get_pixel(x, y);
            assert!((pixel[0] - (6.5 + x as f32) / 128.0).abs() < 1e-6);
            assert!((pixel[1] - (25.5 + y as f32) / 128.0).abs() < 1e-6);
        }
    }
    let wrong =
        source_roi_with_registration(&source, &roi, &geometry, &ResidualWarp::new(), None).unwrap();
    assert!((wrong.get_pixel(0, 0)[0] - actual.get_pixel(0, 0)[0]).abs() > 0.04);
}

#[test]
fn reference_roi_reuses_production_sampler_for_gradient_and_noise_metrics() {
    let source = Rgb32FImage::from_fn(12, 12, |x, y| {
        let value = ((x * x + 5 * x * y + 3 * y + 7) % 23) as f32 / 22.0;
        Rgb([value, value * 0.91, value * 0.83])
    });
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 4,
        region_label: 1,
        owner: 1,
        world_origin: (2.25, 2.5),
        boundary_clearance: 16.0,
    };
    let geometry = SourceGeometry {
        member_to_anchor: Matrix3::identity(),
        tile_to_world: Matrix3::identity(),
        station_id: 0,
    };
    let reference =
        source_roi(&source, &roi, &geometry, &ResidualWarp::new()).expect("identity ROI");
    let source_info = QualitySource {
        native_registration: None,
        owner: 1,
        path: "owner.raw".to_string(),
        geometry: geometry.clone(),
        dimensions: source.dimensions(),
        native_warp: None,
    };
    assert_eq!(
        sample_quality_source_at_world(
            &source,
            Point2::new(roi.world_origin.0, roi.world_origin.1),
            &source_info,
            &ResidualWarp::new(),
        )
        .unwrap(),
        *reference.get_pixel(0, 0),
    );
    let expected =
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(&source, 2.25, 2.5);
    assert_eq!(reference.get_pixel(0, 0), &expected);

    let x0 = 2u32;
    let y0 = 2u32;
    let tx = 0.25f32;
    let ty = 0.5f32;
    let a = *source.get_pixel(x0, y0);
    let b = *source.get_pixel(x0 + 1, y0);
    let c = *source.get_pixel(x0, y0 + 1);
    let d = *source.get_pixel(x0 + 1, y0 + 1);
    let bilinear = Rgb(std::array::from_fn(|channel| {
        let top = a[channel] * (1.0 - tx) + b[channel] * tx;
        let bottom = c[channel] * (1.0 - tx) + d[channel] * tx;
        top * (1.0 - ty) + bottom * ty
    }));
    assert!(
        (reference.get_pixel(0, 0)[0] - bilinear[0]).abs() > 1.0e-6,
        "reference ROI must not silently fall back to the old bilinear kernel"
    );

    let output = Rgb32FImage::from_fn(roi.side, roi.side, |x, y| {
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            &source,
            roi.world_origin.0 + f64::from(x),
            roi.world_origin.1 + f64::from(y),
        )
    });
    let local_scale = 1.125;
    let (gradient_ratio, gradient_passes) = gradient_energy_ratio(&output, &reference, local_scale)
        .expect("the sampled ROIs must remain measurable")
        .expect("the reference gradient must remain non-zero");
    assert!((gradient_ratio - 1.0).abs() <= 1.0e-12);
    assert!(gradient_passes);
    let noise_ratio = quality_gate::noise_sigma_ratio(&output, &reference)
        .expect("identical production and reference sampling must have measurable noise");
    assert!((noise_ratio - 1.0).abs() <= 1.0e-12);
}

#[test]
fn source_roi_mapping_guard_requires_render_provenance_for_focus_band_and_native_delta() {
    // The production Quality_Gate mapping is intentionally only the
    // world -> Residual_Warp inverse -> member/tile inverse path.  A
    // focus render can apply two additional mappings after that path:
    // a local band homography and the accepted native displacement field.
    // Keep a coordinate-coded source so a future provenance integration
    // can compare the exact sample, rather than only comparing geometry
    // matrices.
    let source = Rgb32FImage::from_fn(48, 48, |x, y| {
        Rgb([x as f32 / 48.0, y as f32 / 48.0, (x + y) as f32 / 96.0])
    });
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 2,
        region_label: 1,
        owner: 1,
        world_origin: (10.25, 11.5),
        boundary_clearance: 16.0,
    };
    let geometry = SourceGeometry {
        member_to_anchor: Matrix3::identity(),
        tile_to_world: Matrix3::identity(),
        station_id: 10,
    };
    let qg_reference = source_roi(&source, &roi, &geometry, &ResidualWarp::new())
        .expect("identity world geometry must remain measurable");

    // `render_focus_layer` stores a source->world local band homography;
    // translation(+2, 0) therefore maps a world sample back to source x-2.
    // `warp_rendered_focus_layer` samples the candidate at +1 canvas pixel
    // for this synthetic native field, yielding a final source x-1.  These
    // values model the station 10/12 issue without adding a second,
    // production mapping implementation to the Quality_Gate.
    let world = Point2::new(roi.world_origin.0, roi.world_origin.1);
    let expected_with_render_provenance =
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            &source,
            world.x - 1.0,
            world.y,
        );
    let qg_pixel = qg_reference.get_pixel(0, 0);
    assert_ne!(qg_pixel, &expected_with_render_provenance);
    assert!(
        (qg_pixel[0] - expected_with_render_provenance[0]).abs() > 1.0e-5,
        "the current QG reference must not silently claim local-band/native provenance"
    );
}

#[test]
fn aligned_gradient_ratio_is_invariant_to_shared_local_scale() {
    let output = Rgb32FImage::from_fn(16, 16, |x, y| {
        let value = ((x * x + 3 * y + x * y) % 29) as f32 / 28.0;
        Rgb([value, value * 0.9, value * 0.8])
    });
    let reference = Rgb32FImage::from_fn(16, 16, |x, y| {
        let value = ((2 * x * x + y + x * y) % 31) as f32 / 30.0;
        Rgb([value, value * 0.9, value * 0.8])
    });
    let unit = gradient_energy_ratio(&output, &reference, 1.0)
        .unwrap()
        .unwrap()
        .0;
    for local_scale in [0.8, 1.125] {
        let scaled = gradient_energy_ratio(&output, &reference, local_scale)
            .unwrap()
            .unwrap()
            .0;
        assert!((scaled - unit).abs() <= 1.0e-12);
    }
}
