//! Tiled mask generation against full-frame generation.
//!
//! Programmatic, brush, flow, geometry, sampled-color, luminance and AI masks match
//! across tile seams with exact halos; routing tiles only supported non-empty
//! sub-masks.

use super::*;

fn grayscale_png_data_url(image: &GrayImage) -> String {
    let mut encoded = Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, ImageFormat::Png)
        .expect("encode deterministic AI mask fixture");
    format!(
        "data:image/png;base64,{}",
        general_purpose::STANDARD.encode(encoded.get_ref())
    )
}

fn brush_parameters(flow: bool) -> Value {
    let first_line = if flow {
        serde_json::json!({
            "tool": "brush",
            "brushSize": 29.0,
            "feather": 0.37,
            "flow": 42.0,
            "points": [{ "x": 8.0, "y": 12.0 }, { "x": 128.0, "y": 91.0 }]
        })
    } else {
        serde_json::json!({
            "tool": "brush",
            "brushSize": 29.0,
            "feather": 0.37,
            "points": [{ "x": 8.0, "y": 12.0 }, { "x": 128.0, "y": 91.0 }]
        })
    };
    let eraser_line = if flow {
        serde_json::json!({
            "tool": "eraser",
            "brushSize": 17.0,
            "feather": 0.61,
            "flow": 73.0,
            "points": [{ "x": 76.0, "y": 5.0 }, { "x": 65.0, "y": 104.0 }]
        })
    } else {
        serde_json::json!({
            "tool": "eraser",
            "brushSize": 17.0,
            "feather": 0.61,
            "points": [{ "x": 76.0, "y": 5.0 }, { "x": 65.0, "y": 104.0 }]
        })
    };
    serde_json::json!({ "lines": [first_line, eraser_line] })
}

#[test]
fn tiled_programmatic_and_brush_composition_matches_full_frame_across_seams() {
    const WIDTH: u32 = 141;
    const HEIGHT: u32 = 109;
    const TEST_TILE_EDGE: u32 = 64;
    const SCALE: f32 = 1.25;
    const CROP_OFFSET: (f32, f32) = (11.5, 7.25);

    let cases = [
        (
            "radial",
            serde_json::json!({
                "centerX": 61.0,
                "centerY": 47.0,
                "radiusX": 54.0,
                "radiusY": 31.0,
                "rotation": 23.0,
                "feather": 0.41
            }),
        ),
        (
            "linear",
            serde_json::json!({
                "startX": 17.0,
                "startY": 19.0,
                "endX": 119.0,
                "endY": 88.0,
                "range": 38.0
            }),
        ),
        ("brush", brush_parameters(false)),
        ("clone", brush_parameters(false)),
        ("heal", brush_parameters(false)),
        ("flow", brush_parameters(true)),
        ("all", serde_json::json!({})),
    ];
    let modes = [
        SubMaskMode::Additive,
        SubMaskMode::Subtractive,
        SubMaskMode::Intersect,
    ];

    for (mask_type, parameters) in cases {
        for mode in modes {
            let sub_mask = test_sub_mask(mask_type, parameters.clone(), mode);
            let base = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
                Luma([x.wrapping_mul(11).wrapping_add(y * 7) as u8])
            });

            let mut expected = (mode != SubMaskMode::Additive).then(|| base.clone());
            let mut full_sub_mask =
                generate_sub_mask_bitmap(&sub_mask, WIDTH, HEIGHT, SCALE, CROP_OFFSET, None)
                    .expect("test mask type should rasterize");
            apply_sub_mask_modifiers(&mut full_sub_mask, &sub_mask);
            composite_sub_mask(&mut expected, full_sub_mask, mode);

            let mut actual = (mode != SubMaskMode::Additive).then_some(base);
            composite_sub_mask_tiled_with_edge(
                &mut actual,
                &sub_mask,
                WIDTH,
                HEIGHT,
                SCALE,
                CROP_OFFSET,
                None,
                TEST_TILE_EDGE,
            );

            let actual = actual.expect("tiled composition output");
            let expected = expected.expect("full-frame composition output");
            let first_difference = actual
                .as_raw()
                .iter()
                .zip(expected.as_raw())
                .position(|(actual, expected)| actual != expected)
                .map(|index| {
                    (
                        index as u32 % WIDTH,
                        index as u32 / WIDTH,
                        actual.as_raw()[index],
                        expected.as_raw()[index],
                    )
                });
            assert_eq!(
                first_difference, None,
                "{mask_type} {mode:?} changed at a tile boundary"
            );
        }
    }
}

#[test]
fn first_brush_and_flow_submasks_match_full_frame_when_generated_in_tiles() {
    const WIDTH: u32 = 141;
    const HEIGHT: u32 = 109;

    for (mask_type, parameters) in [
        ("brush", brush_parameters(false)),
        ("flow", brush_parameters(true)),
    ] {
        let sub_mask = test_sub_mask(mask_type, parameters, SubMaskMode::Additive);
        let mut expected_sub_mask =
            generate_sub_mask_bitmap(&sub_mask, WIDTH, HEIGHT, 1.0, (0.0, 0.0), None)
                .expect("brush-like test mask should rasterize");
        apply_sub_mask_modifiers(&mut expected_sub_mask, &sub_mask);

        let mut actual = None;
        composite_sub_mask_tiled_with_edge(
            &mut actual,
            &sub_mask,
            WIDTH,
            HEIGHT,
            1.0,
            (0.0, 0.0),
            None,
            64,
        );

        assert_eq!(actual, Some(expected_sub_mask), "{mask_type}");
    }
}

#[test]
fn tiled_geometry_sampled_color_and_luminance_masks_match_full_frame_with_exact_halo() {
    const WIDTH: u32 = 141;
    const HEIGHT: u32 = 109;
    const TEST_TILE_EDGE: u32 = 37;
    const SCALE: f32 = 1.25;
    const CROP_OFFSET: (f32, f32) = (11.5, 7.25);

    let source_image = DynamicImage::ImageRgba8(RgbaImage::from_fn(181, 137, |x, y| {
        Rgba([
            x.wrapping_mul(17).wrapping_add(y * 3) as u8,
            x.wrapping_mul(5).wrapping_add(y * 19) as u8,
            x.wrapping_mul(11).wrapping_add(y * 7) as u8,
            255,
        ])
    }));
    let geometry_params = crate::image_processing::GeometryParams {
        distortion: 8.0,
        projection: 14.0,
        vertical: 9.0,
        horizontal: -7.0,
        rotate: 2.0,
        aspect: 3.0,
        scale: 98.0,
        x_offset: 1.0,
        y_offset: -1.0,
        lens_dist_k1: 0.008,
        tca_vr: 1.001,
        tca_vb: 0.999,
        vig_k1: -0.04,
        ..crate::image_processing::GeometryParams::default()
    };
    let warped = crate::image_processing::warp_image_geometry(&source_image, geometry_params);
    let range_source = GeometryWarpSampler::new(&source_image, geometry_params);
    let cases = [
        (
            "color",
            serde_json::json!({
                "targetX": 72.0,
                "targetY": 58.0,
                "tolerance": 72.0,
                "grow": 100.0,
                "feather": 100.0,
                "rotation": 17.0,
                "flipHorizontal": true,
                "flipVertical": false,
                "orientationSteps": 1
            }),
        ),
        (
            "luminance",
            serde_json::json!({
                "targetX": 113.0,
                "targetY": 41.0,
                "tolerance": 32.0,
                "grow": -100.0,
                "feather": 100.0,
                "rotation": -13.0,
                "flipHorizontal": false,
                "flipVertical": true,
                "orientationSteps": 2
            }),
        ),
    ];

    for (mask_type, parameters) in cases {
        for mode in [
            SubMaskMode::Additive,
            SubMaskMode::Subtractive,
            SubMaskMode::Intersect,
        ] {
            let sub_mask = test_sub_mask(mask_type, parameters.clone(), mode);
            let base = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
                Luma([x.wrapping_mul(11).wrapping_add(y * 7) as u8])
            });

            let mut expected = (mode != SubMaskMode::Additive).then(|| base.clone());
            let mut full_sub_mask = generate_sub_mask_bitmap(
                &sub_mask,
                WIDTH,
                HEIGHT,
                SCALE,
                CROP_OFFSET,
                Some(&warped),
            )
            .expect("range mask should rasterize");
            let first_pixel = full_sub_mask.get_pixel(0, 0)[0];
            assert!(
                full_sub_mask.pixels().any(|pixel| pixel[0] != first_pixel),
                "{mask_type} fixture must exercise non-uniform pixels"
            );
            apply_sub_mask_modifiers(&mut full_sub_mask, &sub_mask);
            composite_sub_mask(&mut expected, full_sub_mask, mode);

            let mut actual = (mode != SubMaskMode::Additive).then_some(base);
            assert!(composite_sub_mask_tiled_with_edge(
                &mut actual,
                &sub_mask,
                WIDTH,
                HEIGHT,
                SCALE,
                CROP_OFFSET,
                Some(&range_source),
                TEST_TILE_EDGE,
            ));

            let actual = actual.expect("tiled composition output");
            let expected = expected.expect("full-frame composition output");
            let first_difference = actual
                .as_raw()
                .iter()
                .zip(expected.as_raw())
                .position(|(actual, expected)| actual != expected)
                .map(|index| {
                    (
                        index as u32 % WIDTH,
                        index as u32 / WIDTH,
                        actual.as_raw()[index],
                        expected.as_raw()[index],
                    )
                });
            assert_eq!(
                first_difference, None,
                "{mask_type} {mode:?} changed at an overlap seam"
            );
        }
    }
}

#[test]
fn tiled_ai_masks_match_full_frame_with_exact_halo_and_one_decoded_source() {
    const WIDTH: u32 = 141;
    const HEIGHT: u32 = 109;
    const TEST_TILE_EDGE: u32 = 37;
    const SCALE: f32 = 1.25;
    const CROP_OFFSET: (f32, f32) = (11.5, 7.25);

    let full_mask = GrayImage::from_fn(181, 137, |x, y| {
        Luma([x.wrapping_mul(17).wrapping_add(y.wrapping_mul(13)) as u8])
    });
    let data_url = grayscale_png_data_url(&full_mask);
    let binary_parameters = serde_json::json!({
        "startX": 12.0,
        "startY": 9.0,
        "endX": 128.0,
        "endY": 96.0,
        "maskDataBase64": data_url,
        "grow": 100.0,
        "feather": 100.0,
        "rotation": 17.0,
        "flipHorizontal": true,
        "flipVertical": false,
        "orientationSteps": 1
    });
    let cases = vec![
        ("ai-subject", binary_parameters.clone()),
        ("ai-foreground", binary_parameters.clone()),
        ("ai-sky", binary_parameters.clone()),
        ("quick-eraser", binary_parameters),
        (
            "ai-depth",
            serde_json::json!({
                "maskDataBase64": grayscale_png_data_url(&full_mask),
                "minDepth": 18.0,
                "maxDepth": 82.0,
                "minFade": 11.0,
                "maxFade": 9.0,
                "grow": -100.0,
                "feather": 60.0,
                "rotation": -13.0,
                "flipHorizontal": false,
                "flipVertical": true,
                "orientationSteps": 2
            }),
        ),
    ];

    for (mask_type, parameters) in cases {
        for mode in [
            SubMaskMode::Additive,
            SubMaskMode::Subtractive,
            SubMaskMode::Intersect,
        ] {
            let sub_mask = test_sub_mask(mask_type, parameters.clone(), mode);
            let base = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
                Luma([x.wrapping_mul(11).wrapping_add(y * 7) as u8])
            });

            let mut expected = (mode != SubMaskMode::Additive).then(|| base.clone());
            let mut full_sub_mask =
                generate_sub_mask_bitmap(&sub_mask, WIDTH, HEIGHT, SCALE, CROP_OFFSET, None)
                    .expect("AI mask should decode and rasterize");
            let first_pixel = full_sub_mask.get_pixel(0, 0)[0];
            assert!(
                full_sub_mask.pixels().any(|pixel| pixel[0] != first_pixel),
                "{mask_type} fixture must exercise non-uniform pixels"
            );
            apply_sub_mask_modifiers(&mut full_sub_mask, &sub_mask);
            composite_sub_mask(&mut expected, full_sub_mask, mode);

            let mut actual = (mode != SubMaskMode::Additive).then_some(base);
            assert!(composite_sub_mask_tiled_with_edge(
                &mut actual,
                &sub_mask,
                WIDTH,
                HEIGHT,
                SCALE,
                CROP_OFFSET,
                None,
                TEST_TILE_EDGE,
            ));

            let actual = actual.expect("tiled AI composition output");
            let expected = expected.expect("full-frame AI composition output");
            let first_difference = actual
                .as_raw()
                .iter()
                .zip(expected.as_raw())
                .position(|(actual, expected)| actual != expected)
                .map(|index| {
                    (
                        index as u32 % WIDTH,
                        index as u32 / WIDTH,
                        actual.as_raw()[index],
                        expected.as_raw()[index],
                    )
                });
            assert_eq!(
                first_difference, None,
                "{mask_type} {mode:?} changed at an overlap seam"
            );
        }
    }
}

#[test]
fn tiled_generation_routes_supported_nonempty_submasks() {
    let radial = test_sub_mask("radial", serde_json::json!({}), SubMaskMode::Additive);
    assert!(!should_generate_sub_mask_in_tiles(&radial, false, 100, 80));
    assert!(should_generate_sub_mask_in_tiles(&radial, true, 100, 80));

    let brush = test_sub_mask("brush", brush_parameters(false), SubMaskMode::Additive);
    assert!(should_generate_sub_mask_in_tiles(&brush, false, 100, 80));

    let mut empty_brush = brush.clone();
    empty_brush.parameters = serde_json::json!({ "lines": [] });
    assert!(!should_generate_sub_mask_in_tiles(
        &empty_brush,
        false,
        100,
        80
    ));

    let mut hidden_brush = brush;
    hidden_brush.visible = false;
    assert!(!should_generate_sub_mask_in_tiles(
        &hidden_brush,
        true,
        100,
        80
    ));

    let color = test_sub_mask("color", serde_json::json!({}), SubMaskMode::Additive);
    assert!(!should_generate_sub_mask_in_tiles(&color, true, 100, 80));

    let mut color = test_sub_mask(
        "color",
        serde_json::json!({
            "targetX": 0.0,
            "targetY": 0.0,
            "grow": 0.0,
            "feather": 0.0
        }),
        SubMaskMode::Additive,
    );
    assert!(!should_generate_sub_mask_in_tiles(
        &color, false, 9_504, 6_336
    ));
    assert!(should_generate_sub_mask_in_tiles(
        &color, true, 9_504, 6_336
    ));
    color.parameters["feather"] = serde_json::json!(35.0);
    assert!(should_generate_sub_mask_in_tiles(
        &color, false, 9_504, 6_336
    ));

    let mut ai_subject = test_sub_mask(
        "ai-subject",
        serde_json::json!({ "grow": 0.0, "feather": 0.0 }),
        SubMaskMode::Additive,
    );
    assert!(!should_generate_sub_mask_in_tiles(
        &ai_subject,
        false,
        9_504,
        6_336
    ));
    assert!(should_generate_sub_mask_in_tiles(
        &ai_subject,
        true,
        9_504,
        6_336
    ));
    ai_subject.parameters["grow"] = serde_json::json!(50.0);
    assert!(should_generate_sub_mask_in_tiles(
        &ai_subject,
        false,
        9_504,
        6_336
    ));

    let ai_depth = test_sub_mask(
        "ai-depth",
        serde_json::json!({ "grow": 0.0, "feather": 15.0 }),
        SubMaskMode::Additive,
    );
    assert!(should_generate_sub_mask_in_tiles(
        &ai_depth, false, 9_504, 6_336
    ));
}
