//! Row-streaming export.
//!
//! Which exports may stream, encoder parity and the JPEG quality scale, complete-frame
//! checks, streamed resize and the permissions of the temporary output.

use super::*;

#[test]
fn geometry_output_streaming_supports_orthogonal_post_transforms_only() {
    let adjustments = serde_json::json!({
        "transformVertical": 18.0,
        "crop": { "unit": "px", "x": 4.0, "y": 3.0, "width": 80.0, "height": 60.0 }
    });
    let params = get_geometry_params_from_json(&adjustments);
    assert!(can_stream_geometry_output(&adjustments, &params));

    for compatible in [
        serde_json::json!({ "transformVertical": 18.0, "orientationSteps": 1 }),
        serde_json::json!({ "transformVertical": 18.0, "orientationSteps": 2, "flipHorizontal": true }),
        serde_json::json!({ "transformVertical": 18.0, "orientationSteps": 3, "flipVertical": true }),
        serde_json::json!({ "transformVertical": 18.0, "flipHorizontal": true, "flipVertical": true }),
    ] {
        let params = get_geometry_params_from_json(&compatible);
        assert!(can_stream_geometry_output(&compatible, &params));
    }

    for incompatible in [
        serde_json::json!({ "transformVertical": 18.0, "rotation": 0.5 }),
        serde_json::json!({ "transformVertical": 18.0, "lensBlurEnabled": true }),
    ] {
        let params = get_geometry_params_from_json(&incompatible);
        assert!(!can_stream_geometry_output(&incompatible, &params));
    }

    let color_mask: MaskDefinition = serde_json::from_value(serde_json::json!({
        "id": "color-mask",
        "name": "Color",
        "visible": true,
        "invert": false,
        "opacity": 100.0,
        "adjustments": null,
        "subMasks": [{
            "id": "color",
            "type": "color",
            "visible": true,
            "invert": false,
            "opacity": 100.0,
            "mode": "additive",
            "parameters": {}
        }]
    }))
    .expect("color mask fixture");
    assert!(color_mask.requires_range_source());
    assert!(can_stream_geometry_output(&adjustments, &params));
}

fn encode_streaming_fixture(
    image: &DynamicImage,
    output_format: &str,
    jpeg_quality: u8,
) -> Result<Vec<u8>, String> {
    encode_streaming_fixture_with_metadata(
        image,
        output_format,
        jpeg_quality,
        StreamingExportMetadata::None,
    )
}

fn mean_rgb_abs_error(reference: &RgbImage, candidate: &RgbImage) -> f64 {
    assert_eq!(reference.dimensions(), candidate.dimensions());
    let total_delta: u64 = reference
        .pixels()
        .zip(candidate.pixels())
        .map(|(reference, candidate)| {
            (0..3)
                .map(|channel| u64::from(reference[channel].abs_diff(candidate[channel])))
                .sum::<u64>()
        })
        .sum();
    total_delta as f64 / (reference.width() as f64 * reference.height() as f64 * 3.0)
}

#[test]
fn streaming_color_managed_exports_round_trip_dimensions_and_icc() {
    let image = fixture_image(37, 23);
    let expected_rgba = image.to_rgba8();

    let jpeg = encode_streaming_fixture(&image, "jpeg", 92).expect("stream JPEG");
    assert_decoder_contract(
        JpegDecoder::new(Cursor::new(&jpeg)).expect("decode streaming JPEG"),
        (37, 23),
        true,
    );

    let png = encode_streaming_fixture(&image, "png", 92).expect("stream PNG");
    assert_decoder_contract(
        PngDecoder::new(Cursor::new(&png)).expect("decode streaming PNG"),
        (37, 23),
        true,
    );
    let decoded_png = image::load_from_memory_with_format(&png, ImageFormat::Png)
        .expect("load streaming PNG pixels")
        .to_rgba8();
    assert_eq!(decoded_png, expected_rgba);

    let tiff = encode_streaming_fixture(&image, "tiff", 92).expect("stream TIFF");
    assert_decoder_contract(
        TiffDecoder::new(Cursor::new(&tiff)).expect("decode streaming TIFF"),
        (37, 23),
        true,
    );
    let decoded_tiff = image::load_from_memory_with_format(&tiff, ImageFormat::Tiff)
        .expect("load streaming TIFF pixels")
        .to_rgba8();
    assert_eq!(decoded_tiff, expected_rgba);
}

#[test]
fn streaming_jpeg_preserves_the_existing_quality_scale() {
    let image = fixture_image(257, 173);
    let reference = image.to_rgb8();
    let mut previous_streamed_error = f64::INFINITY;
    let mut previous_streamed_bytes = 0_usize;

    for quality in [50, 75, 92] {
        let legacy = encode_image_to_bytes(&image, "jpeg", quality)
            .unwrap_or_else(|error| panic!("encode legacy JPEG Q{quality}: {error}"));
        let streamed = encode_streaming_fixture(&image, "jpeg", quality)
            .unwrap_or_else(|error| panic!("stream JPEG Q{quality}: {error}"));
        let legacy_decoded = image::load_from_memory_with_format(&legacy, ImageFormat::Jpeg)
            .unwrap_or_else(|error| panic!("decode legacy JPEG Q{quality}: {error}"))
            .to_rgb8();
        let streamed_decoded = image::load_from_memory_with_format(&streamed, ImageFormat::Jpeg)
            .unwrap_or_else(|error| panic!("decode streaming JPEG Q{quality}: {error}"))
            .to_rgb8();
        let legacy_error = mean_rgb_abs_error(&reference, &legacy_decoded);
        let streamed_error = mean_rgb_abs_error(&reference, &streamed_decoded);
        let size_delta = streamed.len().abs_diff(legacy.len());
        let allowed_size_delta = (legacy.len() / 20).max(512);

        assert!(
            streamed_error <= legacy_error * 1.05 + 0.15,
            "streamed JPEG Q{quality} mean RGB error {streamed_error:.3} drifted from legacy {legacy_error:.3}"
        );
        assert!(
            size_delta <= allowed_size_delta,
            "streamed JPEG Q{quality} size {} drifted from legacy {}",
            streamed.len(),
            legacy.len()
        );
        assert!(
            streamed_error < previous_streamed_error,
            "streamed JPEG error did not improve at Q{quality}"
        );
        assert!(
            streamed.len() > previous_streamed_bytes,
            "streamed JPEG size did not increase at Q{quality}"
        );
        previous_streamed_error = streamed_error;
        previous_streamed_bytes = streamed.len();
    }
}

#[test]
fn streaming_export_requires_exactly_one_complete_frame() {
    let mut file = tempfile::tempfile().expect("temporary output");
    let error = encode_streaming_rgba_rows(
        &mut file,
        4,
        3,
        "png",
        90,
        8,
        true,
        StreamingExportMetadata::None,
        |sink| {
            sink(&[0; 4 * 4])?;
            sink(&[0; 4 * 4])?;
            Ok(())
        },
    )
    .expect_err("missing final row must fail");
    assert!(error.contains("expected 3"), "unexpected error: {error}");

    let mut overflow_file = tempfile::tempfile().expect("temporary overflow output");
    let error = encode_streaming_rgba_rows(
        &mut overflow_file,
        4,
        1,
        "png",
        90,
        8,
        true,
        StreamingExportMetadata::None,
        |sink| {
            sink(&[0; 4 * 4])?;
            sink(&[0; 4 * 4])?;
            Ok(())
        },
    )
    .expect_err("extra row must fail");
    assert!(
        error.contains("more than the declared 1 rows"),
        "unexpected error: {error}"
    );

    let mut oversized_jpeg = tempfile::tempfile().expect("temporary oversized JPEG");
    let error = encode_streaming_rgba_rows(
        &mut oversized_jpeg,
        u32::from(u16::MAX) + 1,
        1,
        "jpeg",
        90,
        8,
        true,
        StreamingExportMetadata::None,
        |_| Ok(()),
    )
    .expect_err("oversized JPEG dimensions must fail before encoding");
    assert!(error.contains("65535"), "unexpected error: {error}");

    let settings = ExportSettings {
        bit_depth: 8,
        jpeg_quality: 90,
        resize: None,
        keep_metadata: false,
        metadata_overrides: None,
        preserve_timestamps: false,
        strip_gps: false,
        embed_color_profile: true,
        filename_template: None,
        watermark: None,
        export_masks: false,
        preserve_folders: false,
    };
    assert!(supports_streaming_export("jpeg", &settings));
    assert!(supports_streaming_export("png", &settings));
    assert!(supports_streaming_export("tiff", &settings));
    assert!(!supports_streaming_export("webp", &settings));

    let mut resized = settings.clone();
    resized.resize = Some(ResizeOptions {
        mode: ResizeMode::LongEdge,
        value: 2_048,
        dont_enlarge: true,
    });
    assert!(supports_streaming_export("png", &resized));

    let mut watermarked = resized;
    watermarked.watermark = Some(WatermarkSettings {
        path: "unused-routing-fixture.png".to_string(),
        anchor: WatermarkAnchor::BottomRight,
        scale: 10.0,
        spacing: 5.0,
        opacity: 50.0,
    });
    assert!(supports_streaming_export("jpeg", &watermarked));
}

#[test]
fn streaming_resize_matches_the_bounded_batch_reference() {
    let image = fixture_image(37, 23);
    let settings = ExportSettings {
        bit_depth: 8,
        jpeg_quality: 92,
        resize: Some(ResizeOptions {
            mode: ResizeMode::Width,
            value: 19,
            dont_enlarge: false,
        }),
        keep_metadata: false,
        metadata_overrides: None,
        preserve_timestamps: false,
        strip_gps: false,
        embed_color_profile: true,
        filename_template: None,
        watermark: None,
        export_masks: false,
        preserve_folders: false,
    };
    let streamed = collect_streaming_transform(&image, &settings).expect("stream resize");
    let source = image.to_rgba8();
    let (target_width, target_height) = streamed.dimensions();
    let config =
        ResizeConfig::builder(source.width(), source.height(), target_width, target_height)
            .filter(StreamingResizeFilter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build();
    let expected = zenresize::Resizer::new(&config).resize(source.as_raw());

    assert_eq!(streamed.as_raw(), &expected);
    assert_eq!(streamed.dimensions(), (19, 12));

    let legacy = image
        .resize(19, 12, imageops::FilterType::Lanczos3)
        .to_rgba8();
    let mut max_channel_delta = 0_u8;
    let mut total_channel_delta = 0_u64;
    for (streamed, legacy) in streamed.pixels().zip(legacy.pixels()) {
        for channel in 0..3 {
            let delta = streamed[channel].abs_diff(legacy[channel]);
            max_channel_delta = max_channel_delta.max(delta);
            total_channel_delta += u64::from(delta);
        }
    }
    let mean_channel_delta = total_channel_delta as f64 / (19 * 12 * 3) as f64;
    assert!(
        mean_channel_delta <= 0.5,
        "mean channel delta {mean_channel_delta:.3} exceeded the migration bound"
    );
    assert!(
        max_channel_delta <= 2,
        "max channel delta {max_channel_delta} exceeded the migration bound"
    );
}

#[cfg(unix)]
#[test]
fn streamed_export_temp_preserves_existing_file_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().expect("temporary export directory");
    let output_path = directory.path().join("existing.png");
    fs::write(&output_path, b"old").expect("create existing export");
    fs::set_permissions(&output_path, fs::Permissions::from_mode(0o640))
        .expect("set existing permissions");

    let temporary = create_temporary_export(&output_path).expect("create adjacent temporary");
    preserve_existing_export_permissions(&temporary, &output_path)
        .expect("preserve existing permissions");
    let mode = temporary
        .as_file()
        .metadata()
        .expect("temporary metadata")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o640);
}
