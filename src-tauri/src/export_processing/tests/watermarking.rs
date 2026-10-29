//! Watermarks in exports.
//!
//! Imported watermark normalisation and deduplication, the bundled default asset
//! and streamed blending against the full-frame reference.

use super::*;

#[test]
fn imported_watermark_is_normalized_and_deduplicated_in_private_storage() {
    let directory = tempfile::tempdir().expect("watermark import directory");
    let source_path = directory.path().join("My Watermark.png");
    let source = RgbaImage::from_fn(11, 7, |x, y| {
        Rgba([
            (x * 17) as u8,
            (y * 29) as u8,
            ((x + y) * 13) as u8,
            (80 + x * 7 + y * 3) as u8,
        ])
    });
    source.save(&source_path).expect("save source watermark");

    let watermark_directory = directory.path().join("stored-watermarks");
    let first =
        import_watermark_image_impl(&source_path, &watermark_directory).expect("import watermark");
    let second = import_watermark_image_impl(&source_path, &watermark_directory)
        .expect("deduplicate watermark");

    assert_eq!(first, second);
    assert_eq!(first.parent(), Some(watermark_directory.as_path()));
    assert!(
        first
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("My-Watermark-") && name.ends_with(".png"))
    );
    assert_eq!(
        image::image_dimensions(first).expect("stored dimensions"),
        (11, 7)
    );
}

#[test]
fn bundled_default_watermark_matches_the_reference_asset() {
    let watermark =
        load_watermark_image(DEFAULT_WATERMARK_PATH).expect("decode the bundled default watermark");
    assert_eq!(watermark.dimensions(), (1908, 1462));

    let prepared = prepare_watermark(
        1000,
        800,
        &WatermarkSettings {
            path: DEFAULT_WATERMARK_PATH.to_string(),
            anchor: WatermarkAnchor::Center,
            scale: 10.0,
            spacing: 5.0,
            opacity: 80.0,
        },
    )
    .expect("prepare the bundled default watermark")
    .expect("default watermark remains visible");

    assert_eq!(prepared.image.width(), 80);
    assert_eq!(prepared.x, 460);
    assert_eq!(prepared.y, (800 - i64::from(prepared.image.height())) / 2);
}

#[test]
fn streaming_watermark_matches_the_existing_full_frame_blend() {
    let image = fixture_image(37, 23);
    let directory = tempfile::tempdir().expect("watermark directory");
    let watermark_path = directory.path().join("watermark.png");
    let watermark = RgbaImage::from_fn(7, 3, |x, y| {
        Rgba([
            (x * 31) as u8,
            (y * 67) as u8,
            ((x + y) * 23) as u8,
            (80 + x * 17 + y * 11) as u8,
        ])
    });
    watermark.save(&watermark_path).expect("save watermark");
    let settings = ExportSettings {
        bit_depth: 8,
        jpeg_quality: 92,
        resize: None,
        keep_metadata: false,
        metadata_overrides: None,
        preserve_timestamps: false,
        strip_gps: false,
        embed_color_profile: true,
        filename_template: None,
        watermark: Some(WatermarkSettings {
            path: watermark_path.to_string_lossy().into_owned(),
            anchor: WatermarkAnchor::BottomRight,
            scale: 24.0,
            spacing: 7.0,
            opacity: 63.0,
        }),
        export_masks: false,
        preserve_folders: false,
    };

    let streamed = collect_streaming_transform(&image, &settings).expect("stream watermark");
    let expected = apply_export_resize_and_watermark(image, &settings)
        .expect("apply existing watermark")
        .to_rgba8();
    assert_eq!(streamed, expected);
}
