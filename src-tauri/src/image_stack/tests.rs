use std::fs::{self, File};
use std::io::{BufReader, Cursor};
use std::path::Path;

use image::codecs::{jpeg::JpegDecoder, png::PngDecoder, tiff::TiffDecoder};
use image::{
    ColorType, DynamicImage, GenericImageView, ImageBuffer, ImageDecoder, Rgb, Rgb32FImage,
};

use super::{
    DETAIL_PREVIEW_JPEG_QUALITY, FilterType, JpegEncoder, PREVIEW_JPEG_QUALITY, RgbImage,
    derive_preview_images, preview_file_names,
};
use super::{
    DETAIL_PREVIEW_MAX_LONG_SIDE, DETAIL_PREVIEW_MAX_PIXELS, IMAGE_STACK_MAX_SOURCES,
    IMAGE_STACK_PIPELINE_VERSION, ImageStackOutputFormat, PREVIEW_MAX_LONG_SIDE,
    PREVIEW_MAX_PIXELS, canonicalize_image_stack_result, default_image_stack_export_settings,
    detail_preview_dimensions, encode_srgb_tiff, preview_dimensions,
    resolve_image_stack_output_path, validate_image_stack_pipeline_version,
    validate_image_stack_source_count, write_image_stack_output, write_image_stack_output_for_run,
    write_image_stack_output_with_settings, write_srgb_tiff,
};
use crate::panorama_utils::stack_pipeline::degradation::{
    CLOSURE_UNRELIABLE_RESIDUAL, DegradationLedger, GEOMETRY_DISCONNECTED, RUN_CANCELLED_BY_USER,
};

#[test]
fn image_stack_pipeline_version_rejects_stale_frontends() {
    assert!(validate_image_stack_pipeline_version(IMAGE_STACK_PIPELINE_VERSION).is_ok());
    assert!(validate_image_stack_pipeline_version("").is_err());
    assert!(validate_image_stack_pipeline_version("image-stack-legacy").is_err());
}

#[test]
fn task_7_31_source_count_boundaries_are_exact() {
    let outcomes =
        [0usize, 1, 2, 500, 501].map(|count| (count, validate_image_stack_source_count(count)));
    assert!(outcomes[0].1.is_err(), "zero sources must be rejected");
    assert!(outcomes[1].1.is_err(), "one source must be rejected");
    assert!(outcomes[2].1.is_ok(), "two sources start the valid range");
    assert_eq!(IMAGE_STACK_MAX_SOURCES, 500);
    assert!(outcomes[3].1.is_ok(), "500 sources end the valid range");
    assert!(outcomes[4].1.is_err(), "501 sources exceed the valid range");
}

#[test]
fn preview_dimensions_keep_images_within_the_quality_budget() {
    assert_eq!(preview_dimensions(4_000, 3_000), (4_000, 3_000));

    let (wide_width, wide_height) = preview_dimensions(20_000, 2_000);
    assert_eq!(wide_width, PREVIEW_MAX_LONG_SIDE);
    assert_eq!(wide_height, 410);

    let (large_width, large_height) = preview_dimensions(8_256, 5_504);
    assert!(large_width <= PREVIEW_MAX_LONG_SIDE);
    assert!(large_width as u64 * large_height as u64 <= PREVIEW_MAX_PIXELS + 10_000);

    let (portrait_width, portrait_height) = preview_dimensions(9_305, 12_618);
    assert!(portrait_height <= PREVIEW_MAX_LONG_SIDE);
    assert!(portrait_width as u64 * portrait_height as u64 <= PREVIEW_MAX_PIXELS + 10_000);

    let (detail_width, detail_height) = detail_preview_dimensions(9_305, 12_618);
    assert!(detail_height <= DETAIL_PREVIEW_MAX_LONG_SIDE);
    assert!(detail_width as u64 * detail_height as u64 <= DETAIL_PREVIEW_MAX_PIXELS + 10_000);
    assert!(detail_width > portrait_width);
    assert!(detail_height > portrait_height);
}

/// A painted-looking result: smooth colour fields with brush-scale detail.
fn painted_result(width: u32, height: u32, seed: u64) -> DynamicImage {
    let phase = (seed % 997) as f32 * 0.013;
    let image = image::Rgb32FImage::from_fn(width, height, |x, y| {
        let (fx, fy) = (x as f32, y as f32);
        let broad = 0.45 + 0.25 * (fx * 0.021 + phase).sin() * (fy * 0.017 - phase).cos();
        let detail = 0.04 * ((fx * 0.9 + fy * 0.7 + phase * 5.0).sin());
        image::Rgb([
            (broad + detail).clamp(0.0, 1.0),
            (0.85 * broad + 0.05 + detail).clamp(0.0, 1.0),
            (0.6 * broad + 0.15 - detail).clamp(0.0, 1.0),
        ])
    });
    canonicalize_image_stack_result(DynamicImage::ImageRgb32F(image))
}

fn jpeg_round_trip(image: &RgbImage, quality: u8) -> RgbImage {
    let mut bytes = Vec::new();
    JpegEncoder::new_with_quality(&mut bytes, quality)
        .encode_image(image)
        .expect("in-memory preview JPEG");
    image::load_from_memory(&bytes)
        .expect("decode preview JPEG")
        .to_rgb8()
}

fn roi_mean(image: &RgbImage, (x, y, side): (u32, u32, u32)) -> [f32; 3] {
    let mut sum = [0.0f64; 3];
    for py in y..y + side {
        for px in x..x + side {
            for (channel, value) in image.get_pixel(px, py).0.iter().enumerate() {
                sum[channel] += f64::from(*value) / 255.0;
            }
        }
    }
    let count = f64::from(side * side);
    sum.map(|value| (value / count) as f32)
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig {
        cases: 100,
        ..proptest::prelude::ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 58: 对于任意最终结果及其预览，
    // 两者携带同一个结果标识，预览的每个像素都可由最终结果的规范化显示编码像素降采样得到，
    // 且任意对应 ROI 的低频均值 Delta_E00 不超过 1.0。
    //
    // The previews are produced by `derive_preview_images` from the canonical result
    // alone, encoded at the production JPEG qualities, and named by the result
    // identifier. ROIs are aligned to the 2× and 4× downsampling so the preview ROI is
    // exactly the final ROI's footprint.
    //
    // **Validates: Requirements 10.7**
    #[test]
    fn property_58_previews_derive_from_the_result_with_bounded_roi_delta_e(
        seed in proptest::prelude::any::<u64>(),
        cells in 4u32..10,
        roi_x in 0u32..64,
        roi_y in 0u32..64,
    ) {
        let side = cells * 32;
        let result = painted_result(side, side, seed);
        let detail_dims = (side / 2, side / 2);
        let interaction_dims = (side / 4, side / 4);
        let (detail, interaction) = derive_preview_images(&result, detail_dims, interaction_dims);
        // Derived from the canonical pixels and from nothing else.
        let expected_detail = image::imageops::resize(
            &result.to_rgb8(),
            detail_dims.0,
            detail_dims.1,
            FilterType::Lanczos3,
        );
        proptest::prop_assert_eq!(&detail, &expected_detail);
        let interaction = interaction.expect("a smaller interaction preview");
        let (again_detail, again_interaction) =
            derive_preview_images(&result, detail_dims, interaction_dims);
        proptest::prop_assert_eq!(&detail, &again_detail);
        proptest::prop_assert_eq!(Some(&interaction), again_interaction.as_ref());

        let final_rgb = result.to_rgb8();
        let detail = jpeg_round_trip(&detail, DETAIL_PREVIEW_JPEG_QUALITY);
        let interaction = jpeg_round_trip(&interaction, PREVIEW_JPEG_QUALITY);
        // A 64 px final ROI anywhere on the 4-px lattice.
        let roi_side = 64u32;
        let x = (roi_x * 4).min(side - roi_side);
        let y = (roi_y * 4).min(side - roi_side);
        let reference = roi_mean(&final_rgb, (x, y, roi_side));
        for (preview, factor) in [(&detail, 2u32), (&interaction, 4u32)] {
            let mean = roi_mean(preview, (x / factor, y / factor, roi_side / factor));
            let delta = crate::panorama_utils::stack_pipeline::tone::delta_e00_rgb(reference, mean);
            proptest::prop_assert!(
                delta <= 1.0,
                "{factor}x preview ROI ({x}, {y}) Delta_E00 {delta}"
            );
        }

        // The stored result and both preview files carry one identifier.
        let result_id = format!("{seed:016x}");
        let (detail_name, interaction_name) = preview_file_names(&result_id);
        proptest::prop_assert!(detail_name.starts_with(&result_id));
        proptest::prop_assert!(interaction_name.starts_with(&result_id));
        proptest::prop_assert_ne!(detail_name, interaction_name);
    }
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig {
        cases: 100,
        ..proptest::prelude::ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 57: 对于任意最终结果，以支持
    // alpha 的格式写出时未覆盖像素的 alpha 为完全透明、已覆盖像素的 alpha 为完全不透明，
    // 且已覆盖像素的颜色通道值与不写入 alpha 时逐值相同。
    //
    // The canonical result with its Coverage_Mask is written as 16-bit TIFF and PNG and
    // read back; the colour channels are compared with the opaque canonical form.
    //
    // **Validates: Requirements 10.6**
    #[test]
    fn property_57_alpha_follows_coverage_without_changing_colour(
        seed in proptest::prelude::any::<u64>(),
        width in 4u32..40,
        height in 4u32..40,
    ) {
        let result = painted_result(width, height, seed);
        let coverage = image::GrayImage::from_fn(width, height, |x, y| {
            let hash = (u64::from(x) * 73_856_093) ^ (u64::from(y) * 19_349_663) ^ seed;
            image::Luma([if hash % 5 == 0 { 0 } else { 255 }])
        });
        let opaque = super::canonicalize_image_stack_result(result.clone()).to_rgb16();
        let with_alpha =
            super::canonicalize_image_stack_result_with_coverage(result, Some(&coverage));
        for format in [ImageStackOutputFormat::Tiff, ImageStackOutputFormat::Png] {
            let mut bytes = Cursor::new(Vec::new());
            super::encode_srgb_image_stack(&with_alpha, &mut bytes, format, 95, 16, true, None)
                .expect("encode");
            let decoded = image::load_from_memory(bytes.get_ref()).expect("decode");
            proptest::prop_assert_eq!(decoded.color(), ColorType::Rgba16);
            let decoded = decoded.to_rgba16();
            for (x, y, pixel) in decoded.enumerate_pixels() {
                let covered = coverage.get_pixel(x, y)[0] != 0;
                proptest::prop_assert_eq!(pixel[3], if covered { u16::MAX } else { 0 });
                proptest::prop_assert_eq!(&pixel.0[..3], &opaque.get_pixel(x, y).0[..]);
            }
        }
    }
}

#[test]
fn output_fidelity_reports_bit_depth_and_alpha_downgrades_before_writing() {
    let coverage =
        image::GrayImage::from_fn(8, 8, |x, _| image::Luma([if x < 2 { 0 } else { 255 }]));
    let transparent = super::canonicalize_image_stack_result_with_coverage(
        painted_result(8, 8, 3),
        Some(&coverage),
    );
    let tiff = super::OutputFidelity::of(&transparent, ImageStackOutputFormat::Tiff, 16);
    assert_eq!(
        (tiff.bit_depth, tiff.has_transparency, tiff.alpha_preserved),
        (16, true, true)
    );
    assert!(tiff.downgrades(ImageStackOutputFormat::Tiff).is_empty());

    let jpeg = super::OutputFidelity::of(&transparent, ImageStackOutputFormat::Jpeg, 16);
    let notices = jpeg.downgrades(ImageStackOutputFormat::Jpeg);
    assert_eq!(notices.len(), 2, "{notices:?}");
    assert_eq!(notices[0].id, "output_bit_depth_downgraded");
    assert_eq!(notices[0].params.format, "JPEG");
    assert_eq!(notices[0].params.bit_depth, Some(8));
    assert_eq!(notices[1].id, "output_alpha_unsupported");
    assert_eq!(notices[1].params.format, "JPEG");
    assert_eq!(notices[1].params.bit_depth, None);

    // A fully covered result loses nothing to a missing alpha channel.
    let opaque = super::canonicalize_image_stack_result(painted_result(8, 8, 3));
    let jpeg = super::OutputFidelity::of(&opaque, ImageStackOutputFormat::Jpeg, 16);
    assert_eq!(jpeg.downgrades(ImageStackOutputFormat::Jpeg).len(), 1);
    let png8 = super::OutputFidelity::of(&transparent, ImageStackOutputFormat::Png, 8);
    assert_eq!(png8.downgrades(ImageStackOutputFormat::Png).len(), 1);
}

#[test]
fn a_saved_tiff_with_alpha_validates_and_keeps_its_transparency() {
    let coverage = image::GrayImage::from_fn(16, 12, |x, y| {
        image::Luma([if (x + y) % 7 == 0 { 0 } else { 255 }])
    });
    let image = super::canonicalize_image_stack_result_with_coverage(
        painted_result(16, 12, 11),
        Some(&coverage),
    );
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("stack.tiff");
    write_image_stack_output(&image, &path, ImageStackOutputFormat::Tiff)
        .expect("the RGBA result is written and validated");
    let decoded = image::open(&path).expect("reopen").to_rgba16();
    for (x, y, pixel) in decoded.enumerate_pixels() {
        let covered = coverage.get_pixel(x, y)[0] != 0;
        assert_eq!(pixel[3], if covered { u16::MAX } else { 0 }, "({x}, {y})");
    }
}

#[test]
fn saved_output_metadata_matches_tiff_png_alpha_and_jpeg_files() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let coverage = image::GrayImage::from_fn(12, 9, |x, y| {
        image::Luma([if (x + y) % 4 == 0 { 0 } else { 255 }])
    });
    let with_alpha = super::canonicalize_image_stack_result_with_coverage(
        painted_result(12, 9, 17),
        Some(&coverage),
    );
    let tiff_path = directory.path().join("stack.tiff");
    write_image_stack_output(&with_alpha, &tiff_path, ImageStackOutputFormat::Tiff)
        .expect("write 16-bit TIFF");
    assert_eq!(
        super::read_image_stack_output_metadata(&tiff_path, ImageStackOutputFormat::Tiff)
            .expect("read TIFF metadata"),
        super::SavedOutputMetadata {
            format: "TIFF".to_string(),
            bit_depth: 16,
            alpha_preserved: true,
            icc: "sRGB".to_string(),
        }
    );

    let png_path = directory.path().join("stack.png");
    write_image_stack_output(&with_alpha, &png_path, ImageStackOutputFormat::Png)
        .expect("write 16-bit PNG");
    assert_eq!(
        super::read_image_stack_output_metadata(&png_path, ImageStackOutputFormat::Png)
            .expect("read PNG metadata"),
        super::SavedOutputMetadata {
            format: "PNG".to_string(),
            bit_depth: 16,
            alpha_preserved: true,
            icc: "sRGB".to_string(),
        }
    );

    let opaque = super::canonicalize_image_stack_result(painted_result(12, 9, 17));
    let jpeg_path = directory.path().join("stack.jpg");
    write_image_stack_output(&opaque, &jpeg_path, ImageStackOutputFormat::Jpeg)
        .expect("write JPEG");
    assert_eq!(
        super::read_image_stack_output_metadata(&jpeg_path, ImageStackOutputFormat::Jpeg)
            .expect("read JPEG metadata"),
        super::SavedOutputMetadata {
            format: "JPEG".to_string(),
            bit_depth: 8,
            alpha_preserved: false,
            icc: "sRGB".to_string(),
        }
    );
}

#[test]
fn saved_result_updates_report_for_all_formats_and_failed_save_leaves_it_unchanged() {
    let directory = tempfile::tempdir().expect("temporary image-stack directory");
    let coverage = image::GrayImage::from_fn(10, 8, |x, y| {
        image::Luma([if (x + y) % 3 == 0 { 0 } else { 255 }])
    });
    let transparent = super::canonicalize_image_stack_result_with_coverage(
        painted_result(10, 8, 29),
        Some(&coverage),
    );
    let ledger = DegradationLedger::new();
    let result_id = "result-integrated";
    let recorder = crate::panorama_utils::stack_pipeline::report::StackReportRecorder::isolated(
        "stack-test-integrated-output",
        Some(directory.path().to_path_buf()),
    );
    let report_path = recorder.write_once().expect("write Stack_Report");
    crate::panorama_utils::stack_pipeline::report::assign_stack_report_result_id(
        &report_path,
        result_id,
    )
    .expect("assign result ID");

    for (format, extension, expected_bit_depth, expected_alpha) in [
        (ImageStackOutputFormat::Tiff, "tiff", 16, true),
        (ImageStackOutputFormat::Png, "png", 16, true),
        (ImageStackOutputFormat::Jpeg, "jpg", 8, false),
    ] {
        let output_path = directory.path().join(format!("integrated.{extension}"));
        super::save_image_stack_output_for_result(
            &transparent,
            &output_path,
            format,
            &default_image_stack_export_settings(),
            super::ImageStackReportContext {
                source_path: "",
                degradation_ledger: &ledger,
                result_id,
                stack_report_path: Some(&report_path),
            },
        )
        .expect("publish and report the output");

        let metadata = super::read_image_stack_output_metadata(&output_path, format)
            .expect("read published output metadata");
        assert_eq!(metadata.format, format.label());
        assert_eq!(metadata.bit_depth, expected_bit_depth);
        assert_eq!(metadata.alpha_preserved, expected_alpha);
        assert_eq!(metadata.icc, "sRGB");

        let report: crate::panorama_utils::stack_pipeline::report::StackReport =
            serde_json::from_str(&fs::read_to_string(&report_path).expect("read report"))
                .expect("parse report");
        assert!(report.output.written);
        assert_eq!(report.output.result_id, result_id);
        assert_eq!(report.output.format, format.label());
        assert_eq!(report.output.path, output_path.to_string_lossy());
        assert_eq!(report.output.bit_depth, expected_bit_depth);
        assert_eq!(report.output.alpha_preserved, expected_alpha);
        assert_eq!(report.output.icc, "sRGB");

        match format {
            ImageStackOutputFormat::Tiff | ImageStackOutputFormat::Png => {
                let decoded = image::open(&output_path)
                    .expect("decode alpha output")
                    .to_rgba16();
                let transparent_pixels = decoded.pixels().filter(|pixel| pixel[3] == 0).count();
                let expected_transparent_pixels =
                    coverage.pixels().filter(|pixel| pixel[0] == 0).count();
                assert_eq!(transparent_pixels, expected_transparent_pixels);
            }
            ImageStackOutputFormat::Jpeg => {
                let decoded = image::open(&output_path).expect("decode JPEG");
                assert!(!decoded.color().has_alpha());
            }
        }
    }

    let report_before_failure = fs::read(&report_path).expect("read report before failure");
    let blocked_parent = directory.path().join("blocked-output");
    fs::write(&blocked_parent, b"not a directory").expect("create blocked parent");
    let failed_path = blocked_parent.join("failed.tiff");
    let error = super::save_image_stack_output_for_result(
        &transparent,
        &failed_path,
        ImageStackOutputFormat::Tiff,
        &default_image_stack_export_settings(),
        super::ImageStackReportContext {
            source_path: "",
            degradation_ledger: &ledger,
            result_id,
            stack_report_path: Some(&report_path),
        },
    )
    .expect_err("a failed publication must return an error");
    assert!(!error.is_empty());
    assert_eq!(
        fs::read(&report_path).expect("read report after failure"),
        report_before_failure,
        "failed saves must leave Stack_Report untouched"
    );
}

#[test]
fn canonical_stack_result_is_display_encoded_rgb16() {
    let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(2, 1, Rgb([0.25, 0.5, 0.75])));
    let canonical = canonicalize_image_stack_result(source);
    let pixels = canonical.as_rgb16().expect("canonical RGB16 result");

    assert_eq!(pixels.dimensions(), (2, 1));
    assert!((pixels.get_pixel(0, 0)[0] as i32 - 16_384).abs() <= 1);
    assert!((pixels.get_pixel(0, 0)[1] as i32 - 32_768).abs() <= 1);
    assert!((pixels.get_pixel(0, 0)[2] as i32 - 49_151).abs() <= 1);
}

#[test]
fn exported_tiff_preserves_canonical_pixels_and_srgb_profile() {
    let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(3, 2, |x, y| {
        Rgb([
            (x as f32 + 1.0) / 4.0,
            (y as f32 + 1.0) / 3.0,
            (x as f32 + y as f32 + 1.0) / 6.0,
        ])
    }));
    let canonical = canonicalize_image_stack_result(source);
    let expected = canonical.to_rgb16();
    let mut encoded = Cursor::new(Vec::new());
    encode_srgb_tiff(&canonical, &mut encoded).expect("encode canonical TIFF");

    encoded.set_position(0);
    let mut metadata_decoder = TiffDecoder::new(&mut encoded).expect("decode TIFF metadata");
    let profile = metadata_decoder
        .icc_profile()
        .expect("read TIFF ICC")
        .expect("embedded TIFF ICC");
    assert_eq!(profile, crate::color_management::srgb_v4_profile());

    encoded.set_position(0);
    let decoded =
        DynamicImage::from_decoder(TiffDecoder::new(&mut encoded).expect("decode canonical TIFF"))
            .expect("read canonical TIFF pixels")
            .to_rgb16();
    assert_eq!(decoded, expected);
}

#[test]
fn atomic_tiff_save_replaces_stale_output_with_a_valid_image() {
    let directory = tempfile::tempdir().expect("temporary image-stack output directory");
    let output_path = directory.path().join("stack-result.tiff");
    fs::write(&output_path, b"stale").expect("seed stale output");
    let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(4, 3, |x, y| {
        Rgb([
            (x as f32 + 1.0) / 5.0,
            (y as f32 + 1.0) / 4.0,
            (x as f32 + y as f32 + 1.0) / 8.0,
        ])
    }));
    let canonical = canonicalize_image_stack_result(source);

    write_srgb_tiff(&canonical, &output_path).expect("atomic image-stack TIFF save");
    assert!(
        fs::metadata(&output_path)
            .expect("saved TIFF metadata")
            .len()
            > 5
    );

    let mut decoder = TiffDecoder::new(BufReader::new(
        File::open(&output_path).expect("open persisted image-stack TIFF"),
    ))
    .expect("decode persisted image-stack TIFF");
    assert_eq!(decoder.dimensions(), (4, 3));
    assert_eq!(
        decoder
            .icc_profile()
            .expect("read persisted TIFF ICC")
            .as_deref(),
        Some(crate::color_management::srgb_v4_profile())
    );
}

#[test]
fn rejected_stack_save_preserves_existing_outputs_and_removes_only_its_temp() {
    let directory = tempfile::tempdir().expect("temporary rejected stack directory");
    let output_path = directory.path().join("stack-result.tiff");
    let report_path = directory.path().join("stack-report.json");
    let preview_path = directory.path().join("diagnostic-preview.jpg");
    fs::write(&output_path, b"pre-existing final").expect("seed existing final");
    fs::write(&report_path, b"stack report").expect("seed report");
    fs::write(&preview_path, b"diagnostic preview").expect("seed preview");

    let source = DynamicImage::ImageRgb16(ImageBuffer::from_pixel(4, 3, Rgb([1, 2, 3])));
    let mut ledger = DegradationLedger::new();
    ledger.record(CLOSURE_UNRELIABLE_RESIDUAL, serde_json::Value::Null);
    ledger.record(GEOMETRY_DISCONNECTED, serde_json::Value::Null);

    let error = write_image_stack_output_for_run(
        &source,
        &output_path,
        ImageStackOutputFormat::Tiff,
        &default_image_stack_export_settings(),
        "",
        &ledger,
    )
    .expect_err("a rejected run must not publish its staged result");

    assert!(error.contains("rejected"));
    assert_eq!(
        fs::read(&output_path).expect("read existing final"),
        b"pre-existing final"
    );
    assert_eq!(
        fs::read(&report_path).expect("read report"),
        b"stack report"
    );
    assert_eq!(
        fs::read(&preview_path).expect("read preview"),
        b"diagnostic preview"
    );
    let entries = fs::read_dir(directory.path())
        .expect("list rejected stack directory")
        .map(|entry| entry.expect("directory entry").file_name())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        entries,
        [
            "diagnostic-preview.jpg",
            "stack-report.json",
            "stack-result.tiff"
        ]
        .into_iter()
        .map(std::ffi::OsString::from)
        .collect(),
        "the rejected run must remove only its generated temporary file"
    );
}

#[test]
fn cancelled_stack_save_preserves_existing_final_and_removes_only_its_temp() {
    let directory = tempfile::tempdir().expect("temporary cancelled stack directory");
    let output_path = directory.path().join("stack-result.tiff");
    let user_path = directory.path().join("user-notes.txt");
    fs::write(&output_path, b"pre-existing final").expect("seed existing final");
    fs::write(&user_path, b"keep me").expect("seed user file");

    let source = DynamicImage::ImageRgb16(ImageBuffer::from_pixel(4, 3, Rgb([1, 2, 3])));
    let mut ledger = DegradationLedger::new();
    ledger.record(GEOMETRY_DISCONNECTED, serde_json::Value::Null);
    ledger.record(RUN_CANCELLED_BY_USER, serde_json::Value::Null);

    let error = write_image_stack_output_for_run(
        &source,
        &output_path,
        ImageStackOutputFormat::Tiff,
        &default_image_stack_export_settings(),
        "",
        &ledger,
    )
    .expect_err("a cancelled run must not publish its staged result");

    assert!(error.contains("cancelled"));
    assert_eq!(
        fs::read(&output_path).expect("read existing final"),
        b"pre-existing final"
    );
    assert_eq!(fs::read(&user_path).expect("read user file"), b"keep me");
    let entries = fs::read_dir(directory.path())
        .expect("list cancelled stack directory")
        .map(|entry| entry.expect("directory entry").file_name())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        entries,
        ["stack-result.tiff", "user-notes.txt"]
            .into_iter()
            .map(std::ffi::OsString::from)
            .collect(),
        "the cancelled run must remove only its generated temporary file"
    );
}

#[test]
fn save_path_prefers_the_requested_destination_and_normalizes_the_extension() {
    let source_path = Path::new("/photos/source.jpg");
    assert_eq!(
        resolve_image_stack_output_path(
            source_path,
            "focus",
            Some("/exports/custom-stack"),
            ImageStackOutputFormat::Png,
        )
        .expect("explicit PNG save destination"),
        Path::new("/exports/custom-stack.png")
    );
    assert_eq!(
        resolve_image_stack_output_path(
            source_path,
            "focus",
            Some("/exports/custom-stack.jpeg"),
            ImageStackOutputFormat::Jpeg,
        )
        .expect("explicit JPEG save destination"),
        Path::new("/exports/custom-stack.jpeg")
    );
    assert_eq!(
        resolve_image_stack_output_path(
            source_path,
            "focus",
            Some("/exports/custom-stack.png"),
            ImageStackOutputFormat::Tiff,
        )
        .expect("selected format overrides a mismatched extension"),
        Path::new("/exports/custom-stack.tiff")
    );
    assert_eq!(
        resolve_image_stack_output_path(source_path, "focus", None, ImageStackOutputFormat::Tiff,)
            .expect("fallback TIFF save destination"),
        Path::new("/photos/source_FocusStack.tiff")
    );
}

#[test]
fn output_format_accepts_supported_aliases_and_rejects_other_values() {
    assert_eq!(
        ImageStackOutputFormat::from_wire("tif"),
        Ok(ImageStackOutputFormat::Tiff)
    );
    assert_eq!(
        ImageStackOutputFormat::from_wire("JPEG"),
        Ok(ImageStackOutputFormat::Jpeg)
    );
    assert!(ImageStackOutputFormat::from_wire("webp").is_err());
}

#[test]
fn atomic_stack_save_supports_tiff_png_and_jpeg() {
    let directory = tempfile::tempdir().expect("temporary image-stack output directory");
    let source = DynamicImage::ImageRgb16(ImageBuffer::from_fn(8, 5, |x, y| {
        Rgb([
            ((x + 1) * 4_096) as u16,
            ((y + 1) * 8_192) as u16,
            ((x + y + 1) * 2_048) as u16,
        ])
    }));
    let expected_lossless_pixels = source.to_rgb16();

    for (format, extension) in [
        (ImageStackOutputFormat::Tiff, "tiff"),
        (ImageStackOutputFormat::Png, "png"),
        (ImageStackOutputFormat::Jpeg, "jpg"),
    ] {
        let output_path = directory.path().join(format!("stack-result.{extension}"));
        fs::write(&output_path, b"stale").expect("seed stale output");

        write_image_stack_output(&source, &output_path, format)
            .expect("atomic multi-format image-stack save");

        assert!(
            fs::metadata(&output_path)
                .expect("saved output metadata")
                .len()
                > 5
        );
        let decoded = image::open(&output_path).expect("decode saved multi-format output");
        assert_eq!(decoded.dimensions(), source.dimensions());
        if format != ImageStackOutputFormat::Jpeg {
            assert_eq!(decoded.to_rgb16(), expected_lossless_pixels);
        }
    }
}

#[test]
fn stack_export_respects_selected_bit_depth_for_lossless_formats() {
    let directory = tempfile::tempdir().expect("temporary image-stack output directory");
    let source = DynamicImage::ImageRgb16(ImageBuffer::from_fn(8, 5, |x, y| {
        Rgb([
            ((x + 1) * 4_096) as u16,
            ((y + 1) * 8_192) as u16,
            ((x + y + 1) * 2_048) as u16,
        ])
    }));

    for bit_depth in [8_u8, 16_u8] {
        for (format, extension) in [
            (ImageStackOutputFormat::Tiff, "tiff"),
            (ImageStackOutputFormat::Png, "png"),
        ] {
            let output_path = directory
                .path()
                .join(format!("stack-result-{bit_depth}.{extension}"));
            let mut settings = default_image_stack_export_settings();
            settings.bit_depth = bit_depth;
            write_image_stack_output_with_settings(&source, &output_path, format, &settings, "")
                .expect("save selected image-stack bit depth");

            let actual_color_type = match format {
                ImageStackOutputFormat::Tiff => TiffDecoder::new(BufReader::new(
                    File::open(&output_path).expect("open saved TIFF"),
                ))
                .expect("decode saved TIFF")
                .color_type(),
                ImageStackOutputFormat::Png => PngDecoder::new(BufReader::new(
                    File::open(&output_path).expect("open saved PNG"),
                ))
                .expect("decode saved PNG")
                .color_type(),
                ImageStackOutputFormat::Jpeg => unreachable!(),
            };
            assert_eq!(
                actual_color_type,
                if bit_depth == 16 {
                    ColorType::Rgb16
                } else {
                    ColorType::Rgb8
                }
            );
        }
    }
}

#[test]
fn shared_export_settings_resize_stack_output_and_write_selected_metadata() {
    fn ascii_tag(exif_data: &exif::Exif, tag: exif::Tag) -> Option<String> {
        let field = exif_data.get_field(tag, exif::In::PRIMARY)?;
        let exif::Value::Ascii(values) = &field.value else {
            return None;
        };
        Some(
            values
                .iter()
                .map(|value| {
                    String::from_utf8_lossy(value)
                        .trim_end_matches('\0')
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join(" "),
        )
    }

    let directory = tempfile::tempdir().expect("temporary shared export directory");
    let output_path = directory.path().join("stack-settings.jpg");
    let source = DynamicImage::ImageRgb16(ImageBuffer::from_fn(8, 5, |x, y| {
        Rgb([
            ((x + 1) * 4_096) as u16,
            ((y + 1) * 8_192) as u16,
            ((x + y + 1) * 2_048) as u16,
        ])
    }));
    let settings = crate::export_processing::ExportSettings {
        bit_depth: 8,
        jpeg_quality: 37,
        resize: Some(crate::export_processing::ResizeOptions {
            mode: crate::export_processing::ResizeMode::Width,
            value: 4,
            dont_enlarge: false,
        }),
        keep_metadata: false,
        metadata_overrides: Some(crate::exif_processing::ExportMetadataOverrides {
            artist: Some("Museum Imaging Team".to_string()),
            contact: Some("archive@example.test".to_string()),
            copyright: Some("Copyright 2026".to_string()),
            description: Some("Focus-stacked artifact".to_string()),
        }),
        preserve_timestamps: false,
        strip_gps: true,
        embed_color_profile: false,
        filename_template: None,
        watermark: None,
        export_masks: false,
        preserve_folders: false,
    };

    write_image_stack_output_with_settings(
        &source,
        &output_path,
        ImageStackOutputFormat::Jpeg,
        &settings,
        "/missing/source.jpg",
    )
    .expect("save stack through shared export settings");

    let encoded = fs::read(&output_path).expect("read shared export output");
    let mut decoder = JpegDecoder::new(Cursor::new(&encoded)).expect("decode resized stack JPEG");
    assert_eq!(decoder.dimensions(), (4, 3));
    assert_eq!(
        decoder.icc_profile().expect("read optional ICC profile"),
        None
    );

    let exif_data = exif::Reader::new()
        .read_from_container(&mut Cursor::new(&encoded))
        .expect("read selected stack metadata");
    assert_eq!(
        ascii_tag(&exif_data, exif::Tag::Artist).as_deref(),
        Some("Museum Imaging Team")
    );
    assert_eq!(
        ascii_tag(&exif_data, exif::Tag::Copyright).as_deref(),
        Some("Copyright 2026")
    );
    assert_eq!(
        ascii_tag(&exif_data, exif::Tag::ImageDescription).as_deref(),
        Some("Focus-stacked artifact")
    );
    let contact = exif_data
        .get_field(exif::Tag::UserComment, exif::In::PRIMARY)
        .and_then(|field| match &field.value {
            exif::Value::Undefined(value, _) => Some(String::from_utf8_lossy(value).to_string()),
            _ => None,
        });
    assert_eq!(contact.as_deref(), Some("Contact: archive@example.test"));
}

#[test]
#[ignore = "requires RAW_EDITOR_STACK_EXPORT_SOURCE and RAW_EDITOR_STACK_EXPORT_OUTPUT_DIR"]
fn real_stack_export_fixture_from_env() {
    let source_path = std::env::var_os("RAW_EDITOR_STACK_EXPORT_SOURCE")
        .map(std::path::PathBuf::from)
        .expect("RAW_EDITOR_STACK_EXPORT_SOURCE must point to a rendered stack image");
    let output_dir = std::env::var_os("RAW_EDITOR_STACK_EXPORT_OUTPUT_DIR")
        .map(std::path::PathBuf::from)
        .expect("RAW_EDITOR_STACK_EXPORT_OUTPUT_DIR must point to a writable directory");
    fs::create_dir_all(&output_dir).expect("create real image-stack export output directory");

    let mut reader = image::ImageReader::open(&source_path)
        .expect("open real image-stack export fixture")
        .with_guessed_format()
        .expect("detect real image-stack export fixture format");
    reader.no_limits();
    let decoded = reader
        .decode()
        .expect("decode real image-stack export fixture");
    let canonical = if decoded.as_rgb16().is_some() {
        decoded
    } else {
        canonicalize_image_stack_result(decoded)
    };

    for (format, extension) in [
        (ImageStackOutputFormat::Tiff, "tiff"),
        (ImageStackOutputFormat::Png, "png"),
        (ImageStackOutputFormat::Jpeg, "jpg"),
    ] {
        let output_path = output_dir.join(format!("real-stack-export.{extension}"));
        write_image_stack_output(&canonical, &output_path, format)
            .expect("export real image-stack fixture");
        println!("{}", output_path.display());
    }
}
