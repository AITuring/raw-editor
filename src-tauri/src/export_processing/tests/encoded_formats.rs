//! In-memory and BigTIFF encoders.
//!
//! ICC round trips for every color-managed format, the BigTIFF switch and writers,
//! selected PNG/TIFF sample precision and full-size output without a resize.

use super::*;

#[test]
fn bigtiff_budget_switches_before_standard_offset_limit() {
    assert!(!should_use_bigtiff(8_000, 6_000, 3, 2));
    assert!(should_use_bigtiff(1_000_000, 1_000_000, 3, 2));
    assert!(should_use_bigtiff(700_000, 700_000, 4, 2));
}

#[test]
fn bigtiff_encoder_writes_a_decodable_16_bit_rgb_image() {
    let directory = tempdir().expect("temporary BigTIFF directory");
    let path = directory.path().join("sample.bigtiff");
    let mut output = File::create(&path).expect("create BigTIFF");
    encode_rgb16_bigtiff_with_metadata(
        &mut output,
        2,
        2,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        true,
        None,
    )
    .expect("encode BigTIFF");
    drop(output);

    let bytes = std::fs::read(&path).expect("read BigTIFF");
    assert!(
        bytes.get(2..4) == Some(&[43, 0][..]) || bytes.get(2..4) == Some(&[0, 43][..]),
        "expected TIFF version 43 (BigTIFF), header={:?}",
        bytes.get(..16)
    );
    let decoder = TiffDecoder::new(Cursor::new(bytes)).expect("decode BigTIFF");
    assert_eq!(decoder.dimensions(), (2, 2));
}

#[test]
fn streaming_bigtiff_encoder_writes_rgba_rows_as_rgb8() {
    let directory = tempdir().expect("temporary BigTIFF directory");
    let path = directory.path().join("streamed.bigtiff");
    let mut output = File::create(&path).expect("create streamed BigTIFF");
    encode_streaming_bigtiff_rgb8(&mut output, 2, 2, |sink| {
        sink(&[10, 20, 30, 255, 40, 50, 60, 255])?;
        sink(&[70, 80, 90, 255, 100, 110, 120, 255])?;
        Ok(())
    })
    .expect("stream BigTIFF");
    drop(output);

    let bytes = std::fs::read(&path).expect("read streamed BigTIFF");
    assert_eq!(bytes.get(2..4), Some(&[43, 0][..]));
    let decoder = TiffDecoder::new(Cursor::new(bytes)).expect("decode streamed BigTIFF");
    assert_eq!(decoder.dimensions(), (2, 2));
}

#[test]
fn supported_color_managed_exports_round_trip_dimensions_and_icc() {
    let image = fixture_image(37, 23);

    let jpeg = encode_image_to_bytes(&image, "jpeg", 92).expect("encode JPEG");
    assert_decoder_contract(
        JpegDecoder::new(Cursor::new(jpeg)).expect("decode JPEG"),
        (37, 23),
        true,
    );

    let png = encode_image_to_bytes(&image, "png", 100).expect("encode PNG");
    assert_decoder_contract(
        PngDecoder::new(Cursor::new(png)).expect("decode PNG"),
        (37, 23),
        true,
    );

    let tiff = encode_image_to_bytes(&image, "tiff", 100).expect("encode TIFF");
    assert_decoder_contract(
        TiffDecoder::new(Cursor::new(tiff)).expect("decode TIFF"),
        (37, 23),
        true,
    );

    let webp = encode_image_to_bytes(&image, "webp", 90).expect("encode WebP");
    assert_decoder_contract(
        WebPDecoder::new(Cursor::new(webp)).expect("decode WebP"),
        (37, 23),
        true,
    );

    let jxl = encode_image_to_bytes(&image, "jxl", 90).expect("encode JPEG XL");
    assert_decoder_contract(
        JxlDecoder::new(Cursor::new(jxl)).expect("decode JPEG XL"),
        (37, 23),
        false,
    );
}

#[test]
fn export_bit_depth_controls_png_and_tiff_sample_precision() {
    let image = DynamicImage::ImageRgb16(ImageBuffer::from_fn(4, 3, |x, y| {
        Rgb([
            (x * 12_000 + 1_000) as u16,
            (y * 15_000 + 2_000) as u16,
            ((x + y) * 8_000 + 3_000) as u16,
        ])
    }));

    for (output_format, image_format) in [
        ("png", ImageFormat::Png),
        ("tif", ImageFormat::Tiff),
        ("tiff", ImageFormat::Tiff),
    ] {
        for bit_depth in [8_u8, 16_u8] {
            let bytes =
                encode_image_to_bytes_with_profile(&image, output_format, 95, false, bit_depth)
                    .expect("encode selected export bit depth");
            let decoded = image::load_from_memory_with_format(&bytes, image_format)
                .expect("decode selected export bit depth");
            assert_eq!(
                decoded.color(),
                if bit_depth == 16 {
                    image::ColorType::Rgb16
                } else {
                    image::ColorType::Rgb8
                }
            );
        }
    }
}

#[test]
fn no_resize_export_keeps_the_full_input_dimensions() {
    let image = fixture_image(2049, 1367);
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

    let processed =
        apply_export_resize_and_watermark(image, &settings).expect("apply export options");
    assert_eq!(processed.dimensions(), (2049, 1367));

    let bytes = encode_image_to_bytes(&processed, "png", 100).expect("encode full image");
    let decoder = PngDecoder::new(Cursor::new(bytes)).expect("decode full image");
    assert_eq!(decoder.dimensions(), (2049, 1367));
}

#[test]
fn webview_previews_and_exports_match_reference_color_chart_and_icc() {
    use crate::color_management::{
        encode_srgb_jpeg_bytes, encode_srgb_png_bytes, srgb_preview_encoder, srgb_v4_profile,
    };
    use mozjpeg_rs::Preset;

    // 4x2 grid of 16x16 uniform sRGB patches so both lossless and chroma-subsampled
    // lossy codecs have clean interior reference pixels.
    let patches: [[u8; 3]; 8] = [
        [0, 0, 0],       // black
        [24, 24, 24],    // deep shadow toe
        [118, 118, 118], // ~18% linear middle gray
        [255, 255, 255], // white
        [204, 52, 48],   // red patch
        [54, 172, 76],   // green patch
        [48, 104, 212],  // blue patch
        [192, 148, 96],  // warm paper / skin midtone
    ];
    let chart_rgb = RgbImage::from_fn(64, 32, |x, y| {
        let patch_idx = ((y / 16) * 4 + (x / 16)) as usize;
        Rgb(patches[patch_idx])
    });
    let chart = DynamicImage::ImageRgb8(chart_rgb.clone());

    let assert_patch_centers = |label: &str, actual: &RgbImage, max_delta: u8| {
        assert_eq!(actual.dimensions(), (64, 32), "{label}: dimension mismatch");
        for (index, expected) in patches.iter().enumerate() {
            let cx = ((index % 4) as u32) * 16 + 8;
            let cy = ((index / 4) as u32) * 16 + 8;
            let pixel = actual.get_pixel(cx, cy).0;
            for channel in 0..3 {
                let delta = pixel[channel].abs_diff(expected[channel]);
                assert!(
                    delta <= max_delta,
                    "{label} patch {index} channel {channel}: expected {}, got {} (delta {delta} > {max_delta})",
                    expected[channel],
                    pixel[channel]
                );
            }
        }
    };

    // 1. WebView mozjpeg preview
    let mozjpeg_preview = srgb_preview_encoder(Preset::BaselineFastest)
        .quality(95)
        .encode_rgb(chart_rgb.as_raw(), 64, 32)
        .expect("encode WebView mozjpeg preview");
    assert_decoder_contract(
        JpegDecoder::new(Cursor::new(&mozjpeg_preview)).expect("decode WebView mozjpeg"),
        (64, 32),
        true,
    );
    let mozjpeg_decoded = image::load_from_memory_with_format(&mozjpeg_preview, ImageFormat::Jpeg)
        .expect("decode WebView mozjpeg pixels")
        .to_rgb8();
    assert_patch_centers("WebView mozjpeg preview", &mozjpeg_decoded, 4);

    // 2. WebView PNG preview (HDR / legacy panorama / denoise)
    let png_preview = encode_srgb_png_bytes(&chart).expect("encode WebView PNG preview");
    assert_decoder_contract(
        PngDecoder::new(Cursor::new(&png_preview)).expect("decode WebView PNG"),
        (64, 32),
        true,
    );
    let png_preview_decoded = image::load_from_memory_with_format(&png_preview, ImageFormat::Png)
        .expect("decode WebView PNG pixels")
        .to_rgb8();
    assert_patch_centers("WebView PNG preview", &png_preview_decoded, 0);

    // 3. WebView JPEG preview (batch geometry / negative conversion)
    let jpeg_preview = encode_srgb_jpeg_bytes(&chart, 95).expect("encode WebView JPEG preview");
    assert_decoder_contract(
        JpegDecoder::new(Cursor::new(&jpeg_preview)).expect("decode WebView JPEG"),
        (64, 32),
        true,
    );
    let jpeg_preview_decoded =
        image::load_from_memory_with_format(&jpeg_preview, ImageFormat::Jpeg)
            .expect("decode WebView JPEG pixels")
            .to_rgb8();
    assert_patch_centers("WebView JPEG preview", &jpeg_preview_decoded, 4);

    // 4. Lossless export formats (PNG, TIFF, lossless JXL): exact 0 LSB parity + ICC
    for (format_name, image_format) in [("png", ImageFormat::Png), ("tiff", ImageFormat::Tiff)] {
        let bytes = encode_image_to_bytes_with_profile(&chart, format_name, 100, true, 8)
            .expect("encode lossless export");
        let export_icc = match image_format {
            ImageFormat::Tiff => TiffDecoder::new(Cursor::new(&bytes))
                .expect("decode TIFF export")
                .icc_profile()
                .expect("read TIFF export ICC"),
            _ => PngDecoder::new(Cursor::new(&bytes))
                .expect("decode PNG export")
                .icc_profile()
                .expect("read PNG export ICC"),
        };
        assert_eq!(
            export_icc.as_deref(),
            Some(srgb_v4_profile()),
            "{format_name} export must embed sRGB v4 ICC"
        );
        let decoded = image::load_from_memory_with_format(&bytes, image_format)
            .expect("decode lossless export pixels")
            .to_rgb8();
        assert_patch_centers(format_name, &decoded, 0);
    }

    let jxl_bytes = encode_image_to_bytes_with_profile(&chart, "jxl", 100, true, 8)
        .expect("encode lossless JXL export");
    let jxl_decoder = JxlDecoder::new(Cursor::new(&jxl_bytes)).expect("decode lossless JXL");
    let jxl_decoded = DynamicImage::from_decoder(jxl_decoder)
        .expect("read lossless JXL pixels")
        .to_rgb8();
    assert_patch_centers("jxl lossless export", &jxl_decoded, 0);

    // 5. Lossy export formats (JPEG q=100, WebP q=100): ICC + bounded patch-center parity
    let jpeg_export = encode_image_to_bytes_with_profile(&chart, "jpeg", 100, true, 8)
        .expect("encode JPEG export");
    assert_decoder_contract(
        JpegDecoder::new(Cursor::new(&jpeg_export)).expect("decode JPEG export"),
        (64, 32),
        true,
    );
    let jpeg_export_decoded = image::load_from_memory_with_format(&jpeg_export, ImageFormat::Jpeg)
        .expect("decode JPEG export pixels")
        .to_rgb8();
    assert_patch_centers("jpeg export", &jpeg_export_decoded, 4);

    let webp_export = encode_image_to_bytes_with_profile(&chart, "webp", 100, true, 8)
        .expect("encode WebP export");
    assert_decoder_contract(
        WebPDecoder::new(Cursor::new(&webp_export)).expect("decode WebP export"),
        (64, 32),
        true,
    );
    let webp_export_decoded = image::load_from_memory_with_format(&webp_export, ImageFormat::WebP)
        .expect("decode WebP export pixels")
        .to_rgb8();
    assert_patch_centers("webp export", &webp_export_decoded, 4);
}
