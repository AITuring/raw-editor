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
