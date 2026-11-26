use std::io::Cursor;

use image::codecs::{jpeg::JpegEncoder, png::PngEncoder, tiff::TiffEncoder};
use image::{DynamicImage, ExtendedColorType, ImageBuffer, ImageEncoder, ImageReader, Rgb};
use moxcms::ColorProfile;

use super::{load_image_with_orientation, load_psd_image, read_embedded_icc};
use crate::color_management::{MAX_EMBEDDED_ICC_BYTES, linear_to_srgb_channel, srgb_v4_profile};

fn encode_rgb_png(pixel: [u8; 3], profile: Option<Vec<u8>>) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = PngEncoder::new(&mut bytes);
    if let Some(profile) = profile {
        encoder
            .set_icc_profile(profile)
            .expect("attach deterministic test profile");
    }
    encoder
        .write_image(&pixel, 1, 1, ExtendedColorType::Rgb8)
        .expect("encode deterministic PNG");
    bytes
}

fn encode_rgb_jpeg(pixel: [u8; 3], profile: Vec<u8>) -> Vec<u8> {
    let pixels = pixel.repeat(16);
    let mut bytes = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut bytes, 100);
    encoder
        .set_icc_profile(profile)
        .expect("attach deterministic test profile");
    encoder
        .write_image(&pixels, 4, 4, ExtendedColorType::Rgb8)
        .expect("encode deterministic JPEG");
    bytes
}

fn encode_rgb_tiff(pixel: [u8; 3], profile: Vec<u8>) -> Vec<u8> {
    let image = ImageBuffer::<Rgb<u16>, Vec<u16>>::from_pixel(
        37,
        23,
        Rgb([
            u16::from(pixel[0]) * 257,
            u16::from(pixel[1]) * 257,
            u16::from(pixel[2]) * 257,
        ]),
    );
    let mut bytes = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    let mut encoder = TiffEncoder::new(&mut cursor);
    encoder
        .set_icc_profile(profile)
        .expect("attach deterministic test profile");
    DynamicImage::ImageRgb16(image)
        .write_with_encoder(encoder)
        .expect("encode deterministic TIFF");
    bytes
}

fn decoded_pixel(bytes: &[u8]) -> [f32; 3] {
    let decoded = load_image_with_orientation(bytes, None)
        .expect("decode test image")
        .into_rgb32f();
    decoded.get_pixel(0, 0).0
}

fn embedded_icc(bytes: &[u8]) -> Option<Vec<u8>> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .expect("guess test image format");
    let format = reader.format();
    let mut decoder = reader
        .into_decoder()
        .expect("initialize test image decoder");
    read_embedded_icc(bytes, format, &mut decoder).expect("read test image ICC")
}

fn assert_pixel_close(label: &str, actual: [f32; 3], expected: [f32; 3], tolerance: f32) {
    for channel in 0..3 {
        assert!(
            (actual[channel] - expected[channel]).abs() <= tolerance,
            "{label} channel {channel}: expected {}, got {}",
            expected[channel],
            actual[channel]
        );
    }
}

#[test]
fn jpeg_png_and_tiff_display_p3_profiles_are_applied_during_decode() {
    let mut display_p3 = ColorProfile::new_display_p3();
    display_p3.description = None;
    display_p3.copyright = None;
    display_p3.cicp = None;
    let profile = display_p3.encode().expect("encode Display P3 profile");
    assert_eq!(profile.len() % 4, 0, "test ICC must be 4-byte aligned");
    let png = encode_rgb_png([128, 64, 191], Some(profile.clone()));
    let jpeg = encode_rgb_jpeg([128, 64, 191], profile.clone());
    let tiff = encode_rgb_tiff([128, 64, 191], profile);
    let srgb_tiff = encode_rgb_tiff([128, 64, 191], srgb_v4_profile().to_vec());
    assert_eq!(embedded_icc(&srgb_tiff).as_deref(), Some(srgb_v4_profile()));

    for (label, encoded) in [("PNG", &png), ("JPEG", &jpeg), ("TIFF", &tiff)] {
        assert!(
            embedded_icc(encoded).is_some(),
            "{label} must expose its ICC profile"
        );
    }

    for (label, encoded) in [("PNG", &png), ("JPEG", &jpeg), ("TIFF", &tiff)] {
        assert_pixel_close(label, decoded_pixel(encoded), [0.539, 0.233, 0.776], 0.008);
    }
}

#[test]
fn untagged_or_malformed_profile_falls_back_to_srgb_assumption() {
    let expected = [128.0 / 255.0, 64.0 / 255.0, 191.0 / 255.0];
    let untagged = encode_rgb_png([128, 64, 191], None);
    let malformed = encode_rgb_png([128, 64, 191], Some(b"not an ICC profile".to_vec()));

    assert_pixel_close("untagged PNG", decoded_pixel(&untagged), expected, 1.0e-6);
    assert_pixel_close("malformed PNG", decoded_pixel(&malformed), expected, 1.0e-6);
}

fn png_crc32(chunk_type: &[u8; 4], payload: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in chunk_type.iter().chain(payload.iter()) {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn insert_png_chunk_before_idat(png: &[u8], chunk_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut offset = 8usize;
    while offset + 12 <= png.len() {
        let len = u32::from_be_bytes(png[offset..offset + 4].try_into().unwrap()) as usize;
        if &png[offset + 4..offset + 8] == b"IDAT" {
            let mut out = Vec::with_capacity(png.len() + 12 + payload.len());
            out.extend_from_slice(&png[..offset]);
            out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
            out.extend_from_slice(chunk_type);
            out.extend_from_slice(payload);
            out.extend_from_slice(&png_crc32(chunk_type, payload).to_be_bytes());
            out.extend_from_slice(&png[offset..]);
            return out;
        }
        offset += 12 + len;
    }
    panic!("IDAT chunk not found in test PNG");
}

#[test]
fn grayscale_png_with_gray_gamma_icc_normalizes_to_srgb() {
    let gray_profile = ColorProfile::new_gray_with_gamma(1.8)
        .encode()
        .expect("encode Gray Gamma 1.8 profile");
    let mut bytes = Vec::new();
    let mut encoder = PngEncoder::new(&mut bytes);
    encoder
        .set_icc_profile(gray_profile)
        .expect("attach Gray ICC profile");
    encoder
        .write_image(&[128u8], 1, 1, ExtendedColorType::L8)
        .expect("encode grayscale PNG");

    let expected_channel = linear_to_srgb_channel((128.0_f32 / 255.0).powf(1.8));
    assert_pixel_close(
        "Gray Gamma 1.8 PNG",
        decoded_pixel(&bytes),
        [expected_channel; 3],
        0.008,
    );
}

#[test]
fn cmyk_and_oversized_icc_profiles_fall_back_safely_during_decode() {
    let expected = [128.0 / 255.0, 64.0 / 255.0, 191.0 / 255.0];

    let mut cmyk_icc = vec![0u8; 128];
    cmyk_icc[0..4].copy_from_slice(&128u32.to_be_bytes());
    cmyk_icc[16..20].copy_from_slice(b"CMYK");
    cmyk_icc[36..40].copy_from_slice(b"acsp");
    let cmyk_png = encode_rgb_png([128, 64, 191], Some(cmyk_icc));
    assert_pixel_close(
        "CMYK-tagged PNG",
        decoded_pixel(&cmyk_png),
        expected,
        1.0e-6,
    );

    let mut oversized_icc = vec![0u8; MAX_EMBEDDED_ICC_BYTES + 4];
    oversized_icc[0..4].copy_from_slice(&((MAX_EMBEDDED_ICC_BYTES + 4) as u32).to_be_bytes());
    oversized_icc[16..20].copy_from_slice(b"RGB ");
    oversized_icc[36..40].copy_from_slice(b"acsp");
    let oversized_png = encode_rgb_png([128, 64, 191], Some(oversized_icc));
    assert_pixel_close(
        "oversized-ICC PNG",
        decoded_pixel(&oversized_png),
        expected,
        1.0e-6,
    );
}

#[test]
fn cicp_only_display_p3_png_normalizes_and_hdr_cicp_falls_back_safely() {
    let base_png = encode_rgb_png([128, 64, 191], None);
    let p3_cicp_png = insert_png_chunk_before_idat(&base_png, b"cICP", &[12, 13, 0, 1]);
    assert!(embedded_icc(&p3_cicp_png).is_none());
    assert_pixel_close(
        "CICP-only Display P3 PNG",
        decoded_pixel(&p3_cicp_png),
        [0.539, 0.233, 0.776],
        0.008,
    );

    let pq_hdr_png = insert_png_chunk_before_idat(&base_png, b"cICP", &[9, 16, 0, 1]);
    let expected = [128.0 / 255.0, 64.0 / 255.0, 191.0 / 255.0];
    assert_pixel_close(
        "CICP-only HDR PQ PNG fallback",
        decoded_pixel(&pq_hdr_png),
        expected,
        1.0e-6,
    );
}

#[test]
fn configured_psd_fixture_decodes_as_one_source_image() {
    let Some(path) = std::env::var_os("RAW_EDITOR_PSD_FIXTURE") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    let bytes = std::fs::read(&path).expect("read PSD fixture");
    let image = load_psd_image(&bytes, None).expect("decode PSD fixture");
    assert!(image.width() > 0);
    assert!(image.height() > 0);
    assert!(matches!(
        image,
        DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_)
    ));
}
