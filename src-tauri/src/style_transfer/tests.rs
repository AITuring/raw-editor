use std::path::Path;

use super::{
    StyleTransferMode, TransferTransform, apply_transform, downscale_preview_to_edge,
    estimate_transform, hsv_to_rgb, rgb_to_hsv,
};
use image::{DynamicImage, Rgb, Rgb32FImage, RgbImage};

#[test]
fn floating_point_preview_downscale_preserves_midtone_levels() {
    let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(4, 4, Rgb([0.25, 0.5, 0.75])));

    let preview = downscale_preview_to_edge(&source, 2);

    assert_eq!(preview.dimensions(), (2, 2));
    for pixel in preview.pixels() {
        for (actual, expected) in pixel.0.into_iter().zip([64_i16, 128, 191]) {
            assert!((i16::from(actual) - expected).abs() <= 1);
        }
    }
}

#[cfg(not(target_os = "android"))]
use super::{PREVIEW_EDGE, decode_jpeg_preview, jpeg_info, stream_jpeg_export};
#[cfg(not(target_os = "android"))]
use image::{GenericImageView, ImageFormat};

#[test]
fn hsv_round_trip_preserves_primary_colours() {
    for rgb in [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 90, 40]] {
        let (hue, saturation, value) = rgb_to_hsv(
            rgb[0] as f32 / 255.0,
            rgb[1] as f32 / 255.0,
            rgb[2] as f32 / 255.0,
        );
        let result = hsv_to_rgb(hue, saturation, value);
        for (expected, actual) in rgb.into_iter().zip(result) {
            assert!((expected as f32 / 255.0 - actual).abs() < 0.01);
        }
    }
}

#[test]
fn zero_strength_is_non_destructive() {
    let image = RgbImage::from_pixel(2, 2, Rgb([10, 20, 30]));
    let transform = TransferTransform {
        hue_shift: 0.2,
        saturation_scale: 0.4,
        value_offset: 0.2,
        value_contrast: 1.2,
        target_value_mean: 0.2,
        channel_scale: [1.4, 0.7, 1.3],
        channel_offset: [0.1, -0.1, 0.0],
    };
    let result = apply_transform(image.clone(), transform, StyleTransferMode::Mood, 0.0);
    assert_eq!(result, image);
}

fn mean_horizontal_gradient(image: &RgbImage) -> f32 {
    let mut total = 0.0_f32;
    let mut count = 0_u32;
    for y in 0..image.height() {
        for x in 1..image.width() {
            let left = image.get_pixel(x - 1, y).0[0] as f32;
            let right = image.get_pixel(x, y).0[0] as f32;
            total += (right - left).abs();
            count += 1;
        }
    }
    total / count.max(1) as f32
}

#[test]
fn tonal_transfer_preserves_natural_detail_contrast() {
    let target = RgbImage::from_fn(64, 64, |x, y| {
        let value = if (x + y) % 2 == 0 { 96 } else { 112 };
        Rgb([value, value, value])
    });
    let reference = RgbImage::from_fn(64, 64, |x, y| {
        let value = if (x + y) % 2 == 0 { 32 } else { 224 };
        Rgb([value, value, value])
    });
    let transform = estimate_transform(&reference, &target);

    assert!(transform.value_contrast <= 1.2);
    assert!(transform.channel_scale.iter().all(|scale| *scale <= 1.23));

    for mode in [StyleTransferMode::Mood, StyleTransferMode::Distribution] {
        let styled = apply_transform(target.clone(), transform, mode, 1.0);
        let gradient_ratio = mean_horizontal_gradient(&styled) / mean_horizontal_gradient(&target);
        assert!(
            gradient_ratio <= 1.25,
            "{mode:?} amplified detail contrast by {gradient_ratio:.3}x"
        );
    }
}

#[cfg(not(target_os = "android"))]
#[test]
fn scaled_jpeg_preview_uses_the_bounded_decoder_path() {
    let mut image = RgbImage::new(3_200, 2_100);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        *pixel = Rgb([(x % 256) as u8, (y % 256) as u8, 128]);
    }
    let temporary = tempfile::Builder::new()
        .suffix(".jpg")
        .tempfile()
        .expect("create JPEG fixture");
    image
        .save_with_format(temporary.path(), ImageFormat::Jpeg)
        .expect("write JPEG fixture");

    let info = jpeg_info(temporary.path()).expect("inspect JPEG fixture");
    assert_eq!((info.width, info.height), (3_200, 2_100));
    let preview = decode_jpeg_preview(temporary.path(), info).expect("decode JPEG preview");
    assert!(preview.width().max(preview.height()) <= PREVIEW_EDGE);
    assert!(preview.width() > 0 && preview.height() > 0);
}

#[cfg(not(target_os = "android"))]
#[test]
fn streaming_export_writes_a_valid_full_size_jpeg() {
    let reference = tempfile::Builder::new()
        .suffix(".jpg")
        .tempfile()
        .expect("create reference fixture");
    let target = tempfile::Builder::new()
        .suffix(".jpg")
        .tempfile()
        .expect("create target fixture");
    let output = tempfile::Builder::new()
        .suffix(".jpg")
        .tempfile()
        .expect("create output fixture");

    let reference_image = RgbImage::from_fn(48, 32, |x, y| Rgb([180, (x * 3) as u8, y as u8]));
    let target_image = RgbImage::from_fn(48, 32, |x, y| Rgb([(x * 2) as u8, 80, (y * 4) as u8]));
    reference_image
        .save_with_format(reference.path(), ImageFormat::Jpeg)
        .expect("write reference fixture");
    target_image
        .save_with_format(target.path(), ImageFormat::Jpeg)
        .expect("write target fixture");

    let app = tauri::test::mock_app();
    let settings = crate::app_settings::AppSettings::default();
    let saved = stream_jpeg_export(
        reference.path().to_str().expect("reference path"),
        target.path().to_str().expect("target path"),
        output.path().to_str().expect("output path"),
        0.8,
        StyleTransferMode::Mood,
        &settings,
        &app.handle().clone(),
    )
    .expect("stream styled JPEG");
    assert_eq!(Path::new(&saved), output.path());

    let decoded = image::open(output.path()).expect("decode styled JPEG");
    assert_eq!(decoded.dimensions(), target_image.dimensions());
}
