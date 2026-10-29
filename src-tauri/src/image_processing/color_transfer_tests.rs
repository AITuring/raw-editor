use super::{apply_linear_to_srgb, apply_srgb_to_linear};
use image::{DynamicImage, ImageBuffer, Rgba};

#[test]
fn float_rgba_transfer_round_trips_rgb_and_preserves_alpha() {
    let original = Rgba([0.0, 0.040_45, 0.5, 0.37]);
    let encoded = DynamicImage::ImageRgba32F(ImageBuffer::from_pixel(1, 1, original));

    let linear = apply_srgb_to_linear(encoded);
    let linear_pixel = *linear
        .as_rgba32f()
        .expect("sRGB decode keeps RGBA32F storage")
        .get_pixel(0, 0);
    assert!((linear_pixel[2] - 0.214_041_14).abs() <= 1.0e-7);
    assert_eq!(linear_pixel[3], original[3]);

    let round_trip = apply_linear_to_srgb(linear);
    let round_trip_pixel = round_trip
        .as_rgba32f()
        .expect("sRGB encode keeps RGBA32F storage")
        .get_pixel(0, 0);
    for channel in 0..3 {
        assert!((round_trip_pixel[channel] - original[channel]).abs() <= 2.0e-6);
    }
    assert_eq!(round_trip_pixel[3], original[3]);
}
