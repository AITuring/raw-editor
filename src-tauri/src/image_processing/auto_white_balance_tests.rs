use super::*;
use image::Rgb;

fn corrected_spread(pixel: Rgb<u8>, temperature: f64, tint: f64) -> f64 {
    let (red_gain, green_gain, blue_gain) = white_balance_multipliers(temperature, tint);
    let corrected = [
        pixel[0] as f64 * red_gain,
        pixel[1] as f64 * green_gain,
        pixel[2] as f64 * blue_gain,
    ];
    corrected.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - corrected.iter().copied().fold(f64::INFINITY, f64::min)
}

#[test]
fn neutral_image_keeps_neutral_white_balance() {
    let image = RgbImage::from_pixel(32, 32, Rgb([128, 128, 128]));
    assert_eq!(estimate_auto_white_balance(&image), (0.0, 0.0));
}

#[test]
fn warm_cast_is_cooled_toward_neutral() {
    let cast = Rgb([190, 135, 85]);
    let image = RgbImage::from_pixel(32, 32, cast);
    let (temperature, tint) = estimate_auto_white_balance(&image);
    let original_spread = (cast[0] - cast[2]) as f64;

    assert!(temperature < -10.0);
    assert!(corrected_spread(cast, temperature, tint) < original_spread * 0.35);
}

#[test]
fn green_cast_adds_magenta_tint() {
    let cast = Rgb([110, 170, 110]);
    let image = RgbImage::from_pixel(32, 32, cast);
    let (temperature, tint) = estimate_auto_white_balance(&image);
    let original_spread = (cast[1] - cast[0]) as f64;

    assert!(tint > 10.0);
    assert!(corrected_spread(cast, temperature, tint) < original_spread * 0.35);
}
