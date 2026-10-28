use super::*;

fn rgb_fixture(width: u32, height: u32) -> Rgb32FImage {
    Rgb32FImage::from_fn(width, height, |x, y| {
        image::Rgb([
            (x * 7 + y * 3) as f32 / 128.0,
            (x * 5 + y * 11) as f32 / 192.0,
            (x * 13 + y * 2) as f32 / 224.0,
        ])
    })
}

#[test]
fn rgb32f_raw_enhancement_reuses_pixels_and_matches_converted_input() {
    let rgb = rgb_fixture(13, 11);
    let mut reused = DynamicImage::ImageRgb32F(rgb.clone());
    let original_allocation = reused.as_rgb32f().expect("RGB fixture").as_raw().as_ptr();
    let mut converted = DynamicImage::ImageRgba32F(DynamicImage::ImageRgb32F(rgb).to_rgba32f());

    remove_raw_artifacts_and_enhance(&mut reused, 2.0, 0.35);
    remove_raw_artifacts_and_enhance(&mut converted, 2.0, 0.35);

    let reused = reused.as_rgb32f().expect("enhanced RGB output");
    assert_eq!(reused.as_raw().as_ptr(), original_allocation);
    assert_eq!(
        reused.as_raw(),
        converted
            .as_rgb32f()
            .expect("converted enhancement output")
            .as_raw()
    );
}

#[test]
fn rgb32f_default_raw_display_transform_reuses_pixels() {
    let rgb = rgb_fixture(13, 11);
    let mut reused = DynamicImage::ImageRgb32F(rgb.clone());
    let original_allocation = reused.as_rgb32f().expect("RGB fixture").as_raw().as_ptr();
    let mut converted = DynamicImage::ImageRgba32F(DynamicImage::ImageRgb32F(rgb).to_rgba32f());

    apply_cpu_default_raw_processing(&mut reused);
    apply_cpu_default_raw_processing(&mut converted);

    let reused = reused.as_rgb32f().expect("display RGB output");
    assert_eq!(reused.as_raw().as_ptr(), original_allocation);
    assert_eq!(
        reused.as_raw(),
        converted
            .as_rgb32f()
            .expect("converted display output")
            .as_raw()
    );
}
