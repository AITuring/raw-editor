//! Exact-density source prefilter used before a layer is warped and minified.
use super::*;
use crate::panorama_utils::stitching::minify_rgb;

fn filled(width: u32, height: u32, value: impl Fn(u32, u32) -> f32) -> Rgb32FImage {
    Rgb32FImage::from_fn(width, height, |x, y| {
        let level = value(x, y);
        Rgb([level, level, level])
    })
}

fn row_amplitude(image: &Rgb32FImage, y: u32) -> f32 {
    // Ignore the two border pixels on each side, where the kernel is clipped.
    let values = (2..image.width() - 2)
        .map(|x| image.get_pixel(x, y)[0])
        .collect::<Vec<_>>();
    let max = values.iter().copied().fold(f32::MIN, f32::max);
    let min = values.iter().copied().fold(f32::MAX, f32::min);
    (max - min) * 0.5
}

#[test]
fn minify_keeps_level_and_the_shared_pixel_centre_convention() {
    let divisor = 1.6;
    let constant = minify_rgb(&filled(101, 53, |_, _| 0.4), divisor);
    assert_eq!(constant.dimensions(), (64, 34));
    assert!(
        constant
            .pixels()
            .all(|pixel| pixel.0.iter().all(|value| (value - 0.4).abs() < 1e-5)),
        "a flat source must stay flat"
    );
    let ramp = minify_rgb(&filled(160, 16, |x, _| x as f32 / 160.0), divisor);
    for x in 4..ramp.width() - 4 {
        // Output pixel `x` is centred on source `(x + 0.5) * divisor - 0.5`,
        // the coordinate `LayerSampler` reads it at.
        let expected = ((f64::from(x) + 0.5) * divisor - 0.5) / 160.0;
        let actual = f64::from(ramp.get_pixel(x, 8)[0]);
        assert!(
            (actual - expected).abs() < 2e-3,
            "pixel {x}: {actual:.5} != {expected:.5}"
        );
    }
}

#[test]
fn minify_removes_weave_above_the_new_nyquist_and_keeps_detail_below_it() {
    let divisor = 1.6;
    // A two-pixel weave cannot be represented after minifying by 1.6; a point
    // sample would alias it into a false coarse pattern.
    let weave = minify_rgb(
        &filled(320, 8, |x, _| if x % 2 == 0 { 0.2 } else { 0.8 }),
        divisor,
    );
    assert!(
        row_amplitude(&weave, 4) < 0.08,
        "weave amplitude {:.3} survived the prefilter",
        row_amplitude(&weave, 4)
    );
    // A ten-pixel stroke period is well inside the new Nyquist and must keep
    // most of its contrast.
    let strokes = minify_rgb(
        &filled(320, 8, |x, _| {
            0.5 + 0.3 * (std::f32::consts::TAU * x as f32 / 10.0).sin()
        }),
        divisor,
    );
    assert!(
        row_amplitude(&strokes, 4) > 0.2,
        "stroke amplitude {:.3} was blurred away",
        row_amplitude(&strokes, 4)
    );
}

#[test]
fn telephoto_prefilter_keeps_at_least_one_sample_per_canvas_pixel() {
    let source = filled(400, 300, |x, y| ((x * 7 + y * 3) % 11) as f32 / 11.0);
    // A 120 mm frame placed on a 79 mm canvas is drawn at about 0.66 canvas
    // pixels per source pixel. Halving it left 0.76 samples per canvas pixel.
    let (filtered, divisor) = prefilter_minified_source(source.clone(), 0.66);
    let samples_per_canvas_pixel = 1.0 / (divisor * 0.66);
    assert!(
        (samples_per_canvas_pixel - SOURCE_PREFILTER_OVERSAMPLING).abs() < 1e-9,
        "{samples_per_canvas_pixel}"
    );
    assert_eq!(
        filtered.width(),
        (f64::from(source.width()) / divisor).ceil() as u32
    );
    // A source that is only mildly minified is used as it is.
    let (unchanged, unit) = prefilter_minified_source(source.clone(), 0.9);
    assert_eq!(unit, 1.0);
    assert_eq!(unchanged.as_raw(), source.as_raw());
}
