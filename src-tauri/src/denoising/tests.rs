use super::*;

fn mean_squared_error(left: &Rgb32FImage, right: &Rgb32FImage) -> f32 {
    left.pixels()
        .zip(right.pixels())
        .flat_map(|(left, right)| {
            (0..3).map(move |channel| {
                let difference = left[channel] - right[channel];
                difference * difference
            })
        })
        .sum::<f32>()
        / (left.width() * left.height() * 3) as f32
}

#[test]
fn zero_intensity_is_pixel_exact() {
    let source = Rgb32FImage::from_fn(17, 13, |x, y| {
        let value = ((x * 17 + y * 11) % 97) as f32 / 96.0;
        Rgb([value, value * 0.8, value * 0.6])
    });
    let output = run_bm3d_core(&source, 0.0, None).to_rgb32f();
    assert_eq!(source.as_raw(), output.as_raw());
}

#[test]
fn deterministic_bm3d_reduces_noise_without_destroying_primary_edge() {
    let clean = Rgb32FImage::from_fn(48, 48, |x, _| {
        let value = if x < 24 { 0.25 } else { 0.75 };
        Rgb([value, value, value])
    });
    let noisy = Rgb32FImage::from_fn(48, 48, |x, y| {
        let clean_value = if x < 24 { 0.25 } else { 0.75 };
        let hash = x
            .wrapping_mul(1_103_515_245)
            .wrapping_add(y.wrapping_mul(12_345))
            .rotate_left((y % 31) + 1);
        let noise = (hash % 10_001) as f32 / 10_000.0 * 0.16 - 0.08;
        let value = (clean_value + noise).clamp(0.0, 1.0);
        Rgb([value, value, value])
    });

    let denoised = run_bm3d_core(&noisy, 0.25, None).to_rgb32f();
    assert!(
        mean_squared_error(&denoised, &clean) < mean_squared_error(&noisy, &clean),
        "denoise output should be closer to the deterministic clean fixture"
    );
    assert!(
        denoised
            .pixels()
            .flat_map(|pixel| pixel.0)
            .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    );

    let left_mean = denoised
        .enumerate_pixels()
        .filter(|(x, _, _)| *x < 20)
        .map(|(_, _, pixel)| pixel[0])
        .sum::<f32>()
        / (20 * 48) as f32;
    let right_mean = denoised
        .enumerate_pixels()
        .filter(|(x, _, _)| *x >= 28)
        .map(|(_, _, pixel)| pixel[0])
        .sum::<f32>()
        / (20 * 48) as f32;
    assert!(right_mean - left_mean > 0.35);
}

#[test]
fn higher_intensity_produces_a_materially_stronger_result() {
    let source = Rgb32FImage::from_fn(48, 48, |x, y| {
        let base = if x < 24 { 0.28 } else { 0.72 };
        let hash = x
            .wrapping_mul(747_796_405)
            .wrapping_add(y.wrapping_mul(2_891_336_453))
            .rotate_left((x + y) % 31 + 1);
        let noise = (hash % 10_001) as f32 / 10_000.0 * 0.18 - 0.09;
        let red = (base + noise).clamp(0.0, 1.0);
        let green = (base * 0.92 + noise * 0.75).clamp(0.0, 1.0);
        let blue = (base * 0.78 + noise * 0.55).clamp(0.0, 1.0);
        Rgb([red, green, blue])
    });

    let low = run_bm3d_core(&source, 0.15, None).to_rgb32f();
    let high = run_bm3d_core(&source, 0.85, None).to_rgb32f();
    let low_change = mean_squared_error(&low, &source);
    let high_change = mean_squared_error(&high, &source);
    let result_separation = mean_squared_error(&low, &high);

    assert!(
        high_change > low_change * 1.2,
        "high strength should alter the source more (low={low_change}, high={high_change})"
    );
    assert!(
        result_separation > 0.000_01,
        "15 and 85 strength should produce visibly distinct pixels (mse={result_separation})"
    );
}
