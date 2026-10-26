// Quality_Gate module fragment; included at parent module scope.

#[cfg(test)]
mod gradient_tests {
    use super::*;
    use image::{Rgb, Rgb32FImage};

    #[test]
    fn cached_gradient_matches_the_production_acutance_primitive() {
        let mut image = Rgb32FImage::new(17, 19);
        for (i, pixel) in image.pixels_mut().enumerate() {
            let value = ((i * 37 % 251) as f32) / 251.0;
            *pixel = Rgb([value, value * 0.93, value * 0.87]);
        }
        let plane = metric_luminance(&image).unwrap();
        let width = image.width() as usize;
        let height = image.height() as usize;
        let expected = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| {
                crate::panorama_utils::mosaic::acutance_with_step(
                    |sx, sy| {
                        let nx = (sx.round() as isize).clamp(0, width as isize - 1) as usize;
                        let ny = (sy.round() as isize).clamp(0, height as isize - 1) as usize;
                        Some(Rgb([plane[ny * width + nx] as f32; 3]))
                    },
                    x as f64,
                    y as f64,
                    1.0,
                )
            })
            .sum::<f64>()
            / plane.len() as f64;
        let actual = normalized_gradient_energy(&image, 1.0).unwrap();
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }
}
