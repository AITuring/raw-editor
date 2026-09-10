//! Photometrically invariant patch registration in a common geometric frame.
//!
//! A descriptor identifies a neighbourhood, not a subpixel correspondence.
//! Warping the patch support before refinement matters for a lens switch just
//! as much as warping its centre. No image resampling is committed here.
use image::GrayImage;
use nalgebra::{Matrix3, Point2, Point3};

pub(crate) fn sample_gray(image: &GrayImage, x: f64, y: f64) -> Option<f64> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= image.width().saturating_sub(1) as f64
        || y >= image.height().saturating_sub(1) as f64
    {
        return None;
    }
    let ix = x as u32;
    let iy = y as u32;
    let fx = x - ix as f64;
    let fy = y - iy as f64;
    let a = f64::from(image.get_pixel(ix, iy)[0]);
    let b = f64::from(image.get_pixel(ix + 1, iy)[0]);
    let c = f64::from(image.get_pixel(ix, iy + 1)[0]);
    let d = f64::from(image.get_pixel(ix + 1, iy + 1)[0]);
    Some((a + fx * (b - a)) * (1.0 - fy) + (c + fx * (d - c)) * fy)
}

fn project(h: &Matrix3<f64>, p: Point2<f64>) -> Option<Point2<f64>> {
    let q = h * Point3::new(p.x, p.y, 1.0);
    (q.z.abs() > 1e-8).then(|| Point2::new(q.x / q.z, q.y / q.z))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PatchMatch {
    pub(crate) target: Point2<f64>,
    pub(crate) correlation: f64,
}

/// Search a small residual translation after applying the complete geometric
/// transform to every patch sample. Correlation removes gain and black-level
/// differences. The second-peak and boundary checks reject repeated textures
/// and motions outside the capture range instead of inventing a correction.
pub(crate) fn refine_warped_patch(
    source: &GrayImage,
    target: &GrayImage,
    transform: &Matrix3<f64>,
    center: Point2<f64>,
    radius: i32,
    search: i32,
) -> Option<PatchMatch> {
    let predicted = project(transform, center)?;
    let horizontal = project(transform, center + nalgebra::Vector2::new(1.0, 0.0))?;
    let vertical = project(transform, center + nalgebra::Vector2::new(0.0, 1.0))?;
    let scale = ((horizontal - predicted).norm() * (vertical - predicted).norm()).sqrt();
    if !(0.15..6.0).contains(&scale) {
        return None;
    }
    // At least one target pixel between samples when the candidate is smaller.
    let spacing = scale.recip().max(1.0);
    let mut samples = Vec::with_capacity(((2 * radius + 1).pow(2)) as usize);
    let mut sum = 0.0;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let p = center + nalgebra::Vector2::new(dx as f64 * spacing, dy as f64 * spacing);
            let value = sample_gray(source, p.x, p.y)?;
            let q = project(transform, p)?;
            // Reserve the whole search support before comparing hypotheses.
            sample_gray(target, q.x - search as f64, q.y - search as f64)?;
            sample_gray(target, q.x + search as f64, q.y + search as f64)?;
            samples.push((q, value));
            sum += value;
        }
    }
    let count = samples.len() as f64;
    let mean = sum / count;
    let mut variance = 0.0;
    for (_, value) in &mut samples {
        *value -= mean;
        variance += *value * *value;
    }
    if variance / count < 2.0 {
        return None;
    }
    let score = |dx: f64, dy: f64| -> f64 {
        let mut sum = 0.0;
        let mut squares = 0.0;
        let mut product = 0.0;
        for (q, value) in &samples {
            let Some(t) = sample_gray(target, q.x + dx, q.y + dy) else {
                return -1.0;
            };
            sum += t;
            squares += t * t;
            product += value * t;
        }
        let target_variance = squares - sum * sum / count;
        if target_variance / count < 1.0 {
            return -1.0;
        }
        (product / (variance * target_variance).sqrt()).clamp(-1.0, 1.0)
    };
    let mut best = (0.0, 0.0, -1.0);
    let mut candidates = Vec::new();
    for dy in -search..=search {
        for dx in -search..=search {
            let correlation = score(dx as f64, dy as f64);
            candidates.push((dx, dy, correlation));
            if correlation > best.2 {
                best = (dx as f64, dy as f64, correlation);
            }
        }
    }
    if best.2 < 0.80 || best.0.abs() >= search as f64 || best.1.abs() >= search as f64 {
        return None;
    }
    let alternate = candidates
        .iter()
        .filter(|(x, y, _)| (*x as f64 - best.0).hypot(*y as f64 - best.1) >= 2.5)
        .map(|(_, _, s)| *s)
        .fold(-1.0f64, f64::max);
    if best.2 - alternate < 0.006 {
        return None;
    }
    for step in [0.5, 0.25, 0.125, 0.0625] {
        let previous = best;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let x = previous.0 + dx as f64 * step;
                let y = previous.1 + dy as f64 * step;
                let correlation = score(x, y);
                if correlation > best.2 {
                    best = (x, y, correlation);
                }
            }
        }
    }
    Some(PatchMatch {
        target: predicted + nalgebra::Vector2::new(best.0, best.1),
        correlation: best.2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn texture(x: f64, y: f64) -> f64 {
        115.0
            + 31.0 * (x * 0.21 + y * 0.07).sin()
            + 24.0 * (y * 0.31 - x * 0.04).cos()
            + 16.0 * (x * 0.53 + y * 0.41).sin()
    }

    #[test]
    fn lens_scale_and_exposure_are_refined_in_the_same_patch_frame() {
        let source = GrayImage::from_fn(100, 90, |x, y| Luma([texture(x as f64, y as f64) as u8]));
        let actual = Matrix3::new(2.4, 0.018, 9.37, -0.012, 2.4, 12.63, 0.0002, 0.0001, 1.0);
        let inverse = actual.try_inverse().unwrap();
        let target = GrayImage::from_fn(260, 240, |x, y| {
            let p = project(&inverse, Point2::new(x as f64, y as f64)).unwrap();
            Luma([(texture(p.x, p.y) * 0.81 + 19.0).round() as u8])
        });
        let mut initial = actual;
        initial[(0, 2)] += 1.6;
        initial[(1, 2)] -= 1.1;
        for center in [Point2::new(27.0, 30.0), Point2::new(66.0, 61.0)] {
            let found = refine_warped_patch(&source, &target, &initial, center, 7, 4).unwrap();
            assert!(
                (found.target - project(&actual, center).unwrap()).norm() < 0.15,
                "{found:?}"
            );
            assert!(found.correlation > 0.99);
        }
    }

    #[test]
    fn flat_regions_do_not_create_a_registration_constraint() {
        let flat = GrayImage::from_pixel(80, 80, Luma([120]));
        assert!(
            refine_warped_patch(
                &flat,
                &flat,
                &Matrix3::identity(),
                Point2::new(40.0, 40.0),
                7,
                4
            )
            .is_none()
        );
    }
}
