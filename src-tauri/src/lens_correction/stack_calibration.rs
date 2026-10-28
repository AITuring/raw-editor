//! Calibrated lens geometry for focus-stack registration.
//!
//! Exact camera/lens Lensfun matches yield PTLens distortion and PA vignetting
//! without the editor's strength multiplier; the footprint tells which canvas
//! pixels still lie on the original sensor.

use super::{CalibrationElement, LENS_DB_DIR, LensDatabase, makers_match};

/// Calibrated PTLens geometry for registration, without the editor's artistic
/// strength multiplier. The radius unit is half the original short side.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StackLensCalibration {
    pub(super) a: f64,
    pub(super) b: f64,
    pub(super) c: f64,
    pub(super) vignetting: Option<[f64; 3]>,
}

/// Geometric support of a calibrated RAW canvas; black sensor pixels remain
/// covered, while padding outside the original sensor footprint does not.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StackLensFootprint {
    pub(crate) calibration: StackLensCalibration,
    pub(crate) original: (u32, u32),
    pub(crate) corrected: (u32, u32),
}

impl StackLensFootprint {
    pub(crate) fn contains(self, x: f64, y: f64, rendered: (u32, u32)) -> bool {
        if rendered.0 == 0 || rendered.1 == 0 {
            return false;
        }
        let x = (x + 0.5) * f64::from(self.corrected.0) / f64::from(rendered.0) - 0.5;
        let y = (y + 0.5) * f64::from(self.corrected.1) / f64::from(rendered.1) - 0.5;
        let (cx, cy) = (
            (f64::from(self.original.0) - 1.0) * 0.5,
            (f64::from(self.original.1) - 1.0) * 0.5,
        );
        let dx = x - (f64::from(self.corrected.0) - 1.0) * 0.5;
        let dy = y - (f64::from(self.corrected.1) - 1.0) * 0.5;
        let radius = dx.hypot(dy) / (f64::from(self.original.0.min(self.original.1)) * 0.5);
        let ratio = if radius < 1e-12 {
            1.0
        } else {
            self.calibration.distorted_radius(radius) / radius
        };
        let (sx, sy) = (cx + dx * ratio, cy + dy * ratio);
        sx.is_finite()
            && sy.is_finite()
            && sx >= 2.0
            && sy >= 2.0
            && sx < f64::from(self.original.0) - 2.0
            && sy < f64::from(self.original.1) - 2.0
    }
}

impl StackLensCalibration {
    /// PA lens shading is measured on the pristine, linear sensor image.
    /// Unlike distortion, its radius unit is half the original diagonal.
    pub(crate) fn correct_linear_illumination(self, image: &mut image::Rgb32FImage) {
        use rayon::prelude::*;
        let Some(k) = self.vignetting else {
            return;
        };
        let (w, h) = image.dimensions();
        if w < 2 || h < 2 {
            return;
        }
        let (cx, cy) = ((f64::from(w) - 1.0) * 0.5, (f64::from(h) - 1.0) * 0.5);
        let unit_sq = (f64::from(w).powi(2) + f64::from(h).powi(2)) * 0.25;
        let falloff = |r2: f64| 1.0 + k[0] * r2 + k[1] * r2 * r2 + k[2] * r2 * r2 * r2;
        if (0..=1024).any(|i| {
            let v = falloff(f64::from(i) / 1024.0);
            !v.is_finite() || v < 0.05
        }) {
            return;
        }
        image
            .as_mut()
            .par_chunks_mut(w as usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
                    let r2 = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)) / unit_sq;
                    let gain = (1.0 / falloff(r2)) as f32;
                    for channel in pixel {
                        *channel = (*channel * gain).max(0.0);
                    }
                }
            });
    }

    pub(super) fn distorted_radius(self, radius: f64) -> f64 {
        radius
            * (self.a * radius.powi(3) + self.b * radius.powi(2) + self.c * radius + 1.0
                - self.a
                - self.b
                - self.c)
    }

    pub(super) fn undistorted_radius(self, distorted: f64) -> Option<f64> {
        if distorted == 0.0 {
            return Some(0.0);
        }
        let mut low = 0.0;
        let mut high = distorted * 1.25;
        if self.distorted_radius(high) < distorted {
            return None;
        }
        for _ in 0..48 {
            let middle = (low + high) * 0.5;
            if self.distorted_radius(middle) < distorted {
                low = middle;
            } else {
                high = middle;
            }
        }
        Some((low + high) * 0.5)
    }

    /// Expand the corrected canvas to contain every original boundary sample.
    /// Black outside the physical footprint is the stack renderer's existing
    /// unsupported-pixel representation; it must never become image evidence.
    pub(crate) fn correct(self, source: &image::Rgb32FImage) -> Option<image::Rgb32FImage> {
        use rayon::prelude::*;
        let (width, height) = source.dimensions();
        if width < 4 || height < 4 {
            return None;
        }
        let (cx, cy) = (
            (f64::from(width) - 1.0) * 0.5,
            (f64::from(height) - 1.0) * 0.5,
        );
        let unit = f64::from(width.min(height)) * 0.5;
        let corner_radius = cx.hypot(cy) / unit;
        let outer = self.undistorted_radius(corner_radius)?;
        // Reject folding or extreme profiles instead of manufacturing support.
        for step in 0..=1024 {
            let r = outer * f64::from(step) / 1024.0;
            let derivative =
                4.0 * self.a * r.powi(3) + 3.0 * self.b * r.powi(2) + 2.0 * self.c * r + 1.0
                    - self.a
                    - self.b
                    - self.c;
            if !derivative.is_finite() || derivative <= 0.1 {
                return None;
            }
        }
        let mut extent = (cx, cy);
        for (dx, dy) in (0..width)
            .map(|x| (f64::from(x) - cx, cy))
            .chain((0..height).map(|y| (cx, f64::from(y) - cy)))
        {
            let radius = dx.hypot(dy) / unit;
            let ratio = self.undistorted_radius(radius)? / radius;
            extent.0 = extent.0.max(dx.abs() * ratio);
            extent.1 = extent.1.max(dy.abs() * ratio);
        }
        let out_width = (extent.0 * 2.0).ceil() as u32 + 3;
        let out_height = (extent.1 * 2.0).ceil() as u32 + 3;
        if out_width > width * 5 / 4 || out_height > height * 5 / 4 {
            return None;
        }
        let out_cx = (f64::from(out_width) - 1.0) * 0.5;
        let out_cy = (f64::from(out_height) - 1.0) * 0.5;
        let mut pixels = vec![0.0f32; out_width as usize * out_height as usize * 3];
        pixels
            .par_chunks_mut(out_width as usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
                    let (dx, dy) = (x as f64 - out_cx, y as f64 - out_cy);
                    let radius = dx.hypot(dy) / unit;
                    let ratio = if radius < 1e-12 {
                        1.0
                    } else {
                        self.distorted_radius(radius) / radius
                    };
                    let (sx, sy) = (cx + dx * ratio, cy + dy * ratio);
                    if sx >= 0.0
                        && sy >= 0.0
                        && sx <= f64::from(width - 1)
                        && sy <= f64::from(height - 1)
                    {
                        let sampled =
                            crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                                source, sx, sy,
                            );
                        for channel in 0..3 {
                            pixel[channel] = sampled[channel].clamp(0.0, 1.0);
                        }
                    }
                }
            });
        image::Rgb32FImage::from_raw(out_width, out_height, pixels)
    }
}

/// Only exact camera/lens identities and a matching calibrated sensor format
/// are eligible. A fuzzy lens suggestion is unsafe as registration geometry.
pub(crate) fn stack_lens_calibration(
    metadata: &std::collections::HashMap<String, String>,
    dimensions: (u32, u32),
) -> Option<StackLensCalibration> {
    use std::sync::OnceLock;
    static DATABASE: OnceLock<LensDatabase> = OnceLock::new();
    let database = DATABASE.get_or_init(|| {
        let mut db = LensDatabase {
            cameras: Vec::new(),
            lenses: Vec::new(),
        };
        for file in LENS_DB_DIR.files() {
            if let Some(xml) = file.contents_utf8()
                && let Ok(mut part) = quick_xml::de::from_str::<LensDatabase>(xml)
            {
                db.cameras.append(&mut part.cameras);
                db.lenses.append(&mut part.lenses);
            }
        }
        db
    });
    let identity = |name: &str| {
        name.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    let maker = metadata.get("Make")?;
    let camera_model = identity(metadata.get("Model")?);
    let lens_model = identity(metadata.get("LensModel")?);
    let focal: f32 = metadata.get("FocalLength")?.parse().ok()?;
    if !focal.is_finite() || focal <= 0.0 {
        return None;
    }
    let camera = database.cameras.iter().find(|camera| {
        camera
            .maker
            .iter()
            .any(|name| makers_match(maker, &name.value))
            && camera.model.iter().any(|name| {
                let candidate = identity(&name.value);
                candidate == camera_model
                    || candidate == format!("{}{}", identity(maker), camera_model)
            })
    })?;
    // The camera may be operating in a sensor crop mode. Its model alone is
    // insufficient: require the actual 35mm-equivalent EXIF focal length.
    let equivalent: f32 = metadata.get("FocalLengthIn35mmFilm")?.parse().ok()?;
    if !equivalent.is_finite() || (equivalent / focal - camera.cropfactor).abs() > 0.05 {
        return None;
    }
    let lens = database.lenses.iter().find(|lens| {
        lens.mount.contains(&camera.mount)
            && lens
                .model
                .iter()
                .any(|name| identity(&name.value) == lens_model)
            && lens
                .type_
                .as_deref()
                .is_none_or(|kind| kind == "rectilinear")
            && lens.center.is_none()
            && lens
                .cropfactor
                .is_some_and(|crop| (crop - camera.cropfactor).abs() < 0.01)
    })?;
    let aspect = lens.aspect_ratio.as_deref().unwrap_or("1.5");
    let calibrated_aspect = if let Some((a, b)) = aspect.split_once(':') {
        a.parse::<f64>().ok()? / b.parse::<f64>().ok()?
    } else {
        aspect.parse::<f64>().ok()?
    };
    let actual_aspect =
        f64::from(dimensions.0.max(dimensions.1)) / f64::from(dimensions.0.min(dimensions.1));
    if (actual_aspect - calibrated_aspect).abs() > 0.01 {
        return None;
    }
    let distortion = lens
        .calibration
        .as_ref()?
        .elements
        .iter()
        .filter_map(|element| {
            if let CalibrationElement::Distortion(distortion) = element {
                Some(distortion)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let (min, max) = (
        distortion.iter().map(|d| d.focal).min_by(f32::total_cmp)?,
        distortion.iter().map(|d| d.focal).max_by(f32::total_cmp)?,
    );
    if focal < min || focal > max || distortion.iter().any(|d| d.model != "ptlens") {
        return None;
    }
    let parameters = lens.get_distortion_params(focal, None, None)?;
    if parameters.k1.abs() + parameters.k2.abs() + parameters.k3.abs() < 1e-12 {
        return None;
    }
    // Lensfun PA interpolation uses inverse-distance weighting (power 3.5)
    // in normalized focal length, 4/aperture and 0.1/focus-distance space.
    let vignetting = metadata
        .get("FNumber")
        .and_then(|number| number.trim_start_matches("f/").parse::<f64>().ok())
        .filter(|aperture| aperture.is_finite() && *aperture > 0.0)
        .and_then(|aperture| {
            let distance = metadata
                .get("FocusDistance")
                .and_then(|d| d.parse::<f64>().ok())
                .filter(|d| d.is_finite() && *d > 0.0)
                .unwrap_or(1000.0);
            let mut sum = [0.0; 3];
            let mut total = 0.0;
            let mut nearest = f64::INFINITY;
            let range = f64::from(max - min).max(1.0);
            for element in &lens.calibration.as_ref()?.elements {
                let CalibrationElement::Vignetting(v) = element else {
                    continue;
                };
                if v.model != "pa" || v.aperture <= 0.0 {
                    continue;
                }
                let fd = (f64::from(v.focal) - f64::from(focal)) / range;
                let ad = 4.0 / f64::from(v.aperture) - 4.0 / aperture;
                let dd = 0.1 / f64::from(v.distance.unwrap_or(1000.0)) - 0.1 / distance;
                let delta = (fd * fd + ad * ad + dd * dd).sqrt();
                let values = [f64::from(v.k1?), f64::from(v.k2?), f64::from(v.k3?)];
                if delta < 0.0001 {
                    return Some(values);
                }
                nearest = nearest.min(delta);
                let weight = delta.powf(-3.5);
                for channel in 0..3 {
                    sum[channel] += weight * values[channel];
                }
                total += weight;
            }
            (total > 0.0 && nearest <= 1.0).then(|| sum.map(|value| value / total))
        });
    Some(StackLensCalibration {
        a: parameters.k1,
        b: parameters.k2,
        c: parameters.k3,
        vignetting,
    })
}
