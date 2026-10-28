use crate::AppState;
use fuzzy_matcher::FuzzyMatcher;
use include_dir::{Dir, include_dir};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use tauri::{Manager, State};
use walkdir::WalkDir;
static LENS_DB_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/lensfun_db");

/// Calibrated PTLens geometry for registration, without the editor's artistic
/// strength multiplier. The radius unit is half the original short side.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StackLensCalibration {
    a: f64,
    b: f64,
    c: f64,
    vignetting: Option<[f64; 3]>,
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

    fn distorted_radius(self, radius: f64) -> f64 {
        radius
            * (self.a * radius.powi(3) + self.b * radius.powi(2) + self.c * radius + 1.0
                - self.a
                - self.b
                - self.c)
    }

    fn undistorted_radius(self, distorted: f64) -> Option<f64> {
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

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct Distortion {
    #[serde(rename = "@model")]
    pub model: String,
    #[serde(rename = "@focal")]
    pub focal: f32,
    #[serde(rename = "@real-focal")]
    pub real_focal: Option<f32>,
    #[serde(rename = "@k1")]
    pub k1: Option<f32>,
    #[serde(rename = "@k2")]
    pub k2: Option<f32>,
    #[serde(rename = "@k3")]
    pub k3: Option<f32>,
    #[serde(rename = "@a")]
    pub a: Option<f32>,
    #[serde(rename = "@b")]
    pub b: Option<f32>,
    #[serde(rename = "@c")]
    pub c: Option<f32>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct Tca {
    #[serde(rename = "@model")]
    pub model: String,
    #[serde(rename = "@focal")]
    pub focal: f32,
    #[serde(rename = "@vr")]
    pub vr: Option<f32>,
    #[serde(rename = "@vb")]
    pub vb: Option<f32>,
    #[serde(rename = "@cr")]
    pub cr: Option<f32>,
    #[serde(rename = "@cb")]
    pub cb: Option<f32>,
    #[serde(rename = "@br")]
    pub br: Option<f32>,
    #[serde(rename = "@bb")]
    pub bb: Option<f32>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct Vignetting {
    #[serde(rename = "@model")]
    pub model: String,
    #[serde(rename = "@focal")]
    pub focal: f32,
    #[serde(rename = "@aperture")]
    pub aperture: f32,
    #[serde(rename = "@distance")]
    pub distance: Option<f32>,
    #[serde(rename = "@k1")]
    pub k1: Option<f32>,
    #[serde(rename = "@k2")]
    pub k2: Option<f32>,
    #[serde(rename = "@k3")]
    pub k3: Option<f32>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum CalibrationElement {
    Distortion(Distortion),
    Tca(Tca),
    Vignetting(Vignetting),
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct Calibration {
    #[serde(rename = "$value", default)]
    pub elements: Vec<CalibrationElement>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct Focal {
    #[serde(rename = "@value")]
    pub value: Option<f32>,
    #[serde(rename = "@min")]
    pub min: Option<f32>,
    #[serde(rename = "@max")]
    pub max: Option<f32>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct Aperture {
    #[serde(rename = "@min")]
    pub min: Option<f32>,
    #[serde(rename = "@max")]
    pub max: Option<f32>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Lens {
    #[serde(default)]
    pub maker: Vec<MultiName>,
    #[serde(default)]
    pub model: Vec<MultiName>,
    #[serde(default)]
    pub mount: Vec<String>,
    pub cropfactor: Option<f32>,
    pub calibration: Option<Calibration>,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub focal: Option<Focal>,
    pub aspect_ratio: Option<String>,
    pub center: Option<String>,
    pub compat: Option<String>,
    pub notes: Option<String>,
    pub aperture: Option<Aperture>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Camera {
    pub maker: Vec<MultiName>,
    pub model: Vec<MultiName>,
    pub mount: String,
    pub cropfactor: f32,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct LensDatabase {
    #[serde(rename = "camera", default)]
    pub cameras: Vec<Camera>,
    #[serde(rename = "lens", default)]
    pub lenses: Vec<Lens>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct MultiName {
    #[serde(rename = "@lang")]
    lang: Option<String>,
    #[serde(rename = "$value")]
    value: String,
}

#[derive(Debug, Serialize)]
pub struct LensDistortionParams {
    k1: f64,
    k2: f64,
    k3: f64,
    model: u32,
    tca_vr: f64,
    tca_vb: f64,
    vig_k1: f64,
    vig_k2: f64,
    vig_k3: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectedLensProfile {
    maker: String,
    model: String,
    params: Option<LensDistortionParams>,
}

fn strip_maker_prefix(name: &str, maker: &str) -> String {
    if name.to_lowercase().starts_with(&maker.to_lowercase())
        && let Some(rest) = name.get(maker.len()..)
    {
        let trimmed = rest.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    name.to_string()
}

impl Lens {
    pub fn get_full_model_name(&self) -> String {
        self.model
            .iter()
            .find(|m| m.lang.as_deref() == Some("en"))
            .or_else(|| self.model.first())
            .map(|m| m.value.clone())
            .unwrap_or_else(|| "Unknown Model".to_string())
    }

    pub fn get_canonical_model_name(&self) -> String {
        self.model
            .iter()
            .find(|m| m.lang.is_none())
            .or_else(|| self.model.first())
            .map(|m| m.value.clone())
            .unwrap_or_else(|| "Unknown Model".to_string())
    }

    pub fn get_name(&self) -> String {
        let raw_name = self.get_full_model_name();
        let maker = self.get_maker();

        if raw_name.to_lowercase().starts_with(&maker.to_lowercase())
            && let Some(rest) = raw_name.get(maker.len()..)
        {
            let stripped = rest.trim();
            if !stripped.is_empty() {
                return stripped.to_string();
            }
        }

        raw_name
    }

    pub fn get_maker(&self) -> String {
        self.maker
            .iter()
            .find(|m| m.lang.as_deref() == Some("en"))
            .or_else(|| self.maker.first())
            .map(|m| m.value.clone())
            .unwrap_or_else(|| "Misc".to_string())
    }

    pub fn get_display_name(&self, all_maker_lenses: &[&Lens]) -> String {
        let my_short = self.get_name();
        let short_count = all_maker_lenses
            .iter()
            .filter(|l| l.get_name() == my_short)
            .count();

        if short_count <= 1 {
            return my_short;
        }

        let maker = self.get_maker();
        let my_canonical_short = strip_maker_prefix(&self.get_canonical_model_name(), &maker);

        let canonical_short_count = all_maker_lenses
            .iter()
            .filter(|l| {
                strip_maker_prefix(&l.get_canonical_model_name(), &l.get_maker())
                    == my_canonical_short
            })
            .count();

        if canonical_short_count <= 1 {
            return my_canonical_short;
        }

        let my_canonical = self.get_canonical_model_name();
        let canonical_count = all_maker_lenses
            .iter()
            .filter(|l| l.get_canonical_model_name() == my_canonical)
            .count();

        if canonical_count <= 1 {
            return my_canonical;
        }

        if let Some(cf) = self.cropfactor {
            format!("{} (crop {:.1}x)", my_canonical_short, cf)
        } else {
            my_canonical_short
        }
    }

    pub fn get_distortion_params(
        &self,
        focal_length: f32,
        aperture: Option<f32>,
        distance: Option<f32>,
    ) -> Option<LensDistortionParams> {
        let cal = self.calibration.as_ref()?;

        let mut distortions: Vec<&Distortion> = cal
            .elements
            .iter()
            .filter_map(|e| {
                if let CalibrationElement::Distortion(d) = e {
                    Some(d)
                } else {
                    None
                }
            })
            .collect();

        let mut tcas: Vec<&Tca> = cal
            .elements
            .iter()
            .filter_map(|e| {
                if let CalibrationElement::Tca(t) = e {
                    Some(t)
                } else {
                    None
                }
            })
            .collect();

        let mut vignettings: Vec<&Vignetting> = cal
            .elements
            .iter()
            .filter_map(|e| {
                if let CalibrationElement::Vignetting(v) = e {
                    Some(v)
                } else {
                    None
                }
            })
            .collect();

        let (k1, k2, k3, model) = if distortions.is_empty() {
            (0.0, 0.0, 0.0, 0)
        } else {
            distortions.sort_by(|a, b| a.focal.partial_cmp(&b.focal).unwrap_or(Ordering::Equal));

            if let Some(exact) = distortions
                .iter()
                .find(|d| (d.focal - focal_length).abs() < 1e-5)
            {
                extract_dist_params(exact)
            } else if focal_length < distortions[0].focal {
                extract_dist_params(distortions[0])
            } else if focal_length > distortions.last().unwrap().focal {
                extract_dist_params(distortions.last().unwrap())
            } else {
                let mut res = (0.0, 0.0, 0.0, 0);
                for pair in distortions.windows(2) {
                    let (d1, d2) = (&pair[0], &pair[1]);

                    if focal_length >= d1.focal && focal_length <= d2.focal {
                        let p1 = extract_dist_params(d1);
                        let p2 = extract_dist_params(d2);

                        let range = d2.focal - d1.focal;
                        if range.abs() < 1e-5 || p1.3 != p2.3 {
                            res = p1;
                        } else {
                            let t = (focal_length - d1.focal) / range;
                            res = (
                                p1.0 + t as f64 * (p2.0 - p1.0),
                                p1.1 + t as f64 * (p2.1 - p1.1),
                                p1.2 + t as f64 * (p2.2 - p1.2),
                                p1.3,
                            );
                        }
                        break;
                    }
                }
                res
            }
        };

        let (tca_vr, tca_vb) = if tcas.is_empty() {
            (1.0, 1.0)
        } else {
            tcas.sort_by(|a, b| a.focal.partial_cmp(&b.focal).unwrap_or(Ordering::Equal));

            if let Some(exact) = tcas.iter().find(|d| (d.focal - focal_length).abs() < 1e-5) {
                extract_tca_params(exact)
            } else if focal_length < tcas[0].focal {
                extract_tca_params(tcas[0])
            } else if focal_length > tcas.last().unwrap().focal {
                extract_tca_params(tcas.last().unwrap())
            } else {
                let mut res = (1.0, 1.0);
                for pair in tcas.windows(2) {
                    let (d1, d2) = (&pair[0], &pair[1]);
                    if focal_length >= d1.focal && focal_length <= d2.focal {
                        let p1 = extract_tca_params(d1);
                        let p2 = extract_tca_params(d2);

                        let range = d2.focal - d1.focal;
                        if range.abs() < 1e-5 {
                            res = p1;
                        } else {
                            let t = (focal_length - d1.focal) / range;
                            res = (
                                p1.0 + t as f64 * (p2.0 - p1.0),
                                p1.1 + t as f64 * (p2.1 - p1.1),
                            );
                        }
                        break;
                    }
                }
                res
            }
        };

        let (vig_k1, vig_k2, vig_k3) = if vignettings.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let target_aperture = aperture.unwrap_or(3.5);
            let target_distance = distance.unwrap_or(1000.0);

            vignettings.sort_by(|a, b| a.focal.partial_cmp(&b.focal).unwrap_or(Ordering::Equal));

            let find_best_vig = |items: &[&Vignetting]| -> (f64, f64, f64) {
                let best_aperture_item = items.iter().min_by(|a, b| {
                    (a.aperture - target_aperture)
                        .abs()
                        .partial_cmp(&(b.aperture - target_aperture).abs())
                        .unwrap_or(Ordering::Equal)
                });
                if let Some(best_ap) = best_aperture_item {
                    let candidates: Vec<&&Vignetting> = items
                        .iter()
                        .filter(|x| (x.aperture - best_ap.aperture).abs() < 0.01)
                        .collect();
                    let best_dist = candidates.into_iter().min_by(|a, b| {
                        let da = a.distance.unwrap_or(1000.0);
                        let db = b.distance.unwrap_or(1000.0);
                        (da - target_distance)
                            .abs()
                            .partial_cmp(&(db - target_distance).abs())
                            .unwrap_or(Ordering::Equal)
                    });
                    extract_vig_params(best_dist.unwrap_or(best_ap))
                } else {
                    (0.0, 0.0, 0.0)
                }
            };

            if focal_length <= vignettings[0].focal + 0.01 {
                let group: Vec<&Vignetting> = vignettings
                    .iter()
                    .filter(|x| (x.focal - vignettings[0].focal).abs() < 0.01)
                    .copied()
                    .collect();
                find_best_vig(&group)
            } else if focal_length >= vignettings.last().unwrap().focal - 0.01 {
                let last_focal = vignettings.last().unwrap().focal;
                let group: Vec<&Vignetting> = vignettings
                    .iter()
                    .filter(|x| (x.focal - last_focal).abs() < 0.01)
                    .copied()
                    .collect();
                find_best_vig(&group)
            } else {
                let mut res = (0.0, 0.0, 0.0);
                let unique_focals: Vec<f32> = {
                    let mut f: Vec<f32> = vignettings.iter().map(|v| v.focal).collect();
                    f.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                    f.dedup_by(|a, b| (*a - *b).abs() < 0.01);
                    f
                };
                for pair in unique_focals.windows(2) {
                    let (f1, f2) = (pair[0], pair[1]);
                    if focal_length >= f1 && focal_length <= f2 {
                        let group1: Vec<&Vignetting> = vignettings
                            .iter()
                            .filter(|x| (x.focal - f1).abs() < 0.01)
                            .copied()
                            .collect();
                        let group2: Vec<&Vignetting> = vignettings
                            .iter()
                            .filter(|x| (x.focal - f2).abs() < 0.01)
                            .copied()
                            .collect();

                        let p1 = find_best_vig(&group1);
                        let p2 = find_best_vig(&group2);

                        let range = f2 - f1;
                        if range.abs() > 0.01 {
                            let t = (focal_length - f1) / range;
                            res = (
                                p1.0 + t as f64 * (p2.0 - p1.0),
                                p1.1 + t as f64 * (p2.1 - p1.1),
                                p1.2 + t as f64 * (p2.2 - p1.2),
                            );
                        } else {
                            res = p1;
                        }
                        break;
                    }
                }
                res
            }
        };

        Some(LensDistortionParams {
            k1,
            k2,
            k3,
            model,
            tca_vr,
            tca_vb,
            vig_k1,
            vig_k2,
            vig_k3,
        })
    }
}

fn extract_dist_params(dist: &Distortion) -> (f64, f64, f64, u32) {
    match dist.model.as_str() {
        "poly3" | "poly5" => (
            dist.k1.unwrap_or(0.0) as f64,
            dist.k2.unwrap_or(0.0) as f64,
            dist.k3.unwrap_or(0.0) as f64,
            0,
        ),
        "ptlens" => {
            let a = dist.a.unwrap_or(0.0) as f64;
            let b = dist.b.unwrap_or(0.0) as f64;
            let c = dist.c.unwrap_or(0.0) as f64;
            (a, b, c, 1)
        }
        _ => (0.0, 0.0, 0.0, 0),
    }
}

fn extract_tca_params(tca: &Tca) -> (f64, f64) {
    (tca.vr.unwrap_or(1.0) as f64, tca.vb.unwrap_or(1.0) as f64)
}

fn extract_vig_params(vig: &Vignetting) -> (f64, f64, f64) {
    (
        vig.k1.unwrap_or(0.0) as f64,
        vig.k2.unwrap_or(0.0) as f64,
        vig.k3.unwrap_or(0.0) as f64,
    )
}

fn lenses_for_maker<'a>(db: &'a LensDatabase, maker: &str) -> Vec<&'a Lens> {
    db.lenses
        .iter()
        .filter(|l| l.get_maker() == maker)
        .collect()
}

pub fn load_lensfun_db(app_handle: &tauri::AppHandle) -> LensDatabase {
    let mut combined_db = LensDatabase {
        cameras: Vec::new(),
        lenses: Vec::new(),
    };

    #[cfg(target_os = "android")]
    {
        log::info!("Loading Lensfun DB from embedded assets (Android path)");

        for file in LENS_DB_DIR.files() {
            let is_xml = file
                .path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("xml"))
                .unwrap_or(false);

            if is_xml {
                if let Some(xml_content) = file.contents_utf8() {
                    match quick_xml::de::from_str::<LensDatabase>(xml_content) {
                        Ok(mut db) => {
                            combined_db.cameras.append(&mut db.cameras);
                            combined_db.lenses.append(&mut db.lenses);
                        }
                        Err(e) => {
                            log::error!("Failed to parse embedded XML {:?}: {}", file.path(), e)
                        }
                    }
                }
            }
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        let resource_path = app_handle
            .path()
            .resolve("lensfun_db", tauri::path::BaseDirectory::Resource)
            .expect("failed to resolve lensfun_db directory");

        if !resource_path.exists() {
            log::error!("Lensfun DB directory not found at: {:?}", resource_path);
            return combined_db;
        }

        for entry in WalkDir::new(resource_path)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "xml"))
        {
            let path = entry.path();
            log::info!("Processing file: {:?}", path);
            match fs::read_to_string(path) {
                Ok(xml_content) => match quick_xml::de::from_str::<LensDatabase>(&xml_content) {
                    Ok(mut db) => {
                        combined_db.cameras.append(&mut db.cameras);
                        combined_db.lenses.append(&mut db.lenses);
                    }
                    Err(e) => {
                        log::error!("Failed to parse Lensfun XML file {:?}: {}", path, e);
                    }
                },
                Err(e) => log::error!("Failed to read Lensfun XML file {:?}: {}", path, e),
            }
        }
    }

    log::info!(
        "Loaded {} lenses and {} cameras from Lensfun database.",
        combined_db.lenses.len(),
        combined_db.cameras.len()
    );
    combined_db
}

#[tauri::command]
pub fn get_lensfun_makers(state: State<AppState>) -> Result<Vec<String>, String> {
    let db_guard = state
        .lens_db
        .lock()
        .map_err(|e| format!("Lock poisoned: {}", e))?;
    if let Some(db) = &*db_guard {
        let mut makers: Vec<String> = db.lenses.iter().map(|lens| lens.get_maker()).collect();
        makers.sort_unstable();
        makers.dedup();
        Ok(makers)
    } else {
        Err("Lens database not loaded".to_string())
    }
}

#[tauri::command]
pub fn get_lensfun_lenses_for_maker(
    maker: String,
    state: State<AppState>,
) -> Result<Vec<String>, String> {
    let db_guard = state
        .lens_db
        .lock()
        .map_err(|e| format!("Lock poisoned: {}", e))?;
    if let Some(db) = &*db_guard {
        let maker_lenses = lenses_for_maker(db, &maker);

        let mut models: Vec<String> = maker_lenses
            .iter()
            .map(|lens| lens.get_display_name(&maker_lenses))
            .collect();
        models.sort_unstable();
        models.dedup();
        Ok(models)
    } else {
        Err("Lens database not loaded".to_string())
    }
}

pub fn find_best_lens_match(
    db: &LensDatabase,
    maker: &str,
    model: &str,
) -> Option<(String, String)> {
    let clean_maker = maker.trim().trim_matches('"').to_string();
    let clean_model = model.trim().trim_matches('"').to_string();
    if clean_model.len() < 3 {
        return None;
    }

    let matcher = fuzzy_matcher::skim::SkimMatcherV2::default().ignore_case();
    let normalized_query = normalize_search_text(&clean_model);
    let query_tokens = search_tokens(&normalized_query);

    let best_match = db
        .lenses
        .iter()
        .filter_map(|lens| {
            let english_name = lens.get_full_model_name();
            let canonical_name = lens.get_canonical_model_name();

            let score_english = matcher
                .fuzzy_match(&english_name, &clean_model)
                .unwrap_or(0);
            let score_canonical = matcher
                .fuzzy_match(&canonical_name, &clean_model)
                .unwrap_or(0);
            let (score, best_name) = if score_canonical > score_english {
                (score_canonical, canonical_name)
            } else {
                (score_english, english_name)
            };
            let normalized_candidate = normalize_search_text(&best_name);
            let candidate_tokens = search_tokens(&normalized_candidate);
            let token_overlap = query_tokens
                .iter()
                .filter(|token| candidate_tokens.contains(token))
                .count() as i64;
            let distinctive_token_overlap = query_tokens
                .iter()
                .filter(|token| token.len() >= 4 && candidate_tokens.contains(token))
                .count() as i64;
            let contains_model = normalized_candidate.contains(&normalized_query)
                || normalized_query.contains(&normalized_candidate);
            let maker_match = makers_match(&clean_maker, &lens.get_maker());
            let length_penalty =
                (normalized_candidate.len() as i64 - normalized_query.len() as i64).abs() / 3;
            let adjusted_score = score - length_penalty
                + token_overlap * 12
                + distinctive_token_overlap * 40
                + if contains_model { 25 } else { 0 }
                + if maker_match { 14 } else { 0 };

            let has_model_evidence =
                token_overlap >= 2 || contains_model || (score > 0 && token_overlap > 0);

            if adjusted_score >= 24 && has_model_evidence {
                Some((adjusted_score, lens))
            } else {
                None
            }
        })
        .max_by_key(|(score, _): &(i64, _)| *score);

    if let Some((_, best_lens)) = best_match {
        let lens_maker = best_lens.get_maker();
        let maker_lenses = lenses_for_maker(db, &lens_maker);
        return Some((lens_maker, best_lens.get_display_name(&maker_lenses)));
    }

    None
}

fn normalize_search_text(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn search_tokens(value: &str) -> Vec<&str> {
    value
        .split_whitespace()
        .filter(|token| token.len() >= 2)
        .collect()
}

fn makers_match(left: &str, right: &str) -> bool {
    let normalized_left = normalize_search_text(left);
    let normalized_right = normalize_search_text(right);
    !normalized_left.is_empty()
        && !normalized_right.is_empty()
        && (normalized_left == normalized_right
            || normalized_left.contains(&normalized_right)
            || normalized_right.contains(&normalized_left))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack_metadata() -> std::collections::HashMap<String, String> {
        [
            ("Make", "NIKON CORPORATION"),
            ("Model", "NIKON Z 7_2"),
            ("LensModel", "NIKKOR Z 24-120mm f/4 S"),
            ("FocalLength", "24"),
            ("FocalLengthIn35mmFilm", "24"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect()
    }

    #[test]
    fn calibrated_footprint_excludes_padding_and_retains_real_black_pixels() {
        let calibration = stack_lens_calibration(&stack_metadata(), (8256, 5504)).unwrap();
        let white = image::Rgb32FImage::from_pixel(600, 400, image::Rgb([1.0; 3]));
        let corrected = calibration.correct(&white).unwrap();
        let footprint = StackLensFootprint {
            calibration,
            original: white.dimensions(),
            corrected: corrected.dimensions(),
        };
        let mut supported = 0;
        let mut padding = 0;
        for y in 0..corrected.height() {
            for x in 0..corrected.width() {
                if footprint.contains(x as f64, y as f64, corrected.dimensions()) {
                    supported += 1;
                    assert!(corrected.get_pixel(x, y)[0] > 0.99);
                } else if corrected.get_pixel(x, y)[0] == 0.0 {
                    padding += 1;
                }
            }
        }
        assert!(supported > 600 * 350 && padding > 1000);
        let black = image::Rgb32FImage::new(600, 400);
        let corrected_black = calibration.correct(&black).unwrap();
        assert_eq!(
            corrected_black.get_pixel(corrected.width() / 2, corrected.height() / 2)[0],
            0.0
        );
        assert!(footprint.contains(
            (corrected.width() / 2) as f64,
            (corrected.height() / 2) as f64,
            corrected.dimensions()
        ));
        assert!(footprint.contains(150.0, 100.0, (300, 200)));
    }

    #[test]
    fn stack_calibration_requires_exact_known_hardware_and_sensor_format() {
        let mut metadata = stack_metadata();
        let calibration = stack_lens_calibration(&metadata, (8256, 5504)).unwrap();
        assert!((calibration.a - 0.0428727).abs() < 1e-7);
        assert!((calibration.b + 0.140952).abs() < 1e-7);
        assert!((calibration.c - 0.0855397).abs() < 1e-7);
        metadata.insert("LensModel".into(), "NIKKOR Z 24-70mm f/4 S unknown".into());
        assert!(stack_lens_calibration(&metadata, (8256, 5504)).is_none());
        metadata = stack_metadata();
        metadata.insert("Model".into(), "NIKON Z 50".into());
        assert!(stack_lens_calibration(&metadata, (8256, 5504)).is_none());
        metadata = stack_metadata();
        assert!(stack_lens_calibration(&metadata, (8000, 5504)).is_none());
        metadata.insert("FocalLengthIn35mmFilm".into(), "36".into());
        assert!(stack_lens_calibration(&metadata, (8256, 5504)).is_none());
        metadata.remove("FocalLengthIn35mmFilm");
        assert!(stack_lens_calibration(&metadata, (8256, 5504)).is_none());
        metadata.insert("FocalLength".into(), "200".into());
        assert!(stack_lens_calibration(&metadata, (8256, 5504)).is_none());
    }

    #[test]
    fn stack_vignetting_corrects_linear_sensor_values_before_gamma() {
        let mut metadata = stack_metadata();
        metadata.insert("FNumber".into(), "f/4".into());
        let calibration = stack_lens_calibration(&metadata, (8256, 5504)).unwrap();
        let k = calibration.vignetting.unwrap();
        assert!((k[0] + 0.5030).abs() < 1e-6);
        assert!((k[1] + 0.2477).abs() < 1e-6);
        assert!((k[2] + 0.0487).abs() < 1e-6);
        let mut image = image::Rgb32FImage::from_pixel(600, 400, image::Rgb([0.1, 0.2, 0.3]));
        calibration.correct_linear_illumination(&mut image);
        let center = image.get_pixel(300, 200);
        assert!((center[0] - 0.1).abs() < 1e-5);
        let corner = image.get_pixel(0, 0);
        assert!(corner[0] > 0.48 && corner[0] < 0.51);
        assert!(
            corner[2] > 1.0,
            "linear headroom must survive until the display curve"
        );
        assert!((corner[1] / corner[0] - 2.0).abs() < 1e-5);
        metadata.insert("FNumber".into(), "f/5".into());
        let interpolated = stack_lens_calibration(&metadata, (8256, 5504))
            .unwrap()
            .vignetting
            .unwrap();
        assert!(
            interpolated
                .iter()
                .all(|coefficient| coefficient.is_finite())
        );
        let falloff = 1.0 + interpolated.iter().sum::<f64>();
        assert!(falloff > 0.05 && falloff < 0.5);
    }

    #[test]
    fn stack_ptlens_radius_uses_original_short_side_and_full_strength() {
        let calibration = stack_lens_calibration(&stack_metadata(), (8256, 5504)).unwrap();
        assert!((calibration.distorted_radius(1.0) - 1.0).abs() < 1e-12);
        let distorted = calibration.distorted_radius(1.5);
        assert!((distorted - 1.45260376875).abs() < 1e-6);
        assert!((calibration.undistorted_radius(distorted).unwrap() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn stack_lens_canvas_preserves_original_boundary_and_masks_empty_corners() {
        let calibration = stack_lens_calibration(&stack_metadata(), (8256, 5504)).unwrap();
        let source = image::Rgb32FImage::from_pixel(600, 400, image::Rgb([0.4, 0.5, 0.6]));
        let corrected = calibration.correct(&source).unwrap();
        assert!(corrected.width() > source.width());
        assert!(corrected.height() > source.height());
        assert_eq!(corrected.get_pixel(0, 0).0, [0.0; 3]);
        let center = corrected.get_pixel(corrected.width() / 2, corrected.height() / 2);
        assert!((center[0] - 0.4).abs() < 1e-6);
        let unit = 200.0;
        for (x, y) in [(0.0, 0.0), (599.0, 399.0), (299.5, 0.0), (0.0, 199.5)] {
            let (dx, dy) = (x - 299.5f64, y - 199.5f64);
            let r = dx.hypot(dy) / unit;
            let ratio = calibration.undistorted_radius(r).unwrap() / r;
            assert!(dx.abs() * ratio < f64::from(corrected.width()) / 2.0);
            assert!(dy.abs() * ratio < f64::from(corrected.height()) / 2.0);
        }
    }

    #[test]
    fn bundled_sony_lensfun_data_covers_a7r_v_and_calibrated_e_mount_lenses() {
        let database: LensDatabase =
            quick_xml::de::from_str(include_str!("../lensfun_db/mil-sony.xml"))
                .expect("parse bundled Sony Lensfun database");

        let a7r_v = database
            .cameras
            .iter()
            .find(|camera| camera.model.iter().any(|name| name.value == "ILCE-7RM5"));
        let a7r_v = a7r_v.expect("Sony ILCE-7RM5 Lensfun camera entry");
        assert_eq!(a7r_v.mount, "Sony E");
        assert_eq!(a7r_v.cropfactor, 1.0);

        assert!(database.lenses.iter().any(|lens| {
            lens.mount.iter().any(|mount| mount == "Sony E")
                && lens
                    .calibration
                    .as_ref()
                    .is_some_and(|calibration| !calibration.elements.is_empty())
        }));
    }

    #[test]
    fn matches_profiles_for_the_bundled_real_raw_fixtures() {
        let mut sony: LensDatabase =
            quick_xml::de::from_str(include_str!("../lensfun_db/mil-sony.xml"))
                .expect("parse bundled Sony Lensfun database");
        let mut nikon: LensDatabase =
            quick_xml::de::from_str(include_str!("../lensfun_db/mil-nikon.xml"))
                .expect("parse bundled Nikon Lensfun database");
        let mut tamron: LensDatabase =
            quick_xml::de::from_str(include_str!("../lensfun_db/mil-tamron.xml"))
                .expect("parse bundled Tamron Lensfun database");

        sony.lenses.append(&mut nikon.lenses);
        sony.lenses.append(&mut tamron.lenses);

        let sony_match = find_best_lens_match(&sony, "SONY", "FE 20mm F1.8 G")
            .expect("match the Sony a7R V sample lens");
        assert_eq!(sony_match.0, "Sony");
        assert!(sony_match.1.contains("20mm"));
        assert!(
            resolve_lens_params(&sony, &sony_match.0, &sony_match.1, 20.0, Some(8.0), None)
                .is_some()
        );

        let nikon_match = find_best_lens_match(&sony, "NIKON", "NIKKOR Z MC 105mm f/2.8 VR S")
            .expect("match the Nikon Z sample lens");
        assert_eq!(nikon_match.0, "Nikon");
        assert!(nikon_match.1.contains("105mm"));
        assert!(
            resolve_lens_params(
                &sony,
                &nikon_match.0,
                &nikon_match.1,
                105.0,
                Some(4.5),
                None
            )
            .is_some()
        );

        let tamron_match = find_best_lens_match(&sony, "SONY", "E 35-150mm F2.0-F2.8 A058")
            .expect("match the third-party Sony E sample lens");
        assert_eq!(tamron_match.0, "Tamron");
        assert!(
            tamron_match.1.contains("35-150mm"),
            "unexpected Tamron match: {tamron_match:?}"
        );
        assert!(
            resolve_lens_params(
                &sony,
                &tamron_match.0,
                &tamron_match.1,
                45.0,
                Some(4.0),
                None
            )
            .is_some()
        );

        assert!(find_best_lens_match(&sony, "SONY", "Unknown Lens XYZ 999").is_none());
    }
}

#[tauri::command]
pub fn autodetect_lens(
    maker: String,
    model: String,
    state: tauri::State<AppState>,
) -> Result<Option<(String, String)>, String> {
    let db_guard = state
        .lens_db
        .lock()
        .map_err(|e| format!("Lock poisoned: {}", e))?;
    if let Some(db) = &*db_guard {
        Ok(find_best_lens_match(db, &maker, &model))
    } else {
        Ok(None)
    }
}

#[tauri::command]
pub fn autodetect_lens_profile(
    maker: String,
    model: String,
    focal_length: f32,
    aperture: Option<f32>,
    distance: Option<f32>,
    state: tauri::State<AppState>,
) -> Result<Option<DetectedLensProfile>, String> {
    let db_guard = state
        .lens_db
        .lock()
        .map_err(|e| format!("Lock poisoned: {}", e))?;
    let Some(db) = &*db_guard else {
        return Ok(None);
    };
    let Some((detected_maker, detected_model)) = find_best_lens_match(db, &maker, &model) else {
        return Ok(None);
    };

    let params = resolve_lens_params(
        db,
        &detected_maker,
        &detected_model,
        focal_length,
        aperture,
        distance,
    );

    Ok(Some(DetectedLensProfile {
        maker: detected_maker,
        model: detected_model,
        params,
    }))
}

#[tauri::command]
pub fn get_lens_distortion_params(
    maker: String,
    model: String,
    focal_length: f32,
    aperture: Option<f32>,
    distance: Option<f32>,
    state: State<AppState>,
) -> Result<Option<LensDistortionParams>, String> {
    let db_guard = state
        .lens_db
        .lock()
        .map_err(|e| format!("Lock poisoned: {}", e))?;
    if let Some(db) = &*db_guard {
        let maker_lenses = lenses_for_maker(db, &maker);

        if let Some(lens) = maker_lenses
            .iter()
            .find(|l| l.get_display_name(&maker_lenses) == model)
        {
            return Ok(lens.get_distortion_params(focal_length, aperture, distance));
        }
    }
    Ok(None)
}

pub fn resolve_lens_params(
    db: &LensDatabase,
    maker: &str,
    model: &str,
    focal_length: f32,
    aperture: Option<f32>,
    distance: Option<f32>,
) -> Option<LensDistortionParams> {
    let maker_lenses = lenses_for_maker(db, maker);
    if let Some(lens) = maker_lenses
        .iter()
        .find(|l| l.get_display_name(&maker_lenses) == model)
    {
        lens.get_distortion_params(focal_length, aperture, distance)
    } else {
        None
    }
}
