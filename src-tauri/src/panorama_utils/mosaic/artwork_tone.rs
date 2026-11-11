//! Local, overlap-supported photometry for flat artwork. No affine extrapolation.
use super::{Field, median};
use image::{GrayImage, Rgb32FImage};

pub(super) fn artwork_tone(
    base: &Rgb32FImage,
    mask: &GrayImage,
    owner: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    original: &Field<3>,
) -> (Field<3>, serde_json::Value) {
    let (w, h) = base.dimensions();
    let mut field = original.clone();
    let mut cells = vec![Vec::<[f64; 3]>::new(); field.values.len()];
    let mut heldout = vec![Vec::<(u32, u32, [f64; 3])>::new(); field.values.len()];
    let mut regions = vec![0_u8; cells.len()];
    let mut textured = 0usize;
    let mut flat = 0usize;
    let mut rejected = 0usize;
    if w >= 7 && h >= 7 {
        for y in (3..h - 3).step_by(8) {
            for x in (3..w - 3).step_by(8) {
                let prior = owner.get_pixel(x, y)[0];
                let mut a = [[0.0_f64; 3]; 25];
                let mut b = a;
                let mut valid = prior != 0;
                for (i, (dy, dx)) in (-2_i32..=2)
                    .flat_map(|dy| (-2_i32..=2).map(move |dx| (dy, dx)))
                    .enumerate()
                {
                    let px = (x as i32 + dx) as u32;
                    let py = (y as i32 + dy) as u32;
                    if mask.get_pixel(px, py)[0] == 0
                        || candidate_mask.get_pixel(px, py)[0] == 0
                        || owner.get_pixel(px, py)[0] != prior
                    {
                        valid = false;
                        break;
                    }
                    a[i] = base.get_pixel(px, py).0.map(f64::from);
                    b[i] = candidate.get_pixel(px, py).0.map(f64::from);
                }
                let measurement = valid.then(|| match_patch(&a, &b)).flatten();
                let Some((delta, is_flat)) = measurement else {
                    rejected += 1;
                    continue;
                };
                let gx = ((f64::from(x) / field.step).round() as usize).min(field.width - 1);
                let gy = ((f64::from(y) / field.step).round() as usize).min(field.height - 1);
                let index = gy * field.width + gx;
                if (x / 8 + y / 8) % 2 == 0 {
                    cells[index].push(delta);
                } else {
                    heldout[index].push((x, y, delta));
                }
                regions[index] |= 1
                    << (usize::from(f64::from(x) >= gx as f64 * field.step)
                        + 2 * usize::from(f64::from(y) >= gy as f64 * field.step));
                if is_flat {
                    flat += 1;
                } else {
                    textured += 1;
                }
            }
        }
    }
    let mut observations: Vec<(usize, [f64; 3])> = Vec::new();
    for (index, values) in cells.iter().enumerate() {
        if values.len() < 12 || heldout[index].len() < 12 || regions[index].count_ones() < 2 {
            continue;
        }
        let value: [f64; 3] =
            std::array::from_fn(|c| median(&mut values.iter().map(|v| v[c]).collect::<Vec<_>>()));
        let consistent = (0..3).all(|c| {
            median(
                &mut values
                    .iter()
                    .map(|v| (v[c] - value[c]).abs())
                    .collect::<Vec<_>>(),
            ) <= 0.025
        });
        let error = |proposal: Option<[f64; 3]>| {
            median(
                &mut heldout[index]
                    .iter()
                    .map(|&(x, y, target)| {
                        let prediction =
                            proposal.unwrap_or_else(|| original.at(f64::from(x), f64::from(y)));
                        (0..3)
                            .map(|c| (prediction[c] - target[c]).abs())
                            .fold(0.0_f64, f64::max)
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let before = error(None);
        let after = error(Some(value));
        if consistent && before >= 0.01 && after <= 0.025 && after < before * 0.8 {
            let old = original.values[index];
            observations.push((
                index,
                std::array::from_fn(|c| (value[c] - old[c]).clamp(-0.04, 0.04)),
            ));
        }
    }
    // Preserve the baseline outside independently validated local corrections;
    // do not undo a working tone field in areas lacking evidence.
    for gy in 0..field.height {
        for gx in 0..field.width {
            let mut sum = [0.0; 3];
            let mut total = 0.25;
            for &(index, value) in &observations {
                let dx = gx as f64 - (index % field.width) as f64;
                let dy = gy as f64 - (index / field.width) as f64;
                let distance = dx * dx + dy * dy;
                if distance > 9.0 {
                    continue;
                }
                let weight = (-distance / (2.0 * 1.25_f64.powi(2))).exp();
                for c in 0..3 {
                    sum[c] += weight * value[c];
                }
                total += weight;
            }
            let index = gy * field.width + gx;
            field.values[index] = std::array::from_fn(|c| {
                original.values[index][c] + (sum[c] / total).clamp(-0.04, 0.04)
            });
        }
    }
    let summary = serde_json::json!({"method":"matched_local_overlap_tone","textured_samples":textured,"flat_photometric_samples":flat,"rejected_samples":rejected,"accepted_nodes":observations.len(),"max_abs_correction_log_gain":0.04,"heldout_minimum":12,"heldout_relative_error_reduction":0.2,"affine_extrapolation":false,"flat_samples_are_geometry_evidence":false});
    (field, summary)
}

fn match_patch(a: &[[f64; 3]; 25], b: &[[f64; 3]; 25]) -> Option<([f64; 3], bool)> {
    if a.iter()
        .chain(b)
        .flatten()
        .any(|v| !v.is_finite() || *v <= 0.025 || *v >= 0.97)
    {
        return None;
    }
    let ma: [f64; 3] = std::array::from_fn(|c| a.iter().map(|v| v[c]).sum::<f64>() / 25.0);
    let mb: [f64; 3] = std::array::from_fn(|c| b.iter().map(|v| v[c]).sum::<f64>() / 25.0);
    let luma = |v: [f64; 3]| 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
    let la = luma(ma);
    let lb = luma(mb);
    if la < 0.06 || lb < 0.06 || (la / lb).ln().abs() > 0.12 {
        return None;
    }
    let va = a.iter().map(|&v| (luma(v) - la).powi(2)).sum::<f64>();
    let vb = b.iter().map(|&v| (luma(v) - lb).powi(2)).sum::<f64>();
    let flat = va < 0.0004 && vb < 0.0004;
    if flat {
        // These support bounded photometry only. They cannot establish a
        // geometric correspondence or satisfy independent geometry gates.
        let sa = ma.iter().sum::<f64>();
        let sb = mb.iter().sum::<f64>();
        if (0..3).any(|c| (ma[c] / sa - mb[c] / sb).abs() > 0.008) {
            return None;
        }
    } else {
        if va < 0.0004 || vb < 0.0004 {
            return None;
        }
        let cross = a
            .iter()
            .zip(b)
            .map(|(&a, &b)| (luma(a) - la) * (luma(b) - lb))
            .sum::<f64>();
        if cross / (va * vb).sqrt() < 0.85 {
            return None;
        }
    }
    let exposure = (la / lb).ln();
    let delta: [f64; 3] = std::array::from_fn(|c| (ma[c] / mb[c]).ln());
    if delta.iter().any(|v| (*v - exposure).abs() > 0.035) {
        return None;
    }
    Some((delta.map(|v| v.clamp(-0.12, 0.12)), flat))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};
    #[test]
    fn heldout_gate_preserves_an_already_correct_baseline() {
        let base = Rgb32FImage::from_pixel(448, 448, Rgb([0.44; 3]));
        let candidate = Rgb32FImage::from_pixel(448, 448, Rgb([0.4; 3]));
        let mask = GrayImage::from_pixel(448, 448, Luma([255]));
        let owner = GrayImage::from_pixel(448, 448, Luma([1]));
        let mut baseline = Field::new(448, 448, 112.0);
        baseline.values.fill([1.1_f64.ln(); 3]);
        let (field, report) = artwork_tone(&base, &mask, &owner, &candidate, &mask, &baseline);
        assert_eq!(report["accepted_nodes"], 0);
        assert_eq!(field.values, baseline.values);
    }
    #[test]
    fn artwork_tone_rejects_wrong_content_and_colour_but_supports_flat_photometry() {
        let a = std::array::from_fn(|i| [0.25 + i as f64 * 0.007; 3]);
        let b = std::array::from_fn(|i| a[24 - i]);
        assert!(match_patch(&a, &b).is_none());
        assert!(match_patch(&[[0.3, 0.2, 0.1]; 25], &[[0.1, 0.2, 0.3]; 25]).is_none());
        let (gain, flat) = match_patch(&[[0.44; 3]; 25], &[[0.4; 3]; 25]).unwrap();
        assert!(flat);
        assert!((gain[0] - 1.1_f64.ln()).abs() < 1e-8);
    }
    #[test]
    fn artwork_tone_never_samples_invalid_borders_or_extrapolates_far() {
        let base = Rgb32FImage::from_pixel(896, 448, Rgb([0.44; 3]));
        let source = Rgb32FImage::from_pixel(896, 448, Rgb([0.4; 3]));
        let mask = GrayImage::from_fn(896, 448, |x, _| Luma([if x < 112 { 255 } else { 0 }]));
        let full = GrayImage::from_pixel(896, 448, Luma([255]));
        let owner = GrayImage::from_pixel(896, 448, Luma([1]));
        let (field, report) = artwork_tone(
            &base,
            &mask,
            &owner,
            &source,
            &full,
            &Field::new(896, 448, 112.0),
        );
        assert!(report["accepted_nodes"].as_u64().unwrap() > 0);
        assert!(field.at(30.0, 200.0)[0] > 0.02);
        assert_eq!(field.at(850.0, 200.0), [0.0; 3]);
        let (empty, _) = artwork_tone(
            &base,
            &GrayImage::new(896, 448),
            &owner,
            &source,
            &full,
            &Field::new(896, 448, 112.0),
        );
        assert!(empty.values.iter().all(|v| *v == [0.0; 3]));
    }
}

#[derive(Clone)]
struct TonePlan {
    left: u32,
    top: u32,
    scale: f64,
    original: Field<3>,
    matched: Field<3>,
}

/// Replay photometry only after every geometry/ownership decision is frozen.
/// Native owner labels select the source field, without another RAW decode.
pub(super) struct ArtworkToneReplay {
    plans: Vec<Option<TonePlan>>,
}
impl Default for ArtworkToneReplay {
    fn default() -> Self {
        Self {
            plans: vec![None; 256],
        }
    }
}
impl ArtworkToneReplay {
    pub fn record(
        &mut self,
        owner: u8,
        left: u32,
        top: u32,
        scale: f64,
        original: &Field<3>,
        matched: Field<3>,
    ) {
        self.plans[usize::from(owner)] = Some(TonePlan {
            left,
            top,
            scale,
            original: original.clone(),
            matched,
        });
    }
    pub fn apply(
        &self,
        store: &super::StreamingMosaicStore,
        crop: (u32, u32, u32, u32),
        output: &mut Rgb32FImage,
    ) -> Result<serde_json::Value, String> {
        use rayon::prelude::*;
        if self.plans.iter().all(Option::is_none) {
            return Ok(
                serde_json::json!({"method":"frozen_owner_low_frequency_tone_replay","owners":0,"pixels":0,"extra_source_decodes":0,"geometry_feedback":false}),
            );
        }
        let scale = (f64::from(crop.2.max(crop.3)) / 2048.0).max(1.0);
        let w = (f64::from(crop.2) / scale).ceil() as u32;
        let h = (f64::from(crop.3) / scale).ceil() as u32;
        let (_, _, owner) = store.analysis_region(crop.0, crop.1, scale, w, h)?;
        let gains = image::Rgb32FImage::from_fn(w, h, |x, y| {
            let id = usize::from(owner.get_pixel(x, y)[0]);
            let Some(plan) = self.plans[id].as_ref() else {
                return image::Rgb([0.0; 3]);
            };
            let ax = (f64::from(crop.0) + f64::from(x) * scale - f64::from(plan.left)) / plan.scale;
            let ay = (f64::from(crop.1) + f64::from(y) * scale - f64::from(plan.top)) / plan.scale;
            let before = plan.original.at(ax, ay);
            let after = plan.matched.at(ax, ay);
            image::Rgb(std::array::from_fn(|c| {
                (after[c] - before[c]).clamp(-0.04, 0.04) as f32
            }))
        });
        let gains = image::imageops::blur(&gains, 0.75);
        let row_width = output.width() as usize * 3;
        output
            .as_mut()
            .par_chunks_mut(row_width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
                    let correction = super::super::stitching::get_high_quality_interpolated_pixel(
                        &gains,
                        x as f64 / scale,
                        y as f64 / scale,
                    );
                    let before_max = pixel.iter().copied().fold(0.0_f32, f32::max);
                    let corrected: [f32; 3] =
                        std::array::from_fn(|c| pixel[c] * correction[c].clamp(-0.04, 0.04).exp());
                    let after_max = corrected.iter().copied().fold(0.0_f32, f32::max);
                    let headroom = before_max.max(0.995);
                    let compression = if after_max > headroom {
                        headroom / after_max
                    } else {
                        1.0
                    };
                    for c in 0..3 {
                        pixel[c] = (corrected[c] * compression).clamp(0.0, 1.0);
                    }
                }
            });
        Ok(
            serde_json::json!({"method":"frozen_owner_low_frequency_tone_replay","owners":self.plans.iter().filter(|p|p.is_some()).count(),"extra_source_decodes":0,"geometry_feedback":false,"analysis_scale":scale,"gain_blur_sigma":0.75,"max_abs_log_correction":0.04,"highlight_headroom":0.995,"limits":"A smooth correction field bridges the existing ownership feather; RGB detail is not blurred."}),
        )
    }
}

#[cfg(test)]
mod replay_tests {
    use super::*;
    use image::{Luma, Rgb};
    #[test]
    fn replay_preserves_highlight_headroom_and_channel_ratios() {
        let store = super::super::StreamingMosaicStore::new(32, 16).unwrap();
        let original = Rgb32FImage::from_pixel(32, 16, Rgb([0.99, 0.5, 0.3]));
        let mask = GrayImage::from_pixel(32, 16, Luma([255]));
        store.save_tile(0, 0, &original, &mask).unwrap();
        store
            .save_owner_tile(0, 0, &GrayImage::from_pixel(32, 16, Luma([1])))
            .unwrap();
        let old = Field::new(32, 16, 112.0);
        let mut new = old.clone();
        new.values.fill([0.04; 3]);
        let mut replay = ArtworkToneReplay::default();
        replay.record(1, 0, 0, 1.0, &old, new);
        let mut output = original.clone();
        replay.apply(&store, (0, 0, 32, 16), &mut output).unwrap();
        let pixel = output.get_pixel(16, 8);
        assert!(pixel[0] < 1.0);
        assert!((pixel[1] / pixel[0] - 0.5 / 0.99).abs() < 1e-6);
    }
    #[test]
    fn frozen_replay_changes_only_rgb_and_keeps_store_geometry_immutable() {
        let store = super::super::StreamingMosaicStore::new(32, 16).unwrap();
        let original =
            Rgb32FImage::from_fn(32, 16, |x, _| Rgb([if x < 16 { 0.408 } else { 0.515 }; 3]));
        let mask = GrayImage::from_pixel(32, 16, Luma([255]));
        let owner = GrayImage::from_fn(32, 16, |x, _| Luma([if x < 16 { 1 } else { 2 }]));
        store.save_tile(0, 0, &original, &mask).unwrap();
        store.save_owner_tile(0, 0, &owner).unwrap();
        let mut replay = ArtworkToneReplay::default();
        for (id, old, new) in [(1, 1.02_f64, 1.01_f64), (2, 1.03_f64, 1.0_f64)] {
            let mut before = Field::new(32, 16, 112.0);
            before.values.fill([old.ln(); 3]);
            let mut after = Field::new(32, 16, 112.0);
            after.values.fill([new.ln(); 3]);
            replay.record(id, 0, 0, 1.0, &before, after);
        }
        let mut output = image::imageops::crop_imm(&original, 4, 2, 24, 12).to_image();
        let summary = replay.apply(&store, (4, 2, 24, 12), &mut output).unwrap();
        assert!((output.get_pixel(2, 2)[0] - 0.404).abs() < 1e-6);
        assert!((output.get_pixel(20, 2)[0] - 0.50).abs() < 1e-6);
        assert_eq!(output.dimensions(), (24, 12));
        let (rgb, coverage) = store.load_tile(0, 0).unwrap();
        assert_eq!(rgb, original);
        assert_eq!(coverage, mask);
        assert_eq!(store.load_owner_tile(0, 0), owner);
        assert_eq!(summary["geometry_feedback"], false);
        assert_eq!(summary["extra_source_decodes"], 0);
    }
}
