//! Spatially bounded residual exposure correction from independent overlap.
use super::{Field, StreamingMosaicStore, adjusted, median};
use image::{GrayImage, Rgb, Rgb32FImage};
use rayon::prelude::*;
use std::collections::BTreeMap;

const STEP: f64 = 512.0;
const MAX_CELLS: usize = 65_536;
const MAX_SAMPLES: usize = 32;
const MAX_LOG_GAIN: f64 = 0.0769610411361284; // ln(1.08)

#[derive(Default)]
struct Cell {
    values: Vec<f64>,
    quadrants: u8,
}

#[derive(Default)]
pub(super) struct OwnerOverlapGain {
    cells: BTreeMap<(u8, u8, u32, u32), Cell>,
    pub samples: usize,
    pub dropped_cells: usize,
}

impl OwnerOverlapGain {
    pub fn observe(
        &mut self,
        owner: u8,
        base: &Rgb32FImage,
        mask: &GrayImage,
        base_owner: &GrayImage,
        candidate: &Rgb32FImage,
        candidate_mask: &GrayImage,
        tone: &Field<3>,
        left: u32,
        top: u32,
        scale: f64,
    ) {
        let (w, h) = base.dimensions();
        if w < 7 || h < 7 {
            return;
        }
        for y in (3..h - 3).step_by(8) {
            for x in (3..w - 3).step_by(8) {
                let prior = base_owner.get_pixel(x, y)[0];
                if prior == 0 || prior == owner {
                    continue;
                }
                let mut a = [0.0; 25];
                let mut b = [0.0; 25];
                let mut valid = true;
                for (i, (dy, dx)) in (-2_i32..=2)
                    .flat_map(|dy| (-2_i32..=2).map(move |dx| (dy, dx)))
                    .enumerate()
                {
                    let px = (x as i32 + dx) as u32;
                    let py = (y as i32 + dy) as u32;
                    if mask.get_pixel(px, py)[0] == 0
                        || candidate_mask.get_pixel(px, py)[0] == 0
                        || base_owner.get_pixel(px, py)[0] != prior
                    {
                        valid = false;
                        break;
                    }
                    let luma = |p: Rgb<f32>| {
                        0.2126 * f64::from(p[0])
                            + 0.7152 * f64::from(p[1])
                            + 0.0722 * f64::from(p[2])
                    };
                    a[i] = luma(*base.get_pixel(px, py));
                    b[i] = luma(adjusted(
                        *candidate.get_pixel(px, py),
                        tone.at(f64::from(px), f64::from(py)),
                    ));
                }
                if !valid {
                    continue;
                }
                let Some(log_gain) = matched_log_gain(&a, &b) else {
                    continue;
                };
                let wx = f64::from(left) + f64::from(x) * scale;
                let wy = f64::from(top) + f64::from(y) * scale;
                let key = (owner, prior, (wx / STEP) as u32, (wy / STEP) as u32);
                if self.cells.len() >= MAX_CELLS && !self.cells.contains_key(&key) {
                    self.dropped_cells += 1;
                    continue;
                }
                let cell = self.cells.entry(key).or_default();
                if cell.values.len() < MAX_SAMPLES {
                    cell.values.push(log_gain);
                    let q = usize::from(wx % STEP >= STEP * 0.5)
                        + 2 * usize::from(wy % STEP >= STEP * 0.5);
                    cell.quadrants |= 1 << q;
                    self.samples += 1;
                }
            }
        }
    }

    fn accepted(&self) -> BTreeMap<(u8, u32, u32), f64> {
        // Combine only independently accepted pair cells; disagreement fails closed.
        let mut combined = BTreeMap::<(u8, u32, u32), Vec<f64>>::new();
        for (&(owner, _, x, y), cell) in &self.cells {
            if cell.values.len() < 8 || cell.quadrants.count_ones() < 2 {
                continue;
            }
            let mut values = cell.values.clone();
            let value = median(&mut values);
            let mut deviations: Vec<_> = values.iter().map(|v| (v - value).abs()).collect();
            if median(&mut deviations) > 0.025 {
                continue;
            }
            combined.entry((owner, x, y)).or_default().push(value);
        }
        combined
            .into_iter()
            .filter_map(|(key, mut values)| {
                let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (hi - lo <= 0.05)
                    .then(|| (key, median(&mut values).clamp(-MAX_LOG_GAIN, MAX_LOG_GAIN)))
            })
            .collect()
    }

    pub fn apply(
        &self,
        store: &StreamingMosaicStore,
        crop: (u32, u32, u32, u32),
        output: &mut Rgb32FImage,
    ) -> Result<(serde_json::Value, super::MosaicGainReference), String> {
        let accepted = self.accepted();
        let scale = (f64::from(crop.2.max(crop.3)) / 2048.0).max(1.0);
        let w = (f64::from(crop.2) / scale).ceil() as u32;
        let h = (f64::from(crop.3) / scale).ceil() as u32;
        let (_, coverage, owner) = store.analysis_region(crop.0, crop.1, scale, w, h)?;
        let gains = Rgb32FImage::from_fn(w, h, |x, y| {
            if coverage.get_pixel(x, y)[0] == 0 {
                return Rgb([0.0; 3]);
            }
            let value = local_gain(
                &accepted,
                owner.get_pixel(x, y)[0],
                f64::from(crop.0) + f64::from(x) * scale,
                f64::from(crop.1) + f64::from(y) * scale,
            );
            Rgb([value as f32; 3])
        });
        let row_width = output.width() as usize * 3;
        output
            .as_mut()
            .par_chunks_mut(row_width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
                    let log = super::super::stitching::get_high_quality_interpolated_pixel(
                        &gains,
                        x as f64 / scale,
                        y as f64 / scale,
                    )[0]
                    .clamp(-MAX_LOG_GAIN as f32, MAX_LOG_GAIN as f32);
                    for v in pixel {
                        *v = (*v * log.exp()).clamp(0.0, 1.0);
                    }
                }
            });
        Ok((
            serde_json::json!({"method":"same_content_overlap_residual_gain","samples":self.samples,"candidate_pair_cells":self.cells.len(),"accepted_owner_cells":accepted.len(),"dropped_cells":self.dropped_cells,"max_gain":1.08,"min_gain":1.0/1.08,"cell_size_output_pixels":STEP,"support_radius_output_pixels":STEP*1.5,"unsupported_gain":1.0}),
            super::MosaicGainReference {
                gains: std::sync::Arc::new(gains),
                crop,
                scale,
                max_log: MAX_LOG_GAIN as f32,
            },
        ))
    }
}

fn matched_log_gain(a: &[f64; 25], b: &[f64; 25]) -> Option<f64> {
    if a.iter()
        .chain(b)
        .any(|v| !v.is_finite() || *v <= 0.03 || *v >= 0.97)
    {
        return None;
    }
    let ma = a.iter().sum::<f64>() / 25.0;
    let mb = b.iter().sum::<f64>() / 25.0;
    if !(0.06..=0.93).contains(&ma) || !(0.06..=0.93).contains(&mb) {
        return None;
    }
    let va = a.iter().map(|v| (v - ma).powi(2)).sum::<f64>();
    let vb = b.iter().map(|v| (v - mb).powi(2)).sum::<f64>();
    let cross = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - ma) * (b - mb))
        .sum::<f64>();
    if va < 0.0004 || vb < 0.0004 || cross / (va * vb).sqrt() < 0.9 {
        return None;
    }
    let ratio = (ma / mb).ln();
    // Large residuals indicate wrong correspondence or a failed prior tone solve.
    (ratio.abs() <= 0.20).then_some(ratio)
}

fn local_gain(cells: &BTreeMap<(u8, u32, u32), f64>, owner: u8, x: f64, y: f64) -> f64 {
    let gx = (x / STEP) as i32;
    let gy = (y / STEP) as i32;
    let mut sum = 0.0;
    let mut weight = 0.0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            if gx + dx < 0 || gy + dy < 0 {
                continue;
            }
            let cx = (gx + dx) as u32;
            let cy = (gy + dy) as u32;
            if let Some(gain) = cells.get(&(owner, cx, cy)) {
                let wx = (1.0 - (x - (f64::from(cx) + 0.5) * STEP).abs() / (STEP * 1.5)).max(0.0);
                let wy = (1.0 - (y - (f64::from(cy) + 0.5) * STEP).abs() / (STEP * 1.5)).max(0.0);
                let w = wx * wy;
                sum += gain * w;
                weight += w;
            }
        }
    }
    // Do not normalize weak support into a full-strength extrapolation.
    (sum / weight.max(1.0)).clamp(-MAX_LOG_GAIN, MAX_LOG_GAIN)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlap_observes_residual_after_tone_and_never_uses_uncovered_pixels() {
        let base = Rgb32FImage::from_fn(128, 128, |x, y| {
            Rgb([0.3 + ((x * 7 + y * 11) % 17) as f32 * 0.008; 3])
        });
        let candidate = Rgb32FImage::from_fn(128, 128, |x, y| {
            Rgb(base.get_pixel(x, y).0.map(|v| v / 1.05))
        });
        let mask = GrayImage::from_pixel(128, 128, image::Luma([255]));
        let owners = GrayImage::from_pixel(128, 128, image::Luma([1]));
        let mut tone = Field::new(128, 128, 112.0);
        tone.values.fill([1.05_f64.ln(); 3]);
        let mut evidence = OwnerOverlapGain::default();
        evidence.observe(
            2, &base, &mask, &owners, &candidate, &mask, &tone, 0, 0, 8.0,
        );
        let accepted = evidence.accepted();
        assert!(!accepted.is_empty());
        assert!(accepted.values().all(|v| v.abs() < 1e-6));
        let empty = GrayImage::new(128, 128);
        let mut unsupported = OwnerOverlapGain::default();
        unsupported.observe(
            2, &base, &empty, &owners, &candidate, &mask, &tone, 0, 0, 8.0,
        );
        assert_eq!(unsupported.samples, 0);
    }
    #[test]
    fn overlap_rejects_unrelated_texture_and_flat_highlights() {
        let a = std::array::from_fn(|i| 0.2 + i as f64 * 0.01);
        let b = std::array::from_fn(|i| a[24 - i]);
        assert!(matched_log_gain(&a, &b).is_none());
        assert!(matched_log_gain(&[0.5; 25], &[0.6; 25]).is_none());
        let b = a.map(|v| v / 1.05);
        assert!((matched_log_gain(&a, &b).unwrap() - 1.05_f64.ln()).abs() < 1e-8);
    }
    #[test]
    fn overlap_gain_requires_distributed_consistent_support_and_is_bounded() {
        let mut gain = OwnerOverlapGain::default();
        gain.cells.insert(
            (2, 1, 0, 0),
            Cell {
                values: vec![0.15; 8],
                quadrants: 1,
            },
        );
        assert!(gain.accepted().is_empty());
        gain.cells.get_mut(&(2, 1, 0, 0)).unwrap().quadrants = 3;
        let accepted = gain.accepted();
        assert!((local_gain(&accepted, 2, 256.0, 256.0) - MAX_LOG_GAIN).abs() < 1e-8);
        assert_eq!(local_gain(&accepted, 1, 256.0, 256.0), 0.0);
        assert_eq!(local_gain(&accepted, 2, 2048.0, 256.0), 0.0);
        assert!(local_gain(&accepted, 2, 700.0, 256.0) < MAX_LOG_GAIN);
        gain.cells.insert(
            (2, 3, 0, 0),
            Cell {
                values: vec![-0.15; 8],
                quadrants: 3,
            },
        );
        assert!(gain.accepted().is_empty());
    }
}
