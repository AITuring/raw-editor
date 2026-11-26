//! Spatially bounded residual exposure correction from independent overlap.
use super::{Field, StreamingMosaicStore, adjusted, median};
use image::{GrayImage, Rgb, Rgb32FImage};
use rayon::prelude::*;
use std::collections::BTreeMap;

const STEP: f64 = 512.0;
const MAX_CELLS: usize = 65_536;
const MAX_SAMPLES: usize = 32;
const MAX_LOG_GAIN: f64 = 0.0769610411361284; // ln(1.08)
type GainCells = BTreeMap<(u8, u32, u32), f64>;

#[derive(Default)]
struct Cell {
    values: Vec<f64>,
    // Non-overlapping 5x5 patches alternate between fit and independent validation.
    positions: Vec<(f64, f64, bool)>,
    quadrants: u8,
}

#[derive(Default)]
pub(super) struct OwnerOverlapGain {
    cells: BTreeMap<(u8, u8, u32, u32), Cell>,
    pub samples: usize,
    pub dropped_cells: usize,
}

impl OwnerOverlapGain {
    #[allow(clippy::too_many_arguments)]
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
                    cell.positions.push((wx, wy, ((x / 8 + y / 8) & 1) == 0));
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
        let baseline = self.accepted();
        let (accepted, refinement) = self.refined(&baseline);
        #[cfg(test)]
        if let Some(path) = std::env::var_os("RAW_EDITOR_TEST_OWNER_GAIN_DIAGNOSTICS") {
            let cells: Vec<_> = self.cells.iter().map(|(key, cell)| {
                serde_json::json!({"key":key,"values":cell.values,"positions":cell.positions,"quadrants":cell.quadrants})
            }).collect();
            std::fs::write(
                path,
                serde_json::to_vec(&serde_json::json!({"cells":cells,"refinement":refinement}))
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| format!("overlap diagnostics: {e}"))?;
        }
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
            serde_json::json!({"method":"heldout_overlap_graph_residual_gain_v1","samples":self.samples,"candidate_pair_cells":self.cells.len(),"accepted_owner_cells":accepted.len(),"dropped_cells":self.dropped_cells,"max_gain":1.08,"min_gain":1.0/1.08,"cell_size_output_pixels":STEP,"support_radius_output_pixels":STEP*1.5,"unsupported_gain":1.0,"refinement":refinement}),
            super::MosaicGainReference {
                gains: std::sync::Arc::new(gains),
                crop,
                scale,
                max_log: MAX_LOG_GAIN as f32,
            },
        ))
    }

    fn refined(&self, baseline: &GainCells) -> (GainCells, serde_json::Value) {
        let mut regions = BTreeMap::<(u32, u32), Vec<(u8, u8, f64)>>::new();
        let mut heldout = Vec::new();
        let mut training = Vec::new();
        for (&(a, b, x, y), cell) in &self.cells {
            if cell.positions.len() != cell.values.len() || cell.quadrants.count_ones() < 2 {
                continue;
            }
            let mut fit = Vec::new();
            let mut fit_positions = Vec::new();
            let mut validation = Vec::new();
            for (&v, &(wx, wy, train)) in cell.values.iter().zip(&cell.positions) {
                if train {
                    fit.push(v);
                    fit_positions.push((wx, wy));
                } else {
                    validation.push((a, b, wx, wy, v));
                }
            }
            if fit.len() < 4 || validation.len() < 4 {
                continue;
            }
            let r = median(&mut fit);
            let mut deviation: Vec<_> = fit.iter().map(|v| (v - r).abs()).collect();
            if median(&mut deviation) > 0.025 {
                continue;
            }
            regions.entry((x, y)).or_default().push((a, b, r));
            let weight = 1.0 / fit_positions.len() as f64;
            training.extend(
                fit_positions
                    .into_iter()
                    .map(|(wx, wy)| (a, b, wx, wy, r, weight)),
            );
            heldout.extend(validation);
        }
        let mut candidate = baseline.clone();
        for (&(x, y), edges) in &regions {
            let mut anchor = BTreeMap::new();
            for &(a, b, _) in edges {
                for owner in [a, b] {
                    anchor.insert(owner, *baseline.get(&(owner, x, y)).unwrap_or(&0.0));
                }
            }
            let mut values = anchor.clone();
            // Solve g(candidate)-g(base)=measured residual. The ridge retains the
            // original field's gauge and limits drift in weakly connected graphs.
            // Damped Jacobi and sorted maps make the result independent of visits.
            for _ in 0..64 {
                let mut next = BTreeMap::new();
                for (&owner, &old) in &values {
                    let mut sum = 0.05 * anchor[&owner];
                    let mut weight = 0.05;
                    for &(a, b, r) in edges {
                        if owner == a {
                            sum += values[&b] + r;
                            weight += 1.0;
                        } else if owner == b {
                            sum += values[&a] - r;
                            weight += 1.0;
                        }
                    }
                    next.insert(
                        owner,
                        (0.5 * old + 0.5 * sum / weight).clamp(-MAX_LOG_GAIN, MAX_LOG_GAIN),
                    );
                }
                values = next;
            }
            for (owner, value) in values {
                candidate.insert((owner, x, y), value);
            }
        }
        fit_interpolated_field(&mut candidate, baseline, &training);
        let error = |cells: &GainCells| {
            let mut residuals: Vec<_> = heldout
                .iter()
                .map(|&(a, b, x, y, r)| {
                    (r - local_gain(cells, a, x, y) + local_gain(cells, b, x, y)).abs()
                })
                .collect();
            residuals.sort_by(f64::total_cmp);
            if residuals.is_empty() {
                return [0.0; 3];
            }
            [
                residuals.iter().sum::<f64>() / residuals.len() as f64,
                residuals[residuals.len() / 2],
                residuals[residuals.len() * 95 / 100],
            ]
        };
        let before = error(baseline);
        let after = error(&candidate);
        // Judge the *interpolated* field on patches excluded from the fit, rather
        // than the solved cell centres. No improvement means exact old pixels.
        let accept = heldout.len() >= 8
            && after[0] < before[0] * 0.95
            && after[1] <= before[1]
            && after[2] <= before[2];
        let summary = serde_json::json!({"accepted":accept,"fit_pair_cells":regions.values().map(Vec::len).sum::<usize>(),"fit_patches":training.len(),"heldout_patches":heldout.len(),"solver":"bounded_coordinate_descent_on_rendered_gain_basis","baseline_abs_log_residual_mean_median_p95":before,"candidate_abs_log_residual_mean_median_p95":after,"fallback":"original_same_content_overlap_field"});
        (if accept { candidate } else { baseline.clone() }, summary)
    }
}

type FitPatch = (u8, u8, f64, f64, f64, f64);

fn fit_interpolated_field(candidate: &mut GainCells, baseline: &GainCells, patches: &[FitPatch]) {
    let keys: Vec<_> = candidate.keys().copied().collect();
    let indices: BTreeMap<_, _> = keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let mut values: Vec<_> = keys.iter().map(|k| candidate[k]).collect();
    let anchors: Vec<_> = keys
        .iter()
        .map(|k| *baseline.get(k).unwrap_or(&0.0))
        .collect();
    let mut columns = vec![Vec::<(usize, f64)>::new(); keys.len()];
    let mut residuals = Vec::with_capacity(patches.len());
    let mut weights = Vec::with_capacity(patches.len());
    for &(a, b, x, y, target, weight) in patches {
        let mut row = Vec::new();
        for (owner, sign) in [(a, 1.0), (b, -1.0)] {
            let basis = local_basis(candidate, owner, x, y);
            for (key, coefficient) in basis {
                row.push((indices[&key], sign * coefficient));
            }
        }
        let index = residuals.len();
        residuals.push(target - row.iter().map(|&(i, c)| values[i] * c).sum::<f64>());
        weights.push(weight);
        for (i, coefficient) in row {
            columns[i].push((index, coefficient));
        }
    }
    // Fit the same spatial basis that the renderer uses, not hypothetical
    // per-cell constants. Equal total weight per pair cell prevents dense cells
    // from overruling independent overlaps. Coordinate updates preserve bounds.
    for _ in 0..64 {
        let mut largest: f64 = 0.0;
        for (i, column) in columns.iter().enumerate() {
            if column.is_empty() {
                continue;
            }
            let mut gradient = 0.05 * (anchors[i] - values[i]);
            let mut diagonal = 0.05;
            for &(row, c) in column {
                gradient += weights[row] * c * residuals[row];
                diagonal += weights[row] * c * c;
            }
            let next = (values[i] + gradient / diagonal).clamp(-MAX_LOG_GAIN, MAX_LOG_GAIN);
            let delta = next - values[i];
            values[i] = next;
            largest = largest.max(delta.abs());
            for &(row, c) in column {
                residuals[row] -= c * delta;
            }
        }
        if largest < 1e-8 {
            break;
        }
    }
    for (key, value) in keys.into_iter().zip(values) {
        candidate.insert(key, value);
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

fn local_basis(cells: &GainCells, owner: u8, x: f64, y: f64) -> Vec<((u8, u32, u32), f64)> {
    let gx = (x / STEP) as i32;
    let gy = (y / STEP) as i32;
    let mut basis = Vec::with_capacity(9);
    let mut weight = 0.0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            if gx + dx < 0 || gy + dy < 0 {
                continue;
            }
            let cx = (gx + dx) as u32;
            let cy = (gy + dy) as u32;
            if cells.contains_key(&(owner, cx, cy)) {
                let wx = (1.0 - (x - (f64::from(cx) + 0.5) * STEP).abs() / (STEP * 1.5)).max(0.0);
                let wy = (1.0 - (y - (f64::from(cy) + 0.5) * STEP).abs() / (STEP * 1.5)).max(0.0);
                let w = wx * wy;
                basis.push(((owner, cx, cy), w));
                weight += w;
            }
        }
    }
    // Do not normalize weak support into a full-strength extrapolation.
    for (_, w) in &mut basis {
        *w /= weight.max(1.0);
    }
    basis
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
                ..Cell::default()
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
                ..Cell::default()
            },
        );
        assert!(gain.accepted().is_empty());
    }

    fn pair(gain: &mut OwnerOverlapGain, a: u8, b: u8, fit: f64, validation: f64) {
        let values = (0..16)
            .map(|i| if i % 2 == 0 { fit } else { validation })
            .collect();
        gain.cells.insert(
            (a, b, 0, 0),
            Cell {
                values,
                positions: (0..16)
                    .map(|i| (240.0 + i as f64, 256.0, i % 2 == 0))
                    .collect(),
                quadrants: 3,
            },
        );
    }

    #[test]
    fn overlap_graph_closes_transitive_exposure_and_preserves_unsupported_regions() {
        let mut gain = OwnerOverlapGain::default();
        pair(&mut gain, 2, 1, 0.03, 0.03);
        pair(&mut gain, 3, 2, 0.04, 0.04);
        let baseline = gain.accepted();
        let (refined, evidence) = gain.refined(&baseline);
        assert_eq!(evidence["accepted"], true);
        for (a, b, r) in [(2, 1, 0.03), (3, 2, 0.04)] {
            assert!(
                (local_gain(&refined, a, 256.0, 256.0) - local_gain(&refined, b, 256.0, 256.0) - r)
                    .abs()
                    < 0.002
            );
        }
        assert_eq!(local_gain(&refined, 99, 256.0, 256.0), 0.0);
        assert_eq!(local_gain(&refined, 2, 4096.0, 256.0), 0.0);
        assert!(refined.values().all(|v| v.abs() <= MAX_LOG_GAIN));
        let mut reverse = OwnerOverlapGain::default();
        pair(&mut reverse, 3, 2, 0.04, 0.04);
        pair(&mut reverse, 2, 1, 0.03, 0.03);
        assert_eq!(reverse.refined(&reverse.accepted()).0, refined);
    }

    #[test]
    fn overlap_graph_rejects_holdout_regression_and_keeps_exact_original_field() {
        let mut gain = OwnerOverlapGain::default();
        pair(&mut gain, 2, 1, 0.06, -0.06);
        pair(&mut gain, 3, 2, 0.06, -0.06);
        let baseline = gain.accepted();
        let (refined, evidence) = gain.refined(&baseline);
        assert_eq!(evidence["accepted"], false);
        assert_eq!(refined, baseline);
        assert_eq!(
            OwnerOverlapGain::default().refined(&GainCells::new()).0,
            GainCells::new()
        );
    }

    #[test]
    fn fitted_basis_matches_renderer_at_sparse_support_and_cell_boundaries() {
        let mut cells = GainCells::new();
        for (x, y, gain) in [(0, 0, 0.02), (1, 0, -0.03), (0, 1, 0.01)] {
            cells.insert((2, x, y), gain);
        }
        cells.insert((2, 8, 8), 0.01);
        for x in [0.0, 255.0, 512.0, 760.0, 1024.0, 4096.0] {
            for y in [0.0, 256.0, 512.0, 1024.0] {
                let fitted: f64 = local_basis(&cells, 2, x, y)
                    .iter()
                    .map(|(key, weight)| cells[key] * weight)
                    .sum();
                assert!((fitted - local_gain(&cells, 2, x, y)).abs() < 1e-14);
            }
        }
        let baseline = cells.clone();
        let before = (0.03 - local_gain(&cells, 2, 760.0, 256.0)).abs();
        fit_interpolated_field(&mut cells, &baseline, &[(2, 1, 760.0, 256.0, 0.03, 1.0)]);
        let after = (0.03 - local_gain(&cells, 2, 760.0, 256.0)).abs();
        assert!(after < before * 0.5);
        assert_eq!(cells[&(2, 8, 8)], baseline[&(2, 8, 8)]);
        assert!(cells.values().all(|v| v.abs() <= MAX_LOG_GAIN));
    }
}
