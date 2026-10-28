//! Joint BM3D filtering of the Y/Cb/Cr planes.
//!
//! Runs the hard-threshold basic estimate and the Wiener final estimate over
//! the patch grid in parallel and aggregates every filtered group with atomic
//! fixed-point accumulators while reporting progress.

use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering as AtomicOrdering;
use tauri::AppHandle;
use tauri::Emitter;

use super::{
    BLOCK_AREA, BLOCK_SIZE, Bm3dParams, DctTables, FIXED_POINT_SCALE, MAX_GROUP_SIZE, STRIDE,
    block_matching_joint, build_3d_group, inverse_transform_3d, transform_3d,
};

pub(super) struct ProgressReporter<'a> {
    pub(super) counter: &'a Arc<AtomicUsize>,
    pub(super) total_work: usize,
    pub(super) app_handle: Option<&'a AppHandle>,
}

pub(super) fn bm3d_process_joint(
    noisy_channels: &[Vec<f32>],
    width: u32,
    height: u32,
    params: &Bm3dParams,
    tables: &DctTables,
    progress: &ProgressReporter,
) -> Vec<Vec<f32>> {
    let basic_estimate = run_bm3d_step_joint(
        noisy_channels,
        noisy_channels,
        width,
        height,
        params,
        true,
        tables,
        progress,
    );

    run_bm3d_step_joint(
        noisy_channels,
        &basic_estimate,
        width,
        height,
        params,
        false,
        tables,
        progress,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_bm3d_step_joint(
    noisy: &[Vec<f32>],
    guide: &[Vec<f32>],
    width: u32,
    height: u32,
    params: &Bm3dParams,
    is_step_1: bool,
    tables: &DctTables,
    progress: &ProgressReporter,
) -> Vec<Vec<f32>> {
    let w = width as usize;
    let h = height as usize;
    let count = w * h;
    let num_channels = 3;

    let mut numerators = Vec::new();
    let mut denominators = Vec::new();
    for _ in 0..num_channels {
        numerators.push(Arc::new(AtomicAccumulator::new(count)));
        denominators.push(Arc::new(AtomicAccumulator::new(count)));
    }

    let mut ref_patches = Vec::with_capacity((w / STRIDE) * (h / STRIDE));
    for y in (0..h.saturating_sub(BLOCK_SIZE)).step_by(STRIDE) {
        for x in (0..w.saturating_sub(BLOCK_SIZE)).step_by(STRIDE) {
            ref_patches.push((x, y));
        }
    }

    ref_patches.par_iter().for_each(|&(rx, ry)| {
        let c = progress.counter.fetch_add(1, AtomicOrdering::Relaxed);
        if c.is_multiple_of(200) {
            let pct = (c as f32 / progress.total_work as f32) * 100.0;
            let step_str = if is_step_1 { "Step 1/2" } else { "Step 2/2" };
            let msg = format!("{} - {:.0}%", step_str, pct);
            if let Some(app_handle) = progress.app_handle {
                let _ = app_handle.emit("denoise-progress", msg);
            }
        }

        let mut group_locs_buf = [(0, 0); MAX_GROUP_SIZE];
        let group_size =
            block_matching_joint(guide, w, h, rx, ry, is_step_1, params, &mut group_locs_buf);
        let group_locs = &group_locs_buf[0..group_size];

        for ch in 0..num_channels {
            let guide_ch = &guide[ch];
            let noisy_ch = &noisy[ch];

            let ch_sigma = if ch == 0 {
                params.sigma
            } else {
                params.sigma * params.chroma_sigma_scale
            };

            let mut guide_stack = build_3d_group(guide_ch, w, group_locs);
            let mut noisy_stack = if is_step_1 {
                guide_stack.clone()
            } else {
                build_3d_group(noisy_ch, w, group_locs)
            };

            transform_3d(&mut guide_stack, group_size, tables);
            if !is_step_1 {
                transform_3d(&mut noisy_stack, group_size, tables);
            }

            let weight;
            if is_step_1 {
                let threshold = params.hard_th_lambda * ch_sigma;
                let nonzero = hard_threshold(&mut guide_stack, threshold);
                weight = if nonzero > 0 {
                    1.0 / (nonzero as f32)
                } else {
                    1.0
                };
                noisy_stack = guide_stack;
            } else {
                weight = wiener_filter(&mut noisy_stack, &guide_stack, ch_sigma);
            }

            inverse_transform_3d(&mut noisy_stack, group_size, tables);

            let num_acc = &numerators[ch];
            let den_acc = &denominators[ch];

            for (k, &(lx, ly)) in group_locs.iter().enumerate() {
                let patch_offset = k * BLOCK_AREA;
                for dy in 0..BLOCK_SIZE {
                    let row_global = (ly + dy) * w + lx;
                    let row_patch = dy * BLOCK_SIZE;
                    for dx in 0..BLOCK_SIZE {
                        let idx = row_global + dx;
                        let val = noisy_stack[patch_offset + row_patch + dx];
                        let w_val = tables.kaiser[row_patch + dx] * weight;
                        num_acc.add(idx, val * w_val);
                        den_acc.add(idx, w_val);
                    }
                }
            }
        }
    });

    let mut results = Vec::new();
    for ch in 0..num_channels {
        let num_vec = numerators[ch].to_vec();
        let den_vec = denominators[ch].to_vec();
        let final_ch = num_vec
            .iter()
            .zip(den_vec.iter())
            .zip(noisy[ch].iter())
            .map(|((&n, &d), &orig)| if d > 1e-6 { n / d } else { orig })
            .collect();
        results.push(final_ch);
    }
    results
}

pub(super) fn hard_threshold(stack: &mut [f32], th: f32) -> usize {
    let mut c = 0;
    for (i, x) in stack.iter_mut().enumerate() {
        if i == 0 {
            c += 1;
            continue;
        }

        if x.abs() < th {
            *x = 0.0;
        } else {
            c += 1;
        }
    }
    c
}

pub(super) fn wiener_filter(noisy: &mut [f32], guide: &[f32], sigma: f32) -> f32 {
    let mut sum = 0.0;
    let s2 = sigma * sigma;
    for (i, (n, g)) in noisy.iter_mut().zip(guide).enumerate() {
        if i == 0 {
            sum += 1.0;
            continue;
        }

        let energy = g * g;
        let coef = energy / (energy + s2 + 1e-5);
        *n *= coef;
        sum += coef * coef;
    }
    if sum > 0.0 { 1.0 / sum } else { 1.0 }
}

pub(super) struct AtomicAccumulator {
    pub(super) data: Vec<AtomicI64>,
}

impl AtomicAccumulator {
    pub(super) fn new(size: usize) -> Self {
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(AtomicI64::new(0));
        }
        Self { data }
    }
    #[inline(always)]
    pub(super) fn add(&self, index: usize, value: f32) {
        if index < self.data.len() {
            let fixed = (value * FIXED_POINT_SCALE) as i64;
            self.data[index].fetch_add(fixed, AtomicOrdering::Relaxed);
        }
    }
    pub(super) fn to_vec(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|a| a.load(AtomicOrdering::Relaxed) as f32 / FIXED_POINT_SCALE)
            .collect()
    }
}
