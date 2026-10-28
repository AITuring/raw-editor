//! Block matching for BM3D.
//!
//! Ranks the patches inside the search window by their joint SSD over the
//! three planes, keeps a power-of-two group and stacks it into a 3D block.

use std::cmp::Ordering;

use super::{BLOCK_AREA, BLOCK_SIZE, Bm3dParams, MAX_GROUP_SIZE, SEARCH_WINDOW};

#[derive(Clone, Copy)]
pub(super) struct Match {
    pub(super) dist: f32,
    pub(super) x: u16,
    pub(super) y: u16,
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn block_matching_joint(
    channels: &[Vec<f32>],
    w: usize,
    h: usize,
    rx: usize,
    ry: usize,
    is_step_1: bool,
    params: &Bm3dParams,
    out_buf: &mut [(usize, usize)],
) -> usize {
    const MAX_CANDIDATES: usize = 1024;
    let mut candidates: [Match; MAX_CANDIDATES] = [Match {
        dist: f32::MAX,
        x: 0,
        y: 0,
    }; MAX_CANDIDATES];
    let mut cand_count = 0;

    let threshold = if is_step_1 {
        params.max_dist_hard
    } else {
        params.max_dist_hard * 0.5
    };

    let mut ref_r = [0.0; 64];
    let mut ref_g = [0.0; 64];
    let mut ref_b = [0.0; 64];
    extract_patch(&channels[0], w, rx, ry, &mut ref_r);
    extract_patch(&channels[1], w, rx, ry, &mut ref_g);
    extract_patch(&channels[2], w, rx, ry, &mut ref_b);

    let half_sw = SEARCH_WINDOW / 2;
    let sx_start = rx.saturating_sub(half_sw);
    let sx_end = (rx + half_sw).min(w.saturating_sub(BLOCK_SIZE));
    let sy_start = ry.saturating_sub(half_sw);
    let sy_end = (ry + half_sw).min(h.saturating_sub(BLOCK_SIZE));

    candidates[0] = Match {
        dist: 0.0,
        x: rx as u16,
        y: ry as u16,
    };
    cand_count += 1;

    for y in sy_start..=sy_end {
        for x in sx_start..=sx_end {
            if x == rx && y == ry {
                continue;
            }
            let d_r = compute_ssd_flat(&channels[0], w, x, y, &ref_r, threshold);
            if d_r > threshold {
                continue;
            }
            let d_g = compute_ssd_flat(&channels[1], w, x, y, &ref_g, threshold - d_r);
            if d_r + d_g > threshold {
                continue;
            }
            let d_b = compute_ssd_flat(&channels[2], w, x, y, &ref_b, threshold - (d_r + d_g));
            let total_dist = d_r + d_g + d_b;

            if total_dist < threshold && cand_count < MAX_CANDIDATES {
                candidates[cand_count] = Match {
                    dist: total_dist,
                    x: x as u16,
                    y: y as u16,
                };
                cand_count += 1;
            }
        }
    }

    let valid_slice = &mut candidates[0..cand_count];
    valid_slice.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

    let limit = MAX_GROUP_SIZE.min(cand_count);
    let p2_limit = prev_power_of_two(limit);

    for i in 0..p2_limit {
        out_buf[i] = (valid_slice[i].x as usize, valid_slice[i].y as usize);
    }
    p2_limit
}

#[inline(always)]
pub(super) fn compute_ssd_flat(
    img: &[f32],
    w: usize,
    x: usize,
    y: usize,
    ref_patch: &[f32],
    stop_thr: f32,
) -> f32 {
    let mut dist = 0.0;
    for dy in 0..8 {
        let img_base = (y + dy) * w + x;
        let ref_base = dy * 8;
        for dx in 0..8 {
            let diff = img[img_base + dx] - ref_patch[ref_base + dx];
            dist += diff * diff;
        }
        if dist > stop_thr {
            return dist;
        }
    }
    dist / BLOCK_AREA as f32
}

#[inline(always)]
pub(super) fn extract_patch(img: &[f32], w: usize, x: usize, y: usize, out: &mut [f32]) {
    for dy in 0..8 {
        let src_idx = (y + dy) * w + x;
        let dst_idx = dy * 8;
        out[dst_idx..dst_idx + 8].copy_from_slice(&img[src_idx..src_idx + 8]);
    }
}

pub(super) fn build_3d_group(img: &[f32], w: usize, locs: &[(usize, usize)]) -> Vec<f32> {
    let mut stack = vec![0.0; locs.len() * 64];
    for (i, &(lx, ly)) in locs.iter().enumerate() {
        let offset = i * 64;
        extract_patch(img, w, lx, ly, &mut stack[offset..offset + 64]);
    }
    stack
}

pub(super) fn prev_power_of_two(x: usize) -> usize {
    if x == 0 {
        return 0;
    }
    let mut p = 1;
    while p * 2 <= x {
        p *= 2;
    }
    p
}
