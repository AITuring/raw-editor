use rayon::prelude::*;

#[inline(always)]
fn dof_smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn dof_box_filter(buf: &mut [f32], w: usize, h: usize, ch: usize, radius: usize) {
    if radius == 0 || w < 2 || h < 2 {
        return;
    }
    let r = radius as isize;
    let inv = 1.0 / (2 * radius + 1) as f32;
    let wi = w as isize;
    let hi = h as isize;
    let stride = w * ch;

    buf.par_chunks_exact_mut(stride).for_each(|row| {
        let src = row.to_vec();
        let mut sums = vec![0.0f32; ch];
        for i in -r..=r {
            let ix = i.clamp(0, wi - 1) as usize;
            for (c, s) in sums.iter_mut().enumerate() {
                *s += src[ix * ch + c];
            }
        }
        for x in 0..w {
            for (c, s) in sums.iter().enumerate() {
                row[x * ch + c] = *s * inv;
            }
            let add = ((x as isize + r + 1).clamp(0, wi - 1)) as usize;
            let sub = ((x as isize - r).clamp(0, wi - 1)) as usize;
            for (c, s) in sums.iter_mut().enumerate() {
                *s += src[add * ch + c] - src[sub * ch + c];
            }
        }
    });

    let src = buf.to_vec();
    let threads = rayon::current_num_threads().max(1);
    let block_rows = h.div_ceil(threads).max(1);

    buf.par_chunks_mut(block_rows * stride)
        .enumerate()
        .for_each(|(bi, block)| {
            let y_start = bi * block_rows;
            let rows_here = block.len() / stride;
            let mut sums = vec![0.0f32; stride];

            for i in -r..=r {
                let iy = (y_start as isize + i).clamp(0, hi - 1) as usize * stride;
                for (j, s) in sums.iter_mut().enumerate() {
                    *s += src[iy + j];
                }
            }

            for yy in 0..rows_here {
                let y = y_start + yy;
                let base = yy * stride;
                for (j, s) in sums.iter().enumerate() {
                    block[base + j] = *s * inv;
                }
                let add = ((y as isize + r + 1).clamp(0, hi - 1)) as usize * stride;
                let sub = ((y as isize - r).clamp(0, hi - 1)) as usize * stride;
                for (j, s) in sums.iter_mut().enumerate() {
                    *s += src[add + j] - src[sub + j];
                }
            }
        });
}

mod render;

pub use render::apply_lens_blur;

mod coc_field;

use coc_field::build_coc_field;

mod bokeh;

use bokeh::{blur_layer_bokeh, build_bokeh_kernel};
