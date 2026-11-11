//! The detail band a focus scan compares frames in.
//!
//! On a full-size render a 45 MP frame's sensor noise carries most of the
//! energy at the pixel scale, and a defocused frame keeps all of that noise.
//! Neighbour differences of output pixels therefore barely tell a blurred
//! frame from a sharp one: on 祁序2 at 1:1 a visibly blurred frame measured
//! 0.84 of a sharp one (DSC_6773 against DSC_6790), and a sharp frame that
//! should have replaced a soft region measured only 1.05x and lost its seam
//! (DSC_6770). Defocus removes the silk weave a few pixels across, where the
//! noise is much weaker: in that band the same pairs measure 0.49 and 1.45x.
//! A render minified far enough to prefilter its sources has averaged much of
//! the noise already and its weave sits near the output pixel, so it keeps the
//! neighbour difference.

use rayon::prelude::*;

/// Canvas pixels per source pixel from which a run measures focus in the
/// weave band. 祁序2 (0.90) and 兰亭 (1.06) at full size came out sharper
/// with it. 祁序1 at full size (0.68) has a finer weave, about four output
/// pixels across, below the band: there the band preferred DSC_3867, softer in
/// the weave but stronger at ten pixels, over the sharp DSC_3880 on a stroke
/// in the very region the user marked, so it keeps the neighbour difference.
const WEAVE_BAND_MIN_CANVAS_SCALE: f64 = 0.75;
/// Gaussian sigmas, in 2×2-binned output pixels, whose difference is the
/// weave band: it peaks at a period of about ten output pixels.
const WEAVE_BAND_SIGMAS: (f32, f32) = (0.75, 1.5);

/// Band of the focus measure of one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FocusBand {
    /// Neighbour differences of output pixels.
    Pixel,
    /// Difference of Gaussians of 2×2-binned output pixels.
    Weave,
}

impl FocusBand {
    /// The band for a run whose sources land at `canvas_scale` canvas pixels
    /// per source pixel.
    pub(super) fn for_canvas_scale(canvas_scale: f64) -> Self {
        if canvas_scale.is_finite() && canvas_scale >= WEAVE_BAND_MIN_CANVAS_SCALE {
            Self::Weave
        } else {
            Self::Pixel
        }
    }

    /// Output pixels a measured value needs on either side of it.
    pub(super) fn margin(self) -> u32 {
        match self {
            Self::Pixel => 1,
            Self::Weave => 2 * (kernel_radius(WEAVE_BAND_SIGMAS.1) as u32 + 1),
        }
    }
}

/// Luma and weave band of a grid, both on its 2×2-binned grid.
pub(super) struct WeaveBand {
    pub(super) width: usize,
    pub(super) height: usize,
    /// Binned luma; `NaN` where any of the four pixels is.
    pub(super) level: Vec<f32>,
    /// Band-passed binned luma; `NaN` where a kernel reaches a `NaN` or the
    /// grid border.
    pub(super) band: Vec<f32>,
}

/// Weave band of the `width`×`height` luma grid `values` (`NaN` where
/// uncovered).
pub(super) fn weave_band(values: &[f32], width: usize, height: usize) -> WeaveBand {
    let (width, height, level) = bin2(values, width, height);
    let fine = gaussian(&level, width, height, WEAVE_BAND_SIGMAS.0);
    let mut band = gaussian(&level, width, height, WEAVE_BAND_SIGMAS.1);
    for (coarse, fine) in band.iter_mut().zip(&fine) {
        *coarse = fine - *coarse;
    }
    WeaveBand {
        width,
        height,
        level,
        band,
    }
}

fn kernel_radius(sigma: f32) -> usize {
    (3.0 * sigma).ceil() as usize
}

/// 2×2 box average; `NaN` where any of the four pixels is.
fn bin2(values: &[f32], width: usize, height: usize) -> (usize, usize, Vec<f32>) {
    let (binned_width, binned_height) = (width / 2, height / 2);
    let mut binned = vec![f32::NAN; binned_width * binned_height];
    if binned_width == 0 || binned_height == 0 {
        return (binned_width, binned_height, binned);
    }
    binned
        .par_chunks_mut(binned_width)
        .enumerate()
        .for_each(|(y, row)| {
            let top = &values[2 * y * width..(2 * y + 1) * width];
            let bottom = &values[(2 * y + 1) * width..(2 * y + 2) * width];
            for (x, value) in row.iter_mut().enumerate() {
                *value = 0.25 * (top[2 * x] + top[2 * x + 1] + bottom[2 * x] + bottom[2 * x + 1]);
            }
        });
    (binned_width, binned_height, binned)
}

/// Separable Gaussian of a grid; `NaN` wherever the kernel reaches a `NaN`
/// or the grid border.
fn gaussian(values: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let radius = kernel_radius(sigma);
    let mut weights = (0..=2 * radius)
        .map(|index| {
            let offset = index as f32 - radius as f32;
            (-offset * offset / (2.0 * sigma * sigma)).exp()
        })
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    for weight in &mut weights {
        *weight /= total;
    }
    let mut horizontal = vec![f32::NAN; values.len()];
    let mut blurred = vec![f32::NAN; values.len()];
    if width <= 2 * radius || height <= 2 * radius {
        return blurred;
    }
    horizontal
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let source = &values[y * width..(y + 1) * width];
            for x in radius..width - radius {
                row[x] = weights
                    .iter()
                    .zip(&source[x - radius..=x + radius])
                    .map(|(weight, value)| weight * value)
                    .sum();
            }
        });
    blurred
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            if y < radius || y + radius >= height {
                return;
            }
            row.fill(0.0);
            for (index, weight) in weights.iter().enumerate() {
                let source =
                    &horizontal[(y + index - radius) * width..(y + index - radius + 1) * width];
                for (value, sample) in row.iter_mut().zip(source) {
                    *value += weight * sample;
                }
            }
        });
    blurred
}
