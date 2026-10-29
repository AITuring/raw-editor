//! Tiled NIND AI denoising.
//!
//! Mirror-padded tiles run through the denoise session and are blended back
//! with seamless overlap weights.

use anyhow::Result;
use image::DynamicImage;
use image::Rgb;
use image::Rgb32FImage;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;
use tauri::Emitter;

#[derive(Clone, Copy)]
pub(super) struct TileParams {
    pub(super) cs: usize,
    pub(super) ucs: usize,
    pub(super) overlap: usize,
    pub(super) pad: usize,
}

impl TileParams {
    pub(super) const fn new(cs: usize, ucs: usize, overlap: usize) -> Self {
        Self {
            cs,
            ucs,
            overlap,
            pad: (cs - ucs) / 2,
        }
    }
}

pub(super) const TILE_BALANCED: TileParams = TileParams::new(504, 480, 6);

pub(super) const TILE_FASTER: TileParams = TileParams::new(504, 504, 0);

pub(super) const TILE_HIGHER_QUALITY: TileParams = TileParams::new(504, 448, 12);

pub(super) fn select_tile_params(quality_0_1: f32) -> TileParams {
    let q = quality_0_1.clamp(0.0, 1.0);
    if q <= 0.25 {
        TILE_FASTER
    } else if q >= 0.75 {
        TILE_HIGHER_QUALITY
    } else {
        TILE_BALANCED
    }
}

#[inline]
pub(super) fn mirror_coord(c: i32, size: i32) -> i32 {
    if c < 0 {
        (-c).min(size - 1)
    } else if c >= size {
        (2 * size - 1 - c).max(0)
    } else {
        c
    }
}

pub(super) fn extract_tile_mirror(img: &Rgb32FImage, x0: i32, y0: i32, cs: usize) -> Array4<f32> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let mut arr = Array4::zeros((1, 3, cs, cs));
    for dy in 0..cs as i32 {
        for dx in 0..cs as i32 {
            let sx = mirror_coord(x0 + dx, w);
            let sy = mirror_coord(y0 + dy, h);
            let px = img.get_pixel(sx as u32, sy as u32);
            arr[[0, 0, dy as usize, dx as usize]] = px[0];
            arr[[0, 1, dy as usize, dx as usize]] = px[1];
            arr[[0, 2, dy as usize, dx as usize]] = px[2];
        }
    }
    arr
}

pub(super) struct SeamlessBlend {
    pub(super) ud0: usize,
    pub(super) ud1: usize,
    pub(super) ud2: usize,
    pub(super) ud3: usize,
    pub(super) absx0: usize,
    pub(super) absy0: usize,
    pub(super) fswidth: usize,
    pub(super) fsheight: usize,
    pub(super) overlap: usize,
}

pub(super) fn apply_seamless(tile: &mut Array4<f32>, blend: &SeamlessBlend) {
    let SeamlessBlend {
        ud0,
        ud1,
        ud2,
        ud3,
        absx0,
        absy0,
        fswidth,
        fsheight,
        overlap,
    } = *blend;
    let ol = overlap;
    if absx0 > 0 {
        for c in 0..3 {
            for y in ud1..ud3 {
                for x in ud0..(ud0 + ol).min(ud2) {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absy0 > 0 {
        for c in 0..3 {
            for y in ud1..(ud1 + ol).min(ud3) {
                for x in ud0..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absx0 + (ud2 - ud0) < fswidth && ol > 0 {
        let right_start = (ud2 as i32 - ol as i32).max(ud0 as i32) as usize;
        for c in 0..3 {
            for y in ud1..ud3 {
                for x in right_start..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absy0 + (ud3 - ud1) < fsheight && ol > 0 {
        let bottom_start = (ud3 as i32 - ol as i32).max(ud1 as i32) as usize;
        for c in 0..3 {
            for y in bottom_start..ud3 {
                for x in ud0..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
}

pub(super) fn run_native_denoise(
    img: &Rgb32FImage,
    session: &Mutex<Session>,
    accumulator: &mut [f32],
    width: usize,
    height: usize,
    app_handle: &tauri::AppHandle,
    params: TileParams,
) -> Result<()> {
    let w = width as i32;
    let h = height as i32;
    let step = params.ucs.saturating_sub(params.overlap).max(1);
    let iperhl = (width.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let ipervl = (height.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let total = (iperhl + 1) * (ipervl + 1);

    for i in 0..total {
        let yi = i / (iperhl + 1);
        let xi = i % (iperhl + 1);
        let x0 =
            params.ucs as i32 * xi as i32 - params.overlap as i32 * xi as i32 - params.pad as i32;
        let y0 =
            params.ucs as i32 * yi as i32 - params.overlap as i32 * yi as i32 - params.pad as i32;

        if i % 10 == 0 {
            let pct = (i as f32 / total as f32) * 100.0;
            let _ = app_handle.emit("denoise-progress", format!("Denoising… {:.0}%", pct));
        }

        let crop = extract_tile_mirror(img, x0, y0, params.cs);
        let input_values = crop.as_standard_layout().to_owned();
        let t_input = Tensor::from_array(input_values)?;

        let out = {
            let mut sess = session.lock().unwrap();
            let outputs = sess.run(ort::inputs![t_input])?;
            let arr = outputs[0].try_extract_array::<f32>()?.to_owned();
            arr.into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| anyhow::anyhow!("Unexpected output shape: {}", e))?
        };

        let x1pad = (0i32).max(x0 + params.cs as i32 - w) as usize;
        let y1pad = (0i32).max(y0 + params.cs as i32 - h) as usize;
        let ud0 = params.pad;
        let ud1 = params.pad;
        let ud2 = params.cs - params.pad.max(x1pad);
        let ud3 = params.cs - params.pad.max(y1pad);
        let absx0 = (x0 + params.pad as i32).max(0) as usize;
        let absy0 = (y0 + params.pad as i32).max(0) as usize;

        let mut tile = out;
        apply_seamless(
            &mut tile,
            &SeamlessBlend {
                ud0,
                ud1,
                ud2,
                ud3,
                absx0,
                absy0,
                fswidth: width,
                fsheight: height,
                overlap: params.overlap,
            },
        );

        for cy in 0..(ud3 - ud1) {
            for cx in 0..(ud2 - ud0) {
                let gx = absx0 + cx;
                let gy = absy0 + cy;
                if gx < width && gy < height {
                    let base = (gy * width + gx) * 3;
                    accumulator[base] += tile[[0, 0, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                    accumulator[base + 1] += tile[[0, 1, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                    accumulator[base + 2] += tile[[0, 2, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                }
            }
        }
    }
    Ok(())
}

pub(super) fn accumulator_to_rgb32f(acc: &[f32], width: u32, height: u32) -> Rgb32FImage {
    let mut out = Rgb32FImage::new(width, height);
    for (i, p) in out.pixels_mut().enumerate() {
        let i3 = i * 3;
        *p = Rgb([
            acc[i3].clamp(0.0, 1.0),
            acc[i3 + 1].clamp(0.0, 1.0),
            acc[i3 + 2].clamp(0.0, 1.0),
        ]);
    }
    out
}

pub fn run_ai_denoise(
    rgb_img: &Rgb32FImage,
    intensity: f32,
    session: &Mutex<Session>,
    app_handle: &tauri::AppHandle,
) -> Result<DynamicImage> {
    let (width, height) = rgb_img.dimensions();
    let params = select_tile_params(intensity);

    let _ = app_handle.emit("denoise-progress", "Denoising (AI NIND)...");
    let mut accumulator = vec![0.0f32; width as usize * height as usize * 3];
    run_native_denoise(
        rgb_img,
        session,
        &mut accumulator,
        width as usize,
        height as usize,
        app_handle,
        params,
    )?;

    let out_img_buffer = accumulator_to_rgb32f(&accumulator, width, height);
    Ok(DynamicImage::ImageRgb32F(out_img_buffer))
}
