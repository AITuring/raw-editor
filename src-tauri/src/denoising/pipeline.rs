//! Image-level denoising jobs.
//!
//! Loads one source, runs the AI denoiser or BM3D on it, blends luma detail
//! back, converts between RGB and YCbCr planes and emits the before/after
//! preview payloads the commands report.

use crate::app_settings::load_settings;
use crate::formats::is_raw_file;
use crate::image_loader::load_base_image_from_bytes;
use crate::image_processing::apply_cpu_default_raw_processing;
use base64::Engine as _;
use base64::engine::general_purpose;
use image::DynamicImage;
use image::GenericImageView;
use image::ImageFormat;
use image::Rgb;
use image::Rgb32FImage;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use tauri::AppHandle;
use tauri::Emitter;

use super::{BLOCK_SIZE, Bm3dParams, DctTables, ProgressReporter, STRIDE, bm3d_process_joint};

pub(super) fn run_bm3d(
    rgb_img: &Rgb32FImage,
    intensity: f32,
    app_handle: &AppHandle,
) -> Result<DynamicImage, String> {
    Ok(run_bm3d_core(rgb_img, intensity, Some(app_handle)))
}

pub(super) fn run_bm3d_core(
    rgb_img: &Rgb32FImage,
    intensity: f32,
    app_handle: Option<&AppHandle>,
) -> DynamicImage {
    if intensity <= 0.0 {
        return DynamicImage::ImageRgb32F(rgb_img.clone());
    }

    let (width, height) = rgb_img.dimensions();
    let params = Bm3dParams::from_intensity(intensity);
    let dct_tables = Arc::new(DctTables::new());

    let rgb_channels = split_channels(rgb_img);
    let (y, cb, cr) = rgb_to_ycbcr(&rgb_channels[0], &rgb_channels[1], &rgb_channels[2]);
    let original_y = y.clone();
    let channels = vec![y, cb, cr];

    let patches_x = (width as usize).saturating_sub(BLOCK_SIZE) / STRIDE + 1;
    let patches_y = (height as usize).saturating_sub(BLOCK_SIZE) / STRIDE + 1;
    let total_work_units = (patches_x * patches_y) * 2;
    let progress_counter = Arc::new(AtomicUsize::new(0));

    if let Some(app_handle) = app_handle {
        let _ = app_handle.emit("denoise-progress", "Processing (Step 1/2)...");
    }

    let progress = ProgressReporter {
        counter: &progress_counter,
        total_work: total_work_units,
        app_handle,
    };
    let mut denoised_channels =
        bm3d_process_joint(&channels, width, height, &params, &dct_tables, &progress);

    {
        if let Some(app_handle) = app_handle {
            let _ = app_handle.emit("denoise-progress", "Blending detail...");
        }
        let blurred_y = gaussian_blur_1ch(&original_y, width as usize, height as usize, 3.0);
        // Low strength should retain more of the source texture, while high
        // strength should allow the BM3D result to suppress it. Increasing this
        // blend with intensity made the control feel flat (and even reversed).
        let detail_strength = (0.35_f32 * (1.0_f32 - intensity)).clamp(0.0_f32, 0.35_f32);
        let y_ch = &mut denoised_channels[0];
        for i in 0..y_ch.len() {
            let hf = original_y[i] - blurred_y[i];
            y_ch[i] = (y_ch[i] + detail_strength * hf).clamp(0.0, 255.0);
        }
    }

    let (r, g, b) = ycbcr_to_rgb(
        &denoised_channels[0],
        &denoised_channels[1],
        &denoised_channels[2],
    );

    let out_img_buffer = merge_channels(&[r, g, b], width, height);
    DynamicImage::ImageRgb32F(out_img_buffer)
}

pub(super) fn denoise_image(
    path_str: String,
    intensity: f32,
    method: String,
    app_handle: AppHandle,
    ai_session: Option<Arc<Mutex<ort::session::Session>>>,
) -> Result<(DynamicImage, String), String> {
    let path = Path::new(&path_str);
    if !path.exists() {
        return Err("File not found".to_string());
    }

    let is_raw = is_raw_file(&path_str);
    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let _ = app_handle.emit("denoise-progress", "Loading image...");

    let file_bytes = fs::read(path).map_err(|e| e.to_string())?;
    let dynamic_img = load_base_image_from_bytes(&file_bytes, &path_str, false, &settings, None)
        .map_err(|e| e.to_string())?;

    let rgb_img_for_denoiser = dynamic_img.to_rgb32f();

    let out_dynamic = if method == "ai" {
        let session_arc = ai_session.ok_or_else(|| "AI Session not provided".to_string())?;
        crate::ai_processing::run_ai_denoise(
            &rgb_img_for_denoiser,
            intensity,
            &session_arc,
            &app_handle,
        )
        .map_err(|e| e.to_string())?
    } else {
        run_bm3d(&rgb_img_for_denoiser, intensity, &app_handle)?
    };

    let _ = app_handle.emit("denoise-progress", "Finalizing data...");
    let _ = app_handle.emit("denoise-progress", "Generating previews...");

    let (width, height) = out_dynamic.dimensions();
    let (new_width, new_height) = if width > height {
        if width > 4000 {
            (4000, (4000.0 * height as f32 / width as f32).round() as u32)
        } else {
            (width, height)
        }
    } else {
        if height > 4000 {
            ((4000.0 * width as f32 / height as f32).round() as u32, 4000)
        } else {
            (width, height)
        }
    };

    let mut denoised_preview_source = out_dynamic.clone();

    if is_raw {
        apply_cpu_default_raw_processing(&mut denoised_preview_source);
    }

    let denoised_preview = if new_width != width {
        denoised_preview_source.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
    } else {
        denoised_preview_source
    };

    let mut buf_denoised = Cursor::new(Vec::new());
    denoised_preview
        .to_rgb8()
        .write_to(&mut buf_denoised, ImageFormat::Png)
        .map_err(|e| format!("Failed to encode preview: {}", e))?;
    let base64_str_denoised = general_purpose::STANDARD.encode(buf_denoised.get_ref());
    let data_url_denoised = format!("data:image/png;base64,{}", base64_str_denoised);

    let mut original_dynamic = DynamicImage::ImageRgb32F(rgb_img_for_denoiser);

    if is_raw {
        apply_cpu_default_raw_processing(&mut original_dynamic);
    }
    let original_preview = if new_width != width {
        original_dynamic.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
    } else {
        original_dynamic
    };

    let mut buf_orig = Cursor::new(Vec::new());
    original_preview
        .to_rgb8()
        .write_to(&mut buf_orig, ImageFormat::Png)
        .map_err(|e| format!("Failed to encode original preview: {}", e))?;
    let base64_str_orig = general_purpose::STANDARD.encode(buf_orig.get_ref());
    let data_url_orig = format!("data:image/png;base64,{}", base64_str_orig);

    let payload = serde_json::json!({
        "denoised": data_url_denoised,
        "original": data_url_orig
    });

    let _ = app_handle.emit("denoise-complete", &payload);

    Ok((out_dynamic, data_url_denoised))
}

pub(super) fn rgb_to_ycbcr(r: &[f32], g: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut y = vec![0.0f32; n];
    let mut cb = vec![0.0f32; n];
    let mut cr = vec![0.0f32; n];
    for i in 0..n {
        let rv = r[i];
        let gv = g[i];
        let bv = b[i];
        y[i] = 0.299 * rv + 0.587 * gv + 0.114 * bv;
        cb[i] = -0.168736 * rv - 0.331264 * gv + 0.5 * bv + 128.0;
        cr[i] = 0.5 * rv - 0.418688 * gv - 0.081312 * bv + 128.0;
    }
    (y, cb, cr)
}

pub(super) fn ycbcr_to_rgb(y: &[f32], cb: &[f32], cr: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = y.len();
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        let yv = y[i];
        let cbv = cb[i] - 128.0;
        let crv = cr[i] - 128.0;
        r[i] = yv + 1.402 * crv;
        g[i] = yv - 0.344136 * cbv - 0.714136 * crv;
        b[i] = yv + 1.772 * cbv;
    }
    (r, g, b)
}

pub(super) fn split_channels(img: &Rgb32FImage) -> Vec<Vec<f32>> {
    let (w, h) = img.dimensions();
    let size = (w * h) as usize;
    let mut r = vec![0.0; size];
    let mut g = vec![0.0; size];
    let mut b = vec![0.0; size];
    for (i, p) in img.pixels().enumerate() {
        r[i] = p[0] * 255.0;
        g[i] = p[1] * 255.0;
        b[i] = p[2] * 255.0;
    }
    vec![r, g, b]
}

pub(super) fn merge_channels(channels: &[Vec<f32>], w: u32, h: u32) -> Rgb32FImage {
    let mut img = Rgb32FImage::new(w, h);
    for (i, p) in img.pixels_mut().enumerate() {
        let r = channels[0][i].clamp(0.0, 255.0) / 255.0;
        let g = channels[1][i].clamp(0.0, 255.0) / 255.0;
        let b = channels[2][i].clamp(0.0, 255.0) / 255.0;
        *p = Rgb([r, g, b]);
    }
    img
}

pub(super) fn gaussian_blur_1ch(data: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let klen = 2 * radius + 1;
    let mut kernel = vec![0.0f32; klen];
    let two_s2 = 2.0 * sigma * sigma;
    for (i, kernel_val) in kernel.iter_mut().enumerate() {
        let k = i as f32 - radius as f32;
        *kernel_val = (-k * k / two_s2).exp();
    }
    let ksum: f32 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= ksum;
    }

    let mut tmp = vec![0.0f32; width * height];
    for y in 0..height {
        let row_in = &data[y * width..(y + 1) * width];
        let row_out = &mut tmp[y * width..(y + 1) * width];
        for (x, out_val) in row_out.iter_mut().enumerate() {
            let mut val = 0.0f32;
            let mut wsum = 0.0f32;
            let x0 = x as isize - radius as isize;
            for (ki, &kernel_val) in kernel.iter().enumerate() {
                let kx = x0 + ki as isize;
                if kx >= 0 && kx < width as isize {
                    val += row_in[kx as usize] * kernel_val;
                    wsum += kernel_val;
                }
            }
            *out_val = val / wsum;
        }
    }

    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        let y0 = y as isize - radius as isize;
        for x in 0..width {
            let mut val = 0.0f32;
            let mut wsum = 0.0f32;
            for (ki, &kernel_val) in kernel.iter().enumerate() {
                let ky = y0 + ki as isize;
                if ky >= 0 && ky < height as isize {
                    val += tmp[ky as usize * width + x] * kernel_val;
                    wsum += kernel_val;
                }
            }
            out[y * width + x] = val / wsum;
        }
    }

    out
}
