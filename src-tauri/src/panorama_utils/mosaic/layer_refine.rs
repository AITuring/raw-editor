//! Analysis-scale residual refinement of one source layer.
//!
//! Builds the canvas/candidate analysis pair, measures patch offsets on a grid
//! and fits the layer's smooth residual field, for the in-memory and the
//! streaming compositor.

use super::super::registration::refine_warped_patch;
use super::super::stack_pipeline::degradation;
use image::GrayImage;
use image::Luma;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;

use super::{GRID_STEP, LayerSampler, luma, median};

pub(super) fn analysis_pair(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
) -> (GrayImage, GrayImage, GrayImage) {
    let mut a = GrayImage::new(width, height);
    let mut b = GrayImage::new(width, height);
    let mut valid = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let lx = x as f64 * sampler.scale;
            let ly = y as f64 * sampler.scale;
            let gx = (sampler.left as f64 + lx) as u32;
            let gy = (sampler.top as f64 + ly) as u32;
            if gx >= base.width() || gy >= base.height() {
                continue;
            }
            let Some(candidate) = sampler.sample(lx, ly) else {
                continue;
            };
            b.put_pixel(
                x,
                y,
                Luma([(luma(candidate) * 255.0).round().clamp(0.0, 255.0) as u8]),
            );
            if base_mask.get_pixel(gx, gy)[0] == 0 {
                continue;
            }
            let current = *base.get_pixel(gx, gy);
            a.put_pixel(
                x,
                y,
                Luma([(luma(current) * 255.0).round().clamp(0.0, 255.0) as u8]),
            );
            valid.put_pixel(x, y, Luma([255]));
        }
    }
    // Avoid letting resampled weave/noise dominate registration at coarse scale.
    (
        imageproc::filter::gaussian_blur_f32(&a, 0.8),
        imageproc::filter::gaussian_blur_f32(&b, 0.8),
        valid,
    )
}

pub(super) fn refine_layer(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
) {
    let (a, b, valid) = analysis_pair(base, base_mask, sampler, width, height);
    refine_layer_from_analysis(&a, &b, &valid, sampler);
}

pub(super) fn refine_layer_from_analysis(
    a: &GrayImage,
    b: &GrayImage,
    valid: &GrayImage,
    sampler: &mut LayerSampler<'_>,
) {
    let width = a.width().min(b.width()).min(valid.width());
    let height = a.height().min(b.height()).min(valid.height());
    let mut observations = Vec::<(Point2<f64>, [f64; 2], f64)>::new();
    for y in (16..height.saturating_sub(16)).step_by(GRID_STEP as usize) {
        for x in (16..width.saturating_sub(16)).step_by(GRID_STEP as usize) {
            if [-14i32, 0, 14].iter().any(|dy| {
                [-14i32, 0, 14].iter().any(|dx| {
                    valid.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0] == 0
                })
            }) {
                continue;
            }
            let p = Point2::new(x as f64, y as f64);
            let Some(found) = refine_warped_patch(&a, &b, &Matrix3::identity(), p, 9, 7) else {
                continue;
            };
            if found.correlation < 0.88 {
                continue;
            }
            let Some(back) = refine_warped_patch(&b, &a, &Matrix3::identity(), found.target, 9, 7)
            else {
                continue;
            };
            if (back.target - p).norm() > 0.65 {
                continue;
            }
            let d = found.target - p;
            observations.push((p, [d.x, d.y], found.correlation));
        }
    }
    // A repeated stroke can win NCC locally. Require neighbouring measurements
    // to agree before allowing it to move a whole region of the source image.
    let supported = observations
        .iter()
        .filter(|(p, d, _)| {
            let neighbours = observations
                .iter()
                .filter(|(q, _, _)| (q - p).norm() <= 150.0)
                .collect::<Vec<_>>();
            if neighbours.len() < 3 {
                return false;
            }
            let mx = median(&mut neighbours.iter().map(|(_, d, _)| d[0]).collect::<Vec<_>>());
            let my = median(&mut neighbours.iter().map(|(_, d, _)| d[1]).collect::<Vec<_>>());
            (d[0] - mx).hypot(d[1] - my) <= 2.0
        })
        .copied()
        .collect::<Vec<_>>();
    if supported.len() < 4 {
        println!(
            "    - Local registration: insufficient reliable texture; keeping verified global alignment"
        );
        // Observation only (requirement 12.7/12.9): the frame keeps the global
        // model exactly as before, the ledger just records that it did.
        degradation::record_run_degradation(
            degradation::INTRA_STATION_LOCAL_FALLBACK,
            serde_json::json!({
                "stage": "mosaic_local_registration",
                "supported_patches": supported.len(),
                "minimum_patches": 4,
            }),
        );
        return;
    }
    let mut field = sampler.residual.clone();
    for gy in 0..field.height {
        for gx in 0..field.width {
            let p = Point2::new(gx as f64 * field.step, gy as f64 * field.step);
            let mut sum = [0.0; 2];
            let mut total = 0.0f64;
            for (q, d, correlation) in &supported {
                let distance = (p - q).norm_squared();
                let weight = (-distance / (2.0 * 72.0 * 72.0)).exp() * correlation.powi(4);
                sum[0] += weight * d[0];
                sum[1] += weight * d[1];
                total += weight;
            }
            // Unobserved space returns continuously to the global model.
            field.values[gy * field.width + gx][0] += sum[0] / total.max(0.05);
            field.values[gy * field.width + gx][1] += sum[1] / total.max(0.05);
        }
    }
    sampler.residual = field;
    let med = median(
        &mut supported
            .iter()
            .map(|(_, d, _)| d[0].hypot(d[1]) * sampler.scale)
            .collect::<Vec<_>>(),
    );
    println!(
        "    - Local registration: {} consistent patches, median correction {med:.2} output pixels",
        supported.len()
    );
}

pub(super) fn streaming_candidate_analysis(
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
) -> (Rgb32FImage, GrayImage) {
    let mut image = Rgb32FImage::new(width, height);
    let mut mask = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let Some(pixel) = sampler.sample(x as f64 * sampler.scale, y as f64 * sampler.scale)
            else {
                continue;
            };
            image.put_pixel(x, y, pixel);
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    (image, mask)
}

pub(super) fn streaming_refinement_pair(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
) -> (GrayImage, GrayImage, GrayImage) {
    let width = base.width().min(candidate.width());
    let height = base.height().min(candidate.height());
    let mut a = GrayImage::new(width, height);
    let mut b = GrayImage::new(width, height);
    let mut valid = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            if base_mask.get_pixel(x, y)[0] == 0 || candidate_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            a.put_pixel(
                x,
                y,
                Luma([(luma(*base.get_pixel(x, y)) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8]),
            );
            b.put_pixel(
                x,
                y,
                Luma([(luma(*candidate.get_pixel(x, y)) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8]),
            );
            valid.put_pixel(x, y, Luma([255]));
        }
    }
    (
        imageproc::filter::gaussian_blur_f32(&a, 0.8),
        imageproc::filter::gaussian_blur_f32(&b, 0.8),
        valid,
    )
}
