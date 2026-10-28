//! Planar scan mosaic entry point.
//!
//! Sizes the output grid, runs registration, photometry, ownership and
//! rendering in order, and provides the opaque-source mask that decoded PSD
//! composites hand to the compositor.

use super::super::coordinates::pixel_aligned_canvas;
use super::super::shared::ImageInfo;
use super::super::stitching::apply_photometric_model_to_image;
use image::DynamicImage;
use image::GenericImageView;
use image::GrayImage;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use tauri::AppHandle;
use tauri::Emitter;
use tauri::Runtime;

use super::{
    Canvas, LOW_GRID, MAX_OUTPUT_PIXELS, MAX_OUTPUT_UPSCALE, ScanMosaicResult, ScanSource,
    TileWarp, analyse_tile, consensus_low_band, eroded_validity, low_band, plan_ownership, project,
    refine_homographies, refine_mesh, render_tile_detail, solve_photometry,
};
#[cfg(test)]
use super::{OWNER_CELL, Ownership};

/// Opaque-sample mask of a decoded source, resized to the render size.
/// Photoshop stores white under fully transparent composite pixels; those
/// must not become mosaic content.
pub(crate) fn opaque_source_mask(
    image: &DynamicImage,
    width: u32,
    height: u32,
) -> Option<GrayImage> {
    if !image.color().has_alpha() {
        return None;
    }
    let (source_width, source_height) = image.dimensions();
    let mut data = vec![0u8; source_width as usize * source_height as usize];
    let row_length = source_width as usize;
    match image {
        DynamicImage::ImageRgba16(buffer) => {
            let samples = buffer.as_raw();
            data.par_chunks_mut(row_length)
                .enumerate()
                .for_each(|(y, row)| {
                    let base = y * row_length * 4;
                    for (x, value) in row.iter_mut().enumerate() {
                        *value = if samples[base + x * 4 + 3] == u16::MAX {
                            255
                        } else {
                            0
                        };
                    }
                });
        }
        DynamicImage::ImageRgba32F(buffer) => {
            let samples = buffer.as_raw();
            data.par_chunks_mut(row_length)
                .enumerate()
                .for_each(|(y, row)| {
                    let base = y * row_length * 4;
                    for (x, value) in row.iter_mut().enumerate() {
                        *value = if samples[base + x * 4 + 3] >= 0.9999 {
                            255
                        } else {
                            0
                        };
                    }
                });
        }
        other => {
            let rgba = other.to_rgba8();
            let samples = rgba.as_raw();
            data.par_chunks_mut(row_length)
                .enumerate()
                .for_each(|(y, row)| {
                    let base = y * row_length * 4;
                    for (x, value) in row.iter_mut().enumerate() {
                        *value = if samples[base + x * 4 + 3] == u8::MAX {
                            255
                        } else {
                            0
                        };
                    }
                });
        }
    }
    if data.iter().all(|&value| value == 255) {
        return None;
    }
    let mask = GrayImage::from_raw(source_width, source_height, data)?;
    if (source_width, source_height) == (width, height) {
        return Some(mask);
    }
    Some(image::imageops::resize(
        &mask,
        width,
        height,
        image::imageops::FilterType::Nearest,
    ))
}

/// Compose flattened scan tiles into one source-owned mosaic.
pub(crate) fn planar_scan_mosaic<R: Runtime, F>(
    images: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_source: &mut F,
) -> Result<ScanMosaicResult, String>
where
    F: FnMut(&ImageInfo) -> Result<ScanSource, String>,
{
    if images.is_empty() {
        return Err("No PSD tiles to compose.".to_string());
    }
    let started = Instant::now();
    let initial = images
        .iter()
        .map(|image| {
            homographies
                .get(&image.id)
                .copied()
                .ok_or_else(|| format!("Missing alignment for '{}'", image.filename))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The output grid is chosen before analysis so the sharpness band and the
    // registration blur are expressed in final output pixels.
    let world_scales = images
        .iter()
        .zip(&initial)
        .map(|(image, homography)| {
            local_scale(
                homography,
                Point2::new(image.width as f64 * 0.5, image.height as f64 * 0.5),
            )
            .unwrap_or(1.0)
        })
        .collect::<Vec<_>>();
    let finest = world_scales
        .iter()
        .copied()
        .filter(|scale| scale.is_finite() && *scale > 0.0)
        .fold(f64::INFINITY, f64::min);
    let (min_x, max_x, min_y, max_y) = world_bounds(images, &initial);
    let world_pixels = ((max_x - min_x) * (max_y - min_y)).max(1.0);
    let budget = (MAX_OUTPUT_PIXELS / world_pixels).sqrt().max(1.0);
    let upscale = if finest.is_finite() {
        (1.0 / finest).clamp(1.0, MAX_OUTPUT_UPSCALE).min(budget)
    } else {
        1.0
    };
    println!(
        "  - PSD scan mosaic: {} tiles, world/source scale {:.3}..{:.3}, output upscale {:.3}",
        images.len(),
        finest,
        world_scales.iter().copied().fold(0.0f64, f64::max),
        upscale
    );

    let mut analyses = Vec::with_capacity(images.len());
    for (index, image) in images.iter().enumerate() {
        let _ = app_handle.emit(
            progress_event,
            format!(
                "Analysing PSD tile {} of {}: {}",
                index + 1,
                images.len(),
                file_name(&image.filename)
            ),
        );
        let source = load_source(image)?;
        if source.rgb.dimensions() != (image.width, image.height) {
            return Err(format!(
                "PSD tile '{}' decoded as {}x{}, expected {}x{}",
                image.filename,
                source.rgb.width(),
                source.rgb.height(),
                image.width,
                image.height
            ));
        }
        analyses.push(analyse_tile(&source, world_scales[index] * upscale));
    }
    println!(
        "  - PSD tile analysis completed in {:.2?}",
        started.elapsed()
    );

    let _ = app_handle.emit(progress_event, "Refining PSD tile registration...");
    let fixed = initial
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            (*left - Matrix3::identity())
                .norm()
                .total_cmp(&(*right - Matrix3::identity()).norm())
        })
        .map(|(index, _)| index)
        .unwrap_or(0);
    let refined = refine_homographies(&analyses, &initial, fixed);
    // Planar poses leave a smooth residual of a few pixels (lens distortion,
    // a scroll that is not perfectly flat); an ownership seam crossing a
    // stroke there would break it, so each tile also gets a smooth correction.
    let corrections = refine_mesh(&analyses, &refined);

    let photometry = solve_photometry(&analyses, &refined);

    let output_scale = Matrix3::new(upscale, 0.0, 0.0, 0.0, upscale, 0.0, 0.0, 0.0, 1.0);
    let output = refined
        .iter()
        .map(|homography| output_scale * homography)
        .collect::<Vec<_>>();
    let inverses = output
        .iter()
        .map(|homography| {
            homography
                .try_inverse()
                .ok_or_else(|| "A refined PSD tile alignment is not invertible.".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let warps = output
        .iter()
        .zip(&inverses)
        .enumerate()
        .map(|(index, (forward, inverse))| TileWarp {
            forward: *forward,
            inverse: *inverse,
            correction: corrections.as_ref().map(|grids| grids[index].clone()),
        })
        .collect::<Vec<_>>();
    let (min_x, max_x, min_y, max_y) = world_bounds(images, &output);
    let (offset_x, width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, height) = pixel_aligned_canvas(min_y, max_y);
    let canvas = Canvas {
        width,
        height,
        offset_x,
        offset_y,
    };
    println!("  - PSD scan output canvas: {}x{}", width, height);

    let low_bands = analyses
        .par_iter()
        .zip(&warps)
        .zip(&photometry)
        .map(|((analysis, warp), (model, gain))| low_band(analysis, warp, model, *gain, &canvas))
        .collect::<Vec<_>>();
    // Glass reflections and camera-position glare are low-frequency and differ
    // per tile; the robust consensus of all tiles carries the tone instead of
    // the owner, so one tile's reflection cannot become a visible blotch.
    let low_mix = consensus_low_band(&low_bands, &canvas);

    let _ = app_handle.emit(
        progress_event,
        "Selecting the sharpest PSD tile per region...",
    );
    let ownership = plan_ownership(&analyses, &warps, &canvas, &low_bands, &low_mix);
    #[cfg(test)]
    write_test_diagnostics(images, &warps, &canvas, upscale, &ownership);

    let pixel_count = width as usize * height as usize;
    let mut detail = vec![0.0f32; pixel_count * 3];
    let mut weight = vec![0.0f32; pixel_count];
    for (index, image) in images.iter().enumerate() {
        let _ = app_handle.emit(
            progress_event,
            format!(
                "Rendering PSD tile {} of {}: {}",
                index + 1,
                images.len(),
                file_name(&image.filename)
            ),
        );
        let source = load_source(image)?;
        let valid = eroded_validity(&source);
        let mut rgb = source.rgb;
        apply_photometric_model_to_image(&mut rgb, &photometry[index].0, photometry[index].1);
        render_tile_detail(
            index,
            &rgb,
            &valid,
            &warps[index],
            &low_bands[index],
            &ownership,
            &canvas,
            &mut detail,
            &mut weight,
        );
    }

    let row = width as usize;
    let mut coverage = vec![0u8; pixel_count];
    detail
        .par_chunks_mut(row * 3)
        .zip(weight.par_chunks(row))
        .zip(coverage.par_chunks_mut(row))
        .enumerate()
        .for_each(|(y, ((detail_row, weight_row), coverage_row))| {
            let grid_y = (y as f64 - (LOW_GRID as f64 - 1.0) * 0.5) / LOW_GRID as f64;
            for x in 0..row {
                let w = weight_row[x];
                let start = x * 3;
                if w > 0.0 {
                    let grid_x = (x as f64 - (LOW_GRID as f64 - 1.0) * 0.5) / LOW_GRID as f64;
                    let low = low_mix.sample(grid_x, grid_y);
                    for channel in 0..3 {
                        detail_row[start + channel] =
                            (low[channel] + detail_row[start + channel] / w).clamp(0.0, 1.0);
                    }
                    coverage_row[x] = 255;
                } else if w < 0.0 {
                    for channel in 0..3 {
                        detail_row[start + channel] = detail_row[start + channel].clamp(0.0, 1.0);
                    }
                    coverage_row[x] = 255;
                } else {
                    detail_row[start..start + 3].fill(0.0);
                }
            }
        });
    let covered = coverage.iter().filter(|&&value| value > 0).count();
    println!(
        "  - PSD scan mosaic rendered {}x{} ({:.2}% covered) in {:.2?}",
        width,
        height,
        covered as f64 * 100.0 / pixel_count.max(1) as f64,
        started.elapsed()
    );
    let image = Rgb32FImage::from_raw(width, height, detail)
        .ok_or_else(|| "PSD scan mosaic buffer has an invalid size.".to_string())?;
    let coverage = GrayImage::from_raw(width, height, coverage)
        .ok_or_else(|| "PSD scan coverage buffer has an invalid size.".to_string())?;
    Ok(ScanMosaicResult { image, coverage })
}

/// Test-only: persist the output transform and the ownership map so an
/// acceptance run can be inspected against the sources pixel for pixel.
#[cfg(test)]
pub(super) fn write_test_diagnostics(
    images: &[&ImageInfo],
    warps: &[TileWarp],
    canvas: &Canvas,
    upscale: f64,
    ownership: &Ownership,
) {
    let Some(directory) = std::env::var_os("RAW_EDITOR_SCAN_MOSAIC_DIAGNOSTICS_DIR") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    if std::fs::create_dir_all(&directory).is_err() {
        return;
    }
    let tiles = images
        .iter()
        .zip(warps)
        .map(|(image, warp)| {
            serde_json::json!({
                "filename": image.filename,
                "width": image.width,
                "height": image.height,
                "source_to_output_world": warp.forward.iter().copied().collect::<Vec<f64>>(),
                // Forward correction F in source pixels: the tile maps to the
                // output as H(p + F(p)).
                "correction": warp.correction.as_ref().map(|grid| serde_json::json!({
                    "spacing": grid.spacing,
                    "columns": grid.columns,
                    "rows": grid.rows,
                    "values": grid.values.iter().flat_map(|value| *value).collect::<Vec<f64>>(),
                })),
            })
        })
        .collect::<Vec<_>>();
    let record = serde_json::json!({
        "canvas": { "width": canvas.width, "height": canvas.height,
                    "offset_x": canvas.offset_x, "offset_y": canvas.offset_y },
        "upscale": upscale,
        "owner_cell": OWNER_CELL,
        "owner_columns": ownership.columns,
        "owner_rows": ownership.rows,
        "owners": ownership.owners,
        "matrix_order": "column-major nalgebra storage",
        "tiles": tiles,
    });
    let _ = std::fs::write(
        directory.join("scan-mosaic.json"),
        serde_json::to_vec(&record).unwrap_or_default(),
    );
}

pub(super) fn file_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// Square root of the Jacobian determinant of `homography` at `point`.
pub(super) fn local_scale(homography: &Matrix3<f64>, point: Point2<f64>) -> Option<f64> {
    let centre = project(homography, point)?;
    let right = project(homography, point + nalgebra::Vector2::new(1.0, 0.0))?;
    let down = project(homography, point + nalgebra::Vector2::new(0.0, 1.0))?;
    let dx = right - centre;
    let dy = down - centre;
    let determinant = (dx.x * dy.y - dx.y * dy.x).abs();
    (determinant.is_finite() && determinant > 0.0).then(|| determinant.sqrt())
}

pub(super) fn world_bounds(
    images: &[&ImageInfo],
    homographies: &[Matrix3<f64>],
) -> (f64, f64, f64, f64) {
    let mut bounds = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for (image, homography) in images.iter().zip(homographies) {
        for (x, y) in [
            (0.0, 0.0),
            (image.width as f64, 0.0),
            (image.width as f64, image.height as f64),
            (0.0, image.height as f64),
        ] {
            if let Some(point) = project(homography, Point2::new(x, y)) {
                bounds.0 = bounds.0.min(point.x);
                bounds.1 = bounds.1.max(point.x);
                bounds.2 = bounds.2.min(point.y);
                bounds.3 = bounds.3.max(point.y);
            }
        }
    }
    bounds
}
