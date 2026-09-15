//! Focus mosaics: refine residual geometry, choose sharp source detail, match
//! only broad colour. The output is never an average of displaced fine detail.
use super::registration::refine_warped_patch;
use super::seam_cut;
use super::stitching::{
    Projection, crop_to_valid_rectangle, downsample_rgb_half, get_high_quality_interpolated_pixel,
    map_target_to_source, output_bounds, pixel_aligned_canvas, transformed_image_region,
};
use crate::panorama_stitching::ImageInfo;
use image::{GrayImage, Luma, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Runtime};

#[cfg(test)]
#[path = "mosaic_diagnostics.rs"]
mod diagnostics;

const ANALYSIS_LONG_SIDE: u32 = 1400;
// Bound the grid per source layer, not by the full panorama width. At a
// roughly 9504px layer footprint this yields approximately 19px cells, fine
// enough for seams to route around individual strokes.
const SELECTION_LONG_SIDE: u32 = 512;
const LONG_SEQUENCE_SELECTION_LONG_SIDE: u32 = 256;
// Streaming still makes one continuous decision for the whole source layer.
// Keeping the analysis bounded preserves memory while preventing independent
// 1024px tiles from choosing incompatible owners along their shared border.
const STREAMING_ANALYSIS_LONG_SIDE: u32 = 1400;
const STREAMING_SELECTION_LONG_SIDE: u32 = 512;
const STREAMING_HARMONIZATION_LONG_SIDE: u32 = 2400;
// Camera-position transitions can span hundreds of native pixels on a
// high-resolution artwork scan.  A narrow correction merely turns a hard
// exposure rectangle into a thin bright/dark halo.  Keep the selected native
// detail untouched, but spread the low-frequency illumination correction far
// enough that the eye no longer sees the footprint of an individual frame.
const STREAMING_HARMONIZATION_RADIUS: f64 = 48.0;
const STREAMING_HARMONIZATION_BLUR: f32 = 48.0;
const GRID_STEP: u32 = 32;
const NATIVE_REFINE_STEP: u32 = 16;
const NATIVE_FIELD_RADIUS: f64 = 32.0;
const OWNERSHIP_MISMATCH_PENALTY: f64 = 2.4;
const STREAMING_TILE_SIZE: u32 = 1024;
// A float RGB canvas costs twelve bytes per pixel before the ownership mask,
// decoded RAW layer, and canonical 16-bit result are counted.  Switch to the
// tiled backing store by canvas area as well as by capture-sequence metadata:
// evidence-only graph alignment intentionally has no synthetic sequence
// bridges, so using that flag alone made large, valid focus mosaics allocate
// the entire sparse bounding box in memory.
const STREAMING_CANVAS_MIN_PIXELS: u64 = 120_000_000;

struct StreamingMosaicStore {
    temp_dir: tempfile::TempDir,
    width: u32,
    height: u32,
    tile_size: u32,
    tile_columns: u32,
    tile_rows: u32,
}

impl StreamingMosaicStore {
    fn new(width: u32, height: u32) -> Result<Self, String> {
        let temp_dir = tempfile::Builder::new()
            .prefix("raw-editor-focus-stack-")
            .tempdir_in(std::env::temp_dir())
            .map_err(|error| format!("Could not create the tiled focus canvas: {error}"))?;
        Ok(Self {
            temp_dir,
            width,
            height,
            tile_size: STREAMING_TILE_SIZE,
            tile_columns: width.div_ceil(STREAMING_TILE_SIZE),
            tile_rows: height.div_ceil(STREAMING_TILE_SIZE),
        })
    }

    fn tile_extent(&self, column: u32, row: u32) -> (u32, u32, u32, u32) {
        let left = column * self.tile_size;
        let top = row * self.tile_size;
        let width = self.tile_size.min(self.width.saturating_sub(left));
        let height = self.tile_size.min(self.height.saturating_sub(top));
        (left, top, width, height)
    }

    fn tile_path(&self, column: u32, row: u32, suffix: &str) -> PathBuf {
        self.temp_dir
            .path()
            .join(format!("tile-{column}-{row}.{suffix}"))
    }

    fn read_f32_tile(
        &self,
        column: u32,
        row: u32,
        width: u32,
        height: u32,
    ) -> Result<Rgb32FImage, String> {
        let path = self.tile_path(column, row, "rgb");
        if !path.exists() {
            return Ok(Rgb32FImage::new(width, height));
        }
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("Could not read focus tile {}: {error}", path.display()))?;
        let expected = width as usize * height as usize * 3;
        if bytes.len() != expected * std::mem::size_of::<f32>() {
            return Err(format!(
                "Focus tile {} has {} bytes; expected {}",
                path.display(),
                bytes.len(),
                expected * std::mem::size_of::<f32>()
            ));
        }
        let mut values = Vec::with_capacity(expected);
        for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Rgb32FImage::from_raw(width, height, values)
            .ok_or_else(|| format!("Could not decode focus tile {}", path.display()))
    }

    fn read_mask_tile(&self, column: u32, row: u32, width: u32, height: u32) -> Vec<u8> {
        let path = self.tile_path(column, row, "mask");
        let expected = width as usize * height as usize;
        match std::fs::read(&path) {
            Ok(bytes) if bytes.len() == expected => bytes,
            _ => vec![0; expected],
        }
    }

    fn read_owner_tile(&self, column: u32, row: u32, width: u32, height: u32) -> Vec<u8> {
        let path = self.tile_path(column, row, "owner");
        let expected = width as usize * height as usize;
        match std::fs::read(&path) {
            Ok(bytes) if bytes.len() == expected => bytes,
            _ => vec![0; expected],
        }
    }

    fn load_owner_tile(&self, column: u32, row: u32) -> GrayImage {
        let (_, _, width, height) = self.tile_extent(column, row);
        GrayImage::from_raw(
            width,
            height,
            self.read_owner_tile(column, row, width, height),
        )
        .expect("focus tile owner dimensions")
    }

    fn save_owner_tile(&self, column: u32, row: u32, owner: &GrayImage) -> Result<(), String> {
        let path = self.tile_path(column, row, "owner");
        std::fs::write(&path, owner.as_raw()).map_err(|error| {
            format!(
                "Could not write focus owner tile {}: {error}",
                path.display()
            )
        })
    }

    fn load_tile(&self, column: u32, row: u32) -> Result<(Rgb32FImage, GrayImage), String> {
        let (_, _, width, height) = self.tile_extent(column, row);
        let rgb = self.read_f32_tile(column, row, width, height)?;
        let mask = GrayImage::from_raw(
            width,
            height,
            self.read_mask_tile(column, row, width, height),
        )
        .expect("focus tile mask dimensions");
        Ok((rgb, mask))
    }

    fn save_tile(
        &self,
        column: u32,
        row: u32,
        rgb: &Rgb32FImage,
        mask: &GrayImage,
    ) -> Result<(), String> {
        let rgb_path = self.tile_path(column, row, "rgb");
        let mut rgb_bytes = Vec::with_capacity(rgb.as_raw().len() * std::mem::size_of::<f32>());
        for &value in rgb.as_raw() {
            rgb_bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&rgb_path, rgb_bytes).map_err(|error| {
            format!("Could not write focus tile {}: {error}", rgb_path.display())
        })?;
        let mask_path = self.tile_path(column, row, "mask");
        std::fs::write(&mask_path, mask.as_raw()).map_err(|error| {
            format!(
                "Could not write focus tile {}: {error}",
                mask_path.display()
            )
        })?;
        Ok(())
    }

    fn analysis_region(
        &self,
        left: u32,
        top: u32,
        scale: f64,
        width: u32,
        height: u32,
    ) -> Result<(Rgb32FImage, GrayImage, GrayImage), String> {
        let mut image = Rgb32FImage::new(width, height);
        let mut mask = GrayImage::new(width, height);
        let mut owner = GrayImage::new(width, height);
        if width == 0 || height == 0 || !scale.is_finite() || scale <= 0.0 {
            return Ok((image, mask, owner));
        }
        let right = (left as f64 + width.saturating_sub(1) as f64 * scale)
            .floor()
            .clamp(0.0, self.width.saturating_sub(1) as f64) as u32;
        let bottom = (top as f64 + height.saturating_sub(1) as f64 * scale)
            .floor()
            .clamp(0.0, self.height.saturating_sub(1) as f64) as u32;
        for tile_row in top / self.tile_size..=bottom / self.tile_size {
            for tile_column in left / self.tile_size..=right / self.tile_size {
                let (tile_left, tile_top, tile_width, tile_height) =
                    self.tile_extent(tile_column, tile_row);
                let (tile, tile_mask) = self.load_tile(tile_column, tile_row)?;
                let tile_owner = self.load_owner_tile(tile_column, tile_row);
                let x_start = (((tile_left as f64 - left as f64) / scale).ceil() as i64)
                    .clamp(0, width as i64) as u32;
                let x_end = ((((tile_left + tile_width) as f64 - left as f64) / scale).ceil()
                    as i64)
                    .clamp(0, width as i64) as u32;
                let y_start = (((tile_top as f64 - top as f64) / scale).ceil() as i64)
                    .clamp(0, height as i64) as u32;
                let y_end = ((((tile_top + tile_height) as f64 - top as f64) / scale).ceil() as i64)
                    .clamp(0, height as i64) as u32;
                for y in y_start..y_end {
                    let global_y = (top as f64 + y as f64 * scale).floor() as u32;
                    if global_y < tile_top || global_y >= tile_top + tile_height {
                        continue;
                    }
                    for x in x_start..x_end {
                        let global_x = (left as f64 + x as f64 * scale).floor() as u32;
                        if global_x < tile_left || global_x >= tile_left + tile_width {
                            continue;
                        }
                        let tile_x = global_x - tile_left;
                        let tile_y = global_y - tile_top;
                        if tile_mask.get_pixel(tile_x, tile_y)[0] == 0 {
                            continue;
                        }
                        image.put_pixel(x, y, *tile.get_pixel(tile_x, tile_y));
                        mask.put_pixel(x, y, Luma([255]));
                        owner.put_pixel(x, y, *tile_owner.get_pixel(tile_x, tile_y));
                    }
                }
            }
        }
        Ok((image, mask, owner))
    }

    fn largest_valid_rectangle(&self) -> Result<Option<(u32, u32, u32, u32)>, String> {
        let mut heights = vec![0u32; self.width as usize];
        let mut best: Option<(u64, u32, u32, u32, u32)> = None;
        for tile_row in 0..self.tile_rows {
            let tile_masks: Vec<Vec<u8>> = (0..self.tile_columns)
                .map(|column| {
                    let (_, _, width, height) = self.tile_extent(column, tile_row);
                    self.read_mask_tile(column, tile_row, width, height)
                })
                .collect();
            let (_, tile_top, _, tile_height) = self.tile_extent(0, tile_row);
            for local_y in 0..tile_height {
                let y = tile_top + local_y;
                let mut x = 0usize;
                for column in 0..self.tile_columns {
                    let (_, _, tile_width, _) = self.tile_extent(column, tile_row);
                    let tile_row_start = local_y as usize * tile_width as usize;
                    let tile_row_end = tile_row_start + tile_width as usize;
                    for &value in &tile_masks[column as usize][tile_row_start..tile_row_end] {
                        if value != 0 {
                            heights[x] = heights[x].saturating_add(1);
                        } else {
                            heights[x] = 0;
                        }
                        x += 1;
                    }
                }
                let mut stack: Vec<usize> = Vec::new();
                for index in 0..=heights.len() {
                    let current = if index == heights.len() {
                        0
                    } else {
                        heights[index]
                    };
                    while let Some(&last) = stack.last() {
                        if heights[last] <= current {
                            break;
                        }
                        stack.pop();
                        let left = stack.last().map_or(0, |&previous| previous + 1);
                        let width = index - left;
                        let height = heights[last];
                        let area = width as u64 * height as u64;
                        if area > best.as_ref().map_or(0, |entry| entry.0) {
                            best = Some((area, left as u32, y + 1 - height, width as u32, height));
                        }
                    }
                    stack.push(index);
                }
            }
        }
        Ok(best.map(|(_, left, top, width, height)| (left, top, width, height)))
    }

    fn materialize(&self, crop: (u32, u32, u32, u32)) -> Result<Rgb32FImage, String> {
        let (crop_left, crop_top, crop_width, crop_height) = crop;
        let mut output = Rgb32FImage::new(crop_width, crop_height);
        for tile_row in 0..self.tile_rows {
            for tile_column in 0..self.tile_columns {
                let (tile_left, tile_top, tile_width, tile_height) =
                    self.tile_extent(tile_column, tile_row);
                let x_start = crop_left.max(tile_left);
                let y_start = crop_top.max(tile_top);
                let x_end = (crop_left + crop_width).min(tile_left + tile_width);
                let y_end = (crop_top + crop_height).min(tile_top + tile_height);
                if x_start >= x_end || y_start >= y_end {
                    continue;
                }
                let (tile, _) = self.load_tile(tile_column, tile_row)?;
                for y in y_start..y_end {
                    let source_y = (y - tile_top) as usize;
                    let destination_y = (y - crop_top) as usize;
                    let source_start =
                        ((x_start - tile_left) as usize * 3) + source_y * tile_width as usize * 3;
                    let source_end = source_start + (x_end - x_start) as usize * 3;
                    let destination_start = (x_start - crop_left) as usize * 3
                        + destination_y * crop_width as usize * 3;
                    let destination_end = destination_start + (x_end - x_start) as usize * 3;
                    output.as_mut()[destination_start..destination_end]
                        .copy_from_slice(&tile.as_raw()[source_start..source_end]);
                }
            }
        }
        Ok(output)
    }
}

fn streaming_seam_harmonization(
    store: &StreamingMosaicStore,
    crop: (u32, u32, u32, u32),
) -> Result<Option<(Rgb32FImage, f64)>, String> {
    let (left, top, width, height) = crop;
    let scale = (width.max(height) as f64 / STREAMING_HARMONIZATION_LONG_SIDE as f64).max(1.0);
    let analysis_width = (width as f64 / scale).ceil() as u32;
    let analysis_height = (height as f64 / scale).ceil() as u32;
    let (analysis, mask, owner) =
        store.analysis_region(left, top, scale, analysis_width, analysis_height)?;
    let mut distance = vec![u32::MAX / 4; (analysis_width * analysis_height) as usize];
    let mut boundary_count = 0usize;
    for y in 0..analysis_height {
        for x in 0..analysis_width {
            let current = owner.get_pixel(x, y)[0];
            if current == 0 || mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let boundary = [
                x.checked_sub(1).map(|nx| (nx, y)),
                (x + 1 < analysis_width).then_some((x + 1, y)),
                y.checked_sub(1).map(|ny| (x, ny)),
                (y + 1 < analysis_height).then_some((x, y + 1)),
            ]
            .into_iter()
            .flatten()
            .any(|(nx, ny)| {
                let neighbour = owner.get_pixel(nx, ny)[0];
                neighbour != 0 && neighbour != current
            });
            if boundary {
                distance[(y * analysis_width + x) as usize] = 0;
                boundary_count += 1;
            }
        }
    }
    if boundary_count == 0 {
        return Ok(None);
    }
    for y in 0..analysis_height {
        for x in 0..analysis_width {
            let index = (y * analysis_width + x) as usize;
            if x > 0 {
                distance[index] = distance[index].min(distance[index - 1].saturating_add(1));
            }
            if y > 0 {
                distance[index] = distance[index]
                    .min(distance[index - analysis_width as usize].saturating_add(1));
            }
        }
    }
    for y in (0..analysis_height).rev() {
        for x in (0..analysis_width).rev() {
            let index = (y * analysis_width + x) as usize;
            if x + 1 < analysis_width {
                distance[index] = distance[index].min(distance[index + 1].saturating_add(1));
            }
            if y + 1 < analysis_height {
                distance[index] = distance[index]
                    .min(distance[index + analysis_width as usize].saturating_add(1));
            }
        }
    }
    // Remove strokes and paper texture before estimating the illumination on
    // either side of an ownership seam.  Only the low-frequency component is
    // feathered; native calligraphy detail remains from exactly one source.
    let local_low = image::imageops::blur(&analysis, 5.0);
    let seamless_low = image::imageops::blur(&local_low, STREAMING_HARMONIZATION_BLUR);
    let gains = Rgb32FImage::from_fn(analysis_width, analysis_height, |x, y| {
        let index = (y * analysis_width + x) as usize;
        if mask.get_pixel(x, y)[0] == 0 {
            return Rgb([1.0, 1.0, 1.0]);
        }
        let normalized = f64::from(distance[index]) / STREAMING_HARMONIZATION_RADIUS;
        let weight = (-0.5 * normalized * normalized).exp();
        let current = local_low.get_pixel(x, y);
        let target = seamless_low.get_pixel(x, y);
        Rgb(std::array::from_fn(|channel| {
            if current[channel] <= 0.015 || target[channel] <= 0.0 {
                return 1.0;
            }
            let log_gain = f64::from(target[channel] / current[channel])
                .ln()
                .clamp(-0.18, 0.18)
                * weight;
            log_gain.exp() as f32
        }))
    });
    println!(
        "  - Low-frequency seam harmonization: {} group-boundary samples at {}x{}",
        boundary_count, analysis_width, analysis_height
    );
    Ok(Some((gains, scale)))
}

fn apply_streaming_seam_harmonization(output: &mut Rgb32FImage, gains: &Rgb32FImage, scale: f64) {
    let width = output.width() as usize;
    output
        .as_mut()
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let gain_y = (y as f64 / scale).clamp(0.0, gains.height().saturating_sub(1) as f64);
            for x in 0..width {
                let gain_x = (x as f64 / scale).clamp(0.0, gains.width().saturating_sub(1) as f64);
                let gain = get_high_quality_interpolated_pixel(gains, gain_x, gain_y);
                for channel in 0..3 {
                    row[x * 3 + channel] = (row[x * 3 + channel] * gain[channel]).clamp(0.0, 1.0);
                }
            }
        });
}

fn luma(p: Rgb<f32>) -> f64 {
    f64::from(p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114)
}

fn rgb_at(image: &Rgb32FImage, x: f64, y: f64) -> Option<Rgb<f32>> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= image.width() as f64
        || y >= image.height() as f64
    {
        return None;
    }
    Some(get_high_quality_interpolated_pixel(image, x, y))
}

#[derive(Clone)]
struct Field<const N: usize> {
    width: usize,
    height: usize,
    step: f64,
    values: Vec<[f64; N]>,
}

impl<const N: usize> Field<N> {
    fn new(width: u32, height: u32, step: f64) -> Self {
        let width = (width as f64 / step).ceil() as usize + 1;
        let height = (height as f64 / step).ceil() as usize + 1;
        Self {
            width,
            height,
            step,
            values: vec![[0.0; N]; width * height],
        }
    }

    fn at(&self, x: f64, y: f64) -> [f64; N] {
        let gx = (x / self.step).clamp(0.0, self.width as f64 - 1.0);
        let gy = (y / self.step).clamp(0.0, self.height as f64 - 1.0);
        let x0 = gx as usize;
        let y0 = gy as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = gx - x0 as f64;
        let fy = gy - y0 as f64;
        std::array::from_fn(|c| {
            let a = self.values[y0 * self.width + x0][c] * (1.0 - fx)
                + self.values[y0 * self.width + x1][c] * fx;
            let b = self.values[y1 * self.width + x0][c] * (1.0 - fx)
                + self.values[y1 * self.width + x1][c] * fx;
            a * (1.0 - fy) + b * fy
        })
    }
}

struct LayerSampler<'a> {
    info: &'a ImageInfo,
    source: &'a Rgb32FImage,
    source_divisor: f64,
    inverse: Matrix3<f64>,
    projection: Projection,
    offset: (f64, f64),
    left: u32,
    top: u32,
    scale: f64,
    residual: Field<2>,
}

impl LayerSampler<'_> {
    fn source_point(&self, x: f64, y: f64) -> Option<Point2<f64>> {
        let delta = self.residual.at(x / self.scale, y / self.scale);
        let target = Point3::new(
            self.left as f64 + x + delta[0] * self.scale - self.offset.0,
            self.top as f64 + y + delta[1] * self.scale - self.offset.1,
            1.0,
        );
        let source = map_target_to_source(&self.inverse, target, self.info, self.projection)?;
        if source.x < 0.0
            || source.y < 0.0
            || source.x >= self.info.width as f64
            || source.y >= self.info.height as f64
        {
            return None;
        }
        Some(source)
    }

    fn sample(&self, x: f64, y: f64) -> Option<Rgb<f32>> {
        let source = self.source_point(x, y)?;
        // Each prefilter level represents the centre of a 2x2 source block.
        // Retain the half-pixel offset to avoid a lens-dependent sampling shift.
        let sx = ((source.x + 0.5) / self.source_divisor - 0.5)
            .clamp(0.0, self.source.width() as f64 - 1.0);
        let sy = ((source.y + 0.5) / self.source_divisor - 0.5)
            .clamp(0.0, self.source.height() as f64 - 1.0);
        rgb_at(self.source, sx, sy)
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn analysis_pair(
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

fn refine_layer(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
) {
    let (a, b, valid) = analysis_pair(base, base_mask, sampler, width, height);
    refine_layer_from_analysis(&a, &b, &valid, sampler);
}

fn refine_layer_from_analysis(
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

fn streaming_candidate_analysis(
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

fn streaming_refinement_pair(
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

fn streaming_tone_field_from_analysis(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
) -> Field<3> {
    let width = base.width().min(candidate.width());
    let height = base.height().min(candidate.height());
    let mut field = Field::new(width, height, 112.0);
    let mut samples = vec![Vec::<[f64; 3]>::new(); field.values.len()];
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            if base_mask.get_pixel(x, y)[0] == 0 || candidate_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let current = *base.get_pixel(x, y);
            let source = *candidate.get_pixel(x, y);
            if current
                .0
                .iter()
                .chain(source.0.iter())
                .any(|value| !(0.015..0.97).contains(value))
            {
                continue;
            }
            let delta =
                std::array::from_fn(|channel| f64::from(current[channel] / source[channel]).ln());
            if delta.iter().any(|value| value.abs() > 1.2) {
                continue;
            }
            let field_x = ((x as f64 / field.step).round() as usize).min(field.width - 1);
            let field_y = ((y as f64 / field.step).round() as usize).min(field.height - 1);
            samples[field_y * field.width + field_x].push(delta);
        }
    }
    let observations = samples
        .iter()
        .enumerate()
        .filter(|(_, values)| values.len() >= 4)
        .map(|(index, values)| {
            let value: [f64; 3] = std::array::from_fn(|channel| {
                median(
                    &mut values
                        .iter()
                        .map(|delta| delta[channel])
                        .collect::<Vec<_>>(),
                )
            });
            (index, value)
        })
        .collect::<Vec<_>>();
    let global: [f64; 3] = std::array::from_fn(|channel| {
        median(
            &mut samples
                .iter()
                .flatten()
                .map(|delta| delta[channel])
                .collect::<Vec<_>>(),
        )
    });
    for field_y in 0..field.height {
        for field_x in 0..field.width {
            let mut total = 0.02f64;
            let mut sum = global.map(|value| value * total);
            for (index, value) in &observations {
                let dx = field_x as f64 - (index % field.width) as f64;
                let dy = field_y as f64 - (index / field.width) as f64;
                let weight = (-(dx * dx + dy * dy) / 2.0).exp();
                for channel in 0..3 {
                    sum[channel] += value[channel] * weight;
                }
                total += weight;
            }
            field.values[field_y * field.width + field_x] =
                std::array::from_fn(|channel| (sum[channel] / total).clamp(-0.9, 0.9));
        }
    }
    field
}

fn masked_rgb_at(image: &Rgb32FImage, mask: &GrayImage, x: f64, y: f64) -> Option<Rgb<f32>> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= image.width() as f64
        || y >= image.height() as f64
        || mask.get_pixel(x as u32, y as u32)[0] == 0
    {
        return None;
    }
    rgb_at(image, x, y)
}

fn streaming_analysis_disagreement(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    tone: &Field<3>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    let mut values = Vec::new();
    for y_offset in [0.2, 0.5, 0.8] {
        for x_offset in [0.2, 0.5, 0.8] {
            let sample_x = x + cell_size * x_offset;
            let sample_y = y + cell_size * y_offset;
            let Some(current) = masked_rgb_at(base, base_mask, sample_x, sample_y) else {
                continue;
            };
            let Some(source) = masked_rgb_at(candidate, candidate_mask, sample_x, sample_y) else {
                continue;
            };
            let source = adjusted(source, tone.at(sample_x, sample_y));
            values.push(
                (0..3)
                    .map(|channel| f64::from((current[channel] - source[channel]).abs()))
                    .sum::<f64>()
                    / 3.0,
            );
        }
    }
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let high = values[(values.len() * 3 / 4).min(values.len() - 1)];
    (mean * 0.45 + high * 0.55).clamp(0.0, 1.0)
}

fn streaming_ownership_from_analysis(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    tone: &Field<3>,
    base_owner: &GrayImage,
    capture_group_id: u8,
    panorama_transition: bool,
) -> GrayImage {
    let width = base.width().min(candidate.width());
    let height = base.height().min(candidate.height());
    let size = (width.max(height) as f64 / STREAMING_SELECTION_LONG_SIDE.max(1) as f64).max(1.0);
    let grid_width = (width as f64 / size).ceil() as u32;
    let grid_height = (height as f64 / size).ceil() as u32;
    let count = (grid_width * grid_height) as usize;
    let mut preference = vec![0.0f64; count];
    let mut disagreement = vec![0.0f64; count];
    let mut fixed = vec![0i8; count];
    let mut mismatch = vec![0.0f64; count];
    let mut energy_sum = 0.0;
    let mut energy_count = 0usize;
    for grid_y in 0..grid_height {
        for grid_x in 0..grid_width {
            let index = (grid_y * grid_width + grid_x) as usize;
            let x = grid_x as f64 * size;
            let y = grid_y as f64 * size;
            let center_x = x + size * 0.5;
            let center_y = y + size * 0.5;
            let candidate_valid = masked_rgb_at(
                candidate,
                candidate_mask,
                center_x.min(width.saturating_sub(1) as f64),
                center_y.min(height.saturating_sub(1) as f64),
            )
            .is_some();
            if !candidate_valid {
                fixed[index] = -1;
                continue;
            }
            let base_valid = masked_rgb_at(
                base,
                base_mask,
                center_x.min(width.saturating_sub(1) as f64),
                center_y.min(height.saturating_sub(1) as f64),
            )
            .is_some();
            if !base_valid {
                fixed[index] = 1;
                continue;
            }
            let owner_x = center_x
                .floor()
                .clamp(0.0, base_owner.width().saturating_sub(1) as f64)
                as u32;
            let owner_y = center_y
                .floor()
                .clamp(0.0, base_owner.height().saturating_sub(1) as f64)
                as u32;
            // Once a panorama position owns a pixel, later focal planes from
            // another camera position may not reclaim it merely because a
            // repeated brush stroke scores as sharper.  A new camera group is
            // introduced once through a connected panorama seam; subsequent
            // members only compete inside that group's owned region.
            if !panorama_transition && base_owner.get_pixel(owner_x, owner_y)[0] != capture_group_id
            {
                fixed[index] = -1;
                continue;
            }
            let source_boundary = [(-size, 0.0), (size, 0.0), (0.0, -size), (0.0, size)]
                .iter()
                .any(|(dx, dy)| {
                    masked_rgb_at(candidate, candidate_mask, center_x + dx, center_y + dy).is_none()
                });
            if source_boundary {
                fixed[index] = -1;
                continue;
            }
            disagreement[index] = streaming_analysis_disagreement(
                base,
                base_mask,
                candidate,
                candidate_mask,
                tone,
                x,
                y,
                size,
            );
            if panorama_transition {
                // Cross-position stitching is a seam-placement problem, not
                // a focus contest. Prefer the established panorama except for
                // the connected region needed to admit genuinely new source
                // coverage. This prevents rectangular islands and doubled
                // characters inside a broad overlap.
                preference[index] = -0.75;
                continue;
            }
            let base_focus = streaming_cell_focus(
                |sample_x, sample_y| masked_rgb_at(base, base_mask, sample_x, sample_y),
                center_x,
                center_y,
                size,
            );
            let candidate_focus = streaming_cell_focus(
                |sample_x, sample_y| {
                    masked_rgb_at(candidate, candidate_mask, sample_x, sample_y)
                        .map(|pixel| adjusted(pixel, tone.at(sample_x, sample_y)))
                },
                center_x,
                center_y,
                size,
            );
            preference[index] = candidate_focus.powi(2) - base_focus.powi(2) * 1.06;
            energy_sum += (candidate_focus.powi(2) + base_focus.powi(2)) * 0.5;
            energy_count += 1;
            let mismatch_scale = if candidate_focus > base_focus * 1.12 {
                0.30
            } else {
                1.0
            };
            mismatch[index] =
                (disagreement[index] * OWNERSHIP_MISMATCH_PENALTY * mismatch_scale).min(1.0);
        }
    }
    if !panorama_transition {
        let energy_scale = (energy_sum / energy_count.max(1) as f64).max(1e-6);
        for (index, value) in preference.iter_mut().enumerate() {
            if fixed[index] == 0 {
                *value = (*value / energy_scale).clamp(-12.0, 12.0) - mismatch[index];
            }
        }
    }
    let labels = seam_cut::cut_grid(
        grid_width as usize,
        grid_height as usize,
        &preference,
        &disagreement,
        &fixed,
    );
    GrayImage::from_raw(grid_width, grid_height, labels).expect("ownership dimensions")
}

fn refine_native_layer(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
) {
    if sampler.scale <= 2.0 {
        return;
    }
    // Refine against the selected, native output samples. Upsampling a coarse
    // displacement field alone leaves several pixels of error on 200MP input.
    // These tiny patches keep native refinement independent of canvas size.
    for spacing in [2.0, 1.0] {
        let mut observations = Vec::<(Point2<f64>, [f64; 2], f64)>::new();
        for y in (20..height.saturating_sub(20)).step_by(NATIVE_REFINE_STEP as usize) {
            for x in (20..width.saturating_sub(20)).step_by(NATIVE_REFINE_STEP as usize) {
                let lx = x as f64 * sampler.scale;
                let ly = y as f64 * sampler.scale;
                let mut a = GrayImage::new(57, 57);
                let mut b = GrayImage::new(57, 57);
                let mut valid = true;
                'patch: for py in 0..57 {
                    for px in 0..57 {
                        let dx = (px as f64 - 28.0) * spacing;
                        let dy = (py as f64 - 28.0) * spacing;
                        let gx = sampler.left as f64 + lx + dx;
                        let gy = sampler.top as f64 + ly + dy;
                        let Some(current) = rgb_at(base, gx, gy) else {
                            valid = false;
                            break 'patch;
                        };
                        if base_mask.get_pixel(gx as u32, gy as u32)[0] == 0 {
                            valid = false;
                            break 'patch;
                        }
                        let Some(candidate) = sampler.sample(lx + dx, ly + dy) else {
                            valid = false;
                            break 'patch;
                        };
                        a.put_pixel(
                            px,
                            py,
                            Luma([(luma(current) * 255.0).round().clamp(0.0, 255.0) as u8]),
                        );
                        b.put_pixel(
                            px,
                            py,
                            Luma([(luma(candidate) * 255.0).round().clamp(0.0, 255.0) as u8]),
                        );
                    }
                }
                if !valid {
                    continue;
                }
                let p = Point2::new(28.0, 28.0);
                let Some(found) = refine_warped_patch(&a, &b, &Matrix3::identity(), p, 10, 7)
                else {
                    continue;
                };
                if found.correlation < 0.90 {
                    continue;
                }
                let Some(back) =
                    refine_warped_patch(&b, &a, &Matrix3::identity(), found.target, 10, 7)
                else {
                    continue;
                };
                if (back.target - p).norm() > 0.40 {
                    continue;
                }
                let d = (found.target - p) * (spacing / sampler.scale);
                observations.push((
                    Point2::new(x as f64, y as f64),
                    [d.x, d.y],
                    found.correlation,
                ));
            }
        }
        if observations.len() < 6 {
            continue;
        }
        let mut field = sampler.residual.clone();
        for gy in 0..field.height {
            for gx in 0..field.width {
                let p = Point2::new(gx as f64 * field.step, gy as f64 * field.step);
                let mut sum = [0.0; 2];
                let mut total = 0.0f64;
                for (q, d, correlation) in &observations {
                    let weight = (-(p - q).norm_squared()
                        / (2.0 * NATIVE_FIELD_RADIUS * NATIVE_FIELD_RADIUS))
                        .exp()
                        * correlation.powi(8);
                    sum[0] += weight * d[0];
                    sum[1] += weight * d[1];
                    total += weight;
                }
                for (c, value) in sum.into_iter().enumerate() {
                    field.values[gy * field.width + gx][c] += value / total.max(0.1);
                }
            }
        }
        sampler.residual = field;
        let correction = median(
            &mut observations
                .iter()
                .map(|(_, d, _)| d[0].hypot(d[1]) * sampler.scale)
                .collect::<Vec<_>>(),
        );
        println!(
            "    - Native refinement ({spacing:.0}px samples): {} patches, median correction {correction:.2}px",
            observations.len()
        );
    }
}

/// Smooth per-channel log gain. Corrections are estimated from matched overlap
/// blocks; fine texture never becomes a correction field. Robust medians keep
/// an occlusion or a displaced contour from tinting its surrounding region.
fn tone_field(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
) -> Field<3> {
    tone_field_with_sampling(base, base_mask, sampler, width, height, 56.0, 4, 8)
}

#[allow(clippy::too_many_arguments)]
fn tone_field_with_sampling(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
    field_step: f64,
    sample_step: usize,
    minimum_samples: usize,
) -> Field<3> {
    let mut field = Field::new(width, height, field_step);
    let mut samples = vec![Vec::<[f64; 3]>::new(); field.values.len()];
    for y in (0..height).step_by(sample_step) {
        for x in (0..width).step_by(sample_step) {
            let lx = x as f64 * sampler.scale;
            let ly = y as f64 * sampler.scale;
            let gx = (sampler.left as f64 + lx) as u32;
            let gy = (sampler.top as f64 + ly) as u32;
            if gx >= base.width() || gy >= base.height() || base_mask.get_pixel(gx, gy)[0] == 0 {
                continue;
            }
            let Some(candidate) = sampler.sample(lx, ly) else {
                continue;
            };
            let current = base.get_pixel(gx, gy);
            if current
                .0
                .iter()
                .chain(candidate.0.iter())
                .any(|v| !(0.015..0.97).contains(v))
            {
                continue;
            }
            let d = std::array::from_fn(|c| f64::from(current[c] / candidate[c]).ln());
            if d.iter().any(|v| v.abs() > 1.2) {
                continue;
            }
            let ix = ((x as f64 / field.step).round() as usize).min(field.width - 1);
            let iy = ((y as f64 / field.step).round() as usize).min(field.height - 1);
            samples[iy * field.width + ix].push(d);
        }
    }
    let observations = samples
        .iter()
        .enumerate()
        .filter(|(_, v)| v.len() >= minimum_samples)
        .map(|(i, v)| {
            let value =
                std::array::from_fn(|c| median(&mut v.iter().map(|d| d[c]).collect::<Vec<_>>()));
            (i, value)
        })
        .collect::<Vec<(usize, [f64; 3])>>();
    let global: [f64; 3] = std::array::from_fn(|c| {
        median(
            &mut samples
                .iter()
                .flatten()
                .map(|value| value[c])
                .collect::<Vec<_>>(),
        )
    });
    for iy in 0..field.height {
        for ix in 0..field.width {
            // The overlap still supplies a reliable frame-wide exposure when
            // the newly covered side has no local samples. Returning to unity
            // there would put the original exposure step back into the result.
            let mut total = 0.02f64;
            let mut sum = global.map(|gain| gain * total);
            for (i, value) in &observations {
                let dx = ix as f64 - (i % field.width) as f64;
                let dy = iy as f64 - (i / field.width) as f64;
                let weight = (-(dx * dx + dy * dy) / 2.0).exp();
                for c in 0..3 {
                    sum[c] += value[c] * weight;
                }
                total += weight;
            }
            // Extrapolate continuously over the nearby non-overlap border,
            // while bounding unsupported changes far outside the overlap.
            field.values[iy * field.width + ix] =
                std::array::from_fn(|c| (sum[c] / total.max(0.001)).clamp(-0.9, 0.9));
        }
    }
    field
}

fn adjusted(pixel: Rgb<f32>, delta: [f64; 3]) -> Rgb<f32> {
    Rgb(std::array::from_fn(|c| {
        (pixel[c] * delta[c].exp() as f32).clamp(0.0, 1.0)
    }))
}

// Native focus evidence after a separable low-pass filter. Single-pixel
// gradients confuse sensor grain with detail and miss faded coloured marks
// whose luminance nearly matches the substrate. Keep luminance and two colour
// differences, suppress pixel noise in both axes, then measure coherent edges.
fn acutance(mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>, x: f64, y: f64) -> f64 {
    let mut patch = [[[0.0; 3]; 13]; 13];
    for (j, row) in patch.iter_mut().enumerate() {
        for (i, value) in row.iter_mut().enumerate() {
            let Some(p) = sample(x + i as f64 - 6.0, y + j as f64 - 6.0) else {
                return 0.0;
            };
            *value = [
                luma(p),
                f64::from(p[0] - p[1]) * 0.5,
                f64::from(p[2] - p[1]) * 0.5,
            ];
        }
    }
    const KERNEL: [f64; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let mut horizontal = [[[0.0; 3]; 9]; 13];
    for y in 0..13 {
        for x in 0..9 {
            for c in 0..3 {
                horizontal[y][x][c] = (0..5).map(|k| patch[y][x + k][c] * KERNEL[k]).sum();
            }
        }
    }
    let mut smooth = [[[0.0; 3]; 9]; 9];
    for y in 0..9 {
        for x in 0..9 {
            for c in 0..3 {
                smooth[y][x][c] = (0..5).map(|k| horizontal[y + k][x][c] * KERNEL[k]).sum();
            }
        }
    }
    let mut energy = 0.0;
    let mut level = 0.0;
    for y in 2..7 {
        for x in 2..7 {
            for c in 0..3 {
                let dx = (smooth[y][x + 2][c] - smooth[y][x - 2][c]) * 0.25;
                let dy = (smooth[y + 2][x][c] - smooth[y - 2][x][c]) * 0.25;
                energy += dx * dx + dy * dy;
            }
            level += smooth[y][x][0];
        }
    }
    (energy / 25.0).sqrt() / (level / 25.0).max(0.04)
}

/// Evaluate focus at several locations inside an ownership cell. The old
/// single centre sample could land on quiet paper while the candidate held a
/// sharp brush stroke a few pixels away. Average the two strongest spatial
/// probes so sparse detail contributes; each probe filters pixel noise before
/// measuring its edge response.
fn cell_focus(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    let radius = (cell_size * 0.28).clamp(2.0, 18.0);
    // Five probes cover the centre and all four corners. A stroke can cross
    // any cell edge, so sampling only one diagonal can still land entirely
    // on quiet paper and hide an in-focus candidate.
    let offsets = [
        (-radius, -radius),
        (radius, -radius),
        (0.0, 0.0),
        (-radius, radius),
        (radius, radius),
    ];
    let mut values = offsets
        .iter()
        .filter_map(|(dx, dy)| Some(acutance(&mut sample, x + dx, y + dy)))
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    // Preserve sparse in-focus strokes even when other probes see only paper.
    let first = values.len().saturating_sub(2);
    (values[first] + values[first + 1.min(values.len() - 1)]) * 0.5
}

// Streaming stacks may evaluate hundreds of thousands of ownership cells.
// Measure the same five spatial probes with coherent, low-pass cross
// gradients instead of rebuilding a full 13x13 patch for every probe.  Each
// gradient averages along its tangent before measuring energy, so isolated
// sensor noise does not win over a focused brush edge.
fn streaming_acutance(
    sample: &mut impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
) -> Option<f64> {
    const TANGENT: [f64; 3] = [-2.0, 0.0, 2.0];
    const GRADIENTS: [(f64, f64); 2] = [(-3.0, 1.0), (-1.0, 3.0)];
    let mut energy = 0.0;
    let mut level = 0.0;
    let mut level_count = 0.0f64;
    for (near, far) in GRADIENTS {
        let mut dx = [0.0; 3];
        let mut dy = [0.0; 3];
        for tangent in TANGENT {
            let left = sample(x + near, y + tangent)?;
            let right = sample(x + far, y + tangent)?;
            let top = sample(x + tangent, y + near)?;
            let bottom = sample(x + tangent, y + far)?;
            let channels = |pixel: Rgb<f32>| {
                [
                    luma(pixel),
                    f64::from(pixel[0] - pixel[1]) * 0.5,
                    f64::from(pixel[2] - pixel[1]) * 0.5,
                ]
            };
            let left_channels = channels(left);
            let right_channels = channels(right);
            let top_channels = channels(top);
            let bottom_channels = channels(bottom);
            for channel in 0..3 {
                dx[channel] += right_channels[channel] - left_channels[channel];
                dy[channel] += bottom_channels[channel] - top_channels[channel];
            }
            level += luma(left) + luma(right) + luma(top) + luma(bottom);
            level_count += 4.0;
        }
        for channel in 0..3 {
            let dx = dx[channel] / TANGENT.len() as f64;
            let dy = dy[channel] / TANGENT.len() as f64;
            energy += dx * dx + dy * dy;
        }
    }
    Some((energy / (GRADIENTS.len() * 3) as f64).sqrt() / (level / level_count.max(1.0)).max(0.04))
}

fn streaming_cell_focus(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    let radius = (cell_size * 0.28).clamp(2.0, 18.0);
    let mut values = [
        (-radius, -radius),
        (radius, -radius),
        (0.0, 0.0),
        (-radius, radius),
        (radius, radius),
    ]
    .into_iter()
    .filter_map(|(dx, dy)| streaming_acutance(&mut sample, x + dx, y + dy))
    .filter(|value| value.is_finite())
    .collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let first = values.len().saturating_sub(2);
    (values[first] + values[(first + 1).min(values.len() - 1)]) * 0.5
}

fn ownership_disagreement(
    base: &Rgb32FImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    ax: f64,
    ay: f64,
    cell_size: f64,
) -> f64 {
    // A single sample at the cell centre can land on canvas and miss a brush
    // stroke that crosses the same cell near an edge. Sample the whole cell
    // instead, and retain a high percentile so a displaced contour vetoes the
    // candidate even when most of the cell is quiet paper.
    const OFFSETS: [f64; 5] = [0.15, 0.325, 0.5, 0.675, 0.85];
    let side = OFFSETS.len();
    let mut samples = vec![None; side * side];
    for (yi, y_offset) in OFFSETS.iter().copied().enumerate() {
        for (xi, x_offset) in OFFSETS.iter().copied().enumerate() {
            let sample_index = yi * side + xi;
            let sample_ax = ax + cell_size * x_offset;
            let sample_ay = ay + cell_size * y_offset;
            let lx = sample_ax * sampler.scale;
            let ly = sample_ay * sampler.scale;
            let gx = sampler.left as f64 + lx;
            let gy = sampler.top as f64 + ly;
            let Some(candidate) = sampler
                .sample(lx, ly)
                .map(|pixel| adjusted(pixel, tone.at(sample_ax, sample_ay)))
            else {
                continue;
            };
            let Some(base_pixel) = rgb_at(base, gx, gy) else {
                continue;
            };
            let candidate = candidate.0.map(f64::from);
            let base_pixel = base_pixel.0.map(f64::from);
            if candidate
                .iter()
                .chain(base_pixel.iter())
                .all(|v| v.is_finite())
            {
                samples[sample_index] = Some((candidate, base_pixel));
            }
        }
    }
    if samples.is_empty() {
        return 1.0;
    }
    // Compare low-pass colour evidence rather than individual high-frequency
    // pixels. Defocus changes brush-edge samples substantially even when the
    // geometry is correct; a displaced contour still changes the local mean
    // over a 3×3 neighbourhood and remains penalised.
    let side = side as isize;
    let mut disagreements = Vec::with_capacity(samples.len());
    for index in 0..samples.len() {
        let row = index as isize / side;
        let column = index as isize % side;
        let mut candidate_sum = [0.0; 3];
        let mut base_sum = [0.0; 3];
        let mut count = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let y = row + dy;
                let x = column + dx;
                if y < 0 || x < 0 || y >= side || x >= side {
                    continue;
                }
                let neighbour = y as usize * OFFSETS.len() + x as usize;
                let Some(Some((candidate, base))) = samples.get(neighbour) else {
                    continue;
                };
                for channel in 0..3 {
                    candidate_sum[channel] += candidate[channel];
                    base_sum[channel] += base[channel];
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            disagreements.push(
                (0..3)
                    .map(|channel| (candidate_sum[channel] - base_sum[channel]).abs() / count)
                    .sum::<f64>()
                    / 3.0,
            );
        }
    }
    if disagreements.is_empty() {
        return 1.0;
    }
    disagreements.sort_by(f64::total_cmp);
    let mean = disagreements.iter().sum::<f64>() / disagreements.len() as f64;
    let high = disagreements[(disagreements.len() * 3 / 4).min(disagreements.len() - 1)];
    (mean * 0.45 + high * 0.55).clamp(0.0, 1.0)
}

/// Binary ownership regularised on a bounded grid. Pairwise terms penalise
/// seams through high contrast or disagreement. Fixed coverage labels ensure
/// every newly observed pixel is admitted, including fully enclosed detail
/// frames which a one-direction panorama seam can otherwise discard.
fn ownership(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    width: u32,
    height: u32,
) -> GrayImage {
    ownership_with_long_side(
        base,
        base_mask,
        sampler,
        tone,
        width,
        height,
        SELECTION_LONG_SIDE,
    )
}

fn ownership_with_long_side(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    width: u32,
    height: u32,
    selection_long_side: u32,
) -> GrayImage {
    let size = (width.max(height) as f64 / selection_long_side.max(1) as f64).max(1.0);
    ownership_grid(base, base_mask, sampler, tone, width, height, size, false)
}

fn streaming_ownership_with_long_side(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    width: u32,
    height: u32,
    selection_long_side: u32,
) -> GrayImage {
    let size = (width.max(height) as f64 / selection_long_side.max(1) as f64).max(1.0);
    ownership_grid(base, base_mask, sampler, tone, width, height, size, true)
}

fn ownership_grid(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    width: u32,
    height: u32,
    size: f64,
    streaming: bool,
) -> GrayImage {
    let w = (width as f64 / size).ceil() as u32;
    let h = (height as f64 / size).ceil() as u32;
    let count = (w * h) as usize;
    let mut preference = vec![0.0f64; count];
    let mut disagreement = vec![0.0f64; count];
    let mut fixed = vec![0i8; count];
    let mut energy_sum = 0.0;
    let mut energy_count = 0usize;
    let mut mismatch = vec![0.0f64; count];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let ax = x as f64 * size;
            let ay = y as f64 * size;
            let lx = ax * sampler.scale;
            let ly = ay * sampler.scale;
            let gx = sampler.left as f64 + lx;
            let gy = sampler.top as f64 + ly;
            let base_valid = gx >= 0.0
                && gy >= 0.0
                && gx < base.width() as f64
                && gy < base.height() as f64
                && base_mask.get_pixel(gx as u32, gy as u32)[0] > 0;
            let candidate = sampler.sample(lx, ly);
            if candidate.is_none() {
                fixed[i] = -1;
                continue;
            }
            if !base_valid {
                fixed[i] = 1;
                continue;
            }
            // Keep a one-cell guard at the *photographic source* boundary.
            // `width`/`height` can describe a streaming backing tile rather
            // than the source footprint; treating every tile edge as a source
            // edge forced the old owner into a visible 1024px square grid.
            let native_cell = size * sampler.scale;
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                let source_boundary = [
                    (-native_cell, 0.0),
                    (native_cell, 0.0),
                    (0.0, -native_cell),
                    (0.0, native_cell),
                ]
                .iter()
                .any(|(dx, dy)| sampler.sample(lx + dx, ly + dy).is_none());
                if source_boundary {
                    fixed[i] = -1;
                    continue;
                }
            }
            let base_focus = if streaming {
                streaming_cell_focus(
                    |x, y| rgb_at(base, x, y),
                    gx + native_cell * 0.5,
                    gy + native_cell * 0.5,
                    native_cell,
                )
            } else {
                cell_focus(
                    |x, y| rgb_at(base, x, y),
                    gx + native_cell * 0.5,
                    gy + native_cell * 0.5,
                    native_cell,
                )
            };
            let candidate_focus = if streaming {
                streaming_cell_focus(
                    |x, y| sampler.sample(x, y).map(|p| adjusted(p, tone.at(ax, ay))),
                    lx + native_cell * 0.5,
                    ly + native_cell * 0.5,
                    native_cell,
                )
            } else {
                cell_focus(
                    |x, y| sampler.sample(x, y).map(|p| adjusted(p, tone.at(ax, ay))),
                    lx + native_cell * 0.5,
                    ly + native_cell * 0.5,
                    native_cell,
                )
            };
            // Ties retain existing detail, making duplicate/identical frames
            // idempotent. Weak texture never wins solely from sensor noise.
            // Compare edge energy rather than per-cell log ratios. Defocus
            // spreads a contour over more cells: ratios give its weak halo
            // as many votes as a focused edge and can reject the sharp frame.
            // Energy keeps quiet substrate votes small; the relative margin
            // retains an existing source when its detail is equivalent.
            preference[i] = candidate_focus.powi(2) - base_focus.powi(2) * 1.06;
            energy_sum += (candidate_focus.powi(2) + base_focus.powi(2)) * 0.5;
            energy_count += 1;
            disagreement[i] = ownership_disagreement(base, sampler, tone, ax, ay, size);
            // A sharp but displaced candidate can win the acutance test even
            // though it would put a second contour next to the existing
            // stroke. Penalise that candidate directly; otherwise the graph
            // cut can legally place a seam through the high-contrast subject
            // and leave a visible rectangular ghost.
            // Defocus changes high-frequency samples even when geometry is
            // correct. When the candidate is demonstrably sharper, discount
            // that photometric disagreement; a sharp source must be allowed
            // to replace a soft source instead of being vetoed as a ghost.
            let mismatch_scale = if candidate_focus > base_focus * 1.12 {
                0.30
            } else {
                1.0
            };
            mismatch[i] = (disagreement[i] * OWNERSHIP_MISMATCH_PENALTY * mismatch_scale).min(1.0);
        }
    }
    // One shared scale preserves energy comparisons across the overlap. A
    // cell-local denominator would again amplify weak paper/blur-halo votes.
    let energy_scale = (energy_sum / energy_count.max(1) as f64).max(1e-6);
    for i in 0..count {
        preference[i] = (preference[i] / energy_scale).clamp(-12.0, 12.0) - mismatch[i];
    }
    // Solve the ownership globally so a seam can route around a contour and a
    // contained frame can be admitted without a block-shaped boundary.
    let labels = seam_cut::cut_grid(w as usize, h as usize, &preference, &disagreement, &fixed);
    GrayImage::from_raw(w, h, labels).expect("ownership dimensions")
}

pub(super) fn detail_preserving_mosaic<R: Runtime, F>(
    images: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    sequence_gap_aware: bool,
    app: AppHandle<R>,
    event: &str,
    load: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let (min_x, max_x, min_y, max_y) = output_bounds(images, homographies, projection);
    if ![min_x, max_x, min_y, max_y].iter().all(|x| x.is_finite()) {
        return Err("The stack does not have a finite aligned canvas.".into());
    }
    let (offset_x, width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, height) = pixel_aligned_canvas(min_y, max_y);
    let canvas_pixels = u64::from(width).saturating_mul(u64::from(height));
    #[cfg(test)]
    let force_streaming = std::env::var_os("RAW_EDITOR_FORCE_STREAMING_MOSAIC").is_some();
    #[cfg(not(test))]
    let force_streaming = false;
    let capture_group_count = capture_group_ids
        .map(|ids| ids.values().copied().collect::<HashSet<_>>().len())
        .unwrap_or(0);
    if force_streaming
        || capture_group_count > 1
        || (sequence_gap_aware && images.len() > 8)
        || canvas_pixels > STREAMING_CANVAS_MIN_PIXELS
    {
        return detail_preserving_mosaic_streaming(
            images,
            homographies,
            projection,
            capture_group_ids,
            app,
            event,
            load,
            offset_x,
            offset_y,
            width,
            height,
        );
    }
    let mut result = Rgb32FImage::new(width, height);
    let mut mask = GrayImage::new(width, height);
    let mut covered_bounds: Option<(u32, u32, u32, u32)> = None;
    let local_refinement_enabled = !sequence_gap_aware || images.len() <= 8;
    println!("  - Detail-preserving mosaic canvas: {width}x{height}");
    for (index, &info) in images.iter().enumerate() {
        let _ = app.emit(
            event,
            format!(
                "Refining and selecting detail {} of {}",
                index + 1,
                images.len()
            ),
        );
        let mut source = load(info)?;
        let transform = &homographies[&info.id];
        let mut source_divisor = 1.0;
        if projection == Projection::Planar {
            let center = Point3::new(info.width as f64 * 0.5, info.height as f64 * 0.5, 1.0);
            let c = transform * center;
            let dx = transform * (center + nalgebra::Vector3::new(1.0, 0.0, 0.0));
            let dy = transform * (center + nalgebra::Vector3::new(0.0, 1.0, 0.0));
            let sx = (dx.x / dx.z - c.x / c.z).hypot(dx.y / dx.z - c.y / c.z);
            let sy = (dy.x / dy.z - c.x / c.z).hypot(dy.y / dy.z - c.y / c.z);
            let scale = sx.min(sy);
            // Prefilter a telephoto tile before minification. Otherwise weave
            // aliases into false sharpness and defeats native focus selection.
            while scale.is_finite()
                && scale > 0.0
                && scale * source_divisor < 0.67
                && source.width().min(source.height()) > 8
            {
                source = downsample_rgb_half(&source);
                source_divisor *= 2.0;
            }
        }
        let inverse = transform
            .try_inverse()
            .ok_or("The stack contains a non-invertible alignment.")?;
        let Some((left, right, top, bottom)) = transformed_image_region(
            info, transform, projection, offset_x, offset_y, width, height,
        ) else {
            continue;
        };
        let layer_width = right - left + 1;
        let layer_height = bottom - top + 1;
        let scale = (layer_width.max(layer_height) as f64 / ANALYSIS_LONG_SIDE as f64).max(1.0);
        let aw = (layer_width as f64 / scale).ceil() as u32;
        let ah = (layer_height as f64 / scale).ceil() as u32;
        let mut sampler = LayerSampler {
            info,
            source: &source,
            source_divisor,
            inverse,
            projection,
            offset: (offset_x, offset_y),
            left,
            top,
            scale,
            residual: Field::new(aw, ah, GRID_STEP as f64),
        };
        println!("  - Selecting '{}'", info.filename);

        let layer_overlaps_existing = covered_bounds.is_some_and(
            |(covered_left, covered_right, covered_top, covered_bottom)| {
                let overlap_width = right
                    .min(covered_right)
                    .saturating_sub(left.max(covered_left));
                let overlap_height = bottom
                    .min(covered_bottom)
                    .saturating_sub(top.max(covered_top));
                overlap_width >= 128 && overlap_height >= 128
            },
        );
        if sequence_gap_aware && !layer_overlaps_existing {
            // A filename bridge explicitly means that these captures are
            // adjacent but have no measured overlap. Do not spend the costly
            // residual/graph-cut pass trying to register empty canvas against
            // an unrelated block; copy the source tile at its verified global
            // pose, preserving every native detail pixel.
            println!("    - Sequence gap: copying native detail without overlap refinement");
            copy_sequence_gap_layer(&mut result, &mut mask, &sampler, left, right, top, bottom);
            covered_bounds = Some(match covered_bounds {
                Some((covered_left, covered_right, covered_top, covered_bottom)) => (
                    covered_left.min(left),
                    covered_right.max(right),
                    covered_top.min(top),
                    covered_bottom.max(bottom),
                ),
                None => (left, right, top, bottom),
            });
            continue;
        }
        if index > 0 && local_refinement_enabled {
            refine_layer(&result, &mask, &mut sampler, aw, ah);
            refine_layer(&result, &mask, &mut sampler, aw, ah);
            refine_native_layer(&result, &mask, &mut sampler, aw, ah);
        } else if index > 0 && sequence_gap_aware {
            println!("    - Long sequence: using verified global alignment for focus ownership");
        }
        let tone = if index > 0 {
            tone_field(&result, &mask, &sampler, aw, ah)
        } else {
            Field::new(aw, ah, 56.0)
        };
        let decision = if sequence_gap_aware && images.len() > 8 {
            ownership_with_long_side(
                &result,
                &mask,
                &sampler,
                &tone,
                aw,
                ah,
                LONG_SEQUENCE_SELECTION_LONG_SIDE,
            )
        } else {
            ownership(&result, &mask, &sampler, &tone, aw, ah)
        };
        #[cfg(test)]
        diagnostics::capture_layer(index, &sampler, &tone, &decision, aw, ah);
        let stride = width as usize * 3;
        // Render only ownership cells assigned to this layer. The previous
        // scan visited every pixel in the projected rectangle even when the
        // graph cut had already rejected most of those cells. On large
        // 31k×11k stacks that redundant decision check dominated runtime.
        let decision_width = decision.width();
        let decision_height = decision.height();
        result
            .as_mut()
            .par_chunks_mut(stride)
            .zip(mask.as_mut().par_chunks_mut(width as usize))
            .enumerate()
            .skip(top as usize)
            .take(layer_height as usize)
            .for_each(|(y, (row, covered))| {
                let ly = (y - top as usize) as f64;
                let ay = ly / scale;
                let dy =
                    ((ay / ah as f64 * decision_height as f64) as u32).min(decision_height - 1);
                for dx in 0..decision_width {
                    if decision.get_pixel(dx, dy)[0] == 0 {
                        continue;
                    }
                    // These bounds are the inverse of the decision lookup
                    // above. Adjacent cells meet without leaving a pixel
                    // gap, while rejected cells are never sampled.
                    let cell_start =
                        (dx as f64 * aw as f64 / decision_width as f64 * scale).ceil() as u32;
                    let cell_end = (((dx + 1) as f64 * aw as f64 / decision_width as f64 * scale)
                        .ceil() as u32)
                        .min(layer_width);
                    let x_start = left + cell_start.min(layer_width);
                    let x_end = left + cell_end;
                    for x in x_start..x_end {
                        let lx = (x - left) as f64;
                        let ax = lx / scale;
                        let Some(pixel) = sampler.sample(lx, ly) else {
                            continue;
                        };
                        let pixel = adjusted(pixel, tone.at(ax, ay));
                        row[x as usize * 3..x as usize * 3 + 3].copy_from_slice(&pixel.0);
                        covered[x as usize] = 255;
                    }
                }
            });
        covered_bounds = Some(match covered_bounds {
            Some((covered_left, covered_right, covered_top, covered_bottom)) => (
                covered_left.min(left),
                covered_right.max(right),
                covered_top.min(top),
                covered_bottom.max(bottom),
            ),
            None => (left, right, top, bottom),
        });
    }
    let covered = mask.as_raw().iter().filter(|&&v| v != 0).count();
    if covered == 0 {
        return Err("The aligned stack has no covered pixels.".into());
    }
    // An unsupported canvas margin is not photographic detail. Export only an
    // entirely covered rectangle; do not hide holes with stretched/reflected
    // source pixels, which looks like an unfused blur at the image boundary.
    #[cfg(test)]
    diagnostics::capture_crop(&mask);
    let output = crop_to_valid_rectangle(result, &mask);
    println!(
        "  - Verified covered output: {}x{}; every pixel has source ownership",
        output.width(),
        output.height()
    );
    Ok(output)
}

fn detail_preserving_mosaic_streaming<R: Runtime, F>(
    images: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    app: AppHandle<R>,
    event: &str,
    load: &mut F,
    offset_x: f64,
    offset_y: f64,
    width: u32,
    height: u32,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let store = StreamingMosaicStore::new(width, height)?;
    let mut seen_capture_groups = HashSet::new();
    println!(
        "  - Streaming detail-preserving mosaic canvas: {width}x{height}, tiles {}x{} of {}px",
        store.tile_columns, store.tile_rows, STREAMING_TILE_SIZE
    );
    for (index, &info) in images.iter().enumerate() {
        let capture_group_id = capture_group_ids
            .and_then(|groups| groups.get(&info.id))
            .copied()
            .unwrap_or_else(|| {
                u8::try_from(index + 1).expect("focus stack source limit keeps group ids in u8")
            });
        let first_capture_group_layer = seen_capture_groups.insert(capture_group_id);
        let _ = app.emit(
            event,
            format!(
                "Streaming detail and selecting sharp source {} of {}",
                index + 1,
                images.len()
            ),
        );
        let mut source = load(info)?;
        let transform = &homographies[&info.id];
        let mut source_divisor = 1.0;
        if projection == Projection::Planar {
            let center = Point3::new(info.width as f64 * 0.5, info.height as f64 * 0.5, 1.0);
            let c = transform * center;
            let dx = transform * (center + nalgebra::Vector3::new(1.0, 0.0, 0.0));
            let dy = transform * (center + nalgebra::Vector3::new(0.0, 1.0, 0.0));
            let sx = (dx.x / dx.z - c.x / c.z).hypot(dx.y / dx.z - c.y / c.z);
            let sy = (dy.x / dy.z - c.x / c.z).hypot(dy.y / dy.z - c.y / c.z);
            let scale = sx.min(sy);
            while scale.is_finite()
                && scale > 0.0
                && scale * source_divisor < 0.67
                && source.width().min(source.height()) > 8
            {
                source = downsample_rgb_half(&source);
                source_divisor *= 2.0;
            }
        }
        let inverse = transform
            .try_inverse()
            .ok_or("The stack contains a non-invertible alignment.")?;
        let Some((left, right, top, bottom)) = transformed_image_region(
            info, transform, projection, offset_x, offset_y, width, height,
        ) else {
            continue;
        };
        let layer_width = right - left + 1;
        let layer_height = bottom - top + 1;
        let layer_scale =
            (layer_width.max(layer_height) as f64 / STREAMING_ANALYSIS_LONG_SIDE as f64).max(1.0);
        let analysis_width = (layer_width as f64 / layer_scale).ceil() as u32;
        let analysis_height = (layer_height as f64 / layer_scale).ceil() as u32;
        let mut sampler = LayerSampler {
            info,
            source: &source,
            source_divisor,
            inverse,
            projection,
            offset: (offset_x, offset_y),
            left,
            top,
            scale: layer_scale,
            residual: Field::new(analysis_width, analysis_height, GRID_STEP as f64),
        };
        let (base_analysis, base_analysis_mask, base_analysis_owner) =
            store.analysis_region(left, top, layer_scale, analysis_width, analysis_height)?;
        let has_existing_coverage = base_analysis_mask.as_raw().iter().any(|&value| value != 0);
        let (tone, decision) = if has_existing_coverage {
            let group_refinement_mask = (!first_capture_group_layer).then(|| {
                // A later focal plane must register against its own camera
                // position.  Using pixels already owned by another panorama
                // tile pulls repeated calligraphy toward the wrong column.
                // Keep those older positions available for colour matching,
                // but exclude them from residual motion estimation.
                GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
                    Luma([(base_analysis_mask.get_pixel(x, y)[0] != 0
                        && base_analysis_owner.get_pixel(x, y)[0] == capture_group_id)
                        .then_some(255)
                        .unwrap_or(0)])
                })
            });
            let refinement_mask = group_refinement_mask
                .as_ref()
                .unwrap_or(&base_analysis_mask);
            for _ in 0..2 {
                let (candidate, candidate_mask) =
                    streaming_candidate_analysis(&sampler, analysis_width, analysis_height);
                let (a, b, valid) = streaming_refinement_pair(
                    &base_analysis,
                    refinement_mask,
                    &candidate,
                    &candidate_mask,
                );
                refine_layer_from_analysis(&a, &b, &valid, &mut sampler);
            }
            let (candidate, candidate_mask) =
                streaming_candidate_analysis(&sampler, analysis_width, analysis_height);
            let tone = streaming_tone_field_from_analysis(
                &base_analysis,
                &base_analysis_mask,
                &candidate,
                &candidate_mask,
            );
            let decision = streaming_ownership_from_analysis(
                &base_analysis,
                &base_analysis_mask,
                &candidate,
                &candidate_mask,
                &tone,
                &base_analysis_owner,
                capture_group_id,
                first_capture_group_layer,
            );
            (tone, decision)
        } else {
            (
                Field::new(analysis_width, analysis_height, 112.0),
                GrayImage::from_pixel(1, 1, Luma([255])),
            )
        };
        println!(
            "    - Streaming '{}': projected region {}..{} x {}..{}, shared analysis {}x{}",
            info.filename, left, right, top, bottom, analysis_width, analysis_height
        );
        let first_column = left / STREAMING_TILE_SIZE;
        let last_column = right / STREAMING_TILE_SIZE;
        let first_row = top / STREAMING_TILE_SIZE;
        let last_row = bottom / STREAMING_TILE_SIZE;
        for tile_row in first_row..=last_row {
            for tile_column in first_column..=last_column {
                let (tile_left, tile_top, tile_width, tile_height) =
                    store.tile_extent(tile_column, tile_row);
                let x_start = left.max(tile_left);
                let x_end = right.min(tile_left + tile_width - 1);
                let y_start = top.max(tile_top);
                let y_end = bottom.min(tile_top + tile_height - 1);
                if x_start > x_end || y_start > y_end {
                    continue;
                }
                let (mut tile, mut tile_mask) = store.load_tile(tile_column, tile_row)?;
                let mut tile_owner = store.load_owner_tile(tile_column, tile_row);
                if !has_existing_coverage {
                    copy_streaming_tile_region(
                        &mut tile,
                        &mut tile_mask,
                        &mut tile_owner,
                        capture_group_id,
                        &sampler,
                        tile_left,
                        tile_top,
                        x_start,
                        x_end,
                        y_start,
                        y_end,
                    );
                } else {
                    copy_streaming_owned_tile_region(
                        &mut tile,
                        &mut tile_mask,
                        &mut tile_owner,
                        capture_group_id,
                        &sampler,
                        &tone,
                        &decision,
                        tile_left,
                        tile_top,
                        x_start,
                        x_end,
                        y_start,
                        y_end,
                        layer_scale,
                        analysis_width,
                        analysis_height,
                    );
                    // Cell-centre ownership decisions can miss a thin valid
                    // sliver inside an otherwise rejected cell. Preserve the
                    // sharp owner everywhere it already exists, but always
                    // admit real source pixels into still-uncovered holes so
                    // the final crop represents the source union rather than
                    // the largest accidental hole-free island.
                    fill_streaming_uncovered_tile_region(
                        &mut tile,
                        &mut tile_mask,
                        &mut tile_owner,
                        capture_group_id,
                        &sampler,
                        &tone,
                        tile_left,
                        tile_top,
                        x_start,
                        x_end,
                        y_start,
                        y_end,
                        layer_scale,
                    );
                }
                store.save_tile(tile_column, tile_row, &tile, &tile_mask)?;
                store.save_owner_tile(tile_column, tile_row, &tile_owner)?;
            }
        }
    }
    let crop = store
        .largest_valid_rectangle()?
        .ok_or("The aligned stack has no covered pixels.")?;
    println!(
        "  - Streaming coverage rectangle: {}x{} at ({}, {})",
        crop.2, crop.3, crop.0, crop.1
    );
    let harmonization = streaming_seam_harmonization(&store, crop)?;
    let mut output = store.materialize(crop)?;
    if let Some((gains, scale)) = harmonization {
        apply_streaming_seam_harmonization(&mut output, &gains, scale);
    }
    println!(
        "  - Verified covered output: {}x{}; every pixel has source ownership",
        output.width(),
        output.height()
    );
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn fill_streaming_uncovered_tile_region(
    tile: &mut Rgb32FImage,
    tile_mask: &mut GrayImage,
    tile_owner: &mut GrayImage,
    capture_group_id: u8,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    tile_left: u32,
    tile_top: u32,
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
    scale: f64,
) {
    let tile_width = tile.width();
    let stride = tile_width as usize * 3;
    tile.as_mut()
        .par_chunks_mut(stride)
        .zip(tile_mask.as_mut().par_chunks_mut(tile_width as usize))
        .zip(tile_owner.as_mut().par_chunks_mut(tile_width as usize))
        .enumerate()
        .skip(y_start.saturating_sub(tile_top) as usize)
        .take((y_end - y_start + 1) as usize)
        .for_each(|(local_y, ((row, covered), owner))| {
            let global_y = tile_top + local_y as u32;
            let layer_y = global_y.saturating_sub(sampler.top) as f64;
            let analysis_y = layer_y / scale;
            for x in x_start..=x_end {
                let local_x = x - tile_left;
                let layer_x = x.saturating_sub(sampler.left) as f64;
                if covered[local_x as usize] != 0 {
                    continue;
                }
                let Some(pixel) = sampler.sample(layer_x, layer_y) else {
                    continue;
                };
                let adjusted_pixel = adjusted(pixel, tone.at(layer_x / scale, analysis_y));
                let start = local_x as usize * 3;
                row[start..start + 3].copy_from_slice(&adjusted_pixel.0);
                covered[local_x as usize] = 255;
                owner[local_x as usize] = capture_group_id;
            }
        });
}

fn copy_streaming_tile_region(
    tile: &mut Rgb32FImage,
    tile_mask: &mut GrayImage,
    tile_owner: &mut GrayImage,
    capture_group_id: u8,
    sampler: &LayerSampler<'_>,
    tile_left: u32,
    tile_top: u32,
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
) {
    let tile_width = tile.width();
    let stride = tile_width as usize * 3;
    tile.as_mut()
        .par_chunks_mut(stride)
        .zip(tile_mask.as_mut().par_chunks_mut(tile_width as usize))
        .zip(tile_owner.as_mut().par_chunks_mut(tile_width as usize))
        .enumerate()
        .skip(y_start.saturating_sub(tile_top) as usize)
        .take((y_end - y_start + 1) as usize)
        .for_each(|(local_y, ((row, covered), owner))| {
            let global_y = tile_top + local_y as u32;
            let layer_y = global_y.saturating_sub(sampler.top) as f64;
            for x in x_start..=x_end {
                let local_x = x - tile_left;
                let layer_x = x.saturating_sub(sampler.left) as f64;
                let Some(pixel) = sampler.sample(layer_x, layer_y) else {
                    continue;
                };
                let start = local_x as usize * 3;
                row[start..start + 3].copy_from_slice(&pixel.0);
                covered[local_x as usize] = 255;
                owner[local_x as usize] = capture_group_id;
            }
        });
}

fn copy_streaming_owned_tile_region(
    tile: &mut Rgb32FImage,
    tile_mask: &mut GrayImage,
    tile_owner: &mut GrayImage,
    capture_group_id: u8,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    decision: &GrayImage,
    tile_left: u32,
    tile_top: u32,
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
    scale: f64,
    analysis_width: u32,
    analysis_height: u32,
) {
    let decision_width = decision.width();
    let decision_height = decision.height();
    let tile_width = tile.width();
    let stride = tile_width as usize * 3;
    tile.as_mut()
        .par_chunks_mut(stride)
        .zip(tile_mask.as_mut().par_chunks_mut(tile_width as usize))
        .zip(tile_owner.as_mut().par_chunks_mut(tile_width as usize))
        .enumerate()
        .skip(y_start.saturating_sub(tile_top) as usize)
        .take((y_end - y_start + 1) as usize)
        .for_each(|(local_y, ((row, covered), owner))| {
            let global_y = tile_top + local_y as u32;
            let layer_y = global_y.saturating_sub(sampler.top) as f64;
            let ay = layer_y / scale;
            let dy = ((ay / analysis_height as f64 * decision_height as f64) as u32)
                .min(decision_height.saturating_sub(1));
            for dx in 0..decision_width {
                if decision.get_pixel(dx, dy)[0] == 0 {
                    continue;
                }
                let cell_start = (dx as f64 * analysis_width as f64 / decision_width as f64 * scale)
                    .ceil() as u32;
                let cell_end = (((dx + 1) as f64 * analysis_width as f64 / decision_width as f64
                    * scale)
                    .ceil() as u32)
                    .min((analysis_width as f64 * scale).ceil() as u32);
                let cell_x_start = (sampler.left + cell_start).max(x_start);
                let cell_x_end = (sampler.left + cell_end).min(x_end + 1);
                for x in cell_x_start..cell_x_end {
                    let local_x = x - tile_left;
                    let layer_x = x.saturating_sub(sampler.left) as f64;
                    let Some(pixel) = sampler.sample(layer_x, layer_y) else {
                        continue;
                    };
                    let adjusted_pixel = adjusted(pixel, tone.at(layer_x / scale, ay));
                    let start = local_x as usize * 3;
                    row[start..start + 3].copy_from_slice(&adjusted_pixel.0);
                    covered[local_x as usize] = 255;
                    owner[local_x as usize] = capture_group_id;
                }
            }
        });
}

fn copy_sequence_gap_layer(
    result: &mut Rgb32FImage,
    mask: &mut GrayImage,
    sampler: &LayerSampler<'_>,
    left: u32,
    right: u32,
    top: u32,
    bottom: u32,
) {
    let width = result.width();
    let stride = width as usize * 3;
    result
        .as_mut()
        .par_chunks_mut(stride)
        .zip(mask.as_mut().par_chunks_mut(width as usize))
        .enumerate()
        .skip(top as usize)
        .take((bottom - top + 1) as usize)
        .for_each(|(y, (row, covered))| {
            let local_y = (y as u32 - top) as f64;
            for x in left..=right {
                let local_x = (x - left) as f64;
                let Some(pixel) = sampler.sample(local_x, local_y) else {
                    continue;
                };
                let start = x as usize * 3;
                row[start..start + 3].copy_from_slice(&pixel.0);
                covered[x as usize] = 255;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_store_round_trips_across_tile_boundaries() {
        let store = StreamingMosaicStore::new(2050, 1025).expect("tile store should initialize");
        for row in 0..store.tile_rows {
            for column in 0..store.tile_columns {
                let (_, _, width, height) = store.tile_extent(column, row);
                let value = (column + row * 3) as f32;
                let rgb =
                    Rgb32FImage::from_pixel(width, height, Rgb([value, value + 1.0, value + 2.0]));
                let mask = GrayImage::from_pixel(width, height, Luma([255]));
                store
                    .save_tile(column, row, &rgb, &mask)
                    .expect("tile should be writable");
            }
        }
        let crop = store
            .largest_valid_rectangle()
            .expect("coverage should be readable")
            .expect("full tile fixture should be covered");
        assert_eq!(crop, (0, 0, 2050, 1025));
        let output = store.materialize(crop).expect("tiles should materialize");
        assert_eq!(output.dimensions(), (2050, 1025));
        assert_eq!(*output.get_pixel(1023, 10), Rgb([0.0, 1.0, 2.0]));
        assert_eq!(*output.get_pixel(1024, 10), Rgb([1.0, 2.0, 3.0]));
        assert_eq!(*output.get_pixel(10, 1024), Rgb([3.0, 4.0, 5.0]));
    }

    #[test]
    fn streaming_analysis_region_is_continuous_across_tile_boundaries() {
        let store = StreamingMosaicStore::new(2050, 32).expect("tile store should initialize");
        for column in 0..store.tile_columns {
            let (_, _, width, height) = store.tile_extent(column, 0);
            let value = column as f32 + 1.0;
            let rgb = Rgb32FImage::from_pixel(width, height, Rgb([value, value, value]));
            let mask = GrayImage::from_pixel(width, height, Luma([255]));
            store
                .save_tile(column, 0, &rgb, &mask)
                .expect("tile should be writable");
        }

        let (analysis, mask, owner) = store
            .analysis_region(1018, 4, 2.0, 8, 4)
            .expect("analysis region should be readable");
        assert_eq!(analysis.dimensions(), (8, 4));
        assert_eq!(*analysis.get_pixel(2, 1), Rgb([1.0, 1.0, 1.0]));
        assert_eq!(*analysis.get_pixel(3, 1), Rgb([2.0, 2.0, 2.0]));
        assert!(mask.as_raw().iter().all(|&value| value == 255));
        assert!(owner.as_raw().iter().all(|&value| value == 0));
    }

    #[test]
    fn low_frequency_harmonization_softens_group_exposure_steps_without_blurring_detail() {
        let store = StreamingMosaicStore::new(256, 64).expect("tile store should initialize");
        let rgb = Rgb32FImage::from_fn(256, 64, |x, y| {
            let base = if x < 128 { 0.42 } else { 0.58 };
            let detail = if (x / 4 + y / 4) % 2 == 0 {
                0.03
            } else {
                -0.03
            };
            Rgb([base + detail, base + detail, base + detail])
        });
        let mask = GrayImage::from_pixel(256, 64, Luma([255]));
        let owner = GrayImage::from_fn(256, 64, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        store
            .save_owner_tile(0, 0, &owner)
            .expect("owner should be writable");
        let (gains, scale) = streaming_seam_harmonization(&store, (0, 0, 256, 64))
            .expect("harmonization should build")
            .expect("two owners should create a seam field");
        let mut output = rgb.clone();
        apply_streaming_seam_harmonization(&mut output, &gains, scale);
        let band_mean = |image: &Rgb32FImage, start: u32| {
            (start..start + 8)
                .map(|x| luma(*image.get_pixel(x, 32)))
                .sum::<f64>()
                / 8.0
        };
        let before_step = (band_mean(&rgb, 120) - band_mean(&rgb, 128)).abs();
        let after_step = (band_mean(&output, 120) - band_mean(&output, 128)).abs();
        assert!(
            after_step < before_step * 0.72,
            "the exposure seam should soften: {before_step:.4} -> {after_step:.4}"
        );
        let before_detail = (luma(*rgb.get_pixel(20, 20)) - luma(*rgb.get_pixel(24, 20))).abs();
        let after_detail =
            (luma(*output.get_pixel(20, 20)) - luma(*output.get_pixel(24, 20))).abs();
        assert!(
            after_detail >= before_detail * 0.90,
            "multiplicative low-frequency correction must retain local detail"
        );
    }

    #[test]
    fn streaming_uncovered_fill_preserves_owner_and_closes_real_source_holes() {
        let source = Rgb32FImage::from_pixel(16, 8, Rgb([0.8, 0.7, 0.6]));
        let info = image_info(1, &source);
        let mut tile = Rgb32FImage::from_pixel(16, 8, Rgb([0.2, 0.2, 0.2]));
        let mut mask = GrayImage::from_pixel(16, 8, Luma([255]));
        let mut owner = GrayImage::from_pixel(16, 8, Luma([3]));
        mask.put_pixel(5, 3, Luma([0]));
        owner.put_pixel(5, 3, Luma([0]));
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(16, 8, 32.0),
        };
        fill_streaming_uncovered_tile_region(
            &mut tile,
            &mut mask,
            &mut owner,
            4,
            &sampler,
            &Field::new(16, 8, 32.0),
            0,
            0,
            0,
            15,
            0,
            7,
            1.0,
        );

        assert_eq!(*tile.get_pixel(4, 3), Rgb([0.2, 0.2, 0.2]));
        assert_eq!(*tile.get_pixel(5, 3), Rgb([0.8, 0.7, 0.6]));
        assert_eq!(mask.get_pixel(5, 3)[0], 255);
        assert_eq!(owner.get_pixel(4, 3)[0], 3);
        assert_eq!(owner.get_pixel(5, 3)[0], 4);
    }

    #[test]
    fn focus_group_ownership_cannot_reclaim_another_camera_position() {
        let sharp = sharp_fixture();
        let base = image::imageops::blur(&sharp, 2.0);
        let mask = GrayImage::from_pixel(sharp.width(), sharp.height(), Luma([255]));
        let owner = GrayImage::from_fn(sharp.width(), sharp.height(), |x, _| {
            Luma([if x < sharp.width() / 2 { 3 } else { 4 }])
        });
        let decision = streaming_ownership_from_analysis(
            &base,
            &mask,
            &sharp,
            &mask,
            &Field::new(sharp.width(), sharp.height(), 32.0),
            &owner,
            3,
            false,
        );
        let left_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x < decision.width() / 2 && pixel[0] != 0)
            .count();
        let right_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x >= decision.width() / 2 && pixel[0] != 0)
            .count();
        assert!(
            left_selected > 0,
            "the sharper member should contribute inside its group"
        );
        assert_eq!(
            right_selected, 0,
            "a focal member must not overwrite a different camera position"
        );
    }

    #[test]
    fn panorama_group_transition_only_enters_from_new_coverage() {
        let candidate = sharp_fixture();
        let base = image::imageops::blur(&candidate, 1.0);
        let mut base_mask = GrayImage::new(candidate.width(), candidate.height());
        let mut owner = GrayImage::new(candidate.width(), candidate.height());
        for y in 0..candidate.height() {
            for x in 0..candidate.width() * 3 / 4 {
                base_mask.put_pixel(x, y, Luma([255]));
                owner.put_pixel(x, y, Luma([2]));
            }
        }
        let candidate_mask =
            GrayImage::from_pixel(candidate.width(), candidate.height(), Luma([255]));
        let decision = streaming_ownership_from_analysis(
            &base,
            &base_mask,
            &candidate,
            &candidate_mask,
            &Field::new(candidate.width(), candidate.height(), 32.0),
            &owner,
            3,
            true,
        );
        let left_quarter_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x < decision.width() / 4 && pixel[0] != 0)
            .count();
        let new_coverage_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x >= decision.width() * 3 / 4 && pixel[0] != 0)
            .count();
        assert_eq!(
            left_quarter_selected, 0,
            "a new camera group must not form a detached sharpness island"
        );
        assert!(
            new_coverage_selected > 0,
            "new source coverage must be admitted"
        );
    }

    fn image_info(id: usize, image: &Rgb32FImage) -> ImageInfo {
        ImageInfo {
            id,
            filename: format!("source-{id}.png"),
            width: image.width(),
            height: image.height(),
            alignment_image: image::DynamicImage::ImageRgb32F(image.clone()).to_luma8(),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }

    fn sharp_fixture() -> Rgb32FImage {
        Rgb32FImage::from_fn(160, 120, |x, y| {
            let value = 0.36
                + 0.12 * (x as f32 * 0.75).sin()
                + 0.10 * (y as f32 * 0.65).cos()
                + 0.03 * ((x + y) as f32 * 0.19).sin();
            Rgb([value * 1.1, value, value * 0.8])
        })
    }

    fn noisy_colour_stroke_pair() -> (Rgb32FImage, Rgb32FImage) {
        let sharp = Rgb32FImage::from_fn(192, 160, |x, y| {
            let stroke = (x as i32 - 96).abs() <= 3 && (20..140).contains(&y);
            // A faded red mark can have almost the same luminance as its
            // brown substrate; colour edges still carry genuine focus.
            if stroke {
                Rgb([0.46, 0.31, 0.27])
            } else {
                Rgb([0.40, 0.34, 0.27])
            }
        });
        let blurred = image::imageops::blur(&sharp, 3.2);
        let noisy = |im: &Rgb32FImage, amplitude: f32| {
            Rgb32FImage::from_fn(im.width(), im.height(), |x, y| {
                let hash = x.wrapping_mul(0x9e3779b9) ^ y.wrapping_mul(0x85ebca6b);
                let hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
                let noise = ((hash & 65535) as f32 / 65535.0 - 0.5) * amplitude;
                Rgb(im.get_pixel(x, y).0.map(|v| (v + noise).clamp(0.0, 1.0)))
            })
        };
        (noisy(&sharp, 0.020), noisy(&blurred, 0.080))
    }

    #[test]
    fn sensor_noise_cannot_hide_a_sharper_faint_colour_stroke() {
        let (sharp, noisy_blur) = noisy_colour_stroke_pair();
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let mask = GrayImage::from_pixel(192, 160, Luma([255]));
        let chosen = ownership(
            &noisy_blur,
            &mask,
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert!(
            selected >= 90,
            "faint focused colour must beat noisy defocus ({selected}/100)"
        );
    }

    #[test]
    fn extra_sensor_noise_cannot_reclaim_an_already_focused_stroke() {
        let (sharp, noisy_blur) = noisy_colour_stroke_pair();
        let info = image_info(1, &noisy_blur);
        let sampler = LayerSampler {
            info: &info,
            source: &noisy_blur,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let mask = GrayImage::from_pixel(192, 160, Luma([255]));
        let chosen = ownership(
            &sharp,
            &mask,
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert_eq!(
            selected, 0,
            "noisy defocus must not overwrite the focused stroke"
        );
    }

    #[test]
    fn diffuse_halo_cannot_outvote_a_narrow_focused_contour() {
        let sharp = Rgb32FImage::from_fn(192, 160, |x, y| {
            let line = (x as i32 - 96).abs() <= 1 && (20..140).contains(&y);
            let paper = 0.40 + 0.003 * (x as f32 * 0.17).sin();
            let value = if line { 0.28 } else { paper };
            Rgb([value, value * 0.9, value * 0.8])
        });
        let base = image::imageops::blur(&sharp, 5.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let chosen = ownership(
            &base,
            &GrayImage::from_pixel(192, 160, Luma([255])),
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert!(
            selected >= 90,
            "focused contour lost to its spread halo: {selected}/100"
        );
    }

    #[test]
    fn enclosed_sharp_tile_replaces_blur_without_losing_coverage() {
        let sharp = sharp_fixture();
        let base = image::imageops::blur(&sharp, 2.0);
        let candidate = image::imageops::crop_imm(&sharp, 40, 20, 80, 80).to_image();
        let sources = [base.clone(), candidate];
        let infos = [image_info(0, &sources[0]), image_info(1, &sources[1])];
        let transforms = HashMap::from([
            (0, Matrix3::identity()),
            (
                1,
                Matrix3::new(1.0, 0.0, 40.0, 0.0, 1.0, 20.0, 0.0, 0.0, 1.0),
            ),
        ]);
        let app = tauri::test::mock_app();
        let output = detail_preserving_mosaic(
            &[&infos[0], &infos[1]],
            &transforms,
            Projection::Planar,
            None,
            false,
            app.handle().clone(),
            "test-progress",
            &mut |info| Ok(sources[info.id].clone()),
        )
        .unwrap();
        assert_eq!(
            output.dimensions(),
            sharp.dimensions(),
            "coverage must include the last row and column"
        );
        let mut before = 0.0;
        let mut after = 0.0;
        for y in 30..90 {
            for x in 50..110 {
                before += (base.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
                after += (output.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
            }
        }
        assert!(
            after < before * 0.05,
            "the sharp interior must replace the blur: {after} vs {before}"
        );
        assert!(
            output
                .pixels()
                .all(|p| p.0.iter().all(|v| v.is_finite() && *v > 0.0))
        );
    }

    #[test]
    fn blurred_candidate_cannot_replace_an_already_sharp_interior() {
        let base = sharp_fixture();
        let source = image::imageops::blur(&base, 2.0);
        let info = image_info(0, &source);
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let mask = GrayImage::from_pixel(160, 120, Luma([255]));
        let selected = ownership(
            &base,
            &mask,
            &sampler,
            &Field::new(160, 120, 56.0),
            160,
            120,
        );
        let switched = (10..110)
            .flat_map(|y| (10..150).map(move |x| (x, y)))
            .filter(|&(x, y)| selected.get_pixel(x, y)[0] != 0)
            .count();
        assert_eq!(switched, 0);
    }

    #[test]
    fn sparse_in_focus_stroke_is_not_lost_to_cell_background() {
        let sharp = Rgb32FImage::from_fn(160, 120, |x, y| {
            let line = (x as i32 - 80).unsigned_abs() <= 1 && (12..108).contains(&y);
            let value = if line { 0.04 } else { 0.42 };
            Rgb([value, value, value])
        });
        let base = image::imageops::blur(&sharp, 3.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let mask = GrayImage::from_pixel(160, 120, Luma([255]));
        let selected = ownership(
            &base,
            &mask,
            &sampler,
            &Field::new(160, 120, 56.0),
            160,
            120,
        );
        let selected_on_stroke = (12..108)
            .filter(|&y| selected.get_pixel(80, y)[0] != 0)
            .count();
        assert!(
            selected_on_stroke > 70,
            "a sparse sharp stroke must own its pixels ({selected_on_stroke}/96)"
        );
    }

    #[test]
    fn streaming_focus_probe_keeps_a_sparse_sharp_stroke() {
        let sharp = Rgb32FImage::from_fn(160, 128, |x, y| {
            let line = (x as i32 - 80).unsigned_abs() <= 1 && (8..120).contains(&y);
            let value = if line { 0.04 } else { 0.42 };
            Rgb([value, value, value])
        });
        let base = image::imageops::blur(&sharp, 3.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 128, 48.0),
        };
        let selected = streaming_ownership_with_long_side(
            &base,
            &GrayImage::from_pixel(160, 128, Luma([255])),
            &sampler,
            &Field::new(160, 128, 56.0),
            160,
            128,
            5,
        );
        assert!(
            selected.get_pixel(2, 1)[0] != 0 || selected.get_pixel(2, 2)[0] != 0,
            "the bounded streaming probe must retain sparse focused writing"
        );
    }

    #[test]
    fn disagreement_is_lower_for_aligned_defocus_than_for_shifted_detail() {
        let sharp = Rgb32FImage::from_fn(160, 120, |x, y| {
            let value = 0.35 + 0.22 * (x as f32 * 0.37).sin() + 0.16 * (y as f32 * 0.29).cos();
            Rgb([value, value * 0.9, value * 0.75])
        });
        let base = image::imageops::blur(&sharp, 2.5);
        let info = image_info(1, &sharp);
        let tone = Field::new(160, 120, 56.0);
        let aligned = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let aligned_error = ownership_disagreement(&base, &aligned, &tone, 32.0, 32.0, 40.0);
        let shifted = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::new(1.0, 0.0, 4.0, 0.0, 1.0, -3.0, 0.0, 0.0, 1.0),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let shifted_error = ownership_disagreement(&base, &shifted, &tone, 32.0, 32.0, 40.0);
        assert!(
            aligned_error < shifted_error,
            "local averaging should preserve geometric mismatch signal: {aligned_error} vs {shifted_error}"
        );
    }

    #[test]
    fn invalid_sample_gap_cannot_cancel_disconnected_colour_mismatches() {
        let base = Rgb32FImage::from_pixel(100, 100, Rgb([0.5; 3]));
        let source = Rgb32FImage::from_fn(100, 100, |x, _| {
            // Invalid samples must retain their positions. If the valid
            // samples are packed into a smaller grid, these separated light
            // and dark strips become neighbours and falsely cancel out.
            let value = if x < 30 {
                0.8
            } else if x > 60 {
                0.2
            } else {
                f32::NAN
            };
            Rgb([value; 3])
        });
        let info = image_info(1, &source);
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(100, 100, 48.0),
        };
        let error =
            ownership_disagreement(&base, &sampler, &Field::new(100, 100, 56.0), 0.0, 0.0, 80.0);
        assert!(
            (error - 0.3).abs() < 1e-6,
            "missing samples must not hide a real colour mismatch: {error}"
        );
    }

    #[test]
    fn fully_invalid_overlap_is_not_treated_as_an_aligned_match() {
        let source = Rgb32FImage::from_pixel(32, 32, Rgb([0.5; 3]));
        let info = image_info(1, &source);
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::new(1.0, 0.0, 64.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(32, 32, 48.0),
        };
        assert_eq!(
            ownership_disagreement(&source, &sampler, &Field::new(32, 32, 56.0), 0.0, 0.0, 24.0,),
            1.0,
            "absence of common samples must not remove the mismatch penalty"
        );
    }

    #[test]
    fn scaled_sparse_detail_is_sampled_at_the_native_cell_centre() {
        let sharp = Rgb32FImage::from_fn(3_200, 1_600, |x, _| {
            let value = if (x as i32 - 84).unsigned_abs() <= 2 {
                0.04
            } else {
                0.42
            };
            Rgb([value; 3])
        });
        let base = image::imageops::blur(&sharp, 4.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 8.0,
            residual: Field::new(400, 200, 48.0),
        };
        let selected = ownership(
            &base,
            &GrayImage::from_pixel(3_200, 1_600, Luma([255])),
            &sampler,
            &Field::new(400, 200, 56.0),
            400,
            200,
        );
        let selected_on_stroke = (20..180)
            .filter(|&y| selected.get_pixel(10, y)[0] != 0)
            .count();
        assert!(
            selected_on_stroke > 100,
            "a native sharp stroke at a scaled cell centre must own its detail ({selected_on_stroke}/160)"
        );
    }

    #[test]
    fn smooth_tone_correction_preserves_selected_detail_relative_contrast() {
        let p = Rgb([0.2, 0.3, 0.4]);
        let q = Rgb([0.4, 0.5, 0.6]);
        let a = adjusted(p, [0.1, -0.05, 0.02]);
        let b = adjusted(q, [0.1, -0.05, 0.02]);
        for c in 0..3 {
            assert!((b[c] / a[c] - q[c] / p[c]).abs() < 1e-6);
        }
    }

    #[test]
    fn displacement_field_interpolation_is_continuous_at_cell_edges() {
        let mut field = Field::<2>::new(200, 200, 48.0);
        for (i, value) in field.values.iter_mut().enumerate() {
            *value = [
                (i % field.width) as f64 * 0.3,
                (i / field.width) as f64 * -0.2,
            ];
        }
        let a = field.at(48.0 - 1e-5, 73.0);
        let b = field.at(48.0 + 1e-5, 73.0);
        assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
    }
}
