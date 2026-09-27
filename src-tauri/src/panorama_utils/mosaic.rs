//! Focus mosaics: refine residual geometry, choose sharp source detail, match
//! only broad colour. The output is never an average of displaced fine detail.
use super::registration::{PatchMatchGates, probe_warped_patch, refine_warped_patch};
use super::seam_cut;
use super::stack_pipeline::degradation;
use super::stack_pipeline::intra_station;
use super::stack_pipeline::report::{IntraStationFrameRecord, IntraStationFrameStatus};
use super::stack_pipeline::station_degradation::{
    GroupJoinEvidence, StationMember, group_join_rejection, plan_station_fusion,
    record_run_entries, record_run_group_join_rejection, station_plan_entries,
};
use super::stitching::{
    Projection, downsample_rgb_half, get_high_quality_interpolated_pixel, map_target_to_source,
    output_bounds, pixel_aligned_canvas, transformed_image_region,
};
use crate::panorama_stitching::ImageInfo;
use image::{GrayImage, Luma, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Runtime};

#[cfg(test)]
#[path = "mosaic_diagnostics.rs"]
mod diagnostics;

// Actual configured intra-station analysis long side. Keep both compositor
// paths on the same source of truth so Stack_Report records what production
// used rather than the 2048 requirement ceiling (需求 2.2).
const ANALYSIS_LONG_SIDE: u32 = intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE;
// Bound the grid per source layer, not by the full panorama width. At a
// roughly 9504px layer footprint this yields approximately 19px cells, fine
// enough for seams to route around individual strokes.
// Also the Focus_Fuser's ownership cell count (需求 3.1): the station plane's
// long side is divided into at least this many cells.
pub(super) const SELECTION_LONG_SIDE: u32 = 512;
const LONG_SEQUENCE_SELECTION_LONG_SIDE: u32 = 768;
// Streaming still makes one continuous decision for the whole source layer.
// Keeping the analysis bounded preserves memory while preventing independent
// 1024px tiles from choosing incompatible owners along their shared border.
const STREAMING_ANALYSIS_LONG_SIDE: u32 = intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE;
const STREAMING_SELECTION_LONG_SIDE: u32 = 768;
// Ownership is already regularised by the graph cut. A wide blur here mixes
// focused and defocused focal planes across tens of native pixels on a 45MP
// source. Keep only a sub-cell antialiasing transition.
const STREAMING_OWNERSHIP_FEATHER: f32 = 0.65;
const STREAMING_HARMONIZATION_LONG_SIDE: u32 = 2400;
// Keep this narrow enough to cancel the actual source discontinuity at an
// ownership boundary. A broad gain blur follows nearby dark strokes and draws
// a visible halo even though the gain itself is constant per capture group.
const STREAMING_GROUP_GAIN_FEATHER: f32 = 3.0;
const STREAMING_GROUP_GAIN_MAX_LOG: f64 = 0.45;
// Camera-position lighting varies smoothly across a frame, so a seam needs a
// much broader low-frequency transition than the ownership antialiasing.
// These radii operate on the bounded 2400px analysis image.
const STREAMING_ILLUMINATION_LOCAL_SIGMA: f32 = 16.0;
const STREAMING_ILLUMINATION_CONTINUOUS_SIGMA: f32 = 320.0;
const STREAMING_ILLUMINATION_MIN_GAIN: f64 = 0.70;
const STREAMING_ILLUMINATION_MAX_GAIN: f64 = 1.40;
const GRID_STEP: u32 = 32;
const NATIVE_REFINE_STEP: u32 = 16;
const NATIVE_FIELD_RADIUS: f64 = 32.0;
// Selection cost added to a candidate that is sharper than the current
// composite but photometrically inconsistent with it (需求 3.3 requires ≥ 1.0).
pub(super) const OWNERSHIP_MISMATCH_PENALTY: f64 = 2.4;
const STREAMING_TILE_SIZE: u32 = 1024;
// A float RGB canvas costs twelve bytes per pixel before the ownership mask,
// decoded RAW layer, and canonical 16-bit result are counted.  Switch to the
// tiled backing store by canvas area as well as by capture-sequence metadata:
// evidence-only graph alignment intentionally has no synthetic sequence
// bridges, so using that flag alone made large, valid focus mosaics allocate
// the entire sparse bounding box in memory.
const STREAMING_CANVAS_MIN_PIXELS: u64 = 120_000_000;

type GroupToneRelation = ((u8, u8), [f64; 3], f64);

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

    fn covered_bounds(&self) -> Result<Option<(u32, u32, u32, u32)>, String> {
        let mut bounds: Option<(u32, u32, u32, u32)> = None;
        for row in 0..self.tile_rows {
            for column in 0..self.tile_columns {
                let (left, top, width, height) = self.tile_extent(column, row);
                let mask = self.read_mask_tile(column, row, width, height);
                for y in 0..height {
                    let row_start = y as usize * width as usize;
                    for x in 0..width {
                        if mask[row_start + x as usize] == 0 {
                            continue;
                        }
                        let gx = left + x;
                        let gy = top + y;
                        bounds = Some(match bounds {
                            Some((min_x, min_y, max_x, max_y)) => {
                                (min_x.min(gx), min_y.min(gy), max_x.max(gx), max_y.max(gy))
                            }
                            None => (gx, gy, gx, gy),
                        });
                    }
                }
            }
        }
        Ok(bounds.map(|(min_x, min_y, max_x, max_y)| {
            (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        }))
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
    measured_relations: &[GroupToneRelation],
) -> Result<Option<(Rgb32FImage, f64)>, String> {
    let (left, top, width, height) = crop;
    let scale = (width.max(height) as f64 / STREAMING_HARMONIZATION_LONG_SIDE as f64).max(1.0);
    let analysis_width = (width as f64 / scale).ceil() as u32;
    let analysis_height = (height as f64 / scale).ceil() as u32;
    let (analysis, mask, owner) =
        store.analysis_region(left, top, scale, analysis_width, analysis_height)?;
    let mut boundary_count = 0usize;
    let mut owner_pixels = [0usize; 256];
    for y in 0..analysis_height {
        for x in 0..analysis_width {
            let current = owner.get_pixel(x, y)[0];
            if current == 0 || mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            owner_pixels[current as usize] += 1;
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
                boundary_count += 1;
            }
        }
    }
    if boundary_count == 0 {
        return Ok(None);
    }
    // Estimate one exposure/colour correction per camera position from paper
    // immediately across its ownership boundaries.  The previous per-pixel
    // ratio of two blurred images followed dark strokes and created a bright
    // unsharp-mask halo around calligraphy.  Group-constant gains cannot trace
    // glyph contours, and dark/strongly coloured foreground is excluded here.
    let paper = image::imageops::blur(&analysis, 2.0);
    let mut pair_samples: BTreeMap<(u8, u8), Vec<[f64; 3]>> = BTreeMap::new();
    let probe = 6u32;
    let mut sample_pair = |ax: u32, ay: u32, bx: u32, by: u32| {
        if mask.get_pixel(ax, ay)[0] == 0 || mask.get_pixel(bx, by)[0] == 0 {
            return;
        }
        let a_owner = owner.get_pixel(ax, ay)[0];
        let b_owner = owner.get_pixel(bx, by)[0];
        if a_owner == 0 || b_owner == 0 || a_owner == b_owner {
            return;
        }
        let a = paper.get_pixel(ax, ay);
        let b = paper.get_pixel(bx, by);
        let a_luma = luma(*a);
        let b_luma = luma(*b);
        // Museum scans are often deliberately underexposed. Requiring 0.18
        // luma discarded most of this painting and left the problematic
        // camera group disconnected from the exposure solve. Group-constant
        // gains cannot follow brush contours, so darker overlap samples are
        // safe here; retain only a strict near-black and mismatch guard.
        if a_luma < 0.06 || b_luma < 0.06 || (a_luma - b_luma).abs() > 0.20 {
            return;
        }
        let (key, sign) = if a_owner < b_owner {
            ((a_owner, b_owner), 1.0)
        } else {
            ((b_owner, a_owner), -1.0)
        };
        pair_samples
            .entry(key)
            .or_default()
            .push(std::array::from_fn(|channel| {
                ((f64::from(b[channel]).max(0.015) / f64::from(a[channel]).max(0.015)).ln() * sign)
                    .clamp(-STREAMING_GROUP_GAIN_MAX_LOG, STREAMING_GROUP_GAIN_MAX_LOG)
            }));
    };
    for y in 0..analysis_height {
        for x in 0..analysis_width.saturating_sub(1) {
            if owner.get_pixel(x, y)[0] != owner.get_pixel(x + 1, y)[0]
                && x >= probe
                && x + probe < analysis_width
            {
                sample_pair(x - probe, y, x + probe, y);
            }
        }
    }
    for y in 0..analysis_height.saturating_sub(1) {
        for x in 0..analysis_width {
            if owner.get_pixel(x, y)[0] != owner.get_pixel(x, y + 1)[0]
                && y >= probe
                && y + probe < analysis_height
            {
                sample_pair(x, y - probe, x, y + probe);
            }
        }
    }
    // The per-owner log gains below are solved by repeatedly accumulating
    // `total += target * edge_weight` over every pair relation.  That floating
    // point sum is order sensitive, so the relation list must be ordered by owner
    // pair rather than by hash seed (需求 14.6).
    let mut pair_offsets: BTreeMap<(u8, u8), ([f64; 3], f64)> = pair_samples
        .into_iter()
        .filter_map(|(pair, samples)| {
            (samples.len() >= 12).then(|| {
                let offset: [f64; 3] = std::array::from_fn(|channel| {
                    median(
                        &mut samples
                            .iter()
                            .map(|value| value[channel])
                            .collect::<Vec<_>>(),
                    )
                });
                (pair, (offset, samples.len().min(512) as f64))
            })
        })
        .collect();
    // Same-coordinate overlap measurements are more reliable than probing
    // opposite sides of a seam, where the painting content can differ.
    for &(pair, offset, weight) in measured_relations {
        pair_offsets.insert(pair, (offset, weight));
    }
    let pair_offsets = pair_offsets
        .into_iter()
        .map(|(pair, (offset, weight))| (pair, offset, weight))
        .collect::<Vec<_>>();
    let anchor = (1u8..=u8::MAX)
        .max_by_key(|&id| owner_pixels[id as usize])
        .unwrap_or(1);
    let mut log_gains = [[0.0f64; 3]; 256];
    for _ in 0..48 {
        let previous = log_gains;
        for id in 1u8..=u8::MAX {
            if id == anchor || owner_pixels[id as usize] == 0 {
                continue;
            }
            for channel in 0..3 {
                let mut total = 0.0;
                let mut weight = 0.0;
                for &((a, b), offset, edge_weight) in &pair_offsets {
                    let target = if id == a {
                        previous[b as usize][channel] + offset[channel]
                    } else if id == b {
                        previous[a as usize][channel] - offset[channel]
                    } else {
                        continue;
                    };
                    total += target * edge_weight;
                    weight += edge_weight;
                }
                if weight > 0.0 {
                    log_gains[id as usize][channel] = (total / weight)
                        .clamp(-STREAMING_GROUP_GAIN_MAX_LOG, STREAMING_GROUP_GAIN_MAX_LOG);
                }
            }
        }
    }
    let owner_gains = Rgb32FImage::from_fn(analysis_width, analysis_height, |x, y| {
        let id = owner.get_pixel(x, y)[0] as usize;
        if id == 0 {
            Rgb([1.0; 3])
        } else {
            Rgb(std::array::from_fn(|channel| {
                log_gains[id][channel].exp() as f32
            }))
        }
    });
    let gains = image::imageops::blur(&owner_gains, STREAMING_GROUP_GAIN_FEATHER);
    println!(
        "  - Group exposure harmonization: {} boundary samples, {} robust group relations at {}x{}",
        boundary_count,
        pair_offsets.len(),
        analysis_width,
        analysis_height
    );
    Ok(Some((gains, scale)))
}

fn streaming_group_tone_relations_from_analysis(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    base_owner: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    capture_group_id: u8,
) -> Vec<GroupToneRelation> {
    let width = base.width().min(candidate.width()).min(base_owner.width());
    let height = base
        .height()
        .min(candidate.height())
        .min(base_owner.height());
    // Ordered by owner pair: the returned relations are later folded into the
    // shared `pair_offsets` map and into a float accumulation (需求 14.6).
    let mut samples = BTreeMap::<(u8, u8), Vec<[f64; 3]>>::new();
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            if base_mask.get_pixel(x, y)[0] == 0 || candidate_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let base_group = base_owner.get_pixel(x, y)[0];
            if base_group == 0 || base_group == capture_group_id {
                continue;
            }
            let current = *base.get_pixel(x, y);
            let source = *candidate.get_pixel(x, y);
            let current_luma = luma(current);
            let source_luma = luma(source);
            if current_luma < 0.045
                || source_luma < 0.045
                || (current_luma - source_luma).abs() > 0.22
                || current
                    .0
                    .iter()
                    .chain(source.0.iter())
                    .any(|value| !(0.015..0.97).contains(value))
            {
                continue;
            }
            let (pair, sign) = if base_group < capture_group_id {
                ((base_group, capture_group_id), 1.0)
            } else {
                ((capture_group_id, base_group), -1.0)
            };
            samples
                .entry(pair)
                .or_default()
                .push(std::array::from_fn(|channel| {
                    (f64::from(source[channel] / current[channel]).ln() * sign)
                        .clamp(-STREAMING_GROUP_GAIN_MAX_LOG, STREAMING_GROUP_GAIN_MAX_LOG)
                }));
        }
    }
    samples
        .into_iter()
        .filter_map(|(pair, values)| {
            (values.len() >= 24).then(|| {
                let offset = std::array::from_fn(|channel| {
                    median(
                        &mut values
                            .iter()
                            .map(|value| value[channel])
                            .collect::<Vec<_>>(),
                    )
                });
                (pair, offset, values.len().min(2048) as f64)
            })
        })
        .collect()
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

pub(super) fn smooth_streaming_low_frequency_illumination(output: &mut Rgb32FImage) {
    let scale = (output.width().max(output.height()) as f64
        / STREAMING_HARMONIZATION_LONG_SIDE as f64)
        .max(1.0);
    let width = (output.width() as f64 / scale).ceil().max(1.0) as u32;
    let height = (output.height() as f64 / scale).ceil().max(1.0) as u32;
    let analysis =
        image::imageops::resize(output, width, height, image::imageops::FilterType::Triangle);
    let valid = image::ImageBuffer::from_fn(width, height, |x, y| {
        let value = (luma(*analysis.get_pixel(x, y)) > 0.025) as u8 as f32;
        Luma([value])
    });
    let premultiplied = Rgb32FImage::from_fn(width, height, |x, y| {
        let weight = valid.get_pixel(x, y)[0];
        let pixel = analysis.get_pixel(x, y);
        Rgb([pixel[0] * weight, pixel[1] * weight, pixel[2] * weight])
    });
    let normalized_blur = |sigma: f32| {
        let colour = image::imageops::blur(&premultiplied, sigma);
        let weight = image::imageops::blur(&valid, sigma);
        Rgb32FImage::from_fn(width, height, |x, y| {
            let divisor = weight.get_pixel(x, y)[0];
            if divisor <= 1e-4 {
                return Rgb([0.0; 3]);
            }
            let pixel = colour.get_pixel(x, y);
            Rgb([pixel[0] / divisor, pixel[1] / divisor, pixel[2] / divisor])
        })
    };
    let local_low = normalized_blur(STREAMING_ILLUMINATION_LOCAL_SIGMA);
    let continuous_low = normalized_blur(STREAMING_ILLUMINATION_CONTINUOUS_SIGMA);
    let gains = Rgb32FImage::from_fn(width, height, |x, y| {
        if valid.get_pixel(x, y)[0] == 0.0 {
            return Rgb([1.0; 3]);
        }
        let local_pixel = *local_low.get_pixel(x, y);
        let continuous_pixel = *continuous_low.get_pixel(x, y);
        let local = luma(local_pixel);
        let continuous = luma(continuous_pixel);
        let luminance_gain = if local > 0.025 && continuous > 0.025 {
            (continuous / local).clamp(
                STREAMING_ILLUMINATION_MIN_GAIN,
                STREAMING_ILLUMINATION_MAX_GAIN,
            ) as f32
        } else {
            1.0f32
        };
        // Exposure and white-balance changes affect channels together, but a
        // scalar luma gain leaves a coloured seam when two RAW frames have
        // slightly different white balance.  Use the broad luma correction as
        // the anchor and add a restrained per-channel low-frequency term. The
        // high-frequency detail is still taken verbatim from the selected
        // owner, so this cannot blur brush edges.
        Rgb(std::array::from_fn(|channel| {
            let channel_gain = if local_pixel[channel] > 0.025 && continuous_pixel[channel] > 0.025
            {
                (continuous_pixel[channel] / local_pixel[channel]).clamp(
                    STREAMING_ILLUMINATION_MIN_GAIN as f32,
                    STREAMING_ILLUMINATION_MAX_GAIN as f32,
                )
            } else {
                luminance_gain
            };
            (luminance_gain * 0.72 + channel_gain * 0.28).clamp(
                STREAMING_ILLUMINATION_MIN_GAIN as f32,
                STREAMING_ILLUMINATION_MAX_GAIN as f32,
            )
        }))
    });
    let output_width = output.width() as usize;
    output
        .as_mut()
        .par_chunks_mut(output_width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let gain_y = (y as f64 / scale).clamp(0.0, gains.height().saturating_sub(1) as f64);
            for x in 0..output_width {
                let pixel = &mut row[x * 3..x * 3 + 3];
                if pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722 <= 0.025 {
                    continue;
                }
                let gain_x = (x as f64 / scale).clamp(0.0, gains.width().saturating_sub(1) as f64);
                let gain = get_high_quality_interpolated_pixel(&gains, gain_x, gain_y);
                for channel in 0..3 {
                    pixel[channel] = (pixel[channel] * gain[channel]).clamp(0.0, 1.0);
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

fn is_lower_left_black_capture_border(
    image: &Rgb32FImage,
    x: f64,
    y: f64,
    pixel: Rgb<f32>,
) -> bool {
    // Some handheld scan frames contain a genuinely black wedge outside the
    // photographed artwork in their lower-left corner. It is not source
    // content and must not become an owner merely because that frame extends
    // farther than its neighbours. Keep the rule deliberately local and very
    // dark: black robes, ink and furniture inside the image remain eligible.
    x < image.width() as f64 * 0.22
        && y > image.height() as f64 * 0.68
        && pixel[0].max(pixel[1]).max(pixel[2]) < 0.025
}

pub(super) fn corrected_lower_left_capture_sample(
    _info: &ImageInfo,
    image: &Rgb32FImage,
    _source_x: f64,
    _source_y: f64,
    sample_x: f64,
    sample_y: f64,
    pixel: Rgb<f32>,
) -> Option<Rgb<f32>> {
    if is_lower_left_black_capture_border(image, sample_x, sample_y, pixel) {
        return None;
    }
    Some(pixel)
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
        let pixel = rgb_at(self.source, sx, sy)?;
        corrected_lower_left_capture_sample(
            self.info,
            self.source,
            source.x,
            source.y,
            sx,
            sy,
            pixel,
        )
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

fn streaming_transition_structure(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    let mut levels = Vec::with_capacity(9);
    for y_offset in [0.18, 0.5, 0.82] {
        for x_offset in [0.18, 0.5, 0.82] {
            if let Some(pixel) = sample(x + cell_size * x_offset, y + cell_size * y_offset) {
                levels.push(luma(pixel));
            }
        }
    }
    if levels.len() < 5 {
        return 0.0;
    }
    levels.sort_by(f64::total_cmp);
    let dark = levels[levels.len() / 8];
    let paper = levels[levels.len() * 3 / 4].max(0.04);
    ((paper - dark) / paper).clamp(0.0, 1.0)
}

fn streaming_distance_to_fixed_label(
    width: u32,
    height: u32,
    fixed: &[i8],
    label: i8,
) -> Option<Vec<u32>> {
    let mut distances = vec![u32::MAX; fixed.len()];
    let mut pending = VecDeque::new();
    for (index, &value) in fixed.iter().enumerate() {
        if value == label {
            distances[index] = 0;
            pending.push_back(index);
        }
    }
    if pending.is_empty() {
        return None;
    }
    while let Some(index) = pending.pop_front() {
        let x = index as u32 % width;
        let y = index as u32 / width;
        let next_distance = distances[index].saturating_add(1);
        for neighbor in [
            x.checked_sub(1).map(|nx| (y * width + nx) as usize),
            (x + 1 < width).then_some((y * width + x + 1) as usize),
            y.checked_sub(1).map(|ny| (ny * width + x) as usize),
            (y + 1 < height).then_some(((y + 1) * width + x) as usize),
        ]
        .into_iter()
        .flatten()
        {
            if next_distance < distances[neighbor] {
                distances[neighbor] = next_distance;
                pending.push_back(neighbor);
            }
        }
    }
    Some(distances)
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
                // coverage. Make ink/edge cells expensive even when the two
                // sources agree photometrically: a seam through an aligned
                // brush stroke can still expose a tiny residual warp as a
                // horizontally cut glyph. Quiet paper remains the cheapest
                // route between characters.
                let base_structure = streaming_transition_structure(
                    |sample_x, sample_y| masked_rgb_at(base, base_mask, sample_x, sample_y),
                    x,
                    y,
                    size,
                );
                let candidate_structure = streaming_transition_structure(
                    |sample_x, sample_y| {
                        masked_rgb_at(candidate, candidate_mask, sample_x, sample_y)
                            .map(|pixel| adjusted(pixel, tone.at(sample_x, sample_y)))
                    },
                    x,
                    y,
                    size,
                );
                disagreement[index] = disagreement[index]
                    .max(base_structure)
                    .max(candidate_structure);
                preference[index] = 0.0;
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
    if panorama_transition {
        // Put the transition near the middle of the real overlap. With a
        // uniform bias toward the old panorama the minimum cut hugs the first
        // newly uncovered cell and exposes the rectangular source footprint.
        // Distances to the mandatory old/new coverage seeds provide a spatial
        // prior without using filename order or inventing image content.
        if let (Some(base_distance), Some(candidate_distance)) = (
            streaming_distance_to_fixed_label(grid_width, grid_height, &fixed, -1),
            streaming_distance_to_fixed_label(grid_width, grid_height, &fixed, 1),
        ) {
            for index in 0..count {
                if fixed[index] != 0 {
                    continue;
                }
                let base = base_distance[index] as f64;
                let candidate = candidate_distance[index] as f64;
                // Keep most of an already verified panorama intact. The new
                // camera position should own its genuinely new coverage and a
                // narrow connected overlap for a non-rectangular seam, not the
                // middle of a very large overlap. Mid-overlap ownership let a
                // weak defocused bridge (Wen Yuan Tu 3483/3484) replace the
                // tree roots and display rail with a visibly displaced block.
                preference[index] =
                    ((base - candidate) / (base + candidate).max(1.0)) * 0.60 - 0.30;
            }
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

/// Sharpness_Score of a gray analysis image, sampled with `spacing` pixels
/// between probes so a measurement can be taken at a bounded analysis
/// resolution (需求 2.1 / 2.2).  `None` where the probe window would leave the
/// image, which is what "valid pixels" means for a full frame.
pub(crate) fn gray_analysis_sharpness(
    image: &GrayImage,
    x: f64,
    y: f64,
    spacing: f64,
) -> Option<f64> {
    let half = 6.0 * spacing + 1.0;
    if !(x - half >= 0.0
        && y - half >= 0.0
        && x + half < f64::from(image.width())
        && y + half < f64::from(image.height()))
    {
        return None;
    }
    Some(acutance(
        |sample_x, sample_y| {
            intra_station::gray_sample(
                image,
                x + (sample_x - x) * spacing,
                y + (sample_y - y) * spacing,
            )
        },
        x,
        y,
    ))
}

/// Which caller the native patch refinement is serving.
///
/// The intra-station registrar composites at native resolution by definition,
/// so the analysis-scale early exit has no meaning there (需求 2.2).  The legacy
/// mosaic comparison path keeps it, together with its original search radius
/// and its single bidirectional gate, so its output stays byte identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeRefineMode {
    /// Legacy comparison path: `scale <= 2.0` returns without measuring.
    LegacyMosaic,
    /// Intra-station path: every control point passes the 需求 2.4/2.5/2.6 gates
    /// and the frame keeps its local field only when it improves (需求 2.7).
    IntraStation {
        /// Native long side of the anchor frame, for the 需求 2.6 error gate.
        anchor_long_side: u32,
    },
}

impl NativeRefineMode {
    /// `(matching window radius, translation search extent)` in patch samples.
    ///
    /// The legacy pair is unchanged.  The intra-station pair widens the window to
    /// 33 native pixels at the finest pass (需求 2.3) and keeps the search support
    /// inside the patch buffer, which is the only way `refine_warped_patch`
    /// returns a match at all.
    fn patch_geometry(self) -> (i32, i32) {
        match self {
            Self::LegacyMosaic => (10, 7),
            Self::IntraStation { .. } => (
                intra_station::INTRA_STATION_MATCH_RADIUS,
                intra_station::INTRA_STATION_SEARCH_SAMPLES,
            ),
        }
    }

    fn anchor_long_side(self) -> Option<u32> {
        match self {
            Self::LegacyMosaic => None,
            Self::IntraStation { anchor_long_side } => Some(anchor_long_side),
        }
    }

    /// Half side of the patch buffers this mode builds.
    fn patch_half(self) -> i32 {
        match self {
            Self::LegacyMosaic => intra_station::LEGACY_MOSAIC_PATCH_HALF,
            Self::IntraStation { .. } => intra_station::INTRA_STATION_PATCH_HALF,
        }
    }

    /// Similarity gates of one control point.
    fn patch_gates(self) -> PatchMatchGates {
        match self {
            Self::LegacyMosaic => PatchMatchGates::LEGACY,
            Self::IntraStation { .. } => intra_station::INTRA_STATION_PATCH_GATES,
        }
    }

    /// `true` when the two sides of a patch pair have to be brought to a
    /// comparable focal plane before they are correlated.
    ///
    /// Only the intra-station path needs it, because only there are the two
    /// sides deliberately focused on different planes.  The mosaic path matches
    /// two frames of the same plane and its output has to stay byte identical.
    fn matches_focal_plane(self) -> bool {
        matches!(self, Self::IntraStation { .. })
    }
}

/// What one native refinement pass measured (需求 2.9).  Every field is an
/// observation; the caller turns it into the report record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NativeRefineOutcome {
    /// Control points that passed every gate in the final pass.
    pub(crate) accepted_control_points: usize,
    /// Control points whose measurement was attempted in the final pass.
    pub(crate) evaluated_control_points: usize,
    /// `1 - accepted / evaluated` (需求 2.6).
    pub(crate) rejected_control_point_ratio: f64,
    /// Median symmetric reprojection error of the global model, native pixels.
    pub(crate) global_median_symmetric_error_px: f64,
    /// Control points the global baseline median was formed from.  `0` means
    /// 需求 2.7 has no baseline to compare against (需求 2.7 / 2.9).
    pub(crate) global_baseline_samples: usize,
    /// Median symmetric reprojection error after refinement, native pixels.
    pub(crate) local_median_symmetric_error_px: f64,
    /// Control points the refined median was formed from.
    pub(crate) local_baseline_samples: usize,
    /// How 需求 2.7 resolved (需求 2.9).
    pub(crate) local_field_verdict: intra_station::LocalFieldVerdict,
    /// Median `softer / sharper` high-frequency energy ratio of the control
    /// point patch pairs before and after the focal-plane matched low-pass.
    pub(crate) focal_plane_ratio_before: f64,
    pub(crate) focal_plane_ratio_after: f64,
    /// Largest observed bidirectional round trip, native pixels (需求 2.4).
    pub(crate) max_bidirectional_round_trip_px: f64,
    /// 需求 2.8 coverage of the accepted control point cells.
    pub(crate) inlier_area_coverage: f64,
    /// Measured distance between neighbouring control points, in native pixels
    /// of the anchor frame (需求 2.3).  The grid step is fixed in *analysis*
    /// units, so the native spacing is whatever the layer's analysis scale makes
    /// it, and only a measurement can say what that was.
    pub(crate) control_point_spacing_px: f64,
    /// `true` when the whole local field was discarded (需求 2.7).
    pub(crate) reverted_to_global: bool,
    /// `true` when the mode declined to measure at all.
    pub(crate) skipped: bool,
    /// Accepted control point cells of the *final* spacing pass alone.
    ///
    /// Test-only, and the whole point of it is to be different from
    /// `accepted_control_points`: 需求 2.8 counts the cells that contain an
    /// accepted local match over the *whole* refinement, so the coverage is the
    /// union over the spacing passes ([`intra_station::ControlPointGrid::absorb`])
    /// while this is what the last pass on its own saw.  Property 8 asserts which
    /// of the two the reported coverage is built from, and it can only do that if
    /// it can see both.
    #[cfg(test)]
    pub(crate) last_pass_accepted_control_points: usize,
    /// Control point cells whose centre lay on covered base in the final pass,
    /// i.e. the denominator of 需求 2.8 in cells.  Test-only, for the same
    /// reason.
    #[cfg(test)]
    pub(crate) valid_cells: usize,
    /// Area of one control point cell in native pixels of the anchor frame.
    /// Test-only, for the same reason.
    #[cfg(test)]
    pub(crate) cell_area_px: f64,
}

impl Default for NativeRefineOutcome {
    fn default() -> Self {
        Self {
            accepted_control_points: 0,
            evaluated_control_points: 0,
            rejected_control_point_ratio: 0.0,
            global_median_symmetric_error_px: 0.0,
            global_baseline_samples: 0,
            local_median_symmetric_error_px: 0.0,
            local_baseline_samples: 0,
            local_field_verdict: intra_station::LocalFieldVerdict::Indeterminate,
            focal_plane_ratio_before: 1.0,
            focal_plane_ratio_after: 1.0,
            max_bidirectional_round_trip_px: 0.0,
            inlier_area_coverage: 0.0,
            control_point_spacing_px: 0.0,
            reverted_to_global: false,
            skipped: true,
            #[cfg(test)]
            last_pass_accepted_control_points: 0,
            #[cfg(test)]
            valid_cells: 0,
            #[cfg(test)]
            cell_area_px: 0.0,
        }
    }
}

impl NativeRefineOutcome {
    /// 需求 2.7 / 2.8: the frame verdict implied by this measurement.
    pub(crate) fn frame_status(&self) -> IntraStationFrameStatus {
        intra_station::frame_status(
            self.local_field_verdict.field_kept(),
            self.inlier_area_coverage,
        )
    }
}

/// One `intra_station[*].frames[*]` entry (需求 2.9).  Shared by the in-memory
/// and the tiled compositor so the report cannot describe one of them only.
fn intra_station_frame_record(
    path: &str,
    into_anchor: Matrix3<f64>,
    outcome: &NativeRefineOutcome,
) -> IntraStationFrameRecord {
    IntraStationFrameRecord {
        path: path.to_string(),
        transform: [
            into_anchor[(0, 0)],
            into_anchor[(0, 1)],
            into_anchor[(0, 2)],
            into_anchor[(1, 0)],
            into_anchor[(1, 1)],
            into_anchor[(1, 2)],
            into_anchor[(2, 0)],
            into_anchor[(2, 1)],
            into_anchor[(2, 2)],
        ],
        inliers: outcome.accepted_control_points,
        inlier_area_coverage: outcome.inlier_area_coverage,
        control_point_spacing_px: outcome.control_point_spacing_px,
        rejected_control_point_ratio: outcome.rejected_control_point_ratio,
        median_symmetric_error_px: if outcome.reverted_to_global {
            outcome.global_median_symmetric_error_px
        } else {
            outcome.local_median_symmetric_error_px
        },
        global_median_symmetric_error_px: outcome.global_median_symmetric_error_px,
        global_baseline_samples: outcome.global_baseline_samples,
        local_baseline_samples: outcome.local_baseline_samples,
        local_field_verdict: outcome.local_field_verdict.record(),
        focal_plane_ratio_before: outcome.focal_plane_ratio_before,
        focal_plane_ratio_after: outcome.focal_plane_ratio_after,
        status: outcome.frame_status(),
    }
}

/// Median Sharpness_Score over the valid pixels of one Source_RAW, measured the
/// same way the anchor selection of 需求 2.1 measures it.
///
/// This is what 需求 12.1 ranks the members of a degraded Capture_Station by.
/// The measurement runs on the already prepared analysis image and on the fixed
/// probe grid of [`intra_station::median_valid_sharpness`], so it costs a few
/// thousand samples per frame and is independent of the canvas the station
/// happens to be rendered on.
///
/// Shared with the default station path in [`super::stitching`], so a degraded
/// station ranks its members by the same number whichever compositor rendered
/// it.
pub(crate) fn station_member_median_sharpness(info: &ImageInfo) -> f64 {
    let analysis = &info.alignment_image;
    let (width, height) = (analysis.width(), analysis.height());
    let spacing = intra_station::analysis_probe_spacing(width.max(height));
    intra_station::median_valid_sharpness(width, height, |x, y| {
        gray_analysis_sharpness(analysis, x, y, spacing)
    })
}

/// Masked base pixels of the already composited canvas, addressed in global
/// canvas coordinates.
///
/// The non-streaming path wraps the whole canvas.  The streaming path wraps one
/// [`STREAMING_TILE_SIZE`] tile at a time — a canvas of hundreds of megapixels
/// never exists in memory there — so a control point whose matching window
/// leaves the loaded tile has no base to measure against and is skipped, exactly
/// as a control point over uncovered canvas is.  The residual field is a smooth
/// Gaussian interpolation of the accepted observations, so the thin band along
/// each tile border is carried by its neighbours.
struct NativeRefineBase<'a> {
    image: &'a Rgb32FImage,
    mask: &'a GrayImage,
    left: u32,
    top: u32,
}

impl NativeRefineBase<'_> {
    /// The composited pixel at a global canvas position, or `None` where the
    /// position is outside this window or not yet covered.
    fn masked(&self, gx: f64, gy: f64) -> Option<Rgb<f32>> {
        let lx = gx - f64::from(self.left);
        let ly = gy - f64::from(self.top);
        if lx < 0.0 || ly < 0.0 {
            return None;
        }
        let pixel = rgb_at(self.image, lx, ly)?;
        (self.mask.get_pixel(lx as u32, ly as u32)[0] != 0).then_some(pixel)
    }

    fn covered(&self, gx: u32, gy: u32) -> bool {
        let Some(lx) = gx.checked_sub(self.left) else {
            return false;
        };
        let Some(ly) = gy.checked_sub(self.top) else {
            return false;
        };
        lx < self.mask.width() && ly < self.mask.height() && self.mask.get_pixel(lx, ly)[0] != 0
    }
}

/// Observations of one spacing pass.
///
/// Kept apart from the pass that applies them because the streaming path fills
/// one accumulator from several tiles before the displacement field is built.
struct NativeRefinePass {
    observations: Vec<(Point2<f64>, [f64; 2], f64)>,
    accepted_grid: intra_station::ControlPointGrid,
    global_errors: Vec<f64>,
    local_errors: Vec<f64>,
    evaluated: usize,
    valid_cells: usize,
    max_round_trip: f64,
    similarity: SimilarityStats,
}

impl NativeRefinePass {
    fn new(width: u32, height: u32) -> Self {
        let columns = (width.saturating_sub(40) as usize / NATIVE_REFINE_STEP as usize) + 1;
        let rows = (height.saturating_sub(40) as usize / NATIVE_REFINE_STEP as usize) + 1;
        Self {
            observations: Vec::new(),
            accepted_grid: intra_station::ControlPointGrid::new(columns, rows),
            global_errors: Vec::new(),
            local_errors: Vec::new(),
            evaluated: 0,
            valid_cells: 0,
            max_round_trip: 0.0,
            similarity: SimilarityStats::default(),
        }
    }
}

/// Width of one correlation bucket, and the value the lowest bucket starts at.
const SIMILARITY_BUCKET_WIDTH: f64 = 0.05;
const SIMILARITY_BUCKET_FLOOR: f64 = 0.50;
const SIMILARITY_BUCKETS: usize = 10;

/// Correlation the round-trip distribution is reported from, so the log states
/// the tradeoff at the similarity floor that is actually in force.
const SIMILARITY_REPORTED_FLOOR: f64 = intra_station::INTRA_STATION_MIN_CORRELATION;

/// Which correlation bucket a score belongs to, or `None` below the floor.
fn similarity_bucket(correlation: f64) -> Option<usize> {
    if !correlation.is_finite() || correlation < SIMILARITY_BUCKET_FLOOR {
        return None;
    }
    Some(
        (((correlation - SIMILARITY_BUCKET_FLOOR) / SIMILARITY_BUCKET_WIDTH) as usize)
            .min(SIMILARITY_BUCKETS - 1),
    )
}

/// What the similarity gates of one pass actually saw.
///
/// This exists so the correlation threshold can be stated as a measured
/// accept-rate/precision tradeoff rather than as a number someone liked.  The
/// precision proxy is the share of each correlation bucket that then closed the
/// 需求 2.4 bidirectional round trip within 0.40 native pixels: a bucket whose
/// forward peak is real closes the round trip, a bucket matching a repeated
/// texture does not.
#[derive(Debug, Clone, Default)]
struct SimilarityStats {
    /// Forward peak correlation of every control point whose patch pair carried
    /// comparable signal at all.
    correlations: Vec<f64>,
    /// Per bucket, how many control points reached the reverse match.
    bucket_total: [usize; SIMILARITY_BUCKETS],
    /// Per bucket, how many of those closed the round trip.
    bucket_round_trip: [usize; SIMILARITY_BUCKETS],
    /// Round trip in native pixels of every control point that reached the
    /// reverse match at or above [`SIMILARITY_REPORTED_FLOOR`], so the 需求 2.4
    /// tolerance can be read off a distribution instead of guessed.
    accepted_round_trips: Vec<f64>,
    /// Second-peak margin of those same control points, so the margin gate can
    /// be checked against data too.
    accepted_peak_margins: Vec<f64>,
    /// `softer / sharper` high-frequency energy ratio of every measured pair,
    /// before and after the matched low-pass.
    focal_ratio_before: Vec<f64>,
    focal_ratio_matched: Vec<f64>,
    focal_ratio_after: Vec<f64>,
    /// Sigma, in patch samples, the focal-plane match decided the two sides
    /// differ by, and how often the descent ran out of budget.
    focal_passes: Vec<f64>,
    focal_capped: usize,
}

impl SimilarityStats {
    fn absorb(&mut self, point: &NativeControlPoint) {
        if let Some(correlation) = point.best_correlation {
            self.correlations.push(correlation);
            if let Some(bucket) = similarity_bucket(correlation)
                && point.reverse_attempted
            {
                self.bucket_total[bucket] += 1;
                if point.round_trip_closed {
                    self.bucket_round_trip[bucket] += 1;
                }
                if correlation >= SIMILARITY_REPORTED_FLOOR {
                    self.accepted_round_trips.push(point.round_trip_px);
                    if let Some(margin) = point.peak_margin {
                        self.accepted_peak_margins.push(margin);
                    }
                }
            }
        }
        if let Some(focal) = point.focal_plane {
            self.focal_ratio_before.push(focal.sharpness_ratio_before);
            self.focal_ratio_matched.push(focal.sharpness_ratio_matched);
            self.focal_ratio_after.push(focal.sharpness_ratio_after);
            self.focal_passes.push(focal.matched_variance.sqrt());
            self.focal_capped += usize::from(focal.capped);
        }
    }

    /// One log line: the correlation distribution, the measured round-trip pass
    /// rate per bucket and what the focal-plane match did.
    fn summary(&mut self) -> String {
        if self.correlations.is_empty() {
            return "no measurable patch pair".to_string();
        }
        let percentile = |values: &mut Vec<f64>, fraction: f64| -> f64 {
            values.sort_by(|left, right| left.total_cmp(right));
            let index = ((values.len() - 1) as f64 * fraction).round() as usize;
            values[index]
        };
        let p10 = percentile(&mut self.correlations, 0.10);
        let p50 = percentile(&mut self.correlations, 0.50);
        let p90 = percentile(&mut self.correlations, 0.90);
        let buckets = (0..SIMILARITY_BUCKETS)
            .filter(|&bucket| self.bucket_total[bucket] > 0)
            .map(|bucket| {
                format!(
                    "{:.2}:{}/{}",
                    SIMILARITY_BUCKET_FLOOR + bucket as f64 * SIMILARITY_BUCKET_WIDTH,
                    self.bucket_round_trip[bucket],
                    self.bucket_total[bucket]
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let focal = if self.focal_ratio_before.is_empty() {
            String::new()
        } else {
            format!(
                ", focal-plane ratio p50 {:.3} -> {:.3} -> {:.3} at sigma p50/p90 {:.2}/{:.2} samples, {} capped",
                percentile(&mut self.focal_ratio_before, 0.50),
                percentile(&mut self.focal_ratio_matched, 0.50),
                percentile(&mut self.focal_ratio_after, 0.50),
                percentile(&mut self.focal_passes, 0.50),
                percentile(&mut self.focal_passes, 0.90),
                self.focal_capped
            )
        };
        let round_trip = if self.accepted_round_trips.is_empty() {
            String::new()
        } else {
            let total = self.accepted_round_trips.len();
            let round_trips = &mut self.accepted_round_trips;
            let p50 = percentile(round_trips, 0.50);
            let p90 = percentile(round_trips, 0.90);
            let within = |limit: f64| round_trips.iter().filter(|&&value| value <= limit).count();
            let counted = format!(
                ", round trip at correlation >= {SIMILARITY_REPORTED_FLOOR:.2} p50/p90 {p50:.2}/{p90:.2}px, within 0.40/0.70/1.00px {}/{}/{} of {total}",
                within(0.40),
                within(0.70),
                within(1.00)
            );
            let margin = percentile(&mut self.accepted_peak_margins, 0.10);
            format!("{counted}, second-peak margin p10 {margin:.4}")
        };
        format!(
            "correlation p10/p50/p90 {p10:.3}/{p50:.3}/{p90:.3} over {} pairs{focal}{round_trip}, round trip closure by bucket [{buckets}]",
            self.correlations.len()
        )
    }

    fn median_focal_ratios(&mut self) -> (f64, f64) {
        (
            if self.focal_ratio_before.is_empty() {
                1.0
            } else {
                intra_station::median_of(&mut self.focal_ratio_before)
            },
            if self.focal_ratio_after.is_empty() {
                1.0
            } else {
                intra_station::median_of(&mut self.focal_ratio_after)
            },
        )
    }
}

/// What one control point measured, before the order dependent 需求 2.5 gate.
///
/// Every field here is a function of that control point alone, which is what
/// lets the expensive matching run in parallel while the neighbourhood gate
/// stays a serial row-major walk.
#[derive(Debug, Clone, Copy, Default)]
struct NativeControlPoint {
    /// The cell centre lies on covered base, so its area counts towards the
    /// anchor frame's valid pixel area (需求 2.8).
    covered: bool,
    /// The patch pair was complete, so a match was attempted.
    evaluated: bool,
    /// Bidirectional round trip in native pixels, `0.0` when never measured.
    round_trip_px: f64,
    /// Forward peak correlation, `None` when the patch pair carried no
    /// comparable signal at all.
    best_correlation: Option<f64>,
    /// `true` when the forward match cleared the similarity gates, so a reverse
    /// match was run and the round trip is a real measurement.
    reverse_attempted: bool,
    /// `true` when that round trip closed inside the 需求 2.4 tolerance.
    round_trip_closed: bool,
    /// Gap between the correlation peak and the best hypothesis at least 2.5
    /// samples away, `None` when no hypothesis was scored.
    peak_margin: Option<f64>,
    /// What the focal-plane matched low-pass did to this pair.
    focal_plane: Option<intra_station::FocalPlaneMatch>,
    /// Displacement in analysis units, its correlation, the local (round trip)
    /// and global symmetric reprojection errors in native pixels.  `Some` as soon
    /// as both directions matched, i.e. *before* the 需求 2.4 and 需求 2.6 gates.
    ///
    /// Published separately from [`Self::accepted`] so the gates can be stated as
    /// a filter over a measurement rather than as the measurement itself.
    measured: Option<([f64; 2], f64, f64, f64)>,
    /// [`Self::measured`] once the 需求 2.4 and 需求 2.6 gates have passed.
    accepted: Option<([f64; 2], f64, f64, f64)>,
}

/// Measure one control point: build the two patches, match both ways and apply
/// every gate that does not depend on the other control points (需求 2.4 / 2.6).
#[allow(clippy::too_many_arguments)]
fn measure_native_control_point(
    base: &NativeRefineBase<'_>,
    sampler: &LayerSampler<'_>,
    x: u32,
    y: u32,
    spacing: f64,
    mode: NativeRefineMode,
    match_radius: i32,
    search_samples: i32,
) -> NativeControlPoint {
    let lx = x as f64 * sampler.scale;
    let ly = y as f64 * sampler.scale;
    // 需求 2.8 measures coverage against the anchor frame's valid pixel area;
    // the control point cells whose centre is covered are exactly that area,
    // discretised on this same grid.
    let centre_x = (sampler.left as f64 + lx) as u32;
    let centre_y = (sampler.top as f64 + ly) as u32;
    let mut measurement = NativeControlPoint {
        covered: base.covered(centre_x, centre_y),
        ..Default::default()
    };
    let patch_half = mode.patch_half();
    let side = (2 * patch_half + 1) as usize;
    let half = f64::from(patch_half);
    let mut base_patch = vec![0.0f64; side * side];
    let mut candidate_patch = vec![0.0f64; side * side];
    for py in 0..side {
        for px in 0..side {
            let dx = (px as f64 - half) * spacing;
            let dy = (py as f64 - half) * spacing;
            let gx = sampler.left as f64 + lx + dx;
            let gy = sampler.top as f64 + ly + dy;
            let Some(current) = base.masked(gx, gy) else {
                return measurement;
            };
            let Some(candidate) = sampler.sample(lx + dx, ly + dy) else {
                return measurement;
            };
            base_patch[py * side + px] = luma(current) * 255.0;
            candidate_patch[py * side + px] = luma(candidate) * 255.0;
        }
    }
    // A Capture_Station is a focus bracket: at this control point one of the two
    // sides is near its focal plane and the other is not, and raw-intensity
    // correlation cannot see two different point-spread functions as the same
    // content.  Bring them to a comparable scale first (需求 2.3).
    if mode.matches_focal_plane() {
        measurement.focal_plane = Some(intra_station::match_focal_plane(
            &mut base_patch,
            &mut candidate_patch,
            side,
        ));
    }
    let quantise = |values: &[f64]| {
        GrayImage::from_fn(side as u32, side as u32, |x, y| {
            Luma([values[y as usize * side + x as usize]
                .round()
                .clamp(0.0, 255.0) as u8])
        })
    };
    let a = quantise(&base_patch);
    let b = quantise(&candidate_patch);
    let gates = mode.patch_gates();
    let p = Point2::new(half, half);
    measurement.evaluated = true;
    let forward = probe_warped_patch(
        &a,
        &b,
        &Matrix3::identity(),
        p,
        match_radius,
        search_samples,
        gates,
    );
    measurement.best_correlation = forward.best_correlation;
    measurement.peak_margin = forward.peak_margin;
    let Some(found) = forward.matched else {
        return measurement;
    };
    // The mosaic path keeps its second, stricter correlation floor; the
    // intra-station path states its single floor in `gates` instead of gating
    // twice at two different values.
    if !mode.matches_focal_plane() && found.correlation < 0.90 {
        return measurement;
    }
    measurement.reverse_attempted = true;
    let Some(back) = probe_warped_patch(
        &b,
        &a,
        &Matrix3::identity(),
        found.target,
        match_radius,
        search_samples,
        gates,
    )
    .matched
    else {
        return measurement;
    };
    // Both residuals are patch-sample distances; one sample step is `spacing`
    // native pixels, so scaling by `spacing` puts every measurement below in
    // native pixels of the anchor frame.
    let round_trip_patch = (back.target - p).norm();
    let round_trip_px = round_trip_patch * spacing;
    measurement.round_trip_px = round_trip_px;
    // The symmetric reprojection error of the global model at this control point
    // is how far the two-way match says the frame has to move; after the
    // correction the remaining two-way disagreement is the round trip.  Both are
    // measurements of the completed bidirectional match, so they are recorded
    // before either gate looks at them.
    let global_error_px = (found.target - p).norm() * spacing;
    let d = (found.target - p) * (spacing / sampler.scale);
    measurement.measured = Some((
        [d.x, d.y],
        found.correlation,
        round_trip_px,
        global_error_px,
    ));
    measurement.round_trip_closed = match mode {
        // The legacy path compares the round trip in patch sample units,
        // exactly as it always did.
        NativeRefineMode::LegacyMosaic => {
            round_trip_patch <= intra_station::INTRA_STATION_BIDIRECTIONAL_TOLERANCE_PX
        }
        // 需求 2.4 measures in native pixels and allows 1.0; the stricter 0.40
        // of this path is kept and the measured value is reported.
        NativeRefineMode::IntraStation { .. } => {
            intra_station::bidirectional_consistent(round_trip_px)
        }
    };
    if !measurement.round_trip_closed {
        return measurement;
    }
    if let Some(anchor_long_side) = mode.anchor_long_side() {
        // 需求 2.6: an implausible displacement claim keeps the global model
        // here and counts as a rejected control point.
        if !intra_station::symmetric_error_accepted(global_error_px, anchor_long_side) {
            return measurement;
        }
    }
    measurement.accepted = measurement.measured;
    measurement
}

/// Measure every control point whose matching window lies inside `base`.
///
/// Refine against the selected, native output samples. Upsampling a coarse
/// displacement field alone leaves several pixels of error on 200MP input.
/// These tiny patches keep native refinement independent of canvas size.
///
/// The matching runs in parallel because each control point is an independent
/// measurement; the 需求 2.5 neighbourhood gate then walks the same grid
/// serially in row-major order, so the accepted set is exactly what a fully
/// serial walk accepts and is still a deterministic function of the
/// photographs.
fn collect_native_observations(
    base: &NativeRefineBase<'_>,
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
    spacing: f64,
    mode: NativeRefineMode,
    pass: &mut NativeRefinePass,
) {
    let (match_radius, search_samples) = mode.patch_geometry();
    debug_assert!(
        intra_station::patch_support_fits(match_radius, search_samples, mode.patch_half()),
        "the matching window and its search support must fit the patch buffer"
    );
    debug_assert!(
        !matches!(mode, NativeRefineMode::IntraStation { .. })
            || intra_station::native_search_reach_px(spacing)
                <= intra_station::INTRA_STATION_SEARCH_RADIUS,
        "需求 2.3 bounds the native search reach"
    );
    let columns = (20..width.saturating_sub(20))
        .step_by(NATIVE_REFINE_STEP as usize)
        .collect::<Vec<_>>();
    let rows = (20..height.saturating_sub(20))
        .step_by(NATIVE_REFINE_STEP as usize)
        .collect::<Vec<_>>();
    let measured = rows
        .par_iter()
        .map(|&y| {
            columns
                .iter()
                .map(|&x| {
                    measure_native_control_point(
                        base,
                        sampler,
                        x,
                        y,
                        spacing,
                        mode,
                        match_radius,
                        search_samples,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (grid_row, row) in measured.iter().enumerate() {
        for (grid_column, measurement) in row.iter().enumerate() {
            if measurement.covered {
                pass.valid_cells += 1;
            }
            if !measurement.evaluated {
                continue;
            }
            pass.evaluated += 1;
            pass.similarity.absorb(measurement);
            pass.max_round_trip = pass.max_round_trip.max(measurement.round_trip_px);
            let Some((d, correlation, local_error_px, global_error_px)) = measurement.accepted
            else {
                continue;
            };
            if mode.anchor_long_side().is_some() {
                // 需求 2.5: agree with the already accepted 8-neighbourhood, in
                // native pixels.
                let native = [d[0] * sampler.scale, d[1] * sampler.scale];
                if !intra_station::neighbour_median_consistent(
                    native,
                    pass.accepted_grid.neighbour_median(grid_column, grid_row),
                ) {
                    continue;
                }
                pass.accepted_grid.accept(grid_column, grid_row, native);
            }
            pass.global_errors.push(global_error_px);
            pass.local_errors.push(local_error_px);
            pass.observations.push((
                Point2::new(columns[grid_column] as f64, rows[grid_row] as f64),
                d,
                correlation,
            ));
        }
    }
}

/// Fold one spacing pass into the outcome and, when it carries enough accepted
/// control points, into the sampler's displacement field.
#[allow(clippy::too_many_arguments)]
fn apply_native_pass(
    mut pass: NativeRefinePass,
    sampler: &mut LayerSampler<'_>,
    outcome: &mut NativeRefineOutcome,
    accepted_cells: &mut intra_station::ControlPointGrid,
    cell_area_px: f64,
    spacing: f64,
    is_entry_pass: bool,
    mode: NativeRefineMode,
) {
    accepted_cells.absorb(&pass.accepted_grid);
    // The legacy comparison path never populated the grid and never reported
    // these counts, so it keeps counting one pass at a time.
    let accepted = if mode == NativeRefineMode::LegacyMosaic {
        pass.observations.len()
    } else {
        accepted_cells.accepted_count()
    };
    outcome.evaluated_control_points = pass.evaluated;
    outcome.accepted_control_points = accepted;
    outcome.rejected_control_point_ratio = if pass.evaluated == 0 {
        0.0
    } else {
        1.0 - outcome.accepted_control_points as f64 / pass.evaluated as f64
    };
    outcome.max_bidirectional_round_trip_px = pass.max_round_trip;
    // 需求 2.7 compares the refined frame against the *global* model, so the
    // baseline has to be measured while the global model is still what the
    // sampler carries.  That is the entry pass — but a pass whose accepted
    // control points are too few to form a field leaves the sampler untouched,
    // so the *next* pass is still measuring the global model and is the first
    // one that can supply a baseline.  Claiming a baseline from a pass that
    // supplied none is how the comparison ended up against 0.00px.
    let no_baseline_yet =
        outcome.global_baseline_samples < intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES;
    if (is_entry_pass || no_baseline_yet) && !pass.global_errors.is_empty() {
        outcome.global_baseline_samples = pass.global_errors.len();
        outcome.global_median_symmetric_error_px =
            intra_station::median_of(&mut pass.global_errors);
    }
    outcome.local_baseline_samples = pass.local_errors.len();
    outcome.local_median_symmetric_error_px = intra_station::median_of(&mut pass.local_errors);
    outcome.inlier_area_coverage = intra_station::inlier_area_coverage(
        accepted,
        cell_area_px,
        pass.valid_cells as f64 * cell_area_px,
    );
    #[cfg(test)]
    {
        outcome.last_pass_accepted_control_points = pass.accepted_grid.accepted_count();
        outcome.valid_cells = pass.valid_cells;
        outcome.cell_area_px = cell_area_px;
    }
    let (focal_before, focal_after) = pass.similarity.median_focal_ratios();
    outcome.focal_plane_ratio_before = focal_before;
    outcome.focal_plane_ratio_after = focal_after;
    println!(
        "    - Patch similarity ({spacing:.0}px samples): {}",
        pass.similarity.summary()
    );
    if pass.observations.len() < intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES {
        return;
    }
    let observations = &pass.observations;
    let mut field = sampler.residual.clone();
    for gy in 0..field.height {
        for gx in 0..field.width {
            let p = Point2::new(gx as f64 * field.step, gy as f64 * field.step);
            let mut sum = [0.0; 2];
            let mut total = 0.0f64;
            for (q, d, correlation) in observations {
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

/// 需求 2.7 / 2.8: the per-frame verdict of a finished native refinement.
fn finalize_native_refinement(
    outcome: &mut NativeRefineOutcome,
    sampler: &mut LayerSampler<'_>,
    entry_field: Field<2>,
    mode: NativeRefineMode,
) {
    if mode == NativeRefineMode::LegacyMosaic {
        return;
    }
    // 需求 2.7: keep the local field only when it lowered the frame's median
    // symmetric reprojection error.  A refinement that merely moved the error
    // around is a worse description of the same photographs than the global
    // model, so the whole frame goes back to the state it entered with.  When
    // one of the two medians has too few samples to exist the comparison is
    // unevaluable, and the requirement's condition — "not lower than" — is not
    // satisfied by an absent value, so nothing is reverted on its behalf.
    outcome.local_field_verdict = intra_station::local_field_verdict(
        outcome.local_median_symmetric_error_px,
        outcome.local_baseline_samples,
        outcome.global_median_symmetric_error_px,
        outcome.global_baseline_samples,
    );
    match outcome.local_field_verdict {
        intra_station::LocalFieldVerdict::Kept => {}
        intra_station::LocalFieldVerdict::Reverted => {
            sampler.residual = entry_field;
            outcome.reverted_to_global = true;
            println!(
                "    - Native refinement reverted: median symmetric error {:.2}px over {} points is not better than the global model's {:.2}px over {} points",
                outcome.local_median_symmetric_error_px,
                outcome.local_baseline_samples,
                outcome.global_median_symmetric_error_px,
                outcome.global_baseline_samples
            );
            degradation::record_run_degradation(
                degradation::INTRA_STATION_LOCAL_FALLBACK,
                serde_json::json!({
                    "stage": "intra_station_native_refinement",
                    "local_median_symmetric_error_px": outcome.local_median_symmetric_error_px,
                    "local_baseline_samples": outcome.local_baseline_samples,
                    "global_median_symmetric_error_px": outcome.global_median_symmetric_error_px,
                    "global_baseline_samples": outcome.global_baseline_samples,
                    "accepted_control_points": outcome.accepted_control_points,
                }),
            );
        }
        intra_station::LocalFieldVerdict::Indeterminate => {
            println!(
                "    - Native refinement indeterminate: {} local and {} global control points cannot form the 需求 2.7 comparison; the measured field is kept",
                outcome.local_baseline_samples, outcome.global_baseline_samples
            );
            degradation::record_run_degradation(
                degradation::INTRA_STATION_LOCAL_BASELINE_INDETERMINATE,
                serde_json::json!({
                    "stage": "intra_station_native_refinement",
                    "local_median_symmetric_error_px": outcome.local_median_symmetric_error_px,
                    "local_baseline_samples": outcome.local_baseline_samples,
                    "global_median_symmetric_error_px": outcome.global_median_symmetric_error_px,
                    "global_baseline_samples": outcome.global_baseline_samples,
                    "minimum_baseline_samples":
                        intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES,
                    "accepted_control_points": outcome.accepted_control_points,
                }),
            );
        }
    }
    // 需求 2.8: a frame whose accepted control point cells cover less than 20%
    // of the anchor frame's valid pixel area is a registration failure.
    if outcome.inlier_area_coverage < intra_station::INTRA_STATION_MIN_INLIER_AREA_COVERAGE {
        println!(
            "    - Intra-station registration failed: inlier area coverage {:.3} below {:.2}",
            outcome.inlier_area_coverage,
            intra_station::INTRA_STATION_MIN_INLIER_AREA_COVERAGE
        );
        degradation::record_run_degradation(
            degradation::INTRA_STATION_REGISTRATION_FAILED,
            serde_json::json!({
                "stage": "intra_station_native_refinement",
                "inlier_area_coverage": outcome.inlier_area_coverage,
                "minimum_inlier_area_coverage":
                    intra_station::INTRA_STATION_MIN_INLIER_AREA_COVERAGE,
                "accepted_control_points": outcome.accepted_control_points,
                "evaluated_control_points": outcome.evaluated_control_points,
            }),
        );
    }
}

/// Native patch refinement against an in-memory canvas (需求 2.3–2.8).
fn refine_native_layer(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
    mode: NativeRefineMode,
) -> NativeRefineOutcome {
    let mut outcome = NativeRefineOutcome::default();
    if mode == NativeRefineMode::LegacyMosaic && sampler.scale <= 2.0 {
        return outcome;
    }
    outcome.skipped = false;
    outcome.control_point_spacing_px = native_control_point_spacing_px(sampler.scale);
    let entry_field = sampler.residual.clone();
    let cell_area_px = (NATIVE_REFINE_STEP as f64 * sampler.scale).powi(2);
    let mut accepted_cells = NativeRefinePass::new(width, height).accepted_grid;
    let base = NativeRefineBase {
        image: base,
        mask: base_mask,
        left: 0,
        top: 0,
    };
    for (index, spacing) in [2.0, 1.0].into_iter().enumerate() {
        let mut pass = NativeRefinePass::new(width, height);
        collect_native_observations(&base, sampler, width, height, spacing, mode, &mut pass);
        apply_native_pass(
            pass,
            sampler,
            &mut outcome,
            &mut accepted_cells,
            cell_area_px,
            spacing,
            index == 0,
            mode,
        );
    }
    finalize_native_refinement(&mut outcome, sampler, entry_field, mode);
    outcome
}

/// Native patch refinement against a tiled canvas (需求 2.3–2.8).
///
/// This is the same measurement as [`refine_native_layer`]; only the base pixels
/// arrive one tile at a time, because the streaming compositor is used exactly
/// when the canvas is too large to hold.  Without this entry point the whole of
/// 需求 2.3–2.9 would be unreachable for every multi-station stack, which is the
/// only kind of stack the streaming compositor ever sees.
#[allow(clippy::too_many_arguments)]
fn refine_native_layer_streaming(
    store: &StreamingMosaicStore,
    owner: Option<u8>,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
    layer: (u32, u32, u32, u32),
    mode: NativeRefineMode,
) -> Result<NativeRefineOutcome, String> {
    let mut outcome = NativeRefineOutcome {
        skipped: false,
        control_point_spacing_px: native_control_point_spacing_px(sampler.scale),
        ..Default::default()
    };
    let entry_field = sampler.residual.clone();
    let cell_area_px = (NATIVE_REFINE_STEP as f64 * sampler.scale).powi(2);
    let mut accepted_cells = NativeRefinePass::new(width, height).accepted_grid;
    let (left, right, top, bottom) = layer;
    for (index, spacing) in [2.0, 1.0].into_iter().enumerate() {
        let mut pass = NativeRefinePass::new(width, height);
        for tile_row in top / store.tile_size..=bottom / store.tile_size {
            for tile_column in left / store.tile_size..=right / store.tile_size {
                let (tile_left, tile_top, tile_width, tile_height) =
                    store.tile_extent(tile_column, tile_row);
                let (tile, covered) = store.load_tile(tile_column, tile_row)?;
                // A later focal plane of a station must register against its own
                // camera position: pixels owned by another station are available
                // for colour matching but never for residual motion.
                let tile_mask = match owner {
                    Some(group) => {
                        let tile_owner = store.load_owner_tile(tile_column, tile_row);
                        GrayImage::from_fn(tile_width, tile_height, |x, y| {
                            Luma([
                                if covered.get_pixel(x, y)[0] != 0
                                    && tile_owner.get_pixel(x, y)[0] == group
                                {
                                    255
                                } else {
                                    0
                                },
                            ])
                        })
                    }
                    None => covered,
                };
                if tile_mask.as_raw().iter().all(|&value| value == 0) {
                    continue;
                }
                let base = NativeRefineBase {
                    image: &tile,
                    mask: &tile_mask,
                    left: tile_left,
                    top: tile_top,
                };
                collect_native_observations(
                    &base, sampler, width, height, spacing, mode, &mut pass,
                );
            }
        }
        apply_native_pass(
            pass,
            sampler,
            &mut outcome,
            &mut accepted_cells,
            cell_area_px,
            spacing,
            index == 0,
            mode,
        );
    }
    finalize_native_refinement(&mut outcome, sampler, entry_field, mode);
    Ok(outcome)
}

/// Distance between neighbouring control points in native pixels (需求 2.3).
///
/// [`NATIVE_REFINE_STEP`] is a step in *analysis* units, so the native spacing
/// is that step times the layer's analysis scale.  Published so the default
/// station path can compare the measured spacing against the Focus_Fuser's
/// ownership cell side.
pub(crate) fn native_control_point_spacing_px(analysis_scale: f64) -> f64 {
    NATIVE_REFINE_STEP as f64 * analysis_scale
}

/// The displacement field a native refinement left behind, addressed in canvas
/// pixels (需求 2.3–2.7).
///
/// The refinement measures and corrects in the layer's analysis units; a caller
/// that already holds the layer rendered on the canvas needs the same
/// correction in canvas pixels, which is what [`Self::at`] returns.
pub(crate) struct NativeResidualDisplacement {
    field: Field<2>,
    left: u32,
    top: u32,
    scale: f64,
}

impl NativeResidualDisplacement {
    /// Canvas-pixel offset to add to a canvas position before sampling the
    /// layer, i.e. exactly what [`LayerSampler::source_point`] adds.
    pub(crate) fn at(&self, canvas_x: f64, canvas_y: f64) -> [f64; 2] {
        let delta = self.field.at(
            (canvas_x - f64::from(self.left)) / self.scale,
            (canvas_y - f64::from(self.top)) / self.scale,
        );
        [delta[0] * self.scale, delta[1] * self.scale]
    }

    /// Largest correction the field carries, in canvas pixels.  Reported so a
    /// run log shows whether the refinement moved anything at all.
    pub(crate) fn max_offset_px(&self) -> f64 {
        self.field
            .values
            .iter()
            .map(|delta| delta[0].hypot(delta[1]) * self.scale)
            .fold(0.0, f64::max)
    }
}

/// One frame of the Intra_Station_Registrar on the default station path
/// (需求 2.2–2.9).
pub(crate) struct StationFrameRegistration {
    /// The `intra_station[*].frames[*]` entry of 需求 2.9.
    pub(crate) record: IntraStationFrameRecord,
    /// The correction to apply to the already rendered layer, or `None` when
    /// nothing was measured.
    pub(crate) displacement: Option<NativeResidualDisplacement>,
    /// Control points the final pass measured, for the run log.
    pub(crate) evaluated_control_points: usize,
}

/// Register one non-anchor frame of a Capture_Station against the station's
/// already composited canvas at native resolution (需求 2.2–2.9).
///
/// This is the entry point of the *default* station path.  It runs the same
/// measurement as the mosaic comparison path — [`refine_layer`] twice at the
/// analysis resolution of 需求 2.2, then [`refine_native_layer`] with the
/// control point gates of 需求 2.3–2.8 — over a [`LayerSampler`] built from the
/// frame's own source pixels, and hands back both the report record and the
/// displacement field, so the caller's already rendered layer can be corrected
/// by exactly what was measured.
///
/// `layer` is the frame's canvas footprint as the caller rendered it, and
/// `transform` must be the pose that produced it: the sampler reproduces that
/// geometry, which is what makes the measured displacement applicable to those
/// pixels.
#[allow(clippy::too_many_arguments)]
pub(crate) fn register_station_frame_native(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    info: &ImageInfo,
    source: &Rgb32FImage,
    transform: &Matrix3<f64>,
    projection: Projection,
    offset: (f64, f64),
    layer: (u32, u32, u32, u32),
    anchor_long_side: u32,
    into_anchor: Matrix3<f64>,
) -> Option<StationFrameRegistration> {
    let inverse = transform.try_inverse()?;
    let (left, right, top, bottom) = layer;
    if right <= left || bottom <= top {
        return None;
    }
    let layer_width = right - left + 1;
    let layer_height = bottom - top + 1;
    // 需求 2.2: the two global + local rounds run at an analysis resolution whose
    // long side stays inside INTRA_STATION_ANALYSIS_LONG_SIDE; the native patch
    // refinement then runs on the native samples this sampler reaches.
    let scale = (layer_width.max(layer_height) as f64 / ANALYSIS_LONG_SIDE as f64).max(1.0);
    let analysis_width = (layer_width as f64 / scale).ceil() as u32;
    let analysis_height = (layer_height as f64 / scale).ceil() as u32;
    let mut sampler = LayerSampler {
        info,
        source,
        source_divisor: 1.0,
        inverse,
        projection,
        offset,
        left,
        top,
        scale,
        residual: Field::new(analysis_width, analysis_height, GRID_STEP as f64),
    };
    for _ in 0..intra_station::INTRA_STATION_ANALYSIS_ROUNDS {
        refine_layer(
            base,
            base_mask,
            &mut sampler,
            analysis_width,
            analysis_height,
        );
    }
    let outcome = refine_native_layer(
        base,
        base_mask,
        &mut sampler,
        analysis_width,
        analysis_height,
        NativeRefineMode::IntraStation { anchor_long_side },
    );
    let record = intra_station_frame_record(&info.filename, into_anchor, &outcome);
    Some(StationFrameRegistration {
        record,
        displacement: (!outcome.skipped).then(|| NativeResidualDisplacement {
            field: sampler.residual,
            left,
            top,
            scale,
        }),
        evaluated_control_points: outcome.evaluated_control_points,
    })
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
fn acutance(sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>, x: f64, y: f64) -> f64 {
    acutance_with_step(sample, x, y, 1.0)
}

/// Number of patch grid points per axis of one [`acutance`] measurement.
pub(super) const ACUTANCE_PATCH_POINTS: u32 = 13;

/// [`acutance`] with an explicit spacing between the 13×13 patch grid points.
///
/// `step = 1.0` reproduces the legacy measurement exactly (one grid point per
/// native pixel, a ~13px window).  The Focus_Fuser passes a larger step so the
/// effective window reaches the 32 native pixels 需求 3.2 asks for; the gradient
/// stays "four sampling steps apart", so only the sample *positions* scale.
pub(super) fn acutance_with_step(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    step: f64,
) -> f64 {
    let step = if step.is_finite() && step > 0.0 {
        step
    } else {
        1.0
    };
    let mut patch = [[[0.0; 3]; 13]; 13];
    for (j, row) in patch.iter_mut().enumerate() {
        for (i, value) in row.iter_mut().enumerate() {
            let Some(p) = sample(x + (i as f64 - 6.0) * step, y + (j as f64 - 6.0) * step) else {
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
    sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    cell_focus_with_step(sample, x, y, cell_size, 1.0)
}

/// Probe positions of one ownership cell, relative to the cell centre.
///
/// Five probes cover the centre and all four corners. A stroke can cross any
/// cell edge, so sampling only one diagonal can still land entirely on quiet
/// paper and hide an in-focus candidate.  Published so the Focus_Fuser can
/// assert that a candidate and the current composite measure the *same*
/// positions inside a cell (需求 3.2) instead of assuming it.
pub(super) fn cell_probe_offsets(cell_size: f64) -> [(f64, f64); 5] {
    let radius = (cell_size * 0.28).clamp(2.0, 18.0);
    [
        (-radius, -radius),
        (radius, -radius),
        (0.0, 0.0),
        (-radius, radius),
        (radius, radius),
    ]
}

/// [`cell_focus`] with an explicit [`acutance_with_step`] window spacing.
pub(super) fn cell_focus_with_step(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
    step: f64,
) -> f64 {
    let mut values = cell_probe_offsets(cell_size)
        .iter()
        .map(|(dx, dy)| acutance_with_step(&mut sample, x + dx, y + dy, step))
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
    let mut values = cell_probe_offsets(cell_size)
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

/// Fractional positions inside an ownership cell at which the candidate and the
/// current composite are compared.  A single sample at the cell centre can land
/// on canvas and miss a brush stroke that crosses the same cell near an edge.
pub(super) const CELL_DIFFERENCE_OFFSETS: [f64; 5] = [0.15, 0.325, 0.5, 0.675, 0.85];

/// Low-pass absolute colour differences inside one ownership cell.
///
/// Both closures are called with the *same* fractional cell offsets, which is
/// how 需求 3.3's "candidate versus current composite in the same cell" stays a
/// property of the code rather than of the caller: each closure maps the shared
/// offset into its own coordinate space and returns `None` outside its coverage.
///
/// Individual high-frequency pixels are deliberately not compared. Defocus
/// changes brush-edge samples substantially even when the geometry is correct; a
/// displaced contour still changes the local mean over a 3×3 neighbourhood of
/// probes and remains measurable.  The reduction of these values (high
/// percentile for the legacy mosaic path, median for the Focus_Fuser) is left to
/// the caller.
pub(super) fn cell_low_pass_differences(
    mut candidate_at: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    mut base_at: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
) -> Vec<f64> {
    let side = CELL_DIFFERENCE_OFFSETS.len();
    let mut samples = vec![None; side * side];
    for (yi, y_offset) in CELL_DIFFERENCE_OFFSETS.iter().copied().enumerate() {
        for (xi, x_offset) in CELL_DIFFERENCE_OFFSETS.iter().copied().enumerate() {
            let Some(candidate) = candidate_at(x_offset, y_offset) else {
                continue;
            };
            let Some(base_pixel) = base_at(x_offset, y_offset) else {
                continue;
            };
            let candidate = candidate.0.map(f64::from);
            let base_pixel = base_pixel.0.map(f64::from);
            if candidate
                .iter()
                .chain(base_pixel.iter())
                .all(|v| v.is_finite())
            {
                samples[yi * side + xi] = Some((candidate, base_pixel));
            }
        }
    }
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
                let neighbour = y as usize * CELL_DIFFERENCE_OFFSETS.len() + x as usize;
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
    disagreements
}

fn ownership_disagreement(
    base: &Rgb32FImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    ax: f64,
    ay: f64,
    cell_size: f64,
) -> f64 {
    // Retain a high percentile so a displaced contour vetoes the candidate even
    // when most of the cell is quiet paper.  This is the legacy comparison
    // path's reduction; the Focus_Fuser uses the median 需求 3.3 prescribes.
    let mut disagreements = cell_low_pass_differences(
        |x_offset, y_offset| {
            let sample_ax = ax + cell_size * x_offset;
            let sample_ay = ay + cell_size * y_offset;
            sampler
                .sample(sample_ax * sampler.scale, sample_ay * sampler.scale)
                .map(|pixel| adjusted(pixel, tone.at(sample_ax, sample_ay)))
        },
        |x_offset, y_offset| {
            let sample_ax = ax + cell_size * x_offset;
            let sample_ay = ay + cell_size * y_offset;
            rgb_at(
                base,
                sampler.left as f64 + sample_ax * sampler.scale,
                sampler.top as f64 + sample_ay * sampler.scale,
            )
        },
    );
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

fn mask_covered_bounds(mask: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = mask.width();
    let mut min_y = mask.height();
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel[0] == 0 {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    found.then_some((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

pub(crate) fn detail_preserving_mosaic<R: Runtime, F>(
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
    // Everything past the streaming branch above has `capture_group_count <= 1`,
    // so this loop composites one Capture_Station: layer 0 is the frame every
    // later layer registers against, which makes this the intra-station path of
    // 需求 2.2–2.9.  A `sequence_gap_aware` run is different in kind — a filename
    // bridge joined captures that share no measured overlap — and stays on the
    // legacy comparison behaviour, early exit included.
    let intra_station_path = !sequence_gap_aware;
    let anchor_long_side = images
        .first()
        .map(|info| info.width.max(info.height))
        .unwrap_or(0);
    let native_refine_mode = if intra_station_path {
        NativeRefineMode::IntraStation { anchor_long_side }
    } else {
        NativeRefineMode::LegacyMosaic
    };
    let station_index = images
        .first()
        .and_then(|info| capture_group_ids.and_then(|ids| ids.get(&info.id)).copied())
        .map(usize::from)
        .unwrap_or(0);
    let anchor_path = images
        .first()
        .map(|info| info.filename.clone())
        .unwrap_or_default();
    // 需求 2.1: the anchor frame carries the identity transform, so its record
    // is the reference every other frame's transform is expressed against.
    let anchor_to_world_inverse = images
        .first()
        .and_then(|info| homographies.get(&info.id))
        .and_then(|transform| transform.try_inverse());
    // 需求 12.1 / 12.2: the members this station fuses, each with the verdict
    // the registrar reaches below.  The anchor carries the identity transform,
    // so it is registered by definition.
    let mut station_members = Vec::<StationMember>::new();
    if intra_station_path {
        if let Some(info) = images.first() {
            intra_station::record_run_frame(
                station_index,
                &anchor_path,
                IntraStationFrameRecord {
                    path: info.filename.clone(),
                    status: IntraStationFrameStatus::Anchor,
                    inlier_area_coverage: 1.0,
                    ..Default::default()
                },
            );
            station_members.push(StationMember {
                path: info.filename.clone(),
                median_sharpness: station_member_median_sharpness(info),
                status: IntraStationFrameStatus::Anchor,
            });
        }
    }
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
            // 需求 2.2: two global + local rounds at the analysis resolution,
            // then the native patch refinement.
            for _ in 0..intra_station::INTRA_STATION_ANALYSIS_ROUNDS {
                refine_layer(&result, &mask, &mut sampler, aw, ah);
            }
            let outcome =
                refine_native_layer(&result, &mask, &mut sampler, aw, ah, native_refine_mode);
            if intra_station_path {
                let into_anchor = anchor_to_world_inverse
                    .map(|inverse| inverse * transform)
                    .unwrap_or_else(Matrix3::identity);
                let record = intra_station_frame_record(&info.filename, into_anchor, &outcome);
                intra_station::record_run_frame(station_index, &anchor_path, record.clone());
                // 需求 12.5: inlier spatial support below 20%, or an inlier
                // median symmetric reprojection error above 0.01 x the long
                // side, keeps this Source_RAW out of every Capture_Station.
                // 需求 12.2: it is therefore excluded from this station's
                // fusion, with its absolute path and reason identifier
                // reported one by one.  A frame whose evidence was never
                // measured (no control point was even evaluated) has nothing
                // to judge and keeps the behaviour it had before.
                let rejection = (outcome.evaluated_control_points > 0)
                    .then(|| {
                        group_join_rejection(&GroupJoinEvidence {
                            path: info.filename.clone(),
                            inlier_area_coverage: record.inlier_area_coverage,
                            median_symmetric_error_px: record.median_symmetric_error_px,
                            anchor_long_side_px: anchor_long_side,
                        })
                    })
                    .flatten();
                station_members.push(StationMember {
                    path: info.filename.clone(),
                    median_sharpness: station_member_median_sharpness(info),
                    // The member status is what this station actually did with
                    // the frame, so the plan below can never claim a frame was
                    // excluded while its pixels are on the canvas.  With no
                    // control point evaluated there is no local evidence to
                    // judge and the frame is placed by its global model, which
                    // is exactly `GlobalFallback`.
                    status: match (rejection.is_some(), record.status) {
                        (true, _) => IntraStationFrameStatus::Failed,
                        (false, IntraStationFrameStatus::Failed) => {
                            IntraStationFrameStatus::GlobalFallback
                        }
                        (false, status) => status,
                    },
                });
                if let Some(rejection) = rejection {
                    println!(
                        "    - Rejected from every Capture_Station: inlier support {:.3} (minimum {:.2}), median symmetric error {:.2}px (limit {:.2}px)",
                        rejection.inlier_area_coverage,
                        rejection.minimum_inlier_area_coverage,
                        rejection.median_symmetric_error_px,
                        rejection.symmetric_error_limit_px
                    );
                    record_run_group_join_rejection(station_index, &rejection);
                    continue;
                }
            }
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
    // 需求 12.1 / 12.2: the station's fusion path, with every excluded
    // Source_RAW listed individually.  `station_members` carries what the loop
    // above really fused, so the recorded plan and the pixels agree.
    if intra_station_path {
        let plan = plan_station_fusion(&station_members);
        let entries = station_plan_entries(station_index, &station_members, &plan);
        if !entries.is_empty() {
            println!(
                "  - Capture_Station {station_index} fused {} of {} frame(s) ({:?})",
                plan.fused.len(),
                station_members.len(),
                plan.mode
            );
            for excluded in &plan.excluded {
                println!("    - Excluded '{}': {}", excluded.path, excluded.reason);
            }
        }
        record_run_entries(&entries);
    }
    let covered = mask.as_raw().iter().filter(|&&v| v != 0).count();
    if covered == 0 {
        return Err("The aligned stack has no covered pixels.".into());
    }
    // Keep the complete union of real source coverage. Cropping to the largest
    // hole-free rectangle silently removes photographed frame edges whenever
    // the scan is slightly rotated or the coverage is non-rectangular.
    #[cfg(test)]
    diagnostics::capture_crop(&mask);
    let (crop_x, crop_y, crop_width, crop_height) =
        mask_covered_bounds(&mask).ok_or("The aligned stack has no covered pixels.")?;
    let output =
        image::imageops::crop_imm(&result, crop_x, crop_y, crop_width, crop_height).to_image();
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
    let mut group_tone_relations = Vec::<GroupToneRelation>::new();
    // 需求 2.1 / 2.9: the frame every later frame of a Capture_Station registers
    // against is that station's first layer in render order.  Its absolute path,
    // inverse world pose and native long side are what the other frames' records
    // are expressed against.
    let mut station_anchors = HashMap::<u8, (String, Option<Matrix3<f64>>, u32)>::new();
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
        if first_capture_group_layer {
            station_anchors.insert(
                capture_group_id,
                (
                    info.filename.clone(),
                    transform.try_inverse(),
                    info.width.max(info.height),
                ),
            );
            intra_station::record_run_frame(
                usize::from(capture_group_id),
                &info.filename,
                IntraStationFrameRecord {
                    path: info.filename.clone(),
                    status: IntraStationFrameStatus::Anchor,
                    inlier_area_coverage: 1.0,
                    ..Default::default()
                },
            );
        }
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
            // 需求 2.2: two global + local rounds at the analysis resolution,
            // then the native patch refinement below.
            for _ in 0..intra_station::INTRA_STATION_ANALYSIS_ROUNDS {
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
            // 需求 2.2–2.9: the native patch refinement that follows the two
            // analysis rounds.  Only a later layer of a station already on the
            // canvas is an intra-station registration; the first layer of a
            // station has no same-station pixels to register against, so it is
            // the anchor and keeps its global pose.
            if !first_capture_group_layer {
                let anchor = station_anchors.get(&capture_group_id);
                let anchor_long_side = anchor
                    .map(|&(_, _, long_side)| long_side)
                    .unwrap_or_else(|| info.width.max(info.height));
                let outcome = refine_native_layer_streaming(
                    &store,
                    Some(capture_group_id),
                    &mut sampler,
                    analysis_width,
                    analysis_height,
                    (left, right, top, bottom),
                    NativeRefineMode::IntraStation { anchor_long_side },
                )?;
                let into_anchor = anchor
                    .and_then(|&(_, inverse, _)| inverse)
                    .map(|inverse| inverse * transform)
                    .unwrap_or_else(Matrix3::identity);
                intra_station::record_run_frame(
                    usize::from(capture_group_id),
                    anchor.map(|(path, _, _)| path.as_str()).unwrap_or_default(),
                    intra_station_frame_record(&info.filename, into_anchor, &outcome),
                );
            }
            let (candidate, candidate_mask) =
                streaming_candidate_analysis(&sampler, analysis_width, analysis_height);
            if first_capture_group_layer {
                group_tone_relations.extend(streaming_group_tone_relations_from_analysis(
                    &base_analysis,
                    &base_analysis_mask,
                    &base_analysis_owner,
                    &candidate,
                    &candidate_mask,
                    capture_group_id,
                ));
            }
            // A focus bracket may contain several camera positions.  Once a
            // position has entered the mosaic, estimate its colour correction
            // only from overlap owned by that same position.  Comparing a
            // later position against a neighbouring group's pixels compounds
            // exposure drift into broad vertical bands.
            let tone = if first_capture_group_layer {
                // Solve all camera-position gains together after ownership is
                // final. Applying them here would accumulate drift along the
                // scan sequence.
                Field::new(analysis_width, analysis_height, 112.0)
            } else {
                streaming_tone_field_from_analysis(
                    &base_analysis,
                    refinement_mask,
                    &candidate,
                    &candidate_mask,
                )
            };
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
        // Keep focus ownership binary away from a seam, but cross-fade one
        // decision cell on either side of the cut. Rendering the raw graph
        // labels directly reproduced their staircase at native resolution.
        // This narrow transition hides the grid without averaging detail over
        // the rest of either source layer.
        let soft_decision = image::imageops::blur(&decision, STREAMING_OWNERSHIP_FEATHER);
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
                        &soft_decision,
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
        .covered_bounds()?
        .ok_or("The aligned stack has no covered pixels.")?;
    println!(
        "  - Streaming complete source-union bounds: {}x{} at ({}, {})",
        crop.2, crop.3, crop.0, crop.1
    );
    let mut output = materialize_streaming_focus_output(&store, crop)?;
    // The per-layer tone field is estimated only from pixels already owned by
    // the same capture group. That preserves focus detail, but the first layer
    // of each new camera position has no same-group reference and can leave a
    // source-sized exposure step. Reconcile those remaining group constants
    // after ownership is final. The solver uses only blurred, non-black paper
    // samples across robust owner relations and feathers a bounded constant
    // gain; it never averages the high-frequency painting detail.
    if let Some((gains, scale)) = streaming_seam_harmonization(&store, crop, &group_tone_relations)?
    {
        apply_streaming_seam_harmonization(&mut output, &gains, scale);
    }
    smooth_streaming_low_frequency_illumination(&mut output);
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

/// Finalize a tiled focus canvas without inferring colour from ownership
/// boundaries.  Keeping this policy in one helper makes it harder to
/// accidentally reintroduce the broad group-gain pass when changing the
/// streaming renderer.
fn materialize_streaming_focus_output(
    store: &StreamingMosaicStore,
    crop: (u32, u32, u32, u32),
) -> Result<Rgb32FImage, String> {
    store.materialize(crop)
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
    soft_decision: &GrayImage,
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
    let decision_width = soft_decision.width();
    let decision_height = soft_decision.height();
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
            for x in x_start..=x_end {
                let layer_x = x.saturating_sub(sampler.left) as f64;
                let analysis_x = layer_x / scale;
                let alpha = sample_decision_alpha(
                    soft_decision,
                    analysis_x / analysis_width.max(1) as f64 * decision_width as f64,
                    ay / analysis_height.max(1) as f64 * decision_height as f64,
                );
                if alpha <= 1.0 / 255.0 {
                    continue;
                }
                let local_x = x - tile_left;
                let Some(pixel) = sampler.sample(layer_x, layer_y) else {
                    continue;
                };
                let adjusted_pixel = adjusted(pixel, tone.at(analysis_x, ay));
                let start = local_x as usize * 3;
                if covered[local_x as usize] == 0 {
                    row[start..start + 3].copy_from_slice(&adjusted_pixel.0);
                } else {
                    for channel in 0..3 {
                        row[start + channel] =
                            row[start + channel] * (1.0 - alpha) + adjusted_pixel[channel] * alpha;
                    }
                }
                covered[local_x as usize] = 255;
                if alpha >= 0.5 {
                    owner[local_x as usize] = capture_group_id;
                }
            }
        });
}

/// Sample the blurred ownership decision at pixel coordinates rather than
/// assigning one constant alpha to an entire analysis cell.  Constant-cell
/// writes create visible square steps and average sharp source detail over a
/// wide seam.  Bilinear sampling confines the transition to the already
/// narrow blurred decision boundary.
fn sample_decision_alpha(image: &GrayImage, x: f64, y: f64) -> f32 {
    if image.width() == 0 || image.height() == 0 || !x.is_finite() || !y.is_finite() {
        return 0.0;
    }
    let x = x.clamp(0.0, image.width().saturating_sub(1) as f64);
    let y = y.clamp(0.0, image.height().saturating_sub(1) as f64);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width().saturating_sub(1));
    let y1 = (y0 + 1).min(image.height().saturating_sub(1));
    let fx = x as f32 - x0 as f32;
    let fy = y as f32 - y0 as f32;
    let a = f32::from(image.get_pixel(x0, y0)[0]) * (1.0 - fx)
        + f32::from(image.get_pixel(x1, y0)[0]) * fx;
    let b = f32::from(image.get_pixel(x0, y1)[0]) * (1.0 - fx)
        + f32::from(image.get_pixel(x1, y1)[0]) * fx;
    (a * (1.0 - fy) + b * fy) / 255.0
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

/// Test-only access to the Intra_Station_Registrar control point internals of
/// 需求 2.4 to 2.8.
///
/// Property 6 lives in `super::stack_pipeline::properties`, a sibling module that
/// cannot see `LayerSampler`, `NativeRefinePass` or `measure_native_control_point`.
/// The two drivers below assemble exactly what the in-memory compositor assembles
/// — a native-resolution base with a full coverage mask, a layer sampler at
/// `scale = 1.0` and an identity residual field — and then call the production
/// functions unchanged, so the property tests the refinement rather than a model
/// of it.
#[cfg(test)]
pub(crate) mod intra_station_test_access {
    use super::*;

    /// Everything one control point measured, plus the verdict the production
    /// pass gave it.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct ControlPointProbe {
        /// Position in the refinement grid, row major.
        pub(crate) column: usize,
        pub(crate) row: usize,
        /// The cell centre lies on covered base (需求 2.8).
        pub(crate) covered: bool,
        /// The patch pair was complete, so a match was attempted.
        pub(crate) evaluated: bool,
        /// Displacement of a completed bidirectional match, in native pixels of
        /// the anchor frame, before the 需求 2.4 / 2.6 gates.  `None` when the
        /// matcher's similarity gates or the reverse match stopped the
        /// measurement, i.e. when there is no located match to judge.
        pub(crate) displacement_px: Option<[f64; 2]>,
        /// Bidirectional round trip of that match, in native pixels (需求 2.4).
        pub(crate) round_trip_px: f64,
        /// Symmetric reprojection error of the global model there, in native
        /// pixels (需求 2.6).
        pub(crate) symmetric_error_px: f64,
        /// Forward peak correlation, `None` when no hypothesis was scored.
        pub(crate) best_correlation: Option<f64>,
        /// The displacement the production pass accepted at this position, i.e.
        /// what the 需求 2.5 neighbourhood gate let through.
        pub(crate) accepted_px: Option<[f64; 2]>,
    }

    /// One native refinement pass, measured control point by control point.
    ///
    /// Returns every grid position together with the accepted set of the real
    /// pass and the observation positions that entered the displacement field.
    pub(crate) fn probe_native_pass(
        base: &Rgb32FImage,
        layer: &Rgb32FImage,
        anchor_long_side: u32,
        spacing: f64,
    ) -> (Vec<ControlPointProbe>, Vec<(usize, usize)>, usize) {
        let (width, height) = base.dimensions();
        let info = image_info_for_probe(layer);
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        let sampler = LayerSampler {
            info: &info,
            source: layer,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(width, height, GRID_STEP as f64),
        };
        let mode = NativeRefineMode::IntraStation { anchor_long_side };
        let window = NativeRefineBase {
            image: base,
            mask: &mask,
            left: 0,
            top: 0,
        };
        let mut pass = NativeRefinePass::new(width, height);
        collect_native_observations(&window, &sampler, width, height, spacing, mode, &mut pass);

        let (match_radius, search_samples) = mode.patch_geometry();
        let columns = (20..width.saturating_sub(20))
            .step_by(NATIVE_REFINE_STEP as usize)
            .collect::<Vec<_>>();
        let rows = (20..height.saturating_sub(20))
            .step_by(NATIVE_REFINE_STEP as usize)
            .collect::<Vec<_>>();
        let mut probes = Vec::with_capacity(columns.len() * rows.len());
        for (row, &y) in rows.iter().enumerate() {
            for (column, &x) in columns.iter().enumerate() {
                // `collect_native_observations` took `&sampler`, so the sampler is
                // in the state it measured with and re-measuring one control point
                // reproduces its own measurement exactly.
                let measurement = measure_native_control_point(
                    &window,
                    &sampler,
                    x,
                    y,
                    spacing,
                    mode,
                    match_radius,
                    search_samples,
                );
                probes.push(ControlPointProbe {
                    column,
                    row,
                    covered: measurement.covered,
                    evaluated: measurement.evaluated,
                    displacement_px: measurement.measured.map(|(d, _, _, _)| d),
                    round_trip_px: measurement.round_trip_px,
                    symmetric_error_px: measurement
                        .measured
                        .map(|(_, _, _, error)| error)
                        .unwrap_or(0.0),
                    best_correlation: measurement.best_correlation,
                    accepted_px: pass.accepted_grid.accepted_at(column, row),
                });
            }
        }
        let observations = pass
            .observations
            .iter()
            .map(|(position, _, _)| {
                let column = columns
                    .iter()
                    .position(|&x| f64::from(x) == position.x)
                    .expect("an observation sits on a grid column");
                let row = rows
                    .iter()
                    .position(|&y| f64::from(y) == position.y)
                    .expect("an observation sits on a grid row");
                (column, row)
            })
            .collect();
        (probes, observations, pass.evaluated)
    }

    /// What a whole native refinement decided about one frame (需求 2.7 / 2.8).
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct RefinementProbe {
        /// Accepted control point cells over *both* spacing passes, which is the
        /// numerator 需求 2.8 defines.
        pub(crate) accepted_control_points: usize,
        /// Accepted control point cells of the final spacing pass alone.
        pub(crate) last_pass_accepted_control_points: usize,
        /// Control point cells whose centre lay on covered base in the final
        /// pass: the denominator of 需求 2.8, in cells.
        pub(crate) valid_cells: usize,
        /// Area of one control point cell, in native pixels of the anchor frame.
        pub(crate) cell_area_px: f64,
        pub(crate) evaluated_control_points: usize,
        pub(crate) local_baseline_samples: usize,
        pub(crate) global_baseline_samples: usize,
        /// Median symmetric reprojection error after the refinement, in native
        /// pixels (需求 2.7).
        pub(crate) local_median_symmetric_error_px: f64,
        /// The same median for the global model (需求 2.7).
        pub(crate) global_median_symmetric_error_px: f64,
        pub(crate) local_field_verdict: intra_station::LocalFieldVerdict,
        pub(crate) reverted_to_global: bool,
        pub(crate) inlier_area_coverage: f64,
        pub(crate) status: IntraStationFrameStatus,
        /// `true` when the displacement field the sampler carries afterwards is
        /// bit for bit the one it entered with, i.e. every control point position
        /// kept the global model's displacement.
        pub(crate) field_unchanged: bool,
    }

    /// Drive the whole native refinement of one frame (both spacing passes).
    pub(crate) fn probe_native_refinement(
        base: &Rgb32FImage,
        layer: &Rgb32FImage,
        anchor_long_side: u32,
    ) -> RefinementProbe {
        let (width, height) = base.dimensions();
        let info = image_info_for_probe(layer);
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        let entry = Field::new(width, height, GRID_STEP as f64);
        let mut sampler = LayerSampler {
            info: &info,
            source: layer,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: entry.clone(),
        };
        let outcome = refine_native_layer(
            base,
            &mask,
            &mut sampler,
            width,
            height,
            NativeRefineMode::IntraStation { anchor_long_side },
        );
        let field_unchanged = sampler.residual.values.len() == entry.values.len()
            && sampler
                .residual
                .values
                .iter()
                .zip(entry.values.iter())
                .all(|(left, right)| {
                    left[0].to_bits() == right[0].to_bits()
                        && left[1].to_bits() == right[1].to_bits()
                });
        RefinementProbe {
            accepted_control_points: outcome.accepted_control_points,
            last_pass_accepted_control_points: outcome.last_pass_accepted_control_points,
            valid_cells: outcome.valid_cells,
            cell_area_px: outcome.cell_area_px,
            evaluated_control_points: outcome.evaluated_control_points,
            local_baseline_samples: outcome.local_baseline_samples,
            global_baseline_samples: outcome.global_baseline_samples,
            local_median_symmetric_error_px: outcome.local_median_symmetric_error_px,
            global_median_symmetric_error_px: outcome.global_median_symmetric_error_px,
            local_field_verdict: outcome.local_field_verdict,
            reverted_to_global: outcome.reverted_to_global,
            inlier_area_coverage: outcome.inlier_area_coverage,
            status: outcome.frame_status(),
            field_unchanged,
        }
    }

    fn image_info_for_probe(image: &Rgb32FImage) -> ImageInfo {
        ImageInfo {
            id: 0,
            filename: "/tmp/stack-pipeline/intra-station-probe.NEF".to_string(),
            width: image.width(),
            height: image.height(),
            alignment_image: image::DynamicImage::ImageRgb32F(image.clone()).to_luma8(),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: Some(75.0),
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_near_black_pixels_in_the_lower_left_capture_border_are_rejected() {
        let image = Rgb32FImage::new(100, 100);
        assert!(is_lower_left_black_capture_border(
            &image,
            10.0,
            90.0,
            Rgb([0.005, 0.006, 0.004]),
        ));
        assert!(!is_lower_left_black_capture_border(
            &image,
            50.0,
            90.0,
            Rgb([0.005, 0.006, 0.004]),
        ));
        assert!(!is_lower_left_black_capture_border(
            &image,
            10.0,
            90.0,
            Rgb([0.08, 0.03, 0.02]),
        ));
        // A pure-black lower-left sample remains non-photographic even at the
        // extreme source edge. Keeping it created the reported black block.
        assert!(is_lower_left_black_capture_border(
            &image,
            10.0,
            98.0,
            Rgb([0.005, 0.006, 0.004]),
        ));
    }

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
            .covered_bounds()
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
    fn covered_bounds_preserve_non_rectangular_source_edges() {
        let mask = GrayImage::from_fn(8, 6, |x, y| {
            Luma([u8::from((x == 1 && y == 3) || (x >= 3 && x <= 6 && y >= 1 && y <= 4)) * 255])
        });
        assert_eq!(mask_covered_bounds(&mask), Some((1, 1, 6, 4)));
    }

    #[test]
    fn streaming_covered_bounds_do_not_reduce_to_largest_rectangle() {
        let store = StreamingMosaicStore::new(12, 8).expect("tile store should initialize");
        let rgb = Rgb32FImage::new(12, 8);
        let mask = GrayImage::from_fn(12, 8, |x, y| {
            Luma([u8::from((x == 1 && y == 6) || (x >= 4 && x <= 10 && y >= 1 && y <= 5)) * 255])
        });
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        assert_eq!(
            store.covered_bounds().expect("coverage should be readable"),
            Some((1, 1, 10, 6))
        );
    }

    #[test]
    fn streaming_decision_alpha_interpolates_between_cells() {
        let decision =
            GrayImage::from_fn(2, 2, |x, y| Luma([if (x + y) % 2 == 0 { 0 } else { 255 }]));
        let centre = sample_decision_alpha(&decision, 0.5, 0.5);
        assert!((centre - 0.5).abs() < 1e-6);
        let left = sample_decision_alpha(&decision, 0.05, 0.5);
        let right = sample_decision_alpha(&decision, 0.95, 0.5);
        assert!(left > 0.45 && left < 0.55);
        assert!(right > 0.45 && right < 0.55);
        assert!(sample_decision_alpha(&decision, 0.0, 0.0) < 1e-6);
        assert!((sample_decision_alpha(&decision, 1.0, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn streaming_materialize_preserves_a_natural_gradient_at_owner_boundary() {
        let store = StreamingMosaicStore::new(256, 32).expect("tile store should initialize");
        let rgb = Rgb32FImage::from_fn(256, 32, |x, _| {
            let value = 0.30 + 0.40 * x as f32 / 255.0;
            Rgb([value, value * 0.98, value * 0.94])
        });
        let mask = GrayImage::from_pixel(256, 32, Luma([255]));
        let owner = GrayImage::from_fn(256, 32, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        store
            .save_owner_tile(0, 0, &owner)
            .expect("owner should be writable");
        let output = materialize_streaming_focus_output(&store, (0, 0, 256, 32))
            .expect("gradient should materialize");
        let left = luma(*output.get_pixel(126, 16));
        let right = luma(*output.get_pixel(129, 16));
        let expected = luma(*rgb.get_pixel(126, 16)) - luma(*rgb.get_pixel(129, 16));
        assert!((left - right).abs() < expected.abs() * 1.4);
        assert!((left - luma(*rgb.get_pixel(126, 16))).abs() < 1e-6);
        assert!((right - luma(*rgb.get_pixel(129, 16))).abs() < 1e-6);
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
        let (gains, scale) = streaming_seam_harmonization(&store, (0, 0, 256, 64), &[])
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
    fn frequency_separation_removes_station_blocks_but_keeps_detail_and_black_bounds() {
        let mut image = Rgb32FImage::from_fn(512, 128, |x, y| {
            if y < 8 {
                return Rgb([0.0; 3]);
            }
            let base = if x < 256 { 0.36 } else { 0.54 };
            let detail = if (x / 3 + y / 3) % 2 == 0 {
                0.035
            } else {
                -0.035
            };
            Rgb([base + detail, base + detail, base + detail])
        });
        let original = image.clone();
        smooth_streaming_low_frequency_illumination(&mut image);
        let mean = |source: &Rgb32FImage, start: u32| {
            (start..start + 12)
                .map(|x| luma(*source.get_pixel(x, 64)))
                .sum::<f64>()
                / 12.0
        };
        let before = (mean(&original, 238) - mean(&original, 262)).abs();
        let after = (mean(&image, 238) - mean(&image, 262)).abs();
        assert!(
            after < before * 0.82,
            "station step {before:.4} -> {after:.4}"
        );
        let before_detail =
            (luma(*original.get_pixel(80, 64)) - luma(*original.get_pixel(83, 64))).abs();
        let after_detail = (luma(*image.get_pixel(80, 64)) - luma(*image.get_pixel(83, 64))).abs();
        assert!(after_detail >= before_detail * 0.88);
        assert!((0..512).all(|x| image.get_pixel(x, 2).0 == [0.0; 3]));
    }

    #[test]
    fn same_coordinate_group_relations_recover_exposure_offset() {
        let (width, height) = (320, 160);
        let base = Rgb32FImage::from_fn(width, height, |x, y| {
            let detail = if (x / 5 + y / 5) % 2 == 0 {
                0.025
            } else {
                -0.025
            };
            Rgb([0.42 + detail, 0.39 + detail, 0.34 + detail])
        });
        let expected = [0.12f64, 0.08, -0.03];
        let candidate = Rgb32FImage::from_fn(width, height, |x, y| {
            let source = base.get_pixel(x, y);
            Rgb(std::array::from_fn(|channel| {
                source[channel] * expected[channel].exp() as f32
            }))
        });
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        let owner = GrayImage::from_pixel(width, height, Luma([1]));
        let relations = streaming_group_tone_relations_from_analysis(
            &base, &mask, &owner, &candidate, &mask, 2,
        );
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].0, (1, 2));
        for (actual, expected) in relations[0].1.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 0.01, "{actual} vs {expected}");
        }
    }

    #[test]
    #[ignore = "applies production low-frequency correction to a supplied real preview"]
    fn real_preview_low_frequency_correction_from_env() {
        let input = std::env::var_os("RAW_EDITOR_MOSAIC_PREVIEW_INPUT")
            .expect("RAW_EDITOR_MOSAIC_PREVIEW_INPUT is required");
        let output = std::env::var_os("RAW_EDITOR_MOSAIC_PREVIEW_OUTPUT")
            .expect("RAW_EDITOR_MOSAIC_PREVIEW_OUTPUT is required");
        let mut image = image::open(input)
            .expect("real preview should open")
            .to_rgb32f();
        smooth_streaming_low_frequency_illumination(&mut image);
        image::DynamicImage::ImageRgb32F(image)
            .to_rgb8()
            .save(output)
            .expect("corrected preview should save");
    }

    #[test]
    fn group_exposure_harmonization_cannot_draw_a_halo_around_dark_strokes() {
        let store = StreamingMosaicStore::new(256, 64).expect("tile store should initialize");
        let rgb = Rgb32FImage::from_fn(256, 64, |x, _| {
            let paper = if x < 128 { 0.46 } else { 0.56 };
            let value = if (58..=62).contains(&x) { 0.06 } else { paper };
            Rgb([value, value * 0.96, value * 0.90])
        });
        let mask = GrayImage::from_pixel(256, 64, Luma([255]));
        let owner = GrayImage::from_fn(256, 64, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        store
            .save_owner_tile(0, 0, &owner)
            .expect("owner should be writable");
        let (gains, _) = streaming_seam_harmonization(&store, (0, 0, 256, 64), &[])
            .expect("harmonization should build")
            .expect("two owners should create a gain field");
        let paper_gain = gains.get_pixel(30, 32)[0];
        for x in [52, 56, 58, 60, 62, 64, 68] {
            assert!(
                (gains.get_pixel(x, 32)[0] - paper_gain).abs() < 1e-5,
                "a dark stroke must not alter the exposure gain around itself"
            );
        }
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
        let overlap_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| {
                *x >= decision.width() / 2 && *x < decision.width() * 3 / 4 && pixel[0] != 0
            })
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
        assert!(
            overlap_selected > 0,
            "a panorama seam must be allowed to leave the rectangular new-coverage boundary"
        );
    }

    #[test]
    fn panorama_transition_marks_ink_cells_as_expensive_seam_routes() {
        let image = Rgb32FImage::from_fn(96, 64, |x, y| {
            let ink = (38..58).contains(&x) && (18..46).contains(&y);
            let value = if ink { 0.05 } else { 0.52 };
            Rgb([value, value * 0.96, value * 0.90])
        });
        let quiet = streaming_transition_structure(|x, y| rgb_at(&image, x, y), 0.0, 0.0, 24.0);
        let ink = streaming_transition_structure(|x, y| rgb_at(&image, x, y), 30.0, 12.0, 36.0);
        assert!(quiet < 0.05, "plain paper should remain a cheap seam route");
        assert!(
            ink > 0.75,
            "a cell crossed by dark calligraphy must strongly repel the panorama seam"
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

    /// Aperiodic paper grain.  A sum of sines would be rejected by
    /// `refine_warped_patch`'s second-peak guard for looking like a repeated
    /// stroke, which is the guard doing its job rather than a refinement failure.
    fn refine_texture(width: u32, height: u32) -> Rgb32FImage {
        let grain = Rgb32FImage::from_fn(width, height, |x, y| {
            let hash = x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
            let hash = (hash ^ (hash >> 15)).wrapping_mul(0x7feb_352d);
            let value = 0.25 + 0.50 * ((hash >> 8) & 0xffff) as f32 / 65_535.0;
            Rgb([value, value * 0.96, value * 0.90])
        });
        image::imageops::blur(&grain, 1.1)
    }

    /// The intra-station gates must leave usable control points.
    ///
    /// Without this the native refinement could reject every control point and
    /// still look healthy from the outside: the frame simply reverts to its
    /// global model, the output is byte identical to a run with no refinement at
    /// all, and nothing fails.  That is exactly what a matching window radius of
    /// 48 inside a 57×57 patch buffer does — `refine_warped_patch` needs a sample
    /// at −20 and returns `None` for every position.
    #[test]
    fn intra_station_native_refinement_accepts_control_points() {
        let base = refine_texture(256, 256);
        // The layer is the same photograph displaced by a few native pixels,
        // which is what focus breathing does inside one Capture_Station.
        // Two native pixels: inside 需求 2.6's `0.01 × 256` error ceiling for this
        // fixture, so the gate under test is the matching, not the ceiling.
        let shifted = Rgb32FImage::from_fn(256, 256, |x, y| *base.get_pixel((x + 2).min(255), y));
        let info = image_info(1, &shifted);
        let mask = GrayImage::from_pixel(256, 256, Luma([255]));
        let mut sampler = LayerSampler {
            info: &info,
            source: &shifted,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(256, 256, GRID_STEP as f64),
        };
        let outcome = refine_native_layer(
            &base,
            &mask,
            &mut sampler,
            256,
            256,
            NativeRefineMode::IntraStation {
                anchor_long_side: 256,
            },
        );
        assert!(
            !outcome.skipped,
            "需求 2.2 has no analysis-scale early exit"
        );
        assert!(
            outcome.evaluated_control_points >= 64,
            "the control point grid must reach the layer ({} evaluated)",
            outcome.evaluated_control_points
        );
        assert!(
            outcome.accepted_control_points >= 16,
            "the 需求 2.4/2.5/2.6 gates must accept real matches, not reject every one ({}/{} accepted)",
            outcome.accepted_control_points,
            outcome.evaluated_control_points
        );
        assert!(
            outcome.inlier_area_coverage > 0.0,
            "需求 2.8 coverage must be measurable once control points are accepted"
        );
        // 需求 2.7 keeps a local field that lowered the frame's median symmetric
        // reprojection error.  The baseline is the error of the global model, so a
        // frame the refinement actually fixed must not be reverted.
        assert!(
            !outcome.reverted_to_global,
            "a refinement that removed a real 2px displacement must be kept (global {:.2}px, local {:.2}px)",
            outcome.global_median_symmetric_error_px, outcome.local_median_symmetric_error_px
        );
        // The comparison has to be made against a baseline that exists.  The
        // regression this guards reverted every frame because the coarse pass
        // accepted too few control points, `median_of(&[])` returned `0.0`, and
        // no refinement can be below zero.
        assert!(
            outcome.global_baseline_samples >= intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES,
            "需求 2.7 needs a real global baseline, not an empty median ({} samples)",
            outcome.global_baseline_samples
        );
        assert_eq!(
            outcome.local_field_verdict,
            intra_station::LocalFieldVerdict::Kept
        );
        assert_eq!(outcome.frame_status(), IntraStationFrameStatus::Local);
    }

    /// 需求 2.2 caps the *actual* intra-station analysis resolution at 2048
    /// while the production compositors currently configure 1400. This also
    /// guards Stack_Report's source value: both paths share the registrar's
    /// actual configuration instead of reporting the ceiling.
    #[test]
    fn the_analysis_budgets_stay_inside_the_intra_station_bound() {
        const {
            assert!(
                intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE
                    <= intra_station::INTRA_STATION_ANALYSIS_MAX_LONG_SIDE
            );
            assert!(ANALYSIS_LONG_SIDE == intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE);
            assert!(
                STREAMING_ANALYSIS_LONG_SIDE == intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE
            );
            assert!(ANALYSIS_LONG_SIDE <= intra_station::INTRA_STATION_ANALYSIS_MAX_LONG_SIDE);
            assert!(
                STREAMING_ANALYSIS_LONG_SIDE <= intra_station::INTRA_STATION_ANALYSIS_MAX_LONG_SIDE
            );
        };
    }

    /// The tiled compositor is the *only* path a multi-station stack ever takes,
    /// so 需求 2.3–2.9 has to be measurable there.  Before this the native
    /// refinement existed solely on the in-memory path, and every real focus
    /// stack — three camera positions, an 8369×10114 canvas — skipped it
    /// silently and produced output byte identical to a run without it.
    #[test]
    fn streaming_intra_station_refinement_measures_across_tiles() {
        let width = 1_200u32;
        let height = 256u32;
        let base = refine_texture(width, height);
        let store = StreamingMosaicStore::new(width, height).expect("streaming store");
        assert!(
            store.tile_columns > 1,
            "the fixture must straddle a tile boundary"
        );
        for tile_row in 0..store.tile_rows {
            for tile_column in 0..store.tile_columns {
                let (left, top, tile_width, tile_height) = store.tile_extent(tile_column, tile_row);
                let tile = Rgb32FImage::from_fn(tile_width, tile_height, |x, y| {
                    *base.get_pixel(left + x, top + y)
                });
                let covered = GrayImage::from_pixel(tile_width, tile_height, Luma([255]));
                let owner = GrayImage::from_pixel(tile_width, tile_height, Luma([1]));
                store
                    .save_tile(tile_column, tile_row, &tile, &covered)
                    .expect("tile");
                store
                    .save_owner_tile(tile_column, tile_row, &owner)
                    .expect("owner tile");
            }
        }
        let shifted = Rgb32FImage::from_fn(width, height, |x, y| {
            *base.get_pixel((x + 2).min(width - 1), y)
        });
        let info = image_info(1, &shifted);
        fn layer_sampler<'a>(
            source: &'a Rgb32FImage,
            info: &'a ImageInfo,
            width: u32,
            height: u32,
        ) -> LayerSampler<'a> {
            LayerSampler {
                info,
                source,
                source_divisor: 1.0,
                inverse: Matrix3::identity(),
                projection: Projection::Planar,
                offset: (0.0, 0.0),
                left: 0,
                top: 0,
                scale: 1.0,
                residual: Field::new(width, height, GRID_STEP as f64),
            }
        }
        let mode = NativeRefineMode::IntraStation {
            anchor_long_side: 256,
        };
        let mut sampler = layer_sampler(&shifted, &info, width, height);
        let outcome = refine_native_layer_streaming(
            &store,
            Some(1),
            &mut sampler,
            width,
            height,
            (0, width - 1, 0, height - 1),
            mode,
        )
        .expect("streaming native refinement");
        assert!(
            outcome.accepted_control_points >= 16,
            "the tiled path must measure real control points ({}/{} accepted)",
            outcome.accepted_control_points,
            outcome.evaluated_control_points
        );
        assert!(!outcome.reverted_to_global);
        assert!(outcome.inlier_area_coverage > 0.20);
        // 需求 2.2: a later focal plane registers against its own camera
        // position, so canvas owned by another station is not a base at all.
        let mut foreign = layer_sampler(&shifted, &info, width, height);
        let ignored = refine_native_layer_streaming(
            &store,
            Some(7),
            &mut foreign,
            width,
            height,
            (0, width - 1, 0, height - 1),
            mode,
        )
        .expect("streaming native refinement");
        assert_eq!(ignored.accepted_control_points, 0);
        assert_eq!(ignored.evaluated_control_points, 0);
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
        // Four times the old 160x120 fixture, with the candidate a 320x320 crop
        // instead of an 80x80 one.  On the small canvas the native control
        // points had to sit at least one 57x57 patch away from the layer edge,
        // which left 2 measurable cells out of 18 — an 11% inlier spatial
        // support that is an artefact of an 80px layer, not of the photographs,
        // and that 需求 2.8 / 12.2 correctly refuse to fuse.  A 320px layer
        // measures 0.4 of its cells, so the test exercises the selection it is
        // about; the exclusion path has its own unit tests in
        // `stack_pipeline::station_degradation`.
        let sharp = Rgb32FImage::from_fn(640, 480, |x, y| {
            let value = 0.36
                + 0.12 * (x as f32 * 0.75).sin()
                + 0.10 * (y as f32 * 0.65).cos()
                + 0.03 * ((x + y) as f32 * 0.19).sin();
            Rgb([value * 1.1, value, value * 0.8])
        });
        let base = image::imageops::blur(&sharp, 2.0);
        let candidate = image::imageops::crop_imm(&sharp, 160, 80, 320, 320).to_image();
        let sources = [base.clone(), candidate];
        let infos = [image_info(0, &sources[0]), image_info(1, &sources[1])];
        let transforms = HashMap::from([
            (0, Matrix3::identity()),
            (
                1,
                Matrix3::new(1.0, 0.0, 160.0, 0.0, 1.0, 80.0, 0.0, 0.0, 1.0),
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
        for y in 120..360 {
            for x in 200..440 {
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
