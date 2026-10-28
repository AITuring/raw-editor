//! Analysis-scale focus layers and cell measurements.
//!
//! Renders analysis layers, commits their scores and measures per-cell focus,
//! disagreement and texture for the ownership grid.

use super::super::stack_pipeline::focus_fuser::CellSamplingPlan;
use super::super::stack_pipeline::focus_fuser::OwnershipGridGeometry;
use super::super::stack_pipeline::focus_fuser::cell_disagreement_with_seam_evidence;
use image::GrayImage;
use image::Rgb32FImage;
use rayon::prelude::*;

use super::{
    RenderedFocusAnalysisLayer, RenderedFocusLayer, resize_binary_mask, resize_rgb,
    sample_focus_plane, scaled_region_end, scaled_region_start,
};

pub(super) fn render_focus_analysis_layer(
    layer: &RenderedFocusLayer,
    out_width: u32,
    out_height: u32,
    analysis_width: u32,
    analysis_height: u32,
) -> RenderedFocusAnalysisLayer {
    let mut analysis = Rgb32FImage::new(analysis_width, analysis_height);
    let mut analysis_mask = GrayImage::new(analysis_width, analysis_height);
    let mut analysis_foreground_mask = GrayImage::new(analysis_width, analysis_height);
    let (layer_width, layer_height) = layer.image.dimensions();
    if layer_width == 0 || layer_height == 0 || out_width == 0 || out_height == 0 {
        return RenderedFocusAnalysisLayer {
            image: analysis,
            mask: analysis_mask,
            foreground_mask: analysis_foreground_mask,
            relaxed_foreground_mask: GrayImage::new(analysis_width, analysis_height),
        };
    }

    let left = scaled_region_start(layer.left, out_width, analysis_width).min(analysis_width);
    let top = scaled_region_start(layer.top, out_height, analysis_height).min(analysis_height);
    let right = scaled_region_end(
        layer.left.saturating_add(layer_width),
        out_width,
        analysis_width,
    )
    .min(analysis_width);
    let bottom = scaled_region_end(
        layer.top.saturating_add(layer_height),
        out_height,
        analysis_height,
    )
    .min(analysis_height);
    if left >= right || top >= bottom {
        return RenderedFocusAnalysisLayer {
            image: analysis,
            mask: analysis_mask,
            foreground_mask: analysis_foreground_mask,
            relaxed_foreground_mask: GrayImage::new(analysis_width, analysis_height),
        };
    }

    let resized_width = right - left;
    let resized_height = bottom - top;
    let resized = resize_rgb(&layer.image, resized_width, resized_height);
    let resized_mask = resize_binary_mask(&layer.mask, resized_width, resized_height);
    let resized_foreground_mask =
        resize_binary_mask(&layer.foreground_mask, resized_width, resized_height);
    let resized_relaxed_foreground_mask = resize_binary_mask(
        &layer.relaxed_foreground_mask,
        resized_width,
        resized_height,
    );
    let analysis_stride = analysis_width as usize * 3;
    let resized_stride = resized_width as usize * 3;
    analysis
        .as_mut()
        .par_chunks_mut(analysis_stride)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(resized.as_raw().par_chunks(resized_stride))
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize * 3;
            let end = start + resized_stride;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    analysis_mask
        .as_mut()
        .par_chunks_mut(analysis_width as usize)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(resized_mask.as_raw().par_chunks(resized_width as usize))
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    analysis_foreground_mask
        .as_mut()
        .par_chunks_mut(analysis_width as usize)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(
            resized_foreground_mask
                .as_raw()
                .par_chunks(resized_width as usize),
        )
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    let analysis_relaxed_stride = analysis_width as usize;
    let mut relaxed_foreground_mask = GrayImage::new(analysis_width, analysis_height);
    relaxed_foreground_mask
        .as_mut()
        .par_chunks_mut(analysis_relaxed_stride)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(
            resized_relaxed_foreground_mask
                .as_raw()
                .par_chunks(resized_width as usize),
        )
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    RenderedFocusAnalysisLayer {
        image: analysis,
        mask: analysis_mask,
        foreground_mask: analysis_foreground_mask,
        relaxed_foreground_mask,
    }
}

/// Upsample an analysis-resolution focus decision onto a layer's pixel grid.
///
/// Superseded on the station path by [`focus_cell_decision_for_layer`], which
/// decides at ownership cell resolution instead (需求 3.6).  Kept because it is
/// the only reader of the analysis decision and the shifted-mosaic guards still
/// maintain that decision; the comparison path needs it back when it is
/// re-enabled.
#[allow(dead_code)]
pub(super) fn focus_decision_for_layer(
    analysis_decision: &GrayImage,
    layer: &RenderedFocusLayer,
    out_width: u32,
    out_height: u32,
) -> GrayImage {
    let (width, height) = layer.image.dimensions();
    let analysis_width = analysis_decision.width();
    let analysis_height = analysis_decision.height();
    GrayImage::from_fn(width, height, |x, y| {
        if out_width == 0 || out_height == 0 || analysis_width == 0 || analysis_height == 0 {
            return image::Luma([0]);
        }
        let global_x = layer.left.saturating_add(x);
        let global_y = layer.top.saturating_add(y);
        let analysis_x = ((global_x as u64 * analysis_width as u64) / out_width as u64)
            .min(analysis_width.saturating_sub(1) as u64) as u32;
        let analysis_y = ((global_y as u64 * analysis_height as u64) / out_height as u64)
            .min(analysis_height.saturating_sub(1) as u64) as u32;
        *analysis_decision.get_pixel(analysis_x, analysis_y)
    })
}

pub(super) fn commit_focus_analysis_scores(
    best_focus: &mut [f32],
    candidate_focus: &[f32],
    candidate_analysis_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    full_resolution_decision: &GrayImage,
    out_width: u32,
    out_height: u32,
    analysis_width: u32,
    analysis_height: u32,
) {
    if out_width == 0
        || out_height == 0
        || analysis_width == 0
        || analysis_height == 0
        || candidate_analysis_mask.dimensions() != (analysis_width, analysis_height)
        || candidate_focus.len() != analysis_width as usize * analysis_height as usize
        || best_focus.len() != candidate_focus.len()
        || full_resolution_decision.dimensions() != candidate.image.dimensions()
    {
        return;
    }

    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0
        || layer_height == 0
        || candidate.left >= out_width
        || candidate.top >= out_height
        || candidate.left.saturating_add(layer_width) > out_width
        || candidate.top.saturating_add(layer_height) > out_height
    {
        return;
    }

    // The full-resolution ownership guards can veto part of an analysis cell.
    // Downsample the committed decision, rather than blindly copying the
    // low-resolution decision, so the score table remains tied to pixels that
    // really entered the merged image.
    let left = scaled_region_start(candidate.left, out_width, analysis_width).min(analysis_width);
    let top = scaled_region_start(candidate.top, out_height, analysis_height).min(analysis_height);
    let right = scaled_region_end(
        candidate.left.saturating_add(layer_width),
        out_width,
        analysis_width,
    )
    .min(analysis_width);
    let bottom = scaled_region_end(
        candidate.top.saturating_add(layer_height),
        out_height,
        analysis_height,
    )
    .min(analysis_height);
    if left >= right || top >= bottom {
        return;
    }
    let committed_width = right - left;
    let committed_height = bottom - top;
    let committed = resize_binary_mask(full_resolution_decision, committed_width, committed_height);
    let candidate_mask = candidate_analysis_mask.as_raw();
    for local_y in 0..committed_height as usize {
        for local_x in 0..committed_width as usize {
            let analysis_x = left as usize + local_x;
            let analysis_y = top as usize + local_y;
            let analysis_index = analysis_y * analysis_width as usize + analysis_x;
            if committed.as_raw()[local_y * committed_width as usize + local_x] > 0
                && candidate_mask[analysis_index] > 0
            {
                best_focus[analysis_index] = candidate_focus[analysis_index];
            }
        }
    }
}

/// Normalised Sharpness_Score and coverage of every ownership cell of one
/// projected layer (需求 3.2, 3.7).
///
/// A cell counts as covered when the layer's validity mask is set at the cell
/// centre.  The cell is the *decision* unit; a cell only partly inside the
/// projected trapezoid still gets one owner, and the per-pixel write below
/// never copies a pixel the layer does not actually have, so the Coverage_Mask
/// stays exact at pixel resolution.
pub(super) fn measure_focus_cells(
    geometry: &OwnershipGridGeometry,
    plan: &CellSamplingPlan,
    image: &Rgb32FImage,
    mask: &GrayImage,
    left: u32,
    top: u32,
) -> (Vec<f64>, Vec<bool>) {
    // 需求 3.2: the candidate and the current composite are comparable only when
    // they are measured at the same positions with the same window, and the only
    // way to guarantee that across two call sites is to insist that both measure
    // with the plan this station's ownership grid defines.  A plan from anywhere
    // else would make defocus indistinguishable from a change of sampling
    // geometry, which is the one confusion 需求 3.2 exists to rule out.
    plan.assert_matches(&geometry.sampling_plan());
    let count = geometry.cell_count();
    let mut sharpness = vec![0.0f64; count];
    let mut covered = vec![false; count];
    let columns = geometry.columns as usize;
    let half = f64::from(geometry.cell_size_px) * 0.5;
    sharpness
        .par_iter_mut()
        .zip(covered.par_iter_mut())
        .enumerate()
        .for_each(|(cell, (sharpness, covered))| {
            let (column, row) = ((cell % columns) as u32, (cell / columns) as u32);
            let (origin_x, origin_y) = geometry.cell_origin(column, row);
            let sample = |x: f64, y: f64| sample_focus_plane(image, mask, left, top, x, y);
            if sample(origin_x + half, origin_y + half).is_none() {
                return;
            }
            *covered = true;
            *sharpness = plan.measure(origin_x, origin_y, sample);
        });
    (sharpness, covered)
}

/// Pixel inconsistency of every ownership cell between a candidate layer and
/// the current composite (需求 3.3).
pub(super) fn measure_focus_cell_disagreement(
    geometry: &OwnershipGridGeometry,
    candidate: &RenderedFocusLayer,
    base: &Rgb32FImage,
    base_mask: &GrayImage,
) -> (Vec<f64>, Vec<f64>) {
    let count = geometry.cell_count();
    let columns = geometry.columns as usize;
    let cell_size = f64::from(geometry.cell_size_px);
    let mut disagreement = vec![(0.0f64, 0.0f64); count];
    disagreement
        .par_iter_mut()
        .enumerate()
        .for_each(|(cell, disagreement)| {
            let (column, row) = ((cell % columns) as u32, (cell / columns) as u32);
            let (origin_x, origin_y) = geometry.cell_origin(column, row);
            // Both closures receive the same fractional cell offsets and map
            // them into their own plane, which is how 需求 3.3's "same cell"
            // comparison stays a property of the code.
            *disagreement = cell_disagreement_with_seam_evidence(
                |fx, fy| {
                    sample_focus_plane(
                        &candidate.image,
                        &candidate.mask,
                        candidate.left,
                        candidate.top,
                        origin_x + fx * cell_size,
                        origin_y + fy * cell_size,
                    )
                },
                |fx, fy| {
                    sample_focus_plane(
                        base,
                        base_mask,
                        0,
                        0,
                        origin_x + fx * cell_size,
                        origin_y + fy * cell_size,
                    )
                },
            );
        });
    disagreement.into_iter().unzip()
}

/// Rasterise an ownership cell decision onto a layer's own pixel grid
/// (需求 3.6).
///
/// The result is a hard mask whose boundaries are exactly ownership cell
/// boundaries: no feather, no alpha ramp, no transition band.  The legacy
/// streaming mosaic keeps its `STREAMING_OWNERSHIP_FEATHER` sub-cell blur; the
/// station layer no longer has one.
pub(super) fn focus_cell_decision_for_layer(
    geometry: &OwnershipGridGeometry,
    took: &[bool],
    layer: &RenderedFocusLayer,
) -> GrayImage {
    let (width, height) = layer.image.dimensions();
    let mut decision = GrayImage::new(width, height);
    let cell_size = geometry.cell_size_px.max(1);
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    let width_usize = width as usize;
    decision
        .as_mut()
        .par_chunks_mut(width_usize)
        .enumerate()
        .for_each(|(y, row)| {
            let global_y = layer.top as usize + y;
            let cell_row = global_y / cell_size as usize;
            if cell_row >= rows {
                return;
            }
            let base = cell_row * columns;
            for (x, value) in row.iter_mut().enumerate() {
                let cell_column = (layer.left as usize + x) / cell_size as usize;
                if cell_column < columns && took[base + cell_column] {
                    *value = 255;
                }
            }
        });
    decision
}

/// Expand an ownership cell plane to one value per tile pixel (需求 4.6).
pub(super) fn focus_cell_plane_to_pixels(
    geometry: &OwnershipGridGeometry,
    cells: &[f32],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let mut values = vec![0.0f32; width as usize * height as usize];
    let cell_size = geometry.cell_size_px.max(1) as usize;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    values
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let cell_row = y / cell_size;
            if cell_row >= rows {
                return;
            }
            let base = cell_row * columns;
            for (x, value) in row.iter_mut().enumerate() {
                let cell_column = x / cell_size;
                if cell_column < columns {
                    *value = cells[base + cell_column];
                }
            }
        });
    values
}

/// Expand the cell-level Textured_Pixel evidence onto the station plane.
pub(super) fn focus_cell_texture_to_pixels(
    geometry: &OwnershipGridGeometry,
    cells: &[bool],
    width: u32,
    height: u32,
) -> Vec<u8> {
    let mut values = vec![0u8; width as usize * height as usize];
    let cell_size = geometry.cell_size_px.max(1) as usize;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    values
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let cell_row = y / cell_size;
            if cell_row >= rows {
                return;
            }
            let base = cell_row * columns;
            for (x, value) in row.iter_mut().enumerate() {
                let cell_column = x / cell_size;
                if cell_column < columns && cells[base + cell_column] {
                    *value = 255;
                }
            }
        });
    values
}
