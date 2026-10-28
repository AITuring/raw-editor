//! Virtual_Tile records and their station geometry.
//!
//! A Virtual_Tile is one fused Capture_Station rendered in its anchor plane with
//! its Coverage_Mask, ownership and provenance; the helpers build tiles from
//! renders and report the station-to-world geometry the compositors consume.

use super::super::coordinates::Projection;
use super::super::coordinates::pixel_aligned_canvas;
use super::super::render_geometry::focus_output_bounds;
use super::super::stack_pipeline::focus_fuser;
use crate::panorama_utils::shared::FocusLayerWarp;
use crate::panorama_utils::shared::ImageInfo;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use std::collections::HashMap;
use tauri::AppHandle;
use tauri::Runtime;

use super::{
    ColorEncoding, ConfidenceMap, CoverageMask, FocusStackTileRender, NO_OWNER, OwnershipMap,
    SourceProvenance, focus_stack_stitcher_unfilled,
};

/// A fully focus-fused capture station in the coordinate system of the
/// complete mosaic.  The tile pixels are owned by the station's focus
/// decision; `tile_to_world` only places that result in the global planar
/// coordinate system.  Keeping the ownership, coverage, confidence and
/// provenance beside the pixels lets the caller perform tile-level
/// registration and seam/ownership, and lets every output pixel be traced
/// back to one real Source_RAW, without falling back to the individual focus
/// layers.
#[derive(Debug, Clone)]
pub(crate) struct VirtualTile {
    pub(crate) station_index: usize,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tile_to_world: Matrix3<f64>,
    pub(crate) pixels: Rgb32FImage,
    pub(crate) ownership: OwnershipMap,
    pub(crate) sharpness_confidence: ConfidenceMap,
    pub(crate) coverage: CoverageMask,
    pub(crate) color_encoding: ColorEncoding,
    pub(crate) provenance: Vec<SourceProvenance>,
}

impl VirtualTile {
    /// Build a Virtual_Tile, enforcing the four structural invariants of the
    /// design document:
    ///
    /// 1. `provenance` has one entry per participating Source_RAW and matches
    ///    the Ownership_Map legend (需求 4.1);
    /// 2. every `provenance[i].owned_pixels` equals the number of pixels owned
    ///    by that source, so the sum equals the number of pixels with an owner
    ///    (需求 4.1);
    /// 3. Coverage_Mask and Ownership_Map agree in both directions and share
    ///    the pixel geometry of `pixels` (需求 3.11 / 4.6);
    /// 4. the valid pixel extent equals the union bounds of the covered pixels
    ///    (需求 4.7).
    pub(crate) fn new(
        station_index: usize,
        tile_to_world: Matrix3<f64>,
        pixels: Rgb32FImage,
        ownership: OwnershipMap,
        sharpness_confidence: ConfidenceMap,
        coverage: CoverageMask,
        color_encoding: ColorEncoding,
        provenance: Vec<SourceProvenance>,
    ) -> Result<Self, String> {
        let (width, height) = pixels.dimensions();
        // Invariant 3a: one pixel coordinate system for pixels and masks.
        if ownership.dimensions() != (width, height) {
            return Err(format!(
                "ownership map is {:?} for a {}x{} tile",
                ownership.dimensions(),
                width,
                height
            ));
        }
        if coverage.dimensions() != (width, height) {
            return Err(format!(
                "coverage mask is {:?} for a {}x{} tile",
                coverage.dimensions(),
                width,
                height
            ));
        }
        if sharpness_confidence.dimensions() != (width, height) {
            return Err(format!(
                "confidence map is {:?} for a {}x{} tile",
                sharpness_confidence.dimensions(),
                width,
                height
            ));
        }
        if !sharpness_confidence.in_unit_range() {
            return Err("confidence values must stay inside [0, 1]".to_string());
        }
        // Invariant 1: provenance is the Ownership_Map legend.
        if provenance.len() != ownership.legend().len() {
            return Err(format!(
                "{} provenance record(s) for an ownership legend of {} source(s)",
                provenance.len(),
                ownership.legend().len()
            ));
        }
        if let Some((index, record)) = provenance
            .iter()
            .enumerate()
            .find(|(index, record)| ownership.legend()[*index] != record.absolute_path)
        {
            return Err(format!(
                "provenance {} ({}) does not match its ownership legend entry ({})",
                index,
                record.absolute_path.display(),
                ownership.legend()[index].display()
            ));
        }
        // Invariant 2: the provenance pixel counts are the Ownership_Map.
        let counts = ownership.owned_pixel_counts();
        if let Some((index, record)) = provenance
            .iter()
            .enumerate()
            .find(|(index, record)| counts[*index] != record.owned_pixels)
        {
            return Err(format!(
                "provenance {} ({}) claims {} owned pixel(s) but owns {}",
                index,
                record.absolute_path.display(),
                record.owned_pixels,
                counts[index]
            ));
        }
        // Invariant 3b: covered <=> owned, in both directions.
        if let Some(index) = coverage
            .covered()
            .iter()
            .zip(ownership.owners())
            .position(|(&covered, &owner)| (covered > 0) != (owner != NO_OWNER))
        {
            let x = index % width.max(1) as usize;
            let y = index / width.max(1) as usize;
            return Err(format!(
                "pixel ({x}, {y}) is covered={} but owned={}",
                coverage.covered()[index] > 0,
                ownership.owners()[index] != NO_OWNER
            ));
        }
        // Invariant 4: the valid extent is exactly the covered union bounds.
        // Everything outside those bounds must stay untouched, so the tile
        // never carries a pixel that no source produced.
        match coverage.covered_bounds() {
            Some((min_x, min_y, bounds_width, bounds_height)) => {
                let tight =
                    min_x == 0 && min_y == 0 && bounds_width == width && bounds_height == height;
                if !tight {
                    let outside_is_clear = pixels.enumerate_pixels().all(|(x, y, pixel)| {
                        let inside = x >= min_x
                            && y >= min_y
                            && x < min_x + bounds_width
                            && y < min_y + bounds_height;
                        inside || pixel.0 == [0.0, 0.0, 0.0]
                    });
                    if !outside_is_clear {
                        return Err(format!(
                            "tile pixels extend beyond the coverage union bounds ({min_x}, {min_y}, {bounds_width}, {bounds_height})"
                        ));
                    }
                }
            }
            None => {
                if pixels.as_raw().iter().any(|value| *value != 0.0) {
                    return Err("tile has pixels but no coverage".to_string());
                }
            }
        }
        Ok(Self {
            station_index,
            width,
            height,
            tile_to_world,
            pixels,
            ownership,
            sharpness_confidence,
            coverage,
            color_encoding,
            provenance,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FocusVirtualTileGeometry {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tile_to_world: Matrix3<f64>,
}

/// Return the projected bounds needed to place one virtual tile without
/// decoding or fusing any source pixels.  This is used by the bounded-memory
/// compositor to create tile metadata first and render each group lazily.
pub(crate) fn focus_stack_virtual_tile_geometry(
    group: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
) -> Option<FocusVirtualTileGeometry> {
    let native_warp = focus_warp.map(native_station_geometry);
    let focus_warp = native_warp.as_ref().filter(|warp| !warp.bands.is_empty());
    if group.is_empty() {
        return None;
    }
    if group
        .iter()
        .any(|image| !global_homographies.contains_key(&image.id))
    {
        return None;
    }
    let (min_x, max_x, min_y, max_y) =
        focus_output_bounds(group, global_homographies, projection, focus_warp);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return None;
    }
    let (offset_x, width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, height) = pixel_aligned_canvas(min_y, max_y);
    Some(FocusVirtualTileGeometry {
        width,
        height,
        tile_to_world: Matrix3::new(1.0, 0.0, -offset_x, 0.0, 1.0, -offset_y, 0.0, 0.0, 1.0),
    })
}

/// Fuse each capture group independently into one in-memory Virtual_Tile.
///
/// Group membership is supplied by the evidence-backed capture grouping in
/// `panorama_stitching`; this helper does not inspect filenames or assume a
/// particular number of layers.  Every group's existing global homographies
/// are retained, so the returned translation maps tile pixels back into the
/// same world coordinates used by the normal pose graph.
pub(crate) fn focus_stack_virtual_tiles<R: Runtime, F>(
    groups: &[&[&ImageInfo]],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Vec<VirtualTile>, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let mut tiles = Vec::with_capacity(groups.len());
    for (group_index, group) in groups.iter().enumerate() {
        if group.is_empty() {
            continue;
        }
        let geometry =
            focus_stack_virtual_tile_geometry(group, global_homographies, projection, focus_warp)
                .ok_or_else(|| {
                format!(
                    "Capture group {} has invalid focus-tile bounds",
                    group_index + 1
                )
            })?;

        // The regular focus renderer computes its own bounds and uses exactly
        // these pixel-aligned offsets.  Render it once, then expose the
        // inverse translation as the tile-to-world placement.  The unfilled
        // form is the Virtual_Tile form: it keeps the projected validity mask
        // instead of filling the trapezoid margins, which is exactly the
        // Coverage_Mask semantics of 需求 3.7.
        let rendered = focus_stack_stitcher_unfilled(
            group,
            global_homographies,
            projection,
            focus_warp,
            None,
            group_index,
            false,
            app_handle.clone(),
            progress_event,
            load_image,
        )?;
        tiles.push(virtual_tile_from_render(
            group_index,
            geometry.tile_to_world,
            rendered,
        )?);
    }
    Ok(tiles)
}

/// Assemble a Virtual_Tile from a station render.  Every invariant is checked
/// by [`VirtualTile::new`]; this only turns the renderer's internal masks into
/// the published structure (需求 4.1 / 4.2 / 4.6 / 4.7).
pub(crate) fn virtual_tile_from_render(
    station_index: usize,
    tile_to_world: Matrix3<f64>,
    rendered: FocusStackTileRender,
) -> Result<VirtualTile, String> {
    let FocusStackTileRender {
        image,
        masks,
        sampling_origin: _,
    } = rendered;
    let masks = masks.ok_or_else(|| {
        format!(
            "Capture station {} produced no ownership/coverage masks",
            station_index + 1
        )
    })?;
    let mut fusion = masks.fusion;
    fusion.station_index = station_index;
    focus_fuser::record_run_station(fusion);
    let owned_pixels = masks.ownership.owned_pixel_counts();
    let provenance = masks
        .ownership
        .legend()
        .iter()
        .zip(owned_pixels)
        .map(|(path, owned_pixels)| SourceProvenance {
            absolute_path: path.clone(),
            // TODO(task 5.4): the read-only source access records the digest.
            sha256: None,
            owned_pixels,
        })
        .collect::<Vec<_>>();
    VirtualTile::new(
        station_index,
        tile_to_world,
        image,
        masks.ownership,
        masks.confidence,
        masks.coverage,
        // The focus renderer copies decoded render pixels, which are the
        // display-encoded sRGB values used by the rest of the pipeline.
        ColorEncoding::DisplaySrgb,
        provenance,
    )
}

pub(super) fn native_station_geometry(warp: &FocusLayerWarp) -> FocusLayerWarp {
    FocusLayerWarp {
        bands: warp
            .bands
            .iter()
            .filter(|band| band.physical_edge)
            .cloned()
            .collect(),
    }
}
