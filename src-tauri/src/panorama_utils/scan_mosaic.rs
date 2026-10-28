//! Planar scan mosaic for flattened source tiles (Photoshop PSD composites).
//!
//! One PSD is one already-fused capture station of a flat artwork. Its
//! neighbours overlap it heavily but differ in focus, magnification and
//! illumination, and the feature graph only knows their poses from 2400px
//! proxies, so the global poses still disagree by several native pixels.
//! Averaging two such tiles across an overlap doubles or softens brush and
//! silk detail. This compositor instead
//!
//! 1. refines the planar homographies with native-resolution patch
//!    correspondences and one joint robust least-squares adjustment, then
//!    fits every tile a smooth correction on a control grid for what a
//!    planar pose cannot follow (lens distortion, a scroll that is not
//!    perfectly flat), validated on held-out correspondences;
//! 2. renders on an output grid on which the highest-resolution tile is not
//!    downsampled (bounded by the in-memory pixel budget);
//! 3. chooses one owner per output cell by the detail each tile delivers in
//!    *output* pixels, with seams priced by local structure (the multi-label
//!    graph cut shared with the focus fuser);
//! 4. writes the owner's high-frequency detail with a few-pixel seam
//!    transition and blends only the low-frequency band across a wide
//!    transition, so tone steps fade while detail is never averaged.
//!
//! Sources are decoded twice (analysis, then rendering) so only one
//! full-resolution tile is resident at a time.
use super::coordinates::pixel_aligned_canvas;
use super::photometric::{PhotometricModel, PhotometricOptions, calibrate_overlap_photometry};
use super::registration::{PatchMatchGates, probe_warped_patch};
use super::shared::ImageInfo;
use super::stack_pipeline::focus_fuser::{NO_OWNER, OwnershipCostGrid};
use super::stitching::{apply_photometric_model_to_image, flat_scan_paper_median};
use image::{DynamicImage, GenericImageView, GrayImage, Rgb32FImage};
use nalgebra::{DMatrix, DVector, Matrix3, Point2, Vector3};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use tauri::{AppHandle, Emitter, Runtime};

/// Side of one ownership cell in output pixels.
const OWNER_CELL: u32 = 32;
/// A cell may be owned without penalty only when this much valid source
/// surrounds it, so resampling and the low band never read past a tile edge.
const OWNER_MARGIN: f64 = 48.0;
/// Data cost added when a tile owns a cell too close to its own border.
const EDGE_OWNER_PENALTY: f64 = 4.0;
/// Data cost of a tile that delivers no detail where the best tile delivers
/// all of it. The graph cut compares it with the Potts seam weight (0.35+).
const SHARPNESS_COST: f64 = 4.0;
/// Upper bound of the per-cell structure term that prices a seam. Kept low:
/// with the smooth registration correction a seam through a stroke stays
/// aligned, so a softer tile must not win a stroke merely to avoid a seam.
const STRUCTURE_SEAM_WEIGHT: f64 = 0.05;
/// Side of one sharpness measurement block in source pixels.
const SHARPNESS_BLOCK: usize = 32;
/// Decimation of the low-frequency band grid in output pixels.
const LOW_GRID: u32 = 8;
/// Low-pass of every tile's low band on the low grid (grid pixels). Wide
/// enough that a residual registration offset of a pixel or two cannot move
/// the band, narrow enough to carry glare and illumination fields.
const LOW_BAND_SIGMA: f32 = 4.0;
/// Final smoothing of the consensus low band (grid pixels).
const CONSENSUS_SIGMA: f32 = 1.5;
/// Feather distance of a tile's consensus weight from its border (grid px).
const CONSENSUS_FEATHER: f32 = 32.0;
/// A tile's low band joins the consensus only within this relative distance
/// of the per-channel median, so one tile's glass reflection is outvoted.
const CONSENSUS_TOLERANCE: f32 = 0.04;
/// Data cost of a tile whose low band departs from the consensus by more
/// than [`TONE_DEVIATION_SPAN`] beyond [`TONE_DEVIATION_FLOOR`]. The owner's
/// tone never reaches the output (the consensus carries it), so only a
/// departure well beyond the few percent photometric residual every tile
/// shows (a reflection) is priced.
const TONE_DEVIATION_COST: f64 = 2.0;
const TONE_DEVIATION_FLOOR: f64 = 0.03;
const TONE_DEVIATION_SPAN: f64 = 0.05;
/// Half-width of the detail transition across an ownership seam (pixels).
const DETAIL_TRANSITION: f32 = 3.0;
/// Patch spacing of the registration correspondences (world pixels).
const REGISTRATION_STEP: f64 = 384.0;
/// Patch centres stay this far inside both source frames.
const REGISTRATION_INSET: f64 = 96.0;
const REGISTRATION_HUBER_PX: f64 = 1.5;
/// Node spacing of the smooth per-tile registration correction (source
/// pixels). A planar pose cannot follow lens distortion or a scroll that is
/// not perfectly flat; the residual it leaves is smooth over this scale.
const MESH_SPACING: f64 = 256.0;
/// Correspondence spacing of the correction solve (world pixels).
const MESH_STEP: f64 = 160.0;
/// Membrane smoothness between neighbouring correction nodes and the pull of
/// every node towards no correction, both relative to one correspondence of
/// unit weight.
const MESH_SMOOTHNESS: f64 = 1.0;
const MESH_DAMPING: f64 = 0.01;
const MESH_HUBER_PX: f64 = 1.0;
/// Every n-th correspondence stays out of the correction solve; the
/// correction is only used when it also aligns those held-out points.
const MESH_HOLDOUT: usize = 7;
const MAX_OUTPUT_PIXELS: f64 = 240_000_000.0;
const MAX_OUTPUT_UPSCALE: f64 = 1.25;
const PREVIEW_LONG_SIDE: f64 = 512.0;
/// Coordinate unit of the homography solve; keeps the normal equations
/// well conditioned for 10k-pixel frames.
const NORMALIZED_UNIT: f64 = 1000.0;

/// One decoded tile handed to the compositor.
pub(crate) struct ScanSource {
    pub(crate) rgb: Rgb32FImage,
    /// `255` where the source has an opaque sample; `None` when fully opaque.
    pub(crate) valid: Option<GrayImage>,
}

pub(crate) struct ScanMosaicResult {
    pub(crate) image: Rgb32FImage,
    /// `255` where an output pixel is backed by source content.
    pub(crate) coverage: GrayImage,
}

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

struct TileAnalysis {
    width: u32,
    height: u32,
    /// Luma blurred to two output pixels, for native registration.
    reg_full: GrayImage,
    /// Quarter-resolution luma for the coarse registration search.
    reg_quarter: GrayImage,
    /// `255` where all sixteen source pixels of a quarter sample are valid.
    valid_quarter: GrayImage,
    /// Detail energy per [`SHARPNESS_BLOCK`] block, measured in a band fixed
    /// in *output* pixels; NaN where the block is not measurable.
    sharpness: Vec<f32>,
    sharp_columns: usize,
    sharp_rows: usize,
    /// Mean RGB of every `LOW_GRID`×`LOW_GRID` source block.
    low: Vec<[f32; 3]>,
    low_valid: Vec<bool>,
    low_columns: usize,
    low_rows: usize,
    /// Bounded preview for the joint photometric solve.
    preview: Rgb32FImage,
}

#[derive(Clone, Copy)]
struct Canvas {
    width: u32,
    height: u32,
    offset_x: f64,
    offset_y: f64,
}

/// Smooth correction of one tile's planar pose: a displacement in source
/// pixels on a regular control grid, bilinearly interpolated. The tile maps
/// to world as `H(p + F(p))`.
#[derive(Clone, Debug)]
struct DisplacementGrid {
    spacing: f64,
    columns: usize,
    rows: usize,
    values: Vec<[f64; 2]>,
}

impl DisplacementGrid {
    fn zero(width: u32, height: u32, spacing: f64) -> Self {
        let columns = (f64::from(width) / spacing).ceil() as usize + 1;
        let rows = (f64::from(height) / spacing).ceil() as usize + 1;
        Self {
            spacing,
            columns: columns.max(2),
            rows: rows.max(2),
            values: vec![[0.0; 2]; columns.max(2) * rows.max(2)],
        }
    }

    fn len(&self) -> usize {
        self.columns * self.rows
    }

    /// Node indices and bilinear weights of a source point, clamped to the
    /// grid.
    fn stencil(&self, point: Point2<f64>) -> [(usize, f64); 4] {
        let gx = (point.x / self.spacing).clamp(0.0, (self.columns - 1) as f64);
        let gy = (point.y / self.spacing).clamp(0.0, (self.rows - 1) as f64);
        let ix = (gx.floor() as usize).min(self.columns - 2);
        let iy = (gy.floor() as usize).min(self.rows - 2);
        let fx = gx - ix as f64;
        let fy = gy - iy as f64;
        let base = iy * self.columns + ix;
        [
            (base, (1.0 - fx) * (1.0 - fy)),
            (base + 1, fx * (1.0 - fy)),
            (base + self.columns, (1.0 - fx) * fy),
            (base + self.columns + 1, fx * fy),
        ]
    }

    fn at(&self, point: Point2<f64>) -> [f64; 2] {
        let mut value = [0.0; 2];
        for (node, weight) in self.stencil(point) {
            value[0] += weight * self.values[node][0];
            value[1] += weight * self.values[node][1];
        }
        value
    }

    /// The source point `p` with `p + F(p) == point`.
    fn invert(&self, point: Point2<f64>) -> Point2<f64> {
        let mut source = point;
        for _ in 0..4 {
            let offset = self.at(source);
            source = Point2::new(point.x - offset[0], point.y - offset[1]);
        }
        source
    }

    fn largest(&self) -> f64 {
        self.values
            .iter()
            .map(|value| value[0].hypot(value[1]))
            .fold(0.0, f64::max)
    }
}

/// Output-world → source mapping of one tile: the inverse planar pose and,
/// when registration found one, the inverse of the smooth correction.
struct TileWarp {
    forward: Matrix3<f64>,
    inverse: Matrix3<f64>,
    correction: Option<DisplacementGrid>,
}

impl TileWarp {
    #[inline]
    fn source(&self, world: Point2<f64>) -> Option<Point2<f64>> {
        let planar = project(&self.inverse, world)?;
        Some(match &self.correction {
            Some(grid) => grid.invert(planar),
            None => planar,
        })
    }
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
fn write_test_diagnostics(
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

fn file_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn project(homography: &Matrix3<f64>, point: Point2<f64>) -> Option<Point2<f64>> {
    let mapped = homography * Vector3::new(point.x, point.y, 1.0);
    (mapped.z.abs() > 1e-12).then(|| Point2::new(mapped.x / mapped.z, mapped.y / mapped.z))
}

/// Square root of the Jacobian determinant of `homography` at `point`.
fn local_scale(homography: &Matrix3<f64>, point: Point2<f64>) -> Option<f64> {
    let centre = project(homography, point)?;
    let right = project(homography, point + nalgebra::Vector2::new(1.0, 0.0))?;
    let down = project(homography, point + nalgebra::Vector2::new(0.0, 1.0))?;
    let dx = right - centre;
    let dy = down - centre;
    let determinant = (dx.x * dy.y - dx.y * dy.x).abs();
    (determinant.is_finite() && determinant > 0.0).then(|| determinant.sqrt())
}

fn world_bounds(images: &[&ImageInfo], homographies: &[Matrix3<f64>]) -> (f64, f64, f64, f64) {
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

// ---------------------------------------------------------------------------
// Tile analysis
// ---------------------------------------------------------------------------

fn validity_of(source: &ScanSource) -> Vec<bool> {
    let rgb = source.rgb.as_raw();
    let mask = source.valid.as_ref().map(|mask| mask.as_raw());
    (0..rgb.len() / 3)
        .into_par_iter()
        .map(|index| {
            let pixel = &rgb[index * 3..index * 3 + 3];
            mask.is_none_or(|mask| mask[index] > 0) && pixel[0].max(pixel[1]).max(pixel[2]) > 1e-6
        })
        .collect()
}

fn analyse_tile(source: &ScanSource, output_scale: f64) -> TileAnalysis {
    let (width, height) = source.rgb.dimensions();
    let (w, h) = (width as usize, height as usize);
    let rgb = source.rgb.as_raw();
    let valid = validity_of(source);
    let luma = (0..w * h)
        .into_par_iter()
        .map(|index| {
            let pixel = &rgb[index * 3..index * 3 + 3];
            0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
        })
        .collect::<Vec<f32>>();
    // Sigmas are fixed in output pixels so a lower-magnification tile cannot
    // look sharper merely because its pixels are coarser.
    // The detail band stops short of every tile's Nyquist limit on the output
    // grid, so sensor noise of an upsampled tile is not mistaken for focus.
    let scale = output_scale.clamp(0.25, 4.0) as f32;
    let coarse = blur_plane(&luma, w, h, 2.5 / scale);
    let reg_full = to_gray(&coarse, width, height);
    let fine = blur_plane(&luma, w, h, 1.0 / scale);
    let (sharpness, sharp_columns, sharp_rows) =
        block_sharpness(&luma, &fine, &coarse, &valid, w, h, SHARPNESS_BLOCK);
    drop(fine);
    drop(coarse);
    let sharpness = smooth_finite_3x3(&sharpness, sharp_columns, sharp_rows);
    let (quarter, quarter_width, quarter_height) = area_downsample(&luma, w, h, 4);
    drop(luma);
    let quarter = blur_plane(&quarter, quarter_width, quarter_height, 1.0);
    let reg_quarter = to_gray(&quarter, quarter_width as u32, quarter_height as u32);
    let valid_quarter = valid_downsample(&valid, w, h, 4);
    let (low, low_valid, low_columns, low_rows) =
        area_rgb_with_validity(rgb, &valid, w, h, LOW_GRID as usize);
    let preview = photometric_preview(&low, &low_valid, low_columns, low_rows, width, height);
    TileAnalysis {
        width,
        height,
        reg_full,
        reg_quarter,
        valid_quarter,
        sharpness,
        sharp_columns,
        sharp_rows,
        low,
        low_valid,
        low_columns,
        low_rows,
        preview,
    }
}

fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil().max(1.0) as i32;
    let mut kernel = (-radius..=radius)
        .map(|offset| (-(offset * offset) as f32 / (2.0 * sigma * sigma)).exp())
        .collect::<Vec<_>>();
    let sum = kernel.iter().sum::<f32>();
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

/// Separable Gaussian blur with clamped borders.
fn blur_plane(data: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.3 || width == 0 || height == 0 {
        return data.to_vec();
    }
    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() / 2;
    let mut horizontal = vec![0.0f32; width * height];
    horizontal
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let source = &data[y * width..(y + 1) * width];
            for (x, output) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                if x >= radius && x + radius < width {
                    let window = &source[x - radius..=x + radius];
                    for (weight, value) in kernel.iter().zip(window) {
                        sum += weight * value;
                    }
                } else {
                    for (tap, weight) in kernel.iter().enumerate() {
                        let index = (x + tap).saturating_sub(radius).min(width - 1);
                        sum += weight * source[index];
                    }
                }
                *output = sum;
            }
        });
    let mut output = vec![0.0f32; width * height];
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (tap, weight) in kernel.iter().enumerate() {
                let source_y = (y + tap).saturating_sub(radius).min(height - 1);
                let source = &horizontal[source_y * width..(source_y + 1) * width];
                for (value, sample) in row.iter_mut().zip(source) {
                    *value += weight * sample;
                }
            }
        });
    output
}

fn to_gray(data: &[f32], width: u32, height: u32) -> GrayImage {
    let bytes = data
        .par_iter()
        .map(|value| (value * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    GrayImage::from_raw(width, height, bytes).expect("gray buffer matches its dimensions")
}

fn area_downsample(
    data: &[f32],
    width: usize,
    height: usize,
    factor: usize,
) -> (Vec<f32>, usize, usize) {
    let columns = width / factor;
    let rows = height / factor;
    let output = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (column, row) = (index % columns, index / columns);
            let mut sum = 0.0;
            for y in row * factor..(row + 1) * factor {
                let line = &data[y * width + column * factor..y * width + (column + 1) * factor];
                sum += line.iter().sum::<f32>();
            }
            sum / (factor * factor) as f32
        })
        .collect();
    (output, columns, rows)
}

fn valid_downsample(valid: &[bool], width: usize, height: usize, factor: usize) -> GrayImage {
    let columns = width / factor;
    let rows = height / factor;
    let output = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (column, row) = (index % columns, index / columns);
            let all = (row * factor..(row + 1) * factor).all(|y| {
                valid[y * width + column * factor..y * width + (column + 1) * factor]
                    .iter()
                    .all(|&value| value)
            });
            if all { 255u8 } else { 0u8 }
        })
        .collect();
    GrayImage::from_raw(columns as u32, rows as u32, output)
        .expect("quarter validity matches its dimensions")
}

#[allow(clippy::too_many_arguments)]
fn block_sharpness(
    luma: &[f32],
    fine: &[f32],
    coarse: &[f32],
    valid: &[bool],
    width: usize,
    height: usize,
    block: usize,
) -> (Vec<f32>, usize, usize) {
    let columns = width.div_ceil(block);
    let rows = height.div_ceil(block);
    let values = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (column, row) = (index % columns, index / columns);
            let x0 = column * block;
            let y0 = row * block;
            let x1 = (x0 + block).min(width);
            let y1 = (y0 + block).min(height);
            let mut count = 0usize;
            let mut energy = 0.0f64;
            let mut mean = 0.0f64;
            for y in y0..y1 {
                let line = y * width;
                for x in x0..x1 {
                    let index = line + x;
                    if !valid[index] {
                        continue;
                    }
                    let band = f64::from(fine[index] - coarse[index]);
                    energy += band * band;
                    mean += f64::from(luma[index]);
                    count += 1;
                }
            }
            let area = (x1 - x0) * (y1 - y0);
            if count * 10 < area * 9 || count == 0 {
                return f32::NAN;
            }
            let mean = (mean / count as f64).max(0.03);
            ((energy / count as f64).sqrt() / mean) as f32
        })
        .collect();
    (values, columns, rows)
}

fn smooth_finite_3x3(values: &[f32], columns: usize, rows: usize) -> Vec<f32> {
    (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            if !values[index].is_finite() {
                return f32::NAN;
            }
            let (column, row) = (index % columns, index / columns);
            let mut sum = 0.0;
            let mut count = 0.0;
            for y in row.saturating_sub(1)..(row + 2).min(rows) {
                for x in column.saturating_sub(1)..(column + 2).min(columns) {
                    let value = values[y * columns + x];
                    if value.is_finite() {
                        sum += value;
                        count += 1.0;
                    }
                }
            }
            sum / count
        })
        .collect()
}

fn area_rgb_with_validity(
    rgb: &[f32],
    valid: &[bool],
    width: usize,
    height: usize,
    factor: usize,
) -> (Vec<[f32; 3]>, Vec<bool>, usize, usize) {
    let columns = width.div_ceil(factor);
    let rows = height.div_ceil(factor);
    let blocks = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (column, row) = (index % columns, index / columns);
            let x0 = column * factor;
            let y0 = row * factor;
            let x1 = (x0 + factor).min(width);
            let y1 = (y0 + factor).min(height);
            let mut sum = [0.0f64; 3];
            let mut count = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let pixel = y * width + x;
                    if !valid[pixel] {
                        continue;
                    }
                    for channel in 0..3 {
                        sum[channel] += f64::from(rgb[pixel * 3 + channel]);
                    }
                    count += 1;
                }
            }
            let area = (x1 - x0) * (y1 - y0);
            if count == 0 {
                return ([0.0; 3], false);
            }
            (
                sum.map(|value| (value / count as f64) as f32),
                count == area && area == factor * factor,
            )
        })
        .collect::<Vec<_>>();
    let (values, flags): (Vec<_>, Vec<_>) = blocks.into_iter().unzip();
    (values, flags, columns, rows)
}

fn photometric_preview(
    low: &[[f32; 3]],
    low_valid: &[bool],
    columns: usize,
    rows: usize,
    width: u32,
    height: u32,
) -> Rgb32FImage {
    let grid = Rgb32FImage::from_fn(columns as u32, rows as u32, |x, y| {
        let index = y as usize * columns + x as usize;
        if low_valid[index] {
            image::Rgb(low[index])
        } else {
            image::Rgb([0.0; 3])
        }
    });
    let scale = PREVIEW_LONG_SIDE / f64::from(width.max(height).max(1));
    let preview_width = (f64::from(width) * scale).round().max(1.0) as u32;
    let preview_height = (f64::from(height) * scale).round().max(1.0) as u32;
    image::imageops::resize(
        &grid,
        preview_width,
        preview_height,
        image::imageops::FilterType::Triangle,
    )
}

fn quarter_valid(analysis: &TileAnalysis, point: Point2<f64>, radius: f64) -> bool {
    let mask = &analysis.valid_quarter;
    let check = |x: f64, y: f64| {
        let qx = (x - 1.5) / 4.0;
        let qy = (y - 1.5) / 4.0;
        qx >= 0.0
            && qy >= 0.0
            && (qx as u32) < mask.width()
            && (qy as u32) < mask.height()
            && mask.get_pixel(qx as u32, qy as u32)[0] > 0
    };
    check(point.x, point.y)
        && check(point.x - radius, point.y - radius)
        && check(point.x + radius, point.y - radius)
        && check(point.x - radius, point.y + radius)
        && check(point.x + radius, point.y + radius)
}

// ---------------------------------------------------------------------------
// Registration refinement
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Correspondence {
    first: usize,
    second: usize,
    first_point: Point2<f64>,
    second_point: Point2<f64>,
}

fn inside_frame(analysis: &TileAnalysis, point: Point2<f64>, inset: f64) -> bool {
    point.x >= inset
        && point.y >= inset
        && point.x < analysis.width as f64 - inset
        && point.y < analysis.height as f64 - inset
}

fn collect_correspondences(
    analyses: &[TileAnalysis],
    homographies: &[Matrix3<f64>],
    step: f64,
) -> Vec<Correspondence> {
    let inverses = homographies
        .iter()
        .map(|homography| homography.try_inverse())
        .collect::<Vec<_>>();
    let mut jobs = Vec::new();
    for first in 0..analyses.len() {
        for second in first + 1..analyses.len() {
            let (Some(first_inverse), Some(second_inverse)) = (inverses[first], inverses[second])
            else {
                continue;
            };
            let bounds = |index: usize| {
                let analysis = &analyses[index];
                let mut bounds = (
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                );
                for (x, y) in [
                    (0.0, 0.0),
                    (analysis.width as f64, 0.0),
                    (analysis.width as f64, analysis.height as f64),
                    (0.0, analysis.height as f64),
                ] {
                    if let Some(point) = project(&homographies[index], Point2::new(x, y)) {
                        bounds.0 = bounds.0.min(point.x);
                        bounds.1 = bounds.1.max(point.x);
                        bounds.2 = bounds.2.min(point.y);
                        bounds.3 = bounds.3.max(point.y);
                    }
                }
                bounds
            };
            let a = bounds(first);
            let b = bounds(second);
            let x0 = a.0.max(b.0);
            let x1 = a.1.min(b.1);
            let y0 = a.2.max(b.2);
            let y1 = a.3.min(b.3);
            if x1 - x0 < REGISTRATION_INSET * 3.0 || y1 - y0 < REGISTRATION_INSET * 3.0 {
                continue;
            }
            let mut y = y0 + REGISTRATION_INSET;
            while y <= y1 - REGISTRATION_INSET {
                let mut x = x0 + REGISTRATION_INSET;
                while x <= x1 - REGISTRATION_INSET {
                    let world = Point2::new(x, y);
                    if let (Some(a_point), Some(b_point)) = (
                        project(&first_inverse, world),
                        project(&second_inverse, world),
                    ) && inside_frame(&analyses[first], a_point, REGISTRATION_INSET)
                        && inside_frame(&analyses[second], b_point, REGISTRATION_INSET)
                    {
                        jobs.push((first, second, a_point));
                    }
                    x += step;
                }
                y += step;
            }
        }
    }
    let quarter = Matrix3::new(0.25, 0.0, -0.375, 0.0, 0.25, -0.375, 0.0, 0.0, 1.0);
    let quarter_inverse = Matrix3::new(4.0, 0.0, 1.5, 0.0, 4.0, 1.5, 0.0, 0.0, 1.0);
    jobs.par_iter()
        .filter_map(|&(first, second, first_point)| {
            let transform = inverses[second]? * homographies[first];
            let source = &analyses[first];
            let target = &analyses[second];
            if !quarter_valid(source, first_point, 48.0) {
                return None;
            }
            let predicted = project(&transform, first_point)?;
            if !quarter_valid(target, predicted, 48.0) {
                return None;
            }
            // Coarse search on quarter-resolution luma covers the residual
            // the 2400px feature graph leaves (tens of native pixels).
            let coarse_transform = quarter * transform * quarter_inverse;
            let coarse_centre = project(&quarter, first_point)?;
            let coarse = probe_warped_patch(
                &source.reg_quarter,
                &target.reg_quarter,
                &coarse_transform,
                coarse_centre,
                10,
                10,
                PatchMatchGates::LEGACY,
            )
            .matched?;
            let coarse_target = project(&quarter_inverse, coarse.target)?;
            let shift = coarse_target - predicted;
            let fine_transform =
                Matrix3::new(1.0, 0.0, shift.x, 0.0, 1.0, shift.y, 0.0, 0.0, 1.0) * transform;
            let fine = probe_warped_patch(
                &source.reg_full,
                &target.reg_full,
                &fine_transform,
                first_point,
                16,
                4,
                PatchMatchGates::LEGACY,
            )
            .matched?;
            Some(Correspondence {
                first,
                second,
                first_point,
                second_point: fine.target,
            })
        })
        .collect()
}

fn correspondence_residuals(
    homographies: &[Matrix3<f64>],
    correspondences: &[Correspondence],
) -> Vec<f64> {
    correspondences
        .iter()
        .map(|item| {
            match (
                project(&homographies[item.first], item.first_point),
                project(&homographies[item.second], item.second_point),
            ) {
                (Some(a), Some(b)) => (a - b).norm(),
                _ => f64::INFINITY,
            }
        })
        .collect()
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return f64::INFINITY;
    }
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

fn refine_homographies(
    analyses: &[TileAnalysis],
    initial: &[Matrix3<f64>],
    fixed: usize,
) -> Vec<Matrix3<f64>> {
    let mut current = initial.to_vec();
    for round in 0..2 {
        let started = Instant::now();
        let correspondences = collect_correspondences(analyses, &current, REGISTRATION_STEP);
        let before = correspondence_residuals(&current, &correspondences);
        let Some(solved) = solve_joint_homographies(&current, &correspondences, fixed) else {
            println!(
                "  - PSD registration round {}: joint solve unavailable ({} correspondences); keeping current poses",
                round + 1,
                correspondences.len()
            );
            break;
        };
        let after = correspondence_residuals(&solved, &correspondences);
        let (before_median, after_median) = (percentile(&before, 0.5), percentile(&after, 0.5));
        println!(
            "  - PSD registration round {}: {} native correspondences, residual median {:.2}px -> {:.2}px, p90 {:.2}px -> {:.2}px ({:.2?})",
            round + 1,
            correspondences.len(),
            before_median,
            after_median,
            percentile(&before, 0.9),
            percentile(&after, 0.9),
            started.elapsed()
        );
        // Keep a round only when it strictly helped (a NaN never does).
        if after_median.partial_cmp(&before_median) != Some(std::cmp::Ordering::Less) {
            break;
        }
        current = solved;
    }
    current
}

fn parameters_of(homography: &Matrix3<f64>) -> [f64; 8] {
    let normalize = Matrix3::new(
        1.0 / NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0 / NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let denormalize = Matrix3::new(
        NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let normalized = normalize * homography * denormalize;
    let normalized = normalized / normalized[(2, 2)];
    [
        normalized[(0, 0)],
        normalized[(0, 1)],
        normalized[(0, 2)],
        normalized[(1, 0)],
        normalized[(1, 1)],
        normalized[(1, 2)],
        normalized[(2, 0)],
        normalized[(2, 1)],
    ]
}

fn homography_of(parameters: &[f64; 8]) -> Matrix3<f64> {
    let normalized = Matrix3::new(
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
        parameters[4],
        parameters[5],
        parameters[6],
        parameters[7],
        1.0,
    );
    let normalize = Matrix3::new(
        1.0 / NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0 / NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let denormalize = Matrix3::new(
        NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        NORMALIZED_UNIT,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    denormalize * normalized * normalize
}

/// Normalised projection and its Jacobian with respect to the eight
/// parameters.
fn project_normalized(parameters: &[f64; 8], x: f64, y: f64) -> (f64, f64, [f64; 8], [f64; 8]) {
    let denominator = parameters[6] * x + parameters[7] * y + 1.0;
    let u = (parameters[0] * x + parameters[1] * y + parameters[2]) / denominator;
    let v = (parameters[3] * x + parameters[4] * y + parameters[5]) / denominator;
    let inverse = 1.0 / denominator;
    (
        u,
        v,
        [
            x * inverse,
            y * inverse,
            inverse,
            0.0,
            0.0,
            0.0,
            -u * x * inverse,
            -u * y * inverse,
        ],
        [
            0.0,
            0.0,
            0.0,
            x * inverse,
            y * inverse,
            inverse,
            -v * x * inverse,
            -v * y * inverse,
        ],
    )
}

fn huber_cost(residual: f64) -> f64 {
    if residual <= REGISTRATION_HUBER_PX {
        0.5 * residual * residual
    } else {
        REGISTRATION_HUBER_PX * (residual - 0.5 * REGISTRATION_HUBER_PX)
    }
}

/// Joint robust Levenberg–Marquardt adjustment of all free tile poses.
fn solve_joint_homographies(
    initial: &[Matrix3<f64>],
    correspondences: &[Correspondence],
    fixed: usize,
) -> Option<Vec<Matrix3<f64>>> {
    let count = initial.len();
    let mut support = vec![0usize; count];
    for item in correspondences {
        support[item.first] += 1;
        support[item.second] += 1;
    }
    let free = (0..count)
        .filter(|&index| index != fixed && support[index] >= 24)
        .collect::<Vec<_>>();
    if free.is_empty() {
        return None;
    }
    let mut slot = vec![usize::MAX; count];
    for (position, &index) in free.iter().enumerate() {
        slot[index] = position;
    }
    let dimension = free.len() * 8;
    let mut parameters = initial.iter().map(parameters_of).collect::<Vec<_>>();
    let points = correspondences
        .iter()
        .map(|item| {
            (
                item.first,
                item.second,
                item.first_point.x / NORMALIZED_UNIT,
                item.first_point.y / NORMALIZED_UNIT,
                item.second_point.x / NORMALIZED_UNIT,
                item.second_point.y / NORMALIZED_UNIT,
            )
        })
        .collect::<Vec<_>>();
    let mut active = vec![true; points.len()];
    let residual_of = |parameters: &[[f64; 8]], point: &(usize, usize, f64, f64, f64, f64)| {
        let (first, second, ax, ay, bx, by) = *point;
        let (u1, v1, _, _) = project_normalized(&parameters[first], ax, ay);
        let (u2, v2, _, _) = project_normalized(&parameters[second], bx, by);
        ((u1 - u2) * NORMALIZED_UNIT, (v1 - v2) * NORMALIZED_UNIT)
    };
    let total_cost = |parameters: &[[f64; 8]], active: &[bool]| -> f64 {
        points
            .iter()
            .zip(active)
            .filter(|(_, active)| **active)
            .map(|(point, _)| {
                let (rx, ry) = residual_of(parameters, point);
                huber_cost(rx.hypot(ry))
            })
            .sum()
    };
    for pass in 0..2 {
        let mut damping = 1e-3;
        let mut cost = total_cost(&parameters, &active);
        for _ in 0..50 {
            let mut normal = DMatrix::<f64>::zeros(dimension, dimension);
            let mut gradient = DVector::<f64>::zeros(dimension);
            for (point, _) in points.iter().zip(&active).filter(|(_, active)| **active) {
                let (first, second, ax, ay, bx, by) = *point;
                let (u1, v1, ju1, jv1) = project_normalized(&parameters[first], ax, ay);
                let (u2, v2, ju2, jv2) = project_normalized(&parameters[second], bx, by);
                let rx = (u1 - u2) * NORMALIZED_UNIT;
                let ry = (v1 - v2) * NORMALIZED_UNIT;
                let magnitude = rx.hypot(ry);
                let weight = if magnitude <= REGISTRATION_HUBER_PX {
                    1.0
                } else {
                    REGISTRATION_HUBER_PX / magnitude
                };
                let blocks = [(slot[first], ju1, jv1, 1.0), (slot[second], ju2, jv2, -1.0)];
                for &(row_slot, ru, rv, row_sign) in &blocks {
                    if row_slot == usize::MAX {
                        continue;
                    }
                    for i in 0..8 {
                        let jxi = ru[i] * NORMALIZED_UNIT * row_sign;
                        let jyi = rv[i] * NORMALIZED_UNIT * row_sign;
                        gradient[row_slot * 8 + i] += weight * (jxi * rx + jyi * ry);
                        for &(column_slot, cu, cv, column_sign) in &blocks {
                            if column_slot == usize::MAX {
                                continue;
                            }
                            for j in 0..8 {
                                let jxj = cu[j] * NORMALIZED_UNIT * column_sign;
                                let jyj = cv[j] * NORMALIZED_UNIT * column_sign;
                                normal[(row_slot * 8 + i, column_slot * 8 + j)] +=
                                    weight * (jxi * jxj + jyi * jyj);
                            }
                        }
                    }
                }
            }
            let mut improved = false;
            for _ in 0..8 {
                let mut system = normal.clone();
                for d in 0..dimension {
                    system[(d, d)] += damping * normal[(d, d)].max(1e-9);
                }
                let Some(cholesky) = system.cholesky() else {
                    damping *= 10.0;
                    continue;
                };
                let step = cholesky.solve(&(-&gradient));
                let mut trial = parameters.clone();
                for &index in &free {
                    for i in 0..8 {
                        trial[index][i] += step[slot[index] * 8 + i];
                    }
                }
                let trial_cost = total_cost(&trial, &active);
                if trial_cost.is_finite() && trial_cost < cost {
                    let largest = step.amax();
                    parameters = trial;
                    cost = trial_cost;
                    damping = (damping * 0.3).max(1e-9);
                    improved = largest > 1e-12;
                    break;
                }
                damping *= 10.0;
            }
            if !improved {
                break;
            }
        }
        if pass == 0 {
            let residuals = points
                .iter()
                .map(|point| {
                    let (rx, ry) = residual_of(&parameters, point);
                    rx.hypot(ry)
                })
                .collect::<Vec<_>>();
            let median = percentile(&residuals, 0.5);
            let limit = (median * 4.0).max(4.0);
            for (flag, residual) in active.iter_mut().zip(&residuals) {
                *flag = *residual <= limit;
            }
        }
    }
    let solved = parameters.iter().map(homography_of).collect::<Vec<_>>();
    solved
        .iter()
        .all(|matrix| {
            matrix.iter().all(|value| value.is_finite()) && matrix.try_inverse().is_some()
        })
        .then_some(solved)
}

// ---------------------------------------------------------------------------
// Smooth registration correction
// ---------------------------------------------------------------------------

/// Jacobian `[[du/dx, du/dy], [dv/dx, dv/dy]]` of a homography at `point`.
fn homography_jacobian(homography: &Matrix3<f64>, point: Point2<f64>) -> Option<[[f64; 2]; 2]> {
    let mapped = homography * Vector3::new(point.x, point.y, 1.0);
    if mapped.z.abs() < 1e-12 {
        return None;
    }
    let (u, v, w) = (mapped.x / mapped.z, mapped.y / mapped.z, mapped.z);
    Some([
        [
            (homography[(0, 0)] - u * homography[(2, 0)]) / w,
            (homography[(0, 1)] - u * homography[(2, 1)]) / w,
        ],
        [
            (homography[(1, 0)] - v * homography[(2, 0)]) / w,
            (homography[(1, 1)] - v * homography[(2, 1)]) / w,
        ],
    ])
}

/// One correspondence of the correction solve, linearised at no correction:
/// `r(x) = r0 + Ja·F_a(p_a) − Jb·F_b(p_b)` with both corrections bilinear in
/// the node unknowns.
struct MeshRow {
    first: [(usize, f64); 4],
    second: [(usize, f64); 4],
    first_jacobian: [[f64; 2]; 2],
    second_jacobian: [[f64; 2]; 2],
    initial: [f64; 2],
}

impl MeshRow {
    #[inline]
    fn correction(stencil: &[(usize, f64); 4], unknowns: &[f64]) -> [f64; 2] {
        let mut value = [0.0; 2];
        for &(node, weight) in stencil {
            value[0] += weight * unknowns[node * 2];
            value[1] += weight * unknowns[node * 2 + 1];
        }
        value
    }

    #[inline]
    fn linear_part(&self, unknowns: &[f64]) -> [f64; 2] {
        let a = Self::correction(&self.first, unknowns);
        let b = Self::correction(&self.second, unknowns);
        let (ja, jb) = (&self.first_jacobian, &self.second_jacobian);
        [
            ja[0][0] * a[0] + ja[0][1] * a[1] - jb[0][0] * b[0] - jb[0][1] * b[1],
            ja[1][0] * a[0] + ja[1][1] * a[1] - jb[1][0] * b[0] - jb[1][1] * b[1],
        ]
    }

    #[inline]
    fn residual(&self, unknowns: &[f64]) -> [f64; 2] {
        let linear = self.linear_part(unknowns);
        [self.initial[0] + linear[0], self.initial[1] + linear[1]]
    }

    /// Adds `weight · Aᵀ·value` of this row to `output`.
    #[inline]
    fn scatter(&self, value: [f64; 2], weight: f64, output: &mut [f64]) {
        let (ja, jb) = (&self.first_jacobian, &self.second_jacobian);
        let a = [
            weight * (ja[0][0] * value[0] + ja[1][0] * value[1]),
            weight * (ja[0][1] * value[0] + ja[1][1] * value[1]),
        ];
        let b = [
            weight * (jb[0][0] * value[0] + jb[1][0] * value[1]),
            weight * (jb[0][1] * value[0] + jb[1][1] * value[1]),
        ];
        for &(node, stencil_weight) in &self.first {
            output[node * 2] += stencil_weight * a[0];
            output[node * 2 + 1] += stencil_weight * a[1];
        }
        for &(node, stencil_weight) in &self.second {
            output[node * 2] -= stencil_weight * b[0];
            output[node * 2 + 1] -= stencil_weight * b[1];
        }
    }
}

/// `Aᵀ·W·(A·x + shift)` over all rows, accumulated in parallel.
fn mesh_normal_product(
    rows: &[MeshRow],
    weights: &[f64],
    unknowns: &[f64],
    with_initial: bool,
) -> Vec<f64> {
    let length = unknowns.len();
    rows.par_chunks(1024)
        .zip(weights.par_chunks(1024))
        .fold(
            || vec![0.0; length],
            |mut output, (rows, weights)| {
                for (row, &weight) in rows.iter().zip(weights) {
                    if weight <= 0.0 {
                        continue;
                    }
                    let value = if with_initial {
                        row.residual(unknowns)
                    } else {
                        row.linear_part(unknowns)
                    };
                    row.scatter(value, weight, &mut output);
                }
                output
            },
        )
        .reduce(
            || vec![0.0; length],
            |mut left, right| {
                for (value, other) in left.iter_mut().zip(&right) {
                    *value += other;
                }
                left
            },
        )
}

/// `(AᵀWA + λs·L + λ0·I)·x`.
fn mesh_system_product(
    rows: &[MeshRow],
    weights: &[f64],
    edges: &[(usize, usize)],
    unknowns: &[f64],
) -> Vec<f64> {
    let mut output = mesh_normal_product(rows, weights, unknowns, false);
    for &(first, second) in edges {
        for component in 0..2 {
            let difference = MESH_SMOOTHNESS
                * (unknowns[first * 2 + component] - unknowns[second * 2 + component]);
            output[first * 2 + component] += difference;
            output[second * 2 + component] -= difference;
        }
    }
    for (value, unknown) in output.iter_mut().zip(unknowns) {
        *value += MESH_DAMPING * unknown;
    }
    output
}

/// Jacobi-preconditioned conjugate gradients on the correction normal
/// equations, warm-started from `unknowns`.
fn solve_mesh_system(
    rows: &[MeshRow],
    weights: &[f64],
    edges: &[(usize, usize)],
    unknowns: &mut [f64],
) -> usize {
    let length = unknowns.len();
    let mut diagonal = vec![MESH_DAMPING; length];
    for &(first, second) in edges {
        for component in 0..2 {
            diagonal[first * 2 + component] += MESH_SMOOTHNESS;
            diagonal[second * 2 + component] += MESH_SMOOTHNESS;
        }
    }
    for (row, &weight) in rows.iter().zip(weights) {
        if weight <= 0.0 {
            continue;
        }
        for (stencil, jacobian) in [
            (&row.first, &row.first_jacobian),
            (&row.second, &row.second_jacobian),
        ] {
            let column_x = jacobian[0][0] * jacobian[0][0] + jacobian[1][0] * jacobian[1][0];
            let column_y = jacobian[0][1] * jacobian[0][1] + jacobian[1][1] * jacobian[1][1];
            for &(node, stencil_weight) in stencil {
                let squared = weight * stencil_weight * stencil_weight;
                diagonal[node * 2] += squared * column_x;
                diagonal[node * 2 + 1] += squared * column_y;
            }
        }
    }
    // Right-hand side −AᵀW·r0 and the initial residual b − M·x.
    let zero = vec![0.0; length];
    let rhs = mesh_normal_product(rows, weights, &zero, true)
        .into_iter()
        .map(|value| -value)
        .collect::<Vec<_>>();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
    if rhs_norm < 1e-12 {
        return 0;
    }
    let product = mesh_system_product(rows, weights, edges, unknowns);
    let mut residual = rhs
        .iter()
        .zip(&product)
        .map(|(b, ax)| b - ax)
        .collect::<Vec<_>>();
    let mut preconditioned = residual
        .iter()
        .zip(&diagonal)
        .map(|(r, d)| r / d)
        .collect::<Vec<_>>();
    let mut direction = preconditioned.clone();
    let mut rz = residual
        .iter()
        .zip(&preconditioned)
        .map(|(r, z)| r * z)
        .sum::<f64>();
    let mut iterations = 0;
    for iteration in 0..3000 {
        iterations = iteration;
        let residual_norm = residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if residual_norm <= 1e-7 * rhs_norm {
            break;
        }
        let product = mesh_system_product(rows, weights, edges, &direction);
        let curvature = direction
            .iter()
            .zip(&product)
            .map(|(p, ap)| p * ap)
            .sum::<f64>();
        if curvature <= 0.0 || !curvature.is_finite() {
            break;
        }
        let alpha = rz / curvature;
        for (unknown, step) in unknowns.iter_mut().zip(&direction) {
            *unknown += alpha * step;
        }
        for (((r, z), ap), d) in residual
            .iter_mut()
            .zip(preconditioned.iter_mut())
            .zip(&product)
            .zip(&diagonal)
        {
            *r -= alpha * ap;
            *z = *r / d;
        }
        let next = residual
            .iter()
            .zip(&preconditioned)
            .map(|(r, z)| r * z)
            .sum::<f64>();
        let beta = next / rz.max(1e-300);
        rz = next;
        for (p, z) in direction.iter_mut().zip(&preconditioned) {
            *p = z + beta * *p;
        }
    }
    iterations
}

/// Fit one smooth correction per tile so that every native correspondence
/// lands on the same world point. Returns `None` when the correction does not
/// also align the held-out correspondences better than the planar poses.
fn refine_mesh(
    analyses: &[TileAnalysis],
    homographies: &[Matrix3<f64>],
) -> Option<Vec<DisplacementGrid>> {
    let started = Instant::now();
    let correspondences = collect_correspondences(analyses, homographies, MESH_STEP);
    if correspondences.len() < 64 {
        println!(
            "  - PSD registration correction skipped: {} correspondences",
            correspondences.len()
        );
        return None;
    }
    let mut grids = analyses
        .iter()
        .map(|analysis| DisplacementGrid::zero(analysis.width, analysis.height, MESH_SPACING))
        .collect::<Vec<_>>();
    let mut offsets = Vec::with_capacity(grids.len());
    let mut nodes = 0usize;
    for grid in &grids {
        offsets.push(nodes);
        nodes += grid.len();
    }
    let mut edges = Vec::new();
    for (grid, &offset) in grids.iter().zip(&offsets) {
        for row in 0..grid.rows {
            for column in 0..grid.columns {
                let node = offset + row * grid.columns + column;
                if column + 1 < grid.columns {
                    edges.push((node, node + 1));
                }
                if row + 1 < grid.rows {
                    edges.push((node, node + grid.columns));
                }
            }
        }
    }
    let (correspondences, rows): (Vec<_>, Vec<_>) = correspondences
        .into_iter()
        .filter_map(|item| {
            let a = project(&homographies[item.first], item.first_point)?;
            let b = project(&homographies[item.second], item.second_point)?;
            let shift = |stencil: [(usize, f64); 4], offset: usize| {
                stencil.map(|(node, weight)| (node + offset, weight))
            };
            let row = MeshRow {
                first: shift(
                    grids[item.first].stencil(item.first_point),
                    offsets[item.first],
                ),
                second: shift(
                    grids[item.second].stencil(item.second_point),
                    offsets[item.second],
                ),
                first_jacobian: homography_jacobian(&homographies[item.first], item.first_point)?,
                second_jacobian: homography_jacobian(
                    &homographies[item.second],
                    item.second_point,
                )?,
                initial: [a.x - b.x, a.y - b.y],
            };
            Some((item, row))
        })
        .unzip();
    let holdout = (0..rows.len())
        .map(|index| index % MESH_HOLDOUT == MESH_HOLDOUT - 1)
        .collect::<Vec<_>>();
    let mut active = holdout.iter().map(|&held| !held).collect::<Vec<_>>();
    let mut unknowns = vec![0.0; nodes * 2];
    let mut iterations = 0;
    for pass in 0..5 {
        let residuals = rows
            .iter()
            .map(|row| {
                let r = row.residual(&unknowns);
                r[0].hypot(r[1])
            })
            .collect::<Vec<_>>();
        if pass == 3 {
            // Reject what the fitted correction cannot explain (repeated
            // texture, glare) and refit without it.
            let fitted = residuals
                .iter()
                .zip(&active)
                .filter(|(_, active)| **active)
                .map(|(residual, _)| *residual)
                .collect::<Vec<_>>();
            let limit = (percentile(&fitted, 0.5) * 4.0).max(2.0);
            for (flag, residual) in active.iter_mut().zip(&residuals) {
                *flag = *flag && *residual <= limit;
            }
        }
        let weights = residuals
            .iter()
            .zip(&active)
            .map(|(&residual, &active)| {
                if !active {
                    0.0
                } else if residual <= MESH_HUBER_PX {
                    1.0
                } else {
                    MESH_HUBER_PX / residual
                }
            })
            .collect::<Vec<_>>();
        iterations += solve_mesh_system(&rows, &weights, &edges, &mut unknowns);
    }
    for (grid, &offset) in grids.iter_mut().zip(&offsets) {
        for (index, value) in grid.values.iter_mut().enumerate() {
            let node = offset + index;
            *value = [unknowns[node * 2], unknowns[node * 2 + 1]];
        }
    }
    if grids.iter().any(|grid| {
        grid.values
            .iter()
            .any(|value| !value[0].is_finite() || !value[1].is_finite())
    }) {
        println!("  - PSD registration correction rejected: non-finite solution");
        return None;
    }
    // Exact (not linearised) residuals of the corrected poses.
    let corrected = |item: &Correspondence| -> f64 {
        let moved = |index: usize, point: Point2<f64>| {
            let offset = grids[index].at(point);
            project(
                &homographies[index],
                Point2::new(point.x + offset[0], point.y + offset[1]),
            )
        };
        match (
            moved(item.first, item.first_point),
            moved(item.second, item.second_point),
        ) {
            (Some(a), Some(b)) => (a - b).norm(),
            _ => f64::INFINITY,
        }
    };
    let planar = correspondence_residuals(homographies, &correspondences);
    let after = correspondences.iter().map(corrected).collect::<Vec<_>>();
    let pick = |values: &[f64], held: bool| {
        values
            .iter()
            .zip(&holdout)
            .filter(|(_, flag)| **flag == held)
            .map(|(value, _)| *value)
            .collect::<Vec<_>>()
    };
    let (held_before, held_after) = (pick(&planar, true), pick(&after, true));
    let (fit_before, fit_after) = (pick(&planar, false), pick(&after, false));
    let largest = grids
        .iter()
        .map(DisplacementGrid::largest)
        .fold(0.0, f64::max);
    println!(
        "  - PSD registration correction: {} correspondences ({} active), fitted median {:.2}px -> {:.2}px p90 {:.2}px -> {:.2}px; held-out median {:.2}px -> {:.2}px p90 {:.2}px -> {:.2}px; largest correction {:.2}px; {} CG iterations ({:.2?})",
        rows.len(),
        active.iter().filter(|&&flag| flag).count(),
        percentile(&fit_before, 0.5),
        percentile(&fit_after, 0.5),
        percentile(&fit_before, 0.9),
        percentile(&fit_after, 0.9),
        percentile(&held_before, 0.5),
        percentile(&held_after, 0.5),
        percentile(&held_before, 0.9),
        percentile(&held_after, 0.9),
        largest,
        iterations,
        started.elapsed()
    );
    let improves = percentile(&held_after, 0.9) < percentile(&held_before, 0.9)
        && percentile(&held_after, 0.5) <= percentile(&held_before, 0.5);
    if !improves {
        println!("  - PSD registration correction rejected: held-out residuals did not improve");
        return None;
    }
    Some(grids)
}

// ---------------------------------------------------------------------------
// Photometry
// ---------------------------------------------------------------------------

fn solve_photometry(
    analyses: &[TileAnalysis],
    homographies: &[Matrix3<f64>],
) -> Vec<(PhotometricModel, f32)> {
    let tiles = analyses
        .iter()
        .zip(homographies)
        .map(|(analysis, homography)| {
            let scale_x = analysis.preview.width() as f64 / analysis.width.max(1) as f64;
            let scale_y = analysis.preview.height() as f64 / analysis.height.max(1) as f64;
            let preview_to_source = Matrix3::new(
                1.0 / scale_x.max(1e-9),
                0.0,
                0.0,
                0.0,
                1.0 / scale_y.max(1e-9),
                0.0,
                0.0,
                0.0,
                1.0,
            );
            (analysis.preview.clone(), homography * preview_to_source)
        })
        .collect::<Vec<_>>();
    let medians = tiles
        .iter()
        .map(|(preview, _)| flat_scan_paper_median(preview))
        .collect::<Vec<_>>();
    let mut samples = medians.iter().copied().flatten().collect::<Vec<_>>();
    samples.sort_by(f32::total_cmp);
    let target = samples.get(samples.len() / 2).copied().unwrap_or(0.5);
    let gains = medians
        .iter()
        .map(|median| {
            median
                .map(|value| (target / value.max(0.05)).clamp(0.78, 1.28))
                .unwrap_or(1.0)
        })
        .collect::<Vec<_>>();
    let options = PhotometricOptions {
        allow_linear: true,
        min_samples_per_pair: 32,
        max_samples_per_pair: 4096,
        max_pair_log_scatter: 0.30,
        ..PhotometricOptions::default()
    };
    let calibration = calibrate_overlap_photometry(&tiles, &options);
    println!(
        "  - PSD joint photometric solve: {} models, {} reliable pairs, linear={}, paper target {:.3}",
        calibration.models.len(),
        calibration
            .pairs
            .iter()
            .filter(|pair| pair.reliable)
            .count(),
        calibration.linear_used,
        target
    );
    calibration.models.into_iter().zip(gains).collect()
}

fn corrected_low_value(
    analysis: &TileAnalysis,
    model: &PhotometricModel,
    gain: f32,
    source: Point2<f64>,
    value: [f32; 3],
) -> [f32; 3] {
    let preview_x = source.x * analysis.preview.width() as f64 / analysis.width.max(1) as f64;
    let preview_y = source.y * analysis.preview.height() as f64 / analysis.height.max(1) as f64;
    let channel_gain = model.gain_at(preview_x, preview_y);
    let offset = model.offset_at(preview_x, preview_y);
    std::array::from_fn(|channel| {
        ((value[channel] * channel_gain[channel] + offset[channel]) * gain).clamp(0.0, 1.0)
    })
}

// ---------------------------------------------------------------------------
// Ownership
// ---------------------------------------------------------------------------

struct Ownership {
    columns: usize,
    rows: usize,
    /// One-based owner per cell, [`NO_OWNER`] where no tile covers the centre.
    owners: Vec<u16>,
    /// Cells whose pixels may need a non-owner fallback sample.
    fallback: Vec<bool>,
}

impl Ownership {
    fn owner(&self, column: isize, row: isize) -> u16 {
        if column < 0 || row < 0 || column as usize >= self.columns || row as usize >= self.rows {
            return NO_OWNER;
        }
        self.owners[row as usize * self.columns + column as usize]
    }
}

fn sample_sharpness(analysis: &TileAnalysis, source: Point2<f64>) -> Option<f32> {
    let x = source.x / SHARPNESS_BLOCK as f64 - 0.5;
    let y = source.y / SHARPNESS_BLOCK as f64 - 0.5;
    let x0 = x.floor().clamp(0.0, (analysis.sharp_columns - 1) as f64) as usize;
    let y0 = y.floor().clamp(0.0, (analysis.sharp_rows - 1) as f64) as usize;
    let x1 = (x0 + 1).min(analysis.sharp_columns - 1);
    let y1 = (y0 + 1).min(analysis.sharp_rows - 1);
    let fx = (x - x0 as f64).clamp(0.0, 1.0) as f32;
    let fy = (y - y0 as f64).clamp(0.0, 1.0) as f32;
    let mut sum = 0.0;
    let mut weight = 0.0;
    for (column, row, w) in [
        (x0, y0, (1.0 - fx) * (1.0 - fy)),
        (x1, y0, fx * (1.0 - fy)),
        (x0, y1, (1.0 - fx) * fy),
        (x1, y1, fx * fy),
    ] {
        let value = analysis.sharpness[row * analysis.sharp_columns + column];
        if value.is_finite() && w > 0.0 {
            sum += value * w;
            weight += w;
        }
    }
    (weight > 1e-4).then(|| sum / weight)
}

fn sample_gray_bilinear(image: &GrayImage, x: f64, y: f64) -> Option<f32> {
    if x < 0.0 || y < 0.0 || x >= (image.width() - 1) as f64 || y >= (image.height() - 1) as f64 {
        return None;
    }
    let ix = x as u32;
    let iy = y as u32;
    let fx = (x - ix as f64) as f32;
    let fy = (y - iy as f64) as f32;
    let at = |x: u32, y: u32| f32::from(image.get_pixel(x, y)[0]);
    let top = at(ix, iy) * (1.0 - fx) + at(ix + 1, iy) * fx;
    let bottom = at(ix, iy + 1) * (1.0 - fx) + at(ix + 1, iy + 1) * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

/// Mean relative departure of one tile's low band from the consensus over an
/// ownership cell; zero where the tile has no valid low samples there.
fn tone_deviation(band: &LowBand, consensus: &LowBand, column: usize, row: usize) -> f64 {
    let per_cell = (OWNER_CELL / LOW_GRID) as usize;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in row * per_cell..(row + 1) * per_cell {
        for x in column * per_cell..(column + 1) * per_cell {
            if !band.valid_at(x, y) || !consensus.valid_at(x, y) {
                continue;
            }
            let value = band.values[(y - band.y0) * band.columns + (x - band.x0)];
            let reference = consensus.values[y * consensus.columns + x];
            let deviation = (0..3)
                .map(|channel| {
                    (value[channel] - reference[channel]).abs() / reference[channel].max(0.05)
                })
                .fold(0.0f32, f32::max);
            sum += f64::from(deviation);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn plan_ownership(
    analyses: &[TileAnalysis],
    warps: &[TileWarp],
    canvas: &Canvas,
    bands: &[LowBand],
    consensus: &LowBand,
) -> Ownership {
    let started = Instant::now();
    let columns = canvas.width.div_ceil(OWNER_CELL) as usize;
    let rows = canvas.height.div_ceil(OWNER_CELL) as usize;
    let labels = analyses.len();
    let cell = OWNER_CELL as f64;
    let reach = cell * 0.5 + OWNER_MARGIN;
    // (label, sharpness, eligible) of every tile covering a cell centre, and
    // the local structure that prices a seam through the cell.
    let measurements = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let (column, row) = (index % columns, index / columns);
            let centre_x = column as f64 * cell + cell * 0.5 - 0.5;
            let centre_y = row as f64 * cell + cell * 0.5 - 0.5;
            let world = |dx: f64, dy: f64| {
                Point2::new(
                    centre_x + dx - canvas.offset_x,
                    centre_y + dy - canvas.offset_y,
                )
            };
            let mut candidates = Vec::new();
            for (label, (analysis, warp)) in analyses.iter().zip(warps).enumerate() {
                let Some(source) = warp.source(world(0.0, 0.0)) else {
                    continue;
                };
                if !quarter_valid(analysis, source, 0.0) {
                    continue;
                }
                let Some(sharpness) = sample_sharpness(analysis, source) else {
                    continue;
                };
                let eligible = [
                    (-reach, -reach),
                    (reach, -reach),
                    (-reach, reach),
                    (reach, reach),
                ]
                .iter()
                .all(|&(dx, dy)| {
                    warp.source(world(dx, dy)).is_some_and(|point| {
                        inside_frame(analysis, point, 2.0) && quarter_valid(analysis, point, 0.0)
                    })
                });
                let deviation = tone_deviation(&bands[label], consensus, column, row);
                candidates.push((label, sharpness, eligible, deviation));
            }
            // Structure of the sharpest candidate: local contrast of its
            // blurred quarter-resolution luma across the cell.
            let structure = candidates
                .iter()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .map(|&(label, _, _, _)| {
                    let analysis = &analyses[label];
                    let mut values = Vec::with_capacity(36);
                    for sy in 0..6 {
                        for sx in 0..6 {
                            let dx = (sx as f64 + 0.5) / 6.0 * cell - cell * 0.5;
                            let dy = (sy as f64 + 0.5) / 6.0 * cell - cell * 0.5;
                            if let Some(source) = warps[label].source(world(dx, dy))
                                && let Some(value) = sample_gray_bilinear(
                                    &analysis.reg_quarter,
                                    (source.x - 1.5) / 4.0,
                                    (source.y - 1.5) / 4.0,
                                )
                            {
                                values.push(value);
                            }
                        }
                    }
                    if values.len() < 8 {
                        return 0.0;
                    }
                    let mean = values.iter().sum::<f32>() / values.len() as f32;
                    (values
                        .iter()
                        .map(|value| (value - mean).powi(2))
                        .sum::<f32>()
                        / values.len() as f32)
                        .sqrt()
                })
                .unwrap_or(0.0);
            (candidates, structure)
        })
        .collect::<Vec<_>>();

    let mut structures = measurements
        .iter()
        .filter(|(candidates, _)| !candidates.is_empty())
        .map(|(_, structure)| f64::from(*structure))
        .collect::<Vec<_>>();
    structures.sort_by(f64::total_cmp);
    let quantile = |fraction: f64| {
        structures
            .get(((structures.len().max(1) - 1) as f64 * fraction).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let (structure_floor, structure_ceiling) = (quantile(0.5), quantile(0.95));
    let structure_span = (structure_ceiling - structure_floor).max(1e-6);

    let mut grid = OwnershipCostGrid::new(columns, rows, labels);
    let mut eligible_owner = vec![false; columns * rows * labels];
    for (index, (candidates, structure)) in measurements.iter().enumerate() {
        let best = candidates
            .iter()
            .map(|candidate| candidate.1)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        for &(label, sharpness, eligible, deviation) in candidates {
            let mut cost = SHARPNESS_COST * (1.0 - f64::from(sharpness / best)).clamp(0.0, 1.0);
            cost += TONE_DEVIATION_COST
                * ((deviation - TONE_DEVIATION_FLOOR) / TONE_DEVIATION_SPAN).clamp(0.0, 1.0);
            if !eligible {
                cost += EDGE_OWNER_PENALTY;
            }
            grid.set_cost(index, label, Some(cost));
            eligible_owner[index * labels + label] = eligible;
        }
        let disagreement = ((f64::from(*structure) - structure_floor) / structure_span)
            .clamp(0.0, 1.0)
            * STRUCTURE_SEAM_WEIGHT;
        grid.set_disagreement(index, disagreement);
    }
    let mut owners = grid.per_cell_minimum();
    let mut energy = grid.energy(&owners);
    let initial_energy = energy;
    for _ in 0..3 {
        let mut improved = false;
        for label in 0..labels {
            let candidate = grid.fold_candidate(&owners, label);
            let candidate_energy = grid.energy(&candidate);
            if candidate_energy < energy - 1e-9 {
                owners = candidate;
                energy = candidate_energy;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    let mut owned = vec![0usize; labels];
    let mut fallback = vec![false; columns * rows];
    for (index, &owner) in owners.iter().enumerate() {
        if owner == NO_OWNER {
            fallback[index] = true;
            continue;
        }
        let label = usize::from(owner - 1);
        owned[label] += 1;
        if !eligible_owner[index * labels + label] {
            fallback[index] = true;
        }
    }
    println!(
        "  - PSD ownership: {}x{} cells, energy {:.1} -> {:.1}, owned cells per tile {:?} ({:.2?})",
        columns,
        rows,
        initial_energy,
        energy,
        owned,
        started.elapsed()
    );
    Ownership {
        columns,
        rows,
        owners,
        fallback,
    }
}

// ---------------------------------------------------------------------------
// Low-frequency band
// ---------------------------------------------------------------------------

struct LowBand {
    x0: usize,
    y0: usize,
    columns: usize,
    rows: usize,
    values: Vec<[f32; 3]>,
    valid: Vec<bool>,
}

impl LowBand {
    fn empty() -> Self {
        Self {
            x0: 0,
            y0: 0,
            columns: 0,
            rows: 0,
            values: Vec::new(),
            valid: Vec::new(),
        }
    }

    /// Bilinear sample in global low-grid coordinates; clamped to the band.
    fn sample(&self, x: f64, y: f64) -> [f32; 3] {
        if self.columns == 0 || self.rows == 0 {
            return [0.0; 3];
        }
        let lx = (x - self.x0 as f64).clamp(0.0, (self.columns - 1) as f64);
        let ly = (y - self.y0 as f64).clamp(0.0, (self.rows - 1) as f64);
        let ix = (lx as usize).min(self.columns.saturating_sub(2));
        let iy = (ly as usize).min(self.rows.saturating_sub(2));
        let ix1 = (ix + 1).min(self.columns - 1);
        let iy1 = (iy + 1).min(self.rows - 1);
        let fx = (lx - ix as f64).clamp(0.0, 1.0) as f32;
        let fy = (ly - iy as f64).clamp(0.0, 1.0) as f32;
        let a = self.values[iy * self.columns + ix];
        let b = self.values[iy * self.columns + ix1];
        let c = self.values[iy1 * self.columns + ix];
        let d = self.values[iy1 * self.columns + ix1];
        std::array::from_fn(|channel| {
            let top = a[channel] * (1.0 - fx) + b[channel] * fx;
            let bottom = c[channel] * (1.0 - fx) + d[channel] * fx;
            top * (1.0 - fy) + bottom * fy
        })
    }

    fn valid_at(&self, x: usize, y: usize) -> bool {
        x >= self.x0
            && y >= self.y0
            && x < self.x0 + self.columns
            && y < self.y0 + self.rows
            && self.valid[(y - self.y0) * self.columns + (x - self.x0)]
    }
}

fn low_grid_size(canvas: &Canvas) -> (usize, usize) {
    (
        canvas.width.div_ceil(LOW_GRID) as usize,
        canvas.height.div_ceil(LOW_GRID) as usize,
    )
}

fn low_grid_centre(index: usize) -> f64 {
    index as f64 * LOW_GRID as f64 + (LOW_GRID as f64 - 1.0) * 0.5
}

/// One tile's photometrically corrected low band on the output low grid.
fn low_band(
    analysis: &TileAnalysis,
    warp: &TileWarp,
    model: &PhotometricModel,
    gain: f32,
    canvas: &Canvas,
) -> LowBand {
    let (grid_width, grid_height) = low_grid_size(canvas);
    let forward = warp.forward;
    let mut bounds = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for (x, y) in [
        (0.0, 0.0),
        (analysis.width as f64, 0.0),
        (analysis.width as f64, analysis.height as f64),
        (0.0, analysis.height as f64),
    ] {
        if let Some(point) = project(&forward, Point2::new(x, y)) {
            let gx = (point.x + canvas.offset_x) / LOW_GRID as f64;
            let gy = (point.y + canvas.offset_y) / LOW_GRID as f64;
            bounds.0 = bounds.0.min(gx);
            bounds.1 = bounds.1.max(gx);
            bounds.2 = bounds.2.min(gy);
            bounds.3 = bounds.3.max(gy);
        }
    }
    if !bounds.0.is_finite() {
        return LowBand::empty();
    }
    let x0 = (bounds.0.floor() - 2.0).max(0.0) as usize;
    let y0 = (bounds.2.floor() - 2.0).max(0.0) as usize;
    let x1 = ((bounds.1.ceil() + 2.0).max(0.0) as usize).min(grid_width);
    let y1 = ((bounds.3.ceil() + 2.0).max(0.0) as usize).min(grid_height);
    if x1 <= x0 || y1 <= y0 {
        return LowBand::empty();
    }
    let columns = x1 - x0;
    let rows = y1 - y0;
    let lw = analysis.low_columns;
    let lh = analysis.low_rows;
    let sampled = (0..columns * rows)
        .into_par_iter()
        .map(|index| {
            let gx = x0 + index % columns;
            let gy = y0 + index / columns;
            let world = Point2::new(
                low_grid_centre(gx) - canvas.offset_x,
                low_grid_centre(gy) - canvas.offset_y,
            );
            let Some(source) = warp.source(world) else {
                return ([0.0; 3], false);
            };
            let lx = (source.x - (LOW_GRID as f64 - 1.0) * 0.5) / LOW_GRID as f64;
            let ly = (source.y - (LOW_GRID as f64 - 1.0) * 0.5) / LOW_GRID as f64;
            if lx < 0.0 || ly < 0.0 || lx >= (lw - 1) as f64 || ly >= (lh - 1) as f64 {
                return ([0.0; 3], false);
            }
            let ix = lx as usize;
            let iy = ly as usize;
            let fx = (lx - ix as f64) as f32;
            let fy = (ly - iy as f64) as f32;
            let corners = [
                iy * lw + ix,
                iy * lw + ix + 1,
                (iy + 1) * lw + ix,
                (iy + 1) * lw + ix + 1,
            ];
            if corners.iter().any(|&corner| !analysis.low_valid[corner]) {
                return ([0.0; 3], false);
            }
            let weights = [
                (1.0 - fx) * (1.0 - fy),
                fx * (1.0 - fy),
                (1.0 - fx) * fy,
                fx * fy,
            ];
            let mut value = [0.0f32; 3];
            for (&corner, &weight) in corners.iter().zip(&weights) {
                for (sum, sample) in value.iter_mut().zip(&analysis.low[corner]) {
                    *sum += sample * weight;
                }
            }
            (
                corrected_low_value(analysis, model, gain, source, value),
                true,
            )
        })
        .collect::<Vec<_>>();
    let valid = sampled.iter().map(|(_, valid)| *valid).collect::<Vec<_>>();
    // Normalised low-pass on the grid, then extend the band a little past the
    // tile edge so a sample near the border never reads zeros.
    let mask = valid
        .iter()
        .map(|&flag| f32::from(u8::from(flag)))
        .collect::<Vec<_>>();
    let blurred_mask = blur_plane(&mask, columns, rows, LOW_BAND_SIGMA);
    let mut values = vec![[0.0f32; 3]; columns * rows];
    let channels = (0..3)
        .map(|channel| {
            let plane = sampled
                .iter()
                .map(|(value, valid)| if *valid { value[channel] } else { 0.0 })
                .collect::<Vec<_>>();
            blur_plane(&plane, columns, rows, LOW_BAND_SIGMA)
        })
        .collect::<Vec<_>>();
    let mut known = vec![false; columns * rows];
    for index in 0..columns * rows {
        if valid[index] && blurred_mask[index] > 1e-3 {
            for channel in 0..3 {
                values[index][channel] = channels[channel][index] / blurred_mask[index];
            }
            known[index] = true;
        }
    }
    fill_unknown(&mut values, &mut known, columns, rows, 24);
    LowBand {
        x0,
        y0,
        columns,
        rows,
        values,
        valid,
    }
}

/// Chamfer distance of every valid band sample to the nearest invalid sample
/// or band edge, in grid pixels.
fn border_distance(band: &LowBand) -> Vec<f32> {
    let (columns, rows) = (band.columns, band.rows);
    let mut distance = band
        .valid
        .iter()
        .map(|&valid| if valid { f32::MAX } else { 0.0 })
        .collect::<Vec<_>>();
    for y in 0..rows {
        for x in 0..columns {
            let index = y * columns + x;
            if distance[index] == 0.0 {
                continue;
            }
            let edge = (x.min(y).min(columns - 1 - x).min(rows - 1 - y) + 1) as f32;
            let mut best = distance[index].min(edge);
            if x > 0 {
                best = best.min(distance[index - 1] + 1.0);
            }
            if y > 0 {
                best = best.min(distance[index - columns] + 1.0);
                if x > 0 {
                    best = best.min(distance[index - columns - 1] + 1.414);
                }
                if x + 1 < columns {
                    best = best.min(distance[index - columns + 1] + 1.414);
                }
            }
            distance[index] = best;
        }
    }
    for y in (0..rows).rev() {
        for x in (0..columns).rev() {
            let index = y * columns + x;
            if distance[index] == 0.0 {
                continue;
            }
            let mut best = distance[index];
            if x + 1 < columns {
                best = best.min(distance[index + 1] + 1.0);
            }
            if y + 1 < rows {
                best = best.min(distance[index + columns] + 1.0);
                if x + 1 < columns {
                    best = best.min(distance[index + columns + 1] + 1.414);
                }
                if x > 0 {
                    best = best.min(distance[index + columns - 1] + 1.414);
                }
            }
            distance[index] = best;
        }
    }
    distance
}

/// The low band every output pixel receives: a feathered mean of all tiles'
/// low bands that stay within [`CONSENSUS_TOLERANCE`] of their weighted
/// median. A glare field or reflection seen by one tile is outvoted, and
/// tile borders fade in over [`CONSENSUS_FEATHER`] instead of stepping.
fn consensus_low_band(bands: &[LowBand], canvas: &Canvas) -> LowBand {
    let (grid_width, grid_height) = low_grid_size(canvas);
    let count = grid_width * grid_height;
    let feathers = bands
        .par_iter()
        .map(|band| {
            border_distance(band)
                .into_iter()
                .map(|distance| {
                    let ramp = (distance / CONSENSUS_FEATHER).clamp(0.0, 1.0);
                    ramp * ramp + 1e-3
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let samples = (0..count)
        .into_par_iter()
        .map(|index| {
            let (x, y) = (index % grid_width, index / grid_width);
            let mut candidates = Vec::with_capacity(bands.len());
            for (band, feather) in bands.iter().zip(&feathers) {
                if band.valid_at(x, y) {
                    let local = (y - band.y0) * band.columns + (x - band.x0);
                    candidates.push((band.values[local], feather[local]));
                }
            }
            if candidates.is_empty() {
                return None;
            }
            let total = candidates.iter().map(|(_, weight)| weight).sum::<f32>();
            let median: [f32; 3] = std::array::from_fn(|channel| {
                let mut ordered = candidates
                    .iter()
                    .map(|(value, weight)| (value[channel], *weight))
                    .collect::<Vec<_>>();
                ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
                let mut accumulated = 0.0;
                for (value, weight) in &ordered {
                    accumulated += weight;
                    if accumulated >= total * 0.5 {
                        return *value;
                    }
                }
                ordered.last().map_or(0.0, |(value, _)| *value)
            });
            let mut sum = [0.0f32; 3];
            let mut weight_sum = 0.0f32;
            for (value, weight) in &candidates {
                let agrees = (0..3).all(|channel| {
                    (value[channel] - median[channel]).abs()
                        <= CONSENSUS_TOLERANCE * median[channel].max(0.05)
                });
                if agrees {
                    for channel in 0..3 {
                        sum[channel] += weight * value[channel];
                    }
                    weight_sum += weight;
                }
            }
            Some(if weight_sum > 1e-6 {
                sum.map(|value| value / weight_sum)
            } else {
                median
            })
        })
        .collect::<Vec<_>>();
    let valid = samples.iter().map(Option::is_some).collect::<Vec<_>>();
    let mask = valid
        .iter()
        .map(|&flag| f32::from(u8::from(flag)))
        .collect::<Vec<_>>();
    let blurred_mask = blur_plane(&mask, grid_width, grid_height, CONSENSUS_SIGMA);
    let channels = (0..3)
        .map(|channel| {
            let plane = samples
                .iter()
                .map(|sample| sample.map_or(0.0, |value| value[channel]))
                .collect::<Vec<_>>();
            blur_plane(&plane, grid_width, grid_height, CONSENSUS_SIGMA)
        })
        .collect::<Vec<_>>();
    let mut values = vec![[0.0f32; 3]; count];
    let mut known = vec![false; count];
    for index in 0..count {
        if valid[index] && blurred_mask[index] > 1e-3 {
            for channel in 0..3 {
                values[index][channel] = channels[channel][index] / blurred_mask[index];
            }
            known[index] = true;
        }
    }
    fill_unknown(&mut values, &mut known, grid_width, grid_height, 4);
    LowBand {
        x0: 0,
        y0: 0,
        columns: grid_width,
        rows: grid_height,
        values,
        valid: known,
    }
}

/// Extend known samples into unknown ones by averaging known neighbours.
fn fill_unknown(
    values: &mut [[f32; 3]],
    known: &mut [bool],
    columns: usize,
    rows: usize,
    iterations: usize,
) {
    for _ in 0..iterations {
        let snapshot = known.to_vec();
        let previous = values.to_vec();
        let mut changed = false;
        for index in 0..columns * rows {
            if snapshot[index] {
                continue;
            }
            let (x, y) = (index % columns, index / columns);
            let mut sum = [0.0f32; 3];
            let mut count = 0.0f32;
            for ny in y.saturating_sub(1)..(y + 2).min(rows) {
                for nx in x.saturating_sub(1)..(x + 2).min(columns) {
                    let neighbour = ny * columns + nx;
                    if snapshot[neighbour] {
                        for channel in 0..3 {
                            sum[channel] += previous[neighbour][channel];
                        }
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                values[index] = sum.map(|value| value / count);
                known[index] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Detail rendering
// ---------------------------------------------------------------------------

/// Validity eroded by two pixels so a bicubic footprint never reads a
/// transparent or out-of-frame sample.
fn eroded_validity(source: &ScanSource) -> Vec<bool> {
    let (width, height) = source.rgb.dimensions();
    let (w, h) = (width as usize, height as usize);
    let valid = validity_of(source);
    let radius = 2usize;
    let mut horizontal = vec![false; w * h];
    horizontal
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let line = &valid[y * w..(y + 1) * w];
            for (x, output) in row.iter_mut().enumerate() {
                *output = x >= radius
                    && x + radius < w
                    && line[x - radius..=x + radius].iter().all(|&value| value);
            }
        });
    let mut output = vec![false; w * h];
    output.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y < radius || y + radius >= h {
            return;
        }
        for (x, value) in row.iter_mut().enumerate() {
            *value = (y - radius..=y + radius).all(|sy| horizontal[sy * w + x]);
        }
    });
    output
}

#[inline]
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Catmull-Rom sample clamped to its 4×4 neighbourhood (no ringing), the
/// same contract as `get_high_quality_interpolated_pixel`.
#[inline]
fn sample_bicubic(data: &[f32], width: usize, height: usize, x: f64, y: f64) -> [f32; 3] {
    let xf = x.floor();
    let yf = y.floor();
    let ix = xf as isize;
    let iy = yf as isize;
    let wx = catmull_rom_weights((x - xf) as f32);
    let wy = catmull_rom_weights((y - yf) as f32);
    let mut output = [0.0f32; 3];
    let mut low = [f32::MAX; 3];
    let mut high = [f32::MIN; 3];
    for (j, weight_y) in wy.iter().enumerate() {
        let sy = (iy - 1 + j as isize).clamp(0, height as isize - 1) as usize;
        let line = sy * width;
        let mut row = [0.0f32; 3];
        for (i, weight_x) in wx.iter().enumerate() {
            let sx = (ix - 1 + i as isize).clamp(0, width as isize - 1) as usize;
            let start = (line + sx) * 3;
            for channel in 0..3 {
                let value = data[start + channel];
                row[channel] += weight_x * value;
                low[channel] = low[channel].min(value);
                high[channel] = high[channel].max(value);
            }
        }
        for channel in 0..3 {
            output[channel] += weight_y * row[channel];
        }
    }
    std::array::from_fn(|channel| output[channel].clamp(low[channel], high[channel]))
}

/// Weights of the previous, own and next cell along one axis for a pixel at
/// local position `local` (0..OWNER_CELL).
#[inline]
fn axis_transition(local: u32) -> [f32; 3] {
    let position = local as f32 + 0.5;
    let to_previous = position;
    let to_next = OWNER_CELL as f32 - position;
    let ramp = |distance: f32| (0.5 + distance / (2.0 * DETAIL_TRANSITION)).clamp(0.0, 1.0);
    if to_previous < DETAIL_TRANSITION {
        let own = ramp(to_previous);
        [1.0 - own, own, 0.0]
    } else if to_next < DETAIL_TRANSITION {
        let own = ramp(to_next);
        [0.0, own, 1.0 - own]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[allow(clippy::too_many_arguments)]
fn render_tile_detail(
    label: usize,
    rgb: &Rgb32FImage,
    valid: &[bool],
    warp: &TileWarp,
    band: &LowBand,
    ownership: &Ownership,
    canvas: &Canvas,
    detail: &mut [f32],
    weight: &mut [f32],
) {
    let owner = u16::try_from(label + 1).unwrap_or(u16::MAX);
    let (source_width, source_height) = (rgb.width() as usize, rgb.height() as usize);
    let data = rgb.as_raw();
    let width = canvas.width as usize;
    let height = canvas.height as usize;
    let cell = OWNER_CELL as usize;
    let rows_per_band = cell;
    detail
        .par_chunks_mut(width * 3 * rows_per_band)
        .zip(weight.par_chunks_mut(width * rows_per_band))
        .enumerate()
        .for_each(|(cell_row, (detail_band, weight_band))| {
            let band_rows = weight_band.len() / width;
            for column in 0..ownership.columns {
                let (c, r) = (column as isize, cell_row as isize);
                let mut near_owner = false;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if ownership.owner(c + dx, r + dy) == owner {
                            near_owner = true;
                        }
                    }
                }
                let fallback = ownership.fallback[cell_row * ownership.columns + column];
                if !near_owner && !fallback {
                    continue;
                }
                let x_start = column * cell;
                let x_end = ((column + 1) * cell).min(width);
                for local_y in 0..band_rows {
                    let y = cell_row * cell + local_y;
                    if y >= height {
                        break;
                    }
                    let wy = axis_transition(local_y as u32);
                    let world_y = y as f64 - canvas.offset_y;
                    let grid_y = (y as f64 - (LOW_GRID as f64 - 1.0) * 0.5) / LOW_GRID as f64;
                    for x in x_start..x_end {
                        let wx = axis_transition((x - x_start) as u32);
                        let mut own_weight = 0.0f32;
                        if near_owner {
                            for (j, weight_y) in wy.iter().enumerate() {
                                if *weight_y <= 0.0 {
                                    continue;
                                }
                                for (i, weight_x) in wx.iter().enumerate() {
                                    if *weight_x > 0.0
                                        && ownership.owner(c + i as isize - 1, r + j as isize - 1)
                                            == owner
                                    {
                                        own_weight += weight_x * weight_y;
                                    }
                                }
                            }
                        }
                        let pixel = local_y * width + x;
                        let current = weight_band[pixel];
                        if own_weight <= 0.0 && !(fallback && current == 0.0) {
                            continue;
                        }
                        let world_x = x as f64 - canvas.offset_x;
                        let Some(source) = warp.source(Point2::new(world_x, world_y)) else {
                            continue;
                        };
                        let (sx, sy) = (source.x, source.y);
                        if sx < 0.0
                            || sy < 0.0
                            || sx > (source_width - 1) as f64
                            || sy > (source_height - 1) as f64
                        {
                            continue;
                        }
                        let nearest = (sy.round() as usize).min(source_height - 1) * source_width
                            + (sx.round() as usize).min(source_width - 1);
                        if !valid[nearest] {
                            continue;
                        }
                        let value = sample_bicubic(data, source_width, source_height, sx, sy);
                        let start = pixel * 3;
                        if own_weight > 0.0 {
                            if current < 0.0 {
                                detail_band[start..start + 3].fill(0.0);
                                weight_band[pixel] = 0.0;
                            }
                            let grid_x =
                                (x as f64 - (LOW_GRID as f64 - 1.0) * 0.5) / LOW_GRID as f64;
                            let low = band.sample(grid_x, grid_y);
                            for channel in 0..3 {
                                detail_band[start + channel] +=
                                    own_weight * (value[channel] - low[channel]);
                            }
                            weight_band[pixel] += own_weight;
                        } else {
                            // First valid non-owner sample of a pixel no owner
                            // can serve; replaced as soon as an owner writes.
                            detail_band[start..start + 3].copy_from_slice(&value);
                            weight_band[pixel] = -1.0;
                        }
                    }
                }
            }
        });
}
