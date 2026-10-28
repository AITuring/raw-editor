//! Test-only acceptance metrics for real focus-stack regression runs.
//!
//! Production rendering never reads these measurements or a reference image.
//! The ignored acceptance harness measures a finished output only: whether
//! every input reached the stitching order, contrast-invariant tile sharpness
//! and, when a reference rendering is supplied, the per-tile sharpness ratio
//! after registering the output onto that reference. The reference provides
//! measurement windows only; no reference pixel or transform reaches an output.

use std::path::Path;

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, ImageBuffer, ImageReader, Luma};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::panorama_utils::processing;

type LumaF32 = ImageBuffer<Luma<f32>, Vec<f32>>;

/// Tile edge in analysis pixels.
const TILE_PX: u32 = 64;
/// Minimum luma standard deviation (0..1) for a tile to count as content.
const CONTENT_MIN_STD: f32 = 0.04;
/// Variance floor of the contrast-normalised sharpness; keeps nearly flat
/// windows from turning sensor noise into a large score.
const SHARPNESS_VARIANCE_FLOOR: f64 = 0.02 * 0.02;
/// Minimum covered fraction for a tile when the output carries a Coverage_Mask.
const TILE_MIN_COVERED_FRACTION: f64 = 0.99;
const DEFAULT_ANALYSIS_LONG_SIDE: u32 = 4096;
const REGISTRATION_LONG_SIDE: u32 = 2048;
const REGISTRATION_MAX_FEATURES: usize = 4000;
const REGISTRATION_SCALES: [f32; 2] = [0.8, 0.64];
const REGISTRATION_RATIO: f32 = 0.85;
const REGISTRATION_INLIER_PX: f64 = 3.0;
const REGISTRATION_MIN_INLIERS: usize = 40;
const WORST_TILE_COUNT: usize = 12;

/// What the acceptance harness hands to the metric pass.
pub(super) struct AcceptanceMetricsRequest<'a> {
    pub(super) image: &'a DynamicImage,
    pub(super) coverage: Option<&'a GrayImage>,
    pub(super) input_paths: &'a [String],
    pub(super) ordered_paths: &'a [String],
    pub(super) reference: Option<&'a Path>,
    pub(super) analysis_long_side: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
struct RegionStats {
    std: f64,
    sharpness: f64,
}

#[derive(Clone, Copy, Debug)]
struct TileRect {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

#[derive(Clone, Copy, Debug)]
struct ReferenceTile {
    rect: TileRect,
    ratio: f64,
    contrast: f64,
}

struct Registration {
    /// Maps output pixels to reference pixels (both at native resolution).
    output_to_reference: Matrix3<f64>,
    matches: usize,
    inliers: usize,
    local_scale: f64,
}

/// Compute the metrics and write them as pretty JSON to `path`.
pub(super) fn write_acceptance_metrics(
    request: &AcceptanceMetricsRequest<'_>,
    path: &Path,
) -> Result<Value, String> {
    let metrics = acceptance_metrics(request)?;
    let encoded = serde_json::to_vec_pretty(&metrics)
        .map_err(|error| format!("encode acceptance metrics: {error}"))?;
    std::fs::write(path, encoded)
        .map_err(|error| format!("write acceptance metrics {}: {error}", path.display()))?;
    Ok(metrics)
}

pub(super) fn acceptance_metrics(request: &AcceptanceMetricsRequest<'_>) -> Result<Value, String> {
    let (width, height) = (request.image.width(), request.image.height());
    if width < TILE_PX || height < TILE_PX {
        return Err(format!("output {width}x{height} is smaller than one tile"));
    }
    let long_side = request
        .analysis_long_side
        .unwrap_or(DEFAULT_ANALYSIS_LONG_SIDE)
        .max(TILE_PX * 4);
    let (analysis_width, analysis_height) = fit_long_side(width, height, long_side);
    let analysis_scale = f64::from(analysis_width) / f64::from(width);
    let output_luma = luma16_to_f32(&resize_luma16(
        &request.image.to_luma16(),
        analysis_width,
        analysis_height,
    ));
    let coverage = request
        .coverage
        .filter(|mask| mask.dimensions() == (width, height))
        .map(|mask| {
            image::imageops::resize(mask, analysis_width, analysis_height, FilterType::Nearest)
        });
    let grid = tile_grid(analysis_width, analysis_height, coverage.as_ref());
    let self_tiles = grid
        .par_iter()
        .map(|&rect| region_stats(&output_luma, rect).map(|stats| (rect, stats)))
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .filter(|(_, stats)| stats.std >= f64::from(CONTENT_MIN_STD))
        .collect::<Vec<_>>();

    let reference = match request.reference {
        Some(path) => Some(reference_metrics(
            path,
            &output_luma,
            analysis_scale,
            &grid,
        )?),
        None => None,
    };

    Ok(json!({
        "schema": 1,
        "output": { "width": width, "height": height },
        "inputs": input_order_summary(request.input_paths, request.ordered_paths),
        "analysis": {
            "width": analysis_width,
            "height": analysis_height,
            "scale": analysis_scale,
            "tile_px": TILE_PX,
            "content_min_std": CONTENT_MIN_STD,
            "grid_tiles": grid.len(),
        },
        "self": self_summary(&self_tiles, analysis_width),
        "reference": reference,
    }))
}

fn input_order_summary(input_paths: &[String], ordered_paths: &[String]) -> Value {
    let file_name = |path: &String| {
        Path::new(path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone())
    };
    let ordered = ordered_paths.iter().map(file_name).collect::<Vec<_>>();
    let missing = input_paths
        .iter()
        .map(file_name)
        .filter(|name| !ordered.contains(name))
        .collect::<Vec<_>>();
    json!({
        "count": input_paths.len(),
        "ordered": ordered_paths.len(),
        "missing_from_order": missing,
    })
}

fn self_summary(tiles: &[(TileRect, RegionStats)], analysis_width: u32) -> Value {
    let sharpness = tiles
        .iter()
        .map(|(_, stats)| stats.sharpness)
        .collect::<Vec<_>>();
    let thirds = third_bands(analysis_width)
        .into_iter()
        .map(|(name, left, right)| {
            let band = tiles
                .iter()
                .filter(|(rect, _)| (left..right).contains(&rect_center_x(rect)))
                .map(|(_, stats)| stats.sharpness)
                .collect::<Vec<_>>();
            json!({
                "band": name,
                "tiles": band.len(),
                "sharpness_median": percentile(&band, 0.5),
                "sharpness_p10": percentile(&band, 0.1),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "content_tiles": tiles.len(),
        "sharpness": distribution(&sharpness),
        "thirds": thirds,
    })
}

fn reference_metrics(
    path: &Path,
    output_luma: &LumaF32,
    output_analysis_scale: f64,
    grid: &[TileRect],
) -> Result<Value, String> {
    let reference = decode_reference_luma(path)?;
    let (reference_width, reference_height) = reference.dimensions();
    let registration =
        register_output_to_reference(output_luma, output_analysis_scale, &reference)?;

    // Bring the reference to the output analysis content scale: one output
    // analysis pixel covers `local_scale / output_analysis_scale` reference
    // pixels, so both sides measure the same physical window size.
    let reference_scale = (output_analysis_scale / registration.local_scale).min(1.0);
    let reference_analysis_width =
        ((f64::from(reference_width) * reference_scale).round() as u32).max(1);
    let reference_analysis_height =
        ((f64::from(reference_height) * reference_scale).round() as u32).max(1);
    let reference_luma = luma8_to_f32(&image::imageops::resize(
        &reference,
        reference_analysis_width,
        reference_analysis_height,
        FilterType::Triangle,
    ));
    drop(reference);
    let analysis_to_reference = scale_matrix(
        f64::from(reference_analysis_width) / f64::from(reference_width),
        f64::from(reference_analysis_height) / f64::from(reference_height),
    ) * registration.output_to_reference
        * scale_matrix(1.0 / output_analysis_scale, 1.0 / output_analysis_scale);

    let mapped = grid
        .par_iter()
        .map(|&rect| {
            let target = map_rect(&analysis_to_reference, rect, reference_luma.dimensions())?;
            let output = region_stats(output_luma, rect)?;
            let reference = region_stats(&reference_luma, target)?;
            Some((rect, output, reference))
        })
        .collect::<Vec<_>>();
    let mapped_count = mapped.iter().flatten().count();
    let tiles = mapped
        .into_iter()
        .flatten()
        .filter(|(_, _, reference)| reference.std >= f64::from(CONTENT_MIN_STD))
        .map(|(rect, output, reference)| ReferenceTile {
            rect,
            ratio: output.sharpness / reference.sharpness.max(f64::EPSILON),
            contrast: output.std / reference.std.max(f64::EPSILON),
        })
        .collect::<Vec<_>>();
    let ratios = tiles.iter().map(|tile| tile.ratio).collect::<Vec<_>>();
    let contrasts = tiles.iter().map(|tile| tile.contrast).collect::<Vec<_>>();
    let median = percentile(&ratios, 0.5).unwrap_or(0.0);
    let fraction_below = |values: &[f64], limit: f64| {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().filter(|&&value| value < limit).count() as f64 / values.len() as f64
    };
    let normalized = ratios
        .iter()
        .map(|ratio| ratio / median.max(f64::EPSILON))
        .collect::<Vec<_>>();
    let analysis_width = output_luma.width();
    let thirds = third_bands(analysis_width)
        .into_iter()
        .map(|(name, left, right)| {
            let band = tiles
                .iter()
                .filter(|tile| (left..right).contains(&rect_center_x(&tile.rect)))
                .map(|tile| tile.ratio)
                .collect::<Vec<_>>();
            json!({
                "band": name,
                "tiles": band.len(),
                "ratio_median": percentile(&band, 0.5),
                "ratio_p10": percentile(&band, 0.1),
            })
        })
        .collect::<Vec<_>>();
    let mut worst = tiles.clone();
    worst.sort_by(|left, right| {
        left.ratio
            .total_cmp(&right.ratio)
            .then_with(|| left.rect.y0.cmp(&right.rect.y0))
            .then_with(|| left.rect.x0.cmp(&right.rect.x0))
    });
    let worst = worst
        .iter()
        .take(WORST_TILE_COUNT)
        .map(|tile| {
            json!({
                "x": tile.rect.x0,
                "y": tile.rect.y0,
                "output_x": (f64::from(tile.rect.x0) / output_analysis_scale).round(),
                "output_y": (f64::from(tile.rect.y0) / output_analysis_scale).round(),
                "ratio": tile.ratio,
                "normalized": tile.ratio / median.max(f64::EPSILON),
                "contrast": tile.contrast,
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "file": path.file_name().map(|name| name.to_string_lossy().into_owned()),
        "width": reference_width,
        "height": reference_height,
        "registration": {
            "matches": registration.matches,
            "inliers": registration.inliers,
            "local_scale": registration.local_scale,
            "output_to_reference": registration
                .output_to_reference
                .transpose()
                .as_slice()
                .to_vec(),
        },
        "analysis": {
            "width": reference_analysis_width,
            "height": reference_analysis_height,
            "scale": reference_scale,
        },
        "tiles": {
            "grid": grid.len(),
            "mapped": mapped_count,
            "content": tiles.len(),
        },
        "ratio": distribution(&ratios),
        "normalized_below_0_5": fraction_below(&normalized, 0.5),
        "normalized_below_0_7": fraction_below(&normalized, 0.7),
        "contrast": distribution(&contrasts),
        "contrast_below_0_5": fraction_below(&contrasts, 0.5),
        "thirds": thirds,
        "worst_tiles": worst,
    }))
}

fn decode_reference_luma(path: &Path) -> Result<GrayImage, String> {
    let mut reader = ImageReader::open(path)
        .map_err(|error| format!("open reference {}: {error}", path.display()))?
        .with_guessed_format()
        .map_err(|error| format!("probe reference {}: {error}", path.display()))?;
    // A Photoshop reference of a long scroll easily exceeds the decoder's
    // default allocation limit; this is a read-only, test-only measurement.
    reader.no_limits();
    let decoded = reader
        .decode()
        .map_err(|error| format!("decode reference {}: {error}", path.display()))?;
    Ok(decoded.to_luma8())
}

fn register_output_to_reference(
    output_luma: &LumaF32,
    output_analysis_scale: f64,
    reference: &GrayImage,
) -> Result<Registration, String> {
    let (output_width, output_height) = output_luma.dimensions();
    let (output_reg_width, output_reg_height) =
        fit_long_side(output_width, output_height, REGISTRATION_LONG_SIDE);
    let output_reg = f32_to_luma8(&image::imageops::resize(
        output_luma,
        output_reg_width,
        output_reg_height,
        FilterType::Triangle,
    ));
    let (reference_width, reference_height) = reference.dimensions();
    let (reference_reg_width, reference_reg_height) =
        fit_long_side(reference_width, reference_height, REGISTRATION_LONG_SIDE);
    let reference_reg = image::imageops::resize(
        reference,
        reference_reg_width,
        reference_reg_height,
        FilterType::Triangle,
    );
    let pairs = processing::generate_brief_pairs();
    let output_features = processing::find_features_multiscale(
        &output_reg,
        &pairs,
        REGISTRATION_MAX_FEATURES,
        &REGISTRATION_SCALES,
    );
    let reference_features = processing::find_features_multiscale(
        &reference_reg,
        &pairs,
        REGISTRATION_MAX_FEATURES,
        &REGISTRATION_SCALES,
    );
    let matches = processing::match_features_with_ratio(
        &output_features,
        &reference_features,
        REGISTRATION_RATIO,
    );
    let points = matches
        .iter()
        .map(|candidate| {
            let source = output_features[candidate.index1].keypoint;
            let target = reference_features[candidate.index2].keypoint;
            (
                Point2::new(f64::from(source.x), f64::from(source.y)),
                Point2::new(f64::from(target.x), f64::from(target.y)),
            )
        })
        .collect::<Vec<_>>();
    let (ransac, inlier_indices) =
        processing::find_homography_ransac_points_stable_with_min_inliers(
            &points,
            REGISTRATION_INLIER_PX,
            REGISTRATION_MIN_INLIERS,
        )
        .ok_or_else(|| {
            format!(
                "reference registration failed: {} descriptor matches, fewer than {} inliers",
                points.len(),
                REGISTRATION_MIN_INLIERS
            )
        })?;
    let inlier_points = inlier_indices
        .iter()
        .map(|&index| points[index])
        .collect::<Vec<_>>();
    let registration_homography = processing::compute_homography(&inlier_points).unwrap_or(ransac);
    // registration pixels -> native pixels on both sides.
    let output_reg_to_analysis = scale_matrix(
        f64::from(output_width) / f64::from(output_reg_width),
        f64::from(output_height) / f64::from(output_reg_height),
    );
    let reference_reg_to_native = scale_matrix(
        f64::from(reference_width) / f64::from(reference_reg_width),
        f64::from(reference_height) / f64::from(reference_reg_height),
    );
    let output_native_to_analysis = scale_matrix(output_analysis_scale, output_analysis_scale);
    let output_reg_inverse = output_reg_to_analysis
        .try_inverse()
        .ok_or_else(|| "registration scale is singular".to_string())?;
    let output_to_reference = reference_reg_to_native
        * registration_homography
        * output_reg_inverse
        * output_native_to_analysis;
    let center = Point2::new(
        f64::from(output_width) / output_analysis_scale * 0.5,
        f64::from(output_height) / output_analysis_scale * 0.5,
    );
    let local_scale = local_scale_at(&output_to_reference, center)
        .filter(|scale| scale.is_finite() && *scale > 1e-6)
        .ok_or_else(|| "reference registration has a degenerate local scale".to_string())?;
    Ok(Registration {
        output_to_reference,
        matches: points.len(),
        inliers: inlier_indices.len(),
        local_scale,
    })
}

fn local_scale_at(transform: &Matrix3<f64>, point: Point2<f64>) -> Option<f64> {
    let step = 8.0;
    let origin = project(transform, point)?;
    let along_x = project(transform, Point2::new(point.x + step, point.y))?;
    let along_y = project(transform, Point2::new(point.x, point.y + step))?;
    let dx = (along_x - origin) / step;
    let dy = (along_y - origin) / step;
    Some((dx.x * dy.y - dx.y * dy.x).abs().sqrt())
}

fn project(transform: &Matrix3<f64>, point: Point2<f64>) -> Option<Point2<f64>> {
    let projected = transform * Point3::new(point.x, point.y, 1.0);
    (projected.z.abs() > 1e-12)
        .then(|| Point2::new(projected.x / projected.z, projected.y / projected.z))
}

fn map_rect(transform: &Matrix3<f64>, rect: TileRect, bounds: (u32, u32)) -> Option<TileRect> {
    let corners = [
        Point2::new(f64::from(rect.x0), f64::from(rect.y0)),
        Point2::new(f64::from(rect.x1), f64::from(rect.y0)),
        Point2::new(f64::from(rect.x1), f64::from(rect.y1)),
        Point2::new(f64::from(rect.x0), f64::from(rect.y1)),
    ]
    .into_iter()
    .map(|corner| project(transform, corner))
    .collect::<Option<Vec<_>>>()?;
    let min_x = corners
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min)
        .floor();
    let max_x = corners
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil();
    let min_y = corners
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min)
        .floor();
    let max_y = corners
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil();
    if min_x < 1.0
        || min_y < 1.0
        || max_x > f64::from(bounds.0) - 1.0
        || max_y > f64::from(bounds.1) - 1.0
    {
        return None;
    }
    let area = (max_x - min_x) * (max_y - min_y);
    let tile_area = f64::from(rect.x1 - rect.x0) * f64::from(rect.y1 - rect.y0);
    if !(0.5 * tile_area..=2.0 * tile_area).contains(&area) {
        return None;
    }
    Some(TileRect {
        x0: min_x as u32,
        y0: min_y as u32,
        x1: max_x as u32,
        y1: max_y as u32,
    })
}

fn tile_grid(width: u32, height: u32, coverage: Option<&GrayImage>) -> Vec<TileRect> {
    let mut tiles = Vec::new();
    for row in 0..height / TILE_PX {
        for column in 0..width / TILE_PX {
            let rect = TileRect {
                x0: column * TILE_PX,
                y0: row * TILE_PX,
                x1: (column + 1) * TILE_PX,
                y1: (row + 1) * TILE_PX,
            };
            if coverage.is_none_or(|mask| covered_fraction(mask, rect) >= TILE_MIN_COVERED_FRACTION)
            {
                tiles.push(rect);
            }
        }
    }
    tiles
}

fn covered_fraction(mask: &GrayImage, rect: TileRect) -> f64 {
    let mut covered = 0u64;
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            covered += u64::from(mask.get_pixel(x, y)[0] >= 128);
        }
    }
    covered as f64 / (f64::from(rect.x1 - rect.x0) * f64::from(rect.y1 - rect.y0))
}

/// Luma spread and contrast-normalised gradient energy of one window.
///
/// `mean(|∇I|²) / var(I)` does not change when the window's contrast is scaled,
/// so tone or exposure differences between two renderings cancel, while a
/// blurred rendering of the same content loses gradient energy.
fn region_stats(image: &LumaF32, rect: TileRect) -> Option<RegionStats> {
    let (width, height) = image.dimensions();
    if rect.x1 > width || rect.y1 > height || rect.x1 < rect.x0 + 3 || rect.y1 < rect.y0 + 3 {
        return None;
    }
    let value = |x: u32, y: u32| f64::from(image.get_pixel(x, y)[0]);
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut gradient = 0.0;
    let mut gradient_count = 0u64;
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            let current = value(x, y);
            sum += current;
            sum_sq += current * current;
            if x > rect.x0 && x + 1 < rect.x1 && y > rect.y0 && y + 1 < rect.y1 {
                let gx = (value(x + 1, y) - value(x - 1, y)) * 0.5;
                let gy = (value(x, y + 1) - value(x, y - 1)) * 0.5;
                gradient += gx * gx + gy * gy;
                gradient_count += 1;
            }
        }
    }
    let count = f64::from(rect.x1 - rect.x0) * f64::from(rect.y1 - rect.y0);
    let mean = sum / count;
    let variance = (sum_sq / count - mean * mean).max(0.0);
    let gradient_energy = gradient / gradient_count.max(1) as f64;
    Some(RegionStats {
        std: variance.sqrt(),
        sharpness: gradient_energy / variance.max(SHARPNESS_VARIANCE_FLOOR),
    })
}

fn distribution(values: &[f64]) -> Value {
    json!({
        "count": values.len(),
        "min": percentile(values, 0.0),
        "p05": percentile(values, 0.05),
        "p10": percentile(values, 0.1),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
    })
}

fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    Some(sorted[index])
}

fn third_bands(width: u32) -> [(&'static str, u32, u32); 3] {
    let third = width / 3;
    [
        ("left", 0, third),
        ("center", third, width - third),
        ("right", width - third, width + 1),
    ]
}

fn rect_center_x(rect: &TileRect) -> u32 {
    (rect.x0 + rect.x1) / 2
}

fn fit_long_side(width: u32, height: u32, long_side: u32) -> (u32, u32) {
    let current = width.max(height);
    if current <= long_side {
        return (width, height);
    }
    let scale = f64::from(long_side) / f64::from(current);
    (
        ((f64::from(width) * scale).round() as u32).max(1),
        ((f64::from(height) * scale).round() as u32).max(1),
    )
}

fn scale_matrix(x: f64, y: f64) -> Matrix3<f64> {
    Matrix3::new(x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 1.0)
}

fn resize_luma16(
    image: &ImageBuffer<Luma<u16>, Vec<u16>>,
    width: u32,
    height: u32,
) -> ImageBuffer<Luma<u16>, Vec<u16>> {
    if image.dimensions() == (width, height) {
        return image.clone();
    }
    image::imageops::resize(image, width, height, FilterType::Triangle)
}

fn luma16_to_f32(image: &ImageBuffer<Luma<u16>, Vec<u16>>) -> LumaF32 {
    let (width, height) = image.dimensions();
    let data = image
        .as_raw()
        .par_iter()
        .map(|&value| f32::from(value) / 65_535.0)
        .collect::<Vec<_>>();
    LumaF32::from_raw(width, height, data).expect("luma buffer matches its dimensions")
}

fn luma8_to_f32(image: &GrayImage) -> LumaF32 {
    let (width, height) = image.dimensions();
    let data = image
        .as_raw()
        .par_iter()
        .map(|&value| f32::from(value) / 255.0)
        .collect::<Vec<_>>();
    LumaF32::from_raw(width, height, data).expect("luma buffer matches its dimensions")
}

fn f32_to_luma8(image: &LumaF32) -> GrayImage {
    let (width, height) = image.dimensions();
    let data = image
        .as_raw()
        .par_iter()
        .map(|&value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect::<Vec<_>>();
    GrayImage::from_raw(width, height, data).expect("gray buffer matches its dimensions")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn striped(width: u32, height: u32, period: u32) -> LumaF32 {
        LumaF32::from_fn(width, height, |x, _| {
            Luma([if (x / period) % 2 == 0 { 0.2 } else { 0.8 }])
        })
    }

    #[test]
    fn sharpness_is_contrast_invariant_and_drops_with_blur() {
        // Eight-pixel bars behave like brush-stroke edges: a blur widens every
        // edge while the window's contrast stays nearly unchanged.
        let sharp = striped(64, 64, 8);
        let low_contrast = LumaF32::from_fn(64, 64, |x, y| {
            Luma([0.5 + (sharp.get_pixel(x, y)[0] - 0.5) * 0.25])
        });
        let blurred = image::imageops::blur(&sharp, 2.0);
        let rect = TileRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 64,
        };
        let sharp_stats = region_stats(&sharp, rect).unwrap();
        let low_stats = region_stats(&low_contrast, rect).unwrap();
        let blurred_stats = region_stats(&blurred, rect).unwrap();
        assert!((sharp_stats.sharpness - low_stats.sharpness).abs() < 1e-3 * sharp_stats.sharpness);
        // A two-pixel Gaussian halves the score while contrast stays > 70 %.
        assert!(blurred_stats.sharpness < sharp_stats.sharpness * 0.6);
        assert!(blurred_stats.std > sharp_stats.std * 0.7);
    }

    #[test]
    fn tile_mapping_rejects_windows_outside_the_reference() {
        let rect = TileRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 64,
        };
        let shift = Matrix3::new(1.0, 0.0, 10.0, 0.0, 1.0, 10.0, 0.0, 0.0, 1.0);
        let mapped = map_rect(&shift, rect, (200, 200)).unwrap();
        assert_eq!(
            (mapped.x0, mapped.y0, mapped.x1, mapped.y1),
            (10, 10, 74, 74)
        );
        assert!(map_rect(&Matrix3::identity(), rect, (200, 200)).is_none());
    }

    #[test]
    fn input_summary_reports_sources_missing_from_the_order() {
        let inputs = vec!["/a/one.jpg".to_string(), "/a/two.jpg".to_string()];
        let ordered = vec!["/a/two.jpg".to_string()];
        let summary = input_order_summary(&inputs, &ordered);
        assert_eq!(summary["count"], 2);
        assert_eq!(summary["ordered"], 1);
        assert_eq!(summary["missing_from_order"], json!(["one.jpg"]));
    }
}
