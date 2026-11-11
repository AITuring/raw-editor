//! Soft regions of a finished focus scan, reported to the user.
//!
//! After the detail repair, every cell of the repair grid is compared with the
//! detail the canvas reaches around it. A cell well below its surroundings
//! looks soft to a viewer; such cells are grouped into regions, and each
//! region says whether some source was still clearly sharper there (the
//! program could not align it) or every source is soft (the photographs
//! themselves lack the detail). The regions are drawn on an overview so the
//! user can re-shoot or inspect them.

use image::Rgb32FImage;
use image::RgbImage;
use rayon::prelude::*;
use std::collections::HashSet;
use std::collections::VecDeque;

use super::{DetailGrid, REPAIR_DEFICIT_RATIO, RepairOutcome, neighbourhood_percentiles};

/// A cell is soft below this share of its neighbourhood's detail.
const SOFT_SHARE: f64 = 0.55;
/// Neighbours count as the same material here when their mean luma differs
/// by at most this factor; tighter than the repair's, since a border strip or
/// a title label beside the silk is another material, not a soft spot.
const REPORT_LEVEL_SIMILARITY: f64 = 1.25;
/// A cell with more than this share clipped to white has no detail to judge.
const MAX_CLIPPED_SHARE: f64 = 0.05;
/// A cell is judged only where its like neighbours carry an even texture: the
/// median of their detail at least this share of their 90th percentile. Silk
/// weave is even (about 0.75 on 祁序1); paper with strokes is not (blank paper
/// between characters of 兰亭 measured a tenth of the strokes beside it, and
/// was circled as soft all over). A sharper source of the same cell is
/// evidence of its own...
const MIN_TEXTURE_EVENNESS: f64 = 0.6;
/// ...without one the material has to be more even still: a smooth stretch
/// of 兰亭's paper beside its seals passed 0.6.
const MIN_TEXTURE_EVENNESS_SOURCES_SOFT: f64 = 0.75;
/// Soft cells this many cells apart belong to one region.
const REGION_LINK_CELLS: i64 = 2;
/// Regions of fewer cells are not reported.
const MIN_REGION_CELLS: usize = 3;
/// Long side of the overview the regions are drawn on.
const OVERVIEW_LONG_SIDE: u32 = 2400;

/// Why a reported region is soft.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SoftKind {
    /// Every source is soft here: the photographs lack the detail.
    SourcesSoft,
    /// A source was clearly sharper but could not be aligned to the canvas.
    SharperSourceUnused,
}

/// One reported region, in output pixels.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(super) struct SoftRegion {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) cells: usize,
    pub(super) kind: SoftKind,
    /// Median over the region of the canvas detail relative to its
    /// neighbourhood.
    pub(super) relative_detail: f64,
    /// Median over the region of the sharpest source's detail relative to the
    /// canvas.
    pub(super) best_source_gain: f64,
}

/// Soft regions of the canvas `crop` (canvas `x, y, width, height`) after the
/// repair.
pub(super) fn soft_regions(
    grid: &DetailGrid,
    outcome: &RepairOutcome,
    crop: (u32, u32, u32, u32),
) -> Vec<SoftRegion> {
    let (columns, rows) = grid.cell_dimensions();
    let detail = outcome
        .detail_after
        .iter()
        .zip(&outcome.clipped)
        .map(|(&value, &clipped)| {
            if clipped <= MAX_CLIPPED_SHARE {
                value
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    // The 75th percentile is the reference; the median and the 90th tell
    // whether the material is even enough to compare with.
    let neighbourhood = neighbourhood_percentiles(
        &detail,
        &outcome.level,
        grid,
        REPORT_LEVEL_SIMILARITY,
        &[0.75, 0.5, 0.9],
    );
    let reference = neighbourhood
        .iter()
        .enumerate()
        .map(|(cell, values)| {
            let sharper_source = detail[cell] < REPAIR_DEFICIT_RATIO * outcome.best[cell].0;
            let evenness = if sharper_source {
                MIN_TEXTURE_EVENNESS
            } else {
                MIN_TEXTURE_EVENNESS_SOURCES_SOFT
            };
            match values.as_deref() {
                Some(&[reference, median, high]) if median >= evenness * high => reference,
                _ => f64::NAN,
            }
        })
        .collect::<Vec<_>>();
    let soft = (0..columns * rows)
        .filter(|&cell| {
            detail[cell].is_finite()
                && reference[cell].is_finite()
                && detail[cell] < SOFT_SHARE * reference[cell]
        })
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    let mut regions = Vec::new();
    let mut cells_sorted = soft.iter().copied().collect::<Vec<_>>();
    cells_sorted.sort_unstable();
    for start in cells_sorted {
        if !seen.insert(start) {
            continue;
        }
        let mut group = Vec::new();
        let mut pending = VecDeque::from([start]);
        while let Some(cell) = pending.pop_front() {
            group.push(cell);
            let (x, y) = ((cell % columns) as i64, (cell / columns) as i64);
            for dy in -REGION_LINK_CELLS..=REGION_LINK_CELLS {
                for dx in -REGION_LINK_CELLS..=REGION_LINK_CELLS {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= columns as i64 || ny >= rows as i64 {
                        continue;
                    }
                    let neighbour = ny as usize * columns + nx as usize;
                    if soft.contains(&neighbour) && seen.insert(neighbour) {
                        pending.push_back(neighbour);
                    }
                }
            }
        }
        if group.len() < MIN_REGION_CELLS {
            continue;
        }
        let median = |mut values: Vec<f64>| {
            values.retain(|value| value.is_finite());
            if values.is_empty() {
                return f64::NAN;
            }
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        };
        let relative_detail = median(
            group
                .iter()
                .map(|&cell| detail[cell] / reference[cell])
                .collect(),
        );
        let best_source_gain = median(
            group
                .iter()
                .map(|&cell| outcome.best[cell].0 / detail[cell])
                .collect(),
        );
        let unused = group
            .iter()
            .filter(|&&cell| detail[cell] < REPAIR_DEFICIT_RATIO * outcome.best[cell].0)
            .count();
        let cell_size = grid.cell_size();
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for &cell in &group {
            x0 = x0.min(cell % columns);
            x1 = x1.max(cell % columns + 1);
            y0 = y0.min(cell / columns);
            y1 = y1.max(cell / columns + 1);
        }
        let to_output = |cells: usize, origin: u32, extent: u32| {
            ((cells as f64 * cell_size) as u32)
                .saturating_sub(origin)
                .min(extent)
        };
        let (left, top) = (to_output(x0, crop.0, crop.2), to_output(y0, crop.1, crop.3));
        let (right, bottom) = (to_output(x1, crop.0, crop.2), to_output(y1, crop.1, crop.3));
        if right <= left || bottom <= top {
            continue;
        }
        regions.push(SoftRegion {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
            cells: group.len(),
            kind: if unused * 2 >= group.len() {
                SoftKind::SharperSourceUnused
            } else {
                SoftKind::SourcesSoft
            },
            relative_detail,
            best_source_gain,
        });
    }
    regions.sort_by(|a, b| {
        b.cells
            .cmp(&a.cells)
            .then(a.y.cmp(&b.y))
            .then(a.x.cmp(&b.x))
    });
    regions
}

/// An overview of `output` with every region circled: red where the sources
/// are soft, orange where a sharper source could not be aligned.
pub(super) fn soft_region_overview(output: &Rgb32FImage, regions: &[SoftRegion]) -> RgbImage {
    let factor = (output
        .width()
        .max(output.height())
        .div_ceil(OVERVIEW_LONG_SIDE))
    .max(1);
    let (width, height) = (
        output.width().div_ceil(factor),
        output.height().div_ceil(factor),
    );
    let mut overview = RgbImage::new(width, height);
    let rows = (0..height)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(width as usize * 3);
            for x in 0..width {
                let (mut sum, mut count) = ([0.0f32; 3], 0.0f32);
                for sy in y * factor..((y + 1) * factor).min(output.height()) {
                    for sx in x * factor..((x + 1) * factor).min(output.width()) {
                        let pixel = output.get_pixel(sx, sy);
                        for channel in 0..3 {
                            sum[channel] += pixel[channel];
                        }
                        count += 1.0;
                    }
                }
                for value in sum {
                    // The canvas holds display-referred values, as the
                    // previews do.
                    let encoded = (value / count.max(1.0)).clamp(0.0, 1.0);
                    row.push((encoded * 255.0).round() as u8);
                }
            }
            row
        })
        .collect::<Vec<_>>();
    for (y, row) in rows.into_iter().enumerate() {
        for x in 0..width as usize {
            overview.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([row[3 * x], row[3 * x + 1], row[3 * x + 2]]),
            );
        }
    }
    for region in regions {
        let colour = match region.kind {
            SoftKind::SourcesSoft => [255, 30, 30],
            SoftKind::SharperSourceUnused => [255, 160, 0],
        };
        let centre_x = (f64::from(region.x) + f64::from(region.width) * 0.5) / f64::from(factor);
        let centre_y = (f64::from(region.y) + f64::from(region.height) * 0.5) / f64::from(factor);
        // An ellipse around the box, at least 12 overview pixels across.
        let radius_x = (f64::from(region.width) / f64::from(factor) * 0.75).max(12.0);
        let radius_y = (f64::from(region.height) / f64::from(factor) * 0.75).max(12.0);
        let steps = ((radius_x + radius_y) * 4.0) as usize + 32;
        for step in 0..steps {
            let angle = step as f64 / steps as f64 * std::f64::consts::TAU;
            for thickness in [-1.5, -0.5, 0.5, 1.5] {
                let x = centre_x + (radius_x + thickness) * angle.cos();
                let y = centre_y + (radius_y + thickness) * angle.sin();
                if x >= 0.0 && y >= 0.0 && (x as u32) < width && (y as u32) < height {
                    overview.put_pixel(x as u32, y as u32, image::Rgb(colour));
                }
            }
        }
    }
    overview
}

/// Tells the user about `regions`: one line per region in the run log, an
/// overview with the regions circled and a JSON list next to the Stack_Report,
/// and a `scan_soft_region` entry in the run ledger that names both files.
pub(super) fn report_soft_regions<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    output: &Rgb32FImage,
    regions: &[SoftRegion],
) -> Result<(), String> {
    use super::super::stack_pipeline::degradation;
    let sources_soft = regions
        .iter()
        .filter(|region| region.kind == SoftKind::SourcesSoft)
        .count();
    println!(
        "  - Soft regions: {} ({} where every source is soft, {} where a sharper source could not be aligned)",
        regions.len(),
        sources_soft,
        regions.len() - sources_soft
    );
    for region in regions.iter().take(20) {
        println!(
            "    - {:?} at {},{} {}x{} ({} cells): detail {:.0}% of its surroundings, sharpest source {:.2}x the output",
            region.kind,
            region.x,
            region.y,
            region.width,
            region.height,
            region.cells,
            region.relative_detail * 100.0,
            region.best_source_gain
        );
    }
    if regions.is_empty() {
        return Ok(());
    }
    let document = serde_json::json!({
        "output_size": [output.width(), output.height()],
        "regions": regions,
    });
    let overview = soft_region_overview(output, regions);
    let mut written = serde_json::Map::new();
    if let Some(directory) = super::super::stack_pipeline::report::resolve_stack_report_dir(app) {
        let stem = format!("soft-regions-{}", uuid::Uuid::new_v4());
        let image_path = directory.join(format!("{stem}.png"));
        let list_path = directory.join(format!("{stem}.json"));
        let saved = std::fs::create_dir_all(&directory)
            .map_err(|error| error.to_string())
            .and_then(|()| {
                overview
                    .save(&image_path)
                    .map_err(|error| error.to_string())
            })
            .and_then(|()| {
                std::fs::write(
                    &list_path,
                    serde_json::to_vec_pretty(&document).map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())
            });
        match saved {
            Ok(()) => {
                println!("    - Soft-region overview: {}", image_path.display());
                written.insert(
                    "overview".into(),
                    image_path.to_string_lossy().into_owned().into(),
                );
                written.insert(
                    "list".into(),
                    list_path.to_string_lossy().into_owned().into(),
                );
            }
            Err(error) => eprintln!("Soft-region overview could not be written: {error}"),
        }
    }
    #[cfg(test)]
    if let Some(prefix) = std::env::var_os("RAW_EDITOR_SOFT_REGIONS") {
        let prefix = std::path::PathBuf::from(prefix);
        overview
            .save(prefix.with_extension("png"))
            .map_err(|error| format!("write soft-region overview: {error}"))?;
        std::fs::write(
            prefix.with_extension("json"),
            serde_json::to_vec_pretty(&document).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write soft-region list: {error}"))?;
    }
    let mut detail = serde_json::Map::new();
    detail.insert(
        "regions".into(),
        serde_json::to_value(regions).map_err(|error| error.to_string())?,
    );
    detail.insert("sources_soft".into(), sources_soft.into());
    detail.insert(
        "output_size".into(),
        serde_json::json!([output.width(), output.height()]),
    );
    detail.extend(written);
    degradation::record_run_degradation(
        degradation::SCAN_SOFT_REGION,
        serde_json::Value::Object(detail),
    );
    Ok(())
}
