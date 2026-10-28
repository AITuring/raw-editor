//! Test-only station seam diagnostics writer.
//!
//! Serialises seam distributions, costs and stroke evidence for review runs;
//! enabled only from the ignored acceptance harnesses.

use super::super::coordinates::Projection;
#[cfg(test)]
use super::super::stack_pipeline::residual_warp;
use crate::panorama_utils::shared::ImageInfo;
use image::Rgb;
use image::Rgb32FImage;
#[cfg(test)]
use image::RgbImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::path::PathBuf;

use super::{
    NO_OWNER, STROKE_MATCH_CLOSURE_MAX_PX, STROKE_MATCH_CONTRAST_MIN, STROKE_MATCH_CURVATURE_MAX,
    STROKE_MATCH_ORIENTATION_MAX_DEGREES, STROKE_MATCH_PATCH_RMS_MAX, STROKE_MATCH_SEAM_RADIUS_PX,
    STROKE_MATCH_SEARCH_RADIUS_PX, STROKE_MATCH_UNIQUENESS_MARGIN_MIN,
    STROKE_MATCH_UNIQUENESS_RATIO_MAX, SeamOrientation, StationSeamPlan, StationStrokeView,
    station_current_cross_at, station_seam_cross_at, station_seam_edge_misalignment,
    station_seam_luma, station_seam_sample, station_seam_world_gradient,
    station_stroke_correspondences, stroke_translation_holdout,
};

#[cfg(test)]
pub(super) fn station_seam_distribution(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::json!({ "count": 0 });
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let quantile = |numerator: usize, denominator: usize| {
        let rank = (sorted.len() * numerator).div_ceil(denominator).max(1);
        sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
    };
    serde_json::json!({
        "count": sorted.len(),
        "p50": quantile(50, 100),
        "p95": quantile(95, 100),
        "max": sorted[sorted.len() - 1],
    })
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn write_station_seam_diagnostics(
    plan: &StationSeamPlan,
    first_station: usize,
    second_station: usize,
    first: &ImageInfo,
    second: &ImageInfo,
    first_pixels: &Rgb32FImage,
    second_pixels: &Rgb32FImage,
    first_h: &Matrix3<f64>,
    second_h: &Matrix3<f64>,
    first_inverse: &Matrix3<f64>,
    second_inverse: &Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    residual: &residual_warp::ResidualWarp,
    current_owners: &[u16],
    out_width: u32,
    solver_status: &str,
) -> Result<(), String> {
    let Some(directory) = std::env::var_os("RAW_EDITOR_STACK_SEAM_DIAGNOSTICS_DIR") else {
        return Ok(());
    };
    let directory = PathBuf::from(directory);
    std::fs::create_dir_all(&directory).map_err(|error| {
        format!(
            "failed to create station seam diagnostics directory {}: {error}",
            directory.display()
        )
    })?;
    let mut valid_samples = 0usize;
    let mut colour_discontinuity = Vec::new();
    let mut low_frequency_discontinuity = Vec::new();
    let mut edge_misalignment = Vec::new();
    let sample_step = ((plan.max_x - plan.min_x + 1)
        .max(plan.max_y - plan.min_y + 1)
        .div_ceil(768))
    .max(1);
    for y in (plan.min_y..=plan.max_y).step_by(sample_step as usize) {
        for x in (plan.min_x..=plan.max_x).step_by(sample_step as usize) {
            let Some((first_pixel, _)) = station_seam_sample(
                first,
                first_pixels,
                first_inverse,
                projection,
                residual,
                x,
                y,
                offset_x,
                offset_y,
            ) else {
                continue;
            };
            let Some((second_pixel, _)) = station_seam_sample(
                second,
                second_pixels,
                second_inverse,
                projection,
                residual,
                x,
                y,
                offset_x,
                offset_y,
            ) else {
                continue;
            };
            valid_samples += 1;
            colour_discontinuity.push(
                first_pixel
                    .0
                    .iter()
                    .zip(second_pixel.0)
                    .map(|(left, right)| f64::from((*left - right).abs()).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            );
            low_frequency_discontinuity
                .push((station_seam_luma(first_pixel) - station_seam_luma(second_pixel)).abs());
            if let (Some(first_gradient), Some(second_gradient)) = (
                station_seam_world_gradient(
                    first,
                    first_pixels,
                    first_inverse,
                    projection,
                    residual,
                    x,
                    y,
                    offset_x,
                    offset_y,
                ),
                station_seam_world_gradient(
                    second,
                    second_pixels,
                    second_inverse,
                    projection,
                    residual,
                    x,
                    y,
                    offset_x,
                    offset_y,
                ),
            ) {
                edge_misalignment.push(station_seam_edge_misalignment(
                    first_gradient,
                    second_gradient,
                ));
            }
        }
    }
    let source_view = StationStrokeView {
        image: first,
        pixels: first_pixels,
        inverse: first_inverse,
        projection,
        offset_x,
        offset_y,
    };
    let target_view = StationStrokeView {
        image: second,
        pixels: second_pixels,
        inverse: second_inverse,
        projection,
        offset_x,
        offset_y,
    };
    let stroke_diagnostic = station_stroke_correspondences(plan, source_view, target_view);
    let selected = stroke_diagnostic
        .matches
        .iter()
        .map(|item| item.source)
        .collect::<Vec<_>>();
    let residual_before = stroke_diagnostic
        .matches
        .iter()
        .map(|item| item.displacement[0].hypot(item.displacement[1]))
        .collect::<Vec<_>>();
    let residual_after = stroke_diagnostic
        .matches
        .iter()
        .map(|item| {
            let warped_world = residual.warp_inverse(
                Point2::new(f64::from(item.source.0), f64::from(item.source.1)),
                second.id,
            );
            (f64::from(item.target.0) - warped_world.x)
                .hypot(f64::from(item.target.1) - warped_world.y)
        })
        .collect::<Vec<_>>();
    let residual_matches = stroke_diagnostic
        .matches
        .iter()
        .map(|item| {
            serde_json::json!({
                "world": [f64::from(item.source.0) - offset_x, f64::from(item.source.1) - offset_y],
                "source_canvas": item.source,
                "target_canvas": item.target,
                "offset": item.displacement,
                "before_px": item.displacement[0].hypot(item.displacement[1]),
                "match_score": item.score,
                "reverse_score": item.reverse_score,
                "closure_px": item.closure_px,
                "source_strength": item.source_strength,
                "target_strength": item.target_strength,
                "source_station": first_station,
                "target_station": second_station,
                "provenance": "independent_station_renderings",
            })
        })
        .collect::<Vec<_>>();
    let translation_holdout = stroke_translation_holdout(&stroke_diagnostic.matches);

    let mut planned_path = Vec::new();
    let mut current_path = Vec::new();
    for along_index in 0..plan.cross_samples.len().max(2) {
        let along = plan.along_min
            + ((plan.along_max - plan.along_min) as u64 * along_index as u64
                / (plan.cross_samples.len().max(2) - 1) as u64) as u32;
        planned_path.push(station_seam_cross_at(plan, along));
        let mut transitions = Vec::new();
        let (cross_min, cross_max) = match plan.orientation {
            SeamOrientation::Vertical => (plan.min_x, plan.max_x),
            SeamOrientation::Horizontal => (plan.min_y, plan.max_y),
        };
        let mut previous = None;
        for cross in cross_min..=cross_max {
            let (x, y) = match plan.orientation {
                SeamOrientation::Vertical => (cross, along),
                SeamOrientation::Horizontal => (along, cross),
            };
            let owner = current_owners
                .get(y as usize * out_width as usize + x as usize)
                .copied()
                .unwrap_or(NO_OWNER);
            if let Some(prior) = previous
                && owner != prior
                && [owner, prior].contains(&plan.first_owner)
                && [owner, prior].contains(&plan.second_owner)
            {
                transitions.push(f64::from(cross));
            }
            previous = Some(owner);
        }
        current_path.push(transitions.first().copied());
    }
    let applied_path = (0..plan.cross_samples.len().max(2))
        .map(|along_index| {
            let along = plan.along_min
                + ((plan.along_max - plan.along_min) as u64 * along_index as u64
                    / (plan.cross_samples.len().max(2) - 1) as u64) as u32;
            let planned = station_seam_cross_at(plan, along);
            let current = station_current_cross_at(plan, along);
            if (planned - current).abs() <= 32.0 {
                planned
            } else {
                current
            }
        })
        .collect::<Vec<_>>();
    let measure_path = |path: &[f64]| {
        let mut colour = Vec::new();
        let mut stroke = Vec::new();
        for (along_index, &cross) in path.iter().enumerate() {
            let along = plan.along_min
                + ((plan.along_max - plan.along_min) as u64 * along_index as u64
                    / path.len().saturating_sub(1).max(1) as u64) as u32;
            let (x, y) = match plan.orientation {
                SeamOrientation::Vertical => (cross.round() as u32, along),
                SeamOrientation::Horizontal => (along, cross.round() as u32),
            };
            if let (Some((left, _)), Some((right, _))) = (
                station_seam_sample(
                    first,
                    first_pixels,
                    first_inverse,
                    projection,
                    residual,
                    x,
                    y,
                    offset_x,
                    offset_y,
                ),
                station_seam_sample(
                    second,
                    second_pixels,
                    second_inverse,
                    projection,
                    residual,
                    x,
                    y,
                    offset_x,
                    offset_y,
                ),
            ) {
                colour.push(
                    left.0
                        .iter()
                        .zip(right.0)
                        .map(|(first, second)| f64::from((*first - second).abs()).powi(2))
                        .sum::<f64>()
                        .sqrt(),
                );
            }
            if let (Some(left), Some(right)) = (
                station_seam_world_gradient(
                    first,
                    first_pixels,
                    first_inverse,
                    projection,
                    residual,
                    x,
                    y,
                    offset_x,
                    offset_y,
                ),
                station_seam_world_gradient(
                    second,
                    second_pixels,
                    second_inverse,
                    projection,
                    residual,
                    x,
                    y,
                    offset_x,
                    offset_y,
                ),
            ) {
                let strength = left.0.hypot(left.1).max(right.0.hypot(right.1));
                if strength >= 0.02 {
                    stroke.push(station_seam_edge_misalignment(left, right) + strength.min(1.0));
                }
            }
        }
        (colour, stroke)
    };
    let current_metric_path = (0..plan.cross_samples.len().max(2))
        .map(|along_index| {
            let along = plan.along_min
                + ((plan.along_max - plan.along_min) as u64 * along_index as u64
                    / (plan.cross_samples.len().max(2) - 1) as u64) as u32;
            station_current_cross_at(plan, along)
        })
        .collect::<Vec<_>>();
    let (current_path_colour, current_path_stroke) = measure_path(&current_metric_path);
    let (applied_path_colour, applied_path_stroke) = measure_path(&applied_path);

    let overlay_long = 1600u32;
    let overlap_width = plan.max_x - plan.min_x + 1;
    let overlap_height = plan.max_y - plan.min_y + 1;
    let overlay_scale =
        (f64::from(overlay_long) / f64::from(overlap_width.max(overlap_height))).min(1.0);
    let overlay_width = (f64::from(overlap_width) * overlay_scale).ceil().max(1.0) as u32;
    let overlay_height = (f64::from(overlap_height) * overlay_scale).ceil().max(1.0) as u32;
    let mut overlay = RgbImage::new(overlay_width, overlay_height);
    let mut coverage = RgbImage::new(overlay_width, overlay_height);
    for oy in 0..overlay_height {
        for ox in 0..overlay_width {
            let x = plan.min_x + (f64::from(ox) / overlay_scale).round() as u32;
            let y = plan.min_y + (f64::from(oy) / overlay_scale).round() as u32;
            let first_sample = station_seam_sample(
                first,
                first_pixels,
                first_inverse,
                projection,
                residual,
                x,
                y,
                offset_x,
                offset_y,
            );
            let second_sample = station_seam_sample(
                second,
                second_pixels,
                second_inverse,
                projection,
                residual,
                x,
                y,
                offset_x,
                offset_y,
            );
            if let (Some((left, _)), Some((right, _))) = (first_sample, second_sample) {
                let left = (station_seam_luma(left).clamp(0.0, 1.0) * 255.0) as u8;
                let right = (station_seam_luma(right).clamp(0.0, 1.0) * 255.0) as u8;
                overlay.put_pixel(ox, oy, Rgb([left, right, right]));
                coverage.put_pixel(ox, oy, Rgb([255, 255, 255]));
            } else {
                coverage.put_pixel(
                    ox,
                    oy,
                    Rgb([
                        u8::from(first_sample.is_some()) * 255,
                        u8::from(second_sample.is_some()) * 255,
                        0,
                    ]),
                );
            }
            let along = match plan.orientation {
                SeamOrientation::Vertical => y,
                SeamOrientation::Horizontal => x,
            };
            let cross = match plan.orientation {
                SeamOrientation::Vertical => x,
                SeamOrientation::Horizontal => y,
            };
            if (f64::from(cross) - station_seam_cross_at(plan, along)).abs()
                <= (1.0 / overlay_scale).max(1.0)
            {
                overlay.put_pixel(ox, oy, Rgb([255, 255, 0]));
            }
        }
    }
    for matched in &stroke_diagnostic.matches {
        for (point, colour) in [
            (matched.source, Rgb([0, 255, 0])),
            (matched.target, Rgb([255, 0, 255])),
        ] {
            if point.0 < plan.min_x
                || point.0 > plan.max_x
                || point.1 < plan.min_y
                || point.1 > plan.max_y
            {
                continue;
            }
            let ox = (f64::from(point.0 - plan.min_x) * overlay_scale).round() as i32;
            let oy = (f64::from(point.1 - plan.min_y) * overlay_scale).round() as i32;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let x = ox + dx;
                    let y = oy + dy;
                    if x >= 0 && y >= 0 && x < overlay_width as i32 && y < overlay_height as i32 {
                        overlay.put_pixel(x as u32, y as u32, colour);
                    }
                }
            }
        }
    }
    let stem = format!("station-{first_station}-{second_station}");
    overlay
        .save(directory.join(format!("{stem}-alignment-overlay.png")))
        .map_err(|error| error.to_string())?;
    coverage
        .save(directory.join(format!("{stem}-coverage.png")))
        .map_err(|error| error.to_string())?;
    for (index, &(center_x, center_y)) in selected.iter().take(3).enumerate() {
        let side = 256u32;
        let left = center_x.saturating_sub(side / 2);
        let top = center_y.saturating_sub(side / 2);
        let mut crop = RgbImage::new(side, side);
        for y in 0..side {
            for x in 0..side {
                let world_x = left.saturating_add(x);
                let world_y = top.saturating_add(y);
                if let (Some((first_pixel, _)), Some((second_pixel, _))) = (
                    station_seam_sample(
                        first,
                        first_pixels,
                        first_inverse,
                        projection,
                        residual,
                        world_x,
                        world_y,
                        offset_x,
                        offset_y,
                    ),
                    station_seam_sample(
                        second,
                        second_pixels,
                        second_inverse,
                        projection,
                        residual,
                        world_x,
                        world_y,
                        offset_x,
                        offset_y,
                    ),
                ) {
                    let first_luma = (station_seam_luma(first_pixel).clamp(0.0, 1.0) * 255.0) as u8;
                    let second_luma =
                        (station_seam_luma(second_pixel).clamp(0.0, 1.0) * 255.0) as u8;
                    crop.put_pixel(x, y, Rgb([first_luma, second_luma, second_luma]));
                }
            }
        }
        crop.save(directory.join(format!("{stem}-stroke-crop-{index}.png")))
            .map_err(|error| error.to_string())?;
    }
    let report = serde_json::json!({
        "stations": [first_station, second_station],
        "first_owner": plan.first_owner,
        "second_owner": plan.second_owner,
        "first_on_low_side": plan.first_on_low_side,
        "tile_ids": [first.id, second.id],
        "tile_to_world": [first_h.as_slice(), second_h.as_slice()],
        "overlap_canvas": { "left": plan.min_x, "top": plan.min_y, "right": plan.max_x, "bottom": plan.max_y },
        "world_origin": [-offset_x, -offset_y],
        "orientation": match plan.orientation { SeamOrientation::Vertical => "vertical", SeamOrientation::Horizontal => "horizontal" },
        "valid_overlap_samples": valid_samples,
        "residual_warp_identity": residual.identity(),
        "solver_status": solver_status,
        "colour_discontinuity": station_seam_distribution(&colour_discontinuity),
        "low_frequency_luma_discontinuity": station_seam_distribution(&low_frequency_discontinuity),
        "edge_misalignment": station_seam_distribution(&edge_misalignment),
        "stroke_residual_before": station_seam_distribution(&residual_before),
        "stroke_residual_after": station_seam_distribution(&residual_after),
        "stroke_matches": residual_matches,
        "stroke_match_thresholds": {
            "contrast_min": STROKE_MATCH_CONTRAST_MIN,
            "search_radius_px": STROKE_MATCH_SEARCH_RADIUS_PX,
            "seam_radius_px": STROKE_MATCH_SEAM_RADIUS_PX,
            "orientation_max_degrees": STROKE_MATCH_ORIENTATION_MAX_DEGREES,
            "curvature_max": STROKE_MATCH_CURVATURE_MAX,
            "patch_rms_max": STROKE_MATCH_PATCH_RMS_MAX,
            "uniqueness_ratio_max": STROKE_MATCH_UNIQUENESS_RATIO_MAX,
            "uniqueness_margin_min": STROKE_MATCH_UNIQUENESS_MARGIN_MIN,
            "closure_max_px": STROKE_MATCH_CLOSURE_MAX_PX,
            "spatial_dedup_px": 64,
        },
        "stroke_match_attrition": stroke_diagnostic.attrition,
        "translation_holdout": translation_holdout,
        "current_owner_cross": current_path,
        "planned_seam_cross": planned_path,
        "applied_seam_cross": applied_path,
        "current_path_colour_discontinuity": station_seam_distribution(&current_path_colour),
        "applied_path_colour_discontinuity": station_seam_distribution(&applied_path_colour),
        "current_path_stroke_cost": station_seam_distribution(&current_path_stroke),
        "applied_path_stroke_cost": station_seam_distribution(&applied_path_stroke),
    });
    let file = std::fs::File::create(directory.join(format!("{stem}-seam-diagnostics.json")))
        .map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &report)
        .map_err(|error| error.to_string())
}
