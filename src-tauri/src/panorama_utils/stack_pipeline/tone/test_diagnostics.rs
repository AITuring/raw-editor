//! Test-only tone diagnostics writer.
//!
//! Writes tone distributions and before/after evidence when a test sets the
//! tone diagnostics directory.

use super::super::report::ToneReport;
use image::Rgb32FImage;

use super::{
    TONE_BOUNDARY_WINDOW_PX, TONE_MIN_SAMPLES, TONE_SPATIAL_ITERATIONS, ToneHoldoutGate,
    ToneRelation, ToneSpatialField, ToneTile, ToneTransform, delta_e00_rgb, rgb_to_lab,
};

#[cfg(test)]
pub(super) fn tone_distribution_pair(first: [f32; 3], second: [f32; 3]) -> serde_json::Value {
    let linear = |rgb: [f32; 3]| rgb.map(crate::color_management::srgb_to_linear_channel);
    serde_json::json!({
        "display_srgb_mean": [first, second],
        "linear_srgb_mean": [linear(first), linear(second)],
        "lab_mean": [rgb_to_lab(first), rgb_to_lab(second)],
        "channel_ratio": std::array::from_fn::<f64, 3, _>(|channel| {
            f64::from(first[channel]) / f64::from(second[channel]).max(1.0 / 65_535.0)
        }),
        "delta_e00": delta_e00_rgb(first, second),
    })
}

#[cfg(test)]
pub(super) fn write_tone_diagnostics(
    before: &Rgb32FImage,
    after: &Rgb32FImage,
    tiles: &[ToneTile],
    relations: &[ToneRelation],
    transforms: &[ToneTransform],
    spatial_fields: Option<&[ToneSpatialField]>,
    holdout_gate: ToneHoldoutGate,
    world_origin: (f64, f64),
    windows: &[serde_json::Value],
    report: &ToneReport,
) {
    let Some(directory) =
        crate::panorama_utils::stack_pipeline::test_support::tone_diagnostics_dir()
    else {
        return;
    };
    if let Err(error) = std::fs::create_dir_all(&directory) {
        eprintln!("cannot create tone diagnostics directory: {error}");
        return;
    }
    let relation_records = relations
        .iter()
        .map(|relation| {
            let first = &tiles[relation.first];
            let second = &tiles[relation.second];
            let residual = std::array::from_fn::<f64, 3, _>(|channel| {
                f64::from(transforms[relation.first].gain[channel]).ln()
                    - f64::from(transforms[relation.second].gain[channel]).ln()
                    + relation.log_gain[channel]
            });
            let corrected_mean = |tile: usize, owner_side: bool| {
                let transform = &transforms[tile];
                let samples = &relation.samples;
                std::array::from_fn::<f32, 3, _>(|channel| {
                    samples
                        .iter()
                        .map(|sample| {
                            let value = if owner_side {
                                sample.owner[channel]
                            } else {
                                sample.source[channel]
                            };
                            f64::from(transform.gain[channel] * value + transform.offset[channel])
                        })
                        .sum::<f64>() as f32
                        / samples.len().max(1) as f32
                })
            };
            let first_pre = std::array::from_fn::<f32, 3, _>(|channel| {
                relation.samples.iter().map(|sample| sample.owner[channel]).sum::<f32>()
                    / relation.samples.len().max(1) as f32
            });
            let second_pre = std::array::from_fn::<f32, 3, _>(|channel| {
                relation.samples.iter().map(|sample| sample.source[channel]).sum::<f32>()
                    / relation.samples.len().max(1) as f32
            });
            let representative = relation
                .samples
                .iter()
                .take(16)
                .map(|sample| serde_json::json!({ "first": sample.owner, "second": sample.source }))
                .collect::<Vec<_>>();
            serde_json::json!({
                "tile_indices": [relation.first, relation.second],
                "stations": [first.station_index, second.station_index],
                "owner_ids": [first.owner_id, second.owner_id],
                "overlap_world": {
                    "left": first.world_origin.0.max(second.world_origin.0),
                    "top": first.world_origin.1.max(second.world_origin.1),
                    "right": (first.world_origin.0 + first.world_size.0).min(second.world_origin.0 + second.world_size.0),
                    "bottom": (first.world_origin.1 + first.world_size.1).min(second.world_origin.1 + second.world_size.1),
                },
                "pair_samples": {
                    "retained": relation.samples.len(),
                    "minimum": TONE_MIN_SAMPLES,
                    "texture_rejection_enabled": false,
                },
                "log_gain": relation.log_gain,
                "scatter": relation.scatter,
                "weight": relation.weight(),
                "solver_log_residual": residual,
                "pre": tone_distribution_pair(first_pre, second_pre),
                "post": tone_distribution_pair(
                    corrected_mean(relation.first, true),
                    corrected_mean(relation.second, false),
                ),
                "representative_low_frequency_samples": representative,
                "holdout": { "enabled": false },
            })
        })
        .collect::<Vec<_>>();
    let tile_records = tiles
        .iter()
        .zip(transforms)
        .map(|(tile, transform)| {
            serde_json::json!({
                "station": tile.station_index,
                "owner_id": tile.owner_id,
                "world_origin": tile.world_origin,
                "world_size": tile.world_size,
                "world_stride": tile.world_stride,
                "field_dimensions": tile.low.dimensions(),
                "gain": transform.gain,
                "bias": transform.offset,
                "solved_gain": transform.solved_gain,
                "solved_bias": transform.solved_offset,
                "supported": transform.supported,
                "clamped": transform.clamped,
            })
        })
        .collect::<Vec<_>>();
    let cycle_consistency = if relations.len() == 3 {
        let relation_gain = |first: usize, second: usize, channel: usize| {
            relations.iter().find_map(|relation| {
                if relation.first == first && relation.second == second {
                    Some(relation.log_gain[channel])
                } else if relation.first == second && relation.second == first {
                    Some(-relation.log_gain[channel])
                } else {
                    None
                }
            })
        };
        (0..3)
            .map(|channel| {
                relation_gain(0, 1, channel).unwrap_or(0.0)
                    + relation_gain(1, 2, channel).unwrap_or(0.0)
                    + relation_gain(2, 0, channel).unwrap_or(0.0)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let payload = serde_json::json!({
        "schema": 1,
        "color_encoding": "display_srgb_with_explicit_linear_and_lab_diagnostics",
        "canvas": { "width": after.width(), "height": after.height(), "world_origin": world_origin },
        "graph": {
            "reference": "zero_mean_log_gain_per_connected_component",
            "relations": relations.len(),
            "cycle_log_residual": cycle_consistency,
            "deterministic_relation_order": true,
        },
        "tiles": tile_records,
        "relations": relation_records,
        "boundary_windows": windows,
        "summary": {
            "status": report.status,
            "maximum_delta_e00": report.boundary_delta_e.max,
            "threshold": report.boundary_delta_e.threshold,
            "window_count": windows.len(),
            "violations": report.boundary_delta_e.violations.len(),
            "spatial_field_enabled": spatial_fields.is_some(),
            "spatial_iterations": TONE_SPATIAL_ITERATIONS,
            "held_out_improvement_gate": {
                "samples": holdout_gate.samples,
                "before_median_delta_e00": holdout_gate.before_median,
                "before_p95_delta_e00": holdout_gate.before_p95,
                "after_median_delta_e00": holdout_gate.after_median,
                "after_p95_delta_e00": holdout_gate.after_p95,
                "applied": holdout_gate.applied,
            },
        },
    });
    if let Ok(bytes) = serde_json::to_vec_pretty(&payload) {
        if let Err(error) = std::fs::write(directory.join("tone-final-world.json"), bytes) {
            eprintln!("cannot write tone diagnostics: {error}");
        }
    }

    let maximum = windows.iter().max_by(|left, right| {
        left["delta_e00"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&right["delta_e00"].as_f64().unwrap_or(0.0))
    });
    if let Some(maximum) = maximum {
        let world = &maximum["world"];
        let left = world["left"].as_f64().unwrap_or(world_origin.0);
        let top = world["top"].as_f64().unwrap_or(world_origin.1);
        let x = (left - world_origin.0).max(0.0) as u32;
        let y = (top - world_origin.1).max(0.0) as u32;
        let crop_left = x.saturating_sub(TONE_BOUNDARY_WINDOW_PX);
        let crop_top = y.saturating_sub(TONE_BOUNDARY_WINDOW_PX);
        let crop_width = (TONE_BOUNDARY_WINDOW_PX * 3).min(after.width() - crop_left);
        let crop_height = (TONE_BOUNDARY_WINDOW_PX * 3).min(after.height() - crop_top);
        for (name, image) in [("pre", before), ("post", after)] {
            let crop =
                image::imageops::crop_imm(image, crop_left, crop_top, crop_width, crop_height)
                    .to_image();
            let encoded = image::DynamicImage::ImageRgb32F(crop).to_rgb16();
            if let Err(error) = encoded.save(directory.join(format!("tone-max-{name}.png"))) {
                eprintln!("cannot write tone crop: {error}");
            }
        }
    }

    let heat_width = after.width().div_ceil(TONE_BOUNDARY_WINDOW_PX);
    let heat_height = after.height().div_ceil(TONE_BOUNDARY_WINDOW_PX);
    let mut heat = image::RgbImage::new(heat_width, heat_height);
    for window in windows {
        let world = &window["world"];
        let x = ((world["left"].as_f64().unwrap_or(world_origin.0) - world_origin.0)
            / f64::from(TONE_BOUNDARY_WINDOW_PX)) as u32;
        let y = ((world["top"].as_f64().unwrap_or(world_origin.1) - world_origin.1)
            / f64::from(TONE_BOUNDARY_WINDOW_PX)) as u32;
        if x >= heat_width || y >= heat_height {
            continue;
        }
        let value = (window["delta_e00"].as_f64().unwrap_or(0.0) / 20.0).clamp(0.0, 1.0);
        let old = heat.get_pixel(x, y).0;
        heat.put_pixel(
            x,
            y,
            image::Rgb([old[0].max((255.0 * value) as u8), old[1], old[2]]),
        );
    }
    if let Err(error) = heat.save(directory.join("tone-boundary-heatmap.png")) {
        eprintln!("cannot write tone heatmap: {error}");
    }
}
