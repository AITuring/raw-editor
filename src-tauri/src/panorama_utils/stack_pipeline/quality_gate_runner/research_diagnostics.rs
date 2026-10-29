//! Test-only gradient-source and effective-coverage research diagnostics.
//!
//! Written only from the ignored acceptance harnesses.

use super::super::quality_gate;
use super::super::quality_gate::QualityRoi;
use super::super::residual_warp::ResidualWarp;
use image::GrayImage;
use nalgebra::Point2;
#[cfg(test)]
use std::collections::BTreeMap;

use super::{QualityGateInput, SourceLoader, output_roi, source_roi_for_quality};

#[cfg(test)]
#[derive(serde::Serialize)]
pub(super) struct GradientOracleCandidate {
    pub(super) source: String,
    pub(super) station_id: usize,
    pub(super) local_scale: f64,
    pub(super) residual_reference_energy: f64,
    pub(super) identity_reference_energy: Option<f64>,
}

#[cfg(test)]
#[derive(serde::Serialize)]
pub(super) struct GradientOracleRoi {
    pub(super) world: [f64; 2],
    pub(super) owner_source: String,
    pub(super) station_id: usize,
    pub(super) output_energy: f64,
    pub(super) candidates: Vec<GradientOracleCandidate>,
}

#[cfg(test)]
pub(super) fn write_gradient_source_oracle(
    input: &QualityGateInput<'_>,
    rois: &[QualityRoi],
    load_source: &mut SourceLoader<'_>,
) -> Result<(), String> {
    let Some(report_path) = std::env::var_os("RAW_EDITOR_FOCUS_GRADIENT_ORACLE_REPORT") else {
        return Ok(());
    };
    let Some(directory) = std::env::var_os("RAW_EDITOR_FOCUS_GRADIENT_ORACLE_DIR") else {
        return Ok(());
    };
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&report_path)
            .map_err(|error| format!("failed to read gradient oracle report: {error}"))?,
    )
    .map_err(|error| format!("failed to parse gradient oracle report: {error}"))?;
    let requested = report
        .pointer("/quality_gate/criteria")
        .and_then(serde_json::Value::as_array)
        .and_then(|criteria| {
            criteria.iter().find(|criterion| {
                criterion.get("name").and_then(serde_json::Value::as_str)
                    == Some("gradient_energy_normalized")
            })
        })
        .and_then(|criterion| criterion.get("failed"))
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|measurement| {
            Some((
                measurement.pointer("/world/x")?.as_f64()?.to_bits(),
                measurement.pointer("/world/y")?.as_f64()?.to_bits(),
            ))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let owner_sources = input
        .sources
        .iter()
        .map(|source| (source.owner, source))
        .collect::<BTreeMap<_, _>>();
    let selected = rois
        .iter()
        .filter(|roi| {
            requested.contains(&(roi.world_origin.0.to_bits(), roi.world_origin.1.to_bits()))
        })
        .collect::<Vec<_>>();
    let mut records = selected
        .iter()
        .filter_map(|roi| {
            let owner = owner_sources.get(&roi.owner).copied()?;
            let output = output_roi(input.output, roi)?;
            let output_energy = quality_gate::normalized_gradient_energy(&output, 1.0).ok()?;
            Some(GradientOracleRoi {
                world: [roi.world_origin.0, roi.world_origin.1],
                owner_source: owner.path.clone(),
                station_id: owner.geometry.station_id,
                output_energy,
                candidates: Vec::new(),
            })
        })
        .collect::<Vec<_>>();
    let identity = ResidualWarp::new();
    for source_info in input.sources {
        let relevant = records
            .iter()
            .enumerate()
            .filter(|(_, record)| record.station_id == source_info.geometry.station_id)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if relevant.is_empty() {
            continue;
        }
        let source = load_source(source_info)?;
        for index in relevant {
            let Some(roi) = selected.iter().copied().find(|roi| {
                roi.world_origin.0.to_bits() == records[index].world[0].to_bits()
                    && roi.world_origin.1.to_bits() == records[index].world[1].to_bits()
            }) else {
                continue;
            };
            let Ok(reference) = source_roi_for_quality(&source, roi, source_info, input.residual)
            else {
                continue;
            };
            let Ok(residual_reference_energy) =
                quality_gate::normalized_gradient_energy(&reference, 1.0)
            else {
                continue;
            };
            let local_scale = quality_gate::local_scale_for_roi(
                roi,
                &source_info.geometry,
                input.residual,
                input.acceptance_render_scale,
            )
            .map(|measurement| measurement.median)
            .unwrap_or(f64::NAN);
            let identity_reference_energy =
                source_roi_for_quality(&source, roi, source_info, &identity)
                    .ok()
                    .and_then(|reference| {
                        quality_gate::normalized_gradient_energy(&reference, 1.0).ok()
                    });
            records[index].candidates.push(GradientOracleCandidate {
                source: source_info.path.clone(),
                station_id: source_info.geometry.station_id,
                local_scale,
                residual_reference_energy,
                identity_reference_energy,
            });
        }
    }
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create gradient oracle directory: {error}"))?;
    let path = directory.join("gradient-source-oracle.json");
    let file = std::fs::File::create(&path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &records)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

#[cfg(test)]
pub(super) fn write_effective_coverage_diagnostics(
    coverage: &GrayImage,
    source_quadrilaterals: &[[Point2<f64>; 4]],
    canvas_world_origin: (f64, f64),
) {
    let Some(directory) = std::env::var_os("RAW_EDITOR_EFFECTIVE_COVERAGE_DIAGNOSTICS_DIR") else {
        return;
    };
    let (width, height) = coverage.dimensions();
    if width == 0 || height == 0 {
        return;
    }
    let mut projected = vec![0u8; width as usize * height as usize];
    for y in 0..height {
        let scan_y = canvas_world_origin.1 + f64::from(y) + 0.5;
        for quad in source_quadrilaterals {
            let mut intersections = [0.0f64; 4];
            let mut count = 0usize;
            for edge in 0..4 {
                let a = quad[edge];
                let b = quad[(edge + 1) % 4];
                if ((a.y <= scan_y && scan_y < b.y) || (b.y <= scan_y && scan_y < a.y))
                    && count < intersections.len()
                {
                    intersections[count] = a.x + (scan_y - a.y) * (b.x - a.x) / (b.y - a.y);
                    count += 1;
                }
            }
            intersections[..count].sort_by(f64::total_cmp);
            for pair in intersections[..count].chunks_exact(2) {
                let start = (pair[0] - canvas_world_origin.0 - 0.5)
                    .ceil()
                    .max(0.0)
                    .min(f64::from(width)) as u32;
                let end = (pair[1] - canvas_world_origin.0 - 0.5)
                    .ceil()
                    .max(0.0)
                    .min(f64::from(width)) as u32;
                for x in start..end {
                    projected[(y * width + x) as usize] = 1;
                }
            }
        }
    }
    let output_nontransparent = coverage
        .as_raw()
        .iter()
        .filter(|&&value| value != 0)
        .count() as u64;
    let projected_union = projected.iter().filter(|&&value| value != 0).count() as u64;
    let output_outside_projected = coverage
        .as_raw()
        .iter()
        .zip(&projected)
        .filter(|(covered, in_union)| **covered != 0 && **in_union == 0)
        .count() as u64;
    let mut components = Vec::new();
    for start in 0..projected.len() {
        if projected[start] != 1 || coverage.as_raw()[start] != 0 {
            continue;
        }
        projected[start] = 2;
        let mut queue = std::collections::VecDeque::from([start]);
        let mut pixels = 0u64;
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        let mut touches_canvas_edge = false;
        while let Some(index) = queue.pop_front() {
            let x = (index % width as usize) as u32;
            let y = (index / width as usize) as u32;
            pixels += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            touches_canvas_edge |= x == 0 || y == 0 || x + 1 == width || y + 1 == height;
            for neighbour in [
                x.checked_sub(1).map(|nx| (nx, y)),
                (x + 1 < width).then_some((x + 1, y)),
                y.checked_sub(1).map(|ny| (x, ny)),
                (y + 1 < height).then_some((x, y + 1)),
            ]
            .into_iter()
            .flatten()
            {
                let neighbour = (neighbour.1 * width + neighbour.0) as usize;
                if projected[neighbour] == 1 && coverage.as_raw()[neighbour] == 0 {
                    projected[neighbour] = 2;
                    queue.push_back(neighbour);
                }
            }
        }
        components.push(serde_json::json!({
            "pixels": pixels,
            "bounds": { "x": min_x, "y": min_y, "width": max_x - min_x + 1, "height": max_y - min_y + 1 },
            "touches_canvas_edge": touches_canvas_edge,
        }));
    }
    components.sort_by(|left, right| right["pixels"].as_u64().cmp(&left["pixels"].as_u64()));
    let missing_projected = components
        .iter()
        .filter_map(|component| component["pixels"].as_u64())
        .sum::<u64>();
    let canvas_area = u64::from(width) * u64::from(height);
    let document = serde_json::json!({
        "canvas": { "width": width, "height": height, "area": canvas_area, "world_origin": [canvas_world_origin.0, canvas_world_origin.1] },
        "output_nontransparent": output_nontransparent,
        "canvas_transparent": canvas_area - output_nontransparent,
        "projected_union": projected_union,
        "missing_projected": missing_projected,
        "output_outside_projected": output_outside_projected,
        "transparent_outside_projected": canvas_area - output_nontransparent - missing_projected + output_outside_projected,
        "missing_fraction": missing_projected as f64 / projected_union.max(1) as f64,
        "component_count": components.len(),
        "components": components,
    });
    let directory = std::path::PathBuf::from(directory);
    if std::fs::create_dir_all(&directory).is_ok() {
        let _ = std::fs::write(
            directory.join("effective-coverage.json"),
            serde_json::to_vec_pretty(&document).unwrap_or_default(),
        );
    }
}
