//! Tone harmonisation after ownership: the public entry points.
//!
//! One function carries the whole pass by design for now (tiles, relations,
//! solve, fidelity, boundary checks and the report).

use super::super::degradation;
use super::super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD;
use super::super::report::ToneBoundaryViolationRecord;
use super::super::report::TonePairWithoutEvidenceRecord;
use super::super::report::ToneReport;
use super::super::report::ToneStatus;
use super::super::report::ToneTileRecord;
use image::GrayImage;
use image::Rgb32FImage;
use rayon::prelude::*;

use super::{
    PairField, TONE_BOUNDARY_BAND_PX, TONE_BOUNDARY_WINDOW_PX, TONE_MAX_ABS_OFFSET, TONE_MAX_GAIN,
    TONE_MIN_GAIN, TONE_MIN_SAMPLES, TONE_SPATIAL_FIELD_ENABLED, ToneHoldoutGate, ToneRelation,
    ToneTile, build_spatial_field_iterations, consistent_samples, constrain_source_fidelity,
    delta_e00_rgb, record_insufficient_samples, record_run_record, record_run_report,
    run_topology_snapshot, sample_spatial_gain, sample_tile_bilinear, solve_tone_transforms,
    spatial_holdout_gate,
};
#[cfg(test)]
use super::{
    TONE_OWNER_ORACLE_SNAPSHOT, ToneOracleBoundaryWindow, ToneOwnerOracleSnapshot,
    tone_distribution_pair, tone_oracle_cell_sets, write_tone_diagnostics,
};

/// Apply all owner-level tone corrections after the immutable ownership plane
/// has been selected.  Each `ToneTile` is an analysis-grid field in world
/// coordinates, so the function samples at most one low-frequency pixel per
/// output pixel and never allocates a second full-resolution RGB canvas.
/// `relations` are the accepted Station_Relations as index pairs into `tiles`.
pub(crate) fn harmonize_after_ownership(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    tiles: &[ToneTile],
    relations: &[(usize, usize)],
    world_origin: (f64, f64),
) -> ToneReport {
    harmonize_after_ownership_with_evidence(
        panorama,
        owners,
        tiles,
        relations,
        world_origin,
        None,
        false,
    )
}

/// Production-facing entry point used by the owner compositor. `width` is
/// explicit because the ownership plane is commonly held as a flat immutable
/// slice; the height is derived from the output image. `relations` are the
/// accepted Station_Relations as index pairs into `tiles`: only their
/// overlaps are tone evidence (需求 9.1).
pub(crate) fn harmonize_tone_tiles(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    width: u32,
    tiles: &[ToneTile],
    evidence: &GrayImage,
    relations: &[(usize, usize)],
) -> ToneReport {
    harmonize_tone_tiles_with_fidelity(panorama, owners, width, tiles, evidence, relations, false)
}

pub(crate) fn harmonize_tone_tiles_with_fidelity(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    width: u32,
    tiles: &[ToneTile],
    evidence: &GrayImage,
    relations: &[(usize, usize)],
    source_fidelity: bool,
) -> ToneReport {
    if width != panorama.width() {
        let report = ToneReport {
            status: ToneStatus::Identity,
            ..ToneReport::default()
        };
        record_run_report(report.clone());
        return report;
    }
    harmonize_after_ownership_with_evidence(
        panorama,
        owners,
        tiles,
        relations,
        (0.0, 0.0),
        Some(evidence),
        source_fidelity,
    )
}

pub(super) fn harmonize_after_ownership_with_evidence(
    panorama: &mut Rgb32FImage,
    owners: &[u16],
    tiles: &[ToneTile],
    relations: &[(usize, usize)],
    world_origin: (f64, f64),
    evidence: Option<&GrayImage>,
    source_fidelity: bool,
) -> ToneReport {
    let mut report = ToneReport::default();
    if panorama.width() == 0 || panorama.height() == 0 || tiles.is_empty() {
        report.status = ToneStatus::Identity;
        record_run_report(report.clone());
        return report;
    }
    let pixel_count = panorama.width() as usize * panorama.height() as usize;
    if owners.len() != pixel_count {
        report.status = ToneStatus::Identity;
        record_run_report(report.clone());
        return report;
    }
    #[cfg(test)]
    let diagnostic_before =
        crate::panorama_utils::stack_pipeline::test_support::tone_diagnostics_dir()
            .map(|_| panorama.clone());
    #[cfg(test)]
    let owner_oracle_enabled =
        crate::panorama_utils::stack_pipeline::test_support::tone_owner_fidelity_dir().is_some();
    #[cfg(test)]
    {
        *TONE_OWNER_ORACLE_SNAPSHOT
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    // One relation per accepted Station_Relation, measured on its doubly
    // covered overlap only (需求 9.1). Tiles that are merely placed next to
    // each other without an accepted relation contribute nothing.
    let mut pairs = relations
        .iter()
        .filter(|&&(left, right)| left != right && left < tiles.len() && right < tiles.len())
        .map(|&(left, right)| (left.min(right), left.max(right)))
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs.dedup();
    let mut solved_relations = Vec::with_capacity(pairs.len());
    let mut sample_support = vec![0usize; tiles.len()];
    for (first, second) in pairs {
        let samples = PairField::new(&tiles[first], &tiles[second])
            .map(|field| field.samples())
            .unwrap_or_default();
        let retained = consistent_samples(&samples);
        if retained.len() < TONE_MIN_SAMPLES {
            record_insufficient_samples(retained.len());
            report
                .pairs_without_evidence
                .push(TonePairWithoutEvidenceRecord {
                    left: tiles[first].station_index,
                    right: tiles[second].station_index,
                    retained_samples: retained.len() as u64,
                });
            continue;
        }
        sample_support[first] = sample_support[first].max(retained.len());
        sample_support[second] = sample_support[second].max(retained.len());
        solved_relations.push(ToneRelation::new(first, second, retained));
    }

    let mut tile_of_owner =
        vec![None; tiles.iter().map(|tile| tile.owner_id).max().unwrap_or(0) as usize + 1];
    for (index, tile) in tiles.iter().enumerate() {
        tile_of_owner[tile.owner_id as usize].get_or_insert(index);
    }
    // One joint solve over every relation; each transform is bounded once, and
    // the pixels, the boundary re-check and the report read the same bounded
    // values while the report keeps the unbounded solution (需求 9.3, 9.9).
    let mut transforms = solve_tone_transforms(tiles.len(), &solved_relations);
    let local = (source_fidelity && world_origin == (0.0, 0.0))
        .then(|| {
            crate::panorama_utils::stack_pipeline::tone_local::fit_local_tone(
                tiles,
                relations,
                owners,
                panorama.width(),
                panorama.height(),
            )
        })
        .flatten();
    report.geometry_active_relation_count = run_topology_snapshot()
        .map(|t| t.accepted.len())
        .unwrap_or(relations.len());
    if let Some(model) = &local {
        report.correction_model = "supported_spatial_affine".into();
        report.native_fitting_relations = model
            .fitted_relations
            .iter()
            .map(|&(a, b)| [tiles[a].station_index, tiles[b].station_index])
            .collect();
    }
    let mut fidelity_strengths = vec![1.0; tiles.len()];
    if source_fidelity && local.is_none() {
        for (index, (tile, transform)) in tiles.iter().zip(&mut transforms).enumerate() {
            fidelity_strengths[index] = constrain_source_fidelity(tile, transform);
        }
    }
    let (spatial_fields, holdout_gate) = if TONE_SPATIAL_FIELD_ENABLED && local.is_none() {
        let holdout_fields =
            build_spatial_field_iterations(tiles, &solved_relations, &transforms, true);
        let holdout_gate =
            spatial_holdout_gate(tiles, &solved_relations, &transforms, &holdout_fields);
        let spatial_fields = holdout_gate
            .applied
            .then(|| build_spatial_field_iterations(tiles, &solved_relations, &transforms, false));
        (spatial_fields, holdout_gate)
    } else {
        (None, ToneHoldoutGate::default())
    };
    #[cfg(test)]
    let owner_oracle_fields = owner_oracle_enabled
        .then(|| build_spatial_field_iterations(tiles, &solved_relations, &transforms, true));
    #[cfg(test)]
    let (owner_oracle_fit_cells, owner_oracle_overlap_cells) = owner_oracle_fields
        .as_deref()
        .map(|fields| tone_oracle_cell_sets(tiles, &solved_relations, fields))
        .unwrap_or_default();
    for (index, tile) in tiles.iter().enumerate() {
        let transform = &transforms[index];
        if local.is_none() && transform.clamped {
            degradation::record_run_degradation(
                degradation::TONE_GAIN_CLAMPED,
                serde_json::json!({
                    "stage": "tone_harmonizer",
                    "station_index": tile.station_index,
                    "solved_gain": transform.solved_gain,
                    "gain": transform.gain,
                    "solved_offset": transform.solved_offset,
                    "offset": transform.offset,
                }),
            );
        }
        let mut record = ToneTileRecord {
            station_index: tile.station_index,
            gain: transform.gain.map(f64::from),
            offset: transform.offset.map(f64::from),
            samples: sample_support[index] as u64,
            clamped: transform.clamped,
            solved_gain: transform.solved_gain.map(f64::from),
            solved_offset: transform.solved_offset.map(f64::from),
            source_fidelity_strength: fidelity_strengths[index],
            ..Default::default()
        };
        if let Some(model) = &local {
            record.samples = model.retained_samples[index] as u64;
            let values = model.parameters[index]
                .iter()
                .enumerate()
                .map(|(cell, p)| {
                    let x = (cell % model.width) as f64 * f64::from(model.stride)
                        + (f64::from(model.stride) - 1.0) * 0.5;
                    let y = (cell / model.width) as f64 * f64::from(model.stride)
                        + (f64::from(model.stride) - 1.0) * 0.5;
                    sample_tile_bilinear(tile, x, y).map_or([0.0; 6], |low| {
                        crate::panorama_utils::stack_pipeline::tone_local::highlight_supported_parameters(*p, low)
                    })
                })
                .collect::<Vec<_>>();
            for c in 0..3 {
                record.spatial_gain_min[c] = values
                    .iter()
                    .map(|p| 1.0 + f64::from(p[c]))
                    .fold(f64::INFINITY, f64::min);
                record.spatial_gain_max[c] = values
                    .iter()
                    .map(|p| 1.0 + f64::from(p[c]))
                    .fold(f64::NEG_INFINITY, f64::max);
                record.spatial_offset_min[c] = values
                    .iter()
                    .map(|p| f64::from(p[c + 3]))
                    .fold(f64::INFINITY, f64::min);
                record.spatial_offset_max[c] = values
                    .iter()
                    .map(|p| f64::from(p[c + 3]))
                    .fold(f64::NEG_INFINITY, f64::max);
                record.gain[c] = 1.0
                    + values.iter().map(|p| f64::from(p[c])).sum::<f64>()
                        / values.len().max(1) as f64;
                record.offset[c] = values.iter().map(|p| f64::from(p[c + 3])).sum::<f64>()
                    / values.len().max(1) as f64;
            }
            record.clamped = record
                .spatial_gain_min
                .iter()
                .any(|v| *v <= f64::from(TONE_MIN_GAIN) + 1e-6)
                || record
                    .spatial_gain_max
                    .iter()
                    .any(|v| *v >= f64::from(TONE_MAX_GAIN) - 1e-6)
                || record
                    .spatial_offset_min
                    .iter()
                    .any(|v| *v <= -f64::from(TONE_MAX_ABS_OFFSET) + 1e-6)
                || record
                    .spatial_offset_max
                    .iter()
                    .any(|v| *v >= f64::from(TONE_MAX_ABS_OFFSET) - 1e-6);
        } else {
            record.spatial_gain_min = record.gain;
            record.spatial_gain_max = record.gain;
            record.spatial_offset_min = record.offset;
            record.spatial_offset_max = record.offset;
        }
        if local.is_some() && record.clamped {
            degradation::record_run_degradation(
                degradation::TONE_GAIN_CLAMPED,
                serde_json::json!({"stage":"supported_spatial_affine", "station_index":tile.station_index,
                    "gain_min":record.spatial_gain_min,"gain_max":record.spatial_gain_max,
                    "offset_min":record.spatial_offset_min,"offset_max":record.spatial_offset_max}),
            );
        }
        report.tiles.push(record);
    }

    // A tile owns its own high-frequency residual. The owner's low field is
    // expanded bilinearly from the analysis grid (design: "再双线性放大") and
    // only that band is corrected: `gain · low + offset + (owner − low)`.
    // Pixels without a correction, without evidence or without any field cell
    // of their owner keep their exact value.
    let output_width = panorama.width();
    let output_height = panorama.height();
    let corrects = |tile_index: usize| {
        let transform = &transforms[tile_index];
        transform.supported
            && (transform
                .gain
                .iter()
                .any(|value| (*value - 1.0).abs() > f32::EPSILON)
                || transform
                    .offset
                    .iter()
                    .any(|value| value.abs() > f32::EPSILON))
    };
    panorama
        .as_mut()
        .par_chunks_mut(output_width as usize * 3)
        .enumerate()
        .for_each(|(output_y, row)| {
            for (output_x, pixel) in row.chunks_exact_mut(3).enumerate() {
                let owner_id = owners[output_y * output_width as usize + output_x];
                let Some(tile_index) = tile_of_owner.get(owner_id as usize).copied().flatten()
                else {
                    continue;
                };
                if local.is_none() && !corrects(tile_index) {
                    continue;
                }
                if let Some(evidence) = evidence
                    && (output_x as u32 >= evidence.width()
                        || output_y as u32 >= evidence.height()
                        || evidence.get_pixel(output_x as u32, output_y as u32)[0] == 0)
                {
                    continue;
                }
                let Some(low) = sample_tile_bilinear(
                    &tiles[tile_index],
                    world_origin.0 + output_x as f64,
                    world_origin.1 + output_y as f64,
                ) else {
                    continue;
                };
                let transform = &transforms[tile_index];
                let delta = if let Some(model) = &local {
                    let corrected = model.corrected(
                        tile_index,
                        low,
                        world_origin.0 + output_x as f64,
                        world_origin.1 + output_y as f64,
                    );
                    std::array::from_fn(|c| corrected[c] - low[c])
                } else {
                    let local_gain = spatial_fields
                        .as_ref()
                        .and_then(|fields| {
                            sample_spatial_gain(
                                &fields[tile_index],
                                world_origin.0 + output_x as f64,
                                world_origin.1 + output_y as f64,
                            )
                        })
                        .unwrap_or([1.0; 3]);
                    std::array::from_fn::<_, 3, _>(|c| {
                        (transform.gain[c] * local_gain[c] - 1.0) * low[c] + transform.offset[c]
                    })
                };
                for channel in 0..3 {
                    pixel[channel] = (pixel[channel] + delta[channel]).clamp(0.0, 1.0)
                }
            }
        });

    let correct = |tile_index: usize, low: [f32; 3], world_x: f64, world_y: f64| -> [f32; 3] {
        if let Some(model) = &local {
            return model.corrected(tile_index, low, world_x, world_y);
        }
        let transform = &transforms[tile_index];
        let local_gain = spatial_fields
            .as_ref()
            .and_then(|fields| sample_spatial_gain(&fields[tile_index], world_x, world_y))
            .unwrap_or([1.0; 3]);
        std::array::from_fn(|channel| {
            transform.gain[channel] * local_gain[channel] * low[channel] + transform.offset[channel]
        })
    };
    // Tile pair and boundary window → corrected low sums of both tiles. Each
    // adjacent pair is read through its common-support fields, so a band next
    // to either tile's coverage edge compares the same content window.
    let mut pair_fields = std::collections::BTreeMap::<(usize, usize), Option<PairField>>::new();
    let mut boundary_windows = std::collections::BTreeMap::<
        (usize, usize, u32, u32),
        ([f64; 3], [f64; 3], [f64; 3], [f64; 3], u64),
    >::new();
    #[cfg(test)]
    let mut diagnostic_windows = Vec::new();
    #[cfg(test)]
    let mut owner_oracle_boundaries =
        std::collections::BTreeMap::<(usize, usize, u32, u32), ToneOracleBoundaryWindow>::new();
    for y in 0..output_height {
        for x in 0..output_width {
            let index = y as usize * output_width as usize + x as usize;
            let Some(left_tile) = tile_of_owner.get(owners[index] as usize).copied().flatten()
            else {
                continue;
            };
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= output_width || ny >= output_height {
                    continue;
                }
                let right_index = ny as usize * output_width as usize + nx as usize;
                let Some(right_tile) = tile_of_owner
                    .get(owners[right_index] as usize)
                    .copied()
                    .flatten()
                else {
                    continue;
                };
                if right_tile == left_tile {
                    continue;
                }
                // The 16px bands on both sides of this boundary pixel, along
                // its normal. Both owners' corrected low fields are read at
                // the same positions, so the window means differ by the tone
                // step at the boundary and not by the artwork's own gradient
                // across a 32px strip (需求 9.8).
                let (first, second) = (left_tile.min(right_tile), left_tile.max(right_tile));
                let Some(field) = pair_fields
                    .entry((first, second))
                    .or_insert_with(|| PairField::new(&tiles[first], &tiles[second]))
                    .as_ref()
                else {
                    continue;
                };
                let window = (
                    first,
                    second,
                    x / TONE_BOUNDARY_WINDOW_PX,
                    y / TONE_BOUNDARY_WINDOW_PX,
                );
                for distance in 0..TONE_BOUNDARY_BAND_PX {
                    let near = if nx > x {
                        (x.checked_sub(distance), Some(y))
                    } else {
                        (Some(x), y.checked_sub(distance))
                    };
                    let far = if nx > x {
                        (
                            Some(nx + distance).filter(|&px| px < output_width),
                            Some(ny),
                        )
                    } else {
                        (
                            Some(nx),
                            Some(ny + distance).filter(|&py| py < output_height),
                        )
                    };
                    for position in [near, far] {
                        let (Some(px), Some(py)) = position else {
                            continue;
                        };
                        let Some((first_low, second_low)) = field.at(
                            world_origin.0 + f64::from(px),
                            world_origin.1 + f64::from(py),
                        ) else {
                            continue;
                        };
                        let entry = boundary_windows.entry(window).or_insert((
                            [0.0f64; 3],
                            [0.0f64; 3],
                            [0.0f64; 3],
                            [0.0f64; 3],
                            0u64,
                        ));
                        let world_x = world_origin.0 + f64::from(px);
                        let world_y = world_origin.1 + f64::from(py);
                        let first_post = correct(first, first_low, world_x, world_y);
                        let second_post = correct(second, second_low, world_x, world_y);
                        for channel in 0..3 {
                            entry.0[channel] += f64::from(first_low[channel]);
                            entry.1[channel] += f64::from(second_low[channel]);
                            entry.2[channel] += f64::from(first_post[channel]);
                            entry.3[channel] += f64::from(second_post[channel]);
                        }
                        entry.4 += 1;
                        #[cfg(test)]
                        if let Some(fields) = owner_oracle_fields.as_deref() {
                            let oracle = owner_oracle_boundaries.entry(window).or_insert(
                                ToneOracleBoundaryWindow {
                                    key: window,
                                    baseline_first_sum: [0.0; 3],
                                    baseline_second_sum: [0.0; 3],
                                    field_first_delta_sum: [0.0; 3],
                                    field_second_delta_sum: [0.0; 3],
                                    samples: 0,
                                },
                            );
                            let candidate = |tile: usize, low: [f32; 3]| {
                                let transform = transforms[tile];
                                let local = sample_spatial_gain(&fields[tile], world_x, world_y)
                                    .unwrap_or([1.0; 3]);
                                let baseline = std::array::from_fn::<f32, 3, _>(|channel| {
                                    transform.gain[channel] * low[channel]
                                        + transform.offset[channel]
                                });
                                let delta = std::array::from_fn::<f32, 3, _>(|channel| {
                                    transform.gain[channel] * (local[channel] - 1.0) * low[channel]
                                });
                                (baseline, delta)
                            };
                            let (first_baseline, first_delta) = candidate(first, first_low);
                            let (second_baseline, second_delta) = candidate(second, second_low);
                            for channel in 0..3 {
                                oracle.baseline_first_sum[channel] +=
                                    f64::from(first_baseline[channel]);
                                oracle.baseline_second_sum[channel] +=
                                    f64::from(second_baseline[channel]);
                                oracle.field_first_delta_sum[channel] +=
                                    f64::from(first_delta[channel]);
                                oracle.field_second_delta_sum[channel] +=
                                    f64::from(second_delta[channel]);
                            }
                            oracle.samples += 1;
                        }
                    }
                }
            }
        }
    }
    for (
        (first, second, window_x, window_y),
        (first_pre_sum, second_pre_sum, first_sum, second_sum, count),
    ) in boundary_windows
    {
        let mean = |sum: [f64; 3]| sum.map(|channel| (channel / count as f64) as f32);
        let first_pre = mean(first_pre_sum);
        let second_pre = mean(second_pre_sum);
        let first_post = mean(first_sum);
        let second_post = mean(second_sum);
        let delta = delta_e00_rgb(first_post, second_post);
        let world = crate::panorama_utils::stack_pipeline::report::WorldRect {
            left: world_origin.0 + f64::from(window_x * TONE_BOUNDARY_WINDOW_PX),
            top: world_origin.1 + f64::from(window_y * TONE_BOUNDARY_WINDOW_PX),
            width: f64::from(TONE_BOUNDARY_WINDOW_PX),
            height: f64::from(TONE_BOUNDARY_WINDOW_PX),
        };
        #[cfg(test)]
        if diagnostic_before.is_some() {
            diagnostic_windows.push(serde_json::json!({
                "stations": [tiles[first].station_index, tiles[second].station_index],
                "owner_ids": [tiles[first].owner_id, tiles[second].owner_id],
                "world": world,
                "samples": count,
                "distance_to_seam_px": { "minimum": 0, "maximum": TONE_BOUNDARY_BAND_PX - 1 },
                "coverage": { "required": "full_bilateral_cell", "minimum": 1.0 },
                "confidence": 1.0,
                "texture_rejection": { "enabled": false, "rejected": 0 },
                "pre": tone_distribution_pair(first_pre, second_pre),
                "post": tone_distribution_pair(first_post, second_post),
                "delta_e00": delta,
            }));
        }
        report.boundary_delta_e.max = report.boundary_delta_e.max.max(delta);
        if delta <= TONE_BOUNDARY_DELTA_E_THRESHOLD {
            continue;
        }
        if report.boundary_delta_e.violations.len() < 4096 {
            report
                .boundary_delta_e
                .violations
                .push(ToneBoundaryViolationRecord {
                    left: tiles[first].station_index,
                    right: tiles[second].station_index,
                    delta_e00: delta,
                    world,
                });
        }
        degradation::record_run_degradation(
            degradation::TONE_BOUNDARY_DELTA_E_EXCEEDED,
            serde_json::json!({
                "left": tiles[first].station_index,
                "right": tiles[second].station_index,
                "delta_e00": delta,
                "x": world.left,
                "y": world.top,
                "window_px": TONE_BOUNDARY_WINDOW_PX,
                "samples": count,
            }),
        );
    }
    report.status = if solved_relations.is_empty() {
        ToneStatus::Identity
    } else if report.boundary_delta_e.violations.is_empty()
        && report.pairs_without_evidence.is_empty()
    {
        ToneStatus::Applied
    } else {
        ToneStatus::Degraded
    };
    #[cfg(test)]
    if let Some(fields) = owner_oracle_fields {
        let snapshot = ToneOwnerOracleSnapshot {
            baseline: panorama.clone(),
            owners: owners.to_vec(),
            tiles: tiles.to_vec(),
            relations: solved_relations.clone(),
            transforms: transforms.clone(),
            fields,
            fit_cells: owner_oracle_fit_cells,
            overlap_cells: owner_oracle_overlap_cells,
            boundaries: owner_oracle_boundaries.into_values().collect(),
            world_origin,
        };
        *TONE_OWNER_ORACLE_SNAPSHOT
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(snapshot);
    }
    #[cfg(test)]
    if let Some(before) = diagnostic_before.as_ref() {
        write_tone_diagnostics(
            before,
            panorama,
            tiles,
            &solved_relations,
            &transforms,
            spatial_fields.as_deref(),
            holdout_gate,
            world_origin,
            &diagnostic_windows,
            &report,
        );
    }
    for tile in &report.tiles {
        record_run_record(*tile);
    }
    record_run_report(report.clone());
    report
}
