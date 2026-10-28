//! Test-only final-world owner oracle runner.
//!
//! Replays the final world ownership against Source_RAW samples for research
//! reports; never part of a production render.

use super::super::stack_pipeline::focus_fuser;
#[cfg(test)]
use super::super::stack_pipeline::residual_warp;
use image::GrayImage;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;
use std::path::PathBuf;

#[cfg(test)]
use super::final_world_owner_oracle;

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_final_world_owner_oracle<F>(
    output: &mut Rgb32FImage,
    coverage: &GrayImage,
    ownership: &mut [u16],
    confidence: &mut [f32],
    textured: &mut [u8],
    owner_shortfall: &mut [u8],
    owner_disagreement: &mut [u8],
    world_origin: (f64, f64),
    sources: &[crate::panorama_utils::stack_pipeline::quality_gate_runner::QualitySource],
    residual: &residual_warp::ResidualWarp,
    load_source: &mut F,
) -> Result<(), String>
where
    F: FnMut(
        &crate::panorama_utils::stack_pipeline::quality_gate_runner::QualitySource,
    ) -> Result<Rgb32FImage, String>,
{
    use crate::panorama_utils::stack_pipeline::quality_gate;
    use crate::panorama_utils::stack_pipeline::quality_gate_runner;

    let Some(directory) =
        std::env::var_os("RAW_EDITOR_FOCUS_FINAL_WORLD_OWNER_ORACLE_DIR").map(PathBuf::from)
    else {
        return Ok(());
    };
    std::fs::create_dir_all(&directory).map_err(|error| {
        format!(
            "failed to create final-world owner oracle directory {}: {error}",
            directory.display()
        )
    })?;
    let snapshots = final_world_owner_oracle::station_snapshots();
    if snapshots.is_empty() {
        return Err("final-world owner oracle has no current-run station evidence".to_string());
    }
    let (width, height) = output.dimensions();
    let pixel_count = width as usize * height as usize;
    if coverage.dimensions() != (width, height)
        || ownership.len() != pixel_count
        || confidence.len() != pixel_count
        || textured.len() != pixel_count
        || owner_shortfall.len() != pixel_count
        || owner_disagreement.len() != pixel_count
    {
        return Err("final-world owner oracle planes do not share one canvas".to_string());
    }

    fn source_key(path: &str) -> u64 {
        // FNV-1a is used only as a compact, order-independent identity inside
        // this test oracle; output provenance continues to use the real path.
        path.as_bytes()
            .iter()
            .fold(0xcbf29ce484222325u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
            })
    }
    fn map_world(
        world: Point2<f64>,
        source: &crate::panorama_utils::stack_pipeline::quality_gate_runner::QualitySource,
        residual: &residual_warp::ResidualWarp,
        inverse: &Matrix3<f64>,
    ) -> Option<Point2<f64>> {
        quality_gate::map_world_to_source_with_inverse(
            world,
            source.geometry.station_id,
            residual,
            inverse,
        )
        .ok()
    }
    fn station_cell_from_point(
        snapshot: &final_world_owner_oracle::StationSnapshot,
        station: Point2<f64>,
    ) -> Option<usize> {
        let x = station.x;
        let y = station.y;
        if !x.is_finite()
            || !y.is_finite()
            || x < 0.0
            || y < 0.0
            || x >= f64::from(snapshot.dimensions.0)
            || y >= f64::from(snapshot.dimensions.1)
        {
            return None;
        }
        let column = (x.floor() as u32 / snapshot.cell_side_px.max(1))
            .min(snapshot.columns.saturating_sub(1));
        let row =
            (y.floor() as u32 / snapshot.cell_side_px.max(1)).min(snapshot.rows.saturating_sub(1));
        Some((row * snapshot.columns + column) as usize)
    }

    fn station_cell(
        snapshot: &final_world_owner_oracle::StationSnapshot,
        source: &final_world_owner_oracle::StationSourceSnapshot,
        raw: Point2<f64>,
    ) -> Option<usize> {
        let mapped = source.source_to_station * Point3::new(raw.x, raw.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            return None;
        }
        station_cell_from_point(
            snapshot,
            Point2::new(mapped.x / mapped.z, mapped.y / mapped.z),
        )
    }

    fn station_cell_for_world(
        snapshot: &final_world_owner_oracle::StationSnapshot,
        station_source: &final_world_owner_oracle::StationSourceSnapshot,
        source: &crate::panorama_utils::stack_pipeline::quality_gate_runner::QualitySource,
        world: Point2<f64>,
        residual: &residual_warp::ResidualWarp,
        inverse: &Matrix3<f64>,
    ) -> Option<usize> {
        if let Some(native_warp) = source.native_warp.as_ref() {
            let station_inverse = native_warp.station_to_world.try_inverse()?;
            let unwarped = residual.warp_inverse(world, source.geometry.station_id);
            let mapped = station_inverse * Point3::new(unwarped.x, unwarped.y, 1.0);
            if mapped.z.abs() < 1e-8 {
                return None;
            }
            let station = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
            // The production reverse lookup selects the nearest station pixel
            // inside the cubic footprint before consulting the evidence grid.
            return station_cell_from_point(
                snapshot,
                Point2::new(station.x.round(), station.y.round()),
            );
        }
        let raw = map_world(world, source, residual, inverse)?;
        station_cell(snapshot, station_source, raw)
    }

    let mut source_by_owner = std::collections::BTreeMap::new();
    let mut source_by_key = std::collections::BTreeMap::new();
    let mut inverses = std::collections::BTreeMap::new();
    for source in sources {
        let key = source_key(&source.path);
        if source_by_key.insert(key, source).is_some() {
            return Err("final-world owner oracle source identity collision".to_string());
        }
        source_by_owner.insert(source.owner, source);
        let inverse = (source.geometry.tile_to_world * source.geometry.member_to_anchor)
            .try_inverse()
            .ok_or_else(|| format!("source mapping is not invertible: {}", source.path))?;
        inverses.insert(key, inverse);
    }
    let snapshots_by_station = snapshots
        .iter()
        .map(|snapshot| (snapshot.station_index, snapshot))
        .collect::<std::collections::BTreeMap<_, _>>();

    const FINAL_CELL_SIDE_PX: u32 = 16;
    let columns = width.div_ceil(FINAL_CELL_SIDE_PX);
    let rows = height.div_ceil(FINAL_CELL_SIDE_PX);
    let mut world_cells = Vec::with_capacity(columns as usize * rows as usize);
    for row in 0..rows {
        let top = row * FINAL_CELL_SIDE_PX;
        let bottom = top.saturating_add(FINAL_CELL_SIDE_PX).min(height);
        for column in 0..columns {
            let left = column * FINAL_CELL_SIDE_PX;
            let right = left.saturating_add(FINAL_CELL_SIDE_PX).min(width);
            let center_x = left + (right - left) / 2;
            let center_y = top + (bottom - top) / 2;
            let center_index = center_y as usize * width as usize + center_x as usize;
            let production_owner = ownership.get(center_index).copied().unwrap_or(0);
            let production_source = source_by_owner.get(&production_owner).copied();
            let production_station = production_source
                .map(|source| source.geometry.station_id)
                .unwrap_or(usize::MAX);
            let station_is_uniform = production_station != usize::MAX
                && (top..bottom).all(|y| {
                    (left..right).all(|x| {
                        let owner = ownership[y as usize * width as usize + x as usize];
                        source_by_owner
                            .get(&owner)
                            .is_some_and(|source| source.geometry.station_id == production_station)
                    })
                });
            let textured_pixels = (top..bottom)
                .flat_map(|y| (left..right).map(move |x| (x, y)))
                .filter(|&(x, y)| textured[y as usize * width as usize + x as usize] != 0)
                .count() as u64;
            let mut candidates = Vec::new();
            if station_is_uniform {
                let snapshot = snapshots_by_station.get(&production_station).copied();
                if let Some(snapshot) = snapshot {
                    for station_source in &snapshot.sources {
                        let Some(source) = sources.iter().find(|source| {
                            source.geometry.station_id == production_station
                                && source.path == station_source.source
                        }) else {
                            continue;
                        };
                        let key = source_key(&source.path);
                        let Some(inverse) = inverses.get(&key) else {
                            continue;
                        };
                        let points = [
                            (left, top),
                            (right.saturating_sub(1), top),
                            (left, bottom.saturating_sub(1)),
                            (right.saturating_sub(1), bottom.saturating_sub(1)),
                            (center_x, center_y),
                        ];
                        let mut evidence = Vec::with_capacity(points.len());
                        let mut raw_points = Vec::with_capacity(points.len());
                        for (x, y) in points {
                            let world = Point2::new(
                                world_origin.0 + f64::from(x),
                                world_origin.1 + f64::from(y),
                            );
                            let Some(raw) = map_world(world, source, residual, inverse) else {
                                evidence.clear();
                                break;
                            };
                            if raw.x < 2.0
                                || raw.y < 2.0
                                || raw.x + 2.0 >= f64::from(source.dimensions.0)
                                || raw.y + 2.0 >= f64::from(source.dimensions.1)
                            {
                                evidence.clear();
                                break;
                            }
                            let Some(cell) = station_cell(snapshot, station_source, raw) else {
                                evidence.clear();
                                break;
                            };
                            let Some(item) = station_source.cells.get(cell).copied() else {
                                evidence.clear();
                                break;
                            };
                            evidence.push(item);
                            raw_points.push(raw);
                        }
                        if evidence.len() != points.len()
                            || evidence.iter().any(|item| !item.fully_covered)
                        {
                            continue;
                        }
                        let center = raw_points[4];
                        candidates.push(final_world_owner_oracle::CandidateSample {
                            source: key,
                            station: u16::try_from(production_station + 1).unwrap_or(u16::MAX),
                            sharpness: f64::from(evidence[4].sharpness),
                            data_cost: evidence[4].data_cost,
                            // Station ownership is held fixed. This is the exact
                            // no-worse point of the accepted seam objective.
                            seam_cost: 0.0,
                            source_x: center.x,
                            source_y: center.y,
                            fully_covered: true,
                            flat: evidence.iter().all(|item| item.flat),
                            low_frequency_luma: evidence
                                .iter()
                                .map(|item| item.low_frequency_luma)
                                .sum::<f64>()
                                / evidence.len() as f64,
                        });
                    }
                }
            }
            world_cells.push(final_world_owner_oracle::WorldCell {
                production_source: production_source.map_or(0, |source| source_key(&source.path)),
                production_station: u16::try_from(production_station.saturating_add(1))
                    .unwrap_or(0),
                textured_pixels,
                candidates,
            });
        }
    }

    let input = final_world_owner_oracle::OracleInput {
        geometry: final_world_owner_oracle::WorldGeometry {
            columns,
            rows,
            cell_side_px: FINAL_CELL_SIDE_PX,
            qg_aligned_region_grid: true,
            world_origin: (world_origin.0.round() as i64, world_origin.1.round() as i64),
            regions: final_world_owner_oracle::RegionGeometry {
                roi_side_px: quality_gate::QUALITY_ROI_SIDE,
                same_owner_ring_px: quality_gate::QUALITY_ROI_MARGIN as u32,
                sampling_phase_px: quality_gate::QUALITY_ROI_SIDE / 2,
            },
        },
        cells: world_cells,
        required_regions: 8,
        shortfall_limit: f64::from(quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX),
    };
    let phase_x1 = std::env::var_os("RAW_EDITOR_FOCUS_FINAL_WORLD_OWNER_X1").is_some();
    let all_proposals = final_world_owner_oracle::exact_region_proposals(&input);
    let proposal_count = all_proposals.len();
    let proposals = if phase_x1 {
        all_proposals
    } else {
        const EXACT_WINDOWS_PER_STATION: usize = 32;
        const EXACT_DIAGNOSTIC_SIDE_PX: u32 = 128;
        let mut by_station = std::collections::BTreeMap::<
            u16,
            Vec<final_world_owner_oracle::ExactRegionProposal>,
        >::new();
        for proposal in all_proposals {
            by_station
                .entry(proposal.station)
                .or_default()
                .push(proposal);
        }
        by_station
            .into_values()
            .flat_map(|mut station| {
                station.sort_by(|left, right| {
                    right
                        .coarse_current_shortfall_pixels
                        .cmp(&left.coarse_current_shortfall_pixels)
                        .then_with(|| left.top.cmp(&right.top))
                        .then_with(|| left.left.cmp(&right.left))
                        .then_with(|| left.source.cmp(&right.source))
                });
                station
                    .into_iter()
                    .take(EXACT_WINDOWS_PER_STATION)
                    .map(|mut proposal| {
                        let inset = proposal.side.saturating_sub(EXACT_DIAGNOSTIC_SIDE_PX) / 2;
                        proposal.left = proposal.left.saturating_add(inset);
                        proposal.top = proposal.top.saturating_add(inset);
                        proposal.side = proposal.side.min(EXACT_DIAGNOSTIC_SIDE_PX);
                        proposal
                    })
            })
            .collect()
    };
    let shortfall_limit = quality_gate_runner::owner_shortfall_limit();
    let mut evaluations = Vec::with_capacity(proposals.len());
    for proposal in proposals {
        let Some(source) = source_by_key.get(&proposal.source).copied() else {
            continue;
        };
        let Some(snapshot) = snapshots_by_station
            .get(&source.geometry.station_id)
            .copied()
        else {
            continue;
        };
        let Some(selected_station_source) = snapshot
            .sources
            .iter()
            .find(|candidate| candidate.source == source.path)
        else {
            continue;
        };
        let right = proposal.left.saturating_add(proposal.side);
        let bottom = proposal.top.saturating_add(proposal.side);
        let mut evaluation = final_world_owner_oracle::ExactRegionEvaluation {
            proposal,
            valid: right <= width && bottom <= height,
            textured_pixels: 0,
            current_shortfall_pixels: 0,
            baseline_shortfall_pixels: 0,
            candidate_shortfall_pixels: 0,
            owner_change_pixels: 0,
            boundary_change_pixels: 0,
            data_cost_delta: 0.0,
            pairwise_cost: 0.0,
            sparse_run_count: 0,
            prediction_replay_mismatch_pixels: 0,
        };
        if !evaluation.valid {
            evaluations.push(evaluation);
            continue;
        }
        for y in proposal.top..bottom {
            let textured_before_row = evaluation.textured_pixels;
            let current_shortfall_before_row = evaluation.current_shortfall_pixels;
            let candidate_shortfall_before_row = evaluation.candidate_shortfall_pixels;
            let mut row_pixels = Vec::with_capacity(proposal.side as usize);
            for x in proposal.left..right {
                let index = y as usize * width as usize + x as usize;
                let production_owner = ownership[index];
                let Some(production_source) = source_by_owner.get(&production_owner).copied()
                else {
                    evaluation.valid = false;
                    break;
                };
                if production_source.geometry.station_id != source.geometry.station_id
                    || coverage.as_raw()[index] == 0
                {
                    evaluation.valid = false;
                    break;
                }
                let world =
                    Point2::new(world_origin.0 + f64::from(x), world_origin.1 + f64::from(y));
                let Some(selected_inverse) = inverses.get(&proposal.source) else {
                    evaluation.valid = false;
                    break;
                };
                let Some(selected_raw) = map_world(world, source, residual, selected_inverse)
                else {
                    evaluation.valid = false;
                    break;
                };
                if selected_raw.x < 2.0
                    || selected_raw.y < 2.0
                    || selected_raw.x + 2.0 >= f64::from(source.dimensions.0)
                    || selected_raw.y + 2.0 >= f64::from(source.dimensions.1)
                {
                    evaluation.valid = false;
                    break;
                }
                let Some(selected_cell) = station_cell_for_world(
                    snapshot,
                    selected_station_source,
                    source,
                    world,
                    residual,
                    selected_inverse,
                ) else {
                    evaluation.valid = false;
                    break;
                };
                let Some(selected_evidence) = selected_station_source.cells.get(selected_cell)
                else {
                    evaluation.valid = false;
                    break;
                };
                if !selected_evidence.fully_covered
                    || !selected_evidence.flat
                    || !selected_evidence.data_cost.is_finite()
                {
                    evaluation.valid = false;
                    break;
                }
                let available = snapshot
                    .sources
                    .iter()
                    .filter_map(|candidate| candidate.cells.get(selected_cell))
                    .filter(|candidate| candidate.covered)
                    .collect::<Vec<_>>();
                let winner = available
                    .iter()
                    .map(|candidate| candidate.sharpness)
                    .max_by(f32::total_cmp)
                    .unwrap_or(selected_evidence.sharpness);
                let candidate_count = available.len();
                let candidate_shortfall = if candidate_count <= 1 || winner <= 0.0 {
                    0.0
                } else {
                    ((winner - selected_evidence.sharpness) / winner.max(f32::EPSILON))
                        .clamp(0.0, 1.0)
                };
                let candidate_shortfall =
                    quality_gate_runner::quantize_hundredths(candidate_shortfall);
                let candidate_is_textured =
                    winner >= focus_fuser::TEXTURED_PIXEL_SHARPNESS_FLOOR as f32;
                let is_textured = textured[index] != 0;
                if candidate_is_textured != is_textured {
                    evaluation.valid = false;
                    break;
                }
                evaluation.textured_pixels += u64::from(is_textured);
                evaluation.current_shortfall_pixels +=
                    u64::from(is_textured && owner_shortfall[index] > shortfall_limit);
                evaluation.candidate_shortfall_pixels +=
                    u64::from(is_textured && candidate_shortfall > shortfall_limit);

                let production_key = source_key(&production_source.path);
                evaluation.owner_change_pixels = evaluation
                    .owner_change_pixels
                    .saturating_add(u64::from(production_key != proposal.source));
                let on_region_boundary =
                    x == proposal.left || x + 1 == right || y == proposal.top || y + 1 == bottom;
                evaluation.boundary_change_pixels =
                    evaluation.boundary_change_pixels.saturating_add(u64::from(
                        on_region_boundary && production_key != proposal.source,
                    ));
                let seam_boundary = [
                    (x.wrapping_sub(1), y),
                    (x.saturating_add(1), y),
                    (x, y.wrapping_sub(1)),
                    (x, y.saturating_add(1)),
                ]
                .into_iter()
                .filter(|&(nx, ny)| nx < width && ny < height)
                .any(|(nx, ny)| {
                    let neighbour = ownership[ny as usize * width as usize + nx as usize];
                    source_by_owner.get(&neighbour).is_some_and(|neighbour| {
                        neighbour.geometry.station_id != source.geometry.station_id
                    })
                });
                let mut decisions = Vec::with_capacity(snapshot.sources.len());
                let mut source_samples = Vec::with_capacity(snapshot.sources.len());
                for station_source in &snapshot.sources {
                    let Some(candidate_source) = sources.iter().find(|candidate| {
                        candidate.geometry.station_id == source.geometry.station_id
                            && candidate.path == station_source.source
                    }) else {
                        continue;
                    };
                    let key = source_key(&candidate_source.path);
                    let Some(inverse) = inverses.get(&key) else {
                        continue;
                    };
                    let Some(raw) = map_world(world, candidate_source, residual, inverse) else {
                        continue;
                    };
                    let Some(cell) = station_cell_for_world(
                        snapshot,
                        station_source,
                        candidate_source,
                        world,
                        residual,
                        inverse,
                    ) else {
                        continue;
                    };
                    let Some(candidate) = station_source.cells.get(cell) else {
                        continue;
                    };
                    if !candidate.covered
                        || raw.x < 2.0
                        || raw.y < 2.0
                        || raw.x + 2.0 >= f64::from(candidate_source.dimensions.0)
                        || raw.y + 2.0 >= f64::from(candidate_source.dimensions.1)
                    {
                        continue;
                    }
                    let candidate_available = snapshot
                        .sources
                        .iter()
                        .filter_map(|source| source.cells.get(cell))
                        .filter(|evidence| evidence.covered)
                        .collect::<Vec<_>>();
                    let candidate_winner = candidate_available
                        .iter()
                        .map(|evidence| evidence.sharpness)
                        .max_by(f32::total_cmp)
                        .unwrap_or(candidate.sharpness);
                    if (candidate_winner >= focus_fuser::TEXTURED_PIXEL_SHARPNESS_FLOOR as f32)
                        != is_textured
                    {
                        continue;
                    }
                    let shortfall = if candidate_available.len() <= 1 || candidate_winner <= 0.0 {
                        0.0
                    } else {
                        ((candidate_winner - candidate.sharpness)
                            / candidate_winner.max(f32::EPSILON))
                        .clamp(0.0, 1.0)
                    };
                    decisions.push(final_world_owner_oracle::ExactCandidateDecision {
                        source: key,
                        station: u16::try_from(source.geometry.station_id + 1).unwrap_or(u16::MAX),
                        station_cell: u32::try_from(cell).unwrap_or(u32::MAX),
                        sharpness_bits: candidate.sharpness.to_bits(),
                        winner_sharpness_bits: candidate_winner.to_bits(),
                        shortfall: quality_gate_runner::quantize_hundredths(shortfall),
                        disagreement: quality_gate_runner::quantize_hundredths(
                            candidate.disagreement,
                        ),
                        data_cost_bits: candidate.data_cost.to_bits(),
                        pairwise_cost_bits: 0.0f64.to_bits(),
                        seam_boundary,
                    });
                    source_samples.push(final_world_owner_oracle::ExactSourceSample {
                        source: key,
                        x_bits: raw.x.to_bits(),
                        y_bits: raw.y.to_bits(),
                    });
                }
                if !decisions
                    .iter()
                    .any(|candidate| candidate.source == proposal.source)
                    || !decisions
                        .iter()
                        .any(|candidate| candidate.source == production_key)
                {
                    evaluation.valid = false;
                    break;
                }
                row_pixels.push(final_world_owner_oracle::ExactPixelEvidence {
                    x,
                    y,
                    covered: true,
                    textured: is_textured,
                    production_source: production_key,
                    production_station: u16::try_from(source.geometry.station_id + 1)
                        .unwrap_or(u16::MAX),
                    production_shortfall: owner_shortfall[index],
                    production_disagreement: owner_disagreement[index],
                    candidates: decisions,
                    source_samples,
                });
            }
            if !evaluation.valid {
                break;
            }
            let runs = final_world_owner_oracle::aggregate_exact_runs(width, row_pixels)?;
            let current =
                final_world_owner_oracle::exact_run_stats(width, &runs, None, shortfall_limit);
            let baseline =
                final_world_owner_oracle::exact_run_stats_data_min(&runs, shortfall_limit);
            let candidate = final_world_owner_oracle::exact_run_stats_for_source(
                &runs,
                proposal.source,
                shortfall_limit,
            );
            evaluation.baseline_shortfall_pixels = evaluation
                .baseline_shortfall_pixels
                .saturating_add(baseline.shortfall_pixels);
            evaluation.data_cost_delta += candidate.data_cost - baseline.data_cost;
            let expected_textured = evaluation.textured_pixels - textured_before_row;
            let expected_shortfall =
                evaluation.current_shortfall_pixels - current_shortfall_before_row;
            let expected_candidate_shortfall =
                evaluation.candidate_shortfall_pixels - candidate_shortfall_before_row;
            evaluation.prediction_replay_mismatch_pixels = evaluation
                .prediction_replay_mismatch_pixels
                .saturating_add(current.textured_pixels.abs_diff(expected_textured))
                .saturating_add(candidate.textured_pixels.abs_diff(expected_textured))
                .saturating_add(current.shortfall_pixels.abs_diff(expected_shortfall))
                .saturating_add(
                    candidate
                        .shortfall_pixels
                        .abs_diff(expected_candidate_shortfall),
                );
            if current.unresolved_pixels != 0
                || baseline.unresolved_pixels != 0
                || candidate.unresolved_pixels != 0
                || baseline.textured_pixels != expected_textured
                || evaluation.prediction_replay_mismatch_pixels != 0
            {
                evaluation.valid = false;
                break;
            }
            evaluation.sparse_run_count += runs.len() as u64;
        }
        if !evaluation.data_cost_delta.is_finite() || evaluation.boundary_change_pixels != 0 {
            evaluation.valid = false;
        }
        evaluations.push(evaluation);
    }
    let (current_textured_pixels, current_shortfall_pixels) = coverage
        .as_raw()
        .iter()
        .zip(textured.iter())
        .zip(owner_shortfall.iter())
        .fold(
            (0u64, 0u64),
            |(textured_count, shortfall_count), ((covered, textured), shortfall)| {
                let measurable = *covered != 0
                    && *textured != 0
                    && *shortfall != quality_gate_runner::OWNER_EVIDENCE_UNKNOWN;
                (
                    textured_count + u64::from(measurable),
                    shortfall_count + u64::from(measurable && *shortfall > shortfall_limit),
                )
            },
        );
    let mut baseline_shortfall_pixels = 0u64;
    let mut baseline_labels = ownership.to_vec();
    let mut baseline_shortfalls = owner_shortfall.to_vec();
    let mut baseline_resolved = vec![false; pixel_count];
    for index in 0..pixel_count {
        if coverage.as_raw()[index] == 0 {
            continue;
        }
        let Some(production_source) = source_by_owner.get(&ownership[index]).copied() else {
            if textured[index] != 0
                && owner_shortfall[index] != quality_gate_runner::OWNER_EVIDENCE_UNKNOWN
            {
                baseline_shortfall_pixels = baseline_shortfall_pixels
                    .saturating_add(u64::from(owner_shortfall[index] > shortfall_limit));
            }
            continue;
        };
        let Some(snapshot) = snapshots_by_station
            .get(&production_source.geometry.station_id)
            .copied()
        else {
            if textured[index] != 0
                && owner_shortfall[index] != quality_gate_runner::OWNER_EVIDENCE_UNKNOWN
            {
                baseline_shortfall_pixels = baseline_shortfall_pixels
                    .saturating_add(u64::from(owner_shortfall[index] > shortfall_limit));
            }
            continue;
        };
        let x = (index % width as usize) as u32;
        let y = (index / width as usize) as u32;
        let world = Point2::new(world_origin.0 + f64::from(x), world_origin.1 + f64::from(y));
        let mut labels = Vec::<(f64, u64, u16, u8)>::new();
        for station_source in &snapshot.sources {
            let Some(candidate_source) = sources.iter().find(|candidate| {
                candidate.geometry.station_id == production_source.geometry.station_id
                    && candidate.path == station_source.source
            }) else {
                continue;
            };
            let key = source_key(&candidate_source.path);
            let Some(inverse) = inverses.get(&key) else {
                continue;
            };
            let Some(raw) = map_world(world, candidate_source, residual, inverse) else {
                continue;
            };
            if raw.x < 2.0
                || raw.y < 2.0
                || raw.x + 2.0 >= f64::from(candidate_source.dimensions.0)
                || raw.y + 2.0 >= f64::from(candidate_source.dimensions.1)
            {
                continue;
            }
            let Some(cell) = station_cell_for_world(
                snapshot,
                station_source,
                candidate_source,
                world,
                residual,
                inverse,
            ) else {
                continue;
            };
            let Some(candidate) = station_source.cells.get(cell) else {
                continue;
            };
            if !candidate.covered || !candidate.data_cost.is_finite() {
                continue;
            }
            let available = snapshot
                .sources
                .iter()
                .filter_map(|source| source.cells.get(cell))
                .filter(|evidence| evidence.covered)
                .collect::<Vec<_>>();
            let winner = available
                .iter()
                .map(|evidence| evidence.sharpness)
                .max_by(f32::total_cmp)
                .unwrap_or(candidate.sharpness);
            if (winner >= focus_fuser::TEXTURED_PIXEL_SHARPNESS_FLOOR as f32)
                != (textured[index] != 0)
            {
                continue;
            }
            let shortfall = if available.len() <= 1 || winner <= 0.0 {
                0.0
            } else {
                ((winner - candidate.sharpness) / winner.max(f32::EPSILON)).clamp(0.0, 1.0)
            };
            labels.push((
                candidate.data_cost,
                key,
                candidate_source.owner,
                quality_gate_runner::quantize_hundredths(shortfall),
            ));
        }
        let selected = labels.into_iter().min_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let Some((_, _, selected_owner, shortfall)) = selected else {
            if textured[index] != 0
                && owner_shortfall[index] != quality_gate_runner::OWNER_EVIDENCE_UNKNOWN
            {
                baseline_shortfall_pixels = baseline_shortfall_pixels
                    .saturating_add(u64::from(owner_shortfall[index] > shortfall_limit));
            }
            continue;
        };
        baseline_labels[index] = selected_owner;
        baseline_shortfalls[index] = shortfall;
        baseline_resolved[index] = true;
        if textured[index] != 0
            && owner_shortfall[index] != quality_gate_runner::OWNER_EVIDENCE_UNKNOWN
        {
            baseline_shortfall_pixels =
                baseline_shortfall_pixels.saturating_add(u64::from(shortfall > shortfall_limit));
        }
    }
    let data_report = final_world_owner_oracle::solve_exact_regions(
        current_textured_pixels,
        current_shortfall_pixels,
        baseline_shortfall_pixels,
        input.required_regions,
        &evaluations,
    );
    let mut current_evaluations = evaluations.clone();
    for evaluation in &mut current_evaluations {
        evaluation.baseline_shortfall_pixels = evaluation.current_shortfall_pixels;
    }
    let current_report = final_world_owner_oracle::solve_exact_regions(
        current_textured_pixels,
        current_shortfall_pixels,
        current_shortfall_pixels,
        input.required_regions,
        &current_evaluations,
    );
    let use_data_baseline =
        data_report.selected_sharpness_coverage > current_report.selected_sharpness_coverage;
    let report = if use_data_baseline {
        data_report
    } else {
        baseline_labels.clone_from_slice(ownership);
        baseline_shortfalls.clone_from_slice(owner_shortfall);
        baseline_resolved.fill(false);
        current_report
    };
    let report_path = directory.join("final-world-owner-oracle.json");
    let source_records = source_by_key
        .iter()
        .map(|(key, source)| {
            serde_json::json!({
                "key": key,
                "source_raw": source.path,
                "station": source.geometry.station_id,
            })
        })
        .collect::<Vec<_>>();
    let document = serde_json::json!({
        "schema": 2,
        "current_run_only": true,
        "evidence_domain": "exact_final_pixels_grouped_into_exact_station_cell_sparse_runs",
        "phase": if phase_x1 { "X1_full" } else { "X0_exact_windows" },
        "selected_baseline": if use_data_baseline { "exact_data_term" } else { "current_owner" },
        "previous_quality_gate_results_read": false,
        "fixture_identity_read": false,
        "quality_gate_roi_list_read": false,
        "station_boundary_policy": "accepted_compositor_station_held_fixed",
        "resource_policy": "one protected region row resident while exact candidates are evaluated",
        "proposal_count_from_current_masks_and_content": proposal_count,
        "evaluated_exact_windows": evaluations,
        "sources": source_records,
        "oracle": report,
    });
    let file = std::fs::File::create(&report_path)
        .map_err(|error| format!("failed to create {}: {error}", report_path.display()))?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &document)
        .map_err(|error| format!("failed to write {}: {error}", report_path.display()))?;

    if phase_x1
        && report.feasible
        && std::env::var_os("RAW_EDITOR_FOCUS_FINAL_WORLD_OWNER_REPLAY").is_some()
    {
        let mut forced_region = vec![false; pixel_count];
        for evaluation in &report.selected {
            let Some(source) = source_by_key.get(&evaluation.proposal.source).copied() else {
                continue;
            };
            let right = evaluation
                .proposal
                .left
                .saturating_add(evaluation.proposal.side)
                .min(width);
            let bottom = evaluation
                .proposal
                .top
                .saturating_add(evaluation.proposal.side)
                .min(height);
            for y in evaluation.proposal.top..bottom {
                for x in evaluation.proposal.left..right {
                    let index = y as usize * width as usize + x as usize;
                    baseline_labels[index] = source.owner;
                    baseline_resolved[index] = true;
                    forced_region[index] = true;
                }
            }
        }
        let mut replay_skipped_transform = 0u64;
        let mut replay_skipped_bounds = 0u64;
        let mut replay_skipped_cell = 0u64;
        let mut replay_skipped_evidence = 0u64;
        let mut pixels_by_owner = std::collections::BTreeMap::<u16, Vec<u32>>::new();
        for (index, &owner) in baseline_labels.iter().enumerate() {
            if coverage.as_raw()[index] != 0 && baseline_resolved[index] {
                pixels_by_owner
                    .entry(owner)
                    .or_default()
                    .push(u32::try_from(index).map_err(|_| "exact replay canvas is too large")?);
            }
        }
        for (owner, pixels) in pixels_by_owner {
            let Some(source) = source_by_owner.get(&owner).copied() else {
                continue;
            };
            let key = source_key(&source.path);
            let Some(snapshot) = snapshots_by_station
                .get(&source.geometry.station_id)
                .copied()
            else {
                continue;
            };
            let Some(station_source) = snapshot
                .sources
                .iter()
                .find(|candidate| candidate.source == source.path)
            else {
                continue;
            };
            let Some(inverse) = inverses.get(&key) else {
                continue;
            };
            let source_pixels = load_source(source)?;
            for index in pixels {
                let index = index as usize;
                let x = (index % width as usize) as u32;
                let y = (index / width as usize) as u32;
                let world =
                    Point2::new(world_origin.0 + f64::from(x), world_origin.1 + f64::from(y));
                let Some(raw) = map_world(world, source, residual, inverse) else {
                    replay_skipped_transform = replay_skipped_transform.saturating_add(1);
                    continue;
                };
                if raw.x < 2.0
                    || raw.y < 2.0
                    || raw.x + 2.0 >= f64::from(source_pixels.width())
                    || raw.y + 2.0 >= f64::from(source_pixels.height())
                {
                    replay_skipped_bounds = replay_skipped_bounds.saturating_add(1);
                    continue;
                }
                let Some(station_cell) = station_cell_for_world(
                    snapshot,
                    station_source,
                    source,
                    world,
                    residual,
                    inverse,
                ) else {
                    replay_skipped_cell = replay_skipped_cell.saturating_add(1);
                    continue;
                };
                let Some(selected_evidence) = station_source.cells.get(station_cell) else {
                    replay_skipped_evidence = replay_skipped_evidence.saturating_add(1);
                    continue;
                };
                let best = snapshot
                    .sources
                    .iter()
                    .filter_map(|candidate| candidate.cells.get(station_cell))
                    .filter(|candidate| candidate.covered)
                    .map(|candidate| candidate.sharpness)
                    .max_by(f32::total_cmp)
                    .unwrap_or(selected_evidence.sharpness);
                let candidate_count = snapshot
                    .sources
                    .iter()
                    .filter_map(|candidate| candidate.cells.get(station_cell))
                    .filter(|candidate| candidate.covered)
                    .count();
                let shortfall = if candidate_count <= 1 || best <= 0.0 {
                    0.0
                } else {
                    ((best - selected_evidence.sharpness) / best.max(f32::EPSILON)).clamp(0.0, 1.0)
                };
                let sampled = quality_gate_runner::sample_quality_source_at_world(
                    &source_pixels,
                    world,
                    source,
                    residual,
                )?;
                output.put_pixel(x, y, sampled);
                ownership[index] = source.owner;
                textured[index] =
                    u8::from(best >= focus_fuser::TEXTURED_PIXEL_SHARPNESS_FLOOR as f32);
                owner_shortfall[index] = quality_gate_runner::quantize_hundredths(shortfall);
                owner_disagreement[index] =
                    quality_gate_runner::quantize_hundredths(selected_evidence.disagreement);
                confidence[index] = if candidate_count <= 1 || best <= 0.0 {
                    0.0
                } else {
                    let mut scores = snapshot
                        .sources
                        .iter()
                        .filter_map(|candidate| candidate.cells.get(station_cell))
                        .filter(|candidate| candidate.covered)
                        .map(|candidate| candidate.sharpness)
                        .collect::<Vec<_>>();
                    scores.sort_by(|left, right| right.total_cmp(left));
                    ((best - scores.get(1).copied().unwrap_or(best)) / best.max(f32::EPSILON))
                        .clamp(0.0, 1.0)
                };
            }
        }
        let replay_shortfall_pixels = coverage
            .as_raw()
            .iter()
            .zip(textured.iter())
            .zip(owner_shortfall.iter())
            .filter(|((covered, textured), shortfall)| {
                **covered != 0
                    && **textured != 0
                    && **shortfall != quality_gate_runner::OWNER_EVIDENCE_UNKNOWN
                    && **shortfall > shortfall_limit
            })
            .count() as u64;
        let outside_expected = coverage
            .as_raw()
            .iter()
            .zip(textured.iter())
            .zip(baseline_shortfalls.iter())
            .zip(forced_region.iter())
            .filter(|(((covered, textured), shortfall), forced)| {
                !**forced
                    && **covered != 0
                    && **textured != 0
                    && **shortfall != quality_gate_runner::OWNER_EVIDENCE_UNKNOWN
                    && **shortfall > shortfall_limit
            })
            .count() as u64;
        let outside_replayed = coverage
            .as_raw()
            .iter()
            .zip(textured.iter())
            .zip(owner_shortfall.iter())
            .zip(forced_region.iter())
            .filter(|(((covered, textured), shortfall), forced)| {
                !**forced
                    && **covered != 0
                    && **textured != 0
                    && **shortfall != quality_gate_runner::OWNER_EVIDENCE_UNKNOWN
                    && **shortfall > shortfall_limit
            })
            .count() as u64;
        let forced_replayed = replay_shortfall_pixels.saturating_sub(outside_replayed);
        let forced_predicted = report
            .selected
            .iter()
            .map(|evaluation| evaluation.candidate_shortfall_pixels)
            .sum::<u64>();
        let forced_baseline = report
            .selected
            .iter()
            .map(|evaluation| evaluation.baseline_shortfall_pixels)
            .sum::<u64>();
        let predicted_shortfall_pixels = report
            .selected_shortfall_pixels
            .expect("feasible exact owner solve has a predicted numerator");
        if replay_shortfall_pixels != predicted_shortfall_pixels {
            return Err(format!(
                "exact owner prediction/replay mismatch: predicted {predicted_shortfall_pixels}, replayed {replay_shortfall_pixels}, outside expected={outside_expected} replayed={outside_replayed}, forced baseline={forced_baseline} predicted={forced_predicted} replayed={forced_replayed}, skipped transform={replay_skipped_transform} bounds={replay_skipped_bounds} cell={replay_skipped_cell} evidence={replay_skipped_evidence}"
            ));
        }
    }
    Ok(())
}
