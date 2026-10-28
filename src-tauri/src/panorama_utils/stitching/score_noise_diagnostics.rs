//! Test-only score-noise diagnostics recorder.
//!
//! Collects probe measurements per cell and writes the research summary.

use super::super::stack_pipeline::focus_fuser;
#[cfg(test)]
use super::super::stack_pipeline::focus_fuser::CellSamplingPlan;
#[cfg(test)]
use super::super::stack_pipeline::focus_fuser::OwnershipGridGeometry;
use image::GrayImage;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use super::{
    FinalOwnerRoiConstraint, SCORE_NOISE_PHASE_OFFSETS, ScoreNoiseAttributionCounts,
    ScoreNoiseCandidateCell, ScoreNoiseCellRecord, ScoreNoiseRegistrationCell,
    ScoreNoiseRoiSummary, ScoreNoiseSummary, sample_focus_plane, score_noise_aggregate,
    score_noise_owner_regions, score_noise_primary_attribution, score_noise_probe_values,
    score_noise_single_owner_rois, score_noise_smoothness_scale, score_noise_source_roi_origins,
    score_noise_variance, score_noise_winner_owners,
};
#[cfg(test)]
use super::{final_world_owner_oracle, owner_quality_oracle};

#[cfg(test)]
pub(super) struct ScoreNoiseDiagnostics {
    pub(super) station_index: usize,
    pub(super) dimensions: (u32, u32),
    pub(super) geometry: OwnershipGridGeometry,
    pub(super) plan: CellSamplingPlan,
    pub(super) candidates: Vec<Vec<ScoreNoiseCandidateCell>>,
    pub(super) source_to_station: std::collections::BTreeMap<String, Matrix3<f64>>,
    pub(super) source_roi_origins: Vec<owner_quality_oracle::SourceRoiOrigin>,
    pub(super) write_score_noise: bool,
}

#[cfg(test)]
impl ScoreNoiseDiagnostics {
    pub(super) fn new(
        station_index: usize,
        dimensions: (u32, u32),
        geometry: OwnershipGridGeometry,
        plan: CellSamplingPlan,
        write_score_noise: bool,
    ) -> Self {
        Self {
            station_index,
            dimensions,
            geometry,
            plan,
            candidates: vec![Vec::new(); geometry.cell_count()],
            source_to_station: std::collections::BTreeMap::new(),
            source_roi_origins: Vec::new(),
            write_score_noise,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_candidate(
        &mut self,
        label: usize,
        source: &str,
        source_to_station: Matrix3<f64>,
        image: &Rgb32FImage,
        mask: &GrayImage,
        left: u32,
        top: u32,
        sharpness: &[f64],
        covered: &[bool],
        disagreement: &[f64],
        registration: &[ScoreNoiseRegistrationCell],
        base_evidence: Option<&focus_fuser::SharpnessEvidenceGrid>,
    ) {
        self.source_to_station
            .insert(source.to_string(), source_to_station);
        if std::env::var_os("RAW_EDITOR_FOCUS_FINAL_ROI_CONSTRAINTS").is_none() {
            self.source_roi_origins
                .extend(score_noise_source_roi_origins(
                    source,
                    focus_fuser::owner_of(label),
                    mask,
                    left,
                    top,
                    self.dimensions,
                ));
        }
        let columns = self.geometry.columns as usize;
        let write_score_noise = self.write_score_noise;
        let measured = (0..self.geometry.cell_count())
            .into_par_iter()
            .map(|cell| {
                let (column, row) = ((cell % columns) as u32, (cell / columns) as u32);
                let (origin_x, origin_y) = self.geometry.cell_origin(column, row);
                let cell_left = origin_x.round().max(0.0) as u32;
                let cell_top = origin_y.round().max(0.0) as u32;
                let cell_right = cell_left
                    .saturating_add(self.geometry.cell_size_px)
                    .min(self.dimensions.0);
                let cell_bottom = cell_top
                    .saturating_add(self.geometry.cell_size_px)
                    .min(self.dimensions.1);
                let fully_covered = cell_left >= left
                    && cell_top >= top
                    && cell_right > cell_left
                    && cell_bottom > cell_top
                    && cell_right - left <= mask.width()
                    && cell_bottom - top <= mask.height()
                    && (cell_top..cell_bottom).all(|y| {
                        (cell_left..cell_right).all(|x| mask.get_pixel(x - left, y - top)[0] != 0)
                    });
                let mut low_frequency_samples = Vec::with_capacity(9);
                for fy in [0.25, 0.5, 0.75] {
                    for fx in [0.25, 0.5, 0.75] {
                        let x = origin_x + fx * f64::from(self.geometry.cell_size_px);
                        let y = origin_y + fy * f64::from(self.geometry.cell_size_px);
                        if let Some(pixel) = sample_focus_plane(image, mask, left, top, x, y) {
                            low_frequency_samples.push(
                                0.2126 * f64::from(pixel[0])
                                    + 0.7152 * f64::from(pixel[1])
                                    + 0.0722 * f64::from(pixel[2]),
                            );
                        }
                    }
                }
                let low_frequency_luma = if low_frequency_samples.is_empty() {
                    f64::NAN
                } else {
                    low_frequency_samples.iter().sum::<f64>() / low_frequency_samples.len() as f64
                };
                let flat = fully_covered && low_frequency_luma.is_finite();
                let sample = |x: f64, y: f64| sample_focus_plane(image, mask, left, top, x, y);
                let probes = if write_score_noise {
                    score_noise_probe_values(&self.plan, origin_x, origin_y, (0.0, 0.0), sample)
                } else {
                    [0.0; 5]
                };
                let repeated = if write_score_noise {
                    score_noise_probe_values(&self.plan, origin_x, origin_y, (0.0, 0.0), |x, y| {
                        sample_focus_plane(image, mask, left, top, x, y)
                    })
                } else {
                    [0.0; 5]
                };
                let mut phase_scores = [0.0; 9];
                if write_score_noise {
                    for (phase_index, phase) in SCORE_NOISE_PHASE_OFFSETS.into_iter().enumerate() {
                        let values = score_noise_probe_values(
                            &self.plan,
                            origin_x,
                            origin_y,
                            phase,
                            |x, y| sample_focus_plane(image, mask, left, top, x, y),
                        );
                        phase_scores[phase_index] =
                            focus_fuser::normalized_sharpness(score_noise_aggregate(&values));
                    }
                }
                let probe_scores = probes.map(focus_fuser::normalized_sharpness);
                let mut sorted_probe_scores = probe_scores;
                sorted_probe_scores.sort_by(f64::total_cmp);
                let normalized = sharpness.get(cell).copied().unwrap_or(0.0);
                let penalty = base_evidence
                    .and_then(|evidence| {
                        let (x, y) = (
                            column * self.geometry.cell_size_px,
                            row * self.geometry.cell_size_px,
                        );
                        evidence.at_pixel(x, y)
                    })
                    .map_or(0.0, |base| {
                        focus_fuser::mismatch_penalty(
                            disagreement.get(cell).copied().unwrap_or(0.0),
                            normalized,
                            f64::from(base.owner_score),
                        )
                    });
                ScoreNoiseCandidateCell {
                    label,
                    source: source.to_string(),
                    covered: covered.get(cell).copied().unwrap_or(false),
                    fully_covered,
                    flat,
                    low_frequency_luma,
                    probe_raw_acutance: probes,
                    normalized_sharpness: normalized,
                    repeated_measurement_variance: score_noise_variance(
                        &probes
                            .iter()
                            .zip(repeated)
                            .map(|(left, right)| left - right)
                            .collect::<Vec<_>>(),
                    ),
                    phase_normalized_sharpness: phase_scores,
                    phase_offset_variance: score_noise_variance(&phase_scores),
                    probe_mean_sharpness: probe_scores.iter().sum::<f64>() / 5.0,
                    probe_median_sharpness: sorted_probe_scores[2],
                    disagreement: disagreement.get(cell).copied().unwrap_or(0.0),
                    graph_cut_data_cost: covered
                        .get(cell)
                        .copied()
                        .unwrap_or(false)
                        .then(|| focus_fuser::data_cost(normalized, penalty)),
                    registration: registration.get(cell).cloned().unwrap_or_default(),
                }
            })
            .collect::<Vec<_>>();
        for (candidates, candidate) in self.candidates.iter_mut().zip(measured) {
            candidates.push(candidate);
        }
    }

    pub(super) fn write_owner_quality_oracle(
        &self,
        owners: &[u16],
        coverage: &GrayImage,
        directory: &Path,
    ) -> Result<(), String> {
        let mut textured_pixels = vec![0u64; self.geometry.cell_count()];
        if coverage.dimensions() == self.dimensions {
            for y in 0..self.dimensions.1 {
                let row = (y / self.geometry.cell_size_px.max(1))
                    .min(self.geometry.rows.saturating_sub(1));
                for x in 0..self.dimensions.0 {
                    let pixel = (y * self.dimensions.0 + x) as usize;
                    if coverage.as_raw()[pixel] == 0 {
                        continue;
                    }
                    let column = (x / self.geometry.cell_size_px.max(1))
                        .min(self.geometry.columns.saturating_sub(1));
                    let cell = (row * self.geometry.columns + column) as usize;
                    textured_pixels[cell] = textured_pixels[cell].saturating_add(1);
                }
            }
        }
        let cells = self
            .candidates
            .iter()
            .enumerate()
            .map(|(cell, candidates)| {
                let textured = candidates
                    .iter()
                    .filter(|candidate| candidate.covered)
                    .map(|candidate| candidate.normalized_sharpness)
                    .max_by(f64::total_cmp)
                    .is_some_and(|score| score >= focus_fuser::TEXTURED_PIXEL_SHARPNESS_FLOOR);
                owner_quality_oracle::OracleCell {
                    textured_pixels: if textured { textured_pixels[cell] } else { 0 },
                    candidates: candidates
                        .iter()
                        .map(|candidate| owner_quality_oracle::OracleCandidate {
                            owner: focus_fuser::owner_of(candidate.label),
                            source: candidate.source.clone(),
                            sharpness: candidate.normalized_sharpness,
                            data_cost: candidate.graph_cut_data_cost.unwrap_or(f64::INFINITY),
                            covered: candidate.covered,
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        let required_quality_worlds = std::env::var_os("RAW_EDITOR_FOCUS_FINAL_ROI_QUALITY_REPORT")
            .and_then(|path| std::fs::read(path).ok())
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|report| {
                report
                    .pointer("/quality_gate/criteria")?
                    .as_array()?
                    .iter()
                    .find(|criterion| {
                        criterion.get("name").and_then(|name| name.as_str())
                            == Some("noise_sigma_ratio")
                    })
                    .and_then(|criterion| criterion.get("measured"))
                    .and_then(|measured| measured.as_array())
                    .map(|measured| {
                        measured
                            .iter()
                            .filter_map(|item| {
                                Some((
                                    item.pointer("/world/x")?.as_f64()?.to_bits(),
                                    item.pointer("/world/y")?.as_f64()?.to_bits(),
                                ))
                            })
                            .collect::<std::collections::BTreeSet<_>>()
                    })
            });
        let external_roi_options = std::env::var_os("RAW_EDITOR_FOCUS_FINAL_ROI_CONSTRAINTS")
            .and_then(|path| std::fs::read(path).ok())
            .and_then(|bytes| serde_json::from_slice::<Vec<FinalOwnerRoiConstraint>>(&bytes).ok())
            .unwrap_or_default()
            .into_iter()
            .filter(|constraint| constraint.station == self.station_index)
            .filter(|constraint| {
                required_quality_worlds.as_ref().is_none_or(|worlds| {
                    worlds.contains(&(
                        constraint.world_origin[0].to_bits(),
                        constraint.world_origin[1].to_bits(),
                    ))
                })
            })
            .filter_map(|constraint| {
                let source = constraint
                    .cells
                    .first()
                    .and_then(|&cell| cells.get(cell))?
                    .candidates
                    .iter()
                    .find(|candidate| candidate.owner == constraint.local_owner)?
                    .source
                    .clone();
                Some(owner_quality_oracle::ExternalRoiOption {
                    owner: constraint.local_owner,
                    source,
                    x: constraint.x,
                    y: constraint.y,
                    cells: constraint.cells,
                })
            })
            .collect();
        let report = owner_quality_oracle::solve(
            owner_quality_oracle::OracleInput {
                station_index: self.station_index,
                geometry: owner_quality_oracle::OracleGeometry {
                    width: self.dimensions.0,
                    height: self.dimensions.1,
                    columns: self.geometry.columns,
                    rows: self.geometry.rows,
                    cell_size: self.geometry.cell_size_px,
                    roi_side: crate::panorama_utils::stack_pipeline::quality_gate::QUALITY_ROI_SIDE,
                    roi_step: crate::panorama_utils::stack_pipeline::quality_gate::QUALITY_ROI_SIDE
                        / 2,
                    roi_margin:
                        crate::panorama_utils::stack_pipeline::quality_gate::QUALITY_ROI_MARGIN
                            as u32,
                },
                cells,
                production_owners: owners.to_vec(),
                source_roi_origins: self.source_roi_origins.clone(),
                external_roi_options,
            },
            crate::panorama_utils::stack_pipeline::quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX
                as f64,
        );
        let path = directory.join(format!(
            "owner-quality-oracle-station-{}.json",
            self.station_index
        ));
        let file = std::fs::File::create(&path)
            .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
        serde_json::to_writer_pretty(std::io::BufWriter::new(file), &report)
            .map_err(|error| format!("failed to write {}: {error}", path.display()))
    }

    pub(super) fn finish_and_write(
        self,
        owners: &[u16],
        coverage: &GrayImage,
    ) -> Result<(), String> {
        if std::env::var_os("RAW_EDITOR_FOCUS_FINAL_WORLD_OWNER_ORACLE_DIR").is_some() {
            let mut identities = self
                .candidates
                .iter()
                .flat_map(|cell| {
                    cell.iter()
                        .map(|candidate| (candidate.source.clone(), candidate.label))
                })
                .collect::<Vec<_>>();
            identities.sort();
            identities.dedup();
            let sources = identities
                .into_iter()
                .filter_map(|(source, label)| {
                    let source_to_station = self.source_to_station.get(&source).copied()?;
                    let cells = self
                        .candidates
                        .iter()
                        .map(|candidates| {
                            candidates
                                .iter()
                                .find(|candidate| {
                                    candidate.source == source && candidate.label == label
                                })
                                .map_or(
                                    final_world_owner_oracle::StationCandidateCell {
                                        sharpness: 0.0,
                                        data_cost: f64::INFINITY,
                                        disagreement: 1.0,
                                        covered: false,
                                        fully_covered: false,
                                        flat: false,
                                        low_frequency_luma: f64::NAN,
                                    },
                                    |candidate| final_world_owner_oracle::StationCandidateCell {
                                        sharpness: candidate.normalized_sharpness as f32,
                                        data_cost: candidate
                                            .graph_cut_data_cost
                                            .unwrap_or(f64::INFINITY),
                                        disagreement: candidate.disagreement as f32,
                                        covered: candidate.covered,
                                        fully_covered: candidate.fully_covered,
                                        flat: candidate.flat,
                                        low_frequency_luma: candidate.low_frequency_luma,
                                    },
                                )
                        })
                        .collect();
                    Some(final_world_owner_oracle::StationSourceSnapshot {
                        source,
                        local_owner: focus_fuser::owner_of(label),
                        source_to_station,
                        cells,
                    })
                })
                .collect();
            final_world_owner_oracle::record_station_snapshot(
                final_world_owner_oracle::StationSnapshot {
                    station_index: self.station_index,
                    dimensions: self.dimensions,
                    columns: self.geometry.columns,
                    rows: self.geometry.rows,
                    cell_side_px: self.geometry.cell_size_px,
                    production_owners: owners.to_vec(),
                    sources,
                },
            );
        }
        let Some(directory) = std::env::var_os("RAW_EDITOR_FOCUS_OWNER_QUALITY_ORACLE_DIR")
            .or_else(|| std::env::var_os("RAW_EDITOR_FOCUS_SCORE_NOISE_DIAGNOSTICS_DIR"))
            .map(PathBuf::from)
        else {
            return Ok(());
        };
        std::fs::create_dir_all(&directory).map_err(|error| {
            format!("failed to create score-noise diagnostics directory: {error}")
        })?;
        if !self.write_score_noise {
            return self.write_owner_quality_oracle(owners, coverage, &directory);
        }
        let winner_owners = score_noise_winner_owners(&self.candidates);
        let (winner_region_sizes, winner_horizontal_runs) =
            score_noise_owner_regions(&self.geometry, &winner_owners);
        let (production_region_sizes, production_horizontal_runs) =
            score_noise_owner_regions(&self.geometry, owners);
        let mut records = Vec::with_capacity(self.geometry.cell_count());
        let mut data_owners = vec![0u16; self.geometry.cell_count()];
        let mut attribution = ScoreNoiseAttributionCounts::default();
        let mut signals = ScoreNoiseAttributionCounts::default();
        let mut textured = 0u64;
        let mut shortfall = 0u64;
        let mut data_shortfall = 0u64;
        let mut roi_counts: HashMap<
            (u32, u32),
            (
                u64,
                u64,
                ScoreNoiseAttributionCounts,
                ScoreNoiseAttributionCounts,
            ),
        > = HashMap::new();
        for cell in 0..self.geometry.cell_count() {
            let candidates = self.candidates[cell].clone();
            let mut ranked = candidates
                .iter()
                .filter(|candidate| candidate.covered)
                .collect::<Vec<_>>();
            ranked.sort_by(|left, right| {
                right
                    .normalized_sharpness
                    .total_cmp(&left.normalized_sharpness)
                    .then_with(|| left.label.cmp(&right.label))
            });
            let winner = ranked.first().copied();
            let runner = ranked.get(1).copied();
            let production_owner = owners.get(cell).copied().unwrap_or(0);
            let production = candidates
                .iter()
                .find(|candidate| focus_fuser::owner_of(candidate.label) == production_owner);
            let data_owner = candidates
                .iter()
                .filter_map(|candidate| candidate.graph_cut_data_cost.map(|cost| (candidate, cost)))
                .min_by(|(left, left_cost), (right, right_cost)| {
                    left_cost
                        .total_cmp(right_cost)
                        .then_with(|| left.label.cmp(&right.label))
                });
            data_owners[cell] = data_owner
                .map(|(candidate, _)| focus_fuser::owner_of(candidate.label))
                .unwrap_or(0);
            let winner_score = winner.map_or(0.0, |candidate| candidate.normalized_sharpness);
            let runner_score = runner.map_or(0.0, |candidate| candidate.normalized_sharpness);
            let owner_score = production.map_or(0.0, |candidate| candidate.normalized_sharpness);
            let owner_shortfall = if ranked.len() <= 1 || winner_score <= 0.0 {
                0.0
            } else {
                ((winner_score - owner_score) / winner_score).clamp(0.0, 1.0)
            };
            let data_score =
                data_owner.map_or(0.0, |(candidate, _)| candidate.normalized_sharpness);
            let data_owner_shortfall = if ranked.len() <= 1 || winner_score <= 0.0 {
                0.0
            } else {
                ((winner_score - data_score) / winner_score).clamp(0.0, 1.0)
            };
            let is_textured = winner_score >= focus_fuser::TEXTURED_PIXEL_SHARPNESS_FLOOR;
            let is_shortfall = is_textured
                && owner_shortfall
                    > crate::panorama_utils::stack_pipeline::quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64;
            textured += u64::from(is_textured);
            shortfall += u64::from(is_shortfall);
            data_shortfall += u64::from(
                is_textured
                    && data_owner_shortfall
                        > crate::panorama_utils::stack_pipeline::quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64,
            );
            let winner_label = winner.map(|candidate| candidate.label);
            let phase_winners = (0..SCORE_NOISE_PHASE_OFFSETS.len())
                .filter_map(|phase| {
                    ranked
                        .iter()
                        .max_by(|left, right| {
                            left.phase_normalized_sharpness[phase]
                                .total_cmp(&right.phase_normalized_sharpness[phase])
                                .then_with(|| right.label.cmp(&left.label))
                        })
                        .map(|candidate| candidate.label)
                })
                .collect::<Vec<_>>();
            let mean_winner = ranked.iter().max_by(|left, right| {
                left.probe_mean_sharpness
                    .total_cmp(&right.probe_mean_sharpness)
                    .then_with(|| right.label.cmp(&left.label))
            });
            let median_winner = ranked.iter().max_by(|left, right| {
                left.probe_median_sharpness
                    .total_cmp(&right.probe_median_sharpness)
                    .then_with(|| right.label.cmp(&left.label))
            });
            let confidence = if winner_score > 0.0 && ranked.len() > 1 {
                ((winner_score - runner_score) / winner_score).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let phase_instability = phase_winners
                .iter()
                .any(|label| Some(*label) != winner_label);
            let aggregation_instability = mean_winner.map(|candidate| candidate.label)
                != winner_label
                || median_winner.map(|candidate| candidate.label) != winner_label;
            let low_texture_or_confidence = winner_score
                < focus_fuser::TEXTURED_PIXEL_SHARPNESS_FLOOR
                || confidence
                    <= crate::panorama_utils::stack_pipeline::quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64;
            let registration_disagreement = production.is_some_and(|candidate| {
                candidate.disagreement > focus_fuser::OWNERSHIP_DISAGREEMENT_VETO
                    || candidate.registration.median_symmetric_residual_px > 0.5
                    || candidate.registration.residual_displacement[0]
                        .hypot(candidate.registration.residual_displacement[1])
                        > 0.5
                    || (candidate.registration.local_to_anchor_jacobian_determinant - 1.0).abs()
                        > 0.01
            });
            let pairwise_override = data_owners[cell] != production_owner;
            let category = score_noise_primary_attribution(
                is_shortfall,
                pairwise_override,
                registration_disagreement,
                phase_instability,
                aggregation_instability,
                low_texture_or_confidence,
            );
            let increment =
                |counts: &mut ScoreNoiseAttributionCounts, category: Option<&str>| match category {
                    Some("sampling_phase_instability") => counts.sampling_phase_instability += 1,
                    Some("probe_aggregation_instability") => {
                        counts.probe_aggregation_instability += 1
                    }
                    Some("low_texture_or_low_confidence") => {
                        counts.low_texture_or_low_confidence += 1
                    }
                    Some("registration_disagreement") => counts.registration_disagreement += 1,
                    Some("pairwise_smoothness_override") => {
                        counts.pairwise_smoothness_override += 1
                    }
                    Some(_) => counts.unattributed += 1,
                    None => {}
                };
            increment(&mut attribution, category);
            if is_shortfall {
                increment(
                    &mut signals,
                    phase_instability.then_some("sampling_phase_instability"),
                );
                increment(
                    &mut signals,
                    aggregation_instability.then_some("probe_aggregation_instability"),
                );
                increment(
                    &mut signals,
                    low_texture_or_confidence.then_some("low_texture_or_low_confidence"),
                );
                increment(
                    &mut signals,
                    registration_disagreement.then_some("registration_disagreement"),
                );
                increment(
                    &mut signals,
                    pairwise_override.then_some("pairwise_smoothness_override"),
                );
            }
            let column = cell as u32 % self.geometry.columns.max(1);
            let row = cell as u32 / self.geometry.columns.max(1);
            let left = column * self.geometry.cell_size_px;
            let top = row * self.geometry.cell_size_px;
            let roi = roi_counts
                .entry((left / 512, top / 512))
                .or_insert_with(|| {
                    (
                        0,
                        0,
                        ScoreNoiseAttributionCounts::default(),
                        ScoreNoiseAttributionCounts::default(),
                    )
                });
            roi.0 += u64::from(is_textured);
            roi.1 += u64::from(is_shortfall);
            increment(&mut roi.2, category);
            if is_shortfall {
                increment(
                    &mut roi.3,
                    phase_instability.then_some("sampling_phase_instability"),
                );
                increment(
                    &mut roi.3,
                    aggregation_instability.then_some("probe_aggregation_instability"),
                );
                increment(
                    &mut roi.3,
                    low_texture_or_confidence.then_some("low_texture_or_low_confidence"),
                );
                increment(
                    &mut roi.3,
                    registration_disagreement.then_some("registration_disagreement"),
                );
                increment(
                    &mut roi.3,
                    pairwise_override.then_some("pairwise_smoothness_override"),
                );
            }
            let pairwise_weight = |neighbour: usize| {
                if owners.get(neighbour).copied().unwrap_or(0) == 0
                    || production_owner == 0
                    || owners.get(neighbour).copied() == Some(production_owner)
                {
                    0.0
                } else {
                    let neighbour_candidates = self
                        .candidates
                        .get(neighbour)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]);
                    let neighbour_disagreement = neighbour_candidates
                        .iter()
                        .map(|candidate| candidate.disagreement)
                        .max_by(f64::total_cmp)
                        .unwrap_or(0.0);
                    crate::panorama_utils::seam_cut::pairwise_weight(
                        candidates
                            .iter()
                            .map(|candidate| candidate.disagreement)
                            .max_by(f64::total_cmp)
                            .unwrap_or(0.0),
                        neighbour_disagreement,
                    ) * score_noise_smoothness_scale(&candidates)
                        .max(score_noise_smoothness_scale(neighbour_candidates))
                }
            };
            let x = column as usize;
            let y = row as usize;
            let neighbours = [
                (y > 0).then(|| cell - self.geometry.columns as usize),
                (x > 0).then(|| cell - 1),
                (x + 1 < self.geometry.columns as usize).then_some(cell + 1),
                (y + 1 < self.geometry.rows as usize)
                    .then_some(cell + self.geometry.columns as usize),
            ];
            let pairwise_contributions =
                neighbours.map(|neighbour| neighbour.map_or(0.0, pairwise_weight));
            records.push(ScoreNoiseCellRecord {
                station_index: self.station_index,
                cell,
                column,
                row,
                bounds: [
                    left,
                    top,
                    self.geometry
                        .cell_size_px
                        .min(self.dimensions.0.saturating_sub(left)),
                    self.geometry
                        .cell_size_px
                        .min(self.dimensions.1.saturating_sub(top)),
                ],
                winner_source: winner.map(|candidate| candidate.source.clone()),
                production_owner_source: production.map(|candidate| candidate.source.clone()),
                data_term_only_owner_source: data_owner
                    .map(|(candidate, _)| candidate.source.clone()),
                winner_score,
                runner_up_score: runner_score,
                top_two_margin: winner_score - runner_score,
                confidence,
                owner_shortfall,
                winner_horizontal_run_cells: winner_horizontal_runs.get(cell).copied().unwrap_or(0),
                winner_connected_region_cells: winner_region_sizes.get(cell).copied().unwrap_or(0),
                production_owner_horizontal_run_cells: production_horizontal_runs
                    .get(cell)
                    .copied()
                    .unwrap_or(0),
                production_owner_connected_region_cells: production_region_sizes
                    .get(cell)
                    .copied()
                    .unwrap_or(0),
                production_data_term: production
                    .and_then(|candidate| candidate.graph_cut_data_cost),
                data_term_only_data_term: data_owner.map(|(_, cost)| cost),
                pairwise_total: pairwise_contributions.iter().sum(),
                pairwise_contributions,
                attribution: category.map(str::to_string),
                candidates,
            });
        }
        let mut rois = roi_counts
            .into_iter()
            .map(
                |((column, row), (roi_textured, roi_shortfall, attribution, signals))| {
                    let left = column * 512;
                    let top = row * 512;
                    ScoreNoiseRoiSummary {
                        left,
                        top,
                        width: 512.min(self.dimensions.0.saturating_sub(left)),
                        height: 512.min(self.dimensions.1.saturating_sub(top)),
                        textured_cells: roi_textured,
                        shortfall_cells: roi_shortfall,
                        owner_sharpness_coverage: (roi_textured > 0)
                            .then_some(1.0 - roi_shortfall as f64 / roi_textured as f64),
                        attribution,
                        signals,
                    }
                },
            )
            .collect::<Vec<_>>();
        rois.sort_by_key(|roi| (roi.top, roi.left));
        let summary = ScoreNoiseSummary {
            schema: 1,
            station_index: self.station_index,
            dimensions: [self.dimensions.0, self.dimensions.1],
            grid: [
                self.geometry.columns,
                self.geometry.rows,
                self.geometry.cell_size_px,
            ],
            production_window_px: self.plan.window_px(),
            phase_offsets_in_cell_sides: SCORE_NOISE_PHASE_OFFSETS,
            textured_cells: textured,
            shortfall_cells: shortfall,
            owner_sharpness_coverage: (textured > 0)
                .then_some(1.0 - shortfall as f64 / textured as f64),
            data_term_only_shortfall_cells: data_shortfall,
            data_term_only_owner_sharpness_coverage: (textured > 0)
                .then_some(1.0 - data_shortfall as f64 / textured as f64),
            production_512px_single_owner_rois: score_noise_single_owner_rois(
                self.dimensions,
                &self.geometry,
                owners,
            ),
            data_term_only_512px_single_owner_rois: score_noise_single_owner_rois(
                self.dimensions,
                &self.geometry,
                &data_owners,
            ),
            attribution,
            signals,
            rois,
        };
        let mut textured_pixels = vec![0u64; self.geometry.cell_count()];
        if coverage.dimensions() == self.dimensions {
            for y in 0..self.dimensions.1 {
                let row = (y / self.geometry.cell_size_px.max(1))
                    .min(self.geometry.rows.saturating_sub(1));
                for x in 0..self.dimensions.0 {
                    let pixel = (y * self.dimensions.0 + x) as usize;
                    if coverage.as_raw()[pixel] == 0 {
                        continue;
                    }
                    let column = (x / self.geometry.cell_size_px.max(1))
                        .min(self.geometry.columns.saturating_sub(1));
                    let cell = (row * self.geometry.columns + column) as usize;
                    textured_pixels[cell] = textured_pixels[cell].saturating_add(1);
                }
            }
        }
        let oracle_cells = self
            .candidates
            .iter()
            .enumerate()
            .map(|(cell, candidates)| {
                let textured = candidates
                    .iter()
                    .filter(|candidate| candidate.covered)
                    .map(|candidate| candidate.normalized_sharpness)
                    .max_by(f64::total_cmp)
                    .is_some_and(|score| score >= focus_fuser::TEXTURED_PIXEL_SHARPNESS_FLOOR);
                owner_quality_oracle::OracleCell {
                    textured_pixels: if textured { textured_pixels[cell] } else { 0 },
                    candidates: candidates
                        .iter()
                        .map(|candidate| owner_quality_oracle::OracleCandidate {
                            owner: focus_fuser::owner_of(candidate.label),
                            source: candidate.source.clone(),
                            sharpness: candidate.normalized_sharpness,
                            data_cost: candidate.graph_cut_data_cost.unwrap_or(f64::INFINITY),
                            covered: candidate.covered,
                        })
                        .collect(),
                }
            })
            .collect();
        let oracle_report = owner_quality_oracle::solve(
            owner_quality_oracle::OracleInput {
                station_index: self.station_index,
                geometry: owner_quality_oracle::OracleGeometry {
                    width: self.dimensions.0,
                    height: self.dimensions.1,
                    columns: self.geometry.columns,
                    rows: self.geometry.rows,
                    cell_size: self.geometry.cell_size_px,
                    roi_side: crate::panorama_utils::stack_pipeline::quality_gate::QUALITY_ROI_SIDE,
                    roi_step: crate::panorama_utils::stack_pipeline::quality_gate::QUALITY_ROI_SIDE
                        / 2,
                    roi_margin:
                        crate::panorama_utils::stack_pipeline::quality_gate::QUALITY_ROI_MARGIN
                            as u32,
                },
                cells: oracle_cells,
                production_owners: owners.to_vec(),
                source_roi_origins: self.source_roi_origins.clone(),
                external_roi_options: Vec::new(),
            },
            crate::panorama_utils::stack_pipeline::quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX
                as f64,
        );
        let oracle_path = directory.join(format!(
            "owner-quality-oracle-station-{}.json",
            self.station_index
        ));
        let oracle_file = std::fs::File::create(&oracle_path)
            .map_err(|error| format!("failed to create {}: {error}", oracle_path.display()))?;
        serde_json::to_writer_pretty(std::io::BufWriter::new(oracle_file), &oracle_report)
            .map_err(|error| format!("failed to write {}: {error}", oracle_path.display()))?;

        let detail_path = directory.join(format!(
            "score-noise-station-{}-cells.json",
            self.station_index
        ));
        let summary_path = directory.join(format!(
            "score-noise-station-{}-summary.json",
            self.station_index
        ));
        let detail_file = std::fs::File::create(&detail_path)
            .map_err(|error| format!("failed to create {}: {error}", detail_path.display()))?;
        serde_json::to_writer(std::io::BufWriter::new(detail_file), &records)
            .map_err(|error| format!("failed to write {}: {error}", detail_path.display()))?;
        let summary_file = std::fs::File::create(&summary_path)
            .map_err(|error| format!("failed to create {}: {error}", summary_path.display()))?;
        serde_json::to_writer_pretty(std::io::BufWriter::new(summary_file), &summary)
            .map_err(|error| format!("failed to write {}: {error}", summary_path.display()))?;
        Ok(())
    }
}
