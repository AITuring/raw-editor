//! Runtime adapter for the pure Quality_Gate measurements.
//!
//! `quality_gate.rs` deliberately contains no image loading or report policy.
//! This module supplies that small amount of orchestration: it selects ROIs
//! from the final owner planes, decodes one owner source at a time, samples the
//! corresponding reference ROI, and records all nine criteria.  The caller can
//! therefore use it in observation-only mode without changing export pixels.

use image::{GrayImage, ImageBuffer, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Vector3};

use super::degradation;
use super::quality_gate::{self, QualityRoi, SourceGeometry};
use super::report::{UnmeasurableCategory, UnmeasurableRecord, WorldPoint};
use super::residual_warp::ResidualWarp;

/// The native station-local displacement that was actually applied to one
/// Source_RAW before it entered the fused Virtual_Tile.  The homographies are
/// kept alongside the field because the field is addressed in the station
/// canvas, while the Quality_Gate reference is sampled in Source_RAW pixels.
#[derive(Debug, Clone)]
pub(crate) struct NativeSourceWarp {
    /// `Some` when the registrar's native field was applied to the rendered
    /// station layer.  `None` still carries the station provenance: the
    /// reference must reproduce the ordinary raw -> station-layer -> output
    /// sampling chain even when no residual field was needed.
    pub(crate) displacement: Option<super::super::mosaic::NativeResidualDisplacement>,
    pub(crate) source_to_station: Matrix3<f64>,
    pub(crate) station_to_world: Matrix3<f64>,
}

/// A source entry used by the runtime adapter. `owner` is the value copied to
/// the final ownership plane; IDs need not be contiguous.
#[derive(Debug, Clone)]
pub(crate) struct QualitySource {
    pub owner: u16,
    pub path: String,
    pub geometry: SourceGeometry,
    pub dimensions: (u32, u32),
    pub native_warp: Option<NativeSourceWarp>,
    pub native_registration: Option<NativeReferenceMap>,
}

/// Replays the actual station-canvas inverse refinement before the RAW inverse.
#[derive(Debug, Clone)]
pub(crate) struct NativeReferenceMap {
    pub world_to_station: Matrix3<f64>,
    pub station_to_world: Matrix3<f64>,
    pub refinement: crate::panorama_utils::stitching::FocusSourceRefinement,
    pub station_origin: (f64, f64),
}

impl NativeReferenceMap {
    fn corrected_station(&self, world: Point2<f64>) -> Option<Point2<f64>> {
        let station = project(&self.world_to_station, world)?;
        Some(self.refinement.inverse_canvas_point(station))
    }

    fn reference_world(&self, world: Point2<f64>) -> Option<Point2<f64>> {
        project(&self.station_to_world, self.corrected_station(world)?)
    }
}

/// Inputs retained by the compositor after the final image is assembled.
/// `textured` carries the cell-level Textured_Pixel evidence. An absent plane
/// is reported as insufficient evidence instead of widening the statistic to
/// all covered pixels.
pub(crate) struct QualityGateInput<'a> {
    pub output: &'a Rgb32FImage,
    pub coverage: &'a GrayImage,
    pub ownership: &'a [u16],
    pub confidence: &'a [f32],
    pub textured: Option<&'a [u8]>,
    /// Quantised owner shortfall in hundredths (0..=100) for each output
    /// pixel. Keeping this as bytes avoids another full-size f32 plane.
    pub owner_shortfall: Option<&'a [u8]>,
    /// Quantised ownership disagreement in hundredths (0..=100).
    pub owner_disagreement: Option<&'a [u8]>,
    /// Covered output pixels for which the source reverse lookup could not
    /// establish an owner. These are technical evidence gaps.
    pub unresolved_pixel_count: u64,
    pub world_origin: (f64, f64),
    pub sources: &'a [QualitySource],
    /// Accepted Station_Relations as normalized station-index pairs. Boundary
    /// evidence outside these real overlaps is not a production seam metric.
    pub accepted_station_relations: &'a [(usize, usize)],
    pub residual: &'a ResidualWarp,
    pub acceptance_render_scale: f64,
    pub final_sharpen_amount: f64,
}

pub(crate) type SourceLoader<'a> = dyn FnMut(&QualitySource) -> Result<Rgb32FImage, String> + 'a;

mod criteria;

mod source_sampling;

use source_sampling::{apply_run_tone_transform, source_quad, source_roi_for_quality};
pub(crate) use source_sampling::{measure_source_local_scale, sample_quality_source_at_world};
#[cfg(test)]
use source_sampling::{source_roi, source_roi_with_registration};

mod coverage;

pub(crate) use coverage::{accumulate_confidence_score, select_sparse_rois};
use coverage::{confidence_coverage_stats, effective_pixel_coverage_streaming};

mod boundary_pairs;

use boundary_pairs::{
    boundary_pair_error, local_station_relation_for_pair, measure_large_boundaries,
    parse_boundary_owner_pair, parse_boundary_sample_normal, parse_boundary_station_pair,
    sample_quality_source_at_world_with_station_relation,
};

mod boundary_repair;

pub(crate) use boundary_repair::repair_failed_boundary_samples;

mod run_gate;

pub(crate) use run_gate::run_quality_gate;

#[cfg(test)]
use super::report::QualityGateCriterionRecord;
#[cfg(test)]
use super::report::QualityGateReport;
#[cfg(test)]
use super::report::QualityGateVerdict;
#[cfg(test)]
pub(crate) use criteria::criterion_verdict;
#[cfg(test)]
use criteria::owner_disagreement_veto_limit;
#[cfg(test)]
pub(crate) use criteria::owner_sharpness_stats;
pub(crate) use criteria::{
    Criterion, OWNER_EVIDENCE_UNKNOWN, OwnerSharpnessCriterionInput,
    build_owner_sharpness_criterion, overall_verdict, owner_shortfall_limit, quantize_hundredths,
    record_gate_outcome,
};
#[cfg(test)]
use image::Rgb;
#[cfg(test)]
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::BTreeSet;

fn world(roi: &QualityRoi) -> (f64, f64) {
    roi.world_origin
}

fn output_roi(image: &Rgb32FImage, roi: &QualityRoi) -> Option<Rgb32FImage> {
    let side = roi.side;
    if roi.x.checked_add(side)? > image.width() || roi.y.checked_add(side)? > image.height() {
        return None;
    }
    Some(ImageBuffer::from_fn(side, side, |x, y| {
        *image.get_pixel(roi.x + x, roi.y + y)
    }))
}

fn project(matrix: &Matrix3<f64>, p: Point2<f64>) -> Option<Point2<f64>> {
    let v = matrix * Vector3::new(p.x, p.y, 1.0);
    if !v.iter().all(|x| x.is_finite()) || v.z.abs() < 1e-12 {
        return None;
    }
    Some(Point2::new(v.x / v.z, v.y / v.z))
}

#[cfg(test)]
#[derive(Clone)]
struct ToneOwnerEvidence {
    partition: &'static str,
    roi: QualityRoi,
    owner_path: String,
    station_id: usize,
    tile_index: usize,
    baseline_mean: [f32; 3],
    candidate_mean: [f32; 3],
    reference_mean: [f32; 3],
    field_gain: [f32; 3],
    global_gain: [f32; 3],
    global_bias: [f32; 3],
    coverage_min: u8,
    confidence_min: f32,
    confidence_mean: f64,
    overlap_participation: bool,
    fit_disjoint: bool,
    gradient_ratio_candidate_to_source: Option<f64>,
    noise_ratio_candidate_to_source: Option<f64>,
}

#[cfg(test)]
fn tone_owner_distribution(rgb: [f32; 3]) -> serde_json::Value {
    serde_json::json!({
        "display_srgb": rgb,
        "linear_rgb": rgb.map(crate::color_management::srgb_to_linear_channel),
        "lab": super::tone::diagnostic_lab_rgb(rgb),
    })
}

#[cfg(test)]
fn tone_percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * quantile).round() as usize]
}

#[cfg(test)]
fn tone_owner_candidate_roi(
    snapshot: &super::tone::ToneOwnerOracleSnapshot,
    baseline_image: &Rgb32FImage,
    roi: &QualityRoi,
    relaxation: f64,
) -> Option<Rgb32FImage> {
    let baseline = output_roi(baseline_image, roi)?;
    Some(Rgb32FImage::from_fn(roi.side, roi.side, |x, y| {
        let world_x = roi.world_origin.0 + f64::from(x);
        let world_y = roi.world_origin.1 + f64::from(y);
        Rgb(snapshot.candidate_pixel(
            roi.owner,
            world_x,
            world_y,
            baseline.get_pixel(x, y).0,
            relaxation,
        ))
    }))
}

#[cfg(test)]
fn tone_owner_holdout_rois(
    input: &QualityGateInput<'_>,
    formal: &[QualityRoi],
    snapshot: &super::tone::ToneOwnerOracleSnapshot,
) -> Vec<QualityRoi> {
    const SIDE: u32 = 8;
    const STEP: usize = 16;
    const STRATUM: u32 = 512;
    const MAX_HOLDOUTS: usize = 256;
    let (width, height) = input.coverage.dimensions();
    if width < SIDE || height < SIDE {
        return Vec::new();
    }
    let intersects_formal = |x: u32, y: u32| {
        formal.iter().any(|roi| {
            x < roi.x + roi.side && x + SIDE > roi.x && y < roi.y + roi.side && y + SIDE > roi.y
        })
    };
    let mut strata = std::collections::BTreeSet::new();
    let mut result = Vec::new();
    for y in (0..=height - SIDE).step_by(STEP) {
        for x in (0..=width - SIDE).step_by(STEP) {
            if intersects_formal(x, y) {
                continue;
            }
            let owner = input.ownership[(y * width + x) as usize];
            if owner == 0 {
                continue;
            }
            let mut valid = true;
            'rows: for ry in y..y + SIDE {
                for rx in x..x + SIDE {
                    let index = (ry * width + rx) as usize;
                    let world_x = input.world_origin.0 + f64::from(rx);
                    let world_y = input.world_origin.1 + f64::from(ry);
                    if input.coverage.as_raw()[index] == 0
                        || input.ownership[index] != owner
                        || snapshot.is_fit_cell(world_x, world_y)
                    {
                        valid = false;
                        break 'rows;
                    }
                }
            }
            let world_x = input.world_origin.0 + f64::from(x + SIDE / 2);
            let world_y = input.world_origin.1 + f64::from(y + SIDE / 2);
            if !valid || snapshot.field_sample(owner, world_x, world_y).is_none() {
                continue;
            }
            let key = (owner, x / STRATUM, y / STRATUM);
            if !strata.insert(key) {
                continue;
            }
            result.push(QualityRoi {
                x,
                y,
                side: SIDE,
                region_label: 0,
                owner,
                world_origin: (
                    input.world_origin.0 + f64::from(x),
                    input.world_origin.1 + f64::from(y),
                ),
                boundary_clearance: 0.0,
            });
            if result.len() == MAX_HOLDOUTS {
                return result;
            }
        }
    }
    result
}

#[cfg(test)]
fn unique_owner_source(sources: &[QualitySource], owner: u16) -> Result<&QualitySource, String> {
    let matching = sources
        .iter()
        .filter(|source| source.owner == owner)
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(format!(
            "owner {owner} has {} Source_RAW provenance entries",
            matching.len()
        ));
    }
    Ok(matching[0])
}

#[cfg(test)]
fn collect_tone_owner_evidence(
    input: &QualityGateInput<'_>,
    snapshot: &super::tone::ToneOwnerOracleSnapshot,
    formal: &[QualityRoi],
    holdouts: &[QualityRoi],
    load_source: &mut SourceLoader<'_>,
) -> Result<Vec<ToneOwnerEvidence>, String> {
    let mut by_owner = BTreeMap::<u16, Vec<(&'static str, &QualityRoi)>>::new();
    for roi in formal {
        by_owner
            .entry(roi.owner)
            .or_default()
            .push(("formal_quality_gate_verification", roi));
    }
    for roi in holdouts {
        by_owner
            .entry(roi.owner)
            .or_default()
            .push(("owner_raw_fidelity_holdout", roi));
    }
    let mut result = Vec::new();
    for (owner, rois) in by_owner {
        let source_info = unique_owner_source(input.sources, owner)?;
        let source = load_source(source_info)?;
        for (partition, roi) in rois {
            let baseline = output_roi(input.output, roi).ok_or_else(|| {
                "tone owner baseline ROI is outside the current canvas".to_string()
            })?;
            let candidate =
                tone_owner_candidate_roi(snapshot, input.output, roi, 1.0).ok_or_else(|| {
                    "tone owner candidate ROI is outside the current canvas".to_string()
                })?;
            let reference = source_roi_for_quality(&source, roi, source_info, input.residual)
                .map_err(str::to_string)?;
            let baseline_mean = quality_gate::roi_mean_rgb(&baseline)
                .ok_or_else(|| "tone owner baseline ROI is empty".to_string())?;
            let candidate_mean = quality_gate::roi_mean_rgb(&candidate)
                .ok_or_else(|| "tone owner candidate ROI is empty".to_string())?;
            let reference_mean = quality_gate::roi_mean_rgb(&reference)
                .ok_or_else(|| "tone owner Source_RAW ROI is empty".to_string())?;
            let center = (
                roi.world_origin.0 + f64::from(roi.side) * 0.5,
                roi.world_origin.1 + f64::from(roi.side) * 0.5,
            );
            let (_, field_gain, global_gain, global_bias, tile_index) = snapshot
                .field_sample(owner, center.0, center.1)
                .unwrap_or(([0.0; 3], [1.0; 3], [1.0; 3], [0.0; 3], usize::MAX));
            let mut coverage_min = u8::MAX;
            let mut confidence_min = f32::INFINITY;
            let mut confidence_sum = 0.0;
            for y in roi.y..roi.y + roi.side {
                for x in roi.x..roi.x + roi.side {
                    let index = (y * input.coverage.width() + x) as usize;
                    coverage_min = coverage_min.min(input.coverage.as_raw()[index]);
                    confidence_min = confidence_min.min(input.confidence[index]);
                    confidence_sum += f64::from(input.confidence[index]);
                }
            }
            let samples = u64::from(roi.side) * u64::from(roi.side);
            let local_scale = quality_gate::local_scale_for_roi(
                roi,
                &source_info.geometry,
                input.residual,
                input.acceptance_render_scale,
            )
            .ok()
            .map(|measurement| measurement.median);
            let gradient_ratio_candidate_to_source = local_scale.and_then(|scale| {
                gradient_energy_ratio(&candidate, &reference, scale)
                    .ok()
                    .flatten()
                    .map(|measurement| measurement.0)
            });
            let noise_ratio_candidate_to_source =
                quality_gate::noise_sigma_ratio(&candidate, &reference).ok();
            result.push(ToneOwnerEvidence {
                partition,
                roi: roi.clone(),
                owner_path: source_info.path.clone(),
                station_id: source_info.geometry.station_id,
                tile_index,
                baseline_mean,
                candidate_mean,
                reference_mean,
                field_gain,
                global_gain,
                global_bias,
                coverage_min,
                confidence_min,
                confidence_mean: confidence_sum / samples.max(1) as f64,
                overlap_participation: snapshot.overlap_participation(center.0, center.1),
                fit_disjoint: (0..roi.side).all(|y| {
                    (0..roi.side).all(|x| {
                        !snapshot.is_fit_cell(
                            roi.world_origin.0 + f64::from(x),
                            roi.world_origin.1 + f64::from(y),
                        )
                    })
                }),
                gradient_ratio_candidate_to_source,
                noise_ratio_candidate_to_source,
            });
        }
    }
    result.sort_by(|left, right| {
        left.partition
            .cmp(right.partition)
            .then_with(|| left.roi.owner.cmp(&right.roi.owner))
            .then_with(|| left.roi.y.cmp(&right.roi.y))
            .then_with(|| left.roi.x.cmp(&right.roi.x))
    });
    Ok(result)
}

#[cfg(test)]
fn owner_delta_at(evidence: &ToneOwnerEvidence, relaxation: f64) -> f64 {
    let post = std::array::from_fn(|channel| {
        (f64::from(evidence.baseline_mean[channel])
            + relaxation
                * f64::from(evidence.candidate_mean[channel] - evidence.baseline_mean[channel]))
            as f32
    });
    super::tone::delta_e00_rgb(post, evidence.reference_mean)
}

#[cfg(test)]
fn owner_distribution_feasible(evidence: &[&ToneOwnerEvidence], relaxation: f64) -> bool {
    if evidence.is_empty() || evidence.iter().any(|item| !item.fit_disjoint) {
        return false;
    }
    let baseline = evidence
        .iter()
        .map(|item| owner_delta_at(item, 0.0))
        .collect::<Vec<_>>();
    let candidate = evidence
        .iter()
        .map(|item| owner_delta_at(item, relaxation))
        .collect::<Vec<_>>();
    tone_percentile(&candidate, 0.5) <= tone_percentile(&baseline, 0.5) + 1e-9
        && tone_percentile(&candidate, 0.95) <= tone_percentile(&baseline, 0.95) + 1e-9
        && tone_percentile(&candidate, 1.0) <= tone_percentile(&baseline, 1.0) + 1e-9
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct ToneJointRelaxation {
    selected: Option<f64>,
    minimum_boundary: Option<f64>,
    maximum_owner: Option<f64>,
    pareto_gap: Option<f64>,
}

#[cfg(test)]
fn select_joint_relaxation(boundary: &[f64], owner: &[f64]) -> ToneJointRelaxation {
    let selected = boundary.iter().copied().find(|alpha| {
        owner
            .binary_search_by(|value| value.total_cmp(alpha))
            .is_ok()
    });
    let minimum_boundary = boundary.first().copied();
    let maximum_owner = owner.last().copied();
    let pareto_gap = match (minimum_boundary, maximum_owner) {
        (Some(minimum), Some(maximum)) if minimum > maximum => Some(minimum - maximum),
        _ => None,
    };
    ToneJointRelaxation {
        selected,
        minimum_boundary,
        maximum_owner,
        pareto_gap,
    }
}

#[cfg(test)]
fn write_tone_owner_fidelity_oracle(
    input: &QualityGateInput<'_>,
    formal: &[QualityRoi],
    load_source: &mut SourceLoader<'_>,
) -> Result<(), String> {
    let Some(directory) = super::test_support::tone_owner_fidelity_dir() else {
        return Ok(());
    };
    let snapshot = super::tone::take_tone_owner_oracle_snapshot()
        .ok_or_else(|| "current-run tone owner oracle snapshot is unavailable".to_string())?;
    let holdouts = tone_owner_holdout_rois(input, formal, &snapshot);
    let evidence = collect_tone_owner_evidence(input, &snapshot, formal, &holdouts, load_source)?;
    let owner_holdout = evidence
        .iter()
        .filter(|item| item.partition == "owner_raw_fidelity_holdout")
        .collect::<Vec<_>>();
    let formal_evidence = evidence
        .iter()
        .filter(|item| item.partition == "formal_quality_gate_verification")
        .collect::<Vec<_>>();
    const STEPS: usize = 4096;
    let relaxations = (0..=STEPS)
        .map(|step| step as f64 / STEPS as f64)
        .collect::<Vec<_>>();
    let owner_feasible = relaxations
        .iter()
        .copied()
        .filter(|&alpha| owner_distribution_feasible(&owner_holdout, alpha))
        .collect::<Vec<_>>();
    let baseline_holdout = snapshot.checkerboard_holdout(0.0);
    let boundary_feasible = relaxations
        .iter()
        .copied()
        .filter(|&alpha| {
            let checkerboard = snapshot.checkerboard_holdout(alpha);
            snapshot.boundary_max(alpha) <= super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD
                && checkerboard.0 >= super::tone::TONE_MIN_SAMPLES
                && checkerboard.1 < baseline_holdout.1
                && checkerboard.2 < baseline_holdout.2
        })
        .collect::<Vec<_>>();
    let relaxation = select_joint_relaxation(&boundary_feasible, &owner_feasible);
    let selected = relaxation.selected;
    let applied_relaxation = selected.unwrap_or(0.0);
    let formal_values = formal_evidence
        .iter()
        .map(|item| owner_delta_at(item, applied_relaxation))
        .collect::<Vec<_>>();
    let formal_baseline = formal_evidence
        .iter()
        .map(|item| owner_delta_at(item, 0.0))
        .collect::<Vec<_>>();
    let formal_full_candidate = formal_evidence
        .iter()
        .map(|item| owner_delta_at(item, 1.0))
        .collect::<Vec<_>>();
    let formal_pass = selected.is_some()
        && formal_values.len() == formal.len()
        && formal_values
            .iter()
            .all(|value| *value <= quality_gate::QUALITY_ROI_DELTA_E00_MAX)
        && tone_percentile(&formal_values, 0.5) <= tone_percentile(&formal_baseline, 0.5) + 1e-9
        && tone_percentile(&formal_values, 0.95) <= tone_percentile(&formal_baseline, 0.95) + 1e-9;
    let candidate_full_high_frequency_pass = evidence.iter().all(|item| {
        item.gradient_ratio_candidate_to_source
            .is_none_or(|value| value >= quality_gate::QUALITY_GRADIENT_RATIO_MIN)
            && item.noise_ratio_candidate_to_source.is_none_or(|value| {
                (quality_gate::QUALITY_NOISE_RATIO_MIN..=quality_gate::QUALITY_NOISE_RATIO_MAX)
                    .contains(&value)
            })
    });
    let selected_high_frequency_pass = selected.is_none() || candidate_full_high_frequency_pass;
    let field_constraint_stats = snapshot.field_constraint_stats();
    let records = evidence
        .iter()
        .map(|item| {
            let selected_rgb = std::array::from_fn::<f32, 3, _>(|channel| {
                item.baseline_mean[channel]
                    + applied_relaxation as f32
                        * (item.candidate_mean[channel] - item.baseline_mean[channel])
            });
            serde_json::json!({
                "partition": item.partition,
                "world": { "x": item.roi.world_origin.0, "y": item.roi.world_origin.1, "side": item.roi.side },
                "owner": item.roi.owner,
                "station": item.station_id,
                "tile_index": item.tile_index,
                "spatial_stratum": [item.roi.x / 512, item.roi.y / 512],
                "owner_source_raw": item.owner_path,
                "source_provenance_count": 1,
                "mapping": "residual_warp_inverse_then_inverse(tile_to_world*member_to_anchor)",
                "sampler": "production_bounded_catmull_rom",
                "overlap_participation": item.overlap_participation,
                "fit_disjoint": item.fit_disjoint,
                "coverage_min": item.coverage_min,
                "confidence": { "minimum": item.confidence_min, "mean": item.confidence_mean },
                "field_gain": item.field_gain,
                "global_gain": item.global_gain,
                "global_bias": item.global_bias,
                "pre": tone_owner_distribution(item.baseline_mean),
                "candidate_full": tone_owner_distribution(item.candidate_mean),
                "selected": tone_owner_distribution(selected_rgb),
                "source_raw": tone_owner_distribution(item.reference_mean),
                "delta_e00": {
                    "pre": owner_delta_at(item, 0.0),
                    "candidate_full": owner_delta_at(item, 1.0),
                    "selected": owner_delta_at(item, applied_relaxation),
                },
                "high_frequency": {
                    "gradient_ratio_candidate_to_source_raw": item.gradient_ratio_candidate_to_source,
                    "noise_ratio_candidate_to_source_raw": item.noise_ratio_candidate_to_source,
                },
            })
        })
        .collect::<Vec<_>>();
    let checkerboard_selected = snapshot.checkerboard_holdout(applied_relaxation);
    let document = serde_json::json!({
        "schema": 1,
        "source": "current_run_only",
        "previous_quality_gate_results_read": false,
        "production_spatial_field_enabled": super::tone::production_spatial_field_enabled(),
        "evidence": records,
        "partitions": {
            "overlap_fit_cells": snapshot.fit_cell_count(),
            "overlap_checkerboard_holdout_samples": baseline_holdout.0,
            "owner_raw_fidelity_holdout_samples": owner_holdout.len(),
            "formal_quality_gate_rois": formal_evidence.len(),
            "owner_holdout_disjoint_from_fit": owner_holdout.iter().all(|item| item.fit_disjoint),
        },
        "oracle": {
            "family": "bounded_shared_world_field_linear_relaxation",
            "constraints": {
                "field_gain_minimum": field_constraint_stats.0,
                "field_gain_maximum": field_constraint_stats.1,
                "maximum_neighbour_log_gain_delta": field_constraint_stats.2,
                "smoothing": { "kernel": "gaussian", "sigma_cells": super::tone::TONE_LOW_FREQUENCY_SIGMA },
                "graph": "zero_mean_joint_accepted_relation_solve",
                "owner_holdout_distribution_not_worse": ["median", "p95", "max"],
                "formal_roi_delta_e00_threshold_unchanged": quality_gate::QUALITY_ROI_DELTA_E00_MAX,
            },
            "relaxation_resolution": 1.0 / STEPS as f64,
            "selected_relaxation": selected,
            "identity_fallback": selected.is_none(),
            "boundary": {
                "windows": snapshot.boundary_window_count(),
                "baseline_max_delta_e00": snapshot.boundary_max(0.0),
                "selected_max_delta_e00": snapshot.boundary_max(applied_relaxation),
                "threshold": super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD,
                "minimum_feasible_relaxation": relaxation.minimum_boundary,
            },
            "owner_fidelity": {
                "maximum_feasible_relaxation": relaxation.maximum_owner,
                "threshold": quality_gate::QUALITY_ROI_DELTA_E00_MAX,
                "baseline_median": tone_percentile(&formal_baseline, 0.5),
                "baseline_p95": tone_percentile(&formal_baseline, 0.95),
                "baseline_max": tone_percentile(&formal_baseline, 1.0),
                "candidate_full_median": tone_percentile(&formal_full_candidate, 0.5),
                "candidate_full_p95": tone_percentile(&formal_full_candidate, 0.95),
                "candidate_full_max": tone_percentile(&formal_full_candidate, 1.0),
                "candidate_full_violations": formal_full_candidate.iter().filter(|value| **value > quality_gate::QUALITY_ROI_DELTA_E00_MAX).count(),
                "selected_median": tone_percentile(&formal_values, 0.5),
                "selected_p95": tone_percentile(&formal_values, 0.95),
                "selected_max": formal_values.iter().copied().fold(0.0, f64::max),
                "formal_pass": formal_pass,
            },
            "checkerboard_holdout": {
                "samples": checkerboard_selected.0,
                "baseline_median": baseline_holdout.1,
                "baseline_p95": baseline_holdout.2,
                "selected_median": checkerboard_selected.1,
                "selected_p95": checkerboard_selected.2,
            },
            "cycle_residual_max_log": snapshot.cycle_residual_max(),
            "candidate_full_high_frequency_pass": candidate_full_high_frequency_pass,
            "selected_high_frequency_pass": selected_high_frequency_pass,
            "feasible_joint_solution": selected.is_some() && formal_pass && candidate_full_high_frequency_pass,
            "pareto_gap": relaxation.pareto_gap,
        },
    });
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create tone owner fidelity directory: {error}"))?;
    let path = directory.join("tone-owner-source-fidelity.json");
    let file = std::fs::File::create(&path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &document)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

pub(super) fn record_unmeasurable(
    criterion: &mut Criterion,
    records: &mut Vec<UnmeasurableRecord>,
    roi: &QualityRoi,
    reason: &'static str,
) {
    let category = classify_unmeasurable_reason(reason);
    criterion.miss(reason);
    records.push(UnmeasurableRecord {
        criterion: criterion.name.to_string(),
        world: WorldPoint {
            x: roi.world_origin.0,
            y: roi.world_origin.1,
        },
        reason: reason.to_string(),
        category,
    });
}

pub(crate) fn classify_unmeasurable_reason(reason: &str) -> UnmeasurableCategory {
    match reason {
        degradation::ROI_NOT_SLANTED_EDGE
        | degradation::SLANTED_EDGE_ANGLE_OUT_OF_RANGE
        | degradation::SLANTED_EDGE_CONTRAST_INSUFFICIENT
        | degradation::SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED
        | degradation::SLANTED_EDGE_TOO_SHORT
        | degradation::ROI_NOT_FLAT
        | degradation::BOUNDARY_NO_PAIRABLE_EDGE
        | degradation::BOUNDARY_LOW_CONTRAST
        | degradation::BOUNDARY_STROKE_NOT_UNIQUE
        | degradation::BOUNDARY_ORIENTATION_MISMATCH => UnmeasurableCategory::ContentNotApplicable,
        _ => UnmeasurableCategory::Technical,
    }
}

fn ratio(measured: f64, reference: f64, min: f64, max: f64) -> Option<(f64, bool)> {
    quality_gate::compare_ratio(measured, reference, min, max, false).map(|v| (v.value, v.passed))
}

fn gradient_energy_ratio(
    output: &Rgb32FImage,
    reference: &Rgb32FImage,
    local_scale: f64,
) -> Result<Option<(f64, bool)>, &'static str> {
    let output_energy = quality_gate::normalized_gradient_energy(output, local_scale)?;
    // `source_roi` has already resampled the Source_RAW onto the output pixel
    // grid. It therefore has the same local scale as `output`; treating this
    // aligned reference as native-scale evidence divides only the output side
    // and turns an identity comparison into exactly `1 / local_scale`.
    let reference_energy = quality_gate::normalized_gradient_energy(reference, local_scale)?;
    Ok(ratio(
        output_energy,
        reference_energy,
        quality_gate::QUALITY_GRADIENT_RATIO_MIN,
        f64::INFINITY,
    ))
}

#[cfg(test)]
#[derive(serde::Serialize)]
struct GradientOracleCandidate {
    source: String,
    station_id: usize,
    local_scale: f64,
    residual_reference_energy: f64,
    identity_reference_energy: Option<f64>,
}

#[cfg(test)]
#[derive(serde::Serialize)]
struct GradientOracleRoi {
    world: [f64; 2],
    owner_source: String,
    station_id: usize,
    output_energy: f64,
    candidates: Vec<GradientOracleCandidate>,
}

#[cfg(test)]
fn write_gradient_source_oracle(
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
fn write_effective_coverage_diagnostics(
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn merged_quality_reference_prefers_direct_raw_provenance() {
        let image =
            Rgb32FImage::from_fn(64, 64, |x, y| Rgb([x as f32 / 64.0, y as f32 / 64.0, 0.25]));
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 4,
            region_label: 1,
            owner: 1,
            world_origin: (20.0, 20.0),
            boundary_clearance: 16.0,
        };
        let source = QualitySource {
            owner: 1,
            path: "merged-reference.raw".into(),
            dimensions: image.dimensions(),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 0,
            },
            native_warp: Some(NativeSourceWarp {
                displacement: None,
                source_to_station: Matrix3::identity(),
                station_to_world: Matrix3::identity(),
            }),
            native_registration: Some(NativeReferenceMap {
                world_to_station: Matrix3::identity(),
                station_to_world: Matrix3::identity(),
                station_origin: (0.0, 0.0),
                refinement: crate::panorama_utils::stitching::FocusSourceRefinement {
                    translation: (3, -2),
                    ..Default::default()
                },
            }),
        };
        let residual = ResidualWarp::new();
        let reference = source_roi_for_quality(&image, &roi, &source, &residual).unwrap();
        for y in 0..roi.side {
            for x in 0..roi.side {
                let pixel = reference.get_pixel(x, y);
                assert!((pixel[0] - (17.0 + x as f32) / 64.0).abs() < 1e-6);
                assert!((pixel[1] - (22.0 + y as f32) / 64.0).abs() < 1e-6);
            }
        }
        let legacy = QualitySource {
            native_registration: None,
            ..source
        };
        let reference = source_roi_for_quality(&image, &roi, &legacy, &residual).unwrap();
        assert!((reference.get_pixel(0, 0)[0] - 20.0 / 64.0).abs() < 1e-6);
    }

    fn synthetic_owner_evidence(
        baseline: [f32; 3],
        candidate: [f32; 3],
        reference: [f32; 3],
        fit_disjoint: bool,
    ) -> ToneOwnerEvidence {
        ToneOwnerEvidence {
            partition: "owner_raw_fidelity_holdout",
            roi: QualityRoi {
                x: 0,
                y: 0,
                side: 64,
                region_label: 0,
                owner: 1,
                world_origin: (0.0, 0.0),
                boundary_clearance: 0.0,
            },
            owner_path: "synthetic.raw".to_string(),
            station_id: 0,
            tile_index: 0,
            baseline_mean: baseline,
            candidate_mean: candidate,
            reference_mean: reference,
            field_gain: [1.0; 3],
            global_gain: [1.0; 3],
            global_bias: [0.0; 3],
            coverage_min: 255,
            confidence_min: 1.0,
            confidence_mean: 1.0,
            overlap_participation: false,
            fit_disjoint,
            gradient_ratio_candidate_to_source: Some(1.0),
            noise_ratio_candidate_to_source: Some(1.0),
        }
    }

    #[test]
    fn owner_reference_uses_the_rendering_kernel_for_fractional_fine_texture() {
        let source = Rgb32FImage::from_fn(32, 32, |x, y| {
            Rgb([if (x + y) % 2 == 0 { 0.15 } else { 0.85 }; 3])
        });
        let geometry = SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::new(1.0, 0.0, -0.37, 0.0, 1.0, 0.23, 0.0, 0.0, 1.0),
            station_id: 0,
        };
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 4,
            region_label: 1,
            owner: 1,
            world_origin: (10.0, 10.0),
            boundary_clearance: 8.0,
        };
        let sampled =
            source_roi_with_registration(&source, &roi, &geometry, &ResidualWarp::new(), None)
                .unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let expected =
                    crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                        &source,
                        10.37 + f64::from(x),
                        9.77 + f64::from(y),
                    );
                assert!((sampled.get_pixel(x, y)[0] - expected[0]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn owner_fidelity_rejects_joint_colour_shift_and_fit_leakage() {
        let shifted = synthetic_owner_evidence([0.4; 3], [0.5; 3], [0.4; 3], true);
        assert!(owner_distribution_feasible(&[&shifted], 0.0));
        assert!(!owner_distribution_feasible(&[&shifted], 1.0));

        let leaked = synthetic_owner_evidence([0.45; 3], [0.4; 3], [0.4; 3], false);
        assert!(!owner_distribution_feasible(&[&leaked], 1.0));
    }

    #[test]
    fn joint_oracle_recovers_feasible_solution_or_certifies_pareto_gap() {
        let recovered = select_joint_relaxation(&[0.5, 0.75, 1.0], &[0.0, 0.25, 0.5, 0.75]);
        assert_eq!(recovered.selected, Some(0.5));
        assert_eq!(recovered.pareto_gap, None);

        let impossible = select_joint_relaxation(&[0.75, 1.0], &[0.0, 0.25, 0.5]);
        assert_eq!(impossible.selected, None);
        assert_eq!(impossible.minimum_boundary, Some(0.75));
        assert_eq!(impossible.maximum_owner, Some(0.5));
        assert_eq!(impossible.pareto_gap, Some(0.25));
        assert_eq!(impossible.selected.unwrap_or(0.0), 0.0);
    }

    #[test]
    fn low_frequency_candidate_preserves_high_frequency_gradient_and_noise() {
        let baseline = Rgb32FImage::from_fn(64, 64, |x, y| {
            let fine = if (x + y).is_multiple_of(2) {
                0.03
            } else {
                -0.03
            };
            let broad = 0.25 + 0.003 * x as f32 + 0.002 * y as f32;
            Rgb([broad + fine, broad * 0.9 + fine, broad * 0.8 + fine])
        });
        let candidate = Rgb32FImage::from_fn(64, 64, |x, y| {
            let pixel = baseline.get_pixel(x, y).0;
            let low = 0.02 + 0.0001 * x as f32;
            Rgb(pixel.map(|value| value + low))
        });
        let gradient = quality_gate::normalized_gradient_energy(&candidate, 1.0).unwrap()
            / quality_gate::normalized_gradient_energy(&baseline, 1.0).unwrap();
        let noise = quality_gate::noise_sigma_ratio(&candidate, &baseline).unwrap();
        assert!(gradient >= quality_gate::QUALITY_GRADIENT_RATIO_MIN);
        assert!(
            (quality_gate::QUALITY_NOISE_RATIO_MIN..=quality_gate::QUALITY_NOISE_RATIO_MAX)
                .contains(&noise)
        );
    }

    #[test]
    fn owner_oracle_has_no_previous_quality_gate_list_input() {
        let source = include_str!("quality_gate_runner.rs");
        let start = source.find("fn write_tone_owner_fidelity_oracle").unwrap();
        let end = source[start..]
            .find("pub(super) fn record_unmeasurable")
            .map(|offset| start + offset)
            .unwrap();
        let oracle = &source[start..end];
        assert!(!oracle.contains("std::fs::read"));
        assert!(!oracle.contains("GRADIENT_ORACLE_REPORT"));
        assert!(!oracle.contains("fixture"));
        assert!(oracle.contains("current-run tone owner oracle snapshot"));
    }

    #[test]
    fn owner_reference_replays_foreground_native_field_and_translation_in_render_order() {
        use crate::panorama_utils::{
            mosaic::{Field, NativeResidualDisplacement},
            stitching::{FocusForegroundMove, FocusSourceRefinement},
        };
        let source = Rgb32FImage::from_fn(128, 128, |x, y| {
            Rgb([x as f32 / 128.0, y as f32 / 128.0, 0.25])
        });
        let station_to_world = Matrix3::new(1.0, 0.0, 100.0, 0.0, 1.0, 200.0, 0.0, 0.0, 1.0);
        let mut field = Field::<2>::new(128, 128, 16.0);
        field.values.fill([1.5, -0.5]);
        let native = NativeReferenceMap {
            world_to_station: station_to_world.try_inverse().unwrap(),
            station_to_world,
            station_origin: (0.0, 0.0),
            refinement: FocusSourceRefinement {
                translation: (3, -2),
                displacement: Some(NativeResidualDisplacement::from_test_field(field)),
                foreground_move: Some(FocusForegroundMove {
                    delta: (4, 0),
                    left: 0,
                    top: 0,
                    destinations: GrayImage::from_pixel(128, 128, Luma([255])),
                }),
                source_sampling: None,
            },
        };
        let geometry = SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: station_to_world,
            station_id: 0,
        };
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 4,
            region_label: 1,
            owner: 1,
            world_origin: (112.0, 224.0),
            boundary_clearance: 8.0,
        };
        let actual = source_roi_with_registration(
            &source,
            &roi,
            &geometry,
            &ResidualWarp::new(),
            Some(&native),
        )
        .unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let pixel = actual.get_pixel(x, y);
                assert!((pixel[0] - (6.5 + x as f32) / 128.0).abs() < 1e-6);
                assert!((pixel[1] - (25.5 + y as f32) / 128.0).abs() < 1e-6);
            }
        }
        let wrong =
            source_roi_with_registration(&source, &roi, &geometry, &ResidualWarp::new(), None)
                .unwrap();
        assert!((wrong.get_pixel(0, 0)[0] - actual.get_pixel(0, 0)[0]).abs() > 0.04);
    }

    /// 需求 11.12 judges `shortfall > 0.05` on quantised planes; the
    /// quantisation must keep that boundary exact, and the cell definition
    /// must give no shortfall to the best owner or to a lone candidate.
    #[test]
    fn owner_shortfall_quantisation_keeps_the_criterion_boundary() {
        let limit = owner_shortfall_limit();
        assert_eq!(limit, 5);
        assert_eq!(owner_disagreement_veto_limit(), 20);
        assert_eq!(quantize_hundredths(0.05), limit);
        assert!(quantize_hundredths(0.0501) > limit);
        assert!(quantize_hundredths(0.049) <= limit);
        assert_eq!(quantize_hundredths(0.0), 0);
        assert_eq!(quantize_hundredths(1.0), 100);
        assert_eq!(quantize_hundredths(f32::NAN), OWNER_EVIDENCE_UNKNOWN);
        assert_eq!(quantize_hundredths(f32::INFINITY), OWNER_EVIDENCE_UNKNOWN);

        let best_owner = super::super::focus_fuser::SharpnessCellEvidence {
            winner_score: 0.60,
            runner_up_score: 0.50,
            owner_score: 0.60,
            candidate_count: 2,
            disagreement: 0.0,
        };
        assert_eq!(best_owner.owner_shortfall(), 0.0);
        let lone_candidate = super::super::focus_fuser::SharpnessCellEvidence {
            owner_score: 0.30,
            candidate_count: 1,
            ..best_owner
        };
        assert_eq!(lone_candidate.owner_shortfall(), 0.0);
        let weaker_owner = super::super::focus_fuser::SharpnessCellEvidence {
            owner_score: 0.54,
            ..best_owner
        };
        assert!((weaker_owner.owner_shortfall() - 0.10).abs() < 1.0e-6);
        assert!(quantize_hundredths(weaker_owner.owner_shortfall()) > limit);
    }

    #[test]
    fn reference_roi_reuses_production_sampler_for_gradient_and_noise_metrics() {
        let source = Rgb32FImage::from_fn(12, 12, |x, y| {
            let value = ((x * x + 5 * x * y + 3 * y + 7) % 23) as f32 / 22.0;
            Rgb([value, value * 0.91, value * 0.83])
        });
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 4,
            region_label: 1,
            owner: 1,
            world_origin: (2.25, 2.5),
            boundary_clearance: 16.0,
        };
        let geometry = SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 0,
        };
        let reference =
            source_roi(&source, &roi, &geometry, &ResidualWarp::new()).expect("identity ROI");
        let source_info = QualitySource {
            native_registration: None,
            owner: 1,
            path: "owner.raw".to_string(),
            geometry: geometry.clone(),
            dimensions: source.dimensions(),
            native_warp: None,
        };
        assert_eq!(
            sample_quality_source_at_world(
                &source,
                Point2::new(roi.world_origin.0, roi.world_origin.1),
                &source_info,
                &ResidualWarp::new(),
            )
            .unwrap(),
            *reference.get_pixel(0, 0),
        );
        let expected = crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            &source, 2.25, 2.5,
        );
        assert_eq!(reference.get_pixel(0, 0), &expected);

        let x0 = 2u32;
        let y0 = 2u32;
        let tx = 0.25f32;
        let ty = 0.5f32;
        let a = *source.get_pixel(x0, y0);
        let b = *source.get_pixel(x0 + 1, y0);
        let c = *source.get_pixel(x0, y0 + 1);
        let d = *source.get_pixel(x0 + 1, y0 + 1);
        let bilinear = Rgb(std::array::from_fn(|channel| {
            let top = a[channel] * (1.0 - tx) + b[channel] * tx;
            let bottom = c[channel] * (1.0 - tx) + d[channel] * tx;
            top * (1.0 - ty) + bottom * ty
        }));
        assert!(
            (reference.get_pixel(0, 0)[0] - bilinear[0]).abs() > 1.0e-6,
            "reference ROI must not silently fall back to the old bilinear kernel"
        );

        let output = Rgb32FImage::from_fn(roi.side, roi.side, |x, y| {
            crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                &source,
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            )
        });
        let local_scale = 1.125;
        let (gradient_ratio, gradient_passes) =
            gradient_energy_ratio(&output, &reference, local_scale)
                .expect("the sampled ROIs must remain measurable")
                .expect("the reference gradient must remain non-zero");
        assert!((gradient_ratio - 1.0).abs() <= 1.0e-12);
        assert!(gradient_passes);
        let noise_ratio = quality_gate::noise_sigma_ratio(&output, &reference)
            .expect("identical production and reference sampling must have measurable noise");
        assert!((noise_ratio - 1.0).abs() <= 1.0e-12);
    }

    #[test]
    fn owner_fidelity_mapping_composes_member_tile_and_requires_one_source() {
        let source = Rgb32FImage::from_fn(32, 32, |x, y| {
            let value = (x + 3 * y) as f32 / 124.0;
            Rgb([value, value * 0.9, value * 0.8])
        });
        let translate = |x: f64, y: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0);
        let geometry = SourceGeometry {
            member_to_anchor: translate(5.0, 7.0),
            tile_to_world: translate(11.0, 13.0),
            station_id: 4,
        };
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 2,
            region_label: 1,
            owner: 9,
            world_origin: (18.25, 23.5),
            boundary_clearance: 16.0,
        };
        let sampled = source_roi(&source, &roi, &geometry, &ResidualWarp::new()).unwrap();
        let expected = crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            &source, 2.25, 3.5,
        );
        assert_eq!(sampled.get_pixel(0, 0), &expected);

        let info = QualitySource {
            native_registration: None,
            owner: 9,
            path: "owner.raw".to_string(),
            geometry,
            dimensions: source.dimensions(),
            native_warp: None,
        };
        assert_eq!(
            unique_owner_source(std::slice::from_ref(&info), 9)
                .unwrap()
                .path,
            "owner.raw"
        );
        assert!(unique_owner_source(&[info.clone(), info], 9).is_err());
        assert!(unique_owner_source(&[], 9).is_err());
    }

    #[test]
    fn source_roi_mapping_guard_requires_render_provenance_for_focus_band_and_native_delta() {
        // The production Quality_Gate mapping is intentionally only the
        // world -> Residual_Warp inverse -> member/tile inverse path.  A
        // focus render can apply two additional mappings after that path:
        // a local band homography and the accepted native displacement field.
        // Keep a coordinate-coded source so a future provenance integration
        // can compare the exact sample, rather than only comparing geometry
        // matrices.
        let source = Rgb32FImage::from_fn(48, 48, |x, y| {
            Rgb([x as f32 / 48.0, y as f32 / 48.0, (x + y) as f32 / 96.0])
        });
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 2,
            region_label: 1,
            owner: 1,
            world_origin: (10.25, 11.5),
            boundary_clearance: 16.0,
        };
        let geometry = SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 10,
        };
        let qg_reference = source_roi(&source, &roi, &geometry, &ResidualWarp::new())
            .expect("identity world geometry must remain measurable");

        // `render_focus_layer` stores a source->world local band homography;
        // translation(+2, 0) therefore maps a world sample back to source x-2.
        // `warp_rendered_focus_layer` samples the candidate at +1 canvas pixel
        // for this synthetic native field, yielding a final source x-1.  These
        // values model the station 10/12 issue without adding a second,
        // production mapping implementation to the Quality_Gate.
        let world = Point2::new(roi.world_origin.0, roi.world_origin.1);
        let expected_with_render_provenance =
            crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                &source,
                world.x - 1.0,
                world.y,
            );
        let qg_pixel = qg_reference.get_pixel(0, 0);
        assert_ne!(qg_pixel, &expected_with_render_provenance);
        assert!(
            (qg_pixel[0] - expected_with_render_provenance[0]).abs() > 1.0e-5,
            "the current QG reference must not silently claim local-band/native provenance"
        );
    }

    #[test]
    fn aligned_gradient_ratio_is_invariant_to_shared_local_scale() {
        let output = Rgb32FImage::from_fn(16, 16, |x, y| {
            let value = ((x * x + 3 * y + x * y) % 29) as f32 / 28.0;
            Rgb([value, value * 0.9, value * 0.8])
        });
        let reference = Rgb32FImage::from_fn(16, 16, |x, y| {
            let value = ((2 * x * x + y + x * y) % 31) as f32 / 30.0;
            Rgb([value, value * 0.9, value * 0.8])
        });
        let unit = gradient_energy_ratio(&output, &reference, 1.0)
            .unwrap()
            .unwrap()
            .0;
        for local_scale in [0.8, 1.125] {
            let scaled = gradient_energy_ratio(&output, &reference, local_scale)
                .unwrap()
                .unwrap()
                .0;
            assert!((scaled - unit).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn confidence_distribution_accumulates_compact_cell_evidence() {
        let mut result = super::super::report::ConfidenceScoreDistribution {
            winner_histogram: vec![0; 21],
            runner_up_histogram: vec![0; 21],
            ..Default::default()
        };
        let evidence = super::super::focus_fuser::SharpnessCellEvidence {
            winner_score: 0.60,
            runner_up_score: 0.59,
            owner_score: 0.60,
            candidate_count: 2,
            disagreement: 0.0,
        };
        accumulate_confidence_score(&mut result, 255, 0.01, 255, evidence);
        accumulate_confidence_score(&mut result, 255, 0.05, 255, evidence);
        accumulate_confidence_score(&mut result, 255, 0.01, 0, evidence);
        assert_eq!(result.textured_low_count, 1);
        assert_eq!(result.both_scores_at_least_0_10, 1);
        assert_eq!(result.winner_below_0_10, 0);
        assert_eq!(result.winner_histogram[12], 1);
        assert_eq!(result.runner_up_histogram[11], 1);
        assert!((result.winner_sum - 0.60).abs() < 1e-6);
        assert!((result.runner_up_sum - 0.59).abs() < 1e-6);
    }

    #[test]
    fn textured_partition_excludes_transparency_and_keeps_the_fixed_confidence_boundary() {
        let coverage = GrayImage::from_raw(6, 1, vec![255, 255, 255, 255, 0, 0]).unwrap();
        let confidence = [0.049, 0.05, 0.0, 0.8, 0.0, 0.0];
        let textured = [255, 255, 0, 0, 255, 0];
        let stats = confidence_coverage_stats(&coverage, &confidence, &textured).unwrap();
        assert_eq!(stats.textured_pixels, 2);
        assert_eq!(stats.flat_pixels, 2);
        assert_eq!(stats.textured_low, 1, "0.05 itself is not low confidence");
        assert_eq!(stats.flat_low, 1);
        assert_eq!(stats.textured_unknown, 0);
        assert_eq!(stats.flat_unknown, 0);
    }

    #[test]
    fn confidence_stats_report_all_flat_pixels_and_exclude_nan_from_ratios() {
        let coverage = GrayImage::from_raw(5, 1, vec![255; 5]).unwrap();
        let confidence = [f32::NAN, 0.02, f32::INFINITY, 0.049, 0.05];
        let textured = [0, 0, 0, 0, 0];
        let stats = confidence_coverage_stats(&coverage, &confidence, &textured).unwrap();
        assert_eq!(stats.textured_pixels, 0);
        assert_eq!(stats.flat_pixels, 5);
        assert_eq!(stats.flat_low, 2);
        assert_eq!(stats.flat_unknown, 2);
        assert_eq!(stats.textured_unknown, 0);

        let mixed =
            confidence_coverage_stats(&coverage, &confidence, &[255, 255, 0, 0, 0]).unwrap();
        assert_eq!(mixed.textured_pixels, 2);
        assert_eq!(mixed.textured_low, 1);
        assert_eq!(mixed.textured_unknown, 1);
    }

    #[test]
    fn texture_evidence_uses_all_covered_candidates_and_the_inclusive_sharpness_floor() {
        use super::super::focus_fuser::{CandidateCells, StationFusion};

        let mut fusion = StationFusion::new(24, 8, 2);
        assert_eq!(fusion.geometry().cell_count(), 3);
        fusion.seed(0, &[0.099, 0.10, 0.0], &[true, true, true]);
        assert_eq!(fusion.textured_cells(), vec![false, true, false]);
        let mut candidate = CandidateCells::new(3);
        candidate.sharpness = vec![0.10, 0.05, 0.9];
        candidate.covered = vec![true, true, false];
        fusion.fold(1, &candidate);
        assert_eq!(fusion.textured_cells(), vec![true, true, false]);
    }

    #[test]
    fn missing_local_scale_never_supplies_unit_scale_to_mtf_or_gradient() {
        let output = Rgb32FImage::from_pixel(1024, 1024, Rgb([0.5; 3]));
        let coverage = GrayImage::from_pixel(1024, 1024, Luma([255]));
        let ownership = vec![1; 1024 * 1024];
        let confidence = vec![0.8; 1024 * 1024];
        let textured = vec![255; 1024 * 1024];
        let residual = ResidualWarp::new();
        let sources = [QualitySource {
            owner: 1,
            path: "synthetic.raw".to_string(),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 0,
            },
            dimensions: (1024, 1024),
            native_warp: None,
            native_registration: None,
        }];
        let input = QualityGateInput {
            output: &output,
            coverage: &coverage,
            ownership: &ownership,
            confidence: &confidence,
            textured: Some(&textured),
            owner_shortfall: None,
            owner_disagreement: None,
            unresolved_pixel_count: 0,
            world_origin: (0.0, 0.0),
            sources: &sources,
            accepted_station_relations: &[],
            residual: &residual,
            acceptance_render_scale: f64::NAN,
            final_sharpen_amount: 0.0,
        };
        let mut decodes = 0;
        let report = run_quality_gate(&input, &mut |_| {
            decodes += 1;
            Ok(output.clone())
        });
        assert!(
            report.roi_count > 0,
            "the fixture must select an actual ROI"
        );
        assert_eq!(decodes, 1);
        for name in ["mtf50_normalized", "gradient_energy_normalized"] {
            let criterion = report
                .criteria
                .iter()
                .find(|item| item.name == name)
                .unwrap();
            assert_eq!(criterion.measurable_count, 0);
            assert_eq!(criterion.unmeasurable_count, report.roi_count);
            assert_eq!(
                criterion
                    .unmeasurable_reasons
                    .get(degradation::LOCAL_SCALE_UNMEASURABLE),
                Some(&report.roi_count),
            );
            assert!(criterion.measured.is_empty());
            assert!(criterion.failed.is_empty());
        }
        assert_eq!(
            report
                .unmeasurable
                .iter()
                .filter(|item| item.reason == degradation::LOCAL_SCALE_UNMEASURABLE)
                .count(),
            2 * report.roi_count,
        );
    }

    #[test]
    fn world_resampled_reference_uses_the_same_scale_and_still_detects_blur() {
        let _scope = degradation::begin_run_scope();
        // `source_roi` replays the run's recorded tone transform. A previous
        // test that ran the tone harmonizer inside its own run scope leaves
        // that report behind, which would turn this identity comparison into
        // a tone mismatch; production resets it at every run start.
        super::super::tone::reset_run_records();
        let source = Rgb32FImage::from_fn(1024, 1024, |x, y| {
            let value = 0.42 + 0.12 * (x as f32 * 0.27).sin() + 0.08 * (y as f32 * 0.19).cos();
            Rgb([value; 3])
        });
        let scale = 1.2;
        let exact = Rgb32FImage::from_fn(1024, 1024, |x, y| {
            crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                &source,
                f64::from(x) / scale,
                f64::from(y) / scale,
            )
        });
        let blurred = image::imageops::blur(&exact, 3.0);
        let coverage = GrayImage::from_pixel(1024, 1024, Luma([255]));
        let ownership = vec![1; 1024 * 1024];
        let confidence = vec![0.8; 1024 * 1024];
        let textured = vec![255; 1024 * 1024];
        let residual = ResidualWarp::new();
        let sources = [QualitySource {
            native_warp: None,
            owner: 1,
            path: "world-grid.raw".to_string(),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0),
                station_id: 0,
            },
            dimensions: source.dimensions(),
            native_registration: None,
        }];
        for (output, expect_pass) in [(&exact, true), (&blurred, false)] {
            let input = QualityGateInput {
                output,
                coverage: &coverage,
                ownership: &ownership,
                confidence: &confidence,
                textured: Some(&textured),
                owner_shortfall: None,
                owner_disagreement: None,
                unresolved_pixel_count: 0,
                world_origin: (0.0, 0.0),
                sources: &sources,
                accepted_station_relations: &[],
                residual: &residual,
                acceptance_render_scale: 1.0,
                final_sharpen_amount: 0.0,
            };
            let report = run_quality_gate(&input, &mut |_| Ok(source.clone()));
            let gradient = report
                .criteria
                .iter()
                .find(|criterion| criterion.name == "gradient_energy_normalized")
                .unwrap();
            assert!(gradient.measurable_count >= 8);
            if expect_pass {
                assert!(
                    gradient.failed.is_empty(),
                    "identical world samples lost detail"
                );
                assert!(
                    gradient
                        .measured
                        .iter()
                        .all(|m| (m.measured - 1.0).abs() < 1e-6)
                );
            } else {
                assert!(
                    !gradient.failed.is_empty(),
                    "actual blur must still fail the unchanged gate"
                );
            }
        }
    }

    #[test]
    fn quality_gate_record_mode_preserves_output_and_records_all_criteria() {
        // The gate is observation-only at this boundary: it may return a
        // conclusion and diagnostics, but it cannot rewrite the composed
        // pixels or add a publication-blocking degradation of its own.
        let _run_scope = degradation::begin_run_scope();
        degradation::reset_run_ledger();
        let output = Rgb32FImage::from_pixel(1024, 1024, Rgb([0.4, 0.45, 0.5]));
        let before = output.clone();
        let coverage = GrayImage::from_pixel(1024, 1024, Luma([255]));
        let ownership = vec![1u16; 1024 * 1024];
        let confidence = vec![0.8f32; 1024 * 1024];
        let textured = vec![255u8; 1024 * 1024];
        let shortfall = vec![5u8; 1024 * 1024];
        let disagreement = vec![0u8; 1024 * 1024];
        let residual = ResidualWarp::new();
        let sources = [QualitySource {
            owner: 1,
            path: "record-only.raw".to_string(),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 0,
            },
            dimensions: (1024, 1024),
            native_warp: None,
            native_registration: None,
        }];
        let input = QualityGateInput {
            output: &output,
            coverage: &coverage,
            ownership: &ownership,
            confidence: &confidence,
            textured: Some(&textured),
            owner_shortfall: Some(&shortfall),
            owner_disagreement: Some(&disagreement),
            unresolved_pixel_count: 0,
            world_origin: (0.0, 0.0),
            sources: &sources,
            accepted_station_relations: &[],
            residual: &residual,
            acceptance_render_scale: 1.0,
            final_sharpen_amount: 0.0,
        };
        let report = run_quality_gate(&input, &mut |_source| Ok(output.clone()));

        assert_eq!(
            output, before,
            "record-only Quality_Gate must not rewrite pixels"
        );
        assert_eq!(report.criteria.len(), quality_gate::QUALITY_CRITERIA.len());
        assert!(
            report
                .criteria
                .iter()
                .all(|criterion| criterion.threshold.is_finite())
        );
        let effective = report
            .criteria
            .iter()
            .find(|criterion| criterion.name == "effective_pixel_count")
            .expect("effective-pixel criterion must be reported");
        assert_eq!(effective.effective_output_pixels, Some(1024 * 1024));
        // `source_quad` describes the inclusive last source sample at
        // `width - 1`/`height - 1`; the half-open scanline union therefore
        // covers 1023×1023 projected cells for this identity fixture.
        assert_eq!(effective.effective_projected_pixels, Some(1023 * 1023));
        // Record-only means the gate writes no ledger entry of its own.
        assert!(degradation::run_ledger_snapshot().entries().is_empty());
    }

    #[test]
    fn failed_quality_gate_is_promoted_to_export_rejection_at_boundary() {
        let _run_scope = degradation::begin_run_scope();
        degradation::reset_run_ledger();
        let mut report = QualityGateReport {
            verdict: QualityGateVerdict::Fail,
            roi_count: 1,
            ..QualityGateReport::default()
        };
        report.criteria.push(QualityGateCriterionRecord {
            name: "owner_sharpness_coverage".to_string(),
            verdict: QualityGateVerdict::Fail,
            measurable_count: 1,
            ..QualityGateCriterionRecord::default()
        });

        record_gate_outcome(&report);

        let ledger = degradation::run_ledger_snapshot();
        assert_eq!(
            ledger.count_of(degradation::QUALITY_GATE_CRITERION_FAILED),
            1
        );
        assert_eq!(ledger.entries()[0].detail["roi_count"], 1);
        assert_eq!(
            ledger.entries()[0].detail["criteria"][0]["name"],
            "owner_sharpness_coverage"
        );
    }

    #[test]
    fn unmeasurable_reason_histograms_match_the_coordinate_records() {
        let roi = QualityRoi {
            x: 0,
            y: 0,
            side: 512,
            region_label: 1,
            owner: 1,
            world_origin: (-15.0, 72.0),
            boundary_clearance: 16.0,
        };
        let mut criterion = Criterion::new("mtf50_normalized", 0.93);
        let mut records = Vec::new();
        for reason in [
            degradation::SLANTED_EDGE_TOO_SHORT,
            degradation::SLANTED_EDGE_TOO_SHORT,
            degradation::SLANTED_EDGE_CONTRAST_INSUFFICIENT,
            degradation::SLANTED_EDGE_ANGLE_OUT_OF_RANGE,
            degradation::SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED,
        ] {
            record_unmeasurable(&mut criterion, &mut records, &roi, reason);
        }
        let result = criterion.finish(&mut records);
        assert_eq!(result.measurable_count, 0);
        assert_eq!(result.unmeasurable_count, 5);
        assert_eq!(result.unmeasurable_reasons.values().sum::<usize>(), 5);
        assert_eq!(
            result.unmeasurable_reasons[degradation::SLANTED_EDGE_TOO_SHORT],
            2
        );
        for (reason, count) in &result.unmeasurable_reasons {
            assert_eq!(
                records
                    .iter()
                    .filter(|record| &record.reason == reason)
                    .count(),
                *count
            );
        }
        assert!(records.iter().all(|record| {
            record.criterion == "mtf50_normalized"
                && record.world == WorldPoint { x: -15.0, y: 72.0 }
        }));
    }

    fn boundary_sources(width: u32, height: u32) -> [QualitySource; 2] {
        [
            QualitySource {
                native_registration: None,
                owner: 1,
                path: "left.raw".to_string(),
                geometry: SourceGeometry {
                    member_to_anchor: Matrix3::identity(),
                    tile_to_world: Matrix3::identity(),
                    station_id: 0,
                },
                dimensions: (width, height),
                native_warp: None,
            },
            QualitySource {
                native_registration: None,
                owner: 2,
                path: "right.raw".to_string(),
                geometry: SourceGeometry {
                    member_to_anchor: Matrix3::identity(),
                    tile_to_world: Matrix3::identity(),
                    station_id: 1,
                },
                dimensions: (width, height),
                native_warp: None,
            },
        ]
    }

    fn vertical_seam_owners(width: u32, height: u32, seam_x: u32) -> Vec<u16> {
        (0..width * height)
            .map(|index| if index % width < seam_x { 1 } else { 2 })
            .collect()
    }

    fn fragmented_vertical_seam_owners(width: u32, height: u32, seam_x: u32) -> Vec<u16> {
        (0..width * height)
            .map(|index| {
                let x = index % width;
                if x < seam_x.saturating_sub(2) {
                    1
                } else if x < seam_x {
                    3
                } else {
                    2
                }
            })
            .collect()
    }

    fn crossing_stroke(
        width: u32,
        height: u32,
        seam_x: u32,
        left_y: u32,
        right_y: u32,
        invert_right: bool,
    ) -> Rgb32FImage {
        Rgb32FImage::from_fn(width, height, |x, y| {
            let threshold = if x < seam_x { left_y } else { right_y };
            let high = if x >= seam_x && invert_right {
                y < threshold
            } else {
                y >= threshold
            };
            Rgb([if high { 0.9 } else { 0.1 }; 3])
        })
    }

    fn boundary_report(
        output: &Rgb32FImage,
        coverage: &GrayImage,
        ownership: &[u16],
        sources: &[QualitySource],
        accepted: &[(usize, usize)],
    ) -> QualityGateCriterionRecord {
        let residual = ResidualWarp::new();
        let mut criterion = Criterion::p95_with_max("boundary_stroke_alignment", 1.5, 3.0);
        let mut records = Vec::new();
        measure_large_boundaries(
            output,
            coverage,
            ownership,
            (0.0, 0.0),
            sources,
            accepted,
            &residual,
            &mut criterion,
            &mut records,
        );
        criterion.finish(&mut records)
    }

    #[test]
    fn accepted_seam_pairs_independently_detected_crossing_strokes() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 82, false);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = vertical_seam_owners(width, height, seam_x);
        let report = boundary_report(
            &output,
            &coverage,
            &ownership,
            &boundary_sources(width, height),
            &[(0, 1)],
        );
        assert_eq!(report.measurable_count, 1);
        assert_eq!(report.measured[0].measured, 2.0);
        assert!(
            report.measured[0]
                .owner_path
                .contains("station:0/station:1")
        );
        assert!(report.measured[0].owner_path.contains("samples:"));
    }

    #[test]
    fn station_boundary_collapses_fragmented_raw_owners() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 82, false);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let mut sources = boundary_sources(width, height).to_vec();
        sources.push(QualitySource {
            native_registration: None,
            owner: 3,
            path: "left-overlap.raw".to_string(),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: Matrix3::identity(),
                station_id: 0,
            },
            dimensions: (width, height),
            native_warp: None,
        });
        let ownership = fragmented_vertical_seam_owners(width, height, seam_x);
        let report = boundary_report(&output, &coverage, &ownership, &sources, &[(0, 1)]);
        assert_eq!(report.measurable_count, 1);
        assert_eq!(report.measured[0].measured, 2.0);
        assert!(report.measured[0].owner_path.contains("owner:1/owner:2"));
    }

    #[test]
    fn nearby_topology_transitions_do_not_duplicate_one_stroke() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 80, false);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = vertical_seam_owners(width, height, seam_x);
        let sources = boundary_sources(width, height);
        let first = boundary_report(&output, &coverage, &ownership, &sources, &[(0, 1)]);
        let second = boundary_report(&output, &coverage, &ownership, &sources, &[(1, 0)]);
        assert_eq!(first.measurable_count, 1);
        assert_eq!(
            first, second,
            "relation order must not change deterministic evidence"
        );
    }

    #[test]
    fn sampling_is_restricted_to_accepted_cross_station_hard_seams() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 80, false);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = vertical_seam_owners(width, height, seam_x);
        let sources = boundary_sources(width, height);
        let report = boundary_report(&output, &coverage, &ownership, &sources, &[]);
        assert_eq!(report.measurable_count, 0);
        assert_eq!(
            report.unmeasurable_reasons[degradation::BOUNDARY_NO_PAIRABLE_EDGE],
            1
        );
    }

    #[test]
    fn opposite_polarity_strokes_are_rejected_as_wrong_correspondences() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 80, true);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = vertical_seam_owners(width, height, seam_x);
        let report = boundary_report(
            &output,
            &coverage,
            &ownership,
            &boundary_sources(width, height),
            &[(0, 1)],
        );
        assert_eq!(report.measurable_count, 0);
        assert!(
            report
                .unmeasurable_reasons
                .contains_key(degradation::BOUNDARY_ORIENTATION_MISMATCH)
        );
    }

    #[test]
    fn wider_wrong_edge_is_rejected_by_fixed_cross_section() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 70, 82, false);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = vertical_seam_owners(width, height, seam_x);
        let report = boundary_report(
            &output,
            &coverage,
            &ownership,
            &boundary_sources(width, height),
            &[(0, 1)],
        );
        assert_eq!(report.measurable_count, 0);
        assert!(
            report
                .unmeasurable_reasons
                .contains_key(degradation::BOUNDARY_LOW_CONTRAST)
        );
    }

    #[test]
    fn coverage_and_source_provenance_are_required_for_pairability() {
        let (width, height, seam_x) = (192, 192, 96);
        let output = crossing_stroke(width, height, seam_x, 80, 80, false);
        let ownership = vertical_seam_owners(width, height, seam_x);
        let sources = boundary_sources(width, height);
        let coverage = GrayImage::from_fn(width, height, |x, _| {
            Luma([if x == seam_x - 1 || x == seam_x {
                255
            } else {
                0
            }])
        });
        let no_coverage = boundary_report(&output, &coverage, &ownership, &sources, &[(0, 1)]);
        assert_eq!(no_coverage.measurable_count, 0);

        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let mut invalid_sources = boundary_sources(width, height);
        invalid_sources[1].geometry.tile_to_world[(0, 2)] = 1_000.0;
        let no_provenance =
            boundary_report(&output, &coverage, &ownership, &invalid_sources, &[(0, 1)]);
        assert_eq!(no_provenance.measurable_count, 0);
        assert!(
            no_provenance
                .unmeasurable_reasons
                .contains_key(degradation::BOUNDARY_PROVENANCE_UNAVAILABLE)
        );
        assert_eq!(no_provenance.technical_unmeasurable_count, 1);
    }

    #[test]
    fn sparse_roi_cap_covers_the_full_canvas_and_late_source_owners() {
        let (width, height) = (4096, 12288);
        let coverage = GrayImage::from_pixel(width, height, Luma([255]));
        let ownership = (0..height)
            .flat_map(|y| std::iter::repeat_n(if y < height / 2 { 1 } else { 2 }, width as usize))
            .collect::<Vec<_>>();
        let selected = select_sparse_rois(&coverage, &ownership, (10.0, -30.0));
        assert_eq!(selected.len(), quality_gate::QUALITY_ROI_MAX_COUNT);
        assert_eq!(
            selected.iter().map(|r| r.owner).collect::<BTreeSet<_>>(),
            BTreeSet::from([1, 2])
        );
        assert!(selected.iter().map(|r| r.y).max().unwrap() >= height - 1024);
        assert!(selected.iter().map(|r| r.x).max().unwrap() >= width - 1024);
        assert_eq!(
            selected,
            select_sparse_rois(&coverage, &ownership, (10.0, -30.0))
        );
    }
}
