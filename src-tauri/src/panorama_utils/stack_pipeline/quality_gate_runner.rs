//! Runtime adapter for the pure Quality_Gate measurements.
//!
//! `quality_gate.rs` deliberately contains no image loading or report policy.
//! This module supplies that small amount of orchestration: it selects ROIs
//! from the final owner planes, decodes one owner source at a time, samples the
//! corresponding reference ROI, and records all nine criteria.  The caller can
//! therefore use it in observation-only mode without changing export pixels.

#[cfg(test)]
use std::collections::BTreeSet;

#[cfg(test)]
use image::Rgb;
use image::{GrayImage, ImageBuffer, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Vector3};

use super::degradation;
use super::quality_gate::{self, QualityRoi, SourceGeometry};
#[cfg(test)]
use super::report::{QualityGateCriterionRecord, QualityGateReport, QualityGateVerdict};
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

#[cfg(test)]
use criteria::owner_disagreement_veto_limit;
pub(crate) use criteria::{
    Criterion, OWNER_EVIDENCE_UNKNOWN, OwnerSharpnessCriterionInput,
    build_owner_sharpness_criterion, overall_verdict, owner_shortfall_limit, quantize_hundredths,
    record_gate_outcome,
};
#[cfg(test)]
pub(crate) use criteria::{criterion_verdict, owner_sharpness_stats};

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
mod tone_owner_oracle;

#[cfg(test)]
use tone_owner_oracle::{
    ToneOwnerEvidence, collect_tone_owner_evidence, owner_delta_at, owner_distribution_feasible,
    select_joint_relaxation, tone_owner_distribution, tone_owner_holdout_rois, tone_percentile,
    unique_owner_source,
};

#[cfg(test)]
mod research_diagnostics;

#[cfg(test)]
use research_diagnostics::{write_effective_coverage_diagnostics, write_gradient_source_oracle};

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
mod tests;
