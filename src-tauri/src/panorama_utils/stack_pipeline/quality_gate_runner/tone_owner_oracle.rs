//! Test-only evidence helpers of the tone owner-fidelity oracle.
//!
//! Owner distributions, holdout ROIs and joint relaxations measured against
//! Source_RAW; write_tone_owner_fidelity_oracle in the facade consumes them.

use super::super::quality_gate;
use super::super::quality_gate::QualityRoi;
#[cfg(test)]
use image::Rgb;
use image::Rgb32FImage;
#[cfg(test)]
use std::collections::BTreeMap;

use super::{
    QualityGateInput, QualitySource, SourceLoader, gradient_energy_ratio, output_roi,
    source_roi_for_quality,
};

#[cfg(test)]
#[derive(Clone)]
pub(super) struct ToneOwnerEvidence {
    pub(super) partition: &'static str,
    pub(super) roi: QualityRoi,
    pub(super) owner_path: String,
    pub(super) station_id: usize,
    pub(super) tile_index: usize,
    pub(super) baseline_mean: [f32; 3],
    pub(super) candidate_mean: [f32; 3],
    pub(super) reference_mean: [f32; 3],
    pub(super) field_gain: [f32; 3],
    pub(super) global_gain: [f32; 3],
    pub(super) global_bias: [f32; 3],
    pub(super) coverage_min: u8,
    pub(super) confidence_min: f32,
    pub(super) confidence_mean: f64,
    pub(super) overlap_participation: bool,
    pub(super) fit_disjoint: bool,
    pub(super) gradient_ratio_candidate_to_source: Option<f64>,
    pub(super) noise_ratio_candidate_to_source: Option<f64>,
}

#[cfg(test)]
pub(super) fn tone_owner_distribution(rgb: [f32; 3]) -> serde_json::Value {
    serde_json::json!({
        "display_srgb": rgb,
        "linear_rgb": rgb.map(crate::color_management::srgb_to_linear_channel),
        "lab": crate::panorama_utils::stack_pipeline::tone::diagnostic_lab_rgb(rgb),
    })
}

#[cfg(test)]
pub(super) fn tone_percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * quantile).round() as usize]
}

#[cfg(test)]
pub(super) fn tone_owner_candidate_roi(
    snapshot: &crate::panorama_utils::stack_pipeline::tone::ToneOwnerOracleSnapshot,
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
pub(super) fn tone_owner_holdout_rois(
    input: &QualityGateInput<'_>,
    formal: &[QualityRoi],
    snapshot: &crate::panorama_utils::stack_pipeline::tone::ToneOwnerOracleSnapshot,
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
pub(super) fn unique_owner_source(
    sources: &[QualitySource],
    owner: u16,
) -> Result<&QualitySource, String> {
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
pub(super) fn collect_tone_owner_evidence(
    input: &QualityGateInput<'_>,
    snapshot: &crate::panorama_utils::stack_pipeline::tone::ToneOwnerOracleSnapshot,
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
pub(super) fn owner_delta_at(evidence: &ToneOwnerEvidence, relaxation: f64) -> f64 {
    let post = std::array::from_fn(|channel| {
        (f64::from(evidence.baseline_mean[channel])
            + relaxation
                * f64::from(evidence.candidate_mean[channel] - evidence.baseline_mean[channel]))
            as f32
    });
    crate::panorama_utils::stack_pipeline::tone::delta_e00_rgb(post, evidence.reference_mean)
}

#[cfg(test)]
pub(super) fn owner_distribution_feasible(
    evidence: &[&ToneOwnerEvidence],
    relaxation: f64,
) -> bool {
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
pub(super) struct ToneJointRelaxation {
    pub(super) selected: Option<f64>,
    pub(super) minimum_boundary: Option<f64>,
    pub(super) maximum_owner: Option<f64>,
    pub(super) pareto_gap: Option<f64>,
}

#[cfg(test)]
pub(super) fn select_joint_relaxation(boundary: &[f64], owner: &[f64]) -> ToneJointRelaxation {
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
