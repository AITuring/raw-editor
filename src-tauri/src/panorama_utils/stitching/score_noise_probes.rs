//! Test-only score-noise probe records and ROI helpers.
//!
//! Registration cells, candidate cells, attribution counts and owner ROI
//! selection for the score-noise research diagnostics.

use super::super::stack_pipeline::focus_fuser;
#[cfg(test)]
use super::super::stack_pipeline::focus_fuser::CellSamplingPlan;
#[cfg(test)]
use super::super::stack_pipeline::focus_fuser::OwnershipGridGeometry;
#[cfg(test)]
use super::super::stack_pipeline::report::IntraStationFrameRecord;
use image::GrayImage;
use image::Rgb;
use std::path::PathBuf;

#[cfg(test)]
use super::owner_quality_oracle;

// Test-only registration-aligned score-noise evidence. The production score,
// ownership decision, 32px window, and source pixels remain untouched.
#[cfg(test)]
pub(super) const SCORE_NOISE_PHASE_OFFSETS: [(f64, f64); 9] = [
    (-0.25, -0.25),
    (0.0, -0.25),
    (0.25, -0.25),
    (-0.25, 0.0),
    (0.0, 0.0),
    (0.25, 0.0),
    (-0.25, 0.25),
    (0.0, 0.25),
    (0.25, 0.25),
];

#[cfg(test)]
#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct ScoreNoiseRegistrationCell {
    pub(super) local_to_anchor_displacement: [f64; 2],
    pub(super) local_to_anchor_jacobian: [f64; 4],
    pub(super) local_to_anchor_jacobian_determinant: f64,
    pub(super) residual_displacement: [f64; 2],
    pub(super) residual_jacobian: [f64; 4],
    pub(super) median_symmetric_residual_px: f64,
}

#[cfg(test)]
impl Default for ScoreNoiseRegistrationCell {
    fn default() -> Self {
        Self {
            local_to_anchor_displacement: [0.0; 2],
            local_to_anchor_jacobian: [1.0, 0.0, 0.0, 1.0],
            local_to_anchor_jacobian_determinant: 1.0,
            residual_displacement: [0.0; 2],
            residual_jacobian: [0.0; 4],
            median_symmetric_residual_px: 0.0,
        }
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Default, serde::Serialize)]
pub(super) struct ScoreNoiseCandidateCell {
    pub(super) label: usize,
    pub(super) source: String,
    pub(super) covered: bool,
    pub(super) fully_covered: bool,
    pub(super) flat: bool,
    pub(super) low_frequency_luma: f64,
    pub(super) probe_raw_acutance: [f64; 5],
    pub(super) normalized_sharpness: f64,
    pub(super) repeated_measurement_variance: f64,
    pub(super) phase_normalized_sharpness: [f64; 9],
    pub(super) phase_offset_variance: f64,
    pub(super) probe_mean_sharpness: f64,
    pub(super) probe_median_sharpness: f64,
    pub(super) disagreement: f64,
    pub(super) graph_cut_data_cost: Option<f64>,
    pub(super) registration: ScoreNoiseRegistrationCell,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, serde::Serialize)]
pub(super) struct ScoreNoiseAttributionCounts {
    pub(super) sampling_phase_instability: u64,
    pub(super) probe_aggregation_instability: u64,
    pub(super) low_texture_or_low_confidence: u64,
    pub(super) registration_disagreement: u64,
    pub(super) pairwise_smoothness_override: u64,
    pub(super) unattributed: u64,
}

#[cfg(test)]
#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct ScoreNoiseRoiSummary {
    pub(super) left: u32,
    pub(super) top: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) textured_cells: u64,
    pub(super) shortfall_cells: u64,
    pub(super) owner_sharpness_coverage: Option<f64>,
    pub(super) attribution: ScoreNoiseAttributionCounts,
    pub(super) signals: ScoreNoiseAttributionCounts,
}

#[cfg(test)]
#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct ScoreNoiseCellRecord {
    pub(super) station_index: usize,
    pub(super) cell: usize,
    pub(super) column: u32,
    pub(super) row: u32,
    pub(super) bounds: [u32; 4],
    pub(super) candidates: Vec<ScoreNoiseCandidateCell>,
    pub(super) winner_source: Option<String>,
    pub(super) production_owner_source: Option<String>,
    pub(super) data_term_only_owner_source: Option<String>,
    pub(super) winner_score: f64,
    pub(super) runner_up_score: f64,
    pub(super) top_two_margin: f64,
    pub(super) confidence: f64,
    pub(super) owner_shortfall: f64,
    pub(super) winner_horizontal_run_cells: u32,
    pub(super) winner_connected_region_cells: u64,
    pub(super) production_owner_horizontal_run_cells: u32,
    pub(super) production_owner_connected_region_cells: u64,
    pub(super) production_data_term: Option<f64>,
    pub(super) data_term_only_data_term: Option<f64>,
    pub(super) pairwise_contributions: [f64; 4],
    pub(super) pairwise_total: f64,
    pub(super) attribution: Option<String>,
}

#[cfg(test)]
#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct ScoreNoiseSummary {
    pub(super) schema: u32,
    pub(super) station_index: usize,
    pub(super) dimensions: [u32; 2],
    pub(super) grid: [u32; 3],
    pub(super) production_window_px: f64,
    pub(super) phase_offsets_in_cell_sides: [(f64, f64); 9],
    pub(super) textured_cells: u64,
    pub(super) shortfall_cells: u64,
    pub(super) owner_sharpness_coverage: Option<f64>,
    pub(super) data_term_only_shortfall_cells: u64,
    pub(super) data_term_only_owner_sharpness_coverage: Option<f64>,
    pub(super) production_512px_single_owner_rois: u64,
    pub(super) data_term_only_512px_single_owner_rois: u64,
    pub(super) attribution: ScoreNoiseAttributionCounts,
    pub(super) signals: ScoreNoiseAttributionCounts,
    pub(super) rois: Vec<ScoreNoiseRoiSummary>,
}

#[cfg(test)]
#[derive(serde::Deserialize)]
pub(super) struct FinalOwnerRoiConstraint {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) world_origin: [f64; 2],
    pub(super) station: usize,
    pub(super) local_owner: u16,
    pub(super) cells: Vec<usize>,
}

#[cfg(test)]
pub(super) fn score_noise_probe_values(
    plan: &CellSamplingPlan,
    origin_x: f64,
    origin_y: f64,
    phase: (f64, f64),
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
) -> [f64; 5] {
    let half = plan.cell_size_px() * 0.5;
    let center_x = origin_x + half + phase.0 * plan.cell_size_px();
    let center_y = origin_y + half + phase.1 * plan.cell_size_px();
    let mut values = [0.0; 5];
    for (index, (dx, dy)) in plan.probes().into_iter().enumerate() {
        values[index] = crate::panorama_utils::mosaic::acutance_with_step(
            &mut sample,
            center_x + dx,
            center_y + dy,
            plan.step_px(),
        );
    }
    values
}

#[cfg(test)]
pub(super) fn score_noise_aggregate(values: &[f64; 5]) -> f64 {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return 0.0;
    }
    finite.sort_by(f64::total_cmp);
    let first = finite.len().saturating_sub(2);
    (finite[first] + finite[first + 1.min(finite.len() - 1)]) * 0.5
}

#[cfg(test)]
pub(super) fn score_noise_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / values.len() as f64
}

#[cfg(test)]
pub(super) fn score_noise_smoothness_scale(candidates: &[ScoreNoiseCandidateCell]) -> f64 {
    let mut scores = candidates
        .iter()
        .filter(|candidate| candidate.covered)
        .map(|candidate| candidate.normalized_sharpness)
        .collect::<Vec<_>>();
    scores.sort_by(|left, right| right.total_cmp(left));
    if scores.len() <= 1 || scores[0] <= 0.0 {
        return 1.0;
    }
    let confidence = ((scores[0] - scores[1]) / scores[0]).clamp(0.0, 1.0);
    let limit =
        crate::panorama_utils::stack_pipeline::quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64;
    if confidence <= limit {
        1.0
    } else {
        (limit / confidence).clamp(0.05, 1.0)
    }
}

#[cfg(test)]
pub(super) fn score_noise_project(transform: &[f64; 9], point: (f64, f64)) -> Option<(f64, f64)> {
    let denominator = transform[6] * point.0 + transform[7] * point.1 + transform[8];
    if !denominator.is_finite() || denominator.abs() < 1e-12 {
        return None;
    }
    let x = (transform[0] * point.0 + transform[1] * point.1 + transform[2]) / denominator;
    let y = (transform[3] * point.0 + transform[4] * point.1 + transform[5]) / denominator;
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

#[cfg(test)]
pub(super) fn score_noise_registration_cells(
    geometry: &OwnershipGridGeometry,
    record: &IntraStationFrameRecord,
    residual: Option<&crate::panorama_utils::mosaic::NativeResidualDisplacement>,
) -> Vec<ScoreNoiseRegistrationCell> {
    (0..geometry.cell_count())
        .map(|cell| {
            let column = cell as u32 % geometry.columns.max(1);
            let row = cell as u32 / geometry.columns.max(1);
            let (origin_x, origin_y) = geometry.cell_origin(column, row);
            let half = f64::from(geometry.cell_size_px) * 0.5;
            let point = (origin_x + half, origin_y + half);
            let mapped = score_noise_project(&record.transform, point).unwrap_or(point);
            let mapped_x = score_noise_project(&record.transform, (point.0 + 1.0, point.1))
                .unwrap_or((mapped.0 + 1.0, mapped.1));
            let mapped_y = score_noise_project(&record.transform, (point.0, point.1 + 1.0))
                .unwrap_or((mapped.0, mapped.1 + 1.0));
            let jacobian = [
                mapped_x.0 - mapped.0,
                mapped_y.0 - mapped.0,
                mapped_x.1 - mapped.1,
                mapped_y.1 - mapped.1,
            ];
            let residual_at = |x: f64, y: f64| residual.map_or([0.0; 2], |field| field.at(x, y));
            let residual_center = residual_at(point.0, point.1);
            let residual_x = residual_at(point.0 + 1.0, point.1);
            let residual_y = residual_at(point.0, point.1 + 1.0);
            ScoreNoiseRegistrationCell {
                local_to_anchor_displacement: [mapped.0 - point.0, mapped.1 - point.1],
                local_to_anchor_jacobian: jacobian,
                local_to_anchor_jacobian_determinant: jacobian[0] * jacobian[3]
                    - jacobian[1] * jacobian[2],
                residual_displacement: residual_center,
                residual_jacobian: [
                    residual_x[0] - residual_center[0],
                    residual_y[0] - residual_center[0],
                    residual_x[1] - residual_center[1],
                    residual_y[1] - residual_center[1],
                ],
                median_symmetric_residual_px: record.median_symmetric_error_px,
            }
        })
        .collect()
}

#[cfg(test)]
pub(super) fn score_noise_owner_regions(
    geometry: &OwnershipGridGeometry,
    owners: &[u16],
) -> (Vec<u64>, Vec<u32>) {
    let count = geometry.cell_count();
    let mut region_sizes = vec![0u64; count];
    let mut horizontal_runs = vec![0u32; count];
    for row in 0..geometry.rows as usize {
        let mut column = 0usize;
        while column < geometry.columns as usize {
            let start = row * geometry.columns as usize + column;
            let owner = owners.get(start).copied().unwrap_or(0);
            let mut end = column + 1;
            while end < geometry.columns as usize
                && owners.get(row * geometry.columns as usize + end).copied() == Some(owner)
            {
                end += 1;
            }
            let length = (end - column) as u32;
            for x in column..end {
                horizontal_runs[row * geometry.columns as usize + x] = length;
            }
            column = end;
        }
    }
    let mut visited = vec![false; count];
    for start in 0..count {
        if visited[start] || owners.get(start).copied().unwrap_or(0) == 0 {
            continue;
        }
        let owner = owners[start];
        let mut stack = vec![start];
        let mut region = Vec::new();
        visited[start] = true;
        while let Some(cell) = stack.pop() {
            region.push(cell);
            let x = cell % geometry.columns as usize;
            let y = cell / geometry.columns as usize;
            for neighbour in [
                (x > 0).then(|| cell - 1),
                (x + 1 < geometry.columns as usize).then_some(cell + 1),
                (y > 0).then(|| cell - geometry.columns as usize),
                (y + 1 < geometry.rows as usize).then_some(cell + geometry.columns as usize),
            ]
            .into_iter()
            .flatten()
            {
                if !visited[neighbour] && owners[neighbour] == owner {
                    visited[neighbour] = true;
                    stack.push(neighbour);
                }
            }
        }
        let size = region.len() as u64;
        for cell in region {
            region_sizes[cell] = size;
        }
    }
    (region_sizes, horizontal_runs)
}

#[cfg(test)]
pub(super) fn score_noise_single_owner_rois(
    dimensions: (u32, u32),
    geometry: &OwnershipGridGeometry,
    owners: &[u16],
) -> u64 {
    const ROI: u32 = 512;
    if dimensions.0 < ROI || dimensions.1 < ROI {
        return 0;
    }
    let mut count = 0u64;
    for top in (0..=dimensions.1 - ROI).step_by(ROI as usize) {
        for left in (0..=dimensions.0 - ROI).step_by(ROI as usize) {
            let first_column = left / geometry.cell_size_px.max(1);
            let last_column = (left + ROI - 1) / geometry.cell_size_px.max(1);
            let first_row = top / geometry.cell_size_px.max(1);
            let last_row = (top + ROI - 1) / geometry.cell_size_px.max(1);
            let first = first_row as usize * geometry.columns as usize + first_column as usize;
            let owner = owners.get(first).copied().unwrap_or(0);
            if owner == 0 {
                continue;
            }
            let single = (first_row..=last_row).all(|row| {
                (first_column..=last_column).all(|column| {
                    owners.get(row as usize * geometry.columns as usize + column as usize)
                        == Some(&owner)
                })
            });
            count += u64::from(single);
        }
    }
    count
}

#[cfg(test)]
pub(super) fn score_noise_winner_owners(candidates: &[Vec<ScoreNoiseCandidateCell>]) -> Vec<u16> {
    candidates
        .iter()
        .map(|candidates| {
            candidates
                .iter()
                .filter(|candidate| candidate.covered)
                .min_by(|left, right| {
                    right
                        .normalized_sharpness
                        .total_cmp(&left.normalized_sharpness)
                        .then_with(|| left.label.cmp(&right.label))
                })
                .map_or(0, |candidate| focus_fuser::owner_of(candidate.label))
        })
        .collect()
}

#[cfg(test)]
pub(super) fn score_noise_primary_attribution(
    is_shortfall: bool,
    pairwise_override: bool,
    registration_disagreement: bool,
    phase_instability: bool,
    aggregation_instability: bool,
    low_texture_or_confidence: bool,
) -> Option<&'static str> {
    is_shortfall.then(|| {
        // This is the causal ordering: when the production owner differs from
        // the minimum of the actual data terms, the pairwise term directly made
        // the choice. Other, potentially overlapping properties are exported as
        // non-exclusive signals instead of obscuring that direct mechanism.
        if pairwise_override {
            "pairwise_smoothness_override"
        } else if registration_disagreement {
            "registration_disagreement"
        } else if phase_instability {
            "sampling_phase_instability"
        } else if aggregation_instability {
            "probe_aggregation_instability"
        } else if low_texture_or_confidence {
            "low_texture_or_low_confidence"
        } else {
            "unattributed"
        }
    })
}

#[cfg(test)]
pub(super) fn owner_quality_forced_labeling(station_index: usize) -> Option<Vec<u16>> {
    let directory = PathBuf::from(std::env::var_os(
        "RAW_EDITOR_FOCUS_OWNER_QUALITY_LABEL_DIR",
    )?);
    let path = directory.join(format!("owner-quality-oracle-station-{station_index}.json"));
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let field = if std::env::var_os("RAW_EDITOR_FOCUS_OWNER_QUALITY_PRESERVE_REGIONS").is_some() {
        "production_preserving_eight_labeling"
    } else {
        "best_eight_labeling"
    };
    value
        .get(field)?
        .as_array()?
        .iter()
        .map(|owner| owner.as_u64().and_then(|owner| u16::try_from(owner).ok()))
        .collect()
}

#[cfg(test)]
pub(super) fn score_noise_source_roi_origins(
    source: &str,
    owner: u16,
    mask: &GrayImage,
    layer_left: u32,
    layer_top: u32,
    dimensions: (u32, u32),
) -> Vec<owner_quality_oracle::SourceRoiOrigin> {
    let side = crate::panorama_utils::stack_pipeline::quality_gate::QUALITY_ROI_SIDE;
    let margin = crate::panorama_utils::stack_pipeline::quality_gate::QUALITY_ROI_MARGIN as u32;
    if dimensions.0 < side || dimensions.1 < side || mask.width() == 0 || mask.height() == 0 {
        return Vec::new();
    }
    // A summed invalid-pixel table makes every exact 544px padded source
    // coverage query O(1). It is temporary and exists only under the explicit
    // test diagnostic switch.
    let integral_width = mask.width() as usize + 1;
    let mut invalid = vec![0u32; integral_width * (mask.height() as usize + 1)];
    for y in 0..mask.height() as usize {
        let mut row = 0u32;
        for x in 0..mask.width() as usize {
            row += u32::from(mask.as_raw()[y * mask.width() as usize + x] == 0);
            invalid[(y + 1) * integral_width + x + 1] = invalid[y * integral_width + x + 1] + row;
        }
    }
    let all_source_covered = |left: u32, top: u32, right: u32, bottom: u32| {
        let Some(local_left) = left.checked_sub(layer_left) else {
            return false;
        };
        let Some(local_top) = top.checked_sub(layer_top) else {
            return false;
        };
        let Some(local_right) = right.checked_sub(layer_left) else {
            return false;
        };
        let Some(local_bottom) = bottom.checked_sub(layer_top) else {
            return false;
        };
        if local_right > mask.width()
            || local_bottom > mask.height()
            || local_left >= local_right
            || local_top >= local_bottom
        {
            return false;
        }
        let (x0, y0, x1, y1) = (
            local_left as usize,
            local_top as usize,
            local_right as usize,
            local_bottom as usize,
        );
        invalid[y1 * integral_width + x1] + invalid[y0 * integral_width + x0]
            - invalid[y0 * integral_width + x1]
            - invalid[y1 * integral_width + x0]
            == 0
    };
    let mut result = Vec::new();
    for y in (0..=dimensions.1 - side).step_by((side / 2) as usize) {
        for x in (0..=dimensions.0 - side).step_by((side / 2) as usize) {
            let Some(left) = x.checked_sub(margin) else {
                continue;
            };
            let Some(top) = y.checked_sub(margin) else {
                continue;
            };
            let Some(right) = x
                .checked_add(side)
                .and_then(|value| value.checked_add(margin))
            else {
                continue;
            };
            let Some(bottom) = y
                .checked_add(side)
                .and_then(|value| value.checked_add(margin))
            else {
                continue;
            };
            if right <= dimensions.0
                && bottom <= dimensions.1
                && all_source_covered(left, top, right, bottom)
            {
                result.push(owner_quality_oracle::SourceRoiOrigin {
                    owner,
                    source: source.to_string(),
                    x,
                    y,
                });
            }
        }
    }
    result
}
