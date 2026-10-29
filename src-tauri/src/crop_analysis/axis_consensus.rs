//! Axis consensus fitting for Upright.
//!
//! Groups structural lines into vertical and horizontal consensus sets and
//! decides whether they carry enough independent structure for a perspective
//! correction.

use super::{
    StructuralAxis, StructuralLine, UPRIGHT_AXIS_TOLERANCE_DEGREES, structural_line_center,
    structural_line_direction, structural_line_length, structural_line_residual,
};

pub(super) const LONG_STRUCTURE_SPAN_RATIO: f32 = 0.14;

pub(super) const AXIS_CONSENSUS_TOLERANCE_DEGREES: f32 = 2.4;

#[derive(Debug, Clone)]
pub(super) struct AxisConsensus {
    pub(super) axis: StructuralAxis,
    pub(super) lines: Vec<StructuralLine>,
    pub(super) intercept: f32,
    pub(super) slope: f32,
    pub(super) perspective_model: bool,
    pub(super) agreement: f32,
    pub(super) coordinate_span: f32,
    pub(super) spatial_cells: usize,
    pub(super) framed_line_count: usize,
    pub(super) long_line_count: usize,
    pub(super) max_span_ratio: f32,
    pub(super) rms_error: f32,
    pub(super) score: f32,
}

pub(super) fn axis_coordinate(line: &StructuralLine, width: f32, height: f32) -> f32 {
    let center = structural_line_center(line);
    match line.axis {
        StructuralAxis::Vertical => center.0 / width.max(1.0) * 2.0 - 1.0,
        StructuralAxis::Horizontal => center.1 / height.max(1.0) * 2.0 - 1.0,
    }
}

pub(super) fn spatial_cell(line: &StructuralLine, width: f32, height: f32) -> usize {
    let center = structural_line_center(line);
    let column = ((center.0 / width.max(1.0) * 4.0).floor() as i32).clamp(0, 3) as usize;
    let row = ((center.1 / height.max(1.0) * 4.0).floor() as i32).clamp(0, 3) as usize;
    row * 4 + column
}

pub(super) fn orthogonal_frame_support(line: &StructuralLine, lines: &[StructuralLine]) -> usize {
    let line_center = structural_line_center(line);
    let line_length = structural_line_length(line);
    let line_direction = structural_line_direction(line).to_radians();
    lines
        .iter()
        .filter(|candidate| candidate.axis != line.axis)
        .filter(|candidate| {
            let candidate_direction = structural_line_direction(candidate).to_radians();
            (line_direction - candidate_direction).cos().abs() <= 0.24
        })
        .filter(|candidate| {
            let candidate_center = structural_line_center(candidate);
            let centre_distance =
                (candidate_center.0 - line_center.0).hypot(candidate_center.1 - line_center.1);
            let neighbourhood = (line_length + structural_line_length(candidate)) * 0.7 + 6.0;
            centre_distance <= neighbourhood
        })
        .count()
}

pub(super) fn fit_axis_consensus(
    detected_lines: &[StructuralLine],
    axis: StructuralAxis,
    width: f32,
    height: f32,
    intercept_hint: Option<f32>,
) -> Option<AxisConsensus> {
    #[derive(Clone, Copy)]
    struct Observation {
        line: StructuralLine,
        coordinate: f32,
        residual: f32,
        weight: f32,
        cell: usize,
        framed: bool,
    }

    let observations = detected_lines
        .iter()
        .filter(|line| line.axis == axis)
        .map(|line| {
            let span_ratio = structural_line_length(line) / width.min(height).max(1.0);
            let frame_support = orthogonal_frame_support(line, detected_lines);
            let frame_weight = if span_ratio >= LONG_STRUCTURE_SPAN_RATIO {
                1.0
            } else if frame_support == 0 {
                0.35
            } else {
                1.0 + (frame_support.min(3) as f32) * 0.35
            };
            Observation {
                line: *line,
                coordinate: axis_coordinate(line, width, height),
                residual: structural_line_residual(line),
                // Hough support grows much faster for a single long brush
                // stroke than for a collection of small rectangular details.
                // A square root keeps length useful, while nearby orthogonal
                // partners promote stamp/window/tile frames over lone glyphs.
                weight: line.weight.max(0.01).sqrt() * frame_weight,
                cell: spatial_cell(line, width, height),
                framed: frame_support > 0,
            }
        })
        .collect::<Vec<_>>();
    if observations.len() < 2 {
        return None;
    }

    #[derive(Clone, Copy)]
    struct ModelScore {
        intercept: f32,
        slope: f32,
        score: f32,
        inlier_count: usize,
        coordinate_span: f32,
    }

    let evaluate_model = |intercept: f32, slope: f32| {
        let mut inlier_weight = 0.0f32;
        let mut inlier_count = 0usize;
        let mut cell_mask = 0u16;
        let mut minimum_coordinate = f32::INFINITY;
        let mut maximum_coordinate = f32::NEG_INFINITY;
        for observation in &observations {
            let error = (observation.residual - intercept - slope * observation.coordinate).abs();
            if error > AXIS_CONSENSUS_TOLERANCE_DEGREES {
                continue;
            }
            let closeness = 1.0 - 0.2 * error / AXIS_CONSENSUS_TOLERANCE_DEGREES;
            inlier_weight += observation.weight * closeness;
            inlier_count += 1;
            cell_mask |= 1u16 << observation.cell;
            minimum_coordinate = minimum_coordinate.min(observation.coordinate);
            maximum_coordinate = maximum_coordinate.max(observation.coordinate);
        }
        let cells = cell_mask.count_ones() as f32;
        let distribution_bonus = 0.82 + 0.18 * (cells / 6.0).clamp(0.0, 1.0);
        ModelScore {
            intercept,
            slope,
            score: inlier_weight * distribution_bonus,
            inlier_count,
            coordinate_span: (maximum_coordinate - minimum_coordinate).max(0.0),
        }
    };

    let mut best_flat_model = None::<ModelScore>;
    for observation in &observations {
        if intercept_hint.is_some_and(|hint| (observation.residual - hint).abs() > 3.5) {
            continue;
        }
        let model = evaluate_model(observation.residual, 0.0);
        if best_flat_model.is_none_or(|best| model.score > best.score) {
            best_flat_model = Some(model);
        }
    }

    let mut best_perspective_model = None::<ModelScore>;
    for (index, left) in observations.iter().enumerate() {
        for right in observations.iter().skip(index + 1) {
            let coordinate_delta = right.coordinate - left.coordinate;
            if coordinate_delta.abs() < 0.16 {
                continue;
            }
            let slope = (right.residual - left.residual) / coordinate_delta;
            let intercept = left.residual - slope * left.coordinate;
            if slope.abs() <= 40.0
                && intercept.abs() <= UPRIGHT_AXIS_TOLERANCE_DEGREES
                && !intercept_hint.is_some_and(|hint| (intercept - hint).abs() > 3.5)
            {
                let model = evaluate_model(intercept, slope);
                if best_perspective_model.is_none_or(|best| model.score > best.score) {
                    best_perspective_model = Some(model);
                }
            }
        }
    }

    let total_weight = observations
        .iter()
        .map(|observation| observation.weight)
        .sum::<f32>();
    let flat_model = best_flat_model?;
    let perspective_model = best_perspective_model.is_some_and(|model| {
        model.inlier_count >= 4
            && model.coordinate_span >= 0.55
            && model.slope.abs() * model.coordinate_span >= 1.4
            && model.score >= flat_model.score * 1.45
    });
    let selected_model = if perspective_model {
        best_perspective_model.unwrap_or(flat_model)
    } else {
        flat_model
    };
    let mut intercept = selected_model.intercept;
    let mut slope = selected_model.slope;

    // Refit the winning RANSAC family twice with weighted least squares. The
    // first pass removes unrelated glyphs; the second improves sub-degree
    // accuracy for short stamp and window edges.
    for _ in 0..2 {
        let inliers = observations
            .iter()
            .filter(|observation| {
                (observation.residual - intercept - slope * observation.coordinate).abs()
                    <= AXIS_CONSENSUS_TOLERANCE_DEGREES
            })
            .collect::<Vec<_>>();
        if inliers.len() < 2 {
            return None;
        }
        let sum_weight = inliers
            .iter()
            .map(|observation| observation.weight)
            .sum::<f32>();
        let sum_y = inliers
            .iter()
            .map(|observation| observation.residual * observation.weight)
            .sum::<f32>();
        if perspective_model {
            let sum_x = inliers
                .iter()
                .map(|observation| observation.coordinate * observation.weight)
                .sum::<f32>();
            let sum_xx = inliers
                .iter()
                .map(|observation| {
                    observation.coordinate * observation.coordinate * observation.weight
                })
                .sum::<f32>();
            let sum_xy = inliers
                .iter()
                .map(|observation| {
                    observation.coordinate * observation.residual * observation.weight
                })
                .sum::<f32>();
            let denominator = sum_weight * sum_xx - sum_x * sum_x;
            if denominator.abs() > 1e-4 {
                slope = (sum_weight * sum_xy - sum_x * sum_y) / denominator;
                intercept = (sum_y - slope * sum_x) / sum_weight.max(f32::EPSILON);
            }
        } else {
            slope = 0.0;
            intercept = sum_y / sum_weight.max(f32::EPSILON);
        }
    }

    let inliers = observations
        .iter()
        .filter(|observation| {
            (observation.residual - intercept - slope * observation.coordinate).abs()
                <= AXIS_CONSENSUS_TOLERANCE_DEGREES
        })
        .copied()
        .collect::<Vec<_>>();
    if inliers.len() < 2 {
        return None;
    }

    let inlier_weight = inliers
        .iter()
        .map(|observation| observation.weight)
        .sum::<f32>();
    let weighted_error = inliers
        .iter()
        .map(|observation| {
            let error = observation.residual - intercept - slope * observation.coordinate;
            error * error * observation.weight
        })
        .sum::<f32>();
    let rms_error = (weighted_error / inlier_weight.max(f32::EPSILON)).sqrt();
    let minimum_coordinate = inliers
        .iter()
        .map(|observation| observation.coordinate)
        .fold(f32::INFINITY, f32::min);
    let maximum_coordinate = inliers
        .iter()
        .map(|observation| observation.coordinate)
        .fold(f32::NEG_INFINITY, f32::max);
    let cell_mask = inliers
        .iter()
        .fold(0u16, |mask, observation| mask | (1u16 << observation.cell));
    let min_dimension = width.min(height).max(1.0);
    let span_ratios = inliers
        .iter()
        .map(|observation| structural_line_length(&observation.line) / min_dimension)
        .collect::<Vec<_>>();
    let total_span_ratio = span_ratios.iter().sum::<f32>();
    let max_span_ratio = span_ratios.iter().copied().fold(0.0f32, f32::max);
    let long_line_count = span_ratios
        .iter()
        .filter(|span| **span >= LONG_STRUCTURE_SPAN_RATIO)
        .count();
    let agreement = inlier_weight / total_weight.max(f32::EPSILON);
    let spatial_cells = cell_mask.count_ones() as usize;
    let framed_line_count = inliers
        .iter()
        .filter(|observation| observation.framed)
        .count();

    if agreement < 0.2
        || spatial_cells < 2
        || total_span_ratio < 0.22
        || rms_error > AXIS_CONSENSUS_TOLERANCE_DEGREES * 0.8
    {
        return None;
    }

    let score = agreement
        * (spatial_cells as f32 / 6.0).clamp(0.25, 1.0)
        * (total_span_ratio / 0.8).clamp(0.25, 1.0);
    Some(AxisConsensus {
        axis,
        lines: inliers
            .into_iter()
            .map(|observation| observation.line)
            .collect(),
        intercept,
        slope,
        perspective_model,
        agreement,
        coordinate_span: maximum_coordinate - minimum_coordinate,
        spatial_cells,
        framed_line_count,
        long_line_count,
        max_span_ratio,
        rms_error,
        score,
    })
}

pub(super) fn consensus_has_independent_structure(consensus: &AxisConsensus) -> bool {
    consensus.long_line_count >= 2
        || consensus.max_span_ratio >= 0.45
        || (consensus.lines.len() >= 6
            && consensus.framed_line_count >= 4
            && consensus.spatial_cells >= 4
            && consensus.agreement >= 0.35)
}

pub(super) fn consensus_has_distributed_local_structure(consensus: &AxisConsensus) -> bool {
    consensus.lines.len() >= 4
        && consensus.spatial_cells >= 3
        && consensus.agreement >= 0.24
        && consensus.rms_error <= 1.6
}

pub(super) fn consensuses_are_orthogonal_peers(
    horizontal: &AxisConsensus,
    vertical: &AxisConsensus,
) -> bool {
    horizontal.axis == StructuralAxis::Horizontal
        && vertical.axis == StructuralAxis::Vertical
        && (horizontal.intercept - vertical.intercept).abs() <= 3.5
        && consensus_has_distributed_local_structure(horizontal)
        && consensus_has_distributed_local_structure(vertical)
}

pub(super) fn consensus_supports_perspective(consensus: &AxisConsensus) -> bool {
    consensus.perspective_model
        && consensus.lines.len() >= 4
        && consensus.spatial_cells >= 3
        && consensus.coordinate_span >= 0.55
        && consensus.slope.abs() * consensus.coordinate_span >= 1.4
        && consensus.rms_error <= 1.6
}
