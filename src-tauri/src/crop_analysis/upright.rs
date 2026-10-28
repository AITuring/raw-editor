//! Upright perspective analysis.
//!
//! Optimises the perspective parameters against the structural lines and
//! returns the correction for the requested Upright mode.

use image::DynamicImage;
use serde::Deserialize;
use serde::Serialize;

use super::{
    StructuralAxis, StructuralLine, consensus_has_independent_structure,
    consensus_supports_perspective, consensuses_are_orthogonal_peers, extract_structural_lines,
    fit_axis_consensus, normalize_line_angle,
};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum UprightMode {
    Auto,
    Level,
    Vertical,
    Full,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UprightAnalysis {
    pub rotation: f32,
    pub vertical: f32,
    pub horizontal: f32,
    pub confidence: f32,
    pub detected: bool,
    pub line_count: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct UprightParameters {
    pub(super) rotation: f32,
    pub(super) vertical: f32,
    pub(super) horizontal: f32,
}

pub(super) fn project_point(
    point: (f32, f32),
    width: f32,
    height: f32,
    params: UprightParameters,
) -> Option<(f32, f32)> {
    let cx = width * 0.5;
    let cy = height * 0.5;
    let x = point.0 - cx;
    let y = point.1 - cy;
    let radians = params.rotation.to_radians();
    let (sine, cosine) = radians.sin_cos();
    let rotated_x = cosine * x - sine * y;
    let rotated_y = sine * x + cosine * y;
    let perspective_horizontal = -params.horizontal / (50.0 * width.max(1.0));
    let perspective_vertical = params.vertical / (50.0 * height.max(1.0));
    let denominator = 1.0 + perspective_horizontal * rotated_x + perspective_vertical * rotated_y;
    if denominator.abs() < 1e-4 {
        return None;
    }
    Some((cx + rotated_x / denominator, cy + rotated_y / denominator))
}

pub(super) fn line_residual_degrees(
    line: &StructuralLine,
    width: f32,
    height: f32,
    params: UprightParameters,
) -> Option<f32> {
    let start = project_point(line.start, width, height, params)?;
    let end = project_point(line.end, width, height, params)?;
    let direction = normalize_line_angle((end.1 - start.1).atan2(end.0 - start.0).to_degrees());
    Some(match line.axis {
        StructuralAxis::Horizontal => direction,
        StructuralAxis::Vertical => {
            if direction >= 0.0 {
                direction - 90.0
            } else {
                direction + 90.0
            }
        }
    })
}

pub(super) fn upright_objective(
    lines: &[StructuralLine],
    width: f32,
    height: f32,
    params: UprightParameters,
    regularization: f32,
) -> f32 {
    let mut weighted_error = 0.0f32;
    let mut total_weight = 0.0f32;
    for line in lines {
        let Some(residual) = line_residual_degrees(line, width, height, params) else {
            return f32::INFINITY;
        };
        weighted_error += residual * residual * line.weight;
        total_weight += line.weight;
    }
    if total_weight <= f32::EPSILON {
        return f32::INFINITY;
    }

    weighted_error / total_weight
        + regularization
            * ((params.rotation / 15.0).powi(2)
                + (params.vertical / 100.0).powi(2)
                + (params.horizontal / 100.0).powi(2))
}

pub(super) fn optimize_upright(
    lines: &[StructuralLine],
    width: f32,
    height: f32,
    enable_vertical: bool,
    enable_horizontal: bool,
    regularization: f32,
) -> UprightParameters {
    let (weighted_residual, total_weight) = lines.iter().fold((0.0f32, 0.0f32), |acc, line| {
        let direction = normalize_line_angle(
            (line.end.1 - line.start.1)
                .atan2(line.end.0 - line.start.0)
                .to_degrees(),
        );
        let residual = match line.axis {
            StructuralAxis::Horizontal => direction,
            StructuralAxis::Vertical => {
                if direction >= 0.0 {
                    direction - 90.0
                } else {
                    direction + 90.0
                }
            }
        };
        (acc.0 + residual * line.weight, acc.1 + line.weight)
    });

    let mut best = UprightParameters {
        rotation: (-(weighted_residual / total_weight.max(f32::EPSILON))).clamp(-15.0, 15.0),
        ..UprightParameters::default()
    };
    let mut best_score = upright_objective(lines, width, height, best, regularization);
    let rotation_steps = [4.0f32, 2.0, 1.0, 0.5, 0.2, 0.1];
    let perspective_steps = [30.0f32, 15.0, 7.5, 3.0, 1.0, 0.5];

    for index in 0..rotation_steps.len() {
        for _ in 0..3 {
            let mut improved = false;
            for direction in [-1.0f32, 1.0] {
                let mut candidate = best;
                candidate.rotation =
                    (candidate.rotation + direction * rotation_steps[index]).clamp(-15.0, 15.0);
                let score = upright_objective(lines, width, height, candidate, regularization);
                if score + 1e-5 < best_score {
                    best = candidate;
                    best_score = score;
                    improved = true;
                }
            }

            if enable_vertical {
                for direction in [-1.0f32, 1.0] {
                    let mut candidate = best;
                    candidate.vertical = (candidate.vertical
                        + direction * perspective_steps[index])
                        .clamp(-100.0, 100.0);
                    let score = upright_objective(lines, width, height, candidate, regularization);
                    if score + 1e-5 < best_score {
                        best = candidate;
                        best_score = score;
                        improved = true;
                    }
                }
            }

            if enable_horizontal {
                for direction in [-1.0f32, 1.0] {
                    let mut candidate = best;
                    candidate.horizontal = (candidate.horizontal
                        + direction * perspective_steps[index])
                        .clamp(-100.0, 100.0);
                    let score = upright_objective(lines, width, height, candidate, regularization);
                    if score + 1e-5 < best_score {
                        best = candidate;
                        best_score = score;
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }
    }

    best
}

pub fn analyze_upright(
    image: &DynamicImage,
    mode: UprightMode,
    orientation_steps: u8,
) -> UprightAnalysis {
    let detected_lines = extract_structural_lines(image);
    if detected_lines.len() < 2 {
        return UprightAnalysis {
            rotation: 0.0,
            vertical: 0.0,
            horizontal: 0.0,
            confidence: 0.0,
            detected: false,
            line_count: detected_lines.len(),
        };
    }

    let width = image.width() as f32;
    let height = image.height() as f32;
    let mut horizontal_consensus = fit_axis_consensus(
        &detected_lines,
        StructuralAxis::Horizontal,
        width,
        height,
        None,
    );
    let mut vertical_consensus = fit_axis_consensus(
        &detected_lines,
        StructuralAxis::Vertical,
        width,
        height,
        None,
    );

    // A rectangular motif yields two orthogonal measurements of the same
    // centre rotation. If their independent RANSAC modes disagree, use the
    // stronger family only as an intercept prior and refit the weaker family.
    // This recovers faint stamp/window sides without ever substituting a
    // horizontal line for vertical perspective evidence (or vice versa).
    let axis_to_refit = horizontal_consensus
        .as_ref()
        .zip(vertical_consensus.as_ref())
        .filter(|(horizontal, vertical)| (horizontal.intercept - vertical.intercept).abs() > 4.5)
        .map(|(horizontal, vertical)| {
            if horizontal.score >= vertical.score {
                (StructuralAxis::Vertical, horizontal.intercept)
            } else {
                (StructuralAxis::Horizontal, vertical.intercept)
            }
        });
    if let Some((axis, intercept_hint)) = axis_to_refit {
        let refined =
            fit_axis_consensus(&detected_lines, axis, width, height, Some(intercept_hint));
        match axis {
            StructuralAxis::Horizontal => horizontal_consensus = refined,
            StructuralAxis::Vertical => vertical_consensus = refined,
        }
    }
    let paired_local_structure = horizontal_consensus
        .as_ref()
        .zip(vertical_consensus.as_ref())
        .is_some_and(|(horizontal, vertical)| {
            consensuses_are_orthogonal_peers(horizontal, vertical)
        });
    let horizontal_independent = horizontal_consensus
        .as_ref()
        .is_some_and(consensus_has_independent_structure);
    let vertical_independent = vertical_consensus
        .as_ref()
        .is_some_and(consensus_has_independent_structure);
    let horizontal_available =
        horizontal_consensus.is_some() && (horizontal_independent || paired_local_structure);
    let vertical_available =
        vertical_consensus.is_some() && (vertical_independent || paired_local_structure);
    let rotated_quarter_turn = orientation_steps % 2 == 1;
    let (mut use_vertical_lines, mut use_horizontal_lines) = match mode {
        UprightMode::Level | UprightMode::Auto | UprightMode::Full => {
            (vertical_available, horizontal_available)
        }
        UprightMode::Vertical if rotated_quarter_turn => (false, horizontal_available),
        UprightMode::Vertical => (vertical_available, false),
    };

    // At the image centre both orthogonal families should report the same
    // rotation. If two independently strong families disagree, keep the more
    // coherent one instead of averaging them into a correction neither
    // family supports.
    let axis_to_disable = if use_horizontal_lines && use_vertical_lines && !paired_local_structure {
        horizontal_consensus
            .as_ref()
            .zip(vertical_consensus.as_ref())
            .filter(|(horizontal, vertical)| {
                (horizontal.intercept - vertical.intercept).abs() > 4.5
            })
            .map(|(horizontal, vertical)| {
                if horizontal.score >= vertical.score {
                    StructuralAxis::Vertical
                } else {
                    StructuralAxis::Horizontal
                }
            })
    } else {
        None
    };
    match axis_to_disable {
        Some(StructuralAxis::Horizontal) => use_horizontal_lines = false,
        Some(StructuralAxis::Vertical) => use_vertical_lines = false,
        None => {}
    }

    let mut horizontal_lines = if use_horizontal_lines {
        horizontal_consensus
            .as_ref()
            .map(|consensus| consensus.lines.clone())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut vertical_lines = if use_vertical_lines {
        vertical_consensus
            .as_ref()
            .map(|consensus| consensus.lines.clone())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    if horizontal_lines.is_empty() && vertical_lines.is_empty() {
        return UprightAnalysis {
            rotation: 0.0,
            vertical: 0.0,
            horizontal: 0.0,
            confidence: 0.0,
            detected: false,
            line_count: 0,
        };
    }

    // Equalise the two axis families after consensus selection. Ten short
    // stamp edges in one direction should not drown out four equally reliable
    // edges in the perpendicular direction merely because Hough emitted more
    // candidates for them.
    if !horizontal_lines.is_empty() && !vertical_lines.is_empty() {
        let horizontal_weight = horizontal_lines.iter().map(|line| line.weight).sum::<f32>();
        let vertical_weight = vertical_lines.iter().map(|line| line.weight).sum::<f32>();
        let target_weight = (horizontal_weight * vertical_weight).sqrt();
        let horizontal_scale = target_weight / horizontal_weight.max(f32::EPSILON);
        let vertical_scale = target_weight / vertical_weight.max(f32::EPSILON);
        for line in &mut horizontal_lines {
            line.weight *= horizontal_scale;
        }
        for line in &mut vertical_lines {
            line.weight *= vertical_scale;
        }
    }

    let enable_vertical = use_vertical_lines
        && mode != UprightMode::Level
        && (!rotated_quarter_turn || mode != UprightMode::Vertical)
        && vertical_consensus
            .as_ref()
            .is_some_and(consensus_supports_perspective);
    let enable_horizontal = use_horizontal_lines
        && mode != UprightMode::Level
        && (rotated_quarter_turn || mode != UprightMode::Vertical)
        && horizontal_consensus
            .as_ref()
            .is_some_and(consensus_supports_perspective);
    let mut lines = horizontal_lines;
    lines.extend(vertical_lines);
    let regularization = match mode {
        UprightMode::Auto => 0.14,
        UprightMode::Level => 0.04,
        UprightMode::Vertical => 0.035,
        UprightMode::Full => 0.025,
    };
    let mut params = optimize_upright(
        &lines,
        width,
        height,
        enable_vertical,
        enable_horizontal,
        regularization,
    );
    if !enable_vertical {
        params.vertical = 0.0;
    }
    if !enable_horizontal {
        params.horizontal = 0.0;
    }

    let baseline =
        upright_objective(&lines, width, height, UprightParameters::default(), 0.0).sqrt();
    let final_error = upright_objective(&lines, width, height, params, 0.0).sqrt();
    let improvement = if baseline < 0.2 {
        1.0
    } else {
        (1.0 - final_error / baseline).clamp(0.0, 1.0)
    };
    let selected_consensuses = [
        use_horizontal_lines
            .then_some(horizontal_consensus.as_ref())
            .flatten(),
        use_vertical_lines
            .then_some(vertical_consensus.as_ref())
            .flatten(),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    let consensus_count = selected_consensuses.len().max(1) as f32;
    let agreement = selected_consensuses
        .iter()
        .map(|consensus| consensus.agreement)
        .sum::<f32>()
        / consensus_count;
    let spatial_coverage = selected_consensuses
        .iter()
        .map(|consensus| (consensus.spatial_cells as f32 / 6.0).clamp(0.0, 1.0))
        .sum::<f32>()
        / consensus_count;
    let count_strength = (lines.len() as f32 / 10.0).clamp(0.0, 1.0);
    let pixel_support = lines.iter().map(|line| line.support).sum::<usize>() as f32;
    let support_strength = (pixel_support / (width.min(height).max(1.0) * 0.8)).clamp(0.0, 1.0);
    let strength = count_strength * 0.6 + support_strength * 0.4;
    let confidence =
        (agreement * 0.3 + spatial_coverage * 0.25 + strength * 0.25 + improvement * 0.2)
            .clamp(0.0, 1.0);
    let detected = confidence >= 0.42;
    let round_tenth = |value: f32| {
        if value.abs() < 0.15 {
            0.0
        } else {
            (value * 10.0).round() / 10.0
        }
    };

    let (rotation, vertical, horizontal) = if detected {
        (
            round_tenth(params.rotation),
            round_tenth(params.vertical),
            round_tenth(params.horizontal),
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    UprightAnalysis {
        rotation,
        vertical,
        horizontal,
        confidence,
        detected,
        line_count: lines.len(),
    }
}
