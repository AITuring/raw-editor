use image::{DynamicImage, GrayImage};
use imageproc::edges::canny;
use imageproc::hough::{LineDetectionOptions, PolarLine, detect_lines};
use serde::{Deserialize, Serialize};

const MAX_AUTO_ANGLE_DEGREES: f32 = 15.0;
const AXIS_TOLERANCE_DEGREES: f32 = 18.0;
const CLUSTER_RADIUS_DEGREES: f32 = 1.5;
const UPRIGHT_AXIS_TOLERANCE_DEGREES: f32 = 25.0;
const MAX_UPRIGHT_LINES: usize = 48;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StraightenAnalysis {
    pub angle: f32,
    pub confidence: f32,
    pub detected: bool,
    pub line_count: usize,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuralAxis {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy)]
struct StructuralLine {
    start: (f32, f32),
    end: (f32, f32),
    axis: StructuralAxis,
    support: usize,
    weight: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct UprightParameters {
    rotation: f32,
    vertical: f32,
    horizontal: f32,
}

#[derive(Debug, Clone, Copy)]
struct AngleCandidate {
    correction: f32,
    support: usize,
    weight: f32,
}

fn axis_correction(direction_degrees: f32) -> Option<f32> {
    let correction = if direction_degrees.abs() <= AXIS_TOLERANCE_DEGREES {
        -direction_degrees
    } else if direction_degrees >= 90.0 - AXIS_TOLERANCE_DEGREES {
        90.0 - direction_degrees
    } else if direction_degrees <= -90.0 + AXIS_TOLERANCE_DEGREES {
        -90.0 - direction_degrees
    } else {
        return None;
    };

    (correction.abs() <= MAX_AUTO_ANGLE_DEGREES).then_some(correction)
}

fn refine_candidate(edges: &GrayImage, line: PolarLine) -> Option<AngleCandidate> {
    let (width, height) = edges.dimensions();
    let min_dim = width.min(height) as usize;
    let margin_x = (width as f32 * 0.025).round() as u32;
    let margin_y = (height as f32 * 0.025).round() as u32;
    let theta = (line.angle_in_degrees as f32).to_radians();
    let (sin_theta, cos_theta) = theta.sin_cos();

    let mut count = 0usize;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut sum_yy = 0.0f64;

    for y in margin_y..height.saturating_sub(margin_y) {
        for x in margin_x..width.saturating_sub(margin_x) {
            if edges.get_pixel(x, y)[0] == 0 {
                continue;
            }

            let distance = (x as f32 * cos_theta + y as f32 * sin_theta - line.r).abs();
            if distance > 1.75 {
                continue;
            }

            let xf = x as f64;
            let yf = y as f64;
            count += 1;
            sum_x += xf;
            sum_y += yf;
            sum_xx += xf * xf;
            sum_xy += xf * yf;
            sum_yy += yf * yf;
        }
    }

    let minimum_support = (min_dim as f32 * 0.045).round().max(14.0) as usize;
    if count < minimum_support {
        return None;
    }

    let n = count as f64;
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let covariance_xx = sum_xx / n - mean_x * mean_x;
    let covariance_xy = sum_xy / n - mean_x * mean_y;
    let covariance_yy = sum_yy / n - mean_y * mean_y;

    let direction = 0.5 * (2.0 * covariance_xy).atan2(covariance_xx - covariance_yy);
    let direction_degrees = direction.to_degrees() as f32;
    let correction = axis_correction(direction_degrees)?;

    let trace = covariance_xx + covariance_yy;
    let discriminant =
        ((covariance_xx - covariance_yy).powi(2) + 4.0 * covariance_xy.powi(2)).sqrt();
    let major = ((trace + discriminant) * 0.5).max(0.0);
    let minor = ((trace - discriminant) * 0.5).max(0.0);
    if major <= f64::EPSILON {
        return None;
    }

    let straightness = (1.0 - minor / major).clamp(0.0, 1.0) as f32;
    let span = (12.0 * major).sqrt() as f32;
    let coverage = (span / min_dim.max(1) as f32).clamp(0.12, 1.0);
    let weight = count as f32 * straightness.powi(2) * coverage;

    (weight > 0.0).then_some(AngleCandidate {
        correction,
        support: count,
        weight,
    })
}

pub fn analyze_straighten(image: &DynamicImage) -> StraightenAnalysis {
    let grayscale = image.to_luma8();
    let edges = canny(&grayscale, 35.0, 90.0);
    let min_dim = edges.width().min(edges.height());

    if min_dim < 32 {
        return StraightenAnalysis {
            angle: 0.0,
            confidence: 0.0,
            detected: false,
            line_count: 0,
        };
    }

    let lines = detect_lines(
        &edges,
        LineDetectionOptions {
            vote_threshold: (min_dim as f32 * 0.08).round().max(18.0) as u32,
            suppression_radius: 8,
        },
    );

    let candidates: Vec<AngleCandidate> = lines
        .into_iter()
        .filter(|line| {
            let angle = line.angle_in_degrees as f32;
            angle <= AXIS_TOLERANCE_DEGREES
                || angle >= 180.0 - AXIS_TOLERANCE_DEGREES
                || (angle - 90.0).abs() <= AXIS_TOLERANCE_DEGREES
        })
        .filter_map(|line| refine_candidate(&edges, line))
        .collect();

    if candidates.is_empty() {
        return StraightenAnalysis {
            angle: 0.0,
            confidence: 0.0,
            detected: false,
            line_count: 0,
        };
    }

    let total_weight = candidates
        .iter()
        .map(|candidate| candidate.weight)
        .sum::<f32>();
    let best_seed = candidates
        .iter()
        .max_by(|left, right| {
            let left_cluster_weight = candidates
                .iter()
                .filter(|candidate| {
                    (candidate.correction - left.correction).abs() <= CLUSTER_RADIUS_DEGREES
                })
                .map(|candidate| candidate.weight)
                .sum::<f32>();
            let right_cluster_weight = candidates
                .iter()
                .filter(|candidate| {
                    (candidate.correction - right.correction).abs() <= CLUSTER_RADIUS_DEGREES
                })
                .map(|candidate| candidate.weight)
                .sum::<f32>();
            left_cluster_weight.total_cmp(&right_cluster_weight)
        })
        .expect("candidates is not empty");

    let cluster: Vec<&AngleCandidate> = candidates
        .iter()
        .filter(|candidate| {
            (candidate.correction - best_seed.correction).abs() <= CLUSTER_RADIUS_DEGREES
        })
        .collect();
    let cluster_weight = cluster
        .iter()
        .map(|candidate| candidate.weight)
        .sum::<f32>();
    let cluster_support = cluster
        .iter()
        .map(|candidate| candidate.support)
        .sum::<usize>();
    let weighted_angle = cluster
        .iter()
        .map(|candidate| candidate.correction * candidate.weight)
        .sum::<f32>()
        / cluster_weight.max(f32::EPSILON);

    let agreement = cluster_weight / total_weight.max(f32::EPSILON);
    let strength = (cluster_support as f32 / (min_dim as f32 * 0.55)).clamp(0.0, 1.0);
    let confidence = (agreement * strength.sqrt()).clamp(0.0, 1.0);
    let detected = confidence >= 0.12;
    let angle = if detected {
        (weighted_angle.clamp(-MAX_AUTO_ANGLE_DEGREES, MAX_AUTO_ANGLE_DEGREES) * 10.0).round()
            / 10.0
    } else {
        0.0
    };

    StraightenAnalysis {
        angle,
        confidence,
        detected,
        line_count: cluster.len(),
    }
}

fn normalize_line_angle(mut angle: f32) -> f32 {
    while angle > 90.0 {
        angle -= 180.0;
    }
    while angle <= -90.0 {
        angle += 180.0;
    }
    angle
}

fn classify_structural_axis(direction_degrees: f32) -> Option<StructuralAxis> {
    let normalized = normalize_line_angle(direction_degrees);
    if normalized.abs() <= UPRIGHT_AXIS_TOLERANCE_DEGREES {
        Some(StructuralAxis::Horizontal)
    } else if (normalized.abs() - 90.0).abs() <= UPRIGHT_AXIS_TOLERANCE_DEGREES {
        Some(StructuralAxis::Vertical)
    } else {
        None
    }
}

fn refine_structural_line(edges: &GrayImage, line: PolarLine) -> Option<StructuralLine> {
    let (width, height) = edges.dimensions();
    let min_dim = width.min(height) as usize;
    let margin_x = (width as f32 * 0.02).round() as u32;
    let margin_y = (height as f32 * 0.02).round() as u32;
    let theta = (line.angle_in_degrees as f32).to_radians();
    let (sin_theta, cos_theta) = theta.sin_cos();

    let tangent_x = -sin_theta;
    let tangent_y = cos_theta;
    let mut points = Vec::<(f32, f32, f32)>::new();

    for y in margin_y..height.saturating_sub(margin_y) {
        for x in margin_x..width.saturating_sub(margin_x) {
            if edges.get_pixel(x, y)[0] == 0 {
                continue;
            }

            let distance = (x as f32 * cos_theta + y as f32 * sin_theta - line.r).abs();
            if distance > 1.75 {
                continue;
            }

            let xf = x as f32;
            let yf = y as f32;
            points.push((xf, yf, xf * tangent_x + yf * tangent_y));
        }
    }

    let minimum_support = (min_dim as f32 * 0.035).round().max(12.0) as usize;
    if points.len() < minimum_support {
        return None;
    }

    // A polar Hough line is infinite. Without a continuity check, unrelated
    // glyphs or repeated texture that happen to share an x/y coordinate are
    // merged into a fake full-height "structural" line. Keep only the
    // strongest locally continuous run along the Hough tangent.
    points.sort_by(|left, right| left.2.total_cmp(&right.2));
    let maximum_gap = 5.0f32;
    let mut run_start = 0usize;
    let mut best_start = 0usize;
    let mut best_end = 0usize;
    let mut best_score = 0.0f32;
    for run_end in 1..=points.len() {
        let ends_run =
            run_end == points.len() || points[run_end].2 - points[run_end - 1].2 > maximum_gap;
        if !ends_run {
            continue;
        }

        let support = run_end - run_start;
        let span = points[run_end - 1].2 - points[run_start].2;
        let score = span.max(0.0) * (support as f32).sqrt();
        if score > best_score {
            best_score = score;
            best_start = run_start;
            best_end = run_end;
        }
        run_start = run_end;
    }

    if best_end <= best_start {
        return None;
    }
    let points = &points[best_start..best_end];
    let count = points.len();
    let run_span = points[count - 1].2 - points[0].2;
    let minimum_span = min_dim as f32 * 0.14;
    if count < minimum_support || run_span < minimum_span {
        return None;
    }

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut sum_yy = 0.0f64;
    for &(x, y, _) in points {
        let xf = x as f64;
        let yf = y as f64;
        sum_x += xf;
        sum_y += yf;
        sum_xx += xf * xf;
        sum_xy += xf * yf;
        sum_yy += yf * yf;
    }

    let n = count as f64;
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let covariance_xx = sum_xx / n - mean_x * mean_x;
    let covariance_xy = sum_xy / n - mean_x * mean_y;
    let covariance_yy = sum_yy / n - mean_y * mean_y;
    let direction = 0.5 * (2.0 * covariance_xy).atan2(covariance_xx - covariance_yy);
    let direction_degrees = direction.to_degrees() as f32;
    let axis = classify_structural_axis(direction_degrees)?;

    let trace = covariance_xx + covariance_yy;
    let discriminant =
        ((covariance_xx - covariance_yy).powi(2) + 4.0 * covariance_xy.powi(2)).sqrt();
    let major = ((trace + discriminant) * 0.5).max(0.0);
    let minor = ((trace - discriminant) * 0.5).max(0.0);
    if major <= f64::EPSILON {
        return None;
    }

    let straightness = (1.0 - minor / major).clamp(0.0, 1.0) as f32;
    let span = (12.0 * major).sqrt() as f32;
    let coverage = (span / min_dim.max(1) as f32).clamp(0.0, 1.0);
    let continuity = (count as f32 / run_span.max(1.0)).clamp(0.0, 1.0);
    let weight = count as f32 * straightness.powi(2) * coverage * continuity.powi(2);
    if weight <= 0.0 {
        return None;
    }

    let direction_x = direction.cos() as f32;
    let direction_y = direction.sin() as f32;
    let half_span = span * 0.5;
    Some(StructuralLine {
        start: (
            mean_x as f32 - direction_x * half_span,
            mean_y as f32 - direction_y * half_span,
        ),
        end: (
            mean_x as f32 + direction_x * half_span,
            mean_y as f32 + direction_y * half_span,
        ),
        axis,
        support: count,
        weight,
    })
}

fn extract_structural_lines(image: &DynamicImage) -> Vec<StructuralLine> {
    let grayscale = image.to_luma8();
    let edges = canny(&grayscale, 35.0, 90.0);
    let min_dim = edges.width().min(edges.height());
    if min_dim < 32 {
        return Vec::new();
    }

    let detected = detect_lines(
        &edges,
        LineDetectionOptions {
            vote_threshold: (min_dim as f32 * 0.055).round().max(16.0) as u32,
            suppression_radius: 8,
        },
    );

    let mut refined = detected
        .into_iter()
        .filter(|line| {
            let angle = line.angle_in_degrees as f32;
            angle <= UPRIGHT_AXIS_TOLERANCE_DEGREES
                || angle >= 180.0 - UPRIGHT_AXIS_TOLERANCE_DEGREES
                || (angle - 90.0).abs() <= UPRIGHT_AXIS_TOLERANCE_DEGREES
        })
        .take(128)
        .filter_map(|line| refine_structural_line(&edges, line))
        .collect::<Vec<_>>();

    refined.sort_by(|left, right| right.weight.total_cmp(&left.weight));
    let mut unique = Vec::<StructuralLine>::new();
    for candidate in refined {
        let candidate_angle = normalize_line_angle(
            (candidate.end.1 - candidate.start.1)
                .atan2(candidate.end.0 - candidate.start.0)
                .to_degrees(),
        );
        let candidate_center = (
            (candidate.start.0 + candidate.end.0) * 0.5,
            (candidate.start.1 + candidate.end.1) * 0.5,
        );
        let is_duplicate = unique.iter().any(|existing| {
            if existing.axis != candidate.axis {
                return false;
            }
            let existing_angle = normalize_line_angle(
                (existing.end.1 - existing.start.1)
                    .atan2(existing.end.0 - existing.start.0)
                    .to_degrees(),
            );
            let existing_center = (
                (existing.start.0 + existing.end.0) * 0.5,
                (existing.start.1 + existing.end.1) * 0.5,
            );
            (candidate_angle - existing_angle).abs() < 1.2
                && (candidate_center.0 - existing_center.0)
                    .hypot(candidate_center.1 - existing_center.1)
                    < min_dim as f32 * 0.025
        });

        if !is_duplicate {
            unique.push(candidate);
            if unique.len() >= MAX_UPRIGHT_LINES {
                break;
            }
        }
    }
    unique
}

fn project_point(
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

fn line_residual_degrees(
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

fn upright_objective(
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

fn optimize_upright(
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

    let horizontal_count = detected_lines
        .iter()
        .filter(|line| line.axis == StructuralAxis::Horizontal)
        .count();
    let vertical_count = detected_lines.len() - horizontal_count;
    let rotated_quarter_turn = orientation_steps % 2 == 1;
    let minimum_axis_lines = if mode == UprightMode::Auto { 3 } else { 2 };
    let (use_vertical_lines, use_horizontal_lines, enable_vertical, enable_horizontal) = match mode
    {
        UprightMode::Level => (
            vertical_count >= minimum_axis_lines,
            horizontal_count >= minimum_axis_lines,
            false,
            false,
        ),
        UprightMode::Vertical if rotated_quarter_turn => {
            let has_horizontal_structure = horizontal_count >= minimum_axis_lines;
            (
                false,
                has_horizontal_structure,
                false,
                has_horizontal_structure,
            )
        }
        UprightMode::Vertical => {
            let has_vertical_structure = vertical_count >= minimum_axis_lines;
            (has_vertical_structure, false, has_vertical_structure, false)
        }
        UprightMode::Auto | UprightMode::Full => {
            let has_vertical_structure = vertical_count >= minimum_axis_lines;
            let has_horizontal_structure = horizontal_count >= minimum_axis_lines;
            (
                has_vertical_structure,
                has_horizontal_structure,
                has_vertical_structure,
                has_horizontal_structure,
            )
        }
    };
    let lines = detected_lines
        .into_iter()
        .filter(|line| match line.axis {
            StructuralAxis::Horizontal => use_horizontal_lines,
            StructuralAxis::Vertical => use_vertical_lines,
        })
        .collect::<Vec<_>>();
    if lines.len() < minimum_axis_lines {
        return UprightAnalysis {
            rotation: 0.0,
            vertical: 0.0,
            horizontal: 0.0,
            confidence: 0.0,
            detected: false,
            line_count: lines.len(),
        };
    }
    let regularization = if mode == UprightMode::Auto {
        0.1
    } else {
        0.025
    };
    let params = optimize_upright(
        &lines,
        image.width() as f32,
        image.height() as f32,
        enable_vertical,
        enable_horizontal,
        regularization,
    );

    let baseline = upright_objective(
        &lines,
        image.width() as f32,
        image.height() as f32,
        UprightParameters::default(),
        0.0,
    )
    .sqrt();
    let final_error = upright_objective(
        &lines,
        image.width() as f32,
        image.height() as f32,
        params,
        0.0,
    )
    .sqrt();
    let improvement = if baseline < 0.2 {
        1.0
    } else {
        (1.0 - final_error / baseline).clamp(0.0, 1.0)
    };
    let support = lines.iter().map(|line| line.support).sum::<usize>() as f32;
    let strength = (support / (image.width().min(image.height()) as f32 * 0.8)).clamp(0.0, 1.0);
    let coverage = (lines.len() as f32 / 8.0).clamp(0.0, 1.0);
    let confidence = (strength * 0.45 + coverage * 0.35 + improvement * 0.2).clamp(0.0, 1.0);
    let round_tenth = |value: f32| (value * 10.0).round() / 10.0;

    UprightAnalysis {
        rotation: round_tenth(params.rotation),
        vertical: round_tenth(params.vertical),
        horizontal: round_tenth(params.horizontal),
        confidence,
        detected: confidence >= 0.35,
        line_count: lines.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    use imageproc::drawing::draw_line_segment_mut;

    #[test]
    fn detects_tilted_horizontal_structure() {
        let mut image = GrayImage::from_pixel(640, 420, Luma([12]));
        let slope = 4.0f32.to_radians().tan();

        for base_y in [90.0, 205.0, 320.0] {
            draw_line_segment_mut(
                &mut image,
                (28.0, base_y),
                (612.0, base_y + 584.0 * slope),
                Luma([245]),
            );
        }

        let result = analyze_straighten(&DynamicImage::ImageLuma8(image));
        assert!(
            result.detected,
            "analysis should detect the synthetic horizon: {result:?}"
        );
        assert!(
            (result.angle + 4.0).abs() <= 0.8,
            "unexpected correction: {result:?}"
        );
    }

    #[test]
    fn detects_tilted_vertical_structure() {
        let mut image = GrayImage::from_pixel(640, 420, Luma([12]));
        let lean = 3.0f32.to_radians().tan();

        for base_x in [135.0, 320.0, 505.0] {
            draw_line_segment_mut(
                &mut image,
                (base_x, 25.0),
                (base_x + 370.0 * lean, 395.0),
                Luma([245]),
            );
        }

        let result = analyze_straighten(&DynamicImage::ImageLuma8(image));
        assert!(
            result.detected,
            "analysis should detect the synthetic verticals: {result:?}"
        );
        assert!(
            (result.angle - 3.0).abs() <= 0.8,
            "unexpected correction: {result:?}"
        );
    }

    #[test]
    fn ignores_images_without_structural_lines() {
        let image = DynamicImage::ImageLuma8(GrayImage::from_pixel(320, 240, Luma([128])));
        let result = analyze_straighten(&image);
        assert!(!result.detected);
        assert_eq!(result.angle, 0.0);
    }

    fn synthetic_grid_lines(
        width: f32,
        height: f32,
        distortion: UprightParameters,
    ) -> Vec<StructuralLine> {
        let mut lines = Vec::new();
        for x in [width * 0.2, width * 0.4, width * 0.6, width * 0.8] {
            lines.push(StructuralLine {
                start: project_point((x, height * 0.08), width, height, distortion).unwrap(),
                end: project_point((x, height * 0.92), width, height, distortion).unwrap(),
                axis: StructuralAxis::Vertical,
                support: 240,
                weight: 240.0,
            });
        }
        for y in [height * 0.2, height * 0.4, height * 0.6, height * 0.8] {
            lines.push(StructuralLine {
                start: project_point((width * 0.08, y), width, height, distortion).unwrap(),
                end: project_point((width * 0.92, y), width, height, distortion).unwrap(),
                axis: StructuralAxis::Horizontal,
                support: 240,
                weight: 240.0,
            });
        }
        lines
    }

    #[test]
    fn upright_optimizer_reverses_vertical_and_horizontal_keystone() {
        let width = 800.0;
        let height = 600.0;
        let distortion = UprightParameters {
            vertical: 28.0,
            horizontal: -22.0,
            ..UprightParameters::default()
        };
        let lines = synthetic_grid_lines(width, height, distortion);
        let result = optimize_upright(&lines, width, height, true, true, 0.0);

        assert!(
            (result.vertical + distortion.vertical).abs() <= 2.0,
            "{result:?}"
        );
        assert!(
            (result.horizontal + distortion.horizontal).abs() <= 2.0,
            "{result:?}"
        );
        assert!(result.rotation.abs() <= 0.5, "{result:?}");
    }

    #[test]
    fn upright_optimizer_levels_rotated_structure() {
        let width = 800.0;
        let height = 600.0;
        let distortion = UprightParameters {
            rotation: 6.0,
            ..UprightParameters::default()
        };
        let lines = synthetic_grid_lines(width, height, distortion);
        let result = optimize_upright(&lines, width, height, false, false, 0.0);

        assert!(
            (result.rotation + distortion.rotation).abs() <= 0.3,
            "{result:?}"
        );
        assert_eq!(result.vertical, 0.0);
        assert_eq!(result.horizontal, 0.0);
    }

    #[test]
    fn upright_analysis_detects_projected_grid() {
        let width = 800u32;
        let height = 600u32;
        let distortion = UprightParameters {
            vertical: 24.0,
            horizontal: -16.0,
            ..UprightParameters::default()
        };
        let mut image = GrayImage::from_pixel(width, height, Luma([14]));
        for line in synthetic_grid_lines(width as f32, height as f32, distortion) {
            draw_line_segment_mut(&mut image, line.start, line.end, Luma([242]));
        }

        let result = analyze_upright(&DynamicImage::ImageLuma8(image), UprightMode::Full, 0);
        assert!(result.detected, "{result:?}");
        assert!(
            (result.vertical + distortion.vertical).abs() <= 7.0,
            "{result:?}"
        );
        assert!(
            (result.horizontal + distortion.horizontal).abs() <= 7.0,
            "{result:?}"
        );
    }

    #[test]
    fn upright_modes_respect_their_axis_contract() {
        let width = 800u32;
        let height = 600u32;
        let distortion = UprightParameters {
            rotation: 5.0,
            vertical: 24.0,
            horizontal: -16.0,
        };
        let mut image = GrayImage::from_pixel(width, height, Luma([14]));
        for line in synthetic_grid_lines(width as f32, height as f32, distortion) {
            draw_line_segment_mut(&mut image, line.start, line.end, Luma([242]));
        }
        let image = DynamicImage::ImageLuma8(image);

        let auto = analyze_upright(&image, UprightMode::Auto, 0);
        let level = analyze_upright(&image, UprightMode::Level, 0);
        let vertical = analyze_upright(&image, UprightMode::Vertical, 0);
        let full = analyze_upright(&image, UprightMode::Full, 0);

        assert!(auto.detected, "{auto:?}");
        assert!(
            (auto.rotation + distortion.rotation).abs() <= 2.0,
            "{auto:?}"
        );
        assert!(
            (auto.vertical + distortion.vertical).abs() <= 6.0,
            "{auto:?}"
        );
        assert!(
            (auto.horizontal + distortion.horizontal).abs() <= 6.0,
            "{auto:?}"
        );
        assert!(level.detected, "{level:?}");
        assert_eq!(level.vertical, 0.0);
        assert_eq!(level.horizontal, 0.0);
        assert!(
            (level.rotation + distortion.rotation).abs() <= 4.0,
            "{level:?}"
        );
        assert!(vertical.detected, "{vertical:?}");
        assert_eq!(vertical.horizontal, 0.0);
        assert!(
            (vertical.rotation + distortion.rotation).abs() <= 2.0,
            "{vertical:?}"
        );
        assert!(
            (vertical.vertical + distortion.vertical).abs() <= 6.0,
            "{vertical:?}"
        );
        assert!(full.detected, "{full:?}");
        assert!(
            (full.rotation + distortion.rotation).abs() <= 2.0,
            "{full:?}"
        );
        assert!(
            (full.vertical + distortion.vertical).abs() <= 6.0,
            "{full:?}"
        );
        assert!(
            (full.horizontal + distortion.horizontal).abs() <= 6.0,
            "{full:?}"
        );
    }

    #[test]
    fn upright_rejects_disconnected_decorative_marks() {
        let width = 600u32;
        let height = 1000u32;
        let mut image = GrayImage::from_pixel(width, height, Luma([18]));
        for (base_x, slope) in [(130.0f32, -0.12f32), (300.0, 0.0), (470.0, 0.12)] {
            for y in (40..940).step_by(65) {
                let y = y as f32;
                let x = base_x + (y - height as f32 * 0.5) * slope;
                draw_line_segment_mut(
                    &mut image,
                    (x, y),
                    (x + slope * 22.0, y + 22.0),
                    Luma([240]),
                );
            }
        }

        let image = DynamicImage::ImageLuma8(image);
        for mode in [
            UprightMode::Auto,
            UprightMode::Level,
            UprightMode::Vertical,
            UprightMode::Full,
        ] {
            let result = analyze_upright(&image, mode, 0);
            assert!(
                !result.detected,
                "decorative marks are not {mode:?} perspective guides: {result:?}"
            );
            assert_eq!(result.rotation, 0.0);
            assert_eq!(result.vertical, 0.0);
            assert_eq!(result.horizontal, 0.0);
        }
    }

    #[test]
    fn vertical_mode_does_not_use_horizontal_lines_as_vertical_evidence() {
        let mut image = GrayImage::from_pixel(640, 420, Luma([12]));
        let slope = 5.0f32.to_radians().tan();
        for base_y in [90.0, 205.0, 320.0] {
            draw_line_segment_mut(
                &mut image,
                (28.0, base_y),
                (612.0, base_y + 584.0 * slope),
                Luma([245]),
            );
        }
        let image = DynamicImage::ImageLuma8(image);

        let vertical = analyze_upright(&image, UprightMode::Vertical, 0);
        assert!(!vertical.detected, "{vertical:?}");
        assert_eq!(vertical.rotation, 0.0);
        assert_eq!(vertical.vertical, 0.0);
        assert_eq!(vertical.horizontal, 0.0);

        let level = analyze_upright(&image, UprightMode::Level, 0);
        assert!(level.detected, "{level:?}");
        assert!((level.rotation + 5.0).abs() <= 0.8, "{level:?}");
    }
}
