use image::{DynamicImage, GrayImage};
use imageproc::edges::canny;
use imageproc::hough::{LineDetectionOptions, PolarLine, detect_lines};
use serde::{Deserialize, Serialize};

const MAX_AUTO_ANGLE_DEGREES: f32 = 15.0;
const AXIS_TOLERANCE_DEGREES: f32 = 18.0;
const CLUSTER_RADIUS_DEGREES: f32 = 1.5;
const UPRIGHT_AXIS_TOLERANCE_DEGREES: f32 = 25.0;
const MAX_UPRIGHT_LINES: usize = 96;
const LOCAL_STRUCTURE_SPAN_RATIO: f32 = 0.04;
const LONG_STRUCTURE_SPAN_RATIO: f32 = 0.14;
const AXIS_CONSENSUS_TOLERANCE_DEGREES: f32 = 2.4;

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

#[derive(Debug, Clone)]
struct AxisConsensus {
    axis: StructuralAxis,
    lines: Vec<StructuralLine>,
    intercept: f32,
    slope: f32,
    perspective_model: bool,
    agreement: f32,
    coordinate_span: f32,
    spatial_cells: usize,
    framed_line_count: usize,
    long_line_count: usize,
    max_span_ratio: f32,
    rms_error: f32,
    score: f32,
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

fn refine_structural_line(
    edges: &GrayImage,
    edge_points: &[(u32, u32)],
    line: PolarLine,
) -> Option<StructuralLine> {
    let (width, height) = edges.dimensions();
    let min_dim = width.min(height) as usize;
    let margin_x = (width as f32 * 0.02).round() as u32;
    let margin_y = (height as f32 * 0.02).round() as u32;
    let theta = (line.angle_in_degrees as f32).to_radians();
    let (sin_theta, cos_theta) = theta.sin_cos();

    let tangent_x = -sin_theta;
    let tangent_y = cos_theta;
    let mut points = Vec::<(f32, f32, f32)>::new();

    for &(x, y) in edge_points {
        if x < margin_x
            || x >= width.saturating_sub(margin_x)
            || y < margin_y
            || y >= height.saturating_sub(margin_y)
        {
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

    let minimum_support = (min_dim as f32 * 0.025).round().max(10.0) as usize;
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
    let minimum_span = min_dim as f32 * LOCAL_STRUCTURE_SPAN_RATIO;
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

fn structural_edges(image: &DynamicImage) -> GrayImage {
    let grayscale = image.to_luma8();
    let mut merged = canny(&grayscale, 28.0, 72.0);
    let rgb = image.to_rgb8();

    // Coloured architectural details can be nearly invisible after luma
    // conversion (the red seals in the regression fixture are a practical
    // example). Merge channel-specific edges so geometry analysis sees shape
    // contrast without depending on a detail's luminance.
    for channel in 0..3 {
        let plane = GrayImage::from_fn(rgb.width(), rgb.height(), |x, y| {
            image::Luma([rgb.get_pixel(x, y)[channel]])
        });
        let channel_edges = canny(&plane, 14.0, 38.0);
        for (target, source) in merged.pixels_mut().zip(channel_edges.pixels()) {
            target[0] = target[0].max(source[0]);
        }
    }

    merged
}

fn extract_structural_lines(image: &DynamicImage) -> Vec<StructuralLine> {
    let edges = structural_edges(image);
    let min_dim = edges.width().min(edges.height());
    if min_dim < 32 {
        return Vec::new();
    }
    let edge_points = edges
        .enumerate_pixels()
        .filter_map(|(x, y, pixel)| (pixel[0] != 0).then_some((x, y)))
        .collect::<Vec<_>>();

    let detected = detect_lines(
        &edges,
        LineDetectionOptions {
            vote_threshold: (min_dim as f32 * 0.035).round().max(14.0) as u32,
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
        .take(256)
        .filter_map(|line| refine_structural_line(&edges, &edge_points, line))
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

fn structural_line_direction(line: &StructuralLine) -> f32 {
    normalize_line_angle(
        (line.end.1 - line.start.1)
            .atan2(line.end.0 - line.start.0)
            .to_degrees(),
    )
}

fn structural_line_residual(line: &StructuralLine) -> f32 {
    let direction = structural_line_direction(line);
    match line.axis {
        StructuralAxis::Horizontal => direction,
        StructuralAxis::Vertical if direction >= 0.0 => direction - 90.0,
        StructuralAxis::Vertical => direction + 90.0,
    }
}

fn structural_line_length(line: &StructuralLine) -> f32 {
    (line.end.0 - line.start.0).hypot(line.end.1 - line.start.1)
}

fn structural_line_center(line: &StructuralLine) -> (f32, f32) {
    (
        (line.start.0 + line.end.0) * 0.5,
        (line.start.1 + line.end.1) * 0.5,
    )
}

fn axis_coordinate(line: &StructuralLine, width: f32, height: f32) -> f32 {
    let center = structural_line_center(line);
    match line.axis {
        StructuralAxis::Vertical => center.0 / width.max(1.0) * 2.0 - 1.0,
        StructuralAxis::Horizontal => center.1 / height.max(1.0) * 2.0 - 1.0,
    }
}

fn spatial_cell(line: &StructuralLine, width: f32, height: f32) -> usize {
    let center = structural_line_center(line);
    let column = ((center.0 / width.max(1.0) * 4.0).floor() as i32).clamp(0, 3) as usize;
    let row = ((center.1 / height.max(1.0) * 4.0).floor() as i32).clamp(0, 3) as usize;
    row * 4 + column
}

fn orthogonal_frame_support(line: &StructuralLine, lines: &[StructuralLine]) -> usize {
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

fn fit_axis_consensus(
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

fn consensus_has_independent_structure(consensus: &AxisConsensus) -> bool {
    consensus.long_line_count >= 2
        || consensus.max_span_ratio >= 0.45
        || (consensus.lines.len() >= 6
            && consensus.framed_line_count >= 4
            && consensus.spatial_cells >= 4
            && consensus.agreement >= 0.35)
}

fn consensus_has_distributed_local_structure(consensus: &AxisConsensus) -> bool {
    consensus.lines.len() >= 4
        && consensus.spatial_cells >= 3
        && consensus.agreement >= 0.24
        && consensus.rms_error <= 1.6
}

fn consensuses_are_orthogonal_peers(horizontal: &AxisConsensus, vertical: &AxisConsensus) -> bool {
    horizontal.axis == StructuralAxis::Horizontal
        && vertical.axis == StructuralAxis::Vertical
        && (horizontal.intercept - vertical.intercept).abs() <= 3.5
        && consensus_has_distributed_local_structure(horizontal)
        && consensus_has_distributed_local_structure(vertical)
}

fn consensus_supports_perspective(consensus: &AxisConsensus) -> bool {
    consensus.perspective_model
        && consensus.lines.len() >= 4
        && consensus.spatial_cells >= 3
        && consensus.coordinate_span >= 0.55
        && consensus.slope.abs() * consensus.coordinate_span >= 1.4
        && consensus.rms_error <= 1.6
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};
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
    fn upright_uses_distributed_coloured_rectangles_as_geometry_evidence() {
        let width = 800u32;
        let height = 600u32;
        let distortion = UprightParameters {
            rotation: 4.0,
            ..UprightParameters::default()
        };
        // The two colours deliberately have similar luma. Grayscale-only edge
        // detection largely loses these marks, while channel edges retain the
        // repeated rectangular geometry used by stamps, windows and tiles.
        let mut image = RgbImage::from_pixel(width, height, Rgb([166, 105, 70]));
        for (center_x, center_y) in [
            (96.0f32, 92.0f32),
            (304.0, 128.0),
            (520.0, 84.0),
            (704.0, 142.0),
            (142.0, 258.0),
            (366.0, 224.0),
            (646.0, 284.0),
            (82.0, 424.0),
            (282.0, 372.0),
            (488.0, 448.0),
            (714.0, 394.0),
            (190.0, 536.0),
            (410.0, 514.0),
            (650.0, 548.0),
        ] {
            let corners = [
                (center_x - 29.0, center_y - 29.0),
                (center_x + 29.0, center_y - 29.0),
                (center_x + 29.0, center_y + 29.0),
                (center_x - 29.0, center_y + 29.0),
            ]
            .map(|point| project_point(point, width as f32, height as f32, distortion).unwrap());
            for index in 0..corners.len() {
                draw_line_segment_mut(
                    &mut image,
                    corners[index],
                    corners[(index + 1) % corners.len()],
                    Rgb([214, 65, 65]),
                );
            }
        }
        let image = DynamicImage::ImageRgb8(image);

        let auto = analyze_upright(&image, UprightMode::Auto, 0);
        assert!(auto.detected, "{auto:?}");
        assert!(
            (auto.rotation + distortion.rotation).abs() <= 2.0,
            "{auto:?}"
        );
        assert_eq!(auto.vertical, 0.0);
        assert_eq!(auto.horizontal, 0.0);

        let level = analyze_upright(&image, UprightMode::Level, 0);
        assert!(level.detected, "{level:?}");
        assert!(
            (level.rotation + distortion.rotation).abs() <= 2.0,
            "{level:?}"
        );
        assert_eq!(level.vertical, 0.0);
        assert_eq!(level.horizontal, 0.0);

        let vertical = analyze_upright(&image, UprightMode::Vertical, 0);
        assert!(vertical.detected, "{vertical:?}");
        assert!(
            (vertical.rotation + distortion.rotation).abs() <= 2.0,
            "{vertical:?}"
        );
        assert_eq!(vertical.vertical, 0.0);
        assert_eq!(vertical.horizontal, 0.0);
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
