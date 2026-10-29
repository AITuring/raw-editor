//! Structural line extraction for Upright.
//!
//! Edge detection, line refinement and axis classification, plus the line
//! geometry helpers used by the axis consensus fit.

use image::DynamicImage;
use image::GrayImage;
use imageproc::edges::canny;
use imageproc::hough::LineDetectionOptions;
use imageproc::hough::PolarLine;
use imageproc::hough::detect_lines;

use super::{StructuralAxis, StructuralLine, UPRIGHT_AXIS_TOLERANCE_DEGREES, normalize_line_angle};

pub(super) const MAX_UPRIGHT_LINES: usize = 96;

pub(super) const LOCAL_STRUCTURE_SPAN_RATIO: f32 = 0.04;

pub(super) fn classify_structural_axis(direction_degrees: f32) -> Option<StructuralAxis> {
    let normalized = normalize_line_angle(direction_degrees);
    if normalized.abs() <= UPRIGHT_AXIS_TOLERANCE_DEGREES {
        Some(StructuralAxis::Horizontal)
    } else if (normalized.abs() - 90.0).abs() <= UPRIGHT_AXIS_TOLERANCE_DEGREES {
        Some(StructuralAxis::Vertical)
    } else {
        None
    }
}

pub(super) fn refine_structural_line(
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

pub(super) fn structural_edges(image: &DynamicImage) -> GrayImage {
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

pub(super) fn extract_structural_lines(image: &DynamicImage) -> Vec<StructuralLine> {
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

pub(super) fn structural_line_direction(line: &StructuralLine) -> f32 {
    normalize_line_angle(
        (line.end.1 - line.start.1)
            .atan2(line.end.0 - line.start.0)
            .to_degrees(),
    )
}

pub(super) fn structural_line_residual(line: &StructuralLine) -> f32 {
    let direction = structural_line_direction(line);
    match line.axis {
        StructuralAxis::Horizontal => direction,
        StructuralAxis::Vertical if direction >= 0.0 => direction - 90.0,
        StructuralAxis::Vertical => direction + 90.0,
    }
}

pub(super) fn structural_line_length(line: &StructuralLine) -> f32 {
    (line.end.0 - line.start.0).hypot(line.end.1 - line.start.1)
}

pub(super) fn structural_line_center(line: &StructuralLine) -> (f32, f32) {
    (
        (line.start.0 + line.end.0) * 0.5,
        (line.start.1 + line.end.1) * 0.5,
    )
}
