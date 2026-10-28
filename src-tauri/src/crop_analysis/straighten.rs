//! Automatic straighten analysis.
//!
//! Detects dominant line angles and refines the best candidate into a bounded
//! rotation correction.

use image::DynamicImage;
use image::GrayImage;
use imageproc::edges::canny;
use imageproc::hough::LineDetectionOptions;
use imageproc::hough::PolarLine;
use imageproc::hough::detect_lines;
use serde::Serialize;

pub(super) const MAX_AUTO_ANGLE_DEGREES: f32 = 15.0;

pub(super) const AXIS_TOLERANCE_DEGREES: f32 = 18.0;

pub(super) const CLUSTER_RADIUS_DEGREES: f32 = 1.5;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StraightenAnalysis {
    pub angle: f32,
    pub confidence: f32,
    pub detected: bool,
    pub line_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AngleCandidate {
    pub(super) correction: f32,
    pub(super) support: usize,
    pub(super) weight: f32,
}

pub(super) fn axis_correction(direction_degrees: f32) -> Option<f32> {
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

pub(super) fn refine_candidate(edges: &GrayImage, line: PolarLine) -> Option<AngleCandidate> {
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
