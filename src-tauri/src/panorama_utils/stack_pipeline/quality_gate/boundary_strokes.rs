//! Boundary stroke alignment across Owner_Region boundaries (task 15.6).
//!
//! Traces owner-region boundaries and fits paired slanted edges on both sides
//! of each boundary sample to measure the stroke misalignment.

use super::super::degradation;
use image::GrayImage;
use image::Rgb32FImage;
use std::collections::BTreeSet;

use super::OwnerRegions;

// ==================== task 15.6: boundary stroke and confidence ====================
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BoundaryStrokeMeasurement {
    pub world: (f64, f64),
    pub normal_error_px: f64,
    pub orientation_error_deg: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryStrokeReport {
    pub measurements: Vec<BoundaryStrokeMeasurement>,
    pub pairable_count: usize,
    pub unpairable_count: usize,
    pub unmeasurable: Vec<((f64, f64), &'static str)>,
    pub p95_error_px: f64,
    pub max_error_px: f64,
}

pub(crate) fn trace_moore_boundary(
    labels: &OwnerRegions,
    left_label: u32,
    right_label: u32,
) -> Result<Vec<(u32, u32)>, &'static str> {
    if left_label == 0 || right_label == 0 || left_label == right_label {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let w = labels.width as usize;
    let h = labels.height as usize;
    let touches = |x: usize, y: usize| -> bool {
        let i = y * w + x;
        if labels.labels[i] != left_label && labels.labels[i] != right_label {
            return false;
        }
        (-1i32..=1)
            .flat_map(|dy| (-1i32..=1).map(move |dx| (dx, dy)))
            .filter(|(dx, dy)| *dx != 0 || *dy != 0)
            .any(|(dx, dy)| {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                nx >= 0
                    && ny >= 0
                    && (nx as usize) < w
                    && (ny as usize) < h
                    && labels.labels[ny as usize * w + nx as usize]
                        == if labels.labels[i] == left_label {
                            right_label
                        } else {
                            left_label
                        }
            })
    };
    let start = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .find(|&(x, y)| touches(x, y))
        .ok_or(degradation::BOUNDARY_NO_PAIRABLE_EDGE)?;
    let dirs: [(i32, i32); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];
    let mut result = Vec::new();
    let mut current = start;
    let mut backtrack = 4usize;
    let mut states = BTreeSet::new();
    let max_steps = w.saturating_mul(h).saturating_mul(8).max(1);
    for _ in 0..max_steps {
        if !touches(current.0, current.1) {
            break;
        }
        if !states.insert((current, backtrack)) {
            break;
        }
        result.push((current.0 as u32, current.1 as u32));
        let mut found = None;
        for offset in 1..=8 {
            let index = (backtrack + offset) % 8;
            let nx = current.0 as i32 + dirs[index].0;
            let ny = current.1 as i32 + dirs[index].1;
            if nx >= 0
                && ny >= 0
                && (nx as usize) < w
                && (ny as usize) < h
                && touches(nx as usize, ny as usize)
            {
                found = Some(((nx as usize, ny as usize), (index + 4) % 8));
                break;
            }
        }
        let Some((next, next_backtrack)) = found else {
            break;
        };
        current = next;
        backtrack = next_backtrack;
        if current == start && result.len() > 2 {
            break;
        }
    }
    if result.len() < 2 {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    Ok(result)
}

pub(super) fn luminance(pixel: image::Rgb<f32>) -> f64 {
    0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
}

/// Fit the same oriented edge in the two 16px owner bands and compare its
/// intercept at the seam. Contrast mass along the seam normal is not an edge
/// position: it changes under illumination even for a continuous stroke.
pub(crate) fn paired_boundary_edge_error(
    image: &Rgb32FImage,
    x: u32,
    y: u32,
    normal: (f64, f64),
) -> Result<(f64, f64), &'static str> {
    paired_boundary_edge_error_with_coverage(image, None, x, y, normal)
}

#[derive(Debug)]
pub(crate) struct BoundaryEdgeFits {
    pub(crate) left: (f64, f64),
    pub(crate) right: (f64, f64),
    pub(crate) crossing: f64,
    pub(crate) angle: f64,
}

pub(crate) fn paired_boundary_edge_error_with_coverage(
    image: &Rgb32FImage,
    coverage: Option<&GrayImage>,
    x: u32,
    y: u32,
    normal: (f64, f64),
) -> Result<(f64, f64), &'static str> {
    let fit = paired_boundary_edge_fits_with_coverage(image, coverage, x, y, normal)?;
    // Both fitted contours must actually cross the sampled ownership seam.
    // Nearly parallel arcs can otherwise extrapolate different circle/ring
    // segments to a distant intersection and report a fictitious large jump.
    if (fit.left.0 / fit.crossing).abs() > 16.0 || (fit.right.0 / fit.crossing).abs() > 16.0 {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    Ok(((fit.left.0 - fit.right.0).abs(), fit.angle))
}

pub(crate) fn paired_boundary_edge_fits_with_coverage(
    image: &Rgb32FImage,
    coverage: Option<&GrayImage>,
    x: u32,
    y: u32,
    normal: (f64, f64),
) -> Result<BoundaryEdgeFits, &'static str> {
    if coverage.is_some_and(|mask| mask.dimensions() != image.dimensions()) {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    #[derive(Clone, Copy)]
    struct Edge {
        d: f64,
        px: f64,
        py: f64,
        angle: f64,
        weight: f64,
    }
    let length = normal.0.hypot(normal.1);
    if length < 1e-8 {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let normal = (normal.0 / length, normal.1 / length);
    let tangent = (-normal.1, normal.0);
    let (width, height) = image.dimensions();
    let mut sides = [Vec::<Edge>::new(), Vec::<Edge>::new()];
    for (side_index, side) in [-1.0, 1.0].into_iter().enumerate() {
        for distance in 2..=16 {
            for along in -16..=16 {
                let px = side * distance as f64 * normal.0 + along as f64 * tangent.0;
                let py = side * distance as f64 * normal.1 + along as f64 * tangent.1;
                let ix = (x as f64 + px).round() as i32;
                let iy = (y as f64 + py).round() as i32;
                if ix < 1 || iy < 1 || ix + 1 >= width as i32 || iy + 1 >= height as i32 {
                    continue;
                }
                // A coverage/transparent transition is not a photographed
                // stroke. Every gradient stencil must belong to source data;
                // covered black pixels remain valid contrast evidence.
                if coverage.is_some_and(|mask| {
                    [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
                        .iter()
                        .any(|&(dx, dy)| mask.get_pixel((ix + dx) as u32, (iy + dy) as u32)[0] == 0)
                }) {
                    continue;
                }
                let gx = luminance(*image.get_pixel((ix + 1) as u32, iy as u32))
                    - luminance(*image.get_pixel((ix - 1) as u32, iy as u32));
                let gy = luminance(*image.get_pixel(ix as u32, (iy + 1) as u32))
                    - luminance(*image.get_pixel(ix as u32, (iy - 1) as u32));
                let weight = gx.hypot(gy) * 0.5;
                if weight >= 0.15 {
                    // The native Catmull–Rom footprint has radius two and
                    // the gradient adds one pixel. A covered interpolation
                    // halo next to transparency is still a support edge,
                    // rather than a photographed stroke.
                    if coverage.is_some_and(|mask| {
                        (-3..=3).any(|dy| {
                            (-3..=3).any(|dx| {
                                let (sx, sy) = (ix + dx, iy + dy);
                                sx < 0
                                    || sy < 0
                                    || sx >= width as i32
                                    || sy >= height as i32
                                    || mask.get_pixel(sx as u32, sy as u32)[0] == 0
                            })
                        })
                    }) {
                        continue;
                    }
                    sides[side_index].push(Edge {
                        d: px * normal.0 + py * normal.1,
                        px,
                        py,
                        angle: gy.atan2(gx),
                        weight,
                    });
                }
            }
        }
    }
    if sides.iter().any(Vec::is_empty) {
        return Err(degradation::BOUNDARY_LOW_CONTRAST);
    }
    let angle_difference = |a: f64, b: f64| (a - b).sin().atan2((a - b).cos()).abs();
    let tolerance = 10.0f64.to_radians();
    let mut best = (0.0, 0.0);
    for bin in 0..72 {
        let angle = bin as f64 * 5.0f64.to_radians();
        let support = sides
            .iter()
            .map(|side| {
                side.iter()
                    .filter(|p| angle_difference(p.angle, angle) <= tolerance)
                    .map(|p| p.weight)
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
        if support > best.0 {
            best = (support, angle);
        }
    }
    if best.0 <= 0.0 {
        return Err(degradation::BOUNDARY_ORIENTATION_MISMATCH);
    }
    let n = (best.1.cos(), best.1.sin());
    // Select a contour supported in both owner bands before fitting it.
    // Independent strongest peaks can select different concentric rings or
    // parallel strokes when their contrast changes across the ownership seam.
    // If no contour band is shared, retain independent peaks so an actually
    // displaced isolated edge remains a measurable failure.
    let mut shared_contour_support = false;
    let shared_peak = sides
        .iter()
        .flatten()
        .filter(|p| angle_difference(p.angle, best.1) <= tolerance)
        .map(|p| (n.0 * p.px + n.1 * p.py).floor() as i32)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|bin| {
            let peak = bin as f64 + 0.5;
            let samples = sides.each_ref().map(|side| {
                let points = side
                    .iter()
                    .filter(|p| {
                        angle_difference(p.angle, best.1) <= tolerance
                            && (n.0 * p.px + n.1 * p.py - peak).abs() <= 2.5
                    })
                    .collect::<Vec<_>>();
                let span = points
                    .iter()
                    .map(|p| p.d)
                    .max_by(f64::total_cmp)
                    .zip(points.iter().map(|p| p.d).min_by(f64::total_cmp))
                    .map(|(max, min)| max - min)
                    .unwrap_or(0.0);
                (
                    points.len(),
                    if span < 4.0 {
                        0.0
                    } else {
                        points.iter().map(|p| p.weight).sum::<f64>()
                    },
                )
            });
            shared_contour_support |= samples.iter().all(|sample| sample.0 >= 4);
            (
                samples[0].1.min(samples[1].1),
                samples[0].1 + samples[1].1,
                bin,
            )
        })
        .filter(|value| value.0 > 0.0)
        .max_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.total_cmp(&b.1))
                .then_with(|| b.2.cmp(&a.2))
        })
        .map(|value| value.2 as f64 + 0.5);
    if shared_peak.is_none() && shared_contour_support {
        // A common curved/short contour cannot supply the original 4px
        // fitting span. Do not replace it with a different dominant contour.
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let fit = |edges: &[Edge]| -> Option<(f64, f64)> {
        let selected = edges
            .iter()
            .filter(|p| angle_difference(p.angle, best.1) <= tolerance)
            .collect::<Vec<_>>();
        let mut bins = std::collections::BTreeMap::<i32, f64>::new();
        for p in &selected {
            *bins
                .entry((n.0 * p.px + n.1 * p.py).floor() as i32)
                .or_default() += p.weight;
        }
        let independent_peak = *bins
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0)))?
            .0 as f64
            + 0.5;
        let peak = shared_peak.unwrap_or(independent_peak);
        let points = selected
            .into_iter()
            .filter(|p| (n.0 * p.px + n.1 * p.py - peak).abs() <= 2.5)
            .collect::<Vec<_>>();
        let min = points.iter().map(|p| p.d).min_by(f64::total_cmp)?;
        let max = points.iter().map(|p| p.d).max_by(f64::total_cmp)?;
        if max - min < 4.0 {
            return None;
        }
        let sum = points.iter().map(|p| p.weight).sum::<f64>();
        let md = points.iter().map(|p| p.d * p.weight).sum::<f64>() / sum;
        let mr = points
            .iter()
            .map(|p| (n.0 * p.px + n.1 * p.py) * p.weight)
            .sum::<f64>()
            / sum;
        let variance = points
            .iter()
            .map(|p| p.weight * (p.d - md).powi(2))
            .sum::<f64>();
        if variance < 1e-8 {
            return None;
        }
        let slope = points
            .iter()
            .map(|p| p.weight * (p.d - md) * (n.0 * p.px + n.1 * p.py - mr))
            .sum::<f64>()
            / variance;
        Some((mr - slope * md, slope))
    };
    let left = fit(&sides[0]).ok_or(degradation::BOUNDARY_NO_PAIRABLE_EDGE)?;
    let right = fit(&sides[1]).ok_or(degradation::BOUNDARY_NO_PAIRABLE_EDGE)?;
    let crossing = n.0 * tangent.0 + n.1 * tangent.1;
    if crossing.abs() < 1e-6 {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let angle = (left.1.atan() - right.1.atan()).abs().to_degrees();
    if angle > 10.0 {
        return Err(degradation::BOUNDARY_ORIENTATION_MISMATCH);
    }
    Ok(BoundaryEdgeFits {
        left,
        right,
        crossing,
        angle,
    })
}

pub(crate) fn measure_boundary_strokes(
    image: &Rgb32FImage,
    boundary: &[(u32, u32)],
    world_origin: (f64, f64),
) -> Result<BoundaryStrokeReport, &'static str> {
    measure_boundary_strokes_with_coverage(image, None, boundary, world_origin)
}

pub(crate) fn measure_boundary_strokes_with_coverage(
    image: &Rgb32FImage,
    coverage: Option<&GrayImage>,
    boundary: &[(u32, u32)],
    world_origin: (f64, f64),
) -> Result<BoundaryStrokeReport, &'static str> {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 || boundary.len() < 2 {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let mut measurements = Vec::new();
    let mut unpairable_count = 0;
    let mut unmeasurable = Vec::new();
    for (index, &(x, y)) in boundary.iter().enumerate().step_by(256) {
        let prev = boundary[(index + boundary.len() - 1) % boundary.len()];
        let next = boundary[(index + 1) % boundary.len()];
        let tangent = (
            f64::from(next.0) - f64::from(prev.0),
            f64::from(next.1) - f64::from(prev.1),
        );
        let length = tangent.0.hypot(tangent.1);
        if length <= f64::EPSILON {
            unpairable_count += 1;
            continue;
        }
        let normal = (-tangent.1 / length, tangent.0 / length);
        let (normal_error, angle_error) =
            match paired_boundary_edge_error_with_coverage(image, coverage, x, y, normal) {
                Ok(value) => value,
                Err(reason) => {
                    unpairable_count += 1;
                    unmeasurable.push((
                        (world_origin.0 + x as f64, world_origin.1 + y as f64),
                        reason,
                    ));
                    continue;
                }
            };
        measurements.push(BoundaryStrokeMeasurement {
            world: (world_origin.0 + f64::from(x), world_origin.1 + f64::from(y)),
            normal_error_px: normal_error,
            orientation_error_deg: angle_error,
        });
    }
    if measurements.is_empty() {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let mut errors = measurements
        .iter()
        .map(|m| m.normal_error_px)
        .collect::<Vec<_>>();
    errors.sort_by(f64::total_cmp);
    let p95 = errors[((errors.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(errors.len() - 1)];
    Ok(BoundaryStrokeReport {
        pairable_count: measurements.len(),
        unpairable_count,
        unmeasurable,
        p95_error_px: p95,
        max_error_px: *errors.last().unwrap_or(&0.0),
        measurements,
    })
}
