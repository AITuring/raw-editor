//! Final-world sampling for station seams.
//!
//! Relation lookups, coverage-boundary distances, colour/tone samples, edge
//! misalignment and world gradients measured on both stations of a seam pair;
//! the seam planner turns these samples into costs.

use super::super::coordinates::Projection;
use super::super::coordinates::project_point;
use super::super::coordinates::unproject_point;
use super::super::stack_pipeline::residual_warp;
use super::super::stack_pipeline::tone;
use crate::panorama_utils::shared::ImageInfo;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;

use super::{
    SeamOrientation, focus_stack_pixel_is_canvas_like, focus_stack_pixel_is_tone_foreground,
    get_high_quality_interpolated_pixel, map_target_to_source_with_residual,
    source_is_focus_foreground, source_sample_has_support,
};

/// A seam solved on an accepted Virtual_Tile overlap.  The path is sampled in
/// output-world pixels, with one cross-axis coordinate per along-axis sample.
/// Keeping the plan separate from the pixel rewrite lets the compositor solve
/// every relation deterministically before it reloads only the two source
/// tiles needed to apply the result.
#[derive(Debug, Clone)]
pub(super) struct StationSeamPlan {
    pub(super) first_owner: u16,
    pub(super) second_owner: u16,
    pub(super) min_x: u32,
    pub(super) max_x: u32,
    pub(super) min_y: u32,
    pub(super) max_y: u32,
    pub(super) along_min: u32,
    pub(super) along_max: u32,
    pub(super) orientation: SeamOrientation,
    pub(super) cross_samples: Vec<f64>,
    pub(super) current_cross_samples: Vec<f64>,
    pub(super) first_on_low_side: bool,
    /// Optional locally fitted relation for a seam-only pair.  The relation
    /// maps the first tile's projected coordinates into the second tile's
    /// projected coordinates.  It is deliberately absent for authoritative
    /// pose-graph edges, which continue to use the solved global poses.
    pub(super) local_relation: Option<Matrix3<f64>>,
    /// Bounded low-frequency correction measured from the same seam-only
    /// correspondence.  This is applied only to the locally re-rendered
    /// second-owner band, so the candidate cannot introduce a tile-wide tone
    /// step or alter the owner ledger.
    pub(super) local_tone: Option<tone::ToneSolve>,
}

pub(super) fn seam_candidate_relation_for_pair(
    first_id: usize,
    second_id: usize,
) -> Option<Matrix3<f64>> {
    let topology = tone::run_topology_snapshot()?;
    let requested_pair = station_local_relation_pair_requested(&topology, first_id, second_id);
    let accepted_pair = topology.accepted.iter().any(|&(left, right)| {
        (left == first_id && right == second_id) || (left == second_id && right == first_id)
    });
    // Authoritative pose-graph edges stay on the solved global relation by
    // default.  They may opt into this seam-only local relation only when the
    // caller names the pair explicitly; this keeps ordinary runs unchanged
    // while allowing a measured correction for a known visual defect.
    if accepted_pair && !requested_pair {
        return None;
    }
    let (bits, reverse) = topology
        .seam_candidate_transforms
        .iter()
        .find(|(left, right, _)| *left == first_id && *right == second_id)
        .map(|(_, _, bits)| (bits, false))
        .or_else(|| {
            topology
                .seam_candidate_transforms
                .iter()
                .find(|(left, right, _)| *left == second_id && *right == first_id)
                .map(|(_, _, bits)| (bits, true))
        })?;
    let values = bits.map(f64::from_bits);
    let relation = Matrix3::from_row_slice(&values);
    if !relation.iter().all(|value| value.is_finite()) {
        return None;
    }
    if reverse {
        relation.try_inverse()
    } else {
        Some(relation)
    }
}

/// Return whether a pair was explicitly selected for local seam refinement.
/// Values use the one-based owner/station labels exposed in native reports;
/// internally they are compared against the zero-based station indices.
pub(super) fn station_local_relation_pair_requested(
    topology: &tone::ToneStationTopology,
    first_id: usize,
    second_id: usize,
) -> bool {
    let Some(first_station) = topology.station_of(first_id) else {
        return false;
    };
    let Some(second_station) = topology.station_of(second_id) else {
        return false;
    };
    let Some(requested_pairs) = std::env::var("RAW_EDITOR_STATION_LOCAL_RELATION_RESAMPLE_PAIR")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| {
                    let (left, right) = part.trim().split_once('-')?;
                    let left = left.trim().parse::<usize>().ok()?.checked_sub(1)?;
                    let right = right.trim().parse::<usize>().ok()?.checked_sub(1)?;
                    Some((left, right))
                })
                .collect::<Vec<_>>()
        })
    else {
        return false;
    };
    requested_pairs.iter().any(|&(left, right)| {
        (first_station == left && second_station == right)
            || (first_station == right && second_station == left)
    })
}

pub(super) fn station_local_relation_pair_matches_owner(
    first_owner: u16,
    second_owner: u16,
) -> bool {
    let Some(requested_pairs) = std::env::var("RAW_EDITOR_STATION_LOCAL_RELATION_RESAMPLE_PAIR")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| {
                    let (left, right) = part.trim().split_once('-')?;
                    let left = left.trim().parse::<usize>().ok()?.checked_sub(1)?;
                    let right = right.trim().parse::<usize>().ok()?.checked_sub(1)?;
                    Some((left, right))
                })
                .collect::<Vec<_>>()
        })
    else {
        return true;
    };
    let first = usize::from(first_owner.saturating_sub(1));
    let second = usize::from(second_owner.saturating_sub(1));
    requested_pairs.iter().any(|&(left, right)| {
        (first == left && second == right) || (first == right && second == left)
    })
}

/// Width of the region on the locally fitted second owner that is re-rendered
/// through a seam-only relation.  The default stays narrow so an unverified
/// candidate cannot rewrite a whole tile; native acceptance can widen this
/// bounded region when the fitted relation has to carry a visible object
/// across the seam.
pub(super) fn station_local_relation_resample_band_px(first_owner: u16, second_owner: u16) -> f64 {
    if !station_local_relation_pair_matches_owner(first_owner, second_owner) {
        return 0.0;
    }
    std::env::var("RAW_EDITOR_STATION_LOCAL_RELATION_RESAMPLE_BAND_PX")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(64.0)
}

pub(super) fn station_seam_edge_strength_weight() -> f64 {
    std::env::var("RAW_EDITOR_STATION_SEAM_EDGE_STRENGTH_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(2.0)
}

/// Return a bounded cost for cutting through a focus foreground sample on a
/// seam-only closure repair.  The foreground mask is already the source-level
/// evidence used by focus ownership; reusing it here keeps the seam from
/// splitting a shoe, figure, or other near-field contour into two independently
/// sampled owners.  Authoritative pose-graph seams keep their existing
/// objective, while a local relation is only used for the explicitly measured
/// closure defects that need this protection.
pub(super) fn station_seam_foreground_penalty(
    local_relation: Option<&Matrix3<f64>>,
    left: &ImageInfo,
    left_source: Point2<f64>,
    left_pixel: &Rgb<f32>,
    right: &ImageInfo,
    right_source: Point2<f64>,
    right_pixel: &Rgb<f32>,
    edge_strength: f64,
) -> f64 {
    if local_relation.is_none() {
        return 0.0;
    }
    const FOREGROUND_CUT_PENALTY: f64 = 256.0;
    let source_mask_foreground = source_is_focus_foreground(left, left_source)
        || source_is_focus_foreground(right, right_source)
        || focus_stack_pixel_is_tone_foreground(left_pixel.0.as_slice())
        || focus_stack_pixel_is_tone_foreground(right_pixel.0.as_slice());
    // The source foreground mask is deliberately conservative and only
    // covers the detected near-field range.  The shoe in the failing native
    // crop is below that range, so also use the actual seam samples.  A
    // near-black sample or a dark/high-contrast transition is painted
    // foreground evidence, even when the source mask has no label there.
    let left_luma = station_seam_luma(*left_pixel);
    let right_luma = station_seam_luma(*right_pixel);
    let minimum_luma = left_luma.min(right_luma);
    let luma_delta = (left_luma - right_luma).abs();
    let sampled_dark_foreground = minimum_luma <= 0.18
        || (minimum_luma <= 0.30 && (luma_delta >= 0.08 || edge_strength >= 0.04));
    if source_mask_foreground || sampled_dark_foreground {
        FOREGROUND_CUT_PENALTY
    } else {
        0.0
    }
}

pub(super) fn source_to_world_point(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    source_x: f64,
    source_y: f64,
) -> Option<Point2<f64>> {
    let projected = project_point(image, source_x, source_y, projection)?;
    let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
    if mapped.z.abs() < 1e-8 {
        return None;
    }
    let point = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
    if point.x.is_finite() && point.y.is_finite() {
        Some(point)
    } else {
        None
    }
}

/// Approximate the distance from a source sample to its projected valid
/// coverage boundary in output-world pixels. The local Jacobian converts the
/// source-edge distance into world pixels, which is more faithful than using
/// a source-pixel threshold after a perspective transform.
pub(super) fn projected_coverage_boundary_distance(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    source: Point2<f64>,
) -> Option<f64> {
    let width = image.width().max(1) as f64;
    let height = image.height().max(1) as f64;
    let source_edge = source
        .x
        .min(source.y)
        .min((width - 1.0) - source.x)
        .min((height - 1.0) - source.y)
        .max(0.0);
    let center = source_to_world_point(image, homography, projection, source.x, source.y)?;
    let x_step = if source.x + 1.0 < width { 1.0 } else { -1.0 };
    let y_step = if source.y + 1.0 < height { 1.0 } else { -1.0 };
    let x_neighbour = source_to_world_point(
        image,
        homography,
        projection,
        (source.x + x_step).clamp(0.0, width - 1.0),
        source.y,
    )?;
    let y_neighbour = source_to_world_point(
        image,
        homography,
        projection,
        source.x,
        (source.y + y_step).clamp(0.0, height - 1.0),
    )?;
    let local_scale = (x_neighbour - center)
        .norm()
        .min((y_neighbour - center).norm())
        .max(1e-6);
    Some(source_edge * local_scale)
}

pub(super) fn station_seam_sample(
    image: &ImageInfo,
    image_pixels: &Rgb32FImage,
    inverse: &Matrix3<f64>,
    projection: Projection,
    residual: &residual_warp::ResidualWarp,
    x: u32,
    y: u32,
    offset_x: f64,
    offset_y: f64,
) -> Option<(Rgb<f32>, Point2<f64>)> {
    let source = map_target_to_source_with_residual(
        inverse,
        Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0),
        image,
        projection,
        residual,
    )?;
    if source.x < 0.0
        || source.y < 0.0
        || source.x >= image_pixels.width() as f64
        || source.y >= image_pixels.height() as f64
        || !source_sample_has_support(image_pixels, source.x, source.y)
    {
        return None;
    }
    let pixel = get_high_quality_interpolated_pixel(image_pixels, source.x, source.y);
    (pixel[0].max(pixel[1]).max(pixel[2]) > 1e-6).then_some((pixel, source))
}

/// Sample a seam pair with an optional locally fitted relation.  The first
/// tile remains on the authoritative global pose; when a seam-only relation
/// is present, the second tile is read by projecting that same first-tile
/// sample through the local source-to-target fit.  This repairs the visible
/// relative geometry at a rejected closure edge without changing the global
/// pose graph or any accepted relation.
#[allow(clippy::too_many_arguments)]
pub(super) fn station_seam_pair_sample(
    first: &ImageInfo,
    first_pixels: &Rgb32FImage,
    first_inverse: &Matrix3<f64>,
    second: &ImageInfo,
    second_pixels: &Rgb32FImage,
    second_inverse: &Matrix3<f64>,
    relation: Option<&Matrix3<f64>>,
    projection: Projection,
    residual: &residual_warp::ResidualWarp,
    x: u32,
    y: u32,
    offset_x: f64,
    offset_y: f64,
) -> Option<((Rgb<f32>, Point2<f64>), (Rgb<f32>, Point2<f64>))> {
    let first_sample = station_seam_sample(
        first,
        first_pixels,
        first_inverse,
        projection,
        residual,
        x,
        y,
        offset_x,
        offset_y,
    )?;
    let second_sample = if let Some(relation) = relation {
        let target = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
        let first_source =
            map_target_to_source_with_residual(first_inverse, target, first, projection, residual)?;
        let first_projected = project_point(first, first_source.x, first_source.y, projection)?;
        let mapped = relation * Point3::new(first_projected.x, first_projected.y, 1.0);
        if !mapped.x.is_finite() || !mapped.y.is_finite() || mapped.z.abs() < 1e-8 {
            None
        } else {
            let projected_target = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
            let source =
                unproject_point(second, projected_target.x, projected_target.y, projection)?;
            if source.x < 0.0
                || source.y < 0.0
                || source.x >= second_pixels.width() as f64
                || source.y >= second_pixels.height() as f64
                || !source_sample_has_support(second_pixels, source.x, source.y)
            {
                None
            } else {
                let pixel = get_high_quality_interpolated_pixel(second_pixels, source.x, source.y);
                (pixel[0].max(pixel[1]).max(pixel[2]) > 1e-6).then_some((pixel, source))
            }
        }
    } else {
        station_seam_sample(
            second,
            second_pixels,
            second_inverse,
            projection,
            residual,
            x,
            y,
            offset_x,
            offset_y,
        )
    }?;
    Some((first_sample, second_sample))
}

/// Estimate a bounded local tone correction from a seam-only relation.  The
/// pair is sampled through the fitted geometry and only warm, canvas-like
/// samples are retained; painted strokes therefore cannot steer this small
/// boundary repair.  The normal Tone_Harmonizer remains restricted to
/// accepted Station_Relations.
#[allow(clippy::too_many_arguments)]
pub(super) fn station_seam_local_tone(
    first: &ImageInfo,
    first_pixels: &Rgb32FImage,
    first_inverse: &Matrix3<f64>,
    second: &ImageInfo,
    second_pixels: &Rgb32FImage,
    second_inverse: &Matrix3<f64>,
    relation: &Matrix3<f64>,
    projection: Projection,
    residual: &residual_warp::ResidualWarp,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
    offset_x: f64,
    offset_y: f64,
) -> Option<tone::ToneSolve> {
    let area = u64::from(max_x.saturating_sub(min_x).saturating_add(1))
        * u64::from(max_y.saturating_sub(min_y).saturating_add(1));
    let step = ((area as f64 / 4096.0).sqrt().ceil() as u32).max(8);
    let mut samples = Vec::new();
    let mut y = min_y;
    while y <= max_y {
        let mut x = min_x;
        while x <= max_x {
            let Some(((first_pixel, _), (second_pixel, _))) = station_seam_pair_sample(
                first,
                first_pixels,
                first_inverse,
                second,
                second_pixels,
                second_inverse,
                Some(relation),
                projection,
                residual,
                x,
                y,
                offset_x,
                offset_y,
            ) else {
                x = x.saturating_add(step);
                continue;
            };
            if focus_stack_pixel_is_canvas_like(&first_pixel.0)
                && focus_stack_pixel_is_canvas_like(&second_pixel.0)
                && !focus_stack_pixel_is_tone_foreground(&first_pixel.0)
                && !focus_stack_pixel_is_tone_foreground(&second_pixel.0)
            {
                samples.push(tone::ToneSample {
                    owner: first_pixel.0,
                    source: second_pixel.0,
                });
            }
            x = x.saturating_add(step);
        }
        y = y.saturating_add(step);
    }
    (samples.len() >= tone::TONE_MIN_SAMPLES).then(|| tone::solve_tone_pair(&samples))
}

pub(super) fn station_seam_edge_misalignment(left: (f64, f64), right: (f64, f64)) -> f64 {
    const EDGE_EVIDENCE_FLOOR: f64 = 0.005;
    let left_magnitude = left.0.hypot(left.1);
    let right_magnitude = right.0.hypot(right.1);
    if left_magnitude < EDGE_EVIDENCE_FLOOR && right_magnitude < EDGE_EVIDENCE_FLOOR {
        return 0.0;
    }
    if left_magnitude < EDGE_EVIDENCE_FLOOR || right_magnitude < EDGE_EVIDENCE_FLOOR {
        return 1.0;
    }
    let orientation = 1.0
        - ((left.0 * right.0 + left.1 * right.1).abs() / (left_magnitude * right_magnitude))
            .clamp(0.0, 1.0);
    let magnitude = (left_magnitude - right_magnitude).abs()
        / left_magnitude.max(right_magnitude).max(EDGE_EVIDENCE_FLOOR);
    (orientation + magnitude).clamp(0.0, 2.0)
}

pub(super) fn station_seam_luma(pixel: Rgb<f32>) -> f64 {
    0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
}

#[allow(clippy::too_many_arguments)]
pub(super) fn station_seam_world_gradient(
    image: &ImageInfo,
    pixels: &Rgb32FImage,
    inverse: &Matrix3<f64>,
    projection: Projection,
    residual: &residual_warp::ResidualWarp,
    x: u32,
    y: u32,
    offset_x: f64,
    offset_y: f64,
) -> Option<(f64, f64)> {
    let left = station_seam_sample(
        image,
        pixels,
        inverse,
        projection,
        residual,
        x.checked_sub(1)?,
        y,
        offset_x,
        offset_y,
    )?;
    let right = station_seam_sample(
        image,
        pixels,
        inverse,
        projection,
        residual,
        x.checked_add(1)?,
        y,
        offset_x,
        offset_y,
    )?;
    let top = station_seam_sample(
        image,
        pixels,
        inverse,
        projection,
        residual,
        x,
        y.checked_sub(1)?,
        offset_x,
        offset_y,
    )?;
    let bottom = station_seam_sample(
        image,
        pixels,
        inverse,
        projection,
        residual,
        x,
        y.checked_add(1)?,
        offset_x,
        offset_y,
    )?;
    Some((
        (station_seam_luma(right.0) - station_seam_luma(left.0)) * 0.5,
        (station_seam_luma(bottom.0) - station_seam_luma(top.0)) * 0.5,
    ))
}
