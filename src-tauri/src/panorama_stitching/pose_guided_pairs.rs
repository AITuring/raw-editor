//! Pairs the solved poses say overlap but the candidate search never compared.
//!
//! A large focus stack compares only filename and visual-retrieval neighbours
//! (`pairs_to_match_for_images`), so frames that overlap across two scan rows
//! or two captures are often never matched, and the solve places them through
//! long chains. In 枫鹰雉鸡 (110 photos) the pairs the matcher compared agree
//! with the solved poses to 2 px (median), while 646 overlapping pairs were
//! never compared and 54 of them sit more than 60 px apart: the left column of
//! the bottom capture landed 90–110 px off the body frames above it, past what
//! the compositor's local registration corrects, and the grass at their seam
//! broke. The pairs found here go through the same verified matcher, and the
//! poses are solved again with every pair that verifies.

use nalgebra::Matrix3;
use std::collections::{HashMap, HashSet};

use super::ImageInfo;

/// Two frames are compared when their solved footprints share at least this
/// share of the smaller footprint.
pub(super) const MIN_PREDICTED_OVERLAP: f64 = 0.02;

/// Frame corners on the canvas under `pose`, or `None` behind the camera.
fn footprint(image: &ImageInfo, pose: &Matrix3<f64>) -> Option<Vec<(f64, f64)>> {
    let (width, height) = (f64::from(image.width), f64::from(image.height));
    [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
        .into_iter()
        .map(|(x, y)| {
            let point = pose * nalgebra::Vector3::new(x, y, 1.0);
            (point.z > 1e-9).then(|| (point.x / point.z, point.y / point.z))
        })
        .collect()
}

/// Signed area of a polygon (positive counter-clockwise in y-up terms).
fn signed_area(polygon: &[(f64, f64)]) -> f64 {
    polygon
        .iter()
        .zip(polygon.iter().cycle().skip(1))
        .map(|(a, b)| a.0 * b.1 - b.0 * a.1)
        .sum::<f64>()
        * 0.5
}

/// Area shared by two convex polygons (Sutherland–Hodgman clipping).
fn intersection_area(subject: &[(f64, f64)], clip: &[(f64, f64)]) -> f64 {
    let orientation = signed_area(clip).signum();
    let inside = |a: (f64, f64), b: (f64, f64), p: (f64, f64)| {
        ((b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)) * orientation >= 0.0
    };
    let crossing = |a: (f64, f64), b: (f64, f64), p: (f64, f64), q: (f64, f64)| {
        let (dx1, dy1) = (b.0 - a.0, b.1 - a.1);
        let (dx2, dy2) = (q.0 - p.0, q.1 - p.1);
        let denominator = dx1 * dy2 - dy1 * dx2;
        if denominator.abs() < 1e-12 {
            return p;
        }
        let t = ((p.0 - a.0) * dy2 - (p.1 - a.1) * dx2) / denominator;
        (a.0 + t * dx1, a.1 + t * dy1)
    };
    let mut output = subject.to_vec();
    for (index, &a) in clip.iter().enumerate() {
        let b = clip[(index + 1) % clip.len()];
        let input = std::mem::take(&mut output);
        for (position, &q) in input.iter().enumerate() {
            let p = input[(position + input.len() - 1) % input.len()];
            match (inside(a, b, p), inside(a, b, q)) {
                (true, true) => output.push(q),
                (true, false) => output.push(crossing(a, b, p, q)),
                (false, true) => {
                    output.push(crossing(a, b, p, q));
                    output.push(q);
                }
                (false, false) => {}
            }
        }
        if output.is_empty() {
            return 0.0;
        }
    }
    signed_area(&output).abs()
}

/// Index pairs `(i, j)`, `i < j`, of frames whose solved footprints overlap by
/// at least `MIN_PREDICTED_OVERLAP` of the smaller one and that are not in
/// `attempted`, in index order.
pub(super) fn pose_predicted_pairs(
    images: &[ImageInfo],
    poses: &HashMap<usize, Matrix3<f64>>,
    attempted: &HashSet<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let footprints = images
        .iter()
        .map(|image| {
            let polygon = footprint(image, poses.get(&image.id)?)?;
            let area = signed_area(&polygon).abs();
            (area > 0.0).then_some((polygon, area))
        })
        .collect::<Vec<_>>();
    let bounds = footprints
        .iter()
        .map(|footprint| {
            footprint.as_ref().map(|(polygon, _)| {
                polygon.iter().fold(
                    (
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::NEG_INFINITY,
                    ),
                    |(x0, y0, x1, y1), &(x, y)| (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                )
            })
        })
        .collect::<Vec<_>>();
    let mut pairs = Vec::new();
    for left in 0..images.len() {
        let (Some((left_polygon, left_area)), Some(left_box)) = (&footprints[left], bounds[left])
        else {
            continue;
        };
        for right in left + 1..images.len() {
            if attempted.contains(&(left, right)) {
                continue;
            }
            let (Some((right_polygon, right_area)), Some(right_box)) =
                (&footprints[right], bounds[right])
            else {
                continue;
            };
            if left_box.2 <= right_box.0
                || right_box.2 <= left_box.0
                || left_box.3 <= right_box.1
                || right_box.3 <= left_box.1
            {
                continue;
            }
            let shared = intersection_area(left_polygon, right_polygon);
            if shared >= MIN_PREDICTED_OVERLAP * left_area.min(*right_area) {
                pairs.push((left, right));
            }
        }
    }
    pairs
}

/// Grid step, patch half-size and search reach of the guided patch
/// measurement, in alignment pixels (about 5 source pixels each for a 60 MP
/// frame): the reach covers the 140 px the solve was off in 枫鹰雉鸡.
const GUIDED_STEP: u32 = 14;
const GUIDED_PATCH_RADIUS: i32 = 6;
const GUIDED_SEARCH: i32 = 32;
/// Correlation a guided patch needs, and how far its backward match may land
/// from where it started.
const GUIDED_MIN_CORRELATION: f64 = 0.9;
const GUIDED_MAX_ROUND_TRIP: f64 = 1.0;
/// Patches whose offset lies this close to the pair's median offset agree.
const GUIDED_AGREEMENT: f64 = 2.0;
/// A guided relation needs this many agreeing patches, spread over at least
/// this share of the source frame's shorter side.
const GUIDED_MIN_POINTS: usize = 12;
const GUIDED_MIN_SPREAD: f64 = 0.25;

/// A relation between two frames the matcher did not verify, measured by
/// patches where the solved poses put their content (`source_pose`,
/// `target_pose`: source pixels to canvas). Many frames of a scan overlap only
/// in a thin strip, where too few descriptors match for the matcher; a patch
/// predicted from the solved poses only has to be found within
/// `GUIDED_SEARCH` of where they put it. The relation is the poses' own,
/// corrected by the patches' median offset, and its points are the agreeing
/// patches in source pixels.
pub(super) fn patch_guided_match(
    source: &ImageInfo,
    target: &ImageInfo,
    source_pose: &Matrix3<f64>,
    target_pose: &Matrix3<f64>,
) -> Option<super::MatchInfo> {
    use crate::panorama_utils::registration::refine_warped_patch;
    use nalgebra::Point2;

    let predicted = target_pose.try_inverse()? * source_pose;
    let to_alignment = |scale: f64| Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
    let source_scale = to_alignment(source.scale_factor);
    let target_scale_inverse = to_alignment(target.scale_factor.recip());
    let forward = target_scale_inverse * predicted * source_scale;
    let backward = forward.try_inverse()?;
    let (source_image, target_image) = (&source.alignment_image, &target.alignment_image);
    let margin = f64::from(GUIDED_PATCH_RADIUS + GUIDED_SEARCH + 2);
    let inside = |point: Point2<f64>, image: &image::GrayImage| {
        point.x >= margin
            && point.y >= margin
            && point.x < f64::from(image.width()) - margin
            && point.y < f64::from(image.height()) - margin
    };
    let project = |matrix: &Matrix3<f64>, point: Point2<f64>| {
        let mapped = matrix * nalgebra::Point3::new(point.x, point.y, 1.0);
        (mapped.z.abs() > 1e-9).then(|| Point2::new(mapped.x / mapped.z, mapped.y / mapped.z))
    };
    let mut found = Vec::<(Point2<f64>, Point2<f64>, nalgebra::Vector2<f64>)>::new();
    for y in (0..source_image.height()).step_by(GUIDED_STEP as usize) {
        for x in (0..source_image.width()).step_by(GUIDED_STEP as usize) {
            let point = Point2::new(f64::from(x), f64::from(y));
            if !inside(point, source_image) {
                continue;
            }
            let Some(expected) = project(&forward, point) else {
                continue;
            };
            if !inside(expected, target_image) {
                continue;
            }
            let Some(matched) = refine_warped_patch(
                source_image,
                target_image,
                &forward,
                point,
                GUIDED_PATCH_RADIUS,
                GUIDED_SEARCH,
            ) else {
                continue;
            };
            if matched.correlation < GUIDED_MIN_CORRELATION {
                continue;
            }
            // Measured back from where it landed, the patch has to return to
            // where it started: a repeated stroke would not.
            let offset = matched.target - expected;
            let corrected_backward =
                backward * Matrix3::new(1.0, 0.0, -offset.x, 0.0, 1.0, -offset.y, 0.0, 0.0, 1.0);
            let Some(back) = refine_warped_patch(
                target_image,
                source_image,
                &corrected_backward,
                matched.target,
                GUIDED_PATCH_RADIUS,
                3,
            ) else {
                continue;
            };
            if (back.target - point).norm() > GUIDED_MAX_ROUND_TRIP {
                continue;
            }
            found.push((point, matched.target, offset));
        }
    }
    if found.len() < GUIDED_MIN_POINTS {
        return None;
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let median_offset = nalgebra::Vector2::new(
        median(&mut found.iter().map(|(_, _, offset)| offset.x).collect()),
        median(&mut found.iter().map(|(_, _, offset)| offset.y).collect()),
    );
    let agreeing = found
        .into_iter()
        .filter(|(_, _, offset)| (offset - median_offset).norm() <= GUIDED_AGREEMENT)
        .collect::<Vec<_>>();
    if agreeing.len() < GUIDED_MIN_POINTS {
        return None;
    }
    let (min_x, max_x, min_y, max_y) = agreeing.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(x0, x1, y0, y1), (point, _, _)| {
            (
                x0.min(point.x),
                x1.max(point.x),
                y0.min(point.y),
                y1.max(point.y),
            )
        },
    );
    let shorter_side = f64::from(source_image.width().min(source_image.height()));
    if (max_x - min_x).max(max_y - min_y) < GUIDED_MIN_SPREAD * shorter_side {
        return None;
    }
    let shift = median_offset * target.scale_factor;
    let homography = Matrix3::new(1.0, 0.0, shift.x, 0.0, 1.0, shift.y, 0.0, 0.0, 1.0) * predicted;
    let points = agreeing
        .iter()
        .map(|(point, matched, _)| {
            (
                Point2::new(point.x * source.scale_factor, point.y * source.scale_factor),
                Point2::new(
                    matched.x * target.scale_factor,
                    matched.y * target.scale_factor,
                ),
            )
        })
        .collect::<Vec<_>>();
    Some(super::build_match_info(
        homography,
        points.clone(),
        points,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_squares_share_their_common_area() {
        let a = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let b = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)];
        assert!((intersection_area(&a, &b) - 25.0).abs() < 1e-9);
        let clockwise = [(5.0, 5.0), (5.0, 15.0), (15.0, 15.0), (15.0, 5.0)];
        assert!((intersection_area(&a, &clockwise) - 25.0).abs() < 1e-9);
        let apart = [(20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)];
        assert_eq!(intersection_area(&a, &apart), 0.0);
    }
}
