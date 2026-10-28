//! Flat-artwork render order, tree-pose refinement and the blur filter.
//!
//! Orders flat-artwork sources as an overlap walk, closes the translation drift
//! of the panorama tree and drops clear sharpness outliers.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::MatchInfo;
use crate::panorama_utils::stack_pipeline::intra_station;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::HashMap;
use std::collections::HashSet;

use super::{
    Dsu, median_symmetric_error, median_value, solve_robust_closure_translations, transformed_point,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct FlatArtworkRenderRect {
    pub(super) left: f64,
    pub(super) top: f64,
    pub(super) right: f64,
    pub(super) bottom: f64,
}

impl FlatArtworkRenderRect {
    pub(super) fn width(self) -> f64 {
        (self.right - self.left).max(0.0)
    }

    pub(super) fn height(self) -> f64 {
        (self.bottom - self.top).max(0.0)
    }

    pub(super) fn area(self) -> f64 {
        self.width() * self.height()
    }

    pub(super) fn center(self) -> Point2<f64> {
        Point2::new(
            (self.left + self.right) * 0.5,
            (self.top + self.bottom) * 0.5,
        )
    }

    pub(super) fn overlap_area(self, other: Self) -> f64 {
        (self.right.min(other.right) - self.left.max(other.left)).max(0.0)
            * (self.bottom.min(other.bottom) - self.top.max(other.top)).max(0.0)
    }
}

pub(super) fn flat_artwork_projected_rect(
    image: &ImageInfo,
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<FlatArtworkRenderRect> {
    let transform = homographies.get(&image.id)?;
    let corners = [
        Point2::new(0.0, 0.0),
        Point2::new(image.width as f64, 0.0),
        Point2::new(image.width as f64, image.height as f64),
        Point2::new(0.0, image.height as f64),
    ];
    let points = corners
        .into_iter()
        .map(|point| transformed_point(transform, point))
        .collect::<Option<Vec<_>>>()?;
    let left = points
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let right = points
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let top = points
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min);
    let bottom = points
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let rect = FlatArtworkRenderRect {
        left,
        top,
        right,
        bottom,
    };
    (left.is_finite()
        && top.is_finite()
        && right.is_finite()
        && bottom.is_finite()
        && rect.area() > 1.0)
        .then_some(rect)
}

/// Order flat-artwork sources as a centre-outward overlap walk.
///
/// The graph/MST order is a geometry solve order, not a compositor order. A
/// breadth-first traversal of that tree can visit a branch whose projected
/// rectangle is disjoint from the already rendered canvas. The detail-
/// preserving compositor then has no existing pixels from which to estimate
/// that source's low-frequency exposure, so the complete source rectangle
/// survives as a visible block. Start near the projected canvas centre, finish
/// the current focus group, and then choose the unrendered rectangle with the
/// largest overlap with the materialised union. If the solved graph contains a
/// real gap, fall back to the nearest projected centre while retaining a
/// deterministic seed order.
pub(super) fn flat_artwork_overlap_render_order(
    images: &[ImageInfo],
    seed_order: &[usize],
    homographies: &HashMap<usize, Matrix3<f64>>,
    capture_group_ids: Option<&HashMap<usize, u8>>,
) -> Vec<usize> {
    if seed_order.len() < 2 {
        return seed_order.to_vec();
    }
    let mut rank = HashMap::with_capacity(seed_order.len());
    for (position, &index) in seed_order.iter().enumerate() {
        rank.insert(index, position);
    }
    let rects = seed_order
        .iter()
        .filter_map(|&index| {
            images
                .get(index)
                .and_then(|image| flat_artwork_projected_rect(image, homographies))
                .map(|rect| (index, rect))
        })
        .collect::<HashMap<_, _>>();
    if rects.len() < 2 {
        return seed_order.to_vec();
    }

    let mut center_x = rects
        .values()
        .map(|rect| rect.center().x)
        .collect::<Vec<_>>();
    let mut center_y = rects
        .values()
        .map(|rect| rect.center().y)
        .collect::<Vec<_>>();
    let median_x = median_value(&mut center_x).unwrap_or(0.0);
    let median_y = median_value(&mut center_y).unwrap_or(0.0);
    let initial = rects.keys().copied().min_by(|left, right| {
        let left_rect = rects[left];
        let right_rect = rects[right];
        let left_distance =
            (left_rect.center().x - median_x).hypot(left_rect.center().y - median_y);
        let right_distance =
            (right_rect.center().x - median_x).hypot(right_rect.center().y - median_y);
        left_distance
            .total_cmp(&right_distance)
            .then_with(|| rank[left].cmp(&rank[right]))
    });
    let Some(initial) = initial else {
        return seed_order.to_vec();
    };

    let group_for = |index: usize| {
        capture_group_ids
            .and_then(|groups| images.get(index).and_then(|image| groups.get(&image.id)))
            .copied()
    };
    let mut remaining = rects.keys().copied().collect::<HashSet<_>>();
    let mut ordered = Vec::with_capacity(seed_order.len());
    let mut rendered_rects = Vec::with_capacity(rects.len());
    let mut next = Some(initial);
    while let Some(index) = next {
        if !remaining.remove(&index) {
            break;
        }
        let rect = rects[&index];
        ordered.push(index);
        rendered_rects.push(rect);

        // Keep focus layers from one camera position contiguous. Their
        // rectangles are intentionally coincident, and this lets later layers
        // use the station anchor instead of being mistaken for new panorama
        // tiles.
        next = group_for(index).and_then(|group| {
            remaining
                .iter()
                .filter(|&&candidate| group_for(candidate) == Some(group))
                .min_by_key(|&&candidate| rank[&candidate])
                .copied()
        });
        if next.is_some() {
            continue;
        }
        let scene_diagonal = rendered_rects
            .iter()
            .flat_map(|rect| [rect.left, rect.right])
            .fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
            );
        let scene_vertical = rendered_rects
            .iter()
            .flat_map(|rect| [rect.top, rect.bottom])
            .fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
            );
        let scene_scale = (scene_diagonal.1 - scene_diagonal.0)
            .hypot(scene_vertical.1 - scene_vertical.0)
            .max(1.0);
        next = remaining.iter().copied().min_by(|left, right| {
            let score = |candidate: usize| {
                let candidate_rect = rects[&candidate];
                let candidate_area = candidate_rect.area().max(1.0);
                let maximum_overlap = rendered_rects
                    .iter()
                    .map(|rendered| candidate_rect.overlap_area(*rendered))
                    .fold(0.0, f64::max);
                let overlap_fraction = maximum_overlap / candidate_area;
                let candidate_center = candidate_rect.center();
                let nearest_distance = rendered_rects
                    .iter()
                    .map(|rendered| (candidate_center - rendered.center()).norm())
                    .fold(f64::INFINITY, f64::min)
                    / scene_scale;
                // A real overlap dominates proximity. The distance fallback
                // is only relevant when the graph has a genuine coverage gap.
                overlap_fraction * 100.0 - nearest_distance
            };
            score(*left)
                .total_cmp(&score(*right))
                .reverse()
                .then_with(|| rank[left].cmp(&rank[right]))
        });
    }

    // Keep malformed/unprojectable sources in their original solved order so
    // the caller still reports them instead of silently dropping coverage.
    let ordered_set = ordered.iter().copied().collect::<HashSet<_>>();
    ordered.extend(
        seed_order
            .iter()
            .copied()
            .filter(|index| !ordered_set.contains(index)),
    );
    ordered
}

/// Close the translation drift of the maximum-support panorama tree using all
/// measured flat-artwork overlaps. The MST is a stable initialization, but a
/// long raster can otherwise accumulate a small error at every step and make a
/// later source-sized exposure block look like a seam. Keep the measured
/// similarity scale/rotation untouched; solve only a bounded world translation
/// correction from the redundant pair relations.
pub(super) fn refine_flat_artwork_tree_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    base_poses: &HashMap<usize, Matrix3<f64>>,
) -> Option<HashMap<usize, Matrix3<f64>>> {
    if base_poses.len() < 3 || matches.len() <= base_poses.len().saturating_sub(1) {
        return None;
    }
    let mut constraints = Vec::<(usize, usize, [f64; 2], f64)>::new();
    let mut connectivity = Dsu::new(images.len());
    for (&(left, right), relation) in matches {
        let (Some(left_pose), Some(right_pose)) = (
            base_poses.get(&images[left].id),
            base_poses.get(&images[right].id),
        ) else {
            continue;
        };
        if left == right || relation.points.len() < 8 {
            continue;
        }
        let mut residuals = Vec::new();
        for &(left_point, right_point) in &relation.points {
            let Some(left_world) = transformed_point(left_pose, left_point) else {
                continue;
            };
            let Some(right_world) = transformed_point(right_pose, right_point) else {
                continue;
            };
            let delta = left_world - right_world;
            if delta.x.is_finite() && delta.y.is_finite() {
                residuals.push(delta);
            }
        }
        if residuals.len() < 8 {
            continue;
        }
        let mut dx = residuals.iter().map(|point| point.x).collect::<Vec<_>>();
        let mut dy = residuals.iter().map(|point| point.y).collect::<Vec<_>>();
        let (Some(dx), Some(dy)) = (median_value(&mut dx), median_value(&mut dy)) else {
            continue;
        };
        let median_error = median_symmetric_error(&relation.homography, &relation.points);
        let weight = (relation.points.len() as f64).sqrt() / (1.0 + median_error.max(0.0) / 4.0);
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        connectivity.union(left, right);
        constraints.push((left, right, [dx, dy], weight));
    }
    if constraints.len() <= base_poses.len().saturating_sub(1) {
        return None;
    }
    let reference = base_poses
        .keys()
        .copied()
        .min_by(|left, right| images[*left].filename.cmp(&images[*right].filename))?;
    let reference_root = connectivity.find(reference);
    if !base_poses
        .keys()
        .all(|index| connectivity.find(*index) == reference_root)
    {
        return None;
    }
    let solve = solve_robust_closure_translations(images.len(), reference, &constraints)?;
    if !solve.converged {
        return None;
    }
    let mut corrected = base_poses.clone();
    let mut max_correction: f64 = 0.0;
    for (&index, pose) in base_poses {
        let correction = solve.corrections.get(index).copied().unwrap_or([0.0, 0.0]);
        let magnitude = correction[0].hypot(correction[1]);
        // A correction larger than one source frame is evidence that the
        // redundant graph is mixing independent scenes; retain the measured
        // tree in that case instead of making a plausible-looking collage.
        if !magnitude.is_finite()
            || magnitude > images[index].width.max(images[index].height) as f64
        {
            return None;
        }
        max_correction = max_correction.max(magnitude);
        let translation = Matrix3::new(
            1.0,
            0.0,
            correction[0],
            0.0,
            1.0,
            correction[1],
            0.0,
            0.0,
            1.0,
        );
        corrected.insert(index, translation * pose);
    }
    let mut solved_residuals = constraints
        .iter()
        .map(|&(left, right, observed, _)| {
            (solve.corrections[right][0] - solve.corrections[left][0] - observed[0])
                .hypot(solve.corrections[right][1] - solve.corrections[left][1] - observed[1])
        })
        .collect::<Vec<_>>();
    let median_residual = median_value(&mut solved_residuals).unwrap_or(f64::INFINITY);
    println!(
        "  - Flat-artwork pose closure accepted: {} redundant relation(s), median residual {:.2}px, maximum translation correction {:.1}px",
        constraints.len(),
        median_residual,
        max_correction,
    );
    Some(corrected)
}

/// Median Sharpness_Score over the valid pixels of one Source_RAW (需求 2.1).
///
/// Measured on the already prepared analysis image, resampled to the
/// Intra_Station analysis budget of 需求 2.2 so the number is a property of the
/// photograph rather than of whichever alignment budget the run happened to
/// pick.  The score is still the unbounded acutance ratio; task 7.6 normalises
/// it monotonically, which cannot change which frame is sharpest.
pub(super) fn focus_frame_median_sharpness(image: &ImageInfo) -> f64 {
    let analysis = &image.alignment_image;
    let (width, height) = (analysis.width(), analysis.height());
    let spacing = intra_station::analysis_probe_spacing(width.max(height));
    intra_station::median_valid_sharpness(width, height, |x, y| {
        crate::panorama_utils::mosaic::gray_analysis_sharpness(analysis, x, y, spacing)
    })
}

/// Drop only clear outliers in sharpness for a flat-artwork scan. A low score
/// alone is not enough to reject a mostly blank paper tile, so require the
/// frame to be both in the bottom fifth of the selection and less than 55% of
/// the robust median. This keeps low-texture but usable edge coverage while
/// removing visibly defocused frames that would soften an otherwise sharp
/// mosaic.
pub(super) fn filter_flat_artwork_blurry_sources(images: &mut Vec<ImageInfo>) {
    if images.len() < 4 {
        return;
    }
    let mut scores = images
        .iter()
        .map(|image| (focus_frame_median_sharpness(image), image.filename.clone()))
        .collect::<Vec<_>>();
    scores.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let median = scores[scores.len() / 2].0;
    if !median.is_finite() || median <= 0.0 {
        return;
    }
    let bottom_fifth_index = ((scores.len() - 1) * 20) / 100;
    let bottom_fifth = scores[bottom_fifth_index].0;
    let threshold = median * 0.55;
    let rejected = scores
        .iter()
        .filter(|(score, _)| *score <= bottom_fifth && *score < threshold)
        .map(|(_, filename)| filename.clone())
        .collect::<HashSet<_>>();
    if rejected.is_empty() || rejected.len() >= images.len().saturating_sub(2) {
        return;
    }
    for image in images
        .iter()
        .filter(|image| rejected.contains(&image.filename))
    {
        let score = focus_frame_median_sharpness(image);
        println!(
            "  - Flat-artwork blur filter: excluding '{}' (median sharpness {:.4}, selection median {:.4})",
            image.filename, score, median
        );
    }
    images.retain(|image| !rejected.contains(&image.filename));
    println!(
        "  - Flat-artwork blur filter retained {} of {} source images",
        images.len(),
        images.len() + rejected.len()
    );
}
