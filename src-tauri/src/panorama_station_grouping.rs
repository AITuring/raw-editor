//! Capture_Station grouping from verified pairwise evidence.
//!
//! A pair joins one station only when its measured evidence (inliers, overlap
//! NCC, spatial support and scale) passes the station gate and its centre motion
//! stays within a local focus bracket; components that drift too far are split by
//! motion. Every split and rejection is recorded in the grouping outcome.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::MatchInfo;
use crate::panorama_utils::stack_pipeline::degradation;
use crate::panorama_utils::stack_pipeline::determinism::sorted_pairs;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use super::{
    Dsu, FOCUS_BRACKET_MAX_CENTER_MOTION_RATIO, FOCUS_BRACKET_MIN_OVERLAP_SUPPORT,
    FOCUS_MODEL_MIN_INLIERS, SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION,
    STATION_EVIDENCE_SCORE_TIE_BREAK, STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO,
    STATION_MIN_INLIER_MATCHES, STATION_MIN_INLIER_SPATIAL_SUPPORT, STATION_MIN_OVERLAP_NCC,
    STATION_SCALE_RATIO_MAX, STATION_SCALE_RATIO_MIN, canonical_match_direction,
    focus_compatible_focal_length, focus_image_direction_key, focus_overlap_is_verified,
    focus_overlap_quality, homography_scale_ratio, match_has_spatial_support,
    median_symmetric_error, natural_path_cmp, panorama_spatial_support,
    panorama_transform_overlap_support, shifted_focus_structure_is_verified,
    station_member_count_is_supported, transformed_point,
};

/// Frame-centre displacement of an explicit transform, in long sides.
pub(super) fn focus_transform_center_motion_ratio(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<f64> {
    let source_center = Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5);
    let target_center = Point2::new(target.width as f64 * 0.5, target.height as f64 * 0.5);
    let mapped_center = transformed_point(source_to_target, source_center)?;
    let scale = source
        .width
        .max(source.height)
        .max(target.width.max(target.height))
        .max(1) as f64;
    let displacement = (mapped_center - target_center).norm();
    (displacement.is_finite()).then_some(displacement / scale)
}

pub(super) fn focus_match_center_motion_ratio(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> Option<f64> {
    focus_transform_center_motion_ratio(source, target, &match_info.homography)
}

pub(super) fn focus_match_is_local_bracket(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> bool {
    if match_info.sequence_bridge {
        return false;
    }
    let compatible_focal_length = source
        .focal_length_35mm
        .zip(target.focal_length_35mm)
        .map(|(left, right)| {
            let ratio = (left / right.max(1e-8)).max(right / left.max(1e-8));
            ratio.is_finite() && ratio <= 1.10
        })
        .unwrap_or(true);
    if match_info.coarse_bridge {
        let scale = match_info.homography[(0, 0)].abs();
        return compatible_focal_length
            && match_info.points.len() >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION
            && (0.88..=1.12).contains(&scale)
            && focus_match_center_motion_ratio(source, target, match_info)
                .is_some_and(|motion| motion <= 0.12)
            && panorama_transform_overlap_support(
                &match_info.homography,
                source.dimensions(),
                target.dimensions(),
            ) >= 0.60;
    }
    if match_info.points.len() < FOCUS_MODEL_MIN_INLIERS {
        return false;
    }
    compatible_focal_length
        && focus_match_center_motion_ratio(source, target, match_info)
            .is_some_and(|motion| motion <= FOCUS_BRACKET_MAX_CENTER_MOTION_RATIO)
        && panorama_transform_overlap_support(
            &match_info.homography,
            source.dimensions(),
            target.dimensions(),
        ) >= FOCUS_BRACKET_MIN_OVERLAP_SUPPORT
}

/// The four pieces of image evidence 需求 1.1 allows a Capture_Station
/// membership decision to rest on, together with the derived score 需求 1.2
/// compares when it needs a tie-break.
///
/// Every field is a measurement, never a threshold: the accept/reject decision
/// lives in [`StationEvidence::accepts`] so that the numbers can be reported
/// (需求 1.7) for accepted *and* rejected relations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StationEvidence {
    /// Verified overlap inlier correspondences (需求 1.1, ≥ 30).
    pub inliers: usize,
    /// Local normalised pixel correlation over the overlap (需求 1.1, ≥ 0.60).
    ///
    /// Meaningful only when `overlap_ncc_measured` is set.
    pub overlap_ncc: f64,
    /// Whether the overlap correlation could be measured at all.
    ///
    /// The correlation needs at least 120 usable samples out of a 32px-or-larger
    /// alignment image; the design keeps that as the *measurable precondition*
    /// of the threshold. Synthetic sources that carry a placeholder alignment
    /// image report no correlation evidence in either direction, exactly like
    /// the `alignment_is_measurable` gate the station relation scorer uses.
    pub overlap_ncc_measured: bool,
    /// Inlier spread inside the overlap (需求 1.1, ≥ 0.20).
    pub spatial_support: f64,
    /// `sqrt(σ1 · σ2)` of the linear part (需求 1.1, ∈ `[0.98, 1.02]`).
    pub scale_ratio: f64,
    /// Composite of the four measurements above. Ordering only — never a gate.
    pub score: f64,
}

impl Default for StationEvidence {
    fn default() -> Self {
        Self {
            inliers: 0,
            overlap_ncc: 0.0,
            overlap_ncc_measured: false,
            spatial_support: 0.0,
            scale_ratio: 1.0,
            score: 0.0,
        }
    }
}

impl StationEvidence {
    /// Stable identifier of the evidence kind, for the Stack_Report (需求 1.7).
    pub(crate) const KIND: &'static str = "verified_overlap";

    /// Whether all four measurements clear 需求 1.1 at once.
    pub(super) fn accepts(&self) -> bool {
        self.inliers >= STATION_MIN_INLIER_MATCHES
            && (!self.overlap_ncc_measured || self.overlap_ncc >= STATION_MIN_OVERLAP_NCC)
            && self.spatial_support >= STATION_MIN_INLIER_SPATIAL_SUPPORT
            && (STATION_SCALE_RATIO_MIN..=STATION_SCALE_RATIO_MAX).contains(&self.scale_ratio)
    }

    /// Composite score used only where 需求 1.2 permits a comparison: to order
    /// candidates, and to decide whether an absolute-path tie-break is allowed
    /// at all (`|Δscore| ≤ 0.001`).
    ///
    /// Each factor is normalised into `[0, 1]` so the score stays a function of
    /// the measurements and of nothing else.
    pub(super) fn from_measurements(
        inliers: usize,
        overlap_ncc: Option<f64>,
        spatial_support: f64,
        scale_ratio: f64,
    ) -> Self {
        let inlier_term =
            (inliers as f64 / STATION_MIN_INLIER_MATCHES as f64).clamp(0.0, 4.0) / 4.0;
        // An unmeasurable correlation contributes the bar itself, so a source
        // with a placeholder alignment image neither gains nor loses score.
        let correlation_term = overlap_ncc
            .unwrap_or(STATION_MIN_OVERLAP_NCC)
            .clamp(0.0, 1.0);
        let support_term = spatial_support.clamp(0.0, 1.0);
        let scale_term = if scale_ratio.is_finite() && scale_ratio > 0.0 {
            (1.0 - (scale_ratio - 1.0).abs() / (STATION_SCALE_RATIO_MAX - 1.0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            inliers,
            overlap_ncc: overlap_ncc.unwrap_or(0.0),
            overlap_ncc_measured: overlap_ncc.is_some(),
            spatial_support,
            scale_ratio,
            score: 0.25 * (inlier_term + correlation_term + support_term + scale_term),
        }
    }
}

/// Measure the 需求 1.1 evidence of one pairwise relation.
///
/// The measurement runs in the content-derived direction that
/// [`canonical_match_direction`] picks, not in the `(min, max)` orientation the
/// `matches` key happens to use: `focus_overlap_quality` warps the source into
/// the target before correlating and `panorama_spatial_support` weighs the
/// source spread first, so scoring in the key direction let a shuffled import
/// award the same physical relation different numbers (需求 1.2 / 14.6).
/// `canonical_homography` is the untouched fit in exactly that direction, which
/// avoids the lossy `inv(inv(M))` round trip.
pub(super) fn focus_station_link_evidence(
    images: &[ImageInfo],
    first: usize,
    second: usize,
    match_info: &MatchInfo,
) -> Option<StationEvidence> {
    // A coarse correlation peak or a filename-sequence bridge is an ordering
    // cue whose points are synthetic. It may connect two already established
    // stations, but it is not overlap evidence and can never merge two camera
    // positions into one bracket.
    if match_info.sequence_bridge || match_info.coarse_bridge {
        return None;
    }
    let (source_index, target_index, inverted) = canonical_match_direction(images, first, second);
    let source = images.get(source_index)?;
    let target = images.get(target_index)?;
    if !focus_compatible_focal_length(source, target) {
        return None;
    }
    let homography = match match_info.canonical_homography {
        Some(canonical) => canonical,
        None if inverted => match_info.homography.try_inverse()?,
        None => match_info.homography,
    };
    // A coarse precondition, subordinate to the accumulated-displacement split
    // of 需求 1.4: a relation whose own centre displacement is already several
    // times the 0.02 accumulated budget cannot be an intra-station relation, and
    // admitting it would only make the split undo it. The binding motion
    // criterion remains `STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO`.
    if !focus_transform_center_motion_ratio(source, target, &homography)
        .is_some_and(|motion| motion <= FOCUS_BRACKET_MAX_CENTER_MOTION_RATIO)
    {
        return None;
    }
    let points: Cow<'_, [(Point2<f64>, Point2<f64>)]> = if inverted {
        Cow::Owned(
            match_info
                .points
                .iter()
                .map(|&(source_point, target_point)| (target_point, source_point))
                .collect(),
        )
    } else {
        Cow::Borrowed(&match_info.points)
    };
    // `focus_overlap_quality` returns `None` below 120 usable samples, which is
    // the measurable precondition of the correlation threshold.  Distinguish
    // "the overlap correlates badly" (reject) from "this source pair carries no
    // alignment image large enough to correlate on" (no evidence either way),
    // using the same 32px test the station relation scorer applies.
    let alignment_is_measurable = source.alignment_image.width() >= 32
        && source.alignment_image.height() >= 32
        && target.alignment_image.width() >= 32
        && target.alignment_image.height() >= 32;
    let overlap_ncc = match focus_overlap_quality(source, target, &homography) {
        Some((intensity_ncc, _, _, _)) => Some(intensity_ncc),
        None if alignment_is_measurable => return None,
        None => None,
    };
    let spatial_support =
        panorama_spatial_support(&points, source.dimensions(), target.dimensions());
    let scale_ratio = homography_scale_ratio(&homography)?;
    Some(StationEvidence::from_measurements(
        points.len(),
        overlap_ncc,
        spatial_support,
        scale_ratio,
    ))
}

/// Whether two sources belong to the same Capture_Station (需求 1.1).
pub(super) fn focus_match_is_capture_station_link(
    images: &[ImageInfo],
    first: usize,
    second: usize,
    match_info: &MatchInfo,
) -> bool {
    focus_station_link_evidence(images, first, second, match_info)
        .is_some_and(|evidence| evidence.accepts())
}

/// One Capture_Station split, recorded for the Stack_Report (需求 1.4 / 1.10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StationSplit {
    /// Verified-link component the split was applied to, in the component order
    /// the grouper walks. Station indices do not exist yet at split time.
    pub component: usize,
    /// Position inside the candidate order of the component at which the split
    /// falls: the candidate at this position starts the next Capture_Station.
    pub position: usize,
    /// Accumulated frame-centre displacement measured there, in long sides.
    pub motion_ratio: f64,
    /// [`degradation::GROUPING_ACCUMULATED_MOTION`] or
    /// [`degradation::GROUPING_MEMBER_LIMIT_SPLIT`].
    pub reason: &'static str,
}

/// Relative poses of one accepted station link, both derived from the untouched
/// canonical fit with exactly one inversion each.
///
/// Composing `homography` and `homography.try_inverse()` would make one of the
/// two directions an `inv(inv(M))` round trip whenever the `(min, max)` key
/// orientation disagrees with the content-derived measurement direction, and
/// that round trip is not bit exact (需求 14.6).
pub(super) fn station_link_relative_poses(
    images: &[ImageInfo],
    left: usize,
    right: usize,
    match_info: &MatchInfo,
) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
    let (source_index, _, inverted) = canonical_match_direction(images, left, right);
    let canonical = match match_info.canonical_homography {
        Some(canonical) => canonical,
        None if inverted => match_info.homography.try_inverse()?,
        None => match_info.homography,
    };
    let reverse = canonical.try_inverse()?;
    Some(if source_index == left {
        (canonical, reverse)
    } else {
        (reverse, canonical)
    })
}

/// Split one verified-link component into Capture_Stations at the candidates
/// where the accumulated frame-centre displacement leaves the bracket window
/// (需求 1.4), then enforce the 48-member ceiling (需求 1.5 / 1.10).
///
/// The candidate order is the relative world-centre order recovered from the
/// verified link graph; the absolute path only breaks an exact centre tie
/// (需求 1.2 / 1.3).
pub(super) fn split_focus_local_component_by_motion(
    component: Vec<usize>,
    component_index: usize,
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    splits: &mut Vec<StationSplit>,
) -> Vec<Vec<usize>> {
    if component.is_empty() {
        return Vec::new();
    }

    // Recover a relative pose for every member using the verified local-link
    // graph.  Sorting by filename and only testing consecutive pairs would let
    // a renamed or shuffled import split one focus bracket even when the same
    // geometric edges were present.
    let member_set = component.iter().copied().collect::<HashSet<_>>();
    let mut adjacency = HashMap::<usize, Vec<(usize, Matrix3<f64>, f64)>>::new();
    for &left in &component {
        for &right in &component {
            if left >= right || !member_set.contains(&right) {
                continue;
            }
            let Some(match_info) = matches.get(&(left, right)) else {
                continue;
            };
            let Some(evidence) = focus_station_link_evidence(images, left, right, match_info)
                .filter(StationEvidence::accepts)
            else {
                continue;
            };
            let Some((left_to_right, right_to_left)) =
                station_link_relative_poses(images, left, right, match_info)
            else {
                continue;
            };
            adjacency
                .entry(left)
                .or_default()
                .push((right, right_to_left, evidence.score));
            adjacency
                .entry(right)
                .or_default()
                .push((left, left_to_right, evidence.score));
        }
    }

    // 需求 1.2: the score may order candidates, and the absolute path may only
    // step in once two scores are within 0.001 of each other.  Quantise the
    // score before applying the tie-break so the comparator remains a total
    // order.  A direct "within epsilon" comparison is not transitive: A can
    // tie B, B can tie C, while A and C are outside the epsilon, which makes
    // Rust's slice sort panic on real mixed-focal stacks.
    let evidence_cmp = |left_score: f64, right_score: f64, left: usize, right: usize| {
        let score_bucket = |score: f64| {
            if score.is_nan() {
                i64::MIN
            } else if score == f64::INFINITY {
                i64::MAX
            } else if score == f64::NEG_INFINITY {
                i64::MIN + 1
            } else {
                (score / STATION_EVIDENCE_SCORE_TIE_BREAK).floor() as i64
            }
        };
        score_bucket(right_score)
            .cmp(&score_bucket(left_score))
            .then_with(|| {
                focus_image_direction_key(&images[left])
                    .cmp(&focus_image_direction_key(&images[right]))
                    .then_with(|| natural_path_cmp(&images[left].filename, &images[right].filename))
                    .then_with(|| left.cmp(&right))
            })
            .then_with(|| right_score.total_cmp(&left_score))
    };
    let link_support = |image_index: &usize| {
        adjacency
            .get(image_index)
            .into_iter()
            .flatten()
            .map(|(_, _, score)| *score)
            .sum::<f64>()
    };
    let anchor = component
        .iter()
        .copied()
        .min_by(|left, right| evidence_cmp(link_support(left), link_support(right), *left, *right))
        .unwrap_or(component[0]);
    let mut relative_to_anchor = HashMap::<usize, Matrix3<f64>>::new();
    relative_to_anchor.insert(anchor, Matrix3::identity());
    let mut pending = VecDeque::from([anchor]);
    while let Some(current) = pending.pop_front() {
        let current_to_anchor = relative_to_anchor[&current];
        let mut neighbours = adjacency.get(&current).cloned().unwrap_or_default();
        neighbours.sort_by(|left, right| evidence_cmp(left.2, right.2, left.0, right.0));
        for (neighbor, neighbor_to_current, _) in neighbours {
            if relative_to_anchor.contains_key(&neighbor) {
                continue;
            }
            relative_to_anchor.insert(neighbor, current_to_anchor * neighbor_to_current);
            pending.push_back(neighbor);
        }
    }

    // The candidate order is the recovered world-centre order.  Members whose
    // pose could not be recovered carry no centre and sort last; they cannot
    // take part in a displacement measurement either way.
    let member_center = |image_index: usize| {
        relative_to_anchor.get(&image_index).and_then(|relative| {
            transformed_point(
                relative,
                Point2::new(
                    images[image_index].width as f64 * 0.5,
                    images[image_index].height as f64 * 0.5,
                ),
            )
        })
    };
    let mut ordered = component;
    ordered.sort_by(|left, right| {
        let left_center = member_center(*left);
        let right_center = member_center(*right);
        match (left_center, right_center) {
            (Some(left_center), Some(right_center)) => left_center
                .y
                .total_cmp(&right_center.y)
                .then_with(|| left_center.x.total_cmp(&right_center.x)),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
        // 需求 1.2: the absolute path enters only once the world centres compare
        // exactly equal, and the member index is the final stable key.
        .then_with(|| {
            focus_image_direction_key(&images[*left])
                .cmp(&focus_image_direction_key(&images[*right]))
        })
        .then_with(|| natural_path_cmp(&images[*left].filename, &images[*right].filename))
        .then_with(|| left.cmp(right))
    });

    let long_side = |image_index: usize| {
        images[image_index]
            .width
            .max(images[image_index].height)
            .max(1) as f64
    };
    // Step displacement between adjacent candidates, normalised by the longer of
    // the two long sides.  `None` means "not measurable", which is treated as a
    // boundary: two members without a common recovered pose have no evidence
    // that they share a camera position.
    let step_motion = |previous: usize, current: usize| -> Option<f64> {
        let previous_center = member_center(previous)?;
        let current_center = member_center(current)?;
        let scale = long_side(previous).max(long_side(current));
        let ratio = (current_center - previous_center).norm() / scale;
        ratio.is_finite().then_some(ratio)
    };

    // 需求 1.4: walk the candidate order and accumulate the centre displacement
    // since the first member of the current Capture_Station.  A single split
    // position is therefore a function of the measured poses alone.
    let mut groups = Vec::<Vec<usize>>::new();
    let mut current = Vec::<usize>::new();
    let mut accumulated = 0.0f64;
    for (position, image_index) in ordered.iter().copied().enumerate() {
        if current.is_empty() {
            current.push(image_index);
            accumulated = 0.0;
            continue;
        }
        let previous = *current.last().expect("non-empty above");
        match step_motion(previous, image_index) {
            Some(step) => {
                let candidate_accumulated = accumulated + step;
                if candidate_accumulated > STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO {
                    splits.push(StationSplit {
                        component: component_index,
                        position,
                        motion_ratio: candidate_accumulated,
                        reason: degradation::GROUPING_ACCUMULATED_MOTION,
                    });
                    groups.push(std::mem::take(&mut current));
                    current.push(image_index);
                    accumulated = 0.0;
                } else {
                    accumulated = candidate_accumulated;
                    current.push(image_index);
                }
            }
            None => {
                splits.push(StationSplit {
                    component: component_index,
                    position,
                    motion_ratio: f64::NAN,
                    reason: degradation::GROUPING_ACCUMULATED_MOTION,
                });
                groups.push(std::mem::take(&mut current));
                current.push(image_index);
                accumulated = 0.0;
            }
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }

    // 需求 1.5 / 1.10: a Capture_Station holds 1..=48 sources.  Split an
    // oversized one at the adjacent candidate pair carrying the largest
    // displacement, repeatedly, until every station fits.
    let mut bounded = Vec::<Vec<usize>>::with_capacity(groups.len());
    let mut oversized = VecDeque::from(groups);
    while let Some(group) = oversized.pop_front() {
        if station_member_count_is_supported(group.len()) {
            bounded.push(group);
            continue;
        }
        let mut best: Option<(f64, usize)> = None;
        for position in 1..group.len() {
            let step = step_motion(group[position - 1], group[position]).unwrap_or(f64::INFINITY);
            let balance = position.abs_diff(group.len() / 2);
            let better = match best {
                None => true,
                Some((best_step, best_position)) => {
                    if (step - best_step).abs() <= f64::EPSILON {
                        // Equal displacement — all-zero motion included — splits
                        // as evenly as possible so the recursion terminates
                        // quickly and deterministically.
                        balance < best_position.abs_diff(group.len() / 2)
                    } else {
                        step > best_step
                    }
                }
            };
            if better {
                best = Some((step, position));
            }
        }
        let (motion_ratio, position) = best.unwrap_or((f64::NAN, group.len() / 2));
        splits.push(StationSplit {
            component: component_index,
            position,
            motion_ratio,
            reason: degradation::GROUPING_MEMBER_LIMIT_SPLIT,
        });
        let tail = group[position..].to_vec();
        let head = group[..position].to_vec();
        // Push both halves back so a half that is still oversized is split again.
        oversized.push_front(tail);
        oversized.push_front(head);
    }
    bounded
}

/// Capture_Station membership of one selection, with everything 需求 1.6 / 1.7 /
/// 1.9 / 1.4 / 1.10 ask the Stack_Report to carry.
#[derive(Clone, Debug, Default)]
pub(crate) struct StationGroupingOutcome {
    /// Capture_Stations in station order, including connected single-frame scans.
    pub components: Vec<Vec<usize>>,
    /// 需求 1.6: sources with neither same-pose nor verified scan-overlap evidence.
    pub isolated: Vec<usize>,
    /// 需求 1.4 / 1.10: every split position, in the order it was applied.
    pub splits: Vec<StationSplit>,
}

/// Group the selection into Capture_Stations from image evidence only.
///
/// Membership is a DSU over the accepted 需求 1.1 relations — a set operation,
/// so it cannot see the insertion order — followed by the accumulated-motion
/// split above. A verified scan overlap marks a source as connected without
/// unioning the camera poses: an unbracketed scan remains a single-frame station.
pub(super) fn focus_match_has_verified_scan_overlap(
    images: &[ImageInfo],
    source_index: usize,
    target_index: usize,
    match_info: &MatchInfo,
) -> bool {
    if source_index >= images.len() || target_index >= images.len() {
        return false;
    }
    !match_info.coarse_bridge
        && !match_info.sequence_bridge
        && match_info.points.len() >= 30
        && median_symmetric_error(&match_info.homography, &match_info.points) <= 3.0
        && match_has_spatial_support(
            &match_info.points,
            images[source_index].dimensions(),
            images[target_index].dimensions(),
        )
        && (focus_overlap_quality(
            &images[source_index],
            &images[target_index],
            &match_info.homography,
        )
        .is_some_and(|quality| {
            focus_overlap_is_verified(
                quality,
                match_info.points.len(),
                median_symmetric_error(&match_info.homography, &match_info.points),
                false,
            )
        }) || shifted_focus_structure_is_verified(
            &images[source_index],
            &images[target_index],
            &match_info.homography,
            &match_info.points,
            false,
        ))
}

pub(super) fn focus_station_grouping(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> StationGroupingOutcome {
    let mut dsu = Dsu::new(images.len());
    let mut linked = vec![false; images.len()];
    for (&(source_index, target_index), match_info) in sorted_pairs(matches) {
        if source_index >= images.len() || target_index >= images.len() {
            continue;
        }
        if focus_match_is_capture_station_link(images, source_index, target_index, match_info) {
            dsu.union(source_index, target_index);
            linked[source_index] = true;
            linked[target_index] = true;
        } else if (!linked[source_index] || !linked[target_index])
            && focus_match_has_verified_scan_overlap(images, source_index, target_index, match_info)
        {
            // A verified spatial scan edge proves this source is connected,
            // but does not prove the two frames share a camera pose. Preserve
            // each as its own station when it has no focus-bracket partner.
            linked[source_index] = true;
            linked[target_index] = true;
        }
    }
    // Ordered by representative index: the bracket split below is applied in this
    // order and its output feeds the station order (需求 14.6).
    let mut components = BTreeMap::<usize, Vec<usize>>::new();
    for image_index in 0..images.len() {
        let root = dsu.find(image_index);
        components.entry(root).or_default().push(image_index);
    }
    let mut splits = Vec::new();
    let mut components = components
        .into_values()
        .enumerate()
        .flat_map(|(component_index, component)| {
            split_focus_local_component_by_motion(
                component,
                component_index,
                images,
                matches,
                &mut splits,
            )
        })
        .collect::<Vec<_>>();
    // 需求 1.6: a source with no accepted overlap relation at all is isolated.
    // A single member left over by a *split* is not: it had verified evidence,
    // the accumulated motion simply put it at its own camera position.
    let isolated = (0..images.len())
        .filter(|&image_index| !linked[image_index])
        .collect::<Vec<_>>();
    components.retain(|component| component.iter().any(|&index| linked[index]));
    for component in &mut components {
        component.sort_by(|left, right| {
            natural_path_cmp(&images[*left].filename, &images[*right].filename)
                // `natural_path_cmp` can rate two distinct names equal; the
                // member index is the stable secondary key (需求 14.6).
                .then_with(|| left.cmp(right))
        });
    }
    components.sort_by(|left, right| {
        natural_path_cmp(&images[left[0]].filename, &images[right[0]].filename)
            .then_with(|| left[0].cmp(&right[0]))
    });
    StationGroupingOutcome {
        components,
        isolated,
        splits,
    }
}

pub(super) fn focus_local_bracket_components(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<Vec<usize>> {
    focus_station_grouping(images, matches).components
}
