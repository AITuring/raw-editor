//! Capture-station membership, grouping, and anchor-frame helpers.
//!
//! This module is part of the station-solver facade and preserves the
//! production helper visibility used by the focus-stack pipeline.

use super::*;

/// Frame-centre displacement of an explicit transform, in long sides.
pub(crate) fn focus_transform_center_motion_ratio(
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

pub(crate) fn focus_match_center_motion_ratio(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> Option<f64> {
    focus_transform_center_motion_ratio(source, target, &match_info.homography)
}

pub(crate) fn focus_match_is_local_bracket(
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
    pub(crate) fn accepts(&self) -> bool {
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
    fn from_measurements(
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
pub(crate) fn focus_station_link_evidence(
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
pub(crate) fn focus_match_is_capture_station_link(
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
pub(crate) fn station_link_relative_poses(
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
pub(crate) fn split_focus_local_component_by_motion(
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
    // step in once two scores are within 0.001 of each other.
    let evidence_cmp = |left_score: f64, right_score: f64, left: usize, right: usize| {
        if (left_score - right_score).abs() <= STATION_EVIDENCE_SCORE_TIE_BREAK {
            focus_image_direction_key(&images[left])
                .cmp(&focus_image_direction_key(&images[right]))
                .then_with(|| natural_path_cmp(&images[left].filename, &images[right].filename))
                .then_with(|| left.cmp(&right))
        } else {
            right_score.total_cmp(&left_score)
        }
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
pub(crate) fn focus_station_grouping(
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
            && !match_info.coarse_bridge
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

pub(crate) fn focus_local_bracket_components(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<Vec<usize>> {
    focus_station_grouping(images, matches).components
}

/// The selected sources that produced no decoded image (需求 1.9).
///
/// Returned in selection order and by absolute path, which is what the
/// Stack_Report and the degradation ledger record.  Excluding them from every
/// Capture_Station is not a separate step: the grouper only ever sees the decoded
/// images, so a source named here cannot reach it.
pub(crate) fn undecodable_source_paths(selection: &[String], decoded: &[ImageInfo]) -> Vec<String> {
    selection
        .iter()
        .filter(|path| !decoded.iter().any(|image| &image.filename == *path))
        .cloned()
        .collect()
}

/// The Stack_Report records of the sources one grouping excluded (需求 1.6 / 1.9).
///
/// Each record names the absolute path of the excluded Source_RAW and the stable
/// reason identifier it was excluded for, and nothing else: the exclusion itself
/// has already happened by the time this runs.
pub(crate) fn excluded_source_records(
    images: &[ImageInfo],
    indices: &[usize],
    reason: &'static str,
) -> Vec<stack_report::ExcludedSourceRecord> {
    indices
        .iter()
        .filter_map(|&image_index| images.get(image_index))
        .map(|image| stack_report::ExcludedSourceRecord {
            path: image.filename.clone(),
            reason: reason.to_string(),
        })
        .collect()
}

/// Binding 需求 1.1 evidence of one Capture_Station: the weakest accepted
/// relation inside it.
///
/// Every member joined the station through at least this much evidence, so it is
/// the number 需求 1.7 should carry. Returns `None` for a single-member station,
/// which has no intra-station relation to report.
pub(crate) fn focus_station_binding_evidence(
    members: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Option<StationEvidence> {
    let mut weakest: Option<StationEvidence> = None;
    for (offset, &left) in members.iter().enumerate() {
        for &right in &members[offset + 1..] {
            let key = (left.min(right), left.max(right));
            let Some(match_info) = matches.get(&key) else {
                continue;
            };
            let Some(evidence) = focus_station_link_evidence(images, key.0, key.1, match_info)
                .filter(StationEvidence::accepts)
            else {
                continue;
            };
            if weakest.is_none_or(|current| evidence.score < current.score) {
                weakest = Some(evidence);
            }
        }
    }
    weakest
}

pub(crate) fn lock_focus_local_bracket_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    global_homographies: &HashMap<usize, Matrix3<f64>>,
) -> HashMap<usize, Matrix3<f64>> {
    let components = focus_local_bracket_components(images, matches);
    if components.is_empty() {
        return global_homographies.clone();
    }
    let mut locked = global_homographies.clone();
    let mut locked_sources = 0usize;
    let mut maximum_center_shift = 0.0f64;

    for component in &components {
        let members = component.iter().copied().collect::<HashSet<_>>();
        let mut edges = matches
            .iter()
            .filter_map(|(&(source_index, target_index), match_info)| {
                if !members.contains(&source_index)
                    || !members.contains(&target_index)
                    || !focus_match_is_local_bracket(
                        &images[source_index],
                        &images[target_index],
                        match_info,
                    )
                {
                    return None;
                }
                let error = median_symmetric_error(&match_info.homography, &match_info.points);
                let precision = if error.is_finite() {
                    1.0 / (1.0 + error / FOCUS_MODEL_INLIER_THRESHOLD)
                } else {
                    0.0
                };
                Some((
                    match_info.inliers.max(match_info.points.len()) as f64 * precision,
                    source_index,
                    target_index,
                    match_info.homography,
                ))
            })
            .collect::<Vec<_>>();
        edges.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| natural_path_cmp(&images[left.1].filename, &images[right.1].filename))
                .then_with(|| natural_path_cmp(&images[left.2].filename, &images[right.2].filename))
        });

        let mut support = HashMap::<usize, f64>::new();
        for &(weight, source_index, target_index, _) in &edges {
            *support.entry(source_index).or_default() += weight;
            *support.entry(target_index).or_default() += weight;
        }
        let Some(anchor) = component.iter().copied().min_by(|left, right| {
            support
                .get(right)
                .copied()
                .unwrap_or(0.0)
                .total_cmp(&support.get(left).copied().unwrap_or(0.0))
                .then_with(|| natural_path_cmp(&images[*left].filename, &images[*right].filename))
        }) else {
            continue;
        };
        let Some(anchor_global) = global_homographies.get(&images[anchor].id).copied() else {
            continue;
        };

        let mut local_dsu = Dsu::new(images.len());
        let mut adjacency = HashMap::<usize, Vec<(usize, Matrix3<f64>)>>::new();
        for &(_, source_index, target_index, source_to_target) in &edges {
            if local_dsu.find(source_index) == local_dsu.find(target_index) {
                continue;
            }
            let Some(target_to_source) = source_to_target.try_inverse() else {
                continue;
            };
            local_dsu.union(source_index, target_index);
            adjacency
                .entry(source_index)
                .or_default()
                .push((target_index, target_to_source));
            adjacency
                .entry(target_index)
                .or_default()
                .push((source_index, source_to_target));
        }

        let mut relative_to_anchor = HashMap::new();
        relative_to_anchor.insert(anchor, Matrix3::identity());
        let mut pending = VecDeque::from([anchor]);
        while let Some(current) = pending.pop_front() {
            let current_to_anchor = relative_to_anchor[&current];
            for &(neighbor, neighbor_to_current) in adjacency.get(&current).into_iter().flatten() {
                if relative_to_anchor.contains_key(&neighbor) {
                    continue;
                }
                relative_to_anchor.insert(neighbor, current_to_anchor * neighbor_to_current);
                pending.push_back(neighbor);
            }
        }
        if relative_to_anchor.len() != component.len() {
            continue;
        }

        for &image_index in component {
            let replacement = anchor_global * relative_to_anchor[&image_index];
            if !transform_is_stable_for_focus_stack(&replacement, images[image_index].dimensions())
            {
                continue;
            }
            if let Some(previous) = global_homographies.get(&images[image_index].id) {
                let center = Point2::new(
                    images[image_index].width as f64 * 0.5,
                    images[image_index].height as f64 * 0.5,
                );
                if let (Some(before), Some(after)) = (
                    transformed_point(previous, center),
                    transformed_point(&replacement, center),
                ) {
                    maximum_center_shift = maximum_center_shift.max((after - before).norm());
                }
            }
            locked.insert(images[image_index].id, replacement);
            locked_sources += 1;
        }
    }
    println!(
        "  - Locked {} local focus-bracket component(s), {} source poses; maximum center correction {:.2}px",
        components.len(),
        locked_sources,
        maximum_center_shift
    );
    locked
}

pub(crate) struct FocusCaptureGroup {
    pub(crate) members: Vec<usize>,
    pub(crate) anchor: usize,
    pub(crate) local_to_anchor: HashMap<usize, Matrix3<f64>>,
}

pub(crate) struct FocusAnchorRenderFrames {
    pub(crate) homographies: HashMap<usize, Matrix3<f64>>,
    pub(crate) focus_warp: Option<FocusLayerWarp>,
    pub(crate) anchor_to_world: Vec<Matrix3<f64>>,
}

pub(crate) fn focus_anchor_render_frames(
    groups: &[FocusCaptureGroup],
    sources: &[ImageInfo],
    global: &HashMap<usize, Matrix3<f64>>,
    warp: Option<&FocusLayerWarp>,
) -> Option<FocusAnchorRenderFrames> {
    let reference_anchor = sources.get(groups.first()?.anchor)?;
    let world_to_reference = global.get(&reference_anchor.id)?.try_inverse()?;
    let mut local = HashMap::new();
    let mut local_warp = warp.cloned();
    let mut placements = Vec::with_capacity(groups.len());
    for group in groups {
        let anchor = sources.get(group.anchor)?;
        let anchor_global = global.get(&anchor.id)?;
        let world_to_anchor = anchor_global.try_inverse()?;
        placements.push(normalized_homography(
            &(world_to_reference * anchor_global),
        )?);
        for &member in &group.members {
            let source = sources.get(member)?;
            local.insert(
                source.id,
                normalized_homography(&(world_to_anchor * global.get(&source.id)?))?,
            );
            if let Some(warp) = local_warp.as_mut() {
                for band in &mut warp.bands {
                    if let Some(pose) = band.homographies.get_mut(&source.id) {
                        *pose = normalized_homography(&(world_to_anchor * *pose))?;
                    }
                }
            }
        }
    }
    Some(FocusAnchorRenderFrames {
        homographies: local,
        focus_warp: local_warp,
        anchor_to_world: placements,
    })
}
