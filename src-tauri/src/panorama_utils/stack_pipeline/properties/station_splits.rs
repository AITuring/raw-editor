//! Property 3: accumulated-motion split positions are uniquely determined.
//!
//! The candidate order and the accumulated frame-centre displacement are rebuilt
//! from ground truth and compared with the Station_Grouper split.

use super::super::degradation;
use super::super::test_support;
use super::super::test_support::SyntheticScan;
use super::super::test_support::arb_artwork_focus_bracket;
use super::super::test_support::arb_oversized_bracket;
use crate::panorama_stitching::grouping_test_access;
use proptest::prelude::*;
use std::collections::BTreeSet;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use super::{
    REQUIRED_MAX_ACCUMULATED_MOTION_RATIO, REQUIRED_MAX_STATION_MEMBERS, accepted_link_components,
    fit_synthetic_matches,
};

/// Candidate order and per-step frame-centre displacement of one component,
/// computed from the *ground truth* poses of the fixture.
///
/// This is the independent oracle of Property 3.  `split_focus_local_component_by_motion`
/// recovers the same order from the verified link graph and measures the same
/// displacements from the recovered relative poses; the numbers here come from
/// the poses the generator cropped the tiles with, so agreement is a statement
/// about the production code rather than a restatement of it.
///
/// The recovered frame is the plane scaled by the inverse of the anchor frame's
/// focus breathing, so a measured ratio differs from the ratio computed here by
/// up to the breathing amplitude (0.15% in `arb_focus_bracket`).  Property 3
/// therefore skips candidates whose accumulated displacement sits within
/// [`ACCUMULATED_MOTION_DECISION_BAND`] of the 0.02 budget instead of pretending
/// the two frames are the same one.
pub(super) fn ground_truth_candidate_order(
    scan: &SyntheticScan,
    component: &BTreeSet<usize>,
) -> Vec<usize> {
    let mut ordered = component.iter().copied().collect::<Vec<_>>();
    ordered.sort_by(|&left, &right| {
        let left_centre = scan.plane_centre(left);
        let right_centre = scan.plane_centre(right);
        left_centre
            .y
            .total_cmp(&right_centre.y)
            .then_with(|| left_centre.x.total_cmp(&right_centre.x))
    });
    ordered
}

/// Width of the band around 需求 1.4's 0.02 budget in which the ground truth
/// oracle and the production measurement are allowed to disagree, in long sides.
///
/// Five percent of the budget, i.e. 30 times the focus breathing amplitude the
/// two frames differ by.
pub(super) const ACCUMULATED_MOTION_DECISION_BAND: f64 = 1e-3;

/// Accumulated frame-centre displacement per candidate position, with the
/// 需求 1.4 reset at every split, computed from ground truth.
///
/// Returns `(split positions, accumulated value at every position)`.
pub(super) fn ground_truth_accumulated_motion(
    scan: &SyntheticScan,
    ordered: &[usize],
) -> (BTreeSet<usize>, Vec<f64>) {
    let long_side = f64::from(test_support::SYNTHETIC_TILE_SIDE);
    let mut splits = BTreeSet::new();
    let mut accumulated = vec![0.0f64; ordered.len()];
    let mut running = 0.0f64;
    for position in 1..ordered.len() {
        let step = (scan.plane_centre(ordered[position])
            - scan.plane_centre(ordered[position - 1]))
        .norm()
            / long_side;
        let candidate = running + step;
        accumulated[position] = candidate;
        if candidate > REQUIRED_MAX_ACCUMULATED_MOTION_RATIO {
            splits.insert(position);
            running = 0.0;
        } else {
            running = candidate;
        }
    }
    (splits, accumulated)
}

/// Whether Property 3 has seen a component that split and one that did not.
///
/// The accumulated displacement of a drawn bracket is data, not a rotation, so
/// this is a coverage witness rather than a schedule: bit 0 is "a component that
/// stayed whole", bit 1 is "a component that 需求 1.4 split".  Asserted once, on
/// the last case, so a fixture that degenerated into only one of the two outcomes
/// fails instead of quietly testing half the property.
pub(super) static ACCUMULATED_MOTION_OUTCOMES: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 3 so far.
pub(super) static ACCUMULATED_MOTION_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 3 is configured with, mirrored here so the coverage assertion
/// knows which case is the last one.
pub(super) const ACCUMULATED_MOTION_CASES: usize = 100;

proptest! {
    #![proptest_config(ProptestConfig { cases: ACCUMULATED_MOTION_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 3: 对于任意候选序列与其
    // 累计画面中心位移序列，Station_Grouper 的拆分位置集合恰好等于累计位移超过画面长边
    // 0.02 倍的位置集合；且拆分后每个 Capture_Station 的成员数量不超过 48。
    //
    // 两个生成器分别覆盖两条判据：
    //   * `arb_artwork_focus_bracket()` 的机位内层序列驱动需求 1.4 的累计位移拆分。期望
    //     拆分位置由生成器的**真值位姿**独立算出（生产代码是从被接受的链接图里重新恢复
    //     相对位姿再测量的），所以这条断言比较的是两套独立的计算。
    //   * `arb_oversized_bracket()` 的 49..60 成员分量驱动需求 1.5 / 1.10 的成员上限：
    //     该分量的累计位移按构造低于 0.02，因此唯一能拆分它的就是 48 的上限，且首个拆分
    //     必须落在位移最大的相邻候选处。
    //
    // **Validates: Requirements 1.4, 1.5, 1.10**
    #[test]
    fn accumulated_motion_split_positions_are_uniquely_determined(
        scan in arb_artwork_focus_bracket(),
        oversized in arb_oversized_bracket(),
    ) {
        prop_assert_eq!(
            grouping_test_access::MAX_ACCUMULATED_CENTER_MOTION_RATIO,
            REQUIRED_MAX_ACCUMULATED_MOTION_RATIO
        );
        prop_assert_eq!(
            grouping_test_access::MAX_STATION_MEMBERS,
            REQUIRED_MAX_STATION_MEMBERS
        );

        // ---------------------------------------------------------------
        // 需求 1.4: the split positions of one focus bracket
        // ---------------------------------------------------------------
        let images = scan.images();
        let (matches, _measured) = fit_synthetic_matches(&scan, &images);
        let components = accepted_link_components(&images, &matches)
            .into_iter()
            .filter(|component| component.len() >= 2)
            .collect::<Vec<_>>();
        prop_assert!(
            !components.is_empty(),
            "the fixture must produce a verified-link component to split"
        );
        let mut split_components = 0usize;
        let mut whole_components = 0usize;
        for component in &components {
            let ordered = ground_truth_candidate_order(&scan, component);
            let (expected_splits, accumulated) = ground_truth_accumulated_motion(&scan, &ordered);
            // A candidate whose accumulated displacement sits on the budget is a
            // measurement the two frames cannot agree on to the last bit; skip
            // the case rather than assert a coin flip.
            for value in &accumulated {
                prop_assume!(
                    (value - REQUIRED_MAX_ACCUMULATED_MOTION_RATIO).abs()
                        > ACCUMULATED_MOTION_DECISION_BAND
                );
            }
            let (groups, splits) = grouping_test_access::split_component_by_motion(
                component.iter().copied().collect(),
                &images,
                &matches,
            );
            // Every member is measurable here: the layers of one camera position
            // all overlap each other, so the link graph reaches all of them and no
            // split can come from an unrecoverable pose.
            for split in &splits {
                prop_assert!(
                    split.motion_ratio.is_finite(),
                    "a fully linked bracket must not split on an unmeasurable pose: {split:?}"
                );
            }
            let measured_splits = splits
                .iter()
                .filter(|split| split.reason == degradation::GROUPING_ACCUMULATED_MOTION)
                .map(|split| split.position)
                .collect::<BTreeSet<_>>();
            prop_assert_eq!(
                &measured_splits,
                &expected_splits,
                "需求 1.4 split positions must be exactly the positions where the accumulated \
                 frame-centre displacement passes {} (accumulated {:?}, order {:?})",
                REQUIRED_MAX_ACCUMULATED_MOTION_RATIO,
                accumulated,
                ordered
            );
            for split in &splits {
                if split.reason == degradation::GROUPING_ACCUMULATED_MOTION {
                    prop_assert!(
                        (split.motion_ratio - accumulated[split.position]).abs()
                            <= ACCUMULATED_MOTION_DECISION_BAND,
                        "the recorded displacement {} of the split at {} disagrees with the \
                         ground truth {}",
                        split.motion_ratio,
                        split.position,
                        accumulated[split.position]
                    );
                }
            }
            // The stations are the runs of the candidate order between the splits,
            // so the split positions determine the membership completely.
            let mut expected_groups = Vec::<Vec<usize>>::new();
            for (position, &member) in ordered.iter().enumerate() {
                if position == 0 || expected_splits.contains(&position) {
                    expected_groups.push(Vec::new());
                }
                expected_groups
                    .last_mut()
                    .expect("a group was pushed for position 0")
                    .push(member);
            }
            prop_assert_eq!(
                &groups,
                &expected_groups,
                "the Capture_Stations of one component must be the runs of its candidate order"
            );
            for group in &groups {
                prop_assert!(
                    !group.is_empty() && group.len() <= REQUIRED_MAX_STATION_MEMBERS,
                    "a Capture_Station holds 1..={} sources, got {}",
                    REQUIRED_MAX_STATION_MEMBERS,
                    group.len()
                );
            }
            if measured_splits.is_empty() {
                whole_components += 1;
            } else {
                split_components += 1;
            }
        }

        // ---------------------------------------------------------------
        // 需求 1.5 / 1.10: the 48 member ceiling
        // ---------------------------------------------------------------
        prop_assert!(
            oversized.len() > REQUIRED_MAX_STATION_MEMBERS,
            "the ceiling is only reachable above {} members",
            REQUIRED_MAX_STATION_MEMBERS
        );
        prop_assert!(
            oversized.accumulated_motion_ratio < REQUIRED_MAX_ACCUMULATED_MOTION_RATIO,
            "the oversized fixture must not be splittable by 需求 1.4, got {}",
            oversized.accumulated_motion_ratio
        );
        let (bounded, ceiling_splits) = grouping_test_access::split_component_by_motion(
            (0..oversized.len()).collect(),
            &oversized.images,
            &oversized.matches,
        );
        for split in &ceiling_splits {
            prop_assert_eq!(
                split.reason,
                degradation::GROUPING_MEMBER_LIMIT_SPLIT,
                "an accumulated displacement below the budget must not split this component: \
                 {:?}",
                split
            );
        }
        for group in &bounded {
            prop_assert!(
                !group.is_empty() && group.len() <= REQUIRED_MAX_STATION_MEMBERS,
                "the member ceiling was not enforced: a station of {} sources",
                group.len()
            );
        }
        prop_assert_eq!(
            bounded.iter().flatten().copied().collect::<Vec<_>>(),
            (0..oversized.len()).collect::<Vec<_>>(),
            "the ceiling split must keep every member, in candidate order"
        );
        prop_assert_eq!(
            ceiling_splits.len(),
            bounded.len() - 1,
            "one recorded split per new Capture_Station (需求 1.10)"
        );
        // 需求 1.10 splits at the adjacent candidate pair carrying the largest
        // accumulated displacement, and the fixture puts a unique maximum there.
        prop_assert_eq!(
            ceiling_splits[0].position,
            oversized.largest_step_position,
            "the first ceiling split must fall at the largest frame-centre displacement"
        );
        let expected_largest_ratio = test_support::OVERSIZED_BRACKET_LARGEST_STEP_PX
            / f64::from(test_support::OVERSIZED_BRACKET_SIDE);
        prop_assert!(
            (ceiling_splits[0].motion_ratio - expected_largest_ratio).abs() < 1e-9,
            "the recorded displacement {} of the first ceiling split is not the largest step {}",
            ceiling_splits[0].motion_ratio,
            expected_largest_ratio
        );

        // Coverage of the 需求 1.4 dimension across the run.
        let outcomes = ACCUMULATED_MOTION_OUTCOMES.fetch_or(
            usize::from(whole_components > 0) | (usize::from(split_components > 0) << 1),
            Ordering::Relaxed,
        ) | usize::from(whole_components > 0)
            | (usize::from(split_components > 0) << 1);
        let case = ACCUMULATED_MOTION_CURSOR.fetch_add(1, Ordering::Relaxed);
        prop_assert!(
            case + 1 < ACCUMULATED_MOTION_CASES || outcomes == 0b11,
            "the run must have exercised both a bracket 需求 1.4 kept whole and one it split, \
             outcome mask {outcomes:#b}"
        );
    }
}
