//! Property 5: the anchor frame is the sharpest member and maps to itself.
//!
//! The 需求 2.1 anchor choice, including the 0.01 tie window that the absolute path
//! decides, and the identity transform of the chosen anchor.

use super::super::intra_station;
use super::super::test_support;
use super::super::test_support::arb_artwork_focus_bracket;
use crate::panorama_stitching::determinism_test_access;
use crate::panorama_stitching::grouping_test_access;
use nalgebra::Matrix3;
use nalgebra::Point2;
use proptest::prelude::*;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use super::{
    REQUIRED_ANCHOR_SHARPNESS_TIE, SyntheticGeometryOutcome, determinism_images,
    fit_synthetic_matches, map_point, synthetic_geometry_outcome,
};

/// The 需求 2.1 anchor rule, applied to a candidate list from scratch.
///
/// The independent oracle of Property 5: highest median Sharpness_Score wins;
/// every candidate less than 0.01 below the highest is a tie and the lowest
/// absolute path among those wins; a candidate set with no measurable score at
/// all falls back to the lowest absolute path.
pub(super) fn expected_anchor(candidates: &[intra_station::AnchorCandidate]) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let best = candidates
        .iter()
        .map(|candidate| candidate.median_sharpness)
        .filter(|score| !score.is_nan())
        .fold(f64::NEG_INFINITY, f64::max);
    let lowest_path = |allowed: &dyn Fn(&intra_station::AnchorCandidate) -> bool| {
        candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| allowed(candidate))
            .min_by(|(left_index, left), (right_index, right)| {
                left.path
                    .cmp(&right.path)
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(index, _)| index)
    };
    if !best.is_finite() {
        return lowest_path(&|_| true);
    }
    lowest_path(&|candidate| best - candidate.median_sharpness < REQUIRED_ANCHOR_SHARPNESS_TIE)
}

/// Which branch of 需求 2.1 decided the anchor of a drawn candidate list.
///
/// Bit 0: every candidate was inside the 0.01 tie window, so the absolute path
/// decided.  Bit 1: at least one candidate was more than the tie window below the
/// best, so the median Sharpness_Score decided.  Asserted on the last case, so a
/// generator that stopped producing one of the two fails rather than quietly
/// testing half the rule.
pub(super) static ANCHOR_DECISION_BRANCHES: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 5 so far, which also drives the rotation of the
/// station-level half.
pub(super) static ANCHOR_DECISION_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 5 is configured with.
pub(super) const ANCHOR_DECISION_CASES: usize = 100;

/// Whether the station-level half of Property 5 has reached a Capture_Station of
/// several frames.
pub(super) static ANCHOR_MULTI_MEMBER_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Every how many cases the station-level half of Property 5 runs.
///
/// That half needs the station pose solve and `focus_capture_groups`, roughly
/// half a second per case, while the rule half costs nothing.  The wiring it
/// checks — which member is handed to `select_anchor` and whether the anchor's
/// transform is the identity — is not input dependent in the way the rule is, so
/// a quarter of the cases covers it at a quarter of the cost.  The whole run
/// still visits 25 independently drawn scans.
pub(super) const ANCHOR_STATION_STRIDE: usize = 4;

proptest! {
    #![proptest_config(ProptestConfig { cases: ANCHOR_DECISION_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 5: 对于任意
    // Capture_Station，被选为锚点帧的 Source_RAW 是有效像素 Sharpness_Score 中位数最高者；
    // 若最高值之差小于 0.01 则为其中绝对路径升序第一者；且锚点帧到机位坐标系的变换是
    // 恒等矩阵。
    //
    // 两层都要断言，因为它们会各自失效：
    //   1. 规则本身 —— `intra_station::select_anchor` 在任意候选集合上与独立重算的
    //      需求 2.1 规则一致，且结果不随候选顺序改变。分数落在 0.004 的网格上，所以
    //      "分数决定" 与 "并列后按路径决定" 两个分支都会被走到（运行结束时断言两者都出现
    //      过），不可测量的分数集合也在输入空间内。
    //   2. 接线 —— 真实 `focus_capture_groups` 为每个 Capture_Station 选出的锚点确实是
    //      `focus_frame_median_sharpness` 最高的成员（按需求 2.1 的规则重算），且
    //      `local_to_anchor[anchor]` 等于单位矩阵。任务 7.4 之前这里取的是世界中心排序的
    //      第一帧，那个实现能通过第 1 层的全部断言而在这里变红。
    //
    // 实测记录（合成素材，非生产缺陷断言）：`focus_frame_median_sharpness` 在 160px
    // 合成瓦片上的取值区间是 0.0010–0.0021（未归一化的 acutance 比值），因此同一机位内
    // 各层之差最大约 0.001，永远落在需求 2.1 的 0.01 并列窗口内——第 2 层因此只会走到
    // 路径 tie-break 分支。合成瓦片的高频内容只有颗粒（被 acutance 的二项式低通压掉）
    // 与少量笔画，真实 9504px 素材在 spacing≈4.6 下测得的是完全不同的量级，所以这是
    // 素材尺度的限制而不是判定规则的缺陷；规则的另一半由第 1 层在抽样分数上覆盖。
    //
    // **Validates: Requirements 2.1**
    #[test]
    fn the_anchor_frame_is_the_sharpest_member_and_maps_to_itself(
        candidates in test_support::arb_anchor_candidates(),
        scan in arb_artwork_focus_bracket(),
        probe in any::<usize>(),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        prop_assert_eq!(
            intra_station::INTRA_STATION_ANCHOR_SHARPNESS_TIE,
            REQUIRED_ANCHOR_SHARPNESS_TIE
        );

        // 1. The rule.
        let chosen = intra_station::select_anchor(&candidates);
        prop_assert_eq!(
            chosen,
            expected_anchor(&candidates),
            "需求 2.1 picks another candidate from {:?}",
            candidates
        );
        let chosen = chosen.expect("a non-empty candidate list has an anchor");
        let best = candidates
            .iter()
            .map(|candidate| candidate.median_sharpness)
            .filter(|score| !score.is_nan())
            .fold(f64::NEG_INFINITY, f64::max);
        if best.is_finite() {
            // Necessity of the two halves of the rule: the winner is inside the
            // tie window, and nothing inside it sorts before its path.
            prop_assert!(
                best - candidates[chosen].median_sharpness < REQUIRED_ANCHOR_SHARPNESS_TIE,
                "the anchor must be within {} of the highest median Sharpness_Score",
                REQUIRED_ANCHOR_SHARPNESS_TIE
            );
            prop_assert!(
                candidates
                    .iter()
                    .filter(|candidate| {
                        best - candidate.median_sharpness < REQUIRED_ANCHOR_SHARPNESS_TIE
                    })
                    .all(|candidate| candidate.path >= candidates[chosen].path),
                "a tied candidate sorts before the chosen anchor {:?}",
                candidates[chosen]
            );
        }
        // The choice is a property of the photographs, not of the order they are
        // offered in.
        let mut rotated = candidates.clone();
        rotated.rotate_left(probe % candidates.len().max(1));
        let rotated_choice = intra_station::select_anchor(&rotated)
            .expect("a rotation keeps the list non-empty");
        prop_assert_eq!(
            &rotated[rotated_choice].path,
            &candidates[chosen].path,
            "the anchor changed when the candidates were offered in another order"
        );

        // Coverage of the two branches of the rule, on the drawn candidates.
        let branch = if best.is_finite()
            && candidates.iter().any(|candidate| {
                best - candidate.median_sharpness >= REQUIRED_ANCHOR_SHARPNESS_TIE
            })
        {
            0b10
        } else {
            0b01
        };
        let covered = ANCHOR_DECISION_BRANCHES.fetch_or(branch, Ordering::Relaxed) | branch;
        let case = ANCHOR_DECISION_CURSOR.fetch_add(1, Ordering::Relaxed);
        prop_assert!(
            case + 1 < ANCHOR_DECISION_CASES || covered == 0b11,
            "the run must have exercised both an anchor decided by its median Sharpness_Score \
             and one decided by the absolute path tie-break, branch mask {covered:#b}"
        );

        // 2. The wiring: the anchor of a real Capture_Station and its transform.
        //
        //    Rotated, see `ANCHOR_STATION_STRIDE`.  A `prop_assume!` would be
        //    wrong here: a rejected case does not count as a success, so
        //    `proptest` would simply draw until 100 cases *did* run this half.
        if !case.is_multiple_of(ANCHOR_STATION_STRIDE) {
            return Ok(());
        }
        let images = determinism_images(&scan);
        let locked = scan.locked_homographies();
        let (matches, _measured) = fit_synthetic_matches(&scan, &images);
        let solve = determinism_test_access::solve_station_poses_with_report(
            &images,
            &matches,
            &locked,
            scan.geometric_reference_index(),
        );
        let identities = scan.capture_identities();
        let geometry = synthetic_geometry_outcome(&images, &identities, &solve);
        let grouping_poses = match &geometry {
            SyntheticGeometryOutcome::Connected => &solve.poses,
            SyntheticGeometryOutcome::RejectedDisconnected { .. } => &locked,
        };
        let stations =
            determinism_test_access::capture_stations(&images, &matches, grouping_poses)
                .unwrap_or_default();
        if matches!(geometry, SyntheticGeometryOutcome::RejectedDisconnected { .. }) {
            let permuted = scan.with_permuted_import_order();
            let permuted_images = determinism_images(&permuted);
            let permuted_locked = permuted.locked_homographies();
            let (permuted_matches, _) = fit_synthetic_matches(&permuted, &permuted_images);
            let permuted_solve = determinism_test_access::solve_station_poses_with_report(
                &permuted_images,
                &permuted_matches,
                &permuted_locked,
                permuted.geometric_reference_index(),
            );
            let permuted_geometry = synthetic_geometry_outcome(
                &permuted_images,
                &permuted.capture_identities(),
                &permuted_solve,
            );
            prop_assert_eq!(
                &geometry,
                &permuted_geometry,
                "a disconnected anchor fixture must retain the same explicit rejection report under import permutation"
            );
        }
        prop_assert!(
            stations.iter().any(|station| station.members.len() >= 2),
            "the anchor decision is only under test on a station with several frames"
        );
        let mut multi_member_stations = 0usize;
        for station in &stations {
            let station_candidates = station
                .members
                .iter()
                .map(|&member| intra_station::AnchorCandidate {
                    path: images[member].filename.clone(),
                    median_sharpness: grouping_test_access::frame_median_sharpness(&images[member]),
                })
                .collect::<Vec<_>>();
            let expected = expected_anchor(&station_candidates)
                .map(|position| station.members[position])
                .expect("a station has at least one member");
            prop_assert_eq!(
                station.anchor,
                expected,
                "the Capture_Station anchor must be its highest median Sharpness_Score frame \
                 (candidates {:?})",
                station_candidates
            );
            // 锚点帧到机位坐标系的变换是恒等矩阵.
            let into_anchor = station
                .local_to_anchor
                .get(&station.anchor)
                .copied()
                .expect("the anchor carries a transform into its own station frame");
            // `anchor_global.try_inverse() * anchor_global` is a floating point
            // product, so the identity it produces is the identity to within a
            // few ulps rather than bit for bit — the measured worst coefficient
            // over the fixture is 1.4e-14 on the perspective row.  Both halves of
            // "is the identity" are therefore asserted: the normalised
            // coefficients, and the only thing a transform is for, namely that
            // every frame corner maps to itself to far below a pixel.
            let normalised = into_anchor / into_anchor[(2, 2)];
            for (index, (value, expected)) in normalised
                .iter()
                .zip(Matrix3::<f64>::identity().iter())
                .enumerate()
            {
                prop_assert!(
                    (value - expected).abs() <= 1e-12,
                    "coefficient {index} of the anchor transform is {value}, not {expected}"
                );
            }
            let width = f64::from(images[station.anchor].width);
            let height = f64::from(images[station.anchor].height);
            for corner in [
                Point2::new(0.0, 0.0),
                Point2::new(width, 0.0),
                Point2::new(0.0, height),
                Point2::new(width, height),
            ] {
                let mapped = map_point(&into_anchor, corner)
                    .expect("the anchor transform is invertible");
                prop_assert!(
                    (mapped - corner).norm() <= 1e-9,
                    "the anchor frame corner {corner:?} maps to {mapped:?}"
                );
            }
            multi_member_stations += usize::from(station.members.len() >= 2);
        }
        // A bracket whose accumulated drift split every frame into its own
        // Capture_Station is a legitimate 需求 1.4 outcome, and then there is no
        // anchor decision to make; the run as a whole still has to reach one.
        let stations_covered = ANCHOR_MULTI_MEMBER_SEEN
            .fetch_or(usize::from(multi_member_stations >= 1), Ordering::Relaxed)
            | usize::from(multi_member_stations >= 1);
        prop_assert!(
            case + ANCHOR_STATION_STRIDE < ANCHOR_DECISION_CASES || stations_covered == 1,
            "the run must have reached a Capture_Station of several frames, so that 需求 2.1 \
             really had an anchor to choose"
        );
    }
}
