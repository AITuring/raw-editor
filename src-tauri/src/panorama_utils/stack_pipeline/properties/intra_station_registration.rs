//! Intra_Station_Registrar properties (6, 7, 8).
//!
//! The three control point gates of 需求 2.4-2.6 against an oracle rebuilt from
//! the measurements, and the per-frame verdicts of a finished native refinement:
//! the revert of 需求 2.7 and the inlier area coverage of 需求 2.8.

use super::super::intra_station;
use super::super::test_support;
use crate::panorama_utils::mosaic;
use proptest::prelude::*;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// ---------------------------------------------------------------------------
// Property 6 harness: the intra-station control point gates
// ---------------------------------------------------------------------------
//
// The gates of 需求 2.4, 2.5 and 2.6 are applied in two places that cannot be
// separated: `measure_native_control_point` applies the two that depend on the
// control point alone, and `collect_native_observations` applies the
// neighbourhood one while it walks the grid row major.  Property 6 therefore
// drives both — through `mosaic::intra_station_test_access` — and rebuilds the
// accepted set from the *measurements* with the requirement's own three
// thresholds, its own row-major walk and its own median.  Set equality between
// the two is the necessity and the sufficiency of the acceptance condition at
// once: a control point production accepted that the oracle rejects is a gate
// that does not bind, and one the oracle accepts that production rejects is a
// gate that binds harder than the requirement allows.
//
// One thing the oracle cannot derive is whether a *located* match exists at all:
// the matcher's own similarity gates (`INTRA_STATION_MIN_CORRELATION = 0.70`,
// `INTRA_STATION_MIN_PEAK_MARGIN = 0.006`, the variance floors) decide that, and a
// control point without a bidirectional match has no displacement to judge.  That
// is a measurable precondition of the three thresholds rather than a fourth
// threshold, so the oracle treats "no bidirectional measurement" as a rejection
// and the property additionally asserts that such control points contribute
// nothing to the displacement field.

/// 需求 2.4: bidirectional round trip tolerance, in native pixels.
pub(super) const REQUIRED_BIDIRECTIONAL_TOLERANCE_PX: f64 = 1.0;

/// 需求 2.5: neighbourhood median tolerance, in native pixels.
pub(super) const REQUIRED_NEIGHBOUR_TOLERANCE_PX: f64 = 4.0;

/// 需求 2.6: symmetric reprojection error ceiling, as a fraction of the anchor
/// frame's native long side.
pub(super) const REQUIRED_SYMMETRIC_ERROR_RATIO: f64 = 0.01;

/// 需求 2.3: smallest matching window side, in native pixels.
pub(super) const REQUIRED_MIN_WINDOW_SIDE_PX: i32 = 32;

/// 需求 2.3: largest search reach, in native pixels.
pub(super) const REQUIRED_MAX_SEARCH_REACH_PX: f64 = 64.0;

/// Why the oracle rejected one control point, so the property can assert that
/// every gate really rejected something over the run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ControlPointVerdict {
    Accepted,
    /// The patch pair was incomplete, so no match was attempted.
    NotEvaluated,
    /// No bidirectional match: the matcher's similarity gates stopped it.
    NotLocated,
    /// 需求 2.4.
    RoundTrip,
    /// 需求 2.6.
    SymmetricError,
    /// 需求 2.5.
    NeighbourMedian,
}

impl ControlPointVerdict {
    pub(super) fn bit(self) -> usize {
        1 << self as usize
    }
}

/// Per-axis median of the accepted 8-neighbourhood, computed independently of
/// `ControlPointGrid`.
///
/// The lower of the two middle samples for an even count, which is what
/// "中位数" means for a sample of an even size once the answer has to be
/// deterministic.
pub(super) fn oracle_neighbour_median(
    accepted: &HashMap<(usize, usize), [f64; 2]>,
    column: usize,
    row: usize,
) -> Option<[f64; 2]> {
    let mut xs = Vec::with_capacity(8);
    let mut ys = Vec::with_capacity(8);
    for row_offset in -1i64..=1 {
        for column_offset in -1i64..=1 {
            if row_offset == 0 && column_offset == 0 {
                continue;
            }
            let neighbour_row = usize::try_from(row as i64 + row_offset);
            let neighbour_column = usize::try_from(column as i64 + column_offset);
            let (Ok(neighbour_row), Ok(neighbour_column)) = (neighbour_row, neighbour_column)
            else {
                continue;
            };
            if let Some(displacement) = accepted.get(&(neighbour_column, neighbour_row)) {
                xs.push(displacement[0]);
                ys.push(displacement[1]);
            }
        }
    }
    if xs.is_empty() {
        return None;
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(|left, right| left.total_cmp(right));
        values[(values.len() - 1) / 2]
    };
    Some([median(&mut xs), median(&mut ys)])
}

/// The acceptance condition of 需求 2.4 / 2.5 / 2.6, applied to the measurements
/// of one pass from scratch.
///
/// Returns the verdict of every control point in the grid's row-major order,
/// which is the order the requirement's "已接受控制点" makes the neighbourhood
/// gate depend on.
pub(super) fn oracle_control_point_verdicts(
    probes: &[mosaic::intra_station_test_access::ControlPointProbe],
    anchor_long_side: u32,
) -> HashMap<(usize, usize), ControlPointVerdict> {
    let error_limit = f64::from(anchor_long_side) * REQUIRED_SYMMETRIC_ERROR_RATIO;
    let mut accepted = HashMap::new();
    let mut verdicts = HashMap::new();
    let mut ordered = probes.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|probe| (probe.row, probe.column));
    for probe in ordered {
        let key = (probe.column, probe.row);
        let verdict = if !probe.evaluated {
            ControlPointVerdict::NotEvaluated
        } else if let Some(displacement) = probe.displacement_px {
            if !(probe.round_trip_px.is_finite()
                && probe.round_trip_px <= REQUIRED_BIDIRECTIONAL_TOLERANCE_PX)
            {
                ControlPointVerdict::RoundTrip
            } else if !(probe.symmetric_error_px.is_finite()
                && probe.symmetric_error_px <= error_limit)
            {
                ControlPointVerdict::SymmetricError
            } else if oracle_neighbour_median(&accepted, probe.column, probe.row).is_some_and(
                |median| {
                    (displacement[0] - median[0]).hypot(displacement[1] - median[1])
                        > REQUIRED_NEIGHBOUR_TOLERANCE_PX
                },
            ) {
                ControlPointVerdict::NeighbourMedian
            } else {
                accepted.insert(key, displacement);
                ControlPointVerdict::Accepted
            }
        } else {
            ControlPointVerdict::NotLocated
        };
        verdicts.insert(key, verdict);
    }
    verdicts
}

/// Bitmask of the [`ControlPointVerdict`]s the run has produced.
pub(super) static CONTROL_POINT_VERDICTS_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 6 so far, which also drives the rotation of the
/// whole-frame half.
pub(super) static CONTROL_POINT_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 6 is configured with.
///
/// One case measures 25 control points, each of which correlates two 51x51
/// windows over 225 integer hypotheses and runs the focal-plane matched low-pass,
/// so a case costs about a third of a second — the whole suite has a budget, and
/// 24 independently drawn pairs cover the gate combinations (the coverage mask
/// below turns that claim into an assertion) at a tenth of the cost of 240.
pub(super) const CONTROL_POINT_CASES: usize = 20;

/// Every how many cases the whole-frame half of Property 6 runs.
pub(super) const CONTROL_POINT_FRAME_STRIDE: usize = 3;

proptest! {
    #![proptest_config(ProptestConfig { cases: CONTROL_POINT_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 6: 对于任意局部匹配控制点，
    // 该控制点的局部位移被接受当且仅当同时满足：正向与反向位移之和的模长 ≤1.0 个原生像素、
    // 与 8 邻域已接受控制点位移中位数之差 ≤4.0 个原生像素、对称重投影误差 ≤0.01×锚点帧
    // 长边；被拒绝的控制点位置的位移等于全局模型在该位置的位移。
    //
    // 充要性用集合相等断言：生产接受集合 == 用需求原文的三个阈值（1.0 / 4.0 / 0.01×长边，
    // 在测试里按需求写死）、按行优先顺序、用独立实现的中位数从同一批**实测量**重算出的
    // 集合。多接受一个点说明某道门没有起作用，多拒绝一个点说明某道门比需求更严。
    //
    // 哪些判定**能**被触及是被 patch 几何决定的，不是测试的选择，实测如下：
    //
    //   * `probe_warped_patch` 在打分之前要预留整个搜索支撑，而反向探测是以正向匹配点为
    //     中心的，所以它要求
    //     `MATCH_RADIUS(25) + SEARCH_SAMPLES(7) + |d| < PATCH_HALF(33)`，即 `|d|` 必须
    //     小于 1 个 patch 采样（细遍 = 1.0 原生像素，粗遍 = 2.0 原生像素）。超过这个位移的
    //     控制点根本形不成双向测量，会以 `NotLocated` 结束——生产里这没问题，因为细遍面对的
    //     是粗遍改正后的残差。实测本 fixture 上 `|d|` 最大 1.20 原生像素（含亚像素细化的
    //     过冲）。
    //   * 因此需求 2.5 的 4.0 原生像素邻域容差**无法拒绝任何控制点**：同一遍里两个被接受的
    //     控制点各自的位移都小于 1 个采样，其差恒小于 2×spacing ≤ 4.0。这是对生产实现的
    //     一条实测结论，不是本属性的断言范围——集合相等仍然把这道门算在内，只是数据永远不会
    //     落到它拒绝的那一侧。
    //   * 需求 2.6 的上限是 0.01×锚点帧长边，真实 9504px 锚点帧上是 95 原生像素，比可测位移
    //     大两个数量级，所以在真实尺寸下同样无法拒绝。生成器额外抽一个人为很小的锚点长边
    //     （64 → 0.64 原生像素上限）来证明这道门确实接线正确并会拒绝，运行结束时断言它出现过。
    //   * 需求 2.4 的 1.0 原生像素往返容差在原理上可达（反向偏移可到 ±7 个采样），但实测
    //     闭合的往返最大只有 0.18 原生像素：真正在起作用的是同一条约束的"测不出来"那一半。
    //
    // 运行结束时断言"可达的"三种判定（接受、相似度门拒绝、需求 2.6 拒绝）都出现过。
    //
    // 需求 2.7 的 `Indeterminate` 结论在整帧那一半断言：一层与底图无关时没有任何控制点被
    // 接受，两个中位数都无法形成，此时既不能回退也不能假装比较过——位移场必须逐位等于
    // 进入时的全局模型，这同时就是"被拒绝的控制点位置保留全局模型结果"的最强形式。
    //
    // **Validates: Requirements 2.4, 2.5, 2.6**
    #[test]
    fn local_match_acceptance_is_exactly_the_three_requirement_gates(
        pair in test_support::arb_intra_station_layer_pair(),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        // The measured constants of task 7.5 are the requirement's numbers, and
        // the patch geometry they sit on is the one 需求 2.3 allows.
        prop_assert_eq!(
            intra_station::INTRA_STATION_BIDIRECTIONAL_TOLERANCE_PX,
            REQUIRED_BIDIRECTIONAL_TOLERANCE_PX
        );
        prop_assert_eq!(
            intra_station::INTRA_STATION_NEIGHBOUR_MEDIAN_TOLERANCE_PX,
            REQUIRED_NEIGHBOUR_TOLERANCE_PX
        );
        prop_assert_eq!(
            intra_station::INTRA_STATION_SYMMETRIC_ERROR_RATIO,
            REQUIRED_SYMMETRIC_ERROR_RATIO
        );
        prop_assert_eq!(intra_station::INTRA_STATION_MIN_CORRELATION, 0.70);
        prop_assert_eq!(intra_station::INTRA_STATION_MATCH_RADIUS, 25);
        prop_assert_eq!(intra_station::INTRA_STATION_PATCH_HALF, 33);
        prop_assert!(
            2 * intra_station::INTRA_STATION_MATCH_RADIUS + 1 >= REQUIRED_MIN_WINDOW_SIDE_PX,
            "需求 2.3 asks for a matching window of at least {REQUIRED_MIN_WINDOW_SIDE_PX} \
             native pixels"
        );
        prop_assert!(
            intra_station::native_search_reach_px(1.0) <= REQUIRED_MAX_SEARCH_REACH_PX,
            "需求 2.3 bounds the search reach at {REQUIRED_MAX_SEARCH_REACH_PX} native pixels"
        );
        prop_assert!(
            intra_station::patch_support_fits(
                intra_station::INTRA_STATION_MATCH_RADIUS,
                intra_station::INTRA_STATION_SEARCH_SAMPLES,
                intra_station::INTRA_STATION_PATCH_HALF,
            ),
            "the window and its whole search support must fit the patch buffer, or every control \
             point is silently unmatchable"
        );

        // One pass at the finest spacing, which is where the window is 51 native
        // pixels and the reach 32.  Both passes apply the same three gates; the
        // coarse one needs a frame several hundred pixels wide to fit its patch
        // support, and the gates do not know which pass they are in.
        let (probes, observations, evaluated) =
            mosaic::intra_station_test_access::probe_native_pass(
                &pair.base,
                &pair.layer,
                pair.anchor_long_side,
                1.0,
            );
        prop_assert!(
            evaluated >= 1,
            "the fixture must reach at least one measurable control point"
        );

        // 充要性.
        let expected = oracle_control_point_verdicts(&probes, pair.anchor_long_side);
        let mut seen = 0usize;
        for probe in &probes {
            let verdict = expected[&(probe.column, probe.row)];
            seen |= verdict.bit();
            match verdict {
                ControlPointVerdict::Accepted => {
                    let accepted = probe.accepted_px.ok_or_else(|| TestCaseError::fail(format!(
                        "control point ({}, {}) satisfies all three gates (round trip {:.3}px, \
                         symmetric error {:.3}px of {:.3}px, displacement {:?}) but was rejected",
                        probe.column,
                        probe.row,
                        probe.round_trip_px,
                        probe.symmetric_error_px,
                        f64::from(pair.anchor_long_side) * REQUIRED_SYMMETRIC_ERROR_RATIO,
                        probe.displacement_px,
                    )))?;
                    let measured = probe.displacement_px.expect("an accepted point was located");
                    prop_assert_eq!(
                        accepted[0].to_bits(),
                        measured[0].to_bits(),
                        "the accepted displacement must be the measured one"
                    );
                    prop_assert_eq!(accepted[1].to_bits(), measured[1].to_bits());
                    // 需求 2.8 measures its coverage over exactly the accepted
                    // control point cells, so an accepted cell has to be one
                    // whose centre lies on the anchor frame's valid pixels.
                    prop_assert!(
                        probe.covered,
                        "control point ({}, {}) was accepted although its cell centre is not on \
                         covered base",
                        probe.column,
                        probe.row
                    );
                }
                rejected => {
                    prop_assert!(
                        probe.accepted_px.is_none(),
                        "control point ({}, {}) was accepted although it is rejected by \
                         {:?} (round trip {:.3}px, symmetric error {:.3}px of {:.3}px, \
                         correlation {:?})",
                        probe.column,
                        probe.row,
                        rejected,
                        probe.round_trip_px,
                        probe.symmetric_error_px,
                        f64::from(pair.anchor_long_side) * REQUIRED_SYMMETRIC_ERROR_RATIO,
                        probe.best_correlation
                    );
                }
            }
        }

        // 被拒绝的控制点位置的位移等于全局模型在该位置的位移: only the accepted
        // control points enter the displacement field at all, so a rejected one
        // contributes nothing and its position keeps what the global model put
        // there.
        let accepted_positions = probes
            .iter()
            .filter(|probe| probe.accepted_px.is_some())
            .map(|probe| (probe.column, probe.row))
            .collect::<BTreeSet<_>>();
        prop_assert_eq!(
            observations.iter().copied().collect::<BTreeSet<_>>(),
            accepted_positions.clone(),
            "the displacement field must be built from exactly the accepted control points"
        );

        // The whole-frame half, rotated: 需求 2.7's Indeterminate outcome and the
        // strongest form of "a rejected control point keeps the global model".
        let case = CONTROL_POINT_CURSOR.fetch_add(1, Ordering::Relaxed);
        let covered = CONTROL_POINT_VERDICTS_SEEN.fetch_or(seen, Ordering::Relaxed) | seen;
        if pair.unrelated || case.is_multiple_of(CONTROL_POINT_FRAME_STRIDE) {
            let frame = mosaic::intra_station_test_access::probe_native_refinement(
                &pair.base,
                &pair.layer,
                pair.anchor_long_side,
            );
            if pair.unrelated {
                prop_assert_eq!(
                    frame.accepted_control_points,
                    0,
                    "no control point of an unrelated layer may be accepted"
                );
                prop_assert_eq!(frame.local_baseline_samples, 0);
                prop_assert_eq!(
                    frame.global_baseline_samples,
                    0,
                    "需求 2.7's global baseline is formed from accepted control points too"
                );
                prop_assert_eq!(
                    frame.local_field_verdict,
                    intra_station::LocalFieldVerdict::Indeterminate,
                    "需求 2.7 cannot be evaluated without both medians, and an unevaluable \
                     comparison is neither a keep nor a revert"
                );
                prop_assert!(
                    !frame.reverted_to_global,
                    "an absent baseline must not be reported as a revert"
                );
                prop_assert!(
                    frame.field_unchanged,
                    "with no accepted control point every position must keep the global model's \
                     displacement, bit for bit"
                );
                prop_assert_eq!(
                    frame.status,
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed,
                    "需求 2.8 fails a frame whose accepted control point cells cover nothing \
                     (coverage {})",
                    frame.inlier_area_coverage
                );
            } else {
                prop_assert!(
                    frame.evaluated_control_points >= 1,
                    "the whole-frame refinement must reach the control point grid"
                );
                // A displacement field that moved must have been moved by
                // accepted control points, and enough of them to form one.
                prop_assert!(
                    frame.field_unchanged
                        || frame.accepted_control_points
                            >= intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES,
                    "the displacement field moved on {} accepted control point(s), below the \
                     {} a field needs",
                    frame.accepted_control_points,
                    intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES
                );
                prop_assert!(
                    frame.accepted_control_points > 0 || frame.field_unchanged,
                    "with no accepted control point every position must keep the global model's \
                     displacement"
                );
            }
        }

        // The reachable verdicts must all have occurred, or the set equality
        // above could hold for data that never reached a gate.  Which verdicts
        // are reachable at all is a measured property of the patch geometry
        // rather than a choice; see the note on the test.
        let required = ControlPointVerdict::Accepted.bit()
            | ControlPointVerdict::NotLocated.bit()
            | ControlPointVerdict::SymmetricError.bit();
        prop_assert!(
            case + 1 < CONTROL_POINT_CASES || covered & required == required,
            "the run must have produced an accepted control point, a rejection by the matcher's \
             similarity gates and a rejection by 需求 2.6; verdict mask {covered:#b}, required \
             {required:#b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties 7 and 8 harness: the per-frame verdict of a native refinement
// ---------------------------------------------------------------------------
//
// Property 6 above drives the *control point* gates.  These two drive the two
// decisions a finished refinement makes about the frame as a whole, and both are
// taken inside `finalize_native_refinement` / `apply_native_pass`, so both are
// driven through `mosaic::intra_station_test_access::probe_native_refinement`,
// which runs the real `refine_native_layer` over both spacing passes.
//
// The fixture is `arb_intra_station_verdict_pair`, not Property 6's
// `arb_intra_station_layer_pair`: a layer that is always displaced is a layer the
// local refinement can always improve, so on Property 6's fixture 需求 2.7's
// comparison only ever comes out one way and an equivalence asserted on it would
// be half unexercised.  One draw in three of the verdict fixture is an *already
// aligned* layer, where the global model is right and the refinement has nothing
// left to remove.  Measured over 40 draws: 10 Reverted (including 3 exact ties
// between the two medians, which is what tells 需求 2.7's "not lower than" apart
// from "higher than"), 9 Kept, 21 Indeterminate.

/// 需求 2.7: the smallest sample count either median may be formed from, written
/// out here rather than read from the production constant.
pub(super) const REQUIRED_MIN_BASELINE_SAMPLES: usize = 6;

/// 需求 2.8: the inlier area coverage below which a frame is a registration
/// failure.
pub(super) const REQUIRED_MIN_INLIER_AREA_COVERAGE: f64 = 0.20;

/// Bit 0: a Kept verdict occurred.  Bit 1: a Reverted verdict occurred.  Bit 2:
/// the two medians were exactly equal, i.e. the boundary of 需求 2.7.
pub(super) static LOCAL_FIELD_VERDICTS_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 7 so far.
pub(super) static LOCAL_FIELD_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Properties 7 and 8 are configured with.
///
/// One case is a whole two-pass refinement of a 144x144 pair — 50 control points
/// over the two passes, each correlating two 51x51 windows — measured at 25 ms,
/// so the two properties together add about 5 s to the suite.
pub(super) const REFINEMENT_VERDICT_CASES: usize = 100;

proptest! {
    #![proptest_config(ProptestConfig { cases: REFINEMENT_VERDICT_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 7: 对于任意非锚点帧，
    // 该帧的配准状态为已回退当且仅当其 patch refinement 后的中位对称重投影误差不低于其
    // 全局模型的中位对称重投影误差；回退后该帧不携带任何局部位移场。
    //
    // 「不低于」是两个**中位数**之间的比较，而空样本的中位数不是一个值。需求 2.7 因此有
    // 三种结论，本属性断言其中的两种等价关系：两侧样本数都达到 6 时，
    // `Reverted ⟺ ¬(local < global)`，`Kept ⟺ local < global`。第三种结论
    // （`Indeterminate`）在 Property 6 的整帧那一半断言，本属性只断言它与前两种互斥、
    // 且从不被记为回退。
    //
    // 「回退后该帧不携带任何局部位移场」用位移场与进入 refinement 时的全局模型场逐位
    // 相等来断言（`field_unchanged`），这是该子句可测的最强形式。反方向同样断言：
    // Kept 意味着最后一遍至少有 6 个被接受控制点，也就意味着位移场确实被改动过——
    // 否则「不回退」不会有任何可观察的后果。
    //
    // **Validates: Requirements 2.7**
    #[test]
    fn local_refinement_is_reverted_exactly_when_it_does_not_improve(
        pair in test_support::arb_intra_station_verdict_pair(),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        prop_assert_eq!(
            intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES,
            REQUIRED_MIN_BASELINE_SAMPLES
        );

        let frame = mosaic::intra_station_test_access::probe_native_refinement(
            &pair.base,
            &pair.layer,
            pair.anchor_long_side,
        );

        // 需求 2.7's comparison, rebuilt from the reported medians and their
        // sample counts with the requirement's own wording: both medians have to
        // exist before "not lower than" can be true of them.
        let local = frame.local_median_symmetric_error_px;
        let global = frame.global_median_symmetric_error_px;
        let both_formed = frame.local_baseline_samples >= REQUIRED_MIN_BASELINE_SAMPLES
            && frame.global_baseline_samples >= REQUIRED_MIN_BASELINE_SAMPLES
            && local.is_finite()
            && global.is_finite();
        // 需求 2.7 is written as "not lower than", and the negated comparison is
        // kept literal on purpose: `both_formed` has already ruled out the
        // incomparable case clippy warns about, so `!(a < b)` here is exactly the
        // requirement's condition rather than a paraphrase of it.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let expected_reverted = both_formed && !(local < global);
        let expected_kept = both_formed && local < global;

        prop_assert_eq!(
            frame.local_field_verdict == intra_station::LocalFieldVerdict::Reverted,
            expected_reverted,
            "需求 2.7 reverts a frame exactly when its refined median symmetric reprojection \
             error {:.6}px over {} points is not below the global model's {:.6}px over {} \
             points; the verdict was {:?}",
            local,
            frame.local_baseline_samples,
            global,
            frame.global_baseline_samples,
            frame.local_field_verdict
        );
        prop_assert_eq!(
            frame.local_field_verdict == intra_station::LocalFieldVerdict::Kept,
            expected_kept,
            "需求 2.7 keeps a frame exactly when the refinement lowered its median error \
             ({:.6}px vs {:.6}px over {} / {} points); the verdict was {:?}",
            local,
            global,
            frame.local_baseline_samples,
            frame.global_baseline_samples,
            frame.local_field_verdict
        );
        // The report field and the verdict cannot disagree: it is the report the
        // acceptance harness reads.
        prop_assert_eq!(
            frame.reverted_to_global,
            expected_reverted,
            "the reported revert flag must be the verdict"
        );
        // An unevaluable comparison is the third outcome, never a revert.
        prop_assert_eq!(
            frame.local_field_verdict == intra_station::LocalFieldVerdict::Indeterminate,
            !both_formed
        );
        prop_assert!(!(frame.local_field_verdict == intra_station::LocalFieldVerdict::Indeterminate
            && frame.reverted_to_global));

        if expected_reverted {
            // 回退后该帧不携带任何局部位移场: the field is bit for bit the one the
            // frame entered the refinement with.
            prop_assert!(
                frame.field_unchanged,
                "a reverted frame must carry no local displacement field, but the field differs \
                 from the global model's ({} accepted control points)",
                frame.accepted_control_points
            );
            // 需求 2.9 records the revert as the frame's status, unless 需求 2.8
            // already failed the frame on coverage — that verdict is the stronger
            // one and Property 8 owns it.
            let expected_status =
                if frame.inlier_area_coverage < REQUIRED_MIN_INLIER_AREA_COVERAGE {
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed
                } else {
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::GlobalFallback
                };
            prop_assert_eq!(
                frame.status,
                expected_status,
                "a reverted frame with coverage {} must not be reported as locally registered",
                frame.inlier_area_coverage
            );
        }
        if expected_kept {
            // The other direction of "not reverted": a kept refinement has to be
            // observable, and the only observable it has is the field.  A pass
            // only builds a field from at least six accepted control points, and
            // a formed local median means the last pass had at least that many.
            prop_assert!(
                frame.local_baseline_samples >= REQUIRED_MIN_BASELINE_SAMPLES
                    && !frame.field_unchanged,
                "a kept refinement must leave a displacement field behind ({} local samples, \
                 field unchanged {})",
                frame.local_baseline_samples,
                frame.field_unchanged
            );
        }

        let mut seen = 0usize;
        if expected_kept {
            seen |= 1;
        }
        if expected_reverted {
            seen |= 2;
        }
        if both_formed && local == global {
            seen |= 4;
        }
        let covered = LOCAL_FIELD_VERDICTS_SEEN.fetch_or(seen, Ordering::Relaxed) | seen;
        let case = LOCAL_FIELD_CURSOR.fetch_add(1, Ordering::Relaxed);
        prop_assert!(
            case + 1 < REFINEMENT_VERDICT_CASES || covered == 0b111,
            "the run must have produced a kept refinement, a reverted one and a pair of exactly \
             equal medians — the boundary 需求 2.7's 「不低于」 turns on; mask {covered:#b}"
        );
    }
}

/// Bit 0: the union over passes held more accepted cells than the final pass
/// alone.  Bit 1: a frame failed 需求 2.8.  Bit 2: a frame cleared it.
pub(super) static COVERAGE_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 8 so far.
pub(super) static COVERAGE_CURSOR: AtomicUsize = AtomicUsize::new(0);

proptest! {
    #![proptest_config(ProptestConfig { cases: REFINEMENT_VERDICT_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 8: 对于任意非锚点帧，
    // 其内点空间覆盖率等于含已接受局部匹配的控制点单元面积之和除以锚点帧有效像素面积；
    // 该帧被标记为配准失败当且仅当该覆盖率低于 20%。
    //
    // 定义里的「含已接受局部匹配的控制点单元」是**整个 refinement** 接受过的单元，不是
    // 最后一遍 spacing 接受的单元：两遍接受的匹配都进入同一个位移场，所以一个被粗遍匹配
    // 上的单元无论细遍是否再次匹配上，都含有一个已接受的局部匹配。本属性因此同时观测
    // 两个计数（并集与最后一遍），用并集重算覆盖率并断言生产报告的就是它；实测 40 次抽样
    // 里有 9 次两者不同，运行结束时断言这种情形确实出现过，否则这条区分就是没有被触及的。
    //
    // 需求 2.5 的邻域门**不用**并集，那是单遍判定（粗遍的位移已经写进场里，细遍量的是残差，
    // 两者不可比）。这条区分在下面用 `ControlPointGrid` 自身断言：同一批粗/细网格，
    // `absorb` 得到并集，而 `neighbour_median` 只看被查询那一遍自己的已接受集合。
    //
    // **Validates: Requirements 2.8**
    #[test]
    fn inlier_area_coverage_is_the_union_over_passes_and_decides_the_failure(
        pair in test_support::arb_intra_station_verdict_pair(),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        prop_assert_eq!(
            intra_station::INTRA_STATION_MIN_INLIER_AREA_COVERAGE,
            REQUIRED_MIN_INLIER_AREA_COVERAGE
        );

        let frame = mosaic::intra_station_test_access::probe_native_refinement(
            &pair.base,
            &pair.layer,
            pair.anchor_long_side,
        );

        // 需求 2.8's definition, rebuilt here: accepted control point cell area
        // over the anchor frame's valid pixel area.
        let cell_area = frame.cell_area_px;
        let anchor_area = frame.valid_cells as f64 * cell_area;
        let expected = if anchor_area > 0.0 && cell_area > 0.0 {
            ((frame.accepted_control_points as f64 * cell_area) / anchor_area).clamp(0.0, 1.0)
        } else {
            0.0
        };
        prop_assert!(
            (frame.inlier_area_coverage - expected).abs() < 1e-12,
            "需求 2.8's coverage is {} accepted cells of {:.1}px² over {} valid cells = {:.6}, \
             but {:.6} was reported",
            frame.accepted_control_points,
            cell_area,
            frame.valid_cells,
            expected,
            frame.inlier_area_coverage
        );

        // 该帧被标记为配准失败当且仅当该覆盖率低于 20%.
        prop_assert_eq!(
            frame.status == crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed,
            frame.inlier_area_coverage < REQUIRED_MIN_INLIER_AREA_COVERAGE,
            "需求 2.8 fails a frame exactly below {}; coverage {:.6}, status {:?}",
            REQUIRED_MIN_INLIER_AREA_COVERAGE,
            frame.inlier_area_coverage,
            frame.status
        );

        // The union, not the last pass.  The two counts can only differ in one
        // direction, and where they differ the reported coverage has to follow
        // the larger one.
        prop_assert!(
            frame.accepted_control_points >= frame.last_pass_accepted_control_points,
            "the union over passes cannot hold fewer cells than one pass ({} vs {})",
            frame.accepted_control_points,
            frame.last_pass_accepted_control_points
        );
        let mut branches = 0usize;
        if frame.accepted_control_points > frame.last_pass_accepted_control_points {
            branches |= 1;
            let last_pass_only = ((frame.last_pass_accepted_control_points as f64 * cell_area)
                / anchor_area.max(f64::MIN_POSITIVE))
                .clamp(0.0, 1.0);
            prop_assert!(
                frame.inlier_area_coverage > last_pass_only,
                "需求 2.8 counts every cell the refinement accepted a local match in, so the \
                 coverage must exceed the final pass's own {last_pass_only:.6}, not equal it"
            );
        }
        branches |= if frame.status == crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed {
            2
        } else {
            4
        };

        // 需求 2.5's per-pass grid versus 需求 2.8's union, on the grid type that
        // holds both: `absorb` unions the passes, while the neighbourhood median
        // a pass consults sees that pass's accepted set only.
        {
            let mut coarse = intra_station::ControlPointGrid::new(5, 5);
            coarse.accept(0, 0, [3.0, 0.0]);
            coarse.accept(1, 0, [3.0, 0.0]);
            let mut fine = intra_station::ControlPointGrid::new(5, 5);
            // Outside the 8-neighbourhood of the cell queried below, so the two
            // passes really are asked about different neighbourhoods.
            fine.accept(4, 4, [0.5, 0.0]);
            let mut union = intra_station::ControlPointGrid::new(5, 5);
            union.absorb(&coarse);
            union.absorb(&fine);
            prop_assert_eq!(union.accepted_count(), 3, "需求 2.8 counts both passes");
            prop_assert_eq!(
                fine.neighbour_median(1, 1),
                None,
                "需求 2.5 compares a control point against its own pass's accepted \
                 neighbourhood; a coarse-pass displacement has already been applied to the \
                 field and is not comparable with the residual the fine pass measures"
            );
            prop_assert_eq!(union.neighbour_median(1, 1), Some([3.0, 0.0]));
        }

        let covered = COVERAGE_BRANCHES_SEEN.fetch_or(branches, Ordering::Relaxed) | branches;
        let case = COVERAGE_CURSOR.fetch_add(1, Ordering::Relaxed);
        prop_assert!(
            case + 1 < REFINEMENT_VERDICT_CASES || covered == 0b111,
            "the run must have produced a frame whose accepted cells accumulate over the two \
             spacing passes, a frame 需求 2.8 fails and a frame it clears; mask {covered:#b}"
        );
    }
}
