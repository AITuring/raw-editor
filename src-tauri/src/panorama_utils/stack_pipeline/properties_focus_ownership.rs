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
const REQUIRED_BIDIRECTIONAL_TOLERANCE_PX: f64 = 1.0;

/// 需求 2.5: neighbourhood median tolerance, in native pixels.
const REQUIRED_NEIGHBOUR_TOLERANCE_PX: f64 = 4.0;

/// 需求 2.6: symmetric reprojection error ceiling, as a fraction of the anchor
/// frame's native long side.
const REQUIRED_SYMMETRIC_ERROR_RATIO: f64 = 0.01;

/// 需求 2.3: smallest matching window side, in native pixels.
const REQUIRED_MIN_WINDOW_SIDE_PX: i32 = 32;

/// 需求 2.3: largest search reach, in native pixels.
const REQUIRED_MAX_SEARCH_REACH_PX: f64 = 64.0;

/// Why the oracle rejected one control point, so the property can assert that
/// every gate really rejected something over the run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlPointVerdict {
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
    fn bit(self) -> usize {
        1 << self as usize
    }
}

/// Per-axis median of the accepted 8-neighbourhood, computed independently of
/// `ControlPointGrid`.
///
/// The lower of the two middle samples for an even count, which is what
/// "中位数" means for a sample of an even size once the answer has to be
/// deterministic.
fn oracle_neighbour_median(
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
fn oracle_control_point_verdicts(
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
static CONTROL_POINT_VERDICTS_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 6 so far, which also drives the rotation of the
/// whole-frame half.
static CONTROL_POINT_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 6 is configured with.
///
/// One case measures 25 control points, each of which correlates two 51x51
/// windows over 225 integer hypotheses and runs the focal-plane matched low-pass,
/// so a case costs about a third of a second — the whole suite has a budget, and
/// 24 independently drawn pairs cover the gate combinations (the coverage mask
/// below turns that claim into an assertion) at a tenth of the cost of 240.
const CONTROL_POINT_CASES: usize = 20;

/// Every how many cases the whole-frame half of Property 6 runs.
const CONTROL_POINT_FRAME_STRIDE: usize = 3;

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
                    super::report::IntraStationFrameStatus::Failed,
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
const REQUIRED_MIN_BASELINE_SAMPLES: usize = 6;

/// 需求 2.8: the inlier area coverage below which a frame is a registration
/// failure.
const REQUIRED_MIN_INLIER_AREA_COVERAGE: f64 = 0.20;

/// Bit 0: a Kept verdict occurred.  Bit 1: a Reverted verdict occurred.  Bit 2:
/// the two medians were exactly equal, i.e. the boundary of 需求 2.7.
static LOCAL_FIELD_VERDICTS_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 7 so far.
static LOCAL_FIELD_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Properties 7 and 8 are configured with.
///
/// One case is a whole two-pass refinement of a 144x144 pair — 50 control points
/// over the two passes, each correlating two 51x51 windows — measured at 25 ms,
/// so the two properties together add about 5 s to the suite.
const REFINEMENT_VERDICT_CASES: usize = 100;

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
                    super::report::IntraStationFrameStatus::Failed
                } else {
                    super::report::IntraStationFrameStatus::GlobalFallback
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
static COVERAGE_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 8 so far.
static COVERAGE_CURSOR: AtomicUsize = AtomicUsize::new(0);

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
            frame.status == super::report::IntraStationFrameStatus::Failed,
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
        branches |= if frame.status == super::report::IntraStationFrameStatus::Failed {
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

// ---------------------------------------------------------------------------
// Properties 9, 10 and 11 harness: the Focus_Fuser ownership grid
// ---------------------------------------------------------------------------
//
// 需求 3.1, 3.2 and 3.3 are decided in `stack_pipeline::focus_fuser` and consumed
// by `stitching::measure_focus_cells` / `measure_focus_cell_disagreement`, which
// are the *default* path (one Virtual_Tile per Capture_Station, so
// `mosaic::detail_preserving_mosaic` is never reached — see the module header of
// `focus_fuser`).  Properties 10 and 11 therefore drive the production
// measurement itself, through `stitching::focus_fusion_test_access`, and only the
// pure geometry of Property 9 is a function call.

/// 需求 3.1: the ownership cell side bounds, in native pixels.
const REQUIRED_CELL_MIN_PX: u32 = 8;
const REQUIRED_CELL_MAX_PX: u32 = 64;

/// 需求 3.1: the smallest number of ownership cells along the station plane's
/// long side.
const REQUIRED_CELLS_ALONG_LONG_SIDE: u32 = 512;

/// 需求 3.2: the Sharpness_Score window side, in native pixels.
const REQUIRED_SHARPNESS_WINDOW_PX: f64 = 32.0;

/// 需求 3.2: probe positions per ownership cell — the centre and the four
/// corners.
const REQUIRED_PROBES_PER_CELL: usize = 5;

/// 需求 3.3: the inconsistency above which a sharper candidate is penalised.
const REQUIRED_DISAGREEMENT_VETO: f64 = 0.2;

/// 需求 3.3: the smallest penalty such a candidate carries.
const REQUIRED_MIN_MISMATCH_PENALTY: f64 = 1.0;

/// Station plane sides, in native pixels.
///
/// The uniform range covers both sides of the 4,096px crossover the biconditional
/// below turns on, and the selected values pin it exactly, together with the real
/// 9,504px plane of the acceptance material and the 32,768px point where the 64px
/// ceiling starts to bind.
fn arb_station_plane_side() -> impl Strategy<Value = u32> {
    prop_oneof![
        3 => 64u32..=65_536,
        2 => prop::sample::select(vec![
            64u32, 512, 4_095, 4_096, 4_097, 6_000, 8_256, 9_504, 32_767, 32_768, 32_769, 65_536,
        ]),
        // The long side of a plane is the larger of two draws, so without a
        // branch that is small on purpose a plane *below* the crossover is a
        // second-order event and the assertion that both sides of it occur would
        // itself be flaky.
        1 => 64u32..=4_088,
    ]
}

/// Bit 0: a plane whose long side carries at least 512 cells.  Bit 1: a plane
/// where 需求 3.1's two clauses cannot both hold.  Bit 2: a plane where the cell
/// side hit the 64px ceiling.  Bit 3: a plane where rounding *up* would fall
/// below the 512 cell floor.
static CELL_SIZE_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 9 so far.
static CELL_SIZE_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 9 is configured with.  Pure integer geometry, so the cases are
/// free.
const CELL_SIZE_CASES: usize = 256;

proptest! {
    #![proptest_config(ProptestConfig { cases: CELL_SIZE_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 9: 对于任意机位合成平面
    // 尺寸，ownership 单元边长落在 8 至 64 个原生像素之间，且合成平面长边的单元数不少于
    // 512（单元边长已达到 64 个原生像素上限时除外）。
    //
    // 第一条对任意平面尺寸都成立并如此断言。第二条**不能**对任意尺寸成立，这是实测结论而
    // 不是实现缺陷：单元边长的下界是 8，最后一列/行允许越过平面边界但仍然要有 owner，
    // 所以长边上放得下 512 个单元的充要条件是长边不小于 `8 × 511 + 1 = 4089` 个原生像素。
    // 本属性因此断言这条**充要**关系（比设计文档写的单向命题更强），它同时把 4089 这个
    // 分界点钉住：低于它时需求 3.1 的两条子句互相矛盾，任何实现都只能满足其中一条，
    // 而单元边长仍然合法。
    //
    // 设计文档写的「单元边长已达到 64 上限时除外」实测是空的：只有长边 ≥ 32768 才会撞到
    // 64 上限，而那时 `长边 / 64 ≥ 512` 仍然成立。断言里因此不给这个例外留后门。
    //
    // 单元边长取 `floor` 而不是 `ceil`：`ceil` 在 9504px 长边上给 19px / 501 个单元，
    // 低于 512 下界。这条区分用两句断言钉住——`floor` 的单元数恒不少于 `ceil` 的
    // （单调性），且运行结束时断言确实出现过 `ceil` 会掉到 512 以下的平面。
    //
    // **Validates: Requirements 3.1**
    #[test]
    fn ownership_cell_size_and_count_satisfy_the_requirement(
        width in arb_station_plane_side(),
        height in arb_station_plane_side(),
    ) {
        let geometry = focus_fuser::OwnershipGridGeometry::for_plane(width, height);
        let cell = geometry.cell_size_px;
        let long_side = width.max(height);

        // 单元边长落在 8 至 64 个原生像素之间 — for every plane size.
        prop_assert!(
            (REQUIRED_CELL_MIN_PX..=REQUIRED_CELL_MAX_PX).contains(&cell),
            "ownership cell side {cell}px is outside 需求 3.1's [{REQUIRED_CELL_MIN_PX}, \
             {REQUIRED_CELL_MAX_PX}] on a {width}x{height} plane"
        );
        // The cells are square and cover the whole plane, hanging over its edge
        // at most by less than one cell.
        prop_assert_eq!(geometry.columns, width.div_ceil(cell));
        prop_assert_eq!(geometry.rows, height.div_ceil(cell));

        let cells_along_long_side = geometry.columns.max(geometry.rows);
        prop_assert_eq!(cells_along_long_side, long_side.div_ceil(cell));

        // 合成平面长边的单元数不少于 512, exactly when 512 cells of the minimum
        // legal side fit on the plane at all.
        //
        // The last cell may hang over the plane edge and still has to be owned,
        // so 512 cells fit as soon as 511 whole cells of the 8px minimum plus one
        // more pixel do: `8 × 511 + 1 = 4089` native pixels.
        let minimum_long_side_for_floor =
            REQUIRED_CELL_MIN_PX * (REQUIRED_CELLS_ALONG_LONG_SIDE - 1) + 1;
        let reachable = long_side >= minimum_long_side_for_floor;
        prop_assert_eq!(
            cells_along_long_side >= REQUIRED_CELLS_ALONG_LONG_SIDE,
            reachable,
            "a {}px long side carries {} cells of {}px; 需求 3.1's 512 cell floor is satisfiable \
             exactly from {}px up",
            long_side,
            cells_along_long_side,
            cell,
            minimum_long_side_for_floor
        );
        if cell == REQUIRED_CELL_MAX_PX {
            prop_assert!(
                cells_along_long_side >= REQUIRED_CELLS_ALONG_LONG_SIDE,
                "the design document's 「64px 上限除外」 exception is never needed: a plane whose \
                 cells hit the ceiling is at least {}px long and still carries {} cells",
                REQUIRED_CELL_MAX_PX * REQUIRED_CELLS_ALONG_LONG_SIDE,
                cells_along_long_side
            );
        }

        // floor, not ceil.
        let ceil_cell = (f64::from(long_side) / f64::from(REQUIRED_CELLS_ALONG_LONG_SIDE))
            .ceil()
            .max(1.0) as u32;
        let ceil_cell = ceil_cell.clamp(REQUIRED_CELL_MIN_PX, REQUIRED_CELL_MAX_PX);
        let ceil_cells = long_side.div_ceil(ceil_cell);
        prop_assert!(
            cells_along_long_side >= ceil_cells,
            "rounding the cell side down can never give fewer cells than rounding it up \
             ({cells_along_long_side} vs {ceil_cells} on a {long_side}px long side)"
        );

        let mut branches = if reachable { 1 } else { 2 };
        if cell == REQUIRED_CELL_MAX_PX {
            branches |= 4;
        }
        if reachable && ceil_cells < REQUIRED_CELLS_ALONG_LONG_SIDE {
            branches |= 8;
        }
        let covered = CELL_SIZE_BRANCHES_SEEN.fetch_or(branches, Ordering::Relaxed) | branches;
        let case = CELL_SIZE_CURSOR.fetch_add(1, Ordering::Relaxed);
        prop_assert!(
            case + 1 < CELL_SIZE_CASES || covered == 0b1111,
            "the run must have produced a plane that reaches the 512 cell floor, one that cannot, \
             one that hits the 64px ceiling and one where rounding up would miss the floor; \
             mask {covered:#b}"
        );
    }
}

/// A deterministic, aperiodic texture, sampled as a function rather than stored:
/// Property 10 needs two *different* contents measured through one sampling, and
/// a closure is the cheapest way to get them.
fn probe_texture(x: f64, y: f64, seed: u64, detail: f64) -> image::Rgb<f32> {
    let quantise = |value: f64| (value * 4.0).floor() as i64;
    let mut hash = (quantise(x) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (quantise(y) as u64).wrapping_mul(0x85EB_CA6B_C2B2_AE35)
        ^ seed;
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let noise = (hash >> 40) as f32 / 16_777_216.0 - 0.5;
    let smooth = ((x * 0.03).sin() + (y * 0.021).cos()) as f32 * 0.1;
    let value = (0.5 + smooth + noise * detail as f32).clamp(0.0, 1.0);
    image::Rgb([value, value * 0.97, value * 0.93])
}

/// An image of `probe_texture` together with a fully valid mask.
fn probe_layer(side: u32, seed: u64, detail: f64) -> (image::Rgb32FImage, image::GrayImage) {
    let image = image::Rgb32FImage::from_fn(side, side, |x, y| {
        probe_texture(f64::from(x), f64::from(y), seed, detail)
    });
    let mask = image::GrayImage::from_pixel(side, side, image::Luma([255]));
    (image, mask)
}

/// Bit 0: the candidate and the composite measured different scores through the
/// same sampling.  Bit 1: a foreign sampling plan was rejected.
static SAMPLING_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 10 so far.
static SAMPLING_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 10 is configured with.  One case measures two whole layers of
/// ownership cells through the production path and still costs well under a
/// millisecond, so the count is set by the fixture variety rather than by cost:
/// the drawn plane sizes cover the cell side range and the drawn layer sides move
/// the grid under the probes.
const SAMPLING_CASES: usize = 64;

proptest! {
    #![proptest_config(ProptestConfig { cases: SAMPLING_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 10: 对于任意 ownership
    // 单元，候选与当前底图的 Sharpness_Score 测量使用逐元素相同的采样位置集合与相同的
    // 采样窗口尺寸，且每单元的采样位置数为 5（中心与四角）。
    //
    // 「逐元素相同」按字面断言：两次测量各自用一个记录闭包，把生产代码**实际索要的**
    // 每一个采样坐标按顺序记下来，然后逐元素比较位模式。断言的不是两个分数接近——两个
    // 分数本来就应该不同，不同的正是内容——而是索要的坐标序列完全一致，窗口尺寸完全一致。
    //
    // 接线在生产函数 `stitching::measure_focus_cells` 上断言，它就是底图与每个候选各走
    // 一遍的那个函数：把同一张图分别放在相差**恰好一个单元**的两个位置测量，两次结果必须
    // 逐位错位一格相等。只有当两侧的采样位置相对各自单元原点完全一致时这个恒等式才成立。
    // 另外断言生产测量会**拒绝**一个与本机位 ownership 网格不一致的采样方案（需求 3.2
    // 的显式断言），否则「同一采样」只是一个约定而不是一条被强制的性质。
    //
    // **Validates: Requirements 3.2**
    #[test]
    fn sharpness_comparison_uses_one_sampling_for_candidate_and_composite(
        plane_width in arb_station_plane_side(),
        plane_height in arb_station_plane_side(),
        layer_side in 48u32..96,
        seed in any::<u64>(),
    ) {
        // 1. The sampling plan of a real station plane: five probes, one window.
        let geometry = focus_fuser::OwnershipGridGeometry::for_plane(plane_width, plane_height);
        let plan = geometry.sampling_plan();
        let probes = plan.probes();
        prop_assert_eq!(
            probes.len(),
            REQUIRED_PROBES_PER_CELL,
            "需求 3.2 measures five positions per ownership cell"
        );
        let centres = probes.iter().filter(|&&(dx, dy)| dx == 0.0 && dy == 0.0).count();
        prop_assert_eq!(centres, 1, "one probe is the cell centre");
        let mut quadrants = BTreeSet::new();
        for &(dx, dy) in probes.iter().filter(|&&(dx, dy)| (dx, dy) != (0.0, 0.0)) {
            prop_assert!(
                dx.abs() > 0.0 && dx.abs() == dy.abs(),
                "the four corner probes sit on the cell diagonals ({dx}, {dy})"
            );
            quadrants.insert((dx > 0.0, dy > 0.0));
        }
        prop_assert_eq!(quadrants.len(), 4, "one corner probe per quadrant");

        // The window is exactly Requirement 3.2's 32 native pixels. Fractional
        // source coordinates are intentional: rounding to a whole-pixel step
        // made the effective window only 26px.
        let step = plan.step_px();
        prop_assert!(step >= 1.0 && step.is_finite(), "the step is usable: {step}");
        prop_assert!(
            (plan.window_px() - REQUIRED_SHARPNESS_WINDOW_PX).abs() < 1e-12,
            "the Sharpness_Score window is {}px, not {REQUIRED_SHARPNESS_WINDOW_PX}px",
            plan.window_px()
        );
        prop_assert!(
            (plan.window_px() - f64::from(mosaic::ACUTANCE_PATCH_POINTS) * step).abs() < 1e-12
        );

        // 2. 逐元素相同的采样位置集合: the positions the production measurement asks
        //    each side for, in order.
        let (origin_x, origin_y) = geometry.cell_origin(1, 1);
        let mut base_positions = Vec::new();
        let base_score = plan.measure(origin_x, origin_y, |x, y| {
            base_positions.push((x.to_bits(), y.to_bits()));
            Some(probe_texture(x, y, seed, 0.02))
        });
        let mut candidate_positions = Vec::new();
        let candidate_score = plan.measure(origin_x, origin_y, |x, y| {
            candidate_positions.push((x.to_bits(), y.to_bits()));
            // The same scene at another focal plane: the low detail amplitude is
            // what defocus does, and it is the *only* difference between the two
            // measurements.
            Some(probe_texture(x, y, seed, 0.60))
        });
        prop_assert_eq!(
            &base_positions,
            &candidate_positions,
            "需求 3.2: the candidate and the composite must be sampled at the same positions"
        );
        prop_assert_eq!(
            base_positions.len(),
            REQUIRED_PROBES_PER_CELL
                * (mosaic::ACUTANCE_PATCH_POINTS * mosaic::ACUTANCE_PATCH_POINTS) as usize,
            "five probes of a 13x13 window each"
        );
        // Every probe window really is a square grid of the plan's step.
        for window in base_positions.chunks(
            (mosaic::ACUTANCE_PATCH_POINTS * mosaic::ACUTANCE_PATCH_POINTS) as usize,
        ) {
            let xs = window
                .iter()
                .map(|&(x, _)| f64::from_bits(x))
                .collect::<Vec<_>>();
            let (min_x, max_x) = (
                xs.iter().copied().fold(f64::INFINITY, f64::min),
                xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
            prop_assert!(
                (max_x - min_x - (f64::from(mosaic::ACUTANCE_PATCH_POINTS) - 1.0) * step).abs()
                    < 1e-9,
                "a probe window spans 12 steps of {step}px, not {}",
                max_x - min_x
            );
        }
        let mut branches = 0usize;
        if base_score != candidate_score {
            // Which is the point: with the sampling pinned, a difference in the
            // score is a difference in the photographs.
            branches |= 1;
        }

        // 3. The production wiring: the same layer measured at two positions one
        //    whole ownership cell apart.  A small plane, because the assertion is
        //    about the sampling geometry and a 9,504px plane of `f32` RGB is
        //    240 MB; the cell side there is the legal minimum of 8px.
        let (image, mask) = probe_layer(layer_side, seed, 0.25);
        let small = focus_fuser::OwnershipGridGeometry::for_plane(
            layer_side + 3 * REQUIRED_CELL_MIN_PX,
            layer_side + 2 * REQUIRED_CELL_MIN_PX,
        );
        prop_assert_eq!(small.cell_size_px, REQUIRED_CELL_MIN_PX);
        let small_plan = small.sampling_plan();
        let cell = small.cell_size_px;
        let (base_cells, base_covered) = stitching::focus_fusion_test_access::measure_cells(
            &small, &small_plan, &image, &mask, cell, cell,
        );
        let (shifted_cells, shifted_covered) = stitching::focus_fusion_test_access::measure_cells(
            &small, &small_plan, &image, &mask, 2 * cell, cell,
        );
        let columns = small.columns as usize;
        // The identity below is trivially true of two all-zero measurements, and
        // an `acutance` window that leaves the layer returns exactly `0.0`, so the
        // fixture has to reach cells that measured something.
        let measured_cells = base_cells
            .iter()
            .filter(|&&score| score > 0.0)
            .count();
        prop_assert!(
            measured_cells >= 4 && base_covered.iter().filter(|&&covered| covered).count() >= 4,
            "the fixture must reach ownership cells whose whole Sharpness_Score window lies on \
             the layer ({} measured, {} covered of {} cells)",
            measured_cells,
            base_covered.iter().filter(|&&covered| covered).count(),
            small.cell_count()
        );
        for row in 0..small.rows as usize {
            for column in 1..columns {
                let here = row * columns + column;
                let left = row * columns + column - 1;
                prop_assert_eq!(
                    shifted_cells[here].to_bits(),
                    base_cells[left].to_bits(),
                    "a layer displaced by exactly one ownership cell must measure the same \
                     Sharpness_Score one cell over: cell ({}, {}) gave {} against {}",
                    column,
                    row,
                    shifted_cells[here],
                    base_cells[left]
                );
                prop_assert_eq!(shifted_covered[here], base_covered[left]);
            }
        }

        // 4. A sampling plan that is not this station's is rejected rather than
        //    silently compared against (需求 3.2).
        let case = SAMPLING_CURSOR.fetch_add(1, Ordering::Relaxed);
        if case == 0 {
            let foreign = focus_fuser::CellSamplingPlan::for_cell(f64::from(cell) * 2.0);
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                stitching::focus_fusion_test_access::measure_cells(
                    &small, &foreign, &image, &mask, 0, 0,
                )
            }))
            .is_err();
            prop_assert!(
                rejected,
                "需求 3.2: measuring one side with a sampling plan other than the station's \
                 ownership grid plan must be refused, or 「相同采样」 is only a convention"
            );
            branches |= 2;
        }
        let covered = SAMPLING_BRANCHES_SEEN.fetch_or(branches, Ordering::Relaxed) | branches;
        prop_assert!(
            case + 1 < SAMPLING_CASES || covered == 0b11,
            "the run must have measured two different contents through the same sampling and \
             rejected a foreign sampling; mask {covered:#b}"
        );
    }
}

/// Bit 0: the penalty applied.  Bit 1: it did not, although the candidate was
/// sharper.  Bit 2: it did not, although the cell was inconsistent.  Bit 3: the
/// inconsistency sat exactly on 需求 3.3's 0.2, which is the only input that
/// tells 「超过 0.2」 apart from 「不低于 0.2」.
static PENALTY_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 11 so far.
static PENALTY_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 11 is configured with.
const PENALTY_CASES: usize = 100;

proptest! {
    #![proptest_config(ProptestConfig { cases: PENALTY_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 11: 对于任意 ownership
    // 单元与任意候选，像素不一致性等于两者在该单元低通后像素差绝对值的中位数归一化到
    // `[0, 1]` 的结果；且当且仅当该不一致性超过 0.2 且该候选 Sharpness_Score 高于当前
    // 底图时，该候选被施加不小于 1.0 的选择代价惩罚。
    //
    // 中位数那一半：用生产的低通差值序列（`mosaic::cell_low_pass_differences`，两侧拿到
    // 逐元素相同的单元内偏移，这里同样用记录闭包断言）独立重算中位数并与
    // `focus_fuser::cell_disagreement` 比较。旧的 `mosaic::ownership_disagreement`
    // 保留的是高分位数语义，所以「是中位数而不是某个高分位数」是一条会被真正违反的断言。
    //
    // 惩罚那一半：断言充要性，并额外断言它的后果——被惩罚的候选在数据项上永远赢不了当前
    // 底图（归一化 Sharpness_Score 的差最多是 1.0，而惩罚不小于 1.0），这正是需求 3.3
    // 要挡住的「更清晰但错位的帧再画一遍笔画」。
    //
    // **Validates: Requirements 3.3**
    #[test]
    fn disagreement_is_the_low_pass_median_and_the_penalty_follows_it(
        seed in any::<u64>(),
        cell_size in prop::sample::select(vec![8.0f64, 12.0, 18.0, 64.0]),
        candidate_detail in 0.0f64..0.8,
        candidate_offset in 0.0f64..6.0,
        // `k / 40` so that one draw lands exactly on 需求 3.3's 0.2 — `8.0 / 40.0`
        // is the same double as the literal — and one draw in four is that one, so
        // the boundary is not a once-in-forty-one accident.
        disagreement_step in prop_oneof![3 => 0u32..=40, 1 => Just(8u32)],
        sharpness_pair in (0u32..=20, 0u32..=20),
    ) {
        // 1. 低通后像素差绝对值的中位数, on the production difference sequence.
        let mut candidate_positions = Vec::new();
        let mut base_positions = Vec::new();
        let candidate_at = |fx: f64, fy: f64| {
            candidate_positions.push((fx.to_bits(), fy.to_bits()));
            Some(probe_texture(
                fx * cell_size + candidate_offset,
                fy * cell_size,
                seed,
                candidate_detail,
            ))
        };
        let base_at = |fx: f64, fy: f64| {
            base_positions.push((fx.to_bits(), fy.to_bits()));
            Some(probe_texture(fx * cell_size, fy * cell_size, seed, 0.25))
        };
        let differences = mosaic::cell_low_pass_differences(candidate_at, base_at);
        prop_assert_eq!(
            &candidate_positions,
            &base_positions,
            "需求 3.3 compares the candidate and the composite over the same cell offsets"
        );
        prop_assert!(!differences.is_empty());
        // An independent median: ascending order, and the mean of the two middle
        // samples for an even count.
        let expected = {
            let mut sorted = differences.clone();
            sorted.sort_by(f64::total_cmp);
            let middle = sorted.len() / 2;
            let median = if sorted.len() % 2 == 1 {
                sorted[middle]
            } else {
                (sorted[middle - 1] + sorted[middle]) * 0.5
            };
            median.clamp(0.0, 1.0)
        };
        let mut candidate_positions = Vec::new();
        let mut base_positions = Vec::new();
        let measured = focus_fuser::cell_disagreement(
            |fx: f64, fy: f64| {
                candidate_positions.push((fx.to_bits(), fy.to_bits()));
                Some(probe_texture(
                    fx * cell_size + candidate_offset,
                    fy * cell_size,
                    seed,
                    candidate_detail,
                ))
            },
            |fx: f64, fy: f64| {
                base_positions.push((fx.to_bits(), fy.to_bits()));
                Some(probe_texture(fx * cell_size, fy * cell_size, seed, 0.25))
            },
        );
        prop_assert_eq!(&candidate_positions, &base_positions);
        prop_assert!(
            (measured - expected).abs() < 1e-12,
            "需求 3.3's inconsistency is the median {expected:.12} of the low-pass differences, \
             not {measured:.12}"
        );
        prop_assert!(
            (0.0..=1.0).contains(&measured),
            "normalised to [0, 1], got {measured}"
        );
        // A candidate that is the composite disagrees with it not at all, and a
        // candidate with no position valid in both images is maximally
        // inconsistent rather than perfectly consistent.
        prop_assert_eq!(
            focus_fuser::cell_disagreement(
                |fx: f64, fy: f64| Some(probe_texture(fx * cell_size, fy * cell_size, seed, 0.25)),
                |fx: f64, fy: f64| Some(probe_texture(fx * cell_size, fy * cell_size, seed, 0.25)),
            ),
            0.0
        );
        prop_assert_eq!(
            focus_fuser::cell_disagreement(|_, _| None, |_, _| None),
            1.0
        );

        // 2. 当且仅当不一致性超过 0.2 且候选更清晰: `k / 40` puts a draw exactly on
        //    the 0.2 boundary, bit for bit.
        let disagreement = f64::from(disagreement_step) / 40.0;
        let (candidate_sharpness, base_sharpness) = (
            f64::from(sharpness_pair.0) / 20.0,
            f64::from(sharpness_pair.1) / 20.0,
        );
        let penalty = focus_fuser::mismatch_penalty(
            disagreement,
            candidate_sharpness,
            base_sharpness,
        );
        let applies = disagreement > REQUIRED_DISAGREEMENT_VETO
            && candidate_sharpness > base_sharpness;
        prop_assert_eq!(
            penalty > 0.0,
            applies,
            "需求 3.3 penalises a candidate exactly when it is both inconsistent ({} > {}) and \
             sharper ({} > {}); penalty {}",
            disagreement,
            REQUIRED_DISAGREEMENT_VETO,
            candidate_sharpness,
            base_sharpness,
            penalty
        );
        if applies {
            prop_assert!(
                penalty >= REQUIRED_MIN_MISMATCH_PENALTY,
                "需求 3.3 requires a penalty of at least {REQUIRED_MIN_MISMATCH_PENALTY}, got \
                 {penalty}"
            );
            // The consequence 需求 3.3 exists for: a sharper but inconsistent
            // candidate cannot take the cell on the data term, because the
            // sharpness it could gain is at most 1.0 and the penalty is not less
            // than that.
            prop_assert!(
                !focus_fuser::candidate_wins_cell(
                    focus_fuser::data_cost(base_sharpness, 0.0),
                    focus_fuser::data_cost(candidate_sharpness, penalty),
                ),
                "a penalised candidate must not win the cell ({candidate_sharpness} against \
                 {base_sharpness} with penalty {penalty})"
            );
        } else {
            prop_assert_eq!(penalty, 0.0, "an unpenalised candidate carries no penalty");
        }

        let mut branches = if applies { 1 } else { 0 };
        if !applies && candidate_sharpness > base_sharpness {
            branches |= 2;
        }
        if !applies && disagreement > REQUIRED_DISAGREEMENT_VETO {
            branches |= 4;
        }
        if disagreement == REQUIRED_DISAGREEMENT_VETO {
            branches |= 8;
        }
        let covered = PENALTY_BRANCHES_SEEN.fetch_or(branches, Ordering::Relaxed) | branches;
        let case = PENALTY_CURSOR.fetch_add(1, Ordering::Relaxed);
        prop_assert!(
            case + 1 < PENALTY_CASES || covered == 0b1111,
            "the run must have penalised a candidate, declined to penalise a sharper consistent \
             one and an inconsistent softer one, and drawn the exact 0.2 boundary; \
             mask {covered:#b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Property 12 harness: bounded multi-label ownership cost grids
// ---------------------------------------------------------------------------

/// Generated input to the multi-label ownership solver.
///
/// Costs are positive hundredths, so every non-empty grid has a positive oracle
/// optimum and the multiplicative 1.01 bound is meaningful without an additive
/// escape hatch. `None` is a candidate that does not cover that cell.
#[derive(Clone, Debug)]
struct CostGridDraw {
    columns: usize,
    rows: usize,
    labels: usize,
    costs: Vec<Option<u16>>,
    disagreement: Vec<u8>,
}

impl CostGridDraw {
    fn cell_count(&self) -> usize {
        self.columns * self.rows
    }

    fn cost(&self, cell: usize, label: usize) -> Option<f64> {
        self.costs[cell * self.labels + label].map(|units| f64::from(units) / 100.0)
    }

    fn disagreement(&self, cell: usize) -> f64 {
        f64::from(self.disagreement[cell]) / 100.0
    }

    fn is_covered(&self, cell: usize) -> bool {
        (0..self.labels).any(|label| self.cost(cell, label).is_some())
    }

    fn production_grid(&self) -> focus_fuser::OwnershipCostGrid {
        let mut grid = focus_fuser::OwnershipCostGrid::new(self.columns, self.rows, self.labels);
        for cell in 0..self.cell_count() {
            for label in 0..self.labels {
                grid.set_cost(cell, label, self.cost(cell, label));
            }
            grid.set_disagreement(cell, self.disagreement(cell));
        }
        grid
    }

    /// Evaluate the Potts energy independently of `OwnershipCostGrid::energy`.
    fn energy(&self, owners: &[u16]) -> f64 {
        if owners.len() != self.cell_count() {
            return f64::INFINITY;
        }
        let mut total = 0.0;
        for (cell, &owner) in owners.iter().enumerate() {
            if owner == stitching::NO_OWNER {
                if self.is_covered(cell) {
                    return f64::INFINITY;
                }
                continue;
            }
            let label = usize::from(owner - 1);
            let Some(cost) = (label < self.labels)
                .then(|| self.cost(cell, label))
                .flatten()
            else {
                return f64::INFINITY;
            };
            total += cost;
        }
        for cell in 0..self.cell_count() {
            let (x, y) = (cell % self.columns, cell / self.columns);
            for neighbour in [
                (x + 1 < self.columns).then(|| cell + 1),
                (y + 1 < self.rows).then(|| cell + self.columns),
            ]
            .into_iter()
            .flatten()
            {
                if owners[cell] != stitching::NO_OWNER
                    && owners[neighbour] != stitching::NO_OWNER
                    && owners[cell] != owners[neighbour]
                {
                    total += super::super::seam_cut::pairwise_weight(
                        self.disagreement(cell),
                        self.disagreement(neighbour),
                    );
                }
            }
        }
        total
    }
}

/// At most `3^9` legal assignments: uncovered cells have only `NO_OWNER` as an
/// option, while covered cells enumerate only candidates that actually cover
/// them. This avoids the old unit oracle's deliberately-invalid fourth option
/// and keeps 100 exhaustive cases cheap enough for the normal library suite.
const GRAPH_CUT_MAX_ORACLE_LABELINGS: usize = 19_683;
const GRAPH_CUT_PROPERTY_CASES: usize = 100;

fn arb_cost_grid() -> impl Strategy<Value = CostGridDraw> {
    (1usize..=3, 1usize..=3, 2usize..=3).prop_flat_map(|(columns, rows, labels)| {
        let cells = columns * rows;
        (
            Just((columns, rows, labels)),
            prop::collection::vec(
                prop_oneof![
                    1 => Just(None),
                    7 => (1u16..=100).prop_map(Some),
                ],
                cells * labels,
            ),
            prop::collection::vec(0u8..=30, cells),
        )
            .prop_map(|((columns, rows, labels), mut costs, disagreement)| {
                // Keep the multiplicative oracle bound non-vacuous: at least one
                // cell is covered and every finite data cost is strictly positive.
                costs[0] = Some(costs[0].unwrap_or(1));
                CostGridDraw {
                    columns,
                    rows,
                    labels,
                    costs,
                    disagreement,
                }
            })
    })
}

/// Exhaustively enumerate every legal owner assignment and return its minimum
/// energy together with the number of assignments examined.
fn exhaustive_graph_cut_oracle(draw: &CostGridDraw) -> (f64, usize) {
    let options = (0..draw.cell_count())
        .map(|cell| {
            let labels = (0..draw.labels)
                .filter(|&label| draw.cost(cell, label).is_some())
                .map(focus_fuser::owner_of)
                .collect::<Vec<_>>();
            if labels.is_empty() {
                vec![stitching::NO_OWNER]
            } else {
                labels
            }
        })
        .collect::<Vec<_>>();
    let assignments = options.iter().map(Vec::len).product::<usize>();
    assert!(assignments <= GRAPH_CUT_MAX_ORACLE_LABELINGS);

    let mut best = f64::INFINITY;
    for encoded in 0..assignments {
        let mut rest = encoded;
        let owners = options
            .iter()
            .map(|cell_options| {
                let owner = cell_options[rest % cell_options.len()];
                rest /= cell_options.len();
                owner
            })
            .collect::<Vec<_>>();
        best = best.min(draw.energy(&owners));
    }
    (best, assignments)
}

/// A fixed focus-like witness where per-cell minima introduce an expensive
/// isolated seam. It makes a temporary "return per-cell minima" production
/// mutation fail deterministically instead of hoping a random case exposes it.
fn graph_cut_non_vacuity_witness() -> CostGridDraw {
    let mut costs = Vec::with_capacity(18);
    for cell in 0..9 {
        costs.push(Some(30));
        costs.push(Some(if cell == 4 { 1 } else { 60 }));
    }
    CostGridDraw {
        columns: 3,
        rows: 3,
        labels: 2,
        costs,
        disagreement: vec![0; 9],
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: GRAPH_CUT_PROPERTY_CASES as u32,
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 12: 对于任意
    // ownership 代价网格与任意候选集合，每个已覆盖单元恰好得到一个 owner；在不超过
    // 3x3、候选数不超过 3 的网格上，生产总代价不超过穷举最优值的 1.01 倍；相同输入重复
    // 求解得到逐位相同的标签，代价完全相同时按确定的候选顺序解 tie。
    //
    // Exact multi-label Potts optimum is intentionally *not* required here:
    // alpha-expansion is an approximation algorithm and does not guarantee the
    // exact global optimum. The exhaustive oracle instead enforces the approved
    // 1% production bound on generated grids while retaining all owner-validity
    // and determinism conditions.
    //
    // **Validates: Requirements 3.4**
    #[test]
    fn multi_label_graph_cut_is_complete_near_optimal_and_deterministic(
        draw in arb_cost_grid(),
    ) {
        let grid = draw.production_grid();
        let first = focus_fuser::solve_ownership(&grid);
        let repeated = focus_fuser::solve_ownership(&grid);

        prop_assert_eq!(first.solver_status, super::report::FusionSolverStatus::GraphCut);
        prop_assert_eq!(
            &first.owners,
            &repeated.owners,
            "identical cost-grid input must produce bit-identical u16 labels"
        );
        prop_assert_eq!(first.owners.len(), draw.cell_count());
        for cell in 0..draw.cell_count() {
            let owner = first.owners[cell];
            if draw.is_covered(cell) {
                prop_assert_ne!(
                    owner,
                    stitching::NO_OWNER,
                    "covered cell {} has no owner",
                    cell
                );
                let label = usize::from(owner - 1);
                prop_assert!(
                    label < draw.labels && draw.cost(cell, label).is_some(),
                    "covered cell {cell} has invalid owner {owner}"
                );
            } else {
                prop_assert_eq!(
                    owner,
                    stitching::NO_OWNER,
                    "uncovered cell {} must keep the reserved owner",
                    cell
                );
            }
        }

        let achieved = draw.energy(&first.owners);
        let (optimal, assignments) = exhaustive_graph_cut_oracle(&draw);
        prop_assert!(optimal > 0.0 && optimal.is_finite());
        prop_assert!(assignments <= GRAPH_CUT_MAX_ORACLE_LABELINGS);
        prop_assert!(
            achieved <= optimal * 1.01,
            "production cost {achieved:.12} exceeds 1.01 * exhaustive optimum {optimal:.12} \
             on a {}x{} grid with {} labels ({assignments} legal assignments)",
            draw.columns,
            draw.rows,
            draw.labels,
        );

        // A full-grid tie has one seam-free optimum per label. Candidate order
        // is the sole tie-break, so the first candidate must win every cell.
        let mut tied = focus_fuser::OwnershipCostGrid::new(
            draw.columns,
            draw.rows,
            draw.labels,
        );
        for cell in 0..draw.cell_count() {
            for label in 0..draw.labels {
                tied.set_cost(cell, label, Some(0.5));
            }
            tied.set_disagreement(cell, draw.disagreement(cell));
        }
        let tied_first = focus_fuser::solve_ownership(&tied);
        let tied_repeated = focus_fuser::solve_ownership(&tied);
        prop_assert_eq!(&tied_first.owners, &tied_repeated.owners);
        prop_assert_eq!(
            tied_first.owners,
            vec![focus_fuser::owner_of(0); draw.cell_count()],
            "an exact tie must resolve to the first candidate deterministically"
        );

        // Mutation-sensitivity witness: unlike the per-cell fallback, the graph
        // cut must suppress an isolated label change whose seam costs more than
        // its data-term improvement.
        let witness = graph_cut_non_vacuity_witness();
        let witness_solution = focus_fuser::solve_ownership(&witness.production_grid());
        let witness_cost = witness.energy(&witness_solution.owners);
        let (witness_optimum, _) = exhaustive_graph_cut_oracle(&witness);
        prop_assert!(witness_cost <= witness_optimum * 1.01);
    }
}

#[test]
fn graph_cut_saved_counterexample_stays_within_property_12_bound() {
    let draw = CostGridDraw {
        columns: 2,
        rows: 3,
        labels: 3,
        costs: vec![
            Some(1),
            None,
            Some(1),
            None,
            Some(1),
            Some(1),
            None,
            Some(1),
            Some(1),
            Some(3),
            Some(1),
            None,
            None,
            None,
            None,
            Some(1),
            None,
            None,
        ],
        disagreement: vec![0, 0, 0, 24, 0, 22],
    };
    let solution = focus_fuser::solve_ownership(&draw.production_grid());
    let achieved = draw.energy(&solution.owners);
    assert!(
        achieved <= 6.53 * 1.01 + 1.0e-9,
        "counterexample energy {achieved}"
    );
}

// ---------------------------------------------------------------------------
// Properties 14 / 15 / 16 harness: the ownership grid of one Capture_Station
// ---------------------------------------------------------------------------
//
// `focus_fuser::StationFusion` is the entry point the production focus renderer
// drives: `stitching::focus_stack_stitcher_with_margin_policy` builds one, seeds
// it with the anchor frame's per-cell measurements and folds every further frame
// into it with one exact two-label cut.  `low_sharpness_cells`,
// `low_sharpness_region_records` and `confidence` are read straight off that same
// object into the Stack_Report, so driving it here drives the production
// decision rather than a re-implementation of it.
//
// The whole-grid `focus_fuser::solve_ownership` is deliberately *not* used: its
// own module marks it `#![allow(dead_code)]` and documents that the render loop
// cannot afford it, so a property wired to it would assert nothing about a run.

/// `SplitMix64`, so a drawn seed expands into a whole grid of per-cell
/// measurements as a pure function on every platform.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

/// Uniform value in `[0, 1)` from the mixer above.
fn unit_from(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Station plane sides Properties 14–16 fuse over, in native pixels.
///
/// Small on purpose.  Below 4,096px the ownership cell side is the 8px minimum
/// (Property 9), so these planes give a 6×6..12×12 cell grid: enough cells for a
/// percentile and for four-connected regions, few enough that a whole station
/// folds in microseconds.  The cell *geometry* is what Property 9 covers over
/// the full plane size range; what matters here is the number of cells.
fn arb_station_plane() -> impl Strategy<Value = (u32, u32)> {
    (
        prop::sample::select(vec![48u32, 56, 64, 80, 96]),
        prop::sample::select(vec![48u32, 56, 64, 80, 96]),
    )
}

/// The per-cell measurements of one synthetic Capture_Station.
#[derive(Clone)]
struct StationDraw {
    plane: (u32, u32),
    geometry: focus_fuser::OwnershipGridGeometry,
    frames: Vec<focus_fuser::CandidateCells>,
}

/// A summary rather than hundreds of cells: `proptest` prints this on failure.
impl std::fmt::Debug for StationDraw {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StationDraw")
            .field("plane", &self.plane)
            .field("cells", &self.geometry.cell_count())
            .field("frames", &self.frames.len())
            .field(
                "covered_cells_per_frame",
                &self
                    .frames
                    .iter()
                    .map(|frame| frame.covered.iter().filter(|&&covered| covered).count())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Per-cell measurements expanded from a drawn seed.
///
/// A 12×12 grid times three frames is 432 numbers; a `prop::collection::vec`
/// draw of that size prints unreadably and shrinks uselessly, while a seed keeps
/// the case a handful of values and still covers the input space.  The drawn
/// scalars are the ones the assertions actually turn on: the plane, the frame
/// count, the Sharpness_Score quantisation (which decides how often two
/// candidates tie and how often a cell sits exactly *on* the percentile) and the
/// coverage ratio.
#[allow(clippy::too_many_arguments)]
fn station_draw(
    plane: (u32, u32),
    frames: usize,
    seed: u64,
    sharpness_levels: u32,
    coverage_ratio: f64,
    disagreement: f64,
    uncovered_corner: bool,
) -> StationDraw {
    let geometry = focus_fuser::OwnershipGridGeometry::for_plane(plane.0, plane.1);
    let cells = geometry.cell_count();
    let corner = cells - 1;
    let mut state = seed | 1;
    let mut drawn = Vec::with_capacity(frames);
    for _ in 0..frames {
        let mut frame = focus_fuser::CandidateCells::new(cells);
        for cell in 0..cells {
            // 需求 3.7: a cell no frame projects onto has no candidate at all.
            let covered =
                unit_from(&mut state) < coverage_ratio && !(uncovered_corner && cell == corner);
            frame.covered[cell] = covered;
            frame.sharpness[cell] = if covered {
                (unit_from(&mut state) * f64::from(sharpness_levels)).round()
                    / f64::from(sharpness_levels)
            } else {
                0.0
            };
            // One inconsistency for the whole station, so the smoothness term of
            // every round prices seams the same way and the ownership decision
            // is driven by the Sharpness_Score the properties are about.
            frame.disagreement[cell] = disagreement;
        }
        drawn.push(frame);
    }
    StationDraw {
        plane,
        geometry,
        frames: drawn,
    }
}

/// A drawn Capture_Station of 2..=3 registration-successful frames.
fn arb_station_draw() -> impl Strategy<Value = StationDraw> {
    (
        arb_station_plane(),
        2usize..=3,
        any::<u64>(),
        // Coarse quantisation on purpose: with few distinct Sharpness_Scores a
        // cell lands exactly *on* the 10th percentile often, which is the only
        // input that tells 需求 3.8's 「低于」 apart from 「不高于」.
        prop::sample::select(vec![4u32, 8, 10, 20]),
        // Include stations with no projected evidence deliberately.  This makes
        // the production classifier's empty-statistic behavior observable
        // instead of relying on an astronomically unlikely random draw.
        prop_oneof![1 => Just(0.0), 5 => 0.55f64..=1.0],
        prop_oneof![3 => 0.0f64..=0.05, 1 => Just(0.0f64)],
        any::<bool>(),
    )
        .prop_map(
            |(plane, frames, seed, levels, coverage, disagreement, corner)| {
                station_draw(plane, frames, seed, levels, coverage, disagreement, corner)
            },
        )
}

/// Drive the production fusion over a drawn station: seed the first frame, fold
/// the rest, exactly as the render loop does.
fn fuse_station(draw: &StationDraw) -> focus_fuser::StationFusion {
    let mut fusion = focus_fuser::StationFusion::new(draw.plane.0, draw.plane.1, draw.frames.len());
    fusion.seed(0, &draw.frames[0].sharpness, &draw.frames[0].covered);
    for label in 1..draw.frames.len() {
        fusion.fold(label, &draw.frames[label]);
    }
    fusion
}

/// Winning Sharpness_Score, runner-up Sharpness_Score and candidate count of
/// every cell, from the drawn measurements alone.
///
/// The independent oracle both Property 14 and Property 15 compare against: it
/// reads the drawn per-frame numbers, never the fusion's own state.
fn station_cell_evidence(draw: &StationDraw) -> Vec<(f64, f64, usize)> {
    (0..draw.geometry.cell_count())
        .map(|cell| {
            let mut scores = draw
                .frames
                .iter()
                .filter(|frame| frame.covered[cell])
                .map(|frame| frame.sharpness[cell])
                .collect::<Vec<_>>();
            let count = scores.len();
            scores.sort_by(|left, right| right.total_cmp(left));
            (
                scores.first().copied().unwrap_or(0.0),
                scores.get(1).copied().unwrap_or(0.0),
                count,
            )
        })
        .collect()
}

/// Nearest-rank 10th percentile of a sample, recomputed from 需求 3.8's own
/// wording rather than from the production helper.
fn nearest_rank_tenth_percentile(sample: &[f64]) -> f64 {
    let mut sorted = sample.to_vec();
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len() - 1;
    let rank = ((last as f64) * 0.10).round() as usize;
    sorted[rank.min(last)]
}

/// Four-connected components of a boolean cell mask, counted independently of
/// `focus_fuser::low_sharpness_regions`.
fn connected_component_count(columns: usize, rows: usize, mask: &[bool]) -> usize {
    let mut seen = vec![false; mask.len()];
    let mut components = 0usize;
    let mut stack = Vec::new();
    for seed in 0..mask.len() {
        if seen[seed] || !mask[seed] {
            continue;
        }
        components += 1;
        seen[seed] = true;
        stack.push(seed);
        while let Some(cell) = stack.pop() {
            let (x, y) = (cell % columns, cell / columns);
            for neighbour in [
                (y > 0).then(|| cell - columns),
                (x > 0).then(|| cell - 1),
                (x + 1 < columns).then(|| cell + 1),
                (y + 1 < rows).then(|| cell + columns),
            ]
            .into_iter()
            .flatten()
            {
                if !seen[neighbour] && mask[neighbour] {
                    seen[neighbour] = true;
                    stack.push(neighbour);
                }
            }
        }
    }
    components
}

/// 需求 3.8: the percentile of the Sharpness_Score distribution below which a
/// cell is a low sharpness region.
const REQUIRED_LOW_SHARPNESS_PERCENTILE: f64 = 0.10;

/// Amended 需求 11.12: a covered pixel is textured when at least one candidate
/// in its ownership cell reaches this fixed Sharpness_Score floor.  This is a
/// separate absolute boundary from 需求 3.8's station-relative P10; keeping the
/// constant here prevents the property from accidentally reviving the obsolete
/// all-nontransparent-pixel statistic while it exercises shared boundary data.
const TEXTURED_PIXEL_SHARPNESS_FLOOR: f64 = 0.10;

/// Build deterministic stations that exercise every non-vacuity branch of
/// Property 14 without sharing counters with other tests or persisted cases.
///
/// The rich station has two disconnected cells strictly below its nearest-rank
/// P10, cells exactly on P10 (including the inclusive Textured_Pixel floor),
/// ordinary non-low cells, and one uncovered cell. The other stations pin the
/// no-low and no-evidence outcomes that cannot coexist with those facts in one
/// station.
fn low_sharpness_coverage_witnesses() -> [StationDraw; 3] {
    let geometry = focus_fuser::OwnershipGridGeometry::for_plane(48, 48);
    let cells = geometry.cell_count();

    let mut rich_frames = vec![focus_fuser::CandidateCells::new(cells); 2];
    for frame in &mut rich_frames {
        frame.covered.fill(true);
        frame.sharpness.fill(0.5);
        frame.covered[cells - 1] = false;
        frame.sharpness[cells - 1] = 0.0;
        for cell in [0, 2] {
            frame.sharpness[cell] = 0.0;
        }
        for cell in [1, 3] {
            frame.sharpness[cell] = TEXTURED_PIXEL_SHARPNESS_FLOOR;
        }
    }
    let rich = StationDraw {
        plane: (48, 48),
        geometry,
        frames: rich_frames,
    };

    let uniform_frames = (0..2)
        .map(|_| {
            let mut frame = focus_fuser::CandidateCells::new(cells);
            frame.covered.fill(true);
            frame.sharpness.fill(0.5);
            frame
        })
        .collect();
    let uniform = StationDraw {
        plane: (48, 48),
        geometry,
        frames: uniform_frames,
    };

    let empty = StationDraw {
        plane: (48, 48),
        geometry,
        frames: vec![focus_fuser::CandidateCells::new(cells); 2],
    };

    [rich, uniform, empty]
}

/// Classify one station by the exact semantic branches Property 14 asserts.
fn low_sharpness_branch_mask(draw: &StationDraw) -> usize {
    let fusion = fuse_station(draw);
    let evidence = station_cell_evidence(draw);
    let measured = fusion.low_sharpness_cells();
    let low_cells = measured.iter().filter(|&&low| low).count();
    let components = connected_component_count(
        draw.geometry.columns as usize,
        draw.geometry.rows as usize,
        &measured,
    );
    let winning = evidence
        .iter()
        .filter(|(_, _, count)| *count > 0)
        .map(|(best, _, _)| *best)
        .collect::<Vec<_>>();

    let mut branches = if low_cells > 0 { 1 } else { 2 };
    if !winning.is_empty() {
        let threshold = nearest_rank_tenth_percentile(&winning);
        if evidence
            .iter()
            .any(|&(best, _, count)| count > 0 && best.to_bits() == threshold.to_bits())
        {
            branches |= 4;
        }
    }
    if evidence.iter().any(|&(_, _, count)| count == 0) {
        branches |= 8;
    }
    if components > 1 {
        branches |= 16;
    }
    if winning.is_empty() {
        branches |= 32;
    }
    if evidence.iter().any(|&(best, _, count)| {
        count > 0 && best.to_bits() == TEXTURED_PIXEL_SHARPNESS_FLOOR.to_bits()
    }) {
        branches |= 64;
    }
    branches
}

fn assert_local_low_sharpness_coverage() -> Result<(), TestCaseError> {
    let [rich, uniform, empty] = low_sharpness_coverage_witnesses();

    let rich_fusion = fuse_station(&rich);
    let rich_low = rich_fusion.low_sharpness_cells();
    prop_assert!(rich_low[0] && rich_low[2]);
    prop_assert_eq!(rich_low.iter().filter(|&&low| low).count(), 2);
    prop_assert_eq!(
        rich_fusion.low_sharpness_region_records((0.0, 0.0)).len(),
        2
    );
    let rich_winning = station_cell_evidence(&rich)
        .into_iter()
        .filter(|(_, _, count)| *count > 0)
        .map(|(best, _, _)| best)
        .collect::<Vec<_>>();
    prop_assert_eq!(
        focus_fuser::low_sharpness_threshold(&rich_winning).to_bits(),
        TEXTURED_PIXEL_SHARPNESS_FLOOR.to_bits()
    );

    let uniform_fusion = fuse_station(&uniform);
    prop_assert!(uniform_fusion.low_sharpness_cells().iter().all(|&low| !low));

    let empty_fusion = fuse_station(&empty);
    prop_assert!(empty_fusion.low_sharpness_cells().iter().all(|&low| !low));
    prop_assert!(
        empty_fusion
            .low_sharpness_region_records((0.0, 0.0))
            .is_empty()
    );

    let covered = [&rich, &uniform, &empty]
        .into_iter()
        .fold(0usize, |mask, draw| mask | low_sharpness_branch_mask(draw));
    prop_assert_eq!(
        covered,
        0b111_1111,
        "deterministic local witnesses must cover every Property 14 semantic branch"
    );
    Ok(())
}

/// Cases Property 14 is configured with.
const LOW_SHARPNESS_CASES: usize = 100;

proptest! {
    #![proptest_config(ProptestConfig { cases: LOW_SHARPNESS_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 14: 对于任意
    // Capture_Station 的 Sharpness_Score 分布，被标记为低清晰度的 ownership 单元集合
    // 恰好等于所有候选 Sharpness_Score 都低于该分布 10 百分位的单元集合。
    //
    // 断言是**充要**的：逐单元比较生产的 `low_sharpness_cells()` 与独立重算的集合，而不是
    // 只检查被标记的单元确实低于分位数。
    //
    // 「该分布」按生产实现的读法取**每个已 owner 单元的胜出 Sharpness_Score** 的分布，
    // 而不是把所有候选的分数混在一起。两者都能读出需求原文，但只有前者能让「所有候选都低于
    // 分位数」这条子句有意义：把失焦帧的分数一并混入分布会把分位数压到失焦帧那一侧，
    // 于是「低清晰度区域」反而随失焦帧的数量变多而变少。这条选择在断言里被钉住，
    // 并且额外断言「所有候选都低于阈值 ⟺ 胜出候选低于阈值」——生产代码用的正是 `best`，
    // 所以这条等价关系是被断言的性质而不是被假设的。
    //
    // 分位数本身用最近秩（nearest rank）独立重算并断言它是该机位真的产生过的取值：这保证
    // 分布均匀的机位不会有任何单元「严格低于」它，也就没有假的低清晰度区域。
    //
    // 需求 3.8 的上报那一半一并断言：区域是四连通的（用独立的洪泛计数比较），
    // 面积等于低清晰度单元数乘以单元面积（不是包围盒面积，L 形区域会被高估），
    // 世界坐标位置按 `world_origin` 平移。
    //
    // **Validates: Requirements 3.8**
    #[test]
    fn low_sharpness_cells_are_exactly_the_cells_below_the_station_percentile(
        draw in arb_station_draw(),
        world_origin in (-4096.0f64..4096.0, -4096.0f64..4096.0),
    ) {
        let fusion = fuse_station(&draw);
        let cells = draw.geometry.cell_count();
        let evidence = station_cell_evidence(&draw);
        let owners = fusion.owners();

        // A cell has an owner exactly when some candidate covers it (需求 3.7 /
        // 3.11); the low sharpness set is defined over those cells.
        for cell in 0..cells {
            prop_assert_eq!(
                owners[cell] != focus_fuser::NO_OWNER,
                evidence[cell].2 > 0,
                "cell {} has {} candidate(s) but owner {}",
                cell,
                evidence[cell].2,
                owners[cell]
            );
        }

        let winning = (0..cells)
            .filter(|&cell| evidence[cell].2 > 0)
            .map(|cell| evidence[cell].0)
            .collect::<Vec<_>>();
        let measured = fusion.low_sharpness_cells();
        prop_assert_eq!(measured.len(), cells);
        if winning.is_empty() {
            prop_assert!(
                measured.iter().all(|&low| !low),
                "a station with no covered cell has no low sharpness region"
            );
            prop_assert!(fusion.low_sharpness_region_records(world_origin).is_empty());
            assert_local_low_sharpness_coverage()?;
            return Ok(());
        }

        // 该分布的 10 百分位, by nearest rank on the winning scores.
        let threshold = nearest_rank_tenth_percentile(&winning);
        prop_assert_eq!(
            focus_fuser::low_sharpness_threshold(&winning).to_bits(),
            threshold.to_bits(),
            "需求 3.8's {}th percentile must be the nearest rank value of the station's own \
             Sharpness_Score distribution",
            REQUIRED_LOW_SHARPNESS_PERCENTILE * 100.0
        );
        prop_assert!(
            winning.iter().any(|score| *score == threshold),
            "the percentile must be a value the station really produced, so a uniformly sharp \
             station has no cell strictly below it: threshold {threshold}"
        );

        // 所有候选 Sharpness_Score 都低于该分位数 ⟺ 胜出候选低于该分位数.
        let expected = (0..cells)
            .map(|cell| {
                let covering = draw
                    .frames
                    .iter()
                    .filter(|frame| frame.covered[cell])
                    .map(|frame| frame.sharpness[cell])
                    .collect::<Vec<_>>();
                let all_below = !covering.is_empty()
                    && covering.iter().all(|score| *score < threshold);
                all_below
            })
            .collect::<Vec<_>>();
        for cell in 0..cells {
            prop_assert_eq!(
                expected[cell],
                evidence[cell].2 > 0 && evidence[cell].0 < threshold,
                "cell {}: 「所有候选都低于阈值」 and 「胜出候选低于阈值」 must be the same set",
                cell
            );
        }

        // 被标记为低清晰度的单元集合恰好等于该集合.
        for cell in 0..cells {
            prop_assert_eq!(
                measured[cell],
                expected[cell],
                "需求 3.8 on cell {} of a {}x{} grid: winner {}, threshold {}, {} candidate(s); \
                 production marked it {}, the requirement marks it {}",
                cell,
                draw.geometry.columns,
                draw.geometry.rows,
                evidence[cell].0,
                threshold,
                evidence[cell].2,
                measured[cell],
                expected[cell]
            );
        }

        // The reported regions: four-connected, world placed, area by cell count.
        let low_sharpness_records = fusion.low_sharpness_region_records(world_origin);
        let components = connected_component_count(
            draw.geometry.columns as usize,
            draw.geometry.rows as usize,
            &measured,
        );
        prop_assert_eq!(
            low_sharpness_records.len(),
            components,
            "需求 3.8 reports one record per four-connected low sharpness region"
        );
        let cell_area = f64::from(draw.geometry.cell_size_px) * f64::from(draw.geometry.cell_size_px);
        let low_cells = measured.iter().filter(|&&low| low).count();
        let reported_area = low_sharpness_records
            .iter()
            .map(|record| record.area_px)
            .sum::<f64>();
        prop_assert!(
            (reported_area - low_cells as f64 * cell_area).abs() < 1e-6,
            "需求 3.8's area is the covered cell count times the cell area: {} cells of {} px² is \
             {}, reported {}",
            low_cells,
            cell_area,
            low_cells as f64 * cell_area,
            reported_area
        );
        for record in &low_sharpness_records {
            prop_assert!(
                record.world.left >= world_origin.0 - 1e-9
                    && record.world.top >= world_origin.1 - 1e-9
                    && record.world.left + record.world.width
                        <= world_origin.0 + f64::from(draw.geometry.columns) * cell_area.sqrt() + 1e-9,
                "a region's world rectangle lies inside the station plane placed at \
                 {world_origin:?}: {:?}",
                record.world
            );
            prop_assert!(record.area_px > 0.0 && record.world.width > 0.0 && record.world.height > 0.0);
        }

        if evidence.iter().any(|&(best, _, count)| {
            count > 0 && best.to_bits() == TEXTURED_PIXEL_SHARPNESS_FLOOR.to_bits()
        }) {
            // The amended Textured_Pixel boundary is inclusive.  This property
            // does not compute the obsolete confidence ratio over every opaque
            // pixel; it only makes the fixed evidence boundary explicit in the
            // station draws used by the low-sharpness classifier.
            let exact_floor_is_textured = evidence.iter().any(|&(best, _, count)| {
                count > 0
                    && best >= TEXTURED_PIXEL_SHARPNESS_FLOOR
                    && best.to_bits() == TEXTURED_PIXEL_SHARPNESS_FLOOR.to_bits()
            });
            prop_assert!(
                exact_floor_is_textured,
                "a score exactly at the fixed floor must be a Textured_Pixel"
            );
        }
        // Non-vacuity is checked against deterministic local witnesses, not a
        // process-global cursor whose value depends on persisted regression
        // replay and parallel test scheduling.
        assert_local_low_sharpness_coverage()?;
    }
}

/// The production epsilon is deliberately restated here rather than imported:
/// this oracle is derived from 需求 3.9's formula and must fail if the production
/// definition changes independently.
const ORACLE_CONFIDENCE_SCALE_EPSILON: f64 = 1e-6;

/// Independently compute 需求 3.9 for one ownership cell from candidate evidence.
/// `None` is a frame whose valid projection does not cover this cell.
fn sharpness_confidence_oracle(scores: &[Option<f64>]) -> f32 {
    let mut covered = scores.iter().flatten().copied().collect::<Vec<_>>();
    covered.sort_by(|left, right| right.total_cmp(left));
    if covered.len() <= 1 {
        return 0.0;
    }
    let winner = covered[0];
    let runner_up = covered[1];
    (((winner - runner_up) / winner.max(ORACLE_CONFIDENCE_SCALE_EPSILON)).clamp(0.0, 1.0)) as f32
}

/// Drive one cell through the production renderer's `StationFusion` seed/fold
/// sequence.  No assertion in Property 15 calls the copied scalar helper.
fn production_cell_confidence(scores: &[Option<f64>]) -> (f32, u16) {
    assert!(!scores.is_empty());
    let mut fusion = focus_fuser::StationFusion::new(8, 8, scores.len());
    for (label, score) in scores.iter().copied().enumerate() {
        let mut candidate = focus_fuser::CandidateCells::new(1);
        if let Some(score) = score {
            candidate.covered[0] = true;
            candidate.sharpness[0] = score;
        }
        if label == 0 {
            fusion.seed(label, &candidate.sharpness, &candidate.covered);
        } else {
            fusion.fold(label, &candidate);
        }
    }
    (fusion.confidence()[0], fusion.owners()[0])
}

/// Preserve `winner` and `runner_up` as the two highest order statistics while
/// varying the total number of candidates.
fn confidence_scores(winner: f64, runner_up: f64, candidate_count: usize) -> Vec<Option<f64>> {
    let mut scores = Vec::with_capacity(candidate_count);
    // Put the runner-up first so tracking the winner must work across a fold,
    // rather than succeeding merely because the seed happened to be best.
    scores.push(Some(runner_up));
    scores.push(Some(winner));
    for extra in 2..candidate_count {
        scores.push(Some(runner_up * extra as f64 / candidate_count as f64));
    }
    scores
}

/// Cases Property 15 is configured with.
const CONFIDENCE_CASES: usize = 100;

proptest! {
    #![proptest_config(ProptestConfig { cases: CONFIDENCE_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 15: 对于任意 ownership
    // 单元的候选 Sharpness_Score 集合，该单元的 Sharpness_Confidence 落在 `[0, 1]` 内，
    // 随胜出与次优 Sharpness_Score 之差单调不减，且当候选数为 1 时恰为 0。
    //
    // Every measured value comes from `StationFusion::confidence()` after the same
    // seed/fold sequence used by `focus_stack_stitcher_with_margin_policy`.  The
    // independent oracle sorts only the drawn covered scores and applies the
    // requirement formula; it reads no production state and calls no confidence
    // helper.
    //
    // 「随差单调不减」is asserted in the two exact ways permitted by the common
    // gradient scale: hold the runner-up fixed and raise the winner, or hold the
    // winner fixed and lower the runner-up.  A bare gap whose winner and scale
    // both change is intentionally not treated as comparable.
    //
    // **Validates: Requirements 3.9**
    #[test]
    fn sharpness_confidence_is_bounded_monotone_and_zero_for_a_single_candidate(
        draw in arb_station_draw(),
        margin_pairs in prop::collection::vec((0u32..=1_000, 0u32..=1_000), 8),
        winner_raise in 0u32..=1_000,
        runner_lower in 0u32..=1_000,
        candidate_count in 2usize..=8,
    ) {
        // 1. Compare the complete drawn production grid with the independent
        //    requirement oracle, including randomly uncovered and one-candidate
        //    cells.
        let fusion = fuse_station(&draw);
        let measured_grid = fusion.confidence();
        prop_assert_eq!(measured_grid.len(), draw.geometry.cell_count());
        for (cell, &measured) in measured_grid.iter().enumerate() {
            let scores = draw
                .frames
                .iter()
                .map(|frame| frame.covered[cell].then_some(frame.sharpness[cell]))
                .collect::<Vec<_>>();
            let expected = sharpness_confidence_oracle(&scores);
            prop_assert_eq!(
                measured.to_bits(),
                expected.to_bits(),
                "需求 3.9 on cell {}: evidence {:?} gives {}, production published {}",
                cell,
                scores,
                expected,
                measured
            );
            prop_assert!((0.0..=1.0).contains(&measured));
        }

        // 2. Range and both coordinate-wise monotonic margin changes, all through
        //    the production seed/fold path and all checked against the oracle.
        for &(left, right) in &margin_pairs {
            let winner = f64::from(left.max(right)) / 1_000.0;
            let runner_up = f64::from(left.min(right)) / 1_000.0;
            let base_scores = confidence_scores(winner, runner_up, candidate_count);
            let (base, base_owner) = production_cell_confidence(&base_scores);
            prop_assert_ne!(base_owner, focus_fuser::NO_OWNER);
            prop_assert_eq!(
                base.to_bits(),
                sharpness_confidence_oracle(&base_scores).to_bits()
            );
            prop_assert!((0.0..=1.0).contains(&base));

            // Fixed runner-up, raised winner: both the raw margin and confidence
            // are non-decreasing.  Equality at winner == 1 is intentional.
            let raised_winner = winner + (1.0 - winner) * f64::from(winner_raise) / 1_000.0;
            let raised_scores = confidence_scores(raised_winner, runner_up, candidate_count);
            let (raised, _) = production_cell_confidence(&raised_scores);
            prop_assert!(raised_winner - runner_up >= winner - runner_up);
            prop_assert!(
                raised >= base,
                "raising winner {winner} -> {raised_winner} at runner-up {runner_up} widened the \
                 margin but lowered production confidence {base} -> {raised}"
            );
            prop_assert_eq!(
                raised.to_bits(),
                sharpness_confidence_oracle(&raised_scores).to_bits()
            );

            // Fixed winner, lowered runner-up: likewise non-decreasing.
            let lowered_runner = runner_up * (1.0 - f64::from(runner_lower) / 1_000.0);
            let lowered_scores = confidence_scores(winner, lowered_runner, candidate_count);
            let (lowered, _) = production_cell_confidence(&lowered_scores);
            prop_assert!(winner - lowered_runner >= winner - runner_up);
            prop_assert!(
                lowered >= base,
                "lowering runner-up {runner_up} -> {lowered_runner} at winner {winner} widened the \
                 margin but lowered production confidence {base} -> {lowered}"
            );
            prop_assert_eq!(
                lowered.to_bits(),
                sharpness_confidence_oracle(&lowered_scores).to_bits()
            );
        }

        // 3. Mandatory boundaries and equality cases run on every proptest case,
        //    so they cannot depend on random generation luck.
        let boundaries = [
            ("uncovered", vec![None, None], 0.0f32, true),
            ("one candidate in a multi-frame station", vec![None, Some(0.75), None], 0.0, false),
            ("one-frame station", vec![Some(0.75)], 0.0, false),
            ("zero equality", vec![Some(0.0), Some(0.0)], 0.0, false),
            ("positive equality", vec![Some(0.65), Some(0.65)], 0.0, false),
            ("full-scale margin", vec![Some(1.0), Some(0.0)], 1.0, false),
            ("below epsilon scale", vec![Some(0.5e-6), Some(0.0)], 0.5, false),
            ("epsilon scale", vec![Some(1e-6), Some(0.0)], 1.0, false),
        ];
        for (name, scores, expected, uncovered) in boundaries {
            let (measured, owner) = production_cell_confidence(&scores);
            prop_assert_eq!(
                measured.to_bits(),
                expected.to_bits(),
                "{}: production published {}, expected {}",
                name,
                measured,
                expected
            );
            prop_assert_eq!(
                measured.to_bits(),
                sharpness_confidence_oracle(&scores).to_bits(),
                "{}: independent requirement oracle disagreed",
                name
            );
            prop_assert_eq!(owner == focus_fuser::NO_OWNER, uncovered);
            prop_assert!((0.0..=1.0).contains(&measured));
        }

        // The comparison the requirement does not make: when the winner changes,
        // so does the common gradient scale.  Pin this distinction through the
        // production path, not through the scalar implementation helper.
        let (large_absolute_gap, _) =
            production_cell_confidence(&[Some(0.9), Some(0.5)]);
        let (small_absolute_gap, _) =
            production_cell_confidence(&[Some(0.2), Some(0.1)]);
        prop_assert!(large_absolute_gap < small_absolute_gap);
    }
}

// ---------------------------------------------------------------------------
// Property 16 harness: a Capture_Station of one registration-successful frame
// ---------------------------------------------------------------------------
//
// Two levels are asserted, because 需求 3.10 has two halves and they live in
// different places:
//
//   - "跳过图割求解" is a property of `focus_fuser`: the short circuit sits ahead
//     of every budget check and of every cut, and `StationFusion` reports a
//     graph cut duration of exactly `0` because `fold` never runs.
//   - "Ownership_Map 中全部已覆盖像素都指向该 Source_RAW 标识" is a property of the
//     published Virtual_Tile, so it is asserted on
//     `stitching::focus_stack_stitcher_unfilled` — the function the default
//     layered compositor calls once per Capture_Station.

/// Side of the synthetic Source_RAW one-frame stations are rendered from.
///
/// Large enough that the station plane carries several ownership cells and that
/// `render_focus_layer`'s two pixel border leaves an interior, small enough that
/// a float RGB layer is a few hundred kilobytes.
const SINGLE_FRAME_SOURCE_SIDE: u32 = 96;

/// A decodable synthetic Source_RAW: the aperiodic probe texture as float RGB.
fn single_frame_source(side: u32, seed: u64, detail: f64) -> image::Rgb32FImage {
    image::Rgb32FImage::from_fn(side, side, |x, y| {
        probe_texture(f64::from(x), f64::from(y), seed, detail)
    })
}

/// `ImageInfo` of one synthetic Source_RAW, with no feature or foreground
/// evidence: a Capture_Station of one frame has nothing to register against.
fn single_frame_info(id: usize, side: u32) -> ImageInfo {
    ImageInfo {
        id,
        filename: format!("/station/single/{id:04}.NEF"),
        width: side,
        height: side,
        alignment_image: image::GrayImage::new(1, 1),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: Some(75.0),
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

/// Render one Capture_Station through the production Virtual_Tile path.
///
/// This is `stitching::focus_stack_stitcher_unfilled` with the arguments the
/// default layered compositor passes it (`panorama_stitching.rs`): no focus warp,
/// no capture group ids, `sequence_gap_aware = false` — which is what keeps the
/// render on the standard focus fusion branch instead of the shifted mosaic.
fn render_station_tile(
    infos: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    sources: &[image::Rgb32FImage],
) -> Result<stitching::FocusStackTileRender, String> {
    let app = tauri::test::mock_app();
    let refs = infos.iter().collect::<Vec<_>>();
    let mut load = |info: &ImageInfo| {
        sources
            .get(info.id)
            .cloned()
            .ok_or_else(|| format!("no synthetic source for {}", info.filename))
    };
    intra_station::reset_run_records();
    stitching::focus_stack_stitcher_unfilled(
        &refs,
        homographies,
        stitching::Projection::Planar,
        None,
        None,
        0,
        false,
        app.handle().clone(),
        "stack-pipeline-property-progress",
        &mut load,
    )
}

/// Bit 0: the zero budget contrast ran, i.e. a two-candidate grid did fall back
/// while the one-candidate grid did not.  Bit 1: the single frame left cells
/// uncovered.  Bit 2: the published Virtual_Tile was asserted.
static SINGLE_FRAME_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 16 so far.
static SINGLE_FRAME_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 16 is configured with.
///
/// Every case exercises the cheap solver contrast; ten cases render the
/// production Virtual_Tile twice, which covers repeat determinism without
/// dominating the complete `cargo test --lib` budget.
const SINGLE_FRAME_CASES: usize = 100;

/// Cases of Property 16 that also render the production Virtual_Tile twice.
const SINGLE_FRAME_RENDER_STRIDE: usize = 10;

proptest! {
    #![proptest_config(ProptestConfig { cases: SINGLE_FRAME_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 16: 对于任意只包含 1 张
    // 配准成功 Source_RAW 的 Capture_Station，不执行图割求解，且 Ownership_Map 中全部
    // 已覆盖像素都指向该 Source_RAW 标识。
    //
    // 「不执行图割求解」用三条互相独立的观测断言，而不是只看一个状态枚举：
    //   1. `StationFusion` 报告的图割耗时恰好是 `0.00 s`（位模式比较），因为 `fold`
    //      一次都没有被调用；
    //   2. 求解状态是 `SingleFrame` 而不是 `GraphCut`，也不是需求 3.5 的降级状态；
    //   3. **零预算对照**：把预算设成 `Duration::ZERO` 再求解。任何真的进了割的路径都会
    //      在第一次预算检查上翻成 `PerCellFallback`（同一条断言在 2 候选的网格上确认了这一点），
    //      而 1 候选的网格仍然返回 `SingleFrame`——短路发生在预算检查**之前**，
    //      也就是说那条路上根本没有割。
    //
    // 「全部已覆盖像素都指向该 Source_RAW」在生产的 Virtual_Tile 上断言：
    // `stitching::focus_stack_stitcher_unfilled` 是默认分层合成器每个机位各调用一次的
    // 那个函数，断言 Coverage_Mask 与 Ownership_Map 逐像素互为条件、owner 标识恒为该帧、
    // 溯源 legend 只有一条、并且报告里的求解状态与图割耗时与上面一致。
    //
    // **Validates: Requirements 3.10**
    #[test]
    fn a_single_frame_station_skips_the_cut_and_owns_every_covered_pixel(
        draw in arb_station_draw(),
        seed in any::<u64>(),
        detail in 0.05f64..0.6,
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let case = SINGLE_FRAME_CURSOR.fetch_add(1, Ordering::Relaxed);
        let mut branches = 0usize;

        // 1. The fuser: one frame, seeded, never folded.
        let single = StationDraw {
            plane: draw.plane,
            geometry: draw.geometry,
            frames: vec![draw.frames[0].clone()],
        };
        let fusion = fuse_station(&single);
        prop_assert_eq!(
            fusion.solver_status(),
            FusionSolverStatus::SingleFrame,
            "需求 3.10: a station of one Source_RAW reports the single-frame path"
        );
        prop_assert_eq!(
            fusion.graph_cut_seconds().to_bits(),
            0.0f64.to_bits(),
            "需求 3.10: no cut ran, so the reported graph cut duration is exactly 0, not {}",
            fusion.graph_cut_seconds()
        );
        let owner = focus_fuser::owner_of(0);
        let mut uncovered = 0usize;
        for cell in 0..single.geometry.cell_count() {
            if single.frames[0].covered[cell] {
                prop_assert_eq!(
                    fusion.owners()[cell],
                    owner,
                    "需求 3.10: covered cell {} must point at the only Source_RAW",
                    cell
                );
            } else {
                uncovered += 1;
                prop_assert_eq!(
                    fusion.owners()[cell],
                    focus_fuser::NO_OWNER,
                    "需求 3.7: cell {} is outside the frame's projection",
                    cell
                );
            }
        }
        if uncovered > 0 {
            branches |= 2;
        }

        // 2. The zero budget contrast: the short circuit sits ahead of the budget
        //    check, so it cannot be a cut that finished quickly.
        let cells = single.geometry.cell_count().min(9);
        let mut one_candidate = focus_fuser::OwnershipCostGrid::new(cells, 1, 1);
        let mut two_candidates = focus_fuser::OwnershipCostGrid::new(cells, 1, 2);
        for cell in 0..cells {
            let score = f64::from(cell as u32 % 5) / 5.0;
            one_candidate.set_cost(cell, 0, Some(focus_fuser::data_cost(score, 0.0)));
            two_candidates.set_cost(cell, 0, Some(focus_fuser::data_cost(score, 0.0)));
            two_candidates.set_cost(cell, 1, Some(focus_fuser::data_cost(1.0 - score, 0.0)));
            one_candidate.set_disagreement(cell, 0.02);
            two_candidates.set_disagreement(cell, 0.02);
        }
        let clock = focus_fuser::MonotonicGraphCutClock::start();
        let starved = focus_fuser::solve_ownership_with_clock(
            &one_candidate,
            Duration::ZERO,
            &clock,
        );
        prop_assert_eq!(
            starved.solver_status,
            FusionSolverStatus::SingleFrame,
            "需求 3.10: one candidate short circuits before 需求 3.5's budget is even consulted"
        );
        prop_assert_eq!(
            &starved.owners,
            &vec![owner; cells],
            "the short circuit still hands every covered cell to that Source_RAW"
        );
        let starved_pair = focus_fuser::solve_ownership_with_clock(
            &two_candidates,
            Duration::ZERO,
            &clock,
        );
        prop_assert_eq!(
            starved_pair.solver_status,
            FusionSolverStatus::PerCellFallback,
            "the same zero budget does stop a two candidate solve, which is what makes the \
             single-frame result evidence that no cut ran"
        );
        prop_assert!(
            focus_fuser::skips_graph_cut(0) && focus_fuser::skips_graph_cut(1)
                && !focus_fuser::skips_graph_cut(2),
            "需求 3.10's premise is exactly 「至多一张配准成功的帧」"
        );
        branches |= 1;

        // 3. The published Virtual_Tile of a real one-frame render.
        if case % SINGLE_FRAME_RENDER_STRIDE == 0 {
            let side = SINGLE_FRAME_SOURCE_SIDE;
            let info = single_frame_info(0, side);
            let infos = vec![info];
            let homographies = HashMap::from([(0usize, Matrix3::identity())]);
            let sources = vec![single_frame_source(side, seed, detail)];
            let rendered = render_station_tile(&infos, &homographies, &sources)
                .map_err(|error| TestCaseError::fail(format!("the render must succeed: {error}")))?;
            let repeated = render_station_tile(&infos, &homographies, &sources)
                .map_err(|error| TestCaseError::fail(format!("the repeat must succeed: {error}")))?;
            let masks = rendered.masks.as_ref().ok_or_else(|| {
                TestCaseError::fail(
                    "需求 3.7 / 3.11: the unfilled Virtual_Tile form must publish its \
                     Coverage_Mask and Ownership_Map",
                )
            })?;
            let repeated_masks = repeated.masks.as_ref().ok_or_else(|| {
                TestCaseError::fail("the repeated production render must publish its masks")
            })?;
            let (width, height) = masks.ownership.dimensions();
            prop_assert_eq!((width, height), rendered.image.dimensions());
            prop_assert_eq!((width, height), sources[0].dimensions());
            prop_assert_eq!(masks.coverage.dimensions(), (width, height));
            prop_assert_eq!(masks.confidence.dimensions(), (width, height));
            prop_assert_eq!(
                masks.ownership.legend(),
                &[std::path::PathBuf::from(&infos[0].filename)],
                "a station of one Source_RAW has exactly that source in its legend"
            );
            prop_assert_eq!(
                masks.fusion.solver_status,
                FusionSolverStatus::SingleFrame,
                "需求 3.10 must reach the Stack_Report of a real render"
            );
            prop_assert_eq!(
                masks.fusion.graph_cut_seconds.to_bits(),
                0.0f64.to_bits(),
                "需求 3.10: the reported graph cut time of a one-frame station is 0.00 s, not {}",
                masks.fusion.graph_cut_seconds
            );
            let mut covered_pixels = 0u64;
            let mut uncovered_pixels = 0u64;
            for y in 0..height {
                for x in 0..width {
                    let covered = masks.coverage.is_covered(x, y);
                    let owner_id = masks.ownership.owner_at(x, y);
                    prop_assert_eq!(
                        masks.confidence.value_at(x, y).to_bits(),
                        0.0f32.to_bits(),
                        "需求 3.9: a single candidate has zero confidence at ({}, {})",
                        x,
                        y
                    );
                    if covered {
                        covered_pixels += 1;
                        prop_assert_eq!(
                            owner_id,
                            1u16,
                            "需求 3.10: covered pixel ({}, {}) must point at the only Source_RAW",
                            x,
                            y
                        );
                        for channel in 0..3 {
                            prop_assert_eq!(
                                rendered.image.get_pixel(x, y)[channel].to_bits(),
                                sources[0].get_pixel(x, y)[channel].to_bits(),
                                "需求 3.10: covered output ({}, {}) channel {} must be a \
                                 bit-for-bit copy of its sole source",
                                x,
                                y,
                                channel
                            );
                        }
                    } else {
                        uncovered_pixels += 1;
                        prop_assert_eq!(
                            owner_id,
                            0u16,
                            "需求 3.7: uncovered pixel ({}, {}) keeps the reserved identifier",
                            x,
                            y
                        );
                        prop_assert!(
                            rendered.image.get_pixel(x, y).0.iter().all(|&value| value == 0.0),
                            "uncovered backing pixel ({x}, {y}) must remain empty"
                        );
                    }
                }
            }
            prop_assert_eq!(
                covered_pixels,
                masks.coverage.covered_pixels(),
                "the iterated coverage count must agree with Coverage_Mask"
            );
            prop_assert_eq!(
                covered_pixels,
                masks.ownership.assigned_pixels(),
                "需求 3.11: the Coverage_Mask and the Ownership_Map must agree pixel by pixel"
            );
            prop_assert_eq!(masks.fusion.owned_pixels, covered_pixels);
            prop_assert_eq!(masks.fusion.uncovered_pixels, uncovered_pixels);
            prop_assert!(
                covered_pixels > u64::from(width * height) / 2,
                "the fixture must really cover the tile: {covered_pixels} of {} pixels",
                width * height
            );
            prop_assert!(
                uncovered_pixels > 0,
                "the production render must exercise uncovered border pixels"
            );

            // Same build, platform, input and parameters: the real station path
            // must repeat bit-for-bit, including every published semantic plane.
            prop_assert_eq!(rendered.image.as_raw(), repeated.image.as_raw());
            prop_assert_eq!(&masks.coverage, &repeated_masks.coverage);
            prop_assert_eq!(&masks.ownership, &repeated_masks.ownership);
            prop_assert_eq!(&masks.confidence, &repeated_masks.confidence);
            prop_assert_eq!(
                repeated_masks.fusion.solver_status,
                FusionSolverStatus::SingleFrame
            );
            prop_assert_eq!(
                repeated_masks.fusion.graph_cut_seconds.to_bits(),
                0.0f64.to_bits()
            );
            prop_assert_eq!(repeated_masks.fusion.owned_pixels, covered_pixels);
            prop_assert_eq!(repeated_masks.fusion.uncovered_pixels, uncovered_pixels);
            branches |= 4;
        }

        let covered = SINGLE_FRAME_BRANCHES_SEEN.fetch_or(branches, Ordering::Relaxed) | branches;
        prop_assert!(
            case + 1 < SINGLE_FRAME_CASES || covered == 0b111,
            "the run must have contrasted the zero budget paths, seen a station that leaves cells \
             uncovered and asserted a published Virtual_Tile; mask {covered:#b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Property 75 harness: every non-anchor registration fails
// ---------------------------------------------------------------------------

const SINGLE_FRAME_DEGRADATION_CASES: usize = 100;
const SINGLE_FRAME_DEGRADATION_RENDER_STRIDE: usize = 10;
const SINGLE_FRAME_DEGRADATION_SIDE: u32 = 64;
static SINGLE_FRAME_DEGRADATION_CURSOR: AtomicUsize = AtomicUsize::new(0);
static SINGLE_FRAME_DEGRADATION_BRANCHES: AtomicUsize = AtomicUsize::new(0);

fn requirement_12_1_source(
    members: &[super::station_degradation::StationMember],
) -> Option<String> {
    let measurable_maximum = members
        .iter()
        .map(|member| member.median_sharpness)
        .filter(|score| score.is_finite())
        .max_by(f64::total_cmp);
    members
        .iter()
        .filter(|member| {
            measurable_maximum.is_none_or(|maximum| {
                member.median_sharpness.is_finite()
                    && member.median_sharpness.total_cmp(&maximum).is_eq()
            })
        })
        .map(|member| member.path.clone())
        .min()
}

fn degradation_alignment(side: u32, sharp: bool, phase: u32) -> image::GrayImage {
    image::GrayImage::from_fn(side, side, |x, y| {
        let value = if sharp {
            if ((x + phase) / 2 + (y + phase) / 2).is_multiple_of(2) {
                224
            } else {
                32
            }
        } else {
            128
        };
        image::Luma([value])
    })
}

fn degraded_station_info(
    id: usize,
    path: String,
    side: u32,
    alignment_image: image::GrayImage,
) -> ImageInfo {
    ImageInfo {
        id,
        filename: path,
        width: side,
        height: side,
        alignment_image,
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: Some(75.0),
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

fn render_failed_non_anchor_station(
    infos: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    sources: &[image::Rgb32FImage],
) -> Result<
    (
        stitching::FocusStackTileRender,
        Vec<String>,
        DegradationLedger,
    ),
    String,
> {
    let _run_scope = degradation::begin_run_scope();
    let app = tauri::test::mock_app();
    let refs = infos.iter().collect::<Vec<_>>();
    let anchor_path = infos
        .first()
        .map(|info| info.filename.as_str())
        .unwrap_or_default();
    intra_station::reset_run_records();
    degradation::reset_run_ledger();
    for info in infos.iter().skip(1) {
        intra_station::record_run_frame(
            0,
            anchor_path,
            super::report::IntraStationFrameRecord {
                path: info.filename.clone(),
                status: super::report::IntraStationFrameStatus::Failed,
                ..Default::default()
            },
        );
    }
    let mut loaded = Vec::new();
    let mut load = |info: &ImageInfo| {
        loaded.push(info.filename.clone());
        sources
            .get(info.id)
            .cloned()
            .ok_or_else(|| format!("no synthetic source for {}", info.filename))
    };
    let rendered = stitching::focus_stack_stitcher_unfilled(
        &refs,
        homographies,
        stitching::Projection::Planar,
        None,
        None,
        0,
        false,
        app.handle().clone(),
        "stack-pipeline-property-75-progress",
        &mut load,
    )?;
    Ok((rendered, loaded, degradation::run_ledger_snapshot()))
}

fn assert_degraded_station_render(
    rendered: &stitching::FocusStackTileRender,
    loaded: &[String],
    ledger: &DegradationLedger,
    infos: &[ImageInfo],
    selected_source: &image::Rgb32FImage,
) -> Result<(), TestCaseError> {
    let selected_path = &infos[0].filename;
    prop_assert_eq!(
        loaded,
        std::slice::from_ref(selected_path),
        "需求 12.1: excluded frames must never be loaded or offered to GraphCut"
    );
    let masks = rendered.masks.as_ref().ok_or_else(|| {
        TestCaseError::fail("需求 12.1: the production Virtual_Tile must publish ownership")
    })?;
    prop_assert_eq!(
        masks.fusion.solver_status,
        FusionSolverStatus::SingleFrame,
        "需求 12.1: the one-source degraded plan must skip GraphCut"
    );
    prop_assert_eq!(masks.fusion.graph_cut_seconds.to_bits(), 0.0f64.to_bits());

    let degradation_entries = ledger
        .entries()
        .iter()
        .filter(|entry| entry.reason == degradation::FUSION_SINGLE_FRAME_DEGRADED)
        .collect::<Vec<_>>();
    prop_assert_eq!(
        degradation_entries.len(),
        1,
        "Stack_Report must contain one stable single-frame degradation entry"
    );
    let detail = &degradation_entries[0].detail;
    prop_assert_eq!(detail["fused_frames"].as_u64(), Some(1));
    prop_assert_eq!(
        detail["single_frame_path"].as_str(),
        Some(selected_path.as_str())
    );
    let excluded = detail["excluded"].as_array().ok_or_else(|| {
        TestCaseError::fail("the production degradation entry must list excluded sources")
    })?;
    prop_assert_eq!(excluded.len(), infos.len() - 1);
    let reported = excluded
        .iter()
        .map(|record| {
            (
                record["path"].as_str().unwrap_or_default().to_string(),
                record["reason"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect::<BTreeSet<_>>();
    let expected = infos
        .iter()
        .skip(1)
        .map(|info| {
            (
                info.filename.clone(),
                degradation::INTRA_STATION_REGISTRATION_FAILED.to_string(),
            )
        })
        .collect::<BTreeSet<_>>();
    prop_assert_eq!(reported, expected);

    let (width, height) = masks.ownership.dimensions();
    let mut covered = 0u64;
    for y in 0..height {
        for x in 0..width {
            if masks.coverage.is_covered(x, y) {
                covered += 1;
                let owner = masks.ownership.owner_at(x, y);
                prop_assert_ne!(owner, stitching::NO_OWNER);
                prop_assert_eq!(
                    masks.ownership.legend()[owner as usize - 1].as_path(),
                    Path::new(selected_path),
                    "every covered production pixel must name the deterministic source"
                );
                for channel in 0..3 {
                    prop_assert_eq!(
                        rendered.image.get_pixel(x, y)[channel].to_bits(),
                        selected_source.get_pixel(x, y)[channel].to_bits(),
                        "source-ownership invariant failed at ({}, {}) channel {}",
                        x,
                        y,
                        channel
                    );
                }
            } else {
                prop_assert_eq!(masks.ownership.owner_at(x, y), stitching::NO_OWNER);
            }
        }
    }
    prop_assert!(
        covered > 0,
        "the production ownership assertion must be non-vacuous"
    );
    prop_assert_eq!(covered, masks.ownership.assigned_pixels());
    prop_assert_eq!(covered, masks.fusion.owned_pixels);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: SINGLE_FRAME_DEGRADATION_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 75: 对于任意全部非锚点帧
    // 都被标记为配准失败的 Capture_Station，该机位只使用有效面积内中位
    // Sharpness_Score 最高的单张 Source_RAW；完全相同时取绝对路径字典序最小者，
    // Ownership_Map 全部指向它，并把单帧降级及每个排除项写入 Stack_Report。
    //
    // **Validates: Requirements 12.1**
    #[test]
    fn every_failed_non_anchor_degrades_to_one_deterministic_production_source(
        scores in prop::collection::vec(prop_oneof![1 => Just(None), 5 => (0u16..=1_000).prop_map(Some)], 2..=6),
        permutation_seed in any::<usize>(),
        texture_seed in any::<u64>(),
        exact_tie in any::<bool>(),
    ) {
        let case = SINGLE_FRAME_DEGRADATION_CURSOR.fetch_add(1, Ordering::Relaxed);
        let mut members = scores
            .iter()
            .enumerate()
            .map(|(index, score)| super::station_degradation::StationMember {
                path: format!("/station/property-75/{:04}-{index}.NEF", scores.len() - index),
                median_sharpness: score.map_or(f64::NAN, |value| f64::from(value) / 1_000.0),
                status: super::report::IntraStationFrameStatus::Failed,
            })
            .collect::<Vec<_>>();
        if exact_tie {
            let maximum = members
                .iter()
                .map(|member| member.median_sharpness)
                .filter(|score| score.is_finite())
                .max_by(f64::total_cmp)
                .unwrap_or(0.5);
            members[0].median_sharpness = maximum;
            members[1].median_sharpness = maximum;
        }

        let expected_path = requirement_12_1_source(&members)
            .expect("the generated station is non-empty");
        let plan = super::station_degradation::plan_station_fusion(&members);
        prop_assert_eq!(
            plan.mode,
            super::station_degradation::StationFusionMode::SingleFrameDegraded
        );
        prop_assert_eq!(plan.fused.len(), 1);
        prop_assert_eq!(
            plan.single_frame_path(&members),
            Some(expected_path.clone())
        );
        prop_assert_eq!(plan.excluded.len(), members.len() - 1);
        let every_excluded_source_is_reported = plan.excluded.iter().all(|record| {
            record.reason == degradation::INTRA_STATION_REGISTRATION_FAILED
                && record.path != expected_path
        });
        prop_assert!(every_excluded_source_is_reported);

        let shift = permutation_seed % members.len();
        members.rotate_left(shift);
        members.reverse();
        let reordered = super::station_degradation::plan_station_fusion(&members);
        prop_assert_eq!(
            reordered.single_frame_path(&members),
            Some(expected_path.clone()),
            "score/path selection must not depend on import order"
        );

        // Mandatory empty/unmeasurable and exact-tie cases: these run in every
        // generated case instead of depending on random draws.
        prop_assert_eq!(super::station_degradation::single_frame_source(&[]), None);
        let edge_members = vec![
            super::station_degradation::StationMember {
                path: "/z/unmeasurable.NEF".to_string(),
                median_sharpness: f64::NAN,
                status: super::report::IntraStationFrameStatus::Failed,
            },
            super::station_degradation::StationMember {
                path: "/a/unmeasurable.NEF".to_string(),
                median_sharpness: f64::INFINITY,
                status: super::report::IntraStationFrameStatus::Failed,
            },
        ];
        let edge_plan = super::station_degradation::plan_station_fusion(&edge_members);
        prop_assert_eq!(
            edge_plan.single_frame_path(&edge_members),
            Some("/a/unmeasurable.NEF".to_string())
        );
        let tie_members = vec![
            super::station_degradation::StationMember {
                path: "/z/tie.NEF".to_string(),
                median_sharpness: 0.75,
                status: super::report::IntraStationFrameStatus::Failed,
            },
            super::station_degradation::StationMember {
                path: "/a/tie.NEF".to_string(),
                median_sharpness: 0.75,
                status: super::report::IntraStationFrameStatus::Failed,
            },
        ];
        let tie_plan = super::station_degradation::plan_station_fusion(&tie_members);
        prop_assert_eq!(
            tie_plan.single_frame_path(&tie_members),
            Some("/a/tie.NEF".to_string())
        );

        let mut local_ledger = DegradationLedger::new();
        super::station_degradation::record_entries(
            &mut local_ledger,
            &super::station_degradation::station_plan_entries(17, &members, &reordered),
        );
        prop_assert_eq!(
            local_ledger.count_of(degradation::FUSION_SINGLE_FRAME_DEGRADED),
            1,
            "the stable failure identifier must reach the report ledger"
        );

        let mut branches = usize::from(exact_tie);
        if case % SINGLE_FRAME_DEGRADATION_RENDER_STRIDE == 0 {
            let selected_original = scores
                .iter()
                .enumerate()
                .map(|(index, _)| index)
                .min_by_key(|&index| {
                    if exact_tie {
                        format!("{:04}-{index}", scores.len() - index)
                    } else if index == 0 {
                        String::new()
                    } else {
                        format!("z{index:04}")
                    }
                })
                .expect("at least two sources");
            let mut source_order = (0..scores.len()).collect::<Vec<_>>();
            source_order.retain(|&index| index != selected_original);
            source_order.rotate_left(permutation_seed % (scores.len() - 1));
            source_order.insert(0, selected_original);

            let side = SINGLE_FRAME_DEGRADATION_SIDE;
            let sources = (0..scores.len())
                .map(|index| single_frame_source(side, texture_seed ^ index as u64, 0.12 + index as f64 * 0.01))
                .collect::<Vec<_>>();
            let paths = (0..scores.len())
                .map(|index| {
                    if exact_tie {
                        format!("/station/property-75/{:04}-{index}.NEF", scores.len() - index)
                    } else if index == selected_original {
                        "/station/property-75/selected.NEF".to_string()
                    } else {
                        format!("/station/property-75/failed-{index:04}.NEF")
                    }
                })
                .collect::<Vec<_>>();
            let make_infos = |order: &[usize]| {
                order
                    .iter()
                    .map(|&index| {
                        degraded_station_info(
                            index,
                            paths[index].clone(),
                            side,
                            degradation_alignment(side, exact_tie || index == selected_original, index as u32),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let homographies = (0..scores.len())
                .map(|index| (index, Matrix3::identity()))
                .collect::<HashMap<_, _>>();
            let infos = make_infos(&source_order);
            let first = render_failed_non_anchor_station(&infos, &homographies, &sources)
                .map_err(|error| TestCaseError::fail(format!("degraded production render failed: {error}")))?;
            assert_degraded_station_render(&first.0, &first.1, &first.2, &infos, &sources[selected_original])?;

            source_order[1..].reverse();
            let reversed_infos = make_infos(&source_order);
            let second = render_failed_non_anchor_station(&reversed_infos, &homographies, &sources)
                .map_err(|error| TestCaseError::fail(format!("reordered production render failed: {error}")))?;
            assert_degraded_station_render(&second.0, &second.1, &second.2, &reversed_infos, &sources[selected_original])?;
            prop_assert_eq!(first.0.image.as_raw(), second.0.image.as_raw());
            prop_assert_eq!(
                first.0.masks.as_ref().map(|masks| masks.ownership.owners()),
                second.0.masks.as_ref().map(|masks| masks.ownership.owners()),
                "reordering excluded imports must not change production ownership"
            );
            branches |= 2;
        }
        let seen = SINGLE_FRAME_DEGRADATION_BRANCHES.fetch_or(branches, Ordering::Relaxed) | branches;
        prop_assert!(
            case + 1 < SINGLE_FRAME_DEGRADATION_CASES || seen == 0b11,
            "Property 75 must exercise exact ties and full production renders; mask {seen:#b}"
        );
    }
}

// ---------------------------------------------------------------------------
