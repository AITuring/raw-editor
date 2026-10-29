//! Focus_Fuser ownership cell properties (9, 10, 11).
//!
//! Cell size and count, one shared sampling for candidate and composite
//! sharpness, and the low-pass disagreement with its mismatch penalty, measured
//! through the production focus fusion path.

use super::super::focus_fuser;
use crate::panorama_utils::mosaic;
use crate::panorama_utils::stitching;
use proptest::prelude::*;
use std::collections::BTreeSet;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

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
pub(super) const REQUIRED_CELL_MIN_PX: u32 = 8;

pub(super) const REQUIRED_CELL_MAX_PX: u32 = 64;

/// 需求 3.1: the smallest number of ownership cells along the station plane's
/// long side.
pub(super) const REQUIRED_CELLS_ALONG_LONG_SIDE: u32 = 512;

/// 需求 3.2: the Sharpness_Score window side, in native pixels.
pub(super) const REQUIRED_SHARPNESS_WINDOW_PX: f64 = 32.0;

/// 需求 3.2: probe positions per ownership cell — the centre and the four
/// corners.
pub(super) const REQUIRED_PROBES_PER_CELL: usize = 5;

/// 需求 3.3: the inconsistency above which a sharper candidate is penalised.
pub(super) const REQUIRED_DISAGREEMENT_VETO: f64 = 0.2;

/// 需求 3.3: the smallest penalty such a candidate carries.
pub(super) const REQUIRED_MIN_MISMATCH_PENALTY: f64 = 1.0;

/// Station plane sides, in native pixels.
///
/// The uniform range covers both sides of the 4,096px crossover the biconditional
/// below turns on, and the selected values pin it exactly, together with the real
/// 9,504px plane of the acceptance material and the 32,768px point where the 64px
/// ceiling starts to bind.
pub(super) fn arb_station_plane_side() -> impl Strategy<Value = u32> {
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
pub(super) static CELL_SIZE_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 9 so far.
pub(super) static CELL_SIZE_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 9 is configured with.  Pure integer geometry, so the cases are
/// free.
pub(super) const CELL_SIZE_CASES: usize = 256;

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
pub(super) fn probe_texture(x: f64, y: f64, seed: u64, detail: f64) -> image::Rgb<f32> {
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
pub(super) fn probe_layer(
    side: u32,
    seed: u64,
    detail: f64,
) -> (image::Rgb32FImage, image::GrayImage) {
    let image = image::Rgb32FImage::from_fn(side, side, |x, y| {
        probe_texture(f64::from(x), f64::from(y), seed, detail)
    });
    let mask = image::GrayImage::from_pixel(side, side, image::Luma([255]));
    (image, mask)
}

/// Bit 0: the candidate and the composite measured different scores through the
/// same sampling.  Bit 1: a foreign sampling plan was rejected.
pub(super) static SAMPLING_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 10 so far.
pub(super) static SAMPLING_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 10 is configured with.  One case measures two whole layers of
/// ownership cells through the production path and still costs well under a
/// millisecond, so the count is set by the fixture variety rather than by cost:
/// the drawn plane sizes cover the cell side range and the drawn layer sides move
/// the grid under the probes.
pub(super) const SAMPLING_CASES: usize = 64;

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
pub(super) static PENALTY_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 11 so far.
pub(super) static PENALTY_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 11 is configured with.
pub(super) const PENALTY_CASES: usize = 100;

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
