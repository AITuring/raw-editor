//! Station ownership grid properties (14, 15).
//!
//! Low-sharpness cells against the station percentile, and Sharpness_Confidence
//! bounds and monotonicity, read from the production StationFusion. The station
//! draws are shared with the single-frame properties.

use super::super::focus_fuser;
use proptest::prelude::*;

use super::unit_from;

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

/// Station plane sides Properties 14–16 fuse over, in native pixels.
///
/// Small on purpose.  Below 4,096px the ownership cell side is the 8px minimum
/// (Property 9), so these planes give a 6×6..12×12 cell grid: enough cells for a
/// percentile and for four-connected regions, few enough that a whole station
/// folds in microseconds.  The cell *geometry* is what Property 9 covers over
/// the full plane size range; what matters here is the number of cells.
pub(super) fn arb_station_plane() -> impl Strategy<Value = (u32, u32)> {
    (
        prop::sample::select(vec![48u32, 56, 64, 80, 96]),
        prop::sample::select(vec![48u32, 56, 64, 80, 96]),
    )
}

/// The per-cell measurements of one synthetic Capture_Station.
#[derive(Clone)]
pub(super) struct StationDraw {
    pub(super) plane: (u32, u32),
    pub(super) geometry: focus_fuser::OwnershipGridGeometry,
    pub(super) frames: Vec<focus_fuser::CandidateCells>,
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
pub(super) fn station_draw(
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
pub(super) fn arb_station_draw() -> impl Strategy<Value = StationDraw> {
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
pub(super) fn fuse_station(draw: &StationDraw) -> focus_fuser::StationFusion {
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
pub(super) fn station_cell_evidence(draw: &StationDraw) -> Vec<(f64, f64, usize)> {
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
pub(super) fn nearest_rank_tenth_percentile(sample: &[f64]) -> f64 {
    let mut sorted = sample.to_vec();
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len() - 1;
    let rank = ((last as f64) * 0.10).round() as usize;
    sorted[rank.min(last)]
}

/// Four-connected components of a boolean cell mask, counted independently of
/// `focus_fuser::low_sharpness_regions`.
pub(super) fn connected_component_count(columns: usize, rows: usize, mask: &[bool]) -> usize {
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
pub(super) const REQUIRED_LOW_SHARPNESS_PERCENTILE: f64 = 0.10;

/// Amended 需求 11.12: a covered pixel is textured when at least one candidate
/// in its ownership cell reaches this fixed Sharpness_Score floor.  This is a
/// separate absolute boundary from 需求 3.8's station-relative P10; keeping the
/// constant here prevents the property from accidentally reviving the obsolete
/// all-nontransparent-pixel statistic while it exercises shared boundary data.
pub(super) const TEXTURED_PIXEL_SHARPNESS_FLOOR: f64 = 0.10;

/// Build deterministic stations that exercise every non-vacuity branch of
/// Property 14 without sharing counters with other tests or persisted cases.
///
/// The rich station has two disconnected cells strictly below its nearest-rank
/// P10, cells exactly on P10 (including the inclusive Textured_Pixel floor),
/// ordinary non-low cells, and one uncovered cell. The other stations pin the
/// no-low and no-evidence outcomes that cannot coexist with those facts in one
/// station.
pub(super) fn low_sharpness_coverage_witnesses() -> [StationDraw; 3] {
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
pub(super) fn low_sharpness_branch_mask(draw: &StationDraw) -> usize {
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

pub(super) fn assert_local_low_sharpness_coverage() -> Result<(), TestCaseError> {
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
pub(super) const LOW_SHARPNESS_CASES: usize = 100;

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
pub(super) const ORACLE_CONFIDENCE_SCALE_EPSILON: f64 = 1e-6;

/// Independently compute 需求 3.9 for one ownership cell from candidate evidence.
/// `None` is a frame whose valid projection does not cover this cell.
pub(super) fn sharpness_confidence_oracle(scores: &[Option<f64>]) -> f32 {
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
pub(super) fn production_cell_confidence(scores: &[Option<f64>]) -> (f32, u16) {
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
pub(super) fn confidence_scores(
    winner: f64,
    runner_up: f64,
    candidate_count: usize,
) -> Vec<Option<f64>> {
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
pub(super) const CONFIDENCE_CASES: usize = 100;

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
