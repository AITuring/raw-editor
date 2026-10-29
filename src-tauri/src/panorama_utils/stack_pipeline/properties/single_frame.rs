//! Single-frame station properties (16, 75).
//!
//! A one-frame Capture_Station skips the cut and owns every covered pixel, and a
//! station whose non-anchor registrations all fail degrades to one deterministic
//! production source.

use super::super::degradation;
use super::super::degradation::DegradationLedger;
use super::super::focus_fuser;
use super::super::intra_station;
use super::super::report::FusionSolverStatus;
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stitching;
use nalgebra::Matrix3;
use proptest::prelude::*;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::{StationDraw, arb_station_draw, fuse_station, probe_texture};

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
pub(super) const SINGLE_FRAME_SOURCE_SIDE: u32 = 96;

/// A decodable synthetic Source_RAW: the aperiodic probe texture as float RGB.
pub(super) fn single_frame_source(side: u32, seed: u64, detail: f64) -> image::Rgb32FImage {
    image::Rgb32FImage::from_fn(side, side, |x, y| {
        probe_texture(f64::from(x), f64::from(y), seed, detail)
    })
}

/// `ImageInfo` of one synthetic Source_RAW, with no feature or foreground
/// evidence: a Capture_Station of one frame has nothing to register against.
pub(super) fn single_frame_info(id: usize, side: u32) -> ImageInfo {
    ImageInfo {
        id,
        filename: format!("/station/single/{id:04}.NEF"),
        width: side,
        height: side,
        alignment_image: image::GrayImage::new(1, 1),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: Some(75.0),
        source_footprint: None,
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
pub(super) fn render_station_tile(
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
pub(super) static SINGLE_FRAME_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 16 so far.
pub(super) static SINGLE_FRAME_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 16 is configured with.
///
/// Every case exercises the cheap solver contrast; ten cases render the
/// production Virtual_Tile twice, which covers repeat determinism without
/// dominating the complete `cargo test --lib` budget.
pub(super) const SINGLE_FRAME_CASES: usize = 100;

/// Cases of Property 16 that also render the production Virtual_Tile twice.
pub(super) const SINGLE_FRAME_RENDER_STRIDE: usize = 10;

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

pub(super) const SINGLE_FRAME_DEGRADATION_CASES: usize = 100;

pub(super) const SINGLE_FRAME_DEGRADATION_RENDER_STRIDE: usize = 10;

pub(super) const SINGLE_FRAME_DEGRADATION_SIDE: u32 = 64;

pub(super) static SINGLE_FRAME_DEGRADATION_CURSOR: AtomicUsize = AtomicUsize::new(0);

pub(super) static SINGLE_FRAME_DEGRADATION_BRANCHES: AtomicUsize = AtomicUsize::new(0);

pub(super) fn requirement_12_1_source(
    members: &[crate::panorama_utils::stack_pipeline::station_degradation::StationMember],
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

pub(super) fn degradation_alignment(side: u32, sharp: bool, phase: u32) -> image::GrayImage {
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

pub(super) fn degraded_station_info(
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
        source_footprint: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

pub(super) fn render_failed_non_anchor_station(
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
            crate::panorama_utils::stack_pipeline::report::IntraStationFrameRecord {
                path: info.filename.clone(),
                status:
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed,
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

pub(super) fn assert_degraded_station_render(
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
            .map(|(index, score)| crate::panorama_utils::stack_pipeline::station_degradation::StationMember {
                path: format!("/station/property-75/{:04}-{index}.NEF", scores.len() - index),
                median_sharpness: score.map_or(f64::NAN, |value| f64::from(value) / 1_000.0),
                status: crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed,
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
        let plan = crate::panorama_utils::stack_pipeline::station_degradation::plan_station_fusion(&members);
        prop_assert_eq!(
            plan.mode,
            crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::SingleFrameDegraded
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
        let reordered = crate::panorama_utils::stack_pipeline::station_degradation::plan_station_fusion(&members);
        prop_assert_eq!(
            reordered.single_frame_path(&members),
            Some(expected_path.clone()),
            "score/path selection must not depend on import order"
        );

        // Mandatory empty/unmeasurable and exact-tie cases: these run in every
        // generated case instead of depending on random draws.
        prop_assert_eq!(crate::panorama_utils::stack_pipeline::station_degradation::single_frame_source(&[]), None);
        let edge_members = vec![
            crate::panorama_utils::stack_pipeline::station_degradation::StationMember {
                path: "/z/unmeasurable.NEF".to_string(),
                median_sharpness: f64::NAN,
                status: crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed,
            },
            crate::panorama_utils::stack_pipeline::station_degradation::StationMember {
                path: "/a/unmeasurable.NEF".to_string(),
                median_sharpness: f64::INFINITY,
                status: crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed,
            },
        ];
        let edge_plan = crate::panorama_utils::stack_pipeline::station_degradation::plan_station_fusion(&edge_members);
        prop_assert_eq!(
            edge_plan.single_frame_path(&edge_members),
            Some("/a/unmeasurable.NEF".to_string())
        );
        let tie_members = vec![
            crate::panorama_utils::stack_pipeline::station_degradation::StationMember {
                path: "/z/tie.NEF".to_string(),
                median_sharpness: 0.75,
                status: crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed,
            },
            crate::panorama_utils::stack_pipeline::station_degradation::StationMember {
                path: "/a/tie.NEF".to_string(),
                median_sharpness: 0.75,
                status: crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed,
            },
        ];
        let tie_plan = crate::panorama_utils::stack_pipeline::station_degradation::plan_station_fusion(&tie_members);
        prop_assert_eq!(
            tie_plan.single_frame_path(&tie_members),
            Some("/a/tie.NEF".to_string())
        );

        let mut local_ledger = DegradationLedger::new();
        crate::panorama_utils::stack_pipeline::station_degradation::record_entries(
            &mut local_ledger,
            &crate::panorama_utils::stack_pipeline::station_degradation::station_plan_entries(17, &members, &reordered),
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
