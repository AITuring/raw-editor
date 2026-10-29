//! Station level degradation properties (76, 78).
//!
//! A partial registration failure fuses only the frames that registered, and the
//! measured join evidence alone decides whether a source joins a Capture_Station.

use super::super::degradation;
use super::super::degradation::DegradationLedger;
use super::super::intra_station;
use super::super::report::FusionSolverStatus;
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stitching;
use nalgebra::Matrix3;
use proptest::prelude::*;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use super::degraded_station_info;

// ---------------------------------------------------------------------------
// Property 76 harness: partial registration failure keeps only survivors
// ---------------------------------------------------------------------------

pub(super) const PARTIAL_REGISTRATION_CASES: usize = 100;

pub(super) const PARTIAL_REGISTRATION_RENDER_STRIDE: usize = 25;

pub(super) const PARTIAL_REGISTRATION_SIDE: u32 = 64;

pub(super) static PARTIAL_REGISTRATION_CURSOR: AtomicUsize = AtomicUsize::new(0);

pub(super) static PARTIAL_REGISTRATION_BRANCHES: AtomicUsize = AtomicUsize::new(0);

pub(super) fn partial_registration_info(id: usize, failed: bool) -> ImageInfo {
    degraded_station_info(
        id,
        format!(
            "/station/property-76/{}/{id:04}.NEF",
            if failed { "failed" } else { "successful" }
        ),
        PARTIAL_REGISTRATION_SIDE,
        image::GrayImage::new(1, 1),
    )
}

pub(super) fn partial_registration_source(id: usize, failed: bool) -> image::Rgb32FImage {
    let base = if failed {
        0.82 + id as f32 * 0.01
    } else {
        0.08 + id as f32 * 0.025
    };
    image::Rgb32FImage::from_pixel(
        PARTIAL_REGISTRATION_SIDE,
        PARTIAL_REGISTRATION_SIDE,
        image::Rgb([base, base + 0.003, base + 0.006]),
    )
}

pub(super) fn render_registration_boundary(
    infos: &[ImageInfo],
    failed_ids: &BTreeSet<usize>,
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
                status: if failed_ids.contains(&info.id) {
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed
                } else {
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Local
                },
                ..Default::default()
            },
        );
    }
    let homographies = infos
        .iter()
        .map(|info| (info.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();
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
        &homographies,
        stitching::Projection::Planar,
        None,
        None,
        0,
        false,
        app.handle().clone(),
        "stack-pipeline-property-76-progress",
        &mut load,
    )?;
    Ok((rendered, loaded, degradation::run_ledger_snapshot()))
}

pub(super) fn owner_paths(
    rendered: &stitching::FocusStackTileRender,
) -> Result<Vec<Option<String>>, TestCaseError> {
    let masks = rendered
        .masks
        .as_ref()
        .ok_or_else(|| TestCaseError::fail("the production station must publish ownership"))?;
    Ok(masks
        .ownership
        .owners()
        .iter()
        .map(|&owner| {
            (owner != stitching::NO_OWNER).then(|| {
                masks.ownership.legend()[owner as usize - 1]
                    .to_string_lossy()
                    .into_owned()
            })
        })
        .collect())
}

pub(super) fn assert_registration_boundary_render(
    rendered: &stitching::FocusStackTileRender,
    loaded: &[String],
    ledger: &DegradationLedger,
    infos: &[ImageInfo],
    failed_ids: &BTreeSet<usize>,
    sources: &[image::Rgb32FImage],
    expected_mode: crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode,
) -> Result<(), TestCaseError> {
    let successful_paths = infos
        .iter()
        .filter(|info| !failed_ids.contains(&info.id))
        .map(|info| info.filename.clone())
        .collect::<BTreeSet<_>>();
    let loaded_paths = loaded.iter().cloned().collect::<BTreeSet<_>>();
    prop_assert_eq!(loaded.len(), successful_paths.len());
    prop_assert_eq!(
        &loaded_paths,
        &successful_paths,
        "only registration-successful frames may enter production focus fusion"
    );

    let masks = rendered
        .masks
        .as_ref()
        .ok_or_else(|| TestCaseError::fail("the production station must publish ownership"))?;
    let expected_solver = if successful_paths.len() >= 2 {
        FusionSolverStatus::GraphCut
    } else {
        FusionSolverStatus::SingleFrame
    };
    prop_assert_eq!(
        masks.fusion.solver_status,
        expected_solver,
        "the production Focus_Fuser/GraphCut boundary must receive exactly the successful count"
    );

    let by_path = infos
        .iter()
        .map(|info| (info.filename.as_str(), info.id))
        .collect::<HashMap<_, _>>();
    let mut covered = 0u64;
    for y in 0..rendered.image.height() {
        for x in 0..rendered.image.width() {
            if !masks.coverage.is_covered(x, y) {
                prop_assert_eq!(masks.ownership.owner_at(x, y), stitching::NO_OWNER);
                continue;
            }
            covered += 1;
            let owner = masks.ownership.owner_at(x, y);
            prop_assert_ne!(owner, stitching::NO_OWNER);
            let owner_path = masks.ownership.legend()[owner as usize - 1].to_string_lossy();
            prop_assert!(
                successful_paths.contains(owner_path.as_ref()),
                "a failed Source_RAW must never own a production output pixel: {owner_path}"
            );
            let source_id = *by_path
                .get(owner_path.as_ref())
                .ok_or_else(|| TestCaseError::fail("owner legend path must name an input"))?;
            let actual = rendered.image.get_pixel(x, y);
            let expected = sources[source_id].get_pixel(x, y);
            for channel in 0..3 {
                prop_assert_eq!(
                    actual[channel].to_bits(),
                    expected[channel].to_bits(),
                    "covered pixel ({}, {}) channel {} must come from its successful owner",
                    x,
                    y,
                    channel
                );
            }
            for &failed_id in failed_ids {
                prop_assert_ne!(
                    actual.0.map(f32::to_bits),
                    sources[failed_id].get_pixel(x, y).0.map(f32::to_bits),
                    "failed source {} appeared at covered pixel ({}, {})",
                    failed_id,
                    x,
                    y
                );
            }
        }
    }
    prop_assert!(
        covered > 0,
        "production ownership checks must be non-vacuous"
    );
    prop_assert_eq!(covered, masks.ownership.assigned_pixels());

    let partial_entries = ledger
        .entries()
        .iter()
        .filter(|entry| {
            entry.reason == degradation::INTRA_STATION_REGISTRATION_FAILED
                && entry.detail["stage"] == "intra_station_fusion"
        })
        .collect::<Vec<_>>();
    match expected_mode {
        crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::AllFrames => {
            prop_assert!(failed_ids.is_empty());
            prop_assert!(partial_entries.is_empty());
        }
        crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::SuccessfulFramesOnly => {
            prop_assert!(!failed_ids.is_empty());
            prop_assert_eq!(partial_entries.len(), 1);
            let detail = &partial_entries[0].detail;
            prop_assert_eq!(
                detail["fused_frames"].as_u64(),
                Some(successful_paths.len() as u64)
            );
            let reported = detail["excluded"]
                .as_array()
                .ok_or_else(|| TestCaseError::fail("Stack_Report entry must list exclusions"))?
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
                .filter(|info| failed_ids.contains(&info.id))
                .map(|info| {
                    (
                        info.filename.clone(),
                        degradation::INTRA_STATION_REGISTRATION_FAILED.to_string(),
                    )
                })
                .collect::<BTreeSet<_>>();
            prop_assert_eq!(reported, expected);
        }
        crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::SingleFrameDegraded => {
            prop_assert_eq!(successful_paths.len(), 1);
            prop_assert_eq!(
                ledger.count_of(degradation::FUSION_SINGLE_FRAME_DEGRADED),
                1
            );
        }
        crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::Empty => {
            return Err(TestCaseError::fail("the generated station is never empty"));
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: PARTIAL_REGISTRATION_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 76: 对于任意存在配准失败帧且
    // 配准成功帧数不少于 2 的 Capture_Station，参与生产景深合成/GraphCut 与 Ownership_Map
    // 的帧集合恰好等于配准成功帧集合，并逐项报告被排除 Source_RAW 的绝对路径和稳定原因。
    //
    // **Validates: Requirements 12.2**
    #[test]
    fn partial_registration_failure_fuses_only_successful_production_frames(
        successful_count in 2usize..=5,
        failed_count in 1usize..=3,
        permutation_seed in any::<usize>(),
    ) {
        let case = PARTIAL_REGISTRATION_CURSOR.fetch_add(1, Ordering::Relaxed);
        let total = successful_count + failed_count;
        let failed_ids = (successful_count..total).collect::<BTreeSet<_>>();
        let members = (0..total)
            .map(|id| crate::panorama_utils::stack_pipeline::station_degradation::StationMember {
                path: partial_registration_info(id, failed_ids.contains(&id)).filename,
                median_sharpness: 1.0 - id as f64 * 0.01,
                status: if id == 0 {
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Anchor
                } else if failed_ids.contains(&id) {
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed
                } else {
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Local
                },
            })
            .collect::<Vec<_>>();
        let expected_success = members
            .iter()
            .filter(|member| member.registered())
            .map(|member| member.path.clone())
            .collect::<BTreeSet<_>>();
        let expected_failed = members
            .iter()
            .filter(|member| !member.registered())
            .map(|member| {
                (
                    member.path.clone(),
                    degradation::INTRA_STATION_REGISTRATION_FAILED.to_string(),
                )
            })
            .collect::<BTreeSet<_>>();
        let plan = crate::panorama_utils::stack_pipeline::station_degradation::plan_station_fusion(&members);
        prop_assert_eq!(
            plan.mode,
            crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::SuccessfulFramesOnly
        );
        prop_assert_eq!(
            plan.fused
                .iter()
                .map(|&index| members[index].path.clone())
                .collect::<BTreeSet<_>>(),
            expected_success.clone()
        );
        prop_assert_eq!(
            plan.excluded
                .iter()
                .map(|record| (record.path.clone(), record.reason.clone()))
                .collect::<BTreeSet<_>>(),
            expected_failed.clone()
        );

        let mut reordered = members.clone();
        let suffix_len = reordered.len() - 1;
        reordered[1..].rotate_left(permutation_seed % suffix_len);
        if permutation_seed & 1 == 1 {
            reordered[1..].reverse();
        }
        let reordered_plan = crate::panorama_utils::stack_pipeline::station_degradation::plan_station_fusion(&reordered);
        prop_assert_eq!(
            reordered_plan
                .fused
                .iter()
                .map(|&index| reordered[index].path.clone())
                .collect::<BTreeSet<_>>(),
            expected_success
        );
        prop_assert_eq!(
            reordered_plan
                .excluded
                .iter()
                .map(|record| (record.path.clone(), record.reason.clone()))
                .collect::<BTreeSet<_>>(),
            expected_failed
        );

        let all_success = members
            .iter()
            .cloned()
            .map(|mut member| {
                member.status = crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Local;
                member
            })
            .collect::<Vec<_>>();
        prop_assert_eq!(
            crate::panorama_utils::stack_pipeline::station_degradation::plan_station_fusion(&all_success).mode,
            crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::AllFrames
        );
        let one_success = members
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, mut member)| {
                member.status = if index == 0 {
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Anchor
                } else {
                    crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed
                };
                member
            })
            .collect::<Vec<_>>();
        prop_assert_eq!(
            crate::panorama_utils::stack_pipeline::station_degradation::plan_station_fusion(&one_success).mode,
            crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::SingleFrameDegraded
        );

        let mut branches = 0usize;
        if case % PARTIAL_REGISTRATION_RENDER_STRIDE == 0 {
            let sources = (0..total)
                .map(|id| partial_registration_source(id, failed_ids.contains(&id)))
                .collect::<Vec<_>>();
            let mut order = (0..total).collect::<Vec<_>>();
            order[1..].rotate_left(permutation_seed % (total - 1));
            if permutation_seed & 1 == 1 {
                order[1..].reverse();
            }
            let make_infos = |ids: &[usize], failures: &BTreeSet<usize>| {
                ids.iter()
                    .map(|&id| partial_registration_info(id, failures.contains(&id)))
                    .collect::<Vec<_>>()
            };

            let partial_infos = make_infos(&order, &failed_ids);
            let partial = render_registration_boundary(&partial_infos, &failed_ids, &sources)
                .map_err(|error| TestCaseError::fail(format!("partial production render failed: {error}")))?;
            assert_registration_boundary_render(
                &partial.0,
                &partial.1,
                &partial.2,
                &partial_infos,
                &failed_ids,
                &sources,
                crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::SuccessfulFramesOnly,
            )?;

            order[1..].reverse();
            let permuted_infos = make_infos(&order, &failed_ids);
            let permuted = render_registration_boundary(&permuted_infos, &failed_ids, &sources)
                .map_err(|error| TestCaseError::fail(format!("permuted production render failed: {error}")))?;
            assert_registration_boundary_render(
                &permuted.0,
                &permuted.1,
                &permuted.2,
                &permuted_infos,
                &failed_ids,
                &sources,
                crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::SuccessfulFramesOnly,
            )?;
            prop_assert_eq!(partial.0.image.as_raw(), permuted.0.image.as_raw());
            prop_assert_eq!(owner_paths(&partial.0)?, owner_paths(&permuted.0)?);

            let no_failures = BTreeSet::new();
            let all_infos = make_infos(&(0..total).collect::<Vec<_>>(), &no_failures);
            let all = render_registration_boundary(&all_infos, &no_failures, &sources)
                .map_err(|error| TestCaseError::fail(format!("all-frame production render failed: {error}")))?;
            assert_registration_boundary_render(
                &all.0,
                &all.1,
                &all.2,
                &all_infos,
                &no_failures,
                &sources,
                crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::AllFrames,
            )?;

            let single_failures = (1..total).collect::<BTreeSet<_>>();
            let single_infos = make_infos(&(0..total).collect::<Vec<_>>(), &single_failures);
            let single = render_registration_boundary(&single_infos, &single_failures, &sources)
                .map_err(|error| TestCaseError::fail(format!("single-frame production render failed: {error}")))?;
            assert_registration_boundary_render(
                &single.0,
                &single.1,
                &single.2,
                &single_infos,
                &single_failures,
                &sources,
                crate::panorama_utils::stack_pipeline::station_degradation::StationFusionMode::SingleFrameDegraded,
            )?;
            branches = 0b111;
        }
        let seen = PARTIAL_REGISTRATION_BRANCHES.fetch_or(branches, Ordering::Relaxed) | branches;
        prop_assert!(
            case + 1 < PARTIAL_REGISTRATION_CASES || seen == 0b111,
            "Property 76 must exercise AllFrames, SuccessfulFramesOnly and SingleFrameDegraded through production; mask {seen:#b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Property 78 harness: measured evidence controls station membership
// ---------------------------------------------------------------------------

pub(super) const GROUP_JOIN_REJECTION_CASES: usize = 100;

pub(super) static GROUP_JOIN_REJECTION_CURSOR: AtomicUsize = AtomicUsize::new(0);

pub(super) static GROUP_JOIN_REJECTION_BRANCHES: AtomicUsize = AtomicUsize::new(0);

proptest! {
    #![proptest_config(ProptestConfig { cases: GROUP_JOIN_REJECTION_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 78: 对于任意 Source_RAW
    // 与候选 Capture_Station 锚点帧，该 Source_RAW 被拒绝并入该机位当且仅当
    // 内点空间支持 < 20% 或内点中位对称重投影误差 > 0.01 × 锚点长边；拒绝
    // 逐项报告绝对路径、两项实测值、判据标志与稳定原因标识符。
    //
    // **Validates: Requirements 12.5**
    #[test]
    fn measured_group_join_evidence_is_necessary_and_sufficient_across_all_stations(
        anchor_long_sides in prop::collection::vec(320u32..=12_000, 1..=4),
        random_coverage in 0.0f64..=1.0,
        random_error_ratio in 0.0f64..=0.02,
        inliers in 1usize..=2_048,
        path_seed in any::<u64>(),
    ) {
        let case = GROUP_JOIN_REJECTION_CURSOR.fetch_add(1, Ordering::Relaxed);
        let scenario = case % 8;
        let (coverage, error_ratio, branch) = match scenario {
            // Unconstrained finite evidence checks the full iff, not only named examples.
            0 => (random_coverage, random_error_ratio, 1usize),
            // Both exact boundaries are accepted: the defects are strictly < and >.
            1 => (0.20, 0.01, 2usize),
            // Each defect independently and then jointly.
            2 => (0.20 - 1.0e-12, 0.01, 4usize),
            3 => (0.20, 0.01 + 1.0e-12, 8usize),
            4 => (0.20 - 1.0e-12, 0.01 + 1.0e-12, 16usize),
            // Interior accepted evidence proves the implication is not one-way.
            5 => (0.75, 0.005, 32usize),
            // A non-finite value in either measured field is not affirmative evidence.
            6 => (f64::NAN, 0.005, 64usize),
            _ => (0.75, f64::INFINITY, 128usize),
        };
        let source_path = format!("/station/property-78/{path_seed:016x}.NEF");
        let expected_support_defect = !(coverage.is_finite() && coverage >= 0.20);
        let expected_error_defect = !(error_ratio.is_finite() && error_ratio <= 0.01);
        let expected_rejection = expected_support_defect || expected_error_defect;
        let mut ledger = DegradationLedger::new();
        let mut rejected_station_count = 0usize;

        for (station_index, &long_side) in anchor_long_sides.iter().enumerate() {
            let anchor = degraded_station_info(
                station_index * 2,
                format!("/station/property-78/anchor-{station_index}.NEF"),
                long_side,
                image::GrayImage::new(1, 1),
            );
            let source = degraded_station_info(
                station_index * 2 + 1,
                source_path.clone(),
                long_side,
                image::GrayImage::new(1, 1),
            );
            let images = vec![&anchor, &source];
            let measured_error_px = error_ratio * f64::from(long_side);
            let records = HashMap::from([(
                source_path.clone(),
                crate::panorama_utils::stack_pipeline::report::IntraStationFrameRecord {
                    path: source_path.clone(),
                    // Deliberately vary this unrelated count down to one: spatial
                    // support is area coverage, never translation consensus votes.
                    inliers,
                    inlier_area_coverage: coverage,
                    median_symmetric_error_px: measured_error_px,
                    status: crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Local,
                    ..Default::default()
                },
            )]);
            let (members, rejections) =
                stitching::station_members_from_measured_group_join_evidence(&images, &records);

            prop_assert_eq!(rejections.is_empty(), !expected_rejection);
            prop_assert_eq!(
                members[1].status == crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::Failed,
                expected_rejection,
                "station {} membership must be decided only by measured area/error",
                station_index
            );
            if let Some(rejection) = rejections.first() {
                rejected_station_count += 1;
                prop_assert_eq!(rejection.path.as_str(), source_path.as_str());
                prop_assert_eq!(rejection.inlier_area_coverage.to_bits(), coverage.to_bits());
                prop_assert_eq!(
                    rejection.median_symmetric_error_px.to_bits(),
                    measured_error_px.to_bits()
                );
                prop_assert_eq!(rejection.minimum_inlier_area_coverage.to_bits(), 0.20f64.to_bits());
                prop_assert_eq!(
                    rejection.symmetric_error_limit_px.to_bits(),
                    (0.01 * f64::from(long_side)).to_bits()
                );
                prop_assert_eq!(rejection.support_below_minimum, expected_support_defect);
                prop_assert_eq!(rejection.error_above_limit, expected_error_defect);
                ledger.record(
                    degradation::INTRA_STATION_REJECTED_FROM_GROUP,
                    rejection.detail(station_index),
                );
                let entry = ledger.entries().last().expect("rejection must be reported");
                prop_assert_eq!(
                    entry.reason,
                    degradation::INTRA_STATION_REJECTED_FROM_GROUP,
                    "the machine-readable identifier must remain stable"
                );
                prop_assert_eq!(entry.detail["path"].as_str(), Some(source_path.as_str()));
                prop_assert_eq!(
                    entry.detail["support_below_minimum"].as_bool(),
                    Some(expected_support_defect)
                );
                prop_assert_eq!(
                    entry.detail["error_above_limit"].as_bool(),
                    Some(expected_error_defect)
                );
                if coverage.is_finite() {
                    prop_assert_eq!(
                        entry.detail["inlier_area_coverage"].as_f64().map(f64::to_bits),
                        Some(coverage.to_bits())
                    );
                } else {
                    prop_assert!(entry.detail["inlier_area_coverage"].is_null());
                }
                if measured_error_px.is_finite() {
                    prop_assert_eq!(
                        entry.detail["median_symmetric_error_px"].as_f64().map(f64::to_bits),
                        Some(measured_error_px.to_bits())
                    );
                } else {
                    prop_assert!(entry.detail["median_symmetric_error_px"].is_null());
                }
            }

            // The legacy default translation estimator has neither an inlier set
            // nor symmetric error. With no measured record, production must not
            // substitute its consensus patch count for spatial support.
            let (unmeasured_members, unmeasured_rejections) =
                stitching::station_members_from_measured_group_join_evidence(&images, &HashMap::new());
            prop_assert!(unmeasured_rejections.is_empty());
            prop_assert_eq!(
                unmeasured_members[1].status,
                crate::panorama_utils::stack_pipeline::report::IntraStationFrameStatus::GlobalFallback
            );
        }

        prop_assert_eq!(
            rejected_station_count,
            if expected_rejection { anchor_long_sides.len() } else { 0 },
            "the Source_RAW must be rejected from all candidate Capture_Stations iff a defect exists"
        );
        prop_assert_eq!(
            ledger.count_of(degradation::INTRA_STATION_REJECTED_FROM_GROUP),
            rejected_station_count
        );
        let seen = GROUP_JOIN_REJECTION_BRANCHES.fetch_or(branch, Ordering::Relaxed) | branch;
        prop_assert!(
            case + 1 < GROUP_JOIN_REJECTION_CASES || seen == 0xff,
            "Property 78 must cover random iff, exact boundaries, each defect, both defects, accepted evidence, and each non-finite field; mask {seen:#x}"
        );
    }
}
