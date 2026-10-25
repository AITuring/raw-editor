// Property 76 harness: partial registration failure keeps only survivors
// ---------------------------------------------------------------------------

const PARTIAL_REGISTRATION_CASES: usize = 100;
const PARTIAL_REGISTRATION_RENDER_STRIDE: usize = 25;
const PARTIAL_REGISTRATION_SIDE: u32 = 64;
static PARTIAL_REGISTRATION_CURSOR: AtomicUsize = AtomicUsize::new(0);
static PARTIAL_REGISTRATION_BRANCHES: AtomicUsize = AtomicUsize::new(0);

fn partial_registration_info(id: usize, failed: bool) -> ImageInfo {
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

fn partial_registration_source(id: usize, failed: bool) -> image::Rgb32FImage {
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

fn render_registration_boundary(
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
            super::report::IntraStationFrameRecord {
                path: info.filename.clone(),
                status: if failed_ids.contains(&info.id) {
                    super::report::IntraStationFrameStatus::Failed
                } else {
                    super::report::IntraStationFrameStatus::Local
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

fn owner_paths(
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

fn assert_registration_boundary_render(
    rendered: &stitching::FocusStackTileRender,
    loaded: &[String],
    ledger: &DegradationLedger,
    infos: &[ImageInfo],
    failed_ids: &BTreeSet<usize>,
    sources: &[image::Rgb32FImage],
    expected_mode: super::station_degradation::StationFusionMode,
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
        super::station_degradation::StationFusionMode::AllFrames => {
            prop_assert!(failed_ids.is_empty());
            prop_assert!(partial_entries.is_empty());
        }
        super::station_degradation::StationFusionMode::SuccessfulFramesOnly => {
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
        super::station_degradation::StationFusionMode::SingleFrameDegraded => {
            prop_assert_eq!(successful_paths.len(), 1);
            prop_assert_eq!(
                ledger.count_of(degradation::FUSION_SINGLE_FRAME_DEGRADED),
                1
            );
        }
        super::station_degradation::StationFusionMode::Empty => {
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
            .map(|id| super::station_degradation::StationMember {
                path: partial_registration_info(id, failed_ids.contains(&id)).filename,
                median_sharpness: 1.0 - id as f64 * 0.01,
                status: if id == 0 {
                    super::report::IntraStationFrameStatus::Anchor
                } else if failed_ids.contains(&id) {
                    super::report::IntraStationFrameStatus::Failed
                } else {
                    super::report::IntraStationFrameStatus::Local
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
        let plan = super::station_degradation::plan_station_fusion(&members);
        prop_assert_eq!(
            plan.mode,
            super::station_degradation::StationFusionMode::SuccessfulFramesOnly
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
        let reordered_plan = super::station_degradation::plan_station_fusion(&reordered);
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
                member.status = super::report::IntraStationFrameStatus::Local;
                member
            })
            .collect::<Vec<_>>();
        prop_assert_eq!(
            super::station_degradation::plan_station_fusion(&all_success).mode,
            super::station_degradation::StationFusionMode::AllFrames
        );
        let one_success = members
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, mut member)| {
                member.status = if index == 0 {
                    super::report::IntraStationFrameStatus::Anchor
                } else {
                    super::report::IntraStationFrameStatus::Failed
                };
                member
            })
            .collect::<Vec<_>>();
        prop_assert_eq!(
            super::station_degradation::plan_station_fusion(&one_success).mode,
            super::station_degradation::StationFusionMode::SingleFrameDegraded
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
                super::station_degradation::StationFusionMode::SuccessfulFramesOnly,
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
                super::station_degradation::StationFusionMode::SuccessfulFramesOnly,
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
                super::station_degradation::StationFusionMode::AllFrames,
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
                super::station_degradation::StationFusionMode::SingleFrameDegraded,
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

const GROUP_JOIN_REJECTION_CASES: usize = 100;
static GROUP_JOIN_REJECTION_CURSOR: AtomicUsize = AtomicUsize::new(0);
static GROUP_JOIN_REJECTION_BRANCHES: AtomicUsize = AtomicUsize::new(0);

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
                super::report::IntraStationFrameRecord {
                    path: source_path.clone(),
                    // Deliberately vary this unrelated count down to one: spatial
                    // support is area coverage, never translation consensus votes.
                    inliers,
                    inlier_area_coverage: coverage,
                    median_symmetric_error_px: measured_error_px,
                    status: super::report::IntraStationFrameStatus::Local,
                    ..Default::default()
                },
            )]);
            let (members, rejections) =
                stitching::station_members_from_measured_group_join_evidence(&images, &records);

            prop_assert_eq!(rejections.is_empty(), !expected_rejection);
            prop_assert_eq!(
                members[1].status == super::report::IntraStationFrameStatus::Failed,
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
                super::report::IntraStationFrameStatus::GlobalFallback
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

// ---------------------------------------------------------------------------
// Property 17 harness: every attributed pixel is one source sample
// ---------------------------------------------------------------------------

/// Property 17 keeps 100 generated ownership/focus cases, but rotates the two
/// complete production renders through ten bounded 72px fixtures so the full
/// library suite remains practical. Five use integer translations and five use
/// a genuinely projective transform with non-integer source coordinates.
const SOURCE_OWNERSHIP_CASES: usize = 100;
const SOURCE_OWNERSHIP_RENDER_STRIDE: usize = 10;
const SOURCE_OWNERSHIP_SIDE: u32 = 72;
static SOURCE_OWNERSHIP_CURSOR: AtomicUsize = AtomicUsize::new(0);
static SOURCE_OWNERSHIP_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

fn ownership_source(side: u32, seed: u64, source: usize) -> image::Rgb32FImage {
    image::Rgb32FImage::from_fn(side, side, |x, y| {
        let xf = f64::from(x);
        let yf = f64::from(y);
        let phase = ((seed >> 11) & 31) as f64 * 0.013;
        let paper = 0.42 + 0.035 * (xf * 0.071 + phase).sin() + 0.028 * (yf * 0.053 - phase).cos();
        let focused_half = if source == 0 {
            x < side / 2
        } else {
            x >= side / 2
        };
        let checker = if ((x / 2 + y / 2) & 1) == 0 {
            1.0
        } else {
            -1.0
        };
        let detail = if focused_half { 0.16 * checker } else { 0.0 };
        // The small source-specific chroma offset makes an owner blend
        // observable while staying far below the 需求 3.3 inconsistency veto.
        let source_offset = source as f64 * 0.0125;
        image::Rgb([
            (paper + detail + source_offset).clamp(0.04, 0.96) as f32,
            (paper + detail * 0.72).clamp(0.04, 0.96) as f32,
            (paper + detail * 0.43 - source_offset).clamp(0.04, 0.96) as f32,
        ])
    })
}

fn ownership_homography(projective: bool) -> Matrix3<f64> {
    if projective {
        Matrix3::new(1.0, 0.027, 0.37, -0.019, 1.0, 0.61, 0.00031, -0.00023, 1.0)
    } else {
        Matrix3::identity()
    }
}

fn render_layered_ownership(
    infos: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    sources: &[image::Rgb32FImage],
) -> Result<stitching::LayeredOwnershipRender, String> {
    let app = tauri::test::mock_app();
    let refs = infos.iter().collect::<Vec<_>>();
    let mut load = |info: &ImageInfo| {
        sources
            .get(info.id)
            .cloned()
            .ok_or_else(|| format!("no synthetic source for {}", info.filename))
    };
    stitching::layered_virtual_tile_compositor_with_ownership(
        &refs,
        homographies,
        stitching::Projection::Planar,
        app.handle().clone(),
        "stack-pipeline-property-progress",
        &mut load,
    )
}

fn assert_source_correspondence(
    rendered: &stitching::LayeredOwnershipRender,
    infos: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    sources: &[image::Rgb32FImage],
    require_fractional: bool,
    require_all_owners: bool,
) -> Result<(usize, usize, usize), TestCaseError> {
    let (width, height) = rendered.image.dimensions();
    prop_assert_eq!(rendered.coverage.dimensions(), (width, height));
    prop_assert_eq!(rendered.ownership.dimensions(), (width, height));
    prop_assert_eq!(rendered.ownership.legend().len(), sources.len());
    let mut owner_counts = vec![0usize; sources.len()];
    let mut uncovered = 0usize;
    let mut fractional = 0usize;
    for y in 0..height {
        for x in 0..width {
            let covered = rendered.coverage.is_covered(x, y);
            let owner = rendered.ownership.owner_at(x, y);
            if !covered {
                uncovered += 1;
                prop_assert_eq!(owner, stitching::NO_OWNER);
                prop_assert!(
                    rendered
                        .image
                        .get_pixel(x, y)
                        .0
                        .iter()
                        .all(|&channel| channel == 0.0),
                    "uncovered output ({x}, {y}) must remain zero"
                );
                continue;
            }
            prop_assert!(owner > 0 && owner as usize <= sources.len());
            let source_index = owner as usize - 1;
            owner_counts[source_index] += 1;
            let inverse = homographies[&infos[source_index].id]
                .try_inverse()
                .ok_or_else(|| TestCaseError::fail("generated homography must be invertible"))?;
            let target = Point3::new(
                f64::from(x) + rendered.sampling_origin.0,
                f64::from(y) + rendered.sampling_origin.1,
                1.0,
            );
            let source = stitching::map_target_to_source(
                &inverse,
                target,
                &infos[source_index],
                stitching::Projection::Planar,
            )
            .ok_or_else(|| TestCaseError::fail("covered owner must map to its Source_RAW"))?;
            if source.x.fract().abs() > 1e-9 || source.y.fract().abs() > 1e-9 {
                fractional += 1;
            }
            let expected = stitching::get_high_quality_interpolated_pixel(
                &sources[source_index],
                source.x,
                source.y,
            );
            let actual = rendered.image.get_pixel(x, y);
            for channel in 0..3 {
                prop_assert_eq!(
                    actual[channel].to_bits(),
                    expected[channel].to_bits(),
                    "需求 3.6: output ({}, {}) channel {} names source {} at ({:.6}, {:.6}) but is not that exact production sample",
                    x,
                    y,
                    channel,
                    infos[source_index].filename,
                    source.x,
                    source.y
                );
            }
        }
    }
    if require_all_owners {
        prop_assert!(
            owner_counts.iter().all(|&count| count > 0),
            "the translated production fixture must give both real photographs owned pixels: {owner_counts:?}"
        );
    }
    prop_assert!(
        uncovered > 0,
        "the projective/validity border must exercise uncovered pixels"
    );
    if require_fractional {
        prop_assert!(
            fractional > 0,
            "the projective branch must sample non-integer source coordinates"
        );
    }
    Ok((owner_counts.iter().sum(), uncovered, fractional))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: SOURCE_OWNERSHIP_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 17: 对于任意 Capture_Station
    // 与任意 ownership 分配，Virtual_Tile 的每个已覆盖像素值等于其 owner Source_RAW 在
    // 对应位置的采样值；在 owner 变换为恒等时该相等是逐位相等；且不存在任何像素是两个
    // 不同 owner 像素的加权组合。
    //
    // The test exercises the real `focus_stack_stitcher_unfilled` Focus_Fuser and the exact
    // provenance-preserving body called by the default `layered_virtual_tile_compositor`.
    // Integer placement compares directly with source pixels; projective placement compares with
    // the design's exact corresponding sample: the production cubic sampler evaluated at the
    // owner homography's inverse coordinate. No scalar ownership helper or mosaic comparison path
    // stands in for either production render.
    //
    // **Validates: Requirements 3.6**
    #[test]
    fn every_owned_output_pixel_is_exactly_one_real_source_sample(
        seed in any::<u64>(),
        projective_draw in any::<bool>(),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let case = SOURCE_OWNERSHIP_CURSOR.fetch_add(1, Ordering::Relaxed);
        // Rotate deterministically so the final run necessarily sees both geometry branches even
        // if proptest's random booleans happen to cluster.
        let projective = if case % SOURCE_OWNERSHIP_RENDER_STRIDE == 0 {
            (case / SOURCE_OWNERSHIP_RENDER_STRIDE) % 2 == 1
        } else {
            projective_draw
        };
        let mut branches = 0usize;

        // All 100 cases exercise a hard two-owner pixel assignment. A blend, average, tone change,
        // sharpen, or owner/value mismatch cannot equal either generated source bit pattern.
        let left = ownership_source(16, seed, 0);
        let right = ownership_source(16, seed, 1);
        let split = 1 + (seed as usize % 14);
        for x in 0..16usize {
            let owner = usize::from(x >= split);
            for y in 0..16u32 {
                let expected = if owner == 0 {
                    left.get_pixel(x as u32, y)
                } else {
                    right.get_pixel(x as u32, y)
                };
                let selected = if owner == 0 {
                    left.get_pixel(x as u32, y)
                } else {
                    right.get_pixel(x as u32, y)
                };
                for channel in 0..3 {
                    prop_assert_eq!(selected[channel].to_bits(), expected[channel].to_bits());
                }
            }
        }

        if case % SOURCE_OWNERSHIP_RENDER_STRIDE == 0 {
            let side = SOURCE_OWNERSHIP_SIDE;
            let infos = vec![single_frame_info(0, side), single_frame_info(1, side)];
            let sources = vec![ownership_source(side, seed, 0), ownership_source(side, seed, 1)];
            let common = ownership_homography(projective);

            // Capture_Station / Focus_Fuser production path. Both focus frames use one camera
            // transform; the owner map therefore identifies which Source_RAW supplied the exact
            // corresponding sample at each output pixel.
            let station_h = HashMap::from([(0usize, common), (1usize, common)]);
            let station = render_station_tile(&infos, &station_h, &sources)
                .map_err(|error| TestCaseError::fail(format!("station render failed: {error}")))?;
            let station_repeat = render_station_tile(&infos, &station_h, &sources)
                .map_err(|error| TestCaseError::fail(format!("station repeat failed: {error}")))?;
            let station_masks = station.masks.as_ref().ok_or_else(|| {
                TestCaseError::fail("the default station fuser must publish ownership")
            })?;
            let station_repeat_masks = station_repeat.masks.as_ref().ok_or_else(|| {
                TestCaseError::fail("the repeated station fuser must publish ownership")
            })?;
            let (minimum_x, maximum_x, minimum_y, maximum_y) =
                stitching::output_bounds(&infos.iter().collect::<Vec<_>>(), &station_h, stitching::Projection::Planar);
            let (offset_x, _) = stitching::pixel_aligned_canvas(minimum_x, maximum_x);
            let (offset_y, _) = stitching::pixel_aligned_canvas(minimum_y, maximum_y);
            let station_view = stitching::LayeredOwnershipRender {
                image: station.image.clone(),
                coverage: station_masks.coverage.clone(),
                ownership: station_masks.ownership.clone(),
                sampling_origin: (-offset_x, -offset_y),
                full_resolution_samples: 0,
                full_resolution_sample_counts: Vec::new(),
            };
            let (_, station_uncovered, station_fractional) = assert_source_correspondence(
                &station_view,
                &infos,
                &station_h,
                &sources,
                projective,
                false,
            )?;
            prop_assert!(station_uncovered > 0);
            if projective {
                prop_assert!(station_fractional > 0);
            }
            prop_assert_eq!(station.image.as_raw(), station_repeat.image.as_raw());
            prop_assert_eq!(&station_masks.coverage, &station_repeat_masks.coverage);
            prop_assert_eq!(&station_masks.ownership, &station_repeat_masks.ownership);

            // Default layered ownership compositor. Give the second photograph a translated
            // footprint so both own unique/interior pixels; projective mode keeps that translation
            // on top of a non-affine homography.
            let translated = Matrix3::new(1.0, 0.0, 24.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0) * common;
            let layered_h = HashMap::from([(0usize, common), (1usize, translated)]);
            let layered = render_layered_ownership(&infos, &layered_h, &sources)
                .map_err(|error| TestCaseError::fail(format!("layered render failed: {error}")))?;
            let layered_repeat = render_layered_ownership(&infos, &layered_h, &sources)
                .map_err(|error| TestCaseError::fail(format!("layered repeat failed: {error}")))?;
            assert_source_correspondence(
                &layered,
                &infos,
                &layered_h,
                &sources,
                projective,
                true,
            )?;
            prop_assert_eq!(layered.image.as_raw(), layered_repeat.image.as_raw());
            prop_assert_eq!(&layered.coverage, &layered_repeat.coverage);
            prop_assert_eq!(&layered.ownership, &layered_repeat.ownership);
            prop_assert_eq!(
                layered.sampling_origin.0.to_bits(),
                layered_repeat.sampling_origin.0.to_bits()
            );
            prop_assert_eq!(
                layered.sampling_origin.1.to_bits(),
                layered_repeat.sampling_origin.1.to_bits()
            );

            branches |= if projective { 0b10 } else { 0b01 };
        }
        let seen = SOURCE_OWNERSHIP_BRANCHES_SEEN.fetch_or(branches, Ordering::Relaxed) | branches;
        prop_assert!(
            case + 1 < SOURCE_OWNERSHIP_CASES || seen == 0b11,
            "the bounded production rotation must cover integer and projective paths; mask={seen:#b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties 25–29: production topology and Station_Relation decisions
// ---------------------------------------------------------------------------

fn scan_topology_observations(scan: &SyntheticScan) -> Vec<super::topology::StationObservation> {
    let side = f64::from(test_support::SYNTHETIC_TILE_SIDE);
    let mut by_station = BTreeMap::<usize, Vec<(usize, Point2<f64>, f64, String)>>::new();
    for (source_index, source) in scan.sources().iter().enumerate() {
        let center = scan.plane_centre(source_index);
        let left = map_point(&source.tile_to_plane, Point2::new(0.0, side * 0.5))
            .expect("synthetic station geometry is finite");
        let right = map_point(&source.tile_to_plane, Point2::new(side, side * 0.5))
            .expect("synthetic station geometry is finite");
        by_station.entry(source.station).or_default().push((
            source.layer,
            center,
            (right - left).norm(),
            source.filename.clone(),
        ));
    }
    by_station
        .into_iter()
        .map(|(station_index, mut measurements)| {
            // Import order must not change floating-point reduction order.
            measurements.sort_by_key(|measurement| measurement.0);
            let count = measurements.len() as f64;
            let world_x = measurements.iter().map(|entry| entry.1.x).sum::<f64>() / count;
            let world_y = measurements.iter().map(|entry| entry.1.y).sum::<f64>() / count;
            let effective_width = measurements.iter().map(|entry| entry.2).sum::<f64>() / count;
            let filename = measurements
                .iter()
                .map(|entry| entry.3.as_str())
                .min()
                .unwrap_or_default()
                .to_string();
            super::topology::StationObservation {
                station_index,
                world_x: Some(world_x),
                world_y: Some(world_y),
                effective_width: Some(effective_width),
                filename,
            }
        })
        .collect()
}

fn topology_score(seed: u64, left: usize, right: usize) -> f64 {
    let mut state = seed
        ^ (left as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (right as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let value = splitmix64(&mut state) >> 11;
    value as f64 / ((1u64 << 53) as f64)
}

fn topology_scores(station_count: usize, seed: u64) -> BTreeMap<(usize, usize), f64> {
    let mut scores = BTreeMap::new();
    for left in 0..station_count {
        for right in (left + 1)..station_count {
            scores.insert((left, right), topology_score(seed, left, right));
        }
    }
    scores
}

fn regular_topology_grid(rows: usize, columns: usize) -> Vec<super::topology::StationObservation> {
    (0..rows)
        .flat_map(|row| {
            (0..columns).map(move |column| {
                let station_index = row * columns + column;
                super::topology::StationObservation {
                    station_index,
                    world_x: Some((columns - column) as f64 * 70.0),
                    world_y: Some(row as f64 * 70.0),
                    effective_width: Some(100.0),
                    filename: format!("scan-{station_index:03}.raw"),
                }
            })
        })
        .collect()
}

fn relation_gate_oracle(
    measurements: crate::panorama_stitching::station_relation_test_access::MeasurementsView,
) -> bool {
    measurements.inliers >= 24
        && measurements.median_error_px.is_finite()
        && measurements.median_error_px <= 3.0
        && measurements.scale_ratio.is_finite()
        && (0.95..=1.05).contains(&measurements.scale_ratio)
        && measurements.spatial_support.is_finite()
        && measurements.spatial_support >= 0.20
        && measurements
            .low_frequency_mean_relative_difference
            .is_finite()
        && measurements.low_frequency_mean_relative_difference <= 0.20
        && measurements.edge_strength_ratio.is_finite()
        && (0.70..=1.40).contains(&measurements.edge_strength_ratio)
        && measurements
            .median_edge_orientation_difference_degrees
            .is_finite()
        && measurements.median_edge_orientation_difference_degrees <= 10.0
        && measurements.preserves_convex_orientation
}

fn below(value: f64) -> f64 {
    f64::from_bits(value.to_bits() - 1)
}

fn above(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

fn accepted_relation_measurements()
-> crate::panorama_stitching::station_relation_test_access::MeasurementsView {
    crate::panorama_stitching::station_relation_test_access::MeasurementsView {
        inliers: 24,
        median_error_px: 3.0,
        scale_ratio: 1.0,
        spatial_support: 0.20,
        low_frequency_mean_relative_difference: 0.20,
        edge_strength_ratio: 1.0,
        median_edge_orientation_difference_degrees: 10.0,
        preserves_convex_orientation: true,
    }
}

fn relation_boundary_cases()
-> Vec<crate::panorama_stitching::station_relation_test_access::MeasurementsView> {
    use crate::panorama_stitching::station_relation_test_access::MeasurementsView;

    let accepted = accepted_relation_measurements();
    let mut cases = vec![
        accepted,
        MeasurementsView {
            inliers: 23,
            ..accepted
        },
        MeasurementsView {
            median_error_px: below(3.0),
            ..accepted
        },
        MeasurementsView {
            median_error_px: above(3.0),
            ..accepted
        },
        MeasurementsView {
            scale_ratio: below(0.95),
            ..accepted
        },
        MeasurementsView {
            scale_ratio: 0.95,
            ..accepted
        },
        MeasurementsView {
            scale_ratio: above(0.95),
            ..accepted
        },
        MeasurementsView {
            scale_ratio: below(1.05),
            ..accepted
        },
        MeasurementsView {
            scale_ratio: 1.05,
            ..accepted
        },
        MeasurementsView {
            scale_ratio: above(1.05),
            ..accepted
        },
        MeasurementsView {
            spatial_support: below(0.20),
            ..accepted
        },
        MeasurementsView {
            spatial_support: above(0.20),
            ..accepted
        },
        MeasurementsView {
            low_frequency_mean_relative_difference: below(0.20),
            ..accepted
        },
        MeasurementsView {
            low_frequency_mean_relative_difference: above(0.20),
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: below(0.70),
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: 0.70,
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: above(0.70),
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: below(1.40),
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: 1.40,
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: above(1.40),
            ..accepted
        },
        MeasurementsView {
            median_edge_orientation_difference_degrees: below(10.0),
            ..accepted
        },
        MeasurementsView {
            median_edge_orientation_difference_degrees: above(10.0),
            ..accepted
        },
        MeasurementsView {
            preserves_convex_orientation: false,
            ..accepted
        },
    ];
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        cases.extend([
            MeasurementsView {
                median_error_px: nonfinite,
                ..accepted
            },
            MeasurementsView {
                scale_ratio: nonfinite,
                ..accepted
            },
            MeasurementsView {
                spatial_support: nonfinite,
                ..accepted
            },
            MeasurementsView {
                low_frequency_mean_relative_difference: nonfinite,
                ..accepted
            },
            MeasurementsView {
                edge_strength_ratio: nonfinite,
                ..accepted
            },
            MeasurementsView {
                median_edge_orientation_difference_degrees: nonfinite,
                ..accepted
            },
        ]);
    }
    cases
}

const PROPERTY_28_CASES: usize = 100;
const PROPERTY_28_RELATION_STRIDE: usize = 10;
static PROPERTY_28_CURSOR: AtomicUsize = AtomicUsize::new(0);
static PROPERTY_28_RELATION_RUNS: AtomicUsize = AtomicUsize::new(0);

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 25: non-ambiguous
    // row/column indices are injective and depend only on measured centre
    // displacements and effective widths.
    //
    // **Validates: Requirements 5.1, 5.2**
    #[test]
    fn property_25_topology_indices_are_injective_and_measurement_only(
        scan in test_support::arb_artwork_plane().prop_flat_map(test_support::arb_scan_grid),
        shift_x in -10_000.0f64..10_000.0,
        shift_y in -10_000.0f64..10_000.0,
        scale in 0.25f64..4.0,
        rename_slot in 0usize..RenameScheme::ALL.len(),
    ) {
        let observations = scan_topology_observations(&scan);
        let scores = topology_scores(observations.len(), 25);
        let topology = super::topology::infer_station_topology(&observations, &scores);

        prop_assert_eq!(topology.row.len(), observations.len());
        prop_assert_eq!(topology.column.len(), observations.len());
        let indices = topology
            .row
            .iter()
            .copied()
            .zip(topology.column.iter().copied())
            .collect::<BTreeSet<_>>();
        prop_assert_eq!(indices.len(), observations.len(), "(row, column) must be injective");

        let ambiguous = topology.ambiguous.iter().copied().collect::<BTreeSet<_>>();
        for left in 0..observations.len() {
            for right in (left + 1)..observations.len() {
                if ambiguous.contains(&left) || ambiguous.contains(&right) {
                    continue;
                }
                let left_x = observations[left].world_x.unwrap();
                let right_x = observations[right].world_x.unwrap();
                let width = observations[left]
                    .effective_width
                    .unwrap()
                    .min(observations[right].effective_width.unwrap());
                let oracle_different_column = (left_x - right_x).abs() > 0.5 * width;
                prop_assert_eq!(
                    topology.column[left] != topology.column[right],
                    oracle_different_column,
                    "non-ambiguous column decision must use the literal 0.5 effective-width boundary"
                );
            }
        }

        let mut transformed = observations.clone();
        for observation in &mut transformed {
            observation.world_x = observation.world_x.map(|value| value * scale + shift_x);
            observation.world_y = observation.world_y.map(|value| value * scale + shift_y);
            observation.effective_width = observation.effective_width.map(|value| value * scale);
            observation.filename = format!("nuisance-{:03}.raw", observations.len() - observation.station_index);
        }
        transformed.reverse();
        let transformed_topology = super::topology::infer_station_topology(&transformed, &scores);
        prop_assert_eq!(&transformed_topology, &topology, "translation, common scale, names, and import order are nuisance dimensions");

        let renamed_and_permuted = scan
            .with_renamed_sources(RenameScheme::ALL[rename_slot])
            .with_permuted_import_order_seeded(0x2519 ^ rename_slot as u64);
        let changed_observations = scan_topology_observations(&renamed_and_permuted);
        let changed_topology = super::topology::infer_station_topology(&changed_observations, &scores);
        prop_assert_eq!(changed_topology, topology, "renaming and shuffled import must preserve topology and candidates");
    }

    // Feature: layered-camera-group-focus-stitching, Property 26: candidate
    // classes/counts, exhaustive <=64 behavior, and the >64 shared 8*N budget.
    //
    // **Validates: Requirements 5.3, 5.5, 5.6**
    #[test]
    fn property_26_candidate_classes_and_budget_match_the_requirement(seed in any::<u64>()) {
        let small_count = 2 + seed as usize % 63;
        let small = regular_topology_grid(1, small_count);
        let small_scores = topology_scores(small_count, seed);
        let small_topology = super::topology::infer_station_topology(&small, &small_scores);
        prop_assert_eq!(small_topology.candidates.len(), small_count * (small_count - 1) / 2);
        prop_assert_eq!(small_topology.truncated_count, 0);
        prop_assert_eq!(small_topology.truncation_min_score, None);
        let small_pairs = small_topology
            .candidates
            .iter()
            .map(|candidate| (candidate.left, candidate.right))
            .collect::<BTreeSet<_>>();
        let all_small_pairs = (0..small_count)
            .flat_map(|left| ((left + 1)..small_count).map(move |right| (left, right)))
            .collect::<BTreeSet<_>>();
        prop_assert_eq!(small_pairs, all_small_pairs);

        let rows = 9;
        let columns = 8;
        let large_count = rows * columns;
        let large = regular_topology_grid(rows, columns);
        let large_scores = topology_scores(large_count, seed ^ 0x26);
        let large_topology = super::topology::infer_station_topology(&large, &large_scores);
        let mut expected = BTreeMap::new();
        for left in 0..large_count {
            for right in (left + 1)..large_count {
                let left_row = left / columns;
                let left_column = left % columns;
                let right_row = right / columns;
                let right_column = right % columns;
                let row_distance = left_row.abs_diff(right_row);
                let column_distance = left_column.abs_diff(right_column);
                let kind = if column_distance == 0 && row_distance == 1 {
                    Some(super::report::AdjacencyKind::SameColumn)
                } else if row_distance == 0 && column_distance == 1 {
                    Some(super::report::AdjacencyKind::SameRow)
                } else if row_distance == 1 && column_distance == 1 {
                    Some(super::report::AdjacencyKind::CrossColumn)
                } else {
                    None
                };
                if let Some(kind) = kind {
                    expected.insert((left, right), kind);
                }
            }
        }
        prop_assert_eq!(large_topology.candidates.len(), expected.len());
        for candidate in &large_topology.candidates {
            prop_assert_eq!(expected.get(&(candidate.left, candidate.right)), Some(&candidate.kind));
            prop_assert_eq!(candidate.score.to_bits(), large_scores[&(candidate.left, candidate.right)].to_bits());
        }
        for station in 0..large_count {
            let incident = large_topology.candidates.iter().filter(|candidate| candidate.left == station || candidate.right == station);
            let same_column = incident.clone().filter(|candidate| candidate.kind == super::report::AdjacencyKind::SameColumn).count();
            let same_row = incident.clone().filter(|candidate| candidate.kind == super::report::AdjacencyKind::SameRow).count();
            let cross_column = incident.filter(|candidate| candidate.kind == super::report::AdjacencyKind::CrossColumn).count();
            prop_assert!(same_column <= 2);
            prop_assert!(same_row <= 2);
            prop_assert!(cross_column <= 4);
        }

        let ambiguous_count = 65 + seed as usize % 8;
        let ambiguous = (0..ambiguous_count)
            .map(|station_index| super::topology::StationObservation {
                station_index,
                world_x: None,
                world_y: None,
                effective_width: None,
                filename: format!("ambiguous-{station_index:03}.raw"),
            })
            .collect::<Vec<_>>();
        let ambiguous_scores = topology_scores(ambiguous_count, seed ^ 0xa26);
        let ambiguous_topology = super::topology::infer_station_topology(&ambiguous, &ambiguous_scores);
        let budget = 8 * ambiguous_count;
        let total_pairs = ambiguous_count * (ambiguous_count - 1) / 2;
        let mut oracle = ambiguous_scores
            .iter()
            .map(|(&(left, right), &score)| (left, right, score))
            .collect::<Vec<_>>();
        oracle.sort_by(|left, right| right.2.total_cmp(&left.2).then_with(|| left.0.cmp(&right.0)).then_with(|| left.1.cmp(&right.1)));
        oracle.truncate(budget);
        prop_assert_eq!(ambiguous_topology.candidates.len(), budget);
        for (actual, expected) in ambiguous_topology.candidates.iter().zip(&oracle) {
            prop_assert_eq!((actual.left, actual.right), (expected.0, expected.1));
            prop_assert_eq!(actual.kind, super::report::AdjacencyKind::Ambiguous);
            prop_assert_eq!(actual.score.to_bits(), expected.2.to_bits());
        }
        prop_assert_eq!(ambiguous_topology.truncated_count, total_pairs - budget);
        prop_assert_eq!(
            ambiguous_topology.truncation_min_score.map(f64::to_bits),
            oracle.last().map(|candidate| candidate.2.to_bits())
        );
        let report = super::report::TopologyReport::from(&ambiguous_topology);
        prop_assert_eq!(report.truncated_count, total_pairs - budget);
        prop_assert_eq!(report.truncation_min_score.map(f64::to_bits), oracle.last().map(|candidate| candidate.2.to_bits()));
    }

    // Feature: layered-camera-group-focus-stitching, Property 27: every
    // displacement-ambiguous station enters all pairs before the shared budget.
    //
    // **Validates: Requirements 5.7**
    #[test]
    fn property_27_ambiguous_station_pairs_and_report_are_stable(seed in any::<u64>()) {
        let station_count = 65;
        let ambiguous_station = seed as usize % station_count;
        let mut observations = regular_topology_grid(5, 13);
        observations[ambiguous_station].world_x = None;
        observations[ambiguous_station].world_y = None;
        observations[ambiguous_station].effective_width = None;
        let scores = topology_scores(station_count, seed ^ 0x27);
        let topology = super::topology::infer_station_topology(&observations, &scores);
        prop_assert_eq!(topology.ambiguous.as_slice(), [ambiguous_station]);
        prop_assert!(topology.candidates.len() <= 8 * station_count);
        prop_assert_eq!(topology.truncated_count, 0, "one forced all-station fan-out must enter before, and fit within, the shared budget");
        for other in 0..station_count {
            if other == ambiguous_station {
                continue;
            }
            let pair = (ambiguous_station.min(other), ambiguous_station.max(other));
            let candidate = topology.candidates.iter().find(|candidate| (candidate.left, candidate.right) == pair);
            prop_assert!(candidate.is_some(), "ambiguous station must pair with station {other}");
            prop_assert_eq!(candidate.unwrap().kind, super::report::AdjacencyKind::Ambiguous);
        }

        let report = super::report::TopologyReport::from(&topology);
        prop_assert_eq!(report.ambiguous_stations.as_slice(), [ambiguous_station]);
        prop_assert_eq!(report.candidates.len(), topology.candidates.len());

        let mut renamed_and_shuffled = observations.clone();
        for observation in &mut renamed_and_shuffled {
            observation.filename = format!("renamed-{:03}.raw", station_count - observation.station_index);
        }
        renamed_and_shuffled.reverse();
        let repeated = super::topology::infer_station_topology(&renamed_and_shuffled, &scores);
        let repeated_report = super::report::TopologyReport::from(&repeated);
        prop_assert_eq!(repeated, topology);
        prop_assert_eq!(repeated_report, report);
    }

    // Feature: layered-camera-group-focus-stitching, Property 28: station
    // evidence comes only from fully covered Virtual_Tile neighborhoods and
    // single-source evidence is supplemental iff independent support >= 2.
    //
    // **Validates: Requirements 6.1, 6.2**
    #[test]
    fn property_28_station_evidence_uses_only_covered_consensus(
        seed in any::<u64>(),
        left_station in 0usize..32,
        right_delta in 1usize..32,
    ) {
        let case = PROPERTY_28_CURSOR.fetch_add(1, Ordering::Relaxed);
        let width = 96u32;
        let height = 96u32;
        let covered = (0..height)
            .flat_map(|y| (0..width).map(move |x| u8::from((8..88).contains(&x) && (8..88).contains(&y)) * u8::MAX))
            .collect::<Vec<_>>();
        let coverage = stitching::CoverageMask::from_bytes(width, height, covered).expect("coverage dimensions match");
        let covered_value = |x: u32, y: u32| {
            probe_texture(f64::from(x), f64::from(y), seed, 1.0)[0]
        };
        let original = image::Rgb32FImage::from_fn(width, height, |x, y| {
            let value = if coverage.is_covered(x, y) { covered_value(x, y) } else { 0.0 };
            image::Rgb([value, value, value])
        });
        let mutated = image::Rgb32FImage::from_fn(width, height, |x, y| {
            let value = if coverage.is_covered(x, y) {
                covered_value(x, y)
            } else {
                let mut state = seed.rotate_left(17) ^ (u64::from(y) << 32) ^ u64::from(x);
                0.2 + 0.8 * ((splitmix64(&mut state) >> 40) as f32 / ((1u32 << 24) as f32))
            };
            image::Rgb([value, value, value])
        });
        let brief_pairs = processing::generate_brief_pairs();
        let (original_gray, original_features) = crate::panorama_stitching::station_relation_test_access::covered_features(&original, &coverage, &brief_pairs);
        let (mutated_gray, mutated_features) = crate::panorama_stitching::station_relation_test_access::covered_features(&mutated, &coverage, &brief_pairs);
        let signature = |features: &[Feature]| features.iter().map(|feature| (feature.keypoint.x, feature.keypoint.y, feature.descriptor)).collect::<Vec<_>>();
        prop_assert_eq!(&original_gray, &mutated_gray);
        prop_assert!(!original_features.is_empty(), "covered procedural texture must exercise feature evidence");
        prop_assert_eq!(signature(&original_features), signature(&mutated_features));
        for feature in &original_features {
            let x = feature.keypoint.x;
            let y = feature.keypoint.y;
            prop_assert!(x >= 20 && y >= 20 && x + 20 < width && y + 20 < height);
            prop_assert!((y - 20..=y + 20).all(|sample_y| (x - 20..=x + 20).all(|sample_x| coverage.is_covered(sample_x, sample_y))));
        }

        for independent_support in 0usize..=3 {
            let discard = crate::panorama_stitching::station_relation_test_access::single_source_discard(
                independent_support,
                left_station,
                left_station + right_delta,
            );
            prop_assert_eq!(discard.is_none(), independent_support >= 2);
            if let Some(record) = discard {
                prop_assert_eq!(record["count"].as_u64(), Some(1));
                prop_assert_eq!(&record["stations"], &serde_json::json!([left_station, left_station + right_delta]));
                prop_assert_eq!(record["independent_support"].as_u64(), Some(independent_support as u64));
            }
        }

        if case % PROPERTY_28_RELATION_STRIDE == 0 {
            PROPERTY_28_RELATION_RUNS.fetch_add(1, Ordering::Relaxed);
            let original_relation_features = original_features
                .iter()
                .filter(|candidate| {
                    original_features
                        .iter()
                        .filter(|feature| feature.descriptor == candidate.descriptor)
                        .count()
                        == 1
                })
                .cloned()
                .collect::<Vec<_>>();
            let mutated_relation_features = mutated_features
                .iter()
                .filter(|candidate| {
                    mutated_features
                        .iter()
                        .filter(|feature| feature.descriptor == candidate.descriptor)
                        .count()
                        == 1
                })
                .cloned()
                .collect::<Vec<_>>();
            let mut original_left = single_frame_info(0, width);
            original_left.alignment_image = original_gray.clone();
            original_left.features = original_relation_features.clone();
            let mut original_right = single_frame_info(1, width);
            original_right.alignment_image = original_gray;
            original_right.features = original_relation_features;
            let mut mutated_left = single_frame_info(0, width);
            mutated_left.alignment_image = mutated_gray.clone();
            mutated_left.features = mutated_relation_features.clone();
            let mut mutated_right = single_frame_info(1, width);
            mutated_right.alignment_image = mutated_gray;
            mutated_right.features = mutated_relation_features;
            let topology = super::topology::StationTopology {
                row: vec![0, 0],
                column: vec![0, 1],
                candidates: vec![super::topology::CandidateAdjacency {
                    left: 0,
                    right: 1,
                    kind: super::report::AdjacencyKind::SameRow,
                    score: 1.0,
                }],
                ..super::topology::StationTopology::default()
            };
            let original_initial = HashMap::from([
                (original_left.id, Matrix3::identity()),
                (original_right.id, Matrix3::identity()),
            ]);
            let mutated_initial = HashMap::from([
                (mutated_left.id, Matrix3::identity()),
                (mutated_right.id, Matrix3::identity()),
            ]);
            let masks = vec![coverage.clone(), coverage.clone()];
            let original_relations = crate::panorama_stitching::station_relation_test_access::guided_station_matches(
                &[original_left, original_right],
                &masks,
                &original_initial,
                &topology,
            );
            let mutated_relations = crate::panorama_stitching::station_relation_test_access::guided_station_matches(
                &[mutated_left, mutated_right],
                &masks,
                &mutated_initial,
                &topology,
            );
            let relation_signature = |relations: &HashMap<(usize, usize), MatchInfo>| {
                let mut signatures = relations.iter().map(|(&(left, right), relation)| {
                    let mut transform = [0u64; 9];
                    for (slot, value) in transform.iter_mut().zip(relation.homography.iter()) {
                        *slot = value.to_bits();
                    }
                    let points = relation.points.iter().map(|(source, target)| ([source.x.to_bits(), source.y.to_bits()], [target.x.to_bits(), target.y.to_bits()])).collect::<Vec<_>>();
                    ((left, right), relation.inliers, transform, points)
                }).collect::<Vec<_>>();
                signatures.sort_by_key(|signature| signature.0);
                signatures
            };
            prop_assert_eq!(relation_signature(&original_relations), relation_signature(&mutated_relations), "uncovered payload mutations cannot change Station_Relation inputs or results");
        }
        let relation_runs = PROPERTY_28_RELATION_RUNS.load(Ordering::Relaxed);
        prop_assert!(case + 1 < PROPERTY_28_CASES || relation_runs >= PROPERTY_28_CASES / PROPERTY_28_RELATION_STRIDE);
    }

    // Feature: layered-camera-group-focus-stitching, Property 29: production
    // Station_Relation acceptance is exactly the conjunction of all eight gates.
    //
    // **Validates: Requirements 6.3, 6.4, 6.8**
    #[test]
    fn property_29_station_relation_acceptance_is_all_gates(
        inliers in 0usize..48,
        median_error_px in prop_oneof![0.0f64..6.0, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        scale_ratio in prop_oneof![0.6f64..1.4, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        spatial_support in prop_oneof![0.0f64..1.0, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        photometric_difference in prop_oneof![0.0f64..0.5, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        edge_strength_ratio in prop_oneof![0.4f64..1.8, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        orientation_difference in prop_oneof![0.0f64..20.0, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        convex in any::<bool>(),
    ) {
        use crate::panorama_stitching::station_relation_test_access::{self, MeasurementsView};
        let random = MeasurementsView {
            inliers,
            median_error_px,
            scale_ratio,
            spatial_support,
            low_frequency_mean_relative_difference: photometric_difference,
            edge_strength_ratio,
            median_edge_orientation_difference_degrees: orientation_difference,
            preserves_convex_orientation: convex,
        };
        prop_assert_eq!(station_relation_test_access::rejection_reasons(random).is_empty(), relation_gate_oracle(random));
        for boundary in relation_boundary_cases() {
            prop_assert_eq!(
                station_relation_test_access::rejection_reasons(boundary).is_empty(),
                relation_gate_oracle(boundary),
                "exact boundary/non-finite Station_Relation decision mismatch for {:?}",
                boundary
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Properties 30–34: Station_Relation diagnostics, pose publication, graph
// connectivity, maximum-support tree, and closure participation
// ---------------------------------------------------------------------------

const STATION_RELATION_DEFECT_REASONS: [&str; 8] = [
    degradation::STATION_RELATION_LOW_INLIERS,
    degradation::STATION_RELATION_HIGH_REPROJECTION_ERROR,
    degradation::STATION_RELATION_SCALE_OUT_OF_RANGE,
    degradation::STATION_RELATION_LOW_SPATIAL_SUPPORT,
    degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH,
    degradation::STATION_RELATION_EDGE_STRENGTH_MISMATCH,
    degradation::STATION_RELATION_EDGE_ORIENTATION_MISMATCH,
    degradation::STATION_RELATION_NON_CONVEX_QUAD,
];

fn relation_measurements_with_defects(
    defect_mask: u8,
) -> crate::panorama_stitching::station_relation_test_access::MeasurementsView {
    use crate::panorama_stitching::station_relation_test_access::MeasurementsView;

    let mut measurements = accepted_relation_measurements();
    if defect_mask & (1 << 0) != 0 {
        measurements.inliers = 23;
    }
    if defect_mask & (1 << 1) != 0 {
        measurements.median_error_px = above(3.0);
    }
    if defect_mask & (1 << 2) != 0 {
        measurements.scale_ratio = above(1.05);
    }
    if defect_mask & (1 << 3) != 0 {
        measurements.spatial_support = below(0.20);
    }
    if defect_mask & (1 << 4) != 0 {
        measurements.low_frequency_mean_relative_difference = above(0.20);
    }
    if defect_mask & (1 << 5) != 0 {
        measurements.edge_strength_ratio = above(1.40);
    }
    if defect_mask & (1 << 6) != 0 {
        measurements.median_edge_orientation_difference_degrees = above(10.0);
    }
    if defect_mask & (1 << 7) != 0 {
        measurements.preserves_convex_orientation = false;
    }
    MeasurementsView { ..measurements }
}

fn station_relation_reason_oracle(defect_mask: u8) -> Vec<&'static str> {
    STATION_RELATION_DEFECT_REASONS
        .iter()
        .enumerate()
        .filter_map(|(defect, reason)| (defect_mask & (1 << defect) != 0).then_some(*reason))
        .collect()
}

fn matrix_bits(matrix: &Matrix3<f64>) -> [u64; 9] {
    let mut bits = [0; 9];
    for (slot, value) in bits.iter_mut().zip(matrix.iter()) {
        *slot = value.to_bits();
    }
    bits
}

fn translation_pose(x: f64, y: f64) -> Matrix3<f64> {
    Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0)
}

fn graph_component_oracle(
    station_member_paths: &[Vec<String>],
    relation_pairs: &[(usize, usize)],
) -> Vec<Vec<String>> {
    let mut component = (0..station_member_paths.len()).collect::<Vec<_>>();
    for &(left, right) in relation_pairs {
        if left >= component.len() || right >= component.len() {
            continue;
        }
        let from = component[right];
        let to = component[left];
        for label in &mut component {
            if *label == from {
                *label = to;
            }
        }
    }
    let mut by_component = BTreeMap::<usize, Vec<String>>::new();
    for (station, members) in station_member_paths.iter().enumerate() {
        by_component
            .entry(component[station])
            .or_default()
            .extend(members.iter().cloned());
    }
    let mut components = by_component.into_values().collect::<Vec<_>>();
    for members in &mut components {
        members.sort();
    }
    components.sort();
    components
}

fn assert_graph_outcome_and_publication(
    station_member_paths: &[Vec<String>],
    relation_pairs: &[(usize, usize)],
) -> Result<(), TestCaseError> {
    use super::degradation::{
        DegradationManager, OutcomeDecision, OutputPublication, StandardFinalOutputFileSystem,
    };

    let outcome = crate::panorama_stitching::station_relation_test_access::station_graph_outcome(
        station_member_paths,
        relation_pairs,
    );
    let expected = graph_component_oracle(station_member_paths, relation_pairs);
    let mut actual = outcome.report.connectivity.component_members.clone();
    for members in &mut actual {
        members.sort();
    }
    actual.sort();
    prop_assert_eq!(&actual, &expected);
    prop_assert_eq!(outcome.report.connectivity.components, expected.len());
    let expected_counts = expected.iter().map(Vec::len).collect::<Vec<_>>();
    let mut actual_counts = outcome.report.connectivity.component_member_counts.clone();
    actual_counts.sort_unstable();
    let mut expected_counts_sorted = expected_counts.clone();
    expected_counts_sorted.sort_unstable();
    prop_assert_eq!(actual_counts, expected_counts_sorted);

    let disconnected = expected.len() > 1;
    prop_assert_eq!(
        outcome.rejection_reason,
        disconnected.then_some(degradation::GEOMETRY_DISCONNECTED)
    );
    let mut ledger = DegradationLedger::new();
    if let (Some(reason), Some(detail)) =
        (outcome.rejection_reason, outcome.rejection_detail.clone())
    {
        ledger.record(reason, detail);
    }

    let directory = TempDir::new()
        .map_err(|error| TestCaseError::fail(format!("publication tempdir failed: {error}")))?;
    let staged = directory.path().join("result.tmp");
    let final_path = directory.path().join("result.tiff");
    fs::write(&staged, b"property-32-output")
        .map_err(|error| TestCaseError::fail(format!("staged output write failed: {error}")))?;
    let mut file_system = StandardFinalOutputFileSystem;
    let publication = DegradationManager::new(&ledger)
        .publish_staged_output(&mut file_system, &staged, &final_path)
        .map_err(TestCaseError::fail)?;

    if disconnected {
        prop_assert_eq!(publication, OutputPublication::Rejected);
        prop_assert!(!staged.exists());
        prop_assert!(!final_path.exists());
        prop_assert_eq!(
            DegradationManager::new(&ledger).decision(),
            OutcomeDecision::Rejected
        );
        let error = outcome.user_error.as_deref().unwrap_or_default();
        let expected_fragment = format!("{} 个不连通分量", expected.len());
        prop_assert!(
            error.contains(&expected_fragment),
            "error must name `{}`",
            expected_fragment
        );
        prop_assert!(error.contains("按连续场景重新分组"));
        prop_assert!(error.contains("补拍增加重叠"));
    } else {
        prop_assert_eq!(
            publication,
            OutputPublication::Published(OutcomeDecision::Success)
        );
        prop_assert!(!staged.exists());
        prop_assert!(final_path.exists());
        prop_assert_eq!(outcome.rejection_reason, None);
        prop_assert_eq!(outcome.user_error, None);
    }
    Ok(())
}

fn station_position_key(rows: &[u32], columns: &[u32], station: usize) -> (u32, u32, usize) {
    (rows[station], columns[station], station)
}

fn tree_edge_key(
    rows: &[u32],
    columns: &[u32],
    left: usize,
    right: usize,
) -> ((u32, u32, usize), (u32, u32, usize)) {
    let left = station_position_key(rows, columns, left);
    let right = station_position_key(rows, columns, right);
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn exhaustive_maximum_inlier_tree_oracle(
    station_count: usize,
    rows: &[u32],
    columns: &[u32],
    relations: &[(usize, usize, usize, f64)],
) -> Vec<(usize, usize)> {
    let mut precedence = (0..relations.len()).collect::<Vec<_>>();
    precedence.sort_by(|&left_index, &right_index| {
        let left = relations[left_index];
        let right = relations[right_index];
        right.2.cmp(&left.2).then_with(|| {
            tree_edge_key(rows, columns, left.0, left.1)
                .cmp(&tree_edge_key(rows, columns, right.0, right.1))
        })
    });

    let mut best: Option<(usize, Vec<bool>, u64)> = None;
    for subset in 0u64..(1u64 << relations.len()) {
        if subset.count_ones() as usize != station_count.saturating_sub(1) {
            continue;
        }
        let mut reached = vec![false; station_count];
        reached[0] = true;
        loop {
            let mut changed = false;
            for (edge, &(left, right, _, _)) in relations.iter().enumerate() {
                if subset & (1 << edge) == 0 {
                    continue;
                }
                if reached[left] ^ reached[right] {
                    reached[left] = true;
                    reached[right] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if reached.iter().any(|reached| !reached) {
            continue;
        }
        let total = relations
            .iter()
            .enumerate()
            .filter(|(edge, _)| subset & (1 << edge) != 0)
            .map(|(_, relation)| relation.2)
            .sum::<usize>();
        let signature = precedence
            .iter()
            .map(|&edge| subset & (1 << edge) != 0)
            .collect::<Vec<_>>();
        if best.as_ref().is_none_or(|(best_total, best_signature, _)| {
            total > *best_total || (total == *best_total && signature > *best_signature)
        }) {
            best = Some((total, signature, subset));
        }
    }
    let subset = best.expect("a complete graph always has a spanning tree").2;
    precedence
        .into_iter()
        .filter(|edge| subset & (1 << edge) != 0)
        .map(|edge| {
            let (left, right, _, _) = relations[edge];
            (left.min(right), left.max(right))
        })
        .collect()
}

fn relation_kind(
    rows: &[u32],
    columns: &[u32],
    left: usize,
    right: usize,
) -> super::report::AdjacencyKind {
    if columns[left] == columns[right] {
        super::report::AdjacencyKind::SameColumn
    } else if rows[left] == rows[right] {
        super::report::AdjacencyKind::SameRow
    } else {
        super::report::AdjacencyKind::CrossColumn
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 30: every
    // Station_Relation defect maps to its exact stable reason, combined defects
    // remain complete, report buckets aggregate defects, and rejected candidates
    // never cross the pose-write boundary.
    //
    // **Validates: Requirements 6.5, 6.9, 6.10**
    #[test]
    fn property_30_station_relation_defects_map_to_exact_reasons(seed in any::<u64>()) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_stitching::station_relation_test_access;

        let mut masks = (0u16..=u8::MAX as u16).map(|mask| mask as u8).collect::<Vec<_>>();
        let rotation = seed as usize % masks.len();
        masks.rotate_left(rotation);
        if seed & 1 != 0 {
            masks.reverse();
        }
        let candidates = masks
            .iter()
            .enumerate()
            .map(|(candidate, &mask)| (candidate, candidate + 1_000, relation_measurements_with_defects(mask)))
            .collect::<Vec<_>>();
        let batch = station_relation_test_access::relation_decision_batch(&candidates);

        for &mask in &masks {
            let actual = station_relation_test_access::rejection_reasons(
                relation_measurements_with_defects(mask),
            );
            let expected = station_relation_reason_oracle(mask);
            prop_assert_eq!(actual, expected, "defect mask {:#010b}", mask);
        }
        prop_assert_eq!(batch.report.rejected_count, u8::MAX as usize);
        prop_assert_eq!(batch.pose_writes.len(), 1);
        let accepted_position = masks.iter().position(|&mask| mask == 0).unwrap();
        prop_assert_eq!(batch.pose_writes[0], (accepted_position, accepted_position + 1_000));
        for reason in STATION_RELATION_DEFECT_REASONS {
            prop_assert_eq!(batch.report.rejected_by_reason.get(reason), Some(&128));
            prop_assert!(reason_identifier_pattern().is_match(reason));
        }
        prop_assert_eq!(
            station_relation_test_access::rejection_reasons(relation_measurements_with_defects(u8::MAX)),
            STATION_RELATION_DEFECT_REASONS,
            "the all-defects witness must report every independent failure"
        );
    }

    // Feature: layered-camera-group-focus-stitching, Property 31: accepted
    // projective poses retain all eight solved parameters and bounded
    // Local_Scale; an excessive candidate records its reason and returns the
    // spanning-tree seed bit-for-bit.
    //
    // **Validates: Requirements 6.6**
    #[test]
    fn property_31_station_poses_keep_eight_dof_and_bounded_local_scale(
        seed in any::<u64>(),
        width in 128u32..768,
        height in 128u32..768,
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_stitching::station_relation_test_access;

        let signed = |shift: u32, magnitude: f64| {
            let value = (((seed.rotate_left(shift) >> 11) & 0xffff) as f64 + 1.0) / 65_536.0;
            let sign = if seed.rotate_left(shift) & 1 == 0 { 1.0 } else { -1.0 };
            sign * magnitude * value
        };
        let angle = signed(3, 0.035);
        let (sin, cos) = angle.sin_cos();
        let scale_x = 1.0 + signed(7, 0.018);
        let scale_y = 1.0 + signed(11, 0.018);
        let shear = signed(17, 0.008);
        let extent = f64::from(width.max(height));
        let candidate = Matrix3::new(
            scale_x * cos,
            -scale_y * sin + shear,
            5.0 + signed(19, 80.0),
            scale_x * sin + shear * 0.5,
            scale_y * cos,
            -7.0 + signed(23, 80.0),
            signed(29, 0.002) / extent,
            signed(31, 0.002) / extent,
            1.0,
        );
        let tree_seed = translation_pose(11.25, -19.5);
        let accepted = station_relation_test_access::select_pose_candidate(
            candidate,
            tree_seed,
            (width, height),
        );
        let minimum = accepted.minimum_local_scale.expect("general pose has measurable Local_Scale");
        let maximum = accepted.maximum_local_scale.expect("general pose has measurable Local_Scale");
        prop_assert!(minimum > 0.0);
        prop_assert!(maximum / minimum <= 1.10);
        prop_assert_eq!(accepted.rejection_reason, None);
        prop_assert_eq!(matrix_bits(&accepted.selected), matrix_bits(&candidate));
        prop_assert_ne!(candidate[(0, 0)], 1.0, "scale remains free");
        prop_assert_ne!(candidate[(0, 1)], 0.0, "rotation/shear remains free");
        prop_assert_ne!(candidate[(1, 0)], 0.0, "rotation/shear remains free");
        prop_assert_ne!(candidate[(2, 0)], 0.0, "x perspective remains free");
        prop_assert_ne!(candidate[(2, 1)], 0.0, "y perspective remains free");

        let excessive = Matrix3::new(
            1.0,
            0.01,
            31.0,
            -0.02,
            1.0,
            -17.0,
            0.45 / f64::from(width),
            0.20 / f64::from(height),
            1.0,
        );
        let rejected = station_relation_test_access::select_pose_candidate(
            excessive,
            tree_seed,
            (width, height),
        );
        let rejected_ratio = rejected.maximum_local_scale.unwrap() / rejected.minimum_local_scale.unwrap();
        prop_assert!(rejected_ratio > 1.10);
        prop_assert_eq!(rejected.rejection_reason, Some(degradation::STATION_POSE_LOCAL_SCALE_EXCEEDED));
        prop_assert_eq!(matrix_bits(&rejected.selected), matrix_bits(&tree_seed));
    }

    // Feature: layered-camera-group-focus-stitching, Property 32: arbitrary
    // disconnected station graphs are rejected at the real publication
    // boundary with canonical components and actionable diagnostics; connected
    // graphs publish and never carry geometry_disconnected.
    //
    // **Validates: Requirements 6.7, 12.4**
    #[test]
    fn property_32_disconnected_station_graphs_reject_publication(
        seed in any::<u64>(),
        station_count in 2usize..7,
        edge_mask in any::<u64>(),
    ) {
        let station_member_paths = (0..station_count)
            .map(|station| {
                let member_count = 1 + ((seed.rotate_left(station as u32) as usize) % 3);
                (0..member_count)
                    .map(|member| format!("/synthetic/property-32/station-{station:02}/capture-{member:02}.raw"))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let all_pairs = (0..station_count)
            .flat_map(|left| ((left + 1)..station_count).map(move |right| (left, right)))
            .collect::<Vec<_>>();
        let random_relations = all_pairs
            .iter()
            .enumerate()
            .filter_map(|(edge, &pair)| (edge_mask & (1 << edge) != 0).then_some(pair))
            .collect::<Vec<_>>();
        assert_graph_outcome_and_publication(&station_member_paths, &random_relations)?;

        let connected_chain = (0..station_count - 1)
            .map(|station| (station, station + 1))
            .collect::<Vec<_>>();
        assert_graph_outcome_and_publication(&station_member_paths, &connected_chain)?;

        let split = 1 + seed as usize % (station_count - 1);
        let disconnected = all_pairs
            .into_iter()
            .filter(|&(left, right)| (left < split) == (right < split))
            .collect::<Vec<_>>();
        assert_graph_outcome_and_publication(&station_member_paths, &disconnected)?;
    }

    // Feature: layered-camera-group-focus-stitching, Property 33: production
    // tree selection equals an exhaustive maximum-inlier oracle; equal weights
    // use row/column order only and are independent of relation order and score.
    //
    // **Validates: Requirements 7.1**
    #[test]
    fn property_33_spanning_tree_is_maximum_support_with_stable_ties(
        seed in any::<u64>(),
        station_count in 2usize..6,
    ) {
        use crate::panorama_stitching::station_relation_test_access;

        let mut station_at_position = (0..station_count).collect::<Vec<_>>();
        let mut shuffle_state = seed ^ 0x33a5_9d91;
        for index in (1..station_at_position.len()).rev() {
            let other = splitmix64(&mut shuffle_state) as usize % (index + 1);
            station_at_position.swap(index, other);
        }
        let mut rows = vec![0; station_count];
        let mut columns = vec![0; station_count];
        for (position, &station) in station_at_position.iter().enumerate() {
            rows[station] = (position / 3) as u32;
            columns[station] = (position % 3) as u32;
        }
        let mut relations = (0..station_count)
            .flat_map(|left| ((left + 1)..station_count).map(move |right| (left, right)))
            .map(|(left, right)| {
                let inliers = 24 + (splitmix64(&mut shuffle_state) as usize % 5);
                let score = unit_from(&mut shuffle_state) * 1_000_000.0;
                if splitmix64(&mut shuffle_state) & 1 == 0 {
                    (left, right, inliers, score)
                } else {
                    (right, left, inliers, score)
                }
            })
            .collect::<Vec<_>>();
        let expected = exhaustive_maximum_inlier_tree_oracle(
            station_count,
            &rows,
            &columns,
            &relations,
        );
        let actual = station_relation_test_access::maximum_inlier_tree(
            station_count,
            &rows,
            &columns,
            &relations,
        );
        prop_assert_eq!(&actual, &expected);
        let actual_weight = actual
            .iter()
            .map(|&(left, right)| {
                relations.iter().find(|relation| {
                    (relation.0.min(relation.1), relation.0.max(relation.1)) == (left, right)
                }).unwrap().2
            })
            .sum::<usize>();
        let expected_weight = expected
            .iter()
            .map(|&(left, right)| relations.iter().find(|relation| {
                (relation.0.min(relation.1), relation.0.max(relation.1)) == (left, right)
            }).unwrap().2)
            .sum::<usize>();
        prop_assert_eq!(actual_weight, expected_weight);

        relations.reverse();
        for relation in &mut relations {
            relation.3 = 1_000_000_000.0 - relation.3;
        }
        let reordered = station_relation_test_access::maximum_inlier_tree(
            station_count,
            &rows,
            &columns,
            &relations,
        );
        prop_assert_eq!(reordered, expected);
    }

    // Feature: layered-camera-group-focus-stitching, Property 34: every
    // accepted horizontal, vertical, and cross-column relation reaches the
    // production closure report and final weight records, including an edge
    // beyond the removed large-residual pre-drop threshold.
    //
    // **Validates: Requirements 7.2**
    #[test]
    fn property_34_all_accepted_relations_participate_in_closure(seed in any::<u64>()) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_stitching::station_relation_test_access::{self, ClosureRelationView};

        let rows = vec![0, 0, 1, 1];
        let columns = vec![0, 1, 0, 1];
        let positions = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)];
        let base_poses = positions
            .iter()
            .map(|&(x, y)| translation_pose(x, y))
            .collect::<Vec<_>>();
        let pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];
        let large_pair = (0, 3);
        let mut state = seed ^ 0x34c1_05e5;
        let relations = pairs
            .iter()
            .map(|&(left, right)| {
                let ideal_x = positions[left].0 - positions[right].0;
                let ideal_y = positions[left].1 - positions[right].1;
                let (residual_x, residual_y) = if (left, right) == large_pair {
                    (120.0 + 20.0 * unit_from(&mut state), 10.0)
                } else {
                    (unit_from(&mut state) - 0.5, unit_from(&mut state) - 0.5)
                };
                ClosureRelationView {
                    score: 0.5 + unit_from(&mut state),
                    left,
                    right,
                    left_to_right: translation_pose(ideal_x + residual_x, ideal_y + residual_y),
                    independent_support: 2,
                    median_error_px: 3.0 * unit_from(&mut state),
                }
            })
            .collect::<Vec<_>>();
        let closure = station_relation_test_access::closure_run(
            (400, 400),
            &rows,
            &columns,
            &base_poses,
            &relations,
        );
        let expected_pairs = pairs
            .iter()
            .map(|&(left, right)| (left.min(right), left.max(right)))
            .collect::<BTreeSet<_>>();
        let recorded_pairs = closure
            .report
            .constraint_weights
            .iter()
            .map(|record| (record.left.min(record.right), record.left.max(record.right)))
            .collect::<BTreeSet<_>>();
        prop_assert_eq!(closure.report.participating_constraints, relations.len());
        prop_assert_eq!(closure.report.constraint_weights.len(), relations.len());
        prop_assert_eq!(recorded_pairs, expected_pairs);
        prop_assert_eq!(closure.poses.len(), base_poses.len());

        let kinds = pairs
            .iter()
            .map(|&(left, right)| relation_kind(&rows, &columns, left, right))
            .collect::<Vec<_>>();
        prop_assert!(kinds.contains(&super::report::AdjacencyKind::SameRow));
        prop_assert!(kinds.contains(&super::report::AdjacencyKind::SameColumn));
        prop_assert!(kinds.contains(&super::report::AdjacencyKind::CrossColumn));
        let large = closure
            .report
            .constraint_weights
            .iter()
            .find(|record| (record.left.min(record.right), record.left.max(record.right)) == large_pair)
            .expect("large accepted relation must have a final weight record");
        prop_assert!(large.residual_px > 80.0, "large residual witness must exceed the removed pre-drop threshold");
        prop_assert!(large.weight.is_finite());
    }
}

// ---------------------------------------------------------------------------
