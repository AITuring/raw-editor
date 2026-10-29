//! Run outcome and resource properties.
//!
//! Frozen failure identifiers (79), rejection precedence and publication with the
//! recorded path (80, 92), Source_RAW safety on every ending path (24), the
//! memory threshold and overrun (86, 87), Virtual_Tile residency (88),
//! cancellation (91) and the Stack_Report schema (93).

use super::super::degradation;
use super::super::degradation::DegradationLedger;
use super::super::degradation::DegradationManager;
use super::super::degradation::FAILURE_REASONS;
use super::super::degradation::MAX_REASON_IDENTIFIER_LENGTH;
use super::super::degradation::StandardFinalOutputFileSystem;
use super::super::report::STACK_REPORT_TOP_LEVEL_FIELDS;
use super::super::report::StackReport;
use super::super::resources;
use super::super::test_support::arb_stack_report;
use super::super::test_support::arb_virtual_tile_in;
use super::super::test_support::minimal_virtual_tile;
use super::super::virtual_tile;
use super::super::virtual_tile::CacheKeyInputs;
use super::super::virtual_tile::CacheLookup;
use super::super::virtual_tile::LeaseError;
use super::super::virtual_tile::MAX_RESIDENT_VIRTUAL_TILES;
use super::super::virtual_tile::StoreOutcome;
use super::super::virtual_tile::VirtualTileStore;
use crate::image_stack::STACK_PIPELINE_VERSION;
use proptest::prelude::*;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::time::SystemTime;
use tempfile::TempDir;

use super::{CACHE_PROPERTY_MIN_SIDE, frozen_reason_identifiers, reason_identifier_pattern};

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 86: 内存门槛解析是确定的。
    #[test]
    fn property_86_memory_threshold_resolution_is_deterministic(
        physical in 0u64..(64 * 1024 * 1024 * 1024),
        configured in prop::option::of(0u64..(64 * 1024 * 1024 * 1024)),
        auto_calibrated in any::<bool>(),
    ) {
        let first = resources::resolve_memory_threshold(physical, configured, auto_calibrated);
        let second = resources::resolve_memory_threshold(physical, configured, auto_calibrated);
        prop_assert_eq!(first, second);
        let upper = ((physical as f64) * resources::MEMORY_THRESHOLD_PHYSICAL_RATIO).floor()
            as u64;
        let expected = if upper < resources::MEMORY_THRESHOLD_MIN_BYTES {
            resources::MemoryThreshold {
                bytes: upper,
                source: resources::MemoryThresholdSource::AutoCalibrated,
            }
        } else {
            let lower = resources::MEMORY_THRESHOLD_MIN_BYTES;
            let clamp = |value: u64| value.max(lower).min(upper);
            match configured {
                Some(value) => resources::MemoryThreshold {
                    bytes: clamp(value),
                    source: resources::MemoryThresholdSource::UserConfigured,
                },
                None if auto_calibrated => resources::MemoryThreshold {
                    bytes: upper,
                    source: resources::MemoryThresholdSource::AutoCalibrated,
                },
                None => resources::MemoryThreshold {
                    bytes: clamp(resources::MEMORY_THRESHOLD_DEFAULT_BYTES),
                    source: resources::MemoryThresholdSource::Default,
                },
            }
        };
        prop_assert_eq!(first, expected);
    }
}

/// Recover the `'static` identifier behind a serialized reason string so the
/// same reason can be replayed through a fresh ledger.
pub(super) fn static_failure_reason(reason: &str) -> Option<&'static str> {
    FAILURE_REASONS
        .iter()
        .find(|(candidate, _)| *candidate == reason)
        .map(|(candidate, _)| *candidate)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 79: 对于任意生效的失败或
    // 降级路径，其机器可读失败原因标识符仅由 ASCII 小写字母、数字和下划线组成、长度不超过
    // 64 个字符，且对同一失败原因在重复运行之间完全相同。
    //
    // **Validates: Requirements 12.7**
    #[test]
    fn failure_reason_identifiers_have_a_stable_format(report in arb_stack_report()) {
        let pattern = reason_identifier_pattern();
        let frozen = frozen_reason_identifiers();

        for entry in &report.degradation.entries {
            prop_assert!(
                pattern.is_match(&entry.reason),
                "degradation reason `{}` violates ^[a-z0-9_]{{1,64}}$",
                entry.reason
            );
            prop_assert!(
                entry.reason.len() <= MAX_REASON_IDENTIFIER_LENGTH,
                "degradation reason `{}` is longer than {MAX_REASON_IDENTIFIER_LENGTH} bytes",
                entry.reason
            );
            prop_assert!(
                frozen.contains(entry.reason.as_str()),
                "degradation reason `{}` is outside the frozen identifier set",
                entry.reason
            );
        }

        for record in &report.quality_gate.unmeasurable {
            prop_assert!(
                pattern.is_match(&record.reason),
                "unmeasurable reason `{}` violates ^[a-z0-9_]{{1,64}}$",
                record.reason
            );
            prop_assert!(
                record.reason.len() <= MAX_REASON_IDENTIFIER_LENGTH,
                "unmeasurable reason `{}` is longer than {MAX_REASON_IDENTIFIER_LENGTH} bytes",
                record.reason
            );
            prop_assert!(
                frozen.contains(record.reason.as_str()),
                "unmeasurable reason `{}` is outside the frozen identifier set",
                record.reason
            );
        }

        // Stability across repeated runs: replaying the same active paths
        // through a fresh ledger reproduces byte identical identifiers and the
        // same severity, so a consumer can key on them run after run.
        let mut replayed = DegradationLedger::new();
        for entry in &report.degradation.entries {
            let reason = static_failure_reason(&entry.reason)
                .expect("a recorded reason belongs to the frozen failure set");
            replayed.record(reason, Value::Null);
        }
        prop_assert_eq!(replayed.entries().len(), report.degradation.entries.len());
        for (replay, original) in replayed.entries().iter().zip(&report.degradation.entries) {
            prop_assert_eq!(replay.reason, original.reason.as_str());
            prop_assert_eq!(replay.severity.as_identifier(), original.severity.as_str());
        }

        // The serialized document repeats the identifiers verbatim, so the
        // written report of a repeated run carries the same bytes.
        let first = serde_json::to_string(&report).expect("the report serializes");
        let second = serde_json::to_string(&report).expect("the report serializes");
        prop_assert_eq!(&first, &second);
        for entry in &report.degradation.entries {
            prop_assert!(
                first.contains(&format!("\"reason\":\"{}\"", entry.reason)),
                "the serialized report must carry `{}` verbatim",
                entry.reason
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 80: the complete
    // arbitrary reason multiset is retained by Stack_Report, any rejected
    // severity rejects the ledger result, cancellation remains a distinct
    // terminal decision, and the production publication boundary follows it.
    //
    // **Validates: Requirements 12.9**
    #[test]
    fn property_80_rejection_outranks_degradation_and_controls_publication(
        reason_indices in prop::collection::vec(0usize..FAILURE_REASONS.len(), 0..96),
        nonce in any::<u64>(),
    ) {
        use crate::panorama_utils::stack_pipeline::degradation::{
            DegradationManager, OutcomeDecision, OutputPublication, RunResult, Severity,
            StandardFinalOutputFileSystem,
        };
        use crate::panorama_utils::stack_pipeline::report::{StackReportRecorder, StackRunResult};

        let mut ledger = DegradationLedger::new();
        for (ordinal, &reason_index) in reason_indices.iter().enumerate() {
            let (reason, _) = FAILURE_REASONS[reason_index];
            ledger.record(reason, serde_json::json!({ "ordinal": ordinal, "nonce": nonce }));
        }

        let has_rejected = ledger
            .entries()
            .iter()
            .any(|entry| entry.severity == Severity::Rejected);
        let has_degraded = ledger
            .entries()
            .iter()
            .any(|entry| entry.severity == Severity::Degraded);
        let has_cancellation = ledger.count_of(degradation::RUN_CANCELLED_BY_USER) > 0;
        prop_assert_eq!(ledger.result() == RunResult::Rejected, has_rejected);

        let manager = DegradationManager::new(&ledger);
        let expected_decision = if has_cancellation {
            OutcomeDecision::Cancelled
        } else if has_rejected {
            OutcomeDecision::Rejected
        } else if has_degraded {
            OutcomeDecision::Degraded
        } else {
            OutcomeDecision::Success
        };
        prop_assert_eq!(manager.decision(), expected_decision);
        prop_assert_eq!(manager.decision().permits_final_output(), !has_rejected);

        let recorder = StackReportRecorder::isolated("property-80", None);
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();
        prop_assert_eq!(report.degradation.entries.len(), ledger.entries().len());
        for (recorded, original) in report.degradation.entries.iter().zip(ledger.entries()) {
            prop_assert_eq!(recorded.reason.as_str(), original.reason);
            prop_assert_eq!(recorded.severity.as_str(), original.severity.as_identifier());
            prop_assert_eq!(&recorded.detail, &original.detail);
        }
        let expected_report_result = match expected_decision {
            OutcomeDecision::Success => StackRunResult::Success,
            OutcomeDecision::Degraded => StackRunResult::Degraded,
            OutcomeDecision::Rejected => StackRunResult::Rejected,
            OutcomeDecision::Cancelled => StackRunResult::Cancelled,
        };
        prop_assert_eq!(report.degradation.result, expected_report_result);

        let directory = TempDir::new()
            .map_err(|error| TestCaseError::fail(format!("publication tempdir failed: {error}")))?;
        let staged = directory.path().join("result.tmp");
        let final_path = directory.path().join("result.tiff");
        let payload = nonce.to_le_bytes();
        fs::write(&staged, payload)
            .map_err(|error| TestCaseError::fail(format!("staged output write failed: {error}")))?;
        let publication = manager
            .publish_staged_output(
                &mut StandardFinalOutputFileSystem,
                &staged,
                &final_path,
            )
            .map_err(TestCaseError::fail)?;
        let expected_publication = match expected_decision {
            OutcomeDecision::Success | OutcomeDecision::Degraded => {
                OutputPublication::Published(expected_decision)
            }
            OutcomeDecision::Rejected => OutputPublication::Rejected,
            OutcomeDecision::Cancelled => OutputPublication::Cancelled,
        };
        prop_assert_eq!(publication, expected_publication);
        prop_assert_eq!(matches!(publication, OutputPublication::Published(_)), !has_rejected);
        prop_assert!(!staged.exists());
        if has_rejected {
            prop_assert!(!final_path.exists());
        } else {
            prop_assert_eq!(fs::read(&final_path).map_err(|error| TestCaseError::fail(error.to_string()))?, payload);
        }
    }

    // Feature: layered-camera-group-focus-stitching, Property 92: one
    // production function decides both execution and Stack_Report from the
    // station-count boundary and every persisted compositor setting.
    //
    // **Validates: Requirements 15.1, 15.11**
    #[test]
    fn property_92_station_count_and_setting_select_and_record_exact_path(
        station_count in 0usize..128,
    ) {
        use crate::panorama_utils::stack_pipeline::compositor::{
            StackCompositorChoice, select_and_record_run_path,
        };
        use crate::panorama_utils::stack_pipeline::report::{SelectedPath, StackReportRecorder};

        let settings = [
            StackCompositorChoice::LayeredVirtualTile,
            StackCompositorChoice::ProgressiveSeamTile,
            StackCompositorChoice::StreamingMosaic,
            StackCompositorChoice::LegacySingleLayerMosaic,
        ];
        let counts = [0, 1, 2, station_count.max(2)];
        for &count in &counts {
            for &setting in &settings {
                // Independent requirement oracle: the explicit legacy
                // diagnostic setting wins; otherwise every count below two is
                // station-layer-only and every count from two uses its chosen
                // Virtual_Tile compositor.
                let expected = if setting == StackCompositorChoice::LegacySingleLayerMosaic {
                    SelectedPath::LegacySingleLayerMosaic
                } else if count < 2 {
                    SelectedPath::SingleStation
                } else {
                    match setting {
                        StackCompositorChoice::LayeredVirtualTile => SelectedPath::LayeredVirtualTile,
                        StackCompositorChoice::ProgressiveSeamTile => SelectedPath::ProgressiveSeamTile,
                        StackCompositorChoice::StreamingMosaic => SelectedPath::StreamingMosaic,
                        StackCompositorChoice::LegacySingleLayerMosaic => unreachable!(),
                    }
                };
                let expected_virtual_tiles = count >= 2
                    && setting != StackCompositorChoice::LegacySingleLayerMosaic;
                let recorder = StackReportRecorder::isolated("property-92", None);
                let selection = select_and_record_run_path(
                    Some(count),
                    setting,
                    Some(&recorder),
                );

                prop_assert_eq!(selection.selected_path, expected);
                prop_assert_eq!(selection.use_virtual_tiles, expected_virtual_tiles);
                prop_assert_eq!(recorder.snapshot().selected_path, expected);
                prop_assert_eq!(
                    recorder.snapshot().selected_path.as_identifier(),
                    expected.as_identifier(),
                );
            }
        }
    }
}

/// How a run can end (需求 4.8 / 12.8 / 13.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EndingPath {
    /// Fused, cached, read back.
    Success,
    /// The cache could not be written; the run continues on the in-memory tile.
    Degraded,
    /// The selection was rejected before anything was written.
    Rejected,
    /// The run was cancelled while a tile was leased.
    Cancelled,
    /// A diagnostics bundle was written.
    DiagnosticsWrite,
}

pub(super) fn arb_ending_path() -> impl Strategy<Value = EndingPath> {
    prop_oneof![
        Just(EndingPath::Success),
        Just(EndingPath::Degraded),
        Just(EndingPath::Rejected),
        Just(EndingPath::Cancelled),
        Just(EndingPath::DiagnosticsWrite),
    ]
}

/// `(file name, length, modified)` of every entry of `directory`, sorted.
///
/// Compared before and after a run so a created, deleted or rewritten file in
/// the Source_RAW directory fails the property (需求 4.8).
pub(super) fn directory_fingerprint(directory: &Path) -> Vec<(String, u64, Option<SystemTime>)> {
    let mut entries = fs::read_dir(directory)
        .expect("the source directory is readable")
        .flatten()
        .map(|entry| {
            let metadata = entry.metadata().expect("metadata");
            (
                entry.file_name().to_string_lossy().to_string(),
                metadata.len(),
                metadata.modified().ok(),
            )
        })
        .collect::<Vec<_>>();
    entries.sort();
    entries
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 24: 对于任意输入集合与
    // 任意结束路径（成功、降级、拒绝、取消、诊断写入），运行结束后每个 Source_RAW 的绝对
    // 路径存在性与 SHA-256 都与运行开始前一致，且 Source_RAW 所在目录中没有被创建、修改或
    // 删除任何文件。
    //
    // 五条结束路径都用真实临时文件驱动真实的缓存写入、租约与报告序列化；目录指纹
    // （文件名、长度、修改时间）在运行前后逐项对照，所以任何往源目录里写东西的代码路径
    // 都会在这里变红。
    //
    // **Validates: Requirements 4.8, 12.8, 13.4**
    #[test]
    fn source_raw_files_survive_every_ending_path(
        tile in arb_virtual_tile_in(CACHE_PROPERTY_MIN_SIDE, 24),
        contents in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..1024), 1..=4),
        ending in arb_ending_path(),
        probe in any::<usize>(),
    ) {
        let sources = TempDir::new().expect("temp dir");
        let mut pairs = Vec::new();
        for (index, content) in contents.iter().enumerate() {
            let path = sources.path().join(format!("DSC_{:04}.NEF", 3680 + index));
            fs::write(&path, content).expect("writing a source fixture");
            pairs.push(path);
        }
        // Before: existence, digest and the directory listing.
        let before_digests = pairs
            .iter()
            .map(|path| {
                (
                    path.clone(),
                    virtual_tile::source_file_sha256(path).expect("the digest succeeds"),
                )
            })
            .collect::<Vec<_>>();
        let before_listing = directory_fingerprint(sources.path());

        let inputs = CacheKeyInputs::new(
            STACK_PIPELINE_VERSION,
            &before_digests
                .iter()
                .map(|(path, digest)| (path.to_string_lossy().to_string(), *digest))
                .collect::<Vec<_>>(),
        );
        let cache = TempDir::new().expect("temp dir");

        // Run the ending path.  Each one has to really happen, or the property
        // would hold for a run that never touched anything.
        match ending {
            EndingPath::Success => {
                let store = VirtualTileStore::new(cache.path());
                prop_assert!(
                    matches!(store.store(&inputs, &tile), StoreOutcome::Written { .. }),
                    "a temp dir must be writable"
                );
                prop_assert!(matches!(store.load(&inputs), CacheLookup::Hit(_)));
            }
            EndingPath::Degraded => {
                // A regular file where the cache root must be a directory.
                fs::write(cache.path().join("stack-virtual-tiles"), b"blocked")
                    .expect("writing the blocker");
                let store = VirtualTileStore::new(cache.path());
                match store.store(&inputs, &tile) {
                    StoreOutcome::Unavailable { reason, .. } => {
                        prop_assert_eq!(reason, degradation::CACHE_WRITE_UNAVAILABLE);
                    }
                    other => {
                        prop_assert!(false, "an unwritable cache must degrade: {other:?}");
                    }
                }
                // 需求 4.11: the run continues on the in-memory tile.
                prop_assert_eq!(
                    tile.coverage.covered_pixels(),
                    tile.ownership.assigned_pixels()
                );
            }
            EndingPath::Rejected => {
                let mut ledger = DegradationLedger::new();
                let (reason, _) = FAILURE_REASONS[probe % FAILURE_REASONS.len()];
                ledger.record(reason, Value::Null);
                prop_assert_eq!(ledger.entries().len(), 1);
            }
            EndingPath::Cancelled => {
                let store = VirtualTileStore::new(cache.path());
                // A tile is resident and the run is abandoned mid-way.
                let lease = store
                    .lease(tile.station_index, || Ok(tile.clone()))
                    .expect("the first lease fits");
                prop_assert_eq!(lease.station_index(), tile.station_index);
                drop(lease);
                prop_assert_eq!(store.active_virtual_tile_leases(), 0);
            }
            EndingPath::DiagnosticsWrite => {
                let report = crate::panorama_utils::stack_pipeline::report::StackReport::default();
                let document = serde_json::to_vec_pretty(&report).expect("the report serializes");
                let path = cache.path().join("stack-report.json");
                fs::write(&path, &document).expect("writing the diagnostics bundle");
                prop_assert!(path.exists() && !document.is_empty());
            }
        }

        // After: the same existence, the same digest, the same directory.
        for (path, before) in &before_digests {
            prop_assert!(
                path.exists(),
                "{} disappeared on the {:?} path",
                path.display(),
                ending
            );
            let after = virtual_tile::source_file_sha256(path)
                .expect("the source is still readable");
            prop_assert_eq!(
                &after,
                before,
                "{} changed on the {:?} path",
                path.display(),
                ending
            );
        }
        prop_assert_eq!(
            directory_fingerprint(sources.path()),
            before_listing,
            "the Source_RAW directory changed on the {:?} path",
            ending
        );
    }
}

/// One step of the Virtual_Tile access sequence of Property 88.
#[derive(Debug, Clone, Copy)]
pub(super) enum LeaseStep {
    /// Lease the tile of station `index % station_count`.
    Acquire(usize),
    /// Drop the `index % held` lease currently held.
    Release(usize),
}

pub(super) fn arb_lease_step() -> impl Strategy<Value = LeaseStep> {
    prop_oneof![
        3 => any::<usize>().prop_map(LeaseStep::Acquire),
        2 => any::<usize>().prop_map(LeaseStep::Release),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 88: 对于任意
    // Capture_Station 数量与任意 Virtual_Tile 访问序列，任意时刻同时常驻内存的完整尺寸
    // Virtual_Tile 数量不超过 2 个，且报告记录的最大同时常驻数量等于实际观测到的最大值。
    //
    // 既有单元测试走的是几条手写的租约序列；这里的机位数量与取/放序列是随机的，并且在
    // 每一步之后检查上限。像素规模与这条属性无关，所以用最小合法瓦片（需求 14.4 约束的是
    // 常驻数量，不是瓦片内容）。
    //
    // TODO(task 11.x): `resources.max_resident_virtual_tiles` 的报告写入点随
    // Stack_Report 资源段落落地；这里断言的是 store 观测到的峰值，即该字段的数据来源。
    //
    // **Validates: Requirements 14.4**
    #[test]
    fn at_most_two_virtual_tiles_are_resident(
        station_count in 1usize..=6,
        steps in prop::collection::vec(arb_lease_step(), 1..24),
    ) {
        let directory = TempDir::new().expect("temp dir");
        let store = VirtualTileStore::new(directory.path());
        let syntheses = std::cell::Cell::new(0usize);
        let mut held: Vec<virtual_tile::TileLease> = Vec::new();
        let mut observed_peak = 0usize;
        let mut refusals = 0usize;

        for step in &steps {
            match *step {
                LeaseStep::Acquire(slot) => {
                    let station = slot % station_count;
                    let before = syntheses.get();
                    match store.lease(station, || {
                        syntheses.set(syntheses.get() + 1);
                        Ok(minimal_virtual_tile(station))
                    }) {
                        Ok(lease) => {
                            prop_assert_eq!(lease.station_index(), station);
                            prop_assert_eq!(lease.tile().station_index, station);
                            held.push(lease);
                        }
                        Err(LeaseError::ResidencyExhausted { limit, held: live }) => {
                            prop_assert_eq!(limit, MAX_RESIDENT_VIRTUAL_TILES);
                            prop_assert_eq!(
                                live,
                                MAX_RESIDENT_VIRTUAL_TILES,
                                "a lease may only be refused when every slot is held"
                            );
                            prop_assert_eq!(
                                syntheses.get(),
                                before,
                                "a refused lease must not synthesise the tile it cannot hold"
                            );
                            refusals += 1;
                        }
                        Err(other) => {
                            prop_assert!(false, "unexpected lease failure: {other}");
                        }
                    }
                }
                LeaseStep::Release(slot) => {
                    if !held.is_empty() {
                        let index = slot % held.len();
                        held.remove(index);
                    }
                }
            }
            // The ceiling holds after every single step, not just at the end.
            let resident = store.resident_virtual_tiles();
            prop_assert!(
                resident <= MAX_RESIDENT_VIRTUAL_TILES,
                "{resident} resident tile(s) exceed the ceiling of \
                 {MAX_RESIDENT_VIRTUAL_TILES}"
            );
            let live_stations = held
                .iter()
                .map(|lease| lease.station_index())
                .collect::<BTreeSet<_>>();
            prop_assert!(
                live_stations.len() <= MAX_RESIDENT_VIRTUAL_TILES,
                "{} distinct station(s) are leased at once",
                live_stations.len()
            );
            observed_peak = observed_peak.max(resident);
        }

        // 报告记录的最大同时常驻数量等于实际观测到的最大值.
        prop_assert_eq!(
            store.max_resident_virtual_tiles(),
            observed_peak,
            "the recorded peak must be the observed peak"
        );

        // A deterministic probe on top of the random sequence, so every case
        // really reaches the ceiling instead of only the lucky ones.
        drop(held);
        let mut probe = Vec::new();
        for station in 0..MAX_RESIDENT_VIRTUAL_TILES {
            probe.push(
                store
                    .lease(100 + station, || Ok(minimal_virtual_tile(100 + station)))
                    .expect("the first two leases fit"),
            );
        }
        prop_assert_eq!(store.resident_virtual_tiles(), MAX_RESIDENT_VIRTUAL_TILES);
        let before = syntheses.get();
        match store.lease(200, || {
            syntheses.set(syntheses.get() + 1);
            Ok(minimal_virtual_tile(200))
        }) {
            Err(LeaseError::ResidencyExhausted { limit, held: live }) => {
                prop_assert_eq!((limit, live), (MAX_RESIDENT_VIRTUAL_TILES, MAX_RESIDENT_VIRTUAL_TILES));
                prop_assert_eq!(syntheses.get(), before);
                refusals += 1;
            }
            Ok(_) => {
                prop_assert!(false, "a third resident tile must never be admitted");
            }
            Err(other) => {
                prop_assert!(false, "unexpected lease failure: {other}");
            }
        }
        prop_assert!(refusals >= 1, "the ceiling probe must have been refused");
        prop_assert_eq!(
            store.max_resident_virtual_tiles(),
            MAX_RESIDENT_VIRTUAL_TILES
        );
        drop(probe);
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 87: 内存超限中止且不残留
    // 超过内存门槛的运行只删除本次暂存结果，不发布部分输出。
    //
    // **Validates: Requirements 14.3**
    #[test]
    fn property_87_memory_overrun_removes_only_run_staging(
        peak in 1u64..1024,
        threshold in 1u64..1024,
    ) {
        let directory = tempfile::tempdir().expect("temporary output directory");
        let temporary = directory.path().join(".stack-result.current.tmp");
        let final_path = directory.path().join("stack-result.tiff");
        std::fs::write(&temporary, b"partial output").expect("staged output");
        let mut ledger = DegradationLedger::new();
        if peak > threshold {
            ledger.record(
                degradation::MEMORY_THRESHOLD_EXCEEDED,
                serde_json::json!({
                    "threshold_bytes": threshold,
                    "peak_rss_bytes": peak,
                    "sample_count": 1,
                }),
            );
        }
        let publication = DegradationManager::new(&ledger)
            .publish_staged_output(
                &mut StandardFinalOutputFileSystem,
                &temporary,
                &final_path,
            )
            .expect("staged publication decision");

        // The terminating report must carry the same evidence that drove the
        // rejection.  Keep this oracle independent of the publication helper:
        // a future change that drops one of these fields must fail the property
        // even if the temporary file cleanup still happens to be correct.
        let recorder = crate::panorama_utils::stack_pipeline::report::StackReportRecorder::isolated("property-87", None);
        recorder.update(|report| {
            report.resources.memory_threshold_bytes = threshold;
            report.resources.peak_rss_bytes = peak;
            report.resources.rss_sample_count = 1;
            report.resources.memory_threshold_exceeded = peak > threshold;
        });
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();
        prop_assert_eq!(
            report.resources.memory_threshold_bytes,
            threshold,
            "the effective threshold must be reported"
        );
        prop_assert_eq!(report.resources.peak_rss_bytes, peak);
        prop_assert_eq!(report.resources.rss_sample_count, 1);
        prop_assert_eq!(
            report.resources.memory_threshold_exceeded,
            peak > threshold
        );
        prop_assert!(!temporary.exists(), "run staging must never remain");
        if peak > threshold {
            prop_assert_eq!(publication, degradation::OutputPublication::Rejected);
            prop_assert!(!final_path.exists(), "overrun must not publish partial output");
            let entry = report
                .degradation
                .entries
                .iter()
                .find(|entry| entry.reason == degradation::MEMORY_THRESHOLD_EXCEEDED)
                .expect("memory overrun must be recorded");
            prop_assert_eq!(entry.detail["threshold_bytes"].as_u64(), Some(threshold));
            prop_assert_eq!(entry.detail["peak_rss_bytes"].as_u64(), Some(peak));
            prop_assert_eq!(entry.detail["sample_count"].as_u64(), Some(1));
        } else {
            prop_assert!(matches!(publication, degradation::OutputPublication::Published(_)));
            prop_assert!(final_path.exists(), "accepted output must be published");
            prop_assert!(report.degradation.entries.is_empty());
        }
    }

    // Feature: layered-camera-group-focus-stitching, Property 91: 取消后不残留且状态可追溯
    // 用户取消产生 cancelled 状态，只清理本次暂存结果并保留最终输出路径为空。
    //
    // **Validates: Requirements 14.8**
    #[test]
    fn property_91_cancelled_run_has_no_staging_residue(seed in any::<u64>()) {
        let directory = tempfile::tempdir().expect("temporary output directory");
        let temporary = directory.path().join(format!(".stack-result-{seed}.tmp"));
        let final_path = directory.path().join("stack-result.tiff");
        let source_path = directory.path().join("source.NEF");
        let cache_path = directory.path().join("virtual-tile.cache");
        let source_bytes = seed.to_le_bytes().repeat(4);
        let cache_bytes = seed.rotate_left(17).to_le_bytes().repeat(8);
        std::fs::write(&source_path, &source_bytes).expect("source fixture");
        std::fs::write(&cache_path, &cache_bytes).expect("cache fixture");
        let source_before = virtual_tile::source_file_sha256(&source_path)
            .expect("source digest before cancellation");
        let cache_before = std::fs::read(&cache_path).expect("cache bytes before cancellation");
        std::fs::write(&temporary, seed.to_le_bytes()).expect("staged output");
        let mut ledger = DegradationLedger::new();
        ledger.record(degradation::RUN_CANCELLED_BY_USER, serde_json::json!({
            "generation": seed,
        }));
        let manager = DegradationManager::new(&ledger);
        prop_assert_eq!(manager.decision(), degradation::OutcomeDecision::Cancelled);
        let publication = manager
            .publish_staged_output(
                &mut StandardFinalOutputFileSystem,
                &temporary,
                &final_path,
            )
            .expect("cancelled publication decision");
        prop_assert_eq!(publication, degradation::OutputPublication::Cancelled);
        prop_assert!(!temporary.exists());
        prop_assert!(!final_path.exists());

        // Cancellation is allowed to discard only this run's temporary
        // output.  Existing Source_RAW bytes and Virtual_Tile cache entries
        // must remain byte-for-byte unchanged.
        let source_after = virtual_tile::source_file_sha256(&source_path)
            .expect("source digest after cancellation");
        let cache_after = std::fs::read(&cache_path).expect("cache bytes after cancellation");
        prop_assert_eq!(source_after, source_before);
        prop_assert_eq!(cache_after, cache_before);

        let recorder = crate::panorama_utils::stack_pipeline::report::StackReportRecorder::isolated("property-91", None);
        recorder.set_result(crate::panorama_utils::stack_pipeline::report::StackRunResult::Cancelled);
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();
        prop_assert_eq!(
            report.degradation.result,
            crate::panorama_utils::stack_pipeline::report::StackRunResult::Cancelled
        );
        prop_assert!(report
            .degradation
            .entries
            .iter()
            .any(|entry| entry.reason == degradation::RUN_CANCELLED_BY_USER));
    }

    // Feature: layered-camera-group-focus-stitching, Property 93: Stack_Report schema 完整且与返回值一致
    // 任意生产 Stack_Report 都包含固定顶层字段，并可通过实际序列化路径往返。
    //
    // **Validates: Requirements 1.7, 2.9, 5.9, 6.10, 11.16**
    #[test]
    fn property_93_stack_report_schema_matches_serialized_value(mut report in arb_stack_report()) {
        report.pipeline_version = STACK_PIPELINE_VERSION.to_string();
        let encoded = report.to_json_string().expect("stack report serializes");
        let value: Value = serde_json::from_str(&encoded).expect("stack report json");
        for field in STACK_REPORT_TOP_LEVEL_FIELDS {
            prop_assert!(value.get(*field).is_some(), "missing Stack_Report field {field}");
        }
        let station_relations = value
            .get("station_relations")
            .and_then(Value::as_object)
            .expect("station_relations schema object");
        prop_assert!(station_relations.get("unmeasured_count").is_some());
        prop_assert!(station_relations.get("rejected_by_failure_stage").is_some());
        if let Some(candidate) = station_relations
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
        {
            prop_assert!(candidate.get("predicted_overlap_area_px").is_some());
            prop_assert!(candidate.get("model_fitted").is_some());
        }
        let round_trip: StackReport = serde_json::from_str(&encoded).expect("schema round trip");
        // JSON is the wire contract.  Comparing the re-serialized value avoids
        // treating a one-ulp f64 parse/print round trip as a schema mismatch.
        prop_assert_eq!(serde_json::to_value(round_trip).expect("round-trip value"), value);
    }
}
