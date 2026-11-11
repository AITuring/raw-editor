//! Byte-bounded, run-local reuse of complete station fusion results.
//!
//! Every hit keeps the original ownership and native source sampling; callers
//! still build their ordinary compositor payload and run all Quality_Gates.
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use super::native_fusion_key::NativeFusionKey;
use crate::panorama_utils::stack_pipeline::{intra_station, report::NativeFusionCacheReport};
use crate::panorama_utils::stitching::{FocusStackBacking, FocusStackTileRender};

pub(super) enum FusionStage {
    Matching,
    Composition,
}

enum Payload {
    Memory(Arc<FocusStackTileRender>),
    Disk(Box<FocusStackBacking>),
}

struct Entry {
    key: NativeFusionKey,
    input_state: [u8; 32],
    payload: Payload,
    bytes: usize,
}

pub(super) struct NativeFusionCache {
    entries: HashMap<usize, Entry>,
    order: VecDeque<usize>,
    stats: NativeFusionCacheReport,
}

pub(super) fn native_fusion_cache_max_bytes() -> usize {
    std::env::var("RAW_EDITOR_STACK_NATIVE_FUSION_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value <= 4 * 1024 * 1024 * 1024)
        .unwrap_or(1024 * 1024 * 1024)
}

fn state_fingerprint(
    record: Option<&crate::panorama_utils::stack_pipeline::report::IntraStationReport>,
) -> Option<[u8; 32]> {
    // Keep every measured field: frame status, inlier support and errors decide
    // the next render's member plan. The renderer looks records up by path,
    // so first-registration order must not change the cache identity.
    // serde_json writes non-finite floats as null. Reject those snapshots
    // rather than conflating distinct invalid measurements in a cache key.
    if record.is_some_and(|station| {
        station.frames.iter().any(|frame| {
            frame.transform.iter().any(|value| !value.is_finite())
                || [
                    frame.inlier_area_coverage,
                    frame.control_point_spacing_px,
                    frame.rejected_control_point_ratio,
                    frame.median_symmetric_error_px,
                    frame.global_median_symmetric_error_px,
                    frame.focal_plane_ratio_before,
                    frame.focal_plane_ratio_after,
                ]
                .iter()
                .any(|value| !value.is_finite())
        })
    }) {
        return None;
    }
    let canonical = record.map(|station| {
        let mut station = station.clone();
        station
            .frames
            .sort_by(|left, right| left.path.cmp(&right.path));
        station
    });
    serde_json::to_vec(&canonical)
        .ok()
        .map(|bytes| Sha256::digest(bytes).into())
}

fn station_state(station: usize) -> Option<[u8; 32]> {
    let records = intra_station::run_records_snapshot();
    state_fingerprint(
        records
            .iter()
            .find(|record| record.station_index == station),
    )
}

impl NativeFusionCache {
    pub(super) fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            stats: NativeFusionCacheReport {
                max_bytes,
                ..Default::default()
            },
        }
    }

    pub(super) fn with_backing(max_bytes: usize, disk_bytes: usize) -> Self {
        let mut cache = Self::new(max_bytes);
        cache.stats.backing.max_bytes = if max_bytes == 0 { 0 } else { disk_bytes };
        cache
    }

    /// Caller checks cancellation/RSS before entry, including cache hits.
    pub(super) fn get_or_render(
        &mut self,
        station: usize,
        key: NativeFusionKey,
        stage: FusionStage,
        render: impl FnOnce() -> Result<FocusStackTileRender, String>,
    ) -> Result<Arc<FocusStackTileRender>, String> {
        self.get_or_render_with_state(station, key, stage, || station_state(station), render)
    }

    fn get_or_render_with_state(
        &mut self,
        station: usize,
        key: NativeFusionKey,
        stage: FusionStage,
        state: impl Fn() -> Option<[u8; 32]>,
        render: impl FnOnce() -> Result<FocusStackTileRender, String>,
    ) -> Result<Arc<FocusStackTileRender>, String> {
        // Capture the INPUT ledger, not the state left after rendering. The
        // first native registration can mark a frame Failed while retaining
        // its global fallback in that render; the next render must reconsider
        // the frame under the newly recorded exclusion evidence.
        let input_state = state();
        if let Some(entry) = self.entries.get(&station) {
            if entry.key == key && Some(entry.input_state) == input_state {
                let start = Instant::now();
                let rendered = match &entry.payload {
                    Payload::Memory(value) => Some(Arc::clone(value)),
                    Payload::Disk(value) => {
                        let result = value.read();
                        self.stats.backing.read_seconds += start.elapsed().as_secs_f64();
                        match result {
                            Ok(rendered) => {
                                self.stats.backing.hits += 1;
                                self.stats.backing.read_bytes += value.file.bytes;
                                Some(Arc::new(rendered))
                            }
                            Err(_) => {
                                self.stats.backing.failures += 1;
                                None
                            }
                        }
                    }
                };
                if let Some(rendered) = rendered {
                    self.stats.hits += 1;
                    match stage {
                        FusionStage::Matching => self.stats.matching_hits += 1,
                        FusionStage::Composition => self.stats.composition_hits += 1,
                    }
                    self.order.retain(|&id| id != station);
                    self.order.push_back(station);
                    return Ok(rendered);
                }
            }
            self.remove(station);
            self.stats.invalidations += 1;
        }
        self.stats.misses += 1;
        match stage {
            FusionStage::Matching => self.stats.matching_misses += 1,
            FusionStage::Composition => self.stats.composition_misses += 1,
        }
        self.stats.fusion_count += 1;
        let start = Instant::now();
        let result = render();
        self.stats.fusion_seconds += start.elapsed().as_secs_f64();
        let rendered = match result {
            Ok(value) => Arc::new(value),
            Err(error) => {
                self.stats.failed_fusions += 1;
                return Err(error);
            }
        };
        // An unattributed fallback cannot stand in for complete native owner
        // evidence. Preserve its streaming result, without retaining it.
        if rendered.masks.is_none() {
            self.stats.unattributed_bypasses += 1;
            return Ok(rendered);
        }
        let Some(input_state) = input_state else {
            self.stats.state_snapshot_bypasses += 1;
            return Ok(rendered);
        };
        let bytes = rendered
            .retained_bytes()
            .saturating_add(std::mem::size_of::<Entry>())
            .saturating_add(128); // conservative hash/LRU/Arc bookkeeping
        // Keep room for multiple stations. Large results use bounded lossless
        // backing, not a one-entry RAM LRU that thrashes on every tile pass.
        let spill = self.stats.backing.max_bytes > 0 && bytes > self.stats.max_bytes / 4;
        let disk_bytes = if spill {
            FocusStackBacking::disk_bytes(&rendered)
        } else {
            0
        };
        if self.stats.max_bytes == 0
            || (!spill && bytes > self.stats.max_bytes)
            || disk_bytes > self.stats.backing.max_bytes
        {
            self.stats.oversized_bypasses += 1;
            return Ok(rendered);
        }
        // Reserve disk capacity BEFORE writing, including the in-progress file.
        while self.stats.backing.resident_bytes.saturating_add(disk_bytes)
            > self.stats.backing.max_bytes
        {
            let Some(&oldest) = self.order.front() else {
                break;
            };
            self.remove(oldest);
            self.stats.evictions += 1;
            self.stats.backing.evictions += 1;
        }
        let (payload, bytes) = if spill {
            let start = Instant::now();
            let result = FocusStackBacking::write(&rendered);
            self.stats.backing.write_seconds += start.elapsed().as_secs_f64();
            let Ok(backed) = result else {
                self.stats.backing.failures += 1;
                return Ok(rendered);
            };
            self.stats.backing.peak_bytes = self
                .stats
                .backing
                .peak_bytes
                .max(self.stats.backing.resident_bytes.saturating_add(disk_bytes));
            self.stats.backing.writes += 1;
            self.stats.backing.written_bytes += disk_bytes;
            let bytes = backed
                .retained_bytes()
                .saturating_add(std::mem::size_of::<Entry>())
                .saturating_add(128);
            if bytes > self.stats.max_bytes {
                self.stats.oversized_bypasses += 1;
                return Ok(rendered);
            }
            (Payload::Disk(Box::new(backed)), bytes)
        } else {
            (Payload::Memory(Arc::clone(&rendered)), bytes)
        };
        while self.stats.resident_bytes.saturating_add(bytes) > self.stats.max_bytes {
            let Some(&oldest) = self.order.front() else {
                break;
            };
            self.remove(oldest);
            self.stats.evictions += 1;
        }
        self.stats.backing.resident_bytes += disk_bytes;
        self.stats.backing.peak_bytes = self
            .stats
            .backing
            .peak_bytes
            .max(self.stats.backing.resident_bytes);
        self.stats.resident_bytes += bytes;
        self.entries.insert(
            station,
            Entry {
                key,
                input_state,
                payload,
                bytes,
            },
        );
        self.order.push_back(station);
        self.stats.resident_entries = self.entries.len();
        self.stats.peak_entries = self.stats.peak_entries.max(self.entries.len());
        self.stats.peak_memory_entries = self.stats.peak_memory_entries.max(
            self.entries
                .values()
                .filter(|entry| matches!(entry.payload, Payload::Memory(_)))
                .count(),
        );
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.resident_bytes);
        Ok(rendered)
    }

    fn remove(&mut self, station: usize) {
        if let Some(entry) = self.entries.remove(&station) {
            self.stats.resident_bytes = self.stats.resident_bytes.saturating_sub(entry.bytes);
            if let Payload::Disk(backed) = entry.payload {
                self.stats.backing.resident_bytes -= backed.file.bytes;
            }
        }
        self.order.retain(|&id| id != station);
        self.entries.shrink_to_fit();
        self.order.shrink_to_fit();
        self.stats.resident_entries = self.entries.len();
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.entries.shrink_to_fit();
        self.order.clear();
        self.order.shrink_to_fit();
        self.stats.resident_bytes = 0;
        self.stats.backing.resident_bytes = 0;
        self.stats.resident_entries = 0;
    }

    pub(super) fn stats(&self) -> NativeFusionCacheReport {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panorama_utils::shared::ImageInfo;
    use crate::panorama_utils::stack_pipeline::report::{
        IntraStationFrameRecord, IntraStationReport,
    };
    use crate::panorama_utils::stack_pipeline::{focus_fuser, report::FusionReport};
    use crate::panorama_utils::stitching::{
        ConfidenceMap, CoverageMask, FocusSourceRefinement, FocusSourceSampling,
        FocusStackTileMasks, OwnershipMap, Projection,
    };
    use image::{GrayImage, Rgb, Rgb32FImage};
    use nalgebra::{Matrix3, Point2, Point3};
    use std::cell::{Cell, RefCell};
    use std::path::PathBuf;

    #[test]
    fn state_fingerprint_ignores_frame_record_arrival_order_but_keeps_evidence() {
        let mut station = IntraStationReport::default();
        station.frames = vec![
            IntraStationFrameRecord {
                path: "b.NEF".to_string(),
                inliers: 30,
                ..Default::default()
            },
            IntraStationFrameRecord {
                path: "a.NEF".to_string(),
                inliers: 40,
                ..Default::default()
            },
        ];
        let fingerprint = state_fingerprint(Some(&station));
        station.frames.reverse();
        assert_eq!(state_fingerprint(Some(&station)), fingerprint);
        station.frames[0].inliers += 1;
        assert_ne!(state_fingerprint(Some(&station)), fingerprint);
    }

    fn key(value: u8) -> NativeFusionKey {
        NativeFusionKey([value; 32])
    }

    fn rendered() -> FocusStackTileRender {
        let source = ImageInfo {
            id: 7,
            filename: "/fixture/owner.NEF".into(),
            width: 16,
            height: 16,
            alignment_image: GrayImage::new(16, 16),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            source_footprint: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        };
        let refinement = FocusSourceRefinement {
            translation: (3, -2),
            source_sampling: Some(FocusSourceSampling::for_test(
                &source,
                &Matrix3::identity(),
                Projection::Planar,
                None,
            )),
            ..Default::default()
        };
        FocusStackTileRender {
            image: Rgb32FImage::from_pixel(16, 16, Rgb([0.25, 0.5, 0.75])),
            masks: Some(FocusStackTileMasks {
                source_refinements: HashMap::from([(7, refinement)]),
                coverage: CoverageMask::from_bytes(16, 16, vec![255; 256]).unwrap(),
                ownership: OwnershipMap::new(
                    16,
                    16,
                    vec![1; 256],
                    vec![PathBuf::from("/fixture/owner.NEF")],
                )
                .unwrap(),
                confidence: ConfidenceMap::per_pixel(16, 16, vec![0.75; 256]).unwrap(),
                textured: vec![255; 256],
                sharpness_evidence: focus_fuser::StationFusion::new(16, 16, 1).sharpness_evidence(),
                native_displacements: Vec::new(),
                fusion: FusionReport::default(),
            }),
            sampling_origin: (-12.5, 8.25),
        }
    }

    fn read(
        cache: &mut NativeFusionCache,
        station: usize,
        stage: FusionStage,
    ) -> Arc<FocusStackTileRender> {
        cache
            .get_or_render_with_state(
                station,
                key(station as u8),
                stage,
                || Some([0; 32]),
                || Ok(rendered()),
            )
            .unwrap()
    }

    #[test]
    fn native_fusion_cache_hit_shares_complete_native_evidence() {
        let mut cache = NativeFusionCache::new(usize::MAX);
        let first = read(&mut cache, 0, FusionStage::Matching);
        let second = cache
            .get_or_render_with_state(
                0,
                key(0),
                FusionStage::Composition,
                || Some([0; 32]),
                || panic!("cache hit must not repeat fusion"),
            )
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(second.image.get_pixel(0, 0), &Rgb([0.25, 0.5, 0.75]));
        assert_eq!(second.sampling_origin, (-12.5, 8.25));
        let masks = second.masks.as_ref().unwrap();
        assert_eq!(masks.ownership.owner_at(7, 9), 1);
        assert_eq!(
            masks.ownership.legend(),
            &[PathBuf::from("/fixture/owner.NEF")]
        );
        assert!(masks.coverage.is_covered(7, 9));
        assert_eq!(masks.confidence.value_at(7, 9), 0.75);
        assert_eq!(masks.textured[7], 255);
        assert_eq!(masks.source_refinements[&7].translation, (3, -2));
        let sampling = masks.source_refinements[&7]
            .source_sampling
            .as_ref()
            .unwrap();
        assert_eq!(
            sampling.source_point(Point3::new(7.0, 9.0, 1.0)),
            Some(Point2::new(7.0, 9.0))
        );
        assert_eq!(
            sampling.reference_cache_fingerprint(),
            first.masks.as_ref().unwrap().source_refinements[&7]
                .source_sampling
                .as_ref()
                .unwrap()
                .reference_cache_fingerprint()
        );
        assert_eq!(cache.stats().matching_misses, 1);
        assert_eq!(cache.stats().composition_hits, 1);
        assert_eq!(cache.stats().fusion_count, 1);
    }

    #[test]
    fn native_fusion_cache_lru_evicts_within_complete_byte_budget() {
        let mut probe = NativeFusionCache::new(usize::MAX);
        read(&mut probe, 0, FusionStage::Matching);
        let entry_bytes = probe.stats().resident_bytes;
        assert!(entry_bytes > rendered().retained_bytes());
        let mut cache = NativeFusionCache::new(entry_bytes * 2);
        read(&mut cache, 0, FusionStage::Matching);
        read(&mut cache, 1, FusionStage::Matching);
        read(&mut cache, 0, FusionStage::Composition);
        read(&mut cache, 2, FusionStage::Composition);
        assert!(cache.entries.contains_key(&0));
        assert!(!cache.entries.contains_key(&1));
        assert!(cache.entries.contains_key(&2));
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.stats().resident_entries, 2);
        assert_eq!(cache.stats().resident_bytes, entry_bytes * 2);
        assert!(cache.stats().peak_bytes <= cache.stats().max_bytes);
    }

    #[test]
    fn native_fusion_cache_oversized_and_disabled_keep_streaming_results() {
        let mut probe = NativeFusionCache::new(usize::MAX);
        read(&mut probe, 0, FusionStage::Matching);
        for budget in [0, probe.stats().resident_bytes - 1] {
            let mut cache = NativeFusionCache::new(budget);
            let first = read(&mut cache, 0, FusionStage::Matching);
            let second = read(&mut cache, 0, FusionStage::Composition);
            assert!(!Arc::ptr_eq(&first, &second));
            assert_eq!(first.image.as_raw(), second.image.as_raw());
            assert_eq!(cache.stats().fusion_count, 2);
            assert_eq!(cache.stats().oversized_bypasses, 2);
            assert_eq!(cache.stats().resident_bytes, 0);
            assert_eq!(cache.stats().resident_entries, 0);
            assert!(cache.order.is_empty());
        }
    }

    #[test]
    fn native_fusion_cache_rechecks_post_registration_state_before_reuse() {
        use crate::panorama_utils::stack_pipeline::report::{
            IntraStationFrameRecord, IntraStationFrameStatus, IntraStationReport,
        };
        let mut cache = NativeFusionCache::new(usize::MAX);
        let ledger = RefCell::new(IntraStationReport {
            anchor_path: "/fixture/anchor.NEF".into(),
            frames: vec![IntraStationFrameRecord {
                path: "/fixture/owner.NEF".into(),
                status: IntraStationFrameStatus::GlobalFallback,
                ..Default::default()
            }],
            ..Default::default()
        });
        let calls = Cell::new(0);
        let first = cache
            .get_or_render_with_state(
                0,
                key(0),
                FusionStage::Matching,
                || state_fingerprint(Some(&ledger.borrow())),
                || {
                    calls.set(calls.get() + 1);
                    // The first fusion has already considered the global fallback
                    // before the registrar publishes this new failure evidence.
                    ledger.borrow_mut().frames[0].status = IntraStationFrameStatus::Failed;
                    Ok(rendered())
                },
            )
            .unwrap();
        let second = cache
            .get_or_render_with_state(
                0,
                key(0),
                FusionStage::Composition,
                || state_fingerprint(Some(&ledger.borrow())),
                || {
                    calls.set(calls.get() + 1);
                    let mut after_exclusion = rendered();
                    after_exclusion.image.put_pixel(0, 0, Rgb([0.0, 0.0, 0.0]));
                    Ok(after_exclusion)
                },
            )
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_ne!(first.image.get_pixel(0, 0), second.image.get_pixel(0, 0));
        let hit = cache
            .get_or_render_with_state(
                0,
                key(0),
                FusionStage::Composition,
                || state_fingerprint(Some(&ledger.borrow())),
                || panic!("stable input ledger should reuse complete fusion"),
            )
            .unwrap();
        assert!(Arc::ptr_eq(&second, &hit));
        // Unchanged anchor is insufficient: new geometric support may change
        // group-join eligibility even without a frame-status transition.
        ledger.borrow_mut().frames[0].inlier_area_coverage = 0.1;
        cache
            .get_or_render_with_state(
                0,
                key(0),
                FusionStage::Composition,
                || state_fingerprint(Some(&ledger.borrow())),
                || {
                    calls.set(calls.get() + 1);
                    Ok(rendered())
                },
            )
            .unwrap();
        cache
            .get_or_render_with_state(
                0,
                key(1),
                FusionStage::Matching,
                || state_fingerprint(Some(&ledger.borrow())),
                || {
                    calls.set(calls.get() + 1);
                    Ok(rendered())
                },
            )
            .unwrap();
        ledger.borrow_mut().anchor_path = "/fixture/replacement-anchor.NEF".into();
        cache
            .get_or_render_with_state(
                0,
                key(1),
                FusionStage::Composition,
                || state_fingerprint(Some(&ledger.borrow())),
                || {
                    calls.set(calls.get() + 1);
                    Ok(rendered())
                },
            )
            .unwrap();
        assert_eq!(calls.get(), 5);
        assert_eq!(cache.stats().invalidations, 4);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.order.len(), 1);
    }

    #[test]
    fn native_fusion_cache_invalid_record_numbers_cannot_collide_as_json_null() {
        use crate::panorama_utils::stack_pipeline::report::{
            IntraStationFrameRecord, IntraStationReport,
        };
        let mut ledger = IntraStationReport {
            frames: vec![IntraStationFrameRecord::default()],
            ..Default::default()
        };
        assert!(state_fingerprint(Some(&ledger)).is_some());
        let mut cache = NativeFusionCache::new(usize::MAX);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            ledger.frames[0].inlier_area_coverage = value;
            assert!(state_fingerprint(Some(&ledger)).is_none());
            cache
                .get_or_render_with_state(
                    0,
                    key(0),
                    FusionStage::Matching,
                    || state_fingerprint(Some(&ledger)),
                    || Ok(rendered()),
                )
                .unwrap();
        }
        assert_eq!(cache.stats().fusion_count, 3);
        assert_eq!(cache.stats().state_snapshot_bypasses, 3);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().resident_bytes, 0);
    }

    #[test]
    fn native_fusion_cache_unattributed_results_keep_streaming() {
        let mut cache = NativeFusionCache::new(usize::MAX);
        for stage in [FusionStage::Matching, FusionStage::Composition] {
            let value = cache
                .get_or_render_with_state(
                    0,
                    key(0),
                    stage,
                    || Some([0; 32]),
                    || {
                        Ok(FocusStackTileRender {
                            image: Rgb32FImage::new(16, 16),
                            masks: None,
                            sampling_origin: (0.0, 0.0),
                        })
                    },
                )
                .unwrap();
            assert!(value.masks.is_none());
        }
        assert_eq!(cache.stats().fusion_count, 2);
        assert_eq!(cache.stats().unattributed_bypasses, 2);
        assert_eq!(cache.stats().resident_bytes, 0);
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn native_fusion_cache_failed_fusions_do_not_poison_retry() {
        let mut cache = NativeFusionCache::new(usize::MAX);
        read(&mut cache, 0, FusionStage::Matching);
        let result = cache.get_or_render_with_state(
            0,
            key(1),
            FusionStage::Composition,
            || Some([0; 32]),
            || Err("fixture fusion failed".into()),
        );
        assert_eq!(result.unwrap_err(), "fixture fusion failed");
        assert_eq!(cache.stats().failed_fusions, 1);
        assert_eq!(cache.stats().resident_bytes, 0);
        assert!(cache.entries.is_empty());
        assert!(cache.order.is_empty());
        let retried = cache
            .get_or_render_with_state(
                0,
                key(1),
                FusionStage::Composition,
                || Some([0; 32]),
                || Ok(rendered()),
            )
            .unwrap();
        assert!(retried.masks.is_some());
        assert_eq!(cache.stats().fusion_count, 3);
        assert_eq!(cache.stats().resident_entries, 1);
    }

    #[test]
    fn native_fusion_cache_hot_metadata_and_clear_are_bounded() {
        let mut cache = NativeFusionCache::new(usize::MAX);
        let first = read(&mut cache, 0, FusionStage::Matching);
        let weak = Arc::downgrade(&first);
        let retained = cache.stats().resident_bytes;
        let order_capacity = cache.order.capacity();
        for _ in 0..1_000 {
            let hit = read(&mut cache, 0, FusionStage::Composition);
            assert!(Arc::ptr_eq(&first, &hit));
            assert_eq!(cache.order.len(), 1);
            assert_eq!(cache.order.capacity(), order_capacity);
            assert_eq!(cache.stats().resident_bytes, retained);
        }
        assert_eq!(cache.stats().fusion_count, 1);
        assert_eq!(cache.stats().hits, 1_000);
        drop(first);
        cache.clear();
        assert!(weak.upgrade().is_none());
        assert_eq!(cache.stats().resident_bytes, 0);
        assert_eq!(cache.stats().resident_entries, 0);
        assert!(cache.order.is_empty());
        assert_eq!(cache.entries.capacity(), 0);
        assert_eq!(cache.order.capacity(), 0);
    }

    #[test]
    fn backed_native_fusion_preserves_evidence_and_invalidates_state() {
        let size = rendered().retained_bytes() + 1024;
        let disk = FocusStackBacking::disk_bytes(&rendered());
        let mut cache = NativeFusionCache::with_backing(size * 3, disk * 3);
        let first = read(&mut cache, 0, FusionStage::Matching);
        read(&mut cache, 1, FusionStage::Matching);
        read(&mut cache, 2, FusionStage::Matching);
        let hit = cache
            .get_or_render_with_state(
                0,
                key(0),
                FusionStage::Composition,
                || Some([0; 32]),
                || panic!("must reuse backed fusion"),
            )
            .unwrap();
        assert_eq!(hit.image.as_raw(), first.image.as_raw());
        assert_eq!(hit.sampling_origin, first.sampling_origin);
        let (a, b) = (hit.masks.as_ref().unwrap(), first.masks.as_ref().unwrap());
        assert_eq!(a.coverage, b.coverage);
        assert_eq!(a.ownership, b.ownership);
        assert_eq!(a.confidence, b.confidence);
        assert_eq!(a.textured, b.textured);
        assert_eq!(a.sharpness_evidence, b.sharpness_evidence);
        assert_eq!(a.fusion, b.fusion);
        assert_eq!(
            a.source_refinements[&7]
                .source_sampling
                .as_ref()
                .unwrap()
                .reference_cache_fingerprint(),
            b.source_refinements[&7]
                .source_sampling
                .as_ref()
                .unwrap()
                .reference_cache_fingerprint()
        );
        assert_eq!(cache.stats().backing.hits, 1);
        assert_eq!(cache.stats().fusion_count, 3);
        cache
            .get_or_render_with_state(
                0,
                key(0),
                FusionStage::Composition,
                || Some([1; 32]),
                || Ok(rendered()),
            )
            .unwrap();
        assert_eq!(cache.stats().fusion_count, 4);
        assert_eq!(cache.stats().invalidations, 1);
        assert!(cache.stats().peak_bytes <= size * 3);
        assert!(cache.stats().backing.peak_bytes <= disk * 3);
        cache.clear();
        assert_eq!(cache.stats().backing.resident_bytes, 0);
    }

    #[test]
    fn backing_disk_eviction_corruption_and_too_small_budget_fall_back() {
        let disk = FocusStackBacking::disk_bytes(&rendered());
        let mut cache =
            NativeFusionCache::with_backing((rendered().retained_bytes() + 1024) * 3, disk * 2);
        for id in 0..3 {
            read(&mut cache, id, FusionStage::Matching);
        }
        assert!(!cache.entries.contains_key(&0));
        let Payload::Disk(value) = &cache.entries[&1].payload else {
            panic!("expected backing")
        };
        value.file.corrupt();
        read(&mut cache, 1, FusionStage::Composition);
        assert_eq!(cache.stats().backing.failures, 1);
        assert_eq!(cache.stats().fusion_count, 4);
        let mut tiny = NativeFusionCache::with_backing(1, disk * 3);
        read(&mut tiny, 0, FusionStage::Matching);
        read(&mut tiny, 0, FusionStage::Composition);
        assert_eq!(tiny.stats().fusion_count, 2);
        assert_eq!(tiny.stats().resident_bytes, 0);
        assert_eq!(tiny.stats().backing.resident_bytes, 0);
    }

    #[test]
    fn native_fusion_cache_report_roundtrip_and_legacy_performance_defaults() {
        use crate::panorama_utils::stack_pipeline::report::PerformanceReport;
        let mut cache = NativeFusionCache::new(1024 * 1024);
        read(&mut cache, 0, FusionStage::Matching);
        read(&mut cache, 0, FusionStage::Composition);
        let stats = cache.stats();
        let roundtrip: NativeFusionCacheReport =
            serde_json::from_slice(&serde_json::to_vec(&stats).unwrap()).unwrap();
        assert_eq!(roundtrip, stats);

        let mut old_json = serde_json::to_value(PerformanceReport::default()).unwrap();
        old_json
            .as_object_mut()
            .unwrap()
            .remove("native_fusion_cache");
        let old_report: PerformanceReport = serde_json::from_value(old_json).unwrap();
        assert_eq!(
            old_report.native_fusion_cache,
            NativeFusionCacheReport::default()
        );
    }
}
