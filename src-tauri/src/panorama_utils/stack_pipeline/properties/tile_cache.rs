//! Virtual_Tile_Store disk cache properties (19-22).
//!
//! Lossless round trip, the three cache key ingredients, eviction within the byte
//! budget in access order, and every corrupt entry read as a miss.

use super::super::degradation;
use super::super::report::VIRTUAL_TILE_CACHE_LIMIT_BYTES;
use super::super::test_support::arb_virtual_tile_in;
use super::super::test_support::full_coverage_shape;
use super::super::test_support::virtual_tile_with_shape;
use super::super::virtual_tile;
use super::super::virtual_tile::CacheKeyInputs;
use super::super::virtual_tile::CacheLookup;
use super::super::virtual_tile::StoreOutcome;
use super::super::virtual_tile::VirtualTileStore;
use super::super::virtual_tile::test_access;
use crate::image_stack::STACK_PIPELINE_VERSION;
use crate::panorama_utils::stitching;
use half::f16;
use proptest::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Duration;
use tempfile::TempDir;

use super::{CACHE_PROPERTY_MAX_SIDE, CACHE_PROPERTY_MIN_SIDE, frozen_reason_identifiers};

// ---------------------------------------------------------------------------
// Virtual_Tile_Store properties (需求 4.3–4.5, 4.9–4.11, 14.4)
// ---------------------------------------------------------------------------

/// Cache key ingredients for a tile whose sources are not on disk.
///
/// Properties 19 and 21 are about the payload and the index, not about the
/// digest of a real file; Property 20 builds its ingredients from real files
/// instead.
pub(super) fn cache_inputs_for(tile: &stitching::VirtualTile, version: &str) -> CacheKeyInputs {
    let sources = tile
        .provenance
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let mut digest = record.sha256.unwrap_or([0u8; 32]);
            digest[0] = digest[0].wrapping_add(index as u8);
            (record.absolute_path.to_string_lossy().to_string(), digest)
        })
        .collect::<Vec<_>>();
    CacheKeyInputs::new(version, &sources)
}

/// Element-for-element comparison of a stored and a loaded tile (需求 4.3).
pub(super) fn assert_cache_round_trip(
    stored: &stitching::VirtualTile,
    loaded: &stitching::VirtualTile,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(loaded.station_index, stored.station_index);
    prop_assert_eq!((loaded.width, loaded.height), (stored.width, stored.height));
    prop_assert_eq!(loaded.pixels.dimensions(), stored.pixels.dimensions());
    prop_assert_eq!(loaded.color_encoding, stored.color_encoding);
    prop_assert_eq!(&loaded.provenance, &stored.provenance);
    prop_assert_eq!(&loaded.ownership, &stored.ownership);
    prop_assert_eq!(
        loaded.coverage.covered_bounds(),
        stored.coverage.covered_bounds()
    );
    for (index, (left, right)) in loaded
        .tile_to_world
        .iter()
        .zip(stored.tile_to_world.iter())
        .enumerate()
    {
        prop_assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "tile_to_world coefficient {} changed ({} vs {})",
            index,
            left,
            right
        );
    }
    // Pixels: raw bits, so a single rounded mantissa or a lost sign fails.
    for (index, (left, right)) in loaded
        .pixels
        .as_raw()
        .iter()
        .zip(stored.pixels.as_raw().iter())
        .enumerate()
    {
        prop_assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "pixel channel {} changed ({} vs {})",
            index,
            left,
            right
        );
    }
    for y in 0..stored.height {
        for x in 0..stored.width {
            prop_assert_eq!(
                loaded.coverage.is_covered(x, y),
                stored.coverage.is_covered(x, y),
                "coverage at ({}, {})",
                x,
                y
            );
            prop_assert_eq!(
                loaded.ownership.owner_at(x, y),
                stored.ownership.owner_at(x, y),
                "owner at ({}, {})",
                x,
                y
            );
            // Sharpness_Confidence is the one plane the design stores at f16
            // width, so the contract is "exactly the value that was written",
            // which for an f16-representable input is bit equality.
            prop_assert_eq!(
                loaded.sharpness_confidence.value_at(x, y),
                f16::from_f32(stored.sharpness_confidence.value_at(x, y)).to_f32(),
                "confidence at ({}, {})",
                x,
                y
            );
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 19: 对于任意
    // Virtual_Tile，写入缓存后读回的合成像素每通道值、Ownership_Map 每个 owner 标识与
    // Coverage_Mask 每个覆盖标记都与写入时逐元素完全相同。
    //
    // 既有单元测试只往返了一个 4x3 的固定瓦片；这里推广到 16x16..48x48（含非方形与
    // 不是 8 的倍数的宽度，即一位一像素编码的尾字节）、六个覆盖族、两种色彩编码、
    // 两种 Sharpness_Confidence 表示与有无溯源摘要。
    //
    // **Validates: Requirements 4.3**
    #[test]
    fn virtual_tile_cache_round_trip_is_lossless(
        tile in arb_virtual_tile_in(CACHE_PROPERTY_MIN_SIDE, CACHE_PROPERTY_MAX_SIDE),
        drop_digests in any::<bool>(),
    ) {
        let mut tile = tile;
        if drop_digests {
            // The fusion call site does not always have a digest yet, so the
            // absent case has to round trip as "absent" rather than as zeroes.
            for record in &mut tile.provenance {
                record.sha256 = None;
            }
        }
        let directory = TempDir::new().expect("temp dir");
        let store = VirtualTileStore::new(directory.path());
        let inputs = cache_inputs_for(&tile, "stack-prop.19");

        prop_assert!(
            matches!(store.load(&inputs), CacheLookup::Miss),
            "an empty cache must miss"
        );
        let bytes = match store.store(&inputs, &tile) {
            StoreOutcome::Written { bytes, evicted, total_bytes } => {
                prop_assert_eq!(evicted, 0, "a single small entry cannot exceed 64 GiB");
                prop_assert_eq!(total_bytes, bytes);
                bytes
            }
            other => {
                prop_assert!(false, "a temp dir must be writable: {other:?}");
                unreachable!()
            }
        };
        prop_assert!(bytes > 0, "the entry must have a size");

        let loaded = match store.load(&inputs) {
            CacheLookup::Hit(loaded) => *loaded,
            other => {
                prop_assert!(false, "the entry just written must verify: {other:?}");
                unreachable!()
            }
        };
        assert_cache_round_trip(&tile, &loaded)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 20: 对于任意
    // Source_RAW 路径集合、SHA-256 集合与流程版本标识三元组，用同一三元组读取此前以该三元组
    // 写入的缓存条目必然命中且解码的 Source_RAW 数量为 0；三要素中任一项被扰动则必然未命中。
    //
    // "解码的 Source_RAW 数量为 0" 在这里是可证的而非声明的：命中之前源文件已被删除，
    // 任何试图读取源文件的代码路径都会失败。既有单元测试用的是三个手写的扰动，这里扰动的
    // 路径、摘要与文件内容都是随机的，并额外断言集合语义（重排三元组仍然命中）。
    //
    // **Validates: Requirements 4.5, 4.9**
    #[test]
    fn a_cache_hit_requires_all_three_key_ingredients(
        tile in arb_virtual_tile_in(CACHE_PROPERTY_MIN_SIDE, 24),
        contents in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..512), 1..=3),
        probe in any::<usize>(),
    ) {
        let sources = TempDir::new().expect("temp dir");
        let mut pairs = Vec::new();
        for (index, content) in contents.iter().enumerate() {
            let path = sources.path().join(format!("DSC_{:04}.NEF", 3680 + index));
            fs::write(&path, content).expect("writing a source fixture");
            let digest = virtual_tile::source_file_sha256(&path)
                .expect("the read only digest succeeds");
            pairs.push((path.to_string_lossy().to_string(), digest));
        }
        let inputs = CacheKeyInputs::new(STACK_PIPELINE_VERSION, &pairs);

        let cache = TempDir::new().expect("temp dir");
        let store = VirtualTileStore::new(cache.path());
        prop_assert!(
            matches!(store.store(&inputs, &tile), StoreOutcome::Written { .. }),
            "a temp dir must be writable"
        );

        // Every Source_RAW disappears before the hit, so a hit that decoded
        // anything could not have succeeded.
        for (path, _) in &pairs {
            fs::remove_file(path).expect("removing a source fixture");
            prop_assert!(!Path::new(path).exists());
        }
        match store.load(&inputs) {
            CacheLookup::Hit(loaded) => {
                prop_assert_eq!((loaded.width, loaded.height), (tile.width, tile.height));
                prop_assert_eq!(&loaded.ownership, &tile.ownership);
            }
            other => {
                prop_assert!(false, "the same triple must hit: {other:?}");
            }
        }

        // Set semantics: the ingredients are sorted, so a permuted triple is
        // the same triple (需求 4.5).
        let mut permuted = pairs.clone();
        permuted.reverse();
        let permuted = CacheKeyInputs::new(STACK_PIPELINE_VERSION, &permuted);
        prop_assert_eq!(permuted.cache_key(), inputs.cache_key());
        prop_assert!(matches!(store.load(&permuted), CacheLookup::Hit(_)));

        // 1. A different pipeline version.
        let other_version = CacheKeyInputs::new(
            &format!("{STACK_PIPELINE_VERSION}.next"),
            &pairs,
        );
        // 2. A different path set.
        let mut renamed = pairs.clone();
        let slot = probe % renamed.len();
        renamed[slot].0.push_str(".bak");
        let other_paths = CacheKeyInputs::new(STACK_PIPELINE_VERSION, &renamed);
        // 3. A different digest set — one flipped bit of one source file.
        let mut edited = pairs.clone();
        edited[slot].1[probe % 32] ^= 0x01;
        let other_digests = CacheKeyInputs::new(STACK_PIPELINE_VERSION, &edited);

        for (what, probe_inputs) in [
            ("the pipeline version", other_version),
            ("the source path set", other_paths),
            ("the source digest set", other_digests),
        ] {
            prop_assert_ne!(
                probe_inputs.cache_key(),
                inputs.cache_key(),
                "perturbing {} must change the cache key",
                what
            );
            prop_assert!(
                matches!(store.load(&probe_inputs), CacheLookup::Miss),
                "perturbing {what} must miss"
            );
        }
        // None of the misses touched the entry that does verify.
        prop_assert!(matches!(store.load(&inputs), CacheLookup::Hit(_)));
    }
}

/// Access times of `index.json` need to be distinguishable, and the index
/// records them in milliseconds, so the accesses of Property 21 are spaced.
pub(super) const CACHE_ACCESS_SPACING: Duration = Duration::from_millis(2);

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 21: 对于任意缓存条目大小
    // 序列与任意访问序列，淘汰后缓存总占用不超过 64 GiB，且每个被删除条目的最后访问时间都
    // 不晚于所有被保留条目的最后访问时间。
    //
    // 既有单元测试用的是"预算刚好一条"的两条目场景；这里的条目大小序列、访问序列与预算
    // 比例都是随机的。64 GiB 本身不写盘：预算被注入为实测字节数的一个比例，同时断言生产
    // 默认预算确实是 64 GiB。
    //
    // **Validates: Requirements 4.4**
    #[test]
    fn cache_eviction_keeps_the_budget_and_the_access_order(
        sides in prop::collection::vec(4u32..=12, 2..=4),
        access in prop::collection::vec(any::<usize>(), 0..6),
        budget_quarters in 1u64..=4,
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let directory = TempDir::new().expect("temp dir");
        prop_assert_eq!(
            VirtualTileStore::new(directory.path()).limit_bytes(),
            VIRTUAL_TILE_CACHE_LIMIT_BYTES,
            "the production budget is 64 GiB"
        );

        let store = VirtualTileStore::new(directory.path());
        let mut entries = Vec::new();
        for (index, &side) in sides.iter().enumerate() {
            let tile = virtual_tile_with_shape(&full_coverage_shape(side, side + 1), 1);
            let inputs = CacheKeyInputs::new(
                STACK_PIPELINE_VERSION,
                &[(format!("/scan/DSC_{index:04}.NEF"), [index as u8; 32])],
            );
            let bytes = match store.store(&inputs, &tile) {
                StoreOutcome::Written { bytes, evicted, .. } => {
                    prop_assert_eq!(evicted, 0, "64 GiB holds every entry of this property");
                    bytes
                }
                other => {
                    prop_assert!(false, "a temp dir must be writable: {other:?}");
                    unreachable!()
                }
            };
            entries.push((inputs, bytes));
            std::thread::sleep(CACHE_ACCESS_SPACING);
        }

        // The access sequence: every hit refreshes `last_access_epoch_ms`.
        for &slot in &access {
            let (inputs, _) = &entries[slot % entries.len()];
            prop_assert!(matches!(store.load(inputs), CacheLookup::Hit(_)));
            std::thread::sleep(CACHE_ACCESS_SPACING);
        }

        let before = store
            .index_snapshot_for_test()
            .into_iter()
            .map(|(key, last_access, _)| (key, last_access))
            .collect::<HashMap<_, _>>();
        prop_assert_eq!(
            before.len(),
            entries.len(),
            "every stored entry is in the index"
        );

        // A budget that forces at least one eviction: the new entry alone
        // already pushes the cache past the size of everything written so far.
        let total_before = entries.iter().map(|(_, bytes)| *bytes).sum::<u64>();
        let limit = (total_before * budget_quarters / 4).max(1);
        let store = VirtualTileStore::new(directory.path()).with_limit_bytes(limit);
        let extra = CacheKeyInputs::new(
            STACK_PIPELINE_VERSION,
            &[("/scan/DSC_9999.NEF".to_string(), [0xAB; 32])],
        );
        let evicted = match store.store(&extra, &virtual_tile_with_shape(&full_coverage_shape(8, 9), 1)) {
            StoreOutcome::Written { evicted, total_bytes, .. } => {
                prop_assert!(
                    total_bytes <= limit,
                    "the cache holds {total_bytes} byte(s) under a {limit} byte budget"
                );
                evicted
            }
            other => {
                prop_assert!(false, "a temp dir must be writable: {other:?}");
                unreachable!()
            }
        };
        prop_assert!(
            evicted >= 1,
            "a budget of {limit} byte(s) over {total_before} byte(s) must evict"
        );
        prop_assert!(store.total_bytes() <= limit, "the budget is a ceiling");

        // Whole entries only, and the index agrees with the disk.
        let after = store
            .index_snapshot_for_test()
            .into_iter()
            .map(|(key, last_access, _)| (key, last_access))
            .collect::<HashMap<_, _>>();
        let mut deleted_access = Vec::new();
        let mut retained_access = Vec::new();
        for (inputs, _) in &entries {
            let key = inputs.cache_key();
            let entry = store.entry_dir_for_test(&key);
            prop_assert_eq!(
                entry.exists(),
                after.contains_key(&key),
                "the index and the disk disagree about {}",
                key
            );
            let last_access = before[&key];
            if entry.exists() {
                retained_access.push(last_access);
                prop_assert!(
                    entry.join(test_access::META_FILE_NAME).exists(),
                    "a retained entry keeps its meta.json"
                );
            } else {
                deleted_access.push(last_access);
            }
        }
        prop_assert!(
            !deleted_access.is_empty(),
            "the eviction pass reported {evicted} deletion(s) but nothing is gone"
        );
        if let (Some(&newest_deleted), Some(&oldest_retained)) = (
            deleted_access.iter().max(),
            retained_access.iter().min(),
        ) {
            prop_assert!(
                newest_deleted <= oldest_retained,
                "a deleted entry was accessed at {newest_deleted}, later than the retained \
                 entry accessed at {oldest_retained}"
            );
        }
    }
}

/// One way a cache entry can be damaged on disk (需求 4.10).
///
/// The design names three conditions — a missing field, recorded dimensions
/// that disagree with the payload, a SHA-256 set that cannot be re-verified —
/// and each of them has more than one on-disk shape.  This enum is here rather
/// than in `test_support` because it mutates the *cache layout*, which only
/// `virtual_tile::test_access` exposes; it is not an input generator.
#[derive(Debug, Clone, Copy)]
pub(super) enum CacheCorruption {
    /// One required `meta.json` key is gone.
    RemoveMetaKey(usize),
    /// One required key of a nested `meta.json` object is gone: a
    /// `cache_key_inputs` ingredient, a `coverage_bounds` corner, a
    /// `provenance` record field or a `payloads` entry field.
    RemoveNestedMetaKey(usize, usize),
    /// One payload file is gone.
    RemovePayload(usize),
    /// `meta.json` records a taller tile than the payloads hold.
    GrowRecordedHeight,
    /// `meta.json` records a wider tile than the payloads hold.
    GrowRecordedWidth,
    /// One bit of one payload plane is flipped and the plane re-compressed, so
    /// only the recorded digest can catch it.
    FlipPayloadByte(usize, usize),
    /// One payload file is not a valid frame any more.
    TruncatePayload(usize),
    /// The recorded source digest set no longer matches the run.
    RewriteDigestSet,
    /// The recorded source path set no longer matches the run.
    RewritePathSet,
}

pub(super) fn arb_cache_corruption() -> impl Strategy<Value = CacheCorruption> {
    prop_oneof![
        any::<usize>().prop_map(CacheCorruption::RemoveMetaKey),
        (any::<usize>(), any::<usize>())
            .prop_map(|(section, key)| CacheCorruption::RemoveNestedMetaKey(section, key)),
        any::<usize>().prop_map(CacheCorruption::RemovePayload),
        Just(CacheCorruption::GrowRecordedHeight),
        Just(CacheCorruption::GrowRecordedWidth),
        (any::<usize>(), any::<usize>())
            .prop_map(|(payload, byte)| CacheCorruption::FlipPayloadByte(payload, byte)),
        any::<usize>().prop_map(CacheCorruption::TruncatePayload),
        Just(CacheCorruption::RewriteDigestSet),
        Just(CacheCorruption::RewritePathSet),
    ]
}

pub(super) fn read_meta(entry: &Path) -> Value {
    serde_json::from_slice(&fs::read(entry.join(test_access::META_FILE_NAME)).expect("meta.json"))
        .expect("meta.json parses")
}

pub(super) fn write_meta(entry: &Path, meta: &Value) {
    fs::write(
        entry.join(test_access::META_FILE_NAME),
        serde_json::to_vec(meta).expect("meta.json re-serializes"),
    )
    .expect("writing meta.json");
}

impl CacheCorruption {
    /// The reason identifier 需求 4.10 demands for this damage.
    pub(super) fn expected_reason(self) -> &'static str {
        match self {
            Self::RemoveMetaKey(_)
            | Self::RemoveNestedMetaKey(_, _)
            | Self::RemovePayload(_)
            | Self::TruncatePayload(_) => degradation::CACHE_ENTRY_FIELD_MISSING,
            Self::GrowRecordedHeight | Self::GrowRecordedWidth => {
                degradation::CACHE_ENTRY_DIMENSION_MISMATCH
            }
            Self::FlipPayloadByte(_, _) | Self::RewriteDigestSet | Self::RewritePathSet => {
                degradation::CACHE_ENTRY_SHA_MISMATCH
            }
        }
    }

    pub(super) fn apply(self, entry: &Path) {
        match self {
            Self::RemoveMetaKey(which) => {
                let key =
                    test_access::META_REQUIRED_KEYS[which % test_access::META_REQUIRED_KEYS.len()];
                let mut meta = read_meta(entry);
                meta.as_object_mut()
                    .expect("meta.json is an object")
                    .remove(key);
                write_meta(entry, &meta);
            }
            Self::RemoveNestedMetaKey(section, which) => {
                const SECTIONS: &[(&str, &[&str])] = &[
                    ("cache_key_inputs", test_access::CACHE_KEY_INPUTS_KEYS),
                    ("coverage_bounds", test_access::COVERAGE_BOUNDS_KEYS),
                    ("provenance", test_access::PROVENANCE_KEYS),
                    ("payloads", test_access::PAYLOAD_KEYS),
                ];
                let (root_key, keys) = SECTIONS[section % SECTIONS.len()];
                let key = keys[which % keys.len()];
                let mut meta = read_meta(entry);
                let target = match root_key {
                    "provenance" => meta[root_key]
                        .as_array_mut()
                        .and_then(|records| records.first_mut()),
                    "payloads" => meta[root_key]
                        .as_object_mut()
                        .and_then(|payloads| payloads.values_mut().next()),
                    _ => Some(&mut meta[root_key]),
                };
                let removed = target
                    .and_then(Value::as_object_mut)
                    .is_some_and(|object| object.remove(key).is_some());
                if !removed {
                    // The section is nullable or empty for this tile — a fully
                    // uncovered tile records `coverage_bounds: null` — so there
                    // is no nested key to delete.  Deleting the section itself
                    // is the same class of damage.
                    meta.as_object_mut()
                        .expect("meta.json is an object")
                        .remove(root_key);
                }
                write_meta(entry, &meta);
            }
            Self::RemovePayload(which) => {
                let name =
                    test_access::REQUIRED_PAYLOADS[which % test_access::REQUIRED_PAYLOADS.len()];
                fs::remove_file(entry.join(name)).expect("removing a payload");
            }
            Self::GrowRecordedHeight => {
                let mut meta = read_meta(entry);
                let height = meta["height"].as_u64().expect("height is a number");
                meta["height"] = serde_json::json!(height + 1);
                write_meta(entry, &meta);
            }
            Self::GrowRecordedWidth => {
                let mut meta = read_meta(entry);
                let width = meta["width"].as_u64().expect("width is a number");
                meta["width"] = serde_json::json!(width + 1);
                write_meta(entry, &meta);
            }
            Self::FlipPayloadByte(which, byte) => {
                let name =
                    test_access::REQUIRED_PAYLOADS[which % test_access::REQUIRED_PAYLOADS.len()];
                let path = entry.join(name);
                let mut plane = test_access::read_payload(&path).expect("the payload decodes");
                let slot = byte % plane.len();
                plane[slot] ^= 0x01;
                fs::write(&path, test_access::compress_payload(&plane))
                    .expect("writing the payload");
            }
            Self::TruncatePayload(which) => {
                let name =
                    test_access::REQUIRED_PAYLOADS[which % test_access::REQUIRED_PAYLOADS.len()];
                fs::write(entry.join(name), b"not a frame").expect("writing the payload");
            }
            Self::RewriteDigestSet => {
                let mut meta = read_meta(entry);
                meta["cache_key_inputs"]["sorted_source_sha256"][0] =
                    serde_json::json!("0".repeat(64));
                write_meta(entry, &meta);
            }
            Self::RewritePathSet => {
                let mut meta = read_meta(entry);
                meta["cache_key_inputs"]["sorted_paths"][0] = serde_json::json!("/moved.NEF");
                write_meta(entry, &meta);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 22: 对于任意缓存条目与
    // 任意一种损坏（字段缺失、记录尺寸与实际像素尺寸不符、SHA-256 集合无法复核），读取结果
    // 为未命中、该条目被删除、Virtual_Tile 被重新合成，且缓存失效原因标识符被记录。
    //
    // 既有单元测试各自固定了一种损坏与一个受损字段；这里 `meta.json` 的 13 个根级必需键、
    // 嵌套对象（`cache_key_inputs` / `coverage_bounds` / `provenance` / `payloads`）内的
    // 每个键、4 个载荷、载荷内任意一位、以及路径集合与摘要集合都在输入空间内。
    //
    // **Validates: Requirements 4.10**
    #[test]
    fn a_corrupt_cache_entry_is_always_a_miss(
        tile in arb_virtual_tile_in(CACHE_PROPERTY_MIN_SIDE, 24),
        corruption in arb_cache_corruption(),
    ) {
        let directory = TempDir::new().expect("temp dir");
        let store = VirtualTileStore::new(directory.path());
        let inputs = cache_inputs_for(&tile, "stack-prop.22");
        prop_assert!(
            matches!(store.store(&inputs, &tile), StoreOutcome::Written { .. }),
            "a temp dir must be writable"
        );
        let entry = store.entry_dir_for_test(&inputs.cache_key());
        prop_assert!(matches!(store.load(&inputs), CacheLookup::Hit(_)));

        corruption.apply(&entry);

        // 1. 未命中，且失效原因标识符来自冻结标识符集合（需求 12.7）。
        match store.load(&inputs) {
            CacheLookup::Invalid { reason, detail } => {
                prop_assert_eq!(
                    reason,
                    corruption.expected_reason(),
                    "{:?} was reported as {} ({})",
                    corruption,
                    reason,
                    detail
                );
                prop_assert!(
                    frozen_reason_identifiers().contains(reason),
                    "the cache invalidation reason must be a frozen identifier"
                );
                prop_assert!(!detail.is_empty(), "the reason must carry a detail");
            }
            other => {
                prop_assert!(false, "{corruption:?} must invalidate the entry, got {other:?}");
            }
        }

        // 2. 该条目被删除，后续读取是普通未命中（而不是反复失败的坏条目）。
        prop_assert!(!entry.exists(), "a corrupt entry must be deleted whole");
        prop_assert!(matches!(store.load(&inputs), CacheLookup::Miss));

        // 3. Virtual_Tile 被重新合成: the caller re-fuses and stores again, and
        //    the entry that comes back is the original one element for element.
        prop_assert!(
            matches!(store.store(&inputs, &tile), StoreOutcome::Written { .. }),
            "a temp dir must be writable"
        );
        match store.load(&inputs) {
            CacheLookup::Hit(loaded) => assert_cache_round_trip(&tile, &loaded)?,
            other => {
                prop_assert!(false, "the re-fused entry must verify: {other:?}");
            }
        }
    }
}
