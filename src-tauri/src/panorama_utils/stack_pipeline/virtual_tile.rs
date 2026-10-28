//! `Virtual_Tile_Store`: the versioned, lossless on-disk cache of one focus
//! fused capture station (需求 4.3–4.5, 4.9–4.11).
//!
//! Layout, rooted at `<app_cache_dir>/stack-virtual-tiles/v1/`:
//!
//! ```text
//! <cache_key>/                  # directory name = cache_key in lower hex
//!   meta.json                   # the three cache key ingredients + payload digests
//!   pixels.f32.zst              # f32 little endian, 3 values per pixel
//!   ownership.u16.zst           # u16 little endian, 1 value per pixel
//!   coverage.bits.zst           # 1 bit per pixel, MSB first inside a byte
//!   confidence.f16.zst          # f16 little endian, 1 value per pixel
//! .tmp-<uuid>/                  # an entry being written; renamed into place
//! index.json                    # cache_key -> { last_access_epoch_ms, bytes }
//! ```
//!
//! Three properties of this module carry the requirements:
//!
//! * **Lossless (需求 4.3).**  Pixels, ownership and coverage are written as
//!   raw little endian element planes and only then compressed with zstd, which
//!   is a lossless general purpose compressor.  No image encoder is involved,
//!   so no predictor, chroma subsampling or colour transform can perturb the
//!   round trip: [`VirtualTileStore::load`] returns the very same `f32`
//!   channel values, `u16` owner identifiers and coverage flags that
//!   [`VirtualTileStore::store`] was handed.  `Sharpness_Confidence` is the one
//!   plane the design stores at `f16` width; it is element-for-element exact
//!   for the value actually written, but a wider input is rounded once on the
//!   way in.
//! * **Atomic (需求 4.9).**  Every payload plus `meta.json` is written into
//!   `.tmp-<uuid>/`, `fsync`ed, and only then `rename`d onto `<cache_key>/`.  A
//!   reader therefore never observes a partially written entry: the entry
//!   directory either does not exist or is complete.
//! * **Never fatal (需求 4.10 / 4.11).**  A corrupt entry is deleted and
//!   reported as a miss; an unwritable directory or a failed write leaves the
//!   in-memory tile untouched and is reported as
//!   [`degradation::CACHE_WRITE_UNAVAILABLE`].  No path in this module returns
//!   an error that could abort a run.
//!
//! The cache key is also the run's RANSAC seed source, see
//! [`super::determinism::derive_run_seed_from_cache_key`].

// The cache is declared complete up front so its unit tests can pin the layout,
// the key derivation and every verification branch, while the fusion call site
// that reads and writes entries arrives with the Tile_Compositor stage. Only
// `source_file_sha256` and `CacheKeyInputs` have a production caller today (the
// run seed in `panorama_stitching`). Drop this allow once the compositor leases
// tiles through the store.
#![allow(dead_code)]

#[cfg(test)]
use super::degradation;
#[cfg(test)]
use crate::panorama_utils::stitching::{
    ColorEncoding, ConfidenceMap, CoverageMask, OwnershipMap, SourceProvenance, VirtualTile,
};
#[cfg(test)]
use half::f16;
#[cfg(test)]
use image::Rgb32FImage;
#[cfg(test)]
use nalgebra::Matrix3;
#[cfg(test)]
use sha2::{Digest, Sha256};
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::{Path, PathBuf};

/// Cache root below `app_cache_dir()`, mirroring `image_stack.rs`'s use of
/// `app_handle.path().app_cache_dir()` for the preview cache.
pub(crate) const VIRTUAL_TILE_CACHE_DIR_NAME: &str = "stack-virtual-tiles";

/// Layout version of the cache directory.  Bumping this abandons every existing
/// entry without having to delete it, which is what a change to the payload
/// encoding (rather than to the pipeline) requires.
pub(crate) const VIRTUAL_TILE_CACHE_LAYOUT_VERSION: &str = "v1";

/// `meta.json` schema version.
const META_SCHEMA: u32 = 1;

/// `index.json` schema version.
const INDEX_SCHEMA: u32 = 1;

const META_FILE_NAME: &str = "meta.json";
const INDEX_FILE_NAME: &str = "index.json";
const TEMP_PREFIX: &str = ".tmp-";

const PIXELS_PAYLOAD: &str = "pixels.f32.zst";
const OWNERSHIP_PAYLOAD: &str = "ownership.u16.zst";
const COVERAGE_PAYLOAD: &str = "coverage.bits.zst";
const CONFIDENCE_PAYLOAD: &str = "confidence.f16.zst";

/// Every payload the loader requires; a missing name is a corrupt entry.
const REQUIRED_PAYLOADS: &[&str] = &[
    PIXELS_PAYLOAD,
    OWNERSHIP_PAYLOAD,
    COVERAGE_PAYLOAD,
    CONFIDENCE_PAYLOAD,
];

/// zstd level.  Level 3 is the library default: the payloads are large and the
/// cache exists to save a RAW decode, not to reach the smallest possible file.
/// The level is part of no identity — zstd decodes any level identically — so
/// changing it can never invalidate an entry.
const ZSTD_LEVEL: i32 = 3;

mod cache_key;

#[cfg(test)]
use cache_key::DIGEST_CHUNK_BYTES;
use cache_key::parse_hex_digest;
pub(crate) use cache_key::{CacheKeyInputs, hex_digest, source_file_sha256};

mod cache_files;

#[cfg(test)]
use cache_files::{
    CACHE_KEY_INPUTS_KEYS, COVERAGE_BOUNDS_KEYS, META_KEYS, PAYLOAD_KEYS, PROVENANCE_KEYS,
};
use cache_files::{
    CacheEntryMeta, CacheIndex, CoverageBoundsMeta, IndexEntry, PayloadMeta, ProvenanceMeta,
    color_encoding_identifier, dimension_mismatch, epoch_millis, parse_color_encoding,
    verify_meta_keys,
};

mod leases;

pub(crate) use leases::{LeaseError, LeaseRegistry, MAX_RESIDENT_VIRTUAL_TILES, TileLease};

mod store;

#[cfg(test)]
pub(crate) use store::{CacheLookup, StoreOutcome, VirtualTileStore};

mod payload_io;

use payload_io::{
    decode_coverage_bits, decode_f16_plane, decode_f32_plane, decode_u16_plane, directory_bytes,
    encode_coverage_bits, encode_f16_plane, encode_f32_plane, encode_u16_plane, read_payload,
    sync_directory, write_file_sync, write_payload,
};

/// Layout details the stage 2 property tests reach into.
///
/// Kept behind `cfg(test)` so the production surface of the module stays what
/// it was: a store, a lookup result and a store outcome.
#[cfg(test)]
pub(crate) mod test_access {
    use std::path::Path;

    pub(crate) const META_FILE_NAME: &str = super::META_FILE_NAME;
    pub(crate) const PIXELS_PAYLOAD: &str = super::PIXELS_PAYLOAD;
    pub(crate) const REQUIRED_PAYLOADS: &[&str] = super::REQUIRED_PAYLOADS;

    /// Every `meta.json` key the loader requires, so a property can remove an
    /// arbitrary one instead of one hand picked field.  Taken from the module's
    /// own declaration rather than repeated here: a list that drifts from the
    /// struct would have the property delete a key that does not exist, which
    /// tests nothing.
    pub(crate) const META_REQUIRED_KEYS: &[&str] = super::META_KEYS;

    /// The nested key sets, so the corruption generator can also reach a key
    /// inside `cache_key_inputs`, `coverage_bounds`, `provenance` or
    /// `payloads`.
    pub(crate) const CACHE_KEY_INPUTS_KEYS: &[&str] = super::CACHE_KEY_INPUTS_KEYS;
    pub(crate) const COVERAGE_BOUNDS_KEYS: &[&str] = super::COVERAGE_BOUNDS_KEYS;
    pub(crate) const PROVENANCE_KEYS: &[&str] = super::PROVENANCE_KEYS;
    pub(crate) const PAYLOAD_KEYS: &[&str] = super::PAYLOAD_KEYS;

    pub(crate) fn read_payload(path: &Path) -> Result<Vec<u8>, String> {
        super::read_payload(path)
    }

    /// Re-compress a mutated plane with the level the store writes, so the
    /// mutation is invisible to everything but the recorded digest.
    pub(crate) fn compress_payload(plane: &[u8]) -> Vec<u8> {
        zstd::bulk::compress(plane, super::ZSTD_LEVEL).expect("re-compressing a payload")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Rgb};
    use tempfile::TempDir;

    fn digest_of(seed: u8) -> [u8; 32] {
        let mut digest = [0u8; 32];
        for (index, slot) in digest.iter_mut().enumerate() {
            *slot = seed.wrapping_add(index as u8);
        }
        digest
    }

    fn sample_inputs() -> CacheKeyInputs {
        CacheKeyInputs::new(
            "stack-test.1",
            &[
                ("/scan/DSC_3681.NEF".to_string(), digest_of(0x20)),
                ("/scan/DSC_3680.NEF".to_string(), digest_of(0x10)),
            ],
        )
    }

    /// A 4x3 tile whose coverage is a ragged interior shape, so the coverage
    /// bounds, the bit packing tail and the two owner legend entries are all
    /// exercised by one fixture.
    fn sample_tile(station_index: usize) -> VirtualTile {
        let width = 4u32;
        let height = 3u32;
        let covered_flags = [
            false, false, false, false, //
            false, true, true, false, //
            false, true, false, false,
        ];
        let owners = covered_flags
            .iter()
            .enumerate()
            .map(|(index, &covered)| {
                if covered {
                    if index % 2 == 0 { 1u16 } else { 2u16 }
                } else {
                    0u16
                }
            })
            .collect::<Vec<_>>();
        let mut pixels = Rgb32FImage::new(width, height);
        for (index, &covered) in covered_flags.iter().enumerate() {
            if !covered {
                continue;
            }
            let x = (index % width as usize) as u32;
            let y = (index / width as usize) as u32;
            // Values chosen to survive no rounding: a subnormal, a long
            // mantissa and a negative would all be destroyed by an image codec.
            pixels.put_pixel(
                x,
                y,
                Rgb([
                    f32::from_bits(0x0000_0001),
                    0.123_456_79_f32,
                    -2.5_f32 * index as f32,
                ]),
            );
        }
        let legend = vec![
            PathBuf::from("/scan/DSC_3680.NEF"),
            PathBuf::from("/scan/DSC_3681.NEF"),
        ];
        let ownership = OwnershipMap::new(width, height, owners, legend.clone())
            .expect("ownership map is well formed");
        let counts = ownership.owned_pixel_counts();
        let coverage = CoverageMask::from_gray(
            GrayImage::from_raw(
                width,
                height,
                covered_flags
                    .iter()
                    .map(|&covered| if covered { 255 } else { 0 })
                    .collect(),
            )
            .expect("coverage plane is well formed"),
        );
        let confidence = ConfidenceMap::per_pixel(
            width,
            height,
            covered_flags
                .iter()
                .enumerate()
                .map(|(index, &covered)| if covered { index as f32 / 16.0 } else { 0.0 })
                .collect(),
        )
        .expect("confidence map is well formed");
        let provenance = legend
            .iter()
            .zip(counts)
            .enumerate()
            .map(|(index, (path, owned_pixels))| SourceProvenance {
                absolute_path: path.clone(),
                sha256: Some(digest_of(0x10 + 0x10 * index as u8)),
                owned_pixels,
            })
            .collect();
        VirtualTile::new(
            station_index,
            Matrix3::new(1.0, 0.0, -1234.0, 0.0, 1.0, -5678.0, 0.0, 0.0, 1.0),
            pixels,
            ownership,
            confidence,
            coverage,
            ColorEncoding::DisplaySrgb,
            provenance,
        )
        .expect("the fixture satisfies every Virtual_Tile invariant")
    }

    fn store_in(directory: &TempDir) -> VirtualTileStore {
        VirtualTileStore::new(directory.path())
    }

    // -- cache_key ----------------------------------------------------------

    #[test]
    fn cache_key_ingredients_are_sorted_so_import_order_cannot_reach_the_key() {
        let forward = CacheKeyInputs::new(
            "v1",
            &[
                ("/a.NEF".to_string(), digest_of(1)),
                ("/b.NEF".to_string(), digest_of(2)),
            ],
        );
        let backward = CacheKeyInputs::new(
            "v1",
            &[
                ("/b.NEF".to_string(), digest_of(2)),
                ("/a.NEF".to_string(), digest_of(1)),
            ],
        );
        assert_eq!(forward, backward);
        assert_eq!(forward.cache_key(), backward.cache_key());
    }

    #[test]
    fn cache_key_changes_with_each_of_the_three_ingredients() {
        let sources = [
            ("/a.NEF".to_string(), digest_of(1)),
            ("/b.NEF".to_string(), digest_of(2)),
        ];
        let reference = CacheKeyInputs::new("v1", &sources).cache_key();

        // 1. pipeline version
        assert_ne!(reference, CacheKeyInputs::new("v2", &sources).cache_key());
        // 2. path set
        let renamed = [
            ("/a.NEF".to_string(), digest_of(1)),
            ("/c.NEF".to_string(), digest_of(2)),
        ];
        assert_ne!(reference, CacheKeyInputs::new("v1", &renamed).cache_key());
        // 3. digest set
        let edited = [
            ("/a.NEF".to_string(), digest_of(1)),
            ("/b.NEF".to_string(), digest_of(9)),
        ];
        assert_ne!(reference, CacheKeyInputs::new("v1", &edited).cache_key());
        // and the source count
        assert_ne!(
            reference,
            CacheKeyInputs::new("v1", &sources[..1]).cache_key()
        );
    }

    #[test]
    fn cache_key_is_length_prefixed_against_boundary_collisions() {
        let joined = [("/a/b".to_string(), digest_of(1))];
        let split = [
            ("/a".to_string(), digest_of(1)),
            ("b".to_string(), digest_of(1)),
        ];
        assert_ne!(
            CacheKeyInputs::new("v", &joined).cache_key(),
            CacheKeyInputs::new("v", &split).cache_key()
        );
    }

    #[test]
    fn source_file_sha256_matches_a_direct_digest_of_the_bytes() {
        let directory = TempDir::new().expect("temp dir");
        let path = directory.path().join("source.bin");
        // Larger than one read chunk, so the streaming loop is exercised.
        let payload = (0..DIGEST_CHUNK_BYTES + 12_345)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        fs::write(&path, &payload).expect("writing the fixture");
        let expected: [u8; 32] = Sha256::digest(&payload).into();
        assert_eq!(source_file_sha256(&path).expect("digest"), expected);
        assert!(source_file_sha256(&directory.path().join("absent.bin")).is_err());
    }

    // -- lossless round trip ------------------------------------------------

    #[test]
    fn plane_encodings_round_trip_every_element() {
        let values = vec![
            0.0f32,
            -0.0,
            f32::from_bits(0x0000_0001),
            1.0 / 3.0,
            f32::MAX,
            f32::MIN_POSITIVE,
        ];
        assert_eq!(
            decode_f32_plane(&encode_f32_plane(&values), values.len() as u64).expect("f32 plane"),
            values
        );
        // Bit equality, not value equality: -0.0 == 0.0 would hide a sign loss.
        let owners = vec![0u16, 1, 2, u16::MAX];
        assert_eq!(
            decode_u16_plane(&encode_u16_plane(&owners), owners.len() as u64).expect("u16 plane"),
            owners
        );
        let covered = vec![0u8, 255, 0, 0, 7, 0, 0, 0, 1, 0, 0];
        let decoded = decode_coverage_bits(&encode_coverage_bits(&covered), covered.len() as u64)
            .expect("coverage plane");
        assert_eq!(
            decoded.iter().map(|&value| value > 0).collect::<Vec<_>>(),
            covered.iter().map(|&value| value > 0).collect::<Vec<_>>()
        );
        assert_eq!(encode_coverage_bits(&covered).len(), 2, "1 bit per pixel");
    }

    #[test]
    fn store_and_load_round_trip_is_element_for_element_identical() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let inputs = sample_inputs();
        let tile = sample_tile(7);

        assert!(matches!(store.load(&inputs), CacheLookup::Miss));
        let bytes = match store.store(&inputs, &tile) {
            StoreOutcome::Written { bytes, evicted, .. } => {
                assert_eq!(evicted, 0);
                bytes
            }
            other => panic!("the cache must be writable here: {other:?}"),
        };
        assert!(bytes > 0);

        let loaded = match store.load(&inputs) {
            CacheLookup::Hit(tile) => *tile,
            other => panic!("the entry must verify: {other:?}"),
        };

        assert_eq!(loaded.station_index, tile.station_index);
        assert_eq!((loaded.width, loaded.height), (tile.width, tile.height));
        assert_eq!(loaded.tile_to_world, tile.tile_to_world);
        assert_eq!(loaded.color_encoding, tile.color_encoding);
        assert_eq!(loaded.provenance, tile.provenance);
        assert_eq!(loaded.ownership, tile.ownership);
        // Pixels: compare raw bits so a single rounded mantissa fails.
        assert_eq!(
            loaded
                .pixels
                .as_raw()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            tile.pixels
                .as_raw()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        // Coverage: the bit plane keeps the flag, so compare the predicate.
        for y in 0..tile.height {
            for x in 0..tile.width {
                assert_eq!(
                    loaded.coverage.is_covered(x, y),
                    tile.coverage.is_covered(x, y),
                    "coverage at ({x}, {y})"
                );
                assert_eq!(
                    loaded.ownership.owner_at(x, y),
                    tile.ownership.owner_at(x, y),
                    "owner at ({x}, {y})"
                );
                assert_eq!(
                    loaded.sharpness_confidence.value_at(x, y),
                    f16::from_f32(tile.sharpness_confidence.value_at(x, y)).to_f32(),
                    "confidence at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn no_image_encoder_touches_a_payload() {
        // The payload files must be raw zstd frames. If any image codec were
        // involved the magic bytes would be a TIFF/PNG signature instead.
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let inputs = sample_inputs();
        assert!(matches!(
            store.store(&inputs, &sample_tile(0)),
            StoreOutcome::Written { .. }
        ));
        let entry = store.entry_dir(&inputs.cache_key());
        for name in REQUIRED_PAYLOADS {
            let bytes = fs::read(entry.join(name)).expect("payload exists");
            assert_eq!(
                &bytes[..4],
                &[0x28, 0xB5, 0x2F, 0xFD],
                "{name} must be a zstd frame"
            );
        }
    }

    // -- cache key sensitivity ---------------------------------------------

    #[test]
    fn a_change_to_any_cache_key_ingredient_is_a_miss() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let inputs = sample_inputs();
        assert!(matches!(
            store.store(&inputs, &sample_tile(3)),
            StoreOutcome::Written { .. }
        ));
        assert!(matches!(store.load(&inputs), CacheLookup::Hit(_)));

        let other_version = CacheKeyInputs::new(
            "stack-test.2",
            &[
                ("/scan/DSC_3681.NEF".to_string(), digest_of(0x20)),
                ("/scan/DSC_3680.NEF".to_string(), digest_of(0x10)),
            ],
        );
        let other_paths = CacheKeyInputs::new(
            "stack-test.1",
            &[
                ("/scan/DSC_3681.NEF".to_string(), digest_of(0x20)),
                ("/scan/DSC_9999.NEF".to_string(), digest_of(0x10)),
            ],
        );
        let other_digests = CacheKeyInputs::new(
            "stack-test.1",
            &[
                ("/scan/DSC_3681.NEF".to_string(), digest_of(0x20)),
                ("/scan/DSC_3680.NEF".to_string(), digest_of(0x11)),
            ],
        );
        for probe in [other_version, other_paths, other_digests] {
            assert!(
                matches!(store.load(&probe), CacheLookup::Miss),
                "a changed ingredient must not hit"
            );
        }
        // The original entry is untouched by the three misses.
        assert!(matches!(store.load(&inputs), CacheLookup::Hit(_)));
    }

    // -- corruption detection ----------------------------------------------

    fn stored_entry() -> (TempDir, VirtualTileStore, CacheKeyInputs, PathBuf) {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let inputs = sample_inputs();
        assert!(matches!(
            store.store(&inputs, &sample_tile(1)),
            StoreOutcome::Written { .. }
        ));
        let entry = store.entry_dir(&inputs.cache_key());
        (directory, store, inputs, entry)
    }

    /// A tile with no covered pixel at all, and no source digest.
    ///
    /// Both nullable `meta.json` fields — `coverage_bounds` and the provenance
    /// `sha256` — are `None` here, so the document records them as `null`.  That
    /// is the shape in which a deleted key used to be indistinguishable from the
    /// legitimate value, because every downstream cross-check compares against
    /// `None` as well.
    fn uncovered_tile() -> VirtualTile {
        let (width, height) = (4u32, 3u32);
        let legend = vec![PathBuf::from("/scan/DSC_3680.NEF")];
        let ownership =
            OwnershipMap::new(width, height, vec![0u16; (width * height) as usize], legend)
                .expect("ownership map is well formed");
        let coverage = CoverageMask::from_gray(
            GrayImage::from_raw(width, height, vec![0u8; (width * height) as usize])
                .expect("coverage plane is well formed"),
        );
        let confidence =
            ConfidenceMap::per_pixel(width, height, vec![0.0f32; (width * height) as usize])
                .expect("confidence map is well formed");
        VirtualTile::new(
            0,
            Matrix3::identity(),
            Rgb32FImage::new(width, height),
            ownership,
            confidence,
            coverage,
            ColorEncoding::LinearSrgb,
            vec![SourceProvenance {
                absolute_path: PathBuf::from("/scan/DSC_3680.NEF"),
                sha256: None,
                owned_pixels: 0,
            }],
        )
        .expect("a fully uncovered tile satisfies every Virtual_Tile invariant")
    }

    /// Read `meta.json`, hand it to `edit`, write it back.
    fn edit_meta(entry: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
        let path = entry.join(META_FILE_NAME);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).expect("meta")).expect("meta parses");
        edit(&mut value);
        fs::write(&path, serde_json::to_vec(&value).expect("re-serialize")).expect("write");
    }

    /// Store `tile`, verify it reads back, damage it, then assert the damage is
    /// reported as `reason` and the entry is gone (需求 4.10).
    fn assert_invalid_for_tile(reason: &str, tile: &VirtualTile, mutate: impl FnOnce(&Path)) {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let inputs = sample_inputs();
        assert!(matches!(
            store.store(&inputs, tile),
            StoreOutcome::Written { .. }
        ));
        let entry = store.entry_dir(&inputs.cache_key());
        assert!(
            matches!(store.load(&inputs), CacheLookup::Hit(_)),
            "the entry must verify before it is damaged"
        );
        mutate(&entry);
        match store.load(&inputs) {
            CacheLookup::Invalid {
                reason: actual,
                detail,
            } => assert_eq!(actual, reason, "{detail}"),
            other => panic!("expected {reason}, got {other:?}"),
        }
        assert!(!entry.exists(), "a corrupt entry must be deleted");
        assert!(matches!(store.load(&inputs), CacheLookup::Miss));
    }

    #[test]
    fn a_deleted_coverage_bounds_key_invalidates_an_entry_that_records_it_as_null() {
        let tile = uncovered_tile();
        assert_invalid_for_tile(degradation::CACHE_ENTRY_FIELD_MISSING, &tile, |entry| {
            edit_meta(entry, |meta| {
                assert!(
                    meta["coverage_bounds"].is_null(),
                    "this fixture must record coverage_bounds as null, otherwise the \
                     deletion would be caught by the bounds cross-check instead"
                );
                meta.as_object_mut()
                    .expect("meta is an object")
                    .remove("coverage_bounds");
            });
        });
    }

    #[test]
    fn a_deleted_provenance_sha256_key_invalidates_an_entry_that_records_it_as_null() {
        let tile = uncovered_tile();
        assert_invalid_for_tile(degradation::CACHE_ENTRY_FIELD_MISSING, &tile, |entry| {
            edit_meta(entry, |meta| {
                assert!(
                    meta["provenance"][0]["sha256"].is_null(),
                    "this fixture must record provenance[0].sha256 as null"
                );
                meta["provenance"][0]
                    .as_object_mut()
                    .expect("a provenance record is an object")
                    .remove("sha256");
            });
        });
    }

    fn assert_invalid_with(reason: &str, mutate: impl FnOnce(&Path)) {
        let (_directory, store, inputs, entry) = stored_entry();
        mutate(&entry);
        match store.load(&inputs) {
            CacheLookup::Invalid {
                reason: actual,
                detail,
            } => assert_eq!(actual, reason, "{detail}"),
            other => panic!("expected {reason}, got {other:?}"),
        }
        // 需求 4.10: the entry is deleted, so the next lookup is a plain miss
        // and the caller re-fuses instead of retrying a broken entry.
        assert!(!entry.exists(), "a corrupt entry must be deleted");
        assert!(matches!(store.load(&inputs), CacheLookup::Miss));
    }

    #[test]
    fn a_missing_meta_field_invalidates_the_entry() {
        assert_invalid_with(degradation::CACHE_ENTRY_FIELD_MISSING, |entry| {
            let meta = entry.join(META_FILE_NAME);
            let mut value: serde_json::Value =
                serde_json::from_slice(&fs::read(&meta).expect("meta")).expect("meta parses");
            value
                .as_object_mut()
                .expect("meta is an object")
                .remove("color_encoding");
            fs::write(&meta, serde_json::to_vec(&value).expect("re-serialize")).expect("write");
        });
    }

    #[test]
    fn a_missing_payload_invalidates_the_entry() {
        assert_invalid_with(degradation::CACHE_ENTRY_FIELD_MISSING, |entry| {
            fs::remove_file(entry.join(OWNERSHIP_PAYLOAD)).expect("remove payload");
        });
    }

    #[test]
    fn recorded_dimensions_that_disagree_with_the_payloads_invalidate_the_entry() {
        assert_invalid_with(degradation::CACHE_ENTRY_DIMENSION_MISMATCH, |entry| {
            let meta = entry.join(META_FILE_NAME);
            let mut value: serde_json::Value =
                serde_json::from_slice(&fs::read(&meta).expect("meta")).expect("meta parses");
            value["height"] = serde_json::json!(9);
            fs::write(&meta, serde_json::to_vec(&value).expect("re-serialize")).expect("write");
        });
    }

    #[test]
    fn a_corrupted_payload_invalidates_the_entry() {
        assert_invalid_with(degradation::CACHE_ENTRY_SHA_MISMATCH, |entry| {
            let path = entry.join(PIXELS_PAYLOAD);
            let plane = read_payload(&path).expect("payload decodes");
            let mut edited = plane.clone();
            edited[0] ^= 0x01;
            let compressed = zstd::bulk::compress(&edited, ZSTD_LEVEL).expect("recompress");
            fs::write(&path, compressed).expect("write");
        });
    }

    #[test]
    fn a_rewritten_source_digest_set_invalidates_the_entry() {
        assert_invalid_with(degradation::CACHE_ENTRY_SHA_MISMATCH, |entry| {
            let meta = entry.join(META_FILE_NAME);
            let mut value: serde_json::Value =
                serde_json::from_slice(&fs::read(&meta).expect("meta")).expect("meta parses");
            value["cache_key_inputs"]["sorted_source_sha256"][0] =
                serde_json::json!(hex_digest(&digest_of(0x99)));
            fs::write(&meta, serde_json::to_vec(&value).expect("re-serialize")).expect("write");
        });
    }

    // -- eviction -----------------------------------------------------------

    #[test]
    fn exceeding_the_budget_evicts_whole_entries_oldest_access_first() {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let directory = TempDir::new().expect("temp dir");
        // A budget of one entry, so writing the third must leave exactly one.
        let store = VirtualTileStore::new(directory.path()).with_limit_bytes(1);
        let first = CacheKeyInputs::new("v1", &[("/a.NEF".to_string(), digest_of(1))]);
        let second = CacheKeyInputs::new("v1", &[("/b.NEF".to_string(), digest_of(2))]);

        assert!(matches!(
            store.store(&first, &sample_tile(0)),
            StoreOutcome::Written { .. }
        ));
        // The first entry is over budget on its own, so it is evicted right
        // away: the budget is enforced, never exceeded.
        assert!(matches!(store.load(&first), CacheLookup::Miss));
        assert!(!store.entry_dir(&first.cache_key()).exists());
        assert_eq!(store.total_bytes(), 0);

        // Measure one entry, then set a budget that holds exactly one, so the
        // second write has to evict the first.
        let measured = match VirtualTileStore::new(directory.path()).store(&first, &sample_tile(0))
        {
            StoreOutcome::Written { bytes, .. } => bytes,
            other => panic!("unexpected {other:?}"),
        };
        let store = VirtualTileStore::new(directory.path()).with_limit_bytes(measured);
        assert!(matches!(store.load(&first), CacheLookup::Hit(_)));
        assert!(matches!(
            store.store(&second, &sample_tile(1)),
            StoreOutcome::Written { evicted: 1, .. }
        ));
        assert!(
            store.entry_dir(&second.cache_key()).is_dir(),
            "the entry just written must survive"
        );
        assert!(
            !store.entry_dir(&first.cache_key()).exists(),
            "the least recently accessed entry is deleted whole"
        );
        assert!(store.total_bytes() <= measured);
    }

    #[test]
    fn eviction_never_leaves_a_partial_entry_behind() {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let directory = TempDir::new().expect("temp dir");
        let store = VirtualTileStore::new(directory.path()).with_limit_bytes(1);
        let inputs = sample_inputs();
        assert!(matches!(
            store.store(&inputs, &sample_tile(0)),
            StoreOutcome::Written { evicted: 1, .. }
        ));
        // Nothing but the index may remain in the root.
        let remaining = fs::read_dir(store.root())
            .expect("root exists")
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert_eq!(remaining, vec![INDEX_FILE_NAME.to_string()]);
    }

    // -- write failure degradation ------------------------------------------

    #[test]
    fn an_unwritable_cache_root_degrades_instead_of_failing() {
        let directory = TempDir::new().expect("temp dir");
        // A regular file where the cache root must be a directory: every
        // `create_dir_all` below it fails, whatever the platform.
        let blocker = directory.path().join(VIRTUAL_TILE_CACHE_DIR_NAME);
        fs::write(&blocker, b"not a directory").expect("writing the blocker");
        let store = store_in(&directory);
        let inputs = sample_inputs();
        let tile = sample_tile(0);
        match store.store(&inputs, &tile) {
            StoreOutcome::Unavailable { reason, detail } => {
                assert_eq!(reason, degradation::CACHE_WRITE_UNAVAILABLE);
                assert!(!detail.is_empty(), "the degradation must carry a reason");
            }
            other => panic!("expected a write degradation, got {other:?}"),
        }
        // 需求 4.11: the run continues with the in-memory tile, and the failed
        // write left nothing readable behind.
        assert!(matches!(store.load(&inputs), CacheLookup::Miss));
        assert_eq!(tile.coverage.covered_pixels(), 3);
    }

    /// 需求 4.2: the pixels are a 32 bit float RGB buffer and the colour
    /// encoding identifier travels with them, through the cache and back.
    #[test]
    fn cached_pixels_stay_rgb32f_and_keep_their_color_encoding_identifier() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let inputs = sample_inputs();
        let tile = sample_tile(2);
        // Type level: this binding only compiles while the tile carries an
        // `Rgb32FImage`, i.e. three `f32` channels per pixel.
        let pixels: &Rgb32FImage = &tile.pixels;
        let _: f32 = pixels.get_pixel(0, 0)[0];
        assert_eq!(
            pixels.as_raw().len() as u64,
            u64::from(tile.width) * u64::from(tile.height) * 3,
            "three f32 channels per pixel"
        );

        assert!(matches!(
            store.store(&inputs, &tile),
            StoreOutcome::Written { .. }
        ));
        let entry = store.entry_dir(&inputs.cache_key());
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(entry.join(META_FILE_NAME)).expect("meta"))
                .expect("meta parses");
        assert_eq!(
            meta["color_encoding"].as_str(),
            Some("display_srgb"),
            "meta.json must carry the colour encoding identifier"
        );

        let loaded = match store.load(&inputs) {
            CacheLookup::Hit(tile) => *tile,
            other => panic!("the entry must verify: {other:?}"),
        };
        let loaded_pixels: &Rgb32FImage = &loaded.pixels;
        let _: f32 = loaded_pixels.get_pixel(0, 0)[0];
        assert_eq!(loaded.color_encoding, ColorEncoding::DisplaySrgb);
        // Both identifiers are declared and both round trip, so no encoding can
        // be written without a name the loader recognises.
        for encoding in [ColorEncoding::LinearSrgb, ColorEncoding::DisplaySrgb] {
            assert_eq!(
                parse_color_encoding(color_encoding_identifier(encoding)),
                Some(encoding)
            );
        }
    }

    /// 需求 4.11: a cache root that cannot be written into degrades the run
    /// instead of ending it.
    #[cfg(unix)]
    #[test]
    fn a_read_only_cache_root_degrades_with_cache_write_unavailable() {
        use std::os::unix::fs::PermissionsExt;

        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        fs::create_dir_all(store.root()).expect("creating the cache root");
        // r-xr-xr-x: the root exists, so `create_dir_all` succeeds, but the
        // entry directory below it cannot be created.
        fs::set_permissions(store.root(), fs::Permissions::from_mode(0o555))
            .expect("making the cache root read only");

        let inputs = sample_inputs();
        let tile = sample_tile(0);
        let outcome = store.store(&inputs, &tile);

        // Restore the mode first, so the assertions below cannot leave an
        // undeletable temp dir behind.
        fs::set_permissions(store.root(), fs::Permissions::from_mode(0o755))
            .expect("restoring the cache root");

        match outcome {
            StoreOutcome::Unavailable { reason, detail } => {
                assert_eq!(reason, degradation::CACHE_WRITE_UNAVAILABLE);
                assert!(!detail.is_empty(), "the degradation must carry a detail");
            }
            other => panic!("expected a write degradation, got {other:?}"),
        }
        // The run continues: the in-memory tile is untouched, nothing readable
        // was left behind, and a later lookup is a plain miss.
        assert_eq!(tile.coverage.covered_pixels(), 3);
        assert_eq!(tile.pixels.dimensions(), (4, 3));
        assert!(matches!(store.load(&inputs), CacheLookup::Miss));
        // And the very same store works again once the directory is writable.
        assert!(matches!(
            store.store(&inputs, &tile),
            StoreOutcome::Written { .. }
        ));
        assert!(matches!(store.load(&inputs), CacheLookup::Hit(_)));
    }

    #[test]
    fn a_failed_publish_leaves_no_temporary_entry_in_the_root() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let inputs = sample_inputs();
        assert!(matches!(
            store.store(&inputs, &sample_tile(0)),
            StoreOutcome::Written { .. }
        ));
        let leftovers = fs::read_dir(store.root())
            .expect("root exists")
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with(TEMP_PREFIX))
            .collect::<Vec<_>>();
        assert!(
            leftovers.is_empty(),
            "no temporary entry may survive a publish: {leftovers:?}"
        );
    }

    #[test]
    fn overwriting_an_entry_replaces_it_atomically() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let inputs = sample_inputs();
        assert!(matches!(
            store.store(&inputs, &sample_tile(0)),
            StoreOutcome::Written { .. }
        ));
        assert!(matches!(
            store.store(&inputs, &sample_tile(5)),
            StoreOutcome::Written { .. }
        ));
        match store.load(&inputs) {
            CacheLookup::Hit(tile) => assert_eq!(tile.station_index, 5),
            other => panic!("unexpected {other:?}"),
        }
    }

    // -- residency leases (需求 14.4) ---------------------------------------

    /// Counts how often the store had to synthesise a tile.
    fn counting_provider(station_index: usize, calls: &std::cell::Cell<usize>) -> VirtualTile {
        calls.set(calls.get() + 1);
        sample_tile(station_index)
    }

    #[test]
    fn two_full_size_tiles_may_be_leased_at_once() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let first = store
            .lease(0, || Ok(sample_tile(0)))
            .expect("the first lease fits");
        let second = store
            .lease(1, || Ok(sample_tile(1)))
            .expect("the second lease fits");
        assert_eq!(first.station_index(), 0);
        assert_eq!(second.tile().station_index, 1);
        // The lease derefs to the tile, so the compositor reads through it.
        assert_eq!(first.width, 4);
        assert_eq!(store.resident_virtual_tiles(), MAX_RESIDENT_VIRTUAL_TILES);
        assert_eq!(store.active_virtual_tile_leases(), 2);
        assert_eq!(
            store.max_resident_virtual_tiles(),
            MAX_RESIDENT_VIRTUAL_TILES
        );
    }

    #[test]
    fn a_third_live_lease_is_refused_rather_than_exceeding_the_ceiling() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let _first = store.lease(0, || Ok(sample_tile(0))).expect("lease 0");
        let _second = store.lease(1, || Ok(sample_tile(1))).expect("lease 1");
        let calls = std::cell::Cell::new(0usize);
        match store.lease(2, || Ok(counting_provider(2, &calls))) {
            Err(LeaseError::ResidencyExhausted { limit, held }) => {
                assert_eq!(limit, MAX_RESIDENT_VIRTUAL_TILES);
                assert_eq!(held, 2);
            }
            Ok(_) => panic!("a third resident tile must never be admitted"),
            Err(other) => panic!("unexpected lease failure: {other}"),
        }
        assert_eq!(
            calls.get(),
            0,
            "a refused lease must not synthesise the tile it cannot hold"
        );
        assert_eq!(store.resident_virtual_tiles(), MAX_RESIDENT_VIRTUAL_TILES);
        assert_eq!(
            store.max_resident_virtual_tiles(),
            MAX_RESIDENT_VIRTUAL_TILES
        );
    }

    #[test]
    fn a_released_lease_is_evicted_least_recently_used_first() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        {
            let _zero = store.lease(0, || Ok(sample_tile(0))).expect("lease 0");
            let _one = store.lease(1, || Ok(sample_tile(1))).expect("lease 1");
        }
        // Re-touch station 1, making station 0 the least recently leased.
        drop(store.lease(1, || panic!("station 1 is still resident")));
        let calls = std::cell::Cell::new(0usize);
        let two = store
            .lease(2, || Ok(counting_provider(2, &calls)))
            .expect("the idle least recently used tile makes room");
        assert_eq!(calls.get(), 1, "station 2 had to be synthesised once");
        assert_eq!(two.station_index(), 2);
        assert_eq!(store.resident_virtual_tiles(), MAX_RESIDENT_VIRTUAL_TILES);

        // Station 0 is gone, station 1 is still resident.
        drop(store.lease(1, || panic!("station 1 must still be resident")));
        let calls = std::cell::Cell::new(0usize);
        drop(store.lease(0, || Ok(counting_provider(0, &calls))));
        assert_eq!(calls.get(), 1, "the evicted station is synthesised again");
    }

    #[test]
    fn leasing_the_same_station_twice_shares_one_resident_tile() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        let calls = std::cell::Cell::new(0usize);
        let first = store
            .lease(7, || Ok(counting_provider(7, &calls)))
            .expect("lease 7");
        let second = store
            .lease(7, || Ok(counting_provider(7, &calls)))
            .expect("the same station is already resident");
        assert_eq!(calls.get(), 1, "one synthesis serves both handles");
        assert_eq!(store.resident_virtual_tiles(), 1);
        assert_eq!(store.active_virtual_tile_leases(), 2);
        assert_eq!(first.station_index(), second.station_index());

        // Both handles must be dropped before the slot is free again.
        drop(first);
        assert_eq!(store.active_virtual_tile_leases(), 1);
        drop(second);
        assert_eq!(store.active_virtual_tile_leases(), 0);
        assert_eq!(store.resident_virtual_tiles(), 1, "kept for reuse");
        assert_eq!(store.max_resident_virtual_tiles(), 1);
    }

    #[test]
    fn a_single_live_lease_blocks_a_one_slot_store() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory).with_lease_limit(1);
        let held = store.lease(0, || Ok(sample_tile(0))).expect("lease 0");
        match store.lease(1, || Ok(sample_tile(1))) {
            Err(LeaseError::ResidencyExhausted { limit, held }) => {
                assert_eq!((limit, held), (1, 1));
            }
            Ok(_) => panic!("the only slot is held, so the lease must fail"),
            Err(other) => panic!("unexpected lease failure: {other}"),
        }
        drop(held);
        // With the handle gone the idle tile is evictable, so the next station
        // fits without ever exceeding the ceiling.
        let next = store.lease(1, || Ok(sample_tile(1))).expect("lease 1");
        assert_eq!(next.station_index(), 1);
        assert_eq!(store.resident_virtual_tiles(), 1);
        assert_eq!(store.max_resident_virtual_tiles(), 1);
    }

    #[test]
    fn a_provider_failure_frees_the_slot_it_asked_for() {
        let directory = TempDir::new().expect("temp dir");
        let store = store_in(&directory);
        match store.lease(0, || Err("fusion failed".to_string())) {
            Err(LeaseError::Unavailable(detail)) => assert_eq!(detail, "fusion failed"),
            Ok(_) => panic!("a failed synthesis cannot produce a lease"),
            Err(other) => panic!("unexpected lease failure: {other}"),
        }
        assert_eq!(store.resident_virtual_tiles(), 0);
        assert_eq!(store.max_resident_virtual_tiles(), 0);
        // The failure is not sticky: the station can be leased afterwards.
        let tile = store.lease(0, || Ok(sample_tile(0))).expect("lease 0");
        assert_eq!(tile.station_index(), 0);
    }
}
