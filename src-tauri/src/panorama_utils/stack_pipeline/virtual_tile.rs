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
pub(crate) mod test_access;

#[cfg(test)]
mod tests;
