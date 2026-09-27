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

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use half::f16;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::degradation;
use super::report::VIRTUAL_TILE_CACHE_LIMIT_BYTES;
use crate::panorama_utils::stitching::{
    ColorEncoding, ConfidenceMap, CoverageMask, OwnershipMap, SourceProvenance, VirtualTile,
};

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

// ---------------------------------------------------------------------------
// Source content digest (the third cache key ingredient)
// ---------------------------------------------------------------------------

/// Read buffer for the source digest.  Large enough that a 50 MB RAW is a few
/// hundred reads, small enough to stay out of the way of the decoders.
const DIGEST_CHUNK_BYTES: usize = 1 << 20;

/// SHA-256 over every byte of `path`, streamed so a 50 MB RAW never lands in
/// memory as a whole (需求 4.1 / 4.5).
///
/// The file is opened **read only** with [`File::open`] and never written, which
/// is the access mode 需求 4.8 demands of every Source_RAW.
pub(crate) fn source_file_sha256(path: &Path) -> Result<[u8; 32], String> {
    let file = File::open(path)
        .map_err(|error| format!("Failed to open {} for hashing: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut chunk = vec![0u8; DIGEST_CHUNK_BYTES];
    loop {
        let read = reader
            .read(&mut chunk)
            .map_err(|error| format!("Failed to read {} for hashing: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(hasher.finalize().into())
}

/// Lower hex of a 32 byte digest.
pub(crate) fn hex_digest(digest: &[u8; 32]) -> String {
    hex::encode(digest)
}

fn parse_hex_digest(value: &str) -> Option<[u8; 32]> {
    let bytes = hex::decode(value).ok()?;
    <[u8; 32]>::try_from(bytes.as_slice()).ok()
}

// ---------------------------------------------------------------------------
// cache_key
// ---------------------------------------------------------------------------

/// The three ingredients of `cache_key` (需求 4.5), each kept separately so
/// that [`VirtualTileStore::load`] can re-check them item by item (需求 4.9).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CacheKeyInputs {
    /// `STACK_PIPELINE_VERSION`.
    pub pipeline_version: String,
    /// Absolute source paths, ascending by raw bytes.
    pub sorted_paths: Vec<String>,
    /// Source content digests in lower hex, ascending by raw bytes.  Sorted
    /// independently of `sorted_paths` on purpose: the requirement names a
    /// path *set* and a digest *set*, so neither order may depend on the other.
    pub sorted_source_sha256: Vec<String>,
}

impl CacheKeyInputs {
    /// Build the ingredients from `(absolute_path, digest)` pairs in any order.
    pub(crate) fn new(pipeline_version: &str, sources: &[(String, [u8; 32])]) -> Self {
        let mut sorted_paths = sources
            .iter()
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        sorted_paths.sort_unstable();
        let mut sorted_source_sha256 = sources
            .iter()
            .map(|(_, digest)| hex_digest(digest))
            .collect::<Vec<_>>();
        sorted_source_sha256.sort_unstable();
        Self {
            pipeline_version: pipeline_version.to_string(),
            sorted_paths,
            sorted_source_sha256,
        }
    }

    /// `SHA-256( pipeline_version ‖ 0x00 ‖ sorted_paths ‖ 0x00 ‖ sorted_source_sha256 )`.
    ///
    /// Every element is length prefixed as well as separated, so no two
    /// different ingredient lists can produce the same byte stream (the
    /// separator alone would let `["a", "b"]` and `["a\0b"]` collide).
    pub(crate) fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update((self.pipeline_version.len() as u64).to_be_bytes());
        hasher.update(self.pipeline_version.as_bytes());
        hasher.update([0u8]);
        hasher.update((self.sorted_paths.len() as u64).to_be_bytes());
        for path in &self.sorted_paths {
            hasher.update((path.len() as u64).to_be_bytes());
            hasher.update(path.as_bytes());
        }
        hasher.update([0u8]);
        hasher.update((self.sorted_source_sha256.len() as u64).to_be_bytes());
        for digest in &self.sorted_source_sha256 {
            hasher.update((digest.len() as u64).to_be_bytes());
            hasher.update(digest.as_bytes());
        }
        hasher.finalize().into()
    }

    /// `cache_key` in lower hex; also the entry directory name.
    pub(crate) fn cache_key(&self) -> String {
        hex_digest(&self.digest())
    }
}

// ---------------------------------------------------------------------------
// meta.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PayloadMeta {
    /// Size of the compressed file on disk, the unit of the 64 GiB budget.
    bytes: u64,
    /// Number of decompressed elements; the dimension cross-check of 需求 4.10.
    elements: u64,
    /// SHA-256 of the **decompressed** plane, so the check attests the payload
    /// content rather than a particular compressor build.
    sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct CoverageBoundsMeta {
    left: u32,
    top: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ProvenanceMeta {
    path: String,
    /// `None` when the digest was not available at fusion time; the field is
    /// always present in the document so a truly missing key is detectable.
    sha256: Option<String>,
    owned_pixels: u64,
}

/// `meta.json`.  No field carries `#[serde(default)]`, so an absent key fails
/// the parse for every non-`Option` field.  `Option` fields are the exception
/// `serde` cannot cover — an absent key deserialises to `None` without an error
/// — so the document is additionally checked against the declared key sets of
/// [`verify_meta_keys`] before it is trusted (需求 4.10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CacheEntryMeta {
    schema: u32,
    pipeline_version: String,
    station_index: usize,
    width: u32,
    height: u32,
    /// Row major 3x3, as IEEE-754 bit patterns rather than as JSON decimals.
    ///
    /// `serde_json` writes the shortest round-trip decimal but its default
    /// parser is only correct to within one ULP (exact parsing is behind the
    /// crate's `float_roundtrip` feature), so a placement written as a decimal
    /// can come back one ULP away from the value that was fused.  That would
    /// put a cache hit and a cache miss on different sub-pixel placements and
    /// break the byte-for-byte reproducibility of 需求 4.3 / 14.6, so the
    /// coefficients travel as bits.  An entry written before this field existed
    /// fails the parse and is re-fused as a corrupt entry (需求 4.10).
    tile_to_world_bits: Vec<u64>,
    color_encoding: String,
    coverage_bounds: Option<CoverageBoundsMeta>,
    owner_legend: Vec<String>,
    provenance: Vec<ProvenanceMeta>,
    cache_key: String,
    /// The three ingredients, stored for the item by item re-check of 需求 4.9.
    cache_key_inputs: CacheKeyInputs,
    payloads: BTreeMap<String, PayloadMeta>,
}

/// The keys [`CacheEntryMeta`] declares at the document root.
const META_KEYS: &[&str] = &[
    "schema",
    "pipeline_version",
    "station_index",
    "width",
    "height",
    "tile_to_world_bits",
    "color_encoding",
    "coverage_bounds",
    "owner_legend",
    "provenance",
    "cache_key",
    "cache_key_inputs",
    "payloads",
];

/// The keys [`CacheKeyInputs`] declares, under `cache_key_inputs`.
const CACHE_KEY_INPUTS_KEYS: &[&str] =
    &["pipeline_version", "sorted_paths", "sorted_source_sha256"];

/// The keys [`CoverageBoundsMeta`] declares, under `coverage_bounds`.
const COVERAGE_BOUNDS_KEYS: &[&str] = &["left", "top", "width", "height"];

/// The keys [`ProvenanceMeta`] declares, per `provenance` element.
const PROVENANCE_KEYS: &[&str] = &["path", "sha256", "owned_pixels"];

/// The keys [`PayloadMeta`] declares, per `payloads` value.
const PAYLOAD_KEYS: &[&str] = &["bytes", "elements", "sha256"];

/// Require every key of `expected` on the object at `location`.
///
/// Presence only: a key whose value is `null` counts as present, because that
/// is how the two nullable fields (`coverage_bounds` and a provenance
/// `sha256`) legitimately encode "not available".  A key that is *gone* is the
/// corruption of 需求 4.10 and must be reported even for those fields.
fn expect_meta_keys(
    value: &serde_json::Value,
    location: &str,
    expected: &[&str],
) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{location} is not a JSON object"))?;
    for key in expected {
        if !object.contains_key(*key) {
            return Err(format!("{location} is missing {key}"));
        }
    }
    Ok(())
}

/// Check the raw `meta.json` key sets, at every nesting level, before the
/// document is deserialised (需求 4.10).
///
/// This exists because `serde`'s derived `Deserialize` treats an absent key for
/// an `Option` field as `None` rather than as an error.  Without this pass a
/// deleted `coverage_bounds` or `provenance[i].sha256` would parse cleanly, and
/// for a tile whose real value is `None` anyway — a fully uncovered tile, or a
/// source whose digest was unavailable — nothing downstream could tell the
/// difference, so the damaged entry would be served as a hit.
fn verify_meta_keys(meta_bytes: &[u8]) -> Result<(), String> {
    let document: serde_json::Value = serde_json::from_slice(meta_bytes)
        .map_err(|error| format!("meta.json is not a JSON document: {error}"))?;
    expect_meta_keys(&document, "meta.json", META_KEYS)?;
    expect_meta_keys(
        &document["cache_key_inputs"],
        "cache_key_inputs",
        CACHE_KEY_INPUTS_KEYS,
    )?;
    // Nullable: present-and-null is the legitimate encoding of "no covered
    // pixel", so only its own key set is checked when it is an object.
    let bounds = &document["coverage_bounds"];
    if !bounds.is_null() {
        expect_meta_keys(bounds, "coverage_bounds", COVERAGE_BOUNDS_KEYS)?;
    }
    let provenance = document["provenance"]
        .as_array()
        .ok_or_else(|| "provenance is not a JSON array".to_string())?;
    for (index, record) in provenance.iter().enumerate() {
        expect_meta_keys(record, &format!("provenance[{index}]"), PROVENANCE_KEYS)?;
    }
    let payloads = document["payloads"]
        .as_object()
        .ok_or_else(|| "payloads is not a JSON object".to_string())?;
    for (name, payload) in payloads {
        expect_meta_keys(payload, &format!("payloads[{name}]"), PAYLOAD_KEYS)?;
    }
    Ok(())
}

fn color_encoding_identifier(encoding: ColorEncoding) -> &'static str {
    match encoding {
        ColorEncoding::LinearSrgb => "linear_srgb",
        ColorEncoding::DisplaySrgb => "display_srgb",
    }
}

/// Wrap a detail string as the "recorded size disagrees with the payload"
/// verification failure of 需求 4.10.
fn dimension_mismatch(detail: String) -> (&'static str, String) {
    (degradation::CACHE_ENTRY_DIMENSION_MISMATCH, detail)
}

fn parse_color_encoding(value: &str) -> Option<ColorEncoding> {
    match value {
        "linear_srgb" => Some(ColorEncoding::LinearSrgb),
        "display_srgb" => Some(ColorEncoding::DisplaySrgb),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// index.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct IndexEntry {
    last_access_epoch_ms: u64,
    bytes: u64,
}

/// `index.json`.  A `BTreeMap` rather than a `HashMap` so serialisation and
/// eviction tie-breaks are ordered by key (需求 14.6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CacheIndex {
    schema: u32,
    entries: BTreeMap<String, IndexEntry>,
}

impl Default for CacheIndex {
    fn default() -> Self {
        Self {
            schema: INDEX_SCHEMA,
            entries: BTreeMap::new(),
        }
    }
}

impl CacheIndex {
    fn total_bytes(&self) -> u64 {
        self.entries
            .values()
            .fold(0u64, |total, entry| total.saturating_add(entry.bytes))
    }
}

fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Public results
// ---------------------------------------------------------------------------

/// Outcome of a cache lookup.
#[derive(Debug)]
pub(crate) enum CacheLookup {
    /// The entry verified in full; no Source_RAW was decoded (需求 4.5).
    Hit(Box<VirtualTile>),
    /// No entry for this `cache_key` (需求 4.9).
    Miss,
    /// An entry existed but failed verification.  It has already been deleted;
    /// the caller must re-fuse the station and record `reason` (需求 4.10).
    Invalid {
        reason: &'static str,
        detail: String,
    },
}

/// Outcome of a cache write.
#[derive(Debug)]
pub(crate) enum StoreOutcome {
    Written {
        /// Compressed bytes of this entry.
        bytes: u64,
        /// Entries deleted to stay inside the budget (需求 4.4).
        evicted: u64,
        /// Cache size after the write and the eviction pass.
        total_bytes: u64,
    },
    /// The cache is unusable for this entry.  The caller keeps the in-memory
    /// tile and continues (需求 4.11).
    Unavailable {
        reason: &'static str,
        detail: String,
    },
}

// ---------------------------------------------------------------------------
// Residency leases (需求 14.4)
// ---------------------------------------------------------------------------

/// How many full size Virtual_Tiles may be resident in memory at once
/// (需求 14.4).  Two is the minimum a pairwise tile-to-tile step needs.
pub(crate) const MAX_RESIDENT_VIRTUAL_TILES: usize = 2;

/// Why a lease could not be granted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LeaseError {
    /// Every resident slot is held by a live [`TileLease`], so admitting one
    /// more tile would break the ceiling of 需求 14.4.  The caller must drop a
    /// lease it already holds and ask again.
    ResidencyExhausted { limit: usize, held: usize },
    /// The tile itself could not be produced: neither the cache nor the caller's
    /// fusion closure returned one.
    Unavailable(String),
}

impl std::fmt::Display for LeaseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResidencyExhausted { limit, held } => write!(
                formatter,
                "cannot hold more than {limit} full size virtual tiles at once ({held} leased)"
            ),
            Self::Unavailable(detail) => write!(formatter, "{detail}"),
        }
    }
}

/// One tile currently held in memory by the store.
struct Resident {
    station_index: usize,
    tile: Arc<VirtualTile>,
    /// Number of live [`TileLease`] handles on this tile.  A resident with zero
    /// handles is kept for reuse but may be evicted to make room.
    held: usize,
    /// Value of [`LeaseState::clock`] at the last lease.  Lowest loses.
    last_access: u64,
}

#[derive(Default)]
struct LeaseState {
    residents: Vec<Resident>,
    /// Monotonic access counter.  A logical clock rather than a wall clock, so
    /// the eviction order cannot depend on timer resolution (需求 14.6).
    clock: u64,
    /// Highest `residents.len()` this registry ever reached.
    max_resident: usize,
}

/// Enforces the residency ceiling of 需求 14.4 and observes the peak.
///
/// Held behind an `Arc` so a [`TileLease`] can return its slot after the store
/// reference the caller leased through has gone out of scope.
pub(crate) struct LeaseRegistry {
    limit: usize,
    state: Mutex<LeaseState>,
}

impl LeaseRegistry {
    fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(1),
            state: Mutex::new(LeaseState::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LeaseState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Grant a lease on `station_index`, calling `provide` only when the tile is
    /// not resident already.
    fn acquire(
        &self,
        station_index: usize,
        provide: impl FnOnce() -> Result<VirtualTile, String>,
    ) -> Result<Arc<VirtualTile>, LeaseError> {
        let mut state = self.lock();
        state.clock = state.clock.saturating_add(1);
        let now = state.clock;
        if let Some(resident) = state
            .residents
            .iter_mut()
            .find(|resident| resident.station_index == station_index)
        {
            // Already in memory: a second lease on the same station costs no
            // additional memory, so it never has to evict anything.
            resident.held += 1;
            resident.last_access = now;
            return Ok(Arc::clone(&resident.tile));
        }
        // Make room: least recently leased first, and only among tiles nobody
        // holds.  A tile with a live lease is still being read, so evicting it
        // would hand the caller a dangling view of the mosaic.
        while state.residents.len() >= self.limit {
            let victim = state
                .residents
                .iter()
                .enumerate()
                .filter(|(_, resident)| resident.held == 0)
                .min_by(|left, right| {
                    left.1
                        .last_access
                        .cmp(&right.1.last_access)
                        .then_with(|| left.1.station_index.cmp(&right.1.station_index))
                })
                .map(|(index, _)| index);
            let Some(victim) = victim else {
                let held = state
                    .residents
                    .iter()
                    .filter(|resident| resident.held > 0)
                    .count();
                return Err(LeaseError::ResidencyExhausted {
                    limit: self.limit,
                    held,
                });
            };
            state.residents.remove(victim);
        }
        // `provide` may decode and fuse, which is slow, but it runs under the
        // lock on purpose: the slot it was admitted into must not be handed to a
        // second caller in the meantime.
        let tile = Arc::new(provide().map_err(LeaseError::Unavailable)?);
        state.residents.push(Resident {
            station_index,
            tile: Arc::clone(&tile),
            held: 1,
            last_access: now,
        });
        state.max_resident = state.max_resident.max(state.residents.len());
        Ok(tile)
    }

    /// Return one handle.  The tile stays resident for reuse; it becomes
    /// evictable once no handle is left.
    fn release(&self, station_index: usize) {
        let mut state = self.lock();
        if let Some(resident) = state
            .residents
            .iter_mut()
            .find(|resident| resident.station_index == station_index)
        {
            resident.held = resident.held.saturating_sub(1);
        }
    }

    fn resident_count(&self) -> usize {
        self.lock().residents.len()
    }

    fn held_count(&self) -> usize {
        self.lock()
            .residents
            .iter()
            .map(|resident| resident.held)
            .sum()
    }

    fn max_resident(&self) -> usize {
        self.lock().max_resident
    }
}

/// RAII lease on a resident Virtual_Tile.  Dropping it returns the slot; the
/// tile itself may stay in memory for reuse until the store needs the slot.
pub(crate) struct TileLease {
    station_index: usize,
    tile: Arc<VirtualTile>,
    registry: Arc<LeaseRegistry>,
}

impl TileLease {
    pub(crate) fn station_index(&self) -> usize {
        self.station_index
    }

    pub(crate) fn tile(&self) -> &VirtualTile {
        &self.tile
    }
}

impl std::ops::Deref for TileLease {
    type Target = VirtualTile;

    fn deref(&self) -> &Self::Target {
        &self.tile
    }
}

impl Drop for TileLease {
    fn drop(&mut self) {
        self.registry.release(self.station_index);
    }
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

/// Versioned, lossless Virtual_Tile cache.
///
/// `index_lock` serialises the read-modify-write of `index.json`.  A run fuses
/// its stations one after another today, but the compositor may well fuse two
/// in parallel later, and a lost index update would silently leak disk.
pub(crate) struct VirtualTileStore {
    root: PathBuf,
    limit_bytes: u64,
    index_lock: Mutex<()>,
    leases: Arc<LeaseRegistry>,
}

impl VirtualTileStore {
    /// Store rooted at `<app_cache_dir>/stack-virtual-tiles/v1`.
    pub(crate) fn new(app_cache_dir: &Path) -> Self {
        Self::with_root(
            app_cache_dir
                .join(VIRTUAL_TILE_CACHE_DIR_NAME)
                .join(VIRTUAL_TILE_CACHE_LAYOUT_VERSION),
        )
    }

    /// Store rooted at an explicit directory.  Used by the unit tests and by
    /// any caller that already resolved the versioned root.
    pub(crate) fn with_root(root: PathBuf) -> Self {
        Self {
            root,
            limit_bytes: VIRTUAL_TILE_CACHE_LIMIT_BYTES,
            index_lock: Mutex::new(()),
            leases: Arc::new(LeaseRegistry::new(MAX_RESIDENT_VIRTUAL_TILES)),
        }
    }

    /// Lower the 64 GiB budget, so the eviction pass can be exercised without
    /// writing 64 GiB.  Production never calls this.
    #[cfg(test)]
    pub(crate) fn with_limit_bytes(mut self, limit_bytes: u64) -> Self {
        self.limit_bytes = limit_bytes;
        self
    }

    /// Lower the residency ceiling so the "nothing left to evict" branch can be
    /// reached with one tile.  Production always uses
    /// [`MAX_RESIDENT_VIRTUAL_TILES`].
    #[cfg(test)]
    pub(crate) fn with_lease_limit(mut self, limit: usize) -> Self {
        self.leases = Arc::new(LeaseRegistry::new(limit));
        self
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }

    fn entry_dir(&self, cache_key: &str) -> PathBuf {
        self.root.join(cache_key)
    }

    fn index_path(&self) -> PathBuf {
        self.root.join(INDEX_FILE_NAME)
    }

    /// Current cache size according to `index.json`.
    pub(crate) fn total_bytes(&self) -> u64 {
        let _guard = self.lock_index();
        self.read_index().total_bytes()
    }

    fn lock_index(&self) -> std::sync::MutexGuard<'_, ()> {
        self.index_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `index.json`, or an empty index when it is absent or unreadable.  A
    /// broken index costs cache hits, never a run (需求 4.11).
    fn read_index(&self) -> CacheIndex {
        fs::read(self.index_path())
            .ok()
            .and_then(|bytes| serde_json::from_slice::<CacheIndex>(&bytes).ok())
            .filter(|index| index.schema == INDEX_SCHEMA)
            .unwrap_or_default()
    }

    fn write_index(&self, index: &CacheIndex) -> Result<(), String> {
        let serialized = serde_json::to_vec_pretty(index).map_err(|error| {
            format!("Failed to serialize the virtual tile cache index: {error}")
        })?;
        // The index is small; write it through a sibling temp file so a crash
        // cannot leave a truncated document behind.
        let temp = self
            .root
            .join(format!("{TEMP_PREFIX}{}.json", Uuid::new_v4()));
        write_file_sync(&temp, &serialized)?;
        fs::rename(&temp, self.index_path()).map_err(|error| {
            let _ = fs::remove_file(&temp);
            format!("Failed to publish the virtual tile cache index: {error}")
        })
    }

    // -- load ---------------------------------------------------------------

    /// Look up the entry for `inputs`.
    ///
    /// Verification order follows 需求 4.9 / 4.10: the entry must exist, parse
    /// with every field present, agree on all three cache key ingredients,
    /// agree on the recorded dimensions, and re-produce every payload digest.
    /// The first failure deletes the entry and returns
    /// [`CacheLookup::Invalid`]; no Source_RAW is opened on any path.
    pub(crate) fn load(&self, inputs: &CacheKeyInputs) -> CacheLookup {
        let cache_key = inputs.cache_key();
        let entry = self.entry_dir(&cache_key);
        if !entry.is_dir() {
            return CacheLookup::Miss;
        }
        match self.load_verified(&entry, inputs, &cache_key) {
            Ok(tile) => {
                self.touch(&cache_key);
                CacheLookup::Hit(Box::new(tile))
            }
            Err((reason, detail)) => {
                self.discard_entry(&cache_key);
                CacheLookup::Invalid { reason, detail }
            }
        }
    }

    fn load_verified(
        &self,
        entry: &Path,
        inputs: &CacheKeyInputs,
        cache_key: &str,
    ) -> Result<VirtualTile, (&'static str, String)> {
        let meta_bytes = fs::read(entry.join(META_FILE_NAME)).map_err(|error| {
            (
                degradation::CACHE_ENTRY_FIELD_MISSING,
                format!("meta.json is unreadable: {error}"),
            )
        })?;
        // Key sets first, on the raw document: `serde` would silently accept an
        // absent key for either of the two nullable fields (需求 4.10).
        verify_meta_keys(&meta_bytes)
            .map_err(|detail| (degradation::CACHE_ENTRY_FIELD_MISSING, detail))?;
        // A missing or mistyped key fails here, which is exactly the "field
        // missing" condition of 需求 4.10.
        let meta: CacheEntryMeta = serde_json::from_slice(&meta_bytes).map_err(|error| {
            (
                degradation::CACHE_ENTRY_FIELD_MISSING,
                format!("meta.json is incomplete: {error}"),
            )
        })?;
        if meta.schema != META_SCHEMA {
            return Err((
                degradation::CACHE_ENTRY_FIELD_MISSING,
                format!("meta.json schema {} is not {META_SCHEMA}", meta.schema),
            ));
        }
        if meta.tile_to_world_bits.len() != 9 {
            return Err((
                degradation::CACHE_ENTRY_FIELD_MISSING,
                format!(
                    "tile_to_world_bits holds {} values instead of 9",
                    meta.tile_to_world_bits.len()
                ),
            ));
        }
        let color_encoding = parse_color_encoding(&meta.color_encoding).ok_or_else(|| {
            (
                degradation::CACHE_ENTRY_FIELD_MISSING,
                format!("unknown color_encoding {:?}", meta.color_encoding),
            )
        })?;
        for name in REQUIRED_PAYLOADS {
            if !meta.payloads.contains_key(*name) {
                return Err((
                    degradation::CACHE_ENTRY_FIELD_MISSING,
                    format!("meta.json does not describe {name}"),
                ));
            }
        }

        // The three cache key ingredients, re-checked item by item (需求 4.9).
        if meta.cache_key != cache_key {
            return Err((
                degradation::CACHE_ENTRY_SHA_MISMATCH,
                format!(
                    "entry records cache_key {} under {cache_key}",
                    meta.cache_key
                ),
            ));
        }
        if meta.pipeline_version != inputs.pipeline_version
            || meta.cache_key_inputs.pipeline_version != inputs.pipeline_version
        {
            return Err((
                degradation::CACHE_ENTRY_SHA_MISMATCH,
                format!(
                    "entry pipeline version {:?} is not {:?}",
                    meta.cache_key_inputs.pipeline_version, inputs.pipeline_version
                ),
            ));
        }
        if meta.cache_key_inputs.sorted_paths != inputs.sorted_paths {
            return Err((
                degradation::CACHE_ENTRY_SHA_MISMATCH,
                "entry records a different source path set".to_string(),
            ));
        }
        if meta.cache_key_inputs.sorted_source_sha256 != inputs.sorted_source_sha256 {
            return Err((
                degradation::CACHE_ENTRY_SHA_MISMATCH,
                "entry records a different source digest set".to_string(),
            ));
        }
        if meta.cache_key_inputs.digest() != inputs.digest() {
            return Err((
                degradation::CACHE_ENTRY_SHA_MISMATCH,
                "recorded ingredients do not hash to the recorded cache_key".to_string(),
            ));
        }

        let pixel_count = u64::from(meta.width) * u64::from(meta.height);
        let payload = |name: &'static str| -> Result<(Vec<u8>, u64), (&'static str, String)> {
            let recorded = &meta.payloads[name];
            let plane = read_payload(&entry.join(name)).map_err(|error| {
                (
                    degradation::CACHE_ENTRY_FIELD_MISSING,
                    format!("{name}: {error}"),
                )
            })?;
            let digest = Sha256::digest(&plane);
            if hex_digest(&digest.into()) != recorded.sha256 {
                return Err((
                    degradation::CACHE_ENTRY_SHA_MISMATCH,
                    format!("{name} does not match its recorded digest"),
                ));
            }
            Ok((plane, recorded.elements))
        };

        let expect_elements = |name: &'static str,
                               recorded: u64,
                               wanted: u64|
         -> Result<(), (&'static str, String)> {
            (recorded == wanted).then_some(()).ok_or_else(|| {
                (
                    degradation::CACHE_ENTRY_DIMENSION_MISMATCH,
                    format!(
                        "{name} records {recorded} element(s) for a {}x{} tile",
                        meta.width, meta.height
                    ),
                )
            })
        };

        let (pixel_plane, pixel_elements) = payload(PIXELS_PAYLOAD)?;
        expect_elements(PIXELS_PAYLOAD, pixel_elements, pixel_count * 3)?;
        let (ownership_plane, ownership_elements) = payload(OWNERSHIP_PAYLOAD)?;
        expect_elements(OWNERSHIP_PAYLOAD, ownership_elements, pixel_count)?;
        let (coverage_plane, coverage_elements) = payload(COVERAGE_PAYLOAD)?;
        expect_elements(COVERAGE_PAYLOAD, coverage_elements, pixel_count)?;
        let (confidence_plane, confidence_elements) = payload(CONFIDENCE_PAYLOAD)?;
        expect_elements(CONFIDENCE_PAYLOAD, confidence_elements, pixel_count)?;

        let pixels = decode_f32_plane(&pixel_plane, pixel_count * 3)
            .and_then(|values| Rgb32FImage::from_raw(meta.width, meta.height, values))
            .ok_or_else(|| {
                dimension_mismatch(format!(
                    "{PIXELS_PAYLOAD} holds {} byte(s) for a {}x{} tile",
                    pixel_plane.len(),
                    meta.width,
                    meta.height
                ))
            })?;
        let owners = decode_u16_plane(&ownership_plane, pixel_count).ok_or_else(|| {
            dimension_mismatch(format!(
                "{OWNERSHIP_PAYLOAD} holds {} byte(s) for a {}x{} tile",
                ownership_plane.len(),
                meta.width,
                meta.height
            ))
        })?;
        let covered = decode_coverage_bits(&coverage_plane, pixel_count).ok_or_else(|| {
            dimension_mismatch(format!(
                "{COVERAGE_PAYLOAD} holds {} byte(s) for a {}x{} tile",
                coverage_plane.len(),
                meta.width,
                meta.height
            ))
        })?;
        let confidence_values =
            decode_f16_plane(&confidence_plane, pixel_count).ok_or_else(|| {
                dimension_mismatch(format!(
                    "{CONFIDENCE_PAYLOAD} holds {} byte(s) for a {}x{} tile",
                    confidence_plane.len(),
                    meta.width,
                    meta.height
                ))
            })?;

        let legend = meta
            .owner_legend
            .iter()
            .map(PathBuf::from)
            .collect::<Vec<_>>();
        let ownership = OwnershipMap::new(meta.width, meta.height, owners, legend)
            .map_err(dimension_mismatch)?;
        let coverage = CoverageMask::from_bytes(meta.width, meta.height, covered)
            .map_err(dimension_mismatch)?;
        let confidence = ConfidenceMap::per_pixel(meta.width, meta.height, confidence_values)
            .map_err(dimension_mismatch)?;
        let provenance = meta
            .provenance
            .iter()
            .map(|record| {
                Ok(SourceProvenance {
                    absolute_path: PathBuf::from(&record.path),
                    sha256: match record.sha256.as_deref() {
                        None => None,
                        Some(value) => Some(parse_hex_digest(value).ok_or_else(|| {
                            (
                                degradation::CACHE_ENTRY_SHA_MISMATCH,
                                format!("provenance digest {value:?} is not a SHA-256"),
                            )
                        })?),
                    },
                    owned_pixels: record.owned_pixels,
                })
            })
            .collect::<Result<Vec<_>, (&'static str, String)>>()?;

        let coefficients = meta
            .tile_to_world_bits
            .iter()
            .map(|bits| f64::from_bits(*bits))
            .collect::<Vec<_>>();
        let tile_to_world = Matrix3::new(
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
            coefficients[4],
            coefficients[5],
            coefficients[6],
            coefficients[7],
            coefficients[8],
        );
        // Re-running the constructor means a cache hit is held to exactly the
        // same invariants as a fresh fusion, so no cached entry can smuggle a
        // tile past the checks of 需求 4.1 / 4.6 / 4.7.
        let tile = VirtualTile::new(
            meta.station_index,
            tile_to_world,
            pixels,
            ownership,
            confidence,
            coverage,
            color_encoding,
            provenance,
        )
        .map_err(dimension_mismatch)?;
        let recorded_bounds = tile
            .coverage
            .covered_bounds()
            .map(|(left, top, width, height)| CoverageBoundsMeta {
                left,
                top,
                width,
                height,
            });
        if recorded_bounds != meta.coverage_bounds {
            return Err(dimension_mismatch(
                "recorded coverage bounds disagree with the stored coverage mask".to_string(),
            ));
        }
        Ok(tile)
    }

    // -- store --------------------------------------------------------------

    /// Write `tile` under `cache_key`, atomically, then enforce the budget.
    ///
    /// Never returns an error: a cache that cannot be written is reported as
    /// [`StoreOutcome::Unavailable`] and the caller keeps its in-memory tile
    /// (需求 4.11).
    pub(crate) fn store(&self, inputs: &CacheKeyInputs, tile: &VirtualTile) -> StoreOutcome {
        let cache_key = inputs.cache_key();
        match self.store_atomically(inputs, &cache_key, tile) {
            Ok(bytes) => {
                let (evicted, total_bytes) = self.record_and_enforce(&cache_key, bytes);
                StoreOutcome::Written {
                    bytes,
                    evicted,
                    total_bytes,
                }
            }
            Err(detail) => StoreOutcome::Unavailable {
                reason: degradation::CACHE_WRITE_UNAVAILABLE,
                detail,
            },
        }
    }

    fn store_atomically(
        &self,
        inputs: &CacheKeyInputs,
        cache_key: &str,
        tile: &VirtualTile,
    ) -> Result<u64, String> {
        fs::create_dir_all(&self.root).map_err(|error| {
            format!(
                "Failed to create the virtual tile cache at {}: {error}",
                self.root.display()
            )
        })?;
        let temp = self.root.join(format!("{TEMP_PREFIX}{}", Uuid::new_v4()));
        let result = self.fill_temp_entry(inputs, cache_key, tile, &temp);
        let bytes = match result {
            Ok(bytes) => bytes,
            Err(error) => {
                let _ = fs::remove_dir_all(&temp);
                return Err(error);
            }
        };
        // Publish: the previous entry (if any) goes first so `rename` sees a
        // free name, and a reader either finds the old complete entry or the
        // new complete entry.
        let entry = self.entry_dir(cache_key);
        if entry.exists() {
            let _ = fs::remove_dir_all(&entry);
        }
        if let Err(error) = fs::rename(&temp, &entry) {
            let _ = fs::remove_dir_all(&temp);
            return Err(format!(
                "Failed to publish the virtual tile cache entry {cache_key}: {error}"
            ));
        }
        sync_directory(&self.root);
        Ok(bytes)
    }

    fn fill_temp_entry(
        &self,
        inputs: &CacheKeyInputs,
        cache_key: &str,
        tile: &VirtualTile,
        temp: &Path,
    ) -> Result<u64, String> {
        fs::create_dir_all(temp)
            .map_err(|error| format!("Failed to create {}: {error}", temp.display()))?;

        let pixel_count = u64::from(tile.width) * u64::from(tile.height);
        let pixel_plane = encode_f32_plane(tile.pixels.as_raw());
        let ownership_plane = encode_u16_plane(tile.ownership.owners());
        let coverage_plane = encode_coverage_bits(tile.coverage.covered());
        let confidence_plane = encode_f16_plane(&tile.sharpness_confidence.to_row_major());

        let mut payloads = BTreeMap::new();
        for (name, plane, elements) in [
            (PIXELS_PAYLOAD, &pixel_plane, pixel_count * 3),
            (OWNERSHIP_PAYLOAD, &ownership_plane, pixel_count),
            (COVERAGE_PAYLOAD, &coverage_plane, pixel_count),
            (CONFIDENCE_PAYLOAD, &confidence_plane, pixel_count),
        ] {
            let digest: [u8; 32] = Sha256::digest(plane).into();
            let bytes = write_payload(&temp.join(name), plane)?;
            payloads.insert(
                name.to_string(),
                PayloadMeta {
                    bytes,
                    elements,
                    sha256: hex_digest(&digest),
                },
            );
        }

        let meta = CacheEntryMeta {
            schema: META_SCHEMA,
            pipeline_version: inputs.pipeline_version.clone(),
            station_index: tile.station_index,
            width: tile.width,
            height: tile.height,
            tile_to_world_bits: tile
                .tile_to_world
                .transpose()
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            color_encoding: color_encoding_identifier(tile.color_encoding).to_string(),
            coverage_bounds: tile
                .coverage
                .covered_bounds()
                .map(|(left, top, width, height)| CoverageBoundsMeta {
                    left,
                    top,
                    width,
                    height,
                }),
            owner_legend: tile
                .ownership
                .legend()
                .iter()
                .map(|path| path.to_string_lossy().to_string())
                .collect(),
            provenance: tile
                .provenance
                .iter()
                .map(|record| ProvenanceMeta {
                    path: record.absolute_path.to_string_lossy().to_string(),
                    sha256: record.sha256.as_ref().map(hex_digest),
                    owned_pixels: record.owned_pixels,
                })
                .collect(),
            cache_key: cache_key.to_string(),
            cache_key_inputs: inputs.clone(),
            payloads,
        };
        let meta_bytes = serde_json::to_vec_pretty(&meta)
            .map_err(|error| format!("Failed to serialize meta.json: {error}"))?;
        write_file_sync(&temp.join(META_FILE_NAME), &meta_bytes)?;
        sync_directory(temp);

        let total = meta
            .payloads
            .values()
            .fold(0u64, |sum, payload| sum.saturating_add(payload.bytes))
            .saturating_add(meta_bytes.len() as u64);
        Ok(total)
    }

    // -- index and eviction -------------------------------------------------

    /// Refresh `last_access_epoch_ms` after a hit.  Best effort: a failed index
    /// write only risks a premature eviction, never the run.
    fn touch(&self, cache_key: &str) {
        let _guard = self.lock_index();
        let mut index = self.read_index();
        let bytes = index
            .entries
            .get(cache_key)
            .map(|entry| entry.bytes)
            .unwrap_or_else(|| directory_bytes(&self.entry_dir(cache_key)));
        index.entries.insert(
            cache_key.to_string(),
            IndexEntry {
                last_access_epoch_ms: epoch_millis(),
                bytes,
            },
        );
        let _ = self.write_index(&index);
    }

    /// Drop a corrupt entry from disk and from the index (需求 4.10).
    fn discard_entry(&self, cache_key: &str) {
        let _ = fs::remove_dir_all(self.entry_dir(cache_key));
        let _guard = self.lock_index();
        let mut index = self.read_index();
        if index.entries.remove(cache_key).is_some() {
            let _ = self.write_index(&index);
        }
    }

    /// Record the new entry and evict by ascending last access until the cache
    /// fits in [`Self::limit_bytes`] (需求 4.4).  Returns
    /// `(evicted_entries, total_bytes)`.
    fn record_and_enforce(&self, cache_key: &str, bytes: u64) -> (u64, u64) {
        let _guard = self.lock_index();
        let mut index = self.read_index();
        index.entries.insert(
            cache_key.to_string(),
            IndexEntry {
                last_access_epoch_ms: epoch_millis(),
                bytes,
            },
        );
        let mut evicted = 0u64;
        while index.total_bytes() > self.limit_bytes {
            // Oldest last access wins; the cache key breaks ties so the choice
            // never depends on map iteration order.  The entry just written is
            // held back until it is the only one left, because the millisecond
            // clock can tie it with an entry stored in the same millisecond and
            // evicting the newest first would thrash the cache.
            let victim = index
                .entries
                .iter()
                .filter(|(key, _)| key.as_str() != cache_key)
                .min_by(|left, right| {
                    left.1
                        .last_access_epoch_ms
                        .cmp(&right.1.last_access_epoch_ms)
                        .then_with(|| left.0.cmp(right.0))
                })
                .map(|(key, _)| key.clone())
                // A single entry larger than the whole budget is evicted too:
                // the limit is a ceiling, not a target.
                .or_else(|| {
                    index
                        .entries
                        .contains_key(cache_key)
                        .then(|| cache_key.to_string())
                });
            let Some(victim) = victim else {
                break;
            };
            // Whole entries only: a half deleted entry would read as corrupt
            // rather than as absent.
            let _ = fs::remove_dir_all(self.entry_dir(&victim));
            index.entries.remove(&victim);
            evicted += 1;
        }
        if evicted > 0 {
            degradation::record_run_degradation(
                degradation::CACHE_EVICTED_FOR_CAPACITY,
                serde_json::json!({
                    "evicted_entries": evicted,
                    "limit_bytes": self.limit_bytes,
                    "total_bytes": index.total_bytes(),
                }),
            );
        }
        let _ = self.write_index(&index);
        (evicted, index.total_bytes())
    }

    // -- residency leases ---------------------------------------------------

    /// Lease the Virtual_Tile of `station_index`, keeping at most
    /// [`MAX_RESIDENT_VIRTUAL_TILES`] full size tiles in memory (需求 14.4).
    ///
    /// `provide` is called only when the tile is not resident yet; a station
    /// that is leased twice is handed the same tile without a second synthesis.
    /// When the resident slots are full, the least recently leased tile that
    /// nobody holds is dropped first; if every slot has a live lease the call
    /// fails with [`LeaseError::ResidencyExhausted`] rather than exceeding the
    /// ceiling.
    ///
    /// The lease decides only *when* a tile is in memory, never what it
    /// contains: the same `provide` produces the same tile whether it is called
    /// after an eviction or on the first lease, so no pixel, threshold or
    /// fallback decision depends on the residency state.
    pub(crate) fn lease(
        &self,
        station_index: usize,
        provide: impl FnOnce() -> Result<VirtualTile, String>,
    ) -> Result<TileLease, LeaseError> {
        let tile = self.leases.acquire(station_index, provide)?;
        Ok(TileLease {
            station_index,
            tile,
            registry: Arc::clone(&self.leases),
        })
    }

    /// Tiles currently in memory, leased or kept for reuse.
    pub(crate) fn resident_virtual_tiles(&self) -> usize {
        self.leases.resident_count()
    }

    /// Live [`TileLease`] handles across every resident tile.
    pub(crate) fn active_virtual_tile_leases(&self) -> usize {
        self.leases.held_count()
    }

    /// Peak resident tile count of this store, for
    /// `resources.max_resident_virtual_tiles` (需求 14.4).
    pub(crate) fn max_resident_virtual_tiles(&self) -> usize {
        self.leases.max_resident()
    }

    /// On-disk directory of `cache_key`, for the property tests of stage 2.
    /// Read only: nothing in the test build may reach the layout through a
    /// second implementation of the path join.
    #[cfg(test)]
    pub(crate) fn entry_dir_for_test(&self, cache_key: &str) -> PathBuf {
        self.entry_dir(cache_key)
    }

    /// `index.json` as `(cache_key, last_access_epoch_ms, bytes)`, ascending by
    /// key.  Property 21 needs the recorded access times of the entries that
    /// survived an eviction pass, which are only knowable from the index.
    #[cfg(test)]
    pub(crate) fn index_snapshot_for_test(&self) -> Vec<(String, u64, u64)> {
        let _guard = self.lock_index();
        self.read_index()
            .entries
            .iter()
            .map(|(key, entry)| (key.clone(), entry.last_access_epoch_ms, entry.bytes))
            .collect()
    }
}

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

// ---------------------------------------------------------------------------
// Payload encoding
// ---------------------------------------------------------------------------

fn encode_f32_plane(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn decode_f32_plane(bytes: &[u8], elements: u64) -> Option<Vec<f32>> {
    let elements = usize::try_from(elements).ok()?;
    if bytes.len() != elements * 4 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            })
            .collect(),
    )
}

fn encode_u16_plane(values: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_u16_plane(bytes: &[u8], elements: u64) -> Option<Vec<u16>> {
    let elements = usize::try_from(elements).ok()?;
    if bytes.len() != elements * 2 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect(),
    )
}

/// `f16` little endian, one value per pixel.  This is the one plane the design
/// stores narrower than it is held in memory; the stored value round trips
/// exactly, and confidence feeds no threshold that `f16` spacing can move.
fn encode_f16_plane(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
    }
    bytes
}

fn decode_f16_plane(bytes: &[u8], elements: u64) -> Option<Vec<f32>> {
    let elements = usize::try_from(elements).ok()?;
    if bytes.len() != elements * 2 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
    )
}

/// Coverage as one bit per pixel, most significant bit first inside a byte, so
/// the plane is independent of the host byte order.
fn encode_coverage_bits(covered: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0u8; covered.len().div_ceil(8)];
    for (index, &value) in covered.iter().enumerate() {
        if value > 0 {
            bytes[index / 8] |= 0x80 >> (index % 8);
        }
    }
    bytes
}

fn decode_coverage_bits(bytes: &[u8], elements: u64) -> Option<Vec<u8>> {
    let elements = usize::try_from(elements).ok()?;
    if bytes.len() != elements.div_ceil(8) {
        return None;
    }
    let mut covered = vec![0u8; elements];
    for (index, slot) in covered.iter_mut().enumerate() {
        if bytes[index / 8] & (0x80 >> (index % 8)) != 0 {
            *slot = u8::MAX;
        }
    }
    Some(covered)
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

/// Write `plane` zstd compressed and `fsync` it.  Returns the file size.
fn write_payload(path: &Path, plane: &[u8]) -> Result<u64, String> {
    let compressed = zstd::bulk::compress(plane, ZSTD_LEVEL)
        .map_err(|error| format!("Failed to compress {}: {error}", path.display()))?;
    write_file_sync(path, &compressed)
}

/// Read a zstd payload back to its raw plane.
fn read_payload(path: &Path) -> Result<Vec<u8>, String> {
    let compressed =
        fs::read(path).map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    // `zstd` records the decompressed size in the frame header, so no capacity
    // has to be guessed; a truncated frame fails here rather than silently
    // returning a short plane.
    zstd::stream::decode_all(compressed.as_slice())
        .map_err(|error| format!("Failed to decompress {}: {error}", path.display()))
}

fn write_file_sync(path: &Path, bytes: &[u8]) -> Result<u64, String> {
    let mut file = File::create(path)
        .map_err(|error| format!("Failed to create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("Failed to write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("Failed to flush {}: {error}", path.display()))?;
    Ok(bytes.len() as u64)
}

/// `fsync` a directory so a rename that follows is durable.  Best effort:
/// Windows cannot open a directory as a file, and a missing barrier costs
/// durability across a power loss, not correctness within a run.
fn sync_directory(path: &Path) {
    #[cfg(unix)]
    if let Ok(handle) = File::open(path) {
        let _ = handle.sync_all();
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// Total file size below `path`, used when the index lost track of an entry.
fn directory_bytes(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|entry| entry.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .fold(0u64, |total, metadata| total.saturating_add(metadata.len()))
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
