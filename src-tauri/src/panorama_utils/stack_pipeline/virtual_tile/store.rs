//! The Virtual_Tile_Store: lookup, atomic store, verified load and eviction.
//!
//! One implementation block by design; it owns the cache root, the index and
//! the lease registry, and never returns an error that could abort a run.

use super::super::degradation;
use super::super::report::VIRTUAL_TILE_CACHE_LIMIT_BYTES;
use crate::panorama_utils::stitching::ConfidenceMap;
use crate::panorama_utils::stitching::CoverageMask;
use crate::panorama_utils::stitching::OwnershipMap;
use crate::panorama_utils::stitching::SourceProvenance;
use crate::panorama_utils::stitching::VirtualTile;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use uuid::Uuid;

use super::{
    CONFIDENCE_PAYLOAD, COVERAGE_PAYLOAD, CacheEntryMeta, CacheIndex, CacheKeyInputs,
    CoverageBoundsMeta, INDEX_FILE_NAME, INDEX_SCHEMA, IndexEntry, LeaseError, LeaseRegistry,
    MAX_RESIDENT_VIRTUAL_TILES, META_FILE_NAME, META_SCHEMA, OWNERSHIP_PAYLOAD, PIXELS_PAYLOAD,
    PayloadMeta, ProvenanceMeta, REQUIRED_PAYLOADS, TEMP_PREFIX, TileLease,
    VIRTUAL_TILE_CACHE_DIR_NAME, VIRTUAL_TILE_CACHE_LAYOUT_VERSION, color_encoding_identifier,
    decode_coverage_bits, decode_f16_plane, decode_f32_plane, decode_u16_plane, dimension_mismatch,
    directory_bytes, encode_coverage_bits, encode_f16_plane, encode_f32_plane, encode_u16_plane,
    epoch_millis, hex_digest, parse_color_encoding, parse_hex_digest, read_payload, sync_directory,
    verify_meta_keys, write_file_sync, write_payload,
};
use sha2::Digest;

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
// The store
// ---------------------------------------------------------------------------

/// Versioned, lossless Virtual_Tile cache.
///
/// `index_lock` serialises the read-modify-write of `index.json`.  A run fuses
/// its stations one after another today, but the compositor may well fuse two
/// in parallel later, and a lost index update would silently leak disk.
pub(crate) struct VirtualTileStore {
    pub(super) root: PathBuf,
    pub(super) limit_bytes: u64,
    pub(super) index_lock: Mutex<()>,
    pub(super) leases: Arc<LeaseRegistry>,
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

    pub(super) fn entry_dir(&self, cache_key: &str) -> PathBuf {
        self.root.join(cache_key)
    }

    pub(super) fn index_path(&self) -> PathBuf {
        self.root.join(INDEX_FILE_NAME)
    }

    /// Current cache size according to `index.json`.
    pub(crate) fn total_bytes(&self) -> u64 {
        let _guard = self.lock_index();
        self.read_index().total_bytes()
    }

    pub(super) fn lock_index(&self) -> std::sync::MutexGuard<'_, ()> {
        self.index_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `index.json`, or an empty index when it is absent or unreadable.  A
    /// broken index costs cache hits, never a run (需求 4.11).
    pub(super) fn read_index(&self) -> CacheIndex {
        fs::read(self.index_path())
            .ok()
            .and_then(|bytes| serde_json::from_slice::<CacheIndex>(&bytes).ok())
            .filter(|index| index.schema == INDEX_SCHEMA)
            .unwrap_or_default()
    }

    pub(super) fn write_index(&self, index: &CacheIndex) -> Result<(), String> {
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

    pub(super) fn load_verified(
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

    pub(super) fn store_atomically(
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

    pub(super) fn fill_temp_entry(
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
    pub(super) fn touch(&self, cache_key: &str) {
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
    pub(super) fn discard_entry(&self, cache_key: &str) {
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
    pub(super) fn record_and_enforce(&self, cache_key: &str, bytes: u64) -> (u64, u64) {
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
