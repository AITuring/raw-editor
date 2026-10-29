//! The on-disk meta.json and index.json records of the Virtual_Tile cache.
//!
//! Strict key-set verification of meta.json, colour-encoding and dimension
//! checks, and the last-access index used for capacity eviction.

use super::super::degradation;
use crate::panorama_utils::stitching::ColorEncoding;
use serde::Deserialize;
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use super::{CacheKeyInputs, INDEX_SCHEMA};

// ---------------------------------------------------------------------------
// meta.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct PayloadMeta {
    /// Size of the compressed file on disk, the unit of the 64 GiB budget.
    pub(super) bytes: u64,
    /// Number of decompressed elements; the dimension cross-check of 需求 4.10.
    pub(super) elements: u64,
    /// SHA-256 of the **decompressed** plane, so the check attests the payload
    /// content rather than a particular compressor build.
    pub(super) sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct CoverageBoundsMeta {
    pub(super) left: u32,
    pub(super) top: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ProvenanceMeta {
    pub(super) path: String,
    /// `None` when the digest was not available at fusion time; the field is
    /// always present in the document so a truly missing key is detectable.
    pub(super) sha256: Option<String>,
    pub(super) owned_pixels: u64,
}

/// `meta.json`.  No field carries `#[serde(default)]`, so an absent key fails
/// the parse for every non-`Option` field.  `Option` fields are the exception
/// `serde` cannot cover — an absent key deserialises to `None` without an error
/// — so the document is additionally checked against the declared key sets of
/// [`verify_meta_keys`] before it is trusted (需求 4.10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct CacheEntryMeta {
    pub(super) schema: u32,
    pub(super) pipeline_version: String,
    pub(super) station_index: usize,
    pub(super) width: u32,
    pub(super) height: u32,
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
    pub(super) tile_to_world_bits: Vec<u64>,
    pub(super) color_encoding: String,
    pub(super) coverage_bounds: Option<CoverageBoundsMeta>,
    pub(super) owner_legend: Vec<String>,
    pub(super) provenance: Vec<ProvenanceMeta>,
    pub(super) cache_key: String,
    /// The three ingredients, stored for the item by item re-check of 需求 4.9.
    pub(super) cache_key_inputs: CacheKeyInputs,
    pub(super) payloads: BTreeMap<String, PayloadMeta>,
}

/// The keys [`CacheEntryMeta`] declares at the document root.
pub(super) const META_KEYS: &[&str] = &[
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
pub(super) const CACHE_KEY_INPUTS_KEYS: &[&str] =
    &["pipeline_version", "sorted_paths", "sorted_source_sha256"];

/// The keys [`CoverageBoundsMeta`] declares, under `coverage_bounds`.
pub(super) const COVERAGE_BOUNDS_KEYS: &[&str] = &["left", "top", "width", "height"];

/// The keys [`ProvenanceMeta`] declares, per `provenance` element.
pub(super) const PROVENANCE_KEYS: &[&str] = &["path", "sha256", "owned_pixels"];

/// The keys [`PayloadMeta`] declares, per `payloads` value.
pub(super) const PAYLOAD_KEYS: &[&str] = &["bytes", "elements", "sha256"];

/// Require every key of `expected` on the object at `location`.
///
/// Presence only: a key whose value is `null` counts as present, because that
/// is how the two nullable fields (`coverage_bounds` and a provenance
/// `sha256`) legitimately encode "not available".  A key that is *gone* is the
/// corruption of 需求 4.10 and must be reported even for those fields.
pub(super) fn expect_meta_keys(
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
pub(super) fn verify_meta_keys(meta_bytes: &[u8]) -> Result<(), String> {
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

pub(super) fn color_encoding_identifier(encoding: ColorEncoding) -> &'static str {
    match encoding {
        ColorEncoding::LinearSrgb => "linear_srgb",
        ColorEncoding::DisplaySrgb => "display_srgb",
    }
}

/// Wrap a detail string as the "recorded size disagrees with the payload"
/// verification failure of 需求 4.10.
pub(super) fn dimension_mismatch(detail: String) -> (&'static str, String) {
    (degradation::CACHE_ENTRY_DIMENSION_MISMATCH, detail)
}

pub(super) fn parse_color_encoding(value: &str) -> Option<ColorEncoding> {
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
pub(super) struct IndexEntry {
    pub(super) last_access_epoch_ms: u64,
    pub(super) bytes: u64,
}

/// `index.json`.  A `BTreeMap` rather than a `HashMap` so serialisation and
/// eviction tie-breaks are ordered by key (需求 14.6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct CacheIndex {
    pub(super) schema: u32,
    pub(super) entries: BTreeMap<String, IndexEntry>,
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
    pub(super) fn total_bytes(&self) -> u64 {
        self.entries
            .values()
            .fold(0u64, |total, entry| total.saturating_add(entry.bytes))
    }
}

pub(super) fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}
