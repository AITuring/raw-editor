//! Read-only source-locked retouching sidecar parser, SHA-256 verifier,
//! staleness reconciler, constraint resolver, and export binding auditor
//! (Task 9 / Task 20 Phase 3).
//!
//! Implements the Rust counterpart to `src/utils/sourceLockSidecar.ts` and
//! `docs/artwork-lanes/9-source-lock-design.md`.
//!
//! Guarantees:
//! - Read-only: never mutates RGB pixels, TIFF outputs, sidecar files, or the
//!   23-field top-level `Stack_Report` schema.
//! - Deterministic: uses sorted `BTreeMap`/`BTreeSet` structures and canonical
//!   JSON serialization matching ECMAScript `JSON.stringify` so SHA-256 digests
//!   match TypeScript bit-for-bit.
//! - Bounded I/O: reads sidecar JSON with an explicit byte limit and streams
//!   native TIFF files through a fixed 64 KiB buffer via read-only `File::open`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const SOURCE_LOCK_SCHEMA_VERSION: u32 = 1;
pub const SOURCE_LOCK_COORDINATE_UNITS: &str = "native_output_pixels";
/// Upper bound on sidecar JSON / JSONL payloads accepted by the read-only parser.
pub const MAX_SOURCE_LOCK_SIDECAR_BYTES: u64 = 8 * 1024 * 1024;
const STREAMING_HASH_BUFFER_BYTES: usize = 64 * 1024;
const MAX_SAFE_JS_INTEGER: f64 = 9_007_199_254_740_991.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceLockPolicy {
    SingleSource,
    AllowedSources,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceLockStatus {
    Active,
    Stale,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceLockPriority {
    StaleOrUnresolved,
    SingleSource,
    AllowedSources,
    Auto,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordinateFrame {
    pub id: String,
    pub canvas: [f64; 2],
    pub origin: [f64; 2],
    pub units: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceBinding {
    pub source_id: u64,
    pub path: String,
    pub sha256: String,
    pub station_id: u64,
    pub capture_group_id: u64,
    pub source_to_world_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldRegion {
    pub polygon_world: Vec<[f64; 2]>,
    pub bbox: [f64; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceLock {
    pub lock_id: String,
    pub status: SourceLockStatus,
    pub region: WorldRegion,
    pub requested_sources: Vec<u64>,
    pub policy: SourceLockPolicy,
    pub eligible_tiles: u64,
    pub selection_digest: String,
    pub created_event: u64,
    pub stale_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManualReviewRegion {
    pub region_id: String,
    pub polygon_world: Vec<[f64; 2]>,
    pub reason: String,
    pub excluded_from_quality_gate: bool,
    pub created_event: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceLockEventKind {
    AddSourceLock,
    RemoveSourceLock,
    AddManualReviewRegion,
    RemoveManualReviewRegion,
    MarkLockStale,
    Undo,
    Redo,
}

/// Preserves the three-state JSON distinction between an omitted `payload`
/// field (`undefined` in TypeScript), `"payload": null`, and an object value
/// so canonical SHA-256 digests match `sourceLockSidecar.ts` bit-for-bit.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum EventPayloadField {
    #[default]
    Absent,
    Null,
    Present(Value),
}

impl EventPayloadField {
    fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }
}

impl Serialize for EventPayloadField {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Absent | Self::Null => serializer.serialize_unit(),
            Self::Present(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for EventPayloadField {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        if value.is_null() {
            Ok(Self::Null)
        } else {
            Ok(Self::Present(value))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceLockEvent {
    pub event_id: u64,
    pub kind: SourceLockEventKind,
    pub inverse_kind: SourceLockEventKind,
    pub before_digest: String,
    pub after_digest: String,
    pub coordinate_frame_id: String,
    pub target_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverted_event_id: Option<u64>,
    #[serde(default, skip_serializing_if = "EventPayloadField::is_absent")]
    pub payload: EventPayloadField,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceLockHistory {
    pub head: u64,
    pub events: Vec<SourceLockEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceLockDocument {
    pub schema: u32,
    pub run_id: String,
    pub input_manifest_sha256: String,
    pub coordinate_frame: CoordinateFrame,
    pub source_bindings: Vec<SourceBinding>,
    pub locks: Vec<SourceLock>,
    pub manual_review_regions: Vec<ManualReviewRegion>,
    pub history: SourceLockHistory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurrentRunBinding {
    pub run_id: String,
    pub input_manifest_sha256: String,
    pub coordinate_frame: CoordinateFrame,
    pub source_bindings: Vec<SourceBinding>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SidecarReconciliationReport {
    pub doc: SourceLockDocument,
    pub newly_stale_count: usize,
    pub mismatch_reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorldPointConstraint {
    Auto,
    AllowedSources {
        source_ids: Vec<u64>,
        lock_ids: Vec<String>,
    },
    SingleSource {
        source_id: u64,
        lock_ids: Vec<String>,
    },
    BlockedByStaleOrUnresolved {
        lock_ids: Vec<String>,
    },
    Conflict {
        lock_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityGateTileObservation {
    pub tile_id: String,
    pub failed: bool,
    pub manual_region_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcludedQualityGateTile {
    pub tile_id: String,
    pub region_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManualReviewQualityGateVerdict {
    Pass,
    Fail,
    InsufficientEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManualReviewQualityGateAccounting {
    pub all_tiles: usize,
    pub auto_eligible_tiles: usize,
    pub excluded_tiles: Vec<ExcludedQualityGateTile>,
    pub failed_all_tiles: usize,
    pub failed_auto_eligible_tiles: usize,
    pub verdict: ManualReviewQualityGateVerdict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportBindingStatus {
    Valid,
    StaleExport,
    BindingMismatch,
    HasStaleOrUnresolvedLocks,
    EventsLogMismatch,
    IntegrityError,
}

/// Read-only reference summary for `Stack_Report` export audits (Task 9).
/// Computed purely on demand without altering the 23-field top-level
/// `Stack_Report` schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackReportSidecarReferenceSummary {
    pub run_id: String,
    pub sidecar_sha256: String,
    pub tiff_sha256: String,
    pub active_locks: usize,
    pub active_lock_area_px: u64,
    pub stale_or_unresolved_locks: usize,
    pub stale_or_unresolved_area_px: u64,
    pub manual_review_regions: usize,
    pub manual_review_area_px: u64,
    pub locked_eligible_tiles: u64,
    pub history_head: u64,
    pub event_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportBindingAudit {
    pub status: ExportBindingStatus,
    pub can_publish_as_confirmed: bool,
    pub sidecar_sha256: String,
    pub tiff_sha256: String,
    pub summary: StackReportSidecarReferenceSummary,
    pub reconciliation: Option<SidecarReconciliationReport>,
    pub reasons: Vec<String>,
}

pub fn sha256_hex_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Compute a file's SHA-256 hex digest using a read-only `File::open` handle
/// and bounded 64 KiB streaming buffer so multi-gigabyte native TIFF exports
/// never load into memory.
pub fn sha256_file_hex(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| {
        format!(
            "Failed to open '{}' for SHA-256 verification: {error}",
            path.display()
        )
    })?;
    let mut reader = BufReader::with_capacity(STREAMING_HASH_BUFFER_BYTES, file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; STREAMING_HASH_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer).map_err(|error| {
            format!(
                "Failed to read '{}' while computing SHA-256: {error}",
                path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn read_bounded_utf8_file(path: &Path, label: &str) -> Result<String, String> {
    let file = File::open(path)
        .map_err(|error| format!("Failed to open {label} '{}': {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("Failed to inspect {label} '{}': {error}", path.display()))?;
    if metadata.len() > MAX_SOURCE_LOCK_SIDECAR_BYTES {
        return Err(format!(
            "{label} '{}' exceeds maximum supported size ({} > {} bytes)",
            path.display(),
            metadata.len(),
            MAX_SOURCE_LOCK_SIDECAR_BYTES
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_SOURCE_LOCK_SIDECAR_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Failed to read {label} '{}': {error}", path.display()))?;
    if bytes.len() as u64 > MAX_SOURCE_LOCK_SIDECAR_BYTES {
        return Err(format!(
            "{label} '{}' exceeds maximum supported size ({} bytes)",
            path.display(),
            MAX_SOURCE_LOCK_SIDECAR_BYTES
        ));
    }
    String::from_utf8(bytes)
        .map_err(|error| format!("{label} '{}' is not valid UTF-8: {error}", path.display()))
}

fn write_canonical_json(value: &Value, output: &mut String) -> Result<(), String> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(flag) => output.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => {
            if let Some(signed) = number.as_i64() {
                output.push_str(&signed.to_string());
            } else if let Some(unsigned) = number.as_u64() {
                output.push_str(&unsigned.to_string());
            } else if let Some(float) = number.as_f64() {
                if !float.is_finite() {
                    return Err("Non-finite number cannot be serialized to canonical JSON".into());
                }
                if float == 0.0 {
                    output.push('0');
                } else if float.fract() == 0.0
                    && (-MAX_SAFE_JS_INTEGER..=MAX_SAFE_JS_INTEGER).contains(&float)
                {
                    output.push_str(&(float as i64).to_string());
                } else {
                    let encoded = serde_json::to_string(number).map_err(|error| {
                        format!("Failed to format canonical JSON number: {error}")
                    })?;
                    output.push_str(&encoded);
                }
            } else {
                return Err("Unsupported JSON number representation".into());
            }
        }
        Value::String(text) => {
            let encoded = serde_json::to_string(text)
                .map_err(|error| format!("Failed to format canonical JSON string: {error}"))?;
            output.push_str(&encoded);
        }
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical_json(item, output)?;
            }
            output.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by_key(|(key, _)| *key);
            output.push('{');
            for (index, (key, entry_value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                let encoded_key = serde_json::to_string(key)
                    .map_err(|error| format!("Failed to format canonical JSON key: {error}"))?;
                output.push_str(&encoded_key);
                output.push(':');
                write_canonical_json(entry_value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

/// Produce deterministic key-sorted compact JSON matching ECMAScript
/// `JSON.stringify` number formatting for cross-language SHA-256 parity.
pub fn canonical_json_string(value: &Value) -> Result<String, String> {
    let mut output = String::new();
    write_canonical_json(value, &mut output)?;
    Ok(output)
}

fn canonical_sha256_of<T: Serialize>(value: &T) -> Result<String, String> {
    let json_value = serde_json::to_value(value)
        .map_err(|error| format!("Failed to convert sidecar value to JSON: {error}"))?;
    let canonical = canonical_json_string(&json_value)?;
    Ok(sha256_hex_bytes(canonical.as_bytes()))
}

fn validate_polygon_and_bbox(polygon: &[[f64; 2]]) -> Result<[f64; 4], String> {
    if polygon.len() < 3 {
        return Err("World region polygon requires at least 3 vertices.".into());
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &[x, y] in polygon {
        if !x.is_finite() || !y.is_finite() {
            return Err("World region polygon coordinates must be finite numbers.".into());
        }
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    Ok([
        min_x,
        min_y,
        (max_x - min_x).max(0.0),
        (max_y - min_y).max(0.0),
    ])
}

pub fn compute_region_pixel_area(polygon: &[[f64; 2]]) -> u64 {
    if polygon.len() < 3 {
        return 0;
    }
    let mut area_twice = 0.0_f64;
    for (index, &[x0, y0]) in polygon.iter().enumerate() {
        let [x1, y1] = polygon[(index + 1) % polygon.len()];
        area_twice += x0 * y1 - x1 * y0;
    }
    (area_twice.abs() * 0.5).round().max(0.0) as u64
}

fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Strictly parse and normalize a schema-v1 `source-locks.json` sidecar string.
pub fn parse_source_lock_document_str(raw_json: &str) -> Result<SourceLockDocument, String> {
    if raw_json.len() as u64 > MAX_SOURCE_LOCK_SIDECAR_BYTES {
        return Err(format!(
            "Source-lock sidecar exceeds maximum supported size ({} bytes).",
            MAX_SOURCE_LOCK_SIDECAR_BYTES
        ));
    }
    let mut doc: SourceLockDocument = serde_json::from_str(raw_json)
        .map_err(|error| format!("Invalid source-lock sidecar JSON: {error}"))?;

    if doc.schema != SOURCE_LOCK_SCHEMA_VERSION {
        return Err(format!(
            "Unsupported source-lock sidecar schema {} (expected {}).",
            doc.schema, SOURCE_LOCK_SCHEMA_VERSION
        ));
    }
    if doc.run_id.trim().is_empty() {
        return Err("Source-lock sidecar is missing a non-empty run_id.".into());
    }
    if doc.input_manifest_sha256.trim().is_empty() {
        return Err("Source-lock sidecar is missing input_manifest_sha256.".into());
    }
    if doc.coordinate_frame.id.trim().is_empty()
        || !doc.coordinate_frame.canvas[0].is_finite()
        || !doc.coordinate_frame.canvas[1].is_finite()
        || doc.coordinate_frame.canvas[0] <= 0.0
        || doc.coordinate_frame.canvas[1] <= 0.0
        || !doc.coordinate_frame.origin[0].is_finite()
        || !doc.coordinate_frame.origin[1].is_finite()
        || doc.coordinate_frame.units != SOURCE_LOCK_COORDINATE_UNITS
    {
        return Err("Source-lock sidecar has an invalid coordinate_frame.".into());
    }

    let mut seen_sources = BTreeSet::new();
    for binding in &doc.source_bindings {
        if binding.source_id == 0 {
            return Err("Source-lock binding source_id must be positive.".into());
        }
        if !seen_sources.insert(binding.source_id) {
            return Err(format!(
                "Duplicate source_id {} in source_bindings.",
                binding.source_id
            ));
        }
    }
    doc.source_bindings.sort_by_key(|binding| binding.source_id);

    for lock in &mut doc.locks {
        if lock.lock_id.trim().is_empty() {
            return Err("Source lock is missing lock_id.".into());
        }
        let expected_bbox = validate_polygon_and_bbox(&lock.region.polygon_world)?;
        lock.region.bbox = expected_bbox;
        if lock.requested_sources.contains(&0) {
            return Err(format!(
                "Source lock '{}' contains invalid source_id 0.",
                lock.lock_id
            ));
        }
        let unique_sorted: BTreeSet<u64> = lock.requested_sources.iter().copied().collect();
        lock.requested_sources = unique_sorted.into_iter().collect();
    }

    for region in &doc.manual_review_regions {
        if region.region_id.trim().is_empty() {
            return Err("Manual review region is missing region_id.".into());
        }
        validate_polygon_and_bbox(&region.polygon_world)?;
        if region.reason.trim().is_empty() {
            return Err(format!(
                "Manual review region '{}' requires a non-empty reason.",
                region.region_id
            ));
        }
        if !region.excluded_from_quality_gate {
            return Err(format!(
                "Manual review region '{}' must set excluded_from_quality_gate=true.",
                region.region_id
            ));
        }
    }

    let mut prev_event_id = 0_u64;
    for event in &doc.history.events {
        if event.event_id <= prev_event_id {
            return Err(format!(
                "History event_id {} is not strictly monotonic (previous {}).",
                event.event_id, prev_event_id
            ));
        }
        prev_event_id = event.event_id;
    }
    if prev_event_id > doc.history.head {
        return Err(format!(
            "History head {} precedes latest event_id {}.",
            doc.history.head, prev_event_id
        ));
    }

    Ok(doc)
}

/// Read and strictly parse a schema-v1 `source-locks.json` file from disk.
pub fn read_source_lock_document(path: &Path) -> Result<SourceLockDocument, String> {
    let raw = read_bounded_utf8_file(path, "source-lock sidecar")?;
    parse_source_lock_document_str(&raw)
}

/// Compute deterministic SHA-256 selection digest over sorted source IDs,
/// world polygon, coverage/owner tile digest, source-to-world fingerprints,
/// and policy. RGB pixel data is intentionally excluded.
pub fn compute_selection_digest(
    requested_sources: &[u64],
    region: &WorldRegion,
    coverage_owner_tile_digest: &str,
    source_bindings: &[SourceBinding],
    policy: SourceLockPolicy,
    coordinate_frame_id: &str,
) -> Result<String, String> {
    let sorted_ids: Vec<u64> = requested_sources
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let by_id: BTreeMap<u64, &SourceBinding> = source_bindings
        .iter()
        .map(|binding| (binding.source_id, binding))
        .collect();
    let fingerprints: Vec<Value> = sorted_ids
        .iter()
        .map(|&source_id| {
            let binding = by_id.get(&source_id);
            serde_json::json!({
                "source_id": source_id,
                "sha256": binding.map(|entry| entry.sha256.as_str()).unwrap_or(""),
                "source_to_world_sha256": binding
                    .map(|entry| entry.source_to_world_sha256.as_str())
                    .unwrap_or(""),
            })
        })
        .collect();

    #[derive(Serialize)]
    struct SelectionPayload<'a> {
        coordinate_frame_id: &'a str,
        coverage_owner_tile_digest: &'a str,
        policy: SourceLockPolicy,
        polygon_world: &'a [[f64; 2]],
        requested_sources: &'a [u64],
        source_fingerprints: Vec<Value>,
    }

    canonical_sha256_of(&SelectionPayload {
        coordinate_frame_id,
        coverage_owner_tile_digest,
        policy,
        polygon_world: &region.polygon_world,
        requested_sources: &sorted_ids,
        source_fingerprints: fingerprints,
    })
}

pub fn compute_state_digest(
    locks: &[SourceLock],
    manual_review_regions: &[ManualReviewRegion],
) -> Result<String, String> {
    #[derive(Serialize)]
    struct StatePayload<'a> {
        locks: &'a [SourceLock],
        manual_review_regions: &'a [ManualReviewRegion],
    }

    canonical_sha256_of(&StatePayload {
        locks,
        manual_review_regions,
    })
}

pub fn compute_sidecar_digest(doc: &SourceLockDocument) -> Result<String, String> {
    canonical_sha256_of(doc)
}

/// Deep integrity audit of a parsed sidecar: source-manifest references,
/// lock policy arity, unique identifiers, SHA-256 format, and monotonic event
/// digest chaining.
pub fn validate_source_lock_integrity(doc: &SourceLockDocument) -> Vec<String> {
    let mut errors = Vec::new();
    let bindings_by_id: BTreeMap<u64, &SourceBinding> = doc
        .source_bindings
        .iter()
        .map(|binding| (binding.source_id, binding))
        .collect();

    let mut seen_lock_ids = BTreeSet::new();
    for lock in &doc.locks {
        if !seen_lock_ids.insert(lock.lock_id.as_str()) {
            errors.push(format!("duplicate_lock_id:{}", lock.lock_id));
        }
        if lock.requested_sources.is_empty() {
            errors.push(format!("no_requested_sources:{}", lock.lock_id));
        }
        if lock.policy == SourceLockPolicy::SingleSource && lock.requested_sources.len() != 1 {
            errors.push(format!(
                "single_source_policy_requires_exactly_one_source:{}",
                lock.lock_id
            ));
        }
        if !is_valid_sha256_hex(&lock.selection_digest) {
            errors.push(format!("invalid_selection_digest:{}", lock.lock_id));
        }
        for &source_id in &lock.requested_sources {
            match bindings_by_id.get(&source_id) {
                None => errors.push(format!("unregistered_source_id:{source_id}")),
                Some(binding) => {
                    if binding.path.trim().is_empty() {
                        errors.push(format!("missing_source_path:{source_id}"));
                    }
                    if binding.sha256.trim().is_empty() {
                        errors.push(format!("missing_source_sha256:{source_id}"));
                    }
                    if binding.source_to_world_sha256.trim().is_empty() {
                        errors.push(format!("missing_source_to_world:{source_id}"));
                    }
                }
            }
        }
    }

    let mut seen_region_ids = BTreeSet::new();
    for region in &doc.manual_review_regions {
        if !seen_region_ids.insert(region.region_id.as_str()) {
            errors.push(format!(
                "duplicate_manual_review_region:{}",
                region.region_id
            ));
        }
    }

    let empty_state_digest = compute_state_digest(&[], &[]).ok();
    for (index, event) in doc.history.events.iter().enumerate() {
        if event.coordinate_frame_id != doc.coordinate_frame.id {
            errors.push(format!(
                "event_coordinate_frame_mismatch:{}",
                event.event_id
            ));
        }
        if !is_valid_sha256_hex(&event.before_digest) || !is_valid_sha256_hex(&event.after_digest) {
            errors.push(format!("invalid_event_digest:{}", event.event_id));
        }
        if index == 0
            && event.event_id == 1
            && let Some(ref expected_initial) = empty_state_digest
            && event.before_digest != *expected_initial
        {
            errors.push(format!(
                "initial_event_before_digest_mismatch:{}",
                event.event_id
            ));
        }
        if index > 0 {
            let previous = &doc.history.events[index - 1];
            if event.before_digest != previous.after_digest {
                errors.push(format!(
                    "event_digest_chain_broken:{}->{}",
                    previous.event_id, event.event_id
                ));
            }
        }
    }

    let all_locks_active = doc
        .locks
        .iter()
        .all(|lock| lock.status == SourceLockStatus::Active);
    if all_locks_active
        && let Some(last_event) = doc.history.events.last()
        && let Ok(current_state_digest) =
            compute_state_digest(&doc.locks, &doc.manual_review_regions)
        && last_event.after_digest != current_state_digest
    {
        errors.push("head_state_digest_mismatch".into());
    }

    errors
}

/// Verify that `source-locks.events.jsonl` matches `doc.history.events`
/// line-for-line in order, identifiers, kinds, digests, and payloads.
pub fn verify_events_jsonl_str(doc: &SourceLockDocument, events_jsonl: &str) -> Result<(), String> {
    if events_jsonl.len() as u64 > MAX_SOURCE_LOCK_SIDECAR_BYTES {
        return Err(format!(
            "source-locks.events.jsonl exceeds maximum supported size ({} bytes).",
            MAX_SOURCE_LOCK_SIDECAR_BYTES
        ));
    }
    let parsed_events: Vec<SourceLockEvent> = events_jsonl
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line_index, line)| {
            serde_json::from_str::<SourceLockEvent>(line).map_err(|error| {
                format!(
                    "Invalid JSON in source-locks.events.jsonl line {}: {error}",
                    line_index + 1
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if parsed_events.len() != doc.history.events.len() {
        return Err(format!(
            "source-locks.events.jsonl has {} event(s), but sidecar history has {}.",
            parsed_events.len(),
            doc.history.events.len()
        ));
    }

    for (index, (jsonl_event, doc_event)) in parsed_events
        .iter()
        .zip(doc.history.events.iter())
        .enumerate()
    {
        let jsonl_digest = canonical_sha256_of(jsonl_event)?;
        let doc_digest = canonical_sha256_of(doc_event)?;
        if jsonl_digest != doc_digest {
            return Err(format!(
                "source-locks.events.jsonl entry {} (event_id {}) does not match sidecar history.",
                index + 1,
                doc_event.event_id
            ));
        }
    }

    Ok(())
}

/// Reconcile every active lock against the current run's `run_id`,
/// `input_manifest_sha256`, `coordinate_frame`, and `source_bindings`.
/// Affected locks are marked `stale` and never silently continue applying.
pub fn reconcile_document_staleness(
    doc: &SourceLockDocument,
    current: &CurrentRunBinding,
) -> SidecarReconciliationReport {
    let mut reconciled = doc.clone();
    let mut newly_stale_count = 0_usize;
    let mut mismatch_reasons: Vec<String> = Vec::new();

    if reconciled.run_id != current.run_id {
        mismatch_reasons.push(format!(
            "run_id_changed:{}->{}",
            reconciled.run_id, current.run_id
        ));
        for lock in &mut reconciled.locks {
            if lock.status == SourceLockStatus::Active {
                lock.status = SourceLockStatus::Stale;
                lock.stale_reason = Some("run_id_changed".into());
                newly_stale_count += 1;
            }
        }
        return SidecarReconciliationReport {
            doc: reconciled,
            newly_stale_count,
            mismatch_reasons,
        };
    }

    let manifest_changed = current.input_manifest_sha256 != reconciled.input_manifest_sha256;
    let frame_changed = current.coordinate_frame.id != reconciled.coordinate_frame.id
        || current.coordinate_frame.units != reconciled.coordinate_frame.units
        || current.coordinate_frame.canvas != reconciled.coordinate_frame.canvas
        || current.coordinate_frame.origin != reconciled.coordinate_frame.origin;

    let prior_by_id: BTreeMap<u64, &SourceBinding> = reconciled
        .source_bindings
        .iter()
        .map(|binding| (binding.source_id, binding))
        .collect();
    let current_by_id: BTreeMap<u64, &SourceBinding> = current
        .source_bindings
        .iter()
        .map(|binding| (binding.source_id, binding))
        .collect();

    for lock in &mut reconciled.locks {
        if lock.status != SourceLockStatus::Active {
            if let Some(ref reason) = lock.stale_reason
                && !mismatch_reasons.contains(reason)
            {
                mismatch_reasons.push(reason.clone());
            }
            continue;
        }

        let stale_reason = if manifest_changed {
            Some("input_manifest_changed".to_string())
        } else if frame_changed {
            Some("coordinate_frame_or_crop_changed".to_string())
        } else {
            let mut found_reason = None;
            for &source_id in &lock.requested_sources {
                let before = prior_by_id.get(&source_id);
                let after = current_by_id.get(&source_id);
                let Some(after) = after else {
                    found_reason = Some(format!("source_removed:{source_id}"));
                    break;
                };
                if before.is_none_or(|entry| entry.sha256 != after.sha256) {
                    found_reason = Some(format!("source_digest_changed:{source_id}"));
                    break;
                }
                if before.is_some_and(|entry| {
                    entry.source_to_world_sha256 != after.source_to_world_sha256
                }) {
                    found_reason = Some(format!("source_transform_changed:{source_id}"));
                    break;
                }
            }
            found_reason
        };

        if let Some(reason) = stale_reason {
            if !mismatch_reasons.contains(&reason) {
                mismatch_reasons.push(reason.clone());
            }
            lock.status = SourceLockStatus::Stale;
            lock.stale_reason = Some(reason);
            newly_stale_count += 1;
        }
    }

    SidecarReconciliationReport {
        doc: reconciled,
        newly_stale_count,
        mismatch_reasons,
    }
}

fn point_on_segment(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
    const EPSILON: f64 = 1.0e-9;
    let cross = (py - ay) * (bx - ax) - (px - ax) * (by - ay);
    if cross.abs() > EPSILON * ((bx - ax).hypot(by - ay).max(1.0)) {
        return false;
    }
    let min_x = ax.min(bx) - EPSILON;
    let max_x = ax.max(bx) + EPSILON;
    let min_y = ay.min(by) - EPSILON;
    let max_y = ay.max(by) + EPSILON;
    (min_x..=max_x).contains(&px) && (min_y..=max_y).contains(&py)
}

pub fn point_in_polygon(world_x: f64, world_y: f64, polygon: &[[f64; 2]]) -> bool {
    if !world_x.is_finite() || !world_y.is_finite() || polygon.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for (i, &[xi, yi]) in polygon.iter().enumerate() {
        let [xj, yj] = polygon[j];
        if point_on_segment(world_x, world_y, xi, yi, xj, yj) {
            return true;
        }
        let crosses = (yi > world_y) != (yj > world_y);
        if crosses {
            let intersect_x = xj + (world_y - yj) * (xi - xj) / (yi - yj);
            if world_x < intersect_x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn point_in_world_region(world_x: f64, world_y: f64, region: &WorldRegion) -> bool {
    let [bx, by, bw, bh] = region.bbox;
    if world_x < bx - 1.0e-9
        || world_x > bx + bw + 1.0e-9
        || world_y < by - 1.0e-9
        || world_y > by + bh + 1.0e-9
    {
        return false;
    }
    point_in_polygon(world_x, world_y, &region.polygon_world)
}

/// Precedence defined by Task 9:
/// `stale/unresolved` > `single_source` > `allowed_sources` > `auto`.
pub fn resolve_lock_priority(locks_covering_point: &[&SourceLock]) -> SourceLockPriority {
    if locks_covering_point
        .iter()
        .any(|lock| lock.status != SourceLockStatus::Active)
    {
        return SourceLockPriority::StaleOrUnresolved;
    }
    if locks_covering_point.iter().any(|lock| {
        lock.status == SourceLockStatus::Active && lock.policy == SourceLockPolicy::SingleSource
    }) {
        return SourceLockPriority::SingleSource;
    }
    if locks_covering_point.iter().any(|lock| {
        lock.status == SourceLockStatus::Active && lock.policy == SourceLockPolicy::AllowedSources
    }) {
        return SourceLockPriority::AllowedSources;
    }
    SourceLockPriority::Auto
}

/// Resolve the read-only candidate constraint at a world coordinate.
/// Higher-precedence constraints only narrow candidate sets or block publication;
/// overlapping locks intersect deterministically and fail closed on empty overlap.
pub fn resolve_world_point_constraint(
    doc: &SourceLockDocument,
    world_x: f64,
    world_y: f64,
) -> WorldPointConstraint {
    let covering: Vec<&SourceLock> = doc
        .locks
        .iter()
        .filter(|lock| point_in_world_region(world_x, world_y, &lock.region))
        .collect();
    if covering.is_empty() {
        return WorldPointConstraint::Auto;
    }

    let stale_ids: Vec<String> = covering
        .iter()
        .filter(|lock| lock.status != SourceLockStatus::Active)
        .map(|lock| lock.lock_id.clone())
        .collect();
    if !stale_ids.is_empty() {
        return WorldPointConstraint::BlockedByStaleOrUnresolved {
            lock_ids: stale_ids,
        };
    }

    let lock_ids: Vec<String> = covering.iter().map(|lock| lock.lock_id.clone()).collect();
    let mut allowed: Option<BTreeSet<u64>> = None;
    for lock in &covering {
        let lock_set: BTreeSet<u64> = lock.requested_sources.iter().copied().collect();
        allowed = Some(match allowed {
            None => lock_set,
            Some(existing) => existing.intersection(&lock_set).copied().collect(),
        });
    }
    let intersected: Vec<u64> = allowed.unwrap_or_default().into_iter().collect();
    if intersected.is_empty() {
        return WorldPointConstraint::Conflict { lock_ids };
    }

    match resolve_lock_priority(&covering) {
        SourceLockPriority::SingleSource => WorldPointConstraint::SingleSource {
            source_id: intersected[0],
            lock_ids,
        },
        SourceLockPriority::AllowedSources => WorldPointConstraint::AllowedSources {
            source_ids: intersected,
            lock_ids,
        },
        SourceLockPriority::StaleOrUnresolved => {
            WorldPointConstraint::BlockedByStaleOrUnresolved { lock_ids }
        }
        SourceLockPriority::Auto => WorldPointConstraint::Auto,
    }
}

pub fn find_manual_review_region_at_point(
    doc: &SourceLockDocument,
    world_x: f64,
    world_y: f64,
) -> Option<&ManualReviewRegion> {
    doc.manual_review_regions
        .iter()
        .find(|region| point_in_polygon(world_x, world_y, &region.polygon_world))
}

/// Automatic Quality Gate accounting with manual-review exclusions (Task 9 AC5).
/// Manual regions never erase failures from `failed_all_tiles`, and insufficient
/// remaining auto-eligible samples must fail closed as `insufficient_evidence`.
pub fn compute_manual_review_quality_gate_accounting(
    all_tiles: &[QualityGateTileObservation],
    doc: &SourceLockDocument,
    minimum_auto_samples: usize,
) -> ManualReviewQualityGateAccounting {
    let region_by_id: BTreeMap<&str, &ManualReviewRegion> = doc
        .manual_review_regions
        .iter()
        .map(|region| (region.region_id.as_str(), region))
        .collect();

    let mut excluded_tiles = Vec::new();
    let mut auto_eligible_tiles = 0_usize;
    let mut failed_all_tiles = 0_usize;
    let mut failed_auto_eligible_tiles = 0_usize;

    for tile in all_tiles {
        if tile.failed {
            failed_all_tiles += 1;
        }
        let matched = tile
            .manual_region_id
            .as_deref()
            .and_then(|region_id| region_by_id.get(region_id))
            .copied();
        if let Some(region) = matched
            && region.excluded_from_quality_gate
        {
            excluded_tiles.push(ExcludedQualityGateTile {
                tile_id: tile.tile_id.clone(),
                region_id: region.region_id.clone(),
                reason: region.reason.clone(),
            });
        } else {
            auto_eligible_tiles += 1;
            if tile.failed {
                failed_auto_eligible_tiles += 1;
            }
        }
    }

    let verdict = if failed_auto_eligible_tiles > 0 {
        ManualReviewQualityGateVerdict::Fail
    } else if auto_eligible_tiles < minimum_auto_samples {
        ManualReviewQualityGateVerdict::InsufficientEvidence
    } else {
        ManualReviewQualityGateVerdict::Pass
    };

    ManualReviewQualityGateAccounting {
        all_tiles: all_tiles.len(),
        auto_eligible_tiles,
        excluded_tiles,
        failed_all_tiles,
        failed_auto_eligible_tiles,
        verdict,
    }
}

pub fn summarize_sidecar_for_report(
    doc: &SourceLockDocument,
    sidecar_sha256: String,
    tiff_sha256: String,
) -> StackReportSidecarReferenceSummary {
    let mut active_locks = 0_usize;
    let mut active_lock_area_px = 0_u64;
    let mut stale_or_unresolved_locks = 0_usize;
    let mut stale_or_unresolved_area_px = 0_u64;
    let mut locked_eligible_tiles = 0_u64;

    for lock in &doc.locks {
        let area = compute_region_pixel_area(&lock.region.polygon_world);
        if lock.status == SourceLockStatus::Active {
            active_locks += 1;
            active_lock_area_px = active_lock_area_px.saturating_add(area);
            locked_eligible_tiles = locked_eligible_tiles.saturating_add(lock.eligible_tiles);
        } else {
            stale_or_unresolved_locks += 1;
            stale_or_unresolved_area_px = stale_or_unresolved_area_px.saturating_add(area);
        }
    }

    let manual_review_area_px = doc.manual_review_regions.iter().fold(0_u64, |acc, region| {
        acc.saturating_add(compute_region_pixel_area(&region.polygon_world))
    });

    StackReportSidecarReferenceSummary {
        run_id: doc.run_id.clone(),
        sidecar_sha256,
        tiff_sha256,
        active_locks,
        active_lock_area_px,
        stale_or_unresolved_locks,
        stale_or_unresolved_area_px,
        manual_review_regions: doc.manual_review_regions.len(),
        manual_review_area_px,
        locked_eligible_tiles,
        history_head: doc.history.head,
        event_count: doc.history.events.len(),
    }
}

/// Pure in-memory export binding audit matching `verifyExportBinding` in
/// `src/utils/sourceLockSidecar.ts` plus deep integrity validation.
pub fn verify_export_binding(
    doc: &SourceLockDocument,
    current_tiff_sha256: &str,
    recorded_tiff_sha256: &str,
    recorded_sidecar_sha256: &str,
    events_jsonl: Option<&str>,
    current_run: Option<&CurrentRunBinding>,
) -> Result<ExportBindingAudit, String> {
    let sidecar_sha256 = compute_sidecar_digest(doc)?;
    let integrity_errors = validate_source_lock_integrity(doc);
    let events_error = events_jsonl.and_then(|jsonl| verify_events_jsonl_str(doc, jsonl).err());
    let reconciliation = current_run.map(|run| reconcile_document_staleness(doc, run));
    let effective_doc = reconciliation
        .as_ref()
        .map(|report| &report.doc)
        .unwrap_or(doc);

    let summary = summarize_sidecar_for_report(
        effective_doc,
        sidecar_sha256.clone(),
        current_tiff_sha256.to_string(),
    );

    let mut reasons = Vec::new();
    for error in &integrity_errors {
        reasons.push(format!("sidecar integrity error: {error}"));
    }
    if let Some(ref error) = events_error {
        reasons.push(error.clone());
    }
    if summary.stale_or_unresolved_locks > 0 {
        reasons.push(format!(
            "{} stale or unresolved lock(s) present",
            summary.stale_or_unresolved_locks
        ));
    }
    if let Some(ref report) = reconciliation {
        for reason in &report.mismatch_reasons {
            reasons.push(format!("reconciliation mismatch: {reason}"));
        }
    }
    let sidecar_mismatch = sidecar_sha256 != recorded_sidecar_sha256;
    if sidecar_mismatch {
        reasons.push("sidecar changed after TIFF was rendered (stale_export)".into());
    }
    let tiff_mismatch = current_tiff_sha256 != recorded_tiff_sha256;
    if tiff_mismatch {
        reasons
            .push("TIFF digest does not match recorded sidecar binding (binding_mismatch)".into());
    }

    let status = if !integrity_errors.is_empty() {
        ExportBindingStatus::IntegrityError
    } else if events_error.is_some() {
        ExportBindingStatus::EventsLogMismatch
    } else if tiff_mismatch {
        ExportBindingStatus::BindingMismatch
    } else if sidecar_mismatch {
        ExportBindingStatus::StaleExport
    } else if summary.stale_or_unresolved_locks > 0 {
        ExportBindingStatus::HasStaleOrUnresolvedLocks
    } else {
        ExportBindingStatus::Valid
    };

    Ok(ExportBindingAudit {
        status,
        can_publish_as_confirmed: status == ExportBindingStatus::Valid,
        sidecar_sha256,
        tiff_sha256: current_tiff_sha256.to_string(),
        summary,
        reconciliation,
        reasons,
    })
}

/// Read-only on-disk verification of an exported native TIFF,
/// `source-locks.json`, and optional `source-locks.events.jsonl`.
pub fn audit_export_bundle_on_disk(
    tiff_path: &Path,
    sidecar_path: &Path,
    events_jsonl_path: Option<&Path>,
    recorded_tiff_sha256: &str,
    recorded_sidecar_sha256: &str,
    current_run: Option<&CurrentRunBinding>,
) -> Result<ExportBindingAudit, String> {
    let current_tiff_sha256 = sha256_file_hex(tiff_path)?;
    let doc = read_source_lock_document(sidecar_path)?;
    let events_jsonl = match events_jsonl_path {
        Some(path) => Some(read_bounded_utf8_file(path, "source-lock events log")?),
        None => None,
    };
    verify_export_binding(
        &doc,
        &current_tiff_sha256,
        recorded_tiff_sha256,
        recorded_sidecar_sha256,
        events_jsonl.as_deref(),
        current_run,
    )
}

#[cfg(test)]
#[path = "source_lock_sidecar_tests.rs"]
mod tests;
