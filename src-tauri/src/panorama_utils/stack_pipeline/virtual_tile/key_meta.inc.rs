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

