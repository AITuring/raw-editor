//! The Virtual_Tile cache key and its source content digest.
//!
//! SHA-256 of each Source_RAW file plus the stack parameters form the
//! cache_key that names an entry directory and seeds the run.

use serde::Deserialize;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

// ---------------------------------------------------------------------------
// Source content digest (the third cache key ingredient)
// ---------------------------------------------------------------------------

/// Read buffer for the source digest.  Large enough that a 50 MB RAW is a few
/// hundred reads, small enough to stay out of the way of the decoders.
pub(super) const DIGEST_CHUNK_BYTES: usize = 1 << 20;

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

pub(super) fn parse_hex_digest(value: &str) -> Option<[u8; 32]> {
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
