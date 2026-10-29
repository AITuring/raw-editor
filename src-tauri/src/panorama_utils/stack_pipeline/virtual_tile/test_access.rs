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
