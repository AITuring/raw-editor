//! Deterministic iteration helpers (design "逐字节可复现", 需求 14.6).
//!
//! `std::collections::HashMap` seeds its hasher from the operating system once
//! per process, so iterating a map directly produces a different order on every
//! run.  Any decision or floating point reduction that depends on that order
//! makes the final canvas irreproducible even for identical input.
//!
//! Two remedies are used across the pipeline:
//!
//! 1. change the container to `BTreeMap` / `BTreeSet` when the key is `Ord` and
//!    the map is small enough that the lookup cost is irrelevant, and
//! 2. keep the `HashMap` (hot lookup paths) but iterate it through
//!    [`sorted_pairs`] / [`sorted_keys`] / [`into_sorted_pairs`].
//!
//! Every new comparison over floating point values must use `f64::total_cmp`
//! plus a stable secondary key so that equal scores cannot reorder.
//!
//! The second half of this module covers the other order dependency named by
//! the design: floating point **reduction** order.  `rayon`'s `sum()` splits
//! the input into as many pieces as the current thread pool has work to give,
//! so the shape of its reduction tree — and therefore the last bits of the
//! result — changes with `RAYON_NUM_THREADS`.  [`deterministic_sum`] and
//! friends fix the reduction tree to a compile-time block length and let
//! `rayon` vary only *which thread* computes a block, never *how the blocks
//! are cut*.
//!
//! The last part of this module covers the design's fourth source, "时间与随机源":
//! the RANSAC sampling seed.  It is derived from the input content
//! ([`derive_run_seed_from_paths`]) and installed once per run, so the sampling
//! sequence is a function of the selection and of nothing else — not the wall
//! clock, not the thread id, not the order the files were picked in.

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;
use sha2::{Digest, Sha256};

/// Borrow every `(key, value)` pair of `map` ordered by key.
///
/// This is the replacement for `for (key, value) in &map`.
pub(crate) fn sorted_pairs<K, V, S>(map: &HashMap<K, V, S>) -> Vec<(&K, &V)>
where
    K: Ord + std::hash::Hash,
    S: BuildHasher,
{
    let mut pairs = map.iter().collect::<Vec<_>>();
    pairs.sort_unstable_by(|left, right| left.0.cmp(right.0));
    pairs
}

/// Copy every key of `map` in ascending key order.
///
/// This is the replacement for `for key in map.keys()`.
pub(crate) fn sorted_keys<K, V, S>(map: &HashMap<K, V, S>) -> Vec<K>
where
    K: Ord + Copy + std::hash::Hash,
    S: BuildHasher,
{
    let mut keys = map.keys().copied().collect::<Vec<_>>();
    keys.sort_unstable();
    keys
}

/// Consume `map` and yield its `(key, value)` pairs ordered by key.
///
/// This is the replacement for `for (key, value) in map`.
#[allow(dead_code)]
pub(crate) fn into_sorted_pairs<K, V, S>(map: HashMap<K, V, S>) -> Vec<(K, V)>
where
    K: Ord + std::hash::Hash,
    S: BuildHasher,
{
    let mut pairs = map.into_iter().collect::<Vec<_>>();
    pairs.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    pairs
}

/// Number of consecutive values that form one accumulation block.
///
/// This is a **compile-time constant on purpose**: it may not be derived from
/// `rayon::current_num_threads()`, the input length, or any other runtime
/// quantity, because the constant alone (together with the input length) fixes
/// the shape of the reduction tree.
///
/// Inputs of at most `DETERMINISTIC_SUM_BLOCK_LEN` values are summed serially,
/// which is also exactly what the blocked path degenerates to for a single
/// block, so both branches agree bit for bit.
#[allow(dead_code)]
pub(crate) const DETERMINISTIC_SUM_BLOCK_LEN: usize = 4096;

/// Minimum number of blocks handed to one `rayon` task.
///
/// This only controls scheduling granularity. Changing it — like changing the
/// thread count — cannot change the result, because every block is reduced on
/// its own and the block sums are combined by block index afterwards.
#[allow(dead_code)]
const DETERMINISTIC_SUM_MIN_BLOCKS_PER_TASK: usize = 1;

/// Left-to-right serial accumulation: the reference order for every block.
#[inline]
fn block_sum(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0f64, |accumulator, value| accumulator + value)
}

/// Sum `values` so that the result depends only on the values, their order in
/// the slice and [`DETERMINISTIC_SUM_BLOCK_LEN`] — never on the thread count.
///
/// Use this for every floating point reduction whose result reaches a decision
/// (a threshold, a comparison, a reported metric) or an output pixel. Integer
/// reductions and `count()` do not need it: integer addition is associative, so
/// their value is already independent of the reduction order.
#[allow(dead_code)]
pub(crate) fn deterministic_sum(values: &[f64]) -> f64 {
    deterministic_sum_with_granularity(values, DETERMINISTIC_SUM_MIN_BLOCKS_PER_TASK)
}

/// [`deterministic_sum`] with an explicit scheduling granularity.
///
/// Only the unit tests pass a granularity other than the default; they use it
/// to assert that the scheduling decision is invisible in the result.
#[allow(dead_code)]
fn deterministic_sum_with_granularity(values: &[f64], min_blocks_per_task: usize) -> f64 {
    if values.len() <= DETERMINISTIC_SUM_BLOCK_LEN {
        return block_sum(values);
    }
    let block_sums = values
        .par_chunks(DETERMINISTIC_SUM_BLOCK_LEN)
        .with_min_len(min_blocks_per_task.max(1))
        .map(block_sum)
        .collect::<Vec<_>>();
    // `collect` keeps the block order, so this final accumulation runs over
    // ascending block indices no matter which thread produced which block.
    block_sum(&block_sums)
}

/// Deterministic replacement for `items.par_iter().map(map).sum::<f64>()`.
///
/// The mapping still runs in parallel; only the accumulation order is pinned.
#[allow(dead_code)]
pub(crate) fn deterministic_sum_map<T, F>(items: &[T], map: F) -> f64
where
    T: Sync,
    F: Fn(&T) -> f64 + Send + Sync,
{
    if items.len() <= DETERMINISTIC_SUM_BLOCK_LEN {
        return items
            .iter()
            .fold(0.0f64, |accumulator, item| accumulator + map(item));
    }
    let block_sums = items
        .par_chunks(DETERMINISTIC_SUM_BLOCK_LEN)
        .with_min_len(DETERMINISTIC_SUM_MIN_BLOCKS_PER_TASK)
        .map(|block| {
            block
                .iter()
                .fold(0.0f64, |accumulator, item| accumulator + map(item))
        })
        .collect::<Vec<_>>();
    block_sum(&block_sums)
}

/// Arithmetic mean over [`deterministic_sum`]; `None` for an empty slice.
#[allow(dead_code)]
pub(crate) fn deterministic_mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| deterministic_sum(values) / values.len() as f64)
}

// ---------------------------------------------------------------------------
// Content derived RANSAC seed (设计「时间与随机源」, 需求 14.6)
// ---------------------------------------------------------------------------

/// Recorded in `resources.random_seed_source` when no run seed is installed.
///
/// Every RANSAC site mixes the run seed into a per-site compile-time constant,
/// so an absent run seed still gives a fixed, reproducible sampling sequence —
/// it is simply not derived from the input. This is the value the non focus
/// stack panorama path and the unit tests run with.
pub(crate) const RANDOM_SEED_SOURCE_SITE_CONSTANTS: &str = "site_constants_without_run_seed";

/// Recorded in `resources.random_seed_source` when the run seed comes from the
/// Virtual_Tile `cache_key`.
///
/// The `cache_key` is
/// `SHA-256(STACK_PIPELINE_VERSION ‖ sorted_paths ‖ sorted_source_sha256)`
/// (需求 4.5, `stack_pipeline::virtual_tile::CacheKeyInputs`), so the seed is a
/// function of the pipeline version, the selection **and the content** of every
/// Source_RAW. Editing a source without renaming it therefore reseeds the run,
/// which the earlier path-only derivation could not see.
pub(crate) const RANDOM_SEED_SOURCE_SHA256_CACHE_KEY: &str = "sha256_cache_key";

/// Sentinel for "no run seed installed". Mixing zero is the identity, so the
/// RANSAC sites fall back to their site constant alone.
const RUN_SEED_UNSET: u64 = 0;

/// Seed of the run currently in flight.
///
/// The RANSAC entry points are free functions many call levels below the
/// pipeline entry and have no seed parameter; threading one through would touch
/// every geometry call site. A run scoped global keeps the change local, in the
/// same shape as `degradation::RUN_LEDGER`. It is written once per run, before
/// any matching starts, and only read afterwards.
static RUN_RANDOM_SEED: AtomicU64 = AtomicU64::new(RUN_SEED_UNSET);

/// Derive a run seed from the pipeline version and the source paths.
///
/// The digest input is length prefixed per path so that no two different path
/// lists can produce the same byte stream, and the paths are sorted by raw
/// bytes so that the import order of the selection cannot reach the seed
/// (需求 14.6: import order must not change the output).
///
/// Neither the wall clock, the thread id, nor any file system timestamp is
/// consulted. File size is deliberately left out too: it is not a content
/// identity and would make the seed depend on container padding.
///
/// This is **not** the production derivation any more:
/// [`derive_run_seed_from_cache_key`] replaced it once the Virtual_Tile
/// `cache_key` existed. It is kept for the synthetic property tests, which have
/// no files on disk to digest, and as the reference for what the seed looked
/// like before the content digests joined it.
#[allow(dead_code)]
pub(crate) fn derive_run_seed_from_paths(pipeline_version: &str, source_paths: &[String]) -> u64 {
    let mut sorted = source_paths
        .iter()
        .map(String::as_bytes)
        .collect::<Vec<_>>();
    sorted.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update((pipeline_version.len() as u64).to_be_bytes());
    hasher.update(pipeline_version.as_bytes());
    hasher.update((sorted.len() as u64).to_be_bytes());
    for path in sorted {
        hasher.update((path.len() as u64).to_be_bytes());
        hasher.update(path);
    }
    let digest = hasher.finalize();
    let mut leading = [0u8; 8];
    leading.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(leading)
}

/// Derive the run seed from the Virtual_Tile `cache_key` (需求 14.6).
///
/// The `cache_key` already binds the pipeline version, the sorted absolute
/// paths and the sorted per-source content digests (需求 4.5), which is exactly
/// the input identity the sampling sequence must follow. Hashing it once more
/// keeps the seed from being a truncation of a value that also names a
/// directory on disk.
pub(crate) fn derive_run_seed_from_cache_key(cache_key: &[u8; 32]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(cache_key);
    let digest = hasher.finalize();
    let mut leading = [0u8; 8];
    leading.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(leading)
}

/// Install the seed of the run that is about to start.
///
/// A seed that happens to hash to [`RUN_SEED_UNSET`] is nudged to 1 so that
/// "derived zero" and "not installed" stay distinguishable; the nudge is a pure
/// function of the digest, so it does not cost reproducibility.
pub(crate) fn set_run_random_seed(seed: u64) {
    let stored = if seed == RUN_SEED_UNSET { 1 } else { seed };
    RUN_RANDOM_SEED.store(stored, Ordering::Relaxed);
}

/// Remove the run seed, so the RANSAC sites use their site constants alone.
///
/// Called at the start of every run that does not install a seed, so that a
/// run can never inherit the seed of a previous one.
pub(crate) fn clear_run_random_seed() {
    RUN_RANDOM_SEED.store(RUN_SEED_UNSET, Ordering::Relaxed);
}

/// Value every RANSAC site mixes into its own compile-time seed constant.
///
/// Returns 0 when no run seed is installed, which leaves the site constant
/// unchanged.
pub(crate) fn run_random_seed_mix() -> u64 {
    RUN_RANDOM_SEED.load(Ordering::Relaxed)
}

/// Identifier for `resources.random_seed_source`, given the installed seed.
pub(crate) fn run_random_seed_source() -> &'static str {
    if run_random_seed_mix() == RUN_SEED_UNSET {
        RANDOM_SEED_SOURCE_SITE_CONSTANTS
    } else {
        RANDOM_SEED_SOURCE_SHA256_CACHE_KEY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn shuffled_map() -> HashMap<(usize, usize), &'static str> {
        // Insertion order is deliberately different from key order so that a
        // regression to raw `HashMap` iteration is visible.
        let mut map = HashMap::new();
        for &(key, value) in &[
            ((3usize, 1usize), "c"),
            ((0, 9), "a"),
            ((3, 0), "b"),
            ((1, 4), "d"),
        ] {
            map.insert(key, value);
        }
        map
    }

    #[test]
    fn sorted_pairs_orders_by_key() {
        let map = shuffled_map();
        assert_eq!(
            sorted_pairs(&map)
                .into_iter()
                .map(|(key, value)| (*key, *value))
                .collect::<Vec<_>>(),
            vec![((0, 9), "a"), ((1, 4), "d"), ((3, 0), "b"), ((3, 1), "c")]
        );
    }

    #[test]
    fn sorted_keys_orders_by_key() {
        let map = shuffled_map();
        assert_eq!(sorted_keys(&map), vec![(0, 9), (1, 4), (3, 0), (3, 1)]);
    }

    #[test]
    fn into_sorted_pairs_orders_by_key() {
        let map = shuffled_map();
        assert_eq!(
            into_sorted_pairs(map)
                .into_iter()
                .map(|(key, _)| key)
                .collect::<Vec<_>>(),
            vec![(0, 9), (1, 4), (3, 0), (3, 1)]
        );
    }

    /// Values whose sum is deliberately sensitive to accumulation order: the
    /// magnitudes span 40 decades, so adding them in a different order rounds
    /// differently. `order_sensitive_values_really_are_order_sensitive` proves
    /// that this generator does its job, otherwise every assertion below could
    /// pass for the wrong reason.
    fn order_sensitive_values(count: usize) -> Vec<f64> {
        (0..count)
            .map(|index| {
                let magnitude = 10f64.powi(index as i32 % 40 - 20);
                let sign = if index.is_multiple_of(3) { -1.0 } else { 1.0 };
                sign * magnitude * (1.0 + (index % 17) as f64)
            })
            .collect()
    }

    fn naive_serial_sum(values: &[f64]) -> f64 {
        values
            .iter()
            .fold(0.0f64, |accumulator, value| accumulator + value)
    }

    #[test]
    fn order_sensitive_values_really_are_order_sensitive() {
        let values = order_sensitive_values(8 * DETERMINISTIC_SUM_BLOCK_LEN + 7);
        let mut reversed = values.clone();
        reversed.reverse();
        assert_ne!(
            naive_serial_sum(&values).to_bits(),
            naive_serial_sum(&reversed).to_bits(),
            "test data must be able to expose a changed accumulation order"
        );
    }

    #[test]
    fn deterministic_sum_matches_serial_fold_up_to_one_block() {
        for count in [0usize, 1, 2, 4095, DETERMINISTIC_SUM_BLOCK_LEN] {
            let values = order_sensitive_values(count);
            assert_eq!(
                deterministic_sum(&values).to_bits(),
                naive_serial_sum(&values).to_bits(),
                "{count} values must take the serial path"
            );
        }
    }

    #[test]
    fn deterministic_sum_is_independent_of_scheduling_granularity() {
        // One block per task versus one task for everything: the block cut is a
        // compile-time constant, so only the scheduling changes.
        for count in [
            DETERMINISTIC_SUM_BLOCK_LEN + 1,
            3 * DETERMINISTIC_SUM_BLOCK_LEN,
            9 * DETERMINISTIC_SUM_BLOCK_LEN + 123,
        ] {
            let values = order_sensitive_values(count);
            let reference = deterministic_sum_with_granularity(&values, 1);
            for granularity in [1usize, 2, 3, 8, 64, 4096] {
                assert_eq!(
                    deterministic_sum_with_granularity(&values, granularity).to_bits(),
                    reference.to_bits(),
                    "{count} values, granularity {granularity}"
                );
            }
        }
    }

    #[test]
    fn deterministic_sum_is_independent_of_thread_count() {
        let values = order_sensitive_values(11 * DETERMINISTIC_SUM_BLOCK_LEN + 37);
        let mut results = Vec::new();
        for threads in [1usize, 2, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("building a fixed size pool must succeed");
            results.push((
                threads,
                pool.install(|| deterministic_sum(&values)).to_bits(),
            ));
        }
        let (_, expected) = results[0];
        for (threads, bits) in results {
            assert_eq!(
                bits, expected,
                "{threads} threads must produce the same bits as 1 thread"
            );
        }
    }

    #[test]
    fn deterministic_sum_map_matches_deterministic_sum_of_mapped_values() {
        for count in [
            0usize,
            7,
            DETERMINISTIC_SUM_BLOCK_LEN,
            5 * DETERMINISTIC_SUM_BLOCK_LEN + 3,
        ] {
            let values = order_sensitive_values(count);
            let doubled = values.iter().map(|value| value * 2.0).collect::<Vec<_>>();
            assert_eq!(
                deterministic_sum_map(&values, |value| value * 2.0).to_bits(),
                deterministic_sum(&doubled).to_bits(),
                "{count} values"
            );
        }
    }

    #[test]
    fn deterministic_sum_map_is_independent_of_thread_count() {
        let values = order_sensitive_values(6 * DETERMINISTIC_SUM_BLOCK_LEN + 5);
        let mut results = Vec::new();
        for threads in [1usize, 2, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("building a fixed size pool must succeed");
            results.push(
                pool.install(|| deterministic_sum_map(&values, |value| value.abs().sqrt()))
                    .to_bits(),
            );
        }
        assert!(results.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn deterministic_mean_is_none_only_for_an_empty_slice() {
        assert_eq!(deterministic_mean(&[]), None);
        assert_eq!(deterministic_mean(&[2.0, 4.0, 6.0]), Some(4.0));
    }

    fn sample_paths() -> Vec<String> {
        vec![
            "/tmp/stack/DSC_3681.NEF".to_string(),
            "/tmp/stack/DSC_3680.NEF".to_string(),
            "/tmp/stack/DSC_3682.NEF".to_string(),
        ]
    }

    #[test]
    fn run_seed_is_independent_of_the_import_order() {
        let forward = sample_paths();
        let mut backward = forward.clone();
        backward.reverse();
        assert_eq!(
            derive_run_seed_from_paths("stack-test", &forward),
            derive_run_seed_from_paths("stack-test", &backward)
        );
    }

    #[test]
    fn run_seed_changes_with_every_ingredient() {
        let paths = sample_paths();
        let reference = derive_run_seed_from_paths("stack-test", &paths);
        assert_ne!(reference, derive_run_seed_from_paths("stack-other", &paths));

        let mut renamed = paths.clone();
        renamed[0] = "/tmp/stack/DSC_9999.NEF".to_string();
        assert_ne!(
            reference,
            derive_run_seed_from_paths("stack-test", &renamed)
        );

        let mut dropped = paths.clone();
        dropped.pop();
        assert_ne!(
            reference,
            derive_run_seed_from_paths("stack-test", &dropped)
        );
    }

    #[test]
    fn run_seed_is_length_prefixed_against_path_boundary_collisions() {
        // Without the length prefix both selections would hash the same
        // concatenated byte stream.
        let joined = vec!["/a/b".to_string(), "c".to_string()];
        let split = vec!["/a".to_string(), "bc".to_string()];
        assert_ne!(
            derive_run_seed_from_paths("v", &joined),
            derive_run_seed_from_paths("v", &split)
        );
    }

    #[test]
    fn run_seed_is_stable_across_processes() {
        // A literal expectation: the seed must be a pure function of the input
        // bytes, so a change to the derivation shows up here rather than as an
        // unexplained pixel diff.
        assert_eq!(
            derive_run_seed_from_paths("stack-2026.09.22.1", &sample_paths()),
            0xe465_e64a_a71b_aa8c
        );
    }

    #[test]
    fn cache_key_seed_is_a_pure_function_of_the_cache_key() {
        let reference = derive_run_seed_from_cache_key(&[0u8; 32]);
        assert_eq!(reference, derive_run_seed_from_cache_key(&[0u8; 32]));
        let mut flipped = [0u8; 32];
        flipped[31] = 1;
        assert_ne!(reference, derive_run_seed_from_cache_key(&flipped));
        // A literal expectation, so a change to the derivation shows up here
        // rather than as an unexplained pixel diff.
        assert_eq!(reference, 0x6668_7aad_f862_bd77);
        // The seed is not a truncation of the cache key itself.
        assert_ne!(reference, u64::from_be_bytes([0u8; 8]));
    }

    /// This test owns the process wide run seed, so it exercises the install,
    /// report and clear steps in one body instead of racing sibling tests.
    #[test]
    fn installing_and_clearing_the_run_seed_switches_the_recorded_source() {
        clear_run_random_seed();
        assert_eq!(run_random_seed_mix(), RUN_SEED_UNSET);
        assert_eq!(run_random_seed_source(), RANDOM_SEED_SOURCE_SITE_CONSTANTS);

        let seed = derive_run_seed_from_cache_key(&[0x5Au8; 32]);
        set_run_random_seed(seed);
        assert_eq!(run_random_seed_mix(), seed);
        assert_eq!(
            run_random_seed_source(),
            RANDOM_SEED_SOURCE_SHA256_CACHE_KEY
        );

        // A digest of zero must stay distinguishable from "not installed".
        set_run_random_seed(RUN_SEED_UNSET);
        assert_eq!(run_random_seed_mix(), 1);
        assert_eq!(
            run_random_seed_source(),
            RANDOM_SEED_SOURCE_SHA256_CACHE_KEY
        );

        clear_run_random_seed();
        assert_eq!(run_random_seed_source(), RANDOM_SEED_SOURCE_SITE_CONSTANTS);
    }

    #[test]
    fn helpers_are_independent_of_insertion_order() {
        let mut forward = HashMap::new();
        let mut backward = HashMap::new();
        for index in 0..64usize {
            forward.insert(index, index * 2);
        }
        for index in (0..64usize).rev() {
            backward.insert(index, index * 2);
        }
        assert_eq!(sorted_keys(&forward), sorted_keys(&backward));
        assert_eq!(
            sorted_pairs(&forward)
                .into_iter()
                .map(|(key, value)| (*key, *value))
                .collect::<Vec<_>>(),
            sorted_pairs(&backward)
                .into_iter()
                .map(|(key, value)| (*key, *value))
                .collect::<Vec<_>>()
        );
    }
}
