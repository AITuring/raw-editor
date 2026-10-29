//! Shared `proptest` generators for the Stack_Pipeline property tests.
//!
//! The generators live in one module so every property task of the rollout
//! reuses the same inputs instead of inventing a private shape per test.  This
//! file is compiled only under `cfg(test)`; nothing in here is reachable from a
//! production path.
//!
//! Design reference: `Testing Strategy > 生成器设计` of the
//! `layered-camera-group-focus-stitching` design document.  This task adds
//! `arb_stack_report()`, the "random but schema legal Stack_Report, including
//! out-of-range combinations" generator.

// Later property tasks consume the remaining generators; keep them public
// inside the test build without warning noise in the meantime.
#![allow(dead_code)]

use std::path::PathBuf;

mod stack_report;

pub use stack_report::arb_stack_report;

mod artwork;

pub use artwork::{ArtworkPlane, arb_artwork_plane, arb_painted_artwork_plane};

mod scan_grid;

pub use scan_grid::{
    DecoySource, RenameScheme, SYNTHETIC_TILE_SIDE, SyntheticScan, arb_artwork_focus_bracket,
    arb_artwork_scan_grid, arb_scan_grid,
};

mod capture_station;

pub use capture_station::{
    OVERSIZED_BRACKET_LARGEST_STEP_PX, OVERSIZED_BRACKET_SIDE, arb_anchor_candidates,
    arb_intra_station_layer_pair, arb_intra_station_verdict_pair, arb_oversized_bracket,
};

mod virtual_tiles;

pub use virtual_tiles::{
    arb_coverage_shape, arb_virtual_tile, arb_virtual_tile_in, full_coverage_shape,
    minimal_virtual_tile, virtual_tile_with_shape,
};

/// Test-only destination for optional Tone_Harmonizer diagnostics.
pub fn tone_diagnostics_dir() -> Option<PathBuf> {
    std::env::var_os("RAW_EDITOR_TONE_DIAGNOSTICS_DIR").map(PathBuf::from)
}

/// Test-only destination for current-run owner Source_RAW fidelity evidence.
pub fn tone_owner_fidelity_dir() -> Option<PathBuf> {
    std::env::var_os("RAW_EDITOR_TONE_OWNER_FIDELITY_DIR").map(PathBuf::from)
}

/// `SplitMix64`: a tiny, fully specified bit mixer.
///
/// Used instead of a crate RNG so that the generated plane is a pure function
/// of the `proptest` drawn seed on every platform.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

/// Uniform value in `[0, 1)`.
fn unit_interval(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Uniform value in `[low, high)`.
fn in_range(state: &mut u64, low: f64, high: f64) -> f64 {
    low + unit_interval(state) * (high - low)
}
