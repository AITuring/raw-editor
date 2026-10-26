//! Stack_Report: the machine readable JSON report produced by every
//! Stack_Pipeline run.
//!
//! Requirement 10.8 requires a report on *every* terminating path (success,
//! degraded output, rejection, cancellation) within 30 seconds, and
//! requirement 11.16 requires the Quality_Gate metrics to land in the same
//! document.  The schema therefore fixes its top-level fields and writes
//! defaults explicitly, so `tests/focus-stack-quality-contract.mjs` can assert
//! the structure without decoding a single RAW file.
//!
//! This stage records the run metrics. Quality_Gate remains observation-only
//! until the later rejection-path task enables blocking export.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use super::degradation::{self, DegradationLedger, RunResult};
/// Single source of the actual Intra_Station analysis configuration
/// (requirement 2.2); the report echoes the configured 1400 rather than the
/// independent 2048 ceiling.
use super::focus_fuser;
use super::intra_station::{self, INTRA_STATION_ANALYSIS_LONG_SIDE};
use super::residual_warp;
use super::tone;

/// Bumped whenever the serialized shape of [`StackReport`] changes.
pub(crate) const STACK_REPORT_SCHEMA: u32 = 2;

/// 64 GiB Virtual_Tile cache budget (requirement 4.4).
pub(crate) const VIRTUAL_TILE_CACHE_LIMIT_BYTES: u64 = 68_719_476_736;
/// Canvas long-side ceiling (requirement 10.11).
pub(crate) const CANVAS_LONG_SIDE_LIMIT: u64 = 262_144;
/// Default peak-RSS threshold of 24 GiB (requirement 14).
pub(crate) const MEMORY_THRESHOLD_DEFAULT_BYTES: u64 = 25_769_803_776;
/// Owner_Region boundary Delta_E00 ceiling (requirement 9.8).
pub(crate) const TONE_BOUNDARY_DELTA_E_THRESHOLD: f64 = 1.5;
/// Low-frequency band sigma in world pixels (requirement 9.2).
pub(crate) const TONE_LOW_PASS_SIGMA_WORLD_PX: f64 = 64.0;
/// Residual_Warp node spacing in world pixels (requirement 8.2).
pub(crate) const RESIDUAL_WARP_NODE_STEP_PX: u32 = 64;
/// Directory (under the app cache dir) that collects the reports.
pub(crate) const STACK_REPORT_DIR_NAME: &str = "stack-reports";

/// Which composition path a run actually executed (requirement 15.11).
///
/// The identifiers are stable and machine readable; they are the same set the
/// `StackCompositorChoice` setting will select from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SelectedPath {
    LayeredVirtualTile,
    ProgressiveSeamTile,
    StreamingMosaic,
    LegacySingleLayerMosaic,
    SingleStation,
}

impl SelectedPath {
    /// Wired into the `StackCompositorChoice` setting by
    /// [`super::compositor::StackCompositorChoice::selected_path`]; the
    /// serialized form already goes through `#[serde(rename_all)]`.
    pub(crate) fn as_identifier(self) -> &'static str {
        match self {
            Self::LayeredVirtualTile => "layered_virtual_tile",
            Self::ProgressiveSeamTile => "progressive_seam_tile",
            Self::StreamingMosaic => "streaming_mosaic",
            Self::LegacySingleLayerMosaic => "legacy_single_layer_mosaic",
            Self::SingleStation => "single_station",
        }
    }
}

/// Outcome of a single run (requirement 12.9: rejection outranks degradation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StackRunResult {
    /// The run has not reached a terminating path yet.
    Running,
    Success,
    Degraded,
    Rejected,
    Cancelled,
}

impl StackRunResult {
    /// Consumed by the Quality_Gate reporting of a later phase.
    #[allow(dead_code)]
    pub(crate) fn as_identifier(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Success => "success",
            Self::Degraded => "degraded",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
        }
    }

    /// How strongly a state claims the run outcome.  `Cancelled` ranks highest
    /// because a user cancellation is the most specific terminating path: a
    /// cancelled run must never be reported as a success, a degraded output or
    /// a plain rejection.
    fn precedence(self) -> u8 {
        match self {
            Self::Running => 0,
            Self::Success => 1,
            Self::Degraded => 2,
            Self::Rejected => 3,
            Self::Cancelled => 4,
        }
    }

    /// Combine the state the stitching path already recorded with the state the
    /// [`DegradationLedger`] derived (requirement 12.9).  The stronger claim
    /// wins, so a ledger that saw nothing (`Success`) leaves an already
    /// recorded `Cancelled` or `Rejected` untouched, while a single ledger
    /// rejection overrides a `Success` written by the compositor.
    fn merge(self, ledger: Self) -> Self {
        if ledger.precedence() > self.precedence() {
            ledger
        } else {
            self
        }
    }
}

/// Translate a ledger verdict into the report's result vocabulary.  A ledger
/// cancellation is a `Rejected` severity entry, but the report keeps it
/// distinguishable as [`StackRunResult::Cancelled`].
fn ledger_run_result(ledger: &DegradationLedger) -> StackRunResult {
    if ledger.count_of(degradation::RUN_CANCELLED_BY_USER) > 0 {
        return StackRunResult::Cancelled;
    }
    match ledger.result() {
        RunResult::Success => StackRunResult::Success,
        RunResult::Degraded => StackRunResult::Degraded,
        RunResult::Rejected => StackRunResult::Rejected,
    }
}

/// Convert the ledger entries into their report records, preserving order.
fn degradation_entry_records(ledger: &DegradationLedger) -> Vec<DegradationEntryRecord> {
    ledger
        .entries()
        .iter()
        .map(|entry| DegradationEntryRecord {
            reason: entry.reason.to_string(),
            severity: entry.severity.as_identifier().to_string(),
            detail: entry.detail.clone(),
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct WorldRect {
    pub left: f64,
    pub top: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct WorldPoint {
    pub x: f64,
    pub y: f64,
}

// ---------------------------------------------------------------------------

include!("report/input_grouping_station.inc.rs");
include!("report/relations.inc.rs");
include!("report/geometry_tone_composition.inc.rs");
include!("report/quality.inc.rs");
include!("report/runtime.inc.rs");
include!("report/tests.inc.rs");
