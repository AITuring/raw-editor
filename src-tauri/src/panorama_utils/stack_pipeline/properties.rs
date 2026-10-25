//! `proptest` property tests for the Stack_Pipeline Correctness Properties of
//! the `layered-camera-group-focus-stitching` design document.
//!
//! One property test per numbered property, each configured with at least 100
//! cases.  Failing cases are persisted under `proptest-regressions/`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use half::f16;
use nalgebra::{Matrix3, Point2, Point3};
use proptest::prelude::*;
use regex::Regex;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use crate::image_stack::STACK_PIPELINE_VERSION;
use crate::panorama_stitching::{
    Feature, ImageInfo, KeyPoint, MatchInfo, determinism_test_access, grouping_test_access,
};
use crate::panorama_utils::{mosaic, processing, registration, stitching};

use super::degradation::{
    self, DegradationLedger, DegradationManager, FAILURE_REASONS, MAX_REASON_IDENTIFIER_LENGTH,
    StandardFinalOutputFileSystem, UNMEASURABLE_REASONS,
};
use super::determinism::{DETERMINISTIC_SUM_BLOCK_LEN, derive_run_seed_from_paths, sorted_keys};
use super::focus_fuser;
use super::intra_station;
use super::report::{
    ConnectivityReport, FusionReport, FusionSolverStatus, QualityGateCriterionRecord,
    QualityGateVerdict, STACK_REPORT_TOP_LEVEL_FIELDS, SourceRecord, StackReport, ToneStatus,
    VIRTUAL_TILE_CACHE_LIMIT_BYTES,
};
use super::test_support::{
    self, RenameScheme, SyntheticScan, arb_artwork_focus_bracket, arb_artwork_scan_grid,
    arb_coverage_shape, arb_oversized_bracket, arb_stack_report, arb_virtual_tile,
    arb_virtual_tile_in, full_coverage_shape, minimal_virtual_tile, virtual_tile_with_shape,
};
use super::virtual_tile::{
    self, CacheKeyInputs, CacheLookup, LeaseError, MAX_RESIDENT_VIRTUAL_TILES, StoreOutcome,
    VirtualTileStore, test_access,
};
use super::{compositor, resources, tone};

// The property suite is intentionally kept in one module scope because its
// independent oracles share fixtures and private production accessors.  The
// thematic includes below split the source for maintenance without changing
// test names, visibility, or report filters.
include!("properties_core_acceptance.rs");
include!("properties_determinism_grouping.rs");
include!("properties_virtual_tile_cache.rs");
include!("properties_station_grouping.rs");
include!("properties_focus_ownership.rs");
include!("properties_registration_topology.rs");
include!("properties_closure.rs");
include!("properties_quality.rs");
