//! Degradation_Manager: stable machine readable failure reason identifiers plus
//! The ledger and final outcome decision for every degraded or rejected path
//! of a single run.
//!
//! Recording remains observation-only so deep pipeline stages cannot publish a
//! partial result by themselves. At the final publication boundary,
//! [`DegradationManager`] makes the one authoritative decision and either
//! atomically renames the run-owned temporary output or removes only that
//! temporary file. Stack reports and diagnostic previews live outside that
//! transaction and are never cleanup targets.
//!
//! Requirements: 12.7 (identifier format and stability), 12.9 (all active paths
//! recorded, rejection wins over degradation), 12.10 (no rejected output
//! residue while reports and diagnostics remain).

// The identifier set is declared complete up front so the contract tests can
// guard its format and uniqueness, while the call sites that raise each reason
// arrive with the later rollout stages. Drop this allow once every stage
// consumes its identifiers.
#![allow(dead_code)]

use std::cell::Cell;
use std::io;
use std::path::Path;
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, TryLockError};
use std::thread;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

// Stable failure/degradation identifiers grouped by pipeline stage.
include!("degradation/constants.inc.rs");
// Final outcome decision and staged-output publication boundary.
include!("degradation/decision.inc.rs");
// Failure-reason matrix and identifier validation.
include!("degradation/reasons.inc.rs");
// Per-run degradation ledger and result aggregation.
include!("degradation/ledger.inc.rs");
// Process-wide run scope and observation sink.
include!("degradation/runtime.inc.rs");

#[cfg(test)]
include!("degradation/tests.inc.rs");
