//! Stack_Pipeline: the layered capture-station focus stacking and stitching
//! pipeline.
//!
//! This module is introduced observation-first: stage 0 of the rollout only
//! adds the machine readable [`report::StackReport`] so that a real 84-frame
//! regression can produce numbers before any threshold or fallback behaviour
//! changes.  Nothing in here may alter pixel output or an existing fallback
//! decision.
//!
//! Component files are added incrementally by the later stages
//! (`station_grouper`, `intra_station`, `focus_fuser`, `virtual_tile`,
//! `topology`, `station_pose`, `closure`, `residual_warp`, `tone`,
//! `compositor`, `quality_gate`, `degradation`, `diagnostics`).

/// Tile_Compositor path selection: the `StackCompositorChoice` setting
/// (需求 15.1–15.3, 15.10, 15.11).
pub(crate) mod compositor;
pub(crate) mod degradation;
/// Deterministic iteration helpers shared by every stage (需求 14.6).
pub(crate) mod determinism;
/// Focus_Fuser: ownership cell metrics and grid geometry (需求 3.1–3.3, 3.9).
pub(crate) mod focus_fuser;
/// Intra_Station_Registrar: anchor selection and registration gates (需求 2.1–2.9).
pub(crate) mod intra_station;
/// Quality_Gate pure ROI and image measurements (需求 15.1–15.6).
pub(crate) mod quality_gate;
pub(crate) mod quality_gate_runner;
pub(crate) mod report;
/// Residual_Warp: bounded overlap-local correction fields (需求 8.1–8.9).
pub(crate) mod residual_warp;
/// Run-scoped memory threshold and peak RSS sampling (需求 14.1--14.3).
pub(crate) mod resources;
/// Degradation_Manager, station level: the degraded fusion paths and the
/// "joins no Capture_Station" rejection (需求 12.1, 12.2, 12.5).
pub(crate) mod station_degradation;
/// Tone_Harmonizer: robust low-frequency affine correction after ownership.
pub(crate) mod tone;
/// Capture_Topology_Model: pose-free station row/column inference and
/// ambiguity-forced candidate eligibility (需求 5.1, 5.2, 5.4, 5.7).
pub(crate) mod topology;
/// Virtual_Tile_Store: the versioned lossless disk cache (需求 4.3–4.5, 4.9–4.11).
pub(crate) mod virtual_tile;

/// `proptest` generators shared by the property tests of every stage.
#[cfg(test)]
pub(crate) mod test_support;

/// One property test per Correctness Property of the design document.
#[cfg(test)]
mod properties;
