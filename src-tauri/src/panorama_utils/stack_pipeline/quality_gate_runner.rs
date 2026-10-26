//! Runtime adapter for the pure Quality_Gate measurements.
//!
//! `quality_gate.rs` deliberately contains no image loading or report policy.
//! This module supplies that small amount of orchestration: it selects ROIs
//! from the final owner planes, decodes one owner source at a time, samples the
//! corresponding reference ROI, and records all nine criteria.  The caller can
//! therefore use it in observation-only mode without changing export pixels.

use std::collections::BTreeMap;
use std::time::Instant;

use image::{GrayImage, ImageBuffer, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Vector3};

use super::degradation;
use super::quality_gate::{self, EffectivePixelCoverage, QualityRoi, SourceGeometry};
use super::report::{
    FailedMeasurementRecord, QualityGateCriterionRecord, QualityGateReport,
    QualityGateTimingRecord, QualityGateVerdict, QualityStationStatistic, RoiPhotometryRecord,
    UnmeasurableCategory, UnmeasurableRecord, WorldPoint,
};
use super::residual_warp::ResidualWarp;

// Criterion data model, photometry, ROI selection, and coverage metrics.
include!("quality_gate_runner/criteria.inc.rs");
// Accepted-station boundary evidence and rejection classification.
include!("quality_gate_runner/boundaries.inc.rs");
// Owner-sharpness criterion construction from compositor evidence planes.
include!("quality_gate_runner/owner_sharpness.inc.rs");
// Top-level nine-criterion runtime orchestration.
include!("quality_gate_runner/runtime.inc.rs");

include!("quality_gate_runner/tests.inc.rs");
