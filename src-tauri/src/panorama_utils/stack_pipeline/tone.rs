//! Low-frequency affine tone harmonisation for owned focus-stack pixels.
//!
//! Tone correction is deliberately separated from ownership.  The owner image
//! supplies the high-frequency residual while a low-pass version of the source
//! supplies only broad exposure and white-balance changes.  This keeps strokes,
//! edges, and texture tied to the source selected by the ownership map.

#![allow(dead_code)]

use image::{GrayImage, Luma, Rgb, Rgb32FImage};
use rayon::prelude::*;
use std::sync::Mutex;

use super::degradation;
use super::report::{
    TONE_BOUNDARY_DELTA_E_THRESHOLD, ToneBoundaryViolationRecord, TonePairWithoutEvidenceRecord,
    ToneReport, ToneStatus, ToneTileRecord,
};
use crate::panorama_utils::photometric::{PairwiseDifference, solve_pairwise_differences};

include!("tone/model.inc.rs");
include!("tone/solver.inc.rs");
include!("tone/render.inc.rs");
include!("tone/diagnostics.inc.rs");
include!("tone/tests.inc.rs");
