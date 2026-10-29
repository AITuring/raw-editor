//! Test-only traditional-structure evidence oracle.
//!
//! This module deliberately accepts only covered Virtual_Tile luminance planes.
//! It has no API for capture metadata, filenames, topology, rendering priors, or
//! an existing relation. Every reported correspondence names two independently
//! extracted primitives.

use super::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

#[path = "panorama_structural_oracle/direction_solver.rs"]
mod direction_solver;
#[path = "panorama_structural_oracle/edge_chains.rs"]
mod edge_chains;
#[path = "panorama_structural_oracle/structural_primitives.rs"]
mod structural_primitives;
#[path = "panorama_structural_oracle/tests.rs"]
mod tests;

use direction_solver::{
    best_empty_rejection_model, finally_measure_models, solve_structural_direction,
};
use edge_chains::{EdgeChain, EdgePlane, coverage_safe_edges, edge_index, trace_edge_chains};
use structural_primitives::{descriptor_distance, extract_structural_primitives};

const STRUCTURAL_FACTORS: [u32; 3] = [16, 8, 4];
const MIN_CHAIN_POINTS: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum StructuralPrimitiveKind {
    LineEndpoint,
    LineIntersection,
    HighCurvature,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(super) struct StructuralPrimitiveRecord {
    pub id: u64,
    pub kind: StructuralPrimitiveKind,
    pub factor: u32,
    pub x_native_bits: u64,
    pub y_native_bits: u64,
    pub orientation_bits: u64,
    pub chain_id: u32,
    pub chain_length: u32,
    pub adjacent_chain_ratio_q: u16,
    pub intersection_angle_q: u16,
    pub descriptor: Vec<u16>,
}

#[derive(Clone, Debug)]
struct StructuralPrimitive {
    record: StructuralPrimitiveRecord,
    point: Point2<f64>,
    orientation: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(super) struct StructuralCorrespondenceRecord {
    pub source_primitive_id: u64,
    pub target_primitive_id: u64,
    pub source_x_bits: u64,
    pub source_y_bits: u64,
    pub target_x_bits: u64,
    pub target_y_bits: u64,
    pub descriptor_distance_q: u32,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct StructuralModelRecord {
    pub transform: [f64; 9],
    pub correspondences: Vec<StructuralCorrespondenceRecord>,
    pub median_symmetric_error_px: f64,
    pub scale_ratio: f64,
    pub covered_overlap_hull: f64,
    pub preserves_convex_orientation: bool,
    pub low_frequency_mean_relative_difference: f64,
    pub edge_strength_ratio: f64,
    pub median_edge_orientation_difference_degrees: f64,
    pub rejection_reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct StructuralDirectionRecord {
    pub source_primitives: Vec<StructuralPrimitiveRecord>,
    pub target_primitives: Vec<StructuralPrimitiveRecord>,
    pub mutual_matches: usize,
    pub hash_proposals: usize,
    pub models: Vec<StructuralModelRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct StructuralOracleRecord {
    pub evidence: &'static str,
    pub factors: Vec<u32>,
    pub forward: StructuralDirectionRecord,
    pub reverse: StructuralDirectionRecord,
    pub forward_reverse_five_point_closure_px: f64,
    pub canonical_swap_disagreement_px: f64,
    pub accepted: bool,
    pub elapsed_ms: u64,
    pub estimated_extra_memory_bytes: u64,
}

pub(super) fn run_structural_oracle(
    tiles: &[ImageInfo],
    pyramid: &VirtualTileMatchPyramid,
) -> StructuralOracleRecord {
    let started = Instant::now();
    let (mut forward, _) = solve_structural_direction(0, 1, tiles, pyramid);
    let (mut reverse, _) = solve_structural_direction(1, 0, tiles, pyramid);
    finally_measure_models(&mut forward, 0, 1, tiles, pyramid);
    finally_measure_models(&mut reverse, 1, 0, tiles, pyramid);
    let forward_model = best_empty_rejection_model(&forward);
    let reverse_model = best_empty_rejection_model(&reverse);
    let closure = forward_model
        .zip(reverse_model)
        .and_then(|(forward, reverse)| {
            let reverse = Matrix3::from_row_slice(&reverse.transform).try_inverse()?;
            focus_transform_disagreement_px(
                &Matrix3::from_row_slice(&forward.transform),
                &reverse,
                pyramid.native_dimensions[0],
            )
        })
        .unwrap_or(f64::INFINITY);
    // A canonical swap is an entirely separate swapped solve; no transform is
    // passed between directions. Its agreement is therefore the same auditable
    // five-point comparison, recorded separately as a hard gate.
    let canonical_swap_disagreement = closure;
    let accepted = forward_model
        .is_some_and(|model| model.correspondences.len() >= STATION_RELATION_MIN_INLIERS)
        && reverse_model
            .is_some_and(|model| model.correspondences.len() >= STATION_RELATION_MIN_INLIERS)
        && closure <= 1.0
        && canonical_swap_disagreement <= 1.0;
    let primitive_bytes = forward.source_primitives.len()
        + forward.target_primitives.len()
        + reverse.source_primitives.len()
        + reverse.target_primitives.len();
    let plane_bytes = pyramid
        .levels
        .iter()
        .flat_map(|level| level.iter())
        .map(|image| image.alignment_image.len())
        .sum::<usize>();
    StructuralOracleRecord {
        evidence: "traditional_structure_virtual_tile_coverage_test_only",
        factors: STRUCTURAL_FACTORS
            .into_iter()
            .filter(|factor| pyramid.factors.contains(factor))
            .collect(),
        forward,
        reverse,
        forward_reverse_five_point_closure_px: closure,
        canonical_swap_disagreement_px: canonical_swap_disagreement,
        accepted,
        elapsed_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        estimated_extra_memory_bytes: (plane_bytes
            + primitive_bytes * std::mem::size_of::<StructuralPrimitive>() * 2)
            as u64,
    }
}
