//! Feature matching facade.
//!
//! The implementation is split by responsibility so each production component
//! stays below the 1000-line maintenance limit.

use super::*;

#[path = "panorama_feature_candidates.rs"]
mod candidates;
pub(super) use candidates::*;

#[path = "panorama_feature_alignment_features.rs"]
mod alignment_features;
pub(super) use alignment_features::*;

#[path = "panorama_feature_pair_matching.rs"]
mod pair_matching;
pub(super) use pair_matching::*;

#[path = "panorama_feature_transform_fitting.rs"]
mod transform_fitting;
pub(super) use transform_fitting::*;

#[path = "panorama_feature_patch_refinement.rs"]
mod patch_refinement;
pub(super) use patch_refinement::*;
