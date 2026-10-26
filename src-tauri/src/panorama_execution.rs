//! Top-level image-stack execution facade.
//!
//! The command/API surface stays here while entry points, cache/output helpers,
//! and the execution pipeline live in focused sibling modules.

use super::*;

#[path = "panorama_execution_cache.rs"]
mod cache;
#[path = "panorama_execution_entry.rs"]
mod entry;
#[path = "panorama_execution_pipeline.rs"]
mod pipeline;

#[cfg(test)]
pub(super) use cache::write_alignment_cache;
pub(super) use cache::{
    FocusStackSourceDigests, focus_stack_cache_key_inputs, focus_stack_source_digests,
    is_auxiliary_stitch_file, is_generated_stitch_output, source_digest_hex,
};
pub(super) use entry::{save_panorama, stitch_panorama};
pub(crate) use pipeline::stitch_images_with_options;
