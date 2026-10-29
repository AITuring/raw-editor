use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::*;
use crate::cache_utils::BudgetedCache;

mod composition_and_cache;
mod large_mask_harnesses;
mod scratch_bounds;
mod tiled_parity;

fn test_sub_mask(mask_type: &str, parameters: Value, mode: SubMaskMode) -> SubMask {
    SubMask {
        id: format!("test-{mask_type}"),
        mask_type: mask_type.to_string(),
        visible: true,
        invert: true,
        opacity: 63.0,
        mode,
        parameters,
    }
}
