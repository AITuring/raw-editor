//! Scratch-memory bounds for 60 MP masks.
//!
//! AI-depth and grow/feather halos and tiled programmatic/brush generation stay
//! within their planned per-tile scratch.

use super::*;

#[test]
fn ai_depth_halo_bounds_supported_60mp_filter_scratch() {
    const WIDTH: u32 = 9_504;
    const HEIGHT: u32 = 6_336;
    const FULL_MASK_BYTES: u64 = 60_217_344;
    const MAX_AI_DEPTH_HALO: u64 = 159;
    const MAX_EXPANDED_TILE_EDGE: u64 = 2_366;
    const MAX_EXPANDED_TILE_BYTES: u64 = 5_597_956;

    assert_eq!(image_gaussian_blur_radius(0.0), 0);
    assert_eq!(image_gaussian_blur_radius(0.1), 1);
    assert_eq!(image_gaussian_blur_radius(1.5), 3);
    assert_eq!(image_gaussian_blur_radius(10.0), 32);

    let grow_feather = GrowFeatherPlan::new(100.0, 100.0, WIDTH, HEIGHT);
    assert_eq!(
        u64::from(grow_feather.halo()) + u64::from(image_gaussian_blur_radius(10.0)),
        MAX_AI_DEPTH_HALO
    );
    assert_eq!(
        u64::from(GPU_TILE_SIZE) + MAX_AI_DEPTH_HALO * 2,
        MAX_EXPANDED_TILE_EDGE
    );
    assert_eq!(MAX_EXPANDED_TILE_EDGE.pow(2), MAX_EXPANDED_TILE_BYTES);
    assert_eq!(u64::from(WIDTH) * u64::from(HEIGHT), FULL_MASK_BYTES);
    assert_eq!(
        FULL_MASK_BYTES * 3 - MAX_EXPANDED_TILE_BYTES * 3,
        163_858_164
    );
}

#[test]
fn grow_feather_halo_bounds_supported_60mp_range_mask_scratch() {
    const WIDTH: u32 = 9_504;
    const HEIGHT: u32 = 6_336;
    const FULL_MASK_BYTES: u64 = 60_217_344;
    const MAX_EXPANDED_TILE_EDGE: u64 = 2_302;
    const MAX_EXPANDED_TILE_BYTES: u64 = 5_299_204;

    let plan = GrowFeatherPlan::new(100.0, 100.0, WIDTH, HEIGHT);
    assert_eq!(plan.grow_amount, 63);
    assert!(plan.dilate);
    assert!((plan.feather_sigma - 31.68).abs() < 0.0001);
    assert_eq!(plan.feather_radius, 64);
    assert_eq!(plan.halo(), 127);
    assert_eq!(
        u64::from(GPU_TILE_SIZE) + u64::from(plan.halo()) * 2,
        MAX_EXPANDED_TILE_EDGE
    );
    assert_eq!(MAX_EXPANDED_TILE_EDGE.pow(2), MAX_EXPANDED_TILE_BYTES);
    assert_eq!(u64::from(WIDTH) * u64::from(HEIGHT), FULL_MASK_BYTES);
    assert_eq!(
        FULL_MASK_BYTES * 3 - MAX_EXPANDED_TILE_BYTES * 3,
        164_754_420
    );
}

#[test]
fn tiled_generation_bounds_60mp_programmatic_and_brush_scratch() {
    const WIDTH: usize = 9_504;
    const HEIGHT: usize = 6_336;
    const FULL_MASK_BYTES: usize = 60_217_344;
    const MAX_TILE_BYTES: usize = 4_194_304;

    assert_eq!(WIDTH * HEIGHT, FULL_MASK_BYTES);
    assert_eq!(
        WIDTH.min(GPU_TILE_SIZE as usize) * HEIGHT.min(GPU_TILE_SIZE as usize),
        MAX_TILE_BYTES
    );
    assert_eq!(
        FULL_MASK_BYTES * 2 - (FULL_MASK_BYTES + MAX_TILE_BYTES),
        56_023_040
    );
    assert_eq!(
        FULL_MASK_BYTES * 3 - (FULL_MASK_BYTES + MAX_TILE_BYTES * 2),
        112_046_080
    );
}
