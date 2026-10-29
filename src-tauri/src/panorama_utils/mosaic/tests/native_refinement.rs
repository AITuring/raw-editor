//! Native-resolution sampling, control point and intra-station refinement tests.

use super::*;

#[test]
fn native_control_cell_can_find_a_corner_missed_by_the_regular_probe_phase() {
    let image = Rgb32FImage::from_fn(128, 128, |x, y| {
        Rgb([if x >= 56 && y >= 56 { 0.9 } else { 0.1 }; 3])
    });
    let mask = GrayImage::from_pixel(128, 128, Luma([255]));
    let info = image_info(0, &image);
    let sampler = LayerSampler {
        info: &info,
        source: &image,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(128, 128, 48.0),
    };
    let base = NativeRefineBase {
        image: &image,
        mask: &mask,
        left: 0,
        top: 0,
    };
    assert_eq!(
        contrasted_native_control_location(&base, &sampler, 52, 52, 128, 128),
        (56, 56)
    );
    let uncovered = GrayImage::new(128, 128);
    let base = NativeRefineBase {
        mask: &uncovered,
        ..base
    };
    assert_eq!(
        contrasted_native_control_location(&base, &sampler, 52, 52, 128, 128),
        (52, 52)
    );
}

#[test]
fn only_near_black_pixels_in_the_lower_left_capture_border_are_rejected() {
    let image = Rgb32FImage::new(100, 100);
    assert!(is_lower_left_black_capture_border(
        &image,
        10.0,
        90.0,
        Rgb([0.005, 0.006, 0.004]),
    ));
    assert!(!is_lower_left_black_capture_border(
        &image,
        50.0,
        90.0,
        Rgb([0.005, 0.006, 0.004]),
    ));
    assert!(!is_lower_left_black_capture_border(
        &image,
        10.0,
        90.0,
        Rgb([0.08, 0.03, 0.02]),
    ));
    // A pure-black lower-left sample remains non-photographic even at the
    // extreme source edge. Keeping it created the reported black block.
    assert!(is_lower_left_black_capture_border(
        &image,
        10.0,
        98.0,
        Rgb([0.005, 0.006, 0.004]),
    ));
}

/// Aperiodic paper grain.  A sum of sines would be rejected by
/// `refine_warped_patch`'s second-peak guard for looking like a repeated
/// stroke, which is the guard doing its job rather than a refinement failure.
fn refine_texture(width: u32, height: u32) -> Rgb32FImage {
    let grain = Rgb32FImage::from_fn(width, height, |x, y| {
        let hash = x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
        let hash = (hash ^ (hash >> 15)).wrapping_mul(0x7feb_352d);
        let value = 0.25 + 0.50 * ((hash >> 8) & 0xffff) as f32 / 65_535.0;
        Rgb([value, value * 0.96, value * 0.90])
    });
    image::imageops::blur(&grain, 1.1)
}

/// The intra-station gates must leave usable control points.
///
/// Without this the native refinement could reject every control point and
/// still look healthy from the outside: the frame simply reverts to its
/// global model, the output is byte identical to a run with no refinement at
/// all, and nothing fails.  That is exactly what a matching window radius of
/// 48 inside a 57×57 patch buffer does — `refine_warped_patch` needs a sample
/// at −20 and returns `None` for every position.
#[test]
fn intra_station_native_refinement_accepts_control_points() {
    let base = refine_texture(256, 256);
    // The layer is the same photograph displaced by a few native pixels,
    // which is what focus breathing does inside one Capture_Station.
    // Two native pixels: inside 需求 2.6's `0.01 × 256` error ceiling for this
    // fixture, so the gate under test is the matching, not the ceiling.
    let shifted = Rgb32FImage::from_fn(256, 256, |x, y| *base.get_pixel((x + 2).min(255), y));
    let info = image_info(1, &shifted);
    let mask = GrayImage::from_pixel(256, 256, Luma([255]));
    let mut sampler = LayerSampler {
        info: &info,
        source: &shifted,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(256, 256, GRID_STEP as f64),
    };
    let outcome = refine_native_layer(
        &base,
        &mask,
        &mut sampler,
        256,
        256,
        NativeRefineMode::IntraStation {
            anchor_long_side: 256,
        },
    );
    assert!(
        !outcome.skipped,
        "需求 2.2 has no analysis-scale early exit"
    );
    assert!(
        outcome.evaluated_control_points >= 64,
        "the control point grid must reach the layer ({} evaluated)",
        outcome.evaluated_control_points
    );
    assert!(
        outcome.accepted_control_points >= 16,
        "the 需求 2.4/2.5/2.6 gates must accept real matches, not reject every one ({}/{} accepted)",
        outcome.accepted_control_points,
        outcome.evaluated_control_points
    );
    assert!(
        outcome.inlier_area_coverage > 0.0,
        "需求 2.8 coverage must be measurable once control points are accepted"
    );
    // 需求 2.7 keeps a local field that lowered the frame's median symmetric
    // reprojection error.  The baseline is the error of the global model, so a
    // frame the refinement actually fixed must not be reverted.
    assert!(
        !outcome.reverted_to_global,
        "a refinement that removed a real 2px displacement must be kept (global {:.2}px, local {:.2}px)",
        outcome.global_median_symmetric_error_px, outcome.local_median_symmetric_error_px
    );
    // The comparison has to be made against a baseline that exists.  The
    // regression this guards reverted every frame because the coarse pass
    // accepted too few control points, `median_of(&[])` returned `0.0`, and
    // no refinement can be below zero.
    assert!(
        outcome.global_baseline_samples >= intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES,
        "需求 2.7 needs a real global baseline, not an empty median ({} samples)",
        outcome.global_baseline_samples
    );
    assert_eq!(
        outcome.local_field_verdict,
        intra_station::LocalFieldVerdict::Kept
    );
    assert_eq!(outcome.frame_status(), IntraStationFrameStatus::Local);
}

/// 需求 2.2 caps the *actual* intra-station analysis resolution at 2048
/// while the production compositors currently configure 1400. This also
/// guards Stack_Report's source value: both paths share the registrar's
/// actual configuration instead of reporting the ceiling.
#[test]
fn the_analysis_budgets_stay_inside_the_intra_station_bound() {
    const {
        assert!(
            intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE
                <= intra_station::INTRA_STATION_ANALYSIS_MAX_LONG_SIDE
        );
        assert!(ANALYSIS_LONG_SIDE == intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE);
        assert!(STREAMING_ANALYSIS_LONG_SIDE == intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE);
        assert!(ANALYSIS_LONG_SIDE <= intra_station::INTRA_STATION_ANALYSIS_MAX_LONG_SIDE);
        assert!(
            STREAMING_ANALYSIS_LONG_SIDE <= intra_station::INTRA_STATION_ANALYSIS_MAX_LONG_SIDE
        );
    };
}

/// The tiled compositor is the *only* path a multi-station stack ever takes,
/// so 需求 2.3–2.9 has to be measurable there.  Before this the native
/// refinement existed solely on the in-memory path, and every real focus
/// stack — three camera positions, an 8369×10114 canvas — skipped it
/// silently and produced output byte identical to a run without it.
#[test]
fn streaming_intra_station_refinement_measures_across_tiles() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let width = 1_200u32;
    let height = 256u32;
    let base = refine_texture(width, height);
    let store = StreamingMosaicStore::new(width, height).expect("streaming store");
    assert!(
        store.tile_columns > 1,
        "the fixture must straddle a tile boundary"
    );
    for tile_row in 0..store.tile_rows {
        for tile_column in 0..store.tile_columns {
            let (left, top, tile_width, tile_height) = store.tile_extent(tile_column, tile_row);
            let tile = Rgb32FImage::from_fn(tile_width, tile_height, |x, y| {
                *base.get_pixel(left + x, top + y)
            });
            let covered = GrayImage::from_pixel(tile_width, tile_height, Luma([255]));
            let owner = GrayImage::from_pixel(tile_width, tile_height, Luma([1]));
            store
                .save_tile(tile_column, tile_row, &tile, &covered)
                .expect("tile");
            store
                .save_owner_tile(tile_column, tile_row, &owner)
                .expect("owner tile");
        }
    }
    let shifted = Rgb32FImage::from_fn(width, height, |x, y| {
        *base.get_pixel((x + 2).min(width - 1), y)
    });
    let info = image_info(1, &shifted);
    fn layer_sampler<'a>(
        source: &'a Rgb32FImage,
        info: &'a ImageInfo,
        width: u32,
        height: u32,
    ) -> LayerSampler<'a> {
        LayerSampler {
            info,
            source,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(width, height, GRID_STEP as f64),
        }
    }
    let mode = NativeRefineMode::IntraStation {
        anchor_long_side: 256,
    };
    let mut sampler = layer_sampler(&shifted, &info, width, height);
    let outcome = refine_native_layer_streaming(
        &store,
        Some(1),
        &mut sampler,
        width,
        height,
        (0, width - 1, 0, height - 1),
        mode,
    )
    .expect("streaming native refinement");
    assert!(
        outcome.accepted_control_points >= 16,
        "the tiled path must measure real control points ({}/{} accepted)",
        outcome.accepted_control_points,
        outcome.evaluated_control_points
    );
    assert!(!outcome.reverted_to_global);
    assert!(outcome.inlier_area_coverage > 0.20);
    // 需求 2.2: a later focal plane registers against its own camera
    // position, so canvas owned by another station is not a base at all.
    let mut foreign = layer_sampler(&shifted, &info, width, height);
    let ignored = refine_native_layer_streaming(
        &store,
        Some(7),
        &mut foreign,
        width,
        height,
        (0, width - 1, 0, height - 1),
        mode,
    )
    .expect("streaming native refinement");
    assert_eq!(ignored.accepted_control_points, 0);
    assert_eq!(ignored.evaluated_control_points, 0);
}

#[test]
fn native_registration_observes_the_final_depth_band_inverse() {
    let source = Rgb32FImage::new(128, 128);
    let info = image_info(1, &source);
    let mapping = |target: Point3<f64>| Some(Point2::new(target.x + 7.0, target.y - 3.0));
    let mut residual = Field::new(64, 64, 16.0);
    residual.values.fill([1.0, 2.0]);
    let sampler = LayerSampler {
        info: &info,
        source: &source,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: Some(&mapping),
        projection: Projection::Planar,
        offset: (4.0, 5.0),
        left: 10,
        top: 20,
        scale: 2.0,
        residual,
    };
    // Canvas residual precedes the same depth map as final rendering:
    // (10+30+2-4, 20+40+4-5) then (+7,-3).
    assert_eq!(
        sampler.source_point(30.0, 40.0),
        Some(Point2::new(45.0, 56.0))
    );
    let global = LayerSampler {
        native_mapping: None,
        ..sampler
    };
    assert_eq!(
        global.source_point(30.0, 40.0),
        Some(Point2::new(38.0, 59.0))
    );
}

#[test]
fn displacement_field_interpolation_is_continuous_at_cell_edges() {
    let mut field = Field::<2>::new(200, 200, 48.0);
    for (i, value) in field.values.iter_mut().enumerate() {
        *value = [
            (i % field.width) as f64 * 0.3,
            (i / field.width) as f64 * -0.2,
        ];
    }
    let a = field.at(48.0 - 1e-5, 73.0);
    let b = field.at(48.0 + 1e-5, 73.0);
    assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
}
