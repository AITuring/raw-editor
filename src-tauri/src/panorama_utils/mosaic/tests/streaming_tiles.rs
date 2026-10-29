//! Streaming tile store, coverage bounds and materialisation tests.

use super::*;

#[test]
fn streaming_store_round_trips_across_tile_boundaries() {
    let store = StreamingMosaicStore::new(2050, 1025).expect("tile store should initialize");
    for row in 0..store.tile_rows {
        for column in 0..store.tile_columns {
            let (_, _, width, height) = store.tile_extent(column, row);
            let value = (column + row * 3) as f32;
            let rgb =
                Rgb32FImage::from_pixel(width, height, Rgb([value, value + 1.0, value + 2.0]));
            let mask = GrayImage::from_pixel(width, height, Luma([255]));
            store
                .save_tile(column, row, &rgb, &mask)
                .expect("tile should be writable");
        }
    }
    let crop = store
        .covered_bounds()
        .expect("coverage should be readable")
        .expect("full tile fixture should be covered");
    assert_eq!(crop, (0, 0, 2050, 1025));
    let output = store.materialize(crop).expect("tiles should materialize");
    assert_eq!(output.dimensions(), (2050, 1025));
    assert_eq!(*output.get_pixel(1023, 10), Rgb([0.0, 1.0, 2.0]));
    assert_eq!(*output.get_pixel(1024, 10), Rgb([1.0, 2.0, 3.0]));
    assert_eq!(*output.get_pixel(10, 1024), Rgb([3.0, 4.0, 5.0]));
}

#[test]
fn covered_bounds_preserve_non_rectangular_source_edges() {
    let mask = GrayImage::from_fn(8, 6, |x, y| {
        Luma([u8::from((x == 1 && y == 3) || (x >= 3 && x <= 6 && y >= 1 && y <= 4)) * 255])
    });
    assert_eq!(mask_covered_bounds(&mask), Some((1, 1, 6, 4)));
}

#[test]
fn streaming_covered_bounds_do_not_reduce_to_largest_rectangle() {
    let store = StreamingMosaicStore::new(12, 8).expect("tile store should initialize");
    let rgb = Rgb32FImage::new(12, 8);
    let mask = GrayImage::from_fn(12, 8, |x, y| {
        Luma([u8::from((x == 1 && y == 6) || (x >= 4 && x <= 10 && y >= 1 && y <= 5)) * 255])
    });
    store
        .save_tile(0, 0, &rgb, &mask)
        .expect("tile should be writable");
    assert_eq!(
        store.covered_bounds().expect("coverage should be readable"),
        Some((1, 1, 10, 6))
    );
}

#[test]
fn streaming_decision_alpha_interpolates_between_cells() {
    let decision = GrayImage::from_fn(2, 2, |x, y| Luma([if (x + y) % 2 == 0 { 0 } else { 255 }]));
    let centre = sample_decision_alpha(&decision, 0.5, 0.5);
    assert!((centre - 0.5).abs() < 1e-6);
    let left = sample_decision_alpha(&decision, 0.05, 0.5);
    let right = sample_decision_alpha(&decision, 0.95, 0.5);
    assert!(left > 0.45 && left < 0.55);
    assert!(right > 0.45 && right < 0.55);
    assert!(sample_decision_alpha(&decision, 0.0, 0.0) < 1e-6);
    assert!((sample_decision_alpha(&decision, 1.0, 0.0) - 1.0).abs() < 1e-6);
}

#[test]
fn streaming_materialize_preserves_a_natural_gradient_at_owner_boundary() {
    let store = StreamingMosaicStore::new(256, 32).expect("tile store should initialize");
    let rgb = Rgb32FImage::from_fn(256, 32, |x, _| {
        let value = 0.30 + 0.40 * x as f32 / 255.0;
        Rgb([value, value * 0.98, value * 0.94])
    });
    let mask = GrayImage::from_pixel(256, 32, Luma([255]));
    let owner = GrayImage::from_fn(256, 32, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
    store
        .save_tile(0, 0, &rgb, &mask)
        .expect("tile should be writable");
    store
        .save_owner_tile(0, 0, &owner)
        .expect("owner should be writable");
    let output = materialize_streaming_focus_output(&store, (0, 0, 256, 32))
        .expect("gradient should materialize");
    let left = luma(*output.get_pixel(126, 16));
    let right = luma(*output.get_pixel(129, 16));
    let expected = luma(*rgb.get_pixel(126, 16)) - luma(*rgb.get_pixel(129, 16));
    assert!((left - right).abs() < expected.abs() * 1.4);
    assert!((left - luma(*rgb.get_pixel(126, 16))).abs() < 1e-6);
    assert!((right - luma(*rgb.get_pixel(129, 16))).abs() < 1e-6);
}

#[test]
fn streaming_analysis_region_is_continuous_across_tile_boundaries() {
    let store = StreamingMosaicStore::new(2050, 32).expect("tile store should initialize");
    for column in 0..store.tile_columns {
        let (_, _, width, height) = store.tile_extent(column, 0);
        let value = column as f32 + 1.0;
        let rgb = Rgb32FImage::from_pixel(width, height, Rgb([value, value, value]));
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        store
            .save_tile(column, 0, &rgb, &mask)
            .expect("tile should be writable");
    }

    let (analysis, mask, owner) = store
        .analysis_region(1018, 4, 2.0, 8, 4)
        .expect("analysis region should be readable");
    assert_eq!(analysis.dimensions(), (8, 4));
    assert_eq!(*analysis.get_pixel(2, 1), Rgb([1.0, 1.0, 1.0]));
    assert_eq!(*analysis.get_pixel(3, 1), Rgb([2.0, 2.0, 2.0]));
    assert!(mask.as_raw().iter().all(|&value| value == 255));
    assert!(owner.as_raw().iter().all(|&value| value == 0));
}

#[test]
fn streaming_uncovered_fill_preserves_owner_and_closes_real_source_holes() {
    let source = Rgb32FImage::from_pixel(16, 8, Rgb([0.8, 0.7, 0.6]));
    let info = image_info(1, &source);
    let mut tile = Rgb32FImage::from_pixel(16, 8, Rgb([0.2, 0.2, 0.2]));
    let mut mask = GrayImage::from_pixel(16, 8, Luma([255]));
    let mut owner = GrayImage::from_pixel(16, 8, Luma([3]));
    mask.put_pixel(5, 3, Luma([0]));
    owner.put_pixel(5, 3, Luma([0]));
    let sampler = LayerSampler {
        info: &info,
        source: &source,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(16, 8, 32.0),
    };
    fill_streaming_uncovered_tile_region(
        &mut tile,
        &mut mask,
        &mut owner,
        4,
        &sampler,
        &Field::new(16, 8, 32.0),
        0,
        0,
        0,
        15,
        0,
        7,
        1.0,
    );

    assert_eq!(*tile.get_pixel(4, 3), Rgb([0.2, 0.2, 0.2]));
    assert_eq!(*tile.get_pixel(5, 3), Rgb([0.8, 0.7, 0.6]));
    assert_eq!(mask.get_pixel(5, 3)[0], 255);
    assert_eq!(owner.get_pixel(4, 3)[0], 3);
    assert_eq!(owner.get_pixel(5, 3)[0], 4);
}
