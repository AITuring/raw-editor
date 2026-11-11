//! Output-resolution focus evidence for focus-scan ownership.
use super::*;

/// Fine silk-like weave: a two-pixel thread pattern with slow slub variation.
fn weave(width: u32, height: u32) -> Rgb32FImage {
    Rgb32FImage::from_fn(width, height, |x, y| {
        let thread = if (x + y) % 2 == 0 { 0.06 } else { -0.06 };
        let slub = 0.04 * ((x as f32 * 0.031).sin() + (y as f32 * 0.027).cos());
        let value = 0.42 + thread + slub;
        Rgb([value * 1.08, value, value * 0.86])
    })
}

fn store_with(canvas: &Rgb32FImage) -> StreamingMosaicStore {
    let store = StreamingMosaicStore::new(canvas.width(), canvas.height()).expect("store");
    for row in 0..store.tile_rows {
        for column in 0..store.tile_columns {
            let (left, top, width, height) = store.tile_extent(column, row);
            let tile =
                Rgb32FImage::from_fn(width, height, |x, y| *canvas.get_pixel(left + x, top + y));
            let covered = GrayImage::from_pixel(width, height, Luma([255]));
            store.save_tile(column, row, &tile, &covered).expect("tile");
        }
    }
    store
}

fn layer_sampler<'a>(
    info: &'a ImageInfo,
    source: &'a Rgb32FImage,
    scale: f64,
    analysis_width: u32,
    analysis_height: u32,
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
        scale,
        residual: Field::new(analysis_width, analysis_height, GRID_STEP as f64),
    }
}

#[test]
fn output_focus_separates_a_defocused_bracket_the_analysis_grid_cannot_resolve() {
    // A 2400x1200 layer is analysed at 800x400: one analysis sample per three
    // output pixels, so the weave is only aliasing on the analysis grid.
    let (width, height) = (2400u32, 1200u32);
    let sharp = weave(width, height);
    let defocused = image::imageops::blur(&sharp, 1.4);
    let info = image_info(1, &sharp);
    let (analysis_width, analysis_height) = (800, 400);
    let grid = streaming_ownership_grid(analysis_width, analysis_height);
    let layer = (0, width - 1, 0, height - 1);
    let blurred_canvas = store_with(&defocused);
    let sharp_candidate = layer_sampler(&info, &sharp, 3.0, analysis_width, analysis_height);
    let focus = measure_output_focus(
        &blurred_canvas,
        &sharp_candidate,
        layer,
        grid,
        FocusBand::Pixel,
    )
    .expect("focus");
    assert_eq!(focus.base.len(), (grid.0 * grid.1) as usize);
    let measured = (0..focus.base.len())
        .filter(|&index| focus.base[index].is_finite() && focus.candidate[index].is_finite())
        .collect::<Vec<_>>();
    assert!(
        measured.len() * 10 > focus.base.len() * 9,
        "nearly every cell is measured"
    );
    let sharper = measured
        .iter()
        .filter(|&&index| focus.candidate[index] > focus.base[index] * 1.5)
        .count();
    assert!(
        sharper * 100 >= measured.len() * 99,
        "the sharp weave must measure sharper in every cell: {sharper}/{}",
        measured.len()
    );
    // The reverse pairing measures the reverse, so the comparison is about
    // the pixels and not about which side is the canvas.
    let sharp_canvas = store_with(&sharp);
    let defocused_candidate =
        layer_sampler(&info, &defocused, 3.0, analysis_width, analysis_height);
    let reverse = measure_output_focus(
        &sharp_canvas,
        &defocused_candidate,
        layer,
        grid,
        FocusBand::Pixel,
    )
    .expect("focus");
    let softer = measured
        .iter()
        .filter(|&&index| reverse.candidate[index] * 1.5 < reverse.base[index])
        .count();
    assert!(
        softer * 100 >= measured.len() * 99,
        "{softer}/{}",
        measured.len()
    );
}

#[test]
fn output_focus_ignores_pixels_only_one_source_covers() {
    let (width, height) = (1200u32, 600u32);
    let sharp = weave(width, height);
    let info = image_info(1, &sharp);
    let canvas = StreamingMosaicStore::new(width, height).expect("store");
    // Only the left half of the canvas is covered.
    for row in 0..canvas.tile_rows {
        for column in 0..canvas.tile_columns {
            let (left, top, tile_width, tile_height) = canvas.tile_extent(column, row);
            let tile = Rgb32FImage::from_fn(tile_width, tile_height, |x, y| {
                *sharp.get_pixel(left + x, top + y)
            });
            let covered = GrayImage::from_fn(tile_width, tile_height, |x, _| {
                Luma([if left + x < width / 2 { 255 } else { 0 }])
            });
            canvas
                .save_tile(column, row, &tile, &covered)
                .expect("tile");
        }
    }
    let sampler = layer_sampler(&info, &sharp, 1.0, width, height);
    let grid = streaming_ownership_grid(width, height);
    let focus = measure_output_focus(
        &canvas,
        &sampler,
        (0, width - 1, 0, height - 1),
        grid,
        FocusBand::Pixel,
    )
    .expect("focus");
    let columns = grid.0 as usize;
    for (index, value) in focus.base.iter().enumerate() {
        let column = index % columns;
        if column > columns * 3 / 4 {
            assert!(
                value.is_nan(),
                "cell {index} has no covered canvas and must not be measured"
            );
        }
    }
    assert!(focus.base.iter().any(|value| value.is_finite()));
}
