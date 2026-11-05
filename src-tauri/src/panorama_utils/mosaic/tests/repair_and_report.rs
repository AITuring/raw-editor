//! Detail repair of a composed focus scan, its dense alignment and the
//! soft-region report.
use super::*;

/// Silk-like texture: a weave of about four pixels over blobs a low-pass
/// keeps. Canvas position `p` shows the content of `p - shift`.
fn silk(width: u32, height: u32, shift: (i64, i64)) -> Rgb32FImage {
    Rgb32FImage::from_fn(width, height, |x, y| {
        let (u, v) = (
            (i64::from(x) - shift.0) as f32,
            (i64::from(y) - shift.1) as f32,
        );
        let thread = 0.06 * (u * 1.8).sin() * (v * 1.6).sin();
        let blobs =
            0.06 * (u * 0.09).sin() * (v * 0.07).cos() + 0.04 * ((u + 2.0 * v) * 0.031).sin();
        let value = 0.42 + thread + blobs;
        Rgb([value * 1.08, value, value * 0.86])
    })
}

fn luma_plane(image: &Rgb32FImage) -> Plane {
    let mut plane = Plane::new(image.width() as usize, image.height() as usize);
    for (value, pixel) in plane.values.iter_mut().zip(image.pixels()) {
        *value = luma(*pixel) as f32;
    }
    plane
}

/// Repairs `output` from `donor`, the only frame, at the canvas pose.
fn repair_from(output: &mut Rgb32FImage, donor: &Rgb32FImage) -> RepairOutcome {
    let (width, height) = output.dimensions();
    let info = image_info(0, donor);
    let grid = DetailGrid::with_cells(
        width,
        height,
        FocusBand::Pixel,
        REPAIR_CELLS_LONG_SIDE,
        REPAIR_MIN_CELL,
    );
    let sampler = LayerSampler {
        info: &info,
        source: donor,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(1, 1, GRID_STEP as f64),
    };
    let details = vec![measure_frame_detail(
        &sampler,
        (0, width - 1, 0, height - 1),
        &grid,
    )];
    let homographies = HashMap::from([(0usize, Matrix3::identity())]);
    let images = [&info];
    let donor_frames = std::collections::HashSet::new();
    let scene = RepairScene {
        images: &images,
        donor_frames: &donor_frames,
        homographies: &homographies,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        canvas: (width, height),
    };
    repair_detail(
        output,
        (0, 0),
        &scene,
        &grid,
        &details,
        &mut |_: &ImageInfo| Ok::<_, String>(donor.clone()),
        |_, _| {},
    )
    .expect("repair")
}

#[test]
fn dense_alignment_recovers_a_known_displacement() {
    let margin = 48usize;
    let base = luma_plane(&silk(320, 320, (0, 0)));
    // The donor shows the canvas content of p at p + (7, -5); its view starts
    // `margin` pixels before the window on every side.
    let side = 320 + 2 * margin as u32;
    let moving = luma_plane(&silk(side, side, (margin as i64 + 7, margin as i64 - 5)));
    let displacement = align(&base, &moving, margin).expect("aligned");
    // Its affine fit, the fallback where a field folds, holds the same move.
    let smooth = displacement.affine_fit().expect("affine fit");
    for (x, y) in [(80.0, 80.0), (160.0, 160.0), (240.0, 200.0)] {
        for field in [&displacement, &smooth] {
            let [dx, dy] = field.at(x, y);
            assert!(
                (dx - 7.0).abs() < 0.6 && (dy + 5.0).abs() < 0.6,
                "displacement at {x},{y}: {dx:.2},{dy:.2}"
            );
        }
    }
    assert!(smooth.max_strain((0, 0, 319, 319)) < 0.01);
}

#[test]
fn detail_repair_restores_a_defocused_band_across_windows() {
    // A band longer than one repair window, so its cells are split over
    // several windows; every one must still be measured whole and repaired.
    let (width, height) = (2048u32, 480u32);
    let sharp = silk(width, height, (0, 0));
    let blurred = image::imageops::blur(&sharp, 2.0);
    let mut output = sharp.clone();
    for y in 180..300 {
        for x in 200..1800 {
            output.put_pixel(x, y, *blurred.get_pixel(x, y));
        }
    }
    let outcome = repair_from(&mut output, &sharp);
    assert!(outcome.cells.len() >= 40, "{} cells", outcome.cells.len());
    for cell in &outcome.cells {
        assert_eq!(cell.outcome, CellOutcome::Repaired, "{cell:?}");
        assert!(cell.after >= REPAIR_DEFICIT_RATIO * cell.best, "{cell:?}");
    }
    // Away from the band the canvas keeps its pixels.
    assert_eq!(output.get_pixel(1000, 30), sharp.get_pixel(1000, 30));
}

#[test]
fn detail_repair_replaces_a_displaced_soft_patch_with_the_aligned_donor() {
    let (width, height) = (960u32, 960u32);
    let sharp = silk(width, height, (0, 0));
    // A soft frame lying 35 px off holds the patch.
    let displaced = image::imageops::blur(&silk(width, height, (35, 0)), 2.0);
    let mut output = sharp.clone();
    for y in 400..560 {
        for x in 400..560 {
            output.put_pixel(x, y, *displaced.get_pixel(x, y));
        }
    }
    let outcome = repair_from(&mut output, &sharp);
    assert!(
        outcome
            .cells
            .iter()
            .any(|cell| cell.outcome == CellOutcome::Replaced),
        "{:?}",
        outcome.cells
    );
    for cell in &outcome.cells {
        assert!(
            matches!(cell.outcome, CellOutcome::Replaced | CellOutcome::Repaired),
            "{cell:?}"
        );
        assert!(cell.after >= REPAIR_DEFICIT_RATIO * cell.best, "{cell:?}");
    }
    // Inside its replaced cells the patch now holds the content around it,
    // not the displaced frame's (compared below the weave, which the
    // resampling softens a little).
    let (output_low, sharp_low) = (
        image::imageops::blur(&output, 3.0),
        image::imageops::blur(&sharp, 3.0),
    );
    let mut error = 0.0;
    for y in 440..520 {
        for x in 440..520 {
            error += (luma(*output_low.get_pixel(x, y)) - luma(*sharp_low.get_pixel(x, y))).abs();
        }
    }
    let error = error / (80.0 * 80.0);
    assert!(
        error < 0.01,
        "mean error {error:.4}: {:?}",
        outcome
            .cells
            .iter()
            .map(|cell| (cell.cell, cell.outcome))
            .collect::<Vec<_>>()
    );
}

#[test]
fn soft_report_circles_soft_silk_but_not_another_material_uneven_paper_or_clipped_white() {
    let (width, height) = (1600u32, 1600u32);
    let grid = DetailGrid::with_cells(width, height, FocusBand::Pixel, 32.0, 48.0);
    let (columns, rows) = grid.cell_dimensions();
    let count = columns * rows;
    let mut detail = vec![0.0; count];
    let mut level = vec![0.30; count];
    let mut clipped = vec![0.0; count];
    let mut best = vec![(0.0, 0usize); count];
    for cell in 0..count {
        let (x, y) = (cell % columns, cell / columns);
        // Silk with a little spread.
        detail[cell] = 0.20 + 0.01 * ((x * 7 + y * 3) % 5) as f64;
        best[cell] = (detail[cell] * 1.05, 0);
        if y >= 25 {
            // A smooth mount strip, a little brighter and nearly without
            // texture.
            level[cell] = 0.42;
            detail[cell] = 0.05;
            best[cell] = (0.052, 0);
        }
        if (2..6).contains(&x) && (20..24).contains(&y) {
            // White clipped by the exposure.
            level[cell] = 0.99;
            detail[cell] = 0.0;
            clipped[cell] = 0.6;
            best[cell] = (0.0, 0);
        }
        if (10..13).contains(&x) && (10..13).contains(&y) {
            // Silk every source shows soft.
            detail[cell] = 0.06;
            best[cell] = (0.065, 0);
        }
        if (20..23).contains(&x) && (5..8).contains(&y) {
            // Silk a sharper source could have filled.
            detail[cell] = 0.06;
            best[cell] = (0.21, 1);
        }
        if (26..32).contains(&x) && (10..20).contains(&y) {
            // Paper with strokes: blank cells beside inked ones, all sharp.
            level[cell] = 0.6;
            detail[cell] = if (x + y) % 3 == 0 { 0.3 } else { 0.03 };
            best[cell] = (detail[cell], 0);
        }
    }
    let outcome = RepairOutcome {
        measured_cells: count,
        cells: Vec::new(),
        detail_after: detail,
        best,
        level,
        clipped,
    };
    let mut regions = soft_regions(&grid, &outcome, (0, 0, width, height));
    regions.sort_by_key(|region| (region.y, region.x));
    let cell = grid.cell_size() as u32;
    let found = regions
        .iter()
        .map(|region| (region.x / cell, region.y / cell, region.cells, region.kind))
        .collect::<Vec<_>>();
    assert_eq!(
        found,
        vec![
            (20, 5, 9, SoftKind::SharperSourceUnused),
            (10, 10, 9, SoftKind::SourcesSoft),
        ]
    );
}
