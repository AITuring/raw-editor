//! Station stroke matching and station seam planning tests.

use super::*;

fn translated_stroke_source(width: u32, height: u32) -> Rgb32FImage {
    Rgb32FImage::from_fn(width, height, |x, y| {
        let texture = x as f32 * 0.00011 + (y * y) as f32 * 0.000001;
        let value = if x >= 80 { 0.82 } else { 0.14 };
        Rgb([value + texture; 3])
    })
}

fn shifted_render(source: &Rgb32FImage, dx: i32, dy: i32, invert: bool) -> Rgb32FImage {
    Rgb32FImage::from_fn(source.width(), source.height(), |x, y| {
        let Some(sx) = x.checked_add_signed(-dx) else {
            return Rgb([0.2; 3]);
        };
        let Some(sy) = y.checked_add_signed(-dy) else {
            return Rgb([0.2; 3]);
        };
        let Some(pixel) = source.get_pixel_checked(sx, sy) else {
            return Rgb([0.2; 3]);
        };
        if invert {
            Rgb([1.0 - pixel[0]; 3])
        } else {
            *pixel
        }
    })
}

#[test]
fn station_stroke_pairing_requires_independent_provenance_and_closure() {
    let mut source_info = geometry_test_image(0, None);
    let mut target_info = geometry_test_image(1, None);
    source_info.width = 192;
    source_info.height = 160;
    target_info.width = 192;
    target_info.height = 160;
    let source = translated_stroke_source(192, 160);
    let target = shifted_render(&source, 5, 3, false);
    let identity = Matrix3::identity();
    let source_view = StationStrokeView {
        image: &source_info,
        pixels: &source,
        inverse: &identity,
        projection: Projection::Planar,
        offset_x: 0.0,
        offset_y: 0.0,
    };
    let target_view = StationStrokeView {
        image: &target_info,
        pixels: &target,
        inverse: &identity,
        projection: Projection::Planar,
        offset_x: 0.0,
        offset_y: 0.0,
    };
    let feature = (78..=81)
        .filter_map(|x| stroke_feature(source_view, (x, 80)))
        .max_by(|left, right| left.strength.total_cmp(&right.strength))
        .expect("source stroke evidence");
    let expected_source = feature.point;
    let matched = closed_station_stroke_match(feature, source_view, target_view)
        .expect("the independently shifted rendering should close");
    assert_eq!(matched.source, expected_source);
    assert_eq!(
        matched.target,
        (expected_source.0 + 5, expected_source.1 + 3)
    );
    assert_eq!(matched.displacement, [5.0, 3.0]);
    assert!(matched.closure_px <= STROKE_MATCH_CLOSURE_MAX_PX);

    let no_coverage = Rgb32FImage::new(192, 160);
    let unavailable_view = StationStrokeView {
        pixels: &no_coverage,
        ..target_view
    };
    let feature = (78..=81)
        .filter_map(|x| stroke_feature(source_view, (x, 80)))
        .max_by(|left, right| left.strength.total_cmp(&right.strength))
        .expect("source stroke evidence");
    assert_eq!(
        closed_station_stroke_match(feature, source_view, unavailable_view),
        Err("no_target_edge")
    );
}

#[test]
fn station_stroke_pairing_rejects_wrong_polarity_edge() {
    let mut source_info = geometry_test_image(0, None);
    let mut target_info = geometry_test_image(1, None);
    source_info.width = 192;
    source_info.height = 160;
    target_info.width = 192;
    target_info.height = 160;
    let source = translated_stroke_source(192, 160);
    let target = shifted_render(&source, 5, 3, true);
    let identity = Matrix3::identity();
    let source_view = StationStrokeView {
        image: &source_info,
        pixels: &source,
        inverse: &identity,
        projection: Projection::Planar,
        offset_x: 0.0,
        offset_y: 0.0,
    };
    let target_view = StationStrokeView {
        image: &target_info,
        pixels: &target,
        inverse: &identity,
        projection: Projection::Planar,
        offset_x: 0.0,
        offset_y: 0.0,
    };
    let feature = (78..=81)
        .filter_map(|x| stroke_feature(source_view, (x, 80)))
        .max_by(|left, right| left.strength.total_cmp(&right.strength))
        .expect("source stroke evidence");
    assert_eq!(
        closed_station_stroke_match(feature, source_view, target_view),
        Err("orientation_or_polarity")
    );
}

#[test]
fn stroke_translation_uses_disjoint_checkerboard_holdout() {
    let matches = (0..16)
        .map(|index| ClosedStrokeMatch {
            source: ((index % 4) * 128 + 32, (index / 4) * 128 + 32),
            target: ((index % 4) * 128 + 38, (index / 4) * 128 + 28),
            displacement: [6.0, -4.0],
            score: 0.0,
            reverse_score: 0.0,
            closure_px: 0.0,
            source_strength: 0.4,
            target_strength: 0.4,
        })
        .collect::<Vec<_>>();
    let report = stroke_translation_holdout(&matches);
    assert_eq!(report["fit_count"], 8);
    assert_eq!(report["holdout_count"], 8);
    assert_eq!(report["scale"], 1.0);
    assert_eq!(report["held_out_p95_after"], 0.0);
    assert!(report["held_out_p95_before"].as_f64().unwrap() > 7.0);
    assert_eq!(report["applied"], false);
}

#[test]
fn layered_station_seam_uses_centerline_for_narrow_overlap() {
    let _run_scope = degradation::begin_run_scope();
    let mut left = geometry_test_image(0, None);
    let mut right = geometry_test_image(1, None);
    left.width = 64;
    left.height = 64;
    right.width = 64;
    right.height = 64;
    let left_ref = &left;
    let right_ref = &right;
    let images = [left_ref, right_ref];
    let homographies = HashMap::from([
        (0usize, Matrix3::identity()),
        (
            1usize,
            Matrix3::new(1.0, 0.0, 40.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        ),
    ]);
    let source = Rgb32FImage::from_pixel(64, 64, Rgb([0.5, 0.5, 0.5]));
    let mut load = |_: &ImageInfo| Ok(source.clone());
    let plans = build_station_seam_plans(
        &images,
        &homographies,
        Projection::Planar,
        &[(0, 1)],
        0.0,
        0.0,
        104,
        64,
        &residual_warp::ResidualWarp::new(),
        &vec![NO_OWNER; 104 * 64],
        &mut load,
    )
    .expect("synthetic station overlap should produce a seam plan");
    assert_eq!(plans.len(), 1);
    let plan = &plans[0];
    assert_eq!(plan.orientation, SeamOrientation::Vertical);
    assert_eq!(plan.min_x, 40);
    assert_eq!(plan.max_x, 64);
    assert_eq!(plan.cross_samples, vec![52.0, 52.0]);
    assert!(plan.first_on_low_side);
}

#[test]
fn layered_station_seam_trims_uncovered_projective_rows_before_solving() {
    let mut first = geometry_test_image(0, None);
    let mut second = geometry_test_image(1, None);
    first.width = 80;
    first.height = 40;
    second.width = 80;
    second.height = 40;
    let images = [&first, &second];
    let homographies = HashMap::from([
        (first.id, Matrix3::identity()),
        (second.id, Matrix3::identity()),
    ]);
    let source = Rgb32FImage::from_fn(80, 40, |x, _| {
        if (8..=72).contains(&x) {
            Rgb([0.5, 0.5, 0.5])
        } else {
            Rgb([0.0, 0.0, 0.0])
        }
    });
    let mut load = |_: &ImageInfo| Ok(source.clone());
    let plans = build_station_seam_plans(
        &images,
        &homographies,
        Projection::Planar,
        &[(0, 1)],
        0.0,
        0.0,
        80,
        40,
        &residual_warp::ResidualWarp::new(),
        &vec![NO_OWNER; 80 * 40],
        &mut load,
    )
    .expect("valid overlap interior should produce a seam plan");

    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].orientation, SeamOrientation::Horizontal);
    assert!(plans[0].along_min >= 8);
    assert!(plans[0].along_max <= 72);
    assert!(plans[0].cross_samples.len() >= 2);
}

#[test]
fn layered_station_seam_avoids_a_lower_colour_cost_stroke_crossing() {
    let mut first = geometry_test_image(0, None);
    let mut second = geometry_test_image(1, None);
    first.width = 96;
    first.height = 64;
    second.width = 96;
    second.height = 64;
    let images = [&first, &second];
    let homographies = HashMap::from([
        (first.id, Matrix3::identity()),
        (second.id, Matrix3::identity()),
    ]);
    let first_pixels = Rgb32FImage::from_fn(96, 64, |_, y| {
        if y == 31 {
            Rgb([0.0, 0.0, 0.0])
        } else {
            Rgb([0.5, 0.5, 0.5])
        }
    });
    let second_pixels = Rgb32FImage::from_fn(96, 64, |_, y| {
        if y == 33 {
            Rgb([0.0, 0.0, 0.0])
        } else if y == 32 {
            Rgb([0.5, 0.5, 0.5])
        } else {
            Rgb([0.55, 0.55, 0.55])
        }
    });
    let sources = [first_pixels, second_pixels];
    let mut load = |image: &ImageInfo| Ok(sources[image.id].clone());
    let owners = (0..64)
        .flat_map(|y| (0..96).map(move |_| if y <= 32 { 1 } else { 2 }))
        .collect::<Vec<_>>();
    let plans = build_station_seam_plans(
        &images,
        &homographies,
        Projection::Planar,
        &[(0, 1)],
        0.0,
        0.0,
        96,
        64,
        &residual_warp::ResidualWarp::new(),
        &owners,
        &mut load,
    )
    .expect("stroke overlap should produce a seam plan");

    assert_eq!(plans.len(), 1);
    assert!(
        plans[0]
            .cross_samples
            .iter()
            .all(|cross| !(31.0..=33.0).contains(cross)),
        "the production adapter must avoid the deceptively cheap misaligned stroke: {:?}",
        plans[0].cross_samples
    );

    let coverage = GrayImage::from_pixel(96, 64, image::Luma([255]));
    let coverage_before = coverage.clone();
    let mut source_owners = owners.clone();
    let mut owner_map =
        GrayImage::from_raw(96, 64, owners.iter().map(|owner| *owner as u8).collect())
            .expect("owner plane");
    let mut panorama = Rgb32FImage::from_fn(96, 64, |x, y| {
        let owner = owners[(y * 96 + x) as usize];
        *sources[usize::from(owner - 1)].get_pixel(x, y)
    });
    let mut reload = |image: &ImageInfo| Ok(sources[image.id].clone());
    let changed = apply_station_seam_plans(
        &plans,
        &mut panorama,
        &mut owner_map,
        &mut source_owners,
        &images,
        &homographies,
        Projection::Planar,
        0.0,
        0.0,
        &residual_warp::ResidualWarp::new(),
        &mut reload,
    )
    .expect("apply station seam");
    assert!(changed > 0);
    assert_eq!(
        coverage, coverage_before,
        "seam application changed coverage"
    );
    for y in 0..64 {
        for x in 0..96 {
            let index = (y * 96 + x) as usize;
            let owner = source_owners[index];
            assert!([1, 2].contains(&owner));
            assert_eq!(
                panorama.get_pixel(x, y),
                sources[usize::from(owner - 1)].get_pixel(x, y),
                "output pixel lost exact source provenance at ({x},{y})"
            );
        }
    }
}

#[test]
fn station_boundary_term_rejects_a_cheaper_misaligned_stroke_seam() {
    let aligned = station_seam_edge_misalignment((0.2, 0.0), (0.19, 0.0));
    let misaligned = station_seam_edge_misalignment((0.2, 0.0), (0.0, 0.2));
    assert!(aligned < misaligned);
    let aligned_cost = seam_candidate_cost_with_alignment(0.2, 32.0, aligned);
    let deceptively_lower_pixel_cost = seam_candidate_cost_with_alignment(0.0, 32.0, misaligned);
    assert!(
        aligned_cost < deceptively_lower_pixel_cost,
        "edge evidence must reject the lower-colour-cost misaligned seam"
    );
    assert_eq!(station_seam_edge_misalignment((0.0, 0.0), (0.0, 0.0)), 0.0);
}
