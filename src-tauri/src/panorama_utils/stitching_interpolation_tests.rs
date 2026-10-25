use super::*;

fn harmonize_focus_background_tone(
    image: &mut Rgb32FImage,
    image_mask: &GrayImage,
    foreground_mask: &GrayImage,
) {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || image_mask.dimensions() != (width, height)
        || foreground_mask.dimensions() != (width, height)
    {
        return;
    }

    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(width, height, width.max(height));
    let analysis_image = resize_rgb(image, analysis_width, analysis_height);
    let analysis_mask = resize_binary_mask(image_mask, analysis_width, analysis_height);
    // Do not derive the correction domain from the focus ownership mask. That
    // mask is made from sharpness decisions and may contain source-shaped
    // islands on the canvas. Instead, sample only pixels that still look like
    // the warm-brown paper itself; this makes an exposure block eligible for
    // correction while keeping painted colours out of the estimator.
    let analysis_background_like = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let covered = analysis_mask.get_pixel(x, y)[0] > 0;
        let index = y as usize * analysis_width as usize + x as usize;
        let pixel_start = index * 3;
        let pixel = &analysis_image.as_raw()[pixel_start..pixel_start + 3];
        let canvas_like = focus_stack_pixel_is_canvas_like(pixel);
        let tone_foreground = focus_stack_pixel_is_tone_foreground(pixel);
        image::Luma([u8::from(covered && canvas_like && !tone_foreground) * 255])
    });
    let foreground_soft_radius = (analysis_width.max(analysis_height) as f32 * 0.005)
        .round()
        .clamp(2.0, 12.0) as usize;
    let analysis_background_values = analysis_background_like
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let analysis_background_weight = box_blur_focus_map(
        &analysis_background_values,
        analysis_width,
        analysis_height,
        foreground_soft_radius,
    )
    .into_iter()
    .map(|value| value.clamp(0.0, 1.0))
    .collect::<Vec<_>>();
    let analysis_background = analysis_background_like.clone();
    let background_samples = analysis_background
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if background_samples < 128 {
        return;
    }

    let background_pixels = analysis_background.as_raw();
    let mut global_channels = [Vec::new(), Vec::new(), Vec::new()];
    for (index, &background) in background_pixels.iter().enumerate() {
        if background == 0 {
            continue;
        }
        let start = index * 3;
        for channel in 0..3 {
            global_channels[channel].push(analysis_image.as_raw()[start + channel]);
        }
    }
    let global_tone = [
        median_f32(&mut global_channels[0]).unwrap_or(0.0),
        median_f32(&mut global_channels[1]).unwrap_or(0.0),
        median_f32(&mut global_channels[2]).unwrap_or(0.0),
    ];
    let analysis_background_like = analysis_background.clone();
    let background_like_pixels = analysis_background_like
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    println!(
        "  - Background tone sample coverage: {background_like_pixels}/{background_samples} analysis pixels"
    );
    if background_like_pixels < 128 {
        return;
    }

    // The first blur removes weave-scale variation from the estimate. The
    // second, much wider blur provides a continuous target across source-sized
    // exposure blocks. Their difference is a low-frequency correction, so the
    // original high-frequency canvas texture is retained pixel-for-pixel.
    let local_tone = masked_box_blur_rgb(
        &analysis_image,
        &analysis_background_like,
        FOCUS_BACKGROUND_TONE_LOCAL_RADIUS,
    );
    let smooth_tone = masked_box_blur_rgb(
        &local_tone,
        &analysis_background_like,
        FOCUS_BACKGROUND_TONE_SMOOTH_RADIUS,
    );
    let local_pixels = local_tone.as_raw();
    let smooth_pixels = smooth_tone.as_raw();
    let mut correction = vec![[0.0f32; 3]; analysis_width as usize * analysis_height as usize];
    correction
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, output)| {
            if background_like_pixels == 0 || analysis_background_like.as_raw()[index] == 0 {
                return;
            }
            let start = index * 3;
            for channel in 0..3 {
                let target = smooth_pixels[start + channel]
                    * (1.0 - FOCUS_BACKGROUND_TONE_GLOBAL_WEIGHT)
                    + global_tone[channel] * FOCUS_BACKGROUND_TONE_GLOBAL_WEIGHT;
                output[channel] = (target - local_pixels[start + channel]).clamp(
                    -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                    FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                );
            }
        });

    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize;
    let background_weight_ref = &analysis_background_weight;
    let correction_ref = &correction;
    let full_mask = image_mask.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            for x in 0..width as usize {
                let full_index = y * width as usize + x;
                if full_mask[full_index] == 0 {
                    continue;
                }
                let start = x * 3;
                if !focus_stack_pixel_is_canvas_like(&row[start..start + 3])
                    || focus_stack_pixel_is_tone_foreground(&row[start..start + 3])
                {
                    continue;
                }
                let x_sample = x_samples[x];
                let top_left_weight =
                    background_weight_ref[y_sample.lower * analysis_stride + x_sample.lower];
                let top_right_weight =
                    background_weight_ref[y_sample.lower * analysis_stride + x_sample.upper];
                let bottom_left_weight =
                    background_weight_ref[y_sample.upper * analysis_stride + x_sample.lower];
                let bottom_right_weight =
                    background_weight_ref[y_sample.upper * analysis_stride + x_sample.upper];
                let top_weight = top_left_weight * (1.0 - x_sample.upper_weight)
                    + top_right_weight * x_sample.upper_weight;
                let bottom_weight = bottom_left_weight * (1.0 - x_sample.upper_weight)
                    + bottom_right_weight * x_sample.upper_weight;
                let background_weight = (top_weight * (1.0 - y_sample.upper_weight)
                    + bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                if background_weight <= 0.001 {
                    continue;
                }
                let top_left = correction_ref[y_sample.lower * analysis_stride + x_sample.lower];
                let top_right = correction_ref[y_sample.lower * analysis_stride + x_sample.upper];
                let bottom_left = correction_ref[y_sample.upper * analysis_stride + x_sample.lower];
                let bottom_right =
                    correction_ref[y_sample.upper * analysis_stride + x_sample.upper];
                let mut delta = [0.0f32; 3];
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_sample.upper_weight)
                        + top_right[channel] * x_sample.upper_weight;
                    let bottom = bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + bottom_right[channel] * x_sample.upper_weight;
                    delta[channel] = (top * (1.0 - y_sample.upper_weight)
                        + bottom * y_sample.upper_weight)
                        * background_weight;
                }
                for channel in 0..3 {
                    row[start + channel] = (row[start + channel] + delta[channel]).clamp(0.0, 1.0);
                }
            }
        });
}

fn geometry_test_image(id: usize, focal_length_35mm: Option<f64>) -> ImageInfo {
    ImageInfo {
        id,
        filename: format!("geometry-{id}.jpg"),
        width: 1_000,
        height: 800,
        alignment_image: GrayImage::new(1, 1),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

#[test]
fn native_focus_refinement_samples_fine_texture_once_from_the_original() {
    let mut info = geometry_test_image(0, None);
    info.width = 64;
    info.height = 48;
    let source = Rgb32FImage::from_fn(64, 48, |x, y| {
        let value = if (x + y) % 2 == 0 { 0.15 } else { 0.85 };
        Rgb([value, value * 0.8, value * 0.6])
    });
    let mut field = super::super::mosaic::Field::<2>::new(64, 48, 16.0);
    field.values.fill([0.37, -0.23]);
    let refinement = FocusSourceRefinement {
        translation: (2, -1),
        displacement: Some(
            super::super::mosaic::NativeResidualDisplacement::from_test_field(field),
        ),
        ..Default::default()
    };
    let h = Matrix3::new(1.0, 0.0, 0.31, 0.0, 1.0, -0.19, 0.0, 0.0, 1.0);
    let direct = render_focus_layer_with_refinement(
        &info,
        &source,
        &h,
        Projection::Planar,
        None,
        0.0,
        0.0,
        64,
        48,
        Some(&refinement),
        None,
    );
    for y in 12..28 {
        for x in 12..44 {
            let expected = get_high_quality_interpolated_pixel(
                &source,
                x as f64 + 0.37 - 2.0 - 0.31,
                y as f64 - 0.23 + 1.0 + 0.19,
            );
            let actual = direct.image.get_pixel(x - direct.left, y - direct.top);
            for channel in 0..3 {
                assert!((actual[channel] - expected[channel]).abs() < 1e-6);
            }
        }
    }
    // The old two-filter path attenuates this detail despite identical geometry.
    let projected = render_focus_layer(
        &info,
        &source,
        &h,
        Projection::Planar,
        None,
        0.0,
        0.0,
        64,
        48,
    );
    let translated = translate_rendered_focus_layer(projected, 2, -1, 64, 48);
    let old = warp_rendered_focus_layer(translated, refinement.displacement.as_ref().unwrap());
    let actual = direct.image.get_pixel(24 - direct.left, 20 - direct.top)[0];
    let blurred = old.image.get_pixel(24 - old.left, 20 - old.top)[0];
    assert!((actual - blurred).abs() > 0.02);
}

#[test]
fn focus_layer_masks_optical_padding_before_ownership() {
    let mut info = geometry_test_image(0, None);
    info.width = 64;
    info.height = 48;
    let source = Rgb32FImage::from_fn(64, 48, |x, y| {
        if (10..54).contains(&x) && (8..40).contains(&y) {
            Rgb([0.002, 0.003, 0.001])
        } else {
            Rgb([0.0; 3])
        }
    });
    let layer = render_focus_layer(
        &info,
        &source,
        &Matrix3::identity(),
        Projection::Planar,
        None,
        0.0,
        0.0,
        64,
        48,
    );
    assert_eq!(layer.mask.get_pixel(4, 4)[0], 0);
    assert_eq!(layer.mask.get_pixel(32, 24)[0], 255);
    assert_eq!(layer.image.get_pixel(4, 4).0, [0.0; 3]);
    assert_eq!(layer.image.get_pixel(32, 24).0, [0.002, 0.003, 0.001]);
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
    let (plans, _planning) = build_station_seam_plans(
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
    let (plans, _planning) = build_station_seam_plans(
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
    let (plans, _planning) = build_station_seam_plans(
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
    let changed =
        apply_station_seam_plans(&plans, &mut panorama, &mut owner_map, &mut source_owners)
            .expect("apply station seam");
    assert!(changed.changed_pixels > 0);
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

fn seam_application_fixture(
    width: u32,
    height: u32,
    orientation: SeamOrientation,
    reversed: bool,
    transitions: &[u32],
    cross_samples: [f64; 2],
    include_non_pair_hole: bool,
    legacy: bool,
) -> (Rgb32FImage, GrayImage, Vec<u16>, StationSeamApplyStats) {
    let mut first = geometry_test_image(0, None);
    let mut second = geometry_test_image(1, None);
    first.width = width;
    first.height = height;
    second.width = width;
    second.height = height;
    let images = [&first, &second];
    let homographies = HashMap::from([
        (first.id, Matrix3::identity()),
        (second.id, Matrix3::identity()),
    ]);
    let sources = [
        Rgb32FImage::from_fn(width, height, |x, y| {
            Rgb([0.2, x as f32 / width as f32, y as f32 / height as f32])
        }),
        Rgb32FImage::from_fn(width, height, |x, y| {
            Rgb([0.8, y as f32 / height as f32, x as f32 / width as f32])
        }),
    ];
    let (cross_max, along_max) = match orientation {
        SeamOrientation::Vertical => (width - 1, height - 1),
        SeamOrientation::Horizontal => (height - 1, width - 1),
    };
    assert!(transitions.iter().all(|&cross| cross <= cross_max));
    let owner_at = |x: u32, y: u32| {
        let cross = match orientation {
            SeamOrientation::Vertical => x,
            SeamOrientation::Horizontal => y,
        };
        let low_owner = if reversed { 2 } else { 1 };
        let high_owner = if reversed { 1 } else { 2 };
        if transitions
            .iter()
            .filter(|&&transition| cross >= transition)
            .count()
            % 2
            == 0
        {
            low_owner
        } else {
            high_owner
        }
    };
    let mut owners = (0..height)
        .flat_map(|y| (0..width).map(move |x| owner_at(x, y)))
        .collect::<Vec<_>>();
    // Non-pair provenance inside the corridor must remain untouched.
    if include_non_pair_hole {
        let hole = (height / 3 * width + width / 2 + 2) as usize;
        owners[hole] = NO_OWNER;
    }
    let mut owner_map = GrayImage::from_raw(
        width,
        height,
        owners.iter().map(|owner| (*owner).min(255) as u8).collect(),
    )
    .unwrap();
    let mut panorama = Rgb32FImage::from_fn(width, height, |x, y| {
        let owner = owners[(y * width + x) as usize];
        if owner == NO_OWNER {
            Rgb([0.0; 3])
        } else {
            *sources[usize::from(owner - 1)].get_pixel(x, y)
        }
    });
    let mut plan = StationSeamPlan {
        first_owner: 1,
        second_owner: 2,
        min_x: 0,
        max_x: width - 1,
        min_y: 0,
        max_y: height - 1,
        along_min: 0,
        along_max,
        orientation,
        cross_samples: cross_samples.to_vec(),
        replacement_samples: Vec::new(),
    };
    let transition_index = StationOwnerTransitionIndex::build(&owners, width, height);
    materialize_station_seam_samples(
        &mut plan,
        &transition_index,
        &owners,
        width,
        &first,
        &sources[0],
        &Matrix3::identity(),
        &second,
        &sources[1],
        &Matrix3::identity(),
        Projection::Planar,
        &residual_warp::ResidualWarp::new(),
        0.0,
        0.0,
    );
    let mut load = |image: &ImageInfo| Ok(sources[image.id].clone());
    let residual = residual_warp::ResidualWarp::new();
    let stats = if legacy {
        apply_station_seam_plans_legacy_full_scan(
            &[plan],
            &mut panorama,
            &mut owner_map,
            &mut owners,
            &images,
            &homographies,
            Projection::Planar,
            0.0,
            0.0,
            &residual,
            &mut load,
        )
    } else {
        apply_station_seam_plans(&[plan], &mut panorama, &mut owner_map, &mut owners)
    }
    .unwrap();
    (panorama, owner_map, owners, stats)
}

#[test]
fn local32_corridor_matches_legacy_full_scan_for_transition_layouts() {
    for (orientation, reversed) in [
        (SeamOrientation::Vertical, false),
        (SeamOrientation::Vertical, true),
        (SeamOrientation::Horizontal, false),
        (SeamOrientation::Horizontal, true),
    ] {
        let (width, height) = (96, 72);
        let cross_max = match orientation {
            SeamOrientation::Vertical => width - 1,
            SeamOrientation::Horizontal => height - 1,
        };
        let center = cross_max / 2;
        let cases = [
            (
                "nearest transition inside 32",
                vec![center],
                [f64::from(center) + 7.25, f64::from(center) + 11.75],
                true,
                false,
            ),
            (
                "only transition outside 32",
                vec![2],
                [f64::from(cross_max - 2); 2],
                false,
                true,
            ),
            (
                "multiple transitions inside and outside",
                vec![2, center - 20, center + 5, cross_max - 2],
                [f64::from(center) + 12.4, f64::from(center) + 13.6],
                false,
                false,
            ),
            (
                "low endpoint and fractional rounding",
                vec![1, cross_max],
                [33.0, 33.49],
                false,
                false,
            ),
            (
                "high endpoint and fractional rounding",
                vec![1, cross_max],
                [f64::from(cross_max) - 32.49, f64::from(cross_max) - 32.0],
                false,
                false,
            ),
        ];
        for (name, transitions, cross_samples, include_hole, expect_unchanged) in cases {
            let optimized = seam_application_fixture(
                width,
                height,
                orientation,
                reversed,
                &transitions,
                cross_samples,
                include_hole,
                false,
            );
            let legacy = seam_application_fixture(
                width,
                height,
                orientation,
                reversed,
                &transitions,
                cross_samples,
                include_hole,
                true,
            );
            assert_eq!(optimized.0, legacy.0, "panorama: {name}");
            assert_eq!(optimized.1, legacy.1, "owner map: {name}");
            assert_eq!(optimized.2, legacy.2, "owner ledger: {name}");
            assert_eq!(
                optimized.3.changed_pixels, legacy.3.changed_pixels,
                "changed count: {name}"
            );
            if expect_unchanged {
                assert_eq!(optimized.3.changed_pixels, 0, "{name}");
                assert_eq!(optimized.3.rewrite_visits, 0, "{name}");
            } else {
                assert!(optimized.3.changed_pixels > 0, "{name}");
            }
        }
    }
}

#[test]
fn local32_total_owner_reads_scale_with_seam_length_not_cross_extent() {
    for orientation in [SeamOrientation::Vertical, SeamOrientation::Horizontal] {
        let run = |cross_extent: u32, legacy: bool| {
            let (width, height) = match orientation {
                SeamOrientation::Vertical => (cross_extent, 128),
                SeamOrientation::Horizontal => (128, cross_extent),
            };
            let transition = cross_extent / 2;
            seam_application_fixture(
                width,
                height,
                orientation,
                false,
                &[transition],
                [f64::from(transition) + 7.0, f64::from(transition) + 11.0],
                false,
                legacy,
            )
            .3
        };
        let narrow = run(256, false);
        let wide = run(2_048, false);
        let legacy = run(2_048, true);
        let seam_length = 128;
        let fixed_reads_per_along = 66 + 33;
        assert_eq!(wide.total_owner_reads(), narrow.total_owner_reads());
        assert!(
            wide.total_owner_reads() <= seam_length * fixed_reads_per_along,
            "{wide:?}"
        );
        assert!(wide.discovery_owner_reads <= seam_length * 66, "{wide:?}");
        assert!(wide.rewrite_visits <= seam_length * 33, "{wide:?}");
        assert_eq!(legacy.discovery_owner_reads, 2_048 * seam_length);
        assert_eq!(legacy.rewrite_visits, 2_048 * seam_length);
        assert!(legacy.total_owner_reads() > wide.total_owner_reads() * 30);
    }
}

#[test]
fn pixel_aligned_canvas_preserves_integer_reference_coordinates() {
    let (offset, size) = pixel_aligned_canvas(-123.4, 876.2);
    assert_eq!(offset, 124.0);
    assert_eq!(size, 1001);
    assert_eq!(offset.fract(), 0.0);
    assert!(offset - 123.4 >= 0.0);
}

#[test]
fn virtual_tile_geometry_preserves_nonzero_world_origin() {
    let image = geometry_test_image(7, None);
    let image_ref = &image;
    let image_to_world = Matrix3::new(1.0, 0.0, -123.4, 0.0, 1.0, 55.2, 0.0, 0.0, 1.0);
    let homographies = HashMap::from([(image.id, image_to_world)]);
    let geometry =
        focus_stack_virtual_tile_geometry(&[image_ref], &homographies, Projection::Planar, None)
            .expect("translated source must have valid Virtual_Tile geometry");
    assert_eq!(geometry.tile_to_world[(0, 2)], -124.0);
    assert_eq!(geometry.tile_to_world[(1, 2)], 55.0);
    for source in [Point2::new(0.0, 0.0), Point2::new(999.0, 799.0)] {
        let world = image_to_world * Point3::new(source.x, source.y, 1.0);
        let tile = Point2::new(world.x + 124.0, world.y - 55.0);
        let recovered = geometry.tile_to_world * Point3::new(tile.x, tile.y, 1.0);
        assert!((recovered.x - world.x).abs() < 1e-9);
        assert!((recovered.y - world.y).abs() < 1e-9);
    }
}

#[test]
fn cubic_warp_sampling_preserves_more_edge_contrast_than_bilinear() {
    let image = Rgb32FImage::from_fn(8, 8, |x, _| {
        let value = if x < 4 { 0.0 } else { 1.0 };
        Rgb([value, value, value])
    });
    let bilinear_low = get_interpolated_pixel(&image, 3.25, 3.5)[0];
    let bilinear_high = get_interpolated_pixel(&image, 3.75, 3.5)[0];
    let cubic_low = get_high_quality_interpolated_pixel(&image, 3.25, 3.5)[0];
    let cubic_high = get_high_quality_interpolated_pixel(&image, 3.75, 3.5)[0];

    assert!(cubic_high - cubic_low > bilinear_high - bilinear_low);
    assert!((0.0..=1.0).contains(&cubic_low));
    assert!((0.0..=1.0).contains(&cubic_high));
}

#[test]
fn focus_box_blur_preserves_a_constant_score_map() {
    let source = vec![3.5f32; 5 * 4];
    let blurred = box_blur_focus_map(&source, 5, 4, 2);

    assert!(blurred.iter().all(|value| (*value - 3.5).abs() < 1e-6));
}

#[test]
fn overlap_exposure_ignores_coloured_artwork_samples() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.60, 0.50, 0.40]));
    let candidate = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
        if x < WIDTH * 3 / 4 {
            // A saturated red seal must not determine the paper gain.
            Rgb([1.0, 0.02, 0.01])
        } else {
            Rgb([0.60, 0.50, 0.40])
        }
    });
    let mut info = geometry_test_image(0, None);
    info.width = WIDTH;
    info.height = HEIGHT;
    let panorama_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let compensation = estimate_overlap_exposure_compensation(ExposureOverlap {
        panorama: &base,
        panorama_mask: &panorama_mask,
        candidate: &info,
        candidate_image: &candidate,
        candidate_inverse: &Matrix3::identity(),
        projection: Projection::Planar,
        offset_x: 0.0,
        offset_y: 0.0,
    });

    assert!(
        (compensation.representative_gain - 1.0).abs() < 0.02,
        "artwork contaminated exposure estimate: gain={:.3}",
        compensation.representative_gain
    );
}

#[test]
fn overlap_exposure_recovers_bounded_rgb_channel_gain() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.66, 0.52, 0.40]));
    let candidate = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.60, 0.55, 0.44]));
    let mut info = geometry_test_image(0, None);
    info.width = WIDTH;
    info.height = HEIGHT;
    let panorama_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let compensation = estimate_overlap_exposure_compensation(ExposureOverlap {
        panorama: &base,
        panorama_mask: &panorama_mask,
        candidate: &info,
        candidate_image: &candidate,
        candidate_inverse: &Matrix3::identity(),
        projection: Projection::Planar,
        offset_x: 0.0,
        offset_y: 0.0,
    });
    let gains = compensation.representative_channel_gain;
    assert!((gains[0] - 1.10).abs() < 0.02, "red gain={:.3}", gains[0]);
    assert!((gains[1] - 0.95).abs() < 0.02, "green gain={:.3}", gains[1]);
    assert!((gains[2] - 0.91).abs() < 0.02, "blue gain={:.3}", gains[2]);
}

#[test]
fn progressive_exposure_damping_keeps_direction_without_chain_drift() {
    let compensation = ExposureCompensation {
        cell_size: 1,
        grid_width: 1,
        grid_height: 1,
        gains: vec![0.75],
        representative_gain: 0.75,
        channel_gains: vec![[0.75, 1.20, 1.0]],
        representative_channel_gain: [0.75, 1.20, 1.0],
    }
    .damped(0.45);
    let expected = 0.75f32.ln().mul_add(0.45, 0.0).exp();
    assert!((compensation.representative_gain - expected).abs() < 1e-6);
    assert!(compensation.representative_gain < 1.0);
    assert!(compensation.representative_gain > 0.85);
    assert!(compensation.representative_channel_gain[1] > 1.0);
    assert_eq!(compensation.gain_at(0, 0), compensation.representative_gain);
}

#[test]
fn seam_blend_rejects_a_triangular_zero_support_corner() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let paper = Rgb([0.62, 0.51, 0.43]);
    let mut panorama = Rgb32FImage::from_pixel(WIDTH, HEIGHT, paper);
    let mut panorama_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let candidate = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        if x + y < 28 {
            Rgb([0.0, 0.0, 0.0])
        } else {
            paper
        }
    });
    let mut info = geometry_test_image(0, None);
    info.width = WIDTH;
    info.height = HEIGHT;
    let exposure = ExposureCompensation {
        cell_size: 1,
        grid_width: 1,
        grid_height: 1,
        gains: vec![1.0],
        representative_gain: 1.0,
        channel_gains: vec![[1.0; 3]],
        representative_channel_gain: [1.0; 3],
    };
    let seam_coords = vec![HEIGHT as i32 / 2; WIDTH as usize];
    blend_panorama_seam_band(SeamBandBlend {
        panorama: &mut panorama,
        panorama_mask: &mut panorama_mask,
        img_to_add_info: &info,
        img_to_add: &candidate,
        h_add: &Matrix3::identity(),
        projection: Projection::Planar,
        offset_x: 0.0,
        offset_y: 0.0,
        orientation: SeamOrientation::Horizontal,
        seam_coords: &seam_coords,
        new_image_is_dominant_side: true,
        min_x: 0,
        max_x: WIDTH - 1,
        min_y: 0,
        max_y: HEIGHT - 1,
        exposure: &exposure,
    });

    let maximum_error = panorama
        .as_raw()
        .iter()
        .zip(std::iter::repeat(&paper.0).flatten())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        maximum_error < 1e-4,
        "zero-support corner leaked into the seam pyramid: error={maximum_error}"
    );
    assert!(!source_sample_has_support(&candidate, 4.0, 4.0));
    assert!(source_sample_has_support(&candidate, 40.0, 40.0));
}

#[test]
fn focus_tile_tie_break_is_stable_and_quality_first() {
    assert!(focus_tile_ownership_should_replace(0, 0, 1, 3));
    assert!(focus_tile_ownership_should_replace(80, 4, 81, 9));
    assert!(!focus_tile_ownership_should_replace(81, 4, 80, 1));
    // Equal edge distance must resolve to the same tile identity even if
    // the caller's loading order changes.
    assert!(focus_tile_ownership_should_replace(120, 9, 120, 4));
    assert!(!focus_tile_ownership_should_replace(120, 4, 120, 9));
}

#[test]
fn focus_score_prefers_native_edges_over_a_blurred_layer() {
    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 128;
    let sharp = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        // Use aperiodic, high-frequency structure so the metric cannot
        // win by matching a single periodic phase after blur.
        let cell = ((x / 3 + y / 5 + (x * 7 + y * 11) % 5) & 1) as f32;
        let value = 0.18 + cell * 0.62;
        Rgb([value, value * 0.92, value * 0.84])
    });
    let blurred = image::imageops::blur(&sharp, 2.2);
    let mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let sharp_score = focus_score_map(&sharp, &mask);
    let blurred_score = focus_score_map(&blurred, &mask);
    let interior = |scores: &[f32]| {
        let mut values = Vec::new();
        for y in 8..(HEIGHT - 8) as usize {
            let row = y * WIDTH as usize;
            values.extend_from_slice(&scores[row + 8..row + (WIDTH - 8) as usize]);
        }
        values.iter().copied().sum::<f32>() / values.len().max(1) as f32
    };
    let sharp_mean = interior(&sharp_score);
    let blurred_mean = interior(&blurred_score);
    assert!(
        sharp_mean > blurred_mean * 1.8,
        "focus metric did not separate sharp={sharp_mean:.5} from blurred={blurred_mean:.5}"
    );
}

#[test]
fn focus_analysis_keeps_source_sampling_density_on_long_mosaics() {
    let (analysis_width, analysis_height) = focus_analysis_dimensions(31_121, 11_034, 9_504);

    assert!(analysis_width >= 4_500);
    assert!(analysis_height >= 1_500);
    assert!(u64::from(analysis_width) * u64::from(analysis_height) <= FOCUS_ANALYSIS_MAX_PIXELS);
}

#[test]
fn shifted_mosaic_detection_catches_a_lens_scale_change() {
    let first = geometry_test_image(0, Some(35.0));
    let second = geometry_test_image(1, Some(85.0));
    let center_x = first.width as f64 * 0.5;
    let center_y = first.height as f64 * 0.5;
    let scale = 1.25;
    let scale_about_center = Matrix3::new(
        scale,
        0.0,
        center_x * (1.0 - scale),
        0.0,
        scale,
        center_y * (1.0 - scale),
        0.0,
        0.0,
        1.0,
    );
    let homographies = HashMap::from([(0, Matrix3::identity()), (1, scale_about_center)]);
    let images = vec![&first, &second];

    assert!(focus_stack_is_shifted_mosaic(
        &images,
        &homographies,
        Projection::Planar
    ));
}

#[test]
fn shifted_mosaic_detection_uses_focal_metadata_when_geometry_is_centered() {
    let first = geometry_test_image(0, Some(35.0));
    let second = geometry_test_image(1, Some(85.0));
    let homographies = HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())]);
    let images = vec![&first, &second];

    assert!(focus_stack_is_shifted_mosaic(
        &images,
        &homographies,
        Projection::Planar
    ));
}

#[test]
fn shifted_mosaic_detection_keeps_small_focus_breathing_on_the_focus_path() {
    let first = geometry_test_image(0, Some(85.0));
    let second = geometry_test_image(1, Some(85.0));
    let center_x = first.width as f64 * 0.5;
    let center_y = first.height as f64 * 0.5;
    let scale = 1.025;
    let small_breathing = Matrix3::new(
        scale,
        0.0,
        center_x * (1.0 - scale) + 18.0,
        0.0,
        scale,
        center_y * (1.0 - scale) + 12.0,
        0.0,
        0.0,
        1.0,
    );
    let homographies = HashMap::from([(0, Matrix3::identity()), (1, small_breathing)]);
    let images = vec![&first, &second];

    assert!(!focus_stack_is_shifted_mosaic(
        &images,
        &homographies,
        Projection::Planar
    ));
}

#[test]
fn parallel_pyramid_downsample_preserves_constant_pixels_and_odd_edges() {
    let source = Rgb32FImage::from_pixel(5, 7, Rgb([0.2, 0.4, 0.8]));
    let downsampled = downsample_rgb_half(&source);

    assert_eq!(downsampled.dimensions(), (3, 4));
    assert!(downsampled.pixels().all(|pixel| pixel.0 == [0.2, 0.4, 0.8]));
}

#[test]
fn parallel_pyramid_detail_reconstructs_the_original_pixels() {
    let source = Rgb32FImage::from_fn(17, 11, |x, y| {
        let value = (x as f32 * 0.07 + y as f32 * 0.03).sin();
        Rgb([value, value * 0.5, 1.0 - value])
    });
    let coarse = downsample_rgb_half(&source);
    let detail = subtract_upsampled_rgb(&source, &coarse);
    let reconstructed = upsample_and_add_rgb(&coarse, &detail);

    assert_eq!(reconstructed.dimensions(), source.dimensions());
    let maximum_error = reconstructed
        .as_raw()
        .iter()
        .zip(source.as_raw())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(maximum_error < 1e-6, "maximum error: {maximum_error}");
}

#[test]
fn panorama_detail_feather_is_local_and_symmetric() {
    let radius = PANORAMA_DETAIL_SEAM_FEATHER_RADIUS;

    assert_eq!(panorama_detail_alpha(f32::NEG_INFINITY), 0.0);
    assert_eq!(panorama_detail_alpha(-radius), 0.0);
    assert_eq!(panorama_detail_alpha(0.0), 0.5);
    assert_eq!(panorama_detail_alpha(radius), 1.0);
    assert_eq!(panorama_detail_alpha(f32::INFINITY), 1.0);
}

#[test]
fn panorama_global_tone_blend_preserves_medium_frequency_detail() {
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 64;
    let pattern = |x: u32| if (x / 4).is_multiple_of(2) { 0.2 } else { 0.8 };
    let base = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
        let value = pattern(x);
        Rgb([value, value, value])
    });
    let candidate = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
        let value = (pattern((x + 4).min(WIDTH - 1)) + 0.08).min(1.0);
        Rgb([value, value, value])
    });
    let seam_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
        image::Luma([if x >= WIDTH / 2 { 255 } else { 0 }])
    });
    let tone_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
        image::Luma([((x as f32 / (WIDTH - 1) as f32) * 255.0).round() as u8])
    });
    let blended = multiband_blend(
        base.clone(),
        candidate,
        seam_mask,
        Some(tone_mask),
        PANORAMA_BLEND_BANDS,
        false,
    );
    let horizontal_variation = |image: &Rgb32FImage| {
        (32..96)
            .map(|x| {
                (image.get_pixel(x, HEIGHT / 2)[0] - image.get_pixel(x - 1, HEIGHT / 2)[0]).abs()
            })
            .sum::<f32>()
    };

    let source_variation = horizontal_variation(&base);
    let blended_variation = horizontal_variation(&blended);
    assert!(
        blended_variation >= source_variation * 0.9,
        "medium-frequency contrast fell from {source_variation:.3} to {blended_variation:.3}",
    );
}

#[test]
fn focus_translation_recovers_a_non_grid_aligned_displacement() {
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 800;
    const LEFT: i32 = 21;
    const TOP: i32 = 17;
    const SHIFT_X: i32 = 19;
    const SHIFT_Y: i32 = 11;
    let pattern = |x: i32, y: i32| {
        let x = x as f32;
        let y = y as f32;
        // A deterministic aperiodic texture keeps the correspondence
        // unique instead of letting a periodic texture pass the margin at
        // another offset.
        let xi = x as i32;
        let yi = y as i32;
        let hash = ((xi.wrapping_mul(73_856_093)
            ^ yi.wrapping_mul(19_349_663)
            ^ (xi.wrapping_mul(yi)).wrapping_mul(83_492_791))
            & 255) as f32
            / 255.0;
        let wave = (x * 0.017 + y * 0.043).sin() * 0.12;
        (0.12 + hash * 0.72 + wave).clamp(0.03, 0.97)
    };
    let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        let value = pattern(x as i32, y as i32);
        Rgb([value, value * 0.91, value * 0.78])
    });
    let merged_image = Rgb32FImage::from_fn(WIDTH + 80, HEIGHT + 80, |x, y| {
        let source_x = x as i32 - LEFT - SHIFT_X;
        let source_y = y as i32 - TOP - SHIFT_Y;
        let value = pattern(source_x, source_y);
        Rgb([value, value * 0.91, value * 0.78])
    });
    let candidate = RenderedFocusLayer {
        image: candidate_image,
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        left: LEFT as u32,
        top: TOP as u32,
    };
    let merged_mask = GrayImage::from_pixel(WIDTH + 80, HEIGHT + 80, image::Luma([255]));

    let result = estimate_focus_layer_translation(&candidate, &merged_image, &merged_mask)
        .expect("synthetic overlap should produce a translation");

    assert_eq!((result.0, result.1), (SHIFT_X, SHIFT_Y));
    assert!(result.3 >= 8, "insufficient patch consensus: {:?}", result);
}

#[test]
fn focus_translation_accepts_aligned_layers_without_forcing_motion() {
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 640;
    let pattern = |x: i32, y: i32| {
        let v = ((x * 17 + y * 29).rem_euclid(97) as f32) / 97.0;
        0.15 + 0.7 * v
    };
    let image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        let value = pattern(x as i32, y as i32);
        Rgb([value, value * 0.8, value * 0.65])
    });
    let candidate = RenderedFocusLayer {
        image: image.clone(),
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        left: 0,
        top: 0,
    };
    let mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));

    let result = estimate_focus_layer_translation(&candidate, &image, &mask);

    assert!(
        result.is_none(),
        "already aligned layers should remain untouched: {result:?}"
    );
}

#[test]
fn focus_decision_prefers_the_locally_sharper_layer() {
    let base_focus = vec![0.9, 0.8, 0.1, 0.1];
    let candidate_focus = vec![0.1, 0.1, 0.8, 0.9];
    let base_mask = GrayImage::from_pixel(4, 1, image::Luma([255]));
    let candidate_mask = GrayImage::from_pixel(4, 1, image::Luma([255]));

    let decision = focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

    assert_eq!(decision.as_raw(), &[0, 0, 255, 255]);
}

#[test]
fn focus_decision_extends_clear_structure_over_an_adjacent_halo() {
    let base_focus = vec![0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2];
    let candidate_focus = vec![0.1, 0.1, 0.1, 0.8, 0.8, 0.22, 0.2];
    let base_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));
    let candidate_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));

    let decision = focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

    assert_eq!(decision.as_raw()[2], 0);
    assert_eq!(decision.as_raw()[4], 255);
    assert_eq!(decision.as_raw()[5], 255);
}

#[test]
fn focus_decision_protects_base_structure_symmetrically() {
    let base_focus = vec![0.1, 0.1, 0.8, 0.8, 0.22, 0.2, 0.2];
    let candidate_focus = vec![0.9, 0.9, 0.1, 0.1, 0.2, 0.2, 0.2];
    let base_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));
    let candidate_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));

    let decision = focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

    assert_eq!(decision.as_raw()[0], 255);
    assert_eq!(decision.as_raw()[2], 0);
    assert_eq!(decision.as_raw()[4], 0);
}

#[test]
fn focus_canvas_switch_suppression_keeps_canvas_owner_but_allows_ink() {
    let mut decision = GrayImage::from_pixel(2, 1, image::Luma([255]));
    let base = Rgb32FImage::from_fn(2, 1, |x, _| {
        if x == 0 {
            Rgb([0.32, 0.28, 0.24])
        } else {
            Rgb([0.05, 0.05, 0.05])
        }
    });
    let candidate = Rgb32FImage::from_fn(2, 1, |x, _| {
        if x == 0 {
            Rgb([0.40, 0.36, 0.32])
        } else {
            Rgb([0.01, 0.01, 0.01])
        }
    });
    let mask = GrayImage::from_pixel(2, 1, image::Luma([255]));

    suppress_focus_canvas_switches(&mut decision, &base, &candidate, &mask, &mask);

    assert_eq!(decision.as_raw(), &[0, 255]);
}

#[test]
fn focus_seam_keeps_painted_detail_on_one_source() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let canvas = Rgb([0.32, 0.28, 0.24]);
    let candidate_canvas = Rgb([0.40, 0.36, 0.32]);
    let mut base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, canvas);
    let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        if x == 32 && (16..48).contains(&y) {
            Rgb([0.01, 0.01, 0.01])
        } else {
            candidate_canvas
        }
    });
    let candidate = RenderedFocusLayer {
        image: candidate_image,
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        left: 0,
        top: 0,
    };
    let mut base_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let merged_foreground_mask = GrayImage::new(WIDTH, HEIGHT);
    let decision = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
        image::Luma([u8::from(x >= WIDTH / 2) * 255])
    });

    blend_focus_seam_band(
        &mut base,
        &mut base_mask,
        &merged_foreground_mask,
        &candidate,
        &decision,
    );

    assert_eq!(base.get_pixel(32, 32).0, [0.01, 0.01, 0.01]);
}

#[test]
fn hard_focus_selection_copies_source_pixels_without_averaging() {
    let mut base =
        Rgb32FImage::from_raw(3, 1, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
    let candidate =
        Rgb32FImage::from_raw(3, 1, vec![0.0, 0.0, 1.0, 0.2, 0.3, 0.9, 0.7, 0.1, 0.8]).unwrap();
    let mut base_mask = GrayImage::from_raw(3, 1, vec![255, 255, 0]).unwrap();
    let candidate_mask = GrayImage::from_pixel(3, 1, image::Luma([255]));
    let decision = GrayImage::from_raw(3, 1, vec![0, 255, 0]).unwrap();

    hard_select_focus_pixels(
        &mut base,
        &mut base_mask,
        &candidate,
        &candidate_mask,
        &decision,
    );

    assert_eq!(base.get_pixel(0, 0).0, [1.0, 0.0, 0.0]);
    assert_eq!(base.get_pixel(1, 0).0, [0.2, 0.3, 0.9]);
    assert_eq!(base.get_pixel(2, 0).0, [0.7, 0.1, 0.8]);
    assert_eq!(base_mask.as_raw(), &[255, 255, 255]);
}

#[test]
fn focus_color_correction_removes_a_measured_channel_gain() {
    let base = Rgb32FImage::from_pixel(40, 16, Rgb([0.4, 0.5, 0.6]));
    let mut candidate_image = Rgb32FImage::from_pixel(32, 16, Rgb([0.5, 0.625, 0.75]));
    let base_mask = GrayImage::from_pixel(40, 16, image::Luma([255]));
    let candidate = RenderedFocusLayer {
        image: candidate_image.clone(),
        mask: GrayImage::from_pixel(32, 16, image::Luma([255])),
        foreground_mask: GrayImage::new(32, 16),
        relaxed_foreground_mask: GrayImage::new(32, 16),
        left: 4,
        top: 0,
    };

    let correction = estimate_focus_color_correction(
        &base,
        &base_mask,
        &GrayImage::new(40, 16),
        &candidate,
        false,
    );
    assert!(
        correction
            .gains
            .iter()
            .all(|gain| (*gain - 0.8).abs() < 0.01)
    );

    apply_focus_color_correction(
        &mut candidate_image,
        &correction,
        &GrayImage::new(32, 16),
        false,
    );
    let corrected = candidate_image.get_pixel(0, 0);
    assert!((corrected[0] - 0.4).abs() < 0.01);
    assert!((corrected[1] - 0.5).abs() < 0.01);
    assert!((corrected[2] - 0.6).abs() < 0.01);
}

#[test]
fn focus_color_correction_uses_only_consensus_foreground_overlap() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;
    let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.4, 0.4, 0.4]));
    let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |_, y| {
        if y < 24 {
            Rgb([0.5, 0.5, 0.5])
        } else {
            Rgb([0.4, 0.4, 0.4])
        }
    });
    let foreground_mask = GrayImage::from_fn(WIDTH, HEIGHT, |_, y| {
        image::Luma([if y < 24 { 255 } else { 0 }])
    });
    let candidate = RenderedFocusLayer {
        image: candidate_image,
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: foreground_mask.clone(),
        relaxed_foreground_mask: foreground_mask.clone(),
        left: 0,
        top: 0,
    };

    let correction = estimate_focus_color_correction(
        &base,
        &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        &foreground_mask,
        &candidate,
        false,
    );

    assert!(
        correction
            .gains
            .iter()
            .all(|gain| (*gain - 0.8).abs() < 0.01)
    );
}

#[test]
fn focus_color_correction_can_use_bright_consensus_edge_pixels() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;
    let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.94, 0.94, 0.94]));
    let candidate = RenderedFocusLayer {
        image: Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.98, 0.98, 0.98])),
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        relaxed_foreground_mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        left: 0,
        top: 0,
    };

    let correction = estimate_focus_color_correction(
        &base,
        &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        &candidate,
        false,
    );

    assert!(
        correction
            .gains
            .iter()
            .all(|gain| (*gain - (0.94 / 0.98)).abs() < 0.01)
    );
}

#[test]
fn focus_color_correction_reduces_a_broad_local_gain_step() {
    const WIDTH: u32 = 1_536;
    const HEIGHT: u32 = 32;
    let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.4, 0.5, 0.6]));
    let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
        if x < WIDTH / 2 {
            Rgb([0.5, 0.625, 0.75])
        } else {
            Rgb([0.4, 0.5, 0.6])
        }
    });
    let candidate = RenderedFocusLayer {
        image: candidate_image.clone(),
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        left: 0,
        top: 0,
    };
    let base_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let correction = estimate_focus_color_correction(
        &base,
        &base_mask,
        &GrayImage::new(WIDTH, HEIGHT),
        &candidate,
        false,
    );
    assert!(correction.spatial_gains.is_some());

    let mut corrected = candidate_image;
    apply_focus_color_correction(
        &mut corrected,
        &correction,
        &GrayImage::new(WIDTH, HEIGHT),
        false,
    );
    let input_step = 0.5f32 - 0.4;
    let corrected_step =
        (corrected.get_pixel(100, 10)[0] - corrected.get_pixel(1_300, 10)[0]).abs();
    assert!(corrected_step < input_step * 0.8);
}

#[test]
fn focus_background_tone_harmonization_reduces_step_without_touching_foreground() {
    const WIDTH: u32 = 192;
    const HEIGHT: u32 = 96;
    let mut image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        if (80..112).contains(&x) && (32..64).contains(&y) {
            Rgb([0.65, 0.10, 0.08])
        } else if x < WIDTH / 2 {
            Rgb([0.32, 0.28, 0.24])
        } else {
            Rgb([0.40, 0.36, 0.32])
        }
    });
    let original_foreground_pixel = *image.get_pixel(96, 48);
    let image_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let foreground_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
        image::Luma([u8::from((80..112).contains(&x) && (32..64).contains(&y)) * 255])
    });

    let input_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();
    harmonize_focus_background_tone(&mut image, &image_mask, &foreground_mask);
    let output_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();

    assert!(output_step < input_step * 0.5);
    assert_eq!(*image.get_pixel(96, 48), original_foreground_pixel);
}

#[test]
fn focus_owner_tone_harmonization_matches_source_blocks_without_touching_detail() {
    const WIDTH: u32 = 192;
    const HEIGHT: u32 = 96;
    let mut image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        if (80..112).contains(&x) && (32..64).contains(&y) {
            Rgb([0.65, 0.10, 0.08])
        } else if x < WIDTH / 2 {
            Rgb([0.32, 0.28, 0.24])
        } else {
            Rgb([0.40, 0.36, 0.32])
        }
    });
    let original_foreground_pixel = *image.get_pixel(96, 48);
    let image_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let foreground_mask = GrayImage::new(WIDTH, HEIGHT);
    let owner_map = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
        image::Luma([if x < WIDTH / 2 { 1 } else { 2 }])
    });

    let input_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();
    harmonize_focus_background_tone_with_owners(
        &mut image,
        &image_mask,
        &foreground_mask,
        &owner_map,
    );
    let output_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();

    assert!(output_step < input_step * 0.5);
    assert_eq!(*image.get_pixel(96, 48), original_foreground_pixel);
}

#[test]
fn focus_canvas_margin_fill_keeps_the_full_canvas_and_extends_texture() {
    let mut image = Rgb32FImage::from_fn(8, 6, |x, y| {
        let value = (x as f32 * 0.07 + y as f32 * 0.11).fract();
        Rgb([value, value * 0.8, value * 0.6])
    });
    let mut mask = GrayImage::new(8, 6);
    for y in 1..5 {
        for x in 1..7 {
            mask.put_pixel(x, y, image::Luma([255]));
        }
    }

    let dimensions = image.dimensions();
    let (filled, remaining) = fill_focus_canvas_margins(&mut image, &mut mask);

    assert_eq!(image.dimensions(), dimensions);
    assert!(filled > 0);
    assert_eq!(remaining, 0);
    assert!(mask.as_raw().iter().all(|&value| value > 0));
    assert!(image.get_pixel(0, 0).0.iter().any(|&value| value != 0.0));
}
