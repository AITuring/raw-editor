/// The exact shape requirement 12.7 fixes for a machine readable failure reason.
fn reason_identifier_pattern() -> Regex {
    Regex::new(r"^[a-z0-9_]{1,64}$").expect("the reason identifier pattern must compile")
}

// Runtime evidence for Property 44.  The proptest below checks the composed
// production coordinate map for 100 cases; this test enters the same layered
// compositor and counts its full-resolution Virtual_Tile samples.  With one
// station there is no ownership competition, so every covered final pixel must
// be read exactly once.
#[test]
fn property_44_layered_compositor_samples_each_covered_pixel_once() {
    let _run_scope = degradation::begin_run_scope();
    let side = 96;
    let info = single_frame_info(0, side);
    let source = single_frame_source(side, 0x44_11, 1.0);
    let homographies = HashMap::from([(info.id, Matrix3::identity())]);
    let rendered = render_layered_ownership(&[info], &homographies, &[source])
        .expect("single station must render through the production compositor");
    let covered_pixels = rendered
        .coverage
        .covered()
        .iter()
        .filter(|&&covered| covered != 0)
        .count();
    let sampled_pixels = rendered.full_resolution_samples;
    let final_pixels = rendered.image.width() as usize * rendered.image.height() as usize;
    assert_eq!(
        rendered.full_resolution_sample_counts.len(),
        final_pixels,
        "the sampling audit must have one counter for every final pixel"
    );
    assert!(
        rendered
            .full_resolution_sample_counts
            .iter()
            .all(|&count| count <= 1),
        "the production Tile_Compositor must resample each final pixel at most once"
    );
    assert!(
        sampled_pixels <= final_pixels,
        "the production Tile_Compositor must resample each final pixel at most once"
    );
    assert!(
        sampled_pixels >= covered_pixels,
        "every covered final pixel must have a source sample"
    );
    assert_eq!(
        rendered
            .ownership
            .owners()
            .iter()
            .filter(|&&owner| owner != stitching::NO_OWNER)
            .count(),
        covered_pixels,
        "the sampling audit must cover the same pixels as the published ownership plane"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 44: 瓦片到世界的逆映射
    // 先组合 Residual_Warp 与 tile_to_world_inverse，再进行一次坐标投影，不能在
    // 中间生成第二个采样图像。
    //
    // **Validates: Requirements 8.7**
    #[test]
    fn property_44_tile_world_mapping_is_one_composed_coordinate(
        world_x in -1000.0f64..1000.0,
        world_y in -1000.0f64..1000.0,
        tile_tx in -100.0f64..100.0,
        tile_ty in -100.0f64..100.0,
        warp_tx in -100.0f64..100.0,
        warp_ty in -100.0f64..100.0,
    ) {
        let _run_scope = degradation::begin_run_scope();
        let world_x = 256.0 + world_x.rem_euclid(64.0);
        let world_y = 256.0 + world_y.rem_euclid(64.0);
        let warp_tx = warp_tx.rem_euclid(32.0) - 16.0;
        let warp_ty = warp_ty.rem_euclid(32.0) - 16.0;
        let mut tile_inverse = Matrix3::identity();
        tile_inverse[(0, 2)] = tile_tx;
        tile_inverse[(1, 2)] = tile_ty;
        let overlap = super::residual_warp::OverlapResidual::new(
            0,
            7,
            super::report::WorldRect {
                left: 0.0,
                top: 0.0,
                width: 512.0,
                height: 512.0,
            },
            4.0,
        );
        let observations = (0..=8)
            .flat_map(|row| {
                (0..=8).map(move |column| super::residual_warp::WarpObservation::new(
                    super::report::WorldPoint {
                        x: f64::from(column) * 64.0,
                        y: f64::from(row) * 64.0,
                    },
                    [warp_tx, warp_ty],
                    0.0,
                ))
            })
            .collect::<Vec<_>>();
        let mut residual = super::residual_warp::ResidualWarp::new();
        prop_assert!(residual.add_observations(overlap, &observations).is_some());
        let image = single_frame_info(7, 2048);
        let coordinate = stitching::map_target_to_source_with_residual(
            &tile_inverse,
            Point3::new(world_x, world_y, 1.0),
            &image,
            stitching::Projection::Planar,
            &residual,
        )
        .expect("finite composed inverse coordinate");
        prop_assert!((coordinate.x - (world_x + tile_tx - warp_tx)).abs() < 1.0e-8);
        prop_assert!((coordinate.y - (world_y + tile_ty - warp_ty)).abs() < 1.0e-8);
    }

    // Feature: layered-camera-group-focus-stitching, Property 49: Tone_Harmonizer
    // applies the low-frequency correction while retaining the owner's exact
    // high-frequency pixels. The production group entry point is exercised;
    // the oracle independently measures a sigma >= 64px full-resolution
    // Gaussian field and does not call the production low-frequency helper.
    //
    // **Validates: Requirements 9.2**
    #[test]
    fn property_49_tone_keeps_owner_high_frequency_residual(owner_seed in any::<u8>()) {
        // `harmonize_tone_tiles` records tone degradations in the process-wide
        // run ledger, so it must run inside the run scope like a real run.
        let _run_scope = degradation::begin_run_scope();
        degradation::reset_run_ledger();
        let width = 128u32;
        let height = 128u32;
        let level = 0.30 + f32::from(owner_seed % 80) / 400.0;
        let source_level = (level * (0.82 + f32::from(owner_seed % 11) / 100.0)).clamp(0.08, 0.9);
        let owner = image::Rgb32FImage::from_fn(width, height, |x, y| {
            let broad = (f32::from(((x * 7 + y * 11) % 97) as u8) / 96.0) * 0.012;
            let residual = if (x + y + u32::from(owner_seed)) % 2 == 0 {
                0.028
            } else {
                -0.028
            };
            image::Rgb([
                level + broad + residual,
                level * 0.92 + broad + residual,
                level * 0.84 + broad + residual,
            ])
        });
        let mut panorama = owner.clone();
        let constant_tile = |station_index: usize, owner_id: u16, value: f32| tone::ToneTile {
            station_index,
            owner_id,
            low: image::Rgb32FImage::from_pixel(width, height, image::Rgb([value; 3])),
            validity: image::GrayImage::from_pixel(width, height, image::Luma([255])),
            world_origin: (0.0, 0.0),
            world_size: (f64::from(width), f64::from(height)),
            world_stride: 1.0,
            cell_mean: image::Rgb32FImage::from_pixel(width, height, image::Rgb([value; 3])),
            cell_coverage: vec![1.0; (width * height) as usize],
        };
        let tiles = [
            constant_tile(0, 1, level),
            constant_tile(1, 2, source_level),
        ];
        let owners = vec![1u16; (width * height) as usize];
        let evidence = image::GrayImage::from_pixel(width, height, image::Luma([255]));
        let report = tone::harmonize_tone_tiles(
            &mut panorama,
            &owners,
            width,
            &tiles,
            &evidence,
            &[(0, 1)],
        );
        prop_assert_eq!(report.status, ToneStatus::Applied);
        prop_assert!(report.tiles.iter().all(|tile| tile.samples >= tone::TONE_MIN_SAMPLES as u64));

        // Independent oracle: direct full-resolution Gaussian blur with a
        // 64px standard deviation, rather than `low_frequency_field`.
        let sigma = 64.0f32;
        let owner_low = image::imageops::blur(&owner, sigma);
        let corrected_low = image::imageops::blur(&panorama, sigma);
        let kernel_taps = (6.0 * sigma).ceil() + 1.0;
        let tolerance = 8.0 * kernel_taps * f32::EPSILON * (1.0 + level);
        for (index, (owner_pixel, corrected_pixel)) in owner_low
            .pixels()
            .zip(corrected_low.pixels())
            .enumerate()
        {
            let original = owner.get_pixel((index as u32) % width, (index as u32) / width);
            let corrected = panorama.get_pixel((index as u32) % width, (index as u32) / width);
            for channel in 0..3 {
                let owner_residual = original[channel] - owner_pixel[channel];
                let corrected_residual = corrected[channel] - corrected_pixel[channel];
                prop_assert!(
                    (owner_residual - corrected_residual).abs() <= tolerance,
                    "high-frequency residual drift at {index}:{channel}: {} > {}",
                    (owner_residual - corrected_residual).abs(),
                    tolerance
                );
            }
        }
    }

    // Feature: layered-camera-group-focus-stitching, Property 48: Tone_Harmonizer
    // 只有双覆盖、亮度可用且通过一致性筛选的样本进入色调求解。
    //
    // **Validates: Requirements 9.1, 9.4**
    #[test]
    fn property_48_tone_samples_are_usable_and_consistent(seed in any::<u8>()) {
        let value = 0.25 + f32::from(seed % 40) / 200.0;
        let side = 32u32;
        let tile = |coverage: Vec<f32>, pixel: [f32; 3]| {
            tone::ToneTile {
                station_index: 0,
                owner_id: 1,
                low: image::Rgb32FImage::from_pixel(side, side, image::Rgb(pixel)),
                validity: image::GrayImage::from_pixel(side, side, image::Luma([255])),
                world_origin: (0.0, 0.0),
                world_size: (f64::from(side), f64::from(side)),
                world_stride: 1.0,
                cell_mean: image::Rgb32FImage::from_pixel(side, side, image::Rgb(pixel)),
                cell_coverage: coverage,
            }
        };
        let full = vec![1.0; tone::TONE_MIN_SAMPLES];
        let first = tile(full.clone(), [value; 3]);
        let second = tile(full, [value * 0.9; 3]);
        let complete = tone::PairField::new(&first, &second)
            .expect("the accepted overlap has a field")
            .samples();
        prop_assert_eq!(complete.len(), tone::TONE_MIN_SAMPLES);
        let all_usable = complete.iter().all(|sample| {
            let luminance = |pixel: [f32; 3]| {
                0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
            };
            (tone::TONE_MIN_SAMPLE_LUMINANCE..=tone::TONE_MAX_SAMPLE_LUMINANCE)
                .contains(&luminance(sample.owner))
                && (tone::TONE_MIN_SAMPLE_LUMINANCE..=tone::TONE_MAX_SAMPLE_LUMINANCE)
                    .contains(&luminance(sample.source))
        });
        prop_assert!(all_usable);
        let mut partial_coverage = vec![1.0; tone::TONE_MIN_SAMPLES];
        partial_coverage[usize::from(seed) % tone::TONE_MIN_SAMPLES] = 0.0;
        let partial = tile(partial_coverage, [value; 3]);
        let partial_samples = tone::PairField::new(&partial, &second)
            .expect("the overlap remains geometrically valid")
            .samples();
        prop_assert_eq!(partial_samples.len(), tone::TONE_MIN_SAMPLES - 1);
        let mut with_outlier = complete;
        with_outlier.push(tone::ToneSample {
            owner: [0.9; 3],
            source: [0.1; 3],
        });
        let retained = tone::consistent_samples(&with_outlier);
        prop_assert_eq!(retained.len(), tone::TONE_MIN_SAMPLES);
    }

    // Feature: layered-camera-group-focus-stitching, Property 50: Tone_Harmonizer
    // 增益和偏移在应用前始终被限制在设计边界内。
    //
    // **Validates: Requirements 9.3, 9.7, 9.9, 9.10**
    #[test]
    fn property_50_tone_solution_is_bounded(seed in any::<u8>()) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let source = 0.2 + f32::from(seed % 80) / 200.0;
        let samples = vec![tone::ToneSample {
            owner: [(source * 1.4).min(0.99); 3],
            source: [source; 3],
        }; tone::TONE_MIN_SAMPLES];
        let solve = tone::solve_tone_pair(&samples);
        let gains_bounded = solve.gain.iter().all(|value| {
            (*value >= tone::TONE_MIN_GAIN) && (*value <= tone::TONE_MAX_GAIN)
        });
        prop_assert_eq!(gains_bounded, true);
        let offsets_bounded = solve.offset.iter().all(|value| value.abs() <= tone::TONE_MAX_ABS_OFFSET);
        prop_assert_eq!(offsets_bounded, true);
        prop_assert_eq!(solve.solved_gain.len(), solve.gain.len());
        prop_assert_eq!(solve.solved_offset.len(), solve.offset.len());
        for channel in 0..3 {
            prop_assert_eq!(solve.gain[channel], solve.solved_gain[channel].clamp(tone::TONE_MIN_GAIN, tone::TONE_MAX_GAIN));
            prop_assert_eq!(solve.offset[channel], solve.solved_offset[channel].clamp(-tone::TONE_MAX_ABS_OFFSET, tone::TONE_MAX_ABS_OFFSET));
        }
        let insufficient = tone::solve_tone_pair(&samples[..tone::TONE_MIN_SAMPLES - 1]);
        prop_assert_eq!(insufficient.status, tone::ToneSolveStatus::Identity);
        prop_assert_eq!(insufficient.gain, [1.0; 3]);
        prop_assert_eq!(insufficient.offset, [0.0; 3]);
    }

    // Feature: layered-camera-group-focus-stitching, Property 52: Tone_Harmonizer
    // 每个 Owner_Region 边界的 ΔE00 不超过 1.5；超过时必须记录并降级。
    //
    // **Validates: Requirements 9.8, 9.11**
    #[test]
    fn property_52_boundary_delta_is_bounded_or_degraded(
        left in prop::array::uniform3(0.0f32..=1.0),
        right in prop::array::uniform3(0.0f32..=1.0),
    ) {
        let _run_scope = degradation::begin_run_scope();
        degradation::reset_run_ledger();
        tone::reset_run_records();
        let left = left.map(|value| 0.05 + 0.9 * value);
        let right = right.map(|value| 0.05 + 0.9 * value);
        let width = 32u32;
        let height = 32u32;
        let tile = |station_index: usize, owner_id: u16, value: [f32; 3]| tone::ToneTile {
            station_index,
            owner_id,
            low: image::Rgb32FImage::from_pixel(width, height, image::Rgb(value)),
            validity: image::GrayImage::from_pixel(width, height, image::Luma([255])),
            world_origin: (0.0, 0.0),
            world_size: (f64::from(width), f64::from(height)),
            world_stride: 1.0,
            cell_mean: image::Rgb32FImage::from_pixel(width, height, image::Rgb(value)),
            cell_coverage: vec![1.0; (width * height) as usize],
        };
        let tiles = [tile(0, 1, left), tile(1, 2, right)];
        let owners = (0..height)
            .flat_map(|_| (0..width).map(|x| if x < width / 2 { 1 } else { 2 }))
            .collect::<Vec<_>>();
        let mut panorama = image::Rgb32FImage::from_fn(width, height, |x, _| {
            if x < width / 2 {
                image::Rgb(left)
            } else {
                image::Rgb(right)
            }
        });
        let evidence = image::GrayImage::from_pixel(width, height, image::Luma([255]));
        let report = tone::harmonize_tone_tiles(
            &mut panorama,
            &owners,
            width,
            &tiles,
            &evidence,
            &[(0, 1)],
        );
        let delta = report.boundary_delta_e.max;
        prop_assert!(delta.is_finite() && delta >= 0.0);
        if delta > super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD {
            prop_assert!(
                degradation::run_ledger_snapshot()
                    .entries()
                    .iter()
                    .any(|entry| entry.reason == degradation::TONE_BOUNDARY_DELTA_E_EXCEEDED)
            );
        } else {
            prop_assert!(delta <= super::report::TONE_BOUNDARY_DELTA_E_THRESHOLD);
        }
    }

    // Feature: layered-camera-group-focus-stitching, Property 51: Tone_Harmonizer
    // only writes panorama pixels; the production group-tone entry point never
    // mutates Coverage_Mask, seam positions, or the completed Ownership_Map.
    //
    // **Validates: Requirements 9.5, 9.6**
    #[test]
    fn property_51_tone_preserves_ownership_map(owner_seed in any::<u64>()) {
        // Same run-scope requirement as P49: tone writes the run ledger.
        let _run_scope = degradation::begin_run_scope();
        degradation::reset_run_ledger();
        let width = 64u32;
        let height = 64u32;
        let owners = (0..width * height)
            .map(|index| {
                if index % width < width / 2 {
                    1u16
                } else {
                    2u16
                }
            })
            .collect::<Vec<_>>();
        let before = owners.clone();
        let mut panorama = image::Rgb32FImage::from_fn(width, height, |x, y| {
            let value = (owner_seed.wrapping_add(u64::from(x * 17 + y * 31)) % 200) as f32 / 255.0;
            image::Rgb([value, value * 0.9, value * 0.8])
        });
        let panorama_before = panorama.clone();
        let tile = |station_index: usize, owner_id: u16, scale: f32| tone::ToneTile {
            station_index,
            owner_id,
            low: image::Rgb32FImage::from_pixel(width, height, image::Rgb([scale; 3])),
            validity: image::GrayImage::from_pixel(width, height, image::Luma([255])),
            world_origin: (0.0, 0.0),
            world_size: (f64::from(width), f64::from(height)),
            world_stride: 1.0,
            cell_mean: image::Rgb32FImage::from_pixel(width, height, image::Rgb([scale; 3])),
            cell_coverage: vec![1.0; (width * height) as usize],
        };
        let tiles = [tile(0, 1, 0.35), tile(1, 2, 0.55)];
        let evidence = image::GrayImage::from_fn(width, height, |x, y| {
            image::Luma([if (x * 3 + y + owner_seed as u32).is_multiple_of(17) {
                0
            } else {
                255
            }])
        });
        let coverage_before = stitching::CoverageMask::from_gray(evidence.clone());
        let coverage_bytes_before = coverage_before.covered().to_vec();
        let seam_positions = |map: &[u16]| {
            (0..height)
                .flat_map(|y| {
                    (0..width).filter_map(move |x| {
                        let index = y as usize * width as usize + x as usize;
                        let horizontal = x + 1 < width
                            && map[index] != map[index + 1];
                        let vertical = y + 1 < height
                            && map[index] != map[index + width as usize];
                        (horizontal || vertical).then_some((x, y))
                    })
                })
                .collect::<Vec<_>>()
        };
        let seam_before = seam_positions(&owners);
        let tile_coverage_before = tiles
            .iter()
            .map(|tile| tile.cell_coverage.clone())
            .collect::<Vec<_>>();
        let _report = tone::harmonize_tone_tiles(
            &mut panorama,
            &owners,
            width,
            &tiles,
            &evidence,
            &[(0, 1)],
        );
        compositor::assert_ownership_unchanged(&before, &owners);
        prop_assert_eq!(&before, &owners);
        let coverage_after = stitching::CoverageMask::from_gray(evidence.clone());
        prop_assert_eq!(coverage_after.covered(), coverage_bytes_before.as_slice());
        for y in 0..height {
            for x in 0..width {
                if evidence.get_pixel(x, y)[0] == 0 {
                    prop_assert_eq!(panorama.get_pixel(x, y), panorama_before.get_pixel(x, y));
                }
            }
        }
        prop_assert_eq!(seam_positions(&owners), seam_before);
        prop_assert_eq!(
            tiles.iter().map(|tile| tile.cell_coverage.clone()).collect::<Vec<_>>(),
            tile_coverage_before
        );
    }
}

// ---------------------------------------------------------------------------
// Properties 54–56: ownership compositor contract (13.9–13.11)
// ---------------------------------------------------------------------------

fn ownership_contract_fixture() -> (
    Vec<ImageInfo>,
    HashMap<usize, Matrix3<f64>>,
    Vec<image::Rgb32FImage>,
) {
    let side = 48;
    let infos = vec![single_frame_info(0, side), single_frame_info(1, side)];
    let homographies = HashMap::from([
        (0usize, Matrix3::identity()),
        (
            1usize,
            Matrix3::new(1.0, 0.0, 16.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0),
        ),
    ]);
    let sources = vec![
        single_frame_source(side, 11, 1.0),
        single_frame_source(side, 29, 2.0),
    ];
    (infos, homographies, sources)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 54: Layered_Virtual_Tile
    // 的画布边界等于所有 Coverage_Mask 的覆盖联合边界。
    #[test]
    fn property_54_layered_canvas_is_coverage_union(translation_x in -24i32..=24, translation_y in -24i32..=24) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let (infos, mut homographies, sources) = ownership_contract_fixture();
        homographies.insert(1, Matrix3::new(1.0, 0.0, f64::from(translation_x), 0.0, 1.0, f64::from(translation_y), 0.0, 0.0, 1.0));
        let rendered = render_layered_ownership(&infos, &homographies, &sources).expect("compositor render");
        let image_refs = infos.iter().collect::<Vec<_>>();
        let (min_x, max_x, min_y, max_y) =
            stitching::output_bounds(&image_refs, &homographies, stitching::Projection::Planar);
        let (expected_origin_x, expected_width) = stitching::pixel_aligned_canvas(min_x, max_x);
        let (expected_origin_y, expected_height) = stitching::pixel_aligned_canvas(min_y, max_y);
        prop_assert_eq!(
            rendered.sampling_origin,
            (-expected_origin_x, -expected_origin_y)
        );
        prop_assert_eq!(rendered.image.dimensions(), (expected_width, expected_height));
        let (width, height) = rendered.coverage.dimensions();
        let mut union_bounds = (u32::MAX, u32::MAX, 0u32, 0u32);
        for y in 0..height {
            for x in 0..width {
                if rendered.coverage.is_covered(x, y) {
                    union_bounds.0 = union_bounds.0.min(x);
                    union_bounds.1 = union_bounds.1.min(y);
                    union_bounds.2 = union_bounds.2.max(x);
                    union_bounds.3 = union_bounds.3.max(y);
                }
            }
        }
        prop_assert!(union_bounds.0 != u32::MAX);
        prop_assert_eq!(union_bounds.0, 0);
        prop_assert_eq!(union_bounds.1, 0);
        prop_assert_eq!(union_bounds.2 + 1, expected_width);
        prop_assert_eq!(union_bounds.3 + 1, expected_height);
    }

    // Feature: layered-camera-group-focus-stitching, Property 55: 未覆盖像素保持透明且从不被写入。
    #[test]
    fn property_55_layered_uncovered_pixels_are_transparent(translation_x in 96i32..=180) {
        let (infos, mut homographies, sources) = ownership_contract_fixture();
        homographies.insert(1, Matrix3::new(1.0, 0.0, f64::from(translation_x), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0));
        let rendered = render_layered_ownership(&infos, &homographies, &sources).expect("compositor render");
        let mut uncovered = 0usize;
        for (x, y, pixel) in rendered.image.enumerate_pixels() {
            if !rendered.coverage.is_covered(x, y) {
                uncovered += 1;
                prop_assert_eq!(rendered.ownership.owner_at(x, y), stitching::NO_OWNER);
                prop_assert!(pixel.0.iter().all(|&channel| channel == 0.0));
            }
        }
        prop_assert!(uncovered > 0);
    }

    // Feature: layered-camera-group-focus-stitching, Property 56: 输出 Ownership_Map 等于瓦片 ownership。
    #[test]
    fn property_56_layered_output_ownership_matches_tile(translation_x in -12i32..=28) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let (infos, mut homographies, sources) = ownership_contract_fixture();
        homographies.insert(1, Matrix3::new(1.0, 0.0, f64::from(translation_x), 0.0, 1.0, 4.0, 0.0, 0.0, 1.0));
        let rendered = render_layered_ownership(&infos, &homographies, &sources).expect("compositor render");
        let result = assert_source_correspondence(&rendered, &infos, &homographies, &sources, false, true);
        prop_assert!(result.is_ok());
        let (_, _, uncovered) = result.expect("ownership correspondence");
        prop_assert!(uncovered < (rendered.image.width() * rendered.image.height()) as usize);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 86: 内存门槛解析是确定的。
    #[test]
    fn property_86_memory_threshold_resolution_is_deterministic(
        physical in 0u64..(64 * 1024 * 1024 * 1024),
        configured in prop::option::of(0u64..(64 * 1024 * 1024 * 1024)),
        auto_calibrated in any::<bool>(),
    ) {
        let first = resources::resolve_memory_threshold(physical, configured, auto_calibrated);
        let second = resources::resolve_memory_threshold(physical, configured, auto_calibrated);
        prop_assert_eq!(first, second);
        let upper = ((physical as f64) * resources::MEMORY_THRESHOLD_PHYSICAL_RATIO).floor()
            as u64;
        let expected = if upper < resources::MEMORY_THRESHOLD_MIN_BYTES {
            resources::MemoryThreshold {
                bytes: upper,
                source: resources::MemoryThresholdSource::AutoCalibrated,
            }
        } else {
            let lower = resources::MEMORY_THRESHOLD_MIN_BYTES;
            let clamp = |value: u64| value.max(lower).min(upper);
            match configured {
                Some(value) => resources::MemoryThreshold {
                    bytes: clamp(value),
                    source: resources::MemoryThresholdSource::UserConfigured,
                },
                None if auto_calibrated => resources::MemoryThreshold {
                    bytes: upper,
                    source: resources::MemoryThresholdSource::AutoCalibrated,
                },
                None => resources::MemoryThreshold {
                    bytes: clamp(resources::MEMORY_THRESHOLD_DEFAULT_BYTES),
                    source: resources::MemoryThresholdSource::Default,
                },
            }
        };
        prop_assert_eq!(first, expected);
    }
}

// ---------------------------------------------------------------------------
// Property 94 / acceptance harness skeleton (17.5, 17.15)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
struct StackAcceptanceFailure {
    identifier: &'static str,
    measured: String,
    threshold: String,
}

/// The one production acceptance decision.  The harness and P94 both call
/// this function; neither maintains a second copy of the acceptance rules.
fn stack_acceptance_failures(report: &StackReport) -> Vec<StackAcceptanceFailure> {
    let mut failures = Vec::new();
    let mut add = |identifier: &'static str, measured: String, threshold: String| {
        failures.push(StackAcceptanceFailure {
            identifier,
            measured,
            threshold,
        });
    };
    let sources = report
        .input
        .sources
        .iter()
        .map(|source| source.path.as_str())
        .collect::<BTreeSet<_>>();
    let members = report
        .grouping
        .stations
        .iter()
        .flat_map(|station| station.members.iter().map(String::as_str))
        .collect::<Vec<_>>();
    let grouped = members.iter().copied().collect::<BTreeSet<_>>();
    if report.input.source_count != 84
        || sources.len() != 84
        || members.len() != 84
        || grouped != sources
    {
        add(
            "acceptance_source_count",
            format!(
                "input={}, grouped={}, unique={}",
                report.input.source_count,
                members.len(),
                grouped.len()
            ),
            "84".to_string(),
        );
    }
    let station_indices = report
        .grouping
        .stations
        .iter()
        .map(|station| station.index)
        .collect::<BTreeSet<_>>();
    if report.grouping.station_count != 28
        || report.grouping.stations.len() != 28
        || station_indices != (0..28).collect()
    {
        add(
            "acceptance_station_count",
            format!(
                "reported={}, records={}, unique_indices={}",
                report.grouping.station_count,
                report.grouping.stations.len(),
                station_indices.len()
            ),
            "28 stations indexed 0..28".to_string(),
        );
    }
    if !report.grouping.isolated.is_empty() {
        add(
            "acceptance_isolated_sources",
            report.grouping.isolated.len().to_string(),
            "0".to_string(),
        );
    }
    let undecoded_input_count = report
        .input
        .sources
        .iter()
        .filter(|source| !source.decoded)
        .count();
    if undecoded_input_count != 0 || !report.grouping.undecodable.is_empty() {
        add(
            "acceptance_undecodable_sources",
            format!(
                "input={}, grouping={}",
                undecoded_input_count,
                report.grouping.undecodable.len()
            ),
            "input=0, grouping=0".to_string(),
        );
    }

    let mut relation_components = (0..28).collect::<Vec<_>>();
    let mut relation_endpoints_valid = true;
    for relation in &report.station_relations.accepted {
        if relation.left >= 28 || relation.right >= 28 {
            relation_endpoints_valid = false;
            continue;
        }
        let from = relation_components[relation.right];
        let to = relation_components[relation.left];
        for component in &mut relation_components {
            if *component == from {
                *component = to;
            }
        }
    }
    let relation_component_count = relation_components
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    if !relation_endpoints_valid || relation_component_count != 1 {
        add(
            "acceptance_relation_graph",
            format!(
                "endpoints_valid={relation_endpoints_valid}, components={relation_component_count}"
            ),
            "all endpoints in 0..28 and one component".to_string(),
        );
    }

    let connectivity_members = report
        .station_relations
        .connectivity
        .component_members
        .first()
        .map(|members| members.iter().map(String::as_str).collect::<BTreeSet<_>>());
    if report.station_relations.connectivity.components != 1
        || report
            .station_relations
            .connectivity
            .component_members
            .len()
            != 1
        || report
            .station_relations
            .connectivity
            .component_member_counts
            != [84]
        || connectivity_members.as_ref() != Some(&sources)
        || grouped != sources
    {
        add(
            "acceptance_connectivity_membership",
            format!(
                "components={}, member_lists={}, counts={:?}, unique_members={}",
                report.station_relations.connectivity.components,
                report
                    .station_relations
                    .connectivity
                    .component_members
                    .len(),
                report
                    .station_relations
                    .connectivity
                    .component_member_counts,
                connectivity_members.as_ref().map_or(0, BTreeSet::len)
            ),
            "one component containing all 84 grouped inputs".to_string(),
        );
    }
    if report.quality_gate.verdict != QualityGateVerdict::Pass {
        add(
            "acceptance_quality_gate_verdict",
            serde_json::to_string(&report.quality_gate.verdict).unwrap_or_default(),
            "pass".to_string(),
        );
    }
    let failed_criteria = report
        .quality_gate
        .criteria
        .iter()
        .filter(|criterion| {
            !criterion.diagnostic
                && (criterion.verdict == QualityGateVerdict::Fail || !criterion.failed.is_empty())
        })
        .count();
    if failed_criteria != 0 {
        add(
            "acceptance_failed_criteria",
            failed_criteria.to_string(),
            "0".to_string(),
        );
    }
    let union = report.composition.union_projected_pixels;
    let opaque_ratio = if union == 0 {
        0.0
    } else {
        report.composition.opaque_pixels as f64 / union as f64
    };
    if union == 0 || opaque_ratio < 0.98 {
        add(
            "acceptance_opaque_union_ratio",
            format!("{opaque_ratio:.9}"),
            "0.98".to_string(),
        );
    }
    if report.resources.peak_rss_bytes > report.resources.memory_threshold_bytes {
        add(
            "acceptance_peak_rss",
            report.resources.peak_rss_bytes.to_string(),
            report.resources.memory_threshold_bytes.to_string(),
        );
    }
    if report.resources.network_requests != 0 {
        add(
            "acceptance_network_requests",
            report.resources.network_requests.to_string(),
            "0".to_string(),
        );
    }
    let fusion_indices = report
        .fusion
        .iter()
        .map(|fusion| fusion.station_index)
        .collect::<BTreeSet<_>>();
    if report.fusion.len() != 28 || fusion_indices != (0..28).collect() {
        add(
            "acceptance_fusion_coverage",
            format!(
                "records={}, unique_indices={}",
                report.fusion.len(),
                fusion_indices.len()
            ),
            "28 stations indexed 0..28".to_string(),
        );
    }
    let non_graph_cut = report
        .fusion
        .iter()
        .filter(|fusion| fusion.solver_status != FusionSolverStatus::GraphCut)
        .count();
    if non_graph_cut != 0 {
        add(
            "acceptance_fusion_solver_status",
            non_graph_cut.to_string(),
            "0".to_string(),
        );
    }
    let changed_sources = report
        .input
        .sources
        .iter()
        .filter(|source| {
            source.sha256_before.is_empty()
                || source.sha256_after.is_empty()
                || source.sha256_before != source.sha256_after
        })
        .count();
    if report.input.sources.len() != 84 || changed_sources != 0 {
        add(
            "acceptance_source_sha256_stable",
            format!(
                "records={}, changed={changed_sources}",
                report.input.sources.len()
            ),
            "84 records, changed=0".to_string(),
        );
    }
    failures
}

fn stack_acceptance_baseline_report() -> StackReport {
    let mut report = StackReport::default();
    let paths = (0..84)
        .map(|index| format!("/fixture/DSC_{:04}.NEF", 3680 + index))
        .collect::<Vec<_>>();
    report.input.source_count = 84;
    report.input.sources = paths
        .iter()
        .map(|path| SourceRecord {
            path: path.clone(),
            sha256_before: "a".repeat(64),
            sha256_after: "a".repeat(64),
            decoded: true,
        })
        .collect();
    report.grouping.station_count = 28;
    report.grouping.stations = paths
        .chunks_exact(3)
        .enumerate()
        .map(|(index, members)| super::report::StationRecord {
            index,
            members: members.to_vec(),
            ..super::report::StationRecord::default()
        })
        .collect();
    report.station_relations.accepted = (0..27)
        .map(|left| super::report::AcceptedStationRelationRecord {
            left,
            right: left + 1,
            evidence_kind: super::report::StationRelationEvidenceKind::VirtualTile,
            inliers: 24,
            spatial_support: 0.2,
            scale_ratio: 1.0,
            median_error_px: 1.0,
            independent_support: 1,
        })
        .collect();
    report.station_relations.connectivity = ConnectivityReport {
        components: 1,
        component_members: vec![paths.clone()],
        component_member_counts: vec![84],
    };
    report.composition.union_projected_pixels = 10_000;
    report.composition.opaque_pixels = 9_800;
    report.resources.memory_threshold_bytes = 100;
    report.resources.peak_rss_bytes = 100;
    report.resources.network_requests = 0;
    report.fusion = (0..28)
        .map(|station_index| FusionReport {
            station_index,
            solver_status: FusionSolverStatus::GraphCut,
            ..FusionReport::default()
        })
        .collect();
    report.quality_gate.verdict = QualityGateVerdict::Pass;
    report.quality_gate.criteria = vec![QualityGateCriterionRecord {
        name: "baseline".to_string(),
        verdict: QualityGateVerdict::Pass,
        ..QualityGateCriterionRecord::default()
    }];
    report
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AcceptanceViolationKind {
    SourceCount,
    StationCount,
    Isolated,
    Undecodable,
    RelationGraph,
    ConnectivityMembership,
    FusionCoverage,
    QualityVerdict,
    FailedCriterion,
    OpaqueRatio,
    PeakRss,
    Network,
    Solver,
    SourceDigest,
}

fn inject_acceptance_violation(
    report: &mut StackReport,
    kind: AcceptanceViolationKind,
) -> &'static str {
    match kind {
        AcceptanceViolationKind::SourceCount => {
            report.input.source_count = 83;
            "acceptance_source_count"
        }
        AcceptanceViolationKind::StationCount => {
            report.grouping.station_count = 27;
            "acceptance_station_count"
        }
        AcceptanceViolationKind::Isolated => {
            report.grouping.isolated.push(Default::default());
            "acceptance_isolated_sources"
        }
        AcceptanceViolationKind::Undecodable => {
            report.grouping.undecodable.push(Default::default());
            "acceptance_undecodable_sources"
        }
        AcceptanceViolationKind::RelationGraph => {
            report.station_relations.accepted.pop();
            "acceptance_relation_graph"
        }
        AcceptanceViolationKind::ConnectivityMembership => {
            report.station_relations.connectivity.components = 2;
            "acceptance_connectivity_membership"
        }
        AcceptanceViolationKind::FusionCoverage => {
            report.fusion.pop();
            "acceptance_fusion_coverage"
        }
        AcceptanceViolationKind::QualityVerdict => {
            report.quality_gate.verdict = QualityGateVerdict::Fail;
            "acceptance_quality_gate_verdict"
        }
        AcceptanceViolationKind::FailedCriterion => {
            report.quality_gate.criteria[0].verdict = QualityGateVerdict::Fail;
            report.quality_gate.criteria[0]
                .failed
                .push(Default::default());
            "acceptance_failed_criteria"
        }
        AcceptanceViolationKind::OpaqueRatio => {
            report.composition.opaque_pixels = 9_700;
            "acceptance_opaque_union_ratio"
        }
        AcceptanceViolationKind::PeakRss => {
            report.resources.peak_rss_bytes = 101;
            "acceptance_peak_rss"
        }
        AcceptanceViolationKind::Network => {
            report.resources.network_requests = 1;
            "acceptance_network_requests"
        }
        AcceptanceViolationKind::Solver => {
            report.fusion[0].solver_status = FusionSolverStatus::PerCellFallback;
            "acceptance_fusion_solver_status"
        }
        AcceptanceViolationKind::SourceDigest => {
            report.input.sources[0].sha256_after = "b".repeat(64);
            "acceptance_source_sha256_stable"
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]
    // Feature: layered-camera-group-focus-stitching, Property 94: 门禁 verdict 是纯函数。
    #[test]
    fn property_94_stack_acceptance_verdict_is_pure(
        violations in prop::collection::vec(any::<bool>(), 14),
    ) {
        let kinds = [
            AcceptanceViolationKind::SourceCount,
            AcceptanceViolationKind::StationCount,
            AcceptanceViolationKind::Isolated,
            AcceptanceViolationKind::Undecodable,
            AcceptanceViolationKind::RelationGraph,
            AcceptanceViolationKind::ConnectivityMembership,
            AcceptanceViolationKind::FusionCoverage,
            AcceptanceViolationKind::QualityVerdict,
            AcceptanceViolationKind::FailedCriterion,
            AcceptanceViolationKind::OpaqueRatio,
            AcceptanceViolationKind::PeakRss,
            AcceptanceViolationKind::Network,
            AcceptanceViolationKind::Solver,
            AcceptanceViolationKind::SourceDigest,
        ];
        let mut report = stack_acceptance_baseline_report();
        prop_assert!(stack_acceptance_failures(&report).is_empty());
        let expected = kinds
            .into_iter()
            .zip(violations)
            .filter(|&(_, inject)| inject)
            .map(|(kind, _)| inject_acceptance_violation(&mut report, kind))
            .collect::<BTreeSet<_>>();
        let failures = stack_acceptance_failures(&report);
        prop_assert_eq!(failures.iter().map(|failure| failure.identifier).collect::<BTreeSet<_>>(), expected);
    }
}

#[test]
fn acceptance_graph_membership_and_fusion_invariants_are_deterministic() {
    let baseline = stack_acceptance_baseline_report();
    assert!(stack_acceptance_failures(&baseline).is_empty());

    let mut out_of_bounds = baseline.clone();
    out_of_bounds.station_relations.accepted[0].left = 28;
    assert_eq!(
        stack_acceptance_failures(&out_of_bounds)
            .iter()
            .map(|failure| failure.identifier)
            .collect::<Vec<_>>(),
        ["acceptance_relation_graph"]
    );

    let mut duplicate_member = baseline.clone();
    duplicate_member
        .station_relations
        .connectivity
        .component_members[0][1] = duplicate_member
        .station_relations
        .connectivity
        .component_members[0][0]
        .clone();
    assert_eq!(
        stack_acceptance_failures(&duplicate_member)
            .iter()
            .map(|failure| failure.identifier)
            .collect::<Vec<_>>(),
        ["acceptance_connectivity_membership"]
    );

    for duplicate in [false, true] {
        let mut fusion = baseline.clone();
        if duplicate {
            fusion.fusion[27].station_index = 26;
        } else {
            fusion.fusion.pop();
        }
        assert_eq!(
            stack_acceptance_failures(&fusion)
                .iter()
                .map(|failure| failure.identifier)
                .collect::<Vec<_>>(),
            ["acceptance_fusion_coverage"]
        );
    }
}

#[test]
fn acceptance_verdict_rejects_an_input_source_record_marked_undecoded() {
    let mut report = stack_acceptance_baseline_report();
    report.input.sources[17].decoded = false;
    assert!(report.grouping.undecodable.is_empty());
    assert_eq!(
        stack_acceptance_failures(&report)
            .iter()
            .map(|failure| failure.identifier)
            .collect::<Vec<_>>(),
        ["acceptance_undecodable_sources"]
    );
}

fn write_acceptance_source_fixture(source_dir: &Path) {
    fs::create_dir_all(source_dir).expect("create acceptance source fixture");
    for index in 3680..=3763 {
        fs::write(source_dir.join(format!("DSC_{index}.NEF")), [index as u8])
            .expect("write acceptance source fixture");
    }
}

#[test]
fn validate_acceptance_directories_rejects_invalid_paths_and_sources() {
    let root = TempDir::new().expect("acceptance validation tempdir");
    let source = root.path().join("source");
    let report = root.path().join("report");
    write_acceptance_source_fixture(&source);
    assert!(validate_acceptance_directories(&source, &report).is_ok());
    assert!(validate_acceptance_directories(Path::new("relative"), &report).is_err());
    assert!(validate_acceptance_directories(&source, Path::new("relative")).is_err());
    let nested_report = source.join("new/reports");
    let before_nested_rejection = acceptance_source_snapshot(&source);
    assert!(validate_acceptance_directories(&source, &nested_report).is_err());
    assert_eq!(
        acceptance_source_snapshot(&source),
        before_nested_rejection,
        "a rejected preflight must not change the source inventory"
    );
    assert!(!nested_report.exists());

    let missing_root = TempDir::new().expect("missing source tempdir");
    let missing_source = missing_root.path().join("source");
    fs::create_dir_all(&missing_source).unwrap();
    assert!(
        validate_acceptance_directories(&missing_source, &missing_root.path().join("report"))
            .is_err()
    );

    let directory_root = TempDir::new().expect("directory source tempdir");
    let directory_source = directory_root.path().join("source");
    write_acceptance_source_fixture(&directory_source);
    let invalid = directory_source.join("DSC_3680.NEF");
    fs::remove_file(&invalid).unwrap();
    fs::create_dir(&invalid).unwrap();
    assert!(
        validate_acceptance_directories(&directory_source, &directory_root.path().join("report"))
            .is_err()
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let symlink_root = TempDir::new().expect("symlink source tempdir");
        let symlink_source = symlink_root.path().join("source");
        write_acceptance_source_fixture(&symlink_source);
        let invalid = symlink_source.join("DSC_3680.NEF");
        fs::remove_file(&invalid).unwrap();
        symlink(symlink_source.join("DSC_3681.NEF"), &invalid).unwrap();
        assert!(
            validate_acceptance_directories(&symlink_source, &symlink_root.path().join("report"))
                .is_err()
        );
    }
}

#[test]
fn stack_report_selection_requires_one_matching_regular_file() {
    let directory = TempDir::new().expect("report selection tempdir");
    fs::write(directory.path().join("other.json"), b"{}").unwrap();
    assert!(select_unique_stack_report(directory.path()).is_err());
    let first = directory.path().join("stack-report-first.json");
    fs::write(&first, b"{}").unwrap();
    assert_eq!(select_unique_stack_report(directory.path()).unwrap(), first);
    fs::write(directory.path().join("stack-report-second.json"), b"{}").unwrap();
    assert!(select_unique_stack_report(directory.path()).is_err());
}

fn resolve_acceptance_path_without_writes(path: &Path) -> Result<PathBuf, String> {
    let mut resolved = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            std::path::Component::RootDir => resolved.push(std::path::MAIN_SEPARATOR.to_string()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                resolved.pop();
            }
            std::path::Component::Normal(name) => {
                let candidate = resolved.join(name);
                match fs::symlink_metadata(&candidate) {
                    Ok(_) => {
                        resolved = candidate.canonicalize().map_err(|error| {
                            format!("cannot canonicalize {}: {error}", candidate.display())
                        })?;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        resolved = candidate;
                    }
                    Err(error) => {
                        return Err(format!(
                            "cannot inspect acceptance path {}: {error}",
                            candidate.display()
                        ));
                    }
                }
            }
        }
    }
    Ok(resolved)
}

fn validate_acceptance_directories(
    source_dir: &Path,
    report_dir: &Path,
) -> Result<(PathBuf, PathBuf, Vec<PathBuf>), String> {
    if !source_dir.is_absolute() || !report_dir.is_absolute() {
        return Err("acceptance source and report directories must be absolute paths".to_string());
    }
    let source_dir = source_dir.canonicalize().map_err(|error| {
        format!(
            "cannot canonicalize source {}: {error}",
            source_dir.display()
        )
    })?;
    if !source_dir.is_dir() {
        return Err(format!(
            "acceptance source is not a directory: {}",
            source_dir.display()
        ));
    }
    let report_dir = resolve_acceptance_path_without_writes(report_dir)?;
    if source_dir == report_dir
        || source_dir.starts_with(&report_dir)
        || report_dir.starts_with(&source_dir)
    {
        return Err(
            "acceptance source and report directories must be separate and non-nested".to_string(),
        );
    }
    fs::create_dir_all(&report_dir).map_err(|error| {
        format!(
            "cannot create report directory {}: {error}",
            report_dir.display()
        )
    })?;
    let report_dir = report_dir.canonicalize().map_err(|error| {
        format!(
            "cannot canonicalize report {}: {error}",
            report_dir.display()
        )
    })?;
    if !report_dir.is_dir() {
        return Err(format!(
            "acceptance report is not a directory: {}",
            report_dir.display()
        ));
    }

    let mut paths = Vec::with_capacity(84);
    for index in 3680..=3763 {
        let path = source_dir.join(format!("DSC_{index}.NEF"));
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("cannot read metadata for {}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(format!(
                "acceptance source must be a regular non-symlink file: {}",
                path.display()
            ));
        }
        path.to_str()
            .ok_or_else(|| format!("acceptance source path is not UTF-8: {}", path.display()))?;
        fs::File::open(&path).map_err(|error| {
            format!("cannot read acceptance source {}: {error}", path.display())
        })?;
        virtual_tile::source_file_sha256(&path).map_err(|error| {
            format!("cannot hash acceptance source {}: {error}", path.display())
        })?;
        paths.push(path);
    }
    Ok((source_dir, report_dir, paths))
}

fn select_unique_stack_report(report_dir: &Path) -> Result<PathBuf, String> {
    let mut reports = fs::read_dir(report_dir)
        .map_err(|error| {
            format!(
                "cannot read report directory {}: {error}",
                report_dir.display()
            )
        })?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_str()?;
            let regular = fs::symlink_metadata(&path)
                .ok()
                .is_some_and(|metadata| metadata.file_type().is_file());
            (regular && name.starts_with("stack-report-") && name.ends_with(".json"))
                .then_some(path)
        })
        .collect::<Vec<_>>();
    reports.sort();
    match reports.as_slice() {
        [report] => Ok(report.clone()),
        _ => Err(format!(
            "expected exactly one regular stack-report-*.json, found {}: {:?}",
            reports.len(),
            reports
        )),
    }
}

#[test]
#[ignore = "acceptance harness requires the supplied real dataset"]
fn stack_acceptance_harness() {
    let source_dir = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_SOURCE_DIR")
        .map(PathBuf::from)
        .expect("RAW_EDITOR_STACK_ACCEPTANCE_SOURCE_DIR is required");
    let report_dir = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_REPORT_DIR")
        .map(PathBuf::from)
        .expect("RAW_EDITOR_STACK_ACCEPTANCE_REPORT_DIR is required");
    let (source_dir, report_dir, paths) = validate_acceptance_directories(&source_dir, &report_dir)
        .unwrap_or_else(|error| panic!("acceptance directory validation failed: {error}"));
    let before = acceptance_source_snapshot(&source_dir);
    let source_strings = paths
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    // The harness owns this process-wide variable for the duration of the ignored run.
    unsafe {
        std::env::set_var("RAW_EDITOR_STACK_REPORT_DIR", &report_dir);
    }
    crate::sidecar_storage::initialize(Path::new(
        "/private/tmp/raw-editor-stack-acceptance-sidecars",
    ))
    .expect("acceptance sidecar storage should initialize");
    let app = tauri::test::mock_app();
    let pipeline_result = crate::panorama_stitching::stitch_images_with_options(
        source_strings,
        app.handle().clone(),
        crate::panorama_stitching::AlignmentMode::Auto,
        crate::panorama_stitching::BlendMode::FocusStack,
        "test-stack-acceptance-progress",
    );
    let after = acceptance_source_snapshot(&source_dir);
    assert_eq!(
        before, after,
        "acceptance source directory changed during run"
    );
    let report_path = select_unique_stack_report(&report_dir)
        .unwrap_or_else(|error| panic!("acceptance report selection failed: {error}"));
    let report: StackReport = serde_json::from_slice(
        &fs::read(&report_path).expect("acceptance report should be readable"),
    )
    .expect("acceptance report should match Stack_Report schema");
    if let Err(error) = &pipeline_result {
        eprintln!("acceptance pipeline stopped: {error}");
    }
    let failures = stack_acceptance_failures(&report);
    if !failures.is_empty() {
        eprintln!("acceptance unmet conditions: {failures:?}");
    }
    assert!(
        failures.is_empty() && pipeline_result.is_ok(),
        "acceptance conditions were not met"
    );
}

fn acceptance_source_snapshot(source_dir: &Path) -> BTreeMap<String, (u64, String)> {
    let required = (3680..=3763)
        .map(|index| format!("DSC_{index}.NEF"))
        .collect::<BTreeSet<_>>();
    fs::read_dir(source_dir)
        .expect("acceptance source directory should be readable")
        .map(|entry| {
            let entry = entry.expect("every source directory entry must be readable");
            let path = entry.path();
            let name = entry
                .file_name()
                .into_string()
                .expect("source filenames must be UTF-8");
            let size = entry
                .metadata()
                .expect("source metadata must be readable")
                .len();
            // Extra files are inventoried by name/size only; their contents are never read.
            let digest = if required.contains(&name) {
                virtual_tile::hex_digest(
                    &virtual_tile::source_file_sha256(&path)
                        .expect("every required NEF must be hashable"),
                )
            } else {
                String::new()
            };
            (name, (size, digest))
        })
        .collect::<BTreeMap<_, _>>()
}

/// Every identifier a Stack_Report is allowed to carry as a failure or
/// unmeasurable reason.
fn frozen_reason_identifiers() -> BTreeSet<&'static str> {
    FAILURE_REASONS
        .iter()
        .map(|(reason, _)| *reason)
        .chain(UNMEASURABLE_REASONS.iter().copied())
        .collect()
}

/// Recover the `'static` identifier behind a serialized reason string so the
/// same reason can be replayed through a fresh ledger.
fn static_failure_reason(reason: &str) -> Option<&'static str> {
    FAILURE_REASONS
        .iter()
        .find(|(candidate, _)| *candidate == reason)
        .map(|(candidate, _)| *candidate)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 79: 对于任意生效的失败或
    // 降级路径，其机器可读失败原因标识符仅由 ASCII 小写字母、数字和下划线组成、长度不超过
    // 64 个字符，且对同一失败原因在重复运行之间完全相同。
    //
    // **Validates: Requirements 12.7**
    #[test]
    fn failure_reason_identifiers_have_a_stable_format(report in arb_stack_report()) {
        let pattern = reason_identifier_pattern();
        let frozen = frozen_reason_identifiers();

        for entry in &report.degradation.entries {
            prop_assert!(
                pattern.is_match(&entry.reason),
                "degradation reason `{}` violates ^[a-z0-9_]{{1,64}}$",
                entry.reason
            );
            prop_assert!(
                entry.reason.len() <= MAX_REASON_IDENTIFIER_LENGTH,
                "degradation reason `{}` is longer than {MAX_REASON_IDENTIFIER_LENGTH} bytes",
                entry.reason
            );
            prop_assert!(
                frozen.contains(entry.reason.as_str()),
                "degradation reason `{}` is outside the frozen identifier set",
                entry.reason
            );
        }

        for record in &report.quality_gate.unmeasurable {
            prop_assert!(
                pattern.is_match(&record.reason),
                "unmeasurable reason `{}` violates ^[a-z0-9_]{{1,64}}$",
                record.reason
            );
            prop_assert!(
                record.reason.len() <= MAX_REASON_IDENTIFIER_LENGTH,
                "unmeasurable reason `{}` is longer than {MAX_REASON_IDENTIFIER_LENGTH} bytes",
                record.reason
            );
            prop_assert!(
                frozen.contains(record.reason.as_str()),
                "unmeasurable reason `{}` is outside the frozen identifier set",
                record.reason
            );
        }

        // Stability across repeated runs: replaying the same active paths
        // through a fresh ledger reproduces byte identical identifiers and the
        // same severity, so a consumer can key on them run after run.
        let mut replayed = DegradationLedger::new();
        for entry in &report.degradation.entries {
            let reason = static_failure_reason(&entry.reason)
                .expect("a recorded reason belongs to the frozen failure set");
            replayed.record(reason, Value::Null);
        }
        prop_assert_eq!(replayed.entries().len(), report.degradation.entries.len());
        for (replay, original) in replayed.entries().iter().zip(&report.degradation.entries) {
            prop_assert_eq!(replay.reason, original.reason.as_str());
            prop_assert_eq!(replay.severity.as_identifier(), original.severity.as_str());
        }

        // The serialized document repeats the identifiers verbatim, so the
        // written report of a repeated run carries the same bytes.
        let first = serde_json::to_string(&report).expect("the report serializes");
        let second = serde_json::to_string(&report).expect("the report serializes");
        prop_assert_eq!(&first, &second);
        for entry in &report.degradation.entries {
            prop_assert!(
                first.contains(&format!("\"reason\":\"{}\"", entry.reason)),
                "the serialized report must carry `{}` verbatim",
                entry.reason
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 80: the complete
    // arbitrary reason multiset is retained by Stack_Report, any rejected
    // severity rejects the ledger result, cancellation remains a distinct
    // terminal decision, and the production publication boundary follows it.
    //
    // **Validates: Requirements 12.9**
    #[test]
    fn property_80_rejection_outranks_degradation_and_controls_publication(
        reason_indices in prop::collection::vec(0usize..FAILURE_REASONS.len(), 0..96),
        nonce in any::<u64>(),
    ) {
        use super::degradation::{
            DegradationManager, OutcomeDecision, OutputPublication, RunResult, Severity,
            StandardFinalOutputFileSystem,
        };
        use super::report::{StackReportRecorder, StackRunResult};

        let mut ledger = DegradationLedger::new();
        for (ordinal, &reason_index) in reason_indices.iter().enumerate() {
            let (reason, _) = FAILURE_REASONS[reason_index];
            ledger.record(reason, serde_json::json!({ "ordinal": ordinal, "nonce": nonce }));
        }

        let has_rejected = ledger
            .entries()
            .iter()
            .any(|entry| entry.severity == Severity::Rejected);
        let has_degraded = ledger
            .entries()
            .iter()
            .any(|entry| entry.severity == Severity::Degraded);
        let has_cancellation = ledger.count_of(degradation::RUN_CANCELLED_BY_USER) > 0;
        prop_assert_eq!(ledger.result() == RunResult::Rejected, has_rejected);

        let manager = DegradationManager::new(&ledger);
        let expected_decision = if has_cancellation {
            OutcomeDecision::Cancelled
        } else if has_rejected {
            OutcomeDecision::Rejected
        } else if has_degraded {
            OutcomeDecision::Degraded
        } else {
            OutcomeDecision::Success
        };
        prop_assert_eq!(manager.decision(), expected_decision);
        prop_assert_eq!(manager.decision().permits_final_output(), !has_rejected);

        let recorder = StackReportRecorder::isolated("property-80", None);
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();
        prop_assert_eq!(report.degradation.entries.len(), ledger.entries().len());
        for (recorded, original) in report.degradation.entries.iter().zip(ledger.entries()) {
            prop_assert_eq!(recorded.reason.as_str(), original.reason);
            prop_assert_eq!(recorded.severity.as_str(), original.severity.as_identifier());
            prop_assert_eq!(&recorded.detail, &original.detail);
        }
        let expected_report_result = match expected_decision {
            OutcomeDecision::Success => StackRunResult::Success,
            OutcomeDecision::Degraded => StackRunResult::Degraded,
            OutcomeDecision::Rejected => StackRunResult::Rejected,
            OutcomeDecision::Cancelled => StackRunResult::Cancelled,
        };
        prop_assert_eq!(report.degradation.result, expected_report_result);

        let directory = TempDir::new()
            .map_err(|error| TestCaseError::fail(format!("publication tempdir failed: {error}")))?;
        let staged = directory.path().join("result.tmp");
        let final_path = directory.path().join("result.tiff");
        let payload = nonce.to_le_bytes();
        fs::write(&staged, payload)
            .map_err(|error| TestCaseError::fail(format!("staged output write failed: {error}")))?;
        let publication = manager
            .publish_staged_output(
                &mut StandardFinalOutputFileSystem,
                &staged,
                &final_path,
            )
            .map_err(TestCaseError::fail)?;
        let expected_publication = match expected_decision {
            OutcomeDecision::Success | OutcomeDecision::Degraded => {
                OutputPublication::Published(expected_decision)
            }
            OutcomeDecision::Rejected => OutputPublication::Rejected,
            OutcomeDecision::Cancelled => OutputPublication::Cancelled,
        };
        prop_assert_eq!(publication, expected_publication);
        prop_assert_eq!(matches!(publication, OutputPublication::Published(_)), !has_rejected);
        prop_assert!(!staged.exists());
        if has_rejected {
            prop_assert!(!final_path.exists());
        } else {
            prop_assert_eq!(fs::read(&final_path).map_err(|error| TestCaseError::fail(error.to_string()))?, payload);
        }
    }

    // Feature: layered-camera-group-focus-stitching, Property 92: one
    // production function decides both execution and Stack_Report from the
    // station-count boundary and every persisted compositor setting.
    //
    // **Validates: Requirements 15.1, 15.11**
    #[test]
    fn property_92_station_count_and_setting_select_and_record_exact_path(
        station_count in 0usize..128,
    ) {
        use super::compositor::{
            StackCompositorChoice, select_and_record_run_path,
        };
        use super::report::{SelectedPath, StackReportRecorder};

        let settings = [
            StackCompositorChoice::LayeredVirtualTile,
            StackCompositorChoice::ProgressiveSeamTile,
            StackCompositorChoice::StreamingMosaic,
            StackCompositorChoice::LegacySingleLayerMosaic,
        ];
        let counts = [0, 1, 2, station_count.max(2)];
        for &count in &counts {
            for &setting in &settings {
                // Independent requirement oracle: the explicit legacy
                // diagnostic setting wins; otherwise every count below two is
                // station-layer-only and every count from two uses its chosen
                // Virtual_Tile compositor.
                let expected = if setting == StackCompositorChoice::LegacySingleLayerMosaic {
                    SelectedPath::LegacySingleLayerMosaic
                } else if count < 2 {
                    SelectedPath::SingleStation
                } else {
                    match setting {
                        StackCompositorChoice::LayeredVirtualTile => SelectedPath::LayeredVirtualTile,
                        StackCompositorChoice::ProgressiveSeamTile => SelectedPath::ProgressiveSeamTile,
                        StackCompositorChoice::StreamingMosaic => SelectedPath::StreamingMosaic,
                        StackCompositorChoice::LegacySingleLayerMosaic => unreachable!(),
                    }
                };
                let expected_virtual_tiles = count >= 2
                    && setting != StackCompositorChoice::LegacySingleLayerMosaic;
                let recorder = StackReportRecorder::isolated("property-92", None);
                let selection = select_and_record_run_path(
                    Some(count),
                    setting,
                    Some(&recorder),
                );

                prop_assert_eq!(selection.selected_path, expected);
                prop_assert_eq!(selection.use_virtual_tiles, expected_virtual_tiles);
                prop_assert_eq!(recorder.snapshot().selected_path, expected);
                prop_assert_eq!(
                    recorder.snapshot().selected_path.as_identifier(),
                    expected.as_identifier(),
                );
            }
        }
    }
}
