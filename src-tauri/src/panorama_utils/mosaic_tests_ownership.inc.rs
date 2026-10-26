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
        let decision =
            GrayImage::from_fn(2, 2, |x, y| Luma([if (x + y) % 2 == 0 { 0 } else { 255 }]));
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
    fn low_frequency_harmonization_softens_group_exposure_steps_without_blurring_detail() {
        let store = StreamingMosaicStore::new(256, 64).expect("tile store should initialize");
        let rgb = Rgb32FImage::from_fn(256, 64, |x, y| {
            let base = if x < 128 { 0.42 } else { 0.58 };
            let detail = if (x / 4 + y / 4) % 2 == 0 {
                0.03
            } else {
                -0.03
            };
            Rgb([base + detail, base + detail, base + detail])
        });
        let mask = GrayImage::from_pixel(256, 64, Luma([255]));
        let owner = GrayImage::from_fn(256, 64, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        store
            .save_owner_tile(0, 0, &owner)
            .expect("owner should be writable");
        let (gains, scale) = streaming_seam_harmonization(&store, (0, 0, 256, 64), &[])
            .expect("harmonization should build")
            .expect("two owners should create a seam field");
        let mut output = rgb.clone();
        apply_streaming_seam_harmonization(&mut output, &gains, scale);
        let band_mean = |image: &Rgb32FImage, start: u32| {
            (start..start + 8)
                .map(|x| luma(*image.get_pixel(x, 32)))
                .sum::<f64>()
                / 8.0
        };
        let before_step = (band_mean(&rgb, 120) - band_mean(&rgb, 128)).abs();
        let after_step = (band_mean(&output, 120) - band_mean(&output, 128)).abs();
        assert!(
            after_step < before_step * 0.72,
            "the exposure seam should soften: {before_step:.4} -> {after_step:.4}"
        );
        let before_detail = (luma(*rgb.get_pixel(20, 20)) - luma(*rgb.get_pixel(24, 20))).abs();
        let after_detail =
            (luma(*output.get_pixel(20, 20)) - luma(*output.get_pixel(24, 20))).abs();
        assert!(
            after_detail >= before_detail * 0.90,
            "multiplicative low-frequency correction must retain local detail"
        );
    }

    #[test]
    fn frequency_separation_removes_station_blocks_but_keeps_detail_and_black_bounds() {
        let mut image = Rgb32FImage::from_fn(512, 128, |x, y| {
            if y < 8 {
                return Rgb([0.0; 3]);
            }
            let base = if x < 256 { 0.36 } else { 0.54 };
            let detail = if (x / 3 + y / 3) % 2 == 0 {
                0.035
            } else {
                -0.035
            };
            Rgb([base + detail, base + detail, base + detail])
        });
        let original = image.clone();
        smooth_streaming_low_frequency_illumination(&mut image);
        let mean = |source: &Rgb32FImage, start: u32| {
            (start..start + 12)
                .map(|x| luma(*source.get_pixel(x, 64)))
                .sum::<f64>()
                / 12.0
        };
        let before = (mean(&original, 238) - mean(&original, 262)).abs();
        let after = (mean(&image, 238) - mean(&image, 262)).abs();
        assert!(
            after < before * 0.82,
            "station step {before:.4} -> {after:.4}"
        );
        let before_detail =
            (luma(*original.get_pixel(80, 64)) - luma(*original.get_pixel(83, 64))).abs();
        let after_detail = (luma(*image.get_pixel(80, 64)) - luma(*image.get_pixel(83, 64))).abs();
        assert!(after_detail >= before_detail * 0.88);
        assert!((0..512).all(|x| image.get_pixel(x, 2).0 == [0.0; 3]));
    }

    #[test]
    fn same_coordinate_group_relations_recover_exposure_offset() {
        let (width, height) = (320, 160);
        let base = Rgb32FImage::from_fn(width, height, |x, y| {
            let detail = if (x / 5 + y / 5) % 2 == 0 {
                0.025
            } else {
                -0.025
            };
            Rgb([0.42 + detail, 0.39 + detail, 0.34 + detail])
        });
        let expected = [0.12f64, 0.08, -0.03];
        let candidate = Rgb32FImage::from_fn(width, height, |x, y| {
            let source = base.get_pixel(x, y);
            Rgb(std::array::from_fn(|channel| {
                source[channel] * expected[channel].exp() as f32
            }))
        });
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        let owner = GrayImage::from_pixel(width, height, Luma([1]));
        let relations = streaming_group_tone_relations_from_analysis(
            &base, &mask, &owner, &candidate, &mask, 2,
        );
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].0, (1, 2));
        for (actual, expected) in relations[0].1.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 0.01, "{actual} vs {expected}");
        }
    }

    #[test]
    #[ignore = "applies production low-frequency correction to a supplied real preview"]
    fn real_preview_low_frequency_correction_from_env() {
        let input = std::env::var_os("RAW_EDITOR_MOSAIC_PREVIEW_INPUT")
            .expect("RAW_EDITOR_MOSAIC_PREVIEW_INPUT is required");
        let output = std::env::var_os("RAW_EDITOR_MOSAIC_PREVIEW_OUTPUT")
            .expect("RAW_EDITOR_MOSAIC_PREVIEW_OUTPUT is required");
        let mut image = image::open(input)
            .expect("real preview should open")
            .to_rgb32f();
        smooth_streaming_low_frequency_illumination(&mut image);
        image::DynamicImage::ImageRgb32F(image)
            .to_rgb8()
            .save(output)
            .expect("corrected preview should save");
    }

    #[test]
    fn group_exposure_harmonization_cannot_draw_a_halo_around_dark_strokes() {
        let store = StreamingMosaicStore::new(256, 64).expect("tile store should initialize");
        let rgb = Rgb32FImage::from_fn(256, 64, |x, _| {
            let paper = if x < 128 { 0.46 } else { 0.56 };
            let value = if (58..=62).contains(&x) { 0.06 } else { paper };
            Rgb([value, value * 0.96, value * 0.90])
        });
        let mask = GrayImage::from_pixel(256, 64, Luma([255]));
        let owner = GrayImage::from_fn(256, 64, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        store
            .save_owner_tile(0, 0, &owner)
            .expect("owner should be writable");
        let (gains, _) = streaming_seam_harmonization(&store, (0, 0, 256, 64), &[])
            .expect("harmonization should build")
            .expect("two owners should create a gain field");
        let paper_gain = gains.get_pixel(30, 32)[0];
        for x in [52, 56, 58, 60, 62, 64, 68] {
            assert!(
                (gains.get_pixel(x, 32)[0] - paper_gain).abs() < 1e-5,
                "a dark stroke must not alter the exposure gain around itself"
            );
        }
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

    #[test]
    fn focus_group_ownership_cannot_reclaim_another_camera_position() {
        let sharp = sharp_fixture();
        let base = image::imageops::blur(&sharp, 2.0);
        let mask = GrayImage::from_pixel(sharp.width(), sharp.height(), Luma([255]));
        let owner = GrayImage::from_fn(sharp.width(), sharp.height(), |x, _| {
            Luma([if x < sharp.width() / 2 { 3 } else { 4 }])
        });
        let decision = streaming_ownership_from_analysis(
            &base,
            &mask,
            &sharp,
            &mask,
            &Field::new(sharp.width(), sharp.height(), 32.0),
            &owner,
            3,
            false,
        );
        let left_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x < decision.width() / 2 && pixel[0] != 0)
            .count();
        let right_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x >= decision.width() / 2 && pixel[0] != 0)
            .count();
        assert!(
            left_selected > 0,
            "the sharper member should contribute inside its group"
        );
        assert_eq!(
            right_selected, 0,
            "a focal member must not overwrite a different camera position"
        );
    }

    #[test]
    fn panorama_group_transition_only_enters_from_new_coverage() {
        let candidate = sharp_fixture();
        let base = image::imageops::blur(&candidate, 1.0);
        let mut base_mask = GrayImage::new(candidate.width(), candidate.height());
        let mut owner = GrayImage::new(candidate.width(), candidate.height());
        for y in 0..candidate.height() {
            for x in 0..candidate.width() * 3 / 4 {
                base_mask.put_pixel(x, y, Luma([255]));
                owner.put_pixel(x, y, Luma([2]));
            }
        }
        let candidate_mask =
            GrayImage::from_pixel(candidate.width(), candidate.height(), Luma([255]));
        let decision = streaming_ownership_from_analysis(
            &base,
            &base_mask,
            &candidate,
            &candidate_mask,
            &Field::new(candidate.width(), candidate.height(), 32.0),
            &owner,
            3,
            true,
        );
        let left_quarter_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x < decision.width() / 4 && pixel[0] != 0)
            .count();
        let overlap_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| {
                *x >= decision.width() / 2 && *x < decision.width() * 3 / 4 && pixel[0] != 0
            })
            .count();
        let new_coverage_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x >= decision.width() * 3 / 4 && pixel[0] != 0)
            .count();
        assert_eq!(
            left_quarter_selected, 0,
            "a new camera group must not form a detached sharpness island"
        );
        assert!(
            new_coverage_selected > 0,
            "new source coverage must be admitted"
        );
        assert!(
            overlap_selected > 0,
            "a panorama seam must be allowed to leave the rectangular new-coverage boundary"
        );
    }

    #[test]
    fn panorama_transition_marks_ink_cells_as_expensive_seam_routes() {
        let image = Rgb32FImage::from_fn(96, 64, |x, y| {
            let ink = (38..58).contains(&x) && (18..46).contains(&y);
            let value = if ink { 0.05 } else { 0.52 };
            Rgb([value, value * 0.96, value * 0.90])
        });
        let quiet = streaming_transition_structure(|x, y| rgb_at(&image, x, y), 0.0, 0.0, 24.0);
        let ink = streaming_transition_structure(|x, y| rgb_at(&image, x, y), 30.0, 12.0, 36.0);
        assert!(quiet < 0.05, "plain paper should remain a cheap seam route");
        assert!(
            ink > 0.75,
            "a cell crossed by dark calligraphy must strongly repel the panorama seam"
        );
    }

    fn image_info(id: usize, image: &Rgb32FImage) -> ImageInfo {
        ImageInfo {
            id,
            filename: format!("source-{id}.png"),
            width: image.width(),
            height: image.height(),
            alignment_image: image::DynamicImage::ImageRgb32F(image.clone()).to_luma8(),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }

    fn sharp_fixture() -> Rgb32FImage {
        Rgb32FImage::from_fn(160, 120, |x, y| {
            let value = 0.36
                + 0.12 * (x as f32 * 0.75).sin()
                + 0.10 * (y as f32 * 0.65).cos()
                + 0.03 * ((x + y) as f32 * 0.19).sin();
            Rgb([value * 1.1, value, value * 0.8])
        })
    }

    fn noisy_colour_stroke_pair() -> (Rgb32FImage, Rgb32FImage) {
        let sharp = Rgb32FImage::from_fn(192, 160, |x, y| {
            let stroke = (x as i32 - 96).abs() <= 3 && (20..140).contains(&y);
            // A faded red mark can have almost the same luminance as its
            // brown substrate; colour edges still carry genuine focus.
            if stroke {
                Rgb([0.46, 0.31, 0.27])
            } else {
                Rgb([0.40, 0.34, 0.27])
            }
        });
        let blurred = image::imageops::blur(&sharp, 3.2);
        let noisy = |im: &Rgb32FImage, amplitude: f32| {
            Rgb32FImage::from_fn(im.width(), im.height(), |x, y| {
                let hash = x.wrapping_mul(0x9e3779b9) ^ y.wrapping_mul(0x85ebca6b);
                let hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
                let noise = ((hash & 65535) as f32 / 65535.0 - 0.5) * amplitude;
                Rgb(im.get_pixel(x, y).0.map(|v| (v + noise).clamp(0.0, 1.0)))
            })
        };
        (noisy(&sharp, 0.020), noisy(&blurred, 0.080))
    }
