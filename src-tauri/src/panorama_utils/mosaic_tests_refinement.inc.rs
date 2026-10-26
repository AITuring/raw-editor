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
            assert!(
                STREAMING_ANALYSIS_LONG_SIDE == intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE
            );
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
    fn sensor_noise_cannot_hide_a_sharper_faint_colour_stroke() {
        let (sharp, noisy_blur) = noisy_colour_stroke_pair();
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let mask = GrayImage::from_pixel(192, 160, Luma([255]));
        let chosen = ownership(
            &noisy_blur,
            &mask,
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert!(
            selected >= 90,
            "faint focused colour must beat noisy defocus ({selected}/100)"
        );
    }

    #[test]
    fn extra_sensor_noise_cannot_reclaim_an_already_focused_stroke() {
        let (sharp, noisy_blur) = noisy_colour_stroke_pair();
        let info = image_info(1, &noisy_blur);
        let sampler = LayerSampler {
            info: &info,
            source: &noisy_blur,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let mask = GrayImage::from_pixel(192, 160, Luma([255]));
        let chosen = ownership(
            &sharp,
            &mask,
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert_eq!(
            selected, 0,
            "noisy defocus must not overwrite the focused stroke"
        );
    }

    #[test]
    fn diffuse_halo_cannot_outvote_a_narrow_focused_contour() {
        let sharp = Rgb32FImage::from_fn(192, 160, |x, y| {
            let line = (x as i32 - 96).abs() <= 1 && (20..140).contains(&y);
            let paper = 0.40 + 0.003 * (x as f32 * 0.17).sin();
            let value = if line { 0.28 } else { paper };
            Rgb([value, value * 0.9, value * 0.8])
        });
        let base = image::imageops::blur(&sharp, 5.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let chosen = ownership(
            &base,
            &GrayImage::from_pixel(192, 160, Luma([255])),
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert!(
            selected >= 90,
            "focused contour lost to its spread halo: {selected}/100"
        );
    }

    #[test]
    fn enclosed_sharp_tile_replaces_blur_without_losing_coverage() {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        // Four times the old 160x120 fixture, with the candidate a 320x320 crop
        // instead of an 80x80 one.  On the small canvas the native control
        // points had to sit at least one 57x57 patch away from the layer edge,
        // which left 2 measurable cells out of 18 — an 11% inlier spatial
        // support that is an artefact of an 80px layer, not of the photographs,
        // and that 需求 2.8 / 12.2 correctly refuse to fuse.  A 320px layer
        // measures 0.4 of its cells, so the test exercises the selection it is
        // about; the exclusion path has its own unit tests in
        // `stack_pipeline::station_degradation`.
        let sharp = Rgb32FImage::from_fn(640, 480, |x, y| {
            let value = 0.36
                + 0.12 * (x as f32 * 0.75).sin()
                + 0.10 * (y as f32 * 0.65).cos()
                + 0.03 * ((x + y) as f32 * 0.19).sin();
            Rgb([value * 1.1, value, value * 0.8])
        });
        let base = image::imageops::blur(&sharp, 2.0);
        let candidate = image::imageops::crop_imm(&sharp, 160, 80, 320, 320).to_image();
        let sources = [base.clone(), candidate];
        let infos = [image_info(0, &sources[0]), image_info(1, &sources[1])];
        let transforms = HashMap::from([
            (0, Matrix3::identity()),
            (
                1,
                Matrix3::new(1.0, 0.0, 160.0, 0.0, 1.0, 80.0, 0.0, 0.0, 1.0),
            ),
        ]);
        let app = tauri::test::mock_app();
        let output = detail_preserving_mosaic(
            &[&infos[0], &infos[1]],
            &transforms,
            Projection::Planar,
            None,
            false,
            app.handle().clone(),
            "test-progress",
            &mut |info| Ok(sources[info.id].clone()),
        )
        .unwrap();
        assert_eq!(
            output.dimensions(),
            sharp.dimensions(),
            "coverage must include the last row and column"
        );
        let mut before = 0.0;
        let mut after = 0.0;
        for y in 120..360 {
            for x in 200..440 {
                before += (base.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
                after += (output.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
            }
        }
        assert!(
            after < before * 0.05,
            "the sharp interior must replace the blur: {after} vs {before}"
        );
        assert!(
            output
                .pixels()
                .all(|p| p.0.iter().all(|v| v.is_finite() && *v > 0.0))
        );
    }

    #[test]
    fn blurred_candidate_cannot_replace_an_already_sharp_interior() {
        let base = sharp_fixture();
        let source = image::imageops::blur(&base, 2.0);
        let info = image_info(0, &source);
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
            residual: Field::new(160, 120, 48.0),
        };
        let mask = GrayImage::from_pixel(160, 120, Luma([255]));
        let selected = ownership(
            &base,
            &mask,
            &sampler,
            &Field::new(160, 120, 56.0),
            160,
            120,
        );
        let switched = (10..110)
            .flat_map(|y| (10..150).map(move |x| (x, y)))
            .filter(|&(x, y)| selected.get_pixel(x, y)[0] != 0)
            .count();
        assert_eq!(switched, 0);
    }

    #[test]
    fn sparse_in_focus_stroke_is_not_lost_to_cell_background() {
        let sharp = Rgb32FImage::from_fn(160, 120, |x, y| {
            let line = (x as i32 - 80).unsigned_abs() <= 1 && (12..108).contains(&y);
            let value = if line { 0.04 } else { 0.42 };
            Rgb([value, value, value])
        });
        let base = image::imageops::blur(&sharp, 3.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let mask = GrayImage::from_pixel(160, 120, Luma([255]));
        let selected = ownership(
            &base,
            &mask,
            &sampler,
            &Field::new(160, 120, 56.0),
            160,
            120,
        );
        let selected_on_stroke = (12..108)
            .filter(|&y| selected.get_pixel(80, y)[0] != 0)
            .count();
        assert!(
            selected_on_stroke > 70,
            "a sparse sharp stroke must own its pixels ({selected_on_stroke}/96)"
        );
    }

    #[test]
    fn streaming_focus_probe_keeps_a_sparse_sharp_stroke() {
        let sharp = Rgb32FImage::from_fn(160, 128, |x, y| {
            let line = (x as i32 - 80).unsigned_abs() <= 1 && (8..120).contains(&y);
            let value = if line { 0.04 } else { 0.42 };
            Rgb([value, value, value])
        });
        let base = image::imageops::blur(&sharp, 3.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 128, 48.0),
        };
        let selected = streaming_ownership_with_long_side(
            &base,
            &GrayImage::from_pixel(160, 128, Luma([255])),
            &sampler,
            &Field::new(160, 128, 56.0),
            160,
            128,
            5,
        );
        assert!(
            selected.get_pixel(2, 1)[0] != 0 || selected.get_pixel(2, 2)[0] != 0,
            "the bounded streaming probe must retain sparse focused writing"
        );
    }

    #[test]
    fn disagreement_is_lower_for_aligned_defocus_than_for_shifted_detail() {
        let sharp = Rgb32FImage::from_fn(160, 120, |x, y| {
            let value = 0.35 + 0.22 * (x as f32 * 0.37).sin() + 0.16 * (y as f32 * 0.29).cos();
            Rgb([value, value * 0.9, value * 0.75])
        });
        let base = image::imageops::blur(&sharp, 2.5);
        let info = image_info(1, &sharp);
        let tone = Field::new(160, 120, 56.0);
        let aligned = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let aligned_error = ownership_disagreement(&base, &aligned, &tone, 32.0, 32.0, 40.0);
        let shifted = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::new(1.0, 0.0, 4.0, 0.0, 1.0, -3.0, 0.0, 0.0, 1.0),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let shifted_error = ownership_disagreement(&base, &shifted, &tone, 32.0, 32.0, 40.0);
        assert!(
            aligned_error < shifted_error,
            "local averaging should preserve geometric mismatch signal: {aligned_error} vs {shifted_error}"
        );
    }

    #[test]
    fn invalid_sample_gap_cannot_cancel_disconnected_colour_mismatches() {
        let base = Rgb32FImage::from_pixel(100, 100, Rgb([0.5; 3]));
        let source = Rgb32FImage::from_fn(100, 100, |x, _| {
            // Invalid samples must retain their positions. If the valid
            // samples are packed into a smaller grid, these separated light
            // and dark strips become neighbours and falsely cancel out.
            let value = if x < 30 {
                0.8
            } else if x > 60 {
                0.2
            } else {
                f32::NAN
            };
            Rgb([value; 3])
        });
        let info = image_info(1, &source);
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
            residual: Field::new(100, 100, 48.0),
        };
        let error =
            ownership_disagreement(&base, &sampler, &Field::new(100, 100, 56.0), 0.0, 0.0, 80.0);
        assert!(
            (error - 0.3).abs() < 1e-6,
            "missing samples must not hide a real colour mismatch: {error}"
        );
    }

    #[test]
    fn fully_invalid_overlap_is_not_treated_as_an_aligned_match() {
        let source = Rgb32FImage::from_pixel(32, 32, Rgb([0.5; 3]));
        let info = image_info(1, &source);
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::new(1.0, 0.0, 64.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(32, 32, 48.0),
        };
        assert_eq!(
            ownership_disagreement(&source, &sampler, &Field::new(32, 32, 56.0), 0.0, 0.0, 24.0,),
            1.0,
            "absence of common samples must not remove the mismatch penalty"
        );
    }

    #[test]
    fn scaled_sparse_detail_is_sampled_at_the_native_cell_centre() {
        let sharp = Rgb32FImage::from_fn(3_200, 1_600, |x, _| {
            let value = if (x as i32 - 84).unsigned_abs() <= 2 {
                0.04
            } else {
                0.42
            };
            Rgb([value; 3])
        });
        let base = image::imageops::blur(&sharp, 4.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 8.0,
            residual: Field::new(400, 200, 48.0),
        };
        let selected = ownership(
            &base,
            &GrayImage::from_pixel(3_200, 1_600, Luma([255])),
            &sampler,
            &Field::new(400, 200, 56.0),
            400,
            200,
        );
        let selected_on_stroke = (20..180)
            .filter(|&y| selected.get_pixel(10, y)[0] != 0)
            .count();
        assert!(
            selected_on_stroke > 100,
            "a native sharp stroke at a scaled cell centre must own its detail ({selected_on_stroke}/160)"
        );
    }

    #[test]
    fn smooth_tone_correction_preserves_selected_detail_relative_contrast() {
        let p = Rgb([0.2, 0.3, 0.4]);
        let q = Rgb([0.4, 0.5, 0.6]);
        let a = adjusted(p, [0.1, -0.05, 0.02]);
        let b = adjusted(q, [0.1, -0.05, 0.02]);
        for c in 0..3 {
            assert!((b[c] / a[c] - q[c] / p[c]).abs() < 1e-6);
        }
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
