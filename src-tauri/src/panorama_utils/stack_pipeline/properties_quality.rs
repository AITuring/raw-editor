// Stage 7 Quality_Gate self-checks (15.14, 15.16, 15.17).
use super::quality_gate::{noise_sigma, roi_low_frequency_delta_e00, slanted_edge_mtf50};

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 66: MTF50_Normalized 不低于参考的 0.93 倍
    #[test]
    fn property_66_gaussian_slanted_edge_mtf50_self_check(
        sigma in 0.65f32..1.85f32,
    ) {
        let width = 256u32;
        let height = 256u32;
        let angle = 7.0f64.to_radians();
        let edge = image::Rgb32FImage::from_fn(width, height, |x, y| {
            let distance = f64::from(x) - f64::from(y) * angle.tan() - 116.0;
            let value = if distance >= 0.0 { 0.86 } else { 0.12 };
            image::Rgb([value; 3])
        });
        let blurred = image::imageops::blur(&edge, sigma);
        let measurement = slanted_edge_mtf50(&blurred, 1.0)
            .expect("synthetic slanted edge must be measurable");
        let analytic = (2.0f64.ln()).sqrt()
            / (std::f64::consts::PI * 2.0f64.sqrt() * f64::from(sigma));
        prop_assert!(measurement.f50_cycles_per_output_pixel.is_finite());
        prop_assert!(measurement.normalized.is_finite());
        // The finite 256px ROI, 4-point smoothing, Hamming window and the
        // repository's sRGB Gaussian kernel introduce a bounded discrete bias.
        prop_assert!((measurement.f50_cycles_per_output_pixel - analytic).abs()
            <= 0.45 * analytic.max(0.01));
        prop_assert!((measurement.normalized - measurement.f50_cycles_per_output_pixel).abs()
            <= 1.0e-12);
    }

    // Feature: layered-camera-group-focus-stitching, Property 68: Noise_Sigma 比值落在规定范围
    #[test]
    fn property_68_noise_sigma_mad_self_check(
        sigma in 0.002f64..0.030f64,
        seed in any::<u64>(),
    ) {
        let width = 96u32;
        let height = 96u32;
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        // The sRGB EOTF slope at 0.5 is approximately 0.93.
        let srgb_sigma = sigma / 0.93;
        let image = image::Rgb32FImage::from_fn(width, height, |_x, _y| {
            let value = (0.5 + srgb_sigma * next()).clamp(0.0, 1.0) as f32;
            image::Rgb([value; 3])
        });
        let estimated = noise_sigma(&image).expect("finite synthetic noise");
        prop_assert!(estimated.is_finite());
        prop_assert!(estimated / sigma >= 0.55 && estimated / sigma <= 1.45);
    }

    // Feature: layered-camera-group-focus-stitching, Property 69: ROI 低频均值色差不超过 2.0
    #[test]
    fn property_69_ciede2000_is_symmetric_and_zero_for_identity(
        red_a in 0.0f32..1.0,
        green_a in 0.0f32..1.0,
        blue_a in 0.0f32..1.0,
        red_b in 0.0f32..1.0,
        green_b in 0.0f32..1.0,
        blue_b in 0.0f32..1.0,
    ) {
        let left = image::Rgb32FImage::from_pixel(8, 8, image::Rgb([red_a, green_a, blue_a]));
        let right = image::Rgb32FImage::from_pixel(8, 8, image::Rgb([red_b, green_b, blue_b]));
        let forward = roi_low_frequency_delta_e00(&left, &right).expect("finite RGB pair");
        let reverse = roi_low_frequency_delta_e00(&right, &left).expect("finite RGB pair");
        prop_assert!(forward.is_finite());
        prop_assert!((forward - reverse).abs() <= 1.0e-10);
        let identity = roi_low_frequency_delta_e00(&left, &left).expect("identity RGB pair");
        prop_assert!(identity.abs() <= 1.0e-10);
    }
}

// Optional Stage 7 geometry properties (15.9, 15.11).
use super::quality_gate::{
    QUALITY_EFFECTIVE_PIXEL_RATIO_MIN, QUALITY_ROI_MARGIN, QUALITY_ROI_MAX_COUNT, QUALITY_ROI_SIDE,
    QualityRoi, SourceGeometry, effective_pixel_coverage, label_owner_regions, local_scale_for_roi,
    map_roi_corners_to_source, select_quality_rois,
};

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 61: ROI 选取满足全部几何条件
    #[test]
    fn property_61_roi_selection_is_region_complete(split in 544u32..1024u32) {
        let width = 1024u32;
        let height = 512u32;
        let owners = (0..width * height).map(|index| {
            let x = index % width;
            if x < split { 1u16 } else { 2u16 }
        }).collect::<Vec<_>>();
        let coverage = image::GrayImage::from_pixel(width, height, image::Luma([255]));
        let regions = label_owner_regions(&owners, &coverage)
            .expect("synthetic ownership must label");
        let selection = select_quality_rois(&regions, (0.0, 0.0))
            .expect("at least one 512px ROI is measurable");
        prop_assert!(!selection.rois.is_empty());
        prop_assert!(selection.rois.len() <= QUALITY_ROI_MAX_COUNT);
        for roi in selection.rois {
            prop_assert_eq!(roi.side, QUALITY_ROI_SIDE);
            let first = regions.labels[(roi.y * regions.width + roi.x) as usize];
            for y in roi.y..roi.y + roi.side {
                for x in roi.x..roi.x + roi.side {
                    let index = (y * regions.width + x) as usize;
                    prop_assert_eq!(regions.labels[index], first);
                    prop_assert!(regions.distance_to_boundary[index] >= QUALITY_ROI_MARGIN);
                }
            }
        }
    }

    // Feature: layered-camera-group-focus-stitching, Property 63: Local_Scale 由雅可比确定且满足下界
    #[test]
    fn property_63_identity_jacobian_has_unit_local_scale(
        tx in -4096.0f64..4096.0,
        ty in -4096.0f64..4096.0,
        render_scale in 0.25f64..1.5,
    ) {
        let roi = QualityRoi {
            x: 0, y: 0, side: QUALITY_ROI_SIDE, region_label: 1, owner: 1,
            world_origin: (512.0 + tx, 512.0 + ty), boundary_clearance: 64.0,
        };
        let geometry = SourceGeometry {
            member_to_anchor: nalgebra::Matrix3::identity(),
            tile_to_world: nalgebra::Matrix3::identity(),
            station_id: 1,
        };
        let residual = super::residual_warp::ResidualWarp::new();
        let measurement = local_scale_for_roi(&roi, &geometry, &residual, render_scale)
            .expect("identity map is measurable");
        prop_assert_eq!(measurement.samples, 1024);
        prop_assert!((measurement.median - 1.0).abs() < 1.0e-10);
        prop_assert!((measurement.fraction_at_least_095 - 1.0).abs() < 1.0e-10);
        prop_assert_eq!(measurement.diagnostic_render_scale, render_scale < 1.0);
    }
}

// Stage 7 pairing and effective-coverage properties (15.10, 15.12).
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 62: 配对不重采样输出 ROI
    // 输出 ROI 只作为坐标和像素网格的观测窗口传递；配对过程仅把四角映射回
    // owner Source_RAW，并在残余对齐误差超限时拒绝该 ROI。
    // **Validates: Requirements 11.2**
    #[test]
    fn property_62_pairing_keeps_output_roi_and_limits_reference_resampling(
        world_x in -4096.0f64..4096.0,
        world_y in -4096.0f64..4096.0,
        scale_x in 0.5f64..2.0,
        scale_y in 0.5f64..2.0,
        translate_x in -512.0f64..512.0,
        translate_y in -512.0f64..512.0,
        residual_alignment_px in 0.0f64..1.0,
    ) {
        let roi = QualityRoi {
            x: 256,
            y: 512,
            side: QUALITY_ROI_SIDE,
            region_label: 7,
            owner: 11,
            world_origin: (world_x, world_y),
            boundary_clearance: 64.0,
        };
        let before = roi.clone();
        let mut tile_to_world = Matrix3::identity();
        tile_to_world[(0, 0)] = scale_x;
        tile_to_world[(1, 1)] = scale_y;
        tile_to_world[(0, 2)] = translate_x;
        tile_to_world[(1, 2)] = translate_y;
        let geometry = SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world,
            station_id: 11,
        };
        let residual = super::residual_warp::ResidualWarp::new();
        let paired = map_roi_corners_to_source(
            &roi,
            &geometry,
            &residual,
            residual_alignment_px,
        );

        // Pairing does not mutate, crop, or resample the output ROI.  The only
        // allowed reference operation is this one inverse-coordinate mapping.
        prop_assert_eq!(&roi, &before);
        if residual_alignment_px > 0.5 {
            prop_assert_eq!(paired, Err(super::degradation::PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED));
        } else {
            let corners = paired.expect("sub-pixel residual alignment is measurable");
            let expected_x = (world_x - translate_x) / scale_x;
            let expected_y = (world_y - translate_y) / scale_y;
            prop_assert!((corners[0].x - expected_x).abs() < 1.0e-8);
            prop_assert!((corners[0].y - expected_y).abs() < 1.0e-8);
            let expected_last_x = (world_x + f64::from(QUALITY_ROI_SIDE - 1) - translate_x) / scale_x;
            let expected_last_y = (world_y + f64::from(QUALITY_ROI_SIDE - 1) - translate_y) / scale_y;
            prop_assert!((corners[2].x - expected_last_x).abs() < 1.0e-8);
            prop_assert!((corners[2].y - expected_last_y).abs() < 1.0e-8);
        }
    }

    // Feature: layered-camera-group-focus-stitching, Property 64: 有效像素数不低于唯一覆盖面积基准
    // 对整数边界的源图四边形，扫描线填充的并集只计重叠一次；完整输出覆盖
    // 因此满足 0.98×唯一投影面积的有效像素下界。
    // **Validates: Requirements 11.4, 15.7**
    #[test]
    fn property_64_effective_pixels_cover_unique_projected_union(
        ax in 0u32..64,
        ay in 0u32..64,
        aw in 32u32..96,
        ah in 32u32..96,
        bx in 0u32..64,
        by in 0u32..64,
        bw in 32u32..96,
        bh in 32u32..96,
    ) {
        const WIDTH: u32 = 192;
        const HEIGHT: u32 = 192;
        let a_right = ax + aw;
        let a_bottom = ay + ah;
        let b_right = bx + bw;
        let b_bottom = by + bh;
        let quad_a = [
            Point2::new(f64::from(ax), f64::from(ay)),
            Point2::new(f64::from(a_right), f64::from(ay)),
            Point2::new(f64::from(a_right), f64::from(a_bottom)),
            Point2::new(f64::from(ax), f64::from(a_bottom)),
        ];
        let quad_b = [
            Point2::new(f64::from(bx), f64::from(by)),
            Point2::new(f64::from(b_right), f64::from(by)),
            Point2::new(f64::from(b_right), f64::from(b_bottom)),
            Point2::new(f64::from(bx), f64::from(b_bottom)),
        ];
        let coverage = image::GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
            let in_a = x >= ax && x < a_right && y >= ay && y < a_bottom;
            let in_b = x >= bx && x < b_right && y >= by && y < b_bottom;
            image::Luma([if in_a || in_b { 255 } else { 0 }])
        });
        let result = effective_pixel_coverage(&coverage, &[quad_a, quad_b], (0.0, 0.0))
            .expect("integer source quadrilaterals have measurable coverage");
        let overlap_width = a_right.min(b_right).saturating_sub(ax.max(bx));
        let overlap_height = a_bottom.min(b_bottom).saturating_sub(ay.max(by));
        let expected_union = u64::from(aw) * u64::from(ah)
            + u64::from(bw) * u64::from(bh)
            - u64::from(overlap_width) * u64::from(overlap_height);
        prop_assert_eq!(result.projected_union, expected_union);
        prop_assert_eq!(result.output_nontransparent, expected_union);
        prop_assert!(result.ratio + 1.0e-12 >= QUALITY_EFFECTIVE_PIXEL_RATIO_MIN);
    }
}

// Stage 7 Quality_Gate geometry classification properties (15.13, 15.15).
use super::quality_gate::{detect_slanted_edge, flat_roi, normalized_gradient_energy};

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 65: 倾斜边与平坦 ROI 判定互斥且满足测量门槛。
    #[test]
    fn property_65_slanted_edge_and_flat_roi_are_classified(
        angle_deg in 3.5f64..12.5,
        low in 0.04f32..0.20,
        high in 0.78f32..0.96,
        flat in 0.15f32..0.85,
    ) {
        let width = 256u32;
        let height = 256u32;
        let theta = angle_deg.to_radians();
        let edge = image::Rgb32FImage::from_fn(width, height, |x, y| {
            let distance = f64::from(x) - f64::from(y) * theta.tan() - 96.0;
            let value = if distance >= 0.0 { high } else { low };
            image::Rgb([value; 3])
        });
        let coverage = image::GrayImage::from_pixel(width, height, image::Luma([u8::MAX]));
        let evidence = detect_slanted_edge(&edge).expect("qualified synthetic edge must be detected");
        prop_assert!(evidence.angle_deg >= 3.0 && evidence.angle_deg <= 15.0);
        prop_assert!(evidence.length_px >= 128.0);
        prop_assert!(evidence.contrast >= 0.20);
        prop_assert!(evidence.line_fit_rms_px <= 0.5);
        prop_assert!(flat_roi(&edge, &coverage).is_err());

        let flat_image = image::Rgb32FImage::from_pixel(width, height, image::Rgb([flat; 3]));
        let flat_evidence = flat_roi(&flat_image, &coverage)
            .expect("constant, fully covered ROI must be classified as flat");
        prop_assert_eq!(flat_evidence.opaque_pixels, u64::from(width) * u64::from(height));
        prop_assert!(flat_evidence.low_frequency_luma_std <= 0.02);
        prop_assert!(detect_slanted_edge(&flat_image).is_err());
    }

    // Feature: layered-camera-group-focus-stitching, Property 67: 归一化梯度能量在恒等局部尺度下保持非负且对常量 ROI 为零。
    #[test]
    fn property_67_normalized_gradient_energy_is_scale_consistent(
        value in 0.0f32..1.0,
        local_scale in 0.25f64..4.0,
    ) {
        let width = 32u32;
        let height = 32u32;
        let flat = image::Rgb32FImage::from_pixel(width, height, image::Rgb([value; 3]));
        let energy = normalized_gradient_energy(&flat, local_scale)
            .expect("constant finite ROI must be measurable");
        prop_assert!(energy.is_finite());
        prop_assert!(energy.abs() <= 1.0e-12);

        let ramp = image::Rgb32FImage::from_fn(width, height, |x, _y| {
            let channel = (f64::from(x) / f64::from(width - 1)) as f32;
            image::Rgb([channel; 3])
        });
        let unit = normalized_gradient_energy(&ramp, 1.0)
            .expect("finite ramp ROI must be measurable");
        let scaled = normalized_gradient_energy(&ramp, local_scale)
            .expect("finite ramp ROI must be measurable");
        prop_assert!(unit.is_finite() && scaled.is_finite());
        prop_assert!(unit >= 0.0 && scaled >= 0.0);
        prop_assert!((scaled * local_scale - unit).abs() <= 1.0e-10 * unit.max(1.0));
    }
}

// Later Quality_Gate pure-property checks (15.18-15.22).
use super::quality_gate::{measure_boundary_strokes, trace_moore_boundary};
use super::quality_gate_runner::{
    Criterion, OwnerSharpnessCriterionInput, build_owner_sharpness_criterion,
    classify_unmeasurable_reason, criterion_verdict, overall_verdict, record_unmeasurable,
};
use super::report::{
    FailedMeasurementRecord, QualityGateReport, UnmeasurableCategory, UnmeasurableRecord,
    WorldPoint,
};

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 70: 边界笔画配准偏差有界
    #[test]
    fn property_70_boundary_stroke_pairing_is_bounded(
        level in 0.72f32..0.98f32,
    ) {
        let width = 128u32;
        let height = 128u32;
        let mut owners = vec![1u16; (width * height) as usize];
        let mut image = image::Rgb32FImage::from_pixel(width, height, image::Rgb([0.10; 3]));
        for y in 32..96 {
            for x in 32..96 {
                owners[(y * width + x) as usize] = 2;
            }
        }
        // Put equal-width strokes on both sides of the top ownership edge.
        // The measurement samples ±16 px from the boundary, so the paired
        // edges are deliberately symmetric around y=32.
        for y in 0..=20 {
            for x in 0..width {
                image.put_pixel(x, y, image::Rgb([level; 3]));
            }
        }
        for y in 44..height {
            for x in 0..width {
                image.put_pixel(x, y, image::Rgb([level; 3]));
            }
        }
        let coverage = image::GrayImage::from_pixel(width, height, image::Luma([255]));
        let regions = label_owner_regions(&owners, &coverage)
            .expect("closed owner regions must label");
        let mut boundary = trace_moore_boundary(&regions, 1, 2)
            .expect("the inner owner must have a public boundary");
        prop_assert!(!boundary.is_empty());
        // Feed the measurement a straight top-segment window.  It is a
        // contiguous Moore-traced boundary slice, and avoids sampling the
        // square's corner where the normal is intentionally undefined.
        boundary = (48..80).map(|x| (x, 32)).collect();
        let report = measure_boundary_strokes(&image, &boundary, (0.0, 0.0))
            .expect("the synthetic boundary has pairable strokes");
        prop_assert!(report.pairable_count > 0);
        prop_assert!(report.p95_error_px <= 3.0);
        prop_assert!(report.max_error_px <= 3.0);
        for measurement in report.measurements {
            prop_assert!(measurement.orientation_error_deg <= 10.0);
        }
    }

    // Feature: layered-camera-group-focus-stitching, Property 71: 所有者锐度短缺覆盖率按 Textured_Pixel 计算
    #[test]
    fn property_71_owner_sharpness_coverage_uses_runtime_statistics(
        shortfall_count in 0usize..=1024usize,
        transparent_count in 0usize..=64usize,
        unresolved_count in 0usize..=16usize,
    ) {
        let mut coverage = vec![255u8; 1024];
        let transparent_start = 1024 - transparent_count;
        coverage[transparent_start..].fill(0);
        let mut textured = vec![255u8; 1024];
        // The final quarter is flat and must not enter the criterion's
        // denominator, even though it remains covered.
        textured[768..transparent_start].fill(0);
        let mut shortfall = vec![5u8; 1024];
        shortfall[..shortfall_count.min(768)].fill(6);
        let mut disagreement = vec![0u8; 1024];
        disagreement[..shortfall_count.min(768)].fill(21);
        let covered_end = transparent_start.min(1024);
        let unresolved = unresolved_count.min(covered_end);
        let unresolved_start = covered_end.saturating_sub(unresolved);
        for index in unresolved_start..covered_end {
            shortfall[index] = u8::MAX;
            disagreement[index] = u8::MAX;
        }
        let coverage_image = image::GrayImage::from_raw(32, 32, coverage.clone())
            .expect("property coverage plane has the expected dimensions");
        let confidence = vec![0.8f32; 1024];
        let mut evidence_records = Vec::new();
        let criterion = build_owner_sharpness_criterion(
            OwnerSharpnessCriterionInput {
                coverage: &coverage_image,
                confidence: &confidence,
                textured: Some(&textured),
                shortfall: Some(&shortfall),
                disagreement: Some(&disagreement),
                unresolved_pixel_count: unresolved as u64,
                world_origin: (0.0, 0.0),
                unmeasurable: &mut evidence_records,
            },
        );
        let report = criterion.finish(&mut evidence_records);
        let unresolved_textured = (unresolved_start..covered_end)
            .filter(|&index| index < 768)
            .count();
        let expected_textured = (768usize.saturating_sub(unresolved_textured)) as u64;
        let expected_shortfall = shortfall_count.min(expected_textured as usize) as u64;
        let expected_shortfall_ratio = (expected_textured > 0)
            .then_some(expected_shortfall as f64 / expected_textured as f64);
        let coverage_ratio = (expected_textured > 0).then_some(
            1.0 - expected_shortfall as f64 / expected_textured as f64,
        );
        prop_assert_eq!(
            coverage_ratio.map(|value| value >= 0.99),
            (expected_textured > 0).then_some(expected_shortfall * 100 <= expected_textured),
        );
        prop_assert_eq!(report.textured_pixel_count, Some(expected_textured));
        prop_assert_eq!(report.disagreement_veto_count, expected_shortfall);
        prop_assert_eq!(report.unresolved_pixel_count, unresolved as u64);
        prop_assert_eq!(report.textured_shortfall_ratio, expected_shortfall_ratio);
        let expected_verdict = if expected_textured == 0 || unresolved > 0 {
            QualityGateVerdict::InsufficientEvidence
        } else if expected_shortfall * 100 <= expected_textured {
            QualityGateVerdict::Pass
        } else {
            QualityGateVerdict::Fail
        };
        prop_assert_eq!(report.verdict, expected_verdict);
        prop_assert_eq!(report.measurable_count, u64::from(expected_textured > 0) as usize);
        if expected_textured > 0 {
            prop_assert_eq!(report.measured.len(), 1);
            prop_assert_eq!(
                report.measured[0].measured.to_bits(),
                (1.0 - expected_shortfall as f64 / expected_textured as f64).to_bits()
            );
        } else {
            prop_assert!(report.measured.is_empty());
        }
        // Transparent pixels and covered flat pixels are excluded from the
        // Textured_Pixel denominator, so arbitrary evidence in those slices
        // cannot change the owner criterion.
        let mut irrelevant_shortfall = shortfall.clone();
        let mut irrelevant_disagreement = disagreement.clone();
        irrelevant_shortfall[768..].fill(100);
        irrelevant_disagreement[768..].fill(100);
        let mut irrelevant_records = Vec::new();
        let irrelevant = build_owner_sharpness_criterion(
            OwnerSharpnessCriterionInput {
                coverage: &coverage_image,
                confidence: &confidence,
                textured: Some(&textured),
                shortfall: Some(&irrelevant_shortfall),
                disagreement: Some(&irrelevant_disagreement),
                unresolved_pixel_count: unresolved as u64,
                world_origin: (0.0, 0.0),
                unmeasurable: &mut irrelevant_records,
            },
        )
        .finish(&mut irrelevant_records);
        prop_assert_eq!(irrelevant.verdict, report.verdict);
        prop_assert_eq!(&irrelevant.measured, &report.measured);
        // The owner criterion uses only the owner-shortfall plane. Changing
        // Sharpness_Confidence can alter diagnostic ratios, but never the
        // verdict or the measured coverage value.
        let mut changed_records = Vec::new();
        let changed_confidence = vec![f32::NAN; 1024];
        let changed = build_owner_sharpness_criterion(
            OwnerSharpnessCriterionInput {
                coverage: &coverage_image,
                confidence: &changed_confidence,
                textured: Some(&textured),
                shortfall: Some(&shortfall),
                disagreement: Some(&disagreement),
                unresolved_pixel_count: unresolved as u64,
                world_origin: (0.0, 0.0),
                unmeasurable: &mut changed_records,
            },
        )
        .finish(&mut changed_records);
        prop_assert_eq!(changed.verdict, report.verdict);
        prop_assert_eq!(&changed.measured, &report.measured);

        // Exercise the three evidence/threshold boundaries explicitly so the
        // generated data cannot accidentally omit one of the required cases.
        let boundary_coverage = image::GrayImage::from_pixel(10, 10, image::Luma([255]));
        let boundary_confidence = vec![0.8f32; 100];
        let boundary_textured = vec![255u8; 100];
        let zero_disagreement = vec![0u8; 100];
        let mut one_percent_shortfall = vec![5u8; 100];
        one_percent_shortfall[0] = 6;
        let mut boundary_records = Vec::new();
        let boundary_pass = build_owner_sharpness_criterion(
            OwnerSharpnessCriterionInput {
                coverage: &boundary_coverage,
                confidence: &boundary_confidence,
                textured: Some(&boundary_textured),
                shortfall: Some(&one_percent_shortfall),
                disagreement: Some(&zero_disagreement),
                unresolved_pixel_count: 0,
                world_origin: (0.0, 0.0),
                unmeasurable: &mut boundary_records,
            },
        )
        .finish(&mut boundary_records);
        prop_assert_eq!(boundary_pass.verdict, QualityGateVerdict::Pass);
        prop_assert_eq!(boundary_pass.measured[0].measured, 0.99);
        one_percent_shortfall[1] = 6;
        let mut boundary_fail_records = Vec::new();
        let boundary_fail = build_owner_sharpness_criterion(
            OwnerSharpnessCriterionInput {
                coverage: &boundary_coverage,
                confidence: &boundary_confidence,
                textured: Some(&boundary_textured),
                shortfall: Some(&one_percent_shortfall),
                disagreement: Some(&zero_disagreement),
                unresolved_pixel_count: 0,
                world_origin: (0.0, 0.0),
                unmeasurable: &mut boundary_fail_records,
            },
        )
        .finish(&mut boundary_fail_records);
        prop_assert_eq!(boundary_fail.verdict, QualityGateVerdict::Fail);

        let no_textured = vec![0u8; 100];
        let mut no_textured_records = Vec::new();
        let no_textured_result = build_owner_sharpness_criterion(
            OwnerSharpnessCriterionInput {
                coverage: &boundary_coverage,
                confidence: &boundary_confidence,
                textured: Some(&no_textured),
                shortfall: Some(&one_percent_shortfall),
                disagreement: Some(&zero_disagreement),
                unresolved_pixel_count: 0,
                world_origin: (0.0, 0.0),
                unmeasurable: &mut no_textured_records,
            },
        )
        .finish(&mut no_textured_records);
        prop_assert_eq!(no_textured_result.verdict, QualityGateVerdict::InsufficientEvidence);

        let mut unresolved_shortfall = vec![5u8; 100];
        unresolved_shortfall[0] = u8::MAX;
        let mut unresolved_records = Vec::new();
        let unresolved_result = build_owner_sharpness_criterion(
            OwnerSharpnessCriterionInput {
                coverage: &boundary_coverage,
                confidence: &boundary_confidence,
                textured: Some(&boundary_textured),
                shortfall: Some(&unresolved_shortfall),
                disagreement: Some(&zero_disagreement),
                unresolved_pixel_count: 1,
                world_origin: (0.0, 0.0),
                unmeasurable: &mut unresolved_records,
            },
        )
        .finish(&mut unresolved_records);
        prop_assert_eq!(unresolved_result.verdict, QualityGateVerdict::InsufficientEvidence);
    }

    // Feature: layered-camera-group-focus-stitching, Property 72: 测量项计数恒等且证据不足可判定
    #[test]
    fn property_72_measurement_counts_are_conservative(
        measurable in 0usize..=32,
        content_count in 0usize..=32,
        technical_count in 0usize..=16,
        has_failed in any::<bool>(),
        diagnostic in any::<bool>(),
        x in -10000.0f64..10000.0,
        y in -10000.0f64..10000.0,
    ) {
        let content_reasons = [
            degradation::ROI_NOT_SLANTED_EDGE,
            degradation::ROI_NOT_FLAT,
            degradation::BOUNDARY_NO_PAIRABLE_EDGE,
            degradation::SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED,
            degradation::SLANTED_EDGE_TOO_SHORT,
            degradation::SLANTED_EDGE_CONTRAST_INSUFFICIENT,
            degradation::SLANTED_EDGE_ANGLE_OUT_OF_RANGE,
            degradation::BOUNDARY_LOW_CONTRAST,
            degradation::BOUNDARY_ORIENTATION_MISMATCH,
        ];
        let technical_reasons = [
            degradation::OWNER_SOURCE_UNDECODABLE,
            degradation::PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED,
            degradation::LOCAL_SCALE_UNMEASURABLE,
            degradation::TEXTURED_PIXEL_PLANE_UNAVAILABLE,
            degradation::DIAGNOSTICS_ROI_INVALID,
        ];
        for reason in content_reasons {
            prop_assert_eq!(classify_unmeasurable_reason(reason), UnmeasurableCategory::ContentNotApplicable);
        }
        for reason in technical_reasons {
            prop_assert_eq!(classify_unmeasurable_reason(reason), UnmeasurableCategory::Technical);
        }
        let failed = has_failed && measurable > 0;
        let mut criterion_records = Vec::new();
        for name in super::quality_gate::QUALITY_CRITERIA {
            let mut criterion = Criterion::new(name, 1.0);
            criterion.diagnostic = diagnostic;
            for index in 0..measurable {
                criterion.push((x + index as f64, y), 1.0, "owner://property72", !(failed && index == 0));
            }
            let mut unmeasurable = Vec::new();
            for index in 0..content_count + technical_count {
                let reason = if index < content_count {
                    content_reasons[index % content_reasons.len()]
                } else {
                    technical_reasons[(index - content_count) % technical_reasons.len()]
                };
                let roi = QualityRoi {
                    x: 0, y: 0, side: 512, region_label: 1, owner: 1,
                    world_origin: (x + index as f64, y), boundary_clearance: 16.0,
                };
                record_unmeasurable(&mut criterion, &mut unmeasurable, &roi, reason);
            }
            let result = criterion.finish(&mut unmeasurable);
            prop_assert_eq!(result.measurable_count, measurable);
            prop_assert_eq!(result.content_not_applicable_count, content_count);
            prop_assert_eq!(result.technical_unmeasurable_count, technical_count);
            prop_assert_eq!(result.unmeasurable_count, unmeasurable.len());
            prop_assert_eq!(result.measurable_count + result.unmeasurable_count,
                measurable + content_count + technical_count);
            prop_assert_eq!(result.unmeasurable_reasons.values().sum::<usize>(), unmeasurable.len());
            for (index, record) in unmeasurable.iter().enumerate() {
                let expected_category = if index < content_count {
                    UnmeasurableCategory::ContentNotApplicable
                } else {
                    UnmeasurableCategory::Technical
                };
                prop_assert_eq!(&record.criterion, name);
                prop_assert_eq!(record.world, WorldPoint { x: x + index as f64, y });
                prop_assert!(!record.reason.is_empty());
                prop_assert_eq!(record.category, expected_category);
                let encoded = serde_json::to_value(record).unwrap();
                prop_assert_eq!(&encoded["category"], if index < content_count {
                    "content_not_applicable"
                } else {
                    "technical"
                });
            }
            let whole_image = matches!(name, "effective_pixel_count" | "owner_sharpness_coverage");
            let conditional = matches!(name, "mtf50_normalized" | "noise_sigma_ratio" | "boundary_stroke_alignment");
            let expected = if diagnostic {
                QualityGateVerdict::NotApplicable
            } else if whole_image && measurable == 0 {
                QualityGateVerdict::InsufficientEvidence
            } else if whole_image {
                if failed { QualityGateVerdict::Fail } else { QualityGateVerdict::Pass }
            } else if technical_count * 5 > measurable + technical_count {
                QualityGateVerdict::InsufficientEvidence
            } else if failed {
                QualityGateVerdict::Fail
            } else if conditional && measurable < 8 {
                QualityGateVerdict::NotApplicable
            } else if !conditional && measurable < 8 {
                QualityGateVerdict::InsufficientEvidence
            } else {
                QualityGateVerdict::Pass
            };
            prop_assert_eq!(result.verdict, expected);
            criterion_records.push(result);
        }
        let expected_overall = if criterion_records.iter().any(|record| record.verdict == QualityGateVerdict::Fail) {
            QualityGateVerdict::Fail
        } else if criterion_records.iter().any(|record| record.verdict == QualityGateVerdict::InsufficientEvidence) {
            QualityGateVerdict::InsufficientEvidence
        } else {
            QualityGateVerdict::Pass
        };
        prop_assert_eq!(overall_verdict(&criterion_records), expected_overall);
        // Exactly 20% is allowed; content-inapplicable observations never enter this denominator.
        for name in super::quality_gate::QUALITY_CRITERIA {
            let whole_image = matches!(name, "effective_pixel_count" | "owner_sharpness_coverage");
            prop_assert_eq!(criterion_verdict(name, 8, 2, false, false, false), QualityGateVerdict::Pass);
            prop_assert_eq!(criterion_verdict(name, 8, 3, false, false, false), if whole_image {
                QualityGateVerdict::Pass
            } else {
                QualityGateVerdict::InsufficientEvidence
            });
        }
        // A measurable conditional criterion with a failed observation is a
        // real failure even when fewer than eight observations are available.
        prop_assert_eq!(
            criterion_verdict("boundary_stroke_alignment", 1, 0, true, false, false),
            QualityGateVerdict::Fail
        );
        let pass_and_not_applicable = [QualityGateVerdict::Pass, QualityGateVerdict::NotApplicable]
            .map(|verdict| QualityGateCriterionRecord { verdict, ..QualityGateCriterionRecord::default() });
        prop_assert_eq!(overall_verdict(&pass_and_not_applicable), QualityGateVerdict::Pass);
        prop_assert_eq!(serde_json::to_value(QualityGateVerdict::NotApplicable).unwrap(), "not_applicable");
    }

    // Feature: layered-camera-group-focus-stitching, Property 73: 阻止导出时不残留结果且保留诊断
    #[test]
    fn property_73_failed_gate_report_retains_diagnostics(
        measured in 0.0f64..2.0,
        x in -1000.0f64..1000.0,
        y in -1000.0f64..1000.0,
    ) {
        let report = QualityGateReport {
            verdict: QualityGateVerdict::Fail,
            roi_count: 1,
            criteria: vec![QualityGateCriterionRecord {
                name: "mtf50_normalized".to_string(),
                threshold: 0.93,
                threshold_min: None,
                threshold_max: None,
                measurable_count: 1,
                unmeasurable_count: 0,
                content_not_applicable_count: 0,
                technical_unmeasurable_count: 0,
                verdict: QualityGateVerdict::Fail,
                station_statistics: Vec::new(),
                unmeasurable_reasons: BTreeMap::new(),
                measured: Vec::new(),
                diagnostic: false,
                failed: vec![FailedMeasurementRecord {
                    world: WorldPoint { x, y },
                    measured,
                    owner_path: "owner://synthetic".to_string(),
                }],
                textured_pixel_count: None,
                excluded_flat_pixel_count: None,
                textured_low_confidence_ratio: None,
                flat_low_confidence_ratio: None,
                textured_shortfall_ratio: None,
                flat_shortfall_ratio: None,
                shortfall_histogram: Vec::new(),
                disagreement_veto_count: 0,
                unresolved_pixel_count: 0,
            }],
            unmeasurable: vec![UnmeasurableRecord {
                criterion: "noise_sigma_ratio".to_string(),
                world: WorldPoint { x, y },
                reason: "roi_not_flat".to_string(),
                category: UnmeasurableCategory::ContentNotApplicable,
            }],
            timing: super::report::QualityGateTimingRecord::default(),
            confidence_scores: Default::default(),
            roi_photometry: Vec::new(),
            owner_reverse_lookup_failures: Default::default(),
        };
        let encoded = serde_json::to_value(&report).expect("quality report is serializable");
        prop_assert_eq!(&encoded["verdict"], "fail");
        prop_assert_eq!(&encoded["roi_count"], 1);
        prop_assert_eq!(&encoded["criteria"][0]["failed"][0]["owner_path"], "owner://synthetic");
        prop_assert_eq!(&encoded["unmeasurable"][0]["reason"], "roi_not_flat");
    }

    // Feature: layered-camera-group-focus-stitching, Property 74: Quality_Gate 的 ROI 集合与结论可复现
    #[test]
    fn property_74_roi_selection_is_reproducible(
        split in 544u32..1024u32,
        origin_x in -4096.0f64..4096.0,
        origin_y in -4096.0f64..4096.0,
    ) {
        let width = 1024u32;
        let height = 512u32;
        let owners = (0..width * height).map(|index| {
            if index % width < split { 1u16 } else { 2u16 }
        }).collect::<Vec<_>>();
        let coverage = image::GrayImage::from_pixel(width, height, image::Luma([255]));
        let regions_a = label_owner_regions(&owners, &coverage).expect("ownership labels");
        let regions_b = label_owner_regions(&owners, &coverage).expect("same ownership labels");
        let selection_a = select_quality_rois(&regions_a, (origin_x, origin_y));
        let selection_b = select_quality_rois(&regions_b, (origin_x, origin_y));
        prop_assert_eq!(selection_a, selection_b);
        prop_assert_eq!(regions_a.labels, regions_b.labels);
        prop_assert_eq!(regions_a.distance_to_boundary, regions_b.distance_to_boundary);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 77: closure
    // degradation is recorded as geometry evidence while the Quality_Gate's
    // complete measured criterion set, thresholds, and conclusion remain
    // unchanged.
    //
    // **Validates: Requirements 7.7, 7.9, 11.15, 12.3**
    #[test]
    fn property_77_closure_degradation_preserves_quality_gate_record(
        seed in any::<u64>(),
        reliable in any::<bool>(),
    ) {
        use crate::panorama_stitching::station_relation_test_access::{
            self, ClosureRelationView,
        };
        use super::quality_gate::{self, SourceGeometry};
        use super::quality_gate_runner::{run_quality_gate, QualityGateInput, QualitySource};
        use super::residual_warp::ResidualWarp;

        let _run_scope = degradation::begin_run_scope();
        degradation::reset_run_ledger();

        // Run the production Quality_Gate twice around the production closure
        // boundary. The fixture has one full 512px owner region so all nine
        // criterion records are built by `run_quality_gate`, rather than a
        // hand-written report standing in for its output.
        let run_quality_gate_fixture = || {
            let width = 512;
            let height = 512;
            let output = image::Rgb32FImage::from_fn(width, height, |x, y| {
                let ramp = ((x.wrapping_mul(17) + y.wrapping_mul(31)) % 251) as f32 / 251.0;
                image::Rgb([0.2 + ramp * 0.6, 0.25 + ramp * 0.5, 0.3 + ramp * 0.4])
            });
            let coverage = image::GrayImage::from_pixel(width, height, image::Luma([255]));
            let ownership = vec![1u16; (width * height) as usize];
            let confidence = vec![0.8f32; (width * height) as usize];
            let textured = vec![255u8; (width * height) as usize];
            let shortfall = vec![0u8; (width * height) as usize];
            let disagreement = vec![0u8; (width * height) as usize];
            let residual = ResidualWarp::new();
            let source = QualitySource {
                owner: 1,
                path: "property-77-owner.raw".to_string(),
                geometry: SourceGeometry {
                    member_to_anchor: Matrix3::identity(),
                    tile_to_world: Matrix3::identity(),
                    station_id: 0,
                },
                dimensions: (width, height),
                native_registration: None,
            };
            let input = QualityGateInput {
                output: &output,
                coverage: &coverage,
                ownership: &ownership,
                confidence: &confidence,
                textured: Some(&textured),
                owner_shortfall: Some(&shortfall),
                owner_disagreement: Some(&disagreement),
                unresolved_pixel_count: 0,
                world_origin: (0.0, 0.0),
                sources: std::slice::from_ref(&source),
                accepted_station_relations: &[],
                residual: &residual,
                acceptance_render_scale: 1.0,
                final_sharpen_amount: 0.0,
            };
            run_quality_gate(&input, &mut |_source| Ok(output.clone()))
        };
        let quality_before = run_quality_gate_fixture();

        let (run, expected_status) = if reliable {
            let (dimensions, rows, columns, base, relations) = accepted_closure_fixture(seed);
            (
                station_relation_test_access::closure_run(
                    dimensions,
                    &rows,
                    &columns,
                    &base,
                    &relations,
                ),
                super::report::ClosureStatus::Converged,
            )
        } else {
            let signed_zero = if seed & 1 == 0 { 0.0 } else { -0.0 };
            let base = vec![
                translation_pose(signed_zero, -0.0),
                translation_pose(100.25, 0.0),
                translation_pose(200.5, 0.0),
            ];
            let relation = |left, right, tx| ClosureRelationView {
                score: 1.0,
                left,
                right,
                left_to_right: translation_pose(tx, 0.0),
                independent_support: 2,
                median_error_px: 0.0,
            };
            (
                station_relation_test_access::closure_run(
                    (400, 400),
                    &[2, 0, 1],
                    &[1, 2, 0],
                    &base,
                    &[
                        relation(0, 1, -109.25),
                        relation(1, 2, -109.25),
                        relation(0, 2, -191.5),
                    ],
                ),
                super::report::ClosureStatus::Unreliable,
            )
        };
        prop_assert_eq!(run.report.status, expected_status);
        let quality_after = run_quality_gate_fixture();

        let criterion_signature = |report: &QualityGateReport| {
            report
                .criteria
                .iter()
                .map(|criterion| {
                    (
                        criterion.name.clone(),
                        criterion.threshold,
                        criterion.threshold_min,
                        criterion.threshold_max,
                        criterion.verdict,
                    )
                })
                .collect::<Vec<_>>()
        };
        prop_assert_eq!(quality_before.verdict, quality_after.verdict);
        prop_assert_eq!(criterion_signature(&quality_before), criterion_signature(&quality_after));
        prop_assert_eq!(
            quality_before.criteria.iter().map(|criterion| criterion.name.as_str()).collect::<Vec<_>>(),
            quality_gate::QUALITY_CRITERIA.to_vec(),
        );
        for criterion in &quality_after.criteria {
            let expected_threshold = match criterion.name.as_str() {
                "local_scale_median" => quality_gate::QUALITY_LOCAL_SCALE_MEDIAN_MIN,
                "local_scale_pixel_ratio" => quality_gate::QUALITY_LOCAL_SCALE_PIXEL_RATIO_MIN,
                "effective_pixel_count" => quality_gate::QUALITY_EFFECTIVE_PIXEL_RATIO_MIN,
                "mtf50_normalized" => quality_gate::QUALITY_MTF50_RATIO_MIN,
                "gradient_energy_normalized" => quality_gate::QUALITY_GRADIENT_RATIO_MIN,
                "noise_sigma_ratio" => quality_gate::QUALITY_NOISE_RATIO_MIN,
                "roi_delta_e00" => quality_gate::QUALITY_ROI_DELTA_E00_MAX,
                "boundary_stroke_alignment" => quality_gate::QUALITY_BOUNDARY_P95_MAX,
                "owner_sharpness_coverage" => quality_gate::OWNER_SHARPNESS_COVERAGE_MIN,
                _ => unreachable!("run_quality_gate emitted an unknown criterion"),
            };
            prop_assert_eq!(criterion.threshold, expected_threshold);
        }

        let ledger = degradation::run_ledger_snapshot();
        let recorder = super::report::StackReportRecorder::isolated("property-77", None);
        recorder.update(|report| {
            report.quality_gate = quality_after.clone();
            report.closure = run.report.clone();
        });
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();
        prop_assert_eq!(report.closure, run.report);
        if reliable {
            let has_unreliable_degradation = report.degradation.entries.iter().any(|entry| {
                matches!(
                    entry.reason.as_str(),
                    degradation::CLOSURE_UNRELIABLE_RESIDUAL
                        | degradation::CLOSURE_UNRELIABLE_ITERATIONS
                        | degradation::CLOSURE_UNRELIABLE_PAIR_P95
                )
            });
            prop_assert!(!has_unreliable_degradation);
        } else {
            let has_unreliable_degradation = report.degradation.entries.iter().any(|entry| {
                matches!(
                    entry.reason.as_str(),
                    degradation::CLOSURE_UNRELIABLE_RESIDUAL
                        | degradation::CLOSURE_UNRELIABLE_ITERATIONS
                        | degradation::CLOSURE_UNRELIABLE_PAIR_P95
                )
            });
            prop_assert!(has_unreliable_degradation);
        }
    }
}

fn exhaustive_vertical_seam_cost(costs: &[f64], width: usize, height: usize) -> f64 {
    fn visit(costs: &[f64], width: usize, height: usize, row: usize, column: usize) -> f64 {
        let here = costs[row * width + column];
        if row + 1 == height {
            return here;
        }
        let start = column.saturating_sub(1);
        let end = (column + 1).min(width - 1);
        here + (start..=end)
            .map(|next| visit(costs, width, height, row + 1, next))
            .fold(f64::INFINITY, f64::min)
    }
    (0..width)
        .map(|column| visit(costs, width, height, 0, column))
        .fold(f64::INFINITY, f64::min)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 53: 接缝只在双覆盖区且代价最低
    #[test]
    fn property_53_seam_cost_is_minimum_with_boundary_penalty(
        (width, height) in (1usize..=8, 1usize..=8),
        disagreement in prop::collection::vec(0.0f64..=2.0, 1..=64),
        boundary_distance in prop::collection::vec(0.0f64..=32.0, 1..=64),
    ) {
        let cell_count = width * height;
        let costs = (0..cell_count)
            .map(|index| compositor::seam_candidate_cost(
                disagreement[index % disagreement.len()],
                boundary_distance[index % boundary_distance.len()],
            ))
            .collect::<Vec<_>>();
        let (path, selected_cost) = compositor::minimum_vertical_seam(&costs, width, height)
            .expect("a non-empty overlap has a seam");
        prop_assert_eq!(path.len(), height);
        for row in 1..height {
            prop_assert!((path[row] as isize - path[row - 1] as isize).abs() <= 1);
        }
        prop_assert!((selected_cost - exhaustive_vertical_seam_cost(&costs, width, height)).abs() <= 1.0e-10);
        for index in 0..cell_count {
            if boundary_distance[index % boundary_distance.len()] < 16.0 {
                let disagreement = disagreement[index % disagreement.len()];
                prop_assert!(costs[index] - disagreement >= 1.0 - 1.0e-12);
            }
        }
    }

    // Feature: layered-camera-group-focus-stitching, Property 59: 窄重叠沿中线取接缝
    #[test]
    fn property_59_narrow_overlap_uses_its_centerline(
        width in 1usize..32,
        height in 1usize..32,
    ) {
        let Some((vertical, seam)) = compositor::narrow_overlap_centerline(width, height) else {
            prop_assert!(width.max(height) >= 32);
            return Ok(());
        };
        if vertical {
            prop_assert_eq!(seam, vec![width / 2; height]);
        } else {
            prop_assert_eq!(seam, vec![height / 2; width]);
        }
        prop_assert!(width.max(height) < 32);
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 87: 内存超限中止且不残留
    // 超过内存门槛的运行只删除本次暂存结果，不发布部分输出。
    //
    // **Validates: Requirements 14.3**
    #[test]
    fn property_87_memory_overrun_removes_only_run_staging(
        peak in 1u64..1024,
        threshold in 1u64..1024,
    ) {
        let directory = tempfile::tempdir().expect("temporary output directory");
        let temporary = directory.path().join(".stack-result.current.tmp");
        let final_path = directory.path().join("stack-result.tiff");
        std::fs::write(&temporary, b"partial output").expect("staged output");
        let mut ledger = DegradationLedger::new();
        if peak > threshold {
            ledger.record(
                degradation::MEMORY_THRESHOLD_EXCEEDED,
                serde_json::json!({
                    "threshold_bytes": threshold,
                    "peak_rss_bytes": peak,
                    "sample_count": 1,
                }),
            );
        }
        let publication = DegradationManager::new(&ledger)
            .publish_staged_output(
                &mut StandardFinalOutputFileSystem,
                &temporary,
                &final_path,
            )
            .expect("staged publication decision");

        // The terminating report must carry the same evidence that drove the
        // rejection.  Keep this oracle independent of the publication helper:
        // a future change that drops one of these fields must fail the property
        // even if the temporary file cleanup still happens to be correct.
        let recorder = super::report::StackReportRecorder::isolated("property-87", None);
        recorder.update(|report| {
            report.resources.memory_threshold_bytes = threshold;
            report.resources.peak_rss_bytes = peak;
            report.resources.rss_sample_count = 1;
            report.resources.memory_threshold_exceeded = peak > threshold;
        });
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();
        prop_assert_eq!(
            report.resources.memory_threshold_bytes,
            threshold,
            "the effective threshold must be reported"
        );
        prop_assert_eq!(report.resources.peak_rss_bytes, peak);
        prop_assert_eq!(report.resources.rss_sample_count, 1);
        prop_assert_eq!(
            report.resources.memory_threshold_exceeded,
            peak > threshold
        );
        prop_assert!(!temporary.exists(), "run staging must never remain");
        if peak > threshold {
            prop_assert_eq!(publication, degradation::OutputPublication::Rejected);
            prop_assert!(!final_path.exists(), "overrun must not publish partial output");
            let entry = report
                .degradation
                .entries
                .iter()
                .find(|entry| entry.reason == degradation::MEMORY_THRESHOLD_EXCEEDED)
                .expect("memory overrun must be recorded");
            prop_assert_eq!(entry.detail["threshold_bytes"].as_u64(), Some(threshold));
            prop_assert_eq!(entry.detail["peak_rss_bytes"].as_u64(), Some(peak));
            prop_assert_eq!(entry.detail["sample_count"].as_u64(), Some(1));
        } else {
            prop_assert!(matches!(publication, degradation::OutputPublication::Published(_)));
            prop_assert!(final_path.exists(), "accepted output must be published");
            prop_assert!(report.degradation.entries.is_empty());
        }
    }

    // Feature: layered-camera-group-focus-stitching, Property 91: 取消后不残留且状态可追溯
    // 用户取消产生 cancelled 状态，只清理本次暂存结果并保留最终输出路径为空。
    //
    // **Validates: Requirements 14.8**
    #[test]
    fn property_91_cancelled_run_has_no_staging_residue(seed in any::<u64>()) {
        let directory = tempfile::tempdir().expect("temporary output directory");
        let temporary = directory.path().join(format!(".stack-result-{seed}.tmp"));
        let final_path = directory.path().join("stack-result.tiff");
        let source_path = directory.path().join("source.NEF");
        let cache_path = directory.path().join("virtual-tile.cache");
        let source_bytes = seed.to_le_bytes().repeat(4);
        let cache_bytes = seed.rotate_left(17).to_le_bytes().repeat(8);
        std::fs::write(&source_path, &source_bytes).expect("source fixture");
        std::fs::write(&cache_path, &cache_bytes).expect("cache fixture");
        let source_before = virtual_tile::source_file_sha256(&source_path)
            .expect("source digest before cancellation");
        let cache_before = std::fs::read(&cache_path).expect("cache bytes before cancellation");
        std::fs::write(&temporary, seed.to_le_bytes()).expect("staged output");
        let mut ledger = DegradationLedger::new();
        ledger.record(degradation::RUN_CANCELLED_BY_USER, serde_json::json!({
            "generation": seed,
        }));
        let manager = DegradationManager::new(&ledger);
        prop_assert_eq!(manager.decision(), degradation::OutcomeDecision::Cancelled);
        let publication = manager
            .publish_staged_output(
                &mut StandardFinalOutputFileSystem,
                &temporary,
                &final_path,
            )
            .expect("cancelled publication decision");
        prop_assert_eq!(publication, degradation::OutputPublication::Cancelled);
        prop_assert!(!temporary.exists());
        prop_assert!(!final_path.exists());

        // Cancellation is allowed to discard only this run's temporary
        // output.  Existing Source_RAW bytes and Virtual_Tile cache entries
        // must remain byte-for-byte unchanged.
        let source_after = virtual_tile::source_file_sha256(&source_path)
            .expect("source digest after cancellation");
        let cache_after = std::fs::read(&cache_path).expect("cache bytes after cancellation");
        prop_assert_eq!(source_after, source_before);
        prop_assert_eq!(cache_after, cache_before);

        let recorder = super::report::StackReportRecorder::isolated("property-91", None);
        recorder.set_result(super::report::StackRunResult::Cancelled);
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();
        prop_assert_eq!(
            report.degradation.result,
            super::report::StackRunResult::Cancelled
        );
        prop_assert!(report
            .degradation
            .entries
            .iter()
            .any(|entry| entry.reason == degradation::RUN_CANCELLED_BY_USER));
    }

    // Feature: layered-camera-group-focus-stitching, Property 93: Stack_Report schema 完整且与返回值一致
    // 任意生产 Stack_Report 都包含固定顶层字段，并可通过实际序列化路径往返。
    //
    // **Validates: Requirements 1.7, 2.9, 5.9, 6.10, 11.16**
    #[test]
    fn property_93_stack_report_schema_matches_serialized_value(mut report in arb_stack_report()) {
        report.pipeline_version = STACK_PIPELINE_VERSION.to_string();
        let encoded = report.to_json_string().expect("stack report serializes");
        let value: Value = serde_json::from_str(&encoded).expect("stack report json");
        for field in STACK_REPORT_TOP_LEVEL_FIELDS {
            prop_assert!(value.get(*field).is_some(), "missing Stack_Report field {field}");
        }
        let station_relations = value
            .get("station_relations")
            .and_then(Value::as_object)
            .expect("station_relations schema object");
        prop_assert!(station_relations.get("unmeasured_count").is_some());
        prop_assert!(station_relations.get("rejected_by_failure_stage").is_some());
        if let Some(candidate) = station_relations
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
        {
            prop_assert!(candidate.get("predicted_overlap_area_px").is_some());
            prop_assert!(candidate.get("model_fitted").is_some());
        }
        let round_trip: StackReport = serde_json::from_str(&encoded).expect("schema round trip");
        // JSON is the wire contract.  Comparing the re-serialized value avoids
        // treating a one-ulp f64 parse/print round trip as a schema mismatch.
        prop_assert_eq!(serde_json::to_value(round_trip).expect("round-trip value"), value);
    }
}
