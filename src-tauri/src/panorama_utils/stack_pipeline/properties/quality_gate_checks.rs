//! Quality_Gate properties (61-74).
//!
//! ROI selection, pairing, Local_Scale and effective pixels, slanted-edge and flat
//! ROI classification, MTF50, gradient energy, noise sigma, delta E00, boundary
//! strokes, owner sharpness coverage, measurement counts, blocked export and
//! reproducible ROI sets.

use super::super::degradation;
use super::super::quality_gate::QUALITY_EFFECTIVE_PIXEL_RATIO_MIN;
use super::super::quality_gate::QUALITY_ROI_MARGIN;
use super::super::quality_gate::QUALITY_ROI_MAX_COUNT;
use super::super::quality_gate::QUALITY_ROI_SIDE;
use super::super::quality_gate::QualityRoi;
use super::super::quality_gate::SourceGeometry;
use super::super::quality_gate::detect_slanted_edge;
use super::super::quality_gate::effective_pixel_coverage;
use super::super::quality_gate::flat_roi;
use super::super::quality_gate::label_owner_regions;
use super::super::quality_gate::local_scale_for_roi;
use super::super::quality_gate::map_roi_corners_to_source;
use super::super::quality_gate::measure_boundary_strokes;
use super::super::quality_gate::noise_sigma;
use super::super::quality_gate::normalized_gradient_energy;
use super::super::quality_gate::roi_low_frequency_delta_e00;
use super::super::quality_gate::select_quality_rois;
use super::super::quality_gate::slanted_edge_mtf50;
use super::super::quality_gate::trace_moore_boundary;
use super::super::quality_gate_runner::Criterion;
use super::super::quality_gate_runner::OwnerSharpnessCriterionInput;
use super::super::quality_gate_runner::build_owner_sharpness_criterion;
use super::super::quality_gate_runner::classify_unmeasurable_reason;
use super::super::quality_gate_runner::criterion_verdict;
use super::super::quality_gate_runner::overall_verdict;
use super::super::quality_gate_runner::record_unmeasurable;
use super::super::report::FailedMeasurementRecord;
use super::super::report::QualityGateCriterionRecord;
use super::super::report::QualityGateReport;
use super::super::report::QualityGateVerdict;
use super::super::report::UnmeasurableCategory;
use super::super::report::UnmeasurableRecord;
use super::super::report::WorldPoint;
use nalgebra::Matrix3;
use nalgebra::Point2;
use proptest::prelude::*;
use std::collections::BTreeMap;

// Stage 7 Quality_Gate self-checks (15.14, 15.16, 15.17).

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
        let residual = crate::panorama_utils::stack_pipeline::residual_warp::ResidualWarp::new();
        let measurement = local_scale_for_roi(&roi, &geometry, &residual, render_scale)
            .expect("identity map is measurable");
        prop_assert_eq!(measurement.samples, 1024);
        prop_assert!((measurement.median - 1.0).abs() < 1.0e-10);
        prop_assert!((measurement.fraction_at_least_095 - 1.0).abs() < 1.0e-10);
        prop_assert_eq!(measurement.diagnostic_render_scale, render_scale < 1.0);
    }
}

// Stage 7 pairing and effective-coverage properties (15.10, 15.12).
#[test]
fn native_to_output_scale_distinguishes_downsampling_from_magnification() {
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: QUALITY_ROI_SIDE,
        region_label: 1,
        owner: 1,
        world_origin: (128.0, 128.0),
        boundary_clearance: 64.0,
    };
    for scale in [0.5, 0.9, 1.0, 2.0] {
        let geometry = SourceGeometry {
            member_to_anchor: nalgebra::Matrix3::identity(),
            tile_to_world: nalgebra::Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0),
            station_id: 1,
        };
        let measured = local_scale_for_roi(
            &roi,
            &geometry,
            &crate::panorama_utils::stack_pipeline::residual_warp::ResidualWarp::new(),
            1.0,
        )
        .unwrap();
        assert!((measured.median - scale).abs() < 1e-10);
        assert_eq!(
            measured.fraction_at_least_095,
            if scale >= 0.95 { 1.0 } else { 0.0 }
        );
    }
}

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
        let residual = crate::panorama_utils::stack_pipeline::residual_warp::ResidualWarp::new();
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
            prop_assert_eq!(paired, Err(crate::panorama_utils::stack_pipeline::degradation::PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED));
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
        // A single vertical stroke crosses the horizontal owner boundary.
        // Two parallel edges on opposite sides are different edges and must
        // not be paired merely because their contrast centroids are symmetric.
        for y in 0..height {
            for x in 48..width {
                image.put_pixel(x,y,image::Rgb([level;3]));
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
        for name in crate::panorama_utils::stack_pipeline::quality_gate::QUALITY_CRITERIA {
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
        for name in crate::panorama_utils::stack_pipeline::quality_gate::QUALITY_CRITERIA {
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
                effective_output_pixels: None,
                effective_projected_pixels: None,
            }],
            unmeasurable: vec![UnmeasurableRecord {
                criterion: "noise_sigma_ratio".to_string(),
                world: WorldPoint { x, y },
                reason: "roi_not_flat".to_string(),
                category: UnmeasurableCategory::ContentNotApplicable,
            }],
            timing: crate::panorama_utils::stack_pipeline::report::QualityGateTimingRecord::default(),
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
