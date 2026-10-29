//! Residual_Warp and Tone_Harmonizer properties (41-52).
//!
//! The first proptest! block holds Property 44 (one composed tile-to-world
//! mapping) together with the tone properties 48-52; the test before it counts
//! the layered compositor samples behind Property 44 at run time.

use super::super::compositor;
use super::super::degradation;
use super::super::report::ToneStatus;
use super::super::tone;
use crate::panorama_utils::stitching;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;
use proptest::prelude::*;
use std::collections::HashMap;

use super::{render_layered_ownership, single_frame_info, single_frame_source};

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
        let overlap = crate::panorama_utils::stack_pipeline::residual_warp::OverlapResidual::new(
            0,
            7,
            crate::panorama_utils::stack_pipeline::report::WorldRect {
                left: 0.0,
                top: 0.0,
                width: 512.0,
                height: 512.0,
            },
            4.0,
        );
        let observations = (0..=8)
            .flat_map(|row| {
                (0..=8).map(move |column| crate::panorama_utils::stack_pipeline::residual_warp::WarpObservation::new(
                    crate::panorama_utils::stack_pipeline::report::WorldPoint {
                        x: f64::from(column) * 64.0,
                        y: f64::from(row) * 64.0,
                    },
                    [warp_tx, warp_ty],
                    0.0,
                ))
            })
            .collect::<Vec<_>>();
        let mut residual = crate::panorama_utils::stack_pipeline::residual_warp::ResidualWarp::new();
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
            native_detail: None,
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
                native_detail: None,
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
            native_detail: None,
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
        if delta > crate::panorama_utils::stack_pipeline::report::TONE_BOUNDARY_DELTA_E_THRESHOLD {
            prop_assert!(
                degradation::run_ledger_snapshot()
                    .entries()
                    .iter()
                    .any(|entry| entry.reason == degradation::TONE_BOUNDARY_DELTA_E_EXCEEDED)
            );
        } else {
            prop_assert!(delta <= crate::panorama_utils::stack_pipeline::report::TONE_BOUNDARY_DELTA_E_THRESHOLD);
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
            native_detail: None,
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

// Stage 7 backfill: task 11.8 (Property 41).
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 41: 局部形变只在 P95 > 3 的区域启用，64px 网格每方向至少四节点。
    // **Validates: Requirements 8.1, 8.2**
    #[test]
    fn property_41_residual_regions_exactly_match_global_p95_gate(
        origin in (-4096i32..4096, -4096i32..4096),
        overlaps in prop::collection::vec(
            (
                1u16..1025,
                1u16..1025,
                prop_oneof![Just(3.0f64), Just(3.0f64 + f64::EPSILON * 2.0), 0.0f64..12.0],
            ),
            0..16,
        ),
    ) {
        use crate::panorama_utils::stack_pipeline::report::{WorldPoint, WorldRect};
        use crate::panorama_utils::stack_pipeline::residual_warp::{OverlapResidual, ResidualWarp};

        let mut model = ResidualWarp::default();
        prop_assert!(model.regions.is_empty());
        prop_assert!(model.identity());
        let mut expected = Vec::new();
        for (index, &(width, height, p95)) in overlaps.iter().enumerate() {
            let world = WorldRect {
                left: origin.0 as f64 + index as f64 * 2048.0,
                top: origin.1 as f64,
                width: f64::from(width),
                height: f64::from(height),
            };
            let overlap = OverlapResidual::new(index * 2, index * 2 + 1, world, p95);
            let inserted = model.add_overlap(overlap);
            if p95 > 3.0 {
                prop_assert_eq!(inserted, Some(expected.len()));
                expected.push(overlap);
            } else {
                prop_assert_eq!(inserted, None);
                let mut measured = ResidualWarp::default();
                prop_assert_eq!(measured.add_observations(overlap, &[]), None);
                prop_assert!(measured.regions.is_empty());
            }
        }
        prop_assert_eq!(model.regions.len(), expected.len());
        for (region, overlap) in model.regions.iter().zip(&expected) {
            prop_assert_eq!(region.world, overlap.world);
            prop_assert_eq!((region.left_station, region.right_station), (overlap.left_station, overlap.right_station));
            prop_assert_eq!(region.p95_before_px, overlap.p95_px);
            prop_assert_eq!(region.node_step_px, 64);
            prop_assert_eq!(region.columns, ((overlap.world.width / 64.0).ceil() as u32 + 1).max(4));
            prop_assert_eq!(region.rows, ((overlap.world.height / 64.0).ceil() as u32 + 1).max(4));
            prop_assert_eq!(region.node_count(), region.columns as usize * region.rows as usize);
            for (index, node) in region.nodes().iter().enumerate() {
                let column = index % region.columns as usize;
                let row = index / region.columns as usize;
                prop_assert_eq!(node.world, WorldPoint {
                    x: overlap.world.left + column as f64 * 64.0,
                    y: overlap.world.top + row as f64 * 64.0,
                });
                prop_assert_eq!(node.displacement, [0.0, 0.0]);
            }
        }
        // Merely allocating an enabled region must not invent a deformation.
        prop_assert!(model.identity());
        for (station, point) in [
            (usize::MAX, Point2::new(-0.0, origin.1 as f64)),
            (0, Point2::new(origin.0 as f64 - 1.0, origin.1 as f64)),
        ] {
            let mapped = model.warp_inverse(point, station);
            prop_assert_eq!(mapped.x.to_bits(), point.x.to_bits());
            prop_assert_eq!(mapped.y.to_bits(), point.y.to_bits());
        }
    }
}

// Stage 7 backfill snippets for tasks 11.9–11.14. Append each `proptest!`
// block independently to properties.rs after task 11.8 is committed.

// Task 11.9 / Property 42.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 42: 写入的节点位移同时满足位移、邻接差与往返误差三项约束。
    // **Validates: Requirements 8.3, 8.4, 8.5**
    #[test]
    fn property_42_residual_nodes_satisfy_all_three_gates(
        displacement in (-8.0f64..8.0, -8.0f64..8.0),
        round_trip in 0.0f64..=1.0,
    ) {
        use crate::panorama_utils::stack_pipeline::report::WorldRect;
        use crate::panorama_utils::stack_pipeline::residual_warp::{OverlapResidual, WarpObservation};
        let overlap = OverlapResidual::new(1, 2, WorldRect { left: 0.0, top: 0.0, width: 192.0, height: 192.0 }, 4.0);
        let observations = (0..4).flat_map(|row| (0..4).map(move |column| WarpObservation::new(
            crate::panorama_utils::stack_pipeline::report::WorldPoint { x: column as f64 * 64.0, y: row as f64 * 64.0 },
            [displacement.0, displacement.1], round_trip,
        ))).collect::<Vec<_>>();
        let region = crate::panorama_utils::stack_pipeline::residual_warp::WarpRegion::from_observations(overlap, &observations);
        if region.insufficient_evidence {
            prop_assert!(region.is_identity());
        }
        for node in region.nodes() {
            prop_assert!(node.displacement[0].hypot(node.displacement[1]) <= crate::panorama_utils::stack_pipeline::residual_warp::RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX + 1e-9);
            if node.valid {
                prop_assert!(node.round_trip_error_px <= crate::panorama_utils::stack_pipeline::residual_warp::RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX + 1e-9);
            }
        }
        for row in 0..region.rows as usize {
            for column in 0..region.columns as usize {
                let index = row * region.columns as usize + column;
                let mut neighbours = Vec::with_capacity(2);
                if row > 0 {
                    neighbours.push((row - 1) * region.columns as usize + column);
                }
                if column > 0 {
                    neighbours.push(row * region.columns as usize + column - 1);
                }
                for other in neighbours {
                    let delta = (region.nodes()[index].displacement[0] - region.nodes()[other].displacement[0])
                        .hypot(region.nodes()[index].displacement[1] - region.nodes()[other].displacement[1]);
                    prop_assert!(delta <= crate::panorama_utils::stack_pipeline::residual_warp::RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX + 1e-7);
                }
            }
        }
    }
}

// Task 11.10 / Property 43.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 43: 局部形变不优于全局单应性的网格单元回退且计数准确。
    // **Validates: Requirements 8.6**
    #[test]
    fn property_43_residual_cell_reverts_match_error_ordering(
        global in prop::collection::vec(0.0f64..20.0, 9),
        residual in prop::collection::vec(0.0f64..20.0, 9),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_utils::stack_pipeline::report::WorldRect;
        use crate::panorama_utils::stack_pipeline::residual_warp::{OverlapResidual, WarpObservation, WarpRegion};
        let overlap = OverlapResidual::new(0, 1, WorldRect { left: 0.0, top: 0.0, width: 192.0, height: 192.0 }, 4.0);
        let mut region = WarpRegion::from_observations(overlap, &(0..16).map(|index| WarpObservation::new(
            crate::panorama_utils::stack_pipeline::report::WorldPoint { x: (index % 4) as f64 * 64.0, y: (index / 4) as f64 * 64.0 }, [1.0, 0.0], 0.0,
        )).collect::<Vec<_>>());
        region.revert_cells(&global, &residual);
        let expected = global.iter().zip(&residual).filter(|(g, r)| **r >= **g).count();
        prop_assert_eq!(region.reverted_cells as usize, expected);
    }
}

// Task 11.12 / Property 45.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 45: 无效节点只从三节点半径内有效邻域外推，无邻域时严格为零。
    // **Validates: Requirements 8.8**
    #[test]
    fn property_45_invalid_nodes_use_only_three_node_valid_support(
        value in 0.0f64..8.0,
    ) {
        use crate::panorama_utils::stack_pipeline::report::{WorldPoint, WorldRect};
        use crate::panorama_utils::stack_pipeline::residual_warp::{OverlapResidual, WarpObservation, WarpRegion};
        let overlap = OverlapResidual::new(0, 1, WorldRect { left: 0.0, top: 0.0, width: 704.0, height: 704.0 }, 4.0);
        let observations = (0..4).flat_map(|row| (0..4).map(move |column| WarpObservation::new(
            WorldPoint { x: column as f64 * 64.0, y: row as f64 * 64.0 }, [value, 0.0], 0.0,
        ))).collect::<Vec<_>>();
        let region = WarpRegion::from_observations(overlap, &observations);
        for node in region.nodes().iter().filter(|node| !node.valid) {
            let supporting = region.nodes().iter().filter(|candidate| candidate.valid)
                .map(|candidate| (candidate.world.x - node.world.x).hypot(candidate.world.y - node.world.y) / 64.0)
                .fold(f64::INFINITY, f64::min);
            if supporting > crate::panorama_utils::stack_pipeline::residual_warp::RESIDUAL_WARP_EXTRAPOLATION_RADIUS_NODES {
                prop_assert_eq!(node.displacement, [0.0, 0.0]);
            } else if node.displacement != [0.0, 0.0] {
                prop_assert!(supporting <= crate::panorama_utils::stack_pipeline::residual_warp::RESIDUAL_WARP_EXTRAPOLATION_RADIUS_NODES);
            }
        }
    }
}

// Task 11.13 / Property 46.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 46: 形变沿边界法向在128px内单调衰减为零，区域外恒等。
    // **Validates: Requirements 8.9**
    #[test]
    fn property_46_residual_edge_fade_is_monotonic(
        displacement in 1.0f64..32.0,
    ) {
        use crate::panorama_utils::stack_pipeline::report::{WorldPoint, WorldRect};
        use crate::panorama_utils::stack_pipeline::residual_warp::{OverlapResidual, WarpObservation, WarpRegion};
        let overlap = OverlapResidual::new(0, 1, WorldRect { left: 0.0, top: 0.0, width: 1024.0, height: 1024.0 }, 4.0);
        let observations = (0..17).flat_map(|row| (0..17).map(move |column| WarpObservation::new(
            WorldPoint { x: column as f64 * 64.0, y: row as f64 * 64.0 }, [displacement, 0.0], 0.0,
        ))).collect::<Vec<_>>();
        let region = WarpRegion::from_observations(overlap, &observations);
        let mut previous = 0.0;
        for distance in (0..=128).step_by(8) {
            let current = region.displacement_at(distance as f64, 512.0)[0];
            prop_assert!(current + 1e-8 >= previous);
            previous = current;
        }
        prop_assert_eq!(region.displacement_at(-1.0, 512.0), [0.0, 0.0]);
        prop_assert_eq!(region.displacement_at(1025.0, 512.0), [0.0, 0.0]);
    }
}

// Task 11.14 / Property 47.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 47: 证据不足时区域保持恒等并记录不足证据。
    // **Validates: Requirements 8.10**
    #[test]
    fn property_47_insufficient_residual_evidence_is_identity(
        observations in prop::collection::vec(
            (0.0f64..192.0, 0.0f64..192.0, -8.0f64..8.0, -8.0f64..8.0),
            0..16,
        ),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_utils::stack_pipeline::report::{WorldPoint, WorldRect};
        use crate::panorama_utils::stack_pipeline::residual_warp::{OverlapResidual, WarpObservation, WarpRegion, RESIDUAL_WARP_MIN_VERIFIED_POINTS};
        let overlap = OverlapResidual::new(0, 1, WorldRect { left: 0.0, top: 0.0, width: 192.0, height: 192.0 }, 3.01);
        let samples = observations.iter().map(|&(x, y, dx, dy)| WarpObservation::new(WorldPoint { x, y }, [dx, dy], 1.0)).collect::<Vec<_>>();
        let region = WarpRegion::from_observations(overlap, &samples);
        prop_assert!(samples.len() < RESIDUAL_WARP_MIN_VERIFIED_POINTS);
        prop_assert!(region.insufficient_evidence);
        prop_assert!(region.is_identity());
        for node in region.nodes() {
            prop_assert_eq!(node.displacement, [0.0, 0.0]);
        }
    }
}
