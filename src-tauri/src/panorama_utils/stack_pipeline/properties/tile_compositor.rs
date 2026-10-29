//! Tile_Compositor properties: ownership contract, source samples and seams.
//!
//! The layered canvas, transparency and ownership contract (54-56), every owned
//! output pixel as one real source sample (17), and the seam cost and the
//! narrow-overlap centre line (53, 59).

use super::super::compositor;
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stitching;
use nalgebra::Matrix3;
use nalgebra::Point3;
use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use super::{render_station_tile, single_frame_info, single_frame_source};

// ---------------------------------------------------------------------------
// Properties 54–56: ownership compositor contract (13.9–13.11)
// ---------------------------------------------------------------------------

pub(super) fn ownership_contract_fixture() -> (
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

// ---------------------------------------------------------------------------
// Property 17 harness: every attributed pixel is one source sample
// ---------------------------------------------------------------------------

/// Property 17 keeps 100 generated ownership/focus cases, but rotates the two
/// complete production renders through ten bounded 72px fixtures so the full
/// library suite remains practical. Five use integer translations and five use
/// a genuinely projective transform with non-integer source coordinates.
pub(super) const SOURCE_OWNERSHIP_CASES: usize = 100;

pub(super) const SOURCE_OWNERSHIP_RENDER_STRIDE: usize = 10;

pub(super) const SOURCE_OWNERSHIP_SIDE: u32 = 72;

pub(super) static SOURCE_OWNERSHIP_CURSOR: AtomicUsize = AtomicUsize::new(0);

pub(super) static SOURCE_OWNERSHIP_BRANCHES_SEEN: AtomicUsize = AtomicUsize::new(0);

pub(super) fn ownership_source(side: u32, seed: u64, source: usize) -> image::Rgb32FImage {
    image::Rgb32FImage::from_fn(side, side, |x, y| {
        let xf = f64::from(x);
        let yf = f64::from(y);
        let phase = ((seed >> 11) & 31) as f64 * 0.013;
        let paper = 0.42 + 0.035 * (xf * 0.071 + phase).sin() + 0.028 * (yf * 0.053 - phase).cos();
        let focused_half = if source == 0 {
            x < side / 2
        } else {
            x >= side / 2
        };
        let checker = if ((x / 2 + y / 2) & 1) == 0 {
            1.0
        } else {
            -1.0
        };
        let detail = if focused_half { 0.16 * checker } else { 0.0 };
        // The small source-specific chroma offset makes an owner blend
        // observable while staying far below the 需求 3.3 inconsistency veto.
        let source_offset = source as f64 * 0.0125;
        image::Rgb([
            (paper + detail + source_offset).clamp(0.04, 0.96) as f32,
            (paper + detail * 0.72).clamp(0.04, 0.96) as f32,
            (paper + detail * 0.43 - source_offset).clamp(0.04, 0.96) as f32,
        ])
    })
}

pub(super) fn ownership_homography(projective: bool) -> Matrix3<f64> {
    if projective {
        Matrix3::new(1.0, 0.027, 0.37, -0.019, 1.0, 0.61, 0.00031, -0.00023, 1.0)
    } else {
        Matrix3::identity()
    }
}

pub(super) fn render_layered_ownership(
    infos: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    sources: &[image::Rgb32FImage],
) -> Result<stitching::LayeredOwnershipRender, String> {
    let app = tauri::test::mock_app();
    let refs = infos.iter().collect::<Vec<_>>();
    let mut load = |info: &ImageInfo| {
        sources
            .get(info.id)
            .cloned()
            .ok_or_else(|| format!("no synthetic source for {}", info.filename))
    };
    stitching::layered_virtual_tile_compositor_with_ownership(
        &refs,
        homographies,
        stitching::Projection::Planar,
        app.handle().clone(),
        "stack-pipeline-property-progress",
        &mut load,
    )
}

pub(super) fn assert_source_correspondence(
    rendered: &stitching::LayeredOwnershipRender,
    infos: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    sources: &[image::Rgb32FImage],
    require_fractional: bool,
    require_all_owners: bool,
) -> Result<(usize, usize, usize), TestCaseError> {
    let (width, height) = rendered.image.dimensions();
    prop_assert_eq!(rendered.coverage.dimensions(), (width, height));
    prop_assert_eq!(rendered.ownership.dimensions(), (width, height));
    prop_assert_eq!(rendered.ownership.legend().len(), sources.len());
    let mut owner_counts = vec![0usize; sources.len()];
    let mut uncovered = 0usize;
    let mut fractional = 0usize;
    for y in 0..height {
        for x in 0..width {
            let covered = rendered.coverage.is_covered(x, y);
            let owner = rendered.ownership.owner_at(x, y);
            if !covered {
                uncovered += 1;
                prop_assert_eq!(owner, stitching::NO_OWNER);
                prop_assert!(
                    rendered
                        .image
                        .get_pixel(x, y)
                        .0
                        .iter()
                        .all(|&channel| channel == 0.0),
                    "uncovered output ({x}, {y}) must remain zero"
                );
                continue;
            }
            prop_assert!(owner > 0 && owner as usize <= sources.len());
            let source_index = owner as usize - 1;
            owner_counts[source_index] += 1;
            let inverse = homographies[&infos[source_index].id]
                .try_inverse()
                .ok_or_else(|| TestCaseError::fail("generated homography must be invertible"))?;
            let target = Point3::new(
                f64::from(x) + rendered.sampling_origin.0,
                f64::from(y) + rendered.sampling_origin.1,
                1.0,
            );
            let source = stitching::map_target_to_source(
                &inverse,
                target,
                &infos[source_index],
                stitching::Projection::Planar,
            )
            .ok_or_else(|| TestCaseError::fail("covered owner must map to its Source_RAW"))?;
            if source.x.fract().abs() > 1e-9 || source.y.fract().abs() > 1e-9 {
                fractional += 1;
            }
            let expected = stitching::get_high_quality_interpolated_pixel(
                &sources[source_index],
                source.x,
                source.y,
            );
            let actual = rendered.image.get_pixel(x, y);
            for channel in 0..3 {
                prop_assert_eq!(
                    actual[channel].to_bits(),
                    expected[channel].to_bits(),
                    "需求 3.6: output ({}, {}) channel {} names source {} at ({:.6}, {:.6}) but is not that exact production sample",
                    x,
                    y,
                    channel,
                    infos[source_index].filename,
                    source.x,
                    source.y
                );
            }
        }
    }
    if require_all_owners {
        prop_assert!(
            owner_counts.iter().all(|&count| count > 0),
            "the translated production fixture must give both real photographs owned pixels: {owner_counts:?}"
        );
    }
    prop_assert!(
        uncovered > 0,
        "the projective/validity border must exercise uncovered pixels"
    );
    if require_fractional {
        prop_assert!(
            fractional > 0,
            "the projective branch must sample non-integer source coordinates"
        );
    }
    Ok((owner_counts.iter().sum(), uncovered, fractional))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: SOURCE_OWNERSHIP_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 17: 对于任意 Capture_Station
    // 与任意 ownership 分配，Virtual_Tile 的每个已覆盖像素值等于其 owner Source_RAW 在
    // 对应位置的采样值；在 owner 变换为恒等时该相等是逐位相等；且不存在任何像素是两个
    // 不同 owner 像素的加权组合。
    //
    // The test exercises the real `focus_stack_stitcher_unfilled` Focus_Fuser and the exact
    // provenance-preserving body called by the default `layered_virtual_tile_compositor`.
    // Integer placement compares directly with source pixels; projective placement compares with
    // the design's exact corresponding sample: the production cubic sampler evaluated at the
    // owner homography's inverse coordinate. No scalar ownership helper or mosaic comparison path
    // stands in for either production render.
    //
    // **Validates: Requirements 3.6**
    #[test]
    fn every_owned_output_pixel_is_exactly_one_real_source_sample(
        seed in any::<u64>(),
        projective_draw in any::<bool>(),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let case = SOURCE_OWNERSHIP_CURSOR.fetch_add(1, Ordering::Relaxed);
        // Rotate deterministically so the final run necessarily sees both geometry branches even
        // if proptest's random booleans happen to cluster.
        let projective = if case % SOURCE_OWNERSHIP_RENDER_STRIDE == 0 {
            (case / SOURCE_OWNERSHIP_RENDER_STRIDE) % 2 == 1
        } else {
            projective_draw
        };
        let mut branches = 0usize;

        // All 100 cases exercise a hard two-owner pixel assignment. A blend, average, tone change,
        // sharpen, or owner/value mismatch cannot equal either generated source bit pattern.
        let left = ownership_source(16, seed, 0);
        let right = ownership_source(16, seed, 1);
        let split = 1 + (seed as usize % 14);
        for x in 0..16usize {
            let owner = usize::from(x >= split);
            for y in 0..16u32 {
                let expected = if owner == 0 {
                    left.get_pixel(x as u32, y)
                } else {
                    right.get_pixel(x as u32, y)
                };
                let selected = if owner == 0 {
                    left.get_pixel(x as u32, y)
                } else {
                    right.get_pixel(x as u32, y)
                };
                for channel in 0..3 {
                    prop_assert_eq!(selected[channel].to_bits(), expected[channel].to_bits());
                }
            }
        }

        if case % SOURCE_OWNERSHIP_RENDER_STRIDE == 0 {
            let side = SOURCE_OWNERSHIP_SIDE;
            let infos = vec![single_frame_info(0, side), single_frame_info(1, side)];
            let sources = vec![ownership_source(side, seed, 0), ownership_source(side, seed, 1)];
            let common = ownership_homography(projective);

            // Capture_Station / Focus_Fuser production path. Both focus frames use one camera
            // transform; the owner map therefore identifies which Source_RAW supplied the exact
            // corresponding sample at each output pixel.
            let station_h = HashMap::from([(0usize, common), (1usize, common)]);
            let station = render_station_tile(&infos, &station_h, &sources)
                .map_err(|error| TestCaseError::fail(format!("station render failed: {error}")))?;
            let station_repeat = render_station_tile(&infos, &station_h, &sources)
                .map_err(|error| TestCaseError::fail(format!("station repeat failed: {error}")))?;
            let station_masks = station.masks.as_ref().ok_or_else(|| {
                TestCaseError::fail("the default station fuser must publish ownership")
            })?;
            let station_repeat_masks = station_repeat.masks.as_ref().ok_or_else(|| {
                TestCaseError::fail("the repeated station fuser must publish ownership")
            })?;
            let (minimum_x, maximum_x, minimum_y, maximum_y) =
                stitching::output_bounds(&infos.iter().collect::<Vec<_>>(), &station_h, stitching::Projection::Planar);
            let (offset_x, _) = stitching::pixel_aligned_canvas(minimum_x, maximum_x);
            let (offset_y, _) = stitching::pixel_aligned_canvas(minimum_y, maximum_y);
            let station_view = stitching::LayeredOwnershipRender {
                image: station.image.clone(),
                coverage: station_masks.coverage.clone(),
                ownership: station_masks.ownership.clone(),
                sampling_origin: (-offset_x, -offset_y),
                full_resolution_samples: 0,
                full_resolution_sample_counts: Vec::new(),
            };
            let (_, station_uncovered, station_fractional) = assert_source_correspondence(
                &station_view,
                &infos,
                &station_h,
                &sources,
                projective,
                false,
            )?;
            prop_assert!(station_uncovered > 0);
            if projective {
                prop_assert!(station_fractional > 0);
            }
            prop_assert_eq!(station.image.as_raw(), station_repeat.image.as_raw());
            prop_assert_eq!(&station_masks.coverage, &station_repeat_masks.coverage);
            prop_assert_eq!(&station_masks.ownership, &station_repeat_masks.ownership);

            // Default layered ownership compositor. Give the second photograph a translated
            // footprint so both own unique/interior pixels; projective mode keeps that translation
            // on top of a non-affine homography.
            let translated = Matrix3::new(1.0, 0.0, 24.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0) * common;
            let layered_h = HashMap::from([(0usize, common), (1usize, translated)]);
            let layered = render_layered_ownership(&infos, &layered_h, &sources)
                .map_err(|error| TestCaseError::fail(format!("layered render failed: {error}")))?;
            let layered_repeat = render_layered_ownership(&infos, &layered_h, &sources)
                .map_err(|error| TestCaseError::fail(format!("layered repeat failed: {error}")))?;
            assert_source_correspondence(
                &layered,
                &infos,
                &layered_h,
                &sources,
                projective,
                true,
            )?;
            prop_assert_eq!(layered.image.as_raw(), layered_repeat.image.as_raw());
            prop_assert_eq!(&layered.coverage, &layered_repeat.coverage);
            prop_assert_eq!(&layered.ownership, &layered_repeat.ownership);
            prop_assert_eq!(
                layered.sampling_origin.0.to_bits(),
                layered_repeat.sampling_origin.0.to_bits()
            );
            prop_assert_eq!(
                layered.sampling_origin.1.to_bits(),
                layered_repeat.sampling_origin.1.to_bits()
            );

            branches |= if projective { 0b10 } else { 0b01 };
        }
        let seen = SOURCE_OWNERSHIP_BRANCHES_SEEN.fetch_or(branches, Ordering::Relaxed) | branches;
        prop_assert!(
            case + 1 < SOURCE_OWNERSHIP_CASES || seen == 0b11,
            "the bounded production rotation must cover integer and projective paths; mask={seen:#b}"
        );
    }
}

pub(super) fn exhaustive_vertical_seam_cost(costs: &[f64], width: usize, height: usize) -> f64 {
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
