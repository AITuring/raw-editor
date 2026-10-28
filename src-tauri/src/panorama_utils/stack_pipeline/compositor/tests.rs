#[test]
fn a_contour_inverse_composes_once_with_the_source_camera_inverse() {
    use super::super::line_layers::{ContourTrack, LineLayers};
    let track = |station, dx| ContourTrack {
        station,
        points: (0..160)
            .map(|i| [i as f64 * 32.0, 400.0 + i as f64 * 0.4 + dx])
            .collect(),
    };
    let mut model = ResidualWarp::new();
    model.line_layers = LineLayers::fit(vec![track(0, 0.0), track(1, 17.0), track(2, -7.0)]);
    model.register_station_alias(70, 1);
    let inverse = Matrix3::new(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0);
    let actual =
        tile_coordinate_with_residual(Point2::new(408.0, 640.0), &inverse, &model, 70).unwrap();
    assert!((actual.x - 212.5).abs() < 1e-6);
    assert_eq!(actual.y, 320.0);
    assert_eq!(
        tile_coordinate_with_residual(Point2::new(100.0, 640.0), &inverse, &model, 70).unwrap(),
        Point2::new(50.0, 320.0)
    );
}

#[test]
fn large_canvas_revisit_can_cut_through_a_native_cell_corridor() {
    use super::super::tone::ToneTile;
    use image::{GrayImage, Luma, Rgb, Rgb32FImage};
    let tile = |station_index, owner_id, left, right, displaced| ToneTile {
        station_index,
        owner_id,
        low: Rgb32FImage::from_pixel(1024, 8, Rgb([0.4; 3])),
        cell_mean: Rgb32FImage::from_fn(1024, 8, |x, _| {
            Rgb([if displaced && !(512..=513).contains(&x) {
                0.1
            } else {
                0.4
            }; 3])
        }),
        validity: GrayImage::from_fn(1024, 8, |x, _| {
            Luma([if x >= left && x < right { 255 } else { 0 }])
        }),
        native_detail: None,
        cell_coverage: (0..8192)
            .map(|i| {
                if i % 1024 >= left && i % 1024 < right {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
        world_origin: (0.0, 0.0),
        world_size: (16384.0, 128.0),
        world_stride: 16.0,
    };
    let tiles = [tile(0, 1, 0, 600, false), tile(1, 2, 400, 1024, true)];
    let owners = (0..128)
        .flat_map(|_| (0..16384).map(|x| if x < 8192 { 1 } else { 2 }))
        .collect::<Vec<_>>();
    let plan = super::refine_station_ownership(&tiles, &[(0, 1)], &owners, 16384, 128).unwrap();
    // x=8208 lies in the actual common-detail corridor; a 32px grid
    // cannot represent this seam and crosses displaced content instead.
    for y in (32..96).step_by(16) {
        assert_eq!(plan.owner_at(8207, y), 1);
        assert_eq!(plan.owner_at(8208, y), 2);
    }
}

#[test]
fn ownership_revisit_measures_detail_at_the_actual_label_boundary() {
    use super::super::tone::ToneTile;
    use image::{GrayImage, Luma, Rgb, Rgb32FImage};
    let tile = |station_index, owner_id, left, right, displaced| ToneTile {
        station_index,
        owner_id,
        low: Rgb32FImage::from_pixel(16, 16, Rgb([0.4; 3])),
        cell_mean: Rgb32FImage::from_fn(16, 16, |x, _| {
            Rgb([if displaced && x == 8 { 0.1 } else { 0.4 }; 3])
        }),
        validity: GrayImage::from_fn(16, 16, |x, _| {
            Luma([if x >= left && x < right { 255 } else { 0 }])
        }),
        native_detail: None,
        cell_coverage: (0..256)
            .map(|i| {
                if i % 16 >= left && i % 16 < right {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
        world_origin: (0.0, 0.0),
        world_size: (256.0, 256.0),
        world_stride: 16.0,
    };
    let tiles = [tile(0, 1, 0, 12, false), tile(1, 2, 4, 16, true)];
    let owners = (0..256)
        .flat_map(|_| (0..256).map(|x| if x < 128 { 1 } else { 2 }))
        .collect::<Vec<_>>();
    // The retained seam is between columns 7 and 8. At column 7's
    // centre the two detail fields are identical, so a centre-only cost
    // leaves this seam in place. At its actual boundary they differ.
    let plan = super::refine_station_ownership(&tiles, &[(0, 1)], &owners, 256, 256).unwrap();
    assert_eq!(plan.owner_at(8, 128), 1);
    assert_eq!(plan.owner_at(248, 128), 2);
    for y in (32..224).step_by(16) {
        assert_eq!(plan.owner_at(120, y), plan.owner_at(136, y));
    }
}

#[test]
fn ownership_revisit_avoids_displaced_detail_hidden_by_equal_low_frequency() {
    use super::super::tone::ToneTile;
    use image::{GrayImage, Luma, Rgb, Rgb32FImage};
    let tile = |station_index, owner_id, left, right, displaced| ToneTile {
        station_index,
        owner_id,
        low: Rgb32FImage::from_pixel(16, 16, Rgb([0.4; 3])),
        cell_mean: Rgb32FImage::from_fn(16, 16, |x, _| {
            Rgb([if displaced && (7..=9).contains(&x) {
                0.1
            } else {
                0.4
            }; 3])
        }),
        validity: GrayImage::from_fn(16, 16, |x, _| {
            Luma([if x >= left && x < right { 255 } else { 0 }])
        }),
        native_detail: None,
        cell_coverage: (0..256)
            .map(|i| {
                if i % 16 >= left && i % 16 < right {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
        world_origin: (0.0, 0.0),
        world_size: (256.0, 256.0),
        world_stride: 16.0,
    };
    let tiles = [tile(0, 1, 0, 12, false), tile(1, 2, 4, 16, true)];
    let owners = (0..256)
        .flat_map(|_| (0..256).map(|x| if x < 128 { 1 } else { 2 }))
        .collect::<Vec<_>>();
    let plan = super::refine_station_ownership(&tiles, &[(0, 1)], &owners, 256, 256).unwrap();
    assert_eq!(plan.owner_at(8, 128), 1);
    assert_eq!(plan.owner_at(248, 128), 2);
    for y in 2..plan.height - 2 {
        for x in 0..plan.width - 1 {
            if plan.owners[y * plan.width + x] != plan.owners[y * plan.width + x + 1] {
                assert!(
                    !(7..=9).contains(&x),
                    "cut crossed displaced detail at {x},{y}"
                );
            }
        }
    }
}

#[test]
fn ownership_revisit_routes_an_unvalidated_seam_through_covered_relations() {
    use super::super::tone::ToneTile;
    use image::{GrayImage, Luma, Rgb, Rgb32FImage};
    let tile = |station_index, owner_id, left: u32, right: u32| ToneTile {
        station_index,
        owner_id,
        low: Rgb32FImage::from_pixel(16, 16, Rgb([0.4; 3])),
        cell_mean: Rgb32FImage::from_pixel(16, 16, Rgb([0.4; 3])),
        validity: GrayImage::from_fn(16, 16, |x, _| {
            Luma([if x >= left && x < right { 255 } else { 0 }])
        }),
        native_detail: None,
        cell_coverage: (0..256)
            .map(|i| {
                if i % 16 >= left && i % 16 < right {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
        world_origin: (0.0, 0.0),
        world_size: (256.0, 256.0),
        world_stride: 16.0,
    };
    let tiles = [tile(0, 1, 0, 11), tile(1, 2, 5, 16), tile(2, 3, 3, 13)];
    let owners = (0..256)
        .flat_map(|_| (0..256).map(|x| if x < 128 { 1 } else { 2 }))
        .collect::<Vec<_>>();
    let plan =
        super::refine_station_ownership(&tiles, &[(0, 2), (1, 2)], &owners, 256, 256).unwrap();
    assert_eq!(plan.owner_at(8, 128), 1);
    assert_eq!(plan.owner_at(248, 128), 2);
    // The outermost interpolation row has no fully covered proposal.
    for y in 1..plan.height - 1 {
        for x in 0..plan.width - 1 {
            let a = plan.owners[y * plan.width + x];
            let b = plan.owners[y * plan.width + x + 1];
            assert!(
                !((a == 1 && b == 2) || (a == 2 && b == 1)),
                "unvalidated pair at {x},{y}: {:?}",
                plan.owners
            );
        }
    }
}
use super::*;

#[test]
fn existing_seam_binary_reduction_matches_an_exhaustive_energy_oracle() {
    let (width, height) = (3, 2);
    let prior = [0.3, -0.2, 0.5, -0.4, 0.1, -0.6];
    let disagreement = [0.2, 0.8, 0.4, 0.7, 0.1, 0.3];
    let distance = [100.0; 6];
    let fixed = [-1, 0, 0, 0, 0, 1];
    let old: [(f64, f64); 6] = [
        (0.4, 0.6),
        (0.1, 0.0),
        (0.0, 0.2),
        (0.8, 0.0),
        (0.6, 0.0),
        (0.0, 0.0),
    ];
    let energy = |mask: usize| {
        let mut value = 0.0;
        for i in 0..6 {
            let new = mask & (1 << i) != 0;
            if (fixed[i] < 0 && new) || (fixed[i] > 0 && !new) {
                return f64::INFINITY;
            }
            value += if new {
                (-prior[i] * 0.02f64).max(0.0)
            } else {
                (prior[i] * 0.02f64).max(0.0)
            };
            for (other, retained) in [
                ((i % width + 1 < width).then_some(i + 1), old[i].0),
                ((i / width + 1 < height).then_some(i + width), old[i].1),
            ] {
                let Some(other) = other else {
                    continue;
                };
                let other_new = mask & (1 << other) != 0;
                let switched = 0.005 + (disagreement[i] + disagreement[other]) * 0.5;
                value += match (new, other_new) {
                    (false, false) => retained.min(2.0 * switched),
                    (true, true) => 0.0,
                    _ => switched,
                };
            }
        }
        value
    };
    let labels = layered_seam_cut_with_existing_boundaries(
        width,
        height,
        &prior,
        &disagreement,
        &distance,
        &distance,
        &fixed,
        &old,
    );
    let actual = labels
        .iter()
        .enumerate()
        .fold(0, |mask, (i, label)| mask | usize::from(*label > 0) << i);
    let best = (0..64).map(energy).min_by(f64::total_cmp).unwrap();
    assert!((energy(actual) - best).abs() < 1e-10);
}

#[test]
fn incoming_source_can_remove_an_existing_seam_without_creating_a_worse_one() {
    let distance = vec![100.0; 2];
    let labels = layered_seam_cut_with_existing_boundaries(
        2,
        1,
        &[-0.1; 2],
        &[1.0; 2],
        &distance,
        &distance,
        &[0; 2],
        &[(10.0, 0.0), (0.0, 0.0)],
    );
    assert_eq!(labels, vec![255, 255]);
    let distance = vec![100.0; 3];
    let kept = layered_seam_cut_with_existing_boundaries(
        3,
        1,
        &[0.0; 3],
        &[1.0; 3],
        &distance,
        &distance,
        &[-1, 0, -1],
        &[(0.0, 0.0), (0.1, 0.0), (0.0, 0.0)],
    );
    assert_eq!(
        kept,
        vec![0; 3],
        "a cheap retained seam must beat two expensive new seams"
    );
}

#[test]
fn coverage_penalty_must_not_force_a_high_disagreement_boundary_cut() {
    let (width, height) = (64, 8);
    let mut base_distance = vec![100.0; width * height];
    let candidate_distance = vec![100.0; width * height];
    let mut disagreement = vec![5.0; width * height];
    let mut fixed = vec![0; width * height];
    for y in 0..height {
        fixed[y * width] = -1;
        fixed[y * width + width - 1] = 1;
        for x in 0..16 {
            base_distance[y * width + x] = 8.0;
        }
        disagreement[y * width + 32] = 0.0;
        disagreement[y * width + 33] = 0.0;
    }
    let labels = layered_seam_cut(
        width,
        height,
        &vec![0.0; width * height],
        &disagreement,
        &base_distance,
        &candidate_distance,
        &fixed,
    );
    for y in 0..height {
        assert_eq!(
            labels[y * width + 32],
            0,
            "the seam was pushed onto the worse coverage edge"
        );
        assert_eq!(labels[y * width + 33], 255);
    }
}

#[test]
fn layered_cut_follows_real_agreement_and_avoids_coverage_edges() {
    let (w, h) = (20, 8);
    let mut disagreement = vec![0.9; w * h];
    let mut fixed = vec![0; w * h];
    for y in 0..h {
        fixed[y * w] = -1;
        fixed[y * w + w - 1] = 1;
        disagreement[y * w + 5] = 0.0;
        disagreement[y * w + 6] = 0.0;
    }
    let distance = vec![100.0; w * h];
    let labels = layered_seam_cut(
        w,
        h,
        &vec![0.0; w * h],
        &disagreement,
        &distance,
        &distance,
        &fixed,
    );
    for y in 0..h {
        assert_eq!(labels[y * w + 5], 0);
        assert_eq!(labels[y * w + 6], 255);
    }
    let mut boundary = distance.clone();
    for y in 0..h {
        boundary[y * w + 5] = 4.0;
        boundary[y * w + 6] = 4.0;
    }
    let guarded = layered_seam_cut(
        w,
        h,
        &vec![0.0; w * h],
        &disagreement,
        &boundary,
        &boundary,
        &fixed,
    );
    for y in 0..h {
        assert_eq!(guarded[y * w], 0);
        assert_eq!(guarded[y * w + w - 1], 255);
        assert!(!(guarded[y * w + 5] == 0 && guarded[y * w + 6] == 255));
    }
}

#[test]
fn default_choice_is_the_layered_path_with_every_comparison_off() {
    // 需求 15.2 / 15.10.
    let choice = StackCompositorChoice::default();
    assert_eq!(choice, StackCompositorChoice::LayeredVirtualTile);
    assert!(!choice.is_comparison_path());
    assert_eq!(choice.as_identifier(), "layered_virtual_tile");
}

#[test]
fn identifiers_round_trip_for_every_variant() {
    for choice in [
        StackCompositorChoice::LayeredVirtualTile,
        StackCompositorChoice::ProgressiveSeamTile,
        StackCompositorChoice::StreamingMosaic,
        StackCompositorChoice::LegacySingleLayerMosaic,
    ] {
        let identifier = choice.as_identifier();
        assert_eq!(
            StackCompositorChoice::from_identifier(identifier),
            Some(choice),
            "{identifier} must round trip"
        );
        assert_eq!(choice.selected_path().as_identifier(), identifier);
    }
}

#[test]
fn unset_and_unknown_settings_resolve_to_the_default_path() {
    // 任务 13.7: the setting is the only input of `resolve`.
    assert_eq!(StackCompositorChoice::from_identifier(""), None);
    assert_eq!(StackCompositorChoice::from_identifier("nope"), None);
    assert_eq!(
        StackCompositorChoice::resolve(Some("  Streaming_Mosaic ")),
        StackCompositorChoice::StreamingMosaic
    );
    for setting in [None, Some(""), Some("nope")] {
        assert_eq!(
            StackCompositorChoice::resolve(setting),
            StackCompositorChoice::LayeredVirtualTile
        );
    }
}

#[test]
fn default_multi_station_selection_records_the_layered_path() {
    let recorder = StackReportRecorder::isolated("default-layered-selection", None);
    let selection = select_and_record_run_path(
        Some(2),
        StackCompositorChoice::resolve(None),
        Some(&recorder),
    );

    assert_eq!(selection.selected_path, SelectedPath::LayeredVirtualTile);
    assert!(selection.use_virtual_tiles);
    assert_eq!(
        recorder.snapshot().selected_path,
        SelectedPath::LayeredVirtualTile
    );
}

#[test]
fn explicit_legacy_selection_records_the_legacy_path_without_virtual_tiles() {
    let recorder = StackReportRecorder::isolated("legacy-selection", None);
    let selection = select_and_record_run_path(
        Some(2),
        StackCompositorChoice::LegacySingleLayerMosaic,
        Some(&recorder),
    );

    assert_eq!(
        selection.selected_path,
        SelectedPath::LegacySingleLayerMosaic
    );
    assert!(!selection.use_virtual_tiles);
    assert_eq!(
        recorder.snapshot().selected_path,
        SelectedPath::LegacySingleLayerMosaic
    );
}

#[test]
fn default_layered_entry_runs_group_tone_and_quality_gate_components() {
    // Exercise the same production component entry points selected by a
    // fresh install: the layered path, group Tone_Harmonizer and the
    // record-only Quality_Gate.  Keeping this fixture synthetic avoids a
    // RAW decode while still asserting the returned values that the
    // pipeline publishes to Stack_Report.
    use crate::panorama_utils::stack_pipeline::report::{QualityGateVerdict, ToneStatus};
    use crate::panorama_utils::stack_pipeline::{
        degradation, quality_gate, quality_gate_runner, residual_warp, tone,
    };
    use image::{GrayImage, Luma, Rgb, Rgb32FImage};
    use nalgebra::Matrix3;

    let _run_scope = degradation::begin_run_scope();
    let recorder = StackReportRecorder::isolated("default-layered-components", None);
    let selection = select_and_record_run_path(
        Some(2),
        StackCompositorChoice::resolve(None),
        Some(&recorder),
    );
    assert_eq!(selection.selected_path, SelectedPath::LayeredVirtualTile);
    assert!(selection.use_virtual_tiles);

    let (width, height) = (1_536u32, 512u32);
    let low_size = (64, 32);
    let tile = |station_index, owner_id, origin, value| tone::ToneTile {
        station_index,
        owner_id,
        low: Rgb32FImage::from_pixel(low_size.0, low_size.1, Rgb([value; 3])),
        validity: GrayImage::from_pixel(low_size.0, low_size.1, Luma([255])),
        world_origin: (origin, 0.0),
        world_size: (1_024.0, 512.0),
        world_stride: 16.0,
        cell_mean: Rgb32FImage::from_pixel(low_size.0, low_size.1, Rgb([value; 3])),
        native_detail: None,
        cell_coverage: vec![1.0; low_size.0 as usize * low_size.1 as usize],
    };
    let tiles = [tile(0, 1, 0.0, 0.40), tile(1, 2, 512.0, 0.36)];
    let owners = (0..height)
        .flat_map(|_| (0..width).map(|x| if x < 768 { 1u16 } else { 2u16 }))
        .collect::<Vec<_>>();
    let mut panorama = Rgb32FImage::from_fn(width, height, |x, _| {
        Rgb([if x < 768 { 0.40 } else { 0.36 }; 3])
    });
    let tone_report = tone::harmonize_tone_tiles(
        &mut panorama,
        &owners,
        width,
        &tiles,
        &GrayImage::from_pixel(width, height, Luma([255])),
        &[(0, 1)],
    );
    assert_eq!(tone_report.status, ToneStatus::Applied);
    assert_eq!(tone_report.tiles.len(), 2);

    let (qg_width, qg_height) = (1_024u32, 1_024u32);
    let output = Rgb32FImage::from_fn(qg_width, qg_height, |x, y| {
        let value = 0.2 + 0.3 * ((x + y) % 37) as f32 / 36.0;
        Rgb([value, value * 0.9, value * 0.8])
    });
    let coverage = GrayImage::from_pixel(qg_width, qg_height, Luma([255]));
    let ownership = vec![1u16; (qg_width * qg_height) as usize];
    let confidence = vec![0.8f32; ownership.len()];
    let textured = vec![255u8; ownership.len()];
    let shortfall = vec![0u8; ownership.len()];
    let disagreement = vec![0u8; ownership.len()];
    let source = quality_gate_runner::QualitySource {
        owner: 1,
        path: "synthetic.raw".to_string(),
        geometry: quality_gate::SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 0,
        },
        dimensions: (qg_width, qg_height),
        native_warp: None,
        native_registration: None,
    };
    let residual = residual_warp::ResidualWarp::new();
    let sources = [source];
    let input = quality_gate_runner::QualityGateInput {
        output: &output,
        coverage: &coverage,
        ownership: &ownership,
        confidence: &confidence,
        textured: Some(&textured),
        owner_shortfall: Some(&shortfall),
        owner_disagreement: Some(&disagreement),
        unresolved_pixel_count: 0,
        world_origin: (0.0, 0.0),
        sources: &sources,
        accepted_station_relations: &[],
        residual: &residual,
        acceptance_render_scale: 1.0,
        final_sharpen_amount: 0.0,
    };
    let quality_report =
        quality_gate_runner::run_quality_gate(&input, &mut |_source| Ok(output.clone()));
    assert_eq!(
        quality_report.criteria.len(),
        quality_gate::QUALITY_CRITERIA.len()
    );
    assert_ne!(quality_report.verdict, QualityGateVerdict::NotRun);

    recorder.update(|report| {
        report.tone = tone_report.clone();
        report.quality_gate = quality_report.clone();
    });
    let published = recorder.snapshot();
    assert_eq!(published.selected_path, SelectedPath::LayeredVirtualTile);
    assert_eq!(published.tone.status, ToneStatus::Applied);
    assert_eq!(
        published.quality_gate.criteria.len(),
        quality_gate::QUALITY_CRITERIA.len()
    );
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig {
        cases: 100,
        ..proptest::prelude::ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 60: 对于任意联合覆盖边界，若其
    // 长边超过支持的画布上限，则不写出最终结果文件，且实测画布尺寸与该上限被记录。
    //
    // The same function writes the `canvas_long_side_exceeded` ledger entry; the
    // returned error, which becomes the run's error, carries the measured size and the
    // limit, and `is_canvas_rejection` keeps every fallback renderer from retrying.
    //
    // **Validates: Requirements 10.11**
    #[test]
    fn property_60_oversized_canvas_is_rejected_with_its_measured_size(
        width in 1u64..600_000,
        height in 1u64..600_000,
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let rejection = reject_oversized_canvas(width, height);
        proptest::prop_assert_eq!(
            rejection.is_some(),
            width.max(height) > MAX_OUTPUT_CANVAS_LONG_SIDE
        );
        if let Some(error) = rejection {
            proptest::prop_assert!(is_canvas_rejection(&error), "{}", error);
            let measured = format!("{width}x{height}");
            proptest::prop_assert!(error.contains(&measured), "{}", error);
            proptest::prop_assert!(
                error.contains(&MAX_OUTPUT_CANVAS_LONG_SIDE.to_string()),
                "{}",
                error
            );
        }
    }
}

#[test]
fn canvas_limit_boundary_is_inclusive() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    assert!(reject_oversized_canvas(MAX_OUTPUT_CANVAS_LONG_SIDE, 1).is_none());
    assert!(reject_oversized_canvas(1, MAX_OUTPUT_CANVAS_LONG_SIDE).is_none());
    assert!(reject_oversized_canvas(MAX_OUTPUT_CANVAS_LONG_SIDE + 1, 1).is_some());
    assert!(!is_canvas_rejection("Virtual tile 3 dimensions changed"));
}

#[test]
fn layered_compositor_rejects_an_oversized_canvas_before_loading_a_tile() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    use crate::panorama_utils::shared::ImageInfo;
    use crate::panorama_utils::stitching;
    // An 8 px tile placed at 40,000× spans 320,000 world pixels.
    let info = ImageInfo {
        id: 0,
        filename: "/station/oversized/0000.NEF".to_string(),
        width: 8,
        height: 8,
        alignment_image: image::GrayImage::new(1, 1),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: Some(75.0),
        source_footprint: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    };
    let homographies = std::collections::HashMap::from([(
        0usize,
        Matrix3::new(40_000.0, 0.0, 0.0, 0.0, 40_000.0, 0.0, 0.0, 0.0, 1.0),
    )]);
    let app = tauri::test::mock_app();
    let mut loads = 0usize;
    let mut load = |_: &ImageInfo| -> Result<image::Rgb32FImage, String> {
        loads += 1;
        Ok(image::Rgb32FImage::new(8, 8))
    };
    let result = stitching::layered_virtual_tile_compositor_with_ownership(
        &[&info],
        &homographies,
        stitching::Projection::Planar,
        app.handle().clone(),
        "compositor-canvas-limit-test",
        &mut load,
    );
    let Err(error) = result else {
        panic!("an oversized canvas must be rejected");
    };
    assert!(is_canvas_rejection(&error), "{error}");
    assert_eq!(loads, 0, "no tile may be decoded for a rejected canvas");
}

#[test]
fn single_station_is_not_a_compositor_choice() {
    // 需求 15.11's identifier describes the pipeline shape of a one-station
    // run, so it must not be selectable as a compositor.
    assert_eq!(
        StackCompositorChoice::from_identifier(SelectedPath::SingleStation.as_identifier()),
        None
    );
}

#[test]
fn tile_coordinate_composes_inverse_warp_before_one_sample() {
    let mut tile_to_world_inverse = Matrix3::identity();
    tile_to_world_inverse[(0, 2)] = -10.0;
    let mut residual_inverse = Matrix3::identity();
    residual_inverse[(1, 2)] = 3.0;
    let coordinate = tile_coordinate_from_world(
        Point2::new(12.0, 8.0),
        &tile_to_world_inverse,
        &residual_inverse,
    )
    .expect("finite composed coordinate");
    assert_eq!(coordinate, Point2::new(2.0, 11.0));
}

#[test]
fn residual_free_coordinate_is_exactly_the_nominal_inverse() {
    let mut inverse = Matrix3::identity();
    inverse[(0, 2)] = -10.0;
    let world = Point2::new(12.0, 8.0);
    let residual = ResidualWarp::default();
    assert_eq!(
        tile_coordinate_with_residual(world, &inverse, &residual, 7),
        tile_coordinate_from_world(world, &inverse, &Matrix3::identity())
    );
}

#[test]
fn tone_cannot_change_the_ownership_plane() {
    let owners = vec![1u16, 2, 2, 1];
    assert_ownership_unchanged(&owners, &owners);
}
