use super::*;
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stack_pipeline::{degradation, residual_warp, tone};
use crate::panorama_utils::stitching::{
    CoverageMask, FocusSourceRefinement, FocusSourceSampling, FocusTileLoadPurpose,
    FocusTilePayload, FocusTileSource, OwnershipMap, Projection,
};
use image::{GrayImage, Rgb32FImage};
use nalgebra::Matrix3;
use std::collections::HashMap;

struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("native-point-trace-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Override;
impl Override {
    fn set(input: Option<PathBuf>, directory: Option<PathBuf>) -> Self {
        PATHS_OVERRIDE.with(|value| {
            assert!(value.borrow().is_none());
            *value.borrow_mut() = Some((input, directory));
        });
        Self
    }
}
impl Drop for Override {
    fn drop(&mut self) {
        PATHS_OVERRIDE.with(|value| *value.borrow_mut() = None);
    }
}

fn input_value(count: usize) -> Value {
    json!({"schema":1,"reference_canvas_origin":[0.0,0.0],"unknown_future_field":true,
        "points":(0..count).map(|x|json!({"label":format!("p{x}"),"canvas":[x,4],"role":"opaque_control","future":1})).collect::<Vec<_>>()})
}
fn input_file(directory: &Path, points: Value) -> PathBuf {
    let path = directory.join("points.json");
    fs::write(&path, serde_json::to_vec(&points).unwrap()).unwrap();
    path
}
fn info() -> ImageInfo {
    ImageInfo {
        id: 42,
        filename: "/fixture/trace-source.NEF".into(),
        width: 16,
        height: 16,
        alignment_image: GrayImage::new(1, 1),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: None,
        source_footprint: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}
fn sources(image: &ImageInfo) -> FocusTileSources {
    let mut original = Rgb32FImage::from_pixel(16, 16, Rgb([0.6, 0.5, 0.4]));
    original.put_pixel(4, 4, Rgb([0.0; 3]));
    FocusTileSources {
        ownership: OwnershipMap::new(16, 16, vec![1; 256], vec![PathBuf::from(&image.filename)])
            .unwrap(),
        origin: (0.0, 0.0),
        originals: vec![Some(FocusTileSource {
            image: original,
            refinement: FocusSourceRefinement {
                source_sampling: Some(FocusSourceSampling::for_test(
                    image,
                    &Matrix3::identity(),
                    Projection::Planar,
                    None,
                )),
                ..Default::default()
            },
        })],
    }
}

#[test]
fn point_trace_input_caps_all_roles_ignores_future_fields_and_keeps_grid_origin() {
    assert_eq!(
        parse_input(&serde_json::to_vec(&input_value(128)).unwrap())
            .unwrap()
            .points
            .len(),
        128
    );
    assert!(parse_input(&serde_json::to_vec(&input_value(129)).unwrap()).is_err());
    assert!(parse_input(&serde_json::to_vec(&input_value(0)).unwrap()).is_err());
    let mut duplicate = input_value(2);
    duplicate["points"][1]["canvas"] = duplicate["points"][0]["canvas"].clone();
    assert!(parse_input(&serde_json::to_vec(&duplicate).unwrap()).is_err());
    assert!(parse_input(&vec![b' '; MAX_INPUT_BYTES + 1]).is_err());
    assert_eq!(
        pixel_on_grid([9237.0, 473.0], [-108.0, -49.0], [15897, 6720]),
        Some([9345, 522])
    );
    assert_eq!(
        pixel_on_grid([9237.25, 473.0], [-108.0, -49.0], [15897, 6720]),
        None
    );
    assert_eq!(
        pixel_on_grid([-109.0, 473.0], [-108.0, -49.0], [15897, 6720]),
        None
    );
    // A fractional origin can put a point just inside the continuous bound
    // while tolerance-based rounding reaches the first out-of-range index.
    assert_eq!(pixel_on_grid([16.0, 4.0], [1e-7, 0.0], [16, 16]), None);
    assert_eq!(
        pixel_on_grid([15.0, 4.0], [1e-7, 0.0], [16, 16]),
        Some([15, 4])
    );
}

#[test]
fn point_trace_diagnoses_support_threshold_and_missing_owner_without_changing_sample() {
    let info = info();
    let mut sources = sources(&info);
    let black = Point2::new(4.0, 4.0);
    let before = sources.sample(black);
    let diagnostic = diagnose_sources(&sources, black);
    assert_eq!(sources.sample(black), before);
    assert_eq!(before, Some(Rgb([0.0; 3])));
    assert_eq!(
        diagnostic["actual_algorithm_3x3_winner_diagnostic"]["local_owner"],
        1
    );
    assert_eq!(
        diagnostic["legacy_reverse_4x4_winner_diagnostic"]["local_owner"],
        1
    );
    let source = &diagnostic["source_diagnostics"][0];
    assert_eq!(source["source"]["id"], 42);
    assert_eq!(source["source"]["filename"], "/fixture/trace-source.NEF");
    assert_eq!(source["black_filter_le_1e_minus_6"], true);
    assert_eq!(source["nearest_floor_support_diagnostic_only"], false);
    let border = diagnose_sources(&sources, Point2::new(15.5, 4.0));
    assert_eq!(
        border["source_diagnostics"][0]["render_bounds_exclusive_dimensions"],
        true
    );
    assert_eq!(
        border["source_diagnostics"][0]["qg_bounds_inclusive_last_center"],
        false
    );
    sources.originals[0]
        .as_mut()
        .unwrap()
        .refinement
        .translation = (100, 0);
    let invalid = diagnose_sources(&sources, Point2::new(5.0, 4.0));
    assert_eq!(
        invalid["source_diagnostics"][0]["rejection"],
        "source_mapping_returned_none"
    );
    assert!(sources.sample(Point2::new(5.0, 4.0)).is_none());
    sources.originals[0] = None;
    assert_eq!(
        diagnose_sources(&sources, black)["source_diagnostics"][0]["rejection"],
        "original_missing"
    );
}

fn render_fixture() -> LayeredOwnershipRender {
    let info = info();
    let app = tauri::test::mock_app();
    let mut load = |_: &ImageInfo, _: FocusTileLoadPurpose| {
        Ok::<_, String>(FocusTilePayload {
            image: Rgb32FImage::from_pixel(16, 16, Rgb([0.3; 3])),
            sources: Some(sources(&info)),
        })
    };
    super::super::layered_virtual_tile_compositor_with_ownership(
        &[&info],
        &HashMap::from([(info.id, Matrix3::identity())]),
        Projection::Planar,
        app.handle().clone(),
        "native-point-trace-test",
        &mut load,
    )
    .unwrap()
}

#[test]
fn point_trace_off_and_on_preserve_pixels_and_capture_actual_black_rejection() {
    let _scope = degradation::begin_run_scope();
    tone::reset_run_records();
    tone::reset_run_topology();
    residual_warp::reset_run_records();
    let directory = Temporary::new();
    let input = input_file(
        &directory.0,
        json!({"schema":1,"reference_canvas_origin":[0.0,0.0],"points":[
        {"label":"black","canvas":[4,4],"role":"known_hole"},
        {"label":"positive","canvas":[5,4],"role":"opaque_control"}]}),
    );
    let off = {
        let _config = Override::set(None, None);
        assert!(PointTrace::from_env().is_none());
        render_fixture()
    };
    assert!(
        !directory
            .0
            .join(format!("process-{}", std::process::id()))
            .exists()
    );
    let on = {
        let _config = Override::set(Some(input), Some(directory.0.clone()));
        render_fixture()
    };
    assert_eq!(off.image, on.image);
    assert_eq!(off.ownership, on.ownership);
    assert_eq!(off.coverage.covered(), on.coverage.covered());
    assert_eq!(off.sampling_origin, on.sampling_origin);
    let latest: Value = serde_json::from_slice(
        &fs::read(
            directory
                .0
                .join(format!("process-{}/latest.json", std::process::id())),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(latest["latest_iteration_status"], "compositor_complete");
    assert!(latest["final_iteration"].is_null());
    let trace: Value =
        serde_json::from_slice(&fs::read(latest["trace"].as_str().unwrap()).unwrap()).unwrap();
    let stages = trace["stage_snapshots"].as_array().unwrap();
    assert_eq!(stages.len(), 5);
    let names = [
        "before_seam_revisit",
        "after_seam_revisit",
        "before_tone_harmonizer",
        "after_tone_harmonizer",
        "after_finishing_before_crop",
    ];
    for (stage, name) in stages.iter().zip(names) {
        assert_eq!(stage["stage"], name);
        assert_eq!(stage["working_canvas"], trace["working_canvas"]);
        let snapshots = stage["points"].as_array().unwrap();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0]["sample"]["coverage"], 0);
        assert_eq!(snapshots[1]["sample"]["coverage"], 255);
        assert_eq!(snapshots[1]["sample"]["station_owner"], 1);
        for point in snapshots {
            for channel in 0..3 {
                let value = point["sample"]["rgb"][channel].as_f64().unwrap() as f32;
                assert_eq!(point["sample"]["rgb_f32_bits"][channel], value.to_bits());
            }
        }
    }
    let points = &trace["stations"][0]["points"];
    assert_eq!(points[0]["actual"]["status"], "black_filter_rejected");
    assert_eq!(points[0]["actual"]["sample_returned_some"], true);
    assert_eq!(points[0]["actual"]["black_filter_le_1e_minus_6"], true);
    assert_eq!(points[0]["actual"]["mask_written"], false);
    assert_eq!(points[1]["actual"]["status"], "mask_written");
    assert_eq!(
        trace["compositor_return"]["points"][0]["sample"]["coverage"],
        false
    );
    assert_eq!(
        trace["compositor_return"]["points"][1]["sample"]["coverage"],
        true
    );
    // A row without a point never evaluates the lazy evidence closure.
    let mut state = PointTrace::begin_paths(
        Some(input_file(&directory.0, input_value(1))),
        Some(directory.0.clone()),
    )
    .unwrap();
    state.set_canvas([16, 16], [0.0, 0.0]);
    let station = state.station(json!({}), [0, 15, 0, 15], [16, 16], None, |p| {
        Some(Point2::new(p.x, p.y))
    });
    assert!(station.row(3).is_none());
    assert!(
        station
            .row(4)
            .unwrap()
            .pixel(9, || panic!("unselected coordinate evaluated diagnostics"))
            .is_none()
    );
    tone::reset_run_records();
    tone::reset_run_topology();
    residual_warp::reset_run_records();
}

#[test]
fn point_trace_later_failure_supersedes_completed_iteration_and_cannot_claim_final() {
    let directory = Temporary::new();
    let input = input_file(&directory.0, input_value(1));
    let latest_path = directory
        .0
        .join(format!("process-{}/latest.json", std::process::id()));
    let mut first =
        PointTrace::begin_paths(Some(input.clone()), Some(directory.0.clone())).unwrap();
    first.set_canvas([16, 16], [0.0, 0.0]);
    first.complete(&LayeredOwnershipRender {
        image: Rgb32FImage::from_pixel(16, 16, Rgb([0.5; 3])),
        coverage: CoverageMask::from_gray(GrayImage::from_pixel(16, 16, image::Luma([255]))),
        ownership: OwnershipMap::new(16, 16, vec![1; 256], vec![PathBuf::from("/fixture/source")])
            .unwrap(),
        sampling_origin: (0.0, 0.0),
        full_resolution_samples: 0,
        full_resolution_sample_counts: vec![0; 256],
    });
    let first_iteration = first.iteration;
    drop(first);
    let next = PointTrace::begin_paths(Some(input.clone()), Some(directory.0.clone())).unwrap();
    let running: Value = serde_json::from_slice(&fs::read(&latest_path).unwrap()).unwrap();
    assert_eq!(running["latest_iteration_status"], "running");
    assert!(running["final_iteration"].is_null());
    assert!(next.iteration > first_iteration);
    drop(next);
    let failed: Value = serde_json::from_slice(&fs::read(&latest_path).unwrap()).unwrap();
    assert_eq!(failed["latest_iteration_status"], "failed");
    assert!(failed["latest_completed_iteration"].is_null());
    assert!(failed["final_iteration"].is_null());
    fs::write(&input, serde_json::to_vec(&input_value(129)).unwrap()).unwrap();
    assert!(PointTrace::begin_paths(Some(input), Some(directory.0.clone())).is_none());
    let invalid: Value = serde_json::from_slice(&fs::read(&latest_path).unwrap()).unwrap();
    assert_eq!(invalid["latest_iteration_status"], "failed");
    assert!(invalid["final_iteration"].is_null());
}
