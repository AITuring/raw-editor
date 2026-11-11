use super::*;
use crate::panorama_utils::mosaic::{Field, NativeResidualDisplacement};
use crate::panorama_utils::stack_pipeline::degradation;
use crate::panorama_utils::stack_pipeline::quality_gate::SourceGeometry;
use crate::panorama_utils::stack_pipeline::quality_gate_runner::{
    NativeReferenceMap, QualityRoiCache,
};
use crate::panorama_utils::stack_pipeline::residual_warp::ResidualWarp;
use crate::panorama_utils::stitching::{FocusForegroundMove, FocusSourceRefinement};
use image::{GenericImageView, Rgb};

fn source(owner: u16) -> QualitySource {
    QualitySource {
        owner,
        path: format!("source-{owner}.nef"),
        dimensions: (1024, 1024),
        geometry: SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 0,
        },
        native_warp: None,
        native_registration: None,
    }
}

fn input<'a>(
    output: &'a Rgb32FImage,
    coverage: &'a GrayImage,
    ownership: &'a [u16],
    confidence: &'a [f32],
    sources: &'a [QualitySource],
    residual: &'a ResidualWarp,
) -> QualityGateInput<'a> {
    QualityGateInput {
        output,
        coverage,
        ownership,
        confidence,
        textured: None,
        owner_shortfall: None,
        owner_disagreement: None,
        unresolved_pixel_count: 0,
        world_origin: (0.0, 0.0),
        sources,
        accepted_station_relations: &[],
        residual,
        acceptance_render_scale: 1.0,
        final_sharpen_amount: 0.0,
    }
}

fn roi(side: u32) -> QualityRoi {
    QualityRoi {
        x: 0,
        y: 0,
        side,
        owner: 1,
        region_label: quality_gate::station_level_roi_label(0).unwrap(),
        world_origin: (0.0, 0.0),
        boundary_clearance: 0.0,
    }
}

fn config() -> Config {
    Config {
        directory: std::env::temp_dir().join(format!("qg-source-visual-{}", uuid::Uuid::new_v4())),
        profile: "unit-test".into(),
        limit: 1,
    }
}

fn coordinate(bytes: &[u8], pixel: usize) -> (f64, f64) {
    let start = pixel * 16;
    (
        f64::from_le_bytes(bytes[start..start + 8].try_into().unwrap()),
        f64::from_le_bytes(bytes[start + 8..start + 16].try_into().unwrap()),
    )
}

#[test]
fn source_visual_capture_maps_every_owner_through_native_registration() {
    let output = Rgb32FImage::new(2, 2);
    let coverage = GrayImage::from_raw(2, 2, vec![255, 255, 255, 0]).unwrap();
    let ownership = [1, 2, 2, 1];
    let confidence = [1.0; 4];
    let residual = ResidualWarp::new();
    let station_to_world = Matrix3::new(1.0, 0.0, 100.0, 0.0, 1.0, 200.0, 0.0, 0.0, 1.0);
    let mut field = Field::<2>::new(128, 128, 16.0);
    field.values.fill([1.5, -0.5]);
    let native = NativeReferenceMap {
        mosaic_tone: None,
        source_prefilter_divisor: 1.0,
        world_to_station: station_to_world.try_inverse().unwrap(),
        station_to_world,
        station_origin: (0.0, 0.0),
        refinement: FocusSourceRefinement {
            translation: (3, -2),
            displacement: Some(NativeResidualDisplacement::from_test_field(field)),
            foreground_move: Some(FocusForegroundMove {
                delta: (4, 0),
                left: 0,
                top: 0,
                destinations: GrayImage::from_pixel(128, 128, Luma([255])),
            }),
            source_sampling: None,
        },
    };
    let mut sources = [source(1), source(2)];
    for source in &mut sources {
        source.geometry.tile_to_world = station_to_world;
        source.native_registration = Some(native.clone());
    }
    sources[1].geometry.tile_to_world[(0, 2)] = 90.0;
    let input = input(
        &output,
        &coverage,
        &ownership,
        &confidence,
        &sources,
        &residual,
    );
    let mut roi = roi(2);
    roi.world_origin = (112.0, 224.0);
    let evidence = pixel_evidence(&input, &roi);
    assert_eq!(evidence.owners, ownership);
    assert_eq!(evidence.validity, [1, 1, 1, 0]);
    assert_eq!(coordinate(&evidence.coordinates, 0), (6.5, 25.5));
    assert_eq!(coordinate(&evidence.coordinates, 1), (17.5, 25.5));
    assert_eq!(coordinate(&evidence.coordinates, 2), (16.5, 26.5));
    assert!(coordinate(&evidence.coordinates, 3).0.is_nan());
    let counts = owner_counts(&input);
    assert_eq!(counts["covered_pixels"], 3);
    assert_eq!(counts["owners"][0]["pixels"], 1);
    assert_eq!(counts["owners"][1]["pixels"], 2);
}

#[test]
fn source_visual_capture_follows_native_sampling_and_station_origin() {
    use crate::panorama_utils::shared::ImageInfo;
    use crate::panorama_utils::stitching::{FocusSourceSampling, Projection};
    let output = Rgb32FImage::new(1, 1);
    let coverage = GrayImage::from_pixel(1, 1, Luma([255]));
    let ownership = [1];
    let confidence = [1.0];
    let residual = ResidualWarp::new();
    let info = ImageInfo {
        id: 0,
        filename: "prepared.nef".into(),
        width: 128,
        height: 128,
        alignment_image: GrayImage::new(128, 128),
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
    };
    let sampling = FocusSourceSampling::for_test(
        &info,
        &Matrix3::new(1.0, 0.0, 4.0, 0.0, 1.0, 8.0, 0.0, 0.0, 1.0),
        Projection::Planar,
        None,
    );
    let station_to_world = Matrix3::new(1.0, 0.0, 100.0, 0.0, 1.0, 200.0, 0.0, 0.0, 1.0);
    let mut sources = [source(1)];
    sources[0].native_registration = Some(NativeReferenceMap {
        mosaic_tone: None,
        source_prefilter_divisor: 1.0,
        world_to_station: station_to_world.try_inverse().unwrap(),
        station_to_world,
        station_origin: (5.0, -7.0),
        refinement: FocusSourceRefinement {
            translation: (3, -2),
            source_sampling: Some(sampling),
            ..Default::default()
        },
    });
    let input = input(
        &output,
        &coverage,
        &ownership,
        &confidence,
        &sources,
        &residual,
    );
    let mut roi = roi(1);
    roi.world_origin = (112.0, 224.0);
    let evidence = pixel_evidence(&input, &roi);
    assert_eq!(evidence.validity, [1]);
    assert_eq!(coordinate(&evidence.coordinates, 0), (10.0, 11.0));
}

#[test]
fn source_visual_capture_marks_missing_wrong_station_and_outside_mapping() {
    let output = Rgb32FImage::new(2, 2);
    let coverage = GrayImage::from_pixel(2, 2, Luma([255]));
    let ownership = [0, 7, 2, 1];
    let confidence = [1.0; 4];
    let residual = ResidualWarp::new();
    let mut sources = [source(1), source(2)];
    sources[0].geometry.tile_to_world[(0, 2)] = 100.0;
    sources[1].geometry.station_id = 9;
    let input = input(
        &output,
        &coverage,
        &ownership,
        &confidence,
        &sources,
        &residual,
    );
    let evidence = pixel_evidence(&input, &roi(2));
    assert_eq!(evidence.validity, [2, 3, 4, 7]);
}

#[test]
fn source_visual_capture_keeps_exact_pixels_limits_rois_and_supersedes_iterations() {
    let config = config();
    let output = Rgb32FImage::from_fn(2, 2, |x, y| Rgb([x as f32 * 0.25, y as f32 * 0.3, 0.125]));
    let reference = Rgb32FImage::from_pixel(2, 2, Rgb([0.125, 0.25, 0.5]));
    let coverage = GrayImage::from_pixel(2, 2, Luma([255]));
    let ownership = [1; 4];
    let confidence = [1.0; 4];
    let residual = ResidualWarp::new();
    let sources = [source(1)];
    let input = input(
        &output,
        &coverage,
        &ownership,
        &confidence,
        &sources,
        &residual,
    );
    let mut first = Capture::begin(Some(config.clone()), &input).unwrap();
    first.record(&input, &roi(2), &output, &reference, true);
    first.record(&input, &roi(2), &output, &reference, false);
    assert_eq!(first.rois.len(), 1);
    assert_eq!(first.rois[0]["reference_cache_hit"], true);
    let patch_dir = first.iteration_directory.join("roi-00");
    assert_eq!(
        fs::read(patch_dir.join("output.rgb.f32le")).unwrap(),
        output
            .as_raw()
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        image::open(patch_dir.join("output.png"))
            .unwrap()
            .dimensions(),
        (2, 2)
    );
    first.complete(&QualityGateReport::default()).unwrap();
    let mut second = Capture::begin(Some(config.clone()), &input).unwrap();
    assert!(second.iteration > first.iteration);
    let running: Value =
        serde_json::from_slice(&fs::read(second.process_directory.join("latest.json")).unwrap())
            .unwrap();
    assert_eq!(running["latest_iteration_status"], "running");
    assert!(running["final_iteration"].is_null());
    second.complete(&QualityGateReport::default()).unwrap();
    let latest: Value =
        serde_json::from_slice(&fs::read(second.process_directory.join("latest.json")).unwrap())
            .unwrap();
    assert_eq!(latest["latest_completed_iteration"], second.iteration);
    assert!(
        latest["final_iteration"].is_null(),
        "export completion must not be fabricated by the QG helper"
    );
    fs::remove_dir_all(config.directory).unwrap();
}

#[test]
fn source_visual_capture_metadata_write_failure_invalidates_previous_complete() {
    let config = config();
    let output = Rgb32FImage::new(1, 1);
    let coverage = GrayImage::from_pixel(1, 1, Luma([255]));
    let ownership = [1];
    let confidence = [1.0];
    let residual = ResidualWarp::new();
    let sources = [source(1)];
    let input = input(
        &output,
        &coverage,
        &ownership,
        &confidence,
        &sources,
        &residual,
    );
    let mut previous = Capture::begin(Some(config.clone()), &input).unwrap();
    previous.complete(&QualityGateReport::default()).unwrap();
    let latest_path = previous.process_directory.join("latest.json");
    let latest: Value = serde_json::from_slice(&fs::read(&latest_path).unwrap()).unwrap();
    assert_eq!(latest["latest_iteration_status"], "complete");

    let failed =
        Capture::begin_with_metadata_writer(Some(config.clone()), &input, |path, value| {
            // The previous completion must already be superseded before the first
            // iteration metadata write. A directory at its destination forces a
            // real filesystem write/rename failure without relying on permissions.
            let running: Value = serde_json::from_slice(&fs::read(&latest_path).unwrap()).unwrap();
            assert_eq!(running["latest_iteration_status"], "running");
            assert_eq!(running["offline_binding_allowed"], false);
            assert!(running["latest_completed_iteration"].is_null());
            fs::create_dir(path).unwrap();
            write_json(path, value)
        });
    assert!(failed.is_none());
    let latest: Value = serde_json::from_slice(&fs::read(&latest_path).unwrap()).unwrap();
    assert_eq!(latest["latest_iteration_status"], "failed");
    assert_eq!(latest["offline_binding_allowed"], false);
    assert!(latest["latest_started_iteration"].as_u64().unwrap() > previous.iteration);
    assert!(latest["latest_completed_iteration"].is_null());
    assert!(latest["final_iteration"].is_null());
    assert!(!latest["error"].as_str().unwrap().is_empty());
    assert!(previous.iteration_directory.join("metadata.json").is_file());
    fs::remove_dir_all(config.directory).unwrap();
}

struct OverrideGuard;
impl Drop for OverrideGuard {
    fn drop(&mut self) {
        CONFIG_OVERRIDE.with(|value| *value.borrow_mut() = None);
    }
}

#[test]
fn source_visual_capture_selects_each_station_before_extra_rois() {
    let config = Config {
        limit: 4,
        ..config()
    };
    let output = Rgb32FImage::new(2, 2);
    let coverage = GrayImage::from_pixel(2, 2, Luma([255]));
    let ownership = [1; 4];
    let confidence = [1.0; 4];
    let residual = ResidualWarp::new();
    let sources = [source(1)];
    let input = input(
        &output,
        &coverage,
        &ownership,
        &confidence,
        &sources,
        &residual,
    );
    let mut capture = Capture::begin(Some(config.clone()), &input).unwrap();
    let rois = [0, 0, 0, 0, 1, 2, 2, 2]
        .into_iter()
        .enumerate()
        .map(|(index, station)| {
            let mut value = roi(2);
            value.x = index as u32;
            value.region_label = quality_gate::station_level_roi_label(station).unwrap();
            value
        })
        .collect::<Vec<_>>();
    capture.select_rois(&rois, &sources);
    assert_eq!(
        capture.metadata["roi_selection"]["selected_roi_indices"],
        json!([0, 4, 5, 1])
    );
    assert_eq!(
        capture.metadata["roi_selection"]["excluded_station_ids"],
        json!([])
    );
    fs::remove_dir_all(config.directory).unwrap();
}

#[test]
fn source_visual_capture_reuses_cached_reference_without_loading_or_mutating_gate_input() {
    let _scope = degradation::begin_run_scope();
    crate::panorama_utils::stack_pipeline::tone::reset_run_records();
    let config = config();
    let output = Rgb32FImage::from_pixel(1024, 1024, Rgb([0.5; 3]));
    let original_output = output.clone();
    let coverage = GrayImage::from_pixel(1024, 1024, Luma([255]));
    let ownership = vec![1; 1024 * 1024];
    let confidence = vec![1.0; 1024 * 1024];
    let residual = ResidualWarp::new();
    let sources = [source(1)];
    let input = input(
        &output,
        &coverage,
        &ownership,
        &confidence,
        &sources,
        &residual,
    );
    CONFIG_OVERRIDE.with(|value| *value.borrow_mut() = Some(None));
    let _override_guard = OverrideGuard;
    let mut cache = QualityRoiCache::new(64 * 1024 * 1024);
    let mut loads = 0;
    let first = super::super::run_quality_gate_with_cache(
        &input,
        &mut |_| {
            loads += 1;
            Ok(output.clone())
        },
        &mut cache,
    );
    assert!(first.roi_count > 0);
    assert_eq!(loads, 1);
    assert!(
        !config.directory.exists(),
        "disabled capture must not create artifacts"
    );
    CONFIG_OVERRIDE.with(|value| *value.borrow_mut() = Some(Some(config.clone())));
    let second = super::super::run_quality_gate_with_cache(
        &input,
        &mut |_| panic!("cached visual capture must not request a source"),
        &mut cache,
    );
    assert_eq!(second.timing.owner_source_decode_count, 0);
    assert_eq!(
        serde_json::to_value(&first.criteria).unwrap(),
        serde_json::to_value(&second.criteria).unwrap()
    );
    assert_eq!(first.verdict, second.verdict);
    assert_eq!(output, original_output);
    assert!(ownership.iter().all(|owner| *owner == 1));
    assert!(coverage.as_raw().iter().all(|value| *value == 255));
    let process = config
        .directory
        .join(format!("process-{}", std::process::id()));
    let latest: Value =
        serde_json::from_slice(&fs::read(process.join("latest.json")).unwrap()).unwrap();
    let metadata: Value =
        serde_json::from_slice(&fs::read(latest["metadata"].as_str().unwrap()).unwrap()).unwrap();
    assert_eq!(metadata["captured_rois"][0]["reference_cache_hit"], true);
    assert_eq!(metadata["source_loader_calls_added"], 0);
    fs::remove_dir_all(config.directory).unwrap();
}
