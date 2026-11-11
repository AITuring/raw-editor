use super::super::{Capture, Config};
use super::*;
use crate::panorama_utils::mosaic::{Field, NativeResidualDisplacement};
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stack_pipeline::quality_gate::SourceGeometry;
use crate::panorama_utils::stack_pipeline::quality_gate_runner::{
    NativeReferenceMap, NativeSourceWarp, QualitySource,
};
use crate::panorama_utils::stack_pipeline::report::QualityGateReport;
use crate::panorama_utils::stack_pipeline::residual_warp::ResidualWarp;
use crate::panorama_utils::stitching::{FocusSourceRefinement, FocusSourceSampling, Projection};
use image::{GrayImage, Luma, Rgb, Rgb32FImage};
use nalgebra::Matrix3;

struct Fixture {
    output: Rgb32FImage,
    coverage: GrayImage,
    ownership: Vec<u16>,
    confidence: Vec<f32>,
    sources: Vec<QualitySource>,
    residual: ResidualWarp,
}

impl Fixture {
    fn new() -> Self {
        let station_to_world = Matrix3::new(1.0, 0.0, 100.0, 0.0, 1.0, 200.0, 0.0, 0.0, 1.0);
        let mut sources: Vec<_> = (1..=2)
            .map(|owner| QualitySource {
                owner,
                path: format!("source-{owner}.nef"),
                dimensions: (64, 64),
                geometry: SourceGeometry {
                    member_to_anchor: Matrix3::identity(),
                    tile_to_world: station_to_world,
                    station_id: 7,
                },
                native_warp: None,
                native_registration: None,
            })
            .collect();
        let info = ImageInfo {
            id: 2,
            filename: "source-2.nef".into(),
            width: 64,
            height: 64,
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
        };
        let mut field = Field::<2>::new(64, 64, 8.0);
        field.values.fill([0.25, -0.5]);
        sources[1].native_registration = Some(NativeReferenceMap {
            mosaic_tone: None,
            source_prefilter_divisor: 1.0,
            world_to_station: station_to_world.try_inverse().unwrap(),
            station_to_world,
            station_origin: (4.0, 5.0),
            refinement: FocusSourceRefinement {
                translation: (1, -2),
                displacement: Some(NativeResidualDisplacement::from_test_field(field)),
                source_sampling: Some(FocusSourceSampling::for_test(
                    &info,
                    &Matrix3::new(1.0, 0.0, 1.5, 0.0, 1.0, 2.25, 0.0, 0.0, 1.0),
                    Projection::Planar,
                    None,
                )),
                ..Default::default()
            },
        });
        // The direct registration must take priority over legacy provenance,
        // exactly as source_roi_for_quality does for the real QG reference.
        sources[1].native_warp = Some(NativeSourceWarp {
            displacement: None,
            source_to_station: Matrix3::identity(),
            station_to_world: Matrix3::identity(),
        });
        let mut ownership = vec![1; 12];
        ownership[5] = 2;
        ownership[11] = 0;
        let mut coverage = GrayImage::from_pixel(4, 3, Luma([255]));
        coverage.put_pixel(3, 2, Luma([0]));
        let mut output =
            Rgb32FImage::from_fn(4, 3, |x, y| Rgb([x as f32 * 0.125, y as f32 * 0.25, -0.0]));
        output.put_pixel(1, 1, Rgb([0.0; 3]));
        Self {
            output,
            coverage,
            ownership,
            confidence: vec![1.0; 12],
            sources,
            residual: ResidualWarp::new(),
        }
    }

    fn input(&self) -> QualityGateInput<'_> {
        QualityGateInput {
            output: &self.output,
            coverage: &self.coverage,
            ownership: &self.ownership,
            confidence: &self.confidence,
            textured: None,
            owner_shortfall: None,
            owner_disagreement: None,
            unresolved_pixel_count: 0,
            world_origin: (100.0, 200.0),
            sources: &self.sources,
            accepted_station_relations: &[],
            residual: &self.residual,
            acceptance_render_scale: 1.0,
            final_sharpen_amount: 0.0,
        }
    }
}

fn request() -> Value {
    json!({"schema": 1, "reference_canvas_origin": [100.0, 200.0], "points": [
        {"label": "actual-minority-owner", "role": "known_hole", "canvas": [1, 1]},
        {"label": "opaque-control", "role": "opaque_control", "canvas": [2, 1]},
        {"label": "exterior-control", "role": "exterior_control", "canvas": [3, 2]},
    ]})
}

fn observe_request(request: &Value, fixture: &Fixture) -> (Value, Vec<String>) {
    observe(
        &serde_json::to_vec(request).unwrap(),
        Path::new("fixture-points.json"),
        &fixture.input(),
        9,
        "unit-test",
    )
    .unwrap()
}

struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("qg-point-evidence-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct InputOverride;
impl InputOverride {
    fn set(path: Option<PathBuf>) -> Self {
        INPUT_OVERRIDE.with(|value| {
            assert!(value.borrow().is_none());
            *value.borrow_mut() = Some(path);
        });
        Self
    }
}
impl Drop for InputOverride {
    fn drop(&mut self) {
        INPUT_OVERRIDE.with(|value| *value.borrow_mut() = None);
    }
}

#[test]
fn source_visual_point_evidence_reads_actual_raw_owner_and_native_reference_without_mutation() {
    let fixture = Fixture::new();
    let before_rgb: Vec<_> = fixture
        .output
        .as_raw()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let before_owners = fixture.ownership.clone();
    let before_coverage = fixture.coverage.clone();
    let bytes = serde_json::to_vec(&request()).unwrap();
    let (ledger, errors) = observe(
        &bytes,
        Path::new("fixture-points.json"),
        &fixture.input(),
        9,
        "unit-test",
    )
    .unwrap();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(ledger["status"], "captured");
    assert_eq!(ledger["input_sha256"], digest(&bytes));
    assert_eq!(ledger["iteration"], 9);
    assert!(ledger["final_iteration"].is_null());
    assert_eq!(ledger["source_loader_calls_added"], 0);
    let point = &ledger["points"][0];
    assert_eq!(point["point_id"], 0);
    assert_eq!(
        point["actual_raw_owner"], 2,
        "owner 1 dominates the census but cannot replace the actual point owner"
    );
    assert_eq!(point["path"], "source-2.nef");
    assert_eq!(point["station_id"], 7);
    assert_eq!(point["world"], json!([101.0, 201.0]));
    assert_eq!(point["canvas"], json!([1, 1]));
    assert_eq!(point["prepared_source_xy"], json!([2.75, 5.25]));
    assert_eq!(point["output_rgb_f32"], json!([0.0, 0.0, 0.0]));
    assert_eq!(
        point["status"], "covered",
        "black does not erase actual ownership evidence"
    );
    assert_eq!(ledger["points"][1]["prepared_source_xy"], json!([2.0, 1.0]));
    let exterior = &ledger["points"][2];
    assert_eq!(exterior["status"], "uncovered");
    assert_eq!(exterior["actual_raw_owner"], 0);
    assert_eq!(exterior["coverage_u8"], 0);
    assert!(exterior["prepared_source_xy"].is_null() && exterior["path"].is_null());
    assert_eq!(
        fixture
            .output
            .as_raw()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        before_rgb
    );
    assert_eq!(fixture.ownership, before_owners);
    assert_eq!(fixture.coverage, before_coverage);
}

#[test]
fn source_visual_point_evidence_rejects_future_schema_limits_and_invalid_requests() {
    let mut request = request();
    request["schema"] = json!(2);
    assert!(parse_request(&serde_json::to_vec(&request).unwrap()).is_err());
    request["schema"] = json!(1);
    request["points"] = json!([]);
    assert!(parse_request(&serde_json::to_vec(&request).unwrap()).is_err());
    request["points"] = json!(
        (0..129)
            .map(|x| json!({"label": format!("p{x}"), "role":"control", "canvas":[x,0]}))
            .collect::<Vec<_>>()
    );
    assert!(parse_request(&serde_json::to_vec(&request).unwrap()).is_err());
    request["points"].as_array_mut().unwrap().pop();
    assert!(parse_request(&serde_json::to_vec(&request).unwrap()).is_ok());
    request["points"] = json!([{"label":"a","role":"control","canvas":[0,0]},{"label":"b","role":"control","canvas":[0,0]}]);
    assert!(parse_request(&serde_json::to_vec(&request).unwrap()).is_err());
    request["points"] = json!([{"label":"","role":"control","canvas":[0,0]}]);
    assert!(parse_request(&serde_json::to_vec(&request).unwrap()).is_err());
    assert!(parse_request(&vec![b' '; MAX_INPUT_BYTES + 1]).is_err());
    assert!(
        parse_request(br#"{"schema":1,"reference_canvas_origin":[1e400,0],"points":[]}"#).is_err()
    );
}

#[test]
fn source_visual_point_evidence_disabled_path_does_not_read_or_validate_planes() {
    let fixture = Fixture::new();
    let mut input = fixture.input();
    input.ownership = &[]; // Deliberately unusable: OFF must return before access.
    assert!(
        capture_path(
            None,
            &input,
            1,
            "unit-test",
            Path::new("/does-not-exist/qg-evidence")
        )
        .unwrap()
        .is_none()
    );
    let bytes = serde_json::to_vec(&request()).unwrap();
    assert!(observe(&bytes, Path::new("input"), &input, 1, "unit-test").is_err());
}

#[test]
fn source_visual_point_evidence_marks_owner_mapping_and_grid_failures_without_fake_coordinates() {
    for (case, expected) in [
        (0, "covered_owner_zero"),
        (1, "owner_missing_or_duplicate"),
        (2, "owner_missing_or_duplicate"),
        (3, "singular_source_geometry"),
        (4, "mapping_failed"),
        (5, "out_of_bounds"),
    ] {
        let mut fixture = Fixture::new();
        match case {
            0 => fixture.ownership[5] = 0,
            1 => {
                fixture.sources.pop();
            }
            2 => fixture.sources.push(fixture.sources[1].clone()),
            3 => fixture.sources[1].geometry.tile_to_world = Matrix3::zeros(),
            4 => {
                fixture.sources[1]
                    .native_registration
                    .as_mut()
                    .unwrap()
                    .refinement
                    .translation = (1000, 1000)
            }
            5 => fixture.sources[1].dimensions = (1, 1),
            _ => unreachable!(),
        }
        let (ledger, errors) = observe_request(&request(), &fixture);
        assert!(!errors.is_empty(), "case {case}");
        assert_eq!(ledger["status"], "failed");
        assert_eq!(
            ledger["points"][0]["reference_mapping_status"], expected,
            "case {case}"
        );
        assert_eq!(ledger["points"][0]["status"], "invalid");
        if case != 5 {
            assert!(ledger["points"][0]["prepared_source_xy"].is_null());
        }
    }
    assert!(
        canvas_pixel([3.999_999_9, 1.0], (0.0, 0.0), (4, 3)).is_none(),
        "rounding may not step beyond the right boundary"
    );
    assert!(canvas_pixel([1.25, 1.0], (0.0, 0.0), (4, 3)).is_none());
    let mut invalid = request();
    invalid["points"][0]["canvas"] = json!([4, 0]);
    let (ledger, errors) = observe_request(&invalid, &Fixture::new());
    assert!(!errors.is_empty());
    assert!(
        ledger["points"][0]["actual_raw_owner"].is_null(),
        "outside output cannot fabricate owner zero evidence"
    );
}

#[test]
fn source_visual_point_evidence_persists_hash_and_invalid_input_supersedes_completed_capture() {
    let directory = Temporary::new();
    let path = directory.0.join("points.json");
    fs::write(&path, serde_json::to_vec(&request()).unwrap()).unwrap();
    let _override = InputOverride::set(Some(path.clone()));
    let fixture = Fixture::new();
    let config = Config {
        directory: directory.0.join("capture"),
        profile: "unit-test".into(),
        limit: 1,
    };
    let mut first = Capture::begin(Some(config.clone()), &fixture.input()).unwrap();
    assert!(first.errors.is_empty());
    let record = &first.metadata["native_point_evidence"];
    assert_eq!(record["status"], "captured");
    let bytes = fs::read(record["path"].as_str().unwrap()).unwrap();
    assert_eq!(record["sha256"], digest(&bytes));
    assert_eq!(record["point_count"], 3);
    let ledger: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(ledger["points"][0]["actual_raw_owner"], 2);
    first.complete(&QualityGateReport::default()).unwrap();
    fs::write(
        &path,
        br#"{"schema":2,"reference_canvas_origin":[100,200],"points":[]}"#,
    )
    .unwrap();
    let mut second = Capture::begin(Some(config), &fixture.input()).unwrap();
    assert!(
        !second.errors.is_empty(),
        "invalid diagnostics must leave the QG run available but prohibit binding"
    );
    assert_eq!(second.metadata["native_point_evidence"]["status"], "failed");
    second.complete(&QualityGateReport::default()).unwrap();
    let latest: Value =
        serde_json::from_slice(&fs::read(second.process_directory.join("latest.json")).unwrap())
            .unwrap();
    assert!(second.iteration > first.iteration);
    assert_eq!(latest["latest_completed_iteration"], second.iteration);
    assert!(!latest["capture_errors"].as_array().unwrap().is_empty());
    assert!(latest["final_iteration"].is_null());
}

#[test]
fn source_visual_point_evidence_write_failure_is_explicit_and_input_read_is_bounded() {
    let directory = Temporary::new();
    let path = directory.0.join("points.json");
    fs::write(&path, serde_json::to_vec(&request()).unwrap()).unwrap();
    fs::create_dir(directory.0.join("point-evidence.json")).unwrap();
    let fixture = Fixture::new();
    assert!(
        capture_path(
            Some(path.clone()),
            &fixture.input(),
            1,
            "unit-test",
            &directory.0
        )
        .is_err()
    );
    fs::write(&path, vec![b' '; MAX_INPUT_BYTES + 4096]).unwrap();
    let error =
        capture_path(Some(path), &fixture.input(), 1, "unit-test", &directory.0).unwrap_err();
    assert!(error.contains("65536"));
}
