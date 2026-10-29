use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AcceptanceManifest {
    schema_version: u32,
    fixture_root_environment: String,
    files: Vec<Fixture>,
    #[serde(rename = "deferredSonyA7RVScenarios")]
    deferred_sony_a7_rv_scenarios: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Fixture {
    id: String,
    camera: CameraIdentity,
    capture_purpose: String,
    original_filename: String,
    byte_size: u64,
    sha256: String,
    source: String,
    authorization: Authorization,
    expected_sensor_dimensions: Dimensions,
    expected_output_dimensions: Dimensions,
    expected_raw_properties: ExpectedRawProperties,
    classification: Classification,
    #[serde(default, rename = "coveredSonyA7RVScenarios")]
    covered_sony_a7_rv_scenarios: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CameraIdentity {
    make: String,
    model: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Authorization {
    scope: String,
    permission: String,
    redistribution: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
struct Dimensions {
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExpectedRawProperties {
    source_bits_per_sample: usize,
    decoded_bits_per_sample: usize,
    cfa: String,
    compression: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Classification {
    real_camera_sample: bool,
    image_quality_baseline: bool,
    #[serde(default)]
    image_quality_scope: Option<String>,
    performance_baseline: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AcceptanceReport {
    schema_version: u32,
    generated_at: String,
    environment: RunEnvironment,
    #[serde(rename = "sonyA7RVBaselineDeferred")]
    sony_a7_rv_baseline_deferred: bool,
    #[serde(rename = "coveredSonyA7RVScenarios")]
    covered_sony_a7_rv_scenarios: Vec<String>,
    #[serde(rename = "deferredSonyA7RVScenarios")]
    deferred_sony_a7_rv_scenarios: Vec<String>,
    fixtures: Vec<FixtureReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RunEnvironment {
    os: &'static str,
    architecture: &'static str,
    cargo_profile: &'static str,
    logical_cpus: usize,
    available_memory_bytes_at_start: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FixtureReport {
    id: String,
    camera: CameraIdentity,
    capture_purpose: String,
    source: String,
    authorization: Authorization,
    compression: String,
    image_quality_baseline: bool,
    image_quality_scope: Option<String>,
    #[serde(rename = "coveredSonyA7RVScenarios")]
    covered_sony_a7_rv_scenarios: Vec<String>,
    performance_scope: String,
    file_bytes: u64,
    source_sha256: String,
    raw_unpack_ms: f64,
    raw: RawSummary,
    node_probe: NodeProbeReport,
    full_develop: TimedImageSummary,
    display_transform_ms: f64,
    preview: EncodedImageReport,
    full_resolution_export: EncodedImageReport,
    preview_full_channel_mean_max_delta: f32,
    total_ms: f64,
    artifacts: Option<ArtifactReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RawSummary {
    dimensions: Dimensions,
    source_bits_per_sample: usize,
    decoded_bits_per_sample: usize,
    cfa: String,
    components_per_pixel: usize,
    black_level: f32,
    white_level: u32,
    white_balance_rgb_coefficients: [f32; 3],
    color_matrix_count: usize,
    sampled_mosaic_sha256: String,
    sampled_mosaic_values: usize,
    sampled_mosaic_min: f32,
    sampled_mosaic_max: f32,
    sampled_mosaic_mean: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeProbeReport {
    source_origin: Point,
    source_dimensions: Dimensions,
    rescale: TimedStageSummary,
    demosaic_camera_rgb: TimedStageSummary,
    neutral_white_balance_color_transform: TimedStageSummary,
    as_shot_white_balance_color_transform: TimedStageSummary,
    color_transform_mean_absolute_delta: f64,
    white_balance_mean_absolute_delta: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TimedStageSummary {
    elapsed_ms: f64,
    output: StageSummary,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StageSummary {
    dimensions: Dimensions,
    channels: usize,
    sha256: String,
    min: f32,
    max: f32,
    channel_means: Vec<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TimedImageSummary {
    elapsed_ms: f64,
    output: ImageSummary,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageSummary {
    dimensions: Dimensions,
    sampled_pixels: usize,
    sampled_pixel_sha256: String,
    min: [f32; 3],
    max: [f32; 3],
    channel_means: [f64; 3],
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EncodedImageReport {
    dimensions: Dimensions,
    source: ImageSummary,
    source_preparation_ms: f64,
    encode_ms: f64,
    encoded_bytes: usize,
    encoded_sha256: String,
    icc_bytes: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactReport {
    preview_filename: String,
    full_resolution_export_filename: String,
}

#[derive(Debug)]
struct RawSampleSummary {
    sha256: String,
    values: usize,
    min: f32,
    max: f32,
    mean: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct Point {
    x: u32,
    y: u32,
}

#[path = "raw_processing_acceptance/sampling.rs"]
mod sampling;

use sampling::{summarize_image, summarize_raw_samples};

#[path = "raw_processing_acceptance/node_probe.rs"]
mod node_probe;

use node_probe::{centered_cfa_probe, run_node_probe};

#[path = "raw_processing_acceptance/scenario_coverage.rs"]
mod scenario_coverage;

use scenario_coverage::validate_sony_a7_rv_scenario_coverage;

#[path = "raw_processing_acceptance/fixture_run.rs"]
mod fixture_run;

use fixture_run::run_fixture;

#[test]
#[ignore = "requires local licensed RAW files via RAW_EDITOR_ACCEPTANCE_DIR"]
fn local_real_raw_pipeline_acceptance() {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri must have a repository parent")
        .join("tests/acceptance/raw-manifest.json");
    let manifest_bytes = fs::read(&manifest_path).expect("read RAW acceptance manifest");
    let manifest: AcceptanceManifest =
        serde_json::from_slice(&manifest_bytes).expect("parse RAW acceptance manifest");

    assert_eq!(manifest.schema_version, 1, "unsupported manifest schema");
    assert!(
        !manifest.files.is_empty(),
        "the real-RAW manifest must contain at least one fixture"
    );
    let covered_sony_a7_rv_scenarios = validate_sony_a7_rv_scenario_coverage(&manifest);

    let fixture_root = PathBuf::from(env::var(&manifest.fixture_root_environment).unwrap_or_else(
        |_| {
            panic!(
                "set {} to the read-only directory containing the licensed RAW fixtures",
                manifest.fixture_root_environment
            )
        },
    ));
    assert!(
        fixture_root.is_dir(),
        "fixture root does not exist: {}",
        fixture_root.display()
    );

    let output_dir = env::var_os("RAW_EDITOR_ACCEPTANCE_OUTPUT_DIR").map(PathBuf::from);
    if let Some(path) = &output_dir {
        assert_output_is_not_inside_fixture_root(path, &fixture_root);
        fs::create_dir_all(path).expect("create RAW acceptance output directory");
    }

    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let environment = RunEnvironment {
        os: env::consts::OS,
        architecture: env::consts::ARCH,
        cargo_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        logical_cpus: std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1),
        available_memory_bytes_at_start: system.available_memory(),
    };

    let fixture_reports = manifest
        .files
        .iter()
        .map(|fixture| run_fixture(fixture, &fixture_root, output_dir.as_deref()))
        .collect();
    let report = AcceptanceReport {
        schema_version: 1,
        generated_at: chrono::Utc::now().to_rfc3339(),
        environment,
        sony_a7_rv_baseline_deferred: !manifest.deferred_sony_a7_rv_scenarios.is_empty(),
        covered_sony_a7_rv_scenarios,
        deferred_sony_a7_rv_scenarios: manifest.deferred_sony_a7_rv_scenarios.clone(),
        fixtures: fixture_reports,
    };
    let report_json =
        serde_json::to_string_pretty(&report).expect("serialize RAW acceptance report");

    if let Some(report_path) = env::var_os("RAW_EDITOR_ACCEPTANCE_REPORT").map(PathBuf::from) {
        assert_output_is_not_inside_fixture_root(&report_path, &fixture_root);
        if let Some(parent) = report_path.parent() {
            fs::create_dir_all(parent).expect("create RAW acceptance report directory");
        }
        fs::write(&report_path, &report_json).expect("write RAW acceptance report");
        println!("RAW acceptance report: {}", report_path.display());
    }

    println!("{report_json}");
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn assert_output_is_not_inside_fixture_root(output: &Path, fixture_root: &Path) {
    assert!(
        !output.starts_with(fixture_root),
        "acceptance output must not be written beside source photographs"
    );
}

impl Dimensions {
    fn from_u32((width, height): (u32, u32)) -> Self {
        Self { width, height }
    }

    fn from_usize(width: usize, height: usize) -> Self {
        Self {
            width: width as u32,
            height: height as u32,
        }
    }
}
