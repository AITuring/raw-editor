use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::Cursor;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use image::codecs::jpeg::JpegDecoder;
use image::{DynamicImage, GenericImageView, ImageDecoder};
use rawler::decoders::RawDecodeParams;
use rawler::imgop::develop::{DemosaicAlgorithm, Intermediate, ProcessingStep, RawDevelop};
use rawler::rawimage::{RawImage, RawImageData, RawPhotometricInterpretation};
use rawler::rawsource::RawSource;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::develop_raw_image;

const PROBE_EDGE: usize = 512;
const PREVIEW_EDGE: u32 = 1920;
const JPEG_QUALITY: u8 = 92;
const MAX_SAMPLED_PIXELS: usize = 262_144;
const SONY_A7_RV_SCENARIOS: [&str; 8] = [
    "daylight",
    "tungsten",
    "high-iso",
    "underexposed",
    "saturated-light",
    "lossy-arw",
    "lossless-l-arw",
    "uncompressed-arw",
];

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

fn run_fixture(fixture: &Fixture, fixture_root: &Path, output_dir: Option<&Path>) -> FixtureReport {
    let total_start = Instant::now();
    assert_safe_component(&fixture.id, "fixture id");
    assert_safe_component(&fixture.original_filename, "fixture filename");
    assert!(
        fixture.classification.real_camera_sample,
        "{} must be classified as a real-camera sample",
        fixture.id
    );
    if fixture.classification.image_quality_baseline {
        assert_eq!(
            fixture.camera.model, "ILCE-7RM5",
            "{} may only claim α7R V scenario coverage for an ILCE-7RM5 file",
            fixture.id
        );
        assert!(
            fixture
                .classification
                .image_quality_scope
                .as_deref()
                .is_some_and(|scope| !scope.trim().is_empty()),
            "{} image-quality baseline requires an explicit scope",
            fixture.id
        );
        assert!(
            !fixture.covered_sony_a7_rv_scenarios.is_empty(),
            "{} image-quality baseline must cover at least one named α7R V scenario",
            fixture.id
        );
    } else {
        assert!(
            fixture.covered_sony_a7_rv_scenarios.is_empty(),
            "{} cannot cover α7R V scenarios without a scoped image-quality baseline",
            fixture.id
        );
    }
    assert!(!fixture.authorization.scope.trim().is_empty());
    assert!(!fixture.authorization.permission.trim().is_empty());
    assert!(!fixture.source.trim().is_empty());
    assert!(
        !fixture
            .expected_raw_properties
            .compression
            .trim()
            .is_empty()
    );

    let fixture_path = fixture_root.join(&fixture.original_filename);
    let file_bytes = fs::read(&fixture_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
    assert_eq!(
        file_bytes.len() as u64,
        fixture.byte_size,
        "{} byte size changed",
        fixture.id
    );
    let source_sha256 = sha256_hex(&file_bytes);
    assert_eq!(
        source_sha256, fixture.sha256,
        "{} SHA-256 changed; refusing to decode an unmanifested file",
        fixture.id
    );

    let source = RawSource::new_from_slice(&file_bytes);
    let decoder = rawler::get_decoder(&source)
        .unwrap_or_else(|error| panic!("{} decoder selection failed: {error}", fixture.id));
    let unpack_start = Instant::now();
    let raw = decoder
        .raw_image(&source, &RawDecodeParams::default(), false)
        .unwrap_or_else(|error| panic!("{} RAW unpack failed: {error}", fixture.id));
    let raw_unpack_ms = elapsed_ms(unpack_start);

    assert_eq!(raw.make, fixture.camera.make, "{} camera make", fixture.id);
    assert_eq!(
        raw.model, fixture.camera.model,
        "{} camera model",
        fixture.id
    );
    assert_eq!(
        Dimensions::from_usize(raw.width, raw.height),
        fixture.expected_sensor_dimensions,
        "{} decoded sensor mosaic dimensions",
        fixture.id
    );
    assert_eq!(
        raw.bps, fixture.expected_raw_properties.decoded_bits_per_sample,
        "{} decoded container bit depth",
        fixture.id
    );
    assert!(
        fixture.expected_raw_properties.source_bits_per_sample > 0,
        "{} source bit depth must be recorded",
        fixture.id
    );
    assert_eq!(
        raw.cpp, 1,
        "{} expected a single-plane CFA mosaic",
        fixture.id
    );

    let cfa = match &raw.photometric {
        RawPhotometricInterpretation::Cfa(config) => config.cfa.name.clone(),
        other => panic!(
            "{} expected CFA photometric data, got {other:?}",
            fixture.id
        ),
    };
    assert_eq!(
        cfa, fixture.expected_raw_properties.cfa,
        "{} CFA pattern",
        fixture.id
    );
    for coefficient in raw.wb_coeffs.iter().take(3) {
        assert!(
            coefficient.is_finite() && *coefficient > 0.0,
            "{} has invalid as-shot white balance {:?}",
            fixture.id,
            raw.wb_coeffs
        );
    }
    assert!(
        !raw.color_matrix.is_empty(),
        "{} has no camera color matrix",
        fixture.id
    );

    let raw_samples = summarize_raw_samples(&raw.data);
    let decoded_white_level = raw.whitelevel.0.first().copied().unwrap_or(u32::MAX);
    assert_source_bit_depth(
        fixture.expected_raw_properties.source_bits_per_sample,
        decoded_white_level,
        &fixture.id,
    );
    let raw_summary = RawSummary {
        dimensions: Dimensions::from_usize(raw.width, raw.height),
        source_bits_per_sample: fixture.expected_raw_properties.source_bits_per_sample,
        decoded_bits_per_sample: raw.bps,
        cfa,
        components_per_pixel: raw.cpp,
        black_level: raw
            .blacklevel
            .levels
            .first()
            .map(|level| level.as_f32())
            .unwrap_or(0.0),
        white_level: decoded_white_level,
        white_balance_rgb_coefficients: [raw.wb_coeffs[0], raw.wb_coeffs[1], raw.wb_coeffs[2]],
        color_matrix_count: raw.color_matrix.len(),
        sampled_mosaic_sha256: raw_samples.sha256,
        sampled_mosaic_values: raw_samples.values,
        sampled_mosaic_min: raw_samples.min,
        sampled_mosaic_max: raw_samples.max,
        sampled_mosaic_mean: raw_samples.mean,
    };

    let (probe, probe_origin) = centered_cfa_probe(&raw, PROBE_EDGE);
    let node_probe = run_node_probe(&fixture.id, &probe, probe_origin);
    drop(probe);
    drop(raw);

    let full_develop_start = Instant::now();
    let mut full_image = develop_raw_image(&file_bytes, false, 2.5, "auto".to_string(), None)
        .unwrap_or_else(|error| {
            panic!("{} full production development failed: {error}", fixture.id)
        });
    let full_develop_ms = elapsed_ms(full_develop_start);
    assert_eq!(
        Dimensions::from_u32(full_image.dimensions()),
        fixture.expected_output_dimensions,
        "{} full developed output dimensions",
        fixture.id
    );
    let linear_full_summary = summarize_image(&full_image, &format!("{} linear full", fixture.id));

    let display_start = Instant::now();
    crate::image_processing::apply_cpu_default_raw_processing(&mut full_image);
    let display_transform_ms = elapsed_ms(display_start);
    let display_full_summary =
        summarize_image(&full_image, &format!("{} display full", fixture.id));
    assert_unit_display_range(&display_full_summary, &fixture.id);

    let preview_source_start = Instant::now();
    let preview_image =
        crate::image_processing::downscale_f32_image(&full_image, PREVIEW_EDGE, PREVIEW_EDGE);
    let preview_source_ms = elapsed_ms(preview_source_start);
    let expected_preview_dimensions = fitted_dimensions(
        fixture.expected_output_dimensions.width,
        fixture.expected_output_dimensions.height,
        PREVIEW_EDGE,
    );
    assert_eq!(
        Dimensions::from_u32(preview_image.dimensions()),
        expected_preview_dimensions,
        "{} preview dimensions",
        fixture.id
    );
    let preview_summary = summarize_image(&preview_image, &format!("{} preview", fixture.id));
    assert_unit_display_range(&preview_summary, &fixture.id);
    let channel_mean_delta = channel_mean_max_delta(
        &display_full_summary.channel_means,
        &preview_summary.channel_means,
    );
    assert!(
        channel_mean_delta < 0.20,
        "{} preview/full channel means diverged by {channel_mean_delta:.4}",
        fixture.id
    );

    let preview_encode_start = Instant::now();
    let preview_bytes =
        crate::export_processing::encode_image_to_bytes(&preview_image, "jpeg", JPEG_QUALITY)
            .unwrap_or_else(|error| panic!("{} preview JPEG encode failed: {error}", fixture.id));
    let preview_encode_ms = elapsed_ms(preview_encode_start);
    let (preview_dimensions, preview_icc_bytes) =
        inspect_jpeg(&preview_bytes, &format!("{} preview", fixture.id));
    assert_eq!(preview_dimensions, expected_preview_dimensions);

    let export_encode_start = Instant::now();
    let export_bytes =
        crate::export_processing::encode_image_to_bytes(&full_image, "jpeg", JPEG_QUALITY)
            .unwrap_or_else(|error| {
                panic!("{} full-resolution JPEG export failed: {error}", fixture.id)
            });
    let export_encode_ms = elapsed_ms(export_encode_start);
    let (export_dimensions, export_icc_bytes) =
        inspect_jpeg(&export_bytes, &format!("{} full export", fixture.id));
    assert_eq!(export_dimensions, fixture.expected_output_dimensions);

    let artifacts = output_dir.map(|directory| {
        let preview_filename = format!("{}-preview.jpg", fixture.id);
        let full_resolution_export_filename = format!("{}-full.jpg", fixture.id);
        fs::write(directory.join(&preview_filename), &preview_bytes)
            .expect("write acceptance preview");
        fs::write(
            directory.join(&full_resolution_export_filename),
            &export_bytes,
        )
        .expect("write acceptance full-resolution export");
        ArtifactReport {
            preview_filename,
            full_resolution_export_filename,
        }
    });

    FixtureReport {
        id: fixture.id.clone(),
        camera: fixture.camera.clone(),
        capture_purpose: fixture.capture_purpose.clone(),
        source: fixture.source.clone(),
        authorization: fixture.authorization.clone(),
        compression: fixture.expected_raw_properties.compression.clone(),
        image_quality_baseline: fixture.classification.image_quality_baseline,
        image_quality_scope: fixture.classification.image_quality_scope.clone(),
        covered_sony_a7_rv_scenarios: fixture.covered_sony_a7_rv_scenarios.clone(),
        performance_scope: fixture.classification.performance_baseline.clone(),
        file_bytes: file_bytes.len() as u64,
        source_sha256,
        raw_unpack_ms,
        raw: raw_summary,
        node_probe,
        full_develop: TimedImageSummary {
            elapsed_ms: full_develop_ms,
            output: linear_full_summary,
        },
        display_transform_ms,
        preview: EncodedImageReport {
            dimensions: preview_dimensions,
            source: preview_summary,
            source_preparation_ms: preview_source_ms,
            encode_ms: preview_encode_ms,
            encoded_bytes: preview_bytes.len(),
            encoded_sha256: sha256_hex(&preview_bytes),
            icc_bytes: preview_icc_bytes,
        },
        full_resolution_export: EncodedImageReport {
            dimensions: export_dimensions,
            source: display_full_summary,
            source_preparation_ms: 0.0,
            encode_ms: export_encode_ms,
            encoded_bytes: export_bytes.len(),
            encoded_sha256: sha256_hex(&export_bytes),
            icc_bytes: export_icc_bytes,
        },
        preview_full_channel_mean_max_delta: channel_mean_delta,
        total_ms: elapsed_ms(total_start),
        artifacts,
    }
}

fn run_node_probe(id: &str, probe: &RawImage, source_origin: Point) -> NodeProbeReport {
    let (rescale, rescale_ms) = develop_stage(probe, vec![ProcessingStep::Rescale], "rescale", id);
    let (demosaic, demosaic_ms) = develop_stage(
        probe,
        vec![ProcessingStep::Rescale, ProcessingStep::Demosaic],
        "demosaic",
        id,
    );
    let (neutral_color, neutral_color_ms) = develop_stage(
        probe,
        vec![
            ProcessingStep::Rescale,
            ProcessingStep::Demosaic,
            ProcessingStep::Calibrate,
        ],
        "neutral-WB color transform",
        id,
    );
    let (as_shot_color, as_shot_color_ms) = develop_stage(
        probe,
        vec![
            ProcessingStep::Rescale,
            ProcessingStep::Demosaic,
            ProcessingStep::WhiteBalance,
            ProcessingStep::Calibrate,
        ],
        "as-shot-WB color transform",
        id,
    );

    assert_eq!(
        stage_channels(&rescale),
        1,
        "{id} rescale must preserve the mosaic"
    );
    for (stage, name) in [
        (&demosaic, "demosaic"),
        (&neutral_color, "neutral color transform"),
        (&as_shot_color, "as-shot color transform"),
    ] {
        assert_eq!(
            stage_channels(stage),
            3,
            "{id} {name} must produce three channels"
        );
    }

    let color_transform_delta = mean_absolute_delta(&demosaic, &neutral_color);
    let white_balance_delta = mean_absolute_delta(&neutral_color, &as_shot_color);
    assert!(
        color_transform_delta > 1.0e-5,
        "{id} camera-to-RGB color transform had no measurable effect"
    );
    assert!(
        white_balance_delta > 1.0e-5,
        "{id} as-shot white balance had no measurable effect"
    );

    NodeProbeReport {
        source_origin,
        source_dimensions: Dimensions::from_usize(probe.width, probe.height),
        rescale: TimedStageSummary {
            elapsed_ms: rescale_ms,
            output: summarize_stage(&rescale, &format!("{id} rescale")),
        },
        demosaic_camera_rgb: TimedStageSummary {
            elapsed_ms: demosaic_ms,
            output: summarize_stage(&demosaic, &format!("{id} demosaic")),
        },
        neutral_white_balance_color_transform: TimedStageSummary {
            elapsed_ms: neutral_color_ms,
            output: summarize_stage(&neutral_color, &format!("{id} neutral color")),
        },
        as_shot_white_balance_color_transform: TimedStageSummary {
            elapsed_ms: as_shot_color_ms,
            output: summarize_stage(&as_shot_color, &format!("{id} as-shot color")),
        },
        color_transform_mean_absolute_delta: color_transform_delta,
        white_balance_mean_absolute_delta: white_balance_delta,
    }
}

fn develop_stage(
    raw: &RawImage,
    steps: Vec<ProcessingStep>,
    stage_name: &str,
    id: &str,
) -> (Intermediate, f64) {
    let start = Instant::now();
    let output = RawDevelop {
        steps,
        demosaic_algorithm: DemosaicAlgorithm::Quality,
    }
    .develop_intermediate(raw)
    .unwrap_or_else(|error| panic!("{id} {stage_name} failed: {error}"));
    (output, elapsed_ms(start))
}

fn centered_cfa_probe(raw: &RawImage, requested_edge: usize) -> (RawImage, Point) {
    let (period_x, period_y) = match &raw.photometric {
        RawPhotometricInterpretation::Cfa(config) => {
            (config.cfa.width.max(1), config.cfa.height.max(1))
        }
        _ => (1, 1),
    };
    let probe_width = aligned_length(requested_edge.min(raw.width), period_x);
    let probe_height = aligned_length(requested_edge.min(raw.height), period_y);
    let origin_x = ((raw.width - probe_width) / 2 / period_x) * period_x;
    let origin_y = ((raw.height - probe_height) / 2 / period_y) * period_y;
    let row_values = probe_width * raw.cpp;

    let data = match &raw.data {
        RawImageData::Integer(values) => {
            let mut cropped = Vec::with_capacity(probe_width * probe_height * raw.cpp);
            for y in origin_y..origin_y + probe_height {
                let start = (y * raw.width + origin_x) * raw.cpp;
                cropped.extend_from_slice(&values[start..start + row_values]);
            }
            RawImageData::Integer(cropped)
        }
        RawImageData::Float(values) => {
            let mut cropped = Vec::with_capacity(probe_width * probe_height * raw.cpp);
            for y in origin_y..origin_y + probe_height {
                let start = (y * raw.width + origin_x) * raw.cpp;
                cropped.extend_from_slice(&values[start..start + row_values]);
            }
            RawImageData::Float(cropped)
        }
    };

    let probe = RawImage {
        camera: raw.camera.clone(),
        make: raw.make.clone(),
        model: raw.model.clone(),
        clean_make: raw.clean_make.clone(),
        clean_model: raw.clean_model.clone(),
        width: probe_width,
        height: probe_height,
        cpp: raw.cpp,
        bps: raw.bps,
        wb_coeffs: raw.wb_coeffs,
        whitelevel: raw.whitelevel.clone(),
        blacklevel: raw.blacklevel.clone(),
        xyz_to_cam: raw.xyz_to_cam,
        photometric: raw.photometric.clone(),
        active_area: None,
        crop_area: None,
        blackareas: Vec::new(),
        orientation: raw.orientation,
        data,
        color_matrix: raw.color_matrix.clone(),
        dng_tags: raw.dng_tags.clone(),
    };

    (
        probe,
        Point {
            x: origin_x as u32,
            y: origin_y as u32,
        },
    )
}

fn aligned_length(length: usize, period: usize) -> usize {
    let aligned = length - length % period;
    assert!(aligned > 0, "CFA-aligned probe must not be empty");
    aligned
}

fn summarize_raw_samples(data: &RawImageData) -> RawSampleSummary {
    match data {
        RawImageData::Integer(values) => {
            let stride = sampling_stride(values.len());
            let mut hasher = Sha256::new();
            let mut count = 0usize;
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0f64;
            for value in values.iter().step_by(stride) {
                hasher.update(value.to_le_bytes());
                let value = *value as f32;
                min = min.min(value);
                max = max.max(value);
                sum += value as f64;
                count += 1;
            }
            RawSampleSummary {
                sha256: hex::encode(hasher.finalize()),
                values: count,
                min,
                max,
                mean: sum / count as f64,
            }
        }
        RawImageData::Float(values) => {
            let stride = sampling_stride(values.len());
            let mut hasher = Sha256::new();
            let mut count = 0usize;
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0f64;
            for value in values.iter().step_by(stride) {
                assert!(value.is_finite(), "RAW mosaic contains a non-finite value");
                hasher.update(value.to_bits().to_le_bytes());
                min = min.min(*value);
                max = max.max(*value);
                sum += *value as f64;
                count += 1;
            }
            RawSampleSummary {
                sha256: hex::encode(hasher.finalize()),
                values: count,
                min,
                max,
                mean: sum / count as f64,
            }
        }
    }
}

fn summarize_stage(stage: &Intermediate, label: &str) -> StageSummary {
    let dimensions = stage.dim();
    let channels = stage_channels(stage);
    let values = stage_values(stage);
    let mut hasher = Sha256::new();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut channel_sums = vec![0.0f64; channels];
    for (index, value) in values.iter().enumerate() {
        assert!(value.is_finite(), "{label} contains a non-finite value");
        hasher.update(value.to_bits().to_le_bytes());
        min = min.min(*value);
        max = max.max(*value);
        channel_sums[index % channels] += *value as f64;
    }
    let pixels = values.len() / channels;
    StageSummary {
        dimensions: Dimensions::from_usize(dimensions.w, dimensions.h),
        channels,
        sha256: hex::encode(hasher.finalize()),
        min,
        max,
        channel_means: channel_sums
            .into_iter()
            .map(|sum| sum / pixels as f64)
            .collect(),
    }
}

fn stage_values(stage: &Intermediate) -> Vec<f32> {
    match stage {
        Intermediate::Monochrome(pixels) => pixels.data.clone(),
        Intermediate::ThreeColor(pixels) => pixels
            .data
            .iter()
            .flat_map(|pixel| pixel.iter().copied())
            .collect(),
        Intermediate::FourColor(pixels) => pixels
            .data
            .iter()
            .flat_map(|pixel| pixel.iter().copied())
            .collect(),
    }
}

fn stage_channels(stage: &Intermediate) -> usize {
    match stage {
        Intermediate::Monochrome(_) => 1,
        Intermediate::ThreeColor(_) => 3,
        Intermediate::FourColor(_) => 4,
    }
}

fn mean_absolute_delta(left: &Intermediate, right: &Intermediate) -> f64 {
    assert_eq!(left.dim(), right.dim(), "stage dimensions differ");
    assert_eq!(
        stage_channels(left),
        stage_channels(right),
        "stage channel counts differ"
    );
    let left = stage_values(left);
    let right = stage_values(right);
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / left.len() as f64
}

fn summarize_image(image: &DynamicImage, label: &str) -> ImageSummary {
    let dimensions = Dimensions::from_u32(image.dimensions());
    let total_pixels = dimensions.width as usize * dimensions.height as usize;
    let stride = sampling_stride(total_pixels);
    let mut hasher = Sha256::new();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut sums = [0.0f64; 3];
    let mut sampled_pixels = 0usize;

    let mut consume = |rgb: [f32; 3]| {
        for channel in 0..3 {
            assert!(
                rgb[channel].is_finite(),
                "{label} contains a non-finite value"
            );
            hasher.update(rgb[channel].to_bits().to_le_bytes());
            min[channel] = min[channel].min(rgb[channel]);
            max[channel] = max[channel].max(rgb[channel]);
            sums[channel] += rgb[channel] as f64;
        }
        sampled_pixels += 1;
    };

    match image {
        DynamicImage::ImageRgb32F(buffer) => {
            let values = buffer.as_raw();
            for pixel in (0..total_pixels).step_by(stride) {
                let offset = pixel * 3;
                consume([values[offset], values[offset + 1], values[offset + 2]]);
            }
        }
        DynamicImage::ImageRgba32F(buffer) => {
            let values = buffer.as_raw();
            for pixel in (0..total_pixels).step_by(stride) {
                let offset = pixel * 4;
                consume([values[offset], values[offset + 1], values[offset + 2]]);
            }
        }
        _ => {
            let converted = image.to_rgb32f();
            let values = converted.as_raw();
            for pixel in (0..total_pixels).step_by(stride) {
                let offset = pixel * 3;
                consume([values[offset], values[offset + 1], values[offset + 2]]);
            }
        }
    }

    ImageSummary {
        dimensions,
        sampled_pixels,
        sampled_pixel_sha256: hex::encode(hasher.finalize()),
        min,
        max,
        channel_means: sums.map(|sum| sum / sampled_pixels as f64),
    }
}

fn inspect_jpeg(bytes: &[u8], label: &str) -> (Dimensions, usize) {
    let mut decoder = JpegDecoder::new(Cursor::new(bytes))
        .unwrap_or_else(|error| panic!("{label} JPEG decode failed: {error}"));
    let dimensions = Dimensions::from_u32(decoder.dimensions());
    let profile = decoder
        .icc_profile()
        .unwrap_or_else(|error| panic!("{label} ICC read failed: {error}"))
        .unwrap_or_else(|| panic!("{label} must contain an ICC profile"));
    crate::color_management::validate_icc_profile(&profile)
        .unwrap_or_else(|error| panic!("{label} ICC profile is invalid: {error}"));
    assert_eq!(
        profile,
        crate::color_management::srgb_v4_profile(),
        "{label} embedded the wrong ICC profile"
    );
    (dimensions, profile.len())
}

fn assert_unit_display_range(summary: &ImageSummary, id: &str) {
    for channel in 0..3 {
        assert!(
            summary.min[channel] >= -f32::EPSILON && summary.max[channel] <= 1.0 + f32::EPSILON,
            "{id} display channel {channel} escaped [0, 1]: {}..{}",
            summary.min[channel],
            summary.max[channel]
        );
    }
}

fn validate_sony_a7_rv_scenario_coverage(manifest: &AcceptanceManifest) -> Vec<String> {
    let expected: HashSet<&str> = SONY_A7_RV_SCENARIOS.into_iter().collect();
    let deferred: HashSet<&str> = manifest
        .deferred_sony_a7_rv_scenarios
        .iter()
        .map(String::as_str)
        .collect();
    assert_eq!(
        deferred.len(),
        manifest.deferred_sony_a7_rv_scenarios.len(),
        "deferred α7R V scenarios must be unique"
    );

    let mut covered = HashSet::new();
    for fixture in &manifest.files {
        for scenario in &fixture.covered_sony_a7_rv_scenarios {
            assert!(
                expected.contains(scenario.as_str()),
                "{} names unknown α7R V scenario '{scenario}'",
                fixture.id
            );
            assert!(
                covered.insert(scenario.as_str()),
                "α7R V scenario '{scenario}' is covered by more than one fixture"
            );
        }
    }

    assert!(
        deferred.is_disjoint(&covered),
        "covered α7R V scenarios must not remain deferred"
    );
    let accounted_for: HashSet<&str> = deferred.union(&covered).copied().collect();
    assert_eq!(
        accounted_for, expected,
        "every α7R V acceptance scenario must be either covered or deferred"
    );

    let mut covered: Vec<String> = covered.into_iter().map(str::to_string).collect();
    covered.sort();
    covered
}

fn assert_source_bit_depth(source_bits: usize, white_level: u32, id: &str) {
    assert!(
        (1..32).contains(&source_bits),
        "{id} source bit depth {source_bits} cannot be represented safely"
    );
    let maximum_code = (1u32 << source_bits) - 1;
    assert!(
        white_level <= maximum_code && white_level > maximum_code / 2,
        "{id} white level {white_level} is inconsistent with the recorded {source_bits}-bit source"
    );
}

fn channel_mean_max_delta(left: &[f64; 3], right: &[f64; 3]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs() as f32)
        .fold(0.0, f32::max)
}

fn fitted_dimensions(width: u32, height: u32, max_edge: u32) -> Dimensions {
    if width <= max_edge && height <= max_edge {
        return Dimensions { width, height };
    }
    let ratio = (max_edge as f32 / width as f32).min(max_edge as f32 / height as f32);
    Dimensions {
        width: (width as f32 * ratio).round() as u32,
        height: (height as f32 * ratio).round() as u32,
    }
}

fn sampling_stride(values: usize) -> usize {
    values.div_ceil(MAX_SAMPLED_PIXELS).max(1)
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn assert_safe_component(value: &str, label: &str) {
    let mut components = Path::new(value).components();
    assert!(
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
        "{label} must be one safe path component"
    );
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
