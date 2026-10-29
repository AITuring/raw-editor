//! One manifest fixture through the RAW pipeline.
//!
//! Checks the file against its manifest entry, probes the CFA centre, develops the
//! full image and preview, encodes both as JPEG and collects the fixture report.

use super::super::develop_raw_image;
use image::GenericImageView;
use image::ImageDecoder;
use image::codecs::jpeg::JpegDecoder;
use rawler::decoders::RawDecodeParams;
use rawler::rawimage::RawPhotometricInterpretation;
use rawler::rawsource::RawSource;
use sha2::Digest;
use sha2::Sha256;
use std::fs;
use std::io::Cursor;
use std::path::Component;
use std::path::Path;
use std::time::Instant;

use super::{
    ArtifactReport, Dimensions, EncodedImageReport, Fixture, FixtureReport, ImageSummary,
    RawSummary, TimedImageSummary, centered_cfa_probe, elapsed_ms, run_node_probe, summarize_image,
    summarize_raw_samples,
};

pub(super) const PROBE_EDGE: usize = 512;

pub(super) const PREVIEW_EDGE: u32 = 1920;

pub(super) const JPEG_QUALITY: u8 = 92;

pub(super) fn run_fixture(
    fixture: &Fixture,
    fixture_root: &Path,
    output_dir: Option<&Path>,
) -> FixtureReport {
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

pub(super) fn inspect_jpeg(bytes: &[u8], label: &str) -> (Dimensions, usize) {
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

pub(super) fn assert_unit_display_range(summary: &ImageSummary, id: &str) {
    for channel in 0..3 {
        assert!(
            summary.min[channel] >= -f32::EPSILON && summary.max[channel] <= 1.0 + f32::EPSILON,
            "{id} display channel {channel} escaped [0, 1]: {}..{}",
            summary.min[channel],
            summary.max[channel]
        );
    }
}

pub(super) fn assert_source_bit_depth(source_bits: usize, white_level: u32, id: &str) {
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

pub(super) fn channel_mean_max_delta(left: &[f64; 3], right: &[f64; 3]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs() as f32)
        .fold(0.0, f32::max)
}

pub(super) fn fitted_dimensions(width: u32, height: u32, max_edge: u32) -> Dimensions {
    if width <= max_edge && height <= max_edge {
        return Dimensions { width, height };
    }
    let ratio = (max_edge as f32 / width as f32).min(max_edge as f32 / height as f32);
    Dimensions {
        width: (width as f32 * ratio).round() as u32,
        height: (height as f32 * ratio).round() as u32,
    }
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(super) fn assert_safe_component(value: &str, label: &str) {
    let mut components = Path::new(value).components();
    assert!(
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
        "{label} must be one safe path component"
    );
}
