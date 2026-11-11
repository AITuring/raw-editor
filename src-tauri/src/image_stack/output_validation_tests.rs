use super::*;
use image::{DynamicImage, ImageBuffer, Rgba};
use tiff::encoder::{TiffEncoder as RawTiffEncoder, colortype};
use tiff::tags::Tag;

fn rgba_fixture(path: &Path, extra_samples: Option<&[u16]>) {
    let mut encoder = RawTiffEncoder::new(File::create(path).unwrap()).unwrap();
    // Deliberately use the legacy four-sample colour type, so the missing-tag
    // regression does not depend on the production writer under test.
    let mut image = encoder.new_image::<colortype::RGBA16>(2, 1).unwrap();
    if let Some(extra) = extra_samples {
        image.encoder().write_tag(Tag::ExtraSamples, extra).unwrap();
    }
    image
        .write_data(&[12_u16, 34, 56, 0, 789, 1023, 2345, u16::MAX])
        .unwrap();
}

#[test]
fn tiff_artifact_gate_rejects_missing_or_incorrect_alpha_declarations() {
    let directory = tempfile::tempdir().unwrap();
    for (name, extra) in [
        ("missing", None),
        ("unspecified", Some(&[0_u16][..])),
        ("associated", Some(&[1_u16][..])),
        ("two-extras", Some(&[2_u16, 2][..])),
    ] {
        let path = directory.path().join(format!("{name}.tiff"));
        rgba_fixture(&path, extra);
        assert!(
            validate_image_stack_output(
                &path,
                (2, 1),
                ImageStackOutputFormat::Tiff,
                16,
                true,
                false,
            )
            .is_err(),
            "{name} must fail even if a decoder infers RGBA from four samples"
        );
        assert!(
            read_image_stack_output_metadata(&path, ImageStackOutputFormat::Tiff).is_err(),
            "{name} must not be reported as preserved alpha"
        );
    }
}

#[test]
fn tiff_artifact_gate_distinguishes_explicit_alpha_from_rgb_only() {
    let directory = tempfile::tempdir().unwrap();
    let alpha_path = directory.path().join("declared.tiff");
    rgba_fixture(&alpha_path, Some(&[2]));
    let alpha = validate_image_stack_output(
        &alpha_path,
        (2, 1),
        ImageStackOutputFormat::Tiff,
        16,
        true,
        false,
    )
    .unwrap();
    assert_eq!(alpha.dimensions, (2, 1));
    assert_eq!(alpha.tiff_extra_samples, Some(vec![2]));
    assert!(alpha.alpha_preserved);

    let rgb_path = directory.path().join("rgb.tiff");
    RawTiffEncoder::new(File::create(&rgb_path).unwrap())
        .unwrap()
        .write_image::<colortype::RGB16>(2, 1, &[12_u16, 34, 56, 789, 1023, 2345])
        .unwrap();
    let rgb = validate_image_stack_output(
        &rgb_path,
        (2, 1),
        ImageStackOutputFormat::Tiff,
        16,
        false,
        false,
    )
    .unwrap();
    assert_eq!(rgb.tiff_extra_samples, Some(vec![]));
    assert!(!rgb.alpha_preserved);
    assert!(
        validate_image_stack_output(
            &rgb_path,
            (2, 1),
            ImageStackOutputFormat::Tiff,
            16,
            true,
            false,
        )
        .is_err()
    );
}

#[test]
fn native_artifact_export_reports_verified_tiff_and_preserves_quality_evidence() {
    use crate::panorama_utils::stack_pipeline::report::{
        OutputArtifactValidation, QualityGateVerdict, StackReport,
    };
    let directory = tempfile::tempdir().unwrap();
    let report_path = directory.path().join("report.json");
    let mut original = StackReport::default();
    original.run_id = "artifact-test".into();
    original.quality_gate.verdict = QualityGateVerdict::Pass;
    std::fs::write(&report_path, serde_json::to_vec(&original).unwrap()).unwrap();
    let image = DynamicImage::ImageRgba16(ImageBuffer::from_fn(3, 2, |x, y| {
        Rgba([123, 456, 789, if x == y { 0 } else { u16::MAX }])
    }));
    let output = directory.path().join("result.tiff");
    super::super::test_output::write_srgb_tiff_for_acceptance(&image, &output, Some(&report_path))
        .unwrap();
    let saved: StackReport = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert!(saved.output.written);
    assert_eq!(saved.output.format, "TIFF");
    assert_eq!((saved.output.width, saved.output.height), (3, 2));
    assert_eq!(saved.output.bit_depth, 16);
    assert_eq!(saved.output.icc, "sRGB");
    assert_eq!(saved.output.tiff_extra_samples, Some(vec![2]));
    assert_eq!(
        saved.output.artifact_validation,
        OutputArtifactValidation::Pass
    );
    assert_eq!(saved.quality_gate, original.quality_gate);
    assert_eq!(saved.fusion, original.fusion);
    assert_eq!(saved.degradation, original.degradation);
}

#[test]
fn native_artifact_export_failure_cannot_be_hidden_by_quality_gate_pass() {
    use crate::panorama_utils::stack_pipeline::report::{
        OutputArtifactValidation, QualityGateVerdict, StackReport,
    };
    let directory = tempfile::tempdir().unwrap();
    let report_path = directory.path().join("report.json");
    let mut report = StackReport::default();
    report.run_id = "artifact-failure-test".into();
    report.quality_gate.verdict = QualityGateVerdict::Pass;
    std::fs::write(&report_path, serde_json::to_vec(&report).unwrap()).unwrap();
    let blocked = directory.path().join("blocked");
    std::fs::write(&blocked, b"not a directory").unwrap();
    let image = DynamicImage::ImageRgba16(ImageBuffer::from_pixel(2, 2, Rgba([1, 2, 3, 0])));
    assert!(
        super::super::test_output::write_srgb_tiff_for_acceptance(
            &image,
            &blocked.join("result.tiff"),
            Some(&report_path),
        )
        .is_err()
    );
    let saved: StackReport = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert!(!saved.output.written);
    assert_eq!(
        saved.output.artifact_validation,
        OutputArtifactValidation::Fail
    );
    assert!(saved.output.artifact_validation_error.is_some());
    assert_eq!(saved.quality_gate.verdict, QualityGateVerdict::Pass);
}
