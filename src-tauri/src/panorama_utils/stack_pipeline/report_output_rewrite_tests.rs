use super::super::report_schema_metrics::OutputArtifactValidation;
use super::*;

type RawFields = BTreeMap<String, Box<RawValue>>;

fn read_raw(path: &Path) -> RawFields {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn difficult_report(path: &Path) -> RawFields {
    let mut report = StackReport::default();
    report.run_id = "raw-output-evidence".into();
    // These decimal values were observed drifting during the native v18
    // whole-report deserialize/serialize cycle. They must not be parsed by an
    // output-only update at all.
    report.closure.residual_p95_px = 1.9836787414205266;
    report.tone.boundary_delta_e.max = 0.9985258189234775;
    report.output.preview.max_roi_delta_e = 0.9985258189234775;
    let mut fields: RawFields = serde_json::from_str(&report.to_json_string().unwrap()).unwrap();
    fields.insert(
        "future_owner_evidence".into(),
        RawValue::from_string(
            r#"{ "ncc" : 0.9985258189234775, "scale": 0.9999999999999999, "wide": 18446744073709551616, "negative_zero": -0.000, "tiny": 1e-1000 }"#.into(),
        )
        .unwrap(),
    );
    let mut output_fields: RawFields = serde_json::from_str(fields["output"].get()).unwrap();
    output_fields.insert(
        "future_output_evidence".into(),
        RawValue::from_string(
            r#"{ "exact" : 2.2171894826109436e-07, "opaque" : [ 1, 2 ] }"#.into(),
        )
        .unwrap(),
    );
    fields.insert(
        "output".into(),
        serde_json::value::to_raw_value(&output_fields).unwrap(),
    );
    fs::write(path, serde_json::to_string_pretty(&fields).unwrap()).unwrap();
    read_raw(path)
}

fn assert_evidence_unchanged(path: &Path, original: &RawFields) -> OutputReport {
    let updated = read_raw(path);
    assert_eq!(
        updated.len(),
        original.len(),
        "unknown top-level fields must survive"
    );
    for (key, value) in original {
        if key != "output" {
            assert_eq!(
                updated[key].get(),
                value.get(),
                "raw evidence changed: {key}"
            );
        }
    }
    let old_output: RawFields = serde_json::from_str(original["output"].get()).unwrap();
    let new_output: RawFields = serde_json::from_str(updated["output"].get()).unwrap();
    for key in ["preview", "future_output_evidence"] {
        assert_eq!(
            new_output[key].get(),
            old_output[key].get(),
            "unchanged output evidence changed: {key}"
        );
    }
    serde_json::from_str(updated["output"].get()).unwrap()
}

fn record_success(report_path: &Path, output_path: &Path, result_id: &str) -> Result<(), String> {
    update_stack_report_output(
        report_path,
        result_id,
        "TIFF".into(),
        16,
        true,
        "sRGB".into(),
        output_path,
        (15701, 6585),
        Some(vec![2]),
    )
}

#[test]
fn output_rewrites_preserve_raw_non_output_evidence_and_unknown_fields() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("report.json");
    let original = difficult_report(&path);

    assign_stack_report_result_id(&path, "native-fixture").unwrap();
    let output = assert_evidence_unchanged(&path, &original);
    assert_eq!(output.result_id, "native-fixture");
    assert_eq!(output.artifact_validation, OutputArtifactValidation::NotRun);

    let output_path = directory.path().join("result.tiff");
    for _ in 0..3 {
        record_success(&path, &output_path, "native-fixture").unwrap();
        let output = assert_evidence_unchanged(&path, &original);
        assert!(output.written);
        assert_eq!(output.artifact_validation, OutputArtifactValidation::Pass);
        assert_eq!(output.tiff_extra_samples, Some(vec![2]));
        assert_eq!((output.width, output.height), (15701, 6585));

        record_stack_report_artifact_failure(&path, "native-fixture", "fixture failure").unwrap();
        let output = assert_evidence_unchanged(&path, &original);
        assert!(!output.written);
        assert_eq!(output.artifact_validation, OutputArtifactValidation::Fail);
        assert_eq!(
            output.artifact_validation_error.as_deref(),
            Some("fixture failure")
        );
    }
    record_success(&path, &output_path, "native-fixture").unwrap();
    let output = assert_evidence_unchanged(&path, &original);
    assert!(output.written);
    assert!(output.artifact_validation_error.is_none());
    assert_eq!(
        fs::read_dir(directory.path()).unwrap().count(),
        1,
        "no temporary report may remain"
    );
}

#[test]
fn rejected_output_rewrite_leaves_report_bytes_unchanged() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("report.json");
    let original = difficult_report(&path);
    assign_stack_report_result_id(&path, "expected-result").unwrap();
    let before = fs::read(&path).unwrap();
    assert!(assign_stack_report_result_id(&path, "wrong-result").is_err());
    assert!(record_success(&path, &directory.path().join("other.tiff"), "wrong-result").is_err());
    assert!(
        record_stack_report_artifact_failure(&path, "wrong-result", "must not replace").is_err()
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_evidence_unchanged(&path, &original);
}
