use super::*;

#[test]
fn exposure_time_formatter_removes_float_denominator_noise() {
    assert_eq!(
        format_exposure_time_seconds(1.0 / 1.6666666666666667),
        Some("0.6 s".to_string())
    );
    assert_eq!(
        format_exif_exposure_time(1, 200),
        Some("1/200 s".to_string())
    );
    assert_eq!(format_exif_exposure_time(5, 2), Some("2.5 s".to_string()));
    assert_eq!(format_exposure_time_seconds(0.0), None);
}

#[test]
fn sidecar_gps_parser_preserves_display_coordinates() {
    let coordinate = parse_sidecar_gps_coordinate("31 deg 14 min 15.5 sec")
        .expect("parse sidecar GPS coordinate");
    assert_eq!(coordinate.len(), 3);
    assert_eq!(coordinate[0].nominator, 31_000_000);
    assert_eq!(coordinate[1].nominator, 14_000_000);
    assert_eq!(coordinate[2].nominator, 15_500_000);
    assert_eq!(coordinate[2].denominator, 1_000_000);
}

#[test]
fn sidecar_schema_contract_matches_runtime_version() {
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../../../schemas/sidecar-v1.schema.json"))
            .expect("parse sidecar JSON schema");
    assert_eq!(
        schema["properties"]["version"]["const"].as_u64(),
        Some(SIDECAR_SCHEMA_VERSION as u64)
    );
    assert_eq!(
        schema["additionalProperties"],
        serde_json::Value::Bool(true)
    );
}

#[test]
fn migrates_legacy_sidecar_and_preserves_unknown_fields() {
    let directory = tempfile::tempdir().expect("create sidecar test directory");
    let path = directory.path().join("legacy.rrdata");
    fs::write(
        &path,
        br#"{"rating":2,"adjustments":{"exposure":1.25},"futureUi":{"panel":"local"}}"#,
    )
    .expect("write legacy sidecar");

    let metadata = load_sidecar_checked(&path).expect("migrate legacy sidecar");
    assert_eq!(metadata.version, SIDECAR_SCHEMA_VERSION);
    assert_eq!(metadata.rating, 2);
    assert_eq!(metadata.extra["futureUi"]["panel"], "local");

    save_sidecar(&path, &metadata).expect("round-trip migrated sidecar");
    let saved: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).expect("read migrated sidecar"))
            .expect("parse migrated sidecar");
    assert_eq!(saved["version"], SIDECAR_SCHEMA_VERSION);
    assert_eq!(saved["futureUi"]["panel"], "local");
}

#[test]
fn recovers_truncated_primary_from_last_known_good_backup() {
    let directory = tempfile::tempdir().expect("create sidecar test directory");
    let path = directory.path().join("primary.rrdata");

    let first = ImageMetadata {
        rating: 1,
        ..ImageMetadata::default()
    };
    save_sidecar(&path, &first).expect("save first revision");

    let mut second = first.clone();
    second.rating = 4;
    save_sidecar(&path, &second).expect("save second revision");
    assert!(sidecar_backup_path(&path).is_file());

    fs::write(&path, br#"{"version":1,"rating":"#).expect("simulate interrupted write");
    let recovered = load_sidecar_checked(&path).expect("recover from backup");
    assert_eq!(recovered.rating, 1);

    let restored = read_sidecar_document(&path)
        .expect("restored primary is valid")
        .0;
    assert_eq!(restored.rating, 1);
}

#[test]
fn refuses_to_read_or_overwrite_future_schema_versions() {
    let directory = tempfile::tempdir().expect("create sidecar test directory");
    let path = directory.path().join("future.rrdata");
    let future_document = br#"{"version":2,"rating":5,"adjustments":null,"futureOnly":true}"#;
    fs::write(&path, future_document).expect("write future sidecar");

    let load_error = load_sidecar_checked(&path).expect_err("future schema must be rejected");
    assert!(load_error.contains("newer than supported"));

    let save_error = save_sidecar(&path, &ImageMetadata::default())
        .expect_err("future schema must be protected");
    assert!(save_error.contains("refusing to overwrite"));
    assert_eq!(
        fs::read(&path).expect("read protected sidecar"),
        future_document
    );
}

#[test]
fn rejects_malformed_schema_versions_instead_of_treating_them_as_legacy() {
    let directory = tempfile::tempdir().expect("create sidecar test directory");
    let path = directory.path().join("malformed-version.rrdata");
    fs::write(&path, br#"{"version":"1","rating":2,"adjustments":null}"#)
        .expect("write malformed sidecar");

    let error = load_sidecar_checked(&path).expect_err("string version must be rejected");
    assert!(error.contains("non-negative integer"));
}

#[test]
fn invalid_rating_is_rejected_before_persistence() {
    let directory = tempfile::tempdir().expect("create sidecar test directory");
    let path = directory.path().join("invalid.rrdata");
    let metadata = ImageMetadata {
        rating: 6,
        ..ImageMetadata::default()
    };
    assert!(save_sidecar(&path, &metadata).is_err());
    assert!(!path.exists());
}
