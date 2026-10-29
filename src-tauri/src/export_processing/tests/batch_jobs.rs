//! Batch export jobs.
//!
//! Virtual-copy sidecar paths, batch-geometry lens adjustments, output folders,
//! content-orientation decisions and the batch result summary.

use super::*;

#[test]
fn export_adjustment_mode_normalizes_virtual_copy_paths() {
    let mode = ExportAdjustmentsMode::UseSidecars {
        active_path: Some("/photos/source.jpg?vc=3".to_string()),
        active_adjustments: Some(serde_json::json!({ "exposure": 1.0 })),
    }
    .normalize_active_path();

    match mode {
        ExportAdjustmentsMode::UseSidecars {
            active_path,
            active_adjustments,
        } => {
            assert_eq!(active_path.as_deref(), Some("/photos/source.jpg"));
            assert_eq!(
                active_adjustments,
                Some(serde_json::json!({ "exposure": 1.0 }))
            );
        }
        ExportAdjustmentsMode::GlobalOverride(_) | ExportAdjustmentsMode::PerPathOverrides(_) => {
            panic!("unexpected override")
        }
    }
}

#[test]
fn batch_geometry_adjustments_only_enable_a_usable_distortion_profile() {
    let usable_params = serde_json::json!({ "k1": 0.012, "k2": 0.0, "k3": 0.0, "model": 0 });
    let adjustments = build_batch_geometry_adjustments(2, Some(usable_params.clone()), true);

    assert_eq!(adjustments["lensDistortionEnabled"], true);
    assert_eq!(adjustments["lensDistortionParams"], usable_params);
    assert_eq!(adjustments["lensTcaEnabled"], false);
    assert_eq!(adjustments["lensVignetteEnabled"], false);
    assert_eq!(adjustments["orientationSteps"], 2);

    let unavailable =
        build_batch_geometry_adjustments(0, Some(serde_json::json!({ "k1": 0.0 })), true);
    assert_eq!(unavailable["lensDistortionEnabled"], false);

    let disabled =
        build_batch_geometry_adjustments(0, Some(serde_json::json!({ "k1": 0.012 })), false);
    assert_eq!(disabled["lensDistortionEnabled"], false);
}

#[test]
fn batch_geometry_output_folders_are_created_without_overwriting_previous_runs() {
    let temporary_directory = tempfile::tempdir().expect("create temporary output root");

    let first = create_batch_geometry_output_folder(temporary_directory.path())
        .expect("create first batch output folder");
    let second = create_batch_geometry_output_folder(temporary_directory.path())
        .expect("create second batch output folder");

    assert_eq!(
        first.file_name().and_then(|name| name.to_str()),
        Some("RAW Editor Corrected")
    );
    assert_eq!(
        second.file_name().and_then(|name| name.to_str()),
        Some("RAW Editor Corrected 2")
    );
    assert!(first.is_dir());
    assert!(second.is_dir());
}

#[test]
fn batch_geometry_content_orientation_requires_a_clear_win_over_the_current_direction() {
    let decision = decide_content_orientation([0.44, 0.48, 0.78, 0.53]);

    assert_eq!(
        decision,
        ContentOrientationDecision {
            orientation_steps: 2,
            corrected: true,
            confidence: 0.78,
        }
    );

    let ambiguous = decide_content_orientation([0.72, 0.74, 0.77, 0.73]);
    assert_eq!(ambiguous, ContentOrientationDecision::preserve_current());

    let weak = decide_content_orientation([0.35, 0.38, 0.57, 0.41]);
    assert_eq!(weak, ContentOrientationDecision::preserve_current());
}

#[test]
fn batch_summary_retains_later_results_after_an_item_failure() {
    let summary = summarize_batch_export_results([
        Ok(()),
        Err("second.jpg: simulated encoder failure".to_string()),
        Ok(()),
        Err("fourth.jpg: simulated disk failure".to_string()),
        Ok(()),
    ]);

    assert_eq!(summary.completed, 5);
    assert_eq!(summary.errors.len(), 2);
    assert!(summary.errors[0].contains("second.jpg"));
    assert!(summary.errors[1].contains("fourth.jpg"));
}
