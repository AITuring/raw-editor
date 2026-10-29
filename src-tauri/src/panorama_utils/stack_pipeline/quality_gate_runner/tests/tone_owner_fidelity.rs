//! The tone owner fidelity oracle: owner distribution feasibility, the joint
//! relaxation, high-frequency preservation of the candidate, its current-run
//! input and the one-source owner mapping.

use super::*;

fn synthetic_owner_evidence(
    baseline: [f32; 3],
    candidate: [f32; 3],
    reference: [f32; 3],
    fit_disjoint: bool,
) -> ToneOwnerEvidence {
    ToneOwnerEvidence {
        partition: "owner_raw_fidelity_holdout",
        roi: QualityRoi {
            x: 0,
            y: 0,
            side: 64,
            region_label: 0,
            owner: 1,
            world_origin: (0.0, 0.0),
            boundary_clearance: 0.0,
        },
        owner_path: "synthetic.raw".to_string(),
        station_id: 0,
        tile_index: 0,
        baseline_mean: baseline,
        candidate_mean: candidate,
        reference_mean: reference,
        field_gain: [1.0; 3],
        global_gain: [1.0; 3],
        global_bias: [0.0; 3],
        coverage_min: 255,
        confidence_min: 1.0,
        confidence_mean: 1.0,
        overlap_participation: false,
        fit_disjoint,
        gradient_ratio_candidate_to_source: Some(1.0),
        noise_ratio_candidate_to_source: Some(1.0),
    }
}

#[test]
fn owner_fidelity_rejects_joint_colour_shift_and_fit_leakage() {
    let shifted = synthetic_owner_evidence([0.4; 3], [0.5; 3], [0.4; 3], true);
    assert!(owner_distribution_feasible(&[&shifted], 0.0));
    assert!(!owner_distribution_feasible(&[&shifted], 1.0));

    let leaked = synthetic_owner_evidence([0.45; 3], [0.4; 3], [0.4; 3], false);
    assert!(!owner_distribution_feasible(&[&leaked], 1.0));
}

#[test]
fn joint_oracle_recovers_feasible_solution_or_certifies_pareto_gap() {
    let recovered = select_joint_relaxation(&[0.5, 0.75, 1.0], &[0.0, 0.25, 0.5, 0.75]);
    assert_eq!(recovered.selected, Some(0.5));
    assert_eq!(recovered.pareto_gap, None);

    let impossible = select_joint_relaxation(&[0.75, 1.0], &[0.0, 0.25, 0.5]);
    assert_eq!(impossible.selected, None);
    assert_eq!(impossible.minimum_boundary, Some(0.75));
    assert_eq!(impossible.maximum_owner, Some(0.5));
    assert_eq!(impossible.pareto_gap, Some(0.25));
    assert_eq!(impossible.selected.unwrap_or(0.0), 0.0);
}

#[test]
fn low_frequency_candidate_preserves_high_frequency_gradient_and_noise() {
    let baseline = Rgb32FImage::from_fn(64, 64, |x, y| {
        let fine = if (x + y).is_multiple_of(2) {
            0.03
        } else {
            -0.03
        };
        let broad = 0.25 + 0.003 * x as f32 + 0.002 * y as f32;
        Rgb([broad + fine, broad * 0.9 + fine, broad * 0.8 + fine])
    });
    let candidate = Rgb32FImage::from_fn(64, 64, |x, y| {
        let pixel = baseline.get_pixel(x, y).0;
        let low = 0.02 + 0.0001 * x as f32;
        Rgb(pixel.map(|value| value + low))
    });
    let gradient = quality_gate::normalized_gradient_energy(&candidate, 1.0).unwrap()
        / quality_gate::normalized_gradient_energy(&baseline, 1.0).unwrap();
    let noise = quality_gate::noise_sigma_ratio(&candidate, &baseline).unwrap();
    assert!(gradient >= quality_gate::QUALITY_GRADIENT_RATIO_MIN);
    assert!(
        (quality_gate::QUALITY_NOISE_RATIO_MIN..=quality_gate::QUALITY_NOISE_RATIO_MAX)
            .contains(&noise)
    );
}

#[test]
fn owner_oracle_has_no_previous_quality_gate_list_input() {
    let source = include_str!("../../quality_gate_runner.rs");
    let start = source.find("fn write_tone_owner_fidelity_oracle").unwrap();
    let end = source[start..]
        .find("pub(super) fn record_unmeasurable")
        .map(|offset| start + offset)
        .unwrap();
    let oracle = &source[start..end];
    assert!(!oracle.contains("std::fs::read"));
    assert!(!oracle.contains("GRADIENT_ORACLE_REPORT"));
    assert!(!oracle.contains("fixture"));
    assert!(oracle.contains("current-run tone owner oracle snapshot"));
}

#[test]
fn owner_fidelity_mapping_composes_member_tile_and_requires_one_source() {
    let source = Rgb32FImage::from_fn(32, 32, |x, y| {
        let value = (x + 3 * y) as f32 / 124.0;
        Rgb([value, value * 0.9, value * 0.8])
    });
    let translate = |x: f64, y: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0);
    let geometry = SourceGeometry {
        member_to_anchor: translate(5.0, 7.0),
        tile_to_world: translate(11.0, 13.0),
        station_id: 4,
    };
    let roi = QualityRoi {
        x: 0,
        y: 0,
        side: 2,
        region_label: 1,
        owner: 9,
        world_origin: (18.25, 23.5),
        boundary_clearance: 16.0,
    };
    let sampled = source_roi(&source, &roi, &geometry, &ResidualWarp::new()).unwrap();
    let expected =
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(&source, 2.25, 3.5);
    assert_eq!(sampled.get_pixel(0, 0), &expected);

    let info = QualitySource {
        native_registration: None,
        owner: 9,
        path: "owner.raw".to_string(),
        geometry,
        dimensions: source.dimensions(),
        native_warp: None,
    };
    assert_eq!(
        unique_owner_source(std::slice::from_ref(&info), 9)
            .unwrap()
            .path,
        "owner.raw"
    );
    assert!(unique_owner_source(&[info.clone(), info], 9).is_err());
    assert!(unique_owner_source(&[], 9).is_err());
}
