//! Choose the output lattice before composition from the actual native source map.
//! This changes output sampling density only; pair evidence and station-local
//! registration retain their original coordinates and thresholds.

#[cfg(test)]
use super::residual_warp::ResidualWarp;
use crate::panorama_utils::stack_pipeline::report::{ClosureReport, OutputDensityReport};
use crate::panorama_utils::stitching::FocusStackTileMasks;
use nalgebra::{Matrix2, Matrix3, Point2, Point3, Vector2};
use std::collections::HashMap;
use std::path::Path;

use super::ImageInfo;

const PROBE_STEP: u32 = 32;

#[derive(Clone, Debug)]
pub(super) struct DensitySample {
    source_id: usize,
    point: Point2<f64>,
    source_per_station: Matrix2<f64>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct StationDensitySamples {
    samples: Vec<DensitySample>,
    pub(super) source_count: usize,
    pub(super) unresolved_sources: usize,
    invalid_probes: usize,
    boundary_probes: usize,
}

/// Test-only export of the native density evidence used by the output-lattice
/// planner. It is intentionally opt-in and never changes the production path.
#[cfg(test)]
pub(super) fn write_density_diagnostics(
    samples: &[StationDensitySamples],
    station_poses: &HashMap<usize, Matrix3<f64>>,
    station_ids: &[usize],
    directory: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let summary = samples
        .iter()
        .enumerate()
        .map(|(station, evidence)| {
            let tile_id = station_ids.get(station).copied();
            serde_json::json!({
                "station": station,
                "tile_id": tile_id,
                "station_to_world": tile_id.and_then(|id| station_poses.get(&id)).map(|pose| pose.as_slice().to_vec()),
                "source_count": evidence.source_count,
                "sample_count": evidence.samples.len(),
                "unresolved_sources": evidence.unresolved_sources,
                "invalid_probes": evidence.invalid_probes,
                "boundary_probes": evidence.boundary_probes,
            })
        })
        .collect::<Vec<_>>();
    std::fs::write(
        directory.join("summary.json"),
        serde_json::to_vec_pretty(&summary).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    for (station, evidence) in samples.iter().enumerate() {
        let tile_id = station_ids.get(station).copied();
        let probes = evidence
            .samples
            .iter()
            .map(|sample| {
                serde_json::json!({
                    "source_id": sample.source_id,
                    "point": [sample.point.x, sample.point.y],
                    "source_per_station": [
                        [sample.source_per_station[(0, 0)], sample.source_per_station[(0, 1)]],
                        [sample.source_per_station[(1, 0)], sample.source_per_station[(1, 1)]],
                    ],
                })
            })
            .collect::<Vec<_>>();
        std::fs::write(
            directory.join(format!("station-{station:03}.json")),
            serde_json::to_vec(&serde_json::json!({
                "station": station,
                "tile_id": tile_id,
                "station_to_world": tile_id.and_then(|id| station_poses.get(&id)).map(|pose| pose.as_slice().to_vec()),
                "source_count": evidence.source_count,
                "sample_count": evidence.samples.len(),
                "unresolved_sources": evidence.unresolved_sources,
                "invalid_probes": evidence.invalid_probes,
                "boundary_probes": evidence.boundary_probes,
                "probes": probes,
            }))
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn differential(
    map: impl Fn(Point2<f64>) -> Option<Point2<f64>>,
    point: Point2<f64>,
) -> Option<Matrix2<f64>> {
    let dx = map(point + Vector2::new(0.5, 0.0))? - map(point - Vector2::new(0.5, 0.0))?;
    let dy = map(point + Vector2::new(0.0, 0.5))? - map(point - Vector2::new(0.0, 0.5))?;
    let value = Matrix2::from_columns(&[dx, dy]);
    (value.iter().all(|v| v.is_finite()) && value.determinant() > 1e-10).then_some(value)
}

/// Capture only lightweight derivatives from the station fusion already done
/// for matching. No additional decode, color buffer or owner map is retained.
pub(super) fn native_density_samples(
    masks: Option<&FocusStackTileMasks>,
    origin: (f64, f64),
    members: &[&ImageInfo],
) -> StationDensitySamples {
    let Some(masks) = masks else {
        return StationDensitySamples {
            unresolved_sources: 1,
            ..Default::default()
        };
    };
    let (width, height) = masks.coverage.dimensions();
    let mut result = StationDensitySamples::default();
    // The ownership legend, not the available refinement map, defines which
    // real sources require evidence. A missing anchor/refinement must not
    // silently remove the highest-density contributor from the plan. Sources
    // explicitly excluded by fusion own zero pixels and need no output map.
    let owned_counts = masks.ownership.owned_pixel_counts();
    let mut sources = Vec::new();
    for (path, &owned) in masks.ownership.legend().iter().zip(&owned_counts) {
        if owned == 0 {
            continue;
        }
        result.source_count += 1;
        let Some(source) = members
            .iter()
            .find(|source| Path::new(&source.filename) == path)
        else {
            result.unresolved_sources += 1;
            continue;
        };
        let Some(refinement) = masks.source_refinements.get(&source.id) else {
            result.unresolved_sources += 1;
            continue;
        };
        sources.push((source.id, refinement));
    }
    sources.sort_by_key(|(id, _)| *id);
    for (source_id, refinement) in sources {
        let Some(sampling) = refinement.source_sampling.as_ref() else {
            result.unresolved_sources += 1;
            continue;
        };
        let before = result.samples.len();
        let map = |point: Point2<f64>| {
            let corrected = refinement.inverse_canvas_point(point);
            sampling.source_point(Point3::new(
                corrected.x + origin.0,
                corrected.y + origin.1,
                1.0,
            ))
        };
        for y in (1..height.saturating_sub(1)).step_by(PROBE_STEP as usize) {
            for x in (1..width.saturating_sub(1)).step_by(PROBE_STEP as usize) {
                if masks.coverage.covered()[y as usize * width as usize + x as usize] == 0 {
                    continue;
                }
                let point = Point2::new(f64::from(x), f64::from(y));
                // source_point enforces source support/bounds. A footprint
                // outside that domain cannot define a derivative; record it
                // separately from a finite-domain folded/singular Jacobian.
                if [
                    Vector2::new(0.5, 0.0),
                    Vector2::new(-0.5, 0.0),
                    Vector2::new(0.0, 0.5),
                    Vector2::new(0.0, -0.5),
                ]
                .iter()
                .any(|offset| map(point + offset).is_none())
                {
                    result.boundary_probes += 1;
                    continue;
                }
                if let Some(source_per_station) = differential(&map, point) {
                    result.samples.push(DensitySample {
                        source_id,
                        point,
                        source_per_station,
                    });
                } else {
                    result.invalid_probes += 1;
                }
            }
        }
        if result.samples.len() == before {
            result.unresolved_sources += 1;
        }
    }
    if result.source_count == 0 {
        result.unresolved_sources += 1;
    }
    result
}

#[cfg(test)]
fn residual_composed_scale(
    pose: &Matrix3<f64>,
    residual: &ResidualWarp,
    tile_id: usize,
    sample: &DensitySample,
) -> Option<f64> {
    let map = |point: Point2<f64>| {
        let mapped = pose * Point3::new(point.x, point.y, 1.0);
        if !mapped.z.is_finite() || mapped.z.abs() <= 1e-10 {
            return None;
        }
        let world = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
        let warped = residual.forward_world_for_station(world, tile_id);
        (warped.x.is_finite() && warped.y.is_finite()).then_some(warped)
    };
    let world_per_station = differential(map, sample.point)?;
    let inverse = sample.source_per_station.try_inverse()?;
    let jacobian = world_per_station * inverse;
    let smallest = jacobian.svd(false, false).singular_values.min();
    (smallest.is_finite() && smallest > 1e-10).then_some(smallest)
}

/// Export a residual-aware density diagnostic without changing the output
/// lattice. This is deliberately test-only: the production planner still
/// bypasses non-identity residual fields until a full coordinate contract is
/// established. Existing source-boundary/invalid counts are retained and
/// residual-composition failures are reported separately.
#[cfg(test)]
pub(super) fn write_residual_density_diagnostics(
    samples: &[StationDensitySamples],
    station_poses: &HashMap<usize, Matrix3<f64>>,
    station_ids: &[usize],
    residual: &ResidualWarp,
    closure: &ClosureReport,
    directory: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let mut scales = Vec::new();
    let mut worst: Option<(f64, usize, usize, Point2<f64>)> = None;
    let mut residual_invalid_probe_count = 0usize;
    let mut per_station = Vec::with_capacity(samples.len());
    for (station_index, evidence) in samples.iter().enumerate() {
        let tile_id = station_ids.get(station_index).copied();
        let pose = tile_id.and_then(|id| station_poses.get(&id));
        let mut station_scales = Vec::new();
        let mut sample_records = Vec::with_capacity(evidence.samples.len());
        for sample in &evidence.samples {
            let scale = pose.zip(tile_id).and_then(|(pose, tile_id)| {
                residual_composed_scale(pose, residual, tile_id, sample)
            });
            if let Some(scale) = scale {
                station_scales.push(scale);
                scales.push(scale);
                if worst.as_ref().is_none_or(|current| scale < current.0) {
                    worst = Some((scale, station_index, sample.source_id, sample.point));
                }
                sample_records.push(serde_json::json!({
                    "source_id": sample.source_id,
                    "point": [sample.point.x, sample.point.y],
                    "scale": scale,
                }));
            } else {
                residual_invalid_probe_count += 1;
                sample_records.push(serde_json::json!({
                    "source_id": sample.source_id,
                    "point": [sample.point.x, sample.point.y],
                    "scale": null,
                }));
            }
        }
        station_scales.sort_by(f64::total_cmp);
        let station_summary = serde_json::json!({
            "station": station_index,
            "tile_id": tile_id,
            "source_count": evidence.source_count,
            "sample_count": evidence.samples.len(),
            "measurable_count": station_scales.len(),
            "residual_invalid_probe_count": evidence.samples.len().saturating_sub(station_scales.len()),
            "invalid_probe_count": evidence.invalid_probes,
            "boundary_probe_count": evidence.boundary_probes,
            "minimum_scale": station_scales.first().copied(),
            "p01_scale": (!station_scales.is_empty()).then(|| station_scales[(station_scales.len() - 1) / 100]),
            "median_scale": (!station_scales.is_empty()).then(|| station_scales[station_scales.len() / 2]),
        });
        per_station.push(station_summary);
        std::fs::write(
            directory.join(format!("station-{station_index:03}.json")),
            serde_json::to_vec(&serde_json::json!({
                "station": station_index,
                "tile_id": tile_id,
                "samples": sample_records,
            }))
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    }
    scales.sort_by(f64::total_cmp);
    let summary = serde_json::json!({
        "schema": 1,
        "sampling_contract": "residual_forward_world_composed_with_native_source_jacobian; diagnostic_only",
        "residual_identity": residual.identity(),
        "residual_region_count": residual.regions.len(),
        "station_count": samples.len(),
        "source_count": samples.iter().map(|e| e.source_count).sum::<usize>(),
        "sample_count": scales.len(),
        "minimum_scale": scales.first().copied(),
        "p01_scale": (!scales.is_empty()).then(|| scales[(scales.len() - 1) / 100]),
        "median_scale": (!scales.is_empty()).then(|| scales[scales.len() / 2]),
        "direct_pair_p95_output_px": closure.max_direct_pair_p95_px,
        "residual_median_output_px": closure.residual_median_px,
        "max_corner_correction_output_px": closure.max_corner_correction_px,
        "invalid_probe_count": samples.iter().map(|e| e.invalid_probes).sum::<usize>(),
        "boundary_probe_count": samples.iter().map(|e| e.boundary_probes).sum::<usize>(),
        "residual_invalid_probe_count": residual_invalid_probe_count,
        "worst_station_index": worst.map(|(_, station, _, _)| station),
        "worst_source_id": worst.map(|(_, _, source, _)| source),
        "worst_station_point": worst.map(|(_, _, _, point)| [point.x, point.y]),
        "per_station": per_station,
    });
    std::fs::write(
        directory.join("summary.json"),
        serde_json::to_vec_pretty(&summary).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

pub(super) fn plan_output_density(
    station_ids: &[usize],
    poses: &HashMap<usize, Matrix3<f64>>,
    evidence: &[StationDensitySamples],
    residual_identity: bool,
) -> Result<OutputDensityReport, String> {
    let mut report = OutputDensityReport {
        status: "bypassed".into(),
        factor: 1.0,
        target_minimum_scale: 1.0,
        probe_step_px: PROBE_STEP,
        sampling_contract: "sampled_native_jacobian_not_full_pixel_proof".into(),
        source_count: evidence.iter().map(|s| s.source_count).sum(),
        invalid_probe_count: evidence.iter().map(|s| s.invalid_probes).sum(),
        boundary_probe_count: evidence.iter().map(|s| s.boundary_probes).sum(),
        ..Default::default()
    };
    if !residual_identity {
        report.bypass_reason = Some("nonidentity_world_residual".into());
        return Ok(report);
    }
    if evidence.len() != station_ids.len() || evidence.iter().any(|s| s.unresolved_sources > 0) {
        report.bypass_reason = Some("native_sampling_evidence_incomplete".into());
        return Ok(report);
    }
    if report.invalid_probe_count > 0 {
        report.bypass_reason = Some("invalid_native_jacobian_probes".into());
        return Ok(report);
    }
    let mut scales = Vec::new();
    let mut minimum = f64::INFINITY;
    for (station_index, (&id, station)) in station_ids.iter().zip(evidence).enumerate() {
        let pose = poses
            .get(&id)
            .ok_or("output density station pose missing")?;
        let map = |point: Point2<f64>| {
            let mapped = pose * Point3::new(point.x, point.y, 1.0);
            (mapped.z.is_finite() && mapped.z.abs() > 1e-10)
                .then(|| Point2::new(mapped.x / mapped.z, mapped.y / mapped.z))
        };
        for sample in &station.samples {
            let world_per_station = differential(map, sample.point)
                .ok_or("output density has a singular or folded station map")?;
            let inverse = sample
                .source_per_station
                .try_inverse()
                .ok_or("output density native map is singular")?;
            let jacobian = world_per_station * inverse;
            let smallest = jacobian.svd(false, false).singular_values.min();
            if !smallest.is_finite() || smallest <= 1e-10 {
                return Err("output density has an invalid source Jacobian".into());
            }
            if smallest < minimum {
                minimum = smallest;
                report.worst_station_index = Some(station_index);
                report.worst_source_id = Some(sample.source_id);
                report.worst_station_point = Some([sample.point.x, sample.point.y]);
            }
            scales.push(smallest);
        }
    }
    if scales.is_empty() {
        report.bypass_reason = Some("native_sampling_evidence_empty".into());
        return Ok(report);
    }
    scales.sort_by(f64::total_cmp);
    report.sample_count = scales.len();
    report.minimum_scale = Some(scales[0]);
    report.p01_scale = Some(scales[(scales.len() - 1) / 100]);
    report.median_scale = Some(scales[scales.len() / 2]);
    // A sampling target of one output pixel per finest source direction,
    // rounded upward to a stable lattice. Never derive this from QG thresholds.
    report.factor = ((1.0 / scales[0]).max(1.0) * 1024.0).ceil() / 1024.0;
    report.pixel_count_multiplier = report.factor * report.factor;
    if !report.factor.is_finite() || !report.pixel_count_multiplier.is_finite() {
        return Err("output density factor is not finite".into());
    }
    report.status = if report.factor > 1.0 {
        "applied"
    } else {
        "unchanged"
    }
    .into();
    Ok(report)
}

pub(super) fn apply_output_density(poses: &mut HashMap<usize, Matrix3<f64>>, factor: f64) {
    let scale = Matrix3::new(factor, 0.0, 0.0, 0.0, factor, 0.0, 0.0, 0.0, 1.0);
    for pose in poses.values_mut() {
        *pose = scale * *pose;
    }
}

/// Uniform output scale changes world-pixel residuals by the same factor.
/// Preserve the original closure report and independently enforce its limits
/// in the output lattice; never claim invariance of pixel error units.
pub(super) fn validate_output_closure(
    report: &mut OutputDensityReport,
    closure: &ClosureReport,
) -> Result<(), String> {
    let direct = closure.max_direct_pair_p95_px * report.factor;
    let median = closure.residual_median_px * report.factor;
    let corner = closure.max_corner_correction_px * report.factor;
    report.direct_pair_p95_output_px = Some(direct);
    report.residual_median_output_px = Some(median);
    report.max_corner_correction_output_px = Some(corner);
    if !direct.is_finite()
        || direct > super::CLOSURE_DIRECT_PAIR_P95_LIMIT_PX
        || !median.is_finite()
        || median > super::CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX
        || !corner.is_finite()
        || corner > super::CLOSURE_MAX_CORNER_CORRECTION_PX
    {
        report.status = "rejected".into();
        report.bypass_reason = Some("output_pixel_closure_limit_exceeded".into());
        return Err(format!(
            "output density {:.6} exceeds closure output-pixel limits: direct P95 {direct:.6}, median {median:.6}, corner {corner:.6}",
            report.factor
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panorama_utils::stack_pipeline::residual_warp::{OverlapResidual, WarpObservation};
    fn evidence(source_per_station: Matrix2<f64>) -> StationDensitySamples {
        StationDensitySamples {
            samples: vec![DensitySample {
                source_id: 7,
                point: Point2::new(40.0, 60.0),
                source_per_station,
            }],
            source_count: 1,
            unresolved_sources: 0,
            invalid_probes: 0,
            boundary_probes: 0,
        }
    }
    #[test]
    fn preserves_finest_direction_and_scales_translation_before_render() {
        let original = Matrix3::new(0.954, 0.0, 100.0, 0.0, 0.97, -20.0, 0.0, 0.0, 1.0);
        let mut poses = HashMap::from([(0, original)]);
        let plan =
            plan_output_density(&[0], &poses, &[evidence(Matrix2::identity())], true).unwrap();
        assert!(plan.factor >= 1.0 / 0.954 && plan.factor < 1.05);
        apply_output_density(&mut poses, plan.factor);
        assert!(poses[&0][(0, 0)] >= 1.0);
        assert!((poses[&0][(0, 2)] - 100.0 * plan.factor).abs() < 1e-10);
        let source = Point3::new(12.0, 13.0, 1.0);
        assert!((poses[&0].try_inverse().unwrap() * poses[&0] * source - source).norm() < 1e-10);
        assert_eq!(original[(0, 0)], 0.954);
    }
    #[test]
    fn includes_native_source_derivative_and_never_downsamples() {
        let poses = HashMap::from([(0, Matrix3::identity())]);
        let native = evidence(Matrix2::new(1.1, 0.0, 0.0, 1.0));
        assert!(
            plan_output_density(&[0], &poses, &[native], true)
                .unwrap()
                .factor
                >= 1.1
        );
        assert_eq!(
            plan_output_density(&[0], &poses, &[evidence(Matrix2::identity() * 0.9)], true)
                .unwrap()
                .factor,
            1.0
        );
    }
    #[test]
    fn nonidentity_residual_and_missing_evidence_are_explicit_bypasses() {
        let poses = HashMap::from([(0, Matrix3::identity())]);
        let report =
            plan_output_density(&[0], &poses, &[evidence(Matrix2::identity())], false).unwrap();
        assert_eq!(
            report.bypass_reason.as_deref(),
            Some("nonidentity_world_residual")
        );
        assert_eq!(report.factor, 1.0);
        assert_eq!(
            plan_output_density(&[0], &poses, &[], true).unwrap().status,
            "bypassed"
        );
        let mut invalid = evidence(Matrix2::identity());
        invalid.invalid_probes = 1;
        assert_eq!(
            plan_output_density(&[0], &poses, &[invalid], true)
                .unwrap()
                .bypass_reason
                .as_deref(),
            Some("invalid_native_jacobian_probes")
        );
    }

    #[test]
    fn common_density_preserves_direct_station_relation() {
        let left = Matrix3::new(1.0, 0.02, 120.0, 0.0, 1.0, -80.0, 0.0, 0.0, 1.0);
        let right = Matrix3::new(0.95, 0.01, 2120.0, 0.01, 0.97, -50.0, 0.00001, 0.0, 1.0);
        let before = left.try_inverse().unwrap() * right;
        let mut poses = HashMap::from([(0, left), (1, right)]);
        apply_output_density(&mut poses, 1.125);
        let after = poses[&0].try_inverse().unwrap() * poses[&1];
        assert!((before - after).norm() < 1e-10);
    }

    #[test]
    fn invalid_jacobians_cannot_supply_a_density() {
        assert!(differential(|p| Some(Point2::new(-p.x, p.y)), Point2::new(5.0, 5.0)).is_none());
        let poses = HashMap::from([(0, Matrix3::zeros())]);
        assert!(plan_output_density(&[0], &poses, &[evidence(Matrix2::identity())], true).is_err());
    }

    #[test]
    fn residual_composed_scale_keeps_constant_field_derivative() {
        let mut residual = ResidualWarp::new();
        let observations = (0..8)
            .flat_map(|row| {
                (0..8).map(move |column| {
                    WarpObservation::new(
                        crate::panorama_utils::stack_pipeline::report::WorldPoint {
                            x: 100.0 + column as f64 * 64.0,
                            y: 200.0 + row as f64 * 64.0,
                        },
                        [4.0, -3.0],
                        0.1,
                    )
                })
            })
            .collect::<Vec<_>>();
        residual
            .add_observations(
                OverlapResidual::new(
                    0,
                    0,
                    crate::panorama_utils::stack_pipeline::report::WorldRect {
                        left: 100.0,
                        top: 200.0,
                        width: 448.0,
                        height: 448.0,
                    },
                    10.0,
                ),
                &observations,
            )
            .unwrap();
        residual.register_station_alias(99, 0);
        let sample = DensitySample {
            source_id: 7,
            point: Point2::new(324.0, 424.0),
            source_per_station: Matrix2::identity(),
        };
        let scale = residual_composed_scale(&Matrix3::identity(), &residual, 99, &sample).unwrap();
        assert!((scale - 1.0).abs() < 1e-8);
    }

    #[test]
    fn density_cannot_relax_closure_pixel_limits() {
        let mut report = OutputDensityReport {
            factor: 1.1,
            ..Default::default()
        };
        let mut closure = ClosureReport {
            max_direct_pair_p95_px: 2.9,
            ..Default::default()
        };
        assert!(validate_output_closure(&mut report, &closure).is_err());
        assert!((report.direct_pair_p95_output_px.unwrap() - 3.19).abs() < 1e-10);
        assert_eq!(closure.max_direct_pair_p95_px, 2.9);
        closure.max_direct_pair_p95_px = 2.012;
        assert!(validate_output_closure(&mut report, &closure).is_ok());
        closure.max_corner_correction_px = 240.0;
        assert!(validate_output_closure(&mut report, &closure).is_err());
        closure.max_corner_correction_px = 0.0;
        closure.residual_median_px = 1.9;
        assert!(validate_output_closure(&mut report, &closure).is_err());
    }
}
