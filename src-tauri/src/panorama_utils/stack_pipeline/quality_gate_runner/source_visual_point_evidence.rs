//! Bounded, test-only observations of the final RAW owner plane received by QG.
//! No source loader or image sampling is available here. The inverse mapping
//! is the existing QG reference mapping; final export binding remains offline.

use super::{QualityGateInput, map_world_to_source_for_quality, source_point_for_reference};
use nalgebra::Point2;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_POINTS: usize = 128;
const MAX_INPUT_BYTES: usize = 65_536;

thread_local! {
    static INPUT_OVERRIDE: std::cell::RefCell<Option<Option<PathBuf>>> = const { std::cell::RefCell::new(None) };
}

#[derive(Deserialize)]
struct RequestedPoint {
    label: String,
    canvas: [u32; 2],
    role: String,
}

#[derive(Deserialize)]
struct Request {
    schema: u32,
    reference_canvas_origin: [f64; 2],
    points: Vec<RequestedPoint>,
}

fn parse_request(bytes: &[u8]) -> Result<Request, String> {
    if bytes.len() > MAX_INPUT_BYTES {
        return Err("native point evidence input exceeds 65536 bytes".into());
    }
    let request: Request = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    if request.schema != 1
        || !request
            .reference_canvas_origin
            .iter()
            .all(|x| x.is_finite())
    {
        return Err(
            "native point evidence requires schema 1 and a finite reference_canvas_origin".into(),
        );
    }
    if request.points.is_empty() || request.points.len() > MAX_POINTS {
        return Err("native point evidence requires 1..=128 points including controls".into());
    }
    let mut coordinates = BTreeSet::new();
    for point in &request.points {
        if point.label.is_empty()
            || point.label.len() > 256
            || point.role.is_empty()
            || point.role.len() > 128
        {
            return Err("native point evidence label/role is empty or too long".into());
        }
        if !coordinates.insert(point.canvas) {
            return Err("native point evidence has duplicate reference canvas coordinates".into());
        }
    }
    Ok(request)
}

fn input_path() -> Option<PathBuf> {
    if let Some(path) = INPUT_OVERRIDE.with(|value| value.borrow().clone()) {
        return path;
    }
    std::env::var_os("RAW_EDITOR_NATIVE_POINT_TRACE_INPUT").map(PathBuf::from)
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn canvas_pixel(world: [f64; 2], origin: (f64, f64), dimensions: (u32, u32)) -> Option<[u32; 2]> {
    let local = [world[0] - origin.0, world[1] - origin.1];
    let rounded = [local[0].round(), local[1].round()];
    let bounds = [dimensions.0, dimensions.1];
    if (0..2).any(|axis| {
        !local[axis].is_finite()
            || local[axis] < 0.0
            || local[axis] >= f64::from(bounds[axis])
            || (local[axis] - rounded[axis]).abs() > 1e-6
            || rounded[axis] < 0.0
            || rounded[axis] >= f64::from(bounds[axis])
    }) {
        return None;
    }
    Some([rounded[0] as u32, rounded[1] as u32])
}

fn observe(
    bytes: &[u8],
    path: &Path,
    input: &QualityGateInput<'_>,
    iteration: u64,
    profile: &str,
) -> Result<(Value, Vec<String>), String> {
    let request = parse_request(bytes)?;
    let dimensions = input.output.dimensions();
    let pixels = (dimensions.0 as usize)
        .checked_mul(dimensions.1 as usize)
        .ok_or("native point evidence dimensions overflow")?;
    if dimensions.0 == 0
        || dimensions.1 == 0
        || input.coverage.dimensions() != dimensions
        || input.ownership.len() != pixels
        || !input.world_origin.0.is_finite()
        || !input.world_origin.1.is_finite()
        || input.acceptance_render_scale != 1.0
    {
        return Err("native point evidence requires matching nonempty QG planes, finite origin and native scale=1".into());
    }
    let mut errors = Vec::new();
    let mut points = Vec::with_capacity(request.points.len());
    for (point_id, point) in request.points.iter().enumerate() {
        let requested_world = [
            request.reference_canvas_origin[0] + f64::from(point.canvas[0]),
            request.reference_canvas_origin[1] + f64::from(point.canvas[1]),
        ];
        let canvas = canvas_pixel(requested_world, input.world_origin, dimensions);
        let mut evidence = json!({
            "point_id": point_id, "label": point.label, "role": point.role,
            "reference_canvas": point.canvas, "requested_world": requested_world,
            "canvas": canvas, "world": null, "status": "invalid",
            "actual_raw_owner": null, "path": null, "station_id": null,
            "coverage_u8": null, "output_rgb_f32": null, "output_rgb_f32_bits": null,
            "prepared_source_xy": null, "prepared_dimensions": null,
            "reference_mapping_status": "not_observed", "errors": [],
        });
        let mut point_errors = Vec::new();
        if let Some([x, y]) = canvas {
            let world = Point2::new(
                input.world_origin.0 + f64::from(x),
                input.world_origin.1 + f64::from(y),
            );
            let index = y as usize * dimensions.0 as usize + x as usize;
            let owner = input.ownership[index];
            let covered = input.coverage.get_pixel(x, y)[0];
            let rgb = input.output.get_pixel(x, y).0;
            evidence["world"] = json!([world.x, world.y]);
            evidence["actual_raw_owner"] = json!(owner);
            evidence["coverage_u8"] = json!(covered);
            evidence["output_rgb_f32_bits"] = json!(rgb.map(f32::to_bits));
            if rgb.iter().all(|value| value.is_finite()) {
                evidence["output_rgb_f32"] = json!(rgb);
            } else {
                point_errors.push("QG output RGB is non-finite".to_string());
            }
            if covered == 0 {
                // Preserve the actual plane value, including zero; an absent
                // reference is never replaced by a neighbouring/census owner.
                evidence["status"] = json!("uncovered");
                evidence["reference_mapping_status"] = json!("uncovered");
                if owner != 0 {
                    let mut matching = input.sources.iter().filter(|source| source.owner == owner);
                    if let Some(source) = matching.next().filter(|_| matching.next().is_none()) {
                        evidence["path"] = json!(source.path);
                        evidence["station_id"] = json!(source.geometry.station_id);
                    }
                }
            } else if owner == 0 {
                evidence["reference_mapping_status"] = json!("covered_owner_zero");
                point_errors.push("covered QG pixel has RAW owner zero".into());
            } else {
                let mut matching = input.sources.iter().filter(|source| source.owner == owner);
                let source = matching.next().filter(|_| matching.next().is_none());
                if let Some(source) = source {
                    evidence["path"] = json!(source.path);
                    evidence["station_id"] = json!(source.geometry.station_id);
                    evidence["prepared_dimensions"] = json!(source.dimensions);
                    evidence["native_registration"] = json!(source.native_registration.is_some());
                    evidence["legacy_native_warp"] = json!(source.native_warp.is_some());
                    if let Some(inverse) = (source.geometry.tile_to_world
                        * source.geometry.member_to_anchor)
                        .try_inverse()
                    {
                        let mapped = if source.native_registration.is_some() {
                            source_point_for_reference(
                                world,
                                &source.geometry,
                                input.residual,
                                source.native_registration.as_ref(),
                                &inverse,
                            )
                        } else {
                            map_world_to_source_for_quality(
                                world,
                                &source.geometry,
                                input.residual,
                                &inverse,
                                source.native_warp.as_ref(),
                            )
                        };
                        match mapped {
                            Ok(mapped) if mapped.x.is_finite() && mapped.y.is_finite() => {
                                evidence["prepared_source_xy"] = json!([mapped.x, mapped.y]);
                                if source.dimensions.0 != 0
                                    && source.dimensions.1 != 0
                                    && mapped.x >= 0.0
                                    && mapped.y >= 0.0
                                    && mapped.x <= f64::from(source.dimensions.0 - 1)
                                    && mapped.y <= f64::from(source.dimensions.1 - 1)
                                {
                                    evidence["status"] = json!("covered");
                                    evidence["reference_mapping_status"] =
                                        json!("mapped_in_bounds");
                                } else {
                                    evidence["reference_mapping_status"] = json!("out_of_bounds");
                                    point_errors.push(
                                        "QG source reference is outside prepared-source bounds"
                                            .into(),
                                    );
                                }
                            }
                            result => {
                                evidence["reference_mapping_status"] = json!("mapping_failed");
                                point_errors.push(format!(
                                    "QG source reference mapping failed: {result:?}"
                                ));
                            }
                        }
                    } else {
                        evidence["reference_mapping_status"] = json!("singular_source_geometry");
                        point_errors.push("QG source geometry has no inverse".into());
                    }
                } else {
                    evidence["reference_mapping_status"] = json!("owner_missing_or_duplicate");
                    point_errors.push(
                        "actual QG RAW owner has missing or duplicate source metadata".into(),
                    );
                }
            }
        } else {
            evidence["reference_mapping_status"] = json!("outside_output_or_off_grid");
            point_errors.push(
                "requested world point does not identify an in-bounds QG output pixel".into(),
            );
        }
        if !point_errors.is_empty() {
            evidence["status"] = json!("invalid");
            evidence["errors"] = json!(point_errors);
            errors.extend(point_errors.iter().map(|error| {
                format!(
                    "native point evidence {point_id} ({}): {error}",
                    point.label
                )
            }));
        }
        points.push(evidence);
    }
    Ok((
        json!({
            "schema": 1, "diagnostic_only": true, "status": if errors.is_empty() { "captured" } else { "failed" },
            "process_id": std::process::id(), "iteration": iteration, "final_iteration": null,
            "profile": profile, "debug_assertions": cfg!(debug_assertions),
            "input_path": path, "input_sha256": digest(bytes), "input_bytes": bytes.len(),
            "reference_canvas_origin": request.reference_canvas_origin,
            "world_origin": input.world_origin, "output_dimensions": dimensions,
            "acceptance_render_scale": input.acceptance_render_scale,
            "point_count": points.len(), "point_limit_including_controls": MAX_POINTS,
            "source_loader_calls_added": 0, "points": points, "errors": errors,
            "owner_semantics": "actual final RAW owner from this QualityGateInput ownership plane; no neighbouring owner or census inference",
            "coordinate_scope": "QG source reference inverse chain at the actual output pixel world coordinate; legacy native_warp describes its mapped centre, not multistage interpolation taps",
            "finality": "observation of one QG input iteration; requires latest complete capture, successful full run and offline TIFF binding; final_iteration remains null",
        }),
        errors,
    ))
}

pub(super) fn capture(
    input: &QualityGateInput<'_>,
    iteration: u64,
    profile: &str,
    directory: &Path,
) -> Result<Option<(Value, Vec<String>)>, String> {
    capture_path(input_path(), input, iteration, profile, directory)
}

fn capture_path(
    path: Option<PathBuf>,
    input: &QualityGateInput<'_>,
    iteration: u64,
    profile: &str,
    directory: &Path,
) -> Result<Option<(Value, Vec<String>)>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let mut bytes = Vec::new();
    fs::File::open(&path)
        .map_err(|error| error.to_string())?
        .take((MAX_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    let (ledger, errors) = observe(&bytes, &path, input, iteration, profile)?;
    let destination = directory.join("point-evidence.json");
    super::write_json(&destination, &ledger)?;
    let serialized = serde_json::to_vec_pretty(&ledger).map_err(|error| error.to_string())?;
    Ok(Some((
        json!({
            "status": ledger["status"], "path": destination, "sha256": digest(&serialized), "bytes": serialized.len(),
            "input_sha256": ledger["input_sha256"], "point_count": ledger["point_count"],
            "iteration": iteration, "source_loader_calls_added": 0,
        }),
        errors,
    )))
}

#[path = "source_visual_point_evidence_tests.rs"]
mod tests;
