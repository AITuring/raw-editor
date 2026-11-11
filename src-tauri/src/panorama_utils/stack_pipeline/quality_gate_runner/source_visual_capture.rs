//! Opt-in, test-only evidence from existing QG ROIs. Never receives a loader.
//! Images and owner planes are borrowed immutably; this is not a best-focus
//! oracle. Final TIFF binding is an explicit, later offline acceptance step.

use super::super::source_sampling::{map_world_to_source_for_quality, source_point_for_reference};
use super::super::{QualityGateInput, QualityRoi, QualitySource};
use crate::panorama_utils::stack_pipeline::quality_gate;
use crate::panorama_utils::stack_pipeline::report::QualityGateReport;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb32FImage};
use nalgebra::{Matrix3, Point2};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_ROIS: usize = 8;
const MAX_ROI_SIDE: u32 = 512;
static ITERATION: AtomicU64 = AtomicU64::new(0);
type RoiIdentity = (u32, u32, u32, u16, u32, u64, u64);

#[path = "source_visual_point_evidence.rs"]
mod point_evidence;

fn roi_identity(roi: &QualityRoi) -> RoiIdentity {
    (
        roi.x,
        roi.y,
        roi.side,
        roi.owner,
        roi.region_label,
        roi.world_origin.0.to_bits(),
        roi.world_origin.1.to_bits(),
    )
}

fn roi_station(roi: &QualityRoi, sources: &[QualitySource]) -> Option<usize> {
    quality_gate::station_level_roi_station_id(roi.region_label).or_else(|| {
        sources
            .iter()
            .find(|source| source.owner == roi.owner)
            .map(|source| source.geometry.station_id)
    })
}

#[derive(Clone)]
struct Config {
    directory: PathBuf,
    profile: String,
    limit: usize,
}

// Tests inject a thread-local configuration without mutating process env.
thread_local! {
    static CONFIG_OVERRIDE: std::cell::RefCell<Option<Option<Config>>> = const { std::cell::RefCell::new(None) };
}

fn configuration() -> Option<Config> {
    if let Some(config) = CONFIG_OVERRIDE.with(|value| value.borrow().clone()) {
        return config;
    }
    let directory = std::env::var_os("RAW_EDITOR_QG_SOURCE_VISUAL_DIR")?;
    for incompatible in [
        "RAW_EDITOR_FOCUS_FINAL_WORLD_OWNER_ORACLE_DIR",
        "RAW_EDITOR_FOCUS_GRADIENT_ORACLE_DIR",
        "RAW_EDITOR_FOCUS_GRADIENT_ORACLE_REPORT",
    ] {
        if std::env::var_os(incompatible).is_some() {
            eprintln!("QG source visual capture disabled: incompatible oracle {incompatible}");
            return None;
        }
    }
    Some(Config {
        directory: directory.into(),
        profile: std::env::var("RAW_EDITOR_QG_SOURCE_VISUAL_PROFILE")
            .unwrap_or_else(|_| "unspecified".into()),
        limit: std::env::var("RAW_EDITOR_QG_SOURCE_VISUAL_MAX_ROIS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4)
            .clamp(1, MAX_ROIS),
    })
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes).map_err(|error| error.to_string())?;
    fs::rename(temporary, path).map_err(|error| error.to_string())
}

fn remove_latest(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("cannot invalidate {}: {error}", path.display())),
    }
}

fn owner_counts(input: &QualityGateInput<'_>) -> Value {
    let mut counts = BTreeMap::<u16, u64>::new();
    let mut uncovered = 0u64;
    for (&owner, &covered) in input.ownership.iter().zip(input.coverage.as_raw()) {
        if covered == 0 {
            uncovered += 1;
        } else {
            *counts.entry(owner).or_default() += 1;
        }
    }
    // Include zero-contribution sources explicitly; merely appearing in the
    // source legend must never be mistaken for owning final output pixels.
    for source in input.sources {
        counts.entry(source.owner).or_default();
    }
    json!({
        "scope": "actual final QG input owner plane, coverage != 0 only",
        "uncovered_pixels": uncovered,
        "covered_pixels": counts.values().sum::<u64>(),
        "covered_owner_zero_pixels": counts.get(&0).copied().unwrap_or(0),
        "owners": counts.into_iter().map(|(owner, pixels)| json!({
            "owner": owner,
            "path": input.sources.iter().find(|source| source.owner == owner).map(|source| &source.path),
            "pixels": pixels,
        })).collect::<Vec<_>>(),
    })
}

pub(super) struct Capture {
    process_directory: PathBuf,
    iteration_directory: PathBuf,
    iteration: u64,
    limit: usize,
    metadata: Value,
    rois: Vec<Value>,
    errors: Vec<String>,
    selected: Option<Vec<RoiIdentity>>,
}

impl Capture {
    pub(super) fn from_environment(input: &QualityGateInput<'_>) -> Option<Self> {
        Self::begin(configuration(), input)
    }

    fn begin(config: Option<Config>, input: &QualityGateInput<'_>) -> Option<Self> {
        Self::begin_with_metadata_writer(config, input, write_json)
    }

    fn begin_with_metadata_writer(
        config: Option<Config>,
        input: &QualityGateInput<'_>,
        metadata_writer: impl FnOnce(&Path, &Value) -> Result<(), String>,
    ) -> Option<Self> {
        let config = config?; // Disabled capture does not scan or allocate planes.
        let iteration = ITERATION.fetch_add(1, Ordering::Relaxed) + 1;
        let process_directory = config
            .directory
            .join(format!("process-{}", std::process::id()));
        let iteration_directory = process_directory.join(format!("iteration-{iteration:04}"));
        let latest_path = process_directory.join("latest.json");
        let result = (|| {
            // A repair attempt supersedes the previous completed pass even if
            // directory creation or metadata writing for this attempt fails.
            // Remove first so a subsequent failed atomic write cannot expose
            // the old complete marker to an offline TIFF-binding reader.
            remove_latest(&latest_path)?;
            fs::create_dir_all(&process_directory).map_err(|error| error.to_string())?;
            write_json(
                &latest_path,
                &json!({
                    "latest_started_iteration": iteration,
                    "latest_iteration_status": "running",
                    "final_iteration": null,
                    "offline_binding_allowed": false,
                    "finality": "pending full-run completion and offline TIFF binding",
                }),
            )?;
            fs::create_dir(&iteration_directory).map_err(|error| error.to_string())?;
            let mut metadata = json!({
                "schema": 1,
                "diagnostic_only": true,
                "reference_policy": "selected actual owner reference already used by QG; not best-focus comparison across all candidates",
                "process_id": std::process::id(),
                "iteration": iteration,
                "profile": config.profile,
                "debug_assertions": cfg!(debug_assertions),
                "output_dimensions": input.output.dimensions(),
                "world_origin": input.world_origin,
                "acceptance_render_scale": input.acceptance_render_scale,
                "final_sharpen_amount": input.final_sharpen_amount,
                "roi_limit": config.limit,
                "maximum_roi_side": MAX_ROI_SIDE,
                "roi_scope": "existing shared dense QG ROIs; excludes criterion-specific streaming content-search windows",
                "source_loader_calls_added": 0,
                "owner_counts": owner_counts(input),
                "sources": input.sources.iter().map(|source| json!({
                    "owner": source.owner,
                    "path": source.path,
                    "prepared_dimensions": source.dimensions,
                    "station_id": source.geometry.station_id,
                    "native_registration": source.native_registration.is_some(),
                    "legacy_native_warp": source.native_warp.is_some(),
                })).collect::<Vec<_>>(),
                "final_tiff_binding": "pending offline manifest binding after the complete run; no TIFF read here",
            });
            let mut errors = Vec::new();
            match point_evidence::capture(input, iteration, &config.profile, &iteration_directory) {
                Ok(Some((record, point_errors))) => {
                    metadata["native_point_evidence"] = record;
                    errors.extend(point_errors);
                }
                Ok(None) => {}
                Err(error) => {
                    let error = format!("native point evidence capture failed: {error}");
                    metadata["native_point_evidence"] = json!({"status": "failed", "error": error});
                    errors.push(error);
                }
            }
            metadata_writer(&iteration_directory.join("metadata.json"), &metadata)?;
            Ok::<_, String>(Self {
                process_directory: process_directory.clone(),
                iteration_directory,
                iteration,
                limit: config.limit,
                metadata,
                rois: Vec::new(),
                errors,
                selected: None,
            })
        })();
        match result {
            Ok(capture) => Some(capture),
            Err(error) => {
                eprintln!("QG source visual capture could not start: {error}");
                if let Err(error) = remove_latest(&latest_path) {
                    eprintln!("QG source visual capture state invalidation failed: {error}");
                }
                if let Err(state_error) = write_json(
                    &latest_path,
                    &json!({
                        "latest_started_iteration": iteration,
                        "latest_iteration_status": "failed",
                        "final_iteration": null,
                        "offline_binding_allowed": false,
                        "error": error,
                        "finality": "capture start failed; offline TIFF binding is forbidden, including earlier completed iterations",
                    }),
                ) {
                    eprintln!(
                        "QG source visual capture failed-state write failed: {state_error}; offline TIFF binding is forbidden"
                    );
                    // With unavailable storage even the failed marker cannot
                    // be guaranteed; leave no stale marker whenever possible.
                    if let Err(error) = remove_latest(&latest_path) {
                        eprintln!("QG source visual capture stale-state removal failed: {error}");
                    }
                }
                None
            }
        }
    }

    pub(super) fn select_rois(&mut self, rois: &[QualityRoi], sources: &[QualitySource]) {
        let mut first_by_station = BTreeMap::new();
        for (index, roi) in rois.iter().enumerate() {
            first_by_station
                .entry(roi_station(roi, sources))
                .or_insert(index);
        }
        let mut selected = first_by_station
            .values()
            .copied()
            .take(self.limit)
            .collect::<Vec<_>>();
        for index in 0..rois.len() {
            if selected.len() >= self.limit {
                break;
            }
            if !selected.contains(&index) {
                selected.push(index);
            }
        }
        self.metadata["roi_selection"] = json!({
            "policy": "one_per_station_then_fill_original_roi_order_with_hard_limit",
            "available_roi_count": rois.len(),
            "available_station_ids": first_by_station.keys().collect::<Vec<_>>(),
            "selected_roi_indices": selected,
            "selected_station_ids": selected.iter().map(|&index| roi_station(&rois[index],sources)).collect::<std::collections::BTreeSet<_>>(),
            "excluded_station_ids": first_by_station.keys().filter(|station| !selected.iter().any(|&index| roi_station(&rois[index],sources) == **station)).collect::<Vec<_>>(),
            "coverage_claim": "bounded selected windows only; selected references may still be unavailable",
        });
        self.selected = Some(
            selected
                .into_iter()
                .map(|index| roi_identity(&rois[index]))
                .collect(),
        );
    }

    pub(super) fn record(
        &mut self,
        input: &QualityGateInput<'_>,
        roi: &QualityRoi,
        output: &Rgb32FImage,
        reference: &Rgb32FImage,
        was_cached: bool,
    ) {
        if self
            .selected
            .as_ref()
            .is_some_and(|selected| !selected.contains(&roi_identity(roi)))
        {
            return;
        }
        if self.rois.len() >= self.limit {
            return;
        }
        if roi.side == 0 || roi.side > MAX_ROI_SIDE {
            self.errors
                .push(format!("ROI side {} exceeds capture bound", roi.side));
            return;
        }
        let number = self.rois.len();
        let result = self.write_roi(input, roi, output, reference, was_cached, number);
        match result {
            Ok(metadata) => self.rois.push(metadata),
            Err(error) => {
                eprintln!("QG source visual ROI capture failed: {error}");
                self.errors.push(error);
                // An attempted slot is not silently recycled over partial files.
                self.rois
                    .push(json!({"number": number, "status": "capture_failed"}));
            }
        }
    }

    fn write_roi(
        &self,
        input: &QualityGateInput<'_>,
        roi: &QualityRoi,
        output: &Rgb32FImage,
        reference: &Rgb32FImage,
        was_cached: bool,
        number: usize,
    ) -> Result<Value, String> {
        if output.dimensions() != (roi.side, roi.side)
            || reference.dimensions() != output.dimensions()
        {
            return Err("existing ROI dimensions disagree".into());
        }
        let directory = self.iteration_directory.join(format!("roi-{number:02}"));
        fs::create_dir(&directory).map_err(|error| error.to_string())?;
        let evidence = pixel_evidence(input, roi);
        write_rgb(&directory, "output", output)?;
        write_rgb(&directory, "reference", reference)?;
        ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(roi.side, roi.side, evidence.owners)
            .ok_or("invalid owner plane")?
            .save(directory.join("owners.png"))
            .map_err(|error| error.to_string())?;
        GrayImage::from_raw(roi.side, roi.side, evidence.validity)
            .ok_or("invalid validity plane")?
            .save(directory.join("validity.png"))
            .map_err(|error| error.to_string())?;
        fs::write(
            directory.join("prepared-source-xy.f64le"),
            evidence.coordinates,
        )
        .map_err(|error| error.to_string())?;
        let metadata = json!({
            "number": number, "status": "captured", "iteration": self.iteration,
            "canvas_roi": {"x": roi.x, "y": roi.y, "width": roi.side, "height": roi.side},
            "world_origin": roi.world_origin,
            "region_label": roi.region_label,
            "primary_roi_owner": roi.owner,
            "station_id": roi_station(roi, input.sources),
            "reference_cache_hit": was_cached,
            "reference_stage": "existing post-tone QG reference, no new load or resampling",
            "reference_owner_policy": if quality_gate::station_level_roi_station_id(roi.region_label).is_some() {
                "per_pixel_actual_input_ownership"
            } else { "primary_roi_owner; differing actual owners are marked invalid for correspondence" },
            "coordinate_format": "row-major XY f64 little-endian, 16 bytes/pixel; prepared/developed source pixels, not sensor CFA coordinates; unavailable coordinates are NaN",
            "coordinate_scope": "complete current native registration inverse chain; legacy native_warp coordinates describe its mapped centre, not its multistage interpolation taps",
            "validity_legend": {"0":"uncovered", "1":"mapped_in_bounds", "2":"owner_zero", "3":"owner_missing_or_duplicate", "4":"wrong_station", "5":"singular_source_geometry", "6":"mapping_failed", "7":"out_of_bounds", "8":"actual_owner_differs_from_reference_owner"},
            "validity_counts": evidence.validity_counts,
            "validity_scope": "coverage, actual owner, inverse mapping and prepared-image bounds; no additional RAW decode or sensor/CFA validity test",
            "image_format": "PNG16 is quantized visual evidence; *.rgb.f32le is exact existing RGB32F row-major channels; no resize or color-profile conversion",
            "reference_claim": "selected-owner fidelity only; no best-focus/all-candidate claim",
            "source_loader_calls_added": 0,
        });
        write_json(&directory.join("metadata.json"), &metadata)?;
        Ok(metadata)
    }

    fn complete(&mut self, report: &QualityGateReport) -> Result<(), String> {
        self.metadata["gate_verdict"] = json!(report.verdict);
        self.metadata["gate_roi_count"] = json!(report.roi_count);
        self.metadata["capture_errors"] = json!(self.errors);
        self.metadata["captured_rois"] = json!(self.rois);
        self.metadata["captured_station_ids"] = json!(
            self.rois
                .iter()
                .filter_map(|roi| roi["station_id"].as_u64())
                .collect::<std::collections::BTreeSet<_>>()
        );
        self.metadata["iteration_status"] = json!("complete");
        write_json(
            &self.iteration_directory.join("metadata.json"),
            &self.metadata,
        )?;
        write_json(
            &self.process_directory.join("latest.json"),
            &json!({
                "latest_started_iteration": self.iteration,
                "latest_completed_iteration": self.iteration,
                "latest_iteration_status": "complete",
                "metadata": self.iteration_directory.join("metadata.json"),
                "capture_errors": self.errors,
                "final_iteration": null,
                "finality": "latest completed gate only; bind this iteration after successful complete run and offline TIFF comparison; a later repair pass supersedes it",
            }),
        )
    }
}

pub(super) fn finish(capture: &mut Option<Capture>, report: &QualityGateReport) {
    if let Some(capture) = capture.as_mut()
        && let Err(error) = capture.complete(report)
    {
        eprintln!("QG source visual manifest failed: {error}");
    }
}

fn write_rgb(directory: &Path, name: &str, image: &Rgb32FImage) -> Result<(), String> {
    let bytes = image
        .as_raw()
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(directory.join(format!("{name}.rgb.f32le")), bytes)
        .map_err(|error| error.to_string())?;
    DynamicImage::ImageRgb32F(image.clone())
        .to_rgb16()
        .save(directory.join(format!("{name}.png")))
        .map_err(|error| error.to_string())
}

struct PixelEvidence {
    owners: Vec<u16>,
    validity: Vec<u8>,
    coordinates: Vec<u8>,
    validity_counts: BTreeMap<u8, usize>,
}

fn pixel_evidence(input: &QualityGateInput<'_>, roi: &QualityRoi) -> PixelEvidence {
    let count = roi.side as usize * roi.side as usize;
    let mut result = PixelEvidence {
        owners: Vec::with_capacity(count),
        validity: Vec::with_capacity(count),
        coordinates: Vec::with_capacity(count * 16),
        validity_counts: BTreeMap::new(),
    };
    let mut sources = BTreeMap::<u16, Vec<&QualitySource>>::new();
    for source in input.sources {
        sources.entry(source.owner).or_default().push(source);
    }
    let inverses = sources
        .iter()
        .map(|(&owner, entries)| {
            (
                owner,
                (entries[0].geometry.tile_to_world * entries[0].geometry.member_to_anchor)
                    .try_inverse(),
            )
        })
        .collect::<BTreeMap<u16, Option<Matrix3<f64>>>>();
    let station = quality_gate::station_level_roi_station_id(roi.region_label);
    for y in 0..roi.side {
        for x in 0..roi.side {
            let canvas_x = roi.x.checked_add(x);
            let canvas_y = roi.y.checked_add(y);
            let index = canvas_x
                .zip(canvas_y)
                .filter(|&(x, y)| x < input.coverage.width() && y < input.coverage.height())
                .map(|(x, y)| y as usize * input.coverage.width() as usize + x as usize);
            let owner = index
                .and_then(|index| input.ownership.get(index))
                .copied()
                .unwrap_or(0);
            result.owners.push(owner);
            let mut mapped = Point2::new(f64::NAN, f64::NAN);
            let validity = (|| {
                if !index.is_some_and(|index| input.coverage.as_raw()[index] != 0) {
                    return 0;
                }
                if owner == 0 {
                    return 2;
                }
                let Some(entries) = sources.get(&owner).filter(|entries| entries.len() == 1) else {
                    return 3;
                };
                let source = entries[0];
                if station.is_some_and(|station| station != source.geometry.station_id) {
                    return 4;
                }
                let Some(inverse) = inverses[&owner].as_ref() else {
                    return 5;
                };
                let world = Point2::new(
                    roi.world_origin.0 + f64::from(x),
                    roi.world_origin.1 + f64::from(y),
                );
                let point = if source.native_registration.is_some() {
                    source_point_for_reference(
                        world,
                        &source.geometry,
                        input.residual,
                        source.native_registration.as_ref(),
                        inverse,
                    )
                } else {
                    map_world_to_source_for_quality(
                        world,
                        &source.geometry,
                        input.residual,
                        inverse,
                        source.native_warp.as_ref(),
                    )
                };
                let Ok(point) = point else {
                    return 6;
                };
                mapped = point;
                if !point.x.is_finite()
                    || !point.y.is_finite()
                    || point.x < 0.0
                    || point.y < 0.0
                    || source.dimensions.0 == 0
                    || source.dimensions.1 == 0
                    || point.x > f64::from(source.dimensions.0 - 1)
                    || point.y > f64::from(source.dimensions.1 - 1)
                {
                    return 7;
                }
                if station.is_none() && owner != roi.owner {
                    return 8;
                }
                1
            })();
            result.coordinates.extend(mapped.x.to_le_bytes());
            result.coordinates.extend(mapped.y.to_le_bytes());
            result.validity.push(validity);
            *result.validity_counts.entry(validity).or_default() += 1;
        }
    }
    result
}

#[path = "source_visual_capture_tests.rs"]
mod tests;
