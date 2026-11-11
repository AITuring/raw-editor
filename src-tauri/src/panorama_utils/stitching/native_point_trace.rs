//! Bounded, opt-in observations of the existing native compositor writes.
//! This module is cfg(test) only. It never receives a loader or changes a pixel.

use super::{
    FocusTileSources, LayeredOwnershipRender, NativeOwnerGeometry,
    get_high_quality_interpolated_pixel,
};
use crate::panorama_utils::coordinates::map_target_to_source;
use image::Rgb;
use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_POINTS: usize = 128;
const MAX_INPUT_BYTES: usize = 65_536;
static ITERATION: AtomicU64 = AtomicU64::new(0);

// Unit tests select a local configuration without mutating process env.
thread_local! {
    static PATHS_OVERRIDE: std::cell::RefCell<Option<(Option<PathBuf>, Option<PathBuf>)>> = const { std::cell::RefCell::new(None) };
}

#[derive(Clone, Deserialize, Serialize)]
struct RequestedPoint {
    label: String,
    canvas: [u32; 2],
    role: String,
}

#[derive(Deserialize)]
struct Input {
    schema: u32,
    reference_canvas_origin: [f64; 2],
    points: Vec<RequestedPoint>,
}

fn parse_input(bytes: &[u8]) -> Result<Input, String> {
    if bytes.len() > MAX_INPUT_BYTES {
        return Err("point trace input exceeds 65536 bytes".into());
    }
    let input: Input = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    if input.schema != 1 || !input.reference_canvas_origin.iter().all(|x| x.is_finite()) {
        return Err("point trace requires schema 1 and a finite reference_canvas_origin".into());
    }
    if input.points.is_empty() || input.points.len() > MAX_POINTS {
        return Err("point trace requires 1..=128 points including controls".into());
    }
    let mut seen = BTreeSet::new();
    for point in &input.points {
        if point.label.is_empty()
            || point.label.len() > 256
            || point.role.is_empty()
            || point.role.len() > 128
        {
            return Err("point trace label/role is empty or too long".into());
        }
        if !seen.insert(point.canvas) {
            return Err("point trace contains duplicate canvas coordinates".into());
        }
    }
    Ok(input)
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    fs::write(&temporary, bytes).map_err(|error| error.to_string())?;
    fs::rename(&temporary, path).map_err(|error| error.to_string())
}

fn invalidate_latest(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

fn pixel_on_grid(world: [f64; 2], origin: [f64; 2], dimensions: [u32; 2]) -> Option<[u32; 2]> {
    let p = [world[0] - origin[0], world[1] - origin[1]];
    if (0..2).any(|axis| {
        !p[axis].is_finite()
            || p[axis] < 0.0
            || p[axis] >= f64::from(dimensions[axis])
            || (p[axis] - p[axis].round()).abs() > 1e-6
    }) {
        return None;
    }
    let rounded = [p[0].round(), p[1].round()];
    if (0..2).any(|axis| rounded[axis] < 0.0 || rounded[axis] >= f64::from(dimensions[axis])) {
        return None;
    }
    Some([rounded[0] as u32, rounded[1] as u32])
}

/// No image-sized storage: at most 128 points and their per-station evidence.
pub(super) struct PointTrace {
    input: Input,
    process_directory: PathBuf,
    iteration_directory: PathBuf,
    iteration: u64,
    metadata: Value,
    worlds: Vec<[f64; 2]>,
    working_pixels: Vec<Option<[u32; 2]>>,
    rows: BTreeMap<u32, Vec<(u32, usize)>>,
    stations: Vec<Value>,
    finished: bool,
}

impl PointTrace {
    pub(super) fn from_env() -> Option<Self> {
        if let Some((input, directory)) = PATHS_OVERRIDE.with(|value| value.borrow().clone()) {
            return Self::begin_paths(input, directory);
        }
        Self::begin_paths(
            std::env::var_os("RAW_EDITOR_NATIVE_POINT_TRACE_INPUT").map(PathBuf::from),
            std::env::var_os("RAW_EDITOR_NATIVE_POINT_TRACE_DIR").map(PathBuf::from),
        )
    }

    fn begin_paths(input_path: Option<PathBuf>, directory: Option<PathBuf>) -> Option<Self> {
        // OFF does not open a file, create a directory, allocate point records
        // or inspect any image/owner/source. A partial opt-in is invalid.
        if input_path.is_none() && directory.is_none() {
            return None;
        }
        let Some(directory) = directory else {
            eprintln!("Native point trace disabled: output directory is missing");
            return None;
        };
        let iteration = ITERATION.fetch_add(1, Ordering::Relaxed) + 1;
        let process_directory = directory.join(format!("process-{}", std::process::id()));
        let iteration_directory = process_directory.join(format!("iteration-{iteration:04}"));
        let latest_path = process_directory.join("latest.json");
        let result = (|| {
            // Supersede the previous attempt before parsing/configuration I/O.
            invalidate_latest(&latest_path)?;
            fs::create_dir_all(&process_directory).map_err(|error| error.to_string())?;
            write_json(
                &latest_path,
                &json!({
                    "latest_started_iteration": iteration, "latest_iteration_status": "running",
                    "final_iteration": null, "offline_binding_allowed": false,
                }),
            )?;
            let input_path = input_path.ok_or("point trace input path is missing")?;
            let mut bytes = Vec::new();
            fs::File::open(&input_path)
                .map_err(|error| error.to_string())?
                .take((MAX_INPUT_BYTES + 1) as u64)
                .read_to_end(&mut bytes)
                .map_err(|error| error.to_string())?;
            let input = parse_input(&bytes)?;
            fs::create_dir(&iteration_directory).map_err(|error| error.to_string())?;
            let worlds = input
                .points
                .iter()
                .map(|p| {
                    [
                        f64::from(p.canvas[0]) + input.reference_canvas_origin[0],
                        f64::from(p.canvas[1]) + input.reference_canvas_origin[1],
                    ]
                })
                .collect::<Vec<_>>();
            let input_sha256 = Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let metadata = json!({
                "schema": 1, "diagnostic_only": true, "process_id": std::process::id(),
                "iteration": iteration, "iteration_status": "running", "final_iteration": null,
                "input_path": input_path, "input_sha256": input_sha256,
                "reference_canvas_origin": input.reference_canvas_origin, "requests": input.points,
                "point_limit_including_controls": MAX_POINTS, "source_loader_calls_added": 0,
                "owner_semantics": "compositor owner is station-order index + 1; local raw-owner diagnostics are legend index + 1, not final global RAW owner IDs",
                "finality": "compositor observations only; successful full-run QG/TIFF binding must identify the final iteration",
            });
            write_json(&iteration_directory.join("trace.json"), &metadata)?;
            Ok::<_, String>(Self {
                input,
                process_directory: process_directory.clone(),
                iteration_directory: iteration_directory.clone(),
                iteration,
                metadata,
                worlds,
                working_pixels: Vec::new(),
                rows: BTreeMap::new(),
                stations: Vec::new(),
                finished: false,
            })
        })();
        match result {
            Ok(trace) => Some(trace),
            Err(error) => {
                eprintln!("Native point trace could not start: {error}");
                let _ = invalidate_latest(&latest_path);
                let _ = write_json(
                    &latest_path,
                    &json!({"latest_started_iteration": iteration, "latest_iteration_status": "failed", "final_iteration": null, "offline_binding_allowed": false, "error": error}),
                );
                None
            }
        }
    }

    pub(super) fn set_canvas(&mut self, dimensions: [u32; 2], origin: [f64; 2]) {
        self.metadata["working_canvas"] = json!({"dimensions": dimensions, "world_origin": origin});
        self.working_pixels = self
            .worlds
            .iter()
            .map(|&world| pixel_on_grid(world, origin, dimensions))
            .collect();
        for (index, pixel) in self.working_pixels.iter().enumerate() {
            if let Some([x, y]) = pixel {
                self.rows.entry(*y).or_default().push((*x, index));
            }
        }
        for points in self.rows.values_mut() {
            points.sort_unstable();
        }
    }

    pub(super) fn station(
        &self,
        metadata: Value,
        bounds: [u32; 4],
        tile_dimensions: [u32; 2],
        originals: Option<&FocusTileSources>,
        map: impl Fn(Point3<f64>) -> Option<Point2<f64>>,
    ) -> StationTrace {
        let points = self.worlds.iter().enumerate().map(|(index, &world)| {
            let pixel = self.working_pixels.get(index).copied().flatten();
            let eligible = pixel.is_some_and(|[x, y]| x >= bounds[0] && x <= bounds[1] && y >= bounds[2] && y <= bounds[3]);
            let mapped = map(Point3::new(world[0], world[1], 1.0));
            let tile_in_bounds = mapped.is_some_and(|p| p.x >= 0.0 && p.y >= 0.0 && p.x < f64::from(tile_dimensions[0]) && p.y < f64::from(tile_dimensions[1]));
            json!({
                "point_index": index, "request": self.input.points[index], "world": world,
                "working_canvas_pixel": pixel, "station_bbox_eligible": eligible,
                "diagnostic_world_to_tile": mapped.map(|p| [p.x,p.y]), "tile_rectangle_in_bounds": tile_in_bounds,
                "native_candidates": mapped.and_then(|p| originals.map(|sources| diagnose_sources(sources,p))),
                "actual": {"status": if pixel.is_none() {"outside_working_canvas_or_noninteger_grid"} else if !eligible {"outside_station_bbox"} else {"pending"}, "sample_invoked": false, "mask_written": false},
            })
        }).collect();
        StationTrace {
            metadata,
            rows: self.rows.clone(),
            points: Mutex::new(points),
        }
    }

    pub(super) fn push_station(&mut self, station: StationTrace) {
        self.stations.push(json!({"station": station.metadata, "points": station.points.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner())}));
    }

    /// Bounded, read-only stage snapshots on the original working canvas.
    pub(super) fn stage(
        &mut self,
        name: &str,
        image: &image::Rgb32FImage,
        coverage: &image::GrayImage,
        owners: &[u16],
    ) {
        let points = self
            .working_pixels
            .iter()
            .enumerate()
            .map(|(index, pixel)| {
                let sample = pixel.map(|[x, y]| {
                    let rgb = image.get_pixel(x, y).0;
                    json!({"rgb": rgb, "rgb_f32_bits": rgb.map(f32::to_bits),
                    "coverage": coverage.get_pixel(x, y).0[0],
                    "station_owner": owners[y as usize * image.width() as usize + x as usize]})
                });
                json!({"point_index": index, "world": self.worlds[index],
                "working_canvas_pixel": pixel, "sample": sample})
            })
            .collect::<Vec<_>>();
        if self.metadata.get("stage_snapshots").is_none() {
            self.metadata["stage_snapshots"] = json!([]);
        }
        let working_canvas = self.metadata["working_canvas"].clone();
        self.metadata["stage_snapshots"]
            .as_array_mut()
            .unwrap()
            .push(json!({"stage": name, "iteration": self.iteration,
                "working_canvas": working_canvas, "points": points}));
    }

    pub(super) fn complete(&mut self, rendered: &LayeredOwnershipRender) {
        let dimensions = [rendered.image.width(), rendered.image.height()];
        let origin = [rendered.sampling_origin.0, rendered.sampling_origin.1];
        let points = self.worlds.iter().enumerate().map(|(index,&world)| {
            let pixel = pixel_on_grid(world, origin, dimensions);
            let sample = pixel.map(|[x,y]| json!({"coverage": rendered.coverage.is_covered(x,y), "station_owner": rendered.ownership.owner_at(x,y), "rgb": rendered.image.get_pixel(x,y).0}));
            json!({"point_index": index, "request": self.input.points[index], "world": world, "final_compositor_canvas_pixel": pixel, "sample": sample})
        }).collect::<Vec<_>>();
        self.metadata["compositor_return"] = json!({"dimensions": dimensions, "world_origin": origin, "points": points, "scope": "after compositor crop/finishing, before any later QG repair; requires offline binding to final TIFF"});
        self.metadata["iteration_status"] = json!("compositor_complete");
        self.metadata["stations"] = json!(self.stations);
        let result = write_json(&self.iteration_directory.join("trace.json"), &self.metadata).and_then(|_| {
            write_json(&self.process_directory.join("latest.json"), &json!({
                "latest_started_iteration": self.iteration, "latest_completed_iteration": self.iteration,
                "latest_iteration_status": "compositor_complete", "final_iteration": null,
                "trace": self.iteration_directory.join("trace.json"), "offline_binding_allowed": false,
                "finality": "candidate only; successful full-run QG and exact TIFF point comparison must establish finality",
            }))
        });
        if let Err(error) = result {
            self.fail(&format!("diagnostic completion write failed: {error}"));
        }
        self.finished = true;
    }

    fn fail(&mut self, error: &str) {
        self.metadata["iteration_status"] = json!("failed");
        self.metadata["error"] = json!(error);
        self.metadata["stations"] = json!(self.stations);
        let latest = self.process_directory.join("latest.json");
        let _ = invalidate_latest(&latest);
        let _ = write_json(&self.iteration_directory.join("trace.json"), &self.metadata);
        let _ = write_json(
            &latest,
            &json!({"latest_started_iteration": self.iteration, "latest_iteration_status": "failed", "final_iteration": null, "offline_binding_allowed": false, "error": error}),
        );
        eprintln!("Native point trace incomplete: {error}");
    }
}

impl Drop for PointTrace {
    fn drop(&mut self) {
        if !self.finished {
            self.fail("compositor did not return a completed render");
        }
    }
}

pub(super) struct StationTrace {
    metadata: Value,
    rows: BTreeMap<u32, Vec<(u32, usize)>>,
    points: Mutex<Vec<Value>>,
}

#[derive(Clone, Copy)]
pub(super) struct TraceRow<'a> {
    station: &'a StationTrace,
    points: &'a [(u32, usize)],
}

impl StationTrace {
    pub(super) fn row(&self, y: u32) -> Option<TraceRow<'_>> {
        Some(TraceRow {
            station: self,
            points: self.rows.get(&y)?,
        })
    }
}

impl<'a> TraceRow<'a> {
    /// The lazy before-state closure runs only at a selected coordinate.
    pub(super) fn pixel(self, x: u32, before: impl FnOnce() -> Value) -> Option<PixelTrace<'a>> {
        let index = self.points.binary_search_by_key(&x, |&(x, _)| x).ok()?;
        Some(PixelTrace {
            station: self.station,
            index: self.points[index].1,
            actual: json!({"status":"entered", "before":before(), "sample_invoked":false, "mask_written":false}),
        })
    }
}

pub(super) struct PixelTrace<'a> {
    station: &'a StationTrace,
    index: usize,
    actual: Value,
}

impl PixelTrace<'_> {
    /// Geometry returned by the same resolver invocation that produced RGB,
    /// rather than a diagnostic replay of the nearest-owner query.
    pub(super) fn native_resolution(
        &mut self,
        resolved: Option<&NativeOwnerGeometry>,
        sources: &FocusTileSources,
    ) {
        self.actual["native_owner_resolution"] = resolved.map_or(Value::Null, |resolved| {
            let index = usize::from(resolved.local_owner - 1);
            let metadata = sources.originals[index]
                .as_ref()
                .and_then(|source| source.refinement.source_sampling.as_ref())
                .map(|sampling| sampling.snapshot_metadata());
            json!({
                "local_owner": resolved.local_owner, "legend_index": index,
                "evidence_pixel": resolved.evidence_pixel,
                "prepared_point": [resolved.prepared_point.x, resolved.prepared_point.y],
                "used_fallback": resolved.used_fallback, "geometry_valid": true,
                "source_id": metadata.map(|(id,_,_)|id),
                "source_path": metadata.map(|(_,path,_)|path),
                "scope": "returned by actual sample_with_geometry invocation",
            })
        });
    }
    pub(super) fn record(&mut self, key: &str, value: Value) {
        self.actual[key] = value;
    }
    pub(super) fn reject(&mut self, reason: &str) {
        self.actual["status"] = json!(reason);
    }
    pub(super) fn sample(&mut self, rgb: Option<Rgb<f32>>) {
        self.actual["sample_invoked"] = json!(true);
        self.actual["sample_rgb"] = json!(rgb.map(|p| p.0));
        self.actual["sample_returned_some"] = json!(rgb.is_some());
        if self.actual["native_owner_resolution"].is_object() {
            self.actual["native_owner_resolution"]["sample_rgb"] = json!(rgb.map(|p| p.0));
        }
    }
    pub(super) fn written(&mut self, rgb: Rgb<f32>, owner: u16) {
        self.actual["status"] = json!("mask_written");
        self.actual["mask_written"] = json!(true);
        self.actual["after"] = json!({"coverage":255,"station_owner":owner,"rgb":rgb.0});
    }
}

impl Drop for PixelTrace<'_> {
    fn drop(&mut self) {
        if self.actual["after"].is_null() {
            self.actual["after"] = self.actual["before"].clone();
        }
        self.station
            .points
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())[self.index]["actual"] =
            self.actual.take();
    }
}

fn mapped_original(sources: &FocusTileSources, index: usize, point: Point2<f64>) -> Value {
    let mut value = json!({"legend_index":index,"local_owner":index+1,"legend_path":sources.ownership.legend().get(index),"original_present":false});
    let Some(original) = sources.originals.get(index).and_then(Option::as_ref) else {
        value["rejection"] = json!("original_missing");
        return value;
    };
    value["original_present"] = json!(true);
    let corrected = original.refinement.inverse_canvas_point(point);
    let target = Point3::new(
        corrected.x + sources.origin.0,
        corrected.y + sources.origin.1,
        1.0,
    );
    value["refinement"] = json!({"translation":original.refinement.translation,"displacement_present":original.refinement.displacement.is_some(),"foreground_move_present":original.refinement.foreground_move.is_some(),"corrected_tile_point":[corrected.x,corrected.y],"sampling_world_target":[target.x,target.y]});
    let Some(sampling) = original.refinement.source_sampling.as_ref() else {
        value["rejection"] = json!("source_sampling_missing");
        return value;
    };
    value["source"] = json!({"id":sampling.image.id,"filename":sampling.image.filename,"prepared_dimensions":original.image.dimensions(),"mapping_dimensions":sampling.image.dimensions(),"footprint_present":sampling.image.source_footprint.is_some(),"local_band_count":sampling.band_inverses.len()});
    let global = map_target_to_source(
        &sampling.global_inverse,
        target,
        &sampling.image,
        sampling.projection,
    );
    value["global_unfiltered_source_point"] = json!(global.map(|p| [p.x, p.y]));
    value["global_source_supported"] =
        json!(global.is_some_and(|p| sampling.image.source_is_supported(p)));
    let Some(source) = sampling.source_point(target) else {
        value["rejection"] = json!("source_mapping_returned_none");
        return value;
    };
    value["source_point"] = json!([source.x, source.y]);
    let width = f64::from(original.image.width());
    let height = f64::from(original.image.height());
    let render_bounds = source.x >= 0.0 && source.y >= 0.0 && source.x < width && source.y < height;
    let qg_bounds =
        source.x >= 0.0 && source.y >= 0.0 && source.x <= width - 1.0 && source.y <= height - 1.0;
    value["render_bounds_exclusive_dimensions"] = json!(render_bounds);
    value["qg_bounds_inclusive_last_center"] = json!(qg_bounds);
    value["source_footprint_supported"] = json!(sampling.image.source_is_supported(source));
    if !render_bounds {
        value["rejection"] = json!("prepared_image_out_of_bounds");
        return value;
    }
    let floor = original.image.get_pixel(
        (source.x.floor() as u32).min(original.image.width() - 1),
        (source.y.floor() as u32).min(original.image.height() - 1),
    );
    value["nearest_floor_pixel"] = json!(floor.0);
    value["nearest_floor_support_diagnostic_only"] =
        json!(floor.0.iter().all(|x| x.is_finite()) && floor.0.iter().any(|&x| x > 1e-6));
    let rgb = get_high_quality_interpolated_pixel(&original.image, source.x, source.y);
    value["diagnostic_sample_rgb"] = json!(rgb.0);
    value["diagnostic_sample_finite"] = json!(rgb.0.iter().all(|x| x.is_finite()));
    value["black_filter_le_1e_minus_6"] = json!(rgb[0].max(rgb[1]).max(rgb[2]) <= 1e-6);
    value["rejection"] = Value::Null;
    value
}

fn diagnose_sources(sources: &FocusTileSources, point: Point2<f64>) -> Value {
    let (width, height) = sources.ownership.dimensions();
    let mut candidates = Vec::new();
    let mut selected = None;
    let mut minimum = f64::INFINITY;
    let mut raw_sources = BTreeMap::new();
    let rounded = (point.x.round(), point.y.round());
    for dy in -1..=1 {
        for dx in -1..=1 {
            let x = rounded.0 + f64::from(dx);
            let y = rounded.1 + f64::from(dy);
            if x < 0.0
                || y < 0.0
                || x >= f64::from(width)
                || y >= f64::from(height)
                || !x.is_finite()
                || !y.is_finite()
            {
                candidates.push(json!({"pixel":[x,y],"reason":"outside_owner_plane"}));
                continue;
            }
            let owner = sources.ownership.owner_at(x as u32, y as u32);
            let distance = (x - point.x).powi(2) + (y - point.y).powi(2);
            candidates.push(json!({"pixel":[x,y],"local_owner":owner,"distance_squared":distance}));
            if let Some(index) = owner.checked_sub(1) {
                raw_sources
                    .entry(index)
                    .or_insert_with(|| mapped_original(sources, usize::from(index), point));
                if distance < minimum {
                    minimum = distance;
                    selected = Some(
                        json!({"local_owner":owner,"legend_index":index,"pixel":[x,y],"distance_squared":distance}),
                    );
                }
            }
        }
    }
    // This is the old report-side floor-anchored 4x4 local-owner winner only.
    // It does not claim to have executed the global RAW owner ledger lookup.
    let mut reverse = None;
    let mut reverse_distance = f64::INFINITY;
    let floor = (point.x.floor(), point.y.floor());
    for dy in -1..=2 {
        for dx in -1..=2 {
            let x = floor.0 + f64::from(dx);
            let y = floor.1 + f64::from(dy);
            if x < 0.0
                || y < 0.0
                || x >= f64::from(width)
                || y >= f64::from(height)
                || !x.is_finite()
                || !y.is_finite()
            {
                continue;
            }
            let owner = sources.ownership.owner_at(x as u32, y as u32);
            if owner == 0 {
                continue;
            }
            let distance = (x - point.x).mul_add(x - point.x, (y - point.y) * (y - point.y));
            if distance < reverse_distance {
                reverse_distance = distance;
                reverse = Some(
                    json!({"local_owner":owner,"legend_index":owner-1,"pixel":[x,y],"distance_squared":distance}),
                );
            }
        }
    }
    json!({"sampling_origin":sources.origin,"ownership_dimensions":[width,height],"actual_algorithm_3x3_winner_diagnostic":selected,"legacy_reverse_4x4_winner_diagnostic":reverse,"candidates_3x3":candidates,"source_diagnostics":raw_sources.values().collect::<Vec<_>>(),"extra_mapping_and_samples_are_diagnostic_only":true})
}

#[cfg(test)]
#[path = "native_point_trace_tests.rs"]
mod tests;
