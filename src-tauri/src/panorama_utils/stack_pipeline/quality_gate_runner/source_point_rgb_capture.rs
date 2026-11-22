//! Test-only source RGB probes at bounded QG world points.
use super::super::source_sampling::{
    map_world_to_source_for_quality, sample_quality_source_at_world, source_point_for_reference,
};
use super::super::{QualityGateInput, QualitySource, project};
use image::Rgb32FImage;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct Request {
    schema: u32,
    points: Vec<PointRequest>,
}

#[derive(Clone, Debug, Deserialize)]
struct PointRequest {
    label: String,
    world: [f64; 2],
    owner: Option<u16>,
    path: Option<String>,
}

pub(super) struct Capture {
    directory: PathBuf,
    points: Vec<PointRequest>,
    seen: BTreeSet<String>,
    availability_seen: BTreeSet<String>,
    records: Vec<serde_json::Value>,
    availability: Vec<serde_json::Value>,
}

impl Capture {
    pub(super) fn from_environment() -> Option<Self> {
        let input = std::env::var_os("RAW_EDITOR_QG_SOURCE_POINT_RGB_INPUT")?;
        let directory = PathBuf::from(std::env::var_os("RAW_EDITOR_QG_SOURCE_POINT_RGB_DIR")?);
        let bytes = fs::read(input).ok()?;
        let request: Request = serde_json::from_slice(&bytes).ok()?;
        if request.schema != 1
            || request.points.is_empty()
            || request.points.len() > 32
            || request
                .points
                .iter()
                .any(|p| p.owner.is_none() && p.path.is_none())
        {
            return None;
        }
        Some(Self {
            directory,
            points: request.points,
            seen: BTreeSet::new(),
            availability_seen: BTreeSet::new(),
            records: Vec::new(),
            availability: Vec::new(),
        })
    }

    pub(super) fn requested_owners(&self) -> Vec<u16> {
        self.points
            .iter()
            .filter_map(|point| point.owner)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub(super) fn requested_paths(&self) -> Vec<String> {
        self.points
            .iter()
            .filter_map(|point| point.path.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn record_availability(
        &mut self,
        point: &PointRequest,
        source_info: &QualitySource,
        loaded: bool,
        error: Option<String>,
    ) {
        if point.owner.is_some_and(|owner| owner != source_info.owner)
            || point
                .path
                .as_deref()
                .is_some_and(|path| path != source_info.path)
        {
            return;
        }
        if !self.availability_seen.insert(point.label.clone()) {
            return;
        }
        let mut record = json!({
            "label": point.label,
            "requested_owner": point.owner,
            "requested_path": point.path,
            "candidate_owner": source_info.owner,
            "candidate_path": source_info.path,
            "station_id": source_info.geometry.station_id,
            "prepared_dimensions": source_info.dimensions,
            "status": if loaded { "loaded" } else { "unavailable" },
        });
        if let Some(error) = error {
            record["error"] = json!(error);
        }
        self.availability.push(record);
    }

    pub(super) fn record_source_availability(
        &mut self,
        source_info: &QualitySource,
        loaded: bool,
        error: Option<String>,
    ) {
        let points = self.points.clone();
        for point in &points {
            self.record_availability(point, source_info, loaded, error.clone());
        }
    }

    pub(super) fn observe(
        &mut self,
        input: &QualityGateInput<'_>,
        source_info: &QualitySource,
        source: &Rgb32FImage,
    ) {
        for point in &self.points {
            if point.owner.is_some_and(|owner| owner != source_info.owner)
                || point
                    .path
                    .as_deref()
                    .is_some_and(|path| path != source_info.path)
                || self.seen.contains(&point.label)
            {
                continue;
            }
            let world = nalgebra::Point2::new(point.world[0], point.world[1]);
            let result = sample_quality_source_at_world(source, world, source_info, input.residual);
            let key = point.label.clone();
            let mut record = json!({
                "label": point.label,
                "requested_world": point.world,
                "owner": source_info.owner,
                "requested_owner": point.owner,
                "requested_path": point.path,
                "path": source_info.path,
                "station_id": source_info.geometry.station_id,
                "prepared_dimensions": source_info.dimensions,
                "source_image_dimensions": source.dimensions(),
                "sample_stage": "quality_reference_source_sample",
                "mapping": mapping_diagnostic(input, source_info, source, world),
            });
            match result {
                Ok(rgb) => {
                    record["status"] = json!("sampled");
                    record["rgb_f32"] = json!(rgb.0);
                    record["rgb_f32_bits"] = json!(rgb.0.map(f32::to_bits));
                }
                Err(error) => {
                    record["status"] = json!("failed");
                    record["error"] = json!(error);
                }
            }
            self.seen.insert(key);
            self.records.push(record);
        }
    }

    pub(super) fn finish(self) {
        if fs::create_dir_all(&self.directory).is_err() {
            return;
        }
        let _ = fs::write(
            self.directory.join("source-point-rgb.json"),
            serde_json::to_vec_pretty(&json!({
                "schema": 1,
                "diagnostic_only": true,
                "records": self.records,
                "availability": self.availability,
                "requested_labels": self.points.iter().map(|p| &p.label).collect::<Vec<_>>(),
                "unobserved_labels": self.points.iter().filter(|p| !self.seen.contains(&p.label)).map(|p| &p.label).collect::<Vec<_>>(),
                "note": "quality-reference source samples only; no production decision or acceptance claim",
            }))
            .unwrap_or_default(),
        );
    }
}

fn mapping_diagnostic(
    input: &QualityGateInput<'_>,
    source_info: &QualitySource,
    source: &Rgb32FImage,
    world: nalgebra::Point2<f64>,
) -> serde_json::Value {
    let inverse =
        (source_info.geometry.tile_to_world * source_info.geometry.member_to_anchor).try_inverse();
    let Some(inverse) = inverse else {
        return json!({
            "native_registration": source_info.native_registration.is_some(),
            "native_warp": source_info.native_warp.is_some(),
            "status": "inverse_failed",
        });
    };
    if let Some(native) = source_info.native_registration.as_ref() {
        let unwarped = input
            .residual
            .warp_inverse(world, source_info.geometry.station_id);
        if let Some(sampling) = native.refinement.source_sampling.as_ref() {
            let Some(station) = native.corrected_station(unwarped) else {
                return mapping_status(source_info, "corrected_station_failed");
            };
            let target = nalgebra::Point3::new(
                station.x + native.station_origin.0,
                station.y + native.station_origin.1,
                1.0,
            );
            let Some(mapped) = sampling.source_point(target) else {
                return mapping_status(source_info, "native_source_sampling_unmapped");
            };
            return mapping_value(source_info, source, mapped, "native_source_sampling_mapped");
        }
        let Some(corrected) = native.reference_world(unwarped) else {
            return mapping_status(source_info, "reference_world_failed");
        };
        let Some(mapped) = project(&inverse, corrected) else {
            return mapping_status(source_info, "raw_inverse_project_failed");
        };
        return mapping_value(source_info, source, mapped, "raw_inverse_projected");
    }
    let mapped = if source_info.native_registration.is_some() {
        source_point_for_reference(
            world,
            &source_info.geometry,
            input.residual,
            source_info.native_registration.as_ref(),
            &inverse,
        )
    } else {
        map_world_to_source_for_quality(
            world,
            &source_info.geometry,
            input.residual,
            &inverse,
            source_info.native_warp.as_ref(),
        )
    };
    let Ok(mapped) = mapped else {
        return mapping_status(source_info, "mapping_failed");
    };
    mapping_value(source_info, source, mapped, "mapped")
}

fn mapping_status(source_info: &QualitySource, status: &str) -> serde_json::Value {
    json!({
        "native_registration": source_info.native_registration.is_some(),
        "native_warp": source_info.native_warp.is_some(),
        "status": status,
    })
}

fn mapping_value(
    source_info: &QualitySource,
    source: &Rgb32FImage,
    mapped: nalgebra::Point2<f64>,
    mapped_status: &str,
) -> serde_json::Value {
    let in_bounds = mapped.x.is_finite()
        && mapped.y.is_finite()
        && mapped.x >= 0.0
        && mapped.y >= 0.0
        && mapped.x <= f64::from(source.width().saturating_sub(1))
        && mapped.y <= f64::from(source.height().saturating_sub(1));
    json!({
        "native_registration": source_info.native_registration.is_some(),
        "native_warp": source_info.native_warp.is_some(),
        "mapped_source_xy": [mapped.x, mapped.y],
        "mapped_in_bounds": in_bounds,
        "status": if in_bounds { mapped_status } else { "mapped_out_of_bounds" },
    })
}
