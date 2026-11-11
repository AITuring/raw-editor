use super::*;
use crate::panorama_utils::stack_pipeline::report::{FusionReport, PerformanceReport, StackReport};
use crate::panorama_utils::stack_pipeline::{degradation, focus_fuser, residual_warp, tone};
use crate::panorama_utils::stitching::{
    self, ConfidenceMap, CoverageMask, FocusSourceRefinement, FocusSourceSampling,
    FocusTilePayload, LayeredOwnershipRender, OwnershipMap, Projection,
};
use image::{GrayImage, Rgb};
use nalgebra::Matrix3;
use std::collections::HashMap;
use std::path::PathBuf;

fn source_info(id: usize, width: u32, height: u32) -> ImageInfo {
    ImageInfo {
        id,
        filename: format!("/fixture/source-{id}.NEF"),
        width,
        height,
        alignment_image: GrayImage::new(1, 1),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: None,
        source_footprint: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

fn masks(source: &ImageInfo, legend: Vec<PathBuf>) -> FocusStackTileMasks {
    let pixels = source.width as usize * source.height as usize;
    FocusStackTileMasks {
        source_refinements: HashMap::from([(
            source.id,
            FocusSourceRefinement {
                source_sampling: Some(FocusSourceSampling::for_test(
                    source,
                    &Matrix3::identity(),
                    Projection::Planar,
                    None,
                )),
                ..Default::default()
            },
        )]),
        coverage: CoverageMask::from_bytes(source.width, source.height, vec![255; pixels]).unwrap(),
        ownership: OwnershipMap::new(source.width, source.height, vec![1; pixels], legend).unwrap(),
        confidence: ConfidenceMap::zero(source.width, source.height),
        textured: vec![255; pixels],
        sharpness_evidence: focus_fuser::StationFusion::new(source.width, source.height, 1)
            .sharpness_evidence(),
        native_displacements: Vec::new(),
        fusion: FusionReport::default(),
    }
}

#[test]
fn raster_only_payload_skips_owner_resolution_and_source_loading() {
    let source = source_info(0, 4, 4);
    let mut masks = masks(&source, vec![PathBuf::from(&source.filename)]);
    masks.source_refinements.clear();
    let mut stats = TilePayloadReport::default();
    record_request(&mut stats, FocusTileLoadPurpose::RasterOnly);
    let result = load_owner_sources(
        FocusTileLoadPurpose::RasterOnly,
        Some(&masks),
        (1.25, -2.5),
        &[],
        &mut |_| -> Result<Rgb32FImage, String> { panic!("raster-only request loaded an owner") },
        &mut stats,
    )
    .unwrap();
    assert!(result.is_none());
    assert_eq!(stats.raster_only_requests, 1);
    assert_eq!(stats.raster_only_payload_skips, 1);
    assert_eq!(stats.owner_payload_build_requests, 0);
    assert_eq!(stats.owner_source_load_requests, 0);
    // The same attributed owner must still fail without its original when a
    // caller actually needs source sampling.
    let error = load_owner_sources(
        FocusTileLoadPurpose::OwnerSampling,
        Some(&masks),
        (0.0, 0.0),
        &[],
        &mut |_| -> Result<Rgb32FImage, String> { panic!("unresolved source was loaded") },
        &mut stats,
    )
    .err()
    .unwrap();
    assert_eq!(error, "Missing original for owner /fixture/source-0.NEF");
}

#[test]
fn owner_payload_preserves_zero_owned_legend_slots_without_loading_them() {
    let source = source_info(0, 4, 4);
    let masks = masks(
        &source,
        vec![
            PathBuf::from(&source.filename),
            PathBuf::from("/fixture/excluded.NEF"),
        ],
    );
    let mut loaded = Vec::new();
    let mut stats = TilePayloadReport::default();
    let result = load_owner_sources(
        FocusTileLoadPurpose::OwnerSampling,
        Some(&masks),
        (1.25, -2.5),
        &[&source],
        &mut |info| {
            loaded.push(info.id);
            Ok(Rgb32FImage::from_pixel(4, 4, Rgb([0.5; 3])))
        },
        &mut stats,
    )
    .unwrap()
    .unwrap();
    assert_eq!(loaded, vec![0]);
    assert_eq!(result.originals.len(), 2);
    assert!(result.originals[0].is_some());
    assert!(result.originals[1].is_none());
    assert_eq!(result.ownership, masks.ownership);
    assert_eq!(result.origin, (1.25, -2.5));
    assert_eq!(stats.owner_payload_build_requests, 1);
    assert_eq!(stats.owner_source_load_requests, 1);
}

#[test]
fn owner_payload_keeps_missing_sampling_and_loader_errors() {
    let source = source_info(0, 4, 4);
    let mut masks = masks(&source, vec![PathBuf::from(&source.filename)]);
    let mut stats = TilePayloadReport::default();
    masks
        .source_refinements
        .get_mut(&0)
        .unwrap()
        .source_sampling = None;
    let error = load_owner_sources(
        FocusTileLoadPurpose::OwnerSampling,
        Some(&masks),
        (0.0, 0.0),
        &[&source],
        &mut |_| -> Result<Rgb32FImage, String> { panic!("source without sampling was loaded") },
        &mut stats,
    )
    .err()
    .unwrap();
    assert_eq!(error, "Missing source sampling for /fixture/source-0.NEF");
    assert_eq!(stats.owner_source_load_requests, 0);

    masks
        .source_refinements
        .get_mut(&0)
        .unwrap()
        .source_sampling = Some(FocusSourceSampling::for_test(
        &source,
        &Matrix3::identity(),
        Projection::Planar,
        None,
    ));
    let error = load_owner_sources(
        FocusTileLoadPurpose::OwnerSampling,
        Some(&masks),
        (0.0, 0.0),
        &[&source],
        &mut |_| Err("fixture decode failed".to_string()),
        &mut stats,
    )
    .err()
    .unwrap();
    assert_eq!(error, "fixture decode failed");
    assert_eq!(stats.owner_payload_build_requests, 2);
    assert_eq!(stats.owner_source_load_requests, 1);
}

fn compose_fixture(
    always_build: bool,
) -> (
    LayeredOwnershipRender,
    Vec<(usize, FocusTileLoadPurpose)>,
    TilePayloadReport,
) {
    tone::reset_run_records();
    tone::reset_run_topology();
    residual_warp::reset_run_records();
    // Seam-only topology gives a real planning/apply pass without changing the
    // photometric model in this payload-provenance fixture.
    tone::record_run_topology(tone::ToneStationTopology {
        stations: vec![(0, 0), (1, 1)],
        seam_candidates: vec![(0, 1)],
        ..Default::default()
    });
    let infos = [source_info(0, 96, 64), source_info(1, 96, 64)];
    let refs = infos.iter().collect::<Vec<_>>();
    let masks = infos
        .iter()
        .map(|source| masks(source, vec![PathBuf::from(&source.filename)]))
        .collect::<Vec<_>>();
    let homographies = HashMap::from([
        (0, Matrix3::identity()),
        (
            1,
            Matrix3::new(1.0, 0.0, 72.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        ),
    ]);
    let mut requests = Vec::new();
    let mut stats = TilePayloadReport::default();
    let mut loader = |info: &ImageInfo, purpose: FocusTileLoadPurpose| {
        requests.push((info.id, purpose));
        record_request(&mut stats, purpose);
        let sources = load_owner_sources(
            if always_build {
                FocusTileLoadPurpose::OwnerSampling
            } else {
                purpose
            },
            Some(&masks[info.id]),
            (0.0, 0.0),
            &[&infos[info.id]],
            &mut |_| Ok(Rgb32FImage::from_pixel(96, 64, Rgb([0.72, 0.62, 0.52]))),
            &mut stats,
        )?;
        // Deliberately different from the originals: silently sampling this
        // fused raster in the delivered pass cannot pass the provenance check.
        Ok::<_, String>(FocusTilePayload {
            image: Rgb32FImage::from_pixel(96, 64, Rgb([0.2, 0.25, 0.3])),
            sources,
        })
    };
    let app = tauri::test::mock_app();
    let rendered = stitching::layered_virtual_tile_compositor_with_ownership(
        &refs,
        &homographies,
        Projection::Planar,
        app.handle().clone(),
        "tile-payload-purpose-test",
        &mut loader,
    )
    .unwrap();
    tone::reset_run_records();
    tone::reset_run_topology();
    residual_warp::reset_run_records();
    (rendered, requests, stats)
}

#[test]
fn compositor_routes_raster_planning_and_preserves_native_payload_pixels() {
    let _scope = degradation::begin_run_scope();
    let (baseline, baseline_requests, baseline_stats) = compose_fixture(true);
    let (selected, requests, stats) = compose_fixture(false);
    assert_eq!(requests, baseline_requests);
    assert_eq!(requests[0], (0, FocusTileLoadPurpose::OwnerSampling));
    assert_eq!(requests[1], (1, FocusTileLoadPurpose::OwnerSampling));
    let raster = requests
        .iter()
        .enumerate()
        .filter_map(|(index, &(_, purpose))| {
            (purpose == FocusTileLoadPurpose::RasterOnly).then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        raster,
        vec![2, 3],
        "only the seam planning pair needs raster-only payloads"
    );
    assert!(
        requests.len() >= 6,
        "the seam application must request both originals"
    );
    assert!(
        requests[4..]
            .iter()
            .all(|&(_, purpose)| purpose == FocusTileLoadPurpose::OwnerSampling)
    );
    assert_eq!(stats.raster_only_requests, 2);
    assert_eq!(stats.raster_only_payload_skips, 2);
    assert_eq!(
        stats.owner_payload_build_requests,
        stats.owner_sampling_requests
    );
    assert_eq!(
        stats.owner_source_load_requests,
        stats.owner_sampling_requests
    );
    assert_eq!(
        baseline_stats.owner_source_load_requests,
        stats.owner_source_load_requests + 2
    );
    assert_eq!(selected.image.as_raw(), baseline.image.as_raw());
    assert_eq!(
        selected.coverage.dimensions(),
        baseline.coverage.dimensions()
    );
    assert_eq!(selected.coverage.covered(), baseline.coverage.covered());
    assert_eq!(selected.ownership, baseline.ownership);
    assert_eq!(selected.sampling_origin, baseline.sampling_origin);
    assert!(
        selected.image.pixels().any(|pixel| pixel[0] > 0.5),
        "the delivered image must actually sample originals, not the different fused raster"
    );
}

#[test]
fn tile_payload_report_roundtrips_and_defaults_for_older_reports() {
    let mut report = StackReport::default();
    report.performance.tile_payload = TilePayloadReport {
        raster_only_requests: 2,
        owner_sampling_requests: 4,
        owner_payload_build_requests: 4,
        owner_source_load_requests: 6,
        raster_only_payload_skips: 2,
    };
    let mut value = serde_json::to_value(&report).unwrap();
    let restored: StackReport = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(
        restored.performance.tile_payload,
        report.performance.tile_payload
    );
    value["performance"]
        .as_object_mut()
        .unwrap()
        .remove("tile_payload");
    let older: StackReport = serde_json::from_value(value).unwrap();
    assert_eq!(older.performance.tile_payload, TilePayloadReport::default());
    let partial: PerformanceReport = serde_json::from_value(serde_json::json!({
        "tile_payload": {"raster_only_requests": 1}
    }))
    .unwrap();
    assert_eq!(partial.tile_payload.raster_only_requests, 1);
    assert_eq!(partial.tile_payload.owner_source_load_requests, 0);
}

#[test]
fn native_owner_fallback_matches_compositor_reverse_evidence_and_quality_reference() {
    use crate::panorama_utils::stack_pipeline::{quality_gate, quality_gate_runner};
    use nalgebra::Point2;
    let _scope = degradation::begin_run_scope();
    tone::reset_run_records();
    tone::reset_run_topology();
    residual_warp::reset_run_records();
    let a = source_info(0, 16, 16);
    let b = source_info(1, 16, 16);
    let tile = source_info(100, 16, 16);
    let origin = (2.0, -1.0);
    let source_to_station = Matrix3::new(1.0, 0.0, origin.0, 0.0, 1.0, origin.1, 0.0, 0.0, 1.0);
    let mut masks = masks(
        &a,
        vec![a.filename.clone().into(), b.filename.clone().into()],
    );
    masks.ownership = OwnershipMap::new(
        16,
        16,
        (0..256).map(|i| if i % 16 <= 8 { 1 } else { 2 }).collect(),
        masks.ownership.legend().to_vec(),
    )
    .unwrap();
    masks.confidence = ConfidenceMap::per_pixel(
        16,
        16,
        (0..256)
            .map(|i| if i % 16 <= 8 { 0.1 } else { 0.8 })
            .collect(),
    )
    .unwrap();
    masks.textured = (0..256)
        .map(|i| if i % 16 <= 8 { 0 } else { 255 })
        .collect();
    masks.source_refinements = [&a, &b]
        .into_iter()
        .map(|info| {
            (
                info.id,
                FocusSourceRefinement {
                    translation: if info.id == a.id { (9, 0) } else { (0, 0) },
                    source_sampling: Some(FocusSourceSampling::for_test(
                        info,
                        &source_to_station,
                        Projection::Planar,
                        None,
                    )),
                    ..Default::default()
                },
            )
        })
        .collect();
    // Reverse member order deliberately: the published legend is the contract.
    let members = [&b, &a];
    let originals = [
        Rgb32FImage::from_pixel(16, 16, Rgb([0.9, 0.1, 0.1])),
        Rgb32FImage::from_fn(16, 16, |x, y| {
            Rgb([0.2 + x as f32 * 0.01, 0.5 + y as f32 * 0.01, 0.3])
        }),
    ];
    let fused = Rgb([0.3; 3]);
    let mut stats = TilePayloadReport::default();
    let mut loader = |_: &ImageInfo, purpose| {
        Ok::<_, String>(FocusTilePayload {
            image: Rgb32FImage::from_pixel(16, 16, fused),
            sources: load_owner_sources(
                purpose,
                Some(&masks),
                origin,
                &members,
                &mut |info| Ok(originals[info.id].clone()),
                &mut stats,
            )?,
        })
    };
    let station_to_world = Matrix3::new(1.0, 0.0, -0.25, 0.0, 1.0, -0.5, 0.0, 0.0, 1.0);
    let app = tauri::test::mock_app();
    let rendered = stitching::layered_virtual_tile_compositor_with_ownership(
        &[&tile],
        &HashMap::from([(tile.id, station_to_world)]),
        Projection::Planar,
        app.handle().clone(),
        "native-owner-evidence-chain",
        &mut loader,
    )
    .unwrap();
    let world = Point2::new(8.0, 8.0);
    let local = Point2::new(8.25, 8.5);
    let slots = validated_owner_slots(&masks, &members).unwrap();
    let resolved = resolve_owner_evidence(&masks, &slots, &[11, 22], origin, local).unwrap();
    assert!(resolved.geometry.used_fallback);
    assert_eq!(resolved.geometry.local_owner, 2);
    assert_eq!(resolved.raw_owner, 22);
    assert_eq!(resolved.geometry.evidence_pixel, (9, 8));
    assert_eq!(resolved.geometry.prepared_point, local);
    assert_eq!(resolved.confidence, 0.8);
    assert_eq!(resolved.textured, 255);
    assert_eq!(resolved.sharpness, masks.sharpness_evidence.at_pixel(9, 8));
    let selected = slots[usize::from(resolved.geometry.local_owner - 1)]
        .member
        .unwrap();
    let source = quality_gate_runner::QualitySource {
        owner: resolved.raw_owner,
        path: selected.filename.clone(),
        dimensions: selected.dimensions(),
        geometry: quality_gate::SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: station_to_world,
            station_id: 0,
        },
        native_warp: None,
        native_registration: Some(quality_gate_runner::NativeReferenceMap {
            mosaic_tone: None,
            source_prefilter_divisor: 1.0,
            world_to_station: station_to_world.try_inverse().unwrap(),
            station_to_world,
            station_origin: origin,
            refinement: slots[usize::from(resolved.geometry.local_owner - 1)]
                .refinement
                .unwrap()
                .clone(),
        }),
    };
    assert_eq!(source.path, b.filename);
    let reference = quality_gate_runner::sample_quality_source_at_world(
        &originals[selected.id],
        world,
        &source,
        &residual_warp::ResidualWarp::default(),
    )
    .unwrap();
    let output_xy = (
        world.x - rendered.sampling_origin.0,
        world.y - rendered.sampling_origin.1,
    );
    assert_eq!(output_xy.0.fract(), 0.0);
    assert_eq!(output_xy.1.fract(), 0.0);
    let (x, y) = (output_xy.0 as u32, output_xy.1 as u32);
    assert_eq!(
        rendered.coverage.covered()[y as usize * rendered.image.width() as usize + x as usize],
        255
    );
    let actual = *rendered.image.get_pixel(x, y);
    assert_eq!(actual, reference);
    assert_ne!(actual, fused);
    assert_ne!(actual, *originals[0].get_pixel(8, 8));
    tone::reset_run_records();
    tone::reset_run_topology();
    residual_warp::reset_run_records();
}

#[test]
fn owner_payload_rejects_snapshot_and_buffer_dimension_mismatches() {
    let source = source_info(0, 4, 4);
    let mut masks = masks(&source, vec![source.filename.clone().into()]);
    let wrong = source_info(0, 5, 4);
    masks
        .source_refinements
        .get_mut(&0)
        .unwrap()
        .source_sampling = Some(FocusSourceSampling::for_test(
        &wrong,
        &Matrix3::identity(),
        Projection::Planar,
        None,
    ));
    let mut stats = TilePayloadReport::default();
    let error = load_owner_sources(
        FocusTileLoadPurpose::OwnerSampling,
        Some(&masks),
        (0.0, 0.0),
        &[&source],
        &mut |_| -> Result<Rgb32FImage, String> { panic!("invalid snapshot reached loader") },
        &mut stats,
    )
    .err()
    .unwrap();
    assert!(error.starts_with("Source sampling dimensions mismatch"));
    assert_eq!(stats.owner_source_load_requests, 0);
    masks
        .source_refinements
        .get_mut(&0)
        .unwrap()
        .source_sampling = Some(FocusSourceSampling::for_test(
        &source,
        &Matrix3::identity(),
        Projection::Planar,
        None,
    ));
    let error = load_owner_sources(
        FocusTileLoadPurpose::OwnerSampling,
        Some(&masks),
        (0.0, 0.0),
        &[&source],
        &mut |_| Ok(Rgb32FImage::new(5, 4)),
        &mut stats,
    )
    .err()
    .unwrap();
    assert!(error.starts_with("Prepared source dimensions mismatch"));
    assert_eq!(stats.owner_source_load_requests, 1);
}
