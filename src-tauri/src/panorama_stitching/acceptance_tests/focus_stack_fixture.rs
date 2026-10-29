//! The real focus-stack fixture harness run by tests/focus-stack-regression.mjs, its layered
//! native acceptance check, and the tests that read the alignment cache it writes.

use super::*;

fn verify_layered_native_acceptance(report: &stack_report::StackReport) -> Result<(), String> {
    use stack_report::{ClosureStatus, QualityGateVerdict, SelectedPath, ToneStatus};
    if report.selected_path != SelectedPath::LayeredVirtualTile {
        return Ok(());
    }
    if report.station_relations.connectivity.components != 1
        || report.closure.status != ClosureStatus::Converged
    {
        return Err("native station geometry is not connected and converged".into());
    }
    let criteria = &report.quality_gate.criteria;
    if report.quality_gate.verdict != QualityGateVerdict::Pass
        || criteria.len() != quality_gate::QUALITY_CRITERIA.len()
    {
        return Err("native output did not pass the complete quality gate".into());
    }
    for name in quality_gate::QUALITY_CRITERIA {
        let matches = criteria
            .iter()
            .filter(|criterion| criterion.name == name)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(format!("missing or duplicate quality criterion: {name}"));
        }
        let criterion = matches[0];
        let accepted = criterion.verdict == QualityGateVerdict::Pass
            || (name == "mtf50_normalized"
                && criterion.verdict == QualityGateVerdict::NotApplicable
                && criterion.technical_unmeasurable_count == 0);
        if !accepted {
            return Err(format!("native quality criterion failed: {name}"));
        }
    }
    let maximum = report.tone.boundary_delta_e.max;
    if report.tone.status != ToneStatus::Applied
        || !maximum.is_finite()
        || maximum > stack_report::TONE_BOUNDARY_DELTA_E_THRESHOLD
    {
        return Err(format!(
            "native tone boundary failed: {maximum:.6} Delta E00"
        ));
    }
    Ok(())
}

#[test]
fn native_acceptance_rejects_degraded_tone_even_when_all_quality_criteria_pass() {
    let mut report = stack_report::StackReport::default();
    report.station_relations.connectivity.components = 1;
    report.closure.status = stack_report::ClosureStatus::Converged;
    report.quality_gate.verdict = stack_report::QualityGateVerdict::Pass;
    report.quality_gate.criteria = quality_gate::QUALITY_CRITERIA
        .iter()
        .map(|name| stack_report::QualityGateCriterionRecord {
            name: (*name).into(),
            verdict: stack_report::QualityGateVerdict::Pass,
            ..Default::default()
        })
        .collect();
    report.tone.status = stack_report::ToneStatus::Applied;
    assert!(verify_layered_native_acceptance(&report).is_ok());
    report.tone.status = stack_report::ToneStatus::Degraded;
    report.tone.boundary_delta_e.max = 9.8;
    assert!(
        verify_layered_native_acceptance(&report)
            .unwrap_err()
            .contains("tone boundary failed")
    );
    report.tone.status = stack_report::ToneStatus::Applied;
    report.tone.boundary_delta_e.max = 0.0;
    let mtf = report
        .quality_gate
        .criteria
        .iter_mut()
        .find(|c| c.name == "mtf50_normalized")
        .unwrap();
    mtf.verdict = stack_report::QualityGateVerdict::NotApplicable;
    assert!(verify_layered_native_acceptance(&report).is_ok());
    report
        .quality_gate
        .criteria
        .iter_mut()
        .find(|c| c.name == "mtf50_normalized")
        .unwrap()
        .technical_unmeasurable_count = 1;
    assert!(verify_layered_native_acceptance(&report).is_err());
}

#[test]
#[ignore = "renders one real station from the saved source alignment for native diagnostics"]
fn real_saved_capture_station_from_env() {
    let _scope = degradation::begin_run_scope();
    let cache: serde_json::Value = serde_json::from_slice(
        &fs::read(std::env::var("RAW_EDITOR_NATIVE_STATION_ALIGNMENT").unwrap()).unwrap(),
    )
    .unwrap();
    let station: usize = std::env::var("RAW_EDITOR_NATIVE_STATION_INDEX")
        .unwrap()
        .parse()
        .unwrap();
    let output = PathBuf::from(std::env::var("RAW_EDITOR_NATIVE_STATION_OUTPUT").unwrap());
    crate::sidecar_storage::initialize(
        PathBuf::from("/private/tmp/raw-editor-stack-sidecars").as_path(),
    )
    .unwrap();
    let all_paths = cache["sources"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["filename"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    let settings = AppSettings::default();
    let exposure = stack_raw_exposure_reference(&all_paths);
    let group = &cache["capture_groups"][station];
    let ids = group["member_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let matrix = |value: &serde_json::Value| {
        let coefficients = value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        Matrix3::from_row_slice(&coefficients)
    };
    let global = cache["global_homographies"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(id, h)| (id.parse::<usize>().unwrap(), matrix(h)))
        .collect::<HashMap<_, _>>();
    let anchor = group["anchor_id"].as_u64().unwrap() as usize;
    let inverse_anchor = global[&anchor].try_inverse().unwrap();
    let local = ids
        .iter()
        .map(|&id| {
            (
                id,
                normalized_homography(&(inverse_anchor * global[&id])).unwrap(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut warp = FocusLayerWarp {
        bands: cache["focus_warp"]
            .as_array()
            .unwrap()
            .iter()
            .map(|band| {
                let ranges = |key: &str| {
                    band[key]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(id, r)| {
                            (
                                id.parse::<usize>().unwrap(),
                                (r[0].as_f64().unwrap(), r[1].as_f64().unwrap()),
                            )
                        })
                        .collect::<HashMap<_, _>>()
                };
                FocusWarpBand {
                    homographies: band["homographies"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(id, h)| {
                            (
                                id.parse::<usize>().unwrap(),
                                normalized_homography(&(inverse_anchor * matrix(h))).unwrap(),
                            )
                        })
                        .collect(),
                    source_ranges: ranges("source_ranges"),
                    source_x_ranges: ranges("source_x_ranges"),
                    foreground_only: band["foreground_only"].as_bool().unwrap(),
                    relax_foreground_seam: band["relax_foreground_seam"].as_bool().unwrap(),
                    physical_edge: band["physical_edge"].as_bool().unwrap(),
                }
            })
            .collect(),
    };
    if std::env::var_os("RAW_EDITOR_NATIVE_STATION_NO_GENERIC_WARP").is_some() {
        warp.bands.retain(|band| band.physical_edge);
    }
    let mut images = Vec::new();
    for id in ids {
        let source = cache["sources"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["id"].as_u64().unwrap() as usize == id)
            .unwrap();
        let path = source["filename"].as_str().unwrap().to_string();
        let prepared = load_prepared_stack_source(&path, &settings, exposure).unwrap();
        let (w, h, scale) = processing::calculate_downscale_dimensions_capped(
            prepared.image.width(),
            prepared.image.height(),
            2400,
        );
        let alignment_image = prepared
            .image
            .resize_exact(w, h, image::imageops::FilterType::Triangle)
            .to_luma8();
        let foreground_range = detect_foreground_range(&alignment_image);
        images.push(ImageInfo {
            id,
            filename: path,
            width: prepared.image.width(),
            height: prepared.image.height(),
            scale_factor: scale,
            focal_length_35mm: prepared.focal_length_35mm,
            source_footprint: prepared.source_footprint,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_mask: build_foreground_mask(&alignment_image, foreground_range),
            foreground_range,
            horizontal_edge_rows: horizontal_edge_row_candidates(
                &alignment_image,
                foreground_range,
            ),
            vertical_edge_columns: detect_vertical_edge_columns(&alignment_image, foreground_range),
            alignment_image,
            full_image: None,
        });
    }
    if std::env::var_os("RAW_EDITOR_NATIVE_STATION_EDGE_DIAGNOSTICS").is_some() {
        for info in &images {
            println!(
                "Station physical candidates {} H {:?} V {:?}",
                info.id, info.horizontal_edge_rows, info.vertical_edge_columns
            );
            for row in &info.horizontal_edge_rows {
                println!(
                    "Physical curve samples H {} {} {:?}",
                    info.id,
                    row,
                    horizontal_edge_line_samples(info, &local[&info.id], *row)
                );
                println!(
                    "Station horizontal fit {} {} {:?}",
                    info.id,
                    row,
                    fit_focus_horizontal_edge_line(info, &local[&info.id], *row)
                );
            }
            for column in &info.vertical_edge_columns {
                println!(
                    "Physical curve samples V {} {} {:?}",
                    info.id,
                    column,
                    vertical_edge_line_samples(info, &local[&info.id], *column, None)
                );
                println!(
                    "Station vertical fit {} {} {:?}",
                    info.id,
                    column,
                    fit_focus_vertical_edge_line(info, &local[&info.id], *column, None)
                );
            }
        }
        let h = build_focus_horizontal_edge_bands(&images, &local);
        let v = build_focus_vertical_edge_bands_in_region(&images, &local, false);
        println!("Station physical bands H={} V={}", h.len(), v.len());
    }
    let references = images.iter().collect::<Vec<_>>();
    let mut load = |info: &ImageInfo| {
        load_prepared_stack_source(&info.filename, &settings, exposure).map(|p| p.image.to_rgb32f())
    };
    let app = tauri::test::mock_app();
    let rendered = stitching::focus_stack_stitcher_unfilled(
        &references,
        &local,
        Projection::Planar,
        (!warp.bands.is_empty()).then_some(&warp),
        None,
        station,
        false,
        app.handle().clone(),
        "station-native-diagnostic",
        &mut load,
    )
    .unwrap();
    if let Some(masks) = rendered.masks.as_ref() {
        let (width, height) = masks.coverage.dimensions();
        image::GrayImage::from_raw(width, height, masks.coverage.covered().to_vec())
            .unwrap()
            .save(output.with_extension("coverage.png"))
            .unwrap();
    }
    if let (Some(roi), Some(masks)) = (
        std::env::var("RAW_EDITOR_NATIVE_STATION_ROI").ok(),
        rendered.masks.as_ref(),
    ) {
        let r = roi
            .split(',')
            .map(|v| v.parse::<u32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(r.len(), 4);
        let owner =
            image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_fn(r[2], r[3], |x, y| {
                image::Luma([masks.ownership.owner_at(r[0] + x, r[1] + y)])
            });
        owner.save(output.with_extension("roi-owners.png")).unwrap();
        for info in &images {
            let Some(refinement) = masks.source_refinements.get(&info.id) else {
                continue;
            };
            let source = load_prepared_stack_source(&info.filename, &settings, exposure)
                .unwrap()
                .image
                .to_rgb32f();
            let patch = image::Rgb32FImage::from_fn(r[2], r[3], |x, y| {
                let point = refinement
                    .inverse_canvas_point(Point2::new((r[0] + x) as f64, (r[1] + y) as f64));
                refinement
                    .source_sampling
                    .as_ref()
                    .and_then(|s| {
                        s.source_point(Point3::new(
                            point.x + rendered.sampling_origin.0,
                            point.y + rendered.sampling_origin.1,
                            1.0,
                        ))
                    })
                    .map_or(image::Rgb([0.0; 3]), |p| {
                        stitching::get_high_quality_interpolated_pixel(&source, p.x, p.y)
                    })
            });
            DynamicImage::ImageRgb32F(patch)
                .to_rgb16()
                .save(output.with_extension(format!("roi-source-{}.png", info.id)))
                .unwrap();
        }
    }
    let canonical = crate::image_stack::canonicalize_image_stack_result(DynamicImage::ImageRgb32F(
        rendered.image,
    ));
    canonical.save(&output).unwrap();
    canonical
        .resize(1800, 1800, image::imageops::FilterType::Lanczos3)
        .save(output.with_extension("preview.jpg"))
        .unwrap();
    println!(
        "Native station diagnostic only: {}x{} origin {:?}; {} generic/physical bands",
        canonical.width(),
        canonical.height(),
        rendered.sampling_origin,
        warp.bands.len()
    );
}

#[test]
#[ignore = "requires user-provided focus-stack paths and writes a large temporary output"]
fn real_focus_stack_fixture_from_env() {
    stitching::set_test_tone_capture_directory(
        std::env::var_os("RAW_EDITOR_TONE_DIAGNOSTICS_DIR").map(PathBuf::from),
    );
    let encoded_paths = std::env::var_os("RAW_EDITOR_FOCUS_STACK_PATHS")
        .expect("RAW_EDITOR_FOCUS_STACK_PATHS must contain platform-separated image paths");
    let paths: Vec<String> = std::env::split_paths(&encoded_paths)
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    assert!(
        paths.len() >= 2,
        "at least two focus-stack paths are required"
    );

    let alignment_name = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_ALIGNMENT")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    let alignment_mode = AlignmentMode::from_wire(&alignment_name);
    let blend_mode = match std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_BLEND")
        .unwrap_or_else(|_| "focus-stack".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "panorama" => BlendMode::Panorama,
        _ => BlendMode::FocusStack,
    };
    let app = tauri::test::mock_app();
    crate::sidecar_storage::initialize(
        PathBuf::from("/private/tmp/raw-editor-stack-sidecars").as_path(),
    )
    .expect("test sidecar storage should initialize once");
    let input_paths = paths.clone();
    let StitchOutcome {
        image: stitched_image,
        coverage: stitched_coverage,
        ordered_paths: stitched_order,
        stack_report_path,
        ..
    } = stitch_images_with_options(
        paths,
        app.handle().clone(),
        alignment_mode,
        blend_mode,
        "test-image-stack-progress",
    )
    .expect("the provided focus-stack images should align and blend");

    let output_path = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "/private/tmp/raw-editor-focus-stack-{alignment_name}.tiff"
            ))
        });
    // Production writes the Coverage_Mask as alpha; the harness keeps the
    // historical opaque RGB form unless a run asks for the same alpha.
    let canonical = if std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_COVERAGE_ALPHA").is_some() {
        crate::image_stack::canonicalize_image_stack_result_with_coverage(
            stitched_image,
            stitched_coverage.as_ref(),
        )
    } else {
        crate::image_stack::canonicalize_image_stack_result(stitched_image)
    };
    let preview_only = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_PREVIEW_ONLY").is_some();
    let preview_edge = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_PREVIEW_EDGE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value >= 256)
        .unwrap_or(1800);
    let preview = DynamicImage::ImageRgb8(
        canonical
            .resize(
                preview_edge,
                preview_edge,
                image::imageops::FilterType::Lanczos3,
            )
            .to_rgb8(),
    );
    let preview_path = output_path.with_extension("preview.jpg");
    preview
        .save_with_format(&preview_path, ImageFormat::Jpeg)
        .expect("focus-stack preview should be writable");
    if std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_CROPS").is_some() {
        let (width, height) = canonical.dimensions();
        let crop_width = (width / 3).max(1);
        let crop_height = (height / 2).max(1);
        for (name, x, y) in [
            ("top-left", 0, 0),
            ("top-center", width.saturating_sub(crop_width) / 2, 0),
            ("top-right", width.saturating_sub(crop_width), 0),
            ("bottom-left", 0, height.saturating_sub(crop_height)),
            (
                "bottom-center",
                width.saturating_sub(crop_width) / 2,
                height.saturating_sub(crop_height),
            ),
            (
                "bottom-right",
                width.saturating_sub(crop_width),
                height.saturating_sub(crop_height),
            ),
        ] {
            let crop = DynamicImage::ImageRgb8(
                canonical
                    .crop_imm(x, y, crop_width.min(width - x), crop_height.min(height - y))
                    .resize(
                        preview_edge,
                        preview_edge,
                        image::imageops::FilterType::Lanczos3,
                    )
                    .to_rgb8(),
            );
            crop.save_with_format(
                output_path.with_file_name(format!(
                    "{}.{}.jpg",
                    output_path
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .unwrap_or("focus-stack"),
                    name
                )),
                ImageFormat::Jpeg,
            )
            .expect("focus-stack crop preview should be writable");
        }
    }
    let app_jpeg_path = output_path.with_extension("app.jpg");
    if !preview_only {
        // Keep visual diagnostics even if a later large export runs out
        // of disk space. Export failure still fails this acceptance run.
        crate::image_stack::write_srgb_tiff(&canonical, &output_path)
            .expect("full-resolution color-managed TIFF result should be writable");
        crate::image_stack::write_srgb_jpeg(&canonical, &app_jpeg_path)
            .expect("full-resolution application JPEG should be writable");
    }

    // Regression metrics (tests/focus-stack-regression.mjs): input order
    // completeness, tile sharpness and, with a reference rendering, the
    // registered per-tile sharpness ratio. Measurement only.
    if let Some(metrics_path) =
        std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_METRICS").map(PathBuf::from)
    {
        let reference =
            std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_METRICS_REFERENCE").map(PathBuf::from);
        let analysis_long_side = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_METRICS_LONG_SIDE")
            .ok()
            .and_then(|value| value.parse::<u32>().ok());
        let metrics = crate::panorama_stitching::acceptance_metrics::write_acceptance_metrics(
            &crate::panorama_stitching::acceptance_metrics::AcceptanceMetricsRequest {
                image: &canonical,
                coverage: stitched_coverage.as_ref(),
                input_paths: &input_paths,
                ordered_paths: &stitched_order,
                reference: reference.as_deref(),
                analysis_long_side,
            },
            &metrics_path,
        )
        .expect("acceptance metrics should be measurable and writable");
        println!(
            "acceptance metrics: {} (reference ratio median {:?})",
            metrics_path.display(),
            metrics["reference"]["ratio"]["median"]
        );
    }

    if !preview_only
        && let Some(reference_path) =
            std::env::var_os("RAW_EDITOR_FOCUS_STACK_REFERENCE_JPEG").map(PathBuf::from)
    {
        let actual = fs::read(&app_jpeg_path)
            .expect("read the full-resolution application JPEG for comparison");
        let reference =
            fs::read(&reference_path).expect("read the user-provided Photoshop reference JPEG");
        assert!(
            actual == reference,
            "application JPEG does not match the Photoshop reference byte for byte: {} ({} bytes) vs {} ({} bytes)",
            app_jpeg_path.display(),
            actual.len(),
            reference_path.display(),
            reference.len()
        );
        println!(
            "Photoshop parity: byte-identical to {}",
            reference_path.display()
        );
    }
    println!(
        "focus-stack result: {}x{}\nfull: {}\napp JPEG: {}\npreview: {}",
        canonical.width(),
        canonical.height(),
        output_path.display(),
        app_jpeg_path.display(),
        preview_path.display()
    );
    // Save diagnostics first, but successful file writes cannot establish
    // acceptance. In particular, a degraded tone report must fail the run
    // even when all nine independent quality criteria passed.
    if let Some(path) = stack_report_path {
        let report: stack_report::StackReport = serde_json::from_slice(
            &fs::read(&path).expect("native acceptance report should be readable"),
        )
        .expect("native acceptance report should match its schema");
        verify_layered_native_acceptance(&report)
            .unwrap_or_else(|error| panic!("{error}; report: {}", path.display()));
    }
}

#[test]
#[ignore = "reads an alignment cache emitted by real_focus_stack_fixture_from_env"]
fn real_focus_stack_alignment_cache_contract() {
    let cache_path = std::env::var_os("RAW_EDITOR_STACK_ALIGNMENT_CACHE")
        .map(PathBuf::from)
        .expect("RAW_EDITOR_STACK_ALIGNMENT_CACHE must point to an alignment cache");
    let bytes = fs::read(&cache_path).expect("alignment cache should be readable");
    let cache: serde_json::Value =
        serde_json::from_slice(&bytes).expect("alignment cache should be valid JSON");
    assert_eq!(
        cache.get("schema").and_then(|value| value.as_u64()),
        Some(2)
    );
    let sources = cache
        .get("sources")
        .and_then(|value| value.as_array())
        .expect("alignment cache should contain source records");
    assert!(sources.len() >= 2, "alignment cache should contain a stack");
    let source_ids = sources
        .iter()
        .filter_map(|source| source.get("id").and_then(|value| value.as_u64()))
        .collect::<HashSet<_>>();
    let capture_groups = cache
        .get("capture_groups")
        .and_then(|value| value.as_array())
        .expect("focus alignment cache should contain capture groups");
    assert!(
        !capture_groups.is_empty(),
        "capture groups should not be empty"
    );
    let mut grouped_source_ids = HashSet::new();
    for group in capture_groups {
        let members = group
            .get("member_ids")
            .and_then(|value| value.as_array())
            .expect("capture group should list member ids");
        assert!(!members.is_empty(), "capture group should have members");
        for member in members {
            let member_id = member
                .as_u64()
                .expect("capture group member id should be numeric");
            assert!(
                grouped_source_ids.insert(member_id),
                "a source must belong to only one capture group"
            );
        }
        for metric in ["median_center_drift_px", "maximum_center_drift_px"] {
            let value = group
                .get(metric)
                .and_then(|value| value.as_f64())
                .expect("capture group drift should be numeric");
            assert!(value.is_finite(), "capture group drift should be finite");
        }
    }
    assert_eq!(grouped_source_ids, source_ids);
    let homographies = cache
        .get("global_homographies")
        .and_then(|value| value.as_object())
        .expect("alignment cache should contain global homographies");
    for source in sources {
        let id = source
            .get("id")
            .and_then(|value| value.as_u64())
            .expect("source record should have an id");
        let matrix = homographies
            .get(&id.to_string())
            .and_then(|value| value.as_array())
            .expect("every source should have a global homography");
        assert_eq!(matrix.len(), 9, "homography {id} should have nine values");
    }
    let matches = cache
        .get("matches")
        .and_then(|value| value.as_array())
        .expect("alignment cache should contain pair diagnostics");
    assert!(
        !matches.is_empty(),
        "alignment cache should retain at least one verified pair"
    );
    for pair in matches {
        assert!(
            pair.get("source_id")
                .and_then(|value| value.as_u64())
                .is_some()
                && pair
                    .get("target_id")
                    .and_then(|value| value.as_u64())
                    .is_some(),
            "pair diagnostics should identify both sources"
        );
        assert_eq!(
            pair.get("direct_homography")
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(9),
            "pair diagnostics should retain the measured transform"
        );
        assert!(
            pair.get("global_disagreement_px")
                .and_then(|value| value.as_f64())
                .is_some(),
            "pair diagnostics should report pose disagreement"
        );
    }
    let canvas = cache
        .get("canvas")
        .and_then(|value| value.as_object())
        .expect("alignment cache should contain canvas dimensions");
    assert!(
        canvas
            .get("width")
            .and_then(|value| value.as_u64())
            .unwrap_or(0)
            > 0
    );
    assert!(
        canvas
            .get("height")
            .and_then(|value| value.as_u64())
            .unwrap_or(0)
            > 0
    );
    println!(
        "alignment cache contract passed: {} sources, {} bytes, {}",
        sources.len(),
        bytes.len(),
        cache_path.display()
    );
}
