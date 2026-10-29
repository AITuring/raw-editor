//! Replays of saved native diagnostics named by environment variables: contour views, tone
//! fields and local tone, boundary edges, seam grids and joint geometry from measured points.

use super::*;

fn load_saved_tone_tile(path: PathBuf) -> crate::panorama_utils::stack_pipeline::tone::ToneTile {
    use crate::panorama_utils::stack_pipeline::tone::ToneTile;
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let width = value["width"].as_u64().unwrap() as u32;
    let height = value["height"].as_u64().unwrap() as u32;
    let rgb = |key: &str| {
        image::Rgb32FImage::from_raw(
            width,
            height,
            serde_json::from_value(value[key].clone()).unwrap(),
        )
        .unwrap()
    };
    ToneTile {
        station_index: value["station"].as_u64().unwrap() as usize,
        owner_id: value["owner"].as_u64().unwrap() as u16,
        low: rgb("low"),
        cell_mean: rgb("cell_mean"),
        validity: image::GrayImage::from_raw(
            width,
            height,
            serde_json::from_value(value["validity"].clone()).unwrap(),
        )
        .unwrap(),
        world_origin: serde_json::from_value(value["world_origin"].clone()).unwrap(),
        world_size: serde_json::from_value(value["world_size"].clone()).unwrap(),
        world_stride: value["world_stride"].as_f64().unwrap(),
        native_detail: fs::read(path.with_extension("native.f32"))
            .ok()
            .map(|bytes| {
                assert_eq!(bytes.len(), width as usize * height as usize * 16 * 3 * 4);
                image::Rgb32FImage::from_raw(
                    width * 4,
                    height * 4,
                    bytes
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                        .collect(),
                )
                .unwrap()
            }),
        cell_coverage: serde_json::from_value(value["cell_coverage"].clone()).unwrap(),
    }
}

#[test]
#[ignore = "requires original native contour view PNGs, reports measured profiles only"]
fn real_native_contour_views_from_env() {
    let paths =
        std::env::split_paths(&std::env::var_os("RAW_EDITOR_NATIVE_CONTOUR_VIEWS").unwrap())
            .collect::<Vec<_>>();
    let mut records = Vec::new();
    for (station, path) in paths.iter().enumerate() {
        let mut reader = image::ImageReader::open(path).unwrap();
        reader.no_limits();
        let image = reader.decode().unwrap().to_rgb32f();
        let coverage = image::open(path.with_extension("coverage.png"))
            .unwrap()
            .to_luma8();
        assert_eq!(coverage.dimensions(), image.dimensions());
        let tracks = crate::panorama_utils::stack_pipeline::line_layers::detect_native_contours(
            &image,
            |x, y| coverage.get_pixel(x, y)[0] != 0,
        );
        println!(
            "Native contour view {}: {} tracks",
            path.display(),
            tracks.len()
        );
        records.push(serde_json::json!({"path":path,"view_index":station,
            "tracks":tracks.iter().map(|p|p.iter().map(|p|[p.y,p.x]).collect::<Vec<_>>()).collect::<Vec<_>>()}));
    }
    fs::write(
        std::env::var_os("RAW_EDITOR_NATIVE_CONTOUR_OUTPUT").unwrap(),
        serde_json::to_vec(&records).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires two saved native tone fields; exports their unchanged common-support measurements"]
fn real_saved_tone_pair_from_env() {
    use crate::panorama_utils::stack_pipeline::tone::PairField;
    let paths = std::env::split_paths(&std::env::var_os("RAW_EDITOR_TONE_PAIR_FIELDS").unwrap())
        .collect::<Vec<_>>();
    assert_eq!(paths.len(), 2);
    let field = PairField::new(
        &load_saved_tone_tile(paths[0].clone()),
        &load_saved_tone_tile(paths[1].clone()),
    )
    .unwrap();
    fs::write(
        std::env::var_os("RAW_EDITOR_TONE_PAIR_OUTPUT").unwrap(),
        serde_json::to_vec(&field.diagnostic_grid()).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires saved native tone fields, ownership, and native relation proof"]
fn real_saved_local_tone_from_env() {
    use crate::panorama_utils::stack_pipeline::{compositor, tone, tone_local};
    let directory = PathBuf::from(std::env::var_os("RAW_EDITOR_LOCAL_TONE_FIELDS").unwrap());
    let mut reader = image::ImageReader::open(directory.join("tone-owners.png")).unwrap();
    // This is the trusted native ownership artifact emitted by this test
    // pipeline; its 300MP u16 plane exceeds the decoder's default limit.
    reader.no_limits();
    let owners = reader.decode().unwrap().to_luma16();
    let (width, height) = owners.dimensions();
    let mut tiles = Vec::new();
    for owner in 1..=u16::MAX {
        let path = directory.join(format!("tone-owner-{owner:03}.json"));
        if !path.exists() {
            break;
        }
        tiles.push(load_saved_tone_tile(path));
    }
    let native: serde_json::Value = serde_json::from_slice(
        &fs::read(std::env::var_os("RAW_EDITOR_LOCAL_TONE_NATIVE_PROOF").unwrap()).unwrap(),
    )
    .unwrap();
    let tile_index = |station: usize| {
        tiles
            .iter()
            .position(|tile| tile.station_index == station)
            .unwrap()
    };
    let mut relations = native["records"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|record| record["authoritative"].as_bool() != Some(false))
        .map(|record| {
            (
                tile_index(record["left"].as_u64().unwrap() as usize),
                tile_index(record["right"].as_u64().unwrap() as usize),
            )
        })
        .collect::<Vec<_>>();
    if let Some(path) = std::env::var_os("RAW_EDITOR_LOCAL_TONE_CANDIDATE_REPORT") {
        let report: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        for entry in report["degradation"]["entries"].as_array().unwrap() {
            if entry["reason"] != degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH {
                continue;
            }
            let d = &entry["detail"];
            let (left, right) = (
                d["left"].as_u64().unwrap() as usize,
                d["right"].as_u64().unwrap() as usize,
            );
            let Some(candidate) = report["station_relations"]["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| {
                    c["left"].as_u64() == Some(left as u64)
                        && c["right"].as_u64() == Some(right as u64)
                })
            else {
                continue;
            };
            if candidate["measurement_factor"].as_u64() != Some(1)
                || candidate["failure_stage"].as_str() != Some("fitted")
                || !candidate["residual_model"]
                    .as_str()
                    .is_some_and(|v| v.starts_with("native_polished_"))
                || candidate["bidirectional_match_count"].as_u64().unwrap_or(0)
                    < STATION_RELATION_MIN_INLIERS as u64
            {
                continue;
            }
            let measurements = StationRelationMeasurements {
                inliers: d["inliers"].as_u64().unwrap() as usize,
                median_error_px: d["median_error_px"].as_f64().unwrap(),
                scale_ratio: d["scale_ratio"].as_f64().unwrap(),
                spatial_support: d["spatial_support"].as_f64().unwrap(),
                low_frequency_mean_relative_difference: d["low_frequency_mean_relative_difference"]
                    .as_f64()
                    .unwrap(),
                edge_strength_ratio: d["edge_strength_ratio"].as_f64().unwrap(),
                median_edge_orientation_difference_degrees:
                    d["median_edge_orientation_difference_degrees"]
                        .as_f64()
                        .unwrap(),
                preserves_convex_orientation: d["preserves_convex_orientation"].as_bool().unwrap(),
            };
            if native_photometric_measurements_are_usable(&measurements) {
                println!(
                    "Original native photometric evidence {left}/{right}: {} inliers, {:.3}px, support {:.3}",
                    measurements.inliers,
                    measurements.median_error_px,
                    measurements.spatial_support
                );
                relations.push((tile_index(left), tile_index(right)));
            }
        }
        for pair in &mut relations {
            if pair.0 > pair.1 {
                *pair = (pair.1, pair.0);
            }
        }
        relations.sort_unstable();
        relations.dedup();
    }
    let use_saved = std::env::var_os("RAW_EDITOR_LOCAL_TONE_USE_SAVED_OWNERS").is_some();
    let plan = (!use_saved).then(|| {
        compositor::refine_station_ownership(&tiles, &relations, owners.as_raw(), width, height)
            .unwrap()
    });
    let planned = if let Some(plan) = plan.as_ref() {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| {
                if owners.get_pixel(x, y)[0] == 0 {
                    0
                } else {
                    plan.owner_at(x, y)
                }
            })
            .collect::<Vec<_>>()
    } else {
        owners.as_raw().clone()
    };
    if let Some(path) = std::env::var_os("RAW_EDITOR_LOCAL_TONE_PLANNED_OWNERS") {
        image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(width, height, planned.clone())
            .unwrap()
            .save(path)
            .unwrap();
    }
    let model = tone_local::fit_local_tone(&tiles, &relations, &planned, width, height).unwrap();
    let native_roi_prediction=std::env::var_os("RAW_EDITOR_LOCAL_TONE_ROI_JSON").map(|path| {
        let d:serde_json::Value=serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let reference=image::open(d["reference"].as_str().unwrap()).unwrap().to_rgb32f();
        let actual=image::open(d["actual"].as_str().unwrap()).unwrap().to_rgb32f();
        let x=d["x"].as_f64().unwrap();let y=d["y"].as_f64().unwrap();let scale=d["local_scale"].as_f64().unwrap();
        let tile=tiles.iter().position(|t|t.owner_id==d["owner"].as_u64().unwrap() as u16).unwrap();
        let old=tone_local::fit_local_tone_with_identity_prior(&tiles,&relations,&planned,width,height,d["old_prior"].as_f64().unwrap() as f32).unwrap();
        let predict=|model:&tone_local::LocalToneField| image::Rgb32FImage::from_fn(reference.width(),reference.height(),|px,py| {
            let (wx,wy)=(x+px as f64,y+py as f64);
            let low=tone::sample_tile_bilinear(&tiles[tile],wx,wy).unwrap();
            let corrected=model.corrected(tile,low,wx,wy);
            let source=reference.get_pixel(px,py);
            image::Rgb(std::array::from_fn(|c|(source[c]+corrected[c]-low[c]).clamp(0.0,1.0)))
        });
        let previous=predict(&old);let proposed=predict(&model);
        let error=previous.as_raw().iter().zip(actual.as_raw()).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
        let reference_energy=quality_gate::normalized_gradient_energy(&reference,scale).unwrap();
        let old_ratio=quality_gate::normalized_gradient_energy(&previous,scale).unwrap()/reference_energy;
        let new_ratio=quality_gate::normalized_gradient_energy(&proposed,scale).unwrap()/reference_energy;
        println!("True RAW ROI backtest max reconstruction error={error} old={old_ratio} proposed={new_ratio}");
        serde_json::json!({"true_reference_used":true,"old_reconstruction_error":error,"old_ratio":old_ratio,"proposed_ratio":new_ratio})
    });
    // Match production: every adjacent ownership transition, all 16 pixels
    // on each side, accumulated in the boundary pixel's 256px window.
    let step = 1u32;
    let columns = width.div_ceil(step);
    let rows = height.div_ceil(step);
    let mut fields = std::collections::BTreeMap::new();
    let mut windows =
        std::collections::BTreeMap::<(usize, usize, u32, u32), ([f64; 3], [f64; 3], usize)>::new();
    let witness_window = std::env::var("RAW_EDITOR_LOCAL_TONE_WITNESS_WINDOW")
        .ok()
        .map(|s| {
            let values = s
                .split(',')
                .map(|v| v.parse::<u32>().unwrap() / 256)
                .collect::<Vec<_>>();
            (values[0], values[1])
        });
    let mut witnesses =
        std::collections::BTreeMap::<(usize, usize, u32, u32), Vec<serde_json::Value>>::new();
    for gy in 0..rows {
        for gx in 0..columns {
            let (x, y) = (
                (gx * step + step / 2).min(width - 1),
                (gy * step + step / 2).min(height - 1),
            );
            let owner = planned[y as usize * width as usize + x as usize];
            for (nx, ny, horizontal) in [
                (x.saturating_add(step), y, true),
                (x, y.saturating_add(step), false),
            ] {
                if nx >= width || ny >= height {
                    continue;
                }
                let other = planned[ny as usize * width as usize + nx as usize];
                if owner == 0 || other == 0 || owner == other {
                    continue;
                }
                let a = tiles
                    .iter()
                    .position(|tile| tile.owner_id == owner)
                    .unwrap();
                let b = tiles
                    .iter()
                    .position(|tile| tile.owner_id == other)
                    .unwrap();
                let (first, second) = (a.min(b), a.max(b));
                let Some(field) = fields
                    .entry((first, second))
                    .or_insert_with(|| tone::PairField::new(&tiles[first], &tiles[second]))
                    .as_ref()
                else {
                    continue;
                };
                for offset in -15i64..=16 {
                    let px = i64::from(x) + if horizontal { offset } else { 0 };
                    let py = i64::from(y) + if horizontal { 0 } else { offset };
                    if px < 0 || py < 0 || px >= i64::from(width) || py >= i64::from(height) {
                        continue;
                    }
                    let Some((a, b)) = field.at(px as f64, py as f64) else {
                        continue;
                    };
                    if witness_window == Some((x / 256, y / 256)) {
                        let list = witnesses
                            .entry((first, second, x / 256, y / 256))
                            .or_default();
                        if list.len() < 128 {
                            list.push(serde_json::json!({"x":px,"y":py,"first_rgb":a,"second_rgb":b,
                                "first_parameters":model.at(first,px as f64,py as f64),"second_parameters":model.at(second,px as f64,py as f64)}));
                        }
                    }
                    let (a, b) = (
                        model.corrected(first, a, px as f64, py as f64),
                        model.corrected(second, b, px as f64, py as f64),
                    );
                    let entry = windows
                        .entry((first, second, x / 256, y / 256))
                        .or_insert(([0.0; 3], [0.0; 3], 0));
                    for c in 0..3 {
                        entry.0[c] += f64::from(a[c]);
                        entry.1[c] += f64::from(b[c])
                    }
                    entry.2 += 1;
                }
            }
        }
    }
    let mut maximum = 0.0f64;
    let mut failed = Vec::new();
    for ((a, b, x, y), (first, second, count)) in windows {
        let delta = tone::delta_e00_rgb(
            first.map(|v| (v / count as f64) as f32),
            second.map(|v| (v / count as f64) as f32),
        );
        maximum = maximum.max(delta);
        if delta > stack_report::TONE_BOUNDARY_DELTA_E_THRESHOLD {
            let (cx, cy) = ((x * 256 + 128) as f64, (y * 256 + 128) as f64);
            failed.push(serde_json::json!({"first":tiles[a].station_index,"second":tiles[b].station_index,"x":x*256,"y":y*256,"delta":delta,"samples":count,
                "first_parameters":model.at(a,cx,cy),"second_parameters":model.at(b,cx,cy),
                "witnesses":witnesses.remove(&(a,b,x,y)),
                "common_rgb":fields.get(&(a,b)).and_then(|f| f.as_ref()).and_then(|f| f.at(cx,cy))}))
        }
    }
    let result = serde_json::json!({"diagnostic_only":true,"native_roi_prediction":native_roi_prediction,"maximum":maximum,"failed":failed,"iterations":model.iterations,"maximum_parameter_change":model.maximum_parameter_change,"fitted_relations":model.fitted_relations.len(),"plan_step":plan.as_ref().map(|p|p.step)});
    fs::write(
        std::env::var_os("RAW_EDITOR_LOCAL_TONE_OUTPUT").unwrap(),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    println!("Local native tone prediction: {maximum:.6}");
    assert!(
        failed.is_empty(),
        "local native tone prediction failed: {result}"
    );
}

#[test]
#[ignore = "requires an actual saved native render and a boundary coordinate"]
fn real_saved_boundary_edge_from_env() {
    let image = image::open(std::env::var("RAW_EDITOR_BOUNDARY_IMAGE").unwrap())
        .unwrap()
        .to_rgb32f();
    let x = std::env::var("RAW_EDITOR_BOUNDARY_X")
        .unwrap()
        .parse()
        .unwrap();
    let y = std::env::var("RAW_EDITOR_BOUNDARY_Y")
        .unwrap()
        .parse()
        .unwrap();
    let coverage = std::env::var_os("RAW_EDITOR_BOUNDARY_COVERAGE")
        .map(|path| image::open(path).unwrap().to_luma8());
    let normal = if std::env::var_os("RAW_EDITOR_BOUNDARY_VERTICAL_NORMAL").is_some() {
        (0.0, 1.0)
    } else {
        (1.0, 0.0)
    };
    println!(
        "Actual paired boundary fits: {:?}",
        quality_gate::paired_boundary_edge_fits_with_coverage(
            &image,
            coverage.as_ref(),
            x,
            y,
            normal
        )
    );
    println!(
        "Actual paired boundary edge: {:?}",
        quality_gate::paired_boundary_edge_error_with_coverage(
            &image,
            coverage.as_ref(),
            x,
            y,
            normal
        )
    );
}

#[test]
#[ignore = "replays an actual saved seam grid using its native measurements and coverage constraints"]
fn real_saved_seam_grid_from_env() {
    use crate::panorama_utils::stack_pipeline::compositor;
    let d: serde_json::Value = serde_json::from_slice(
        &fs::read(std::env::var_os("RAW_EDITOR_SEAM_GRID").unwrap()).unwrap(),
    )
    .unwrap();
    let dir = PathBuf::from(std::env::var_os("RAW_EDITOR_TONE_FIELDS_DIR").unwrap());
    let owner = d["owner"].as_u64().unwrap() as u16;
    let mut tiles = Vec::new();
    for id in 1..=owner {
        tiles.push(load_saved_tone_tile(
            dir.join(format!("tone-owner-{id:03}.json")),
        ));
    }
    let width = d["columns"].as_u64().unwrap() as usize;
    let height = d["rows"].as_u64().unwrap() as usize;
    let floats = |key: &str| serde_json::from_value::<Vec<f64>>(d[key].clone()).unwrap();
    let base: Vec<u16> = serde_json::from_value(d["base_owners"].clone()).unwrap();
    let fixed: Vec<i8> = serde_json::from_value(d["fixed"].clone()).unwrap();
    let base_distance = floats("base_distance");
    let candidate_distance = floats("candidate_distance");
    let preference = floats("preference");
    let disagreement = floats("disagreement");
    let bounds = (
        d["left"].as_u64().unwrap() as u32,
        d["top"].as_u64().unwrap() as u32,
        d["right"].as_u64().unwrap() as u32,
        d["bottom"].as_u64().unwrap() as u32,
    );
    let step = d["step"].as_u64().unwrap() as u32;
    let existing = compositor::existing_owner_boundary_costs(
        &tiles,
        &base,
        &base_distance,
        width,
        height,
        bounds,
        step,
    );
    let labels = compositor::layered_seam_cut_with_existing_boundaries(
        width,
        height,
        &preference,
        &disagreement,
        &base_distance,
        &candidate_distance,
        &fixed,
        &existing,
    );
    for (label, pin) in labels.iter().zip(&fixed) {
        if *pin < 0 {
            assert_eq!(*label, 0);
        } else if *pin > 0 {
            assert_eq!(*label, 255);
        }
    }
    fs::write(
        std::env::var_os("RAW_EDITOR_SEAM_REPLAY_OUTPUT").unwrap(),
        serde_json::to_vec(
            &serde_json::json!({"labels":labels,"existing_costs":existing,"diagnostic_only":true}),
        )
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires saved measured native relation points and station poses"]
fn real_joint_geometry_from_saved_native_points() {
    let _scope = degradation::begin_run_scope();
    residual_warp::reset_run_records();
    let data: serde_json::Value = serde_json::from_slice(
        &fs::read(std::env::var("RAW_EDITOR_JOINT_PRODUCTION_INPUT").unwrap()).unwrap(),
    )
    .unwrap();
    let tiles = data["dimensions"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(station, d)| ImageInfo {
            id: usize::MAX - station,
            filename: format!("virtual://saved-native-station-{station}"),
            width: d[0].as_u64().unwrap() as u32,
            height: d[1].as_u64().unwrap() as u32,
            alignment_image: GrayImage::new(1, 1),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: Some(24.0),
            source_footprint: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        })
        .collect::<Vec<_>>();
    let initial = data["poses"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&tiles)
        .map(|(p, tile)| (tile.id, Matrix3::from_fn(|r, c| p[r][c].as_f64().unwrap())))
        .collect::<HashMap<_, _>>();
    let mut matches = HashMap::new();
    let mut report = stack_report::StationRelationsReport::default();
    for record in data["records"].as_array().unwrap() {
        if record["authoritative"].as_bool() == Some(false) {
            continue;
        }
        let (left, right) = (
            record["left"].as_u64().unwrap() as usize,
            record["right"].as_u64().unwrap() as usize,
        );
        let points = record["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    Point2::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()),
                    Point2::new(p[2].as_f64().unwrap(), p[3].as_f64().unwrap()),
                )
            })
            .collect::<Vec<_>>();
        let native = record["native_verified_points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    Point2::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()),
                    Point2::new(p[2].as_f64().unwrap(), p[3].as_f64().unwrap()),
                    p[4].as_f64().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let coefficients = record["homography"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        matches.insert(
            (left, right),
            MatchInfo {
                homography: Matrix3::from_column_slice(&coefficients),
                inliers: points.len(),
                sequence_bridge: false,
                coarse_bridge: false,
                points,
                candidate_points: Vec::new(),
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
                canonical_homography: None,
                native_verified_points: native,
            },
        );
        report
            .accepted
            .push(stack_report::AcceptedStationRelationRecord {
                left,
                right,
                evidence_kind: stack_report::StationRelationEvidenceKind::VirtualTile,
                inliers: matches[&(left, right)].inliers,
                spatial_support: record["diagnostic"]["hull_support"].as_f64().unwrap(),
                scale_ratio: record["scale"].as_f64().unwrap(),
                median_error_px: record["error"].as_f64().unwrap(),
                independent_support: 0,
            });
    }
    let mut closure = stack_report::ClosureReport {
        status: stack_report::ClosureStatus::Unreliable,
        ..stack_report::ClosureReport::default()
    };
    let solved =
        refine_virtual_tile_joint_geometry(&tiles, &matches, &initial, &mut closure, &mut report)
            .expect("production joint geometry refused the measured native evidence");
    let model = residual_warp::run_model_snapshot();
    let fields=model.regions.iter().map(|region|serde_json::json!({
        "station":region.right_station,"world":region.world,"columns":region.columns,"rows":region.rows,
        "values":region.nodes().iter().map(|node|node.displacement).collect::<Vec<_>>(),
    })).collect::<Vec<_>>();
    let poses = tiles
        .iter()
        .map(|tile| {
            let h = solved[&tile.id];
            (0..3)
                .map(|r| (0..3).map(|c| h[(r, c)]).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    fs::write(
        std::env::var("RAW_EDITOR_JOINT_PRODUCTION_OUTPUT").unwrap(),
        serde_json::to_vec(
            &serde_json::json!({"poses":poses,"fields":fields,"closure":closure,
            "station_relations":report}),
        )
        .unwrap(),
    )
    .unwrap();
}
