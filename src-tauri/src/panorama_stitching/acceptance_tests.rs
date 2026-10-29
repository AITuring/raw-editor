use super::*;
use image::{ImageFormat, Rgb, RgbImage};
use std::fs;
use std::path::PathBuf;

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

fn ordered_panorama_fixture_paths() -> Vec<PathBuf> {
    let fixture_root = std::env::var("RAW_EDITOR_ORDERED_PANORAMA_DIR")
        .map(PathBuf::from)
        .expect("set RAW_EDITOR_ORDERED_PANORAMA_DIR to the source-image directory");
    let mut paths = fs::read_dir(&fixture_root)
        .expect("ordered panorama fixture directory must be readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            !is_auxiliary_stitch_file(&path.to_string_lossy())
                && !is_generated_stitch_output(&path.to_string_lossy())
                && path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        matches!(
                            extension.to_ascii_lowercase().as_str(),
                            "jpg" | "jpeg" | "png" | "nef"
                        )
                    })
        })
        .collect::<Vec<_>>();
    paths
        .sort_by(|left, right| natural_path_cmp(&left.to_string_lossy(), &right.to_string_lossy()));
    if std::env::var_os("RAW_EDITOR_ORDERED_PANORAMA_REVERSE_INPUT").is_some() {
        paths.reverse();
    }
    assert!(paths.len() >= 2, "fixture must contain at least two images");
    paths
}

#[test]
#[ignore = "requires an external ordered panorama fixture directory"]
fn real_ordered_panorama_pair_diagnostics() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    crate::sidecar_storage::initialize(
        PathBuf::from("/private/tmp/raw-editor-ordered-panorama-sidecars").as_path(),
    )
    .expect("test sidecar storage should initialize once");
    let alignment_name = std::env::var("RAW_EDITOR_ORDERED_PANORAMA_ALIGNMENT")
        .unwrap_or_else(|_| "auto".to_string());
    let alignment_mode = AlignmentMode::from_wire(&alignment_name.to_ascii_lowercase());
    let blend_mode = match std::env::var("RAW_EDITOR_ORDERED_PANORAMA_BLEND")
        .unwrap_or_else(|_| "panorama".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "focus" => BlendMode::FocusStack,
        _ => BlendMode::Panorama,
    };
    let paths = ordered_panorama_fixture_paths();

    let (max_dimension, max_features) = scalable_alignment_budget(paths.len());
    let brief_pairs = processing::generate_brief_pairs();
    let images = paths
        .par_iter()
        .enumerate()
        .map(|(id, path)| {
            // The acceptance fixture may contain the same RAW sources that
            // the production image-stack path accepts. `image::open`
            // only handles raster formats, so using it here made a valid
            // NEF fixture fail before alignment was even exercised. Keep
            // diagnostics on the production loader so this harness tests
            // the actual RAW decode and preprocessing path.
            let prepared =
                load_prepared_stack_source(&path.to_string_lossy(), &AppSettings::default(), None)
                    .expect("fixture image must decode through the production loader");
            let source = prepared.image;
            let (width, height) = source.dimensions();
            let focal_length_35mm = prepared.focal_length_35mm;
            let (new_width, new_height, scale_factor) =
                processing::calculate_downscale_dimensions_capped(width, height, max_dimension);
            let alignment_image = source
                .resize_exact(new_width, new_height, image::imageops::FilterType::Triangle)
                .to_luma8();
            let features = find_alignment_features(
                &alignment_image,
                &brief_pairs,
                max_features,
                true,
                focal_length_35mm,
            );
            let foreground_range = detect_foreground_range(&alignment_image);
            let top_features =
                find_top_alignment_features(&alignment_image, &brief_pairs, foreground_range);
            let foreground_mask = build_foreground_mask(&alignment_image, foreground_range);
            let horizontal_edge_rows =
                horizontal_edge_row_candidates(&alignment_image, foreground_range);
            let vertical_edge_columns =
                detect_vertical_edge_columns(&alignment_image, foreground_range);
            ImageInfo {
                id,
                filename: path.to_string_lossy().into_owned(),
                width,
                height,
                alignment_image,
                full_image: None,
                scale_factor,
                focal_length_35mm,
                source_footprint: None,
                overview_reference: false,
                features,
                top_features,
                foreground_range,
                foreground_mask,
                horizontal_edge_rows,
                vertical_edge_columns,
            }
        })
        .collect::<Vec<_>>();
    println!(
        "focus top feature counts: min={} max={} total={}",
        images
            .iter()
            .map(|image| image.top_features.len())
            .min()
            .unwrap_or(0),
        images
            .iter()
            .map(|image| image.top_features.len())
            .max()
            .unwrap_or(0),
        images
            .iter()
            .map(|image| image.top_features.len())
            .sum::<usize>(),
    );
    println!(
        "focus foreground ranges: {}",
        images
            .iter()
            .map(|image| {
                let name = Path::new(&image.filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy();
                match image.foreground_range {
                    Some((minimum, maximum)) => format!("{name}=({minimum:.3},{maximum:.3})"),
                    None => format!("{name}=none"),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "focus foreground corner energy: {}",
        images
            .iter()
            .filter_map(|image| {
                if image.top_features.is_empty() {
                    return None;
                }
                let width = image.alignment_image.width() as i32;
                let height = image.alignment_image.height() as i32;
                let mut values = image
                    .top_features
                    .iter()
                    .filter_map(|feature| {
                        let x = feature.keypoint.x as i32;
                        let y = feature.keypoint.y as i32;
                        (x >= FOCUS_FOREGROUND_PATCH_RADIUS
                            && y >= FOCUS_FOREGROUND_PATCH_RADIUS
                            && x + FOCUS_FOREGROUND_PATCH_RADIUS < width
                            && y + FOCUS_FOREGROUND_PATCH_RADIUS < height)
                            .then(|| {
                                gradient_patch_corner_energy(
                                    &LumaPlane::Gray(&image.alignment_image),
                                    x,
                                    y,
                                    FOCUS_FOREGROUND_PATCH_RADIUS,
                                )
                            })
                    })
                    .filter(|value| value.is_finite())
                    .collect::<Vec<_>>();
                values.sort_unstable_by(f64::total_cmp);
                let percentile = |fraction: f64| {
                    values
                        .get(((values.len().saturating_sub(1)) as f64 * fraction) as usize)
                        .copied()
                        .unwrap_or(0.0)
                };
                Some(format!(
                    "{} n={} p50={:.2} p75={:.2} p90={:.2} max={:.2}",
                    Path::new(&image.filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    values.len(),
                    percentile(0.50),
                    percentile(0.75),
                    percentile(0.90),
                    percentile(1.0),
                ))
            })
            .collect::<Vec<_>>()
            .join(" | ")
    );
    let candidate_pairs = pairs_to_match_for_images(&images, blend_mode);
    let matches = candidate_pairs
        .par_iter()
        .filter_map(|&(source, target)| {
            let (source_index, target_index, invert_for_storage) =
                if blend_mode == BlendMode::FocusStack {
                    canonical_match_direction(&images, source, target)
                } else {
                    (source, target, false)
                };
            let mut match_info = match_image_pair(
                &images[source_index],
                &images[target_index],
                Projection::Planar,
                blend_mode,
                alignment_mode,
                true,
                false,
                false,
            )?;
            if blend_mode == BlendMode::FocusStack {
                // Record the fit before it is rotated into the key
                // orientation; see `MatchInfo::canonical_homography`
                // (需求 14.6).
                match_info.canonical_homography = Some(match_info.homography);
            }
            if invert_for_storage {
                match_info.homography = match_info.homography.try_inverse()?;
                match_info.points = match_info
                    .points
                    .into_iter()
                    .map(|(source, target)| (target, source))
                    .collect();
                match_info.candidate_points = match_info
                    .candidate_points
                    .into_iter()
                    .map(|(source, target)| (target, source))
                    .collect();
                match_info.top_candidate_points = match_info
                    .top_candidate_points
                    .into_iter()
                    .map(|(source, target)| (target, source))
                    .collect();
                match_info.dense_focus_points = match_info
                    .dense_focus_points
                    .into_iter()
                    .map(|(source, target)| (target, source))
                    .collect();
                match_info.foreground_feature_points = match_info
                    .foreground_feature_points
                    .into_iter()
                    .map(|(source, target)| (target, source))
                    .collect();
            }
            Some(((source, target), match_info))
        })
        .collect::<HashMap<_, _>>();
    for (&(source, target), match_info) in &matches {
        let Some(source_range) = images[source].foreground_range else {
            continue;
        };
        let Some(target_range) = images[target].foreground_range else {
            continue;
        };
        let active_top_count = match_info
            .top_candidate_points
            .iter()
            .filter(|(source_point, target_point)| {
                let source_y = source_point.y / images[source].height as f64;
                let target_y = target_point.y / images[target].height as f64;
                (source_range.0..=source_range.1).contains(&source_y)
                    && (target_range.0..=target_range.1).contains(&target_y)
            })
            .count();
        let active_dense_count = match_info
            .dense_focus_points
            .iter()
            .filter(|(source_point, target_point)| {
                let source_y = source_point.y / images[source].height as f64;
                let target_y = target_point.y / images[target].height as f64;
                (source_range.0..=source_range.1).contains(&source_y)
                    && (target_range.0..=target_range.1).contains(&target_y)
            })
            .count();
        if active_top_count >= FOCUS_MODEL_MIN_INLIERS
            || active_dense_count >= FOCUS_MODEL_MIN_INLIERS
        {
            println!(
                "focus active foreground pair {}<->{}: top={} dense={} local={}",
                Path::new(&images[source].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                Path::new(&images[target].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                active_top_count,
                active_dense_count,
                focus_match_points_for_region(
                    &images[source],
                    &images[target],
                    match_info,
                    (0.04, 0.34),
                )
                .len(),
            );
        }
    }
    let (connected_order, homographies) = if blend_mode == BlendMode::FocusStack {
        build_focus_stack_stitching_order(&images, &matches)
    } else {
        build_stitching_order(&images, &matches)
    };
    if homographies.len() != images.len() {
        let missing = images
            .iter()
            .filter(|image| !homographies.contains_key(&image.id))
            .map(|image| {
                Path::new(&image.filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        println!(
            "ordered {} diagnostics: geometry disconnected; poses for {}/{} images are missing: {}",
            if blend_mode == BlendMode::FocusStack {
                "focus-stack"
            } else {
                "panorama"
            },
            missing.len(),
            images.len(),
            missing.join(", "),
        );
        return;
    }
    if blend_mode == BlendMode::FocusStack {
        let mut band_counts = [0usize; 6];
        let mut band_error_sums = [0.0f64; 6];
        let mut band_error_counts = [0usize; 6];
        for (&(source, target), match_info) in &matches {
            let Some(source_homography) = homographies.get(&images[source].id) else {
                continue;
            };
            let Some(target_homography) = homographies.get(&images[target].id) else {
                continue;
            };
            for &(source_point, target_point) in &match_info.points {
                let source_normalized_y = source_point.y / images[source].height as f64;
                let target_normalized_y = target_point.y / images[target].height as f64;
                let band = (((source_normalized_y + target_normalized_y) * 0.5) * 6.0)
                    .floor()
                    .clamp(0.0, 5.0) as usize;
                band_counts[band] += 1;
                let source_world =
                    source_homography * nalgebra::Point3::new(source_point.x, source_point.y, 1.0);
                let target_world =
                    target_homography * nalgebra::Point3::new(target_point.x, target_point.y, 1.0);
                if source_world.z.abs() >= 1e-8 && target_world.z.abs() >= 1e-8 {
                    let source_world = nalgebra::Point2::new(
                        source_world.x / source_world.z,
                        source_world.y / source_world.z,
                    );
                    let target_world = nalgebra::Point2::new(
                        target_world.x / target_world.z,
                        target_world.y / target_world.z,
                    );
                    band_error_sums[band] += (source_world - target_world).norm();
                    band_error_counts[band] += 1;
                }
            }
        }
        println!(
            "focus residual bands: counts={band_counts:?} mean_px={:?}",
            band_error_sums
                .iter()
                .zip(band_error_counts.iter())
                .map(|(sum, count)| {
                    if *count == 0 {
                        0.0
                    } else {
                        *sum / *count as f64
                    }
                })
                .collect::<Vec<_>>()
        );
        let mut top_candidate_count = 0usize;
        let mut top_local_count = 0usize;
        for (&(source, target), match_info) in &matches {
            let candidates = if !match_info.top_candidate_points.is_empty() {
                &match_info.top_candidate_points
            } else if match_info.candidate_points.is_empty() {
                &match_info.points
            } else {
                &match_info.candidate_points
            };
            top_candidate_count += candidates
                .iter()
                .filter(|(source_point, target_point)| {
                    let source_y = source_point.y / images[source].height as f64;
                    let target_y = target_point.y / images[target].height as f64;
                    (0.12..=0.28).contains(&source_y) && (0.12..=0.28).contains(&target_y)
                })
                .count();
            top_local_count += focus_match_points_for_region(
                &images[source],
                &images[target],
                match_info,
                (0.12, 0.28),
            )
            .len();
        }
        println!("focus top candidate/local points: {top_candidate_count}/{top_local_count}");
        let mut dense_offsets = [Vec::<(f64, f64)>::new(), Vec::new(), Vec::new()];
        for (&(source, target), match_info) in &matches {
            if source.abs_diff(target) > 4 {
                continue;
            }
            let source_image = &images[source];
            let target_image = &images[target];
            for (band_index, y_fraction) in [0.12, 0.20, 0.28].into_iter().enumerate() {
                for x_fraction in [0.1, 0.3, 0.5, 0.7, 0.9] {
                    let source_x = (source_image.width as f64 * x_fraction).round() as i32;
                    let source_y = (source_image.height as f64 * y_fraction).round() as i32;
                    let predicted = match_info.homography
                        * nalgebra::Point3::new(source_x as f64, source_y as f64, 1.0);
                    if predicted.z.abs() < 1e-8 {
                        continue;
                    }
                    let target_x = (predicted.x / predicted.z).round() as i32;
                    let target_y = (predicted.y / predicted.z).round() as i32;
                    if target_x < 12
                        || target_y < 12
                        || target_x + 12 >= target_image.width as i32
                        || target_y + 12 >= target_image.height as i32
                    {
                        continue;
                    }
                    let Some((best_x, best_y, subpixel_x, subpixel_y)) = refine_patch_position(
                        &LumaPlane::Gray(&source_image.alignment_image),
                        &LumaPlane::Gray(&target_image.alignment_image),
                        source_x,
                        source_y,
                        target_x,
                        target_y,
                        10,
                        20,
                    ) else {
                        continue;
                    };
                    dense_offsets[band_index].push((
                        best_x as f64 + subpixel_x - predicted.x / predicted.z,
                        best_y as f64 + subpixel_y - predicted.y / predicted.z,
                    ));
                }
            }
        }
        println!(
            "focus dense top offsets: {:?}",
            dense_offsets
                .iter()
                .map(|offsets| {
                    let (sum_x, sum_y) = offsets
                        .iter()
                        .fold((0.0, 0.0), |(sum_x, sum_y), (x, y)| (sum_x + x, sum_y + y));
                    if offsets.is_empty() {
                        (0, 0.0, 0.0)
                    } else {
                        (
                            offsets.len(),
                            sum_x / offsets.len() as f64,
                            sum_y / offsets.len() as f64,
                        )
                    }
                })
                .collect::<Vec<_>>()
        );
        let top_homographies = optimize_focus_stack_global_homographies_in_region(
            &images,
            &matches,
            &homographies,
            Some((0.12, 0.28)),
        );
        let mut maximum_top_delta = (0.0f64, String::new(), 0.0f64, 0.0f64);
        for &index in connected_order.iter().take(6) {
            let image = &images[index];
            let point =
                nalgebra::Point3::new(image.width as f64 * 0.5, image.height as f64 * 0.2, 1.0);
            let global = homographies[&image.id] * point;
            let top = top_homographies[&image.id] * point;
            println!(
                "focus top model {}: delta=({:.1},{:.1})",
                Path::new(&image.filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                top.x / top.z - global.x / global.z,
                top.y / top.z - global.y / global.z,
            );
        }
        for image in &images {
            let point =
                nalgebra::Point3::new(image.width as f64 * 0.5, image.height as f64 * 0.2, 1.0);
            let (Some(global_homography), Some(top_homography)) =
                (homographies.get(&image.id), top_homographies.get(&image.id))
            else {
                // A rejected/disconnected source has no global pose. Keep
                // the diagnostic report readable instead of panicking on
                // a map lookup before the explicit connectivity assertion
                // below explains the real failure.
                continue;
            };
            let global = global_homography * point;
            let top = top_homography * point;
            let delta_x = top.x / top.z - global.x / global.z;
            let delta_y = top.y / top.z - global.y / global.z;
            let magnitude = (delta_x * delta_x + delta_y * delta_y).sqrt();
            if magnitude > maximum_top_delta.0 {
                maximum_top_delta = (
                    magnitude,
                    Path::new(&image.filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                    delta_x,
                    delta_y,
                );
            }
        }
        println!("focus top model maximum delta: {maximum_top_delta:?}");
        for (&(source, target), match_info) in &matches {
            let top_count = match_info
                .points
                .iter()
                .filter(|(source_point, target_point)| {
                    source_point.y < images[source].height as f64 * 0.28
                        || target_point.y < images[target].height as f64 * 0.28
                })
                .count();
            if top_count >= 8 && (source.abs_diff(target) <= 4 || source == 0) {
                println!(
                    "focus top-points {}<->{}: {}/{}",
                    Path::new(&images[source].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    Path::new(&images[target].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    top_count,
                    match_info.points.len(),
                );
            }
        }
        for (position, &index) in connected_order.iter().enumerate() {
            let image = &images[index];
            let homography = homographies
                .get(&image.id)
                .expect("focus diagnostic image must have a transform");
            let center = homography
                * nalgebra::Point3::new(image.width as f64 * 0.5, image.height as f64 * 0.5, 1.0);
            let top_left = homography * nalgebra::Point3::new(0.0, 0.0, 1.0);
            let top_right = homography * nalgebra::Point3::new(image.width as f64, 0.0, 1.0);
            let top_left = nalgebra::Point2::new(top_left.x / top_left.z, top_left.y / top_left.z);
            let top_right =
                nalgebra::Point2::new(top_right.x / top_right.z, top_right.y / top_right.z);
            println!(
                "focus frame {position:02} {}: center=({:.1},{:.1}) top_scale={:.6}",
                Path::new(&image.filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                center.x / center.z,
                center.y / center.z,
                (top_right - top_left).norm() / image.width as f64,
            );
        }
    }
    let minimum_features = images
        .iter()
        .map(|image| image.features.len())
        .min()
        .unwrap_or(0);
    let maximum_features = images
        .iter()
        .map(|image| image.features.len())
        .max()
        .unwrap_or(0);
    let mut minimum_x = f64::INFINITY;
    let mut maximum_x = f64::NEG_INFINITY;
    let mut minimum_y = f64::INFINITY;
    let mut maximum_y = f64::NEG_INFINITY;
    for image in &images {
        let homography = homographies
            .get(&image.id)
            .expect("every connected fixture image should have a transform");
        for (x, y) in [
            (0.0, 0.0),
            (image.width as f64, 0.0),
            (image.width as f64, image.height as f64),
            (0.0, image.height as f64),
        ] {
            let mapped = homography * nalgebra::Point3::new(x, y, 1.0);
            assert!(mapped.z.abs() >= 1e-8, "fixture corner must remain finite");
            let mapped_x = mapped.x / mapped.z;
            let mapped_y = mapped.y / mapped.z;
            minimum_x = minimum_x.min(mapped_x);
            maximum_x = maximum_x.max(mapped_x);
            minimum_y = minimum_y.min(mapped_y);
            maximum_y = maximum_y.max(mapped_y);
        }
    }
    let output_width = (maximum_x + (-minimum_x).ceil()).ceil().max(1.0) as u64;
    let output_height = (maximum_y + (-minimum_y).ceil()).ceil().max(1.0) as u64;
    let output_pixels = output_width.saturating_mul(output_height);
    let rgb32f_gib = output_pixels.saturating_mul(12) as f64 / 1024_f64.powi(3);
    let safe_scale = memory_safe_panorama_render_scale(output_width as u32, output_height as u32);
    let safe_homographies = scaled_homographies(&homographies, safe_scale);
    let image_refs = connected_order
        .iter()
        .map(|&index| &images[index])
        .collect::<Vec<_>>();
    let (safe_width, safe_height) =
        stitching::output_canvas_dimensions(&image_refs, &safe_homographies, Projection::Planar);
    println!(
        "ordered {} diagnostics ({alignment_name}): {} images, {} candidate pairs, {} matched pairs, {} connected images, features {}..{}, canvas {}x{} ({} pixels, {:.2} GiB RGB32F), safe {}x{} ({:.1}%)",
        if blend_mode == BlendMode::FocusStack {
            "focus-stack"
        } else {
            "panorama"
        },
        images.len(),
        candidate_pairs.len(),
        matches.len(),
        connected_order.len(),
        minimum_features,
        maximum_features,
        output_width,
        output_height,
        output_pixels,
        rgb32f_gib,
        safe_width,
        safe_height,
        safe_scale * 100.0,
    );
    assert_eq!(connected_order.len(), images.len());
    assert_eq!(homographies.len(), images.len());
}

#[test]
#[ignore = "requires an external ordered panorama fixture directory and renders a large result"]
fn real_ordered_panorama_full_render() {
    let paths = ordered_panorama_fixture_paths()
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let alignment_name = std::env::var("RAW_EDITOR_ORDERED_PANORAMA_ALIGNMENT")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    let alignment_mode = AlignmentMode::from_wire(&alignment_name);
    let blend_mode = match std::env::var("RAW_EDITOR_ORDERED_PANORAMA_BLEND")
        .unwrap_or_else(|_| "panorama".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "focus" => BlendMode::FocusStack,
        _ => BlendMode::Panorama,
    };
    let app = tauri::test::mock_app();
    crate::sidecar_storage::initialize(
        PathBuf::from("/private/tmp/raw-editor-ordered-panorama-sidecars").as_path(),
    )
    .expect("test sidecar storage should initialize once");

    let started = Instant::now();
    let outcome = stitch_images_with_options(
        paths,
        app.handle().clone(),
        alignment_mode,
        blend_mode,
        "test-image-stack-progress",
    )
    .expect("the complete ordered panorama fixture should align and render");
    let (rendered_width, rendered_height) = outcome.image.dimensions();
    let full_canvas_width = outcome.full_canvas_width;
    let full_canvas_height = outcome.full_canvas_height;
    let render_scale = outcome.render_scale;
    let rendered_pixels = u64::from(rendered_width) * u64::from(rendered_height);
    assert!(rendered_width > 0 && rendered_height > 0);
    // Panorama renders are memory-bounded by the production path. Focus stacks
    // intentionally keep the source resolution so the exported TIFF does not
    // lose detail; their large canvas is therefore expected in this visual
    // acceptance fixture too.
    if blend_mode == BlendMode::Panorama {
        assert!(rendered_pixels <= MAX_IN_MEMORY_PANORAMA_PIXELS + 1_000_000);
    }
    assert!(outcome.full_canvas_width >= rendered_width);
    assert!(outcome.full_canvas_height >= rendered_height);

    let preview_scale = (4_000.0 / f64::from(rendered_width.max(rendered_height))).min(1.0);
    let preview_width = (f64::from(rendered_width) * preview_scale).round().max(1.0) as u32;
    let preview_height = (f64::from(rendered_height) * preview_scale)
        .round()
        .max(1.0) as u32;
    let preview =
        crate::image_processing::downscale_f32_image(&outcome.image, preview_width, preview_height);
    let preview_path = std::env::var_os("RAW_EDITOR_ORDERED_PANORAMA_PREVIEW")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/private/tmp/raw-editor-ordered-panorama-preview.jpg"));
    preview
        .save_with_format(&preview_path, ImageFormat::Jpeg)
        .expect("ordered panorama preview should be writable");
    let full_output_path =
        std::env::var_os("RAW_EDITOR_ORDERED_PANORAMA_OUTPUT").map(PathBuf::from);
    if let Some(output_path) = full_output_path.as_ref() {
        let canonical = crate::image_stack::canonicalize_image_stack_result(outcome.image);
        crate::image_stack::write_srgb_jpeg(&canonical, output_path)
            .expect("full-resolution ordered panorama JPEG should be writable");
    }
    println!(
        "ordered panorama full render ({alignment_name}): {}x{} from full canvas {}x{} at {:.1}% in {:.2?}\npreview: {}{}",
        rendered_width,
        rendered_height,
        full_canvas_width,
        full_canvas_height,
        render_scale * 100.0,
        started.elapsed(),
        preview_path.display(),
        full_output_path
            .as_ref()
            .map(|path| format!("\nfull: {}", path.display()))
            .unwrap_or_default(),
    );
}

#[test]
#[ignore = "requires saved real Virtual_Tile grayscale and coverage planes"]
fn real_virtual_tile_pair_diagnostics_from_env() {
    let _run_scope = degradation::begin_run_scope();
    degradation::reset_run_ledger();
    if let Ok(path) = std::env::var("RAW_EDITOR_TILE_REPLAY_MANIFEST") {
        let manifest: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let sources = manifest
            .as_array()
            .unwrap()
            .iter()
            .map(|source| {
                let hex = source["sha256_before"].as_str().unwrap();
                let mut digest = [0u8; 32];
                for (index, byte) in digest.iter_mut().enumerate() {
                    *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
                }
                (source["path"].as_str().unwrap().to_string(), digest)
            })
            .collect::<Vec<_>>();
        let key =
            virtual_tile::CacheKeyInputs::new(crate::image_stack::STACK_PIPELINE_VERSION, &sources);
        let seed = determinism::derive_run_seed_from_cache_key(&key.digest());
        determinism::set_run_random_seed(seed);
        println!("TILE_REPLAY run seed={seed}");
    }
    let directory = PathBuf::from(
        std::env::var_os("RAW_EDITOR_TILE_REPLAY_DIR").expect("saved tile directory"),
    );
    let count: usize = std::env::var("RAW_EDITOR_TILE_REPLAY_COUNT")
        .unwrap()
        .parse()
        .unwrap();
    let brief_pairs = processing::generate_brief_pairs();
    let mut tiles = Vec::new();
    let mut native = Vec::new();
    let mut masks = Vec::new();
    let factors = std::env::var("RAW_EDITOR_TILE_REPLAY_FACTORS")
        .map(|value| {
            value
                .split(',')
                .map(|factor| factor.parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|_| vec![1]);
    let mut pyramid = VirtualTileMatchPyramid::new(factors, count);
    for station in 0..count {
        let gray = image::open(directory.join(format!("station-{station:03}.gray.png")))
            .unwrap()
            .to_luma8();
        let mask = image::open(directory.join(format!("station-{station:03}.coverage.png")))
            .unwrap()
            .to_luma8();
        let coverage =
            stitching::CoverageMask::from_bytes(mask.width(), mask.height(), mask.into_raw())
                .unwrap();
        let mut tile = ImageInfo {
            id: station,
            filename: format!("station-{station:03}"),
            width: gray.width(),
            height: gray.height(),
            alignment_image: gray.clone(),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            source_footprint: None,
            features: Vec::new(),
            top_features: Vec::new(),
            overview_reference: false,
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        };
        install_virtual_tile_match_planes(
            &mut tile,
            &mut pyramid,
            station,
            &gray,
            &coverage,
            &brief_pairs,
        )
        .unwrap();
        tiles.push(tile);
        native.push(gray);
        masks.push(coverage);
    }
    let native_sigma = std::env::var("RAW_EDITOR_TILE_REPLAY_NATIVE_ROUND_TRIP_SIGMA")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);
    let native_structure = native
        .iter()
        .map(|image| {
            if native_sigma > 0.0 {
                imageproc::filter::gaussian_blur_f32(image, native_sigma)
            } else {
                image.clone()
            }
        })
        .collect::<Vec<_>>();
    let native_half = std::env::var("RAW_EDITOR_TILE_REPLAY_NATIVE_ROUND_TRIP_HALF")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(7);
    let mut records = Vec::new();
    let mut replay_matches = HashMap::new();
    let selected_pairs = std::env::var("RAW_EDITOR_TILE_REPLAY_PAIRS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|pair| {
                    let (left, right) = pair.split_once(':').unwrap();
                    (
                        left.parse::<usize>().unwrap(),
                        right.parse::<usize>().unwrap(),
                    )
                })
                .collect::<BTreeSet<_>>()
        });
    if let Ok(sigma) = std::env::var("RAW_EDITOR_TILE_REPLAY_POLISH_BLUR") {
        let sigma: f32 = sigma.parse().unwrap();
        for plane in pyramid.levels.last_mut().unwrap() {
            plane.alignment_image =
                imageproc::filter::gaussian_blur_f32(&plane.alignment_image, sigma);
        }
    }
    let quality_factors: Vec<u32> = std::env::var("RAW_EDITOR_TILE_REPLAY_QUALITY_FACTORS")
        .map(|value| {
            value
                .split(',')
                .map(|factor| factor.parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|_| vec![1, 2, 4, 12]);
    let seed_poses = std::env::var("RAW_EDITOR_TILE_REPLAY_SEED_POSES")
        .ok()
        .map(|path| {
            let poses: Vec<Vec<Vec<f64>>> =
                serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
            assert_eq!(poses.len(), count);
            poses
                .into_iter()
                .map(|p| {
                    Matrix3::new(
                        p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2], p[2][0], p[2][1],
                        p[2][2],
                    )
                })
                .collect::<Vec<_>>()
        });
    for left in 0..count {
        for right in left + 1..count {
            if selected_pairs
                .as_ref()
                .is_some_and(|pairs| !pairs.contains(&(left, right)))
            {
                continue;
            }
            let feature_seed = virtual_tile_prior_free_relation(
                pyramid.feature_factors[left],
                &tiles[left].features,
                pyramid.feature_factors[right],
                &tiles[right].features,
                tiles[left].dimensions(),
            );
            let Some((seed, feature_inliers)) = seed_poses
                .as_ref()
                .and_then(|poses| Some((poses[right].try_inverse()? * poses[left], 0)))
                .or(feature_seed)
            else {
                continue;
            };
            let (relation, mut diagnostic) =
                virtual_tile_pyramid_station_match(left, right, &pyramid, &seed, "saved_tiles");
            let Some(selection) = select_virtual_tile_relation_hypothesis(
                &tiles,
                left,
                right,
                &relation,
                pyramid.solver_coverages(),
            ) else {
                println!(
                    "TILE_NO_FIT {left}->{right} stage={} inliers={} support={:.4}",
                    diagnostic.failure_stage, diagnostic.fitted_inliers, diagnostic.hull_support
                );
                continue;
            };
            let mut relation = selection.relation;
            if std::env::var_os("RAW_EDITOR_TILE_REPLAY_POLISH").is_some() {
                let polished = virtual_tile_polished_station_match(
                    &pyramid.levels[0][left],
                    &pyramid.coverages[0][left],
                    &pyramid.levels[0][right],
                    &pyramid.coverages[0][right],
                    1,
                    &relation.homography,
                    (tiles[left].dimensions(), tiles[right].dimensions()),
                    &mut diagnostic,
                );
                let Some(selection) = select_virtual_tile_relation_hypothesis(
                    &tiles,
                    left,
                    right,
                    &polished,
                    pyramid.solver_coverages(),
                ) else {
                    continue;
                };
                relation = selection.relation;
            }
            let native_verified_points =
                if std::env::var_os("RAW_EDITOR_TILE_REPLAY_NATIVE_ROUND_TRIP").is_some() {
                    let backward = relation.homography.try_inverse().unwrap();
                    let step = (relation.points.len() / 400).max(1);
                    relation
                        .points
                        .par_iter()
                        .enumerate()
                        .filter(|(index, _)| index % step == 0)
                        .filter_map(|(_, &(center, _))| {
                            let check = residual_warp::verify_round_trip_match_with_template(
                                &native_structure[left],
                                &native_structure[right],
                                &relation.homography,
                                &backward,
                                center,
                                native_half,
                            )?;
                            Some([
                                center.x,
                                center.y,
                                check.forward.x,
                                check.forward.y,
                                check.round_trip_error_px,
                            ])
                        })
                        .collect::<Vec<_>>()
                } else {
                    relation
                        .native_verified_points
                        .iter()
                        .map(|&(a, b, error)| [a.x, a.y, b.x, b.y, error])
                        .collect::<Vec<_>>()
                };
            let authoritative = virtual_tile_provisional_authoritative_candidate(
                &tiles,
                left,
                right,
                std::slice::from_ref(&relation),
                pyramid.solver_coverages(),
            );
            if authoritative {
                replay_matches.insert((left, right), relation.clone());
            }
            let (canonical_left, canonical_right, inverted) =
                canonical_match_direction(&tiles, left, right);
            let canonical_pose = if inverted {
                relation.homography.try_inverse().unwrap()
            } else {
                relation.homography
            };
            let canonical_quality = focus_overlap_quality_measurements_on_coverage(
                &tiles[canonical_left],
                &tiles[canonical_right],
                &canonical_pose,
                Some((
                    &pyramid.quality_coverages[canonical_left],
                    &pyramid.quality_coverages[canonical_right],
                )),
            );
            println!(
                "TILE_GATE {left}->{right} authoritative={authoritative} inverted={inverted} quality={canonical_quality:?}"
            );
            for &factor in &quality_factors {
                let lp = VirtualTileMatchPlane::from_native(&native[left], &masks[left], factor)
                    .unwrap();
                let rp = VirtualTileMatchPlane::from_native(&native[right], &masks[right], factor)
                    .unwrap();
                let mut l = virtual_tile_plane_image_info(&tiles[left], &lp);
                let mut r = virtual_tile_plane_image_info(&tiles[right], &rp);
                for (t, p) in [(&mut l, &lp), (&mut r, &rp)] {
                    t.width = p.gray.width() * factor;
                    t.height = p.gray.height() * factor;
                    t.scale_factor = f64::from(factor);
                    t.alignment_image = imageproc::filter::gaussian_blur_f32(&p.gray, 1.0);
                }
                let q = focus_overlap_quality_measurements_on_coverage(
                    &l,
                    &r,
                    &relation.homography,
                    Some((&lp.coverage, &rp.coverage)),
                );
                println!(
                    "TILE_REPLAY {left}->{right} factor={factor} features={feature_inliers} fit={} error={:.3} quality={q:?}",
                    relation.points.len(),
                    median_symmetric_error(&relation.homography, &relation.points)
                );
                if let Some(q) = q {
                    let (photometric, strength) = virtual_tile_photometric_measurements(q);
                    records.push(serde_json::json!({"left":left,"right":right,"factor":factor,
                        "inliers":relation.points.len(),"error":median_symmetric_error(&relation.homography,&relation.points),
                        "scale":station_relation_overlap_scale(&relation.homography,&relation.points),
                        "authoritative":authoritative,
                        "orientation":q.median_edge_orientation_difference_degrees,"strength":strength,
                        "photometric":photometric,"raw_photometric":q.low_frequency_mean_relative_difference,
                        "exposure_gain":q.exposure_gain,"normalized_difference":q.exposure_normalized_difference,"intensity_ncc":q.intensity_ncc,
                        "edge_ncc":q.edge_ncc,"homography":relation.homography.as_slice(),
                        "points":relation.points.iter().map(|(a,b)| [a.x,a.y,b.x,b.y]).collect::<Vec<_>>(),
                        "native_verified_points": native_verified_points, "diagnostic":diagnostic}));
                }
            }
        }
    }
    if let Ok(path) = std::env::var("RAW_EDITOR_TILE_REPLAY_INITIAL_POSES") {
        let poses: Vec<Vec<Vec<f64>>> = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let initial = poses
            .iter()
            .enumerate()
            .map(|(index, p)| {
                (
                    index,
                    Matrix3::new(
                        p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2], p[2][0], p[2][1],
                        p[2][2],
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        let groups = (0..tiles.len())
            .map(|anchor| FocusCaptureGroup {
                anchor,
                members: vec![anchor],
                local_to_anchor: HashMap::from([(anchor, Matrix3::identity())]),
            })
            .collect::<Vec<_>>();
        let topology = focus_station_topology(&tiles, &groups, &initial);
        let mut closure = stack_report::ClosureReport::default();
        let mut relations = stack_report::StationRelationsReport::default();
        let mut solved = solve_virtual_tile_station_poses_with_report(
            &tiles,
            pyramid.solver_coverages(),
            &replay_matches,
            &initial,
            &topology,
            &mut closure,
            &mut relations,
        );
        if std::env::var_os("RAW_EDITOR_TILE_REPLAY_JOINT").is_some() {
            // Preserve the measured evidence before an intentional guard
            // assertion, so a refused model can be diagnosed without
            // repeating the native patch measurements.
            fs::write(
                directory.join("replayed-relations.json"),
                serde_json::to_vec_pretty(&records).unwrap(),
            )
            .unwrap();
            fs::write(
                directory.join("joint-production-input.json"),
                serde_json::to_vec(&serde_json::json!({
                    "poses":tiles.iter().map(|tile| {
                        let h=solved[&tile.id];(0..3).map(|r|(0..3).map(|c|h[(r,c)])
                            .collect::<Vec<_>>()).collect::<Vec<_>>()
                    }).collect::<Vec<_>>(),
                    "dimensions":tiles.iter().map(ImageInfo::dimensions).collect::<Vec<_>>(),
                    "records":records,
                }))
                .unwrap(),
            )
            .unwrap();
            solved = refine_virtual_tile_joint_geometry(
                &tiles,
                &replay_matches,
                &solved,
                &mut closure,
                &mut relations,
            )
            .expect("native joint model failed its guards");
            let model = residual_warp::run_model_snapshot();
            let fields = model.regions.iter().map(|region| serde_json::json!({
                "station":region.right_station,"world":region.world,"columns":region.columns,
                "rows":region.rows,"values":region.nodes().iter().map(|node|node.displacement).collect::<Vec<_>>(),
            })).collect::<Vec<_>>();
            fs::write(
                directory.join("joint-render-fields.json"),
                serde_json::to_vec(&fields).unwrap(),
            )
            .unwrap();
        }
        fs::write(directory.join("solver-replay.json"),serde_json::to_vec_pretty(&serde_json::json!({
            "poses":solved.iter().map(|(id,h)| (id.to_string(),h.as_slice())).collect::<BTreeMap<_,_>>(),
            "closure":closure,"station_relations":relations,
        })).unwrap()).unwrap();
    }
    fs::write(
        directory.join("replayed-relations.json"),
        serde_json::to_vec_pretty(&records).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires RAW_EDITOR_FOCUS_STACK_PATHS and an external diagnostic directory"]
fn real_focus_pair_diagnostics_from_env() {
    let paths = std::env::split_paths(
        &std::env::var_os("RAW_EDITOR_FOCUS_STACK_PATHS").expect("source paths"),
    )
    .map(|p| p.to_string_lossy().into_owned())
    .collect::<Vec<_>>();
    let output = PathBuf::from(
        std::env::var_os("RAW_EDITOR_FOCUS_PAIR_DIAGNOSTICS_DIR").expect("diagnostic directory"),
    );
    fs::create_dir_all(&output).unwrap();
    crate::sidecar_storage::initialize(&output.join("sidecars")).unwrap();
    let brief_pairs = processing::generate_brief_pairs();
    let settings = AppSettings::default();
    let raw_exposure_reference = stack_raw_exposure_reference(&paths);
    let mut images = Vec::new();
    for (id, filename) in paths.iter().enumerate() {
        let prepared =
            load_prepared_stack_source(filename, &settings, raw_exposure_reference).unwrap();
        let (width, height) = prepared.image.dimensions();
        let (w, h, scale_factor) =
            processing::calculate_downscale_dimensions_capped(width, height, 2400);
        let alignment_image = prepared
            .image
            .resize_exact(w, h, image::imageops::FilterType::Triangle)
            .to_luma8();
        let foreground_range = detect_foreground_range(&alignment_image);
        let features = find_alignment_features(
            &alignment_image,
            &brief_pairs,
            1600,
            true,
            prepared.focal_length_35mm,
        );
        alignment_image
            .save(output.join(format!(
                "{}.png",
                Path::new(filename).file_stem().unwrap().to_string_lossy()
            )))
            .unwrap();
        images.push(ImageInfo {
            id,
            filename: filename.clone(),
            width,
            height,
            scale_factor,
            focal_length_35mm: prepared.focal_length_35mm,
            source_footprint: prepared.source_footprint,
            features,
            top_features: find_top_alignment_features(
                &alignment_image,
                &brief_pairs,
                foreground_range,
            ),
            foreground_mask: build_foreground_mask(&alignment_image, foreground_range),
            horizontal_edge_rows: horizontal_edge_row_candidates(
                &alignment_image,
                foreground_range,
            ),
            vertical_edge_columns: detect_vertical_edge_columns(&alignment_image, foreground_range),
            alignment_image,
            foreground_range,
            overview_reference: false,
            full_image: None,
        });
    }
    for i in 0..images.len() {
        for j in i + 1..images.len() {
            let (a, b, _) = canonical_match_direction(&images, i, j);
            println!(
                "PROBE {} -> {}",
                Path::new(&images[a].filename)
                    .file_name()
                    .unwrap()
                    .to_string_lossy(),
                Path::new(&images[b].filename)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
            );
            let result = match_image_pair(
                &images[a],
                &images[b],
                Projection::Planar,
                BlendMode::FocusStack,
                AlignmentMode::Auto,
                true,
                true,
                false,
            );
            println!(
                "PROBE_RESULT {} {} {:?}",
                i,
                j,
                result.as_ref().map(|m| (
                    m.inliers,
                    median_symmetric_error(&m.homography, &m.points),
                    m.homography
                ))
            );
            if result.is_none() {
                if let Some(m) = match_image_pair(
                    &images[a],
                    &images[b],
                    Projection::Planar,
                    BlendMode::Panorama,
                    AlignmentMode::Auto,
                    true,
                    true,
                    false,
                ) {
                    println!(
                        "PANORAMA_CANDIDATE {:?} quality={:?}",
                        m.homography,
                        focus_overlap_quality(&images[a], &images[b], &m.homography)
                    );
                    for factor in [1u32, 2, 4] {
                        let structural = |info: &ImageInfo| ImageInfo {
                            id: info.id,
                            filename: info.filename.clone(),
                            width: info.width,
                            height: info.height,
                            scale_factor: info.scale_factor * factor as f64,
                            alignment_image: imageproc::filter::gaussian_blur_f32(
                                &image::imageops::resize(
                                    &info.alignment_image,
                                    info.alignment_image.width() / factor,
                                    info.alignment_image.height() / factor,
                                    image::imageops::FilterType::Triangle,
                                ),
                                1.0,
                            ),
                            full_image: None,
                            focal_length_35mm: info.focal_length_35mm,
                            source_footprint: info.source_footprint,
                            features: Vec::new(),
                            top_features: Vec::new(),
                            overview_reference: false,
                            foreground_range: None,
                            foreground_mask: None,
                            horizontal_edge_rows: Vec::new(),
                            vertical_edge_columns: Vec::new(),
                        };
                        println!(
                            "STRUCTURAL factor={} quality={:?}",
                            factor,
                            focus_overlap_quality(
                                &structural(&images[a]),
                                &structural(&images[b]),
                                &m.homography
                            )
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "temporary fixture diagnostics"]
fn temporary_pair_probe() {
    let root = std::env::var("RAW_EDITOR_ORDERED_PANORAMA_DIR").expect("fixture dir");
    crate::sidecar_storage::initialize(
        PathBuf::from("/private/tmp/raw-editor-temporary-pair-probe-sidecars").as_path(),
    )
    .expect("temporary probe sidecar storage should initialize");
    let mut paths = fs::read_dir(&root)
        .expect("fixture dir")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("jpg"))
        })
        .collect::<Vec<_>>();
    paths.sort_by(|a, b| natural_path_cmp(&a.to_string_lossy(), &b.to_string_lossy()));
    let (max_dimension, max_features) = scalable_alignment_budget(paths.len());
    let pairs = processing::generate_brief_pairs();
    let images = paths
        .iter()
        .enumerate()
        .map(|(id, path)| {
            let bytes = fs::read(path).unwrap();
            let focal = crate::exif_processing::focal_length_35mm_from_bytes(&bytes);
            let source = crate::image_loader::load_base_image_from_bytes(
                &bytes,
                &path.to_string_lossy(),
                false,
                &AppSettings::default(),
                None,
            )
            .unwrap();
            let (w, h, sf) = processing::calculate_downscale_dimensions_capped(
                source.width(),
                source.height(),
                max_dimension,
            );
            let alignment = source
                .resize_exact(w, h, image::imageops::FilterType::Triangle)
                .to_luma8();
            let features = find_alignment_features(&alignment, &pairs, max_features, true, focal);
            ImageInfo {
                id,
                filename: path.to_string_lossy().into_owned(),
                width: source.width(),
                height: source.height(),
                alignment_image: alignment,
                full_image: Some(source.to_rgb32f()),
                scale_factor: sf,
                focal_length_35mm: focal,
                source_footprint: None,
                overview_reference: false,
                features,
                top_features: Vec::new(),
                foreground_range: None,
                foreground_mask: None,
                horizontal_edge_rows: Vec::new(),
                vertical_edge_columns: Vec::new(),
            }
        })
        .collect::<Vec<_>>();
    for i in 0..images.len() {
        for j in i + 1..images.len() {
            let a = &images[i];
            let b = &images[j];
            let raw = processing::match_features_with_ratio(
                &a.features,
                &b.features,
                SCALABLE_MATCH_RATIO_THRESHOLD,
            );
            if raw.len() < 5 {
                continue;
            }
            let kp1 = a.features.iter().map(|f| f.keypoint).collect::<Vec<_>>();
            let kp2 = b.features.iter().map(|f| f.keypoint).collect::<Vec<_>>();
            let pts = raw
                .iter()
                .map(|m| {
                    (
                        Point2::new(
                            kp1[m.index1].x as f64 * a.scale_factor,
                            kp1[m.index1].y as f64 * a.scale_factor,
                        ),
                        Point2::new(
                            kp2[m.index2].x as f64 * b.scale_factor,
                            kp2[m.index2].y as f64 * b.scale_factor,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let r = processing::find_homography_ransac_points_stable(
                &pts,
                FULL_RES_RANSAC_INLIER_THRESHOLD,
            );
            let (n, spread) = r
                .as_ref()
                .map(|(_, idx)| {
                    let p = idx.iter().map(|&k| pts[k]).collect::<Vec<_>>();
                    (
                        idx.len(),
                        match_has_spatial_support(&p, a.dimensions(), b.dimensions()),
                    )
                })
                .unwrap_or((0, false));
            if n >= 5 {
                let accepted = match_image_pair(
                    a,
                    b,
                    Projection::Planar,
                    BlendMode::Panorama,
                    AlignmentMode::Auto,
                    true,
                    false,
                    false,
                )
                .is_some();
                println!(
                    "PAIR {} {} focal={:?}/{:?} raw={} inliers={} spatial={} accepted={}",
                    Path::new(&a.filename)
                        .file_name()
                        .unwrap()
                        .to_string_lossy(),
                    Path::new(&b.filename)
                        .file_name()
                        .unwrap()
                        .to_string_lossy(),
                    a.focal_length_35mm,
                    b.focal_length_35mm,
                    raw.len(),
                    n,
                    spread,
                    accepted
                );
            }
        }
    }
}

fn synthetic_texture_pixel(x: u32, y: u32) -> Rgb<u8> {
    let mut hash = x.wrapping_mul(0x9E37_79B9) ^ y.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x7FEB_352D);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x846C_A68B);
    hash ^= hash >> 16;
    let value = hash as u8;
    Rgb([
        value,
        value.rotate_left(3) ^ ((x.wrapping_add(y) * 13) as u8),
        value.rotate_right(2) ^ ((x.wrapping_mul(3).wrapping_add(y * 5)) as u8),
    ])
}

/// 需求 1.9 excludes an undecodable Source_RAW and continues; 需求 1.8 then
/// rejects the run when fewer than two decodable sources are left.  The
/// panorama path shares this entry point and keeps its original behaviour:
/// it has no Capture_Station layer to exclude a source from, so it still
/// aborts on the first preparation error.
#[test]
// Like every other test that drives the whole entry point, this one needs the
// process-global sidecar storage, whose `OnceCell` cannot be shared with
// `sidecar_storage`'s own tests inside one test binary.
#[ignore = "initializes the process-global sidecar storage; run explicitly"]
fn undecodable_source_is_excluded_before_the_input_count_is_rejected() {
    let fixture_dir = tempfile::tempdir().expect("temporary stack fixture should be writable");
    let readable = fixture_dir.path().join("tile-000.png");
    RgbImage::from_fn(96, 96, synthetic_texture_pixel)
        .save_with_format(&readable, ImageFormat::Png)
        .expect("synthetic stack tile should be writable");
    let broken = fixture_dir.path().join("tile-001.png");
    std::fs::write(&broken, b"not an image").expect("broken fixture should be writable");
    let paths = vec![
        readable.to_string_lossy().into_owned(),
        broken.to_string_lossy().into_owned(),
    ];

    // Process global, set once by whichever test gets there first; any
    // initialized root is usable here because this run never reaches a
    // sidecar write.
    let _ = crate::sidecar_storage::initialize(
        &std::env::temp_dir().join("raw-editor-stack-decode-test-sidecars"),
    );
    let app = tauri::test::mock_app();
    let focus_error = stitch_images_with_options(
        paths.clone(),
        app.handle().clone(),
        AlignmentMode::Position,
        BlendMode::FocusStack,
        "test-image-stack-progress",
    )
    .err()
    .expect("one decodable source is below the 需求 1.8 minimum of two");
    assert!(
        focus_error.contains("Only 1 of 2"),
        "需求 1.9 must exclude the undecodable source and 需求 1.8 must then reject on the \
         surviving count, got: {focus_error}"
    );

    let panorama_error = stitch_images_with_options(
        paths,
        app.handle().clone(),
        AlignmentMode::Position,
        BlendMode::Panorama,
        "test-image-stack-progress",
    )
    .err()
    .expect("the panorama path still aborts on the first preparation error");
    assert!(
        !panorama_error.contains("Only 1 of 2"),
        "the panorama path must keep its original abort, got: {panorama_error}"
    );
}

#[test]
#[ignore = "manual deterministic 200-image scalable stitching smoke test"]
fn synthetic_two_hundred_image_scalable_path() {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;
    const HORIZONTAL_STEP: u32 = 2;

    let fixture_dir = tempfile::tempdir().expect("temporary stack fixture should be writable");
    let mut paths = Vec::with_capacity(MAX_STITCH_SOURCE_IMAGES);
    for index in 0..MAX_STITCH_SOURCE_IMAGES {
        let horizontal_offset = index as u32 * HORIZONTAL_STEP;
        let image = RgbImage::from_fn(WIDTH, HEIGHT, |x, y| {
            synthetic_texture_pixel(x + horizontal_offset, y)
        });
        let path = fixture_dir.path().join(format!("tile-{index:03}.png"));
        image
            .save_with_format(&path, ImageFormat::Png)
            .expect("synthetic stack tile should be writable");
        paths.push(path.to_string_lossy().into_owned());
    }

    crate::sidecar_storage::initialize(&fixture_dir.path().join("sidecars"))
        .expect("test sidecar storage should initialize once");
    let app = tauri::test::mock_app();
    let started = Instant::now();
    let result = stitch_images_with_options(
        paths,
        app.handle().clone(),
        AlignmentMode::Position,
        BlendMode::Panorama,
        "test-image-stack-progress",
    )
    .expect("the 200 overlapping tiles should stitch through the scalable path");
    let image = result.image;

    let expected_width = WIDTH + (MAX_STITCH_SOURCE_IMAGES as u32 - 1) * HORIZONTAL_STEP;
    assert!(image.width().abs_diff(expected_width) <= 2);
    assert!(image.height().abs_diff(HEIGHT) <= 2);
    println!(
        "synthetic 200-image stack: {}x{} in {:.2?}",
        image.width(),
        image.height(),
        started.elapsed()
    );
}

#[test]
#[ignore = "requires the real three-image fixture and writes a large temporary output"]
fn real_three_image_panorama_fixture() {
    let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../src/assets/test")
        .canonicalize()
        .expect("test fixture directory must exist");
    let paths = ["DSC01721_1.tif", "DSC01700.tif", "DSC01728.tif"]
        .iter()
        .map(|name| fixture_root.join(name).to_string_lossy().into_owned())
        .collect();
    let alignment_name = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_ALIGNMENT")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    let alignment_mode = AlignmentMode::from_wire(&alignment_name);
    let app = tauri::test::mock_app();
    crate::sidecar_storage::initialize(
        PathBuf::from("/private/tmp/raw-editor-stack-sidecars").as_path(),
    )
    .expect("test sidecar storage should initialize once");
    let result = stitch_images_with_options(
        paths,
        app.handle().clone(),
        alignment_mode,
        BlendMode::Panorama,
        "test-image-stack-progress",
    )
    .expect("the three real overlapping images should stitch");
    let image = result.image;

    assert!(image.width() > 0 && image.height() > 0);
    let output_path = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "/private/tmp/raw-editor-three-image-panorama-{alignment_name}.tiff"
            ))
        });
    image
        .save_with_format(&output_path, ImageFormat::Tiff)
        .expect("full-resolution TIFF result should be writable");

    let preview = crate::image_processing::downscale_f32_image(&image, 1800, 1800);
    let preview_path = output_path.with_extension("jpg");
    preview
        .save_with_format(&preview_path, ImageFormat::Jpeg)
        .expect("panorama preview should be writable");
    println!(
        "three-image panorama result: {}x{}\nfull: {}\npreview: {}",
        image.width(),
        image.height(),
        output_path.display(),
        preview_path.display()
    );
}

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

pub(crate) fn capture_native_gradient_roi(
    world: (f64, f64),
    path: &str,
    value: f64,
    local_scale: f64,
    output: &image::Rgb32FImage,
    reference: &image::Rgb32FImage,
) {
    let Some(directory) =
        std::env::var_os("RAW_EDITOR_NATIVE_GRADIENT_DIAGNOSTICS").map(PathBuf::from)
    else {
        return;
    };
    let witness = std::env::var("RAW_EDITOR_NATIVE_GRADIENT_ROIS")
        .ok()
        .is_some_and(|text| {
            text.split(';').any(|pair| {
                let values = pair
                    .split(',')
                    .filter_map(|v| v.parse::<f64>().ok())
                    .collect::<Vec<_>>();
                values.len() == 2 && values[0] == world.0 && values[1] == world.1
            })
        });
    if !witness && value >= quality_gate::QUALITY_GRADIENT_RATIO_MIN {
        return;
    }
    fs::create_dir_all(&directory).unwrap();
    let prefix = directory.join(format!("gradient-{}-{}", world.0, world.1));
    for (name, image) in [("output", output), ("reference", reference)] {
        DynamicImage::ImageRgb32F(image.clone())
            .save(prefix.with_extension(format!("{name}.float.tiff")))
            .unwrap();
        DynamicImage::ImageRgb32F(image.clone())
            .to_rgb16()
            .save(prefix.with_extension(format!("{name}.png")))
            .unwrap();
    }
    fs::write(prefix.with_extension("json"),serde_json::to_vec_pretty(&serde_json::json!({
        "diagnostic_only":true,"world":world,"owner_path":path,"measured":value,"local_scale":local_scale,
        "threshold":quality_gate::QUALITY_GRADIENT_RATIO_MIN,
    })).unwrap()).unwrap();
}

pub(super) fn capture_failed_native_boundaries(
    input: &quality_gate_runner::QualityGateInput<'_>,
    report: &stack_report::QualityGateReport,
) {
    let Some(directory) =
        std::env::var_os("RAW_EDITOR_FAILED_NATIVE_BOUNDARIES").map(PathBuf::from)
    else {
        return;
    };
    fs::create_dir_all(&directory).unwrap();
    for criterion in &report.criteria {
        if criterion.name != "boundary_stroke_alignment" {
            continue;
        }
        for (index, failed) in criterion.failed.iter().enumerate() {
            let x = (failed.world.x - input.world_origin.0).round() as u32;
            let y = (failed.world.y - input.world_origin.1).round() as u32;
            let left = x.saturating_sub(48);
            let top = y.saturating_sub(48);
            let width = 96.min(input.output.width().saturating_sub(left));
            let height = 96.min(input.output.height().saturating_sub(top));
            let crop = image::imageops::crop_imm(input.output, left, top, width, height).to_image();
            let prefix = directory.join(format!("boundary-{index:03}"));
            DynamicImage::ImageRgb32F(crop.clone())
                .save(prefix.with_extension("float.tiff"))
                .unwrap();
            DynamicImage::ImageRgb32F(crop)
                .to_rgb16()
                .save(prefix.with_extension("visible.png"))
                .unwrap();
            image::imageops::crop_imm(input.coverage, left, top, width, height)
                .to_image()
                .save(prefix.with_extension("coverage.png"))
                .unwrap();
            let owners = (top..top + height)
                .flat_map(|row| {
                    (left..left + width).map(move |column| {
                        input.ownership
                            [row as usize * input.output.width() as usize + column as usize]
                    })
                })
                .collect::<Vec<_>>();
            image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(width, height, owners)
                .unwrap()
                .save(prefix.with_extension("owners.png"))
                .unwrap();
            let legend = input
                .sources
                .iter()
                .map(|source| {
                    serde_json::json!({
                        "owner":source.owner,"path":source.path,"station":source.geometry.station_id
                    })
                })
                .collect::<Vec<_>>();
            fs::write(prefix.with_extension("json"),serde_json::to_vec_pretty(&serde_json::json!({
                "diagnostic_only":true,"reported":failed,"world_origin":input.world_origin,
                "canvas":[x,y],"crop":[left,top,width,height],"crop_point":[x-left,y-top],"legend":legend,
            })).unwrap()).unwrap();
        }
    }
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
        let metrics = super::acceptance_metrics::write_acceptance_metrics(
            &super::acceptance_metrics::AcceptanceMetricsRequest {
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
