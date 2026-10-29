//! Pair diagnostics on real inputs: saved Virtual_Tile planes, focus pairs from
//! RAW_EDITOR_FOCUS_STACK_PATHS and the temporary pair probe.

use super::*;

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
