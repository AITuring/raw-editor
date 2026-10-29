//! Ordered panorama fixture diagnostics: pairwise matching and the full render of an external
//! ordered panorama directory.

use super::*;

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
