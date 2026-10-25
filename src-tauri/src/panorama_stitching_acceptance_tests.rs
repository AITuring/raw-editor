use super::*;
use image::{ImageFormat, Rgb, RgbImage};
use std::fs;
use std::path::PathBuf;

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
            let source = image::open(path).expect("fixture image must decode");
            let (width, height) = source.dimensions();
            let focal_length_35mm = fs::read(path)
                .ok()
                .and_then(|bytes| crate::exif_processing::focal_length_35mm_from_bytes(&bytes));
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
            let global = homographies[&image.id] * point;
            let top = top_homographies[&image.id] * point;
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
    for left in 0..count {
        for right in left + 1..count {
            if selected_pairs
                .as_ref()
                .is_some_and(|pairs| !pairs.contains(&(left, right)))
            {
                continue;
            }
            let Some((seed, feature_inliers)) = virtual_tile_prior_free_relation(
                pyramid.feature_factors[left],
                &tiles[left].features,
                pyramid.feature_factors[right],
                &tiles[right].features,
                tiles[left].dimensions(),
            ) else {
                continue;
            };
            let (relation, mut diagnostic) =
                virtual_tile_pyramid_station_match(left, right, &pyramid, &seed, "saved_tiles");
            let Some(mut relation) = relation.into_iter().next() else {
                println!(
                    "TILE_NO_FIT {left}->{right} stage={} inliers={} support={:.4}",
                    diagnostic.failure_stage, diagnostic.fitted_inliers, diagnostic.hull_support
                );
                continue;
            };
            if std::env::var_os("RAW_EDITOR_TILE_REPLAY_POLISH").is_some() {
                let Some(polished) = virtual_tile_polished_station_match(
                    &pyramid.levels[0][left],
                    &pyramid.coverages[0][left],
                    &pyramid.levels[0][right],
                    &pyramid.coverages[0][right],
                    1,
                    &relation.homography,
                    (tiles[left].dimensions(), tiles[right].dimensions()),
                    &mut diagnostic,
                )
                .into_iter()
                .next() else {
                    continue;
                };
                relation = polished;
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
                    Vec::new()
                };
            let authoritative = virtual_tile_relation_is_authoritative(
                left,
                right,
                &relation,
                &tiles,
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
        let solved = solve_virtual_tile_station_poses_with_report(
            &tiles,
            pyramid.solver_coverages(),
            &replay_matches,
            &initial,
            &topology,
            &mut closure,
            &mut relations,
        );
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

#[test]
#[ignore = "requires user-provided focus-stack paths and writes a large temporary output"]
fn real_focus_stack_fixture_from_env() {
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
    let result = stitch_images_with_options(
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
    let canonical = crate::image_stack::canonicalize_image_stack_result(result.image);
    let preview_only = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_PREVIEW_ONLY").is_some();
    if !preview_only {
        crate::image_stack::write_srgb_tiff(&canonical, &output_path)
            .expect("full-resolution color-managed TIFF result should be writable");
    }
    let preview_edge = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_PREVIEW_EDGE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value >= 256)
        .unwrap_or(1800);
    let preview = canonical.resize(
        preview_edge,
        preview_edge,
        image::imageops::FilterType::Lanczos3,
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
            let crop = canonical
                .crop_imm(x, y, crop_width.min(width - x), crop_height.min(height - y))
                .resize(
                    preview_edge,
                    preview_edge,
                    image::imageops::FilterType::Lanczos3,
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
        crate::image_stack::write_srgb_jpeg(&canonical, &app_jpeg_path)
            .expect("full-resolution application JPEG should be writable");
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
