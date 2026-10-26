use super::*;

pub(super) fn focal_lengths_span_multiple_lenses<I>(focal_lengths: I) -> bool
where
    I: IntoIterator<Item = f64>,
{
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut count = 0usize;
    for focal_length in focal_lengths {
        if !focal_length.is_finite() || focal_length <= 0.0 {
            continue;
        }
        minimum = minimum.min(focal_length);
        maximum = maximum.max(focal_length);
        count += 1;
    }
    count >= 2
        && minimum.is_finite()
        && maximum.is_finite()
        && maximum / minimum >= MIXED_FOCAL_LENGTH_RATIO
}

pub(super) fn selection_has_mixed_focal_lengths(image_paths: &[String]) -> bool {
    // Large stacks already use the scale-robust path. This lightweight EXIF
    // pass exists for small selections, where a count-only switch can miss a
    // camera-module change.
    focal_lengths_span_multiple_lenses(image_paths.iter().filter_map(|filename| {
        read_file_mapped(Path::new(filename))
            .ok()
            .and_then(|bytes| crate::exif_processing::focal_length_35mm_from_bytes(&bytes))
    }))
}

pub(super) fn image_pair_has_mixed_focal_lengths(source: &ImageInfo, target: &ImageInfo) -> bool {
    focal_lengths_span_multiple_lenses(
        [source.focal_length_35mm, target.focal_length_35mm]
            .into_iter()
            .flatten(),
    )
}

pub(super) fn images_have_mixed_focal_lengths(images: &[ImageInfo]) -> bool {
    focal_lengths_span_multiple_lenses(images.iter().filter_map(|image| image.focal_length_35mm))
}

pub(super) struct AreaContributions {
    offsets: Vec<usize>,
    indices: Vec<usize>,
    weights: Vec<f32>,
}

pub(super) fn area_contributions(source_length: u32, target_length: u32) -> AreaContributions {
    let mut offsets = Vec::with_capacity(target_length as usize + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let scale = f64::from(source_length) / f64::from(target_length.max(1));
    offsets.push(0);
    for target in 0..target_length.max(1) {
        let start = f64::from(target) * scale;
        let end = (f64::from(target) + 1.0) * scale;
        let first = start.floor() as usize;
        let last = (end.ceil() as usize)
            .saturating_sub(1)
            .min(source_length.saturating_sub(1) as usize);
        let normalization = (end - start).recip();
        for source in first..=last {
            let overlap = ((source + 1) as f64).min(end) - (source as f64).max(start);
            if overlap > 0.0 {
                indices.push(source);
                weights.push((overlap * normalization) as f32);
            }
        }
        offsets.push(indices.len());
    }
    AreaContributions {
        offsets,
        indices,
        weights,
    }
}

pub(super) fn resize_rgb8_area_to_rgb32f(
    source: &RgbImage,
    target_width: u32,
    target_height: u32,
) -> Rgb32FImage {
    let (source_width, source_height) = source.dimensions();
    debug_assert!(target_width > 0 && target_height > 0);
    debug_assert!(target_width <= source_width && target_height <= source_height);

    let horizontal_contributions = area_contributions(source_width, target_width);
    let vertical_contributions = area_contributions(source_height, target_height);
    let source_stride = source_width as usize * 3;
    let horizontal_stride = target_width as usize * 3;
    let mut horizontal = vec![0.0f32; horizontal_stride * source_height as usize];
    horizontal
        .par_chunks_mut(horizontal_stride)
        .zip(source.as_raw().par_chunks(source_stride))
        .for_each(|(output_row, source_row)| {
            for target_x in 0..target_width as usize {
                let contribution_start = horizontal_contributions.offsets[target_x];
                let contribution_end = horizontal_contributions.offsets[target_x + 1];
                let output_start = target_x * 3;
                for contribution in contribution_start..contribution_end {
                    let source_start = horizontal_contributions.indices[contribution] * 3;
                    let weight = horizontal_contributions.weights[contribution] / 255.0;
                    for channel in 0..3 {
                        output_row[output_start + channel] +=
                            f32::from(source_row[source_start + channel]) * weight;
                    }
                }
            }
        });

    let mut output = vec![0.0f32; horizontal_stride * target_height as usize];
    output
        .par_chunks_mut(horizontal_stride)
        .enumerate()
        .for_each(|(target_y, output_row)| {
            let contribution_start = vertical_contributions.offsets[target_y];
            let contribution_end = vertical_contributions.offsets[target_y + 1];
            for contribution in contribution_start..contribution_end {
                let source_y = vertical_contributions.indices[contribution];
                let weight = vertical_contributions.weights[contribution];
                let horizontal_row =
                    &horizontal[source_y * horizontal_stride..(source_y + 1) * horizontal_stride];
                for (output, source) in output_row.iter_mut().zip(horizontal_row) {
                    *output += source * weight;
                }
            }
        });

    Rgb32FImage::from_raw(target_width, target_height, output)
        .expect("area-resized RGB buffer dimensions must match")
}

pub(super) fn source_to_render_rgb32f(
    source: DynamicImage,
    target_width: u32,
    target_height: u32,
) -> Rgb32FImage {
    if source.dimensions() == (target_width, target_height) {
        return source.to_rgb32f();
    }
    if source.color() == ColorType::Rgb8
        && target_width <= source.width()
        && target_height <= source.height()
    {
        return resize_rgb8_area_to_rgb32f(&source.into_rgb8(), target_width, target_height);
    }
    source
        .resize_exact(
            target_width,
            target_height,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb32f()
}

pub(super) fn pairs_to_match(image_count: usize) -> Vec<(usize, usize)> {
    let neighbor_window = if image_count > SCALABLE_STACK_THRESHOLD {
        LARGE_STACK_NEIGHBOR_WINDOW
    } else {
        image_count.saturating_sub(1)
    };
    (0..image_count)
        .flat_map(|first| {
            let end = image_count.min(first.saturating_add(neighbor_window + 1));
            (first + 1..end).map(move |second| (first, second))
        })
        .collect()
}

pub(super) fn all_image_pairs(image_count: usize) -> Vec<(usize, usize)> {
    (0..image_count)
        .flat_map(|first| (first + 1..image_count).map(move |second| (first, second)))
        .collect()
}

pub(super) fn natural_path_cmp(left: &str, right: &str) -> Ordering {
    let left = Path::new(left)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(left)
        .as_bytes();
    let right = Path::new(right)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(right)
        .as_bytes();
    let mut left_index = 0usize;
    let mut right_index = 0usize;

    while left_index < left.len() && right_index < right.len() {
        if left[left_index].is_ascii_digit() && right[right_index].is_ascii_digit() {
            let left_start = left_index;
            let right_start = right_index;
            while left_index < left.len() && left[left_index].is_ascii_digit() {
                left_index += 1;
            }
            while right_index < right.len() && right[right_index].is_ascii_digit() {
                right_index += 1;
            }
            let left_digits = &left[left_start..left_index];
            let right_digits = &right[right_start..right_index];
            let left_trimmed = left_digits
                .iter()
                .position(|digit| *digit != b'0')
                .map_or(&left_digits[left_digits.len()..], |index| {
                    &left_digits[index..]
                });
            let right_trimmed = right_digits
                .iter()
                .position(|digit| *digit != b'0')
                .map_or(&right_digits[right_digits.len()..], |index| {
                    &right_digits[index..]
                });
            let number_order = left_trimmed
                .len()
                .cmp(&right_trimmed.len())
                .then_with(|| left_trimmed.cmp(right_trimmed))
                .then_with(|| left_digits.len().cmp(&right_digits.len()));
            if number_order != Ordering::Equal {
                return number_order;
            }
        } else {
            let byte_order = left[left_index]
                .to_ascii_lowercase()
                .cmp(&right[right_index].to_ascii_lowercase());
            if byte_order != Ordering::Equal {
                return byte_order;
            }
            left_index += 1;
            right_index += 1;
        }
    }

    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

pub(super) fn add_neighbor_pairs(
    order: &[usize],
    neighbor_window: usize,
    unique_pairs: &mut HashSet<(usize, usize)>,
) {
    for (position, &first) in order.iter().enumerate() {
        let end = order
            .len()
            .min(position.saturating_add(neighbor_window + 1));
        for &second in &order[position + 1..end] {
            unique_pairs.insert((first.min(second), first.max(second)));
        }
    }
}

pub(super) fn pairs_to_match_for_images(
    images: &[ImageInfo],
    blend_mode: BlendMode,
) -> Vec<(usize, usize)> {
    if images.len() <= SCALABLE_STACK_THRESHOLD {
        return pairs_to_match(images.len());
    }
    // The 65–96 image range is common for high-resolution artwork scans and is
    // still small enough for an exhaustive bounded-resolution pass.  A lens
    // switch can move the only useful bridge dozens of filenames away, so a
    // neighbour-only graph is too brittle here.  Larger selections retain the
    // bounded visual-retrieval path below.
    if (SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES..=SCALE_ROBUST_EXHAUSTIVE_MAX_SOURCES)
        .contains(&images.len())
    {
        return all_image_pairs(images.len());
    }
    if blend_mode == BlendMode::FocusStack
        && images.len() <= FOCUS_AUTO_ORDER_EXHAUSTIVE_MAX_SOURCES
    {
        return all_image_pairs(images.len());
    }

    let input_order = (0..images.len()).collect::<Vec<_>>();
    let mut filename_order = input_order.clone();
    filename_order.sort_by(|&left, &right| {
        natural_path_cmp(&images[left].filename, &images[right].filename)
            .then_with(|| left.cmp(&right))
    });

    let mut unique_pairs = HashSet::new();
    add_neighbor_pairs(&input_order, LARGE_STACK_NEIGHBOR_WINDOW, &mut unique_pairs);
    add_neighbor_pairs(
        &filename_order,
        LARGE_STACK_NEIGHBOR_WINDOW,
        &mut unique_pairs,
    );
    let mut pairs = unique_pairs.into_iter().collect::<Vec<_>>();
    pairs.sort_unstable();
    augment_large_stack_candidate_pairs(images, &mut pairs);
    pairs
}

pub(super) fn retrieval_feature_subset(features: &[Feature], limit: usize) -> Vec<Feature> {
    if features.len() <= limit {
        return features.to_vec();
    }
    // Feature vectors are ordered by detector strength.  Evenly sampling the
    // vector retains strong corners throughout the image instead of taking only
    // the first (often high-contrast) region.
    (0..limit)
        .map(|slot| {
            let index = slot
                .saturating_mul(features.len())
                .checked_div(limit)
                .unwrap_or(0)
                .min(features.len().saturating_sub(1));
            features[index].clone()
        })
        .collect()
}

/// Add a bounded set of visually likely pairs to the normal input/filename
/// neighbour graph.  File-picker order is often random, and a focal-length
/// switch can put the first useful bridge well outside the neighbour window.
/// The retrieval pass only counts compact descriptor matches; full-resolution
/// geometric verification still decides whether a pair is accepted.
pub(super) fn augment_large_stack_candidate_pairs(
    images: &[ImageInfo],
    pairs: &mut Vec<(usize, usize)>,
) {
    if images.len() <= SCALABLE_STACK_THRESHOLD || images.len() < 2 {
        return;
    }
    let compact_features = images
        .iter()
        .map(|image| retrieval_feature_subset(&image.features, LARGE_STACK_RETRIEVAL_FEATURES))
        .collect::<Vec<_>>();
    if compact_features.iter().all(Vec::is_empty) {
        return;
    }

    let existing = pairs.iter().copied().collect::<HashSet<_>>();
    let scores = all_image_pairs(images.len())
        .into_par_iter()
        .filter_map(|(left, right)| {
            let score = processing::count_mutual_descriptor_matches_with_ratio(
                &compact_features[left],
                &compact_features[right],
                SCALABLE_MATCH_RATIO_THRESHOLD,
            );
            (score >= LARGE_STACK_RETRIEVAL_MIN_MATCHES).then_some((score, left, right))
        })
        .collect::<Vec<_>>();
    if scores.is_empty() {
        return;
    }

    let mut by_image = vec![Vec::<(usize, usize)>::new(); images.len()];
    for (score, left, right) in scores {
        by_image[left].push((score, right));
        by_image[right].push((score, left));
    }
    let mut additions = HashSet::new();
    for (image_index, neighbours) in by_image.iter_mut().enumerate() {
        neighbours.sort_by(|(left_score, left_index), (right_score, right_index)| {
            right_score
                .cmp(left_score)
                .then_with(|| {
                    natural_path_cmp(
                        &images[*left_index].filename,
                        &images[*right_index].filename,
                    )
                })
                .then_with(|| left_index.cmp(right_index))
        });
        for &(_, neighbour) in neighbours.iter().take(LARGE_STACK_RETRIEVAL_NEIGHBORS) {
            let pair = (image_index.min(neighbour), image_index.max(neighbour));
            if !existing.contains(&pair) {
                additions.insert(pair);
            }
        }
    }
    pairs.extend(additions);
    pairs.sort_unstable();
    pairs.dedup();
}

pub(super) fn find_alignment_features(
    alignment_image: &GrayImage,
    brief_pairs: &[(nalgebra::Point2<i32>, nalgebra::Point2<i32>)],
    max_features: usize,
    scalable_stack: bool,
    focal_length_35mm: Option<f64>,
) -> Vec<Feature> {
    // A fixed-size BRIEF patch is not scale invariant.  Large phone stacks can
    // switch lenses midway through a capture (for example 35mm -> 85mm), so
    // keep descriptors from a small image pyramid on the bounded-memory path.
    // The native level remains dominant; the extra levels only provide a bridge
    // when the same detail is rendered at a different magnification.
    let mut features = if scalable_stack {
        let mut scales = vec![0.38, 0.5, std::f32::consts::FRAC_1_SQRT_2];
        // A metadata-guided level places the two camera modules on a common
        // support scale.  The generic levels remain in place for cropped or
        // non-EXIF sources.
        if let Some(focal) = focal_length_35mm.filter(|value| value.is_finite() && *value > 0.0) {
            let metadata_scale = (50.0 / focal).clamp(0.28, 2.5) as f32;
            if scales
                .iter()
                .all(|existing| (*existing - metadata_scale).abs() >= 0.04)
            {
                scales.push(metadata_scale);
            }
        }
        processing::find_features_multiscale(alignment_image, brief_pairs, max_features, &scales)
    } else {
        processing::find_features(alignment_image, brief_pairs)
    };
    if scalable_stack && features.len() < SCALABLE_LOW_TEXTURE_FEATURE_TARGET {
        let normalized = processing::normalize_grayscale(alignment_image);
        let fallback = processing::find_features_tuned(
            &normalized,
            brief_pairs,
            SCALABLE_LOW_TEXTURE_FAST_THRESHOLD,
            SCALABLE_LOW_TEXTURE_NMS_RADIUS,
        );
        if fallback.len() > features.len() {
            // Keep the pyramid descriptors even when the normalized native
            // pass finds more corners. Replacing the vector here used to erase
            // the only scale bridge on a low-texture frame—the exact case in
            // which the 35mm/85mm pair is hardest to register.
            let remaining = max_features.saturating_sub(features.len());
            features.extend(fallback.into_iter().take(remaining));
        }
    }
    features.truncate(max_features);
    features
}

pub(super) fn detect_foreground_range(alignment_image: &GrayImage) -> Option<(f64, f64)> {
    let width = alignment_image.width();
    let height = alignment_image.height();
    if width < 64 || height < 128 {
        return None;
    }

    let scan_end = ((height as f64 * FOCUS_FOREGROUND_SCAN_MAX_Y).round() as u32)
        .clamp(1, height.saturating_sub(1));
    let minimum_bright_pixels = (width as f64 * FOCUS_FOREGROUND_MIN_BRIGHT_FRACTION).ceil() as u32;
    let bright_rows = (0..scan_end)
        .map(|y| {
            let bright_pixels = alignment_image
                .rows()
                .nth(y as usize)
                .into_iter()
                .flatten()
                .filter(|pixel| pixel[0] >= FOCUS_FOREGROUND_LUMA_THRESHOLD)
                .count() as u32;
            bright_pixels >= minimum_bright_pixels
        })
        .collect::<Vec<_>>();

    // This is only a conservative seed for an optional depth mask. The main
    // focus registration does not depend on it and samples the complete frame.
    let max_gap = ((height as f64 * 0.012).round() as usize).max(2);
    let mut first = None;
    let mut last = None;
    let mut gap = 0usize;
    for (row, is_bright) in bright_rows.iter().copied().enumerate() {
        if is_bright {
            if first.is_none() {
                first = Some(row);
            }
            last = Some(row);
            gap = 0;
        } else if first.is_some() {
            gap += 1;
            if gap > max_gap {
                break;
            }
        }
    }
    let (first, last) = first.zip(last)?;
    let run_height = last.saturating_sub(first) + 1;
    if (run_height as f64) < height as f64 * FOCUS_FOREGROUND_MIN_HEIGHT_RATIO {
        return None;
    }

    let padding = (height as f64 * 0.012).max(3.0);
    let minimum = ((first as f64 - padding) / height as f64).clamp(0.0, 1.0);
    let maximum = (((last + 1) as f64 + padding) / height as f64).clamp(0.0, 1.0);
    Some((minimum, maximum))
}

pub(super) fn dilate_binary_mask(
    mask: &GrayImage,
    horizontal_radius: usize,
    vertical_radius: usize,
) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return mask.clone();
    }
    let width = width as usize;
    let height = height as usize;
    let source = mask.as_raw();
    let mut horizontal = vec![0u8; width * height];
    let mut output = vec![0u8; width * height];

    for y in 0..height {
        let row_start = y * width;
        let mut prefix = vec![0u32; width + 1];
        for x in 0..width {
            prefix[x + 1] = prefix[x] + u32::from(source[row_start + x] > 0);
        }
        for x in 0..width {
            let start = x.saturating_sub(horizontal_radius);
            let end = (x + horizontal_radius).min(width - 1);
            if prefix[end + 1] > prefix[start] {
                horizontal[row_start + x] = 255;
            }
        }
    }

    for x in 0..width {
        let mut prefix = vec![0u32; height + 1];
        for y in 0..height {
            prefix[y + 1] = prefix[y] + u32::from(horizontal[y * width + x] > 0);
        }
        for y in 0..height {
            let start = y.saturating_sub(vertical_radius);
            let end = (y + vertical_radius).min(height - 1);
            if prefix[end + 1] > prefix[start] {
                output[y * width + x] = 255;
            }
        }
    }

    GrayImage::from_raw(width as u32, height as u32, output)
        .expect("foreground mask dimensions must match")
}

pub(super) fn build_foreground_mask(
    alignment_image: &GrayImage,
    foreground_range: Option<(f64, f64)>,
) -> Option<GrayImage> {
    let (minimum_y, maximum_y) = foreground_range?;
    let (width, height) = alignment_image.dimensions();
    if width == 0 || height == 0 || minimum_y >= maximum_y {
        return None;
    }
    let base = GrayImage::from_fn(width, height, |x, y| {
        let normalized_y = y as f64 / height.max(1) as f64;
        let value = alignment_image.get_pixel(x, y)[0];
        image::Luma([
            if (minimum_y..=maximum_y).contains(&normalized_y)
                && value >= FOCUS_FOREGROUND_LUMA_THRESHOLD
            {
                255
            } else {
                0
            },
        ])
    });
    // Include the narrow shadow and metallic lip around the bright core so
    // ownership remains stable at its silhouette. This is a mask-space margin;
    // it does not alter the generic registration path.
    let horizontal_radius = ((width as f64 * 0.012).round() as usize).clamp(2, 32);
    let vertical_radius = ((height as f64 * 0.025).round() as usize).clamp(3, 48);
    Some(dilate_binary_mask(
        &base,
        horizontal_radius,
        vertical_radius,
    ))
}

pub(super) fn focus_alignment_keypoint_is_foreground(
    image: &ImageInfo,
    keypoint: KeyPoint,
) -> bool {
    let Some((minimum_y, maximum_y)) = image.foreground_range else {
        return false;
    };
    let (width, height) = image.alignment_image.dimensions();
    if keypoint.x >= width || keypoint.y >= height {
        return false;
    }
    let normalized_y = keypoint.y as f64 / height.max(1) as f64;
    if !(minimum_y..=maximum_y).contains(&normalized_y) {
        return false;
    }
    image.foreground_mask.as_ref().map_or_else(
        || {
            image.alignment_image.get_pixel(keypoint.x, keypoint.y)[0]
                >= FOCUS_FOREGROUND_LUMA_THRESHOLD
        },
        |mask| mask.get_pixel(keypoint.x, keypoint.y)[0] > 0,
    )
}

pub(super) fn find_top_alignment_features(
    alignment_image: &GrayImage,
    brief_pairs: &[(nalgebra::Point2<i32>, nalgebra::Point2<i32>)],
    foreground_range: Option<(f64, f64)>,
) -> Vec<Feature> {
    let Some((minimum_y, maximum_y)) = foreground_range else {
        return Vec::new();
    };
    if alignment_image.height() < 128 || minimum_y >= maximum_y {
        return Vec::new();
    }
    let top_start = (alignment_image.height() as f64 * minimum_y).round() as u32;
    let top_end = (alignment_image.height() as f64 * maximum_y)
        .round()
        .max((top_start + 64) as f64)
        .min(alignment_image.height() as f64) as u32;
    let top_region = image::imageops::crop_imm(
        alignment_image,
        0,
        top_start,
        alignment_image.width(),
        top_end.saturating_sub(top_start),
    )
    .to_image();
    let mut features = processing::find_features_tuned(
        &top_region,
        brief_pairs,
        SCALABLE_LOW_TEXTURE_FAST_THRESHOLD,
        (SCALABLE_LOW_TEXTURE_NMS_RADIUS * 0.75).max(6.0),
    );
    let normalized_region = processing::normalize_grayscale(&top_region);
    features.extend(processing::find_features_tuned(
        &normalized_region,
        brief_pairs,
        5,
        8.0,
    ));
    for feature in &mut features {
        feature.keypoint.y = feature.keypoint.y.saturating_add(top_start);
    }
    features.retain(|feature| {
        let y = feature.keypoint.y as f64 / alignment_image.height().max(1) as f64;
        (minimum_y..=maximum_y).contains(&y)
            && alignment_image
                .get_pixel(feature.keypoint.x, feature.keypoint.y)
                .0[0]
                >= FOCUS_FOREGROUND_LUMA_THRESHOLD
    });
    features
}

pub(super) fn match_image_pair(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    projection: Projection,
    blend_mode: BlendMode,
    alignment_mode: AlignmentMode,
    stable_four_point_solver: bool,
    log_match: bool,
    allow_low_texture_boundary: bool,
) -> Option<MatchInfo> {
    let mixed_focal_pair = image_pair_has_mixed_focal_lengths(source_image, target_image);
    let features1 = &source_image.features;
    let features2 = &target_image.features;
    let minimum_inliers = if stable_four_point_solver {
        SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION
    } else {
        processing::MIN_INLIERS_FOR_CONNECTION
    };
    let initial_matches = if stable_four_point_solver {
        let ratio_threshold = if mixed_focal_pair {
            MIXED_FOCAL_MATCH_RATIO_THRESHOLD
        } else {
            SCALABLE_MATCH_RATIO_THRESHOLD
        };
        let mut matches = if mixed_focal_pair {
            match (
                source_image.focal_length_35mm,
                target_image.focal_length_35mm,
            ) {
                (Some(source_focal), Some(target_focal)) if source_focal > 0.0 => {
                    processing::match_features_with_ratio_and_scale(
                        features1,
                        features2,
                        MIXED_FOCAL_EMERGENCY_MATCH_RATIO_THRESHOLD,
                        (source_focal / target_focal) as f32,
                        1.65,
                    )
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        if matches.len() < minimum_inliers {
            matches = processing::match_features_with_ratio(features1, features2, ratio_threshold);
        }
        if mixed_focal_pair && matches.len() < minimum_inliers {
            matches = processing::match_features_with_ratio(
                features1,
                features2,
                MIXED_FOCAL_EMERGENCY_MATCH_RATIO_THRESHOLD,
            );
            if log_match {
                println!(
                    "  - Mixed-focal descriptor fallback produced {} candidates",
                    matches.len()
                );
            }
        }
        matches
    } else {
        processing::match_features(features1, features2)
    };
    // Filename adjacency is only a candidate-order hint. It must not enable
    // a coarse transform during pair matching: with shuffled imports, an
    // arbitrary numeric suffix can otherwise turn two scan positions into a
    // false focus-layer edge before geometry has been established.
    let adjacent_same_focal_focus_pair = false;
    let severe_focus_feature_imbalance = features1.len().min(features2.len()).saturating_mul(4)
        < features1.len().max(features2.len());
    if adjacent_same_focal_focus_pair
        && severe_focus_feature_imbalance
        && let Some((homography, points, score)) = {
            let source_focal = source_image
                .focal_length_35mm
                .or(target_image.focal_length_35mm)
                .unwrap_or(50.0);
            let target_focal = target_image
                .focal_length_35mm
                .or(source_image.focal_length_35mm)
                .unwrap_or(source_focal);
            estimate_mixed_focal_coarse_registration(
                source_image,
                target_image,
                source_focal,
                target_focal,
                true,
                log_match,
            )
        }
    {
        if log_match {
            println!(
                "  - Defocused adjacent bracket registration recovered {} points (NCC {:.3})",
                points.len(),
                score
            );
        }
        return Some(MatchInfo {
            homography,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: true,
            points: points.clone(),
            candidate_points: points,
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            canonical_homography: None,
        });
    }
    if initial_matches.len() < minimum_inliers {
        // A focus bracket can contain one deliberately defocused endpoint
        // whose local descriptors collapse even though the frame has the same
        // camera pose as its immediate neighbour. Photoshop can still align
        // these pairs from their global image structure. Give only adjacent,
        // same-focal captures that bounded fallback: the coarse registrar
        // validates luminance and edges before returning a transform, while
        // the filename/focal gates prevent it from connecting unrelated scan
        // tiles merely because they contain repeated artwork texture.
        if adjacent_same_focal_focus_pair
            && let Some((homography, points, score)) = {
                let source_focal = source_image
                    .focal_length_35mm
                    .or(target_image.focal_length_35mm)
                    .unwrap_or(50.0);
                let target_focal = target_image
                    .focal_length_35mm
                    .or(source_image.focal_length_35mm)
                    .unwrap_or(source_focal);
                estimate_mixed_focal_coarse_registration(
                    source_image,
                    target_image,
                    source_focal,
                    target_focal,
                    true,
                    log_match,
                )
            }
        {
            if log_match {
                println!(
                    "  - Adjacent focus-bracket image registration recovered {} points (NCC {:.3})",
                    points.len(),
                    score
                );
            }
            return Some(MatchInfo {
                homography,
                inliers: points.len(),
                sequence_bridge: false,
                coarse_bridge: true,
                points: points.clone(),
                candidate_points: points,
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
                canonical_homography: None,
            });
        }
        if log_match && mixed_focal_pair {
            println!(
                "  - Rejecting mixed-focal pair before geometry: {} descriptor candidates (need {})",
                initial_matches.len(),
                minimum_inliers
            );
        }
        return None;
    }

    let keypoints1 = features1
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let keypoints2 = features2
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let projected_points1 = keypoints1
        .iter()
        .map(|point| {
            project_point(
                source_image,
                point.x as f64 * source_image.scale_factor,
                point.y as f64 * source_image.scale_factor,
                projection,
            )
            .expect("projection should produce finite feature coordinates")
        })
        .collect::<Vec<_>>();
    let projected_points2 = keypoints2
        .iter()
        .map(|point| {
            project_point(
                target_image,
                point.x as f64 * target_image.scale_factor,
                point.y as f64 * target_image.scale_factor,
                projection,
            )
            .expect("projection should produce finite feature coordinates")
        })
        .collect::<Vec<_>>();
    let projected_match_points = initial_matches
        .iter()
        .map(|matched| {
            (
                projected_points1[matched.index1],
                projected_points2[matched.index2],
            )
        })
        .collect::<Vec<_>>();
    let background_match_indices = if blend_mode == BlendMode::FocusStack
        && (source_image.foreground_range.is_some() || target_image.foreground_range.is_some())
    {
        initial_matches
            .iter()
            .enumerate()
            .filter_map(|(index, matched)| {
                (!focus_alignment_keypoint_is_foreground(source_image, keypoints1[matched.index1])
                    && !focus_alignment_keypoint_is_foreground(
                        target_image,
                        keypoints2[matched.index2],
                    ))
                .then_some(index)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    // A depth-discontinuous object can dominate a local descriptor match (the
    // bright holder in this fixture is one example). Solve the global camera
    // pose from the paper/background whenever there is enough background
    // support, then let the optional regional model handle the object itself.
    // If a pair contains too little background, retain the old all-point solver
    // so unusual stacks do not become disconnected merely because an object
    // detector fired.
    let solver_indices = if background_match_indices.len() >= minimum_inliers {
        background_match_indices
    } else {
        (0..projected_match_points.len()).collect::<Vec<_>>()
    };
    let solver_points = solver_indices
        .iter()
        .map(|&index| projected_match_points[index])
        .collect::<Vec<_>>();
    let top_matches =
        processing::match_features(&source_image.top_features, &target_image.top_features);
    let top_candidate_points = top_matches
        .iter()
        .filter_map(|matched| {
            let source = source_image.top_features.get(matched.index1)?.keypoint;
            let target = target_image.top_features.get(matched.index2)?.keypoint;
            Some((
                project_point(
                    source_image,
                    source.x as f64 * source_image.scale_factor,
                    source.y as f64 * source_image.scale_factor,
                    projection,
                )?,
                project_point(
                    target_image,
                    target.x as f64 * target_image.scale_factor,
                    target.y as f64 * target_image.scale_factor,
                    projection,
                )?,
            ))
        })
        .collect::<Vec<_>>();
    // The detector works in analysis pixels. The 12px native-image gate is
    // smaller than two quantized detector pixels on 200MP phone photographs.
    let seed_threshold = FULL_RES_RANSAC_INLIER_THRESHOLD
        .max(source_image.scale_factor.max(target_image.scale_factor) * 3.5);
    let mut solver_indices_used = solver_indices.clone();
    let ransac_result = if stable_four_point_solver {
        processing::find_homography_ransac_points_stable_with_min_inliers(
            &solver_points,
            seed_threshold,
            minimum_inliers,
        )
    } else {
        processing::find_homography_ransac_points(&solver_points, seed_threshold)
    };
    let mut mixed_focal_fallback_used = false;
    let ransac_result = ransac_result.or_else(|| {
        if !(mixed_focal_pair && projection == Projection::Planar) {
            return None;
        }
        let (Some(source_focal), Some(target_focal)) = (
            source_image.focal_length_35mm,
            target_image.focal_length_35mm,
        ) else {
            return None;
        };
        let fallback = find_mixed_focal_similarity_ransac(
            &solver_points,
            source_focal,
            target_focal,
            MIXED_FOCAL_SIMILARITY_RANSAC_THRESHOLD,
            minimum_inliers,
        )
        .or_else(|| {
            // Foreground masking is normally valuable for focus stacks, but a
            // lens switch can leave the paper/object boundary outside one
            // mask. Retry the complete candidate set before declaring the
            // focal-length component disconnected.
            if solver_indices.len() == projected_match_points.len() {
                return None;
            }
            solver_indices_used = (0..projected_match_points.len()).collect();
            find_mixed_focal_similarity_ransac(
                &projected_match_points,
                source_focal,
                target_focal,
                MIXED_FOCAL_SIMILARITY_RANSAC_THRESHOLD,
                minimum_inliers,
            )
        });
        if log_match && fallback.is_some() {
            mixed_focal_fallback_used = true;
            println!(
                "  - Mixed-focal similarity fallback recovered a geometric bridge ({} inliers)",
                fallback.as_ref().map_or(0, |(_, indices)| indices.len())
            );
        }
        fallback
    });
    let Some((projected_homography, projected_inlier_indices)) = ransac_result else {
        if mixed_focal_pair
            && projection == Projection::Planar
            && let (Some(source_focal), Some(target_focal)) = (
                source_image.focal_length_35mm,
                target_image.focal_length_35mm,
            )
            && let Some((coarse_homography, coarse_points, coarse_score)) =
                estimate_mixed_focal_coarse_registration(
                    source_image,
                    target_image,
                    source_focal,
                    target_focal,
                    false,
                    log_match,
                )
        {
            if log_match {
                println!(
                    "  - Mixed-focal coarse image registration recovered {} points (NCC {:.3})",
                    coarse_points.len(),
                    coarse_score
                );
            }
            return Some(MatchInfo {
                homography: coarse_homography,
                inliers: coarse_points.len(),
                sequence_bridge: false,
                coarse_bridge: true,
                points: coarse_points.clone(),
                candidate_points: coarse_points,
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
                canonical_homography: None,
            });
        }
        if log_match && mixed_focal_pair {
            println!(
                "  - Rejecting mixed-focal pair after RANSAC: {} descriptor candidates",
                initial_matches.len()
            );
        }
        return None;
    };
    if mixed_focal_pair
        && projection == Projection::Planar
        && let (Some(source_focal), Some(target_focal)) = (
            source_image.focal_length_35mm,
            target_image.focal_length_35mm,
        )
        && !mixed_focal_scale_is_plausible(&projected_homography, source_focal, target_focal)
    {
        return None;
    }
    let projected_inlier_indices = projected_inlier_indices
        .into_iter()
        .map(|index| solver_indices_used[index])
        .collect::<Vec<_>>();
    if mixed_focal_pair
        && projection == Projection::Planar
        && (source_image.foreground_range.is_some() || target_image.foreground_range.is_some())
    {
        let foreground_support = projected_inlier_indices
            .iter()
            .filter(|&&index| {
                let matched = initial_matches[index];
                focus_alignment_keypoint_is_foreground(source_image, keypoints1[matched.index1])
                    || focus_alignment_keypoint_is_foreground(
                        target_image,
                        keypoints2[matched.index2],
                    )
            })
            .count();
        if log_match {
            println!(
                "  - Mixed-focal foreground support: {foreground_support}/{} inliers",
                projected_inlier_indices.len()
            );
        }
        // A scale-correct homography supported only by a repeated wall weave
        // is still a false bridge between independent phone captures. When
        // foreground masks exist, require a few identity-bearing points too.
        if foreground_support < minimum_inliers.min(8) {
            let descriptor_pose_verified = focus_overlap_quality(
                source_image,
                target_image,
                &projected_homography,
            )
            .is_some_and(|(intensity_ncc, edge_ncc, edge_orientation, samples)| {
                if log_match {
                    println!(
                        "  - Rail-supported mixed-focal pose validation: intensity NCC {:.3}, edge NCC {:.3}, edge orientation {:.3} ({} samples)",
                        intensity_ncc, edge_ncc, edge_orientation, samples
                    );
                }
                intensity_ncc >= MIXED_FOCAL_COARSE_MIN_INTENSITY_NCC
                    && edge_ncc >= MIXED_FOCAL_COARSE_MIN_EDGE_NCC
                    && edge_orientation >= MIXED_FOCAL_COARSE_MIN_EDGE_ORIENTATION
            });
            if !descriptor_pose_verified {
                // The descriptor consensus may have landed entirely on the
                // display rail even when a genuine cross-lens artwork overlap
                // is present elsewhere. Give the denser image-level validator
                // one chance; it will reject a repeated rail-only pose by its
                // full-overlap structure score.
                if let (Some(source_focal), Some(target_focal)) = (
                    source_image.focal_length_35mm,
                    target_image.focal_length_35mm,
                ) && let Some((coarse_homography, coarse_points, coarse_score)) =
                    estimate_mixed_focal_coarse_registration(
                        source_image,
                        target_image,
                        source_focal,
                        target_focal,
                        false,
                        log_match,
                    )
                {
                    if log_match {
                        println!(
                            "  - Mixed-focal coarse image registration recovered {} points (NCC {:.3})",
                            coarse_points.len(),
                            coarse_score
                        );
                    }
                    return Some(MatchInfo {
                        homography: coarse_homography,
                        inliers: coarse_points.len(),
                        sequence_bridge: false,
                        coarse_bridge: true,
                        points: coarse_points.clone(),
                        candidate_points: coarse_points,
                        top_candidate_points: Vec::new(),
                        dense_focus_points: Vec::new(),
                        foreground_feature_points: Vec::new(),
                        canonical_homography: None,
                    });
                }
                return None;
            }
        }
    }
    let (dense_focus_points, foreground_feature_points) = if blend_mode == BlendMode::FocusStack {
        let dense_focus_points = collect_dense_focus_region_points(
            source_image,
            target_image,
            &projected_homography,
            projection,
        );
        let mut foreground_feature_points = collect_foreground_feature_points(
            source_image,
            target_image,
            &projected_homography,
            projection,
        );
        // Descriptor matches are sparse but carry a stronger identity signal
        // than a patch search on a long edge. Keep them alongside the strictly
        // corner-filtered matches; the regional residual/RANSAC checks below
        // decide whether a point is geometrically usable.
        foreground_feature_points.extend(top_candidate_points.iter().copied());
        (dense_focus_points, foreground_feature_points)
    } else {
        (Vec::new(), Vec::new())
    };
    let mut inlier_points = projected_inlier_indices
        .iter()
        .map(|&index| {
            let matched = initial_matches[index];
            if mixed_focal_pair && projection == Projection::Planar {
                // Refine the complete patch under the scale-aware model below.
                // Returning the descriptor point here is only a fallback for a
                // low-texture patch; a fixed-size native NCC patch compares the
                // wrong physical support after a 35mm/85mm module switch.
                projected_match_points[index]
            } else {
                refine_match_point_from_homography(
                    source_image,
                    target_image,
                    keypoints1[matched.index1],
                    keypoints2[matched.index2],
                    projection,
                    &projected_homography,
                    false,
                    blend_mode == BlendMode::FocusStack,
                )
                .unwrap_or(projected_match_points[index])
            }
        })
        .collect::<Vec<_>>();
    if mixed_focal_pair && projection == Projection::Planar {
        // The descriptors live in the reduced alignment images while the RANSAC
        // model above lives in source pixels. Convert the model back into the
        // descriptor coordinate system, warp every patch sample, then convert
        // the refined target observations to source pixels. This removes the
        // several-pixel quantisation error that otherwise becomes a visible
        // soft edge at 85mm/35mm overlaps.
        let source_scale = source_image.scale_factor.max(1e-9);
        let target_scale = target_image.scale_factor.max(1e-9);
        let to_target_alignment = Matrix3::new(
            target_scale.recip(),
            0.0,
            0.0,
            0.0,
            target_scale.recip(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let from_source_alignment = Matrix3::new(
            source_scale,
            0.0,
            0.0,
            0.0,
            source_scale,
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let alignment_homography =
            to_target_alignment * projected_homography * from_source_alignment;
        let mut refined_points = Vec::new();
        for &index in &projected_inlier_indices {
            let matched = initial_matches[index];
            let center = Point2::new(
                keypoints1[matched.index1].x as f64,
                keypoints1[matched.index1].y as f64,
            );
            if let Some(refined) = registration::refine_warped_patch(
                &source_image.alignment_image,
                &target_image.alignment_image,
                &alignment_homography,
                center,
                7,
                4,
            ) {
                let source = projected_match_points[index].0;
                let target = Point2::new(
                    refined.target.x * target_scale,
                    refined.target.y * target_scale,
                );
                if target.x.is_finite() && target.y.is_finite() {
                    refined_points.push((source, target));
                }
            }
        }
        if refined_points.len() >= minimum_inliers
            && match_has_spatial_support(
                &refined_points,
                source_image.dimensions(),
                target_image.dimensions(),
            )
        {
            println!(
                "  - Cross-scale warped-patch refinement retained {}/{} observations",
                refined_points.len(),
                projected_inlier_indices.len()
            );
            inlier_points = refined_points;
        }
    }
    let model_refinement_threshold = if blend_mode == BlendMode::FocusStack {
        FOCUS_MODEL_INLIER_THRESHOLD
    } else {
        FULL_RES_REFINEMENT_THRESHOLD
    };
    let alignment_quantization = source_image.scale_factor.max(target_image.scale_factor) * 1.5;
    let refinement_threshold = if source_image.full_image.is_some()
        && target_image.full_image.is_some()
        && !mixed_focal_pair
    {
        model_refinement_threshold
    } else {
        model_refinement_threshold
            .max(alignment_quantization)
            .max(if mixed_focal_pair {
                PANORAMA_MODEL_INLIER_THRESHOLD
            } else {
                0.0
            })
    };
    let refined_homography = if mixed_focal_fallback_used {
        // The two-point scale-constrained fallback has already performed its
        // own robust inlier/refit loop. Re-running the four-point homography
        // refit on only a dozen cross-scale observations can reject the valid
        // bridge because the sparse close-up has different lens distortion.
        projected_homography
    } else {
        refine_homography_inliers(&mut inlier_points, refinement_threshold, minimum_inliers)?
    };
    if stable_four_point_solver
        && !match_has_spatial_support(
            &inlier_points,
            source_image.dimensions(),
            target_image.dimensions(),
        )
    {
        // Descriptor consensus concentrated on one repeated face/stroke is not
        // enough to place a large tile. Reject it before it can become a strong
        // graph edge and pull an otherwise coherent mosaic into a false overlap.
        return None;
    }
    let mut homography = if alignment_mode == AlignmentMode::Position {
        estimate_translation(&inlier_points)
    } else if mixed_focal_fallback_used {
        // The fallback is already a scale-constrained similarity fit. Running
        // the generic focus model selector on a sparse lens-switch bridge can
        // turn its valid scale into an arbitrary affine shear, so preserve the
        // model that passed the focal-length plausibility gate.
        projected_homography
    } else if blend_mode == BlendMode::FocusStack {
        select_focus_stack_transform(
            &refined_homography,
            &inlier_points,
            source_image.dimensions(),
            alignment_mode,
        )
    } else if stable_four_point_solver && alignment_mode == AlignmentMode::Auto {
        select_large_panorama_transform(
            &refined_homography,
            &inlier_points,
            source_image.dimensions(),
            log_match,
        )
    } else {
        refined_homography
    };
    // A shifted artwork scan can require perspective even when every frame
    // uses the same lens. The low-DOF selector may favour a precise minority
    // while rejecting the correctly registered full overlap. Keep the original
    // feature/RANSAC fit as a candidate; only a distributed, independently
    // image-verified fit may supersede that minority.
    if blend_mode == BlendMode::FocusStack
        && stable_four_point_solver
        && !mixed_focal_pair
        && !mixed_focal_fallback_used
        && alignment_mode == AlignmentMode::Auto
    {
        let selected_indices =
            symmetric_inlier_indices(&homography, &inlier_points, FOCUS_MODEL_INLIER_THRESHOLD);
        let selected_points = selected_indices
            .iter()
            .map(|&index| inlier_points[index])
            .collect::<Vec<_>>();
        let selected_verified = focus_overlap_quality(source_image, target_image, &homography)
            .is_some_and(|quality| {
                focus_overlap_is_verified(
                    quality,
                    selected_points.len(),
                    median_symmetric_error(&homography, &selected_points),
                    allow_low_texture_boundary,
                )
            });
        if !selected_verified {
            if shifted_focus_structure_is_verified(
                source_image,
                target_image,
                &homography,
                &selected_points,
                log_match,
            ) {
                // Keep the simpler model when its structure is independently
                // supported; do not promote perspective solely for more points.
            } else if shifted_focus_structure_is_verified(
                source_image,
                target_image,
                &refined_homography,
                &inlier_points,
                log_match,
            ) {
                homography = refined_homography;
                if log_match {
                    println!(
                        "  - Retaining structure-verified shifted focus projective fit instead of a low-DOF minority"
                    );
                }
            }
        }
    }
    if blend_mode == BlendMode::FocusStack
        && !mixed_focal_fallback_used
        && !retain_model_inliers(
            &mut inlier_points,
            &homography,
            FOCUS_MODEL_INLIER_THRESHOLD,
            minimum_inliers,
        )
    {
        // The projective RANSAC fit is only an intermediate consensus. A
        // lower-DOF focus-stack model must get its own residual-consistent
        // observations; otherwise the global pose solve receives points that
        // support a different warp and can introduce a soft/double edge.
        if log_match {
            println!(
                "  - Rejecting focus match after selected-model validation: fewer than {} consistent observations",
                minimum_inliers
            );
        }
        return None;
    }
    let inlier_count = inlier_points.len();
    let median_error = median_symmetric_error(&homography, &inlier_points);
    let focus_overlap_quality = (blend_mode == BlendMode::FocusStack && stable_four_point_solver)
        .then(|| focus_overlap_quality(source_image, target_image, &homography))
        .flatten();
    let focus_overlap_verified = focus_overlap_quality.is_some_and(|quality| {
        focus_overlap_is_verified(
            quality,
            inlier_count,
            median_error,
            allow_low_texture_boundary,
        )
    }) || (blend_mode == BlendMode::FocusStack
        && stable_four_point_solver
        && !mixed_focal_pair
        && shifted_focus_structure_is_verified(
            source_image,
            target_image,
            &homography,
            &inlier_points,
            log_match,
        ));
    if log_match
        && let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) = focus_overlap_quality
    {
        println!(
            "  - Focus overlap validation: intensity NCC {:.3}, edge NCC {:.3}, edge orientation {:.3} ({} samples, {})",
            intensity_ncc,
            edge_ncc,
            edge_orientation,
            samples,
            if focus_overlap_verified {
                "verified"
            } else {
                "rejected"
            }
        );
    }
    if mixed_focal_pair
        && blend_mode == BlendMode::FocusStack
        && (inlier_count < MIXED_FOCAL_MIN_INLIERS
            || !median_error.is_finite()
            || median_error > MIXED_FOCAL_MAX_MEDIAN_ERROR
            || (stable_four_point_solver && !focus_overlap_verified))
    {
        if mixed_focal_fallback_used && focus_overlap_verified {
            // Detector coordinates are heavily quantized when a small
            // overview is matched to a long-lens frame. The focal-constrained
            // similarity can therefore miss the full-resolution point-error
            // cutoff while the actual warped luminance and edge structure are
            // clearly the same region. Preserve that image-verified pose as a
            // graph-only bridge; its noisy points must not enter bundle
            // adjustment.
            if log_match {
                println!(
                    "  - Accepting image-verified mixed-focal graph bridge despite {:.3}px quantized point error",
                    median_error
                );
            }
            return Some(MatchInfo {
                homography,
                inliers: inlier_count,
                sequence_bridge: false,
                coarse_bridge: true,
                points: inlier_points.clone(),
                candidate_points: projected_match_points,
                top_candidate_points,
                dense_focus_points,
                foreground_feature_points,
                canonical_homography: None,
            });
        }
        if let (Some(source_focal), Some(target_focal)) = (
            source_image.focal_length_35mm,
            target_image.focal_length_35mm,
        ) && let Some((coarse_homography, coarse_points, coarse_score)) =
            estimate_mixed_focal_coarse_registration(
                source_image,
                target_image,
                source_focal,
                target_focal,
                false,
                log_match,
            )
        {
            if log_match {
                println!(
                    "  - Mixed-focal coarse image registration recovered {} points (NCC {:.3})",
                    coarse_points.len(),
                    coarse_score
                );
            }
            return Some(MatchInfo {
                homography: coarse_homography,
                inliers: coarse_points.len(),
                sequence_bridge: false,
                coarse_bridge: true,
                points: coarse_points.clone(),
                candidate_points: coarse_points,
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
                canonical_homography: None,
            });
        }
        if log_match {
            println!(
                "  - Rejecting weak mixed-focal bridge: {} inliers, {:.3}px median symmetric error (need at least {} and <= {:.2}px)",
                inlier_count, median_error, MIXED_FOCAL_MIN_INLIERS, MIXED_FOCAL_MAX_MEDIAN_ERROR
            );
        }
        return None;
    }
    if blend_mode == BlendMode::FocusStack && stable_four_point_solver && !focus_overlap_verified {
        if focus_adjacent_same_focal_pair(source_image, target_image)
            && let Some(match_info) =
                recover_adjacent_focus_bracket_match(source_image, target_image, log_match)
        {
            if log_match {
                println!(
                    "  - Focus overlap check failed; recovered same-pose adjacent bracket instead"
                );
            }
            return Some(match_info);
        }
        if log_match {
            println!(
                "  - Rejecting focus match: the proposed overlap does not preserve the source structure"
            );
        }
        return None;
    }
    if log_match {
        println!(
            "  - Good match found: '{}' <-> '{}' ({} inliers)",
            Path::new(&source_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            inlier_count
        );
        let reprojection_error = symmetric_reprojection_rmse(&homography, &inlier_points);
        println!(
            "  - Refined match: '{}' <-> '{}' ({} inliers, {:.3}px symmetric RMS)",
            Path::new(&source_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            inlier_count,
            reprojection_error
        );
    }
    Some(MatchInfo {
        homography,
        inliers: inlier_count,
        sequence_bridge: false,
        coarse_bridge: false,
        points: inlier_points,
        candidate_points: projected_match_points,
        top_candidate_points,
        dense_focus_points,
        foreground_feature_points,
        canonical_homography: None,
    })
}

pub(super) fn estimate_translation(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> Matrix3<f64> {
    let mut dx: Vec<f64> = points.iter().map(|(a, b)| b.x - a.x).collect();
    let mut dy: Vec<f64> = points.iter().map(|(a, b)| b.y - a.y).collect();
    dx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = |values: &[f64]| values[values.len() / 2];
    Matrix3::new(1.0, 0.0, median(&dx), 0.0, 1.0, median(&dy), 0.0, 0.0, 1.0)
}

pub(super) fn estimate_similarity(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> Option<Matrix3<f64>> {
    if points.len() < 2 {
        return None;
    }
    let count = points.len() as f64;
    let source_center = points
        .iter()
        .fold(nalgebra::Vector2::zeros(), |sum, (source, _)| {
            sum + source.coords
        })
        / count;
    let target_center = points
        .iter()
        .fold(nalgebra::Vector2::zeros(), |sum, (_, target)| {
            sum + target.coords
        })
        / count;

    let mut denominator = 0.0;
    let mut dot = 0.0;
    let mut cross = 0.0;
    for (source, target) in points {
        let source_delta = source.coords - source_center;
        let target_delta = target.coords - target_center;
        denominator += source_delta.norm_squared();
        dot += source_delta.x * target_delta.x + source_delta.y * target_delta.y;
        cross += source_delta.x * target_delta.y - source_delta.y * target_delta.x;
    }
    if denominator <= f64::EPSILON {
        return None;
    }
    let a = dot / denominator;
    let b = cross / denominator;
    let translation_x = target_center.x - a * source_center.x + b * source_center.y;
    let translation_y = target_center.y - b * source_center.x - a * source_center.y;
    let transform = Matrix3::new(a, -b, translation_x, b, a, translation_y, 0.0, 0.0, 1.0);
    transform
        .iter()
        .all(|value| value.is_finite())
        .then_some(transform)
}

pub(super) fn estimate_affine(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> Option<Matrix3<f64>> {
    if points.len() < 3 {
        return None;
    }
    let mut design_data = Vec::with_capacity(points.len() * 3);
    let mut target_x = Vec::with_capacity(points.len());
    let mut target_y = Vec::with_capacity(points.len());
    for (source, target) in points {
        design_data.extend_from_slice(&[source.x, source.y, 1.0]);
        target_x.push(target.x);
        target_y.push(target.y);
    }
    let design = nalgebra::DMatrix::from_row_slice(points.len(), 3, &design_data);
    let decomposition = design.svd(true, true);
    let x_solution = decomposition
        .solve(&nalgebra::DVector::from_vec(target_x), 1e-10)
        .ok()?;
    let y_solution = decomposition
        .solve(&nalgebra::DVector::from_vec(target_y), 1e-10)
        .ok()?;
    let transform = Matrix3::new(
        x_solution[0],
        x_solution[1],
        x_solution[2],
        y_solution[0],
        y_solution[1],
        y_solution[2],
        0.0,
        0.0,
        1.0,
    );
    transform
        .iter()
        .all(|value| value.is_finite())
        .then_some(transform)
}

pub(super) fn transformed_point(
    transform: &Matrix3<f64>,
    point: nalgebra::Point2<f64>,
) -> Option<nalgebra::Point2<f64>> {
    let mapped = transform * nalgebra::Point3::new(point.x, point.y, 1.0);
    if mapped.z.abs() < 1e-8 {
        return None;
    }
    let mapped = nalgebra::Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
    (mapped.x.is_finite() && mapped.y.is_finite()).then_some(mapped)
}

pub(super) fn homography_preserves_focus_orientation(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    if width <= 1.0 || height <= 1.0 {
        return false;
    }
    let source_corners = [
        Point2::new(0.0, 0.0),
        Point2::new(width, 0.0),
        Point2::new(width, height),
        Point2::new(0.0, height),
    ];
    let mut mapped_corners = Vec::with_capacity(source_corners.len());
    let mut depth_sign = 0.0;
    for corner in source_corners {
        let mapped = transform * Point3::new(corner.x, corner.y, 1.0);
        if !mapped.iter().all(|value| value.is_finite()) || mapped.z.abs() < 1e-8 {
            return false;
        }
        let sign = mapped.z.signum();
        if depth_sign == 0.0 {
            depth_sign = sign;
        } else if sign != depth_sign {
            // A corner crossing the projective horizon can fold the image even
            // when the matrix is numerically invertible. Such a pose is never
            // a valid focus-stack registration.
            return false;
        }
        mapped_corners.push(Point2::new(mapped.x / mapped.z, mapped.y / mapped.z));
    }
    let signed_double_area = mapped_corners
        .iter()
        .zip(mapped_corners.iter().cycle().skip(1))
        .take(4)
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>();
    signed_double_area.is_finite() && signed_double_area > 0.0
}

pub(super) fn transform_is_stable_for_focus_stack(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    if width <= 1.0 || height <= 1.0 || transform.try_inverse().is_none() {
        return false;
    }
    let source_corners = [
        nalgebra::Point2::new(0.0, 0.0),
        nalgebra::Point2::new(width, 0.0),
        nalgebra::Point2::new(width, height),
        nalgebra::Point2::new(0.0, height),
    ];
    let Some(corners) = source_corners
        .into_iter()
        .map(|point| transformed_point(transform, point))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };

    let top = (corners[1] - corners[0]).norm() / width;
    let right = (corners[2] - corners[1]).norm() / height;
    let bottom = (corners[2] - corners[3]).norm() / width;
    let left = (corners[3] - corners[0]).norm() / height;
    let edge_scales = [top, right, bottom, left];
    if edge_scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale < 0.65 || *scale > 1.45)
    {
        return false;
    }
    let min_scale = edge_scales.iter().copied().fold(f64::INFINITY, f64::min);
    let max_scale = edge_scales
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_scale / min_scale > 1.18 {
        return false;
    }

    let top_vector = corners[1] - corners[0];
    let left_vector = corners[3] - corners[0];
    let orthogonality = top_vector.dot(&left_vector) / (top_vector.norm() * left_vector.norm());
    if !orthogonality.is_finite() || orthogonality.abs() > 0.22 {
        return false;
    }

    let signed_double_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .map(|(a, b)| a.x * b.y - b.x * a.y)
        .sum::<f64>();
    let area_ratio = signed_double_area / (2.0 * width * height);
    area_ratio.is_finite() && (0.5..=2.0).contains(&area_ratio)
}

pub(super) fn median_symmetric_error(
    transform: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> f64 {
    let Some(inverse) = transform.try_inverse() else {
        return f64::INFINITY;
    };
    let mut errors: Vec<f64> = points
        .iter()
        .map(|(source, target)| symmetric_point_error(transform, &inverse, *source, *target))
        .filter(|error| error.is_finite())
        .collect();
    if errors.is_empty() {
        return f64::INFINITY;
    }
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    errors[errors.len() / 2]
}

#[derive(Debug, Clone)]
pub(super) struct RobustTransformFit {
    pub(super) transform: Matrix3<f64>,
    pub(super) inlier_indices: Vec<usize>,
    pub(super) median_error: f64,
}

pub(super) fn symmetric_inlier_indices(
    transform: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    threshold: f64,
) -> Vec<usize> {
    let Some(inverse) = transform.try_inverse() else {
        return Vec::new();
    };
    points
        .iter()
        .enumerate()
        .filter_map(|(index, (source, target))| {
            (symmetric_point_error(transform, &inverse, *source, *target) <= threshold)
                .then_some(index)
        })
        .collect()
}

pub(super) fn robust_transform_fit<F>(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    sample_size: usize,
    seed: u64,
    estimator: F,
) -> Option<RobustTransformFit>
where
    F: Fn(&[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)]) -> Option<Matrix3<f64>>,
{
    robust_transform_fit_with_threshold(
        points,
        sample_size,
        seed,
        FOCUS_MODEL_INLIER_THRESHOLD,
        estimator,
    )
}

/// [`robust_transform_fit`] with a caller-chosen symmetric inlier threshold.
pub(super) fn robust_transform_fit_with_threshold<F>(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    sample_size: usize,
    seed: u64,
    inlier_threshold: f64,
    estimator: F,
) -> Option<RobustTransformFit>
where
    F: Fn(&[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)]) -> Option<Matrix3<f64>>,
{
    if points.len() < sample_size {
        return None;
    }

    // `seed` is the caller's site constant; the run seed below is derived from
    // the input content (需求 14.6).
    let mut rng = StdRng::seed_from_u64(
        seed ^ (points.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ determinism::run_random_seed_mix(),
    );
    let all_indices: Vec<usize> = (0..points.len()).collect();
    let mut best_transform = None;
    let mut best_inliers = Vec::new();
    let mut best_median = f64::INFINITY;

    let iterations = if points.len() == sample_size {
        1
    } else {
        FOCUS_MODEL_RANSAC_ITERATIONS
    };
    for _ in 0..iterations {
        let sample_indices: Vec<usize> =
            all_indices.sample(&mut rng, sample_size).copied().collect();
        if sample_indices.len() != sample_size {
            continue;
        }
        let sample: Vec<_> = sample_indices.iter().map(|&index| points[index]).collect();
        let Some(transform) = estimator(&sample) else {
            continue;
        };
        let inlier_indices = symmetric_inlier_indices(&transform, points, inlier_threshold);
        if inlier_indices.len() < sample_size {
            continue;
        }
        let inlier_points: Vec<_> = inlier_indices.iter().map(|&index| points[index]).collect();
        let median = median_symmetric_error(&transform, &inlier_points);
        if inlier_indices.len() > best_inliers.len()
            || (inlier_indices.len() == best_inliers.len() && median < best_median)
        {
            best_transform = Some(transform);
            best_inliers = inlier_indices;
            best_median = median;
        }
    }

    let mut transform = best_transform?;
    let mut inlier_indices = best_inliers;
    for _ in 0..4 {
        let inlier_points: Vec<_> = inlier_indices.iter().map(|&index| points[index]).collect();
        let Some(refitted) = estimator(&inlier_points) else {
            break;
        };
        let refitted_inliers = symmetric_inlier_indices(&refitted, points, inlier_threshold);
        if refitted_inliers.len() < sample_size {
            break;
        }
        transform = refitted;
        if refitted_inliers == inlier_indices {
            break;
        }
        inlier_indices = refitted_inliers;
    }

    let inlier_points: Vec<_> = inlier_indices.iter().map(|&index| points[index]).collect();
    Some(RobustTransformFit {
        transform,
        inlier_indices,
        median_error: median_symmetric_error(&transform, &inlier_points),
    })
}

pub(super) fn focus_fit_is_competitive(
    candidate: &RobustTransformFit,
    selected: &RobustTransformFit,
) -> bool {
    if candidate.inlier_indices.len() < FOCUS_MODEL_MIN_INLIERS {
        return false;
    }
    // A focus stack is normally captured from one stable camera pose. Prefer the
    // lower-error, lower-DOF model instead of allowing a slightly larger inlier
    // consensus to select an affine warp that bends the paper differently at each
    // edge. Extra support may win only when it does not materially worsen the fit.
    let materially_more_precise = candidate.median_error + 0.25 < selected.median_error
        && candidate.median_error <= selected.median_error * 0.90;
    let similar_precision_with_more_support = candidate.inlier_indices.len()
        >= selected.inlier_indices.len() + 4
        && candidate.median_error <= selected.median_error * 1.04;
    materially_more_precise || similar_precision_with_more_support
}

pub(super) fn select_focus_stack_transform(
    projective: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    source_dimensions: (u32, u32),
    alignment_mode: AlignmentMode,
) -> Matrix3<f64> {
    // A scale change is a shifted mosaic even when the optical centres line up
    // perfectly.  Use the broader, bounded geometry guard for that case; the
    // normal focus-stack guard intentionally rejects large scale changes so a
    // bad repeated-stroke match cannot bend a fixed-camera stack.
    let shifted_mosaic = focus_stack_motion_is_shifted_mosaic(points, source_dimensions);
    let stable_transform = |transform: &Matrix3<f64>| {
        if shifted_mosaic {
            panorama_transform_is_stable(transform, source_dimensions)
        } else {
            transform_is_stable_for_focus_stack(transform, source_dimensions)
        }
    };
    let translation = estimate_translation(points);
    let translation_inliers =
        symmetric_inlier_indices(&translation, points, FOCUS_MODEL_INLIER_THRESHOLD);
    let translation_is_valid = translation_inliers.len() >= FOCUS_MODEL_MIN_INLIERS;
    let translation_points: Vec<_> = translation_inliers
        .iter()
        .map(|&index| points[index])
        .collect();
    let mut selected = RobustTransformFit {
        transform: translation,
        inlier_indices: if translation_is_valid {
            translation_inliers
        } else {
            Vec::new()
        },
        median_error: if translation_is_valid {
            median_symmetric_error(&translation, &translation_points)
        } else {
            f64::INFINITY
        },
    };
    let mut selected_name = "translation";

    let similarity = robust_transform_fit(points, 2, 0xA24B_AED4_963E_E407, estimate_similarity);
    if let Some(model) = similarity.as_ref()
        && stable_transform(&model.transform)
        && focus_fit_is_competitive(model, &selected)
    {
        selected = model.clone();
        selected_name = "similarity";
    }

    let affine = robust_transform_fit(points, 3, 0x9FB2_1C65_1E98_DF25, estimate_affine);
    if let Some(model) = affine.as_ref()
        && stable_transform(&model.transform)
        && focus_fit_is_competitive(model, &selected)
    {
        selected = model.clone();
        selected_name = "affine";
    }

    let projective_error = median_symmetric_error(projective, points);
    let explicit_projective = matches!(
        alignment_mode,
        AlignmentMode::Perspective | AlignmentMode::Cylindrical | AlignmentMode::Spherical
    );
    // Auto mode may still need a homography for a handheld view of a flat
    // artwork. Require the inliers to cover a substantial part of both
    // frames before accepting it; a compact repeated brush stroke must stay
    // on the lower-DOF model even if its projective fit has a tiny residual.
    let projective_spatial_support =
        panorama_spatial_support(points, source_dimensions, source_dimensions);
    let allow_projective =
        explicit_projective || projective_spatial_support >= FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT;
    if allow_projective
        && stable_transform(projective)
        && projective_error <= FOCUS_MODEL_INLIER_THRESHOLD * 0.5
        && projective_error + 0.15 < selected.median_error
        && projective_error <= selected.median_error * 0.80
    {
        selected.transform = *projective;
        selected.inlier_indices = (0..points.len()).collect();
        selected.median_error = projective_error;
        selected_name = "projective";
    }

    let similarity_summary = similarity
        .as_ref()
        .map(|fit| format!("{:.3}px/{}", fit.median_error, fit.inlier_indices.len()))
        .unwrap_or_else(|| "n/a".to_string());
    let affine_summary = affine
        .as_ref()
        .map(|fit| format!("{:.3}px/{}", fit.median_error, fit.inlier_indices.len()))
        .unwrap_or_else(|| "n/a".to_string());
    let translation_summary = if translation_is_valid {
        format!(
            "{:.3}px/{}",
            median_symmetric_error(&translation, &translation_points),
            translation_points.len()
        )
    } else {
        format!("invalid/{}", translation_points.len())
    };
    println!(
        "  - Focus alignment selected {selected_name}: median symmetric error {:.3}px with {} inliers (translation {translation_summary}, similarity {similarity_summary}, affine {affine_summary}, projective {:.3}px/{}, shifted-mosaic {})",
        selected.median_error,
        selected.inlier_indices.len(),
        projective_error,
        points.len(),
        shifted_mosaic
    );
    selected.transform
}

pub(super) fn focus_stack_motion_is_shifted_mosaic(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    source_dimensions: (u32, u32),
) -> bool {
    if points.len() < FOCUS_MODEL_MIN_INLIERS {
        return false;
    }
    let translation = estimate_translation(points);
    let motion = nalgebra::Vector2::new(translation[(0, 2)], translation[(1, 2)]).norm();
    let image_scale = source_dimensions.0.max(source_dimensions.1) as f64;
    let translation_shift = motion.is_finite()
        && image_scale > 1.0
        && motion > image_scale * FOCUS_SHIFTED_MOSAIC_MOTION_RATIO;
    let scale_or_rotation_shift = estimate_similarity(points)
        .map(|similarity| {
            let scale_x = similarity[(0, 0)].hypot(similarity[(1, 0)]);
            let scale_y = similarity[(0, 1)].hypot(similarity[(1, 1)]);
            let scale = (scale_x.max(0.0) * scale_y.max(0.0)).sqrt();
            let rotation = 0.5
                * (similarity[(1, 0)] - similarity[(0, 1)])
                    .atan2(similarity[(0, 0)] + similarity[(1, 1)]);
            (scale.is_finite() && (scale - 1.0).abs() > 0.10)
                || (rotation.is_finite() && rotation.abs() > 0.035)
        })
        .unwrap_or(false);
    translation_shift || scale_or_rotation_shift
}

pub(super) fn refine_match_point_from_homography(
    image1: &ImageInfo,
    image2: &ImageInfo,
    keypoint1: KeyPoint,
    keypoint2: KeyPoint,
    projection: Projection,
    homography: &Matrix3<f64>,
    prefer_feature_center: bool,
    high_precision: bool,
) -> Option<(nalgebra::Point2<f64>, nalgebra::Point2<f64>)> {
    let source_full_x = (keypoint1.x as f64 * image1.scale_factor).round() as i32;
    let source_full_y = (keypoint1.y as f64 * image1.scale_factor).round() as i32;
    let fallback_target_full_x = (keypoint2.x as f64 * image2.scale_factor).round() as i32;
    let fallback_target_full_y = (keypoint2.y as f64 * image2.scale_factor).round() as i32;
    let (target_full_x, target_full_y) =
        if projection == Projection::Planar && !prefer_feature_center {
            let predicted =
                homography * nalgebra::Point3::new(source_full_x as f64, source_full_y as f64, 1.0);
            if predicted.z.abs() < 1e-8 {
                (fallback_target_full_x, fallback_target_full_y)
            } else {
                (
                    (predicted.x / predicted.z).round() as i32,
                    (predicted.y / predicted.z).round() as i32,
                )
            }
        } else {
            (fallback_target_full_x, fallback_target_full_y)
        };

    let (
        source_plane,
        target_plane,
        source_x,
        source_y,
        target_x,
        target_y,
        target_scale,
        patch_radius,
        search_radius,
    ) = if let (Some(source), Some(target)) = (&image1.full_image, &image2.full_image) {
        (
            LumaPlane::Rgb(source),
            LumaPlane::Rgb(target),
            source_full_x,
            source_full_y,
            target_full_x,
            target_full_y,
            1.0,
            if high_precision {
                FOCUS_MATCH_REFINE_PATCH_RADIUS
            } else {
                MATCH_REFINE_PATCH_RADIUS
            },
            if high_precision {
                FOCUS_MATCH_REFINE_SEARCH_RADIUS
            } else {
                MATCH_REFINE_SEARCH_RADIUS
            },
        )
    } else {
        (
            LumaPlane::Gray(&image1.alignment_image),
            LumaPlane::Gray(&image2.alignment_image),
            keypoint1.x as i32,
            keypoint1.y as i32,
            (target_full_x as f64 / image2.scale_factor).round() as i32,
            (target_full_y as f64 / image2.scale_factor).round() as i32,
            image2.scale_factor,
            if high_precision { 8 } else { 4 },
            if high_precision { 12 } else { 6 },
        )
    };

    let (best_x, best_y, subpixel_x, subpixel_y) = refine_patch_position(
        &source_plane,
        &target_plane,
        source_x,
        source_y,
        target_x,
        target_y,
        patch_radius,
        search_radius,
    )?;
    let source = project_point(
        image1,
        source_full_x as f64,
        source_full_y as f64,
        projection,
    )?;
    let target = project_point(
        image2,
        (best_x as f64 + subpixel_x) * target_scale,
        (best_y as f64 + subpixel_y) * target_scale,
        projection,
    )?;
    Some((source, target))
}

pub(super) enum LumaPlane<'a> {
    Rgb(&'a Rgb32FImage),
    Gray(&'a GrayImage),
}

impl LumaPlane<'_> {
    fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Rgb(image) => image.dimensions(),
            Self::Gray(image) => image.dimensions(),
        }
    }

    pub(super) fn luma_at(&self, x: i32, y: i32) -> Option<f64> {
        let (width, height) = self.dimensions();
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            return None;
        }
        match self {
            Self::Rgb(image) => {
                let pixel = image.get_pixel(x as u32, y as u32);
                Some(
                    (pixel[0] as f64 * 0.299)
                        + (pixel[1] as f64 * 0.587)
                        + (pixel[2] as f64 * 0.114),
                )
            }
            Self::Gray(image) => Some(image.get_pixel(x as u32, y as u32)[0] as f64),
        }
    }

    fn gradient_at(&self, x: i32, y: i32) -> Option<(f64, f64)> {
        let horizontal = self.luma_at(x + 1, y)? - self.luma_at(x - 1, y)?;
        let vertical = self.luma_at(x, y + 1)? - self.luma_at(x, y - 1)?;
        Some((horizontal, vertical))
    }
}

pub(super) fn gradient_patch_energy(
    image: &LumaPlane<'_>,
    center_x: i32,
    center_y: i32,
    radius: i32,
) -> f64 {
    let mut energy = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some((horizontal, vertical)) = image.gradient_at(center_x + dx, center_y + dy)
            else {
                return 0.0;
            };
            energy += horizontal * horizontal + vertical * vertical;
            sample_count += 1;
        }
    }
    if sample_count == 0 {
        0.0
    } else {
        (energy / sample_count as f64).sqrt()
    }
}

pub(super) fn gradient_patch_corner_energy(
    image: &LumaPlane<'_>,
    center_x: i32,
    center_y: i32,
    radius: i32,
) -> f64 {
    let mut sum_horizontal_squared = 0.0;
    let mut sum_vertical_squared = 0.0;
    let mut sum_cross = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some((horizontal, vertical)) = image.gradient_at(center_x + dx, center_y + dy)
            else {
                return 0.0;
            };
            sum_horizontal_squared += horizontal * horizontal;
            sum_vertical_squared += vertical * vertical;
            sum_cross += horizontal * vertical;
            sample_count += 1;
        }
    }
    if sample_count == 0 {
        return 0.0;
    }
    let sample_count = sample_count as f64;
    let horizontal_squared = sum_horizontal_squared / sample_count;
    let vertical_squared = sum_vertical_squared / sample_count;
    let cross = sum_cross / sample_count;
    (horizontal_squared * vertical_squared - cross * cross)
        .max(0.0)
        .sqrt()
}

pub(super) fn gradient_patch_ncc(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    center1_x: i32,
    center1_y: i32,
    center2_x: i32,
    center2_y: i32,
    radius: i32,
) -> f64 {
    let mut sum1_horizontal = 0.0;
    let mut sum1_vertical = 0.0;
    let mut sum2_horizontal = 0.0;
    let mut sum2_vertical = 0.0;
    let mut sum_squared1 = 0.0;
    let mut sum_squared2 = 0.0;
    let mut sum_product = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some((horizontal1, vertical1)) = image1.gradient_at(center1_x + dx, center1_y + dy)
            else {
                return f64::NEG_INFINITY;
            };
            let Some((horizontal2, vertical2)) = image2.gradient_at(center2_x + dx, center2_y + dy)
            else {
                return f64::NEG_INFINITY;
            };
            sum1_horizontal += horizontal1;
            sum1_vertical += vertical1;
            sum2_horizontal += horizontal2;
            sum2_vertical += vertical2;
            sum_squared1 += horizontal1 * horizontal1 + vertical1 * vertical1;
            sum_squared2 += horizontal2 * horizontal2 + vertical2 * vertical2;
            sum_product += horizontal1 * horizontal2 + vertical1 * vertical2;
            sample_count += 1;
        }
    }

    let sample_count = sample_count as f64;
    let covariance = sum_product
        - (sum1_horizontal * sum2_horizontal + sum1_vertical * sum2_vertical) / sample_count;
    let variance1 = sum_squared1
        - (sum1_horizontal * sum1_horizontal + sum1_vertical * sum1_vertical) / sample_count;
    let variance2 = sum_squared2
        - (sum2_horizontal * sum2_horizontal + sum2_vertical * sum2_vertical) / sample_count;
    if variance1 <= f64::EPSILON || variance2 <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        covariance / (variance1 * variance2).sqrt()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn refine_foreground_patch_position(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    source_x: i32,
    source_y: i32,
    target_x: i32,
    target_y: i32,
    patch_radius: i32,
    search_radius: i32,
) -> Option<(i32, i32, f64, f64)> {
    let source_energy = gradient_patch_energy(image1, source_x, source_y, patch_radius);
    if source_energy < FOCUS_FOREGROUND_MIN_GRADIENT_ENERGY {
        return None;
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best_target = None;
    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let candidate_x = target_x + dx;
            let candidate_y = target_y + dy;
            let gradient_score = gradient_patch_ncc(
                image1,
                image2,
                source_x,
                source_y,
                candidate_x,
                candidate_y,
                patch_radius,
            );
            if !gradient_score.is_finite() {
                continue;
            }
            // Gradient correlation identifies stable depth-layer edges and
            // structure; a small luminance contribution keeps the position from
            // becoming arbitrary on a long, nearly uniform surface.
            let luminance_score = patch_ncc(
                image1,
                image2,
                source_x,
                source_y,
                candidate_x,
                candidate_y,
                patch_radius,
            );
            let score = if luminance_score.is_finite() {
                gradient_score * 0.75 + luminance_score * 0.25
            } else {
                gradient_score
            };
            if score > best_score {
                best_score = score;
                best_target = Some((candidate_x, candidate_y));
            }
        }
    }

    let (best_x, best_y) = best_target?;
    if !best_score.is_finite() {
        return None;
    }
    let sample = |x, y| {
        let gradient_score =
            gradient_patch_ncc(image1, image2, source_x, source_y, x, y, patch_radius);
        let luminance_score = patch_ncc(image1, image2, source_x, source_y, x, y, patch_radius);
        if luminance_score.is_finite() {
            gradient_score * 0.75 + luminance_score * 0.25
        } else {
            gradient_score
        }
    };
    let subpixel_offset = |negative: f64, center: f64, positive: f64| {
        if !negative.is_finite() || !center.is_finite() || !positive.is_finite() {
            return 0.0;
        }
        let denominator = negative - 2.0 * center + positive;
        if denominator.abs() < 1e-8 {
            0.0
        } else {
            (0.5 * (negative - positive) / denominator).clamp(-1.0, 1.0)
        }
    };
    let subpixel_x = subpixel_offset(
        sample(best_x - 1, best_y),
        best_score,
        sample(best_x + 1, best_y),
    );
    let subpixel_y = subpixel_offset(
        sample(best_x, best_y - 1),
        best_score,
        sample(best_x, best_y + 1),
    );
    Some((best_x, best_y, subpixel_x, subpixel_y))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn refine_patch_position(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    source_x: i32,
    source_y: i32,
    target_x: i32,
    target_y: i32,
    patch_radius: i32,
    search_radius: i32,
) -> Option<(i32, i32, f64, f64)> {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_target = None;

    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let candidate_x = target_x + dx;
            let candidate_y = target_y + dy;
            let score = patch_ncc(
                image1,
                image2,
                source_x,
                source_y,
                candidate_x,
                candidate_y,
                patch_radius,
            );
            if score > best_score {
                best_score = score;
                best_target = Some((candidate_x, candidate_y));
            }
        }
    }

    let (best_x, best_y) = best_target?;
    if !best_score.is_finite() {
        return None;
    }
    let subpixel_offset = |negative: f64, center: f64, positive: f64| {
        if !negative.is_finite() || !center.is_finite() || !positive.is_finite() {
            return 0.0;
        }
        let denominator = negative - 2.0 * center + positive;
        if denominator.abs() < 1e-8 {
            0.0
        } else {
            (0.5 * (negative - positive) / denominator).clamp(-1.0, 1.0)
        }
    };
    let sample = |x, y| patch_ncc(image1, image2, source_x, source_y, x, y, patch_radius);
    let subpixel_x = subpixel_offset(
        sample(best_x - 1, best_y),
        best_score,
        sample(best_x + 1, best_y),
    );
    let subpixel_y = subpixel_offset(
        sample(best_x, best_y - 1),
        best_score,
        sample(best_x, best_y + 1),
    );
    Some((best_x, best_y, subpixel_x, subpixel_y))
}

pub(super) fn patch_ncc(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    center1_x: i32,
    center1_y: i32,
    center2_x: i32,
    center2_y: i32,
    radius: i32,
) -> f64 {
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum_squares1 = 0.0;
    let mut sum_squares2 = 0.0;
    let mut sum_products = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some(value1) = image1.luma_at(center1_x + dx, center1_y + dy) else {
                return f64::NEG_INFINITY;
            };
            let Some(value2) = image2.luma_at(center2_x + dx, center2_y + dy) else {
                return f64::NEG_INFINITY;
            };
            sum1 += value1;
            sum2 += value2;
            sum_squares1 += value1 * value1;
            sum_squares2 += value2 * value2;
            sum_products += value1 * value2;
            sample_count += 1;
        }
    }

    let sample_count = sample_count as f64;
    let covariance = sum_products - sum1 * sum2 / sample_count;
    let variance1 = sum_squares1 - sum1 * sum1 / sample_count;
    let variance2 = sum_squares2 - sum2 * sum2 / sample_count;
    if variance1 <= f64::EPSILON || variance2 <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        covariance / (variance1 * variance2).sqrt()
    }
}

pub(super) fn refine_homography_inliers(
    points: &mut Vec<(nalgebra::Point2<f64>, nalgebra::Point2<f64>)>,
    refinement_threshold: f64,
    minimum_inliers: usize,
) -> Option<Matrix3<f64>> {
    if points.len() < minimum_inliers {
        return None;
    }

    // Patch correlation can occasionally lock onto a nearby repeated stroke or
    // texture. Starting the least-squares refinement from every correlation lets
    // a coherent minority bend a projective model enough that none of those bad
    // points exceeds the later residual threshold. Re-run a deterministic robust
    // fit at full resolution before optimizing all remaining correspondences.
    let (_, robust_inlier_indices) =
        processing::find_homography_ransac_points_stable_with_min_inliers(
            points,
            refinement_threshold,
            minimum_inliers,
        )?;
    if robust_inlier_indices.len() < minimum_inliers {
        return None;
    }
    if robust_inlier_indices.len() < points.len() {
        *points = robust_inlier_indices
            .into_iter()
            .map(|index| points[index])
            .collect();
    }

    let mut homography = processing::compute_homography(points)?;
    for _ in 0..3 {
        let Some(inverse) = homography.try_inverse() else {
            break;
        };
        let refined_points: Vec<_> = points
            .iter()
            .copied()
            .filter(|(source, target)| {
                symmetric_point_error(&homography, &inverse, *source, *target)
                    <= refinement_threshold
            })
            .collect();
        if refined_points.len() < minimum_inliers || refined_points.len() == points.len() {
            break;
        }
        *points = refined_points;
        homography = processing::compute_homography(points)?;
    }
    Some(homography)
}

pub(super) fn retain_model_inliers(
    points: &mut Vec<(nalgebra::Point2<f64>, nalgebra::Point2<f64>)>,
    transform: &Matrix3<f64>,
    threshold: f64,
    minimum_inliers: usize,
) -> bool {
    let Some(inverse) = transform.try_inverse() else {
        return false;
    };
    let filtered = points
        .iter()
        .copied()
        .filter(|(source, target)| {
            symmetric_point_error(transform, &inverse, *source, *target) <= threshold
        })
        .collect::<Vec<_>>();
    if filtered.len() < minimum_inliers {
        return false;
    }
    *points = filtered;
    true
}

pub(super) fn symmetric_point_error(
    homography: &Matrix3<f64>,
    inverse: &Matrix3<f64>,
    source: nalgebra::Point2<f64>,
    target: nalgebra::Point2<f64>,
) -> f64 {
    let forward = homography * nalgebra::Point3::new(source.x, source.y, 1.0);
    let reverse = inverse * nalgebra::Point3::new(target.x, target.y, 1.0);
    if forward.z.abs() < 1e-8 || reverse.z.abs() < 1e-8 {
        return f64::INFINITY;
    }
    let forward_point = nalgebra::Point2::new(forward.x / forward.z, forward.y / forward.z);
    let reverse_point = nalgebra::Point2::new(reverse.x / reverse.z, reverse.y / reverse.z);
    ((forward_point - target).norm_squared() + (reverse_point - source).norm_squared()).sqrt()
}

pub(super) fn symmetric_reprojection_rmse(
    homography: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> f64 {
    let Some(inverse) = homography.try_inverse() else {
        return f64::INFINITY;
    };
    let sum_squared_error = points
        .iter()
        .filter_map(|(source, target)| {
            let forward = homography * nalgebra::Point3::new(source.x, source.y, 1.0);
            let reverse = inverse * nalgebra::Point3::new(target.x, target.y, 1.0);
            if forward.z.abs() < 1e-8 || reverse.z.abs() < 1e-8 {
                return None;
            }
            let forward_point = nalgebra::Point2::new(forward.x / forward.z, forward.y / forward.z);
            let reverse_point = nalgebra::Point2::new(reverse.x / reverse.z, reverse.y / reverse.z);
            Some((forward_point - target).norm_squared() + (reverse_point - source).norm_squared())
        })
        .sum::<f64>();
    if points.is_empty() {
        f64::INFINITY
    } else {
        (sum_squared_error / (points.len() as f64 * 2.0)).sqrt()
    }
}
