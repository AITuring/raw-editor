//! Focus-stack local warp geometry and physical-edge consensus helpers.
//!
//! The parent module keeps the orchestration and optimization flow; this module
//! owns the source-image edge detectors, robust line fits, and the narrow warp
//! bands derived from repeated physical edges.

use super::*;

#[derive(Clone)]
pub(crate) struct FocusWarpBand {
    pub(crate) homographies: HashMap<usize, Matrix3<f64>>,
    pub(crate) source_ranges: HashMap<usize, (f64, f64)>,
    pub(crate) source_x_ranges: HashMap<usize, (f64, f64)>,
    // A repeated, content-independent long edge may safely change ownership
    // at its narrow silhouette. Generic/depth bands keep the stricter
    // foreground ownership rule so a repeated stroke cannot be reintroduced.
    pub(crate) relax_foreground_seam: bool,
    // A depth-layer correction is only valid for pixels that belong to the
    // detected layer. Keeping this bit on the band prevents a near-field
    // correction from bending the continuous paper plane beside it.
    pub(crate) foreground_only: bool,
    // Long-edge consensus is a separate, content-independent geometry cue.
    // It may be retained in a shifted mosaic to keep a physical frame/rail
    // continuous, while generic regional focus fits remain disabled there.
    pub(crate) physical_edge: bool,
}

#[derive(Clone)]
pub(crate) struct FocusLayerWarp {
    pub(crate) bands: Vec<FocusWarpBand>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FocusHorizontalEdgeLine {
    pub(crate) image_id: usize,
    pub(crate) source_row: f64,
    pub(crate) world_x_center: f64,
    pub(crate) slope: f64,
    pub(crate) intercept: f64,
    pub(crate) median_error: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FocusVerticalEdgeLine {
    pub(crate) image_id: usize,
    pub(crate) source_column: f64,
    pub(crate) world_y_center: f64,
    pub(crate) slope: f64,
    pub(crate) intercept: f64,
    pub(crate) median_error: f64,
}

pub(crate) fn detect_horizontal_edge_rows(alignment_image: &GrayImage) -> Vec<f64> {
    let (width, height) = alignment_image.dimensions();
    if width < 96 || height < 96 {
        return Vec::new();
    }

    // A long paper/holder boundary has vertical gradient support across a large
    // fraction of the frame. Measuring that support rather than looking for a
    // particular luminance keeps this cue useful for white, dark, or coloured
    // borders alike, while suppressing the short horizontal strokes of text.
    let pixels = alignment_image.as_raw();
    let stride = width as usize;
    let interior_width = width.saturating_sub(2) as f64;
    let mut profile = vec![0.0f64; height as usize];
    for y in 1..height.saturating_sub(1) {
        let mut gradient_sum = 0.0f64;
        let mut supported = 0usize;
        for x in 1..width.saturating_sub(1) {
            let above = pixels[(y as usize - 1) * stride + x as usize] as i32;
            let below = pixels[(y as usize + 1) * stride + x as usize] as i32;
            let gradient = (below - above).unsigned_abs() as f64 * 0.5;
            gradient_sum += gradient;
            if gradient >= 14.0 {
                supported += 1;
            }
        }
        let mean_gradient = gradient_sum / interior_width;
        let support_ratio = supported as f64 / interior_width;
        profile[y as usize] = mean_gradient * (0.35 + support_ratio * 0.65);
    }

    let peak = profile.iter().copied().fold(0.0f64, f64::max);
    if !peak.is_finite() || peak < 4.0 {
        return Vec::new();
    }
    // Keep the detector permissive enough for a low-contrast paper/holder
    // boundary. Short character strokes still lose at the later line-fit
    // stage because they cannot form a low-residual line across the frame.
    let threshold = (peak * 0.10).max(2.5);
    let minimum_separation =
        ((height as f64 * FOCUS_HORIZONTAL_EDGE_MIN_SEPARATION_RATIO).round() as usize).max(16);
    let mut candidates = (1..height.saturating_sub(1))
        .filter(|&y| {
            let score = profile[y as usize];
            score >= threshold
                && score >= profile[y as usize - 1]
                && score >= profile[y as usize + 1]
        })
        .map(|y| (y, profile[y as usize]))
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));

    let mut selected = Vec::with_capacity(FOCUS_HORIZONTAL_EDGE_MAX_ROWS);
    for (row, _) in candidates {
        if selected.iter().all(|selected_row: &u32| {
            (*selected_row as usize).abs_diff(row as usize) >= minimum_separation
        }) {
            selected.push(row);
            if selected.len() == FOCUS_HORIZONTAL_EDGE_MAX_ROWS {
                break;
            }
        }
    }
    selected.sort_unstable();
    selected
        .into_iter()
        .map(|row| row as f64 / height.max(1) as f64)
        .collect()
}

pub(crate) fn horizontal_edge_row_candidates(
    alignment_image: &GrayImage,
    optional_region: Option<(f64, f64)>,
) -> Vec<f64> {
    let mut rows = detect_horizontal_edge_rows(alignment_image);
    // The gradient profile is the primary, object-agnostic detector. When the
    // existing optional near-field detector has already found a bright/dark
    // region, also probe its silhouettes: a boundary can be partially cropped
    // or low-contrast enough to miss the broad profile while still being a
    // valid long edge. These are only extra candidates and still need the
    // robust line fit plus multi-image consensus below.
    if let Some((minimum, maximum)) = optional_region {
        for row in [minimum, maximum] {
            if row.is_finite()
                && (0.0..=1.0).contains(&row)
                && rows.iter().all(|existing| (*existing - row).abs() >= 0.005)
            {
                rows.push(row);
            }
        }
    }
    rows.sort_unstable_by(f64::total_cmp);
    rows
}

pub(crate) fn horizontal_edge_line_samples(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    normalized_row: f64,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let (width, height) = image.alignment_image.dimensions();
    if width < 96 || height < 96 || !normalized_row.is_finite() {
        return Vec::new();
    }

    let center_y = (normalized_row.clamp(0.02, 0.98) * height as f64).round() as i32;
    let search_radius = ((height as f64 * 0.025).round() as i32)
        .max(FOCUS_HORIZONTAL_EDGE_SEARCH_RADIUS)
        .min((height as i32 / 5).max(FOCUS_HORIZONTAL_EDGE_SEARCH_RADIUS));
    let x_start = ((width as f64 * 0.04).round() as i32).max(2);
    let x_end = ((width as f64 * 0.96).round() as i32).min(width as i32 - 3);
    if x_end <= x_start {
        return Vec::new();
    }
    let pixels = image.alignment_image.as_raw();
    let stride = width as usize;
    let y_start = (center_y - search_radius).max(2);
    let y_end = (center_y + search_radius).min(height as i32 - 3);
    let responses = (x_start..=x_end)
        .map(|x| {
            (y_start..=y_end)
                .map(|y| {
                    let above = pixels[(y as usize - 1) * stride + x as usize] as i32;
                    let below = pixels[(y as usize + 1) * stride + x as usize] as i32;
                    (y, (below - above).unsigned_abs() as f64 * 0.5)
                })
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .unwrap_or((center_y, 0.0))
        })
        .collect::<Vec<_>>();
    let maximum_gap = ((width as f64 * 0.01).round() as usize).max(2);
    let mut runs = Vec::<(usize, usize)>::new();
    let mut run_start = None;
    let mut last_supported = None;
    let mut gap = 0usize;
    for (index, (_, gradient)) in responses.iter().enumerate() {
        if *gradient >= FOCUS_HORIZONTAL_EDGE_MIN_GRADIENT {
            if run_start.is_none() {
                run_start = Some(index);
            }
            last_supported = Some(index);
            gap = 0;
        } else if let (Some(start), Some(last)) = (run_start, last_supported) {
            gap += 1;
            if gap > maximum_gap {
                runs.push((start, last));
                run_start = None;
                last_supported = None;
                gap = 0;
            }
        }
    }
    if let (Some(start), Some(last)) = (run_start, last_supported) {
        runs.push((start, last));
    }
    let Some((run_start, run_end)) = runs
        .into_iter()
        .max_by_key(|(start, end)| end.saturating_sub(*start) + 1)
    else {
        return Vec::new();
    };
    let minimum_span = if image.foreground_range.is_some() {
        FOCUS_HORIZONTAL_EDGE_FOREGROUND_MIN_SOURCE_SPAN_RATIO
    } else {
        FOCUS_HORIZONTAL_EDGE_MIN_SOURCE_SPAN_RATIO
    };
    if run_end.saturating_sub(run_start) + 1 < (width as f64 * minimum_span).round() as usize {
        return Vec::new();
    }

    let run_length = run_end.saturating_sub(run_start) + 1;
    let sample_count = FOCUS_HORIZONTAL_EDGE_SAMPLE_COUNT.min(run_length).max(1);
    let mut samples = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let fraction = if sample_count == 1 {
            0.0
        } else {
            sample_index as f64 / (sample_count - 1) as f64
        };
        let response_index =
            run_start + (run_length.saturating_sub(1) as f64 * fraction).round() as usize;
        let x = x_start + response_index as i32;
        let (best_y, _) = responses[response_index];
        let source = Point2::new(
            x as f64 * image.scale_factor,
            best_y as f64 * image.scale_factor,
        );
        let Some(world) = transformed_point(homography, source) else {
            continue;
        };
        samples.push((source, world));
    }
    samples
}

pub(crate) fn fit_focus_horizontal_edge_line(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    normalized_row: f64,
) -> Option<FocusHorizontalEdgeLine> {
    let samples = horizontal_edge_line_samples(image, homography, normalized_row);
    if samples.len() < FOCUS_HORIZONTAL_EDGE_MIN_SAMPLES {
        return None;
    }
    let source_min_x = samples
        .iter()
        .map(|(source, _)| source.x)
        .fold(f64::INFINITY, f64::min);
    let source_max_x = samples
        .iter()
        .map(|(source, _)| source.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let minimum_source_span = if image.foreground_range.is_some() {
        FOCUS_HORIZONTAL_EDGE_FOREGROUND_MIN_SOURCE_SPAN_RATIO
    } else {
        FOCUS_HORIZONTAL_EDGE_MIN_SOURCE_SPAN_RATIO
    };
    if source_max_x - source_min_x < image.width as f64 * minimum_source_span {
        return None;
    }

    let mut slopes = Vec::new();
    for (index, (_, first)) in samples.iter().enumerate() {
        for (_, second) in samples.iter().skip(index + 1) {
            let delta_x = second.x - first.x;
            if delta_x.abs() < image.width as f64 * 0.02 {
                continue;
            }
            let slope = (second.y - first.y) / delta_x;
            if slope.is_finite() && slope.abs() <= FOCUS_HORIZONTAL_EDGE_MAX_SLOPE_DELTA * 2.0 {
                slopes.push(slope);
            }
        }
    }
    let slope = median_value(&mut slopes)?;
    let mut intercepts = samples
        .iter()
        .map(|(_, world)| world.y - slope * world.x)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let intercept = median_value(&mut intercepts)?;
    let mut errors = samples
        .iter()
        .map(|(_, world)| (world.y - (slope * world.x + intercept)).abs())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let median_error = median_value(&mut errors)?;
    let maximum_error =
        image.width.max(image.height) as f64 * FOCUS_HORIZONTAL_EDGE_MAX_LINE_RESIDUAL_RATIO;
    if !median_error.is_finite() || median_error > maximum_error {
        return None;
    }
    let mut source_rows = samples
        .iter()
        .map(|(source, _)| source.y / image.height.max(1) as f64)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let source_row = median_value(&mut source_rows)?;
    let mut world_xs = samples
        .iter()
        .map(|(_, world)| world.x)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let world_x_center = median_value(&mut world_xs)?;
    Some(FocusHorizontalEdgeLine {
        image_id: image.id,
        source_row,
        world_x_center,
        slope,
        intercept,
        median_error,
    })
}

pub(crate) fn deduplicate_focus_horizontal_edge_lines(
    lines: Vec<FocusHorizontalEdgeLine>,
    coordinate_scale: f64,
) -> Vec<FocusHorizontalEdgeLine> {
    if lines.len() < 2 {
        return lines;
    }
    let tolerance = coordinate_scale.max(1.0) * FOCUS_HORIZONTAL_EDGE_DEDUP_TOLERANCE_RATIO;
    let mut deduplicated = Vec::with_capacity(lines.len());
    for line in lines {
        let duplicate_index = deduplicated
            .iter()
            .position(|existing: &FocusHorizontalEdgeLine| {
                if existing.image_id != line.image_id
                    || (existing.slope - line.slope).abs() > FOCUS_HORIZONTAL_EDGE_MAX_SLOPE_DELTA
                {
                    return false;
                }
                let reference_x = (existing.world_x_center + line.world_x_center) * 0.5;
                (focus_horizontal_edge_line_y(existing, reference_x)
                    - focus_horizontal_edge_line_y(&line, reference_x))
                .abs()
                    <= tolerance
            });
        if let Some(index) = duplicate_index {
            if line.median_error < deduplicated[index].median_error {
                deduplicated[index] = line;
            }
        } else {
            deduplicated.push(line);
        }
    }
    deduplicated
}

pub(crate) fn detect_vertical_edge_columns(
    alignment_image: &GrayImage,
    optional_region: Option<(f64, f64)>,
) -> Vec<f64> {
    let (width, height) = alignment_image.dimensions();
    if width < 96 || height < 96 {
        return Vec::new();
    }

    // This is the transposed counterpart of the horizontal long-edge detector.
    // A vertical holder end or frame edge has horizontal-gradient support over
    // a long part of the frame, whereas individual character strokes are
    // rejected later by the robust line fit and cross-image consensus.
    let pixels = alignment_image.as_raw();
    let stride = width as usize;
    let region_start = optional_region
        .map(|(minimum, _)| (minimum.clamp(0.0, 1.0) * height as f64).round() as u32)
        .unwrap_or(1)
        .max(1)
        .min(height.saturating_sub(2));
    let region_end = optional_region
        .map(|(_, maximum)| (maximum.clamp(0.0, 1.0) * height as f64).round() as u32)
        .unwrap_or(height.saturating_sub(2))
        .max(region_start + 1)
        .min(height.saturating_sub(2));
    let interior_height = region_end.saturating_sub(region_start) as f64;
    if interior_height < 8.0 {
        return Vec::new();
    }
    let mut profile = vec![0.0f64; width as usize];
    for x in 1..width.saturating_sub(1) {
        let mut gradient_sum = 0.0f64;
        let mut supported = 0usize;
        for y in region_start..region_end {
            let left = pixels[y as usize * stride + (x as usize - 1)] as i32;
            let right = pixels[y as usize * stride + (x as usize + 1)] as i32;
            let gradient = (right - left).unsigned_abs() as f64 * 0.5;
            gradient_sum += gradient;
            if gradient >= 14.0 {
                supported += 1;
            }
        }
        let mean_gradient = gradient_sum / interior_height;
        let support_ratio = supported as f64 / interior_height;
        profile[x as usize] = mean_gradient * (0.35 + support_ratio * 0.65);
    }

    let peak = profile.iter().copied().fold(0.0f64, f64::max);
    if !peak.is_finite() || peak < 4.0 {
        return Vec::new();
    }
    let threshold = (peak * 0.10).max(2.5);
    let minimum_separation =
        ((width as f64 * FOCUS_VERTICAL_EDGE_MIN_SEPARATION_RATIO).round() as usize).max(16);
    let mut candidates = (1..width.saturating_sub(1))
        .filter(|&x| {
            let score = profile[x as usize];
            score >= threshold
                && score >= profile[x as usize - 1]
                && score >= profile[x as usize + 1]
        })
        .map(|x| (x, profile[x as usize]))
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));

    let mut selected = Vec::with_capacity(FOCUS_VERTICAL_EDGE_MAX_COLUMNS);
    for (column, _) in candidates {
        if selected.iter().all(|selected_column: &u32| {
            (*selected_column as usize).abs_diff(column as usize) >= minimum_separation
        }) {
            selected.push(column);
            if selected.len() == FOCUS_VERTICAL_EDGE_MAX_COLUMNS {
                break;
            }
        }
    }
    selected.sort_unstable();
    selected
        .into_iter()
        .map(|column| column as f64 / width.max(1) as f64)
        .collect()
}

pub(crate) fn vertical_edge_line_samples(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    normalized_column: f64,
    optional_region: Option<(f64, f64)>,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let (width, height) = image.alignment_image.dimensions();
    if width < 96 || height < 96 || !normalized_column.is_finite() {
        return Vec::new();
    }

    let center_x = (normalized_column.clamp(0.02, 0.98) * width as f64).round() as i32;
    let search_radius = ((width as f64 * 0.025).round() as i32)
        .max(FOCUS_VERTICAL_EDGE_SEARCH_RADIUS)
        .min((width as i32 / 5).max(FOCUS_VERTICAL_EDGE_SEARCH_RADIUS));
    let y_start = optional_region
        .map(|(minimum, _)| (minimum.clamp(0.0, 1.0) * height as f64).round() as i32)
        .unwrap_or_else(|| (height as f64 * 0.04).round() as i32)
        .max(2);
    let y_end = optional_region
        .map(|(_, maximum)| (maximum.clamp(0.0, 1.0) * height as f64).round() as i32)
        .unwrap_or_else(|| (height as f64 * 0.96).round() as i32)
        .min(height as i32 - 3);
    if y_end <= y_start {
        return Vec::new();
    }
    let pixels = image.alignment_image.as_raw();
    let stride = width as usize;
    let x_start = (center_x - search_radius).max(2);
    let x_end = (center_x + search_radius).min(width as i32 - 3);
    let responses = (y_start..=y_end)
        .map(|y| {
            (x_start..=x_end)
                .map(|x| {
                    let left = pixels[y as usize * stride + (x as usize - 1)] as i32;
                    let right = pixels[y as usize * stride + (x as usize + 1)] as i32;
                    (x, (right - left).unsigned_abs() as f64 * 0.5)
                })
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .unwrap_or((center_x, 0.0))
        })
        .collect::<Vec<_>>();
    let maximum_gap = ((height as f64 * 0.01).round() as usize).max(2);
    let mut runs = Vec::<(usize, usize)>::new();
    let mut run_start = None;
    let mut last_supported = None;
    let mut gap = 0usize;
    for (index, (_, gradient)) in responses.iter().enumerate() {
        if *gradient >= FOCUS_VERTICAL_EDGE_MIN_GRADIENT {
            if run_start.is_none() {
                run_start = Some(index);
            }
            last_supported = Some(index);
            gap = 0;
        } else if let (Some(start), Some(last)) = (run_start, last_supported) {
            gap += 1;
            if gap > maximum_gap {
                runs.push((start, last));
                run_start = None;
                last_supported = None;
                gap = 0;
            }
        }
    }
    if let (Some(start), Some(last)) = (run_start, last_supported) {
        runs.push((start, last));
    }
    let Some((run_start, run_end)) = runs
        .into_iter()
        .max_by_key(|(start, end)| end.saturating_sub(*start) + 1)
    else {
        return Vec::new();
    };
    if run_end.saturating_sub(run_start) + 1
        < (height as f64 * FOCUS_VERTICAL_EDGE_MIN_SOURCE_SPAN_RATIO).round() as usize
    {
        return Vec::new();
    }

    let run_length = run_end.saturating_sub(run_start) + 1;
    let sample_count = FOCUS_VERTICAL_EDGE_SAMPLE_COUNT.min(run_length).max(1);
    let mut samples = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let fraction = if sample_count == 1 {
            0.0
        } else {
            sample_index as f64 / (sample_count - 1) as f64
        };
        let response_index =
            run_start + (run_length.saturating_sub(1) as f64 * fraction).round() as usize;
        let y = y_start + response_index as i32;
        let (best_x, _) = responses[response_index];
        let source = Point2::new(
            best_x as f64 * image.scale_factor,
            y as f64 * image.scale_factor,
        );
        let Some(world) = transformed_point(homography, source) else {
            continue;
        };
        samples.push((source, world));
    }
    samples
}

pub(crate) fn fit_focus_vertical_edge_line(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    normalized_column: f64,
    optional_region: Option<(f64, f64)>,
) -> Option<FocusVerticalEdgeLine> {
    let samples = vertical_edge_line_samples(image, homography, normalized_column, optional_region);
    if samples.len() < FOCUS_VERTICAL_EDGE_MIN_SAMPLES {
        return None;
    }
    let source_min_y = samples
        .iter()
        .map(|(source, _)| source.y)
        .fold(f64::INFINITY, f64::min);
    let source_max_y = samples
        .iter()
        .map(|(source, _)| source.y)
        .fold(f64::NEG_INFINITY, f64::max);
    if source_max_y - source_min_y < image.height as f64 * FOCUS_VERTICAL_EDGE_MIN_SOURCE_SPAN_RATIO
    {
        return None;
    }

    let mut slopes = Vec::new();
    for (index, (_, first)) in samples.iter().enumerate() {
        for (_, second) in samples.iter().skip(index + 1) {
            let delta_y = second.y - first.y;
            if delta_y.abs() < image.height as f64 * 0.02 {
                continue;
            }
            let slope = (second.x - first.x) / delta_y;
            if slope.is_finite() && slope.abs() <= FOCUS_VERTICAL_EDGE_MAX_SLOPE_DELTA * 2.0 {
                slopes.push(slope);
            }
        }
    }
    let slope = median_value(&mut slopes)?;
    let mut intercepts = samples
        .iter()
        .map(|(_, world)| world.x - slope * world.y)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let intercept = median_value(&mut intercepts)?;
    let mut errors = samples
        .iter()
        .map(|(_, world)| (world.x - (slope * world.y + intercept)).abs())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let median_error = median_value(&mut errors)?;
    let maximum_error =
        image.width.max(image.height) as f64 * FOCUS_VERTICAL_EDGE_MAX_LINE_RESIDUAL_RATIO;
    if !median_error.is_finite() || median_error > maximum_error {
        return None;
    }
    let mut source_columns = samples
        .iter()
        .map(|(source, _)| source.x / image.width.max(1) as f64)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let source_column = median_value(&mut source_columns)?;
    let mut world_ys = samples
        .iter()
        .map(|(_, world)| world.y)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let world_y_center = median_value(&mut world_ys)?;
    Some(FocusVerticalEdgeLine {
        image_id: image.id,
        source_column,
        world_y_center,
        slope,
        intercept,
        median_error,
    })
}

pub(crate) fn deduplicate_focus_vertical_edge_lines(
    lines: Vec<FocusVerticalEdgeLine>,
    coordinate_scale: f64,
) -> Vec<FocusVerticalEdgeLine> {
    if lines.len() < 2 {
        return lines;
    }
    let tolerance = coordinate_scale.max(1.0) * FOCUS_VERTICAL_EDGE_DEDUP_TOLERANCE_RATIO;
    let mut deduplicated = Vec::with_capacity(lines.len());
    for line in lines {
        let duplicate_index = deduplicated
            .iter()
            .position(|existing: &FocusVerticalEdgeLine| {
                if existing.image_id != line.image_id
                    || (existing.slope - line.slope).abs() > FOCUS_VERTICAL_EDGE_MAX_SLOPE_DELTA
                {
                    return false;
                }
                let reference_y = (existing.world_y_center + line.world_y_center) * 0.5;
                (focus_vertical_edge_line_x(existing, reference_y)
                    - focus_vertical_edge_line_x(&line, reference_y))
                .abs()
                    <= tolerance
            });
        if let Some(index) = duplicate_index {
            if line.median_error < deduplicated[index].median_error {
                deduplicated[index] = line;
            }
        } else {
            deduplicated.push(line);
        }
    }
    deduplicated
}

pub(crate) fn focus_vertical_edge_line_x(line: &FocusVerticalEdgeLine, reference_y: f64) -> f64 {
    line.slope * reference_y + line.intercept
}

pub(crate) fn focus_vertical_edge_line_clusters(
    lines: &[FocusVerticalEdgeLine],
    coordinate_scale: f64,
) -> Vec<Vec<usize>> {
    if lines.len() < FOCUS_VERTICAL_EDGE_MIN_CLUSTER_IMAGES {
        return Vec::new();
    }
    let mut center_ys = lines
        .iter()
        .map(|line| line.world_y_center)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let Some(reference_y) = median_value(&mut center_ys) else {
        return Vec::new();
    };
    let tolerance = coordinate_scale.max(1.0) * FOCUS_VERTICAL_EDGE_CLUSTER_TOLERANCE_RATIO;
    let mut ordered_indices = (0..lines.len()).collect::<Vec<_>>();
    ordered_indices.sort_unstable_by(|left, right| {
        focus_vertical_edge_line_x(&lines[*left], reference_y)
            .total_cmp(&focus_vertical_edge_line_x(&lines[*right], reference_y))
    });

    let mut clusters = Vec::<Vec<usize>>::new();
    for index in ordered_indices {
        let line = &lines[index];
        let line_x = focus_vertical_edge_line_x(line, reference_y);
        let mut best_cluster = None;
        let mut best_distance = f64::INFINITY;
        for (cluster_index, cluster) in clusters.iter().enumerate() {
            if cluster
                .iter()
                .any(|member| lines[*member].image_id == line.image_id)
            {
                continue;
            }
            let mut cluster_xs = cluster
                .iter()
                .map(|member| focus_vertical_edge_line_x(&lines[*member], reference_y))
                .collect::<Vec<_>>();
            let Some(cluster_x) = median_value(&mut cluster_xs) else {
                continue;
            };
            let mut cluster_slopes = cluster
                .iter()
                .map(|member| lines[*member].slope)
                .collect::<Vec<_>>();
            let Some(cluster_slope) = median_value(&mut cluster_slopes) else {
                continue;
            };
            let distance = (line_x - cluster_x).abs();
            if distance <= tolerance
                && (line.slope - cluster_slope).abs() <= FOCUS_VERTICAL_EDGE_MAX_SLOPE_DELTA
                && distance < best_distance
            {
                best_cluster = Some(cluster_index);
                best_distance = distance;
            }
        }
        if let Some(cluster_index) = best_cluster {
            clusters[cluster_index].push(index);
        } else {
            clusters.push(vec![index]);
        }
    }

    clusters
        .into_iter()
        .filter(|cluster| {
            cluster
                .iter()
                .map(|index| lines[*index].image_id)
                .collect::<HashSet<_>>()
                .len()
                >= FOCUS_VERTICAL_EDGE_MIN_CLUSTER_IMAGES
        })
        .collect()
}

pub(crate) fn focus_vertical_edge_correction(
    local_line: &FocusVerticalEdgeLine,
    target_slope: f64,
    target_intercept: f64,
) -> Option<Matrix3<f64>> {
    let shear = target_slope - local_line.slope;
    let translation = target_intercept - local_line.intercept;
    if !shear.is_finite()
        || !translation.is_finite()
        || shear.abs() > FOCUS_VERTICAL_EDGE_MAX_SLOPE_DELTA
    {
        return None;
    }
    Some(Matrix3::new(
        1.0,
        shear,
        translation,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ))
}

pub(crate) fn build_focus_vertical_edge_bands(
    images: &[ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Vec<FocusWarpBand> {
    let mut lines = Vec::new();
    for image in images {
        let Some(homography) = global_homographies.get(&image.id) else {
            continue;
        };
        for &normalized_column in &image.vertical_edge_columns {
            if let Some(line) = fit_focus_vertical_edge_line(
                image,
                homography,
                normalized_column,
                image.foreground_range,
            ) {
                lines.push(line);
            }
        }
    }
    let coordinate_scale = images
        .iter()
        .map(|image| image.width.max(image.height) as f64)
        .fold(1.0, f64::max);
    let raw_line_count = lines.len();
    let lines = deduplicate_focus_vertical_edge_lines(lines, coordinate_scale);
    println!(
        "  - Vertical-edge fitted candidates: {raw_line_count} (after dedup: {})",
        lines.len()
    );
    if raw_line_count != lines.len() {
        println!(
            "  - Collapsed {} duplicate vertical-edge candidate line(s)",
            raw_line_count - lines.len()
        );
    }
    if lines.len() < FOCUS_VERTICAL_EDGE_MIN_CLUSTER_IMAGES {
        return Vec::new();
    }
    let clusters = focus_vertical_edge_line_clusters(&lines, coordinate_scale);
    if clusters.is_empty() {
        println!(
            "  - No repeated vertical-edge consensus found from {} detected edge line(s)",
            lines.len()
        );
        return Vec::new();
    }

    let reference_y = {
        let mut center_ys = lines
            .iter()
            .map(|line| line.world_y_center)
            .collect::<Vec<_>>();
        median_value(&mut center_ys).unwrap_or(0.0)
    };
    let mut bands = Vec::new();
    for cluster in clusters {
        let mut slopes = cluster
            .iter()
            .map(|index| lines[*index].slope)
            .collect::<Vec<_>>();
        let Some(target_slope) = median_value(&mut slopes) else {
            continue;
        };
        let mut target_xs = cluster
            .iter()
            .map(|index| focus_vertical_edge_line_x(&lines[*index], reference_y))
            .collect::<Vec<_>>();
        let Some(target_x_at_reference) = median_value(&mut target_xs) else {
            continue;
        };
        let target_intercept = target_x_at_reference - target_slope * reference_y;
        let mut homographies = HashMap::new();
        let mut source_ranges = HashMap::new();
        let mut source_x_ranges = HashMap::new();
        for &index in &cluster {
            let line = lines[index];
            let Some(global) = global_homographies.get(&line.image_id).copied() else {
                continue;
            };
            let Some(image) = images.iter().find(|image| image.id == line.image_id) else {
                continue;
            };
            let Some(correction) =
                focus_vertical_edge_correction(&line, target_slope, target_intercept)
            else {
                continue;
            };
            let corrected = correction * global;
            if !transform_is_stable_for_focus_stack(&corrected, image.dimensions()) {
                continue;
            }
            let minimum_source_x =
                (line.source_column - FOCUS_VERTICAL_EDGE_BAND_HALF_WIDTH_RATIO).max(0.0);
            let maximum_source_x =
                (line.source_column + FOCUS_VERTICAL_EDGE_BAND_HALF_WIDTH_RATIO).min(1.0);
            let displacement =
                focus_band_maximum_displacement(image, &global, &corrected, (0.0, 1.0));
            let maximum_allowed = image.width.max(image.height).max(1) as f64
                * FOCUS_VERTICAL_EDGE_MAX_DISPLACEMENT_RATIO;
            if !displacement.is_finite() || displacement > maximum_allowed {
                continue;
            }
            homographies.insert(line.image_id, corrected);
            source_ranges.insert(line.image_id, (0.0, 1.0));
            source_x_ranges.insert(line.image_id, (minimum_source_x, maximum_source_x));
        }
        if homographies.len() < FOCUS_VERTICAL_EDGE_MIN_CLUSTER_IMAGES {
            continue;
        }
        let average_error = cluster
            .iter()
            .map(|index| lines[*index].median_error)
            .sum::<f64>()
            / cluster.len().max(1) as f64;
        let maximum_consensus_error =
            FOCUS_VERTICAL_EDGE_MAX_CONSENSUS_ERROR_PX.max(coordinate_scale * 0.0005);
        if !average_error.is_finite() || average_error > maximum_consensus_error {
            println!(
                "  - Rejected vertical-edge consensus band: {} image(s), fit error {:.2}px exceeds {:.2}px",
                homographies.len(),
                average_error,
                maximum_consensus_error
            );
            continue;
        }
        println!(
            "  - Vertical-edge consensus band: {} image(s), world x {:.1}, fit error {:.2}px",
            homographies.len(),
            target_x_at_reference,
            average_error
        );
        bands.push(FocusWarpBand {
            homographies,
            source_ranges,
            source_x_ranges,
            relax_foreground_seam: true,
            foreground_only: false,
            physical_edge: true,
        });
    }
    bands
}

pub(crate) fn focus_horizontal_edge_line_y(
    line: &FocusHorizontalEdgeLine,
    reference_x: f64,
) -> f64 {
    line.slope * reference_x + line.intercept
}

pub(crate) fn focus_horizontal_edge_line_clusters(
    lines: &[FocusHorizontalEdgeLine],
    coordinate_scale: f64,
) -> Vec<Vec<usize>> {
    if lines.len() < FOCUS_HORIZONTAL_EDGE_MIN_CLUSTER_IMAGES {
        return Vec::new();
    }
    let mut center_xs = lines
        .iter()
        .map(|line| line.world_x_center)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let Some(reference_x) = median_value(&mut center_xs) else {
        return Vec::new();
    };
    let tolerance = coordinate_scale.max(1.0) * FOCUS_HORIZONTAL_EDGE_CLUSTER_TOLERANCE_RATIO;
    let mut ordered_indices = (0..lines.len()).collect::<Vec<_>>();
    ordered_indices.sort_unstable_by(|left, right| {
        focus_horizontal_edge_line_y(&lines[*left], reference_x)
            .total_cmp(&focus_horizontal_edge_line_y(&lines[*right], reference_x))
    });

    let mut clusters = Vec::<Vec<usize>>::new();
    for index in ordered_indices {
        let line = &lines[index];
        let line_y = focus_horizontal_edge_line_y(line, reference_x);
        let mut best_cluster = None;
        let mut best_distance = f64::INFINITY;
        for (cluster_index, cluster) in clusters.iter().enumerate() {
            if cluster
                .iter()
                .any(|member| lines[*member].image_id == line.image_id)
            {
                continue;
            }
            let mut cluster_ys = cluster
                .iter()
                .map(|member| focus_horizontal_edge_line_y(&lines[*member], reference_x))
                .collect::<Vec<_>>();
            let Some(cluster_y) = median_value(&mut cluster_ys) else {
                continue;
            };
            let mut cluster_slopes = cluster
                .iter()
                .map(|member| lines[*member].slope)
                .collect::<Vec<_>>();
            let Some(cluster_slope) = median_value(&mut cluster_slopes) else {
                continue;
            };
            let distance = (line_y - cluster_y).abs();
            if distance <= tolerance
                && (line.slope - cluster_slope).abs() <= FOCUS_HORIZONTAL_EDGE_MAX_SLOPE_DELTA
                && distance < best_distance
            {
                best_cluster = Some(cluster_index);
                best_distance = distance;
            }
        }
        if let Some(cluster_index) = best_cluster {
            clusters[cluster_index].push(index);
        } else {
            clusters.push(vec![index]);
        }
    }

    clusters
        .into_iter()
        .filter(|cluster| {
            cluster
                .iter()
                .map(|index| lines[*index].image_id)
                .collect::<HashSet<_>>()
                .len()
                >= FOCUS_HORIZONTAL_EDGE_MIN_CLUSTER_IMAGES
        })
        .collect()
}

pub(crate) fn focus_horizontal_edge_correction(
    local_line: &FocusHorizontalEdgeLine,
    target_slope: f64,
    target_intercept: f64,
) -> Option<Matrix3<f64>> {
    let shear = target_slope - local_line.slope;
    let translation = target_intercept - local_line.intercept;
    if !shear.is_finite()
        || !translation.is_finite()
        || shear.abs() > FOCUS_HORIZONTAL_EDGE_MAX_SLOPE_DELTA
    {
        return None;
    }
    Some(Matrix3::new(
        1.0,
        0.0,
        0.0,
        shear,
        1.0,
        translation,
        0.0,
        0.0,
        1.0,
    ))
}

pub(crate) fn focus_horizontal_cluster_geometry(
    lines: &[FocusHorizontalEdgeLine],
    cluster: &[usize],
    reference_x: f64,
) -> Option<(f64, f64, f64)> {
    let mut slopes = cluster
        .iter()
        .map(|index| lines[*index].slope)
        .collect::<Vec<_>>();
    let target_slope = median_value(&mut slopes)?;
    let mut target_ys = cluster
        .iter()
        .map(|index| focus_horizontal_edge_line_y(&lines[*index], reference_x))
        .collect::<Vec<_>>();
    let target_y_at_reference = median_value(&mut target_ys)?;
    let target_intercept = target_y_at_reference - target_slope * reference_x;
    Some((target_slope, target_intercept, target_y_at_reference))
}

pub(crate) fn build_focus_horizontal_edge_bands(
    images: &[ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Vec<FocusWarpBand> {
    let mut lines = Vec::new();
    for image in images {
        let Some(homography) = global_homographies.get(&image.id) else {
            continue;
        };
        for &normalized_row in &image.horizontal_edge_rows {
            if let Some(line) = fit_focus_horizontal_edge_line(image, homography, normalized_row) {
                lines.push(line);
            }
        }
    }
    let coordinate_scale = images
        .iter()
        .map(|image| image.width.max(image.height) as f64)
        .fold(1.0, f64::max);
    let raw_line_count = lines.len();
    let lines = deduplicate_focus_horizontal_edge_lines(lines, coordinate_scale);
    if raw_line_count != lines.len() {
        println!(
            "  - Collapsed {} duplicate long-edge candidate line(s)",
            raw_line_count - lines.len()
        );
    }
    if lines.len() < FOCUS_HORIZONTAL_EDGE_MIN_CLUSTER_IMAGES {
        return Vec::new();
    }
    let clusters = focus_horizontal_edge_line_clusters(&lines, coordinate_scale);
    if clusters.is_empty() {
        println!(
            "  - No repeated long-edge consensus found from {} detected edge line(s)",
            lines.len()
        );
        return Vec::new();
    }

    let reference_x = {
        let mut center_xs = lines
            .iter()
            .map(|line| line.world_x_center)
            .collect::<Vec<_>>();
        median_value(&mut center_xs).unwrap_or(0.0)
    };
    let mut bands = Vec::new();
    for cluster in &clusters {
        let Some((target_slope, target_intercept, target_y_at_reference)) =
            focus_horizontal_cluster_geometry(&lines, cluster, reference_x)
        else {
            continue;
        };
        let mut homographies = HashMap::new();
        let mut source_ranges = HashMap::new();
        for &index in cluster {
            let line = lines[index];
            let Some(global) = global_homographies.get(&line.image_id).copied() else {
                continue;
            };
            let Some(image) = images.iter().find(|image| image.id == line.image_id) else {
                continue;
            };
            let Some(correction) =
                focus_horizontal_edge_correction(&line, target_slope, target_intercept)
            else {
                continue;
            };
            let corrected = correction * global;
            if !transform_is_stable_for_focus_stack(&corrected, image.dimensions()) {
                continue;
            }
            let minimum_source_y =
                (line.source_row - FOCUS_HORIZONTAL_EDGE_BAND_HALF_HEIGHT_RATIO).max(0.0);
            let maximum_source_y =
                (line.source_row + FOCUS_HORIZONTAL_EDGE_BAND_HALF_HEIGHT_RATIO).min(1.0);
            let displacement = focus_band_maximum_displacement(
                image,
                &global,
                &corrected,
                (minimum_source_y, maximum_source_y),
            );
            let maximum_allowed = image.width.max(image.height).max(1) as f64
                * FOCUS_HORIZONTAL_EDGE_MAX_DISPLACEMENT_RATIO;
            if !displacement.is_finite() || displacement > maximum_allowed {
                continue;
            }
            homographies.insert(line.image_id, corrected);
            source_ranges.insert(line.image_id, (minimum_source_y, maximum_source_y));
        }
        if homographies.len() < FOCUS_HORIZONTAL_EDGE_MIN_CLUSTER_IMAGES {
            continue;
        }
        let average_error = cluster
            .iter()
            .map(|index| lines[*index].median_error)
            .sum::<f64>()
            / cluster.len().max(1) as f64;
        let maximum_consensus_error =
            FOCUS_HORIZONTAL_EDGE_MAX_CONSENSUS_ERROR_PX.max(coordinate_scale * 0.0005);
        if !average_error.is_finite() || average_error > maximum_consensus_error {
            println!(
                "  - Rejected long-edge consensus band: {} image(s), fit error {:.2}px exceeds {:.2}px",
                homographies.len(),
                average_error,
                maximum_consensus_error
            );
            continue;
        }
        println!(
            "  - Long-edge consensus band: {} image(s), world y {:.1}, fit error {:.2}px",
            homographies.len(),
            target_y_at_reference,
            average_error
        );
        bands.push(FocusWarpBand {
            homographies,
            source_ranges,
            source_x_ranges: HashMap::new(),
            relax_foreground_seam: false,
            foreground_only: false,
            physical_edge: true,
        });
    }

    bands
}
