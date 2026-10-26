struct LayerSampler<'a> {
    info: &'a ImageInfo,
    source: &'a Rgb32FImage,
    source_divisor: f64,
    inverse: Matrix3<f64>,
    projection: Projection,
    offset: (f64, f64),
    left: u32,
    top: u32,
    scale: f64,
    residual: Field<2>,
}

impl LayerSampler<'_> {
    fn source_point(&self, x: f64, y: f64) -> Option<Point2<f64>> {
        let delta = self.residual.at(x / self.scale, y / self.scale);
        let target = Point3::new(
            self.left as f64 + x + delta[0] * self.scale - self.offset.0,
            self.top as f64 + y + delta[1] * self.scale - self.offset.1,
            1.0,
        );
        let source = map_target_to_source(&self.inverse, target, self.info, self.projection)?;
        if source.x < 0.0
            || source.y < 0.0
            || source.x >= self.info.width as f64
            || source.y >= self.info.height as f64
        {
            return None;
        }
        Some(source)
    }

    fn sample(&self, x: f64, y: f64) -> Option<Rgb<f32>> {
        let source = self.source_point(x, y)?;
        // Each prefilter level represents the centre of a 2x2 source block.
        // Retain the half-pixel offset to avoid a lens-dependent sampling shift.
        let sx = ((source.x + 0.5) / self.source_divisor - 0.5)
            .clamp(0.0, self.source.width() as f64 - 1.0);
        let sy = ((source.y + 0.5) / self.source_divisor - 0.5)
            .clamp(0.0, self.source.height() as f64 - 1.0);
        let pixel = rgb_at(self.source, sx, sy)?;
        corrected_lower_left_capture_sample(
            self.info,
            self.source,
            source.x,
            source.y,
            sx,
            sy,
            pixel,
        )
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn analysis_pair(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
) -> (GrayImage, GrayImage, GrayImage) {
    let mut a = GrayImage::new(width, height);
    let mut b = GrayImage::new(width, height);
    let mut valid = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let lx = x as f64 * sampler.scale;
            let ly = y as f64 * sampler.scale;
            let gx = (sampler.left as f64 + lx) as u32;
            let gy = (sampler.top as f64 + ly) as u32;
            if gx >= base.width() || gy >= base.height() {
                continue;
            }
            let Some(candidate) = sampler.sample(lx, ly) else {
                continue;
            };
            b.put_pixel(
                x,
                y,
                Luma([(luma(candidate) * 255.0).round().clamp(0.0, 255.0) as u8]),
            );
            if base_mask.get_pixel(gx, gy)[0] == 0 {
                continue;
            }
            let current = *base.get_pixel(gx, gy);
            a.put_pixel(
                x,
                y,
                Luma([(luma(current) * 255.0).round().clamp(0.0, 255.0) as u8]),
            );
            valid.put_pixel(x, y, Luma([255]));
        }
    }
    // Avoid letting resampled weave/noise dominate registration at coarse scale.
    (
        imageproc::filter::gaussian_blur_f32(&a, 0.8),
        imageproc::filter::gaussian_blur_f32(&b, 0.8),
        valid,
    )
}

fn refine_layer(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
) {
    let (a, b, valid) = analysis_pair(base, base_mask, sampler, width, height);
    refine_layer_from_analysis(&a, &b, &valid, sampler);
}

fn refine_layer_from_analysis(
    a: &GrayImage,
    b: &GrayImage,
    valid: &GrayImage,
    sampler: &mut LayerSampler<'_>,
) {
    let width = a.width().min(b.width()).min(valid.width());
    let height = a.height().min(b.height()).min(valid.height());
    let mut observations = Vec::<(Point2<f64>, [f64; 2], f64)>::new();
    for y in (16..height.saturating_sub(16)).step_by(GRID_STEP as usize) {
        for x in (16..width.saturating_sub(16)).step_by(GRID_STEP as usize) {
            if [-14i32, 0, 14].iter().any(|dy| {
                [-14i32, 0, 14].iter().any(|dx| {
                    valid.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0] == 0
                })
            }) {
                continue;
            }
            let p = Point2::new(x as f64, y as f64);
            let Some(found) = refine_warped_patch(&a, &b, &Matrix3::identity(), p, 9, 7) else {
                continue;
            };
            if found.correlation < 0.88 {
                continue;
            }
            let Some(back) = refine_warped_patch(&b, &a, &Matrix3::identity(), found.target, 9, 7)
            else {
                continue;
            };
            if (back.target - p).norm() > 0.65 {
                continue;
            }
            let d = found.target - p;
            observations.push((p, [d.x, d.y], found.correlation));
        }
    }
    // A repeated stroke can win NCC locally. Require neighbouring measurements
    // to agree before allowing it to move a whole region of the source image.
    let supported = observations
        .iter()
        .filter(|(p, d, _)| {
            let neighbours = observations
                .iter()
                .filter(|(q, _, _)| (q - p).norm() <= 150.0)
                .collect::<Vec<_>>();
            if neighbours.len() < 3 {
                return false;
            }
            let mx = median(&mut neighbours.iter().map(|(_, d, _)| d[0]).collect::<Vec<_>>());
            let my = median(&mut neighbours.iter().map(|(_, d, _)| d[1]).collect::<Vec<_>>());
            (d[0] - mx).hypot(d[1] - my) <= 2.0
        })
        .copied()
        .collect::<Vec<_>>();
    if supported.len() < 4 {
        println!(
            "    - Local registration: insufficient reliable texture; keeping verified global alignment"
        );
        // Observation only (requirement 12.7/12.9): the frame keeps the global
        // model exactly as before, the ledger just records that it did.
        degradation::record_run_degradation(
            degradation::INTRA_STATION_LOCAL_FALLBACK,
            serde_json::json!({
                "stage": "mosaic_local_registration",
                "supported_patches": supported.len(),
                "minimum_patches": 4,
            }),
        );
        return;
    }
    let mut field = sampler.residual.clone();
    for gy in 0..field.height {
        for gx in 0..field.width {
            let p = Point2::new(gx as f64 * field.step, gy as f64 * field.step);
            let mut sum = [0.0; 2];
            let mut total = 0.0f64;
            for (q, d, correlation) in &supported {
                let distance = (p - q).norm_squared();
                let weight = (-distance / (2.0 * 72.0 * 72.0)).exp() * correlation.powi(4);
                sum[0] += weight * d[0];
                sum[1] += weight * d[1];
                total += weight;
            }
            // Unobserved space returns continuously to the global model.
            field.values[gy * field.width + gx][0] += sum[0] / total.max(0.05);
            field.values[gy * field.width + gx][1] += sum[1] / total.max(0.05);
        }
    }
    sampler.residual = field;
    let med = median(
        &mut supported
            .iter()
            .map(|(_, d, _)| d[0].hypot(d[1]) * sampler.scale)
            .collect::<Vec<_>>(),
    );
    println!(
        "    - Local registration: {} consistent patches, median correction {med:.2} output pixels",
        supported.len()
    );
}

fn streaming_candidate_analysis(
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
) -> (Rgb32FImage, GrayImage) {
    let mut image = Rgb32FImage::new(width, height);
    let mut mask = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let Some(pixel) = sampler.sample(x as f64 * sampler.scale, y as f64 * sampler.scale)
            else {
                continue;
            };
            image.put_pixel(x, y, pixel);
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    (image, mask)
}

fn streaming_refinement_pair(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
) -> (GrayImage, GrayImage, GrayImage) {
    let width = base.width().min(candidate.width());
    let height = base.height().min(candidate.height());
    let mut a = GrayImage::new(width, height);
    let mut b = GrayImage::new(width, height);
    let mut valid = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            if base_mask.get_pixel(x, y)[0] == 0 || candidate_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            a.put_pixel(
                x,
                y,
                Luma([(luma(*base.get_pixel(x, y)) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8]),
            );
            b.put_pixel(
                x,
                y,
                Luma([(luma(*candidate.get_pixel(x, y)) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8]),
            );
            valid.put_pixel(x, y, Luma([255]));
        }
    }
    (
        imageproc::filter::gaussian_blur_f32(&a, 0.8),
        imageproc::filter::gaussian_blur_f32(&b, 0.8),
        valid,
    )
}

fn streaming_tone_field_from_analysis(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
) -> Field<3> {
    let width = base.width().min(candidate.width());
    let height = base.height().min(candidate.height());
    let mut field = Field::new(width, height, 112.0);
    let mut samples = vec![Vec::<[f64; 3]>::new(); field.values.len()];
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            if base_mask.get_pixel(x, y)[0] == 0 || candidate_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let current = *base.get_pixel(x, y);
            let source = *candidate.get_pixel(x, y);
            if current
                .0
                .iter()
                .chain(source.0.iter())
                .any(|value| !(0.015..0.97).contains(value))
            {
                continue;
            }
            let delta =
                std::array::from_fn(|channel| f64::from(current[channel] / source[channel]).ln());
            if delta.iter().any(|value| value.abs() > 1.2) {
                continue;
            }
            let field_x = ((x as f64 / field.step).round() as usize).min(field.width - 1);
            let field_y = ((y as f64 / field.step).round() as usize).min(field.height - 1);
            samples[field_y * field.width + field_x].push(delta);
        }
    }
    let observations = samples
        .iter()
        .enumerate()
        .filter(|(_, values)| values.len() >= 4)
        .map(|(index, values)| {
            let value: [f64; 3] = std::array::from_fn(|channel| {
                median(
                    &mut values
                        .iter()
                        .map(|delta| delta[channel])
                        .collect::<Vec<_>>(),
                )
            });
            (index, value)
        })
        .collect::<Vec<_>>();
    let global: [f64; 3] = std::array::from_fn(|channel| {
        median(
            &mut samples
                .iter()
                .flatten()
                .map(|delta| delta[channel])
                .collect::<Vec<_>>(),
        )
    });
    for field_y in 0..field.height {
        for field_x in 0..field.width {
            let mut total = 0.02f64;
            let mut sum = global.map(|value| value * total);
            for (index, value) in &observations {
                let dx = field_x as f64 - (index % field.width) as f64;
                let dy = field_y as f64 - (index / field.width) as f64;
                let weight = (-(dx * dx + dy * dy) / 2.0).exp();
                for channel in 0..3 {
                    sum[channel] += value[channel] * weight;
                }
                total += weight;
            }
            field.values[field_y * field.width + field_x] =
                std::array::from_fn(|channel| (sum[channel] / total).clamp(-0.9, 0.9));
        }
    }
    field
}

fn masked_rgb_at(image: &Rgb32FImage, mask: &GrayImage, x: f64, y: f64) -> Option<Rgb<f32>> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= image.width() as f64
        || y >= image.height() as f64
        || mask.get_pixel(x as u32, y as u32)[0] == 0
    {
        return None;
    }
    rgb_at(image, x, y)
}

fn streaming_analysis_disagreement(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    tone: &Field<3>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    let mut values = Vec::new();
    for y_offset in [0.2, 0.5, 0.8] {
        for x_offset in [0.2, 0.5, 0.8] {
            let sample_x = x + cell_size * x_offset;
            let sample_y = y + cell_size * y_offset;
            let Some(current) = masked_rgb_at(base, base_mask, sample_x, sample_y) else {
                continue;
            };
            let Some(source) = masked_rgb_at(candidate, candidate_mask, sample_x, sample_y) else {
                continue;
            };
            let source = adjusted(source, tone.at(sample_x, sample_y));
            values.push(
                (0..3)
                    .map(|channel| f64::from((current[channel] - source[channel]).abs()))
                    .sum::<f64>()
                    / 3.0,
            );
        }
    }
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let high = values[(values.len() * 3 / 4).min(values.len() - 1)];
    (mean * 0.45 + high * 0.55).clamp(0.0, 1.0)
}

fn streaming_transition_structure(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    let mut levels = Vec::with_capacity(9);
    for y_offset in [0.18, 0.5, 0.82] {
        for x_offset in [0.18, 0.5, 0.82] {
            if let Some(pixel) = sample(x + cell_size * x_offset, y + cell_size * y_offset) {
                levels.push(luma(pixel));
            }
        }
    }
    if levels.len() < 5 {
        return 0.0;
    }
    levels.sort_by(f64::total_cmp);
    let dark = levels[levels.len() / 8];
    let paper = levels[levels.len() * 3 / 4].max(0.04);
    ((paper - dark) / paper).clamp(0.0, 1.0)
}

fn streaming_distance_to_fixed_label(
    width: u32,
    height: u32,
    fixed: &[i8],
    label: i8,
) -> Option<Vec<u32>> {
    let mut distances = vec![u32::MAX; fixed.len()];
    let mut pending = VecDeque::new();
    for (index, &value) in fixed.iter().enumerate() {
        if value == label {
            distances[index] = 0;
            pending.push_back(index);
        }
    }
    if pending.is_empty() {
        return None;
    }
    while let Some(index) = pending.pop_front() {
        let x = index as u32 % width;
        let y = index as u32 / width;
        let next_distance = distances[index].saturating_add(1);
        for neighbor in [
            x.checked_sub(1).map(|nx| (y * width + nx) as usize),
            (x + 1 < width).then_some((y * width + x + 1) as usize),
            y.checked_sub(1).map(|ny| (ny * width + x) as usize),
            (y + 1 < height).then_some(((y + 1) * width + x) as usize),
        ]
        .into_iter()
        .flatten()
        {
            if next_distance < distances[neighbor] {
                distances[neighbor] = next_distance;
                pending.push_back(neighbor);
            }
        }
    }
    Some(distances)
}

fn streaming_ownership_from_analysis(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    tone: &Field<3>,
    base_owner: &GrayImage,
    capture_group_id: u8,
    panorama_transition: bool,
) -> GrayImage {
    let width = base.width().min(candidate.width());
    let height = base.height().min(candidate.height());
    let size = (width.max(height) as f64 / STREAMING_SELECTION_LONG_SIDE.max(1) as f64).max(1.0);
    let grid_width = (width as f64 / size).ceil() as u32;
    let grid_height = (height as f64 / size).ceil() as u32;
    let count = (grid_width * grid_height) as usize;
    let mut preference = vec![0.0f64; count];
    let mut disagreement = vec![0.0f64; count];
    let mut fixed = vec![0i8; count];
    let mut mismatch = vec![0.0f64; count];
    let mut energy_sum = 0.0;
    let mut energy_count = 0usize;
    for grid_y in 0..grid_height {
        for grid_x in 0..grid_width {
            let index = (grid_y * grid_width + grid_x) as usize;
            let x = grid_x as f64 * size;
            let y = grid_y as f64 * size;
            let center_x = x + size * 0.5;
            let center_y = y + size * 0.5;
            let candidate_valid = masked_rgb_at(
                candidate,
                candidate_mask,
                center_x.min(width.saturating_sub(1) as f64),
                center_y.min(height.saturating_sub(1) as f64),
            )
            .is_some();
            if !candidate_valid {
                fixed[index] = -1;
                continue;
            }
            let base_valid = masked_rgb_at(
                base,
                base_mask,
                center_x.min(width.saturating_sub(1) as f64),
                center_y.min(height.saturating_sub(1) as f64),
            )
            .is_some();
            if !base_valid {
                fixed[index] = 1;
                continue;
            }
            let owner_x = center_x
                .floor()
                .clamp(0.0, base_owner.width().saturating_sub(1) as f64)
                as u32;
            let owner_y = center_y
                .floor()
                .clamp(0.0, base_owner.height().saturating_sub(1) as f64)
                as u32;
            // Once a panorama position owns a pixel, later focal planes from
            // another camera position may not reclaim it merely because a
            // repeated brush stroke scores as sharper.  A new camera group is
            // introduced once through a connected panorama seam; subsequent
            // members only compete inside that group's owned region.
            if !panorama_transition && base_owner.get_pixel(owner_x, owner_y)[0] != capture_group_id
            {
                fixed[index] = -1;
                continue;
            }
            let source_boundary = [(-size, 0.0), (size, 0.0), (0.0, -size), (0.0, size)]
                .iter()
                .any(|(dx, dy)| {
                    masked_rgb_at(candidate, candidate_mask, center_x + dx, center_y + dy).is_none()
                });
            if source_boundary {
                fixed[index] = -1;
                continue;
            }
            disagreement[index] = streaming_analysis_disagreement(
                base,
                base_mask,
                candidate,
                candidate_mask,
                tone,
                x,
                y,
                size,
            );
            if panorama_transition {
                // Cross-position stitching is a seam-placement problem, not
                // a focus contest. Prefer the established panorama except for
                // the connected region needed to admit genuinely new source
                // coverage. Make ink/edge cells expensive even when the two
                // sources agree photometrically: a seam through an aligned
                // brush stroke can still expose a tiny residual warp as a
                // horizontally cut glyph. Quiet paper remains the cheapest
                // route between characters.
                let base_structure = streaming_transition_structure(
                    |sample_x, sample_y| masked_rgb_at(base, base_mask, sample_x, sample_y),
                    x,
                    y,
                    size,
                );
                let candidate_structure = streaming_transition_structure(
                    |sample_x, sample_y| {
                        masked_rgb_at(candidate, candidate_mask, sample_x, sample_y)
                            .map(|pixel| adjusted(pixel, tone.at(sample_x, sample_y)))
                    },
                    x,
                    y,
                    size,
                );
                disagreement[index] = disagreement[index]
                    .max(base_structure)
                    .max(candidate_structure);
                preference[index] = 0.0;
                continue;
            }
            let base_focus = streaming_cell_focus(
                |sample_x, sample_y| masked_rgb_at(base, base_mask, sample_x, sample_y),
                center_x,
                center_y,
                size,
            );
            let candidate_focus = streaming_cell_focus(
                |sample_x, sample_y| {
                    masked_rgb_at(candidate, candidate_mask, sample_x, sample_y)
                        .map(|pixel| adjusted(pixel, tone.at(sample_x, sample_y)))
                },
                center_x,
                center_y,
                size,
            );
            preference[index] = candidate_focus.powi(2) - base_focus.powi(2) * 1.06;
            energy_sum += (candidate_focus.powi(2) + base_focus.powi(2)) * 0.5;
            energy_count += 1;
            let mismatch_scale = if candidate_focus > base_focus * 1.12 {
                0.30
            } else {
                1.0
            };
            mismatch[index] =
                (disagreement[index] * OWNERSHIP_MISMATCH_PENALTY * mismatch_scale).min(1.0);
        }
    }
    if panorama_transition {
        // Put the transition near the middle of the real overlap. With a
        // uniform bias toward the old panorama the minimum cut hugs the first
        // newly uncovered cell and exposes the rectangular source footprint.
        // Distances to the mandatory old/new coverage seeds provide a spatial
        // prior without using filename order or inventing image content.
        if let (Some(base_distance), Some(candidate_distance)) = (
            streaming_distance_to_fixed_label(grid_width, grid_height, &fixed, -1),
            streaming_distance_to_fixed_label(grid_width, grid_height, &fixed, 1),
        ) {
            for index in 0..count {
                if fixed[index] != 0 {
                    continue;
                }
                let base = base_distance[index] as f64;
                let candidate = candidate_distance[index] as f64;
                // Keep most of an already verified panorama intact. The new
                // camera position should own its genuinely new coverage and a
                // narrow connected overlap for a non-rectangular seam, not the
                // middle of a very large overlap. Mid-overlap ownership let a
                // weak defocused bridge (Wen Yuan Tu 3483/3484) replace the
                // tree roots and display rail with a visibly displaced block.
                preference[index] =
                    ((base - candidate) / (base + candidate).max(1.0)) * 0.60 - 0.30;
            }
        }
    }
    if !panorama_transition {
        let energy_scale = (energy_sum / energy_count.max(1) as f64).max(1e-6);
        for (index, value) in preference.iter_mut().enumerate() {
            if fixed[index] == 0 {
                *value = (*value / energy_scale).clamp(-12.0, 12.0) - mismatch[index];
            }
        }
    }
    let labels = seam_cut::cut_grid(
        grid_width as usize,
        grid_height as usize,
        &preference,
        &disagreement,
        &fixed,
    );
    GrayImage::from_raw(grid_width, grid_height, labels).expect("ownership dimensions")
}
