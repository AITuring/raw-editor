fn streaming_seam_harmonization(
    store: &StreamingMosaicStore,
    crop: (u32, u32, u32, u32),
    measured_relations: &[GroupToneRelation],
) -> Result<Option<(Rgb32FImage, f64)>, String> {
    let (left, top, width, height) = crop;
    let scale = (width.max(height) as f64 / STREAMING_HARMONIZATION_LONG_SIDE as f64).max(1.0);
    let analysis_width = (width as f64 / scale).ceil() as u32;
    let analysis_height = (height as f64 / scale).ceil() as u32;
    let (analysis, mask, owner) =
        store.analysis_region(left, top, scale, analysis_width, analysis_height)?;
    let mut boundary_count = 0usize;
    let mut owner_pixels = [0usize; 256];
    for y in 0..analysis_height {
        for x in 0..analysis_width {
            let current = owner.get_pixel(x, y)[0];
            if current == 0 || mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            owner_pixels[current as usize] += 1;
            let boundary = [
                x.checked_sub(1).map(|nx| (nx, y)),
                (x + 1 < analysis_width).then_some((x + 1, y)),
                y.checked_sub(1).map(|ny| (x, ny)),
                (y + 1 < analysis_height).then_some((x, y + 1)),
            ]
            .into_iter()
            .flatten()
            .any(|(nx, ny)| {
                let neighbour = owner.get_pixel(nx, ny)[0];
                neighbour != 0 && neighbour != current
            });
            if boundary {
                boundary_count += 1;
            }
        }
    }
    if boundary_count == 0 {
        return Ok(None);
    }
    // Estimate one exposure/colour correction per camera position from paper
    // immediately across its ownership boundaries.  The previous per-pixel
    // ratio of two blurred images followed dark strokes and created a bright
    // unsharp-mask halo around calligraphy.  Group-constant gains cannot trace
    // glyph contours, and dark/strongly coloured foreground is excluded here.
    let paper = image::imageops::blur(&analysis, 2.0);
    let mut pair_samples: BTreeMap<(u8, u8), Vec<[f64; 3]>> = BTreeMap::new();
    let probe = 6u32;
    let mut sample_pair = |ax: u32, ay: u32, bx: u32, by: u32| {
        if mask.get_pixel(ax, ay)[0] == 0 || mask.get_pixel(bx, by)[0] == 0 {
            return;
        }
        let a_owner = owner.get_pixel(ax, ay)[0];
        let b_owner = owner.get_pixel(bx, by)[0];
        if a_owner == 0 || b_owner == 0 || a_owner == b_owner {
            return;
        }
        let a = paper.get_pixel(ax, ay);
        let b = paper.get_pixel(bx, by);
        let a_luma = luma(*a);
        let b_luma = luma(*b);
        // Museum scans are often deliberately underexposed. Requiring 0.18
        // luma discarded most of this painting and left the problematic
        // camera group disconnected from the exposure solve. Group-constant
        // gains cannot follow brush contours, so darker overlap samples are
        // safe here; retain only a strict near-black and mismatch guard.
        if a_luma < 0.06 || b_luma < 0.06 || (a_luma - b_luma).abs() > 0.20 {
            return;
        }
        let (key, sign) = if a_owner < b_owner {
            ((a_owner, b_owner), 1.0)
        } else {
            ((b_owner, a_owner), -1.0)
        };
        pair_samples
            .entry(key)
            .or_default()
            .push(std::array::from_fn(|channel| {
                ((f64::from(b[channel]).max(0.015) / f64::from(a[channel]).max(0.015)).ln() * sign)
                    .clamp(-STREAMING_GROUP_GAIN_MAX_LOG, STREAMING_GROUP_GAIN_MAX_LOG)
            }));
    };
    for y in 0..analysis_height {
        for x in 0..analysis_width.saturating_sub(1) {
            if owner.get_pixel(x, y)[0] != owner.get_pixel(x + 1, y)[0]
                && x >= probe
                && x + probe < analysis_width
            {
                sample_pair(x - probe, y, x + probe, y);
            }
        }
    }
    for y in 0..analysis_height.saturating_sub(1) {
        for x in 0..analysis_width {
            if owner.get_pixel(x, y)[0] != owner.get_pixel(x, y + 1)[0]
                && y >= probe
                && y + probe < analysis_height
            {
                sample_pair(x, y - probe, x, y + probe);
            }
        }
    }
    // The per-owner log gains below are solved by repeatedly accumulating
    // `total += target * edge_weight` over every pair relation.  That floating
    // point sum is order sensitive, so the relation list must be ordered by owner
    // pair rather than by hash seed (需求 14.6).
    let mut pair_offsets: BTreeMap<(u8, u8), ([f64; 3], f64)> = pair_samples
        .into_iter()
        .filter_map(|(pair, samples)| {
            (samples.len() >= 12).then(|| {
                let offset: [f64; 3] = std::array::from_fn(|channel| {
                    median(
                        &mut samples
                            .iter()
                            .map(|value| value[channel])
                            .collect::<Vec<_>>(),
                    )
                });
                (pair, (offset, samples.len().min(512) as f64))
            })
        })
        .collect();
    // Same-coordinate overlap measurements are more reliable than probing
    // opposite sides of a seam, where the painting content can differ.
    for &(pair, offset, weight) in measured_relations {
        pair_offsets.insert(pair, (offset, weight));
    }
    let pair_offsets = pair_offsets
        .into_iter()
        .map(|(pair, (offset, weight))| (pair, offset, weight))
        .collect::<Vec<_>>();
    let anchor = (1u8..=u8::MAX)
        .max_by_key(|&id| owner_pixels[id as usize])
        .unwrap_or(1);
    let mut log_gains = [[0.0f64; 3]; 256];
    for _ in 0..48 {
        let previous = log_gains;
        for id in 1u8..=u8::MAX {
            if id == anchor || owner_pixels[id as usize] == 0 {
                continue;
            }
            for channel in 0..3 {
                let mut total = 0.0;
                let mut weight = 0.0;
                for &((a, b), offset, edge_weight) in &pair_offsets {
                    let target = if id == a {
                        previous[b as usize][channel] + offset[channel]
                    } else if id == b {
                        previous[a as usize][channel] - offset[channel]
                    } else {
                        continue;
                    };
                    total += target * edge_weight;
                    weight += edge_weight;
                }
                if weight > 0.0 {
                    log_gains[id as usize][channel] = (total / weight)
                        .clamp(-STREAMING_GROUP_GAIN_MAX_LOG, STREAMING_GROUP_GAIN_MAX_LOG);
                }
            }
        }
    }
    let owner_gains = Rgb32FImage::from_fn(analysis_width, analysis_height, |x, y| {
        let id = owner.get_pixel(x, y)[0] as usize;
        if id == 0 {
            Rgb([1.0; 3])
        } else {
            Rgb(std::array::from_fn(|channel| {
                log_gains[id][channel].exp() as f32
            }))
        }
    });
    let gains = image::imageops::blur(&owner_gains, STREAMING_GROUP_GAIN_FEATHER);
    println!(
        "  - Group exposure harmonization: {} boundary samples, {} robust group relations at {}x{}",
        boundary_count,
        pair_offsets.len(),
        analysis_width,
        analysis_height
    );
    Ok(Some((gains, scale)))
}

fn streaming_group_tone_relations_from_analysis(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    base_owner: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    capture_group_id: u8,
) -> Vec<GroupToneRelation> {
    let width = base.width().min(candidate.width()).min(base_owner.width());
    let height = base
        .height()
        .min(candidate.height())
        .min(base_owner.height());
    // Ordered by owner pair: the returned relations are later folded into the
    // shared `pair_offsets` map and into a float accumulation (需求 14.6).
    let mut samples = BTreeMap::<(u8, u8), Vec<[f64; 3]>>::new();
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            if base_mask.get_pixel(x, y)[0] == 0 || candidate_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let base_group = base_owner.get_pixel(x, y)[0];
            if base_group == 0 || base_group == capture_group_id {
                continue;
            }
            let current = *base.get_pixel(x, y);
            let source = *candidate.get_pixel(x, y);
            let current_luma = luma(current);
            let source_luma = luma(source);
            if current_luma < 0.045
                || source_luma < 0.045
                || (current_luma - source_luma).abs() > 0.22
                || current
                    .0
                    .iter()
                    .chain(source.0.iter())
                    .any(|value| !(0.015..0.97).contains(value))
            {
                continue;
            }
            let (pair, sign) = if base_group < capture_group_id {
                ((base_group, capture_group_id), 1.0)
            } else {
                ((capture_group_id, base_group), -1.0)
            };
            samples
                .entry(pair)
                .or_default()
                .push(std::array::from_fn(|channel| {
                    (f64::from(source[channel] / current[channel]).ln() * sign)
                        .clamp(-STREAMING_GROUP_GAIN_MAX_LOG, STREAMING_GROUP_GAIN_MAX_LOG)
                }));
        }
    }
    samples
        .into_iter()
        .filter_map(|(pair, values)| {
            (values.len() >= 24).then(|| {
                let offset = std::array::from_fn(|channel| {
                    median(
                        &mut values
                            .iter()
                            .map(|value| value[channel])
                            .collect::<Vec<_>>(),
                    )
                });
                (pair, offset, values.len().min(2048) as f64)
            })
        })
        .collect()
}

fn apply_streaming_seam_harmonization(output: &mut Rgb32FImage, gains: &Rgb32FImage, scale: f64) {
    let width = output.width() as usize;
    output
        .as_mut()
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let gain_y = (y as f64 / scale).clamp(0.0, gains.height().saturating_sub(1) as f64);
            for x in 0..width {
                let gain_x = (x as f64 / scale).clamp(0.0, gains.width().saturating_sub(1) as f64);
                let gain = get_high_quality_interpolated_pixel(gains, gain_x, gain_y);
                for channel in 0..3 {
                    row[x * 3 + channel] = (row[x * 3 + channel] * gain[channel]).clamp(0.0, 1.0);
                }
            }
        });
}

pub(super) fn smooth_streaming_low_frequency_illumination(output: &mut Rgb32FImage) {
    let scale = (output.width().max(output.height()) as f64
        / STREAMING_HARMONIZATION_LONG_SIDE as f64)
        .max(1.0);
    let width = (output.width() as f64 / scale).ceil().max(1.0) as u32;
    let height = (output.height() as f64 / scale).ceil().max(1.0) as u32;
    let analysis =
        image::imageops::resize(output, width, height, image::imageops::FilterType::Triangle);
    let valid = image::ImageBuffer::from_fn(width, height, |x, y| {
        let value = (luma(*analysis.get_pixel(x, y)) > 0.025) as u8 as f32;
        Luma([value])
    });
    let premultiplied = Rgb32FImage::from_fn(width, height, |x, y| {
        let weight = valid.get_pixel(x, y)[0];
        let pixel = analysis.get_pixel(x, y);
        Rgb([pixel[0] * weight, pixel[1] * weight, pixel[2] * weight])
    });
    let normalized_blur = |sigma: f32| {
        let colour = image::imageops::blur(&premultiplied, sigma);
        let weight = image::imageops::blur(&valid, sigma);
        Rgb32FImage::from_fn(width, height, |x, y| {
            let divisor = weight.get_pixel(x, y)[0];
            if divisor <= 1e-4 {
                return Rgb([0.0; 3]);
            }
            let pixel = colour.get_pixel(x, y);
            Rgb([pixel[0] / divisor, pixel[1] / divisor, pixel[2] / divisor])
        })
    };
    let local_low = normalized_blur(STREAMING_ILLUMINATION_LOCAL_SIGMA);
    let continuous_low = normalized_blur(STREAMING_ILLUMINATION_CONTINUOUS_SIGMA);
    let gains = Rgb32FImage::from_fn(width, height, |x, y| {
        if valid.get_pixel(x, y)[0] == 0.0 {
            return Rgb([1.0; 3]);
        }
        let local_pixel = *local_low.get_pixel(x, y);
        let continuous_pixel = *continuous_low.get_pixel(x, y);
        let local = luma(local_pixel);
        let continuous = luma(continuous_pixel);
        let luminance_gain = if local > 0.025 && continuous > 0.025 {
            (continuous / local).clamp(
                STREAMING_ILLUMINATION_MIN_GAIN,
                STREAMING_ILLUMINATION_MAX_GAIN,
            ) as f32
        } else {
            1.0f32
        };
        // Exposure and white-balance changes affect channels together, but a
        // scalar luma gain leaves a coloured seam when two RAW frames have
        // slightly different white balance.  Use the broad luma correction as
        // the anchor and add a restrained per-channel low-frequency term. The
        // high-frequency detail is still taken verbatim from the selected
        // owner, so this cannot blur brush edges.
        Rgb(std::array::from_fn(|channel| {
            let channel_gain = if local_pixel[channel] > 0.025 && continuous_pixel[channel] > 0.025
            {
                (continuous_pixel[channel] / local_pixel[channel]).clamp(
                    STREAMING_ILLUMINATION_MIN_GAIN as f32,
                    STREAMING_ILLUMINATION_MAX_GAIN as f32,
                )
            } else {
                luminance_gain
            };
            (luminance_gain * 0.72 + channel_gain * 0.28).clamp(
                STREAMING_ILLUMINATION_MIN_GAIN as f32,
                STREAMING_ILLUMINATION_MAX_GAIN as f32,
            )
        }))
    });
    let output_width = output.width() as usize;
    output
        .as_mut()
        .par_chunks_mut(output_width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let gain_y = (y as f64 / scale).clamp(0.0, gains.height().saturating_sub(1) as f64);
            for x in 0..output_width {
                let pixel = &mut row[x * 3..x * 3 + 3];
                if pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722 <= 0.025 {
                    continue;
                }
                let gain_x = (x as f64 / scale).clamp(0.0, gains.width().saturating_sub(1) as f64);
                let gain = get_high_quality_interpolated_pixel(&gains, gain_x, gain_y);
                for channel in 0..3 {
                    pixel[channel] = (pixel[channel] * gain[channel]).clamp(0.0, 1.0);
                }
            }
        });
}

fn luma(p: Rgb<f32>) -> f64 {
    f64::from(p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114)
}

fn rgb_at(image: &Rgb32FImage, x: f64, y: f64) -> Option<Rgb<f32>> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= image.width() as f64
        || y >= image.height() as f64
    {
        return None;
    }
    Some(get_high_quality_interpolated_pixel(image, x, y))
}

fn is_lower_left_black_capture_border(
    image: &Rgb32FImage,
    x: f64,
    y: f64,
    pixel: Rgb<f32>,
) -> bool {
    // Some handheld scan frames contain a genuinely black wedge outside the
    // photographed artwork in their lower-left corner. It is not source
    // content and must not become an owner merely because that frame extends
    // farther than its neighbours. Keep the rule deliberately local and very
    // dark: black robes, ink and furniture inside the image remain eligible.
    x < image.width() as f64 * 0.22
        && y > image.height() as f64 * 0.68
        && pixel[0].max(pixel[1]).max(pixel[2]) < 0.025
}

pub(super) fn corrected_lower_left_capture_sample(
    _info: &ImageInfo,
    image: &Rgb32FImage,
    _source_x: f64,
    _source_y: f64,
    sample_x: f64,
    sample_y: f64,
    pixel: Rgb<f32>,
) -> Option<Rgb<f32>> {
    if is_lower_left_black_capture_border(image, sample_x, sample_y, pixel) {
        return None;
    }
    Some(pixel)
}

#[derive(Clone, Debug)]
pub(crate) struct Field<const N: usize> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) step: f64,
    pub(crate) values: Vec<[f64; N]>,
}

impl<const N: usize> Field<N> {
    pub(crate) fn new(width: u32, height: u32, step: f64) -> Self {
        let width = (width as f64 / step).ceil() as usize + 1;
        let height = (height as f64 / step).ceil() as usize + 1;
        Self {
            width,
            height,
            step,
            values: vec![[0.0; N]; width * height],
        }
    }

    pub(crate) fn at(&self, x: f64, y: f64) -> [f64; N] {
        let gx = (x / self.step).clamp(0.0, self.width as f64 - 1.0);
        let gy = (y / self.step).clamp(0.0, self.height as f64 - 1.0);
        let x0 = gx as usize;
        let y0 = gy as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = gx - x0 as f64;
        let fy = gy - y0 as f64;
        std::array::from_fn(|c| {
            let a = self.values[y0 * self.width + x0][c] * (1.0 - fx)
                + self.values[y0 * self.width + x1][c] * fx;
            let b = self.values[y1 * self.width + x0][c] * (1.0 - fx)
                + self.values[y1 * self.width + x1][c] * fx;
            a * (1.0 - fy) + b * fy
        })
    }

    /// Construct a field with an explicit lower bound on the number of
    /// control nodes in each direction.  Residual_Warp uses this to retain a
    /// four-node support even for a very small overlap; the legacy mosaic
    /// field keeps the original geometry through [`Self::new`].
    pub(crate) fn new_with_min_nodes(width: u32, height: u32, step: f64, min_nodes: usize) -> Self {
        let mut field = Self::new(width, height, step);
        let min_nodes = min_nodes.max(1);
        if field.width < min_nodes || field.height < min_nodes {
            field.width = field.width.max(min_nodes);
            field.height = field.height.max(min_nodes);
            field.values.resize(field.width * field.height, [0.0; N]);
        }
        field
    }
}

